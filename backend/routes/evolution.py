"""
Evolution routes for starting and monitoring evolution
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
import uuid

from models.problem import ProblemInput
from services.shinka_bridge import (
    ShinkaEvolutionBridge,
    generate_initial_program_code,
    generate_evaluate_program_code
)
from services.websocket_manager import connection_manager

router = APIRouter(prefix="/api/evolution", tags=["evolution"])


# Dependency to get database (will be injected)
async def get_db() -> Optional[AsyncIOMotorDatabase]:
    return None


# Active evolution sessions
active_sessions: Dict[str, ShinkaEvolutionBridge] = {}


@router.post("/configure/{problem_id}")
async def configure_evolution(
    problem_id: str,
    user_config: Dict[str, Any],
    db: Optional[AsyncIOMotorDatabase] = Depends(get_db)
):
    """
    Configure evolution for a problem and generate code
    
    Args:
        problem_id: Problem identifier
        user_config: User configuration for evolution
        db: Database connection
        
    Returns:
        Session ID and configuration
    """
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not configured"
        )
    
    # Get problem from database
    problem = await db.problems.find_one({"problem_id": problem_id})
    if not problem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Problem {problem_id} not found"
        )
    
    # Get analysis
    analysis = await db.analyses.find_one({"problem_id": problem_id})
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis for problem {problem_id} not found. Run analysis first."
        )
    
    # Create session
    session_id = str(uuid.uuid4())
    work_dir = Path(f"/tmp/emergent_evolution/{session_id}")
    
    # Initialize bridge
    bridge = ShinkaEvolutionBridge(session_id, work_dir)
    
    # Generate initial.py
    problem_type = problem["problem_input"]["problem_type"]
    params = problem["problem_input"]["constraints"]
    
    initial_code = generate_initial_program_code(problem_type, params)
    bridge.save_program("initial.py", initial_code)
    
    # Generate evaluate.py
    eval_code = generate_evaluate_program_code(problem_type, params)
    bridge.save_program("evaluate.py", eval_code)
    
    # Initialize runner with user config
    bridge.initialize_runner(user_config)
    
    # Store session
    active_sessions[session_id] = bridge
    
    # Save session to database
    await db.evolution_sessions.insert_one({
        "session_id": session_id,
        "problem_id": problem_id,
        "user_config": user_config,
        "work_dir": str(work_dir),
        "status": "configured",
        "created_at": problem["created_at"]
    })
    
    return {
        "session_id": session_id,
        "problem_id": problem_id,
        "work_dir": str(work_dir),
        "ws_url": f"/api/evolution/ws/{session_id}",
        "initial_code": initial_code,
        "evaluate_code": eval_code
    }


@router.post("/start/{session_id}")
async def start_evolution(
    session_id: str,
    db: Optional[AsyncIOMotorDatabase] = Depends(get_db)
):
    """
    Start evolution for a configured session
    
    Args:
        session_id: Evolution session ID
        db: Database connection
        
    Returns:
        Status message
    """
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found. Configure first."
        )
    
    bridge = active_sessions[session_id]
    
    if bridge.is_running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evolution already running"
        )
    
    # Update status in database
    if db is not None:
        await db.evolution_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"status": "running"}}
        )
    
    # Start evolution in background
    asyncio.create_task(run_evolution_with_monitoring(session_id, bridge, db))
    
    return {
        "session_id": session_id,
        "status": "started",
        "message": "Evolution started. Connect to WebSocket for updates."
    }


async def run_evolution_with_monitoring(
    session_id: str,
    bridge: ShinkaEvolutionBridge,
    db: Optional[AsyncIOMotorDatabase]
):
    """
    Run evolution and broadcast updates via WebSocket
    """
    try:
        # Send start message
        await connection_manager.broadcast_to_session(session_id, {
            "type": "evolution_start",
            "session_id": session_id,
            "message": "Evolution started"
        })
        
        # Start monitoring task
        monitor_task = asyncio.create_task(
            monitor_evolution_progress(session_id, bridge)
        )
        
        # Run evolution
        await bridge.start_evolution()
        
        # Cancel monitoring
        monitor_task.cancel()
        
        # Send completion message
        best_solution = bridge.get_best_solution()
        
        await connection_manager.broadcast_to_session(session_id, {
            "type": "evolution_complete",
            "session_id": session_id,
            "best_solution": best_solution,
            "message": "Evolution completed successfully"
        })
        
        # Update database
        if db is not None:
            await db.evolution_sessions.update_one(
                {"session_id": session_id},
                {"$set": {
                    "status": "completed",
                    "best_solution": best_solution
                }}
            )
        
    except Exception as e:
        print(f"❌ Evolution failed for session {session_id}: {e}")
        
        await connection_manager.broadcast_to_session(session_id, {
            "type": "evolution_error",
            "session_id": session_id,
            "error": str(e),
            "message": "Evolution failed"
        })
        
        if db is not None:
            await db.evolution_sessions.update_one(
                {"session_id": session_id},
                {"$set": {
                    "status": "failed",
                    "error": str(e)
                }}
            )


async def monitor_evolution_progress(session_id: str, bridge: ShinkaEvolutionBridge):
    """
    Monitor database for changes and broadcast updates
    """
    last_generation = 0
    
    while True:
        try:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            current_gen = bridge.get_latest_generation()
            
            if current_gen is None:
                continue
            
            if current_gen > last_generation:
                # New generation completed
                gen_data = bridge.get_generation_data(current_gen)
                
                await connection_manager.broadcast_to_session(session_id, {
                    "type": "generation_complete",
                    **gen_data
                })
                
                # Get island status
                islands = bridge.get_island_status()
                
                await connection_manager.broadcast_to_session(session_id, {
                    "type": "islands_update",
                    "islands": islands
                })
                
                last_generation = current_gen
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error monitoring evolution: {e}")
            await asyncio.sleep(5)


@router.websocket("/ws/{session_id}")
async def evolution_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time evolution updates
    """
    await connection_manager.connect(websocket, session_id)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "WebSocket connected"
        })
        
        # Keep connection alive
        while True:
            # Wait for messages from client (ping/pong)
            data = await websocket.receive_text()
            
            # Echo back
            await websocket.send_json({
                "type": "pong",
                "data": data
            })
    
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket, session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await connection_manager.disconnect(websocket, session_id)


@router.get("/status/{session_id}")
async def get_evolution_status(
    session_id: str,
    db: Optional[AsyncIOMotorDatabase] = Depends(get_db)
):
    """
    Get current evolution status
    """
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not configured"
        )
    
    session = await db.evolution_sessions.find_one({"session_id": session_id})
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    # Remove MongoDB _id
    session.pop("_id", None)
    
    # Add runtime info if session is active
    if session_id in active_sessions:
        bridge = active_sessions[session_id]
        session["is_running"] = bridge.is_running
        session["latest_generation"] = bridge.get_latest_generation()
        session["best_solution"] = bridge.get_best_solution()
        session["islands"] = bridge.get_island_status()
    
    return session
