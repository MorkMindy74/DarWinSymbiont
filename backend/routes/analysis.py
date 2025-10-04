"""
Analysis routes
"""
from fastapi import APIRouter, HTTPException, status, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional
from models.problem import ProblemAnalysis, ProblemInput
from services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


# Dependency to get database (will be injected)
async def get_db() -> Optional[AsyncIOMotorDatabase]:
    return None


@router.post("/analyze/{problem_id}", response_model=ProblemAnalysis)
async def analyze_problem(
    problem_id: str, 
    problem_input: ProblemInput,
    db: Optional[AsyncIOMotorDatabase] = Depends(get_db)
):
    """
    Analyze a problem using LLM
    
    Args:
        problem_id: Problem identifier
        problem_input: Problem details
        db: Database connection
        
    Returns:
        Structured analysis
    """
    try:
        # Initialize analysis service
        analysis_service = AnalysisService()
        
        # Perform analysis
        analysis = await analysis_service.analyze_problem(problem_id, problem_input)
        
        # Save analysis to database if db provided
        if db is not None:
            analysis_dict = analysis.model_dump()
            # Upsert (update or insert)
            await db.analyses.update_one(
                {"problem_id": problem_id},
                {"$set": analysis_dict},
                upsert=True
            )
        
        return analysis
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/{problem_id}", response_model=ProblemAnalysis)
async def get_analysis(problem_id: str, db: Optional[AsyncIOMotorDatabase] = Depends(get_db)):
    """
    Get existing analysis for a problem
    
    Args:
        problem_id: Problem identifier
        db: Database connection
        
    Returns:
        Problem analysis
    """
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not configured"
        )
    
    analysis = await db.analyses.find_one({"problem_id": problem_id})
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis for problem {problem_id} not found"
        )
    
    # Remove MongoDB _id field
    analysis.pop("_id", None)
    
    return ProblemAnalysis(**analysis)


@router.post("/generate-code/{problem_id}")
async def generate_code(
    problem_id: str,
    code_type: str = "evaluate",
    db: Optional[AsyncIOMotorDatabase] = Depends(get_db)
):
    """
    Generate evaluate.py or initial.py code
    
    Args:
        problem_id: Problem identifier
        code_type: "evaluate" or "initial"
        db: Database connection
        
    Returns:
        Generated code
    """
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not configured"
        )
    
    # Get problem
    problem = await db.problems.find_one({"problem_id": problem_id})
    if not problem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Problem {problem_id} not found"
        )
    
    # Get analysis
    analysis_dict = await db.analyses.find_one({"problem_id": problem_id})
    if not analysis_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis for problem {problem_id} not found. Run analysis first."
        )
    
    analysis_dict.pop("_id", None)
    analysis = ProblemAnalysis(**analysis_dict)
    
    # Generate code
    try:
        analysis_service = AnalysisService()
        
        if code_type == "evaluate":
            code = await analysis_service.generate_evaluation_code(
                problem_type=problem["problem_input"]["problem_type"],
                analysis=analysis
            )
        elif code_type == "initial":
            code = await analysis_service.generate_initial_code(
                problem_type=problem["problem_input"]["problem_type"],
                analysis=analysis
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid code_type: {code_type}. Must be 'evaluate' or 'initial'"
            )
        
        return {"code": code, "code_type": code_type}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code generation failed: {str(e)}"
        )
