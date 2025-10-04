"""
Problem routes
"""
from fastapi import APIRouter, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List

try:
    from ..models.problem import ProblemInput, ProblemCreate, ProblemWithAnalysis
except ImportError:
    from backend.models.problem import ProblemInput, ProblemCreate, ProblemWithAnalysis

router = APIRouter(prefix="/api/problem", tags=["problem"])


# Dependency to get database (will be injected)
def get_db():
    pass


@router.post("/create", response_model=ProblemCreate, status_code=status.HTTP_201_CREATED)
async def create_problem(problem_input: ProblemInput, db: AsyncIOMotorDatabase = None):
    """
    Create a new optimization problem
    
    Args:
        problem_input: Problem details from user
        db: Database connection
        
    Returns:
        Created problem with ID
    """
    # Create problem object
    problem = ProblemCreate(problem_input=problem_input)
    
    # Save to database if db provided
    if db:
        problem_dict = problem.model_dump()
        await db.problems.insert_one(problem_dict)
    
    return problem


@router.get("/{problem_id}", response_model=ProblemWithAnalysis)
async def get_problem(problem_id: str, db: AsyncIOMotorDatabase = None):
    """
    Get a problem by ID
    
    Args:
        problem_id: Problem identifier
        db: Database connection
        
    Returns:
        Problem with analysis if available
    """
    if not db:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not configured"
        )
    
    problem = await db.problems.find_one({"problem_id": problem_id})
    
    if not problem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Problem {problem_id} not found"
        )
    
    # Remove MongoDB _id field
    problem.pop("_id", None)
    
    # Get analysis if exists
    analysis = await db.analyses.find_one({"problem_id": problem_id})
    if analysis:
        analysis.pop("_id", None)
        return ProblemWithAnalysis(problem=problem, analysis=analysis)
    
    return ProblemWithAnalysis(problem=problem, analysis=None)


@router.get("/", response_model=List[ProblemCreate])
async def list_problems(db: AsyncIOMotorDatabase = None, limit: int = 50):
    """
    List all problems
    
    Args:
        db: Database connection
        limit: Maximum number of problems to return
        
    Returns:
        List of problems
    """
    if not db:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not configured"
        )
    
    problems = await db.problems.find().sort("created_at", -1).limit(limit).to_list(length=limit)
    
    # Remove MongoDB _id field
    for problem in problems:
        problem.pop("_id", None)
    
    return problems
