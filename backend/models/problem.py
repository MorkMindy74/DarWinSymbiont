"""
Pydantic models for problem domain and analysis
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid


class ProblemConstraints(BaseModel):
    """Constraints specific to problem type"""
    max_distance: Optional[float] = None
    max_time: Optional[float] = None
    num_locations: Optional[int] = None
    time_windows: Optional[List[Dict[str, Any]]] = None
    vehicles: Optional[int] = None
    capacity: Optional[float] = None
    custom: Optional[Dict[str, Any]] = None


class ProblemInput(BaseModel):
    """Problem input from user"""
    problem_type: Literal["tsp", "tsp_tw", "scheduling", "circle_packing", "nas"]
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10)
    constraints: ProblemConstraints
    dataset_url: Optional[str] = None
    dataset_content: Optional[str] = None


class ProblemCreate(BaseModel):
    """Problem creation request"""
    problem_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    problem_input: ProblemInput
    status: str = "created"


class ParameterSuggestion(BaseModel):
    """Suggested parameter for the problem"""
    name: str
    value: Any
    description: str
    rationale: str
    adjustable: bool = True


class ConstraintAnalysis(BaseModel):
    """Analysis of a constraint"""
    constraint_type: str
    description: str
    importance: Literal["critical", "high", "medium", "low"]
    impact_on_solution: str


class ProblemAnalysis(BaseModel):
    """LLM-generated problem analysis"""
    problem_id: str
    problem_characterization: str
    complexity_assessment: str
    key_challenges: List[str]
    parameter_suggestions: List[ParameterSuggestion]
    constraints_analysis: List[ConstraintAnalysis]
    solution_strategy: str
    estimated_search_space: str
    recommended_evolution_config: Dict[str, Any]
    analysis_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ProblemWithAnalysis(BaseModel):
    """Complete problem with analysis"""
    problem: ProblemCreate
    analysis: Optional[ProblemAnalysis] = None
