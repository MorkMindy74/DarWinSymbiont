"""
Analysis Service for problem processing
"""
import json
from typing import Dict, Any
from services.llm_service import LLMService
from models.problem import (
    ProblemInput, 
    ProblemAnalysis,
    ParameterSuggestion,
    ConstraintAnalysis
)


class AnalysisService:
    """Service for analyzing problems"""
    
    def __init__(self):
        self.llm_service = LLMService()
    
    async def analyze_problem(self, problem_id: str, problem_input: ProblemInput) -> ProblemAnalysis:
        """
        Analyze a problem using LLM
        
        Args:
            problem_id: Unique problem identifier
            problem_input: Problem input data
            
        Returns:
            Structured problem analysis
        """
        # Convert constraints to dict
        constraints_dict = problem_input.constraints.model_dump(exclude_none=True)
        
        # Get LLM analysis
        llm_response = await self.llm_service.analyze_problem(
            problem_type=problem_input.problem_type,
            description=problem_input.description,
            constraints=constraints_dict
        )
        
        # Parse LLM response (it should be JSON)
        try:
            analysis_data = json.loads(llm_response)
        except json.JSONDecodeError:
            # Fallback: extract JSON from markdown code blocks if present
            llm_response = llm_response.strip()
            if "```json" in llm_response:
                json_start = llm_response.find("```json") + 7
                json_end = llm_response.find("```", json_start)
                json_str = llm_response[json_start:json_end].strip()
                analysis_data = json.loads(json_str)
            elif "```" in llm_response:
                json_start = llm_response.find("```") + 3
                json_end = llm_response.find("```", json_start)
                json_str = llm_response[json_start:json_end].strip()
                analysis_data = json.loads(json_str)
            else:
                raise ValueError("Could not parse LLM response as JSON")
        
        # Convert to Pydantic models
        parameter_suggestions = [
            ParameterSuggestion(**param) 
            for param in analysis_data.get("parameter_suggestions", [])
        ]
        
        constraints_analysis = [
            ConstraintAnalysis(**constraint) 
            for constraint in analysis_data.get("constraints_analysis", [])
        ]
        
        # Create ProblemAnalysis object
        analysis = ProblemAnalysis(
            problem_id=problem_id,
            problem_characterization=analysis_data.get("problem_characterization", ""),
            complexity_assessment=analysis_data.get("complexity_assessment", ""),
            key_challenges=analysis_data.get("key_challenges", []),
            parameter_suggestions=parameter_suggestions,
            constraints_analysis=constraints_analysis,
            solution_strategy=analysis_data.get("solution_strategy", ""),
            estimated_search_space=analysis_data.get("estimated_search_space", ""),
            recommended_evolution_config=analysis_data.get("recommended_evolution_config", {})
        )
        
        return analysis
    
    async def generate_evaluation_code(
        self, 
        problem_type: str, 
        analysis: ProblemAnalysis
    ) -> str:
        """Generate evaluate.py code"""
        analysis_dict = analysis.model_dump()
        code = await self.llm_service.generate_code(
            problem_type=problem_type,
            analysis=analysis_dict,
            code_type="evaluate"
        )
        return code
    
    async def generate_initial_code(
        self, 
        problem_type: str, 
        analysis: ProblemAnalysis
    ) -> str:
        """Generate initial.py code"""
        analysis_dict = analysis.model_dump()
        code = await self.llm_service.generate_code(
            problem_type=problem_type,
            analysis=analysis_dict,
            code_type="initial"
        )
        return code
