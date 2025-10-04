"""
LLM Service for problem analysis using Emergent Universal Key
"""
import os
from typing import Optional
from dotenv import load_dotenv
from emergentintegrations.llm.chat import LlmChat, UserMessage

load_dotenv()


class LLMService:
    """Service for LLM interactions"""
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4o"):
        self.api_key = os.getenv("EMERGENT_LLM_KEY")
        if not self.api_key:
            raise ValueError("EMERGENT_LLM_KEY not found in environment")
        
        self.model_provider = model_provider
        self.model_name = model_name
        
    async def analyze_problem(self, problem_type: str, description: str, constraints: dict) -> str:
        """
        Analyze a problem and return structured analysis
        
        Args:
            problem_type: Type of problem (tsp, tsp_tw, scheduling, etc.)
            description: User's problem description
            constraints: Problem constraints
            
        Returns:
            JSON string with structured analysis
        """
        system_message = """You are an expert in optimization problems and evolutionary algorithms. 
You analyze user-provided optimization problems and provide detailed, structured analysis 
for configuring evolutionary search algorithms. Your responses must be in valid JSON format."""

        # Create problem-specific analysis prompt
        user_prompt = self._create_analysis_prompt(problem_type, description, constraints)
        
        # Initialize chat with unique session
        chat = LlmChat(
            api_key=self.api_key,
            session_id=f"problem_analysis_{problem_type}",
            system_message=system_message
        ).with_model(self.model_provider, self.model_name)
        
        # Send message
        user_message = UserMessage(text=user_prompt)
        response = await chat.send_message(user_message)
        
        return response
    
    def _create_analysis_prompt(self, problem_type: str, description: str, constraints: dict) -> str:
        """Create problem-specific analysis prompt"""
        
        base_prompt = f"""Analyze the following {problem_type.upper()} optimization problem:

**Problem Description:**
{description}

**Constraints:**
{self._format_constraints(constraints)}

Please provide a comprehensive analysis in the following JSON format:
{{
  "problem_characterization": "Brief characterization of this specific problem instance",
  "complexity_assessment": "Assessment of computational complexity and search space size",
  "key_challenges": ["challenge1", "challenge2", "challenge3"],
  "parameter_suggestions": [
    {{
      "name": "parameter_name",
      "value": "suggested_value",
      "description": "What this parameter controls",
      "rationale": "Why this value is suggested",
      "adjustable": true
    }}
  ],
  "constraints_analysis": [
    {{
      "constraint_type": "constraint_name",
      "description": "What this constraint means",
      "importance": "critical|high|medium|low",
      "impact_on_solution": "How this affects solution quality"
    }}
  ],
  "solution_strategy": "Recommended approach for solving this problem",
  "estimated_search_space": "Estimate of solution space size (e.g., 'n!', '2^n', etc.)",
  "recommended_evolution_config": {{
    "population_size": 50,
    "num_generations": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "selection_method": "tournament",
    "other_params": {{}}
  }}
}}

"""
        
        # Add problem-specific guidance
        if problem_type == "tsp":
            base_prompt += """
**Specific Guidance for TSP:**
- Consider the number of cities/locations
- Analyze distance matrix properties if available
- Suggest appropriate mutation operators (2-opt, 3-opt, etc.)
- Recommend local search strategies
"""
        elif problem_type == "tsp_tw":
            base_prompt += """
**Specific Guidance for TSP with Time Windows:**
- Analyze time window constraints critically
- Consider feasibility vs optimality tradeoffs
- Suggest penalty functions for time window violations
- Recommend repair mechanisms
"""
        elif problem_type == "scheduling":
            base_prompt += """
**Specific Guidance for Scheduling:**
- Identify resource constraints
- Analyze precedence relationships
- Consider makespan vs other objectives
- Suggest appropriate scheduling heuristics
"""
        
        return base_prompt
    
    def _format_constraints(self, constraints: dict) -> str:
        """Format constraints for display"""
        formatted = []
        for key, value in constraints.items():
            if value is not None:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted) if formatted else "No specific constraints provided"
    
    async def generate_code(self, problem_type: str, analysis: dict, code_type: str = "evaluate") -> str:
        """
        Generate evaluate.py or initial.py code for the problem
        
        Args:
            problem_type: Type of problem
            analysis: Problem analysis dict
            code_type: "evaluate" or "initial"
            
        Returns:
            Generated Python code as string
        """
        system_message = """You are an expert Python programmer specializing in optimization algorithms.
You generate clean, efficient, well-documented code for evolutionary algorithms."""

        if code_type == "evaluate":
            prompt = self._create_evaluate_prompt(problem_type, analysis)
        else:
            prompt = self._create_initial_prompt(problem_type, analysis)
        
        chat = LlmChat(
            api_key=self.api_key,
            session_id=f"code_gen_{problem_type}_{code_type}",
            system_message=system_message
        ).with_model(self.model_provider, self.model_name)
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
        return response
    
    def _create_evaluate_prompt(self, problem_type: str, analysis: dict) -> str:
        """Create prompt for evaluate.py generation"""
        return f"""Generate an evaluate.py file for a {problem_type.upper()} problem.

**Problem Context:**
{analysis.get('problem_characterization', '')}

**Key Constraints:**
{analysis.get('constraints_analysis', [])}

The evaluate.py file should:
1. Define a fitness function that evaluates solution quality
2. Handle constraint violations appropriately
3. Return a single float score (higher is better)
4. Be efficient for repeated evaluations
5. Include proper error handling

Generate ONLY the Python code, no explanations. Start with imports.
"""
    
    def _create_initial_prompt(self, problem_type: str, analysis: dict) -> str:
        """Create prompt for initial.py generation"""
        return f"""Generate an initial.py file for a {problem_type.upper()} problem.

**Problem Context:**
{analysis.get('problem_characterization', '')}

**Solution Strategy:**
{analysis.get('solution_strategy', '')}

The initial.py file should:
1. Generate a valid initial solution
2. Respect all constraints
3. Use appropriate heuristics for the problem type
4. Be randomized but not completely random
5. Include proper documentation

Generate ONLY the Python code, no explanations. Start with imports.
"""
