"""
Quick test for LLM service
"""
import asyncio
import sys
import os

# Add parent to path for imports
sys.path.insert(0, '/app')

from backend.services.llm_service import LLMService


async def test_llm():
    """Test LLM integration"""
    print("🧪 Testing LLM Service...")
    
    try:
        llm = LLMService()
        print(f"✅ LLM Service initialized with key: {llm.api_key[:20]}...")
        
        # Test simple problem analysis
        problem_type = "tsp"
        description = "Find the shortest route visiting 10 cities"
        constraints = {"num_locations": 10, "max_distance": 1000}
        
        print(f"\n📝 Analyzing {problem_type.upper()} problem...")
        response = await llm.analyze_problem(problem_type, description, constraints)
        
        print(f"\n✅ LLM Response received ({len(response)} chars)")
        print(f"\n📄 Response preview:\n{response[:500]}...\n")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_llm())
    sys.exit(0 if result else 1)
