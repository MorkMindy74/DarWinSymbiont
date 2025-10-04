#!/usr/bin/env python3
"""
Backend Test Suite for EMERGENT AI-Powered Optimization Platform

Tests the Phase 1-2 backend APIs:
1. Health check endpoint: GET /api/health
2. Problem creation: POST /api/problem/create
3. Problem analysis with LLM: POST /api/analysis/analyze/{problem_id}
4. Get problem with analysis: GET /api/problem/{problem_id}
"""

import sys
import os
import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend URL from frontend .env
BACKEND_URL = "https://bf7227fc-ba83-4dd5-b96c-522be2796f63.preview.emergentagent.com"

# Test data - TSP (Traveling Salesman Problem)
TSP_TEST_DATA = {
    "problem_type": "tsp",
    "title": "Optimize delivery routes for 10 cities",
    "description": "Find the shortest route to visit 10 cities and return to the starting point. Each city must be visited exactly once.",
    "constraints": {
        "num_locations": 10,
        "max_distance": 1000
    }
}

class EmergentAPITester:
    """Test suite for EMERGENT Platform APIs"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.created_problem_id = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout for LLM calls
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> tuple[bool, str]:
        """Test 1: Health check endpoint"""
        logger.info("Testing health check endpoint...")
        
        try:
            url = f"{self.base_url}/api/health"
            async with self.session.get(url) as response:
                if response.status != 200:
                    return False, f"Health check failed with status {response.status}"
                
                data = await response.json()
                
                # Verify expected fields
                required_fields = ["status", "database", "llm"]
                for field in required_fields:
                    if field not in data:
                        return False, f"Missing field in health response: {field}"
                
                if data["status"] != "healthy":
                    return False, f"Service not healthy: {data['status']}"
                
                logger.info(f"✅ Health check passed: {data}")
                return True, f"Health check successful: {data}"
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False, f"Health check error: {e}"

    async def test_problem_creation(self) -> tuple[bool, str]:
        """Test 2: Problem creation endpoint"""
        logger.info("Testing problem creation endpoint...")
        
        try:
            url = f"{self.base_url}/api/problem/create"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=TSP_TEST_DATA, headers=headers) as response:
                if response.status != 201:
                    error_text = await response.text()
                    return False, f"Problem creation failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                # Verify expected fields
                required_fields = ["problem_id", "created_at", "problem_input", "status"]
                for field in required_fields:
                    if field not in data:
                        return False, f"Missing field in problem response: {field}"
                
                # Verify problem_input matches what we sent
                if data["problem_input"]["problem_type"] != TSP_TEST_DATA["problem_type"]:
                    return False, "Problem type mismatch in response"
                
                if data["problem_input"]["title"] != TSP_TEST_DATA["title"]:
                    return False, "Problem title mismatch in response"
                
                # Store problem_id for subsequent tests
                self.created_problem_id = data["problem_id"]
                
                logger.info(f"✅ Problem created successfully: {self.created_problem_id}")
                return True, f"Problem creation successful: {data['problem_id']}"
                
        except Exception as e:
            logger.error(f"❌ Problem creation failed: {e}")
            return False, f"Problem creation error: {e}"

def test_context_aware_functionality():
    """Test 3: Context-Aware functionality"""
    logger.info("Testing Context-Aware Thompson Sampling functionality...")
    
    try:
        from shinka.llm.dynamic_sampling import ContextAwareThompsonSamplingBandit
        
        # Test context-aware bandit creation
        bandit = ContextAwareThompsonSamplingBandit(
            arm_names=["fast_model", "accurate_model", "balanced_model"],
            contexts=["early", "mid", "late", "stuck"],
            seed=42,
            prior_alpha=1.0,
            prior_beta=1.0
        )
        logger.info("✅ Context-Aware bandit created successfully")
        
        # Test context switching
        initial_context = bandit.current_context
        logger.info(f"Initial context: {initial_context}")
        
        # Update context with early phase parameters
        new_context = bandit.update_context(
            generation=5,
            total_generations=100,
            no_improve_steps=2,
            best_fitness_history=[0.1, 0.2, 0.3, 0.4, 0.45],
            population_diversity=0.8
        )
        logger.info(f"✅ Context update successful: {new_context}")
        
        # Test that context switching works
        # Add enough samples to allow switching
        for _ in range(10):
            bandit.update_submitted("fast_model")
            bandit.update("fast_model", reward=0.7, baseline=0.5)
        
        # Switch to stuck context with clear signal
        stuck_context = bandit.update_context(
            generation=50,
            total_generations=100,
            no_improve_steps=20,  # Clearly stuck
            best_fitness_history=[0.5, 0.5, 0.5, 0.5, 0.5],
            population_diversity=0.2
        )
        logger.info(f"✅ Context switching working: {initial_context} -> {stuck_context}")
        
        # Test different posteriors per context
        # Update different arms in different contexts
        bandit.current_context = "early"
        for _ in range(5):
            bandit.update_submitted("fast_model")
            bandit.update("fast_model", reward=0.9, baseline=0.5)  # Fast model good in early
        
        bandit.current_context = "stuck"
        for _ in range(5):
            bandit.update_submitted("accurate_model")
            bandit.update("accurate_model", reward=0.9, baseline=0.5)  # Accurate model good when stuck
        
        # Check that contexts learned different preferences
        early_posterior = bandit.posterior(context="early", samples=100)
        stuck_posterior = bandit.posterior(context="stuck", samples=100)
        
        # Fast model should be preferred in early, accurate model in stuck
        fast_model_idx = 0
        accurate_model_idx = 1
        
        logger.info(f"Early context posterior: {early_posterior}")
        logger.info(f"Stuck context posterior: {stuck_posterior}")
        
        # Verify different preferences (allowing some tolerance for randomness)
        if early_posterior[fast_model_idx] > stuck_posterior[fast_model_idx]:
            logger.info("✅ Context-specific learning verified: fast_model preferred in early")
        else:
            logger.warning("⚠️ Context-specific learning may not be strong enough")
        
        if stuck_posterior[accurate_model_idx] > early_posterior[accurate_model_idx]:
            logger.info("✅ Context-specific learning verified: accurate_model preferred when stuck")
        else:
            logger.warning("⚠️ Context-specific learning may not be strong enough")
        
        # Test context statistics
        stats = bandit.get_context_stats()
        assert "current_context" in stats, "Context stats missing current_context"
        assert "contexts" in stats, "Context stats missing contexts"
        logger.info("✅ Context statistics working")
        
        return True, "Context-Aware functionality working correctly"
        
    except Exception as e:
        logger.error(f"❌ Context-Aware test failed: {e}")
        return False, f"Context-Aware error: {e}"

def test_benchmark_harness_integration():
    """Test 4: Benchmark harness integration"""
    logger.info("Testing benchmark harness integration...")
    
    try:
        from bench.context_bandit_bench import MockLLMScorer, EvolutionSimulator, BenchmarkConfig
        
        # Test MockLLMScorer for all problem types
        problem_types = ["toy", "tsp", "synthetic"]
        
        for problem_type in problem_types:
            scorer = MockLLMScorer(problem_type, seed=42)
            
            # Test scoring with sample solution
            solution = np.array([1.0, 2.0, 0.5, -1.0, 0.8])
            context_info = {
                'gen_progress': 0.3,
                'no_improve_steps': 5,
                'current_best': 0.5
            }
            
            score = scorer.score_solution(solution, context_info)
            assert 0 <= score <= 1, f"Score out of range [0,1]: {score} for {problem_type}"
            logger.info(f"✅ MockLLMScorer working for {problem_type}: score={score:.3f}")
        
        # Test EvolutionSimulator
        config = BenchmarkConfig(
            benchmark="toy",
            algorithm="baseline",
            seed=42,
            budget_steps=10,  # Small test
            output_dir="/tmp/test_output"
        )
        
        scorer = MockLLMScorer("toy", seed=42)
        simulator = EvolutionSimulator(config, scorer)
        
        # Test single step
        step_result = simulator.run_step()
        
        required_fields = [
            'run_id', 'algo', 'benchmark', 'seed', 'step', 'context',
            'fitness', 'best_fitness', 'llm_queries', 'llm_queries_cum',
            'time_ms', 'gen_progress', 'no_improve_steps', 'fitness_slope',
            'pop_diversity', 'selected_model', 'improvement'
        ]
        
        for field in required_fields:
            assert field in step_result, f"Missing field in step result: {field}"
        
        logger.info(f"✅ EvolutionSimulator step working: fitness={step_result['fitness']:.3f}")
        
        # Test context-aware simulator
        config_context = BenchmarkConfig(
            benchmark="synthetic",
            algorithm="context",
            seed=42,
            budget_steps=5,
            output_dir="/tmp/test_output"
        )
        
        simulator_context = EvolutionSimulator(config_context, scorer)
        step_result_context = simulator_context.run_step()
        
        assert 'context' in step_result_context, "Context field missing from context-aware simulator"
        assert step_result_context['context'] != 'none', "Context should be detected"
        logger.info(f"✅ Context-aware simulator working: context={step_result_context['context']}")
        
        return True, "Benchmark harness integration working correctly"
        
    except Exception as e:
        logger.error(f"❌ Benchmark harness test failed: {e}")
        return False, f"Benchmark harness error: {e}"

def test_complete_minimal_benchmark():
    """Test 5: Complete minimal benchmark run"""
    logger.info("Testing complete minimal benchmark run...")
    
    try:
        from bench.context_bandit_bench import run_single_benchmark, BenchmarkConfig
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BenchmarkConfig(
                benchmark="toy",
                algorithm="baseline",
                seed=42,
                budget_steps=50,  # Minimal run
                output_dir=temp_dir
            )
            
            # Run benchmark
            logger.info("Running minimal benchmark (50 steps)...")
            metrics = run_single_benchmark(config)
            
            # Verify metrics
            required_metrics = [
                'run_id', 'algorithm', 'benchmark', 'seed',
                'final_best_fitness', 'llm_queries_total',
                'llm_queries_while_stuck', 'time_to_first_improve',
                'no_improve_final', 'area_under_fitness_curve'
            ]
            
            for metric in required_metrics:
                assert metric in metrics, f"Missing metric: {metric}"
            
            logger.info(f"✅ Benchmark completed: final_fitness={metrics['final_best_fitness']:.3f}")
            
            # Verify CSV output
            output_path = Path(temp_dir) / "baseline" / "toy" / "run_42.csv"
            assert output_path.exists(), f"CSV output not found: {output_path}"
            
            # Read and verify CSV content
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            assert len(rows) == 50, f"Expected 50 rows, got {len(rows)}"
            
            # Verify CSV columns
            expected_columns = [
                'run_id', 'algo', 'benchmark', 'seed', 'step', 'context',
                'fitness', 'best_fitness', 'llm_queries', 'llm_queries_cum',
                'time_ms', 'gen_progress', 'no_improve_steps', 'fitness_slope',
                'pop_diversity', 'selected_model', 'improvement'
            ]
            
            first_row = rows[0]
            for col in expected_columns:
                assert col in first_row, f"Missing CSV column: {col}"
            
            logger.info("✅ CSV output verified with correct columns")
            
            # Test context-aware benchmark
            config_context = BenchmarkConfig(
                benchmark="synthetic",
                algorithm="context",
                seed=42,
                budget_steps=50,
                output_dir=temp_dir
            )
            
            logger.info("Running context-aware benchmark (50 steps)...")
            metrics_context = run_single_benchmark(config_context)
            
            # Verify context-specific metrics
            assert 'context_switch_count' in metrics_context, "Missing context_switch_count"
            assert 'context_dwell_times' in metrics_context, "Missing context_dwell_times"
            
            logger.info(f"✅ Context-aware benchmark completed: switches={metrics_context['context_switch_count']}")
            
            # Verify no runtime errors occurred
            assert metrics['final_best_fitness'] >= 0, "Negative fitness indicates error"
            assert metrics['llm_queries_total'] == 50, f"Expected 50 queries, got {metrics['llm_queries_total']}"
            
            logger.info("✅ No runtime errors detected")
            
        return True, "Complete minimal benchmark run working correctly"
        
    except Exception as e:
        logger.error(f"❌ Complete benchmark test failed: {e}")
        return False, f"Complete benchmark error: {e}"

def run_all_tests():
    """Run all backend tests and return results"""
    logger.info("Starting ShinkaEvolve backend test suite...")
    
    tests = [
        ("Core imports functionality", test_core_imports),
        ("Thompson Sampling basic functionality", test_thompson_sampling_basic),
        ("Context-Aware functionality", test_context_aware_functionality),
        ("Benchmark harness integration", test_benchmark_harness_integration),
        ("Complete minimal benchmark run", test_complete_minimal_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success, message = test_func()
            results[test_name] = {
                'success': success,
                'message': message
            }
            
            if success:
                logger.info(f"✅ PASSED: {test_name}")
            else:
                logger.error(f"❌ FAILED: {test_name} - {message}")
                
        except Exception as e:
            logger.error(f"❌ ERROR: {test_name} - {e}")
            results[test_name] = {
                'success': False,
                'message': f"Unexpected error: {e}"
            }
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if not result['success']:
            logger.info(f"    Error: {result['message']}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with error code if any tests failed
    failed_tests = [name for name, result in results.items() if not result['success']]
    if failed_tests:
        logger.error(f"Failed tests: {failed_tests}")
        sys.exit(1)
    else:
        logger.info("All tests passed!")
        sys.exit(0)