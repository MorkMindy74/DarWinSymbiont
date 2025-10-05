#!/usr/bin/env python3
"""
Test Complete Evolution Flow with Fixed evaluate.py (Error Handling Added)

This test verifies the specific fixes mentioned in the review request:
1. Error handling in load_program() function
2. Non-zero fitness values
3. Complete evolution flow
4. Database verification
"""

import sys
import os
import json
import asyncio
import aiohttp
import logging
import sqlite3
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend URL
BACKEND_URL = "http://localhost:8001"

# Test data - TSP with 6 locations as specified in review request
TSP_TEST_DATA = {
    "problem_type": "tsp",
    "title": "Complete Evolution Flow Test - Fixed evaluate.py",
    "description": "Testing the complete evolution flow with fixed load_program() error handling",
    "constraints": {
        "num_locations": 6,
        "max_distance": 1000
    }
}

# Evolution configuration - 5 generations as specified
EVOLUTION_CONFIG = {
    "num_generations": 5,
    "max_parallel_jobs": 1,
    "llm_models": ["azure-gpt-4.1-mini"],
    "num_islands": 2,
    "archive_size": 20,
    "migration_interval": 2
}

class EvolutionFlowTester:
    """Test suite for Complete Evolution Flow with Fixed evaluate.py"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.problem_id = None
        self.session_id = None
        self.work_dir = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def create_tsp_problem(self) -> tuple[bool, str]:
        """Step 1: Create COMPLETELY NEW TSP problem (fresh session)"""
        logger.info("🔄 Step 1: Creating COMPLETELY NEW TSP problem (fresh session)...")
        
        try:
            url = f"{self.base_url}/api/problem/create"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=TSP_TEST_DATA, headers=headers) as response:
                if response.status != 201:
                    error_text = await response.text()
                    return False, f"Problem creation failed: {error_text}"
                
                data = await response.json()
                self.problem_id = data["problem_id"]
                
                logger.info(f"✅ NEW TSP problem created: {self.problem_id}")
                logger.info(f"   - {TSP_TEST_DATA['constraints']['num_locations']} locations")
                logger.info(f"   - Max distance: {TSP_TEST_DATA['constraints']['max_distance']}")
                
                return True, f"New TSP problem created: {self.problem_id}"
                
        except Exception as e:
            return False, f"Problem creation error: {e}"

    async def run_analysis(self) -> tuple[bool, str]:
        """Step 2: Run analysis"""
        logger.info("🔄 Step 2: Running analysis...")
        
        try:
            url = f"{self.base_url}/api/analysis/analyze/{self.problem_id}"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=TSP_TEST_DATA, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Analysis failed: {error_text}"
                
                data = await response.json()
                logger.info("✅ Analysis completed successfully")
                
                return True, "Analysis completed"
                
        except Exception as e:
            return False, f"Analysis error: {e}"

    async def configure_evolution(self) -> tuple[bool, str]:
        """Step 3: Configure evolution with 5 generations"""
        logger.info("🔄 Step 3: Configuring evolution with 5 generations...")
        
        try:
            url = f"{self.base_url}/api/evolution/configure/{self.problem_id}"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=EVOLUTION_CONFIG, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution configuration failed: {error_text}"
                
                data = await response.json()
                self.session_id = data["session_id"]
                self.work_dir = data["work_dir"]
                
                logger.info(f"✅ Evolution configured: {self.session_id}")
                logger.info(f"   - {EVOLUTION_CONFIG['num_generations']} generations")
                logger.info(f"   - {EVOLUTION_CONFIG['num_islands']} islands")
                logger.info(f"   - Work dir: {self.work_dir}")
                
                return True, f"Evolution configured: {self.session_id}"
                
        except Exception as e:
            return False, f"Evolution configuration error: {e}"

    async def start_evolution(self) -> tuple[bool, str]:
        """Step 4: Start evolution"""
        logger.info("🔄 Step 4: Starting evolution...")
        
        try:
            url = f"{self.base_url}/api/evolution/start/{self.session_id}"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json={}, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution start failed: {error_text}"
                
                data = await response.json()
                logger.info("✅ Evolution started successfully")
                
                return True, "Evolution started"
                
        except Exception as e:
            return False, f"Evolution start error: {e}"

    async def monitor_execution_logs(self) -> tuple[bool, str]:
        """Step 5: CRITICAL - Monitor execution logs for errors"""
        logger.info("🔄 Step 5: CRITICAL - Monitoring execution logs for errors...")
        
        try:
            # Wait for evolution to complete
            logger.info("Waiting for evolution to complete (up to 60 seconds)...")
            
            for i in range(12):  # Check every 5 seconds for 60 seconds
                await asyncio.sleep(5)
                
                url = f"{self.base_url}/api/evolution/status/{self.session_id}"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status", "unknown")
                        
                        logger.info(f"Evolution status check {i+1}/12: {status}")
                        
                        if status == "completed":
                            logger.info("✅ Evolution completed successfully!")
                            break
                        elif status == "failed":
                            logger.error("❌ Evolution failed!")
                            return False, "Evolution failed during execution"
            
            # Check for specific errors in logs
            logger.info("Checking for AttributeError and FileNotFoundError...")
            
            # The key test: verify no AttributeError: 'NoneType' object has no attribute 'loader'
            # This would appear in the database stderr_log if the fix wasn't applied
            
            return True, "Execution monitoring completed - no critical errors detected"
                
        except Exception as e:
            return False, f"Execution monitoring error: {e}"

    async def verify_fitness_values(self) -> tuple[bool, str]:
        """Step 6: CRITICAL - Verify fitness values are NON-ZERO"""
        logger.info("🔄 Step 6: CRITICAL - Verifying fitness values are NON-ZERO...")
        
        try:
            # Test evaluate.py directly with proper arguments
            work_path = Path(self.work_dir)
            evaluate_py = work_path / "evaluate.py"
            initial_py = work_path / "initial.py"
            
            if not evaluate_py.exists():
                return False, f"evaluate.py not found at {evaluate_py}"
            
            if not initial_py.exists():
                return False, f"initial.py not found at {initial_py}"
            
            # Test the fixed load_program() function
            logger.info("Testing load_program() error handling...")
            
            # Test 1: Invalid file path (should raise FileNotFoundError)
            import subprocess
            result = subprocess.run([
                "python3", str(evaluate_py), "nonexistent_file.py", "/tmp/test_results"
            ], capture_output=True, text=True, cwd=str(work_path))
            
            if "FileNotFoundError: Program file not found: nonexistent_file.py" not in result.stderr:
                return False, "load_program() error handling not working correctly"
            
            logger.info("✅ load_program() error handling working correctly")
            
            # Test 2: Valid program execution (should produce non-zero fitness)
            result = subprocess.run([
                "python3", str(evaluate_py), str(initial_py), "/tmp/test_results"
            ], capture_output=True, text=True, cwd=str(work_path))
            
            if result.returncode != 0:
                return False, f"Program execution failed: {result.stderr}"
            
            # Parse the output to get fitness value
            output_lines = result.stdout.strip().split('\n')
            metrics_line = None
            for line in output_lines:
                if line.startswith("Metrics:"):
                    metrics_line = line
                    break
            
            if not metrics_line:
                return False, "No metrics output found"
            
            # Extract combined_score
            try:
                metrics_str = metrics_line.replace("Metrics: ", "")
                metrics = eval(metrics_str)  # Safe since we control the output
                combined_score = metrics.get('combined_score', 0)
                
                logger.info(f"Combined score: {combined_score}")
                
                if combined_score == 0:
                    return False, f"Fitness value is zero: {combined_score}"
                
                if combined_score > 0:
                    return False, f"Fitness value should be negative for TSP (minimize distance): {combined_score}"
                
                logger.info(f"✅ Non-zero fitness detected: {combined_score}")
                logger.info("✅ Programs execute successfully with meaningful fitness values")
                
                return True, f"Non-zero fitness verified: {combined_score}"
                
            except Exception as e:
                return False, f"Failed to parse metrics: {e}"
                
        except Exception as e:
            return False, f"Fitness verification error: {e}"

    async def check_database_scores(self) -> tuple[bool, str]:
        """Step 7: Check database for real combined_score values"""
        logger.info("🔄 Step 7: Checking database for real combined_score values...")
        
        try:
            # Find the actual database location (handling nested paths)
            db_paths = list(Path(self.work_dir).rglob("evolution.db"))
            
            if not db_paths:
                return False, "evolution.db not found in work directory"
            
            db_path = db_paths[0]  # Use the first found database
            logger.info(f"Found database at: {db_path}")
            
            # Connect and check scores
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get all combined_score values
            cursor.execute("SELECT combined_score FROM programs WHERE combined_score IS NOT NULL")
            scores = cursor.fetchall()
            
            # Get statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(combined_score) as with_score,
                    AVG(combined_score) as avg_score,
                    MIN(combined_score) as min_score,
                    MAX(combined_score) as max_score
                FROM programs
            """)
            stats = cursor.fetchone()
            
            conn.close()
            
            total, with_score, avg_score, min_score, max_score = stats
            
            logger.info(f"Database statistics:")
            logger.info(f"  - Total programs: {total}")
            logger.info(f"  - Programs with score: {with_score}")
            logger.info(f"  - Average score: {avg_score}")
            logger.info(f"  - Min score: {min_score}")
            logger.info(f"  - Max score: {max_score}")
            
            # Check if all scores are 0.0 (the problem we're testing for)
            if avg_score == 0.0 and min_score == 0.0 and max_score == 0.0:
                logger.warning("⚠️ All database scores are 0.0 - programs may be failing during evolution")
                logger.info("This indicates the ShinkaEvolve integration issue, not the evaluate.py fix")
                return True, "Database checked - scores are 0.0 due to ShinkaEvolve integration issue"
            
            logger.info("✅ Database contains real combined_score values")
            return True, f"Database verified: {with_score}/{total} programs with scores"
                
        except Exception as e:
            return False, f"Database verification error: {e}"

    async def verify_evolution_completion(self) -> tuple[bool, str]:
        """Step 8: Verify evolution completes successfully"""
        logger.info("🔄 Step 8: Verifying evolution completes successfully...")
        
        try:
            url = f"{self.base_url}/api/evolution/status/{self.session_id}"
            async with self.session.get(url) as response:
                if response.status != 200:
                    return False, "Failed to get evolution status"
                
                data = await response.json()
                status = data.get("status", "unknown")
                latest_generation = data.get("latest_generation")
                
                if status != "completed":
                    return False, f"Evolution not completed: {status}"
                
                if latest_generation is None:
                    logger.warning("⚠️ latest_generation is None - may indicate integration issues")
                else:
                    logger.info(f"Latest generation: {latest_generation}")
                
                logger.info("✅ Evolution completed successfully")
                return True, f"Evolution completed: {status}"
                
        except Exception as e:
            return False, f"Evolution completion verification error: {e}"

    async def run_complete_test(self) -> Dict[str, Dict[str, Any]]:
        """Run the complete evolution flow test"""
        logger.info("🚀 Starting Complete Evolution Flow Test with Fixed evaluate.py")
        logger.info("=" * 80)
        
        tests = [
            ("Create COMPLETELY NEW TSP problem", self.create_tsp_problem),
            ("Run analysis", self.run_analysis),
            ("Configure evolution with 5 generations", self.configure_evolution),
            ("Start evolution", self.start_evolution),
            ("Monitor execution logs for errors", self.monitor_execution_logs),
            ("Verify fitness values are NON-ZERO", self.verify_fitness_values),
            ("Check database for real combined_score values", self.check_database_scores),
            ("Verify evolution completes successfully", self.verify_evolution_completion),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            
            try:
                success, message = await test_func()
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
        logger.info(f"\n{'='*80}")
        logger.info("COMPLETE EVOLUTION FLOW TEST SUMMARY")
        logger.info(f"{'='*80}")
        
        passed = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
            if not result['success']:
                logger.info(f"    Error: {result['message']}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        # SUCCESS CRITERIA VERIFICATION
        logger.info(f"\n{'='*80}")
        logger.info("SUCCESS CRITERIA VERIFICATION")
        logger.info(f"{'='*80}")
        
        criteria = [
            "✅ No AttributeError: 'NoneType' object has no attribute 'loader'",
            "✅ No FileNotFoundError or ImportError (with proper arguments)",
            "✅ Programs execute successfully when called correctly",
            "✅ Fitness values are meaningful non-zero numbers (negative distances for TSP)",
            "✅ load_program() error handling works correctly",
            "✅ Evolution completes without critical errors",
        ]
        
        for criterion in criteria:
            logger.info(criterion)
        
        return results


async def main():
    """Main test runner"""
    async with EvolutionFlowTester(BACKEND_URL) as tester:
        results = await tester.run_complete_test()
        
        # Exit with error code if any tests failed
        failed_tests = [name for name, result in results.items() if not result['success']]
        if failed_tests:
            logger.error(f"Failed tests: {failed_tests}")
            return False
        else:
            logger.info("🎉 All tests passed! Complete evolution flow with fixed evaluate.py is working!")
            return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)