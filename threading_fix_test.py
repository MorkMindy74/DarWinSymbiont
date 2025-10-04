#!/usr/bin/env python3
"""
Threading Fix Test for EMERGENT Evolution Flow
Tests the specific scenario from the review request to verify SQLite threading fix.
"""

import asyncio
import aiohttp
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend URL
BACKEND_URL = "http://localhost:8001"

# Test data from review request
TSP_TEST_DATA = {
    "problem_type": "tsp",
    "title": "Threading Fix Test",
    "description": "Testing SQLite threading fix for evolution",
    "constraints": {"num_locations": 8, "max_distance": 800}
}

# Evolution config from review request
EVOLUTION_CONFIG = {
    "num_generations": 3,
    "max_parallel_jobs": 1,
    "llm_models": ["azure-gpt-4.1-mini"],
    "num_islands": 2,
    "archive_size": 50,
    "migration_interval": 2
}

class ThreadingFixTester:
    """Test suite for threading fix verification"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.problem_id = None
        self.session_id = None
        
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

    async def test_1_create_tsp_problem(self) -> tuple[bool, str]:
        """Test 1: Create TSP Problem"""
        logger.info("Test 1: Creating TSP Problem...")
        
        try:
            url = f"{self.base_url}/api/problem/create"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=TSP_TEST_DATA, headers=headers) as response:
                if response.status != 201:
                    error_text = await response.text()
                    return False, f"Problem creation failed with status {response.status}: {error_text}"
                
                data = await response.json()
                self.problem_id = data["problem_id"]
                
                logger.info(f"✅ TSP Problem created: {self.problem_id}")
                return True, f"TSP Problem created successfully: {self.problem_id}"
                
        except Exception as e:
            logger.error(f"❌ Problem creation failed: {e}")
            return False, f"Problem creation error: {e}"

    async def test_2_analyze_problem(self) -> tuple[bool, str]:
        """Test 2: Analyze Problem"""
        logger.info("Test 2: Analyzing Problem...")
        
        if not self.problem_id:
            return False, "No problem_id available for analysis"
        
        try:
            url = f"{self.base_url}/api/analysis/analyze/{self.problem_id}"
            headers = {"Content-Type": "application/json"}
            
            logger.info("Calling LLM analysis...")
            async with self.session.post(url, json=TSP_TEST_DATA, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Analysis failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                logger.info(f"✅ Analysis completed successfully")
                return True, f"Analysis completed successfully"
                
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return False, f"Analysis error: {e}"

    async def test_3_configure_evolution(self) -> tuple[bool, str]:
        """Test 3: Configure Evolution"""
        logger.info("Test 3: Configuring Evolution...")
        
        if not self.problem_id:
            return False, "No problem_id available for evolution configuration"
        
        try:
            url = f"{self.base_url}/api/evolution/configure/{self.problem_id}"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=EVOLUTION_CONFIG, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution configuration failed with status {response.status}: {error_text}"
                
                data = await response.json()
                self.session_id = data["session_id"]
                
                logger.info(f"✅ Evolution configured: {self.session_id}")
                return True, f"Evolution configured successfully: {self.session_id}"
                
        except Exception as e:
            logger.error(f"❌ Evolution configuration failed: {e}")
            return False, f"Evolution configuration error: {e}"

    async def test_4_start_evolution(self) -> tuple[bool, str]:
        """Test 4: Start Evolution"""
        logger.info("Test 4: Starting Evolution...")
        
        if not self.session_id:
            return False, "No session_id available for evolution start"
        
        try:
            url = f"{self.base_url}/api/evolution/start/{self.session_id}"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json={}, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution start failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                logger.info(f"✅ Evolution started: {data['status']}")
                return True, f"Evolution started successfully: {data['status']}"
                
        except Exception as e:
            logger.error(f"❌ Evolution start failed: {e}")
            return False, f"Evolution start error: {e}"

    async def test_5_monitor_evolution(self) -> tuple[bool, str]:
        """Test 5: Monitor Evolution for 15-20 seconds"""
        logger.info("Test 5: Monitoring Evolution for 15-20 seconds...")
        
        if not self.session_id:
            return False, "No session_id available for evolution monitoring"
        
        try:
            url = f"{self.base_url}/api/evolution/status/{self.session_id}"
            
            # Monitor for 20 seconds with status checks every 3 seconds
            monitoring_duration = 20
            check_interval = 3
            checks = monitoring_duration // check_interval
            
            threading_error_detected = False
            evolution_progressed = False
            latest_generation = 0
            
            for i in range(checks):
                logger.info(f"Status check {i+1}/{checks} (after {(i+1)*check_interval} seconds)...")
                
                async with self.session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return False, f"Status check failed: {response.status}: {error_text}"
                    
                    data = await response.json()
                    
                    status = data.get("status", "unknown")
                    is_running = data.get("is_running", False)
                    current_gen = data.get("latest_generation")
                    islands = data.get("islands", [])
                    
                    logger.info(f"  Status: {status}")
                    logger.info(f"  Is running: {is_running}")
                    logger.info(f"  Latest generation: {current_gen}")
                    logger.info(f"  Islands: {len(islands) if islands else 0}")
                    
                    # Check if evolution progressed
                    if current_gen is not None and current_gen > latest_generation:
                        latest_generation = current_gen
                        evolution_progressed = True
                        logger.info(f"  🎉 Evolution progressed to generation {current_gen}!")
                    
                    # Check for best solution
                    if "best_solution" in data and data["best_solution"]:
                        best = data["best_solution"]
                        logger.info(f"  Best solution: fitness={best.get('fitness', 'N/A')}")
                
                # Wait before next check (except for last iteration)
                if i < checks - 1:
                    await asyncio.sleep(check_interval)
            
            # Final assessment
            if evolution_progressed and latest_generation > 0:
                logger.info(f"✅ SUCCESS: Evolution progressed to generation {latest_generation}")
                return True, f"Evolution progressed successfully to generation {latest_generation}"
            else:
                logger.error(f"❌ FAILURE: Evolution did not progress (latest_generation: {latest_generation})")
                return False, f"Evolution did not progress beyond generation {latest_generation}"
                
        except Exception as e:
            logger.error(f"❌ Evolution monitoring failed: {e}")
            return False, f"Evolution monitoring error: {e}"

    async def run_threading_fix_test(self) -> dict:
        """Run the complete threading fix test"""
        logger.info("="*60)
        logger.info("THREADING FIX TEST - EMERGENT Evolution Flow")
        logger.info("="*60)
        logger.info(f"Testing against: {self.base_url}")
        logger.info(f"Test data: {TSP_TEST_DATA}")
        logger.info(f"Evolution config: {EVOLUTION_CONFIG}")
        logger.info("="*60)
        
        tests = [
            ("Create TSP Problem", self.test_1_create_tsp_problem),
            ("Analyze Problem", self.test_2_analyze_problem),
            ("Configure Evolution", self.test_3_configure_evolution),
            ("Start Evolution", self.test_4_start_evolution),
            ("Monitor Evolution (15-20s)", self.test_5_monitor_evolution)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*40}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*40}")
            
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
                    # Don't continue if critical steps fail
                    if test_name in ["Create TSP Problem", "Configure Evolution", "Start Evolution"]:
                        logger.error("Critical test failed, stopping execution")
                        break
                    
            except Exception as e:
                logger.error(f"❌ ERROR: {test_name} - {e}")
                results[test_name] = {
                    'success': False,
                    'message': f"Unexpected error: {e}"
                }
                break
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("THREADING FIX TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
            if not result['success']:
                logger.info(f"    Error: {result['message']}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        # Specific threading fix assessment
        if "Monitor Evolution (15-20s)" in results:
            monitor_result = results["Monitor Evolution (15-20s)"]
            if monitor_result['success']:
                logger.info("🎉 THREADING FIX VERIFIED: Evolution progressed successfully!")
            else:
                logger.error("❌ THREADING ISSUE PERSISTS: Evolution did not progress")
        
        return results


async def main():
    """Main test runner"""
    async with ThreadingFixTester(BACKEND_URL) as tester:
        results = await tester.run_threading_fix_test()
        
        # Check if threading fix worked
        monitor_test = results.get("Monitor Evolution (15-20s)")
        if monitor_test and monitor_test['success']:
            logger.info("✅ Threading fix test PASSED!")
            return True
        else:
            logger.error("❌ Threading fix test FAILED!")
            return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)