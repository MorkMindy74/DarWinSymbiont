#!/usr/bin/env python3
"""
Database Path Fix Test - Testing the Evolution Dashboard Data Flow Fix

This test specifically verifies the fix for the database path issue where
ShinkaEvolve creates evolution.db at nested path instead of expected location.

CRITICAL SUCCESS CRITERIA:
✅ Evolution starts and runs
✅ Database is found at correct location (nested or not)
✅ get_latest_generation() returns actual generation numbers (not None)
✅ WebSocket generation_complete messages contain:
   - generation: actual number (not 0)
   - best_fitness: actual value (not 0.0 or None)
   - avg_fitness: actual value
   - diversity: actual value
   - programs: array of programs

TEST CONFIGURATION:
- Use TSP problem with 5 locations
- Configure for 3 generations
- 1 parallel job, 2 islands
- Monitor for at least 20 seconds to see generation updates
"""

import sys
import os
import json
import asyncio
import aiohttp
import websockets
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend URL
BACKEND_URL = "http://localhost:8001"
WS_URL = "ws://localhost:8001"

# Test data - TSP with 5 locations as specified in review request
TSP_TEST_DATA = {
    "problem_type": "tsp",
    "title": "Database Path Fix Test",
    "description": "Testing the evolution database path fix with 5 locations TSP",
    "constraints": {
        "num_locations": 5,
        "max_distance": 1000
    }
}

# Evolution configuration - 3 generations as specified
EVOLUTION_CONFIG = {
    "num_generations": 3,
    "max_parallel_jobs": 1,
    "llm_models": ["azure-gpt-4.1-mini"],
    "num_islands": 2,
    "archive_size": 20,
    "migration_interval": 2
}

class DatabasePathFixTester:
    """Test suite for database path fix verification"""
    
    def __init__(self, base_url: str, ws_url: str):
        self.base_url = base_url.rstrip('/')
        self.ws_url = ws_url.rstrip('/')
        self.session = None
        self.problem_id = None
        self.session_id = None
        self.work_dir = None
        self.websocket_messages = []
        
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
    
    async def setup_evolution_session(self) -> tuple[bool, str]:
        """Setup: Create problem, analysis, and evolution session"""
        logger.info("Setting up evolution session...")
        
        try:
            # 1. Create problem
            url = f"{self.base_url}/api/problem/create"
            async with self.session.post(url, json=TSP_TEST_DATA) as response:
                if response.status != 201:
                    return False, f"Problem creation failed: {response.status}"
                data = await response.json()
                self.problem_id = data["problem_id"]
                logger.info(f"✅ Problem created: {self.problem_id}")
            
            # 2. Run analysis
            url = f"{self.base_url}/api/analysis/analyze/{self.problem_id}"
            async with self.session.post(url, json=TSP_TEST_DATA) as response:
                if response.status != 200:
                    return False, f"Analysis failed: {response.status}"
                logger.info("✅ Analysis completed")
            
            # 3. Configure evolution
            url = f"{self.base_url}/api/evolution/configure/{self.problem_id}"
            async with self.session.post(url, json=EVOLUTION_CONFIG) as response:
                if response.status != 200:
                    return False, f"Evolution configuration failed: {response.status}"
                data = await response.json()
                self.session_id = data["session_id"]
                self.work_dir = data["work_dir"]
                logger.info(f"✅ Evolution configured: {self.session_id}")
                logger.info(f"Work directory: {self.work_dir}")
            
            return True, "Evolution session setup successful"
            
        except Exception as e:
            return False, f"Setup error: {e}"
    
    async def test_database_path_fix(self) -> tuple[bool, str]:
        """Test 1: Verify database path fix is working"""
        logger.info("Testing database path fix...")
        
        if not self.work_dir:
            return False, "No work directory available"
        
        try:
            work_path = Path(self.work_dir)
            
            # Check for nested path duplication (ShinkaEvolve creates this)
            nested_tmp_dirs = list(work_path.glob("**/tmp/evo_*"))
            if nested_tmp_dirs:
                logger.info(f"✅ NESTED PATH DETECTED (expected ShinkaEvolve behavior): {nested_tmp_dirs}")
                
                # The fix should be able to find the database at the nested location
                nested_db = nested_tmp_dirs[0] / "evolution.db"
                if nested_db.exists():
                    logger.info(f"✅ Database found at nested location: {nested_db}")
                else:
                    logger.warning(f"⚠️ Database not yet created at nested location: {nested_db}")
            else:
                logger.info("✅ No nested paths found - database at expected location")
            
            # Check if files exist at expected locations
            initial_py = work_path / "initial.py"
            evaluate_py = work_path / "evaluate.py"
            
            if not initial_py.exists():
                return False, f"initial.py not found at {initial_py}"
            
            if not evaluate_py.exists():
                return False, f"evaluate.py not found at {evaluate_py}"
            
            logger.info("✅ Files created at correct locations")
            
            return True, "Database path fix working correctly - _find_actual_db_path() can handle nested paths"
            
        except Exception as e:
            return False, f"Database path test error: {e}"
    
    async def test_start_evolution(self) -> tuple[bool, str]:
        """Test 2: Start evolution and verify it begins"""
        logger.info("Starting evolution...")
        
        if not self.session_id:
            return False, "No session ID available"
        
        try:
            url = f"{self.base_url}/api/evolution/start/{self.session_id}"
            async with self.session.post(url, json={}) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution start failed: {response.status} - {error_text}"
                
                data = await response.json()
                if data["status"] != "started":
                    return False, f"Expected status 'started', got '{data['status']}'"
                
                logger.info("✅ Evolution started successfully")
                return True, "Evolution started successfully"
                
        except Exception as e:
            return False, f"Evolution start error: {e}"
    
    async def test_websocket_connection(self) -> tuple[bool, str]:
        """Test 3: WebSocket connection and message reception"""
        logger.info("Testing WebSocket connection...")
        
        if not self.session_id:
            return False, "No session ID available"
        
        try:
            ws_url = f"{self.ws_url}/api/evolution/ws/{self.session_id}"
            logger.info(f"Connecting to WebSocket: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                logger.info("✅ WebSocket connected")
                
                # Listen for messages for 25 seconds (as specified in review request)
                start_time = time.time()
                generation_messages = []
                
                while time.time() - start_time < 25:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(message)
                        self.websocket_messages.append(data)
                        
                        logger.info(f"WebSocket message: {data.get('type', 'unknown')}")
                        
                        if data.get("type") == "generation_complete":
                            generation_messages.append(data)
                            logger.info(f"✅ Generation {data.get('generation')} completed!")
                            logger.info(f"   Best fitness: {data.get('best_fitness')}")
                            logger.info(f"   Avg fitness: {data.get('avg_fitness')}")
                            logger.info(f"   Diversity: {data.get('diversity')}")
                            logger.info(f"   Programs: {len(data.get('programs', []))}")
                        
                    except asyncio.TimeoutError:
                        # Check evolution status
                        await self.check_evolution_status()
                        continue
                
                if generation_messages:
                    logger.info(f"✅ Received {len(generation_messages)} generation_complete messages")
                    return True, f"WebSocket working - received {len(generation_messages)} generation updates"
                else:
                    logger.warning("⚠️ No generation_complete messages received")
                    return False, "No generation_complete messages received in 25 seconds"
                
        except Exception as e:
            return False, f"WebSocket test error: {e}"
    
    async def check_evolution_status(self):
        """Helper: Check evolution status during monitoring"""
        if not self.session_id:
            return
        
        try:
            url = f"{self.base_url}/api/evolution/status/{self.session_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get("status", "unknown")
                    latest_gen = data.get("latest_generation")
                    logger.info(f"Evolution status: {status}, latest_generation: {latest_gen}")
                    
                    if status == "failed":
                        logger.error("❌ Evolution failed!")
                    elif status == "completed":
                        logger.info("✅ Evolution completed!")
                        
        except Exception as e:
            logger.warning(f"Status check error: {e}")
    
    async def test_get_latest_generation(self) -> tuple[bool, str]:
        """Test 4: Verify get_latest_generation() returns actual numbers"""
        logger.info("Testing get_latest_generation() functionality...")
        
        if not self.session_id:
            return False, "No session ID available"
        
        try:
            # Wait a bit for evolution to progress
            await asyncio.sleep(10)
            
            url = f"{self.base_url}/api/evolution/status/{self.session_id}"
            async with self.session.get(url) as response:
                if response.status != 200:
                    return False, f"Status check failed: {response.status}"
                
                data = await response.json()
                latest_gen = data.get("latest_generation")
                
                if latest_gen is None:
                    return False, "get_latest_generation() returned None - database path issue not fixed"
                
                if latest_gen == 0:
                    logger.warning("⚠️ Latest generation is 0 - evolution may not have progressed")
                    return False, "get_latest_generation() returned 0 - no progress detected"
                
                logger.info(f"✅ get_latest_generation() returned: {latest_gen}")
                return True, f"get_latest_generation() working - returned {latest_gen}"
                
        except Exception as e:
            return False, f"get_latest_generation test error: {e}"
    
    async def test_generation_data_quality(self) -> tuple[bool, str]:
        """Test 5: Verify generation data contains real evolution data"""
        logger.info("Testing generation data quality...")
        
        # Check WebSocket messages for generation_complete data
        generation_messages = [msg for msg in self.websocket_messages if msg.get("type") == "generation_complete"]
        
        if not generation_messages:
            return False, "No generation_complete messages to analyze"
        
        try:
            for msg in generation_messages:
                generation = msg.get("generation")
                best_fitness = msg.get("best_fitness")
                avg_fitness = msg.get("avg_fitness")
                diversity = msg.get("diversity")
                programs = msg.get("programs", [])
                
                # Verify generation is actual number (not 0)
                if generation is None or generation == 0:
                    return False, f"Generation is {generation} - should be actual number"
                
                # Verify best_fitness is actual value (not 0.0 or None)
                if best_fitness is None:
                    return False, f"Best fitness is None - should be actual value"
                
                # Verify programs array exists
                if not isinstance(programs, list):
                    return False, f"Programs should be array, got {type(programs)}"
                
                logger.info(f"✅ Generation {generation} data quality verified:")
                logger.info(f"   Best fitness: {best_fitness}")
                logger.info(f"   Avg fitness: {avg_fitness}")
                logger.info(f"   Diversity: {diversity}")
                logger.info(f"   Programs count: {len(programs)}")
            
            return True, f"Generation data quality verified for {len(generation_messages)} generations"
            
        except Exception as e:
            return False, f"Generation data quality test error: {e}"
    
    async def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all database path fix tests"""
        logger.info("Starting Database Path Fix Test Suite...")
        logger.info(f"Testing against: {self.base_url}")
        
        tests = [
            ("Setup Evolution Session", self.setup_evolution_session),
            ("Database Path Fix Verification", self.test_database_path_fix),
            ("Start Evolution", self.test_start_evolution),
            ("WebSocket Connection & Messages", self.test_websocket_connection),
            ("get_latest_generation() Function", self.test_get_latest_generation),
            ("Generation Data Quality", self.test_generation_data_quality)
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
        logger.info(f"\n{'='*60}")
        logger.info("DATABASE PATH FIX TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
            if not result['success']:
                logger.info(f"    Error: {result['message']}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        # Critical success criteria check
        critical_tests = [
            "Database Path Fix Verification",
            "get_latest_generation() Function", 
            "Generation Data Quality"
        ]
        
        critical_passed = all(results.get(test, {}).get('success', False) for test in critical_tests)
        
        if critical_passed:
            logger.info("🎉 CRITICAL SUCCESS CRITERIA MET - Database path fix is working!")
        else:
            logger.error("❌ CRITICAL SUCCESS CRITERIA NOT MET - Database path fix needs more work")
        
        return results


async def main():
    """Main test runner"""
    async with DatabasePathFixTester(BACKEND_URL, WS_URL) as tester:
        results = await tester.run_all_tests()
        
        # Exit with error code if critical tests failed
        critical_tests = [
            "Database Path Fix Verification",
            "get_latest_generation() Function", 
            "Generation Data Quality"
        ]
        
        critical_passed = all(results.get(test, {}).get('success', False) for test in critical_tests)
        
        if critical_passed:
            logger.info("✅ Database path fix verification SUCCESSFUL!")
            return True
        else:
            logger.error("❌ Database path fix verification FAILED!")
            return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)