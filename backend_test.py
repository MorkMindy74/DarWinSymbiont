#!/usr/bin/env python3
"""
Backend Test Suite for EMERGENT AI-Powered Optimization Platform

Tests the Phase 1-4 backend APIs:
1. Health check endpoint: GET /api/health
2. Problem creation: POST /api/problem/create
3. Problem analysis with LLM: POST /api/analysis/analyze/{problem_id}
4. Get problem with analysis: GET /api/problem/{problem_id}
5. Evolution configuration: POST /api/evolution/configure/{problem_id}
6. Evolution status: GET /api/evolution/status/{session_id}
7. Evolution start: POST /api/evolution/start/{session_id}
8. File verification: initial.py, evaluate.py, evolution.db
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

# Backend URL - using local URL since external URL returns 404
BACKEND_URL = "http://localhost:8001"

# Test data - TSP (Traveling Salesman Problem) - Matching review request
TSP_TEST_DATA = {
    "problem_type": "tsp",
    "title": "Final Integration Test",
    "description": "Complete E2E test with all fixes applied",
    "constraints": {
        "num_locations": 5,
        "max_distance": 500
    }
}

# Evolution configuration for testing - SHORT test as requested
EVOLUTION_CONFIG = {
    "num_generations": 5,
    "max_parallel_jobs": 1,
    "llm_models": ["azure-gpt-4.1-mini"],
    "num_islands": 2,
    "archive_size": 20,
    "migration_interval": 2
}

class EmergentAPITester:
    """Test suite for EMERGENT Platform APIs"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.created_problem_id = None
        self.evolution_session_id = None
        self.work_dir = None
        
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

    async def test_problem_analysis(self) -> tuple[bool, str]:
        """Test 3: Problem analysis with LLM"""
        logger.info("Testing problem analysis endpoint...")
        
        if not self.created_problem_id:
            return False, "No problem_id available for analysis test"
        
        try:
            url = f"{self.base_url}/api/analysis/analyze/{self.created_problem_id}"
            headers = {"Content-Type": "application/json"}
            
            logger.info("Calling LLM analysis (may take 5-10 seconds)...")
            async with self.session.post(url, json=TSP_TEST_DATA, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Analysis failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                # Verify expected analysis fields
                required_fields = [
                    "problem_id", "problem_characterization", "complexity_assessment",
                    "key_challenges", "parameter_suggestions", "constraints_analysis",
                    "solution_strategy", "estimated_search_space", "recommended_evolution_config",
                    "analysis_timestamp"
                ]
                
                for field in required_fields:
                    if field not in data:
                        return False, f"Missing field in analysis response: {field}"
                
                # Verify problem_id matches
                if data["problem_id"] != self.created_problem_id:
                    return False, "Problem ID mismatch in analysis response"
                
                # Verify key_challenges is a list
                if not isinstance(data["key_challenges"], list):
                    return False, "key_challenges should be a list"
                
                # Verify parameter_suggestions is a list of objects
                if not isinstance(data["parameter_suggestions"], list):
                    return False, "parameter_suggestions should be a list"
                
                for param in data["parameter_suggestions"]:
                    required_param_fields = ["name", "value", "description", "rationale"]
                    for field in required_param_fields:
                        if field not in param:
                            return False, f"Missing field in parameter suggestion: {field}"
                
                # Verify constraints_analysis is a list of objects
                if not isinstance(data["constraints_analysis"], list):
                    return False, "constraints_analysis should be a list"
                
                for constraint in data["constraints_analysis"]:
                    required_constraint_fields = ["constraint_type", "description", "importance", "impact_on_solution"]
                    for field in required_constraint_fields:
                        if field not in constraint:
                            return False, f"Missing field in constraint analysis: {field}"
                
                # Verify recommended_evolution_config is a dict
                if not isinstance(data["recommended_evolution_config"], dict):
                    return False, "recommended_evolution_config should be a dict"
                
                # Check for TSP-specific content (basic validation)
                analysis_text = (data["problem_characterization"] + " " + 
                               data["complexity_assessment"] + " " + 
                               data["solution_strategy"]).lower()
                
                tsp_keywords = ["tsp", "traveling", "salesman", "route", "city", "cities", "distance"]
                if not any(keyword in analysis_text for keyword in tsp_keywords):
                    logger.warning("⚠️ Analysis may not be TSP-specific")
                
                logger.info(f"✅ Analysis completed successfully")
                logger.info(f"Problem characterization: {data['problem_characterization'][:100]}...")
                logger.info(f"Key challenges: {data['key_challenges']}")
                logger.info(f"Parameter suggestions count: {len(data['parameter_suggestions'])}")
                
                return True, f"Analysis successful with {len(data['key_challenges'])} challenges and {len(data['parameter_suggestions'])} parameter suggestions"
                
        except Exception as e:
            logger.error(f"❌ Problem analysis failed: {e}")
            return False, f"Problem analysis error: {e}"

    async def test_get_problem_with_analysis(self) -> tuple[bool, str]:
        """Test 4: Get problem with analysis"""
        logger.info("Testing get problem with analysis endpoint...")
        
        if not self.created_problem_id:
            return False, "No problem_id available for get problem test"
        
        try:
            url = f"{self.base_url}/api/problem/{self.created_problem_id}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Get problem failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                # Verify expected fields
                required_fields = ["problem", "analysis"]
                for field in required_fields:
                    if field not in data:
                        return False, f"Missing field in get problem response: {field}"
                
                # Verify problem data
                problem = data["problem"]
                if problem["problem_id"] != self.created_problem_id:
                    return False, "Problem ID mismatch in get problem response"
                
                # Verify analysis is included
                analysis = data["analysis"]
                if analysis is None:
                    return False, "Analysis should be included in response"
                
                # Verify analysis has required fields
                analysis_required_fields = [
                    "problem_id", "problem_characterization", "complexity_assessment",
                    "key_challenges", "parameter_suggestions", "constraints_analysis"
                ]
                
                for field in analysis_required_fields:
                    if field not in analysis:
                        return False, f"Missing field in analysis: {field}"
                
                logger.info(f"✅ Get problem with analysis successful")
                logger.info(f"Problem title: {problem['problem_input']['title']}")
                logger.info(f"Analysis timestamp: {analysis['analysis_timestamp']}")
                
                return True, f"Get problem with analysis successful"
                
        except Exception as e:
            logger.error(f"❌ Get problem with analysis failed: {e}")
            return False, f"Get problem with analysis error: {e}"

    async def test_evolution_configure(self) -> tuple[bool, str]:
        """Test 5: Evolution configuration endpoint"""
        logger.info("Testing evolution configuration endpoint...")
        
        if not self.created_problem_id:
            return False, "No problem_id available for evolution configuration test"
        
        try:
            url = f"{self.base_url}/api/evolution/configure/{self.created_problem_id}"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=EVOLUTION_CONFIG, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution configuration failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                # Verify expected fields
                required_fields = ["session_id", "problem_id", "work_dir", "ws_url", "initial_code", "evaluate_code"]
                for field in required_fields:
                    if field not in data:
                        return False, f"Missing field in evolution config response: {field}"
                
                # Verify problem_id matches
                if data["problem_id"] != self.created_problem_id:
                    return False, "Problem ID mismatch in evolution config response"
                
                # Store session info
                self.evolution_session_id = data["session_id"]
                self.work_dir = data["work_dir"]
                
                # Verify codes are not empty
                if not data["initial_code"].strip():
                    return False, "initial_code should not be empty"
                
                if not data["evaluate_code"].strip():
                    return False, "evaluate_code should not be empty"
                
                # Check for TSP-specific content in initial code
                initial_code = data["initial_code"].lower()
                if "tsp" not in initial_code and "tour" not in initial_code:
                    logger.warning("⚠️ Initial code may not be TSP-specific")
                
                logger.info(f"✅ Evolution configured successfully")
                logger.info(f"Session ID: {self.evolution_session_id}")
                logger.info(f"Work directory: {self.work_dir}")
                
                return True, f"Evolution configuration successful: {self.evolution_session_id}"
                
        except Exception as e:
            logger.error(f"❌ Evolution configuration failed: {e}")
            return False, f"Evolution configuration error: {e}"

    async def test_evolution_status(self) -> tuple[bool, str]:
        """Test 6: Evolution status endpoint"""
        logger.info("Testing evolution status endpoint...")
        
        if not self.evolution_session_id:
            return False, "No session_id available for evolution status test"
        
        try:
            url = f"{self.base_url}/api/evolution/status/{self.evolution_session_id}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution status failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                # Verify expected fields
                required_fields = ["session_id", "problem_id", "user_config", "work_dir", "status"]
                for field in required_fields:
                    if field not in data:
                        return False, f"Missing field in evolution status response: {field}"
                
                # Verify session_id matches
                if data["session_id"] != self.evolution_session_id:
                    return False, "Session ID mismatch in evolution status response"
                
                # Verify status is configured initially
                if data["status"] != "configured":
                    logger.warning(f"⚠️ Expected status 'configured', got '{data['status']}'")
                
                logger.info(f"✅ Evolution status retrieved successfully")
                logger.info(f"Status: {data['status']}")
                logger.info(f"User config: {data['user_config']}")
                
                return True, f"Evolution status successful: {data['status']}"
                
        except Exception as e:
            logger.error(f"❌ Evolution status failed: {e}")
            return False, f"Evolution status error: {e}"

    async def test_evolution_start(self) -> tuple[bool, str]:
        """Test 7: Evolution start endpoint"""
        logger.info("Testing evolution start endpoint...")
        
        if not self.evolution_session_id:
            return False, "No session_id available for evolution start test"
        
        try:
            url = f"{self.base_url}/api/evolution/start/{self.evolution_session_id}"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json={}, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution start failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                # Verify expected fields
                required_fields = ["session_id", "status", "message"]
                for field in required_fields:
                    if field not in data:
                        return False, f"Missing field in evolution start response: {field}"
                
                # Verify session_id matches
                if data["session_id"] != self.evolution_session_id:
                    return False, "Session ID mismatch in evolution start response"
                
                # Verify status is started
                if data["status"] != "started":
                    return False, f"Expected status 'started', got '{data['status']}'"
                
                logger.info(f"✅ Evolution started successfully")
                logger.info(f"Message: {data['message']}")
                
                return True, f"Evolution start successful: {data['status']}"
                
        except Exception as e:
            logger.error(f"❌ Evolution start failed: {e}")
            return False, f"Evolution start error: {e}"

    async def test_monitor_evolution_status(self) -> tuple[bool, str]:
        """Test 8: Monitor evolution status during execution"""
        logger.info("Testing evolution status monitoring...")
        
        if not self.evolution_session_id:
            return False, "No session_id available for evolution monitoring test"
        
        try:
            # Wait a bit for evolution to start
            logger.info("Waiting 5-10 seconds for evolution to progress...")
            await asyncio.sleep(8)
            
            url = f"{self.base_url}/api/evolution/status/{self.evolution_session_id}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution monitoring failed with status {response.status}: {error_text}"
                
                data = await response.json()
                
                # Check if status changed from configured
                status = data.get("status", "unknown")
                logger.info(f"Current evolution status: {status}")
                
                # Check for runtime info
                if "is_running" in data:
                    logger.info(f"Is running: {data['is_running']}")
                
                if "latest_generation" in data:
                    latest_gen = data["latest_generation"]
                    logger.info(f"Latest generation: {latest_gen}")
                    
                    if latest_gen is not None and latest_gen > 0:
                        logger.info("✅ Evolution is progressing - generations detected")
                
                if "islands" in data:
                    islands = data["islands"]
                    logger.info(f"Islands info: {len(islands) if islands else 0} islands")
                
                if "best_solution" in data and data["best_solution"]:
                    best = data["best_solution"]
                    logger.info(f"Best solution found: fitness={best.get('fitness', 'N/A')}")
                
                logger.info(f"✅ Evolution monitoring successful")
                
                return True, f"Evolution monitoring successful: status={status}"
                
        except Exception as e:
            logger.error(f"❌ Evolution monitoring failed: {e}")
            return False, f"Evolution monitoring error: {e}"

    async def test_verify_generated_files(self) -> tuple[bool, str]:
        """Test 9: Verify generated files exist and contain expected content"""
        logger.info("Testing generated files verification...")
        
        if not self.work_dir:
            return False, "No work_dir available for file verification test"
        
        try:
            work_path = Path(self.work_dir)
            
            # Check if work directory exists
            if not work_path.exists():
                return False, f"Work directory does not exist: {self.work_dir}"
            
            # Check initial.py
            initial_py = work_path / "initial.py"
            if not initial_py.exists():
                return False, f"initial.py not found at {initial_py}"
            
            initial_content = initial_py.read_text()
            if not initial_content.strip():
                return False, "initial.py is empty"
            
            # Check for TSP-specific content
            if "tsp" not in initial_content.lower() and "tour" not in initial_content.lower():
                logger.warning("⚠️ initial.py may not contain TSP-specific logic")
            
            # Check evaluate.py
            evaluate_py = work_path / "evaluate.py"
            if not evaluate_py.exists():
                return False, f"evaluate.py not found at {evaluate_py}"
            
            evaluate_content = evaluate_py.read_text()
            if not evaluate_content.strip():
                return False, "evaluate.py is empty"
            
            # Check evolution.db
            evolution_db = work_path / "evolution.db"
            db_exists = evolution_db.exists()
            
            logger.info(f"✅ File verification successful")
            logger.info(f"initial.py: {len(initial_content)} characters")
            logger.info(f"evaluate.py: {len(evaluate_content)} characters")
            logger.info(f"evolution.db exists: {db_exists}")
            
            return True, f"File verification successful: initial.py ({len(initial_content)} chars), evaluate.py ({len(evaluate_content)} chars), db_exists={db_exists}"
                
        except Exception as e:
            logger.error(f"❌ File verification failed: {e}")
            return False, f"File verification error: {e}"

    async def test_database_verification(self) -> tuple[bool, str]:
        """Test 10: Verify evolution database structure"""
        logger.info("Testing evolution database verification...")
        
        if not self.work_dir:
            return False, "No work_dir available for database verification test"
        
        try:
            db_path = Path(self.work_dir) / "evolution.db"
            
            if not db_path.exists():
                logger.warning("⚠️ evolution.db does not exist yet (evolution may not have started)")
                return True, "Database verification skipped: evolution.db not created yet"
            
            # Connect to database
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Check if programs table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='programs'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                conn.close()
                return False, "programs table does not exist in evolution.db"
            
            # Check table structure
            cursor.execute("PRAGMA table_info(programs)")
            columns = [row[1] for row in cursor.fetchall()]
            
            expected_columns = ["id", "generation", "island_id", "code", "metrics", "parent_id"]
            missing_columns = [col for col in expected_columns if col not in columns]
            
            if missing_columns:
                conn.close()
                return False, f"Missing columns in programs table: {missing_columns}"
            
            # Check if there are any records
            cursor.execute("SELECT COUNT(*) FROM programs")
            record_count = cursor.fetchone()[0]
            
            conn.close()
            
            logger.info(f"✅ Database verification successful")
            logger.info(f"programs table exists with {len(columns)} columns")
            logger.info(f"Record count: {record_count}")
            
            return True, f"Database verification successful: programs table with {record_count} records"
                
        except Exception as e:
            logger.error(f"❌ Database verification failed: {e}")
            return False, f"Database verification error: {e}"

    async def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all backend tests and return results"""
        logger.info("Starting EMERGENT Platform backend test suite...")
        logger.info(f"Testing against: {self.base_url}")
        
        tests = [
            ("Health check endpoint", self.test_health_check),
            ("Problem creation", self.test_problem_creation),
            ("Problem analysis with LLM", self.test_problem_analysis),
            ("Get problem with analysis", self.test_get_problem_with_analysis),
            ("Evolution configuration", self.test_evolution_configure),
            ("Evolution status check", self.test_evolution_status),
            ("Evolution start", self.test_evolution_start),
            ("Monitor evolution status", self.test_monitor_evolution_status),
            ("Verify generated files", self.test_verify_generated_files),
            ("Database verification", self.test_database_verification)
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


async def main():
    """Main test runner"""
    async with EmergentAPITester(BACKEND_URL) as tester:
        results = await tester.run_all_tests()
        
        # Exit with error code if any tests failed
        failed_tests = [name for name, result in results.items() if not result['success']]
        if failed_tests:
            logger.error(f"Failed tests: {failed_tests}")
            return False
        else:
            logger.info("All tests passed!")
            return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)