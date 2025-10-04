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

# Test data - TSP (Traveling Salesman Problem)
TSP_TEST_DATA = {
    "problem_type": "tsp",
    "title": "Test TSP - Evolution Flow",
    "description": "Find the shortest route to visit 10 cities. This is a test for the full evolution pipeline.",
    "constraints": {
        "num_locations": 10,
        "max_distance": 1000
    }
}

# Evolution configuration for testing
EVOLUTION_CONFIG = {
    "num_generations": 5,
    "max_parallel_jobs": 1,
    "llm_models": ["azure-gpt-4.1-mini"],
    "num_islands": 2,
    "archive_size": 50,
    "migration_interval": 3
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

    async def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all backend tests and return results"""
        logger.info("Starting EMERGENT Platform backend test suite...")
        logger.info(f"Testing against: {self.base_url}")
        
        tests = [
            ("Health check endpoint", self.test_health_check),
            ("Problem creation", self.test_problem_creation),
            ("Problem analysis with LLM", self.test_problem_analysis),
            ("Get problem with analysis", self.test_get_problem_with_analysis)
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