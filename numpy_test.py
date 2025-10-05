#!/usr/bin/env python3
"""
Numpy Integration Test - Final verification after numpy installation
Matches exact review request requirements:
1. Create TSP problem (5 locations)
2. Analyze
3. Configure evolution (3 generations, quick test)
4. Start evolution
5. Wait 25 seconds
6. Check status - verify latest_generation = 3, best_fitness != 0.0
"""

import asyncio
import aiohttp
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend URL
BACKEND_URL = "http://localhost:8001"

# Test data - TSP with 5 locations as requested
TSP_TEST_DATA = {
    "problem_type": "tsp",
    "title": "Numpy Integration Test - 5 Locations",
    "description": "Final test after numpy installation to verify programs work",
    "constraints": {
        "num_locations": 5,
        "max_distance": 1000
    }
}

# Evolution config - 3 generations as requested
EVOLUTION_CONFIG = {
    "num_generations": 3,
    "max_parallel_jobs": 1,
    "llm_models": ["azure-gpt-4.1-mini"],
    "num_islands": 2,
    "archive_size": 20,
    "migration_interval": 2
}

async def run_numpy_test():
    """Run the exact test scenario from review request"""
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        
        # Step 1: Create TSP problem (5 locations)
        logger.info("Step 1: Creating TSP problem with 5 locations...")
        url = f"{BACKEND_URL}/api/problem/create"
        async with session.post(url, json=TSP_TEST_DATA) as response:
            if response.status != 201:
                error_text = await response.text()
                logger.error(f"❌ Problem creation failed: {response.status} - {error_text}")
                return False
            
            data = await response.json()
            problem_id = data["problem_id"]
            logger.info(f"✅ Problem created: {problem_id}")
        
        # Step 2: Analyze
        logger.info("Step 2: Analyzing problem...")
        url = f"{BACKEND_URL}/api/analysis/analyze/{problem_id}"
        async with session.post(url, json=TSP_TEST_DATA) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"❌ Analysis failed: {response.status} - {error_text}")
                return False
            
            logger.info("✅ Analysis completed")
        
        # Step 3: Configure evolution (3 generations)
        logger.info("Step 3: Configuring evolution with 3 generations...")
        url = f"{BACKEND_URL}/api/evolution/configure/{problem_id}"
        async with session.post(url, json=EVOLUTION_CONFIG) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"❌ Evolution configuration failed: {response.status} - {error_text}")
                return False
            
            data = await response.json()
            session_id = data["session_id"]
            work_dir = data["work_dir"]
            logger.info(f"✅ Evolution configured: {session_id}")
            logger.info(f"Work directory: {work_dir}")
        
        # Step 4: Start evolution
        logger.info("Step 4: Starting evolution...")
        url = f"{BACKEND_URL}/api/evolution/start/{session_id}"
        async with session.post(url, json={}) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"❌ Evolution start failed: {response.status} - {error_text}")
                return False
            
            logger.info("✅ Evolution started")
        
        # Step 5: Wait 25 seconds
        logger.info("Step 5: Waiting 25 seconds for evolution to progress...")
        await asyncio.sleep(25)
        
        # Step 6: Check status - verify latest_generation = 3, best_fitness != 0.0
        logger.info("Step 6: Checking final status...")
        url = f"{BACKEND_URL}/api/evolution/status/{session_id}"
        async with session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"❌ Status check failed: {response.status} - {error_text}")
                return False
            
            data = await response.json()
            
            # Extract key metrics
            status = data.get("status", "unknown")
            latest_generation = data.get("latest_generation")
            best_fitness = data.get("best_fitness")
            is_running = data.get("is_running", False)
            islands = data.get("islands", [])
            
            logger.info(f"Evolution status: {status}")
            logger.info(f"Latest generation: {latest_generation}")
            logger.info(f"Best fitness: {best_fitness}")
            logger.info(f"Is running: {is_running}")
            logger.info(f"Islands count: {len(islands) if islands else 0}")
            
            # Verify requirements from review request
            success = True
            issues = []
            
            # Check latest_generation = 3
            if latest_generation != 3:
                issues.append(f"Expected latest_generation=3, got {latest_generation}")
                success = False
            else:
                logger.info("✅ Latest generation = 3 (as expected)")
            
            # Check best_fitness != 0.0 (programs should work now!)
            if best_fitness is None or best_fitness == 0.0:
                issues.append(f"Expected best_fitness != 0.0, got {best_fitness} (programs may still be failing)")
                success = False
            else:
                logger.info(f"✅ Best fitness = {best_fitness} (programs are working!)")
            
            # Check for at least one program with positive score
            positive_programs = 0
            if islands:
                for island in islands:
                    if isinstance(island, dict) and "programs" in island:
                        for program in island["programs"]:
                            if isinstance(program, dict) and "fitness" in program:
                                if program["fitness"] > 0:
                                    positive_programs += 1
            
            if positive_programs == 0:
                issues.append("No programs with positive fitness found")
                success = False
            else:
                logger.info(f"✅ Found {positive_programs} programs with positive fitness")
            
            # Final result
            if success:
                logger.info("🎉 NUMPY INTEGRATION TEST PASSED!")
                logger.info("✅ Evolution completed 3 generations")
                logger.info("✅ Programs are working (non-zero fitness)")
                logger.info("✅ At least one program has positive score")
                return True
            else:
                logger.error("❌ NUMPY INTEGRATION TEST FAILED!")
                for issue in issues:
                    logger.error(f"   - {issue}")
                return False

if __name__ == "__main__":
    success = asyncio.run(run_numpy_test())
    exit(0 if success else 1)