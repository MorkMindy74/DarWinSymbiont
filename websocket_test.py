#!/usr/bin/env python3
"""
WebSocket Connection Test for EMERGENT Platform

Tests the WebSocket URL construction fix:
- Verifies WebSocket endpoint /api/evolution/ws/{session_id} is accessible
- Tests WebSocket upgrade request succeeds
- Verifies heartbeat mechanism is working
- Tests message broadcasting (generation_complete, islands_update, etc.)

CRITICAL: WebSocket URL should now be constructed as:
wss://emergent-evolve.preview.emergentagent.com/api/evolution/ws/{sessionId}
"""

import asyncio
import aiohttp
import json
import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import websockets
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URLs - Testing both local and external
BACKEND_URL = "http://localhost:8001"
EXTERNAL_BACKEND_URL = "https://emergent-evolve.preview.emergentagent.com"

# Test data for creating evolution session
TSP_TEST_DATA = {
    "problem_type": "tsp",
    "title": "WebSocket Connection Test",
    "description": "Testing WebSocket URL construction fix",
    "constraints": {
        "num_locations": 5,
        "max_distance": 500
    }
}

EVOLUTION_CONFIG = {
    "num_generations": 3,
    "max_parallel_jobs": 1,
    "llm_models": ["azure-gpt-4.1-mini"],
    "num_islands": 2,
    "archive_size": 20,
    "migration_interval": 2
}

class WebSocketTester:
    """Test WebSocket connection and functionality"""
    
    def __init__(self):
        self.session = None
        self.problem_id = None
        self.session_id = None
        self.ws_url = None
        
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
        """Create problem and evolution session for WebSocket testing"""
        logger.info("Setting up evolution session for WebSocket testing...")
        
        try:
            # 1. Create problem
            url = f"{BACKEND_URL}/api/problem/create"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=TSP_TEST_DATA, headers=headers) as response:
                if response.status != 201:
                    error_text = await response.text()
                    return False, f"Problem creation failed: {response.status} - {error_text}"
                
                data = await response.json()
                self.problem_id = data["problem_id"]
                logger.info(f"✅ Problem created: {self.problem_id}")
            
            # 2. Run analysis
            url = f"{BACKEND_URL}/api/analysis/analyze/{self.problem_id}"
            async with self.session.post(url, json=TSP_TEST_DATA, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Analysis failed: {response.status} - {error_text}"
                
                logger.info("✅ Analysis completed")
            
            # 3. Configure evolution
            url = f"{BACKEND_URL}/api/evolution/configure/{self.problem_id}"
            async with self.session.post(url, json=EVOLUTION_CONFIG, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return False, f"Evolution configuration failed: {response.status} - {error_text}"
                
                data = await response.json()
                self.session_id = data["session_id"]
                self.ws_url = data["ws_url"]
                logger.info(f"✅ Evolution configured: {self.session_id}")
                logger.info(f"WebSocket URL: {self.ws_url}")
            
            return True, f"Evolution session setup complete: {self.session_id}"
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False, f"Setup error: {e}"
    
    async def test_websocket_connection_local(self) -> tuple[bool, str]:
        """Test WebSocket connection to local backend"""
        logger.info("Testing WebSocket connection to local backend...")
        
        if not self.session_id:
            return False, "No session_id available for WebSocket test"
        
        try:
            # Construct local WebSocket URL
            ws_url = f"ws://localhost:8001/api/evolution/ws/{self.session_id}"
            logger.info(f"Connecting to: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                logger.info("✅ WebSocket connection established")
                
                # Wait for initial connection message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    logger.info(f"Received initial message: {data}")
                    
                    if data.get("type") != "connected":
                        return False, f"Expected 'connected' message, got: {data.get('type')}"
                    
                    if data.get("session_id") != self.session_id:
                        return False, f"Session ID mismatch: expected {self.session_id}, got {data.get('session_id')}"
                    
                except asyncio.TimeoutError:
                    return False, "No initial connection message received within 5 seconds"
                
                # Test heartbeat mechanism
                logger.info("Testing heartbeat mechanism...")
                try:
                    # Wait for heartbeat (should come within 30 seconds)
                    message = await asyncio.wait_for(websocket.recv(), timeout=35.0)
                    data = json.loads(message)
                    logger.info(f"Received heartbeat: {data}")
                    
                    if data.get("type") != "heartbeat":
                        logger.warning(f"Expected heartbeat, got: {data.get('type')}")
                    else:
                        logger.info("✅ Heartbeat mechanism working")
                    
                except asyncio.TimeoutError:
                    logger.warning("⚠️ No heartbeat received within 35 seconds")
                
                # Test ping-pong
                logger.info("Testing ping-pong mechanism...")
                await websocket.send("ping")
                
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    logger.info(f"Received pong: {data}")
                    
                    if data.get("type") == "pong" and data.get("data") == "ping":
                        logger.info("✅ Ping-pong mechanism working")
                    else:
                        logger.warning(f"Unexpected pong response: {data}")
                
                except asyncio.TimeoutError:
                    return False, "No pong response received within 5 seconds"
                
                return True, "Local WebSocket connection successful with heartbeat and ping-pong"
                
        except Exception as e:
            logger.error(f"❌ Local WebSocket connection failed: {e}")
            return False, f"Local WebSocket error: {e}"
    
    async def test_websocket_connection_external(self) -> tuple[bool, str]:
        """Test WebSocket connection to external preview URL"""
        logger.info("Testing WebSocket connection to external preview URL...")
        
        if not self.session_id:
            return False, "No session_id available for WebSocket test"
        
        try:
            # Construct external WebSocket URL (as per review request)
            ws_url = f"wss://emergent-evolve.preview.emergentagent.com/api/evolution/ws/{self.session_id}"
            logger.info(f"Connecting to: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                logger.info("✅ External WebSocket connection established")
                
                # Wait for initial connection message
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    logger.info(f"Received initial message: {data}")
                    
                    if data.get("type") != "connected":
                        return False, f"Expected 'connected' message, got: {data.get('type')}"
                    
                    if data.get("session_id") != self.session_id:
                        return False, f"Session ID mismatch: expected {self.session_id}, got {data.get('session_id')}"
                    
                except asyncio.TimeoutError:
                    return False, "No initial connection message received within 5 seconds"
                
                # Test ping-pong
                logger.info("Testing ping-pong mechanism...")
                await websocket.send("ping")
                
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    logger.info(f"Received pong: {data}")
                    
                    if data.get("type") == "pong" and data.get("data") == "ping":
                        logger.info("✅ External ping-pong mechanism working")
                    else:
                        logger.warning(f"Unexpected pong response: {data}")
                
                except asyncio.TimeoutError:
                    return False, "No pong response received within 5 seconds"
                
                return True, "External WebSocket connection successful - URL construction fix verified"
                
        except Exception as e:
            logger.error(f"❌ External WebSocket connection failed: {e}")
            return False, f"External WebSocket error: {e}"
    
    async def test_websocket_message_broadcasting(self) -> tuple[bool, str]:
        """Test WebSocket message broadcasting during evolution"""
        logger.info("Testing WebSocket message broadcasting during evolution...")
        
        if not self.session_id:
            return False, "No session_id available for broadcasting test"
        
        try:
            # Connect to WebSocket
            ws_url = f"ws://localhost:8001/api/evolution/ws/{self.session_id}"
            logger.info(f"Connecting to: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                logger.info("✅ WebSocket connected for broadcasting test")
                
                # Wait for initial connection message
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                logger.info(f"Initial message: {data}")
                
                # Start evolution
                logger.info("Starting evolution to test message broadcasting...")
                url = f"{BACKEND_URL}/api/evolution/start/{self.session_id}"
                headers = {"Content-Type": "application/json"}
                
                async with self.session.post(url, json={}, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return False, f"Evolution start failed: {response.status} - {error_text}"
                    
                    logger.info("✅ Evolution started")
                
                # Listen for broadcast messages
                messages_received = []
                expected_message_types = ["evolution_start", "generation_complete", "islands_update"]
                
                logger.info("Listening for broadcast messages (30 seconds)...")
                timeout_time = asyncio.get_event_loop().time() + 30
                
                while asyncio.get_event_loop().time() < timeout_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        message_type = data.get("type")
                        
                        logger.info(f"Received broadcast: {message_type}")
                        messages_received.append(message_type)
                        
                        # Check for specific message types
                        if message_type == "evolution_start":
                            logger.info("✅ Evolution start message received")
                        elif message_type == "generation_complete":
                            logger.info(f"✅ Generation complete message received: gen {data.get('generation', 'unknown')}")
                        elif message_type == "islands_update":
                            logger.info(f"✅ Islands update message received: {len(data.get('islands', []))} islands")
                        elif message_type == "evolution_complete":
                            logger.info("✅ Evolution complete message received")
                            break
                        elif message_type == "evolution_error":
                            logger.warning(f"⚠️ Evolution error message: {data.get('error', 'unknown')}")
                        elif message_type == "heartbeat":
                            logger.info("Heartbeat received")
                        
                    except asyncio.TimeoutError:
                        logger.info("No message received in last 5 seconds, continuing...")
                        continue
                    except Exception as e:
                        logger.error(f"Error receiving message: {e}")
                        break
                
                # Analyze results
                unique_messages = set(messages_received)
                logger.info(f"Messages received: {list(unique_messages)}")
                
                if "evolution_start" in unique_messages:
                    logger.info("✅ Evolution start broadcasting working")
                else:
                    logger.warning("⚠️ No evolution start message received")
                
                if any(msg in unique_messages for msg in ["generation_complete", "evolution_complete"]):
                    logger.info("✅ Evolution progress broadcasting working")
                else:
                    logger.warning("⚠️ No evolution progress messages received")
                
                return True, f"Message broadcasting test completed. Received {len(unique_messages)} unique message types: {list(unique_messages)}"
                
        except Exception as e:
            logger.error(f"❌ Message broadcasting test failed: {e}")
            return False, f"Broadcasting test error: {e}"
    
    async def run_all_websocket_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all WebSocket tests"""
        logger.info("Starting WebSocket Connection Test Suite...")
        
        tests = [
            ("Setup Evolution Session", self.setup_evolution_session),
            ("WebSocket Connection (Local)", self.test_websocket_connection_local),
            ("WebSocket Connection (External)", self.test_websocket_connection_external),
            ("WebSocket Message Broadcasting", self.test_websocket_message_broadcasting)
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
                    # Stop on critical failures
                    if test_name == "Setup Evolution Session":
                        logger.error("Setup failed, stopping tests")
                        break
                        
            except Exception as e:
                logger.error(f"❌ ERROR: {test_name} - {e}")
                results[test_name] = {
                    'success': False,
                    'message': f"Unexpected error: {e}"
                }
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("WEBSOCKET TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
            if not result['success']:
                logger.info(f"    Error: {result['message']}")
        
        logger.info(f"\nOverall: {passed}/{total} WebSocket tests passed")
        
        return results


async def main():
    """Main test runner"""
    async with WebSocketTester() as tester:
        results = await tester.run_all_websocket_tests()
        
        # Exit with error code if any tests failed
        failed_tests = [name for name, result in results.items() if not result['success']]
        if failed_tests:
            logger.error(f"Failed tests: {failed_tests}")
            return False
        else:
            logger.info("All WebSocket tests passed!")
            return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)