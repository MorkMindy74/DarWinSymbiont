#!/usr/bin/env python3
"""
Regression test script for archived agents
"""

import sys
import json
sys.path.insert(0, '/app')

from shinka.archive import list_archived_agents, reproduce_agent, create_agent_archive

def main():
    print("🔍 Finding archived agents for regression test...")
    
    # Find latest stable agent
    agents = list_archived_agents()
    if not agents:
        print("No agents found - creating reference agent...")
        
        # First run a quick benchmark to get realistic data
        import subprocess
        
        try:
            result = subprocess.run([
                "python", "-m", "bench.context_bandit_bench",
                "--benchmark", "toy",
                "--algo", "context",
                "--seed", "42", 
                "--budget_steps", "200",
                "--model", "mock"
            ], cwd="/app", capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"Warning: Benchmark failed: {result.stderr}")
        except Exception as e:
            print(f"Warning: Could not run benchmark: {e}")
        
        # Create reference agent
        archive = create_agent_archive()
        config = {
            'algorithm': 'context', 
            'seed': 42, 
            'benchmark': 'toy', 
            'stable': True,
            'budget_steps': 200
        }
        agent_id = archive.save_agent(config)
        print(f"Created reference agent: {agent_id}")
        agents = [{'id': agent_id}]
    
    # Use most recent agent
    target_agent = agents[0]
    agent_id = target_agent['id']
    print(f"Testing agent: {agent_id}")
    
    # Run reproduction test
    print("🧪 Running reproduction test...")
    result = reproduce_agent(agent_id, tolerance_pct=1.0)
    
    if result['success']:
        print("✅ REGRESSION TEST PASSED")
        print("All metrics reproduced within ±1% tolerance")
        
        # Print summary
        if 'verification_results' in result:
            for benchmark, verification in result['verification_results'].items():
                status = "✅" if verification['passed'] else "❌"
                print(f"  {status} {benchmark}")
        
        return 0
    else:
        print(f"❌ REGRESSION TEST FAILED: {result.get('error', 'Unknown error')}")
        
        # Print details
        if 'verification_results' in result:
            print("\nDetails:")
            print(json.dumps(result['verification_results'], indent=2, default=str))
        
        return 1

if __name__ == "__main__":
    sys.exit(main())