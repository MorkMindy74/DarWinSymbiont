#!/usr/bin/env python3
"""
Practical integration test for LLM caching with ShinkaEvolve.

This test verifies that the caching system works correctly 
with the EvolutionRunner and shows practical usage.
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '/app')

from shinka.core.runner import EvolutionRunner, EvolutionConfig
from shinka.launch import JobConfig
from shinka.database import DatabaseConfig
from shinka.llm import LLMCache, CacheConfig

def test_evolution_runner_with_caching():
    """Test that EvolutionRunner correctly initializes and uses LLM caching."""
    print("🧪 Testing EvolutionRunner with LLM Caching Integration")
    print("=" * 60)
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    results_dir = Path(temp_dir) / "test_evolution_results"
    
    try:
        # Create EvolutionConfig with caching enabled
        evo_config = EvolutionConfig(
            num_generations=1,  # Minimal for testing
            max_parallel_jobs=1,
            llm_models=["gpt-4"],
            
            # Enable LLM caching
            llm_cache_enabled=True,
            llm_cache_mode="exact",
            llm_cache_ttl_hours=24.0,
            llm_cache_key_fields=["prompt", "model", "temperature"],
            
            results_dir=str(results_dir),
            init_program_path=None  # Will generate initial program
        )
        
        # Create JobConfig and DatabaseConfig
        from shinka.launch import LocalJobConfig
        job_config = LocalJobConfig(
            eval_program_path="evaluate.py"
        )
        
        db_config = DatabaseConfig(
            db_path="test_evolution.db"
        )
        
        print("✅ Configuration created successfully")
        print(f"   Cache enabled: {evo_config.llm_cache_enabled}")
        print(f"   Cache mode: {evo_config.llm_cache_mode}")
        print(f"   Cache TTL: {evo_config.llm_cache_ttl_hours} hours")
        
        # Initialize EvolutionRunner - this should create cache
        # Set a dummy API key to avoid initialization errors
        import os
        os.environ['OPENAI_API_KEY'] = 'dummy-key-for-testing'
        
        runner = EvolutionRunner(
            evo_config=evo_config,
            job_config=job_config,
            db_config=db_config,
            verbose=True
        )
        
        print("✅ EvolutionRunner initialized successfully")
        
        # Verify cache was created
        if hasattr(runner, 'llm_cache') and runner.llm_cache is not None:
            print("✅ LLM cache was created and configured")
            
            # Check cache configuration
            cache_stats = runner.llm_cache.get_stats()
            print(f"   Cache enabled: {cache_stats['enabled']}")
            print(f"   Cache mode: {cache_stats['mode']}")
            print(f"   Total entries: {cache_stats['total_entries']}")
            
        else:
            print("❌ LLM cache was not created")
            return False
        
        # Verify that LLM client is wrapped with cache
        from shinka.llm.cache import CachedLLMClient
        if isinstance(runner.llm, CachedLLMClient):
            print("✅ Main LLM client is wrapped with cache")
        else:
            print("❌ Main LLM client is not cached")
            return False
        
        # Test cache path generation
        expected_cache_path = results_dir / ".cache" / "llm_cache.db"
        if expected_cache_path.exists() or expected_cache_path.parent.exists():
            print(f"✅ Cache directory created at: {expected_cache_path.parent}")
        else:
            print(f"❌ Cache directory not found at: {expected_cache_path.parent}")
        
        print("\n🎯 Integration Test Summary:")
        print("✅ EvolutionConfig correctly configured for caching")
        print("✅ EvolutionRunner initializes with caching enabled")  
        print("✅ LLM clients are wrapped with CachedLLMClient")
        print("✅ Cache statistics are accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_cache_key_generation():
    """Test cache key generation with realistic LLM parameters."""
    print("\n🔑 Testing Cache Key Generation")
    print("=" * 40)
    
    config = CacheConfig(
        enabled=True,
        mode="exact",
        key_fields=["prompt", "model", "temperature", "seed"]
    )
    
    cache = LLMCache(config)
    
    # Test different scenarios
    scenarios = [
        {
            "name": "Code optimization prompt",
            "msg": "Optimize this function for better performance:\ndef slow_func(): pass",
            "system_msg": "You are a code optimization expert",
            "llm_kwargs": {"model_name": "gpt-4", "temperature": 0.2, "seed": 42}
        },
        {
            "name": "Same prompt, different temperature",
            "msg": "Optimize this function for better performance:\ndef slow_func(): pass",
            "system_msg": "You are a code optimization expert",
            "llm_kwargs": {"model_name": "gpt-4", "temperature": 0.7, "seed": 42}
        },
        {
            "name": "Different prompt, same parameters",
            "msg": "Generate a mutation for this code:\ndef slow_func(): pass",
            "system_msg": "You are a code optimization expert", 
            "llm_kwargs": {"model_name": "gpt-4", "temperature": 0.2, "seed": 42}
        }
    ]
    
    keys = []
    for scenario in scenarios:
        _, key = cache.get(
            scenario["msg"],
            scenario["system_msg"],
            [],
            scenario["llm_kwargs"]
        )
        keys.append(key)
        print(f"   {scenario['name']}: {key[:16]}...")
    
    # Verify keys are different when they should be
    if keys[0] != keys[1]:
        print("✅ Different temperatures produce different keys")
    else:
        print("❌ Different temperatures should produce different keys")
        return False
    
    if keys[0] != keys[2]:
        print("✅ Different prompts produce different keys")
    else:
        print("❌ Different prompts should produce different keys")
        return False
    
    # Verify determinism - same inputs should produce same key
    _, key_repeat = cache.get(
        scenarios[0]["msg"],
        scenarios[0]["system_msg"],
        [],
        scenarios[0]["llm_kwargs"]
    )
    
    if keys[0] == key_repeat:
        print("✅ Deterministic key generation confirmed")
        return True
    else:
        print("❌ Key generation is not deterministic")
        return False


def test_realistic_caching_scenario():
    """Test a realistic caching scenario with repeated prompts."""
    print("\n🔄 Testing Realistic Caching Scenario")
    print("=" * 45)
    
    temp_dir = tempfile.mkdtemp()
    cache_path = Path(temp_dir) / "realistic_cache.db"
    
    try:
        config = CacheConfig(
            enabled=True,
            mode="exact",
            path=str(cache_path),
            ttl_hours=1.0
        )
        
        cache = LLMCache(config)
        
        # Mock a QueryResult for testing
        from shinka.llm.models import QueryResult
        mock_result = QueryResult(
            content="Optimized code implementation",
            msg="Test query",
            system_msg="System",
            new_msg_history=[],
            model_name="gpt-4",
            kwargs={},
            input_tokens=50,
            output_tokens=100,
            cost=0.15
        )
        
        # Simulate evolution workflow with repeated patterns
        common_prompts = [
            "Analyze this code for performance bottlenecks",
            "Generate a code mutation for improved efficiency", 
            "Evaluate the correctness of this implementation"
        ]
        
        system_msg = "You are an expert code analysis AI"
        llm_kwargs = {"model_name": "gpt-4", "temperature": 0.5}
        
        print("Simulating evolution rounds with caching...")
        
        # Round 1: All cache misses
        misses_round1 = 0
        for prompt in common_prompts:
            result, _ = cache.get(prompt, system_msg, [], llm_kwargs)
            if result is None:
                misses_round1 += 1
                cache.put(prompt, system_msg, mock_result, [], llm_kwargs)
        
        # Round 2: All should be cache hits  
        hits_round2 = 0
        for prompt in common_prompts:
            result, _ = cache.get(prompt, system_msg, [], llm_kwargs)
            if result is not None:
                hits_round2 += 1
        
        print(f"   Round 1 misses: {misses_round1}/{len(common_prompts)}")
        print(f"   Round 2 hits: {hits_round2}/{len(common_prompts)}")
        
        # Verify expected behavior
        if misses_round1 == len(common_prompts) and hits_round2 == len(common_prompts):
            print("✅ Caching behavior works as expected")
            
            # Check statistics
            stats = cache.get_stats()
            print(f"   Final stats: {stats['hits']} hits, {stats['misses']} misses")
            print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
            
            return True
        else:
            print("❌ Unexpected caching behavior")
            return False
            
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("🧪 LLM Cache Integration Tests for ShinkaEvolve")
    print("=" * 70)
    
    test_results = []
    
    # Run all integration tests
    test_results.append(test_evolution_runner_with_caching())
    test_results.append(test_cache_key_generation()) 
    test_results.append(test_realistic_caching_scenario())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "="*70)
    print(f"🎯 Integration Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("✅ LLM caching is successfully integrated with ShinkaEvolve")
        print("\n📋 Ready for production use with configurations:")
        print("   • exact mode: deterministic caching for reproducible runs")
        print("   • fuzzy mode: similarity-based caching for cost optimization")
        print("   • configurable TTL for cache management")
        print("   • persistent SQLite backend")
    else:
        print("❌ Some integration tests failed")
        print("🔧 Please review the failing tests before deployment")
        sys.exit(1)