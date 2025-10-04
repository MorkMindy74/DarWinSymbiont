#!/usr/bin/env python3
"""
Advanced LLM Cache Usage Examples for ShinkaEvolve

This file demonstrates both the Darwin-style simple interface and 
the full ShinkaEvolve native interface with all advanced features.
"""

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, '/app')

from shinka.llm import (
    # Darwin-style simple interface
    CachedLLM,
    
    # Native ShinkaEvolve interface
    LLMCache, 
    CacheConfig,
    CachedLLMClient,
    LLMClient
)

def demo_darwin_style():
    """Demo using Darwin-style simple interface."""
    print("🔹 Darwin-Style Simple Interface")
    print("=" * 50)
    
    # Simple config dictionary
    config = {
        "enabled": True,
        "mode": "exact",
        "path": "./cache_darwin.db",
        "ttl_hours": 24.0
    }
    
    # Initialize and set model
    llm = CachedLLM(config)
    llm.set_model(lambda prompt: f"Mock response for: {prompt[:30]}...")
    
    # Use like a simple function
    result1 = llm("What is machine learning?")
    result2 = llm("What is machine learning?")  # Should hit cache
    
    print(f"First call:  {result1}")
    print(f"Second call: {result2}")
    print(f"Results identical: {result1 == result2}")
    
    # Show stats
    stats = llm.get_stats()
    print(f"Hit rate: {stats['local_hit_rate']:.1f}%")


def demo_native_interface():
    """Demo using full ShinkaEvolve native interface."""
    print("\n🔸 ShinkaEvolve Native Interface")
    print("=" * 50)
    
    # Advanced configuration
    config = CacheConfig(
        enabled=True,
        mode="fuzzy",  # Enable fuzzy matching
        path="./cache_native.db",
        ttl_hours=12.0,
        key_fields=["prompt", "model", "temperature"],
        fuzzy_threshold=0.75,  # Lower threshold for more matches
        minhash_perm=64  # Fewer permutations for speed
    )
    
    # Create cache
    cache = LLMCache(config)
    
    # Mock LLM client
    class MockLLMClient:
        def query(self, msg, system_msg, msg_history=None, llm_kwargs=None):
            from shinka.llm.models import QueryResult
            return QueryResult(
                content=f"Advanced response to: {msg[:40]}...",
                msg=msg,
                system_msg=system_msg,
                new_msg_history=[],
                model_name=llm_kwargs.get("model_name", "mock-model"),
                kwargs=llm_kwargs or {},
                input_tokens=len(msg.split()),
                output_tokens=10,
                cost=0.05
            )
    
    # Wrap with cache
    base_llm = MockLLMClient()
    cached_llm = CachedLLMClient(base_llm, cache)
    
    # Test exact and fuzzy matching
    queries = [
        "Optimize this code for better performance",
        "Optimize this code for improved performance",  # Similar, should fuzzy match
        "Generate a mutation of this function"
    ]
    
    llm_kwargs = {"model_name": "gpt-4", "temperature": 0.7}
    
    print("Testing fuzzy matching:")
    for i, query in enumerate(queries):
        result = cached_llm.query(
            msg=query,
            system_msg="You are a coding expert",
            llm_kwargs=llm_kwargs
        )
        print(f"Query {i+1}: {result.content}")
    
    # Show detailed stats
    stats = cache.get_stats()
    print(f"\nAdvanced Statistics:")
    print(f"  Fuzzy matches: {stats['fuzzy_matches']}")
    print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")
    print(f"  Active entries: {stats['active_entries']}")


def demo_integration_with_evolution():
    """Demo integration with EvolutionConfig."""
    print("\n🔷 Integration with EvolutionRunner")
    print("=" * 50)
    
    from shinka.core.runner import EvolutionConfig
    
    # Create config with caching
    evo_config = EvolutionConfig(
        # Basic evolution settings
        num_generations=5,
        max_parallel_jobs=1,
        llm_models=["gpt-4"],
        
        # Cache settings
        llm_cache_enabled=True,
        llm_cache_mode="exact",
        llm_cache_ttl_hours=48.0,
        llm_cache_key_fields=["prompt", "model", "seed"]
    )
    
    print("EvolutionConfig with caching:")
    print(f"  Cache enabled: {evo_config.llm_cache_enabled}")
    print(f"  Cache mode: {evo_config.llm_cache_mode}")
    print(f"  TTL: {evo_config.llm_cache_ttl_hours} hours")
    print(f"  Key fields: {evo_config.llm_cache_key_fields}")
    
    # In real usage, EvolutionRunner would use these settings automatically
    print("\nℹ️  In real usage:")
    print("   runner = EvolutionRunner(evo_config, job_config, db_config)")
    print("   runner.run()  # Caching happens transparently")


def demo_performance_comparison():
    """Demo performance comparison with/without caching."""
    print("\n⚡ Performance Comparison")
    print("=" * 50)
    
    def slow_model_call(prompt):
        """Simulate slow LLM call."""
        time.sleep(0.1)  # 100ms delay
        return f"Processed: {prompt[:20]}..."
    
    # Test without caching
    start_time = time.time()
    for _ in range(5):
        result = slow_model_call("Repeated query for testing")
    no_cache_time = time.time() - start_time
    
    # Test with caching
    config = {"enabled": True, "mode": "exact", "path": ":memory:"}
    cached_llm = CachedLLM(config)
    cached_llm.set_model(slow_model_call)
    
    start_time = time.time()
    for _ in range(5):
        result = cached_llm("Repeated query for testing")
    cache_time = time.time() - start_time
    
    print(f"Without cache: {no_cache_time:.2f}s")
    print(f"With cache:    {cache_time:.2f}s")
    print(f"Speedup:       {no_cache_time/cache_time:.1f}x")
    
    # Show cache efficiency
    stats = cached_llm.get_stats()
    print(f"Cache hit rate: {stats['local_hit_rate']:.1f}%")


def demo_cache_modes_comparison():
    """Compare exact vs fuzzy cache modes."""
    print("\n🎯 Cache Modes Comparison")
    print("=" * 50)
    
    queries = [
        "Optimize this function for better speed",
        "Optimize this function for improved speed",
        "Optimize this function for faster execution",
        "Make this function run faster"
    ]
    
    def mock_model(prompt):
        return f"Optimized version of your code"
    
    # Test exact mode
    exact_cache = CachedLLM({
        "enabled": True,
        "mode": "exact",
        "path": ":memory:"
    })
    exact_cache.set_model(mock_model)
    
    print("Exact mode results:")
    for q in queries:
        result = exact_cache(q)
        
    exact_stats = exact_cache.get_stats()
    print(f"  Hits: {exact_stats['local_hits']}")
    print(f"  Misses: {exact_stats['local_misses']}")
    
    # Test fuzzy mode  
    fuzzy_cache = CachedLLM({
        "enabled": True,
        "mode": "fuzzy",
        "path": ":memory:",
        "ttl_hours": 24
    })
    fuzzy_cache.set_model(mock_model)
    
    print("\nFuzzy mode results:")
    for q in queries:
        result = fuzzy_cache(q)
        
    fuzzy_stats = fuzzy_cache.get_stats()
    print(f"  Hits: {fuzzy_stats['local_hits']}")
    print(f"  Misses: {fuzzy_stats['local_misses']}")
    print(f"  Fuzzy matches: {fuzzy_stats.get('fuzzy_matches', 0)}")


if __name__ == "__main__":
    print("🧪 Advanced LLM Cache Usage Examples")
    print("=" * 70)
    
    # Run all demos
    demo_darwin_style()
    demo_native_interface()
    demo_integration_with_evolution()
    demo_performance_comparison()
    demo_cache_modes_comparison()
    
    print("\n" + "=" * 70)
    print("🎉 All demos completed!")
    print("\n📋 Summary:")
    print("✅ Darwin-style interface: Simple, drop-in replacement")
    print("✅ Native interface: Full control and advanced features")
    print("✅ EvolutionRunner integration: Transparent caching")
    print("✅ Performance benefits: Significant speedup for repeated queries")
    print("✅ Fuzzy matching: Better hit rates with similarity-based matching")
    
    # Cleanup temp files
    for db_file in ["cache_darwin.db", "cache_native.db"]:
        try:
            Path(db_file).unlink()
        except FileNotFoundError:
            pass