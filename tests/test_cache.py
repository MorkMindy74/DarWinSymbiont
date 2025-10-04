"""
Unit tests for LLM caching system.

Tests cover:
1. Cache hits and misses
2. TTL expiration
3. Fuzzy matching with MinHash
4. Persistence between runs
5. Edge cases and error handling
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, '/app')

from shinka.llm.cache import LLMCache, CachedLLMClient, CacheConfig, MinHasher
from shinka.llm.models import QueryResult


class TestMinHasher(unittest.TestCase):
    """Test MinHash implementation for fuzzy matching."""
    
    def setUp(self):
        self.minhasher = MinHasher(num_perm=64)
    
    def test_identical_strings_high_similarity(self):
        """Identical strings should have similarity close to 1.0."""
        text = "This is a test string for MinHash similarity"
        sig1 = self.minhasher.compute_signature(text)
        sig2 = self.minhasher.compute_signature(text)
        
        similarity = self.minhasher.jaccard_similarity(sig1, sig2)
        self.assertGreater(similarity, 0.99)
    
    def test_similar_strings_moderate_similarity(self):
        """Similar strings should have moderate similarity."""
        text1 = "This is a test string for MinHash similarity testing"
        text2 = "This is a test string for MinHash similarity analysis"
        
        sig1 = self.minhasher.compute_signature(text1)
        sig2 = self.minhasher.compute_signature(text2)
        
        similarity = self.minhasher.jaccard_similarity(sig1, sig2)
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 0.99)
    
    def test_different_strings_low_similarity(self):
        """Completely different strings should have low similarity."""
        text1 = "This is about machine learning and artificial intelligence"
        text2 = "The weather today is sunny with a chance of rain"
        
        sig1 = self.minhasher.compute_signature(text1)
        sig2 = self.minhasher.compute_signature(text2)
        
        similarity = self.minhasher.jaccard_similarity(sig1, sig2)
        self.assertLess(similarity, 0.3)
    
    def test_empty_string_handling(self):
        """Empty strings should be handled gracefully."""
        sig = self.minhasher.compute_signature("")
        self.assertEqual(len(sig), 64)
        self.assertTrue(all(isinstance(x, int) for x in sig))


class TestLLMCache(unittest.TestCase):
    """Test LLM cache functionality."""
    
    def setUp(self):
        """Set up test environment with temporary cache."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.temp_dir, "test_cache.db")
        
        self.config = CacheConfig(
            enabled=True,
            mode="exact",
            path=self.cache_path,
            ttl_hours=1.0  # 1 hour for testing
        )
        
        self.cache = LLMCache(self.config)
        
        # Create real QueryResult for testing
        self.mock_result = QueryResult(
            content="Test response",
            msg="Test query",
            system_msg="System message", 
            new_msg_history=[],
            model_name="gpt-4",
            kwargs={"temperature": 0.7},
            input_tokens=10,
            output_tokens=15,
            cost=0.05
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_miss_and_hit(self):
        """Test basic cache miss followed by cache hit."""
        msg = "What is machine learning?"
        system_msg = "You are a helpful assistant."
        llm_kwargs = {"model_name": "gpt-4", "temperature": 0.7}
        
        # First query - should be cache miss
        result, cache_key = self.cache.get(msg, system_msg, [], llm_kwargs)
        self.assertIsNone(result)
        self.assertTrue(cache_key)
        
        # Store result in cache
        stored_key = self.cache.put(msg, system_msg, self.mock_result, [], llm_kwargs)
        self.assertEqual(cache_key, stored_key)
        
        # Second query - should be cache hit
        result, cache_key = self.cache.get(msg, system_msg, [], llm_kwargs)
        self.assertIsNotNone(result)
        self.assertEqual(result.content, "Test response")
        
        # Check statistics
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
    
    def test_deterministic_keys(self):
        """Test that identical queries produce identical keys."""
        msg = "Hello world"
        system_msg = "Be helpful"
        llm_kwargs = {"model_name": "gpt-4", "seed": 42}
        
        # Query multiple times with same parameters
        _, key1 = self.cache.get(msg, system_msg, [], llm_kwargs)
        _, key2 = self.cache.get(msg, system_msg, [], llm_kwargs)
        _, key3 = self.cache.get(msg, system_msg, [], llm_kwargs)
        
        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)
    
    def test_different_parameters_different_keys(self):
        """Test that different parameters produce different keys."""
        msg = "Hello world"
        system_msg = "Be helpful"
        
        kwargs1 = {"model_name": "gpt-4", "temperature": 0.5}
        kwargs2 = {"model_name": "gpt-4", "temperature": 0.7}
        kwargs3 = {"model_name": "claude-3", "temperature": 0.5}
        
        _, key1 = self.cache.get(msg, system_msg, [], kwargs1)
        _, key2 = self.cache.get(msg, system_msg, [], kwargs2)
        _, key3 = self.cache.get(msg, system_msg, [], kwargs3)
        
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key2, key3)
    
    def test_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        # Use very short TTL for testing
        short_config = CacheConfig(
            enabled=True,
            mode="exact",
            path=self.cache_path,
            ttl_hours=0.001  # ~3.6 seconds
        )
        cache = LLMCache(short_config)
        
        msg = "Test TTL"
        system_msg = "System"
        llm_kwargs = {"model_name": "gpt-4"}
        
        # Store entry
        cache.put(msg, system_msg, self.mock_result, [], llm_kwargs)
        
        # Immediately retrieve - should hit
        result, _ = cache.get(msg, system_msg, [], llm_kwargs)
        self.assertIsNotNone(result)
        
        # Wait for expiration and retrieve - should miss
        time.sleep(4)
        result, _ = cache.get(msg, system_msg, [], llm_kwargs)
        self.assertIsNone(result)
        
        # Check expired count in stats
        stats = cache.get_stats()
        self.assertEqual(stats["expired"], 1)
    
    def test_persistence_between_runs(self):
        """Test that cache persists between different cache instances."""
        msg = "Persistent test"
        system_msg = "System"
        llm_kwargs = {"model_name": "gpt-4"}
        
        # Store with first cache instance
        self.cache.put(msg, system_msg, self.mock_result, [], llm_kwargs)
        
        # Create new cache instance with same path
        new_cache = LLMCache(self.config)
        
        # Should be able to retrieve from new instance
        result, _ = new_cache.get(msg, system_msg, [], llm_kwargs)
        self.assertIsNotNone(result)
        self.assertEqual(result.content, "Test response")
    
    def test_fuzzy_matching(self):
        """Test fuzzy matching with MinHash."""
        fuzzy_config = CacheConfig(
            enabled=True,
            mode="fuzzy",
            path=self.cache_path,
            ttl_hours=1.0,
            fuzzy_threshold=0.7
        )
        fuzzy_cache = LLMCache(fuzzy_config)
        
        # Store original query
        original_msg = "What is machine learning and how does it work?"
        system_msg = "You are an AI expert"
        llm_kwargs = {"model_name": "gpt-4"}
        
        fuzzy_cache.put(original_msg, system_msg, self.mock_result, [], llm_kwargs)
        
        # Try similar query - should match
        similar_msg = "What is machine learning and how does it function?"
        result, _ = fuzzy_cache.get(similar_msg, system_msg, [], llm_kwargs)
        self.assertIsNotNone(result)
        
        # Check fuzzy match stats
        stats = fuzzy_cache.get_stats()
        self.assertEqual(stats["fuzzy_matches"], 1)
        
        # Try very different query - should not match
        different_msg = "What is the weather like today?"
        result, _ = fuzzy_cache.get(different_msg, system_msg, [], llm_kwargs)
        self.assertIsNone(result)
    
    def test_cleanup_expired_entries(self):
        """Test cleanup of expired entries."""
        short_config = CacheConfig(
            enabled=True,
            mode="exact", 
            path=self.cache_path,
            ttl_hours=0.001
        )
        cache = LLMCache(short_config)
        
        # Store multiple entries
        for i in range(5):
            msg = f"Test message {i}"
            cache.put(msg, "System", self.mock_result, [], {"model_name": "gpt-4"})
        
        # Wait for expiration
        time.sleep(4)
        
        # Cleanup expired entries
        removed_count = cache.cleanup_expired()
        self.assertEqual(removed_count, 5)
        
        # Verify stats
        stats = cache.get_stats()
        self.assertEqual(stats["total_entries"], 0)
    
    def test_disabled_cache(self):
        """Test that disabled cache doesn't store or retrieve."""
        disabled_config = CacheConfig(enabled=False)
        disabled_cache = LLMCache(disabled_config)
        
        msg = "Test disabled"
        system_msg = "System"
        llm_kwargs = {"model_name": "gpt-4"}
        
        # Try to store
        key = disabled_cache.put(msg, system_msg, self.mock_result, [], llm_kwargs)
        self.assertEqual(key, "")
        
        # Try to retrieve
        result, key = disabled_cache.get(msg, system_msg, [], llm_kwargs)
        self.assertIsNone(result)
        self.assertEqual(key, "")
    
    def test_tool_state_in_cache_key(self):
        """Test that tool_state affects cache key generation."""
        msg = "Test tool state"
        system_msg = "System"
        llm_kwargs = {"model_name": "gpt-4"}
        
        tool_state1 = {"active_tools": ["calculator"]}
        tool_state2 = {"active_tools": ["web_search"]}
        
        _, key1 = self.cache.get(msg, system_msg, [], llm_kwargs, tool_state1)
        _, key2 = self.cache.get(msg, system_msg, [], llm_kwargs, tool_state2)
        
        self.assertNotEqual(key1, key2)
    
    def test_message_history_affects_key(self):
        """Test that message history affects cache key."""
        msg = "Continue the conversation"
        system_msg = "Be helpful"
        llm_kwargs = {"model_name": "gpt-4"}
        
        history1 = [{"role": "user", "content": "Hello"}]
        history2 = [{"role": "user", "content": "Hi there"}]
        
        _, key1 = self.cache.get(msg, system_msg, history1, llm_kwargs)
        _, key2 = self.cache.get(msg, system_msg, history2, llm_kwargs)
        
        self.assertNotEqual(key1, key2)


class TestCachedLLMClient(unittest.TestCase):
    """Test the CachedLLMClient wrapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.temp_dir, "test_cache.db")
        
        self.config = CacheConfig(
            enabled=True,
            mode="exact",
            path=self.cache_path,
            ttl_hours=1.0
        )
        
        self.cache = LLMCache(self.config)
        
        # Mock underlying LLM client
        self.mock_llm_client = Mock()
        self.mock_result = Mock(spec=QueryResult)
        self.mock_result.content = "LLM Response"
        self.mock_result.cost = 0.1
        self.mock_llm_client.query.return_value = self.mock_result
        
        self.cached_client = CachedLLMClient(self.mock_llm_client, self.cache)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_miss_calls_llm(self):
        """Test that cache miss results in LLM call."""
        msg = "Test query"
        system_msg = "System"
        llm_kwargs = {"model_name": "gpt-4"}
        
        result = self.cached_client.query(msg, system_msg, llm_kwargs=llm_kwargs)
        
        # Should call underlying LLM
        self.mock_llm_client.query.assert_called_once()
        self.assertEqual(result, self.mock_result)
    
    def test_cache_hit_skips_llm(self):
        """Test that cache hit skips LLM call."""
        msg = "Cached query"
        system_msg = "System"
        llm_kwargs = {"model_name": "gpt-4"}
        
        # First call - cache miss, should call LLM
        result1 = self.cached_client.query(msg, system_msg, llm_kwargs=llm_kwargs)
        self.assertEqual(self.mock_llm_client.query.call_count, 1)
        
        # Second call - cache hit, should not call LLM
        result2 = self.cached_client.query(msg, system_msg, llm_kwargs=llm_kwargs) 
        self.assertEqual(self.mock_llm_client.query.call_count, 1)  # Still 1
        
        # Results should be equal
        self.assertEqual(result1.content, result2.content)
    
    def test_no_cache_always_calls_llm(self):
        """Test that client without cache always calls LLM."""
        no_cache_client = CachedLLMClient(self.mock_llm_client, cache=None)
        
        msg = "No cache query"
        system_msg = "System"
        
        # Multiple calls should all go to LLM
        no_cache_client.query(msg, system_msg)
        no_cache_client.query(msg, system_msg)
        
        self.assertEqual(self.mock_llm_client.query.call_count, 2)
    
    def test_attribute_forwarding(self):
        """Test that unknown attributes are forwarded to underlying client."""
        # Access attribute that should be forwarded
        self.mock_llm_client.some_method = Mock(return_value="forwarded")
        
        result = self.cached_client.some_method()
        self.assertEqual(result, "forwarded")
        self.mock_llm_client.some_method.assert_called_once()
    
    def test_batch_operations_not_cached(self):
        """Test that batch operations bypass cache."""
        self.mock_llm_client.batch_query.return_value = ["result1", "result2"]
        
        result = self.cached_client.batch_query(2, "msg", "system")
        
        self.mock_llm_client.batch_query.assert_called_once_with(2, "msg", "system")
        self.assertEqual(result, ["result1", "result2"])


class TestCacheIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.temp_dir, "integration_cache.db")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_evolution_workflow_simulation(self):
        """Simulate realistic evolution workflow with caching."""
        config = CacheConfig(
            enabled=True,
            mode="exact",
            path=self.cache_path,
            ttl_hours=24.0
        )
        cache = LLMCache(config)
        
        # Simulate common evolution queries that might be repeated
        common_queries = [
            ("Analyze this code for optimization", "You are a code optimization expert"),
            ("Generate a mutation for this function", "You are a code mutator"), 
            ("Evaluate the correctness of this solution", "You are a code evaluator")
        ]
        
        mock_results = []
        for i, (msg, sys_msg) in enumerate(common_queries):
            mock_result = Mock(spec=QueryResult)
            mock_result.content = f"Response {i}"
            mock_result.cost = 0.05
            mock_results.append(mock_result)
        
        # First round - all cache misses
        for i, (msg, sys_msg) in enumerate(common_queries):
            result, _ = cache.get(msg, sys_msg, [], {"model_name": "gpt-4"})
            self.assertIsNone(result)
            
            # Store result
            cache.put(msg, sys_msg, mock_results[i], [], {"model_name": "gpt-4"})
        
        # Second round - all cache hits
        for i, (msg, sys_msg) in enumerate(common_queries):
            result, _ = cache.get(msg, sys_msg, [], {"model_name": "gpt-4"})
            self.assertIsNotNone(result)
            self.assertEqual(result.content, f"Response {i}")
        
        # Verify hit rate
        stats = cache.get_stats()
        self.assertEqual(stats["hits"], 3)
        self.assertEqual(stats["misses"], 3)
        self.assertAlmostEqual(stats["hit_rate_percent"], 50.0, places=1)
    
    def test_fuzzy_matching_evolution_patterns(self):
        """Test fuzzy matching with typical evolution code patterns."""
        config = CacheConfig(
            enabled=True,
            mode="fuzzy",
            path=self.cache_path,
            ttl_hours=1.0,
            fuzzy_threshold=0.6
        )
        cache = LLMCache(config)
        
        # Store original optimization query
        original_query = """
        Optimize this function for better performance:
        def calculate_fitness(population):
            scores = []
            for individual in population:
                score = evaluate(individual)
                scores.append(score)
            return scores
        """
        
        mock_result = Mock(spec=QueryResult)
        mock_result.content = "Use list comprehension for better performance"
        
        cache.put(
            original_query,
            "You are a performance optimization expert",
            mock_result,
            [],
            {"model_name": "gpt-4"}
        )
        
        # Similar query with slight variations should match
        similar_query = """
        Optimize this function for improved performance:
        def calculate_fitness(population):
            scores = []
            for individual in population:
                score = evaluate(individual)
                scores.append(score)
            return scores
        """
        
        result, _ = cache.get(
            similar_query,
            "You are a performance optimization expert",
            [],
            {"model_name": "gpt-4"}
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.content, "Use list comprehension for better performance")
        
        # Verify fuzzy match was used
        stats = cache.get_stats()
        self.assertEqual(stats["fuzzy_matches"], 1)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)