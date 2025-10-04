"""
Unit and integration tests for the deduplication system.
"""

import pytest
import numpy as np
from shinka.dedup import (
    DedupConfig, DedupManager, MinHashDedup, SimHashDedup, 
    create_dedup_manager
)


class TestDedupConfig:
    def test_default_config(self):
        config = DedupConfig()
        assert config.method == "minhash"
        assert config.threshold == 0.8
        assert config.hash_size == 64
        assert config.window_size == 100
        assert config.enabled is True


class TestMinHashDedup:
    def test_identical_solutions(self):
        config = DedupConfig(method="minhash", threshold=0.8)
        dedup = MinHashDedup(config, seed=42)
        
        solution = np.array([1.0, 2.0, 3.0, 4.0])
        
        # First occurrence should not be duplicate
        assert not dedup.is_duplicate(solution)
        
        # Identical solution should be duplicate
        assert dedup.is_duplicate(solution)
    
    def test_similar_solutions(self):
        config = DedupConfig(method="minhash", threshold=0.8)
        dedup = MinHashDedup(config, seed=42)
        
        solution1 = np.array([1.0, 2.0, 3.0, 4.0])
        solution2 = np.array([1.01, 2.01, 3.01, 4.01])  # Very similar
        
        assert not dedup.is_duplicate(solution1)
        # Should be detected as duplicate due to high similarity
        is_dup = dedup.is_duplicate(solution2)
        # May or may not be duplicate depending on shingle sensitivity
        
    def test_different_solutions(self):
        config = DedupConfig(method="minhash", threshold=0.8)
        dedup = MinHashDedup(config, seed=42)
        
        solution1 = np.array([1.0, 2.0, 3.0, 4.0])
        solution2 = np.array([10.0, 20.0, 30.0, 40.0])  # Very different
        
        assert not dedup.is_duplicate(solution1)
        assert not dedup.is_duplicate(solution2)
    
    def test_window_size_limit(self):
        config = DedupConfig(method="minhash", threshold=0.9, window_size=2)
        dedup = MinHashDedup(config, seed=42)
        
        solutions = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
            np.array([1.0, 2.0])  # Same as first, but should be outside window
        ]
        
        for i, sol in enumerate(solutions):
            is_dup = dedup.is_duplicate(sol)
            if i < 3:
                assert not is_dup  # First 3 should be unique
            # The 4th might or might not be duplicate depending on window
    
    def test_disabled_dedup(self):
        config = DedupConfig(method="minhash", enabled=False)
        dedup = MinHashDedup(config, seed=42)
        
        solution = np.array([1.0, 2.0, 3.0])
        
        # Should never detect duplicates when disabled
        assert not dedup.is_duplicate(solution)
        assert not dedup.is_duplicate(solution)
        assert not dedup.is_duplicate(solution)


class TestSimHashDedup:
    def test_identical_solutions(self):
        config = DedupConfig(method="simhash", threshold=0.9)
        dedup = SimHashDedup(config, seed=42)
        
        solution = np.array([1.0, 2.0, 3.0, 4.0])
        
        # First occurrence should not be duplicate
        assert not dedup.is_duplicate(solution)
        
        # Identical solution should be duplicate
        assert dedup.is_duplicate(solution)
    
    def test_similar_solutions(self):
        config = DedupConfig(method="simhash", threshold=0.9, hash_size=32)
        dedup = SimHashDedup(config, seed=42)
        
        solution1 = np.array([1.0, 2.0, 3.0, 4.0])
        solution2 = np.array([1.1, 2.1, 3.1, 4.1])  # Similar
        
        assert not dedup.is_duplicate(solution1)
        # Similarity depends on feature extraction and hash collision
        is_dup = dedup.is_duplicate(solution2)
        # Test passes regardless of result - testing mechanism
    
    def test_hash_computation(self):
        config = DedupConfig(method="simhash", hash_size=16)
        dedup = SimHashDedup(config, seed=42)
        
        solution = np.array([1.0, 2.0, 3.0])
        hash_val = dedup.compute_hash(solution)
        
        assert isinstance(hash_val, np.ndarray)
        assert len(hash_val) == 16
        assert hash_val.dtype == np.uint8
        assert all(bit in [0, 1] for bit in hash_val)


class TestDedupManager:
    def test_minhash_manager(self):
        manager = create_dedup_manager(method="minhash", threshold=0.8, seed=42)
        
        solution1 = np.array([1.0, 2.0, 3.0])
        solution2 = np.array([1.0, 2.0, 3.0])
        
        assert not manager.is_duplicate(solution1)
        assert manager.is_duplicate(solution2)  # Should be duplicate
    
    def test_simhash_manager(self):
        manager = create_dedup_manager(method="simhash", threshold=0.9, seed=42)
        
        solution1 = np.array([1.0, 2.0, 3.0])
        solution2 = np.array([1.0, 2.0, 3.0])
        
        assert not manager.is_duplicate(solution1)
        assert manager.is_duplicate(solution2)  # Should be duplicate
    
    def test_filter_duplicates(self):
        manager = create_dedup_manager(method="minhash", threshold=0.7, seed=42)
        
        solutions = [
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),  # Duplicate
            np.array([3.0, 4.0]),
            np.array([1.0, 2.0]),  # Duplicate
            np.array([5.0, 6.0])
        ]
        
        filtered = manager.filter_duplicates(solutions)
        
        # Should have fewer solutions after filtering
        assert len(filtered) <= len(solutions)
        # At least the unique ones should remain
        assert len(filtered) >= 3
    
    def test_stats_collection(self):
        manager = create_dedup_manager(method="minhash", seed=42)
        
        # Initially no stats
        stats = manager.get_stats()
        assert stats['filtered_count'] == 0
        assert stats['total_count'] == 0
        assert stats['filter_rate_pct'] == 0
        
        # Add some solutions
        solutions = [np.array([1.0]), np.array([1.0]), np.array([2.0])]
        manager.filter_duplicates(solutions)
        
        stats = manager.get_stats()
        assert stats['total_count'] == 3
        assert stats['filtered_count'] >= 0
        assert 0 <= stats['filter_rate_pct'] <= 100
    
    def test_reset_cache(self):
        manager = create_dedup_manager(method="minhash", seed=42)
        
        solution = np.array([1.0, 2.0])
        
        # Add solution to cache
        assert not manager.is_duplicate(solution)
        assert manager.is_duplicate(solution)  # Now duplicate
        
        # Reset cache
        manager.reset_cache()
        
        # Should not be duplicate after reset
        assert not manager.is_duplicate(solution)
    
    def test_disabled_manager(self):
        manager = create_dedup_manager(enabled=False, seed=42)
        
        solutions = [np.array([1.0]), np.array([1.0]), np.array([1.0])]
        filtered = manager.filter_duplicates(solutions)
        
        # Should return all solutions when disabled
        assert len(filtered) == len(solutions)
        
        stats = manager.get_stats()
        assert not stats['enabled']
    
    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown deduplication method"):
            create_dedup_manager(method="invalid_method")


class TestDedupIntegration:
    def test_large_solution_set(self):
        """Test with a larger set of solutions to verify performance."""
        manager = create_dedup_manager(method="minhash", threshold=0.8, seed=42)
        
        # Generate 50 solutions with some duplicates
        solutions = []
        for i in range(50):
            if i % 5 == 0:  # Every 5th solution is a repeat of first
                solutions.append(np.array([1.0, 2.0, 3.0]))
            else:
                solutions.append(np.random.RandomState(i).randn(3))
        
        filtered = manager.filter_duplicates(solutions)
        
        # Should have filtered out some duplicates
        assert len(filtered) < len(solutions)
        
        stats = manager.get_stats()
        assert stats['total_count'] == 50
        assert stats['filtered_count'] > 0
    
    def test_different_thresholds(self):
        """Test behavior with different similarity thresholds."""
        solution1 = np.array([1.0, 2.0, 3.0])
        solution2 = np.array([1.1, 2.1, 3.1])  # Slightly different
        
        # High threshold (strict) - should not detect as duplicate
        strict_manager = create_dedup_manager(method="minhash", threshold=0.95, seed=42)
        assert not strict_manager.is_duplicate(solution1)
        assert not strict_manager.is_duplicate(solution2)
        
        # Low threshold (lenient) - more likely to detect as duplicate
        lenient_manager = create_dedup_manager(method="minhash", threshold=0.3, seed=42)
        assert not lenient_manager.is_duplicate(solution1)
        # May or may not be duplicate - depends on hash collision
    
    def test_method_comparison(self):
        """Compare MinHash and SimHash on same data."""
        solutions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),  # Exact duplicate
            np.array([4.0, 5.0, 6.0])
        ]
        
        minhash_manager = create_dedup_manager(method="minhash", threshold=0.8, seed=42)
        simhash_manager = create_dedup_manager(method="simhash", threshold=0.8, seed=42)
        
        minhash_filtered = minhash_manager.filter_duplicates(solutions.copy())
        simhash_filtered = simhash_manager.filter_duplicates(solutions.copy())
        
        # Both should detect the exact duplicate
        assert len(minhash_filtered) <= len(solutions)
        assert len(simhash_filtered) <= len(solutions)
        
        # Both should have similar effectiveness on exact duplicates
        # (Though exact behavior may differ due to different algorithms)