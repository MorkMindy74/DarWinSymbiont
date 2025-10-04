"""
Low-cost deduplication system using MinHash and SimHash
for filtering near-identical mutations in evolutionary algorithms.
"""

import numpy as np
import hashlib
from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    """Configuration for deduplication system."""
    method: str = "minhash"  # "minhash" or "simhash"
    threshold: float = 0.8   # Similarity threshold for deduplication
    hash_size: int = 64      # Number of hash functions for MinHash or bits for SimHash
    window_size: int = 100   # Size of recent solutions cache
    enabled: bool = True     # Enable/disable deduplication


class DedupBase(ABC):
    """Base class for deduplication strategies."""
    
    def __init__(self, config: DedupConfig):
        self.config = config
        self.recent_hashes: List[Any] = []
        self.filtered_count = 0
        self.total_count = 0
        
    @abstractmethod
    def compute_hash(self, solution: np.ndarray) -> Any:
        """Compute hash representation of solution."""
        pass
    
    @abstractmethod
    def similarity(self, hash1: Any, hash2: Any) -> float:
        """Compute similarity between two hashes."""
        pass
    
    def is_duplicate(self, solution: np.ndarray) -> bool:
        """Check if solution is too similar to recent solutions."""
        if not self.config.enabled:
            return False
            
        self.total_count += 1
        solution_hash = self.compute_hash(solution)
        
        # Check similarity against recent solutions
        for recent_hash in self.recent_hashes:
            similarity_score = self.similarity(solution_hash, recent_hash)
            if similarity_score >= self.config.threshold:
                self.filtered_count += 1
                logger.debug(f"Filtered duplicate solution (similarity: {similarity_score:.3f})")
                return True
        
        # Add to recent cache
        self.recent_hashes.append(solution_hash)
        if len(self.recent_hashes) > self.config.window_size:
            self.recent_hashes.pop(0)
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        filter_rate = (self.filtered_count / self.total_count) * 100 if self.total_count > 0 else 0
        return {
            'method': self.config.method,
            'filtered_count': self.filtered_count,
            'total_count': self.total_count,
            'filter_rate_pct': filter_rate,
            'threshold': self.config.threshold,
            'enabled': self.config.enabled
        }


class MinHashDedup(DedupBase):
    """MinHash-based deduplication using Jaccard similarity."""
    
    def __init__(self, config: DedupConfig, seed: int = 42):
        super().__init__(config)
        self.rng = np.random.RandomState(seed)
        
        # Generate random hash functions (coefficients for linear hashing)
        self.hash_funcs = []
        for _ in range(config.hash_size):
            a = self.rng.randint(1, 2**31 - 1)
            b = self.rng.randint(0, 2**31 - 1) 
            self.hash_funcs.append((a, b))
    
    def compute_hash(self, solution: np.ndarray) -> np.ndarray:
        """Compute MinHash signature for solution."""
        # Convert solution to shingle set (overlapping k-grams)
        shingles = self._create_shingles(solution, k=2)
        
        # Compute MinHash signature
        signature = np.full(self.config.hash_size, np.inf, dtype=np.uint32)
        
        for shingle in shingles:
            shingle_int = hash(shingle) & 0x7FFFFFFF  # Ensure positive
            
            for i, (a, b) in enumerate(self.hash_funcs):
                hash_val = (a * shingle_int + b) & 0x7FFFFFFF
                signature[i] = min(signature[i], hash_val)
        
        return signature
    
    def _create_shingles(self, solution: np.ndarray, k: int = 2) -> Set[Tuple]:
        """Create k-gram shingles from solution."""
        # Quantize solution to create discrete shingles
        quantized = np.round(solution * 100).astype(int)  # 2 decimal precision
        
        shingles = set()
        for i in range(len(quantized) - k + 1):
            shingle = tuple(quantized[i:i+k])
            shingles.add(shingle)
        
        # Add individual elements as 1-grams
        for val in quantized:
            shingles.add((val,))
            
        return shingles if shingles else {(0,)}  # Avoid empty set
    
    def similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Compute Jaccard similarity from MinHash signatures."""
        matches = np.sum(hash1 == hash2)
        return matches / len(hash1)


class SimHashDedup(DedupBase):
    """SimHash-based deduplication using Hamming distance."""
    
    def __init__(self, config: DedupConfig, seed: int = 42):
        super().__init__(config)
        self.rng = np.random.RandomState(seed)
        
        # Generate random projection vectors
        self.projection_matrix = self.rng.randn(config.hash_size, 1000)  # 1000-dim features
    
    def compute_hash(self, solution: np.ndarray) -> np.ndarray:
        """Compute SimHash fingerprint for solution."""
        # Extract features from solution
        features = self._extract_features(solution)
        
        # Project to hash_size dimensions
        projections = np.dot(self.projection_matrix, features)
        
        # Convert to binary hash
        hash_bits = (projections >= 0).astype(np.uint8)
        
        return hash_bits
    
    def _extract_features(self, solution: np.ndarray) -> np.ndarray:
        """Extract feature vector from solution."""
        features = np.zeros(1000)
        
        # Basic statistical features
        features[0] = np.mean(solution)
        features[1] = np.std(solution)
        features[2] = np.min(solution)
        features[3] = np.max(solution)
        features[4] = np.median(solution)
        
        # Higher-order moments
        features[5] = np.mean(solution**2)
        features[6] = np.mean(solution**3)
        
        # Differences and gradients
        if len(solution) > 1:
            diffs = np.diff(solution)
            features[7] = np.mean(diffs)
            features[8] = np.std(diffs)
            features[9] = np.mean(np.abs(diffs))
        
        # Quantized values (binned features)
        for i, val in enumerate(solution[:min(100, len(solution))]):
            bin_idx = int(np.clip((val + 5) * 10, 0, 99))  # Map [-5,5] to [0,99]
            features[10 + i] = bin_idx
        
        # Fourier transform features (if solution is long enough)
        if len(solution) >= 8:
            fft = np.fft.fft(solution[:8])
            fft_magnitudes = np.abs(fft)
            features[110:118] = fft_magnitudes
        
        # Remaining features filled with solution cycled and transformed
        remaining = 1000 - 118
        extended_solution = np.tile(solution, (remaining // len(solution) + 1))[:remaining]
        features[118:] = extended_solution * np.arange(remaining) / remaining
        
        return features
    
    def similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Compute similarity from Hamming distance."""
        hamming_distance = np.sum(hash1 != hash2)
        max_distance = len(hash1)
        return 1.0 - (hamming_distance / max_distance)


class DedupManager:
    """Main deduplication manager."""
    
    def __init__(self, config: DedupConfig, seed: int = 42):
        self.config = config
        
        if config.method == "minhash":
            self.dedup = MinHashDedup(config, seed)
        elif config.method == "simhash":
            self.dedup = SimHashDedup(config, seed)
        else:
            raise ValueError(f"Unknown deduplication method: {config.method}")
        
        logger.info(f"Deduplication initialized: {config.method} "
                   f"(threshold={config.threshold}, enabled={config.enabled})")
    
    def filter_duplicates(self, solutions: List[np.ndarray]) -> List[np.ndarray]:
        """Filter out duplicate solutions from a list."""
        if not self.config.enabled:
            return solutions
        
        filtered = []
        for solution in solutions:
            if not self.dedup.is_duplicate(solution):
                filtered.append(solution)
        
        stats = self.get_stats()
        if len(solutions) > 0:
            logger.debug(f"Filtered {len(solutions) - len(filtered)}/{len(solutions)} "
                        f"solutions ({stats['filter_rate_pct']:.1f}% total rate)")
        
        return filtered
    
    def is_duplicate(self, solution: np.ndarray) -> bool:
        """Check if single solution is duplicate."""
        return self.dedup.is_duplicate(solution)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return self.dedup.get_stats()
    
    def reset_cache(self):
        """Reset the deduplication cache."""
        self.dedup.recent_hashes.clear()
        logger.info("Deduplication cache reset")


# Convenience function
def create_dedup_manager(method: str = "minhash", 
                        threshold: float = 0.8,
                        hash_size: int = 64,
                        window_size: int = 100,
                        enabled: bool = True,
                        seed: int = 42) -> DedupManager:
    """Create a deduplication manager with specified parameters."""
    config = DedupConfig(
        method=method,
        threshold=threshold,
        hash_size=hash_size,
        window_size=window_size,
        enabled=enabled
    )
    return DedupManager(config, seed)