"""
Deterministic LLM Caching System for ShinkaEvolve

This module provides persistent, deterministic caching for LLM queries with:
- SHA256-based deterministic keys (prompt, seed, model, tool_state)
- SQLite backend for persistence
- Configurable TTL (time-to-live)
- Exact and fuzzy matching modes
- Comprehensive logging

Example Usage:
    cache = LLMCache(enabled=True, mode="exact", ttl_hours=168)
    cached_llm = CachedLLMClient(llm_client, cache)
    result = cached_llm.query(msg="Hello", system_msg="You are helpful")
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import pickle
import threading
from datetime import datetime, timedelta

from ..models import QueryResult

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for LLM caching system."""
    enabled: bool = True
    mode: str = "exact"  # "exact" or "fuzzy"
    backend: str = "sqlite"
    path: str = "./.cache/llm_cache.db"
    ttl_hours: float = 168.0  # 7 days default
    key_fields: List[str] = field(default_factory=lambda: [
        "prompt", "seed", "model", "tool_state"
    ])
    # Fuzzy matching parameters (MinHash)
    minhash_perm: int = 128  # Number of hash functions for MinHash
    fuzzy_threshold: float = 0.8  # Similarity threshold for fuzzy matching


class CacheEntry:
    """Represents a cached LLM query and response."""
    
    def __init__(
        self,
        cache_key: str,
        query_data: Dict[str, Any],
        result_data: bytes,  # Pickled QueryResult
        timestamp: float,
        ttl_hours: float,
        minhash_signature: Optional[List[int]] = None
    ):
        self.cache_key = cache_key
        self.query_data = query_data
        self.result_data = result_data
        self.timestamp = timestamp
        self.ttl_hours = ttl_hours
        self.minhash_signature = minhash_signature
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired based on TTL."""
        if self.ttl_hours <= 0:
            return False  # Never expires if TTL is 0 or negative
        
        expiry_time = self.timestamp + (self.ttl_hours * 3600)
        return time.time() > expiry_time
    
    @property 
    def age_hours(self) -> float:
        """Get age of cache entry in hours."""
        return (time.time() - self.timestamp) / 3600
    
    def to_query_result(self) -> QueryResult:
        """Deserialize the cached result back to QueryResult."""
        return pickle.loads(self.result_data)


class MinHasher:
    """MinHash implementation for fuzzy prompt matching."""
    
    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        # Generate random hash coefficients
        import random
        random.seed(42)  # Fixed seed for deterministic hashing
        self.coeffs_a = [random.randint(1, 2**32 - 1) for _ in range(num_perm)]
        self.coeffs_b = [random.randint(0, 2**32 - 1) for _ in range(num_perm)]
    
    def _hash_function(self, x: int, a: int, b: int) -> int:
        """Universal hash function."""
        return (a * x + b) % (2**32 - 1)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into shingles for MinHash."""
        # Create 3-character shingles
        shingles = []
        for i in range(len(text) - 2):
            shingle = text[i:i+3].lower()
            shingles.append(shingle)
        return list(set(shingles))  # Remove duplicates
    
    def compute_signature(self, text: str) -> List[int]:
        """Compute MinHash signature for given text."""
        tokens = self._tokenize(text)
        if not tokens:
            return [0] * self.num_perm
        
        # Convert tokens to integer hashes
        token_hashes = [hash(token) % (2**32 - 1) for token in tokens]
        
        signature = []
        for i in range(self.num_perm):
            min_hash = float('inf')
            for token_hash in token_hashes:
                h = self._hash_function(token_hash, self.coeffs_a[i], self.coeffs_b[i])
                min_hash = min(min_hash, h)
            signature.append(int(min_hash))
        
        return signature
    
    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class LLMCache:
    """SQLite-backed persistent cache for LLM queries."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.db_path = Path(config.path)
        self.db_lock = threading.RLock()  # Thread-safe access
        
        if config.mode == "fuzzy":
            self.minhasher = MinHasher(config.minhash_perm)
        else:
            self.minhasher = None
        
        # Ensure cache directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "expired": 0,
            "fuzzy_matches": 0
        }
        
        logger.info(f"LLM Cache initialized: {config.mode} mode, TTL: {config.ttl_hours}h, Path: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                # Create cache table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        cache_key TEXT PRIMARY KEY,
                        query_data TEXT NOT NULL,
                        result_data BLOB NOT NULL,
                        timestamp REAL NOT NULL,
                        ttl_hours REAL NOT NULL,
                        minhash_signature TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for timestamp-based queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON cache_entries(timestamp)
                """)
                
                # Create metadata table for statistics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.debug("Database initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise
            finally:
                conn.close()
    
    def _generate_cache_key(self, query_params: Dict[str, Any]) -> str:
        """Generate deterministic SHA256 hash key from query parameters."""
        # Extract only the configured key fields
        key_data = {}
        for field in self.config.key_fields:
            if field in query_params:
                key_data[field] = query_params[field]
        
        # Sort keys for deterministic ordering
        sorted_data = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        
        # Generate SHA256 hash
        hash_object = hashlib.sha256(sorted_data.encode('utf-8'))
        return hash_object.hexdigest()
    
    def _normalize_query_params(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict],
        llm_kwargs: Dict[str, Any],
        tool_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Normalize query parameters for consistent caching."""
        # Combine message components into a single prompt
        prompt_parts = [system_msg, msg]
        if msg_history:
            # Add message history to prompt
            for hist_msg in msg_history:
                if isinstance(hist_msg, dict):
                    content = hist_msg.get('content', str(hist_msg))
                    prompt_parts.append(content)
        
        prompt = "\n".join(filter(None, prompt_parts))
        
        # Extract relevant parameters
        normalized = {
            "prompt": prompt,
            "model": llm_kwargs.get("model_name", "unknown"),
            "seed": llm_kwargs.get("seed"),
            "temperature": llm_kwargs.get("temperature"),
            "max_tokens": llm_kwargs.get("max_tokens"),
            "tool_state": tool_state
        }
        
        # Remove None values to avoid inconsistent keys
        return {k: v for k, v in normalized.items() if v is not None}
    
    def get(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = None,
        llm_kwargs: Dict[str, Any] = None,
        tool_state: Optional[Dict] = None
    ) -> Tuple[Optional[QueryResult], str]:
        """
        Retrieve cached result for given query parameters.
        
        Returns:
            Tuple of (QueryResult if found, cache_key used)
        """
        if not self.config.enabled:
            return None, ""
        
        msg_history = msg_history or []
        llm_kwargs = llm_kwargs or {}
        
        query_params = self._normalize_query_params(
            msg, system_msg, msg_history, llm_kwargs, tool_state
        )
        cache_key = self._generate_cache_key(query_params)
        
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                
                # Try exact match first
                cursor.execute(
                    "SELECT query_data, result_data, timestamp, ttl_hours, minhash_signature "
                    "FROM cache_entries WHERE cache_key = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()
                
                if row:
                    query_data_json, result_data, timestamp, ttl_hours, minhash_sig = row
                    entry = CacheEntry(
                        cache_key=cache_key,
                        query_data=json.loads(query_data_json),
                        result_data=result_data,
                        timestamp=timestamp,
                        ttl_hours=ttl_hours,
                        minhash_signature=json.loads(minhash_sig) if minhash_sig else None
                    )
                    
                    if entry.is_expired:
                        self.stats["expired"] += 1
                        logger.info(f"[CACHE] EXPIRED key={cache_key[:12]} age={entry.age_hours:.1f}h")
                        # Remove expired entry
                        cursor.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
                        conn.commit()
                        return None, cache_key
                    
                    # Cache hit!
                    self.stats["hits"] += 1
                    result = entry.to_query_result()
                    prompt_len = len(query_params.get("prompt", ""))
                    model = query_params.get("model", "unknown")
                    logger.info(f"[CACHE] HIT key={cache_key[:12]} (model={model}, prompt_len={prompt_len})")
                    return result, cache_key
                
                # No exact match, try fuzzy matching if enabled
                if self.config.mode == "fuzzy" and self.minhasher:
                    return self._fuzzy_lookup(query_params, cursor, conn)
                
                # Cache miss
                self.stats["misses"] += 1
                return None, cache_key
                
            except Exception as e:
                logger.error(f"Error retrieving from cache: {e}")
                return None, cache_key
            finally:
                conn.close()
    
    def _fuzzy_lookup(
        self, 
        query_params: Dict[str, Any], 
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection
    ) -> Tuple[Optional[QueryResult], str]:
        """Perform fuzzy matching using MinHash signatures."""
        prompt = query_params.get("prompt", "")
        if not prompt:
            return None, ""
        
        # Compute MinHash signature for current prompt
        current_signature = self.minhasher.compute_signature(prompt)
        
        # Get all non-expired entries with MinHash signatures
        cursor.execute("""
            SELECT cache_key, query_data, result_data, timestamp, ttl_hours, minhash_signature
            FROM cache_entries 
            WHERE minhash_signature IS NOT NULL
              AND (timestamp + ttl_hours * 3600) > ?
        """, (time.time(),))
        
        best_match = None
        best_similarity = 0.0
        best_cache_key = ""
        
        for row in cursor.fetchall():
            cache_key, query_data_json, result_data, timestamp, ttl_hours, minhash_sig = row
            
            try:
                stored_signature = json.loads(minhash_sig)
                similarity = self.minhasher.jaccard_similarity(current_signature, stored_signature)
                
                if similarity > best_similarity and similarity >= self.config.fuzzy_threshold:
                    best_similarity = similarity
                    best_match = CacheEntry(
                        cache_key=cache_key,
                        query_data=json.loads(query_data_json),
                        result_data=result_data,
                        timestamp=timestamp,
                        ttl_hours=ttl_hours,
                        minhash_signature=stored_signature
                    )
                    best_cache_key = cache_key
            except (json.JSONDecodeError, ValueError):
                continue
        
        if best_match:
            self.stats["fuzzy_matches"] += 1
            result = best_match.to_query_result()
            prompt_len = len(prompt)
            model = query_params.get("model", "unknown")
            logger.info(
                f"[CACHE] FUZZY_MATCH key={best_cache_key[:12]} "
                f"similarity={best_similarity:.3f} (model={model}, prompt_len={prompt_len})"
            )
            return result, best_cache_key
        
        # No fuzzy match found
        self.stats["misses"] += 1
        return None, ""
    
    def put(
        self,
        msg: str,
        system_msg: str,
        result: QueryResult,
        msg_history: List[Dict] = None,
        llm_kwargs: Dict[str, Any] = None,
        tool_state: Optional[Dict] = None
    ) -> str:
        """
        Store query result in cache.
        
        Returns:
            The cache key used for storage
        """
        if not self.config.enabled:
            return ""
        
        msg_history = msg_history or []
        llm_kwargs = llm_kwargs or {}
        
        query_params = self._normalize_query_params(
            msg, system_msg, msg_history, llm_kwargs, tool_state
        )
        cache_key = self._generate_cache_key(query_params)
        
        try:
            # Serialize result
            result_data = pickle.dumps(result)
            
            # Compute MinHash signature if fuzzy mode is enabled
            minhash_signature = None
            if self.config.mode == "fuzzy" and self.minhasher:
                prompt = query_params.get("prompt", "")
                if prompt:
                    minhash_signature = self.minhasher.compute_signature(prompt)
            
            with self.db_lock:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    cursor = conn.cursor()
                    
                    # Insert or replace cache entry
                    cursor.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (cache_key, query_data, result_data, timestamp, ttl_hours, minhash_signature)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        cache_key,
                        json.dumps(query_params),
                        result_data,
                        time.time(),
                        self.config.ttl_hours,
                        json.dumps(minhash_signature) if minhash_signature else None
                    ))
                    
                    conn.commit()
                    
                    prompt_len = len(query_params.get("prompt", ""))
                    model = query_params.get("model", "unknown")
                    logger.info(
                        f"[CACHE] STORE key={cache_key[:12]} "
                        f"(model={model}, prompt_len={prompt_len})"
                    )
                    
                except Exception as e:
                    logger.error(f"Error storing to cache: {e}")
                finally:
                    conn.close()
                    
        except Exception as e:
            logger.error(f"Error serializing result for cache: {e}")
            
        return cache_key
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache. Returns number of entries removed."""
        if not self.config.enabled:
            return 0
        
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                
                # Delete expired entries
                cursor.execute("""
                    DELETE FROM cache_entries 
                    WHERE (timestamp + ttl_hours * 3600) <= ?
                """, (time.time(),))
                
                removed_count = cursor.rowcount
                conn.commit()
                
                if removed_count > 0:
                    logger.info(f"[CACHE] CLEANUP removed {removed_count} expired entries")
                
                return removed_count
                
            except Exception as e:
                logger.error(f"Error during cache cleanup: {e}")
                return 0
            finally:
                conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                
                # Count total entries
                cursor.execute("SELECT COUNT(*) FROM cache_entries")
                total_entries = cursor.fetchone()[0]
                
                # Count expired entries
                cursor.execute("""
                    SELECT COUNT(*) FROM cache_entries 
                    WHERE (timestamp + ttl_hours * 3600) <= ?
                """, (time.time(),))
                expired_entries = cursor.fetchone()[0]
                
                # Calculate hit rate
                total_requests = self.stats["hits"] + self.stats["misses"]
                hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
                
                return {
                    "enabled": self.config.enabled,
                    "mode": self.config.mode,
                    "total_entries": total_entries,
                    "expired_entries": expired_entries,
                    "active_entries": total_entries - expired_entries,
                    "hit_rate_percent": hit_rate,
                    **self.stats
                }
                
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
                return {"error": str(e)}
            finally:
                conn.close()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.db_lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries")
                conn.commit()
                logger.info("[CACHE] All entries cleared")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
            finally:
                conn.close()


class CachedLLMClient:
    """
    Wrapper around LLMClient that adds deterministic caching.
    
    This class provides a drop-in replacement for LLMClient with caching capabilities.
    """
    
    def __init__(self, llm_client, cache: Optional[LLMCache] = None):
        """
        Initialize cached LLM client.
        
        Args:
            llm_client: The underlying LLMClient instance
            cache: LLMCache instance, or None to disable caching
        """
        self.llm_client = llm_client
        self.cache = cache
    
    def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = None,
        llm_kwargs: Optional[Dict] = None,
        tool_state: Optional[Dict] = None
    ) -> Optional[QueryResult]:
        """
        Execute a query with caching support.
        
        Args:
            msg: The message to query the LLM with
            system_msg: The system message
            msg_history: Message history
            llm_kwargs: LLM parameters
            tool_state: Additional tool state for cache key
            
        Returns:
            QueryResult from cache or fresh LLM call
        """
        msg_history = msg_history or []
        llm_kwargs = llm_kwargs or {}
        
        # Try cache first
        if self.cache:
            cached_result, cache_key = self.cache.get(
                msg, system_msg, msg_history, llm_kwargs, tool_state
            )
            if cached_result:
                return cached_result
        
        # Cache miss - query the actual LLM
        result = self.llm_client.query(
            msg=msg,
            system_msg=system_msg,
            msg_history=msg_history,
            llm_kwargs=llm_kwargs
        )
        
        # Store result in cache
        if self.cache and result:
            self.cache.put(
                msg, system_msg, result, msg_history, llm_kwargs, tool_state
            )
        
        return result
    
    def batch_query(self, *args, **kwargs):
        """Forward batch_query to underlying client (no caching for batch operations)."""
        return self.llm_client.batch_query(*args, **kwargs)
    
    def batch_kwargs_query(self, *args, **kwargs):
        """Forward batch_kwargs_query to underlying client (no caching for batch operations)."""
        return self.llm_client.batch_kwargs_query(*args, **kwargs)
    
    def get_kwargs(self):
        """Forward get_kwargs to underlying client."""
        return self.llm_client.get_kwargs()
    
    def __getattr__(self, name):
        """Forward all other attributes to the underlying client."""
        return getattr(self.llm_client, name)