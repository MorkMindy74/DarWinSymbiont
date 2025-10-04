"""
Darwin-style adapter for ShinkaEvolve LLM caching system.

This module provides a simplified interface that matches the expected
darwin.cache.CachedLLM API for backward compatibility and ease of use.
"""

import logging
from typing import Dict, Any, Callable, Optional
from .cache import LLMCache, CacheConfig
from .models import QueryResult

logger = logging.getLogger(__name__)


class CachedLLM:
    """
    Darwin-style cached LLM wrapper with simplified interface.
    
    This class provides a more straightforward API that matches 
    the expected darwin.cache interface while using ShinkaEvolve's
    robust caching backend.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cached LLM with config dictionary.
        
        Args:
            config: Dictionary with cache configuration
                - enabled: bool, whether caching is enabled
                - mode: str, "exact" or "fuzzy"
                - backend: str, "sqlite" (only supported backend)
                - path: str, path to cache database
                - ttl_hours: float, time-to-live in hours
        """
        # Convert dict config to CacheConfig object
        self.cache_config = CacheConfig(
            enabled=config.get("enabled", True),
            mode=config.get("mode", "exact"),
            path=config.get("path", "./.cache/llm_cache.db"),
            ttl_hours=config.get("ttl_hours", 168.0),
            key_fields=config.get("key_fields", ["prompt", "model"])
        )
        
        # Initialize cache
        self.cache = LLMCache(self.cache_config)
        
        # Model function placeholder
        self.model_fn: Optional[Callable[[str], str]] = None
        
        # Statistics tracking
        self.call_count = 0
        self.hit_count = 0
        self.miss_count = 0
        
    def set_model(self, model_fn: Callable[[str], str]) -> None:
        """
        Set the underlying model function to use for cache misses.
        
        Args:
            model_fn: Function that takes a prompt string and returns a result string
        """
        self.model_fn = model_fn
        logger.info("Model function set for CachedLLM")
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Execute a cached LLM call.
        
        Args:
            prompt: The input prompt string
            **kwargs: Additional parameters (model_name, temperature, etc.)
            
        Returns:
            Result string from cache or model call
        """
        if self.model_fn is None:
            raise ValueError("Model function not set. Call set_model() first.")
        
        self.call_count += 1
        
        # Prepare parameters for cache lookup
        model_name = kwargs.get("model_name", "default_model")
        system_msg = kwargs.get("system_msg", "")
        llm_kwargs = {
            "model_name": model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "seed": kwargs.get("seed"),
        }
        
        # Try cache first
        cached_result, cache_key = self.cache.get(
            msg=prompt,
            system_msg=system_msg,
            msg_history=[],
            llm_kwargs=llm_kwargs,
            tool_state=kwargs.get("tool_state")
        )
        
        if cached_result is not None:
            # Cache hit
            self.hit_count += 1
            result_str = cached_result.content
            logger.info(f"[CACHE] HIT key={cache_key[:12]}... result_len={len(result_str)}")
            return result_str
        
        # Cache miss - call actual model
        self.miss_count += 1
        logger.info(f"[CACHE] MISS key={cache_key[:12]}... calling model")
        
        result_str = self.model_fn(prompt)
        
        # Create QueryResult for caching
        query_result = QueryResult(
            content=result_str,
            msg=prompt,
            system_msg=system_msg,
            new_msg_history=[],
            model_name=model_name,
            kwargs=llm_kwargs,
            input_tokens=len(prompt.split()),  # Rough estimate
            output_tokens=len(result_str.split()),  # Rough estimate
            cost=0.0  # Unknown for fake model
        )
        
        # Store in cache
        self.cache.put(
            msg=prompt,
            system_msg=system_msg,
            result=query_result,
            msg_history=[],
            llm_kwargs=llm_kwargs,
            tool_state=kwargs.get("tool_state")
        )
        
        return result_str
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about cache performance."""
        cache_stats = self.cache.get_stats()
        
        # Merge with our local stats
        combined_stats = {
            **cache_stats,
            "total_calls": self.call_count,
            "local_hits": self.hit_count,
            "local_misses": self.miss_count,
            "local_hit_rate": (self.hit_count / max(self.call_count, 1)) * 100
        }
        
        return combined_stats
    
    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        removed = self.cache.cleanup_expired()
        logger.info(f"Removed {removed} expired cache entries")
        return removed


# Backward compatibility alias
CachedLLMClient = CachedLLM