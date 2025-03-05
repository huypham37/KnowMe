import os
import json
import hashlib
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import concurrent.futures

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """
        Encode a single text into an embedding vector
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts into embedding vectors
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model"""
        pass


class EmbeddingCache:
    """Cache for storing and retrieving embeddings"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the embedding cache
        
        Args:
            cache_dir: Directory to store the cache files, defaults to ~/.cache/tutorllm/embeddings
        """
        if cache_dir is None:
            home_dir = os.path.expanduser("~")
            cache_dir = os.path.join(home_dir, ".cache", "tutorllm", "embeddings")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Embedding cache initialized at {self.cache_dir}")
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get embedding from cache if available
        
        Args:
            text: The text that was embedded
            model_name: Name of the model used for embedding
            
        Returns:
            Cached embedding vector or None if not found
        """
        cache_key = self._get_cache_key(text, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                self.logger.debug(f"Cache hit for text: {text[:30]}...")
                return cache_data.get("embedding")
            except Exception as e:
                self.logger.warning(f"Failed to read cache file {cache_path}: {str(e)}")
        
        return None
    
    def store(self, text: str, model_name: str, embedding: List[float]) -> None:
        """
        Store embedding in cache
        
        Args:
            text: The text that was embedded
            model_name: Name of the model used for embedding
            embedding: The embedding vector
        """
        cache_key = self._get_cache_key(text, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                "model": model_name,
                "text_hash": hashlib.md5(text.encode()).hexdigest(),
                "embedding": embedding,
                "timestamp": time.time()
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
            self.logger.debug(f"Stored embedding in cache for text: {text[:30]}...")
        except Exception as e:
            self.logger.warning(f"Failed to write to cache file {cache_path}: {str(e)}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate a unique cache key for a text and model combination"""
        combined = f"{model_name}:{text}"
        hash_obj = hashlib.md5(combined.encode())
        return hash_obj.hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.json"


class EmbeddingService:
    """Main service for text embedding with support for multiple models and caching"""
    
    def __init__(
        self,
        primary_model: EmbeddingModel,
        fallback_model: Optional[EmbeddingModel] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding service
        
        Args:
            primary_model: Primary embedding model to use
            fallback_model: Fallback model to use if primary fails
            use_cache: Whether to use caching
            cache_dir: Directory for cache storage
        """
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.use_cache = use_cache
        self.cache = EmbeddingCache(cache_dir) if use_cache else None
        self.logger = logging.getLogger(__name__)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        # Check cache first if enabled
        if self.use_cache:
            cached_embedding = self.cache.get(text, self.primary_model.model_name)
            if cached_embedding:
                return cached_embedding
        
        # Try primary model
        try:
            embedding = self.primary_model.encode(text)
            
            # Store in cache if enabled
            if self.use_cache and embedding:
                self.cache.store(text, self.primary_model.model_name, embedding)
                
            return embedding
            
        except Exception as e:
            self.logger.warning(f"Primary model embedding failed: {str(e)}")
            
            # Try fallback model if available
            if self.fallback_model:
                try:
                    self.logger.info("Using fallback embedding model")
                    embedding = self.fallback_model.encode(text)
                    
                    # Store in cache if enabled
                    if self.use_cache and embedding:
                        self.cache.store(text, self.fallback_model.model_name, embedding)
                        
                    return embedding
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback model embedding also failed: {str(fallback_error)}")
            
            # Re-raise the original error if no fallback or fallback fails
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed multiple texts, with caching and fallback support
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches for processing
            
        Returns:
            List of embedding vectors
        """
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first if enabled
        if self.use_cache:
            for i, text in enumerate(texts):
                cached_embedding = self.cache.get(text, self.primary_model.model_name)
                if cached_embedding:
                    results.append(cached_embedding)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # If all were in cache, return early
        if not uncached_texts:
            return results
        
        # Process uncached texts in batches
        uncached_results = [None] * len(uncached_texts)
        
        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i + batch_size]
            
            try:
                # Try primary model for this batch
                batch_embeddings = self.primary_model.encode_batch(batch)
                
                # Store results and update cache
                for j, embedding in enumerate(batch_embeddings):
                    idx = i + j
                    if idx < len(uncached_texts):
                        uncached_results[idx] = embedding
                        if self.use_cache:
                            self.cache.store(uncached_texts[idx], self.primary_model.model_name, embedding)
                            
            except Exception as e:
                self.logger.warning(f"Primary model batch embedding failed: {str(e)}")
                self._handle_batch_fallback(batch, i, uncached_texts, uncached_results)
        
        # Merge cached and newly computed results
        return self._merge_results(texts, results, uncached_results, uncached_indices)
    
    def _handle_batch_fallback(self, batch, start_idx, uncached_texts, uncached_results):
        """Handle fallback processing for a batch of texts"""
        # First try fallback model if available
        if self.fallback_model:
            self.logger.info("Using fallback model for individual processing")
            for j, text in enumerate(batch):
                idx = start_idx + j
                if idx < len(uncached_texts):
                    try:
                        embedding = self.fallback_model.encode(text)
                        uncached_results[idx] = embedding
                        if self.use_cache:
                            self.cache.store(text, self.fallback_model.model_name, embedding)
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback embedding failed: {str(fallback_error)}")
                        uncached_results[idx] = []
        else:
            # No fallback, try individual processing with primary model
            for j, text in enumerate(batch):
                idx = start_idx + j
                if idx < len(uncached_texts):
                    try:
                        embedding = self.primary_model.encode(text)
                        uncached_results[idx] = embedding
                        if self.use_cache:
                            self.cache.store(text, self.primary_model.model_name, embedding)
                    except Exception:
                        uncached_results[idx] = []
    
    def _merge_results(self, texts, cached_results, uncached_results, uncached_indices):
        """Merge cached and newly computed results in the correct order"""
        merged_results = [None] * len(texts)
        cached_idx = 0
        
        for i in range(len(texts)):
            if i in uncached_indices:
                idx = uncached_indices.index(i)
                merged_results[i] = uncached_results[idx]
            else:
                merged_results[i] = cached_results[cached_idx]
                cached_idx += 1
        
        return merged_results