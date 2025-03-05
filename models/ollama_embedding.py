import time
import logging
from typing import List, Optional
import concurrent.futures
import torch
import torch.nn.functional as F
import ollama
import numpy as np
import sys
import os

# Ensure TutorLLM is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from TutorLLM.core.embedding import EmbeddingModel

class OllamaEmbedding(EmbeddingModel):
    """Embedding model using the official Ollama Python library with Nomic-AI embedding approach"""
    
    def __init__(
        self, 
        model_name: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_workers: int = 4,  # For parallel processing
        matryoshka_dim: int = 512,  # Configurable embedding dimension
        **client_kwargs
    ):
        """
        Initialize the Nomic-AI compatible embedding model
        
        Args:
            model_name: Name of the model (default: nomic-embed-text-v1.5)
            host: Host URL for the Ollama API
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            max_workers: Maximum number of worker threads for batch processing
            matryoshka_dim: Dimension of the embedding (512, 256, 128, or 64)
            client_kwargs: Additional kwargs to pass to the Ollama client
        """
        self._model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        self.matryoshka_dim = matryoshka_dim
        self.logger = logging.getLogger(__name__)
        
        # Create Ollama client
        self.client = ollama.Client(host=host, **client_kwargs)
        
        # Cache dimensions after first call
        self._dimension = None
        
        # Validate matryoshka dimension
        valid_dims = [512, 256, 128, 64]
        if matryoshka_dim not in valid_dims:
            raise ValueError(f"matryoshka_dim must be one of {valid_dims}")

    def _post_process_embedding(self, embedding: List[float]) -> List[float]:
        """
        Apply Nomic-AI's post-processing steps to the embedding
        
        Args:
            embedding: Raw embedding from model
            
        Returns:
            Processed embedding following Nomic-AI specs
        """
        # Convert to tensor
        emb_tensor = torch.tensor(embedding)
        
        # Apply layer normalization
        emb_tensor = F.layer_norm(emb_tensor.unsqueeze(0), 
                                normalized_shape=(emb_tensor.shape[0],))
        
        # Slice to desired dimension
        emb_tensor = emb_tensor[0, :self.matryoshka_dim]
        
        # L2 normalize
        emb_tensor = F.normalize(emb_tensor.unsqueeze(0), p=2, dim=1)[0]
        
        return emb_tensor.tolist()

    def encode(self, text: str) -> List[float]:
        """Generate embedding for a single text string using Nomic-AI approach"""
        # Add required search_document prefix if not present
        if not text.startswith(('search_document:', 'search_query:', 'clustering:', 'classification:')):
            text = f'search_document: {text}'
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embed(
                    model=self.model_name,
                    input=text
                )
                
                if isinstance(response, dict) and 'embeddings' in response:
                    embedding = response['embeddings'][0]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        return self._post_process_embedding(embedding)
                
            except Exception as e:
                self.logger.error(f"Embedding error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
        
        # Fallback to random vector if all attempts fail
        fallback_vector = np.random.rand(768).tolist()
        return self._post_process_embedding(fallback_vector)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding lists
        """
        # Add required prefix to all texts
        texts = [
            f'search_document: {text}' if not text.startswith(('search_document:', 'search_query:', 'clustering:', 'classification:')) 
            else text 
            for text in texts
        ]
        
        self.logger.debug(f"Batch encoding {len(texts)} texts")
        
        try:
            response = self.client.embed(
                model=self.model_name,
                input=texts
            )
            
            if isinstance(response, dict) and 'embeddings' in response:
                raw_embeddings = response['embeddings']
                if all(isinstance(emb, list) for emb in raw_embeddings):
                    # Process all embeddings
                    return [self._post_process_embedding(emb) for emb in raw_embeddings]
                    
            self.logger.error(f"Failed to extract embeddings from batch response")
            self.logger.debug(f"Response structure: {response}")
                
        except Exception as e:
            self.logger.error(f"Batch embedding error: {e}")
                
        # Fall back to individual processing
        self.logger.debug("Falling back to individual processing")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.encode, texts))
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if self._dimension is None:
            # Call encode with a sample text to determine dimension
            sample_embedding = self.encode("This is a sample text to determine dimension.")
            self._dimension = len(sample_embedding)
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Return the name of the model"""
        return self._model_name