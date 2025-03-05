import time
import logging
import requests
from typing import List, Optional
from ..embedding import EmbeddingModel

class HuggingFaceEmbedding(EmbeddingModel):
    """Embedding model using HuggingFace API or models"""
    
    def __init__(
        self, 
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        use_api: bool = False,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the HuggingFace embedding model
        
        Args:
            model_name_or_path: Model name or path for HuggingFace model
            api_key: HuggingFace API key if using API
            api_url: HuggingFace API URL if using custom endpoint
            use_api: Whether to use the HuggingFace API or load the model locally
            batch_size: Batch size for encoding multiple texts
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Delay between retries in seconds
        """
        self._model_name = model_name_or_path
        self.api_key = api_key
        self.api_url = api_url or f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name_or_path}"
        self.use_api = use_api
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # Cache dimensions after first call
        self._dimension = None
        self._model = None
        
        if not use_api:
            self._load_local_model()
    
    def _load_local_model(self):
        """Load the model locally"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self.logger.info(f"Loaded local HuggingFace model: {self._model_name}")
        except ImportError:
            self.logger.error("Failed to import sentence_transformers. Please install with: pip install sentence-transformers")
            raise
    
    def encode(self, text: str) -> List[float]:
        """Encode a single text using HuggingFace"""
        if self.use_api:
            return self._encode_api(text)
        else:
            return self._encode_local(text)
    
    def _encode_local(self, text: str) -> List[float]:
        """Encode text using local model"""
        if self._model is None:
            self._load_local_model()
            
        try:
            embedding = self._model.encode(text)
            
            # Cache dimension if not already set
            if self._dimension is None and len(embedding) > 0:
                self._dimension = len(embedding)
                
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
        except Exception as e:
            self.logger.error(f"Error encoding with local model: {str(e)}")
            raise
    
    def _encode_api(self, text: str) -> List[float]:
        """Encode text using HuggingFace API"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json={"inputs": text}
                )
                response.raise_for_status()
                
                embedding = response.json()
                # HuggingFace API returns a list of lists for a single text
                if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list):
                    embedding = embedding[0]
                    
                # Cache dimension if not already set
                if self._dimension is None and len(embedding) > 0:
                    self._dimension = len(embedding)
                    
                return embedding
                
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt+1} failed for API: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Failed after {self.max_retries} attempts")
                    raise
        
        return []  # Should not reach here due to raise above
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts using HuggingFace"""
        if self.use_api:
            return self._encode_batch_api(texts)
        else:
            return self._encode_batch_local(texts)
    
    def _encode_batch_local(self, texts: List[str]) -> List[List[float]]:
        """Encode batch of texts using local model"""
        if self._model is None:
            self._load_local_model()
            
        try:
            embeddings = self._model.encode(texts, batch_size=self.batch_size)
            
            # Convert to list format
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            return [list(emb) for emb in embeddings]
            
        except Exception as e:
            self.logger.error(f"Error batch encoding with local model: {str(e)}")
            raise
    
    def _encode_batch_api(self, texts: List[str]) -> List[List[float]]:
        """Encode batch of texts using HuggingFace API"""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        results = []
        # Process in smaller batches to avoid API limitations
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json={"inputs": batch}
                    )
                    response.raise_for_status()
                    
                    batch_embeddings = response.json()
                    results.extend(batch_embeddings)
                    break
                    
                except requests.RequestException as e:
                    self.logger.warning(f"Attempt {attempt+1} failed for batch API: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        self.logger.error(f"Failed batch processing after {self.max_retries} attempts")
                        raise
        
        return results
    
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