from typing import Optional
from ..embedding import EmbeddingService
from .ollama_embedding import OllamaEmbedding
from .huggingface_embedding import HuggingFaceEmbedding

def create_embedding_service(
    primary_model_type: str = "ollama",
    primary_model_name: str = "nomic-embed-text",
    fallback_model_type: str = "huggingface",
    fallback_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    hf_api_key: Optional[str] = None,
    use_api: bool = False
) -> EmbeddingService:
    """
    Create an EmbeddingService with primary and fallback models
    
    Args:
        primary_model_type: Type of primary model ('ollama' or 'huggingface')
        primary_model_name: Name of primary model
        fallback_model_type: Type of fallback model ('ollama' or 'huggingface')
        fallback_model_name: Name of fallback model
        use_cache: Whether to use caching
        cache_dir: Directory for cache storage
        hf_api_key: HuggingFace API key if using API
        use_api: Whether to use the HuggingFace API
        
    Returns:
        Configured EmbeddingService
    """
    # Create primary model
    if primary_model_type.lower() == "ollama":
        primary_model = OllamaEmbedding(model_name=primary_model_name)
    else:  # huggingface
        primary_model = HuggingFaceEmbedding(
            model_name_or_path=primary_model_name, 
            api_key=hf_api_key,
            use_api=use_api
        )
    
    # Create fallback model if different from primary
    fallback_model = None
    if fallback_model_type and fallback_model_name:
        if fallback_model_type.lower() == "ollama":
            fallback_model = OllamaEmbedding(model_name=fallback_model_name)
        else:  # huggingface
            fallback_model = HuggingFaceEmbedding(
                model_name_or_path=fallback_model_name,
                api_key=hf_api_key,
                use_api=use_api
            )
    
    # Create and return the service
    return EmbeddingService(
        primary_model=primary_model,
        fallback_model=fallback_model,
        use_cache=use_cache,
        cache_dir=cache_dir
    )