from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class RAGModel(ABC):
    """
    Base class for SelfRAG models that provides a high-level interface.
    Subclasses must implement the abstract methods defined here.
    """
    
    @abstractmethod
    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Initialize the SelfRAG model and its components.
        
        Args:
            model_name_or_path: Path to the model or model identifier
            **kwargs: Additional arguments specific to the implementation
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, model_name_or_path: str, **kwargs):
        """
        Alternative constructor to load and initialize the model.
        
        Args:
            model_name_or_path: Path to the model or model identifier
            **kwargs: Additional arguments specific to the implementation
            
        Returns:
            An instance of SelfRAGModel
        """
        pass
    
    
    
    @abstractmethod
    def generate_response(self, input_text: str, **kwargs) -> str:
        """
        Generates content without retrieval augmentation.
        
        Args:
            input_text: The input query or text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    