import logging
import os
import sys
from typing import List, Dict, Any, Optional, Union
import ollama
from ollama import Client, ResponseError

# Add parent directory to Python path to find imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the RAGModel class
from TutorLLM.core.rag_model import RAGModel

logger = logging.getLogger(__name__)

class Llama3RAGModel(RAGModel):
    """
    Implementation of RAGModel using the official Ollama Python library for Llama3 models.
    Focuses on generation capabilities without handling retrieval decisions.
    """
    
    def __init__(self, model_name_or_path: str, host: str = "http://localhost:11434", **kwargs):
        """
        Initialize the Llama3 RAG model using Ollama.
        
        Args:
            model_name_or_path: Name of the Ollama model (e.g., "llama3")
            host: Host URL for the Ollama API
            **kwargs: Additional arguments for model configuration
        """
        self.model_name = model_name_or_path
        self.client = Client(host=host)
        self.generation_params = {}
        
        # Check if the model is available
        self._check_model_availability()
        
    @classmethod
    def load(cls, model_name_or_path: str, **kwargs):
        """
        Load the Llama3 model from Ollama.
        
        Args:
            model_name_or_path: Name of the Ollama model
            **kwargs: Additional arguments for model configuration
            
        Returns:
            An instance of Llama3RAGModel
        """
        return cls(model_name_or_path, **kwargs)
    
    def _check_model_availability(self):
        """
        Check if the specified model is available in Ollama.
        If not, log a warning.
        """
        try:
            # Fix the 'name' KeyError by adjusting how we access the model list
            models_response = self.client.list()
            # Print the response structure for debugging
            logger.debug(f"Models response structure: {models_response}")
            
            # Handle both possible response structures
            if isinstance(models_response, dict) and 'models' in models_response:
                available_models = []
                for model in models_response['models']:
                    if isinstance(model, dict) and 'name' in model:
                        available_models.append(model['name'])
            else:
                # Fall back to just checking if our model name appears in the response
                available_models = []
                logger.warning(f"Unexpected model list response format: {models_response}")
                
            if not available_models or self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} might not be available. Consider pulling it.")
                logger.info(f"You can pull the model using: ollama.pull('{self.model_name}')")
        except ResponseError as e:
            logger.warning(f"Error checking model availability: {str(e)}")
    
    def predict_retrieval_decision(self, input_text: str) -> bool:
        """
        Determines if retrieval is needed for the given input.
        
        Args:
            input_text: The input query or text
            
        Returns:
            Boolean indicating whether retrieval is needed
        """
        # Implement a simple heuristic - always use retrieval for simplicity
        # In a real implementation, you might use the model to decide if retrieval is needed
        logger.info("Predict retrieval decision: Always retrieving for now")
        return True
            
    def generate_response(self, input_text: str, **kwargs) -> str:
        """
        Generates content without retrieval augmentation.
        
        Args:
            input_text: The input query or text
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Update generation parameters with any provided kwargs
        gen_params = self.generation_params.copy()
        gen_params.update(kwargs)
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": input_text}],
                **gen_params
            )
            return response["message"]["content"]
            
        except ResponseError as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_context(self, input_text: str, context: Union[str, List[str]], **kwargs) -> str:
        """
        Generates content with retrieval augmentation from provided context.
        
        Args:
            input_text: The input query or text
            context: Retrieved context information as string or list of strings
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response with context integration
        """
        # Update generation parameters with any provided kwargs
        gen_params = self.generation_params.copy()
        gen_params.update(kwargs)
        
        # Format the context
        if isinstance(context, list):
            formatted_context = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(context)])
        else:
            formatted_context = f"Context:\n{context}"
        
        system_prompt = (
            "You are a helpful AI assistant. Use the provided context to answer the user's question. "
            "If the context doesn't contain relevant information, use your knowledge to provide the best response."
        )
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{formatted_context}\n\nQuestion: {input_text}"}
                ],
                **gen_params
            )
            return response["message"]["content"]
            
        except ResponseError as e:
            logger.error(f"Error in generate_with_context: {str(e)}")
            return f"Error generating response with context: {str(e)}"
    
    def generate_with_streaming(self, input_text: str, context: Optional[Union[str, List[str]]] = None, **kwargs):
        """
        Generates content with streaming capability, optionally using context.
        This is an additional method not required by the base class but provides
        streaming functionality that's useful for interactive applications.
        
        Args:
            input_text: The input query or text
            context: Optional retrieved context information
            **kwargs: Additional generation parameters
            
        Returns:
            A generator yielding response chunks
        """
        # Update generation parameters with any provided kwargs
        gen_params = self.generation_params.copy()
        gen_params.update(kwargs)
        gen_params["stream"] = True
        
        messages = []
        
        if context:
            # Format the context
            if isinstance(context, list):
                formatted_context = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(context)])
            else:
                formatted_context = f"Context:\n{context}"
            
            system_prompt = (
                "You are a helpful AI assistant. Use the provided context to answer the user's question. "
                "If the context doesn't contain relevant information, use your knowledge to provide the best response."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{formatted_context}\n\nQuestion: {input_text}"}
            ]
        else:
            messages = [{"role": "user", "content": input_text}]
        
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                **gen_params
            )
            
            for chunk in stream:
                yield chunk["message"]["content"]
                
        except ResponseError as e:
            logger.error(f"Error in generate_with_streaming: {str(e)}")
            yield f"Error generating streaming response: {str(e)}"


# Test function that can be run directly when the file is executed
def main():
    # Configure logging - set to DEBUG to see more details
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the model
    model_name = "llama3.2:1b"
    logger.info(f"Initializing Llama3RAGModel with model: {model_name}")
    
    try:
        # Try to pull the model if it doesn't exist
        logger.info(f"Attempting to pull model {model_name} if not already available")
        try:
            ollama.pull(model_name)
        except Exception as e:
            logger.warning(f"Model pull attempt resulted in: {str(e)}")
        
        model = Llama3RAGModel(model_name)
        
        # Test 1: Simple generation
        test_query = "What is the capital of France?"
        logger.info(f"Testing generate_response with query: '{test_query}'")
        response = model.generate_response(test_query)  # Limit token generation
        logger.info(f"Response: {response}")
        
        # Test 2: Generation with context
        test_context = "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
        logger.info(f"Testing generate_with_context with the same query and provided context")
        response_with_context = model.generate_with_context(test_query, test_context)
        logger.info(f"Response with context: {response_with_context}")
        
        # Test 3: Streaming generation - optional, may take longer
        test_streaming = input("Run streaming tests? (y/n): ").lower().strip() == 'y'
        if test_streaming:
            logger.info(f"Testing generate_with_streaming with query: 'Tell me about quantum physics'")
            print("\nStreaming response:")
            for chunk in model.generate_with_streaming("Tell me about quantum physics", 
                                                    
                                                    ): # the temp parameter and max token is left for later
                print(chunk, end="", flush=True)
            print("\n")
            
            # Test 4: Streaming with context
            logger.info(f"Testing streaming with context")
            quantum_context = [
                "Quantum physics is a branch of physics that deals with the behavior of matter at atomic and subatomic scales.",
                "Key concepts include wave-particle duality, quantization of energy, and the uncertainty principle."
            ]
            print("\nStreaming response with context:")
            for chunk in model.generate_with_streaming("Explain the uncertainty principle", 
                                                    context=quantum_context,
                                                    
                                                    ): # the temp parameter and max token is left for later 
                print(chunk, end="", flush=True)
            print("\n")
    
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()