import logging
from typing import List, Dict, Any, Optional, Union
import sys
import os

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import components
from TutorLLM.core.selfrag_model import SelfRAGModel
from TutorLLM.utils.token_extractor import SelfRAGTokenExtractor
from TutorLLM.storage.vector_db import VectorDB
from TutorLLM.core.llama3_model import Llama3RAGModel
from TutorLLM.models.ollama_embedding import OllamaEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RAGOrchestrator:
    """
    Orchestrator class for the Self-RAG system.
    
    Coordinates the workflow between:
    1. SelfRAG model - Initial model that decides if retrieval is needed
    2. TokenExtractor - Extracts retrieval signals from SelfRAG output
    3. VectorDB - Provides relevant context from the knowledge base
    4. Llama3 model - Generates final responses with retrieved context
    """
    
    def __init__(
        self, 
        selfrag_model_path: str = "/Users/mac/mlx-model/selfrag_llama2_7b_mlx",
        llama3_model_name: str = "llama3.2:1b",
        vector_db_path: str = "./storage/vector_db",
        collection_name: str = "knowledge_vault",
        embedding_model_name: str = "nomic-embed-text",
        embedding_host: str = "http://localhost:11434",
        max_retrieval_results: int = 5
    ):
        """
        Initialize the RAG Orchestrator with all required components.
        
        Args:
            selfrag_model_path: Path to the SelfRAG model
            llama3_model_name: Name of the Llama3 model in Ollama
            vector_db_path: Path to the vector database storage directory
            collection_name: Name of the collection within the vector database
            embedding_model_name: Name of the embedding model in Ollama
            embedding_host: Host URL for the Ollama API
            max_retrieval_results: Maximum number of results to retrieve
        """
        self.max_retrieval_results = max_retrieval_results
        
        # Initialize components
        logger.info("Initializing RAG Orchestrator components")
        
        # Initialize SelfRAG model
        logger.info(f"Loading SelfRAG model from {selfrag_model_path}")
        self.selfrag_model = SelfRAGModel(model_path=selfrag_model_path)
        
        # Initialize token extractor
        logger.info("Initializing SelfRAG token extractor")
        self.token_extractor = SelfRAGTokenExtractor()
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = OllamaEmbedding(
            model_name=embedding_model_name,
            host=embedding_host
        )
        
        # Initialize vector database
        logger.info(f"Initializing Vector DB at {vector_db_path}")
        self.vector_db = VectorDB(
            persist_directory=vector_db_path,
            collection_name=collection_name
        )
        
        # Initialize Llama3 model
        logger.info(f"Loading Llama3 model: {llama3_model_name}")
        self.llama3_model = Llama3RAGModel(model_name_or_path=llama3_model_name)
        
        logger.info("RAG Orchestrator initialization complete")
        
    def _retrieve_context(self, query: str) -> List[str]:
        """
        Retrieve relevant context from the vector database.
        
        Args:
            query: The query to search for
            
        Returns:
            List of relevant text passages
        """
        logger.info(f"Retrieving context for query: {query}")
        
        try:
            # Use query_by_text with our embedding model
            results = self.vector_db.query_by_text(
                query_text=query,
                embedding_function=self.embedding_model.encode,
                n_results=self.max_retrieval_results
            )
            
            # Extract text from results
            context_texts = []
            if results and 'documents' in results:
                context_texts = results['documents']
                
            logger.info(f"Retrieved {len(context_texts)} context passages")
            return context_texts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def run(self, query: str, has_prior_evidence: bool = False) -> str:
        """
        Run the complete RAG pipeline.
        
        Args:
            query: The user's query
            has_prior_evidence: Whether prior evidence exists in the conversation
            
        Returns:
            The final generated response
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Get initial response from SelfRAG model to decide if retrieval is needed
            logger.info("Getting initial response from SelfRAG model")
            initial_response = self.selfrag_model.generate(instruction=query)
            
            # Step 2: Analyze the response to see if retrieval is needed
            logger.info("Analyzing response for retrieval signals")
            should_retrieve = self.token_extractor.should_trigger_retrieval(
                response=initial_response, 
                has_prior_evidence=has_prior_evidence
            )
            
            # Step 3: If retrieval is needed, get context from vector DB
            if should_retrieve:
                logger.info("Retrieval triggered - fetching relevant context")
                context_passages = self._retrieve_context(query)
                
                # Step 4a: Generate final response with context
                if context_passages:
                    logger.info(f"Generating response with {len(context_passages)} context passages")
                    final_response = self.llama3_model.generate_with_context(
                        input_text=query,
                        context=context_passages
                    )
                else:
                    logger.warning("No context retrieved, falling back to direct generation")
                    final_response = self.llama3_model.generate_response(query)
            else:
                # Step 4b: Generate response without context if retrieval not needed
                logger.info("Retrieval not triggered - generating direct response")
                final_response = self.llama3_model.generate_response(query)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in RAG orchestrator: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Initialize the orchestrator
    orchestrator = RAGOrchestrator()
    
    # Test the orchestrator
    query = "big endian vs little endian"
    response = orchestrator.run(query)
    print(f"Query: {query}")
    print(f"Response: {response}")