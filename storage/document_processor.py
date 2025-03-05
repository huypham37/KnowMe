import os
import logging
import uuid
from typing import List, Dict, Any, Optional
import sys
from tqdm import tqdm

# Ensure TutorLLM is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from TutorLLM.storage.knowledge_vault import KnowledgeVault
from TutorLLM.models.ollama_embedding import OllamaEmbedding
from TutorLLM.storage.vector_db import VectorDB

class DocumentProcessor:
    """
    Processes markdown documents from a directory:
    1. Uses KnowledgeVault to extract and chunk content
    2. Embeds chunks using OllamaEmbedding model
    3. Stores results in VectorDB for retrieval
    """
    
    def __init__(
        self,
        docs_directory: str = "/Users/mac/Documents/UM",
        db_directory: str = "./data/vector_db",
        collection_name: str = "um_docs",
        embedding_model_name: str = "nomic-embed-text",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        matryoshka_dim: int = 512,
        ollama_host: str = "http://localhost:11434",
        log_level: int = logging.INFO
    ):
        """
        Initialize the document processor
        
        Args:
            docs_directory: Directory containing markdown documents to process
            db_directory: Directory to store vector database
            collection_name: Name of the vector database collection
            embedding_model_name: Name of the Ollama embedding model
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between document chunks
            matryoshka_dim: Dimension for embeddings
            ollama_host: Host URL for Ollama API
            log_level: Logging level
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.docs_directory = docs_directory
        self.db_directory = db_directory
        self.collection_name = collection_name
            
        # Create directory for vector database if it doesn't exist
        os.makedirs(db_directory, exist_ok=True)
            
        # Initialize the embedding model
        self.logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embedding_model = OllamaEmbedding(
            model_name=embedding_model_name,
            host=ollama_host,
            matryoshka_dim=matryoshka_dim
        )
            
        # Initialize the knowledge vault
        self.logger.info("Initializing knowledge vault")
        self.knowledge_vault = KnowledgeVault(
            embedding_model=self.embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
            
        # Initialize the vector database
        self.logger.info(f"Initializing vector database: {collection_name}")
        self.vector_db = VectorDB(
            persist_directory=db_directory,
            collection_name=collection_name,
            embedding_dimension=matryoshka_dim,
            create_if_not_exists=True
        )
    
    def process_documents(self, show_progress: bool = True) -> int:
        """
        Process all markdown documents in the configured directory
        
        Args:
            show_progress: Whether to show tqdm progress bars
            
        Returns:
            Number of documents processed and added to the vector database
        """
        self.logger.info(f"Processing documents from: {self.docs_directory}")
        
        try:
            # Find markdown files first to show accurate progress
            markdown_files = self.knowledge_vault.find_markdown_files(self.docs_directory)
            self.logger.info(f"Found {len(markdown_files)} markdown files to process")
            
            if not markdown_files:
                self.logger.warning(f"No markdown files found in {self.docs_directory}")
                return 0
            
            # Process documents manually with progress bar
            all_documents = []
            
            if show_progress:
                file_pbar = tqdm(markdown_files, desc="Processing files", unit="file")
                for file_path in file_pbar:
                    file_pbar.set_postfix({"file": os.path.basename(file_path)})
                    file_documents = self.knowledge_vault.process_file(file_path)
                    all_documents.extend(file_documents)
                    file_pbar.set_postfix({"file": os.path.basename(file_path), 
                                         "chunks": len(file_documents)})
            else:
                # Use the knowledge vault's parallel processing if no progress bar
                all_documents = self.knowledge_vault.process_directory(self.docs_directory)
            
            self.logger.info(f"Processed {len(all_documents)} document chunks")
            
            # Generate IDs for the documents
            doc_ids = [str(uuid.uuid4()) for _ in range(len(all_documents))]
            
            # Add documents to the vector database with progress bar
            if show_progress and all_documents:
                batch_size = 100  # Number of documents to add at once
                batches = [(all_documents[i:i + batch_size], doc_ids[i:i + batch_size]) 
                           for i in range(0, len(all_documents), batch_size)]
                
                with tqdm(total=len(all_documents), desc="Adding to vector DB", unit="doc") as pbar:
                    for docs_batch, ids_batch in batches:
                        self.vector_db.add_documents(docs_batch, ids=ids_batch)
                        pbar.update(len(docs_batch))
            else:
                self.vector_db.add_documents(all_documents, ids=doc_ids)
                
            self.logger.info(f"Added {len(all_documents)} document chunks to vector database")
            
            return len(all_documents)
            
        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}")
            return 0
    
    def query_similar(
        self, 
        query_text: str, 
        n_results: int = 5, 
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database for documents similar to the query text
        
        Args:
            query_text: Text to find similar documents for
            n_results: Number of results to return
            where: Optional filtering criteria for metadata
            
        Returns:
            Dictionary with query results
        """
        self.logger.info(f"Querying for: '{query_text[:50]}...' with {n_results} results")
        
        try:
            # Generate embedding for the query text
            query_embedding = self.embedding_model.encode(query_text)
            
            # Query the vector database using the embedding
            raw_response = self.vector_db.query_similar(
                query_embedding=query_embedding,
                n_results=n_results,
                where=where
            )
            
            # Format results into a standard structure
            results = []
            
            # Try to extract results from different possible response structures
            
            # Case 1: Direct results in raw_response
            if isinstance(raw_response, dict) and "documents" in raw_response:
                documents = raw_response.get("documents", [])
                metadatas = raw_response.get("metadatas", [])
                distances = raw_response.get("distances", [])
                
                for i, doc in enumerate(documents):
                    if i < len(metadatas) and i < len(distances):
                        results.append({
                            "text": doc,
                            "metadata": metadatas[i],
                            "similarity": 1 - distances[i]
                        })
            
            # Case 2: Nested results in raw_response['results']
            elif isinstance(raw_response, dict) and "results" in raw_response:
                nested_results = raw_response.get("results", [])
                if isinstance(nested_results, list) and nested_results:
                    for item in nested_results:
                        if isinstance(item, dict):
                            # If item is already formatted properly
                            if "text" in item and "metadata" in item:
                                results.append(item)
                            # If we need to extract from further nesting
                            elif "document" in item or "content" in item:
                                text = item.get("document", item.get("content", ""))
                                metadata = item.get("metadata", {})
                                distance = item.get("distance", 0)
                                results.append({
                                    "text": text,
                                    "metadata": metadata,
                                    "similarity": 1 - distance if isinstance(distance, (int, float)) else 0.0
                                })
            
            # Case 3: Double-nested raw_response['raw_response']
            elif isinstance(raw_response, dict) and "raw_response" in raw_response:
                inner_raw = raw_response.get("raw_response", {})
                
                # Check for standard format in inner_raw
                if isinstance(inner_raw, dict) and "documents" in inner_raw:
                    documents = inner_raw.get("documents", [])
                    if isinstance(documents, list) and documents:
                        # Handle case where documents is a list of lists
                        if isinstance(documents[0], list):
                            documents = documents[0]
                            
                        metadatas = inner_raw.get("metadatas", [[]])[0] if isinstance(inner_raw.get("metadatas"), list) else []
                        distances = inner_raw.get("distances", [[]])[0] if isinstance(inner_raw.get("distances"), list) else []
                        
                        for i, doc in enumerate(documents):
                            meta = metadatas[i] if i < len(metadatas) else {}
                            dist = distances[i] if i < len(distances) else 0
                            
                            results.append({
                                "text": doc,
                                "metadata": meta,
                                "similarity": 1 - dist if isinstance(dist, (int, float)) else 0.0
                            })
            
            self.logger.info(f"Found {len(results)} results")
            return {
                "results": results,
                "raw_response": raw_response
            }
            
        except Exception as e:
            self.logger.error(f"Error querying vector database: {str(e)}")
            return {"results": [], "error": str(e)}
    
    def get_document_count(self) -> int:
        """Get the count of documents in the vector database"""
        return self.vector_db.count()
    
    def get_document_sample(self, n: int = 5) -> Dict[str, Any]:
        """Get a sample of documents from the vector database"""
        return self.vector_db.peek(n)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection"""
        doc_count = self.vector_db.count()
        metadata_summary = self.vector_db.get_metadata_summary()
        
        return {
            "document_count": doc_count,
            "metadata_fields": metadata_summary.get("fields", {}),
            "collection_name": self.collection_name
        }
        
    def verify_database_population(self, sample_size: int = 3) -> Dict[str, Any]:
        """
        Verify that the vector database is populated with documents and return
        information about the population status.
        
        Args:
            sample_size: Number of sample documents to retrieve
            
        Returns:
            Dictionary with verification results
        """
        self.logger.info("Verifying vector database population")
        
        verification_results = {
            "is_populated": False,
            "document_count": 0,
            "sample_documents": [],
            "metadata_fields": {},
            "collection_name": self.collection_name
        }
        
        try:
            # Check document count
            doc_count = self.get_document_count()
            verification_results["document_count"] = doc_count
            verification_results["is_populated"] = doc_count > 0
            
            self.logger.info(f"Database contains {doc_count} documents")
            
            if doc_count > 0:
                # Get metadata summary
                stats = self.get_collection_stats()
                verification_results["metadata_fields"] = stats.get("metadata_fields", {})
                
                # Get sample documents
                sample = self.get_document_sample(sample_size)
                if sample and "documents" in sample:
                    # Process sample for readability
                    for i, doc in enumerate(sample["documents"][:sample_size]):
                        sample_meta = sample["metadatas"][i] if "metadatas" in sample else {}
                        doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
                        
                        verification_results["sample_documents"].append({
                            "text_preview": doc_preview,
                            "metadata": sample_meta
                        })
                
                self.logger.info(f"Retrieved {len(verification_results['sample_documents'])} sample documents")
            
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Error verifying database population: {str(e)}")
            verification_results["error"] = str(e)
            return verification_results