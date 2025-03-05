#!/usr/bin/env python3
import os
import argparse
import logging
import time
import json
import sys
from typing import Dict, Any, List, Optional


# Ensure TutorLLM is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from TutorLLM.storage.document_processor import DocumentProcessor

class KnowledgeImporter:
    """
    Main class for importing markdown files into the knowledge vector database.
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
        log_level: int = logging.INFO,
        show_progress: bool = True
    ):
        """Initialize the knowledge importer"""
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.docs_directory = docs_directory
        self.db_directory = db_directory
        self.collection_name = collection_name
        self.show_progress = show_progress
        
        self.logger.info(f"Initializing knowledge importer for {docs_directory}")
        self.logger.info(f"Vector database will be stored in {db_directory}")
        
        # Initialize document processor
        self.processor = DocumentProcessor(
            docs_directory=docs_directory,
            db_directory=db_directory,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            matryoshka_dim=matryoshka_dim,
            ollama_host=ollama_host,
            log_level=log_level
        )
    
    def import_documents(self) -> int:
        """
        Import documents from the configured directory into the vector database
        
        Returns:
            Number of document chunks imported
        """
        self.logger.info(f"Starting document import from {self.docs_directory}")
        
        # Check if Ollama server is running by trying to ping it
        try:
            self.processor.embedding_model.client.list()
            self.logger.info("Successfully connected to Ollama server")
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
            self.logger.error("Make sure Ollama is running with: ollama serve")
            return 0
        
        # Start timing the import
        start_time = time.time()
        
        # Process and import documents
        doc_count = self.processor.process_documents(show_progress=self.show_progress)
        
        # Calculate time taken
        elapsed_time = time.time() - start_time
        
        if doc_count > 0:
            self.logger.info(f"Successfully imported {doc_count} document chunks in {elapsed_time:.2f} seconds")
            self.logger.info(f"Average time per chunk: {elapsed_time / doc_count:.4f} seconds")
        else:
            self.logger.warning("No documents were imported")
        
        return doc_count
    
    def verify_import(self, sample_size: int = 3) -> Dict[str, Any]:
        """
        Verify that documents have been successfully imported
        
        Args:
            sample_size: Number of sample documents to retrieve
        
        Returns:
            Dictionary with verification results
        """
        self.logger.info("Verifying document import")
        return self.processor.verify_database_population(sample_size)
    
    def query_knowledge(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge base for information
        
        Args:
            query: The query text
            n_results: Number of results to return
        
        Returns:
            Dictionary with query results
        """
        self.logger.info(f"Querying knowledge base: '{query}'")
        return self.processor.query_similar(query, n_results=n_results)
    
    def print_query_results(self, results: Dict[str, Any]) -> None:
        """
        Print query results in a formatted way
        
        Args:
            results: Query results dictionary
        """
        if not results or "documents" not in results:
            print("No results found.")
            return
            
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        
        print(f"\nFound {len(documents)} matching documents:\n")
        
        for i, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
            print(f"Result {i+1} [Similarity: {1-distance:.4f}]")
            print(f"Title: {meta.get('doc_title', 'Unknown')}")
            print(f"Source: {os.path.basename(meta.get('source_file', 'Unknown'))}")
            print(f"Tags: {', '.join(meta.get('tags', []))}")
            print(f"Excerpt: {doc[:200]}...\n")


def main():
    """Command line interface for the knowledge importer"""
    parser = argparse.ArgumentParser(description='Import markdown documents into a vector database')
    parser.add_argument('--docs-dir', type=str, default="/Users/mac/Documents/UM",
                      help='Directory containing markdown documents')
    parser.add_argument('--db-dir', type=str, default="./data/vector_db",
                      help='Directory to store the vector database')
    parser.add_argument('--collection', type=str, default="um_docs",
                      help='Name of the vector database collection')
    parser.add_argument('--query', type=str,
                      help='Query to search for after importing')
    parser.add_argument('--verify-only', action='store_true',
                      help='Only verify the database without importing documents')
    parser.add_argument('--no-progress', action='store_true',
                      help='Disable progress bars')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    args = parser.parse_args()
    
    # Set log level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Create knowledge importer
    importer = KnowledgeImporter(
        docs_directory=args.docs_dir,
        db_directory=args.db_dir,
        collection_name=args.collection,
        log_level=log_level,
        show_progress=not args.no_progress
    )
    
    # Verify only mode
    if args.verify_only:
        verification = importer.verify_import(sample_size=3)
        doc_count = verification.get("document_count", 0)
        
        print(f"\nVector database contains {doc_count} document chunks")
        
        if doc_count > 0:
            # Display sample documents
            print("\nSample documents:")
            for i, sample in enumerate(verification.get("sample_documents", [])):
                print(f"\n--- Sample {i+1} ---")
                print(f"Text preview: {sample.get('text_preview', '')}")
                print(f"Metadata: {json.dumps(sample.get('metadata', {}), indent=2)}")
            
            # Display available metadata fields
            print("\nAvailable metadata fields:")
            for field, info in verification.get("metadata_fields", {}).items():
                print(f"- {field}: {info}")
        
        # If query argument is provided, run a query
        if args.query:
            results = importer.query_knowledge(args.query)
            importer.print_query_results(results)
        
        return 0
    
    # Check if database already has documents
    verification = importer.verify_import()
    if verification.get("is_populated", False):
        doc_count = verification.get("document_count", 0)
        print(f"\nDatabase already contains {doc_count} document chunks")
        user_input = input("Do you want to import more documents? (y/n): ").lower().strip()
        
        if user_input != 'y':
            print("Exiting without importing documents")
            return 0
    
    # Import documents
    doc_count = importer.import_documents()
    
    # Run verification after import
    if doc_count > 0:
        verification = importer.verify_import()
        print(f"\nVerification after import: Database contains {verification.get('document_count', 0)} document chunks")
    
    # Run query if provided
    if args.query:
        results = importer.query_knowledge(args.query)
        importer.print_query_results(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())