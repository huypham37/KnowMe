#!/usr/bin/env python3
import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List
import traceback

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from TutorLLM.storage.document_processor import DocumentProcessor

def setup_logging(verbose: bool = False):
    """Configure logging"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Reduce noise from other loggers when not in verbose mode
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

def format_results(results: Dict[str, Any]) -> None:
    """Format and print query results"""
    # Debug the actual results structure
    print("\n----- DEBUG: Results Structure -----")
    print(f"Keys in results: {list(results.keys())}")
    for key, value in results.items():
        if isinstance(value, list):
            print(f"- {key}: {len(value)} items")
        else:
            print(f"- {key}: {type(value)}")
    
    # Find the deepest raw_response with documents
    raw_response = results.get('raw_response', {})
    
    # Debug the raw_response
    if isinstance(raw_response, dict):
        print(f"Raw response keys: {list(raw_response.keys())}")
        for key, value in raw_response.items():
            if isinstance(value, list):
                print(f"  - {key}: {len(value)} items")
                if value and key == 'documents' and len(value) > 0:
                    print(f"    First document sample: {str(value[0])[:100]}...")
                if key == 'distances' and value:
                    print(f"    Distances type: {type(value[0])}")
            elif isinstance(value, dict):
                print(f"  - {key}: {type(value)} with keys {list(value.keys())}")
            else:
                print(f"  - {key}: {type(value)}")
    
        # If raw_response has a nested raw_response, examine it too
        if 'raw_response' in raw_response:
            nested_raw = raw_response.get('raw_response', {})
            print(f"  Nested raw_response keys: {list(nested_raw.keys())}")
    
    print("-----------------------------------\n")
    
    # Option 1: Check if results are in 'results' field and not empty
    if 'results' in results and results['results']:
        print("Using 'results' field for data")
        print_formatted_results(results['results'])
        return
    
    # Option 2: Check if there are results in raw_response['results']
    elif 'raw_response' in results and isinstance(results['raw_response'], dict) and 'results' in results['raw_response']:
        nested_results = results['raw_response'].get('results', [])
        if nested_results:
            print("Using nested raw_response['results'] field for data")
            # Try to convert to standard format
            formatted_results = []
            for item in nested_results:
                if isinstance(item, dict):
                    doc = item.get('document', item.get('content', ''))
                    meta = item.get('metadata', {})
                    dist = item.get('distance', 0)
                    formatted_results.append({
                        'text': doc,
                        'metadata': meta,
                        'similarity': 1 - dist if isinstance(dist, (int, float)) else 0
                    })
            
            print_formatted_results(formatted_results)
            return
    
    # Option 3: Check if results are in raw_response directly
    elif 'raw_response' in results:
        raw = results['raw_response']
        if 'documents' in raw and isinstance(raw['documents'], list) and raw['documents']:
            print("Using direct raw_response for data")
            documents = raw.get('documents', [])
            metadatas = raw.get('metadatas', [])
            distances = raw.get('distances', [])
            
            print(f"\nFound {len(documents)} matching documents:\n")
            
            for i, doc in enumerate(documents):
                print_document(i+1, doc, 
                              meta=metadatas[i] if i < len(metadatas) else None,
                              distance=distances[i] if i < len(distances) else None)
            
            return
    
    # Option 4: Check doubly-nested raw_response
    elif 'raw_response' in results and 'raw_response' in results['raw_response']:
        inner_raw = results['raw_response'].get('raw_response', {})
        if 'documents' in inner_raw:
            print("Using doubly-nested raw_response for data")
            
            documents = inner_raw.get('documents', [[]])
            # Handle ChromaDB's nested lists format
            if documents and isinstance(documents[0], list):
                documents = documents[0]
                
            metadatas = inner_raw.get('metadatas', [[]])
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
                
            distances = inner_raw.get('distances', [[]])
            if distances and isinstance(distances[0], list):
                distances = distances[0]
            
            print(f"\nFound {len(documents)} matching documents:\n")
            
            for i, doc in enumerate(documents):
                print_document(i+1, doc, 
                              meta=metadatas[i] if i < len(metadatas) else None,
                              distance=distances[i] if i < len(distances) else None)
            
            return
    
    print("No document content found in results. Check the document_processor.query_similar method.")

def print_document(index: int, doc: Any, meta: dict = None, distance: Any = None):
    """Print a single document result with proper error handling"""
    print(f"Result {index}")
    
    # Handle similarity score
    if distance is not None:
        try:
            # Try to calculate similarity from distance
            if isinstance(distance, (int, float)):
                similarity = 1 - distance
                print(f"Similarity: {similarity:.4f}")
            elif isinstance(distance, list):
                # If distance is a list, take the first element if available
                if distance and isinstance(distance[0], (int, float)):
                    similarity = 1 - distance[0]
                    print(f"Similarity: {similarity:.4f}")
                else:
                    print(f"Distance: {distance} (raw value)")
            else:
                print(f"Distance: {distance} (raw value)")
        except Exception as e:
            print(f"Could not calculate similarity: {str(e)}")
    
    # Handle metadata
    if meta:
        try:
            print(f"Title: {meta.get('doc_title', 'Unknown')}")
            print(f"Source: {os.path.basename(meta.get('source_file', 'Unknown'))}")
            
            # Handle tags
            tags = meta.get('tags', [])
            if isinstance(tags, str) and (tags.startswith('[') and tags.endswith(']')):
                try:
                    tags = json.loads(tags.replace("'", '"'))
                except:
                    pass
            
            if tags:
                print(f"Tags: {', '.join(str(tag) for tag in tags)}")
        except Exception as e:
            print(f"Error processing metadata: {str(e)}")
    
    # Handle document text
    try:
        if isinstance(doc, list):
            # If document is a list, concatenate the text elements
            text = " ".join(str(item) for item in doc)
        else:
            text = str(doc)
        
        # Display document excerpt
        print(f"\nExcerpt: {text[:300]}...\n")
    except Exception as e:
        print(f"Error processing document text: {str(e)}")
    
    print("-" * 50)

def print_formatted_results(results):
    """Print pre-formatted results"""
    if not results:
        print("No results found.")
        return
        
    print(f"\nFound {len(results)} matching documents:\n")
    
    for i, item in enumerate(results):
        try:
            print(f"Result {i+1}")
            
            if 'similarity' in item:
                print(f"Similarity: {item['similarity']:.4f}")
            
            if 'metadata' in item and item['metadata']:
                meta = item['metadata']
                print(f"Title: {meta.get('doc_title', 'Unknown')}")
                print(f"Source: {os.path.basename(meta.get('source_file', 'Unknown'))}")
                
                # Handle tags
                tags = meta.get('tags', [])
                if isinstance(tags, str) and tags.startswith('[') and tags.endswith(']'):
                    try:
                        tags = json.loads(tags.replace("'", '"'))
                    except:
                        pass
                
                if tags:
                    print(f"Tags: {', '.join(str(tag) for tag in tags)}")
            
            if 'text' in item and item['text']:
                print(f"\nExcerpt: {item['text'][:300]}...\n")
            
            print("-" * 50)
        except Exception as e:
            print(f"Error formatting result {i+1}: {str(e)}")

def main():
    """Main function to query the knowledge base"""
    parser = argparse.ArgumentParser(description="Query your document knowledge base")
    parser.add_argument("query", help="The text to search for")
    parser.add_argument("--db-dir", default="./data/vector_db", help="Vector database directory")
    parser.add_argument("--collection", default="um_docs", help="Collection name")
    parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize document processor
        processor = DocumentProcessor(
            docs_directory="",  # Not needed for querying
            db_directory=args.db_dir,
            collection_name=args.collection
        )
        
        # Get stats about the database
        stats = processor.get_collection_stats()
        print(f"Database '{args.collection}' contains {stats['document_count']} document chunks")
        
        # Execute the query
        print(f"\nQuerying for: '{args.query}'")
        print("Processing...")
        
        # Get raw results with debug info
        results = processor.query_similar(args.query, n_results=args.results)
        
        # Format and display results
        format_results(results)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())