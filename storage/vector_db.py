import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import uuid
import json
import sys
from datetime import datetime  # Add this import for create_backup method
# Ensure TutorLLM is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    logging.error("ChromaDB not installed. Please install with: pip install chromadb")
    raise

class VectorDB:
    """
    Vector database manager for storing and retrieving document embeddings.
    Uses ChromaDB for vector similarity search.
    """
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = "knowledge_vault",
        embedding_dimension: int = 384,  # Default dimension for nomic-embed-text
        distance_metric: str = "cosine",
        create_if_not_exists: bool = True
    ):
        """
        Initialize the vector database
        
        Args:
            persist_directory: Directory to persist the database, if None uses in-memory DB
            collection_name: Name of the collection to store embeddings
            embedding_dimension: Dimension of the embeddings
            distance_metric: Distance metric for similarity search (cosine, l2, or ip)
            create_if_not_exists: Create the collection if it doesn't exist
        """
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        
        # Configure and initialize ChromaDB client
        client_settings = Settings()
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.logger.info(f"Using persistent ChromaDB at: {persist_directory}")
            self.client = chromadb.PersistentClient(path=persist_directory, settings=client_settings)
        else:
            self.logger.info("Using in-memory ChromaDB")
            self.client = chromadb.Client(settings=client_settings)
            
        # Get or create the collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # We'll provide embeddings directly
            )
            self.logger.info(f"Connected to existing collection: {collection_name}")
        except (ValueError, chromadb.errors.InvalidCollectionException) as e:
            if create_if_not_exists:
                self.logger.info(f"Creating new collection: {collection_name}")
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=None,  # We'll provide embeddings directly
                    metadata={"hnsw:space": distance_metric}
                )
            else:
                self.logger.error(f"Collection '{collection_name}' does not exist and create_if_not_exists=False")
                raise
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents with their embeddings to the vector database
        
        Args:
            documents: List of document dictionaries with 'text', 'embedding', and 'metadata'
            ids: Optional list of IDs for the documents, if None, UUIDs will be generated
            batch_size: Batch size for adding documents
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
            
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        elif len(ids) != len(documents):
            raise ValueError(f"Length of ids ({len(ids)}) must match length of documents ({len(documents)})")
            
        # Prepare data for ChromaDB
        texts = [doc.get('text', '') for doc in documents]
        embeddings = [doc.get('embedding', []) for doc in documents]
        
        # Process metadatas: convert any non-primitive types to strings
        metadatas = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            processed_metadata = {}
            
            for key, value in metadata.items():
                # Convert lists and other non-primitive types to strings
                if isinstance(value, (list, dict)):
                    processed_metadata[key] = json.dumps(value)
                else:
                    processed_metadata[key] = value
                    
            metadatas.append(processed_metadata)
        
        # Validate embeddings
        for i, emb in enumerate(embeddings):
            if not emb or len(emb) != self.embedding_dimension:
                self.logger.warning(
                    f"Invalid embedding for document {i} (ID: {ids[i]}): "
                    f"expected dimension {self.embedding_dimension}, got {len(emb) if emb else 0}"
                )
                return ids[:i]  # Return IDs of successfully added documents
                
        # Add documents in batches to avoid memory issues or API limits
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch_ids = ids[i:batch_end]
            batch_texts = texts[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                self.logger.debug(f"Added batch of {len(batch_ids)} documents")
            except Exception as e:
                self.logger.error(f"Error adding batch to ChromaDB: {str(e)}")
                return ids[:i]  # Return IDs of successfully added documents
                
        return ids
    
    def query_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Any]:
        """
        Query for similar documents based on embedding
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional filtering criteria for metadata
            where_document: Optional filtering criteria for document content
            include: What to include in the results
            
        Returns:
            Dictionary with query results
        """
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} doesn't match expected dimension {self.embedding_dimension}"
            )
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
            # Format the results as a more convenient structure
            formatted_results = []
            for i in range(len(results.get('ids', [[]])[0])):
                item = {
                    'id': results['ids'][0][i] if 'ids' in results and results['ids'] else None
                }
                
                if 'documents' in results and results['documents'] and i < len(results['documents'][0]):
                    item['text'] = results['documents'][0][i]
                    
                if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas'][0]):
                    item['metadata'] = results['metadatas'][0][i]
                    
                if 'distances' in results and results['distances'] and i < len(results['distances'][0]):
                    item['distance'] = results['distances'][0][i]
                    
                formatted_results.append(item)
                
            return {
                'results': formatted_results,
                'raw_response': results
            }
            
        except Exception as e:
            self.logger.error(f"Error querying ChromaDB: {str(e)}")
            return {'results': [], 'raw_response': None, 'error': str(e)}
    
    def query_by_text(
        self,
        query_text: str,
        embedding_function: callable,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> Dict[str, Any]:
        """
        Query for similar documents based on text
        
        Args:
            query_text: Query text to embed
            embedding_function: Function to convert text to embedding
            n_results: Number of results to return
            where: Optional filtering criteria for metadata
            where_document: Optional filtering criteria for document content
            include: What to include in the results
            
        Returns:
            Dictionary with query results
        """
        try:
            # Get embedding for query text
            query_embedding = embedding_function(query_text)
            
            # Use the embedding to query
            return self.query_similar(
                query_embedding=query_embedding,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
        except Exception as e:
            self.logger.error(f"Error in query_by_text: {str(e)}")
            return {'results': [], 'raw_response': None, 'error': str(e)}
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Check if the document exists
            if not result or 'ids' not in result or len(result['ids']) == 0:
                return None
            
            # Get the first document's data
            doc_index = 0
            
            # Process metadata if available
            metadata = {}
            if ('metadatas' in result and result['metadatas'] is not None
                    and len(result['metadatas']) > doc_index):
                raw_metadata = result['metadatas'][doc_index]
                
                # Process each metadata field
                for key, value in raw_metadata.items():
                    # Try to parse JSON strings back to their original form
                    if isinstance(value, str):
                        try:
                            metadata[key] = json.loads(value)
                        except json.JSONDecodeError:
                            metadata[key] = value
                    else:
                        metadata[key] = value
            
            # Get embedding
            embedding = None
            if ('embeddings' in result and result['embeddings'] is not None
                    and len(result['embeddings']) > doc_index):
                raw_embedding = result['embeddings'][doc_index]
                
                # Convert numpy array to list if needed
                if hasattr(raw_embedding, 'tolist'):
                    embedding = raw_embedding.tolist()
                else:
                    embedding = raw_embedding
            
            # Get text
            text = None
            if ('documents' in result and result['documents'] is not None
                    and len(result['documents']) > doc_index):
                text = result['documents'][doc_index]
            
            return {
                'id': document_id,
                'text': text,
                'embedding': embedding,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document {document_id}: {str(e)}")
            return None
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not ids:
            return True
            
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def update_document(
        self,
        document_id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing document
        
        Args:
            document_id: ID of the document to update
            text: Optional new text
            embedding: Optional new embedding
            metadata: Optional new metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Get current document to update only specified fields
        current_doc = self.get_document(document_id)
        if not current_doc:
            self.logger.error(f"Document {document_id} not found for update")
            return False
            
        # Prepare update values
        update_text = text if text is not None else current_doc.get('text')
        update_embedding = embedding if embedding is not None else current_doc.get('embedding', [])
        
        # For metadata, merge instead of replace if both exist
        if metadata and current_doc.get('metadata'):
            update_metadata = {**current_doc['metadata'], **metadata}
        else:
            update_metadata = metadata if metadata is not None else current_doc.get('metadata', {})
        
        # Process metadata to ensure all values are of allowed types (str, int, float, bool)
        processed_metadata = {}
        for key, value in update_metadata.items():
            # Convert lists and other non-primitive types to strings
            if isinstance(value, (list, dict)):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = value
        
        try:
            self.collection.update(
                ids=[document_id],
                documents=[update_text],
                embeddings=[update_embedding],
                metadatas=[processed_metadata]
            )
            self.logger.info(f"Updated document {document_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating document {document_id}: {str(e)}")
            return False
    
    def count(self) -> int:
        """Get the count of documents in the collection"""
        try:
            # In newer ChromaDB versions
            if hasattr(self.collection, 'count'):
                return self.collection.count()
            
            # Fallback for older versions
            result = self.collection.get(limit=1)
            return len(result.get('ids', []))
        except Exception as e:
            self.logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    def get_all_ids(self) -> List[str]:
        """Get all document IDs in the collection"""
        try:
            # This could be resource-intensive for large collections
            result = self.collection.get(include=["documents"]) 
            return result.get('ids', [])
        except Exception as e:
            self.logger.error(f"Error getting all document IDs: {str(e)}")
            return []
            
    def peek(self, n: int = 5) -> Dict[str, Any]:
        """
        Get a sample of documents from the collection
        
        Args:
            n: Number of documents to retrieve
            
        Returns:
            Dictionary with document data
        """
        try:
            result = self.collection.get(
                limit=n,
                include=["documents", "metadatas"]
            )
            
            samples = []
            for i in range(len(result.get('ids', []))):
                sample = {
                    'id': result['ids'][i],
                    'text': result['documents'][i] if 'documents' in result else None
                }
                
                if 'metadatas' in result and result['metadatas'] and i < len(result['metadatas']):
                    sample['metadata'] = result['metadatas'][i]
                    
                samples.append(sample)
                
            return {'samples': samples, 'count': len(samples)}
            
        except Exception as e:
            self.logger.error(f"Error peeking at collection: {str(e)}")
            return {'samples': [], 'count': 0, 'error': str(e)}
            
    def get_metadata_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metadata fields and their distribution
        
        Returns:
            Dictionary with metadata summary statistics
        """
        try:
            # Get all metadata
            result = self.collection.get(include=["metadatas"])
            metadatas = result.get('metadatas', [])
            
            if not metadatas:
                return {'fields': {}, 'count': 0}
                
            # Analyze metadata fields
            fields = {}
            for meta in metadatas:
                if not meta:
                    continue
                    
                for key, value in meta.items():
                    if key not in fields:
                        fields[key] = {
                            'count': 0,
                            'types': {},
                            'sample_values': []
                        }
                        
                    fields[key]['count'] += 1
                    value_type = type(value).__name__
                    
                    if value_type not in fields[key]['types']:
                        fields[key]['types'][value_type] = 0
                        
                    fields[key]['types'][value_type] += 1
                    
                    # Store a few sample values
                    if len(fields[key]['sample_values']) < 5 and value not in fields[key]['sample_values']:
                        # Convert to string for complex types
                        if isinstance(value, (dict, list)):
                            fields[key]['sample_values'].append(str(value))
                        else:
                            fields[key]['sample_values'].append(value)
            
            return {'fields': fields, 'count': len(metadatas)}
            
        except Exception as e:
            self.logger.error(f"Error getting metadata summary: {str(e)}")
            return {'fields': {}, 'count': 0, 'error': str(e)}
            
    def create_backup(self, backup_path: str) -> bool:
        """
        Create a backup of the current collection
        
        Args:
            backup_path: Directory path to store the backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(backup_path, exist_ok=True)
            
            # Get all documents
            result = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Convert NumPy arrays to lists for JSON serialization
            embeddings = []
            if 'embeddings' in result and result['embeddings'] is not None:
                for emb in result['embeddings']:
                    if hasattr(emb, 'tolist'):
                        embeddings.append(emb.tolist())
                    else:
                        embeddings.append(emb)
            
            # Prepare backup data
            backup = {
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_dimension,
                'distance_metric': self.distance_metric,
                'count': len(result.get('ids', [])),
                'ids': result.get('ids', []),
                'documents': result.get('documents', []),
                'metadatas': result.get('metadatas', []),
                'embeddings': embeddings,
                'timestamp': str(datetime.now())
            }
            
            # Save backup file
            backup_file = os.path.join(backup_path, f"{self.collection_name}_backup.json")
            with open(backup_file, 'w') as f:
                json.dump(backup, f)
                
            self.logger.info(f"Created backup at {backup_file} with {backup['count']} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False
            
    def restore_from_backup(self, backup_file: str) -> bool:
        """
        Restore collection from backup file
        
        Args:
            backup_file: Path to backup JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load backup data
            with open(backup_file, 'r') as f:
                backup = json.load(f)
                
            # Validate backup data
            required_fields = ['ids', 'documents', 'metadatas', 'embeddings']
            for field in required_fields:
                if field not in backup or not backup[field]:
                    raise ValueError(f"Invalid backup file: missing or empty '{field}' field")
                    
            # Delete all existing documents
            existing_ids = self.get_all_ids()
            if existing_ids:
                self.delete_documents(existing_ids)
                
            # Add documents from backup
            ids = backup['ids']
            documents = backup['documents']
            embeddings = backup['embeddings']
            metadatas = backup['metadatas']
            
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.logger.info(f"Restored {len(ids)} documents from backup {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring from backup: {str(e)}")
            return False