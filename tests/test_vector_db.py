import unittest
import tempfile
import os
import shutil
import logging
import numpy as np
from typing import List, Dict, Any

from TutorLLM.storage.vector_db import VectorDB

class TestVectorDB(unittest.TestCase):
    """Test suite for the VectorDB class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(levelname)s:%(name)s:%(message)s'
        )
        cls.logger = logging.getLogger("vector_db_test")
        
        # Define constants for testing
        cls.TEST_DIMENSION = 128
        cls.COLLECTION_NAME = "test_collection"

    def setUp(self):
        """Set up before each test"""
        # Create a temporary directory for the database
        self.test_dir = tempfile.mkdtemp()
        self.logger.info(f"Created test directory: {self.test_dir}")
        
        # Create test VectorDB instance with create_if_not_exists=True
        self.db = VectorDB(
            persist_directory=self.test_dir,
            collection_name=self.COLLECTION_NAME,
            embedding_dimension=self.TEST_DIMENSION,
            create_if_not_exists=True  # Ensure collection is created
        )
        
        # Generate sample documents for testing
        self.sample_docs = self.generate_sample_documents(10)

    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            self.logger.info(f"Removed test directory: {self.test_dir}")

    def generate_sample_documents(self, n: int) -> List[Dict[str, Any]]:
        """Generate sample documents with embeddings for testing"""
        docs = []
        for i in range(n):
            # Create a deterministic but varied embedding
            np.random.seed(i)
            embedding = np.random.rand(self.TEST_DIMENSION).tolist()
            
            docs.append({
                'text': f"Sample document {i} for testing vector database functionality",
                'embedding': embedding,
                'metadata': {
                    'source': f"test_doc_{i}",
                    'category': f"category_{i % 3}",
                    'priority': i % 5,
                    'tags': [f"tag_{j}" for j in range(i % 3 + 1)]
                }
            })
        
        return docs

    def test_initialization(self):
        """Test VectorDB initialization and collection creation"""
        # Check that the collection was created
        self.assertIsNotNone(self.db.collection)
        
        # Check configuration
        self.assertEqual(self.db.collection_name, self.COLLECTION_NAME)
        self.assertEqual(self.db.embedding_dimension, self.TEST_DIMENSION)
        self.assertEqual(self.db.distance_metric, "cosine")
        
        # Test with in-memory client
        in_memory_db = VectorDB(
            persist_directory=None,
            collection_name="memory_test",
            embedding_dimension=self.TEST_DIMENSION
        )
        self.assertIsNotNone(in_memory_db.collection)

    def test_add_documents(self):
        """Test adding documents to the vector database"""
        # Add sample documents
        ids = self.db.add_documents(self.sample_docs)
        
        # Check that IDs were returned
        self.assertEqual(len(ids), len(self.sample_docs))
        
        # Check document count
        self.assertEqual(self.db.count(), len(self.sample_docs))
        
        # Add documents with predefined IDs
        custom_ids = [f"custom_id_{i}" for i in range(5)]
        custom_docs = self.generate_sample_documents(5)
        added_ids = self.db.add_documents(custom_docs, ids=custom_ids)
        
        # Check that the correct IDs were used
        self.assertEqual(added_ids, custom_ids)
        
        # Check total count
        self.assertEqual(self.db.count(), len(self.sample_docs) + len(custom_docs))

    def test_query_similar(self):
        """Test querying for similar documents"""
        # Add sample documents
        ids = self.db.add_documents(self.sample_docs)
        
        # Query using the embedding from the first document
        query_embedding = self.sample_docs[0]['embedding']
        results = self.db.query_similar(
            query_embedding=query_embedding,
            n_results=3
        )
        
        # Check that results were returned
        self.assertIn('results', results)
        self.assertEqual(len(results['results']), 3)
        
        # The most similar document should be the original document
        most_similar = results['results'][0]
        self.assertEqual(most_similar['id'], ids[0])
        
        # Test with metadata filter
        filtered_results = self.db.query_similar(
            query_embedding=query_embedding,
            n_results=3,
            where={"category": "category_0"}
        )
        
        # Check that all results match the filter
        for item in filtered_results['results']:
            self.assertEqual(item['metadata']['category'], "category_0")

    def test_get_document(self):
        """Test retrieving a specific document by ID"""
        # Add sample documents
        ids = self.db.add_documents(self.sample_docs)
        
        # Get a specific document
        doc = self.db.get_document(ids[2])
        
        # Check that the document was retrieved
        self.assertIsNotNone(doc)
        self.assertEqual(doc['id'], ids[2])
        self.assertEqual(doc['metadata']['source'], self.sample_docs[2]['metadata']['source'])
        
        # Test with non-existent ID
        non_existent = self.db.get_document("non_existent_id")
        self.assertIsNone(non_existent)

    def test_delete_documents(self):
        """Test deleting documents"""
        # Add sample documents
        ids = self.db.add_documents(self.sample_docs)
        initial_count = self.db.count()
        
        # Delete a few documents
        to_delete = ids[:3]
        success = self.db.delete_documents(to_delete)
        
        # Check deletion was successful
        self.assertTrue(success)
        self.assertEqual(self.db.count(), initial_count - len(to_delete))
        
        # Check documents no longer exist
        for doc_id in to_delete:
            self.assertIsNone(self.db.get_document(doc_id))

    def test_update_document(self):
        """Test updating an existing document"""
        # Add sample documents
        ids = self.db.add_documents(self.sample_docs)
        
        # Update a document's text and metadata
        doc_id = ids[1]
        new_text = "Updated text for testing"
        new_metadata = {"category": "updated", "new_field": "new_value"}
        
        success = self.db.update_document(
            document_id=doc_id,
            text=new_text,
            metadata=new_metadata
        )
        
        # Check update was successful
        self.assertTrue(success)
        
        # Retrieve updated document
        updated_doc = self.db.get_document(doc_id)
        
        # Check updates were applied
        self.assertEqual(updated_doc['text'], new_text)
        self.assertEqual(updated_doc['metadata']['category'], "updated")
        self.assertEqual(updated_doc['metadata']['new_field'], "new_value")
        
        # The original fields should still be present
        self.assertEqual(updated_doc['metadata']['source'], self.sample_docs[1]['metadata']['source'])

    def test_invalid_embedding_dimension(self):
        """Test handling of invalid embedding dimensions"""
        # Create document with wrong embedding dimension
        invalid_doc = {
            'text': "Invalid document",
            'embedding': [0.1, 0.2, 0.3],  # Wrong dimension
            'metadata': {'source': "invalid"}
        }
        
        # Adding should fail or at least warn
        ids = self.db.add_documents([invalid_doc])
        self.assertEqual(len(ids), 0)
        
        # Querying with wrong dimension should raise ValueError
        with self.assertRaises(ValueError):
            self.db.query_similar([0.1, 0.2, 0.3])

    def test_query_by_text(self):
        """Test querying by text using an embedding function"""
        # Add sample documents
        self.db.add_documents(self.sample_docs)
        
        # Mock embedding function that returns a consistent embedding
        def mock_embedding_function(text: str) -> List[float]:
            # Simple deterministic function based on text length
            seed = len(text)
            np.random.seed(seed)
            return np.random.rand(self.TEST_DIMENSION).tolist()
        
        # Query by text
        results = self.db.query_by_text(
            query_text="Sample query text",
            embedding_function=mock_embedding_function,
            n_results=5
        )
        
        # Check that results were returned
        self.assertIn('results', results)
        self.assertEqual(len(results['results']), 5)

    def test_metadata_filtering(self):
        """Test filtering by metadata in queries"""
        # Add sample documents
        self.db.add_documents(self.sample_docs)
        
        # Query with specific metadata filter
        query_embedding = self.sample_docs[0]['embedding']
        
        # Filter by priority
        priority_results = self.db.query_similar(
            query_embedding=query_embedding,
            n_results=10,
            where={"priority": 2}
        )
        
        # Check that all results have the correct priority
        for item in priority_results['results']:
            self.assertEqual(item['metadata']['priority'], 2)
        
        # Filter by category
        category_results = self.db.query_similar(
            query_embedding=query_embedding,
            n_results=10,
            where={"category": "category_1"}
        )
        
        # Check that all results have the correct category
        for item in category_results['results']:
            self.assertEqual(item['metadata']['category'], "category_1")

    def test_peek_function(self):
        """Test the peek function for sampling documents"""
        # Add sample documents
        self.db.add_documents(self.sample_docs)
        
        # Peek at 3 documents
        peek_results = self.db.peek(3)
        
        # Check that the correct number of samples was returned
        self.assertEqual(peek_results['count'], 3)
        self.assertEqual(len(peek_results['samples']), 3)
        
        # Each sample should have id, text, and metadata
        for sample in peek_results['samples']:
            self.assertIn('id', sample)
            self.assertIn('text', sample)
            self.assertIn('metadata', sample)

    def test_metadata_summary(self):
        """Test metadata summary generation"""
        # Add sample documents
        self.db.add_documents(self.sample_docs)
        
        # Get metadata summary
        summary = self.db.get_metadata_summary()
        
        # Check summary structure
        self.assertIn('fields', summary)
        self.assertIn('count', summary)
        self.assertEqual(summary['count'], len(self.sample_docs))
        
        # Check that known fields are in the summary
        self.assertIn('source', summary['fields'])
        self.assertIn('category', summary['fields'])
        self.assertIn('priority', summary['fields'])
        self.assertIn('tags', summary['fields'])
        
        # Check field details
        source_field = summary['fields']['source']
        self.assertEqual(source_field['count'], len(self.sample_docs))
        self.assertIn('types', source_field)
        self.assertIn('sample_values', source_field)

    def test_backup_and_restore(self):
        """Test backup and restore functionality"""
        # Add sample documents
        original_ids = self.db.add_documents(self.sample_docs)
        
        # Create a backup directory
        backup_dir = os.path.join(self.test_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup
        success = self.db.create_backup(backup_dir)
        self.assertTrue(success)
        
        # Find the backup file
        backup_file = os.path.join(backup_dir, f"{self.COLLECTION_NAME}_backup.json")
        self.assertTrue(os.path.exists(backup_file))
        
        # Create a new VectorDB for restoration
        restore_dir = os.path.join(self.test_dir, "restore")
        os.makedirs(restore_dir, exist_ok=True)
        
        restore_db = VectorDB(
            persist_directory=restore_dir,
            collection_name=self.COLLECTION_NAME,
            embedding_dimension=self.TEST_DIMENSION
        )
        
        # Restore from backup
        success = restore_db.restore_from_backup(backup_file)
        self.assertTrue(success)
        
        # Check document count
        self.assertEqual(restore_db.count(), len(self.sample_docs))
        
        # Check that all original IDs are present
        restored_ids = restore_db.get_all_ids()
        self.assertEqual(set(restored_ids), set(original_ids))
        
        # Compare original and restored documents
        for doc_id in original_ids:
            original_doc = self.db.get_document(doc_id)
            restored_doc = restore_db.get_document(doc_id)
            
            self.assertEqual(original_doc['text'], restored_doc['text'])
            self.assertEqual(original_doc['metadata'], restored_doc['metadata'])

    def test_batch_operations(self):
        """Test adding documents in batches"""
        # Generate a larger set of documents
        large_doc_set = self.generate_sample_documents(150)
        
        # Add with a small batch size
        batch_size = 30
        ids = self.db.add_documents(large_doc_set, batch_size=batch_size)
        
        # Check that all documents were added
        self.assertEqual(len(ids), len(large_doc_set))
        self.assertEqual(self.db.count(), len(large_doc_set))

    def test_edge_cases(self):
        """Test various edge cases"""
        # Test empty document list
        ids = self.db.add_documents([])
        self.assertEqual(len(ids), 0)
        
        # Test empty ID list for deletion
        success = self.db.delete_documents([])
        self.assertTrue(success)
        
        # Test updating non-existent document
        success = self.db.update_document("non_existent_id", text="New text")
        self.assertFalse(success)
        
        # Test query with empty where clause
        self.db.add_documents(self.sample_docs[:3])
        results = self.db.query_similar(
            query_embedding=self.sample_docs[0]['embedding'],
            where={}
        )
        self.assertIn('results', results)
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
                # Fix: check if embeddings exists and has content without direct boolean evaluation
                if result.get('embeddings') is not None and len(result.get('embeddings')) > 0:
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

if __name__ == '__main__':
    unittest.main()