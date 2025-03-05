import os
import shutil
import tempfile
import unittest
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime

from TutorLLM.storage.knowledge_vault import KnowledgeVault
from TutorLLM.models.ollama_embedding import OllamaEmbedding
from TutorLLM.storage.vector_db import VectorDB

class MockEmbeddingModel:
    """Mock embedding model for testing"""
    
    def __init__(self, dimension=512):
        self.dimension = dimension
        self.encode_called = 0
        self.batch_encode_called = 0
        self.texts = []
        self.model_name = "nomic-embed-text"  # Add default model_name
        
    def encode(self, text):
        """Return fake embeddings"""
        self.encode_called += 1
        self.texts.append(text)
        # Generate deterministic fake embeddings based on text length
        return [0.1] * self.dimension
        
    def encode_batch(self, texts):
        """Return fake batch embeddings"""
        self.batch_encode_called += 1
        self.texts.extend(texts)
        # Generate deterministic fake embeddings based on text length
        return [[0.1] * self.dimension for _ in texts]
        
    @property
    def dimension(self):
        return self._dimension
        
    @dimension.setter
    def dimension(self, value):
        self._dimension = value


class TestKnowledgeIntegration(unittest.TestCase):
    """Test integration between KnowledgeVault, OllamaEmbedding and VectorDB"""

    def setUp(self):
        """Set up test environment"""
        # Create temp dir for test files
        self.test_dir = tempfile.mkdtemp()
        self.knowledge_dir = os.path.join(self.test_dir, "knowledge")
        self.db_dir = os.path.join(self.test_dir, "vectordb")
        self.backup_dir = os.path.join(self.test_dir, "backup")
        
        # Create directory structure
        os.makedirs(self.knowledge_dir)
        os.makedirs(self.db_dir)
        os.makedirs(self.backup_dir)
        
        # Create test markdown files
        self.create_test_markdown_files()
        
        # Create mock embedding model
        self.mock_embedding = MockEmbeddingModel(dimension=512)
        
        # Configure logging for tests
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("test_knowledge_integration")
        
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.test_dir)

    def create_test_markdown_files(self):
        """Create test markdown files in the knowledge directory"""
        # File 1: Simple markdown
        with open(os.path.join(self.knowledge_dir, "test1.md"), "w") as f:
            f.write("""# Test Document 1
            
					Tags: #test, #document
								
					This is a test document with some content.
					It should be processed correctly by the knowledge vault.
						
					## Section 1
					Some content in section 1.
						
					## Section 2
					More content in section 2.
						""")
        
        # File 2: Longer markdown with code blocks
        with open(os.path.join(self.knowledge_dir, "test2.md"), "w") as f:
            f.write("""# Test Document 2
            
				Tags: #code, #python
							
				This document contains some code examples.
					
				```python
				def hello_world():
					print("Hello, world!")
				```
					""")

    def test_full_integration_flow(self):
        """Test the full integration flow from KnowledgeVault through embeddings to VectorDB"""
        # Initialize components
        knowledge_vault = KnowledgeVault(
            embedding_model=self.mock_embedding,
            chunk_size=200,
            chunk_overlap=50
        )
        
        vector_db = VectorDB(
            persist_directory=self.db_dir,
            collection_name="test_collection",
            embedding_dimension=512,
            distance_metric="cosine"
        )
        
        # Process directory and get embedded chunks
        chunks = knowledge_vault.process_directory(self.knowledge_dir)
        
        # Assert chunks were created
        self.assertGreater(len(chunks), 0, "No chunks were created")
        
        # Verify chunks have required fields
        for chunk in chunks:
            self.assertIn("text", chunk, "Chunk missing text field")
            self.assertIn("embedding", chunk, "Chunk missing embedding field")
            self.assertIn("metadata", chunk, "Chunk missing metadata field")
            
            # Check metadata fields
            metadata = chunk["metadata"]
            self.assertIn("source_file", metadata, "Metadata missing source field")
            self.assertIn("doc_title", metadata, "Metadata missing title field")
            
        # Add documents to vector DB
        doc_ids = vector_db.add_documents(chunks)
        
        # Verify documents were added
        self.assertEqual(len(doc_ids), len(chunks), "Not all documents were added to the vector DB")
        
        # Verify count matches
        self.assertEqual(vector_db.count(), len(chunks), "Vector DB count doesn't match number of chunks")
        
        # Test query functionality with mock embedding
        query_text = "test query"
        query_embedding = self.mock_embedding.encode(query_text)
        
        # Query by embedding
        results = vector_db.query_similar(
            query_embedding=query_embedding,
            n_results=2
        )
        
        # Verify query results
        self.assertIn("results", results, "Query results missing results list")
        self.assertIn("raw_response", results, "Query results missing raw_response")
        raw_response = results["raw_response"]
        self.assertIn("ids", raw_response, "Query results missing ids in raw_response")
        self.assertIn("documents", raw_response, "Query results missing documents in raw_response")
        self.assertIn("metadatas", raw_response, "Query results missing metadatas in raw_response")
        self.assertIn("distances", raw_response, "Query results missing distances in raw_response")
        
        # Test backup and restore
        success = vector_db.create_backup(self.backup_dir)
        self.assertTrue(success, "Backup creation failed")
        
        # Check backup file exists
        backup_file = os.path.join(self.backup_dir, "test_collection_backup.json")
        self.assertTrue(os.path.exists(backup_file), "Backup file not created")
        
        # Create a new vector DB instance
        new_db_dir = os.path.join(self.test_dir, "new_vectordb")
        os.makedirs(new_db_dir)
        
        new_vector_db = VectorDB(
            persist_directory=new_db_dir,
            collection_name="test_collection",
            embedding_dimension=512,
            distance_metric="cosine"
        )
        
        # Restore from backup
        success = new_vector_db.restore_from_backup(backup_file)
        self.assertTrue(success, "Restore from backup failed")
        
        # Verify count matches original DB
        self.assertEqual(new_vector_db.count(), vector_db.count(), "Restored DB count doesn't match original")
        
        # Create a test file to process
        with open(os.path.join(self.knowledge_dir, "test_single.md"), "w") as f:
            f.write("""# Single Test
            
This is a single test file for the integration test.
""")

        # Process file
        chunks = knowledge_vault.process_file(os.path.join(self.knowledge_dir, "test_single.md"))
        
        original_count = vector_db.count()  # Store the original count before adding new chunks
        
        # Add to vector DB
        doc_ids = vector_db.add_documents(chunks)
        
        # Verify document was added
        self.assertEqual(len(doc_ids), len(chunks), "Not all documents were added")
        self.assertEqual(vector_db.count(), original_count + len(chunks), "Vector DB count doesn't match")
        
        # Verify Ollama client was called
        self.assertGreater(self.mock_embedding.encode_called, 0, "Mock embedding model encode method was not called")

    def test_metadata_integrity(self):
        """Test that metadata is preserved throughout the pipeline"""
        # Initialize components
        knowledge_vault = KnowledgeVault(
            embedding_model=self.mock_embedding,
            chunk_size=200,
            chunk_overlap=50
        )
        
        vector_db = VectorDB(
            persist_directory=self.db_dir,
            collection_name="test_collection",
            embedding_dimension=512,
            distance_metric="cosine"
        )
        
        # Process directory
        chunks = knowledge_vault.process_directory(self.knowledge_dir)
        
        # Verify chunks have metadata
        for chunk in chunks:
            self.assertIn("metadata", chunk)
            metadata = chunk["metadata"]
            self.assertIn("source_file", metadata)
            self.assertIn("doc_title", metadata)
            if "tags" in metadata:
                self.assertIsInstance(metadata["tags"], list)
        
        # Add to vector DB
        doc_ids = vector_db.add_documents(chunks)
        
        # Get metadata summary
        summary = vector_db.get_metadata_summary()
        
        # Verify metadata fields are present in summary
        self.assertIn("fields", summary)
        fields = summary["fields"]
        self.assertIn("source_file", fields)
        self.assertIn("doc_title", fields)
        
        # Get a sample document
        peek = vector_db.peek(1)
        
        # Verify metadata is preserved
        self.assertIn("samples", peek)
        self.assertGreater(len(peek["samples"]), 0)
        sample = peek["samples"][0]
        self.assertIn("metadata", sample)
        self.assertIn("source_file", sample["metadata"])  # Change from "source" to "source_file"
        self.assertIn("doc_title", sample["metadata"])  # This likely needs to be kept as "doc_title"

    @patch('TutorLLM.models.ollama_embedding.ollama')
    def test_with_real_embedding_model(self, mock_ollama):
        """Test using a patched real embedding model instead of mock"""
        # Configure mock Ollama client with embed method
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        
        # Mock the embed method call - this matches how OllamaEmbedding actually uses the client
        mock_embeddings_response = {
            'embeddings': [[0.1] * 768]  # Format that matches what the embed method returns
        }
        
        # Set up the mock for the embed method - this is what the class actually calls
        mock_client.embed = MagicMock(return_value=mock_embeddings_response)
        
        # Initialize real embedding model with mocked backend
        embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            matryoshka_dim=512,
        )
        
        # Initialize components
        knowledge_vault = KnowledgeVault(
            embedding_model=embedding_model,
            chunk_size=200,
            chunk_overlap=50
        )
        
        vector_db = VectorDB(
            persist_directory=self.db_dir,
            collection_name="test_collection",
            embedding_dimension=512,
            distance_metric="cosine"
        )
        
        # Create a test file to process
        with open(os.path.join(self.knowledge_dir, "test_single.md"), "w") as f:
            f.write("""# Single Test
            
This is a single test file for the real embedding model test.
""")

        # Process file
        chunks = knowledge_vault.process_file(os.path.join(self.knowledge_dir, "test_single.md"))
        
        # Verify chunks were created
        self.assertGreater(len(chunks), 0, "No chunks were created")
        
        # Add to vector DB
        doc_ids = vector_db.add_documents(chunks)
        
        # Verify document was added
        self.assertEqual(len(doc_ids), len(chunks), "Not all documents were added")
        
        # Reset mock to ensure we only check calls related to the query
        mock_client.embed.reset_mock()
        
        # Test query functionality
        query_text = "test query"
        
        # Query by text using the embedding model
        results = vector_db.query_by_text(
            query_text=query_text,
            embedding_function=embedding_model.encode,
            n_results=2
        )
        
        # Verify query results
        self.assertIn("results", results, "Query results missing results list")
        self.assertIn("raw_response", results, "Query results missing raw_response")
        raw_response = results["raw_response"]
        self.assertIn("ids", raw_response, "Query results missing ids in raw_response")
        self.assertIn("documents", raw_response, "Query results missing documents in raw_response")
        
        # Check that embed was called - this matches the actual implementation
        self.assertTrue(
            mock_client.embed.called,
            "Ollama embed method was not called"
        )


if __name__ == '__main__':
    unittest.main()


