import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np
import torch
from core.embedding import EmbeddingModel
from typing import List

# Import the class to test - adjust the import path as needed
from models.ollama_embedding import OllamaEmbedding


class TestOllamaEmbedding(unittest.TestCase):
    """Test suite for OllamaEmbedding class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock the ollama client
        self.mock_client_patcher = patch('ollama.Client')
        self.mock_client = self.mock_client_patcher.start()
        
        # Mock torch functions
        self.mock_layer_norm_patcher = patch('torch.nn.functional.layer_norm')
        self.mock_layer_norm = self.mock_layer_norm_patcher.start()
        self.mock_normalize_patcher = patch('torch.nn.functional.normalize')
        self.mock_normalize = self.mock_normalize_patcher.start()
        
        # Setup default layer_norm and normalize behavior
        self.mock_layer_norm.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        self.mock_normalize.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        
        # Setup default client behavior
        self.client_instance = self.mock_client.return_value
        self.client_instance.embed.return_value = {
            'embeddings': [[0.1, 0.2, 0.3, 0.4, 0.5]]
        }

    def tearDown(self):
        """Tear down test fixtures after each test method"""
        self.mock_client_patcher.stop()
        self.mock_layer_norm_patcher.stop()
        self.mock_normalize_patcher.stop()

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters"""
        embedding = OllamaEmbedding()
        
        # Assert client was created with correct parameters
        self.mock_client.assert_called_once_with(host="http://localhost:11434")
        
        # Assert default properties
        self.assertEqual(embedding.model_name, "nomic-embed-text")
        self.assertEqual(embedding.max_retries, 3)
        self.assertEqual(embedding.retry_delay, 1.0)
        self.assertEqual(embedding.max_workers, 4)
        self.assertEqual(embedding.matryoshka_dim, 512)

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        embedding = OllamaEmbedding(
            model_name="custom-model",
            host="http://custom-host:1234",
            max_retries=5,
            retry_delay=2.0,
            max_workers=8,
            matryoshka_dim=256,
            timeout=10
        )
        
        # Assert client was created with correct parameters
        self.mock_client.assert_called_once_with(
            host="http://custom-host:1234", 
            timeout=10
        )
        
        # Assert custom properties
        self.assertEqual(embedding.model_name, "custom-model")
        self.assertEqual(embedding.max_retries, 5)
        self.assertEqual(embedding.retry_delay, 2.0)
        self.assertEqual(embedding.max_workers, 8)
        self.assertEqual(embedding.matryoshka_dim, 256)

    def test_invalid_matryoshka_dim(self):
        """Test initialization with invalid matryoshka_dim"""
        with self.assertRaises(ValueError) as context:
            OllamaEmbedding(matryoshka_dim=100)
        
        self.assertIn("matryoshka_dim must be one of", str(context.exception))

    def test_post_process_embedding(self):
        """Test the post-processing of embeddings"""
        # Use a valid matryoshka_dim value
        matryoshka_dim = 64
        embedding = OllamaEmbedding(matryoshka_dim=matryoshka_dim)
        
        # Create a raw embedding with enough elements
        raw_embedding = [0.1] * 512  # Create longer embedding
        
        # Setup specific mocks for this test
        # Create tensor of correct length to simulate layer_norm output
        layer_norm_output = torch.tensor([[0.1] * 512])
        self.mock_layer_norm.return_value = layer_norm_output
        
        # Create tensor of correct length for normalize output
        normalize_output = torch.tensor([[0.1] * matryoshka_dim])
        self.mock_normalize.return_value = normalize_output
        
        # Call the method being tested
        processed = embedding._post_process_embedding(raw_embedding)
        
        # Assert layer_norm and normalize were called
        self.mock_layer_norm.assert_called_once()
        self.mock_normalize.assert_called_once()
        
        # Assert result has correct dimension
        self.assertEqual(len(processed), matryoshka_dim)
        
        # Instead of exact comparison, check that values are close enough (account for float precision)
        # First make sure it's the right length
        self.assertEqual(len(processed), matryoshka_dim)
        
        # Then check that all values are approximately 0.1
        for value in processed:
            self.assertAlmostEqual(value, 0.1, places=6)

    def test_encode_adds_prefix(self):
        """Test that encode adds the search_document prefix when missing"""
        embedding = OllamaEmbedding()
        embedding.encode("sample text")
        
        # Check that the client was called with prefixed text
        self.client_instance.embed.assert_called_with(
            model="nomic-embed-text",
            input="search_document: sample text"
        )

    def test_encode_keeps_existing_prefix(self):
        """Test that encode keeps existing prefixes"""
        embedding = OllamaEmbedding()
        
        # Test different valid prefixes
        prefixes = [
            "search_document: text",
            "search_query: text",
            "clustering: text",
            "classification: text"
        ]
        
        for prefixed_text in prefixes:
            embedding.encode(prefixed_text)
            self.client_instance.embed.assert_called_with(
                model="nomic-embed-text",
                input=prefixed_text
            )

    def test_encode_retries(self):
        """Test retry logic in encode method"""
        # Setup client to fail twice then succeed
        self.client_instance.embed.side_effect = [
            Exception("Connection error"),
            Exception("Timeout error"),
            {'embeddings': [[0.1, 0.2, 0.3, 0.4, 0.5]]}
        ]
        
        embedding = OllamaEmbedding(retry_delay=0.01)  # Short delay for testing
        result = embedding.encode("test text")
        
        # Assert embed was called 3 times
        self.assertEqual(self.client_instance.embed.call_count, 3)
        self.assertIsInstance(result, list)

    def test_encode_fallback(self):
        """Test fallback to random vector when all retries fail"""
        # Setup client to always fail
        self.client_instance.embed.side_effect = Exception("Always fails")
        
        with patch('numpy.random.rand') as mock_rand:
            mock_rand.return_value = np.array([0.1] * 768)
            
            embedding = OllamaEmbedding(max_retries=2, retry_delay=0.01)
            result = embedding.encode("test text")
            
            # Assert fallback was used
            mock_rand.assert_called_once()
            self.assertIsInstance(result, list)

    def test_encode_batch(self):
        """Test batch encoding"""
        texts = ["text1", "text2", "text3"]
        
        # Setup client response for batch
        self.client_instance.embed.return_value = {
            'embeddings': [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7]
            ]
        }
        
        embedding = OllamaEmbedding()
        results = embedding.encode_batch(texts)
        
        # Assert client was called with prefixed texts
        expected_input = [
            "search_document: text1",
            "search_document: text2", 
            "search_document: text3"
        ]
        self.client_instance.embed.assert_called_with(
            model="nomic-embed-text",
            input=expected_input
        )
        
        # Assert results are correct format
        self.assertEqual(len(results), 3)
        for emb in results:
            self.assertIsInstance(emb, list)

    def test_encode_batch_fallback(self):
        """Test fallback to individual processing when batch fails"""
        texts = ["text1", "text2"]
        
        # Setup client to fail on batch but succeed on individual calls
        self.client_instance.embed.side_effect = [
            Exception("Batch error"),  # Fail on batch
            {'embeddings': [[0.1, 0.2, 0.3, 0.4, 0.5]]},  # Succeed on first individual
            {'embeddings': [[0.2, 0.3, 0.4, 0.5, 0.6]]}   # Succeed on second individual
        ]
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            # Setup mock executor to call encode directly
            mock_executor.return_value.__enter__.return_value.map.side_effect = \
                lambda func, texts: [func(text) for text in texts]
            
            embedding = OllamaEmbedding()
            results = embedding.encode_batch(texts)
            
            # Assert executor was created with correct max_workers
            mock_executor.assert_called_with(max_workers=4)
            
            # Assert results are correct format
            self.assertEqual(len(results), 2)
            for emb in results:
                self.assertIsInstance(emb, list)

    def test_dimension_property_caching(self):
        """Test that dimension property caches result"""
        embedding = OllamaEmbedding()
        
        # First call should encode a sample text
        dim1 = embedding.dimension
        call_count = self.client_instance.embed.call_count
        
        # Second call should use cached value
        dim2 = embedding.dimension
        self.assertEqual(self.client_instance.embed.call_count, call_count)
        
        # Dimensions should be equal
        self.assertEqual(dim1, dim2)

    def test_integration_with_embedding_model_base_class(self):
        """Test that OllamaEmbedding properly inherits from EmbeddingModel"""
        embedding = OllamaEmbedding()
        self.assertIsInstance(embedding, EmbeddingModel)

    @patch('ollama.Client')
    def test_encode_with_proper_mock(self, mock_client):
        # Create mock response with properly sized embeddings
        # For nomic-embed-text, the full embedding size is typically 768
        mock_embedding = np.random.rand(768).tolist()
        
        # Configure the mock client
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Configure the embed method to return proper response structure
        mock_client_instance.embed.return_value = {
            'embeddings': [mock_embedding]
        }
        
        # Initialize the embedding model
        embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            matryoshka_dim=512
        )
        
        # Test encode method
        result = embedding_model.encode("Test document")
        
        # Verify the result
        self.assertEqual(len(result), 512)  # Should match the matryoshka_dim
        
        # Verify normalization (L2 norm should be close to 1.0)
        norm = sum(x*x for x in result) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)
        
        # Verify the client was called correctly
        mock_client_instance.embed.assert_called_once()
        call_args = mock_client_instance.embed.call_args[1]
        self.assertEqual(call_args['model'], "nomic-embed-text")
        self.assertTrue(call_args['input'].startswith('search_document:'))

    @patch('ollama.Client')
    def test_encode_batch(self, mock_client):
        # Create mock responses for batch
        batch_size = 3
        mock_embeddings = [np.random.rand(768).tolist() for _ in range(batch_size)]
        
        # Configure the mock
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.embed.return_value = {
            'embeddings': mock_embeddings
        }
        
        # Initialize the embedding model
        embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            matryoshka_dim=256  # Testing different dimension
        )
        
        # Test encode_batch method
        texts = ["Document 1", "Document 2", "Document 3"]
        results = embedding_model.encode_batch(texts)
        
        # Verify results
        self.assertEqual(len(results), batch_size)
        for emb in results:
            self.assertEqual(len(emb), 256)  # Should match the matryoshka_dim
            # Verify normalization
            norm = sum(x*x for x in emb) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=6)

if __name__ == '__main__':
    unittest.main()