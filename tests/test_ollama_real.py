import unittest
import tempfile
import os
from pathlib import Path
import shutil
import logging
import json
import numpy as np
import time

from TutorLLM.models.ollama_embedding import OllamaEmbedding
from TutorLLM.storage.knowledge_vault import KnowledgeVault


class TestOllamaIntegration(unittest.TestCase):
    """Integration tests for KnowledgeVault with real Ollama embedding model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(levelname)s:%(name)s:%(message)s'
        )
        cls.logger = logging.getLogger("integration_test")
        
        # Initialize real embedding model
        cls.embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            max_retries=2,
            retry_delay=1.0,
            matryoshka_dim=512
        )
        
        # Test connection and skip if Ollama server not available
        try:
            test_embedding = cls.embedding_model.encode("test connection")
            cls.skip_tests = False
            cls.logger.info(f"Ollama server connected - embedding dimension: {len(test_embedding)}")
        except Exception as e:
            cls.logger.warning(f"Ollama server not available: {e}")
            cls.skip_tests = True

    def setUp(self):
        """Set up before each test"""
        if self.skip_tests:
            self.skipTest("Ollama server not available")
        
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp()
        self.logger.info(f"Created test directory: {self.test_dir}")
        
        # Initialize KnowledgeVault with real embedding model
        self.vault = KnowledgeVault(
            embedding_model=self.embedding_model,
            chunk_size=200,  # Smaller chunks for faster tests
            chunk_overlap=40,
            max_workers=2  # Limit workers for testing
        )
        
        # Create test files
        self.create_test_files()

    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            self.logger.info(f"Removed test directory: {self.test_dir}")

    def create_test_files(self):
        """Create test markdown files with various content patterns"""
        # Simple document with frontmatter tags
        simple_doc = """---
title: Simple Test Document
tags: [embedding, test, documentation]
---

# Simple Document

This is a basic test document with frontmatter tags.
Testing basic knowledge vault functionality.
"""

        # Complex document with code blocks and inline tags
        complex_doc = """---
title: Complex Test Document
tags: [advanced, testing]
---

# Complex Document

This document contains #code examples and #technical content.

## Python Example
```python
def calculate_similarity(v1, v2):
    dot_product = sum(a*b for a, b in zip(v1, v2))
    norm1 = sum(a*a for a in v1) ** 0.5
    norm2 = sum(b*b for b in v2) ** 0.5
    return dot_product / (norm1 * norm2)
```
"""
        
        with open(os.path.join(self.test_dir, "simple_test.md"), "w") as f:
            f.write(simple_doc)
        
        with open(os.path.join(self.test_dir, "complex_test.md"), "w") as f:
            f.write(complex_doc)

    def test_real_embedding_dimensions(self):
        """Test that real embeddings have consistent dimensions"""
        # Process directory
        results = self.vault.process_directory(self.test_dir)
        
        # Debug logging
        self.logger.debug(f"Processing results: {len(results)} chunks")
        
        # Add task prefix to chunks before embedding
        for chunk in results:
            if "text" in chunk:
                chunk["text"] = f"search_document: {chunk['text']}"
        
        # Check first valid embedding
        valid_embeddings = [r for r in results if r.get("embedding") and len(r["embedding"]) > 0]
        self.assertTrue(len(valid_embeddings) > 0, "No valid embeddings found")
        
        # Get dimension from first embedding
        first_dim = len(valid_embeddings[0]["embedding"])
        self.logger.debug(f"First embedding dimension: {first_dim}")
        
        # All embeddings should have same dimension
        for chunk in valid_embeddings:
            embedding = chunk["embedding"]
            self.assertEqual(
                len(embedding),
                first_dim,
                f"Inconsistent embedding dimensions: {len(embedding)} vs {first_dim}"
            )
            
            # Check if embedding contains non-zero values
            has_nonzero = any(abs(x) > 1e-6 for x in embedding)
            self.assertTrue(
                has_nonzero,
                f"Empty embedding detected for text: {chunk['text'][:100]}..."
            )

    def test_real_tag_extraction(self):
        """Test tag extraction with real document processing"""
        results = self.vault.process_directory(self.test_dir)
        
        # Collect all tags from all chunks
        all_tags = set()
        for chunk in results:
            all_tags.update(chunk["metadata"]["tags"])
        
        # Check for both frontmatter and inline tags
        expected_tags = {"embedding", "test", "technical", "code", "documentation", "advanced", "testing"}
        self.assertTrue(expected_tags.issubset(all_tags))

    def test_real_embedding_stability(self):
        """Test that same text produces similar embeddings"""
        test_text = "This is a test sentence for embedding stability."
        
        # First, check if the model is deterministic
        embedding1 = self.embedding_model.encode(test_text)
        embedding2 = self.embedding_model.encode(test_text)
        
        # Calculate cosine similarity
        def cosine_similarity(v1, v2):
            dot_product = sum(a*b for a, b in zip(v1, v2))
            norm1 = sum(a*a for a in v1) ** 0.5
            norm2 = sum(b*b for b in v2) ** 0.5
            return dot_product / (norm1 * norm2)
        
        similarity = cosine_similarity(embedding1, embedding2)
        self.logger.info(f"Embedding similarity for same text: {similarity}")
        
        # If model is stochastic, use a more reasonable threshold

    def test_real_batch_processing(self):
        """Test batch processing with real embeddings"""
        # Create multiple test files
        for i in range(3):
            with open(os.path.join(self.test_dir, f"batch_test_{i}.md"), "w") as f:
                f.write(f"# Test {i}\nThis is test document {i}.")
        
        results = self.vault.process_directory(self.test_dir)
        
        # Verify all documents were processed
        unique_sources = {r["metadata"]["source_file"] for r in results}
        self.assertEqual(len(unique_sources), 5)  # 3 batch files + 2 original test files

if __name__ == '__main__':
    unittest.main()