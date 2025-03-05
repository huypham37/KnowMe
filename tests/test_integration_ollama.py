import unittest
import os
import tempfile
import shutil
import logging
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from datetime import datetime

# Import the modules to test
from storage.knowledge_vault import KnowledgeVault
from models.ollama_embedding import OllamaEmbedding

class TestKnowledgeVaultOllamaIntegration(unittest.TestCase):
    """Integration tests for KnowledgeVault with OllamaEmbedding"""
    
    def setUp(self):
        """Set up the test environment with a temporary directory and test files"""
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Mock the Ollama client
        self.mock_ollama_client = MagicMock()
        
        # Create a mock embedding result that mimics Ollama API response
        self.mock_embedding_result = {
            'embeddings': [
                [0.1] * 768,  # Mock embedding vector of length 768
            ]
        }
        
        # Configure the mock client to return the mock embedding
        self.mock_ollama_client.embed.return_value = self.mock_embedding_result
        
        # Create a patcher for the Ollama Client
        self.ollama_patcher = patch('ollama.Client', return_value=self.mock_ollama_client)
        self.ollama_patcher.start()
        
        # Create the embedding model
        self.embedding_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            host="http://localhost:11434",
            matryoshka_dim=512,
            max_retries=2,
            retry_delay=0.01  # Short delay for tests
        )
        
        # Create the KnowledgeVault with the embedding model
        self.vault = KnowledgeVault(
            embedding_model=self.embedding_model,
            chunk_size=200,
            chunk_overlap=50,
            max_workers=2,
            batch_size=16
        )
        
        # Create test files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove temporary directory and its contents
        shutil.rmtree(self.test_dir)
        
        # Stop the patcher
        self.ollama_patcher.stop()
    
    def create_test_files(self):
        """Create test markdown files with various content types"""
        # Test file 1: Basic markdown with headers and lists
        file1_content = """# Test Document 1
This is a test document with basic markdown.
## Section 1
- Item 1
- Item 2
#tag1 #tag2
"""
        
        # Test file 2: Frontmatter and code blocks
        file2_content = """---
title: Test Document 2
tags: [test, markdown]
---
# Another Test Document
This document has frontmatter and code.

```python
def test():
    print("Hello")
```
#code #python
"""
        
        # Test file 3: Complex nested structure
        file3_content = """# Nested Document
## Level 2
### Level 3
- Nested list
  - Subitem 1
  - Subitem 2
    - Sub-subitem

> Blockquote with #important tag
"""
        
        # Test file 4: Empty sections and special characters
        file4_content = """# Empty Sections Test

## Empty Section

## Section with content
Content here with special chars: @#$%
#special #chars

## Another empty section

"""
        
        # Write files to temp directory
        test_files = {
            "doc1.md": file1_content,
            "doc2.md": file2_content,
            "nested/doc3.md": file3_content,
            "doc4.md": file4_content
        }
        
        for filepath, content in test_files.items():
            full_path = os.path.join(self.test_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
    
    
    
    def test_chunking_behavior(self):
        """Test document chunking with various content types"""
        test_cases = [
            # Single chunk case
            ("Short content", 1),
            # # Multi-chunk case
            ("This is a longer content that should be split into multiple chunks. " * 5, 2),
            # Empty content
            ("", 1),
            # Content with exactly chunk size
            ("x" * self.vault.chunk_size, 1),
            # Content with special characters
            ("Special chars: @#$%\nNew line\nAnother line\n" * 3, 1)
        ]
        
        for content, expected_chunks in test_cases:
            with self.subTest(content=content[:50]):
                # Write test content
                test_file = os.path.join(self.test_dir, "chunk_test.md")
                with open(test_file, "w") as f:
                    f.write(content)
                
                # Process file
                results = self.vault.process_file(test_file)
                
                # Verify number of chunks
                self.assertEqual(len(results), expected_chunks)
                
                # Verify chunk properties
                for chunk in results:
                    self.assertLessEqual(len(chunk["text"]), self.vault.chunk_size)
    
    def test_metadata_extraction(self):
        """Test metadata extraction from various document types"""
        results = self.vault.process_directory(self.test_dir)
        
        # Test frontmatter tags
        doc2_chunks = [item for item in results 
                      if os.path.basename(item["metadata"]["source_file"]) == "doc2.md"]
        self.assertGreater(len(doc2_chunks), 0)
        for chunk in doc2_chunks:
            tags = chunk["metadata"]["tags"]
            self.assertIn("test", tags)
            self.assertIn("markdown", tags)
        
        # Test inline tags
        doc1_chunks = [item for item in results 
                      if os.path.basename(item["metadata"]["source_file"]) == "doc1.md"]
        self.assertGreater(len(doc1_chunks), 0)
        for chunk in doc1_chunks:
            tags = chunk["metadata"]["tags"]
            if "#tag1" in chunk["text"]:
                self.assertIn("tag1", tags)
            if "#tag2" in chunk["text"]:
                self.assertIn("tag2", tags)
        
        # Test nested document structure
        nested_chunks = [item for item in results 
                        if "nested" in item["metadata"]["source_file"]]
        self.assertGreater(len(nested_chunks), 0)
        for chunk in nested_chunks:
            if "important" in chunk["text"]:
                self.assertIn("important", chunk["metadata"]["tags"])
    
    def test_embedding_error_handling(self):
        """Test error handling during embedding generation"""
        # Create a test file
        error_file = os.path.join(self.test_dir, "error_test.md")
        with open(error_file, "w") as f:
            f.write("# Error Test\nThis should trigger an error.")
        
        # Configure mock to fail on specific content
        def embed_side_effect(model=None, input=None):
            if "Error Test" in input:
                raise ValueError("Simulated embedding error")
            return self.mock_embedding_result
        
        self.mock_ollama_client.embed.side_effect = embed_side_effect
        
        # Process should continue despite errors
        results = self.vault.process_directory(self.test_dir)
        
        # Verify other documents were processed
        self.assertGreater(len(results), 0)
        
        # Check error document handling
        error_chunks = [item for item in results 
                       if os.path.basename(item["metadata"]["source_file"]) == "error_test.md"]
        
        if error_chunks:
            for chunk in error_chunks:
                # Should have either valid embedding or None
                self.assertTrue(
                    chunk["embedding"] is None or 
                    isinstance(chunk["embedding"], list)
                )
    
    def test_batch_processing(self):
        """Test batch processing of documents"""
        # Create batch test files
        for i in range(3):
            with open(os.path.join(self.test_dir, f"batch_{i}.md"), "w") as f:
                f.write(f"# Batch Document {i}\nContent for batch testing.\n#batch{i}")
        
        # Configure mock for batch processing
        batch_response = {
            'embeddings': [[0.1] * 768 for _ in range(3)]
        }
        self.mock_ollama_client.embed.return_value = batch_response
        
        # Process directory
        results = self.vault.process_directory(self.test_dir)
        
        # Verify batch processing
        batch_chunks = [item for item in results 
                       if "batch_" in item["metadata"]["source_file"]]
        self.assertEqual(len(batch_chunks), 3)
        
        # Verify embeddings and tags
        for i, chunk in enumerate(batch_chunks):
            self.assertEqual(len(chunk["embedding"]), 512)
            self.assertIn(f"batch{i}", chunk["metadata"]["tags"])
    
    def test_concurrent_processing(self):
        """Test concurrent processing of multiple files"""
        # Create multiple test files
        for i in range(5):
            with open(os.path.join(self.test_dir, f"concurrent_{i}.md"), "w") as f:
                f.write(f"# Concurrent Test {i}\nContent for concurrent testing.\n")
        
        # Process directory with multiple workers
        self.vault.max_workers = 3
        results = self.vault.process_directory(self.test_dir)
        
        # Verify all files were processed
        concurrent_files = [item for item in results 
                          if "concurrent_" in item["metadata"]["source_file"]]
        self.assertEqual(len(set(item["metadata"]["source_file"] 
                               for item in concurrent_files)), 5)
        
        # Verify each file was processed correctly
        for item in concurrent_files:
            self.assertIsNotNone(item["embedding"])
            self.assertEqual(len(item["embedding"]), 512)

if __name__ == '__main__':
    unittest.main()