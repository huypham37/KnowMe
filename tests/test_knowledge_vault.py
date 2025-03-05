import unittest
import tempfile
import os
from unittest.mock import Mock, patch
import logging
from datetime import datetime

from storage.knowledge_vault import KnowledgeVault

class TestKnowledgeVault(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock embedding model
        self.mock_model = Mock()
        self.mock_model.encode.return_value = [0.1, 0.2, 0.3]
        
        # Initialize KnowledgeVault
        self.vault = KnowledgeVault(
            embedding_model=self.mock_model,
            chunk_size=100,
            chunk_overlap=20
        )

    def tearDown(self):
        """Clean up after tests"""
        import shutil
        shutil.rmtree(self.test_dir)

    def create_test_file(self, content, filename="test.md"):
        """Helper to create test markdown files"""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath

    def test_chunk_document_empty(self):
        """Test chunking empty document"""
        chunks = self.vault.chunk_document("")
        self.assertEqual(chunks, [""])

    def test_chunk_document_small(self):
        """Test chunking document smaller than chunk size"""
        content = "Small test document."
        chunks = self.vault.chunk_document(content)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], content)

    def test_chunk_document_exact_size(self):
        """Test chunking document exactly chunk size"""
        content = "x" * self.vault.chunk_size
        chunks = self.vault.chunk_document(content)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), self.vault.chunk_size)

    def test_chunk_document_with_overlap(self):
        """Test chunking with overlap"""
        # Create content that should split into two chunks with overlap
        content = "First chunk. " + "Middle part. " * 10 + "Last chunk."
        chunks = self.vault.chunk_document(content)
        
        self.assertTrue(len(chunks) >= 2)
        # Check overlap
        self.assertTrue(chunks[0][-20:] in chunks[1])

    def test_preprocess_text(self):
        """Test text preprocessing"""
        test_cases = [
            # Empty text
            ("", "empty_document"),
            # Whitespace only
            ("   \n   \t   ", "empty_document"),
            # Multiple spaces
            ("multiple   spaces   here", "multiple spaces here"),
            # Special characters
            ("Special $#@ characters!", "Special characters!"),
            # Multiple periods
            ("Multiple periods...here", "Multiple periods.here"),
            # Code blocks
            ("```python\ndef test():\n    pass\n```", "python def test pass"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.vault.preprocess_text(input_text)
                self.assertEqual(result, expected)

    def test_extract_tags_frontmatter(self):
        """Test extracting tags from frontmatter"""
        content = """---
        title: Test
        tags: [tag1, tag2, tag3]
        ---
        # Content
        """
        tags = self.vault.extract_tags(content)
        self.assertEqual(set(tags), {"tag1", "tag2", "tag3"})

    def test_extract_tags_inline(self):
        """Test extracting inline tags"""
        content = "Content with #tag1 and #tag2 tags"
        tags = self.vault.extract_tags(content)
        self.assertEqual(set(tags), {"tag1", "tag2"})

    def test_get_title(self):
        """Test title extraction"""
        test_cases = [
            ("# Main Title\nContent", "Main Title"),
            ("Content without title", "Content Without Title"),
            ("## Secondary Title\nContent", "Secondary Title"),
        ]
        
        for content, expected in test_cases:
            with self.subTest(content=content):
                result = self.vault.get_title("test.md", content)
                self.assertEqual(result, expected)

    def test_process_file_with_empty_chunks(self):
        """Test processing file that might generate empty chunks"""
        content = """---
        title: Test
        tags: [test]
        ---
        
        # Test Document
        
        
        
        ## Empty Section
        
        
        ## Another Section
        Content here
        """
        
        filepath = self.create_test_file(content)
        results = self.vault.process_file(filepath)
        
        # Check that no empty chunks made it to embedding
        for chunk in results:
            self.assertTrue(len(chunk["text"].strip()) > 0)
            self.assertIsNotNone(chunk["embedding"])

    def test_process_directory_structure(self):
        """Test processing nested directory structure"""
        # Create nested directory structure
        os.makedirs(os.path.join(self.test_dir, "subdir"))
        
        # Create test files
        files = [
            ("test1.md", "# Test 1\nContent 1"),
            ("subdir/test2.md", "# Test 2\nContent 2"),
            ("test3.txt", "Not a markdown file"),
        ]
        
        for filename, content in files:
            filepath = os.path.join(self.test_dir, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(content)
        
        results = self.vault.process_directory(self.test_dir)
        
        # Should only process markdown files
        markdown_files = [r["metadata"]["source_file"] for r in results]
        self.assertEqual(len(markdown_files), 2)
        self.assertTrue(any("test1.md" in f for f in markdown_files))
        self.assertTrue(any("test2.md" in f for f in markdown_files))

if __name__ == '__main__':
    unittest.main()