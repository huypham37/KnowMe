import os
import glob
from pathlib import Path
import concurrent.futures
import time
from datetime import datetime
import re
from typing import List, Dict, Any, Optional, Callable
import logging
import sys
# Ensure TutorLLM is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class KnowledgeVault:
    """
    KnowledgeVault processes markdown documents to create embeddings for a vector database.
    It handles document discovery, chunking, metadata extraction, and embedding generation.
    """
    
    def __init__(
        self,
        embedding_model=None,  # Will be injected
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_workers: int = 4,
        batch_size: int = 32
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all markdown files in the given directory and return embedded chunks with metadata.
        
        Args:
            directory_path: Path to the directory containing markdown files
            
        Returns:
            List of dictionaries containing embedded chunks with metadata
        """
        self.logger.info(f"Processing directory: {directory_path}")
        markdown_files = self.find_markdown_files(directory_path)
        self.logger.info(f"Found {len(markdown_files)} markdown files")
        
        results = []
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in markdown_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_results = future.result()
                    results.extend(file_results)
                    self.logger.info(f"Processed file: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")
                    
        return results
    
    def find_markdown_files(self, directory_path: str) -> List[str]:
        """
        Find all markdown files in the given directory and its subdirectories.
        
        Args:
            directory_path: Path to the directory to search
            
        Returns:
            List of paths to markdown files
        """
        markdown_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.md', '.markdown')):
                    markdown_files.append(os.path.join(root, file))
        return markdown_files
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single markdown file: read, chunk, extract metadata, and embed.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            List of dictionaries containing embedded chunks with metadata
        """
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.logger.debug(f"File content length: {len(content)}")
            
            # Split the document into chunks
            chunks = self.chunk_document(content)
            self.logger.debug(f"Generated {len(chunks)} chunks")
            
            # Extract title and create basic metadata
            doc_title = self.get_title(file_path, content)
            creation_date = self.get_file_date(file_path)
            
            embedded_chunks = []
            
            # Process each chunk
            for index, chunk_text in enumerate(chunks):
                self.logger.debug(f"Processing chunk {index+1}/{len(chunks)}")
                self.logger.debug(f"Chunk length: {len(chunk_text)}")
                
                # Extract tags from chunk content
                tags = self.extract_tags(chunk_text)
                
                try:
                    # Embed the chunk
                    embedding = self.embed_chunk(chunk_text) if self.embedding_model else None
                    
                    embedded_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source_file": file_path,
                            "doc_title": doc_title,
                            "chunk_index": index,
                            "total_chunks": len(chunks),
                            "creation_date": creation_date,
                            "tags": tags
                        },
                        "embedding": embedding
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {index}: {str(e)}")
                    continue
                    
            return embedded_chunks
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def chunk_document(self, content: str) -> List[str]:
        """
        Split document content into overlapping chunks.
        """
        if not content or len(content) <= self.chunk_size:
            return [content]
        
        chunks = []
        start = 0
        content_length = len(content)
        
        while start < content_length:
            end = min(start + self.chunk_size, content_length)
            chunk = content[start:end]
            chunks.append(chunk)
            
            # If we've reached the end, break out of the loop.
            if end == content_length:
                break
            
            # Move start position, accounting for overlap.
            start = end - self.chunk_overlap
        
        return chunks
    
    def get_title(self, file_path: str, content: str) -> str:
        """
        Extract title from markdown content or generate from content.
        
        Args:
            file_path: Path to the source file
            content: Document content as string
            
        Returns:
            Document title as string
            
        Examples:
            >>> vault = KnowledgeVault()
            >>> vault.get_title("test.md", "# Main Title\nContent")
            "Main Title"
            >>> vault.get_title("test.md", "## Secondary Title\nContent")
            "Secondary Title"
            >>> vault.get_title("test.md", "Content without title")
            "Content Without Title"
        """
        # Try to find a markdown header (# or ##)
        header_match = re.search(r'^#+ (.*?)$', content, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()
        
        # If no header found, use first line of content
        first_line = content.strip().split('\n')[0]
        
        # Clean and capitalize the title
        title = ' '.join(
            word.capitalize() 
            for word in first_line.split()
            if word and not word.startswith('#')
        )
        
        # If still empty, use filename without extension
        if not title:
            title = Path(file_path).stem.replace('_', ' ').title()
            
        return title
    
    def get_file_date(self, file_path: str) -> str:
        """
        Get the file creation or modification date.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File date as ISO format string
        """
        try:
            mtime = os.path.getmtime(file_path)
            return datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            return datetime.now().isoformat()
    
    def extract_tags(self, content: str) -> List[str]:
        """
        Extract tags from both frontmatter and inline markdown content
        
        Args:
            content: The document content as a string
            
        Returns:
            List of unique tags found in the document
        """
        tags = set()
        
        # Extract frontmatter tags
        if content.startswith('---'):
            try:
                end_frontmatter = content.find('---', 3)
                if end_frontmatter != -1:
                    frontmatter = content[3:end_frontmatter]
                    # Look for tags: [...] pattern
                    tag_match = re.search(r'tags:\s*\[(.*?)\]', frontmatter)
                    if tag_match:
                        frontmatter_tags = [t.strip() for t in tag_match.group(1).split(',')]
                        tags.update(frontmatter_tags)
            except Exception as e:
                self.logger.error(f"Error parsing frontmatter tags: {e}")
        
        # Extract inline hashtags
        inline_tags = re.findall(r'#(\w+)', content)
        tags.update(inline_tags)
        
        return list(tags)
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text before embedding
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
            
        Examples:
            >>> vault = KnowledgeVault()
            >>> vault.preprocess_text("Special $#@ characters!")
            "Special characters!"
            >>> vault.preprocess_text("multiple   spaces   here")
            "multiple spaces here"
        """
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return "empty_document"
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize whitespace (replace multiple spaces with single space)
        text = ' '.join(text.split())
        
        # Remove multiple periods/ellipsis
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()

    def embed_chunk(self, text: str) -> List[float]:
        """
        Generate embedding for a chunk of text using the configured model.
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        try:
            # Preprocess text
            clean_text = self.preprocess_text(text)
            self.logger.debug(f"Original text length: {len(text)}")
            self.logger.debug(f"Cleaned text length: {len(clean_text)}")
            
            # Log model status
            self.logger.debug(f"Using embedding model: {self.embedding_model.model_name}")
            
            # Generate embedding
            self.logger.debug("Generating embedding...")
            embedding = self.embedding_model.encode(clean_text)
            
            # Log embedding details
            if hasattr(embedding, 'shape'):
                self.logger.debug(f"Embedding shape: {embedding.shape}")
            self.logger.debug(f"Embedding type: {type(embedding)}")
            
            # Convert to list if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            else:
                embedding = list(embedding)
            
            # Validate embedding
            if not embedding:
                self.logger.error(f"Empty embedding generated for text: {clean_text[:100]}...")
                raise ValueError("Generated embedding is empty")
            
            self.logger.debug(f"Successfully generated embedding of length {len(embedding)}")
            return embedding
                
        except Exception as e:
            self.logger.error(f"Error embedding text: {str(e)}")
            self.logger.error(f"Problematic text chunk: {text[:100]}...")
            raise  # Re-raise to handle in calling function