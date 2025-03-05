Collecting workspace informationI'll help you create a comprehensive README for your KnowMe project. Based on the workspace structure and files, I'll write a README that explains the project's purpose, features, architecture, and usage instructions.

# KnowMe

A sophisticated tutoring system powered by Large Language Models with self-retrieval augmented generation capabilities.

## Overview

KnowMe is an advanced tutoring system that combines modern LLMs with sophisticated context retrieval to deliver highly personalized learning experiences. The system dynamically determines when to retrieve external knowledge based on the input query, enhancing responses with relevant information without unnecessary context bloat.

## Features

- **Self-RAG Architecture**: Intelligently determines when to use retrieval augmentation for responses
- **Nomic Embeddings Integration**: Utilizes `nomic-embed-text` for high-quality vector embeddings
- **Document Processing**: Automatically processes, chunks, and embeds markdown documents
- **Knowledge Vault**: Maintains a repository of knowledge that can be efficiently retrieved
- **Context Fusion**: Intelligently combines retrieved context with user queries
- **Modular Design**: Components are designed to be easily swapped or upgraded

## Architecture

The system follows a modern RAG architecture with self-retrieval decision making:

1. User input is analyzed to determine if external knowledge is needed
2. If retrieval is necessary, relevant documents are fetched from the vector database
3. Retrieved context is fused with the original query using the `context_fusion` module
4. The enhanced query is sent to the `llama3_model` for final response generation

The `orchestrator` manages this entire workflow seamlessly.

## Installation

```bash
# Clone the repository
git clone https://github.com/huypham37/KnowMe.git
cd KnowMe

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from TutorLLM.core.orchestrator import Orchestrator

# Initialize the orchestrator with default settings
orchestrator = Orchestrator()

# Get a response
response = orchestrator("Explain the concept of matryoshka representation learning")
print(response)
```

For document processing:

```python
from TutorLLM.storage.document_processor import DocumentProcessor

# Initialize document processor
processor = DocumentProcessor(
    docs_directory="/path/to/your/documents",
    embedding_model_name="nomic-embed-text",
    matryoshka_dim=512
)

# Process all documents in the directory
num_processed = processor.process_documents()
print(f"Processed {num_processed} documents")
```

## Configuration

Key settings can be configured in the `settings.py` file, including:

- Model selection and parameters
- Vector database settings
- Document processing parameters
- Embedding dimensions

## Known Issues

See the `known-issues` directory for information about current limitations and workarounds, including:

- Quantization issues on Apple Silicon Macs

## Testing

Run the test suite with:

```bash
python -m unittest discover -s TutorLLM/tests
```

## License

[Include your license information here]

## Credits

This project builds upon several open-source technologies:
- [Nomic AI](https://nomic.ai/) for embedding models
- [Ollama](https://github.com/ollama/ollama) for local model integration
- [SelfRAG](https://huggingface.co/selfrag/selfrag_llama2_7b) for retrieval augmented generation components