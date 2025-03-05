## SelfRAGModel class:
This is a base class that provides a high-level interface for the model. Another class will inherit from this class, it can be from hugging face, or local model. The class will have the following methods:
- **__init__() or load()** - Initializes the model and its components. 
- **predict_retrieval_decision(input_text)** - Determines if retrieval is needed for the given input
- **generate_response(input_text)** - Generates content without retrieval augmentation
- **generate_with_context(input_text, context)** - Generates content with retrieved context

## EmbeddingModel class:

- **encode(text)** - Creates vector embeddings for retrieval queries

## ContextFusion class:

- **fuse_context(original_input, retrieved_contexts)** - Combines original query with retrieved information
- **prioritize_contexts(retrieved_contexts)** - Ranks and selects most relevant contexts

## Orchestrator class:

- **__call__(input_text)** or **forward(input_text)** - Main entry point handling the flow
- **handle_retrieval_path(input_text)** - Manages the "Yes" path with retrieval
- **handle_direct_path(input_text)** - Manages the "No" path without retrieval