import mlx.core as mx
from mlx_lm import load, generate
from transformers import AutoTokenizer

class SelfRAGModel:
    def __init__(self, model_path="/Users/mac/mlx-model/selfrag_llama2_7b_mlx", max_tokens=100):
        """Initialize the SelfRAG model with MLX."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("selfrag/selfrag_llama2_7b")
        
        # Load the quantized MLX model and config
        self.model, self.config = load(model_path)  # Uses mlx_lm.load
        self.max_tokens = max_tokens

    def format_prompt(self, instruction, paragraph=None):
        """Format the input prompt."""
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        if paragraph:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
        return prompt

    def generate(self, instruction, paragraph=None):
        """Generate raw text output using MLX."""
        # Format the prompt
        formatted_prompt = self.format_prompt(instruction, paragraph)
        
        # Generate output using mlx_lm.generate
        output_text = generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=self.max_tokens,
            verbose=True  # Set to True for debugging
        )
        
        return output_text

# Example usage
if __name__ == "__main__":
    model = SelfRAGModel(model_path="/Users/mac/mlx-model/selfrag_llama2_7b_mlx")
    query = "Do you know Ashish Shai professor in Maastricht University?"
    paragraph = ""
    response = model.generate(query, paragraph)
    print(query)
    print(response)