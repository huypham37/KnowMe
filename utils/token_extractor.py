import re
import logging
from typing import List, Tuple, Optional

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SelfRAGTokenExtractor:
    def __init__(self):
        # Define regex patterns for reflection tokens
        self.retrieval_pattern = r"\[Retrieval\]"
        self.no_retrieval_pattern = r"\[No Retrieval\]"
        self.continue_evidence_pattern = r"\[Continue to Use Evidence\]"
        self.utility_pattern = r"\[Utility:[1-5]\]"

    def extract_tokens(self, response: str) -> Tuple[List[str], List[str], List[str], Optional[int]]:
        """
        Extract reflection tokens from the model's response.

        Args:
            response (str): The model's generated response containing tokens.

        Returns:
            Tuple containing:
            - List of [Retrieval] occurrences
            - List of [No Retrieval] occurrences
            - List of [Continue to Use Evidence] occurrences
            - Utility score (int between 1 and 5) if present, else None
        """
        # Extract tokens using regex
        retrieval_tokens = re.findall(self.retrieval_pattern, response)
        no_retrieval_tokens = re.findall(self.no_retrieval_pattern, response)
        continue_evidence_tokens = re.findall(self.continue_evidence_pattern, response)

        # Extract utility score
        utility_match = re.search(self.utility_pattern, response)
        utility_score = None
        if utility_match:
            try:
                utility_score = int(utility_match.group(0)[-2])  # Extract the number from [Utility:X]
            except (IndexError, ValueError) as e:
                logger.warning(f"Failed to parse utility score from {utility_match.group(0)}: {e}")

        return retrieval_tokens, no_retrieval_tokens, continue_evidence_tokens, utility_score

    def should_trigger_retrieval(self, response: str, has_prior_evidence: bool = False) -> bool:
        """
        Determine whether to trigger retrieval based on the response tokens.

        Args:
            response (str): The model's generated response.
            has_prior_evidence (bool): Whether prior retrieved passages exist in the session/context.

        Returns:
            bool: True if retrieval should be triggered, False otherwise.
        """
        # Extract tokens
        retrieval_tokens, no_retrieval_tokens, continue_evidence_tokens, utility_score = self.extract_tokens(response)

        # Log the extracted tokens for debugging
        logger.info(f"Extracted tokens: Retrieval={len(retrieval_tokens)}, NoRetrieval={len(no_retrieval_tokens)}, "
                    f"ContinueEvidence={len(continue_evidence_tokens)}, Utility={utility_score}")

        # Rule 1: If [Retrieval] is present, trigger retrieval
        if retrieval_tokens:
            logger.info("Found [Retrieval] token(s). Triggering retrieval.")
            return True

        # Rule 2: If [Continue to Use Evidence] is present and prior evidence exists, don't trigger new retrieval
        if continue_evidence_tokens and has_prior_evidence:
            logger.info("Found [Continue to Use Evidence] and prior evidence exists. Not triggering new retrieval.")
            return False
        elif continue_evidence_tokens:
            logger.info("Found [Continue to Use Evidence] but no prior evidence. Treating as [Retrieval].")
            return True

        # Rule 3: If no [Retrieval] but [No Retrieval] is present, don't trigger retrieval
        if no_retrieval_tokens:
            logger.info("Found [No Retrieval] token(s) and no [Retrieval]. Not triggering retrieval.")
            return False

        # Rule 4: If neither [Retrieval] nor [No Retrieval] is present, trigger retrieval as a precaution
        logger.info("No [Retrieval] or [No Retrieval] tokens found. Triggering retrieval as precaution.")
        return True

# Example usage
if __name__ == "__main__":
    extractor = SelfRAGTokenExtractor()

    # Test cases from your logs
    test_responses = [
        "The capital of the Netherlands is Amsterdam.[Utility:5]",
        "Hanoi.[Utility:5]",
        "Yes, I am aware of Ashish Shai.[No Retrieval]He is a professor of economics at Maastricht University in the Netherlands.<paragraph>I am an AI language model and do not have personal knowledge or experience.[No Retrieval]I can provide information about Ashish Shai and his work, but I do not have any personal knowledge or experience with him.[Utility:5]"
    ]

    for response in test_responses:
        print(f"\nTesting response: {response}")
        should_retrieve = extractor.should_trigger_retrieval(response, has_prior_evidence=False)
        print(f"Should trigger retrieval: {should_retrieve}")