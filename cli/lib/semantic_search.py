from sentence_transformers import SentenceTransformer


class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.max_sequence_length = 0

    def verify(self):
        """Verify the model information"""
        verify_model()

    
def verify_model():
    model = SemanticSearch()

    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.max_sequence_length}")

        