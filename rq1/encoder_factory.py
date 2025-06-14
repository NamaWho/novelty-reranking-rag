
from sentence_transformers import SentenceTransformer

def get_encoder(name: str):
    """
    Return a sentence embedding model by name.
    """
    try:
        model = SentenceTransformer(name)
    except Exception as e:
        raise ValueError(f"Encoder '{name}' could not be loaded: {e}")
    return model
