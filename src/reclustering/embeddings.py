import numpy as np
from .encoders import get_encoder

def compute_embeddings(texts: list[str], encoder_name: str) -> np.ndarray:
    model = get_encoder(encoder_name)
    embs = model.encode(texts, show_progress_bar=True)
    return np.array(embs)
