import numpy as np

from config import Config
from encoder_factory import get_encoder


def compute_embeddings(sentences: list[str], encoder_name: str) -> np.ndarray:
    """
    Compute embeddings for a list of sentences using the specified encoder.
    """
    encoder = get_encoder(encoder_name)
    embeddings = encoder.encode(sentences, show_progress_bar=True)
    return np.array(embeddings)


def apply_threshold(similarity_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply a similarity threshold to filter pairs.
    Returns a boolean mask of shape same as similarity_matrix.
    """
    return similarity_matrix >= threshold


def build_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Build a cosine similarity matrix from embeddings.
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    sim_matrix = np.dot(normalized, normalized.T)
    return sim_matrix


def run_pipeline(
    sentences: list[str],
    encoder_name: str,
    threshold: float
) -> dict:
    """
    Full pipeline: compute embeddings, similarity matrix, apply threshold.
    Returns a dict with embeddings, sim_matrix, mask.
    """
    embeddings = compute_embeddings(sentences, encoder_name)
    sim_matrix = build_similarity_matrix(embeddings)
    mask = apply_threshold(sim_matrix, threshold)
    return {
        "embeddings": embeddings,
        "similarity_matrix": sim_matrix,
        "mask": mask,
    }