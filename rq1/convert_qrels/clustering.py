import numpy as np
from sklearn.cluster import AgglomerativeClustering

def reassign_clusters(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """
    Performs agglomerative clustering with cosine affinity and a distance threshold.
    Returns an array of new cluster labels.
    """
    # convert cosine similarity threshold to distance threshold (1 - sim)
    distance_threshold = 1.0 - threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='complete',
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(embeddings)
    return labels
