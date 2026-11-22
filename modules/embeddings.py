
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    """
    Load the embedding model once and reuse it.
    """
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Embedding model loaded successfully!")

    return model


def generate_embeddings(texts):
    """
    Convert a list of text strings into embeddings.

    Returns:
        NumPy array with shape (number_of_texts, 384).
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list.")

    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if not all(
        isinstance(text, str) and text.strip()
        for text in texts
    ):
        raise ValueError(
            "All texts must be non-empty strings."
        )

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


def generate_query_embedding(query):
    """
    Generate an embedding for a user question.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    return generate_embeddings([query])[0]
