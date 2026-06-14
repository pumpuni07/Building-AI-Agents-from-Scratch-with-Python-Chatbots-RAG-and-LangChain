"""
RAG with PyTorch – Cosine Similarity Exercise

Task: Modify RAG_QA() to replace dot product with cosine similarity.
"""

from typing import List, Tuple
import torch


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between each row of a and each row of b.

    a: (N, D)
    b: (M, D)
    returns: (N, M) similarity matrix
    """
    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-12)
    b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-12)
    return torch.matmul(a_norm, b_norm.T)


def RAG_QA(
    embeddings_questions: torch.Tensor,
    embeddings_responses: torch.Tensor,
    query_embedding: torch.Tensor,
    responses: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k responses using cosine similarity between query and response embeddings.

    embeddings_questions: (N, D) – question embeddings (not strictly needed here but kept for API)
    embeddings_responses: (N, D) – response embeddings aligned with 'responses'
    query_embedding: (D,) or (1, D)
    responses: list of candidate response texts
    """
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)  # (1, D)

    # Cosine similarity between query and all response embeddings
    sims = cosine_similarity_matrix(query_embedding, embeddings_responses)  # (1, N)
    sims = sims.squeeze(0)  # (N,)

    # Sort and select top-k
    topk_vals, topk_idx = torch.topk(sims, k=min(top_k, len(responses)))
    results = [(responses[i], float(topk_vals[j])) for j, i in enumerate(topk_idx)]
    return results


if __name__ == "__main__":
    # Example dummy data to illustrate the exercise
    torch.manual_seed(0)

    num_items = 5
    dim = 4
    embeddings_questions = torch.randn(num_items, dim)
    embeddings_responses = torch.randn(num_items, dim)
    responses = [
        "This song is appropriate for children.",
        "This song contains explicit language.",
        "This song references violence.",
        "This song is recommended for all ages.",
        "This song includes suggestive content.",
    ]

    # Simulate a new song query embedding
    query_embedding = torch.randn(dim)

    top_responses = RAG_QA(
        embeddings_questions,
        embeddings_responses,
        query_embedding,
        responses,
        top_k=3,
    )

    print("Top responses (cosine similarity):")
    for text, score in top_responses:
        print(f"{score:.3f} -> {text}")
