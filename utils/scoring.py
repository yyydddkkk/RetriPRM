"""
Scoring utilities for ECS calculation and trajectory evaluation.
"""

from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_ecs(
    retrieved_titles: List[str],
    retrieved_scores: List[float],
    supporting_facts: Dict
) -> float:
    """
    Calculate Expected Coverage Score (ECS) with BM25 soft scoring.

    Args:
        retrieved_titles: List of retrieved document titles
        retrieved_scores: List of BM25 scores for retrieved documents
        supporting_facts: Dict with 'title' field containing gold document titles

    Returns:
        ECS score in [0, 1]
    """
    gold_titles = set(supporting_facts["title"])
    if len(gold_titles) == 0:
        return 0.0

    total_score = 0.0
    for i, title in enumerate(retrieved_titles):
        bm25_score = retrieved_scores[i] if i < len(retrieved_scores) else 0.0
        normalized = min(bm25_score / 30.0, 0.2)
        is_gold = title in gold_titles

        if is_gold:
            total_score += 1.0
        else:
            total_score += normalized

    ecs = min(total_score / len(gold_titles), 1.0)
    return ecs


def score_trajectory(
    hop1_titles: List[str],
    hop1_scores: List[float],
    hop2_titles: List[str],
    hop2_scores: List[float],
    supporting_facts: Dict,
    w1: float = 0.4,
    w2: float = 0.6
) -> Dict[str, float]:
    """
    Score a complete two-hop trajectory.

    Args:
        hop1_titles: First hop retrieved titles
        hop1_scores: First hop BM25 scores
        hop2_titles: Second hop retrieved titles
        hop2_scores: Second hop BM25 scores
        supporting_facts: Gold supporting facts
        w1: Weight for hop1 ECS (default 0.4)
        w2: Weight for hop2 ECS (default 0.6)

    Returns:
        Dict with 'ecs_hop1', 'ecs_hop2', 'ecs_combined'
    """
    ecs_hop1 = calculate_ecs(hop1_titles, hop1_scores, supporting_facts)
    ecs_hop2 = calculate_ecs(hop2_titles, hop2_scores, supporting_facts)
    ecs_combined = w1 * ecs_hop1 + w2 * ecs_hop2

    return {
        "ecs_hop1": round(ecs_hop1, 4),
        "ecs_hop2": round(ecs_hop2, 4),
        "ecs_combined": round(ecs_combined, 4)
    }


def is_query_degenerate(
    original_q: str,
    hop2_query: str,
    threshold: float = 0.85
) -> Tuple[bool, float]:
    """
    Check if hop2 query is degenerate (too similar to original question).

    Args:
        original_q: Original question
        hop2_query: Generated hop2 query
        threshold: Similarity threshold (default 0.85)

    Returns:
        (is_degenerate, similarity_score)
    """
    if original_q.strip().lower() == hop2_query.strip().lower():
        return True, 1.0

    try:
        vec = TfidfVectorizer().fit_transform([original_q, hop2_query])
        sim = float(cosine_similarity(vec[0], vec[1])[0][0])
        return sim > threshold, sim
    except Exception:
        return False, 0.0


def check_answer_correctness(
    answer: str,
    retrieved_docs: List[str],
    strict: bool = False
) -> bool:
    """
    Check if answer appears in retrieved documents.

    Args:
        answer: Gold answer string
        retrieved_docs: List of retrieved document texts
        strict: If True, only check first document

    Returns:
        True if answer found in documents
    """
    answer_lower = answer.lower().strip()
    if not retrieved_docs:
        return False

    if strict:
        top_doc = retrieved_docs[0].lower()
        return answer_lower in top_doc
    else:
        combined_text = " ".join(retrieved_docs).lower()
        return answer_lower in combined_text
