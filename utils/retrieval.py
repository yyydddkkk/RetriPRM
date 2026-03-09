"""
Retrieval utilities for BM25-based multi-hop retrieval.
"""

from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi


def preprocess_text(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    return text.lower().split()


def build_corpus(context: Dict) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Build corpus from context paragraphs.

    Args:
        context: HotpotQA context dict with 'title' and 'sentences' fields

    Returns:
        corpus: List of paragraph texts
        metadata: List of (title, num_sentences) tuples
    """
    corpus = []
    metadata = []

    for title, sentences in zip(context["title"], context["sentences"]):
        paragraph_text = " ".join(sentences)
        corpus.append(paragraph_text)
        metadata.append((title, len(sentences)))

    return corpus, metadata


def simulate_two_hop_retrieval(
    question: str,
    context: Dict,
    generate_hop2_query_func,
    top_k: int = 2,
    use_llm_hop2: bool = True
) -> Tuple[Tuple[List[str], List[str], List[float]], Tuple[List[str], List[str], List[float]], str]:
    """
    Simulate two-hop BM25 retrieval.

    Args:
        question: Original question
        context: HotpotQA context dict
        generate_hop2_query_func: Function to generate hop2 query using LLM
        top_k: Number of documents to retrieve per hop
        use_llm_hop2: Whether to use LLM for hop2 query generation

    Returns:
        (hop1_docs, hop1_titles, hop1_scores): First hop results
        (hop2_docs, hop2_titles, hop2_scores): Second hop results
        query_hop2: Generated query for second hop
    """
    corpus, metadata = build_corpus(context)

    tokenized_corpus = [preprocess_text(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # First hop: retrieve based on question
    query_tokens = preprocess_text(question)
    scores = bm25.get_scores(query_tokens)
    top_indices_hop1 = np.argsort(scores)[::-1][:top_k]

    hop1_docs = [corpus[i] for i in top_indices_hop1]
    hop1_titles = [metadata[i][0] for i in top_indices_hop1]
    hop1_scores = [scores[i] for i in top_indices_hop1]

    # Second hop: use LLM or BM25 augmented query
    if use_llm_hop2:
        query_hop2 = generate_hop2_query_func(question, hop1_docs)
        query_hop2_tokens = preprocess_text(query_hop2)
    else:
        query_hop2 = question + " " + " ".join(hop1_docs)
        query_hop2_tokens = preprocess_text(query_hop2)

    scores = bm25.get_scores(query_hop2_tokens)
    top_indices_hop2 = np.argsort(scores)[::-1][:top_k]

    hop2_docs = [corpus[i] for i in top_indices_hop2]
    hop2_titles = [metadata[i][0] for i in top_indices_hop2]
    hop2_scores = [scores[i] for i in top_indices_hop2]

    return (hop1_docs, hop1_titles, hop1_scores), (hop2_docs, hop2_titles, hop2_scores), query_hop2
