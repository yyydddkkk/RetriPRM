"""
LLM utilities for Qwen API interaction.
"""

import os
from typing import List
from openai import OpenAI


def get_qwen_client(api_key: str = None, base_url: str = None, model: str = None):
    """
    Initialize Qwen API client.

    Args:
        api_key: Qwen API key (default from env QWEN_API_KEY)
        base_url: API base URL (default Qwen compatible mode)
        model: Model name (default qwen-turbo)

    Returns:
        Configured OpenAI client and model name
    """
    api_key = api_key or os.getenv("QWEN_API_KEY", "sk-9233a091f6d44f79bdddf896e24cd141")
    base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = model or "qwen-turbo"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    return client, model


def generate_hop2_query(
    question: str,
    hop1_docs: List[str],
    client: OpenAI = None,
    model: str = None,
    temperature: float = 0.7
) -> str:
    """
    Use Qwen to generate second hop query based on first hop results.

    Args:
        question: Original question
        hop1_docs: Documents retrieved in first hop
        client: OpenAI client (will create if None)
        model: Model name (default qwen-turbo)
        temperature: Sampling temperature (default 0.7)

    Returns:
        Generated query for second hop
    """
    if client is None:
        client, model = get_qwen_client()

    # Truncate each doc to first 150 words to avoid token limit
    context_text = " ".join(" ".join(doc.split()[:150]) for doc in hop1_docs)

    system_prompt = """You are a multi-hop reasoning assistant.
Given the original question and the documents retrieved in the first hop,
generate a focused search query for the second hop retrieval.
The second hop query should target the MISSING information needed to answer the question,
NOT repeat what was already found in the first hop.
Return ONLY the search query, no explanation."""

    user_message = f"""Original question: {question}

First hop retrieved documents:
{context_text}

What should the second hop search query be?"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=64,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Qwen API 调用失败: {e}")
        return question
