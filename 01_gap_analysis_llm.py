"""
Gap Analysis for Multi-hop RAG Reasoning Trajectories (with LLM-based Hop2 Query)

This script analyzes the credit assignment confusion problem in multi-hop RAG systems
by comparing retrieval quality (ECS) with final answer correctness.

Dependencies (install via uv):
    uv pip install datasets rank-bm25 numpy matplotlib tqdm openai scikit-learn
"""

import json
import os
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
# [MODIFIED] More precise ECS thresholds for fine-grained analysis
ECS_PERFECT = 1.0      # Perfect retrieval: all gold documents retrieved
ECS_PARTIAL = 0.5      # Partial retrieval: some gold documents retrieved
ECS_FAILED = 0.0       # Failed retrieval: no gold documents retrieved

# For backward compatibility
ECS_BAD_THRESHOLD = ECS_PERFECT  # Anything < 1.0 is considered imperfect
ECS_GOOD_THRESHOLD = ECS_PARTIAL  # >= 0.5 is considered acceptable

# [MODIFIED] Qwen API Configuration
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-9233a091f6d44f79bdddf896e24cd141")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"  # Using qwen-turbo for cost efficiency

qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)


def preprocess_text(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    return text.lower().split()


def get_paragraph_text(context: Dict, title: str, sent_id: int = None) -> str:
    """Extract paragraph text from context."""
    try:
        title_idx = context["title"].index(title)
        sentences = context["sentences"][title_idx]
        if sent_id is not None:
            return sentences[sent_id] if sent_id < len(sentences) else ""
        return " ".join(sentences)
    except (ValueError, IndexError):
        return ""


def build_corpus(context: Dict) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Build corpus from context paragraphs."""
    corpus = []
    metadata = []

    for title, sentences in zip(context["title"], context["sentences"]):
        paragraph_text = " ".join(sentences)
        corpus.append(paragraph_text)
        metadata.append((title, len(sentences)))

    return corpus, metadata


def generate_hop2_query(question: str, hop1_docs: List[str]) -> str:
    """Use Qwen to generate second hop query based on first hop results."""
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
        response = qwen_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=64,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Qwen API 调用失败，使用原始问题作为 fallback: {e}")
        return question


def is_query_degenerate(original_q: str, hop2_query: str,
                         threshold: float = 0.85) -> Tuple[bool, float]:
    if original_q.strip().lower() == hop2_query.strip().lower():
        return True, 1.0
    try:
        vec = TfidfVectorizer().fit_transform([original_q, hop2_query])
        sim = float(cosine_similarity(vec[0], vec[1])[0][0])
        return sim > threshold, sim
    except Exception:
        return False, 0.0


def simulate_two_hop_retrieval(
    question: str,
    context: Dict,
    top_k: int = 2,
    use_llm_hop2: bool = True
) -> Tuple[Tuple[List[str], List[str], List[float]], Tuple[List[str], List[str], List[float]], str]:
    """
    Simulate two-hop retrieval.

    Returns:
        ((hop1_docs, hop1_titles, hop1_scores), (hop2_docs, hop2_titles, hop2_scores), query_hop2)
    """
    corpus, metadata = build_corpus(context)

    tokenized_corpus = [preprocess_text(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = preprocess_text(question)
    scores = bm25.get_scores(query_tokens)
    top_indices_hop1 = np.argsort(scores)[::-1][:top_k]

    hop1_docs = [corpus[i] for i in top_indices_hop1]
    hop1_titles = [metadata[i][0] for i in top_indices_hop1]
    hop1_scores = [scores[i] for i in top_indices_hop1]

    if use_llm_hop2:
        query_hop2 = generate_hop2_query(question, hop1_docs)
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


def calculate_ecs(retrieved_titles: List[str],
                  retrieved_scores: List[float],
                  supporting_facts: Dict) -> float:
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


def categorize_ecs(ecs: float) -> str:
    """
    Categorize ECS into fine-grained levels.

    Returns:
        - "Perfect (1.0)": All gold documents retrieved
        - "Partial (0.5-1.0)": Some but not all gold documents retrieved
        - "Minimal (0.0-0.5)": Very few gold documents retrieved
        - "Failed (0.0)": No gold documents retrieved
    """
    if ecs == 1.0:
        return "Perfect (1.0)"
    elif ecs >= 0.5:
        return "Partial (0.5-1.0)"
    elif ecs > 0.0:
        return "Minimal (0.0-0.5)"
    else:
        return "Failed (0.0)"


def categorize_hop_pair(ecs_hop1: float, ecs_hop2: float) -> str:
    """
    Categorize the quality of a two-hop retrieval pair.

    Returns a descriptive label for the retrieval pattern.
    """
    cat1 = categorize_ecs(ecs_hop1)
    cat2 = categorize_ecs(ecs_hop2)

    if ecs_hop1 == 1.0 and ecs_hop2 == 1.0:
        return "Both Perfect"
    elif ecs_hop1 >= 0.5 and ecs_hop2 >= 0.5:
        if ecs_hop1 == 0.5 or ecs_hop2 == 0.5:
            return "Partial Success (含0.5)"
        else:
            return "Good Quality (≥0.5)"
    elif ecs_hop1 == 0.0 and ecs_hop2 == 0.0:
        return "Both Failed"
    else:
        return "Mixed Quality"


def check_answer_correctness(answer: str, retrieved_docs: List[str],
                              strict: bool = False) -> bool:
    answer_lower = answer.lower().strip()
    if not retrieved_docs:
        return False
    if strict:
        top_doc = retrieved_docs[0].lower()
        return answer_lower in top_doc
    else:
        combined_text = " ".join(retrieved_docs).lower()
        return answer_lower in combined_text


def analyze_sample(sample: Dict, use_llm_hop2: bool = True, sample_idx: int = 0) -> Dict:
    """Analyze a single sample."""
    question = sample["question"]
    answer = sample["answer"]
    supporting_facts = sample["supporting_facts"]
    context = sample["context"]

    (hop1_docs, hop1_titles, hop1_scores), (hop2_docs, hop2_titles, hop2_scores), query_hop2 = simulate_two_hop_retrieval(
        question, context, top_k=2, use_llm_hop2=use_llm_hop2
    )

    ecs_hop1 = calculate_ecs(hop1_titles, hop1_scores, supporting_facts)
    ecs_hop2 = calculate_ecs(hop2_titles, hop2_scores, supporting_facts)

    all_docs = hop1_docs + hop2_docs
    is_correct_loose = check_answer_correctness(answer, all_docs, strict=False)
    is_correct_strict = check_answer_correctness(answer, all_docs, strict=True)

    result = {
        "question": question,
        "answer": answer,
        "is_correct": is_correct_loose,
        "is_correct_strict": is_correct_strict,
        "ecs_hop1": round(ecs_hop1, 4),
        "ecs_hop2": round(ecs_hop2, 4),
        "hop1_titles": hop1_titles,
        "hop2_titles": hop2_titles,
        "hop1_scores": [round(s, 3) for s in hop1_scores],
        "hop2_scores": [round(s, 3) for s in hop2_scores],
        "gold_titles": supporting_facts["title"],
        "query_hop2_llm": query_hop2 if use_llm_hop2 else None,
        "hop2_same_docs_as_hop1": set(hop2_titles) == set(hop1_titles),
    }

    if use_llm_hop2:
        is_degen, degen_sim = is_query_degenerate(question, query_hop2)
        result["query_hop2_same_as_q"] = is_degen
        result["query_hop2_degen_sim"] = round(degen_sim, 3)
    else:
        result["query_hop2_same_as_q"] = None
        result["query_hop2_degen_sim"] = None

    return result


def main():
    print("=" * 60)
    print("多跳 RAG 信用分配混淆分析 (LLM Hop2)".center(60))
    print("=" * 60)
    print()

    print("[步骤 1/5] 加载 HotpotQA 数据集...")
    print("  ├─ 数据源: Hugging Face hotpot_qa/distractor")
    print("  ├─ 分割: validation")
    print("  └─ 正在下载和加载数据...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"  ✓ 数据集加载完成，共 {len(dataset)} 条样本")
    print()

    random.seed(42)
    sample_size = 500
    print(f"[步骤 2/5] 随机采样 {sample_size} 条数据...")
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    samples = [dataset[i] for i in indices]
    print(f"  ✓ 采样完成，实际样本数: {len(samples)}")
    print()

    USE_LLM_HOP2 = True

    print(f"[步骤 3/5] 模拟两跳检索并计算 ECS...")
    print(f"  ├─ 检索模式: {'LLM 生成第二跳 Query' if USE_LLM_HOP2 else 'BM25 增强 Query'}")
    print("  ├─ 每个样本执行两跳检索")
    print("  ├─ 计算每跳的 ECS（检索质量）")
    print("  └─ 判断答案正确性（EM 代理）")
    print()

    results = []
    desc = "  处理进度（含LLM调用，约5~8分钟）" if USE_LLM_HOP2 else "  处理进度"
    for i, sample in enumerate(tqdm(samples, desc=desc, ncols=80)):
        result = analyze_sample(sample, use_llm_hop2=USE_LLM_HOP2, sample_idx=i)
        results.append(result)

        if (i + 1) % 100 == 0:
            correct_so_far = sum(1 for r in results if r["is_correct"])
            print(f"  └─ 已处理 {i+1}/{len(samples)} 条，当前正确率: {correct_so_far/(i+1)*100:.1f}%")

    print(f"  ✓ 检索分析完成")
    print()

    print("[步骤 4/5] 统计分析...")
    print("  ├─ 计算答案正确率")
    correct_samples = [r for r in results if r["is_correct"]]
    incorrect_samples = [r for r in results if not r["is_correct"]]
    correct_samples_strict = [r for r in results if r["is_correct_strict"]]
    incorrect_samples_strict = [r for r in results if not r["is_correct_strict"]]

    total = len(results)
    correct_count = len(correct_samples)
    incorrect_count = len(incorrect_samples)
    correct_count_strict = len(correct_samples_strict)
    incorrect_count_strict = len(incorrect_samples_strict)
    print(f"  ├─ 答案正确（宽松）: {correct_count} 题 ({correct_count/total*100:.1f}%)")
    print(f"  ├─ 答案正确（严格）: {correct_count_strict} 题 ({correct_count_strict/total*100:.1f}%)")
    print(f"  ├─ 答案错误（宽松）: {incorrect_count} 题 ({incorrect_count/total*100:.1f}%)")

    if USE_LLM_HOP2:
        llm_degraded_count = sum(1 for r in results if r.get("query_hop2_same_as_q", False))
        llm_degraded_ratio = (llm_degraded_count / total * 100) if total > 0 else 0
        print(f"  ├─ LLM Query 退化率（hop2 query ≈ 原始问题）: {llm_degraded_count} 题 ({llm_degraded_ratio:.1f}%)")

        hop2_same_docs_count = sum(1 for r in results if r.get("hop2_same_docs_as_hop1", False))
        hop2_same_docs_ratio = (hop2_same_docs_count / total * 100) if total > 0 else 0
        print(f"  ├─ Hop2 检索结果与 Hop1 完全相同: {hop2_same_docs_count} 题 ({hop2_same_docs_ratio:.1f}%)")

    print(f"  ├─ 细粒度 ECS 分析")

    hop1_categories = Counter([categorize_ecs(r["ecs_hop1"]) for r in results])
    hop2_categories = Counter([categorize_ecs(r["ecs_hop2"]) for r in results])
    pair_categories = Counter([categorize_hop_pair(r["ecs_hop1"], r["ecs_hop2"]) for r in results])

    print(f"  ├─ 分析 Bad Hop（ECS < {ECS_BAD_THRESHOLD}）分布")
    def has_bad_hop(result):
        return result["ecs_hop1"] < ECS_BAD_THRESHOLD or result["ecs_hop2"] < ECS_BAD_THRESHOLD

    correct_bad_hop = sum(1 for r in correct_samples if has_bad_hop(r))
    incorrect_bad_hop = sum(1 for r in incorrect_samples if has_bad_hop(r))

    def has_perfect_retrieval(result):
        return result["ecs_hop1"] == 1.0 and result["ecs_hop2"] == 1.0

    def has_partial_retrieval(result):
        return (result["ecs_hop1"] >= 0.5 or result["ecs_hop2"] >= 0.5) and not has_perfect_retrieval(result)

    def has_weak_retrieval(result):
        return (result["ecs_hop1"] > 0.0 and result["ecs_hop1"] < 0.5) and (result["ecs_hop2"] > 0.0 and result["ecs_hop2"] < 0.5)

    def has_failed_retrieval(result):
        return result["ecs_hop1"] == 0.0 or result["ecs_hop2"] == 0.0

    perfect_count = sum(1 for r in results if has_perfect_retrieval(r))
    partial_count = sum(1 for r in results if has_partial_retrieval(r))
    weak_count = sum(1 for r in results if has_weak_retrieval(r))
    failed_count = sum(1 for r in results if has_failed_retrieval(r))

    print("  ├─ 识别幸存者偏差样本")
    survivor_bias_count = correct_bad_hop
    survivor_bias_ratio = (survivor_bias_count / correct_count * 100) if correct_count > 0 else 0

    print(f"  ├─ 识别隐性失败样本（所有跳 ECS >= {ECS_GOOD_THRESHOLD}）")
    def all_hops_good(result):
        return result["ecs_hop1"] >= ECS_GOOD_THRESHOLD and result["ecs_hop2"] >= ECS_GOOD_THRESHOLD

    hidden_failure_count = sum(1 for r in incorrect_samples if all_hops_good(r))
    hidden_failure_ratio = (hidden_failure_count / incorrect_count * 100) if incorrect_count > 0 else 0

    correct_bad_hop_ratio = (correct_bad_hop / correct_count * 100) if correct_count > 0 else 0
    incorrect_bad_hop_ratio = (incorrect_bad_hop / incorrect_count * 100) if incorrect_count > 0 else 0
    print("  ✓ 统计分析完成")
    print()

    print("[步骤 5/5] 保存结果和生成可视化...")
    output_data = {
        "total_samples": total,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "correct_ratio": correct_count / total * 100,
        "correct_count_strict": correct_count_strict,
        "incorrect_count_strict": incorrect_count_strict,
        "correct_ratio_strict": correct_count_strict / total * 100,
        "survivor_bias_count": survivor_bias_count,
        "survivor_bias_ratio": survivor_bias_ratio,
        "hidden_failure_count": hidden_failure_count,
        "hidden_failure_ratio": hidden_failure_ratio,
        "correct_bad_hop_ratio": correct_bad_hop_ratio,
        "incorrect_bad_hop_ratio": incorrect_bad_hop_ratio,
        "use_llm_hop2": USE_LLM_HOP2,
        "ecs_fine_grained": {
            "perfect_retrieval_count": perfect_count,
            "perfect_retrieval_ratio": perfect_count / total * 100,
            "partial_retrieval_count": partial_count,
            "partial_retrieval_ratio": partial_count / total * 100,
            "weak_retrieval_count": weak_count,
            "weak_retrieval_ratio": weak_count / total * 100,
            "failed_retrieval_count": failed_count,
            "failed_retrieval_ratio": failed_count / total * 100,
            "hop1_categories": dict(hop1_categories),
            "hop2_categories": dict(hop2_categories),
            "pair_categories": dict(pair_categories)
        }
    }

    if USE_LLM_HOP2:
        output_data["llm_degraded_count"] = llm_degraded_count
        output_data["llm_degraded_ratio"] = llm_degraded_ratio
        output_data["hop2_same_docs_count"] = hop2_same_docs_count
        output_data["hop2_same_docs_ratio"] = hop2_same_docs_ratio

    with open("gap_analysis_results_llm.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print("  ✓ 结果已保存到 gap_analysis_results_llm.json")

    # Save detailed sample data
    with open("gap_analysis_samples_detailed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  ✓ 详细样本数据已保存到 gap_analysis_samples_detailed.json")

    print("\n" + "=" * 50)
    print("GAP ANALYSIS SUMMARY".center(50))
    print("=" * 50)
    print(f"总样本数: {total}")
    print(f"检索模式: {'LLM 生成第二跳 Query' if USE_LLM_HOP2 else 'BM25 增强 Query'}")
    print(f"答案正确率（宽松EM）: {correct_count} 题 ({correct_count / total * 100:.1f}%)")
    print(f"答案正确率（严格EM）: {correct_count_strict} 题 ({correct_count_strict / total * 100:.1f}%)")
    print()

    print("[评估说明]")
    print("⚠ EM 代理使用字符串匹配（答案词是否出现在检索文档中）")
    print("  真实 RAG 系统的 EM 会更低，因为 LLM 可能推理失败")
    print("  但相对提升的方向性结论仍然有效")
    print()

    print("[检索质量细分]")
    print(f"✓ 完美检索（两跳都 ECS=1.0）: {perfect_count} 题 ({perfect_count/total*100:.1f}%)")
    print(f"✓ 部分成功（至少一跳 ECS≥0.5）: {partial_count} 题 ({partial_count/total*100:.1f}%)")
    print(f"✓ 微弱检索（两跳 0<ECS<0.5）: {weak_count} 题 ({weak_count/total*100:.1f}%)")
    print(f"✓ 完全失败（至少一跳 ECS=0.0）: {failed_count} 题 ({failed_count/total*100:.1f}%)")
    print()

    if USE_LLM_HOP2:
        print("[LLM Query 质量]")
        print(f"✓ LLM Query 退化率（hop2 query ≈ 原始问题）: {llm_degraded_count} 题 ({llm_degraded_ratio:.1f}%)")
        print(f"✓ Hop2 检索结果与 Hop1 完全相同: {hop2_same_docs_count} 题 ({hop2_same_docs_ratio:.1f}%)")
        print()

    print("[信用分配混淆证据]")
    print(f"✓ 幸存者偏差样本（答案对 但检索有Bad Hop）: {survivor_bias_count} 题 ({survivor_bias_ratio:.1f}%)")
    print(f"✓ 隐性失败样本（答案错 但检索全部正常）: {hidden_failure_count} 题 ({hidden_failure_ratio:.1f}%)")
    print()
    print("[各组 Bad Hop 比例]")
    print(f"答案正确组 Bad Hop 率: {correct_bad_hop_ratio:.1f}%")
    print(f"答案错误组 Bad Hop 率: {incorrect_bad_hop_ratio:.1f}%")
    print()

    if survivor_bias_ratio > 15:
        conclusion = "Gap 显著，继续实验"
    else:
        conclusion = "Gap 不明显，请检查数据"
    print(f"结论: {conclusion}")
    print("=" * 50)

    print("  ├─ 生成 ECS 分布对比图...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    correct_ecs = [r["ecs_hop1"] for r in correct_samples] + [r["ecs_hop2"] for r in correct_samples]
    incorrect_ecs = [r["ecs_hop1"] for r in incorrect_samples] + [r["ecs_hop2"] for r in incorrect_samples]

    axes[0].hist(correct_ecs, bins=20, alpha=0.7, color="green", edgecolor="black")
    axes[0].set_xlabel("ECS")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("ECS Distribution (Correct Answers)")
    axes[0].axvline(x=ECS_BAD_THRESHOLD, color="red", linestyle="--", label="Bad Hop Threshold")
    axes[0].legend()

    axes[1].hist(incorrect_ecs, bins=20, alpha=0.7, color="red", edgecolor="black")
    axes[1].set_xlabel("ECS")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("ECS Distribution (Incorrect Answers)")
    axes[1].axvline(x=ECS_BAD_THRESHOLD, color="red", linestyle="--", label="Bad Hop Threshold")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("ecs_distribution_comparison_llm.png", dpi=300, bbox_inches="tight")
    print("  ✓ 保存图表: ecs_distribution_comparison_llm.png")

    print("  ├─ 生成 ECS 箱线图...")
    fig, ax = plt.subplots(figsize=(10, 6))

    hop1_ecs = [r["ecs_hop1"] for r in results]
    hop2_ecs = [r["ecs_hop2"] for r in results]

    box_data = [hop1_ecs, hop2_ecs]
    ax.boxplot(box_data, labels=["Hop 1", "Hop 2"])
    ax.set_ylabel("ECS")
    ax.set_title("ECS Distribution by Hop")
    ax.axhline(y=ECS_BAD_THRESHOLD, color="red", linestyle="--", label="Bad Hop Threshold")
    ax.axhline(y=ECS_GOOD_THRESHOLD, color="orange", linestyle="--", label="Good Hop Threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("ecs_boxplot_by_hop_llm.png", dpi=300, bbox_inches="tight")
    print("  ✓ 保存图表: ecs_boxplot_by_hop_llm.png")
    print()


if __name__ == "__main__":
    main()
