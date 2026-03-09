"""
Best-of-N (BoN) Selection Strategy Comparison Experiment - Parallel Version

并行化策略：
- 将 500 题分成 10 个 shard（每个 50 题）
- 使用 multiprocessing 并行处理多个 shard
- 支持断点续传（已完成的 shard 会跳过）
- 最后自动合并所有 shard 结果

预计费用：约 15-20 元（qwen-turbo）
预计时间：约 8-15 分钟（4-8 进程并行）

Dependencies: datasets, rank-bm25, numpy, matplotlib, tqdm, openai, scikit-learn
"""

import json
import os
import random
import time
import signal
import sys
from collections import Counter
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Qwen API Configuration
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-9233a091f6d44f79bdddf896e24cd141")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"

# Experiment Configuration
N_VALUES = [1, 2, 4, 8]
SAMPLE_SIZE = 500
MAX_N = 8  # Generate 8 trajectories per question
NUM_SHARDS = 10  # Split into 10 shards
NUM_WORKERS = 8  # Number of parallel workers (adjust based on API limit)

# Output directories
SHARD_DIR = Path("data/shards")
SHARD_DIR.mkdir(parents=True, exist_ok=True)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n\n⚠ 收到中断信号，正在安全退出...")
    print("  已完成的 shard 会保存，下次运行会自动跳过")
    shutdown_requested = True
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_qwen_client():
    """Initialize Qwen API client."""
    return OpenAI(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )


def preprocess_text(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    return text.lower().split()


def build_corpus(context: Dict) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Build corpus from context paragraphs."""
    corpus = []
    metadata = []

    for title, sentences in zip(context["title"], context["sentences"]):
        paragraph_text = " ".join(sentences)
        corpus.append(paragraph_text)
        metadata.append((title, len(sentences)))

    return corpus, metadata


def generate_hop2_query(question: str, hop1_docs: List[str], client: OpenAI) -> str:
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
        response = client.chat.completions.create(
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
        print(f"  ⚠ Qwen API 调用失败: {e}")
        return question


def simulate_two_hop_retrieval(
    question: str,
    context: Dict,
    client: OpenAI,
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
        query_hop2 = generate_hop2_query(question, hop1_docs, client)
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
    """Calculate Expected Coverage Score (ECS)."""
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


def check_answer_correctness(answer: str, retrieved_docs: List[str],
                              strict: bool = False) -> bool:
    """Check if answer appears in retrieved documents."""
    answer_lower = answer.lower().strip()
    if not retrieved_docs:
        return False
    if strict:
        top_doc = retrieved_docs[0].lower()
        return answer_lower in top_doc
    else:
        combined_text = " ".join(retrieved_docs).lower()
        return answer_lower in combined_text


def generate_trajectory(question: str, answer: str, context: Dict,
                        supporting_facts: Dict, client: OpenAI) -> Dict:
    """Generate a single trajectory for a question."""
    (hop1_docs, hop1_titles, hop1_scores), (hop2_docs, hop2_titles, hop2_scores), query_hop2 = \
        simulate_two_hop_retrieval(question, context, client, top_k=2, use_llm_hop2=True)

    ecs_hop1 = calculate_ecs(hop1_titles, hop1_scores, supporting_facts)
    ecs_hop2 = calculate_ecs(hop2_titles, hop2_scores, supporting_facts)
    ecs_combined = 0.4 * ecs_hop1 + 0.6 * ecs_hop2

    all_docs = hop1_docs + hop2_docs
    answer_loose = check_answer_correctness(answer, all_docs, strict=False)
    answer_strict = check_answer_correctness(answer, hop2_docs, strict=True) if hop2_docs else False

    return {
        "hop1_docs": hop1_docs,
        "hop1_titles": hop1_titles,
        "hop1_scores": [round(s, 3) for s in hop1_scores],
        "hop2_docs": hop2_docs,
        "hop2_titles": hop2_titles,
        "hop2_scores": [round(s, 3) for s in hop2_scores],
        "query_hop2": query_hop2,
        "ecs_hop1": round(ecs_hop1, 4),
        "ecs_hop2": round(ecs_hop2, 4),
        "ecs_combined": round(ecs_combined, 4),
        "answer_loose": answer_loose,
        "answer_strict": answer_strict,
    }


# ============================================================================
# Selection Strategies
# ============================================================================

def select_random(trajectories: List[Dict]) -> Dict:
    """Random selection baseline."""
    return random.choice(trajectories)


def select_majority(trajectories: List[Dict]) -> Dict:
    """Vote by document set frequency (corrected implementation)."""
    # Count frequency of each unique document set
    traj_signature_counter = Counter()
    traj_map = {}

    for traj in trajectories:
        # Create signature from sorted document titles
        signature = tuple(sorted(traj["hop1_titles"] + traj["hop2_titles"]))
        traj_signature_counter[signature] += 1
        if signature not in traj_map:
            traj_map[signature] = traj

    # Select the most common document set
    most_common_signature = traj_signature_counter.most_common(1)[0][0]
    return traj_map[most_common_signature]


def select_ecs_bon(trajectories: List[Dict]) -> Dict:
    """Select by highest ECS score."""
    return max(trajectories, key=lambda t: t["ecs_combined"])


def select_ecs_majority(trajectories: List[Dict]) -> Dict:
    """ECS filtering + majority vote."""
    k = max(2, len(trajectories) // 2)
    sorted_trajs = sorted(trajectories, key=lambda t: t["ecs_combined"], reverse=True)
    top_k = sorted_trajs[:k]
    return select_majority(top_k)


def select_oracle(trajectories: List[Dict]) -> Dict:
    """Oracle: select correct trajectory if exists (upper bound)."""
    strict_correct = [t for t in trajectories if t["answer_strict"]]
    if strict_correct:
        return strict_correct[0]

    loose_correct = [t for t in trajectories if t["answer_loose"]]
    if loose_correct:
        return loose_correct[0]

    return select_random(trajectories)


# ============================================================================
# Parallel Processing
# ============================================================================

def process_shard(shard_info: Tuple[int, List[Dict]]) -> str:
    """
    Process a single shard of questions.

    Args:
        shard_info: (shard_id, samples)

    Returns:
        Path to saved shard result file
    """
    shard_id, samples = shard_info
    shard_file = SHARD_DIR / f"bon_shard_{shard_id:02d}.json"

    # Skip if already processed
    if shard_file.exists():
        print(f"  ✓ Shard {shard_id} already exists, skipping")
        return str(shard_file)

    print(f"  → Processing shard {shard_id} ({len(samples)} questions)...")

    # Initialize client for this worker
    client = get_qwen_client()

    all_question_data = []

    try:
        for q_idx, sample in enumerate(samples):
            question = sample["question"]
            answer = sample["answer"]
            context = sample["context"]
            supporting_facts = sample["supporting_facts"]

            # Generate MAX_N trajectories for this question
            trajectories = []
            for traj_idx in range(MAX_N):
                traj = generate_trajectory(question, answer, context, supporting_facts, client)
                trajectories.append(traj)

                # Rate limiting: sleep after every 5 API calls
                if (traj_idx + 1) % 5 == 0:
                    time.sleep(0.5)

            all_question_data.append({
                "question": question,
                "answer": answer,
                "gold_titles": supporting_facts["title"],
                "trajectories": trajectories
            })

            # Progress update every 10 questions
            if (q_idx + 1) % 10 == 0:
                print(f"    Shard {shard_id}: {q_idx + 1}/{len(samples)} questions done")

    except KeyboardInterrupt:
        print(f"\n  ⚠ Shard {shard_id} interrupted, saving partial results...")
        # Save partial results if any
        if all_question_data:
            shard_data = {
                "shard_id": shard_id,
                "num_questions": len(all_question_data),
                "per_question": all_question_data,
                "partial": True
            }
            partial_file = SHARD_DIR / f"bon_shard_{shard_id:02d}_partial.json"
            with open(partial_file, "w", encoding="utf-8") as f:
                json.dump(shard_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Shard {shard_id} partial results saved to {partial_file.name}")
        raise

    # Save shard result
    shard_data = {
        "shard_id": shard_id,
        "num_questions": len(samples),
        "per_question": all_question_data
    }

    with open(shard_file, "w", encoding="utf-8") as f:
        json.dump(shard_data, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Shard {shard_id} completed and saved")
    return str(shard_file)


def merge_shard_results(shard_files: List[str]) -> Dict:
    """Merge all shard results into final output."""
    print("\n[合并 Shard 结果]")

    all_question_data = []

    for shard_file in sorted(shard_files):
        with open(shard_file, "r", encoding="utf-8") as f:
            shard_data = json.load(f)
            all_question_data.extend(shard_data["per_question"])
            print(f"  ✓ 加载 {Path(shard_file).name}: {shard_data['num_questions']} 题")

    print(f"  总计: {len(all_question_data)} 题")
    return all_question_data


def evaluate_strategies(all_question_data: List[Dict]) -> Dict:
    """Evaluate all selection strategies."""
    print("\n[评估五种选择策略]")

    strategies = {
        "random": select_random,
        "majority": select_majority,
        "ecs_bon": select_ecs_bon,
        "ecs_majority": select_ecs_majority,
        "oracle": select_oracle
    }

    results = {}

    for n in N_VALUES:
        print(f"  评估 N={n}...")
        results[f"N={n}"] = {}

        for strategy_name, strategy_func in strategies.items():
            loose_correct = 0
            strict_correct = 0

            for qdata in all_question_data:
                trajs_subset = qdata["trajectories"][:n]
                selected = strategy_func(trajs_subset)

                if selected["answer_loose"]:
                    loose_correct += 1
                if selected["answer_strict"]:
                    strict_correct += 1

            loose_em = loose_correct / len(all_question_data) * 100
            strict_em = strict_correct / len(all_question_data) * 100

            results[f"N={n}"][strategy_name] = {
                "loose": round(loose_em, 2),
                "strict": round(strict_em, 2)
            }

            print(f"    {strategy_name:15s} | 宽松EM: {loose_em:5.1f}% | 严格EM: {strict_em:5.1f}%")

    return results


def generate_plots(results: Dict):
    """Generate visualization plots."""
    strategies = ["random", "majority", "ecs_bon", "ecs_majority", "oracle"]
    strategy_labels = {
        "random": "Random",
        "majority": "Majority",
        "ecs_bon": "ECS-BoN",
        "ecs_majority": "ECS+Majority",
        "oracle": "Oracle"
    }
    strategy_styles = {
        "random": {"color": "gray", "linestyle": "--", "linewidth": 1.5},
        "majority": {"color": "blue", "linestyle": "-", "linewidth": 2},
        "ecs_bon": {"color": "red", "linestyle": "-", "linewidth": 3},
        "ecs_majority": {"color": "orange", "linestyle": "-", "linewidth": 2},
        "oracle": {"color": "green", "linestyle": "--", "linewidth": 1.5}
    }

    n_values = [1, 2, 4, 8]

    # Plot 1: Loose EM
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in strategies:
        loose_scores = [results[f"N={n}"][strategy]["loose"] for n in n_values]
        ax.plot(n_values, loose_scores,
                label=strategy_labels[strategy],
                marker='o',
                **strategy_styles[strategy])

    ax.set_xlabel("N (Number of Trajectories)", fontsize=12)
    ax.set_ylabel("Loose EM (%)", fontsize=12)
    ax.set_title("Best-of-N: Loose EM", fontsize=14, fontweight='bold')
    ax.set_xticks(n_values)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/bon_results_loose.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Strict EM
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in strategies:
        strict_scores = [results[f"N={n}"][strategy]["strict"] for n in n_values]
        ax.plot(n_values, strict_scores,
                label=strategy_labels[strategy],
                marker='o',
                **strategy_styles[strategy])

    ax.set_xlabel("N (Number of Trajectories)", fontsize=12)
    ax.set_ylabel("Strict EM (%)", fontsize=12)
    ax.set_title("Best-of-N: Strict EM", fontsize=14, fontweight='bold')
    ax.set_xticks(n_values)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/bon_results_strict.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def run_bon_experiment_parallel():
    """Run Best-of-N experiment with parallel processing."""
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("Best-of-N Parallel Experiment".center(70))
    print("=" * 70)
    print()

    # Load dataset
    print("[步骤 1/5] 加载 HotpotQA 数据集...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"  ✓ 数据集加载完成，共 {len(dataset)} 条样本")
    print()

    # Sample questions
    random.seed(42)
    print(f"[步骤 2/5] 随机采样 {SAMPLE_SIZE} 题...")
    indices = random.sample(range(len(dataset)), min(SAMPLE_SIZE, len(dataset)))
    samples = [dataset[i] for i in indices]
    print(f"  ✓ 采样完成，实际样本数: {len(samples)}")
    print()

    # Split into shards
    print(f"[步骤 3/5] 分片处理（{NUM_SHARDS} 个 shard，{NUM_WORKERS} 个并行进程）...")
    shard_size = len(samples) // NUM_SHARDS
    shards = []
    for i in range(NUM_SHARDS):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size if i < NUM_SHARDS - 1 else len(samples)
        shard_samples = samples[start_idx:end_idx]
        shards.append((i, shard_samples))

    print(f"  ✓ 分片完成: {NUM_SHARDS} 个 shard，每个约 {shard_size} 题")
    print(f"  预计 API 调用次数: {len(samples)} × {MAX_N} = {len(samples) * MAX_N}")
    print(f"  预计费用: 约 15-20 元")
    print(f"  预计时间: 约 {30 // NUM_WORKERS}-{50 // NUM_WORKERS} 分钟（{NUM_WORKERS} 进程并行）")
    print()

    # Process shards in parallel
    print("[步骤 4/5] 并行处理 shard...")
    print("  提示: 按 Ctrl+C 可以安全中断，已完成的 shard 会保存")
    start_time = time.time()

    try:
        # Use ProcessPoolExecutor for better Windows compatibility
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_shard = {executor.submit(process_shard, shard): shard[0] for shard in shards}

            shard_files = []
            completed = 0
            total = len(shards)

            # Process completed tasks as they finish
            for future in as_completed(future_to_shard):
                shard_id = future_to_shard[future]
                try:
                    shard_file = future.result()
                    shard_files.append(shard_file)
                    completed += 1
                    print(f"  进度: {completed}/{total} shard 完成")
                except Exception as e:
                    print(f"  ❌ Shard {shard_id} 失败: {e}")

            # Sort by shard ID to maintain order
            shard_files.sort()

    except KeyboardInterrupt:
        print("\n\n⚠ 实验被中断")
        print("  已完成的 shard 已保存到 data/shards/")
        print("  重新运行脚本会自动跳过已完成的 shard")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        print("  已完成的 shard 已保存到 data/shards/")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"  ✓ 所有 shard 处理完成，耗时: {elapsed/60:.1f} 分钟")
    print()

    # Merge results
    all_question_data = merge_shard_results(shard_files)

    # Evaluate strategies
    print("[步骤 5/5] 评估策略...")
    results = evaluate_strategies(all_question_data)
    print(f"  ✓ 评估完成")
    print()

    # Save final results
    output_data = {
        "config": {
            "n_values": N_VALUES,
            "sample_size": SAMPLE_SIZE,
            "max_n": MAX_N,
            "num_shards": NUM_SHARDS,
            "num_workers": NUM_WORKERS
        },
        "results": results,
        "per_question": all_question_data
    }

    with open("data/bon_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print("  ✓ 结果已保存到 data/bon_results.json")

    # Generate visualizations
    print()
    print("[生成可视化图表]")
    generate_plots(results)
    print("  ✓ 图表已保存到 outputs/")

    # Print summary
    print()
    print("=" * 70)
    print("实验总结".center(70))
    print("=" * 70)
    print()
    print("各策略在 N=8 时的表现：")
    strategies = ["random", "majority", "ecs_bon", "ecs_majority", "oracle"]
    for strategy_name in strategies:
        loose = results["N=8"][strategy_name]["loose"]
        strict = results["N=8"][strategy_name]["strict"]
        print(f"  {strategy_name:15s} | 宽松EM: {loose:5.1f}% | 严格EM: {strict:5.1f}%")
    print()
    print("=" * 70)


if __name__ == "__main__":
    run_bon_experiment_parallel()
