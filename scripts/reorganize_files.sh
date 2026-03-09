#!/bin/bash
# RetriPRM 文件整理脚本

set -e

echo "=========================================="
echo "RetriPRM 文件整理脚本"
echo "=========================================="
echo ""

# 创建必要的目录
echo "[1/5] 创建目录结构..."
mkdir -p experiments/deprecated
mkdir -p data/deprecated
mkdir -p outputs/gap_analysis
mkdir -p outputs/bon_experiment
mkdir -p outputs/deprecated
mkdir -p docs
mkdir -p tests

# 整理实验脚本
echo "[2/5] 整理实验脚本..."

# 保留并重命名
if [ -f "01_gap_analysis_llm.py" ]; then
    cp 01_gap_analysis_llm.py experiments/01_gap_analysis.py
    echo "  ✓ 01_gap_analysis_llm.py → experiments/01_gap_analysis.py"
fi

if [ -f "03_bon_experiment_parallel.py" ]; then
    cp 03_bon_experiment_parallel.py experiments/02_bon_experiment.py
    echo "  ✓ 03_bon_experiment_parallel.py → experiments/02_bon_experiment.py"
fi

# 移动到 deprecated
for file in 01_gap_analysis.py 03_bon_experiment.py 02_sample_hidden_failures.py \
            03_ecs_breakdown_analysis.py 04_ecs_fine_grained_report.py \
            main.py test_qwen_api.py; do
    if [ -f "$file" ]; then
        mv "$file" experiments/deprecated/
        echo "  ✓ $file → experiments/deprecated/"
    fi
done

# 整理数据文件
echo "[3/5] 整理数据文件..."

# 保留并重命名
if [ -f "data/gap_analysis_results_llm.json" ]; then
    cp data/gap_analysis_results_llm.json data/gap_analysis_results.json
    echo "  ✓ gap_analysis_results_llm.json → gap_analysis_results.json"
fi

if [ -f "data/bon_results_corrected.json" ]; then
    cp data/bon_results_corrected.json data/bon_results.json
    echo "  ✓ bon_results_corrected.json → bon_results.json"
fi

# 移动到 deprecated
for file in gap_analysis_results.json bon_results.json ecs_breakdown_results.json; do
    if [ -f "data/$file" ]; then
        mv "data/$file" data/deprecated/
        echo "  ✓ data/$file → data/deprecated/"
    fi
done

# 整理可视化图表
echo "[4/5] 整理可视化图表..."

# Gap Analysis 图表
if [ -f "outputs/ecs_distribution_comparison_llm.png" ]; then
    cp outputs/ecs_distribution_comparison_llm.png outputs/gap_analysis/ecs_distribution_comparison.png
    echo "  ✓ ecs_distribution_comparison_llm.png → outputs/gap_analysis/"
fi

if [ -f "outputs/ecs_boxplot_by_hop_llm.png" ]; then
    cp outputs/ecs_boxplot_by_hop_llm.png outputs/gap_analysis/ecs_boxplot_by_hop.png
    echo "  ✓ ecs_boxplot_by_hop_llm.png → outputs/gap_analysis/"
fi

# Best-of-N 图表
if [ -f "outputs/bon_results_strict_corrected.png" ]; then
    cp outputs/bon_results_strict_corrected.png outputs/bon_experiment/bon_results_strict.png
    echo "  ✓ bon_results_strict_corrected.png → outputs/bon_experiment/"
fi

for file in bon_results_high_diversity.png bon_results_low_diversity.png bon_comparison_by_diversity.png; do
    if [ -f "outputs/$file" ]; then
        cp "outputs/$file" outputs/bon_experiment/
        echo "  ✓ $file → outputs/bon_experiment/"
    fi
done

# 移动旧图表到 deprecated
for file in ecs_distribution_comparison.png ecs_boxplot_by_hop.png \
            bon_results_loose.png bon_results_strict.png; do
    if [ -f "outputs/$file" ]; then
        mv "outputs/$file" outputs/deprecated/
        echo "  ✓ outputs/$file → outputs/deprecated/"
    fi
done

# 整理文档
echo "[5/5] 整理文档..."

for file in EXPERIMENT_RESULTS.md PROJECT_STATUS.md PARALLEL_CONFIG.md FILE_ORGANIZATION.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/
        echo "  ✓ $file → docs/"
    fi
done

# 移动开题报告（如果存在）
if [ -f "doc/开题报告_RetriPRM检索阶段过程奖励模型.docx" ]; then
    cp "doc/开题报告_RetriPRM检索阶段过程奖励模型.docx" docs/
    echo "  ✓ 开题报告 → docs/"
fi

echo ""
echo "=========================================="
echo "文件整理完成！"
echo "=========================================="
echo ""
echo "核心文件清单："
echo ""
echo "实验脚本:"
echo "  - experiments/01_gap_analysis.py"
echo "  - experiments/02_bon_experiment.py"
echo ""
echo "数据文件:"
echo "  - data/gap_analysis_results.json"
echo "  - data/gap_analysis_samples_detailed.json"
echo "  - data/bon_results.json"
echo "  - data/shards/bon_shard_*.json"
echo ""
echo "可视化图表:"
echo "  - outputs/gap_analysis/ (2 个图表)"
echo "  - outputs/bon_experiment/ (4 个图表)"
echo ""
echo "文档:"
echo "  - docs/EXPERIMENT_RESULTS.md"
echo "  - docs/PROJECT_STATUS.md"
echo "  - docs/PARALLEL_CONFIG.md"
echo "  - docs/FILE_ORGANIZATION.md"
echo ""
echo "旧文件已移动到 */deprecated/ 目录"
echo ""
