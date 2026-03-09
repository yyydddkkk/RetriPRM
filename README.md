# RetriPRM: Retrieval-Stage Process Reward Modeling

面向多跳推理 RAG 的检索阶段过程奖励模型研究

## 项目简介

本项目聚焦于多跳 RAG 中被系统性忽视的检索侧过程监督问题，提出 RetriPRM（Retrieval-Stage Process Reward Model）框架，通过细粒度的检索质量评估（ECS、RRS、RFS）解决现有方法中的信用分配混淆问题。

## 核心发现

基于 HotpotQA 5000 题的初步实验：

- **83.8%** 的正确答案来自不完美检索（ECS < 1.0）—— 幸存者偏差
- **44.0%** 的错误答案发生在检索正常的情况下（ECS ≥ 0.5）—— 隐性失败
- **仅 11.6%** 的样本达到完美检索（两跳都 ECS=1.0）
- **31.2%** 的第二跳检索到与第一跳完全相同的文档

这些数据强有力地证明了检索质量与最终答案正确性的解耦必要性。

## 项目结构

```
RetriPRM/
│
├── experiments/              # 实验脚本
│   ├── 01_gap_analysis.py   # Gap Analysis 实验
│   ├── 02_bon_experiment.py # Best-of-N 实验
│   └── deprecated/          # 废弃的实验脚本
│
├── data/                    # 实验数据
│   ├── gap_analysis_results.json          # Gap Analysis 结果
│   ├── gap_analysis_samples_detailed.json # 详细样本数据
│   ├── bon_results.json                   # Best-of-N 结果
│   ├── shards/                            # Best-of-N 分片数据
│   └── deprecated/                        # 废弃的数据文件
│
├── outputs/                 # 可视化图表
│   ├── gap_analysis/        # Gap Analysis 图表
│   ├── bon_experiment/      # Best-of-N 图表
│   └── deprecated/          # 废弃的图表
│
├── utils/                   # 工具模块
│   ├── retrieval.py        # BM25 检索
│   ├── scoring.py          # ECS 计算
│   └── llm.py              # Qwen API
│
├── docs/                    # 文档
│   ├── EXPERIMENT_RESULTS.md      # 实验结果整理（论文素材）
│   ├── PROJECT_STATUS.md          # 项目进度跟踪
│   ├── CONTROL_GROUP_DESIGN.md    # 对照组设置说明
│   ├── FILE_ORGANIZATION.md       # 文件组织说明
│   └── 开题报告_RetriPRM检索阶段过程奖励模型.docx
│
├── .gitignore
├── .env.example
├── pyproject.toml
└── README.md
```

## 快速开始

### 1. 环境配置

```bash
# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv sync
```

### 2. 配置 API Key

创建 `.env` 文件：

```bash
QWEN_API_KEY=your_api_key_here
```

### 3. 运行实验

```bash
# 实验 1: Gap 分析（验证信用分配混淆问题）
uv run python experiments/01_gap_analysis.py

# 实验 2: Best-of-N 对比实验
uv run python experiments/02_bon_experiment.py
```

## 实验结果

详细的实验结果和论文素材请查看：
- **实验结果整理**: `docs/EXPERIMENT_RESULTS.md`
- **项目进度**: `docs/PROJECT_STATUS.md`
- **对照组设计**: `docs/CONTROL_GROUP_DESIGN.md`

## 实验说明

### 实验 1: Gap Analysis

**目标**：验证检索质量与答案正确性的解耦必要性

**方法**：
- 在 HotpotQA 上采样 5000 题
- 使用 LLM 生成第二跳 Query（temperature=0.7）
- 计算 ECS（Expected Coverage Score）评估检索质量
- 统计幸存者偏差和隐性失败样本

**输出**：
- `data/gap_analysis_results_llm.json`：统计结果
- `data/gap_analysis_samples_detailed.json`：详细样本数据
- `outputs/ecs_distribution_comparison_llm.png`：ECS 分布对比图
- `outputs/ecs_boxplot_by_hop_llm.png`：各跳 ECS 箱线图

### 实验 2: Best-of-N Experiment

**目标**：验证 ECS 作为轨迹选择信号的有效性

**方法**：
- 为每题生成 N=8 条轨迹
- 对比 5 种选择策略：Random、Majority、ECS-BoN、ECS+Majority、Oracle
- 在 N=1,2,4,8 下评估宽松 EM 和严格 EM

**输出**：
- `data/bon_results.json`：完整实验结果
- `outputs/bon_results_loose.png`：宽松 EM 对比图
- `outputs/bon_results_strict.png`：严格 EM 对比图

## 核心模块

### utils/retrieval.py

BM25 检索相关功能：
- `build_corpus()`: 构建检索语料
- `preprocess_text()`: 文本预处理
- `simulate_two_hop_retrieval()`: 模拟两跳检索

### utils/scoring.py

ECS 计算和轨迹评分：
- `calculate_ecs()`: 计算 ECS（带 BM25 软分）
- `score_trajectory()`: 评估完整轨迹
- `is_query_degenerate()`: 检测 Query 退化
- `check_answer_correctness()`: 答案正确性判断

### utils/llm.py

Qwen API 交互：
- `get_qwen_client()`: 初始化 API 客户端
- `generate_hop2_query()`: 生成第二跳 Query

## 依赖

- Python 3.10+
- datasets
- rank-bm25
- numpy
- matplotlib
- tqdm
- openai
- scikit-learn

## 成本估算

- **Gap Analysis**：5000 题 × 1 次调用 = ~20 元
- **Best-of-N**：5000 题 × 8 次调用 = ~150-200 元

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{retriprm2026,
  title={RetriPRM: Retrieval-Stage Process Reward Modeling for Multi-Hop RAG},
  author={yyydddkkk},
  year={2026}
}
```

## 许可证

MIT License

## 联系方式

- 作者：jackyyyyyy
- 邮箱：yangjackk04@gmail.com
- 项目主页：https://github.com/yyydddkkk
