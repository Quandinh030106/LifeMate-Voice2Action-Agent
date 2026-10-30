# 🎙️ LifeMate — Agentic AI Voice-to-Action Pipeline
### Zalo AI Challenge 2026 Solution

This repository contains the complete, production-ready, competition-grade solution for the **LifeMate (Voice-to-Action Agentic AI)** track in Zalo AI Challenge 2026.

---

## 🏗️ Architecture Overview

The pipeline turns Vietnamese spoken requests into verifiable application actions via an end-to-end interactive agentic loop:

```
[Audio .wav] ──> [Vietnamese ASR] ──> [Transcript]
                                           │
[User Context & Installed Apps] ───────────┼──> [Prompt Builder]
                                           │
[Conversation History] ────────────────────┘
                                           │
                                           ▼
                                [Qwen2.5 Local LLM Engine]
                                  (Offline, SDPA Attention)
                                           │
                                    (Tool Call)
                                           │
                                           ▼
                              [zaic_tools Execution Env]
                                   (Single Session)
                                           │
                                    (Observation)
                                           │
                                           ▼
                            [Next Action / State Hash]
                                           │
                                           ▼
                                 [submission.jsonl]
```

### Key Modules:
- **`src/asr.py`**: Automatic Speech Recognition using `vinai/PhoWhisper-small` (244M params, specialized for Vietnamese dialects across North, Central, and South).
- **`src/prompt_builder.py`**: Dynamic schema condensation that filters only tools in `installed_apps`, constructs temporal reasoning constraints ("mai" -> `+1 day` in ISO-8601), and sets up context.
- **`src/llm_engine.py`**: Deterministic local LLM runner (`Qwen/Qwen2.5-1.5B-Instruct` or `3B`), using SDPA attention, greedy decoding (`temperature=0.0`), and strictly offline cache.
- **`src/parser.py`**: Multi-format tool call parser supporting Qwen `<tool_call>` tags, Markdown codeblocks, and raw JSON.
- **`src/agent.py`**: Multi-turn ReAct interactive agent executing tools on `zaic_tools.ToolEnv` and retrieving deterministic `state_hash`.
- **`src/evaluator.py`**: Full implementation of the official evaluation formula (State Match check, Non-fuzzy exact match, Fuzzy Token-F1 > 0.6, Precision, Recall, Out-of-Scope, Missing Info).
- **`inference.py`**: Official submission generator streaming lines directly to `result/submission.jsonl`.
- **`run_benchmark.py`**: Automated benchmarking and latency validation suite.

---

## ⚙️ Competition Constraints Compliance

| Requirement | Rule | Our Solution | Status |
|---|---|---|---|
| **System Size** | Total < 6B params; LLM < 5B params | PhoWhisper (244M) + Qwen2.5-1.5B (~1.7B total) |  Well within budget |
| **Context Length** | $\le 16\text{K}$ tokens | Schema condensation (~2K tokens max) |  Well within budget |
| **Network Access** | 100% Offline at inference | `local_files_only=True`, `HF_HUB_OFFLINE=1` |  Fully offline |
| **Reproducibility** | Fixed random seeds, greedy decoding | `seed_everything(42)`, `do_sample=False`, temp=0 |  Deterministic |
| **Data Access** | Never read `state.json` directly | Strictly interacts via `zaic_tools.ToolEnv` |  Fully compliant |
| **Output Format** | `submission.jsonl` matching official schema | Validated via `python -m zaic_tools --validate` |  100% Valid schema |
| **Docker** | Mount `/data`, output to `/app/result/submission.jsonl` | Provided `Dockerfile` matching specification |  Production Ready |

---

## 🚀 How to Run

### 1. High-Performance v2.0 Pipeline (SFT + NVIDIA Parakeet ASR)
```bash
# Step 1: Prepare SFT training data from train.jsonl (1545 multi-turn samples)
python scripts/prepare_sft_data.py --output_dir data/sft

# Step 2: Fine-tune Qwen2.5-3B-Instruct via LoRA and auto-merge standalone model
python train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --train_file data/sft/sft_train.jsonl \
    --output_dir models/qwen-lora \
    --merged_dir models/lifemate-qwen-sft

# Step 3: Run inference on public test with NVIDIA Parakeet + Fine-Tuned Agent
python inference.py \
    --split public_test \
    --asr_engine auto \
    --llm_model models/lifemate-qwen-sft \
    --output result/submission.jsonl

# Step 4: Validate generated submission schema
python -m zaic_tools --split public_test --validate result/submission.jsonl
```

### 2. Fast Baseline Inference (Without Fine-Tuning)
```bash
python inference.py --split public_test --output result/submission.jsonl
```

### 3. Running Benchmark & Score Evaluation
```bash
python run_benchmark.py --num_samples 10
```

---

## ☁️ Running Fast on Kaggle GPU (Tesla T4 x2 / P100)

We provide two pre-configured Kaggle notebooks:
- **`kaggle_train_and_inference_v2.ipynb`** *(Recommended)*: Complete pipeline that fine-tunes Qwen2.5-3B on train trajectories, merges model, and generates top-tier submission with NVIDIA Parakeet ASR.
- **`kaggle_submission.ipynb`**: Zero-shot baseline inference runner.

**Steps to run on Kaggle:**
1. **Upload Dataset**: Upload `train+public_test` to Kaggle as a private dataset.
2. **Import Notebook**: In Kaggle, click **File -> Import Notebook**, select `kaggle_train_and_inference_v2.ipynb`.
3. **Configure Environment**:
   - Accelerator: **GPU T4 x2** (or GPU P100).
   - Internet: **ON**.
4. **Run All Cells**:
   - The notebook automatically prepares data, runs SFT (~20 mins), merges weights, executes public test inference, validates with `zaic_tools --validate`, and outputs `/kaggle/working/submission.jsonl`.
5. **Download Submission**: Download `submission.jsonl` from Kaggle Output and submit to Zalo AI Challenge portal.

---

## 🐳 Docker Build & Run (Private Test)

```bash
# 1. Build Docker image
docker build -t lifemate_solution .

# 2. Test run container with mounted data folder
docker run --gpus all --network none \
    -v /path/to/data:/data \
    -v $(pwd)/result:/app/result \
    lifemate_solution
```
