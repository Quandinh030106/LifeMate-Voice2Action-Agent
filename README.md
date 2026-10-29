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

### 1. Generating Submission for Public Test
```bash
# Run full audio pipeline on public test
python inference.py --split public_test --output result/submission.jsonl

# Validate generated submission schema
python -m zaic_tools --split public_test --validate result/submission.jsonl
```

### 2. Fast Text-only Verification on Train Split
```bash
python inference.py --split train --no_audio --limit 20 --output result/train_sub.jsonl
```

### 3. Running Benchmark & Score Evaluation
```bash
python run_benchmark.py --num_samples 10
```

---

## ☁️ Running Fast on Kaggle GPU

For fast submission generation on free Tesla T4 x2 or P100 GPUs:

1. **Upload Dataset**: Upload `train+public_test` to Kaggle as a private dataset.
2. **Import Notebook**: Create a new Kaggle Notebook, click **File -> Import Notebook**, and select [`kaggle_submission.ipynb`](file:///d:/PROJECTS/Zalo AIC 2026/LifeMate Agentic AI/kaggle_submission.ipynb).
3. **Configure Environment**:
   - Accelerator: **GPU T4 x2** (or GPU P100).
   - Internet: **ON**.
   - (For Private Repo) Add-ons -> Secrets: add secret named `GITHUB_PAT` with your GitHub token (`repo` scope).
4. **Run All Cells**:
   - The notebook clones the repo, resolves the dataset paths automatically, runs inference across all 634 public test audio samples, validates the output with `zaic_tools --validate`, and produces `/kaggle/working/submission.jsonl`.
5. **Download Submission**: Download `submission.jsonl` and submit to Zalo AI Challenge portal.

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
