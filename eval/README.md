# Evaluation

This directory contains the evaluation scripts for SportsTime.

## Overview

SportsTime uses a **dual-track evaluation protocol**:

1. **Open-ended QA** — LLM-as-Judge scores model answers against ground truth
2. **Step-wise Grounding Alignment (SGA)** — Measures temporal evidence quality in reasoning chains

## Scripts

| Script | Description |
|---|---|
| **`evaluate.py`** | **Unified entry point** — runs judge + accuracy + SGA in one command |
| `judge.py` | LLM-as-Judge: scores predictions using Qwen2.5-VL / Qwen3-VL |
| `sga_eval.py` | SGA evaluation: computes Anchor(%), mIoU, H@0.5 for temporal grounding |
| `task_accuracy.py` | Per-task-type accuracy breakdown |
| `judge_consistency.py` | Inter-judge agreement: pairwise agreement, Cohen's & Fleiss' kappa |

## Requirements

```
torch>=2.1
transformers>=4.45
tqdm
```

## Prediction Format

Your model should produce a JSONL file where each line is a JSON object:

```json
{
  "id": "Basketball_Full_001_1_1",
  "question": "At the start of the game, what was the most direct cause of ...",
  "gt_text": "Warriors #30 was caught on the screen while defending",
  "pred_answer": "The foul occurred because the defender was blocked by a screen",
  "pred_answer_raw": "<thinking>1. At 01:48, Thunder #35 holds the ball...\n2. At 01:49, ...</thinking>\n<answer>The foul occurred because the defender was blocked by a screen</answer>"
}
```

| Field | Description |
|---|---|
| `id` | Sample ID matching the SportsTime data files |
| `question` | Question text |
| `gt_text` | Ground-truth answer text |
| `pred_answer` | Extracted model answer (used by LLM-as-Judge) |
| `pred_answer_raw` | Full model output including `<thinking>` tags (used by SGA eval) |

## Quick Start

All commands are run from the repository root.

### Full Pipeline (Recommended)

```bash
# Run all three evaluation steps in one command
python eval/evaluate.py \
  --pred_jsonl predictions.jsonl \
  --gt_dir data/ \
  --judge_model_path /path/to/Qwen2.5-VL-7B-Instruct

# If predictions are already judged, skip the judge step
python eval/evaluate.py \
  --pred_jsonl predictions.judged.jsonl \
  --gt_dir data/ \
  --skip_judge

# Run only SGA evaluation
python eval/evaluate.py \
  --pred_jsonl predictions.jsonl \
  --gt_dir data/ \
  --only sga
```

### Individual Scripts

### 1. LLM-as-Judge

```bash
# Single GPU
python eval/judge.py \
  --in_jsonl predictions.jsonl \
  --judge_model_path /path/to/Qwen2.5-VL-7B-Instruct

# Multi-GPU (data parallel)
torchrun --nproc_per_node 4 eval/judge.py \
  --in_jsonl predictions.jsonl \
  --judge_model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --parallel dp
```

Output: `predictions.judged.jsonl` with added `judge`, `correct`, and `score` fields.

### 2. Per-Task Accuracy

```bash
python eval/task_accuracy.py \
  --pred_jsonl predictions.judged.jsonl \
  --gt_dir data/
```

Reads `task_type` from the SportsTime data files and reports accuracy per task type.

### 3. SGA Evaluation (Temporal Grounding Quality)

```bash
python eval/sga_eval.py \
  --pred_jsonl predictions.jsonl \
  --gt_dir data/
```

Extracts temporal anchors from `pred_answer_raw` and compares against `CoT` in ground-truth data.

### 4. Judge Consistency

```bash
python eval/judge_consistency.py \
  --judges qwen.judged.jsonl minimax.judged.jsonl glm.judged.jsonl \
  --names Qwen MiniMax GLM
```
