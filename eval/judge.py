"""
LLM-as-Judge Evaluation for SportsTime

Scores model predictions against ground-truth answers using Qwen2.5-VL / Qwen3-VL
as an impartial judge. Outputs a judged JSONL with label (CORRECT/INCORRECT/UNSURE),
score (0-1), and reason for each sample.

Supports:
  - Data-parallel (DP) sharding via torchrun
  - Resume from partial runs
  - Flexible judge model family (qwen2.5 / qwen3)

Usage:
  # Single GPU
  python judge.py --in_jsonl predictions.jsonl --judge_model_path /path/to/Qwen2.5-VL-7B-Instruct

  # Multi-GPU (DP)
  torchrun --nproc_per_node 4 judge.py \
    --in_jsonl predictions.jsonl \
    --judge_model_path /path/to/Qwen2.5-VL-7B-Instruct \
    --parallel dp
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are an impartial grader.
Given a question, a ground-truth answer, and a model answer, judge whether the model answer is correct.
Guidelines:
1) Consider semantic equivalence; allow paraphrase/synonyms.
2) If the model answer is partially correct but misses key constraints, mark INCORRECT and give a lower score.
3) If the question expects a specific location/option and the model answer is ambiguous, mark UNSURE.
Output STRICTLY a JSON object with keys: label, score, reason.
label must be one of: CORRECT, INCORRECT, UNSURE.
score must be a number between 0 and 1.

Question:
{question}

Ground-truth answer:
{gt_text}

Model answer:
{pred_text}
"""


def build_judge_prompt(question: str, gt_text: str, pred_text: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(question=question, gt_text=gt_text, pred_text=pred_text)


def parse_judge_json(text: str) -> Dict[str, Any]:
    """Parse judge output into {label, score, reason} with multiple fallbacks."""
    if not text:
        return {"label": "UNSURE", "score": 0.0, "reason": "empty_judge_output"}
    t = text.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return {
                "label": str(obj.get("label", "UNSURE")).upper(),
                "score": float(obj.get("score", 0.0)),
                "reason": str(obj.get("reason", ""))[:500],
            }
    except Exception:
        pass

    # Try extracting JSON substring
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {
                    "label": str(obj.get("label", "UNSURE")).upper(),
                    "score": float(obj.get("score", 0.0)),
                    "reason": str(obj.get("reason", ""))[:500],
                }
        except Exception:
            pass

    # Regex fallback
    label = "UNSURE"
    m1 = re.search(r"\b(CORRECT|INCORRECT|UNSURE)\b", t, flags=re.IGNORECASE)
    if m1:
        label = m1.group(1).upper()
    score = 0.0
    m2 = re.search(r"score\s*[:=]\s*([0-1](?:\.\d+)?)", t, flags=re.IGNORECASE)
    if m2:
        score = float(m2.group(1))

    return {"label": label, "score": max(0.0, min(1.0, score)), "reason": t[:500]}


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------

def load_judge_model(model_path: str, model_family: str, device_map):
    if model_family == "qwen3":
        if AutoModelForImageTextToText is None:
            raise RuntimeError("AutoModelForImageTextToText not available. Upgrade transformers.")
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map=device_map)
    elif model_family == "qwen2.5":
        if Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available. Upgrade transformers.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map=device_map)
    else:
        raise ValueError(f"Unknown model_family: {model_family}")
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


@torch.inference_mode()
def generate_text(model, processor, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def judge_one(model, processor, question: str, gt_text: str, pred_text: str,
              max_new_tokens: int = 256) -> Tuple[Dict[str, Any], str]:
    prompt = build_judge_prompt(question, gt_text, pred_text)
    raw = generate_text(model, processor, prompt, max_new_tokens)
    parsed = parse_judge_json(raw)
    parsed["label"] = parsed.get("label", "UNSURE")
    parsed["score"] = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
    return parsed, raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="LLM-as-Judge for SportsTime")
    ap.add_argument("--in_jsonl", required=True, help="Prediction JSONL (must have pred_answer, gt_text, question)")
    ap.add_argument("--out_jsonl", default=None, help="Output JSONL (default: in_jsonl with .judged suffix)")
    ap.add_argument("--judge_model_path", required=True, help="Path to judge model (e.g. Qwen2.5-VL-7B-Instruct)")
    ap.add_argument("--judge_model_family", choices=["qwen2.5", "qwen3"], default="qwen2.5")
    ap.add_argument("--judge_max_new_tokens", type=int, default=256)
    ap.add_argument("--parallel", choices=["dp", "none"], default="none")
    ap.add_argument("--resume", action="store_true", help="Skip already-judged samples")
    args = ap.parse_args()

    # Distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if args.parallel == "dp" and world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        rank, world_size = 0, 1
        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

    # Output path
    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
    else:
        out_path = Path(args.in_jsonl).with_suffix(".judged.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        out_path = out_path.with_suffix(f".shard{rank:02d}.jsonl")

    # Resume
    done_indices = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if "index" in obj:
                        done_indices.add(int(obj["index"]))
                except Exception:
                    pass

    # Load judge model
    model, processor = load_judge_model(args.judge_model_path, args.judge_model_family, device_map)

    # Read predictions
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    judged = correct = 0
    score_sum = 0.0
    t0 = time.time()

    with out_path.open("a" if args.resume else "w", encoding="utf-8") as wf:
        for i, line in enumerate(tqdm(lines, desc=f"Judging (rank={rank})")):
            rec = json.loads(line)
            idx = rec.get("index", i)
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = i

            if world_size > 1 and (idx % world_size) != rank:
                continue
            if args.resume and idx in done_indices:
                continue

            pred = rec.get("pred_answer")
            gt = rec.get("gt_text")
            question = str(rec.get("question", ""))

            if pred is None or gt is None:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            try:
                parsed, raw = judge_one(model, processor, question, str(gt), str(pred),
                                        args.judge_max_new_tokens)
                rec["judge"] = parsed
                rec["judge_raw"] = raw
                rec["correct"] = parsed["label"] == "CORRECT"
                rec["score"] = parsed["score"]
            except Exception as e:
                rec["judge"] = {"label": "UNSURE", "score": 0.0, "reason": str(e)}
                rec["correct"] = False
                rec["score"] = 0.0

            judged += 1
            correct += int(rec["correct"])
            score_sum += rec["score"]

            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wf.flush()

    dt = time.time() - t0
    acc = correct / max(1, judged)

    if rank == 0:
        print(f"\n[Judge Summary] judged={judged}, correct={correct}, acc={acc:.4f}, "
              f"avg_score={score_sum / max(1, judged):.4f}, time={dt:.1f}s")

    if args.parallel == "dp" and world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
