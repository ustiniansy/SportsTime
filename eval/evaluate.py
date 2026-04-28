"""
Unified Evaluation Entry Point for SportsTime

Runs the full evaluation pipeline:
  1. LLM-as-Judge scoring (open-ended QA accuracy)
  2. Per-task-type accuracy breakdown
  3. Step-wise Grounding Alignment (SGA) evaluation

Usage:
  # Full pipeline (judge + accuracy + SGA)
  python eval/evaluate.py \
    --pred_jsonl predictions.jsonl \
    --gt_dir data/ \
    --judge_model_path /path/to/Qwen2.5-VL-7B-Instruct

  # Skip judging (if predictions are already judged)
  python eval/evaluate.py \
    --pred_jsonl predictions.judged.jsonl \
    --gt_dir data/ \
    --skip_judge

  # Only SGA (no judge, no accuracy)
  python eval/evaluate.py \
    --pred_jsonl predictions.jsonl \
    --gt_dir data/ \
    --only sga
"""

import argparse
import json
import os
import subprocess
import sys


def run_cmd(cmd: list, desc: str):
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {desc}", flush=True)
    print(f"{'=' * 60}\n", flush=True)
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"\n[ERROR] {desc} failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(
        description="SportsTime unified evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python eval/evaluate.py --pred_jsonl pred.jsonl --gt_dir data/ \\
      --judge_model_path /path/to/Qwen2.5-VL-7B-Instruct

  # Already judged, just compute metrics
  python eval/evaluate.py --pred_jsonl pred.judged.jsonl --gt_dir data/ --skip_judge

  # Only SGA evaluation
  python eval/evaluate.py --pred_jsonl pred.jsonl --gt_dir data/ --only sga
        """,
    )

    ap.add_argument("--pred_jsonl", required=True,
                     help="Prediction JSONL file")
    ap.add_argument("--gt_dir", default="data/",
                     help="Path to SportsTime data/ directory (default: data/)")

    # Judge options
    judge_group = ap.add_argument_group("LLM-as-Judge")
    judge_group.add_argument("--judge_model_path", default=None,
                              help="Path to judge model (e.g. Qwen2.5-VL-7B-Instruct)")
    judge_group.add_argument("--judge_model_family", choices=["qwen2.5", "qwen3"], default="qwen2.5")
    judge_group.add_argument("--skip_judge", action="store_true",
                              help="Skip judging (use if predictions already have judge results)")

    # SGA options
    sga_group = ap.add_argument_group("SGA Evaluation")
    sga_group.add_argument("--delta", type=float, default=5.0,
                            help="Point expansion half-window in seconds (default: 5.0)")

    # Control
    ap.add_argument("--only", choices=["judge", "accuracy", "sga"],
                     default=None, help="Run only a specific evaluation step")

    args = ap.parse_args()

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    pred_jsonl = args.pred_jsonl
    judged_jsonl = pred_jsonl

    # ── Step 1: LLM-as-Judge ──
    run_judge = (args.only is None or args.only == "judge") and not args.skip_judge
    if run_judge:
        if not args.judge_model_path:
            print("[ERROR] --judge_model_path is required unless --skip_judge is set.")
            sys.exit(1)

        judged_jsonl = pred_jsonl.replace(".jsonl", ".judged.jsonl")
        run_cmd([
            sys.executable, os.path.join(eval_dir, "judge.py"),
            "--in_jsonl", pred_jsonl,
            "--out_jsonl", judged_jsonl,
            "--judge_model_path", args.judge_model_path,
            "--judge_model_family", args.judge_model_family,
        ], "Step 1/3: LLM-as-Judge")

    if args.only == "judge":
        return

    # ── Step 2: Per-Task Accuracy ──
    if args.only is None or args.only == "accuracy":
        input_for_acc = judged_jsonl if (run_judge or args.skip_judge) else pred_jsonl
        run_cmd([
            sys.executable, os.path.join(eval_dir, "task_accuracy.py"),
            "--pred_jsonl", input_for_acc,
            "--gt_dir", args.gt_dir,
        ], "Step 2/3: Per-Task Accuracy")

    if args.only == "accuracy":
        return

    # ── Step 3: SGA Evaluation ──
    if args.only is None or args.only == "sga":
        run_cmd([
            sys.executable, os.path.join(eval_dir, "sga_eval.py"),
            "--pred_jsonl", pred_jsonl,
            "--gt_dir", args.gt_dir,
            "--delta", str(args.delta),
        ], "Step 3/3: Step-wise Grounding Alignment (SGA)")

    if args.only is None:
        print(f"\n{'=' * 60}", flush=True)
        print("  All evaluations complete.", flush=True)
        print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
