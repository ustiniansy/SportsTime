"""
Per-Task-Type Accuracy Breakdown for SportsTime

Computes accuracy for each reasoning task type (Perception, Temporal,
Tactical, Causal, Counterfactual) from judged prediction results.

The ground-truth task_type is read directly from the SportsTime data/ files.

Usage:
  python eval/task_accuracy.py \
    --pred_jsonl predictions.judged.jsonl \
    --gt_dir data/
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


def load_task_types(data_dir: str) -> Dict[str, str]:
    """Load task_type for each item from SportsTime data/ directory, keyed by 'id'."""
    task_map: Dict[str, str] = {}
    for sport in os.listdir(data_dir):
        sport_dir = os.path.join(data_dir, sport)
        if not os.path.isdir(sport_dir):
            continue
        for fname in os.listdir(sport_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(sport_dir, fname), "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                item_id = item.get("id")
                task = item.get("task_type")
                if item_id and task:
                    task_map[str(item_id)] = str(task)
    return task_map


def infer_correct(sample: Dict[str, Any]) -> Optional[bool]:
    """Infer correctness from 'correct' field or 'judge.label'."""
    if "correct" in sample:
        c = sample["correct"]
        if isinstance(c, bool):
            return c
        if isinstance(c, (int, float)):
            return bool(c)
        if isinstance(c, str):
            return c.lower() in ("true", "1", "yes")

    j = sample.get("judge")
    if isinstance(j, dict):
        label = str(j.get("label", "")).upper()
        if label == "CORRECT":
            return True
        if label == "INCORRECT":
            return False
    return None


def main():
    ap = argparse.ArgumentParser(description="Per-task accuracy for SportsTime")
    ap.add_argument("--pred_jsonl", required=True, help="Judged prediction JSONL (must have 'id' and judge results)")
    ap.add_argument("--gt_dir", default="data/", help="Path to SportsTime data/ directory (default: data/)")
    args = ap.parse_args()

    # Load task types from ground truth
    task_map = load_task_types(args.gt_dir)
    if not task_map:
        print(f"[ERROR] No items loaded from {args.gt_dir}. "
              "Make sure it contains {{sport}}/full_game.json and {{sport}}/highlight.json.")
        return
    print(f"Loaded {len(task_map)} items with task_type from {args.gt_dir}")

    # Load predictions
    preds = []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    missing = 0

    for s in preds:
        item_id = str(s.get("id", ""))
        task = task_map.get(item_id)
        if task is None:
            missing += 1
            continue

        stats[task]["total"] += 1
        c = infer_correct(s)
        if c:
            stats[task]["correct"] += 1

    rows = []
    for task, d in stats.items():
        acc = d["correct"] / max(1, d["total"])
        rows.append((task, d["total"], d["correct"], acc))
    rows.sort(key=lambda x: (-x[1], x[0]))

    overall_total = sum(r[1] for r in rows)
    overall_correct = sum(r[2] for r in rows)
    overall_acc = overall_correct / max(1, overall_total)

    print(f"\n{'Task Type':<20} | {'N':>6} | {'Correct':>7} | {'Acc':>8}")
    print("-" * 50)
    for task, total, correct, acc in rows:
        print(f"{task:<20} | {total:>6d} | {correct:>7d} | {acc:>7.2%}")
    print("-" * 50)
    print(f"{'Overall':<20} | {overall_total:>6d} | {overall_correct:>7d} | {overall_acc:>7.2%}")

    if missing > 0:
        print(f"\n[WARN] {missing} predictions could not be matched to ground truth (missing 'id').")


if __name__ == "__main__":
    main()
