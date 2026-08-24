"""
Judge Consistency Analysis for SportsTime

Computes inter-judge agreement metrics (pairwise agreement, Cohen's kappa,
Fleiss' kappa) across multiple LLM judges and optionally against human labels.

Usage:
  python judge_consistency.py \
    --judges qwen_judged.jsonl minimax_judged.jsonl glm_judged.jsonl \
    --names Qwen MiniMax GLM \
    [--human human_labels.jsonl]
"""

import argparse
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple


def load_labels(path: str) -> Dict[str, int]:
    """Load binary labels from a judged JSONL file. Returns {sample_id: 0/1}."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("id") or obj.get("problem_id")
            if pid is None:
                continue

            if "correct" in obj:
                label = bool(obj["correct"])
            elif "judge" in obj and isinstance(obj["judge"], dict):
                label = str(obj["judge"].get("label", "")).upper() == "CORRECT"
            else:
                continue

            data[str(pid)] = int(label)
    return data


def pairwise_agreement(a: Dict[str, int], b: Dict[str, int]) -> Tuple[float, int]:
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan"), 0
    agree = sum(1 for k in keys if a[k] == b[k])
    return agree / len(keys), len(keys)


def cohen_kappa(a: Dict[str, int], b: Dict[str, int]) -> Tuple[float, int]:
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan"), 0
    n = len(keys)
    a_vals = [a[k] for k in keys]
    b_vals = [b[k] for k in keys]
    p0 = sum(1 for x, y in zip(a_vals, b_vals) if x == y) / n
    a_cnt, b_cnt = Counter(a_vals), Counter(b_vals)
    pe = sum((a_cnt[c] / n) * (b_cnt[c] / n) for c in [0, 1])
    if abs(1 - pe) < 1e-12:
        return float("nan"), n
    return (p0 - pe) / (1 - pe), n


def fleiss_kappa(rating_matrix: List[List[int]]) -> float:
    N = len(rating_matrix)
    if N == 0:
        return float("nan")
    k = len(rating_matrix[0])
    n = sum(rating_matrix[0])
    P_i = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in rating_matrix]
    P_bar = sum(P_i) / N
    p = [sum(row[j] for row in rating_matrix) / (N * n) for j in range(k)]
    P_e = sum(x * x for x in p)
    if abs(1 - P_e) < 1e-12:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def main():
    ap = argparse.ArgumentParser(description="Judge consistency analysis for SportsTime")
    ap.add_argument("--judges", nargs="+", required=True, help="Paths to judged JSONL files")
    ap.add_argument("--names", nargs="+", default=None, help="Names for each judge (default: Judge_0, Judge_1, ...)")
    ap.add_argument("--human", default=None, help="Optional human labels JSONL for alignment comparison")
    args = ap.parse_args()

    names = args.names or [f"Judge_{i}" for i in range(len(args.judges))]
    judges = [load_labels(p) for p in args.judges]

    print("=== Individual Accuracy ===")
    for name, labels in zip(names, judges):
        acc = sum(labels.values()) / max(1, len(labels))
        print(f"  {name}: {acc * 100:.2f}% (n={len(labels)})")

    print("\n=== Pairwise Agreement ===")
    agreements = []
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            agr, n = pairwise_agreement(judges[i], judges[j])
            agreements.append(agr)
            print(f"  {names[i]} vs {names[j]}: {agr * 100:.2f}% (n={n})")
    if agreements:
        print(f"  Avg. pairwise agreement: {sum(agreements) / len(agreements) * 100:.2f}%")

    print("\n=== Pairwise Cohen's Kappa ===")
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            kap, n = cohen_kappa(judges[i], judges[j])
            print(f"  {names[i]} vs {names[j]}: kappa={kap:.4f} (n={n})")

    if len(judges) >= 3:
        common = sorted(set.intersection(*(set(d) for d in judges)))
        matrix = []
        for k in common:
            votes = [d[k] for d in judges]
            matrix.append([len(votes) - sum(votes), sum(votes)])
        fk = fleiss_kappa(matrix)
        print(f"\n=== Fleiss' Kappa ({', '.join(names)}) ===")
        print(f"  Fleiss' kappa = {fk:.4f} (n={len(common)})")

    if args.human:
        human = load_labels(args.human)
        print("\n=== Human Alignment (Cohen's Kappa) ===")
        for name, labels in zip(names, judges):
            kap, n = cohen_kappa(labels, human)
            print(f"  {name} vs Human: kappa={kap:.4f} (n={n})")


if __name__ == "__main__":
    main()
