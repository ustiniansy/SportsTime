"""
Step-wise Grounding Alignment (SGA) Evaluation for SportsTime

Evaluates the quality of temporal evidence in Chain-of-Time reasoning outputs.

Metrics:
  - Anchor(%): Percentage of predictions containing any temporal reference
  - mIoU: Mean temporal span IoU between predicted and ground-truth anchors
  - H@0.5(%): Fraction of samples whose best span IoU >= 0.5

Usage:
  python eval/sga_eval.py \
    --pred_jsonl predictions.jsonl \
    --gt_dir data/

  The --gt_dir should point to the SportsTime data/ directory containing
  {sport}/full_game.json and {sport}/highlight.json files. Each ground-truth
  item must have an "id" field and a "CoT" field with temporal references.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Anchor dataclass & regex patterns
# ---------------------------------------------------------------------------

@dataclass
class Anchor:
    kind: str  # "span" or "point"
    s: float = 0.0
    e: float = 0.0
    t: float = 0.0

    def as_key(self) -> Tuple:
        if self.kind == "span":
            return ("span", self.s, self.e)
        return ("point", self.t)


_HMS_SPAN_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})\s*[-–—~～]\s*(\d{1,2}):(\d{2}):(\d{2})"
)
_HMS_POINT_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
_MMSS_SPAN_RE = re.compile(
    r"(\d{1,3}):(\d{2})\s*[-–—~～]\s*(\d{1,3}):(\d{2})"
)
_MMSS_POINT_RE = re.compile(r"(\d{1,3}):(\d{2})")


def _extract_think(text: str) -> str:
    for tag in ("thinking", "think"):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1)
    return text


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _mmss_to_sec(mm: str, ss: str) -> float:
    return float(int(mm) * 60 + int(ss))


def _hms_to_sec(hh: str, mm: str, ss: str) -> Optional[float]:
    """Convert HH:MM:SS to seconds."""
    return float(int(hh) * 3600 + int(mm) * 60 + int(ss))


def time_str_to_sec(value: Any) -> Optional[float]:
    """Convert a timestamp string in MM:SS or HH:MM:SS format to seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    hms = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", text)
    if hms:
        return _hms_to_sec(hms.group(1), hms.group(2), hms.group(3))

    mmss = re.fullmatch(r"(\d{1,3}):(\d{2})", text)
    if mmss:
        return _mmss_to_sec(mmss.group(1), mmss.group(2))

    return None


def extract_anchors(text: str) -> Tuple[List[Anchor], bool]:
    """
    Extract temporal anchors from text (model output or CoT annotation).

    Parsing order: HH:MM:SS spans/points first (occupy character ranges to
    prevent MM:SS sub-matches), then MM:SS spans/points.

    Returns:
        anchors: List of valid Anchor objects usable for IoU
        has_any_time: Whether the text contains any temporal reference
    """
    if not text:
        return [], False

    text = str(text).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
    think = _extract_think(text)

    anchors: List[Anchor] = []
    occupied: List[Tuple[int, int]] = []
    has_any_time = False

    for m in _HMS_SPAN_RE.finditer(think):
        has_any_time = True
        occupied.append((m.start(), m.end()))
        s = _hms_to_sec(m.group(1), m.group(2), m.group(3))
        e = _hms_to_sec(m.group(4), m.group(5), m.group(6))
        if e < s:
            s, e = e, s
        anchors.append(Anchor(kind="span", s=s, e=e))

    for m in _HMS_POINT_RE.finditer(think):
        has_any_time = True
        occupied.append((m.start(), m.end()))
        t = _hms_to_sec(m.group(1), m.group(2), m.group(3))
        anchors.append(Anchor(kind="point", t=t))

    for m in _MMSS_SPAN_RE.finditer(think):
        rng = (m.start(), m.end())
        if any(_overlaps(rng, o) for o in occupied):
            continue
        has_any_time = True
        s = _mmss_to_sec(m.group(1), m.group(2))
        e = _mmss_to_sec(m.group(3), m.group(4))
        if e < s:
            s, e = e, s
        anchors.append(Anchor(kind="span", s=s, e=e))
        occupied.append(rng)

    for m in _MMSS_POINT_RE.finditer(think):
        rng = (m.start(), m.end())
        if any(_overlaps(rng, o) for o in occupied):
            continue
        has_any_time = True
        t = _mmss_to_sec(m.group(1), m.group(2))
        anchors.append(Anchor(kind="point", t=t))
        occupied.append(rng)

    uniq: Dict[Tuple, Anchor] = {}
    for a in anchors:
        uniq[a.as_key()] = a
    return list(uniq.values()), has_any_time


def anchors_from_time_refs(time_refs: Any) -> Tuple[List[Anchor], bool]:
    """Extract anchors from the structured ground-truth time_refs field."""
    if not isinstance(time_refs, list):
        return [], False

    anchors: List[Anchor] = []
    has_any_time = False

    for ref in time_refs:
        if not isinstance(ref, dict):
            continue

        ref_type = str(ref.get("type", "")).lower()
        if ref_type in {"ts", "timestamp", "point"} and "ts" in ref:
            has_any_time = True
            t = time_str_to_sec(ref.get("ts"))
            if t is not None:
                anchors.append(Anchor(kind="point", t=t))
            continue

        if ref_type in {"span", "range"} and "start" in ref and "end" in ref:
            has_any_time = True
            s = time_str_to_sec(ref.get("start"))
            e = time_str_to_sec(ref.get("end"))
            if s is None or e is None:
                continue
            if e < s:
                s, e = e, s
            anchors.append(Anchor(kind="span", s=s, e=e))

    uniq: Dict[Tuple, Anchor] = {}
    for a in anchors:
        uniq[a.as_key()] = a
    return list(uniq.values()), has_any_time


def gt_anchors_from_item(item: Dict[str, Any]) -> Tuple[List[Anchor], bool, str]:
    """Prefer structured time_refs for GT anchors; fall back to CoT text parsing."""
    anchors, has_time = anchors_from_time_refs(item.get("time_refs"))
    if has_time:
        return anchors, has_time, "time_refs"

    anchors, has_time = extract_anchors(item.get("CoT", ""))
    return anchors, has_time, "cot_text"


def anchor_to_span(anchor: Anchor, delta: float = 5.0) -> Tuple[float, float]:
    """Convert anchor to [start, end] span. Points are expanded by +/- delta seconds."""
    if anchor.kind == "span":
        s, e = anchor.s, anchor.e
        if e < s:
            s, e = e, s
        return max(0.0, s), max(0.0, e)
    return max(0.0, anchor.t - delta), max(0.0, anchor.t + delta)


def span_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(1e-8, max(a[1], b[1]) - min(a[0], b[0]))
    return inter / union


def load_gt_from_data_dir(data_dir: str) -> Dict[str, Dict]:
    """Load all ground-truth items from the SportsTime data/ directory, keyed by 'id'."""
    gt_map: Dict[str, Dict] = {}
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
                if item_id:
                    gt_map[str(item_id)] = item
    return gt_map


def main():
    ap = argparse.ArgumentParser(description="SGA Evaluation for SportsTime")
    ap.add_argument("--pred_jsonl", required=True,
                     help="Model predictions JSONL. Each line must have 'id' and 'pred_answer_raw'.")
    ap.add_argument("--gt_dir", default="data/",
                     help="Path to SportsTime data/ directory (default: data/)")
    ap.add_argument("--delta", type=float, default=5.0,
                     help="Point-to-span expansion half-window in seconds (default: 5.0)")
    args = ap.parse_args()

    preds = []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    gt_map = load_gt_from_data_dir(args.gt_dir)
    if not gt_map:
        print(f"[ERROR] No ground-truth items loaded from {args.gt_dir}. "
              "Make sure it contains {sport}/full_game.json and {sport}/highlight.json.")
        return

    total = len(preds)
    pred_has_time_cnt = 0
    eligible = 0
    ious: List[float] = []
    hit05 = 0
    missing_gt = 0
    gt_time_ref_source = {"time_refs": 0, "cot_text": 0}

    for row in preds:
        item_id = str(row.get("id", ""))

        pred_anchors, pred_has_time = extract_anchors(row.get("pred_answer_raw", ""))
        if pred_has_time:
            pred_has_time_cnt += 1

        gt = gt_map.get(item_id)
        if gt is None:
            missing_gt += 1
            continue

        gt_anchors, gt_has_time, source = gt_anchors_from_item(gt)
        if gt_has_time:
            gt_time_ref_source[source] += 1
        if not gt_has_time or not pred_has_time:
            continue

        eligible += 1

        if not pred_anchors or not gt_anchors:
            best = 0.0
        else:
            best = max(
                span_iou(anchor_to_span(pa, args.delta), anchor_to_span(ga, args.delta))
                for pa in pred_anchors for ga in gt_anchors
            )

        ious.append(best)
        if best >= 0.5:
            hit05 += 1

    anchor_pct = 100.0 * pred_has_time_cnt / max(1, total)
    miou = sum(ious) / max(1, len(ious))
    hit05_pct = 100.0 * hit05 / max(1, eligible)

    print("=" * 50)
    print("Step-wise Grounding Alignment (SGA) Evaluation")
    print("=" * 50)
    print(f"Total predictions       : {total}")
    print(f"GT items loaded         : {len(gt_map)}")
    print(f"Predictions missing GT  : {missing_gt}")
    print(f"GT time source          : time_refs={gt_time_ref_source['time_refs']}, "
          f"cot_text={gt_time_ref_source['cot_text']}")
    print(f"Anchor(%)               : {anchor_pct:.2f}")
    print(f"Eligible (both w/ time) : {eligible}")
    print(f"mIoU                    : {miou:.4f}")
    print(f"H@0.5(%)                : {hit05_pct:.2f}")
    print(f"Point delta (sec)       : {args.delta:.1f}")


if __name__ == "__main__":
    main()
