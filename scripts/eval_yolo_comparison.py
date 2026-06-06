"""Compare original vs fine-tuned YOLO on the exported val set.

For each image in yolo_dataset/images/val/, runs both models and compares
their top prediction against the ground-truth label. Produces a per-class
breakdown and categorises every image as:

  improvement  – fine-tuned correct, original wrong
  regression   – original correct, fine-tuned wrong
  both_correct – both models right
  both_wrong   – both models wrong

Usage
-----
  # After fine-tuning completes:
  python -m scripts.eval_yolo_comparison \\
      --fine-tuned runs/detect/train1/weights/best.pt

  # Custom val dir or original model:
  python -m scripts.eval_yolo_comparison \\
      --original yolov8n.pt \\
      --fine-tuned runs/detect/train2/weights/best.pt \\
      --val-dir yolo_dataset/images/val \\
      --labels-dir yolo_dataset/labels/val
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"), format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_yolo")

_CLASS_NAMES = {0: "person", 1: "chalker"}
_DEFAULT_ORIGINAL  = "yolov8n.pt"
_DEFAULT_VAL       = "yolo_dataset/images/val"
_DEFAULT_LABELS    = "yolo_dataset/labels/val"
_CONF_THRESHOLD    = 0.25   # minimum detection confidence to count


def _load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def _predict_class(model, img_path: Path, original_is_stock: bool) -> tuple[int, float]:
    """Return (predicted_class_id, confidence) for the highest-confidence detection."""
    results = model.predict(str(img_path), verbose=False, conf=_CONF_THRESHOLD)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return -1, 0.0
    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    cls   = boxes.cls.cpu().numpy().astype(int)
    best  = int(confs.argmax())
    pred_cls = int(cls[best])
    # Stock yolov8n only knows class 0 (person) — map anything it finds to 0.
    if original_is_stock:
        pred_cls = 0
    return pred_cls, float(confs[best])


def _read_gt(label_path: Path) -> int:
    """Read the first class id from a YOLO label file."""
    if not label_path.exists():
        return -1
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return -1
    return int(lines[0].split()[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original",    default=_DEFAULT_ORIGINAL,
                    help="Path to the original/base YOLO model weights")
    ap.add_argument("--fine-tuned",  required=True,
                    help="Path to the fine-tuned weights (e.g. runs/detect/train1/weights/best.pt)")
    ap.add_argument("--val-dir",     default=_DEFAULT_VAL,
                    help="Directory of val images")
    ap.add_argument("--labels-dir",  default=_DEFAULT_LABELS,
                    help="Directory of val label .txt files")
    ap.add_argument("--conf",        type=float, default=_CONF_THRESHOLD,
                    help="Minimum confidence threshold for a detection to count")
    ap.add_argument("--show-regressions", action="store_true",
                    help="Print filenames of every regression case")
    ap.add_argument("--show-improvements", action="store_true",
                    help="Print filenames of every improvement case")
    args = ap.parse_args()

    val_dir    = _ROOT / args.val_dir
    labels_dir = _ROOT / args.labels_dir
    images     = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))

    if not images:
        print(f"No images found in {val_dir}")
        sys.exit(1)

    print(f"Loading original model  : {args.original}")
    orig_model = _load_model(args.original)
    print(f"Loading fine-tuned model: {args.fine_tuned}")
    ft_model   = _load_model(args.fine_tuned)
    # Detect whether the original is a stock COCO model (no chalker class).
    orig_is_stock = orig_model.names.get(1) != "chalker"

    print(f"\nEvaluating {len(images)} images ...\n")

    # Per-class accumulators: { gt_class: Counter of outcome }
    by_class: dict[int, Counter] = defaultdict(Counter)
    regressions:   list[str] = []
    improvements:  list[str] = []

    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")
        gt = _read_gt(label_path)
        if gt < 0:
            continue

        orig_cls, orig_conf = _predict_class(orig_model, img_path, orig_is_stock)
        ft_cls,   ft_conf   = _predict_class(ft_model,   img_path, False)

        orig_correct = (orig_cls == gt)
        ft_correct   = (ft_cls   == gt)

        if ft_correct and not orig_correct:
            outcome = "improvement"
            improvements.append(f"  {img_path.name}  gt={_CLASS_NAMES.get(gt,gt)}  orig={_CLASS_NAMES.get(orig_cls,'no-det')}({orig_conf:.0%})  ft={_CLASS_NAMES.get(ft_cls,'no-det')}({ft_conf:.0%})")
        elif orig_correct and not ft_correct:
            outcome = "regression"
            regressions.append(f"  {img_path.name}  gt={_CLASS_NAMES.get(gt,gt)}  orig={_CLASS_NAMES.get(orig_cls,'no-det')}({orig_conf:.0%})  ft={_CLASS_NAMES.get(ft_cls,'no-det')}({ft_conf:.0%})")
        elif orig_correct and ft_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"

        by_class[gt][outcome] += 1

    # ── Print results ─────────────────────────────────────────────────────────
    total_impr = sum(c["improvement"] for c in by_class.values())
    total_regr = sum(c["regression"]  for c in by_class.values())
    total_both = sum(c["both_correct"] for c in by_class.values())
    total_bad  = sum(c["both_wrong"]   for c in by_class.values())
    total      = total_impr + total_regr + total_both + total_bad

    print("=" * 56)
    print(f"  {'CLASS':<12} {'BOTH OK':>8} {'IMPROVE':>8} {'REGRESS':>8} {'BOTH BAD':>9}")
    print("-" * 56)
    for cls_id in sorted(by_class):
        c    = by_class[cls_id]
        name = _CLASS_NAMES.get(cls_id, str(cls_id))
        n    = sum(c.values())
        print(f"  {name:<12} {c['both_correct']:>8} {c['improvement']:>8} {c['regression']:>8} {c['both_wrong']:>9}   (n={n})")
    print("-" * 56)
    print(f"  {'TOTAL':<12} {total_both:>8} {total_impr:>8} {total_regr:>8} {total_bad:>9}   (n={total})")
    print("=" * 56)

    if total > 0:
        print(f"\n  Net gain : +{total_impr} improvements, -{total_regr} regressions")
        delta = total_impr - total_regr
        verdict = "BETTER" if delta > 0 else ("WORSE" if delta < 0 else "NEUTRAL")
        print(f"  Verdict  : {verdict} ({delta:+d})\n")

    if args.show_improvements and improvements:
        print(f"Improvements ({len(improvements)}):")
        print("\n".join(improvements[:50]))
        if len(improvements) > 50:
            print(f"  ... and {len(improvements)-50} more")
        print()

    if args.show_regressions and regressions:
        print(f"Regressions ({len(regressions)}):")
        print("\n".join(regressions[:50]))
        if len(regressions) > 50:
            print(f"  ... and {len(regressions)-50} more")
        print()


if __name__ == "__main__":
    main()
