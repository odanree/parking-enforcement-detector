"""Export ChromaDB events to a YOLOv8-format dataset for fine-tuning.

The stored images are person crops (person fills most of the frame), so the
YOLO bounding-box annotation is approximated as the full image bounds.
This is suitable for fine-tuning a two-class person/chalker discriminator or
for scene-specific person detection adaptation.

Classes
-------
  0  person   – any non-chalker detection
  1  chalker  – person_type == 'chalker' OR label in ('true_positive',)

Output layout
-------------
  <out>/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    data.yaml

Usage
-----
  # Dry-run — prints stats, writes nothing:
  python -m scripts.export_yolo_dataset

  # Write dataset:
  python -m scripts.export_yolo_dataset --apply

  # Custom output dir, only chalker + high-confidence negatives:
  python -m scripts.export_yolo_dataset --apply --out yolo_out --min-yolo-conf 0.65
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import chromadb  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_yolo")

_DEFAULT_DB      = os.getenv("VECTOR_DB_PATH", "data/vectors")
_DEFAULT_DATA    = os.getenv("DATASET_PATH",   "dataset")
_DEFAULT_OUT     = "yolo_dataset"
_COLLECTION      = "chalking_evals"
_VAL_RATIO       = 0.20
_RANDOM_SEED     = 42

# YOLO annotation: (cx, cy, w, h) normalised — person fills ~95% of each crop.
_PERSON_BBOX     = "0.500 0.500 0.950 0.950"

_VLM_ERROR_PREFIXES = ("VLM ", "Model not found", "Classification failed")
_VLM_ERROR_SUBS     = ("VLM timeout", "VLM unreachable", "VLM HTTP", "VLM error")


def _is_vlm_error(desc: str) -> bool:
    if not desc:
        return True
    if any(desc.startswith(p) for p in _VLM_ERROR_PREFIXES):
        return True
    return any(s in desc for s in _VLM_ERROR_SUBS)


def _yolo_class(meta: dict) -> int | None:
    """Return YOLO class id or None to skip this row."""
    pt    = (meta.get("person_type") or "").lower()
    label = (meta.get("label") or "").lower()
    if pt == "chalker" or label == "true_positive":
        return 1   # chalker
    # Skip rows with no usable signal (unlabeled negatives are fine as class 0).
    return 0


def _best_image(meta: dict, data_dir: Path) -> Path | None:
    """Return the best available image path for this event, or None."""
    for key in ("thumb_file", "hires_file"):
        val = meta.get(key, "")
        if val:
            p = data_dir / val
            if p.exists():
                return p
    # Fall back to first frame file
    for fname in (meta.get("frame_files") or "").split(","):
        fname = fname.strip()
        if fname:
            p = data_dir / fname
            if p.exists():
                return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db",             default=_DEFAULT_DB,   help="Path to ChromaDB directory")
    ap.add_argument("--data",           default=_DEFAULT_DATA, help="Path to dataset image directory")
    ap.add_argument("--out",            default=_DEFAULT_OUT,  help="Output directory for YOLO dataset")
    ap.add_argument("--min-yolo-conf",  type=float, default=0.0,
                    help="Minimum YOLO detection confidence to include a row (0 = all)")
    ap.add_argument("--max-negatives",  type=int,   default=0,
                    help="Cap the number of class-0 samples (0 = no cap). "
                         "Useful to balance against the small chalker set.")
    ap.add_argument("--chalker-only-labeled", action="store_true",
                    help="Only include chalker rows that have an explicit label "
                         "(true_positive). By default person_type=='chalker' also counts.")
    ap.add_argument("--val-ratio",      type=float, default=_VAL_RATIO)
    ap.add_argument("--apply",          action="store_true", help="Write files. Default is dry-run.")
    args = ap.parse_args()

    db_path   = _ROOT / args.db
    data_dir  = _ROOT / args.data
    out_dir   = _ROOT / args.out

    client = chromadb.PersistentClient(path=str(db_path))
    col    = client.get_or_create_collection(_COLLECTION)

    logger.info("Loading all events from %s …", db_path)
    result = col.get(include=["documents", "metadatas"], limit=100_000)
    ids       = result["ids"]
    docs      = result["documents"]
    metadatas = result["metadatas"]
    logger.info("Loaded %d events", len(ids))

    # ── Classify and filter ──────────────────────────────────────────────────
    chalkers:  list[tuple[Path, int, str]] = []  # (img_path, class_id, event_id)
    negatives: list[tuple[Path, int, str]] = []

    skipped_no_image  = 0
    skipped_vlm_error = 0
    skipped_low_conf  = 0

    for eid, doc, meta in zip(ids, docs, metadatas):
        if _is_vlm_error(doc):
            skipped_vlm_error += 1
            continue

        yolo_conf = float(meta.get("yolo_confidence") or 0.0)
        if args.min_yolo_conf > 0 and yolo_conf >= 0 and yolo_conf < args.min_yolo_conf:
            skipped_low_conf += 1
            continue

        img = _best_image(meta, data_dir)
        if img is None:
            skipped_no_image += 1
            continue

        pt    = (meta.get("person_type") or "").lower()
        label = (meta.get("label") or "").lower()

        if args.chalker_only_labeled:
            is_chalker = label == "true_positive"
        else:
            is_chalker = pt == "chalker" or label == "true_positive"

        cls = 1 if is_chalker else 0
        entry = (img, cls, eid)
        if cls == 1:
            chalkers.append(entry)
        else:
            negatives.append(entry)

    # ── Balance ──────────────────────────────────────────────────────────────
    rng = random.Random(_RANDOM_SEED)
    rng.shuffle(negatives)
    if args.max_negatives and len(negatives) > args.max_negatives:
        negatives = negatives[:args.max_negatives]

    all_entries = chalkers + negatives
    rng.shuffle(all_entries)

    n_val   = max(1, int(len(all_entries) * args.val_ratio))
    val_set = set(e[2] for e in all_entries[:n_val])

    class_counts = Counter(e[1] for e in all_entries)

    print("\n--- Export summary ---")
    print(f"  Total events in DB    : {len(ids)}")
    print(f"  Skipped (VLM error)   : {skipped_vlm_error}")
    print(f"  Skipped (no image)    : {skipped_no_image}")
    print(f"  Skipped (low conf)    : {skipped_low_conf}")
    print(f"  Exportable            : {len(all_entries)}")
    print(f"    class 0  person     : {class_counts[0]}")
    print(f"    class 1  chalker    : {class_counts[1]}")
    print(f"  Train / val split     : {len(all_entries)-n_val} / {n_val}")
    if not args.apply:
        print("\n  DRY-RUN — pass --apply to write files.")
        print("---------------------\n")
        return

    # ── Write files ───────────────────────────────────────────────────────────
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = 0
    for img_path, cls_id, eid in all_entries:
        split = "val" if eid in val_set else "train"
        stem  = f"{eid}_{img_path.stem}"

        dst_img = out_dir / "images" / split / f"{stem}.jpg"
        dst_lbl = out_dir / "labels" / split / f"{stem}.txt"

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text(f"{cls_id} {_PERSON_BBOX}\n")
        written += 1

    # data.yaml
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: person\n"
        f"  1: chalker\n"
    )

    print(f"\n  Written {written} images + labels → {out_dir}")
    print(f"  data.yaml → {yaml_path}")
    print("\nTo fine-tune:")
    print(f"  yolo train model=yolov8n.pt data={yaml_path} epochs=50 imgsz=320 batch=32")
    print("---------------------\n")


if __name__ == "__main__":
    main()
