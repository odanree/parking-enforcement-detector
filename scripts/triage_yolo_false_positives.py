"""LLM-assisted triage of suspected YOLO false positives.

Pulls high-confidence YOLO detections that the pipeline rejected (detected=0)
and asks Claude Vision whether a real person is actually present. Results:

  "no"        -> label = false_positive  (confirmed YOLO FP, use as hard negative)
  "yes"       -> label = true_negative   (VLM was wrong, YOLO was right)
  "uncertain" -> queued for human review

Writes results back to ChromaDB (--apply) and produces an HTML review queue
for the uncertain cases.

Usage
-----
  # Dry-run — shows what would happen, writes nothing:
  python -m scripts.triage_yolo_false_positives

  # Run and write labels:
  python -m scripts.triage_yolo_false_positives --apply

  # Limit batch size for testing:
  python -m scripts.triage_yolo_false_positives --apply --limit 50

  # Use a cheaper model:
  python -m scripts.triage_yolo_false_positives --apply --model claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import anthropic
import chromadb

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("triage_yolo_fp")

_DEFAULT_DB     = os.getenv("VECTOR_DB_PATH", "data/vectors")
_DEFAULT_DATA   = os.getenv("DATASET_PATH",   "dataset")
_DEFAULT_MODEL  = "claude-haiku-4-5-20251001"   # cheap + fast for binary vision task
_COLLECTION     = "chalking_evals"
_MIN_YOLO_CONF  = 0.60   # only triage confident YOLO detections
_RATE_LIMIT_S   = 0.3    # seconds between API calls

_SYSTEM_PROMPT = (
    "You are reviewing crops from a parking surveillance camera. "
    "Your only job is to decide whether a real human person is clearly "
    "visible in the image. Inanimate objects — trees, bushes, signs, "
    "fence posts, fire hydrants, car parts, shadows — are NOT people. "
    "Answer with exactly one word: yes, no, or uncertain. "
    "Use 'uncertain' only when you genuinely cannot tell."
)

_USER_PROMPT = (
    "Is there a real human person visible in this image? "
    "Answer with one word only: yes, no, or uncertain."
)


def _load_image_b64(path: Path) -> str | None:
    try:
        return base64.standard_b64encode(path.read_bytes()).decode()
    except Exception:
        return None


def _ask_claude(client: anthropic.Anthropic, img_b64: str, model: str) -> str:
    """Returns 'yes', 'no', or 'uncertain'."""
    msg = client.messages.create(
        model=model,
        max_tokens=10,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": _USER_PROMPT},
            ],
        }],
    )
    answer = msg.content[0].text.strip().lower().rstrip(".")
    if "uncertain" in answer:
        return "uncertain"
    if answer.startswith("yes"):
        return "yes"
    if answer.startswith("no"):
        return "no"
    return "uncertain"   # treat unexpected responses as uncertain


def _best_image(meta: dict, data_dir: Path) -> Path | None:
    for key in ("thumb_file", "hires_file"):
        val = meta.get(key, "")
        if val:
            p = data_dir / val
            if p.exists():
                return p
    return None


def _build_review_html(uncertain: list[tuple[dict, str, Path]], out_path: Path) -> None:
    cards = []
    for meta, doc, img_path in uncertain:
        yc = float(meta.get("yolo_confidence") or 0)
        ts = meta.get("timestamp", 0)
        from datetime import datetime
        dt = datetime.fromtimestamp(float(ts)).strftime("%m/%d %H:%M") if ts else ""
        eid = meta.get("id", "")
        cards.append(f"""
        <div class="card">
          <img src="{img_path.as_posix()}" />
          <div class="meta">
            <b>{dt}</b> yolo={yc:.0%}<br>
            <small>{doc[:100]}</small><br>
            <code style="font-size:10px">{eid}</code>
          </div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Human Review Queue ({len(uncertain)} uncertain)</title>
<style>
body {{ background:#111; color:#eee; font-family:sans-serif; margin:16px; }}
h1 {{ color:#fa0 }}
p {{ color:#aaa }}
.grid {{ display:flex; flex-wrap:wrap; gap:12px; }}
.card {{ background:#1a1a1a; border:1px solid #555; border-radius:6px; width:220px; overflow:hidden; }}
.card img {{ width:100%; height:170px; object-fit:cover; display:block; }}
.meta {{ padding:6px; font-size:11px; line-height:1.6; }}
</style></head>
<body>
<h1>Human Review Queue — {len(uncertain)} uncertain cases</h1>
<p>Claude could not confidently classify these. Label them manually in the
pipeline kanban using the FP / TN buttons.</p>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db",           default=_DEFAULT_DB)
    ap.add_argument("--data",         default=_DEFAULT_DATA)
    ap.add_argument("--model",        default=_DEFAULT_MODEL)
    ap.add_argument("--min-yolo-conf", type=float, default=_MIN_YOLO_CONF)
    ap.add_argument("--limit",        type=int, default=0, help="Max rows to process (0=all)")
    ap.add_argument("--apply",        action="store_true", help="Write labels to ChromaDB")
    args = ap.parse_args()

    db_path  = _ROOT / args.db
    data_dir = _ROOT / args.data

    client  = anthropic.Anthropic()
    chroma  = chromadb.PersistentClient(path=str(db_path))
    col     = chroma.get_or_create_collection(_COLLECTION)

    logger.info("Loading events ...")
    result = col.get(include=["metadatas", "documents"], limit=100_000)

    # Candidates: YOLO confident, VLM rejected, not yet human-labeled, has image
    candidates = []
    for eid, meta, doc in zip(result["ids"], result["metadatas"], result["documents"]):
        if int(meta.get("detected", 0)):
            continue
        if meta.get("label", ""):
            continue   # already labeled — skip
        yc = float(meta.get("yolo_confidence") or -1)
        if yc < args.min_yolo_conf:
            continue
        img = _best_image(meta, data_dir)
        if img is None:
            continue
        meta["id"] = eid
        candidates.append((meta, doc, img))

    total = len(candidates)
    if args.limit:
        candidates = candidates[:args.limit]

    logger.info("Candidates: %d total, processing %d (min_yolo_conf=%.0f%%)",
                total, len(candidates), args.min_yolo_conf * 100)

    if not candidates:
        print("No candidates found.")
        return

    counts = {"no": 0, "yes": 0, "uncertain": 0, "error": 0}
    uncertain_rows: list[tuple[dict, str, Path]] = []

    for i, (meta, doc, img_path) in enumerate(candidates, 1):
        img_b64 = _load_image_b64(img_path)
        if not img_b64:
            counts["error"] += 1
            continue

        try:
            verdict = _ask_claude(client, img_b64, args.model)
        except Exception as e:
            logger.warning("Claude error on %s: %s", meta["id"], e)
            counts["error"] += 1
            continue

        label_map = {"no": "false_positive", "yes": "true_negative"}
        new_label = label_map.get(verdict)

        if verdict == "uncertain":
            uncertain_rows.append((meta, doc, img_path))

        counts[verdict] += 1

        if args.apply and new_label:
            try:
                existing = col.get(ids=[meta["id"]], include=["metadatas"])
                m = existing["metadatas"][0]
                m["label"] = new_label
                col.update(ids=[meta["id"]], metadatas=[m])
            except Exception as e:
                logger.warning("ChromaDB update failed for %s: %s", meta["id"], e)

        if i % 10 == 0 or i == len(candidates):
            logger.info("[%d/%d] no=%d yes=%d uncertain=%d error=%d",
                        i, len(candidates), counts["no"], counts["yes"],
                        counts["uncertain"], counts["error"])

        time.sleep(_RATE_LIMIT_S)

    # Summary
    print("\n--- Triage results ---")
    print(f"  Processed      : {len(candidates)}")
    print(f"  false_positive : {counts['no']}  (confirmed YOLO FP -> hard negative training data)")
    print(f"  true_negative  : {counts['yes']} (VLM was wrong, YOLO was right)")
    print(f"  uncertain      : {counts['uncertain']} (needs human review)")
    print(f"  errors         : {counts['error']}")
    if not args.apply:
        print("\n  DRY-RUN -- pass --apply to write labels to ChromaDB")

    if uncertain_rows:
        out = _ROOT / "yolo_fp_review_queue.html"
        _build_review_html(uncertain_rows, out)
        print(f"\n  Human review queue -> {out}")

    print("---------------------\n")


if __name__ == "__main__":
    main()
