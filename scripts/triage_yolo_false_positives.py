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

# If the VLM description already confirms a person, YOLO was correct — skip.
# We only want events where the detection may be a tree/hydrant/fence FP.
_SKIP_DOC_PHRASES = [
    # VLM described a real person in the scene
    "the person is",
    "near the person",
    "person's hand",
    "person is present",
    "individual is",
    "subject is",
    "human is",
    "pedestrian",
    "person walking",
    "person standing",
    "person crouching",
    # Person pre-classified by person_classifier — YOLO was right
    "classifier short-circuit",
    # Chalking-related rejections (real person, not chalking)
    "no chalking",
    "not engaged",
    "no evidence of chalking",
    "no chalk",
    # Model/pipeline errors — not useful training data
    "model not found",
]

_SYSTEM_PROMPT = (
    "You are auditing YOLO person detections from a parking surveillance camera. "
    "Each image has a GREEN bounding box drawn around what YOLO classified as a 'person'. "
    "Your only job: is the object INSIDE the green box a real human person, "
    "or is it a false alarm? "
    "False alarms include trees, bushes, shrubs, fire hydrants, fence posts, signs, "
    "car mirrors, shadows, or any inanimate object. "
    "Ignore everything outside the green box — focus only on what is boxed. "
    "Answer with exactly one word: yes (real person), no (false alarm), or uncertain."
)

_USER_PROMPT = (
    "Is the object inside the GREEN bounding box a real human person, "
    "or a false alarm (tree, hydrant, fence, shadow, etc.)? "
    "Answer with one word: yes, no, or uncertain."
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
    # Prefer hires (NVR full-res) — distant detections are too small in the 640px thumb
    for key in ("hires_file", "thumb_file"):
        val = meta.get(key, "")
        if val:
            p = data_dir / val
            if p.exists():
                return p
    return None


_LABEL_STYLE = {
    "false_positive": ("fp",   "#e74",  "FALSE POSITIVE",  "YOLO fired on non-person"),
    "true_negative":  ("tn",   "#4a4",  "TRUE NEGATIVE",   "Real person, not chalking"),
    "uncertain":      ("unk",  "#fa0",  "UNCERTAIN",       "Needs human review"),
}


def _card_html(meta: dict, doc: str, img_path: Path, verdict: str) -> str:
    from datetime import datetime
    yc  = float(meta.get("yolo_confidence") or 0)
    ts  = meta.get("timestamp", 0)
    dt  = datetime.fromtimestamp(float(ts)).strftime("%m/%d %H:%M") if ts else ""
    eid = meta.get("id", "")
    cls, color, badge, tip = _LABEL_STYLE.get(verdict, ("unk", "#888", verdict.upper(), ""))
    return f"""
    <div class="card {cls}">
      <div class="badge" style="background:{color}">{badge}</div>
      <a href="{img_path.as_posix()}" target="_blank" title="Click for full size">
        <img src="{img_path.as_posix()}" />
      </a>
      <div class="meta">
        <b>{dt}</b> yolo={yc:.0%}<br>
        <small style="color:#aaa">{tip}</small><br>
        <small>{doc[:120]}</small><br>
        <code style="font-size:9px;color:#777">{eid}</code>
      </div>
    </div>"""


def _build_results_html(
    labeled: list[tuple[dict, str, Path, str]],   # (meta, doc, img, verdict)
    uncertain: list[tuple[dict, str, Path]],
    out_path: Path,
) -> None:
    fp_cards  = [_card_html(m, d, p, "false_positive") for m, d, p, v in labeled if v == "no"]
    tn_cards  = [_card_html(m, d, p, "true_negative")  for m, d, p, v in labeled if v == "yes"]
    unk_cards = [_card_html(m, d, p, "uncertain")       for m, d, p in uncertain]

    def section(title: str, cards: list[str], color: str) -> str:
        if not cards:
            return ""
        return f"""
        <h2 style="color:{color}">{title} ({len(cards)})</h2>
        <div class="grid">{''.join(cards)}</div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Triage Results — {len(labeled) + len(uncertain)} events</title>
<style>
body {{ background:#111; color:#eee; font-family:sans-serif; margin:16px; }}
h1 {{ color:#fff }} h2 {{ margin-top:24px }}
p  {{ color:#aaa }}
.grid {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:8px; }}
.card {{ background:#1a1a1a; border-radius:6px; width:220px; overflow:hidden; position:relative; }}
.card.fp {{ border:2px solid #e74 }}
.card.tn {{ border:2px solid #4a4 }}
.card.unk {{ border:2px solid #fa0 }}
.badge {{ font-size:10px; font-weight:bold; color:#fff; padding:2px 6px; position:absolute; top:4px; left:4px; border-radius:3px; }}
.card img {{ width:100%; height:160px; object-fit:cover; display:block; cursor:pointer; }}
.card a {{ display:block; }}
.meta {{ padding:6px; font-size:11px; line-height:1.6; }}
</style></head>
<body>
<h1>Triage Results</h1>
<p>
  <span style="color:#e74">■</span> {len(fp_cards)} false positives (YOLO FP → hard negative training data) &nbsp;
  <span style="color:#4a4">■</span> {len(tn_cards)} true negatives (real person, not chalking) &nbsp;
  <span style="color:#fa0">■</span> {len(unk_cards)} uncertain (needs human review)
</p>
{section("FALSE POSITIVES — YOLO fired on non-person (use as hard negatives)", fp_cards, "#e74")}
{section("UNCERTAIN — needs human review", unk_cards, "#fa0")}
{section("TRUE NEGATIVES — real person, not chalking", tn_cards, "#4a4")}
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db",           default=_DEFAULT_DB)
    ap.add_argument("--data",         default=_DEFAULT_DATA)
    ap.add_argument("--model",        default=_DEFAULT_MODEL)
    ap.add_argument("--min-yolo-conf", type=float, default=_MIN_YOLO_CONF)
    ap.add_argument("--limit",         type=int, default=0, help="Max rows to process (0=all)")
    ap.add_argument("--apply",         action="store_true", help="Write labels to ChromaDB")
    ap.add_argument("--no-doc-filter", action="store_true",
                    help="Disable VLM-description pre-filter (include person-present events too)")
    ap.add_argument("--relabel-fps",   action="store_true",
                    help="Re-evaluate existing false_positive labels using hires images")
    args = ap.parse_args()

    db_path  = _ROOT / args.db
    data_dir = _ROOT / args.data

    client  = anthropic.Anthropic()
    chroma  = chromadb.PersistentClient(path=str(db_path))
    col     = chroma.get_or_create_collection(_COLLECTION)

    logger.info("Loading events ...")
    result = col.get(include=["metadatas", "documents"], limit=100_000)

    # Candidates: YOLO confident, VLM rejected, not yet human-labeled, has image.
    # Skip events whose VLM description already mentions a person — those are
    # true negatives (YOLO was right, person wasn't chalking) and don't need triage.
    # We only want events where YOLO may have fired on a non-person object.
    #
    # --relabel-fps mode: re-evaluate existing false_positive labels using hires images.
    candidates = []
    skipped_person_desc = 0
    for eid, meta, doc in zip(result["ids"], result["metadatas"], result["documents"]):
        if args.relabel_fps:
            if meta.get("label") != "false_positive":
                continue
        else:
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
        doc_lower = (doc or "").lower()
        if not args.no_doc_filter and not args.relabel_fps and any(p in doc_lower for p in _SKIP_DOC_PHRASES):
            skipped_person_desc += 1
            continue
        meta["id"] = eid
        candidates.append((meta, doc, img))

    total = len(candidates)
    if args.limit:
        candidates = candidates[:args.limit]

    logger.info(
        "Candidates: %d (skipped %d person-description events), processing %d (min_yolo_conf=%.0f%%)",
        total, skipped_person_desc, len(candidates), args.min_yolo_conf * 100,
    )

    if not candidates:
        print("No candidates found.")
        return

    if not args.apply:
        print(f"\n  DRY-RUN -- would process {len(candidates)} candidates with {args.model}")
        print(f"  Estimated cost: ~${len(candidates) * 0.0006:.2f} (haiku) or ~${len(candidates) * 0.0018:.2f} (sonnet)")
        print("  Pass --apply to run.\n")
        return

    counts = {"no": 0, "yes": 0, "uncertain": 0, "error": 0}
    labeled_rows: list[tuple[dict, str, Path, str]] = []   # (meta, doc, img, verdict)
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

        labeled_rows.append((meta, doc, img_path, verdict))
        if verdict == "uncertain":
            uncertain_rows.append((meta, doc, img_path))

        counts[verdict] += 1

        if new_label:
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
    print(f"  true_negative  : {counts['yes']} (real person, not chalking)")
    print(f"  uncertain      : {counts['uncertain']} (needs human review)")
    print(f"  errors         : {counts['error']}")

    # Always generate a results gallery so you can audit Claude's verdicts
    gallery_path = _ROOT / "triage_results.html"
    _build_results_html(labeled_rows, uncertain_rows, gallery_path)
    print(f"\n  Results gallery -> {gallery_path}")
    print("---------------------\n")


if __name__ == "__main__":
    main()
