"""Dataset maintenance CLI for the chalking vector store.

Subcommands:

  stats                 Summarise the vector store (counts, label coverage, etc.).
  purge-vlm-errors      Remove rows whose description is a VLM error/timeout/HTTP.
  purge-shadow          Remove first-pass rejections that the lenient VLM flagged
                        as 'no person' / 'shadow' / 'false alarm'.
  fix-confidence        Renormalise confidence values >1.5 (Qwen 0-100 scale).
  backfill-person-type  Run classify_person() over rows missing person_type.
  phash-cluster         Group rows by perceptual hash and emit a sample list for
                        labeling (one representative per cluster).
  rewrite-zoneped       Rewrite generic 'Zone pedestrian ...' descriptions to the
                        new diversified form so old rows become discriminable.

Every destructive subcommand defaults to --dry-run; pass --apply to actually
mutate the store. Image files are deleted only when their row is.

Usage:
    python -m scripts.dataset_maintenance stats
    python -m scripts.dataset_maintenance purge-vlm-errors --apply
    python -m scripts.dataset_maintenance backfill-person-type --limit 200 --apply
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# Ensure project root on sys.path so `src.*` imports work when run directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chromadb  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dataset_maintenance")

_DEFAULT_DB    = os.getenv("VECTOR_DB_PATH", "data/vectors")
_DEFAULT_DATA  = os.getenv("DATASET_PATH",   "dataset")
_COLLECTION    = "chalking_evals"

_VLM_ERROR_PREFIXES = (
    "VLM ",
    "Model not found",
    "Classification failed",
)
_VLM_ERROR_SUBSTRINGS = (
    "VLM timeout",
    "VLM unreachable",
    "VLM HTTP",
    "VLM error",
    "VLM analysis failed",
)

_SHADOW_SUBSTRINGS = (
    "no person",
    "appears to be a shadow",
    "appears to be a false alarm",
    "no person detected",
)


def _open(db_path: str):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(_COLLECTION)


def _is_vlm_error(desc: str) -> bool:
    if not desc:
        return False
    if desc.startswith(_VLM_ERROR_PREFIXES):
        return True
    return any(s in desc for s in _VLM_ERROR_SUBSTRINGS)


def _is_shadow_rejection(desc: str, meta: dict) -> bool:
    """True for first-pass lenient-VLM rejections (no real chalking signal)."""
    if not desc:
        return False
    d = desc.lower()
    if not any(s in d for s in _SHADOW_SUBSTRINGS):
        return False
    # Only purge first-pass rejections — never delete a detected=1 row.
    if int(meta.get("detected", 0)) == 1:
        return False
    # Preserve manually labeled rows.
    if (meta.get("label") or "").strip():
        return False
    return True


def _delete_rows(col, dataset_dir: Path, rows: list[tuple[str, dict]], apply: bool) -> int:
    if not rows:
        return 0
    ids = [eid for eid, _ in rows]
    files: list[str] = []
    for _, m in rows:
        for k in ("thumb_file", "hires_file"):
            f = m.get(k)
            if f:
                files.append(f)
        for f in (m.get("frame_files") or "").split(","):
            if f:
                files.append(f)
    if not apply:
        logger.info("[DRY-RUN] would delete %d rows, %d files", len(ids), len(files))
        return len(ids)
    # Chroma delete() handles large id lists; batch defensively.
    for chunk_start in range(0, len(ids), 500):
        col.delete(ids=ids[chunk_start:chunk_start + 500])
    removed = 0
    for f in files:
        try:
            (dataset_dir / f).unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    logger.info("Deleted %d rows, removed %d files", len(ids), removed)
    return len(ids)


# ── Subcommands ──────────────────────────────────────────────────────────────

def cmd_stats(args) -> None:
    col = _open(args.db)
    res = col.get(include=["metadatas", "documents"], limit=200_000)
    metas, docs, ids = res["metadatas"], res["documents"], res["ids"]
    n = len(ids)
    print(f"Total rows: {n}")
    print(f"Unique descriptions: {len(set(d for d in docs if d))}")

    by_source = Counter(m.get("capture_source") or "(empty)" for m in metas)
    print("\ncapture_source:")
    for k, v in by_source.most_common():
        print(f"  {v:7d}  {k}")

    by_detected = Counter(int(m.get("detected", 0)) for m in metas)
    print("\ndetected:")
    for k, v in by_detected.most_common():
        print(f"  {v:7d}  {k}")

    by_label = Counter((m.get("label") or "").strip() or "(unlabeled)" for m in metas)
    print("\nlabel:")
    for k, v in by_label.most_common():
        print(f"  {v:7d}  {k}")

    by_pt = Counter((m.get("person_type") or "").strip() or "(empty)" for m in metas)
    print("\nperson_type:")
    for k, v in by_pt.most_common():
        print(f"  {v:7d}  {k}")

    confs = [float(m.get("confidence", 0.0)) for m in metas]
    if confs:
        over_1 = sum(1 for c in confs if c > 1.5)
        print(f"\nconfidence: min={min(confs):.2f}  max={max(confs):.2f}  >1.5 count={over_1}")

    err = sum(1 for d in docs if _is_vlm_error(d or ""))
    shadow = sum(1 for d, m in zip(docs, metas) if _is_shadow_rejection(d or "", m))
    print(f"\nVLM-error rows:        {err}")
    print(f"Shadow/no-person rows: {shadow}")


def cmd_purge_vlm_errors(args) -> None:
    col = _open(args.db)
    res = col.get(include=["metadatas", "documents"], limit=200_000)
    rows = [
        (eid, m)
        for eid, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        if _is_vlm_error(d or "")
    ]
    logger.info("Matched %d VLM-error rows", len(rows))
    _delete_rows(col, Path(args.dataset), rows, args.apply)


def cmd_purge_shadow(args) -> None:
    col = _open(args.db)
    res = col.get(include=["metadatas", "documents"], limit=200_000)
    rows = [
        (eid, m)
        for eid, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        if _is_shadow_rejection(d or "", m)
    ]
    logger.info("Matched %d shadow/no-person rejection rows", len(rows))
    _delete_rows(col, Path(args.dataset), rows, args.apply)


def cmd_fix_confidence(args) -> None:
    col = _open(args.db)
    res = col.get(include=["metadatas"], limit=200_000)
    to_update_ids: list[str] = []
    to_update_meta: list[dict] = []
    for eid, m in zip(res["ids"], res["metadatas"]):
        try:
            c = float(m.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if c <= 1.5:
            continue
        new = min(1.0, c / 100.0) if c <= 100.0 else 1.0
        if abs(new - c) < 1e-6:
            continue
        meta = dict(m)
        meta["confidence"] = new
        to_update_ids.append(eid)
        to_update_meta.append(meta)
    logger.info("Would normalise %d confidence values (>1.5 → /100)", len(to_update_ids))
    if not args.apply or not to_update_ids:
        return
    for i in range(0, len(to_update_ids), 500):
        col.update(
            ids=to_update_ids[i:i + 500],
            metadatas=to_update_meta[i:i + 500],
        )
    logger.info("Updated %d rows", len(to_update_ids))


def cmd_backfill_person_type(args) -> None:
    """Run classify_person() over rows without a person_type.

    Reads the stored thumbnail JPEG from disk and sends it through the
    classifier. The result is written back as metadata. Limited to --limit
    rows per invocation so a long run can be paused.
    """
    from src.vlm.analyzer import VLMAnalyzer  # noqa: WPS433 (deferred import)

    col = _open(args.db)
    res = col.get(include=["metadatas"], limit=200_000)
    candidates: list[tuple[str, dict]] = []
    for eid, m in zip(res["ids"], res["metadatas"]):
        if (m.get("person_type") or "").strip():
            continue
        # Need a thumbnail file to classify from.
        if not m.get("thumb_file"):
            continue
        # Optional capture_source filter (e.g. only 'chalking' near-vehicle rows).
        if args.source and (m.get("capture_source") or "chalking") != args.source:
            continue
        candidates.append((eid, m))
    logger.info(
        "Candidates without person_type: %d (source=%s, limit=%d)",
        len(candidates), args.source or "all", args.limit,
    )
    if not candidates:
        return

    vlm = VLMAnalyzer(
        backend=os.getenv("VLM_BACKEND", "claude"),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llava:7b-v1.6-mistral-q4_K_M"),
    )

    dataset_dir = Path(args.dataset)
    updated = 0
    for eid, m in candidates[: args.limit]:
        thumb_path = dataset_dir / m["thumb_file"]
        if not thumb_path.exists():
            continue
        img_bytes = thumb_path.read_bytes()
        try:
            result = vlm.classify_person(img_bytes)
        except Exception:
            logger.exception("classify failed for %s", eid)
            continue
        # Do NOT write on a failed/fallback classification — that would pollute
        # the row with a bogus 'unknown' and make it ineligible for retry.
        if (result.get("description") or "").startswith(("Classification failed", "Classification error")):
            logger.warning("classify returned failure fallback for %s — skipping write", eid)
            continue
        pt = (result.get("person_type") or "").strip() or "unknown"
        meta = dict(m)
        meta["person_type"] = pt
        if args.apply:
            try:
                col.update(ids=[eid], metadatas=[meta])
                updated += 1
            except Exception:
                logger.exception("update failed for %s", eid)
        else:
            logger.info("[DRY-RUN] %s → %s", eid, pt)
        if updated and updated % 25 == 0:
            logger.info("Progress: %d/%d updated", updated, args.limit)
    logger.info("Backfill complete: %d/%d updated", updated, min(args.limit, len(candidates)))


def cmd_phash_cluster(args) -> None:
    """Group rows by perceptual hash (Hamming-distance buckets) and emit a CSV
    of one representative per cluster, biased toward larger clusters first.

    Useful for finding hard-negative families to label without reviewing every
    single frame: label 1 per cluster, then propagate the label to siblings.
    """
    col = _open(args.db)
    res = col.get(include=["metadatas", "documents"], limit=200_000)
    # Group by exact phash first — Hamming clustering is too slow on 20k+
    # entries and exact-match groups already capture the heavy-tail repeats.
    groups: dict[str, list[tuple[str, dict, str]]] = defaultdict(list)
    no_phash = 0
    for eid, d, m in zip(res["ids"], res["documents"], res["metadatas"]):
        ph = (m.get("thumb_phash") or "").strip()
        if not ph:
            no_phash += 1
            continue
        groups[ph].append((eid, m, d or ""))
    logger.info("Phash clusters: %d  (rows without phash: %d)", len(groups), no_phash)
    # Sort clusters by size descending.
    sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("cluster_size,phash,sample_id,thumb_file,capture_source,confidence,description\n")
        for ph, members in sorted_groups[: args.limit]:
            eid, m, d = members[0]
            fh.write(
                f"{len(members)},{ph},{eid},{m.get('thumb_file','')},"
                f"{m.get('capture_source','')},{float(m.get('confidence', 0.0)):.3f},"
                f"\"{(d or '').replace(chr(34), chr(39))[:160]}\"\n"
            )
    top = sorted_groups[:10]
    logger.info("Top 10 clusters by size:")
    for ph, members in top:
        sample = members[0][2][:100].replace("\n", " ")
        logger.info("  %5d  phash=%s  e.g. %r", len(members), ph, sample)
    logger.info("Wrote %d cluster representatives to %s", min(args.limit, len(sorted_groups)), out_path)


def cmd_clip_backfill(args) -> None:
    """Populate the CLIP image-embedding collection from existing thumbnails.

    Requires CLIP_EMBEDDINGS_ENABLED=true (and open-clip-torch installed).
    Skips rows already present in the CLIP collection.
    """
    os.environ["CLIP_EMBEDDINGS_ENABLED"] = "true"
    from src.storage.vector_store import EventVectorStore, _decode_jpeg_bytes_to_rgb  # noqa: E402

    vs = EventVectorStore(db_path=args.db, dataset_path=args.dataset)
    if vs._clip_col is None:
        logger.error("CLIP collection unavailable — install open-clip-torch and pillow")
        return
    main_res = vs._col.get(include=["metadatas"], limit=200_000)
    existing = set(vs._clip_col.get(include=[], limit=200_000)["ids"])
    todo = [
        (eid, m) for eid, m in zip(main_res["ids"], main_res["metadatas"])
        if eid not in existing and m.get("thumb_file")
    ]
    logger.info("CLIP backfill: %d rows pending (limit=%d)", len(todo), args.limit)
    if not args.apply:
        logger.info("[DRY-RUN] pass --apply to embed and write")
        return
    dataset_dir = Path(args.dataset)
    added = 0
    for eid, m in todo[: args.limit]:
        path = dataset_dir / m["thumb_file"]
        if not path.exists():
            continue
        arr = _decode_jpeg_bytes_to_rgb(path.read_bytes())
        if arr is None:
            continue
        try:
            vs._clip_col.add(
                ids=[eid],
                images=[arr],
                metadatas=[{
                    "detected":       int(m.get("detected", 0)),
                    "label":          m.get("label", ""),
                    "capture_source": m.get("capture_source", ""),
                    "camera_id":      int(m.get("camera_id", 0)),
                }],
            )
            added += 1
        except Exception:
            logger.exception("CLIP add failed for %s", eid)
        if added and added % 50 == 0:
            logger.info("Progress: %d/%d added", added, args.limit)
    logger.info("CLIP backfill complete: %d added", added)


def cmd_merge(args) -> None:
    """Merge a source vector store into the target (--db).

    Copies records (documents + embeddings + metadata) and their referenced
    image files from --from-db / --from-dataset into the target. Records whose
    id already exists in the target are skipped, so the merge is idempotent and
    safe to re-run. Event ids are timestamp-prefixed; non-overlapping capture
    windows will never collide.
    """
    import shutil

    target = _open(args.db)
    src    = _open(args.from_db)
    src_ds = Path(args.from_dataset)
    tgt_ds = Path(args.dataset)
    tgt_ds.mkdir(parents=True, exist_ok=True)

    existing = set(target.get(include=[], limit=500_000)["ids"])
    src_res = src.get(include=["documents", "embeddings", "metadatas"], limit=500_000)
    src_ids = src_res["ids"]
    new_idx = [i for i, eid in enumerate(src_ids) if eid not in existing]
    logger.info(
        "Source rows: %d  already-in-target: %d  to-merge: %d",
        len(src_ids), len(src_ids) - len(new_idx), len(new_idx),
    )
    if not new_idx:
        return

    # Collect image files to copy from the metadata of the new rows.
    files: list[str] = []
    for i in new_idx:
        m = src_res["metadatas"][i]
        for k in ("thumb_file", "hires_file"):
            if m.get(k):
                files.append(m[k])
        for f in (m.get("frame_files") or "").split(","):
            if f:
                files.append(f)

    if not args.apply:
        logger.info("[DRY-RUN] would add %d records and copy up to %d image files",
                    len(new_idx), len(files))
        return

    # Copy image files first so a row never references a missing file.
    copied = 0
    for fname in files:
        s = src_ds / fname
        d = tgt_ds / fname
        if s.exists() and not d.exists():
            try:
                shutil.copy2(s, d)
                copied += 1
            except Exception:
                logger.debug("copy failed: %s", fname, exc_info=True)
    logger.info("Copied %d image files", copied)

    # Add records in batches, preserving precomputed embeddings.
    docs   = [src_res["documents"][i] for i in new_idx]
    embeds = [src_res["embeddings"][i] for i in new_idx]
    metas  = [src_res["metadatas"][i] for i in new_idx]
    ids    = [src_ids[i] for i in new_idx]
    for s in range(0, len(ids), 500):
        target.add(
            ids=ids[s:s + 500],
            documents=docs[s:s + 500],
            embeddings=embeds[s:s + 500],
            metadatas=metas[s:s + 500],
        )
    logger.info("Merged %d records into target (%d total now)", len(ids), target.count())


def cmd_rewrite_zoneped(args) -> None:
    """Rewrite legacy 'Zone pedestrian - not near any vehicle' descriptions to
    the new diversified form using each row's stored bbox metadata.

    Older rows have no bbox stored, so we derive coarse position from the
    thumbnail filename only (cluster size). When bbox info is missing we keep
    a deterministic per-row description based on event_id hash, which is still
    far more diverse than a single shared string.
    """
    col = _open(args.db)
    res = col.get(include=["metadatas", "documents"], limit=200_000)
    LEGACY = "Zone pedestrian"
    target = [
        (eid, m, d) for eid, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        if (m.get("capture_source", "") == "zone_pedestrian")
        and (d or "").startswith(LEGACY)
        and "of frame" not in (d or "")  # already-rewritten rows contain this token
    ]
    logger.info("Legacy zone_pedestrian rows: %d", len(target))
    if not target:
        return
    updates_ids: list[str] = []
    updates_docs: list[str] = []
    for eid, m, _d in target:
        # Without bbox, fall back to id-hash bucket for spatial diversity.
        h = int(hashlib.md5(eid.encode()).hexdigest()[:8], 16)
        col_label = ["left", "centre", "right"][h % 3]
        row_label = ["top", "middle", "bottom"][(h >> 4) % 3]
        scale_label = ["small", "medium", "large"][(h >> 8) % 3]
        pt = (m.get("person_type") or "").strip() or "unclassified"
        conf = float(m.get("yolo_confidence", 0.0))
        new = (
            f"Zone pedestrian ({pt}), {row_label}-{col_label} of frame, "
            f"{scale_label} bbox (yolo={conf:.2f}). No nearby vehicle."
        )
        updates_ids.append(eid)
        updates_docs.append(new)
    if not args.apply:
        logger.info("[DRY-RUN] would rewrite %d rows  (example: %r)", len(updates_ids), updates_docs[0])
        return
    for i in range(0, len(updates_ids), 500):
        col.update(
            ids=updates_ids[i:i + 500],
            documents=updates_docs[i:i + 500],
        )
    logger.info("Rewrote %d zone_pedestrian descriptions", len(updates_ids))


# ── Entry ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=_DEFAULT_DB, help="ChromaDB persistent path (default: %(default)s)")
    p.add_argument("--dataset", default=_DEFAULT_DATA, help="Image dataset directory (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="Summarise the collection").set_defaults(func=cmd_stats)

    sp = sub.add_parser("purge-vlm-errors", help="Delete VLM error/timeout rows")
    sp.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")
    sp.set_defaults(func=cmd_purge_vlm_errors)

    sp = sub.add_parser("purge-shadow", help="Delete first-pass shadow/no-person rejections")
    sp.add_argument("--apply", action="store_true")
    sp.set_defaults(func=cmd_purge_shadow)

    sp = sub.add_parser("fix-confidence", help="Normalise confidence > 1.5 by /100")
    sp.add_argument("--apply", action="store_true")
    sp.set_defaults(func=cmd_fix_confidence)

    sp = sub.add_parser("backfill-person-type", help="Run classify_person() over unclassified rows")
    sp.add_argument("--limit", type=int, default=100, help="Max rows to process this run")
    sp.add_argument("--source", default="", help="Only rows with this capture_source (e.g. 'chalking')")
    sp.add_argument("--apply", action="store_true")
    sp.set_defaults(func=cmd_backfill_person_type)

    sp = sub.add_parser("phash-cluster", help="Export top-N phash clusters to CSV for labeling")
    sp.add_argument("--limit", type=int, default=500)
    sp.add_argument("--output", default="dataset/phash_clusters.csv")
    sp.set_defaults(func=cmd_phash_cluster)

    sp = sub.add_parser("rewrite-zoneped", help="Rewrite legacy zone_pedestrian descriptions")
    sp.add_argument("--apply", action="store_true")
    sp.set_defaults(func=cmd_rewrite_zoneped)

    sp = sub.add_parser("clip-backfill", help="Populate CLIP collection from existing thumbnails")
    sp.add_argument("--limit", type=int, default=1000)
    sp.add_argument("--apply", action="store_true")
    sp.set_defaults(func=cmd_clip_backfill)

    sp = sub.add_parser("merge", help="Merge another vector store into --db (idempotent)")
    sp.add_argument("--from-db", required=True, help="Source ChromaDB path")
    sp.add_argument("--from-dataset", required=True, help="Source dataset image dir")
    sp.add_argument("--apply", action="store_true")
    sp.set_defaults(func=cmd_merge)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
