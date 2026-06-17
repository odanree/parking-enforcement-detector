"""Rebuild ChromaDB from the SQLite metadata store.

The HNSW vector index (link_lists.bin) can grow corrupt/bloated when Docker
is force-killed mid-write.  The SQLite metadata store is always intact.

This script:
  1. Reads every event directly from chroma.sqlite3 (bypasses the Rust backend)
  2. Writes a fresh ChromaDB at data/vectors_new
  3. Swaps data/vectors_new -> data/vectors (after user confirmation or --apply)

Run inside Docker so the DB lives on ext4:
  docker compose run --rm detector python -m scripts.repair_chromadb [--apply]
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
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

import chromadb

_DB_DIR   = _ROOT / "data/vectors"
_NEW_DIR  = _ROOT / "data/vectors_new"
_SQLITE   = _DB_DIR / "chroma.sqlite3"


def read_events_from_sqlite(sqlite_path: Path) -> list[dict]:
    """Read all events + metadata + documents directly from chroma.sqlite3."""
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row

    # Get collection ID for chalking_evals
    col_row = conn.execute(
        "SELECT id FROM collections WHERE name = 'chalking_evals'"
    ).fetchone()
    if col_row is None:
        print("ERROR: collection 'chalking_evals' not found in SQLite")
        return []
    col_id = col_row["id"]

    # Get metadata segment id for this collection
    seg_row = conn.execute(
        "SELECT id FROM segments WHERE collection = ? AND type = 'urn:chroma:segment/metadata/sqlite'",
        (col_id,)
    ).fetchone()
    if seg_row is None:
        print("ERROR: metadata segment not found")
        return []
    seg_id = seg_row["id"]
    print(f"Collection '{col_id}', segment '{seg_id}'")

    # embeddings.id   = internal integer PK
    # embeddings.embedding_id = user-visible string ID ("1748xxx_abc123")
    # embedding_metadata.id   = FK to embeddings.id (integer)
    rows = conn.execute(
        """
        SELECT e.id AS int_id, e.embedding_id AS event_id,
               em.key, em.string_value, em.int_value, em.float_value, em.bool_value
        FROM embeddings e
        LEFT JOIN embedding_metadata em ON em.id = e.id
        WHERE e.segment_id = ?
        ORDER BY e.id
        """,
        (seg_id,)
    ).fetchall()

    events: dict[int, dict] = {}
    for row in rows:
        int_id   = row["int_id"]
        event_id = row["event_id"]
        if int_id not in events:
            events[int_id] = {"id": event_id, "metadata": {}, "document": ""}
        key = row["key"]
        if key is None:
            continue
        if row["string_value"] is not None:
            val = row["string_value"]
        elif row["int_value"] is not None:
            val = row["int_value"]
        elif row["float_value"] is not None:
            val = row["float_value"]
        elif row["bool_value"] is not None:
            val = bool(row["bool_value"])
        else:
            continue

        if key == "chroma:document":
            events[int_id]["document"] = val
        else:
            events[int_id]["metadata"][key] = val

    conn.close()
    return [ev for _, ev in sorted(events.items())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Actually swap the new DB in")
    ap.add_argument("--batch", type=int, default=100, help="Insert batch size")
    args = ap.parse_args()

    if not _SQLITE.exists():
        print(f"ERROR: {_SQLITE} not found")
        sys.exit(1)

    print(f"Reading events from {_SQLITE} ...")
    events = read_events_from_sqlite(_SQLITE)
    print(f"Found {len(events)} events in SQLite")
    if not events:
        print("Nothing to do.")
        return

    # Dry-run: just show summary
    labeled = sum(1 for e in events if e["metadata"].get("label"))
    print(f"  labeled: {labeled}, unlabeled: {len(events) - labeled}")
    label_counts: dict[str, int] = {}
    for e in events:
        lbl = e["metadata"].get("label", "") or "unlabeled"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {lbl}: {cnt}")

    if not args.apply:
        print(f"\nDRY-RUN: would rebuild ChromaDB at {_NEW_DIR}")
        print("Pass --apply to actually do it.\n")
        return

    # Build fresh DB
    if _NEW_DIR.exists():
        print(f"Removing existing {_NEW_DIR} ...")
        shutil.rmtree(_NEW_DIR)
    _NEW_DIR.mkdir(parents=True)

    print(f"Creating fresh ChromaDB at {_NEW_DIR} ...")
    client = chromadb.PersistentClient(path=str(_NEW_DIR))

    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    ef(["warmup"])

    col = client.get_or_create_collection("chalking_evals", embedding_function=ef)

    total = len(events)
    batch_size = args.batch
    inserted = 0
    t0 = time.time()

    for i in range(0, total, batch_size):
        batch = events[i : i + batch_size]
        ids       = [e["id"]       for e in batch]
        docs      = [e["document"] or "no description" for e in batch]
        metadatas = [e["metadata"] for e in batch]

        col.add(ids=ids, documents=docs, metadatas=metadatas)
        inserted += len(batch)
        elapsed = time.time() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        remaining = (total - inserted) / rate if rate > 0 else 0
        print(f"  [{inserted}/{total}] {rate:.1f} events/s, ~{remaining:.0f}s remaining")

    print(f"\nInserted {inserted} events into fresh DB.")
    print(f"Verifying count: {col.count()}")

    # Swap
    old_backup = _ROOT / "data/vectors_old"
    if old_backup.exists():
        shutil.rmtree(old_backup)
    print(f"\nSwapping: {_DB_DIR} -> {old_backup}")
    shutil.move(str(_DB_DIR), str(old_backup))
    print(f"Swapping: {_NEW_DIR} -> {_DB_DIR}")
    shutil.move(str(_NEW_DIR), str(_DB_DIR))
    print(f"\nDone! Old DB backed up at {old_backup}")
    print("You can delete it once you verify the container starts OK.")


if __name__ == "__main__":
    main()
