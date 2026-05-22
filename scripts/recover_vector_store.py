"""One-off recovery: rebuild a chalking_evals collection from raw SQLite.

The in-place merge was interrupted mid-bulk-add, leaving the host collection's
vector segment inconsistent (chromadb segfaults on count/get). The underlying
records — ids, documents, and metadata — are still intact in the SQLite
metadata tables. Description embeddings are deterministic (all-MiniLM-L6-v2),
so we can rebuild the collection by re-adding records and letting the embedding
function recompute vectors via the normal, stable add() path.

Usage:
    python -m scripts.recover_vector_store \
        --sources data/vectors _container_db_tmp \
        --out data/vectors_rebuilt

Then (after verifying counts) swap data/vectors_rebuilt into place.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chromadb  # noqa: E402

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recover")

_DOC_KEY = "chroma:document"


def _extract(db_path: str) -> dict[str, dict]:
    """Return {event_id: {"document": str, "metadata": dict}} from raw SQLite.

    Avoids the chromadb runtime entirely so a damaged vector segment cannot
    crash the read.
    """
    sql = Path(db_path) / "chroma.sqlite3"
    con = sqlite3.connect(str(sql))
    cur = con.cursor()

    # int rowid -> string event id
    id_map: dict[int, str] = {}
    for rid, eid in cur.execute("SELECT id, embedding_id FROM embeddings"):
        id_map[rid] = eid

    records: dict[str, dict] = {}
    for rid, key, sval, ival, fval, bval in cur.execute(
        "SELECT id, key, string_value, int_value, float_value, bool_value FROM embedding_metadata"
    ):
        eid = id_map.get(rid)
        if eid is None:
            continue
        rec = records.setdefault(eid, {"document": "", "metadata": {}})
        if key == _DOC_KEY:
            rec["document"] = sval or ""
            continue
        # Reconstruct typed value (exactly one column is non-null).
        if sval is not None:
            val = sval
        elif ival is not None:
            val = int(ival)
        elif fval is not None:
            val = float(fval)
        elif bval is not None:
            val = bool(bval)
        else:
            continue
        rec["metadata"][key] = val
    con.close()
    logger.info("Extracted %d records from %s", len(records), db_path)
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True, help="One or more chromadb dirs to union")
    p.add_argument("--out", required=True, help="Output chromadb dir (must not exist or be empty)")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--collection", default="chalking_evals")
    args = p.parse_args(argv)

    # Union all sources; later sources do NOT overwrite earlier ones (first wins,
    # which preserves any labels already present on the host copy).
    merged: dict[str, dict] = {}
    for src in args.sources:
        recs = _extract(src)
        for eid, rec in recs.items():
            if eid not in merged:
                merged[eid] = rec
            else:
                # Keep whichever has a non-empty label.
                if not (merged[eid]["metadata"].get("label") or "").strip() and \
                        (rec["metadata"].get("label") or "").strip():
                    merged[eid] = rec
    logger.info("Total unique records to rebuild: %d", len(merged))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(out))
    # Default embedding function recomputes vectors from the document text.
    col = client.get_or_create_collection(args.collection)

    items = list(merged.items())
    added = 0
    for i in range(0, len(items), args.batch):
        chunk = items[i:i + args.batch]
        col.add(
            ids=[eid for eid, _ in chunk],
            documents=[r["document"] or "no description" for _, r in chunk],
            metadatas=[r["metadata"] for _, r in chunk],
        )
        added += len(chunk)
        if added % 1000 == 0:
            logger.info("  added %d/%d", added, len(items))
            time.sleep(0.1)
    logger.info("Rebuild complete: %d records, collection.count()=%d", added, col.count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
