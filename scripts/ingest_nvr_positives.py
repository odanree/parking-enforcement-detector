"""Ingest ground-truth chalking frames into the vector store as labeled positives.

Seeds RAG with real `chalker` examples (the dataset had ~1 positive in 12k rows).

Two phases, because of two hard environment constraints:
  • EXTRACT must run on the Windows host — only the MSMF backend decodes these
    Dahua/Amcrest NVR mp4s; the Linux container's ffmpeg cannot.
  • INGEST (the ChromaDB write) must run inside the container — host-side
    chromadb segfaults on the live store (native-lib divergence), while the
    container's libs read/write it cleanly. Run with the main detector
    container STOPPED so there is a single writer.

Phase 1 (host):
    python -m scripts.ingest_nvr_positives extract \
        --video /path/to/nvr_clip.mp4 \
        --seconds 119 122 136 406 423 429 445 --out _ingest_payload

Phase 2 (container one-off, main container stopped):
    docker compose run --rm \
        -v "$PWD/_ingest_payload:/app/_ingest_payload" \
        -v "$PWD/scripts:/app/scripts" \
        detector python -m scripts.ingest_nvr_positives ingest \
        --dir _ingest_payload --camera-id 1 --apply
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except Exception:
    pass

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_positives")

_VLM_W, _VLM_H = 1280, 720
_CANONICAL = ("Parking enforcement officer in dark uniform walking the parked-car line "
              "holding a long light-coloured chalking wand angled down to the ground near a tyre.")


def _jpeg(img: np.ndarray, q: int = 90) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


def _b64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode()


# ── Phase 1: extract (host / MSMF) ───────────────────────────────────────────

def cmd_extract(args) -> None:
    from ultralytics import YOLO
    from src.detection.wand_detector import WandDetector

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8m.pt")
    wd = WandDetector(require_dark_torso=False)

    cap = cv2.VideoCapture(args.video, cv2.CAP_MSMF)
    if not cap.isOpened():
        logger.error("Cannot open %s (need Windows MSMF)", args.video); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    saved = 0
    for sec in args.seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ok, fr = cap.read()
        if not ok:
            logger.warning("no frame at s=%.0f", sec); continue
        res = model.predict(fr, imgsz=1920, conf=0.30, classes=[0], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        # pick the officer: person the wand detector fires on (highest conf), else darkest torso
        best = None
        H, W = fr.shape[:2]
        for b in boxes:
            bw, bh = b[2]-b[0], b[3]-b[1]
            if bw < 14 or bh < 24:
                continue
            padx, padt, padb = int(bw*1.3), int(bh*0.15), int(bh*0.6)
            cx1, cy1 = max(0, b[0]-padx), max(0, b[1]-padt)
            cx2, cy2 = min(W, b[2]+padx), min(H, b[3]+padb)
            crop = fr[cy1:cy2, cx1:cx2]
            sig = wd.detect(crop, (b[0]-cx1, b[1]-cy1, b[2]-cx1, b[3]-cy1))
            torso = cv2.cvtColor(fr[b[1]:b[1]+int(bh*0.6), b[0]:b[2]], cv2.COLOR_BGR2HSV)[..., 2]
            dark = 255 - (float(torso.mean()) if torso.size else 255)
            score = (sig.confidence * 100 if sig.present else 0) + dark * 0.1
            if best is None or score > best[0]:
                best = (score, b.tolist(), sig.present)
        if best is None:
            logger.warning("no person at s=%.0f", sec); continue
        _, bbox, wand_present = best
        cv2.imwrite(str(out / f"s{int(sec):04d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / f"s{int(sec):04d}.json").write_text(json.dumps({"sec": sec, "bbox": bbox, "wand": wand_present}))
        logger.info("extracted s=%.0f bbox=%s wand=%s", sec, bbox, wand_present)
        saved += 1
    cap.release()
    logger.info("Extracted %d officer frames to %s", saved, out)


# ── Phase 2: ingest (container) ──────────────────────────────────────────────

def cmd_ingest(args) -> None:
    from src.storage.vector_store import EventVectorStore

    d = Path(args.dir)
    metas = sorted(d.glob("s*.json"))
    if not metas:
        logger.error("no payload in %s", d); return
    vs = EventVectorStore(db_path=args.db, dataset_path=args.dataset) if args.apply else None
    added = 0
    for mj in metas:
        meta = json.loads(mj.read_text())
        img_path = mj.with_suffix(".jpg")
        fr = cv2.imread(str(img_path))
        if fr is None:
            logger.warning("missing %s", img_path); continue
        b = meta["bbox"]; H, W = fr.shape[:2]
        cx, cy = (b[0]+b[2])//2, (b[1]+b[3])//2
        x1 = max(0, min(cx-_VLM_W//2, W-_VLM_W)) if W >= _VLM_W else 0
        y1 = max(0, min(cy-_VLM_H//2, H-_VLM_H)) if H >= _VLM_H else 0
        crop = fr[y1:y1+_VLM_H, x1:x1+_VLM_W] if (W >= _VLM_W and H >= _VLM_H) else fr
        vis = fr.copy()
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        thumb = _b64(_jpeg(cv2.resize(vis, (640, int(640*H/W)))))
        desc = f"{_CANONICAL} [s={int(meta['sec'])}]"
        logger.info("ingest s=%s bbox=%s", meta["sec"], b)
        if not args.apply:
            continue
        eid = vs.add(
            description=desc, detected=True, confidence=0.9,
            camera_id=args.camera_id, thumbnail_b64=thumb,
            frames_b64=[_b64(_jpeg(crop))], person_type=args.person_type,
            capture_source=args.source, hires_b64=_b64(_jpeg(fr, 90)),
            model_primary="manual",
        )
        if eid:
            vs.update_label(eid, args.label)
            added += 1
    logger.info("Ingested %d labeled positives (label=%s person_type=%s source=%s)",
                added, args.label, args.person_type, args.source)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="HOST: extract officer frames to a payload dir")
    e.add_argument("--video", required=True)
    e.add_argument("--seconds", type=float, nargs="+", required=True)
    e.add_argument("--out", default="_ingest_payload")
    e.set_defaults(func=cmd_extract)

    i = sub.add_parser("ingest", help="CONTAINER: write payload to the vector store")
    i.add_argument("--dir", default="_ingest_payload")
    i.add_argument("--db", default=os.getenv("VECTOR_DB_PATH", "data/vectors"))
    i.add_argument("--dataset", default=os.getenv("DATASET_PATH", "dataset"))
    i.add_argument("--camera-id", type=int, default=1)
    i.add_argument("--label", default="true_positive")
    i.add_argument("--person-type", default="chalker")
    i.add_argument("--source", default="external")
    i.add_argument("--apply", action="store_true")
    i.set_defaults(func=cmd_ingest)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
