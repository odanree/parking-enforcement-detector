"""Offline VLM footprint/accuracy comparison for the person-type classifier.

Compares candidate ollama models against the current qwen2.5vl:7b on the
classify task, scoring predicted person_type vs the human label over a
stratified sample of definitely-labeled events. Measures each model's loaded
VRAM. Per ADR-024: validate model swaps offline before deploying.

Usage:  python scripts/vlm_footprint_eval.py [--n 200] [--models a,b,c]
Run from the detector host (talks to the running container on :8000 / ollama).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict

BASE = "http://localhost:8000"
OLLAMA_CONTAINER = "parking-enforcement-detector-ollama-1"
DEFINITE = {"pedestrian", "occupant", "worker_delivery", "worker_landscape", "resident", "chalker"}


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.load(r)


def _post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def build_sample(n: int, seed: int = 42) -> list[dict]:
    items = _get("/api/dataset?limit=3000")["items"]
    pool = [e for e in items if (e.get("person_type") or "") in DEFINITE and e.get("frame_files")]
    by_type: dict[str, list] = defaultdict(list)
    for e in pool:
        by_type[e["person_type"]].append(e)
    rng = random.Random(seed)
    # Take all of the rare classes; sample the rest proportionally to fill n.
    sample, rest_types = [], []
    for t, evs in by_type.items():
        if len(evs) <= 20:
            sample.extend(evs)
        else:
            rest_types.append(t)
    remaining = max(0, n - len(sample))
    rest_total = sum(len(by_type[t]) for t in rest_types)
    for t in rest_types:
        k = round(remaining * len(by_type[t]) / rest_total)
        sample.extend(rng.sample(by_type[t], min(k, len(by_type[t]))))
    return sample


def vram_for(model: str) -> str:
    # Warm the model, then read its resident size from `ollama ps`.
    subprocess.run(["docker", "exec", OLLAMA_CONTAINER, "ollama", "run", model, "hi"],
                   capture_output=True, timeout=120)
    out = subprocess.run(["docker", "exec", OLLAMA_CONTAINER, "ollama", "ps"],
                         capture_output=True, text=True, timeout=30).stdout
    for line in out.splitlines():
        if line.startswith(model.split(":")[0]):
            # columns: NAME ID SIZE PROCESSOR ... — SIZE is cols 2-3 like "6.0 GB"
            m = re.search(r"(\d+\.?\d*)\s*GB", line)
            if m:
                return f"{m.group(1)} GB"
    return "?"


def run_eval(model: str, ids: list[str]) -> None:
    _post("/api/dataset/model-eval", {"model_name": model, "backend": "ollama",
                                      "task": "classify", "ids": ids})
    while True:
        p = _get("/api/dataset/model-eval/progress")
        print(f"  {model}: {p['done']}/{p['total']} (errors {p['errors']})", end="\r")
        if not p["running"]:
            break
        time.sleep(5)
    print()


PRED_RE = re.compile(r"\[([a-z_]+)\]")


def score(models: list[str], sample_ids: set[str]) -> dict:
    items = {e["id"]: e for e in _get("/api/dataset?limit=3000")["items"] if e["id"] in sample_ids}
    results: dict[str, dict] = {}
    for model in models:
        key = f"{model}:classify"
        correct = total = unknown = 0
        confusion: dict = defaultdict(Counter)
        for e in items.values():
            label = e["person_type"]
            evals = e.get("model_evals") or []
            if isinstance(evals, str):
                try:
                    evals = json.loads(evals) if evals else []
                except Exception:
                    evals = []
            ev = next((x for x in evals if x.get("model_name") == key), None)
            if not ev:
                continue
            m = PRED_RE.search(ev.get("description", ""))
            pred = m.group(1) if m else "unknown"
            total += 1
            if pred in ("unknown", ""):
                unknown += 1
            confusion[label][pred] += 1
            if pred == label:
                correct += 1
        results[model] = {
            "n": total,
            "accuracy": round(correct / total, 3) if total else None,
            "unknown_rate": round(unknown / total, 3) if total else None,
            "confusion": {k: dict(v) for k, v in confusion.items()},
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", default="qwen2.5vl:7b,gemma3:4b,moondream")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    sample = build_sample(args.n)
    ids = [e["id"] for e in sample]
    print(f"Sample: {len(ids)} events — {dict(Counter(e['person_type'] for e in sample))}\n")

    vram = {}
    for model in models:
        print(f"VRAM probe: {model}")
        vram[model] = vram_for(model)
        print(f"  loaded size: {vram[model]}")
        print(f"Eval (classify): {model}")
        run_eval(model, ids)

    res = score(models, set(ids))
    print("\n==== RESULTS (person-type classification) ====")
    print(f"{'model':<16}{'size':>8}{'n':>6}{'acc':>8}{'unknown':>9}")
    for model in models:
        r = res[model]
        print(f"{model:<16}{vram.get(model,'?'):>8}{r['n']:>6}"
              f"{(r['accuracy'] or 0):>8.1%}{(r['unknown_rate'] or 0):>9.1%}")
    print("\nPer-label confusion (label -> predictions):")
    for model in models:
        print(f"\n{model}:")
        for label, preds in sorted(res[model]["confusion"].items()):
            print(f"  {label:<18} {dict(preds)}")


if __name__ == "__main__":
    main()
