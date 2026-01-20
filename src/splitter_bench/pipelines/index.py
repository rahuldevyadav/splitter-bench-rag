from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import yaml

from splitter_bench.splitters.build import build_splitter, split_text
from splitter_bench.vectorstore.pgvector_store import get_store, upsert_texts
from dotenv import load_dotenv
load_dotenv()


def load_processed(corpus_id: str) -> List[Dict[str, Any]]:
    p = Path(f"data/processed/{corpus_id}.jsonl")
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows

def stable_key(splitter_name: str, params: Dict[str, Any]) -> str:
    s = splitter_name + "|" + json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def iter_splitter_configs(cfg_path: str = "src/splitter_bench/splitters/configs.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    for s in cfg["splitters"]:
        base_name = s["name"]
        s_type = s["type"]
        grid = s["params"]

        # simple cartesian product
        keys = list(grid.keys())
        def rec(i, cur):
            if i == len(keys):
                yield cur
                return
            k = keys[i]
            for v in grid[k]:
                cur2 = dict(cur)
                cur2[k] = v
                yield from rec(i+1, cur2)

        for params in rec(0, {}):
            key = stable_key(base_name, params)
            yield {
                "splitter_name": base_name,
                "splitter_type": s_type,
                "params": params,
                "splitter_key": key,
                "splitter_config": ",".join(f"{k}={params[k]}" for k in sorted(params.keys())),
            }

def main(all_splitters: bool = True, manifest_path: str = "data/corpora_manifest.yaml"):
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    corpora = [c["id"] for c in manifest["corpora"]]

    splitters = list(iter_splitter_configs()) if all_splitters else []
    if not splitters:
        raise RuntimeError("No splitters found. Check configs.yaml")

    for corpus_id in corpora:
        docs = load_processed(corpus_id)
        for s in splitters:
            table = f"vs_{corpus_id}__{s['splitter_key']}"
            store = get_store(table)

            splitter = build_splitter(s["splitter_type"], s["params"])

            all_chunks: List[str] = []
            all_meta: List[dict] = []

            for row in docs:
                text = row["text"]
                meta = row["metadata"]
                chunks = split_text(splitter, text)

                for j, ch in enumerate(chunks):
                    all_chunks.append(ch)
                    all_meta.append({
                        **meta,
                        "chunk_index": j,
                        "splitter_name": s["splitter_name"],
                        "splitter_key": s["splitter_key"],
                        "splitter_config": s["splitter_config"],
                    })

            upsert_texts(store, all_chunks, all_meta)
            print(f"[ok] indexed corpus={corpus_id} splitter={s['splitter_name']} ({s['splitter_config']}) -> {table}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-splitters", action="store_true", default=True)
    args = ap.parse_args()
    main(all_splitters=args.all_splitters)
