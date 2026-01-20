from __future__ import annotations

import json
from pathlib import Path
from typing import List

import yaml
from langchain_core.documents import Document


from splitter_bench.corpora.loaders import load_corpus

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def save_docs(corpus_id: str, docs: List[Document]) -> None:
    path = OUT / f"{corpus_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {"text": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main(manifest_path: str = "data/corpora_manifest.yaml") -> None:
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    for c in manifest["corpora"]:
        docs = load_corpus(c)
        if not docs:
            print(f"[skip] {c['id']}: no docs returned (skipped or empty)")
            continue
        save_docs(c["id"], docs)
        print(f"[ok] {c['id']}: {len(docs)} docs -> data/processed/{c['id']}.jsonl")

if __name__ == "__main__":
    main()
