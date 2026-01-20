import json
import os
import random
import pathlib
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv(".env", override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Ensure .env is in repo root and contains OPENAI_API_KEY=...")

QGEN_MODEL = os.getenv("OPENAI_QGEN_MODEL", "gpt-4.1-mini")
print("[env] OPENAI_QGEN_MODEL =", QGEN_MODEL)

client = OpenAI(api_key=API_KEY)

# ----------------------------
# 1) Define schema (Pydantic)
# ----------------------------

class Gold(BaseModel):
    quote: str = Field(..., description="A short verbatim quote from the passage (<= 40 words). Can be empty if unavailable.")
    metadata_match: Dict[str, Any] = Field(
        default_factory=dict,
        description="Subset of metadata keys to match for gold retrieval scoring (e.g., source, page, path)."
    )

class EvalItem(BaseModel):
    question: str = Field(..., description="Natural, realistic question answerable from the passage.")
    answer: str = Field(..., description="Concise answer (1-4 sentences) using only the passage.")
    gold: Gold
    difficulty: str = Field(..., description="One of: easy, medium, hard")
    type: str = Field(..., description="One of: definition, procedure, compare, exception, constraint")

parser = PydanticOutputParser(pydantic_object=EvalItem)

# ----------------------------
# 2) Prompt template
# ----------------------------

BASE_INSTRUCTIONS = f"""
You create evaluation questions for a RAG benchmark.

Rules:
- Use ONLY the SOURCE_PASSAGE.
- The answer must be supported by the passage.
- Prefer realistic queries: definition, procedure, compare, exceptions, constraints.
- Keep the answer concise (1-4 sentences).
- Include a short verbatim quote from the passage if possible (<= 40 words).
- metadata_match should include stable keys (e.g., corpus_id, source, page, local_path, path).

{parser.get_format_instructions()}
""".strip()

def load_processed(corpus_id: str) -> List[Dict[str, Any]]:
    path = pathlib.Path(f"data/processed/{corpus_id}.jsonl")
    if not path.exists():
        raise RuntimeError(f"Missing processed file: {path}. Run ingest first.")
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        txt = (r.get("text") or "").strip()
        if len(txt) >= 400:
            docs.append(r)
    return docs

def call_model(prompt: str) -> str:
    resp = client.responses.create(
        model=QGEN_MODEL,
        input=prompt,
    )
    return resp.output_text

def parse_output(text: str) -> EvalItem:
    try:
        return parser.parse(text)
    except OutputParserException:
        # Common: model wraps JSON in ```json fences or adds commentary.
        # Try to extract likely JSON block and parse again.
        a = text.find("{")
        b = text.rfind("}")
        if a != -1 and b != -1 and b > a:
            return parser.parse(text[a:b+1])
        raise

def make_one(corpus_id: str, doc: Dict[str, Any], idx: int) -> Dict[str, Any]:
    passage = (doc.get("text") or "")[:4500]
    meta = doc.get("metadata") or {}

    prompt = (
        BASE_INSTRUCTIONS
        + "\n\nMETADATA:\n"
        + json.dumps(meta, ensure_ascii=False)
        + "\n\nSOURCE_PASSAGE:\n"
        + passage
    )

    last_err: Optional[Exception] = None
    last_out: str = ""

    for attempt in range(1, 4):
        try:
            out = call_model(prompt)
            last_out = out
            item = parse_output(out)

            obj = item.model_dump()
            obj["id"] = f"{corpus_id}_{idx:05d}"
            obj["corpus_id"] = corpus_id

            # If quote isn’t reliably verbatim, allow it to be empty (don’t drop the item)
            # You can tighten later once pipeline works end-to-end.
            if not isinstance(obj.get("gold", {}).get("quote", ""), str):
                obj["gold"]["quote"] = ""

            return obj
        except Exception as e:
            last_err = e
            time.sleep(0.6 * attempt)

    raise RuntimeError(f"QGEN failed after retries: {last_err}\nLast output (truncated): {last_out[:400]}")

def main():
    random.seed(7)

    manifest_path = pathlib.Path("data/evalsets/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets: Dict[str, int] = manifest.get("targets", {})

    out_dir = pathlib.Path("data/evalsets")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for corpus_id, n in targets.items():
        docs = load_processed(corpus_id)
        pool = random.sample(docs, k=min(len(docs), n * 8))

        items: List[Dict[str, Any]] = []
        errors: List[str] = []

        i = 0
        for doc in pool:
            if i >= n:
                break
            try:
                items.append(make_one(corpus_id, doc, i))
                i += 1
                print(f"[ok] {corpus_id}: {i}/{n}")
            except Exception as e:
                errors.append(str(e))

        (out_dir / f"{corpus_id}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in items),
            encoding="utf-8",
        )
        (log_dir / f"{corpus_id}_errors.txt").write_text(
            "\n---\n".join(errors) if errors else "no errors",
            encoding="utf-8",
        )

        print(f"[done] {corpus_id} -> {len(items)} items")
        print(f"[log]  data/evalsets/_logs/{corpus_id}_errors.txt")

if __name__ == "__main__":
    main()
