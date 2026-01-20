from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import yaml
import tiktoken
from openai import OpenAI

from splitter_bench.config import TOP_K, OPENAI_EMBED_MODEL, OPENAI_ANSWER_MODEL, OPENAI_JUDGE_MODEL
from splitter_bench.costs import per_query_cost_usd, cost_per_1k_queries_usd
from splitter_bench.vectorstore.pgvector_store import get_store, similarity_search
from splitter_bench.pipelines.index import iter_splitter_configs

from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enc = tiktoken.get_encoding("cl100k_base")

JUDGE_RUBRIC = """You are grading a RAG answer.
Score 0-5:
5 = fully correct and supported by retrieved context
3 = partially correct or weakly supported
1 = mostly incorrect or hallucinated
0 = irrelevant/unsafe

Return strict JSON:
{"score": int, "rationale": str}
"""

def token_count(s: str) -> int:
    return len(enc.encode(s or ""))

def read_evalset(corpus_id: str) -> List[Dict[str, Any]]:
    p = Path(f"data/evalsets/{corpus_id}.jsonl")
    items = []
    for line in p.read_text(encoding="utf-8").splitlines():
        items.append(json.loads(line))
    return items

def gold_match(meta: dict, match: dict) -> bool:
    # match is a dict of key->value that must exist in meta
    for k, v in match.items():
        if meta.get(k) != v:
            return False
    return True

def compute_rank_and_hit(retrieved: List[Tuple[Any, float]], gold_match_dict: dict, k: int) -> Tuple[int | None, int]:
    # retrieved is list of (Document, score)
    for i, (doc, _score) in enumerate(retrieved[:k], start=1):
        if gold_match(doc.metadata, gold_match_dict):
            return i, 1
    return None, 0

def mrr_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / float(rank)

def answer_with_context(question: str, contexts: List[str]) -> Tuple[str, int, int]:
    prompt = "Use the CONTEXT to answer the QUESTION. If context is insufficient, say you don't know.\n\n" \
             "CONTEXT:\n" + "\n---\n".join(contexts) + "\n\nQUESTION:\n" + question
    resp = client.responses.create(model=OPENAI_ANSWER_MODEL, input=prompt)
    out_text = resp.output_text
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) if usage else 0
    out_tok = getattr(usage, "output_tokens", 0) if usage else 0
    return out_text, int(in_tok), int(out_tok)

def judge_answer(question: str, answer: str, contexts: List[str]) -> Tuple[int, str, int, int]:
    prompt = JUDGE_RUBRIC + "\n\nQUESTION:\n" + question + "\n\nANSWER:\n" + answer + "\n\nCONTEXTS:\n" + "\n---\n".join(contexts)
    resp = client.responses.create(model=OPENAI_JUDGE_MODEL, input=prompt)
    txt = resp.output_text
    obj = json.loads(txt[txt.find("{"):txt.rfind("}")+1])
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) if usage else 0
    out_tok = getattr(usage, "output_tokens", 0) if usage else 0
    return int(obj["score"]), str(obj.get("rationale","")), int(in_tok), int(out_tok)

def main(all_splitters: bool = True, out_csv: str = "results/results.csv", manifest_path: str = "data/corpora_manifest.yaml"):
    Path("results").mkdir(parents=True, exist_ok=True)

    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    corpora = [c["id"] for c in manifest["corpora"]]

    splitters = list(iter_splitter_configs()) if all_splitters else []
    if not splitters:
        raise RuntimeError("No splitters found. Check configs.yaml")

    # We'll write row-per-query to CSV
    fieldnames = [
        "corpus_id","query_id","question","difficulty","type",
        "splitter_name","splitter_key","splitter_config",
        "trackA_hit_at_k","trackA_rank_of_gold","trackA_mrr_at_k","retrieval_latency_ms",
        "trackB_answer","trackB_judge_score","trackB_judge_rationale",
        "trackB_contexts","trackB_context_metadatas",
        "embed_query_tokens",
        "answer_input_tokens","answer_output_tokens",
        "judge_input_tokens","judge_output_tokens",
        "per_query_cost_usd","cost_per_1k_queries_usd",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for corpus_id in corpora:
            evalset = read_evalset(corpus_id)

            for s in splitters:
                table = f"vs_{corpus_id}__{s['splitter_key']}"
                store = get_store(table)

                for item in evalset:
                    qid = item["id"]
                    question = item["question"]
                    gold = item.get("gold", {})
                    meta_match = gold.get("metadata_match", {})
                    k = TOP_K

                    # Track A: retrieval
                    t0 = time.perf_counter()
                    retrieved = similarity_search(store, question, k=k)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    rank, hit = compute_rank_and_hit(retrieved, meta_match, k)
                    mrr = mrr_at_k(rank, k)

                    # Prepare contexts for Track B
                    contexts = [doc.page_content for (doc, _score) in retrieved]
                    ctx_meta = [doc.metadata for (doc, _score) in retrieved]

                    # Query embedding tokens (estimate)
                    embed_q_toks = token_count(question)

                    # Track B: answer + judge
                    answer, ans_in, ans_out = answer_with_context(question, contexts)
                    score, rationale, judge_in, judge_out = judge_answer(question, answer, contexts)

                    pq_cost = per_query_cost_usd(
                        embed_model=OPENAI_EMBED_MODEL,
                        llm_model=OPENAI_ANSWER_MODEL,
                        embed_query_tokens=embed_q_toks,
                        answer_in=ans_in, answer_out=ans_out,
                        judge_in=judge_in, judge_out=judge_out,
                    )
                    c1k = cost_per_1k_queries_usd(pq_cost)

                    w.writerow({
                        "corpus_id": corpus_id,
                        "query_id": qid,
                        "question": question,
                        "difficulty": item.get("difficulty",""),
                        "type": item.get("type",""),
                        "splitter_name": s["splitter_name"],
                        "splitter_key": s["splitter_key"],
                        "splitter_config": s["splitter_config"],

                        "trackA_hit_at_k": hit,
                        "trackA_rank_of_gold": rank if rank is not None else "",
                        "trackA_mrr_at_k": mrr,
                        "retrieval_latency_ms": latency_ms,

                        "trackB_answer": answer,
                        "trackB_judge_score": score,
                        "trackB_judge_rationale": rationale,
                        "trackB_contexts": json.dumps(contexts, ensure_ascii=False),
                        "trackB_context_metadatas": json.dumps(ctx_meta, ensure_ascii=False),

                        "embed_query_tokens": embed_q_toks,
                        "answer_input_tokens": ans_in,
                        "answer_output_tokens": ans_out,
                        "judge_input_tokens": judge_in,
                        "judge_output_tokens": judge_out,

                        "per_query_cost_usd": pq_cost,
                        "cost_per_1k_queries_usd": c1k,
                    })

                print(f"[ok] benchmarked corpus={corpus_id} splitter={s['splitter_name']} ({s['splitter_config']})")

    print(f"[ok] wrote {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-splitters", action="store_true", default=True)
    ap.add_argument("--out", default="results/results.csv")
    args = ap.parse_args()
    main(all_splitters=args.all_splitters, out_csv=args.out)
