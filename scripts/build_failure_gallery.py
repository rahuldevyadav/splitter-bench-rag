import json
import pandas as pd
import re
from pathlib import Path

def has_unclosed_code_fence(s: str) -> bool:
    return s.count("```") % 2 == 1

def tabley(s: str) -> bool:
    lines = s.splitlines()
    pipe_lines = sum(1 for l in lines if l.count("|") >= 3)
    return pipe_lines >= 3

def bad_boundary(s: str) -> list[str]:
    reasons = []
    if s.rstrip().endswith("-"):
        reasons.append("ends_with_hyphen")
    if has_unclosed_code_fence(s):
        reasons.append("unclosed_code_fence")
    if tabley(s):
        reasons.append("markdown_table_split")
    if re.search(r"\bArticle\s+\d+\b", s) and len(s) < 700:
        reasons.append("article_header_fragment")
    return reasons

def main():
    df = pd.read_csv("results/results.csv")
    out = Path("results/FAILURES.md")

    parts = ["# Failure Gallery\n\n"]
    parts.append("Examples where retrieval succeeded (Hit@K=1) but the judged answer was poor (<=2).\n\n")

    grp = df.groupby(["corpus_id","splitter_name","splitter_config"], dropna=False)
    for (corpus_id, splitter_name, splitter_config), sub in grp:
        sub = sub[(sub["trackA_hit_at_k"] == 1) & (sub["trackB_judge_score"] <= 2)]
        if sub.empty:
            continue

        def priority(r):
            try:
                ctxs = json.loads(r["trackB_contexts"])
                if not ctxs:
                    return 0
                return 10 * len(bad_boundary(ctxs[0]))
            except Exception:
                return 0

        sub = sub.copy()
        sub["priority"] = sub.apply(priority, axis=1)
        sub = sub.sort_values(["priority","trackB_judge_score"], ascending=[False, True]).head(5)

        parts.append(f"## {corpus_id} — {splitter_name} ({splitter_config})\n\n")

        for _, r in sub.iterrows():
            qid = r["query_id"]
            parts.append(f"### Query `{qid}` (Judge={r['trackB_judge_score']})\n\n")

            parts.append("**Question**\n\n")
            parts.append(f"> {r['question']}\n\n")

            parts.append("**Answer (model)**\n\n")
            parts.append(f"{r['trackB_answer']}\n\n")

            parts.append("**Judge rationale**\n\n")
            parts.append(f"> {r['trackB_judge_rationale']}\n\n")

            parts.append("**Top retrieved chunk (snippet)**\n\n")
            try:
                ctxs = json.loads(r["trackB_contexts"])
                snippet = (ctxs[0] if ctxs else "")[:1200]
                parts.append("```text\n" + snippet + "\n```\n\n")
                issues = bad_boundary(ctxs[0]) if ctxs else []
                if issues:
                    parts.append("**Boundary issues detected:** " + ", ".join(issues) + "\n\n")
            except Exception:
                parts.append("_Missing contexts_\n\n")

    out.write_text("".join(parts), encoding="utf-8")
    print("[ok] wrote results/FAILURES.md")

if __name__ == "__main__":
    main()
