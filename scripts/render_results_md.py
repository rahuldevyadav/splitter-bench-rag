import pandas as pd
import numpy as np
from pathlib import Path

def to_md_table(d: pd.DataFrame) -> str:
    return d.to_markdown(index=False)

def main():
    df = pd.read_csv("results/results.csv")

    # Standardize names for reporting
    df["hit"] = df["trackA_hit_at_k"]
    df["mrr"] = df["trackA_mrr_at_k"]
    df["score"] = df["trackB_judge_score"]
    df["lat"] = df["retrieval_latency_ms"]
    df["cost1k"] = df["cost_per_1k_queries_usd"]

    # Overall leaderboard
    overall = (
        df.groupby(["splitter_name","splitter_config"])
          .agg(
              avg_score=("score","mean"),
              pct_ge4=("score", lambda x: float((x>=4).mean())),
              mrr_at_k=("mrr","mean"),
              hit_at_k=("hit","mean"),
              p95_latency_ms=("lat", lambda x: float(np.nanpercentile(x, 95))),
              cost_per_1k_usd=("cost1k","mean"),
          )
          .reset_index()
          .sort_values(["avg_score","mrr_at_k"], ascending=False)
          .head(15)
    )

    parts = []
    parts.append("# Results\n\n")
    parts.append("Benchmarked corpora:\n")
    parts.append("- GDPR (EUR-Lex PDF)\n")
    parts.append("- LangChain docs (sitemap crawl)\n")
    parts.append("- LangChain GitHub markdown\n\n")

    parts.append("## Overall leaderboard (Top 15)\n\n")
    parts.append(to_md_table(overall))
    parts.append("\n\n")

    for corpus_id, sub in df.groupby("corpus_id"):
        summary = (
            sub.groupby(["splitter_name","splitter_config"])
               .agg(
                   avg_score=("score","mean"),
                   pct_ge4=("score", lambda x: float((x>=4).mean())),
                   mrr_at_k=("mrr","mean"),
                   hit_at_k=("hit","mean"),
                   p95_latency_ms=("lat", lambda x: float(np.nanpercentile(x, 95))),
                   cost_per_1k_usd=("cost1k","mean"),
               )
               .reset_index()
               .sort_values(["avg_score","mrr_at_k"], ascending=False)
               .head(10)
        )
        parts.append(f"## Corpus: `{corpus_id}`\n\n")
        parts.append(to_md_table(summary))
        parts.append("\n\n")

        # link radar charts by convention
        parts.append("### Radar charts\n\n")
        parts.append(f"See `results/plots/` for radar charts for `{corpus_id}`.\n\n")

    Path("results/RESULTS.md").write_text("".join(parts), encoding="utf-8")
    print("[ok] wrote results/RESULTS.md")

if __name__ == "__main__":
    main()
