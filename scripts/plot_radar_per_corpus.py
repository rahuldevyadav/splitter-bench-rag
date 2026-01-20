import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def minmax(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.ones_like(x) * 0.5
    return (x - lo) / (hi - lo)

def invert01(x):
    return 1.0 - x

def radar(ax, labels, values, title):
    vals = np.concatenate([values, [values[0]]])
    angs = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angs = np.concatenate([angs, [angs[0]]])

    ax.plot(angs, vals)
    ax.fill(angs, vals, alpha=0.15)
    ax.set_title(title, pad=18)
    ax.set_xticks(angs[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])

def main():
    df = pd.read_csv("results/results.csv")
    outdir = Path("results/plots")
    outdir.mkdir(parents=True, exist_ok=True)

    # pick top 5 splitters overall by judge score to keep plots readable
    top = (
        df.groupby(["splitter_name","splitter_config"])
          .agg(avg_score=("trackB_judge_score","mean"))
          .reset_index()
          .sort_values("avg_score", ascending=False)
          .head(5)
    )
    top_set = set(map(tuple, top[["splitter_name","splitter_config"]].values.tolist()))

    for corpus_id, sub in df.groupby("corpus_id"):
        sub = sub[sub.apply(lambda r: (r["splitter_name"], r["splitter_config"]) in top_set, axis=1)]
        if sub.empty:
            continue

        g = (
            sub.groupby(["splitter_name","splitter_config"])
               .agg(
                   mrr=("trackA_mrr_at_k","mean"),
                   hit=("trackA_hit_at_k","mean"),
                   score=("trackB_judge_score","mean"),
                   pctge4=("trackB_judge_score", lambda x: float((x>=4).mean())),
                   p95lat=("retrieval_latency_ms", lambda x: float(np.nanpercentile(x, 95))),
                   cost=("cost_per_1k_queries_usd","mean"),
               )
               .reset_index()
        )

        # normalize within corpus
        mrr = minmax(g["mrr"].to_numpy())
        hit = minmax(g["hit"].to_numpy())
        scr = minmax(g["score"].to_numpy())
        pct = minmax(g["pctge4"].to_numpy())
        lat = invert01(minmax(g["p95lat"].to_numpy()))
        cst = invert01(minmax(g["cost"].to_numpy()))

        labels = ["MRR@K","Hit@K","Judge","%>=4","Latency(P95)","Cost/1k"]

        for i, row in g.iterrows():
            fig = plt.figure(figsize=(6,6))
            ax = plt.subplot(111, polar=True)
            vals = np.array([mrr[i], hit[i], scr[i], pct[i], lat[i], cst[i]])
            title = f"{corpus_id}\n{row['splitter_name']} | {row['splitter_config']}"
            radar(ax, labels, vals, title)
            fname = outdir / f"radar_{corpus_id}_{i}.png"
            fig.savefig(fname, dpi=180, bbox_inches="tight")
            plt.close(fig)

        print("[ok] radar charts:", corpus_id, "->", len(g))

if __name__ == "__main__":
    main()
