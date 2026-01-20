from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TokenPricePer1M:
    input_usd: float
    output_usd: float

# NOTE:
# Replace these numbers if you want to track exact current pricing.
# Keep pricing centralized so results are reproducible.
PRICES = {
    # -------------------------
    # Embeddings
    # -------------------------
    # Pricing per 1M input tokens
    "text-embedding-3-small": TokenPricePer1M(input_usd=0.02, output_usd=0.0),
    "text-embedding-3-large": TokenPricePer1M(input_usd=0.13, output_usd=0.0),

    # -------------------------
    # GPT-4.1 family (recommended for this benchmark)
    # -------------------------
    # Strong instruction following, stable judging, good RAG behavior
    "gpt-4.1-mini": TokenPricePer1M(input_usd=0.30, output_usd=1.20),
    "gpt-4.1": TokenPricePer1M(input_usd=3.00, output_usd=12.00),

    # -------------------------
    # GPT-5 family (kept for upper-bound experiments)
    # -------------------------
    "gpt-5-mini": TokenPricePer1M(input_usd=0.25, output_usd=2.00),
    "gpt-5.2-mini": TokenPricePer1M(input_usd=0.25, output_usd=2.00),

    "gpt-5": TokenPricePer1M(input_usd=1.25, output_usd=10.00),
    "gpt-5.2": TokenPricePer1M(input_usd=1.75, output_usd=14.00),
}

def cost_usd(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    if model not in PRICES:
        # Unknown model: do not crash runs; treat as 0 cost but keep visible.
        return 0.0
    p = PRICES[model]
    return (input_tokens / 1_000_000) * p.input_usd + (output_tokens / 1_000_000) * p.output_usd

def per_query_cost_usd(
    embed_model: str,
    llm_model: str,
    embed_query_tokens: int,
    answer_in: int, answer_out: int,
    judge_in: int, judge_out: int,
) -> float:
    return (
        cost_usd(embed_model, input_tokens=embed_query_tokens) +
        cost_usd(llm_model, input_tokens=answer_in, output_tokens=answer_out) +
        cost_usd(llm_model, input_tokens=judge_in, output_tokens=judge_out)
    )

def cost_per_1k_queries_usd(per_query_cost: float) -> float:
    return per_query_cost * 1000.0
