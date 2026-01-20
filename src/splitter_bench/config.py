from __future__ import annotations
import os

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v and v.strip() else default

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v and v.strip() else default

TOP_K = env_int("TOP_K", 5)

OPENAI_EMBED_MODEL = env_str("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_QGEN_MODEL = env_str("OPENAI_QGEN_MODEL", "gpt-4.1-mini")
OPENAI_ANSWER_MODEL = env_str("OPENAI_ANSWER_MODEL", "gpt-4.1-mini")
OPENAI_JUDGE_MODEL = env_str("OPENAI_JUDGE_MODEL", "gpt-4.1-mini")
