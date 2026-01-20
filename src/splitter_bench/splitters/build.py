from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
except Exception:
    MarkdownHeaderTextSplitter = None

@dataclass
class BuiltSplitter:
    name: str
    splitter: Any
    type: str
    params: Dict[str, Any]
    key: str  # stable key used for table names etc.

def build_splitter(splitter_type: str, params: Dict[str, Any]) -> Any:
    if splitter_type == "character":
        return CharacterTextSplitter(**params)

    if splitter_type == "recursive_character":
        return RecursiveCharacterTextSplitter(**params)

    if splitter_type == "token":
        return TokenTextSplitter(**params)

    if splitter_type == "markdown_header_pipeline":
        if MarkdownHeaderTextSplitter is None:
            raise RuntimeError("MarkdownHeaderTextSplitter not available in your installed version.")
        headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        inner = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params.get("chunk_overlap", 0),
        )
        return ("pipeline", header_splitter, inner)

    raise ValueError(f"Unknown splitter type: {splitter_type}")

def split_text(built: Any, text: str) -> List[str]:
    if isinstance(built, tuple) and built and built[0] == "pipeline":
        _, first, inner = built
        docs = first.split_text(text) if hasattr(first, "split_text") else first.split_documents(text)
        parts: List[str] = []
        for d in docs:
            content = getattr(d, "page_content", d)
            parts.extend(inner.split_text(content))
        return parts
    return built.split_text(text)
