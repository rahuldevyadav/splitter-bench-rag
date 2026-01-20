from __future__ import annotations

import os
import re
import pathlib
import requests
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.document_loaders import SitemapLoader, GithubFileLoader

def _ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def load_pdf_from_url(corpus_id: str, url: str, tags: List[str]) -> List[Document]:
    """
    Downloads a PDF and extracts text via PyMuPDF.
    Robust: validates PDF bytes, retries, and supports manual local fallback.
    """
    _ensure_dir("data/raw")
    local_path = f"data/raw/{corpus_id}.pdf"

    def looks_like_pdf(path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) < 1024:
                return False
            with open(path, "rb") as f:
                return f.read(5) == b"%PDF-"
        except Exception:
            return False

    headers = {
        "User-Agent": os.getenv("USER_AGENT", "splitter-bench-rag/0.1"),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    # Try download if file missing / invalid
    if not looks_like_pdf(local_path):
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass

        last_err = None
        for attempt in range(1, 4):
            try:
                r = requests.get(url, headers=headers, timeout=180, allow_redirects=True)
                r.raise_for_status()
                content = r.content or b""

                with open(local_path, "wb") as f:
                    f.write(content)

                if looks_like_pdf(local_path):
                    break

                # If it's HTML or empty, wipe and retry
                try:
                    os.remove(local_path)
                except Exception:
                    pass
                last_err = RuntimeError(
                    f"Downloaded content is not a valid PDF (attempt {attempt}). "
                    f"Content-Type={r.headers.get('Content-Type')}, bytes={len(content)}"
                )
            except Exception as e:
                last_err = e

        # Fallback: allow user to manually place the PDF
        if not looks_like_pdf(local_path):
            raise RuntimeError(
                "Failed to download a valid GDPR PDF (received empty/HTML). "
                "Fix options:\n"
                "1) Manually download GDPR PDF in a browser and save it as:\n"
                f"   {local_path}\n"
                "2) Or update the URL in corpora_manifest.yaml to a working PDF endpoint.\n"
                f"Last error: {last_err}"
            )

    import fitz  # PyMuPDF

    docs: List[Document] = []
    pdf = fitz.open(local_path)
    for page_idx in range(len(pdf)):
        text = pdf[page_idx].get_text("text")
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "corpus_id": corpus_id,
                    "source_type": "pdf_url",
                    "source": url,
                    "local_path": local_path,
                    "page": page_idx + 1,
                    "tags": tags,
                },
            )
        )
    return docs


def load_sitemap_docs(corpus_id: str, url: str, include_regex: str | None, tags: List[str]) -> List[Document]:
    loader = SitemapLoader(url)
    docs = loader.load()

    rx = re.compile(include_regex) if include_regex else None
    out: List[Document] = []

    for d in docs:
        src = d.metadata.get("source", "")
        if rx and not rx.search(src):
            continue
        out.append(
            Document(
                page_content=d.page_content,
                metadata={
                    **d.metadata,
                    "corpus_id": corpus_id,
                    "source_type": "sitemap",
                    "tags": tags,
                },
            )
        )
    return out

def load_github_markdown(
    corpus_id: str,
    repo: str,
    branch: str,
    file_glob_suffix: str,
    tags: List[str],
) -> List[Document]:
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        print("[warn] GITHUB_PERSONAL_ACCESS_TOKEN not set; skipping GitHub corpus:", corpus_id)
        return []


    loader = GithubFileLoader(
        repo=repo,
        branch=branch,
        access_token=token,
        github_api_url="https://api.github.com",
        file_filter=lambda p: p.endswith(file_glob_suffix),
    )
    docs = loader.load()

    out: List[Document] = []
    for d in docs:
        out.append(
            Document(
                page_content=d.page_content,
                metadata={
                    **d.metadata,
                    "corpus_id": corpus_id,
                    "source_type": "github_markdown",
                    "tags": tags,
                },
            )
        )
    return out

def load_corpus(c: Dict[str, Any]) -> List[Document]:
    kind = c["kind"]
    corpus_id = c["id"]
    tags = c.get("tags", [])

    if kind == "pdf_url":
        return load_pdf_from_url(corpus_id, c["url"], tags)

    if kind == "sitemap":
        return load_sitemap_docs(corpus_id, c["url"], c.get("include_regex"), tags)

    if kind == "github_markdown":
        return load_github_markdown(
            corpus_id=corpus_id,
            repo=c["repo"],
            branch=c.get("branch", "master"),
            file_glob_suffix=c.get("file_glob_suffix", ".md"),
            tags=tags,
        )

    raise ValueError(f"Unknown corpus kind: {kind}")
