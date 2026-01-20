from __future__ import annotations

import os
from typing import Any, List

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

def pg_connection_url() -> str:
    return (
        f"postgresql+psycopg://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}"
    )

def ensure_database_exists() -> None:
    db_name = os.environ["PGDATABASE"]
    admin_db = os.getenv("PGADMIN_DATABASE", "postgres")
    admin_url = (
        f"postgresql+psycopg://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{admin_db}"
    )
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name},
            ).scalar()
            if not exists:
                safe_name = db_name.replace('"', '""')
                conn.execute(text(f'CREATE DATABASE "{safe_name}"'))
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Failed to verify/create the Postgres database. "
            "Ensure Postgres is running and the PG* environment variables are set. "
            "If the database doesn't exist, create it or run docker compose up -d."
        ) from exc
    finally:
        engine.dispose()

def get_embeddings() -> Any:
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, api_key=os.environ["OPENAI_API_KEY"])

def get_store(collection_name: str) -> PGVector:
    ensure_database_exists()
    return PGVector(
        connection=pg_connection_url(),
        embeddings=get_embeddings(),
        collection_name=collection_name,
        use_jsonb=True,
    )

def upsert_texts(store: PGVector, chunks: List[str], metadatas: List[dict]) -> None:
    docs = [Document(page_content=t, metadata=m) for t, m in zip(chunks, metadatas)]
    store.add_documents(docs)

def similarity_search(store: PGVector, query: str, k: int):
    return store.similarity_search_with_score(query, k=k)
