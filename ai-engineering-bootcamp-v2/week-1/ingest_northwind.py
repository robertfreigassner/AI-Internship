"""Ingest Northwind sample docs through POST /ingest.

Reads every doc*.txt next to the Week 2 notebook, posts the text,
prints per-file chunk counts, then prints total vectors in Pinecone.

Run from anywhere (uvicorn must be up on port 8000):

  cd ai-engineering-bootcamp-v2/week-1
  source .venv/bin/activate
  python ingest_northwind.py
"""

from pathlib import Path

import httpx

API = "http://127.0.0.1:8000"
DOCS_DIR = Path(__file__).resolve().parent.parent / "week-2" / "rag-vector-databases"


def main() -> None:
    files = sorted(DOCS_DIR.glob("doc*.txt"))
    if not files:
        raise SystemExit(f"No doc*.txt files in {DOCS_DIR}")

    total_indexed = 0
    with httpx.Client(timeout=120.0) as client:
        for path in files:
            document_id = path.stem  # stable id, e.g. doc1_handbook
            text = path.read_text(encoding="utf-8")
            response = client.post(
                f"{API}/ingest",
                json={
                    "document_id": document_id,
                    "source": path.name,
                    "text": text,
                },
            )
            response.raise_for_status()
            body = response.json()
            chunks = body["chunks_indexed"]
            total_indexed += chunks
            print(f"{path.name}  document_id={document_id}  chunks_indexed={chunks}  status={body['status']}")

        stats = client.get(f"{API}/debug/pinecone").json()
        print(f"total_vector_count={stats.get('total_vector_count')}")
        print(f"(this run indexed {total_indexed} chunks across {len(files)} file(s))")


if __name__ == "__main__":
    main()
