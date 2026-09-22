"""Week 1 live demo — five stages in one file, built up live in class."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

from pinecone_store import embedding_model, ingest_document, pinecone_status, query_similar

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


DEFAULT_RAG_TOP_K = 5

GROUNDING_PROMPT_TEMPLATE = """You answer questions using ONLY the context chunks below.

Rules:
- Use only facts that appear in the context. Do not use outside knowledge.
- Whenever you use a fact, cite its document_id in square brackets, like [handbook].
- If the context is missing or not enough to answer, refuse. Say you cannot answer from the provided documents. Set sources_needed to true and keep confidence low.
- If you can answer from the context, set sources_needed to false.

Question:
{question}

Context:
{context}
"""


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    top_k: int | None = None  # RAG: how many chunks to retrieve (default RAG_TOP_K / 5).


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str]


class IngestRequest(BaseModel):
    text: str = ""
    document_id: str = ""
    source: str | None = None
    metadata: dict | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


def rag_top_k(override: int | None) -> int:
    if override is not None:
        return override
    raw = os.getenv("RAG_TOP_K", str(DEFAULT_RAG_TOP_K)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_RAG_TOP_K


def format_context(matches: list[dict]) -> str:
    if not matches:
        return "(no chunks retrieved)"
    blocks = []
    for match in matches:
        metadata = match.get("metadata") or {}
        chunk_id = match.get("id") or ""
        document_id = metadata.get("document_id") or ""
        text = metadata.get("text") or ""
        blocks.append(
            f"[chunk_id={chunk_id} document_id={document_id}]\n{text}"
        )
    return "\n\n".join(blocks)


def build_grounding_prompt(question: str, matches: list[dict]) -> str:
    """Fill GROUNDING_PROMPT_TEMPLATE with the question and retrieved chunks."""

    return GROUNDING_PROMPT_TEMPLATE.format(
        question=question,
        context=format_context(matches),
    )


def retrieve_matches(question: str, top_k: int) -> tuple[list[dict], list[str]]:
    raw_matches = query_similar(question, top_k=top_k)
    chunk_ids = [str(match.get("id")) for match in raw_matches if match.get("id")]
    return raw_matches, chunk_ids


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(user_content: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    RAG passes a grounding prompt here instead of the raw question.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": user_content}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(user_content: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{user_content}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.get("/health")
def health() -> dict:
    """Liveness plus a Pinecone reachability check. Never returns secrets."""

    payload: dict = {"status": "ok", "service": "week-1-ask"}
    try:
        payload["pinecone"] = pinecone_status()
    except Exception as exc:
        payload["status"] = "degraded"
        payload["pinecone"] = {"ok": False, "error": str(exc)}
    return payload


@app.get("/debug/pinecone")
def debug_pinecone() -> dict:
    """Call this to confirm Pinecone is reachable and the index dimensions match embeddings."""

    try:
        return pinecone_status()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/debug/retrieve")
def debug_retrieve(q: str = "", top_k: int = 5) -> dict:
    """Embed the question and return nearest chunks. Does not call the chat model.

    curl -s "http://127.0.0.1:8000/debug/retrieve?q=What%20is%20RAG"
    """

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="q must not be empty")
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1")

    try:
        raw_matches = query_similar(question, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {exc}") from exc

    matches = []
    for match in raw_matches:
        metadata = match.get("metadata") or {}
        matches.append(
            {
                "id": match.get("id"),
                "score": match.get("score"),
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "source": metadata.get("source"),
                "text": metadata.get("text"),
            }
        )

    return {
        "query": question,
        "embedding_model": embedding_model(),
        "top_k": top_k,
        "llm_called": False,
        "matches": matches,
    }


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk, embed with text-embedding-3-small, and upsert into Pinecone.

    curl -s -X POST http://127.0.0.1:8000/ingest \
      -H "Content-Type: application/json" \
      -d '{"document_id": "doc-1", "source": "notes.txt", "text": "RAG retrieves relevant chunks before the model answers."}'
    """

    document_id = body.document_id.strip()
    text = body.text.strip()
    source = (body.source or "").strip()
    if not source and isinstance(body.metadata, dict):
        raw_source = body.metadata.get("source")
        source = str(raw_source).strip() if raw_source is not None else ""

    if not document_id:
        raise HTTPException(status_code=400, detail="document_id must not be empty")
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        chunks_indexed = ingest_document(document_id, text, source=source)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {exc}") from exc

    if chunks_indexed == 0:
        raise HTTPException(status_code=400, detail="text produced no chunks to index")

    return IngestResponse(
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        status="indexed",
    )


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Retrieve top-k chunks, ground the Session 1 generation path, return cost + chunk ids.

    curl -s -X POST http://127.0.0.1:8000/ask \
      -H "Content-Type: application/json" \
      -d '{"question": "Where is Northwind headquarters?", "model": "gpt-4o-mini"}'
    """

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    model = body.model or DEFAULT_MODEL
    top_k = rag_top_k(body.top_k)
    start = time.perf_counter()

    try:
        matches, retrieved_chunk_ids = retrieve_matches(question, top_k)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieve failed: {exc}") from exc

    grounded = build_grounding_prompt(question, matches)
    last_error: str | None = None

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    grounded, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounded, model
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )
