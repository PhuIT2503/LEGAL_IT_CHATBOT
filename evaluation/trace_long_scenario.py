#!/usr/bin/env python3
"""Capture a stage-by-stage trace for the long AI voice scenario.

This is diagnostic tooling only.  It observes the existing pipeline without
changing retrieval, applicability, generation, the public API, or benchmark
runner.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LONG_SCENARIO = (
    "Nhân viên A dùng AI giả giọng giám đốc, gọi kế toán yêu cầu chuyển "
    "300.000.000 đồng. Kế toán đã chuyển đủ 300.000.000 đồng. Ngân hàng "
    "phong tỏa được 180.000.000 đồng. 120.000.000 đồng đã được chuyển tiếp "
    "sang tài khoản khác. A khai rằng chỉ thử nghiệm công nghệ và chưa sử "
    "dụng số tiền. Hãy phân tích hành vi, mục đích, hậu quả, trách nhiệm pháp "
    "lý và biện pháp xử lý."
)


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding a runtime dependency."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFC", question).split())


def _question_sections(question: str) -> list[str]:
    return [
        section.strip()
        for section in re.split(r"(?<=[.!?])\s+|\n+", question)
        if section.strip()
    ]


def _build_llm(model: str):
    from langchain_openai import ChatOpenAI

    if model == "qwen2.5:7b":
        return ChatOpenAI(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            temperature=0.2,
            max_completion_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "1600")),
            timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
            max_retries=0,
            stream_usage=True,
        )
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the proxy model")
    return ChatOpenAI(
        model=model,
        base_url="https://api.shopaikey.com/v1",
        api_key=api_key,
        temperature=0.2,
        max_completion_tokens=int(os.getenv("PROXY_MAX_TOKENS", "1600")),
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "300")),
        stream_usage=True,
    )


def run_trace(model: str, question: str) -> dict[str, Any]:
    from qdrant_client import models
    from src.agents.agent_generation.prompts import (
        canonical_generation_payload_hash,
    )
    from src.retrieval.legal_domains import select_legal_domains
    from src.workflow.pipeline import ChatbotWorkflow
    import src.agents.common.legal_relevance_filter as relevance_module
    import src.agents.agent_retrieval.node_hybrid_search as retrieval_node_module

    llm = _build_llm(model)
    pipeline = ChatbotWorkflow(
        llm=llm,
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_path=os.getenv(
            "QDRANT_PATH", str(PROJECT_ROOT / "data" / ".qdrant_base")
        ),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding_v2"
        ),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pass=os.getenv("NEO4J_PASSWORD", "legal_kg_2024"),
        top_k=int(os.getenv("TOP_K", "5")),
        recursive_max_depth=int(os.getenv("RECURSIVE_MAX_DEPTH", "3")),
        recursive_max_iterations=int(os.getenv("RECURSIVE_MAX_ITERATIONS", "5")),
        grounding_repair_attempts=0,
        skip_router=True,
    )

    retrieval_output: dict[str, Any] = {}
    generation_calls: list[dict[str, Any]] = []
    applicability_calls: list[dict[str, Any]] = []
    reranked_pool: list[dict[str, Any]] = []

    original_retrieval = pipeline._retrieval_agent.run
    original_generation = pipeline._generation_agent.run_draft
    original_applicability = relevance_module.check_legal_applicability
    original_rerank = retrieval_node_module.rerank_context_records_with_behavior

    def traced_retrieval(state: dict[str, Any]) -> dict[str, Any]:
        output = original_retrieval(state)
        retrieval_output.update(copy.deepcopy(dict(output)))
        return output

    def traced_generation(state: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        output = original_generation(state)
        generation_calls.append(
            {
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "input": copy.deepcopy(dict(state)),
                "output": copy.deepcopy(dict(output)),
            }
        )
        return output

    def traced_applicability(*args: Any, **kwargs: Any):
        started = time.perf_counter()
        result = original_applicability(*args, **kwargs)
        applicability_calls.append(
            {
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "decisions": [_json_safe(decision) for decision in result.decisions],
                "retrieval_gap": bool(result.retrieval_gap),
                "gap_reason": str(result.gap_reason or ""),
            }
        )
        return result

    def traced_rerank(*args: Any, **kwargs: Any):
        result = original_rerank(*args, **kwargs)
        reranked_pool.extend(copy.deepcopy(list(result)))
        return result

    pipeline._retrieval_agent.run = traced_retrieval
    pipeline._generation_agent.run_draft = traced_generation
    relevance_module.check_legal_applicability = traced_applicability
    retrieval_node_module.rerank_context_records_with_behavior = traced_rerank
    started = time.perf_counter()
    try:
        public_result = pipeline.run(question, mode="critic")
    finally:
        total_latency_ms = (time.perf_counter() - started) * 1000.0
        pipeline._retrieval_agent.run = original_retrieval
        pipeline._generation_agent.run_draft = original_generation
        relevance_module.check_legal_applicability = original_applicability
        retrieval_node_module.rerank_context_records_with_behavior = original_rerank

    generation = generation_calls[-1]["output"] if generation_calls else {}
    final_context = list(generation.get("context_records", []))
    generation_payload = generation.get("generation_payload") or {
        "normalized_question": _normalized_question(question),
        "question": question,
        "context_texts": list(generation.get("context_texts", [])),
        "retrieval_is_complete": generation.get("retrieval_is_complete", True),
    }
    fact_state = generation.get("scenario_fact_state") or {}
    domain_selection = select_legal_domains(question)
    target_id = "Lu_t_An_ninh_m_ng_2025_D7_K2_Pg"
    target_reranked = next(
        (record for record in reranked_pool if record.get("chunk_id") == target_id),
        {},
    )

    domain_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.legal_domains",
                match=models.MatchAny(any=list(domain_selection.selected)),
            )
        ]
    )
    dense_vector = pipeline._retrieval_agent.embedding_model.encode(
        [question], normalize_embeddings=True
    )[0].tolist()
    sparse_indices, sparse_values = pipeline._retrieval_agent.bm25.encode_query(
        question
    )
    dense_points = pipeline._retrieval_agent.qdrant_client.query_points(
        collection_name=pipeline._retrieval_agent.qdrant_child_col,
        query=dense_vector,
        using="dense",
        query_filter=domain_filter,
        limit=pipeline._retrieval_agent.prefetch_limit,
        with_payload=True,
    ).points
    sparse_points = (
        pipeline._retrieval_agent.qdrant_client.query_points(
            collection_name=pipeline._retrieval_agent.qdrant_child_col,
            query=models.SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            ),
            using="bm25",
            query_filter=domain_filter,
            limit=pipeline._retrieval_agent.prefetch_limit,
            with_payload=True,
        ).points
        if sparse_indices
        else []
    )

    def rank_of(points: list[Any]) -> int | None:
        for rank, point in enumerate(points, start=1):
            if str((point.payload or {}).get("id") or "") == target_id:
                return rank
        return None

    raw_ce_order = sorted(
        reranked_pool,
        key=lambda record: -float(record.get("cross_encoder_score") or 0.0),
    )
    final_rerank_order = sorted(
        reranked_pool,
        key=lambda record: -float(record.get("score") or 0.0),
    )
    tracked_provision = {
        "provision": "Luật An ninh mạng 2025 Điều 7 Khoản 2 Điểm g",
        "target_chunk_id": target_id,
        "domain_opened": "cybersecurity" in domain_selection.selected,
        "dense_rank": rank_of(dense_points),
        "bm25_rank": rank_of(sparse_points),
        "hybrid_rank": target_reranked.get("original_rank"),
        "hybrid_rrf_score": target_reranked.get("original_rrf_score"),
        "cross_encoder_score": target_reranked.get("cross_encoder_score"),
        "cross_encoder_rank": next(
            (
                rank
                for rank, record in enumerate(raw_ce_order, start=1)
                if record.get("chunk_id") == target_id
            ),
            None,
        ),
        "post_behavior_rank": next(
            (
                rank
                for rank, record in enumerate(final_rerank_order, start=1)
                if record.get("chunk_id") == target_id
            ),
            None,
        ),
        "behavior_score": target_reranked.get("behavior_score"),
        "behavior_gate_status": (
            "PASS"
            if any(
                record.get("chunk_id") == target_id
                for record in retrieval_output.get("retrieved_chunks", [])
            )
            else "REMOVED"
        ),
        "applicability": next(
            (
                decision
                for call in applicability_calls
                for decision in call["decisions"]
                if decision.get("document") == "Luật An ninh mạng 2025"
                and str(decision.get("article")) == "7"
            ),
            None,
        ),
        "in_final_context": any(
            record.get("chunk_id") == target_id for record in final_context
        ),
    }
    return {
        "model": model,
        "normalized_question": fact_state.get(
            "normalized_question", _normalized_question(question)
        ),
        "question_sections": fact_state.get(
            "question_sections", _question_sections(question)
        ),
        "extracted_facts": fact_state,
        "behavior_card": retrieval_output.get("behavior_profile", {}),
        "selected_domains": list(domain_selection.selected),
        "retrieval_candidates": retrieval_output.get("retrieved_chunks", []),
        "recursive_candidate_pool": retrieval_output.get("context_records", []),
        "applicability_results": applicability_calls,
        "final_context": final_context,
        "generation_payload": generation_payload,
        "generation_payload_hash": canonical_generation_payload_hash(
            generation_payload
        ),
        "answer_assessment": generation.get("answer_assessment", {}),
        "raw_generation": generation.get("draft_response", ""),
        "final_answer": public_result.get("final_response", ""),
        "token_usage": public_result.get("token_usage", {}),
        "latency_ms": {
            "total": total_latency_ms,
            "generation_stage_including_applicability": (
                generation_calls[-1]["latency_ms"] if generation_calls else 0.0
            ),
            "applicability": sum(
                float(call["latency_ms"]) for call in applicability_calls
            ),
        },
        "tracked_provision": tracked_provision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--question", default=LONG_SCENARIO)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _load_dotenv(PROJECT_ROOT / ".env")
    payload = run_trace(args.model, args.question)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
