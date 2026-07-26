#!/usr/bin/env python3
"""Run the legal chatbot benchmark without changing the production pipeline.

The runner calls ``ChatbotWorkflow.run`` exactly once per testcase.  Temporary
wrappers only observe existing Retrieval, Generation and Applicability calls;
they are restored after every case and never invoke an extra LLM judge.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (  # noqa: E402
    METRIC_NAMES,
    aggregate_by,
    aggregate_metrics,
    applicability_accuracy,
    citation_accuracy,
    citations_from_answer,
    classify_errors,
    count_errors,
    expected_legal_units,
    hallucinated_citations,
    legal_recall_at,
    legal_unit_from_record,
    reciprocal_rank,
    recursive_metrics,
    set_precision_recall,
    slug,
)
from evaluation.reporting import write_report_artifacts  # noqa: E402


LOGGER = logging.getLogger("legal_benchmark")
REQUIRED_FIELDS = (
    "id",
    "category",
    "difficulty",
    "question",
    "expected_domains",
    "expected_behaviors",
    "expected_documents",
    "expected_articles",
    "expected_clauses",
    "expected_points",
)
REQUIRED_CATEGORIES = {
    "ai_deepfake",
    "personal_data",
    "cyber_attack",
    "sql_injection",
    "malware",
    "ai_copyright",
    "advertising",
    "consumer_protection",
    "network_security",
    "electronic_transactions",
}
REQUIRED_DIFFICULTIES = {"easy", "medium", "hard"}
SOURCE_MARKER_RE = re.compile(r"\[\[(?:CITE|QUOTE):(?P<ids>S\d+(?:\s*,\s*S\d+)*)\]\]")


def _safe_copy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("benchmark root phải là một JSON array")
    ids: set[str] = set()
    errors: list[str] = []
    for index, case in enumerate(payload):
        label = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{label} không phải object")
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in case]
        if missing:
            errors.append(f"{label} thiếu field: {', '.join(missing)}")
        case_id = str(case.get("id") or "")
        if not case_id:
            errors.append(f"{label} có id rỗng")
        elif case_id in ids:
            errors.append(f"id bị lặp: {case_id}")
        ids.add(case_id)
        for field in REQUIRED_FIELDS[4:]:
            if field in case and not isinstance(case[field], list):
                errors.append(f"{case_id or label}.{field} phải là array")
    if errors:
        raise ValueError("Benchmark không hợp lệ:\n- " + "\n- ".join(errors))
    return payload


def validate_coverage(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    category_map: dict[str, set[str]] = {}
    for case in cases:
        category_map.setdefault(slug(case.get("category")), set()).add(slug(case.get("difficulty")))
    missing_categories = REQUIRED_CATEGORIES.difference(category_map)
    if missing_categories:
        warnings.append("Thiếu category: " + ", ".join(sorted(missing_categories)))
    for category in sorted(REQUIRED_CATEGORIES.intersection(category_map)):
        missing = REQUIRED_DIFFICULTIES.difference(category_map[category])
        if missing:
            warnings.append(f"{category} thiếu difficulty: {', '.join(sorted(missing))}")
    return warnings


def filter_cases(
    cases: Sequence[dict[str, Any]],
    *,
    category: str | None,
    difficulty: str | None,
    case_id: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = list(cases)
    if category:
        wanted = slug(category)
        selected = [case for case in selected if wanted in slug(case["category"]) or slug(case["category"]) in wanted]
    if difficulty:
        wanted = slug(difficulty)
        selected = [case for case in selected if slug(case["difficulty"]) == wanted]
    if case_id:
        selected = [case for case in selected if str(case["id"]) == case_id]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


class PipelineTrace(AbstractContextManager["PipelineTrace"]):
    """Observe internal stages by wrapping existing calls, then restore them."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.retrieval_input: dict[str, Any] = {}
        self.retrieval_output: dict[str, Any] = {}
        self.retrieval_latency_ms = 0.0
        self.generation_calls: list[dict[str, Any]] = []
        self.applicability_calls: list[dict[str, Any]] = []
        self._originals: list[tuple[Any, str, Any]] = []

    def _replace(self, owner: Any, name: str, value: Any) -> None:
        self._originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def __enter__(self) -> "PipelineTrace":
        retrieval_agent = self.pipeline._retrieval_agent
        original_retrieval = retrieval_agent.run

        def traced_retrieval(state: dict[str, Any]) -> dict[str, Any]:
            self.retrieval_input = _safe_copy(dict(state))
            started = time.perf_counter()
            try:
                output = original_retrieval(state)
                self.retrieval_output = _safe_copy(dict(output))
                return output
            finally:
                self.retrieval_latency_ms += (time.perf_counter() - started) * 1000.0

        self._replace(retrieval_agent, "run", traced_retrieval)

        generation_agent = self.pipeline._generation_agent
        for method_name in ("run_draft", "run_final"):
            original = getattr(generation_agent, method_name)

            def make_wrapper(method: Any, stage: str):
                def wrapper(state: dict[str, Any]) -> dict[str, Any]:
                    call: dict[str, Any] = {
                        "stage": stage,
                        "input_context_records": _safe_copy(list(state.get("context_records", []))),
                        "input_context_texts": _safe_copy(list(state.get("context_texts", []))),
                    }
                    started = time.perf_counter()
                    output = method(state)
                    call.update(
                        {
                            "latency_ms": (time.perf_counter() - started) * 1000.0,
                            "output_context_records": _safe_copy(list(output.get("context_records", []))),
                            "output_context_texts": _safe_copy(list(output.get("context_texts", []))),
                            "retrieval_decisions": _safe_copy(list(output.get("retrieval_decisions", []))),
                            "draft_response": str(output.get("draft_response") or ""),
                            "final_response": str(output.get("final_response") or ""),
                            "retrieval_is_complete": bool(output.get("retrieval_is_complete", True)),
                        }
                    )
                    self.generation_calls.append(call)
                    return output

                return wrapper

            self._replace(generation_agent, method_name, make_wrapper(original, method_name))

        # legal_relevance_filter imported this function directly, therefore the
        # wrapper must be installed on that module binding (not the origin module).
        import src.agents.common.legal_relevance_filter as relevance_module

        original_applicability = relevance_module.check_legal_applicability

        def traced_applicability(*args: Any, **kwargs: Any):
            started = time.perf_counter()
            result = original_applicability(*args, **kwargs)
            self.applicability_calls.append(
                {
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "decisions": [_json_safe(decision) for decision in result.decisions],
                    "retrieval_gap": bool(result.retrieval_gap),
                    "gap_reason": str(result.gap_reason or ""),
                }
            )
            return result

        self._replace(relevance_module, "check_legal_applicability", traced_applicability)
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> bool:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        return False


def candidate_view(record: Mapping[str, Any], *, reason_removed: str = "") -> dict[str, Any]:
    unit = legal_unit_from_record(record)
    metadata = record.get("metadata") or {}
    return {
        **unit.as_dict(),
        "chunk_id": str(record.get("chunk_id") or record.get("id") or ""),
        "parent_id": str(record.get("parent_id") or metadata.get("parent_id") or ""),
        "score": float(record.get("score") or record.get("retrieval_score") or 0.0),
        "retrieval_score": record.get("retrieval_score"),
        "cross_encoder_score": record.get("cross_encoder_score"),
        "behavior_score": float(record.get("behavior_score") or 0.0),
        "recursive_depth": int(record.get("recursive_depth") or 0),
        "is_seed": bool(record.get("is_seed", True)),
        "expansion_reason": str(record.get("expansion_reason") or ""),
        "reason_kept": str(
            record.get("reason_kept")
            or record.get("expansion_reason")
            or "selected_by_phase2_ranking"
        ),
        "legal_domains": list(metadata.get("legal_domains") or []),
        "provenance_chain": list(record.get("provenance_chain") or []),
        "seed_preserved": bool(record.get("seed_preserved", False)),
        "seed_survived": bool(record.get("seed_survived", False)),
        "seed_removed": bool(record.get("seed_removed", False)),
        "behavior_preserved": bool(record.get("behavior_preserved", False)),
        "relevance_removed": bool(record.get("relevance_removed", False)),
        "applicability_removed": bool(record.get("applicability_removed", False)),
        "decision_stage": str(record.get("decision_stage") or ""),
        "reason_removed": str(record.get("reason_removed") or reason_removed),
        "text": str(record.get("text") or record.get("content") or ""),
    }


def _identity(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return legal_unit_from_record(record).identity


def _source_dict(source: Any) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "document": source.document,
        "article": source.article,
        "clause": source.clause or "",
        "point": source.point or "",
        "text": source.text,
    }


def _marker_citations(answer: str, source_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source_by_id = {str(source.get("source_id")): source for source in source_records}
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for match in SOURCE_MARKER_RE.finditer(answer or ""):
        for source_id in (item.strip() for item in match.group("ids").split(",")):
            if source_id in seen:
                continue
            seen.add(source_id)
            result.append(dict(source_by_id.get(source_id) or {"source_id": source_id, "missing": True}))
    return result


def _flatten_behaviors(profile: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ("actions", "objects", "purposes", "conditions"):
        for value in profile.get(field, []) or []:
            if value not in values:
                values.append(str(value))
    return values


def _git_metadata() -> dict[str, Any]:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "dirty": bool(command("git", "status", "--porcelain")),
    }


def run_case(pipeline: Any, case: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    from src.agents.common.grounded_validation import (
        build_grounded_sources,
        validate_grounded_draft,
    )
    from src.retrieval.legal_domains import count_candidates_by_domain, select_legal_domains

    question = str(case["question"])
    started = time.perf_counter()
    with PipelineTrace(pipeline) as trace:
        public_result = pipeline.run(question, mode=mode)
    total_latency_ms = (time.perf_counter() - started) * 1000.0

    retrieval_output = trace.retrieval_output
    retrieved = list(retrieval_output.get("retrieved_chunks", []))
    recursive_pool = list(retrieval_output.get("context_records", []))
    recursive_candidates = [record for record in recursive_pool if not bool(record.get("is_seed", True))]

    final_generation = trace.generation_calls[-1] if trace.generation_calls else {}
    pre_generation_records = list(final_generation.get("input_context_records", recursive_pool))
    final_records = list(final_generation.get("output_context_records", []))
    final_context_texts = list(final_generation.get("output_context_texts", []))
    retrieval_decisions = list(final_generation.get("retrieval_decisions", []))
    if not final_context_texts:
        final_context_texts = [str(record.get("text") or "") for record in final_records if record.get("text")]

    final_ids = {_identity(record) for record in final_records}
    removed_records = [
        candidate_view(record, reason_removed="removed_by_relevance_or_applicability_before_generation")
        for record in pre_generation_records
        if _identity(record) not in final_ids
    ]

    behavior_profile = dict(retrieval_output.get("behavior_profile") or {})
    actual_behaviors = _flatten_behaviors(behavior_profile)
    domain_selection = select_legal_domains(question)
    selected_domains = list(domain_selection.selected)

    sources = build_grounded_sources(final_context_texts, query=question)
    source_records = [_source_dict(source) for source in sources]
    raw_generation = str(
        final_generation.get("final_response")
        or final_generation.get("draft_response")
        or public_result.get("draft_response")
        or ""
    )
    grounding_validation = validate_grounded_draft(
        raw_generation,
        sources,
        is_complete=bool(public_result.get("retrieval_is_complete", True)),
        query=question,
    )
    generated_citations = _marker_citations(raw_generation, source_records)
    rendered_answer = str(public_result.get("final_response") or "")
    rendered_citations = citations_from_answer(rendered_answer)
    hallucinated = hallucinated_citations(rendered_citations, source_records)

    expected_units = expected_legal_units(case)
    domain_precision, domain_recall = set_precision_recall(selected_domains, case["expected_domains"])
    behavior_precision, behavior_recall = set_precision_recall(actual_behaviors, case["expected_behaviors"])
    recursive_precision, recursive_noise = recursive_metrics(recursive_pool, expected_units)

    applicability_decisions = [
        decision
        for call in trace.applicability_calls
        for decision in call.get("decisions", [])
    ]
    if applicability_decisions:
        applicability_candidates = [
            {"document": item.get("document", ""), "article": item.get("article", "")}
            for item in applicability_decisions
        ]
        applicability_kept = [
            {"document": item.get("document", ""), "article": item.get("article", "")}
            for item in applicability_decisions
            if str(item.get("decision", "")).upper() in {"KEEP", "WEAK_KEEP"}
            or (
                not item.get("decision")
                and str(item.get("level", "")).upper() in {"HIGH", "MEDIUM"}
            )
        ]
    else:
        applicability_candidates = pre_generation_records
        applicability_kept = final_records

    metrics = {
        "domain_recall": domain_recall,
        "domain_precision": domain_precision,
        "behavior_recall": behavior_recall,
        "behavior_precision": behavior_precision,
        "recall_at_5": legal_recall_at(retrieved, expected_units, 5),
        "recall_at_10": legal_recall_at(retrieved, expected_units, 10),
        "mrr": reciprocal_rank(retrieved, expected_units),
        "citation_accuracy": citation_accuracy(rendered_citations, expected_units),
        "wrong_domain_rate": 1.0 - domain_precision,
        "wrong_behavior_rate": 1.0 - behavior_precision,
        "recursive_precision": recursive_precision,
        "recursive_noise_rate": recursive_noise,
        "applicability_accuracy": applicability_accuracy(
            applicability_candidates,
            applicability_kept,
            expected_units,
        ),
        "retrieval_latency_ms": trace.retrieval_latency_ms,
        "total_latency_ms": total_latency_ms,
    }
    errors = classify_errors(
        metrics=metrics,
        selected_domains=selected_domains,
        expected_domains=case["expected_domains"],
        actual_behaviors=actual_behaviors,
        expected_behaviors=case["expected_behaviors"],
        citations=rendered_citations,
        expected_units=expected_units,
        hallucinated=hallucinated,
        grounding_issues=list(grounding_validation.issues),
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "question": question,
        "status": "completed",
        "expected": {
            "domains": list(case["expected_domains"]),
            "behaviors": list(case["expected_behaviors"]),
            "legal_units": [unit.as_dict() for unit in expected_units],
        },
        "actual": {
            "selected_domains": selected_domains,
            "filtered_domains": list(domain_selection.filtered),
            "domain_scores": domain_selection.scores,
            "domain_fallback": domain_selection.used_fallback,
            "behavior_card": behavior_profile,
            "behaviors": actual_behaviors,
            "candidate_count_by_domain": count_candidates_by_domain(retrieved, selected_domains),
            "retrieval_candidates": [candidate_view(record) for record in retrieved],
            "recursive_candidates": [candidate_view(record) for record in recursive_candidates],
            "final_candidate_pool": [candidate_view(record) for record in pre_generation_records],
            "removed_before_generation": removed_records,
            "final_context": [candidate_view(record) for record in final_records],
            "applicability": trace.applicability_calls,
            "retrieval_decisions": retrieval_decisions,
            "grounded_sources": source_records,
            "citations_by_generation": generated_citations,
            "citations_in_rendered_answer": rendered_citations,
            "hallucinated_citations": hallucinated,
            "raw_generation": raw_generation,
            "rendered_answer": rendered_answer,
            "retrieval_is_complete": bool(public_result.get("retrieval_is_complete", True)),
            "retrieval_is_relevant": bool(public_result.get("retrieval_is_relevant", True)),
        },
        "metrics": metrics,
        "errors": errors,
        "grounding_validation": {
            "is_valid": grounding_validation.is_valid,
            "issues": list(grounding_validation.issues),
            "used_source_ids": list(grounding_validation.used_source_ids),
        },
        "token_usage": public_result.get("token_usage", {}),
    }


def failed_case(case: Mapping[str, Any], exc: BaseException, elapsed_ms: float) -> dict[str, Any]:
    metrics = {name: 0.0 for name in METRIC_NAMES}
    metrics["total_latency_ms"] = elapsed_ms
    return {
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "difficulty": case.get("difficulty", ""),
        "question": case.get("question", ""),
        "status": "runtime_error",
        "runtime_error": f"{type(exc).__name__}: {exc}",
        "runtime_traceback": traceback.format_exc(),
        "metrics": metrics,
        "errors": ["Missing Relevant Law", "Generation Grounding Error"],
    }


def build_summary(
    *,
    benchmark_path: Path,
    benchmark_sha256: str,
    selected_cases: Sequence[Mapping[str, Any]],
    details: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    completed = [item for item in details if item.get("status") == "completed"]
    return {
        "schema_version": "1.0",
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "generated_at": generated_at,
        "benchmark_file": str(benchmark_path),
        "benchmark_sha256": benchmark_sha256,
        "selected_cases": len(selected_cases),
        "completed_cases": len(completed),
        "runtime_error_cases": len(details) - len(completed),
        "filters": {
            "category": args.category,
            "difficulty": args.difficulty,
            "case_id": args.case_id,
            "limit": args.limit,
        },
        "pipeline_config": {
            "mode": args.mode,
            "top_k": args.top_k,
            "qdrant_path": os.getenv("QDRANT_PATH", "data/.qdrant_base"),
            "qdrant_url_configured": bool(os.getenv("QDRANT_URL")),
            "embedding_model": os.getenv(
                "EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding_v2"
            ),
            "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        },
        "git": _git_metadata(),
        "overall_metrics": aggregate_metrics(details),
        "metrics_by_category": aggregate_by(details, "category"),
        "metrics_by_difficulty": aggregate_by(details, "difficulty"),
        "error_counts": count_errors(details),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Retrieval Pipeline của chatbot pháp lý")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "benchmark" / "benchmark.json",
        help="Đường dẫn benchmark JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results",
        help="Thư mục chứa summary/details/report/plots",
    )
    parser.add_argument("--category", help="Tên hoặc slug category, ví dụ deepfake/sql_injection")
    parser.add_argument("--difficulty", help="Easy, Medium hoặc Hard")
    parser.add_argument("--case-id", help="Chạy đúng một testcase theo ID")
    parser.add_argument("--limit", type=int, help="Giới hạn số case sau khi lọc")
    parser.add_argument("--top-k", type=int, default=10, help="Candidate top-k; mặc định 10 để tính Recall@10")
    parser.add_argument("--mode", choices=("naive", "article_expand", "critic"), default="critic")
    parser.add_argument("--no-plots", action="store_true", help="Không sinh biểu đồ")
    parser.add_argument("--fail-fast", action="store_true", help="Dừng ở runtime error đầu tiên")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Chỉ kiểm tra schema/coverage benchmark, không load model hoặc gọi pipeline",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    benchmark_path = args.benchmark.expanduser().resolve()
    cases = load_benchmark(benchmark_path)
    coverage_warnings = validate_coverage(cases)
    for warning in coverage_warnings:
        LOGGER.warning("Benchmark coverage: %s", warning)
    selected = filter_cases(
        cases,
        category=args.category,
        difficulty=args.difficulty,
        case_id=args.case_id,
        limit=args.limit,
    )
    if not selected:
        LOGGER.error("Không có testcase khớp bộ lọc.")
        return 2
    LOGGER.info("Benchmark hợp lệ: %d case; đã chọn %d case.", len(cases), len(selected))
    if args.validate_only:
        for case in selected:
            LOGGER.info("  %s | %s | %s", case["id"], case["category"], case["difficulty"])
        return 0

    # Import runtime only after validation, so dataset CI does not need Qdrant,
    # embeddings, Neo4j, Ollama or LangChain installed.
    LOGGER.info("Khởi tạo pipeline hiện tại (mode=%s, top_k=%d).", args.mode, args.top_k)
    from langchain_openai import ChatOpenAI
    from src.workflow.pipeline import ChatbotWorkflow

    llm = ChatOpenAI(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        temperature=0.2,
    )
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
        top_k=args.top_k,
        recursive_max_depth=int(os.getenv("RECURSIVE_MAX_DEPTH", "3")),
        recursive_max_iterations=int(os.getenv("RECURSIVE_MAX_ITERATIONS", "5")),
        grounding_repair_attempts=int(os.getenv("GROUNDING_REPAIR_ATTEMPTS", "0")),
        skip_router=True,
    )
    details: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        LOGGER.info("[%d/%d] %s", index, len(selected), case["id"])
        started = time.perf_counter()
        try:
            detail = run_case(pipeline, case, mode=args.mode)
            LOGGER.info(
                "  R@5=%.3f R@10=%.3f MRR=%.3f total=%.1fms errors=%s",
                detail["metrics"]["recall_at_5"],
                detail["metrics"]["recall_at_10"],
                detail["metrics"]["mrr"],
                detail["metrics"]["total_latency_ms"],
                detail["errors"] or "none",
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.exception("Case %s thất bại.", case["id"])
            if args.fail_fast:
                raise
            detail = failed_case(case, exc, elapsed_ms)
        details.append(_json_safe(detail))

    benchmark_sha256 = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    summary = build_summary(
        benchmark_path=benchmark_path,
        benchmark_sha256=benchmark_sha256,
        selected_cases=selected,
        details=details,
        args=args,
    )
    artifacts = write_report_artifacts(
        summary,
        details,
        args.output_dir.expanduser().resolve(),
        make_plots=not args.no_plots,
    )
    LOGGER.info("Hoàn tất. Report: %s", artifacts["report"])
    if artifacts.get("plot_error"):
        LOGGER.warning("Không sinh được plot: %s", artifacts["plot_error"])
    return 0 if not summary["runtime_error_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
