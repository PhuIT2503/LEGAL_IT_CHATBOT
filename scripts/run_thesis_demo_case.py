"""Run one thesis case study with the production workflow and save an evidence trace.

This helper does not replace or force any retrieval/Critic decision.  It wraps the
existing relevance-gate function only to record the boolean result that the
production function returned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_chatbot import build_llm, build_pipeline  # noqa: E402
from src.agents.agent_critic import relevance_gate  # noqa: E402
from src.agents.agent_critic import node_critic_check  # noqa: E402


class WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def load_case(case_id: str) -> dict:
    path = PROJECT_ROOT / "data" / "eval_testset.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") == case_id:
                return row
    raise KeyError(f"Không tìm thấy case {case_id!r} trong {path}")


def compact_candidate_ids(report: dict) -> list[str]:
    ids: list[str] = []
    for item in report.get("missing_references", []):
        value = item.get("missing_dieu_id")
        if value:
            ids.append(value)
    for key in ("compound_penalty_behaviors", "structurally_incomplete_dieu"):
        for item in report.get(key, []):
            value = item.get("dieu_id")
            if value:
                ids.append(value)
    return list(dict.fromkeys(ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    output_path = Path(args.output)
    log_path = Path(args.log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format, datefmt="%H:%M:%S"))
    warning_collector = WarningCollector()
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(warning_collector)

    case = load_case(args.case_id)
    gate_observations: list[dict] = []
    original_gate = relevance_gate.is_candidate_relevant

    def observed_gate(query: str, candidate_content: str, *, llm_client) -> bool:
        warning_count_before = len(warning_collector.records)
        decision = original_gate(query, candidate_content, llm_client=llm_client)
        new_warnings = warning_collector.records[warning_count_before:]
        fail_open = any("Relevance gate lỗi" in row["message"] for row in new_warnings)
        gate_observations.append(
            {
                "decision": "fail_open" if fail_open else ("accept" if decision else "reject"),
                "returned_boolean": decision,
                "candidate_article_id": "not_logged_by_gate_function",
                "candidate_content_sha256": hashlib.sha256(candidate_content.encode("utf-8")).hexdigest(),
                "candidate_preview": candidate_content[:240],
            }
        )
        return decision

    relevance_gate.is_candidate_relevant = observed_gate
    node_critic_check.is_candidate_relevant = observed_gate

    errors: list[dict] = []
    result: dict = {}
    try:
        llm = build_llm()
        # Matches scripts/run_evaluation.py: the testset contains legal questions,
        # so the router is skipped to avoid an unrelated classification call.
        pipeline = build_pipeline(llm, top_k=args.top_k, skip_router=True)
        result = pipeline.run(case["question"], mode="critic")
        usage_by_tag = dict(pipeline._llm_client.token_usage_by_tag)
    except Exception as exc:  # preserve a structured failure instead of hiding it
        logging.exception("Demo case failed")
        usage_by_tag = {}
        errors.append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        relevance_gate.is_candidate_relevant = original_gate
        node_critic_check.is_candidate_relevant = original_gate

    report = result.get("critic_report") or {}
    fetched_ids = result.get("graph_fetched_dieu_ids") or []
    rejected_ids = report.get("rejected_by_relevance_gate") or []
    accepted_ids = [item for item in fetched_ids if item not in rejected_ids]
    retrieved_article_ids = result.get("retrieved_dieu_ids", [])
    final_context_article_ids = list(dict.fromkeys([*retrieved_article_ids, *fetched_ids]))
    detected_gap_types = []
    for key, value in {
        "missing_references": report.get("missing_references", []),
        "compound_penalty_behaviors": report.get("compound_penalty_behaviors", []),
        "structurally_incomplete_articles": report.get("structurally_incomplete_dieu", []),
        "multi_hop_references": report.get("multi_hop_references", []),
        "mixed_document_safeguard": report.get("reinforced_top_dieu_multi_van_ban"),
    }.items():
        if value:
            detected_gap_types.append(key)

    retrieved_children = []
    for item in result.get("retrieved_chunks") or []:
        retrieved_children.append(
            {
                "child_id": item.get("chunk_id"),
                "article_id_raw": item.get("dieu_id_raw"),
                "document_id_raw": item.get("van_ban_id_raw"),
                "score": item.get("score"),
                "text": item.get("text"),
            }
        )

    trace = {
        "run": {
            "timestamp": datetime.now().astimezone().isoformat(),
            "source_snapshot": "9bf145e6744599e1e15c21e2011a02f7ecff87f5",
            "testset": "data/eval_testset.jsonl",
            "retriever": "Base",
            "qdrant_path": "data/.qdrant_base",
            "embedding_model": os.getenv("EMBEDDING_MODEL", "not_set"),
            "embedding_dimension": 1024,
            "llm_provider": "Ollama native",
            "llm_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            "mode": "critic",
            "top_k": args.top_k,
            "skip_router": True,
        },
        "case": {
            "id": case.get("id"),
            "category": case.get("category"),
            "question": case.get("question"),
            "required_facts": case.get("required_facts", []),
            "gold_article_ids": case.get("dieu_ids", []),
            "reference_answer": case.get("reference_answer"),
        },
        "evidence_summary": {
            "testset_id": case.get("id"),
            "category": case.get("category"),
            "question": case.get("question"),
            "required_facts": case.get("required_facts", []),
            "gold_dieu_ids": case.get("dieu_ids", []),
            "mode": "critic",
            "top_k": args.top_k,
            "retrieved_child_ids": [item.get("child_id") for item in retrieved_children],
            "retrieved_dieu_ids": retrieved_article_ids,
            "draft_answer": result.get("draft_response") or None,
            "detected_gap_types": detected_gap_types,
            "candidate_dieu_ids": compact_candidate_ids(report),
            "graph_fetched_dieu_ids": fetched_ids,
            "gate_decisions": [item.get("decision") for item in gate_observations],
            "accepted_dieu_ids": accepted_ids,
            "rejected_dieu_ids": rejected_ids,
            "final_context_dieu_ids": final_context_article_ids,
            "final_answer": result.get("final_response") or None,
            "required_facts_coverage": "not_logged_by_pipeline",
            "token_usage": result.get("token_usage", {}),
            "llm_call_count": (result.get("token_usage") or {}).get("call_count"),
            "errors": errors,
            "warnings": warning_collector.records,
        },
        "retrieval": {
            "retrieved_children": retrieved_children,
            "retrieved_child_ids": [item.get("child_id") for item in retrieved_children],
            "retrieved_article_ids": retrieved_article_ids,
        },
        "critic": {
            "draft_answer": result.get("draft_response") or None,
            "is_complete_before_gate": report.get("is_complete"),
            "detected_gaps": {
                "missing_references": report.get("missing_references", []),
                "compound_penalty_behaviors": report.get("compound_penalty_behaviors", []),
                "structurally_incomplete_articles": report.get("structurally_incomplete_dieu", []),
                "multi_hop_references": report.get("multi_hop_references", []),
                "mixed_document_safeguard": report.get("reinforced_top_dieu_multi_van_ban"),
            },
            "candidate_article_ids": compact_candidate_ids(report),
            "graph_fetched_article_ids": fetched_ids,
            "gate_observations": gate_observations,
            "gate_decision_by_article": {
                "accepted_article_ids": accepted_ids,
                "rejected_article_ids": rejected_ids,
                "mapping_note": "Accepted IDs are recorded from the production code path: an article appears in graph_fetched_dieu_ids and not in rejected_by_relevance_gate. The gate function does not log the article ID argument directly.",
            },
            "critic_report": report,
        },
        "generation": {
            "final_context": {
                "retrieved_contexts": [item.get("text") for item in retrieved_children if item.get("text")],
                "critic_graph_context": result.get("graph_context") or None,
            },
            "final_answer": result.get("final_response") or None,
            "regenerated": bool(result.get("graph_context")),
        },
        "usage": {
            "total": result.get("token_usage", {}),
            "final_answer": result.get("final_answer_token_usage", {}),
            "by_tag": usage_by_tag,
        },
        "warnings": warning_collector.records,
        "errors": errors,
        "not_logged_fields": [
            "per-step wall-clock latency",
            "candidate article ID inside each individual gate call",
            "explicit required-fact coverage judgment",
        ],
    }

    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(trace, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps({
        "output": str(output_path),
        "case_id": case.get("id"),
        "retrieved_article_ids": result.get("retrieved_dieu_ids", []),
        "candidate_article_ids": compact_candidate_ids(report),
        "graph_fetched_article_ids": fetched_ids,
        "gate_observations": gate_observations,
        "regenerated": bool(result.get("graph_context")),
        "errors": errors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
