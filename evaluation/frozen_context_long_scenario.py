#!/usr/bin/env python3
"""Compare Generation models using one byte-identical canonical payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.trace_long_scenario import _build_llm, _load_dotenv
from src.agents.agent_generation.prompts import (
    build_answer_prompt,
    canonical_generation_payload_hash,
)
from src.agents.common.grounded_validation import (
    build_grounded_sources,
    validate_grounded_draft,
)
from src.agents.common.llm_client import LLMClient


FORBIDDEN_MISSING = (
    "Tiền hoặc tài sản đã được chuyển hay chưa.",
    "Hành vi mới ở giai đoạn chuẩn bị hay đã được thực hiện.",
    "Số tiền chuyển là bao nhiêu.",
)


def run_model(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    client = LLMClient(_build_llm(model))
    prompt = build_answer_prompt(
        str(payload["normalized_question"]),
        str(payload["final_context"]),
        is_complete=bool(payload["retrieval_is_complete"]),
        generation_payload=payload,
    )
    started = time.perf_counter()
    response = client.invoke(prompt, tag="frozen_generation")
    latency_ms = (time.perf_counter() - started) * 1000.0
    answer = str(response.content or "")
    sources = build_grounded_sources(
        [str(payload["final_context"])],
        query=str(payload["normalized_question"]),
    )
    validation = validate_grounded_draft(
        answer,
        sources,
        is_complete=bool(payload["retrieval_is_complete"]),
        query=str(payload["normalized_question"]),
    )
    return {
        "model": model,
        "canonical_payload_hash": canonical_generation_payload_hash(payload),
        "latency_ms": latency_ms,
        "token_usage": dict(client.token_usage),
        "answer": answer,
        "fact_preservation": {
            key: str(value) in answer
            for key, value in {
                "transferred_amount_vnd": "300.000.000",
                "frozen_amount_vnd": "180.000.000",
                "onward_transferred_amount_vnd": "120.000.000",
            }.items()
        },
        "missing_fact_contradictions": [
            text for text in FORBIDDEN_MISSING if text in answer
        ],
        "citation_correctness": {
            "is_valid": validation.is_valid,
            "used_source_ids": list(validation.used_source_ids),
        },
        "unsupported_claims_or_instruction_issues": list(validation.issues),
        "instruction_compliance": {
            "required_headings": {
                heading: heading in answer
                for heading in (
                    "## Tóm tắt tình huống",
                    "## Các vấn đề pháp lý",
                    "## Phân tích",
                    "## Chế tài",
                    "## Trả lời câu hỏi của người dùng",
                )
            },
            "has_valid_citation_marker": bool(
                re.search(r"\[\[CITE:S\d+(?:,S\d+)*\]\]", answer)
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _load_dotenv(PROJECT_ROOT / ".env")
    trace = json.loads(args.input_trace.read_text(encoding="utf-8"))
    payload = dict(trace["generation_payload"])
    payload.pop("model_config", None)
    result = {
        "canonical_payload_hash": canonical_generation_payload_hash(payload),
        "models": [
            run_model("qwen2.5:7b", payload),
            run_model("gpt-4o-mini", payload),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
