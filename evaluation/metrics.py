"""Pure, deterministic metrics for the legal retrieval benchmark.

This module deliberately has no dependency on the chatbot runtime.  Keeping the
scoring contract separate makes historical result files reproducible even when
the retrieval implementation changes later.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


METRIC_NAMES = (
    "domain_recall",
    "domain_precision",
    "behavior_recall",
    "behavior_precision",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "citation_accuracy",
    "wrong_domain_rate",
    "wrong_behavior_rate",
    "recursive_precision",
    "recursive_noise_rate",
    "applicability_accuracy",
    "retrieval_latency_ms",
    "total_latency_ms",
)

ERROR_TYPES = (
    "Wrong Domain",
    "Wrong Behavior",
    "Wrong Citation",
    "Missing Relevant Law",
    "Recursive Noise",
    "Applicability Error",
    "Generation Grounding Error",
    "Hallucinated Citation",
)

_CHUNK_COORDINATES_RE = re.compile(
    r"_D(?P<article>\d+[a-zA-Z]*)"
    r"(?:_K(?P<clause>\d+))?"
    r"(?:_P(?P<point>[^_]+))?",
    re.IGNORECASE,
)


def normalise(value: Any) -> str:
    """Accent/punctuation-insensitive key used by every comparison."""

    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    return "".join(char for char in text if char.isalnum())


def slug(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


@dataclass(frozen=True)
class LegalUnit:
    document: str = ""
    article: str = ""
    clause: str = ""
    point: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "document": self.document,
            "article": self.article,
            "clause": self.clause,
            "point": self.point,
        }

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return tuple(
            normalise(value)
            for value in (self.document, self.article, self.clause, self.point)
        )


def _at(values: Sequence[Any], index: int, *, reuse_single: bool = True) -> str:
    if index < len(values):
        return str(values[index] or "")
    if reuse_single and len(values) == 1:
        return str(values[0] or "")
    return ""


def expected_legal_units(case: Mapping[str, Any]) -> list[LegalUnit]:
    """Build coordinate tuples from the required parallel benchmark arrays.

    One document is reused for many articles.  Otherwise arrays are paired by
    index.  Clause/point may be empty when the expected granularity is an
    article.  Duplicate coordinates are collapsed without reordering.
    """

    documents = list(case.get("expected_documents") or [])
    articles = list(case.get("expected_articles") or [])
    clauses = list(case.get("expected_clauses") or [])
    points = list(case.get("expected_points") or [])
    width = max(len(documents), len(articles), len(clauses), len(points), 0)
    units: list[LegalUnit] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index in range(width):
        unit = LegalUnit(
            document=_at(documents, index),
            article=_at(articles, index),
            clause=_at(clauses, index),
            point=_at(points, index),
        )
        if not unit.document and not unit.article:
            continue
        if unit.identity in seen:
            continue
        seen.add(unit.identity)
        units.append(unit)
    return units


def legal_unit_from_record(record: Mapping[str, Any]) -> LegalUnit:
    metadata = record.get("metadata") or {}
    chunk_id = str(record.get("chunk_id") or record.get("id") or "")
    chunk_match = _CHUNK_COORDINATES_RE.search(chunk_id)
    return LegalUnit(
        document=str(
            record.get("document")
            or record.get("source")
            or metadata.get("source")
            or ""
        ).removesuffix(".docx"),
        article=str(
            record.get("article")
            or metadata.get("article_number")
            or (chunk_match.group("article") if chunk_match else "")
        ),
        clause=str(
            record.get("clause")
            or metadata.get("clause_number")
            or (chunk_match.group("clause") if chunk_match and chunk_match.group("clause") else "")
        ),
        point=str(
            record.get("point")
            or metadata.get("point_number")
            or (chunk_match.group("point") if chunk_match and chunk_match.group("point") else "")
        ),
    )


def _document_matches(actual: str, expected: str) -> bool:
    if not expected:
        return True
    left, right = normalise(actual), normalise(expected)
    if not left or not right:
        return False
    return left == right or (min(len(left), len(right)) >= 10 and (left in right or right in left))


def unit_matches(actual: LegalUnit, expected: LegalUnit) -> bool:
    """Expected empty clause/point acts as a deliberate article-level wildcard."""

    if not _document_matches(actual.document, expected.document):
        return False
    for actual_value, expected_value in (
        (actual.article, expected.article),
        (actual.clause, expected.clause),
        (actual.point, expected.point),
    ):
        if expected_value and normalise(actual_value) != normalise(expected_value):
            return False
    return True


def deduplicate_units(records: Iterable[Mapping[str, Any]]) -> list[LegalUnit]:
    result: list[LegalUnit] = []
    seen: set[tuple[str, str, str, str]] = set()
    for record in records:
        unit = legal_unit_from_record(record)
        if not unit.document and not unit.article:
            continue
        if unit.identity in seen:
            continue
        seen.add(unit.identity)
        result.append(unit)
    return result


def set_precision_recall(actual: Iterable[Any], expected: Iterable[Any]) -> tuple[float, float]:
    actual_set = {normalise(value) for value in actual if normalise(value)}
    expected_set = {normalise(value) for value in expected if normalise(value)}
    true_positive = len(actual_set.intersection(expected_set))
    precision = true_positive / len(actual_set) if actual_set else (1.0 if not expected_set else 0.0)
    recall = true_positive / len(expected_set) if expected_set else 1.0
    return precision, recall


def legal_recall_at(
    ranked_records: Sequence[Mapping[str, Any]],
    expected: Sequence[LegalUnit],
    k: int,
) -> float:
    if not expected:
        return 1.0
    actual = deduplicate_units(ranked_records[:k])
    hits = sum(any(unit_matches(item, target) for item in actual) for target in expected)
    return hits / len(expected)


def reciprocal_rank(
    ranked_records: Sequence[Mapping[str, Any]], expected: Sequence[LegalUnit]
) -> float:
    if not expected:
        return 1.0
    for rank, record in enumerate(ranked_records, start=1):
        actual = legal_unit_from_record(record)
        if any(unit_matches(actual, target) for target in expected):
            return 1.0 / rank
    return 0.0


def unit_precision(records: Sequence[Mapping[str, Any]], expected: Sequence[LegalUnit]) -> float:
    actual = deduplicate_units(records)
    if not actual:
        return 1.0 if not expected else 0.0
    hits = sum(any(unit_matches(item, target) for target in expected) for item in actual)
    return hits / len(actual)


def recursive_metrics(
    records: Sequence[Mapping[str, Any]], expected: Sequence[LegalUnit]
) -> tuple[float, float]:
    recursive = [record for record in records if not bool(record.get("is_seed", True))]
    if not recursive:
        return 1.0, 0.0
    # Recursive expansion commonly fetches an article-level parent that contains
    # the expected clause/point.  Score that parent at its real granularity.
    article_expected = [LegalUnit(item.document, item.article) for item in expected]
    precision = unit_precision(recursive, article_expected)
    return precision, 1.0 - precision


def applicability_accuracy(
    candidate_records: Sequence[Mapping[str, Any]],
    final_records: Sequence[Mapping[str, Any]],
    expected: Sequence[LegalUnit],
) -> float:
    """Binary keep/drop accuracy over the candidate pool before applicability."""

    candidates = deduplicate_units(candidate_records)
    if not candidates:
        return 1.0 if not expected else 0.0
    kept = deduplicate_units(final_records)
    # Applicability decisions are intentionally article-level, even when the
    # benchmark ground truth is more granular.
    article_expected = [LegalUnit(item.document, item.article) for item in expected]
    correct = 0
    for candidate in candidates:
        should_keep = any(unit_matches(candidate, target) for target in article_expected)
        was_kept = any(item.identity == candidate.identity for item in kept)
        correct += int(should_keep == was_kept)
    return correct / len(candidates)


_CITATION_RE = re.compile(
    r"(?P<document>"
    r"(?:Luật|Bộ [Ll]uật|Nghị [Đđ]ịnh|Thông tư)[^\n,;()]{2,100}?"
    r"|\d{4}_[^\n,;()]{2,100}?"
    r"),\s*"
    r"Điều\s+(?P<article>[0-9A-Za-z]+)"
    r"(?:,\s*Khoản\s+(?P<clause>[0-9A-Za-z]+))?"
    r"(?:,\s*Điểm\s+(?P<point>[^\s,;.)]+))?",
)


def citations_from_answer(answer: str) -> list[dict[str, str]]:
    """Extract rendered Vietnamese legal citations without trusting the LLM trace."""

    clean = str(answer or "").replace("**", "")
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for match in _CITATION_RE.finditer(clean):
        document = match.group("document").strip(" -*>")
        document = re.sub(
            r"^(?:căn cứ|theo|trích từ)\s*:?\s*",
            "",
            document,
            flags=re.IGNORECASE,
        ).strip()
        record = {
            "document": document,
            "article": match.group("article") or "",
            "clause": match.group("clause") or "",
            "point": match.group("point") or "",
        }
        unit = legal_unit_from_record(record)
        if unit.identity not in seen:
            seen.add(unit.identity)
            records.append(record)
    return records


def citation_accuracy(citations: Sequence[Mapping[str, Any]], expected: Sequence[LegalUnit]) -> float:
    actual = deduplicate_units(citations)
    if not actual and not expected:
        return 1.0
    hits = sum(any(unit_matches(item, target) for target in expected) for item in actual)
    precision = hits / len(actual) if actual else 0.0
    recalled = sum(any(unit_matches(item, target) for item in actual) for target in expected)
    recall = recalled / len(expected) if expected else 1.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def hallucinated_citations(
    citations: Sequence[Mapping[str, Any]], final_sources: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    sources = deduplicate_units(final_sources)
    result: list[dict[str, str]] = []
    for record in citations:
        citation = legal_unit_from_record(record)
        if not any(unit_matches(citation, source) for source in sources):
            result.append(citation.as_dict())
    return result


def classify_errors(
    *,
    metrics: Mapping[str, float],
    selected_domains: Sequence[str],
    expected_domains: Sequence[str],
    actual_behaviors: Sequence[str],
    expected_behaviors: Sequence[str],
    citations: Sequence[Mapping[str, Any]],
    expected_units: Sequence[LegalUnit],
    hallucinated: Sequence[Mapping[str, Any]],
    grounding_issues: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    domain_precision, domain_recall = set_precision_recall(selected_domains, expected_domains)
    behavior_precision, behavior_recall = set_precision_recall(actual_behaviors, expected_behaviors)
    if domain_precision < 1.0 or domain_recall < 1.0:
        errors.append("Wrong Domain")
    if behavior_precision < 1.0 or behavior_recall < 1.0:
        errors.append("Wrong Behavior")
    if expected_units and metrics.get("citation_accuracy", 0.0) < 1.0:
        errors.append("Wrong Citation")
    if metrics.get("recall_at_10", 0.0) < 1.0:
        errors.append("Missing Relevant Law")
    if metrics.get("recursive_noise_rate", 0.0) > 0.0:
        errors.append("Recursive Noise")
    if metrics.get("applicability_accuracy", 0.0) < 1.0:
        errors.append("Applicability Error")
    if grounding_issues:
        errors.append("Generation Grounding Error")
    if hallucinated:
        errors.append("Hallucinated Citation")
    return [error for error in ERROR_TYPES if error in errors]


def aggregate_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Macro average so each legal question has equal weight."""

    if not details:
        return {name: 0.0 for name in METRIC_NAMES}
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = [float(item.get("metrics", {}).get(name, 0.0)) for item in details]
        result[name] = sum(values) / len(values)
    return result


def aggregate_by(details: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in details:
        groups.setdefault(str(item.get(field) or "unknown"), []).append(item)
    return {
        name: {"case_count": len(items), "metrics": aggregate_metrics(items)}
        for name, items in sorted(groups.items())
    }


def count_errors(details: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(error for item in details for error in item.get("errors", []))
    return {name: int(counter.get(name, 0)) for name in ERROR_TYPES}
