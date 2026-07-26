"""JSON, Markdown and matplotlib outputs for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PLOT_METRICS = (
    ("domain_recall", "Domain Recall"),
    ("behavior_recall", "Behavior Recall"),
    ("recall_at_5", "Recall@5"),
    ("citation_accuracy", "Citation Accuracy"),
    ("wrong_domain_rate", "Wrong Domain Rate"),
)


def _percent(value: Any) -> str:
    return f"{100.0 * float(value or 0.0):.2f}%"


def _milliseconds(value: Any) -> str:
    return f"{float(value or 0.0):.1f} ms"


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def render_markdown(summary: Mapping[str, Any], details: Sequence[Mapping[str, Any]]) -> str:
    metrics = summary.get("overall_metrics", {})
    lines = [
        "# Legal Retrieval Benchmark Report",
        "",
        f"- Run ID: `{summary.get('run_id', '')}`",
        f"- Generated: `{summary.get('generated_at', '')}`",
        f"- Benchmark SHA-256: `{summary.get('benchmark_sha256', '')}`",
        f"- Cases: **{summary.get('completed_cases', 0)}/{summary.get('selected_cases', 0)}** completed",
        f"- Runtime errors: **{summary.get('runtime_error_cases', 0)}**",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    metric_labels = (
        ("domain_recall", "Domain Recall", False),
        ("domain_precision", "Domain Precision", False),
        ("behavior_recall", "Behavior Recall", False),
        ("behavior_precision", "Behavior Precision", False),
        ("recall_at_5", "Recall@5", False),
        ("recall_at_10", "Recall@10", False),
        ("mrr", "MRR", False),
        ("citation_accuracy", "Citation Accuracy", False),
        ("wrong_domain_rate", "Wrong Domain Rate", False),
        ("wrong_behavior_rate", "Wrong Behavior Rate", False),
        ("recursive_precision", "Recursive Precision", False),
        ("recursive_noise_rate", "Recursive Noise Rate", False),
        ("applicability_accuracy", "Applicability Accuracy", False),
        ("retrieval_latency_ms", "Average Retrieval Latency", True),
        ("total_latency_ms", "Average Total Latency", True),
    )
    for key, label, latency in metric_labels:
        formatter = _milliseconds if latency else _percent
        lines.append(f"| {label} | {formatter(metrics.get(key, 0.0))} |")

    lines.extend(["", "## Metrics by Category", ""])
    category_metrics = summary.get("metrics_by_category", {})
    lines.extend(
        [
            "| Category | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Wrong domain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, payload in category_metrics.items():
        item = payload.get("metrics", {})
        lines.append(
            f"| {name} | {payload.get('case_count', 0)} | {_percent(item.get('domain_recall'))} "
            f"| {_percent(item.get('behavior_recall'))} | {_percent(item.get('recall_at_5'))} "
            f"| {_percent(item.get('recall_at_10'))} | {_percent(item.get('citation_accuracy'))} "
            f"| {_percent(item.get('wrong_domain_rate'))} |"
        )

    lines.extend(["", "## Metrics by Difficulty", ""])
    lines.extend(
        [
            "| Difficulty | Cases | Domain R | Behavior R | R@5 | R@10 | Citation | Applicability |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, payload in summary.get("metrics_by_difficulty", {}).items():
        item = payload.get("metrics", {})
        lines.append(
            f"| {name} | {payload.get('case_count', 0)} | {_percent(item.get('domain_recall'))} "
            f"| {_percent(item.get('behavior_recall'))} | {_percent(item.get('recall_at_5'))} "
            f"| {_percent(item.get('recall_at_10'))} | {_percent(item.get('citation_accuracy'))} "
            f"| {_percent(item.get('applicability_accuracy'))} |"
        )

    lines.extend(["", "## Top Recurring Errors", ""])
    error_counts = summary.get("error_counts", {})
    if any(error_counts.values()):
        for name, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- **{name}**: {count}")
    else:
        lines.append("- No classified errors.")

    lines.extend(["", "## Top 20 Failed Cases", ""])
    failed = [item for item in details if item.get("errors") or item.get("status") != "completed"]
    failed.sort(
        key=lambda item: (
            -len(item.get("errors", [])),
            float(item.get("metrics", {}).get("recall_at_10", 0.0)),
            float(item.get("metrics", {}).get("citation_accuracy", 0.0)),
            str(item.get("id", "")),
        )
    )
    if not failed:
        lines.append("No failed cases.")
    for index, item in enumerate(failed[:20], start=1):
        error_text = ", ".join(item.get("errors", [])) or item.get("runtime_error", "Runtime error")
        lines.extend(
            [
                f"### {index}. `{item.get('id', '')}` — {item.get('category', '')} / {item.get('difficulty', '')}",
                "",
                f"- Errors: {error_text}",
                f"- Recall@10: {_percent(item.get('metrics', {}).get('recall_at_10'))}",
                f"- Citation Accuracy: {_percent(item.get('metrics', {}).get('citation_accuracy'))}",
                f"- Question: {item.get('question', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            "- `benchmark_summary.json`: aggregate metrics and run metadata.",
            "- `benchmark_details.json`: per-case trace, metrics and errors.",
            "- `plots/`: visual metrics generated from the same summary.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_plots(summary: Mapping[str, Any], plots_dir: Path) -> list[str]:
    """Create one stable PNG per requested metric; import matplotlib lazily."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    categories = list(summary.get("metrics_by_category", {}).keys())
    written: list[str] = []
    for key, title in PLOT_METRICS:
        labels = ["Overall", *categories]
        values = [float(summary.get("overall_metrics", {}).get(key, 0.0))]
        values.extend(
            float(summary["metrics_by_category"][category]["metrics"].get(key, 0.0))
            for category in categories
        )
        width = max(10.0, 0.62 * len(labels))
        figure, axis = plt.subplots(figsize=(width, 5.5))
        color = "#b42318" if key.startswith("wrong_") else "#175cd3"
        bars = axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.set_ylabel("Rate")
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=35)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 1.02),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        figure.tight_layout()
        output = plots_dir / f"{key}.png"
        figure.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written.append(str(output))
    return written


def write_report_artifacts(
    summary: dict[str, Any],
    details: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    make_plots: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.json"
    details_path = output_dir / "benchmark_details.json"
    report_path = output_dir / "benchmark_report.md"

    plot_files: list[str] = []
    plot_error = ""
    if make_plots:
        try:
            plot_files = generate_plots(summary, output_dir / "plots")
        except (ImportError, ModuleNotFoundError) as exc:
            plot_error = f"matplotlib unavailable: {exc}"
    summary["plot_files"] = plot_files
    summary["plot_error"] = plot_error

    _json_dump(summary_path, summary)
    _json_dump(details_path, list(details))
    report_path.write_text(render_markdown(summary, details), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "details": str(details_path),
        "report": str(report_path),
        "plots": plot_files,
        "plot_error": plot_error,
    }

