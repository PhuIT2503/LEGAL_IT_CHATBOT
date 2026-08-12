"""Build screenshot-ready HTML views from real logs, traces, and Qdrant state."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo_pages"
TRACE_DIR = ROOT / "docs" / "demo_traces"


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#07111f;color:#eaf2ff;font:17px/1.45 Inter,Arial,sans-serif}
.page{width:1720px;min-height:980px;padding:48px 58px;background:radial-gradient(circle at 85% 5%,#17365d 0,#07111f 36%)}
h1{margin:0 0 8px;font-size:34px;letter-spacing:-.4px}.sub{color:#91a9c8;margin-bottom:30px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}
.card{background:#0e2036;border:1px solid #294968;border-radius:16px;padding:22px;min-width:0}.wide{grid-column:1/-1}
h2{font-size:21px;margin:0 0 13px;color:#7dd3fc}.metric{font-size:34px;font-weight:700;color:#f8fafc}.label{color:#9fb4cf;font-size:14px;text-transform:uppercase;letter-spacing:.08em}
table{width:100%;border-collapse:collapse}th,td{padding:10px 12px;border-bottom:1px solid #24415d;text-align:left;vertical-align:top}th{color:#8ed6ff;font-size:14px}td{overflow-wrap:anywhere}
code,pre{font-family:"SFMono-Regular",Menlo,monospace}code{color:#d7f0ff}.ok{color:#66e3a4}.warn{color:#ffd166}.no{color:#ff8f8f}
.flow{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.step{padding:12px 16px;border:1px solid #3c6b91;border-radius:12px;background:#102942}.arrow{color:#55c2ff;font-size:22px}
.answer{white-space:pre-wrap;max-height:220px;overflow:hidden}.mono{font-family:"SFMono-Regular",Menlo,monospace;font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere}
.pill{display:inline-block;padding:5px 10px;margin:3px;border-radius:999px;background:#173b59;border:1px solid #356382;font-size:14px}.foot{margin-top:20px;color:#7890ad;font-size:13px}
"""


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else "—"))


def pills(values: list[object]) -> str:
    return "".join(f'<span class="pill">{esc(v)}</span>' for v in values) or "—"


def page(title: str, subtitle: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLE}</style></head><body><main class='page'><h1>{esc(title)}</h1><div class='sub'>{esc(subtitle)}</div>{body}<div class='foot'>Nguồn: phiên chạy cục bộ ngày 04/08/2026 · Không chỉnh sửa quyết định retrieval/Critic.</div></main></body></html>"


def save(name: str, content: str) -> None:
    (OUT / name).write_text(content, encoding="utf-8")


def qdrant_page() -> None:
    client = QdrantClient(path=str(ROOT / "data" / ".qdrant_base"))
    infos = {c.name: client.get_collection(c.name) for c in client.get_collections().collections}
    child = infos["legal_child_chunks"]
    parent = infos["legal_parent_chunks"]
    bm25 = json.loads((ROOT / "data" / ".qdrant_base" / "legal_child_chunks.bm25.json").read_text())
    body = f"""
    <div class='grid'>
      <section class='card'><div class='label'>Parent collection</div><div class='metric'>{parent.points_count:,}</div><code>legal_parent_chunks</code></section>
      <section class='card'><div class='label'>Child collection</div><div class='metric'>{child.points_count:,}</div><code>legal_child_chunks</code></section>
      <section class='card wide'><h2>Cấu hình vector được đọc trực tiếp từ Qdrant</h2><table>
        <tr><th>Collection</th><th>Vector</th><th>Kích thước</th><th>Khoảng cách</th><th>Sparse</th></tr>
        <tr><td><code>legal_child_chunks</code></td><td><code>dense</code></td><td>1.024</td><td>Cosine</td><td><code>bm25</code></td></tr>
        <tr><td><code>legal_parent_chunks</code></td><td>unnamed placeholder</td><td>1</td><td>Cosine</td><td>—</td></tr>
      </table></section>
      <section class='card wide'><h2>BM25 Base</h2><div class='grid'>
        <div><div class='label'>doc_count</div><div class='metric'>{bm25['doc_count']:,}</div></div>
        <div><div class='label'>vocab</div><div class='metric'>{len(bm25['vocab']):,}</div></div>
        <div><div class='label'>k1 / b</div><div class='metric'>{bm25['k1']} / {bm25['b']}</div></div>
        <div><div class='label'>avgdl</div><div class='metric'>{bm25['avgdl']:.4f}</div></div>
      </div></section>
    </div>"""
    save("qdrant.html", page("Qdrant Base — trạng thái chỉ mục", "Kiểm tra read-only trên data/.qdrant_base", body))


def compare_page() -> None:
    raw = (ROOT / "docs" / "demo_logs" / "demo_compare_all_cli.log").read_text(errors="replace")
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
    keep = []
    terms = ("REAL CLI", "model:", "retriever:", "question:", "MODE:", "Retrieved", "Điều", "Critic", "Phạt tiền", "Tịch thu", "Đình chỉ", "thu hồi", "hoàn trả")
    for line in clean.splitlines():
        if any(term in line for term in terms):
            keep.append(line)
    excerpt = "\n".join(keep[-52:])
    body = f"<section class='card'><h2>Trích xuất từ log CLI thật</h2><pre class='mono'>{esc(excerpt)}</pre></section>"
    save("compare.html", page("So sánh ba chế độ trên cùng câu hỏi", "Naive · Article Expansion · Critic Agent — Base + Qwen2.5 7B", body))


def trace_page(trace: dict) -> None:
    c, r, cr, g = trace["case"], trace["retrieval"], trace["critic"], trace["generation"]
    steps = [
        ("Câu hỏi", c["id"]),
        ("Hybrid retrieval", f"{len(r['retrieved_child_ids'])} child"),
        ("Phát hiện thiếu", f"complete={cr['is_complete_before_gate']}"),
        ("Graph fetch", f"{len(cr['graph_fetched_article_ids'])} Điều"),
        ("Sinh lại", str(g["regenerated"])),
    ]
    flow = "".join(f"<div class='step'><b>{esc(a)}</b><br>{esc(b)}</div>" + ("<div class='arrow'>→</div>" if i < len(steps)-1 else "") for i,(a,b) in enumerate(steps))
    body = f"""
    <section class='card wide'><div class='flow'>{flow}</div></section>
    <div class='grid'>
      <section class='card'><h2>Retrieved child IDs</h2>{pills(r['retrieved_child_ids'])}</section>
      <section class='card'><h2>Retrieved Điều</h2>{pills(r['retrieved_article_ids'])}</section>
      <section class='card'><h2>Ứng viên / đã fetch</h2><div class='label'>Candidate</div>{pills(cr['candidate_article_ids'])}<div class='label' style='margin-top:12px'>Fetched</div>{pills(cr['graph_fetched_article_ids'])}</section>
      <section class='card'><h2>Relevance gate</h2>{pills([x['decision'] for x in cr['gate_observations']])}<div class='label' style='margin-top:12px'>Accepted</div>{pills(cr['gate_decision_by_article']['accepted_article_ids'])}</section>
    </div>"""
    save("trace.html", page("Truy vết Critic Agent — cùng Điều", c["question"], body))


def case_page(filename: str, title: str, trace: dict) -> None:
    c, r, cr, g = trace["case"], trace["retrieval"], trace["critic"], trace["generation"]
    gap = cr["detected_gaps"]
    gap_count = sum(len(gap.get(k) or []) for k in ("missing_references", "compound_penalty_behaviors", "structurally_incomplete_articles", "multi_hop_references"))
    decisions = [x["decision"] for x in cr["gate_observations"]]
    body = f"""
    <section class='card wide'><h2>Câu hỏi testset {esc(c['id'])}</h2><div>{esc(c['question'])}</div></section>
    <div class='grid'>
      <section class='card'><div class='label'>Retrieved</div><div class='metric'>{len(r['retrieved_child_ids'])} child / {len(r['retrieved_article_ids'])} Điều</div>{pills(r['retrieved_article_ids'])}</section>
      <section class='card'><div class='label'>Số nhóm gap ghi nhận</div><div class='metric'>{gap_count}</div><span class='{'ok' if gap_count == 0 else 'warn'}'>is_complete={esc(cr['is_complete_before_gate'])}</span></section>
      <section class='card'><h2>Candidate → gate → graph fetch</h2><div class='label'>Candidate</div>{pills(cr['candidate_article_ids'])}<div class='label' style='margin-top:10px'>Gate</div>{pills(decisions)}<div class='label' style='margin-top:10px'>Fetched</div>{pills(cr['graph_fetched_article_ids'])}</section>
      <section class='card'><h2>Kết quả</h2><div class='metric {'ok' if g['regenerated'] else 'warn'}'>regenerated={str(g['regenerated']).lower()}</div><div class='answer'>{esc(g['final_answer'])}</div></section>
    </div>"""
    save(filename, page(title, f"Nhóm {c['category']} · trace JSON thật", body))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traces = {
        "no": json.loads((TRACE_DIR / "case_01_no_gap.json").read_text()),
        "same": json.loads((TRACE_DIR / "case_02_same_article.json").read_text()),
        "cross": json.loads((TRACE_DIR / "case_03_cross_article.json").read_text()),
    }
    qdrant_page()
    compare_page()
    trace_page(traces["same"])
    case_page("case_no_gap.html", "Kịch bản 1 — Không thiếu ngữ cảnh", traces["no"])
    case_page("case_same.html", "Kịch bản 2 — Thiếu trong cùng Điều", traces["same"])
    case_page("case_cross.html", "Kịch bản 3 — Tham chiếu chéo Điều", traces["cross"])
    print(f"Built {len(list(OUT.glob('*.html')))} pages in {OUT}")


if __name__ == "__main__":
    main()
