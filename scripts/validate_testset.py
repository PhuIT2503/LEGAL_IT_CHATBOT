"""
scripts/validate_testset.py
=============================
Kiểm định độ tin cậy của CHÍNH bộ test set (data/eval_testset.jsonl, do LLM —
Opus 4.8 — sinh ra) — KHÔNG đánh giá chatbot, mà đánh giá xem câu hỏi/fact/đáp
án mẫu trong benchmark có đáng tin để dùng làm ground-truth xuyên suốt khóa
luận hay không. Nếu required_facts/reference_answer sai lệch so với văn bản
luật gốc, mọi số liệu Completeness Rate tính từ trước tới giờ đều mất giá trị
— đây là lý do bắt buộc phải kiểm định riêng bộ test set.

Gồm 2 lớp kiểm tra:

A. KHÔNG cần LLM (deterministic, chắc chắn tuyệt đối):
   A1. existence  — dieu_id có tồn tại thật trong corpus (data/keep) không
                     (bắt Điều bị bịa/ghi sai số — lỗi nghiêm trọng nhất)
   A2. schema     — cross_reference phải có >=2 dieu_ids, 3 nhóm còn lại đúng 1
   A3. duplicate  — câu hỏi có bị trùng/gần trùng câu khác trong bộ test không

B. LLM-as-judge (mặc định gpt-4o-mini), đối chiếu với ĐÚNG văn bản Điều gốc
   (lấy từ data/keep, không suy đoán):
   B1. facts_grounded          — từng required_fact có đúng với văn bản gốc
   B2. answer_grounded         — reference_answer có trung thực với văn bản gốc
   B3. facts_covered_by_answer — reference_answer có nêu đủ mọi required_fact
                                 (tự nhất quán nội bộ giữa 2 trường cùng 1 câu)
   B4. natural_clear           — câu hỏi có tự nhiên, rõ ràng, giống người thật hỏi
   B5. requires_citation       — có bắt buộc phải biết đúng Điều này mới trả lời
                                 đúng/đủ được không (hay chỉ cần kiến thức chung)
   B6. category_correct        — nhãn category có đúng với nội dung câu hỏi/Điều

Cách dùng:
    python scripts/validate_testset.py
    python scripts/validate_testset.py --limit 20          # test nhanh 20 câu đầu
    python scripts/validate_testset.py --judge-model gpt-4o-mini --max-retries 2
    python scripts/validate_testset.py --output-suffix _v1

Checkpoint: kết quả LLM-judge từng câu lưu tại
data/testset_validation_checkpoint<suffix>.json ngay sau khi chấm xong — chạy
lại đúng lệnh cũ sẽ chỉ chấm tiếp câu chưa có, không mất tiền/thời gian chấm
lại (giống cơ chế đã dùng cho score_evaluation.py).
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Chỉ cần PyTorch/text — ép transformers (kéo theo bởi vài import gián tiếp)
# bỏ qua hẳn việc detect/import TensorFlow, tránh xung đột protobuf trên Colab.
os.environ.setdefault("USE_TF", "0")

from langchain_openai import ChatOpenAI  # noqa: E402

from src.data_processing.chunking import VBPLChunker  # noqa: E402

# Endpoint OpenAI-compatible dùng cho LLM chấm điểm — hardcode tại đây (đổi
# thẳng dòng này nếu muốn dùng proxy khác hoặc để "" để dùng OpenAI gốc).
OPENAI_BASE_URL = "https://api.shopaikey.com/v1"

CATEGORY_DEFINITIONS = {
    "same_dieu_compound_penalty": (
        "ĐIỀU LUẬT ĐƯỢC TRÍCH DẪN (KHÔNG PHẢI cách đặt câu hỏi) quy định NHIỀU hình "
        "thức xử phạt/hậu quả pháp lý khác nhau (hình phạt kép, vd phạt tiền + tịch "
        "thu + biện pháp khắc phục) cho CÙNG một hành vi vi phạm — nên để trả lời "
        "ĐÚNG VÀ ĐỦ, câu trả lời bắt buộc phải liệt kê hết các hình thức đó. Câu hỏi "
        "vẫn có thể đặt dạng mở, tự nhiên (vd 'hành vi X bị xử phạt thế nào?') mà "
        "KHÔNG tự liệt kê trước các hình thức xử phạt trong câu hỏi — điều đó KHÔNG "
        "vi phạm nhãn này, vì tính 'kép' nằm ở nội dung Điều luật/đáp án đúng, không "
        "nằm ở cách hỏi."
    ),
    "cross_reference": (
        "Câu hỏi mà để trả lời ĐẦY ĐỦ, bắt buộc phải kết hợp thông tin từ HAI Điều "
        "khác nhau có quan hệ dẫn chiếu qua lại (1 Điều dẫn chiếu sang Điều kia)."
    ),
    "structural_multi_part": (
        "Câu hỏi cần thông tin từ NHIỀU khoản/điểm khác nhau trong CÙNG MỘT Điều để "
        "trả lời đầy đủ — không thể trả lời đủ chỉ bằng 1 khoản/điểm đơn lẻ."
    ),
    "control_no_gap": (
        "TOÀN BỘ nội dung cần thiết để trả lời ĐẦY ĐỦ nằm trong ĐÚNG 1 Điều luật được "
        "trích dẫn — có thể cần MỘT hoặc VÀI khoản/điểm trong CHÍNH Điều đó (không giới "
        "hạn số khoản), miễn là KHÔNG cần bất kỳ Điều luật nào khác, KHÔNG có hình phạt "
        "kép cần liệt kê đủ, KHÔNG có khoảng trống nào cần tham chiếu/mở rộng RA NGOÀI "
        "Điều này. 'no_gap' ở đây nghĩa là không có gap CROSS-ĐIỀU (không cần Điều khác), "
        "KHÔNG phải yêu cầu chỉ đúng 1 khoản."
    ),
}

GROUNDEDNESS_PROMPT = """Bạn là chuyên gia pháp lý, kiểm tra độ CHÍNH XÁC của 1 mục dữ liệu benchmark so với văn bản luật GỐC.

VĂN BẢN ĐIỀU LUẬT GỐC (nguồn DUY NHẤT được phép dùng để đối chiếu, không dùng kiến thức ngoài):
---
{source_text}
---

CÁC FACT BẮT BUỘC (required_facts) cần kiểm tra từng cái một, theo đúng thứ tự:
{facts_numbered}

CÂU TRẢ LỜI MẪU (reference_answer) cần kiểm tra:
{reference_answer}

Trả lời CHỈ bằng JSON (không thêm chữ nào khác, không markdown code fence), đúng định dạng:
{{
  "facts_grounded": [true, false, ...],
  "answer_grounded": true,
  "facts_covered_by_answer": [true, false, ...]
}}
Trong đó:
- facts_grounded: 1 giá trị cho MỖI fact theo đúng thứ tự trên — true nếu fact đó được văn bản gốc nêu đúng (chấp nhận diễn đạt lại nhưng phải đúng ý, đúng số liệu cụ thể), false nếu văn bản gốc không nêu hoặc nêu khác.
- answer_grounded: true nếu TOÀN BỘ câu trả lời mẫu trung thực với văn bản gốc, không bịa thêm nội dung không có, không mâu thuẫn với văn bản gốc.
- facts_covered_by_answer: 1 giá trị cho MỖI fact theo đúng thứ tự trên — true nếu fact đó ĐƯỢC nhắc tới trong câu trả lời mẫu (bất kể đúng/sai so với gốc, chỉ xét có được đề cập hay không)."""

QUESTION_QUALITY_PROMPT = """Bạn đang kiểm định chất lượng câu hỏi trong 1 bộ benchmark hỏi-đáp pháp luật.

VĂN BẢN ĐIỀU LUẬT LIÊN QUAN (để đánh giá câu hỏi có thực sự cần đến nó không):
---
{source_text}
---

CÂU HỎI CẦN ĐÁNH GIÁ:
{question}

Trả lời CHỈ bằng JSON (không thêm chữ nào khác, không markdown code fence):
{{
  "natural_clear": true,
  "requires_citation": true
}}
Trong đó:
- natural_clear: true nếu câu hỏi được diễn đạt tự nhiên, rõ ràng, giống cách một người dùng thật sẽ hỏi (không lộ vẻ máy tạo/kiểu đề thi cứng nhắc, không mơ hồ đa nghĩa).
- requires_citation: true nếu để trả lời ĐẦY ĐỦ VÀ CHÍNH XÁC câu hỏi này, bắt buộc phải biết đúng nội dung văn bản Điều luật cụ thể trên (không thể trả lời đúng/đủ chỉ bằng kiến thức pháp luật chung chung)."""

CATEGORY_PROMPT = """Bạn đang kiểm định xem 1 câu hỏi trong bộ benchmark có được gán ĐÚNG nhãn phân loại hay không.

LƯU Ý QUAN TRỌNG: các nhãn dưới đây mô tả CẤU TRÚC CỦA VĂN BẢN ĐIỀU LUẬT/ĐÁP ÁN ĐÚNG,
KHÔNG PHẢI cách đặt câu hỏi. Câu hỏi hoàn toàn có thể diễn đạt dạng mở, tự nhiên (vd
"hành vi X bị xử phạt thế nào?") mà KHÔNG tự liệt kê/gợi ý trước nội dung — đó KHÔNG
phải lỗi. Việc bạn cần đánh giá là: để trả lời ĐÚNG VÀ ĐỦ câu hỏi này, đáp án BẮT BUỘC
phải có đúng cấu trúc nội dung mô tả trong định nghĩa nhãn hay không (dựa trên chính
văn bản Điều luật, KHÔNG dựa vào cách hành văn của câu hỏi).

LƯU Ý THÊM về cấu trúc Nghị định xử phạt hành chính: "Hình thức xử phạt bổ sung" và
"Biện pháp khắc phục hậu quả" THƯỜNG được quy định ở 1 khoản RIÊNG (thường ở CUỐI
Điều), ghi rõ dạng "đối với hành vi vi phạm quy định tại khoản X Điều này" — PHẢI đọc
kỹ để xác định khoản X đó có TRÙNG với khoản chứa đúng hành vi vi phạm trong câu hỏi
hay không. Nếu trùng, Điều này VẪN CÓ hình phạt kép cho đúng hành vi đó, dù hình thức
bổ sung/khắc phục nằm tách riêng ở khoản khác, KHÔNG nằm chung khoản với phạt tiền.

ĐỊNH NGHĨA NHÃN "{category}":
{category_definition}

VĂN BẢN ĐIỀU LUẬT LIÊN QUAN:
---
{source_text}
---

CÂU HỎI ĐƯỢC GÁN NHÃN NÀY:
{question}

Để trả lời ĐÚNG VÀ ĐỦ câu hỏi này, đáp án có BẮT BUỘC phải khớp đúng cấu trúc nội dung mô tả trong nhãn "{category}" ở trên không (dựa trên chính nội dung văn bản Điều luật, KHÔNG dựa vào cách hành văn của câu hỏi)? Trả lời CHỈ bằng JSON (không thêm chữ nào khác, không markdown code fence):
{{
  "category_correct": true,
  "reason": "giải thích ngắn gọn 1 câu"
}}"""


def load_corpus_dieu_map(data_dir: str) -> dict:
    """dieu_id (viết thường) -> toàn bộ nội dung (nối các parent chunk cùng dieu_id,
    hiếm khi >1). Viết thường vì data/eval_testset.jsonl dùng quy ước dieu_id kiểu
    Neo4j (viết thường) — KHÁC với dieu_id gốc do VBPLChunker sinh ra (giữ nguyên
    hoa/thường từ tên file, xem comment tương tự trong chatbot_pipeline.py)."""
    chunker = VBPLChunker()
    result = chunker.process_directory_parent_child(data_dir)
    dieu_map = defaultdict(list)
    for p in result["parents"]:
        dieu_map[p["dieu_id"].lower()].append(p["content"])
    return {k: "\n".join(v) for k, v in dieu_map.items()}


def load_testset(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def build_source_text(row: dict, dieu_map: dict) -> str:
    parts = []
    for d in row.get("dieu_ids", []):
        parts.append(dieu_map.get(d.lower(), f"[KHÔNG TÌM THẤY VĂN BẢN GỐC CHO dieu_id={d}]"))
    return "\n\n---\n\n".join(parts)


_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")


def build_llm(model: str, api_key: str, max_retries: int) -> ChatOpenAI:
    if not api_key:
        raise RuntimeError("Cần OPENAI_API_KEY (truyền qua --openai-api-key hoặc biến môi trường OPENAI_API_KEY).")
    kwargs = dict(model=model, base_url=OPENAI_BASE_URL or None, api_key=api_key, max_retries=max_retries)
    # Model dòng o1/o3/o4 (reasoning) KHÔNG hỗ trợ temperature tùy chỉnh (chỉ nhận
    # giá trị mặc định) — truyền temperature=0.0 như các model chat thường sẽ lỗi API.
    if not model.startswith(_REASONING_MODEL_PREFIXES):
        kwargs["temperature"] = 0.0
    return ChatOpenAI(**kwargs)


def _extract_json_object(text: str) -> str:
    """Bóc đúng khối {...} đầu tiên trong text — model đôi khi thêm chữ dẫn/giải
    thích trước hoặc sau JSON dù đã yêu cầu 'chỉ trả lời JSON' (vd 'Here is the
    JSON:\\n{...}'), nên KHÔNG thể giả định cả chuỗi output là JSON thuần."""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def invoke_json(llm, prompt: str, parse_retries: int = 2) -> dict:
    """Gọi LLM, parse JSON — thử lại nếu output không phải JSON hợp lệ (lỗi mạng/rate
    limit đã do ChatOpenAI(max_retries=...) tự xử lý ở tầng dưới, đây chỉ retry lỗi
    format output của model)."""
    text = ""
    last_err = None
    for _ in range(parse_retries + 1):
        resp = llm.invoke(prompt)
        text = _extract_json_object(resp.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Không parse được JSON output sau nhiều lần thử: {last_err}. Output cuối: {text[:300]!r}")


# ---------------- A. Kiểm tra không cần LLM ----------------

def check_existence(rows: list, dieu_map: dict) -> list:
    missing = []
    for r in rows:
        for d in r.get("dieu_ids", []):
            if d.lower() not in dieu_map:
                missing.append({"id": r["id"], "missing_dieu_id": d})
    return missing


def check_schema(rows: list) -> list:
    violations = []
    for r in rows:
        n = len(r.get("dieu_ids", []))
        cat = r.get("category")
        if cat == "cross_reference" and n < 2:
            violations.append({"id": r["id"], "category": cat, "dieu_ids_count": n})
        elif cat != "cross_reference" and n != 1:
            violations.append({"id": r["id"], "category": cat, "dieu_ids_count": n})
    return violations


def check_duplicates(rows: list, threshold: float = 0.9) -> list:
    dupes = []
    items = [(r["id"], r["question"]) for r in rows]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ratio = difflib.SequenceMatcher(None, items[i][1], items[j][1]).ratio()
            if ratio >= threshold:
                dupes.append({"id_a": items[i][0], "id_b": items[j][0], "similarity": round(ratio, 3)})
    return dupes


# ---------------- B. LLM-as-judge ----------------

def _load_checkpoint(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(path)


def run_llm_judge(rows: list, dieu_map: dict, llm, checkpoint_path: Path = None, recheck_category: bool = False) -> dict:
    """
    recheck_category=True: dùng khi CHỈ sửa CATEGORY_PROMPT/CATEGORY_DEFINITIONS
    (vd sau khi phát hiện rubric chấm nhầm) — với câu ĐÃ có sẵn trong checkpoint
    (groundedness/quality đã chấm đúng, không cần chấm lại), chỉ gọi lại lệnh
    CATEGORY_PROMPT để cập nhật đúng "category", giữ nguyên 2 phần kia — tiết
    kiệm 2/3 chi phí so với xóa checkpoint chấm lại từ đầu.
    """
    per_row = _load_checkpoint(checkpoint_path)
    if per_row:
        print(f"  Đã nạp checkpoint: {len(per_row)}/{len(rows)} câu đã chấm từ lần chạy trước.")

    todo = [r for r in rows if r["id"] not in per_row or (recheck_category and not per_row[r["id"]].get("error"))]
    for i, row in enumerate(todo, 1):
        source_text = build_source_text(row, dieu_map)
        existing = per_row.get(row["id"])

        try:
            if recheck_category and existing and not existing.get("error"):
                category = invoke_json(llm, CATEGORY_PROMPT.format(
                    category=row["category"],
                    category_definition=CATEGORY_DEFINITIONS.get(row["category"], "(không có định nghĩa)"),
                    source_text=source_text, question=row["question"],
                ))
                per_row[row["id"]] = {**existing, "category": category}
            else:
                facts = row.get("required_facts", [])
                facts_numbered = "\n".join(f"{idx + 1}. {f}" for idx, f in enumerate(facts)) or "(không có fact nào)"
                groundedness = invoke_json(llm, GROUNDEDNESS_PROMPT.format(
                    source_text=source_text, facts_numbered=facts_numbered,
                    reference_answer=row["reference_answer"],
                ))
                quality = invoke_json(llm, QUESTION_QUALITY_PROMPT.format(
                    source_text=source_text, question=row["question"],
                ))
                category = invoke_json(llm, CATEGORY_PROMPT.format(
                    category=row["category"],
                    category_definition=CATEGORY_DEFINITIONS.get(row["category"], "(không có định nghĩa)"),
                    source_text=source_text, question=row["question"],
                ))
                per_row[row["id"]] = {"groundedness": groundedness, "quality": quality, "category": category, "error": None}
        except Exception as e:
            per_row[row["id"]] = {"error": str(e)}

        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, per_row)
        if i % 10 == 0 or i == len(todo):
            print(f"  LLM-judge: đã chấm {i}/{len(todo)} câu mới ({len(per_row)}/{len(rows)} tổng cộng)...")

    return per_row


# ---------------- Tổng hợp kết quả ----------------

def aggregate(rows: list, per_row: dict):
    by_cat = defaultdict(lambda: defaultdict(list))
    fails = defaultdict(list)
    errors = []

    for r in rows:
        res = per_row.get(r["id"])
        if not res or res.get("error"):
            errors.append({"id": r["id"], "error": res.get("error") if res else "không có kết quả"})
            continue

        g = res["groundedness"]
        q = res["quality"]
        c = res["category"]
        cat = r["category"]

        facts_grounded = g.get("facts_grounded", [])
        facts_covered = g.get("facts_covered_by_answer", [])
        fg_rate = (sum(facts_grounded) / len(facts_grounded)) if facts_grounded else None
        fc_rate = (sum(facts_covered) / len(facts_covered)) if facts_covered else None
        answer_grounded = bool(g.get("answer_grounded"))
        natural_clear = bool(q.get("natural_clear"))
        requires_citation = bool(q.get("requires_citation"))
        category_correct = bool(c.get("category_correct"))

        if fg_rate is not None:
            by_cat[cat]["facts_grounded_rate"].append(fg_rate)
        by_cat[cat]["answer_grounded"].append(1.0 if answer_grounded else 0.0)
        if fc_rate is not None:
            by_cat[cat]["facts_covered_rate"].append(fc_rate)
        by_cat[cat]["natural_clear"].append(1.0 if natural_clear else 0.0)
        by_cat[cat]["requires_citation"].append(1.0 if requires_citation else 0.0)
        by_cat[cat]["category_correct"].append(1.0 if category_correct else 0.0)

        if fg_rate is not None and fg_rate < 1.0:
            fails["facts_grounded"].append(r["id"])
        if not answer_grounded:
            fails["answer_grounded"].append(r["id"])
        if fc_rate is not None and fc_rate < 1.0:
            fails["facts_covered"].append(r["id"])
        if not natural_clear:
            fails["natural_clear"].append(r["id"])
        if not requires_citation:
            fails["requires_citation"].append(r["id"])
        if not category_correct:
            fails["category_correct"].append({"id": r["id"], "reason": c.get("reason", "")})

    return by_cat, fails, errors


def print_and_save_report(rows, dieu_map, per_row, existence_issues, schema_issues, dup_issues, output_suffix, judge_model="?"):
    print("\n" + "=" * 70)
    print("A. KIỂM TRA KHÔNG CẦN LLM (deterministic)")
    print("=" * 70)
    print(f"A1. existence  : {len(existence_issues)} dieu_id KHÔNG tồn tại trong corpus"
          f"{' — ' + str(existence_issues[:5]) if existence_issues else ' (0 lỗi, đạt 100%)'}")
    print(f"A2. schema     : {len(schema_issues)} câu sai schema (số dieu_ids không khớp category)"
          f"{' — ' + str(schema_issues[:5]) if schema_issues else ' (0 lỗi, đạt 100%)'}")
    print(f"A3. duplicate  : {len(dup_issues)} cặp câu hỏi trùng/gần trùng (similarity>=0.9)"
          f"{' — ' + str(dup_issues[:5]) if dup_issues else ' (0 cặp trùng)'}")

    by_cat, fails, errors = aggregate(rows, per_row)

    print("\n" + "=" * 70)
    print(f"B. LLM-AS-JUDGE ({judge_model}) — % đạt theo từng chỉ số")
    print("=" * 70)
    if errors:
        print(f"  CẢNH BÁO: {len(errors)} câu bị lỗi khi gọi LLM-judge (không tính vào tỷ lệ dưới đây): "
              f"{[e['id'] for e in errors][:10]}")

    metric_labels = {
        "facts_grounded_rate": "B1. facts_grounded          (fact đúng với văn bản gốc)",
        "answer_grounded": "B2. answer_grounded         (đáp án mẫu trung thực với gốc)",
        "facts_covered_rate": "B3. facts_covered_by_answer (đáp án mẫu nêu đủ facts)",
        "natural_clear": "B4. natural_clear           (câu hỏi tự nhiên, rõ ràng)",
        "requires_citation": "B5. requires_citation       (bắt buộc cần đúng Điều này)",
        "category_correct": "B6. category_correct        (nhãn category đúng)",
    }

    summary_rows = []
    all_metrics = list(metric_labels.keys())
    categories = sorted(by_cat.keys())

    for metric in all_metrics:
        print(f"\n{metric_labels[metric]}")
        overall_vals = []
        for cat in categories:
            vals = by_cat[cat].get(metric, [])
            overall_vals.extend(vals)
            avg = sum(vals) / len(vals) if vals else 0.0
            print(f"    {cat:<28}: {avg:.1%} (n={len(vals)})")
            summary_rows.append({"category": cat, "metric": metric, "value": avg, "n": len(vals)})
        overall_avg = sum(overall_vals) / len(overall_vals) if overall_vals else 0.0
        print(f"    {'TỔNG':<28}: {overall_avg:.1%} (n={len(overall_vals)})")
        summary_rows.append({"category": "ALL", "metric": metric, "value": overall_avg, "n": len(overall_vals)})

    print("\n" + "=" * 70)
    print("Danh sách câu CẦN XEM LẠI THỦ CÔNG (không đạt >=1 chỉ số)")
    print("=" * 70)
    for name, ids in fails.items():
        print(f"  {name}: {len(ids)} câu — {ids[:10]}{' ...' if len(ids) > 10 else ''}")

    # Lưu file chi tiết + tổng hợp
    summary_path = PROJECT_ROOT / "data" / f"testset_validation_summary{output_suffix}.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "metric", "value", "n"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nĐã lưu bảng tổng hợp: {summary_path}")

    detail_path = PROJECT_ROOT / "data" / f"testset_validation_details{output_suffix}.csv"
    with open(detail_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "category", "facts_grounded_rate", "answer_grounded",
            "facts_covered_rate", "natural_clear", "requires_citation",
            "category_correct", "category_reason", "error",
        ])
        for r in rows:
            res = per_row.get(r["id"], {})
            if res.get("error"):
                writer.writerow([r["id"], r["category"], "", "", "", "", "", "", "", res["error"]])
                continue
            g = res.get("groundedness", {})
            q = res.get("quality", {})
            c = res.get("category", {})
            fg = g.get("facts_grounded", [])
            fc = g.get("facts_covered_by_answer", [])
            writer.writerow([
                r["id"], r["category"],
                (sum(fg) / len(fg)) if fg else "",
                g.get("answer_grounded", ""),
                (sum(fc) / len(fc)) if fc else "",
                q.get("natural_clear", ""),
                q.get("requires_citation", ""),
                c.get("category_correct", ""),
                c.get("reason", ""),
                "",
            ])
    print(f"Đã lưu chi tiết từng câu: {detail_path}")

    existence_path = PROJECT_ROOT / "data" / f"testset_validation_issues{output_suffix}.json"
    with open(existence_path, "w", encoding="utf-8") as f:
        json.dump({
            "existence_issues": existence_issues,
            "schema_issues": schema_issues,
            "duplicate_issues": dup_issues,
            "llm_errors": errors,
            "fails": fails,
        }, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu chi tiết các lỗi/câu cần xem lại: {existence_path}")


def main():
    parser = argparse.ArgumentParser(description="Kiểm định độ tin cậy của data/eval_testset.jsonl")
    parser.add_argument("--testset", type=str, default=str(PROJECT_ROOT / "data" / "eval_testset.jsonl"))
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "keep"),
                         help="Thư mục chứa văn bản luật gốc (.docx) để đối chiếu groundedness")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ kiểm tra N câu đầu (test nhanh)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--max-retries", type=int, default=2, help="Số lần ChatOpenAI tự retry khi lỗi mạng/rate limit")
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--skip-llm-judge", action="store_true", help="Chỉ chạy phần A (không cần LLM), bỏ qua phần B")
    parser.add_argument("--recheck-category", action="store_true",
                         help="Chỉ chấm lại B6 category_correct cho câu ĐÃ có trong checkpoint (giữ nguyên groundedness/quality đã chấm) — dùng sau khi sửa CATEGORY_PROMPT/CATEGORY_DEFINITIONS, tiết kiệm 2/3 chi phí so với xóa checkpoint chấm lại từ đầu.")
    args = parser.parse_args()

    print("Đang parse corpus gốc (data/keep) để lấy văn bản đối chiếu...")
    dieu_map = load_corpus_dieu_map(args.data_dir)
    print(f"Đã parse {len(dieu_map)} Điều từ corpus gốc.")

    rows = load_testset(args.testset, args.limit)
    print(f"Đã load {len(rows)} câu từ {args.testset}")

    existence_issues = check_existence(rows, dieu_map)
    schema_issues = check_schema(rows)
    dup_issues = check_duplicates(rows)

    per_row = {}
    if not args.skip_llm_judge:
        llm = build_llm(args.judge_model, args.openai_api_key, args.max_retries)
        checkpoint_path = None
        if not args.no_checkpoint:
            checkpoint_path = PROJECT_ROOT / "data" / f"testset_validation_checkpoint{args.output_suffix}.json"
        per_row = run_llm_judge(rows, dieu_map, llm, checkpoint_path, recheck_category=args.recheck_category)

    print_and_save_report(rows, dieu_map, per_row, existence_issues, schema_issues, dup_issues, args.output_suffix, args.judge_model)


if __name__ == "__main__":
    main()
