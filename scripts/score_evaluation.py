"""
scripts/score_evaluation.py
=============================
Tính điểm cho kết quả đã chạy (data/eval_results_<mode>.jsonl, xem
run_evaluation.py) — gồm 2 lớp chỉ số:

1. RAGAS chuẩn (nếu đã `pip install ragas`): faithfulness, answer_relevancy,
   context_precision, answer_correctness (cần `reference` — đã có sẵn trong
   test set). Bọc trong try/except: nếu chưa cài ragas hoặc lệch API version,
   script vẫn chạy tiếp phần chỉ số tùy biến bên dưới, không crash toàn bộ.

2. Legal Completeness Rate (tùy biến, LLM-as-judge) — chỉ số TRUNG TÂM của
   khóa luận: với mỗi fact trong required_facts của từng câu hỏi, hỏi LLM xem
   câu trả lời (response) có thể hiện đúng nội dung fact đó không. Tỷ lệ fact
   được thể hiện đúng = Legal Completeness Rate của câu đó.

Cả 2 lớp chỉ số dùng CHUNG đúng 1 LLM chấm (--judge-provider/--judge-model,
xem build_llm_for_provider()) — đo nhất quán, và tách biệt khỏi model SINH
câu trả lời (Ollama/qwen của cả 3 kịch bản) để tránh self-preference bias.

Output: in bảng tổng hợp theo mode x category, lưu CSV chi tiết từng câu tại
data/eval_scores_<mode>.csv và bảng tổng hợp tại data/eval_summary.csv.

Cách dùng:
    python scripts/score_evaluation.py --modes naive article_expand critic
    python scripts/score_evaluation.py --modes critic --skip-ragas   # chỉ tính Completeness Rate, bỏ qua RAGAS

    # Tự do chọn LLM chấm điểm (CẢ Completeness Rate VÀ RAGAS), KHÔNG hardcode:
    python scripts/score_evaluation.py --modes critic --judge-provider openai --judge-model gpt-4o-mini
        (cần set OPENAI_API_KEY trong môi trường — rẻ, ~vài đô cho vài trăm câu)
    python scripts/score_evaluation.py --modes critic --judge-provider gemini --judge-model gemini-1.5-flash
        (cần set GOOGLE_API_KEY trong môi trường)
    python scripts/score_evaluation.py --modes critic --judge-provider ollama
        (dùng lại Ollama local đang chạy sẵn — miễn phí, không cần API key,
        nhưng có rủi ro self-preference bias vì đây cũng là model sinh câu
        trả lời của cả 3 kịch bản)

    RAGAS mặc định chấm theo BATCH (10 câu/lần, xem --ragas-batch-size) và lưu
    checkpoint tại data/ragas_checkpoint_<mode><suffix>.json sau mỗi batch —
    nếu bị ngắt giữa chừng (mất mạng, hết quota...), chạy lại ĐÚNG lệnh cũ sẽ
    tự động chấm tiếp phần còn thiếu, không mất tiền/thời gian chấm lại từ đầu.
    Dùng --no-ragas-checkpoint để tắt, chấm 1 lần nguyên khối như bản cũ.

    Legal Completeness Rate cũng lưu checkpoint TỪNG CÂU tại
    data/completeness_checkpoint_<mode><suffix>.json ngay sau khi chấm xong —
    cùng cơ chế resume như RAGAS ở trên. Dùng --no-completeness-checkpoint để
    tắt.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.embeddings import Embeddings as _LangchainEmbeddingsBase  # noqa: E402


def build_llm_for_provider(provider: str, model_name: str = None):
    """
    Dựng 1 LLM DUY NHẤT (temperature=0) — dùng CHUNG cho cả 2 lớp chỉ số
    (Legal Completeness Rate VÀ RAGAS), để đo bằng đúng 1 model nhất quán,
    không lệch chuẩn giữa 2 lớp khi đưa số liệu vào khóa luận.

    Trước đây Completeness Rate luôn dùng cứng Ollama/qwen (model SINH câu trả
    lời của cả 3 kịch bản) để chấm — có rủi ro lý thuyết "self-preference
    bias" (model có xu hướng tự chấm câu trả lời của chính nó/model cùng họ
    cao hơn thực tế). Gộp về 1 provider tự chọn (mặc định openai/gpt-4o-mini,
    độc lập với model sinh câu trả lời) loại bỏ rủi ro này, và không tốn thêm
    đáng kể (mỗi lệnh gọi chấm 1 fact rất ngắn).

    temperature=0 (KHÁC với LLM sinh câu trả lời, temperature=0.2) vì đây là
    tác vụ PHÂN LOẠI yes/no đơn giản — quan sát thực tế: CÙNG 1 câu trả lời
    (chữ giống hệt nhau giữa 2 mode) nhưng judge chấm khác nhau giữa các lần
    chạy nếu để temperature>0 — nhiễu ngẫu nhiên của chính bước chấm, không
    phải khác biệt chất lượng thật.

      - "openai": ChatOpenAI thật, cần OPENAI_API_KEY trong môi trường.
      - "gemini": ChatGoogleGenerativeAI, cần GOOGLE_API_KEY trong môi trường.
      - "ollama": Ollama local đang chạy sẵn — miễn phí, không cần API key.
    """
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("--judge-provider openai cần biến môi trường OPENAI_API_KEY.")
        return ChatOpenAI(model=model_name or "gpt-4o-mini", temperature=0.0)
    elif provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("--judge-provider gemini cần biến môi trường GOOGLE_API_KEY.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name or "gemini-1.5-flash", temperature=0.0)
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        return ChatOpenAI(model=model, base_url=base_url, api_key="ollama", temperature=0.0)
    else:
        raise ValueError(f"--judge-provider không hợp lệ: {provider!r} (chọn openai|gemini|ollama)")


# Tên hiển thị RÕ RÀNG cho từng chỉ số RAGAS khi in ra console — key thật trả
# về từ ragas.evaluate() là tên ngắn (faithfulness, answer_relevancy, ...).
RAGAS_METRIC_LABELS = {
    "faithfulness": "Faithfulness (độ trung thực với ngữ cảnh)",
    "answer_relevancy": "Answer Relevancy (độ liên quan của câu trả lời)",
    "context_precision": "Context Precision (độ chính xác ngữ cảnh, theo RAGAS)",
    "answer_correctness": "Answer Correctness (độ đúng đắn câu trả lời)",
}


class _LocalSentenceTransformerEmbeddings(_LangchainEmbeddingsBase):
    """
    Wrapper Embeddings kiểu LangChain (embed_documents/embed_query) quanh
    SentenceTransformer cục bộ (model fine-tune VBPL) — dùng cho RAGAS thay vì
    bắt buộc phải có thêm 1 API key riêng cho embeddings. Dùng CHUNG cho mọi
    provider LLM (openai/gemini/ollama) vì embeddings và LLM chấm là 2 việc
    độc lập trong RAGAS.

    PHẢI kế thừa langchain_core.embeddings.Embeddings (không phải object
    thường) — RAGAS chấm bất đồng bộ (asyncio) và cần aembed_documents/
    aembed_query, mà class cha này tự cung cấp bản async mặc định (chạy
    embed_documents/embed_query đồng bộ qua thread executor). Thiếu bước kế
    thừa này thì mọi lệnh gọi async đều lỗi AttributeError, khiến các chỉ số
    phụ thuộc embedding (answer_relevancy, answer_correctness) bị NaN/thiếu
    dữ liệu ở một phần câu hỏi mà không crash toàn bộ — rất dễ bị bỏ sót.
    """

    def __init__(self, model):
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


def build_ragas_llm_and_embeddings(provider: str, model_name: str = None):
    """
    Dựng (llm, embeddings) cho RAGAS — dùng lại ĐÚNG build_llm_for_provider()
    (cùng 1 model với Completeness Rate, xem docstring hàm đó). Embeddings
    LUÔN dùng model fine-tune VBPL cục bộ (miễn phí, nhất quán với embeddings
    dùng cho retrieval) — không phụ thuộc provider LLM ở trên.
    """
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src.llm.embedding_model import load_embedding_model

    llm = build_llm_for_provider(provider, model_name)

    embed_model = load_embedding_model(
        os.getenv("RAGAS_EMBEDDING_MODEL", "data/ai_vietnamese_embedding_v2_finetuned_final")
    )
    embeddings = LangchainEmbeddingsWrapper(_LocalSentenceTransformerEmbeddings(embed_model))
    return LangchainLLMWrapper(llm), embeddings


def load_results(mode: str, suffix: str = ""):
    path = PROJECT_ROOT / "data" / f"eval_results_{mode}{suffix}.jsonl"
    if not path.exists():
        print(f"CẢNH BÁO: chưa có {path} — hãy chạy run_evaluation.py --mode {mode} trước.")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def judge_fact_covered(llm, response: str, fact: str) -> bool:
    """LLM-as-judge: fact có được thể hiện ĐÚNG trong response hay không (yes/no)."""
    prompt = (
        "Bạn là giám khảo chấm điểm câu trả lời pháp luật. Cho một CÂU TRẢ LỜI và một YÊU CẦU "
        "(1 fact bắt buộc phải có), hãy xác định xem CÂU TRẢ LỜI có thể hiện ĐÚNG nội dung của YÊU CẦU "
        "hay không — chấp nhận diễn đạt khác nhau miễn là ĐÚNG Ý và ĐÚNG SỐ LIỆU cụ thể (nếu yêu cầu có số "
        "liệu). Nếu câu trả lời thiếu hẳn ý đó, diễn đạt mơ hồ né tránh, hoặc nêu sai số liệu/nội dung thì "
        "tính là KHÔNG đạt.\n\n"
        f"YÊU CẦU (fact bắt buộc): {fact}\n\n"
        f"CÂU TRẢ LỜI CẦN CHẤM:\n{response}\n\n"
        "Chỉ trả lời đúng 1 từ: 'yes' nếu câu trả lời có thể hiện đúng fact này, 'no' nếu không."
    )
    resp = llm.invoke(prompt)
    return "yes" in resp.content.strip().lower()


def compute_completeness(llm, rows: list, checkpoint_path: Path = None) -> list:
    """Gắn thêm completeness_rate + facts_covered vào từng row (mutate + return).

    checkpoint_path: nếu có, lưu kết quả từng câu (theo id) ra file JSON ngay
    sau khi chấm xong câu đó — nếu bị ngắt giữa chừng (mất mạng, hết quota,
    Colab bị disconnect...), chạy lại đúng lệnh cũ sẽ chỉ chấm tiếp các câu
    CHƯA có trong checkpoint, không mất tiền/thời gian chấm lại từ đầu. Không
    truyền checkpoint_path -> hành vi cũ, không lưu tạm, mất là mất hết.
    """
    per_row = _load_json_checkpoint(checkpoint_path) if checkpoint_path else {}
    if per_row:
        print(f"  Đã nạp checkpoint Completeness Rate: {len(per_row)}/{len(rows)} câu đã chấm từ lần chạy trước.")

    todo_count = sum(1 for r in rows if r["id"] not in per_row)
    done_count = 0
    for row in rows:
        cached = per_row.get(row["id"])
        if cached is not None:
            row["completeness_rate"] = cached.get("completeness_rate")
            row["facts_covered"] = cached.get("facts_covered")
            continue

        facts = row.get("required_facts", [])
        if not facts:
            row["completeness_rate"] = None
            row["facts_covered"] = None
        else:
            covered = [judge_fact_covered(llm, row["response"], f) for f in facts]
            row["facts_covered"] = covered
            row["completeness_rate"] = sum(covered) / len(covered)

        per_row[row["id"]] = {"completeness_rate": row["completeness_rate"], "facts_covered": row["facts_covered"]}
        done_count += 1
        if checkpoint_path is not None:
            _save_json_checkpoint(checkpoint_path, per_row)
            if done_count % 20 == 0 or done_count == todo_count:
                print(f"  Completeness Rate: đã chấm {done_count}/{todo_count} câu mới ({len(per_row)}/{len(rows)} tổng cộng)...")
    return rows


def _load_json_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path, encoding="utf-8") as f:
        return json.load(f)


def _save_json_checkpoint(checkpoint_path: Path, data: dict) -> None:
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp_path.replace(checkpoint_path)


def try_compute_ragas(
    rows: list,
    ragas_provider: str,
    ragas_model: str = None,
    checkpoint_path: Path = None,
    batch_size: int = 10,
) -> dict:
    """Trả về dict {metric_name: avg_score} nếu ragas cài được và chạy được, ngược lại {}.

    ragas_provider: "openai" | "gemini" | "ollama" — xem
    build_ragas_llm_and_embeddings() và docstring đầu file để biết cách
    truyền API key cho từng provider.

    checkpoint_path: nếu có, chấm theo TỪNG BATCH (batch_size câu/lần) và lưu
    điểm từng câu (theo id) ra file JSON sau MỖI batch — nếu bị ngắt giữa
    chừng (mất mạng, hết quota, Colab bị disconnect...), chạy lại đúng lệnh
    cũ sẽ chỉ chấm tiếp phần CHƯA ĐỦ 4 chỉ số trong checkpoint, không mất
    tiền/thời gian chấm lại từ đầu. Không truyền checkpoint_path -> chấm 1
    lần nguyên khối như cũ (không lưu tạm, mất là mất hết).

    Một câu được coi "xong" chỉ khi checkpoint có ĐỦ CẢ 4 chỉ số — vì trong
    1 batch, evaluate() có thể chấm THÀNH CÔNG cho batch nhưng vẫn thiếu 1-2
    chỉ số ở MỘT VÀI câu do lỗi tạm thời ở đúng job đó (rate limit, timeout —
    xem comment ở batch_ok bên dưới). Nếu chỉ kiểm tra "câu đã có trong
    checkpoint" mà không kiểm tra đủ chỉ số, các câu thiếu 1 phần này sẽ bị
    bỏ sót vĩnh viễn, không bao giờ được chấm lại dù resume bao nhiêu lần.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, answer_correctness
    except ImportError as e:
        print(f"Bỏ qua RAGAS (chưa cài đặt hoặc thiếu dependency): {e}")
        print("Cài đặt: pip install ragas datasets")
        return {}

    try:
        llm, embeddings = build_ragas_llm_and_embeddings(ragas_provider, ragas_model)
    except Exception as e:
        print(f"Bỏ qua RAGAS (không dựng được LLM/embeddings cho provider={ragas_provider!r}): {e}")
        return {}

    metrics = [faithfulness, answer_relevancy, context_precision, answer_correctness]
    metric_names = [m.name for m in metrics]

    def _to_ragas_item(r):
        return {
            "question": r["user_input"],
            "answer": r["response"],
            "contexts": r["retrieved_contexts"] or [""],
            "ground_truth": r["reference"],
        }

    if checkpoint_path is None:
        # Hành vi CŨ — chấm 1 lần nguyên khối, không lưu tạm.
        ds = Dataset.from_list([_to_ragas_item(r) for r in rows])
        try:
            result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
            return {k: float(v) for k, v in result.items()}
        except Exception as e:
            print(f"RAGAS evaluate() lỗi (có thể do khác version API) — báo cáo lỗi để tự điều chỉnh: {e}")
            return {}

    per_row_scores = _load_json_checkpoint(checkpoint_path)
    if per_row_scores:
        complete = sum(1 for s in per_row_scores.values() if len(s) >= len(metric_names))
        print(f"  Đã nạp checkpoint RAGAS: {len(per_row_scores)}/{len(rows)} câu có trong checkpoint "
              f"({complete} câu đủ cả {len(metric_names)} chỉ số, {len(per_row_scores) - complete} câu thiếu "
              f"một phần do lỗi tạm thời trước đó — sẽ chấm lại các câu thiếu này).")

    # Coi 1 câu là "xong" chỉ khi ĐỦ cả 4 chỉ số — nếu batch trước bị rate limit/
    # timeout giữa chừng, checkpoint có thể lưu câu đó với chỉ 1-3/4 chỉ số (xem
    # comment ở dưới). Nếu chỉ kiểm tra "đã có trong checkpoint" (không kiểm tra đủ
    # chỉ số), các câu thiếu 1 phần này sẽ bị coi là xong VĨNH VIỄN, không bao giờ
    # được chấm lại dù chạy lại bao nhiêu lần.
    todo_rows = [r for r in rows if len(per_row_scores.get(r["id"], {})) < len(metric_names)]
    consecutive_bad_batches = 0
    for i in range(0, len(todo_rows), batch_size):
        batch = todo_rows[i : i + batch_size]
        ds = Dataset.from_list([_to_ragas_item(r) for r in batch])
        try:
            result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeddings)
            df = result.to_pandas()
        except Exception as e:
            print(f"  RAGAS lỗi ở batch {i}-{i + len(batch)}: {e}")
            print(f"  Đã lưu checkpoint tới {len(per_row_scores)}/{len(rows)} câu — chạy lại lệnh cũ để tiếp tục.")
            break

        # LƯU Ý: evaluate() KHÔNG raise exception khi từng job con lỗi (RateLimitError,
        # TimeoutError...) — ragas tự bắt lỗi ở mức job, chỉ in "Exception raised in
        # Job[N]" và trả về NaN cho đúng ô đó, evaluate() vẫn coi là "thành công". Vì
        # vậy try/except phía trên KHÔNG bắt được tình trạng hết quota API (vd RPD của
        # OpenAI cạn) — phải tự đếm tỷ lệ NaN mỗi batch để phát hiện và DỪNG kịp thời,
        # nếu không sẽ chạy hết batch còn lại (có thể hàng giờ) mà toàn ra dữ liệu rỗng.
        batch_ok = 0
        for row, (_, score_row) in zip(batch, df.iterrows()):
            scores = {}
            for name in metric_names:
                val = score_row.get(name)
                if val is not None and val == val:  # loại NaN (NaN != NaN)
                    scores[name] = float(val)
                    batch_ok += 1
            per_row_scores[row["id"]] = scores
        _save_json_checkpoint(checkpoint_path, per_row_scores)

        batch_total = len(batch) * len(metric_names)
        success_rate = batch_ok / batch_total if batch_total else 1.0
        print(f"  RAGAS: đã chấm {min(i + batch_size, len(todo_rows))}/{len(todo_rows)} câu mới "
              f"({len(per_row_scores)}/{len(rows)} tổng cộng, batch này thành công {success_rate:.0%})...")

        if success_rate < 0.5:
            consecutive_bad_batches += 1
        else:
            consecutive_bad_batches = 0

        if consecutive_bad_batches >= 2:
            print(f"  DỪNG: 2 batch liên tiếp thất bại phần lớn (khả năng hết quota/rate limit API, xem "
                  f"'Exception raised in Job[...]' ở log phía trên). Đã lưu checkpoint tới "
                  f"{len(per_row_scores)}/{len(rows)} câu — chạy lại ĐÚNG lệnh cũ sau khi hết bị rate limit "
                  f"để tự động chấm tiếp phần còn thiếu, không mất tiền/thời gian chấm lại.")
            break

    if not per_row_scores:
        return {}
    agg = {}
    for name in metric_names:
        vals = [per_row_scores[r["id"]][name] for r in rows if name in per_row_scores.get(r["id"], {})]
        if vals:
            agg[name] = sum(vals) / len(vals)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Tính điểm đánh giá (RAGAS + Legal Completeness Rate tùy biến)")
    parser.add_argument("--modes", nargs="+", default=["naive", "article_expand", "critic"])
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--suffix", type=str, default="", help="Hậu tố file input/output (vd '_stratified10'), phải khớp với --output-suffix đã dùng ở run_evaluation.py")
    parser.add_argument("--judge-provider", choices=["openai", "gemini", "ollama"], default="openai",
                         help="LLM DUY NHẤT dùng để chấm CẢ Completeness Rate VÀ RAGAS — xem docstring build_llm_for_provider() để biết API key cần set cho từng provider")
    parser.add_argument("--judge-model", type=str, default=None,
                         help="Tên model cụ thể cho --judge-provider (vd gpt-4o-mini, gemini-1.5-flash, qwen2.5:7b). Bỏ trống dùng mặc định của provider.")
    parser.add_argument("--ragas-batch-size", type=int, default=10,
                         help="Số câu chấm RAGAS mỗi batch trước khi lưu checkpoint (mặc định 10) — batch nhỏ hơn = lưu thường xuyên hơn, an toàn hơn nếu hay bị ngắt, nhưng chậm hơn 1 chút.")
    parser.add_argument("--no-ragas-checkpoint", action="store_true",
                         help="Tắt checkpoint, chấm RAGAS 1 lần nguyên khối như bản cũ (mất là mất hết nếu bị ngắt giữa chừng).")
    parser.add_argument("--no-completeness-checkpoint", action="store_true",
                         help="Tắt checkpoint, chấm Completeness Rate lại từ đầu mỗi lần (mất là mất hết nếu bị ngắt giữa chừng).")
    args = parser.parse_args()

    llm = build_llm_for_provider(args.judge_provider, args.judge_model)

    summary_rows = []
    for mode in args.modes:
        rows = load_results(mode, args.suffix)
        if not rows:
            continue

        print(f"\n=== Chấm điểm mode={mode} ({len(rows)} câu) ===")
        completeness_ckpt_path = None
        if not args.no_completeness_checkpoint:
            completeness_ckpt_path = PROJECT_ROOT / "data" / f"completeness_checkpoint_{mode}{args.suffix}.json"
        rows = compute_completeness(llm, rows, completeness_ckpt_path)

        ragas_scores = {}
        if not args.skip_ragas:
            ckpt_path = None
            if not args.no_ragas_checkpoint:
                ckpt_path = PROJECT_ROOT / "data" / f"ragas_checkpoint_{mode}{args.suffix}.json"
            ragas_scores = try_compute_ragas(rows, args.judge_provider, args.judge_model, ckpt_path, args.ragas_batch_size)

        # Lưu chi tiết từng câu
        detail_path = PROJECT_ROOT / "data" / f"eval_scores_{mode}{args.suffix}.csv"
        with open(detail_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "category", "completeness_rate", "response_preview"])
            for r in rows:
                writer.writerow([r["id"], r["category"], r.get("completeness_rate"), r["response"][:200].replace("\n", " ")])
        print(f"Đã lưu chi tiết: {detail_path}")

        # Tổng hợp theo category
        by_cat = defaultdict(list)
        for r in rows:
            if r.get("completeness_rate") is not None:
                by_cat[r["category"]].append(r["completeness_rate"])

        print(f"\n--- Legal Completeness Rate theo nhóm (mode={mode}) ---")
        for cat, vals in by_cat.items():
            avg = sum(vals) / len(vals) if vals else 0
            print(f"  {cat}: {avg:.2%} (n={len(vals)})")
            summary_rows.append({"mode": mode, "category": cat, "metric": "completeness_rate", "value": avg, "n": len(vals)})

        all_vals = [r["completeness_rate"] for r in rows if r.get("completeness_rate") is not None]
        overall = sum(all_vals) / len(all_vals) if all_vals else 0
        print(f"  TỔNG: {overall:.2%} (n={len(all_vals)})")
        summary_rows.append({"mode": mode, "category": "ALL", "metric": "completeness_rate", "value": overall, "n": len(all_vals)})

        # Chi phí token (prompt+completion cộng dồn qua MỌI lệnh gọi LLM trong 1 câu
        # hỏi — xem token_usage ghi bởi run_evaluation.py) — so sánh chi phí thực tế
        # giữa 3 kịch bản, không chỉ completeness_rate.
        token_totals = [r["token_usage"]["total_tokens"] for r in rows if r.get("token_usage")]
        prompt_totals = [r["token_usage"]["prompt_tokens"] for r in rows if r.get("token_usage")]
        completion_totals = [r["token_usage"]["completion_tokens"] for r in rows if r.get("token_usage")]
        call_counts = [r["token_usage"]["call_count"] for r in rows if r.get("token_usage")]
        if token_totals:
            avg_total = sum(token_totals) / len(token_totals)
            avg_prompt = sum(prompt_totals) / len(prompt_totals)
            avg_completion = sum(completion_totals) / len(completion_totals)
            avg_calls = sum(call_counts) / len(call_counts)
            print(f"  Token TB/câu: total={avg_total:.0f} (prompt={avg_prompt:.0f}, completion={avg_completion:.0f}), "
                  f"số lệnh gọi LLM TB={avg_calls:.1f} (n={len(token_totals)})")
            summary_rows.append({"mode": mode, "category": "ALL", "metric": "avg_total_tokens", "value": avg_total, "n": len(token_totals)})
            summary_rows.append({"mode": mode, "category": "ALL", "metric": "avg_prompt_tokens", "value": avg_prompt, "n": len(token_totals)})
            summary_rows.append({"mode": mode, "category": "ALL", "metric": "avg_completion_tokens", "value": avg_completion, "n": len(token_totals)})
            summary_rows.append({"mode": mode, "category": "ALL", "metric": "avg_llm_calls", "value": avg_calls, "n": len(token_totals)})

        # Token của ĐÚNG lệnh gọi sinh câu trả lời cuối cùng (không cộng router/gate/
        # draft-bị-bỏ) — chỉ số đúng cho "hiệu quả ngữ cảnh" khi so sánh 3 kịch bản,
        # tách biệt khỏi avg_total_tokens (tổng chi phí cả pipeline) ở trên.
        final_totals = [r["final_answer_token_usage"]["total_tokens"] for r in rows if r.get("final_answer_token_usage")]
        final_prompt = [r["final_answer_token_usage"]["prompt_tokens"] for r in rows if r.get("final_answer_token_usage")]
        final_completion = [r["final_answer_token_usage"]["completion_tokens"] for r in rows if r.get("final_answer_token_usage")]
        if final_totals:
            avg_final_total = sum(final_totals) / len(final_totals)
            avg_final_prompt = sum(final_prompt) / len(final_prompt)
            avg_final_completion = sum(final_completion) / len(final_completion)
            print(f"  Token TB/câu (CHỈ lệnh gọi sinh câu trả lời cuối): total={avg_final_total:.0f} "
                  f"(prompt={avg_final_prompt:.0f}, completion={avg_final_completion:.0f}) (n={len(final_totals)})")
            summary_rows.append({"mode": mode, "category": "ALL", "metric": "avg_final_answer_tokens", "value": avg_final_total, "n": len(final_totals)})
            summary_rows.append({"mode": mode, "category": "ALL", "metric": "avg_final_answer_prompt_tokens", "value": avg_final_prompt, "n": len(final_totals)})
            summary_rows.append({"mode": mode, "category": "ALL", "metric": "avg_final_answer_completion_tokens", "value": avg_final_completion, "n": len(final_totals)})

        if ragas_scores:
            print(f"\n--- RAGAS (mode={mode}) ---")
            for k, v in ragas_scores.items():
                label = RAGAS_METRIC_LABELS.get(k, k)
                print(f"  {label}: {v:.3f}")
                summary_rows.append({"mode": mode, "category": "ALL", "metric": f"ragas_{k}", "value": v, "n": len(rows)})

    summary_path = PROJECT_ROOT / "data" / f"eval_summary{args.suffix}.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "category", "metric", "value", "n"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nĐã lưu bảng tổng hợp: {summary_path}")


if __name__ == "__main__":
    main()
