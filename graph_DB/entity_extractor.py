"""
entity_extractor.py
===================
Gọi Qwen/Qwen2.5-7B-Instruct để trích xuất entities và relations
từ các chunks văn bản pháp luật IT.

Thiết kế cho Google Colab với GPU T4 (16GB VRAM).
"""

import os
import gc
import json
import re
import time
import logging
from typing import Optional

# Giảm memory fragmentation cho CUDA allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_llm(model_id: str, use_4bit: bool = True):
    """
    Load Qwen2.5-7B-Instruct với 4-bit quantization (BitsAndBytes).
    Chạy trên Colab T4 (16GB).
    
    Returns:
        (model, tokenizer) tuple
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    messages: list[dict],
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
) -> str:
    """
    Chạy inference với Qwen2.5-7B-Instruct.
    Sử dụng apply_chat_template để format messages.
    
    Returns:
        Raw text output từ LLM
    """
    import torch

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Chỉ lấy phần generated (bỏ phần input)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Giải phóng bộ nhớ GPU ngay sau inference
    del model_inputs, generated_ids
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _repair_truncated_json(text: str) -> Optional[dict]:
    """
    Sửa JSON bị cắt giữa chừng do max_new_tokens hết.

    Vấn đề: text có thể bị cắt ngay GIỮA một string (ví dụ: `{"id": "D4_K...`)
    khiến việc đếm bracket trực tiếp từ đầu đến cuối bị sai (vì in_string=True).

    Chiến lược đúng:
    1. Scan từng ký tự, track depth và in_string state
    2. Ghi lại "last_safe_pos" = vị trí SAU KHI đóng một object ở depth=1
       (tức là vừa hoàn thành 1 entry trong array)
    3. Cắt tại last_safe_pos, bỏ phần dở dang sau đó
    4. Đóng các bracket còn mở (array ] và root object })
    """
    if not text or not text.strip().startswith('{'):
        return None

    # Bước 1: Tìm last_safe_pos
    depth = 0
    in_string = False
    escape_next = False
    last_safe_pos = -1  # vị trí sau khi đóng element ở depth=1

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch in ('{', '['):
            depth += 1
        elif ch in ('}', ']'):
            depth -= 1
            # depth=1 nghĩa là vừa đóng xong 1 element trong array/object con
            if depth == 1:
                last_safe_pos = i + 1

    # Nếu JSON hoàn chỉnh (depth=0), thử parse trực tiếp
    if depth == 0 and last_safe_pos != -1:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass  # Có lỗi syntax khác, tiếp tục với repair

    # Nếu không tìm được điểm cắt an toàn → từ bỏ
    if last_safe_pos == -1:
        return None

    # Bước 2: Cắt tại điểm an toàn, bỏ phần dở dang
    truncated = text[:last_safe_pos].rstrip().rstrip(',').rstrip()

    # Bước 3: Đếm lại bracket trong phần đã cắt sạch (không còn unclosed string)
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in truncated:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # Bước 4: Đóng bracket còn mở
    closing = ']' * max(0, open_brackets) + '}' * max(0, open_braces)
    repaired = truncated + closing

    try:
        result = json.loads(repaired)
        if isinstance(result, dict) and ('entities' in result or 'relations' in result):
            result.setdefault('entities', [])
            result.setdefault('relations', [])
            n_e = len(result['entities'])
            n_r = len(result['relations'])
            logger.info(
                f"Repaired truncated JSON → {n_e} entities, "
                f"{n_r} relations (partial, cut at safe boundary)"
            )
            return result
    except json.JSONDecodeError:
        pass

    return None



def extract_json_from_output(raw_output: str) -> Optional[dict]:
    """
    Trích xuất JSON từ output LLM.
    Hỗ trợ: json trong code block, raw JSON, và JSON bị truncate.

    Returns:
        dict hoặc None nếu parse thất bại
    """
    # Thử 1: tìm ```json ... ``` block (kể cả bị cắt → không dùng regex greedy)
    code_start = raw_output.find('```json')
    if code_start != -1:
        inner = raw_output[code_start + 7:]
        code_end = inner.find('```')
        candidate = inner[:code_end].strip() if code_end != -1 else inner.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(candidate)
            if repaired:
                return repaired

    # Thử 2: tìm ``` ... ``` block thông thường
    if '```' in raw_output:
        parts = raw_output.split('```')
        for part in parts:
            part = part.strip()
            if part.startswith('{'):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    repaired = _repair_truncated_json(part)
                    if repaired:
                        return repaired

    # Thử 3: tìm JSON object từ { đầu tiên
    start = raw_output.find('{')
    if start != -1:
        candidate = raw_output[start:]
        # Thử parse với } cuối cùng
        end = candidate.rfind('}')
        if end != -1:
            try:
                return json.loads(candidate[:end + 1])
            except json.JSONDecodeError:
                pass
        # JSON bị truncate → thử repair
        repaired = _repair_truncated_json(candidate)
        if repaired:
            return repaired

    logger.warning(f"Failed to parse JSON from output: {raw_output[:200]}...")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class LegalEntityExtractor:
    """
    Extractor sử dụng Qwen2.5-7B-Instruct để trích xuất entities và relations
    từ các chunks văn bản pháp luật.
    """

    def __init__(
        self,
        model,
        tokenizer,
        temperature: float = 0.0,
        max_new_tokens: int = 1536,
        max_retry: int = 3,
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retry = max_retry
        self.verbose = verbose

    def extract_from_chunk(
        self,
        van_ban_name: str,
        van_ban_id: str,
        chunk_id: str,
        content: str,
        use_few_shot: bool = True,
    ) -> Optional[dict]:
        """
        Trích xuất entities và relations từ một chunk.
        
        Args:
            van_ban_name: Tên văn bản (ví dụ: "Luật An ninh mạng 2025")
            van_ban_id: ID văn bản trong graph (snake_case)
            chunk_id: ID chunk theo chuẩn VBPLChunker
            content: Nội dung chunk
            use_few_shot: Có dùng few-shot examples không
            
        Returns:
            dict {"entities": [...], "relations": [...]} hoặc None
        """
        from prompts import build_extraction_prompt
        import torch

        if self.verbose:
            logger.info(f"Extracting: {chunk_id}")

        for attempt in range(self.max_retry):
            # Lần đầu dùng few-shot, lần sau fallback sang no-shot để tiết kiệm VRAM
            attempt_few_shot = use_few_shot and (attempt == 0)

            # Rebuild messages cho mỗi attempt (có thể đổi few-shot)
            messages = build_extraction_prompt(
                van_ban_name=van_ban_name,
                chunk_id=chunk_id,
                content=content,
                van_ban_id=van_ban_id,
                use_few_shot=attempt_few_shot,
            )

            try:
                raw_output = run_inference(
                    self.model,
                    self.tokenizer,
                    messages,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                )

                result = extract_json_from_output(raw_output)

                if result and "entities" in result and "relations" in result:
                    # Thêm metadata vào các entity
                    for entity in result["entities"]:
                        entity.setdefault("van_ban_id", van_ban_id)
                        entity.setdefault("source_chunk_id", chunk_id)
                    return result
                else:
                    logger.warning(f"Invalid JSON structure (attempt {attempt + 1}): {chunk_id}")
                    if attempt < self.max_retry - 1:
                        gc.collect()
                        torch.cuda.empty_cache()
                        time.sleep(1)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM (attempt {attempt + 1}), clearing cache and retrying without few-shot: {chunk_id}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(2)
                else:
                    logger.error(f"Error during extraction (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retry - 1:
                        time.sleep(2)
            except Exception as e:
                logger.error(f"Error during extraction (attempt {attempt + 1}): {e}")
                if attempt < self.max_retry - 1:
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(2)

        logger.error(f"All {self.max_retry} attempts failed for chunk: {chunk_id}")
        return None

    def extract_cross_references(
        self,
        chunk_id: str,
        content: str,
    ) -> Optional[dict]:
        """
        Phát hiện tham chiếu chéo giữa các điều khoản.
        """
        from prompts import build_cross_ref_prompt

        messages = build_cross_ref_prompt(chunk_id=chunk_id, content=content)

        try:
            raw_output = run_inference(
                self.model,
                self.tokenizer,
                messages,
                temperature=0.0,
                max_new_tokens=512,
            )
            return extract_json_from_output(raw_output)
        except Exception as e:
            logger.error(f"Cross-ref extraction failed for {chunk_id}: {e}")
            return None

    def extract_batch(
        self,
        chunks: list[dict],
        van_ban_name: str,
        van_ban_id: str,
        save_dir: Optional[str] = None,
    ) -> list[dict]:
        """
        Trích xuất từ danh sách chunks của một văn bản.
        
        Args:
            chunks: Danh sách chunks từ VBPLChunker (parent chunks)
            van_ban_name: Tên văn bản
            van_ban_id: ID văn bản
            save_dir: Thư mục lưu JSON trung gian (để debug / resume)
            
        Returns:
            Danh sách kết quả extraction
        """
        import os

        results = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            content = chunk.get("content", "")

            if not content.strip():
                continue

            # Kiểm tra file đã extract chưa (để resume)
            if save_dir:
                save_path = os.path.join(save_dir, f"{chunk_id}.json")
                if os.path.exists(save_path):
                    logger.info(f"[SKIP] Already extracted: {chunk_id}")
                    with open(save_path, "r", encoding="utf-8") as f:
                        results.append(json.load(f))
                    continue

            result = self.extract_from_chunk(
                van_ban_name=van_ban_name,
                van_ban_id=van_ban_id,
                chunk_id=chunk_id,
                content=content,
            )

            if result:
                result["chunk_id"] = chunk_id
                result["van_ban_id"] = van_ban_id
                results.append(result)

                # Lưu JSON trung gian
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{chunk_id}.json")
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved: {save_path}")
            else:
                logger.warning(f"Extraction failed for chunk: {chunk_id}")

            if self.verbose:
                logger.info(f"Progress: {i + 1}/{len(chunks)} chunks")

        return results
