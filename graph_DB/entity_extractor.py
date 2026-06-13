"""
entity_extractor.py
===================
Gọi Qwen/Qwen2.5-7B-Instruct để trích xuất entities và relations
từ các chunks văn bản pháp luật IT.

Thiết kế cho Google Colab với GPU T4 (16GB VRAM).
"""

import json
import re
import time
import logging
from typing import Optional

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
    max_new_tokens: int = 2048,
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

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSER
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_from_output(raw_output: str) -> Optional[dict]:
    """
    Trích xuất JSON từ output LLM.
    LLM có thể thêm text giải thích trước/sau JSON block.
    
    Thứ tự ưu tiên:
    1. Tìm ```json ... ``` block
    2. Tìm { ... } block lớn nhất
    3. Parse trực tiếp
    
    Returns:
        dict hoặc None nếu parse thất bại
    """
    # Thử 1: tìm ```json ... ```
    json_block_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
    match = json_block_pattern.search(raw_output)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Thử 2: tìm ``` ... ``` (không có json label)
    code_block_pattern = re.compile(r'```\s*(.*?)\s*```', re.DOTALL)
    match = code_block_pattern.search(raw_output)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Thử 3: tìm JSON object { ... } lớn nhất
    # Tìm từ dấu { đầu tiên đến } cuối cùng
    start = raw_output.find('{')
    end = raw_output.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_output[start:end + 1])
        except json.JSONDecodeError:
            pass

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
        max_new_tokens: int = 2048,
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

        if self.verbose:
            logger.info(f"Extracting: {chunk_id}")

        messages = build_extraction_prompt(
            van_ban_name=van_ban_name,
            chunk_id=chunk_id,
            content=content,
            van_ban_id=van_ban_id,
            use_few_shot=use_few_shot,
        )

        for attempt in range(self.max_retry):
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
                        time.sleep(1)

            except Exception as e:
                logger.error(f"Error during extraction (attempt {attempt + 1}): {e}")
                if attempt < self.max_retry - 1:
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
