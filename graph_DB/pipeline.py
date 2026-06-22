"""
pipeline.py
===========
Entry point chạy toàn bộ pipeline xây dựng Legal Knowledge Graph.

CÁCH DÙNG TRÊN GOOGLE COLAB:
────────────────────────────
1. Mount Google Drive hoặc upload data/keep lên Colab
2. Cài đặt dependencies (xem phần INSTALLATION bên dưới)
3. Set các biến môi trường NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
4. Chạy: python pipeline.py --data-dir /path/to/data/keep

INSTALLATION (chạy trong Colab cell):
    !pip install transformers accelerate bitsandbytes neo4j python-docx torch
    !pip install huggingface_hub

THỨ TỰ PIPELINE:
    Step 1: Parse .docx → chunks (VBPLChunker)
    Step 2: LLM extract entities/relations từ mỗi Điều (parent chunk)
    Step 3: Build unified graph (normalize, deduplicate)
    Step 4: Ingest vào Neo4j AuraDB
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP (cho Colab)
# ─────────────────────────────────────────────────────────────────────────────

# Thêm thư mục gốc project vào sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def step1_parse_documents(data_dir: str) -> dict[str, list[dict]]:
    """
    Step 1: Parse tất cả .docx trong data_dir thành parent chunks.
    
    Returns:
        dict {filename -> list of parent_chunks}
    """
    logger.info(f"=== STEP 1: Parsing documents from {data_dir} ===")
    
    from chunking import VBPLChunker
    chunker = VBPLChunker()
    
    all_doc_chunks = {}
    docx_files = [f for f in os.listdir(data_dir) if f.endswith(".docx")]
    
    logger.info(f"Found {len(docx_files)} .docx files")
    
    for filename in docx_files:
        file_path = os.path.join(data_dir, filename)
        logger.info(f"Parsing: {filename}")
        
        result = chunker.chunk_document_parent_child(file_path)
        parents = result["parents"]
        
        logger.info(f"  → {len(parents)} parent chunks (Điều luật)")
        all_doc_chunks[filename] = parents
    
    total = sum(len(v) for v in all_doc_chunks.values())
    logger.info(f"Step 1 complete: {total} total parent chunks from {len(all_doc_chunks)} documents")
    return all_doc_chunks


def step2_extract_entities(
    all_doc_chunks: dict[str, list[dict]],
    model,
    tokenizer,
    output_dir: str,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    max_retry: int = 3,
    verbose: bool = True,
) -> list[dict]:
    """
    Step 2: Dùng LLM để extract entities/relations từ mỗi parent chunk.
    
    Returns:
        Danh sách tất cả extraction results
    """
    logger.info("=== STEP 2: LLM Entity Extraction ===")
    
    from entity_extractor import LegalEntityExtractor
    from graph_builder import get_van_ban_meta, get_van_ban_id
    
    extractor = LegalEntityExtractor(
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_retry=max_retry,
        verbose=verbose,
    )
    
    all_results = []
    
    for filename, parent_chunks in all_doc_chunks.items():
        # Lấy metadata văn bản
        meta = get_van_ban_meta(filename)
        if meta:
            van_ban_name = meta["ten"]
            van_ban_id_str = get_van_ban_id(filename)
        else:
            van_ban_name = filename.replace(".docx", "")
            van_ban_id_str = get_van_ban_id(filename)
            logger.warning(f"No metadata found for: {filename}")
        
        logger.info(f"\nProcessing: {van_ban_name} ({len(parent_chunks)} Điều luật)")
        
        # Tạo output dir cho từng văn bản
        doc_output_dir = os.path.join(output_dir, van_ban_id_str)
        os.makedirs(doc_output_dir, exist_ok=True)
        
        doc_results = extractor.extract_batch(
            chunks=parent_chunks,
            van_ban_name=van_ban_name,
            van_ban_id=van_ban_id_str,
            save_dir=doc_output_dir,
        )
        
        all_results.extend(doc_results)
        logger.info(f"  → Extracted {len(doc_results)} results")
    
    logger.info(f"Step 2 complete: {len(all_results)} total extraction results")
    return all_results


def step3_build_graph(
    all_results: list[dict],
    all_doc_chunks: dict[str, list[dict]],
    output_path: str,
) -> "LegalGraphBuilder":
    """
    Step 3: Tổng hợp extraction results thành unified graph.
    
    Returns:
        LegalGraphBuilder instance
    """
    logger.info("=== STEP 3: Building Knowledge Graph ===")
    
    from graph_builder import LegalGraphBuilder, get_van_ban_meta, get_van_ban_id
    
    builder = LegalGraphBuilder()
    
    # Thêm VanBan nodes từ registry
    for filename in all_doc_chunks.keys():
        meta = get_van_ban_meta(filename)
        if meta:
            van_ban_id_str = get_van_ban_id(filename)
            builder.add_van_ban_node(
                van_ban_id=van_ban_id_str,
                ten=meta["ten"],
                loai=meta["loai"],
                nam=meta.get("nam"),
                so_hieu=meta.get("so_hieu", ""),
            )
    
    # Process tất cả extraction results
    builder.process_all_results(all_results)
    
    # Resolve cross-document references
    builder.resolve_cross_document_references()
    
    # Summary
    summary = builder.get_summary()
    logger.info(f"Graph summary: {json.dumps(summary, ensure_ascii=False, indent=2)}")
    
    # Save graph
    builder.save(output_path)
    logger.info(f"Step 3 complete: Graph saved to {output_path}")
    
    return builder


def step4_ingest_neo4j(
    builder: "LegalGraphBuilder",
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_first: bool = False,
):
    """
    Step 4: Ingest graph vào Neo4j AuraDB.
    """
    logger.info("=== STEP 4: Ingesting to Neo4j AuraDB ===")
    
    from neo4j_ingest import Neo4jGraphIngestor
    
    with Neo4jGraphIngestor(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    ) as ingestor:
        # Kiểm tra kết nối
        if not ingestor.verify_connection():
            raise ConnectionError("Cannot connect to Neo4j. Check credentials.")
        
        # Ingest
        graph_data = builder.to_dict()
        ingestor.ingest_graph(graph_data, clear_first=clear_first)
        
        # Thống kê sau ingest
        stats = ingestor.get_stats()
        logger.info(f"Neo4j stats: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    
    logger.info("Step 4 complete: Graph ingested to Neo4j!")


# ─────────────────────────────────────────────────────────────────────────────
# RESUME FROM JSON (nếu LLM extraction đã chạy xong)
# ─────────────────────────────────────────────────────────────────────────────

def load_extracted_results_from_dir(extracted_json_dir: str) -> list[dict]:
    """
    Load tất cả JSON/JSONL extraction results từ thư mục (để resume).
    Hữu ích khi Step 2 đã chạy xong nhưng Step 3/4 cần chạy lại.
    """
    all_results = []
    seen_keys = set()

    def add_result(result: dict, fpath: str):
        if not isinstance(result, dict):
            logger.warning(f"Invalid extraction result in {fpath}: expected object")
            return

        key = (result.get("van_ban_id"), result.get("chunk_id"))
        if all(key):
            if key in seen_keys:
                return
            seen_keys.add(key)

        all_results.append(result)

    for root, dirs, files in os.walk(extracted_json_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.endswith(".jsonl"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line_no, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            try:
                                add_result(json.loads(line), f"{fpath}:{line_no}")
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse {fpath}:{line_no}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load {fpath}: {e}")
            elif fname.endswith(".json"):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    add_result(result, fpath)
                except Exception as e:
                    logger.warning(f"Failed to load {fpath}: {e}")
    
    logger.info(f"Loaded {len(all_results)} extraction results from {extracted_json_dir}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Legal Knowledge Graph Builder")
    
    parser.add_argument(
        "--data-dir",
        default="/content/LEGAL_IT_CHATBOT/data/keep",
        help="Thư mục chứa file .docx văn bản pháp luật"
    )
    parser.add_argument(
        "--output-dir",
        default="/content/LEGAL_IT_CHATBOT/graph_DB/extracted_json",
        help="Thư mục lưu JSON extraction (để debug/resume)"
    )
    parser.add_argument(
        "--graph-output",
        default="/content/LEGAL_IT_CHATBOT/graph_DB/legal_knowledge_graph.json",
        help="Đường dẫn lưu file JSON graph"
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", ""),
        help="Neo4j AuraDB URI"
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", ""),
        help="Neo4j password"
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Không dùng 4-bit quantization (cần nhiều VRAM hơn)"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Bỏ qua Step 2 (LLM extraction), load từ --output-dir"
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Bỏ qua Step 4 (Neo4j ingest)"
    )
    parser.add_argument(
        "--clear-neo4j",
        action="store_true",
        help="Xóa graph Neo4j cũ trước khi ingest (CẢNH BÁO!)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LEGAL KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 60)
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Model: {args.model_id}")
    
    # ── STEP 1: Parse documents ──
    all_doc_chunks = step1_parse_documents(args.data_dir)
    
    # ── STEP 2: LLM Extraction ──
    if args.skip_extraction:
        logger.info("Skipping Step 2 (--skip-extraction flag). Loading from disk...")
        all_results = load_extracted_results_from_dir(args.output_dir)
    else:
        from entity_extractor import load_llm
        logger.info(f"Loading LLM: {args.model_id}")
        model, tokenizer = load_llm(args.model_id, use_4bit=not args.no_4bit)
        
        all_results = step2_extract_entities(
            all_doc_chunks=all_doc_chunks,
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
        )
        
        # Giải phóng GPU memory sau extraction
        import torch
        del model
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared after extraction.")
    
    # ── STEP 3: Build Graph ──
    builder = step3_build_graph(
        all_results=all_results,
        all_doc_chunks=all_doc_chunks,
        output_path=args.graph_output,
    )
    
    # ── STEP 4: Ingest Neo4j ──
    if args.skip_neo4j:
        logger.info("Skipping Step 4 (--skip-neo4j flag).")
    elif not args.neo4j_uri or not args.neo4j_password:
        logger.warning(
            "NEO4J_URI or NEO4J_PASSWORD not set. "
            "Set environment variables or use --neo4j-uri/--neo4j-password flags."
        )
        logger.info("Graph saved to JSON. Run again with Neo4j credentials to ingest.")
    else:
        step4_ingest_neo4j(
            builder=builder,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            clear_first=args.clear_neo4j,
        )
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
