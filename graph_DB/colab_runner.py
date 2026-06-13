"""
colab_runner.py
===============
Script xây dựng Legal Knowledge Graph chạy trên Google Colab.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÁCH 1 — Git clone (KHUYẾN NGHỊ, không cần upload gì)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chép đoạn này vào 1 cell Colab rồi chạy là xong:

    # ── Cài thư viện ──
    !pip install transformers accelerate bitsandbytes python-docx tqdm -q

    # ── Clone repo ──
    !git clone https://github.com/PhuIT2503/LEGAL_IT_CHATBOT.git /content/LEGAL_IT_CHATBOT

    # ── Setup path ──
    import sys
    sys.path.insert(0, '/content/LEGAL_IT_CHATBOT')
    sys.path.insert(0, '/content/LEGAL_IT_CHATBOT/graph_DB')

    # ── Chạy pipeline ──
    exec(open('/content/LEGAL_IT_CHATBOT/graph_DB/colab_runner.py').read())

Output sẽ lưu vào /content/LEGAL_IT_CHATBOT/graph_DB/
→ Sau đó tải về máy hoặc copy sang Drive.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÁCH 2 — Google Drive (nếu repo private hoặc có data riêng)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chỉ cần upload 2 thứ lên Drive:
    Drive/MyDrive/LEGAL_IT_CHATBOT/data/keep/      ← chỉ folder data
    Drive/MyDrive/LEGAL_IT_CHATBOT/graph_DB/       ← chỉ folder code

Rồi chạy trong Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/LEGAL_IT_CHATBOT')
    sys.path.insert(0, '/content/drive/MyDrive/LEGAL_IT_CHATBOT/graph_DB')
    # Đổi USE_DRIVE = True bên dưới
    exec(open('/content/drive/MyDrive/LEGAL_IT_CHATBOT/graph_DB/colab_runner.py').read())
"""

import os
import sys
import json
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("colab_runner")

# ══════════════════════════════════════════════════════════════
# ⚙️  CONFIGURATION — Chỉnh tại đây nếu cần
# ══════════════════════════════════════════════════════════════

# Chọn 1 trong 2 cách:
USE_DRIVE = False   # False = git clone (Cách 1) | True = Google Drive (Cách 2)

if USE_DRIVE:
    # Cách 2: Drive
    PROJECT_ROOT = "/content/drive/MyDrive/LEGAL_IT_CHATBOT"
else:
    # Cách 1: git clone (mặc định)
    PROJECT_ROOT = "/content/LEGAL_IT_CHATBOT"

# Đường dẫn cụ thể (thường không cần đổi)
DATA_DIR     = f"{PROJECT_ROOT}/data/keep"
OUTPUT_DIR   = f"{PROJECT_ROOT}/graph_DB/extracted_json"   # JSON trung gian
GRAPH_OUTPUT = f"{PROJECT_ROOT}/graph_DB/legal_knowledge_graph.json"
CYPHER_DIR   = f"{PROJECT_ROOT}/graph_DB/neo4j_import"

# LLM config
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
USE_4BIT = True    # True = 4-bit (T4 OK) | False = bf16 (cần A100)

# Test nhanh 1 file trước khi chạy toàn bộ
TEST_MODE = False
TEST_FILE = "Luật An ninh mạng 2025.docx"

# Tiếp tục từ chỗ dở nếu Colab bị ngắt
RESUME = True

# ══════════════════════════════════════════════════════════════
# SETUP PATHS
# ══════════════════════════════════════════════════════════════

for _p in [PROJECT_ROOT, f"{PROJECT_ROOT}/graph_DB"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CYPHER_DIR, exist_ok=True)

logger.info("=" * 58)
logger.info("  LEGAL KNOWLEDGE GRAPH BUILDER")
logger.info(f"  Project: {PROJECT_ROOT}")
logger.info(f"  Data   : {DATA_DIR}")
logger.info(f"  Mode   : {'Drive' if USE_DRIVE else 'Git clone'}")
logger.info("=" * 58)

t_start = time.time()

# ══════════════════════════════════════════════════════════════
# STEP 1 — PARSE .docx → PARENT CHUNKS
# ══════════════════════════════════════════════════════════════

logger.info("\n── STEP 1: Parse .docx → parent chunks ──")

from chunking import VBPLChunker
from graph_builder import get_van_ban_meta, get_van_ban_id

chunker = VBPLChunker()
all_doc_chunks = {}  # filename -> list[parent_chunk]

docx_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".docx"))

if TEST_MODE:
    docx_files = [TEST_FILE] if TEST_FILE in docx_files else docx_files[:1]
    logger.info(f"⚠ TEST MODE: [{', '.join(docx_files)}]")

for fn in docx_files:
    fp      = os.path.join(DATA_DIR, fn)
    parents = chunker.chunk_document_parent_child(fp)["parents"]
    all_doc_chunks[fn] = parents
    logger.info(f"  ✓ {fn[:52]:<52} {len(parents):>3} Điều")

total_dieu = sum(len(v) for v in all_doc_chunks.values())
logger.info(f"\n  → {len(all_doc_chunks)} văn bản | {total_dieu} Điều luật")

# ══════════════════════════════════════════════════════════════
# STEP 2 — LLM EXTRACTION
# ══════════════════════════════════════════════════════════════

logger.info("\n── STEP 2: LLM Extraction (Qwen2.5-7B-Instruct) ──")
logger.info(f"  4-bit quant: {USE_4BIT} | Resume: {RESUME}")

from entity_extractor import load_llm, LegalEntityExtractor

logger.info("  Loading model... (~3-5 phút lần đầu)")
model, tokenizer = load_llm(MODEL_ID, use_4bit=USE_4BIT)
logger.info("  Model loaded ✓")

extractor = LegalEntityExtractor(
    model=model,
    tokenizer=tokenizer,
    temperature=0.0,
    max_new_tokens=2048,
    max_retry=3,
    verbose=False,
)

all_results = []
done = 0

for fn, parent_chunks in all_doc_chunks.items():
    meta           = get_van_ban_meta(fn)
    van_ban_name   = meta["ten"] if meta else fn.replace(".docx", "")
    van_ban_id_str = get_van_ban_id(fn)
    doc_out_dir    = os.path.join(OUTPUT_DIR, van_ban_id_str) if RESUME else None

    results = extractor.extract_batch(
        chunks=parent_chunks,
        van_ban_name=van_ban_name,
        van_ban_id=van_ban_id_str,
        save_dir=doc_out_dir,
    )
    all_results.extend(results)
    done += len(parent_chunks)
    mins  = (time.time() - t_start) / 60
    logger.info(
        f"  ✓ {van_ban_name[:40]:<40} "
        f"{len(results):>3} results | "
        f"{done}/{total_dieu} Điều | "
        f"{mins:.1f} phút"
    )

# Giải phóng GPU
import torch
del model; torch.cuda.empty_cache()
logger.info(f"\n  GPU released | Tổng: {len(all_results)} extraction results")

# ══════════════════════════════════════════════════════════════
# STEP 3 — BUILD UNIFIED GRAPH
# ══════════════════════════════════════════════════════════════

logger.info("\n── STEP 3: Build Knowledge Graph ──")

from graph_builder import LegalGraphBuilder

builder = LegalGraphBuilder()

for fn in all_doc_chunks:
    meta = get_van_ban_meta(fn)
    if meta:
        builder.add_van_ban_node(
            van_ban_id=get_van_ban_id(fn),
            ten=meta["ten"], loai=meta["loai"],
            nam=meta.get("nam"), so_hieu=meta.get("so_hieu", ""),
        )

builder.process_all_results(all_results)
builder.resolve_cross_document_references()

s = builder.get_summary()
logger.info(f"  Nodes : {s['total_nodes']:,}  |  Edges : {s['total_edges']:,}")
logger.info("  Node types:")
for nt, cnt in sorted(s["node_types"].items(), key=lambda x: -x[1]):
    logger.info(f"    {nt:<20} {cnt:>5}")

# Lưu JSON
builder.save(GRAPH_OUTPUT)
logger.info(f"\n  ✅ Saved: {GRAPH_OUTPUT}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — EXPORT CYPHER SCRIPTS (sẵn sàng import Neo4j sau)
# ══════════════════════════════════════════════════════════════

logger.info("\n── STEP 4: Export Cypher scripts ──")

from cypher_export import export_cypher_scripts

export_cypher_scripts(graph_data=builder.to_dict(), output_dir=CYPHER_DIR)
logger.info(f"  ✅ Saved: {CYPHER_DIR}/")

# ══════════════════════════════════════════════════════════════
# XONG — Hướng dẫn tải về / import Neo4j
# ══════════════════════════════════════════════════════════════

total_mins = (time.time() - t_start) / 60
logger.info("\n" + "=" * 58)
logger.info(f"  🎉 HOÀN THÀNH!  Tổng thời gian: {total_mins:.1f} phút")
logger.info("=" * 58)
logger.info(f"\n  📄 Graph JSON    : {GRAPH_OUTPUT}")
logger.info(f"  📁 Cypher scripts: {CYPHER_DIR}/")
logger.info(f"  📁 JSON trung gian: {OUTPUT_DIR}/")

if not USE_DRIVE:
    logger.info("""
  ─── Tải kết quả về máy (chạy trong cell mới) ───
  from google.colab import files
  files.download('{graph}')

  # Hoặc copy sang Drive:
  from google.colab import drive
  drive.mount('/content/drive')
  !cp -r {graph_db} /content/drive/MyDrive/LEGAL_IT_CHATBOT_output/
""".format(
        graph=GRAPH_OUTPUT,
        graph_db=f"{PROJECT_ROOT}/graph_DB"
    ))

logger.info("""
  ─── Khi có Neo4j AuraDB ───────────────────────────
  from cypher_export import import_to_neo4j
  import_to_neo4j(
      graph_json_path = '{graph}',
      neo4j_uri       = 'neo4j+s://xxxx.databases.neo4j.io',
      neo4j_user      = 'neo4j',
      neo4j_password  = 'your_password',
  )
""".format(graph=GRAPH_OUTPUT))
