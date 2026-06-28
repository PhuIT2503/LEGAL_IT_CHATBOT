"""
config.py
=========
Cấu hình cho pipeline xây dựng Knowledge Graph pháp luật IT.
Chạy trên Google Colab với Neo4j AuraDB và Qwen/Qwen2.5-7B-Instruct.
"""

import os

# ─────────────────────────────────────────────
# Neo4j AuraDB
# ─────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j+s://REPLACE_ME.databases.neo4j.io")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "REPLACE_ME")

# ─────────────────────────────────────────────
# LLM (Qwen2.5-7B-Instruct)
# ─────────────────────────────────────────────
# Tên model trên HuggingFace Hub
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# 4-bit quantization để chạy trên Colab T4 (GPU 16GB)
USE_4BIT_QUANT = True

# Nhiệt độ sampling – 0.0 để output JSON ổn định
LLM_TEMPERATURE = 0.0

# Số token tối đa LLM sinh ra
LLM_MAX_NEW_TOKENS = 2048

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
# Thư mục chứa file .docx đã được giữ lại
DATA_DIR = "/content/LEGAL_IT_CHATBOT/data/keep"  # Đường dẫn trên Colab

# Thư mục lưu kết quả extract (JSON) để debug / retry
EXTRACT_OUTPUT_DIR = "/content/LEGAL_IT_CHATBOT/graph_DB/extracted_json"

# Số Điều xử lý trong 1 batch (để tránh OOM)
DIEU_BATCH_SIZE = 5

# Retry nếu LLM trả về JSON lỗi
MAX_RETRY = 3

# Ghi log chi tiết
VERBOSE = True
