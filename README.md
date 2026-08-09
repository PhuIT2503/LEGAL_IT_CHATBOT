# LEGAL IT CHATBOT

Hệ thống RAG tra cứu và phân tích văn bản pháp luật công nghệ thông tin Việt
Nam (sở hữu trí tuệ, an toàn thông tin, dữ liệu cá nhân, giao dịch điện tử,
viễn thông, công nghiệp công nghệ số).

Điểm khác biệt so với RAG thông thường: hệ thống có **Critic Agent** dùng
Knowledge Graph (Neo4j) để tự kiểm tra xem câu trả lời nháp đã đủ căn cứ pháp
lý chưa — thiếu Điều tham chiếu, thiếu chế tài bổ sung, hay thiếu Khoản của
chính Điều đang dẫn — rồi bổ sung ngữ cảnh và viết lại.

---

## 1. Chạy nhanh bằng Docker

```bash
cp .env.example .env          # điền OPENAI_API_KEY (model mặc định gpt-4o-mini qua api.shopaikey)

# TÙY CHỌN nhưng NÊN LÀM — chép 2 thư mục Qdrant đã ingest sẵn vào data/
# (xin từ người bàn giao; chúng nằm trong .gitignore nên không đi theo repo):
#     data/.qdrant_base/
#     data/.qdrant_gte_base/
# Có 2 thư mục này thì bước nạp vector chỉ mất vài phút thay vì 1-3 tiếng.

docker compose --profile app up -d app     # 1 lệnh duy nhất
```

Lệnh đó tự dựng đủ chuỗi phụ thuộc, đúng thứ tự:

```
neo4j (chờ healthy) ──► kg-ingest ─────────────┐
                        nạp Knowledge Graph     │
                        (3 script, vài phút)    ├──► app
qdrant (chờ healthy) ─► qdrant-ingest ─────────┘    Chainlit :8000
                        nạp 4 collection vector
```

`qdrant-ingest` **tự chọn 1 trong 2 đường**, không phải bấm gì:

| Điều kiện | Đường đi | Thời gian |
|---|---|---|
| Có `data/.qdrant_base` **và** `data/.qdrant_gte_base` | Copy thẳng vector + payload sang server, không chạy embedding model | **vài phút** |
| Không có | Chunking `data/keep` → encode dense + BM25 → upsert, lần lượt 2 model | **1–3 tiếng** |

Đường thứ hai là lý do repo vẫn tự đứng được một mình: chỉ cần clone là dựng
lại toàn bộ vector DB từ số 0, không phụ thuộc file nào bên ngoài.

Xem tiến độ: `docker compose logs -f qdrant-ingest`.

> Cả 2 đường đều **idempotent**. Đường nhanh bỏ qua collection đã có trên
> server; đường chậm dùng `--resume`, đối chiếu ID từng batch nên bị ngắt giữa
> chừng thì chạy lại là đi tiếp. Nhờ vậy mỗi lần `up` sau chỉ tốn khoảng một
> phút, và lỡ xóa mất volume `legal_qdrant_data` thì cứ chạy lại đúng lệnh cũ
> là nó nạp lại đúng phần thiếu.

Xong thì mở `http://localhost:8000`, đăng nhập `admin` / `admin` (đổi bằng biến
môi trường `CHAINLIT_DEMO_USER` / `CHAINLIT_DEMO_PASSWORD`).

| Dịch vụ | Địa chỉ | Ghi chú |
|---|---|---|
| Giao diện chat (Chainlit) | http://localhost:8000 | `app.py` |
| Qdrant dashboard | http://localhost:6333/dashboard | xem trực tiếp 4 collection |
| Neo4j Browser | http://localhost:7474 | `neo4j` / `legal_kg_2024` |
| Ollama (tùy chọn) | http://localhost:11434 | `docker compose --profile ollama up -d ollama` |

Ollama (Qwen2.5 7B chạy cục bộ) **không bắt buộc** — máy thiếu RAM/CPU thì bỏ
qua, dùng model qua `api.shopaikey` là đủ. Đổi model LLM và model embedding
ngay trong khung chat, không cần sửa code.

---

## 2. Cấu trúc source code

Thư mục `src/` xếp theo **đúng thứ tự các bước của pipeline**:

```text
src/
├── data_processing/     ①  Đọc .docx → chunk theo Điều/Khoản/Điểm
│   └── chunking.py
├── embedding/           ②  Nạp model embedding tiếng Việt
│   └── embedding_model.py
├── indexing/            ③  Encode chunk → upsert lên Qdrant
│   └── qdrant_ingest.py
├── retrieval/           ④  Truy xuất hybrid (dense + BM25, hợp nhất RRF)
│   ├── bm25_sparse.py
│   └── qdrant_hybrid_search.py
├── knowledge_graph/     ⑤  Trích xuất thực thể/quan hệ → Neo4j
│   ├── extraction_prompts.py
│   ├── entity_extractor.py
│   ├── graph_builder.py
│   ├── neo4j_ingest.py
│   └── build_graph.py
├── agents/              ⑥  5 agent LangGraph, mỗi agent 1 nhiệm vụ
│   ├── agent_router/          phân loại chit-chat vs câu hỏi pháp lý
│   ├── agent_retrieval/       viết lại truy vấn + gọi hybrid search
│   ├── agent_article_expand/  mở rộng child chunk → toàn văn Điều
│   ├── agent_critic/          kiểm tra đủ căn cứ bằng Knowledge Graph
│   ├── agent_generation/      sinh câu trả lời nháp và câu trả lời cuối
│   └── common/                tài nguyên dùng chung (LLM client, kho Điều)
└── workflow/            ⑦  Nối 5 agent thành 1 đồ thị LangGraph
    ├── pipeline.py            ChatbotWorkflow — điểm vào của toàn bộ luồng chat
    ├── state.py
    └── node_agent_*.py
```

### Từng file làm gì

| Bước | File | Nhiệm vụ |
|---|---|---|
| ① Chunking | [src/data_processing/chunking.py](src/data_processing/chunking.py) | `VBPLChunker` parse `.docx` theo cấu trúc Điều – Khoản – Điểm, sinh **parent chunk** (trọn 1 Điều) và **child chunk** (từng Khoản/Điểm) |
| ② Embedding | [src/embedding/embedding_model.py](src/embedding/embedding_model.py) | `load_embedding_model()` — nạp `AITeamVN/Vietnamese_Embedding_v2` (1024d) hoặc `Alibaba-NLP/gte-multilingual-base` (768d); tự giải nén model finetune trong `references/` nếu có |
| ③ Ingest vector | [src/indexing/qdrant_ingest.py](src/indexing/qdrant_ingest.py) | Chunk → encode dense + BM25 sparse → upsert lên Qdrant, tạo cả collection child lẫn parent |
| ③ BM25 | [src/retrieval/bm25_sparse.py](src/retrieval/bm25_sparse.py) | `BM25SparseVectorizer` — xây/nạp chỉ mục BM25, lưu ra `data/bm25/<collection>.bm25.json` |
| ④ Truy xuất | [src/retrieval/qdrant_hybrid_search.py](src/retrieval/qdrant_hybrid_search.py) | `hybrid_search()` — chạy song song dense + sparse trên child chunk, Qdrant hợp nhất bằng RRF, rồi lấy parent chunk theo `parent_id` |
| ⑤ Prompt trích xuất | [src/knowledge_graph/extraction_prompts.py](src/knowledge_graph/extraction_prompts.py) | Few-shot + CoT prompt cho Qwen2.5-7B trích xuất entity/relation từ mỗi Điều |
| ⑤ Trích xuất | [src/knowledge_graph/entity_extractor.py](src/knowledge_graph/entity_extractor.py) | Gọi LLM, parse JSON, retry khi hỏng — kết quả ghi ra `data/extracted_json/` |
| ⑤ Dựng đồ thị | [src/knowledge_graph/graph_builder.py](src/knowledge_graph/graph_builder.py) | Chuẩn hóa + khử trùng lặp node/edge, giải tham chiếu chéo giữa các văn bản; chứa `to_dieu_node_id()` nối payload Qdrant ↔ node Neo4j |
| ⑤ Nạp Neo4j | [src/knowledge_graph/neo4j_ingest.py](src/knowledge_graph/neo4j_ingest.py) | `Neo4jGraphIngestor` — MERGE node/relationship vào Neo4j (idempotent) |
| ⑤ Entry point | [src/knowledge_graph/build_graph.py](src/knowledge_graph/build_graph.py) | Chạy 4 bước trên tuần tự — **chỉ chạy 1 lần trên Colab** (cần GPU), runtime không dùng |
| ⑥ Router | [src/agents/agent_router/](src/agents/agent_router/) | Chit-chat thì trả lời luôn, câu hỏi pháp lý thì đẩy sang retrieval |
| ⑥ Retrieval | [src/agents/agent_retrieval/](src/agents/agent_retrieval/) | Viết lại truy vấn, gọi hybrid search, trả top-k child chunk |
| ⑥ Article expand | [src/agents/agent_article_expand/](src/agents/agent_article_expand/) | Từ child chunk lấy trọn toàn văn Điều chứa nó |
| ⑥ Critic | [src/agents/agent_critic/](src/agents/agent_critic/) | Truy vấn Neo4j tìm căn cứ còn thiếu, lọc ứng viên qua cổng LLM, bơm thêm ngữ cảnh — xem `critic_query.py`, `node_critic_check.py`, `relevance_gate.py` |
| ⑥ Generation | [src/agents/agent_generation/](src/agents/agent_generation/) | Sinh câu trả lời nháp và câu trả lời cuối; prompt ở `prompts.py` |
| ⑦ Orchestrator | [src/workflow/pipeline.py](src/workflow/pipeline.py) | `ChatbotWorkflow` — dựng tài nguyên dùng chung 1 lần, lắp 5 agent thành đồ thị LangGraph, hỗ trợ 3 kịch bản |
| Giao diện | [app.py](app.py) | Chainlit: chọn kịch bản, đổi model LLM/embedding, hiển thị Critic Report và chế độ Dev (các bước pipeline đã chạy) |

### Thư mục khác

```text
scripts/     # các lệnh chạy tay (xem mục 5)
notebooks/   # 2 notebook Colab: chạy đánh giá 301 câu, kiểm định test set
data/
├── keep/            # corpus gốc: 18 file .docx (đã parse) + 5 .pdf (nguồn tham khảo)
├── extracted_json/  # entity/relation LLM đã trích xuất sẵn → nguồn nạp Neo4j
├── bm25/            # chỉ mục BM25 (phần sparse của hybrid search)
└── eval_testset.jsonl   # bộ 301 câu hỏi đánh giá
docs/        # tài liệu khóa luận, sơ đồ kiến trúc, phương pháp đánh giá
```

---

## 3. Luồng xử lý một câu hỏi

![RAG query flow](docs/assets/rag-query-flow.jpg)

Hệ thống chạy được 3 kịch bản (chọn ở Chat Profile góc trên bên trái) để so
sánh A/B/C trong khóa luận:

| Kịch bản | Luồng |
|---|---|
| `naive` | router → retrieval → sinh câu trả lời từ đúng top-k child chunk |
| `article_expand` | thêm bước mở rộng child chunk → toàn văn Điều trước khi sinh |
| `critic` | thêm Critic Agent: sinh nháp → soi Knowledge Graph tìm căn cứ thiếu → bơm ngữ cảnh → viết lại |

Critic Agent kiểm tra 3 tín hiệu trên đồ thị:

1. **missing_references** — Điều đang dẫn có `THAM_CHIEU` sang Điều khác chưa được đưa vào ngữ cảnh.
2. **compound_penalty** — hành vi có cả chế tài chính lẫn chế tài bổ sung (`CHE_TAI_CHINH` + `CHE_TAI_BO_SUNG`) nhưng ngữ cảnh mới có một loại.
3. **structurally_incomplete** — mới lấy được vài Khoản/Điểm của một Điều nhiều Khoản.

Mỗi Điều ứng viên phải qua **cổng lọc LLM** (`relevance_gate.py`) hỏi "đoạn này
có liên quan câu hỏi không" mới được bơm vào ngữ cảnh — tránh làm loãng prompt.

---

## 4. Hai cơ sở dữ liệu

### Knowledge Graph (Neo4j) — 3 bước nạp

Service `kg-ingest` chạy tự động khi `docker compose --profile app up -d app`,
gồm 3 script tuần tự, tất cả đều idempotent (chạy lại không tạo trùng):

1. `scripts/run_neo4j_ingest.py` — nạp entity/quan hệ đã trích xuất sẵn
   (`data/extracted_json/`) vào Neo4j.
2. `scripts/build_cross_references.py` — vá cạnh `THAM_CHIEU` bằng regex trên
   text gốc, vì bước trích xuất LLM chỉ nhìn trong phạm vi 1 chunk nên bỏ sót
   tham chiếu chéo sang Điều khác.
3. `scripts/repair_che_tai_links.py --apply` — vá cạnh `CHE_TAI_CHINH` /
   `CHE_TAI_BO_SUNG` cho các node chế tài bị bỏ rơi (trích xuất ra nhưng không
   nối vào `HanhVi` nào). **Không có bước này thì tín hiệu "chế tài kép" của
   Critic Agent chỉ kích hoạt được ở đúng 1 Điều trên tổng 1211 Điều** — đo
   thật: sau khi vá tăng lên 5 Điều, số `HanhVi` có cả 2 loại chế tài tăng từ
   3 lên 16.

Chạy riêng bước 3 khi cần:

```bash
docker compose run --rm --entrypoint bash kg-ingest \
  -c "pip install neo4j --quiet && python scripts/repair_che_tai_links.py"
```

(bỏ `--apply` để xem trước, thêm `--undo` để hoàn tác.)

### Vector DB (Qdrant) — container riêng, cổng 6333

Qdrant chạy như một service trong `docker-compose.yml` (container
`legal_qdrant`), **không còn** ở chế độ embedded/thư mục trong repo. Dữ liệu
nằm trong volume Docker `legal_qdrant_data`.

4 collection = 2 model embedding × cặp parent/child của kiến trúc chunking:

| Collection | Model embedding | Dim | Số point |
|---|---|---|---|
| `legal_child_chunks_base` / `legal_parent_chunks_base` | `AITeamVN/Vietnamese_Embedding_v2` | 1024 | 10.355 / 1.211 |
| `legal_child_chunks_gte` / `legal_parent_chunks_gte` | `Alibaba-NLP/gte-multilingual-base` | 768 | 10.355 / 1.211 |

Hai không gian embedding khác dimension nên **bắt buộc tách tên collection**,
không được dùng lẫn. BM25 index (phần sparse) là file cục bộ, không nằm trong
Qdrant: `data/bm25/<tên child collection>.bm25.json`.

Container `qdrant` khởi động lên là **rỗng**. Dữ liệu do service `qdrant-ingest`
nạp vào, chạy tự động cùng `docker compose --profile app up -d app`, tự chọn
đường nhanh hay chậm theo bảng ở [mục 1](#1-chạy-nhanh-bằng-docker).

```bash
docker compose --profile app up qdrant-ingest        # chạy và xem log trực tiếp
docker compose logs -f qdrant-ingest                 # xem log khi đang chạy nền
```

**Đường nhanh** dùng `scripts/migrate_qdrant_to_server.py`: copy nguyên vector +
payload từ 2 thư mục embedded sang server, không load embedding model. Nó bỏ
qua collection đã có; muốn ghi đè thì thêm `--recreate`. Chạy trên máy không có
2 thư mục đó thì báo lỗi kèm chỉ dẫn chứ không im lặng "thành công".

**Đường chậm** dùng `src/indexing/qdrant_ingest.py`, mỗi model 3 việc: chunking
`data/keep` → encode dense + fit BM25 → upsert. Nạp lại từ đầu cho một model:

```bash
docker compose --profile app run --rm --no-deps qdrant-ingest \
  python src/indexing/qdrant_ingest.py --data-dir data/keep \
  --model AITeamVN/Vietnamese_Embedding_v2 \
  --child-collection legal_child_chunks_base \
  --parent-collection legal_parent_chunks_base --recreate
```

---

## 5. Chạy từng bước bằng tay

```bash
# Kiểm thử chunking
python src/data_processing/chunking.py

# Kiểm thử hybrid search (mặc định đọc container qdrant cổng 6333)
python src/retrieval/qdrant_hybrid_search.py \
  --query "Doanh nghiệp cần làm gì khi xử lý dữ liệu cá nhân?" \
  --model AITeamVN/Vietnamese_Embedding_v2 --limit 5 --include-parent

# Chat trong terminal (không cần Chainlit)
python scripts/run_chatbot.py --mode critic
```

Xây lại Knowledge Graph từ đầu (cần GPU, chạy trên Colab):

```bash
python src/knowledge_graph/build_graph.py --data-dir data/keep
```

---

## 6. Đánh giá

| Script | Việc |
|---|---|
| `scripts/run_evaluation.py` | Chạy bộ test qua pipeline thật, xuất `data/eval_results_<mode>.jsonl` theo chuẩn RAGAS |
| `scripts/score_evaluation.py` | Chấm điểm: RAGAS (faithfulness, answer_relevancy, context_precision, answer_correctness) + **Legal Completeness Rate** (chỉ số trung tâm của khóa luận) |
| `scripts/validate_testset.py` | Kiểm định độ tin cậy của chính bộ test set so với văn bản luật gốc |
| `notebooks/colab_full_evaluation.ipynb` | Chạy toàn bộ 301 câu × 3 kịch bản trên Colab T4 |
| `notebooks/colab_validate_testset.ipynb` | Chạy kiểm định test set trên Colab |

```bash
python scripts/run_evaluation.py --all-modes
python scripts/score_evaluation.py --modes naive article_expand critic
```

Chi tiết phương pháp: [docs/evaluation_guide.md](docs/evaluation_guide.md),
[docs/testset_validation_methodology.md](docs/testset_validation_methodology.md),
[docs/reproducibility_supplement.md](docs/reproducibility_supplement.md).

---

## 7. Dữ liệu

`data/keep/` chứa corpus chính: **18 file `.docx`** đã được chunk và ingest
(→ 1.211 Điều = 1.211 parent chunk, 10.355 child chunk, và 18 thư mục tương
ứng trong `data/extracted_json/`). 5 file `.pdf` còn lại lưu như nguồn tài liệu
gốc — chunker hiện chỉ xử lý `.docx`, chưa parse `.pdf`.

Bộ đánh giá `data/eval_testset.jsonl` gồm **301 câu hỏi**, chia 4 nhóm: chế tài
kép, cross-Điều, structural đa Khoản, và nhóm control.

---

## 8. Công nghệ

Python 3.11 · `python-docx` · `sentence-transformers` · Qdrant · Neo4j 5.26 ·
LangGraph · LangChain · Chainlit · Qwen2.5-7B-Instruct (Ollama) hoặc
gpt-4o-mini · RAGAS.

---

## 9. Bảo mật

Không commit: `.env`, `.env.*`, `.venv/`, `data/.qdrant*/`, API key, service
key, database password. Đã cấu hình sẵn trong `.gitignore`.
