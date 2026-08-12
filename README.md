# LEGAL IT CHATBOT

Hệ thống hỏi đáp pháp luật tiếng Việt kết hợp **Retrieval-Augmented
Generation (RAG)**, **Knowledge Graph** và **Critic Agent**. Dự án tập trung
vào các văn bản pháp luật liên quan đến công nghệ thông tin, dữ liệu cá nhân,
an toàn thông tin, giao dịch điện tử, viễn thông, sở hữu trí tuệ và công nghiệp
công nghệ số.

Đây là sản phẩm của khóa luận:

> **Xây dựng hệ thống hỏi đáp pháp luật tiếng Việt kết hợp
> Retrieval-Augmented Generation, đồ thị tri thức và Critic Agent**

Sinh viên thực hiện: **Phan Quyết Tâm Phú** và **Nguyễn Gia Huy**<br>
Giảng viên hướng dẫn: **ThS. Huỳnh Thanh Việt**<br>
Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh, năm 2026.

> Hệ thống là nguyên mẫu hỗ trợ tra cứu và nghiên cứu. Câu trả lời do mô hình
> sinh ra không thay thế ý kiến thẩm định của chuyên gia pháp luật.

---

## 1. Bài toán và giải pháp

RAG thông thường lấy các đoạn văn bản có độ tương đồng cao với câu hỏi. Cách
tiếp cận này có thể tìm đúng một Khoản hoặc Điểm nhưng vẫn bỏ sót:

- điều kiện, ngoại lệ hoặc thành phần khác trong cùng một Điều;
- hình thức xử phạt bổ sung và biện pháp khắc phục hậu quả;
- căn cứ nằm tại Điều khác được dẫn chiếu trong văn bản.

LEGAL IT CHATBOT xử lý vấn đề trên bằng ba lớp:

1. **Chunking theo Điều – Khoản – Điểm:** child chunk phục vụ truy xuất chi
   tiết; parent Article giữ toàn văn Điều để khôi phục ngữ cảnh.
2. **Hybrid retrieval:** dense embedding và BM25 được hợp nhất bằng Reciprocal
   Rank Fusion (RRF) trong Qdrant.
3. **Kiểm tra độ đầy đủ sau truy xuất:** Critic Agent dùng Knowledge Graph
   trong Neo4j để phát hiện khoảng trống bằng chứng, lọc Điều ứng viên qua
   semantic relevance gate và chỉ sinh lại câu trả lời khi có bằng chứng phù
   hợp.

Knowledge Graph không thay thế bộ truy xuất văn bản. Graph giúp định vị quan
hệ và Điều có khả năng bị thiếu; nội dung pháp lý đưa vào mô hình sinh vẫn
được lấy từ parent Article trong Qdrant.

![Kiến trúc hệ thống RAG](docs/assets/rag-system-architecture.jpg)

---

## 2. Kết quả nghiên cứu

Các số liệu dưới đây được tổng hợp từ Chương 5 của bản khóa luận cuối và dùng
dấu phẩy làm dấu thập phân theo quy ước trình bày tiếng Việt.

### 2.1. Thiết kế thực nghiệm

Ba chế độ được đánh giá trên cùng corpus, cùng câu hỏi và cùng retriever trong
từng cấu hình:

| Chế độ | Cách xây dựng ngữ cảnh |
|---|---|
| **Naive RAG** | Sinh câu trả lời trực tiếp từ top-5 child chunks |
| **Article Expansion** | Mở rộng các child chunks thành toàn văn Điều |
| **KG-based Critic** | Sinh nháp, kiểm tra evidence gap bằng graph, bổ sung bằng chứng có chọn lọc rồi sinh lại khi cần |

Benchmark gồm **301 câu hỏi**:

| Nhóm | Số câu | Mục tiêu đánh giá |
|---|---:|---|
| same_dieu_compound_penalty | 51 | Nhiều chế tài hoặc thành phần trong cùng Điều |
| cross_reference | 16 | Cần căn cứ từ Điều được dẫn chiếu |
| structural_multi_part | 130 | Cần tổng hợp nhiều Khoản/Điểm |
| control_no_gap | 104 | Điều đã đủ căn cứ, dùng làm nhóm đối chứng |

Cấu hình chính sử dụng **AITeamVN/Vietnamese_Embedding_v2**; cấu hình
**Alibaba-NLP/gte-multilingual-base** được chạy bổ sung để kiểm tra độ ổn định
của xu hướng.

### 2.2. Kết quả tổng hợp

| Embedding | MRR | nDCG@5 | Naive RAG LCR | Article Expansion LCR | KG-based Critic LCR |
|---|---:|---:|---:|---:|---:|
| Vietnamese_Embedding_v2 | **0,856** | **0,866** | 43,77% | 72,72% | **81,33%** |
| GTE Multilingual Base | 0,831 | 0,843 | 40,28% | 73,51% | **77,44%** |

Với cấu hình chính, Article Expansion tăng Legal Completeness Rate (LCR) từ
43,77% lên 72,72%. KG-based Critic đạt 81,33%, cao hơn Naive RAG **37,56 điểm
phần trăm** và cao hơn Article Expansion **8,61 điểm phần trăm**. Trên GTE,
Critic vẫn đạt LCR tổng cao nhất, cho thấy xu hướng chính được duy trì khi đổi
embedding.

### 2.3. Legal Completeness Rate theo nhóm

**Vietnamese_Embedding_v2 — cấu hình phân tích chính**

| Nhóm câu hỏi | Naive RAG | Article Expansion | KG-based Critic |
|---|---:|---:|---:|
| Cùng Điều, chế tài kép | 27,45% | 58,01% | **75,49%** |
| Tham chiếu chéo | 45,83% | 49,48% | **70,83%** |
| Nhiều thành phần cấu trúc | 28,27% | 68,64% | **77,67%** |
| Nhóm đối chứng | 70,83% | 88,62% | **90,38%** |
| **Toàn bộ 301 câu** | **43,77%** | **72,72%** | **81,33%** |

**GTE Multilingual Base — cấu hình đối chiếu**

| Nhóm câu hỏi | Naive RAG | Article Expansion | KG-based Critic |
|---|---:|---:|---:|
| Cùng Điều, chế tài kép | 25,33% | 58,17% | **71,73%** |
| Tham chiếu chéo | 39,58% | 56,77% | **61,98%** |
| Nhiều thành phần cấu trúc | 24,35% | 68,98% | **74,04%** |
| Nhóm đối chứng | 67,63% | **89,26%** | 86,86% |
| **Toàn bộ 301 câu** | **40,28%** | **73,51%** | **77,44%** |

Lợi thế rõ nhất của Critic trên cấu hình chính nằm ở nhóm tham chiếu chéo:
70,83%, so với 49,48% của Article Expansion. Điều này phù hợp với vai trò của
cạnh THAM_CHIEU trong việc tìm Điều liên quan ngoài phạm vi Điều ban đầu.
Tuy nhiên, Critic không đứng đầu ở mọi nhóm: với nhóm đối chứng trên GTE,
Article Expansion đạt 89,26%, cao hơn 86,86% của Critic.

### 2.4. Chất lượng câu trả lời theo RAGAS

| Cấu hình | Chế độ | Faithfulness | Answer Relevancy | Context Precision | Answer Correctness |
|---|---|---:|---:|---:|---:|
| Vietnamese_Embedding_v2 | Naive RAG | 0,705 | 0,349 | **0,965** | 0,544 |
| Vietnamese_Embedding_v2 | Article Expansion | 0,841 | **0,597** | 0,963 | 0,657 |
| Vietnamese_Embedding_v2 | KG-based Critic | **0,857** | 0,593 | 0,961 | **0,668** |
| GTE Base | Naive RAG | 0,656 | 0,408 | **0,974** | 0,521 |
| GTE Base | Article Expansion | 0,858 | **0,776** | 0,944 | **0,717** |
| GTE Base | KG-based Critic | **0,873** | 0,709 | 0,943 | 0,707 |

Critic có Faithfulness cao nhất ở cả hai embedding. Các chỉ số còn lại không
cho thấy một cấu hình hoặc một chế độ tốt nhất tuyệt đối: Article Expansion
nhỉnh hơn Critic về Answer Relevancy ở cả hai cấu hình và về Answer Correctness
trên GTE.

### 2.5. Độ dài ngữ cảnh và chi phí

| Cấu hình | Chế độ | Context Recall | Ký tự context/câu | Tổng token/câu | Token lượt sinh cuối | Lượt gọi LLM/câu |
|---|---|---:|---:|---:|---:|---:|
| Vietnamese_Embedding_v2 | Naive RAG | 0,952 | 1.594 | 1.516 | 1.352 | 1,0 |
| Vietnamese_Embedding_v2 | Article Expansion | 0,952 | 8.296 | 2.782 | 2.618 | 1,0 |
| Vietnamese_Embedding_v2 | KG-based Critic | **0,962** | 4.748 | 6.540 | 2.187 | 5,7 |
| GTE Base | Naive RAG | 0,942 | 1.612 | 1.358 | 1.358 | 1,0 |
| GTE Base | Article Expansion | 0,945 | 8.769 | 2.775 | 2.775 | 1,0 |
| GTE Base | KG-based Critic | **0,955** | 4.512 | 6.308 | 2.190 | 5,7 |

Critic tạo context cuối ngắn hơn Article Expansion nhưng đạt Context Recall cao
hơn trên cả hai embedding. Đổi lại, Critic dùng trung bình **5,7 lượt gọi
LLM/câu** và có tổng token toàn pipeline cao nhất. Vì vậy, đây là đánh đổi giữa
độ đầy đủ pháp lý với chi phí và độ trễ, không phải một phương án rẻ hơn.

### 2.6. Độ tin cậy của test set

Kiểm định test set trong khóa luận đạt:

| Tiêu chí | Kết quả toàn bộ |
|---|---:|
| B1 — Required facts có căn cứ | 99,7% |
| B2 — Reference answer có căn cứ | 99,3% |
| B3 — Reference answer bao phủ required facts | 100,0% |
| B4 — Câu hỏi tự nhiên, rõ nghĩa | 100,0% |
| B5 — Câu hỏi cần viện dẫn pháp luật | 100,0% |
| B6 — Nhãn category chính xác | 71,4% |

B6 thấp hơn các tiêu chí còn lại nên kết quả theo từng category cần được diễn
giải thận trọng. Kết quả hiện tại cũng chưa thay thế đánh giá mù từ nhiều
chuyên gia pháp luật.

---

## 3. Luồng xử lý

![Luồng xử lý một câu hỏi](docs/assets/rag-query-flow.jpg)

Một legal query đi qua các bước:

1. Router phân biệt hội thoại thông thường và câu hỏi pháp luật.
2. Retrieval Agent viết lại truy vấn, tìm kiếm dense và BM25 trên child chunks,
   sau đó hợp nhất thứ hạng bằng RRF.
3. Tùy chế độ, hệ thống giữ nguyên top-k child, mở rộng toàn Điều hoặc sinh câu
   trả lời nháp để Critic kiểm tra.
4. Critic Agent tìm ba loại khoảng trống:
   - thiếu Điều được dẫn qua quan hệ THAM_CHIEU;
   - thiếu chế tài chính hoặc chế tài bổ sung;
   - thiếu Khoản/Điểm thuộc cùng một Điều nhiều thành phần.
5. Candidate evidence phải qua semantic relevance gate trước khi được thêm vào
   ngữ cảnh.
6. Generator giữ nguyên bản nháp nếu không có bằng chứng mới, hoặc sinh lại câu
   trả lời cuối nếu Critic tìm thấy phần cần bổ sung.

Thông số thực nghiệm chính:

| Tham số | Giá trị |
|---|---:|
| Top-k child | 5 |
| Candidate pool của Article Expansion | 20 |
| Ngưỡng Article Expansion | 0,40 × điểm cao nhất |
| Số Điều mở rộng tối đa | 5 |
| Ngưỡng Critic | 0,60 × điểm cao nhất |
| Số Điều Critic kiểm tra tối đa | 4 |
| Duyệt THAM_CHIEU | Tối đa 2 hop, tối đa 8 Điều |
| Generator/semantic gate | Qwen2.5-7B, temperature 0,2 |
| LCR judge | GPT-4o-mini, temperature 0 |

---

## 4. Dữ liệu và chỉ mục

Corpus chính trong **data/keep/** gồm **18 tệp DOCX** được pipeline parse và
**5 tệp PDF** lưu làm nguồn tham khảo. Kết quả ingest hiện tại:

| Thành phần | Quy mô |
|---|---:|
| Parent Article | 1.211 |
| Child chunk | 10.355 |
| Bộ câu hỏi đánh giá | 301 |
| Nhóm câu hỏi đánh giá | 4 |

Qdrant sử dụng bốn collection độc lập:

| Collection | Embedding | Số chiều | Số point |
|---|---|---:|---:|
| legal_child_chunks_base | Vietnamese_Embedding_v2 | 1.024 | 10.355 |
| legal_parent_chunks_base | Vietnamese_Embedding_v2 | 1.024 | 1.211 |
| legal_child_chunks_gte | GTE Multilingual Base | 768 | 10.355 |
| legal_parent_chunks_gte | GTE Multilingual Base | 768 | 1.211 |

Hai embedding có số chiều khác nhau nên không được dùng lẫn collection. Chỉ
mục BM25 của mỗi child collection nằm tại **data/bm25/**.

---

## 5. Chạy nhanh bằng Docker

### 5.1. Yêu cầu

- Docker và Docker Compose;
- API key nếu dùng các LLM qua endpoint tương thích OpenAI; hoặc Ollama nếu
  chạy Qwen2.5-7B cục bộ;
- đủ dung lượng để tải hai embedding model nếu chưa có Qdrant snapshot.

### 5.2. Khởi động

~~~bash
cp .env.example .env
# Điền OPENAI_API_KEY trong .env nếu dùng model qua API.

docker compose --profile app up -d app
~~~

Lệnh trên tự khởi động Neo4j và Qdrant, nạp Knowledge Graph, tạo/nạp bốn
collection vector rồi mới mở ứng dụng Chainlit.

Nếu có sẵn hai snapshot **data/.qdrant_base/** và
**data/.qdrant_gte_base/**, service ingest sẽ copy vector vào Qdrant trong vài
phút. Nếu không có snapshot, hệ thống tự xây lại chỉ mục từ **data/keep/**;
bước encode hai embedding có thể mất khoảng 1–3 giờ tùy phần cứng.

Theo dõi tiến độ:

~~~bash
docker compose logs -f qdrant-ingest
~~~

Chạy Qwen2.5-7B bằng Ollama:

~~~bash
docker compose --profile ollama up -d ollama
docker exec legal_ollama ollama pull qwen2.5:7b
~~~

Sau khi hệ thống sẵn sàng, mở:

| Dịch vụ | Địa chỉ |
|---|---|
| Chainlit | http://localhost:8000 |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Neo4j Browser | http://localhost:7474 |
| Ollama API | http://localhost:11434 |

Tài khoản Chainlit mặc định là **admin / admin**. Có thể đổi bằng
**CHAINLIT_DEMO_USER** và **CHAINLIT_DEMO_PASSWORD**.

---

## 6. Chạy bằng dòng lệnh

Sau khi các dịch vụ dữ liệu và môi trường Python đã sẵn sàng:

~~~bash
# Một chế độ
python scripts/run_chatbot.py --query "Doanh nghiệp phải làm gì khi xử lý dữ liệu cá nhân?" --mode critic

# So sánh cả ba chế độ trên cùng câu hỏi
python scripts/run_chatbot.py --query "Hành vi này bị xử phạt và khắc phục hậu quả như thế nào?" --compare-all
~~~

Đổi sang GTE và cặp collection tương ứng:

~~~bash
QDRANT_CHILD_COLLECTION=legal_child_chunks_gte QDRANT_PARENT_COLLECTION=legal_parent_chunks_gte EMBEDDING_MODEL=Alibaba-NLP/gte-multilingual-base python scripts/run_chatbot.py --query "Điều kiện của hợp đồng điện tử là gì?"
~~~

---

## 7. Tái lập đánh giá

~~~bash
# Chạy 301 câu qua cả ba chế độ
python scripts/run_evaluation.py --all-modes

# Chấm Legal Completeness Rate và bốn chỉ số RAGAS
python scripts/score_evaluation.py --modes naive article_expand critic --judge-provider openai --judge-model gpt-4o-mini

# Kiểm định bộ test set
python scripts/validate_testset.py
~~~

Các tài liệu liên quan:

- [Hướng dẫn đánh giá](docs/evaluation_guide.md)
- [Phương pháp kiểm định test set](docs/testset_validation_methodology.md)
- [Bổ sung khả năng tái lập](docs/reproducibility_supplement.md)
- [Audit hiệu lực văn bản](docs/legal_effectiveness_audit_2026-07-01.md)

---

## 8. Cấu trúc mã nguồn

~~~text
.
├── app.py                         # Giao diện Chainlit
├── src/
│   ├── data_processing/           # Parse DOCX, parent/child chunking
│   ├── embedding/                 # Nạp embedding model
│   ├── indexing/                  # Ingest Qdrant
│   ├── retrieval/                 # Dense + BM25 + RRF
│   ├── knowledge_graph/           # Trích xuất và nạp Neo4j
│   ├── agents/
│   │   ├── agent_router/          # Phân loại truy vấn
│   │   ├── agent_retrieval/       # Hybrid retrieval
│   │   ├── agent_article_expand/  # Khôi phục toàn văn Điều
│   │   ├── agent_critic/          # Phát hiện evidence gap
│   │   └── agent_generation/      # Sinh nháp và câu trả lời cuối
│   └── workflow/                  # Đồ thị điều phối LangGraph
├── scripts/                       # Chat CLI, ingest và evaluation
├── notebooks/                     # Notebook Colab
├── data/                          # Corpus, BM25, test set, kết quả
├── docs/                          # Tài liệu kỹ thuật và đánh giá
└── tests/                         # Kiểm thử
~~~

Các điểm vào chính:

| Tệp | Vai trò |
|---|---|
| [app.py](app.py) | Giao diện chat và lựa chọn model/chế độ |
| [src/workflow/pipeline.py](src/workflow/pipeline.py) | Điều phối toàn bộ pipeline |
| [src/data_processing/chunking.py](src/data_processing/chunking.py) | Chunking Điều – Khoản – Điểm |
| [src/retrieval/qdrant_hybrid_search.py](src/retrieval/qdrant_hybrid_search.py) | Hybrid search và RRF |
| [src/agents/agent_critic/node_critic_check.py](src/agents/agent_critic/node_critic_check.py) | Logic Critic Agent |
| [scripts/run_evaluation.py](scripts/run_evaluation.py) | Chạy benchmark |
| [scripts/score_evaluation.py](scripts/score_evaluation.py) | Tính LCR và RAGAS |

---

## 9. Giới hạn

- Corpus mới bao phủ một số lĩnh vực pháp luật Việt Nam và phụ thuộc vào
  snapshot hiệu lực của văn bản.
- LCR và RAGAS có sử dụng đánh giá tự động; chưa có đánh giá mù quy mô lớn từ
  nhiều chuyên gia pháp luật.
- Độ chính xác nhãn category của test set đạt 71,4%, thấp hơn các tiêu chí chất
  lượng nội dung còn lại.
- Chưa có component ablation đầy đủ để cô lập đóng góp của từng tín hiệu
  Critic, semantic gate và bước sinh lại.
- Knowledge Graph có thể thiếu hoặc sai entity/relation, từ đó làm Critic bỏ
  sót hoặc lấy thừa bằng chứng.
- Critic cải thiện độ đầy đủ nhưng tăng số lượt gọi LLM, tổng token và độ trễ.

---

## 10. Công nghệ

Python 3.11 · LangGraph · LangChain · Chainlit · Qdrant · Neo4j 5.26 ·
sentence-transformers · BM25 · Qwen2.5-7B · GPT-4o-mini · RAGAS.

## 11. Bảo mật

Không commit API key, mật khẩu thật, tệp **.env**, thư mục môi trường ảo hoặc
snapshot Qdrant chứa dữ liệu riêng. Mẫu cấu hình được cung cấp tại
[.env.example](.env.example).
