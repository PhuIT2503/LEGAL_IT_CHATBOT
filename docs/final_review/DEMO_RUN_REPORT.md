# Báo cáo chạy demo

## Phạm vi và nguyên tắc

- Ngày kiểm tra: 2026-08-04 (Asia/Ho_Chi_Minh).
- Snapshot mã nguồn: `9bf145e6744599e1e15c21e2011a02f7ecff87f5`.
- Working tree có thay đổi sẵn của người dùng; quá trình demo không xóa, reset hoặc ghi đè các thay đổi đó.
- Chỉ mục sử dụng: Qdrant embedded tại `data/.qdrant_base` với `AITeamVN/Vietnamese_Embedding_v2`.
- Mô hình sinh ưu tiên: Ollama native, `qwen2.5:7b`.
- Không đọc hoặc ghi lại giá trị khóa API. Báo cáo chỉ xác nhận tên biến cấu hình khi cần.
- Ba case demo không thay thế phép đánh giá đầy đủ 301 câu.

## Điều kiện đầu vào khóa luận

`KLTN_Final.docx` đã được bổ sung vào thư mục gốc sau phiên chạy demo. Tệp được sao lưu nguyên trạng thành `KLTN_Final_Backup.docx`; bản hoàn thiện và PDF đã được tạo dưới tên mới, không ghi đè tệp nguồn.

## Trạng thái môi trường ban đầu

| Thành phần | Trạng thái | Bằng chứng |
|---|---|---|
| Docker daemon | Hoạt động | Docker Engine 29.2.1 |
| Neo4j | Hoạt động, healthy | HTTP 7474 trả 200; Bolt 7687 mở |
| Knowledge Graph | Có dữ liệu | 15.666 node; 14.137 quan hệ; 1.140 node `Dieu`; 1.612 quan hệ `THAM_CHIEU` |
| Qdrant Base | Có dữ liệu, embedded | 1.211 parent; 10.355 child; dense 1.024 chiều, cosine; sparse BM25 |
| BM25 Base | Có dữ liệu | 10.355 tài liệu; 1.914 từ vựng; `k1=1,5`; `b=0,75`; `avgdl=57,5987` |
| Ollama native | Hoạt động | Có model `qwen2.5:7b` |
| Chainlit | Đã chạy và trả lời thành công | HTTP 8000 trả 200; ảnh `demo_chainlit_overview.png`; sau khi chụp đã dừng app để giải phóng khóa Qdrant embedded |

## Đối chiếu hành vi mã nguồn

- CLI hỗ trợ `--mode {naive,article_expand,critic}`, `--compare-all` và `--top-k`; mặc định `top_k=5`.
- Pipeline dùng chung bước hybrid retrieval cho ba chế độ; candidate pool mặc định 20 và hợp nhất dense + BM25 bằng RRF.
- Critic giới hạn phạm vi Điều theo `critic_score_ratio=0,6`, tối đa 4 Điều ở vòng đầu.
- Quan hệ `THAM_CHIEU` được kiểm tra theo cả chiều đi ra và chiều đi vào. Mã nguồn đọc toàn bộ cạnh có hướng rồi ánh xạ hai trường hợp đối xứng tại `src/agents/agent_critic/critic_query.py`.
- Duyệt bắc cầu từ các Điều mới fetch có `max_hops=2`, tổng số Điều fetch không vượt quá `2 × critic_max_dieu`.
- Cổng liên quan dùng LLM theo quyết định yes/no. Nếu cổng gặp ngoại lệ, mã nguồn hiện hành fail-open và ghi cảnh báo; vì vậy log lỗi phải được giữ để không diễn giải nhầm thành một quyết định xác nhận thành công.
- Trace trả về trực tiếp từ `ChatbotWorkflow.run()` có answer, retrieved chunks/Điều, critic report, graph context/fetched Điều và token usage; một số trường chi tiết theo yêu cầu không được mã nguồn log sẵn sẽ ghi `null` hoặc `not_logged` trong trace demo, không suy đoán.

## Lệnh và phiên chạy đã xác nhận

1. Kiểm tra CLI bằng `scripts/run_chatbot.py --help` với Qdrant Base và embedding Base.
2. Khởi động `neo4j`, chạy lại `kg-ingest` theo Compose; ingest kết thúc mã 0 và số node/quan hệ không đổi sau thao tác idempotent.
3. Khởi động profile `app`, đăng nhập tài khoản demo cục bộ, chọn `qwen2.5:7b`, nhập câu hỏi `cat1_01` và mở bảng `Critic Report`. Router phân loại `legal`; hybrid retrieval lấy 5 chunk thuộc 2 Điều; Critic phát hiện Điều 98 có 9 phần nhưng mới retrieve 4 phần; pipeline lấy toàn văn Điều và sinh lại câu trả lời.
4. Dừng container `app` để tránh hai tiến trình đồng thời mở Qdrant embedded, sau đó chạy `--compare-all --top-k 5 --query ...`. Lượt hợp lệ kết thúc mã 0; log lưu tại `docs/demo_logs/demo_compare_all_cli.log`.
5. Chạy riêng ba câu trong testset bằng workflow production, `skip_router=True` giống `scripts/run_evaluation.py`, giữ nguyên toàn bộ quyết định retrieval/Critic; trace JSON lưu tại `docs/demo_traces/`.
6. Chạy Cypher read-only cho case tham chiếu chéo trong Neo4j Browser; nhận 5 node `Dieu`, 4 quan hệ `THAM_CHIEU`, 4 record, hoàn tất sau 322 ms.

Lượt gọi helper đầu tiên thiếu cờ `--query`, kết thúc mã 2. Log lỗi được giữ tại `docs/demo_logs/demo_compare_all_cli.failed.log`; helper đã được sửa và lượt chạy lại thành công. Không có thay đổi nào trong pipeline để tạo kết quả mong muốn.

## Kết quả compare-all

| Chế độ | Retrieval dùng chung | Phần ngữ cảnh bổ sung | Hành vi quan sát |
|---|---|---|---|
| Naive | 5 child, 2 Điều | Không | Trả lời mức phạt 50–70 triệu đồng nhưng ghép thêm nội dung không đúng trọng tâm từ các khoản retrieved khác. |
| Article Expansion | 5 child, 2 Điều; phạm vi mở rộng chọn 1 Điều | Toàn văn Điều 98, 1.966 ký tự | Bổ sung tịch thu, đình chỉ và thu hồi/hoàn trả tên miền. |
| Critic | 5 child, 2 Điều | Critic phát hiện Điều 98 có 9 phần nhưng retrieved 4; fetch lại toàn Điều | Sinh lại câu trả lời với mức phạt chính, hình phạt bổ sung và biện pháp khắc phục. |

## Ba case study Critic Agent

### Case 1 — không thiếu ngữ cảnh

- Testset ID/category: `cat4_02` / `control_no_gap`.
- Câu hỏi: “Theo Luật Giao dịch điện tử 2023, thông báo dưới dạng thông điệp dữ liệu trong giao kết và thực hiện hợp đồng điện tử có giá trị pháp lý như thế nào?”
- Retrieved: 5 child thuộc 5 Điều (`D38`, `D8`, `D7`, `D3`, `D10`).
- Critic: `is_complete=True`; không tạo candidate, không gọi relevance gate, không fetch graph.
- Kết quả: giữ câu trả lời nháp; `regenerated=false`; 1.048 token, 1 LLM call; không có lỗi.

### Case 2 — thiếu nội dung trong cùng Điều

- Testset ID/category: `cat1_01` / `same_dieu_compound_penalty`.
- Câu hỏi: “Doanh nghiệp thiết lập mạng xã hội nhưng không có giấy phép thì bị xử phạt như thế nào theo Nghị định 15/2020/NĐ-CP?”
- Retrieved: 5 child thuộc 2 Điều (`D98`, `D15`); trong Điều 98 mới lấy 4/9 phần.
- Candidate/fetch: Điều 98; relevance gate trả `accept`; không có fail-open.
- Kết quả: lấy toàn văn Điều 98 và sinh lại; `regenerated=true`; 3.696 token, 3 LLM call; không có lỗi.

### Case 3 — thiếu do tham chiếu chéo

- Testset ID/category: `cat2_07` / `cross_reference`.
- Câu hỏi: “Khi tôi gửi đơn đặt hàng điện tử cho một cửa hàng trực tuyến nhưng không thấy họ phản hồi xác nhận, tôi có thể coi như đơn hàng đó chưa được xác lập không? Và nói chung khi nào một thông điệp dữ liệu như đơn đặt hàng được xem là bên kia đã ‘nhận’ được về mặt pháp lý?”
- Retrieved: 5 child thuộc 3 Điều (`D16`, `D17`, `D14`).
- Candidate/fetch: `D37`, `D16`; cổng trả hai `accept`; hai ứng viên duyệt sâu khác bị `reject`; không có fail-open.
- Neo4j xác nhận Điều 37 có bốn cạnh đi ra đến Điều 15, 16, 17, 18. Trong truy vết production, Điều 37 được phát hiện từ cạnh đi vào đối với Điều 16 đã retrieved, chứng minh mã nguồn xử lý hai chiều của cạnh có hướng.
- Kết quả: bổ sung ngữ cảnh graph và sinh lại; `regenerated=true`; 6.316 token, 6 LLM call; không có lỗi.

## Tám hình minh họa

| Tệp | Bản chất bằng chứng | Kích thước |
|---|---|---|
| `demo_qdrant_collection.png` | Trang kiểm tra read-only sinh từ Qdrant Base thực | 1720×980 |
| `demo_neo4j_tham_chieu.png` | Neo4j Browser thật, Cypher read-only, 5 node/4 cạnh | 1715×1100 |
| `demo_chainlit_overview.png` | Chainlit thật, Qwen local, câu trả lời và Critic Report | 1715×1100 |
| `demo_compare_all_cli.png` | Ảnh đọc từ log CLI thật, exit 0 | 1720×1224 |
| `demo_critic_trace_json.png` | Tóm tắt trực quan từ trace JSON thật của `cat1_01` | 1720×980 |
| `demo_critic_no_gap.png` | Kịch bản `cat4_02` | 1720×980 |
| `demo_critic_same_article.png` | Kịch bản `cat1_01` | 1720×980 |
| `demo_critic_cross_article.png` | Kịch bản `cat2_07` | 1720×980 |

## Trạng thái đánh giá 301 câu

- Testset hiện có đúng 301 dòng: 51 `same_dieu`, 16 `cross`, 130 `structural`, 104 `control`.
- Artefact cũ ngày 2026-07-31 có đủ 301 dòng cho Naive và Article Expansion, nhưng Critic chỉ có 23 dòng. Do đó không có một bộ kết quả ba chế độ đầy đủ, cùng phiên và đủ 301 câu để báo cáo so sánh tổng thể.
- Không tái chạy 301 câu trong đợt hoàn thiện này. Không dùng smoke 1 câu hoặc ba case study để suy rộng thành metric toàn bộ testset.

## Cảnh báo quan sát được

- Hugging Face cảnh báo request chưa xác thực; model embedding vẫn tải/đọc thành công.
- Neo4j cảnh báo property `r.ghi_chu` chưa tồn tại; truy vấn vẫn trả cạnh và pipeline vẫn hoàn tất.
- Chainlit data layer báo lỗi lưu thread vì trường `tags` nhận Python list trong khi SQLite binding không hỗ trợ trực tiếp. Câu hỏi và phản hồi vẫn hoàn thành, nhưng lịch sử phiên này có thể không được persist đúng.
- Chainlit cảnh báo chưa cấu hình blob storage cho element; không ảnh hưởng phần trả lời văn bản đã chụp.

## Trạng thái dịch vụ sau demo

- Neo4j: đang chạy, healthy.
- Ollama native: đang chạy, có `qwen2.5:7b`.
- Chainlit app: đã dừng có chủ đích sau khi chụp để Qdrant embedded có thể được CLI và bước kiểm tra read-only mở an toàn.
- HTTP server tạm dùng để chụp các trang bằng chứng không phải thành phần của hệ thống chatbot.
