# Ground Truth Legal Adjudication

Ngày audit: 2026-07-27

## Executive outcome

Audit này không sửa benchmark. Đây là engineering adjudication để xác định nhãn
nào đủ tin cậy cho việc tối ưu Retrieval/Cross Encoder; không thay thế ý kiến
pháp lý có thẩm quyền.

Kết quả đối với 44 expected provisions:

| Classification | Provisions | Tỷ lệ |
|---|---:|---:|
| Confirmed correct | 29 | 65.91% |
| Likely incorrect | 10 | 22.73% |
| Ambiguous | 5 | 11.36% |

Phát hiện quan trọng:

1. Sáu nhãn `Luật An ninh mạng 2025, Điều 13 khoản 3 điểm h` trong các
   testcase cyber attack, SQL injection và malware là likely incorrect. Điểm h
   điều chỉnh mạo danh/giả mạo thông tin, hình ảnh, giọng nói gây ảnh hưởng đến
   uy tín, danh dự, nhân phẩm; nó không điều chỉnh hành vi khai thác lỗ hổng,
   truy cập trái phép, SQL injection hoặc cài mã độc.
2. Ba nhãn `Nghị định 15/2020/NĐ-CP, Điều 81` không có dữ kiện về mục đích
   chiếm đoạt tài sản. Điều 81 là chế tài cho việc sử dụng mạng nhằm chiếm đoạt
   tài sản, không phải chế tài chung cho mọi truy cập trái phép hoặc lấy dữ liệu.
3. Nhãn `Luật Bảo vệ dữ liệu cá nhân 2025, Điều 20` trong
   `sql_injection_hard_001` thiếu yếu tố chuyển dữ liệu xuyên biên giới.
4. Ba nhãn Điều 13 khoản 3 điểm h trong nhóm deepfake có quan hệ chủ đề nhưng
   query không xác lập hậu quả ảnh hưởng uy tín, danh dự hoặc nhân phẩm. Chúng
   được giữ ở trạng thái ambiguous, không kết luận sai.
5. Trong năm expected provisions được forensic trace gán là CE loss, bốn nhãn
   là confirmed và một nhãn (`cyber_attack_medium_001` — NĐ15 Điều 81) là
   likely incorrect. CE score thấp của nhãn này không phải bằng chứng về CE
   false negative.

## Root cause

Ground truth hiện trộn ba tiêu chuẩn khác nhau:

- **Direct applicability**: điều luật trực tiếp chứa các yếu tố của query.
- **Topic adjacency**: điều luật cùng lĩnh vực nhưng thiếu một yếu tố cấu thành
  quan trọng.
- **Negative/rule-out relevance**: điều luật có thể được viện dẫn để giải thích
  vì sao không áp dụng, nhưng benchmark vẫn tính như một positive hit.

Benchmark contract không ghi loại quan hệ này. Vì vậy một Cross Encoder ưu tiên
đúng semantic có thể bị chấm sai khi hạ hạng một annotation chỉ cùng chủ đề.
Ngược lại, tối ưu theo mọi expected label có thể dạy retriever đưa một căn cứ
không điều chỉnh hành vi lên top-10.

## Evidence and method

### Frozen inputs

- Benchmark: `evaluation/benchmark/benchmark.json`
- Benchmark SHA-256:
  `126438a83f9902d950b140f74fd1f527b899ec9dc160ce1163f47c0c5d3e114c`
- Authoritative run:
  `evaluation/results/benchmark_30_applicability_recovery_20260726/`
- Corpus snapshot: `data/.qdrant_base`
- Corpus coverage: 44/44 expected provisions tồn tại.
- Retrieval-only replay: top-10 chunk list khớp 30/30 testcase, mismatch = 0.

Mỗi annotation được đối chiếu theo bốn lớp:

1. Query và exact document/article/clause/point trong benchmark.
2. Exact provision text trong Qdrant child/parent payload.
3. First failed stage, score và rank từ retrieval-only forensic replay.
4. Văn bản chính thức hoặc metadata văn bản chính thức.

### Official source register

- [Luật An ninh mạng số 116/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=216499&orggroupid=1&pageid=27160)
- [Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroupid=3)
- [Nghị định 15/2020/NĐ-CP](https://vanban.chinhphu.vn/?docid=199053&pageid=27160)
- [Nghị định 14/2022/NĐ-CP sửa đổi Nghị định 15](https://vanban.chinhphu.vn/?classid=1&docid=205256&orggroupid=2&pageid=27160)
- [Nghị định 53/2022/NĐ-CP](https://vanban.chinhphu.vn/?classid=1&docid=206381&orggroupid=2&pageid=27160)
- [Luật Giao dịch điện tử số 20/2023/QH15](https://vanban.chinhphu.vn/?classid=1&docid=208421&pageid=27160&typegroupid=3)
- [Luật Bảo vệ quyền lợi người tiêu dùng số 19/2023/QH15](https://vanban.chinhphu.vn/?classid=1&docid=208363&orggroupid=1&pageid=27160)
- [Nghị định 17/2023/NĐ-CP](https://vanban.chinhphu.vn/?classid=1&docid=207842&pageid=27160&typegroupid=4)
- [Văn bản hợp nhất Luật Sở hữu trí tuệ hiện hành](https://vanban.chinhphu.vn/?classid=0&docid=215309&pageid=27160)

Nguồn chính thức xác nhận danh tính, hiệu lực và bản đính kèm của văn bản.
Việc so khớp nội dung chi tiết dùng exact text đã ingest từ các file trong
`data/keep/`; không suy diễn từ tên Điều.

### Classification rules

- **Confirmed correct**: provision trực tiếp chứa quy tắc hoặc yếu tố mà query
  hỏi; không cần thêm một fact quan trọng chưa có trong query.
- **Likely incorrect**: provision điều chỉnh hành vi/mục đích/hậu quả khác và
  thiếu một điều kiện vật chất không thể suy ra an toàn từ query.
- **Ambiguous**: có quan hệ pháp lý đáng kể nhưng việc áp dụng phụ thuộc fact
  chưa nêu, cách hiểu annotation positive-vs-rule-out, hoặc vấn đề pháp lý chưa
  đủ rõ. Phải chuyển legal reviewer.

## Full adjudication — all 44 expected provisions

Trace notation:

- `RET rN`: đã vào retrieval top-10 ở rank N;
- `DOMAIN`, `CAND`, `BEHAVIOR`, `CE`, `TOPK`: first failed stage;
- trace là bằng chứng kỹ thuật, không được dùng để quyết định tính đúng pháp lý.

| # | Testcase | Query | Expected document | Article | Clause | Point | Trace | Adjudication | Evidence |
|---:|---|---|---|---:|---:|---:|---|---|---|
| 1 | `deepfake_easy_001` | Dùng AI tạo video deepfake từ hình ảnh người nổi tiếng để quảng cáo sản phẩm khi chưa được họ đồng ý có vi phạm không? | Luật An ninh mạng 2025 | 7 | 2 | g | RET r1 | Confirmed | Điểm g trực tiếp cấm dùng AI/công nghệ mới giả mạo video, hình ảnh, giọng nói của người khác trái pháp luật. |
| 2 | `deepfake_easy_001` | Dùng AI tạo video deepfake từ hình ảnh người nổi tiếng để quảng cáo sản phẩm khi chưa được họ đồng ý có vi phạm không? | Luật An ninh mạng 2025 | 13 | 3 | h | CAND | Ambiguous | Có hành vi giả mạo hình ảnh, nhưng điểm h còn yêu cầu ảnh hưởng uy tín, danh dự, nhân phẩm; query chỉ nêu thiếu đồng ý và quảng cáo. |
| 3 | `deepfake_medium_001` | Một nhãn hàng giả giọng nói và khuôn mặt của ca sĩ bằng AI, đăng video lên mạng để bán hàng dù ca sĩ không cho phép. Hành vi nào cần được xem xét? | Luật An ninh mạng 2025 | 7 | 2 | g | DOMAIN | Confirmed | Khớp trực tiếp AI giả mạo video/hình ảnh/giọng nói và đăng tải. |
| 4 | `deepfake_medium_001` | Một nhãn hàng giả giọng nói và khuôn mặt của ca sĩ bằng AI, đăng video lên mạng để bán hàng dù ca sĩ không cho phép. Hành vi nào cần được xem xét? | Luật An ninh mạng 2025 | 13 | 3 | h | DOMAIN | Ambiguous | Giả mạo hình ảnh/giọng nói có mặt, nhưng hậu quả danh dự/uy tín không được nêu. |
| 5 | `deepfake_hard_001` | Công ty thu thập ảnh khách hàng để huấn luyện AI, tạo video deepfake quảng cáo rồi công khai video mà không xin đồng ý. Công ty có vi phạm và phải xử lý dữ liệu thế nào? | Luật Bảo vệ dữ liệu cá nhân 2025 | 28 | 1 | — | BEHAVIOR | Confirmed | Khoản 1 điều chỉnh việc thu thập, sử dụng, chuyển giao dữ liệu cá nhân để kinh doanh dịch vụ quảng cáo và yêu cầu bảo đảm quyền chủ thể dữ liệu. |
| 6 | `deepfake_hard_001` | Công ty thu thập ảnh khách hàng để huấn luyện AI, tạo video deepfake quảng cáo rồi công khai video mà không xin đồng ý. Công ty có vi phạm và phải xử lý dữ liệu thế nào? | Luật An ninh mạng 2025 | 13 | 3 | h | CAND | Ambiguous | Có giả mạo/công khai hình ảnh nhưng chưa có fact về ảnh hưởng uy tín, danh dự hoặc nhân phẩm. |
| 7 | `personal_data_easy_001` | Doanh nghiệp dùng số điện thoại khách hàng để gửi quảng cáo khi chưa có sự đồng ý thì có được phép không? | Luật Bảo vệ dữ liệu cá nhân 2025 | 28 | 1 | — | TOPK | Confirmed | Điều 28 trực tiếp điều chỉnh dữ liệu khách hàng dùng cho quảng cáo và dẫn chiếu quyền của chủ thể dữ liệu. |
| 8 | `personal_data_medium_001` | Bên kiểm soát chuyển dữ liệu cá nhân của người dùng Việt Nam sang máy chủ ở nước ngoài thì phải đáp ứng yêu cầu nào? | Luật Bảo vệ dữ liệu cá nhân 2025 | 20 | — | — | RET r1 | Confirmed | Điều 20 trực tiếp quy định chuyển dữ liệu cá nhân xuyên biên giới, gồm chuyển sang hệ thống lưu trữ ngoài Việt Nam và hồ sơ đánh giá tác động. |
| 9 | `personal_data_hard_001` | Nền tảng vừa quyết định mục đích xử lý dữ liệu vừa thuê nhà cung cấp lưu trữ. Hãy xác định vai trò của các bên và nghĩa vụ khi chia sẻ dữ liệu. | Luật Bảo vệ dữ liệu cá nhân 2025 | 2 | — | — | DOMAIN | Confirmed | Điều 2 định nghĩa bên kiểm soát, bên xử lý, bên kiểm soát và xử lý, bên thứ ba. |
| 10 | `cyber_attack_easy_001` | Một người dùng mật khẩu lấy cắp để đăng nhập trái phép vào tài khoản của người khác bị xử lý thế nào? | Nghị định 15/2020/NĐ-CP | 80 | — | — | RET r2 | Confirmed | Điều 80 quy định trực tiếp hành vi trộm cắp/sử dụng mật khẩu và truy cập trái phép vào mạng hoặc thiết bị số. |
| 11 | `cyber_attack_medium_001` | Kẻ tấn công xâm nhập tài khoản quản trị website và chiếm quyền điều khiển hệ thống để lấy dữ liệu thì căn cứ nào điều chỉnh? | Nghị định 15/2020/NĐ-CP | 81 | — | — | CE | Likely incorrect | Điều 81 yêu cầu sử dụng mạng nhằm chiếm đoạt tài sản; query chỉ nêu chiếm quyền hệ thống/lấy dữ liệu, không có tài sản hoặc mục đích chiếm đoạt tài sản. |
| 12 | `cyber_attack_medium_001` | Kẻ tấn công xâm nhập tài khoản quản trị website và chiếm quyền điều khiển hệ thống để lấy dữ liệu thì căn cứ nào điều chỉnh? | Luật An ninh mạng 2025 | 13 | 3 | h | CAND | Likely incorrect | Điểm h điều chỉnh mạo danh/giả mạo thông tin, hình ảnh, giọng nói gây tổn hại danh dự; query là truy cập trái phép và chiếm quyền hệ thống. |
| 13 | `cyber_attack_hard_001` | Nhóm tấn công khai thác lỗ hổng, chiếm quyền máy chủ của hệ thống thông tin quan trọng rồi sao chép dữ liệu. Phân tích hành vi và chế tài có thể áp dụng. | Luật An ninh mạng 2025 | 13 | 3 | h | CAND | Likely incorrect | Không có hành vi mạo danh/giả mạo hoặc hậu quả danh dự theo điểm h. |
| 14 | `cyber_attack_hard_001` | Nhóm tấn công khai thác lỗ hổng, chiếm quyền máy chủ của hệ thống thông tin quan trọng rồi sao chép dữ liệu. Phân tích hành vi và chế tài có thể áp dụng. | Nghị định 15/2020/NĐ-CP | 81 | — | — | CAND | Likely incorrect | Sao chép dữ liệu không tự động đồng nghĩa chiếm đoạt tài sản; query không nêu yếu tố mục đích/tài sản của Điều 81. |
| 15 | `sql_injection_easy_001` | Dùng SQL Injection khai thác lỗ hổng website và tải xuống cơ sở dữ liệu có phải là truy cập trái phép không? | Luật An ninh mạng 2025 | 13 | 3 | h | CAND | Likely incorrect | SQL injection/truy cập trái phép không thuộc hành vi mạo danh, giả mạo thông tin, hình ảnh, giọng nói tại điểm h. |
| 16 | `sql_injection_medium_001` | Lập trình viên thử SQL Injection trên website của khách hàng khi chưa được ủy quyền, đọc bảng tài khoản nhưng chưa phát tán dữ liệu. Trách nhiệm nào cần xem xét? | Luật An ninh mạng 2025 | 13 | 3 | h | CAND | Likely incorrect | Không có giả mạo danh tính/hình ảnh/giọng nói hoặc hậu quả danh dự. |
| 17 | `sql_injection_medium_001` | Lập trình viên thử SQL Injection trên website của khách hàng khi chưa được ủy quyền, đọc bảng tài khoản nhưng chưa phát tán dữ liệu. Trách nhiệm nào cần xem xét? | Nghị định 15/2020/NĐ-CP | 81 | — | — | RET r5 | Likely incorrect | Điều 81 yêu cầu mục đích chiếm đoạt tài sản; đọc bảng tài khoản khi chưa được phép chưa đáp ứng yếu tố đó. |
| 18 | `sql_injection_hard_001` | Nhân viên vượt phạm vi quyền được cấp, dùng blind SQL Injection trích xuất từng phần cơ sở dữ liệu khách hàng và bán cho bên thứ ba. Hãy xác định các nhóm hành vi pháp lý. | Luật An ninh mạng 2025 | 13 | 3 | h | CAND | Likely incorrect | Hành vi là vượt quyền, trích xuất và bán dữ liệu; điểm h là mạo danh/giả mạo gây ảnh hưởng danh dự. |
| 19 | `sql_injection_hard_001` | Nhân viên vượt phạm vi quyền được cấp, dùng blind SQL Injection trích xuất từng phần cơ sở dữ liệu khách hàng và bán cho bên thứ ba. Hãy xác định các nhóm hành vi pháp lý. | Luật Bảo vệ dữ liệu cá nhân 2025 | 20 | — | — | DOMAIN | Likely incorrect | Điều 20 chỉ điều chỉnh chuyển dữ liệu xuyên biên giới; “bên thứ ba” trong query không được xác định ở nước ngoài hoặc dùng hạ tầng ngoài Việt Nam. |
| 20 | `malware_easy_001` | Hệ thống thông tin phải thực hiện biện pháp nào để phát hiện và loại bỏ mã độc trong phần cứng, phần mềm? | Nghị định 53/2022/NĐ-CP | 10 | — | — | RET r2 | Confirmed | Điều 10 trực tiếp yêu cầu kiểm tra phần cứng/phần mềm để phát hiện điểm yếu, lỗ hổng, mã độc và xử lý sản phẩm đã bị cảnh báo. |
| 21 | `malware_medium_001` | Tổ chức nghiên cứu mã độc và lỗ hổng mạng để xây dựng giải pháp phòng chống có thuộc hoạt động nghiên cứu an ninh mạng không? | Luật An ninh mạng 2025 | 36 | — | — | CE | Confirmed | Điều 36 liệt kê nghiên cứu phần mềm bảo vệ, thẩm định lỗ hổng/phần mềm độc hại và giải quyết nguy cơ an ninh mạng. |
| 22 | `malware_hard_001` | Tin tặc cài mã độc vào hệ thống thông tin quan trọng, duy trì quyền truy cập và gửi dữ liệu ra máy chủ bên ngoài. Cần truy xuất căn cứ về tấn công và bảo vệ hệ thống nào? | Luật An ninh mạng 2025 | 13 | 3 | h | CAND | Likely incorrect | Cài mã độc, duy trì truy cập và exfiltration không phải mạo danh/giả mạo gây ảnh hưởng danh dự theo điểm h. |
| 23 | `malware_hard_001` | Tin tặc cài mã độc vào hệ thống thông tin quan trọng, duy trì quyền truy cập và gửi dữ liệu ra máy chủ bên ngoài. Cần truy xuất căn cứ về tấn công và bảo vệ hệ thống nào? | Nghị định 53/2022/NĐ-CP | 11 | — | — | TOPK | Confirmed | Query hỏi cả căn cứ bảo vệ hệ thống; Điều 11 quy định môi trường, phân vùng mạng, kiểm soát truy cập, phát hiện/ngăn chặn xâm nhập và ứng phó tấn công. |
| 24 | `ai_copyright_easy_001` | Dùng tác phẩm có bản quyền của người khác làm dữ liệu đầu vào cho công cụ AI có liên quan đến quyền tài sản nào của tác giả? | VBHN Luật Sở hữu trí tuệ (`2023_361 + 362_11-VBHN-VPQH`) | 20 | — | — | RET r7 | Confirmed | Điều 20 quy định quyền làm tác phẩm phái sinh và quyền sao chép; đây là các quyền tài sản mà việc tạo bản sao làm dữ liệu đầu vào có thể liên quan. |
| 25 | `ai_copyright_medium_001` | Sao chép một phần tác phẩm để nghiên cứu, huấn luyện mô hình AI có thể thuộc trường hợp ngoại lệ không xâm phạm quyền tác giả nào? | VBHN Luật Sở hữu trí tuệ (`2023_361 + 362_11-VBHN-VPQH`) | 25 | — | — | RET r2 | Confirmed | Điều 25 quy định ngoại lệ sao chép hợp lý để nghiên cứu/học tập, kèm giới hạn mục đích thương mại và điều kiện sử dụng. |
| 26 | `ai_copyright_hard_001` | Công ty sao chép hàng nghìn bản ghi có bản quyền để huấn luyện AI thương mại rồi phân phối sản phẩm mô phỏng nội dung gốc. Hãy phân biệt ngoại lệ và hành vi xâm phạm. | VBHN Luật Sở hữu trí tuệ (`2023_361 + 362_11-VBHN-VPQH`) | 35 | — | — | TOPK | Ambiguous | Điều 35 điều chỉnh quyền liên quan của người biểu diễn, nhà sản xuất bản ghi âm/ghi hình và tổ chức phát sóng. “Bản ghi” trong query không đủ xác định đây là đối tượng quyền liên quan thay vì tác phẩm nói chung. |
| 27 | `ai_copyright_hard_001` | Công ty sao chép hàng nghìn bản ghi có bản quyền để huấn luyện AI thương mại rồi phân phối sản phẩm mô phỏng nội dung gốc. Hãy phân biệt ngoại lệ và hành vi xâm phạm. | Nghị định 17/2023/NĐ-CP | 28 | — | — | CAND | Ambiguous | Điều 28 là điều kiện trích dẫn hợp lý. Mass copying để huấn luyện thương mại không giống trích dẫn, nhưng provision có thể được dùng như rule-out nếu benchmark coi negative applicability là relevant. |
| 28 | `advertising_easy_001` | Có được dùng dữ liệu cá nhân của khách hàng để quảng cáo trực tiếp khi họ chưa đồng ý không? | Luật Bảo vệ dữ liệu cá nhân 2025 | 28 | 1 | — | RET r2 | Confirmed | Khớp trực tiếp xử lý dữ liệu cá nhân để quảng cáo và quyền chủ thể dữ liệu. |
| 29 | `advertising_medium_001` | Người bán quảng cáo sai công dụng sản phẩm khiến người tiêu dùng hiểu nhầm thì hành vi bị cấm nào được áp dụng? | Luật Bảo vệ quyền lợi người tiêu dùng 2023 | 10 | — | — | RET r1 | Confirmed | Điều 10 cấm cung cấp thông tin sai lệch/không chính xác gây nhầm lẫn về sản phẩm, hàng hóa, dịch vụ. |
| 30 | `advertising_hard_001` | Mạng xã hội phân tích hồ sơ người dùng để nhắm mục tiêu quảng cáo, đồng thời để người bán đăng nội dung quảng cáo gây nhầm lẫn. Nền tảng phải tuân thủ những nhóm nghĩa vụ nào? | Luật Bảo vệ dữ liệu cá nhân 2025 | 28 | 1 | — | DOMAIN | Confirmed | Phân tích hồ sơ để quảng cáo thuộc xử lý dữ liệu cá nhân trong hoạt động quảng cáo. |
| 31 | `advertising_hard_001` | Mạng xã hội phân tích hồ sơ người dùng để nhắm mục tiêu quảng cáo, đồng thời để người bán đăng nội dung quảng cáo gây nhầm lẫn. Nền tảng phải tuân thủ những nhóm nghĩa vụ nào? | Luật Bảo vệ quyền lợi người tiêu dùng 2023 | 10 | — | — | CE | Confirmed | Nội dung quảng cáo gây nhầm lẫn khớp hành vi bị cấm tại Điều 10. |
| 32 | `consumer_easy_001` | Người bán cung cấp thông tin sai về chất lượng hàng hóa cho người tiêu dùng có thuộc hành vi bị cấm không? | Luật Bảo vệ quyền lợi người tiêu dùng 2023 | 10 | — | — | RET r1 | Confirmed | Điều 10 trực tiếp cấm thông tin sai lệch/không chính xác về hàng hóa. |
| 33 | `consumer_medium_001` | Sản phẩm có khuyết tật gây thiệt hại cho người mua thì tổ chức kinh doanh phải bồi thường khi nào và được miễn trách nhiệm trong trường hợp nào? | Luật Bảo vệ quyền lợi người tiêu dùng 2023 | 34 | — | — | DOMAIN | Confirmed | Điều 34 quy định trách nhiệm bồi thường thiệt hại do sản phẩm, hàng hóa có khuyết tật. |
| 34 | `consumer_medium_001` | Sản phẩm có khuyết tật gây thiệt hại cho người mua thì tổ chức kinh doanh phải bồi thường khi nào và được miễn trách nhiệm trong trường hợp nào? | Luật Bảo vệ quyền lợi người tiêu dùng 2023 | 35 | — | — | DOMAIN | Confirmed | Điều 35 trực tiếp quy định các trường hợp miễn trách nhiệm bồi thường. |
| 35 | `consumer_hard_001` | Sàn thương mại điện tử đưa vào điều khoản mẫu cho phép đơn phương thay đổi giá và loại trừ toàn bộ trách nhiệm với người mua. Điều khoản nào cần bị kiểm tra? | Luật Bảo vệ quyền lợi người tiêu dùng 2023 | 25 | — | — | DOMAIN | Confirmed | Điều 25 cấm điều khoản loại trừ trách nhiệm và các điều khoản bất lợi trong hợp đồng theo mẫu/điều kiện giao dịch chung. |
| 36 | `network_security_easy_001` | Chủ quản hệ thống phải kiểm tra phần cứng và phần mềm để phát hiện yếu tố gây mất an ninh mạng theo quy định nào? | Nghị định 53/2022/NĐ-CP | 10 | — | — | RET r2 | Confirmed | Điều 10 trực tiếp quy định kiểm tra thiết bị, phần cứng, phần mềm để phát hiện lỗ hổng, mã độc, phần cứng độc hại. |
| 37 | `network_security_medium_001` | Hệ thống thông tin quan trọng về an ninh quốc gia phải áp dụng biện pháp bảo vệ và giám sát nào? | Nghị định 53/2022/NĐ-CP | 11 | — | — | DOMAIN | Confirmed | Điều 11 có tiêu đề và nội dung trực tiếp về biện pháp kỹ thuật giám sát, bảo vệ an ninh mạng. |
| 38 | `network_security_hard_001` | Doanh nghiệp nghiên cứu giải pháp phát hiện lỗ hổng, mã độc và phương thức tấn công để bảo vệ hệ thống. Hoạt động nghiên cứu này được quy định ra sao? | Luật An ninh mạng 2025 | 36 | — | — | CE | Confirmed | Điều 36 trực tiếp liệt kê nghiên cứu giải pháp, lỗ hổng, phần mềm độc hại và nguy cơ an ninh mạng. |
| 39 | `electronic_transactions_easy_001` | Thông báo điện tử trong giao kết và thực hiện hợp đồng có giá trị pháp lý như thế nào? | Luật Giao dịch điện tử 2023 | 38 | — | — | RET r1 | Confirmed | Điều 38 quy định thông báo bằng thông điệp dữ liệu có giá trị pháp lý như thông báo bằng văn bản giấy. |
| 40 | `electronic_transactions_medium_001` | Hợp đồng được hệ thống tự động tạo và gửi nhưng có lỗi nhập liệu thì bên nhập sai có quyền rút lại phần dữ liệu đó không? | Luật Giao dịch điện tử 2023 | 18 | — | — | CE | Confirmed | Điều 18 điều chỉnh gửi/nhận tự động và dẫn chiếu trực tiếp các Điều 14–17; query có hệ thống tự động. |
| 41 | `electronic_transactions_medium_001` | Hợp đồng được hệ thống tự động tạo và gửi nhưng có lỗi nhập liệu thì bên nhập sai có quyền rút lại phần dữ liệu đó không? | Luật Giao dịch điện tử 2023 | 14 | — | — | RET r1 | Confirmed | Khoản 3 Điều 14 trực tiếp cho phép rút thông tin nhập lỗi khi hệ thống tự động không cho cơ hội sửa, kèm điều kiện. |
| 42 | `electronic_transactions_hard_001` | Hai doanh nghiệp tranh chấp thời điểm và địa điểm gửi, nhận thông điệp dữ liệu vì hệ thống không gửi xác nhận. Cần áp dụng các điều nào để xác định hiệu lực giao dịch? | Luật Giao dịch điện tử 2023 | 15 | — | — | RET r5 | Confirmed | Điều 15 trực tiếp quy định thời điểm, địa điểm gửi thông điệp dữ liệu. |
| 43 | `electronic_transactions_hard_001` | Hai doanh nghiệp tranh chấp thời điểm và địa điểm gửi, nhận thông điệp dữ liệu vì hệ thống không gửi xác nhận. Cần áp dụng các điều nào để xác định hiệu lực giao dịch? | Luật Giao dịch điện tử 2023 | 16 | — | — | RET r3 | Confirmed | Điều 16 quy định việc nhận và nghĩa vụ/thông báo xác nhận nhận thông điệp dữ liệu. |
| 44 | `electronic_transactions_hard_001` | Hai doanh nghiệp tranh chấp thời điểm và địa điểm gửi, nhận thông điệp dữ liệu vì hệ thống không gửi xác nhận. Cần áp dụng các điều nào để xác định hiệu lực giao dịch? | Luật Giao dịch điện tử 2023 | 17 | — | — | RET r2 | Confirmed | Điều 17 trực tiếp quy định thời điểm, địa điểm nhận thông điệp dữ liệu. |

## Confirmed correct

29 annotations có direct textual/legal fit. Trong số này:

- 15 đang được retrieve top-10;
- 4 bị CE threshold loại;
- 2 qua CE/threshold nhưng mất ở top-k;
- 1 bị Behavior Gate loại;
- 7 bị Domain hoặc candidate generation chặn.

Điều này xác nhận Retrieval Recall vẫn là vấn đề thật ngay cả khi loại annotation
nhiễu khỏi sensitivity analysis.

## Likely incorrect

| Failure pattern | Count | Provisions |
|---|---:|---|
| Điều 13(3)(h) gán cho hacking/SQL/malware | 6 | Rows 12, 13, 15, 16, 18, 22 |
| Điều 81 thiếu mục đích/yếu tố chiếm đoạt tài sản | 3 | Rows 11, 14, 17 |
| Điều 20 thiếu yếu tố xuyên biên giới | 1 | Row 19 |

Sáu nhãn Điều 13(3)(h) chiếm 21.43% của 28 retrieval losses. Nếu dùng chúng
làm target, Candidate Generation sẽ bị thúc đẩy tìm một đoạn luật semantic
không liên quan đến hacking.

## Ambiguous

| Pattern | Count | Required adjudication |
|---|---:|---|
| Deepfake nhưng chưa nêu hậu quả danh dự/uy tín | 3 | Quyết định có cho phép suy ra hậu quả hay benchmark phải nêu fact này |
| “Bản ghi” chưa xác định đối tượng quyền liên quan | 1 | Xác định loại bản ghi/tác phẩm trong fact pattern |
| Quy tắc trích dẫn được dùng như negative/rule-out relevance | 1 | Xác định benchmark chấm positive applicability hay cả rule-out provision |

## Oracle upper bound and expected benchmark gain

Benchmark vẫn giữ nguyên 44 labels; các con số chính thức không được tính lại.
Sensitivity analysis chỉ dùng để tránh lựa chọn CE strategy dựa trên nhãn nhiễu:

| Scope | Baseline | CE ranking + threshold oracle |
|---|---:|---:|
| Official benchmark | 16/44; macro R@10 41.67% | 24/44; macro R@10 60.00% |
| Confirmed-only sensitivity | 15/29; macro trên 24 case có confirmed label 52.08% | 21/29; macro 72.92% |

Confirmed-only sensitivity không phải evaluation metric mới và không thay
benchmark runner. Nó chỉ chứng minh rằng CE recovery còn upper bound thật sau
khi loại likely-incorrect/ambiguous labels khỏi phép chọn thiết kế.

Ground Truth Adjudication tự nó có expected benchmark gain bằng 0 vì không sửa
benchmark hoặc pipeline. Bảng oracle chỉ là upper bound cho thiết kế CE tiếp
theo; không được ghi nhận như kết quả benchmark mới.

## Decision and next action

Không sửa benchmark trong task này.

Trước mọi Retrieval/CE implementation:

1. Legal reviewer adjudicate 10 likely-incorrect và 5 ambiguous annotations.
2. Mỗi annotation phải được gắn một trong các relation:
   `directly_applicable`, `conditional`, `rule_out`, `incorrect`.
3. Giữ benchmark hiện tại làm frozen historical control; mọi bản sửa sau này
   phải là dataset version mới, có changelog và reviewer.
4. Không dùng 15 annotations chưa adjudicate làm tuning target.

## Rollback plan

Không có code, benchmark hoặc index mutation để rollback. Nếu report bị bác bởi
legal reviewer, rollback là thu hồi classification tương ứng trong tài liệu;
authoritative benchmark và mọi baseline artifact vẫn nguyên trạng.
