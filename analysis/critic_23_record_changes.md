# Biên bản thay đổi sau kiểm toán artefact Critic

Tài liệu nguồn: `/Users/nguyengiahuy/Desktop/KLTN_dcs.docx`  
Tài liệu đầu ra: `/Users/nguyengiahuy/Project/LEGAL_IT_CHATBOT/KLTN_Final_CriticAuditFix.docx`

## 1. Tóm tắt khóa luận

- **Vị trí:** Đoạn 46, phần Tóm tắt.
- **Câu cũ:** “Nguyên nhân là đầu ra hiện có của Critic chỉ bao gồm 23 trên tổng số 301 bản ghi, chưa đủ để tái tạo toàn bộ quá trình chấm điểm.”
- **Câu mới:** “Các chỉ số LCR, RAGAS và mức tiêu thụ tài nguyên được giữ ở trạng thái số liệu tổng hợp theo báo cáo của phiên bản bài báo cuối cùng vì repository hiện chưa lưu đầy đủ raw output Critic cho 301 câu và raw scoring artifacts cần thiết để tái tính độc lập toàn bộ bảng. Số record của artefact cục bộ không được diễn giải thành phạm vi thực thi của toàn bộ thí nghiệm.” Đồng thời sửa tên chỉ số từ “Legal Coverage Rate” thành “Legal Completeness Rate”.
- **Lý do:** Số dòng của tệp đang lưu chỉ chứng minh phạm vi của artefact cục bộ, không chứng minh toàn bộ thí nghiệm chỉ chạy từng ấy câu. Tên đầy đủ của LCR cần thống nhất với định nghĩa trong khóa luận.
- **Bằng chứng:** `data/eval_results_critic_p0_base_qwen25_20260731.jsonl` có 23 JSONL record đúng schema evaluation; repository không có log kết thúc hoặc provenance đủ để xác định phạm vi thực thi của toàn bộ run. Chi tiết tại `analysis/critic_23_record_audit.md`.

## 2. Thiết kế đánh giá

- **Vị trí:** Đoạn 269, Mục 3.9.
- **Câu cũ:** “Tệp kết quả ngày 31/07/2026 có 301 bản ghi cho Naive và 301 bản ghi cho Article Expansion nhưng chỉ 23 bản ghi cho Critic; vì vậy, phần kết quả tách rõ chỉ số có thể tái tính từ đầu ra chi tiết Naive với chỉ số tổng hợp do tài liệu nguồn báo cáo, và không thực hiện kiểm định cặp.”
- **Câu mới:** “Tệp kết quả ngày 31/07/2026 có 301 bản ghi cho Naive và 301 bản ghi cho Article Expansion; artefact Critic hiện lưu là một tập con có đúng schema evaluation. Repository không có log đủ để kết luận toàn bộ run Critic chỉ xử lý số câu tương ứng với tập con này. Vì chưa có raw Critic đủ 301 câu và raw judge/checkpoint đầy đủ, phần kết quả tách rõ chỉ số có thể tái tính từ artefact hiện có với chỉ số tổng hợp do tài liệu nguồn báo cáo; kiểm định cặp không được thực hiện do thiếu đầy đủ các cặp kết quả và repeated runs.”
- **Lý do:** Cần tách “số record hiện lưu” khỏi “số câu đã chạy trong thí nghiệm”.
- **Bằng chứng:** Script `scripts/run_evaluation.py` ghi một dòng cho mỗi ca thành công và bỏ qua ca gặp ngoại lệ; không có completion log của run Critic. Naive và Expansion có 301 record; artefact Critic hiện lưu có 23 record.

## 3. Phạm vi và khả năng diễn giải kết quả

- **Vị trí:** Đoạn 340, Mục 4.6.
- **Câu cũ:** “Đầu ra chi tiết Base ngày 31/07/2026 hoàn chỉnh ở Naive và Expansion, nhưng Critic dừng ở 23 dòng.”
- **Câu mới:** “Đầu ra chi tiết Base ngày 31/07/2026 hoàn chỉnh ở Naive và Expansion; artefact Critic đang lưu có schema evaluation nhưng chỉ bao phủ một tập con. Repository không có log đủ để xác định phạm vi thực thi của toàn bộ run Critic.”
- **Lý do:** Cụm “Critic dừng” là kết luận về lịch sử thực thi mà artefact hiện có không chứng minh được.
- **Bằng chứng:** Tệp Critic là evaluation output thật, gồm các ID liên tiếp từ `cat1_01` đến `cat2_10`; không tìm thấy log ghi trạng thái dừng hoặc tổng số ca đã hoàn thành.

## 4. Mức tái lập

- **Vị trí:** Đoạn 349, Mục 4.9.
- **Câu cũ:** Đoạn nêu “chưa có ... đầu ra Critic đủ 301 câu” nhưng chưa giới hạn rõ đây là tình trạng lưu trữ trong repository.
- **Câu mới:** Đổi thành “repository chưa lưu ... raw output Critic đủ 301 câu” và bổ sung: “Việc thiếu artefact không được diễn giải thành kết luận về số câu mà thí nghiệm đầy đủ đã thực sự chạy.”
- **Lý do:** Tránh đồng nhất sự vắng mặt của artefact với sự vắng mặt của một lần chạy trong quá khứ.
- **Bằng chứng:** Không tìm thấy raw Critic 301 câu, checkpoint LCR/RAGAS đầy đủ hoặc nhật ký API trong repository; vì vậy chỉ có thể kết luận về khả năng tái lập hiện tại.

## 5. Phân tích nhóm no-gap

- **Vị trí:** Đoạn 363, Mục 5.3.
- **Câu cũ:** “Nhóm no-gap chỉ tăng 1,76 điểm từ Expansion lên Critic; thiếu đầu ra Critic đầy đủ nên chưa xác định được bao nhiêu trường hợp gate/regeneration làm tốt hơn hoặc làm xấu đi một câu vốn đã đủ ngữ cảnh.”
- **Câu mới:** “Nhóm no-gap chỉ tăng 1,76 điểm từ Expansion lên Critic; do repository chưa lưu raw output Critic và judge decisions đủ 301 câu, chưa xác định được bao nhiêu trường hợp gate/regeneration làm tốt hơn hoặc làm xấu đi một câu vốn đã đủ ngữ cảnh.”
- **Lý do:** Nêu đúng loại bằng chứng còn thiếu và phạm vi nơi đã kiểm tra.
- **Bằng chứng:** Artefact Critic cục bộ không bao phủ đủ testset; không có quyết định judge theo từng câu để tái lập phân tích lỗi.

## 6. Hạn chế thống kê

- **Vị trí:** Đoạn 388, Mục 5.9.
- **Câu cũ:** “Đầu ra Critic hiện chỉ có 23 bản ghi. Mọi so sánh là mô tả, không phải bằng chứng về ý nghĩa thống kê.”
- **Câu mới:** “Repository chưa lưu đầy đủ raw output Critic 301 câu, quyết định judge, repeated runs, confidence interval hoặc dữ liệu cần thiết cho paired statistical test. Mọi so sánh là mô tả, không phải bằng chứng về ý nghĩa thống kê; số record của artefact Critic cục bộ không cho phép suy ra phạm vi thực thi của toàn bộ thí nghiệm.”
- **Lý do:** Lý do khoa học cho việc không kiểm định cặp là thiếu đầy đủ cặp kết quả và repeated runs, không phải riêng con số 23.
- **Bằng chứng:** Không tìm thấy repeated-run outputs, confidence interval, paired-test data hoặc full per-case Critic/judge artifacts.

## 7. Kết luận về trạng thái số liệu

- **Vị trí:** Đoạn 400, Mục 6.1.
- **Câu cũ:** “Đầu ra Critic chỉ có 23/301 dòng, nên LCR/RAGAS tiếp tục được ghi là số liệu tổng hợp `REPORTED`.”
- **Câu mới:** “Do repository chưa lưu đầy đủ raw output Critic 301 câu và raw judge/checkpoint cần thiết để tái tính độc lập, LCR/RAGAS tiếp tục được ghi là số liệu tổng hợp `REPORTED`; số record của artefact cục bộ không được dùng để kết luận toàn bộ Critic chỉ chạy từng ấy câu.”
- **Lý do:** Giữ nguyên trạng thái thận trọng `REPORTED`, nhưng gắn trạng thái này với bằng chứng tái tính thực sự còn thiếu.
- **Bằng chứng:** Không có full raw Critic, per-question LCR/RAGAS decisions hoặc checkpoint hoàn chỉnh để tái tính độc lập các số liệu tổng hợp.

## 8. Hướng phát triển reproducibility package

- **Vị trí:** Đoạn 403, Mục 6.2.
- **Câu cũ:** “Hướng ưu tiên thứ nhất là hoàn thiện reproducibility package: chạy đủ 301 câu cho ba mode Base...”
- **Câu mới:** “Hướng ưu tiên thứ nhất là hoàn thiện reproducibility package: phát hành đầy đủ raw artifacts theo từng câu cho cả ba mode Base trên bộ 301 câu...”
- **Lý do:** Repository hiện tại không đủ bằng chứng để kết luận Critic chưa từng chạy đủ 301 câu; hành động cần thiết có thể khẳng định là phát hành đầy đủ artefact.
- **Bằng chứng:** Naive và Expansion có raw output 301 câu; Critic không có full raw output trong repository hiện tại. Nguyên nhân và lịch sử của phần thiếu: **Chưa đủ bằng chứng để xác minh.**

## Nội dung được giữ nguyên sau kiểm toán

Các kịch bản `cat1_02` và `cat1_03` tại Mục 4.5 không bị xóa hoặc hạ trạng thái, vì cả hai ID đều có trong artefact Critic và mỗi record chứa câu trả lời, ngữ cảnh truy xuất, token usage, draft response và critic report. Chúng là bằng chứng cấp tình huống hợp lệ, nhưng không đại diện cho toàn bộ 301 câu.
