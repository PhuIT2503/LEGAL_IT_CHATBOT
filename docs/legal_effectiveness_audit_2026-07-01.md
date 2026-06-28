# Audit hiệu lực văn bản pháp luật trong `data/`

Ngày rà soát: 2026-06-11. Giả định mốc "1/07" là **01/07/2026** vì hiện tại đang là tháng 06/2026.

Mục tiêu cho RAG/chatbot: không xóa vật lý ngay lập tức; nên loại khỏi corpus chính hoặc gắn nhãn `expired/superseded` để tránh trả lời bằng văn bản hết hiệu lực, bị thay thế, trùng lặp hoặc không chính thức.

## Kết Luận Nhanh

### Cần bỏ khỏi corpus chính ngay

| File | Lý do | Đề xuất thay thế |
| --- | --- | --- |
| `data/Nghị Định 13 2023 NĐ-CP.docx` | Nghị định 356/2025/NĐ-CP có hiệu lực từ 01/01/2026 và quy định chi tiết Luật Bảo vệ dữ liệu cá nhân. Nguồn công khai nêu Nghị định 13/2023/NĐ-CP hết hiệu lực từ 01/01/2026. | Thêm `Nghị định 356/2025/NĐ-CP` vào `data/keep`. |
| `data/Nghị Định 130 2018 NĐ-CP.docx` | Nghị định 23/2025/NĐ-CP về chữ ký điện tử và dịch vụ tin cậy có hiệu lực từ 10/04/2025; thay khung chữ ký số cũ. | Thêm `Nghị định 23/2025/NĐ-CP` vào `data/keep`. |
| `data/29.pdf` | Luật Khoa học và Công nghệ 2013, số 29/2013/QH13, đã bị thay thế bởi Luật Khoa học, Công nghệ và Đổi mới sáng tạo 2025, số 93/2025/QH15, hiệu lực từ 01/10/2025. | Bỏ khỏi corpus chính; thêm Luật 93/2025/QH15 và các nghị định hướng dẫn năm 2025. |
| `data/131-nd.pdf` | Nghị định 131/2013/NĐ-CP đã hết hiệu lực từ 15/02/2026 theo Nghị định 341/2025/NĐ-CP. | Bỏ khỏi corpus chính; thay bằng `Nghị định 341/2025/NĐ-CP`. |
| `data/08-bvhttdl.pdf` | File trong repo là bản không có số/không ngày ký (`Số: /2023/TT-BVHTTDL`), không nên coi là văn bản chính thức. | Không ingest file này; nếu cần nội dung mẫu đăng ký quyền tác giả thì thay bằng bản chính thức `Thông tư 08/2023/TT-BVHTTDL` và kiểm tra `Thông tư 21/2025/TT-BVHTTDL` sửa đổi. |
| Một trong hai file SHTT: `data/2023_361 + 362_11-VBHN-VPQH.docx` hoặc `data/Luật Sở hữu trí tuệ 2005 (sửa đổi, bổ sung năm 2009, 2019, 2022).docx` | Hai file đều là nội dung Luật Sở hữu trí tuệ hợp nhất/sửa đổi đến 2022, dễ gây trùng embedding và nhiễu kết quả trả lời. | Giữ một bản chuẩn, ưu tiên bản hợp nhất rõ nguồn. |

### Sau 01/07/2026 cần bỏ khỏi corpus chính

| File | Tình trạng sau 01/07/2026 | Đề xuất |
| --- | --- | --- |
| `data/Luật ATTT mạng 2015 (sửa đổi, bổ sung năm 2018).docx` | Hết hiệu lực từ 01/07/2026 theo Điều 44 Luật An ninh mạng 2025. | Bỏ khỏi corpus chính; nếu cần lịch sử thì gắn `expired_after=2026-07-01`. |
| `data/Luật An ninh mạng 2018.docx` | Hết hiệu lực từ 01/07/2026 theo Điều 44 Luật An ninh mạng 2025. | Bỏ khỏi corpus chính; thay bằng `data/keep/Luật An ninh mạng 2025.docx`. |

### Sau 01/07/2026 cần giữ nhưng cập nhật bản hợp nhất

| File | Lý do cần cập nhật |
| --- | --- |
| `data/keep/Luật An ninh mạng 2025.docx` | Từ 01/07/2026 trở thành luật chính thay Luật An toàn thông tin mạng 2015 và Luật An ninh mạng 2018. |
| `data/keep/Luật Viễn thông 2023.docx` | Luật An ninh mạng 2025 sửa/bãi bỏ một số cụm từ trong Luật Viễn thông 2023. Không bỏ, chỉ cập nhật bản hợp nhất. |
| `data/keep/Luật Giao dịch điện tử 2023.docx` | Luật An ninh mạng 2025 sửa/bãi bỏ một số cụm từ. Không bỏ, chỉ cập nhật bản hợp nhất. |
| `data/keep/Luật Dữ liệu 2024.docx` | Luật An ninh mạng 2025 sửa/bãi bỏ một số cụm từ. Không bỏ, chỉ cập nhật bản hợp nhất. |
| `data/keep/Luật Bảo vệ quyền lợi người tiêu dùng 2023.docx` | Luật An ninh mạng 2025 sửa/bãi bỏ một số cụm từ. Không bỏ, chỉ cập nhật bản hợp nhất. |
| `data/keep/Luật Công nghiệp công nghệ số 2025.docx` | Đã có hiệu lực toàn bộ từ 01/01/2026, nhưng cũng bị Luật An ninh mạng 2025 sửa/bãi bỏ cụm từ. |
| `data/keep/Luật CNTT 2006 (sửa đổi, bổ sung bởi Luật Quy hoạch 2017, Luật GDĐT 2023, Luật Viễn thông 2023).docx` | Luật Công nghiệp công nghệ số 2025 bãi bỏ các khoản 9, 10, 11, 12 Điều 4 và Mục 3, Mục 4 Chương III của Luật CNTT. Cần bản hợp nhất mới sau 01/01/2026. |

## Bảng Rà Soát Từng DOCX

| File | Hiệu lực hiện tại 2026-06-11 | Hành động |
| --- | --- | --- |
| `Luật ATTT mạng 2015...docx` | Còn áp dụng đến trước 01/07/2026; sẽ hết hiệu lực từ 01/07/2026. | Bỏ sau 01/07/2026. |
| `Luật An ninh mạng 2018.docx` | Còn áp dụng đến trước 01/07/2026; sẽ hết hiệu lực từ 01/07/2026. | Bỏ sau 01/07/2026. |
| `Luật An ninh mạng 2025.docx` | Chưa có hiệu lực đến 30/06/2026; có hiệu lực từ 01/07/2026. | Giữ, ưu tiên sau mốc 01/07/2026. |
| `Luật Bảo vệ dữ liệu cá nhân 2025.docx` | Có hiệu lực từ 01/01/2026. | Giữ; bổ sung Nghị định 356/2025/NĐ-CP. |
| `Nghị Định 13 2023 NĐ-CP.docx` | Đã bị thay bằng Nghị định 356/2025/NĐ-CP từ 01/01/2026. | Bỏ khỏi corpus chính ngay. |
| `Nghị Định 130 2018 NĐ-CP.docx` | Đã bị thay bằng Nghị định 23/2025/NĐ-CP từ 10/04/2025. | Bỏ khỏi corpus chính ngay. |
| `Luật Công nghiệp công nghệ số 2025.docx` | Có hiệu lực toàn bộ từ 01/01/2026; một số điều đã có hiệu lực từ 01/07/2025. | Giữ, cập nhật theo Luật An ninh mạng 2025. |
| `Luật Dữ liệu 2024.docx` | Có hiệu lực từ 01/07/2025. | Giữ, cập nhật theo Luật An ninh mạng 2025. |
| `Luật Giao dịch điện tử 2023.docx` | Có hiệu lực từ 01/07/2024. | Giữ, cập nhật theo Luật An ninh mạng 2025. |
| `Luật Viễn thông 2023.docx` | Có hiệu lực từ 01/07/2024; một phần từ 01/01/2025. | Giữ, cập nhật theo Luật An ninh mạng 2025. |
| `Luật Bảo vệ quyền lợi người tiêu dùng 2023.docx` | Có hiệu lực từ 01/07/2024. | Giữ, cập nhật theo Luật An ninh mạng 2025. |
| `Luật CNTT 2006...docx` | Còn hiệu lực một phần. | Giữ bản hợp nhất mới; không nên dùng file hiện tại nếu hỏi về công nghiệp CNTT sau 01/01/2026. |
| `Luật Sở hữu trí tuệ 2005...docx` | Còn hiệu lực/sửa đổi đến 2022. | Chỉ giữ một bản để tránh trùng với `2023_361 + 362_11-VBHN-VPQH.docx`. |
| `2023_361 + 362_11-VBHN-VPQH.docx` | Bản hợp nhất Luật Sở hữu trí tuệ. | Giữ một bản, tránh trùng với file SHTT còn lại. |
| `Nghị Định 15 2020...docx` | Còn hiệu lực nhưng file chỉ hợp nhất đến Nghị định 14/2022; Nghị định 211/2025 đã sửa đổi, bổ sung. | Giữ nhưng thay bằng bản hợp nhất mới sau Nghị định 211/2025. |
| `Nghị Định 211 2025 NĐ-CP.docx` | Có hiệu lực từ 09/09/2025. | Giữ. |
| `Nghị Định 53 2022 NĐ-CP.docx` | VBPL ghi còn hiệu lực, ngày có hiệu lực 01/10/2022. | Giữ tạm; sau 01/07/2026 gắn nhãn `under_old_cybersecurity_law`, chờ nghị định thay thế/hướng dẫn Luật An ninh mạng 2025. |
| `Nghị Định 85 2016 NĐ-CP.docx` | VBPL ghi còn hiệu lực, ngày có hiệu lực 01/07/2016. | Giữ tạm; sau 01/07/2026 gắn nhãn `under_old_network_information_security_law`, chờ nghị định thay thế/hướng dẫn Luật An ninh mạng 2025. |
| `Nghị Định 147 2024 NĐ-CP.docx` | Có hiệu lực từ 25/12/2024; thay Nghị định 72/2013 và Nghị định 27/2018, VBPL ghi hết hiệu lực một phần. | Giữ, cần kiểm tra văn bản sửa đổi mới nhất trước ingest. |
| `Nghị Định 52 2013...docx` | Đang là khung thương mại điện tử đã sửa đổi bởi Nghị định 85/2021. | Giữ nếu corpus cần thương mại điện tử; nên thay bằng bản hợp nhất mới nhất nếu có. |
| `Nghị Định 52 2024 NĐ-CP.docx` | Có hiệu lực; thay khung thanh toán không dùng tiền mặt cũ. | Giữ. |
| `Nghị Định 17 2023 NĐ-CP.docx` | Có hiệu lực từ 26/04/2023. | Giữ. |
| `Nghị Định 71 2007 NĐ-CP.docx` | VBPL ghi hết hiệu lực một phần; nội dung gắn với công nghiệp CNTT cũ. | Không bỏ toàn bộ nếu chưa có văn bản thay thế rõ, nhưng nên gắn nhãn `partially_expired/outdated` và đối chiếu Luật Công nghiệp công nghệ số 2025. |

## PDF Đã Định Danh

| File | Định danh sơ bộ | Hành động |
| --- | --- | --- |
| `07.signed.pdf` | Luật Chuyển giao công nghệ 2017, số 07/2017/QH14. | Giữ; VBPL ghi còn hiệu lực, ngày có hiệu lực 01/07/2018. |
| `76.signed.pdf` | Nghị định 76/2018/NĐ-CP hướng dẫn Luật Chuyển giao công nghệ. | Giữ; VBPL ghi còn hiệu lực, ngày có hiệu lực 01/07/2018. |
| `29.pdf` | Luật Khoa học và Công nghệ 2013, số 29/2013/QH13. | Bỏ khỏi corpus chính; đã bị thay thế bởi Luật 93/2025/QH15 từ 01/10/2025. |
| `131-nd.pdf` | Nghị định 131/2013/NĐ-CP về xử phạt VPHC quyền tác giả, quyền liên quan. | Bỏ khỏi corpus chính; Nghị định 341/2025/NĐ-CP có hiệu lực 15/02/2026 làm Nghị định 131/2013 hết hiệu lực. |
| `08-bvhttdl.pdf` | Bản không có số/không ngày ký của thông tư về mẫu đăng ký quyền tác giả, quyền liên quan. | Không ingest file này; thay bằng bản chính thức Thông tư 08/2023/TT-BVHTTDL nếu cần. |
| `65-nd-cp.signed.pdf` | Nghị định 65/2023/NĐ-CP về SHTT/SHTT công nghiệp. | Giữ. |
| `23-bkhcn.pdf` | Thông tư 23/2023/TT-BKHCN. | Giữ nhưng phải patch theo Quyết định 4448/QĐ-BKHCN ngày 26/12/2025. |
| `4448-bkhcn.pdf` | Quyết định 4448/QĐ-BKHCN bãi bỏ một phần Thông tư 23/2023/TT-BKHCN. | Không ingest như văn bản hỏi đáp độc lập; dùng làm amendment patch cho `23-bkhcn.pdf`. |

## Kết Quả Tách Thư Mục

### `data/keep`

Nhóm này dùng cho corpus chính và pipeline ingest:

- `07.signed.pdf`
- `76.signed.pdf`
- `65-nd-cp.signed.pdf`
- `23-bkhcn.pdf`
- `4448-bkhcn.pdf`
- `2023_361 + 362_11-VBHN-VPQH.docx`
- `Luật An ninh mạng 2025.docx`
- `Luật Bảo vệ dữ liệu cá nhân 2025.docx`
- `Luật Bảo vệ quyền lợi người tiêu dùng 2023.docx`
- `Luật CNTT 2006...docx`
- `Luật Công nghiệp công nghệ số 2025.docx`
- `Luật Dữ liệu 2024.docx`
- `Luật Giao dịch điện tử 2023.docx`
- `Luật Viễn thông 2023.docx`
- `Nghị Định 147 2024 NĐ-CP.docx`
- `Nghị Định 15 2020...docx`
- `Nghị Định 17 2023 NĐ-CP.docx`
- `Nghị Định 211 2025 NĐ-CP.docx`
- `Nghị Định 52 2013...docx`
- `Nghị Định 52 2024 NĐ-CP.docx`
- `Nghị Định 53 2022 NĐ-CP.docx`
- `Nghị Định 71 2007 NĐ-CP.docx`
- `Nghị Định 85 2016 NĐ-CP.docx`

### `data/remove`

Nhóm này không dùng cho corpus chính:

- `08-bvhttdl.pdf`
- `29.pdf`
- `131-nd.pdf`
- `Luật ATTT mạng 2015...docx`
- `Luật An ninh mạng 2018.docx`
- `Luật Sở hữu trí tuệ 2005...docx`
- `Nghị Định 13 2023 NĐ-CP.docx`
- `Nghị Định 130 2018 NĐ-CP.docx`

## Nguồn Đã Đối Chiếu

- Luật An ninh mạng 2025, Điều 43-45: https://vbpl.vn/soctrang/Pages/vbpq-toanvan.aspx?ItemID=187039&Keyword=
- Nghị định 356/2025/NĐ-CP trên Cổng TTĐT Chính phủ: https://vanban.chinhphu.vn/?docid=216387&pageid=27160
- Nghị định 23/2025/NĐ-CP trên Cổng TTĐT Chính phủ: https://vanban.chinhphu.vn/?docid=212829&pageid=27160
- Nghị định 53/2022/NĐ-CP, lịch sử hiệu lực VBPL: https://vbpl.vn/bocongan/Pages/vbpq-lichsu.aspx?ItemID=180306&Keyword=
- Nghị định 85/2016/NĐ-CP trên VBPL: https://vbpl.vn/bothongtin/Pages/vbpq-toanvan.aspx?ItemID=112057
- Nghị định 15/2020/NĐ-CP, lược đồ VBPL: https://vbpl.vn/bothongtin/Pages/vbpq-luocdo.aspx?ItemID=140561
- Nghị định 147/2024/NĐ-CP, lược đồ VBPL: https://vbpl.vn/sonla/Pages/vbpq-luocdo.aspx?ItemID=171689
- Luật Chuyển giao công nghệ 2017 trên VBPL: https://vbpl.vn/TW/Pages/vbpq-van-ban-goc.aspx?ItemID=123514
- Nghị định 76/2018/NĐ-CP trên VBPL: https://vbpl.vn/dongthap/Pages/vbpq-thuoctinh.aspx?ItemID=129237&Keyword=
- Luật Khoa học và Công nghệ 2013 bị thay thế bởi Luật 93/2025/QH15: https://luatvietnam.vn/khoa-hoc/luat-khoa-hoc-cong-nghe-2013-79401-d1.html
- Luật 93/2025/QH15, ngày có hiệu lực 01/10/2025: https://vbpl.vn/cantho/Pages/vbpq-vanbanlienquan.aspx?ItemID=185343&Keyword=&dvid=285
- Nghị định 341/2025/NĐ-CP thay Nghị định 131/2013/NĐ-CP từ 15/02/2026: https://luatvietnam.vn/hanh-chinh/nghi-dinh-341-2025-nd-cp-xu-phat-vi-pham-hanh-chinh-ve-quyen-tac-gia-quyen-lien-quan-423032-d1.html
- Thông tư 23/2023/TT-BKHCN trên VBPL: https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=164373
- Quyết định 4448/QĐ-BKHCN trên Công báo: https://congbao.chinhphu.vn/van-ban/quyet-dinh-so-4448-qd-bkhcn-468051.htm
- Thông tư 08/2023/TT-BVHTTDL và Thông tư 21/2025/TT-BVHTTDL liên quan: https://vbpl.vn/hagiang/Pages/vbpq-vanbanlienquan.aspx?ItemID=186002&Keyword=
