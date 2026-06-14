"""
prompts.py
==========
Few-shot + Chain-of-Thought prompts cho Qwen/Qwen2.5-7B-Instruct
để trích xuất thực thể (entity) và quan hệ (relation) từ văn bản pháp luật IT Việt Nam.

Mục tiêu: phục vụ xây dựng Knowledge Graph cho Critic Agent.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam, \
chuyên trích xuất thông tin có cấu trúc từ luật và nghị định.

Nhiệm vụ của bạn là phân tích đoạn văn bản pháp luật được cung cấp và trả về \
một đối tượng JSON hợp lệ chứa danh sách thực thể (entities) và quan hệ (relations).

## Các loại thực thể (Entity Types):
- **VanBan**: Văn bản pháp luật (Luật, Nghị định, Thông tư, v.v.)
- **Dieu**: Điều luật (ví dụ: "Điều 5", "Điều 12a")
- **Khoan**: Khoản trong điều (ví dụ: "Khoản 1", "Khoản 2")
- **Diem**: Điểm trong khoản (ví dụ: "Điểm a", "Điểm b")
- **HanhVi**: Hành vi vi phạm hoặc hành động được pháp luật điều chỉnh
- **CheTai**: Chế tài, hình phạt (chính, bổ sung, hành chính)
- **ChuThe**: Chủ thể pháp lý (cá nhân, tổ chức, cơ quan nhà nước)
- **NghiaVu**: Nghĩa vụ pháp lý phải thực hiện
- **QuyenHan**: Quyền hạn hoặc quyền được phép làm
- **KhaiNiem**: Khái niệm/thuật ngữ được định nghĩa
- **DieuKien**: Điều kiện để áp dụng quy định

## Các loại quan hệ (Relation Types):
- **CO_KHOAN**: Điều → Khoản (Điều chứa Khoản)
- **CO_DIEM**: Khoản → Điểm (Khoản chứa Điểm)
- **THAM_CHIEU**: Điều/Khoản/Điểm → Điều/Khoản (tham chiếu chéo)
- **QUY_DINH_HANH_VI**: Điều/Khoản → HanhVi (quy định về hành vi)
- **CHE_TAI_CHINH**: HanhVi → CheTai (hình phạt chính)
- **CHE_TAI_BO_SUNG**: HanhVi → CheTai (hình phạt bổ sung)
- **AP_DUNG_VOI**: Điều/Khoản → ChuThe (áp dụng với chủ thể nào)
- **CO_NGHIA_VU**: ChuThe → NghiaVu (chủ thể có nghĩa vụ)
- **CO_QUYEN**: ChuThe → QuyenHan (chủ thể có quyền)
- **DINH_NGHIA**: Điều/Khoản → KhaiNiem (định nghĩa khái niệm)
- **CO_DIEU_KIEN**: CheTai/QuyenHan → DieuKien (có điều kiện áp dụng)

## Quy tắc bắt buộc:
1. Mỗi entity phải có trường `id` (duy nhất, snake_case) và `type`.
2. Mỗi relation phải có `source_id`, `target_id`, `relation_type`.
3. Nếu không tìm thấy thực thể nào, trả về `{"entities": [], "relations": []}`.
4. `id` của Điều/Khoản/Điểm phải dùng chuẩn: `D<số>`, `D<số>_K<số>`, `D<số>_K<số>_P<ký_hiệu>`.
5. Đối với CheTai, bắt buộc điền `loai`: "chinh" | "bo_sung" | "hanh_chinh" | "hinh_su".

## Định dạng output BẮT BUỘC:
Bước 1 — Suy nghĩ bên trong thẻ `<thinking>` (phân tích cấu trúc, xác định entity/relation).
Bước 2 — Sau thẻ `</thinking>`, trả về ĐÚNG 1 JSON object thô, bắt đầu bằng `{`, kết thúc bằng `}`.
TUYỆT ĐỐI KHÔNG bọc JSON trong markdown code block.

Ví dụ cấu trúc output:
<thinking>
[phân tích ở đây]
</thinking>
{"entities": [...], "relations": [...]}
"""


# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [

    # ════════════════════════════════════════════════════════════════════════
    # EXAMPLE 1: Điều định nghĩa khái niệm (Luật An ninh mạng)
    # ════════════════════════════════════════════════════════════════════════
    {
        "role": "user",
        "content": """Hãy phân tích đoạn văn bản pháp luật sau và trích xuất thực thể, quan hệ.

## Văn bản nguồn:
- Tên văn bản: Luật An ninh mạng 2025
- Chunk ID: Luat_An_ninh_mang_2025_D2_preamble

## Nội dung:
Điều 2. Giải thích từ ngữ
Trong Luật này, các từ ngữ dưới đây được hiểu như sau:
1. An ninh mạng là sự ổn định, an ninh, an toàn của không gian mạng; bảo vệ hệ thống thông tin và bảo đảm thông tin, dữ liệu, hoạt động trên không gian mạng không gây phương hại đến an ninh quốc gia, trật tự an toàn xã hội.
2. An ninh thông tin mạng là sự bảo đảm tính nguyên vẹn, tính bảo mật, tính khả dụng của thông tin trên không gian mạng.
5. Không gian mạng là môi trường được hình thành bởi hệ thống mạng lưới kết nối của cơ sở hạ tầng công nghệ thông tin.
12. Tội phạm mạng là hành vi nguy hiểm cho xã hội được quy định trong Bộ luật Hình sự, do cá nhân hoặc tổ chức thực hiện trên không gian mạng.

## Hãy suy nghĩ từng bước (Chain-of-Thought):"""
    },
    {
        "role": "assistant",
        "content": """<thinking>
Điều 2 là điều giải thích từ ngữ, gồm Khoản 1, 2, 5, 12. Mỗi khoản định nghĩa 1 KhaiNiem.
Khoản 12 có ChuThe là cá nhân và tổ chức thực hiện tội phạm mạng → AP_DUNG_VOI.
</thinking>
{"entities":[{"id":"van_ban_luat_an_ninh_mang_2025","type":"VanBan","ten":"Luật An ninh mạng 2025","loai":"Luat","so_hieu":"Luật An ninh mạng","nam":2025},{"id":"D2","type":"Dieu","so":"2","ten":"Giải thích từ ngữ","van_ban_id":"van_ban_luat_an_ninh_mang_2025","chunk_id":"Luat_An_ninh_mang_2025_D2_preamble"},{"id":"D2_K1","type":"Khoan","so":"1","dieu_id":"D2","content":"An ninh mạng là sự ổn định, an ninh, an toàn của không gian mạng..."},{"id":"D2_K2","type":"Khoan","so":"2","dieu_id":"D2","content":"An ninh thông tin mạng là sự bảo đảm tính nguyên vẹn, tính bảo mật..."},{"id":"D2_K5","type":"Khoan","so":"5","dieu_id":"D2","content":"Không gian mạng là môi trường được hình thành bởi hệ thống mạng lưới kết nối..."},{"id":"D2_K12","type":"Khoan","so":"12","dieu_id":"D2","content":"Tội phạm mạng là hành vi nguy hiểm cho xã hội..."},{"id":"kn_an_ninh_mang","type":"KhaiNiem","ten":"An ninh mạng","dinh_nghia":"Sự ổn định, an ninh, an toàn của không gian mạng."},{"id":"kn_an_ninh_thong_tin_mang","type":"KhaiNiem","ten":"An ninh thông tin mạng","dinh_nghia":"Sự bảo đảm tính nguyên vẹn, tính bảo mật, tính khả dụng của thông tin."},{"id":"kn_khong_gian_mang","type":"KhaiNiem","ten":"Không gian mạng","dinh_nghia":"Môi trường được hình thành bởi hệ thống mạng lưới kết nối của cơ sở hạ tầng công nghệ thông tin."},{"id":"kn_toi_pham_mang","type":"KhaiNiem","ten":"Tội phạm mạng","dinh_nghia":"Hành vi nguy hiểm cho xã hội được quy định trong Bộ luật Hình sự, do cá nhân hoặc tổ chức thực hiện."},{"id":"chu_the_ca_nhan","type":"ChuThe","ten":"Cá nhân","loai":"ca_nhan"},{"id":"chu_the_to_chuc","type":"ChuThe","ten":"Tổ chức","loai":"to_chuc"}],"relations":[{"source_id":"D2","relation_type":"CO_KHOAN","target_id":"D2_K1"},{"source_id":"D2","relation_type":"CO_KHOAN","target_id":"D2_K2"},{"source_id":"D2","relation_type":"CO_KHOAN","target_id":"D2_K5"},{"source_id":"D2","relation_type":"CO_KHOAN","target_id":"D2_K12"},{"source_id":"D2_K1","relation_type":"DINH_NGHIA","target_id":"kn_an_ninh_mang"},{"source_id":"D2_K2","relation_type":"DINH_NGHIA","target_id":"kn_an_ninh_thong_tin_mang"},{"source_id":"D2_K5","relation_type":"DINH_NGHIA","target_id":"kn_khong_gian_mang"},{"source_id":"D2_K12","relation_type":"DINH_NGHIA","target_id":"kn_toi_pham_mang"},{"source_id":"D2_K12","relation_type":"AP_DUNG_VOI","target_id":"chu_the_ca_nhan"},{"source_id":"D2_K12","relation_type":"AP_DUNG_VOI","target_id":"chu_the_to_chuc"}]}"""
    },

    # ════════════════════════════════════════════════════════════════════════
    # EXAMPLE 2: Điều quy định hành vi bị cấm (An ninh mạng)
    # ════════════════════════════════════════════════════════════════════════
    {
        "role": "user",
        "content": """Hãy phân tích đoạn văn bản pháp luật sau và trích xuất thực thể, quan hệ.

## Văn bản nguồn:
- Tên văn bản: Luật An ninh mạng 2025
- Chunk ID: Luat_An_ninh_mang_2025_D7_K1

## Nội dung:
Điều 7. Các hành vi bị nghiêm cấm về an ninh mạng
1. Đăng tải, phát tán thông tin có nội dung sau trên không gian mạng:
a) Tuyên truyền chống Nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam bao gồm: tuyên truyền xuyên tạc, phỉ báng chính quyền nhân dân;
b) Xuyên tạc lịch sử, phủ nhận thành tựu cách mạng, phá hoại khối đại đoàn kết toàn dân tộc;
c) Bôi nhọ, vu khống, thông tin sai sự thật, xâm phạm nhân phẩm, danh dự, uy tín của người khác.
3. Thực hiện tấn công mạng, khủng bố mạng, gián điệp mạng, tội phạm mạng.

## Hãy suy nghĩ từng bước (Chain-of-Thought):"""
    },
    {
        "role": "assistant",
        "content": """**Bước 1 - Xác định Điều:**
- Điều 7: quy định các hành vi bị cấm → đây là Điều về hành vi vi phạm (HanhVi)

**Bước 2 - Xác định các Khoản và Điểm:**
- Khoản 1: nghiêm cấm đăng tải/phát tán thông tin vi phạm
  - Điểm a: tuyên truyền chống nhà nước
  - Điểm b: xuyên tạc lịch sử
  - Điểm c: bôi nhọ, vu khống
- Khoản 3: cấm tấn công mạng, khủng bố mạng, gián điệp mạng, tội phạm mạng

**Bước 3 - Xác định HanhVi:**
- "Đăng tải, phát tán thông tin vi phạm" → HanhVi tổng quát
- "Tuyên truyền chống nhà nước" → HanhVi cụ thể
- "Xuyên tạc lịch sử" → HanhVi cụ thể
- "Bôi nhọ, vu khống" → HanhVi cụ thể
- "Tấn công mạng" → HanhVi cụ thể
- "Khủng bố mạng", "Gián điệp mạng", "Tội phạm mạng" → HanhVi cụ thể

**Bước 4 - Quan hệ:**
- Điều 7 → CO_KHOAN → Khoản 1, 3
- Khoản 1 → CO_DIEM → Điểm a, b, c
- Điều 7/Khoản/Điểm → QUY_DINH_HANH_VI → các HanhVi

```json
{
  "entities": [
    {
      "id": "D7",
      "type": "Dieu",
      "so": "7",
      "ten": "Các hành vi bị nghiêm cấm về an ninh mạng",
      "van_ban_id": "van_ban_luat_an_ninh_mang_2025",
      "chunk_id": "Luat_An_ninh_mang_2025_D7_K1"
    },
    {
      "id": "D7_K1",
      "type": "Khoan",
      "so": "1",
      "dieu_id": "D7",
      "content": "Đăng tải, phát tán thông tin có nội dung vi phạm trên không gian mạng."
    },
    {
      "id": "D7_K3",
      "type": "Khoan",
      "so": "3",
      "dieu_id": "D7",
      "content": "Thực hiện tấn công mạng, khủng bố mạng, gián điệp mạng, tội phạm mạng."
    },
    {
      "id": "D7_K1_Pa",
      "type": "Diem",
      "ky_hieu": "a",
      "khoan_id": "D7_K1",
      "content": "Tuyên truyền chống Nhà nước CHXHCN Việt Nam."
    },
    {
      "id": "D7_K1_Pb",
      "type": "Diem",
      "ky_hieu": "b",
      "khoan_id": "D7_K1",
      "content": "Xuyên tạc lịch sử, phủ nhận thành tựu cách mạng."
    },
    {
      "id": "D7_K1_Pc",
      "type": "Diem",
      "ky_hieu": "c",
      "khoan_id": "D7_K1",
      "content": "Bôi nhọ, vu khống, thông tin sai sự thật, xâm phạm nhân phẩm, danh dự."
    },
    {
      "id": "hv_dang_tai_phat_tan_thong_tin_vi_pham",
      "type": "HanhVi",
      "mo_ta": "Đăng tải, phát tán thông tin vi phạm trên không gian mạng",
      "bi_cam": true
    },
    {
      "id": "hv_tuyen_truyen_chong_nha_nuoc",
      "type": "HanhVi",
      "mo_ta": "Tuyên truyền chống Nhà nước CHXHCN Việt Nam, phỉ báng chính quyền nhân dân",
      "bi_cam": true
    },
    {
      "id": "hv_xuyen_tac_lich_su",
      "type": "HanhVi",
      "mo_ta": "Xuyên tạc lịch sử, phủ nhận thành tựu cách mạng, phá hoại đoàn kết",
      "bi_cam": true
    },
    {
      "id": "hv_boi_nho_vu_khong",
      "type": "HanhVi",
      "mo_ta": "Bôi nhọ, vu khống, thông tin sai sự thật xâm phạm nhân phẩm, danh dự",
      "bi_cam": true
    },
    {
      "id": "hv_tan_cong_mang",
      "type": "HanhVi",
      "mo_ta": "Thực hiện tấn công mạng",
      "bi_cam": true
    },
    {
      "id": "hv_khung_bo_mang",
      "type": "HanhVi",
      "mo_ta": "Thực hiện khủng bố mạng",
      "bi_cam": true
    },
    {
      "id": "hv_gian_diep_mang",
      "type": "HanhVi",
      "mo_ta": "Thực hiện gián điệp mạng",
      "bi_cam": true
    }
  ],
  "relations": [
    {"source_id": "D7", "relation_type": "CO_KHOAN", "target_id": "D7_K1"},
    {"source_id": "D7", "relation_type": "CO_KHOAN", "target_id": "D7_K3"},
    {"source_id": "D7_K1", "relation_type": "CO_DIEM", "target_id": "D7_K1_Pa"},
    {"source_id": "D7_K1", "relation_type": "CO_DIEM", "target_id": "D7_K1_Pb"},
    {"source_id": "D7_K1", "relation_type": "CO_DIEM", "target_id": "D7_K1_Pc"},
    {"source_id": "D7_K1", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_dang_tai_phat_tan_thong_tin_vi_pham"},
    {"source_id": "D7_K1_Pa", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_tuyen_truyen_chong_nha_nuoc"},
    {"source_id": "D7_K1_Pb", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_xuyen_tac_lich_su"},
    {"source_id": "D7_K1_Pc", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_boi_nho_vu_khong"},
    {"source_id": "D7_K3", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_tan_cong_mang"},
    {"source_id": "D7_K3", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_khung_bo_mang"},
    {"source_id": "D7_K3", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_gian_diep_mang"}
  ]
}
```"""
    },

    # ════════════════════════════════════════════════════════════════════════
    # EXAMPLE 3: Điều quy định chế tài (hình phạt chính + bổ sung) – NĐ xử phạt
    # ════════════════════════════════════════════════════════════════════════
    {
        "role": "user",
        "content": """Hãy phân tích đoạn văn bản pháp luật sau và trích xuất thực thể, quan hệ.

## Văn bản nguồn:
- Tên văn bản: Nghị Định 15 2020 NĐ-CP (sửa đổi, bổ sung Nghị Định 14 2022)
- Chunk ID: ND_15_2020_D90_K1

## Nội dung:
Điều 90. Vi phạm quy định về cung cấp thông tin trên mạng
1. Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với hành vi cung cấp thông tin giả mạo, thông tin sai sự thật, xuyên tạc, vu khống, xúc phạm uy tín của tổ chức, danh dự, nhân phẩm của cá nhân.
2. Phạt tiền từ 20.000.000 đồng đến 30.000.000 đồng đối với một trong các hành vi sau:
a) Giả mạo trang thông tin điện tử của tổ chức, cá nhân khác;
b) Cung cấp thông tin giả mạo nhằm chiếm đoạt tài sản.
3. Hình thức xử phạt bổ sung: Tịch thu tang vật, phương tiện vi phạm đối với hành vi quy định tại khoản 2 Điều này.
4. Biện pháp khắc phục hậu quả: Buộc gỡ bỏ thông tin sai sự thật hoặc gây nhầm lẫn hoặc thông tin vi phạm pháp luật do thực hiện hành vi vi phạm quy định tại khoản 1 và khoản 2 Điều này.

## Hãy suy nghĩ từng bước (Chain-of-Thought):"""
    },
    {
        "role": "assistant",
        "content": """**Bước 1 - Xác định Điều và cấu trúc:**
- Điều 90: quy định vi phạm và chế tài hành chính
- Cấu trúc: Khoản 1 (hình phạt chính K1), Khoản 2 (hình phạt chính K2 với điểm a, b), Khoản 3 (hình phạt bổ sung), Khoản 4 (biện pháp khắc phục)

**Bước 2 - Xác định HanhVi:**
- "Cung cấp thông tin giả mạo, sai sự thật, xuyên tạc, vu khống" (Khoản 1)
- "Giả mạo trang thông tin điện tử" (Khoản 2, Điểm a)
- "Cung cấp thông tin giả mạo nhằm chiếm đoạt tài sản" (Khoản 2, Điểm b)

**Bước 3 - Xác định CheTai:**
- Phạt tiền 5-10 triệu → CheTai chính (loai: "hanh_chinh") cho HanhVi K1
- Phạt tiền 20-30 triệu → CheTai chính (loai: "hanh_chinh") cho HanhVi K2
- Tịch thu tang vật → CheTai bổ sung cho HanhVi K2
- Buộc gỡ bỏ thông tin → CheTai bổ sung (biện pháp khắc phục) cho K1, K2

**Bước 4 - Quan hệ tham chiếu:**
- Khoản 3 tham chiếu Khoản 2 → THAM_CHIEU
- Khoản 4 tham chiếu Khoản 1, Khoản 2 → THAM_CHIEU

```json
{
  "entities": [
    {
      "id": "D90",
      "type": "Dieu",
      "so": "90",
      "ten": "Vi phạm quy định về cung cấp thông tin trên mạng",
      "van_ban_id": "van_ban_nd_15_2020",
      "chunk_id": "ND_15_2020_D90_K1"
    },
    {
      "id": "D90_K1",
      "type": "Khoan",
      "so": "1",
      "dieu_id": "D90",
      "content": "Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với hành vi cung cấp thông tin giả mạo..."
    },
    {
      "id": "D90_K2",
      "type": "Khoan",
      "so": "2",
      "dieu_id": "D90",
      "content": "Phạt tiền từ 20.000.000 đồng đến 30.000.000 đồng đối với một trong các hành vi sau..."
    },
    {
      "id": "D90_K3",
      "type": "Khoan",
      "so": "3",
      "dieu_id": "D90",
      "content": "Hình thức xử phạt bổ sung: Tịch thu tang vật, phương tiện vi phạm..."
    },
    {
      "id": "D90_K4",
      "type": "Khoan",
      "so": "4",
      "dieu_id": "D90",
      "content": "Biện pháp khắc phục hậu quả: Buộc gỡ bỏ thông tin sai sự thật..."
    },
    {
      "id": "D90_K2_Pa",
      "type": "Diem",
      "ky_hieu": "a",
      "khoan_id": "D90_K2",
      "content": "Giả mạo trang thông tin điện tử của tổ chức, cá nhân khác."
    },
    {
      "id": "D90_K2_Pb",
      "type": "Diem",
      "ky_hieu": "b",
      "khoan_id": "D90_K2",
      "content": "Cung cấp thông tin giả mạo nhằm chiếm đoạt tài sản."
    },
    {
      "id": "hv_cung_cap_thong_tin_gia_mao_k1",
      "type": "HanhVi",
      "mo_ta": "Cung cấp thông tin giả mạo, thông tin sai sự thật, xuyên tạc, vu khống, xúc phạm uy tín tổ chức, cá nhân",
      "bi_cam": true
    },
    {
      "id": "hv_gia_mao_trang_thong_tin",
      "type": "HanhVi",
      "mo_ta": "Giả mạo trang thông tin điện tử của tổ chức, cá nhân khác",
      "bi_cam": true
    },
    {
      "id": "hv_cung_cap_thong_tin_chiem_doat",
      "type": "HanhVi",
      "mo_ta": "Cung cấp thông tin giả mạo nhằm chiếm đoạt tài sản",
      "bi_cam": true
    },
    {
      "id": "che_tai_phat_tien_5_10tr",
      "type": "CheTai",
      "loai": "hanh_chinh",
      "mo_ta": "Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng",
      "muc_phat_min": 5000000,
      "muc_phat_max": 10000000,
      "don_vi": "VND"
    },
    {
      "id": "che_tai_phat_tien_20_30tr",
      "type": "CheTai",
      "loai": "hanh_chinh",
      "mo_ta": "Phạt tiền từ 20.000.000 đồng đến 30.000.000 đồng",
      "muc_phat_min": 20000000,
      "muc_phat_max": 30000000,
      "don_vi": "VND"
    },
    {
      "id": "che_tai_tich_thu_tang_vat",
      "type": "CheTai",
      "loai": "bo_sung",
      "mo_ta": "Tịch thu tang vật, phương tiện vi phạm"
    },
    {
      "id": "che_tai_go_bo_thong_tin",
      "type": "CheTai",
      "loai": "bo_sung",
      "mo_ta": "Buộc gỡ bỏ thông tin sai sự thật hoặc gây nhầm lẫn hoặc thông tin vi phạm pháp luật"
    }
  ],
  "relations": [
    {"source_id": "D90", "relation_type": "CO_KHOAN", "target_id": "D90_K1"},
    {"source_id": "D90", "relation_type": "CO_KHOAN", "target_id": "D90_K2"},
    {"source_id": "D90", "relation_type": "CO_KHOAN", "target_id": "D90_K3"},
    {"source_id": "D90", "relation_type": "CO_KHOAN", "target_id": "D90_K4"},
    {"source_id": "D90_K2", "relation_type": "CO_DIEM", "target_id": "D90_K2_Pa"},
    {"source_id": "D90_K2", "relation_type": "CO_DIEM", "target_id": "D90_K2_Pb"},
    {"source_id": "D90_K1", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_cung_cap_thong_tin_gia_mao_k1"},
    {"source_id": "D90_K2_Pa", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_gia_mao_trang_thong_tin"},
    {"source_id": "D90_K2_Pb", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_cung_cap_thong_tin_chiem_doat"},
    {"source_id": "hv_cung_cap_thong_tin_gia_mao_k1", "relation_type": "CHE_TAI_CHINH", "target_id": "che_tai_phat_tien_5_10tr"},
    {"source_id": "hv_gia_mao_trang_thong_tin", "relation_type": "CHE_TAI_CHINH", "target_id": "che_tai_phat_tien_20_30tr"},
    {"source_id": "hv_cung_cap_thong_tin_chiem_doat", "relation_type": "CHE_TAI_CHINH", "target_id": "che_tai_phat_tien_20_30tr"},
    {"source_id": "hv_gia_mao_trang_thong_tin", "relation_type": "CHE_TAI_BO_SUNG", "target_id": "che_tai_tich_thu_tang_vat"},
    {"source_id": "hv_cung_cap_thong_tin_chiem_doat", "relation_type": "CHE_TAI_BO_SUNG", "target_id": "che_tai_tich_thu_tang_vat"},
    {"source_id": "hv_cung_cap_thong_tin_gia_mao_k1", "relation_type": "CHE_TAI_BO_SUNG", "target_id": "che_tai_go_bo_thong_tin"},
    {"source_id": "hv_gia_mao_trang_thong_tin", "relation_type": "CHE_TAI_BO_SUNG", "target_id": "che_tai_go_bo_thong_tin"},
    {"source_id": "D90_K3", "relation_type": "THAM_CHIEU", "target_id": "D90_K2",
     "ghi_chu": "Khoản 3 áp dụng cho hành vi tại Khoản 2"},
    {"source_id": "D90_K4", "relation_type": "THAM_CHIEU", "target_id": "D90_K1",
     "ghi_chu": "Khoản 4 áp dụng cho hành vi tại Khoản 1"},
    {"source_id": "D90_K4", "relation_type": "THAM_CHIEU", "target_id": "D90_K2",
     "ghi_chu": "Khoản 4 áp dụng cho hành vi tại Khoản 2"}
  ]
}
```"""
    },

    # ════════════════════════════════════════════════════════════════════════
    # EXAMPLE 4: Điều quy định nghĩa vụ + tham chiếu chéo
    # ════════════════════════════════════════════════════════════════════════
    {
        "role": "user",
        "content": """Hãy phân tích đoạn văn bản pháp luật sau và trích xuất thực thể, quan hệ.

## Văn bản nguồn:
- Tên văn bản: Luật Bảo vệ dữ liệu cá nhân 2025
- Chunk ID: Luat_BVDLCN_2025_D24_K1

## Nội dung:
Điều 24. Trách nhiệm của bên kiểm soát dữ liệu cá nhân
1. Bên kiểm soát dữ liệu cá nhân có trách nhiệm:
a) Thực hiện các biện pháp kỹ thuật, tổ chức phù hợp để bảo vệ dữ liệu cá nhân theo quy định tại Điều 26 Luật này;
b) Thực hiện đánh giá tác động xử lý dữ liệu cá nhân trong trường hợp quy định tại Điều 25 Luật này;
c) Thông báo cho Ủy ban Bảo vệ dữ liệu cá nhân trong trường hợp xảy ra sự cố lộ, mất dữ liệu cá nhân trong thời hạn 72 giờ kể từ khi phát hiện sự cố.
2. Trường hợp bên kiểm soát dữ liệu vi phạm quy định tại khoản 1 Điều này, sẽ bị xử phạt theo quy định của pháp luật về xử lý vi phạm hành chính.

## Hãy suy nghĩ từng bước (Chain-of-Thought):"""
    },
    {
        "role": "assistant",
        "content": """**Bước 1 - Xác định cấu trúc:**
- Điều 24: quy định trách nhiệm (NghiaVu) của ChuThe "Bên kiểm soát dữ liệu cá nhân"
- Khoản 1: liệt kê nghĩa vụ cụ thể (3 điểm)
- Khoản 2: hệ quả pháp lý (vi phạm → bị xử phạt)

**Bước 2 - Xác định tham chiếu chéo:**
- Điểm a tham chiếu Điều 26 cùng Luật
- Điểm b tham chiếu Điều 25 cùng Luật
→ Quan hệ THAM_CHIEU quan trọng cho Critic Agent!

**Bước 3 - Xác định ChuThe và NghiaVu:**
- ChuThe: "Bên kiểm soát dữ liệu cá nhân"
- NghiaVu 1: thực hiện biện pháp bảo vệ dữ liệu
- NghiaVu 2: thực hiện đánh giá tác động
- NghiaVu 3: thông báo sự cố trong 72 giờ

**Bước 4 - Chế tài:**
- Khoản 2 quy định vi phạm → bị xử phạt hành chính (tham chiếu pháp luật xử lý vi phạm)

```json
{
  "entities": [
    {
      "id": "D24",
      "type": "Dieu",
      "so": "24",
      "ten": "Trách nhiệm của bên kiểm soát dữ liệu cá nhân",
      "van_ban_id": "van_ban_luat_bvdlcn_2025",
      "chunk_id": "Luat_BVDLCN_2025_D24_K1"
    },
    {
      "id": "D24_K1",
      "type": "Khoan",
      "so": "1",
      "dieu_id": "D24",
      "content": "Bên kiểm soát dữ liệu cá nhân có trách nhiệm..."
    },
    {
      "id": "D24_K2",
      "type": "Khoan",
      "so": "2",
      "dieu_id": "D24",
      "content": "Trường hợp vi phạm quy định tại khoản 1 Điều này, sẽ bị xử phạt..."
    },
    {
      "id": "D24_K1_Pa",
      "type": "Diem",
      "ky_hieu": "a",
      "khoan_id": "D24_K1",
      "content": "Thực hiện các biện pháp kỹ thuật, tổ chức phù hợp để bảo vệ dữ liệu cá nhân theo Điều 26."
    },
    {
      "id": "D24_K1_Pb",
      "type": "Diem",
      "ky_hieu": "b",
      "khoan_id": "D24_K1",
      "content": "Thực hiện đánh giá tác động xử lý dữ liệu cá nhân theo Điều 25."
    },
    {
      "id": "D24_K1_Pc",
      "type": "Diem",
      "ky_hieu": "c",
      "khoan_id": "D24_K1",
      "content": "Thông báo cho Ủy ban Bảo vệ dữ liệu cá nhân trong 72 giờ khi xảy ra sự cố lộ, mất dữ liệu."
    },
    {
      "id": "chu_the_ben_kiem_soat_dlcn",
      "type": "ChuThe",
      "ten": "Bên kiểm soát dữ liệu cá nhân",
      "loai": "to_chuc_ca_nhan"
    },
    {
      "id": "nv_thuc_hien_bien_phap_bao_ve",
      "type": "NghiaVu",
      "mo_ta": "Thực hiện các biện pháp kỹ thuật, tổ chức phù hợp để bảo vệ dữ liệu cá nhân"
    },
    {
      "id": "nv_danh_gia_tac_dong",
      "type": "NghiaVu",
      "mo_ta": "Thực hiện đánh giá tác động xử lý dữ liệu cá nhân"
    },
    {
      "id": "nv_thong_bao_su_co_72h",
      "type": "NghiaVu",
      "mo_ta": "Thông báo cho Ủy ban Bảo vệ dữ liệu cá nhân trong thời hạn 72 giờ khi xảy ra sự cố lộ, mất dữ liệu"
    },
    {
      "id": "D26",
      "type": "Dieu",
      "so": "26",
      "ten": "Điều 26 (tham chiếu - chưa extract)",
      "van_ban_id": "van_ban_luat_bvdlcn_2025"
    },
    {
      "id": "D25",
      "type": "Dieu",
      "so": "25",
      "ten": "Điều 25 (tham chiếu - chưa extract)",
      "van_ban_id": "van_ban_luat_bvdlcn_2025"
    },
    {
      "id": "hv_vi_pham_trach_nhiem_kiem_soat_dlcn",
      "type": "HanhVi",
      "mo_ta": "Vi phạm trách nhiệm bảo vệ dữ liệu cá nhân của bên kiểm soát",
      "bi_cam": true
    },
    {
      "id": "che_tai_xu_phat_hanh_chinh_dlcn",
      "type": "CheTai",
      "loai": "hanh_chinh",
      "mo_ta": "Xử phạt theo quy định pháp luật về xử lý vi phạm hành chính"
    }
  ],
  "relations": [
    {"source_id": "D24", "relation_type": "CO_KHOAN", "target_id": "D24_K1"},
    {"source_id": "D24", "relation_type": "CO_KHOAN", "target_id": "D24_K2"},
    {"source_id": "D24_K1", "relation_type": "CO_DIEM", "target_id": "D24_K1_Pa"},
    {"source_id": "D24_K1", "relation_type": "CO_DIEM", "target_id": "D24_K1_Pb"},
    {"source_id": "D24_K1", "relation_type": "CO_DIEM", "target_id": "D24_K1_Pc"},
    {"source_id": "D24_K1", "relation_type": "AP_DUNG_VOI", "target_id": "chu_the_ben_kiem_soat_dlcn"},
    {"source_id": "chu_the_ben_kiem_soat_dlcn", "relation_type": "CO_NGHIA_VU", "target_id": "nv_thuc_hien_bien_phap_bao_ve"},
    {"source_id": "chu_the_ben_kiem_soat_dlcn", "relation_type": "CO_NGHIA_VU", "target_id": "nv_danh_gia_tac_dong"},
    {"source_id": "chu_the_ben_kiem_soat_dlcn", "relation_type": "CO_NGHIA_VU", "target_id": "nv_thong_bao_su_co_72h"},
    {
      "source_id": "D24_K1_Pa",
      "relation_type": "THAM_CHIEU",
      "target_id": "D26",
      "loai_tham_chieu": "ap_dung",
      "ghi_chu": "Thực hiện biện pháp bảo vệ theo quy định tại Điều 26"
    },
    {
      "source_id": "D24_K1_Pb",
      "relation_type": "THAM_CHIEU",
      "target_id": "D25",
      "loai_tham_chieu": "ap_dung",
      "ghi_chu": "Thực hiện đánh giá tác động theo Điều 25"
    },
    {"source_id": "D24_K2", "relation_type": "THAM_CHIEU", "target_id": "D24_K1",
     "loai_tham_chieu": "ap_dung", "ghi_chu": "Vi phạm khoản 1 thì áp dụng khoản 2"},
    {"source_id": "D24_K2", "relation_type": "QUY_DINH_HANH_VI", "target_id": "hv_vi_pham_trach_nhiem_kiem_soat_dlcn"},
    {"source_id": "hv_vi_pham_trach_nhiem_kiem_soat_dlcn", "relation_type": "CHE_TAI_CHINH", "target_id": "che_tai_xu_phat_hanh_chinh_dlcn"}
  ]
}
```"""
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_extraction_prompt(
    van_ban_name: str,
    chunk_id: str,
    content: str,
    van_ban_id: str,
    use_few_shot: bool = True,
) -> list[dict]:
    """
    Xây dựng danh sách messages cho Qwen2.5-7B-Instruct (chat format).

    Args:
        van_ban_name: Tên văn bản pháp luật (ví dụ: "Luật An ninh mạng 2025")
        chunk_id: ID chunk theo chuẩn VBPLChunker (ví dụ: "Luat_ANM_2025_D7_K1")
        content: Nội dung chunk (Điều + các Khoản/Điểm)
        van_ban_id: ID văn bản trong graph (snake_case)
        use_few_shot: Có dùng few-shot examples không (tắt nếu context window nhỏ)

    Returns:
        List of messages theo định dạng chat của Qwen
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if use_few_shot:
        # Chỉ lấy 1 few-shot example để tiết kiệm VRAM trên Colab T4
        # Example 1: định nghĩa khái niệm (đại diện cấu trúc tổng quát nhất)
        selected_examples = [FEW_SHOT_EXAMPLES[0]]
        messages.extend(selected_examples)

    # User message thực tế
    user_message = f"""Phân tích đoạn văn bản pháp luật sau và trích xuất thực thể, quan hệ.

## Văn bản nguồn:
- Tên văn bản: {van_ban_name}
- Văn bản ID: {van_ban_id}
- Chunk ID: {chunk_id}

## Nội dung:
{content}

## Yêu cầu:
Xác định Khoản, Điểm, hành vi, chế tài, chủ thể, nghĩa vụ và tham chiếu.
Trả về một JSON object DUY NHẤT (không bọc trong code block, không giải thích trước hay sau):
- id của Điều: "D<số>" (ví dụ: "D7", "D24")
- id của Khoản: "D<số>_K<số>" (ví dụ: "D7_K1")
- id của Điểm: "D<số>_K<số>_P<ký_hiệu>" (ví dụ: "D7_K1_Pa")
- Nếu có THAM_CHIEU đến Điều khác, phải tạo entity Dieu placeholder với van_ban_id tương ứng.
"""
    messages.append({"role": "user", "content": user_message})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# 2-PASS EXTRACTION — PASS 1: ENTITIES ONLY
# ─────────────────────────────────────────────────────────────────────────────

ENTITIES_ONLY_SYSTEM_PROMPT = """Bạn là chuyên gia trích xuất thực thể từ văn bản pháp luật Việt Nam.

## Nhiệm vụ: Chỉ trích xuất THỰC THỂ (entities), KHÔNG cần quan hệ.

## Loại thực thể cần trích xuất:
- **Dieu**: Điều luật → id: "D<số>" (ví dụ: "D25", "D114")
- **Khoan**: Khoản trong điều → id: "D<số>_K<số>" (ví dụ: "D25_K1")
- **Diem**: Điểm trong khoản → id: "D<số>_K<số>_P<ký_hiệu>" (ví dụ: "D25_K1_Pa")
- **HanhVi**: Hành vi vi phạm hoặc được pháp luật điều chỉnh
- **CheTai**: Chế tài, hình phạt (bắt buộc có field `loai`: "chinh"|"bo_sung"|"hanh_chinh"|"hinh_su")
- **ChuThe**: Chủ thể pháp lý (cá nhân, tổ chức, cơ quan)
- **NghiaVu**: Nghĩa vụ pháp lý phải thực hiện
- **QuyenHan**: Quyền hạn hoặc quyền được phép làm
- **KhaiNiem**: Khái niệm/thuật ngữ được định nghĩa
- **DieuKien**: Điều kiện áp dụng quy định

## Quy tắc output COMPACT (quan trọng để tránh truncation):
1. Mỗi entity CHỈ CẦN: `id`, `type`, và TỐI ĐA 2 field ngắn (ví dụ: `so`, `ten`, `loai`, `mo_ta` ngắn).
2. KHÔNG sinh field `content`, `dinh_nghia`, `chu_thich` dài dòng.
3. Trả về JSON: `{"entities": [...]}` — KHÔNG có `relations`.

## Định dạng output BẮT BUỘC:
<thinking>
[suy nghĩ ngắn gọn về cấu trúc Điều/Khoản/Điểm và các thực thể]
</thinking>
{"entities": [...]}
"""

# Few-shot cho Pass 1: compact entities only
ENTITIES_ONLY_FEW_SHOT = [
    {
        "role": "user",
        "content": """Trích xuất THỰC THỂ từ đoạn văn bản sau (không cần quan hệ).

Văn bản: Nghị định 100/2019/NĐ-CP | van_ban_id: nd_100_2019 | chunk: nd_100_2019_D5_PARENT

Nội dung:
Điều 5. Xử phạt người điều khiển xe ô tô và các loại xe tương tự xe ô tô vi phạm quy tắc giao thông đường bộ
1. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe thực hiện một trong các hành vi sau đây:
a) Vượt đèn tín hiệu giao thông;
b) Không tuân thủ hiệu lệnh của người điều khiển giao thông.
2. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe có một trong các hành vi sau:
a) Điều khiển xe chạy quá tốc độ quy định từ 20 km/h đến dưới 35 km/h;
3. Ngoài bị phạt tiền, người điều khiển xe còn bị áp dụng các hình thức xử phạt bổ sung: tước quyền sử dụng Giấy phép lái xe từ 01 đến 03 tháng đối với hành vi quy định tại khoản 2 điều này."""
    },
    {
        "role": "assistant",
        "content": """<thinking>
Điều 5 có 3 khoản. Khoản 1,2 quy định phạt tiền cho các hành vi vi phạm. Khoản 3 là hình phạt bổ sung.
Entities: D5, K1, K2, K3, điểm a/b của K1 và K2, các HanhVi, các CheTai, ChuThe.
</thinking>
{"entities":[{"id":"D5","type":"Dieu","so":"5","ten":"Xử phạt người điều khiển xe ô tô vi phạm quy tắc giao thông","van_ban_id":"nd_100_2019"},{"id":"D5_K1","type":"Khoan","so":"1","dieu_id":"D5"},{"id":"D5_K2","type":"Khoan","so":"2","dieu_id":"D5"},{"id":"D5_K3","type":"Khoan","so":"3","dieu_id":"D5"},{"id":"D5_K1_Pa","type":"Diem","ky_hieu":"a","khoan_id":"D5_K1"},{"id":"D5_K1_Pb","type":"Diem","ky_hieu":"b","khoan_id":"D5_K1"},{"id":"D5_K2_Pa","type":"Diem","ky_hieu":"a","khoan_id":"D5_K2"},{"id":"hv_vuot_den_do","type":"HanhVi","mo_ta":"Vượt đèn tín hiệu giao thông","bi_cam":true},{"id":"hv_khong_tuan_hieu_lenh","type":"HanhVi","mo_ta":"Không tuân thủ hiệu lệnh người điều khiển giao thông","bi_cam":true},{"id":"hv_qua_toc_do_20_35","type":"HanhVi","mo_ta":"Chạy quá tốc độ từ 20km/h đến dưới 35km/h","bi_cam":true},{"id":"ct_phat_tien_800k_1tr","type":"CheTai","loai":"hanh_chinh","mo_ta":"Phạt tiền 800.000đ–1.000.000đ"},{"id":"ct_phat_tien_4tr_6tr","type":"CheTai","loai":"hanh_chinh","mo_ta":"Phạt tiền 4.000.000đ–6.000.000đ"},{"id":"ct_tuoc_gplx_1_3_thang","type":"CheTai","loai":"bo_sung","mo_ta":"Tước GPLX từ 1 đến 3 tháng"},{"id":"chu_the_nguoi_dieu_khien_oto","type":"ChuThe","ten":"Người điều khiển xe ô tô","loai":"ca_nhan"}]}"""
    }
]


def build_entities_only_prompt(
    van_ban_name: str,
    van_ban_id: str,
    chunk_id: str,
    content: str,
) -> list[dict]:
    """
    Pass 1: Chỉ extract entities (compact format, không có relations).
    Output nhỏ hơn ~60% so với single-pass → không bị truncate.

    Args:
        van_ban_name: Tên văn bản pháp luật
        van_ban_id: ID văn bản (snake_case)
        chunk_id: ID chunk
        content: Nội dung chunk (Điều + Khoản/Điểm)

    Returns:
        List messages theo chat format của Qwen
    """
    messages = [
        {"role": "system", "content": ENTITIES_ONLY_SYSTEM_PROMPT},
        *ENTITIES_ONLY_FEW_SHOT,
        {
            "role": "user",
            "content": f"""Trích xuất THỰC THỂ từ đoạn văn bản sau (không cần quan hệ).

Văn bản: {van_ban_name} | van_ban_id: {van_ban_id} | chunk: {chunk_id}

Nội dung:
{content}"""
        }
    ]
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# 2-PASS EXTRACTION — PASS 2: RELATIONS ONLY
# ─────────────────────────────────────────────────────────────────────────────

RELATIONS_ONLY_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích quan hệ pháp lý.

## Nhiệm vụ: Chỉ xác định QUAN HỆ (relations) giữa các thực thể đã cho.

## Loại quan hệ cần tìm:
- CO_KHOAN: Dieu → Khoan
- CO_DIEM: Khoan → Diem
- QUY_DINH_HANH_VI: Dieu/Khoan/Diem → HanhVi
- CHE_TAI_CHINH: HanhVi → CheTai (hình phạt chính)
- CHE_TAI_BO_SUNG: HanhVi → CheTai (hình phạt bổ sung/tước quyền)
- AP_DUNG_VOI: Dieu/Khoan → ChuThe
- CO_NGHIA_VU: ChuThe → NghiaVu
- CO_QUYEN: ChuThe → QuyenHan
- DINH_NGHIA: Dieu/Khoan → KhaiNiem
- THAM_CHIEU: Dieu/Khoan/Diem → Dieu/Khoan (tham chiếu chéo)
- CO_DIEU_KIEN: CheTai/QuyenHan → DieuKien

## Quy tắc:
1. Chỉ tạo quan hệ giữa các entity IDs ĐÃ ĐƯỢC CUNG CẤP.
2. Mỗi relation: {"source_id": "...", "relation_type": "...", "target_id": "..."}
3. Trả về JSON: {"relations": [...]}

## Định dạng output BẮT BUỘC:
<thinking>
[suy nghĩ ngắn về các quan hệ chính]
</thinking>
{"relations": [...]}
"""

# Few-shot cho Pass 2: relations only
RELATIONS_ONLY_FEW_SHOT = [
    {
        "role": "user",
        "content": """Xác định QUAN HỆ giữa các thực thể đã biết dựa trên nội dung.

Entity IDs đã có:
D5, D5_K1, D5_K2, D5_K3, D5_K1_Pa, D5_K1_Pb, D5_K2_Pa,
hv_vuot_den_do, hv_khong_tuan_hieu_lenh, hv_qua_toc_do_20_35,
ct_phat_tien_800k_1tr, ct_phat_tien_4tr_6tr, ct_tuoc_gplx_1_3_thang,
chu_the_nguoi_dieu_khien_oto

Nội dung:
Điều 5. Xử phạt người điều khiển xe ô tô...
1. Phạt tiền từ 800.000đ đến 1.000.000đ đối với:
a) Vượt đèn tín hiệu giao thông;
b) Không tuân thủ hiệu lệnh người điều khiển giao thông.
2. Phạt tiền từ 4.000.000đ đến 6.000.000đ đối với:
a) Chạy quá tốc độ từ 20km/h đến dưới 35km/h;
3. Hình thức xử phạt bổ sung: tước GPLX 1–3 tháng đối với hành vi tại khoản 2."""
    },
    {
        "role": "assistant",
        "content": """<thinking>
CO_KHOAN: D5→K1,K2,K3. CO_DIEM: K1→Pa,Pb; K2→Pa.
K1 áp dụng với chu_the. K1_Pa→hv_vuot_den_do→ct_phat_tien_800k_1tr (CHE_TAI_CHINH).
K2_Pa→hv_qua_toc_do→ct_phat_tien_4tr_6tr (CHE_TAI_CHINH). K3→hv_qua_toc_do→ct_tuoc_gplx (CHE_TAI_BO_SUNG).
THAM_CHIEU: K3→K2.
</thinking>
{"relations":[{"source_id":"D5","relation_type":"CO_KHOAN","target_id":"D5_K1"},{"source_id":"D5","relation_type":"CO_KHOAN","target_id":"D5_K2"},{"source_id":"D5","relation_type":"CO_KHOAN","target_id":"D5_K3"},{"source_id":"D5_K1","relation_type":"CO_DIEM","target_id":"D5_K1_Pa"},{"source_id":"D5_K1","relation_type":"CO_DIEM","target_id":"D5_K1_Pb"},{"source_id":"D5_K2","relation_type":"CO_DIEM","target_id":"D5_K2_Pa"},{"source_id":"D5_K1","relation_type":"AP_DUNG_VOI","target_id":"chu_the_nguoi_dieu_khien_oto"},{"source_id":"D5_K1_Pa","relation_type":"QUY_DINH_HANH_VI","target_id":"hv_vuot_den_do"},{"source_id":"D5_K1_Pb","relation_type":"QUY_DINH_HANH_VI","target_id":"hv_khong_tuan_hieu_lenh"},{"source_id":"D5_K2_Pa","relation_type":"QUY_DINH_HANH_VI","target_id":"hv_qua_toc_do_20_35"},{"source_id":"hv_vuot_den_do","relation_type":"CHE_TAI_CHINH","target_id":"ct_phat_tien_800k_1tr"},{"source_id":"hv_khong_tuan_hieu_lenh","relation_type":"CHE_TAI_CHINH","target_id":"ct_phat_tien_800k_1tr"},{"source_id":"hv_qua_toc_do_20_35","relation_type":"CHE_TAI_CHINH","target_id":"ct_phat_tien_4tr_6tr"},{"source_id":"hv_qua_toc_do_20_35","relation_type":"CHE_TAI_BO_SUNG","target_id":"ct_tuoc_gplx_1_3_thang"},{"source_id":"D5_K3","relation_type":"THAM_CHIEU","target_id":"D5_K2"}]}"""
    }
]


def build_relations_only_prompt(
    entity_ids: list[str],
    content: str,
    chunk_id: str,
) -> list[dict]:
    """
    Pass 2: Chỉ extract relations dựa trên entity IDs đã có từ Pass 1.
    Output cực nhỏ (~400–800 tokens) — không bao giờ bị truncate.

    Args:
        entity_ids: Danh sách ID của các entities từ Pass 1
        content: Nội dung chunk gốc (để model hiểu ngữ cảnh)
        chunk_id: ID chunk (dùng để log)

    Returns:
        List messages theo chat format của Qwen
    """
    entity_ids_str = ", ".join(entity_ids)
    messages = [
        {"role": "system", "content": RELATIONS_ONLY_SYSTEM_PROMPT},
        *RELATIONS_ONLY_FEW_SHOT,
        {
            "role": "user",
            "content": f"""Xác định QUAN HỆ giữa các thực thể đã biết dựa trên nội dung.

Entity IDs đã có:
{entity_ids_str}

Nội dung (chunk: {chunk_id}):
{content}"""
        }
    ]
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-REFERENCE DETECTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

CROSS_REF_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam. \
Nhiệm vụ của bạn là tìm tất cả các tham chiếu chéo (cross-references) trong đoạn văn bản được cung cấp.

Tham chiếu chéo là khi một Điều/Khoản/Điểm đề cập đến Điều/Khoản/Điểm khác hoặc văn bản khác.

Ví dụ tham chiếu:
- "theo quy định tại Điều 5 Luật này" → tham chiếu đến Điều 5 cùng luật
- "quy định tại khoản 2 Điều 10" → tham chiếu đến Khoản 2 Điều 10
- "theo quy định của Bộ luật Hình sự" → tham chiếu đến văn bản khác
- "theo Nghị định hướng dẫn" → tham chiếu đến NĐ

Trả về JSON:
{
  "cross_references": [
    {
      "source_chunk_id": "ID chunk đang phân tích",
      "source_location": "tên Điều/Khoản/Điểm chứa tham chiếu",
      "target_dieu": "số điều được tham chiếu (null nếu không rõ)",
      "target_khoan": "số khoản (null nếu không có)",
      "target_van_ban": "tên văn bản được tham chiếu (null nếu cùng luật)",
      "loai": "ap_dung | xem_them | sua_doi | bai_bo | huong_dan",
      "text_goc": "đoạn text gốc chứa tham chiếu"
    }
  ]
}
"""


def build_cross_ref_prompt(chunk_id: str, content: str) -> list[dict]:
    """Prompt để detect tham chiếu chéo giữa các điều khoản."""
    return [
        {"role": "system", "content": CROSS_REF_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Tìm tất cả tham chiếu chéo trong đoạn văn bản sau:

Chunk ID: {chunk_id}

Nội dung:
{content}

Trả về JSON:"""
        }
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CRITIC AGENT VALIDATION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

CRITIC_VALIDATION_SYSTEM_PROMPT = """Bạn là Critic Agent của hệ thống RAG pháp luật. \
Nhiệm vụ của bạn là kiểm tra xem context đã retrieved có đủ thông tin để trả lời câu hỏi không.

Bạn được cung cấp:
1. Câu hỏi của người dùng
2. Context đã retrieved (các chunks từ vector DB)
3. Knowledge Graph: danh sách các Điều/Khoản/Điểm liên quan và các tham chiếu của chúng

Hãy xác định:
- Câu hỏi cần những thông tin gì?
- Context đã có đủ chưa?
- Nếu thiếu, cần retrieve thêm gì? (cụ thể: Điều/Khoản/Điểm nào, văn bản nào)

Trả về JSON:
{
  "is_sufficient": true/false,
  "reasoning": "Giải thích ngắn gọn",
  "missing_info": [
    {
      "mo_ta": "Thiếu thông tin gì",
      "can_retrieve": ["chunk_id_1", "chunk_id_2"],
      "ly_do": "Tại sao cần thông tin này"
    }
  ],
  "confidence": 0.0-1.0
}
"""


def build_critic_prompt(
    question: str,
    retrieved_context: str,
    graph_context: str,
) -> list[dict]:
    """
    Prompt cho Critic Agent khi cần validate độ đầy đủ của retrieved context.

    Args:
        question: Câu hỏi người dùng
        retrieved_context: Nội dung đã retrieve được (ghép từ các chunks)
        graph_context: Thông tin từ graph (các Điều liên quan, tham chiếu)
    """
    return [
        {"role": "system", "content": CRITIC_VALIDATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""## Câu hỏi người dùng:
{question}

## Context đã retrieved:
{retrieved_context}

## Thông tin từ Knowledge Graph:
{graph_context}

## Đánh giá độ đầy đủ của context:"""
        }
    ]
