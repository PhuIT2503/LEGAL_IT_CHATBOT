"""
Script tạm: thu thập ỨNG VIÊN THẬT cho bộ test đánh giá (4 nhóm), lấy trực tiếp
từ Neo4j (KG) + Qdrant (raw text) — KHÔNG tự bịa nội dung. Output 1 file JSON
duy nhất để dùng làm nguyên liệu viết câu hỏi + checklist đáp án.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_graph.neo4j_ingest import Neo4jGraphIngestor
from src.retrieval.qdrant_hybrid_search import _make_client, _point_id

db = Neo4jGraphIngestor(uri="bolt://neo4j:7687", user="neo4j", password="legal_kg_2024")
qc = _make_client("data/.qdrant", None)


def fetch_parent_by_dieu(van_ban_id_raw: str, dieu_id_raw: str):
    """Tìm parent content thật trong Qdrant theo (van_ban_id, dieu_id) — dùng scroll vì id gốc giữ hoa/thường khác Neo4j."""
    offset = None
    while True:
        pts, offset = qc.scroll(collection_name="legal_parent_chunks", with_payload=True, limit=256, offset=offset)
        for p in pts:
            pl = p.payload or {}
            if pl.get("van_ban_id") == van_ban_id_raw and pl.get("dieu_id") == dieu_id_raw:
                return pl.get("content", "")
        if not offset:
            break
    return None


# ── Nhóm 1: same-Dieu multi-part CHE TAI (co "Hinh thuc xu phat bo sung" that trong van ban) ──
CAT1_DIEU_IDS = [
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D98",   # mang xa hoi
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D105",  # tro choi dien tu
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D99",   # trang thong tin dien tu
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D47",   # ten mien
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D102",  # luu tru truyen dua du lieu
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D93",   # nhap khau san pham an toan TT mang
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D95",   # thu dien tu tin nhan
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D111",  # chung thu so nuoc ngoai
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D33",   # hop dong theo mau TMDT
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D21",   # dich vu vien thong thue bao
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D26",   # dich vu vien thong cong ich
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D65",   # chung chi vo tuyen dien vien
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D69",   # an toan buc xa vo tuyen dien
    "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022_D64",   # (case da verify truoc do)
]
VB_ND15 = "Ngh_nh_15_2020_N_CP_s_a_i_b_sung_Ngh_nh_14_2022"

cat1 = []
for dieu_id in CAT1_DIEU_IDS:
    content = fetch_parent_by_dieu(VB_ND15, dieu_id)
    if content:
        cat1.append({"dieu_id": dieu_id, "van_ban": "Nghị định 15/2020/NĐ-CP (sửa đổi, bổ sung NĐ 14/2022)", "content": content})

# ── Nhóm 2: cross-Dieu THAM_CHIEU that (da tung kham pha 1 phan truoc do, mo rong them) ──
cross_ref_rows = db.run_query("""
    MATCH (a:Dieu)-[:THAM_CHIEU]->(b:Dieu)
    WHERE a.id <> b.id AND a.ten IS NOT NULL AND b.ten IS NOT NULL
    RETURN DISTINCT a.van_ban_id AS vb, a.id AS a_id, a.ten AS a_ten, b.id AS b_id, b.ten AS b_ten
""")
cat2_raw = [dict(r) for r in cross_ref_rows]

# Loc da dang theo van_ban, uu tien cac van ban chua dung o nhom 1
seen_vb = set()
cat2_candidates = []
for r in cat2_raw:
    key = (r["a_id"], r["b_id"])
    if r["vb"] == "ngh_nh_15_2020_n_cp_s_a_i_b_sung_ngh_nh_14_2022":
        continue  # danh rieng cho nhom 1
    cat2_candidates.append(r)

print(f"Cross-ref candidates (excl ND15): {len(cat2_candidates)}")
for r in cat2_candidates[:30]:
    print(r["vb"], "|", r["a_id"], "->", r["b_id"], "|", r["a_ten"], "->", r["b_ten"])

with open("data/test_gather_raw.json", "w", encoding="utf-8") as f:
    json.dump({"cat1_compound_penalty": cat1, "cat2_cross_ref_candidates": cat2_candidates}, f, ensure_ascii=False, indent=2)

print(f"\nCat1: {len(cat1)} candidates lay duoc content that.")
print("Da luu data/test_gather_raw.json")
