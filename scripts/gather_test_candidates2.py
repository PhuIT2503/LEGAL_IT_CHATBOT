"""Thu thap nhom 2 (loc + fetch content that), nhom 3 (structural), nhom 4 (control)."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_graph.neo4j_ingest import Neo4jGraphIngestor
from src.retrieval.qdrant_hybrid_search import _make_client

db = Neo4jGraphIngestor(uri="bolt://neo4j:7687", user="neo4j", password="legal_kg_2024")
qc = _make_client("data/.qdrant", None)

# ── Build 1 lan bang tra: (van_ban_id_lower, dieu_id_lower_suffix) -> content ──
# de tra cuu nhanh theo id Neo4j (van_ban_id da chuan hoa) thay vi phai biet id goc Qdrant.
print("Dang quet toan bo parent chunks (1 lan)...")
all_parents = []
offset = None
while True:
    pts, offset = qc.scroll(collection_name="legal_parent_chunks", with_payload=True, limit=256, offset=offset)
    for p in pts:
        pl = p.payload or {}
        all_parents.append(pl)
    if not offset:
        break
print(f"Tong {len(all_parents)} parent chunks.")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from knowledge_graph.graph_builder import to_dieu_node_id  # noqa: E402

dieu_to_content = {}
for pl in all_parents:
    canon = to_dieu_node_id(pl.get("van_ban_id", ""), pl.get("dieu_id", ""))
    if canon:
        dieu_to_content[canon] = pl.get("content", "")

print(f"Bang tra: {len(dieu_to_content)} Dieu co content.")


def get_content(dieu_id: str):
    return dieu_to_content.get(dieu_id)


# ── Nhom 2: loc ung vien cross-ref (bo placeholder/ten xau, verify content 2 phia) ──
cross_ref_rows = db.run_query("""
    MATCH (a:Dieu)-[:THAM_CHIEU]->(b:Dieu)
    WHERE a.id <> b.id AND a.ten IS NOT NULL AND b.ten IS NOT NULL
      AND NOT a.van_ban_id = "ngh_nh_15_2020_n_cp_s_a_i_b_sung_ngh_nh_14_2022"
    RETURN DISTINCT a.van_ban_id AS vb, a.id AS a_id, a.ten AS a_ten, b.id AS b_id, b.ten AS b_ten
""")
cat2 = []
for r in cross_ref_rows:
    r = dict(r)
    a_content = get_content(r["a_id"])
    b_content = get_content(r["b_id"])
    bad_title = "[" in (r["a_ten"] or "") or "[" in (r["b_ten"] or "") or "..." == r["a_ten"] or "..." == r["b_ten"]
    if a_content and b_content and not bad_title:
        cat2.append({
            "a_dieu_id": r["a_id"], "a_ten": r["a_ten"], "a_content": a_content,
            "b_dieu_id": r["b_id"], "b_ten": r["b_ten"], "b_content": b_content,
            "van_ban": r["vb"],
        })

print(f"\nCat2 (cross-ref, da loc+verify content): {len(cat2)}")
for c in cat2:
    print(" -", c["van_ban"], "|", c["a_dieu_id"], "->", c["b_dieu_id"])

# ── Nhom 3: structural multi-part (KHONG trung nhom 1 (ND15) va nhom 2) ──
used_dieu_ids = {c["a_dieu_id"] for c in cat2} | {c["b_dieu_id"] for c in cat2}
struct_rows = db.run_query("""
    MATCH (d:Dieu)
    WHERE NOT d.van_ban_id = "ngh_nh_15_2020_n_cp_s_a_i_b_sung_ngh_nh_14_2022"
    OPTIONAL MATCH (d)-[:CO_KHOAN|CO_DIEM*1..4]->(struc)
    WITH d, count(DISTINCT struc) AS total
    WHERE total >= 4
    RETURN d.id AS dieu_id, d.ten AS ten, d.van_ban_id AS vb, total
    ORDER BY total DESC
""")
cat3 = []
seen_vb_count = {}
for r in struct_rows:
    r = dict(r)
    if r["dieu_id"] in used_dieu_ids:
        continue
    content = get_content(r["dieu_id"])
    if not content:
        continue
    # Uu tien da dang van ban - toi da 3 case moi van ban de trai deu 13 case tren nhieu luat
    if seen_vb_count.get(r["vb"], 0) >= 3:
        continue
    cat3.append({"dieu_id": r["dieu_id"], "ten": r["ten"], "van_ban": r["vb"], "total_parts": r["total"], "content": content})
    seen_vb_count[r["vb"]] = seen_vb_count.get(r["vb"], 0) + 1
    if len(cat3) >= 15:
        break

print(f"\nCat3 (structural multi-part): {len(cat3)}")
for c in cat3:
    print(" -", c["van_ban"], "|", c["dieu_id"], "| total_parts=", c["total_parts"], "|", c["ten"])

# ── Nhom 4: control (Dieu don gian - <=1 phan, KHONG co THAM_CHIEU vao/ra) ──
control_rows = db.run_query("""
    MATCH (d:Dieu)
    OPTIONAL MATCH (d)-[:CO_KHOAN|CO_DIEM*1..4]->(struc)
    WITH d, count(DISTINCT struc) AS total
    WHERE total <= 1
    OPTIONAL MATCH (d)-[:THAM_CHIEU]-(other)
    WITH d, total, count(other) AS ref_count
    WHERE ref_count = 0
    RETURN d.id AS dieu_id, d.ten AS ten, d.van_ban_id AS vb, total
    ORDER BY rand()
""")
cat4 = []
seen_vb_count4 = {}
for r in control_rows:
    r = dict(r)
    content = get_content(r["dieu_id"])
    if not content or len(content) < 80 or len(content) > 1200:
        continue  # tranh Dieu qua ngan (thieu ngu canh) hoac qua dai (mau thuan voi dinh nghia "don gian")
    if seen_vb_count4.get(r["vb"], 0) >= 3:
        continue
    cat4.append({"dieu_id": r["dieu_id"], "ten": r["ten"], "van_ban": r["vb"], "content": content})
    seen_vb_count4[r["vb"]] = seen_vb_count4.get(r["vb"], 0) + 1
    if len(cat4) >= 15:
        break

print(f"\nCat4 (control): {len(cat4)}")
for c in cat4:
    print(" -", c["van_ban"], "|", c["dieu_id"], "|", c["ten"])

# ── Luu tat ca ──
existing = json.load(open("data/test_gather_raw.json", encoding="utf-8"))
existing["cat2_cross_ref"] = cat2
existing["cat3_structural"] = cat3
existing["cat4_control"] = cat4
with open("data/test_gather_raw.json", "w", encoding="utf-8") as f:
    json.dump(existing, f, ensure_ascii=False, indent=2)
print("\nDa cap nhat data/test_gather_raw.json")
