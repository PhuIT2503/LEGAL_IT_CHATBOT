"""Khao sat tran thuc te so ung vien THAT con lai cho 4 nhom, loai tru dieu_id da dung trong 50 case dau."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge_graph.neo4j_ingest import Neo4jGraphIngestor
from src.retrieval.qdrant_hybrid_search import _make_client

db = Neo4jGraphIngestor(uri="bolt://neo4j:7687", user="neo4j", password="legal_kg_2024")
qc = _make_client("data/.qdrant", None)

# ── Lay danh sach dieu_id DA DUNG trong 50 case dau (loai tru) ──
existing = json.load(open("data/eval_testset.json", encoding="utf-8"))
used_dieu_ids = set()
for c in existing:
    for d in c.get("dieu_ids", []):
        used_dieu_ids.add(d)
print(f"Da dung {len(used_dieu_ids)} dieu_id trong 50 case dau.")

# ── Bang tra content (dung chung) ──
print("Dang quet toan bo parent chunks...")
all_parents = []
offset = None
while True:
    pts, offset = qc.scroll(collection_name="legal_parent_chunks", with_payload=True, limit=256, offset=offset)
    all_parents.extend(p.payload or {} for p in pts)
    if not offset:
        break

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from knowledge_graph.graph_builder import to_dieu_node_id  # noqa: E402

dieu_to_content = {}
dieu_to_raw = {}  # canonical -> (van_ban_id_raw, dieu_id_raw) de tra Qdrant content chinh xac hon neu can
for pl in all_parents:
    canon = to_dieu_node_id(pl.get("van_ban_id", ""), pl.get("dieu_id", ""))
    if canon:
        dieu_to_content[canon] = pl.get("content", "")

print(f"Bang tra: {len(dieu_to_content)} Dieu co content.")


def get_content(dieu_id):
    return dieu_to_content.get(dieu_id)


# ── NHOM 1: compound-penalty, mo rong QUA MOI van ban co CHE_TAI_BO_SUNG-like text,
#    khong chi ND15/2020 -- quet raw text "Hinh thuc xu phat bo sung" toan corpus ──
cat1_matches = []
for pl in all_parents:
    content = pl.get("content", "")
    if "Hình thức xử phạt bổ sung" in content or "hình thức xử phạt bổ sung" in content:
        canon = to_dieu_node_id(pl.get("van_ban_id", ""), pl.get("dieu_id", ""))
        if canon and canon not in used_dieu_ids:
            cat1_matches.append({"dieu_id": canon, "van_ban": pl.get("van_ban_id", ""), "content": content})

print(f"\nCat1 (compound-penalty) ung vien MOI (chua dung): {len(cat1_matches)}")
vb_count1 = {}
for m in cat1_matches:
    vb_count1[m["van_ban"]] = vb_count1.get(m["van_ban"], 0) + 1
print("  Theo van ban:", vb_count1)

# ── NHOM 2: cross-Dieu THAM_CHIEU, mo rong TOAN BO van ban (ke ca ND15/2020),
#    loai tru cap da dung ──
cross_ref_rows = db.run_query("""
    MATCH (a:Dieu)-[:THAM_CHIEU]->(b:Dieu)
    WHERE a.id <> b.id AND a.ten IS NOT NULL AND b.ten IS NOT NULL
    RETURN DISTINCT a.van_ban_id AS vb, a.id AS a_id, a.ten AS a_ten, b.id AS b_id, b.ten AS b_ten
""")
cat2_candidates = []
for r in cross_ref_rows:
    r = dict(r)
    if r["a_id"] in used_dieu_ids or r["b_id"] in used_dieu_ids:
        continue
    a_content = get_content(r["a_id"])
    b_content = get_content(r["b_id"])
    bad_title = "[" in (r["a_ten"] or "") or "[" in (r["b_ten"] or "") or r["a_ten"] in ("...", None) or r["b_ten"] in ("...", None)
    if a_content and b_content and not bad_title:
        cat2_candidates.append({
            "a_dieu_id": r["a_id"], "a_ten": r["a_ten"], "a_content": a_content,
            "b_dieu_id": r["b_id"], "b_ten": r["b_ten"], "b_content": b_content,
            "van_ban": r["vb"],
        })

print(f"\nCat2 (cross-ref) ung vien MOI (chua dung): {len(cat2_candidates)}")
vb_count2 = {}
for c in cat2_candidates:
    vb_count2[c["van_ban"]] = vb_count2.get(c["van_ban"], 0) + 1
print("  Theo van ban:", vb_count2)

# ── NHOM 3: structural multi-part, ha nguong xuong >=3, khong gioi han per-van-ban (lay het) ──
struct_rows = db.run_query("""
    MATCH (d:Dieu)
    OPTIONAL MATCH (d)-[:CO_KHOAN|CO_DIEM*1..4]->(struc)
    WITH d, count(DISTINCT struc) AS total
    WHERE total >= 3
    RETURN d.id AS dieu_id, d.ten AS ten, d.van_ban_id AS vb, total
    ORDER BY total DESC
""")
cat3_candidates = []
for r in struct_rows:
    r = dict(r)
    if r["dieu_id"] in used_dieu_ids:
        continue
    content = get_content(r["dieu_id"])
    if not content:
        continue
    cat3_candidates.append({"dieu_id": r["dieu_id"], "ten": r["ten"], "van_ban": r["vb"], "total_parts": r["total"], "content": content})

print(f"\nCat3 (structural) ung vien MOI (chua dung, total_parts>=3): {len(cat3_candidates)}")
vb_count3 = {}
for c in cat3_candidates:
    vb_count3[c["van_ban"]] = vb_count3.get(c["van_ban"], 0) + 1
print("  Theo van ban:", vb_count3)

# ── NHOM 4: control, <=1 phan, khong THAM_CHIEU, khong gioi han per-van-ban ──
control_rows = db.run_query("""
    MATCH (d:Dieu)
    OPTIONAL MATCH (d)-[:CO_KHOAN|CO_DIEM*1..4]->(struc)
    WITH d, count(DISTINCT struc) AS total
    WHERE total <= 1
    OPTIONAL MATCH (d)-[:THAM_CHIEU]-(other)
    WITH d, total, count(other) AS ref_count
    WHERE ref_count = 0
    RETURN d.id AS dieu_id, d.ten AS ten, d.van_ban_id AS vb, total
""")
cat4_candidates = []
for r in control_rows:
    r = dict(r)
    if r["dieu_id"] in used_dieu_ids:
        continue
    content = get_content(r["dieu_id"])
    if not content or len(content) < 80 or len(content) > 1200:
        continue
    cat4_candidates.append({"dieu_id": r["dieu_id"], "ten": r["ten"], "van_ban": r["vb"], "content": content})

print(f"\nCat4 (control) ung vien MOI (chua dung): {len(cat4_candidates)}")
vb_count4 = {}
for c in cat4_candidates:
    vb_count4[c["van_ban"]] = vb_count4.get(c["van_ban"], 0) + 1
print("  Theo van ban:", vb_count4)

with open("data/test_gather_raw2.json", "w", encoding="utf-8") as f:
    json.dump({
        "cat1_compound_penalty": cat1_matches,
        "cat2_cross_ref": cat2_candidates,
        "cat3_structural": cat3_candidates,
        "cat4_control": cat4_candidates,
    }, f, ensure_ascii=False, indent=2)

print(f"\n=== TONG UNG VIEN THAT CON LAI: cat1={len(cat1_matches)}, cat2={len(cat2_candidates)}, "
      f"cat3={len(cat3_candidates)}, cat4={len(cat4_candidates)} ===")
print("Da luu data/test_gather_raw2.json")
