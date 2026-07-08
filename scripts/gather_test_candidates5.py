"""Chon them ung vien de bo sung cho du 250 case moi (hien co 234), loai tru
dung dieu_id da dung (ke ca 234 case moi da gop), uu tien lay du: cat1 +8, cat2 +tat ca con lai,
cat3 +12, cat4 +6 (co du de bu tru sau khi agent tu loc), chia batch ~15 item/file."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from src.knowledge_graph.neo4j_ingest import Neo4jGraphIngestor

db = Neo4jGraphIngestor(uri="bolt://neo4j:7687", user="neo4j", password="legal_kg_2024")
qc = QdrantClient(path="data/.qdrant")

existing = json.load(open("data/eval_testset.json", encoding="utf-8"))
used_lower = set()
for c in existing:
    for d in c.get("dieu_ids", []):
        used_lower.add(d.lower())
print(f"Da dung {len(used_lower)} dieu_id (tu {len(existing)} case hien co).")

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
for pl in all_parents:
    canon = to_dieu_node_id(pl.get("van_ban_id", ""), pl.get("dieu_id", ""))
    if canon:
        dieu_to_content[canon] = pl.get("content", "")


def get_content(dieu_id):
    return dieu_to_content.get(dieu_id)


def is_used(dieu_id):
    return dieu_id.lower() in used_lower


def batch_write(items, prefix, batch_size=15):
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    for idx, b in enumerate(batches, 1):
        path = f"data/{prefix}_batch{idx}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(b, f, ensure_ascii=False, indent=2)
        print(f"  {path}: {len(b)} item")
    return len(batches)


# ── CAT1: +8 case du phong (bu tru filter cua agent, muc tieu thuc te +5) ──
cat1_all = []
for pl in all_parents:
    content = pl.get("content", "")
    if "Hình thức xử phạt bổ sung" in content or "hình thức xử phạt bổ sung" in content:
        canon = to_dieu_node_id(pl.get("van_ban_id", ""), pl.get("dieu_id", ""))
        if canon and not is_used(canon):
            cat1_all.append({"dieu_id": canon, "van_ban_raw": pl.get("van_ban_id", ""), "content": content})

seen = set()
cat1_dedup = []
for c in cat1_all:
    if c["dieu_id"] not in seen:
        seen.add(c["dieu_id"])
        cat1_dedup.append(c)

cat1_final = cat1_dedup[:8]
print(f"\nCAT1: con lai {len(cat1_dedup)} ung vien chua dung, chon {len(cat1_final)}.")
n1 = batch_write(cat1_final, "cat1more", batch_size=15)

# ── CAT2: lay HET ung vien sach con lai ──
cross_ref_rows = db.run_query("""
    MATCH (a:Dieu)-[:THAM_CHIEU]->(b:Dieu)
    WHERE a.id <> b.id AND a.ten IS NOT NULL AND b.ten IS NOT NULL
    RETURN DISTINCT a.van_ban_id AS vb, a.id AS a_id, a.ten AS a_ten, b.id AS b_id, b.ten AS b_ten
""")
cat2_final = []
for r in cross_ref_rows:
    r = dict(r)
    if is_used(r["a_id"]) or is_used(r["b_id"]):
        continue
    a_content = get_content(r["a_id"])
    b_content = get_content(r["b_id"])
    bad_title = "[" in (r["a_ten"] or "") or "[" in (r["b_ten"] or "") or r["a_ten"] in ("...", None) or r["b_ten"] in ("...", None)
    if a_content and b_content and not bad_title:
        cat2_final.append({
            "a_dieu_id": r["a_id"], "a_ten": r["a_ten"], "a_content": a_content,
            "b_dieu_id": r["b_id"], "b_ten": r["b_ten"], "b_content": b_content,
            "van_ban": r["vb"],
        })
print(f"\nCAT2: con lai {len(cat2_final)} ung vien sach chua dung (lay het).")
n2 = batch_write(cat2_final, "cat2more", batch_size=15)

# ── CAT3: +12 case du phong (muc tieu thuc te +8) ──
struct_rows = db.run_query("""
    MATCH (d:Dieu)
    OPTIONAL MATCH (d)-[:CO_KHOAN|CO_DIEM*1..4]->(struc)
    WITH d, count(DISTINCT struc) AS total
    WHERE total >= 3
    RETURN d.id AS dieu_id, d.ten AS ten, d.van_ban_id AS vb, total
    ORDER BY total DESC
""")
cat3_pool = []
for r in struct_rows:
    r = dict(r)
    if is_used(r["dieu_id"]):
        continue
    content = get_content(r["dieu_id"])
    if not content:
        continue
    cat3_pool.append({"dieu_id": r["dieu_id"], "ten": r["ten"], "van_ban": r["vb"], "total_parts": r["total"], "content": content})

from collections import defaultdict
by_vb = defaultdict(list)
for c in cat3_pool:
    by_vb[c["van_ban"]].append(c)
cat3_final = []
while len(cat3_final) < 12 and any(by_vb.values()):
    for vb in list(by_vb.keys()):
        if by_vb[vb]:
            cat3_final.append(by_vb[vb].pop(0))
        if len(cat3_final) >= 12:
            break
print(f"\nCAT3: con lai {len(cat3_pool)} ung vien chua dung (total_parts>=3), chon {len(cat3_final)}.")
n3 = batch_write(cat3_final, "cat3more", batch_size=15)

# ── CAT4: +6 case du phong ──
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
cat4_pool = []
for r in control_rows:
    r = dict(r)
    if is_used(r["dieu_id"]):
        continue
    content = get_content(r["dieu_id"])
    if not content or len(content) < 80 or len(content) > 1200:
        continue
    cat4_pool.append({"dieu_id": r["dieu_id"], "ten": r["ten"], "van_ban": r["vb"], "content": content})

by_vb4 = defaultdict(list)
for c in cat4_pool:
    by_vb4[c["van_ban"]].append(c)
cat4_final = []
while len(cat4_final) < 6 and any(by_vb4.values()):
    for vb in list(by_vb4.keys()):
        if by_vb4[vb]:
            cat4_final.append(by_vb4[vb].pop(0))
        if len(cat4_final) >= 6:
            break
print(f"\nCAT4: con lai {len(cat4_pool)} ung vien chua dung, chon {len(cat4_final)}.")
n4 = batch_write(cat4_final, "cat4more", batch_size=15)

print(f"\n=== TONG: cat1={len(cat1_final)} ({n1} batch), cat2={len(cat2_final)} ({n2} batch), "
      f"cat3={len(cat3_final)} ({n3} batch), cat4={len(cat4_final)} ({n4} batch) ===")
print(f"TONG CONG: {len(cat1_final)+len(cat2_final)+len(cat3_final)+len(cat4_final)} case ung vien moi")
