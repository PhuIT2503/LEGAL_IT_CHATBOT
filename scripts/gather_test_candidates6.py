"""Lay them 1 mieng nho ung vien cat3 (structural_multi_part) de bu du 300 tong
(hien 297), loai tru dung dieu_id da dung."""
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
while len(cat3_final) < 7 and any(by_vb.values()):
    for vb in list(by_vb.keys()):
        if by_vb[vb]:
            cat3_final.append(by_vb[vb].pop(0))
        if len(cat3_final) >= 7:
            break

print(f"CAT3: con lai {len(cat3_pool)} ung vien, chon {len(cat3_final)}.")
with open("data/cat3top_batch1.json", "w", encoding="utf-8") as f:
    json.dump(cat3_final, f, ensure_ascii=False, indent=2)
print("Wrote data/cat3top_batch1.json")
