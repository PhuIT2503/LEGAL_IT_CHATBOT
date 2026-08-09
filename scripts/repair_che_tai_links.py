"""
scripts/repair_che_tai_links.py
================================
Vá các node CheTai bị "mồ côi" trong Neo4j — node chế tài đã được trích xuất
nhưng KHÔNG có cạnh CHE_TAI_CHINH/CHE_TAI_BO_SUNG nào trỏ vào, nên Critic Agent
không bao giờ nhìn thấy.

VÌ SAO CẦN (đo thật trên đồ thị hiện tại, trước khi vá):
    CheTai                        1026 node
    CHE_TAI_CHINH                  488 cạnh
    CHE_TAI_BO_SUNG                108 cạnh
    -> CheTai mồ côi               842 node
    HanhVi có CẢ hình phạt chính lẫn bổ sung:   3
    Số Điều kích hoạt được tín hiệu "chế tài kép": 1  (trên tổng 1211 Điều)

Tín hiệu "chế tài kép" là 1 trong 3 tín hiệu cốt lõi của Critic Agent (xem
src/agents/agent_critic/node_critic_check.py) nhưng thực tế chỉ bắn được ở ĐÚNG
1 Điều — không phải vì luật ít chế tài bổ sung, mà vì bước trích xuất bằng LLM
tạo ra node chế tài rồi không nối nó vào HanhVi nào.

CÁCH VÁ (thuần luật lệ, KHÔNG gọi lại LLM):
  1. Lấy các CheTai mồ côi CÓ mo_ta và chunk_id (chunk_id trỏ tới Điều chứa nó).
  2. Phân loại mo_ta theo TỪ VỰNG CHUẨN của Luật Xử lý vi phạm hành chính:
       - "Phạt tiền", "Phạt cảnh cáo"              -> hình phạt CHÍNH
       - "Tịch thu", "Tước quyền sử dụng",
         "Đình chỉ hoạt động", "Trục xuất"          -> hình phạt BỔ SUNG
       - "Buộc ..."                                 -> biện pháp khắc phục hậu quả
                                                       (Critic xếp chung nhóm bổ sung)
  3. Nối vào MỌI HanhVi thuộc CÙNG MỘT ĐIỀU (cùng chunk_id).
  4. BỎ QUA phần còn lại. Đây là điểm quan trọng: 756/842 node mồ côi KHÔNG
     PHẢI chế tài (mo_ta kiểu "Quyền tác giả", "Quyền tác giả kịch bản" — LLM
     gán nhầm nhãn CheTai cho khái niệm). Nối bừa chúng vào sẽ khiến Critic
     báo "chế tài kép" ở những Điều không hề có chế tài nào.

VÌ SAO NỐI VÀO MỌI HanhVi TRONG ĐIỀU LÀ CHẤP NHẬN ĐƯỢC:
Đồ thị không đủ thông tin để biết chế tài bổ sung thuộc về hành vi CỤ THỂ nào —
property chunk_id chỉ trỏ tới Điều, không tới Khoản (đã ghi rõ trong docstring
của find_compound_penalty_behaviors). Mà hành động cuối cùng của Critic khi
phát hiện chế tài kép cũng chỉ là "lấy TOÀN VĂN Điều đó", nên nối thừa trong
phạm vi cùng 1 Điều không làm sai kết quả retrieval.

Cách chạy:
    docker compose --profile app run --rm --no-deps app python scripts/repair_che_tai_links.py            # xem trước, KHÔNG ghi
    docker compose --profile app run --rm --no-deps app python scripts/repair_che_tai_links.py --apply    # ghi thật

Chạy lại nhiều lần an toàn: dùng MERGE nên không tạo cạnh trùng.
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase

# Từ vựng chuẩn của Luật Xử lý vi phạm hành chính 2012 (Điều 21: hình thức xử
# phạt chính / bổ sung) — dùng làm luật lệ phân loại thay cho LLM.
RE_CHINH = re.compile(r"^\s*(phạt\s+tiền|phạt\s+cảnh\s+cáo|cảnh\s+cáo|phạt\s+từ|phạt\s+\d)", re.IGNORECASE)
RE_BO_SUNG = re.compile(
    r"(tịch\s+thu|tước\s+quyền\s+sử\s+dụng|tước\s+gplx|tước\s+giấy\s+phép|đình\s+chỉ\s+hoạt\s+động|trục\s+xuất)",
    re.IGNORECASE,
)
RE_KHAC_PHUC = re.compile(r"^\s*buộc\s+", re.IGNORECASE)


def classify(mo_ta: str):
    """-> ("CHE_TAI_CHINH"|"CHE_TAI_BO_SUNG", loai) hoặc (None, None) nếu không phải chế tài."""
    if RE_CHINH.search(mo_ta):
        return "CHE_TAI_CHINH", "chinh"
    if RE_BO_SUNG.search(mo_ta):
        return "CHE_TAI_BO_SUNG", "bo_sung"
    if RE_KHAC_PHUC.search(mo_ta):
        return "CHE_TAI_BO_SUNG", "bo_sung"
    return None, None


def do_bo_chi_so(session) -> dict:
    """Các chỉ số dùng để so sánh trước/sau khi vá."""
    one = lambda c: session.run(c).single()[0]
    return {
        "CHE_TAI_CHINH": one("MATCH ()-[r:CHE_TAI_CHINH]->() RETURN count(r)"),
        "CHE_TAI_BO_SUNG": one("MATCH ()-[r:CHE_TAI_BO_SUNG]->() RETURN count(r)"),
        "CheTai mồ côi": one(
            "MATCH (n:CheTai) WHERE NOT ()-[:CHE_TAI_CHINH|CHE_TAI_BO_SUNG]->(n) RETURN count(n)"
        ),
        "HanhVi có cả 2 loại chế tài": one(
            "MATCH (hv:HanhVi)-[:CHE_TAI_CHINH]->() WITH hv "
            "MATCH (hv)-[:CHE_TAI_BO_SUNG]->() RETURN count(DISTINCT hv)"
        ),
        "Điều kích hoạt tín hiệu chế tài kép": one(
            "MATCH (d:Dieu)-[:CO_KHOAN|CO_DIEM*0..4]->(s)-[:QUY_DINH_HANH_VI]->(hv:HanhVi) "
            "MATCH (hv)-[:CHE_TAI_CHINH]->() MATCH (hv)-[:CHE_TAI_BO_SUNG]->() "
            "RETURN count(DISTINCT d)"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vá cạnh CHE_TAI_* cho các node CheTai mồ côi trong Neo4j")
    parser.add_argument("--apply", action="store_true", help="Ghi thật vào Neo4j (mặc định chỉ xem trước)")
    parser.add_argument("--undo", action="store_true", help="Xóa toàn bộ cạnh do script này tạo, trả đồ thị về trạng thái cũ")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "legal_kg_2024"))
    args = parser.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            if args.undo:
                n = session.run(
                    "MATCH ()-[r:CHE_TAI_CHINH|CHE_TAI_BO_SUNG]->() "
                    "WHERE r.repaired_by = 'repair_che_tai_links' "
                    "DELETE r RETURN count(r) AS n"
                ).single()["n"]
                print(f"Đã xóa {n} cạnh do script tạo. Chỉ số hiện tại:")
                for k, v in do_bo_chi_so(session).items():
                    print(f"  {k:<40} {v}")
                return

            truoc = do_bo_chi_so(session)
            print("=== TRƯỚC KHI VÁ ===")
            for k, v in truoc.items():
                print(f"  {k:<40} {v}")

            orphans = [
                dict(r)
                for r in session.run(
                    """
                    MATCH (n:CheTai)
                    WHERE NOT ()-[:CHE_TAI_CHINH|CHE_TAI_BO_SUNG]->(n)
                      AND n.mo_ta IS NOT NULL AND n.chunk_id IS NOT NULL
                    RETURN n.id AS id, n.mo_ta AS mo_ta, n.chunk_id AS chunk_id
                    """
                )
            ]
            print(f"\nCheTai mồ côi có đủ mo_ta + chunk_id: {len(orphans)}")

            thong_ke = Counter()
            ke_hoach = []
            for o in orphans:
                rel, loai = classify(o["mo_ta"])
                if rel is None:
                    thong_ke["bỏ qua (không phải chế tài)"] += 1
                    continue
                so_hanh_vi = session.run(
                    "MATCH (hv:HanhVi {chunk_id: $c}) RETURN count(hv) AS n", {"c": o["chunk_id"]}
                ).single()["n"]
                if so_hanh_vi == 0:
                    thong_ke["bỏ qua (Điều không có HanhVi nào để nối)"] += 1
                    continue
                thong_ke[f"sẽ nối {rel}"] += 1
                ke_hoach.append({**o, "rel": rel, "loai": loai, "so_hanh_vi": so_hanh_vi})

            print("\n=== PHÂN LOẠI ===")
            for k, v in thong_ke.most_common():
                print(f"  {k:<45} {v}")

            print("\n=== VÍ DỤ SẼ NỐI (tối đa 8) ===")
            for k in ke_hoach[:8]:
                print(f"  [{k['rel']}] \"{k['mo_ta'][:60]}\"  -> {k['so_hanh_vi']} HanhVi trong {k['chunk_id']}")

            if not args.apply:
                print(f"\nXem trước — CHƯA ghi gì. Chạy lại với --apply để nối {len(ke_hoach)} chế tài này.")
                return

            print(f"\n=== ĐANG GHI {len(ke_hoach)} chế tài vào Neo4j ===")
            so_canh = 0
            for k in ke_hoach:
                res = session.run(
                    f"""
                    MATCH (ct:CheTai {{id: $ct_id}})
                    SET ct.loai = $loai, ct.repaired_by = 'repair_che_tai_links'
                    WITH ct
                    MATCH (hv:HanhVi {{chunk_id: $chunk_id}})
                    MERGE (hv)-[r:{k['rel']}]->(ct)
                    // Đánh dấu trên CẠNH để hoàn tác được, xem --undo
                    SET r.repaired_by = 'repair_che_tai_links'
                    RETURN count(r) AS n
                    """,
                    {"ct_id": k["id"], "loai": k["loai"], "chunk_id": k["chunk_id"]},
                ).single()
                so_canh += res["n"]
            print(f"Đã tạo/đảm bảo {so_canh} cạnh.")

            sau = do_bo_chi_so(session)
            print("\n=== SAU KHI VÁ ===")
            for k, v in sau.items():
                delta = v - truoc[k]
                dau = f"  ({delta:+d})" if delta else ""
                print(f"  {k:<40} {v}{dau}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
