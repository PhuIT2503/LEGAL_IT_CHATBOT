"""
cypher_export.py
================
Chuyển đổi Legal Knowledge Graph (JSON) sang Cypher scripts
để import vào Neo4j AuraDB sau khi đã có account.

OUTPUT:
    neo4j_import/
    ├── 00_schema.cypher           ← Tạo constraints + indexes (chạy đầu tiên)
    ├── 01_nodes_VanBan.cypher     ← Nodes từng loại
    ├── 01_nodes_Dieu.cypher
    ├── 01_nodes_Khoan.cypher
    ├── 01_nodes_Diem.cypher
    ├── 01_nodes_HanhVi.cypher
    ├── 01_nodes_CheTai.cypher
    ├── 01_nodes_ChuThe.cypher
    ├── 01_nodes_NghiaVu.cypher
    ├── 01_nodes_QuyenHan.cypher
    ├── 01_nodes_KhaiNiem.cypher
    ├── 01_nodes_DieuKien.cypher
    ├── 02_edges_all.cypher        ← Tất cả relationships (chia batch)
    └── README.md                  ← Hướng dẫn import vào Neo4j AuraDB

CÁCH IMPORT VÀO NEO4J AURADB:
    Cách 1 (Browser): Mở Neo4j Browser → paste từng file .cypher → Run
    Cách 2 (Python) : Gọi hàm import_to_neo4j() khi có credentials
"""

import os
import json
import logging
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA CYPHER (chạy đầu tiên trong Neo4j Browser)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_CYPHER = """\
// ══════════════════════════════════════════════════════════════════
// 00_schema.cypher
// Tạo constraints và indexes cho Legal Knowledge Graph
// Chạy file này ĐẦU TIÊN trong Neo4j Browser
// ══════════════════════════════════════════════════════════════════

// ── Unique constraints (mỗi lệnh chạy riêng) ──
CREATE CONSTRAINT van_ban_id IF NOT EXISTS FOR (n:VanBan) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dieu_id IF NOT EXISTS FOR (n:Dieu) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT khoan_id IF NOT EXISTS FOR (n:Khoan) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT diem_id IF NOT EXISTS FOR (n:Diem) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT hanh_vi_id IF NOT EXISTS FOR (n:HanhVi) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT che_tai_id IF NOT EXISTS FOR (n:CheTai) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT chu_the_id IF NOT EXISTS FOR (n:ChuThe) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT nghia_vu_id IF NOT EXISTS FOR (n:NghiaVu) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT quyen_han_id IF NOT EXISTS FOR (n:QuyenHan) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT khai_niem_id IF NOT EXISTS FOR (n:KhaiNiem) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dieu_kien_id IF NOT EXISTS FOR (n:DieuKien) REQUIRE n.id IS UNIQUE;

// ── Full-text indexes ──
CREATE FULLTEXT INDEX hanh_vi_search IF NOT EXISTS FOR (n:HanhVi) ON EACH [n.mo_ta];
CREATE FULLTEXT INDEX khai_niem_search IF NOT EXISTS FOR (n:KhaiNiem) ON EACH [n.ten, n.dinh_nghia];
CREATE FULLTEXT INDEX che_tai_search IF NOT EXISTS FOR (n:CheTai) ON EACH [n.mo_ta];
CREATE FULLTEXT INDEX dieu_search IF NOT EXISTS FOR (n:Dieu) ON EACH [n.ten];
"""

README_CONTENT = """\
# Neo4j AuraDB Import Guide

## Thứ tự import

Chạy lần lượt trong **Neo4j Browser** (db.neo4j.com):

```
1. 00_schema.cypher       ← Tạo constraints & indexes (BẮT BUỘC chạy trước)
2. 01_nodes_VanBan.cypher ← Nodes văn bản pháp luật
3. 01_nodes_Dieu.cypher
4. 01_nodes_Khoan.cypher
5. 01_nodes_Diem.cypher
6. 01_nodes_HanhVi.cypher
7. 01_nodes_CheTai.cypher
8. 01_nodes_ChuThe.cypher
9. 01_nodes_NghiaVu.cypher
10. 01_nodes_QuyenHan.cypher
11. 01_nodes_KhaiNiem.cypher
12. 01_nodes_DieuKien.cypher
13. 02_edges_all.cypher   ← Relationships (chạy SAU KHI đã có tất cả nodes)
```

## Cách paste vào Neo4j Browser

1. Mở https://browser.neo4j.io hoặc AuraDB console
2. Copy nội dung file `.cypher`
3. Paste vào thanh query → Nhấn **Run** (▶)
4. Đợi confirm "Nodes created" / "Relationships created"

## Cách import bằng Python (khi có credentials)

```python
from cypher_export import import_to_neo4j

import_to_neo4j(
    graph_json_path="legal_knowledge_graph.json",
    neo4j_uri="neo4j+s://xxxx.databases.neo4j.io",
    neo4j_user="neo4j",
    neo4j_password="your_password",
)
```

## Lưu ý

- Mỗi lệnh MERGE là idempotent: chạy lại không bị duplicate
- Nếu file `.cypher` quá lớn, chia nhỏ bằng cách copy từng khối `MERGE`
- AuraDB Free tier: giới hạn 200k nodes và 400k relationships
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _escape_str(value) -> str:
    """Escape string value cho Cypher."""
    if value is None:
        return "null"
    s = str(value).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "")
    return f"'{s}'"


def _format_props(props: dict, exclude_keys: set = None) -> str:
    """
    Format dict thành Cypher property map string.
    Ví dụ: {id: 'abc', ten: 'xyz', nam: 2025}
    """
    exclude_keys = exclude_keys or {"type"}
    parts = []
    for k, v in props.items():
        if k in exclude_keys or v is None:
            continue
        if isinstance(v, bool):
            parts.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
        elif isinstance(v, str):
            parts.append(f"{k}: {_escape_str(v)}")
        # Bỏ qua list/dict (Neo4j không support nested object)
    return "{" + ", ".join(parts) + "}"


def _format_props_set(props: dict, var: str = "n", exclude_keys: set = None) -> str:
    """Format SET clauses cho MERGE ... SET n += {...}"""
    exclude_keys = exclude_keys or {"type", "id"}
    parts = []
    for k, v in props.items():
        if k in exclude_keys or v is None:
            continue
        if isinstance(v, bool):
            parts.append(f"{var}.{k} = {str(v).lower()}")
        elif isinstance(v, (int, float)):
            parts.append(f"{var}.{k} = {v}")
        elif isinstance(v, str):
            parts.append(f"{var}.{k} = {_escape_str(v)}")
    return ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# NODE CYPHER GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _generate_node_cypher(node: dict) -> Optional[str]:
    """Tạo Cypher MERGE statement cho một node."""
    node_type = node.get("type", "")
    node_id   = node.get("id", "")

    if not node_type or not node_id:
        return None

    set_clause = _format_props_set(node, var="n")

    if set_clause:
        return (
            f"MERGE (n:{node_type} {{id: {_escape_str(node_id)}}})\n"
            f"SET {set_clause};"
        )
    else:
        return f"MERGE (n:{node_type} {{id: {_escape_str(node_id)}}});"


def _generate_edge_cypher(edge: dict) -> Optional[str]:
    """Tạo Cypher MERGE statement cho một relationship."""
    source_id     = edge.get("source", "")
    target_id     = edge.get("target", "")
    relation_type = edge.get("relation", "")
    props         = edge.get("properties", {})

    if not source_id or not target_id or not relation_type:
        return None

    # Chỉ lấy properties là scalar (str/int/float/bool)
    clean_props = {
        k: v for k, v in props.items()
        if v is not None and isinstance(v, (str, int, float, bool))
    }

    if clean_props:
        set_clause = _format_props_set(clean_props, var="r", exclude_keys=set())
        return (
            f"MATCH (a {{id: {_escape_str(source_id)}}})\n"
            f"MATCH (b {{id: {_escape_str(target_id)}}})\n"
            f"MERGE (a)-[r:{relation_type}]->(b)\n"
            f"SET {set_clause};"
        )
    else:
        return (
            f"MATCH (a {{id: {_escape_str(source_id)}}})\n"
            f"MATCH (b {{id: {_escape_str(target_id)}}})\n"
            f"MERGE (a)-[:{relation_type}]->(b);"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPORT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def export_cypher_scripts(
    graph_data: dict,
    output_dir: str,
    edge_batch_size: int = 500,
):
    """
    Xuất toàn bộ graph thành Cypher scripts.

    Args:
        graph_data   : dict {"nodes": [...], "edges": [...]}
        output_dir   : Thư mục lưu .cypher files
        edge_batch_size: Số edges mỗi file (tránh file quá lớn)
    """
    os.makedirs(output_dir, exist_ok=True)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # ── Schema ──────────────────────────────────────────────
    schema_path = os.path.join(output_dir, "00_schema.cypher")
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(SCHEMA_CYPHER)
    logger.info(f"  Wrote: 00_schema.cypher")

    # ── Nodes (tách theo loại) ───────────────────────────────
    from collections import defaultdict
    nodes_by_type: dict[str, list] = defaultdict(list)
    for node in nodes:
        node_type = node.get("type", "Unknown")
        nodes_by_type[node_type].append(node)

    for node_type, type_nodes in sorted(nodes_by_type.items()):
        lines = [
            f"// ══════════════════════════════════════════════════════\n"
            f"// 01_nodes_{node_type}.cypher\n"
            f"// {len(type_nodes)} nodes loại {node_type}\n"
            f"// ══════════════════════════════════════════════════════\n"
        ]
        for node in type_nodes:
            cypher = _generate_node_cypher(node)
            if cypher:
                lines.append(cypher)
                lines.append("")  # blank line giữa các MERGE

        fname = f"01_nodes_{node_type}.cypher"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"  Wrote: {fname} ({len(type_nodes)} nodes)")

    # ── Edges (chia batch nếu nhiều) ─────────────────────────
    edge_stmts = []
    for edge in edges:
        cypher = _generate_edge_cypher(edge)
        if cypher:
            edge_stmts.append(cypher)

    if len(edge_stmts) <= edge_batch_size:
        # Tất cả vừa 1 file
        _write_edge_file(edge_stmts, output_dir, "02_edges_all.cypher")
    else:
        # Chia batch
        for i in range(0, len(edge_stmts), edge_batch_size):
            batch_num  = i // edge_batch_size + 1
            batch      = edge_stmts[i:i + edge_batch_size]
            fname      = f"02_edges_batch{batch_num:02d}.cypher"
            _write_edge_file(batch, output_dir, fname)

    logger.info(f"  Wrote: {len(edge_stmts)} edge statements")

    # ── README ───────────────────────────────────────────────
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README_CONTENT)
    logger.info(f"  Wrote: README.md")

    # ── Summary JSON ─────────────────────────────────────────
    summary = {
        "total_nodes" : len(nodes),
        "total_edges" : len(edges),
        "node_types"  : {k: len(v) for k, v in nodes_by_type.items()},
        "cypher_files": sorted(os.listdir(output_dir)),
    }
    summary_path = os.path.join(output_dir, "export_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ Cypher export complete → {output_dir}/")
    logger.info(f"   {len(nodes):,} nodes | {len(edges):,} edges | "
                f"{len(list(nodes_by_type.keys()))} node types")
    return summary


def _write_edge_file(stmts: list[str], output_dir: str, fname: str):
    """Ghi danh sách Cypher statements ra file."""
    header = (
        f"// ══════════════════════════════════════════════════════\n"
        f"// {fname}\n"
        f"// {len(stmts)} relationships\n"
        f"// Chạy SAU KHI đã import tất cả nodes\n"
        f"// ══════════════════════════════════════════════════════\n\n"
    )
    fpath = os.path.join(output_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n\n".join(stmts))
        f.write("\n")
    logger.info(f"  Wrote: {fname} ({len(stmts)} edges)")


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Load JSON → Export Cypher (dùng độc lập)
# ─────────────────────────────────────────────────────────────────────────────

def export_from_json(graph_json_path: str, output_dir: Optional[str] = None):
    """
    Load graph từ JSON và export Cypher scripts.
    Dùng độc lập khi đã có file legal_knowledge_graph.json.

    Args:
        graph_json_path: Đường dẫn file JSON graph
        output_dir     : Thư mục output (mặc định: cùng thư mục với JSON)

    Example:
        from cypher_export import export_from_json
        export_from_json("/content/drive/.../legal_knowledge_graph.json")
    """
    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(graph_json_path), "neo4j_import")

    return export_cypher_scripts(graph_data, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Import thẳng vào Neo4j khi có credentials
# ─────────────────────────────────────────────────────────────────────────────

def import_to_neo4j(
    graph_json_path: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    clear_first: bool = False,
):
    """
    Import graph JSON trực tiếp vào Neo4j AuraDB.
    Gọi hàm này khi đã có Neo4j credentials.

    Args:
        graph_json_path: Đường dẫn file legal_knowledge_graph.json
        neo4j_uri      : URI AuraDB (ví dụ: neo4j+s://xxxx.databases.neo4j.io)
        neo4j_user     : Username (thường là 'neo4j')
        neo4j_password : Password
        clear_first    : Xóa graph cũ trước khi import (CẢNH BÁO!)

    Example:
        from cypher_export import import_to_neo4j
        import_to_neo4j(
            graph_json_path = "/content/drive/.../legal_knowledge_graph.json",
            neo4j_uri       = "neo4j+s://xxxx.databases.neo4j.io",
            neo4j_user      = "neo4j",
            neo4j_password  = "your_password",
        )
    """
    from neo4j_ingest import Neo4jGraphIngestor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    with Neo4jGraphIngestor(neo4j_uri, neo4j_user, neo4j_password) as ingestor:
        if not ingestor.verify_connection():
            raise ConnectionError(f"Cannot connect to Neo4j: {neo4j_uri}")

        ingestor.ingest_graph(graph_data, clear_first=clear_first)
        stats = ingestor.get_stats()

        logger.info("\n✅ Neo4j Import Complete!")
        logger.info(f"  Nodes: {stats['total_nodes']:,}")
        logger.info(f"  Edges: {stats['total_edges']:,}")
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI (chạy trực tiếp)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Legal KG JSON → Cypher scripts cho Neo4j"
    )
    parser.add_argument(
        "--graph-json",
        required=True,
        help="Đường dẫn file legal_knowledge_graph.json"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Thư mục output cho Cypher scripts (mặc định: neo4j_import/ cạnh file JSON)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    export_from_json(args.graph_json, args.output_dir)
