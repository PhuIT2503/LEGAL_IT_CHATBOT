"""
graph_builder.py
================
Xây dựng graph objects từ LLM extraction results.
Normalize entities, resolve cross-references, deduplicate.
"""

import re
import json
import logging
import unicodedata
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────

def normalize_id(text: str) -> str:
    """
    Chuẩn hóa ID thành snake_case, bỏ dấu tiếng Việt.
    """
    # Bỏ dấu tiếng Việt cơ bản
    replacements = {
        'à':'a','á':'a','ả':'a','ã':'a','ạ':'a',
        'ă':'a','ắ':'a','ặ':'a','ẳ':'a','ẵ':'a','ằ':'a',
        'â':'a','ấ':'a','ậ':'a','ẩ':'a','ẫ':'a','ầ':'a',
        'è':'e','é':'e','ẻ':'e','ẽ':'e','ẹ':'e',
        'ê':'e','ế':'e','ệ':'e','ể':'e','ễ':'e','ề':'e',
        'ì':'i','í':'i','ỉ':'i','ĩ':'i','ị':'i',
        'ò':'o','ó':'o','ỏ':'o','õ':'o','ọ':'o',
        'ô':'o','ố':'o','ộ':'o','ổ':'o','ỗ':'o','ồ':'o',
        'ơ':'o','ớ':'o','ợ':'o','ở':'o','ỡ':'o','ờ':'o',
        'ù':'u','ú':'u','ủ':'u','ũ':'u','ụ':'u',
        'ư':'u','ứ':'u','ự':'u','ử':'u','ữ':'u','ừ':'u',
        'ỳ':'y','ý':'y','ỷ':'y','ỹ':'y','ỵ':'y',
        'đ':'d','Đ':'D',
        'À':'A','Á':'A','Ả':'A','Ã':'A','Ạ':'A',
    }
    for viet, ascii_char in replacements.items():
        text = text.replace(viet, ascii_char)

    # Chuyển về lowercase, bỏ ký tự đặc biệt
    text = re.sub(r'[^a-zA-Z0-9_]', '_', text.lower())
    text = re.sub(r'_+', '_', text).strip('_')
    return text


def get_van_ban_id(filename: str) -> str:
    """Tạo van_ban_id từ tên file .docx (giống VBPLChunker.extract_doc_id)."""
    base = re.sub(r'\.docx$', '', filename, flags=re.IGNORECASE)
    slug = re.sub(r'[^a-zA-Z0-9]', '_', base)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug.lower()


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class LegalGraphBuilder:
    """
    Tổng hợp tất cả extraction results thành một graph thống nhất.
    
    Graph được biểu diễn bằng:
    - nodes: dict {node_id: node_data}
    - edges: list of {source, target, relation, properties}
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}      # id -> node_data
        self.edges: list[dict] = []            # list of edge dicts
        self._edge_set: set = set()            # để deduplicate edges
        self.van_ban_nodes: dict[str, dict] = {}  # van_ban_id -> VanBan node

    def add_van_ban_node(self, van_ban_id: str, ten: str, loai: str, nam: Optional[int] = None, so_hieu: str = ""):
        """Thêm node VanBan (văn bản pháp luật)."""
        node = {
            "id": van_ban_id,
            "type": "VanBan",
            "ten": ten,
            "loai": loai,  # "Luat" | "NghiDinh" | "ThongTu" | ...
            "so_hieu": so_hieu,
            "nam": nam,
        }
        self.nodes[van_ban_id] = node
        self.van_ban_nodes[van_ban_id] = node
        return node

    def merge_entities(self, entities: list[dict], van_ban_id: str) -> dict[str, str]:
        """
        Merge entities vào graph, tránh duplicate.
        
        Returns:
            Mapping {original_id -> canonical_id} để fix relations
        """
        id_mapping = {}

        for entity in entities:
            original_id = entity.get("id", "")
            entity_type = entity.get("type", "")

            if not original_id or not entity_type:
                continue

            # Tạo canonical ID bằng cách prefix với van_ban_id cho các entity Điều/Khoản/Điểm
            # Để tránh conflict giữa các văn bản khác nhau (đều có D1, D2, ...)
            if entity_type in ("Dieu", "Khoan", "Diem"):
                # Thêm prefix van_ban_id
                canonical_id = f"{van_ban_id}__{original_id}"
            else:
                # Với HanhVi, CheTai, KhaiNiem: dùng ID như LLM tạo
                # nhưng cũng prefix để tránh conflict
                canonical_id = f"{van_ban_id}__{original_id}"

            id_mapping[original_id] = canonical_id

            # Nếu node đã tồn tại, merge properties (không ghi đè)
            if canonical_id not in self.nodes:
                entity_data = dict(entity)
                entity_data["id"] = canonical_id
                entity_data["original_id"] = original_id
                entity_data["van_ban_id"] = van_ban_id
                self.nodes[canonical_id] = entity_data
            else:
                # Merge properties mới vào node hiện tại
                existing = self.nodes[canonical_id]
                for k, v in entity.items():
                    if k not in existing or existing[k] is None:
                        existing[k] = v

        return id_mapping

    def add_relations(self, relations: list[dict], id_mapping: dict[str, str]):
        """
        Thêm relations vào graph, sử dụng canonical IDs từ id_mapping.
        """
        for rel in relations:
            source_id = rel.get("source_id", "")
            target_id = rel.get("target_id", "")
            relation_type = rel.get("relation_type", "")

            if not source_id or not target_id or not relation_type:
                continue

            # Map sang canonical IDs
            canonical_source = id_mapping.get(source_id, source_id)
            canonical_target = id_mapping.get(target_id, target_id)

            # Deduplicate
            edge_key = (canonical_source, canonical_target, relation_type)
            if edge_key in self._edge_set:
                continue
            self._edge_set.add(edge_key)

            edge_data = {
                "source": canonical_source,
                "target": canonical_target,
                "relation": relation_type,
                "properties": {
                    k: v for k, v in rel.items()
                    if k not in ("source_id", "target_id", "relation_type")
                }
            }
            self.edges.append(edge_data)

    def process_extraction_result(self, result: dict):
        """
        Xử lý một extraction result (từ một chunk của một văn bản).
        """
        van_ban_id = result.get("van_ban_id", "unknown")
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # Merge entities và lấy ID mapping
        id_mapping = self.merge_entities(entities, van_ban_id)

        # Thêm relations
        self.add_relations(relations, id_mapping)

    def process_all_results(self, all_results: list[dict]):
        """
        Xử lý toàn bộ extraction results từ tất cả văn bản.
        """
        logger.info(f"Processing {len(all_results)} extraction results...")
        for result in all_results:
            self.process_extraction_result(result)
        logger.info(f"Graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def resolve_cross_document_references(self):
        """
        Sau khi build xong graph, resolve các tham chiếu chéo giữa văn bản.
        Ví dụ: Điều 24 Luật BVDLCN tham chiếu đến "Bộ luật Hình sự" → tìm node tương ứng.
        """
        # Tìm tất cả THAM_CHIEU edges
        tham_chieu_edges = [e for e in self.edges if e["relation"] == "THAM_CHIEU"]

        for edge in tham_chieu_edges:
            target_id = edge["target"]
            # Nếu target node chưa tồn tại → đó là placeholder từ văn bản khác
            if target_id not in self.nodes:
                # Tạo placeholder node
                self.nodes[target_id] = {
                    "id": target_id,
                    "type": "Dieu",
                    "ten": f"[Placeholder] {target_id}",
                    "is_placeholder": True,
                }
                logger.warning(f"Created placeholder node: {target_id}")

    def get_summary(self) -> dict:
        """Thống kê graph."""
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            type_counts[node.get("type", "unknown")] += 1

        relation_counts = defaultdict(int)
        for edge in self.edges:
            relation_counts[edge["relation"]] += 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": dict(type_counts),
            "relation_types": dict(relation_counts),
        }

    def to_dict(self) -> dict:
        """Export graph dưới dạng dict (để lưu JSON)."""
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "summary": self.get_summary(),
        }

    def save(self, path: str):
        """Lưu graph ra file JSON."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Graph saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "LegalGraphBuilder":
        """Load graph từ file JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        builder = cls()
        for node in data.get("nodes", []):
            builder.nodes[node["id"]] = node
        for edge in data.get("edges", []):
            edge_key = (edge["source"], edge["target"], edge["relation"])
            builder._edge_set.add(edge_key)
            builder.edges.append(edge)

        return builder


# ─────────────────────────────────────────────────────────────────────────────
# VAN BAN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# Danh sách metadata văn bản trong data/keep
# Được dùng để tạo VanBan nodes và map tên file → van_ban_id
VAN_BAN_REGISTRY = [
    {
        "filename": "Luật An ninh mạng 2025.docx",
        "ten": "Luật An ninh mạng 2025",
        "loai": "Luat",
        "so_hieu": "Luật An ninh mạng",
        "nam": 2025,
    },
    {
        "filename": "Luật Bảo vệ dữ liệu cá nhân 2025.docx",
        "ten": "Luật Bảo vệ dữ liệu cá nhân 2025",
        "loai": "Luat",
        "so_hieu": "Luật BVDLCN",
        "nam": 2025,
    },
    {
        "filename": "Luật Bảo vệ quyền lợi người tiêu dùng 2023.docx",
        "ten": "Luật Bảo vệ quyền lợi người tiêu dùng 2023",
        "loai": "Luat",
        "so_hieu": "Luật BVQLNTD",
        "nam": 2023,
    },
    {
        "filename": "Luật CNTT 2006 (sửa đổi, bổ sung bởi Luật Quy hoạch 2017, Luật GDĐT 2023, Luật Viễn thông 2023).docx",
        "ten": "Luật Công nghệ thông tin 2006 (sửa đổi 2023)",
        "loai": "Luat",
        "so_hieu": "67/2006/QH11",
        "nam": 2006,
    },
    {
        "filename": "Luật Công nghiệp công nghệ số 2025.docx",
        "ten": "Luật Công nghiệp công nghệ số 2025",
        "loai": "Luat",
        "so_hieu": "Luật CNCNS",
        "nam": 2025,
    },
    {
        "filename": "Luật Dữ liệu 2024.docx",
        "ten": "Luật Dữ liệu 2024",
        "loai": "Luat",
        "so_hieu": "Luật Dữ liệu",
        "nam": 2024,
    },
    {
        "filename": "Luật Giao dịch điện tử 2023.docx",
        "ten": "Luật Giao dịch điện tử 2023",
        "loai": "Luat",
        "so_hieu": "20/2023/QH15",
        "nam": 2023,
    },
    {
        "filename": "Luật Viễn thông 2023.docx",
        "ten": "Luật Viễn thông 2023",
        "loai": "Luat",
        "so_hieu": "24/2023/QH15",
        "nam": 2023,
    },
    {
        "filename": "Nghị Định 147 2024 NĐ-CP.docx",
        "ten": "Nghị định 147/2024/NĐ-CP về quản lý, cung cấp, sử dụng dịch vụ Internet và thông tin mạng",
        "loai": "NghiDinh",
        "so_hieu": "147/2024/NĐ-CP",
        "nam": 2024,
    },
    {
        "filename": "Nghị Định 15 2020 NĐ-CP (sửa đổi, bổ sung Nghị Định 14 2022).docx",
        "ten": "Nghị định 15/2020/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực bưu chính, viễn thông, tần số vô tuyến điện, CNTT và giao dịch điện tử",
        "loai": "NghiDinh",
        "so_hieu": "15/2020/NĐ-CP",
        "nam": 2020,
    },
    {
        "filename": "Nghị Định 17 2023 NĐ-CP.docx",
        "ten": "Nghị định 17/2023/NĐ-CP quy định chi tiết một số điều và biện pháp thi hành Luật Sở hữu trí tuệ",
        "loai": "NghiDinh",
        "so_hieu": "17/2023/NĐ-CP",
        "nam": 2023,
    },
    {
        "filename": "Nghị Định 211 2025 NĐ-CP.docx",
        "ten": "Nghị định 211/2025/NĐ-CP",
        "loai": "NghiDinh",
        "so_hieu": "211/2025/NĐ-CP",
        "nam": 2025,
    },
    {
        "filename": "Nghị Định 52 2013 NĐ-CP (sửa đổi bởi Nghị Định 08 2018 NĐ-CP và Nghị Định 85 2021 NĐ-CP).docx",
        "ten": "Nghị định 52/2013/NĐ-CP về thương mại điện tử (sửa đổi 2018, 2021)",
        "loai": "NghiDinh",
        "so_hieu": "52/2013/NĐ-CP",
        "nam": 2013,
    },
    {
        "filename": "Nghị Định 52 2024 NĐ-CP.docx",
        "ten": "Nghị định 52/2024/NĐ-CP",
        "loai": "NghiDinh",
        "so_hieu": "52/2024/NĐ-CP",
        "nam": 2024,
    },
    {
        "filename": "Nghị Định 53 2022 NĐ-CP.docx",
        "ten": "Nghị định 53/2022/NĐ-CP quy định chi tiết một số điều của Luật An ninh mạng",
        "loai": "NghiDinh",
        "so_hieu": "53/2022/NĐ-CP",
        "nam": 2022,
    },
    {
        "filename": "Nghị Định 71 2007 NĐ-CP.docx",
        "ten": "Nghị định 71/2007/NĐ-CP quy định chi tiết và hướng dẫn thi hành một số điều của Luật CNTT",
        "loai": "NghiDinh",
        "so_hieu": "71/2007/NĐ-CP",
        "nam": 2007,
    },
    {
        "filename": "Nghị Định 85 2016 NĐ-CP.docx",
        "ten": "Nghị định 85/2016/NĐ-CP về bảo đảm an toàn hệ thống thông tin theo cấp độ",
        "loai": "NghiDinh",
        "so_hieu": "85/2016/NĐ-CP",
        "nam": 2016,
    },
    {
        "filename": "2023_361 + 362_11-VBHN-VPQH.docx",
        "ten": "11-VBHN-VPQH (Văn bản hợp nhất Luật Sở hữu trí tuệ 2022)",
        "loai": "Luat",
        "so_hieu": "11-VBHN-VPQH",
        "nam": 2023,
    },
]


def get_van_ban_meta(filename: str) -> Optional[dict]:
    """Lấy metadata văn bản từ filename."""
    normalized_filename = unicodedata.normalize("NFC", filename)
    for meta in VAN_BAN_REGISTRY:
        if unicodedata.normalize("NFC", meta["filename"]) == normalized_filename:
            return meta
    return None
