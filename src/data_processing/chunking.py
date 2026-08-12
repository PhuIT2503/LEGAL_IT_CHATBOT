import os
import re #(biểu thức chính quy)
import sys
from typing import List, Dict, Any

# Thêm thư mục gốc vào sys.path để import utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from utils.logger import get_logger
    logger = get_logger("ingestion")
except Exception:
    class _FallbackLogger:
        def info(self, message: str) -> None:
            print(message)

        def error(self, message: str) -> None:
            print(message)

    logger = _FallbackLogger()

try:
    import docx
except ImportError:
    raise ImportError("Vui lòng cài đặt python-docx: pip install python-docx")

class VBPLChunker:
    """
    Class hỗ trợ chunking văn bản pháp luật (VBPL) theo cấu trúc Điều - Khoản - Điểm.
    Mỗi phần nhỏ (Lời dẫn Điều, Khoản, Điểm) được chia thành một chunk riêng biệt.
    Tiêu đề Điều đứng một mình được gộp vào chunk nội dung đầu tiên của Điều đó
    để giảm nhiễu khi retrieval; Điều bị bãi bỏ/không có nội dung con vẫn được giữ.
    
    Tất cả các chunk thuộc cùng một Điều sẽ chia sẻ chung một trường `dieu_id`.
    Khi truy xuất (retrieval) trúng một chunk bất kỳ (VD: Điểm), ta dùng `dieu_id` 
    để gom lại toàn bộ các chunk cùng Điều nhằm khôi phục trọn vẹn văn cảnh của Điều đó.
    """
    def __init__(self):
        # Regular expressions để bắt các thành phần của VBPL
        # Bắt "Điều 114.", "Điều 1.", "Điều 1a."
        self.dieu_pattern = re.compile(r'^Điều\s+(\d+[a-zA-Z]*)\s*[\.\:\-]?\s*(.*)', re.IGNORECASE)
        # Bắt "1.", "2.", cả trường hợp văn bản hợp nhất gắn chú thích như "4.[16]"
        self.khoan_pattern = re.compile(r'^(\d+)\.(?:\[\d+\]|\d+)?\s*(.+)')
        # Bắt "a)", "b)", "đ)", cả trường hợp "c)[15]"
        self.diem_pattern = re.compile(r'^([a-zđ])\)(?:\[\d+\]|\d+)?\s*(.+)', re.IGNORECASE)
        # Chú thích cuối văn bản thường không thuộc thân Điều và dễ bị dính vào Điều cuối.
        self.trailing_note_pattern = re.compile(
            r'^(?:\d+\s+|\[\d+\]\s*)(?:Tên|Luật|Nghị định|Thông tư|Quyết định|Điểm|Khoản|Điều|Cụm từ|Từ|Các|Mục)\b',
            re.IGNORECASE
        )
    def extract_doc_id(self, filename: str) -> str:
        """
        Trích xuất doc_id từ tên file. 
        Mặc định tạo slug từ tên file để làm prefix cho các ID.
        Nếu muốn lấy chính xác "95_2015_QH13" thì có thể viết logic Regex chuyên sâu hơn 
        tùy vào cấu trúc tên file chuẩn của bạn. Ở đây đang chuyển đổi tên file thành chuỗi slug.
        """
        base = os.path.splitext(filename)[0]
        # Loại bỏ các ký tự không phải chữ/số, thay bằng "_"
        slug = re.sub(r'[^a-zA-Z0-9]', '_', base)
        # Gộp các "_" liên tiếp
        slug = re.sub(r'_+', '_', slug).strip('_')
        return slug

    def parse_docx(self, file_path: str) -> List[str]:
        """
        Đọc file docx và trả về danh sách các đoạn văn bản (paragraphs).
        """
        try:
            doc = docx.Document(file_path)
            # Chỉ lấy các paragraph có nội dung
            return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {file_path}: {e}")
            return []

    def _is_trailing_note(self, line: str) -> bool:
        """
        Nhận diện chú thích/cuối văn bản hợp nhất để không nhập vào Điều cuối.
        """
        return bool(self.trailing_note_pattern.match(line))

    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Phân tách nội dung file thành các chunk nhỏ (lời dẫn, khoản, điểm).
        """
        filename = os.path.basename(file_path)
        doc_id = self.extract_doc_id(filename)
        doc_name = os.path.splitext(filename)[0] # Dùng làm tên luật trong prefix
        
        lines = self.parse_docx(file_path)
        
        chunks = []
        
        # State variables
        current_dieu_id = None
        current_dieu_title = None
        current_dieu_num = None
        
        current_khoan_num = None
        current_diem_char = None
        
        current_chunk_lines = []
        current_chunk_type = None
        current_chunk_id = None
        dieu_occurrences: Dict[str, int] = {}
        chunk_id_occurrences: Dict[str, int] = {}
        pending_title_preamble = None
        
        def build_prefix(chunk_type, dieu_num, khoan_num=None, diem_char=None):
            if chunk_type == "dieu_preamble":
                return f"[Điều {dieu_num}, {doc_name}]"
            if chunk_type == "khoan":
                return f"[Khoản {khoan_num}, Điều {dieu_num}, {doc_name}]"
            if chunk_type == "diem":
                if khoan_num:
                    return f"[Điểm {diem_char}, Khoản {khoan_num}, Điều {dieu_num}, {doc_name}]"
                return f"[Điểm {diem_char}, Điều {dieu_num}, {doc_name}]"
            return f"[Điều {dieu_num}, {doc_name}]"

        def append_chunk(chunk_data):
            if not chunk_data["lines"]:
                return

            prefix = build_prefix(
                chunk_data["type"],
                chunk_data["dieu_num"],
                chunk_data.get("khoan_num"),
                chunk_data.get("diem_char"),
            )
            content_text = "\n".join(chunk_data["lines"])
            content_with_prefix = f"{prefix} {content_text}"
            base_chunk_id = chunk_data["id"]
            chunk_id_occurrences[base_chunk_id] = chunk_id_occurrences.get(base_chunk_id, 0) + 1
            unique_chunk_id = base_chunk_id
            if chunk_id_occurrences[base_chunk_id] > 1:
                unique_chunk_id = f"{base_chunk_id}_C{chunk_id_occurrences[base_chunk_id]}"

            chunks.append({
                "id": unique_chunk_id,
                "dieu_id": chunk_data["dieu_id"],
                "type": chunk_data["type"],
                "content": content_with_prefix,
                "order": len(chunks),
                "metadata": {
                    "doc_id": doc_id,
                    "dieu_title": chunk_data["dieu_title"],
                    "source": filename
                }
            })

        def flush_pending_title_preamble():
            nonlocal pending_title_preamble
            if pending_title_preamble:
                append_chunk(pending_title_preamble)
                pending_title_preamble = None

        def save_chunk():
            nonlocal pending_title_preamble
            if not (current_chunk_lines and current_dieu_id and current_chunk_type):
                return

            chunk_data = {
                "id": current_chunk_id,
                "dieu_id": current_dieu_id,
                "type": current_chunk_type,
                "lines": list(current_chunk_lines),
                "dieu_title": current_dieu_title,
                "dieu_num": current_dieu_num,
                "khoan_num": current_khoan_num,
                "diem_char": current_diem_char,
            }

            if current_chunk_type == "dieu_preamble" and len(current_chunk_lines) == 1:
                flush_pending_title_preamble()
                pending_title_preamble = chunk_data
                return

            if pending_title_preamble and pending_title_preamble["dieu_id"] == current_dieu_id:
                chunk_data["lines"] = pending_title_preamble["lines"] + chunk_data["lines"]
                pending_title_preamble = None
            elif pending_title_preamble:
                flush_pending_title_preamble()

            append_chunk(chunk_data)

        stopped_at_tail = False
        for line in lines:
            if current_dieu_id and self._is_trailing_note(line):
                save_chunk()
                flush_pending_title_preamble()
                stopped_at_tail = True
                break

            dieu_match = self.dieu_pattern.match(line)
            khoan_match = self.khoan_pattern.match(line)
            diem_match = self.diem_pattern.match(line)
            
            if dieu_match:
                # Lưu lại chunk trước đó
                save_chunk()
                flush_pending_title_preamble()
                
                current_dieu_num = dieu_match.group(1)
                dieu_title = dieu_match.group(2)
                
                current_khoan_num = None # Reset Khoản khi sang Điều mới
                current_diem_char = None # Reset Điểm
                
                # Tạo dieu_id giống mẫu "95_2015_QH13_D114"
                base_dieu_id = f"{doc_id}_D{current_dieu_num}"
                dieu_occurrences[base_dieu_id] = dieu_occurrences.get(base_dieu_id, 0) + 1
                current_dieu_id = base_dieu_id
                if dieu_occurrences[base_dieu_id] > 1:
                    current_dieu_id = f"{base_dieu_id}_O{dieu_occurrences[base_dieu_id]}"
                current_dieu_title = f"Điều {current_dieu_num}. {dieu_title}".strip()
                
                # Bắt đầu chunk mới: Phần lời dẫn của Điều (preamble)
                current_chunk_type = "dieu_preamble"
                current_chunk_id = f"{current_dieu_id}_preamble"
                current_chunk_lines = [line]
                
            elif khoan_match and current_dieu_id:
                # Chuyển sang Khoản mới -> lưu chunk trước đó
                save_chunk()
                current_khoan_num = khoan_match.group(1)
                current_diem_char = None # Reset Điểm khi sang Khoản mới
                
                current_chunk_type = "khoan"
                current_chunk_id = f"{current_dieu_id}_K{current_khoan_num}"
                current_chunk_lines = [line]
                
            elif diem_match and current_dieu_id:
                # Chuyển sang Điểm mới -> lưu chunk trước đó
                save_chunk()
                current_diem_char = diem_match.group(1)
                
                current_chunk_type = "diem"
                # ID của điểm: D114_K1_Pa hoặc D114_Pa (nếu không có Khoản)
                if current_khoan_num:
                    current_chunk_id = f"{current_dieu_id}_K{current_khoan_num}_P{current_diem_char}"
                else:
                    current_chunk_id = f"{current_dieu_id}_P{current_diem_char}"
                
                current_chunk_lines = [line]
                
            else:
                # Text bình thường không chứa pattern -> nối vào chunk đang mở
                if current_dieu_id:
                    current_chunk_lines.append(line)
                    
        # Lưu chunk cuối cùng
        if not stopped_at_tail:
            save_chunk()
            flush_pending_title_preamble()
        
        return chunks

    def build_parent_chunks(self, child_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Gom các child chunks theo dieu_id để tạo parent chunk cho từng Điều.
        """
        by_dieu: Dict[str, List[Dict[str, Any]]] = {}
        for ch in child_chunks:
            by_dieu.setdefault(ch["dieu_id"], []).append(ch)

        parent_chunks = []
        for dieu_id, items in by_dieu.items():
            # Giữ đúng thứ tự xuất hiện trong văn bản khi ghép parent context.
            items_sorted = sorted(items, key=lambda x: x.get("order", 0))
            content_text = "\n".join([i["content"] for i in items_sorted])
            meta = items_sorted[0]["metadata"] if items_sorted else {}
            parent_chunks.append({
                "id": f"{dieu_id}_PARENT",
                "dieu_id": dieu_id,
                "type": "parent",
                "content": content_text,
                "metadata": meta
            })

        return parent_chunks

    def chunk_document_parent_child(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Trả về cấu trúc parent-child:
        - children: các chunk nhỏ (preamble/khoan/diem)
        - parents: chunk gộp theo Điều
        """
        children = self.chunk_document(file_path)
        for ch in children:
            ch["parent_id"] = f"{ch['dieu_id']}_PARENT"
        parents = self.build_parent_chunks(children)
        return {"parents": parents, "children": children}

    def expand_with_parents(
        self,
        child_hits: List[Dict[str, Any]],
        parent_index: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Mở rộng kết quả retrieval từ child chunks sang parent chunk tương ứng.
        child_hits: danh sách kết quả truy xuất từ child chunks (có parent_id).
        parent_index: dict {parent_id: parent_chunk}.
        """
        expanded = []
        seen_parent = set()
        for hit in child_hits:
            parent_id = hit.get("parent_id")
            if not parent_id:
                continue
            if parent_id in seen_parent:
                continue
            parent = parent_index.get(parent_id)
            if parent:
                expanded.append(parent)
                seen_parent.add(parent_id)
        return expanded

    def process_directory(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        Xử lý tất cả file docx trong thư mục.
        """
        all_chunks = []
        
        if not os.path.exists(data_dir):
            logger.error(f"Thư mục {data_dir} không tồn tại!")
            return []
            
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".docx"):
                file_path = os.path.join(data_dir, filename)
                logger.info(f"Processing: {filename}")
                chunks = self.chunk_document(file_path)
                all_chunks.extend(chunks)
                
        return all_chunks

    def process_directory_parent_child(self, data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Xử lý tất cả file docx trong thư mục, trả về parents và children.
        """
        all_children = []
        all_parents = []

        if not os.path.exists(data_dir):
            logger.error(f"Thư mục {data_dir} không tồn tại!")
            return {"parents": [], "children": []}

        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".docx"):
                file_path = os.path.join(data_dir, filename)
                logger.info(f"Processing: {filename}")
                result = self.chunk_document_parent_child(file_path)
                all_children.extend(result["children"])
                all_parents.extend(result["parents"])

        return {"parents": all_parents, "children": all_children}

if __name__ == "__main__":
    # Ví dụ sử dụng
    chunker = VBPLChunker()
    # Đường dẫn thư mục data/keep nằm cạnh file chunking.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(base_dir, "data", "keep")
    
    chunks = chunker.process_directory(data_directory)
    logger.info(f"Tổng số chunks (Lời dẫn Điều, Khoản, Điểm): {len(chunks)}")
    
    if chunks:
        logger.info("--- Ví dụ 1 Chunk ---")
        example_chunk = chunks[2] if len(chunks) > 1 else chunks[0]
        logger.info(f"id: {example_chunk['id']}")
        logger.info(f"dieu_id: {example_chunk['dieu_id']}")
        logger.info(f"type: {example_chunk['type']}")
        logger.info(f"content: {example_chunk['content']}")
