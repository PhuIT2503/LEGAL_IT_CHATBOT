import os
import re
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
    
    Tất cả các chunk thuộc cùng một Điều sẽ chia sẻ chung một trường `dieu_id`.
    Khi truy xuất (retrieval) trúng một chunk bất kỳ (VD: Điểm), ta dùng `dieu_id` 
    để gom lại toàn bộ các chunk cùng Điều nhằm khôi phục trọn vẹn văn cảnh của Điều đó.
    """
    def __init__(self):
        # Regular expressions để bắt các thành phần của VBPL
        # Bắt "Điều 114.", "Điều 1.", "Điều 1a."
        self.dieu_pattern = re.compile(r'^Điều\s+(\d+[a-zA-Z]*)\s*[\.\:\-]?\s*(.*)', re.IGNORECASE)
        # Bắt "1.", "2."
        self.khoan_pattern = re.compile(r'^(\d+)\.\s+(.*)')
        # Bắt "a)", "b)", "đ)"
        self.diem_pattern = re.compile(r'^([a-zđ])\)\s+(.*)', re.IGNORECASE)

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
        
        def save_chunk():
            if current_chunk_lines and current_dieu_id and current_chunk_type:
                # Tạo prefix để BM25 bắt keyword dễ hơn
                prefix = ""
                if current_chunk_type == "dieu_preamble":
                    prefix = f"[Điều {current_dieu_num}, {doc_name}]"
                elif current_chunk_type == "khoan":
                    prefix = f"[Khoản {current_khoan_num}, Điều {current_dieu_num}, {doc_name}]"
                elif current_chunk_type == "diem":
                    if current_khoan_num:
                        prefix = f"[Điểm {current_diem_char}, Khoản {current_khoan_num}, Điều {current_dieu_num}, {doc_name}]"
                    else:
                        prefix = f"[Điểm {current_diem_char}, Điều {current_dieu_num}, {doc_name}]"
                        
                content_text = "\n".join(current_chunk_lines)
                content_with_prefix = f"{prefix} {content_text}"
                chunk_id_occurrences[current_chunk_id] = chunk_id_occurrences.get(current_chunk_id, 0) + 1
                unique_chunk_id = current_chunk_id
                if chunk_id_occurrences[current_chunk_id] > 1:
                    unique_chunk_id = f"{current_chunk_id}_C{chunk_id_occurrences[current_chunk_id]}"
                
                chunks.append({
                    "id": unique_chunk_id,
                    "dieu_id": current_dieu_id,
                    "type": current_chunk_type,
                    "content": content_with_prefix,
                    "order": len(chunks),
                    "metadata": {
                        "doc_id": doc_id,
                        "dieu_title": current_dieu_title,
                        "source": filename
                    }
                })

        for line in lines:
            dieu_match = self.dieu_pattern.match(line)
            khoan_match = self.khoan_pattern.match(line)
            diem_match = self.diem_pattern.match(line)
            
            if dieu_match:
                # Lưu lại chunk trước đó
                save_chunk()
                
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
        save_chunk()
        
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
            
        for filename in os.listdir(data_dir):
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

        for filename in os.listdir(data_dir):
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
