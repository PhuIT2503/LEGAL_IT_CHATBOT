import os
import zipfile
from typing import Optional

# Chỉ cần PyTorch cho embedding — ép transformers bỏ qua hẳn việc detect/import
# TensorFlow (PHẢI set TRƯỚC khi import sentence_transformers/transformers).
# Một số môi trường (vd Colab) có sẵn TF với bản protobuf khác/không tương thích
# với bản transformers đang dùng, khiến import transformers crash ngay cả khi
# code chỉ dùng PyTorch — vì transformers vẫn thử import nhánh TF nếu thấy TF
# có sẵn trong môi trường.
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer  # noqa: E402


DEFAULT_FINETUNED_ZIP = os.path.join(
    "references",
    "ai_vietnamese_embedding_v2_finetuned_final (1).zip",
)
DEFAULT_FINETUNED_DIR = os.path.join(
    "references",
    "ai_vietnamese_embedding_v2_finetuned_final",
)


def _has_sentence_transformer_files(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "modules.json"))
        and os.path.exists(os.path.join(path, "config_sentence_transformers.json"))
    )


def resolve_model_path(model: str = DEFAULT_FINETUNED_DIR) -> str:
    if _has_sentence_transformer_files(model):
        return model

    if os.path.isfile(model) and model.endswith(".zip"):
        target_dir = os.path.splitext(model)[0]
        return extract_finetuned_model(model, target_dir)

    if model == DEFAULT_FINETUNED_DIR and os.path.isfile(DEFAULT_FINETUNED_ZIP):
        return extract_finetuned_model(DEFAULT_FINETUNED_ZIP, DEFAULT_FINETUNED_DIR)

    return model


def extract_finetuned_model(zip_path: str, target_dir: str) -> str:
    if _has_sentence_transformer_files(target_dir):
        return target_dir

    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        modules_member = next((name for name in members if name.endswith("/modules.json")), None)
        if not modules_member:
            raise RuntimeError(f"Không tìm thấy modules.json trong model zip: {zip_path}")

        model_root = modules_member[: -len("modules.json")]
        for member in members:
            if not member.startswith(model_root):
                continue
            relative = member[len(model_root) :]
            if not relative:
                continue
            destination = os.path.join(target_dir, relative)
            if member.endswith("/"):
                os.makedirs(destination, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as output:
                output.write(source.read())

    if not _has_sentence_transformer_files(target_dir):
        raise RuntimeError(f"Extract model không đầy đủ vào: {target_dir}")
    return target_dir


def load_embedding_model(
    model: str = DEFAULT_FINETUNED_DIR,
    device: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    trust_remote_code: bool = True,
) -> SentenceTransformer:
    # trust_remote_code=True mặc định: một số model HF (vd Alibaba-NLP/gte-multilingual-base)
    # dùng kiến trúc/code tùy biến, bắt buộc cờ này mới load được — không ảnh hưởng
    # tới model không cần custom code (sentence-transformers bỏ qua nếu không cần).
    model_path = resolve_model_path(model)
    sentence_model = SentenceTransformer(model_path, device=device, trust_remote_code=trust_remote_code)
    if max_seq_length is not None:
        sentence_model.max_seq_length = max_seq_length
    return sentence_model
