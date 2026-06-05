import os
import zipfile
from typing import Optional

from sentence_transformers import SentenceTransformer


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
) -> SentenceTransformer:
    model_path = resolve_model_path(model)
    sentence_model = SentenceTransformer(model_path, device=device)
    if max_seq_length is not None:
        sentence_model.max_seq_length = max_seq_length
    return sentence_model
