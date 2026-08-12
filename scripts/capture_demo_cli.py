"""Capture one real compare-all CLI run without changing the chatbot pipeline."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "demo_logs" / "demo_compare_all_cli.log"
QUESTION = (
    "Doanh nghiệp thiết lập mạng xã hội nhưng không có giấy phép thì bị xử phạt "
    "như thế nào theo Nghị định 15/2020/NĐ-CP?"
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "QDRANT_PATH": "data/.qdrant_base",
            "EMBEDDING_MODEL": "AITeamVN/Vietnamese_Embedding_v2",
            "LLM_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434/v1",
            "OLLAMA_MODEL": "qwen2.5:7b",
        }
    )
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "scripts/run_chatbot.py",
        "--compare-all",
        "--top-k",
        "5",
        "--query",
        QUESTION,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    header = (
        "# REAL CLI RUN — thesis demo evidence\n"
        f"# cwd: {ROOT}\n"
        f"# model: qwen2.5:7b (Ollama native)\n"
        "# retriever: Base / AITeamVN/Vietnamese_Embedding_v2\n"
        f"# question: {QUESTION}\n"
        f"# exit_code: {completed.returncode}\n\n"
    )
    OUTPUT.write_text(header + completed.stdout, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes), exit={completed.returncode}")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
