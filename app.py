"""
app.py
======
Giao diện chat (Chainlit) cho Chatbot pháp luật — cho phép người dùng chọn 1
trong 3 kịch bản (Chat Profile ở góc trên bên trái, giống chọn phiên bản model
của ChatGPT), có lịch sử chat lưu lại ở thanh bên trái (yêu cầu đăng nhập, xem
ghi chú AUTH bên dưới). Ngay tại khung nhập chat có 2 chỗ đổi model, không cần
sửa code/biến môi trường:
  - Nút chọn model (Command, cạnh nút đính kèm file) — đổi model LLM ngay tức
    thời (Qwen2.5 7B qua Ollama hoặc gpt-4o-mini/gpt-5-nano/gpt-4.1-nano qua
    api.shopaikey).
  - Nút Cài đặt (icon bánh răng) — đổi model embedding (kéo theo Qdrant index
    tương ứng), tốn vài giây vì phải load lại pipeline.

Cách chạy (qua Docker, đã có sẵn đủ dependency — xem docker-compose.yml):
    docker compose up -d neo4j                            # chờ healthy
    docker compose --profile ingest up kg-ingest          # 1 LẦN DUY NHẤT, nạp Neo4j
    (copy .env.example -> .env, điền OPENAI_API_KEY — xem bên dưới)
    docker compose up -d app
    Mở trình duyệt: http://localhost:8000

    Ollama (Qwen2.5 7B local) là TÙY CHỌN, không bắt buộc — máy không host được
    Ollama (thiếu RAM/CPU, hoặc chỉ cần chạy nhanh gọn) thì bỏ qua hẳn bước này,
    dùng model qua api.shopaikey (mặc định) là đủ. Chỉ khi thật sự muốn Qwen2.5
    7B local mới cần thêm: `docker compose --profile ollama up -d ollama`.

    Lưu ý: data/.qdrant_base/ và data/.qdrant_gte_base/ (Qdrant embedded, ~100-
    150MB mỗi cái) KHÔNG nằm trong git (xem .gitignore) — phải copy kèm theo
    khi chuyển code sang máy khác. CHAINLIT_AUTH_SECRET đã hardcode sẵn trong
    docker-compose.yml, không cần tự tạo.

Đăng nhập demo (đổi qua biến môi trường CHAINLIT_DEMO_USER/CHAINLIT_DEMO_PASSWORD
nếu muốn) — Chainlit BẮT BUỘC phải có đăng nhập mới hiển thị được lịch sử chat
ở thanh bên trái, dù chỉ chạy local 1 người dùng:
    Username: admin   (mặc định)
    Password: admin   (mặc định)

Model LLM mặc định (gpt-4o-mini qua api.shopaikey) cần OPENAI_API_KEY: copy
.env.example thành .env, điền key, rồi `docker compose up -d app` lại. Muốn
dùng Qwen2.5 7B qua Ollama (không cần key, nhưng cần bật service ollama ở
trên): chọn trong nút chọn model cạnh ô nhập chat.
"""

import os
import sys
import logging
import threading
from pathlib import Path

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflow.pipeline import ChatbotWorkflow  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("chainlit_app")

DEFAULT_MODE = "critic"

MODE_INFO = {
    "critic": {
        "display_name": "Critic Agent (đề xuất khóa luận)",
        "description": (
            "**Critic Agent** — sinh câu trả lời nháp, tự kiểm tra chéo qua Knowledge "
            "Graph (Neo4j) để phát hiện phần còn thiếu (tham chiếu chéo Điều khác, cấu "
            "trúc chưa đủ), rồi bổ sung lại nếu cần. Chính xác và đầy đủ nhất, nhưng "
            "chậm hơn 2 kịch bản còn lại."
        ),
    },
    "article_expand": {
        "display_name": "Mở rộng toàn Điều",
        "description": (
            "**Mở rộng toàn Điều** — quét rộng hơn để lấy đủ các Điều liên quan, rồi mở "
            "rộng lấy TOÀN VĂN từng Điều làm ngữ cảnh (không dùng Knowledge Graph, không "
            "có bước tự kiểm tra)."
        ),
    },
    "naive": {
        "display_name": "RAG truyền thống (baseline)",
        "description": (
            "**RAG truyền thống** — chỉ lấy đúng top-k đoạn (chunk) liên quan nhất làm "
            "ngữ cảnh, không mở rộng, không dùng Knowledge Graph. Nhanh nhất nhưng dễ "
            "thiếu ngữ cảnh với câu hỏi phức tạp."
        ),
    },
}


# ---------------------------------------------------------------------------
# Embedding model — mỗi model ứng với 1 Qdrant index đã ingest riêng (2 không
# gian embedding khác nhau TUYỆT ĐỐI không được lẫn vào nhau). Model fine-tune
# + data/.qdrant cũ không còn dùng nữa — chỉ còn lại 2 lựa chọn dưới đây.
# ---------------------------------------------------------------------------

EMBEDDING_INFO = {
    "vietnamese_embedding_v2": {
        "display_name": "Vietnamese Embedding v2 (base)",
        "model_name": "AITeamVN/Vietnamese_Embedding_v2",
        "qdrant_path": str(PROJECT_ROOT / "data" / ".qdrant_base"),
    },
    "gte": {
        "display_name": "GTE Multilingual Base",
        "model_name": "Alibaba-NLP/gte-multilingual-base",
        "qdrant_path": str(PROJECT_ROOT / "data" / ".qdrant_gte_base"),
    },
}
DEFAULT_EMBEDDING_KEY = "vietnamese_embedding_v2"


# ---------------------------------------------------------------------------
# LLM — Qwen2.5 7B qua Ollama (local, không cần API key, nhưng CẦN container
# ollama — xem docker-compose.yml, profile "ollama", KHÔNG bắt buộc phải có)
# hoặc 1 trong các model qua proxy api.shopaikey (cần OPENAI_API_KEY, không
# cần Ollama — mặc định dùng nhóm này để chạy được ngay cả trên máy không host
# được Ollama).
# ---------------------------------------------------------------------------

OPENAI_BASE_URL = "https://api.shopaikey.com/v1"

LLM_INFO = {
    "qwen2.5:7b": {"display_name": "Qwen2.5 7B (Ollama, local)", "provider": "ollama"},
    "gpt-4o-mini": {"display_name": "GPT-4o-mini (api.shopaikey)", "provider": "openai_proxy"},
    "gpt-5-nano": {"display_name": "GPT-5-nano (api.shopaikey)", "provider": "openai_proxy"},
    "gpt-4.1-nano": {"display_name": "GPT-4.1-nano (api.shopaikey)", "provider": "openai_proxy"},
}
# Mặc định KHÔNG chọn Qwen2.5 (qua Ollama) vì service ollama là tùy chọn (xem
# docker-compose.yml) — máy không chạy được Ollama vẫn phải vào chat được ngay
# lần đầu mà không cần đổi model trước. Đổi lại "qwen2.5:7b" nếu máy bạn CÓ
# chạy service ollama và muốn nó làm mặc định.
DEFAULT_LLM_KEY = "gpt-4o-mini"

# Command hiện ngay trong khung nhập chat (cạnh nút đính kèm file/Cài đặt) —
# persistent=True để lựa chọn giữ nguyên qua nhiều tin nhắn (không phải chỉ áp
# dụng cho 1 tin nhắn rồi mất), selected đánh dấu model mặc định khi mới vào chat.
LLM_COMMANDS = [
    {
        "id": key,
        "description": info["display_name"],
        "icon": "server" if info["provider"] == "ollama" else "sparkles",
        "persistent": True,
        "selected": key == DEFAULT_LLM_KEY,
    }
    for key, info in LLM_INFO.items()
]


def _build_llm(llm_key: str) -> ChatOpenAI:
    info = LLM_INFO.get(llm_key, LLM_INFO[DEFAULT_LLM_KEY])
    if info["provider"] == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        logger.info(f"LLM: {llm_key} (Ollama) @ {base_url}")
        return ChatOpenAI(model=llm_key, base_url=base_url, api_key="ollama", temperature=0.2)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Cần biến môi trường OPENAI_API_KEY để dùng model qua api.shopaikey "
            f"({llm_key})."
        )
    logger.info(f"LLM: {llm_key} (api.shopaikey) @ {OPENAI_BASE_URL}")
    return ChatOpenAI(model=llm_key, base_url=OPENAI_BASE_URL, api_key=api_key, temperature=0.2)


# ---------------------------------------------------------------------------
# Data layer (lịch sử chat) — lưu SQLite cục bộ tại data/chainlit_history.db
# ---------------------------------------------------------------------------

# SQLAlchemyDataLayer KHÔNG tự tạo schema — bản gốc trong docs Chainlit dùng kiểu
# Postgres (UUID/JSONB/TEXT[]) không tương thích SQLite, nên phải tự tạo bảng
# tương đương bằng kiểu SQLite (TEXT cho UUID/JSONB/mảng). Chạy 1 lần, idempotent
# (IF NOT EXISTS), mỗi lần app khởi động — không cần bước cài đặt thủ công riêng.
_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    "autoCollapse" BOOLEAN,
    "icon" TEXT,
    "modes" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "path" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    "autoPlay" BOOLEAN,
    "playerConfig" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
"""


# Cột thêm SAU khi bảng đã tồn tại (Chainlit ra bản mới, StepDict/ElementDict có
# thêm field) — CREATE TABLE IF NOT EXISTS ở trên KHÔNG tự thêm cột vào bảng đã
# có sẵn, nên cần ALTER TABLE riêng, chỉ thêm cột nào còn thiếu (an toàn để chạy
# lại nhiều lần, không đụng tới dữ liệu lịch sử chat đã lưu).
_EXTRA_COLUMNS = {
    "steps": {"autoCollapse": "BOOLEAN", "icon": "TEXT"},
    "elements": {"path": "TEXT", "autoPlay": "BOOLEAN", "playerConfig": "TEXT"},
}


def _add_missing_columns(conn, table: str, columns: dict) -> None:
    existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    for col, coltype in columns.items():
        if col not in existing:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {coltype}')


def _ensure_sqlite_schema(db_path: Path) -> None:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SQLITE_SCHEMA)
        for table, columns in _EXTRA_COLUMNS.items():
            _add_missing_columns(conn, table, columns)
        conn.commit()
    finally:
        conn.close()


@cl.data_layer
def get_data_layer():
    db_path = PROJECT_ROOT / "data" / "chainlit_history.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_sqlite_schema(db_path)
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{db_path}")


# ---------------------------------------------------------------------------
# Đăng nhập — chỉ 1 tài khoản demo, BẮT BUỘC phải có để Chainlit hiển thị
# được lịch sử chat ở thanh bên trái (giới hạn của chính Chainlit, không phải
# lựa chọn thiết kế ở đây).
# ---------------------------------------------------------------------------

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    demo_user = os.getenv("CHAINLIT_DEMO_USER", "admin")
    demo_pass = os.getenv("CHAINLIT_DEMO_PASSWORD", "admin")
    if username == demo_user and password == demo_pass:
        return cl.User(identifier=username)
    return None


# ---------------------------------------------------------------------------
# Chat Profile — dropdown chọn kịch bản ở góc trên bên trái (giống chọn model
# GPT trên ChatGPT).
# ---------------------------------------------------------------------------

@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name=mode,
            display_name=info["display_name"],
            markdown_description=info["description"],
        )
        for mode, info in MODE_INFO.items()
    ]


# ---------------------------------------------------------------------------
# Pipeline — 1 pipeline (nặng: embedding model + Qdrant + Neo4j) cho MỖI model
# embedding, cache lại và dùng chung cho mọi phiên chat chọn cùng embedding đó.
# LLM thì rẻ (chỉ là 1 client HTTP) nên KHÔNG cache theo pipeline — pipeline.llm
# được gán lại đúng lúc chạy (xem run_query), có khóa riêng để 2 phiên chat
# cùng dùng chung 1 pipeline (cùng embedding, khác LLM) không ghi đè lẫn nhau.
# ---------------------------------------------------------------------------

_pipelines: dict = {}
_pipelines_lock = threading.Lock()

_llms: dict = {}
_llms_lock = threading.Lock()

_run_locks: dict = {}
_run_locks_master_lock = threading.Lock()


def _get_llm(llm_key: str) -> ChatOpenAI:
    llm = _llms.get(llm_key)
    if llm is not None:
        return llm
    with _llms_lock:
        llm = _llms.get(llm_key)
        if llm is not None:
            return llm
        llm = _build_llm(llm_key)
        _llms[llm_key] = llm
        return llm


def get_pipeline(embedding_key: str) -> ChatbotWorkflow:
    """Trả về (hoặc khởi tạo lần đầu) pipeline ứng với 1 model embedding.

    Khởi tạo với LLM mặc định (Ollama, không cần API key) chỉ để bootstrap —
    LLM thật sự dùng cho từng câu hỏi được gán lại trong run_query() ngay
    trước khi gọi pipeline.run(), bên trong khóa riêng của embedding_key đó.
    """
    pipeline = _pipelines.get(embedding_key)
    if pipeline is not None:
        return pipeline
    # Khóa BẮT BUỘC: get_pipeline() chạy trong thread pool riêng (cl.make_async),
    # nên 2 phiên chat mở gần như đồng thời (2 tab, hoặc trình duyệt tự kết nối
    # lại) có thể cùng thấy pipeline chưa tồn tại và cùng cố mở QdrantClient trên
    # CÙNG 1 đường dẫn — Qdrant chế độ local/embedded không cho phép 2 instance
    # mở cùng lúc dù trong cùng 1 process, gây lỗi "already accessed by another
    # instance". Khóa đảm bảo chỉ 1 luồng thực sự khởi tạo, các luồng khác chờ
    # rồi dùng lại kết quả đã có.
    with _pipelines_lock:
        pipeline = _pipelines.get(embedding_key)
        if pipeline is not None:
            return pipeline
        info = EMBEDDING_INFO[embedding_key]
        pipeline = ChatbotWorkflow(
            llm=_get_llm(DEFAULT_LLM_KEY),
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_path=info["qdrant_path"],
            embedding_model_name=info["model_name"],
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_pass=os.getenv("NEO4J_PASSWORD", "legal_kg_2024"),
            top_k=int(os.getenv("TOP_K", "5")),
        )
        _pipelines[embedding_key] = pipeline
    return pipeline


def _get_run_lock(embedding_key: str) -> threading.Lock:
    lock = _run_locks.get(embedding_key)
    if lock is not None:
        return lock
    with _run_locks_master_lock:
        lock = _run_locks.get(embedding_key)
        if lock is None:
            lock = threading.Lock()
            _run_locks[embedding_key] = lock
        return lock


def run_query(embedding_key: str, llm_key: str, query: str, mode: str):
    """Chạy 1 câu hỏi với đúng (embedding, LLM) người dùng đang chọn.

    pipeline được cache theo embedding_key (có thể dùng chung giữa nhiều phiên
    chat) — gán pipeline.llm ngay trước khi chạy, bên trong khóa riêng của
    embedding_key đó, để 2 phiên chat chọn CÙNG embedding nhưng KHÁC LLM không
    bị lẫn LLM của nhau khi chạy đồng thời.
    """
    pipeline = get_pipeline(embedding_key)
    llm = _get_llm(llm_key)
    with _get_run_lock(embedding_key):
        pipeline.llm = llm
        return pipeline.run(query, mode=mode)


# ---------------------------------------------------------------------------
# Cài đặt (Settings) — icon bánh răng kế nút đính kèm file ở khung chat, cho
# đổi model embedding (kéo theo Qdrant index tương ứng). Model LLM đổi qua
# Command riêng (xem LLM_COMMANDS + on_message) vì nhẹ/tức thời hơn, không cần
# vào modal Settings.
# ---------------------------------------------------------------------------

async def _apply_settings(settings: dict, announce: bool = False):
    embedding_key = settings.get("embedding_model") or DEFAULT_EMBEDDING_KEY
    if embedding_key not in EMBEDDING_INFO:
        embedding_key = DEFAULT_EMBEDDING_KEY

    prev_embedding_key = cl.user_session.get("embedding_key")
    cl.user_session.set("embedding_key", embedding_key)

    # Chỉ hiện Step "đang tải" khi thực sự cần load 1 embedding model MỚI.
    if embedding_key not in _pipelines:
        async with cl.Step(
            name=f"Tải embedding model: {EMBEDDING_INFO[embedding_key]['display_name']}", type="run"
        ):
            await cl.make_async(get_pipeline)(embedding_key)
    else:
        await cl.make_async(get_pipeline)(embedding_key)

    if announce and (prev_embedding_key is not None):
        await cl.Message(
            content=f"Đã đổi embedding model sang **{EMBEDDING_INFO[embedding_key]['display_name']}**."
        ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    await _apply_settings(settings, announce=True)


# ---------------------------------------------------------------------------
# Chainlit hooks
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    mode = cl.user_session.get("chat_profile") or DEFAULT_MODE
    if mode not in MODE_INFO:
        mode = DEFAULT_MODE
    cl.user_session.set("mode", mode)
    cl.user_session.set("llm_key", DEFAULT_LLM_KEY)

    settings = await cl.ChatSettings(
        [
            Select(
                id="embedding_model",
                label="Embedding model (Qdrant index)",
                # items = {label hiển thị: value trả về} — chainlit.Select KHÔNG dùng
                # thứ tự (key=id nội bộ, value=label) như trực giác, mà ngược lại;
                # initial (khi dùng items) bị ghi đè bởi initial_value, không phải initial.
                items={info["display_name"]: key for key, info in EMBEDDING_INFO.items()},
                initial_value=DEFAULT_EMBEDDING_KEY,
            ),
        ]
    ).send()

    # Command chọn LLM — hiện ngay trong khung nhập chat (cạnh nút Cài đặt),
    # không cần mở modal Settings. message.command trong on_message cho biết
    # người dùng vừa chọn command nào (xem LLM_COMMANDS).
    await cl.context.emitter.set_commands(LLM_COMMANDS)

    # Load pipeline trong thread riêng — tránh chặn event loop khi model
    # embedding/Neo4j đang khởi tạo lần đầu (có thể mất vài giây - vài chục giây).
    async with cl.Step(name="Khởi tạo pipeline", type="run"):
        await _apply_settings(settings, announce=False)

    embedding_key = cl.user_session.get("embedding_key", DEFAULT_EMBEDDING_KEY)
    llm_key = cl.user_session.get("llm_key", DEFAULT_LLM_KEY)
    await cl.Message(
        content=(
            f"Xin chào! Đang dùng chế độ **{MODE_INFO[mode]['display_name']}**, "
            f"embedding **{EMBEDDING_INFO[embedding_key]['display_name']}** (đổi ở nút Cài đặt), "
            f"LLM **{LLM_INFO[llm_key]['display_name']}** (đổi ở nút chọn model cạnh ô nhập chat).\n\n"
            "Hãy đặt câu hỏi pháp luật của bạn (an ninh mạng, giao dịch điện tử, "
            "bảo vệ dữ liệu cá nhân, viễn thông, ...)."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # message.command: id của LLM_COMMANDS đang được chọn (persistent=True nên
    # tự động gửi kèm mỗi tin nhắn cho tới khi người dùng đổi command khác).
    if message.command and message.command in LLM_INFO:
        cl.user_session.set("llm_key", message.command)

    mode = cl.user_session.get("mode", DEFAULT_MODE)
    embedding_key = cl.user_session.get("embedding_key", DEFAULT_EMBEDDING_KEY)
    llm_key = cl.user_session.get("llm_key", DEFAULT_LLM_KEY)

    async with cl.Step(name=f"Đang xử lý ({MODE_INFO[mode]['display_name']})", type="run") as step:
        result = await cl.make_async(run_query)(embedding_key, llm_key, message.content, mode)
        step.output = (
            f"Số chunk retrieved: {len(result.get('retrieved_chunks', []))} | "
            f"Số Điều: {len(result.get('retrieved_dieu_ids', []))}"
        )

    final_response = result.get("final_response", "(không có câu trả lời)")

    elements = []
    if mode == "critic" and result.get("critic_report"):
        report = result["critic_report"]
        suggestions = report.get("suggestions", [])
        if suggestions:
            detail = "\n".join(f"- **[{s['action']}]** {s['reason']}" for s in suggestions)
            elements.append(
                cl.Text(
                    name="Critic Report",
                    content=f"is_complete={report.get('is_complete')}\n\n{detail}",
                    display="side",
                )
            )

    await cl.Message(content=final_response, elements=elements).send()


if __name__ == "__main__":
    print(__doc__)
