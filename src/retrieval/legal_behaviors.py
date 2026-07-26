"""Behavior Extraction và rule-based legal behavior matching cho Phase 2.

Phase này không gọi LLM. Query gốc được chuẩn hóa thành một behavior card gồm
hành vi, đối tượng, mục đích và điều kiện. Card vừa làm rõ input cho Cross
Encoder, vừa tạo một tín hiệu độc lập để không nhầm "trùng bối cảnh/từ khóa"
với "điều chỉnh đúng hành vi".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").casefold())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value.replace("đ", "d")).strip()


@dataclass(frozen=True)
class BehaviorDefinition:
    key: str
    description: str
    query_signals: tuple[str, ...]
    provision_signals: tuple[str, ...]


ACTION_DEFINITIONS: tuple[BehaviorDefinition, ...] = (
    BehaviorDefinition(
        "create_ai_deepfake",
        "tạo hoặc dùng AI/công nghệ để giả mạo video, hình ảnh hoặc giọng nói",
        (
            "deepfake",
            "ai tao video gia",
            "ai tao hinh anh gia",
            "gia mao video",
            "gia mao hinh anh",
            "gia mao giong noi",
        ),
        (
            "deepfake",
            "gia mao video",
            "gia mao hinh anh",
            "gia mao giong noi",
            "video, hinh anh, giong noi gia mao",
            "tri tue nhan tao de gia mao",
            "cong nghe moi de gia mao",
        ),
    ),
    BehaviorDefinition(
        "use_person_likeness",
        "sử dụng hình ảnh, khuôn mặt hoặc giọng nói của người khác",
        (
            "hinh anh nguoi khac",
            "hinh anh ca nhan",
            "hinh anh nguoi noi tieng",
            "nguoi noi tieng",
            "khuon mat nguoi khac",
            "giong noi nguoi khac",
            "quyen hinh anh",
        ),
        (
            "su dung hinh anh cua ca nhan",
            "quyen cua ca nhan doi voi hinh anh",
            "hinh anh cua nguoi khac",
            "hinh anh ca nhan",
            "khuon mat cua ca nhan",
            "giong noi cua ca nhan",
        ),
    ),
    BehaviorDefinition(
        "unauthorized_access",
        "truy cập hoặc xâm nhập trái phép website/hệ thống thông tin",
        (
            "sql injection",
            "truy cap trai phep",
            "xam nhap trai phep",
            "hack website",
            "hack he thong",
        ),
        (
            "truy cap trai phep",
            "xam nhap trai phep",
            "xam nhap vao he thong",
            "truy cap bat hop phap",
            "chiem quyen dieu khien",
            "tan cong mang",
        ),
    ),
    BehaviorDefinition(
        "exploit_vulnerability",
        "khai thác điểm yếu hoặc lỗ hổng bảo mật",
        (
            "sql injection",
            "khai thac lo hong",
            "lo hong website",
            "lo hong bao mat",
            "diem yeu bao mat",
        ),
        (
            "khai thac diem yeu",
            "khai thac lo hong",
            "diem yeu, lo hong",
            "lo hong bao mat",
            "loi bao mat",
        ),
    ),
    BehaviorDefinition(
        "extract_or_download_data",
        "chiếm đoạt, trích xuất hoặc tải xuống dữ liệu",
        (
            "tai xuong co so du lieu",
            "tai co so du lieu",
            "tai database",
            "download database",
            "lay cap du lieu",
            "chiem doat du lieu",
            "trich xuat du lieu",
        ),
        (
            "chiem doat thong tin",
            "chiem doat du lieu",
            "lay cap thong tin",
            "lay cap du lieu",
            "tai xuong du lieu",
            "sao chep trai phep du lieu",
        ),
    ),
    BehaviorDefinition(
        "collect_personal_data",
        "thu thập dữ liệu cá nhân",
        (
            "thu thap du lieu ca nhan",
            "thu thap thong tin ca nhan",
            "lay du lieu ca nhan",
        ),
        (
            "thu thap du lieu ca nhan",
            "thu thap thong tin ca nhan",
            "hoat dong thu thap du lieu",
        ),
    ),
    BehaviorDefinition(
        "share_personal_data",
        "chia sẻ, cung cấp hoặc chuyển dữ liệu cá nhân cho bên khác",
        (
            "chia se du lieu ca nhan",
            "cung cap du lieu ca nhan",
            "chuyen du lieu ca nhan",
            "chia se thong tin ca nhan",
        ),
        (
            "chia se du lieu ca nhan",
            "cung cap du lieu ca nhan",
            "chuyen du lieu ca nhan",
            "cung cap thong tin cua ca nhan",
            "chia se thong tin cua ca nhan",
        ),
    ),
    BehaviorDefinition(
        "process_personal_data_without_consent",
        "xử lý dữ liệu cá nhân khi chưa có sự đồng ý",
        (
            "xu ly du lieu ca nhan khi chua co su dong y",
            "xu ly du lieu ca nhan khong co su dong y",
            "khong xin phep chu the du lieu",
            "chua duoc dong y",
        ),
        (
            "xu ly du lieu ca nhan khi chua co su dong y",
            "xu ly du lieu ca nhan khong co su dong y",
            "su dong y cua chu the du lieu",
            "khong co su dong y cua chu the du lieu",
        ),
    ),
    BehaviorDefinition(
        "publish_false_content",
        "đăng hoặc phát tán thông tin giả/sai sự thật trên mạng",
        (
            "dang tin gia",
            "phat tan tin gia",
            "thong tin sai su that",
            "phat tan thong tin sai",
        ),
        (
            "cung cap thong tin gia mao",
            "thong tin sai su that",
            "phat tan tin gia",
            "dang tai thong tin sai su that",
        ),
    ),
    BehaviorDefinition(
        "misleading_advertising",
        "thực hiện quảng cáo sai sự thật hoặc gây nhầm lẫn",
        (
            "quang cao sai su that",
            "quang cao gay nham lan",
            "lua doi trong quang cao",
        ),
        (
            "quang cao sai su that",
            "quang cao gay nham lan",
            "thong tin quang cao khong chinh xac",
            "lua doi nguoi tieu dung",
        ),
    ),
    BehaviorDefinition(
        "use_copyrighted_work",
        "sao chép hoặc sử dụng tác phẩm/bản ghi được bảo hộ",
        (
            "sao chep tac pham",
            "su dung tac pham",
            "su dung ban ghi",
            "video co ban quyen",
            "xam pham ban quyen",
        ),
        (
            "sao chep tac pham",
            "su dung tac pham",
            "su dung ban ghi am, ghi hinh",
            "quyen tac gia",
            "quyen lien quan",
        ),
    ),
)


OBJECT_DEFINITIONS: tuple[BehaviorDefinition, ...] = (
    BehaviorDefinition(
        "synthetic_media",
        "video, hình ảnh hoặc giọng nói tổng hợp/giả mạo",
        ("deepfake", "video gia", "hinh anh gia", "giong noi gia"),
        ("video", "hinh anh", "giong noi", "noi dung gia mao"),
    ),
    BehaviorDefinition(
        "person_likeness",
        "hình ảnh, khuôn mặt hoặc giọng nói của cá nhân",
        ("nguoi noi tieng", "hinh anh ca nhan", "khuon mat", "giong noi"),
        ("hinh anh ca nhan", "hinh anh cua ca nhan", "khuon mat", "giong noi"),
    ),
    BehaviorDefinition(
        "website_or_information_system",
        "website hoặc hệ thống thông tin",
        ("website", "he thong thong tin", "may chu", "server"),
        ("website", "he thong thong tin", "may chu", "he thong mang"),
    ),
    BehaviorDefinition(
        "database",
        "cơ sở dữ liệu",
        ("co so du lieu", "database"),
        ("co so du lieu", "database", "du lieu luu tru"),
    ),
    BehaviorDefinition(
        "personal_data",
        "dữ liệu hoặc thông tin cá nhân",
        ("du lieu ca nhan", "thong tin ca nhan", "chu the du lieu"),
        ("du lieu ca nhan", "thong tin ca nhan", "chu the du lieu"),
    ),
    BehaviorDefinition(
        "biometric_data",
        "dữ liệu sinh trắc học",
        ("du lieu sinh trac", "sinh trac hoc", "van tay"),
        ("du lieu sinh trac", "sinh trac hoc", "van tay"),
    ),
    BehaviorDefinition(
        "copyrighted_work_or_recording",
        "tác phẩm hoặc bản ghi được bảo hộ",
        ("tac pham", "ban ghi", "ban quyen"),
        ("tac pham", "ban ghi am", "ban ghi hinh", "ban quyen"),
    ),
)


PURPOSE_DEFINITIONS: tuple[BehaviorDefinition, ...] = (
    BehaviorDefinition(
        "advertising",
        "dùng cho quảng cáo hoặc tiếp thị",
        ("quang cao", "tiep thi", "marketing", "quang ba san pham"),
        ("quang cao", "tiep thi", "marketing", "muc dich thuong mai"),
    ),
    BehaviorDefinition(
        "commercial_gain",
        "nhằm mục đích thương mại hoặc thu lợi",
        ("muc dich thuong mai", "kinh doanh", "thu loi", "kiem tien"),
        ("muc dich thuong mai", "kinh doanh", "thu loi", "thu tien"),
    ),
    BehaviorDefinition(
        "data_exfiltration",
        "lấy hoặc chiếm đoạt dữ liệu",
        ("tai xuong co so du lieu", "lay cap du lieu", "chiem doat du lieu"),
        ("chiem doat thong tin", "lay cap du lieu", "chiem doat du lieu"),
    ),
)


CONDITION_DEFINITIONS: tuple[BehaviorDefinition, ...] = (
    BehaviorDefinition(
        "without_consent",
        "chưa có sự đồng ý của chủ thể/cá nhân",
        ("khong co su dong y", "chua co su dong y", "khong xin phep", "chua xin phep"),
        ("khong co su dong y", "chua co su dong y", "su dong y cua chu the"),
    ),
    BehaviorDefinition(
        "without_authorization",
        "không được phép hoặc không có quyền truy cập",
        ("trai phep", "khong duoc phep", "khong co quyen"),
        ("trai phep", "khong duoc phep", "khong co quyen"),
    ),
    BehaviorDefinition(
        "public_distribution",
        "đăng tải hoặc phát tán công khai",
        ("dang tai", "phat tan", "cong khai", "mang xa hoi"),
        ("dang tai", "phat tan", "cong khai", "tren mang"),
    ),
)


_ACTION_BY_KEY = {definition.key: definition for definition in ACTION_DEFINITIONS}
_OBJECT_BY_KEY = {definition.key: definition for definition in OBJECT_DEFINITIONS}
_PURPOSE_BY_KEY = {definition.key: definition for definition in PURPOSE_DEFINITIONS}
_CONDITION_BY_KEY = {definition.key: definition for definition in CONDITION_DEFINITIONS}

# Hành vi có tính định danh cao hơn nhận trọng số lớn hơn khi query chứa nhiều
# action. Nếu query chỉ có một action thì coverage vẫn là 1.0 như bình thường.
_ACTION_IMPORTANCE: dict[str, float] = {
    "create_ai_deepfake": 2.0,
    "exploit_vulnerability": 1.3,
    "extract_or_download_data": 1.2,
}


@dataclass(frozen=True)
class BehaviorProfile:
    actions: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    purposes: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.actions or self.objects or self.purposes or self.conditions)

    @property
    def has_primary_action(self) -> bool:
        return bool(self.actions)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "actions": list(self.actions),
            "objects": list(self.objects),
            "purposes": list(self.purposes),
            "conditions": list(self.conditions),
        }

    def enrich_query(self, original_query: str) -> str:
        """Thêm mô tả tiếng Việt, không thay query dùng để candidate search."""

        if self.is_empty:
            return original_query
        parts = [f"Câu hỏi gốc: {original_query}"]
        mappings = (
            ("Hành vi", self.actions, _ACTION_BY_KEY),
            ("Đối tượng", self.objects, _OBJECT_BY_KEY),
            ("Mục đích", self.purposes, _PURPOSE_BY_KEY),
            ("Điều kiện", self.conditions, _CONDITION_BY_KEY),
        )
        for label, keys, definitions in mappings:
            if keys:
                parts.append(
                    f"{label}: "
                    + "; ".join(definitions[key].description for key in keys)
                )
        return "\n".join(parts)


@dataclass(frozen=True)
class BehaviorMatch:
    score: float
    action_score: float
    object_score: float
    purpose_score: float
    condition_score: float
    matched_actions: tuple[str, ...] = ()
    matched_objects: tuple[str, ...] = ()
    matched_purposes: tuple[str, ...] = ()
    matched_conditions: tuple[str, ...] = ()


def _extract_keys(text: str, definitions: tuple[BehaviorDefinition, ...]) -> list[str]:
    return [
        definition.key
        for definition in definitions
        if any(signal in text for signal in definition.query_signals)
    ]


def extract_legal_behavior(query: str) -> BehaviorProfile:
    """Trích behavior card xác định từ query gốc, không dùng query expansion."""

    folded = _fold(query)
    actions = _extract_keys(folded, ACTION_DEFINITIONS)
    objects = _extract_keys(folded, OBJECT_DEFINITIONS)
    purposes = _extract_keys(folded, PURPOSE_DEFINITIONS)
    conditions = _extract_keys(folded, CONDITION_DEFINITIONS)

    # Deepfake tự mang đối tượng synthetic media dù câu hỏi viết ngắn chỉ có
    # đúng từ "deepfake". SQL Injection tương tự là hành vi trên hệ thống.
    if "create_ai_deepfake" in actions and "synthetic_media" not in objects:
        objects.append("synthetic_media")
    if (
        {"unauthorized_access", "exploit_vulnerability"}.intersection(actions)
        and "website_or_information_system" not in objects
    ):
        objects.append("website_or_information_system")

    return BehaviorProfile(
        actions=tuple(actions),
        objects=tuple(objects),
        purposes=tuple(purposes),
        conditions=tuple(conditions),
    )


def _matched_keys(
    keys: tuple[str, ...], definitions: dict[str, BehaviorDefinition], text: str
) -> tuple[str, ...]:
    return tuple(
        key
        for key in keys
        if any(signal in text for signal in definitions[key].provision_signals)
    )


def _coverage(matched: tuple[str, ...], expected: tuple[str, ...]) -> float:
    return len(matched) / len(expected) if expected else 0.0


def _weighted_action_coverage(
    matched: tuple[str, ...], expected: tuple[str, ...]
) -> float:
    if not expected:
        return 0.0
    total = sum(_ACTION_IMPORTANCE.get(key, 1.0) for key in expected)
    covered = sum(_ACTION_IMPORTANCE.get(key, 1.0) for key in matched)
    return covered / total if total else 0.0


def score_behavior_relevance(profile: BehaviorProfile, provision_text: str) -> BehaviorMatch:
    """Chấm mức điều luật khớp hành vi, ưu tiên action hơn từ khóa bối cảnh."""

    if profile.is_empty:
        return BehaviorMatch(0.0, 0.0, 0.0, 0.0, 0.0)

    folded = _fold(provision_text)
    matched_actions = _matched_keys(profile.actions, _ACTION_BY_KEY, folded)
    matched_objects = _matched_keys(profile.objects, _OBJECT_BY_KEY, folded)
    matched_purposes = _matched_keys(profile.purposes, _PURPOSE_BY_KEY, folded)
    matched_conditions = _matched_keys(profile.conditions, _CONDITION_BY_KEY, folded)

    action_score = _weighted_action_coverage(matched_actions, profile.actions)
    object_score = _coverage(matched_objects, profile.objects)
    purpose_score = _coverage(matched_purposes, profile.purposes)
    condition_score = _coverage(matched_conditions, profile.conditions)

    if profile.has_primary_action:
        if action_score == 0.0:
            # Trùng video/quảng cáo/dữ liệu nhưng không khớp động từ pháp lý
            # không được vượt cổng behavior.
            score = min(0.15, 0.10 * object_score + 0.04 * purpose_score + 0.01 * condition_score)
        else:
            score = (
                0.72 * action_score
                + 0.15 * object_score
                + 0.08 * purpose_score
                + 0.05 * condition_score
            )
    else:
        score = 0.65 * object_score + 0.25 * purpose_score + 0.10 * condition_score

    return BehaviorMatch(
        score=min(1.0, score),
        action_score=action_score,
        object_score=object_score,
        purpose_score=purpose_score,
        condition_score=condition_score,
        matched_actions=matched_actions,
        matched_objects=matched_objects,
        matched_purposes=matched_purposes,
        matched_conditions=matched_conditions,
    )
