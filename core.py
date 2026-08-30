"""消息防抖动的纯逻辑工具。

本模块不依赖 NekroAgent 运行时，方便单元测试和后续复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


_UNFINISHED_ENDINGS = ("...", "…", "，", "、", "：", ":")
_QUESTION_ENDINGS = ("？", "?")
_SENTENCE_ENDINGS = ("。", "！", "!")
_PUNCTUATION = "。！？!?，、：:；;,.…"


@dataclass
class DebounceState:
    """一个聊天来源的防抖状态。"""

    started_at: float
    short_message_count: int = 0
    texts: list[str] = field(default_factory=list)


def is_command(text: str, prefixes: Iterable[str]) -> bool:
    """判断消息是否以配置的命令前缀开头。"""

    clean = (text or "").strip()
    return bool(clean) and any(clean.startswith(prefix) for prefix in prefixes if prefix)


def is_short_message(text: str, threshold: int) -> bool:
    return len((text or "").strip()) <= max(0, threshold)


def calculate_wait_duration(
    text: str,
    *,
    state: DebounceState | None,
    now: float,
    enabled: bool,
    fixed_wait: float,
    min_wait: float,
    max_wait: float,
    max_total_wait: float,
    short_threshold: int,
) -> tuple[float, bool, str]:
    """计算下一次结算等待时间。

    返回 ``(等待秒数, 是否应立即结算, 原因)``。等待会受最大总等待时间限制，
    这样高速连续发送短消息时也不会无限期占用会话。
    """

    if not enabled:
        return max(0.0, fixed_wait), False, "fixed_debounce"

    clean = (text or "").strip()
    length = len(clean)
    reasons: list[str] = []

    if length <= 3:
        wait = 4.0
        reasons.append("very_short")
    elif length <= 10:
        wait = 3.2
        reasons.append("short")
    elif length <= 30:
        wait = 2.7
        reasons.append("medium")
    elif length <= 80:
        wait = 1.8
        reasons.append("long")
    else:
        wait = 1.0
        reasons.append("very_long")

    if clean.endswith(_UNFINISHED_ENDINGS):
        wait += 1.8
        reasons.append("unfinished_punctuation")
    elif clean.endswith(_QUESTION_ENDINGS):
        wait -= 0.8
        reasons.append("question_end")
    elif clean.endswith(_SENTENCE_ENDINGS):
        wait -= 0.5
        reasons.append("sentence_end")
    elif clean and clean[-1] not in _PUNCTUATION:
        wait += 0.8 if length <= 10 else 0.7 if length <= 30 else 0.3 if length <= 80 else 0
        reasons.append("chat_plain_end")

    if state is not None:
        if is_short_message(clean, short_threshold):
            state.short_message_count += 1
        else:
            state.short_message_count = 0
        if state.short_message_count >= 4:
            wait += 0.8
            reasons.append("short_streak_4plus")
        elif state.short_message_count == 3:
            wait += 0.6
            reasons.append("short_streak_3")
        elif state.short_message_count == 2:
            wait += 0.3
            reasons.append("short_streak_2")

        elapsed = max(0.0, now - state.started_at)
        remaining = max_total_wait - elapsed
        if remaining <= 0:
            return 0.0, True, ",".join(reasons + ["max_total_wait_reached"])
        wait = min(wait, remaining)
        if wait <= 0:
            return 0.0, True, ",".join(reasons + ["max_total_wait_reached"])
        if wait < max(min_wait, 0.0):
            wait = max(0.0, remaining)
            if wait <= 0:
                return 0.0, True, ",".join(reasons + ["max_total_wait_reached"])
            reasons.append("limited_by_total_wait")

    wait = max(max(0.0, min_wait), min(wait, max(max(0.0, min_wait), max_wait)))
    return wait, False, ",".join(reasons) or "adaptive"


def merge_text(texts: Iterable[str], separator: str) -> str:
    """清理并合并文本，跳过空消息。"""

    return separator.join(text.strip() for text in texts if text and text.strip()).strip()


def format_wait_reason(reason: str) -> str:
    labels = {
        "fixed_debounce": "固定防抖",
        "very_short": "极短消息",
        "short": "短消息",
        "medium": "中等长度",
        "long": "较长消息",
        "very_long": "长消息",
        "unfinished_punctuation": "延续信号",
        "question_end": "问号结尾",
        "sentence_end": "主动结束标点",
        "chat_plain_end": "口语无标点",
        "short_streak_2": "连续短句x2",
        "short_streak_3": "连续短句x3",
        "short_streak_4plus": "连续短句x4+",
        "limited_by_total_wait": "受总等待上限限制",
        "max_total_wait_reached": "达到总等待上限",
        "adaptive": "自适应防抖",
    }
    return "+".join(labels.get(item, item) for item in (reason or "adaptive").split(",") if item) or labels["adaptive"]
