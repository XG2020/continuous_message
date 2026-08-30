"""消息防抖动插件。

NekroAgent 的消息回调和消息服务。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import Field

from nekro_agent.api.plugin import ConfigBase, NekroPlugin
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.schemas.chat_message import (
    ChatMessage,
    ChatMessageSegment,
    ChatMessageSegmentType,
    ChatType,
)
from nekro_agent.schemas.signal import MsgSignal

from .core import (
    DebounceState,
    calculate_wait_duration,
    format_wait_reason,
    is_command,
    merge_text,
)


plugin = NekroPlugin(
    name="消息防抖动",
    module_name="continuous_message",
    description="合并短时间内连续发送的消息，改善多条碎片消息的上下文和触发体验。",
    version="1.0.0",
    author="XGGM",
    url="https://github.com/XG2020/continuous_message",
    allow_sleep=False,
    sleep_brief="仅在收到聊天消息时合并短时间内的连续消息。",
)


@plugin.mount_config()
class ContinuousMessageConfig(ConfigBase):
    ENABLE: bool = Field(default=True, title="启用消息防抖动", description="是否启用连续消息合并。")
    ENABLE_PRIVATE: bool = Field(default=True, title="启用私聊防抖", description="是否处理私聊消息。")
    ENABLE_GROUP: bool = Field(default=False, title="启用群聊防抖", description="是否处理群聊消息。群聊会按发送者分别建立会话。")
    DEBOUNCE_TIME: float = Field(default=2.0, title="固定防抖秒数", description="关闭自适应防抖时使用的等待时间。")
    MERGE_SEPARATOR: str = Field(default="\n", title="消息合并分隔符", description="多条消息合并时插入的分隔符。")
    COMMAND_PREFIXES: list[str] = Field(default=["/"], title="命令前缀", description="以这些前缀开头的消息不参与合并，并放行给命令系统。")

    ENABLE_ADAPTIVE: bool = Field(default=True, title="启用自适应防抖", description="根据消息长度、标点和连续短消息数量调整等待时间。")
    ADAPTIVE_MIN_WAIT: float = Field(default=1.0, title="自适应最短等待", description="自适应模式的最短等待秒数。")
    ADAPTIVE_MAX_WAIT: float = Field(default=6.0, title="自适应最长等待", description="单轮自适应等待的上限。")
    ADAPTIVE_MAX_TOTAL_WAIT: float = Field(default=12.0, title="最大总等待", description="同一会话从第一条消息开始计算的最大等待秒数。")
    ADAPTIVE_SHORT_MESSAGE_THRESHOLD: int = Field(default=10, title="短消息长度阈值", description="不超过该字符数的消息计入连续短消息。")

    ENABLE_TYPING_DETECTION: bool = Field(default=True, title="启用输入状态感知", description="识别适配器提供的输入状态字段，并在输入期间延长等待。")
    MAX_TYPING_WAIT: float = Field(default=60.0, title="输入状态最大等待", description="用户持续输入时的超时保护秒数。")
    ENABLE_RECALL_FILTER: bool = Field(default=True, title="过滤撤回消息", description="在结算前移除已标记为撤回的消息。")

    ID_ACCESS_MODE: str = Field(default="all", title="用户访问模式", description="all=全部用户；whitelist=仅名单；blacklist=名单用户放行但不防抖。")
    ID_LIST: list[str] = Field(default_factory=list, title="用户 ID 名单", description="按平台用户 ID 匹配，支持字符串或数字形式。")


config: ContinuousMessageConfig = plugin.get_config(ContinuousMessageConfig)


@dataclass
class _PendingSession:
    key: str
    chat_key: str
    first_message: ChatMessage
    user: Any = None
    messages: list[ChatMessage] = field(default_factory=list)
    flush_event: asyncio.Event = field(default_factory=asyncio.Event)
    timer_task: Optional[asyncio.Task] = None
    started_at: float = 0.0
    short_message_count: int = 0
    is_typing: bool = False

    @property
    def state(self) -> DebounceState:
        return DebounceState(
            started_at=self.started_at,
            short_message_count=self.short_message_count,
            texts=[message.content_text for message in self.messages],
        )


_sessions: dict[str, _PendingSession] = {}


def _chat_type_value(message: ChatMessage) -> str:
    value = message.chat_type
    return value.value if isinstance(value, ChatType) else str(value)


def _user_id(message: ChatMessage) -> str:
    return str(message.platform_userid or message.sender_id or "")


def _session_key(message: ChatMessage) -> str:
    if _chat_type_value(message) == ChatType.GROUP.value:
        return f"{message.chat_key}::user:{_user_id(message)}"
    return message.chat_key


def _is_typing_notice(message: ChatMessage) -> Optional[bool]:
    """从适配器扩展字段中读取通用输入状态。

    不同适配器字段命名可能不同，因此这里只识别明确的布尔值或常见状态文本，
    不把普通聊天内容误判为输入状态通知。
    """

    ext = message.ext_data if isinstance(message.ext_data, dict) else {}
    raw = ext.get("raw") if isinstance(ext.get("raw"), dict) else ext
    event_type = str(raw.get("event_type", raw.get("notice_type", ""))).lower()
    status_text = str(raw.get("status_text", raw.get("status", ""))).lower()
    has_typing_flag = "is_typing" in raw or "typing" in raw
    if not has_typing_flag and not any(token in event_type for token in ("typing", "input", "input_status")) and "正在输入" not in status_text:
        return None
    if "stopped" in event_type or "stop" in event_type or "结束" in status_text or "停止" in status_text:
        return False
    if raw.get("is_typing") is False or raw.get("typing") is False:
        return False
    return True


def _copy_non_text_segments(messages: list[ChatMessage]) -> list[ChatMessageSegment]:
    segments: list[ChatMessageSegment] = []
    for message in messages:
        for segment in message.content_data:
            segment_type = segment.type.value if isinstance(segment.type, ChatMessageSegmentType) else str(segment.type)
            if segment_type == ChatMessageSegmentType.TEXT.value:
                continue
            try:
                segments.append(segment.model_copy(deep=True))
            except AttributeError:
                segments.append(segment.copy(deep=True))
    return segments


def _make_merged_message(session: _PendingSession) -> Optional[ChatMessage]:
    if not session.messages:
        return None
    merged_text = merge_text((message.content_text for message in session.messages), config.MERGE_SEPARATOR)
    segments: list[ChatMessageSegment] = []
    if merged_text:
        segments.append(ChatMessageSegment(type=ChatMessageSegmentType.TEXT, text=merged_text))
    segments.extend(_copy_non_text_segments(session.messages))
    if not merged_text and not segments:
        return None

    first = session.first_message
    source_ids = [str(message.message_id) for message in session.messages if message.message_id]
    ext_data = dict(first.ext_data or {})
    ext_data.update(
        {
            "continuous_message_merged": True,
            "continuous_message_count": len(session.messages),
            "continuous_message_source_ids": source_ids,
        }
    )
    # 沿用首条真实消息 ID，便于适配器设置处理状态；原始碎片消息未单独写入历史。
    message_id = first.message_id or f"continuous-{uuid.uuid4().hex}"
    return first.model_copy(
        update={
            "message_id": message_id,
            "content_text": merged_text,
            "content_data": segments,
            "raw_cq_code": first.raw_cq_code,
            "ext_data": ext_data,
            "send_timestamp": int(time.time()),
        },
        deep=True,
    )


async def _timer(session: _PendingSession, duration: float) -> None:
    try:
        await asyncio.sleep(max(0.0, duration))
        session.flush_event.set()
    except asyncio.CancelledError:
        return


async def _flush_session(session: _PendingSession) -> None:
    merged = _make_merged_message(session)
    if merged is None:
        return
    try:
        from nekro_agent.services.message_service import message_service

        await message_service.push_human_message(merged, user=session.user, trigger_agent=True)
        plugin.logger.info(
            f"会话结算完成：{session.chat_key} 合并 {len(session.messages)} 条消息，"
            f"文本 {len(merged.content_text)} 字，媒体 {len(merged.content_data) - (1 if merged.content_text else 0)} 段"
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        plugin.logger.exception(f"会话结算失败：{session.chat_key}")


async def _wait_and_flush(session: _PendingSession) -> None:
    await session.flush_event.wait()
    current = _sessions.get(session.key)
    if current is not session:
        return
    _sessions.pop(session.key, None)
    if session.timer_task and session.timer_task is not asyncio.current_task() and not session.timer_task.done():
        session.timer_task.cancel()
    await _flush_session(session)


def _cancel_session(key: str) -> None:
    session = _sessions.pop(key, None)
    if not session:
        return
    if session.timer_task and not session.timer_task.done():
        session.timer_task.cancel()
    session.flush_event.set()


@plugin.mount_on_user_message()
async def handle_user_message(ctx: AgentCtx, message: ChatMessage) -> MsgSignal:
    """收集消息并在窗口结束时注入一条合并消息。"""

    if not config.ENABLE or config.DEBOUNCE_TIME <= 0:
        return MsgSignal.CONTINUE

    # 合并消息再次进入消息管线时直接放行，避免递归拦截。
    if isinstance(message.ext_data, dict) and message.ext_data.get("continuous_message_merged"):
        return MsgSignal.CONTINUE

    typing_state = _is_typing_notice(message) if config.ENABLE_TYPING_DETECTION else None
    key = _session_key(message)
    if typing_state is not None:
        session = _sessions.get(key)
        if session:
            session.is_typing = typing_state
            if session.timer_task and not session.timer_task.done():
                session.timer_task.cancel()
            wait = config.MAX_TYPING_WAIT if typing_state else config.DEBOUNCE_TIME
            session.timer_task = asyncio.create_task(_timer(session, wait))
            plugin.logger.debug(f"输入状态变更：{message.chat_key} typing={typing_state}，等待 {wait:.2f}s")
        return MsgSignal.BLOCK_ALL

    if config.ENABLE_RECALL_FILTER and message.is_recalled:
        session = _sessions.get(key)
        if session:
            session.messages = [item for item in session.messages if str(item.message_id) != str(message.message_id)]
            if not session.messages:
                _cancel_session(key)
        return MsgSignal.BLOCK_ALL

    chat_type = _chat_type_value(message)
    if chat_type == ChatType.PRIVATE.value:
        if not config.ENABLE_PRIVATE:
            return MsgSignal.CONTINUE
    elif chat_type == ChatType.GROUP.value:
        if not config.ENABLE_GROUP:
            return MsgSignal.CONTINUE
    else:
        return MsgSignal.CONTINUE

    user_id = _user_id(message)
    access_mode = str(config.ID_ACCESS_MODE or "all").strip().lower()
    allowed_ids = {str(item) for item in (config.ID_LIST or [])}
    if access_mode == "whitelist" and user_id not in allowed_ids:
        return MsgSignal.CONTINUE
    if access_mode == "blacklist" and user_id in allowed_ids:
        return MsgSignal.CONTINUE

    text = (message.content_text or "").strip()
    has_media = bool(message.content_data) and any(
        (segment.type.value if isinstance(segment.type, ChatMessageSegmentType) else str(segment.type)) != ChatMessageSegmentType.TEXT.value
        for segment in message.content_data
    )
    if is_command(text, config.COMMAND_PREFIXES):
        if key in _sessions:
            _sessions[key].flush_event.set()
        return MsgSignal.CONTINUE
    if not text and not has_media:
        return MsgSignal.CONTINUE

    session = _sessions.get(key)
    if session is None:
        session = _PendingSession(
            key=key,
            chat_key=message.chat_key,
            first_message=message,
            user=getattr(ctx, "_trigger_db_user", None),
            messages=[message],
            started_at=time.monotonic(),
        )
        _sessions[key] = session
        wait, _, reason = calculate_wait_duration(
            text,
            state=None,
            now=time.monotonic(),
            enabled=config.ENABLE_ADAPTIVE,
            fixed_wait=config.DEBOUNCE_TIME,
            min_wait=config.ADAPTIVE_MIN_WAIT,
            max_wait=config.ADAPTIVE_MAX_WAIT,
            max_total_wait=config.ADAPTIVE_MAX_TOTAL_WAIT,
            short_threshold=config.ADAPTIVE_SHORT_MESSAGE_THRESHOLD,
        )
        session.short_message_count = 1 if len(text) <= config.ADAPTIVE_SHORT_MESSAGE_THRESHOLD else 0
        plugin.logger.info(f"开始收集：{message.chat_key}，首次等待 {wait:.2f}s（{format_wait_reason(reason)}）")
        session.timer_task = asyncio.create_task(_timer(session, wait))
        await _wait_and_flush(session)
        return MsgSignal.BLOCK_ALL

    session.messages.append(message)
    session.user = session.user or getattr(ctx, "_trigger_db_user", None)
    if session.timer_task and not session.timer_task.done():
        session.timer_task.cancel()
    state = session.state
    wait, flush_now, reason = calculate_wait_duration(
        text,
        state=state,
        now=time.monotonic(),
        enabled=config.ENABLE_ADAPTIVE,
        fixed_wait=config.DEBOUNCE_TIME,
        min_wait=config.ADAPTIVE_MIN_WAIT,
        max_wait=config.ADAPTIVE_MAX_WAIT,
        max_total_wait=config.ADAPTIVE_MAX_TOTAL_WAIT,
        short_threshold=config.ADAPTIVE_SHORT_MESSAGE_THRESHOLD,
    )
    session.short_message_count = state.short_message_count
    if flush_now:
        session.flush_event.set()
    else:
        session.timer_task = asyncio.create_task(_timer(session, wait))
    plugin.logger.debug(
        f"追加消息：{message.chat_key}，累计 {len(session.messages)} 条，等待 {wait:.2f}s（{format_wait_reason(reason)}）"
    )
    return MsgSignal.BLOCK_ALL


@plugin.mount_on_channel_reset()
async def reset_channel(ctx: AgentCtx) -> None:
    """频道重置时清理相关会话和计时器。"""

    chat_key = ctx.chat_key
    for key in [key for key, session in _sessions.items() if session.chat_key == chat_key]:
        _cancel_session(key)


@plugin.mount_cleanup_method()
async def cleanup() -> None:
    """插件卸载时取消全部计时器。"""

    for key in list(_sessions):
        _cancel_session(key)


