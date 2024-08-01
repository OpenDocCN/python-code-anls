# `.\DB-GPT-src\dbgpt\storage\chat_history\storage_adapter.py`

```py
"""Adapter for chat history storage."""

import json  # 导入 JSON 模块，用于处理 JSON 数据
from typing import Dict, List, Optional, Type  # 导入类型提示相关的模块

from sqlalchemy.orm import Session  # 导入 SQLAlchemy 的 Session 类

from dbgpt.core.interface.message import (  # 导入消息相关的接口和函数
    BaseMessage,
    ConversationIdentifier,
    MessageIdentifier,
    MessageStorageItem,
    StorageConversation,
    _conversation_to_dict,
    _messages_from_dict,
)
from dbgpt.core.interface.storage import StorageItemAdapter  # 导入存储适配器接口

from .chat_history_db import ChatHistoryEntity, ChatHistoryMessageEntity  # 导入本地的数据库实体


class DBStorageConversationItemAdapter(
    StorageItemAdapter[StorageConversation, ChatHistoryEntity]
):
    """Adapter for chat history storage."""

    def to_storage_format(self, item: StorageConversation) -> ChatHistoryEntity:
        """Convert to storage format."""
        message_ids = ",".join(item.message_ids)  # 将消息 ID 列表转换为逗号分隔的字符串
        messages = None
        if not item.save_message_independent and item.messages:
            message_dict_list = [_conversation_to_dict(item)]
            messages = json.dumps(message_dict_list, ensure_ascii=False)  # 将消息字典列表转换为 JSON 字符串
        summary = item.summary  # 获取会话摘要信息
        latest_user_message = item.get_latest_user_message()  # 获取最新的用户消息
        if not summary and latest_user_message is not None:
            summary = latest_user_message.content  # 如果没有摘要且有最新用户消息，则摘要为最新消息的内容
        return ChatHistoryEntity(
            conv_uid=item.conv_uid,  # 设置会话唯一标识符
            chat_mode=item.chat_mode,  # 设置会话的聊天模式
            summary=summary,  # 设置会话的摘要信息
            user_name=item.user_name,  # 设置用户名称
            # We not save messages to chat_history table in new design
            messages=messages,  # 设置消息内容（如果有的话）
            message_ids=message_ids,  # 设置消息的 ID 列表
            sys_code=item.sys_code,  # 设置系统代码
        )

    def from_storage_format(self, model: ChatHistoryEntity) -> StorageConversation:
        """Convert from storage format."""
        message_ids = model.message_ids.split(",") if model.message_ids else []  # 拆分消息 ID 字符串为列表
        old_conversations: List[Dict] = (
            json.loads(model.messages) if model.messages else []  # 解析 JSON 格式的旧会话数据
        )
        old_messages = []
        save_message_independent = True
        if old_conversations:
            # Load old messages from old conversations, in old design, we save messages
            # to chat_history table
            save_message_independent = False  # 标记为不独立保存消息
            old_messages = _parse_old_conversations(old_conversations)  # 解析旧会话数据为消息列表
        return StorageConversation(
            conv_uid=model.conv_uid,  # 设置会话唯一标识符
            chat_mode=model.chat_mode,  # 设置会话的聊天模式
            summary=model.summary,  # 设置会话的摘要信息
            user_name=model.user_name,  # 设置用户名称
            message_ids=message_ids,  # 设置消息的 ID 列表
            sys_code=model.sys_code,  # 设置系统代码
            save_message_independent=save_message_independent,  # 设置是否独立保存消息的标志
            messages=old_messages,  # 设置消息内容（如果有的话）
        )

    def get_query_for_identifier(
        self,
        storage_format: Type[ChatHistoryEntity],  # 定义存储格式类型
        resource_id: ConversationIdentifier,  # 定义对话标识符类型
        **kwargs,  # 其他关键字参数

        ):
        """Generate query based on conversation identifier."""
        # This method generates a query based on the conversation identifier and other optional parameters.
        # It is used to retrieve data from the database related to a specific conversation.
        pass
        """Get query for identifier."""
        # 定义函数的文档字符串，说明这个函数的作用是获取特定标识符的查询对象

        # 从关键字参数中获取会话对象，类型为可选的会话对象
        session: Optional[Session] = kwargs.get("session")
        
        # 如果会话对象为 None，则抛出异常
        if session is None:
            raise Exception("session is None")
        
        # 返回查询对象，使用 ChatHistoryEntity 表格，过滤条件为 conv_uid 等于 resource_id.conv_uid
        return session.query(ChatHistoryEntity).filter(
            ChatHistoryEntity.conv_uid == resource_id.conv_uid
        )
class DBMessageStorageItemAdapter(
    StorageItemAdapter[MessageStorageItem, ChatHistoryMessageEntity]
):
    """Adapter for chat history message storage."""

    def to_storage_format(self, item: MessageStorageItem) -> ChatHistoryMessageEntity:
        """Convert to storage format."""
        # 获取消息详情中的轮次索引，如果不存在则默认为 0
        round_index = item.message_detail.get("round_index", 0)
        # 将消息详情转换为 JSON 字符串
        message_detail = json.dumps(item.message_detail, ensure_ascii=False)
        # 返回 ChatHistoryMessageEntity 对象，用于存储
        return ChatHistoryMessageEntity(
            conv_uid=item.conv_uid,
            index=item.index,
            round_index=round_index,
            message_detail=message_detail,
        )

    def from_storage_format(
        self, model: ChatHistoryMessageEntity
    ) -> MessageStorageItem:
        """Convert from storage format."""
        # 将存储格式中的消息详情 JSON 字符串解析为 Python 对象
        message_detail = (
            json.loads(model.message_detail)  # type: ignore
            if model.message_detail
            else {}
        )
        # 返回 MessageStorageItem 对象，表示从存储格式中解析出的消息
        return MessageStorageItem(
            conv_uid=model.conv_uid,  # type: ignore
            index=model.index,  # type: ignore
            message_detail=message_detail,
        )

    def get_query_for_identifier(
        self,
        storage_format: Type[ChatHistoryMessageEntity],
        resource_id: MessageIdentifier,  # type: ignore
        **kwargs,
    ):
        """Get query for identifier."""
        # 获取会话对象，如果不存在则抛出异常
        session: Optional[Session] = kwargs.get("session")
        if session is None:
            raise Exception("session is None")
        # 返回查询对象，用于从数据库中获取与 resource_id 匹配的记录
        return session.query(ChatHistoryMessageEntity).filter(
            ChatHistoryMessageEntity.conv_uid == resource_id.conv_uid,
            ChatHistoryMessageEntity.index == resource_id.index,
        )


def _parse_old_conversations(old_conversations: List[Dict]) -> List[BaseMessage]:
    # 初始化空列表，用于存储旧消息的字典形式
    old_messages_dict = []
    # 遍历旧对话列表中的每个对话
    for old_conversation in old_conversations:
        # 获取对话中的消息列表，如果不存在则为空列表
        messages = (
            old_conversation["messages"] if "messages" in old_conversation else []
        )
        # 遍历对话中的每条消息
        for message in messages:
            # 如果消息中包含"data"字段
            if "data" in message:
                # 获取消息的数据部分
                message_data = message["data"]
                # 获取消息数据中的额外关键字参数字典，如果不存在则为空字典
                additional_kwargs = message_data.get("additional_kwargs", {})
                # 设置额外关键字参数的值，这些值来自于对话本身的参数
                additional_kwargs["param_value"] = old_conversation.get("param_value")
                additional_kwargs["param_type"] = old_conversation.get("param_type")
                additional_kwargs["model_name"] = old_conversation.get("model_name")
                # 更新消息数据的额外关键字参数
                message_data["additional_kwargs"] = additional_kwargs

        # 将处理过的消息列表添加到旧消息字典列表中
        old_messages_dict.extend(messages)

    # 将旧消息字典列表转换为消息对象列表
    old_messages: List[BaseMessage] = _messages_from_dict(old_messages_dict)
    # 返回解析得到的旧消息对象列表
    return old_messages
```