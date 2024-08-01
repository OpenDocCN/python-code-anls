# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\editor\_chat_history\meta_db_history.py`

```py
# 导入必要的模块和库
import json  # 导入用于 JSON 操作的模块
import logging  # 导入日志记录模块
from typing import Dict, List, Optional  # 导入类型提示相关的模块

from dbgpt._private.config import Config  # 从私有库中导入 Config 类
from dbgpt.core.interface.message import OnceConversation, _conversation_to_dict  # 导入消息接口相关模块
from dbgpt.storage.chat_history.chat_history_db import ChatHistoryDao, ChatHistoryEntity  # 导入聊天历史数据库相关模块

from .base import BaseChatHistoryMemory, MemoryStoreType  # 从当前包中导入基础聊天历史内存存储相关模块

CFG = Config()  # 创建 Config 对象实例
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器实例


class DbHistoryMemory(BaseChatHistoryMemory):
    """Db history memory storage.

    It is deprecated.
    """

    store_type: str = MemoryStoreType.DB.value  # 设置存储类型为数据库，标记为可忽略的类型提示

    def __init__(self, chat_session_id: str):
        self.chat_seesion_id = chat_session_id  # 初始化聊天会话 ID
        self.chat_history_dao = ChatHistoryDao()  # 初始化聊天历史数据访问对象

    def messages(self) -> List[OnceConversation]:
        # 获取特定聊天会话 ID 的聊天历史记录实体
        chat_history: Optional[ChatHistoryEntity] = self.chat_history_dao.get_by_uid(
            self.chat_seesion_id
        )
        if chat_history:
            context = chat_history.messages  # 获取聊天历史记录中的消息内容
            if context:
                conversations: List[OnceConversation] = json.loads(
                    context  # 将 JSON 字符串解析为对话列表，标记为可忽略的类型提示
                )
                return conversations  # 返回解析后的对话列表
        return []  # 如果没有有效的消息内容，则返回空列表

    # def create(self, chat_mode, summary: str, user_name: str) -> None:
    #     try:
    #         chat_history: ChatHistoryEntity = ChatHistoryEntity()
    #         chat_history.chat_mode = chat_mode
    #         chat_history.summary = summary
    #         chat_history.user_name = user_name
    #
    #         self.chat_history_dao.raw_update(chat_history)
    #     except Exception as e:
    #         logger.error("init create conversation log error！" + str(e))
    #

    def append(self, once_message: OnceConversation) -> None:
        logger.debug(f"db history append: {once_message}")  # 记录调试日志，显示追加的一次消息对象

        # 获取特定聊天会话 ID 的聊天历史记录实体
        chat_history: Optional[ChatHistoryEntity] = self.chat_history_dao.get_by_uid(
            self.chat_seesion_id
        )
        conversations: List[Dict] = []  # 初始化对话列表

        # 获取最新用户消息的摘要内容
        latest_user_message = once_message.get_latest_user_message()
        summary = latest_user_message.content if latest_user_message else ""

        if chat_history:
            context = chat_history.messages  # 获取聊天历史记录中的消息内容
            if context:
                conversations = json.loads(context)  # 解析已有的 JSON 内容为对话列表，标记为可忽略的类型提示
            else:
                chat_history.summary = summary  # 如果没有消息内容，则设置摘要内容到聊天历史实体中
        else:
            # 如果没有找到聊天历史记录，则创建新的聊天历史实体对象并初始化相关字段
            chat_history = ChatHistoryEntity()
            chat_history.conv_uid = self.chat_seesion_id  # 设置会话 ID
            chat_history.chat_mode = once_message.chat_mode  # 设置聊天模式
            chat_history.user_name = once_message.user_name  # 设置用户名称
            chat_history.sys_code = once_message.sys_code  # 设置系统代码
            chat_history.summary = summary  # 设置摘要内容

        conversations.append(_conversation_to_dict(once_message))  # 将一次对话消息对象转换为字典并添加到对话列表中
        chat_history.messages = json.dumps(
            conversations, ensure_ascii=False
        )  # 将对话列表转换为 JSON 字符串，并设置到聊天历史记录实体中

        self.chat_history_dao.raw_update(chat_history)  # 执行原始更新操作，更新聊天历史记录实体
    # 更新聊天历史记录，将消息列表转换成 JSON 字符串并更新到数据库中
    def update(self, messages: List[OnceConversation]) -> None:
        self.chat_history_dao.update_message_by_uid(
            json.dumps(messages, ensure_ascii=False), self.chat_seesion_id
        )

    # 删除指定会话的聊天历史记录
    def delete(self) -> bool:
        self.chat_history_dao.raw_delete(self.chat_seesion_id)
        return True

    # 获取指定会话的所有消息
    def get_messages(self) -> List[Dict]:
        # 从数据库中获取指定会话的聊天历史记录对象
        chat_history = self.chat_history_dao.get_by_uid(self.chat_seesion_id)
        if chat_history:
            # 从聊天历史记录对象中获取消息内容的 JSON 字符串并解析为 Python 对象
            context = chat_history.messages
            return json.loads(context)  # type: ignore
        return []

    # 静态方法：获取最近的 20 条聊天历史记录，转换为字典列表返回
    @staticmethod
    def conv_list(
        user_name: Optional[str] = None, sys_code: Optional[str] = None
    ) -> List[Dict]:
        # 创建聊天历史记录 DAO 对象
        chat_history_dao = ChatHistoryDao()
        # 获取最近的 20 条聊天历史记录
        history_list = chat_history_dao.list_last_20(user_name, sys_code)
        result = []
        # 将每条聊天历史记录对象的属性字典添加到结果列表中
        for history in history_list:
            result.append(history.__dict__)
        return result
```