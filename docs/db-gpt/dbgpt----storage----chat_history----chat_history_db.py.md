# `.\DB-GPT-src\dbgpt\storage\chat_history\chat_history_db.py`

```py
"""Chat history database model."""
# 导入必要的模块和类
from datetime import datetime
from typing import Optional

# 导入 SQLAlchemy 相关的模块和类
from sqlalchemy import Column, DateTime, Index, Integer, String, Text, UniqueConstraint

# 导入自定义的模块
from ..metadata import BaseDao, Model


class ChatHistoryEntity(Model):
    """Chat history entity."""

    # 定义数据表名称
    __tablename__ = "chat_history"
    # 定义表格参数，设置唯一约束
    __table_args__ = (UniqueConstraint("conv_uid", name="uk_conv_uid"),)
    
    # 主键字段，自增长的整数类型
    id = Column(
        Integer, primary_key=True, autoincrement=True, comment="autoincrement id"
    )
    # 对话记录唯一标识，字符串类型，不可为空
    conv_uid = Column(
        String(255),
        unique=False,  # 修改为 False 时，alembic 迁移将失败，因此用 UniqueConstraint 替代
        nullable=False,
        comment="Conversation record unique id",
    )
    # 对话模式，字符串类型，不可为空
    chat_mode = Column(String(255), nullable=False, comment="Conversation scene mode")
    # 对话记录摘要，文本类型，不可为空
    summary = Column(
        Text(length=2**31 - 1), nullable=False, comment="Conversation record summary"
    )
    # 对话者名称，字符串类型，可为空
    user_name = Column(String(255), nullable=True, comment="interlocutor")
    # 对话消息内容，文本类型，可为空
    messages = Column(
        Text(length=2**31 - 1), nullable=True, comment="Conversation details"
    )
    # 消息 ID，逗号分隔的文本类型，可为空
    message_ids = Column(
        Text(length=2**31 - 1), nullable=True, comment="Message ids, split by comma"
    )
    # 系统代码，字符串类型，可为空，创建索引
    sys_code = Column(String(128), index=True, nullable=True, comment="System code")
    # 记录创建时间，日期时间类型，默认为当前时间
    gmt_created = Column(DateTime, default=datetime.now, comment="Record creation time")
    # 记录更新时间，日期时间类型，默认为当前时间
    gmt_modified = Column(DateTime, default=datetime.now, comment="Record update time")

    # 创建索引，用于查询用户名称
    Index("idx_q_user", "user_name")
    # 创建索引，用于查询对话模式
    Index("idx_q_mode", "chat_mode")
    # 创建索引，用于查询对话摘要
    Index("idx_q_conv", "summary")


class ChatHistoryMessageEntity(Model):
    """Chat history message entity."""

    # 定义数据表名称
    __tablename__ = "chat_history_message"
    # 定义表格参数，设置唯一约束
    __table_args__ = (
        UniqueConstraint("conv_uid", "index", name="uk_conversation_message"),
    )
    
    # 主键字段，自增长的整数类型
    id = Column(
        Integer, primary_key=True, autoincrement=True, comment="autoincrement id"
    )
    # 对话记录唯一标识，字符串类型，不可为空
    conv_uid = Column(
        String(255),
        unique=False,
        nullable=False,
        comment="Conversation record unique id",
    )
    # 消息索引，整数类型，不可为空
    index = Column(Integer, nullable=False, comment="Message index")
    # 消息轮次索引，整数类型，不可为空
    round_index = Column(Integer, nullable=False, comment="Message round index")
    # 消息详细内容，文本类型，可为空，以 JSON 格式存储
    message_detail = Column(
        Text(length=2**31 - 1), nullable=True, comment="Message details, json format"
    )
    # 记录创建时间，日期时间类型，默认为当前时间
    gmt_created = Column(DateTime, default=datetime.now, comment="Record creation time")
    # 记录更新时间，日期时间类型，默认为当前时间
    gmt_modified = Column(DateTime, default=datetime.now, comment="Record update time")


class ChatHistoryDao(BaseDao):
    """Chat history dao."""

    # 获取最近的 20 条对话记录
    def list_last_20(
        self, user_name: Optional[str] = None, sys_code: Optional[str] = None
    ):
    ):
        """Retrieve the last 20 chat history records."""
        # 获取原始数据库会话
        session = self.get_raw_session()
        # 查询所有的聊天历史记录
        chat_history = session.query(ChatHistoryEntity)
        # 如果提供了用户名，筛选该用户的聊天记录
        if user_name:
            chat_history = chat_history.filter(ChatHistoryEntity.user_name == user_name)
        # 如果提供了系统代码，筛选该系统的聊天记录
        if sys_code:
            chat_history = chat_history.filter(ChatHistoryEntity.sys_code == sys_code)

        # 按照记录ID倒序排序
        chat_history = chat_history.order_by(ChatHistoryEntity.id.desc())

        # 获取最多20条记录
        result = chat_history.limit(20).all()
        # 关闭数据库会话
        session.close()
        return result

    def raw_update(self, entity: ChatHistoryEntity):
        """Update the chat history record."""
        # 获取原始数据库会话
        session = self.get_raw_session()
        try:
            # 合并给定的聊天历史记录实体
            updated = session.merge(entity)
            # 提交事务
            session.commit()
            return updated.id
        finally:
            # 关闭数据库会话
            session.close()

    def update_message_by_uid(self, message: str, conv_uid: str):
        """Update the chat history record."""
        # 获取原始数据库会话
        session = self.get_raw_session()
        try:
            # 查询特定会话UID的聊天历史记录
            chat_history = session.query(ChatHistoryEntity)
            chat_history = chat_history.filter(ChatHistoryEntity.conv_uid == conv_uid)
            # 更新该会话的消息内容
            updated = chat_history.update({ChatHistoryEntity.messages: message})
            # 提交事务
            session.commit()
            return updated
        finally:
            # 关闭数据库会话
            session.close()

    def raw_delete(self, conv_uid: str):
        """Delete the chat history record."""
        # 如果会话UID为空，抛出异常
        if conv_uid is None:
            raise Exception("conv_uid is None")
        # 使用上下文管理器获取数据库会话
        with self.session() as session:
            # 查询特定会话UID的聊天历史记录
            chat_history = session.query(ChatHistoryEntity)
            chat_history = chat_history.filter(ChatHistoryEntity.conv_uid == conv_uid)
            # 删除该会话的所有记录
            chat_history.delete()

    def get_by_uid(self, conv_uid: str) -> Optional[ChatHistoryEntity]:
        """Retrieve the chat history record by conv_uid."""
        # 使用上下文管理器获取数据库会话，不自动提交
        with self.session(commit=False) as session:
            # 查询特定会话UID的聊天历史记录，返回第一条记录
            return session.query(ChatHistoryEntity).filter_by(conv_uid=conv_uid).first()
```