# `.\DB-GPT-src\dbgpt\serve\agent\db\gpts_messages_db.py`

```py
from datetime import datetime  # 导入datetime模块中的datetime类，用于处理日期时间
from typing import List, Optional  # 导入类型提示相关的模块

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, and_, desc, or_  # 导入SQLAlchemy相关的类和函数

from dbgpt.storage.metadata import BaseDao, Model  # 导入自定义的数据库访问类和数据模型


class GptsMessagesEntity(Model):
    __tablename__ = "gpts_messages"  # 设置表名为"gpts_messages"

    id = Column(Integer, primary_key=True, comment="autoincrement id")  # 主键，自增长的整数id

    conv_id = Column(
        String(255), nullable=False, comment="The unique id of the conversation record"
    )  # 字符串，255字符长度，不可为空，对话记录的唯一标识

    sender = Column(
        String(255),
        nullable=False,
        comment="Who speaking in the current conversation turn",
    )  # 字符串，255字符长度，不可为空，当前对话轮次中发言者

    receiver = Column(
        String(255),
        nullable=False,
        comment="Who receive message in the current conversation turn",
    )  # 字符串，255字符长度，不可为空，当前对话轮次中接收消息者

    model_name = Column(String(255), nullable=True, comment="message generate model")  # 字符串，255字符长度，可为空，生成消息的模型

    rounds = Column(Integer, nullable=False, comment="dialogue turns")  # 整数，不可为空，对话轮次

    content = Column(
        Text(length=2**31 - 1), nullable=True, comment="Content of the speech"
    )  # 文本类型，长度较大，可为空，发言内容

    current_goal = Column(
        Text, nullable=True, comment="The target corresponding to the current message"
    )  # 文本类型，可为空，当前消息对应的目标

    context = Column(Text, nullable=True, comment="Current conversation context")  # 文本类型，可为空，当前对话上下文

    review_info = Column(
        Text, nullable=True, comment="Current conversation review info"
    )  # 文本类型，可为空，当前对话的审核信息

    action_report = Column(
        Text(length=2**31 - 1),
        nullable=True,
        comment="Current conversation action report",
    )  # 文本类型，较大长度，可为空，当前对话的行动报告

    role = Column(
        String(255), nullable=True, comment="The role of the current message content"
    )  # 字符串，255字符长度，可为空，当前消息内容的角色

    created_at = Column(DateTime, default=datetime.utcnow, comment="create time")  # 日期时间类型，默认为当前UTC时间，表示创建时间

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="last update time",
    )  # 日期时间类型，更新时自动更新为当前UTC时间，表示最后更新时间

    __table_args__ = (Index("idx_q_messages", "conv_id", "rounds", "sender"),)  # 定义表的索引，加速查询


class GptsMessagesDao(BaseDao):
    def append(self, entity: dict):
        session = self.get_raw_session()  # 获取数据库会话
        message = GptsMessagesEntity(  # 创建GptsMessagesEntity对象，用于插入数据库
            conv_id=entity.get("conv_id"),  # 对话ID
            sender=entity.get("sender"),  # 发送者
            receiver=entity.get("receiver"),  # 接收者
            content=entity.get("content"),  # 内容
            role=entity.get("role", None),  # 角色，默认为空
            model_name=entity.get("model_name", None),  # 模型名称，默认为空
            context=entity.get("context", None),  # 对话上下文，默认为空
            rounds=entity.get("rounds", None),  # 对话轮次，默认为空
            current_goal=entity.get("current_goal", None),  # 当前目标，默认为空
            review_info=entity.get("review_info", None),  # 审核信息，默认为空
            action_report=entity.get("action_report", None),  # 行动报告，默认为空
        )
        session.add(message)  # 将消息对象添加到会话中
        session.commit()  # 提交会话，保存数据到数据库
        id = message.id  # 获取插入后的自增长ID
        session.close()  # 关闭会话
        return id  # 返回插入数据的ID

    def get_by_agent(
        self, conv_id: str, agent: str
    # 获取一个原始数据库会话
    session = self.get_raw_session()
    # 从 GptsMessagesEntity 表中查询所有记录
    gpts_messages = session.query(GptsMessagesEntity)
    # 如果指定了代理者(agent)，则筛选会话ID为 conv_id，并且发送者或接收者为代理者的消息记录
    if agent:
        gpts_messages = gpts_messages.filter(
            GptsMessagesEntity.conv_id == conv_id
        ).filter(
            or_(
                GptsMessagesEntity.sender == agent,
                GptsMessagesEntity.receiver == agent,
            )
        )
    # 按照消息轮次 rounds 排序所有记录，并将结果返回为列表
    result = gpts_messages.order_by(GptsMessagesEntity.rounds).all()
    # 关闭数据库会话
    session.close()
    # 返回查询结果列表
    return result

# 根据会话ID获取所有与该会话相关的消息记录
def get_by_conv_id(self, conv_id: str) -> Optional[List[GptsMessagesEntity]]:
    # 获取一个原始数据库会话
    session = self.get_raw_session()
    # 从 GptsMessagesEntity 表中查询所有记录
    gpts_messages = session.query(GptsMessagesEntity)
    # 如果指定了会话ID，则筛选会话ID为 conv_id 的消息记录
    if conv_id:
        gpts_messages = gpts_messages.filter(GptsMessagesEntity.conv_id == conv_id)
    # 按照消息轮次 rounds 排序所有记录，并将结果返回为列表
    result = gpts_messages.order_by(GptsMessagesEntity.rounds).all()
    # 关闭数据库会话
    session.close()
    # 返回查询结果列表
    return result

# 根据会话ID、两个代理者以及可选的当前目标获取消息记录
def get_between_agents(
    self,
    conv_id: str,
    agent1: str,
    agent2: str,
    current_goal: Optional[str] = None,
) -> Optional[List[GptsMessagesEntity]]:
    # 获取一个原始数据库会话
    session = self.get_raw_session()
    # 从 GptsMessagesEntity 表中查询所有记录
    gpts_messages = session.query(GptsMessagesEntity)
    # 如果指定了两个代理者(agent1 和 agent2)，则筛选会话ID为 conv_id，发送者和接收者分别为两个代理者的消息记录
    if agent1 and agent2:
        gpts_messages = gpts_messages.filter(
            GptsMessagesEntity.conv_id == conv_id
        ).filter(
            or_(
                and_(
                    GptsMessagesEntity.sender == agent1,
                    GptsMessagesEntity.receiver == agent2,
                ),
                and_(
                    GptsMessagesEntity.sender == agent2,
                    GptsMessagesEntity.receiver == agent1,
                ),
            )
        )
    # 如果指定了当前目标(current_goal)，则筛选当前目标为 current_goal 的消息记录
    if current_goal:
        gpts_messages = gpts_messages.filter(
            GptsMessagesEntity.current_goal == current_goal
        )
    # 按照消息轮次 rounds 排序所有记录，并将结果返回为列表
    result = gpts_messages.order_by(GptsMessagesEntity.rounds).all()
    # 关闭数据库会话
    session.close()
    # 返回查询结果列表
    return result

# 根据会话ID获取最后一条消息记录
def get_last_message(self, conv_id: str) -> Optional[GptsMessagesEntity]:
    # 获取一个原始数据库会话
    session = self.get_raw_session()
    # 从 GptsMessagesEntity 表中查询所有记录
    gpts_messages = session.query(GptsMessagesEntity)
    # 如果指定了会话ID，则筛选会话ID为 conv_id 的消息记录，并按照消息轮次 rounds 降序排序
    if conv_id:
        gpts_messages = gpts_messages.filter(
            GptsMessagesEntity.conv_id == conv_id
        ).order_by(desc(GptsMessagesEntity.rounds))

    # 获取排序后的第一条记录作为结果
    result = gpts_messages.first()
    # 关闭数据库会话
    session.close()
    # 返回查询结果
    return result
```