# `.\AutoGPT\autogpts\forge\forge\db.py`

```py
# 导入必要的模块
import datetime
import uuid

from sqlalchemy import Column, DateTime, String
from sqlalchemy.exc import SQLAlchemyError

# 导入自定义模块
from .sdk import AgentDB, Base, ForgeLogger, NotFoundError

# 创建日志对象
LOG = ForgeLogger(__name__)

# 定义 ChatModel 数据库表结构
class ChatModel(Base):
    __tablename__ = "chat"
    msg_id = Column(String, primary_key=True, index=True)  # 消息ID
    task_id = Column(String)  # 任务ID
    role = Column(String)  # 角色
    content = Column(String)  # 内容
    created_at = Column(DateTime, default=datetime.datetime.utcnow)  # 创建时间
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )  # 修改时间

# 定义 ActionModel 数据库表结构
class ActionModel(Base):
    __tablename__ = "action"
    action_id = Column(String, primary_key=True, index=True)  # 动作ID
    task_id = Column(String)  # 任务ID
    name = Column(String)  # 名称
    args = Column(String)  # 参数
    created_at = Column(DateTime, default=datetime.datetime.utcnow)  # 创建时间
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )  # 修改时间

# 定义 ForgeDatabase 类，继承自 AgentDB
class ForgeDatabase(AgentDB):
    # 异步方法，用于添加聊天记录
    async def add_chat_history(self, task_id, messages):
        # 遍历消息列表
        for message in messages:
            # 调用方法添加聊天消息
            await self.add_chat_message(task_id, message["role"], message["content"])
    # 异步方法，用于向聊天消息中添加新消息
    async def add_chat_message(self, task_id, role, content):
        # 如果调试模式开启，则记录日志
        if self.debug_enabled:
            LOG.debug("Creating new task")
        try:
            # 使用会话上下文管理器创建数据库会话
            with self.Session() as session:
                # 创建新的聊天消息对象
                mew_msg = ChatModel(
                    msg_id=str(uuid.uuid4()),
                    task_id=task_id,
                    role=role,
                    content=content,
                )
                # 将新消息添加到数据库会话中
                session.add(mew_msg)
                # 提交会话中的所有更改
                session.commit()
                # 刷新新消息对象，以获取数据库中的最新数据
                session.refresh(mew_msg)
                # 如果调试模式开启，则记录创建新聊天消息的日志
                if self.debug_enabled:
                    LOG.debug(
                        f"Created new Chat message with task_id: {mew_msg.msg_id}"
                    )
                # 返回新创建的聊天消息对象
                return mew_msg
        except SQLAlchemyError as e:
            # 如果发生 SQLAlchemy 错误，则记录错误日志并抛出异常
            LOG.error(f"SQLAlchemy error while creating task: {e}")
            raise
        except NotFoundError as e:
            # 如果发生 NotFoundError 异常，则直接抛出异常
            raise
        except Exception as e:
            # 如果发生意外异常，则记录错误日志并抛出异常
            LOG.error(f"Unexpected error while creating task: {e}")
            raise
    # 异步方法，用于获取与给定任务ID相关的聊天记录
    async def get_chat_history(self, task_id):
        # 如果启用了调试模式，则记录获取聊天记录的任务ID
        if self.debug_enabled:
            LOG.debug(f"Getting chat history with task_id: {task_id}")
        try:
            # 使用会话对象查询数据库中与给定任务ID相关的聊天记录
            with self.Session() as session:
                # 如果找到了聊天记录
                if messages := (
                    session.query(ChatModel)
                    .filter(ChatModel.task_id == task_id)
                    .order_by(ChatModel.created_at)
                    .all()
                ):
                    # 返回聊天记录的角色和内容组成的列表
                    return [{"role": m.role, "content": m.content} for m in messages]

                else:
                    # 如果未找到聊天记录，则记录错误并抛出异常
                    LOG.error(f"Chat history not found with task_id: {task_id}")
                    raise NotFoundError("Chat history not found")
        except SQLAlchemyError as e:
            # 记录SQLAlchemy错误并抛出异常
            LOG.error(f"SQLAlchemy error while getting chat history: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            # 记录未预期的错误并抛出异常
            LOG.error(f"Unexpected error while getting chat history: {e}")
            raise

    # 创建新的动作记录
    async def create_action(self, task_id, name, args):
        try:
            # 使用会话对象创建新的动作记录
            with self.Session() as session:
                new_action = ActionModel(
                    action_id=str(uuid.uuid4()),
                    task_id=task_id,
                    name=name,
                    args=str(args),
                )
                session.add(new_action)
                session.commit()
                session.refresh(new_action)
                if self.debug_enabled:
                    # 如果启用了调试模式，则记录创建动作记录的任务ID
                    LOG.debug(
                        f"Created new Action with task_id: {new_action.action_id}"
                    )
                return new_action
        except SQLAlchemyError as e:
            # 记录SQLAlchemy错误并抛出异常
            LOG.error(f"SQLAlchemy error while creating action: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            # 记录未预期的错误并抛出异常
            LOG.error(f"Unexpected error while creating action: {e}")
            raise
    # 异步方法，用于获取指定任务ID的操作历史记录
    async def get_action_history(self, task_id):
        # 如果启用了调试模式，则记录获取操作历史记录的任务ID
        if self.debug_enabled:
            LOG.debug(f"Getting action history with task_id: {task_id}")
        try:
            # 使用会话对象查询数据库中符合条件的操作记录
            with self.Session() as session:
                if actions := (
                    session.query(ActionModel)
                    .filter(ActionModel.task_id == task_id)
                    .order_by(ActionModel.created_at)
                    .all()
                ):
                    # 将查询到的操作记录转换为包含名称和参数的字典列表并返回
                    return [{"name": a.name, "args": a.args} for a in actions]

                else:
                    # 如果未找到符合条件的操作记录，则记录错误并抛出异常
                    LOG.error(f"Action history not found with task_id: {task_id}")
                    raise NotFoundError("Action history not found")
        except SQLAlchemyError as e:
            # 捕获SQLAlchemy错误并记录错误信息，然后重新抛出异常
            LOG.error(f"SQLAlchemy error while getting action history: {e}")
            raise
        except NotFoundError as e:
            # 捕获自定义的未找到错误并重新抛出异常
            raise
        except Exception as e:
            # 捕获其他异常并记录错误信息，然后重新抛出异常
            LOG.error(f"Unexpected error while getting action history: {e}")
            raise
```