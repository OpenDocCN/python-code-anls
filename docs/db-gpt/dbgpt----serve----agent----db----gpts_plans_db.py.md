# `.\DB-GPT-src\dbgpt\serve\agent\db\gpts_plans_db.py`

```py
from datetime import datetime  # 导入datetime模块中的datetime类

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint  # 导入SQLAlchemy模块中的相关类和函数

from dbgpt.agent.core.schema import Status  # 导入dbgpt.agent.core.schema模块中的Status类
from dbgpt.storage.metadata import BaseDao, Model  # 导入dbgpt.storage.metadata模块中的BaseDao和Model类


class GptsPlansEntity(Model):
    __tablename__ = "gpts_plans"  # 数据库表名为'gpts_plans'

    id = Column(Integer, primary_key=True, comment="autoincrement id")  # 主键id，自增长

    conv_id = Column(
        String(255), nullable=False, comment="The unique id of the conversation record"
    )  # 字符串类型，长度255，不可为空，用于表示对话记录的唯一id

    sub_task_num = Column(Integer, nullable=False, comment="Subtask number")  # 整数类型，不可为空，表示子任务编号

    sub_task_title = Column(String(255), nullable=False, comment="subtask title")  # 字符串类型，长度255，不可为空，表示子任务标题

    sub_task_content = Column(Text, nullable=False, comment="subtask content")  # 文本类型，不可为空，表示子任务内容

    sub_task_agent = Column(
        String(255), nullable=True, comment="Available agents corresponding to subtasks"
    )  # 字符串类型，长度255，可为空，表示与子任务对应的可用代理人

    resource_name = Column(String(255), nullable=True, comment="resource name")  # 字符串类型，长度255，可为空，表示资源名称

    rely = Column(
        String(255), nullable=True, comment="Subtask dependencies，like: 1,2,3"
    )  # 字符串类型，长度255，可为空，表示子任务的依赖关系，如: 1,2,3

    agent_model = Column(
        String(255),
        nullable=True,
        comment="LLM model used by subtask processing agents",
    )  # 字符串类型，长度255，可为空，表示子任务处理代理所使用的LLM模型

    retry_times = Column(Integer, default=False, comment="number of retries")  # 整数类型，默认值为False，表示重试次数

    max_retry_times = Column(
        Integer, default=False, comment="Maximum number of retries"
    )  # 整数类型，默认值为False，表示最大重试次数

    state = Column(String(255), nullable=True, comment="subtask status")  # 字符串类型，长度255，可为空，表示子任务状态

    result = Column(Text(length=2**31 - 1), nullable=True, comment="subtask result")  # 文本类型，长度约2GB，可为空，表示子任务结果

    created_at = Column(DateTime, default=datetime.utcnow, comment="create time")  # 日期时间类型，默认值为当前UTC时间，表示创建时间

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="last update time",
    )  # 日期时间类型，默认值为当前UTC时间，更新时自动更新为当前UTC时间，表示最后更新时间

    __table_args__ = (UniqueConstraint("conv_id", "sub_task_num", name="uk_sub_task"),)  # 唯一约束，确保(conv_id, sub_task_num)组合唯一


class GptsPlansDao(BaseDao):
    def batch_save(self, plans: list[dict]):
        session = self.get_raw_session()  # 获取数据库会话
        session.bulk_insert_mappings(GptsPlansEntity, plans)  # 批量插入数据到GptsPlansEntity表
        session.commit()  # 提交事务
        session.close()  # 关闭会话

    def get_by_conv_id(self, conv_id: str) -> list[GptsPlansEntity]:
        session = self.get_raw_session()  # 获取数据库会话
        gpts_plans = session.query(GptsPlansEntity)  # 查询GptsPlansEntity表的查询对象
        if conv_id:
            gpts_plans = gpts_plans.filter(GptsPlansEntity.conv_id == conv_id)  # 根据conv_id过滤查询结果
        result = gpts_plans.all()  # 获取所有查询结果
        session.close()  # 关闭会话
        return result  # 返回查询结果列表

    def get_by_task_id(self, task_id: int) -> list[GptsPlansEntity]:
        session = self.get_raw_session()  # 获取数据库会话
        gpts_plans = session.query(GptsPlansEntity)  # 查询GptsPlansEntity表的查询对象
        if task_id:
            gpts_plans = gpts_plans.filter(GptsPlansEntity.id == task_id)  # 根据task_id过滤查询结果
        result = gpts_plans.first()  # 获取第一个查询结果
        session.close()  # 关闭会话
        return result  # 返回查询结果

    def get_by_conv_id_and_num(
        self, conv_id: str, task_nums: list
    ):  # 获取符合conv_id和task_nums条件的查询结果
        session = self.get_raw_session()  # 获取数据库会话
        gpts_plans = session.query(GptsPlansEntity)  # 查询GptsPlansEntity表的查询对象
        if conv_id:
            gpts_plans = gpts_plans.filter(GptsPlansEntity.conv_id == conv_id)  # 根据conv_id过滤查询结果
        if task_nums:
            gpts_plans = gpts_plans.filter(GptsPlansEntity.sub_task_num.in_(task_nums))  # 根据task_nums过滤查询结果
        result = gpts_plans.all()  # 获取所有查询结果
        session.close()  # 关闭会话
        return result  # 返回查询结果列表
    def get_plans_by_task_nums(
        self, conv_id: Optional[str], task_nums: List[int]
    ) -> list[GptsPlansEntity]:
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        # 查询所有的 GptsPlansEntity 对象
        gpts_plans = session.query(GptsPlansEntity)
        # 如果提供了 conv_id，则筛选符合该 conv_id 和 task_nums 的记录
        if conv_id:
            gpts_plans = gpts_plans.filter(GptsPlansEntity.conv_id == conv_id).filter(
                GptsPlansEntity.sub_task_num.in_(task_nums)
            )
        # 检索所有符合条件的记录
        result = gpts_plans.all()
        # 关闭数据库会话
        session.close()
        # 返回查询结果列表
        return result

    def get_todo_plans(self, conv_id: str) -> list[GptsPlansEntity]:
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        # 查询所有的 GptsPlansEntity 对象
        gpts_plans = session.query(GptsPlansEntity)
        # 如果 conv_id 为空，则返回空列表
        if not conv_id:
            return []
        # 筛选符合 conv_id 和状态为 TODO 或 RETRYING 的记录
        gpts_plans = gpts_plans.filter(GptsPlansEntity.conv_id == conv_id).filter(
            GptsPlansEntity.state.in_([Status.TODO.value, Status.RETRYING.value])
        )
        # 按 sub_task_num 排序，并检索所有符合条件的记录
        result = gpts_plans.order_by(GptsPlansEntity.sub_task_num).all()
        # 关闭数据库会话
        session.close()
        # 返回查询结果列表
        return result

    def complete_task(self, conv_id: str, task_num: int, result: str):
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        # 查询所有的 GptsPlansEntity 对象，筛选符合 conv_id 和 task_num 的记录
        gpts_plans = session.query(GptsPlansEntity).filter(
            GptsPlansEntity.conv_id == conv_id,
            GptsPlansEntity.sub_task_num == task_num
        )
        # 更新匹配的记录状态为 COMPLETE，并设置结果为给定值
        gpts_plans.update(
            {
                GptsPlansEntity.state: Status.COMPLETE.value,
                GptsPlansEntity.result: result,
            },
            synchronize_session="fetch",
        )
        # 提交事务
        session.commit()
        # 关闭数据库会话
        session.close()

    def update_task(
        self,
        conv_id: str,
        task_num: int,
        state: str,
        retry_times: int,
        agent: str = None,
        model: str = None,
        result: str = None,
    ):
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        # 查询所有的 GptsPlansEntity 对象，筛选符合 conv_id 和 task_num 的记录
        gpts_plans = session.query(GptsPlansEntity).filter(
            GptsPlansEntity.conv_id == conv_id,
            GptsPlansEntity.sub_task_num == task_num
        )
        # 准备更新的字段及其值
        update_param = {}
        update_param[GptsPlansEntity.state] = state
        update_param[GptsPlansEntity.retry_times] = retry_times
        update_param[GptsPlansEntity.result] = result
        # 如果提供了 agent，则更新 sub_task_agent 字段
        if agent:
            update_param[GptsPlansEntity.sub_task_agent] = agent
        # 如果提供了 model，则更新 agent_model 字段
        if model:
            update_param[GptsPlansEntity.agent_model] = model

        # 执行更新操作，同步会话以获取最新数据
        gpts_plans.update(update_param, synchronize_session="fetch")
        # 提交事务
        session.commit()
        # 关闭数据库会话
        session.close()

    def remove_by_conv_id(self, conv_id: str):
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        # 如果 conv_id 为 None，则抛出异常
        if conv_id is None:
            raise Exception("conv_id is None")

        # 查询所有的 GptsPlansEntity 对象，筛选符合 conv_id 的记录并删除
        gpts_plans = session.query(GptsPlansEntity)
        gpts_plans.filter(GptsPlansEntity.conv_id == conv_id).delete()
        # 提交事务
        session.commit()
        # 关闭数据库会话
        session.close()
```