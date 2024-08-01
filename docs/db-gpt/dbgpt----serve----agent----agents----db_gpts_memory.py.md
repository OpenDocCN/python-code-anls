# `.\DB-GPT-src\dbgpt\serve\agent\agents\db_gpts_memory.py`

```py
from typing import List, Optional

from dbgpt.agent.core.memory.gpts.base import GptsPlan  # 导入 GptsPlan 类
from dbgpt.agent.core.memory.gpts.gpts_memory import (  # 导入 GptsMessageMemory 和 GptsPlansMemory 类
    GptsMessage,
    GptsMessageMemory,
    GptsPlansMemory,
)

from ..db.gpts_messages_db import GptsMessagesDao  # 导入 GptsMessagesDao 类
from ..db.gpts_plans_db import GptsPlansDao, GptsPlansEntity  # 导入 GptsPlansDao 和 GptsPlansEntity 类


class MetaDbGptsPlansMemory(GptsPlansMemory):
    def __init__(self):
        self.gpts_plan = GptsPlansDao()  # 初始化 gpts_plan 属性为 GptsPlansDao 的实例

    def batch_save(self, plans: List[GptsPlan]):
        self.gpts_plan.batch_save([item.to_dict() for item in plans])  # 批量保存计划到数据库

    def get_by_conv_id(self, conv_id: str) -> List[GptsPlan]:
        db_results: List[GptsPlansEntity] = self.gpts_plan.get_by_conv_id(
            conv_id=conv_id
        )  # 根据对话 ID 从数据库获取计划信息
        results = []
        for item in db_results:
            results.append(GptsPlan.from_dict(item.__dict__))  # 将数据库实体转换为 GptsPlan 对象并添加到结果列表
        return results  # 返回查询结果的列表

    def get_by_conv_id_and_num(
        self, conv_id: str, task_nums: List[int]
    ) -> List[GptsPlan]:
        db_results: List[GptsPlansEntity] = self.gpts_plan.get_by_conv_id_and_num(
            conv_id=conv_id, task_nums=task_nums
        )  # 根据对话 ID 和任务号列表从数据库获取计划信息
        results = []
        for item in db_results:
            results.append(GptsPlan.from_dict(item.__dict__))  # 将数据库实体转换为 GptsPlan 对象并添加到结果列表
        return results  # 返回查询结果的列表

    def get_todo_plans(self, conv_id: str) -> List[GptsPlan]:
        db_results: List[GptsPlansEntity] = self.gpts_plan.get_todo_plans(
            conv_id=conv_id
        )  # 获取待办计划列表
        results = []
        for item in db_results:
            results.append(GptsPlan.from_dict(item.__dict__))  # 将数据库实体转换为 GptsPlan 对象并添加到结果列表
        return results  # 返回待办计划的列表

    def complete_task(self, conv_id: str, task_num: int, result: str):
        self.gpts_plan.complete_task(conv_id=conv_id, task_num=task_num, result=result)  # 标记指定对话 ID 和任务号的任务为已完成

    def update_task(
        self,
        conv_id: str,
        task_num: int,
        state: str,
        retry_times: int,
        agent: Optional[str] = None,
        model: Optional[str] = None,
        result: Optional[str] = None,
    ):
        self.gpts_plan.update_task(
            conv_id=conv_id,
            task_num=task_num,
            state=state,
            retry_times=retry_times,
            agent=agent,
            model=model,
            result=result,
        )  # 更新指定对话 ID 和任务号的任务状态及相关信息

    def remove_by_conv_id(self, conv_id: str):
        self.gpts_plan.remove_by_conv_id(conv_id=conv_id)  # 删除指定对话 ID 相关的所有计划


class MetaDbGptsMessageMemory(GptsMessageMemory):
    def __init__(self):
        self.gpts_message = GptsMessagesDao()  # 初始化 gpts_message 属性为 GptsMessagesDao 的实例

    def append(self, message: GptsMessage):
        self.gpts_message.append(message.to_dict())  # 将消息对象转换为字典并追加到数据库中

    def get_by_agent(self, conv_id: str, agent: str) -> Optional[List[GptsMessage]]:
        db_results = self.gpts_message.get_by_agent(conv_id, agent)  # 根据对话 ID 和代理名从数据库获取消息列表
        results = []
        db_results = sorted(db_results, key=lambda x: x.rounds)
        for item in db_results:
            results.append(GptsMessage.from_dict(item.__dict__))  # 将数据库实体转换为 GptsMessage 对象并添加到结果列表
        return results  # 返回查询结果的列表
    # 获取两个代理之间的消息记录，根据会话 ID、代理1、代理2和当前目标（可选），从数据库中检索消息记录
    def get_between_agents(
        self,
        conv_id: str,
        agent1: str,
        agent2: str,
        current_goal: Optional[str] = None,
    ) -> List[GptsMessage]:
        # 从数据库中获取两个代理之间的消息记录
        db_results = self.gpts_message.get_between_agents(
            conv_id, agent1, agent2, current_goal
        )
        # 初始化结果列表
        results = []
        # 按消息轮次（rounds）对数据库结果进行排序
        db_results = sorted(db_results, key=lambda x: x.rounds)
        # 将每个数据库结果转换为 GptsMessage 对象并添加到结果列表中
        for item in db_results:
            results.append(GptsMessage.from_dict(item.__dict__))
        return results

    # 根据会话 ID 获取相关的消息记录列表
    def get_by_conv_id(self, conv_id: str) -> List[GptsMessage]:
        # 从数据库中获取指定会话 ID 相关的消息记录
        db_results = self.gpts_message.get_by_conv_id(conv_id)
        # 初始化结果列表
        results = []
        # 按消息轮次（rounds）对数据库结果进行排序
        db_results = sorted(db_results, key=lambda x: x.rounds)
        # 将每个数据库结果转换为 GptsMessage 对象并添加到结果列表中
        for item in db_results:
            results.append(GptsMessage.from_dict(item.__dict__))
        return results

    # 获取指定会话 ID 的最后一条消息记录
    def get_last_message(self, conv_id: str) -> Optional[GptsMessage]:
        # 从数据库中获取指定会话 ID 的最后一条消息记录
        db_result = self.gpts_message.get_last_message(conv_id)
        # 如果存在数据库结果，则将其转换为 GptsMessage 对象并返回；否则返回 None
        if db_result:
            return GptsMessage.from_dict(db_result.__dict__)
        else:
            return None
```