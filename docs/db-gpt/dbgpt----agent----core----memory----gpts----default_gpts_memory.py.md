# `.\DB-GPT-src\dbgpt\agent\core\memory\gpts\default_gpts_memory.py`

```py
"""Default memory for storing plans and messages."""

from dataclasses import fields  # 导入dataclasses模块中的fields函数，用于获取数据类的字段信息
from typing import List, Optional  # 导入List和Optional类型提示

import pandas as pd  # 导入pandas库并使用pd作为别名

from ...schema import Status  # 从上级目录中的schema模块导入Status
from .base import GptsMessage, GptsMessageMemory, GptsPlan, GptsPlansMemory  # 从当前目录中的base模块导入GptsMessage, GptsMessageMemory, GptsPlan, GptsPlansMemory


class DefaultGptsPlansMemory(GptsPlansMemory):
    """Default memory for storing plans."""

    def __init__(self):
        """Create a memory to store plans."""
        self.df = pd.DataFrame(columns=[field.name for field in fields(GptsPlan)])
        # 初始化一个空的DataFrame，列名为GptsPlan数据类中各字段的名称

    def batch_save(self, plans: list[GptsPlan]):
        """Save plans in batch."""
        new_rows = pd.DataFrame([item.to_dict() for item in plans])
        # 将传入的plans列表中的每个GptsPlan对象转换为字典，并创建一个新的DataFrame
        self.df = pd.concat([self.df, new_rows], ignore_index=True)
        # 将新创建的DataFrame与现有的self.df合并，并忽略索引，更新self.df

    def get_by_conv_id(self, conv_id: str) -> List[GptsPlan]:
        """Get plans by conv_id."""
        result = self.df.query(f"conv_id==@conv_id")  # noqa: F541
        # 从self.df中查询符合conv_id条件的行，并将结果存储在result中，禁用F541警告
        plans = []
        for row in result.itertuples(index=False, name=None):
            row_dict = dict(zip(self.df.columns, row))
            plans.append(GptsPlan.from_dict(row_dict))
        # 遍历查询结果中的每一行，将其转换为字典，再使用GptsPlan.from_dict方法转换为GptsPlan对象，并添加到plans列表中
        return plans  # 返回符合条件的GptsPlan对象列表

    def get_by_conv_id_and_num(
        self, conv_id: str, task_nums: List[int]
    ) -> List[GptsPlan]:
        """Get plans by conv_id and task number."""
        task_nums_int = [int(num) for num in task_nums]  # noqa:F841
        # 将task_nums列表中的每个元素转换为整数类型，禁用F841警告
        result = self.df.query(
            f"conv_id==@conv_id and sub_task_num in @task_nums_int"  # noqa
        )
        # 从self.df中查询符合conv_id和task_nums_int条件的行，并将结果存储在result中，禁用F841和F541警告
        plans = []
        for row in result.itertuples(index=False, name=None):
            row_dict = dict(zip(self.df.columns, row))
            plans.append(GptsPlan.from_dict(row_dict))
        # 遍历查询结果中的每一行，将其转换为字典，再使用GptsPlan.from_dict方法转换为GptsPlan对象，并添加到plans列表中
        return plans  # 返回符合条件的GptsPlan对象列表

    def get_todo_plans(self, conv_id: str) -> List[GptsPlan]:
        """Get unfinished planning steps."""
        todo_states = [Status.TODO.value, Status.RETRYING.value]  # noqa: F841
        # 定义一个包含未完成状态的列表，禁用F841警告
        result = self.df.query(f"conv_id==@conv_id and state in @todo_states")  # noqa
        # 从self.df中查询符合conv_id和todo_states条件的行，并将结果存储在result中，禁用F841和F541警告
        plans = []
        for row in result.itertuples(index=False, name=None):
            row_dict = dict(zip(self.df.columns, row))
            plans.append(GptsPlan.from_dict(row_dict))
        # 遍历查询结果中的每一行，将其转换为字典，再使用GptsPlan.from_dict方法转换为GptsPlan对象，并添加到plans列表中
        return plans  # 返回符合条件的GptsPlan对象列表

    def complete_task(self, conv_id: str, task_num: int, result: str):
        """Set the planning step to complete."""
        condition = (self.df["conv_id"] == conv_id) & (
            self.df["sub_task_num"] == task_num
        )
        # 定义筛选条件，选择conv_id和task_num匹配的行
        self.df.loc[condition, "state"] = Status.COMPLETE.value
        # 将匹配的行的state列设置为完成状态的值
        self.df.loc[condition, "result"] = result
        # 将匹配的行的result列设置为给定的结果值

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
        """Update the planning task."""
        # 更新计划任务的方法
        pass
        # 空方法，暂未实现
    ):
        """Update the state of the planning step."""
        # 定义一个方法，用于更新计划步骤的状态

        # 根据会话 ID (conv_id) 和子任务编号 (task_num) 确定更新条件
        condition = (self.df["conv_id"] == conv_id) & (
            self.df["sub_task_num"] == task_num
        )

        # 更新数据框中符合条件的记录的状态 (state)、重试次数 (retry_times) 和结果 (result)
        self.df.loc[condition, "state"] = state
        self.df.loc[condition, "retry_times"] = retry_times
        self.df.loc[condition, "result"] = result

        # 如果有指定代理 (agent)，更新相应的子任务代理信息
        if agent:
            self.df.loc[condition, "sub_task_agent"] = agent

        # 如果有指定模型 (model)，更新代理模型信息
        if model:
            self.df.loc[condition, "agent_model"] = model

    def remove_by_conv_id(self, conv_id: str):
        """Remove all plans in the conversation."""
        # 根据会话 ID (conv_id) 删除数据框中所有相关的计划记录
        self.df.drop(self.df[self.df["conv_id"] == conv_id].index, inplace=True)
class DefaultGptsMessageMemory(GptsMessageMemory):
    """Default memory for storing messages."""

    def __init__(self):
        """Create a memory to store messages."""
        # 创建一个空的 Pandas DataFrame 来存储消息
        self.df = pd.DataFrame(columns=[field.name for field in fields(GptsMessage)])

    def append(self, message: GptsMessage):
        """Append a message to the memory."""
        # 将消息对象转换为字典并添加到 DataFrame 的末尾
        self.df.loc[len(self.df)] = message.to_dict()

    def get_by_agent(self, conv_id: str, agent: str) -> Optional[List[GptsMessage]]:
        """Get all messages sent or received by the agent in the conversation."""
        # 查询指定会话中由指定代理人发送或接收的所有消息
        result = self.df.query(
            f"conv_id==@conv_id and (sender==@agent or receiver==@agent)"  # noqa: F541
        )
        messages = []
        for row in result.itertuples(index=False, name=None):
            row_dict = dict(zip(self.df.columns, row))
            # 将 DataFrame 行转换为消息对象并添加到列表中
            messages.append(GptsMessage.from_dict(row_dict))
        return messages

    def get_between_agents(
        self,
        conv_id: str,
        agent1: str,
        agent2: str,
        current_goal: Optional[str] = None,
    ) -> List[GptsMessage]:
        """Get all messages between two agents in the conversation."""
        if current_goal:
            # 查询指定会话中由两个指定代理人之间互相发送或接收的消息，并且满足当前目标条件
            result = self.df.query(
                f"conv_id==@conv_id and ((sender==@agent1 and receiver==@agent2) or (sender==@agent2 and receiver==@agent1)) and current_goal==@current_goal"  # noqa
            )
        else:
            # 查询指定会话中由两个指定代理人之间互相发送或接收的消息
            result = self.df.query(
                f"conv_id==@conv_id and ((sender==@agent1 and receiver==@agent2) or (sender==@agent2 and receiver==@agent1))"  # noqa
            )
        messages = []
        for row in result.itertuples(index=False, name=None):
            row_dict = dict(zip(self.df.columns, row))
            # 将 DataFrame 行转换为消息对象并添加到列表中
            messages.append(GptsMessage.from_dict(row_dict))
        return messages

    def get_by_conv_id(self, conv_id: str) -> List[GptsMessage]:
        """Get all messages in the conversation."""
        # 查询指定会话中的所有消息
        result = self.df.query(f"conv_id==@conv_id")  # noqa: F541
        messages = []
        for row in result.itertuples(index=False, name=None):
            row_dict = dict(zip(self.df.columns, row))
            # 将 DataFrame 行转换为消息对象并添加到列表中
            messages.append(GptsMessage.from_dict(row_dict))
        return messages

    def get_last_message(self, conv_id: str) -> Optional[GptsMessage]:
        """Get the last message in the conversation."""
        # 暂未实现获取会话中最后一条消息的功能
        return None
```