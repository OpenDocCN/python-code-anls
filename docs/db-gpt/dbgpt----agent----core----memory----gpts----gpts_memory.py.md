# `.\DB-GPT-src\dbgpt\agent\core\memory\gpts\gpts_memory.py`

```py
"""GPTs memory."""

import json
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

from dbgpt.vis.client import VisAgentMessages, VisAgentPlans, vis_client

from ...action.base import ActionOutput
from .base import GptsMessage, GptsMessageMemory, GptsPlansMemory
from .default_gpts_memory import DefaultGptsMessageMemory, DefaultGptsPlansMemory

NONE_GOAL_PREFIX: str = "none_goal_count_"


class GptsMemory:
    """GPTs memory."""

    def __init__(
        self,
        plans_memory: Optional[GptsPlansMemory] = None,
        message_memory: Optional[GptsMessageMemory] = None,
    ):
        """Create a memory to store plans and messages."""
        # 初始化计划和消息存储器
        self._plans_memory: GptsPlansMemory = (
            plans_memory if plans_memory is not None else DefaultGptsPlansMemory()
        )
        self._message_memory: GptsMessageMemory = (
            message_memory if message_memory is not None else DefaultGptsMessageMemory()
        )

    @property
    def plans_memory(self) -> GptsPlansMemory:
        """Return the plans memory."""
        return self._plans_memory

    @property
    def message_memory(self) -> GptsMessageMemory:
        """Return the message memory."""
        return self._message_memory

    async def _message_group_vis_build(self, message_group):
        """Build visualization items based on message group."""
        # 如果消息组为空，则返回空字符串
        if not message_group:
            return ""
        
        # 初始化计数器和获取最后一个目标
        num: int = 0
        last_goal = next(reversed(message_group))
        last_goal_messages = message_group[last_goal]

        # 获取最后一个目标的最后一条消息
        last_goal_message = last_goal_messages[-1]
        vis_items = []

        plan_temps = []
        for key, value in message_group.items():
            num = num + 1
            # 如果键以指定前缀开头，处理为计划可视化项目
            if key.startswith(NONE_GOAL_PREFIX):
                vis_items.append(await self._messages_to_plan_vis(plan_temps))
                plan_temps = []
                num = 0
                vis_items.append(await self._messages_to_agents_vis(value))
            else:
                num += 1
                # 将消息转换为计划可视化条目并添加到临时列表中
                plan_temps.append(
                    {
                        "name": key,
                        "num": num,
                        "status": "complete",
                        "agent": value[0].receiver if value else "",
                        "markdown": await self._messages_to_agents_vis(value),
                    }
                )

        # 处理剩余的计划临时条目
        if len(plan_temps) > 0:
            vis_items.append(await self._messages_to_plan_vis(plan_temps))
        # 将最后一个目标的最后一条消息转换为代理可视化项目并添加到列表中
        vis_items.append(await self._messages_to_agents_vis([last_goal_message]))
        # 将所有可视化项目以换行符连接成字符串并返回
        return "\n".join(vis_items)
    # 异步方法，用于构建计划可视化
    async def _plan_vis_build(self, plan_group: dict[str, list]):
        # 初始化计数器
        num: int = 0
        # 初始化计划项列表
        plan_items = []
        # 遍历计划组中的每个键值对
        for key, value in plan_group.items():
            # 更新计数器
            num = num + 1
            # 构建计划项字典，并添加到计划项列表中
            plan_items.append(
                {
                    "name": key,
                    "num": num,
                    "status": "complete",
                    "agent": value[0].receiver if value else "",  # 根据值获取接收者信息，若值不存在则为空字符串
                    "markdown": await self._messages_to_agents_vis(value),  # 将消息转换为代理可视化的Markdown格式
                }
            )
        # 返回计划项列表的可视化结果
        return await self._messages_to_plan_vis(plan_items)

    # 生成对话的可视化，异步方法
    async def one_chat_completions_v2(self, conv_id: str):
        """Generate a visualization of the conversation."""
        # 获取特定会话ID的消息列表
        messages = self.message_memory.get_by_conv_id(conv_id=conv_id)
        # 临时分组，有序字典，用于存储按目标分组的消息列表
        temp_group: Dict[str, List[GptsMessage]] = OrderedDict()
        # 未指定目标的消息计数器
        none_goal_count = 1
        # 计数器，用于迭代消息
        count: int = 0
        # 遍历消息列表
        for message in messages:
            # 更新消息计数器
            count = count + 1
            # 跳过第一条消息
            if count == 1:
                continue
            # 获取当前消息的目标
            current_goal = message.current_goal

            # 获取临时分组中最后一个目标
            last_goal = next(reversed(temp_group)) if temp_group else None
            # 如果存在最后一个目标
            if last_goal:
                # 获取最后一个目标的消息列表
                last_goal_messages = temp_group[last_goal]
                # 如果当前目标与最后一个目标相同
                if current_goal:
                    if current_goal == last_goal:
                        # 将消息添加到最后一个目标的消息列表中
                        last_goal_messages.append(message)
                    else:
                        # 创建新的目标消息列表
                        temp_group[current_goal] = [message]
                else:
                    # 将消息添加到未指定目标的消息列表中
                    temp_group[f"{NONE_GOAL_PREFIX}{none_goal_count}"] = [message]
                    none_goal_count += 1
            else:
                # 如果不存在最后一个目标
                if current_goal:
                    # 创建新的目标消息列表
                    temp_group[current_goal] = [message]
                else:
                    # 将消息添加到未指定目标的消息列表中
                    temp_group[f"{NONE_GOAL_PREFIX}{none_goal_count}"] = [message]
                    none_goal_count += 1

        # 构建消息分组的可视化，并返回结果
        return await self._message_group_vis_build(temp_group)
    async def one_chat_completions(self, conv_id: str):
        """Generate a visualization of the conversation."""
        # 从消息内存中获取指定会话ID的消息列表
        messages = self.message_memory.get_by_conv_id(conv_id=conv_id)
        # 使用 defaultdict 创建临时分组字典，键为字符串，值为 GptsMessage 对象列表
        temp_group: Dict[str, List[GptsMessage]] = defaultdict(list)
        # 初始化临时消息列表和可视化项目列表
        temp_messages = []
        vis_items = []
        # 初始化计数器
        count: int = 0
        # 遍历消息列表
        for message in messages:
            count = count + 1
            # 跳过第一条消息
            if count == 1:
                continue
            # 如果消息的当前目标为空或长度为0
            if not message.current_goal or len(message.current_goal) <= 0:
                # 如果临时分组中有内容，则生成计划可视化并清空临时分组
                if len(temp_group) > 0:
                    vis_items.append(await self._plan_vis_build(temp_group))
                    temp_group.clear()

                # 将消息添加到临时消息列表中
                temp_messages.append(message)
            else:
                # 如果临时消息列表中有内容，则生成消息到代理的可视化并清空临时消息列表
                if len(temp_messages) > 0:
                    vis_items.append(await self._messages_to_agents_vis(temp_messages))
                    temp_messages.clear()

                # 获取消息的最后一个目标，并将消息添加到临时分组中
                last_goal = message.current_goal
                temp_group[last_goal].append(message)

        # 处理剩余的临时分组和临时消息列表中的内容
        if len(temp_group) > 0:
            vis_items.append(await self._plan_vis_build(temp_group))
            temp_group.clear()
        if len(temp_messages) > 0:
            vis_items.append(await self._messages_to_agents_vis(temp_messages, True))
            temp_messages.clear()

        # 将可视化项目列表转换为字符串并返回
        return "\n".join(vis_items)

    async def _messages_to_agents_vis(
        self, messages: List[GptsMessage], is_last_message: bool = False
    ):
        # 如果消息列表为空，返回空字符串
        if messages is None or len(messages) <= 0:
            return ""
        # 初始化消息视图列表
        messages_view = []
        # 遍历消息列表
        for message in messages:
            # 获取消息的动作报告和视图信息
            action_report_str = message.action_report
            view_info = message.content
            # 如果存在动作报告且长度大于0
            if action_report_str and len(action_report_str) > 0:
                # 尝试从动作报告字符串解析 ActionOutput 对象
                action_out = ActionOutput.from_dict(json.loads(action_report_str))
                # 如果成功解析，并且执行成功或为最后一条消息，则使用其视图信息
                if action_out is not None and (
                    action_out.is_exe_success or is_last_message
                ):
                    view = action_out.view
                    view_info = view if view else action_out.content

            # 将消息的相关信息添加到消息视图列表中
            messages_view.append(
                {
                    "sender": message.sender,
                    "receiver": message.receiver,
                    "model": message.model_name,
                    "markdown": view_info,
                }
            )
        # 获取消息代理可视化组件，并显示消息视图内容
        vis_compent = vis_client.get(VisAgentMessages.vis_tag())
        return await vis_compent.display(content=messages_view)

    async def _messages_to_plan_vis(self, messages: List[Dict]):
        # 如果消息列表为空，返回空字符串
        if messages is None or len(messages) <= 0:
            return ""
        # 获取计划可视化组件，并显示计划内容
        return await vis_client.get(VisAgentPlans.vis_tag()).display(content=messages)
```