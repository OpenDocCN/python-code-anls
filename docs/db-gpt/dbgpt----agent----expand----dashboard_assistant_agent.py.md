# `.\DB-GPT-src\dbgpt\agent\expand\dashboard_assistant_agent.py`

```py
"""Dashboard Assistant Agent."""

# 导入所需模块
from typing import List

from ..core.agent import AgentMessage
from ..core.base_agent import ConversableAgent
from ..core.profile import DynConfig, ProfileConfig
from ..resource.database import DBResource
from .actions.dashboard_action import DashboardAction


class DashboardAssistantAgent(ConversableAgent):
    """Dashboard Assistant Agent."""

    # 定义代理配置
    profile: ProfileConfig = ProfileConfig(
        # 代理名称动态配置
        name=DynConfig(
            "Visionary",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_name",
        ),
        # 代理角色动态配置
        role=DynConfig(
            "Reporter",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_role",
        ),
        # 代理目标动态配置
        goal=DynConfig(
            "Read the provided historical messages, collect various analysis SQLs "
            "from them, and assemble them into professional reports.",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_goal",
        ),
        # 代理约束动态配置
        constraints=DynConfig(
            [
                "You are only responsible for collecting and sorting out the analysis "
                "SQL that already exists in historical messages, and do not generate "
                "any analysis sql yourself.",
                "In order to build a report with rich display types, you can "
                "appropriately adjust the display type of the charts you collect so "
                "that you can build a better report. Of course, you can choose from "
                "the following available display types: {{ display_type }}",
                "Please read and completely collect all analysis sql in the "
                "historical conversation, and do not omit or modify the content of "
                "the analysis sql.",
            ],
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_constraints",
        ),
        # 代理描述动态配置
        desc=DynConfig(
            "Observe and organize various analysis results and construct "
            "professional reports",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_desc",
        ),
    )

    def __init__(self, **kwargs):
        """Create a new instance of DashboardAssistantAgent."""
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 初始化操作动作
        self._init_actions([DashboardAction])

    def _init_reply_message(self, received_message: AgentMessage) -> AgentMessage:
        # 调用父类方法初始化回复消息
        reply_message = super()._init_reply_message(received_message)

        # 从资源中获取数据库资源列表
        dbs: List[DBResource] = DBResource.from_resource(self.resource)

        # 如果数据库资源列表为空，抛出数值错误异常
        if not dbs:
            raise ValueError(
                f"Resource type {self.actions[0].resource_need} is not supported."
            )

        # 获取第一个数据库资源
        db = dbs[0]
        # 设置回复消息的上下文信息
        reply_message.context = {
            "display_type": self.actions[0].render_prompt(),
            "dialect": db.dialect,
        }
        return reply_message
```