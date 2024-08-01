# `.\DB-GPT-src\dbgpt\agent\expand\Indicator_assistant_agent.py`

```py
"""
Indicator Assistant Agent.
"""

import logging  # 导入日志模块

from ..core.base_agent import ConversableAgent  # 导入基础Agent类
from ..core.profile import DynConfig, ProfileConfig  # 导入动态配置和配置文件类
from .actions.indicator_action import IndicatorAction  # 导入指示器动作类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class IndicatorAssistantAgent(ConversableAgent):
    """Indicator Assistant Agent."""

    profile: ProfileConfig = ProfileConfig(
        name=DynConfig(
            "Indicator",  # 设置代理名称为“Indicator”
            category="agent",  # 分类为“agent”
            key="dbgpt_agent_expand_indicator_assistant_agent_profile_name",  # 关键字配置
        ),
        role=DynConfig(
            "Indicator",  # 设置角色为“Indicator”
            category="agent",  # 分类为“agent”
            key="dbgpt_agent_expand_indicator_assistant_agent_profile_role",  # 关键字配置
        ),
        goal=DynConfig(
            "Summarize answer summaries based on user questions from provided "
            "resource information or from historical conversation memories.",  # 设置代理目标描述
            category="agent",  # 分类为“agent”
            key="dbgpt_agent_expand_indicator_assistant_agent_profile_goal",  # 关键字配置
        ),
        constraints=DynConfig(
            [
                "Prioritize the summary of answers to user questions from the "
                "improved resource text. If no relevant information is found, "
                "summarize it from the historical dialogue memory given. It is "
                "forbidden to make up your own.",
                "You need to first detect user's question that you need to answer "
                "with your summarization.",
                "Extract the provided text content used for summarization.",
                "Then you need to summarize the extracted text content.",
                "Output the content of summarization ONLY related to user's question. "
                "The output language must be the same to user's question language.",
                "If you think the provided text content is not related to user "
                "questions at all, ONLY output 'Did not find the information you "
                "want.'!!.",
            ],  # 设置代理的约束条件描述
            category="agent",  # 分类为“agent”
            key="dbgpt_agent_expand_indicator_assistant_agent_profile_constraints",  # 关键字配置
        ),
        desc=DynConfig(
            "You can summarize provided text content according to user's questions "
            "and output the summarization.",  # 设置代理的描述
            category="agent",  # 分类为“agent”
            key="dbgpt_agent_expand_indicator_assistant_agent_profile_desc",  # 关键字配置
        ),
    )

    def __init__(self, **kwargs):
        """Create a new instance."""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self._init_actions([IndicatorAction])  # 初始化动作列表，添加指示器动作类
```