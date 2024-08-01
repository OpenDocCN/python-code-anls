# `.\DB-GPT-src\dbgpt\agent\expand\summary_assistant_agent.py`

```py
"""Summary Assistant Agent."""

# 导入日志模块
import logging

# 导入核心空动作类
from ..core.action.blank_action import BlankAction

# 导入对话代理基类
from ..core.base_agent import ConversableAgent

# 导入动态配置模块
from ..core.profile import DynConfig, ProfileConfig

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class SummaryAssistantAgent(ConversableAgent):
    """Summary Assistant Agent."""

    # 定义代理配置文件，包括代理名称、角色、目标、约束和描述信息
    profile: ProfileConfig = ProfileConfig(
        # 代理名称，动态配置
        name=DynConfig(
            "Aristotle",
            category="agent",
            key="dbgpt_agent_expand_summary_assistant_agent_profile_name",
        ),
        # 代理角色，动态配置
        role=DynConfig(
            "Summarizer",
            category="agent",
            key="dbgpt_agent_expand_summary_assistant_agent_profile_role",
        ),
        # 代理目标，动态配置
        goal=DynConfig(
            "Summarize answer summaries based on user questions from provided "
            "resource information or from historical conversation memories.",
            category="agent",
            key="dbgpt_agent_expand_summary_assistant_agent_profile_goal",
        ),
        # 代理约束，动态配置列表
        constraints=DynConfig(
            [
                "Prioritize the summary of answers to user questions from the improved "
                "resource text. If no relevant information is found, summarize it from "
                "the historical dialogue memory given. It is forbidden to make up your "
                "own.",
                "You need to first detect user's question that you need to answer with "
                "your summarization.",
                "Extract the provided text content used for summarization.",
                "Then you need to summarize the extracted text content.",
                "Output the content of summarization ONLY related to user's question. "
                "The output language must be the same to user's question language.",
                "If you think the provided text content is not related to user "
                "questions at all, ONLY output 'Did not find the information you "
                "want.'!!.",
            ],
            category="agent",
            key="dbgpt_agent_expand_summary_assistant_agent_profile_constraints",
        ),
        # 代理描述信息，动态配置
        desc=DynConfig(
            "You can summarize provided text content according to user's questions"
            " and output the summarization.",
            category="agent",
            key="dbgpt_agent_expand_summary_assistant_agent_profile_desc",
        ),
    )

    def __init__(self, **kwargs):
        """Create a new SummaryAssistantAgent instance."""
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 初始化空动作列表
        self._init_actions([BlankAction])
```