# `.\DB-GPT-src\dbgpt\agent\core\agent_manage.py`

```py
"""Manages the registration and retrieval of agents."""

import logging  # 导入日志记录模块
import re  # 导入正则表达式模块
from collections import defaultdict  # 导入默认字典模块
from typing import Dict, List, Optional, Set, Tuple, Type, cast  # 导入类型提示模块

from dbgpt.component import BaseComponent, ComponentType, SystemApp  # 导入基础组件和系统应用相关模块

from .agent import Agent  # 导入本地的Agent类
from .base_agent import ConversableAgent  # 导入本地的ConversableAgent类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def participant_roles(agents: List[Agent]) -> str:
    """Return a string listing the roles of the agents."""
    # Default to all agents registered
    roles = []
    for agent in agents:
        roles.append(f"{agent.name}: {agent.desc}")  # 构建每个代理的名称和描述字符串
    return "\n".join(roles)  # 将所有代理角色信息以换行连接成一个字符串返回


def mentioned_agents(message_content: str, agents: List[Agent]) -> Dict:
    """Return a dictionary mapping agent names to mention counts.

    Finds and counts agent mentions in the string message_content, taking word
    boundaries into account.

    Returns: A dictionary mapping agent names to mention counts (to be included,
    at least one mention must occur)
    """
    mentions = dict()
    for agent in agents:
        regex = (
            r"(?<=\W)" + re.escape(agent.name) + r"(?=\W)"
        )  # 构建正则表达式，用于匹配包含代理名称的单词
        count = len(
            re.findall(regex, " " + message_content + " ")
        )  # 在消息内容两端加空格，以确保匹配到单词边界
        if count > 0:
            mentions[agent.name] = count  # 如果至少有一个匹配，则记录代理名称及其出现次数
    return mentions  # 返回代理名称到出现次数的字典


class AgentManager(BaseComponent):
    """Manages the registration and retrieval of agents."""

    name = ComponentType.AGENT_MANAGER  # 组件类型为代理管理器

    def __init__(self, system_app: SystemApp):
        """Create a new AgentManager."""
        super().__init__(system_app)
        self.system_app = system_app  # 初始化系统应用实例
        self._agents: Dict[
            str, Tuple[Type[ConversableAgent], ConversableAgent]
        ] = defaultdict()  # 初始化代理字典为默认字典

        self._core_agents: Set[str] = set()  # 初始化核心代理集合为空集合

    def init_app(self, system_app: SystemApp):
        """Initialize the AgentManager."""
        self.system_app = system_app  # 初始化系统应用实例

    def after_start(self):
        """Register all agents."""
        from ..expand.code_assistant_agent import CodeAssistantAgent  # 导入代码助手代理
        from ..expand.dashboard_assistant_agent import DashboardAssistantAgent  # 导入仪表板助手代理
        from ..expand.data_scientist_agent import DataScientistAgent  # 导入数据科学家代理
        from ..expand.summary_assistant_agent import SummaryAssistantAgent  # 导入摘要助手代理
        from ..expand.tool_assistant_agent import ToolAssistantAgent  # 导入工具助手代理

        core_agents = set()  # 创建一个集合用于存储核心代理
        core_agents.add(self.register_agent(CodeAssistantAgent))  # 注册代码助手代理并添加到核心代理集合
        core_agents.add(self.register_agent(DashboardAssistantAgent))  # 注册仪表板助手代理并添加到核心代理集合
        core_agents.add(self.register_agent(DataScientistAgent))  # 注册数据科学家代理并添加到核心代理集合
        core_agents.add(self.register_agent(SummaryAssistantAgent))  # 注册摘要助手代理并添加到核心代理集合
        core_agents.add(self.register_agent(ToolAssistantAgent))  # 注册工具助手代理并添加到核心代理集合
        self._core_agents = core_agents  # 将集合赋值给核心代理集合属性

    def register_agent(
        self, cls: Type[ConversableAgent], ignore_duplicate: bool = False
    ):
        """Register an agent class."""
        # 省略方法体，实现代理注册的功能
    ) -> str:
        """Register an agent."""
        # 创建一个新的代理实例
        inst = cls()
        # 获取该实例的角色名称
        profile = inst.role
        # 检查代理是否已经注册，如果已注册且不允许重复注册，则引发异常
        if profile in self._agents and (
            profile in self._core_agents or not ignore_duplicate
        ):
            raise ValueError(f"Agent:{profile} already register!")
        # 将代理及其类对象注册到 _agents 字典中
        self._agents[profile] = (cls, inst)
        # 返回代理的角色名称
        return profile

    def get_by_name(self, name: str) -> Type[ConversableAgent]:
        """Return an agent by name.

        Args:
            name (str): The name of the agent to retrieve.

        Returns:
            Type[ConversableAgent]: The agent with the given name.

        Raises:
            ValueError: If the agent with the given name is not registered.
        """
        # 检索并返回指定名称的代理类对象
        if name not in self._agents:
            raise ValueError(f"Agent:{name} not register!")
        return self._agents[name][0]

    def get_describe_by_name(self, name: str) -> str:
        """Return the description of an agent by name."""
        # 返回指定名称代理实例的描述信息，如果没有描述信息则返回空字符串
        return self._agents[name][1].desc or ""

    def all_agents(self) -> Dict[str, str]:
        """Return a dictionary of all registered agents and their descriptions."""
        # 返回所有已注册代理及其描述信息的字典
        result = {}
        for name, value in self._agents.items():
            result[name] = value[1].desc or ""
        return result

    def list_agents(self):
        """Return a list of all registered agents and their descriptions."""
        # 返回所有已注册代理及其角色名称和目标的列表形式
        result = []
        for name, value in self._agents.items():
            result.append(
                {
                    "name": value[1].role,
                    "desc": value[1].goal,
                }
            )
        return result
# 全局变量，用于存储系统应用的实例，初始值为 None
_SYSTEM_APP: Optional[SystemApp] = None


def initialize_agent(system_app: SystemApp):
    """初始化代理管理器。

    Args:
        system_app (SystemApp): 系统应用的实例
    """
    # 将全局变量 _SYSTEM_APP 设置为传入的系统应用实例
    global _SYSTEM_APP
    _SYSTEM_APP = system_app
    # 创建代理管理器对象，并注册到系统应用实例中
    agent_manager = AgentManager(system_app)
    system_app.register_instance(agent_manager)


def get_agent_manager(system_app: Optional[SystemApp] = None) -> AgentManager:
    """返回代理管理器对象。

    Args:
        system_app (Optional[SystemApp], optional): 系统应用的实例。默认为 None。

    Returns:
        AgentManager: 代理管理器对象
    """
    # 如果 _SYSTEM_APP 还未初始化
    if not _SYSTEM_APP:
        # 如果传入的系统应用实例为空，则创建一个新的 SystemApp 实例
        if not system_app:
            system_app = SystemApp()
        # 初始化代理管理器，并注册到系统应用实例中
        initialize_agent(system_app)
    # 确定要返回的系统应用实例，可以是传入的参数或者全局变量 _SYSTEM_APP
    app = system_app or _SYSTEM_APP
    # 返回代理管理器的单例实例
    return AgentManager.get_instance(cast(SystemApp, app))
```