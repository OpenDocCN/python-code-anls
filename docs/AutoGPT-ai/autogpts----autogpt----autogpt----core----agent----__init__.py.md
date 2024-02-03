# `.\AutoGPT\autogpts\autogpt\autogpt\core\agent\__init__.py`

```py
"""The Agent is an autonomouos entity guided by a LLM provider."""
# 该注释描述了 Agent 类是一个由 LLM 提供者指导的自主实体

from autogpt.core.agent.base import Agent
# 从 autogpt.core.agent.base 模块导入 Agent 类

from autogpt.core.agent.simple import AgentSettings, SimpleAgent
# 从 autogpt.core.agent.simple 模块导入 AgentSettings 和 SimpleAgent 类

__all__ = [
    "Agent",
    "AgentSettings",
    "SimpleAgent",
]
# 将 Agent、AgentSettings 和 SimpleAgent 类添加到 __all__ 列表中，表示它们是该模块的公开接口
```