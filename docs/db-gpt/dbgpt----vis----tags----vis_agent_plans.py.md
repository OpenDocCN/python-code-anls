# `.\DB-GPT-src\dbgpt\vis\tags\vis_agent_plans.py`

```py
"""Agent Plans Vis Protocol."""
# 导入父模块中的 Vis 类
from ..base import Vis

# 定义 AgentPlansVis 类，继承自 Vis 类
class VisAgentPlans(Vis):
    """Agent Plans Vis Protocol."""

    @classmethod
    def vis_tag(cls) -> str:
        """Return the tag name of the vis protocol module."""
        # 返回该可视化协议模块的标签名称
        return "agent-plans"
```