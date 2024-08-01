# `.\DB-GPT-src\dbgpt\vis\tags\vis_gpts_execution.py`

```py
"""Agent Plans Vis Protocol."""

# 导入 Vis 类，.. 表示从当前包的父级包中导入
from ..base import Vis

# 定义 VisDbgptsFlow 类，继承自 Vis 类
class VisDbgptsFlow(Vis):
    """DBGPts Flow Vis Protocol."""

    @classmethod
    def vis_tag(cls) -> str:
        """Return the tag name of the vis protocol module."""
        # 返回字符串 "dbgpts-flow"，表示该可视化协议模块的标签名
        return "dbgpts-flow"
```