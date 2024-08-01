# `.\DB-GPT-src\dbgpt\vis\tags\vis_gpts_result.py`

```py
"""Visualize the result of the DBGPts flow."""

# 导入基于上级目录的 Vis 模块
from ..base import Vis

# 定义一个用于可视化 DBGPts 流结果的类，继承自 Vis 类
class VisDbgptsFlowResult(Vis):
    """Protocol for visualizing the result of the DBGPts flow."""

    @classmethod
    def vis_tag(cls) -> str:
        """Return the tag name of the vis protocol module."""
        # 返回可视化协议模块的标签名称
        return "dbgpts-result"
```