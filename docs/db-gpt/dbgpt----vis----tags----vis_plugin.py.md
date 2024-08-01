# `.\DB-GPT-src\dbgpt\vis\tags\vis_plugin.py`

```py
"""Visualize plugins."""
# 导入 Vis 类，从上层模块 ..base 中
from ..base import Vis

# 定义 VisPlugin 类，继承自 Vis 类
class VisPlugin(Vis):
    """Protocol for visualizing plugins."""

    @classmethod
    def vis_tag(cls) -> str:
        """Return the tag name of the vis protocol module."""
        # 返回该可视化插件的标签名
        return "vis-plugin"
```