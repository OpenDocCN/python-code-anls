# `.\DB-GPT-src\dbgpt\vis\tags\vis_code.py`

```py
"""Code visualization protocol."""
# 引入 Vis 基类模块，该模块位于当前模块的上级目录下的 base 子目录中
from ..base import Vis

# 定义 VisCode 类，继承自 Vis 基类
class VisCode(Vis):
    """Protocol for visualizing code."""

    # 定义类方法 vis_tag，返回字符串类型的标签名称，用于表示可视化协议模块
    @classmethod
    def vis_tag(cls) -> str:
        """Return the tag name of the vis protocol module."""
        return "vis-code"
```