# `.\DB-GPT-src\dbgpt\vis\base.py`

```py
"""Base class for vis protocol module."""
# 导入所需模块和类型提示
import json
from typing import Any, Dict, Optional

# 导入自定义的序列化函数
from dbgpt.util.json_utils import serialize


class Vis:
    """Vis protocol base class."""

    def render_prompt(self) -> Optional[str]:
        """Return the prompt for the vis protocol."""
        # 返回空值作为 vis 协议的提示信息
        return None

    async def generate_param(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate the parameters required by the vis protocol.

        Display corresponding content using vis protocol
        Args:
            **kwargs: 传入的参数字典

        Returns:
        vis protocol text 返回使用 vis 协议显示的内容
        """
        # 返回传入参数中的内容作为 vis 协议的参数
        return kwargs["content"]

    async def display(self, **kwargs) -> Optional[str]:
        """Display the content using the vis protocol."""
        # 生成参数内容的 JSON 字符串，使用自定义的序列化函数
        content = json.dumps(
            await self.generate_param(**kwargs), default=serialize, ensure_ascii=False
        )
        # 构造包含 vis 标签的 Markdown 代码块，用于显示内容
        return f"```{self.vis_tag()}\n{content}\n```py"

    @classmethod
    def vis_tag(cls) -> str:
        """Return current vis protocol module tag name."""
        # 返回当前 vis 协议模块的标签名称，这里默认为空字符串
        return ""
```