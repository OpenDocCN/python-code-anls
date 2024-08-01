# `.\DB-GPT-src\dbgpt\app\scene\chat_normal\chat.py`

```py
from typing import Dict  # 导入 Dict 类型提示

from dbgpt._private.config import Config  # 导入 Config 类
from dbgpt.app.scene import BaseChat, ChatScene  # 导入 BaseChat 和 ChatScene 类
from dbgpt.util.tracer import trace  # 导入 trace 函数

CFG = Config()  # 创建 Config 对象实例

class ChatNormal(BaseChat):
    chat_scene: str = ChatScene.ChatNormal.value()  # 设置 chat_scene 属性为 ChatNormal 的值
    keep_end_rounds: int = 10  # 设置 keep_end_rounds 属性为整数 10

    """Number of results to return from the query"""

    def __init__(self, chat_param: Dict):
        """Initialize ChatNormal instance with chat_param dictionary."""
        chat_param["chat_mode"] = ChatScene.ChatNormal  # 设置 chat_param 字典的 chat_mode 键
        super().__init__(
            chat_param=chat_param,
        )

    @trace()  # 使用 trace 装饰器进行函数追踪
    async def generate_input_values(self) -> Dict:
        """Generate input values dictionary asynchronously."""
        input_values = {"input": self.current_user_input}  # 创建包含当前用户输入的字典
        return input_values  # 返回 input_values 字典

    @property
    def chat_type(self) -> str:
        """Return the type of chat scene."""
        return ChatScene.ChatNormal.value  # 返回 ChatNormal 的值作为聊天类型
```