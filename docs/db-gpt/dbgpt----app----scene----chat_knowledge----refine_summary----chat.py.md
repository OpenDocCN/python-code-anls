# `.\DB-GPT-src\dbgpt\app\scene\chat_knowledge\refine_summary\chat.py`

```py
from typing import Dict

from dbgpt.app.scene import BaseChat, ChatScene


class ExtractRefineSummary(BaseChat):
    # 设置聊天场景为ExtractRefineSummary
    chat_scene: str = ChatScene.ExtractRefineSummary.value()

    """extract final summary by llm"""

    def __init__(self, chat_param: Dict):
        """初始化函数"""
        # 设置chat_mode为ExtractRefineSummary
        chat_param["chat_mode"] = ChatScene.ExtractRefineSummary
        # 调用父类的初始化函数
        super().__init__(
            chat_param=chat_param,
        )

        # 获取select_param作为existing_answer
        self.existing_answer = chat_param["select_param"]

    async def generate_input_values(self):
        # 生成输入值字典
        input_values = {
            # "context": self.user_input,
            "existing_answer": self.existing_answer,
        }
        return input_values

    def stream_plugin_call(self, text):
        """返回带有summary标签的文本"""
        return f"<summary>{text}</summary>"

    @property
    def chat_type(self) -> str:
        # 返回ExtractRefineSummary的值
        return ChatScene.ExtractRefineSummary.value
```