# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\client_lib\logging\helpers.py`

```py
# 从 math 模块中导入 ceil 和 floor 函数
# 从 typing 模块中导入 TYPE_CHECKING 类型
from math import ceil, floor
from typing import TYPE_CHECKING

# 如果 TYPE_CHECKING 为真，则从 autogpt.core.prompting 模块中导入 ChatPrompt 类
if TYPE_CHECKING:
    from autogpt.core.prompting import ChatPrompt

# 定义常量 SEPARATOR_LENGTH 为 42
SEPARATOR_LENGTH = 42

# 定义函数 dump_prompt，接受一个类型为 ChatPrompt 的参数，返回一个字符串
def dump_prompt(prompt: "ChatPrompt") -> str:
    # 定义内部函数 separator，接受一个字符串参数 text，返回一个带有分隔符的字符串
    def separator(text: str):
        # 计算分隔符两侧的长度
        half_sep_len = (SEPARATOR_LENGTH - 2 - len(text)) / 2
        # 返回带有分隔符的字符串
        return f"{floor(half_sep_len)*'-'} {text.upper()} {ceil(half_sep_len)*'-'}"

    # 格式化消息列表，每个消息包含一个分隔符和内容
    formatted_messages = "\n".join(
        [f"{separator(m.role)}\n{m.content}" for m in prompt.messages]
    )
    # 返回格式化后的字符串，包含类名、消息长度和格式化消息列表
    return f"""
============== {prompt.__class__.__name__} ==============
Length: {len(prompt.messages)} messages
{formatted_messages}
==========================================
"""
```