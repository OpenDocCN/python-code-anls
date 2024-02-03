# `.\AutoGPT\autogpts\autogpt\autogpt\core\prompting\__init__.py`

```py
# 从当前目录下的 base 模块中导入 PromptStrategy 类
from .base import PromptStrategy
# 从当前目录下的 schema 模块中导入 ChatPrompt 和 LanguageModelClassification 类
from .schema import ChatPrompt, LanguageModelClassification

# 定义一个列表，包含需要导出的类名
__all__ = [
    "LanguageModelClassification",
    "ChatPrompt",
    "PromptStrategy",
]
```