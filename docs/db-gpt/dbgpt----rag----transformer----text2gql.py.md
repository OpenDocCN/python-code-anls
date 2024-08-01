# `.\DB-GPT-src\dbgpt\rag\transformer\text2gql.py`

```py
"""Text2GQL class."""
# 引入日志模块，用于记录程序运行时的信息
import logging

# 从 dbgpt.rag.transformer.base 模块中导入 TranslatorBase 类
from dbgpt.rag.transformer.base import TranslatorBase

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义 Text2GQL 类，继承自 TranslatorBase 类
class Text2GQL(TranslatorBase):
    """Text2GQL class."""
```