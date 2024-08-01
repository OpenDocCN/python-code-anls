# `.\DB-GPT-src\dbgpt\rag\transformer\text2vector.py`

```py
"""Text2Vector class."""
# 导入日志模块
import logging

# 导入基类 EmbedderBase
from dbgpt.rag.transformer.base import EmbedderBase

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# Text2Vector 类，继承自 EmbedderBase 基类
class Text2Vector(EmbedderBase):
    """Text2Vector class."""
```