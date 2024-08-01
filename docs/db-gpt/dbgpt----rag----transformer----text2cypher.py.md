# `.\DB-GPT-src\dbgpt\rag\transformer\text2cypher.py`

```py
"""Text2Cypher class."""
# 引入 logging 模块，用于记录日志信息
import logging

# 从 dbgpt.rag.transformer.base 模块中导入 TranslatorBase 类
from dbgpt.rag.transformer.base import TranslatorBase

# 获取当前模块的 logger 对象
logger = logging.getLogger(__name__)


class Text2Cypher(TranslatorBase):
    """Text2Cypher class."""
    # Text2Cypher 类继承自 TranslatorBase 类，用于将文本转换为 Cypher 查询语句
```