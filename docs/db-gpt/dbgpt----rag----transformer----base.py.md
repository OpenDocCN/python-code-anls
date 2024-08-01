# `.\DB-GPT-src\dbgpt\rag\transformer\base.py`

```py
"""Transformer base class."""
# 导入日志模块
import logging
# 导入抽象基类相关模块
from abc import ABC, abstractmethod
# 导入类型提示相关模块
from typing import List, Optional

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


# 定义TransformerBase类，作为所有转换器类的基类
class TransformerBase:
    """Transformer base class."""


# 定义EmbedderBase类，继承自TransformerBase和ABC，作为所有嵌入器类的基类
class EmbedderBase(TransformerBase, ABC):
    """Embedder base class."""


# 定义ExtractorBase类，继承自TransformerBase和ABC，作为所有抽取器类的基类
class ExtractorBase(TransformerBase, ABC):
    """Extractor base class."""

    @abstractmethod
    async def extract(self, text: str, limit: Optional[int] = None) -> List:
        """Extract results from text."""
        # 抽象方法，用于从文本中提取结果


# 定义TranslatorBase类，继承自TransformerBase和ABC，作为所有翻译器类的基类
class TranslatorBase(TransformerBase, ABC):
    """Translator base class."""
```