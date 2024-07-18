# `.\graphrag\graphrag\index\verbs\text\translate\strategies\typing.py`

```py
# 导入必要的模块和类
from collections.abc import Awaitable, Callable  # 导入Awaitable和Callable类
from dataclasses import dataclass  # 导入dataclass装饰器
from typing import Any  # 导入Any类型

from datashaper import VerbCallbacks  # 从datashaper模块导入VerbCallbacks类

from graphrag.index.cache import PipelineCache  # 从graphrag.index.cache模块导入PipelineCache类

# 使用dataclass装饰器定义一个数据类TextTranslationResult，用于表示文本翻译的结果
@dataclass
class TextTranslationResult:
    """Text translation result class definition."""
    
    translations: list[str]  # 类型注解，表示translations属性是一个字符串列表，存储翻译结果

# 定义一个类型别名TextTranslationStrategy，它是一个可调用类型，接受特定参数和返回一个Awaitable的TextTranslationResult对象
TextTranslationStrategy = Callable[
    [list[str], dict[str, Any], VerbCallbacks, PipelineCache],
    Awaitable[TextTranslationResult],
]
```