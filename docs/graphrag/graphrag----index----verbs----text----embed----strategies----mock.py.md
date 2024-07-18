# `.\graphrag\graphrag\index\verbs\text\embed\strategies\mock.py`

```py
# 导入必要的模块和类
"""A module containing run and _embed_text methods definitions."""
import random  # 导入随机数生成模块
from collections.abc import Iterable  # 从标准库的 collections.abc 模块导入 Iterable 类
from typing import Any  # 从 typing 模块导入 Any 类型

# 导入 datashaper 模块中的 ProgressTicker, VerbCallbacks, progress_ticker 函数
from datashaper import ProgressTicker, VerbCallbacks, progress_ticker  

# 从 graphrag.index.cache 模块导入 PipelineCache 类
from graphrag.index.cache import PipelineCache  

# 从当前目录下的 typing 模块中导入 TextEmbeddingResult 类型
from .typing import TextEmbeddingResult  


async def run(  # noqa RUF029 async is required for interface
    input: list[str],  # 定义函数参数 input，类型为 list，其中每个元素为 str 类型
    callbacks: VerbCallbacks,  # 定义函数参数 callbacks，类型为 VerbCallbacks 类
    cache: PipelineCache,  # 定义函数参数 cache，类型为 PipelineCache 类
    _args: dict[str, Any],  # 定义函数参数 _args，类型为 dict，其中键为 str，值为 Any 类型
) -> TextEmbeddingResult:
    """Run the Claim extraction chain."""
    input = input if isinstance(input, Iterable) else [input]  # 如果 input 不是 Iterable 类型，则转为列表形式
    ticker = progress_ticker(callbacks.progress, len(input))  # 创建进度条 ticker 对象
    return TextEmbeddingResult(
        embeddings=[_embed_text(cache, text, ticker) for text in input]  # 调用 _embed_text 函数生成文本嵌入
    )


def _embed_text(_cache: PipelineCache, _text: str, tick: ProgressTicker) -> list[float]:
    """Embed a single piece of text."""
    tick(1)  # 调用 tick 函数，表示处理了一个文本
    return [random.random(), random.random(), random.random()]  # 返回包含三个随机数的列表  # noqa S311
```