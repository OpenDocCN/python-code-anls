# `.\graphrag\graphrag\query\llm\text_utils.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和库
"""Text Utilities for LLM."""
from collections.abc import Iterator  # 导入迭代器抽象基类
from itertools import islice  # 导入切片迭代器

# 导入自定义模块
import tiktoken  # 导入名为tiktoken的模块

# 定义函数：计算文本中的令牌数量
def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    # 如果未提供编码器，则使用默认编码器"cl100k_base"
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    # 返回文本经编码器编码后的令牌数量
    return len(token_encoder.encode(text))  # type: ignore

# 定义函数：将可迭代对象分批处理为长度为n的元组
def batched(iterable: Iterator, n: int):
    """
    Batch data into tuples of length n. The last batch may be shorter.

    Taken from Python's cookbook: https://docs.python.org/3/library/itertools.html#itertools.batched
    """
    # 如果n小于1，则抛出值错误异常
    if n < 1:
        value_error = "n must be at least one"
        raise ValueError(value_error)
    # 创建迭代器对象it，用于迭代可迭代对象iterable
    it = iter(iterable)
    # 使用islice函数将迭代器it的内容切片为长度为n的元组，直到迭代器it耗尽
    while batch := tuple(islice(it, n)):
        yield batch

# 定义函数：按令牌长度将文本分块
def chunk_text(
    text: str, max_tokens: int, token_encoder: tiktoken.Encoding | None = None
):
    """Chunk text by token length."""
    # 如果未提供编码器，则使用默认编码器"cl100k_base"
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    # 使用编码器对文本进行编码，得到令牌列表
    tokens = token_encoder.encode(text)  # type: ignore
    # 将令牌列表转换为长度为max_tokens的批次迭代器
    chunk_iterator = batched(iter(tokens), max_tokens)
    # 通过生成器从chunk_iterator中产生每个批次
    yield from chunk_iterator
```