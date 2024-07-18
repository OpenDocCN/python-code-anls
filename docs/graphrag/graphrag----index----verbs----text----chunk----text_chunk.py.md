# `.\graphrag\graphrag\index\verbs\text\chunk\text_chunk.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing _get_num_total, chunk, run_strategy and load_strategy methods definitions."""

from enum import Enum  # 导入枚举类型 Enum
from typing import Any, cast  # 导入类型提示 Any 和 cast

import pandas as pd  # 导入 pandas 库
from datashaper import (  # 从 datashaper 模块导入多个符号
    ProgressTicker,  # 进度条类
    TableContainer,  # 表格容器类
    VerbCallbacks,  # 动词回调类型
    VerbInput,  # 动词输入类型
    progress_ticker,  # 进度条函数
    verb,  # 动词装饰器
)

from .strategies.typing import ChunkStrategy as ChunkStrategy  # 导入本地的 ChunkStrategy 类型别名
from .typing import ChunkInput  # 导入本地的 ChunkInput 类型别名


def _get_num_total(output: pd.DataFrame, column: str) -> int:
    """
    计算指定列中包含的总项数。

    Args:
        output (pd.DataFrame): 数据框
        column (str): 列名

    Returns:
        int: 总项数
    """
    num_total = 0  # 初始化总项数为 0
    for row in output[column]:  # 遍历指定列中的每一行
        if isinstance(row, str):  # 如果行是字符串类型
            num_total += 1  # 总项数加一
        else:  # 如果行不是字符串类型
            num_total += len(row)  # 总项数加上行的长度
    return num_total  # 返回计算得到的总项数


class ChunkStrategyType(str, Enum):
    """ChunkStrategy class definition."""

    tokens = "tokens"  # 定义 tokens 策略
    sentence = "sentence"  # 定义 sentence 策略

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'  # 返回枚举值的字符串表示形式


@verb(name="chunk")
def chunk(
    input: VerbInput,
    column: str,
    to: str,
    callbacks: VerbCallbacks,
    strategy: dict[str, Any] | None = None,
    **_kwargs,
) -> TableContainer:
    """
    将文本分块为更小的部分。

    ## Usage
    ```yaml
    verb: text_chunk
    args:
        column: <column name> # 包含要分块文本的列名，可以是包含文本的列，也可以是包含列表的列[tuple[doc_id, str]]
        to: <column name> # 输出分块结果的列名
        strategy: <strategy config> # 用于分块文本的策略，详细信息见下文
    ```py

    ## Strategies
    文本分块动词使用策略来进行文本分块。策略是一个定义要使用的策略的对象。以下是可用的策略：

    ### tokens
    该策略使用 [tokens] 库来对文本进行分块。策略配置如下：

    > 注意：将来可能会将其重命名为更通用的名称，如 "openai_tokens"。

    ```yaml
    strategy:
        type: tokens
        chunk_size: 1200 # 可选项，要使用的分块大小，默认为 1200
        chunk_overlap: 100 # 可选项，要使用的分块重叠，默认为 100
    ```py

    ### sentence
    该策略使用 nltk 库将文本分块为句子。策略配置如下：

    ```yaml
    strategy:
        type: sentence
    ```py
    """
    if strategy is None:
        strategy = {}  # 如果未指定策略，则初始化为空字典
    output = cast(pd.DataFrame, input.get_input())  # 获取输入数据，并将其转换为 pandas 数据框
    strategy_name = strategy.get("type", ChunkStrategyType.tokens)  # 获取策略名称，默认为 tokens
    strategy_config = {**strategy}  # 复制策略配置

    strategy_exec = load_strategy(strategy_name)  # 加载指定的策略执行函数

    num_total = _get_num_total(output, column)  # 计算总项数
    tick = progress_ticker(callbacks.progress, num_total)  # 创建进度条对象

    output[to] = output.apply(
        cast(
            Any,
            lambda x: run_strategy(strategy_exec, x[column], strategy_config, tick),
        ),
        axis=1,  # 按行应用函数
    )
    # 将变量 `output` 封装在 `TableContainer` 对象中，并作为返回值返回
    return TableContainer(table=output)
# 定义了一个名为 run_strategy 的函数，用于执行指定策略的方法
def run_strategy(
    strategy: ChunkStrategy,                     # 接收一个 ChunkStrategy 类型的参数 strategy，表示执行的策略
    input: ChunkInput,                           # 接收一个 ChunkInput 类型的参数 input，表示输入的数据块
    strategy_args: dict[str, Any],               # 接收一个字典类型的参数 strategy_args，表示策略执行的额外参数
    tick: ProgressTicker,                        # 接收一个 ProgressTicker 类型的参数 tick，表示进度更新器
) -> list[str | tuple[list[str] | None, str, int]]:
    """Run strategy method definition."""     # 函数的文档字符串，说明该函数是用来执行策略的方法

    # 如果输入是字符串类型，则将其转换为列表处理
    if isinstance(input, str):
        return [item.text_chunk for item in strategy([input], {**strategy_args}, tick)]

    # 可以处理输入为文本内容列表或文本内容元组（文档 ID, 文本内容）
    # text_to_chunk = '''
    texts = []
    for item in input:
        if isinstance(item, str):
            texts.append(item)
        else:
            texts.append(item[1])

    # 使用给定的策略处理文本列表 texts，并传递策略参数 strategy_args 和进度更新器 tick
    strategy_results = strategy(texts, {**strategy_args}, tick)

    results = []
    # 遍历策略处理结果
    for strategy_result in strategy_results:
        doc_indices = strategy_result.source_doc_indices
        # 如果输入中对应的项是字符串类型，则将策略结果中的文本块添加到结果列表中
        if isinstance(input[doc_indices[0]], str):
            results.append(strategy_result.text_chunk)
        else:
            # 否则，获取文档 ID 列表，并将策略结果中的文本块、标记数添加为元组形式的结果项
            doc_ids = [input[doc_idx][0] for doc_idx in doc_indices]
            results.append((
                doc_ids,
                strategy_result.text_chunk,
                strategy_result.n_tokens,
            ))
    return results


# 定义了一个名为 load_strategy 的函数，用于加载指定的策略类型
def load_strategy(strategy: ChunkStrategyType) -> ChunkStrategy:
    """Load strategy method definition."""   # 函数的文档字符串，说明该函数是用来加载策略的方法

    # 根据不同的策略类型进行匹配加载
    match strategy:
        # 如果策略类型为 ChunkStrategyType.tokens，则导入并返回 tokens 策略的运行函数
        case ChunkStrategyType.tokens:
            from .strategies.tokens import run as run_tokens
            return run_tokens
        # 如果策略类型为 ChunkStrategyType.sentence，则初始化 NLTK 并导入返回 sentence 策略的运行函数
        case ChunkStrategyType.sentence:
            from graphrag.index.bootstrap import bootstrap
            bootstrap()   # 初始化 NLTK
            from .strategies.sentence import run as run_sentence
            return run_sentence
        # 对于未知的策略类型，抛出 ValueError 异常
        case _:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)
```