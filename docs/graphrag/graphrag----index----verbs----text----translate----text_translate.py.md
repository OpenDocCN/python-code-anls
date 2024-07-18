# `.\graphrag\graphrag\index\verbs\text\translate\text_translate.py`

```py
# 版权声明及许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的库和模块
"""A module containing text_translate methods definition."""
from enum import Enum
from typing import Any, cast

import pandas as pd
from datashaper import (
    AsyncType,
    TableContainer,
    VerbCallbacks,
    VerbInput,
    derive_from_rows,
    verb,
)

# 导入本地模块和策略相关的库
from graphrag.index.cache import PipelineCache
from .strategies.typing import TextTranslationStrategy

# 定义一个枚举类，表示文本翻译策略的类型
class TextTranslateStrategyType(str, Enum):
    """TextTranslateStrategyType class definition."""
    openai = "openai"
    mock = "mock"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'

# 定义异步函数修饰器，用于文本翻译操作
@verb(name="text_translate")
async def text_translate(
    input: VerbInput,
    cache: PipelineCache,
    callbacks: VerbCallbacks,
    text_column: str,
    to: str,
    strategy: dict[str, Any],
    async_mode: AsyncType = AsyncType.AsyncIO,
    **kwargs,
) -> TableContainer:
    """
    Translate a piece of text into another language.

    ## Usage
    ```yaml
    verb: text_translate
    args:
        text_column: <column name> # The name of the column containing the text to translate
        to: <column name> # The name of the column to write the translated text to
        strategy: <strategy config> # The strategy to use to translate the text, see below for more details
    ```py

    ## Strategies
    The text translate verb uses a strategy to translate the text. The strategy is an object which defines the strategy to use. The following strategies are available:

    ### openai
    This strategy uses openai to translate a piece of text. In particular it uses a LLM to translate a piece of text. The strategy config is as follows:

    ```yaml
    strategy:
        type: openai
        language: english # The language to translate to, default: english
        prompt: <prompt> # The prompt to use for the translation, default: None
        chunk_size: 2500 # The chunk size to use for the translation, default: 2500
        chunk_overlap: 0 # The chunk overlap to use for the translation, default: 0
        llm: # The configuration for the LLM
            type: openai_chat # the type of llm to use, available options are: openai_chat, azure_openai_chat
            api_key: !ENV ${GRAPHRAG_OPENAI_API_KEY} # The api key to use for openai
            model: !ENV ${GRAPHRAG_OPENAI_MODEL:gpt-4-turbo-preview} # The model to use for openai
            max_tokens: !ENV ${GRAPHRAG_MAX_TOKENS:6000} # The max tokens to use for openai
            organization: !ENV ${GRAPHRAG_OPENAI_ORGANIZATION} # The organization to use for openai
    ```py
    """
    
    # 将输入转换为 DataFrame
    output_df = cast(pd.DataFrame, input.get_input())
    
    # 从策略中获取翻译策略的类型
    strategy_type = strategy["type"]
    
    # 复制策略参数以备后用
    strategy_args = {**strategy}
    
    # 根据策略类型加载相应的翻译策略执行函数
    strategy_exec = _load_strategy(strategy_type)
    async def run_strategy(row):
        # 从输入行中获取文本数据
        text = row[text_column]
        # 调用策略执行函数，获取执行结果
        result = await strategy_exec(text, strategy_args, callbacks, cache)

        # 如果输入是单个字符串，则返回该字符串的翻译结果
        if isinstance(text, str):
            return result.translations[0]

        # 否则，返回一个翻译结果列表，每个输入项对应一个翻译结果
        return list(result.translations)

    # 使用给定的函数对输出数据框中的每一行进行处理
    results = await derive_from_rows(
        output_df,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=kwargs.get("num_threads", 4),
    )

    # 将处理后的结果赋值给输出数据框中指定的列
    output_df[to] = results

    # 返回包含输出数据框的表格容器对象
    return TableContainer(table=output_df)
# 定义一个函数_load_strategy，用于根据指定的翻译策略类型加载相应的翻译策略对象
def _load_strategy(strategy: TextTranslateStrategyType) -> TextTranslationStrategy:
    # 使用匹配语法根据strategy的取值进行分支处理
    match strategy:
        # 当strategy为TextTranslateStrategyType.openai时执行以下代码块
        case TextTranslateStrategyType.openai:
            # 从.strategies.openai模块导入名为run的函数，并赋给变量run_openai
            from .strategies.openai import run as run_openai
            # 返回导入的run_openai函数对象
            return run_openai

        # 当strategy为TextTranslateStrategyType.mock时执行以下代码块
        case TextTranslateStrategyType.mock:
            # 从.strategies.mock模块导入名为run的函数，并赋给变量run_mock
            from .strategies.mock import run as run_mock
            # 返回导入的run_mock函数对象
            return run_mock

        # 默认情况，如果strategy的取值不匹配上述任何一种情况，则执行以下代码块
        case _:
            # 构建一条错误消息，指出未知的策略类型
            msg = f"Unknown strategy: {strategy}"
            # 抛出值错误异常，将错误消息作为异常信息抛出
            raise ValueError(msg)
```