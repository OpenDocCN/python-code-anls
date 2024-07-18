# `.\graphrag\graphrag\index\verbs\text\embed\text_embed.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing text_embed, load_strategy and create_row_from_embedding_data methods definition."""

import logging
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd
from datashaper import TableContainer, VerbCallbacks, VerbInput, verb

from graphrag.index.cache import PipelineCache
from graphrag.vector_stores import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreFactory,
)

# Set up logging for the current module
log = logging.getLogger(__name__)

# Per Azure OpenAI Limits
# https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
DEFAULT_EMBEDDING_BATCH_SIZE = 500

# Define an enumeration for different text embedding strategy types
class TextEmbedStrategyType(str, Enum):
    """TextEmbedStrategyType class definition."""
    openai = "openai"
    mock = "mock"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'

@verb(name="text_embed")
async def text_embed(
    input: VerbInput,
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    column: str,
    strategy: dict,
    **kwargs,
) -> TableContainer:
    """
    Embed a piece of text into a vector space. The verb outputs a new column containing a mapping between doc_id and vector.

    ## Usage
    ```yaml
    verb: text_embed
    args:
        column: text # The name of the column containing the text to embed, this can either be a column with text, or a column with a list[tuple[doc_id, str]]
        to: embedding # The name of the column to output the embedding to
        strategy: <strategy config> # See strategies section below
    ```py

    ## Strategies
    The text embed verb uses a strategy to embed the text. The strategy is an object which defines the strategy to use. The following strategies are available:

    ### openai
    This strategy uses openai to embed a piece of text. In particular it uses a LLM to embed a piece of text. The strategy config is as follows:

    ```yaml
    strategy:
        type: openai
        llm: # The configuration for the LLM
            type: openai_embedding # the type of llm to use, available options are: openai_embedding, azure_openai_embedding
            api_key: !ENV ${GRAPHRAG_OPENAI_API_KEY} # The api key to use for openai
            model: !ENV ${GRAPHRAG_OPENAI_MODEL:gpt-4-turbo-preview} # The model to use for openai
            max_tokens: !ENV ${GRAPHRAG_MAX_TOKENS:6000} # The max tokens to use for openai
            organization: !ENV ${GRAPHRAG_OPENAI_ORGANIZATION} # The organization to use for openai
        vector_store: # The optional configuration for the vector store
            type: lancedb # The type of vector store to use, available options are: azure_ai_search, lancedb
            <...>
    ```py
    """
    # Extract the configuration for the vector store from the strategy dictionary
    vector_store_config = strategy.get("vector_store")
    # 如果 vector_store_config 不为空，则进入条件判断
    if vector_store_config:
        # 获取参数中的 embedding_name，如果不存在则使用默认值 "default"
        embedding_name = kwargs.get("embedding_name", "default")
        # 根据 vector_store_config 和 embedding_name 获取集合名称
        collection_name = _get_collection_name(vector_store_config, embedding_name)
        # 根据 vector_store_config 创建 BaseVectorStore 对象，并使用得到的集合名称
        vector_store: BaseVectorStore = _create_vector_store(
            vector_store_config, collection_name
        )
        # 获取与当前 embedding_name 相关的 vector_store_workflow_config
        vector_store_workflow_config = vector_store_config.get(
            embedding_name, vector_store_config
        )
        # 使用 _text_embed_with_vector_store 函数进行文本嵌入，传入相关参数
        return await _text_embed_with_vector_store(
            input,
            callbacks,
            cache,
            column,
            strategy,
            vector_store,
            vector_store_workflow_config,
            # 检查是否需要将结果存储在表格中，默认为 False
            vector_store_config.get("store_in_table", False),
            # 获取参数中的 to，如果不存在则使用默认的列名 + "_embedding"
            kwargs.get("to", f"{column}_embedding"),
        )
    
    # 如果 vector_store_config 为空，则进入此分支，使用 _text_embed_in_memory 函数进行内存中的文本嵌入
    return await _text_embed_in_memory(
        input,
        callbacks,
        cache,
        column,
        strategy,
        # 获取参数中的 to，如果不存在则使用默认的列名 + "_embedding"
        kwargs.get("to", f"{column}_embedding"),
    )
# 异步函数，将文本嵌入内存中
async def _text_embed_with_vector_store(
    input: VerbInput,                     # 输入对象
    callbacks: VerbCallbacks,             # 回调函数对象
    cache: PipelineCache,                 # 流水线缓存对象
    column: str,                          # 待处理的列名
    strategy: dict[str, Any],             # 策略参数字典
    vector_store: BaseVectorStore,        # 向量存储对象
    vector_store_config: dict,            # 向量存储配置参数
    store_in_table: bool = False,         # 是否存储在表中的标志，默认为False
    to: str = "",                         # 存储结果的目标列名，默认为空字符串
):
    output_df = cast(pd.DataFrame, input.get_input())  # 获得输入数据并转换为DataFrame类型
    strategy_type = strategy["type"]       # 获取策略类型
    strategy_exec = load_strategy(strategy_type)  # 根据策略类型加载对应的执行器
    strategy_args = {**strategy}           # 复制策略参数

    # 获取向量存储的配置信息
    insert_batch_size: int = (
        vector_store_config.get("batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE
    )
    title_column: str = vector_store_config.get("title_column", "title")
    id_column: str = vector_store_config.get("id_column", "id")
    overwrite: bool = vector_store_config.get("overwrite", True)

    # 检查输入数据中是否包含指定的列名
    if column not in output_df.columns:
        msg = f"Column {column} not found in input dataframe with columns {output_df.columns}"
        raise ValueError(msg)
    if title_column not in output_df.columns:
        msg = f"Column {title_column} not found in input dataframe with columns {output_df.columns}"
        raise ValueError(msg)
    if id_column not in output_df.columns:
        msg = f"Column {id_column} not found in input dataframe with columns {output_df.columns}"
        raise ValueError(msg)

    total_rows = 0
    # 计算待处理列中的总行数
    for row in output_df[column]:
        if isinstance(row, list):
            total_rows += len(row)
        else:
            total_rows += 1

    i = 0
    starting_index = 0

    all_results = []  # 初始化存储所有处理结果的列表
    # 当前批次的起始索引乘以插入批量大小小于输入数据的行数时，执行循环
    while insert_batch_size * i < input.get_input().shape[0]:
        # 获取当前批次的数据
        batch = input.get_input().iloc[
            insert_batch_size * i : insert_batch_size * (i + 1)
        ]
        # 提取批次中指定列的文本数据并转换为列表
        texts: list[str] = batch[column].to_numpy().tolist()
        # 提取批次中指定列的标题数据并转换为列表
        titles: list[str] = batch[title_column].to_numpy().tolist()
        # 提取批次中指定列的ID数据并转换为列表
        ids: list[str] = batch[id_column].to_numpy().tolist()
        # 执行策略函数，传入文本数据、回调函数、缓存、策略参数
        result = await strategy_exec(
            texts,
            callbacks,
            cache,
            strategy_args,
        )
        # 如果需要存储在表中并且结果中包含嵌入向量
        if store_in_table and result.embeddings:
            # 筛选出不为None的嵌入向量并加入到总结果列表中
            embeddings = [
                embedding for embedding in result.embeddings if embedding is not None
            ]
            all_results.extend(embeddings)

        # 获取结果中的嵌入向量列表，如果结果为空则返回空列表
        vectors = result.embeddings or []
        # 存储向量对应的文档信息列表
        documents: list[VectorStoreDocument] = []
        # 遍历ID、文本、标题、向量的组合，构建文档对象并加入文档列表中
        for id, text, title, vector in zip(ids, texts, titles, vectors):
            # 如果向量是numpy数组，则转换为列表形式
            if type(vector) is np.ndarray:
                vector = vector.tolist()
            # 创建文档对象，包括ID、文本、向量以及额外属性（如标题）
            document = VectorStoreDocument(
                id=id,
                text=text,
                vector=vector,
                attributes={"title": title},
            )
            # 将文档对象添加到文档列表中
            documents.append(document)

        # 将文档列表加载到向量存储中，如果覆盖选项为真且是第一次迭代，则覆盖已有文档
        vector_store.load_documents(documents, overwrite and i == 0)
        # 更新起始索引，增加当前批次处理的文档数
        starting_index += len(documents)
        # 更新迭代计数器
        i += 1

    # 如果需要将结果存储在表中，则将所有结果嵌入向量存储到输出数据框的指定列中
    if store_in_table:
        output_df[to] = all_results

    # 返回包含输出数据框的表容器对象
    return TableContainer(table=output_df)
def _create_vector_store(
    vector_store_config: dict, collection_name: str
) -> BaseVectorStore:
    # 获取向量存储的类型
    vector_store_type: str = str(vector_store_config.get("type"))
    # 如果指定了 collection_name，则更新配置
    if collection_name:
        vector_store_config.update({"collection_name": collection_name})

    # 使用工厂方法获取向量存储对象
    vector_store = VectorStoreFactory.get_vector_store(
        vector_store_type, kwargs=vector_store_config
    )

    # 连接向量存储
    vector_store.connect(**vector_store_config)
    # 返回向量存储对象
    return vector_store


def _get_collection_name(vector_store_config: dict, embedding_name: str) -> str:
    # 获取配置中的 collection_name
    collection_name = vector_store_config.get("collection_name")
    # 如果未指定 collection_name，则从 collection_names 中获取对应的值
    if not collection_name:
        collection_names = vector_store_config.get("collection_names", {})
        collection_name = collection_names.get(embedding_name, embedding_name)

    # 记录日志信息，使用的向量存储类型和 collection_name
    msg = f"using {vector_store_config.get('type')} collection_name {collection_name} for embedding {embedding_name}"
    log.info(msg)
    # 返回最终确定的 collection_name
    return collection_name


def load_strategy(strategy: TextEmbedStrategyType) -> TextEmbeddingStrategy:
    """Load strategy method definition."""
    # 根据不同的策略类型加载相应的策略实现
    match strategy:
        case TextEmbedStrategyType.openai:
            from .strategies.openai import run as run_openai

            return run_openai
        case TextEmbedStrategyType.mock:
            from .strategies.mock import run as run_mock

            return run_mock
        case _:
            # 如果策略类型未知，则抛出 ValueError 异常
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)
```