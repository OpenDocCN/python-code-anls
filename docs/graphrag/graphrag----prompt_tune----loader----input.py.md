# `.\graphrag\graphrag\prompt_tune\loader\input.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Input loading module."""

from typing import cast

import pandas as pd
from datashaper import NoopVerbCallbacks, TableContainer, VerbInput

# 导入图形分析配置和输入加载功能
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.input import load_input
from graphrag.index.progress.types import ProgressReporter
from graphrag.index.verbs import chunk

# 定义最小分块大小和最小分块重叠
MIN_CHUNK_SIZE = 200
MIN_CHUNK_OVERLAP = 0


async def load_docs_in_chunks(
    root: str,
    config: GraphRagConfig,
    select_method: str,
    limit: int,
    reporter: ProgressReporter,
    chunk_size: int = MIN_CHUNK_SIZE,
) -> list[str]:
    """Load docs into chunks for generating prompts."""
    
    # 使用异步方式加载输入数据集
    dataset = await load_input(config.input, reporter, root)

    # 将数据集封装为文本单元的输入对象
    input = VerbInput(input=TableContainer(table=dataset))
    
    # 解析并设置分块策略
    chunk_strategy = config.chunks.resolved_strategy()

    # 设置较小的分块大小，以避免生成过大的提示
    chunk_strategy["chunk_size"] = chunk_size
    chunk_strategy["chunk_overlap"] = MIN_CHUNK_OVERLAP

    # 使用分块策略对输入数据集进行分块处理
    dataset_chunks_table_container = chunk(
        input,
        column="text",
        to="chunks",
        callbacks=NoopVerbCallbacks(),
        strategy=chunk_strategy,
    )

    # 将分块后的数据集转换为 Pandas 的 DataFrame 格式
    dataset_chunks = cast(pd.DataFrame, dataset_chunks_table_container.table)

    # 从分块数据中选择具体的分块并扩展成新的 DataFrame
    chunks_df = pd.DataFrame(dataset_chunks["chunks"].explode())  # type: ignore

    # 根据选择方法和限制数量构建数据集
    if limit <= 0 or limit > len(chunks_df):
        limit = len(chunks_df)

    if select_method == "top":
        chunks_df = chunks_df[:limit]
    elif select_method == "random":
        chunks_df = chunks_df.sample(n=limit)

    # 将数据集转换为列表形式，以便得到文档列表
    return chunks_df["chunks"].tolist()
```