# `.\graphrag\graphrag\index\verbs\graph\embed\embed_graph.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和类
"""A module containing embed_graph and run_embeddings methods definition."""
from enum import Enum
from typing import Any, cast

import networkx as nx
import pandas as pd
from datashaper import TableContainer, VerbCallbacks, VerbInput, derive_from_rows, verb

# 从本地模块导入 load_graph 函数
from graphrag.index.utils import load_graph

# 导入 NodeEmbeddings 类型定义
from .typing import NodeEmbeddings

# 定义策略类型枚举
class EmbedGraphStrategyType(str, Enum):
    """EmbedGraphStrategyType class definition."""
    node2vec = "node2vec"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'

# 定义异步谓词函数 embed_graph
@verb(name="embed_graph")
async def embed_graph(
    input: VerbInput,
    callbacks: VerbCallbacks,
    strategy: dict[str, Any],
    column: str,
    to: str,
    **kwargs,
) -> TableContainer:
    """
    Embed a graph into a vector space. The graph is expected to be in graphml format. The verb outputs a new column containing a mapping between node_id and vector.

    ## Usage
    ```yaml
    verb: embed_graph
    args:
        column: clustered_graph # The name of the column containing the graph, should be a graphml graph
        to: embeddings # The name of the column to output the embeddings to
        strategy: <strategy config> # See strategies section below
    ```py

    ## Strategies
    The embed_graph verb uses a strategy to embed the graph. The strategy is an object which defines the strategy to use. The following strategies are available:

    ### node2vec
    This strategy uses the node2vec algorithm to embed a graph. The strategy config is as follows:

    ```yaml
    strategy:
        type: node2vec
        dimensions: 1536 # Optional, The number of dimensions to use for the embedding, default: 1536
        num_walks: 10 # Optional, The number of walks to use for the embedding, default: 10
        walk_length: 40 # Optional, The walk length to use for the embedding, default: 40
        window_size: 2 # Optional, The window size to use for the embedding, default: 2
        iterations: 3 # Optional, The number of iterations to use for the embedding, default: 3
        random_seed: 86 # Optional, The random seed to use for the embedding, default: 86
    ```py
    """
    # 将输入的 DataFrame 赋值给 output_df
    output_df = cast(pd.DataFrame, input.get_input())

    # 获取策略类型，默认为 node2vec
    strategy_type = strategy.get("type", EmbedGraphStrategyType.node2vec)
    # 复制策略参数
    strategy_args = {**strategy}

    # 定义异步函数 run_strategy，用于执行策略
    async def run_strategy(row):  # noqa RUF029 async is required for interface
        return run_embeddings(strategy_type, cast(Any, row[column]), strategy_args)

    # 异步处理 DataFrame 的每一行，并执行 run_strategy 函数
    results = await derive_from_rows(
        output_df,
        run_strategy,
        callbacks=callbacks,
        num_threads=kwargs.get("num_threads", None),
    )
    # 将结果存储到输出 DataFrame 的新列中
    output_df[to] = list(results)
    # 返回包含输出 DataFrame 的 TableContainer
    return TableContainer(table=output_df)

# 定义函数 run_embeddings，用于执行嵌入策略
def run_embeddings(
    strategy: EmbedGraphStrategyType,
    graphml_or_graph: str | nx.Graph,
    args: dict[str, Any],
) -> NodeEmbeddings:
    """Run embeddings method definition."""
    # 加载指定的图形表示，可以是 GraphML 文件或已加载的图对象
    graph = load_graph(graphml_or_graph)
    
    # 根据策略选择不同的嵌入方法
    match strategy:
        # 如果策略是 node2vec
        case EmbedGraphStrategyType.node2vec:
            # 导入并运行 node2vec 策略
            from .strategies.node_2_vec import run as run_node_2_vec
            return run_node_2_vec(graph, args)
    
        # 如果策略未知或未实现
        case _:
            # 抛出值错误，指出未知的策略
            msg = f"Unknown strategy {strategy}"
            raise ValueError(msg)
```