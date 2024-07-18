# `.\graphrag\graphrag\index\verbs\graph\layout\layout_graph.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing layout_graph, _run_layout and _apply_layout_to_graph methods definition."""

from enum import Enum  # 导入枚举类型的支持
from typing import Any, cast  # 导入类型提示相关的模块

import networkx as nx  # 导入网络图处理库 NetworkX
import pandas as pd  # 导入数据处理库 Pandas
from datashaper import TableContainer, VerbCallbacks, VerbInput, progress_callback, verb  # 导入数据处理相关模块和自定义的回调函数

from graphrag.index.graph.visualization import GraphLayout  # 从本地导入图形布局相关模块
from graphrag.index.utils import load_graph  # 从本地导入加载图形数据的模块
from graphrag.index.verbs.graph.embed.typing import NodeEmbeddings  # 导入节点嵌入的类型定义


class LayoutGraphStrategyType(str, Enum):
    """LayoutGraphStrategyType class definition."""
    
    umap = "umap"  # 定义布局策略类型为 umap
    zero = "zero"  # 定义布局策略类型为 zero

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'  # 返回枚举值的字符串表示形式


@verb(name="layout_graph")
def layout_graph(
    input: VerbInput,  # 输入参数类型为 VerbInput 类
    callbacks: VerbCallbacks,  # 回调函数参数类型为 VerbCallbacks 类
    strategy: dict[str, Any],  # 策略参数为字典，键为字符串，值为任意类型
    embeddings_column: str,  # 嵌入列的名称，字符串类型
    graph_column: str,  # 图形列的名称，字符串类型
    to: str,  # 输出列的名称，字符串类型
    graph_to: str | None = None,  # 输出图形的列的名称，可选字符串或空
    **_kwargs: dict,  # 其他关键字参数，字典类型
) -> TableContainer:
    """
    Apply a layout algorithm to a graph. The graph is expected to be in graphml format. The verb outputs a new column containing the laid out graph.

    ## Usage
    ```yaml
    verb: layout_graph
    args:
        graph_column: clustered_graph # The name of the column containing the graph, should be a graphml graph
        embeddings_column: embeddings # The name of the column containing the embeddings
        to: node_positions # The name of the column to output the node positions to
        graph_to: positioned_graph # The name of the column to output the positioned graph to
        strategy: <strategy config> # See strategies section below
    ```py

    ## Strategies
    The layout graph verb uses a strategy to layout the graph. The strategy is a json object which defines the strategy to use. The following strategies are available:

    ### umap
    This strategy uses the umap algorithm to layout a graph. The strategy config is as follows:
    ```yaml
    strategy:
        type: umap
        n_neighbors: 5 # Optional, The number of neighbors to use for the umap algorithm, default: 5
        min_dist: 0.75 # Optional, The min distance to use for the umap algorithm, default: 0.75
    ```py
    """
    output_df = cast(pd.DataFrame, input.get_input())  # 从输入参数中获取 DataFrame 数据

    num_items = len(output_df)  # 获取 DataFrame 中条目的数量
    strategy_type = strategy.get("type", LayoutGraphStrategyType.umap)  # 获取布局策略的类型，默认为 umap
    strategy_args = {**strategy}  # 获取并复制所有策略参数

    has_embeddings = embeddings_column in output_df.columns  # 检查 DataFrame 是否包含嵌入列

    # 对 DataFrame 中的每一行应用进度回调和布局运行函数
    layouts = output_df.apply(
        progress_callback(
            lambda row: _run_layout(
                strategy_type,
                row[graph_column],  # 当前行的图形数据
                row[embeddings_column] if has_embeddings else {},  # 当前行的嵌入数据（如果有）
                strategy_args,  # 使用的布局策略参数
                callbacks,  # 回调函数
            ),
            callbacks.progress,  # 进度回调函数
            num_items,  # 总条目数
        ),
        axis=1,  # 沿行方向进行应用
    )
    output_df[to] = layouts.apply(lambda layout: [pos.to_pandas() for pos in layout])  # 将布局结果保存到指定输出列
    # 如果指定了 graph_to 参数，则将计算的图形布局应用到输出数据框的指定列
    output_df[graph_to] = output_df.apply(
        # 对每一行应用 _apply_layout_to_graph 函数，将图形列和对应的布局类型作为参数传入
        lambda row: _apply_layout_to_graph(
            row[graph_column], cast(GraphLayout, layouts[row.name])
        ),
        # 按行(axis=1)进行应用
        axis=1,
    )
    # 返回一个包含表格的容器，表格内容为输出数据框
    return TableContainer(table=output_df)
# 定义一个函数 `_run_layout`，用于根据给定的策略对图进行布局，并返回布局结果。
def _run_layout(
    strategy: LayoutGraphStrategyType,
    graphml_or_graph: str | nx.Graph,
    embeddings: NodeEmbeddings,
    args: dict[str, Any],
    reporter: VerbCallbacks,
) -> GraphLayout:
    # 载入图数据，根据输入参数确定是从 GraphML 文件载入还是直接使用 NetworkX 图对象
    graph = load_graph(graphml_or
```