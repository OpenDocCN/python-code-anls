# `.\graphrag\graphrag\index\verbs\graph\layout\methods\zero.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run and _create_node_position methods definitions."""

import logging  # 导入日志模块
import traceback  # 导入异常跟踪模块
from typing import Any  # 导入类型提示模块

import networkx as nx  # 导入网络图模块

from graphrag.index.graph.visualization import (  # 导入图形布局相关模块
    GraphLayout,  # 导入图形布局类
    NodePosition,  # 导入节点位置类
    get_zero_positions,  # 导入获取零位置方法
)
from graphrag.index.typing import ErrorHandlerFn  # 导入错误处理函数类型定义

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


def run(  # 定义运行方法，接收网络图、参数字典和错误处理函数作为参数
    graph: nx.Graph,  # 输入参数：网络图对象
    _args: dict[str, Any],  # 输入参数：其它参数字典
    on_error: ErrorHandlerFn,  # 输入参数：错误处理函数
) -> GraphLayout:
    """Run method definition."""
    node_clusters = []  # 存储节点群集信息的列表
    node_sizes = []  # 存储节点尺寸信息的列表

    nodes = list(graph.nodes)  # 获取网络图中所有节点的列表

    for node_id in nodes:
        node = graph.nodes[node_id]  # 获取节点的属性字典
        cluster = node.get("cluster", node.get("community", -1))  # 获取节点的群集信息或默认值
        node_clusters.append(cluster)  # 将节点群集信息添加到列表
        size = node.get("degree", node.get("size", 0))  # 获取节点的度或尺寸信息或默认值
        node_sizes.append(size)  # 将节点尺寸信息添加到列表

    additional_args = {}  # 存储额外参数的字典
    if len(node_clusters) > 0:
        additional_args["node_categories"] = node_clusters  # 若存在节点群集信息，则存入额外参数字典
    if len(node_sizes) > 0:
        additional_args["node_sizes"] = node_sizes  # 若存在节点尺寸信息，则存入额外参数字典

    try:
        return get_zero_positions(node_labels=nodes, **additional_args)  # 调用获取零位置的方法并返回结果
    except Exception as e:
        log.exception("Error running zero-position")  # 记录异常信息到日志
        on_error(e, traceback.format_exc(), None)  # 调用错误处理函数处理异常
        # Umap may fail due to input sparseness or memory pressure.
        # For now, in these cases, we'll just return a layout with all nodes at (0, 0)
        result = []  # 存储结果的列表
        for i in range(len(nodes)):
            cluster = node_clusters[i] if len(node_clusters) > 0 else 1  # 获取节点群集信息或默认值
            result.append(
                NodePosition(x=0, y=0, label=nodes[i], size=0, cluster=str(cluster))
                # 将节点位置信息加入结果列表
            )
        return result  # 返回生成的节点位置信息列表作为最终结果
```