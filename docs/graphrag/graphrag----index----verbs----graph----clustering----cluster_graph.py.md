# `.\graphrag\graphrag\index\verbs\graph\clustering\cluster_graph.py`

```py
# 导入必要的库和模块
import logging  # 导入日志记录模块
from enum import Enum  # 导入枚举类型支持
from random import Random  # 导入随机数生成器
from typing import Any, cast  # 导入类型提示相关模块

import networkx as nx  # 导入网络图操作库
import pandas as pd  # 导入数据处理库
from datashaper import TableContainer, VerbCallbacks, VerbInput, progress_iterable, verb  # 导入数据格式化相关模块

from graphrag.index.utils import gen_uuid, load_graph  # 导入图形操作工具函数

from .typing import Communities  # 导入社区类型的类型提示

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@verb(name="cluster_graph")  # 注册名为cluster_graph的自定义操作
def cluster_graph(
    input: VerbInput,  # 输入参数，数据格式化相关
    callbacks: VerbCallbacks,  # 回调函数，用于操作进度更新等
    strategy: dict[str, Any],  # 策略参数，定义了聚类算法的配置
    column: str,  # 输入数据中包含图形的列名
    to: str,  # 输出聚类后图形的列名
    level_to: str | None = None,  # 可选参数，输出层级信息的列名，默认为None
    **_kwargs,  # 其他未命名参数，不做处理
) -> TableContainer:  # 返回值为数据格式化容器
    """
    Apply a hierarchical clustering algorithm to a graph. The graph is expected to be in graphml format. The verb outputs a new column containing the clustered graph, and a new column containing the level of the graph.

    ## Usage
    ```yaml
    verb: cluster_graph
    args:
        column: entity_graph # The name of the column containing the graph, should be a graphml graph
        to: clustered_graph # The name of the column to output the clustered graph to
        level_to: level # The name of the column to output the level to
        strategy: <strategy config> # See strategies section below
    ```py

    ## Strategies
    The cluster graph verb uses a strategy to cluster the graph. The strategy is a json object which defines the strategy to use. The following strategies are available:

    ### leiden
    This strategy uses the leiden algorithm to cluster a graph. The strategy config is as follows:
    ```yaml
    strategy:
        type: leiden
        max_cluster_size: 10 # Optional, The max cluster size to use, default: 10
        use_lcc: true # Optional, if the largest connected component should be used with the leiden algorithm, default: true
        seed: 0xDEADBEEF # Optional, the seed to use for the leiden algorithm, default: 0xDEADBEEF
        levels: [0, 1] # Optional, the levels to output, default: all the levels detected

    ```py
    """
    output_df = cast(pd.DataFrame, input.get_input())  # 获取输入数据并转换为DataFrame格式

    # 对每个图应用布局运行策略，返回聚类结果
    results = output_df[column].apply(lambda graph: run_layout(strategy, graph))

    community_map_to = "communities"  # 社区映射结果列名
    output_df[community_map_to] = results  # 将聚类结果存入数据框中

    level_to = level_to or f"{to}_level"  # 确定输出层级信息的列名，默认为to_level
    # 计算每行数据中社区的层级信息并存入数据框
    output_df[level_to] = output_df.apply(
        lambda x: list({level for level, _, _ in x[community_map_to]}), axis=1
    )
    output_df[to] = [None] * len(output_df)  # 初始化聚类图结果列为空值

    num_total = len(output_df)  # 数据框总行数

    # 遍历数据框中的每一行，为每个图构建层级-社区映射
    graph_level_pairs_column: list[list[tuple[int, str]]] = []
    for _, row in progress_iterable(
        output_df.iterrows(), callbacks.progress, num_total
        # 显示进度并迭代处理每一行数据
    ):
        # 获取当前行中的级别数据
        levels = row[level_to]
        # 初始化存储级别和对应图形的列表
        graph_level_pairs: list[tuple[int, str]] = []

        # 遍历每个级别，生成对应的图形并添加到列表中
        for level in levels:
            # 应用聚类算法并生成 GraphML 格式的图形数据
            graph = "\n".join(
                nx.generate_graphml(
                    apply_clustering(
                        cast(str, row[column]),  # 将列转换为字符串
                        cast(Communities, row[community_map_to]),  # 将社区映射转换为 Communities 类型
                        level,  # 当前处理的级别
                    )
                )
            )
            # 将当前级别和对应的图形数据组成元组，添加到列表中
            graph_level_pairs.append((level, graph))
        # 将每个 (级别, 图形数据) 对的列表添加到输出数据框中
        graph_level_pairs_column.append(graph_level_pairs)
    # 将含有多个 (级别, 图形数据) 列表的列展开为多行，忽略索引
    output_df = output_df.explode(to, ignore_index=True)

    # 将展开后的 (级别, 图形数据) 对分隔成单独的列
    # TODO: 可能有更好的方法来实现这一步骤
    output_df[[level_to, to]] = pd.DataFrame(
        output_df[to].tolist(), index=output_df.index
    )

    # 清理不再需要的社区映射列
    output_df.drop(columns=[community_map_to], inplace=True)

    # 返回包含处理后数据的 TableContainer 对象
    return TableContainer(table=output_df)
# TODO: This should support str | nx.Graph as a graphml param
def apply_clustering(
    graphml: str, communities: Communities, level=0, seed=0xF001
) -> nx.Graph:
    """Apply clustering to a graphml string."""
    random = Random(seed)  # noqa S311
    # 解析输入的 GraphML 字符串，返回一个 NetworkX 图对象
    graph = nx.parse_graphml(graphml)
    # 遍历每个社区的层级、ID和节点
    for community_level, community_id, nodes in communities:
        # 如果当前层级匹配指定层级，则为每个节点添加集群信息和层级信息
        if level == community_level:
            for node in nodes:
                graph.nodes[node]["cluster"] = community_id
                graph.nodes[node]["level"] = level

    # 添加节点的度数信息
    for node_degree in graph.degree:
        graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])

    # 为每个节点生成人类可读的 ID 和递增的记录 ID（在最终报告中作为参考）
    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["human_readable_id"] = index
        graph.nodes[node]["id"] = str(gen_uuid(random))

    # 添加边的唯一 ID 和人类可读的 ID，以及层级信息
    for index, edge in enumerate(graph.edges()):
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level
    return graph


class GraphCommunityStrategyType(str, Enum):
    """GraphCommunityStrategyType class definition."""

    leiden = "leiden"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


def run_layout(
    strategy: dict[str, Any], graphml_or_graph: str | nx.Graph
) -> Communities:
    """Run layout method definition."""
    # 根据输入的图形描述（GraphML 或 NetworkX 图对象）加载图形数据
    graph = load_graph(graphml_or_graph)
    # 如果图中没有节点，记录警告信息并返回空列表
    if len(graph.nodes) == 0:
        log.warning("Graph has no nodes")
        return []

    # 初始化聚类结果字典
    clusters: dict[int, dict[str, list[str]]] = {}
    # 获取策略的类型，若未指定，默认为 Leiden 算法
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)
    # 根据策略类型选择不同的聚类算法进行计算
    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            from .strategies.leiden import run as run_leiden
            # 调用 Leiden 算法执行聚类
            clusters = run_leiden(graph, strategy)
        case _:
            msg = f"Unknown clustering strategy {strategy_type}"
            raise ValueError(msg)

    # 将聚类结果格式化为指定的 Communities 类型并返回
    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, nodes))
    return results
```