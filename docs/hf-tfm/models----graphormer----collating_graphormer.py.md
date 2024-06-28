# `.\models\graphormer\collating_graphormer.py`

```py
# 从 Microsoft Corporation 和 HuggingFace 中导入必要的模块
# 使用 MIT 许可证进行许可

from typing import Any, Dict, List, Mapping  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 库

from ...utils import is_cython_available, requires_backends  # 导入自定义模块，检查是否有 Cython 可用

if is_cython_available():  # 如果 Cython 可用，则导入相关模块
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from . import algos_graphormer  # noqa E402  # 导入算法相关模块，忽略 E402 错误

def convert_to_single_emb(x, offset: int = 512):
    # 将输入 x 转换为单一嵌入表示，并返回转换后的结果
    feature_num = x.shape[1] if len(x.shape) > 1 else 1  # 计算特征数量
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)  # 计算特征偏移量
    x = x + feature_offset  # 将 x 加上特征偏移量
    return x  # 返回转换后的 x

def preprocess_item(item, keep_features=True):
    # 预处理给定的项目 item，根据需要保留特征

    requires_backends(preprocess_item, ["cython"])  # 检查是否需要 Cython 支持

    if keep_features and "edge_attr" in item.keys():  # 如果需要保留特征并且有边属性
        edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)  # 转换边属性为 NumPy 数组
    else:
        edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # 否则，默认为所有边相同的嵌入

    if keep_features and "node_feat" in item.keys():  # 如果需要保留特征并且有节点特征
        node_feature = np.asarray(item["node_feat"], dtype=np.int64)  # 转换节点特征为 NumPy 数组
    else:
        node_feature = np.ones((item["num_nodes"], 1), dtype=np.int64)  # 否则，默认为所有节点相同的嵌入

    edge_index = np.asarray(item["edge_index"], dtype=np.int64)  # 转换边索引为 NumPy 数组
    input_nodes = convert_to_single_emb(node_feature) + 1  # 转换输入节点为单一嵌入表示并加一

    num_nodes = item["num_nodes"]  # 获取节点数量

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]  # 如果边属性的形状为一维，则扩展为二维

    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64)  # 初始化注意力边类型矩阵

    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1  # 设置注意力边类型

    adj = np.zeros([num_nodes, num_nodes], dtype=bool)  # 初始化邻接矩阵，布尔类型

    adj[edge_index[0], edge_index[1]] = True  # 根据边索引设置邻接矩阵的值为 True

    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)  # 计算最短路径和路径
    max_dist = np.amax(shortest_path_result)  # 计算最大距离

    input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)  # 生成输入边

    attn_bias = np.zeros([num_nodes + 1, num_nodes + 1], dtype=np.single)  # 初始化注意力偏置矩阵，单精度浮点数

    item["input_nodes"] = input_nodes + 1  # 将输入节点加一，用于填充
    item["attn_bias"] = attn_bias  # 设置注意力偏置矩阵
    item["attn_edge_type"] = attn_edge_type  # 设置注意力边类型
    item["spatial_pos"] = shortest_path_result.astype(np.int64) + 1  # 设置空间位置，加一用于填充
    item["in_degree"] = np.sum(adj, axis=1).reshape(-1) + 1  # 计算入度并加一用于填充
    item["out_degree"] = item["in_degree"]  # 对于无向图，出度等同于入度
    item["input_edges"] = input_edges + 1  # 设置输入边，加一用于填充

    if "labels" not in item:
        item["labels"] = item["y"]  # 如果没有标签，则使用 y 属性

    return item  # 返回预处理后的项目

class GraphormerDataCollator:
    # Graphormer 数据收集器类，用于收集和处理数据
    # 定义初始化方法，用于初始化对象
    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False):
        # 检查是否有 Cython 可用，如果没有则抛出 ImportError
        if not is_cython_available():
            raise ImportError("Graphormer preprocessing needs Cython (pyximport)")

        # 设置对象的空间位置最大值属性
        self.spatial_pos_max = spatial_pos_max
        # 设置对象的动态处理属性
        self.on_the_fly_processing = on_the_fly_processing
```