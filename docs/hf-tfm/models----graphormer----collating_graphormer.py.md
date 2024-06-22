# `.\models\graphormer\collating_graphormer.py`

```py
# 导入必要的库和模块
from typing import Any, Dict, List, Mapping
import numpy as np
import torch
from ...utils import is_cython_available, requires_backends

# 如果 Cython 可用，则导入相关模块
if is_cython_available():
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from . import algos_graphormer  # noqa E402

# 将输入数据转换为单个嵌入向量
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x

# 预处理数据项
def preprocess_item(item, keep_features=True):
    requires_backends(preprocess_item, ["cython"])

    # 处理边属性
    if keep_features and "edge_attr" in item.keys():  # edge_attr
        edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    else:
        edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # same embedding for all

    # 处理节点特征
    if keep_features and "node_feat" in item.keys():  # input_nodes
        node_feature = np.asarray(item["node_feat"], dtype=np.int64)
    else:
        node_feature = np.ones((item["num_nodes"], 1), dtype=np.int64)  # same embedding for all

    edge_index = np.asarray(item["edge_index"], dtype=np.int64)

    input_nodes = convert_to_single_emb(node_feature) + 1
    num_nodes = item["num_nodes"]

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    # 创建节点邻接矩阵
    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    adj[edge_index[0], edge_index[1]] = True

    # 计算最短路径和最大距离
    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
    max_dist = np.amax(shortest_path_result)

    # 生成边输入
    input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    attn_bias = np.zeros([num_nodes + 1, num_nodes + 1], dtype=np.single)  # with graph token

    # 组合数据
    item["input_nodes"] = input_nodes + 1  # we shift all indices by one for padding
    item["attn_bias"] = attn_bias
    item["attn_edge_type"] = attn_edge_type
    item["spatial_pos"] = shortest_path_result.astype(np.int64) + 1  # we shift all indices by one for padding
    item["in_degree"] = np.sum(adj, axis=1).reshape(-1) + 1  # we shift all indices by one for padding
    item["out_degree"] = item["in_degree"]  # for undirected graph
    item["input_edges"] = input_edges + 1  # we shift all indices by one for padding
    if "labels" not in item:
        item["labels"] = item["y"]

    return item

# 定义 GraphormerDataCollator 类
class GraphormerDataCollator:
    # 初始化函数，设置默认参数空间位置最大值为20，是否进行即时处理为False
    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False):
        # 检查是否有 Cython 可用
        if not is_cython_available():
            # 如果没有 Cython，抛出 ImportError 异常
            raise ImportError("Graphormer preprocessing needs Cython (pyximport)")

        # 设置空间位置最大值
        self.spatial_pos_max = spatial_pos_max
        # 设置是否进行即时处理
        self.on_the_fly_processing = on_the_fly_processing
```