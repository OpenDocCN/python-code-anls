# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hdbscan\_linkage.pyx`

```
# Minimum spanning tree single linkage implementation for hdbscan
# Authors: Leland McInnes <leland.mcinnes@gmail.com>
#          Steve Astels <sastels@gmail.com>
#          Meekail Zain <zainmeekail@gmail.com>
# Copyright (c) 2015, Leland McInnes
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cimport numpy as cnp
from libc.float cimport DBL_MAX

import numpy as np
from ...metrics._dist_metrics cimport DistanceMetric64
from ...cluster._hierarchical_fast cimport UnionFind
from ...cluster._hdbscan._tree cimport HIERARCHY_t
from ...cluster._hdbscan._tree import HIERARCHY_dtype
from ...utils._typedefs cimport intp_t, float64_t, int64_t, uint8_t

cnp.import_array()

cdef extern from "numpy/arrayobject.h":
    intp_t * PyArray_SHAPE(cnp.PyArrayObject *)

# Numpy structured dtype representing a single ordered edge in Prim's algorithm
MST_edge_dtype = np.dtype([
    ("current_node", np.int64),
    ("next_node", np.int64),
    ("distance", np.float64),
])

# Packed shouldn't make a difference since they're all 8-byte quantities,
# but it's included just to be safe.
ctypedef packed struct MST_edge_t:
    int64_t current_node
    int64_t next_node
    float64_t distance

cpdef cnp.ndarray[MST_edge_t, ndim=1, mode='c'] mst_from_mutual_reachability(
    cnp.ndarray[float64_t, ndim=2] mutual_reachability
):
    """Compute the Minimum Spanning Tree (MST) representation of the mutual-
    reachability graph using Prim's algorithm.

    Parameters
    ----------
    mutual_reachability : 2D numpy array, shape (n_samples, n_samples)
        Mutual reachability matrix representing pairwise distances between points.

    Returns
    -------
    mst : 1D numpy structured array of MST_edge_t
        Array representing the edges of the Minimum Spanning Tree.

    Notes
    -----
    This function computes the MST using Prim's algorithm on the mutual reachability graph.
    Each edge in the MST is represented as a structured numpy array with fields for
    current_node, next_node, and distance.

    """
    # mutual_reachability 是一个形状为 (n_samples, n_samples) 的 ndarray，存储样本间的互相可达性。

    Returns
    -------
    mst : 形状为 (n_samples - 1,)，数据类型为 MST_edge_dtype 的 ndarray
        mutual-reachability 图的最小生成树（MST）表示。MST 被表示为一组边的集合。
    """
    cdef:
        # 注意：我们利用 ndarray 而非内存视图以便在下面使用 numpy 的二进制索引和子选择。
        cnp.ndarray[int64_t, ndim=1, mode='c'] current_labels
        cnp.ndarray[float64_t, ndim=1, mode='c'] min_reachability, left, right
        cnp.ndarray[MST_edge_t, ndim=1, mode='c'] mst

        cnp.ndarray[uint8_t, mode='c'] label_filter

        int64_t n_samples = PyArray_SHAPE(<cnp.PyArrayObject*> mutual_reachability)[0]
        int64_t current_node, new_node_index, new_node, i

    # 创建一个形状为 (n_samples - 1,)，数据类型为 MST_edge_dtype 的空数组来存储 MST
    mst = np.empty(n_samples - 1, dtype=MST_edge_dtype)
    # 创建一个长度为 n_samples 的整数数组，包含从 0 到 n_samples-1 的整数
    current_labels = np.arange(n_samples, dtype=np.int64)
    # 初始化当前节点为第一个节点
    current_node = 0
    # 创建一个长度为 n_samples 的浮点数数组，每个元素初始化为正无穷
    min_reachability = np.full(n_samples, fill_value=np.inf, dtype=np.float64)
    # 迭代 n_samples - 1 次，构建 MST
    for i in range(0, n_samples - 1):
        # 根据当前节点过滤掉与当前节点相同的标签
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]
        # 获取剩余节点的最小可达性数组
        left = min_reachability[label_filter]
        # 获取当前节点到剩余节点的可达性数组
        right = mutual_reachability[current_node][current_labels]
        # 计算最小可达性数组
        min_reachability = np.minimum(left, right)

        # 找出最小可达性数组中的最小值所对应的新节点索引
        new_node_index = np.argmin(min_reachability)
        # 根据索引找出对应的新节点
        new_node = current_labels[new_node_index]
        # 将当前节点与新节点及其距离存储为 MST 中的一条边
        mst[i].current_node = current_node
        mst[i].next_node = new_node
        mst[i].distance = min_reachability[new_node_index]
        # 更新当前节点为新节点，准备处理下一条边
        current_node = new_node

    # 返回构建好的最小生成树 MST
    return mst
    # 声明一个一维的 NumPy 数组，用于表示节点是否已经在生成树中，每个节点对应一个字节
    cdef:
        uint8_t[::1] in_tree
        # 声明一个一维的 NumPy 数组，用于存储每个节点的最小可达距离
        float64_t[::1] min_reachability
        # 声明一个一维的 NumPy 数组，用于存储当前正在处理的源节点
        int64_t[::1] current_sources
        # 声明一个二维的 NumPy 数组，用于存储最小生成树的边集合，每个元素为 MST_edge_t 类型，以 C 顺序访问
        cnp.ndarray[MST_edge_t, ndim=1, mode='c'] mst

        # 声明一些变量来存储当前节点、源节点、新节点等信息
        int64_t current_node, source_node, new_node, next_node_source
        # 声明变量 i, j，以及样本数和特征数
        int64_t i, j, n_samples, num_features

        # 声明一些变量来存储当前节点的核心距离、新的可达距离等信息
        float64_t current_node_core_dist, new_reachability, mutual_reachability_distance
        float64_t next_node_min_reach, pair_distance, next_node_core_dist

    # 获取数据集的样本数和特征数
    n_samples = raw_data.shape[0]
    num_features = raw_data.shape[1]

    # 初始化一个空的最小生成树数组，长度为样本数减一，数据类型为 MST_edge_dtype
    mst = np.empty(n_samples - 1, dtype=MST_edge_dtype)

    # 初始化一个数组，表示哪些节点已经在生成树中，全部初始化为 0 (False)
    in_tree = np.zeros(n_samples, dtype=np.uint8)
    # 初始化一个数组，用于存储每个节点的最小可达距离，全部初始化为无穷大
    min_reachability = np.full(n_samples, fill_value=np.inf, dtype=np.float64)
    # 初始化一个数组，用于表示当前正在处理的源节点，默认所有节点都是源节点
    current_sources = np.ones(n_samples, dtype=np.int64)

    # 初始化当前节点为第一个节点
    current_node = 0
    # 遍历样本数据，从第一个样本到倒数第二个样本
    for i in range(0, n_samples - 1):

        # 将当前节点标记为已访问
        in_tree[current_node] = 1

        # 获取当前节点的核心距离
        current_node_core_dist = core_distances[current_node]

        # 初始化新的可达距离、源节点和新节点
        new_reachability = DBL_MAX
        source_node = 0
        new_node = 0

        # 遍历所有样本
        for j in range(n_samples):
            # 如果节点已在树中，则跳过
            if in_tree[j]:
                continue

            # 获取下一个节点的最小可达距离和源节点
            next_node_min_reach = min_reachability[j]
            next_node_source = current_sources[j]

            # 计算当前节点和下一个节点之间的距离
            pair_distance = dist_metric.dist(
                &raw_data[current_node, 0],
                &raw_data[j, 0],
                num_features
            )

            # 将距离除以 alpha
            pair_distance /= alpha

            # 获取下一个节点的核心距离
            next_node_core_dist = core_distances[j]

            # 计算互达距离
            mutual_reachability_distance = max(
                current_node_core_dist,
                next_node_core_dist,
                pair_distance
            )

            # 如果互达距离大于下一个节点的最小可达距离
            if mutual_reachability_distance > next_node_min_reach:
                # 如果下一个节点的最小可达距离小于新的可达距离
                if next_node_min_reach < new_reachability:
                    new_reachability = next_node_min_reach
                    source_node = next_node_source
                    new_node = j
                continue

            # 如果互达距离小于下一个节点的最小可达距离
            if mutual_reachability_distance < next_node_min_reach:
                min_reachability[j] = mutual_reachability_distance
                current_sources[j] = current_node
                # 如果互达距离小于新的可达距离
                if mutual_reachability_distance < new_reachability:
                    new_reachability = mutual_reachability_distance
                    source_node = current_node
                    new_node = j
            else:
                # 如果下一个节点的最小可达距离小于新的可达距离
                if next_node_min_reach < new_reachability:
                    new_reachability = next_node_min_reach
                    source_node = next_node_source
                    new_node = j

        # 更新最小生成树的当前节点、下一个节点和距离
        mst[i].current_node = source_node
        mst[i].next_node = new_node
        mst[i].distance = new_reachability
        current_node = new_node

    # 返回最小生成树
    return mst
# 定义一个 Cython 函数，构建单链接树从最小生成树（MST）中
cpdef cnp.ndarray[HIERARCHY_t, ndim=1, mode="c"] make_single_linkage(const MST_edge_t[::1] mst):
    """Construct a single-linkage tree from an MST.

    Parameters
    ----------
    mst : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges.

    Returns
    -------
    single_linkage : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree (dendrogram) built from the MST. Each
        element of the array represents:
        - left node/cluster
        - right node/cluster
        - distance
        - new cluster size
    """
    cdef:
        cnp.ndarray[HIERARCHY_t, ndim=1, mode="c"] single_linkage
        # 注意 mst.shape[0] 比样本数少一个
        int64_t n_samples = mst.shape[0] + 1
        intp_t current_node_cluster, next_node_cluster
        int64_t current_node, next_node, i
        float64_t distance
        UnionFind U = UnionFind(n_samples)

    # 初始化一个全零数组，用于存储单链接树的结果
    single_linkage = np.zeros(n_samples - 1, dtype=HIERARCHY_dtype)

    # 遍历每条最小生成树的边
    for i in range(n_samples - 1):
        # 获取当前边的起始节点、目标节点和距离
        current_node = mst[i].current_node
        next_node = mst[i].next_node
        distance = mst[i].distance

        # 使用并查集快速查找当前节点所属的簇和目标节点所属的簇
        current_node_cluster = U.fast_find(current_node)
        next_node_cluster = U.fast_find(next_node)

        # 将簇的信息存入单链接树的当前位置
        single_linkage[i].left_node = current_node_cluster
        single_linkage[i].right_node = next_node_cluster
        single_linkage[i].value = distance
        # 更新新簇的大小
        single_linkage[i].cluster_size = U.size[current_node_cluster] + U.size[next_node_cluster]

        # 将当前节点所在的簇和目标节点所在的簇进行合并
        U.union(current_node_cluster, next_node_cluster)

    # 返回构建好的单链接树
    return single_linkage
```