# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hdbscan\_tree.pyx`

```
```python`
# Tree handling (condensing, finding stable clusters) for hdbscan
# Authors: Leland McInnes
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

# 导入 numpy 库的 c 语言接口
cimport numpy as cnp
# 从 libc.math 模块导入 isinf 函数，检查浮点数是否为无穷大
from libc.math cimport isinf
# 导入 cython 模块
import cython

# 导入 numpy 库
import numpy as np

# 初始化 numpy 的 C 数组接口
cnp.import_array()

# 定义外部函数，访问 numpy 的 arrayobject.h 头文件中的 PyArray_SHAPE 函数
cdef extern from "numpy/arrayobject.h":
    intp_t * PyArray_SHAPE(cnp.PyArrayObject *)

# 定义一个常量 INFTY，值为 numpy 的无穷大
cdef cnp.float64_t INFTY = np.inf
# 定义一个常量 NOISE，值为 -1，表示噪声点
cdef cnp.intp_t NOISE = -1

# 定义 HIERARCHY_dtype 数据类型，用于存储树的层次结构信息
HIERARCHY_dtype = np.dtype([
    ("left_node", np.intp),  # 左子节点的索引
    ("right_node", np.intp),  # 右子节点的索引
    ("value", np.float64),    # 连接的距离或相似度
    ("cluster_size", np.intp),# 当前簇的大小
])

# 定义 CONDENSED_dtype 数据类型，用于存储压缩后的树信息
CONDENSED_dtype = np.dtype([
    ("parent", np.intp),      # 父节点的索引
    ("child", np.intp),       # 子节点的索引
    ("value", np.float64),    # 连接的距离或相似度
    ("cluster_size", np.intp),# 当前簇的大小
])

# 定义一个函数，转换树结构为标签数组，处理稳定簇的选择
cpdef tuple tree_to_labels(
    const HIERARCHY_t[::1] single_linkage_tree,  # 输入的单链接树
    cnp.intp_t min_cluster_size=10,             # 最小簇大小，默认为 10
    cluster_selection_method="eom",             # 聚类选择方法，默认为 "eom"
    bint allow_single_cluster=False,             # 是否允许单一簇，默认为 False
    cnp.float64_t cluster_selection_epsilon=0.0,  # 聚类选择的容差值，默认为 0.0
    max_cluster_size=None,                       # 最大簇大小，默认为 None
):
    cdef:
        cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] condensed_tree  # 定义压缩树数组
        cnp.ndarray[cnp.intp_t, ndim=1, mode='c'] labels            # 定义标签数组
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] probabilities   # 定义概率数组

    # 使用 _condense_tree 函数将单链接树压缩为 condensed_tree
    condensed_tree = _condense_tree(single_linkage_tree, min_cluster_size)
    # 使用 _get_clusters 函数计算聚类标签和概率
    labels, probabilities = _get_clusters(
        condensed_tree,
        _compute_stability(condensed_tree),  # 计算稳定性
        cluster_selection_method,            # 聚类选择方法
        allow_single_cluster,                # 是否允许单一簇
        cluster_selection_epsilon,           # 聚类选择容差值
        max_cluster_size,                    # 最大簇大小
    )

    # 返回标签数组和概率数组作为元组
    return (labels, probabilities)
cdef list bfs_from_hierarchy(
    const HIERARCHY_t[::1] hierarchy,
    cnp.intp_t bfs_root
):
    """
    Perform a breadth first search on a tree in scipy hclust format.
    """

    cdef list process_queue, next_queue, result
    cdef cnp.intp_t n_samples = hierarchy.shape[0] + 1
    cdef cnp.intp_t node
    process_queue = [bfs_root]
    result = []

    while process_queue:
        result.extend(process_queue)
        # By construction, node i is formed by the union of nodes
        # hierarchy[i - n_samples, 0] and hierarchy[i - n_samples, 1]
        # 将当前处理队列中节点扩展为其子节点
        process_queue = [
            x - n_samples  # 转换成实际节点索引
            for x in process_queue
            if x >= n_samples  # 只考虑有效的节点
        ]
        if process_queue:
            next_queue = []
            for node in process_queue:
                next_queue.extend(
                    [
                        hierarchy[node].left_node,   # 添加左子节点
                        hierarchy[node].right_node,  # 添加右子节点
                    ]
                )
            process_queue = next_queue  # 更新当前处理队列为下一层的节点

    return result  # 返回广度优先搜索结果列表
        

cpdef cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] _condense_tree(
    const HIERARCHY_t[::1] hierarchy,
    cnp.intp_t min_cluster_size=10
):
    """Condense a tree according to a minimum cluster size. This is akin
    to the runt pruning procedure of Stuetzle. The result is a much simpler
    tree that is easier to visualize. We include extra information on the
    lambda value at which individual points depart clusters for later
    analysis and computation.

    Parameters
    ----------
    hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype
        A single linkage hierarchy in scipy.cluster.hierarchy format.

    min_cluster_size : int, optional (default 10)
        The minimum size of clusters to consider. Clusters smaller than this
        are pruned from the tree.

    Returns
    -------
    condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype
        Effectively an edgelist encoding a parent/child pair, along with a
        value and the corresponding cluster_size in each row providing a tree
        structure.
    """

    cdef:
        cnp.intp_t root = 2 * hierarchy.shape[0]  # 根节点索引
        cnp.intp_t n_samples = hierarchy.shape[0] + 1  # 样本数
        cnp.intp_t next_label = n_samples + 1  # 下一个标签
        list result_list, node_list = bfs_from_hierarchy(hierarchy, root)  # 执行广度优先搜索

        cnp.intp_t[::1] relabel  # 重新标记数组
        cnp.uint8_t[::1] ignore  # 忽略标记数组

        cnp.intp_t node, sub_node, left, right  # 节点相关变量
        cnp.float64_t lambda_value, distance  # λ值和距离
        cnp.intp_t left_count, right_count  # 左右子节点数目
        HIERARCHY_t children  # 子节点结构体

    relabel = np.empty(root + 1, dtype=np.intp)  # 初始化重新标记数组
    relabel[root] = n_samples  # 设置根节点重新标记值为样本数
    result_list = []  # 初始化结果列表
    ignore = np.zeros(len(node_list), dtype=bool)  # 初始化忽略标记数组
    # 遍历节点列表中的每一个节点
    for node in node_list:
        # 如果 ignore 中的节点被标记为忽略或者节点编号小于样本数，则跳过当前循环
        if ignore[node] or node < n_samples:
            continue

        # 从层次结构中获取当前节点的子节点信息
        children = hierarchy[node - n_samples]
        # 获取左子节点编号
        left = children.left_node
        # 获取右子节点编号
        right = children.right_node
        # 获取子节点之间的距离值
        distance = children.value

        # 根据距离值计算 lambda 值，若距离大于 0，则 lambda 值为 1.0 / distance，否则为无穷大
        if distance > 0.0:
            lambda_value = 1.0 / distance
        else:
            lambda_value = INFTY

        # 如果左子节点编号大于等于样本数，则获取左子节点的聚类大小，否则默认为 1
        if left >= n_samples:
            left_count = hierarchy[left - n_samples].cluster_size
        else:
            left_count = 1

        # 如果右子节点编号大于等于样本数，则获取右子节点的聚类大小，否则默认为 1
        if right >= n_samples:
            right_count = hierarchy[right - n_samples].cluster_size
        else:
            right_count = 1

        # 如果左右子节点的聚类大小都大于等于指定的最小聚类大小
        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            # 给左子节点重新标记一个下一个可用的标签
            relabel[left] = next_label
            next_label += 1
            # 将结果添加到结果列表中，包括节点重标记后的值、左子节点的重标记值、lambda 值以及左子节点的聚类大小
            result_list.append(
                (relabel[node], relabel[left], lambda_value, left_count)
            )

            # 给右子节点重新标记一个下一个可用的标签
            relabel[right] = next_label
            next_label += 1
            # 将结果添加到结果列表中，包括节点重标记后的值、右子节点的重标记值、lambda 值以及右子节点的聚类大小
            result_list.append(
                (relabel[node], relabel[right], lambda_value, right_count)
            )

        # 如果左右子节点的聚类大小都小于指定的最小聚类大小
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            # 从左子节点开始的广度优先搜索，并将结果添加到结果列表中
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < n_samples:
                    result_list.append(
                        (relabel[node], sub_node, lambda_value, 1)
                    )
                # 将搜索到的子节点标记为已忽略
                ignore[sub_node] = True

            # 从右子节点开始的广度优先搜索，并将结果添加到结果列表中
            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < n_samples:
                    result_list.append(
                        (relabel[node], sub_node, lambda_value, 1)
                    )
                # 将搜索到的子节点标记为已忽略
                ignore[sub_node] = True

        # 如果只有左子节点的聚类大小小于指定的最小聚类大小
        elif left_count < min_cluster_size:
            # 将右子节点标记为与当前节点相同的重标记值
            relabel[right] = relabel[node]
            # 从左子节点开始的广度优先搜索，并将结果添加到结果列表中
            for sub_node in bfs_from_hierarchy(hierarchy, left):
                if sub_node < n_samples:
                    result_list.append(
                        (relabel[node], sub_node, lambda_value, 1)
                    )
                # 将搜索到的子节点标记为已忽略
                ignore[sub_node] = True

        # 如果只有右子节点的聚类大小小于指定的最小聚类大小
        else:
            # 将左子节点标记为与当前节点相同的重标记值
            relabel[left] = relabel[node]
            # 从右子节点开始的广度优先搜索，并将结果添加到结果列表中
            for sub_node in bfs_from_hierarchy(hierarchy, right):
                if sub_node < n_samples:
                    result_list.append(
                        (relabel[node], sub_node, lambda_value, 1)
                    )
                # 将搜索到的子节点标记为已忽略
                ignore[sub_node] = True

    # 将结果列表转换为 NumPy 数组，并指定数据类型为 CONDENSED_dtype
    return np.array(result_list, dtype=CONDENSED_dtype)
# 定义一个Cython函数，计算稳定性指标，返回一个字典
cdef dict _compute_stability(
    cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] condensed_tree
):

    cdef:
        cnp.float64_t[::1] result, births  # 结果数组和出生数组
        cnp.intp_t[:] parents = condensed_tree['parent']  # 父节点数组

        cnp.intp_t parent, cluster_size, result_index, idx  # 父节点、簇大小、结果索引和迭代索引
        cnp.float64_t lambda_val  # Lambda 值
        CONDENSED_t condensed_node  # 精简树节点
        cnp.intp_t largest_child = condensed_tree['child'].max()  # 最大子节点
        cnp.intp_t smallest_cluster = np.min(parents)  # 最小簇标签
        cnp.intp_t num_clusters = np.max(parents) - smallest_cluster + 1  # 簇的数量
        dict stability_dict = {}  # 稳定性字典，用于存储结果

    largest_child = max(largest_child, smallest_cluster)  # 确保最大子节点不小于最小簇标签
    births = np.full(largest_child + 1, np.nan, dtype=np.float64)  # 初始化出生数组，NaN填充

    # 遍历精简树，填充出生数组
    for idx in range(PyArray_SHAPE(<cnp.PyArrayObject*> condensed_tree)[0]):
        condensed_node = condensed_tree[idx]
        births[condensed_node.child] = condensed_node.value

    births[smallest_cluster] = 0.0  # 最小簇标签的出生时间设为0.0

    result = np.zeros(num_clusters, dtype=np.float64)  # 初始化结果数组
    # 计算稳定性指标的主循环
    for idx in range(PyArray_SHAPE(<cnp.PyArrayObject*> condensed_tree)[0]):
        condensed_node = condensed_tree[idx]
        parent = condensed_node.parent
        lambda_val = condensed_node.value
        cluster_size = condensed_node.cluster_size

        result_index = parent - smallest_cluster  # 计算结果数组的索引
        result[result_index] += (lambda_val - births[parent]) * cluster_size  # 更新结果数组中的值

    # 将结果放入稳定性字典
    for idx in range(num_clusters):
        stability_dict[idx + smallest_cluster] = result[idx]

    return stability_dict  # 返回稳定性字典


# 定义一个Cython函数，从簇树开始进行广度优先搜索，并返回遍历结果的列表
cdef list bfs_from_cluster_tree(
    cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] condensed_tree,
    cnp.intp_t bfs_root
):

    cdef:
        list result = []  # 结果列表
        cnp.ndarray[cnp.intp_t, ndim=1] process_queue = (
            np.array([bfs_root], dtype=np.intp)  # 处理队列，初始包含根节点
        )
        cnp.ndarray[cnp.intp_t, ndim=1] children = condensed_tree['child']  # 子节点数组
        cnp.intp_t[:] parents = condensed_tree['parent']  # 父节点数组

    while len(process_queue) > 0:  # 循环直到处理队列为空
        result.extend(process_queue.tolist())  # 将处理队列中的节点添加到结果列表
        process_queue = children[np.isin(parents, process_queue)]  # 更新处理队列为当前处理队列节点的子节点

    return result  # 返回广度优先搜索结果的列表


# 定义一个Cython函数，计算每个父节点的最大 Lambda 值，返回一个数组
cdef cnp.float64_t[::1] max_lambdas(cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] condensed_tree):

    cdef:
        cnp.intp_t parent, current_parent, idx  # 父节点、当前父节点和迭代索引
        cnp.float64_t lambda_val, max_lambda  # Lambda 值和最大 Lambda 值
        cnp.float64_t[::1] deaths  # 死亡数组
        cnp.intp_t largest_parent = condensed_tree['parent'].max()  # 最大父节点

    deaths = np.zeros(largest_parent + 1, dtype=np.float64)  # 初始化死亡数组

    current_parent = condensed_tree[0].parent  # 初始化当前父节点
    max_lambda = condensed_tree[0].value  # 初始化最大 Lambda 值

    # 遍历精简树，计算每个父节点的最大 Lambda 值
    for idx in range(1, PyArray_SHAPE(<cnp.PyArrayObject*> condensed_tree)[0]):
        parent = condensed_tree[idx].parent
        lambda_val = condensed_tree[idx].value

        if parent == current_parent:
            max_lambda = max(max_lambda, lambda_val)
        else:
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_val

    deaths[current_parent] = max_lambda  # 最后一个父节点的最大 Lambda 值

    return deaths  # 返回每个父节点的最大 Lambda 值数组
cdef class TreeUnionFind:
    # 定义一个 Cython 扩展类型 TreeUnionFind，用于并查集的实现

    cdef cnp.intp_t[:, ::1] data
    # 二维数组 data，存储每个节点的父节点及秩信息

    cdef cnp.uint8_t[::1] is_component
    # 一维数组 is_component，标记每个节点是否是一个独立的组件

    def __init__(self, size):
        # 构造函数，初始化并查集
        cdef cnp.intp_t idx
        self.data = np.zeros((size, 2), dtype=np.intp)
        # 初始化并查集的 data 数组，每个节点的初始父节点为自身，秩为 0
        for idx in range(size):
            self.data[idx, 0] = idx
        self.is_component = np.ones(size, dtype=np.uint8)
        # 初始化 is_component 数组，所有节点初始标记为独立组件

    cdef void union(self, cnp.intp_t x, cnp.intp_t y):
        # 并操作，将 x 所在集合和 y 所在集合合并
        cdef cnp.intp_t x_root = self.find(x)
        cdef cnp.intp_t y_root = self.find(y)

        if self.data[x_root, 1] < self.data[y_root, 1]:
            self.data[x_root, 0] = y_root
        elif self.data[x_root, 1] > self.data[y_root, 1]:
            self.data[y_root, 0] = x_root
        else:
            self.data[y_root, 0] = x_root
            self.data[x_root, 1] += 1
        return

    cdef cnp.intp_t find(self, cnp.intp_t x):
        # 查找操作，找到 x 所在集合的根节点，并进行路径压缩优化
        if self.data[x, 0] != x:
            self.data[x, 0] = self.find(self.data[x, 0])
            self.is_component[x] = False
        return self.data[x, 0]

cpdef cnp.ndarray[cnp.intp_t, ndim=1, mode='c'] labelling_at_cut(
        const HIERARCHY_t[::1] linkage,
        cnp.float64_t cut,
        cnp.intp_t min_cluster_size
):
    """Given a single linkage tree and a cut value, return the
    vector of cluster labels at that cut value. This is useful
    for Robust Single Linkage, and extracting DBSCAN results
    from a single HDBSCAN run.

    Parameters
    ----------
    linkage : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype
        The single linkage tree in scipy.cluster.hierarchy format.

    cut : double
        The cut value at which to find clusters.

    min_cluster_size : int
        The minimum cluster size; clusters below this size at
        the cut will be considered noise.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The cluster labels for each point in the data set;
        a label of -1 denotes a noise assignment.
    """

    cdef:
        cnp.intp_t n, cluster, root, n_samples, cluster_label
        cnp.intp_t[::1] unique_labels, cluster_size
        cnp.ndarray[cnp.intp_t, ndim=1, mode='c'] result
        TreeUnionFind union_find
        dict cluster_label_map
        HIERARCHY_t node

    root = 2 * linkage.shape[0]
    # 计算根节点的索引
    n_samples = root // 2 + 1
    # 计算样本数目
    result = np.empty(n_samples, dtype=np.intp)
    # 初始化结果数组，用于存储每个样本点的簇标签
    union_find = TreeUnionFind(root + 1)
    # 初始化并查集对象

    cluster = n_samples
    for node in linkage:
        if node.value < cut:
            union_find.union(node.left_node, cluster)
            union_find.union(node.right_node, cluster)
        cluster += 1
    # 遍历 linkage 中的节点，根据 cut 值进行聚类操作

    cluster_size = np.zeros(cluster, dtype=np.intp)
    # 初始化簇大小数组
    for n in range(n_samples):
        cluster = union_find.find(n)
        cluster_size[cluster] += 1
        result[n] = cluster
    # 统计每个簇的大小，并将每个样本点的簇标签存入 result 数组

    cluster_label_map = {-1: NOISE}
    # 初始化簇标签映射，-1 表示噪声点
    cluster_label = 0
    unique_labels = np.unique(result)
    # 获取 result 中的唯一簇标签
    # 遍历所有唯一的簇标签
    for cluster in unique_labels:
        # 如果当前簇的大小小于最小簇大小阈值
        if cluster_size[cluster] < min_cluster_size:
            # 将该簇标记为噪声（NOISE）
            cluster_label_map[cluster] = NOISE
        else:
            # 否则，为该簇分配一个新的簇标签，并增加簇标签计数器
            cluster_label_map[cluster] = cluster_label
            cluster_label += 1

    # 根据簇标签映射将结果集中的每个样本重新标记
    for n in range(n_samples):
        result[n] = cluster_label_map[result[n]]

    # 返回重新标记后的结果集
    return result
    cpdef cnp.ndarray[cnp.intp_t, ndim=1, mode='c'] _do_labelling(
        cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] condensed_tree,
        set clusters,
        dict cluster_label_map,
        cnp.intp_t allow_single_cluster,
        cnp.float64_t cluster_selection_epsilon
):
    """
    Given a condensed tree, clusters, and a labeling map for the clusters,
    return an array containing the labels of each point based on cluster
    membership. Note that this is where points may be marked as noisy
    outliers. The determination of some points as noise in large, single-
    cluster datasets is controlled by the `allow_single_cluster` and
    `cluster_selection_epsilon` parameters.

    Parameters
    ----------
    condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype
        Effectively an edgelist encoding a parent/child pair, along with a
        value and the corresponding cluster_size in each row providing a tree
        structure.

    clusters : set
        The set of nodes corresponding to identified clusters. These node
        values should be the same as those present in `condensed_tree`.

    cluster_label_map : dict
        A mapping from the node values present in `clusters` to the labels
        which will be returned.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The cluster labels for each point in the data set;
        a label of -1 denotes a noise assignment.
    """
    # Declare variables with C types for optimized performance
    cdef:
        cnp.intp_t root_cluster  # Stores the minimum value in parent_array
        cnp.ndarray[cnp.intp_t, ndim=1, mode='c'] result  # Resultant label array
        cnp.ndarray[cnp.intp_t, ndim=1] parent_array, child_array  # Arrays from condensed_tree
        cnp.ndarray[cnp.float64_t, ndim=1] lambda_array  # Value array from condensed_tree
        TreeUnionFind union_find  # Custom union-find data structure
        cnp.intp_t n, parent, child, cluster  # Loop and data index variables
        cnp.float64_t threshold  # Threshold value for cluster selection

    # Assign arrays from condensed_tree to local variables
    child_array = condensed_tree['child']
    parent_array = condensed_tree['parent']
    lambda_array = condensed_tree['value']

    # Determine the root cluster using the minimum value in parent_array
    root_cluster = np.min(parent_array)

    # Initialize result array with empty values based on root_cluster
    result = np.empty(root_cluster, dtype=np.intp)

    # Initialize union-find data structure with max value from parent_array
    union_find = TreeUnionFind(np.max(parent_array) + 1)

    # Iterate through each element in condensed_tree
    for n in range(PyArray_SHAPE(<cnp.PyArrayObject*> condensed_tree)[0]):
        child = child_array[n]
        parent = parent_array[n]
        # If the child node is not in the clusters set, perform union operation
        if child not in clusters:
            union_find.union(parent, child)
    # 对每个 root_cluster 中的节点 n 进行处理
    for n in range(root_cluster):
        # 查找节点 n 所属的集合的根节点
        cluster = union_find.find(n)
        
        # 默认将节点 n 标记为噪声
        label = NOISE
        
        # 如果节点 n 不是其自身的根节点，则根据其所属集合确定标签
        if cluster != root_cluster:
            label = cluster_label_map[cluster]
        
        # 如果节点 n 是其自身的根节点，并且允许单个集群存在且当前集群只有一个节点时
        elif len(clusters) == 1 and allow_single_cluster:
            # 从 lambda_array 中提取与节点 n 对应的 parent_lambda
            parent_lambda = lambda_array[child_array == n]
            
            # 根据 cluster_selection_epsilon 计算阈值
            if cluster_selection_epsilon != 0.0:
                threshold = 1 / cluster_selection_epsilon
            else:
                # 如果 cluster_selection_epsilon 为零，则基于同一父节点下所有节点的 lambda_array 的最大值计算阈值
                threshold = lambda_array[parent_array == cluster].max()
            
            # 如果 parent_lambda 大于等于阈值，则将节点 n 标记为对应集合的标签
            if parent_lambda >= threshold:
                label = cluster_label_map[cluster]
        
        # 将节点 n 的标签存入结果数组中
        result[n] = label
    
    # 返回处理完毕的结果数组
    return result
cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] get_probabilities(
    cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] condensed_tree,
    dict cluster_map,
    cnp.intp_t[::1] labels
):
    # 定义函数，计算并返回每个数据点的概率
    cdef:
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] result  # 结果数组，存储计算得到的概率值
        cnp.float64_t[:] lambda_array  # lambda 数组，存储聚类树中的 lambda 值
        cnp.float64_t[::1] deaths  # 死亡时间数组，由最大 lambda 值计算得到
        cnp.intp_t[:] child_array, parent_array  # 子节点和父节点数组
        cnp.intp_t root_cluster, n, point, cluster_num, cluster  # 定义一些整数变量和节点编号

    # 初始化子节点和父节点数组
    child_array = condensed_tree['child']
    parent_array = condensed_tree['parent']
    # 获取 lambda 数组
    lambda_array = condensed_tree['value']

    # 初始化结果数组为零
    result = np.zeros(labels.shape[0])
    # 计算最大 lambda 值对应的死亡时间数组
    deaths = max_lambdas(condensed_tree)
    # 获取根节点的集群编号
    root_cluster = np.min(parent_array)

    # 遍历聚类树中的每个节点
    for n in range(PyArray_SHAPE(<cnp.PyArrayObject*> condensed_tree)[0]):
        # 获取当前节点的子节点
        point = child_array[n]
        # 如果节点编号大于等于根节点的集群编号，则跳过
        if point >= root_cluster:
            continue

        # 获取当前节点对应的集群编号
        cluster_num = labels[point]
        # 如果集群编号为 -1，则跳过
        if cluster_num == -1:
            continue

        # 根据集群编号获取对应的集群
        cluster = cluster_map[cluster_num]
        # 获取该集群的最大死亡时间
        max_lambda = deaths[cluster]
        # 如果最大死亡时间为 0 或者 lambda_array 中的值为无穷大，则将结果设置为 1.0
        if max_lambda == 0.0 or isinf(lambda_array[n]):
            result[point] = 1.0
        else:
            # 否则计算当前节点的 lambda 值与最大死亡时间的比值，并存储到结果数组中
            lambda_val = min(lambda_array[n], max_lambda)
            result[point] = lambda_val / max_lambda

    # 返回计算得到的概率数组
    return result


cpdef list recurse_leaf_dfs(
    cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] cluster_tree,
    cnp.intp_t current_node
):
    # 定义函数，递归深度优先搜索叶子节点，并返回叶子节点列表
    cdef cnp.intp_t[:] children  # 子节点数组
    cdef cnp.intp_t child  # 子节点变量

    # 获取当前节点的所有子节点
    children = cluster_tree[cluster_tree['parent'] == current_node]['child']
    # 如果当前节点没有子节点，则返回包含当前节点的列表
    if children.shape[0] == 0:
        return [current_node,]
    else:
        # 否则递归地对每个子节点调用该函数，并将结果列表展平返回
        return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])


cpdef list get_cluster_tree_leaves(cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] cluster_tree):
    # 定义函数，获取聚类树的所有叶子节点
    cdef cnp.intp_t root  # 根节点变量
    # 如果聚类树为空，则返回空列表
    if PyArray_SHAPE(<cnp.PyArrayObject*> cluster_tree)[0] == 0:
        return []
    # 获取根节点的编号
    root = cluster_tree['parent'].min()
    # 调用递归深度优先搜索函数，返回所有叶子节点的列表
    return recurse_leaf_dfs(cluster_tree, root)


cdef cnp.intp_t traverse_upwards(
    cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] cluster_tree,
    cnp.float64_t cluster_selection_epsilon,
    cnp.intp_t leaf,
    cnp.intp_t allow_single_cluster
):
    # 定义函数，向上遍历聚类树，根据给定条件选择合适的节点
    cdef cnp.intp_t root, parent  # 根节点和父节点变量
    cdef cnp.float64_t parent_eps  # 父节点的 epsilon 值

    # 获取聚类树中的最小根节点编号
    root = cluster_tree['parent'].min()
    # 获取当前叶子节点的父节点编号
    parent = cluster_tree[cluster_tree['child'] == leaf]['parent']
    # 如果父节点是根节点
    if parent == root:
        # 如果允许返回单个集群，则返回父节点；否则返回最接近根节点的叶子节点
        if allow_single_cluster:
            return parent
        else:
            return leaf  # 返回最接近根节点的节点

    # 计算父节点的 epsilon 值
    parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']
    # 如果父节点的 epsilon 值大于给定的 cluster_selection_epsilon，则返回父节点
    if parent_eps > cluster_selection_epsilon:
        return parent
    else:
        # 否则递归地调用该函数，向上遍历直到找到合适的节点
        return traverse_upwards(
            cluster_tree,
            cluster_selection_epsilon,
            parent,
            allow_single_cluster
        )
    cnp.intp_t allow_single_cluster


// 声明一个名为 allow_single_cluster 的 cnp.intp_t 类型变量
):
    # 初始化空列表，用于存储选定的簇标签
    cdef:
        list selected_clusters = list()
        # 初始化空列表，用于跟踪已处理的簇标签
        list processed = list()
        # 声明变量 leaf, epsilon_child, sub_node 为 int 型
        cnp.intp_t leaf, epsilon_child, sub_node
        # 声明变量 eps 为 float64 型
        cnp.float64_t eps
        # 声明变量 leaf_nodes 为 uint8 数组
        cnp.uint8_t[:] leaf_nodes
        # 从聚类树中获取子节点数组，并赋值给 children
        cnp.ndarray[cnp.intp_t, ndim=1] children = cluster_tree['child']
        # 从聚类树中获取距离值数组，并赋值给 distances
        cnp.ndarray[cnp.float64_t, ndim=1] distances = cluster_tree['value']

    # 遍历每一个叶子节点
    for leaf in leaves:
        # 找到聚类树中与当前叶子节点相关的叶子节点
        leaf_nodes = children == leaf
        # 计算当前叶子节点的 epsilon 值
        eps = 1 / distances[leaf_nodes][0]
        
        # 如果 epsilon 值小于指定的簇选择阈值
        if eps < cluster_selection_epsilon:
            # 如果当前叶子节点尚未处理过
            if leaf not in processed:
                # 从当前叶子节点向上遍历聚类树，获取符合条件的子节点
                epsilon_child = traverse_upwards(
                    cluster_tree,
                    cluster_selection_epsilon,
                    leaf,
                    allow_single_cluster
                )
                # 将找到的子节点添加到选定的簇列表中
                selected_clusters.append(epsilon_child)

                # 从 epsilon_child 开始广度优先搜索聚类树，并将搜索到的节点加入已处理列表
                for sub_node in bfs_from_cluster_tree(cluster_tree, epsilon_child):
                    # 排除 epsilon_child 本身，避免重复处理
                    if sub_node != epsilon_child:
                        processed.append(sub_node)
        else:
            # 如果 epsilon 值大于等于簇选择阈值，则直接将当前叶子节点加入选定的簇列表
            selected_clusters.append(leaf)

    # 返回最终选定的簇标签的集合
    return set(selected_clusters)


# 使用 Cython 对该函数进行装饰，支持负数索引
@cython.wraparound(True)
# 声明私有函数 _get_clusters，接收以下参数
cdef tuple _get_clusters(
    # 压缩树结构的 ndarray，每个元素为 CONDENSED_t 类型
    cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] condensed_tree,
    # 稳定性字典，映射聚类 ID 到稳定性值
    dict stability,
    # 簇选择方法，默认为 'eom'
    cluster_selection_method='eom',
    # 是否允许单一簇标签，默认为 False
    cnp.uint8_t allow_single_cluster=False,
    # 簇选择阈值，默认为 0.0
    cnp.float64_t cluster_selection_epsilon=0.0,
    # 最大簇大小限制，默认为 None
    max_cluster_size=None
):
    """给定一个树结构和稳定性字典，根据选择的簇选择方法生成平面聚类的簇标签（和概率）。

    Parameters
    ----------
    condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype
        有效地是一个边列表，编码父/子对，以及树结构中每一行的值和相应的簇大小。

    stability : dict
        将聚类 ID 映射到稳定性值的字典。

    cluster_selection_method : string, optional (default 'eom')
        选择簇的方法。默认为 Excess of Mass 算法 'eom'。可选的替代选项为 'leaf'。

    allow_single_cluster : boolean, optional (default False)
        是否允许 Excess of Mass 算法选择单一簇标签。

    cluster_selection_epsilon: double, optional (default 0.0)
        用于簇分裂的距离阈值。

    max_cluster_size: int, default=None
        EOM 聚类器定位的簇的最大大小。在罕见情况下，可能会被 cluster_selection_epsilon 参数覆盖。

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        整数数组，表示簇标签，其中 -1 表示噪声。

    probabilities : ndarray (n_samples,)
        每个样本的簇成员强度。

    stabilities : ndarray (n_clusters,)
        每个簇的聚类一致性强度。
    """
    cdef:
        # 定义变量，用于存储节点列表、聚类树、子节点选择、标签、是否为聚类、聚类大小等信息
        list node_list
        cnp.ndarray[CONDENSED_t, ndim=1, mode='c'] cluster_tree
        cnp.uint8_t[::1] child_selection
        cnp.ndarray[cnp.intp_t, ndim=1, mode='c'] labels
        dict is_cluster, cluster_sizes
        cnp.float64_t subtree_stability
        cnp.intp_t node, sub_node, cluster, n_samples
        cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] probs

    # 假设聚类按照数值 ID 排序，相当于树的拓扑排序；这在上述当前实现中是有效的，因此不要更改它...
    # 如果更改了上述实现，请相应地更改这里！
    if allow_single_cluster:
        # 如果允许单个聚类，则将节点列表按照稳定性降序排序
        node_list = sorted(stability.keys(), reverse=True)
    else:
        # 如果不允许单个聚类，则将节点列表按照稳定性降序排序，并排除最后一个节点（即根节点）
        node_list = sorted(stability.keys(), reverse=True)[:-1]
        # （排除根节点）

    # 从压缩树中选择出聚类大小大于1的部分作为聚类树
    cluster_tree = condensed_tree[condensed_tree['cluster_size'] > 1]

    # 对于节点列表中的每个节点，标记其为聚类节点
    is_cluster = {cluster: True for cluster in node_list}

    # 计算只包含一个样本的节点的数量
    n_samples = np.max(condensed_tree[condensed_tree['cluster_size'] == 1]['child']) + 1

    # 如果最大聚类大小未指定，则设置为一个不会触发的值
    if max_cluster_size is None:
        max_cluster_size = n_samples + 1  # 设置为一个不会触发的值

    # 创建字典，将聚类树中每个子节点与其对应的聚类大小关联起来
    cluster_sizes = {
        child: cluster_size for child, cluster_size
        in zip(cluster_tree['child'], cluster_tree['cluster_size'])
    }

    if allow_single_cluster:
        # 计算根节点的聚类大小
        cluster_sizes[node_list[-1]] = np.sum(
            cluster_tree[cluster_tree['parent'] == node_list[-1]]['cluster_size'])
    # 如果聚类选择方法是'eom'，执行以下操作
    if cluster_selection_method == 'eom':
        # 遍历节点列表中的每个节点
        for node in node_list:
            # 选择子树中父节点为当前节点的子节点
            child_selection = (cluster_tree['parent'] == node)
            # 计算子树稳定性，将所有子节点稳定性值求和
            subtree_stability = np.sum([
                stability[child] for
                child in cluster_tree['child'][child_selection]])
            
            # 如果子树稳定性大于当前节点稳定性或者当前节点的聚类大小大于最大聚类大小
            if subtree_stability > stability[node] or cluster_sizes[node] > max_cluster_size:
                # 将当前节点标记为非聚类节点
                is_cluster[node] = False
                # 更新当前节点的稳定性值为子树稳定性值
                stability[node] = subtree_stability
            else:
                # 对于从当前节点开始的 BFS 遍历的每个子节点
                for sub_node in bfs_from_cluster_tree(cluster_tree, node):
                    # 如果子节点不等于当前节点
                    if sub_node != node:
                        # 将子节点标记为非聚类节点
                        is_cluster[sub_node] = False

        # 如果聚类选择 epsilon 不等于 0 并且 cluster_tree 数组的第一个元素大于 0
        if cluster_selection_epsilon != 0.0 and PyArray_SHAPE(<cnp.PyArrayObject*> cluster_tree)[0] > 0:
            # 从 is_cluster 中选择所有标记为聚类的节点形成 eom_clusters 列表
            eom_clusters = [c for c in is_cluster if is_cluster[c]]
            selected_clusters = []

            # 首先检查 eom_clusters 是否仅包含根节点，跳过 epsilon 检查
            if (len(eom_clusters) == 1 and eom_clusters[0] == cluster_tree['parent'].min()):
                # 如果允许单个聚类，并且 eom_clusters 只包含根节点，则选定的聚类为 eom_clusters
                if allow_single_cluster:
                    selected_clusters = eom_clusters
            else:
                # 否则调用 epsilon_search 函数查找符合条件的聚类节点
                selected_clusters = epsilon_search(
                    set(eom_clusters),
                    cluster_tree,
                    cluster_selection_epsilon,
                    allow_single_cluster
                )
            
            # 更新 is_cluster 中的节点，若节点在 selected_clusters 中，则标记为聚类节点，否则标记为非聚类节点
            for c in is_cluster:
                if c in selected_clusters:
                    is_cluster[c] = True
                else:
                    is_cluster[c] = False

    # 如果聚类选择方法是'leaf'
    elif cluster_selection_method == 'leaf':
        # 获取聚类树的所有叶子节点形成集合 leaves
        leaves = set(get_cluster_tree_leaves(cluster_tree))
        
        # 如果叶子节点数为 0
        if len(leaves) == 0:
            # 将所有节点标记为非聚类节点
            for c in is_cluster:
                is_cluster[c] = False
            # 将压缩树的根节点标记为聚类节点
            is_cluster[condensed_tree['parent'].min()] = True

        # 如果聚类选择 epsilon 不等于 0
        if cluster_selection_epsilon != 0.0:
            # 调用 epsilon_search 函数查找符合条件的聚类节点形成 selected_clusters
            selected_clusters = epsilon_search(
                leaves,
                cluster_tree,
                cluster_selection_epsilon,
                allow_single_cluster
            )
        else:
            # 否则选定的聚类节点为 leaves
            selected_clusters = leaves
        
        # 更新 is_cluster 中的节点，若节点在 selected_clusters 中，则标记为聚类节点，否则标记为非聚类节点
        for c in is_cluster:
            if c in selected_clusters:
                is_cluster[c] = True
            else:
                is_cluster[c] = False

    # 从 is_cluster 中选出所有标记为聚类的节点形成集合 clusters
    clusters = set([c for c in is_cluster if is_cluster[c]])
    # 构建聚类节点到索引的映射字典 cluster_map
    cluster_map = {c: n for n, c in enumerate(sorted(list(clusters)))}
    # 构建索引到聚类节点的反向映射字典 reverse_cluster_map
    reverse_cluster_map = {n: c for c, n in cluster_map.items()}

    # 调用 _do_labelling 函数进行标签化处理，返回标签 labels
    labels = _do_labelling(
        condensed_tree,
        clusters,
        cluster_map,
        allow_single_cluster,
        cluster_selection_epsilon
    )
    # 调用 get_probabilities 函数获取概率值，返回概率 probs
    probs = get_probabilities(condensed_tree, reverse_cluster_map, labels)

    # 返回标签和概率值的元组
    return (labels, probs)
```