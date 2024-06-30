# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hdbscan\_reachability.pyx`

```
# mutual reachability distance computations
# Authors: Leland McInnes <leland.mcinnes@gmail.com>
#          Meekail Zain <zainmeekail@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
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

cimport numpy as cnp  # 导入Cython版本的NumPy库

import numpy as np  # 导入NumPy库
from scipy.sparse import issparse  # 导入判断稀疏矩阵的函数
from cython cimport floating, integral  # 从Cython中导入浮点和整型类型
from libc.math cimport isfinite, INFINITY  # 从C标准库中导入数学函数和常数
from ...utils._typedefs cimport intp_t  # 从指定路径导入类型定义
cnp.import_array()  # 导入Cython版本的NumPy数组操作函数


def mutual_reachability_graph(
    distance_matrix, min_samples=5, max_distance=0.0
):
    """Compute the weighted adjacency matrix of the mutual reachability graph.

    The mutual reachability distance used to build the graph is defined as::

        max(d_core(x_p), d_core(x_q), d(x_p, x_q))

    and the core distance `d_core` is defined as the distance between a point
    `x_p` and its k-th nearest neighbor.

    Note that all computations are done in-place.

    Parameters
    ----------
    distance_matrix : {ndarray, sparse matrix} of shape (n_samples, n_samples)
        Array of distances between samples. If sparse, the array must be in
        `CSR` format.

    min_samples : int, default=5
        The number of points in a neighbourhood for a point to be considered
        a core point.
    """
    # 设置默认的最大距离，用来替换无穷大的距离。当距离矩阵是稀疏矩阵时使用。
    max_distance : float, default=0.0
        The distance which `np.inf` is replaced with. When the true mutual-
        reachability distance is measured to be infinite, it is instead
        truncated to `max_dist`. Only used when `distance_matrix` is a sparse
        matrix.

    Returns
    -------
    # 返回一个形状为 (n_samples, n_samples) 的互达图的加权邻接矩阵，可以是 ndarray 或稀疏矩阵。
    mututal_reachability_graph: {ndarray, sparse matrix} of shape \
            (n_samples, n_samples)
        Weighted adjacency matrix of the mutual reachability graph.

    References
    ----------
    # 参考文献引用，详细描述了基于密度估计的基于密度的聚类方法。
    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.
    """
    # 进一步的邻居索引，即最小样本数减一
    further_neighbor_idx = min_samples - 1
    # 如果距离矩阵是稀疏矩阵
    if issparse(distance_matrix):
        # 确保距离矩阵格式为 CSR，否则抛出错误
        if distance_matrix.format != "csr":
            raise ValueError(
                "Only sparse CSR matrices are supported for `distance_matrix`."
            )
        # 调用函数处理稀疏矩阵的互达图计算
        _sparse_mutual_reachability_graph(
            distance_matrix.data,
            distance_matrix.indices,
            distance_matrix.indptr,
            distance_matrix.shape[0],
            further_neighbor_idx=further_neighbor_idx,
            max_distance=max_distance,
        )
    else:
        # 调用函数处理稠密矩阵的互达图计算
        _dense_mutual_reachability_graph(
            distance_matrix, further_neighbor_idx=further_neighbor_idx
        )
    # 返回处理后的距离矩阵（可能被修改）
    return distance_matrix
def _dense_mutual_reachability_graph(
    floating[:, :] distance_matrix,
    intp_t further_neighbor_idx,
):
    """Dense implementation of mutual reachability graph.

    The computation is done in-place, i.e. the distance matrix is modified
    directly.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n_samples, n_samples)
        Array of distances between samples.

    further_neighbor_idx : int
        The index of the furthest neighbor to use to define the core distances.
    """
    cdef:
        intp_t i, j, n_samples = distance_matrix.shape[0]
        floating mutual_reachibility_distance
        floating[::1] core_distances

    # We assume that the distance matrix is symmetric. We choose to sort every
    # row to have the same implementation than the sparse case that requires
    # CSR matrix.
    core_distances = np.ascontiguousarray(
        np.partition(
            distance_matrix, further_neighbor_idx, axis=1
        )[:, further_neighbor_idx]
    )

    with nogil:
        # TODO: Update w/ prange with thread count based on
        # _openmp_effective_n_threads
        for i in range(n_samples):
            for j in range(n_samples):
                # Calculate the mutual reachability distance as the maximum
                # of core distances of points i and j, and their direct distance.
                mutual_reachibility_distance = max(
                    core_distances[i],
                    core_distances[j],
                    distance_matrix[i, j],
                )
                # Update the distance matrix with the computed mutual reachability distance.
                distance_matrix[i, j] = mutual_reachibility_distance


def _sparse_mutual_reachability_graph(
    cnp.ndarray[floating, ndim=1, mode="c"] data,
    cnp.ndarray[integral, ndim=1, mode="c"] indices,
    cnp.ndarray[integral, ndim=1, mode="c"] indptr,
    intp_t n_samples,
    intp_t further_neighbor_idx,
    floating max_distance,
):
    """Sparse implementation of mutual reachability graph.

    The computation is done in-place, i.e. the distance matrix is modified
    directly. This implementation only accepts `CSR` format sparse matrices.

    Parameters
    ----------
    distance_matrix : sparse matrix of shape (n_samples, n_samples)
        Sparse matrix of distances between samples. The sparse format should
        be `CSR`.

    further_neighbor_idx : int
        The index of the furthest neighbor to use to define the core distances.

    max_distance : float
        The distance which `np.inf` is replaced with. When the true mutual-
        reachability distance is measured to be infinite, it is instead
        truncated to `max_dist`. Only used when `distance_matrix` is a sparse
        matrix.
    """
    cdef:
        integral i, col_ind, row_ind
        floating mutual_reachibility_distance
        floating[:] core_distances
        floating[:] row_data

    # Determine the appropriate dtype based on the floating type
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # Initialize an array to hold core distances
    core_distances = np.empty(n_samples, dtype=dtype)
    # 遍历每个样本的索引范围
    for i in range(n_samples):
        # 提取当前样本的数据行
        row_data = data[indptr[i]:indptr[i + 1]]
        
        # 检查进一步的邻居索引是否在当前行数据范围内
        if further_neighbor_idx < row_data.size:
            # 计算当前样本的核心距离，使用快速分割算法找到第 further_neighbor_idx 小的元素
            core_distances[i] = np.partition(
                row_data, further_neighbor_idx
            )[further_neighbor_idx]
        else:
            # 如果进一步的邻居索引超出了行数据范围，则将核心距离设为无穷大
            core_distances[i] = INFINITY

    # 使用 nogil 上下文进行并行处理
    with nogil:
        # 遍历每个行索引
        for row_ind in range(n_samples):
            # 遍历当前行的非零元素索引范围
            for i in range(indptr[row_ind], indptr[row_ind + 1]):
                # 提取当前非零元素对应的列索引
                col_ind = indices[i]
                
                # 计算互达距离，取当前行和列的核心距离及数据中的最大值
                mutual_reachibility_distance = max(
                    core_distances[row_ind], core_distances[col_ind], data[i]
                )
                
                # 如果互达距离有限，则更新数据中的当前元素
                if isfinite(mutual_reachibility_distance):
                    data[i] = mutual_reachibility_distance
                # 否则，如果最大距离大于零，则更新数据中的当前元素为最大距离
                elif max_distance > 0:
                    data[i] = max_distance
```