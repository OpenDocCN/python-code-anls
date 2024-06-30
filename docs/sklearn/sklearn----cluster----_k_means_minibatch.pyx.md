# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_k_means_minibatch.pyx`

```
# 导入必要的 Cython 库中的特定模块和函数声明
from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.stdlib cimport malloc, free

# 定义函数 _minibatch_update_dense，用于执行稠密 MiniBatchKMeans 的中心更新操作
def _minibatch_update_dense(
        const floating[:, ::1] X,            # 输入参数：观测数据矩阵 X
        const floating[::1] sample_weight,   # 输入参数：每个观测的权重数组
        const floating[:, ::1] centers_old,  # 输入参数：上一迭代的中心点位置
        floating[:, ::1] centers_new,        # 输出参数：当前迭代的新中心点位置
        floating[::1] weight_sums,           # 输入输出参数：各中心的权重累计和
        const int[::1] labels,               # 输入参数：每个观测的聚类标签
        int n_threads):                      # 输入参数：用于 OpenMP 的线程数
    """Update of the centers for dense MiniBatchKMeans.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The observations to cluster.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    weight_sums : ndarray of shape (n_clusters,), dtype=floating
        Current sums of the accumulated weights for each center.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    n_threads : int
        The number of threads to be used by openmp.
    """
    cdef:
        int n_samples = X.shape[0]           # 观测数据的样本数
        int n_clusters = centers_old.shape[0]# 聚类中心的数量
        int cluster_idx                      # 聚类中心索引

        int *indices                          # 用于存储符合条件的观测索引数组

    # 使用 Cython 的 nogil 和 parallel 指令并行执行以下代码段
    with nogil, parallel(num_threads=n_threads):
        indices = <int*> malloc(n_samples * sizeof(int))  # 分配内存以存储符合条件的观测索引

        # 使用 prange 遍历每个聚类中心
        for cluster_idx in prange(n_clusters, schedule="static"):
            # 调用 update_center_dense 函数更新指定聚类中心的位置
            update_center_dense(cluster_idx, X, sample_weight,
                                centers_old, centers_new, weight_sums, labels,
                                indices)

        free(indices)  # 释放动态分配的内存空间


# 定义函数 update_center_dense，用于更新稠密 MiniBatchKMeans 的单个聚类中心
cdef void update_center_dense(
        int cluster_idx,                       # 输入参数：要更新的聚类中心的索引
        const floating[:, ::1] X,              # 输入参数：观测数据矩阵 X
        const floating[::1] sample_weight,     # 输入参数：每个观测的权重数组
        const floating[:, ::1] centers_old,    # 输入参数：上一迭代的中心点位置
        floating[:, ::1] centers_new,          # 输出参数：当前迭代的新中心点位置
        floating[::1] weight_sums,             # 输入输出参数：各中心的权重累计和
        const int[::1] labels,                 # 输入参数：每个观测的聚类标签
        int *indices) noexcept nogil:          # 临时参数：用于存储符合条件的观测索引
    """Update of a single center for dense MinibatchKMeans"""
    cdef:
        int n_samples = sample_weight.shape[0]  # 观测数据的样本数
        int n_features = centers_old.shape[1]   # 观测数据的特征数
        floating alpha                          # 临时变量：用于更新聚类中心位置的系数
        int n_indices                           # 符合条件的观测索引数
        int k, sample_idx, feature_idx          # 循环变量：用于迭代观测数据的索引、特征索引

        floating wsum = 0                       # 临时变量：用于计算权重总和的累加器

    # 遍历所有观测数据，选出符合条件的观测索引，存储在 indices 数组中
    k = 0
    for sample_idx in range(n_samples):
        if labels[sample_idx] == cluster_idx:
            indices[k] = sample_idx
            wsum += sample_weight[sample_idx]
            k += 1
    n_indices = k  # 记录符合条件的观测索引数量
    if wsum > 0:
        # 如果该簇中有样本被分配到，则执行以下操作

        # 取消之前基于计数的缩放操作
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] = centers_old[cluster_idx, feature_idx] * weight_sums[cluster_idx]

        # 使用新的点更新簇的中心
        for k in range(n_indices):
            sample_idx = indices[k]
            for feature_idx in range(n_features):
                centers_new[cluster_idx, feature_idx] += X[sample_idx, feature_idx] * sample_weight[sample_idx]

        # 更新该簇的加权和统计信息
        weight_sums[cluster_idx] += wsum

        # 重新缩放以计算所有点（旧点和新点）的平均值
        alpha = 1 / weight_sums[cluster_idx]
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] *= alpha
    else:
        # 在本批数据中，没有样本被分配到该簇
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] = centers_old[cluster_idx, feature_idx]
def _minibatch_update_sparse(
        X,                                   # IN：稀疏矩阵 X，形状为 (n_samples, n_features)，数据类型为浮点数
        const floating[::1] sample_weight,   # IN：每个观测的权重数组，数据类型为浮点数
        const floating[:, ::1] centers_old,  # IN：上一次迭代后的中心点数组，数据类型为浮点数
        floating[:, ::1] centers_new,        # OUT：当前迭代计算得到的新中心点数组，数据类型为浮点数
        floating[::1] weight_sums,           # INOUT：每个中心点累积权重的数组，数据类型为浮点数
        const int[::1] labels,               # IN：观测点的标签分配数组，数据类型为整数
        int n_threads):
    """Update of the centers for sparse MiniBatchKMeans.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The observations to cluster. Must be in CSR format.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    weight_sums : ndarray of shape (n_clusters,), dtype=floating
        Current sums of the accumulated weights for each center.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    n_threads : int
        The number of threads to be used by openmp.
    """
    cdef:
        floating[::1] X_data = X.data        # X 稀疏矩阵的非零元素数组
        int[::1] X_indices = X.indices      # X 稀疏矩阵的非零元素所在列索引数组
        int[::1] X_indptr = X.indptr        # X 稀疏矩阵行指针数组
        int n_samples = X.shape[0]          # 观测点数量
        int n_clusters = centers_old.shape[0]  # 簇的数量
        int cluster_idx                     # 簇索引

        int *indices                         # 用于存储标签为 cluster_idx 的观测点索引数组

    with nogil, parallel(num_threads=n_threads):
        indices = <int*> malloc(n_samples * sizeof(int))  # 分配内存空间用于存储索引数组

        for cluster_idx in prange(n_clusters, schedule="static"):
            update_center_sparse(cluster_idx, X_data, X_indices, X_indptr,
                                 sample_weight, centers_old, centers_new,
                                 weight_sums, labels, indices)

        free(indices)  # 释放索引数组的内存空间


cdef void update_center_sparse(
        int cluster_idx,
        const floating[::1] X_data,          # IN：观测点的非零元素数据数组，数据类型为浮点数
        const int[::1] X_indices,            # IN：观测点的非零元素所在列索引数组，数据类型为整数
        const int[::1] X_indptr,             # IN：观测点行指针数组，数据类型为整数
        const floating[::1] sample_weight,   # IN：每个观测点的权重数组，数据类型为浮点数
        const floating[:, ::1] centers_old,  # IN：上一次迭代后的中心点数组，数据类型为浮点数
        floating[:, ::1] centers_new,        # OUT：当前迭代计算得到的新中心点数组，数据类型为浮点数
        floating[::1] weight_sums,           # INOUT：每个中心点累积权重的数组，数据类型为浮点数
        const int[::1] labels,               # IN：观测点的标签分配数组，数据类型为整数
        int *indices) noexcept nogil:        # TMP：用于存储标签为 cluster_idx 的观测点索引数组，无 gil

    """Update of a single center for sparse MinibatchKMeans"""
    cdef:
        int n_samples = sample_weight.shape[0]  # 观测点数量
        int n_features = centers_old.shape[1]   # 特征数量
        floating alpha                          # 临时变量 alpha
        int n_indices                           # 标签为 cluster_idx 的观测点数量
        int k, sample_idx, feature_idx          # 临时变量 k、样本索引、特征索引

        floating wsum = 0                       # 权重总和初始化为 0

    # indices = np.where(labels == cluster_idx)[0]
    k = 0  # 初始化 k 为 0
    # 遍历每个样本的索引，检查其标签是否等于当前簇的索引
    for sample_idx in range(n_samples):
        if labels[sample_idx] == cluster_idx:
            # 如果样本属于当前簇，则将其索引添加到indices数组中
            indices[k] = sample_idx
            # 更新加权和，加上当前样本的权重
            wsum += sample_weight[sample_idx]
            # 更新索引计数器
            k += 1
    # 记录有效索引的数量
    n_indices = k

    # 如果加权和大于0，执行以下操作
    if wsum > 0:
        # 恢复之前针对该簇中心的基于计数的缩放
        for feature_idx in range(n_features):
            # 更新centers_new数组，按簇中心权重之和缩放旧中心的每个特征
            centers_new[cluster_idx, feature_idx] = centers_old[cluster_idx, feature_idx] * weight_sums[cluster_idx]

        # 使用新点成员更新簇
        for k in range(n_indices):
            sample_idx = indices[k]
            # 遍历样本中的每个特征值，根据样本权重更新centers_new数组
            for feature_idx in range(X_indptr[sample_idx], X_indptr[sample_idx + 1]):
                centers_new[cluster_idx, X_indices[feature_idx]] += X_data[feature_idx] * sample_weight[sample_idx]

        # 更新该中心的计数统计信息
        weight_sums[cluster_idx] += wsum

        # 重新缩放以计算所有点（旧点和新点）的平均值
        alpha = 1 / weight_sums[cluster_idx]
        # 对于每个特征，按簇中心的权重之和重新缩放centers_new数组
        for feature_idx in range(n_features):
            centers_new[cluster_idx, feature_idx] *= alpha
    else:
        # 如果在此批数据中没有样本分配给该簇
        for feature_idx in range(n_features):
            # 将centers_new数组设置为与centers_old数组相同的值，即保持不变
            centers_new[cluster_idx, feature_idx] = centers_old[cluster_idx, feature_idx]
```