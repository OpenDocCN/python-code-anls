# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_k_means_common.pyx`

```
# 导入必要的库和模块
import numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport sqrt

# 导入外部工具函数
from ..utils.extmath import row_norms

# 每个数据块的样本数定义为全局常量
CHUNK_SIZE = 256

# 定义一个 Cython 函数来计算两个密集型数组之间的欧氏距离
cdef floating _euclidean_dense_dense(
        const floating* a,  # 输入参数 a，密集型数组 a 的指针
        const floating* b,  # 输入参数 b，密集型数组 b 的指针
        int n_features,     # 特征数目
        bint squared       # 如果为真，则计算距离的平方；否则计算距离
) noexcept nogil:
    """计算密集型数组 a 和 b 之间的欧氏距离"""
    cdef:
        int i
        int n = n_features // 4   # 将特征数目除以 4，以进行优化
        int rem = n_features % 4  # 计算余数，用于处理特征数目无法被 4 整除的情况
        floating result = 0

    # 手动展开循环，以便更好地进行缓存优化
    for i in range(n):
        result += (
            (a[0] - b[0]) * (a[0] - b[0]) +
            (a[1] - b[1]) * (a[1] - b[1]) +
            (a[2] - b[2]) * (a[2] - b[2]) +
            (a[3] - b[3]) * (a[3] - b[3])
        )
        a += 4  # 移动指针到下一个块的起始位置
        b += 4  # 同上

    # 处理剩余的特征
    for i in range(rem):
        result += (a[i] - b[i]) * (a[i] - b[i])

    return result if squared else sqrt(result)


# 包装函数，用于测试目的，调用 _euclidean_dense_dense 函数
def _euclidean_dense_dense_wrapper(
    const floating[::1] a,     # 输入参数 a，密集型数组 a 的引用
    const floating[::1] b,     # 输入参数 b，密集型数组 b 的引用
    bint squared              # 如果为真，则计算距离的平方；否则计算距离
):
    """_euclidean_dense_dense 的包装器，用于测试目的"""
    return _euclidean_dense_dense(&a[0], &b[0], a.shape[0], squared)


# 定义一个 Cython 函数来计算稀疏数组和密集型数组之间的欧氏距离
cdef floating _euclidean_sparse_dense(
        const floating[::1] a_data,    # 输入参数 a_data，稀疏数组的数据部分
        const int[::1] a_indices,      # 输入参数 a_indices，稀疏数组的索引部分
        const floating[::1] b,         # 输入参数 b，密集型数组 b 的引用
        floating b_squared_norm,       # 密集型数组 b 的平方范数
        bint squared                  # 如果为真，则计算距离的平方；否则计算距离
) noexcept nogil:
    """计算稀疏数组 a 和密集型数组 b 之间的欧氏距离"""
    cdef:
        int nnz = a_indices.shape[0]   # 获取稀疏数组的非零元素数目
        int i
        floating tmp, bi
        floating result = 0.0

    # 遍历稀疏数组中的非零元素
    for i in range(nnz):
        bi = b[a_indices[i]]    # 获取密集型数组 b 中对应索引的值
        tmp = a_data[i] - bi    # 计算差值
        result += tmp * tmp - bi * bi   # 计算欧氏距离的平方

    result += b_squared_norm    # 加上密集型数组 b 的平方范数

    if result < 0:
        result = 0.0

    return result if squared else sqrt(result)


# 包装函数，用于测试目的，调用 _euclidean_sparse_dense 函数
def _euclidean_sparse_dense_wrapper(
        const floating[::1] a_data,         # 输入参数 a_data，稀疏数组的数据部分
        const int[::1] a_indices,           # 输入参数 a_indices，稀疏数组的索引部分
        const floating[::1] b,              # 输入参数 b，密集型数组 b 的引用
        floating b_squared_norm,            # 密集型数组 b 的平方范数
        bint squared                       # 如果为真，则计算距离的平方；否则计算距离
):
    """_euclidean_sparse_dense 的包装器，用于测试目的"""
    return _euclidean_sparse_dense(
        a_data, a_indices, b, b_squared_norm, squared)


# 定义一个 Cython 函数来计算密集型输入数据的惯性
cpdef floating _inertia_dense(
        const floating[:, ::1] X,            # 输入参数 X，密集型输入数据
        const floating[::1] sample_weight,   # 输入参数 sample_weight，样本权重
        const floating[:, ::1] centers,      # 输入参数 centers，聚类中心
        const int[::1] labels,               # 输入参数 labels，标签
        int n_threads,                       # 输入参数 n_threads，线程数
        int single_label=-1,                 # 输入参数 single_label，默认为 -1
):
    """计算密集型输入数据的惯性

    每个样本与其分配的中心之间的平方距离之和。

    如果 single_label >= 0，则仅计算该标签的惯性。
    """
    cdef:
        # 样本数量，即 X 的行数
        int n_samples = X.shape[0]
        # 特征数量，即 X 的列数
        int n_features = X.shape[1]
        # 循环变量 i, j
        int i, j

        # 浮点型变量，用于存储计算得到的平方距离
        floating sq_dist = 0.0
        # 浮点型变量，用于存储惯性或者聚类的总误差平方和
        floating inertia = 0.0

    # 使用并行化循环对每个样本进行处理
    for i in prange(n_samples, nogil=True, num_threads=n_threads,
                    schedule='static'):
        # 获取当前样本的类别标签
        j = labels[i]
        # 如果 single_label 小于 0 或者等于当前类别标签 j，则进行计算
        if single_label < 0 or single_label == j:
            # 计算当前样本到其所属类别中心的欧氏距离的平方
            sq_dist = _euclidean_dense_dense(&X[i, 0], &centers[j, 0],
                                             n_features, True)
            # 将该距离的平方乘以样本权重，加入总的惯性值中
            inertia += sq_dist * sample_weight[i]

    # 返回最终的惯性值
    return inertia
# 计算稀疏输入数据的惯性，即每个样本与其分配的中心点之间的平方距离之和
cpdef floating _inertia_sparse(
        X,                                  # IN 稀疏输入数据
        const floating[::1] sample_weight,  # IN 每个样本的权重
        const floating[:, ::1] centers,     # IN 中心点的坐标
        const int[::1] labels,              # IN 每个样本分配的中心点标签
        int n_threads,                      # IN 线程数
        int single_label=-1,                # IN 仅计算指定标签的惯性（可选）
):
    """Compute inertia for sparse input data

    Sum of squared distance between each sample and its assigned center.

    If single_label is >= 0, the inertia is computed only for that label.
    """
    cdef:
        floating[::1] X_data = X.data        # 稀疏数据的值数组
        int[::1] X_indices = X.indices       # 稀疏数据的列索引数组
        int[::1] X_indptr = X.indptr         # 稀疏数据的行指针数组

        int n_samples = X.shape[0]           # 样本数量
        int i, j

        floating sq_dist = 0.0               # 存储平方距离
        floating inertia = 0.0               # 最终惯性值

        floating[::1] centers_squared_norms = row_norms(centers, squared=True)  # 中心点的平方范数数组

    for i in prange(n_samples, nogil=True, num_threads=n_threads,
                    schedule='static'):
        j = labels[i]
        if single_label < 0 or single_label == j:
            # 计算当前样本与其分配的中心点之间的欧氏距离的平方
            sq_dist = _euclidean_sparse_dense(
                X_data[X_indptr[i]: X_indptr[i + 1]],
                X_indices[X_indptr[i]: X_indptr[i + 1]],
                centers[j], centers_squared_norms[j], True)
            # 将距离的平方乘以样本的权重，并累加到惯性值中
            inertia += sq_dist * sample_weight[i]

    return inertia


# 重新定位没有样本分配的聚类中心
cpdef void _relocate_empty_clusters_dense(
        const floating[:, ::1] X,            # IN 输入数据
        const floating[::1] sample_weight,   # IN 每个样本的权重
        const floating[:, ::1] centers_old,  # IN 旧的聚类中心坐标
        floating[:, ::1] centers_new,        # INOUT 新的聚类中心坐标（可变）
        floating[::1] weight_in_clusters,    # INOUT 每个聚类中心的权重总和
        const int[::1] labels                # IN 每个样本分配的聚类中心标签
):
    """Relocate centers which have no sample assigned to them."""
    cdef:
        int[::1] empty_clusters = np.where(np.equal(weight_in_clusters, 0))[0].astype(np.int32)  # 没有样本分配的聚类中心的索引数组
        int n_empty = empty_clusters.shape[0]  # 没有样本分配的聚类中心的数量

    if n_empty == 0:
        return  # 如果没有空的聚类中心，直接返回

    cdef:
        int n_features = X.shape[1]  # 数据的特征数

        # 计算每个样本到其分配的中心点的距离的平方，并求出距离最远的样本的索引
        floating[::1] distances = ((np.asarray(X) - np.asarray(centers_old)[labels])**2).sum(axis=1)
        int[::1] far_from_centers = np.argpartition(distances, -n_empty)[:-n_empty-1:-1].astype(np.int32)

        int new_cluster_id, old_cluster_id, far_idx, idx, k
        floating weight

    if np.max(distances) == 0:
        # 当存在更多的聚类中心而非重复样本时会发生这种情况。在这种情况下重新定位是无意义的。
        return  # 如果所有样本都是重复的，无需重新定位聚类中心

    for idx in range(n_empty):
        new_cluster_id = empty_clusters[idx]  # 获取当前需要重新定位的新聚类中心的索引

        far_idx = far_from_centers[idx]  # 获取当前离新聚类中心最远的样本的索引
        weight = sample_weight[far_idx]  # 获取离新聚类中心最远的样本的权重

        old_cluster_id = labels[far_idx]  # 获取离新聚类中心最远的样本原先分配的旧聚类中心的索引

        for k in range(n_features):
            # 更新新旧聚类中心的坐标，根据离新聚类中心最远的样本的权重
            centers_new[old_cluster_id, k] -= X[far_idx, k] * weight
            centers_new[new_cluster_id, k] = X[far_idx, k] * weight

        weight_in_clusters[new_cluster_id] = weight  # 更新新聚类中心的权重总和
        weight_in_clusters[old_cluster_id] -= weight  # 减去旧聚类中心的权重总和
# 重新定位没有样本分配给它们的中心点
cpdef void _relocate_empty_clusters_sparse(
        const floating[::1] X_data,          # 输入：稀疏数据的值部分
        const int[::1] X_indices,            # 输入：稀疏数据的索引部分
        const int[::1] X_indptr,             # 输入：稀疏数据的指针部分
        const floating[::1] sample_weight,   # 输入：样本权重
        const floating[:, ::1] centers_old,  # 输入：旧的聚类中心
        floating[:, ::1] centers_new,        # 输入/输出：新的聚类中心
        floating[::1] weight_in_clusters,    # 输入/输出：聚类中心的权重
        const int[::1] labels                # 输入：每个样本的聚类标签
):
    """重新定位没有样本分配给它们的中心点"""
    cdef:
        # 找出权重为零的聚类
        int[::1] empty_clusters = np.where(np.equal(weight_in_clusters, 0))[0].astype(np.int32)
        int n_empty = empty_clusters.shape[0]

    if n_empty == 0:
        return

    cdef:
        int n_samples = X_indptr.shape[0] - 1
        int i, j, k

        # 初始化距离数组
        floating[::1] distances = np.zeros(n_samples, dtype=X_data.base.dtype)
        # 计算旧聚类中心的平方范数
        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

    for i in range(n_samples):
        j = labels[i]
        # 计算当前样本到其对应聚类中心的欧氏距离
        distances[i] = _euclidean_sparse_dense(
            X_data[X_indptr[i]: X_indptr[i + 1]],
            X_indices[X_indptr[i]: X_indptr[i + 1]],
            centers_old[j], centers_squared_norms[j], True)

    if np.max(distances) == 0:
        # 当存在的聚类数多于非重复样本数时会发生这种情况，此时重新定位聚类中心没有意义
        return

    cdef:
        # 找出距离最远的样本索引
        int[::1] far_from_centers = np.argpartition(distances, -n_empty)[:-n_empty-1:-1].astype(np.int32)

        int new_cluster_id, old_cluster_id, far_idx, idx
        floating weight

    for idx in range(n_empty):
        new_cluster_id = empty_clusters[idx]

        far_idx = far_from_centers[idx]
        weight = sample_weight[far_idx]

        old_cluster_id = labels[far_idx]

        for k in range(X_indptr[far_idx], X_indptr[far_idx + 1]):
            # 调整新旧聚类中心的值以反映样本移动
            centers_new[old_cluster_id, X_indices[k]] -= X_data[k] * weight
            centers_new[new_cluster_id, X_indices[k]] = X_data[k] * weight

        # 更新新旧聚类中心的权重
        weight_in_clusters[new_cluster_id] = weight
        weight_in_clusters[old_cluster_id] -= weight


cdef void _average_centers(
        floating[:, ::1] centers,               # 输入/输出：聚类中心
        const floating[::1] weight_in_clusters  # 输入：聚类中心的权重
):
    """根据权重平均新的聚类中心"""
    cdef:
        int n_clusters = centers.shape[0]
        int n_features = centers.shape[1]
        int j, k
        floating alpha
        # 找出权重最大的聚类
        int argmax_weight = np.argmax(weight_in_clusters)

    for j in range(n_clusters):
        if weight_in_clusters[j] > 0:
            # 计算权重的倒数
            alpha = 1.0 / weight_in_clusters[j]
            for k in range(n_features):
                # 对每个特征维度上的聚类中心进行加权平均
                centers[j, k] *= alpha
        else:
            # 为了方便起见，我们避免将空聚类设在原点，而是放置在权重最大的聚类位置
            for k in range(n_features):
                centers[j, k] = centers[argmax_weight, k]
cdef void _center_shift(
        const floating[:, ::1] centers_old,  # 旧中心点数组，输入参数
        const floating[:, ::1] centers_new,  # 新中心点数组，输入参数
        floating[::1] center_shift           # 中心点位移数组，输出参数
):
    """计算旧中心点和新中心点之间的位移。"""
    cdef:
        int n_clusters = centers_old.shape[0]  # 簇的数量
        int n_features = centers_old.shape[1]  # 特征的数量
        int j  # 循环变量

    for j in range(n_clusters):
        center_shift[j] = _euclidean_dense_dense(
            &centers_new[j, 0], &centers_old[j, 0], n_features, False)
        # 调用欧氏距离计算函数，计算新旧中心点之间的欧氏距离


def _is_same_clustering(
    const int[::1] labels1,  # 第一个标签数组，输入参数
    const int[::1] labels2,  # 第二个标签数组，输入参数
    n_clusters               # 簇的数量，输入参数
):
    """检查两个标签数组是否相同，允许标签的排列顺序不同。"""
    cdef int[::1] mapping = np.full(fill_value=-1, shape=(n_clusters,), dtype=np.int32)
    cdef int i  # 循环变量

    for i in range(labels1.shape[0]):
        if mapping[labels1[i]] == -1:
            mapping[labels1[i]] = labels2[i]
        elif mapping[labels1[i]] != labels2[i]:
            return False
    return True
```