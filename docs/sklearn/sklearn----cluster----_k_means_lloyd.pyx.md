# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_k_means_lloyd.pyx`

```
# Licence: BSD 3 clause

# 从 Cython 导入特定的类型和函数
from cython cimport floating
from cython.parallel import prange, parallel
# 从 C 标准库导入内存管理相关的函数
from libc.stdlib cimport malloc, calloc, free
# 从 C 标准库导入字符串处理相关的函数
from libc.string cimport memset
# 从 C 标准库导入浮点数的最大值常量
from libc.float cimport DBL_MAX, FLT_MAX

# 从当前包的工具模块中导入 OpenMP 相关的辅助函数和类型
from ..utils._openmp_helpers cimport omp_lock_t
from ..utils._openmp_helpers cimport omp_init_lock
from ..utils._openmp_helpers cimport omp_destroy_lock
from ..utils._openmp_helpers cimport omp_set_lock
from ..utils._openmp_helpers cimport omp_unset_lock
# 从当前包的工具模块中导入数学扩展相关的函数
from ..utils.extmath import row_norms
# 从当前包的工具模块中导入 Cython 实现的 BLAS 函数
from ..utils._cython_blas cimport _gemm
from ..utils._cython_blas cimport RowMajor, Trans, NoTrans
# 从当前包的 K-means 通用模块中导入常量和函数
from ._k_means_common import CHUNK_SIZE
from ._k_means_common cimport _relocate_empty_clusters_dense
from ._k_means_common cimport _relocate_empty_clusters_sparse
from ._k_means_common cimport _average_centers, _center_shift

# 定义函数 lloyd_iter_chunked_dense，实现 K-means 算法的单个迭代过程，适用于密集输入数据
def lloyd_iter_chunked_dense(
        const floating[:, ::1] X,            # IN 输入数据矩阵，形状为 (样本数, 特征数)
        const floating[::1] sample_weight,   # IN 每个样本的权重数组，形状为 (样本数,)
        const floating[:, ::1] centers_old,  # IN 上一迭代结束后的聚类中心，形状为 (聚类数, 特征数)
        floating[:, ::1] centers_new,        # OUT 当前迭代结束后的聚类中心，形状为 (聚类数, 特征数)
        floating[::1] weight_in_clusters,    # OUT 每个聚类的样本权重和，形状为 (聚类数,)
        int[::1] labels,                     # OUT 每个样本分配的聚类标签，形状为 (样本数,)
        floating[::1] center_shift,          # OUT 聚类中心的变化量，形状为 (聚类数,)
        int n_threads,                       # IN 使用的线程数
        bint update_centers=True             # IN 是否更新聚类中心的布尔标志
):
    """Single iteration of K-means lloyd algorithm with dense input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

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
        computed during this iteration. `centers_new` can be `None` if
        `update_centers` is False.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center. `weight_in_clusters` can be `None` if `update_centers`
        is False.

    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.

    n_threads : int
        The number of threads to be used by openmp.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    cdef:
        int n_samples = X.shape[0]           # 获取样本数量
        int n_features = X.shape[1]          # 获取特征数量
        int n_clusters = centers_old.shape[0] # 获取旧的聚类中心数量

    if n_samples == 0:
        # 如果样本数量为0，则返回空，不进行后续计算（在尝试计算n_chunks之前）。这通常发生在使用分层k均值模型的预测函数时，数据中包含大量异常值时。
        return

    cdef:
        # 每个数据块中固定的样本数。在所有情况下似乎都接近最优。
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples  # 计算每个数据块中的样本数
        int n_chunks = n_samples // n_samples_chunk  # 计算数据块的数量
        int n_samples_rem = n_samples % n_samples_chunk  # 计算余下的样本数
        int chunk_idx    # 数据块的索引
        int start, end   # 数据块的起始和结束索引

        int j, k         # 循环变量

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)  # 计算旧聚类中心的平方范数

        floating *centers_new_chunk            # 新数据块的聚类中心
        floating *weight_in_clusters_chunk     # 新数据块的聚类中心权重
        floating *pairwise_distances_chunk     # 新数据块的成对距离

        omp_lock_t lock                        # OpenMP锁对象

    # 将余数块计入总块数
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # 线程数不应超过块数
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))  # 将新聚类中心初始化为零
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))           # 将聚类中心权重初始化为零
        omp_init_lock(&lock)                                                      # 初始化OpenMP锁对象
    # 使用 nogil 和 parallel 指令开启并行区域，不使用全局解释器锁（GIL），并设置线程数量为 n_threads。
    with nogil, parallel(num_threads=n_threads):
        # 分配线程局部缓冲区
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        # 分配线程局部缓冲区
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))
        # 分配线程局部缓冲区
        pairwise_distances_chunk = <floating*> malloc(n_samples_chunk * n_clusters * sizeof(floating))

        # 并行循环处理数据块
        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            # 调用函数处理当前数据块
            _update_chunk_dense(
                X[start: end],                   # 当前数据块的特征向量
                sample_weight[start: end],       # 当前数据块的样本权重
                centers_old,                     # 当前聚类中心
                centers_squared_norms,           # 当前聚类中心平方范数
                labels[start: end],              # 当前数据块的样本标签
                centers_new_chunk,               # 线程局部的新聚类中心缓冲区
                weight_in_clusters_chunk,        # 线程局部的聚类权重缓冲区
                pairwise_distances_chunk,        # 线程局部的距离缓冲区
                update_centers                  # 是否更新聚类中心的标志
            )

        # 使用锁进行数据汇总
        if update_centers:
            # 使用 OpenMP 锁避免多个线程对数据进行竞争修改
            omp_set_lock(&lock)
            # 将线程局部的聚类权重和新聚类中心累加到全局变量中
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]

            omp_unset_lock(&lock)

        # 释放线程局部缓冲区
        free(centers_new_chunk)
        free(weight_in_clusters_chunk)
        free(pairwise_distances_chunk)

    # 如果需要更新聚类中心，则进行后续处理
    if update_centers:
        # 销毁 OpenMP 锁
        omp_destroy_lock(&lock)
        # 重新定位空的聚类中心
        _relocate_empty_clusters_dense(
            X, sample_weight, centers_old, centers_new, weight_in_clusters, labels
        )
        # 计算聚类中心的加权平均值
        _average_centers(centers_new, weight_in_clusters)
        # 计算聚类中心的偏移量
        _center_shift(centers_old, centers_new, center_shift)
cdef void _update_chunk_dense(
        const floating[:, ::1] X,                   # 输入：稠密数据块的观测值矩阵 X
        const floating[::1] sample_weight,          # 输入：每个观测值的权重数组 sample_weight
        const floating[:, ::1] centers_old,         # 输入：旧的聚类中心矩阵 centers_old
        const floating[::1] centers_squared_norms,  # 输入：旧的聚类中心的平方范数数组 centers_squared_norms
        int[::1] labels,                            # 输出：每个观测值的标签数组 labels
        floating *centers_new,                      # 输出：更新后的聚类中心矩阵 centers_new
        floating *weight_in_clusters,               # 输出：更新后的聚类中心权重数组 weight_in_clusters
        floating *pairwise_distances,               # 输出：更新后的两两距离矩阵 pairwise_distances
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one dense data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]             # 观测值数量
        int n_clusters = centers_old.shape[0]       # 聚类中心数量
        int n_features = centers_old.shape[1]       # 特征数量

        floating sq_dist, min_sq_dist               # 浮点型变量：距离的平方值和最小平方距离
        int i, j, k, label                          # 整型变量：循环索引和标签

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers.
    # pairwise_distances = ||C||²
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i * n_clusters + j] = centers_squared_norms[j]

    # pairwise_distances += -2 * X.dot(C.T)
    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          -2.0, &X[0, 0], n_features, &centers_old[0, 0], n_features,
          1.0, pairwise_distances, n_clusters)

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i * n_clusters]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i * n_clusters + j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]
            for k in range(n_features):
                centers_new[label * n_features + k] += X[i, k] * sample_weight[i]


def lloyd_iter_chunked_sparse(
        X,                                   # 输入：稀疏数据块的观测值矩阵 X
        const floating[::1] sample_weight,   # 输入：每个观测值的权重数组 sample_weight
        const floating[:, ::1] centers_old,  # 输入：旧的聚类中心矩阵 centers_old
        floating[:, ::1] centers_new,        # 输出：更新后的聚类中心矩阵 centers_new
        floating[::1] weight_in_clusters,    # 输出：更新后的聚类中心权重数组 weight_in_clusters
        int[::1] labels,                     # 输出：每个观测值的标签数组 labels
        floating[::1] center_shift,          # 输出：每个聚类中心的位移数组 center_shift
        int n_threads,                       # 整数：并行线程数
        bint update_centers=True):           # 布尔值：是否更新聚类中心，默认为True
    """Single iteration of K-means lloyd algorithm with sparse input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features), dtype=floating
        The observations to cluster. Must be in CSR format.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.
    """
    """
    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        上一次迭代前的聚类中心，用于存放上一次迭代后的聚类中心。

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        上一次迭代后的聚类中心，用于存放当前迭代期间计算的新聚类中心。如果 `update_centers` 为 False，则 `centers_new` 可以为 None。

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        每个聚类中心分配的所有观测权重的总和的占位符。如果 `update_centers` 为 False，则 `weight_in_clusters` 可以为 None。

    labels : ndarray of shape (n_samples,), dtype=int
        样本的标签分配。

    center_shift : ndarray of shape (n_clusters,), dtype=floating
        老聚类中心和新聚类中心之间的距离。

    n_threads : int
        OpenMP 使用的线程数。

    update_centers : bool
        - 如果为 True，则计算标签和新的聚类中心，即运行算法的 E 步和 M 步。
        - 如果为 False，则仅计算标签，即运行算法的 E 步。当在拟合模型上调用预测函数时特别有用。
    """
    cdef:
        int n_samples = X.shape[0]  # 样本数量
        int n_features = X.shape[1]  # 特征数量
        int n_clusters = centers_old.shape[0]  # 聚类中心数量

    if n_samples == 0:
        # 传入了一个空数组，直接返回（在计算 n_chunks 之前提前返回）。当在具有大量离群点的双分 k 均值模型上调用预测函数时可能会发生这种情况。
        return

    cdef:
        # 选择与稠密数据相同的值。由于稀疏数据中没有预先计算的成对距离矩阵，因此不会有相同的影响。
        # 然而，分块是必要的以获取并行性。
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples  # 每个块的样本数量
        int n_chunks = n_samples // n_samples_chunk  # 块的数量
        int n_samples_rem = n_samples % n_samples_chunk  # 剩余未分块的样本数量
        int chunk_idx  # 块索引
        int start = 0, end = 0  # 块的起始和结束索引

        int j, k  # 循环索引变量

        floating[::1] X_data = X.data  # 稀疏矩阵的数据部分
        int[::1] X_indices = X.indices  # 稀疏矩阵的列索引部分
        int[::1] X_indptr = X.indptr  # 稀疏矩阵的行指针部分

        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)  # 聚类中心的平方范数

        floating *centers_new_chunk  # 新聚类中心的指针
        floating *weight_in_clusters_chunk  # 聚类中心权重的指针

        omp_lock_t lock  # OpenMP 锁

    # 将剩余的块数加到总块数中
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # 线程数不应大于块的数量
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))  # 初始化新聚类中心数组
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))  # 初始化聚类中心权重数组
        omp_init_lock(&lock)  # 初始化 OpenMP 锁
    # 使用 nogil 和 parallel 声明并行区域，指定线程数为 n_threads
    with nogil, parallel(num_threads=n_threads):
        # 分配内存以存储新的聚类中心和聚类权重
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))

        # 遍历每个数据块进行并行更新
        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            # 调用更新函数，处理稀疏数据块的更新操作
            _update_chunk_sparse(
                X_data[X_indptr[start]: X_indptr[end]],         # 稀疏数据的实际数据部分
                X_indices[X_indptr[start]: X_indptr[end]],      # 稀疏数据的列索引
                X_indptr[start: end+1],                         # 稀疏数据的行指针
                sample_weight[start: end],                      # 数据块对应的样本权重
                centers_old,                                    # 当前聚类中心
                centers_squared_norms,                          # 当前聚类中心的平方范数
                labels[start: end],                             # 数据块对应的样本标签
                centers_new_chunk,                              # 新聚类中心的局部缓冲区
                weight_in_clusters_chunk,                       # 聚类权重的局部缓冲区
                update_centers)                                # 是否需要更新聚类中心

        # 合并各线程的局部缓冲区数据到全局数据结构中
        # 如果需要更新聚类中心
        if update_centers:
            # 使用锁保证在多线程环境下对全局数据的安全访问
            omp_set_lock(&lock)
            for j in range(n_clusters):
                # 累加每个聚类的权重
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    # 累加每个聚类的新中心坐标
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]
            omp_unset_lock(&lock)

        # 释放动态分配的内存空间
        free(centers_new_chunk)
        free(weight_in_clusters_chunk)

    # 如果需要更新聚类中心
    if update_centers:
        # 销毁使用的锁对象
        omp_destroy_lock(&lock)
        # 处理稀疏数据中的空聚类
        _relocate_empty_clusters_sparse(
            X_data, X_indices, X_indptr, sample_weight,
            centers_old, centers_new, weight_in_clusters, labels)

        # 计算各聚类中心的平均值
        _average_centers(centers_new, weight_in_clusters)
        # 计算聚类中心的移动量
        _center_shift(centers_old, centers_new, center_shift)
cdef void _update_chunk_sparse(
        const floating[::1] X_data,                 # 输入参数，稀疏数据块的数据
        const int[::1] X_indices,                   # 输入参数，稀疏数据块的列索引
        const int[::1] X_indptr,                    # 输入参数，稀疏数据块的行指针
        const floating[::1] sample_weight,          # 输入参数，每个样本的权重
        const floating[:, ::1] centers_old,         # 输入参数，旧的聚类中心
        const floating[::1] centers_squared_norms,  # 输入参数，聚类中心的平方范数
        int[::1] labels,                            # 输出参数，每个样本的标签
        floating *centers_new,                      # 输出参数，更新后的聚类中心
        floating *weight_in_clusters,               # 输出参数，聚类中心的权重和
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one sparse data chunk.

    计算稀疏数据块对标签和聚类中心的部分贡献。
    """
    cdef:
        int n_samples = labels.shape[0]              # 样本数
        int n_clusters = centers_old.shape[0]         # 聚类中心数
        int n_features = centers_old.shape[1]         # 特征数

        floating sq_dist, min_sq_dist                 # 用于存储距离和最小距离的变量
        int i, j, k, label                            # 循环索引和标签变量
        floating max_floating = FLT_MAX if floating is float else DBL_MAX  # 浮点数最大值

        int s = X_indptr[0]                           # 稀疏数据块的起始索引

    # XXX Precompute the pairwise distances matrix is not worth for sparse
    # currently. Should be tested when BLAS (sparse x dense) matrix
    # multiplication is available.
    for i in range(n_samples):
        min_sq_dist = max_floating                   # 初始化最小平方距离为最大浮点数
        label = 0                                    # 初始化标签为0

        for j in range(n_clusters):
            sq_dist = 0.0                            # 初始化距离为0
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                sq_dist += centers_old[j, X_indices[k]] * X_data[k]  # 计算样本与聚类中心的乘积

            # Instead of computing the full squared distance with each cluster,
            # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to compute
            # the - 2 X.C^T + ||C||² term since the argmin for a given sample
            # only depends on the centers C.
            sq_dist = centers_squared_norms[j] - 2 * sq_dist  # 计算更新后的平方距离
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist                # 更新最小平方距离
                label = j                            # 更新样本的标签为当前聚类中心的索引

        labels[i] = label                             # 将样本的最终标签赋值给输出数组

        if update_centers:
            weight_in_clusters[label] += sample_weight[i]  # 更新聚类中心的权重和
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                centers_new[label * n_features + X_indices[k]] += X_data[k] * sample_weight[i]  # 更新聚类中心的数据
```