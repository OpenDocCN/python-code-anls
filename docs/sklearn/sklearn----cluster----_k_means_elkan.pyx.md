# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_k_means_elkan.pyx`

```
# Author: Andreas Mueller
#
# Licence: BSD 3 clause

# 导入必要的 Cython 模块和 C 库函数
from cython cimport floating
from cython.parallel import prange, parallel
from libc.stdlib cimport calloc, free
from libc.string cimport memset

# 导入并初始化 OpenMP 相关的辅助函数和数据结构
from ..utils._openmp_helpers cimport omp_lock_t
from ..utils._openmp_helpers cimport omp_init_lock
from ..utils._openmp_helpers cimport omp_destroy_lock
from ..utils._openmp_helpers cimport omp_set_lock
from ..utils._openmp_helpers cimport omp_unset_lock

# 导入扩展数学函数和 K-means 相关的共享函数和常量
from ..utils.extmath import row_norms
from ._k_means_common import CHUNK_SIZE
from ._k_means_common cimport _relocate_empty_clusters_dense
from ._k_means_common cimport _relocate_empty_clusters_sparse
from ._k_means_common cimport _euclidean_dense_dense
from ._k_means_common cimport _euclidean_sparse_dense
from ._k_means_common cimport _average_centers
from ._k_means_common cimport _center_shift


def init_bounds_dense(
        const floating[:, ::1] X,                      # IN 输入数据矩阵
        const floating[:, ::1] centers,                # IN 聚类中心
        const floating[:, ::1] center_half_distances,  # IN 聚类中心之间距离的一半
        int[::1] labels,                               # OUT 每个样本的类别标签
        floating[::1] upper_bounds,                    # OUT 每个样本的最大距离上界
        floating[:, ::1] lower_bounds,                 # OUT 每个样本的距离下界矩阵
        int n_threads):                               # 线程数

    """Initialize upper and lower bounds for each sample for dense input data.

    给定稠密输入数据 X、聚类中心 centers 和聚类中心之间的距离的一半，
    初始化每个样本的上界和下界。

    Given X, centers and the pairwise distances divided by 2.0 between the
    centers this calculates the upper bounds and lower bounds for each sample.
    The upper bound for each sample is set to the distance between the sample
    and the closest center.

    The lower bound for each sample is a one-dimensional array of n_clusters.
    For each sample i assume that the previously assigned cluster is c1 and the
    previous closest distance is dist, for a new cluster c2, the
    lower_bound[i][c2] is set to distance between the sample and this new
    cluster, if and only if dist > center_half_distances[c1][c2]. This prevents
    computation of unnecessary distances for each sample to the clusters that
    it is unlikely to be assigned to.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=floating
        The input data.

    centers : ndarray of shape (n_clusters, n_features), dtype=floating
        The cluster centers.

    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        The half of the distance between any 2 clusters centers.

    labels : ndarray of shape(n_samples), dtype=int
        The label for each sample. This array is modified in place.

    upper_bounds : ndarray of shape(n_samples,), dtype=floating
        The upper bound on the distance between each sample and its closest
        cluster center. This array is modified in place.

    lower_bounds : ndarray of shape(n_samples, n_clusters), dtype=floating
        The lower bound on the distance between each sample and each cluster.
        This matrix is modified in place.

    n_threads : int
        The number of OpenMP threads to use.

    """
    # 定义一个二维数组 lower_bounds，表示每个样本到每个聚类中心的最小距离的下界
    lower_bounds : ndarray, of shape(n_samples, n_clusters), dtype=floating
        The lower bound on the distance between each sample and each cluster
        center. This array is modified in place.

    # 定义一个整型变量 n_threads，表示OpenMP将要使用的线程数
    n_threads : int
        The number of threads to be used by openmp.
    """
    # 使用Cython语法定义变量，获取样本数、聚类数和特征数
    cdef:
        int n_samples = X.shape[0]
        int n_clusters = centers.shape[0]
        int n_features = X.shape[1]

        # 定义浮点数变量 min_dist 和 dist，整型变量 best_cluster, i, j
        floating min_dist, dist
        int best_cluster, i, j

    # 使用prange函数并行迭代样本集合
    for i in prange(
        n_samples, num_threads=n_threads, schedule='static', nogil=True
    ):
        # 初始化当前样本最近聚类中心的距离信息
        best_cluster = 0
        min_dist = _euclidean_dense_dense(&X[i, 0], &centers[0, 0],
                                          n_features, False)
        lower_bounds[i, 0] = min_dist
        # 遍历剩余聚类中心，更新当前样本到各个聚类中心的距离信息
        for j in range(1, n_clusters):
            # 检查是否存在更接近的聚类中心，更新最小距离信息
            if min_dist > center_half_distances[best_cluster, j]:
                dist = _euclidean_dense_dense(&X[i, 0], &centers[j, 0],
                                              n_features, False)
                lower_bounds[i, j] = dist
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
        # 记录当前样本所属的最近聚类中心的索引
        labels[i] = best_cluster
        # 记录当前样本到最近聚类中心的最小距离上界
        upper_bounds[i] = min_dist
    # 初始化稀疏输入数据的上下界限制

    # 从参数中获取样本数据 X，是一个稀疏矩阵，以CSR格式存储
    # 输入数据 X 的数据部分
    # 输入数据 X 的列索引部分
    # 输入数据 X 的行指针部分
    def init_bounds_sparse(
        X,
        # 聚类中心的坐标数组，类型为浮点数，输入参数
        const floating[:, ::1] centers,
        # 聚类中心间距的一半的数组，类型为浮点数，输入参数
        const floating[:, ::1] center_half_distances,
        # 每个样本对应的聚类标签数组，类型为整数，输出参数
        int[::1] labels,
        # 每个样本与其最近聚类中心的上界数组，类型为浮点数，输出参数
        floating[::1] upper_bounds,
        # 每个样本与每个聚类中心的下界数组，类型为浮点数，输出参数
        floating[:, ::1] lower_bounds,
        # 使用的线程数
        int n_threads):
        
        """Initialize upper and lower bounds for each sample for sparse input data.

        Given X, centers and the pairwise distances divided by 2.0 between the
        centers this calculates the upper bounds and lower bounds for each sample.
        The upper bound for each sample is set to the distance between the sample
        and the closest center.

        The lower bound for each sample is a one-dimensional array of n_clusters.
        For each sample i assume that the previously assigned cluster is c1 and the
        previous closest distance is dist, for a new cluster c2, the
        lower_bound[i][c2] is set to distance between the sample and this new
        cluster, if and only if dist > center_half_distances[c1][c2]. This prevents
        computation of unnecessary distances for each sample to the clusters that
        it is unlikely to be assigned to.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features), dtype=floating
            The input data. Must be in CSR format.

        centers : ndarray of shape (n_clusters, n_features), dtype=floating
            The cluster centers.

        center_half_distances : ndarray of shape (n_clusters, n_clusters), \
                dtype=floating
            The half of the distance between any 2 clusters centers.

        labels : ndarray of shape(n_samples), dtype=int
            The label for each sample. This array is modified in place.

        upper_bounds : ndarray of shape(n_samples,), dtype=floating
            The upper bound on the distance between each sample and its closest
            cluster center. This array is modified in place.

        lower_bounds : ndarray of shape(n_samples, n_clusters), dtype=floating
            The lower bound on the distance between each sample and each cluster
            center. This array is modified in place.

        n_threads : int
            The number of threads to be used by openmp.
        """

        # 获取样本数和聚类中心数
        cdef:
            int n_samples = X.shape[0]
            int n_clusters = centers.shape[0]

            # 提取稀疏矩阵 X 的数据、列索引和行指针
            floating[::1] X_data = X.data
            int[::1] X_indices = X.indices
            int[::1] X_indptr = X.indptr

            # 声明用于计算最小距离和当前距离的变量
            floating min_dist, dist
            # 用于记录最佳聚类中心和迭代变量
            int best_cluster, i, j

            # 存储聚类中心的范数平方
            floating[::1] centers_squared_norms = row_norms(centers, squared=True)

        # 并行迭代每个样本
        for i in prange(
            n_samples, num_threads=n_threads, schedule='static', nogil=True
        best_cluster = 0
        # 计算当前样本点与第一个簇中心的欧氏距离（稠密向量与稀疏矩阵之间的距离）
        min_dist = _euclidean_sparse_dense(
            X_data[X_indptr[i]: X_indptr[i + 1]],  # 从稀疏数据中获取当前样本的数据部分
            X_indices[X_indptr[i]: X_indptr[i + 1]],  # 获取当前样本的列索引部分
            centers[0],  # 第一个簇的中心点向量
            centers_squared_norms[0],  # 第一个簇的中心点的平方范数
            False)  # 不使用平方根优化

        lower_bounds[i, 0] = min_dist
        # 对于剩余的簇中心，计算最近的簇中心并更新 lower bound
        for j in range(1, n_clusters):
            if min_dist > center_half_distances[best_cluster, j]:
                # 如果当前距离大于某个簇中心的一半距离，则重新计算距离
                dist = _euclidean_sparse_dense(
                    X_data[X_indptr[i]: X_indptr[i + 1]],  # 稀疏数据中当前样本的数据部分
                    X_indices[X_indptr[i]: X_indptr[i + 1]],  # 当前样本的列索引部分
                    centers[j],  # 当前簇中心的向量
                    centers_squared_norms[j],  # 当前簇中心的平方范数
                    False)  # 不使用平方根优化
                lower_bounds[i, j] = dist
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
        # 将当前样本分配到最近的簇
        labels[i] = best_cluster
        # 更新 upper bound
        upper_bounds[i] = min_dist
# 密集型 Elkan K-means 算法的单次迭代，处理密集型输入数据

def elkan_iter_chunked_dense(
        const floating[:, ::1] X,                      # 输入数据集 X，形状为 (样本数, 特征数)
        const floating[::1] sample_weight,             # 每个样本的权重数组
        const floating[:, ::1] centers_old,            # 上一次迭代后的聚类中心数组
        floating[:, ::1] centers_new,                  # 本次迭代计算得到的新聚类中心数组，作为输出
        floating[::1] weight_in_clusters,              # 每个聚类中心所包含样本的权重和，作为输出
        const floating[:, ::1] center_half_distances,  # 聚类中心间的一半距离的数组
        const floating[::1] distance_next_center,      # 每个聚类中心到其最近的下一个聚类中心的距离数组
        floating[::1] upper_bounds,                    # 每个样本到其对应聚类中心的上界距离数组，作为输入输出
        floating[:, ::1] lower_bounds,                 # 每个样本到每个聚类中心的下界距离数组，作为输入输出
        int[::1] labels,                               # 每个样本的聚类标签，作为输入输出
        floating[::1] center_shift,                    # 每个聚类中心的偏移量，作为输出
        int n_threads,                                 # 使用的线程数
        bint update_centers=True):                     # 是否更新聚类中心的标志，True 表示更新

    """Single iteration of K-means Elkan algorithm with dense input.

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
        computed during this iteration.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.

    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        Half pairwise distances between centers.

    distance_next_center : ndarray of shape (n_clusters,), dtype=floating
        Distance between each center its closest center.

    upper_bounds : ndarray of shape (n_samples,), dtype=floating
        Upper bound for the distance between each sample and its center,
        updated inplace.

    lower_bounds : ndarray of shape (n_samples, n_clusters), dtype=floating
        Lower bound for the distance between each sample and each center,
        updated inplace.

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
        int n_samples = X.shape[0]  # 获取样本数目
        int n_features = X.shape[1]  # 获取特征数目
        int n_clusters = centers_new.shape[0]  # 获取聚类中心数目

    if n_samples == 0:
        # 如果样本数为0，则直接返回，不执行后续计算。
        # 这通常发生在使用二分k均值模型进行预测时，大部分样本都是异常值时。
        return

    cdef:
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        # 将样本分成若干块以进行并行计算，每块的大小选择为与Lloyd算法相同的大小。
        int n_chunks = n_samples // n_samples_chunk  # 计算块的数量
        int n_samples_rem = n_samples % n_samples_chunk  # 计算余下的样本数量
        int chunk_idx  # 块的索引
        int start, end  # 块的起始和结束索引

        int i, j, k  # 循环变量

        floating *centers_new_chunk  # 用于存储局部聚类中心的缓冲区
        floating *weight_in_clusters_chunk  # 用于存储局部聚类权重的缓冲区

        omp_lock_t lock  # OpenMP锁

    n_chunks += n_samples != n_chunks * n_samples_chunk
    # 如果总样本数不能整除每块的样本数，则余下的样本也算作一块。

    n_threads = min(n_threads, n_chunks)
    # 确保线程数不超过块的数量。

    if update_centers:
        # 如果需要更新聚类中心，则初始化centers_new和weight_in_clusters为零。
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
        omp_init_lock(&lock)
        # 初始化OpenMP锁。

    with nogil, parallel(num_threads=n_threads):
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))
        # 在每个线程中分配局部缓冲区。

        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk
            # 计算当前块的起始和结束索引。

            _update_chunk_dense(
                X[start: end],
                sample_weight[start: end],
                centers_old,
                center_half_distances,
                distance_next_center,
                labels[start: end],
                upper_bounds[start: end],
                lower_bounds[start: end],
                centers_new_chunk,
                weight_in_clusters_chunk,
                update_centers)
            # 调用更新函数处理当前块的数据。

        if update_centers:
            omp_set_lock(&lock)
            # 使用锁保护以避免不同线程间的竞态条件。
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]
            omp_unset_lock(&lock)
            # 累加局部缓冲区中的结果到全局变量。

        free(centers_new_chunk)
        free(weight_in_clusters_chunk)
        # 释放局部缓冲区的内存。
    # 如果需要更新聚类中心
    if update_centers:
        # 销毁 omp 中的锁对象 &lock
        omp_destroy_lock(&lock)
        # 将空的聚类重新定位到密集形式的数据点
        _relocate_empty_clusters_dense(X, sample_weight, centers_old,
                                       centers_new, weight_in_clusters, labels)

        # 对新的聚类中心进行平均化
        _average_centers(centers_new, weight_in_clusters)
        # 计算聚类中心的变化量
        _center_shift(centers_old, centers_new, center_shift)

        # 更新每个样本的上界和下界
        for i in range(n_samples):
            # 根据聚类中心的移动量更新上界
            upper_bounds[i] += center_shift[labels[i]]

            for j in range(n_clusters):
                # 根据聚类中心的移动量更新下界，并确保下界不小于0
                lower_bounds[i, j] -= center_shift[j]
                if lower_bounds[i, j] < 0:
                    lower_bounds[i, j] = 0
cdef void _update_chunk_dense(
        const floating[:, ::1] X,                      # 输入参数：密集数据块的数据集
        const floating[::1] sample_weight,             # 输入参数：样本权重数组
        const floating[:, ::1] centers_old,            # 输入参数：旧的聚类中心数组
        const floating[:, ::1] center_half_distances,  # 输入参数：聚类中心半距离数组
        const floating[::1] distance_next_center,      # 输入参数：距离下一个聚类中心的数组
        int[::1] labels,                               # 输入输出参数：样本的聚类标签
        floating[::1] upper_bounds,                    # 输入输出参数：每个样本的上界
        floating[:, ::1] lower_bounds,                 # 输入输出参数：每个样本的下界
        floating *centers_new,                         # 输出参数：新的聚类中心数组
        floating *weight_in_clusters,                  # 输出参数：聚类中每个簇的权重
        bint update_centers) noexcept nogil:
    """K-means combined EM step for one dense data chunk.

    计算一个密集数据块对标签和聚类中心的部分贡献。
    """
    cdef:
        int n_samples = labels.shape[0]                # 样本数量
        int n_clusters = centers_old.shape[0]          # 聚类中心数量
        int n_features = centers_old.shape[1]          # 特征数量

        floating upper_bound, distance                 # 临时变量：上界和距离
        int i, j, k, label                            # 循环索引和标签变量
    # 对于每一个样本，依次处理
    for i in range(n_samples):
        # 获取当前样本的上界
        upper_bound = upper_bounds[i]
        # 标志是否边界紧绷
        bounds_tight = 0
        # 获取当前样本的标签
        label = labels[i]

        # 如果当前样本与其当前分配中心的下一个中心距离不大
        # 则可能需要重新分配给另一个中心
        if not distance_next_center[label] >= upper_bound:

            # 遍历所有的聚类中心
            for j in range(n_clusters):

                # 如果以下条件成立，则 center_index 是样本重新标记的良好候选者
                # 我们需要通过重新计算上界和下界来确认这一点
                if (
                    j != label
                    and (upper_bound > lower_bounds[i, j])
                    and (upper_bound > center_half_distances[label, j])
                ):

                    # 通过计算样本与其当前分配中心的实际距离来重新计算上界
                    if not bounds_tight:
                        upper_bound = _euclidean_dense_dense(
                            &X[i, 0], &centers_old[label, 0], n_features, False)
                        lower_bounds[i, label] = upper_bound
                        bounds_tight = 1

                    # 如果条件仍然成立，则计算样本与中心的实际距离
                    # 如果这个距离小于之前的距离，则重新分配标签
                    if (
                        upper_bound > lower_bounds[i, j]
                        or (upper_bound > center_half_distances[label, j])
                    ):
                        distance = _euclidean_dense_dense(
                            &X[i, 0], &centers_old[j, 0], n_features, False)
                        lower_bounds[i, j] = distance
                        if distance < upper_bound:
                            label = j
                            upper_bound = distance

            # 更新样本的标签和上界
            labels[i] = label
            upper_bounds[i] = upper_bound

        # 如果需要更新中心点
        if update_centers:
            # 在相应聚类中心的权重中增加当前样本的权重
            weight_in_clusters[label] += sample_weight[i]
            # 更新相应聚类中心的坐标
            for k in range(n_features):
                centers_new[label * n_features + k] += X[i, k] * sample_weight[i]
def elkan_iter_chunked_sparse(
        X,                                             # 输入：稀疏矩阵，形状为 (n_samples, n_features)，必须是 CSR 格式
        const floating[::1] sample_weight,             # 输入：每个观测的权重，形状为 (n_samples,)
        const floating[:, ::1] centers_old,            # 输入：上一次迭代后的中心点，形状为 (n_clusters, n_features)
        floating[:, ::1] centers_new,                  # 输出：当前迭代计算得到的新中心点，形状为 (n_clusters, n_features)
        floating[::1] weight_in_clusters,              # 输出：每个中心点所分配的所有观测的权重总和，形状为 (n_clusters,)
        const floating[:, ::1] center_half_distances,  # 输入：中心点之间的一半距离的矩阵，形状为 (n_clusters, n_clusters)
        const floating[::1] distance_next_center,      # 输入：每个中心点到其最近中心点的距离，形状为 (n_clusters,)
        floating[::1] upper_bounds,                    # 输入输出：每个样本到其中心点的距离的上界，形状为 (n_samples,)
        floating[:, ::1] lower_bounds,                 # 输入输出：每个样本到每个中心点的距离的下界，形状为 (n_samples, n_clusters)
        int[::1] labels,                               # 输入输出：样本的标签分配，形状为 (n_samples,)
        floating[::1] center_shift,                    # 输出：老中心点和新中心点之间的距离，形状为 (n_clusters,)
        int n_threads,                                 # 输入：要使用的线程数目
        bint update_centers=True):                     # 输入：是否更新中心点的标志，True 表示运行算法的 E 步和 M 步，False 表示只运行 E 步，通常用于预测

    """Single iteration of K-means Elkan algorithm with sparse input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        The observations to cluster. Must be in CSR format.

    sample_weight : ndarray of shape (n_samples,), dtype=floating
        The weights for each observation in X.

    centers_old : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : ndarray of shape (n_clusters, n_features), dtype=floating
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    weight_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.

    center_half_distances : ndarray of shape (n_clusters, n_clusters), \
            dtype=floating
        Half pairwise distances between centers.

    distance_next_center : ndarray of shape (n_clusters,), dtype=floating
        Distance between each center its closest center.

    upper_bounds : ndarray of shape (n_samples,), dtype=floating
        Upper bound for the distance between each sample and its center,
        updated inplace.

    lower_bounds : ndarray of shape (n_samples, n_clusters), dtype=floating
        Lower bound for the distance between each sample and each center,
        updated inplace.

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
    # 定义变量，分别表示样本数、特征数和簇的数量
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_new.shape[0]

    # 如果样本数为0，则输入的数组为空，直接返回，不进行后续操作
    if n_samples == 0:
        # An empty array was passed, do nothing and return early (before
        # attempting to compute n_chunks). This can typically happen when
        # calling the prediction function of a bisecting k-means model with a
        # large fraction of outiers.
        return

    # 使用 Cython 定义变量，存储稀疏矩阵 X 的数据、索引和指针
    cdef:
        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

        # 硬编码每个数据块的样本数。将数据分块是为了并行处理。数据块的大小选择与 Lloyd 算法相同
        int n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_rem = n_samples % n_samples_chunk
        int chunk_idx
        int start, end

        int i, j, k

        # 计算每个旧簇中心的平方范数
        floating[::1] centers_squared_norms = row_norms(centers_old, squared=True)

        # 定义指向新中心点和簇权重的指针
        floating *centers_new_chunk
        floating *weight_in_clusters_chunk

        # 定义 OpenMP 锁
        omp_lock_t lock

    # 如果有余数，则增加一个数据块以包含所有样本
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # 确保线程数不超过数据块数
    n_threads = min(n_threads, n_chunks)

    # 如果需要更新中心点，则初始化 centers_new 和 weight_in_clusters 为零，并初始化 OpenMP 锁
    if update_centers:
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))
        omp_init_lock(&lock)
    with nogil, parallel(num_threads=n_threads):
        # 分配每个线程的本地缓冲区
        centers_new_chunk = <floating*> calloc(n_clusters * n_features, sizeof(floating))
        weight_in_clusters_chunk = <floating*> calloc(n_clusters, sizeof(floating))

        # 并行处理每个数据块
        for chunk_idx in prange(n_chunks, schedule='static'):
            start = chunk_idx * n_samples_chunk
            if chunk_idx == n_chunks - 1 and n_samples_rem > 0:
                end = start + n_samples_rem
            else:
                end = start + n_samples_chunk

            # 调用稀疏更新函数处理当前数据块
            _update_chunk_sparse(
                X_data[X_indptr[start]: X_indptr[end]],
                X_indices[X_indptr[start]: X_indptr[end]],
                X_indptr[start: end+1],
                sample_weight[start: end],
                centers_old,
                centers_squared_norms,
                center_half_distances,
                distance_next_center,
                labels[start: end],
                upper_bounds[start: end],
                lower_bounds[start: end],
                centers_new_chunk,
                weight_in_clusters_chunk,
                update_centers)

        # 从本地缓冲区合并结果
        if update_centers:
            # 使用锁来避免多个线程同时修改共享数据结构造成的竞争条件
            omp_set_lock(&lock)
            for j in range(n_clusters):
                weight_in_clusters[j] += weight_in_clusters_chunk[j]
                for k in range(n_features):
                    centers_new[j, k] += centers_new_chunk[j * n_features + k]
            omp_unset_lock(&lock)

        # 释放本地缓冲区的内存
        free(centers_new_chunk)
        free(weight_in_clusters_chunk)

    # 如果需要更新中心点
    if update_centers:
        # 销毁使用的锁
        omp_destroy_lock(&lock)

        # 重新定位空簇的数据点
        _relocate_empty_clusters_sparse(
            X_data, X_indices, X_indptr, sample_weight,
            centers_old, centers_new, weight_in_clusters, labels)

        # 对新中心点进行加权平均
        _average_centers(centers_new, weight_in_clusters)

        # 计算中心点的移动距离
        _center_shift(centers_old, centers_new, center_shift)

        # 更新数据点的上下界
        for i in range(n_samples):
            upper_bounds[i] += center_shift[labels[i]]

            for j in range(n_clusters):
                lower_bounds[i, j] -= center_shift[j]
                if lower_bounds[i, j] < 0:
                    lower_bounds[i, j] = 0
cdef void _update_chunk_sparse(
        const floating[::1] X_data,                    # 输入参数：稀疏数据块的数据数组
        const int[::1] X_indices,                      # 输入参数：稀疏数据块的索引数组
        const int[::1] X_indptr,                       # 输入参数：稀疏数据块的指针数组
        const floating[::1] sample_weight,             # 输入参数：样本权重数组
        const floating[:, ::1] centers_old,            # 输入参数：旧的聚类中心数组
        const floating[::1] centers_squared_norms,     # 输入参数：聚类中心的平方范数数组
        const floating[:, ::1] center_half_distances,  # 输入参数：聚类中心之间的一半距离数组
        const floating[::1] distance_next_center,      # 输入参数：到下一个聚类中心的距离数组
        int[::1] labels,                               # 输入输出参数：标签数组
        floating[::1] upper_bounds,                    # 输入输出参数：上界数组
        floating[:, ::1] lower_bounds,                 # 输入输出参数：下界数组
        floating *centers_new,                         # 输出参数：新的聚类中心
        floating *weight_in_clusters,                  # 输出参数：聚类中心的权重
        bint update_centers) noexcept nogil:
    """K-means算法的组合EM步骤，用于处理稀疏数据块。

    计算单个数据块对标签和聚类中心的部分贡献。
    """
    cdef:
        int n_samples = labels.shape[0]                # 样本数量
        int n_clusters = centers_old.shape[0]          # 聚类中心的数量
        int n_features = centers_old.shape[1]          # 特征的数量

        floating upper_bound, distance                 # 上界和距离的临时变量
        int i, j, k, label                             # 循环索引和标签

        int s = X_indptr[0]                            # 稀疏数据块的起始位置索引
    # 遍历样本数量的范围
    for i in range(n_samples):
        # 获取当前样本的上界
        upper_bound = upper_bounds[i]
        # 初始化边界紧密度为0
        bounds_tight = 0
        # 获取当前样本的标签
        label = labels[i]

        # 如果当前样本到下一个中心的距离未超过其上界
        if not distance_next_center[label] >= upper_bound:

            # 遍历所有的聚类中心
            for j in range(n_clusters):

                # 如果当前聚类中心不是样本的当前标签，并且满足重新标记的条件
                if (
                    j != label
                    and (upper_bound > lower_bounds[i, j])
                    and (upper_bound > center_half_distances[label, j])
                ):

                    # 如果边界紧密度尚未更新，则重新计算当前样本与其当前分配的中心之间的距离
                    if not bounds_tight:
                        upper_bound = _euclidean_sparse_dense(
                            X_data[X_indptr[i] - s: X_indptr[i + 1] - s],
                            X_indices[X_indptr[i] - s: X_indptr[i + 1] - s],
                            centers_old[label], centers_squared_norms[label], False)
                        lower_bounds[i, label] = upper_bound
                        bounds_tight = 1

                    # 如果条件仍然满足，则计算当前样本与另一个中心之间的距离，并检查是否小于之前的上界
                    if (
                        upper_bound > lower_bounds[i, j]
                        or (upper_bound > center_half_distances[label, j])
                    ):
                        distance = _euclidean_sparse_dense(
                            X_data[X_indptr[i] - s: X_indptr[i + 1] - s],
                            X_indices[X_indptr[i] - s: X_indptr[i + 1] - s],
                            centers_old[j], centers_squared_norms[j], False)
                        lower_bounds[i, j] = distance
                        if distance < upper_bound:
                            label = j
                            upper_bound = distance

            # 更新当前样本的标签和上界
            labels[i] = label
            upper_bounds[i] = upper_bound

        # 如果需要更新中心点
        if update_centers:
            # 更新聚类中心的权重
            weight_in_clusters[label] += sample_weight[i]
            # 遍历当前样本的所有特征
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                # 更新新的聚类中心的值
                centers_new[label * n_features + X_indices[k]] += X_data[k] * sample_weight[i]
```