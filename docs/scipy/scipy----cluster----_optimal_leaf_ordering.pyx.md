# `D:\src\scipysrc\scipy\scipy\cluster\_optimal_leaf_ordering.pyx`

```
# 导入必要的库
import numpy as np
# 导入 Cython 所需的 numpy 功能
cimport numpy as np
# 导入 Cython 编译器
cimport cython
# 导入 C 标准库中的 malloc 和 free 函数
from libc.stdlib cimport malloc, free

# 导入 scipy.spatial.distance 中的 squareform, is_valid_y, is_valid_dm 函数
from scipy.spatial.distance import squareform, is_valid_y, is_valid_dm

# 调用 numpy 的 C API 函数
np.import_array()

# 定义 Cython 函数的装饰器，关闭函数性能分析
@cython.profile(False)
# 关闭边界检查
@cython.boundscheck(False)
# 关闭负索引的循环检查
@cython.wraparound(False)
# 定义内联函数 dual_swap，交换数组 darr 和 iarr 中索引 i1 和 i2 处的值
cdef inline void dual_swap(float* darr, int* iarr,
                           int i1, int i2) noexcept:
    """
    [Taken from Scikit-learn.]

    swap the values at inex i1 and i2 of both darr and iarr
    """
    cdef float dtmp = darr[i1]
    darr[i1] = darr[i2]
    darr[i2] = dtmp

    cdef int itmp = iarr[i1]
    iarr[i1] = iarr[i2]
    iarr[i2] = itmp

# 定义 Cython 函数的装饰器，关闭函数性能分析
@cython.profile(False)
# 关闭边界检查
@cython.boundscheck(False)
# 关闭负索引的循环检查
@cython.wraparound(False)
# 定义函数 _simultaneous_sort，对 dist 和 idx 数组执行同时排序，返回 -1 表示异常
cdef int _simultaneous_sort(float* dist, int* idx,
                            int size) except -1:
    """
    [Taken from Scikit-learn.]


    Perform a recursive quicksort on the dist array, simultaneously
    performing the same swaps on the idx array.  The equivalent in
    numpy (though quite a bit slower) is
    def simultaneous_sort(dist, idx):
        i = np.argsort(dist)
        return dist[i], idx[i]
    """
    cdef int pivot_idx, i, store_idx
    cdef float pivot_val

    # 在小数组情况下，高效地执行排序
    if size <= 1:
        pass
    elif size == 2:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
    elif size == 3:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
        if dist[1] > dist[2]:
            dual_swap(dist, idx, 1, 2)
            if dist[0] > dist[1]:
                dual_swap(dist, idx, 0, 1)
    else:
        # 使用中位数法确定枢轴值（pivot）。
        # 三个值中最小的移动到数组的开头，
        # 中间值（枢轴值）移动到末尾，最大值移动到枢轴值的位置。
        pivot_idx = size // 2
        if dist[0] > dist[size - 1]:
            dual_swap(dist, idx, 0, size - 1)
        if dist[size - 1] > dist[pivot_idx]:
            dual_swap(dist, idx, size - 1, pivot_idx)
            if dist[0] > dist[size - 1]:
                dual_swap(dist, idx, 0, size - 1)
        pivot_val = dist[size - 1]

        # 围绕枢轴值进行分区。在此操作结束时，
        # pivot_idx 将包含枢轴值，左侧的值都比它小，右侧的值都比它大。
        store_idx = 0
        for i in range(size - 1):
            if dist[i] < pivot_val:
                dual_swap(dist, idx, i, store_idx)
                store_idx += 1
        dual_swap(dist, idx, store_idx, size - 1)
        pivot_idx = store_idx

        # 递归地对枢轴两侧的部分进行排序
        if pivot_idx > 1:
            _simultaneous_sort(dist, idx, pivot_idx)
        if pivot_idx + 2 < size:
            _simultaneous_sort(dist + pivot_idx + 1,
                               idx + pivot_idx + 1,
                               size - pivot_idx - 1)
    return 0


注释：

# 否则情况下执行以下操作：
# 使用中位数法确定枢轴值（pivot）。
# 将三个值中最小的移动到数组开头，
# 中间值（枢轴值）移动到数组末尾，最大值移动到枢轴值位置。
pivot_idx = size // 2

# 如果数组第一个元素大于最后一个元素，则交换它们。
if dist[0] > dist[size - 1]:
    dual_swap(dist, idx, 0, size - 1)

# 如果最后一个元素大于中间元素，则将它们交换位置。
if dist[size - 1] > dist[pivot_idx]:
    dual_swap(dist, idx, size - 1, pivot_idx)
    
    # 再次检查并交换第一个和最后一个元素的位置。
    if dist[0] > dist[size - 1]:
        dual_swap(dist, idx, 0, size - 1)

# 将枢轴值设为数组最后一个元素的值。
pivot_val = dist[size - 1]

# 初始化存储索引为0，
# 遍历数组中除了最后一个元素外的所有元素。
# 如果元素小于枢轴值，则将其与存储索引指向的位置交换，并增加存储索引。
store_idx = 0
for i in range(size - 1):
    if dist[i] < pivot_val:
        dual_swap(dist, idx, i, store_idx)
        store_idx += 1

# 将枢轴值移到其最终位置，即存储索引指向的位置。
dual_swap(dist, idx, store_idx, size - 1)
pivot_idx = store_idx

# 递归地对枢轴值两侧的子数组进行排序。
if pivot_idx > 1:
    _simultaneous_sort(dist, idx, pivot_idx)
if pivot_idx + 2 < size:
    _simultaneous_sort(dist + pivot_idx + 1,
                       idx + pivot_idx + 1,
                       size - pivot_idx - 1)

# 返回0，表示排序完成。
return 0
cdef inline void _sort_M_slice(float[:, ::1] M,
                               float* vals, int* idx,
                               int dim1_min, int dim1_max, int dim2_val) noexcept:
    """
    Simultaneously sort indices and values of M[{m}, u] using
    `_simultaneous_sort`

    This is equivalent to :
       m_sort = M[dim1_min:dim1_max, dim2_val].argsort()
       m_iter = np.arange(dim1_min, dim1_max)[m_sort]

    but much faster because we don't have to pay the numpy overhead. This
    matters a lot for the sorting of M[{k}, w] which is executed many times.
    """
    cdef int i
    # Iterate over the range defined by dim1_min and dim1_max
    for i in range(0, dim1_max - dim1_min):
        # Copy values from M to vals array
        vals[i] = M[dim1_min + i, dim2_val]
        # Populate idx array with corresponding indices
        idx[i] = dim1_min + i
    # Call a C function to sort vals and idx simultaneously
    _simultaneous_sort(vals, idx, dim1_max - dim1_min)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:] identify_swaps(int[:, ::1] sorted_Z,
                           double[:, ::1] sorted_D,
                           int[:, ::1] cluster_ranges) noexcept:
    """
    Implements the Optimal Leaf Ordering algorithm described in
    "Fast Optimal leaf ordering for hierarchical clustering"
        Ziv Bar-Joseph, David K. Gifford, Tommi S. Jaakkola
        Bioinformatics, 2001, :doi:`10.1093/bioinformatics/17.suppl_1.S22`

    `sorted_Z` : Linkage list, with 'height' column removed.

    """
    # Determine the number of points based on the length of sorted_Z
    cdef int n_points = len(sorted_Z) + 1

    cdef:
        # (n x n) floats: Matrix M initialized to zeros
        float[:, ::1] M = np.zeros((n_points, n_points), dtype=np.float32)
        # (n x n x 2) booleans: 3D array swap_status initialized to zeros
        int[:, :, :] swap_status = np.zeros((n_points, n_points, 2),
                                            dtype=np.intc)
        # (len(sorted_Z),) integers: Array must_swap initialized to zeros
        int[:] must_swap = np.zeros((len(sorted_Z),), dtype=np.intc)

        # Variables for iteration and cluster indices
        int i, v_l, v_r
        int v_l_min, v_l_max, v_r_min, v_r_max

        # Arrays to store cluster indices
        int u_clusters[2]
        int m_clusters[2]
        int w_clusters[2]
        int k_clusters[2]
        # Variables to store total number of clusters
        int total_u_clusters, total_w_clusters

        # Variables for loop indices
        int u, w, m, k
        int u_min, u_max, m_min, m_max, w_min, w_max, k_min, k_max
        # Variables for swap calculation
        int swap_L, swap_R

        # Pointers for float and integer arrays
        float* m_vals
        int* m_idx
        float* k_vals
        int* k_idx
        int mi, ki

        # Variables for distance calculation
        float min_km_dist
        float cur_min_M, current_M

        # Variables to store best indices
        int best_u = 0, best_w = 0

    # Return the array must_swap
    return must_swap


def optimal_leaf_ordering(Z, D):
    """
    Compute the optimal leaf order for Z (according to D) and return an
    optimally sorted Z.

    We start by sorting and relabelling Z and D according to the current leaf
    order in Z.

    This is because when everything is sorted each cluster (including
    singletons) can be defined by its range over (0...n_points).

    This is used extensively to loop efficiently over the various arrays in the
    algorithm.

    """
    # Import here to avoid import cycles
    from scipy.cluster.hierarchy import leaves_list, is_valid_linkage

    # Check if Z is a valid linkage matrix
    is_valid_linkage(Z, throw=True, name='Z')

    # Check if D is a valid condensed distance matrix
    if is_valid_y(D):
        # Convert D to a squareform distance matrix
        sorted_D = squareform(D)
    # 如果输入的距离矩阵 D 是有效的距离矩阵（condensed 或 square form），则直接使用它，否则抛出异常
    elif is_valid_dm(D):
        sorted_D = D
    else:
        raise("Not a valid distance matrix (neither condensed nor square form)")

    # 计算点的数量 n_points 和聚类的数量 n_clusters
    n_points = Z.shape[0] + 1
    n_clusters = 2*n_points - 1

    # 获取当前的线性叶节点排序
    sorted_leaves = leaves_list(Z)

    # 创建从原始顺序到排序顺序的映射
    original_order_to_sorted_order = dict((orig_i, sorted_i) for sorted_i, orig_i
                                          in enumerate(sorted_leaves))

    # 重新构建 Z 连接矩阵，以引用排序后的位置而不是输入位置。同时移除 'height' 列以便整体转换为整数类型并简化传递给上述 C 函数的过程。
    sorted_Z = []
    for (v_l, v_r, _, v_size) in Z:
        if v_l < n_points:
            v_l = original_order_to_sorted_order[int(v_l)]
        if v_r < n_points:
            v_r = original_order_to_sorted_order[int(v_r)]

        sorted_Z.append([v_l, v_r, v_size])
    sorted_Z = np.array(sorted_Z, dtype=np.int32)

    # 根据叶节点的顺序对距离矩阵 D 进行排序
    sorted_D = sorted_D[sorted_leaves, :]
    sorted_D = sorted_D[:, sorted_leaves].copy(order='C')

    # 定义每个集群的范围，解释见上文。
    cluster_ranges = np.zeros((n_clusters, 2))
    cluster_ranges[np.arange(n_points), 0] = np.arange(n_points)
    cluster_ranges[np.arange(n_points), 1] = np.arange(n_points) + 1
    for link_i, (v_l, v_r, v_size) in enumerate(sorted_Z):
        v = link_i + n_points
        cluster_ranges[v, 0] = cluster_ranges[v_l, 0]
        cluster_ranges[v, 1] = cluster_ranges[v_r, 1]
    cluster_ranges = cluster_ranges.astype(np.int32).copy(order='C')

    # 获取必须进行交换的节点
    must_swap = identify_swaps(sorted_Z, sorted_D, cluster_ranges)

    # 为了"旋转"树节点，需要考虑目标节点及其所有后代节点的左右孩子节点
    # 通过记录每个节点需要进行交换的总次数（如果节点本身需要交换一次，每个父节点需要交换一次），并取模2得出是否需要进行交换
    is_descendant = np.zeros((n_clusters - n_points, n_clusters - n_points),
                             dtype=int)
    for i, (v_l, v_r, v_size) in enumerate(sorted_Z):
        is_descendant[i, i] = 1
        if v_l >= n_points:
            is_descendant[i, v_l - n_points] = 1
            is_descendant[i, :] += is_descendant[v_l - n_points, :]
        if v_r >= n_points:
            is_descendant[i, v_r - n_points] = 1
            is_descendant[i, :] += is_descendant[v_r - n_points, :]

    # 为了"旋转"树节点，需要交换其左右孩子节点，并递归处理其所有子孙节点
    applied_swap = (np.array(is_descendant).astype(bool)
                    * np.array(must_swap).reshape(-1, 1))
    final_swap = applied_swap.sum(axis=0) % 2
    # 创建一个新的联接矩阵，根据需要进行交换操作。
    swapped_Z = []
    for i, (in_l, in_r, h, v_size) in enumerate(Z):
        # 如果 final_swap 列表中对应位置为 True，则交换左右节点
        if final_swap[i]:
            out_l = in_r
            out_r = in_l
        else:
            out_r = in_r
            out_l = in_l
        # 将交换后的节点以及原始高度和节点尺寸添加到新的联接矩阵中
        swapped_Z.append((out_l, out_r, h, v_size))
    # 将列表转换为 NumPy 数组
    swapped_Z = np.array(swapped_Z)

    # 返回经过节点交换后的新联接矩阵
    return swapped_Z
```