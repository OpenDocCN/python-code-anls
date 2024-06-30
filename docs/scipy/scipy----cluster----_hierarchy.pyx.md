# `D:\src\scipysrc\scipy\scipy\cluster\_hierarchy.pyx`

```
# 设置 Cython 的编译选项：禁用边界检查、禁用循环包装、启用 C 除法
# 这些选项可以提高程序性能，但需要谨慎使用以确保代码的正确性和安全性
cython: boundscheck=False, wraparound=False, cdivision=True

# 导入 NumPy 库，并将其作为 np 别名导入
import numpy as np

# 使用 cimport 导入 NumPy 库的 C 扩展版本
cimport numpy as np

# 从 libc.math 中导入 sqrt 和 INFINITY 函数
from libc.math cimport sqrt, INFINITY

# 从 libc.string 中导入 memset 函数
from libc.string cimport memset

# 从 cpython.mem 中导入 PyMem_Malloc 和 PyMem_Free 函数
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# 定义一个无符号字符类型的别名 uchar
ctypedef unsigned char uchar

# 调用 NumPy 的 import_array() 函数，初始化 NumPy C API
np.import_array()

# 包含 _hierarchy_distance_update.pxi 文件，该文件包括 linkage_distance_update 函数和
# 支持的链接方法的距离更新函数
include "_hierarchy_distance_update.pxi"

# 定义一个 linkage_distance_update 函数指针数组，包含多个链接方法的距离更新函数
cdef linkage_distance_update *linkage_methods = [
    _single, _complete, _average, _centroid, _median, _ward, _weighted]

# 包含 _structures.pxi 文件，用于定义数据结构和常量
include "_structures.pxi"

# 定义一个内联函数 condensed_index，计算在一个 n x n 的压缩矩阵中元素 (i, j) 的压缩索引
cdef inline np.npy_int64 condensed_index(np.npy_int64 n, np.npy_int64 i,
                                         np.npy_int64 j) noexcept:
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - (i * (i + 1) / 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) / 2) + (i - j - 1)

# 定义一个内联函数 is_visited，检查节点 i 是否已被访问
cdef inline int is_visited(uchar *bitset, int i) noexcept:
    """
    Check if node i was visited.
    """
    return bitset[i >> 3] & (1 << (i & 7))

# 定义一个内联函数 set_visited，标记节点 i 已被访问
cdef inline void set_visited(uchar *bitset, int i) noexcept:
    """
    Mark node i as visited.
    """
    bitset[i >> 3] |= 1 << (i & 7)

# 定义一个 Cython 可调用函数 calculate_cluster_sizes，计算每个聚类的大小
# 结果存储在链接矩阵的第四列中
cpdef void calculate_cluster_sizes(double[:, :] Z, double[:] cs, int n) noexcept:
    """
    Calculate the size of each cluster. The result is the fourth column of
    the linkage matrix.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix. The fourth column can be empty.
    cs : ndarray
        The array to store the sizes.
    n : ndarray
        The number of observations.
    """
    cdef int i, child_l, child_r

    for i in range(n - 1):
        child_l = <int>Z[i, 0]
        child_r = <int>Z[i, 1]

        if child_l >= n:
            cs[i] += cs[child_l - n]
        else:
            cs[i] += 1

        if child_r >= n:
            cs[i] += cs[child_r - n]
        else:
            cs[i] += 1

# 定义一个 Cython 可调用函数 cluster_dist，根据距离准则形成扁平聚类
# 参数说明见函数内部注释
def cluster_dist(const double[:, :] Z, int[:] T, double cutoff, int n):
    """
    Form flat clusters by distance criterion.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
    T : ndarray
        The array to store the cluster numbers. The i'th observation belongs to
        cluster ``T[i]``.
    cutoff : double
        Clusters are formed when distances are less than or equal to `cutoff`.
    n : int
        The number of observations.
    """
    cdef double[:] max_dists = np.ndarray(n, dtype=np.float64)
    get_max_dist_for_each_cluster(Z, max_dists, n)
    cluster_monocrit(Z, max_dists, T, cutoff, n)

# 定义一个 Cython 可调用函数 cluster_in，根据不一致性准则形成扁平聚类
# 参数说明见函数内部注释
def cluster_in(const double[:, :] Z, const double[:, :] R, int[:] T, double cutoff, int n):
    """
    Form flat clusters by inconsistent criterion.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
    R : ndarray
        The inconsistent matrix.
    # 创建一个双精度浮点数数组，用于存储每个观测的最大不一致值
    cdef double[:] max_inconsists = np.ndarray(n, dtype=np.float64)
    
    # 调用函数 `get_max_Rfield_for_each_cluster`，计算每个观测的最大R字段值，并存储在 max_inconsists 中
    get_max_Rfield_for_each_cluster(Z, R, max_inconsists, n, 3)
    
    # 调用函数 `cluster_monocrit`，进行聚类并将结果存储在数组 T 中，使用给定的不一致性阈值 cutoff
    cluster_monocrit(Z, max_inconsists, T, cutoff, n)
    # 定义一个函数，用于根据 maxclust 准则形成扁平聚类
    def cluster_maxclust_dist(const double[:, :] Z, int[:] T, int n, int mc):
        """
        Form flat clusters by maxclust criterion.

        Parameters
        ----------
        Z : ndarray
            The linkage matrix.
        T : ndarray
            The array to store the cluster numbers. The i'th observation belongs to
            cluster ``T[i]``.
        n : int
            The number of observations.
        mc : int
            The maximum number of clusters.
        """
        # 创建一个数组 max_dists，用于存储每个观测点的最大距离
        cdef double[:] max_dists = np.ndarray(n, dtype=np.float64)
        # 调用函数 get_max_dist_for_each_cluster，计算每个聚类的最大距离
        get_max_dist_for_each_cluster(Z, max_dists, n)
        # 调用 cluster_maxclust_monocrit 函数，使用 maxclust 准则形成扁平聚类
        # 这个函数应该使用 O(n) 算法
        cluster_maxclust_monocrit(Z, max_dists, T, n, mc)


    # 定义一个 CPython 函数，用于根据 maxclust_monocrit 准则形成扁平聚类
    cpdef void cluster_maxclust_monocrit(const double[:, :] Z, const double[:] MC, int[:] T,
                                         int n, int max_nc) noexcept:
        """
        Form flat clusters by maxclust_monocrit criterion.

        Parameters
        ----------
        Z : ndarray
            The linkage matrix.
        MC : ndarray
            The monotonic criterion array.
        T : ndarray
            The array to store the cluster numbers. The i'th observation belongs to
            cluster ``T[i]``.
        n : int
            The number of observations.
        max_nc : int
            The maximum number of clusters.
        """
        # 定义变量和数组
        cdef int i, k, i_lc, i_rc, root, nc, lower_idx, upper_idx
        cdef double thresh
        # 创建一个整型数组 curr_node，用于存储当前节点信息
        cdef int[:] curr_node = np.ndarray(n, dtype=np.intc)

        # 计算需要分配的 visited 数组大小
        cdef int visited_size = (((n * 2) - 1) >> 3) + 1
        # 分配内存并将 visited 强制转换为 uchar 指针
        cdef uchar *visited = <uchar *>PyMem_Malloc(visited_size)
        if not visited:
            raise MemoryError

        # 初始化 lower_idx 和 upper_idx
        lower_idx = 0
        upper_idx = n - 1
    # 当上界索引与下界索引之差大于1时，执行循环
    while upper_idx - lower_idx > 1:
        # 计算中间索引
        i = (lower_idx + upper_idx) >> 1
        # 获取当前阈值
        thresh = MC[i]

        # 将 visited 数组初始化为0
        memset(visited, 0, visited_size)
        # 初始化变量 nc 为0，k 为0，并设置当前节点为最后一个非叶子节点的索引
        nc = 0
        k = 0
        curr_node[0] = 2 * n - 2

        # 进入内部循环，处理节点栈中的节点
        while k >= 0:
            # 获取当前栈顶节点的左右子节点的索引
            root = curr_node[k] - n
            i_lc = <int>Z[root, 0]
            i_rc = <int>Z[root, 1]

            # 如果当前节点的 MC 值小于等于阈值 thresh，则形成一个簇
            if MC[root] <= thresh:  # this subtree forms a cluster
                nc += 1
                # 如果簇的数量超过了 max_nc，则视为非法情况，中断循环
                if nc > max_nc:  # illegal
                    break
                # 弹出栈顶节点，并标记其左右子节点为已访问
                k -= 1
                set_visited(visited, i_lc)
                set_visited(visited, i_rc)
                continue

            # 如果左子节点未访问过，则标记为已访问并入栈
            if not is_visited(visited, i_lc):
                set_visited(visited, i_lc)
                if i_lc >= n:
                    k += 1
                    curr_node[k] = i_lc
                    continue
                else:  # singleton cluster，单节点簇
                    nc += 1
                    if nc > max_nc:
                        break

            # 如果右子节点未访问过，则标记为已访问并入栈
            if not is_visited(visited, i_rc):
                set_visited(visited, i_rc)
                if i_rc >= n:
                    k += 1
                    curr_node[k] = i_rc
                    continue
                else:  # singleton cluster，单节点簇
                    nc += 1
                    if nc > max_nc:
                        break

            # 弹出栈顶节点
            k -= 1

        # 如果簇的数量超过了 max_nc，则将下界索引设置为当前索引 i
        if nc > max_nc:
            lower_idx = i
        else:
            # 否则将上界索引设置为当前索引 i
            upper_idx = i

    # 释放 visited 数组所占内存
    PyMem_Free(visited)
    # 对找到的最优阈值进行簇分析
    cluster_monocrit(Z, MC, T, MC[upper_idx], n)
# 定义了一个 Cython 函数，用于根据单调性准则形成扁平化的聚类

cpdef void cluster_monocrit(const double[:, :] Z, const double[:] MC, int[:] T,
                            double cutoff, int n) noexcept:
    """
    Form flat clusters by monocrit criterion.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
        链接矩阵，描述聚类过程中的合并顺序和距离
    MC : ndarray
        The monotonic criterion array.
        单调性准则数组，描述聚类过程中的单调性规则
    T : ndarray
        The array to store the cluster numbers. The i'th observation belongs to
        cluster ``T[i]``.
        存储聚类编号的数组，第 i 个观测属于聚类 ``T[i]``
    cutoff : double
        Clusters are formed when the MC values are less than or equal to
        `cutoff`.
        当单调性准则值小于或等于 `cutoff` 时形成聚类
    n : int
        The number of observations.
        观测数目
    """
    cdef int k, i_lc, i_rc, root, n_cluster = 0, cluster_leader = -1
    cdef int[:] curr_node = np.ndarray(n, dtype=np.intc)

    # 计算 visited 数组的大小并申请内存
    cdef int visited_size = (((n * 2) - 1) >> 3) + 1
    cdef uchar *visited = <uchar *>PyMem_Malloc(visited_size)
    if not visited:
        raise MemoryError
    # 将 visited 数组初始化为 0
    memset(visited, 0, visited_size)

    k = 0
    curr_node[0] = 2 * n - 2
    while k >= 0:
        root = curr_node[k] - n
        i_lc = <int>Z[root, 0]
        i_rc = <int>Z[root, 1]

        # 如果找到了一个聚类且符合单调性准则
        if cluster_leader == -1 and MC[root] <= cutoff:
            cluster_leader = root
            n_cluster += 1

        # 处理左子节点
        if i_lc >= n and not is_visited(visited, i_lc):
            set_visited(visited, i_lc)
            k += 1
            curr_node[k] = i_lc
            continue

        # 处理右子节点
        if i_rc >= n and not is_visited(visited, i_rc):
            set_visited(visited, i_rc)
            k += 1
            curr_node[k] = i_rc
            continue

        # 处理叶子节点的情况
        if i_lc < n:
            if cluster_leader == -1:
                n_cluster += 1
            T[i_lc] = n_cluster

        if i_rc < n:
            if cluster_leader == -1:
                n_cluster += 1
            T[i_rc] = n_cluster

        # 如果回到了聚类的领导节点
        if cluster_leader == root:
            cluster_leader = -1
        k -= 1

    # 释放 visited 数组占用的内存
    PyMem_Free(visited)


def cophenetic_distances(const double[:, :] Z, double[:] d, int n):
    """
    Calculate the cophenetic distances between each observation

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
        链接矩阵，描述聚类过程中的合并顺序和距离
    d : ndarray
        The condensed matrix to store the cophenetic distances.
        存储 cophenetic 距离的压缩矩阵
    n : int
        The number of observations.
        观测数目
    """
    cdef int i, j, k, root, i_lc, i_rc, n_lc, n_rc, right_start
    cdef double dist
    cdef int[:] curr_node = np.ndarray(n, dtype=np.intc)
    cdef int[:] members = np.ndarray(n, dtype=np.intc)
    cdef int[:] left_start = np.ndarray(n, dtype=np.intc)

    # 计算 visited 数组的大小并申请内存
    cdef int visited_size = (((n * 2) - 1) >> 3) + 1
    cdef uchar *visited = <uchar *>PyMem_Malloc(visited_size)
    if not visited:
        raise MemoryError
    # 将 visited 数组初始化为 0
    memset(visited, 0, visited_size)

    k = 0
    curr_node[0] = 2 * n - 2
    left_start[0] = 0
    # 当 k 大于等于 0 时，执行循环
    while k >= 0:
        # 从当前节点中获取根节点的索引，并转换成原始节点的索引
        root = curr_node[k] - n
        # 获取根节点的左子节点和右子节点的索引
        i_lc = <int>Z[root, 0]
        i_rc = <int>Z[root, 1]

        # 如果左子节点不是叶子节点
        if i_lc >= n:
            # 获取左子节点的聚类大小
            n_lc = <int>Z[i_lc - n, 3]

            # 如果左子节点未被访问过
            if not is_visited(visited, i_lc):
                # 标记左子节点为已访问
                set_visited(visited, i_lc)
                # 将左子节点加入当前节点列表
                k += 1
                curr_node[k] = i_lc
                # 继承父节点的左边界
                left_start[k] = left_start[k - 1]
                continue  # 继续访问左子树
        else:
            # 如果左子节点是叶子节点，设定聚类大小为1，并将其加入成员列表
            n_lc = 1
            members[left_start[k]] = i_lc

        # 如果右子节点不是叶子节点
        if i_rc >= n:
            # 获取右子节点的聚类大小
            n_rc = <int>Z[i_rc - n, 3]

            # 如果右子节点未被访问过
            if not is_visited(visited, i_rc):
                # 标记右子节点为已访问
                set_visited(visited, i_rc)
                # 将右子节点加入当前节点列表
                k += 1
                curr_node[k] = i_rc
                # 计算右子节点的左边界
                left_start[k] = left_start[k - 1] + n_lc
                continue  # 继续访问右子树
        else:
            # 如果右子节点是叶子节点，设定聚类大小为1，并将其加入成员列表
            n_rc = 1
            members[left_start[k] + n_lc] = i_rc

        # 回到当前子树的根节点
        dist = Z[root, 2]
        right_start = left_start[k] + n_lc
        # 对当前子树中的节点进行两两组合，计算它们之间的距离，并存入距离数组中
        for i in range(left_start[k], right_start):
            for j in range(right_start, right_start + n_rc):
                d[condensed_index(n, members[i], members[j])] = dist

        k -= 1  # 回到父节点

    # 释放访问数组的内存
    PyMem_Free(visited)
# 定义 Cython 的 cpdef 函数，获取每个非单点聚类的最大 R 值统计量
# 对于第 i 个非单点聚类，max_rfs[i] = max{R[j, rf] j 是 i 的后代}
cpdef void get_max_Rfield_for_each_cluster(const double[:, :] Z, const double[:, :] R,
                                           double[:] max_rfs, int n, int rf) noexcept:
    """
    Get the maximum statistic for each non-singleton cluster. For the i'th
    non-singleton cluster, max_rfs[i] = max{R[j, rf] j is a descendent of i}.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
    R : ndarray
        The R matrix.
    max_rfs : ndarray
        The array to store the result. Note that this input arrays gets
        modified in-place.
    n : int
        The number of observations.
    rf : int
        Indicate which column of `R` is used.
    """
    # 定义局部变量
    cdef int k, i_lc, i_rc, root
    cdef double max_rf, max_l, max_r
    # 创建整数数组 curr_node，用于存储当前节点
    cdef int[:] curr_node = np.ndarray(n, dtype=np.intc)

    # 计算需要分配的 visited 数组大小
    cdef int visited_size = (((n * 2) - 1) >> 3) + 1
    # 分配并初始化 visited 数组
    cdef uchar *visited = <uchar *>PyMem_Malloc(visited_size)
    if not visited:
        raise MemoryError
    memset(visited, 0, visited_size)

    # 初始化根节点
    k = 0
    curr_node[0] = 2 * n - 2
    # 开始深度优先搜索
    while k >= 0:
        root = curr_node[k] - n
        i_lc = <int>Z[root, 0]
        i_rc = <int>Z[root, 1]

        # 处理左子节点
        if i_lc >= n and not is_visited(visited, i_lc):
            set_visited(visited, i_lc)
            k += 1
            curr_node[k] = i_lc
            continue

        # 处理右子节点
        if i_rc >= n and not is_visited(visited, i_rc):
            set_visited(visited, i_rc)
            k += 1
            curr_node[k] = i_rc
            continue

        # 计算当前节点的最大 R 值统计量
        max_rf = R[root, rf]
        if i_lc >= n:
            max_l = max_rfs[i_lc - n]
            if max_l > max_rf:
                max_rf = max_l
        if i_rc >= n:
            max_r = max_rfs[i_rc - n]
            if max_r > max_rf:
                max_rf = max_r
        max_rfs[root] = max_rf

        k -= 1

    # 释放内存
    PyMem_Free(visited)


# 定义 Cython 的 cpdef 函数，获取每个非单点聚类的最大不一致系数
cpdef get_max_dist_for_each_cluster(const double[:, :] Z, double[:] MD, int n):
    """
    Get the maximum inconsistency coefficient for each non-singleton cluster.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
    MD : ndarray
        The array to store the result (hence this input array gets modified
        in-place).
    n : int
        The number of observations.
    """
    # 定义局部变量
    cdef int k, i_lc, i_rc, root
    cdef double max_dist, max_l, max_r
    cdef int[:] curr_node = np.ndarray(n, dtype=np.intc)

    # 计算需要分配的 visited 数组大小
    cdef int visited_size = (((n * 2) - 1) >> 3) + 1
    # 分配并初始化 visited 数组
    cdef uchar *visited = <uchar *>PyMem_Malloc(visited_size)
    if not visited:
        raise MemoryError
    memset(visited, 0, visited_size)

    # 初始化根节点
    k = 0
    curr_node[0] = 2 * n - 2
    # 当 k 大于等于 0 时执行循环，k 表示当前处理的节点在 curr_node 中的索引
    while k >= 0:
        # 获取当前处理的节点在原始矩阵 Z 中的索引
        root = curr_node[k] - n
        # 获取当前节点的左子节点和右子节点在 Z 矩阵中的索引
        i_lc = <int>Z[root, 0]
        i_rc = <int>Z[root, 1]

        # 如果左子节点是一个真正的叶子节点且未被访问过，则继续处理左子节点
        if i_lc >= n and not is_visited(visited, i_lc):
            set_visited(visited, i_lc)  # 标记左子节点为已访问
            k += 1  # 将左子节点索引加入 curr_node 列表，继续处理
            curr_node[k] = i_lc
            continue

        # 如果右子节点是一个真正的叶子节点且未被访问过，则继续处理右子节点
        if i_rc >= n and not is_visited(visited, i_rc):
            set_visited(visited, i_rc)  # 标记右子节点为已访问
            k += 1  # 将右子节点索引加入 curr_node 列表，继续处理
            curr_node[k] = i_rc
            continue

        # 计算当前节点的最大距离
        max_dist = Z[root, 2]

        # 如果左子节点是内部节点，则比较其最大距离和当前节点的最大距离
        if i_lc >= n:
            max_l = MD[i_lc - n]
            if max_l > max_dist:
                max_dist = max_l

        # 如果右子节点是内部节点，则比较其最大距离和当前节点的最大距离
        if i_rc >= n:
            max_r = MD[i_rc - n]
            if max_r > max_dist:
                max_dist = max_r

        # 将当前节点的最大距离记录到 MD 数组中
        MD[root] = max_dist

        # 处理完当前节点后，将 k 减一，准备处理下一个节点
        k -= 1

    # 释放 visited 的内存
    PyMem_Free(visited)
# 计算不一致性统计信息
def inconsistent(const double[:, :] Z, double[:, :] R, int n, int d):
    """
    Calculate the inconsistency statistics.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
    R : ndarray
        A (n - 1) x 4 matrix to store the result (hence this input array is
        modified in-place). The inconsistency statistics ``R[i]`` are calculated
        over `d` levels below cluster ``i``.
        ``R[i, 0]`` is the mean of distances.
        ``R[i, 1]`` is the standard deviation of distances.
        ``R[i, 2]`` is the number of clusters included.
        ``R[i, 3]`` is the inconsistency coefficient.

        .. math:: \\frac{\\mathtt{Z[i,2]}-\\mathtt{R[i,0]}} {R[i,1]}

    n : int
        The number of observations.
    d : int
        The number of levels included in calculation below a node.
    """
    # 定义变量和数组
    cdef int i, k, i_lc, i_rc, root, level_count
    cdef int[:] curr_node = np.ndarray(n, dtype=np.intc)
    cdef double level_sum, level_std_sum, level_std, dist

    # 计算 visited 数组的大小，并分配内存
    cdef int visited_size = (((n * 2) - 1) >> 3) + 1
    cdef uchar *visited = <uchar *>PyMem_Malloc(visited_size)
    if not visited:
        raise MemoryError

    # 遍历 Z 中的每个节点
    for i in range(n - 1):
        k = 0
        level_count = 0
        level_sum = 0
        level_std_sum = 0
        memset(visited, 0, visited_size)
        curr_node[0] = i

        # 深度优先搜索遍历每个节点的子节点
        while k >= 0:
            root = curr_node[k]

            # 如果未达到指定的层数 d，继续向下遍历子节点
            if k < d - 1:
                i_lc = <int>Z[root, 0]
                if i_lc >= n and not is_visited(visited, i_lc):
                    set_visited(visited, i_lc)
                    k += 1
                    curr_node[k] = i_lc - n
                    continue

                i_rc = <int>Z[root, 1]
                if i_rc >= n and not is_visited(visited, i_rc):
                    set_visited(visited, i_rc)
                    k += 1
                    curr_node[k] = i_rc - n
                    continue

            # 计算距离的统计信息
            dist = Z[root, 2]
            level_count += 1
            level_sum += dist
            level_std_sum += dist * dist
            k -= 1

        # 计算平均距离和包含的子节点数目
        R[i, 0] = level_sum / level_count
        R[i, 2] = level_count

        # 计算标准差和不一致性系数
        if level_count < 2:
            level_std = (level_std_sum - (level_sum * level_sum)) / level_count
        else:
            level_std = ((level_std_sum -
                         ((level_sum * level_sum) / level_count)) /
                         (level_count - 1))
        if level_std > 0:
            level_std = sqrt(level_std)
            R[i, 1] = level_std
            R[i, 3] = (Z[i, 2] - R[i, 0]) / level_std
        else:
            R[i, 1] = 0

    # 释放分配的内存
    PyMem_Free(visited)
    # 声明整型变量 k, i_lc, i_rc, root, cid_lc, cid_rc, leader_idx, result，并初始化为 -1
    cdef int k, i_lc, i_rc, root, cid_lc, cid_rc, leader_idx, result = -1
    
    # 创建一个大小为 n 的整型数组 curr_node，并初始化为 0
    cdef int[:] curr_node = np.ndarray(n, dtype=np.intc)
    
    # 创建一个大小为 n*2-1 的整型数组 cluster_ids，并初始化为 0
    cdef int[:] cluster_ids = np.ndarray(n * 2 - 1, dtype=np.intc)

    # 计算 visited 数组的大小，并分配内存
    cdef int visited_size = (((n * 2) - 1) >> 3) + 1
    cdef uchar *visited = <uchar *>PyMem_Malloc(visited_size)
    
    # 检查内存分配是否成功，若失败则抛出 MemoryError 异常
    if not visited:
        raise MemoryError
    
    # 将 visited 数组的内存空间初始化为 0
    memset(visited, 0, visited_size)

    # 将 T 数组的前 n 个元素复制到 cluster_ids 数组的前 n 个位置
    cluster_ids[:n] = T[:]
    
    # 将 cluster_ids 数组的剩余位置（n 到 n*2-2）初始化为 -1
    cluster_ids[n:] = -1
    
    # 初始化 k 为 0，并将 curr_node 数组的第一个元素设置为 2*n-2
    k = 0
    curr_node[0] = 2 * n - 2
    leader_idx = 0
    
    # 开始迭代处理 curr_node 数组直到结束
    while k >= 0:
        # 计算当前节点的左右子节点在 cluster linkage matrix Z 中的索引
        root = curr_node[k] - n
        i_lc = <int>Z[root, 0]
        i_rc = <int>Z[root, 1]

        # 如果左子节点超过了 n 且未被访问过，则标记为已访问并将其加入 curr_node 数组
        if i_lc >= n and not is_visited(visited, i_lc):
            set_visited(visited, i_lc)
            k += 1
            curr_node[k] = i_lc
            continue

        # 如果右子节点超过了 n 且未被访问过，则标记为已访问并将其加入 curr_node 数组
        if i_rc >= n and not is_visited(visited, i_rc):
            set_visited(visited, i_rc)
            k += 1
            curr_node[k] = i_rc
            continue

        # 获取左右子节点所属的簇 ID
        cid_lc = cluster_ids[i_lc]
        cid_rc = cluster_ids[i_rc]

        # 如果左右子节点属于同一个簇，则将当前节点标记为该簇
        if cid_lc == cid_rc:
            cluster_ids[root + n] = cid_lc
        else:
            # 如果左右子节点分别为两个不同的簇的领导节点，则将其记录到 L 和 M 数组中
            if cid_lc != -1:
                if leader_idx >= nc:
                    result = root + n
                    break
                L[leader_idx] = i_lc
                M[leader_idx] = cid_lc
                leader_idx += 1

            if cid_rc != -1:
                if leader_idx >= nc:
                    result = root + n
                    break
                L[leader_idx] = i_rc
                M[leader_idx] = cid_rc
                leader_idx += 1

            # 将当前节点标记为未分配任何簇
            cluster_ids[root + n] = -1

        # 继续处理 curr_node 数组的下一个节点
        k -= 1

    # 如果没有找到错误节点，则检查最后一次合并的节点是否属于相同簇，并记录结果
    if result == -1:
        i_lc = <int>Z[n - 2, 0]
        i_rc = <int>Z[n - 2, 1]
        cid_lc = cluster_ids[i_lc]
        cid_rc = cluster_ids[i_rc]
        if cid_lc == cid_rc and cid_lc != -1:
            if leader_idx >= nc:
                result = 2 * n - 2
            else:
                L[leader_idx] = 2 * n - 2
                M[leader_idx] = cid_lc

    # 释放 visited 数组的内存空间
    PyMem_Free(visited)
    
    # 返回处理结果，-1 表示成功
    return result
# 执行层次聚类的函数
def linkage(double[:] dists, np.npy_int64 n, int method):
    """
    Perform hierarchy clustering.

    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    method : int
        The linkage method. 0: single 1: complete 2: average 3: centroid
        4: median 5: ward 6: weighted

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """
    # 创建一个空的数组来存储层次聚类的结果
    Z_arr = np.empty((n - 1, 4))
    # 将数组转换为Cython的双精度二维数组
    cdef double[:, :] Z = Z_arr

    # 定义整数变量和索引
    cdef int i, j, k, x = 0, y = 0, nx, ny, ni, id_x, id_y, id_i
    cdef np.npy_int64 i_start
    cdef double current_min

    # 初始化一个存储两两距离的数组
    cdef double[:] D = np.ndarray(n * (n - 1) / 2, dtype=np.float64)
    # 初始化一个索引映射表
    cdef int[:] id_map = np.ndarray(n, dtype=np.intc)
    # 定义一个函数指针，用于根据不同的方法选择距离更新策略
    cdef linkage_distance_update new_dist

    # 根据给定的距离数组初始化
    D[:] = dists
    # 初始化索引映射表
    for i in range(n):
        id_map[i] = i

    # 开始层次聚类算法的迭代过程
    for k in range(n - 1):
        # 寻找最接近的两个簇 x, y (其中 x < y)
        current_min = INFINITY
        for i in range(n - 1):
            if id_map[i] == -1:
                continue

            i_start = condensed_index(n, i, i + 1)
            for j in range(n - i - 1):
                if D[i_start + j] < current_min:
                    current_min = D[i_start + j]
                    x = i
                    y = i + j + 1

        # 获取簇 x 和 y 的原始点数
        id_x = id_map[x]
        id_y = id_map[y]
        nx = 1 if id_x < n else <int>Z[id_x - n, 3]
        ny = 1 if id_y < n else <int>Z[id_y - n, 3]

        # 记录新节点
        Z[k, 0] = min(id_x, id_y)
        Z[k, 1] = max(id_y, id_x)
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        id_map[x] = -1  # 簇 x 将被移除
        id_map[y] = n + k  # 簇 y 将被新的聚类替代

        # 更新距离矩阵
        for i in range(n):
            id_i = id_map[i]
            if id_i == -1 or id_i == n + k:
                continue

            ni = 1 if id_i < n else <int>Z[id_i - n, 3]
            D[condensed_index(n, i, y)] = new_dist(
                D[condensed_index(n, i, x)],
                D[condensed_index(n, i, y)],
                current_min, nx, ny, ni)
            if i < x:
                D[condensed_index(n, i, x)] = INFINITY

    # 返回层次聚类的结果数组
    return Z_arr


# 找到与簇 x 最近的簇 y 的函数
cdef Pair find_min_dist(int n, double[:] D, int[:] size, int x):
    """
    Find the cluster y closest to cluster x.

    Parameters
    ----------
    n : int
        The number of clusters.
    D : double[:]
        Array storing pairwise distances between clusters.
    size : int[:]
        Array storing the size of each cluster.
    x : int
        Index of the current cluster.

    Returns
    -------
    Pair
        Pair object containing the index of the closest cluster y and its distance.
    """
    cdef double current_min = INFINITY
    cdef int y = -1
    cdef int i
    cdef double dist

    # 寻找最接近簇 x 的簇 y
    for i in range(x + 1, n):
        if size[i] == 0:
            continue

        dist = D[condensed_index(n, x, i)]
        if dist < current_min:
            current_min = dist
            y = i

    return Pair(y, current_min)


# 快速层次聚类算法的函数
def fast_linkage(const double[:] dists, int n, int method):
    """Perform hierarchy clustering.
    
    Parameters
    ----------
    dists : double[:]
        A condensed matrix storing pairwise distances of observations.
    n : int
        The number of observations.
    method : int
        The linkage method. 0: single 1: complete 2: average 3: centroid
        4: median 5: ward 6: weighted

    Returns
    -------
    ndarray
        Computed linkage matrix.
    """
    """
    It implements "Generic Clustering Algorithm" from [1]. The worst case
    time complexity is O(N^3), but the best case time complexity is O(N^2) and
    it usually works quite close to the best case.

    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    method : int
        The linkage method. 0: single 1: complete 2: average 3: centroid
        4: median 5: ward 6: weighted

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.

    References
    ----------
    .. [1] Daniel Mullner, "Modern hierarchical, agglomerative clustering
       algorithms", :arXiv:`1109.2378v1`.
    """
    # Allocate memory for the linkage matrix Z
    cdef double[:, :] Z = np.empty((n - 1, 4))

    # Make a copy of the distances matrix
    cdef double[:] D = dists.copy()  # Distances between clusters.

    # Initialize cluster sizes to 1 for each observation
    cdef int[:] size = np.ones(n, dtype=np.intc)  # Sizes of clusters.

    # Initialize cluster IDs
    cdef int[:] cluster_id = np.arange(n, dtype=np.intc)
    # ID of a cluster to put into linkage matrix.

    # Initialize arrays to store nearest neighbor candidates and distances
    cdef int[:] neighbor = np.empty(n - 1, dtype=np.intc)
    cdef double[:] min_dist = np.empty(n - 1)

    # Select the linkage method function based on the given method integer
    cdef linkage_distance_update new_dist = linkage_methods[method]

    # Initialize variables used in the algorithm
    cdef int i, k
    cdef int x = 0, y = 0, z
    cdef int nx, ny, nz
    cdef int id_x, id_y
    cdef double dist = 0
    cdef Pair pair

    # Iterate to find nearest neighbors and their distances
    for x in range(n - 1):
        pair = find_min_dist(n, D, size, x)
        neighbor[x] = pair.key
        min_dist[x] = pair.value

    # Create a heap data structure using the minimum distances found
    cdef Heap min_dist_heap = Heap(min_dist)
    for k in range(n - 1):
        # 这里使用一个固定大小的循环来处理最近邻聚类的计算，而不是使用无限循环"while True"，
        # 这在涉及浮点计算时更加可靠。我们的目标是在不超过 n - k 次（最后一次迭代只有一次）距离更新内找到两个最接近的聚类。
        for i in range(n - k):
            # 从最小堆中获取距离最小的聚类对
            pair = min_dist_heap.get_min()
            x, dist = pair.key, pair.value
            # 获取聚类 x 的最近邻
            y = neighbor[x]

            # 如果距离等于当前紧凑距离矩阵中的距离，则停止当前迭代
            if dist == D[condensed_index(n, x, y)]:
                break

            # 找到聚类 x 的最近邻和对应的距离
            pair = find_min_dist(n, D, size, x)
            y, dist = pair.key, pair.value
            # 更新聚类 x 的最近邻
            neighbor[x] = y
            min_dist[x] = dist
            min_dist_heap.change_value(x, dist)
        # 从最小堆中移除距离最小的聚类对
        min_dist_heap.remove_min()

        # 获取聚类 x 和 y 的标识 ID，以及它们的大小
        id_x = cluster_id[x]
        id_y = cluster_id[y]
        nx = size[x]
        ny = size[y]

        # 确保 id_x <= id_y，以便于更新层次聚类矩阵
        if id_x > id_y:
            id_x, id_y = id_y, id_x

        # 更新层次聚类矩阵 Z 的当前行
        Z[k, 0] = id_x
        Z[k, 1] = id_y
        Z[k, 2] = dist
        Z[k, 3] = nx + ny

        # 标记聚类 x 将被丢弃，聚类 y 将被新的聚类替换
        size[x] = 0
        size[y] = nx + ny
        cluster_id[y] = n + k  # 更新聚类 y 的标识 ID

        # 更新距离矩阵 D
        for z in range(n):
            nz = size[z]
            if nz == 0 or z == y:
                continue

            # 计算并更新距离矩阵 D 中新的距离下界
            D[condensed_index(n, z, y)] = new_dist(
                D[condensed_index(n, z, x)], D[condensed_index(n, z, y)],
                dist, nx, ny, nz)

        # 重新分配从聚类 x 到聚类 y 的最近邻候选项
        for z in range(x):
            if size[z] > 0 and neighbor[z] == x:
                neighbor[z] = y

        # 更新距离的下界
        for z in range(y):
            if size[z] == 0:
                continue

            dist = D[condensed_index(n, z, y)]
            if dist < min_dist[z]:
                neighbor[z] = y
                min_dist[z] = dist
                min_dist_heap.change_value(z, dist)

        # 查找聚类 y 的最近邻
        if y < n - 1:
            pair = find_min_dist(n, D, size, y)
            z, dist = pair.key, pair.value
            if z != -1:
                neighbor[y] = z
                min_dist[y] = dist
                min_dist_heap.change_value(y, dist)

    return Z.base
# 定义一个函数，执行最近邻链算法进行层次聚类

def nn_chain(const double[:] dists, int n, int method):
    """Perform hierarchy clustering using nearest-neighbor chain algorithm.

    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    method : int
        The linkage method. 0: single 1: complete 2: average 3: centroid
        4: median 5: ward 6: weighted

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """
    
    # 创建一个空的数组用于存储 linkage 矩阵
    Z_arr = np.empty((n - 1, 4))
    # 将数组转换为 Cython 的 double 类型二维数组
    cdef double[:, :] Z = Z_arr

    # 复制距离矩阵，用于存储集群之间的距离
    cdef double[:] D = dists.copy()
    # 初始化一个整数数组，表示每个集群的大小，初始都为1
    cdef int[:] size = np.ones(n, dtype=np.intc)

    # 获取指定的 linkage 方法对应的更新距离函数
    cdef linkage_distance_update new_dist = linkage_methods[method]

    # 变量用于存储邻居链
    cdef int[:] cluster_chain = np.ndarray(n, dtype=np.intc)
    cdef int chain_length = 0

    # 初始化循环中使用的变量
    cdef int i, k, x, y = 0, nx, ny, ni
    cdef double dist, current_min
    # Iterate through clusters to perform agglomerative clustering until one cluster remains.
    for k in range(n - 1):
        if chain_length == 0:
            # Start a new cluster chain with the first available element.
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # Extend the chain by finding the next neighbor with minimal distance.
        while True:
            x = cluster_chain[chain_length - 1]

            # Prefer the previous element in the chain as the minimum distance to avoid cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[condensed_index(n, x, y)]
            else:
                current_min = INFINITY

            # Find the nearest neighbor to x that is not yet in the chain.
            for i in range(n):
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                if dist < current_min:
                    current_min = dist
                    y = i

            # If we have closed the loop, stop extending the chain.
            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break

            # Add the new neighbor to the chain and increase chain length.
            cluster_chain[chain_length] = y
            chain_length += 1

        # Merge clusters x and y and reduce chain length accordingly.
        chain_length -= 2

        # Ensure x < y to maintain convention in fastcluster.
        if x > y:
            x, y = y, x

        # Retrieve sizes of clusters x and y before merging.
        nx = size[x]
        ny = size[y]

        # Record the new node in the linkage matrix Z.
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new merged cluster.

        # Update the distance matrix D to reflect the merging of clusters x and y.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue

            # Calculate the new distance between cluster i and the merged cluster (x, y).
            D[condensed_index(n, i, y)] = new_dist(
                D[condensed_index(n, i, x)],
                D[condensed_index(n, i, y)],
                current_min, nx, ny, ni)

    # Sort the linkage matrix Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]

    # Assign correct cluster labels inplace using a labeling function.
    label(Z_arr, n)

    # Return the final linkage matrix Z_arr.
    return Z_arr
# 使用 MST 算法进行单链接层次聚类。

def mst_single_linkage(const double[:] dists, int n):
    """Perform hierarchy clustering using MST algorithm for single linkage.

    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    """

    Z_arr = np.empty((n - 1, 4))
    cdef double[:, :] Z = Z_arr

    # Which nodes were already merged.
    cdef int[:] merged = np.zeros(n, dtype=np.intc)

    cdef double[:] D = np.empty(n)
    D[:] = INFINITY

    cdef int i, k, x, y = 0
    cdef double dist, current_min

    x = 0
    for k in range(n - 1):
        current_min = INFINITY
        merged[x] = 1
        for i in range(n):
            if merged[i] == 1:
                continue

            dist = dists[condensed_index(n, x, i)]
            if D[i] > dist:
                D[i] = dist

            if D[i] < current_min:
                y = i
                current_min = D[i]

        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        x = y

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]

    # Find correct cluster labels and compute cluster sizes inplace.
    label(Z_arr, n)

    return Z_arr


cdef class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""
    cdef int[:] parent
    cdef int[:] size
    cdef int next_label

    def __init__(self, int n):
        # Initialize parent array and sizes for union-find structure.
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    cdef int merge(self, int x, int y) noexcept:
        # Merge clusters x and y, returning the size of the merged cluster.
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        cdef int size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    cdef find(self, int x):
        # Find the root of the cluster containing x using path compression.
        cdef int p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


cdef label(double[:, :] Z, int n):
    """Correctly label clusters in unsorted dendrogram."""
    cdef LinkageUnionFind uf = LinkageUnionFind(n)
    cdef int i, x, y, x_root, y_root

    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)


def prelist(const double[:, :] Z, int[:] members, int n):
    """
    Perform a pre-order traversal on the linkage tree and get a list of ids
    of the leaves.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix.
    """
    members : ndarray
        用于存储结果的数组。注意，这个输入数组将被原地修改。

    n : int
        观测值的数量。

    """
    cdef int k, i_lc, i_rc, root, mem_idx
    cdef int[:] curr_node = np.ndarray(n, dtype=np.intc)

    # 计算 visited 数组的大小
    cdef int visited_size = (((n * 2) - 1) >> 3) + 1
    # 分配内存并将其初始化为0，用于标记访问过的节点
    cdef uchar *visited = <uchar *>PyMem_Malloc(visited_size)
    if not visited:
        raise MemoryError
    memset(visited, 0, visited_size)

    mem_idx = 0
    k = 0
    # 初始化根节点为最后一个节点的索引
    curr_node[0] = 2 * n - 2
    while k >= 0:
        # 获取当前节点的索引并转换为实际的节点索引
        root = curr_node[k] - n

        # 获取左子节点的索引
        i_lc = <int>Z[root, 0]
        # 如果左子节点未访问过，则标记为访问过，并根据情况继续处理
        if not is_visited(visited, i_lc):
            set_visited(visited, i_lc)
            if i_lc >= n:
                # 如果左子节点是内部节点，则将其推入栈中
                k += 1
                curr_node[k] = i_lc
                continue
            else:
                # 如果左子节点是叶子节点，则将其加入结果数组中
                members[mem_idx] = i_lc
                mem_idx += 1

        # 获取右子节点的索引
        i_rc = <int>Z[root, 1]
        # 如果右子节点未访问过，则标记为访问过，并根据情况继续处理
        if not is_visited(visited, i_rc):
            set_visited(visited, i_rc)
            if i_rc >= n:
                # 如果右子节点是内部节点，则将其推入栈中
                k += 1
                curr_node[k] = i_rc
                continue
            else:
                # 如果右子节点是叶子节点，则将其加入结果数组中
                members[mem_idx] = i_rc
                mem_idx += 1

        # 处理完当前节点后，退栈
        k -= 1

    # 释放分配的 visited 数组的内存
    PyMem_Free(visited)
```