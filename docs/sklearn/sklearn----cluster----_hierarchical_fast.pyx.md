# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hierarchical_fast.pyx`

```
# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>

# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库
cimport cython  # 导入 Cython 扩展类型声明

# 导入特定的 Cython 类型和函数
from ..metrics._dist_metrics cimport DistanceMetric64
from ..utils._fast_dict cimport IntFloatDict
from ..utils._typedefs cimport float64_t, intp_t, uint8_t

# 导入 C++ 标准库中的特定功能
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.map cimport map as cpp_map
from libc.math cimport fmax, INFINITY


###############################################################################
# 计算 Ward 距离的实用函数

def compute_ward_dist(
    const float64_t[::1] m_1,
    const float64_t[:, ::1] m_2,
    const intp_t[::1] coord_row,
    const intp_t[::1] coord_col,
    float64_t[::1] res
):
    # 初始化变量
    cdef intp_t size_max = coord_row.shape[0]  # 计算 coord_row 的大小
    cdef intp_t n_features = m_2.shape[1]  # 获取 m_2 的特征数
    cdef intp_t i, j, row, col
    cdef float64_t pa, n

    # 循环遍历计算 Ward 距离
    for i in range(size_max):
        row = coord_row[i]
        col = coord_col[i]
        n = (m_1[row] * m_1[col]) / (m_1[row] + m_1[col])  # 计算 n
        pa = 0.
        # 计算对应的 pa 值
        for j in range(n_features):
            pa += (m_2[row, j] / m_1[row] - m_2[col, j] / m_1[col]) ** 2
        res[i] = pa * n  # 计算结果存入 res


###############################################################################
# 切割和探索层次树的实用函数

def _hc_get_descendent(intp_t node, children, intp_t n_leaves):
    """
    返回树中指定节点的所有叶子后代节点。

    参数
    ----------
    node : 整数
        想要获取后代节点的节点。

    children : 长度为 n_nodes 的节点对列表
        每个非叶子节点的子节点。小于 `n_samples` 的值表示树的叶子节点。
        大于 `i` 的值表示具有子节点 `children[i - n_samples]` 的节点。

    n_leaves : 整数
        叶子节点的数量。

    返回
    -------
    descendent : 整数列表
    """
    ind = [node]  # 初始化节点列表

    # 如果节点小于叶子节点数，则直接返回节点列表
    if node < n_leaves:
        return ind

    descendent = []  # 初始化后代节点列表

    # 手动计数元素列表比使用 len 更快
    cdef intp_t i, n_indices = 1

    # 迭代查找后代节点
    while n_indices:
        i = ind.pop()
        if i < n_leaves:
            descendent.append(i)
            n_indices -= 1
        else:
            ind.extend(children[i - n_leaves])
            n_indices += 1
    return descendent


def hc_get_heads(intp_t[:] parents, copy=True):
    """
    返回森林（树的集合）中的头节点，根据父节点定义。

    参数
    ----------
    parents : 整数数组
        定义森林（树集合）的父节点结构
    copy : 布尔值
        如果 copy 是 False，则在原地修改输入的 'parents' 数组

    返回
    -------
    heads : 整数数组，与 parents 的形状相同
        'parents' 中树头节点的索引

    """
    cdef intp_t parent, node0, node, size

    # 如果需要复制，则复制父节点数组
    if copy:
        parents = np.copy(parents)

    size = parents.size  # 获取父节点数组的大小
    # 从树的顶部开始向下遍历
    for node0 in range(size - 1, -1, -1):
        # 将当前节点初始化为 node0
        node = node0
        # 获取当前节点的父节点
        parent = parents[node]
        # 当父节点不等于当前节点时，进行循环，寻找最终的根节点
        while parent != node:
            # 将当前节点的父节点更新为根节点，路径压缩
            parents[node0] = parent
            # 将当前节点更新为其父节点
            node = parent
            # 获取当前节点的新的父节点
            parent = parents[node]
    # 返回具有路径压缩的父节点数组
    return parents
def _get_parents(
    nodes,
    heads,
    const intp_t[:] parents,
    uint8_t[::1] not_visited
):
    """Returns the heads of the given nodes, as defined by parents.

    Modifies 'heads' and 'not_visited' in-place.

    Parameters
    ----------
    nodes : list of integers
        The nodes to start from
    heads : list of integers
        A list to hold the results (modified inplace)
    parents : array of integers
        The parent structure defining the tree
    not_visited : array of unsigned bytes
        The tree nodes to consider (modified inplace)

    """
    cdef intp_t parent, node

    for node in nodes:
        # Traverse upwards to find the root parent of the current node
        parent = parents[node]
        while parent != node:
            node = parent
            parent = parents[node]
        
        # Mark the root parent as visited if it hasn't been visited before
        if not_visited[node]:
            not_visited[node] = 0  # Mark node as visited
            heads.append(node)  # Append the root parent to the 'heads' list


###############################################################################
# merge strategies implemented on IntFloatDicts

# These are used in the hierarchical clustering code, to implement
# merging between two clusters, defined as a dict containing node number
# as keys and edge weights as values.


def max_merge(
    IntFloatDict a,
    IntFloatDict b,
    const intp_t[:] mask,
    intp_t n_a,
    intp_t n_b
):
    """Merge two IntFloatDicts with the max strategy: when the same key is
    present in the two dicts, the max of the two values is used.

    Parameters
    ==========
    a, b : IntFloatDict object
        The IntFloatDicts to merge
    mask : ndarray array of dtype integer and of dimension 1
        a mask for keys to ignore: if not mask[key] the corresponding key
        is skipped in the output dictionary
    n_a, n_b : float
        n_a and n_b are weights for a and b for the merge strategy.
        They are not used in the case of a max merge.

    Returns
    =======
    out : IntFloatDict object
        The IntFloatDict resulting from the merge
    """
    cdef IntFloatDict out_obj = IntFloatDict.__new__(IntFloatDict)
    cdef cpp_map[intp_t, float64_t].iterator a_it = a.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator a_end = a.my_map.end()
    cdef intp_t key
    cdef float64_t value
    
    # First copy values from dictionary 'a' into 'out_obj'
    while a_it != a_end:
        key = deref(a_it).first
        if mask[key]:
            out_obj.my_map[key] = deref(a_it).second
        inc(a_it)

    # Then merge values from dictionary 'b' into 'out_obj'
    cdef cpp_map[intp_t, float64_t].iterator out_it = out_obj.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator out_end = out_obj.my_map.end()
    cdef cpp_map[intp_t, float64_t].iterator b_it = b.my_map.begin()
    cdef cpp_map[intp_t, float64_t].iterator b_end = b.my_map.end()
    # 当 b_it 没有达到 b_end 时，继续循环
    while b_it != b_end:
        # 获取 b_it 指向的元素的第一个和第二个值作为 key 和 value
        key = deref(b_it).first
        value = deref(b_it).second
        # 如果 mask[key] 为真
        if mask[key]:
            # 在 out_obj.my_map 中查找 key
            out_it = out_obj.my_map.find(key)
            # 如果找不到 key
            if out_it == out_end:
                # 将 key-value 添加到 out_obj.my_map 中
                out_obj.my_map[key] = value
            else:
                # 如果找到 key，则更新其对应的 value，取原值和新值的最大值
                deref(out_it).second = fmax(deref(out_it).second, value)
        # 移动 b_it 到下一个元素
        inc(b_it)
    # 返回更新后的 out_obj
    return out_obj
def average_merge(
    IntFloatDict a,
    IntFloatDict b,
    const intp_t[:] mask,
    intp_t n_a,
    intp_t n_b
):
    """Merge two IntFloatDicts with the average strategy: when the
    same key is present in the two dicts, the weighted average of the two
    values is used.

    Parameters
    ==========
    a, b : IntFloatDict object
        The IntFloatDicts to merge
    mask : ndarray array of dtype integer and of dimension 1
        a mask for keys to ignore: if not mask[key] the corresponding key
        is skipped in the output dictionary
    n_a, n_b : float
        n_a and n_b are weights for a and b for the merge strategy.
        They are used for a weighted mean.

    Returns
    =======
    out : IntFloatDict object
        The IntFloatDict resulting from the merge
    """
    # Create a new instance of IntFloatDict for the merged output
    cdef IntFloatDict out_obj = IntFloatDict.__new__(IntFloatDict)
    
    # Iterator over the elements of dictionary 'a'
    cdef cpp_map[intp_t, float64_t].iterator a_it = a.my_map.begin()
    # End iterator for dictionary 'a'
    cdef cpp_map[intp_t, float64_t].iterator a_end = a.my_map.end()
    # Declare variables for key and value
    cdef intp_t key
    cdef float64_t value
    # Compute the total weight for the weighted mean
    cdef float64_t n_out = <float64_t> (n_a + n_b)
    
    # Copy elements from 'a' to 'out_obj', applying the mask
    while a_it != a_end:
        key = deref(a_it).first
        if mask[key]:
            out_obj.my_map[key] = deref(a_it).second
        inc(a_it)

    # Iterator over the elements of dictionary 'b'
    cdef cpp_map[intp_t, float64_t].iterator b_it = b.my_map.begin()
    # End iterator for dictionary 'b'
    cdef cpp_map[intp_t, float64_t].iterator b_end = b.my_map.end()
    
    # Merge elements from 'b' into 'out_obj', applying the mask and computing weighted averages
    while b_it != b_end:
        key = deref(b_it).first
        value = deref(b_it).second
        if mask[key]:
            # Attempt to find the key in 'out_obj'
            cdef cpp_map[intp_t, float64_t].iterator out_it = out_obj.my_map.find(key)
            # End iterator for 'out_obj'
            cdef cpp_map[intp_t, float64_t].iterator out_end = out_obj.my_map.end()
            
            if out_it == out_end:
                # Key not found in 'out_obj', add it with the value from 'b'
                out_obj.my_map[key] = value
            else:
                # Key found in 'out_obj', update its value using weighted average
                deref(out_it).second = (n_a * deref(out_it).second
                                        + n_b * value) / n_out
        inc(b_it)
    
    # Return the merged dictionary 'out_obj'
    return out_obj


###############################################################################
# An edge object for fast comparisons

cdef class WeightedEdge:
    cdef public intp_t a
    cdef public intp_t b
    cdef public float64_t weight

    def __init__(self, float64_t weight, intp_t a, intp_t b):
        self.weight = weight
        self.a = a
        self.b = b
    # 定义一个特殊的比较方法，用于 Cython 编译器

    def __richcmp__(self, WeightedEdge other, int op):
        """Cython-specific comparison method.

        op is the comparison code::
            <   0
            ==  2
            >   4
            <=  1
            !=  3
            >=  5

        根据给定的比较操作码（op），比较当前对象和另一个 WeightedEdge 对象的权重值。
        """
        if op == 0:
            return self.weight < other.weight
        elif op == 1:
            return self.weight <= other.weight
        elif op == 2:
            return self.weight == other.weight
        elif op == 3:
            return self.weight != other.weight
        elif op == 4:
            return self.weight > other.weight
        elif op == 5:
            return self.weight >= other.weight

    # 返回对象的字符串表示，包括类名、权重值、a 和 b 属性的格式化输出
    def __repr__(self):
        return "%s(weight=%f, a=%i, b=%i)" % (self.__class__.__name__,
                                              self.weight,
                                              self.a, self.b)
################################################################################
# Efficient labelling/conversion of MSTs to single linkage hierarchies

cdef class UnionFind(object):
    """
    Union-find data structure for efficient clustering operations.

    Attributes:
    - parent: Array to store parent nodes for each element.
    - next_label: Next available label for merging clusters.
    - size: Array to store size of each cluster.
    """

    def __init__(self, N):
        # Initialize parent array with -1 indicating each element is its own parent
        self.parent = np.full(2 * N - 1, -1., dtype=np.intp, order='C')
        # Initialize next_label to N, indicating the next available label for merging
        self.next_label = N
        # Initialize size array with 1s for N elements and 0s for N-1 virtual nodes
        self.size = np.hstack((np.ones(N, dtype=np.intp),
                               np.zeros(N - 1, dtype=np.intp)))

    cdef void union(self, intp_t m, intp_t n) noexcept:
        """
        Union operation to merge clusters represented by elements m and n.

        Parameters:
        - m: Element to merge.
        - n: Element to merge.

        Returns: None
        """
        # Set parent of m and n to the next available label
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        # Update size of the new cluster formed by merging m and n
        self.size[self.next_label] = self.size[m] + self.size[n]
        # Increment next_label for the next merge operation
        self.next_label += 1
        return

    @cython.wraparound(True)
    cdef intp_t fast_find(self, intp_t n) noexcept:
        """
        Find operation to retrieve the root of the cluster containing element n.

        Parameters:
        - n: Element to find.

        Returns:
        - Root of the cluster containing element n.
        """
        cdef intp_t p
        p = n
        # Traverse up to find the root of the cluster
        while self.parent[n] != -1:
            n = self.parent[n]
        # Path compression to speed up future operations
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n


def _single_linkage_label(const float64_t[:, :] L):
    """
    Convert an linkage array or MST to a tree by labelling clusters at merges.
    This is done by using a Union find structure to keep track of merges
    efficiently. This is the private version of the function that assumes that
    ``L`` has been properly validated. See ``single_linkage_label`` for the
    user facing version of this function.

    Parameters
    ----------
    L: array of shape (n_samples - 1, 3)
        The linkage array or MST where each row specifies two samples
        to be merged and a distance or weight at which the merge occurs. This
        array is assumed to be sorted by the distance/weight.

    Returns
    -------
    A tree in the format used by scipy.cluster.hierarchy.
    """

    cdef float64_t[:, ::1] result_arr

    cdef intp_t left, left_cluster, right, right_cluster, index
    cdef float64_t delta

    # Initialize result_arr to store the resulting tree structure
    result_arr = np.zeros((L.shape[0], 4), dtype=np.float64)
    # Initialize UnionFind data structure with size equal to number of merges + 1
    U = UnionFind(L.shape[0] + 1)

    # Iterate through each merge in the linkage array
    for index in range(L.shape[0]):
        # Extract left and right elements to be merged and the delta (distance/weight)
        left = <intp_t> L[index, 0]
        right = <intp_t> L[index, 1]
        delta = L[index, 2]

        # Find the root clusters for left and right elements
        left_cluster = U.fast_find(left)
        right_cluster = U.fast_find(right)

        # Populate result_arr with left and right cluster labels, delta, and merged cluster size
        result_arr[index][0] = left_cluster
        result_arr[index][1] = right_cluster
        result_arr[index][2] = delta
        result_arr[index][3] = U.size[left_cluster] + U.size[right_cluster]

        # Union operation to merge left and right clusters
        U.union(left_cluster, right_cluster)

    # Convert result_arr to numpy array and return
    return np.asarray(result_arr)


@cython.wraparound(True)
def single_linkage_label(L):
    """
    Convert an linkage array or MST to a tree by labelling clusters at merges.
    This is done by using a Union find structure to keep track of merges
    efficiently.

    Parameters
    ----------
    L: array of shape (n_samples - 1, 3)
        The linkage array or MST where each row specifies two samples
        to be merged and a distance or weight at which the merge occurs. This
        array is assumed to be sorted by the distance/weight.

    Returns
    -------
    A tree in the format used by scipy.cluster.hierarchy.
    """
    # 定义变量 L，表示一个形状为 (n_samples - 1, 3) 的数组
    # 这个数组是层次聚类的链接数组或最小生成树 (MST)，每一行指定了要合并的两个样本及其合并发生的距离或权重
    # 假设该数组已按距离/权重排序
    
    # 返回值
    # 返回一个以 scipy.cluster.hierarchy 使用的格式表示的树
    
    # 检验 L 的有效性
    if L[:, :2].min() < 0 or L[:, :2].max() >= 2 * L.shape[0] + 1:
        # 如果 L 的前两列的最小值小于 0 或者最大值大于等于 2 * L.shape[0] + 1，则抛出值错误异常
        raise ValueError("Input MST array is not a validly formatted MST array")
    
    # 定义一个 lambda 函数 is_sorted，用于检查数组是否按权重排序
    is_sorted = lambda x: np.all(x[:-1] <= x[1:])
    if not is_sorted(L[:, 2]):
        # 如果 L 的第三列（权重列）没有按升序排序，则抛出值错误异常
        raise ValueError("Input MST array must be sorted by weight")
    
    # 调用 _single_linkage_label 函数，并返回其结果
    return _single_linkage_label(L)
# Implements MST-LINKAGE-CORE from https://arxiv.org/abs/1109.2378
def mst_linkage_core(
        const float64_t [:, ::1] raw_data,
        DistanceMetric64 dist_metric):
    """
    Compute the necessary elements of a minimum spanning
    tree for computation of single linkage clustering. This
    represents the MST-LINKAGE-CORE algorithm (Figure 6) from
    :arxiv:`Daniel Mullner, "Modern hierarchical, agglomerative clustering
    algorithms" <1109.2378>`.

    In contrast to the scipy implementation is never computes
    a full distance matrix, generating distances only as they
    are needed and releasing them when no longer needed.

    Parameters
    ----------
    raw_data: array of shape (n_samples, n_features)
        The array of feature data to be clustered. Must be C-aligned

    dist_metric: DistanceMetric64
        A DistanceMetric64 object conforming to the API from
        ``sklearn.metrics._dist_metrics.pxd`` that will be
        used to compute distances.

    Returns
    -------
    mst_core_data: array of shape (n_samples, 3)
        An array providing information from which one
        can either compute an MST, or the linkage hierarchy
        very efficiently. See :arxiv:`Daniel Mullner, "Modern hierarchical,
        agglomerative clustering algorithms" <1109.2378>` algorithm
        MST-LINKAGE-CORE for more details.
    """
    cdef:
        # 获取样本数
        intp_t n_samples = raw_data.shape[0]
        # 用于标记节点是否在树中的布尔数组
        uint8_t[:] in_tree = np.zeros(n_samples, dtype=bool)
        # 存储最终结果的数组，每行包含两个节点和它们之间的距离
        float64_t[:, ::1] result = np.zeros((n_samples - 1, 3))

        # 当前节点的索引
        intp_t current_node = 0
        # 新节点的索引
        intp_t new_node
        # 循环变量 i 和 j
        intp_t i
        intp_t j
        # 特征数
        intp_t num_features = raw_data.shape[1]

        # 用于存储当前节点到其他节点的距离的数组，初始值为无穷大
        float64_t right_value
        float64_t left_value
        float64_t new_distance

        # 存储当前节点到所有其他节点的距离
        float64_t[:] current_distances = np.full(n_samples, INFINITY)

    for i in range(n_samples - 1):
        # 将当前节点标记为在树中
        in_tree[current_node] = 1

        # 初始化新距离和新节点索引
        new_distance = INFINITY
        new_node = 0

        for j in range(n_samples):
            if in_tree[j]:
                continue

            # 计算当前节点与 j 节点之间的距离
            right_value = current_distances[j]
            left_value = dist_metric.dist(&raw_data[current_node, 0],
                                          &raw_data[j, 0],
                                          num_features)

            # 如果左侧的距离小于右侧的距离，则更新当前节点到 j 节点的距离
            if left_value < right_value:
                current_distances[j] = left_value

            # 更新最小距离和新节点的索引
            if current_distances[j] < new_distance:
                new_distance = current_distances[j]
                new_node = j

        # 将当前节点、新节点及其距离存入结果数组
        result[i, 0] = current_node
        result[i, 1] = new_node
        result[i, 2] = new_distance
        # 更新当前节点为新节点
        current_node = new_node

    # 返回最终的结果数组
    return np.array(result)
```