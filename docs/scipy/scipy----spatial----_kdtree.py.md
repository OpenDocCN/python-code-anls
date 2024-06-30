# `D:\src\scipysrc\scipy\scipy\spatial\_kdtree.py`

```
# 导入 NumPy 库并将其命名为 np
import numpy as np
# 从 _ckdtree 模块中导入 cKDTree 和 cKDTreeNode 类
from ._ckdtree import cKDTree, cKDTreeNode

# 定义模块可导出的变量列表
__all__ = ['minkowski_distance_p', 'minkowski_distance',
           'distance_matrix',
           'Rectangle', 'KDTree']

# 定义计算 Minkowski 距离的函数，返回 L**p 距离的 p 次幂
def minkowski_distance_p(x, y, p=2):
    """Compute the pth power of the L**p distance between two arrays.

    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.

    The last dimensions of `x` and `y` must be the same length.  Any
    other dimensions must be compatible for broadcasting.

    Parameters
    ----------
    x : (..., K) array_like
        Input array.
    y : (..., K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Returns
    -------
    dist : ndarray
        pth power of the distance between the input arrays.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0, 0], [0, 0]], [[1, 1], [0, 1]])
    array([2, 1])

    """
    # 将输入转换为 NumPy 数组
    x = np.asarray(x)
    y = np.asarray(y)

    # 找到与 float64（函数返回类型）兼容的最小公共数据类型 - 解决 #10262。
    # 对于复数输入情况，不要只是转换为 float64。
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype),
                                       'float64')

    # 确保 x 和 y 是正确数据类型的 NumPy 数组
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    # 根据 p 的不同取值计算 Minkowski 距离的 p 次幂
    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)


# 定义计算 Minkowski 距离的函数，返回 L**p 距离
def minkowski_distance(x, y, p=2):
    """Compute the L**p distance between two arrays.

    The last dimensions of `x` and `y` must be the same length.  Any
    other dimensions must be compatible for broadcasting.

    Parameters
    ----------
    x : (..., K) array_like
        Input array.
    y : (..., K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Returns
    -------
    dist : ndarray
        Distance between the input arrays.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance
    >>> minkowski_distance([[0, 0], [0, 0]], [[1, 1], [0, 1]])
    array([ 1.41421356,  1.        ])

    """
    # 将输入转换为 NumPy 数组
    x = np.asarray(x)
    y = np.asarray(y)

    # 如果 p 是无穷大或者 p 是 1，直接调用 minkowski_distance_p 计算距离
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        # 否则计算 p 次幂的 Minkowski 距离
        return minkowski_distance_p(x, y, p)**(1./p)


class Rectangle:
    """Hyperrectangle class.

    Represents a Cartesian product of intervals.
    """
    def __init__(self, maxes, mins):
        """Construct a hyperrectangle."""
        # 初始化超矩形对象的最大值和最小值数组，并将它们转换为浮点数类型
        self.maxes = np.maximum(maxes, mins).astype(float)
        self.mins = np.minimum(maxes, mins).astype(float)
        # 计算超矩形的维度数
        self.m, = self.maxes.shape

    def __repr__(self):
        # 返回超矩形对象的字符串表示形式，包括最小值和最大值的列表
        return "<Rectangle %s>" % list(zip(self.mins, self.maxes))

    def volume(self):
        """Total volume."""
        # 计算超矩形的体积
        return np.prod(self.maxes - self.mins)

    def split(self, d, split):
        """Produce two hyperrectangles by splitting.

        In general, if you need to compute maximum and minimum
        distances to the children, it can be done more efficiently
        by updating the maximum and minimum distances to the parent.

        Parameters
        ----------
        d : int
            Axis to split hyperrectangle along.
        split : float
            Position along axis `d` to split at.

        """
        # 创建两个新的超矩形对象，通过在指定轴上进行分割
        mid = np.copy(self.maxes)
        mid[d] = split
        less = Rectangle(self.mins, mid)
        mid = np.copy(self.mins)
        mid[d] = split
        greater = Rectangle(mid, self.maxes)
        return less, greater

    def min_distance_point(self, x, p=2.):
        """
        Return the minimum distance between input and points in the
        hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input.
        p : float, optional
            Input.

        """
        # 计算输入点与超矩形内点的最小距离，采用Minkowski距离
        return minkowski_distance(
            0, np.maximum(0, np.maximum(self.mins - x, x - self.maxes)),
            p
        )

    def max_distance_point(self, x, p=2.):
        """
        Return the maximum distance between input and points in the hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input array.
        p : float, optional
            Input.

        """
        # 计算输入点与超矩形内点的最大距离，采用Minkowski距离
        return minkowski_distance(0, np.maximum(self.maxes - x, x - self.mins), p)

    def min_distance_rectangle(self, other, p=2.):
        """
        Compute the minimum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float
            Input.

        """
        # 计算两个超矩形之间的最小距离，采用Minkowski距离
        return minkowski_distance(
            0,
            np.maximum(0, np.maximum(self.mins - other.maxes,
                                     other.mins - self.maxes)),
            p
        )

    def max_distance_rectangle(self, other, p=2.):
        """
        Compute the maximum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float, optional
            Input.

        """
        # 计算两个超矩形之间的最大距离，采用Minkowski距离
        return minkowski_distance(
            0, np.maximum(self.maxes - other.mins, other.maxes - self.mins), p)
# 继承自 cKDTree 的 KD 树类，用于快速最近邻搜索。

"""kd-tree for quick nearest-neighbor lookup.

This class provides an index into a set of k-dimensional points
which can be used to rapidly look up the nearest neighbors of any
point.
"""

class KDTree(cKDTree):
    """
    Parameters
    ----------
    data : array_like, shape (n,m)
        要建立索引的 n 个 m 维数据点。除非需要生成一个连续的双精度数组，
        否则不会复制此数组，因此修改此数据将导致错误的结果。如果使用
        copy_data=True 来构建 KD 树，则数据也会被复制。
    leafsize : 正整数, 可选
        算法切换到蛮力的点数。默认：10。
    compact_nodes : 布尔值, 可选
        如果为 True，则构建的 KD 树将缩小超矩形到实际数据范围。
        这通常会生成一个更紧凑的树，能够处理退化的输入数据并提供更快的查询，
        但构建时间较长。默认：True。
    copy_data : 布尔值, 可选
        如果为 True，则始终复制数据以保护 KD 树免受数据损坏。默认：False。
    balanced_tree : 布尔值, 可选
        如果为 True，则使用中位数来分割超矩形，而不是中点。
        这通常会生成一个更紧凑的树，并提供更快的查询，但构建时间较长。
        默认：True。
    boxsize : array_like 或 标量, 可选
        对 KD 树应用 m 维环形拓扑。拓扑由 :math:`x_i + n_i L_i` 生成，
        其中 :math:`n_i` 是整数，:math:`L_i` 是第 i 维度的 boxsize。
        输入数据必须被包装到 :math:`[0, L_i)` 范围内。如果任何数据超出此范围，
        将引发 ValueError。

    Notes
    -----
    该算法描述于 Maneewongvatana 和 Mount 1999 年的论文中。
    总体思想是 KD 树是一个二叉树，每个节点表示一个轴对齐的超矩形。
    每个节点指定一个轴，并根据它们在该轴上的坐标是大于还是小于特定值来分割点集。

    在构建过程中，通过 "sliding midpoint" 规则选择轴和分割点，
    确保单元格不会全部变得长而细。

    可以查询树以获取任意给定点的 r 个最近邻居（可选仅返回某些最大距离内的邻居）。
    还可以查询树，通过显著提高效率，以获取 r 个近似最近邻居。

    对于大维度（20 已经很大），不要期望它比蛮力运行得显著更快。
    高维度的最近邻查询是计算机科学中一个重要的开放问题。

    Attributes
    ----------
    """
    data : ndarray, shape (n,m)
        数据点的数组，形状为 (n,m)，其中 n 表示数据点数目，m 表示每个数据点的维度。
        除非需要生成连续的双精度数组，否则不会复制此数组。如果使用 ``copy_data=True`` 构建了 kd 树，则也会复制数据。
    leafsize : positive int
        算法切换到蛮力方法的点数阈值。
    m : int
        单个数据点的维度。
    n : int
        数据点的数目。
    maxes : ndarray, shape (m,)
        数据点每个维度的最大值数组，形状为 (m,)。
    mins : ndarray, shape (m,)
        数据点每个维度的最小值数组，形状为 (m,)。
    size : int
        树中节点的数目。
    """

    class node:
        @staticmethod
        def _create(ckdtree_node=None):
            """创建内部节点或叶节点，包装 cKDTreeNode 实例"""
            if ckdtree_node is None:
                return KDTree.node(ckdtree_node)
            elif ckdtree_node.split_dim == -1:
                return KDTree.leafnode(ckdtree_node)
            else:
                return KDTree.innernode(ckdtree_node)

        def __init__(self, ckdtree_node=None):
            """初始化节点，如果未提供 cKDTreeNode 实例，则使用默认实例"""
            if ckdtree_node is None:
                ckdtree_node = cKDTreeNode()
            self._node = ckdtree_node

        def __lt__(self, other):
            """比较两个节点的 ID，用于排序"""
            return id(self) < id(other)

        def __gt__(self, other):
            """比较两个节点的 ID，用于排序"""
            return id(self) > id(other)

        def __le__(self, other):
            """比较两个节点的 ID，用于排序"""
            return id(self) <= id(other)

        def __ge__(self, other):
            """比较两个节点的 ID，用于排序"""
            return id(self) >= id(other)

        def __eq__(self, other):
            """比较两个节点的 ID，用于判断相等"""
            return id(self) == id(other)

    class leafnode(node):
        @property
        def idx(self):
            """返回叶节点的索引数组"""
            return self._node.indices

        @property
        def children(self):
            """返回叶节点的子节点"""
            return self._node.children

    class innernode(node):
        def __init__(self, ckdtreenode):
            """初始化内部节点，确保提供的是 cKDTreeNode 实例"""
            assert isinstance(ckdtreenode, cKDTreeNode)
            super().__init__(ckdtreenode)
            self.less = KDTree.node._create(ckdtreenode.lesser)
            self.greater = KDTree.node._create(ckdtreenode.greater)

        @property
        def split_dim(self):
            """返回内部节点的分割维度"""
            return self._node.split_dim

        @property
        def split(self):
            """返回内部节点的分割值"""
            return self._node.split

        @property
        def children(self):
            """返回内部节点的子节点"""
            return self._node.children

    @property
    def tree(self):
        """返回 KD 树的根节点"""
        if not hasattr(self, "_tree"):
            self._tree = KDTree.node._create(super().tree)

        return self._tree
    # 初始化 KD 树对象
    def __init__(self, data, leafsize=10, compact_nodes=True, copy_data=False,
                 balanced_tree=True, boxsize=None):
        # 将数据转换为 NumPy 数组
        data = np.asarray(data)
        # 检查数据类型，如果是复数则抛出类型错误
        if data.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")

        # 调用父类的初始化方法，传入各个参数
        # 注意：KDTree 的默认 leafsize 不同于 cKDTree
        super().__init__(data, leafsize, compact_nodes, copy_data,
                         balanced_tree, boxsize)

    # 在两棵 KD 树之间找出距离不超过 r 的点对
    def query_ball_tree(self, other, r, p=2., eps=0):
        """
        Find all pairs of points between `self` and `other` whose distance is
        at most r.

        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Returns
        -------
        results : list of lists
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.

        Examples
        --------
        You can search all pairs of points between two kd-trees within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((15, 2))
        >>> points2 = rng.random((15, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)
        >>> plt.plot(points2[:, 0], points2[:, 1], "og", markersize=14)
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> for i in range(len(indexes)):
        ...     for j in indexes[i]:
        ...         plt.plot([points1[i, 0], points2[j, 0]],
        ...                  [points1[i, 1], points2[j, 1]], "-r")
        >>> plt.show()

        """
        # 调用父类的查询方法，返回距离不超过 r 的点对的索引列表
        return super().query_ball_tree(other, r, p, eps)
    def query_pairs(self, r, p=2., eps=0, output_type='set'):
        """
        Find all pairs of points in `self` whose distance is at most r.

        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        output_type : string, optional
            Choose the output container, 'set' or 'ndarray'. Default: 'set'

            .. versionadded:: 1.6.0

        Returns
        -------
        results : set or ndarray
            Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
            positions are close. If output_type is 'ndarray', an ndarray is
            returned instead of a set.

        Examples
        --------
        You can search all pairs of points in a kd-tree within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points = rng.random((20, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)
        >>> kd_tree = KDTree(points)
        >>> pairs = kd_tree.query_pairs(r=0.2)
        >>> for (i, j) in pairs:
        ...     plt.plot([points[i, 0], points[j, 0]],
        ...             [points[i, 1], points[j, 1]], "-r")
        >>> plt.show()

        """
        # 调用父类的方法来查询所有距离在 r 以内的点对
        return super().query_pairs(r, p, eps, output_type)
    def sparse_distance_matrix(
            self, other, max_distance, p=2., output_type='dok_matrix'):
        """
        Compute a sparse distance matrix.

        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : KDTree
            另一个 KD 树对象，用于计算距离矩阵。

        max_distance : positive float
            最大距离阈值，超过此距离的距离将被设为零。

        p : float, 1<=p<=infinity
            Minkowski 范数中的 p 值，用于计算距离。如果 p 太大可能会导致溢出错误。

        output_type : string, optional
            输出数据的容器类型。可选值包括 'dok_matrix', 'coo_matrix', 'dict', 或 'ndarray'。
            默认为 'dok_matrix'。

            .. versionadded:: 1.6.0

        Returns
        -------
        result : dok_matrix, coo_matrix, dict or ndarray
            表示结果的稀疏矩阵，使用“字典键”格式。如果返回 dict，则键是 (i,j) 索引对。
            如果 output_type 是 'ndarray'，则返回一个字段包含 'i', 'j', 'v' 的记录数组。

        Examples
        --------
        你可以计算两个 KD 树之间的稀疏距离矩阵:

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.3)
        >>> sdm.toarray()
        array([[0.        , 0.        , 0.12295571, 0.        , 0.        ],
               [0.        , 0.        , 0.        , 0.        , 0.        ],
               [0.28942611, 0.        , 0.        , 0.2333084 , 0.        ],
               [0.        , 0.        , 0.        , 0.        , 0.        ],
               [0.24617575, 0.29571802, 0.26836782, 0.        , 0.        ]])

        你可以检查超过 `max_distance` 的距离是否为零:

        >>> from scipy.spatial import distance_matrix
        >>> distance_matrix(points1, points2)
        array([[0.56906522, 0.39923701, 0.12295571, 0.8658745 , 0.79428925],
               [0.37327919, 0.7225693 , 0.87665969, 0.32580855, 0.75679479],
               [0.28942611, 0.30088013, 0.6395831 , 0.2333084 , 0.33630734],
               [0.31994999, 0.72658602, 0.71124834, 0.55396483, 0.90785663],
               [0.24617575, 0.29571802, 0.26836782, 0.57714465, 0.6473269 ]])

        """
        return super().sparse_distance_matrix(
            other, max_distance, p, output_type)
def distance_matrix(x, y, p=2, threshold=1000000):
    """
    Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    y : (N, K) array_like
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    threshold : positive int
        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
        of large temporary arrays.

    Returns
    -------
    result : (M, N) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.

    Examples
    --------
    >>> from scipy.spatial import distance_matrix
    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])

    """

    x = np.asarray(x)  # Convert input x to NumPy array
    m, k = x.shape  # Get dimensions of x: m rows and k columns
    y = np.asarray(y)  # Convert input y to NumPy array
    n, kk = y.shape  # Get dimensions of y: n rows and kk columns

    if k != kk:
        raise ValueError(f"x contains {k}-dimensional vectors but y contains "
                         f"{kk}-dimensional vectors")

    if m*n*k <= threshold:
        # Use minkowski_distance function directly for small enough matrices
        return minkowski_distance(x[:, np.newaxis, :], y[np.newaxis, :, :], p)
    else:
        result = np.empty((m, n), dtype=float)  # Initialize result matrix
        if m < n:
            # Loop over rows of x and compute distances
            for i in range(m):
                result[i, :] = minkowski_distance(x[i], y, p)
        else:
            # Loop over columns of y and compute distances
            for j in range(n):
                result[:, j] = minkowski_distance(x, y[j], p)
        return result
```