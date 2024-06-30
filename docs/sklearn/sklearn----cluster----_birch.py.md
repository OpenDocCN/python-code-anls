# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_birch.py`

```
# 导入警告模块，用于处理警告信息
import warnings
# 导入平方根函数
from math import sqrt
# 导入整数和实数类型
from numbers import Integral, Real

# 导入 NumPy 库，并重命名为 np
import numpy as np
# 导入 SciPy 中的稀疏矩阵处理模块
from scipy import sparse

# 导入配置上下文
from .._config import config_context
# 导入基本估计器类及特征输出前缀类、聚类混合类、转换器类和拟合上下文
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
# 导入收敛警告异常类
from ..exceptions import ConvergenceWarning
# 导入计算最近距离的函数
from ..metrics import pairwise_distances_argmin
# 导入欧氏距离计算函数
from ..metrics.pairwise import euclidean_distances
# 导入参数验证相关的类和函数
from ..utils._param_validation import Hidden, Interval, StrOptions
# 导入行范数计算函数
from ..utils.extmath import row_norms
# 导入检查是否已拟合函数
from ..utils.validation import check_is_fitted
# 导入凝聚聚类模块
from . import AgglomerativeClustering


def _iterate_sparse_X(X):
    """This little hack returns a densified row when iterating over a sparse
    matrix, instead of constructing a sparse matrix for every row that is
    expensive.
    """
    # 获取稀疏矩阵的样本数
    n_samples = X.shape[0]
    # 获取稀疏矩阵的非零元素的列索引
    X_indices = X.indices
    # 获取稀疏矩阵的非零元素的值
    X_data = X.data
    # 获取稀疏矩阵每行非零元素在数据数组中的起始位置和结束位置
    X_indptr = X.indptr

    # 遍历稀疏矩阵的每一行
    for i in range(n_samples):
        # 创建一个全零数组，用于存放当前行的密集表示
        row = np.zeros(X.shape[1])
        # 获取当前行非零元素的列索引
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        # 将当前行非零元素的值放入密集数组对应的位置
        row[nonzero_indices] = X_data[startptr:endptr]
        # 返回当前行的密集表示
        yield row


def _split_node(node, threshold, branching_factor):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    # 初始化两个空子节点和两个空子聚类
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    # 根据指定阈值和分支因子创建两个新节点
    new_node1 = _CFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_node2 = _CFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    # 将两个新节点设置为两个新子聚类的子节点
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    # 如果当前节点是叶子节点
    if node.is_leaf:
        # 更新叶子节点的前驱和后继节点
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2

    # 计算当前节点中心点间的欧氏距离矩阵
    dist = euclidean_distances(
        node.centroids_, Y_norm_squared=node.squared_norm_, squared=True
    )
    n_clusters = dist.shape[0]

    # 找到距离最远的两个节点的索引
    farthest_idx = np.unravel_index(dist.argmax(), (n_clusters, n_clusters))
    node1_dist, node2_dist = dist[(farthest_idx,)]
    # 判断哪个节点更靠近它自身，即使所有节点之间的距离都相等时也要保证 node1 是最近的。
    # 这种情况只会在所有 node.centroids_ 都是重复的情况下发生，导致所有中心点之间的距离都为零。
    node1_closer = node1_dist < node2_dist
    
    # 确保 node1 在距离相等的情况下也被标记为最近的节点。
    # 这种情况可能会出现在所有 node.centroids_ 都是重复的情况下，导致所有中心点之间的距离都为零。
    node1_closer[farthest_idx[0]] = True

    # 遍历节点 node 的子聚类列表，根据 node1_closer 的标记将子聚类分配到新的两个节点中。
    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            # 如果 subcluster 更接近 node1，则将其添加到 new_node1 中，并更新 new_subcluster1。
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            # 如果 subcluster 更接近 node2，则将其添加到 new_node2 中，并更新 new_subcluster2。
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    
    # 返回更新后的两个子聚类 new_subcluster1 和 new_subcluster2。
    return new_subcluster1, new_subcluster2
class _CFNode:
    """Each node in a CFTree is called a CFNode.

    The CFNode can have a maximum of branching_factor
    number of CFSubclusters.

    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a CFSubcluster.

    branching_factor : int
        Maximum number of CF subclusters in each node.

    is_leaf : bool
        We need to know if the CFNode is a leaf or not, in order to
        retrieve the final subclusters.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : list
        List of subclusters for a particular CFNode.

    prev_leaf_ : _CFNode
        Useful only if is_leaf is True.

    next_leaf_ : _CFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.

    init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
        Manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    init_sq_norm_ : ndarray of shape (branching_factor + 1,)
        manipulate init_sq_norm_ throughout. similar to ``init_centroids_``.

    centroids_ : ndarray of shape (branching_factor + 1, n_features)
        View of ``init_centroids_``.

    squared_norm_ : ndarray of shape (branching_factor + 1,)
        View of ``init_sq_norm_``.

    """

    def __init__(self, *, threshold, branching_factor, is_leaf, n_features, dtype):
        # 设置阈值以便新的子簇进入 CFSubcluster
        self.threshold = threshold
        # 每个节点最多允许的 CF 子簇数量
        self.branching_factor = branching_factor
        # 是否为叶子节点，用于检索最终的子簇
        self.is_leaf = is_leaf
        # 特征的数量
        self.n_features = n_features

        # 列表，用于存储特定 CFNode 的子簇
        self.subclusters_ = []
        # 初始化用于存储质心和平方范数的数组
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features), dtype=dtype)
        self.init_sq_norm_ = np.zeros((branching_factor + 1), dtype)
        # 将 centroids_ 和 squared_norm_ 视作视图
        self.centroids_ = None  # 未初始化的占位符
        self.squared_norm_ = None  # 未初始化的占位符
        # 用于叶子节点时，存储前一个和下一个叶子节点
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        # 获取当前子簇数量
        n_samples = len(self.subclusters_)
        # 将新的子簇添加到列表中
        self.subclusters_.append(subcluster)
        # 将子簇的质心和平方范数添加到相应的数组中
        self.init_centroids_[n_samples] = subcluster.centroid_
        self.init_sq_norm_[n_samples] = subcluster.sq_norm_

        # 更新 centroids_ 和 squared_norm_ 的视图
        self.centroids_ = self.init_centroids_[: n_samples + 1, :]
        self.squared_norm_ = self.init_sq_norm_[: n_samples + 1]
    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        # 找到要移除的子聚类在列表中的索引位置
        ind = self.subclusters_.index(subcluster)
        # 用新的第一个子聚类替换原来的子聚类
        self.subclusters_[ind] = new_subcluster1
        # 更新初始质心向量到新的第一个子聚类的质心向量
        self.init_centroids_[ind] = new_subcluster1.centroid_
        # 更新初始平方范数到新的第一个子聚类的平方范数
        self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
        # 将新的第二个子聚类添加到列表末尾
        self.append_subcluster(new_subcluster2)
    def insert_cf_subcluster(self, subcluster):
        """Insert a new subcluster into the node."""
        # 如果没有子集群，则直接添加新的子集群并返回 False
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        threshold = self.threshold
        branching_factor = self.branching_factor
        
        # 计算当前节点的所有子集群与新子集群的距离矩阵
        dist_matrix = np.dot(self.centroids_, subcluster.centroid_)
        dist_matrix *= -2.0
        dist_matrix += self.squared_norm_
        
        # 找出距离最近的子集群的索引
        closest_index = np.argmin(dist_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # 如果最近的子集群有子节点，则采用递归策略插入新的子集群
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_cf_subcluster(subcluster)

            if not split_child:
                # 如果确定子节点不需要分裂，则更新最近的子集群
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[
                    closest_index
                ].centroid_
                self.init_sq_norm_[closest_index] = self.subclusters_[
                    closest_index
                ].sq_norm_
                return False

            # 需要重新分配子集群并在父子集群中添加新的子集群
            else:
                new_subcluster1, new_subcluster2 = _split_node(
                    closest_subcluster.child_,
                    threshold,
                    branching_factor,
                )
                self.update_split_subclusters(
                    closest_subcluster, new_subcluster1, new_subcluster2
                )

                # 如果子集群数量超过了分支因子，则返回 True 表示需要分裂当前节点
                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # 如果没有子节点，则直接合并或添加新的子集群
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            if merged:
                self.init_centroids_[closest_index] = closest_subcluster.centroid_
                self.init_sq_norm_[closest_index] = closest_subcluster.sq_norm_
                return False

            # 如果没有接近其他子集群，并且还有空间，则直接添加新的子集群
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False

            # 如果没有足够的空间，并且没有接近其他子集群，则需要分裂当前节点
            else:
                self.append_subcluster(subcluster)
                return True
class _CFSubcluster:
    """Each subcluster in a CFNode is called a CFSubcluster.

    A CFSubcluster can have a CFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    squared_sum_ : float
        Sum of the squared l2 norms of all samples belonging to a subcluster.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``CFNode.centroids_`` is called.

    child_ : _CFNode
        Child Node of the subcluster. Once a given _CFNode is set as the child
        of the _CFNode, it is set to ``self.child_``.

    sq_norm_ : ndarray of shape (branching_factor + 1,)
        Squared norm of the subcluster. Used to prevent recomputing when
        pairwise minimum distances are computed.
    """

    def __init__(self, *, linear_sum=None):
        # 如果 linear_sum 为 None，表示初始化一个空的子聚类
        if linear_sum is None:
            # 初始化空子聚类的属性
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.centroid_ = self.linear_sum_ = 0  # 线性和和质心初始化为0
        else:
            # 使用给定的 linear_sum 初始化子聚类
            self.n_samples_ = 1  # 子聚类中有一个样本
            self.centroid_ = self.linear_sum_ = linear_sum  # 初始化质心和线性和
            self.squared_sum_ = self.sq_norm_ = np.dot(
                self.linear_sum_, self.linear_sum_
            )  # 计算并初始化平方和及平方范数
        self.child_ = None  # 初始化子节点为 None

    def update(self, subcluster):
        # 更新子聚类的属性，将另一个子聚类的信息合并到当前子聚类中
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_  # 更新质心
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)  # 更新平方范数
    def merge_subcluster(self, nominee_cluster, threshold):
        """
        Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        # 计算合并后的子簇的新平方和、新线性和、新样本数
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        # 计算新的质心
        new_centroid = (1 / new_n) * new_ls
        # 计算新的质心的平方范数
        new_sq_norm = np.dot(new_centroid, new_centroid)

        # 定义子簇的平方半径：
        #   r^2  = sum_i ||x_i - c||^2 / n
        # 其中 x_i 是分配给子簇的 n 个点，c 是其质心：
        #   c = sum_i x_i / n
        # 可以展开为：
        #   r^2 = sum_i ||x_i||^2 / n - 2 < sum_i x_i / n, c> + n ||c||^2 / n
        # 简化后得到：
        #   r^2 = sum_i ||x_i||^2 / n - ||c||^2
        sq_radius = new_ss / new_n - new_sq_norm

        # 判断子簇的平方半径是否小于等于给定阈值的平方
        if sq_radius <= threshold**2:
            # 如果满足条件，则更新当前子簇的样本数、线性和、平方和、质心和质心的平方范数
            (
                self.n_samples_,
                self.linear_sum_,
                self.squared_sum_,
                self.centroid_,
                self.sq_norm_,
            ) = (new_n, new_ls, new_ss, new_centroid, new_sq_norm)
            return True
        # 如果不满足条件，则不进行合并操作
        return False

    @property
    def radius(self):
        """Return radius of the subcluster"""
        # 由于数值问题，这里的计算可能会得到负数，因此取其与0的最大值再开平方
        sq_radius = self.squared_sum_ / self.n_samples_ - self.sq_norm_
        return sqrt(max(0, sq_radius))
class Birch(
    ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin, BaseEstimator
):
    """Implements the BIRCH clustering algorithm.

    It is a memory-efficient, online-learning algorithm provided as an
    alternative to :class:`MiniBatchKMeans`. It constructs a tree
    data structure with the cluster centroids being read off the leaf.
    These can be either the final cluster centroids or can be provided as input
    to another clustering algorithm such as :class:`AgglomerativeClustering`.

    Read more in the :ref:`User Guide <birch>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    threshold : float, default=0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default=50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    n_clusters : int, instance of sklearn.cluster model or None, default=3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples.

        - `None` : the final clustering step is not performed and the
          subclusters are returned as they are.

        - :mod:`sklearn.cluster` Estimator : If a model is provided, the model
          is fit treating the subclusters as new samples and the initial data
          is mapped to the label of the closest subcluster.

        - `int` : the model fit is :class:`AgglomerativeClustering` with
          `n_clusters` set to be equal to the int.

    compute_labels : bool, default=True
        Whether or not to compute labels for each fit.

    copy : bool, default=True
        Whether or not to make a copy of the given data. If set to False,
        the initial data will be overwritten.

        .. deprecated:: 1.6
            `copy` was deprecated in 1.6 and will be removed in 1.8. It has no effect
            as the estimator does not perform in-place operations on the input data.

    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.

    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    subcluster_labels_ : ndarray
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.
"""
    # 存储输入数据的标签数组，形状为 (n_samples,)
    labels_ : ndarray of shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    # 记录在拟合过程中观察到的特征数量
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    # 记录在拟合过程中观察到的特征名称数组，只有当输入数据 X 中的特征名都是字符串时定义
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    # 另请参阅
    # --------
    # MiniBatchKMeans : 使用小批量更新中心位置的替代实现
    #     of the centers' positions using mini-batches.

    # 注意事项
    # -----
    # 树数据结构由节点组成，每个节点包含多个子簇。节点中最大子簇数由分支因子确定。
    # 每个子簇维护线性和、平方和及其样本数。此外，如果子簇不是叶节点的成员，
    # 则每个子簇还可以有一个节点作为其子节点。

    # 对于进入根节点的新点，它将与最接近的子簇合并，并更新该子簇的线性和、平方和
    # 及其样本数。这个过程递归地进行，直到更新叶节点的属性。

    # 参考文献
    # ----------
    # * Tian Zhang, Raghu Ramakrishnan, Maron Livny
    #   BIRCH: An efficient data clustering method for large databases.
    #   https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

    # * Roberto Perdisci
    #   JBirch - Java implementation of BIRCH clustering algorithm
    #   https://code.google.com/archive/p/jbirch

    # 示例
    # --------
    # >>> from sklearn.cluster import Birch
    # >>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    # >>> brc = Birch(n_clusters=None)
    # >>> brc.fit(X)
    # Birch(n_clusters=None)
    # >>> brc.predict(X)
    # array([0, 0, 0, 1, 1, 1])

    # 存储参数约束的字典，每个参数对应其值的验证条件
    _parameter_constraints: dict = {
        "threshold": [Interval(Real, 0.0, None, closed="neither")],
        "branching_factor": [Interval(Integral, 1, None, closed="neither")],
        "n_clusters": [None, ClusterMixin, Interval(Integral, 1, None, closed="left")],
        "compute_labels": ["boolean"],
        "copy": ["boolean", Hidden(StrOptions({"deprecated"}))],
    }

    # 初始化方法，接受一系列可选参数，用于设置 BIRCH 聚类算法的初始状态
    def __init__(
        self,
        *,
        threshold=0.5,
        branching_factor=50,
        n_clusters=3,
        compute_labels=True,
        copy="deprecated",
    ):
        self.threshold = threshold  # 设置阈值参数
        self.branching_factor = branching_factor  # 设置分支因子参数
        self.n_clusters = n_clusters  # 设置聚类数参数
        self.compute_labels = compute_labels  # 设置是否计算标签参数
        self.copy = copy  # 设置是否复制参数

    # 应用装饰器，用于拟合过程中的上下文管理
    @_fit_context(prefer_skip_nested_validation=True)
    # 定义一个方法 `fit`，用于构建 CF 树（Collaborative Filtering Tree）来处理输入数据
    def fit(self, X, y=None):
        """
        Build a CF Tree for the input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        # 调用对象内部的 `_fit` 方法来进行具体的拟合操作，`partial=False` 表示不使用部分拟合
        return self._fit(X, partial=False)
    def _fit(self, X, partial):
        # 检查是否已经存在根节点
        has_root = getattr(self, "root_", None)
        # 判断是否为首次调用_fit方法
        first_call = not (partial and has_root)

        # 检查是否需要警告关于`copy`参数已弃用的警告信息
        if self.copy != "deprecated" and first_call:
            warnings.warn(
                "`copy` was deprecated in 1.6 and will be removed in 1.8 since it "
                "has no effect internally. Simply leave this parameter to its default "
                "value to avoid this warning.",
                FutureWarning,
            )

        # 验证数据X的有效性，接受稀疏矩阵格式csr，如果是首次调用_fit方法则重置数据
        X = self._validate_data(
            X,
            accept_sparse="csr",
            reset=first_call,
            dtype=[np.float64, np.float32],
        )
        # 获取阈值和分支因子
        threshold = self.threshold
        branching_factor = self.branching_factor

        # 获取样本数和特征数
        n_samples, n_features = X.shape

        # 如果是首次调用_fit方法，初始化树结构
        if first_call:
            # 第一个根节点是叶子节点，用于整个操作过程中的对象
            self.root_ = _CFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=X.dtype,
            )

            # 用于获取子簇的虚拟叶子节点
            self.dummy_leaf_ = _CFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=X.dtype,
            )
            # 将虚拟叶子节点的下一个节点设置为根节点，用于后续操作
            self.dummy_leaf_.next_leaf_ = self.root_
            # 将根节点的前一个节点设置为虚拟叶子节点
            self.root_.prev_leaf_ = self.dummy_leaf_

        # 如果X不是稀疏矩阵，则迭代函数为iter，否则为特定的稀疏矩阵迭代函数_iterate_sparse_X
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        # 对X中的每个样本进行迭代
        for sample in iter_func(X):
            # 创建_CFSubcluster对象，表示线性求和的子簇
            subcluster = _CFSubcluster(linear_sum=sample)
            # 将子簇插入根节点的聚类簇中
            split = self.root_.insert_cf_subcluster(subcluster)

            # 如果发生分裂
            if split:
                # 对根节点进行分裂，生成两个新的子簇
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                # 删除旧的根节点
                del self.root_
                # 创建新的根节点作为_CFNode对象
                self.root_ = _CFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=X.dtype,
                )
                # 将新生成的两个子簇添加到新根节点下
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        # 获取所有叶子节点的质心并连接成一个numpy数组
        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        # 将质心作为子簇中心点存储
        self.subcluster_centers_ = centroids
        # 设置输出特征数为子簇中心点的数量
        self._n_features_out = self.subcluster_centers_.shape[0]

        # 对全局数据进行聚类操作
        self._global_clustering(X)
        # 返回对象自身
        return self
    def _get_leaves(self):
        """
        Retrieve the leaves of the CF Node.

        Returns
        -------
        leaves : list of shape (n_leaves,)
            List of the leaf nodes.
        """
        # Start from the first dummy leaf and traverse the linked list of leaf nodes
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            # Add each leaf node to the list
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        # Return the list of leaf nodes
        return leaves

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X=None, y=None):
        """
        Online learning. Prevents rebuilding of CFTree from scratch.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), \
            default=None
            Input data. If X is not provided, only the global clustering
            step is done.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        if X is None:
            # If X is not provided, perform only the global clustering step
            self._global_clustering()
            return self
        else:
            # Otherwise, perform partial fitting with the provided data
            return self._fit(X, partial=True)

    def predict(self, X):
        """
        Predict data using the ``centroids_`` of subclusters.

        Avoid computation of the row norms of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        labels : ndarray of shape(n_samples,)
            Labelled data.
        """
        # Ensure the estimator is fitted
        check_is_fitted(self)
        # Validate input X and ensure it is in CSR format without resetting
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        # Perform prediction using subcluster centroids
        return self._predict(X)

    def _predict(self, X):
        """Predict data using the ``centroids_`` of subclusters."""
        # Prepare metric keyword arguments for pairwise distance computation
        kwargs = {"Y_norm_squared": self._subcluster_norms}

        # Compute pairwise distances to find the closest subcluster centers
        with config_context(assume_finite=True):
            argmin = pairwise_distances_argmin(
                X, self.subcluster_centers_, metric_kwargs=kwargs
            )
        # Return labels of the closest subclusters
        return self.subcluster_labels_[argmin]

    def transform(self, X):
        """
        Transform X into subcluster centroids dimension.

        Each dimension represents the distance from the sample point to each
        cluster centroid.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_trans : {array-like, sparse matrix} of shape (n_samples, n_clusters)
            Transformed data.
        """
        # Ensure the estimator is fitted
        check_is_fitted(self)
        # Validate input X and ensure it is in CSR format without resetting
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        # Compute Euclidean distances from X to subcluster centers
        with config_context(assume_finite=True):
            return euclidean_distances(X, self.subcluster_centers_)
    # 定义一个方法用于对子簇进行全局聚类
    def _global_clustering(self, X=None):
        """
        Global clustering for the subclusters obtained after fitting
        子簇拟合后的全局聚类
        """
        # 获取聚类器的数量
        clusterer = self.n_clusters
        # 获取子簇的质心
        centroids = self.subcluster_centers_
        # 是否需要计算标签（如果输入数据不为None且需要计算标签）
        compute_labels = (X is not None) and self.compute_labels

        # 全局聚类的预处理
        not_enough_centroids = False
        # 如果clusterer是整数类型
        if isinstance(clusterer, Integral):
            # 使用层次聚类作为聚类器，并指定聚类数为self.n_clusters
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
            # 如果质心数量小于self.n_clusters，则不需要执行全局聚类步骤
            if len(centroids) < self.n_clusters:
                not_enough_centroids = True

        # 用于在预测中避免重新计算的子簇归一化后的范数
        self._subcluster_norms = row_norms(self.subcluster_centers_, squared=True)

        # 如果clusterer为None或者质心数量不足，则直接将子簇标签设置为索引数组
        if clusterer is None or not_enough_centroids:
            self.subcluster_labels_ = np.arange(len(centroids))
            # 如果质心数量不足，发出警告
            if not_enough_centroids:
                warnings.warn(
                    "Number of subclusters found (%d) by BIRCH is less "
                    "than (%d). Decrease the threshold."
                    % (len(centroids), self.n_clusters),
                    ConvergenceWarning,
                )
        else:
            # 执行全局聚类步骤，对叶子节点的子簇进行聚类
            # 假设子簇的质心作为样本，找到最终的质心
            self.subcluster_labels_ = clusterer.fit_predict(self.subcluster_centers_)

        # 如果需要计算标签，则调用_predict方法进行预测
        if compute_labels:
            self.labels_ = self._predict(X)

    # 返回更多标签的方法
    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}
```