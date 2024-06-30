# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_bisect_k_means.py`

```
"""Bisecting K-means clustering."""

# Author: Michal Krawczyk <mkrwczyk.1@gmail.com>

import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库
import scipy.sparse as sp  # 导入SciPy稀疏矩阵模块

from ..base import _fit_context  # 导入拟合上下文模块
from ..utils._openmp_helpers import _openmp_effective_n_threads  # 导入OpenMP帮助函数
from ..utils._param_validation import Integral, Interval, StrOptions  # 导入参数验证相关函数
from ..utils.extmath import row_norms  # 导入行范数计算函数
from ..utils.validation import _check_sample_weight, check_is_fitted, check_random_state  # 导入样本权重验证及其他验证函数
from ._k_means_common import _inertia_dense, _inertia_sparse  # 导入密集和稀疏数据的惯性计算函数
from ._kmeans import (
    _BaseKMeans,
    _kmeans_single_elkan,
    _kmeans_single_lloyd,
    _labels_inertia_threadpool_limit,
)  # 导入K均值聚类的基类及不同方法的实现

class _BisectingTree:
    """Tree structure representing the hierarchical clusters of BisectingKMeans."""

    def __init__(self, center, indices, score):
        """Create a new cluster node in the tree.

        The node holds the center of this cluster and the indices of the data points
        that belong to it.
        """
        self.center = center  # 当前簇的中心点
        self.indices = indices  # 当前簇包含的数据点索引
        self.score = score  # 当前簇的得分（用于决定下一步分裂）

        self.left = None  # 左子树
        self.right = None  # 右子树

    def split(self, labels, centers, scores):
        """Split the cluster node into two subclusters."""
        # 根据指定的标签（0和1）将当前节点分裂为两个子簇
        self.left = _BisectingTree(
            indices=self.indices[labels == 0], center=centers[0], score=scores[0]
        )
        self.right = _BisectingTree(
            indices=self.indices[labels == 1], center=centers[1], score=scores[1]
        )

        # 为了节省内存，重置当前节点的数据点索引为空
        self.indices = None

    def get_cluster_to_bisect(self):
        """Return the cluster node to bisect next.

        It's based on the score of the cluster, which can be either the number of
        data points assigned to that cluster or the inertia of that cluster
        (see `bisecting_strategy` for details).
        """
        max_score = None

        # 遍历树中所有的叶子节点，找到得分最高的簇节点
        for cluster_leaf in self.iter_leaves():
            if max_score is None or cluster_leaf.score > max_score:
                max_score = cluster_leaf.score
                best_cluster_leaf = cluster_leaf

        return best_cluster_leaf

    def iter_leaves(self):
        """Iterate over all the cluster leaves in the tree."""
        # 如果当前节点是叶子节点，则直接返回当前节点
        if self.left is None:
            yield self
        else:
            # 否则递归遍历左右子树的叶子节点
            yield from self.left.iter_leaves()
            yield from self.right.iter_leaves()


class BisectingKMeans(_BaseKMeans):
    """Bisecting K-Means clustering.

    Read more in the :ref:`User Guide <bisect_k_means>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    # 初始化方法，用于确定初始聚类中心点的方式。可以是字符串 'k-means++' 或 'random'，或者是一个可调用对象。
    # 'k-means++': 使用智能方式选择初始聚类中心点，加速收敛。
    # 'random': 从数据中随机选择 `n_clusters` 个观察值作为初始质心。
    # 如果传入一个可调用对象，它应该接受参数 X、n_clusters 和 random state，并返回一个初始化结果。
    init : {'k-means++', 'random'} or callable, default='random'

    # 内部 k-means 算法运行的次数，每次使用不同的质心种子进行双分过程。
    # 每次双分将产生 n_init 次连续运行的最佳输出，这些输出在惯性方面表现最佳。
    n_init : int, default=1

    # 内部 K-Means 算法中用于质心初始化的随机数生成器。使用整数值使随机性确定性化。
    # 可以是 int、RandomState 实例或 None。
    random_state : int, RandomState instance or None, default=None

    # 每次双分时内部 k-means 算法的最大迭代次数。
    max_iter : int, default=300

    # 内部 k-means 算法的迭代收敛性判据，以聚类中心的两次连续迭代之间的 Frobenius 范数相对差为标准。
    tol : float, default=1e-4

    # 当预先计算距离时，先居中数据通常更加数值稳定。如果 copy_x 是 True（默认值），则不修改原始数据。
    # 如果为 False，则修改原始数据，并在函数返回前恢复原始数据，但通过减去和添加数据均值可能引入小的数值差异。
    # 注意，如果原始数据不是 C 连续的，即使 copy_x 为 False，也会进行复制。如果原始数据是稀疏的但不是 CSR 格式，则即使 copy_x 为 False，也会进行复制。
    copy_x : bool, default=True

    # 双分过程中使用的内部 K-means 算法。经典的 EM 风格算法是 "lloyd"。
    # "elkan" 变体在某些具有明确定义聚类的数据集上可能更有效，因为利用三角不等式。但由于需要分配额外的数组（形状为 `(n_samples, n_clusters)`），它更占用内存。
    algorithm : {"lloyd", "elkan"}, default="lloyd"

    # 内部 K-means 算法在双分过程中使用的详细程度。
    verbose : int, default=0
    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,  # 继承基类 _BaseKMeans 的参数约束字典
        "init": [StrOptions({"k-means++", "random"}), callable],  # 初始化方法参数可选值和可调用对象
        "n_init": [Interval(Integral, 1, None, closed="left")],  # 聚类中心初始化的次数约束
        "copy_x": ["boolean"],  # 是否复制输入数据的布尔值约束
        "algorithm": [StrOptions({"lloyd", "elkan"})],  # K-Means 算法的选择，可选值为 "lloyd" 或 "elkan"
        "bisecting_strategy": [StrOptions({"biggest_inertia", "largest_cluster"})],  # 二分 K-Means 的分裂策略选择
    }
    def __init__(
        self,
        n_clusters=8,
        *,
        init="random",
        n_init=1,
        random_state=None,
        max_iter=300,
        verbose=0,
        tol=1e-4,
        copy_x=True,
        algorithm="lloyd",
        bisecting_strategy="biggest_inertia",
    ):
        # 调用父类的初始化方法，设置聚类器的参数
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
        )

        # 设置当前对象的属性
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.bisecting_strategy = bisecting_strategy

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        # 发出警告，说明在 Windows 下使用 MKL 时可能会出现内存泄漏问题
        warnings.warn(
            "BisectingKMeans is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}."
        )

    def _inertia_per_cluster(self, X, centers, labels, sample_weight):
        """Calculate the sum of squared errors (inertia) per cluster.

        Parameters
        ----------
        X : {ndarray, csr_matrix} of shape (n_samples, n_features)
            The input samples.

        centers : ndarray of shape (n_clusters=2, n_features)
            The cluster centers.

        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.

        Returns
        -------
        inertia_per_cluster : ndarray of shape (n_clusters=2,)
            Sum of squared errors (inertia) for each cluster.
        """
        n_clusters = centers.shape[0]  # = 2 since centers comes from a bisection
        _inertia = _inertia_sparse if sp.issparse(X) else _inertia_dense

        # 创建一个空的数组，用于存储每个簇的误差平方和（inertia）
        inertia_per_cluster = np.empty(n_clusters)
        # 对每个簇进行计算误差平方和
        for label in range(n_clusters):
            inertia_per_cluster[label] = _inertia(
                X, sample_weight, centers, labels, self._n_threads, single_label=label
            )

        return inertia_per_cluster
    # 将给定的集群分成两个子集群。

    # 参数说明：
    # X: 形状为 (n_samples, n_features) 的 ndarray 或 csr_matrix
    #    训练实例数据，用于聚类。
    # x_squared_norms: 形状为 (n_samples,) 的 ndarray
    #    每个数据点的平方欧氏范数。
    # sample_weight: 形状为 (n_samples,) 的 ndarray
    #    X 中每个观测的权重。
    # cluster_to_bisect: _BisectingTree 节点对象
    #    要拆分的集群节点。

    X = X[cluster_to_bisect.indices]
    x_squared_norms = x_squared_norms[cluster_to_bisect.indices]
    sample_weight = sample_weight[cluster_to_bisect.indices]

    best_inertia = None

    # 将 X 中的样本分成两个集群。
    # 重复 `n_init` 次以获得最佳集群
    for _ in range(self.n_init):
        centers_init = self._init_centroids(
            X,
            x_squared_norms=x_squared_norms,
            init=self.init,
            random_state=self._random_state,
            n_centroids=2,
            sample_weight=sample_weight,
        )

        # 进行单次 K-means 聚类
        labels, inertia, centers, _ = self._kmeans_single(
            X,
            sample_weight,
            centers_init,
            max_iter=self.max_iter,
            verbose=self.verbose,
            tol=self.tol,
            n_threads=self._n_threads,
        )

        # 允许对惯性有小的容差，以适应由于并行计算而产生的非确定性舍入误差
        if best_inertia is None or inertia < best_inertia * (1 - 1e-6):
            best_labels = labels
            best_centers = centers
            best_inertia = inertia

    if self.verbose:
        # 打印从双分叉得到的新质心
        print(f"New centroids from bisection: {best_centers}")

    if self.bisecting_strategy == "biggest_inertia":
        # 计算每个集群的惯性分数
        scores = self._inertia_per_cluster(
            X, best_centers, best_labels, sample_weight
        )
    else:  # bisecting_strategy == "largest_cluster"
        # 使用 minlength 确保即使所有样本标记为 0，也有两个标记的计数
        scores = np.bincount(best_labels, minlength=2)

    # 执行集群拆分操作，传递最佳标签、最佳质心和分数
    cluster_to_bisect.split(best_labels, best_centers, scores)
    def predict(self, X):
        """Predict which cluster each sample in X belongs to.

        Prediction is made by going down the hierarchical tree
        in searching of closest leaf cluster.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        # 确保模型已经拟合（即已经训练好）
        check_is_fitted(self)

        # 检查和处理输入数据，确保格式正确
        X = self._check_test_data(X)

        # 计算输入数据点的平方范数
        x_squared_norms = row_norms(X, squared=True)

        # 在Cython辅助函数中，样本权重是必需的，但在此处未使用
        sample_weight = np.ones_like(x_squared_norms)

        # 递归预测数据点所属的聚类标签
        labels = self._predict_recursive(X, sample_weight, self._bisecting_tree)

        return labels

    def _predict_recursive(self, X, sample_weight, cluster_node):
        """Predict recursively by going down the hierarchical tree.

        Parameters
        ----------
        X : {ndarray, csr_matrix} of shape (n_samples, n_features)
            The data points, currently assigned to `cluster_node`, to predict between
            the subclusters of this node.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.

        cluster_node : _BisectingTree node object
            The cluster node of the hierarchical tree.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        # 如果当前集群节点没有子集群，则直接返回该节点的标签作为所有数据点的标签
        if cluster_node.left is None:
            return np.full(X.shape[0], cluster_node.label, dtype=np.int32)

        # 确定数据点属于左子集群还是右子集群
        centers = np.vstack((cluster_node.left.center, cluster_node.right.center))
        if hasattr(self, "_X_mean"):
            centers += self._X_mean

        # 使用线程池限制的函数计算数据点的聚类标签
        cluster_labels = _labels_inertia_threadpool_limit(
            X,
            sample_weight,
            centers,
            self._n_threads,
            return_inertia=False,
        )

        # 确定属于左子集群的数据点的掩码
        mask = cluster_labels == 0

        # 初始化标签数组
        labels = np.full(X.shape[0], -1, dtype=np.int32)

        # 递归计算左子集群数据点的标签
        labels[mask] = self._predict_recursive(
            X[mask], sample_weight[mask], cluster_node.left
        )

        # 递归计算右子集群数据点的标签
        labels[~mask] = self._predict_recursive(
            X[~mask], sample_weight[~mask], cluster_node.right
        )

        return labels

    def _more_tags(self):
        # 返回关于模型的额外标签信息
        return {"preserves_dtype": [np.float64, np.float32]}
```