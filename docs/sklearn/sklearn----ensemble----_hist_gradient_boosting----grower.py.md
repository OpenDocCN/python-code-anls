# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\grower.py`

```
"""
This module contains the TreeGrower class.

TreeGrower builds a regression tree fitting a Newton-Raphson step, based on
the gradients and hessians of the training data.
"""

# Author: Nicolas Hug

# 引入必要的库和模块
import numbers  # 用于数值判断的库
from heapq import heappop, heappush  # 用于堆操作的库
from timeit import default_timer as time  # 用于时间测量的库

# 引入第三方库
import numpy as np  # 用于科学计算的库

# 引入自定义模块和函数
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads  # 多线程帮助函数
from ...utils.arrayfuncs import sum_parallel  # 并行求和函数
from ._bitset import set_raw_bitset_from_binned_bitset  # 位集合操作函数
from .common import (
    PREDICTOR_RECORD_DTYPE,
    X_BITSET_INNER_DTYPE,
    MonotonicConstraint,
)  # 导入通用数据结构和约束条件类
from .histogram import HistogramBuilder  # 直方图构建器类
from .predictor import TreePredictor  # 树预测器类
from .splitting import Splitter  # 分裂器类

# 定义树节点类
class TreeNode:
    """Tree Node class used in TreeGrower.

    This isn't used for prediction purposes, only for training (see
    TreePredictor).

    Parameters
    ----------
    depth : int
        The depth of the node, i.e. its distance from the root.
    sample_indices : ndarray of shape (n_samples_at_node,), dtype=np.uint32
        The indices of the samples at the node.
    partition_start : int
        start position of the node's sample_indices in splitter.partition.
    partition_stop : int
        stop position of the node's sample_indices in splitter.partition.
    sum_gradients : float
        The sum of the gradients of the samples at the node.
    sum_hessians : float
        The sum of the hessians of the samples at the node.

    Attributes
    ----------
    depth : int
        The depth of the node, i.e. its distance from the root.
    sample_indices : ndarray of shape (n_samples_at_node,), dtype=np.uint32
        The indices of the samples at the node.
    sum_gradients : float
        The sum of the gradients of the samples at the node.
    sum_hessians : float
        The sum of the hessians of the samples at the node.
    split_info : SplitInfo or None
        The result of the split evaluation.
    is_leaf : bool
        True if node is a leaf
    left_child : TreeNode or None
        The left child of the node. None for leaves.
    right_child : TreeNode or None
        The right child of the node. None for leaves.
    value : float or None
        The value of the leaf, as computed in finalize_leaf(). None for
        non-leaf nodes.
    partition_start : int
        start position of the node's sample_indices in splitter.partition.
    partition_stop : int
        stop position of the node's sample_indices in splitter.partition.
    allowed_features : None or ndarray, dtype=int
        Indices of features allowed to split for children.
    interaction_cst_indices : None or list of ints
        Indices of the interaction sets that have to be applied on splits of
        child nodes. The fewer sets the stronger the constraint as fewer sets
        contain fewer features.
    children_lower_bound : float
    children_upper_bound : float
    """
    def __init__(
        self,
        *,
        depth,
        sample_indices,
        partition_start,
        partition_stop,
        sum_gradients,
        sum_hessians,
        value=None,
    ):
        # 初始化函数，设置节点的各种属性
        self.depth = depth  # 树的深度
        self.sample_indices = sample_indices  # 节点包含的样本索引
        self.n_samples = sample_indices.shape[0]  # 样本数量
        self.sum_gradients = sum_gradients  # 节点内样本梯度总和
        self.sum_hessians = sum_hessians  # 节点内样本Hessian矩阵总和
        self.value = value  # 节点的值（可选）
        self.is_leaf = False  # 是否为叶子节点的标志
        self.allowed_features = None  # 允许使用的特征
        self.interaction_cst_indices = None  # 交互常数索引（未指定）
        self.set_children_bounds(float("-inf"), float("+inf"))  # 设置子节点值的边界
        self.split_info = None  # 分裂信息（未指定）
        self.left_child = None  # 左子节点（未指定）
        self.right_child = None  # 右子节点（未指定）
        self.histograms = None  # 直方图（未指定）

        # 节点在分裂器（splitter）分区数组中的起始和终止索引
        # 具体来说，
        # self.sample_indices = view(self.splitter.partition[start:stop])
        # 有关splitter.partition和splitter.split_indices的更多信息，请参见注释。
        # 这两个属性仅在_update_raw_prediction中使用，因为我们需要遍历叶子节点，
        # 并且我不知道如何高效地存储sample_indices视图，因为它们的大小都不同。
        self.partition_start = partition_start  # 分区起始索引
        self.partition_stop = partition_stop  # 分区终止索引

    def set_children_bounds(self, lower, upper):
        """设置子节点值的边界，以满足单调性约束。"""

        # 这些是节点的*子节点*值的边界，而不是节点自身的值。这些边界在分裂器（splitter）
        # 考虑潜在的左子节点和右子节点时使用。
        self.children_lower_bound = lower  # 子节点值的下界
        self.children_upper_bound = upper  # 子节点值的上界

    def __lt__(self, other_node):
        """优先队列的比较方法。

        具有较高增益的节点比具有较低增益的节点具有更高的优先级。

        heapq.heappush只需要'<'操作符。
        heapq.heappop首先取最小的项（较小的是更高的优先级）。

        Parameters
        ----------
        other_node : TreeNode
            要比较的节点。
        """
        return self.split_info.gain > other_node.split_info.gain
# 定义一个树生长器类，用于构建树模型。

class TreeGrower:
    """Tree grower class used to build a tree.

    The tree is fitted to predict the values of a Newton-Raphson step. The
    splits are considered in a best-first fashion, and the quality of a
    split is defined in splitting._split_gain.
    
    树生长器类，用于构建树模型。

    Parameters
    ----------
    X_binned : ndarray of shape (n_samples, n_features), dtype=np.uint8
        The binned input samples. Must be Fortran-aligned.
        输入样本的分箱表示。必须是Fortran对齐的。

    gradients : ndarray of shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration ``i - 1``.
        每个训练样本的梯度。这些是损失相对于预测的梯度，在第``i - 1``次迭代中评估。

    hessians : ndarray of shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration ``i - 1``.
        每个训练样本的Hessian矩阵。这些是损失相对于预测的Hessian矩阵，在第``i - 1``次迭代中评估。

    max_leaf_nodes : int, default=None
        The maximum number of leaves for each tree. If None, there is no
        maximum limit.
        每棵树的最大叶子节点数。如果为None，则没有最大限制。

    max_depth : int, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
        每棵树的最大深度。树的深度是从根到最深叶子的边数。默认情况下深度没有约束。

    min_samples_leaf : int, default=20
        The minimum number of samples per leaf.
        每个叶子节点的最小样本数。

    min_gain_to_split : float, default=0.
        The minimum gain needed to split a node. Splits with lower gain will
        be ignored.
        拆分节点所需的最小增益。低于该增益的拆分将被忽略。

    min_hessian_to_split : float, default=1e-3
        The minimum sum of hessians needed in each node. Splits that result in
        at least one child having a sum of hessians less than
        ``min_hessian_to_split`` are discarded.
        每个节点所需的最小Hessian矩阵之和。导致至少一个子节点的Hessian矩阵之和小于``min_hessian_to_split``的拆分将被丢弃。

    n_bins : int, default=256
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
        总的箱数，包括缺失值的箱。用于定义直方图的形状。

    n_bins_non_missing : ndarray, dtype=np.uint32, default=None
        For each feature, gives the number of bins actually used for
        non-missing values. For features with a lot of unique values, this
        is equal to ``n_bins - 1``. If it's an int, all features are
        considered to have the same number of bins. If None, all features
        are considered to have ``n_bins - 1`` bins.
        每个特征的实际使用的非缺失值箱数。对于具有大量唯一值的特征，这等于``n_bins - 1``。如果是int型，则所有特征被认为有相同数量的箱。如果是None，则所有特征被认为有``n_bins - 1``个箱。

    has_missing_values : bool or ndarray, dtype=bool, default=False
        Whether each feature contains missing values (in the training data).
        If it's a bool, the same value is used for all features.
        每个特征是否包含缺失值（在训练数据中）。如果是bool型，则所有特征使用相同的值。

    is_categorical : ndarray of bool of shape (n_features,), default=None
        Indicates categorical features.
        表示分类特征的布尔型数组。

    monotonic_cst : array-like of int of shape (n_features,), dtype=int, default=None
        Indicates the monotonic constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease
        表示要在每个特征上强制执行的单调约束。

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

    interaction_cst : list of sets of integers, default=None
        List of interaction constraints.
        交互约束的列表。

    """
    l2_regularization : float, default=0.
        The L2 regularization parameter penalizing leaves with small hessians.
        Use ``0`` for no regularization (default).
    feature_fraction_per_split : float, default=1
        Proportion of randomly chosen features in each and every node split.
        This is a form of regularization, smaller values make the trees weaker
        learners and might prevent overfitting.
    rng : Generator
        Numpy random Generator used for feature subsampling.
    shrinkage : float, default=1.
        The shrinkage parameter to apply to the leaves values, also known as
        learning rate.
    n_threads : int, default=None
        Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
        to determine the effective number of threads use, which takes cgroups CPU
        quotes into account. See the docstring of `_openmp_effective_n_threads`
        for details.

    Attributes
    ----------
    histogram_builder : HistogramBuilder
        Builds histograms needed for splitting nodes in the tree.
    splitter : Splitter
        Object responsible for making decisions on how to split nodes.
    root : TreeNode
        The root node of the gradient boosted decision tree.
    finalized_leaves : list of TreeNode
        List containing nodes that are finalized (no further splits will be made).
    splittable_nodes : list of TreeNode
        List of nodes that are eligible for further splitting.
    missing_values_bin_idx : int
        Index representing the bin for missing values, equals n_bins - 1.
    n_categorical_splits : int
        Number of categorical splits encountered during tree building.
    n_features : int
        Number of features in the dataset.
    n_nodes : int
        Total number of nodes in the decision tree.
    total_find_split_time : float
        Time spent finding the best splits during training.
    total_compute_hist_time : float
        Time spent computing histograms for nodes.
    total_apply_split_time : float
        Time spent applying splits to nodes.
    with_monotonic_cst : bool
        Indicates whether monotonic constraints are applied. False if monotonic_cst is None.
    """

    def __init__(
        self,
        X_binned,
        gradients,
        hessians,
        max_leaf_nodes=None,
        max_depth=None,
        min_samples_leaf=20,
        min_gain_to_split=0.0,
        min_hessian_to_split=1e-3,
        n_bins=256,
        n_bins_non_missing=None,
        has_missing_values=False,
        is_categorical=None,
        monotonic_cst=None,
        interaction_cst=None,
        l2_regularization=0.0,
        feature_fraction_per_split=1.0,
        rng=np.random.default_rng(),
        shrinkage=1.0,
        n_threads=None,
    ):
        """
        Constructor for the GradientBoostingTree class.

        Parameters:
        -----------
        X_binned : array-like of shape (n_samples, n_features)
            Binned input data used for training the tree.
        gradients : array-like of shape (n_samples,)
            Gradient values for each sample.
        hessians : array-like of shape (n_samples,)
            Hessian values for each sample, used for calculating leaf weights.
        max_leaf_nodes : int or None, optional, default=None
            Maximum number of leaves per tree. If None, unlimited growth of leaves.
        max_depth : int or None, optional, default=None
            Maximum depth of the tree. If None, nodes are expanded until min_samples_leaf.
        min_samples_leaf : int, optional, default=20
            Minimum number of samples required to be at a leaf node.
        min_gain_to_split : float, optional, default=0.0
            Minimum gain required for a node to be split further.
        min_hessian_to_split : float, optional, default=1e-3
            Minimum sum of hessian required for a node to be split further.
        n_bins : int, optional, default=256
            Number of bins used for binning continuous features.
        n_bins_non_missing : int or None, optional, default=None
            Number of bins excluding missing values. If None, defaults to n_bins.
        has_missing_values : bool, optional, default=False
            Whether the dataset contains missing values.
        is_categorical : array-like of shape (n_features,) or None, optional, default=None
            Indicates which features are categorical. If None, inferred from data.
        monotonic_cst : array-like of shape (n_features,) or None, optional, default=None
            Monotonic constraints for each feature. None means no constraints.
        interaction_cst : array-like of shape (n_features, n_features) or None, optional, default=None
            Interaction constraints for feature pairs. None means no interaction constraints.
        l2_regularization : float, optional, default=0.0
            L2 regularization parameter.
        feature_fraction_per_split : float, optional, default=1.0
            Proportion of features randomly chosen per split.
        rng : Generator, optional, default=np.random.default_rng()
            Random number generator.
        shrinkage : float, optional, default=1.0
            Learning rate or shrinkage parameter.
        n_threads : int or None, optional, default=None
            Number of OpenMP threads to use for parallel computation.

        """

    def _validate_parameters(
        self,
        X_binned,
        min_gain_to_split,
        min_hessian_to_split,
        ):
        """
        Validates the parameters used for building the gradient boosted tree.

        Parameters:
        -----------
        X_binned : array-like of shape (n_samples, n_features)
            Binned input data used for training the tree.
        min_gain_to_split : float
            Minimum gain required for a node to be split further.
        min_hessian_to_split : float
            Minimum sum of hessian required for a node to be split further.
        """
    ):
        """Validate parameters passed to __init__.

        Also validate parameters passed to splitter.
        """
        # 检查 X_binned 的数据类型是否为 uint8
        if X_binned.dtype != np.uint8:
            raise NotImplementedError("X_binned must be of type uint8.")
        # 检查 X_binned 是否按照 Fortran 连续性存储，以确保最大效率
        if not X_binned.flags.f_contiguous:
            raise ValueError(
                "X_binned should be passed as Fortran contiguous "
                "array for maximum efficiency."
            )
        # 检查 min_gain_to_split 是否大于等于 0
        if min_gain_to_split < 0:
            raise ValueError(
                "min_gain_to_split={} must be positive.".format(min_gain_to_split)
            )
        # 检查 min_hessian_to_split 是否大于等于 0
        if min_hessian_to_split < 0:
            raise ValueError(
                "min_hessian_to_split={} must be positive.".format(min_hessian_to_split)
            )

    def grow(self):
        """Grow the tree, from root to leaves."""
        # 不断进行节点分裂，直到没有可分裂节点为止
        while self.splittable_nodes:
            self.split_next()

        # 应用收缩因子到叶子节点的值
        self._apply_shrinkage()

    def _apply_shrinkage(self):
        """Multiply leaves values by shrinkage parameter.

        This must be done at the very end of the growing process. If this were
        done during the growing process e.g. in finalize_leaf(), then a leaf
        would be shrunk but its sibling would potentially not be (if it's a
        non-leaf), which would lead to a wrong computation of the 'middle'
        value needed to enforce the monotonic constraints.
        """
        # 将所有已确定的叶子节点的值乘以收缩因子
        for leaf in self.finalized_leaves:
            leaf.value *= self.shrinkage
    def _initialize_root(self, gradients, hessians):
        """Initialize root node and finalize it if needed."""
        # 获取样本数量
        n_samples = self.X_binned.shape[0]
        # 根节点深度初始化为0
        depth = 0
        # 并行求和梯度
        sum_gradients = sum_parallel(gradients, self.n_threads)
        # 如果节点的 Hessian 矩阵是常数，则直接使用第一个元素乘以样本数量
        if self.histogram_builder.hessians_are_constant:
            sum_hessians = hessians[0] * n_samples
        else:
            # 否则并行求和 Hessian 矩阵
            sum_hessians = sum_parallel(hessians, self.n_threads)
        # 创建根节点
        self.root = TreeNode(
            depth=depth,
            sample_indices=self.splitter.partition,
            partition_start=0,
            partition_stop=n_samples,
            sum_gradients=sum_gradients,
            sum_hessians=sum_hessians,
            value=0,
        )

        # 如果根节点样本数小于两倍的最小叶子节点样本数，则直接完成叶子节点的初始化并返回
        if self.root.n_samples < 2 * self.min_samples_leaf:
            self._finalize_leaf(self.root)
            return
        # 如果根节点的 Hessian 和小于最小切分所需的 Hessian 值，则直接完成叶子节点的初始化并返回
        if sum_hessians < self.splitter.min_hessian_to_split:
            self._finalize_leaf(self.root)
            return

        # 如果存在交互约束，则设置根节点的交互约束索引和允许特征集合
        if self.interaction_cst is not None:
            self.root.interaction_cst_indices = range(len(self.interaction_cst))
            allowed_features = set().union(*self.interaction_cst)
            self.root.allowed_features = np.fromiter(
                allowed_features, dtype=np.uint32, count=len(allowed_features)
            )

        # 计算直方图
        tic = time()
        self.root.histograms = self.histogram_builder.compute_histograms_brute(
            self.root.sample_indices, self.root.allowed_features
        )
        self.total_compute_hist_time += time() - tic

        # 计算最佳切分并将节点推入可切分节点堆中
        tic = time()
        self._compute_best_split_and_push(self.root)
        self.total_find_split_time += time() - tic

    def _compute_best_split_and_push(self, node):
        """Compute the best possible split (SplitInfo) of a given node.

        Also push it in the heap of splittable nodes if gain isn't zero.
        The gain of a node is 0 if either all the leaves are pure
        (best gain = 0), or if no split would satisfy the constraints,
        (min_hessians_to_split, min_gain_to_split, min_samples_leaf)
        """
        # 找到节点的最佳切分
        node.split_info = self.splitter.find_node_split(
            n_samples=node.n_samples,
            histograms=node.histograms,
            sum_gradients=node.sum_gradients,
            sum_hessians=node.sum_hessians,
            value=node.value,
            lower_bound=node.children_lower_bound,
            upper_bound=node.children_upper_bound,
            allowed_features=node.allowed_features,
        )

        # 如果节点的增益小于等于0，表示无效切分，则完成叶子节点的初始化
        if node.split_info.gain <= 0:
            self._finalize_leaf(node)
        else:
            # 否则将节点推入可切分节点堆中
            heappush(self.splittable_nodes, node)
    def _compute_interactions(self, node):
        r"""Compute features allowed by interactions to be inherited by child nodes.

        Example: Assume constraints [{0, 1}, {1, 2}].
           1      <- Both constraint groups could be applied from now on
          / \
         1   2    <- Left split still fulfills both constraint groups.
        / \ / \      Right split at feature 2 has only group {1, 2} from now on.

        LightGBM uses the same logic for overlapping groups. See
        https://github.com/microsoft/LightGBM/issues/4481 for details.

        Parameters:
        ----------
        node : TreeNode
            A node that might have children. Based on its feature_idx, the interaction
            constraints for possible child nodes are computed.

        Returns
        -------
        allowed_features : ndarray, dtype=uint32
            Indices of features allowed to split for children.
        interaction_cst_indices : list of ints
            Indices of the interaction sets that have to be applied on splits of
            child nodes. The fewer sets the stronger the constraint as fewer sets
            contain fewer features.
        """
        # Note:
        #  - Case of no interactions is already captured before function call.
        #  - This is for nodes that are already split and have a
        #    node.split_info.feature_idx.
        
        # 初始化一个空集合，用于存储允许进行分裂的特征索引
        allowed_features = set()
        # 初始化一个空列表，用于存储需要应用于子节点分裂的交互约束集合索引
        interaction_cst_indices = []
        
        # 遍历当前节点的交互约束集合索引列表
        for i in node.interaction_cst_indices:
            # 如果当前节点的分裂信息中包含在交互约束集合中
            if node.split_info.feature_idx in self.interaction_cst[i]:
                # 将该集合索引添加到需要应用的交互约束集合索引列表中
                interaction_cst_indices.append(i)
                # 更新允许分裂的特征索引集合，包含当前交互约束集合中的所有特征索引
                allowed_features.update(self.interaction_cst[i])
        
        # 将允许分裂的特征索引集合转换为 numpy 数组，并返回允许分裂的特征索引数组和需要应用的交互约束集合索引列表
        return (
            np.fromiter(allowed_features, dtype=np.uint32, count=len(allowed_features)),
            interaction_cst_indices,
        )

    def _finalize_leaf(self, node):
        """Make node a leaf of the tree being grown."""
        
        # 将当前节点标记为叶节点
        node.is_leaf = True
        # 将当前节点添加到已最终化的叶节点列表中
        self.finalized_leaves.append(node)

    def _finalize_splittable_nodes(self):
        """Transform all splittable nodes into leaves.

        Used when some constraint is met e.g. maximum number of leaves or
        maximum depth."""
        
        # 当有可分裂节点时循环执行以下操作
        while len(self.splittable_nodes) > 0:
            # 从可分裂节点列表中取出一个节点
            node = self.splittable_nodes.pop()
            # 最终化该节点，即将其转换为叶节点
            self._finalize_leaf(node)
    # 创建一个方法，用于生成 TreePredictor 对象，基于当前的树结构

    # 创建 predictor_nodes 数组，用于存储预测节点信息，初始为全零数组
    predictor_nodes = np.zeros(self.n_nodes, dtype=PREDICTOR_RECORD_DTYPE)
    
    # 创建 binned_left_cat_bitsets 数组，存储分类特征左子树的二进制位集合，初始为全零数组
    binned_left_cat_bitsets = np.zeros(
        (self.n_categorical_splits, 8), dtype=X_BITSET_INNER_DTYPE
    )
    
    # 创建 raw_left_cat_bitsets 数组，存储原始分类特征左子树的二进制位集合，初始为全零数组
    raw_left_cat_bitsets = np.zeros(
        (self.n_categorical_splits, 8), dtype=X_BITSET_INNER_DTYPE
    )
    
    # 调用 _fill_predictor_arrays 函数，填充上述数组和节点信息
    _fill_predictor_arrays(
        predictor_nodes,
        binned_left_cat_bitsets,
        raw_left_cat_bitsets,
        self.root,
        binning_thresholds,
        self.n_bins_non_missing,
    )
    
    # 返回一个新的 TreePredictor 对象，该对象由上述数组组成
    return TreePredictor(
        predictor_nodes, binned_left_cat_bitsets, raw_left_cat_bitsets
    )
def _fill_predictor_arrays(
    predictor_nodes,
    binned_left_cat_bitsets,
    raw_left_cat_bitsets,
    grower_node,
    binning_thresholds,
    n_bins_non_missing,
    next_free_node_idx=0,
    next_free_bitset_idx=0,
):
    """Helper used in make_predictor to set the TreePredictor fields."""
    # 获取当前节点
    node = predictor_nodes[next_free_node_idx]
    # 设置节点的样本计数为生长节点的样本数
    node["count"] = grower_node.n_samples
    # 设置节点的深度为生长节点的深度
    node["depth"] = grower_node.depth
    # 如果存在分裂信息，则设置节点的增益为分裂信息中的增益值，否则设为-1
    if grower_node.split_info is not None:
        node["gain"] = grower_node.split_info.gain
    else:
        node["gain"] = -1

    # 设置节点的值为生长节点的值
    node["value"] = grower_node.value

    # 如果是叶子节点
    if grower_node.is_leaf:
        node["is_leaf"] = True  # 标记节点为叶子节点
        return next_free_node_idx + 1, next_free_bitset_idx

    # 如果不是叶子节点，获取分裂信息
    split_info = grower_node.split_info
    feature_idx, bin_idx = split_info.feature_idx, split_info.bin_idx
    # 设置节点的特征索引和分裂阈值
    node["feature_idx"] = feature_idx
    node["bin_threshold"] = bin_idx
    # 设置节点的缺失值处理方式和是否为分类特征
    node["missing_go_to_left"] = split_info.missing_go_to_left
    node["is_categorical"] = split_info.is_categorical

    # 根据分裂信息设置节点的阈值
    if split_info.bin_idx == n_bins_non_missing[feature_idx] - 1:
        # 如果分裂是在最后一个非缺失的箱子上：这是一个“缺失值分裂”。
        # 所有的缺失值都走右边，其余的走左边。
        # 注意：对于分类分裂，bin_idx为0，我们依赖于位集
        node["num_threshold"] = np.inf
    elif split_info.is_categorical:
        # 如果是分类特征
        categories = binning_thresholds[feature_idx]
        node["bitset_idx"] = next_free_bitset_idx
        # 将左侧类别位集设置为分裂信息中的左侧类别位集
        binned_left_cat_bitsets[next_free_bitset_idx] = split_info.left_cat_bitset
        # 从分类位集设置原始位集
        set_raw_bitset_from_binned_bitset(
            raw_left_cat_bitsets[next_free_bitset_idx],
            split_info.left_cat_bitset,
            categories,
        )
        next_free_bitset_idx += 1
    else:
        # 如果是数值特征，设置节点的数值阈值
        node["num_threshold"] = binning_thresholds[feature_idx][bin_idx]

    # 更新下一个可用节点索引
    next_free_node_idx += 1

    # 设置节点的左子节点索引
    node["left"] = next_free_node_idx
    # 递归填充左子树
    next_free_node_idx, next_free_bitset_idx = _fill_predictor_arrays(
        predictor_nodes,
        binned_left_cat_bitsets,
        raw_left_cat_bitsets,
        grower_node.left_child,
        binning_thresholds=binning_thresholds,
        n_bins_non_missing=n_bins_non_missing,
        next_free_node_idx=next_free_node_idx,
        next_free_bitset_idx=next_free_bitset_idx,
    )

    # 设置节点的右子节点索引
    node["right"] = next_free_node_idx
    # 递归填充右子树
    return _fill_predictor_arrays(
        predictor_nodes,
        binned_left_cat_bitsets,
        raw_left_cat_bitsets,
        grower_node.right_child,
        binning_thresholds=binning_thresholds,
        n_bins_non_missing=n_bins_non_missing,
        next_free_node_idx=next_free_node_idx,
        next_free_bitset_idx=next_free_bitset_idx,
    )
```