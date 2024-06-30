# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\splitting.pyx`

```
# 定义一个 Cython 扩展模块，包含了用于决策树中节点分裂的相关数据结构和函数
# 以及找到节点最佳分裂和应用分裂的功能

# 导入 Cython 相关模块和库
cimport cython
from cython.parallel import prange
import numpy as np

# 导入 C 语言的数学库函数和标准库函数
from libc.math cimport INFINITY, ceil
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy

# 导入自定义类型定义
from ...utils._typedefs cimport uint8_t
from .common cimport X_BINNED_DTYPE_C
from .common cimport Y_DTYPE_C
from .common cimport hist_struct
from .common cimport BITSET_INNER_DTYPE_C
from .common cimport BITSET_DTYPE_C
from .common cimport MonotonicConstraint

# 导入位集合操作的相关函数
from ._bitset cimport init_bitset
from ._bitset cimport set_bitset
from ._bitset cimport in_bitset


# 定义一个 C 结构体，用于存储节点分裂的相关信息
cdef struct split_info_struct:
    # 分裂信息结构体，用于在 nogil 环境下和数组中使用
    Y_DTYPE_C gain
    int feature_idx
    unsigned int bin_idx
    unsigned char missing_go_to_left
    Y_DTYPE_C sum_gradient_left
    Y_DTYPE_C sum_gradient_right
    Y_DTYPE_C sum_hessian_left
    Y_DTYPE_C sum_hessian_right
    unsigned int n_samples_left
    unsigned int n_samples_right
    Y_DTYPE_C value_left
    Y_DTYPE_C value_right
    unsigned char is_categorical
    BITSET_DTYPE_C left_cat_bitset


# 用于分类特征分裂的排序分类信息结构体
cdef struct categorical_info:
    X_BINNED_DTYPE_C bin_idx
    Y_DTYPE_C value


# 定义一个 Python 类，用于存储潜在分裂的信息，这是一个纯数据类
class SplitInfo:
    """存储潜在分裂信息的纯数据类。

    Parameters
    ----------
    gain : float
        分裂的增益值。
    feature_idx : int
        要分裂的特征的索引。
    bin_idx : int
        分裂的箱的索引。如果 `is_categorical` 为 True，则应忽略此值，
        而使用 `left_cat_bitset` 来确定分裂。
    missing_go_to_left : bool
        缺失值是否归属于左侧子节点。无论分裂是否是分类的都会使用此选项。
    sum_gradient_left : float
        左侧子节点所有样本梯度的总和。
    sum_hessian_left : float
        左侧子节点所有样本 Hessian 矩阵的总和。
    sum_gradient_right : float
        右侧子节点所有样本梯度的总和。
    sum_hessian_right : float
        右侧子节点所有样本 Hessian 矩阵的总和。
    n_samples_left : int, 默认=0
        左侧子节点中的样本数。
    n_samples_right : int
        右侧子节点中的样本数。
    is_categorical : bool
        分裂是否基于分类特征。
    """
    # 表示左侧分类的位集合，类型为 ndarray，形状为 (8,)，数据类型为 uint32 或 None
    # 当 is_categorical 为 True 时使用，表示哪些类别被归为左侧
    # 注意，如果训练数据中存在缺失值，缺失值也将包含在该位集合中
    # 在分裂时，对缺失值依赖于该位集合，但在预测时依赖于 missing_go_to_left
    left_cat_bitset : ndarray of shape=(8,), dtype=uint32 or None
        Bitset representing the categories that go to the left. This is used
        only when `is_categorical` is True.
        Note that missing values are part of that bitset if there are missing
        values in the training data. For missing values, we rely on that
        bitset for splitting, but at prediction time, we rely on
        missing_go_to_left.
    """
    def __init__(self, gain, feature_idx, bin_idx,
                 missing_go_to_left, sum_gradient_left, sum_hessian_left,
                 sum_gradient_right, sum_hessian_right, n_samples_left,
                 n_samples_right, value_left, value_right,
                 is_categorical, left_cat_bitset):
        self.gain = gain
        self.feature_idx = feature_idx
        self.bin_idx = bin_idx
        self.missing_go_to_left = missing_go_to_left
        self.sum_gradient_left = sum_gradient_left
        self.sum_hessian_left = sum_hessian_left
        self.sum_gradient_right = sum_gradient_right
        self.sum_hessian_right = sum_hessian_right
        self.n_samples_left = n_samples_left
        self.n_samples_right = n_samples_right
        self.value_left = value_left
        self.value_right = value_right
        self.is_categorical = is_categorical
        self.left_cat_bitset = left_cat_bitset
# 使用 Cython 的 final 关键字定义一个不可继承的类 Splitter
@cython.final
cdef class Splitter:
    """Splitter used to find the best possible split at each node.

    A split (see SplitInfo) is characterized by a feature and a bin.

    The Splitter is also responsible for partitioning the samples among the
    leaves of the tree (see split_indices() and the partition attribute).

    Parameters
    ----------
    X_binned : ndarray of int, shape (n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    n_bins_non_missing : ndarray, shape (n_features,)
        For each feature, gives the number of bins actually used for
        non-missing values.
    missing_values_bin_idx : uint8
        Index of the bin that is used for missing values. This is the index of
        the last bin and is always equal to max_bins (as passed to the GBDT
        classes), or equivalently to n_bins - 1.
    has_missing_values : ndarray, shape (n_features,)
        Whether missing values were observed in the training data, for each
        feature.
    is_categorical : ndarray of bool of shape (n_features,)
        Indicates categorical features.
    monotonic_cst : ndarray of int of shape (n_features,), dtype=int
        Indicates the monotonic constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.
    l2_regularization : float
        The L2 regularization parameter.
    min_hessian_to_split : float, default=1e-3
        The minimum sum of hessians needed in each node. Splits that result in
        at least one child having a sum of hessians less than
        min_hessian_to_split are discarded.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf.
    min_gain_to_split : float, default=0.0
        The minimum gain needed to split a node. Splits with lower gain will
        be ignored.
    hessians_are_constant: bool, default is False
        Whether hessians are constant.
    feature_fraction_per_split : float, default=1
        Proportion of randomly chosen features in each and every node split.
        This is a form of regularization, smaller values make the trees weaker
        learners and might prevent overfitting.
    rng : Generator
    n_threads : int, default=1
        Number of OpenMP threads to use.
    """
    # 定义公共成员变量，用于存储梯度提升决策树（GBDT）训练过程中的各种数据和配置信息

    # 二维数组，存储经过分箱处理后的特征数据
    const X_BINNED_DTYPE_C [::1, :] X_binned

    # 特征数量
    unsigned int n_features

    # 一维数组，存储非缺失值样本的每个特征的分箱数量
    const unsigned int [::1] n_bins_non_missing

    # 无效值（如缺失值）的分箱索引
    unsigned char missing_values_bin_idx

    # 一维数组，标记每个特征是否存在缺失值
    const unsigned char [::1] has_missing_values

    # 一维数组，标记每个特征是否是分类特征
    const unsigned char [::1] is_categorical

    # 一维数组，记录特征的单调性常数
    const signed char [::1] monotonic_cst

    # 标志变量，指示样本上的 Hessian 矩阵是否是常数
    unsigned char hessians_are_constant

    # L2 正则化参数
    Y_DTYPE_C l2_regularization

    # 划分节点所需的最小 Hessian 矩阵值
    Y_DTYPE_C min_hessian_to_split

    # 叶子节点所需的最小样本数
    unsigned int min_samples_leaf

    # 分裂节点所需的最小增益
    Y_DTYPE_C min_gain_to_split

    # 每次分裂所使用的特征比例
    Y_DTYPE_C feature_fraction_per_split

    # 随机数生成器对象
    rng

    # 一维数组，存储样本索引的分区信息
    unsigned int [::1] partition

    # 用于存储左子节点索引的缓冲区
    unsigned int [::1] left_indices_buffer

    # 用于存储右子节点索引的缓冲区
    unsigned int [::1] right_indices_buffer

    # 并行线程数
    int n_threads
    def __init__(self,
                 const X_BINNED_DTYPE_C [::1, :] X_binned,  # 二维数组，用于存储经过分箱处理的特征数据
                 const unsigned int [::1] n_bins_non_missing,  # 一维数组，存储每个特征非缺失值的箱数
                 const unsigned char missing_values_bin_idx,  # 无符号字符，表示缺失值的箱索引
                 const unsigned char [::1] has_missing_values,  # 一维数组，标记每个特征是否含有缺失值
                 const unsigned char [::1] is_categorical,  # 一维数组，标记每个特征是否为分类特征
                 const signed char [::1] monotonic_cst,  # 一维数组，存储每个特征的单调性约束
                 Y_DTYPE_C l2_regularization,  # 类型为Y_DTYPE_C的参数，L2正则化强度
                 Y_DTYPE_C min_hessian_to_split=1e-3,  # 类型为Y_DTYPE_C的参数，默认最小的Hessian值用于分割
                 unsigned int min_samples_leaf=20,  # 无符号整数，叶子节点的最小样本数
                 Y_DTYPE_C min_gain_to_split=0.,  # 类型为Y_DTYPE_C的参数，分割特征所需的最小增益
                 unsigned char hessians_are_constant=False,  # 布尔值，指示Hessian值是否恒定
                 Y_DTYPE_C feature_fraction_per_split=1.0,  # 类型为Y_DTYPE_C的参数，每次分割使用的特征比例
                 rng=np.random.RandomState(),  # 随机数生成器对象，默认使用NumPy的随机数生成器
                 unsigned int n_threads=1):  # 无符号整数，指定并行处理的线程数

        self.X_binned = X_binned  # 将参数 X_binned 赋值给对象属性 X_binned
        self.n_features = X_binned.shape[1]  # 计算特征的数量并将其存储在对象属性 n_features 中
        self.n_bins_non_missing = n_bins_non_missing  # 将参数 n_bins_non_missing 赋值给对象属性 n_bins_non_missing
        self.missing_values_bin_idx = missing_values_bin_idx  # 将参数 missing_values_bin_idx 赋值给对象属性 missing_values_bin_idx
        self.has_missing_values = has_missing_values  # 将参数 has_missing_values 赋值给对象属性 has_missing_values
        self.is_categorical = is_categorical  # 将参数 is_categorical 赋值给对象属性 is_categorical
        self.monotonic_cst = monotonic_cst  # 将参数 monotonic_cst 赋值给对象属性 monotonic_cst
        self.l2_regularization = l2_regularization  # 将参数 l2_regularization 赋值给对象属性 l2_regularization
        self.min_hessian_to_split = min_hessian_to_split  # 将参数 min_hessian_to_split 赋值给对象属性 min_hessian_to_split
        self.min_samples_leaf = min_samples_leaf  # 将参数 min_samples_leaf 赋值给对象属性 min_samples_leaf
        self.min_gain_to_split = min_gain_to_split  # 将参数 min_gain_to_split 赋值给对象属性 min_gain_to_split
        self.hessians_are_constant = hessians_are_constant  # 将参数 hessians_are_constant 赋值给对象属性 hessians_are_constant
        self.feature_fraction_per_split = feature_fraction_per_split  # 将参数 feature_fraction_per_split 赋值给对象属性 feature_fraction_per_split
        self.rng = rng  # 将参数 rng 赋值给对象属性 rng
        self.n_threads = n_threads  # 将参数 n_threads 赋值给对象属性 n_threads

        # The partition array maps each sample index into the leaves of the
        # tree (a leaf in this context is a node that isn't split yet, not
        # necessarily a 'finalized' leaf). Initially, the root contains all
        # the indices, e.g.:
        # partition = [abcdefghijkl]
        # After a call to split_indices, it may look e.g. like this:
        # partition = [cef|abdghijkl]
        # we have 2 leaves, the left one is at position 0 and the second one at
        # position 3. The order of the samples is irrelevant.
        self.partition = np.arange(X_binned.shape[0], dtype=np.uint32)
        # buffers used in split_indices to support parallel splitting.
        self.left_indices_buffer = np.empty_like(self.partition)
        self.right_indices_buffer = np.empty_like(self.partition)

    cdef int _find_best_feature_to_split_helper(
        self,
        split_info_struct * split_infos,  # 输入参数，一个结构体数组，包含每个特征的分割信息
        int n_allowed_features,  # 输入参数，允许考虑的特征数量
    ) noexcept nogil:
        """Return the index of split_infos with the best feature split."""
        cdef:
            int split_info_idx
            int best_split_info_idx = 0

        for split_info_idx in range(1, n_allowed_features):
            if (split_infos[split_info_idx].gain > split_infos[best_split_info_idx].gain):
                best_split_info_idx = split_info_idx
        return best_split_info_idx
# 定义一个 Cython 函数，用于比较两个 categorical_info 结构体的值，作为排序的依据
cdef int compare_cat_infos(const void * a, const void * b) noexcept nogil:
    return -1 if (<categorical_info *>a).value < (<categorical_info *>b).value else 1

# 定义一个 Cython 内联函数，计算在执行分裂操作后相对于保持节点为叶子节点的损失减少量
# 参数说明:
# sum_gradient_left: 左子节点的梯度总和
# sum_hessian_left: 左子节点的Hessian矩阵总和
# sum_gradient_right: 右子节点的梯度总和
# sum_hessian_right: 右子节点的Hessian矩阵总和
# loss_current_node: 当前节点的损失
# monotonic_cst: 单调性约束，1为正单调性，-1为负单调性
# lower_bound: 分裂特征的下界
# upper_bound: 分裂特征的上界
# l2_regularization: L2正则化参数
# 返回值:
# 返回损失的减少量（gain）
cdef inline Y_DTYPE_C _split_gain(
        Y_DTYPE_C sum_gradient_left,
        Y_DTYPE_C sum_hessian_left,
        Y_DTYPE_C sum_gradient_right,
        Y_DTYPE_C sum_hessian_right,
        Y_DTYPE_C loss_current_node,
        signed char monotonic_cst,
        Y_DTYPE_C lower_bound,
        Y_DTYPE_C upper_bound,
        Y_DTYPE_C l2_regularization) noexcept nogil:
    """Loss reduction

    Compute the reduction in loss after taking a split, compared to keeping
    the node a leaf of the tree.

    See Equation 7 of:
    :arxiv:`T. Chen, C. Guestrin, (2016) XGBoost: A Scalable Tree Boosting System,
    <1603.02754>.`
    """
    cdef:
        Y_DTYPE_C gain
        Y_DTYPE_C value_left
        Y_DTYPE_C value_right

    # 计算潜在左右子节点的值
    value_left = compute_node_value(sum_gradient_left, sum_hessian_left,
                                    lower_bound, upper_bound,
                                    l2_regularization)
    value_right = compute_node_value(sum_gradient_right, sum_hessian_right,
                                     lower_bound, upper_bound,
                                     l2_regularization)

    # 如果不满足单调性约束，则不考虑该分裂
    if ((monotonic_cst == MonotonicConstraint.POS and value_left > value_right) or
            (monotonic_cst == MonotonicConstraint.NEG and value_left < value_right)):
        # 不考虑该分裂，因为它不符合单调性约束。
        # 需要注意，这些比较已经基于已经考虑单调性约束的值进行了修剪。
        return -1

    # 计算损失减少量
    gain = loss_current_node
    gain -= _loss_from_value(value_left, sum_gradient_left)
    gain -= _loss_from_value(value_right, sum_gradient_right)
    # 注意，为了正确计算损益（并使 min_gain_to_split 正常工作），所有值都必须是有界的
    # （当前节点、左子节点和右子节点）。

    return gain

# 计算节点从其（有界）值得到的损失
# 参数说明:
# value: 节点的值
# sum_gradient: 梯度总和
# 返回值:
# 返回节点从其值得到的损失
cdef inline Y_DTYPE_C _loss_from_value(
        Y_DTYPE_C value,
        Y_DTYPE_C sum_gradient) noexcept nogil:
    """Return loss of a node from its (bounded) value

    See Equation 6 of:
    :arxiv:`T. Chen, C. Guestrin, (2016) XGBoost: A Scalable Tree Boosting System,
    <1603.02754>.`
    """
    return sum_gradient * value

# 辅助函数，用于决定样本应该去左子节点还是右子节点
# 参数说明:
# missing_go_to_left: 如果缺失值应该进入左子节点，则为1，否则为0
# missing_values_bin_idx: 缺失值的二进制索引
# split_bin_idx: 分裂的二进制索引
# bin_value: 二进制值
# is_categorical: 是否为分类特征
# left_cat_bitset: 左子节点的分类位集合
# 返回值:
# 如果样本应该进入左子节点，则返回1，否则返回0
cdef inline unsigned char sample_goes_left(
        unsigned char missing_go_to_left,
        unsigned char missing_values_bin_idx,
        X_BINNED_DTYPE_C split_bin_idx,
        X_BINNED_DTYPE_C bin_value,
        unsigned char is_categorical,
        BITSET_DTYPE_C left_cat_bitset) noexcept nogil:
    """Helper to decide whether sample should go to left or right child."""

    if is_categorical:
        # 注意：如果有的话，缺失值被编码在 left_cat_bitset 中
        return in_bitset(left_cat_bitset, bin_value)
    else:
        return (
            (
                missing_go_to_left and  # 如果缺失值需要进入左子树，并且当前值等于缺失值的分箱索引
                bin_value == missing_values_bin_idx  # 则返回真
            )
            or (
                bin_value <= split_bin_idx  # 或者当前值小于等于分箱索引
            ))
# 定义一个内联函数 compute_node_value，计算节点的值
cpdef inline Y_DTYPE_C compute_node_value(
        Y_DTYPE_C sum_gradient,        # 总梯度和
        Y_DTYPE_C sum_hessian,         # 总黑塞矩阵和
        Y_DTYPE_C lower_bound,         # 下界
        Y_DTYPE_C upper_bound,         # 上界
        Y_DTYPE_C l2_regularization) noexcept nogil:
    """Compute a node's value.

    The value is capped in the [lower_bound, upper_bound] interval to respect
    monotonic constraints. Shrinkage is ignored.

    See Equation 5 of:
    :arxiv:`T. Chen, C. Guestrin, (2016) XGBoost: A Scalable Tree Boosting System,
    <1603.02754>.`
    """

    cdef:
        Y_DTYPE_C value               # 定义节点值变量

    # 计算节点值，忽略收缩项（shrinkage），确保节点值在指定的下界和上界之间
    value = -sum_gradient / (sum_hessian + l2_regularization + 1e-15)

    # 如果计算得到的节点值小于下界，则将其设为下界
    if value < lower_bound:
        value = lower_bound
    # 如果计算得到的节点值大于上界，则将其设为上界
    elif value > upper_bound:
        value = upper_bound

    # 返回最终计算得到的节点值
    return value
```