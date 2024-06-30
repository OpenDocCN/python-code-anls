# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\histogram.pyx`

```
# 导入Cython模块的声明，用于编写Cython扩展
cimport cython
# 导入Cython并行处理的prange函数
from cython.parallel import prange
# 导入C语言的memset函数
from libc.string cimport memset

# 导入NumPy库
import numpy as np

# 从当前包中导入common模块中定义的常量和结构体
from .common import HISTOGRAM_DTYPE
from .common cimport hist_struct
from .common cimport X_BINNED_DTYPE_C
from .common cimport G_H_DTYPE_C

# 注意事项的说明部分，详细介绍了代码中的视图和循环展开优化的使用情况

# 定义一个Cython类HistogramBuilder，用于构建直方图
@cython.final
cdef class HistogramBuilder:
    """A Histogram builder... used to build histograms.

    A histogram is an array with n_bins entries of type HISTOGRAM_DTYPE. Each
    feature has its own histogram. A histogram contains the sum of gradients
    and hessians of all the samples belonging to each bin.

    There are different ways to build a histogram:
    - by subtraction: hist(child) = hist(parent) - hist(sibling)
    - from scratch. In this case we have routines that update the hessians
      or not (not useful when hessians are constant for some losses e.g.
      least squares). Also, there's a special case for the root which
      contains all the samples, leading to some possible optimizations.
      Overall all the implementations look the same, and are optimized for
      cache hit.

    Parameters
    ----------
    X_binned : ndarray of int, shape (n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    n_bins : int
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
    gradients : ndarray, shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians : ndarray, shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians_are_constant : bool
        Whether hessians are constant.
    """
    # 定义公共成员变量，用于存储离散化后的特征矩阵 X_binned
    cdef public:
        const X_BINNED_DTYPE_C [::1, :] X_binned
        # 特征数量
        unsigned int n_features
        # 每个特征的离散化分箱数
        unsigned int n_bins
        # 梯度数组
        G_H_DTYPE_C [::1] gradients
        # 海森矩阵数组
        G_H_DTYPE_C [::1] hessians
        # 排序后的梯度数组
        G_H_DTYPE_C [::1] ordered_gradients
        # 排序后的海森矩阵数组
        G_H_DTYPE_C [::1] ordered_hessians
        # 海森矩阵是否常数
        unsigned char hessians_are_constant
        # 线程数
        int n_threads

    # 初始化方法，接受离散化后的特征矩阵 X_binned、分箱数、梯度和海森矩阵、海森矩阵是否常数、线程数作为参数
    def __init__(self, const X_BINNED_DTYPE_C [::1, :] X_binned,
                 unsigned int n_bins, G_H_DTYPE_C [::1] gradients,
                 G_H_DTYPE_C [::1] hessians,
                 unsigned char hessians_are_constant,
                 int n_threads):

        # 将离散化后的特征矩阵 X_binned 赋给对象的 X_binned 属性
        self.X_binned = X_binned
        # 计算特征的数量，即 X_binned 的列数
        self.n_features = X_binned.shape[1]
        # 所有直方图将具有相同的分箱数 n_bins，但如果某些特征的唯一值较少，则可能存在未使用的分箱
        self.n_bins = n_bins
        # 将梯度数组赋给对象的 gradients 属性
        self.gradients = gradients
        # 将海森矩阵数组赋给对象的 hessians 属性
        self.hessians = hessians
        # 对于根节点，梯度和海森矩阵已经是有序的，将其拷贝给对象的 ordered_gradients 属性
        self.ordered_gradients = gradients.copy()
        # 将海森矩阵拷贝给对象的 ordered_hessians 属性
        self.ordered_hessians = hessians.copy()
        # 标记海森矩阵是否为常数，赋值给对象的 hessians_are_constant 属性
        self.hessians_are_constant = hessians_are_constant
        # 设置对象的线程数属性
        self.n_threads = n_threads

    # 计算直方图的粗力方法，接受直方图构建器对象 self、样本索引数组和可选的允许特征数组作为参数
    def compute_histograms_brute(
        HistogramBuilder self,
        const unsigned int [::1] sample_indices,       # IN
        const unsigned int [:] allowed_features=None,  # IN
    cdef void _compute_histogram_brute_single_feature(
            HistogramBuilder self,
            const int feature_idx,
            const unsigned int [::1] sample_indices,  # IN: 存储要处理的样本索引数组
            hist_struct [:, ::1] histograms) noexcept nogil:  # OUT: 存储直方图数据的数组

        """Compute the histogram for a given feature."""
        # 计算给定特征的直方图

        cdef:
            unsigned int n_samples = sample_indices.shape[0]  # 样本数量
            const X_BINNED_DTYPE_C [::1] X_binned = \
                self.X_binned[:, feature_idx]  # 特征索引对应的离散化后的数据
            unsigned int root_node = X_binned.shape[0] == n_samples  # 判断是否为根节点
            G_H_DTYPE_C [::1] ordered_gradients = \
                self.ordered_gradients[:n_samples]  # 排序后的梯度数组
            G_H_DTYPE_C [::1] ordered_hessians = \
                self.ordered_hessians[:n_samples]  # 排序后的Hessian矩阵数组
            unsigned char hessians_are_constant = \
                self.hessians_are_constant  # 指示Hessian矩阵是否为常数

        # 将直方图数据初始化为零
        memset(&histograms[feature_idx, 0], 0, self.n_bins * sizeof(hist_struct))

        # 根据根节点和Hessian矩阵的情况选择构建直方图的方法
        if root_node:
            if hessians_are_constant:
                _build_histogram_root_no_hessian(feature_idx, X_binned,
                                                 ordered_gradients,
                                                 histograms)
            else:
                _build_histogram_root(feature_idx, X_binned,
                                      ordered_gradients, ordered_hessians,
                                      histograms)
        else:
            if hessians_are_constant:
                _build_histogram_no_hessian(feature_idx,
                                            sample_indices, X_binned,
                                            ordered_gradients, histograms)
            else:
                _build_histogram(feature_idx, sample_indices,
                                 X_binned, ordered_gradients,
                                 ordered_hessians, histograms)

    def compute_histograms_subtraction(
        HistogramBuilder self,
        hist_struct [:, ::1] parent_histograms,        # IN and OUT: 父节点直方图数据，同时也是输出
        hist_struct [:, ::1] sibling_histograms,       # IN: 兄弟节点直方图数据
        const unsigned int [:] allowed_features=None,  # IN: 允许处理的特征索引数组，可选参数
        """
        使用减法技巧计算节点的直方图。

        hist(parent) = hist(left_child) + hist(right_child)

        对于给定的特征，复杂度为 O(n_bins)。这比 compute_histograms_brute 更高效，
        但只适用于兄弟节点之一。

        参数
        ----------
        parent_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            父节点的直方图。
        sibling_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            兄弟节点的直方图。
        allowed_features : None 或 ndarray, dtype=np.uint32
            指示允许进行分裂的特征索引，由交互作用约束确定。

        返回
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape(n_features, n_bins)
            当前节点的计算直方图。
            我们重用 parent_histograms，无需分配新的内存。
        """

        cdef:
            int feature_idx
            int f_idx
            int n_allowed_features = self.n_features
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        # 计算每个特征的直方图
        for f_idx in prange(n_allowed_features, schedule='static', nogil=True,
                            num_threads=n_threads):
            if has_interaction_cst:
                feature_idx = allowed_features[f_idx]
            else:
                feature_idx = f_idx

            _subtract_histograms(
                feature_idx,
                self.n_bins,
                parent_histograms,
                sibling_histograms,
            )
        # 返回计算后的父节点直方图
        return parent_histograms
# 定义了一个 Cython 函数，用于在朴素方式下构建直方图，未优化以提高缓存命中率。
# 用于测试，与优化版本进行比较。
cpdef void _build_histogram_naive(
        const int feature_idx,  # 特征索引，表示要构建直方图的特征列
        unsigned int [:] sample_indices,  # 输入参数，样本索引数组
        X_BINNED_DTYPE_C [:] binned_feature,  # 输入参数，特征列的分箱后的值
        G_H_DTYPE_C [:] ordered_gradients,  # 输入参数，排序后的梯度值
        G_H_DTYPE_C [:] ordered_hessians,  # 输入参数，排序后的 Hessian 矩阵值
        hist_struct [:, :] out) noexcept nogil:  # 输出参数，用于存储直方图结果
    """Build histogram in a naive way, without optimizing for cache hit.

    Used in tests to compare with the optimized version."""
    cdef:
        unsigned int i  # 循环变量，迭代样本索引
        unsigned int n_samples = sample_indices.shape[0]  # 样本数量
        unsigned int sample_idx  # 单个样本索引
        unsigned int bin_idx  # 分箱后的特征值索引

    for i in range(n_samples):
        sample_idx = sample_indices[i]  # 获取当前样本索引
        bin_idx = binned_feature[sample_idx]  # 获取当前样本对应的分箱后的特征值索引
        # 将当前样本的梯度值和 Hessian 矩阵值累加到直方图中对应的箱子里
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx].count += 1


# 定义了一个 Cython 函数，用于计算直方图的差值，即 hist_a = hist_a - hist_b
cpdef void _subtract_histograms(
        const int feature_idx,  # 特征索引，表示要处理的特征列
        unsigned int n_bins,  # 分箱数目
        hist_struct [:, ::1] hist_a,  # 输入输出参数，第一个直方图
        hist_struct [:, ::1] hist_b,  # 输入参数，第二个直方图
) noexcept nogil:  # 输出参数，用于存储差值后的直方图结果
    """compute hist_a = hist_a - hist_b"""
    # 注意，在这里对大量浮点数进行减法操作可能会出现灾难性的取消效应。
    # 特别是对梯度而言，由于可能为正数或负数，而 Hessian 矩阵值为非负数。
    # 需要记住，梯度和 Hessian 矩阵值最初是在 float32 精度下计算的。
    # 因此，如果 sum_gradients 和 sum_hessians 使用了 float64，则不会失去精度。
    # 但如果我们也用 float32 进行求和，则需要注意浮点数误差。
    #
    # 可以通过设置保护负数 Hessian 矩阵值的方法：
    #     sum_hessians = max(0, sum_hessians)
    # 但由于我们使用 float64 对 float32 进行求和，这种情况非常不太可能发生。
    cdef:
        unsigned int i = 0  # 循环变量，迭代分箱索引
    for i in range(n_bins):
        # 对于每个分箱，将 hist_b 的梯度和 Hessian 矩阵值从 hist_a 对应的梯度和 Hessian 矩阵值中减去
        hist_a[feature_idx, i].sum_gradients -= hist_b[feature_idx, i].sum_gradients
        hist_a[feature_idx, i].sum_hessians -= hist_b[feature_idx, i].sum_hessians
        hist_a[feature_idx, i].count -= hist_b[feature_idx, i].count


# 定义了一个 Cython 函数，用于构建给定特征的直方图
cpdef void _build_histogram(
        const int feature_idx,  # 特征索引，表示要构建直方图的特征列
        const unsigned int [::1] sample_indices,  # 输入参数，样本索引数组
        const X_BINNED_DTYPE_C [::1] binned_feature,  # 输入参数，特征列的分箱后的值
        const G_H_DTYPE_C [::1] ordered_gradients,  # 输入参数，排序后的梯度值
        const G_H_DTYPE_C [::1] ordered_hessians,  # 输入参数，排序后的 Hessian 矩阵值
        hist_struct [:, ::1] out) noexcept nogil:  # 输出参数，用于存储直方图结果
    """Return histogram for a given feature."""
    cdef:
        unsigned int i = 0  # 循环变量，迭代样本索引
        unsigned int n_node_samples = sample_indices.shape[0]  # 样本数量
        unsigned int unrolled_upper = (n_node_samples // 4) * 4  # 循环展开的上界

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx  # 分箱后的特征值索引
    # 循环处理每四个样本索引，依次处理每个特征的相关数据
    for i in range(0, unrolled_upper, 4):
        # 获取当前四个样本索引对应的特征分箱值
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        # 累加有序梯度到输出中对应特征、分箱的梯度总和
        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        # 累加有序 Hessian 到输出中对应特征、分箱的 Hessian 总和
        out[feature_idx, bin_0].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_1].sum_hessians += ordered_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += ordered_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += ordered_hessians[i + 3]

        # 增加输出中对应特征、分箱的计数器
        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    # 处理余下的样本索引，更新对应特征、分箱的梯度总和、Hessian 总和及计数器
    for i in range(unrolled_upper, n_node_samples):
        # 获取当前样本索引对应的特征分箱值
        bin_idx = binned_feature[sample_indices[i]]
        # 累加有序梯度到输出中对应特征、分箱的梯度总和
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]
        # 累加有序 Hessian 到输出中对应特征、分箱的 Hessian 总和
        out[feature_idx, bin_idx].sum_hessians += ordered_hessians[i]
        # 增加输出中对应特征、分箱的计数器
        out[feature_idx, bin_idx].count += 1
# 构建不包含 Hessian 更新的直方图函数，用于计算给定特征的直方图
cpdef void _build_histogram_no_hessian(
        const int feature_idx,  # 特征索引
        const unsigned int [::1] sample_indices,  # 输入：样本索引数组
        const X_BINNED_DTYPE_C [::1] binned_feature,  # 输入：特征的分箱数据
        const G_H_DTYPE_C [::1] ordered_gradients,  # 输入：按顺序排列的梯度数据
        hist_struct [:, ::1] out) noexcept nogil:  # 输出：直方图结构数组

    """Return histogram for a given feature, not updating hessians.
    
    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0  # 循环计数器
        unsigned int n_node_samples = sample_indices.shape[0]  # 样本索引数组的长度
        unsigned int unrolled_upper = (n_node_samples // 4) * 4  # 可以进行4个样本一组的循环的上限

        unsigned int bin_0  # 第一个分箱的索引
        unsigned int bin_1  # 第二个分箱的索引
        unsigned int bin_2  # 第三个分箱的索引
        unsigned int bin_3  # 第四个分箱的索引
        unsigned int bin_idx  # 当前分箱的索引

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]  # 获取第一个样本的分箱索引
        bin_1 = binned_feature[sample_indices[i + 1]]  # 获取第二个样本的分箱索引
        bin_2 = binned_feature[sample_indices[i + 2]]  # 获取第三个样本的分箱索引
        bin_3 = binned_feature[sample_indices[i + 3]]  # 获取第四个样本的分箱索引

        # 将对应分箱的梯度值加到直方图中
        out[feature_idx, bin_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += ordered_gradients[i + 3]

        # 增加对应分箱的样本计数
        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    # 处理剩余不足一组的样本
    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]  # 获取当前样本的分箱索引
        out[feature_idx, bin_idx].sum_gradients += ordered_gradients[i]  # 加入梯度值到直方图
        out[feature_idx, bin_idx].count += 1  # 增加样本计数


# 计算根节点的直方图
cpdef void _build_histogram_root(
        const int feature_idx,  # 特征索引
        const X_BINNED_DTYPE_C [::1] binned_feature,  # 输入：特征的分箱数据
        const G_H_DTYPE_C [::1] all_gradients,  # 输入：所有样本的梯度数据
        const G_H_DTYPE_C [::1] all_hessians,  # 输入：所有样本的 Hessian 数据
        hist_struct [:, ::1] out) noexcept nogil:  # 输出：直方图结构数组

    """Compute histogram of the root node.

    Unlike other nodes, the root node has to find the split among *all* the
    samples from the training set. binned_feature and all_gradients /
    all_hessians already have a consistent ordering.
    """

    cdef:
        unsigned int i = 0  # 循环计数器
        unsigned int n_samples = binned_feature.shape[0]  # 样本数
        unsigned int unrolled_upper = (n_samples // 4) * 4  # 可以进行4个样本一组的循环的上限

        unsigned int bin_0  # 第一个分箱的索引
        unsigned int bin_1  # 第二个分箱的索引
        unsigned int bin_2  # 第三个分箱的索引
        unsigned int bin_3  # 第四个分箱的索引
        unsigned int bin_idx  # 当前分箱的索引
    # 遍历从0到unrolled_upper（不包含）的索引，步长为4
    for i in range(0, unrolled_upper, 4):
        # 分别获取当前索引及其后三个索引位置的分箱特征值
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        # 更新输出矩阵中对应特征索引和分箱索引的梯度总和
        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        # 更新输出矩阵中对应特征索引和分箱索引的黑塞矩阵总和
        out[feature_idx, bin_0].sum_hessians += all_hessians[i]
        out[feature_idx, bin_1].sum_hessians += all_hessians[i + 1]
        out[feature_idx, bin_2].sum_hessians += all_hessians[i + 2]
        out[feature_idx, bin_3].sum_hessians += all_hessians[i + 3]

        # 更新输出矩阵中对应特征索引和分箱索引的样本计数
        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    # 对于剩余的样本（索引从unrolled_upper到n_samples），更新输出矩阵中对应特征索引和分箱索引的统计量
    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx].sum_hessians += all_hessians[i]
        out[feature_idx, bin_idx].count += 1
# 定义一个 Cython 函数，计算不更新 Hessian 的根节点直方图
# 该函数不抛出异常，且在 GIL（全局解释器锁）未被获取时运行

cpdef void _build_histogram_root_no_hessian(
        const int feature_idx,  # 特征索引，输入参数
        const X_BINNED_DTYPE_C [::1] binned_feature,  # 输入参数，特征的离散化版本
        const G_H_DTYPE_C [::1] all_gradients,  # 输入参数，所有样本的梯度
        hist_struct [:, ::1] out) noexcept nogil:  # 输出参数，直方图结构体数组
    """Compute histogram of the root node, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0  # 循环索引变量
        unsigned int n_samples = binned_feature.shape[0]  # 样本数量
        unsigned int unrolled_upper = (n_samples // 4) * 4  # 可以进行向量化计算的样本数量上限

        unsigned int bin_0  # 第一个样本对应的离散化特征值索引
        unsigned int bin_1  # 第二个样本对应的离散化特征值索引
        unsigned int bin_2  # 第三个样本对应的离散化特征值索引
        unsigned int bin_3  # 第四个样本对应的离散化特征值索引
        unsigned int bin_idx  # 当前样本对应的离散化特征值索引

    # 循环处理每四个样本，进行直方图更新
    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        # 更新直方图对应的梯度总和
        out[feature_idx, bin_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3].sum_gradients += all_gradients[i + 3]

        # 增加每个离散化特征值对应的样本计数
        out[feature_idx, bin_0].count += 1
        out[feature_idx, bin_1].count += 1
        out[feature_idx, bin_2].count += 1
        out[feature_idx, bin_3].count += 1

    # 处理剩余的不足四个样本的情况
    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        # 更新直方图对应的梯度总和
        out[feature_idx, bin_idx].sum_gradients += all_gradients[i]
        # 增加每个离散化特征值对应的样本计数
        out[feature_idx, bin_idx].count += 1
```