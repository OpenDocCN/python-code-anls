# `D:\src\scipysrc\scikit-learn\sklearn\tree\_criterion.pxd`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从 utils 模块中导入所需的类型定义
from ..utils._typedefs cimport float64_t, int8_t, intp_t

# 定义一个 Cython 类 Criterion，用于计算节点的不纯度和分裂后不纯度的减少，
# 同时计算回归中的均值和分类中的类概率等输出统计信息。
cdef class Criterion:
    # 内部结构

    # 常量定义：y 的值，即目标变量的值
    cdef const float64_t[:, ::1] y
    # 样本权重
    cdef const float64_t[:] sample_weight

    # 样本索引在 X 和 y 中的位置
    cdef const intp_t[:] sample_indices
    # 左子节点中的样本起始索引
    cdef intp_t start
    # 右子节点中的样本结束索引
    cdef intp_t pos
    # 总体样本结束索引
    cdef intp_t end
    # 正在评估的特征中缺失的值的数量
    cdef intp_t n_missing
    # 缺失值是否放到左子节点的标志

    # 输出的数量
    cdef intp_t n_outputs
    # 样本的数量
    cdef intp_t n_samples
    # 节点中的样本数量 (end-start)
    cdef intp_t n_node_samples
    # 加权样本数（总体）
    cdef float64_t weighted_n_samples
    # 加权节点样本数
    cdef float64_t weighted_n_node_samples
    # 左子节点中加权样本数
    cdef float64_t weighted_n_left
    # 右子节点中加权样本数
    cdef float64_t weighted_n_right
    # 缺失值的加权样本数

    # Criterion 对象的维护，左右收集的统计数据对应 samples[start:pos] 和 samples[pos:end]。

    # 方法

    # 初始化方法，设置 Criterion 对象的初始状态
    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end
    ) except -1 nogil

    # 初始化缺失值处理
    cdef void init_sum_missing(self)

    # 初始化缺失值，设置特征中的缺失值数量
    cdef void init_missing(self, intp_t n_missing) noexcept nogil

    # 重置 Criterion 对象的状态
    cdef int reset(self) except -1 nogil

    # 反向重置 Criterion 对象的状态
    cdef int reverse_reset(self) except -1 nogil

    # 更新 Criterion 对象的状态，根据新的位置 new_pos
    cdef int update(self, intp_t new_pos) except -1 nogil

    # 计算节点的不纯度
    cdef float64_t node_impurity(self) noexcept nogil

    # 计算左右子节点的不纯度
    cdef void children_impurity(
        self,
        float64_t* impurity_left,
        float64_t* impurity_right
    ) noexcept nogil

    # 计算节点的值
    cdef void node_value(
        self,
        float64_t* dest
    ) noexcept nogil

    # 将节点的值限制在指定的下界和上界之间
    cdef void clip_node_value(
        self,
        float64_t* dest,
        float64_t lower_bound,
        float64_t upper_bound
    ) noexcept nogil

    # 计算节点值的中间值
    cdef float64_t middle_value(self) noexcept nogil
    # 计算节点分裂前后不纯度的改善度
    cdef float64_t impurity_improvement(
        self,
        float64_t impurity_parent,
        float64_t impurity_left,
        float64_t impurity_right
    ) noexcept nogil

    # 计算代理节点分裂的不纯度改善度
    cdef float64_t proxy_impurity_improvement(self) noexcept nogil

    # 检查变量的单调性是否满足指定的上下界限制
    cdef bint check_monotonicity(
            self,
            int8_t monotonic_cst,
            float64_t lower_bound,
            float64_t upper_bound,
    ) noexcept nogil

    # 内部函数：检查变量的单调性是否满足指定的上下界限制，并计算左右子节点的总和
    cdef inline bint _check_monotonicity(
            self,
            int8_t monotonic_cst,
            float64_t lower_bound,
            float64_t upper_bound,
            float64_t sum_left,
            float64_t sum_right,
    ) noexcept nogil
# 定义一个 Cython 的类 ClassificationCriterion，继承自 Criterion，用于分类任务的抽象标准。
cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    # 使用 cdef 声明一个一维的整型数组 n_classes，表示类别的数量。
    cdef intp_t[::1] n_classes
    # 使用 cdef 声明一个整型变量 max_n_classes，表示最大类别数量。

    # 使用 cdef 声明一个二维的浮点数数组 sum_total，存储每个标签的加权计数的总和。
    cdef float64_t[:, ::1] sum_total    # The sum of the weighted count of each label.
    # 使用 cdef 声明一个二维的浮点数数组 sum_left，与 sum_total 类似，但用于分裂后左侧的数据。
    cdef float64_t[:, ::1] sum_left     # Same as above, but for the left side of the split
    # 使用 cdef 声明一个二维的浮点数数组 sum_right，与 sum_total 类似，但用于分裂后右侧的数据。
    cdef float64_t[:, ::1] sum_right    # Same as above, but for the right side of the split
    # 使用 cdef 声明一个二维的浮点数数组 sum_missing，与 sum_total 类似，但用于 X 中的缺失值。
    cdef float64_t[:, ::1] sum_missing  # Same as above, but for missing values in X

# 定义一个 Cython 的类 RegressionCriterion，继承自 Criterion，用于回归任务的抽象标准。
cdef class RegressionCriterion(Criterion):
    """Abstract regression criterion."""

    # 使用 cdef 声明一个浮点数变量 sq_sum_total，表示加权总和的平方。
    cdef float64_t sq_sum_total

    # 使用 cdef 声明一个一维的浮点数数组 sum_total，存储加权目标变量的总和。
    cdef float64_t[::1] sum_total    # The sum of w*y.
    # 使用 cdef 声明一个一维的浮点数数组 sum_left，与 sum_total 类似，但用于分裂后左侧的数据。
    cdef float64_t[::1] sum_left     # Same as above, but for the left side of the split
    # 使用 cdef 声明一个一维的浮点数数组 sum_right，与 sum_total 类似，但用于分裂后右侧的数据。
    cdef float64_t[::1] sum_right    # Same as above, but for the right side of the split
    # 使用 cdef 声明一个一维的浮点数数组 sum_missing，与 sum_total 类似，但用于 X 中的缺失值。
    cdef float64_t[::1] sum_missing  # Same as above, but for missing values in X
```