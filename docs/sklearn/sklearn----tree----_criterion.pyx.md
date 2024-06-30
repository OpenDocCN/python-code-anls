# `D:\src\scipysrc\scikit-learn\sklearn\tree\_criterion.pyx`

```
# 从libc.string库中导入memcpy和memset函数，用于内存操作
# 从libc.math库中导入fabs和INFINITY常量，用于浮点数计算
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs, INFINITY

# 导入numpy库，并将其命名为np，用于数值计算
import numpy as np
# 导入cnp模块，并将其导入为cnp，用于与Cython结合使用numpy
cimport numpy as cnp
# 调用cnp的import_array()函数，初始化cnp数组接口
cnp.import_array()

# 从scipy.special.cython_special中导入xlogy函数，用于特殊数学计算
from scipy.special.cython_special cimport xlogy

# 从._utils模块中导入log和WeightedMedianCalculator类
from ._utils cimport log
from ._utils cimport WeightedMedianCalculator

# EPSILON常量用于Poisson标准
# 使用np.finfo('double').eps获取双精度浮点数的最小精度，乘以10作为EPSILON值
cdef float64_t EPSILON = 10 * np.finfo('double').eps

# Criterion类定义
cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    # __getstate__()方法，返回空字典，用于对象状态的序列化
    def __getstate__(self):
        return {}

    # __setstate__()方法，接受字典d，用于对象状态的反序列化
    def __setstate__(self, d):
        pass

    # init()方法，用于初始化标准
    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=float64_t
            y is a buffer that can store values for n_outputs target variables
            stored as a Cython memoryview.
        sample_weight : ndarray, dtype=float64_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : float64_t
            The total weight of the samples being considered
        sample_indices : ndarray, dtype=intp_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : intp_t
            The first sample to be used on this node
        end : intp_t
            The last sample used on this node

        """
        pass

    # init_missing()方法，初始化缺失值相关的操作
    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]

        Parameters
        ----------
        n_missing: intp_t
            Number of missing values for specific feature.
        """
        pass

    # reset()方法，重置标准，从位置start开始
    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """
        pass

    # reverse_reset()方法，反向重置标准，从位置end开始
    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass
    # 定义一个 Cython 的函数，用于更新统计信息，移动 sample_indices[pos:new_pos] 到左子树
    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        This updates the collected statistics by moving sample_indices[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : intp_t
            New starting index position of the sample_indices in the right child
        """
        pass

    # 定义一个 Cython 的函数，用于计算节点的不纯度指标
    cdef float64_t node_impurity(self) noexcept nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of sample_indices[start:end]. This is the
        primary function of the criterion class. The smaller the impurity the
        better.
        """
        pass

    # 定义一个 Cython 的函数，用于计算子节点的不纯度指标
    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of sample_indices[start:pos] + the impurity
        of sample_indices[pos:end].

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : float64_t pointer
            The memory address where the impurity of the right child should be
            stored
        """
        pass

    # 定义一个 Cython 的函数，用于计算节点值并保存到指定的内存地址
    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of sample_indices[start:end] and save the value into dest.

        Parameters
        ----------
        dest : float64_t pointer
            The memory address where the node value should be stored.
        """
        pass

    # 定义一个 Cython 的函数，用于将节点值限制在指定的下界和上界之间
    cdef void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        pass

    # 定义一个 Cython 的函数，用于计算分裂的中间值，以满足单调性约束
    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints

        This method is implemented in ClassificationCriterion and RegressionCriterion.
        """
        pass
    # 计算一个代理的不纯度减少量
    # 这个方法用于加速查找最佳分裂点
    # 它是一个代理量，最大化这个值的分裂点也会最大化不纯度的改善
    # 它忽略了给定分裂点的所有常数项的不纯度减少
    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        # 调用方法计算左右子节点的不纯度
        self.children_impurity(&impurity_left, &impurity_right)

        # 返回计算得到的代理不纯度改善量
        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    # 计算不纯度的改善量
    # 这个方法计算分裂时不纯度的改善
    # 权重不纯度改善方程如下：
    # N_t / N * (impurity - N_t_R / N_t * right_impurity
    #                    - N_t_L / N_t * left_impurity)
    # 其中 N 是总样本数，N_t 是当前节点的样本数，N_t_L 是左子节点的样本数，
    # N_t_R 是右子节点的样本数
    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right /
                                    self.weighted_n_node_samples * impurity_right)
                                 - (self.weighted_n_left /
                                    self.weighted_n_node_samples * impurity_left)))

    # 检查单调性
    # 这个方法用于检查节点值的单调性
    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        pass

    # 内联方法：检查单调性
    # 这个方法用于内联检查节点值的单调性
    cdef inline bint _check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
        float64_t value_left,
        float64_t value_right,
    ) noexcept nogil:
        pass
    ) noexcept nogil:
        cdef:
            # 检查左右值是否均大于或等于下界
            bint check_lower_bound = (
                (value_left >= lower_bound) &
                (value_right >= lower_bound)
            )
            # 检查左右值是否均小于或等于上界
            bint check_upper_bound = (
                (value_left <= upper_bound) &
                (value_right <= upper_bound)
            )
            # 检查左右值之差乘以单调常数是否小于等于零，即检查是否单调
            bint check_monotonic_cst = (
                (value_left - value_right) * monotonic_cst <= 0
            )
        # 返回三个条件的逻辑与结果
        return check_lower_bound & check_upper_bound & check_monotonic_cst

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
# 定义一个内联函数 _move_sums_classification，用于按照指定的标准将总和和缺失值分布到 sum_1 和 sum_2 中。

# 如果存在缺失值，并且 put_missing_in_1 为 True，则将缺失值分配到 sum_1 中：
# 具体操作是将 sum_1 设置为 sum_missing，将 sum_2 设置为 sum_total 减去 sum_missing。

# 如果 put_missing_in_1 为 False，则将缺失值分配到 sum_2 中：
# 具体操作是将 sum_1 设置为全零，将 sum_2 设置为 sum_total。

cdef inline void _move_sums_classification(
    ClassificationCriterion criterion,
    float64_t[:, ::1] sum_1,
    float64_t[:, ::1] sum_2,
    float64_t* weighted_n_1,
    float64_t* weighted_n_2,
    bint put_missing_in_1,
) noexcept nogil:
    """Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values to go sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    cdef intp_t k, c, n_bytes
    
    # 如果存在缺失值并且 put_missing_in_1 为 True，则执行以下操作：
    if criterion.n_missing != 0 and put_missing_in_1:
        # 对于每个输出 k，将 sum_missing 的数据复制到 sum_1 中
        for k in range(criterion.n_outputs):
            n_bytes = criterion.n_classes[k] * sizeof(float64_t)
            memcpy(&sum_1[k, 0], &criterion.sum_missing[k, 0], n_bytes)

        # 对于每个输出 k 和每个类别 c，计算 sum_2 为 sum_total 减去 sum_missing
        for k in range(criterion.n_outputs):
            for c in range(criterion.n_classes[k]):
                sum_2[k, c] = criterion.sum_total[k, c] - criterion.sum_missing[k, c]

        # 将 weighted_n_1 和 weighted_n_2 分别设置为权重下的缺失样本数和非缺失样本数
        weighted_n_1[0] = criterion.weighted_n_missing
        weighted_n_2[0] = criterion.weighted_n_node_samples - criterion.weighted_n_missing
    else:
        # 如果不存在缺失值或者 put_missing_in_1 为 False，则执行以下操作：
        
        # 将 sum_1 设置为全零
        for k in range(criterion.n_outputs):
            n_bytes = criterion.n_classes[k] * sizeof(float64_t)
            memset(&sum_1[k, 0], 0, n_bytes)
        
        # 将 sum_2 设置为 sum_total 的值
        memcpy(&sum_2[k, 0], &criterion.sum_total[k, 0], n_bytes)

        # 将 weighted_n_1 设置为 0，将 weighted_n_2 设置为权重下的节点样本数
        weighted_n_1[0] = 0.0
        weighted_n_2[0] = criterion.weighted_n_node_samples


cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""
    # 定义类的初始化方法，接受输出数量和每个输出的类别数量数组作为参数
    def __cinit__(self, intp_t n_outputs,
                  cnp.ndarray[intp_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=intp_t
            The number of unique classes in each target
        """
        # 初始化起始、当前位置和结束位置为零
        self.start = 0
        self.pos = 0
        self.end = 0
        self.missing_go_to_left = 0

        # 设置输出数量和样本数量等属性
        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        # 创建一个空的整数数组以存储每个输出的类别数量
        self.n_classes = np.empty(n_outputs, dtype=np.intp)

        # 初始化循环计数器和最大类别数量
        cdef intp_t k = 0
        cdef intp_t max_n_classes = 0

        # 遍历每个输出，设置其对应的类别数量，并计算所有输出中最大的类别数量
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            # 更新最大类别数量
            if n_classes[k] > max_n_classes:
                max_n_classes = n_classes[k]

        # 存储最大类别数量
        self.max_n_classes = max_n_classes

        # 初始化用于存储总和的数组，用于每个输出和每个类别
        self.sum_total = np.zeros((n_outputs, max_n_classes), dtype=np.float64)
        self.sum_left = np.zeros((n_outputs, max_n_classes), dtype=np.float64)
        self.sum_right = np.zeros((n_outputs, max_n_classes), dtype=np.float64)

    # 定义用于序列化的方法，返回一个元组，包括类型、输出数量和类别数量数组的序列化状态
    def __reduce__(self):
        return (type(self),
                (self.n_outputs, np.asarray(self.n_classes)), self.__getstate__())

    # 定义初始化方法，接受训练数据、样本权重、加权样本数量、样本索引、起始和结束位置作为参数
    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end
    ) except -1 nogil:
        """
        Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=float64_t
            The target stored as a buffer for memory efficiency.
        sample_weight : ndarray, dtype=float64_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : float64_t
            The total weight of all samples
        sample_indices : ndarray, dtype=intp_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : intp_t
            The first sample to use in the mask
        end : intp_t
            The last sample to use in the mask
        """
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef intp_t i        # 定义整数变量 i
        cdef intp_t p        # 定义整数变量 p
        cdef intp_t k        # 定义整数变量 k
        cdef intp_t c        # 定义整数变量 c
        cdef float64_t w = 1.0   # 定义浮点数变量 w，并初始化为 1.0

        # 针对每个输出维度，将 self.sum_total[k, 0] 到 self.sum_total[k, n_classes[k]-1] 的内存清零
        for k in range(self.n_outputs):
            memset(&self.sum_total[k, 0], 0, self.n_classes[k] * sizeof(float64_t))

        # 遍历从 start 到 end 的每个样本索引
        for p in range(start, end):
            i = sample_indices[p]

            # 如果提供了样本权重 sample_weight，则更新 w 为样本 i 的权重值
            if sample_weight is not None:
                w = sample_weight[i]

            # 对于每个输出维度 k，将样本 i 的目标值 y[i, k] 转换为整数 c，并将权重 w 加到 self.sum_total[k, c] 上
            for k in range(self.n_outputs):
                c = <intp_t> self.y[i, k]
                self.sum_total[k, c] += w

            # 累加权重 w 到 self.weighted_n_node_samples
            self.weighted_n_node_samples += w

        # 调用 self.reset() 方法重置对象状态到初始位置
        self.reset()
        return 0

    cdef void init_sum_missing(self):
        """
        Init sum_missing to hold sums for missing values.
        
        初始化 sum_missing 用于保存缺失值的和。
        """
        self.sum_missing = np.zeros((self.n_outputs, self.max_n_classes), dtype=np.float64)
    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        # 初始化 sum_missing 如果存在缺失值

        cdef intp_t i, p, k, c
        cdef float64_t w = 1.0

        self.n_missing = n_missing
        # 设置对象的 n_missing 属性为传入的 n_missing 值
        if n_missing == 0:
            return
        # 如果 n_missing 为 0，则直接返回，不进行后续操作

        # 使用 memset 函数将 self.sum_missing 的部分内存初始化为 0
        memset(&self.sum_missing[0, 0], 0, self.max_n_classes * self.n_outputs * sizeof(float64_t))

        self.weighted_n_missing = 0.0
        # 将 weighted_n_missing 初始化为 0.0

        # 缺失的样本被假定在 self.sample_indices[-n_missing:] 中
        # 对于处于末尾 self.end - n_missing 到 self.end 位置的 sample_indices 的样本
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            # 获取样本索引 i

            if self.sample_weight is not None:
                w = self.sample_weight[i]
                # 如果样本权重数组 sample_weight 不为空，则获取第 i 个样本的权重值

            # 遍历每个输出和类别，更新 sum_missing 中的值
            for k in range(self.n_outputs):
                c = <intp_t> self.y[i, k]
                # 获取目标变量 y 中第 i 行、第 k 列的值，并转换为 intp_t 类型

                self.sum_missing[k, c] += w
                # 将权重 w 加到 sum_missing 的相应位置上

            self.weighted_n_missing += w
            # 将权重 w 加到 weighted_n_missing 上

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 将标准重新设置到位置 start 处

        self.pos = self.start
        # 将对象的 pos 属性设置为 start

        _move_sums_classification(
            self,
            self.sum_left,
            self.sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left,
        )
        # 调用 _move_sums_classification 函数，重置分类标准的汇总值

        return 0
        # 返回 0 表示成功重置

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 将标准反向重置到位置 end 处

        self.pos = self.end
        # 将对象的 pos 属性设置为 end

        _move_sums_classification(
            self,
            self.sum_right,
            self.sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left
        )
        # 调用 _move_sums_classification 函数，反向重置分类标准的汇总值

        return 0
        # 返回 0 表示成功反向重置
    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : intp_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        # Retrieve current position from self.pos
        cdef intp_t pos = self.pos

        # Calculate the end position of non-missing samples
        cdef intp_t end_non_missing = self.end - self.n_missing

        # Reference to self.sample_indices and self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef const float64_t[:] sample_weight = self.sample_weight

        # Declare variables for iteration and computation
        cdef intp_t i  # Index variable for sample indices
        cdef intp_t p  # Loop variable for iterating over indices
        cdef intp_t k  # Loop variable for iterating over outputs
        cdef intp_t c  # Loop variable for iterating over classes
        cdef float64_t w = 1.0  # Weight variable initialized to 1.0

        # Update statistics up to new_pos based on the smaller segment
        #
        # Given that
        #   sum_left[x] + sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that requires the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                # If sample weights are provided, update weight
                if sample_weight is not None:
                    w = sample_weight[i]

                # Update sum_left for each output and corresponding class
                for k in range(self.n_outputs):
                    self.sum_left[k, <intp_t> self.y[i, k]] += w

                # Accumulate weighted samples for the left child
                self.weighted_n_left += w

        else:
            # Reverse reset operation if updating from end to new_pos
            self.reverse_reset()

            # Update sum_left for the range from end_non_missing to new_pos in reverse
            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                # If sample weights are provided, update weight
                if sample_weight is not None:
                    w = sample_weight[i]

                # Subtract weights from sum_left for each output and corresponding class
                for k in range(self.n_outputs):
                    self.sum_left[k, <intp_t> self.y[i, k]] -= w

                # Decrement weighted samples for the left child
                self.weighted_n_left -= w

        # Update statistics for the right child
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                # Calculate sum_right based on the updated sum_left and sum_total
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]

        # Update the position marker to new_pos
        self.pos = new_pos

        # Return 0 to indicate successful completion of the update
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        pass

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        pass
    # 计算样本索引范围内节点值，并将结果保存到目标数组 dest 中
    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] and save it into dest.

        Parameters
        ----------
        dest : float64_t pointer
            The memory address which we will save the node value into.
        """
        cdef intp_t k, c
        
        # 遍历输出节点和类别，计算节点值并存入 dest 数组
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                dest[c] = self.sum_total[k, c] / self.weighted_n_node_samples
            # 更新 dest 指针位置，移动到下一个输出节点
            dest += self.max_n_classes

    # 将目标数组中的值裁剪到指定的下限和上限之间，确保预测概率在指定边界内，并考虑单调性约束
    cdef inline void clip_node_value(
        self, float64_t * dest, float64_t lower_bound, float64_t upper_bound
    ) noexcept nogil:
        """Clip the values in dest such that predicted probabilities stay between
        `lower_bound` and `upper_bound` when monotonic constraints are enforced.
        Note that monotonicity constraints are only supported for:
        - single-output trees and
        - binary classifications.
        """
        # 如果目标值小于下限，则将其设置为下限值
        if dest[0] < lower_bound:
            dest[0] = lower_bound
        # 如果目标值大于上限，则将其设置为上限值
        elif dest[0] > upper_bound:
            dest[0] = upper_bound
        
        # 对于二分类问题，确保两个类别的概率值之和为 1
        dest[1] = 1 - dest[0]

    # 计算分裂的中间值，用于单调性约束，计算方法是左右子节点值的简单平均
    cdef inline float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Note that monotonicity constraints are only supported for:
        - single-output trees and
        - binary classifications.
        """
        return (
            (self.sum_left[0, 0] / (2 * self.weighted_n_left)) +
            (self.sum_right[0, 0] / (2 * self.weighted_n_right))
        )

    # 检查当前分类分裂点是否满足单调性约束
    cdef inline bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current classification split"""
        cdef:
            float64_t value_left = self.sum_left[0][0] / self.weighted_n_left
            float64_t value_right = self.sum_right[0][0] / self.weighted_n_right
        
        # 调用内部方法检查单调性约束
        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)
# 定义一个 C 扩展类 Entropy，继承自 ClassificationCriterion
cdef class Entropy(ClassificationCriterion):
    r"""Cross Entropy impurity criterion.

    这是交叉熵不纯度准则。

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    # 定义一个返回浮点数的方法，计算当前节点的不纯度
    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        评估当前节点的不纯度，即 sample_indices[start:end] 的不纯度。不纯度越小越好。
        """
        # 初始化 entropy 变量为 0.0
        cdef float64_t entropy = 0.0
        # 声明 count_k 和索引变量 k, c
        cdef float64_t count_k
        cdef intp_t k
        cdef intp_t c

        # 遍历输出类别数目的范围
        for k in range(self.n_outputs):
            # 遍历每个类别内部的类别数目
            for c in range(self.n_classes[k]):
                # 获取当前类别和输出类别的总和
                count_k = self.sum_total[k, c]
                # 如果 count_k 大于 0.0
                if count_k > 0.0:
                    # 将 count_k 除以加权节点样本
                    count_k /= self.weighted_n_node_samples
                    # 计算 entropy
                    entropy -= count_k * log(count_k)

        # 返回平均 entropy 除以输出类别数目
        return entropy / self.n_outputs

    # 定义一个返回空值的方法，计算左右孩子节点的不纯度
    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        评估子节点的不纯度，即左子节点 (sample_indices[start:pos]) 和右子节点 (sample_indices[pos:end]) 的不纯度。

        Parameters
        ----------
        impurity_left : float64_t pointer
            保存左节点不纯度的内存地址
        impurity_right : float64_t pointer
            保存右节点不纯度的内存地址
        """
        # 初始化左右节点的 entropy 为 0.0
        cdef float64_t entropy_left = 0.0
        cdef float64_t entropy_right = 0.0
        # 声明 count_k 和索引变量 k, c
        cdef float64_t count_k
        cdef intp_t k
        cdef intp_t c

        # 遍历输出类别数目的范围
        for k in range(self.n_outputs):
            # 遍历每个类别内部的类别数目
            for c in range(self.n_classes[k]):
                # 获取左子节点的总和
                count_k = self.sum_left[k, c]
                # 如果 count_k 大于 0.0
                if count_k > 0.0:
                    # 将 count_k 除以加权左节点样本
                    count_k /= self.weighted_n_left
                    # 计算左子节点的 entropy
                    entropy_left -= count_k * log(count_k)

                # 获取右子节点的总和
                count_k = self.sum_right[k, c]
                # 如果 count_k 大于 0.0
                if count_k > 0.0:
                    # 将 count_k 除以加权右节点样本
                    count_k /= self.weighted_n_right
                    # 计算右子节点的 entropy
                    entropy_right -= count_k * log(count_k)

        # 将左右子节点的平均 entropy 存储在 impurity_left 和 impurity_right 中
        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs
    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """



# 定义基尼系数

cdef float64_t node_impurity(self) noexcept nogil:
    """Evaluate the impurity of the current node.

    Evaluate the Gini criterion as impurity of the current node,
    i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
    better.
    """
    # 初始化基尼系数为0
    cdef float64_t gini = 0.0
    cdef float64_t sq_count
    cdef float64_t count_k
    cdef intp_t k
    cdef intp_t c

    # 遍历每个输出（可能的类别）
    for k in range(self.n_outputs):
        sq_count = 0.0

        # 遍历当前输出的每个类别
        for c in range(self.n_classes[k]):
            # 获取当前类别的样本数量
            count_k = self.sum_total[k, c]
            # 计算该类别样本数量的平方和
            sq_count += count_k * count_k

        # 计算基尼系数的一部分，并加到总的基尼系数上
        gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                  self.weighted_n_node_samples)

    # 返回平均每个输出的基尼系数
    return gini / self.n_outputs

cdef void children_impurity(self, float64_t* impurity_left,
                            float64_t* impurity_right) noexcept nogil:
    """Evaluate the impurity in children nodes.

    i.e. the impurity of the left child (sample_indices[start:pos]) and the
    impurity the right child (sample_indices[pos:end]) using the Gini index.

    Parameters
    ----------
    impurity_left : float64_t pointer
        The memory address to save the impurity of the left node to
    impurity_right : float64_t pointer
        The memory address to save the impurity of the right node to
    """
    # 初始化左右子节点的基尼系数为0
    cdef float64_t gini_left = 0.0
    cdef float64_t gini_right = 0.0
    cdef float64_t sq_count_left
    cdef float64_t sq_count_right
    cdef float64_t count_k
    cdef intp_t k
    cdef intp_t c

    # 遍历每个输出（可能的类别）
    for k in range(self.n_outputs):
        sq_count_left = 0.0
        sq_count_right = 0.0

        # 遍历当前输出的每个类别
        for c in range(self.n_classes[k]):
            # 获取左子节点和右子节点中当前类别的样本数量
            count_k = self.sum_left[k, c]
            sq_count_left += count_k * count_k

            count_k = self.sum_right[k, c]
            sq_count_right += count_k * count_k

        # 计算左子节点和右子节点的基尼系数的一部分，并加到总的基尼系数上
        gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                            self.weighted_n_left)

        gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                              self.weighted_n_right)

    # 将计算得到的左子节点和右子节点的基尼系数写入指定的内存地址中
    impurity_left[0] = gini_left / self.n_outputs
    impurity_right[0] = gini_right / self.n_outputs
cdef inline void _move_sums_regression(
    RegressionCriterion criterion,
    float64_t[::1] sum_1,
    float64_t[::1] sum_2,
    float64_t* weighted_n_1,
    float64_t* weighted_n_2,
    bint put_missing_in_1,
) noexcept nogil:
    """Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values to go sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    cdef:
        intp_t i                   # 循环变量
        intp_t n_bytes = criterion.n_outputs * sizeof(float64_t)  # 计算数组大小的字节数
        bint has_missing = criterion.n_missing != 0  # 是否存在缺失值的标志

    if has_missing and put_missing_in_1:
        memcpy(&sum_1[0], &criterion.sum_missing[0], n_bytes)
        for i in range(criterion.n_outputs):
            sum_2[i] = criterion.sum_total[i] - criterion.sum_missing[i]
        weighted_n_1[0] = criterion.weighted_n_missing  # 加权样本数，包含缺失值的部分
        weighted_n_2[0] = criterion.weighted_n_node_samples - criterion.weighted_n_missing  # 加权样本数，不含缺失值的部分
    else:
        memset(&sum_1[0], 0, n_bytes)  # 将 sum_1 数组清零
        # 将 sum_2 数组赋值为 sum_total 数组的内容
        memcpy(&sum_2[0], &criterion.sum_total[0], n_bytes)
        weighted_n_1[0] = 0.0  # 加权样本数，没有缺失值的部分
        weighted_n_2[0] = criterion.weighted_n_node_samples  # 加权样本数，总体的部分


cdef class RegressionCriterion(Criterion):
    r"""Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, intp_t n_outputs, intp_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets to be predicted

        n_samples : intp_t
            The total number of samples to fit on
        """
        # Default values
        self.start = 0  # 起始位置
        self.pos = 0  # 当前位置
        self.end = 0  # 结束位置

        self.n_outputs = n_outputs  # 目标的数量
        self.n_samples = n_samples  # 要拟合的总样本数
        self.n_node_samples = 0  # 节点样本数
        self.weighted_n_node_samples = 0.0  # 加权节点样本数
        self.weighted_n_left = 0.0  # 左侧加权样本数
        self.weighted_n_right = 0.0  # 右侧加权样本数
        self.weighted_n_missing = 0.0  # 缺失值的加权样本数

        self.sq_sum_total = 0.0  # 总体平方和

        self.sum_total = np.zeros(n_outputs, dtype=np.float64)  # 总体和的数组
        self.sum_left = np.zeros(n_outputs, dtype=np.float64)   # 左侧和的数组
        self.sum_right = np.zeros(n_outputs, dtype=np.float64)  # 右侧和的数组

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())
    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        # Initialize fields
        self.y = y  # 设置类成员变量 y，用于存储传入的 y 参数
        self.sample_weight = sample_weight  # 设置类成员变量 sample_weight，用于存储传入的 sample_weight 参数
        self.sample_indices = sample_indices  # 设置类成员变量 sample_indices，用于存储传入的 sample_indices 参数
        self.start = start  # 设置类成员变量 start，用于存储传入的 start 参数
        self.end = end  # 设置类成员变量 end，用于存储传入的 end 参数
        self.n_node_samples = end - start  # 计算节点样本数并存储在类成员变量 n_node_samples 中
        self.weighted_n_samples = weighted_n_samples  # 设置类成员变量 weighted_n_samples，用于存储传入的 weighted_n_samples 参数
        self.weighted_n_node_samples = 0.  # 初始化节点加权样本数为 0.0

        cdef intp_t i  # 声明 C 语言级别的整型变量 i
        cdef intp_t p  # 声明 C 语言级别的整型变量 p
        cdef intp_t k  # 声明 C 语言级别的整型变量 k
        cdef float64_t y_ik  # 声明 C 语言级别的双精度浮点数变量 y_ik
        cdef float64_t w_y_ik  # 声明 C 语言级别的双精度浮点数变量 w_y_ik
        cdef float64_t w = 1.0  # 声明 C 语言级别的双精度浮点数变量 w，并初始化为 1.0
        self.sq_sum_total = 0.0  # 初始化 sq_sum_total 类成员变量为 0.0
        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(float64_t))  # 使用 memset 函数将 self.sum_total 数组初始化为 0

        for p in range(start, end):  # 遍历从 start 到 end 的 sample_indices
            i = sample_indices[p]  # 获取当前索引对应的样本索引 i

            if sample_weight is not None:  # 如果 sample_weight 不为空
                w = sample_weight[i]  # 获取当前样本的权重值 w

            for k in range(self.n_outputs):  # 遍历输出维度
                y_ik = self.y[i, k]  # 获取样本 i 在输出维度 k 上的值
                w_y_ik = w * y_ik  # 计算加权值 w_y_ik
                self.sum_total[k] += w_y_ik  # 累加到 sum_total 数组的对应位置上
                self.sq_sum_total += w_y_ik * y_ik  # 更新平方和的累加值

            self.weighted_n_node_samples += w  # 更新节点加权样本数

        # Reset to pos=start
        self.reset()  # 调用类方法 reset()，重置状态
        return 0  # 返回 0 表示初始化成功

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.sum_missing = np.zeros(self.n_outputs, dtype=np.float64)  # 初始化 sum_missing 数组为 n_outputs 维度的零数组，用于存储缺失值的和

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        cdef intp_t i, p, k  # 声明 C 语言级别的整型变量 i, p, k
        cdef float64_t y_ik  # 声明 C 语言级别的双精度浮点数变量 y_ik
        cdef float64_t w_y_ik  # 声明 C 语言级别的双精度浮点数变量 w_y_ik
        cdef float64_t w = 1.0  # 声明 C 语言级别的双精度浮点数变量 w，并初始化为 1.0

        self.n_missing = n_missing  # 设置类成员变量 n_missing，用于存储传入的 n_missing 参数
        if n_missing == 0:  # 如果没有缺失值，则直接返回
            return

        memset(&self.sum_missing[0], 0, self.n_outputs * sizeof(float64_t))  # 使用 memset 函数将 sum_missing 数组初始化为 0

        self.weighted_n_missing = 0.0  # 初始化加权缺失值数量为 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        # 假设缺失样本位于 self.sample_indices[-n_missing:] 中
        for p in range(self.end - n_missing, self.end):  # 遍历最后 n_missing 个 sample_indices
            i = self.sample_indices[p]  # 获取当前索引对应的样本索引 i
            if self.sample_weight is not None:  # 如果 sample_weight 不为空
                w = self.sample_weight[i]  # 获取当前样本的权重值 w

            for k in range(self.n_outputs):  # 遍历输出维度
                y_ik = self.y[i, k]  # 获取样本 i 在输出维度 k 上的值
                w_y_ik = w * y_ik  # 计算加权值 w_y_ik
                self.sum_missing[k] += w_y_ik  # 累加到 sum_missing 数组的对应位置上

            self.weighted_n_missing += w  # 更新加权缺失值数量
    # 定义一个Cython函数，重置当前位置为起始位置
    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        # 将当前位置重置为起始位置
        self.pos = self.start
        # 调用底层C函数，重新计算左右子节点的统计信息
        _move_sums_regression(
            self,
            self.sum_left,
            self.sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left
        )
        # 返回0表示成功重置
        return 0

    # 定义一个Cython函数，将当前位置反向重置为结束位置
    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        # 将当前位置反向重置为结束位置
        self.pos = self.end
        # 调用底层C函数，反向重新计算左右子节点的统计信息
        _move_sums_regression(
            self,
            self.sum_right,
            self.sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left
        )
        # 返回0表示成功反向重置
        return 0

    # 定义一个Cython函数，根据新的位置更新统计信息
    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        # 获取样本权重和样本索引的视图
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        # 获取当前位置
        cdef intp_t pos = self.pos

        # 计算非缺失样本的结束位置
        # 缺失的样本假设位于self.sample_indices[-self.n_missing:]，即self.sample_indices[end_non_missing:self.end]。
        cdef intp_t end_non_missing = self.end - self.n_missing
        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0

        # 根据新位置更新统计信息
        #
        # 根据以下关系：
        #           sum_left[x] + sum_right[x] = sum_total[x]
        # 并已知sum_total，我们将从需要计算量最小的方向更新sum_left，即从pos到new_pos或从end到new_pos。
        if (new_pos - pos) <= (end_non_missing - new_pos):
            # 从pos到new_pos逐步更新sum_left和weighted_n_left
            for p in range(pos, new_pos):
                i = sample_indices[p]

                # 如果样本权重不为空，则考虑样本权重
                if sample_weight is not None:
                    w = sample_weight[i]

                # 对于每个输出维度，更新sum_left
                for k in range(self.n_outputs):
                    self.sum_left[k] += w * self.y[i, k]

                # 更新左子节点的加权样本数
                self.weighted_n_left += w
        else:
            # 否则，调用反向重置方法
            self.reverse_reset()

            # 从end_non_missing到new_pos的反向更新sum_left和weighted_n_left
            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                # 如果样本权重不为空，则考虑样本权重
                if sample_weight is not None:
                    w = sample_weight[i]

                # 对于每个输出维度，更新sum_left
                for k in range(self.n_outputs):
                    self.sum_left[k] -= w * self.y[i, k]

                # 更新左子节点的加权样本数
                self.weighted_n_left -= w

        # 更新右子节点的加权样本数
        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)

        # 对于每个输出维度，计算sum_right
        for k in range(self.n_outputs):
            self.sum_right[k] = self.sum_total[k] - self.sum_left[k]

        # 将当前位置更新为new_pos
        self.pos = new_pos

        # 返回0表示成功更新统计信息
        return 0

    # 定义一个Cython函数，计算节点的不纯度指标
    cdef float64_t node_impurity(self) noexcept nogil:
        pass

    # 定义一个Cython函数，计算子节点的不纯度指标
    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        pass
    # 定义一个 C 语言扩展函数，计算节点值，将计算结果存储到 dest 中
    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        # 声明一个整型变量 k，用于迭代 self.n_outputs 次
        cdef intp_t k

        # 循环计算每个输出的节点值
        for k in range(self.n_outputs):
            # 将节点总和除以加权节点样本数，存储到 dest 中的第 k 个位置
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples

    # 定义一个内联函数，用于将 dest 中的值限制在 lower_bound 和 upper_bound 之间，用于单调性约束
    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""
        # 如果 dest[0] 的值小于 lower_bound，则将其设为 lower_bound
        if dest[0] < lower_bound:
            dest[0] = lower_bound
        # 如果 dest[0] 的值大于 upper_bound，则将其设为 upper_bound
        elif dest[0] > upper_bound:
            dest[0] = upper_bound

    # 定义一个 C 语言扩展函数，计算分裂点的中间值，用于单调性约束
    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Monotonicity constraints are only supported for single-output trees we can safely assume
        n_outputs == 1.
        """
        # 返回左右子节点值的简单平均值作为分裂点的中间值
        return (
            (self.sum_left[0] / (2 * self.weighted_n_left)) +
            (self.sum_right[0] / (2 * self.weighted_n_right))
        )

    # 定义一个 C 语言扩展函数，检查当前回归分裂点是否满足单调性约束
    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current regression split"""
        # 声明两个 float64_t 类型的变量，计算左右子节点的值
        cdef:
            float64_t value_left = self.sum_left[0] / self.weighted_n_left
            float64_t value_right = self.sum_right[0] / self.weighted_n_right

        # 调用内部方法 _check_monotonicity，检查单调性约束是否满足
        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)
# 定义一个 C 扩展类 MSE，继承自 RegressionCriterion
cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """

    # 定义一个 C 扩展方法 node_impurity，计算当前节点的不纯度
    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        # 声明变量 impurity 为 float64_t 类型
        cdef float64_t impurity
        # 声明变量 k 为 intp_t 类型

        # 计算当前节点的不纯度，基于总平方和除以加权节点样本数
        impurity = self.sq_sum_total / self.weighted_n_node_samples
        # 对于每个输出维度 k，减去平方总和除以加权节点样本数的平方
        for k in range(self.n_outputs):
            impurity -= (self.sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    # 定义一个 C 扩展方法 proxy_impurity_improvement，计算不纯度减少的代理值
    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The MSE proxy is derived from

            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2

        Neglecting constant terms, this gives:

            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """
        # 声明变量 k 为 intp_t 类型
        cdef intp_t k
        # 声明变量 proxy_impurity_left 和 proxy_impurity_right 为 float64_t 类型，初始化为 0.0
        cdef float64_t proxy_impurity_left = 0.0
        cdef float64_t proxy_impurity_right = 0.0

        # 对于每个输出维度 k，计算左子节点和右子节点的代理不纯度
        for k in range(self.n_outputs):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k]

        # 返回左子节点和右子节点的代理不纯度除以加权左子节点样本数和加权右子节点样本数的和
        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)
    # 定义一个 C 语言扩展函数，计算子节点的不纯度指标
    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """
        # 获得样本权重和样本索引
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef intp_t pos = self.pos
        cdef intp_t start = self.start

        # 初始化左右子节点的平方和
        cdef float64_t sq_sum_left = 0.0
        cdef float64_t sq_sum_right

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0

        cdef intp_t end_non_missing

        # 计算左子节点的平方和
        for p in range(start, pos):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        # 如果遗漏值归于左子节点
        if self.missing_go_to_left:
            # 累加这些遗漏值对左子节点统计数据的影响
            # 注意：这只影响了平方和，因为和已在其他地方修改
            end_non_missing = self.end - self.n_missing

            for p in range(end_non_missing, self.end):
                i = sample_indices[p]
                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    sq_sum_left += w * y_ik * y_ik

        # 计算右子节点的平方和
        sq_sum_right = self.sq_sum_total - sq_sum_left

        # 计算左右子节点的加权平均不纯度
        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        # 根据左右子节点的总和，调整不纯度
        for k in range(self.n_outputs):
            impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0

        # 最后除以输出的数量，得到最终的不纯度
        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs
cdef class MAE(RegressionCriterion):
    r"""Mean absolute error impurity criterion.

       MAE = (1 / n)*(\sum_i |y_i - f_i|), where y_i is the true
       value and f_i is the predicted value."""

    cdef cnp.ndarray left_child
    cdef cnp.ndarray right_child
    cdef void** left_child_ptr
    cdef void** right_child_ptr
    cdef float64_t[::1] node_medians

    def __cinit__(self, intp_t n_outputs, intp_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets to be predicted

        n_samples : intp_t
            The total number of samples to fit on
        """
        # Default values
        self.start = 0  # 初始值为0，表示开始位置
        self.pos = 0    # 初始值为0，表示当前位置
        self.end = 0    # 初始值为0，表示结束位置

        self.n_outputs = n_outputs  # 设置目标数，即要预测的目标数量
        self.n_samples = n_samples  # 设置样本数，即要拟合的总样本数
        self.n_node_samples = 0    # 节点样本数，初始化为0
        self.weighted_n_node_samples = 0.0  # 加权节点样本数，初始化为0.0
        self.weighted_n_left = 0.0   # 加权左子节点样本数，初始化为0.0
        self.weighted_n_right = 0.0  # 加权右子节点样本数，初始化为0.0

        self.node_medians = np.zeros(n_outputs, dtype=np.float64)  # 创建一个全零数组，存储中位数值

        self.left_child = np.empty(n_outputs, dtype='object')  # 创建一个空数组，用于存储左子节点的WeightedMedianCalculator对象
        self.right_child = np.empty(n_outputs, dtype='object')  # 创建一个空数组，用于存储右子节点的WeightedMedianCalculator对象
        # 初始化WeightedMedianCalculator对象
        for k in range(n_outputs):
            self.left_child[k] = WeightedMedianCalculator(n_samples)
            self.right_child[k] = WeightedMedianCalculator(n_samples)

        self.left_child_ptr = <void**> cnp.PyArray_DATA(self.left_child)  # 获取左子节点数据的指针
        self.right_child_ptr = <void**> cnp.PyArray_DATA(self.right_child)  # 获取右子节点数据的指针

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """在不持有全局解释器锁的情况下初始化准则对象。

        这里初始化了准则对象，使用了从节点样本索引（sample_indices[start:end]）和子节点
        样本索引（sample_indices[start:start] 和 sample_indices[start:end]）得到的数据。

        """
        cdef intp_t i, p, k
        cdef float64_t w = 1.0

        # 初始化字段
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        for k in range(self.n_outputs):
            (<WeightedMedianCalculator> left_child[k]).reset()
            (<WeightedMedianCalculator> right_child[k]).reset()

        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                # push方法最终调用了safe_realloc，因此使用`except -1`
                # 将所有值推送到右侧，
                # 因为初始时pos = start
                (<WeightedMedianCalculator> right_child[k]).push(self.y[i, k], w)

            self.weighted_n_node_samples += w
        # 计算节点的中位数
        for k in range(self.n_outputs):
            self.node_medians[k] = (<WeightedMedianCalculator> right_child[k]).get_median()

        # 重置到pos=start
        self.reset()
        return 0

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """如果n_missing不为0，则引发错误异常。"""
        if n_missing == 0:
            return
        with gil:
            raise ValueError("对于MAE，不支持缺失值。")
    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 声明整型变量 i, k
        cdef intp_t i, k
        # 声明浮点型变量 value, weight
        cdef float64_t value
        cdef float64_t weight

        # 获取左右子节点指针数组的引用
        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        # 初始化权重为 0
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        # 设置当前位置为起始位置
        self.pos = self.start

        # 重置 WeightedMedianCalculator 对象，左边不含元素，右边包含所有元素
        for k in range(self.n_outputs):
            # 如果左边没有元素，则已经重置
            for i in range((<WeightedMedianCalculator> left_child[k]).size()):
                # 从左边弹出所有元素并放入右边
                (<WeightedMedianCalculator> left_child[k]).pop(&value,
                                                               &weight)
                # 调用 push 方法最终会调用 safe_realloc，因此用 `except -1` 处理异常
                (<WeightedMedianCalculator> right_child[k]).push(value,
                                                                 weight)
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 将权重初始化为 0
        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        # 将当前位置设置为结束位置
        self.pos = self.end

        # 获取左右子节点指针数组的引用
        cdef float64_t value
        cdef float64_t weight
        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        # 反向重置 WeightedMedianCalculator 对象，右边不含元素，左边包含所有元素
        for k in range(self.n_outputs):
            # 如果右边没有元素，则已经重置
            for i in range((<WeightedMedianCalculator> right_child[k]).size()):
                # 从右边弹出所有元素并放入左边
                (<WeightedMedianCalculator> right_child[k]).pop(&value,
                                                                &weight)
                # 调用 push 方法最终会调用 safe_realloc，因此用 `except -1` 处理异常
                (<WeightedMedianCalculator> left_child[k]).push(value,
                                                                weight)
        return 0
    # 定义一个 Cython 的 cdef 函数，用于更新统计信息并移动样本索引在 pos 到 new_pos 之间的部分到左侧。

    # 如果分配内存失败则返回 -1 （并引发 MemoryError），否则返回 0。
    cdef int update(self, intp_t new_pos) except -1 nogil:
        
        # 将样本权重和样本索引声明为常量数组
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        # 左子节点和右子节点的指针
        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        # 当前位置 pos 和结束位置 end
        cdef intp_t pos = self.pos
        cdef intp_t end = self.end

        # 循环中使用的变量声明
        cdef intp_t i, p, k
        cdef float64_t w = 1.0

        # 根据 new_pos 和 pos 的关系选择更新的方向
        if (new_pos - pos) <= (end - new_pos):
            # 从 pos 到 new_pos 更新统计信息
            for p in range(pos, new_pos):
                i = sample_indices[p]

                # 如果有样本权重，则更新权重值
                if sample_weight is not None:
                    w = sample_weight[i]

                # 对每个输出维度 k 进行操作：
                # 从右边移除 y_ik 及其权重 w 并加入左边
                (<WeightedMedianCalculator> right_child[k]).remove(self.y[i, k], w)
                # 调用 push 方法最终调用 safe_realloc，因此需要 except -1
                (<WeightedMedianCalculator> left_child[k]).push(self.y[i, k], w)

                # 左边的加权样本数量增加
                self.weighted_n_left += w
        else:
            # 如果从 end 到 new_pos 更新
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]

                # 如果有样本权重，则更新权重值
                if sample_weight is not None:
                    w = sample_weight[i]

                # 对每个输出维度 k 进行操作：
                # 从左边移除 y_ik 及其权重 w 并加入右边
                (<WeightedMedianCalculator> left_child[k]).remove(self.y[i, k], w)
                (<WeightedMedianCalculator> right_child[k]).push(self.y[i, k], w)

                # 左边的加权样本数量减少
                self.weighted_n_left -= w

        # 更新右边的加权样本数量
        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        
        # 更新当前位置 pos
        self.pos = new_pos
        
        # 返回成功状态
        return 0

    # 定义一个 Cython 的 cdef 函数，计算样本索引从 start 到 end 的节点值到 dest 数组中
    cdef void node_value(self, float64_t* dest) noexcept nogil:
        
        # 对每个输出维度 k 进行操作：
        for k in range(self.n_outputs):
            # 将节点的中位数值赋给 dest 数组
            dest[k] = <float64_t> self.node_medians[k]
    # 计算当前节点分裂的中间值，用于单调性约束，是左右子节点值的简单平均值
    cdef inline float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.
    
        Monotonicity constraints are only supported for single-output trees we can safely assume
        n_outputs == 1.
        """
        return (
                (<WeightedMedianCalculator> self.left_child_ptr[0]).get_median() +
                (<WeightedMedianCalculator> self.right_child_ptr[0]).get_median()
        ) / 2
    
    # 检查当前回归分裂点是否满足单调性约束
    cdef inline bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current regression split"""
        # 获取左右子节点的中位数值
        cdef:
            float64_t value_left = (<WeightedMedianCalculator> self.left_child_ptr[0]).get_median()
            float64_t value_right = (<WeightedMedianCalculator> self.right_child_ptr[0]).get_median()
    
        # 调用内部函数检查单调性约束是否满足
        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)
    
    # 计算当前节点的不纯度
    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.
    
        Evaluate the MAE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef intp_t i, p, k
        cdef float64_t w = 1.0
        cdef float64_t impurity = 0.0
    
        for k in range(self.n_outputs):
            for p in range(self.start, self.end):
                i = sample_indices[p]
    
                # 如果有样本权重，则更新权重值
                if sample_weight is not None:
                    w = sample_weight[i]
    
                # 计算绝对误差的加权和，用于评估不纯度
                impurity += fabs(self.y[i, k] - self.node_medians[k]) * w
    
        # 返回加权平均不纯度
        return impurity / (self.weighted_n_node_samples * self.n_outputs)
    # 定义 Cython 函数，评估子节点的不纯度
    cdef void children_impurity(self, float64_t* p_impurity_left,
                                float64_t* p_impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """
        # 获取样本权重和样本索引
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        # 获取起始点、分割点和结束点的索引
        cdef intp_t start = self.start
        cdef intp_t pos = self.pos
        cdef intp_t end = self.end

        # 初始化循环变量和中位数
        cdef intp_t i, p, k
        cdef float64_t median
        cdef float64_t w = 1.0
        cdef float64_t impurity_left = 0.0
        cdef float64_t impurity_right = 0.0

        # 获取左右子节点指针
        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        # 计算左子节点的不纯度
        for k in range(self.n_outputs):
            # 获取左子节点的权重中位数
            median = (<WeightedMedianCalculator> left_child[k]).get_median()
            # 遍历左子节点的样本区间
            for p in range(start, pos):
                i = sample_indices[p]

                # 如果存在样本权重，则更新权重值
                if sample_weight is not None:
                    w = sample_weight[i]

                # 根据权重计算左子节点的不纯度
                impurity_left += fabs(self.y[i, k] - median) * w
        # 计算并存储左子节点的不纯度值
        p_impurity_left[0] = impurity_left / (self.weighted_n_left *
                                              self.n_outputs)

        # 计算右子节点的不纯度
        for k in range(self.n_outputs):
            # 获取右子节点的权重中位数
            median = (<WeightedMedianCalculator> right_child[k]).get_median()
            # 遍历右子节点的样本区间
            for p in range(pos, end):
                i = sample_indices[p]

                # 如果存在样本权重，则更新权重值
                if sample_weight is not None:
                    w = sample_weight[i]

                # 根据权重计算右子节点的不纯度
                impurity_right += fabs(self.y[i, k] - median) * w
        # 计算并存储右子节点的不纯度值
        p_impurity_right[0] = impurity_right / (self.weighted_n_right *
                                                self.n_outputs)
cdef class FriedmanMSE(MSE):
    """Mean squared error impurity criterion with improvement score by Friedman.

    Uses the formula (35) in Friedman's original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    # 计算用于加速最佳分割搜索的不纯度减少的代理量
    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef float64_t total_sum_left = 0.0
        cdef float64_t total_sum_right = 0.0

        cdef intp_t k
        cdef float64_t diff = 0.0

        # 计算左右子节点的总和
        for k in range(self.n_outputs):
            total_sum_left += self.sum_left[k]
            total_sum_right += self.sum_right[k]

        # 计算 diff^2 / (n_left * n_right)，代表不纯度改善的估计
        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right)

        return diff * diff / (self.weighted_n_left * self.weighted_n_right)

    # 计算真实不纯度改善的方法
    cdef float64_t impurity_improvement(self, float64_t impurity_parent, float64_t
                                        impurity_left, float64_t impurity_right) noexcept nogil:
        # Note: none of the arguments are used here
        cdef float64_t total_sum_left = 0.0
        cdef float64_t total_sum_right = 0.0

        cdef intp_t k
        cdef float64_t diff = 0.0

        # 计算左右子节点的总和
        for k in range(self.n_outputs):
            total_sum_left += self.sum_left[k]
            total_sum_right += self.sum_right[k]

        # 计算 diff^2 / (n_left * n_right * n_node_samples)，用于真实的不纯度改善
        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right) / self.n_outputs

        return (diff * diff / (self.weighted_n_left * self.weighted_n_right *
                               self.weighted_n_node_samples))


cdef class Poisson(RegressionCriterion):
    """Half Poisson deviance as impurity criterion.

    Poisson deviance = 2/n * sum(y_true * log(y_true/y_pred) + y_pred - y_true)

    Note that the deviance is >= 0, and since we have `y_pred = mean(y_true)`
    at the leaves, one always has `sum(y_pred - y_true) = 0`. It remains the
    implemented impurity (factor 2 is skipped):
        1/n * sum(y_true * log(y_true/y_pred)
    """
    # FIXME in 1.0:
    # min_impurity_split with default = 0 forces us to use a non-negative
    # impurity like the Poisson deviance. Without this restriction, one could
    # throw away the 'constant' term sum(y_true * log(y_true)) and just use
    # Poisson loss = - 1/n * sum(y_true * log(y_pred))
    #              = - 1/n * sum(y_true * log(mean(y_true))
    #              = - mean(y_true) * log(mean(y_true))
    # 定义一个Cython函数，计算当前节点的不纯度，返回值为float64_t
    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the Poisson criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        # 调用poisson_loss函数计算当前节点的不纯度，基于start、end、sum_total和weighted_n_node_samples参数
        return self.poisson_loss(self.start, self.end, self.sum_total,
                                 self.weighted_n_node_samples)

    # 定义一个Cython函数，计算一个不纯度改善的代理量，返回值为float64_t
    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The Poisson proxy is derived from:

              sum_{i left }(y_i * log(y_i / y_pred_L))
            + sum_{i right}(y_i * log(y_i / y_pred_R))
            = sum(y_i * log(y_i) - n_L * mean_{i left}(y_i) * log(mean_{i left}(y_i))
                                 - n_R * mean_{i right}(y_i) * log(mean_{i right}(y_i))

        Neglecting constant terms, this gives

            - sum{i left }(y_i) * log(mean{i left}(y_i))
            - sum{i right}(y_i) * log(mean{i right}(y_i))
        """
        # 定义变量和初始化
        cdef intp_t k
        cdef float64_t proxy_impurity_left = 0.0
        cdef float64_t proxy_impurity_right = 0.0
        cdef float64_t y_mean_left = 0.
        cdef float64_t y_mean_right = 0.

        # 遍历每个输出变量
        for k in range(self.n_outputs):
            # 如果左子树或右子树的和小于等于EPSILON，则返回负无穷
            if (self.sum_left[k] <= EPSILON) or (self.sum_right[k] <= EPSILON):
                # Poisson损失不允许非正预测值，因此禁止具有子节点和小于等于0的分割
                # 由于sum_right = sum_total - sum_left，这可能会导致浮点舍入误差，不会得到零。因此，我们放宽对sum(y_i) <= EPSILON的比较。
                return -INFINITY
            else:
                # 计算左子树和右子树的平均值
                y_mean_left = self.sum_left[k] / self.weighted_n_left
                y_mean_right = self.sum_right[k] / self.weighted_n_right
                # 计算左子树和右子树的代理不纯度改善
                proxy_impurity_left -= self.sum_left[k] * log(y_mean_left)
                proxy_impurity_right -= self.sum_right[k] * log(y_mean_right)

        # 返回左子树和右子树的代理不纯度改善之和的负值
        return - proxy_impurity_left - proxy_impurity_right
    # 定义一个 Cython 的函数，计算子节点的不纯度（impurity）
    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity of the right child (sample_indices[pos:end]) for Poisson.
        """
        # 从 self 对象中提取起始、中间和结束索引
        cdef intp_t start = self.start
        cdef intp_t pos = self.pos
        cdef intp_t end = self.end

        # 计算左子节点的不纯度，调用 poisson_loss 函数计算
        impurity_left[0] = self.poisson_loss(start, pos, self.sum_left,
                                             self.weighted_n_left)

        # 计算右子节点的不纯度，调用 poisson_loss 函数计算
        impurity_right[0] = self.poisson_loss(pos, end, self.sum_right,
                                              self.weighted_n_right)

    # 定义一个内联函数，计算给定节点的 Poisson 损失（即偏差）
    cdef inline float64_t poisson_loss(
        self,
        intp_t start,
        intp_t end,
        const float64_t[::1] y_sum,
        float64_t weight_sum
    ) noexcept nogil:
        """Helper function to compute Poisson loss (~deviance) of a given node.
        """
        # 从 self 对象中提取 y 数组、样本权重和样本索引
        cdef const float64_t[:, ::1] y = self.y
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        # 初始化 y_mean 和 poisson_loss
        cdef float64_t y_mean = 0.
        cdef float64_t poisson_loss = 0.
        cdef float64_t w = 1.0
        cdef intp_t i, k, p
        cdef intp_t n_outputs = self.n_outputs

        # 遍历输出的维度
        for k in range(n_outputs):
            # 如果 y_sum[k] 小于等于 EPSILON，返回无穷大，避免除以零或接近零的数
            if y_sum[k] <= EPSILON:
                # y_sum 可能是从 sum_total - sum_left 的减法计算得来，可能存在浮点舍入误差。
                # 因此，我们用 y_sum <= EPSILON 来放宽比较条件，防止出现问题。
                return INFINITY

            # 计算 y_mean，即 y_sum[k] / weight_sum
            y_mean = y_sum[k] / weight_sum

            # 遍历样本索引的范围，计算 Poisson 损失
            for p in range(start, end):
                i = sample_indices[p]

                # 如果有样本权重，则更新 w
                if sample_weight is not None:
                    w = sample_weight[i]

                # 计算 Poisson 损失的一部分，使用 xlogy 函数
                poisson_loss += w * xlogy(y[i, k], y[i, k] / y_mean)

        # 返回 Poisson 损失除以权重总和和输出维度的乘积
        return poisson_loss / (weight_sum * n_outputs)
```