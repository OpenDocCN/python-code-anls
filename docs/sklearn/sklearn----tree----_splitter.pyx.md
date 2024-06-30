# `D:\src\scipysrc\scikit-learn\sklearn\tree\_splitter.pyx`

```
# 从Cython中导入final修饰符，标记此类为最终类，禁止继承
from cython cimport final
# 从libc.math中导入isnan函数，用于检查浮点数是否为NaN
from libc.math cimport isnan
# 从libc.stdlib中导入qsort函数，用于快速排序
from libc.stdlib cimport qsort
# 从libc.string中导入memcpy函数，用于内存复制操作

# 从_criterion模块中导入Criterion类
from ._criterion cimport Criterion
# 从_utils模块中导入log函数
from ._utils cimport log
# 从_utils模块中导入rand_int函数
from ._utils cimport rand_int
# 从_utils模块中导入rand_uniform函数
from ._utils cimport rand_uniform
# 从_utils模块中导入RAND_R_MAX常量
from ._utils cimport RAND_R_MAX
# 从..utils._typedefs模块中导入int8_t类型定义

# 导入NumPy库并使用别名np
import numpy as np
# 从SciPy.sparse模块中导入issparse函数

# 定义浮点类型常量INFINITY，值为正无穷大
cdef float64_t INFINITY = np.inf

# 定义浮点类型常量FEATURE_THRESHOLD，用于缓解32位和64位精度差异
cdef float32_t FEATURE_THRESHOLD = 1e-7

# 定义浮点类型常量EXTRACT_NNZ_SWITCH，用于在SparsePartitioner中切换算法非零值提取算法
cdef float32_t EXTRACT_NNZ_SWITCH = 0.1

# 定义内联函数_init_split，初始化SplitRecord对象的各个属性值
cdef inline void _init_split(SplitRecord* self, intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False
    self.n_missing = 0

# 定义Splitter类，用于抽象分割器类，被树构建器调用以逐步找到最佳分割点（用于稀疏和密集数据）
cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    # 构造函数__cinit__，初始化Splitter对象的各个属性
    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : intp_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : intp_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : float64_t
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const int8_t[:]
            Monotonicity constraints

        """

        # 初始化Splitter对象的属性
        self.criterion = criterion
        self.n_samples = 0
        self.n_features = 0
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None

    # 方法__getstate__，返回对象的状态信息
    def __getstate__(self):
        return {}

    # 方法__setstate__，设置对象的状态信息
    def __setstate__(self, d):
        pass
    def __reduce__(self):
        # 返回一个元组，用于对象的序列化和反序列化
        return (type(self), (self.criterion,
                             self.max_features,
                             self.min_samples_leaf,
                             self.min_weight_leaf,
                             self.random_state,
                             self.monotonic_cst), self.__getstate__())

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        """初始化分裂器。

        接收输入数据 X、目标 y 和可选的样本权重。

        在内存分配失败时返回 -1（并引发 MemoryError），否则返回 0。

        Parameters
        ----------
        X : object
            包含输入数据的对象，通常是一个二维 numpy 数组。

        y : ndarray, dtype=float64_t
            样本的目标或真实标签，以 Cython 内存视图表示。

        sample_weight : ndarray, dtype=float64_t
            样本的权重，权重较高的样本比权重较低的样本拟合得更紧密。如果未提供，则假定所有样本权重均相同。
            以 Cython 内存视图表示。

        has_missing : bool
            X 中至少有一个缺失值。
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef intp_t n_samples = X.shape[0]

        # 创建一个新数组，用于存储感兴趣特征的非零样本
        self.samples = np.empty(n_samples, dtype=np.intp)
        cdef intp_t[::1] samples = self.samples

        cdef intp_t i, j
        cdef float64_t weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # 仅处理权重为正的样本
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # 样本数量为权重为正的样本数
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef intp_t n_features = X.shape[1]
        self.features = np.arange(n_features, dtype=np.intp)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.constant_features = np.empty(n_features, dtype=np.intp)

        self.y = y

        self.sample_weight = sample_weight
        if missing_values_in_feature_mask is not None:
            self.criterion.init_sum_missing()
        return 0

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
        ):
        """重置节点的状态。

        重置节点在样本索引范围内的状态，包括节点的加权样本数。

        Parameters
        ----------
        start : intp_t
            起始样本的索引。

        end : intp_t
            结束样本的索引。

        weighted_n_node_samples : float64_t*
            指向节点加权样本数的指针。

        Returns
        -------
        int
            返回 0 表示成功重置节点状态。
        """
    ) except -1 nogil:
        """
        重置节点样本 samples[start:end] 的分割器。

        在分配内存失败时返回 -1（并引发 MemoryError 异常），否则返回 0。

        参数
        ----------
        start : intp_t
            考虑的第一个样本的索引
        end : intp_t
            考虑的最后一个样本的索引
        weighted_n_node_samples : ndarray, dtype=float64_t 指针
            这些样本的总权重
        """

        self.start = start
        self.end = end

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end
        )

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        """
        在节点样本 samples[start:end] 上找到最佳分割点。

        这是一个占位符方法。大部分计算将在这里完成。

        如果发生错误应返回 -1。
        """

        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """
        将节点样本 samples[start:end] 的值复制到 dest 中。
        """

        self.criterion.node_value(dest)

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """
        对于单调性约束，将 dest 中的值剪切在 lower_bound 和 upper_bound 之间。
        """

        self.criterion.clip_node_value(dest, lower_bound, upper_bound)

    cdef float64_t node_impurity(self) noexcept nogil:
        """
        返回当前节点的不纯度。
        """

        return self.criterion.node_impurity()
# 定义一个内联函数，用于将缺失值的样本索引移到分裂点左侧（如果需要的话）
# 函数参数说明：
# - best: 指向最佳分裂记录的指针
# - samples: 整数指针数组，存储样本索引
# - end: 整数，表示样本数组的结束索引（不包括）
# 注意事项：
# - 这个函数必须在最后调用，因为它会移动样本，从而影响计算标准
# - 这会影响子节点的不纯度计算，进而影响下一个节点的计算
cdef inline void shift_missing_values_to_left_if_required(
    SplitRecord* best,
    intp_t[::1] samples,
    intp_t end,
) noexcept nogil:
    cdef intp_t i, p, current_end
    # 如果需要将缺失值移到分裂点左侧，则执行以下操作：
    # - 对于每个缺失值，将其索引从右侧移到分裂点左侧
    # - 更新最佳分裂记录中的位置指针 `best.pos`
    if best.n_missing > 0 and best.missing_go_to_left:
        for p in range(best.n_missing):
            i = best.pos + p
            current_end = end - 1 - p
            samples[i], samples[current_end] = samples[current_end], samples[i]
        best.pos += best.n_missing

# 引入一个融合类，以便在 `node_split_best` 和 `node_split_random` 函数中共享分裂实现
# 这个类用于在节点分裂函数中密集和稀疏情况下使用
# 注意事项：
# - 使用融合类的方法避免了基于继承的多态性带来的性能损耗
# - 继承会导致频繁的虚拟方法查找，造成整体树拟合性能下降约10%
ctypedef fused Partitioner:
    DensePartitioner  # 密集分裂器类
    SparsePartitioner  # 稀疏分裂器类

cdef inline int node_split_best(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    """在节点样本 samples[start:end] 上寻找最佳分裂点

    如果内存分配失败，则返回 -1（并引发 MemoryError），否则返回 0
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # 寻找最佳分裂点
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t end_non_missing
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_searches
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    # 获取样本索引、特征索引和常量特征
    cdef intp_t[::1] samples = splitter.samples
    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    # 获取特征值、最大特征数、最小叶子样本数和最小叶子权重
    cdef float32_t[::1] feature_values = splitter.feature_values
    cdef intp_t max_features = splitter.max_features
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    # 当前代理改进值初始化为负无穷大
    cdef float64_t current_proxy_improvement = -INFINITY
    # 最佳代理改进值初始化为负无穷大
    cdef float64_t best_proxy_improvement = -INFINITY

    # 从父节点记录中获取不纯度、下界和上界的值
    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    # 初始化变量 f_i 为特征数，f_j、p 和 p_prev 为整数类型
    cdef intp_t f_i = n_features
    cdef intp_t f_j
    cdef intp_t p
    cdef intp_t p_prev

    # 已访问的特征数初始化为 0
    cdef intp_t n_visited_features = 0
    # 在分割搜索过程中发现的常数特征数
    cdef intp_t n_found_constants = 0
    # 已绘制且不替换的常数特征数
    cdef intp_t n_drawn_constants = 0
    # 已知的常数特征数由父节点记录提供
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # 总的常数特征数为已知和发现的常数特征数之和
    cdef intp_t n_total_constants = n_known_constants

    # 初始化最佳分割点
    _init_split(&best_split, end)

    # 初始化节点分割器
    partitioner.init_node_split(start, end)

    # 使用 Fisher-Yates 算法无替换地对 max_features 进行抽样
    # 跳过已被祖先节点检测为常数特征的计算不纯度标准的步骤，
    # 并保存新发现常数特征的信息，以节省后代节点的计算资源
    # 重新组织样本为 samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        partitioner.partition_samples_final(
            best_split.pos,
            best_split.threshold,
            best_split.feature,
            best_split.n_missing
        )
        criterion.init_missing(best_split.n_missing)
        criterion.missing_go_to_left = best_split.missing_go_to_left

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

        shift_missing_values_to_left_if_required(&best_split, samples, end)

    # 保持常数特征的不变性：features[:n_known_constants] 的原始顺序
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # 复制新发现的常数特征到 constant_features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # 更新父节点记录中的常数特征数
    parent_record.n_constant_features = n_total_constants
    # 将最佳分割信息写入 split 数组的第一个元素
    split[0] = best_split
    # 返回值 0 表示函数成功执行
    return 0
# 根据特征值 feature_values 和样本 samples 同时对指向的 n 元素数组进行排序。
# 使用 Introsort 算法（Musser, SP&E, 1997）。
cdef inline void sort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil:
    if n == 0:
        return
    # 计算最大深度 maxd，这里使用对数函数 log
    cdef intp_t maxd = 2 * <intp_t>log(n)
    # 调用 introsort 函数进行排序
    introsort(feature_values, samples, n, maxd)


cdef inline void swap(float32_t* feature_values, intp_t* samples,
                      intp_t i, intp_t j) noexcept nogil:
    # 辅助函数，用于在 sort 函数中交换 feature_values 和 samples 中的元素
    feature_values[i], feature_values[j] = feature_values[j], feature_values[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline float32_t median3(float32_t* feature_values, intp_t n) noexcept nogil:
    # 三数中值法选择枢纽元素，参考 Bentley 和 McIlroy (1993) 的工程排序方法。
    # 算法工程化，SP&E。平均需要 8/3 次比较。
    cdef float32_t a = feature_values[0], b = feature_values[n / 2], c = feature_values[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# 使用三数中值法选择枢纽元素和三路划分函数的 Introsort
# 对重复元素具有鲁棒性，例如大量零特征。
cdef void introsort(float32_t* feature_values, intp_t *samples,
                    intp_t n, intp_t maxd) noexcept nogil:
    cdef float32_t pivot
    cdef intp_t i, l, r

    while n > 1:
        if maxd <= 0:   # 达到最大深度限制，进入堆排序（heapsort）
            heapsort(feature_values, samples, n)
            return
        maxd -= 1

        # 使用 median3 函数选择枢纽元素
        pivot = median3(feature_values, n)

        # 三路划分
        i = l = 0
        r = n
        while i < r:
            if feature_values[i] < pivot:
                swap(feature_values, samples, i, l)
                i += 1
                l += 1
            elif feature_values[i] > pivot:
                r -= 1
                swap(feature_values, samples, i, r)
            else:
                i += 1

        # 递归调用 introsort 对左侧子数组进行排序
        introsort(feature_values, samples, l, maxd)
        # 更新 feature_values 和 samples 指针及剩余元素数量
        feature_values += r
        samples += r
        n -= r


cdef inline void sift_down(float32_t* feature_values, intp_t* samples,
                           intp_t start, intp_t end) noexcept nogil:
    # 将 feature_values[start:end] 中的最大元素移动到 start 位置，恢复堆的顺序
    cdef intp_t child, maxind, root
    # 进入循环，该循环用于维护最大堆的性质
    while True:
        # 计算当前根节点的左子节点索引
        child = root * 2 + 1

        # 找到根节点、左子节点和右子节点中的最大值的索引
        maxind = root
        # 检查左子节点是否存在且比根节点大
        if child < end and feature_values[maxind] < feature_values[child]:
            maxind = child
        # 检查右子节点是否存在且比当前最大值大
        if child + 1 < end and feature_values[maxind] < feature_values[child + 1]:
            maxind = child + 1

        # 如果最大值索引仍然是根节点索引，表示当前节点已经满足最大堆性质，结束循环
        if maxind == root:
            break
        else:
            # 否则，交换根节点和最大值节点，然后继续向下调整
            swap(feature_values, samples, root, maxind)
            root = maxind
cdef void heapsort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil:
    cdef intp_t start, end

    # heapify
    start = (n - 2) / 2  # 计算堆化的起始索引
    end = n
    while True:
        sift_down(feature_values, samples, start, end)  # 对堆进行调整
        if start == 0:  # 如果已经处理到堆顶
            break
        start -= 1  # 继续处理上一个节点

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(feature_values, samples, 0, end)  # 交换堆顶和当前最后一个元素
        sift_down(feature_values, samples, 0, end)  # 对剩余堆重新调整
        end = end - 1  # 缩小堆范围，继续下一轮排序

cdef inline int node_split_random(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    """Find the best random split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Draw random splits and pick the best
    cdef intp_t start = splitter.start  # 节点数据的起始索引
    cdef intp_t end = splitter.end  # 节点数据的结束索引

    cdef intp_t[::1] features = splitter.features  # 可用于分裂的特征索引列表
    cdef intp_t[::1] constant_features = splitter.constant_features  # 已知为常量的特征索引列表
    cdef intp_t n_features = splitter.n_features  # 可用于分裂的特征总数

    cdef intp_t max_features = splitter.max_features  # 最大用于分裂的特征数
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf  # 叶子节点的最小样本数
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf  # 叶子节点的最小权重
    cdef uint32_t* random_state = &splitter.rand_r_state  # 随机数生成器状态指针

    cdef SplitRecord best_split, current_split
    cdef float64_t current_proxy_improvement = - INFINITY  # 当前最佳分裂的提升值
    cdef float64_t best_proxy_improvement = - INFINITY  # 最好的分裂的提升值

    cdef float64_t impurity = parent_record.impurity  # 父节点的不纯度
    cdef float64_t lower_bound = parent_record.lower_bound  # 父节点的下界
    cdef float64_t upper_bound = parent_record.upper_bound  # 父节点的上界

    cdef intp_t f_i = n_features  # 特征索引的迭代变量
    cdef intp_t f_j
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0  # 在分裂搜索中发现的常量特征数
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0  # 已知为常量且不再进行替换的特征数
    cdef intp_t n_known_constants = parent_record.n_constant_features  # 父节点已知为常量的特征数
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants  # 总的已知常量特征数
    cdef intp_t n_visited_features = 0  # 已访问的特征数
    cdef float32_t min_feature_value
    cdef float32_t max_feature_value

    _init_split(&best_split, end)  # 初始化最佳分裂记录

    partitioner.init_node_split(start, end)  # 初始化节点分裂过程

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # 新发现的常量特征，以减少后续节点的计算量。
    # 重新组织样本数组，将 samples[start:best.pos] + samples[best.pos:end] 合并
    if best_split.pos < end:
        if current_split.feature != best_split.feature:
            # TODO: 当随机分裂器支持缺失值时，传入 best.n_missing
            partitioner.partition_samples_final(
                best_split.pos, best_split.threshold, best_split.feature, 0
            )

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity, best_split.impurity_left, best_split.impurity_right
        )

    # 保持常量特征的不变性：features[:n_known_constants] 中元素的原始顺序必须对兄弟节点和子节点保持不变
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # 复制新发现的常量特征
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # 返回值设定
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split
    return 0
@final
cdef class DensePartitioner:
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef:
        const float32_t[:, :] X   # 常量二维浮点数数组 X，存储输入的密集数据
        cdef intp_t[::1] samples  # 一维整数数组 samples，存储样本索引
        cdef float32_t[::1] feature_values  # 一维浮点数数组 feature_values，存储特征值
        cdef intp_t start         # 起始索引 start，用于节点分割的起始位置
        cdef intp_t end           # 终止索引 end，用于节点分割的结束位置
        cdef intp_t n_missing     # 缺失值数量 n_missing，当前节点中的缺失值计数
        cdef const unsigned char[::1] missing_values_in_feature_mask  # 常量一维无符号字符数组 missing_values_in_feature_mask，
                                                                      # 用于表示特征中的缺失值情况

    def __init__(
        self,
        const float32_t[:, :] X,
        intp_t[::1] samples,
        float32_t[::1] feature_values,
        const unsigned char[::1] missing_values_in_feature_mask,
    ):
        self.X = X  # 初始化密集数据 X
        self.samples = samples  # 初始化样本索引数组 samples
        self.feature_values = feature_values  # 初始化特征值数组 feature_values
        self.missing_values_in_feature_mask = missing_values_in_feature_mask  # 初始化特征中缺失值的掩码数组

    cdef inline void init_node_split(self, intp_t start, intp_t end) noexcept nogil:
        """Initialize splitter at the beginning of node_split."""
        self.start = start  # 设置节点分割的起始位置
        self.end = end      # 设置节点分割的结束位置
        self.n_missing = 0  # 初始化当前节点的缺失值计数为 0

    cdef inline void sort_samples_and_feature_values(
        self, intp_t current_feature
        # 内联函数，用于对样本索引和特征值按当前特征进行排序
    ) noexcept nogil:
        """
        同时根据特征值对样本进行排序。

        缺失值存储在特征值末尾。
        观察到的特征值中的缺失值数量存储在 self.n_missing 中。
        """
        cdef:
            intp_t i, current_end
            float32_t[::1] feature_values = self.feature_values
            const float32_t[:, :] X = self.X
            intp_t[::1] samples = self.samples
            intp_t n_missing = 0
            const unsigned char[::1] missing_values_in_feature_mask = self.missing_values_in_feature_mask

        # 根据特征值排序样本；通过将值复制到数组中并以更有效利用缓存的方式进行排序。
        if missing_values_in_feature_mask is not None and missing_values_in_feature_mask[current_feature]:
            i, current_end = self.start, self.end - 1
            # 缺失值放置在末尾，并且不参与排序。
            while i <= current_end:
                # 找到最右边的非缺失值，以便与其左边的缺失值交换位置。
                if isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] 是一个非缺失值
                if isnan(X[samples[i], current_feature]):
                    samples[i], samples[current_end] = samples[current_end], samples[i]
                    n_missing += 1
                    current_end -= 1

                feature_values[i] = X[samples[i], current_feature]
                i += 1
        else:
            # 当没有缺失值时，只需将数据复制到 feature_values 中
            for i in range(self.start, self.end):
                feature_values[i] = X[samples[i], current_feature]

        sort(&feature_values[self.start], &samples[self.start], self.end - self.start - n_missing)
        self.n_missing = n_missing

    cdef inline void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil:
        """Find the minimum and maximum value for current_feature."""
        cdef:
            intp_t p                    # 声明变量p，用于迭代
            float32_t current_feature_value  # 当前特征值的浮点数变量
            const float32_t[:, :] X = self.X  # 声明常量X，表示输入数据的二维浮点数数组
            intp_t[::1] samples = self.samples  # 声明一维整数数组samples，表示样本索引
            float32_t min_feature_value = X[samples[self.start], current_feature]  # 初始化最小特征值为起始样本的特征值
            float32_t max_feature_value = min_feature_value  # 初始化最大特征值为最小特征值
            float32_t[::1] feature_values = self.feature_values  # 声明一维浮点数数组feature_values，表示特征值列表

        feature_values[self.start] = min_feature_value  # 将起始位置的特征值设为最小特征值

        for p in range(self.start + 1, self.end):  # 迭代从起始位置+1到结束位置的样本
            current_feature_value = X[samples[p], current_feature]  # 获取当前样本的特征值
            feature_values[p] = current_feature_value  # 将当前样本的特征值存入feature_values数组中

            if current_feature_value < min_feature_value:  # 如果当前特征值小于最小特征值
                min_feature_value = current_feature_value  # 更新最小特征值
            elif current_feature_value > max_feature_value:  # 如果当前特征值大于最大特征值
                max_feature_value = current_feature_value  # 更新最大特征值

        min_feature_value_out[0] = min_feature_value  # 将最小特征值输出到min_feature_value_out数组的第一个位置
        max_feature_value_out[0] = max_feature_value  # 将最大特征值输出到max_feature_value_out数组的第一个位置

    cdef inline void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil:
        """Compute the next p_prev and p for iteratiing over feature values.

        The missing values are not included when iterating through the feature values.
        """
        cdef:
            float32_t[::1] feature_values = self.feature_values  # 声明一维浮点数数组feature_values，表示特征值列表
            intp_t end_non_missing = self.end - self.n_missing  # 计算无缺失值时的结束位置

        while (
            p[0] + 1 < end_non_missing and
            feature_values[p[0] + 1] <= feature_values[p[0]] + FEATURE_THRESHOLD
        ):
            p[0] += 1  # 递增p的值，直到满足条件

        p_prev[0] = p[0]  # 将p的当前值赋给p_prev

        # By adding 1, we have
        # (feature_values[p] >= end) or (feature_values[p] > feature_values[p - 1])
        p[0] += 1  # 递增p的值

    cdef inline intp_t partition_samples(self, float64_t current_threshold) noexcept nogil:
        """Partition samples for feature_values at the current_threshold."""
        cdef:
            intp_t p = self.start  # 初始化p为起始位置
            intp_t partition_end = self.end  # 初始化partition_end为结束位置
            intp_t[::1] samples = self.samples  # 声明一维整数数组samples，表示样本索引
            float32_t[::1] feature_values = self.feature_values  # 声明一维浮点数数组feature_values，表示特征值列表

        while p < partition_end:  # 当p小于partition_end时循环
            if feature_values[p] <= current_threshold:  # 如果特征值小于当前阈值
                p += 1  # 递增p的值
            else:
                partition_end -= 1  # 减少partition_end的值

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p]
                )  # 交换feature_values中p和partition_end位置的值
                samples[p], samples[partition_end] = samples[partition_end], samples[p]  # 交换samples中p和partition_end位置的值

        return partition_end  # 返回partition_end的值

    cdef inline void partition_samples_final(
        self,
        intp_t best_pos,
        float64_t best_threshold,
        intp_t best_feature,
        intp_t best_n_missing,
    ) noexcept nogil:
        """Partition samples for X at the best_threshold and best_feature.

        If missing values are present, this method partitions `samples`
        so that the `best_n_missing` missing values' indices are in the
        right-most end of `samples`, that is `samples[end_non_missing:end]`.
        """
        cdef:
            # Local invariance: start <= p <= partition_end <= end
            intp_t start = self.start               # 起始索引
            intp_t p = start                        # 当前索引
            intp_t end = self.end - 1               # 结束索引
            intp_t partition_end = end - best_n_missing  # 分区结束索引，考虑了缺失值的影响
            intp_t[::1] samples = self.samples      # 样本索引数组
            const float32_t[:, :] X = self.X        # 特征矩阵 X
            float32_t current_value                 # 当前特征值

        if best_n_missing != 0:
            # Move samples with missing values to the end while partitioning the
            # non-missing samples
            while p < partition_end:
                # Keep samples with missing values at the end
                if isnan(X[samples[end], best_feature]):
                    end -= 1
                    continue

                # Swap sample with missing values with the sample at the end
                current_value = X[samples[p], best_feature]
                if isnan(current_value):
                    samples[p], samples[end] = samples[end], samples[p]
                    end -= 1

                    # The swapped sample at the end is always a non-missing value, so
                    # we can continue the algorithm without checking for missingness.
                    current_value = X[samples[p], best_feature]

                # Partition the non-missing samples
                if current_value <= best_threshold:
                    p += 1
                else:
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    partition_end -= 1
        else:
            # Partitioning routine when there are no missing values
            while p < partition_end:
                if X[samples[p], best_feature] <= best_threshold:
                    p += 1
                else:
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    partition_end -= 1
@final
cdef class SparsePartitioner:
    """Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    # 定义稀疏分区器类，专门用于稀疏的CSC数据结构

    cdef intp_t[::1] samples
    # 声明一个一维连续内存的整型指针数组 samples

    cdef float32_t[::1] feature_values
    # 声明一个一维连续内存的单精度浮点型指针数组 feature_values

    cdef intp_t start
    # 声明一个整型变量 start，表示分区的起始位置

    cdef intp_t end
    # 声明一个整型变量 end，表示分区的结束位置

    cdef intp_t n_missing
    # 声明一个整型变量 n_missing，表示缺失值的数量

    cdef const unsigned char[::1] missing_values_in_feature_mask
    # 声明一个一维连续内存的无符号字符型常量指针数组 missing_values_in_feature_mask

    cdef const float32_t[::1] X_data
    # 声明一个一维连续内存的单精度浮点型常量指针数组 X_data

    cdef const int32_t[::1] X_indices
    # 声明一个一维连续内存的32位整型常量指针数组 X_indices

    cdef const int32_t[::1] X_indptr
    # 声明一个一维连续内存的32位整型常量指针数组 X_indptr

    cdef intp_t n_total_samples
    # 声明一个整型变量 n_total_samples，表示总样本数

    cdef intp_t[::1] index_to_samples
    # 声明一个一维连续内存的整型指针数组 index_to_samples，用于索引到样本的映射

    cdef intp_t[::1] sorted_samples
    # 声明一个一维连续内存的整型指针数组 sorted_samples，用于存储排序后的样本索引

    cdef intp_t start_positive
    # 声明一个整型变量 start_positive，表示正样本的起始位置

    cdef intp_t end_negative
    # 声明一个整型变量 end_negative，表示负样本的结束位置

    cdef bint is_samples_sorted
    # 声明一个布尔型变量 is_samples_sorted，表示样本是否已排序

    def __init__(
        self,
        object X,
        intp_t[::1] samples,
        intp_t n_samples,
        float32_t[::1] feature_values,
        const unsigned char[::1] missing_values_in_feature_mask,
    ):
        # 初始化方法，接受稀疏矩阵 X，样本索引数组 samples，样本数量 n_samples，
        # 特征值数组 feature_values，缺失值掩码数组 missing_values_in_feature_mask

        if not (issparse(X) and X.format == "csc"):
            # 检查稀疏矩阵 X 是否为 CSC 格式，如果不是则抛出 ValueError
            raise ValueError("X should be in csc format")

        self.samples = samples
        # 将参数 samples 赋值给实例变量 samples

        self.feature_values = feature_values
        # 将参数 feature_values 赋值给实例变量 feature_values

        # Initialize X
        cdef intp_t n_total_samples = X.shape[0]
        # 获取稀疏矩阵 X 的总样本数

        self.X_data = X.data
        # 将稀疏矩阵 X 的数据部分赋值给实例变量 X_data

        self.X_indices = X.indices
        # 将稀疏矩阵 X 的索引部分赋值给实例变量 X_indices

        self.X_indptr = X.indptr
        # 将稀疏矩阵 X 的指针部分赋值给实例变量 X_indptr

        self.n_total_samples = n_total_samples
        # 将总样本数赋值给实例变量 n_total_samples

        # Initialize auxiliary array used to perform split
        self.index_to_samples = np.full(n_total_samples, fill_value=-1, dtype=np.intp)
        # 初始化一个大小为总样本数的数组，用于将索引映射到样本

        self.sorted_samples = np.empty(n_samples, dtype=np.intp)
        # 初始化一个大小为样本数的空数组，用于存储排序后的样本索引

        cdef intp_t p
        # 声明整型变量 p，用于循环迭代

        for p in range(n_samples):
            # 遍历样本索引数组 samples
            self.index_to_samples[samples[p]] = p
            # 将当前索引位置 p 映射到 samples 数组中的样本索引位置

        self.missing_values_in_feature_mask = missing_values_in_feature_mask
        # 将缺失值掩码数组赋值给实例变量 missing_values_in_feature_mask

    cdef inline void init_node_split(self, intp_t start, intp_t end) noexcept nogil:
        """Initialize splitter at the beginning of node_split."""
        # 内联函数，初始化节点分割的起始和结束位置
        self.start = start
        # 设置起始位置
        self.end = end
        # 设置结束位置
        self.is_samples_sorted = 0
        # 将样本排序标志位设为假
        self.n_missing = 0
        # 将缺失值数量初始化为零

    cdef inline void sort_samples_and_feature_values(
        self, intp_t current_feature
        # 内联函数，用于排序样本和特征值
    ) noexcept nogil:
        """定义一个 C++ 风格的函数，不抛出异常且不涉及全局解锁。
        同时基于特征值进行排序。
        """
        cdef:
            float32_t[::1] feature_values = self.feature_values  # 将 self.feature_values 赋值给 feature_values
            intp_t[::1] index_to_samples = self.index_to_samples  # 将 self.index_to_samples 赋值给 index_to_samples
            intp_t[::1] samples = self.samples  # 将 self.samples 赋值给 samples

        self.extract_nnz(current_feature)
        # 对 feature_values 的正负部分进行排序
        sort(&feature_values[self.start], &samples[self.start], self.end_negative - self.start)
        if self.start_positive < self.end:
            # 如果有正部分，对其进行排序
            sort(
                &feature_values[self.start_positive],
                &samples[self.start_positive],
                self.end - self.start_positive
            )

        # 更新 index_to_samples 以反映排序后的顺序
        for p in range(self.start, self.end_negative):
            index_to_samples[samples[p]] = p
        for p in range(self.start_positive, self.end):
            index_to_samples[samples[p]] = p

        # 在 feature_values 中添加一个或两个零，如果有的话
        if self.end_negative < self.start_positive:
            self.start_positive -= 1
            feature_values[self.start_positive] = 0.

            if self.end_negative != self.start_positive:
                feature_values[self.end_negative] = 0.
                self.end_negative += 1

        # XXX: 当稀疏支持缺失值时，应设置为当前特征的缺失值数量
        self.n_missing = 0

    cdef inline void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil:
        """定义一个没有异常、无全局解锁（nogil）的Cython函数，用于查找当前特征的最小值和最大值。"""
        cdef:
            intp_t p  # 声明一个Cython整型指针变量p
            float32_t current_feature_value, min_feature_value, max_feature_value  # 声明三个Cython单精度浮点数变量
            float32_t[::1] feature_values = self.feature_values  # 声明一个Cython浮点数数组变量，并将其初始化为self.feature_values数组的视图

        self.extract_nnz(current_feature)  # 调用类中的方法extract_nnz，处理当前特征

        if self.end_negative != self.start_positive:
            # 存在零值的情况
            min_feature_value = 0  # 最小特征值设置为0
            max_feature_value = 0  # 最大特征值设置为0
        else:
            min_feature_value = feature_values[self.start]  # 从feature_values数组中获取起始位置self.start处的值作为最小特征值
            max_feature_value = min_feature_value  # 将最小特征值同时赋给最大特征值

        # 在feature_values数组的范围[self.start, self.end_negative)内查找最小值和最大值
        for p in range(self.start, self.end_negative):
            current_feature_value = feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        # 在feature_values数组的范围[self.start_positive, self.end)内更新最小值和最大值
        for p in range(self.start_positive, self.end):
            current_feature_value = feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        min_feature_value_out[0] = min_feature_value  # 将计算得到的最小特征值存入min_feature_value_out数组
        max_feature_value_out[0] = max_feature_value  # 将计算得到的最大特征值存入max_feature_value_out数组

    cdef inline void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil:
        """计算迭代特征值时的下一个p_prev和p。"""
        cdef:
            intp_t p_next  # 声明一个Cython整型指针变量p_next
            float32_t[::1] feature_values = self.feature_values  # 声明一个Cython浮点数数组变量，并将其初始化为self.feature_values数组的视图

        if p[0] + 1 != self.end_negative:
            p_next = p[0] + 1
        else:
            p_next = self.start_positive

        # 循环直到找到满足条件(feature_values[p_next] <= feature_values[p[0]] + FEATURE_THRESHOLD)的p_next
        while (p_next < self.end and
                feature_values[p_next] <= feature_values[p[0]] + FEATURE_THRESHOLD):
            p[0] = p_next
            if p[0] + 1 != self.end_negative:
                p_next = p[0] + 1
            else:
                p_next = self.start_positive

        p_prev[0] = p[0]  # 更新p_prev为当前的p
        p[0] = p_next  # 更新p为找到的p_next

    cdef inline intp_t partition_samples(self, float64_t current_threshold) noexcept nogil:
        """根据当前阈值对feature_values数组进行分区。"""
        return self._partition(current_threshold, self.start_positive)  # 调用类中的方法_partition进行分区，并返回结果

    cdef inline void partition_samples_final(
        self,
        intp_t best_pos,
        float64_t best_threshold,
        intp_t best_feature,
        intp_t n_missing,
    ) noexcept nogil:
        """对X使用最佳阈值和最佳特征进行样本分区。"""
        self.extract_nnz(best_feature)  # 处理最佳特征
        self._partition(best_threshold, best_pos)  # 使用最佳阈值和位置进行分区
    cdef inline intp_t _partition(self, float64_t threshold, intp_t zero_pos) noexcept nogil:
        """Partition samples[start:end] based on threshold."""
        # 定义变量 p 和 partition_end，分别表示当前位置和分区结束位置
        cdef:
            intp_t p, partition_end
            # 使用 Cython 的特定语法声明并初始化数组变量
            intp_t[::1] index_to_samples = self.index_to_samples
            float32_t[::1] feature_values = self.feature_values
            intp_t[::1] samples = self.samples

        # 根据阈值确定初始分区位置和结束位置
        if threshold < 0.:
            p = self.start
            partition_end = self.end_negative
        elif threshold > 0.:
            p = self.start_positive
            partition_end = self.end
        else:
            # 如果数据已经分割完成，则直接返回零点位置
            return zero_pos

        # 开始执行分区操作，直到 p >= partition_end
        while p < partition_end:
            if feature_values[p] <= threshold:
                # 如果当前特征值小于等于阈值，则继续向后移动 p
                p += 1
            else:
                # 如果当前特征值大于阈值，则将 partition_end 向前移动一位
                partition_end -= 1

                # 交换 feature_values 中 p 和 partition_end 处的值
                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p]
                )
                # 调用 sparse_swap 函数交换 index_to_samples 和 samples 中的数据
                sparse_swap(index_to_samples, samples, p, partition_end)

        # 返回最终的 partition_end，表示分区结束位置
        return partition_end
    # 定义一个内联函数，提取给定特征的非零值并进行分区操作
    # 函数是无异常、无GIL的
    cdef inline void extract_nnz(self, intp_t feature) noexcept nogil:
        """Extract and partition values for a given feature.

        The extracted values are partitioned between negative values
        feature_values[start:end_negative[0]] and positive values
        feature_values[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.

        Parameters
        ----------
        feature : intp_t,
            Index of the feature we want to extract non zero value.
        """
        # 将本地变量绑定到类成员变量或全局变量
        cdef intp_t[::1] samples = self.samples
        cdef float32_t[::1] feature_values = self.feature_values
        cdef intp_t indptr_start = self.X_indptr[feature],
        cdef intp_t indptr_end = self.X_indptr[feature + 1]
        cdef intp_t n_indices = <intp_t>(indptr_end - indptr_start)
        cdef intp_t n_samples = self.end - self.start
        cdef intp_t[::1] index_to_samples = self.index_to_samples
        cdef intp_t[::1] sorted_samples = self.sorted_samples
        cdef const int32_t[::1] X_indices = self.X_indices
        cdef const float32_t[::1] X_data = self.X_data

        # 如果使用二分搜索效率更高，根据公式计算判断条件
        if ((1 - self.is_samples_sorted) * n_samples * log(n_samples) +
                n_samples * log(n_indices) < EXTRACT_NNZ_SWITCH * n_indices):
            # 调用二分搜索方法提取非零值
            extract_nnz_binary_search(X_indices, X_data,
                                      indptr_start, indptr_end,
                                      samples, self.start, self.end,
                                      index_to_samples,
                                      feature_values,
                                      &self.end_negative, &self.start_positive,
                                      sorted_samples, &self.is_samples_sorted)

        # 否则使用基于 index_to_samples 的方法提取非零值
        else:
            extract_nnz_index_to_samples(X_indices, X_data,
                                         indptr_start, indptr_end,
                                         samples, self.start, self.end,
                                         index_to_samples,
                                         feature_values,
                                         &self.end_negative, &self.start_positive)
cdef int compare_SIZE_t(const void* a, const void* b) noexcept nogil:
    """Comparison function for sort.

    This function compares two elements pointed to by `a` and `b`.
    It subtracts the first element from the second and returns an integer result.
    This is required by the `qsort` function in the C standard library.
    """
    return <int>((<intp_t*>a)[0] - (<intp_t*>b)[0])


cdef inline void binary_search(const int32_t[::1] sorted_array,
                               int32_t start, int32_t end,
                               intp_t value, intp_t* index,
                               int32_t* new_start) noexcept nogil:
    """Return the index of `value` in `sorted_array` using binary search.

    If `value` is found in `sorted_array`, its index is stored in `index`.
    If `value` is not found, `index` remains -1.
    `new_start` is updated to the last pivot index + 1 after the search.
    """
    cdef int32_t pivot
    index[0] = -1
    while start < end:
        pivot = start + (end - start) / 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


cdef inline void extract_nnz_index_to_samples(const int32_t[::1] X_indices,
                                              const float32_t[::1] X_data,
                                              int32_t indptr_start,
                                              int32_t indptr_end,
                                              intp_t[::1] samples,
                                              intp_t start,
                                              intp_t end,
                                              intp_t[::1] index_to_samples,
                                              float32_t[::1] feature_values,
                                              intp_t* end_negative,
                                              intp_t* start_positive) noexcept nogil:
    """Extract and partition feature values and corresponding indices.

    This function iterates over `X_indices` and `X_data` within the range specified by `indptr_start` and `indptr_end`.
    It populates `feature_values` based on the conditions specified:
    - If `X_data[k] > 0`, it updates `start_positive` and swaps values using `sparse_swap`.
    - If `X_data[k] < 0`, it updates `end_negative` and swaps values using `sparse_swap`.
    `index_to_samples` is used to map indices to samples efficiently.
    """
    cdef int32_t k
    cdef intp_t index
    cdef intp_t end_negative_ = start
    cdef intp_t start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                feature_values[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                feature_values[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    # Assign final values to output variables
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_
cdef inline void extract_nnz_binary_search(const int32_t[::1] X_indices,
                                           const float32_t[::1] X_data,
                                           int32_t indptr_start,
                                           int32_t indptr_end,
                                           intp_t[::1] samples,
                                           intp_t start,
                                           intp_t end,
                                           intp_t[::1] index_to_samples,
                                           float32_t[::1] feature_values,
                                           intp_t* end_negative,
                                           intp_t* start_positive,
                                           intp_t[::1] sorted_samples,
                                           bint* is_samples_sorted) noexcept nogil:
    """Extract and partition values for a given feature using binary search.

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    """
    cdef intp_t n_samples

    # If samples are not yet sorted, sort them and mark as sorted
    if not is_samples_sorted[0]:
        n_samples = end - start
        # Copy samples to sorted_samples and sort
        memcpy(&sorted_samples[start], &samples[start],
               n_samples * sizeof(intp_t))
        qsort(&sorted_samples[start], n_samples, sizeof(intp_t),
              compare_SIZE_t)
        is_samples_sorted[0] = 1

    # Adjust indptr_start to find the starting point for the search
    while (indptr_start < indptr_end and
           sorted_samples[start] > X_indices[indptr_start]):
        indptr_start += 1

    # Adjust indptr_end to find the ending point for the search
    while (indptr_start < indptr_end and
           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
        indptr_end -= 1

    # Initialize variables for iteration
    cdef intp_t p = start
    cdef intp_t index
    cdef intp_t k
    cdef intp_t end_negative_ = start
    cdef intp_t start_positive_ = end

    # Iterate over sorted_samples and X_indices to partition values
    while (p < end and indptr_start < indptr_end):
        # Find index of sorted_samples[p] in X_indices using binary search
        binary_search(X_indices, indptr_start, indptr_end,
                      sorted_samples[p], &k, &indptr_start)

        if k != -1:
            # If k != -1, we have found a non-zero value

            if X_data[k] > 0:
                # Positive value found, move it to start_positive_
                start_positive_ -= 1
                feature_values[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                # Negative value found, move it to end_negative_
                feature_values[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    # Returned values: end_negative and start_positive
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void sparse_swap(intp_t[::1] index_to_samples, intp_t[::1] samples,
                             intp_t pos_1, intp_t pos_2) noexcept nogil:
    """Swap sample pos_1 and pos_2 preserving sparse invariant."""
    # 将 samples 中位置 pos_1 和 pos_2 的样本进行交换，保持稀疏不变性
    samples[pos_1], samples[pos_2] = samples[pos_2], samples[pos_1]
    # 更新 index_to_samples 字典，更新 pos_1 和 pos_2 的位置映射
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2
cdef class BestSplitter(Splitter):
    """Splitter for finding the best split on dense data."""

    # 密集数据的最佳分割器的初始化方法
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        # 调用父类 Splitter 的初始化方法，传入参数
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        # 使用密集数据分割器初始化 self.partitioner
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    # 节点分割方法，用于密集数据的最佳分割
    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        # 调用 node_split_best 函数执行最佳分割
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class BestSparseSplitter(Splitter):
    """Splitter for finding the best split, using the sparse data."""

    # 稀疏数据的最佳分割器的初始化方法
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        # 调用父类 Splitter 的初始化方法，传入参数
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        # 使用稀疏数据分割器初始化 self.partitioner
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )

    # 节点分割方法，用于稀疏数据的最佳分割
    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        # 调用 node_split_best 函数执行最佳分割
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSplitter(Splitter):
    """Splitter for finding the best random split on dense data."""

    # 密集数据的随机分割器的初始化方法
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        # 调用父类 Splitter 的初始化方法，传入参数
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        # 使用密集数据分割器初始化 self.partitioner
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    # 节点分割方法，用于密集数据的随机分割
    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        # 调用 node_split_random 函数执行随机分割
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSparseSplitter(Splitter):
    """Splitter for finding the best random split, using the sparse data."""

    # 稀疏数据的随机分割器的初始化方法
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1:
        # 调用父类 Splitter 的初始化方法，传入参数
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        # 使用稀疏数据分割器初始化 self.partitioner
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )
    # 尝试执行以下代码块，如果返回错误码 -1，则执行异常处理
    ) except -1:
        # 调用 Splitter 类的 init 方法，初始化决策树节点分裂所需的数据结构和参数
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        # 创建 SparsePartitioner 对象，用于稀疏数据集的分区管理
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )
    # 定义一个 C 语言风格的函数，用于执行节点的随机分裂操作
    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        # 调用 node_split_random 函数，进行节点的随机分裂操作，并返回结果
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )
```