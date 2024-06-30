# `D:\src\scipysrc\scikit-learn\sklearn\tree\_splitter.pxd`

```
# 从 _splitter.pyx 文件导入详细信息。
from ._criterion cimport Criterion
from ._tree cimport ParentInfo

# 从 ..utils._typedefs 中导入特定数据类型，包括 float32_t, float64_t, intp_t, int8_t, int32_t, uint32_t

cdef struct SplitRecord:
    # 用于跟踪样本分割的数据
    intp_t feature         # 要进行分割的特征。
    intp_t pos             # 在给定位置分割样本数组，
                           # 即在特征阈值下方的样本数量。
                           # 如果节点是叶子节点，则 pos >= end。
    float64_t threshold    # 分割的阈值。
    float64_t improvement  # 相对于父节点的不纯度改善。
    float64_t impurity_left   # 左侧分割的不纯度。
    float64_t impurity_right  # 右侧分割的不纯度。
    float64_t lower_bound     # 对于单调性的两个子节点的值的下界。
    float64_t upper_bound     # 对于单调性的两个子节点的值的上界。
    unsigned char missing_go_to_left  # 控制缺失值是否进入左侧节点。
    intp_t n_missing        # 被分割特征上的缺失值数量。

cdef class Splitter:
    # Splitter 在输入空间中搜索特征和阈值来分割样本 samples[start:end]。
    #
    # 不纯度计算由 Criterion 对象处理。

    # 内部结构
    cdef public Criterion criterion      # 不纯度准则
    cdef public intp_t max_features      # 测试的特征数量上限
    cdef public intp_t min_samples_leaf  # 叶子节点的最小样本数
    cdef public float64_t min_weight_leaf   # 叶子节点的最小权重

    cdef object random_state             # 随机状态
    cdef uint32_t rand_r_state           # sklearn_rand_r 随机数状态

    cdef intp_t[::1] samples             # X, y 中的样本索引
    cdef intp_t n_samples                # X.shape[0]
    cdef float64_t weighted_n_samples    # 加权样本数
    cdef intp_t[::1] features            # X 中的特征索引
    cdef intp_t[::1] constant_features   # 常数特征的索引
    cdef intp_t n_features               # X.shape[1]
    cdef float32_t[::1] feature_values   # 临时数组，保存特征值

    cdef intp_t start                    # 当前节点的起始位置
    cdef intp_t end                      # 当前节点的结束位置

    cdef const float64_t[:, ::1] y
    # 每个特征的单调性约束。
    # 编码如下：
    #   -1: 单调递减
    #    0: 无约束
    #   +1: 单调递增
    cdef const int8_t[:] monotonic_cst
    cdef bint with_monotonic_cst
    cdef const float64_t[:] sample_weight

    # Splitter 对象维护的样本向量 `samples`，使得节点中包含的样本是连续的。通过这种设置，
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.
    # `node_split` 方法重新组织节点样本 `samples[start:end]`，分成两个子集 `samples[start:pos]` 和 `samples[pos:end]`。

    # The 1-d `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.
    # 大小为 n_features 的一维 `features` 数组包含特征索引，允许快速无重复采样特征。

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.
    # 大小为 n_features 的一维 `constant_features` 数组在 `constant_features[:n_constant_features]` 中保存了具有恒定值的特征标识符，
    # 对于所有到达特定节点的样本。`n_constant_features` 的值由父节点传递给其子节点。
    # 范围 `[n_constant_features:]` 的内容未定义，但为了性能原因预先分配。
    # 这允许基于深度的树构建进行优化。

    # Methods
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
    ) except -1
    # `init` 方法初始化对象，接受 X、y、sample_weight 和 missing_values_in_feature_mask 参数。

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil
    # `node_reset` 方法重置节点，处理从 start 到 end 的范围内的节点数据，并使用 weighted_n_node_samples 参数。

    cdef int node_split(
        self,
        ParentInfo* parent,
        SplitRecord* split,
    ) except -1 nogil
    # `node_split` 方法实现节点分裂操作，接受 ParentInfo 和 SplitRecord 参数。

    cdef void node_value(self, float64_t* dest) noexcept nogil
    # `node_value` 方法获取节点的值，并将结果存储到 dest 中，使用 noexcept 关键字表明不会抛出异常，使用 nogil 关键字表明没有 GIL。

    cdef void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil
    # `clip_node_value` 方法裁剪节点值，将节点值限制在 lower_bound 和 upper_bound 之间，使用 noexcept 和 nogil 优化性能。

    cdef float64_t node_impurity(self) noexcept nogil
    # `node_impurity` 方法计算节点的不纯度，并返回浮点数结果，使用 noexcept 和 nogil 优化性能。
```