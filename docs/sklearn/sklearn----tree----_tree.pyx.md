# `D:\src\scipysrc\scikit-learn\sklearn\tree\_tree.pyx`

```
# 导入必要的 C 语言库和 Python 模块
from cpython cimport Py_INCREF, PyObject, PyTypeObject  # 导入 CPython 的相关库
from libc.stdlib cimport free  # 导入 C 标准库中的 free 函数
from libc.string cimport memcpy, memset  # 导入 C 标准库中的内存操作函数 memcpy 和 memset
from libc.stdint cimport INTPTR_MAX  # 导入 C 标准库中的整数类型极限值 INTPTR_MAX
from libc.math cimport isnan  # 导入 C 标准库中的浮点数判断是否为 NaN 的函数
from libcpp.vector cimport vector  # 导入 C++ 标准库中的 vector 容器
from libcpp.algorithm cimport pop_heap, push_heap  # 导入 C++ 标准库中的堆操作函数 pop_heap 和 push_heap
from libcpp.stack cimport stack  # 导入 C++ 标准库中的 stack 容器
from libcpp cimport bool  # 导入 C++ 中的布尔类型

import struct  # 导入 Python 标准库中的 struct 模块

import numpy as np  # 导入 NumPy 库，并使用 np 别名
cimport numpy as cnp  # 使用 cnp 别名导入 NumPy 库

cnp.import_array()  # 导入 NumPy 数组的 C 接口

from scipy.sparse import issparse  # 从 SciPy 稀疏矩阵模块中导入 issparse 函数
from scipy.sparse import csr_matrix  # 从 SciPy 稀疏矩阵模块中导入 csr_matrix 类型

from ._utils cimport safe_realloc  # 从当前包中的 _utils 模块导入 safe_realloc 函数
from ._utils cimport sizet_ptr_to_ndarray  # 从当前包中的 _utils 模块导入 sizet_ptr_to_ndarray 函数

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE  # 从 NumPy 中导入 float32 类型，并使用 DTYPE 别名
from numpy import float64 as DOUBLE  # 从 NumPy 中导入 float64 类型，并使用 DOUBLE 别名

cdef float64_t INFINITY = np.inf  # 定义 C 语言级别的 float64_t 类型常量 INFINITY，值为正无穷大
cdef float64_t EPSILON = np.finfo('double').eps  # 定义 C 语言级别的 float64_t 类型常量 EPSILON，值为 double 类型的最小精度值

# Some handy constants (BestFirstTreeBuilder)
cdef bint IS_FIRST = 1  # 定义 C 语言级别的 bint 类型常量 IS_FIRST，值为 1，表示是第一个
cdef bint IS_NOT_FIRST = 0  # 定义 C 语言级别的 bint 类型常量 IS_NOT_FIRST，值为 0，表示不是第一个
cdef bint IS_LEFT = 1  # 定义 C 语言级别的 bint 类型常量 IS_LEFT，值为 1，表示是左侧
cdef bint IS_NOT_LEFT = 0  # 定义 C 语言级别的 bint 类型常量 IS_NOT_LEFT，值为 0，表示不是左侧

TREE_LEAF = -1  # Python 级别的常量 TREE_LEAF，表示树节点是叶子节点的标识
TREE_UNDEFINED = -2  # Python 级别的常量 TREE_UNDEFINED，表示树节点未定义的标识
cdef intp_t _TREE_LEAF = TREE_LEAF  # 定义 C 语言级别的 intp_t 类型常量 _TREE_LEAF，值为 TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED  # 定义 C 语言级别的 intp_t 类型常量 _TREE_UNDEFINED，值为 TREE_UNDEFINED

# Build the corresponding numpy dtype for Node.
# This works by casting `dummy` to an array of Node of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Node dummy  # 声明 C 语言级别的 Node 类型变量 dummy

# 创建 NODE_DTYPE，通过将 dummy 强制转换为长度为 1 的 Node 数组来构建相应的 NumPy dtype
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

cdef inline void _init_parent_record(ParentInfo* record) noexcept nogil:
    # 初始化父节点信息结构体的函数，不抛出异常，不涉及 Python 全局解释器锁（GIL）
    record.n_constant_features = 0  # 设置常量特征数为 0
    record.impurity = INFINITY  # 设置不纯度为正无穷大
    record.lower_bound = -INFINITY  # 设置下界为负无穷大
    record.upper_bound = INFINITY  # 设置上界为正无穷大

# =============================================================================
# TreeBuilder
# =============================================================================

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const unsigned char[::1] missing_values_in_feature_mask=None,
    ):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
    ):
        """
        检查输入数据的数据类型、布局和格式
        """
        if issparse(X):
            # 如果输入是稀疏矩阵，转换为压缩列稀疏矩阵并排序索引
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                # 如果数据的数据类型不是指定的DTYPE，则转换为指定的数据类型
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                # 如果索引不是np.int32类型，则抛出异常，不支持np.int64类型的稀疏矩阵
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # 如果输入不是稀疏矩阵且数据类型不是指定的DTYPE，则转换为指定的数据类型和Fortran格式
            X = np.asfortranarray(X, dtype=DTYPE)

        # TODO: 这里对y的检查似乎是多余的，因为在BaseDecisionTree的fit方法中也有相同的检查，可以移除。
        if y.base.dtype != DOUBLE or not y.base.flags.contiguous:
            # 如果y的数据类型不是DOUBLE或者不是连续存储，则转换为连续存储的DOUBLE类型
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (
            sample_weight is not None and
            (
                sample_weight.base.dtype != DOUBLE or
                not sample_weight.base.flags.contiguous
            )
        ):
            # 如果sample_weight不为None且其数据类型不是DOUBLE或者不是连续存储，则转换为指定的DOUBLE类型和C顺序
            sample_weight = np.asarray(sample_weight, dtype=DOUBLE, order="C")

        return X, y, sample_weight
# Depth first builder ---------------------------------------------------------
# 深度优先树生成器
cdef struct StackRecord:
    intp_t start  # 节点起始索引
    intp_t end  # 节点结束索引
    intp_t depth  # 节点深度
    intp_t parent  # 父节点索引
    bint is_left  # 是否为左子节点
    float64_t impurity  # 节点不纯度
    intp_t n_constant_features  # 常量特征数量
    float64_t lower_bound  # 下界
    float64_t upper_bound  # 上界

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease):
        self.splitter = splitter  # 分裂器
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.min_samples_leaf = min_samples_leaf  # 最小叶子样本数
        self.min_weight_leaf = min_weight_leaf  # 最小叶子权重
        self.max_depth = max_depth  # 最大深度
        self.min_impurity_decrease = min_impurity_decrease  # 最小不纯度减少量

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const unsigned char[::1] missing_values_in_feature_mask=None,
    ):
        # 构建决策树
        pass

# Best first builder ----------------------------------------------------------
cdef struct FrontierRecord:
    # 前沿记录，记录一个节点的信息，用于优先级队列以便按照提升不纯度的大小贪婪地生长树
    intp_t node_id  # 节点 ID
    intp_t start  # 节点起始索引
    intp_t end  # 节点结束索引
    intp_t pos  # 位置
    intp_t depth  # 深度
    bint is_leaf  # 是否为叶子节点
    float64_t impurity  # 节点不纯度
    float64_t impurity_left  # 左子节点不纯度
    float64_t impurity_right  # 右子节点不纯度
    float64_t improvement  # 提升值
    float64_t lower_bound  # 下界
    float64_t upper_bound  # 上界
    float64_t middle_value  # 中间值

cdef inline bool _compare_records(
    const FrontierRecord& left,
    const FrontierRecord& right,
):
    # 比较两个前沿记录的提升值
    return left.improvement < right.improvement

cdef inline void _add_to_frontier(
    FrontierRecord rec,
    vector[FrontierRecord]& frontier,
) noexcept nogil:
    """Adds record `rec` to the priority queue `frontier`."""
    # 将记录 `rec` 添加到优先级队列 `frontier` 中
    frontier.push_back(rec)
    push_heap(frontier.begin(), frontier.end(), &_compare_records)


cdef class BestFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.
    """
    cdef intp_t max_leaf_nodes  # 最大叶子节点数
    # 定义构造函数，初始化决策树拆分器、最小分裂样本数、最小叶子节点样本数、最小叶子节点权重、
    # 最大深度、最大叶子节点数、最小不纯度减少值
    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf,  min_weight_leaf,
                  intp_t max_depth, intp_t max_leaf_nodes,
                  float64_t min_impurity_decrease):
        # 将参数赋值给实例变量
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    # 定义构建方法，用于构建决策树
    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const unsigned char[::1] missing_values_in_feature_mask=None,
    ):
        # 内部方法：添加分裂节点
        cdef inline int _add_split_node(
            self,
            Splitter splitter,
            Tree tree,
            intp_t start,
            intp_t end,
            bint is_first,
            bint is_left,
            Node* parent,
            intp_t depth,
            ParentInfo* parent_record,
            FrontierRecord* res
        ):
# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : intp_t
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : intp_t
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : intp_t
        The depth of the tree, i.e. the maximum depth of its leaves.

    children_left : array of intp_t, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of intp_t, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    n_leaves : intp_t
        Number of leaves in the tree.

    feature : array of intp_t, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of float64_t, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of float64_t, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of float64_t, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of intp_t, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of float64_t, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.

    missing_go_to_left : array of bool, shape [node_count]
        missing_go_to_left[i] holds a bool indicating whether or not there were
        missing values at node i.
    """
    
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    @property
    # This property wraps access to internal arrays `nodes` and `value`, 
    # warning about the need to avoid freeing them during subsequent memory allocations.
    # 返回一个将指针转换为 NumPy 数组后的类的输出类别数
    def n_classes(self):
        return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    @property
    # 返回左子节点的数组，该数组长度为节点总数
    def children_left(self):
        return self._get_node_ndarray()['left_child'][:self.node_count]

    @property
    # 返回右子节点的数组，该数组长度为节点总数
    def children_right(self):
        return self._get_node_ndarray()['right_child'][:self.node_count]

    @property
    # 返回叶子节点的数量
    def n_leaves(self):
        return np.sum(np.logical_and(
            self.children_left == -1,
            self.children_right == -1))

    @property
    # 返回节点对应的特征索引数组，长度为节点总数
    def feature(self):
        return self._get_node_ndarray()['feature'][:self.node_count]

    @property
    # 返回节点对应的阈值数组，长度为节点总数
    def threshold(self):
        return self._get_node_ndarray()['threshold'][:self.node_count]

    @property
    # 返回节点对应的不纯度数组，长度为节点总数
    def impurity(self):
        return self._get_node_ndarray()['impurity'][:self.node_count]

    @property
    # 返回节点对应的样本数数组，长度为节点总数
    def n_node_samples(self):
        return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    @property
    # 返回节点对应的加权样本数数组，长度为节点总数
    def weighted_n_node_samples(self):
        return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    @property
    # 返回节点对应的缺失值处理方式数组，长度为节点总数
    def missing_go_to_left(self):
        return self._get_node_ndarray()['missing_go_to_left'][:self.node_count]

    @property
    # 返回节点对应的值数组，长度为节点总数
    def value(self):
        return self._get_value_ndarray()[:self.node_count]

    # TODO: Convert n_classes to cython.integral memory view once
    #  https://github.com/cython/cython/issues/5243 is fixed
    def __cinit__(self, intp_t n_features, cnp.ndarray n_classes, intp_t n_outputs):
        """Constructor."""
        cdef intp_t dummy = 0
        size_t_dtype = np.array(dummy).dtype

        # 检查并调整类别数组的数据类型
        n_classes = _check_n_classes(n_classes, size_t_dtype)

        # 初始化输入输出布局
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        # 计算类别数的最大值
        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef intp_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # 初始化内部结构
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # 释放所有内部结构的内存
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        # 用于序列化的 reduce 方法的实现
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())
    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        # 初始化一个空字典，用于存储对象状态
        d = {}
        # 将最大深度存入状态字典
        d["max_depth"] = self.max_depth
        # 将节点计数存入状态字典
        d["node_count"] = self.node_count
        # 获取节点数组并存入状态字典
        d["nodes"] = self._get_node_ndarray()
        # 获取值数组并存入状态字典
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        # 从状态字典中恢复最大深度
        self.max_depth = d["max_depth"]
        # 从状态字典中恢复节点计数
        self.node_count = d["node_count"]

        # 如果状态字典中没有节点数组，抛出错误
        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        # 从状态字典中获取节点数组
        node_ndarray = d['nodes']
        # 从状态字典中获取值数组
        value_ndarray = d['values']

        # 定义值数组的期望形状
        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)

        # 检查节点数组的数据类型并转换为期望的数据类型
        node_ndarray = _check_node_ndarray(node_ndarray, expected_dtype=NODE_DTYPE)
        # 检查值数组的数据类型和形状并转换为期望的数据类型和形状
        value_ndarray = _check_value_ndarray(
            value_ndarray,
            expected_dtype=np.dtype(np.float64),
            expected_shape=value_shape
        )

        # 设置对象的容量为节点数组的长度
        self.capacity = node_ndarray.shape[0]
        # 如果调整内部结构大小失败，则抛出内存错误
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)

        # 将节点数组的数据复制到对象的节点数组中
        memcpy(self.nodes, cnp.PyArray_DATA(node_ndarray),
               self.capacity * sizeof(Node))
        # 将值数组的数据复制到对象的值数组中
        memcpy(self.value, cnp.PyArray_DATA(value_ndarray),
               self.capacity * self.value_stride * sizeof(float64_t))

    cdef int _resize(self, intp_t capacity) except -1 nogil:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 如果调用C函数调整容量失败，则通过GIL抛出内存错误
        if self._resize_c(capacity) != 0:
            # 只有在需要时才获取全局解释器锁（GIL）
            with gil:
                raise MemoryError()
    cdef int _resize_c(self, intp_t capacity=INTPTR_MAX) except -1 nogil:
        """Guts of _resize

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # 如果请求的容量与当前容量相同且节点数组不为空，则无需调整
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        # 如果请求的容量是最大整数值，则根据当前容量决定新的容量大小
        if capacity == INTPTR_MAX:
            if self.capacity == 0:
                capacity = 3  # 默认初始值
            else:
                capacity = 2 * self.capacity

        # 调整节点数组的大小
        safe_realloc(&self.nodes, capacity)
        # 调整值数组的大小
        safe_realloc(&self.value, capacity * self.value_stride)

        # 如果新的容量大于当前容量，则初始化新增部分的值数组为0，以便后续使用分类器的argmax
        memset(<void*>(self.value + self.capacity * self.value_stride), 0,
               (capacity - self.capacity) * self.value_stride *
               sizeof(float64_t))
        # 初始化新增部分的节点数组为0，以保证序列化时的确定性（Node 结构体的填充）
        memset(<void*>(self.nodes + self.capacity), 0, (capacity - self.capacity) * sizeof(Node))

        # 如果请求的容量小于节点计数，调整节点计数
        if capacity < self.node_count:
            self.node_count = capacity

        # 更新当前容量
        self.capacity = capacity
        return 0


    cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
                          intp_t feature, float64_t threshold, float64_t impurity,
                          intp_t n_node_samples,
                          float64_t weighted_n_node_samples,
                          unsigned char missing_go_to_left) except -1 nogil:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        # 新增一个节点到树中

        # 获取当前节点的ID
        cdef intp_t node_id = self.node_count

        # 如果节点ID超过了当前容量，则进行扩容
        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return INTPTR_MAX

        # 获取新节点的指针
        cdef Node* node = &self.nodes[node_id]
        # 设置新节点的属性
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        # 如果有父节点，则将当前节点注册为父节点的子节点
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        # 如果是叶子节点，则设置左右子节点为叶子标记，并清空特征和阈值
        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # 如果不是叶子节点，设置节点的特征、阈值和是否向左移动
            # 左右子节点将在后续设置
            node.feature = feature
            node.threshold = threshold
            node.missing_go_to_left = missing_go_to_left

        # 增加节点计数
        self.node_count += 1

        # 返回新节点的ID
        return node_id
    # 使用 Cython 声明的 cpdef，可在 Python 和 Cython 中调用的方法，返回预测结果
    cpdef cnp.ndarray predict(self, object X):
        """Predict target for X."""
        # 调用内部方法获取节点值的 ndarray，并在指定轴上根据应用结果进行取值，使用 clip 模式
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        # 如果只有一个输出，将 out 重新整形为 (样本数, 最大类别数)
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    # 使用 Cython 声明的 cpdef，可在 Python 和 Cython 中调用的方法，根据输入 X 查找每个样本的叶子节点
    cpdef cnp.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        # 如果 X 是稀疏矩阵，调用稀疏 CSR 格式的应用方法
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            # 否则调用密集格式的应用方法
            return self._apply_dense(X)

    # 使用 Cython 声明的 cdef 内联方法，用于处理密集格式的输入 X
    cdef inline cnp.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # 检查输入是否为 np.ndarray
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        # 检查输入的数据类型是否为预期的 DTYPE（np.float32）
        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # 提取输入数据
        cdef const float32_t[:, :] X_ndarray = X
        cdef intp_t n_samples = X.shape[0]
        cdef float32_t X_i_node_feature

        # 初始化输出数组
        cdef intp_t[:] out = np.zeros(n_samples, dtype=np.intp)

        # 初始化辅助数据结构
        cdef Node* node = NULL
        cdef intp_t i = 0

        # 使用 nogil 块，允许在没有 GIL（全局解释器锁）的情况下执行并行化的代码
        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # 当节点不是叶子节点时循环
                while node.left_child != _TREE_LEAF:
                    X_i_node_feature = X_ndarray[i, node.feature]
                    # 如果节点的特征值是 NaN
                    if isnan(X_i_node_feature):
                        # 根据节点的 missing_go_to_left 属性决定向左还是向右移动
                        if node.missing_go_to_left:
                            node = &self.nodes[node.left_child]
                        else:
                            node = &self.nodes[node.right_child]
                    elif X_i_node_feature <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # 将最终到达的叶子节点的偏移量作为输出结果
                out[i] = <intp_t>(node - self.nodes)  # node offset

        # 将输出数组转换为 np.ndarray 格式并返回
        return np.asarray(out)
    # 使用 Cython 定义内联函数，处理稀疏的 CSR 格式输入 X，找到每个样本的叶节点（终端区域）。
    cdef inline cnp.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.
        """
        # 检查输入
        if not (issparse(X) and X.format == 'csr'):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # 提取输入数据
        cdef const float32_t[:] X_data = X.data  # 提取稀疏矩阵的数据部分
        cdef const int32_t[:] X_indices = X.indices  # 提取稀疏矩阵的索引部分
        cdef const int32_t[:] X_indptr = X.indptr  # 提取稀疏矩阵的指针部分

        cdef intp_t n_samples = X.shape[0]  # 样本数
        cdef intp_t n_features = X.shape[1]  # 特征数

        # 初始化输出数组
        cdef intp_t[:] out = np.zeros(n_samples, dtype=np.intp)

        # 初始化辅助数据结构
        cdef float32_t feature_value = 0.  # 特征值
        cdef Node* node = NULL  # 节点指针
        cdef float32_t* X_sample = NULL  # 样本数据指针
        cdef intp_t i = 0  # 循环变量
        cdef int32_t k = 0  # 循环变量

        # feature_to_sample 是一个数据结构，记录每个特征最后出现的样本索引，
        # 在功能上，它是一种有效的方法来识别当前样本中非零特征。
        cdef intp_t* feature_to_sample = NULL  # 特征到样本的映射数组

        # 分配内存并检查安全性
        safe_realloc(&X_sample, n_features)
        safe_realloc(&feature_to_sample, n_features)

        # 使用 nogil 上下文以释放全局解释器锁（GIL）
        with nogil:
            # 将 feature_to_sample 初始化为 -1
            memset(feature_to_sample, -1, n_features * sizeof(intp_t))

            # 遍历每个样本
            for i in range(n_samples):
                node = self.nodes  # 将节点指针指向树的根节点

                # 遍历当前样本的特征值
                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i  # 记录当前特征最后出现的样本索引
                    X_sample[X_indices[k]] = X_data[k]  # 提取当前特征的值作为样本数据

                # 当节点不是叶节点时持续循环
                while node.left_child != _TREE_LEAF:
                    # 如果当前节点不是叶节点
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]
                    else:
                        feature_value = 0.

                    # 根据特征值与节点阈值的比较选择下一个节点
                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out[i] = <intp_t>(node - self.nodes)  # 记录样本所达到的叶节点偏移量

            # 释放辅助数组内存
            free(X_sample)
            free(feature_to_sample)

        return np.asarray(out)  # 返回结果数组转换为 NumPy 数组形式

    # Cython 对外暴露的方法，根据输入的 X 类型（稀疏或密集），调用相应的决策路径计算方法
    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        if issparse(X):
            return self._decision_path_sparse_csr(X)  # 调用稀疏矩阵的决策路径计算方法
        else:
            return self._decision_path_dense(X)  # 调用密集矩阵的决策路径计算方法
    # 定义一个内联函数用于计算每个样本在决策树中的决策路径（即节点）。
    cdef inline object _decision_path_dense(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # 检查输入是否为 numpy 数组
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        # 检查输入数组的数据类型是否为指定的 DTYPE（np.float32）
        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # 将输入转换为 C 语言风格的二维 np.ndarray
        cdef const float32_t[:, :] X_ndarray = X
        # 获取样本数
        cdef intp_t n_samples = X.shape[0]

        # 初始化输出的索引指针数组，长度为样本数加一
        cdef intp_t[:] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        # 初始化输出的节点索引数组，预计分配足够的空间
        cdef intp_t[:] indices = np.zeros(
            n_samples * (1 + self.max_depth), dtype=np.intp
        )

        # 初始化辅助数据结构，包括节点指针和迭代器 i
        cdef Node* node = NULL
        cdef intp_t i = 0

        # 使用 nogil 块并行化循环，以提高性能
        with nogil:
            for i in range(n_samples):
                # 从根节点开始遍历
                node = self.nodes
                # 设置当前样本的起始索引
                indptr[i + 1] = indptr[i]

                # 添加所有的外部节点到索引数组
                while node.left_child != _TREE_LEAF:
                    # 当左右子节点都不为叶子节点时继续
                    indices[indptr[i + 1]] = <intp_t>(node - self.nodes)
                    indptr[i + 1] += 1

                    # 根据节点特征和阈值判断下一个节点的位置
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # 添加叶子节点到索引数组
                indices[indptr[i + 1]] = <intp_t>(node - self.nodes)
                indptr[i + 1] += 1

        # 截取有效长度的索引数组
        indices = indices[:indptr[n_samples]]
        # 初始化数据数组，所有值为 1
        cdef intp_t[:] data = np.ones(shape=len(indices), dtype=np.intp)
        # 创建压缩稀疏行矩阵，表示每个样本的节点路径
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    # 定义一个公共内联函数，用于计算树中每个节点的深度
    cpdef compute_node_depths(self):
        """Compute the depth of each node in a tree.

        .. versionadded:: 1.3

        Returns
        -------
        depths : ndarray of shape (self.node_count,), dtype=np.int64
            The depth of each node in the tree.
        """
        # 定义 C 语言风格的局部变量
        cdef:
            cnp.int64_t[::1] depths = np.empty(self.node_count, dtype=np.int64)
            cnp.npy_intp[:] children_left = self.children_left
            cnp.npy_intp[:] children_right = self.children_right
            cnp.npy_intp node_id
            cnp.npy_intp node_count = self.node_count
            cnp.int64_t depth

        # 初始化根节点的深度为 1
        depths[0] = 1  # init root node
        # 遍历所有节点，计算每个节点的深度
        for node_id in range(node_count):
            if children_left[node_id] != _TREE_LEAF:
                # 如果当前节点有左子节点，计算左右子节点的深度
                depth = depths[node_id] + 1
                depths[children_left[node_id]] = depth
                depths[children_right[node_id]] = depth

        return depths.base
    # 定义一个 Cython 的 cpdef 函数，计算每个特征（变量）的重要性
    def compute_feature_importances(self, normalize=True):
        # 声明指向 Node 结构体的指针
        cdef Node* left
        cdef Node* right
        # 获取 self 对象中的 nodes 数组的指针
        cdef Node* nodes = self.nodes
        # 初始化 node 指针，指向 nodes 数组的第一个元素
        cdef Node* node = nodes
        # end_node 指向 nodes 数组的尾部
        cdef Node* end_node = node + self.node_count

        # 初始化一个浮点数 normalizer
        cdef float64_t normalizer = 0.

        # 创建一个大小为 self.n_features 的浮点数数组 importances，用于存储特征重要性
        cdef cnp.float64_t[:] importances = np.zeros(self.n_features)

        # 使用 nogil 关键字进行无 GIL（全局解释器锁）区域的并行处理
        with nogil:
            # 遍历所有节点
            while node != end_node:
                # 如果当前节点不是叶子节点
                if node.left_child != _TREE_LEAF:
                    # 获取当前节点的左右子节点的指针
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    # 计算当前节点的特征重要性
                    importances[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                # 移动到下一个节点
                node += 1

        # 对每个特征的重要性进行归一化处理
        for i in range(self.n_features):
            importances[i] /= nodes[0].weighted_n_node_samples

        # 如果指定进行归一化
        if normalize:
            # 计算所有特征重要性的总和
            normalizer = np.sum(importances)

            # 如果总和大于 0，避免除以零的情况（例如根节点为纯净节点）
            if normalizer > 0.0:
                # 对每个特征的重要性进行最终的归一化处理
                for i in range(self.n_features):
                    importances[i] /= normalizer

        # 返回特征重要性数组作为 NumPy 数组
        return np.asarray(importances)

    # 定义一个 Cython 的 cdef 函数，返回节点值作为一个 3 维 NumPy 数组
    def _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        # 定义一个用于存储数组形状的数组 shape
        cdef cnp.npy_intp shape[3]
        shape[0] = <cnp.npy_intp> self.node_count
        shape[1] = <cnp.npy_intp> self.n_outputs
        shape[2] = <cnp.npy_intp> self.max_n_classes
        # 声明一个新的 NumPy 数组 arr，将 self.value 作为数据源，并设置类型为 NPY_DOUBLE
        cdef cnp.ndarray arr
        arr = cnp.PyArray_SimpleNewFromData(3, shape, cnp.NPY_DOUBLE, self.value)
        # 增加对 self 对象的引用计数
        Py_INCREF(self)
        # 将 arr 与 self 关联起来，使其管理底层内存
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    # 定义一个 Cython 的 cdef 函数，返回节点作为一个 NumPy 结构化数组
    def _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        # 定义一个数组 shape，用于存储结构化数组的形状
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.node_count
        # 定义一个数组 strides，用于定义结构化数组的步幅
        cdef cnp.npy_intp strides[1]
        strides[0] = sizeof(Node)
        # 声明一个新的结构化 NumPy 数组 arr，使用 self.nodes 作为数据源，并设置类型为 NODE_DTYPE
        cdef cnp.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> cnp.ndarray,
                                   <cnp.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   cnp.NPY_ARRAY_DEFAULT, None)
        # 增加对 self 对象的引用计数
        Py_INCREF(self)
        # 将 arr 与 self 关联起来，使其管理底层内存
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr
# 检查 n_classes 的维度是否为1，如果不是则引发 ValueError 异常
def _check_n_classes(n_classes, expected_dtype):
    if n_classes.ndim != 1:
        raise ValueError(
            f"Wrong dimensions for n_classes from the pickle: "
            f"expected 1, got {n_classes.ndim}"
        )

    # 检查 n_classes 的数据类型是否符合期望的数据类型，如果符合则返回 n_classes
    if n_classes.dtype == expected_dtype:
        return n_classes

    # 处理不同的字节顺序或位数不同的情况，尝试将 n_classes 转换为期望的数据类型
    if n_classes.dtype.kind == "i" and n_classes.dtype.itemsize in [4, 8]:
        return n_classes.astype(expected_dtype, casting="same_kind")

    # 如果无法转换，引发 ValueError 异常
    raise ValueError(
        "n_classes from the pickle has an incompatible dtype:\n"
        f"- expected: {expected_dtype}\n"
        f"- got:      {n_classes.dtype}"
    )


# 检查 value_ndarray 的形状是否与期望的形状相同，不同则引发 ValueError 异常
def _check_value_ndarray(value_ndarray, expected_dtype, expected_shape):
    if value_ndarray.shape != expected_shape:
        raise ValueError(
            "Wrong shape for value array from the pickle: "
            f"expected {expected_shape}, got {value_ndarray.shape}"
        )

    # 检查 value_ndarray 是否为 C 连续数组，如果不是则引发 ValueError 异常
    if not value_ndarray.flags.c_contiguous:
        raise ValueError(
            "value array from the pickle should be a C-contiguous array"
        )

    # 检查 value_ndarray 的数据类型是否符合期望的数据类型，如果符合则返回 value_ndarray
    if value_ndarray.dtype == expected_dtype:
        return value_ndarray

    # 处理不同字节顺序的情况，尝试将 value_ndarray 转换为期望的数据类型
    if value_ndarray.dtype.str.endswith('f8'):
        return value_ndarray.astype(expected_dtype, casting='equiv')

    # 如果无法转换，引发 ValueError 异常
    raise ValueError(
        "value array from the pickle has an incompatible dtype:\n"
        f"- expected: {expected_dtype}\n"
        f"- got:      {value_ndarray.dtype}"
    )


# 将 dtype 对象转换为其字段名称及对应的字符串格式字典
def _dtype_to_dict(dtype):
    return {name: dt.str for name, (dt, *rest) in dtype.fields.items()}


# 根据当前平台的位数调整 dtype_dict 中特定字段的位数，以应对不同的部署场景
def _dtype_dict_with_modified_bitness(dtype_dict):
    # 定义 Node 结构体中使用 intp_t 类型的字段名称（参见 sklearn/tree/_tree.pxd）
    indexing_field_names = ["left_child", "right_child", "feature", "n_node_samples"]

    # 计算当前平台指针大小的字符串表示
    expected_dtype_size = str(struct.calcsize("P"))
    # 根据平台位数调整允许的 dtype 字典，使其符合当前平台的需求
    allowed_dtype_size = "8" if expected_dtype_size == "4" else "4"

    allowed_dtype_dict = dtype_dict.copy()
    # 遍历字段名称列表，根据当前平台位数调整相应字段的位数信息
    for name in indexing_field_names:
        allowed_dtype_dict[name] = allowed_dtype_dict[name].replace(
            expected_dtype_size, allowed_dtype_size
        )

    return allowed_dtype_dict


# 返回适用于不同平台位数的 dtype 字典，以便在跨位部署场景中使用
def _all_compatible_dtype_dicts(dtype):
    # 决策树的 Cython 代码使用平台特定的 intp_t 类型的索引字段，对应于 numpy 数组中的 i4 或 i8 dtype
    # 根据平台位数（32位或64位）在 pickle 加载时需要对 NODE_DTYPE-dtyped 数组的索引字段进行转换，
    # 以支持跨位部署场景。通常我们希望在64位服务器平台上运行昂贵的 fit 方法，然后将估计器进行序列化以在
    # 低功耗32位边缘平台上运行 predict 方法。
    #
    # 这段注释用于说明为什么在 pickle 加载时需要对 dtype 进行调整以适应不同的位数需求。
    pass
    # 类似的情况也适用于字节序，保存 pickle 的机器可能与加载 pickle 的机器字节序不同

    # 将给定的 dtype 转换为字典形式
    dtype_dict = _dtype_to_dict(dtype)
    
    # 修改字典中的比特数信息
    dtype_dict_with_modified_bitness = _dtype_dict_with_modified_bitness(dtype_dict)
    
    # 使用新的字节序修改后的 dtype 转换为字典形式
    dtype_dict_with_modified_endianness = _dtype_to_dict(dtype.newbyteorder())
    
    # 修改了比特数和字节序后的字典形式的 dtype
    dtype_dict_with_modified_bitness_and_endianness = _dtype_dict_with_modified_bitness(
        dtype_dict_with_modified_endianness
    )

    # 返回一个列表，包含不同修改组合后的 dtype 字典
    return [
        dtype_dict,
        dtype_dict_with_modified_bitness,
        dtype_dict_with_modified_endianness,
        dtype_dict_with_modified_bitness_and_endianness,
    ]
# 定义一个函数，用于检查节点数组是否符合预期的数据类型和形状要求
def _check_node_ndarray(node_ndarray, expected_dtype):
    # 检查节点数组的维度是否为1，如果不是则引发数值错误异常
    if node_ndarray.ndim != 1:
        raise ValueError(
            "Wrong dimensions for node array from the pickle: "
            f"expected 1, got {node_ndarray.ndim}"
        )

    # 检查节点数组是否是C连续的，如果不是则引发数值错误异常
    if not node_ndarray.flags.c_contiguous:
        raise ValueError(
            "node array from the pickle should be a C-contiguous array"
        )

    # 获取节点数组的数据类型
    node_ndarray_dtype = node_ndarray.dtype
    # 如果节点数组的数据类型与期望的数据类型相同，则直接返回节点数组
    if node_ndarray_dtype == expected_dtype:
        return node_ndarray

    # 将节点数组的数据类型转换成字典形式
    node_ndarray_dtype_dict = _dtype_to_dict(node_ndarray_dtype)
    # 获取与期望数据类型兼容的所有数据类型字典列表
    all_compatible_dtype_dicts = _all_compatible_dtype_dicts(expected_dtype)

    # 如果节点数组的数据类型字典不在兼容的数据类型字典列表中，则引发数值错误异常
    if node_ndarray_dtype_dict not in all_compatible_dtype_dicts:
        raise ValueError(
            "node array from the pickle has an incompatible dtype:\n"
            f"- expected: {expected_dtype}\n"
            f"- got     : {node_ndarray_dtype}"
        )

    # 将节点数组转换为期望的数据类型并返回，使用同一类型转换的方式
    return node_ndarray.astype(expected_dtype, casting="same_kind")


# =============================================================================
# Build Pruned Tree
# =============================================================================


# 定义一个Cython类，用于控制剪枝过程
cdef class _CCPPruneController:
    """Base class used by build_pruned_tree_ccp and ccp_pruning_path
    to control pruning.
    """
    # 停止剪枝方法，根据有效的alpha值返回1以停止剪枝，返回0以继续剪枝
    cdef bint stop_pruning(self, float64_t effective_alpha) noexcept nogil:
        """Return 1 to stop pruning and 0 to continue pruning"""
        return 0

    # 保存剪枝时的指标数据
    cdef void save_metrics(self, float64_t effective_alpha,
                           float64_t subtree_impurities) noexcept nogil:
        """Save metrics when pruning"""
        pass

    # 剪枝后调用的方法，更新子树中叶子节点的数量
    cdef void after_pruning(self, unsigned char[:] in_subtree) noexcept nogil:
        """Called after pruning"""
        pass


# 定义一个Cython类，继承自_CCPRuneController，用于根据alpha值控制何时停止剪枝
cdef class _AlphaPruner(_CCPPruneController):
    """Use alpha to control when to stop pruning."""
    cdef float64_t ccp_alpha
    cdef intp_t capacity

    def __cinit__(self, float64_t ccp_alpha):
        # 初始化方法，设定ccp_alpha的值和capacity的初始值为0
        self.ccp_alpha = ccp_alpha
        self.capacity = 0

    # 停止剪枝方法的实现，根据有效的alpha值决定是否停止剪枝
    cdef bint stop_pruning(self, float64_t effective_alpha) noexcept nogil:
        # 如果当前的ccp_alpha小于有效的alpha值，则返回1停止剪枝，否则返回0继续剪枝
        return self.ccp_alpha < effective_alpha

    # 剪枝后调用的方法，更新子树中叶子节点的数量
    cdef void after_pruning(self, unsigned char[:] in_subtree) noexcept nogil:
        """Updates the number of leaves in subtree"""
        # 遍历in_subtree数组，如果某个元素为真，则增加self.capacity的值
        for i in range(in_subtree.shape[0]):
            if in_subtree[i]:
                self.capacity += 1


# 定义一个Cython类，继承自_CCPRuneController，用于记录用于返回成本复杂性路径的指标数据
cdef class _PathFinder(_CCPPruneController):
    """Record metrics used to return the cost complexity path."""
    cdef float64_t[:] ccp_alphas
    cdef float64_t[:] impurities
    cdef uint32_t count

    def __cinit__(self,  intp_t node_count):
        # 初始化方法，创建长度为node_count的ccp_alphas和impurities数组，以及初始值为0的count变量
        self.ccp_alphas = np.zeros(shape=(node_count), dtype=np.float64)
        self.impurities = np.zeros(shape=(node_count), dtype=np.float64)
        self.count = 0
    # 定义一个 Cython 的函数，保存计算得到的指标值到对应的数组中
    cdef void save_metrics(self,
                           float64_t effective_alpha,
                           float64_t subtree_impurities) noexcept nogil:
        # 将计算得到的 effective_alpha 存入 self.ccp_alphas 数组的当前位置
        self.ccp_alphas[self.count] = effective_alpha
        # 将计算得到的 subtree_impurities 存入 self.impurities 数组的当前位置
        self.impurities[self.count] = subtree_impurities
        # 自增 count 变量，以便下次保存时指向下一个位置
        self.count += 1
cdef struct CostComplexityPruningRecord:
    intp_t node_idx
    intp_t parent

cdef _cost_complexity_prune(unsigned char[:] leaves_in_subtree,  # OUT
                            Tree orig_tree,
                            _CCPPruneController controller):
    """Perform cost complexity pruning.

    This function takes an already grown tree, `orig_tree` and outputs a
    boolean mask `leaves_in_subtree` which are the leaves in the pruned tree.
    During the pruning process, the controller is passed the effective alpha and
    the subtree impurities. Furthermore, the controller signals when to stop
    pruning.

    Parameters
    ----------
    leaves_in_subtree : unsigned char[:]
        Output for leaves of subtree
    orig_tree : Tree
        Original tree
    ccp_controller : _CCPPruneController
        Cost complexity controller
    """

    cdef:
        intp_t i
        intp_t n_nodes = orig_tree.node_count
        # prior probability using weighted samples
        float64_t[:] weighted_n_node_samples = orig_tree.weighted_n_node_samples
        float64_t total_sum_weights = weighted_n_node_samples[0]
        float64_t[:] impurity = orig_tree.impurity
        # weighted impurity of each node
        float64_t[:] r_node = np.empty(shape=n_nodes, dtype=np.float64)

        intp_t[:] child_l = orig_tree.children_left
        intp_t[:] child_r = orig_tree.children_right
        intp_t[:] parent = np.zeros(shape=n_nodes, dtype=np.intp)

        stack[CostComplexityPruningRecord] ccp_stack
        CostComplexityPruningRecord stack_record
        intp_t node_idx
        stack[intp_t] node_indices_stack

        intp_t[:] n_leaves = np.zeros(shape=n_nodes, dtype=np.intp)
        float64_t[:] r_branch = np.zeros(shape=n_nodes, dtype=np.float64)
        float64_t current_r
        intp_t leaf_idx
        intp_t parent_idx

        # candidate nodes that can be pruned
        unsigned char[:] candidate_nodes = np.zeros(shape=n_nodes,
                                                    dtype=np.uint8)
        # nodes in subtree
        unsigned char[:] in_subtree = np.ones(shape=n_nodes, dtype=np.uint8)
        intp_t pruned_branch_node_idx
        float64_t subtree_alpha
        float64_t effective_alpha
        intp_t n_pruned_leaves
        float64_t r_diff
        float64_t max_float64 = np.finfo(np.float64).max

    # find parent node ids and leaves
    for i in range(n_nodes):
        parent[child_l[i]] = i
        parent[child_r[i]] = i

def _build_pruned_tree_ccp(
    Tree tree,  # OUT
    Tree orig_tree,
    float64_t ccp_alpha
):
    """Build a pruned tree from the original tree using cost complexity
    pruning.

    The values and nodes from the original tree are copied into the pruned
    tree.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    ccp_alpha : float64_t
        Cost complexity parameter alpha for pruning
    """
    # ccp_alpha : 正的 float64_t 类型
    # 复杂度参数。选择成本复杂度小于 ``ccp_alpha`` 的最大子树。
    # 默认情况下，不执行修剪。

    cdef:
        # 计算原始决策树的节点数目
        intp_t n_nodes = orig_tree.node_count
        # 创建一个数组表示子树中的叶子节点，默认为零
        unsigned char[:] leaves_in_subtree = np.zeros(
            shape=n_nodes, dtype=np.uint8)

    # 创建 _AlphaPruner 类的实例，使用给定的 ccp_alpha 参数
    pruning_controller = _AlphaPruner(ccp_alpha=ccp_alpha)

    # 执行成本复杂度修剪，更新 leaves_in_subtree 数组
    _cost_complexity_prune(leaves_in_subtree, orig_tree, pruning_controller)

    # 根据修剪后的子树信息构建修剪后的树
    _build_pruned_tree(tree, orig_tree, leaves_in_subtree,
                       pruning_controller.capacity)
def ccp_pruning_path(Tree orig_tree):
    """Computes the cost complexity pruning path.

    Parameters
    ----------
    orig_tree : Tree
        Original tree.

    Returns
    -------
    path_info : dict
        Information about pruning path with attributes:

        ccp_alphas : ndarray
            Effective alphas of subtree during pruning.

        impurities : ndarray
            Sum of the impurities of the subtree leaves for the
            corresponding alpha value in ``ccp_alphas``.
    """
    # 初始化一个布尔数组，表示原始树中的每个节点是否是子树中的叶子节点
    cdef:
        unsigned char[:] leaves_in_subtree = np.zeros(
            shape=orig_tree.node_count, dtype=np.uint8)

    # 创建 PathFinder 对象，用于查找成本复杂度剪枝路径
    path_finder = _PathFinder(orig_tree.node_count)

    # 执行成本复杂度剪枝算法，修改 leaves_in_subtree 和 path_finder 对象
    _cost_complexity_prune(leaves_in_subtree, orig_tree, path_finder)

    # 初始化用于存储结果的数组
    cdef:
        uint32_t total_items = path_finder.count
        float64_t[:] ccp_alphas = np.empty(shape=total_items, dtype=np.float64)
        float64_t[:] impurities = np.empty(shape=total_items, dtype=np.float64)
        uint32_t count = 0

    # 将 path_finder 中的 ccp_alphas 和 impurities 复制到结果数组中
    while count < total_items:
        ccp_alphas[count] = path_finder.ccp_alphas[count]
        impurities[count] = path_finder.impurities[count]
        count += 1

    # 返回结果字典，包含剪枝路径的 ccp_alphas 和 impurities
    return {
        'ccp_alphas': np.asarray(ccp_alphas),
        'impurities': np.asarray(impurities),
    }


cdef struct BuildPrunedRecord:
    intp_t start
    intp_t depth
    intp_t parent
    bint is_left

cdef _build_pruned_tree(
    Tree tree,  # OUT
    Tree orig_tree,
    const unsigned char[:] leaves_in_subtree,
    intp_t capacity
):
    """Build a pruned tree.

    Build a pruned tree from the original tree by transforming the nodes in
    ``leaves_in_subtree`` into leaves.

    Parameters
    ----------
    tree : Tree
        Location to place the pruned tree
    orig_tree : Tree
        Original tree
    leaves_in_subtree : unsigned char memoryview, shape=(node_count, )
        Boolean mask for leaves to include in subtree
    capacity : intp_t
        Number of nodes to initially allocate in pruned tree
    """
    # 调整新建树的容量
    tree._resize(capacity)

    cdef:
        intp_t orig_node_id
        intp_t new_node_id
        intp_t depth
        intp_t parent
        bint is_left
        bint is_leaf

        # 原始树和新树的值步长相同
        intp_t value_stride = orig_tree.value_stride
        intp_t max_depth_seen = -1
        int rc = 0
        Node* node
        float64_t* orig_value_ptr
        float64_t* new_value_ptr

        # 用于存储剪枝过程中节点信息的堆栈
        stack[BuildPrunedRecord] prune_stack
        BuildPrunedRecord stack_record
    # 使用 nogil 上下文，表示在此期间不会释放全局解释器锁 (GIL)
    with nogil:
        # 将根节点推入修剪栈
        prune_stack.push({"start": 0, "depth": 0, "parent": _TREE_UNDEFINED, "is_left": 0})

        # 当修剪栈不为空时循环执行
        while not prune_stack.empty():
            # 获取栈顶记录
            stack_record = prune_stack.top()
            # 弹出栈顶记录
            prune_stack.pop()

            # 原始节点 ID
            orig_node_id = stack_record.start
            # 当前深度
            depth = stack_record.depth
            # 父节点 ID
            parent = stack_record.parent
            # 是否为左子节点
            is_left = stack_record.is_left

            # 检查当前节点是否为叶子节点
            is_leaf = leaves_in_subtree[orig_node_id]
            # 获取原始树中的节点
            node = &orig_tree.nodes[orig_node_id]

            # 向新树添加节点
            new_node_id = tree._add_node(
                parent, is_left, is_leaf, node.feature, node.threshold,
                node.impurity, node.n_node_samples,
                node.weighted_n_node_samples, node.missing_go_to_left)

            # 检查节点添加是否成功
            if new_node_id == INTPTR_MAX:
                rc = -1
                break

            # 从原始树复制值到新树
            orig_value_ptr = orig_tree.value + value_stride * orig_node_id
            new_value_ptr = tree.value + value_stride * new_node_id
            memcpy(new_value_ptr, orig_value_ptr, sizeof(float64_t) * value_stride)

            # 如果当前节点不是叶子节点，将右子节点推入栈
            if not is_leaf:
                prune_stack.push({"start": node.right_child, "depth": depth + 1,
                                  "parent": new_node_id, "is_left": 0})
                # 将左子节点推入栈
                prune_stack.push({"start": node.left_child, "depth": depth + 1,
                                  "parent": new_node_id, "is_left": 1})

            # 更新最大深度记录
            if depth > max_depth_seen:
                max_depth_seen = depth

        # 如果执行正常完成，则将最大深度记录更新到树对象中
        if rc >= 0:
            tree.max_depth = max_depth_seen

    # 如果出现内存错误 (rc == -1)，则抛出 MemoryError 异常
    if rc == -1:
        raise MemoryError("pruning tree")
```