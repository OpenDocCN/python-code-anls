# `D:\src\scipysrc\scikit-learn\sklearn\tree\_tree.pxd`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _tree.pyx for details.

# 导入需要的模块和类型
import numpy as np
cimport numpy as cnp

# 从相对路径导入特定类型
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t

# 从指定路径导入Splitter类和SplitRecord类
from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

# 定义一个结构体Node，用于存储决策树中节点的基本信息
cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    intp_t left_child                    # id of the left child of the node
    intp_t right_child                   # id of the right child of the node
    intp_t feature                       # Feature used for splitting the node
    float64_t threshold                  # Threshold value at the node
    float64_t impurity                   # Impurity of the node (i.e., the value of the criterion)
    intp_t n_node_samples                # Number of samples at the node
    float64_t weighted_n_node_samples    # Weighted number of samples at the node
    unsigned char missing_go_to_left     # Whether features have missing values

# 定义一个结构体ParentInfo，用于存储节点父节点的信息
cdef struct ParentInfo:
    # Structure to store information about the parent of a node
    # This is passed to the splitter, to provide information about the previous split

    float64_t lower_bound           # the lower bound of the parent's impurity
    float64_t upper_bound           # the upper bound of the parent's impurity
    float64_t impurity              # the impurity of the parent
    intp_t n_constant_features      # the number of constant features found in parent

# 定义一个Tree类，表示决策树对象，用于预测和特征重要性评估
cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public intp_t n_features        # Number of features in X
    cdef intp_t* n_classes               # Number of classes in y[:, k]
    cdef public intp_t n_outputs         # Number of outputs in y
    cdef public intp_t max_n_classes     # max(n_classes)

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public intp_t max_depth         # Max depth of the tree
    cdef public intp_t node_count        # Counter for node IDs
    cdef public intp_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef float64_t* value                # (capacity, n_outputs, max_n_classes) array of values
    cdef intp_t value_stride             # = n_outputs * max_n_classes

    # Methods
    # 添加一个新节点到树中
    cdef intp_t _add_node(self, intp_t parent, bint is_left, bint is_leaf,
                          intp_t feature, float64_t threshold, float64_t impurity,
                          intp_t n_node_samples,
                          float64_t weighted_n_node_samples,
                          unsigned char missing_go_to_left) except -1 nogil
    # 调整树的容量大小
    cdef int _resize(self, intp_t capacity) except -1 nogil
    # 声明一个 C 函数 _resize_c，参数为 self 和 capacity，返回类型为 int，可能会抛出 -1 异常，使用 GIL 之外的线程
    cdef int _resize_c(self, intp_t capacity=*) except -1 nogil

    # 声明一个 C 函数 _get_value_ndarray，返回一个 NumPy 数组（ndarray），不接受参数类型声明
    cdef cnp.ndarray _get_value_ndarray(self)

    # 声明一个 C 函数 _get_node_ndarray，返回一个 NumPy 数组（ndarray），不接受参数类型声明
    cdef cnp.ndarray _get_node_ndarray(self)

    # 定义一个公共 Cython/Python 函数 predict，接受一个对象 X，并返回一个 NumPy 数组（ndarray）
    cpdef cnp.ndarray predict(self, object X)

    # 定义一个公共 Cython/Python 函数 apply，接受一个对象 X，并返回一个 NumPy 数组（ndarray）
    cpdef cnp.ndarray apply(self, object X)
    
    # 声明一个 C 函数 _apply_dense，接受一个对象 X，返回一个 NumPy 数组（ndarray）
    cdef cnp.ndarray _apply_dense(self, object X)
    
    # 声明一个 C 函数 _apply_sparse_csr，接受一个对象 X，返回一个 NumPy 数组（ndarray）
    cdef cnp.ndarray _apply_sparse_csr(self, object X)

    # 定义一个公共 Cython/Python 函数 decision_path，接受一个对象 X，返回一个对象
    cpdef object decision_path(self, object X)
    
    # 声明一个 C 函数 _decision_path_dense，接受一个对象 X，返回一个对象
    cdef object _decision_path_dense(self, object X)
    
    # 声明一个 C 函数 _decision_path_sparse_csr，接受一个对象 X，返回一个对象
    cdef object _decision_path_sparse_csr(self, object X)

    # 定义一个公共 Cython/Python 函数 compute_node_depths，不接受参数，返回对象
    cpdef compute_node_depths(self)
    
    # 定义一个公共 Cython/Python 函数 compute_feature_importances，接受一个参数 normalize，默认为 *
    cpdef compute_feature_importances(self, normalize=*)
# =============================================================================
# Tree builder
# =============================================================================

# 定义一个 Cython 扩展类型 TreeBuilder，用于递归地从训练样本构建树对象。
# 使用 Splitter 对象进行内部节点的分裂，并为叶节点分配值。

cdef class TreeBuilder:
    # Splitter 算法对象，用于节点分裂
    cdef Splitter splitter

    # 内部节点的最小样本数
    cdef intp_t min_samples_split
    # 叶节点的最小样本数
    cdef intp_t min_samples_leaf
    # 叶节点的最小权重
    cdef float64_t min_weight_leaf
    # 树的最大深度
    cdef intp_t max_depth
    # 早停止的不纯度阈值
    cdef float64_t min_impurity_decrease

    # 构建树的方法，接受 Tree 对象、特征矩阵 X、目标值矩阵 y、样本权重 sample_weight
    # 和特征缺失值掩码 missing_values_in_feature_mask
    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=*,
        const unsigned char[::1] missing_values_in_feature_mask=*,
    )

    # 检查输入数据的合法性，包括特征矩阵 X、目标值矩阵 y 和样本权重 sample_weight
    cdef _check_input(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
    )
```