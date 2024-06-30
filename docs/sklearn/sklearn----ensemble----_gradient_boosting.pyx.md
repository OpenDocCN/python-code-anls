# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_gradient_boosting.pyx`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入标准库函数和memset函数
from libc.stdlib cimport free
from libc.string cimport memset

# 导入NumPy库和issparse函数
import numpy as np
from scipy.sparse import issparse

# 导入自定义类型定义
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t

# 注意：_tree使用了cimport numpy和cnp.import_array，所以需要在扩展模块的构建配置中包含NumPy头文件
from ..tree._tree cimport Node
from ..tree._tree cimport Tree
from ..tree._utils cimport safe_realloc

# 从NumPy导入zeros函数，并且将其重命名为np_zeros，不涉及名称空间查找
from numpy import zeros as np_zeros

# 常量，用于标记树的叶子节点
cdef intp_t TREE_LEAF = -1

# 定义函数_predict_regression_tree_inplace_fast_dense，用于快速在密集数据上进行回归树预测
cdef void _predict_regression_tree_inplace_fast_dense(
    const float32_t[:, ::1] X,
    Node* root_node,
    double *value,
    double scale,
    Py_ssize_t k,
    float64_t[:, :] out
) noexcept nogil:
    """Predicts output for regression tree and stores it in ``out[i, k]``.

    This function operates directly on the data arrays of the tree
    data structures. This is 5x faster than the variant above because
    it allows us to avoid buffer validation.

    The function assumes that the ndarray that wraps ``X`` is
    c-continuous.

    Parameters
    ----------
    X : float32_t 2d memory view
        The memory view on the data ndarray of the input ``X``.
        Assumes that the array is c-continuous.
    root_node : tree Node pointer
        Pointer to the main node array of the :class:``sklearn.tree.Tree``.
    value : np.float64_t pointer
        The pointer to the data array of the ``value`` array attribute
        of the :class:``sklearn.tree.Tree``.
    scale : double
        A constant to scale the predictions.
    k : int
        The index of the tree output to be predicted. Must satisfy
        0 <= ``k`` < ``K``.
    out : memory view on array of type np.float64_t
        The data array where the predictions are stored.
        ``out`` is assumed to be a two-dimensional array of
        shape ``(n_samples, K)``.
    """
    # 获取样本数
    cdef intp_t n_samples = X.shape[0]
    cdef Py_ssize_t i
    cdef Node *node
    # 遍历每个样本
    for i in range(n_samples):
        node = root_node
        # 当节点不是叶子节点时执行循环
        while node.left_child != TREE_LEAF:
            # 根据节点的阈值将样本分类到左子树或右子树
            if X[i, node.feature] <= node.threshold:
                node = root_node + node.left_child
            else:
                node = root_node + node.right_child
        # 计算并更新预测值到输出数组中
        out[i, k] += scale * value[node - root_node]

# 定义函数_predict_regression_tree_stages_sparse，用于稀疏数据上的回归树预测
def _predict_regression_tree_stages_sparse(
    object[:, :] estimators,
    object X,
    double scale,
    float64_t[:, :] out
):
    """Predicts output for regression tree inplace and adds scaled value to ``out[i, k]``.

    The function assumes that the ndarray that wraps ``X`` is csr_matrix.
    """
    # 提取稀疏矩阵X的data, indices, indptr属性
    cdef const float32_t[::1] X_data = X.data
    cdef const int32_t[::1] X_indices = X.indices
    cdef const int32_t[::1] X_indptr = X.indptr

    # 获取样本数和特征数
    cdef intp_t n_samples = X.shape[0]
    cdef intp_t n_features = X.shape[1]
    # 获取估计器的阶段数和输出数
    cdef intp_t n_stages = estimators.shape[0]
    cdef intp_t n_outputs = estimators.shape[1]

    # 索引和临时变量
    cdef intp_t sample_i  # 样本索引
    cdef intp_t feature_i  # 特征索引
    cdef intp_t stage_i  # 阶段索引
    cdef intp_t output_i  # 输出索引
    cdef Node *root_node = NULL  # 根节点指针
    cdef Node *node = NULL  # 节点指针
    cdef double *value = NULL  # 值指针

    cdef Tree tree  # 树对象
    cdef Node** nodes = NULL  # 节点数组指针
    cdef double** values = NULL  # 值数组指针
    safe_realloc(&nodes, n_stages * n_outputs)  # 安全重新分配节点数组
    safe_realloc(&values, n_stages * n_outputs)  # 安全重新分配值数组
    for stage_i in range(n_stages):
        for output_i in range(n_outputs):
            tree = estimators[stage_i, output_i].tree_
            nodes[stage_i * n_outputs + output_i] = tree.nodes  # 将树的节点数组赋给节点指针数组
            values[stage_i * n_outputs + output_i] = tree.value  # 将树的值数组赋给值指针数组

    # 初始化辅助数据结构
    cdef float32_t feature_value = 0.  # 特征值
    cdef float32_t* X_sample = NULL  # 样本数据指针

    # feature_to_sample作为数据结构记录每个特征最后见到的样本，
    # 在功能上，它是一种有效识别当前样本中哪些特征为非零的方法。
    cdef intp_t* feature_to_sample = NULL  # 特征到样本的映射数组指针

    safe_realloc(&X_sample, n_features)  # 安全重新分配样本数据指针
    safe_realloc(&feature_to_sample, n_features)  # 安全重新分配特征到样本的映射数组指针

    memset(feature_to_sample, -1, n_features * sizeof(intp_t))  # 初始化特征到样本映射为-1

    # 循环遍历所有样本
    for sample_i in range(n_samples):
        for feature_i in range(X_indptr[sample_i], X_indptr[sample_i + 1]):
            feature_to_sample[X_indices[feature_i]] = sample_i
            X_sample[X_indices[feature_i]] = X_data[feature_i]

        # 循环遍历所有阶段
        for stage_i in range(n_stages):
            # 循环遍历所有输出
            for output_i in range(n_outputs):
                root_node = nodes[stage_i * n_outputs + output_i]
                value = values[stage_i * n_outputs + output_i]
                node = root_node

                # 当节点不是叶子节点时进行循环
                while node.left_child != TREE_LEAF:
                    # 如果节点的特征在当前样本中存在，则特征值为当前样本的特征值，否则为0
                    if feature_to_sample[node.feature] == sample_i:
                        feature_value = X_sample[node.feature]
                    else:
                        feature_value = 0.

                    # 根据特征值和节点的阈值更新节点指针
                    if feature_value <= node.threshold:
                        node = root_node + node.left_child
                    else:
                        node = root_node + node.right_child

                # 将计算结果累加到输出数组中
                out[sample_i, output_i] += scale * value[node - root_node]

    # 释放辅助数组内存空间
    free(X_sample)
    free(feature_to_sample)
    free(nodes)
    free(values)
def predict_stages(
    object[:, :] estimators,  # 二维数组，包含所有预测器对象
    object X,  # 输入数据
    double scale,  # 缩放因子
    float64_t[:, :] out  # 输出数组，用于存储预测结果
):
    """Add predictions of ``estimators`` to ``out``.

    Each estimator is scaled by ``scale`` before its prediction
    is added to ``out``.
    """
    cdef Py_ssize_t i  # 循环变量：预测器索引
    cdef Py_ssize_t k  # 循环变量：子模型索引
    cdef Py_ssize_t n_estimators = estimators.shape[0]  # 预测器数量
    cdef Py_ssize_t K = estimators.shape[1]  # 每个预测器包含的子模型数量
    cdef Tree tree  # 决策树对象

    if issparse(X):  # 如果输入数据是稀疏矩阵
        if X.format != 'csr':  # 确保稀疏矩阵格式为CSR
            raise ValueError("When X is a sparse matrix, a CSR format is"
                             " expected, got {!r}".format(type(X)))
        _predict_regression_tree_stages_sparse(
            estimators=estimators, X=X, scale=scale, out=out
        )
    else:
        if not isinstance(X, np.ndarray) or np.isfortran(X):
            raise ValueError(f"X should be C-ordered np.ndarray, got {type(X)}")

        for i in range(n_estimators):  # 遍历所有预测器
            for k in range(K):  # 遍历每个预测器中的子模型
                tree = estimators[i, k].tree_

                # 避免缓冲区验证，将数据转换为ndarray并获取数据指针
                _predict_regression_tree_inplace_fast_dense(
                    X=X,
                    root_node=tree.nodes,
                    value=tree.value,
                    scale=scale,
                    k=k,
                    out=out
                )
                # out[:, k] += scale * tree.predict(X).ravel()


def predict_stage(
    object[:, :] estimators,  # 二维数组，包含所有预测器对象
    int stage,  # 预测阶段索引
    object X,  # 输入数据
    double scale,  # 缩放因子
    float64_t[:, :] out  # 输出数组，用于存储预测结果
):
    """Add predictions of ``estimators[stage]`` to ``out``.

    Each estimator in the stage is scaled by ``scale`` before
    its prediction is added to ``out``.
    """
    return predict_stages(
        estimators=estimators[stage:stage + 1], X=X, scale=scale, out=out
    )


def _random_sample_mask(
    intp_t n_total_samples,  # 总样本数
    intp_t n_total_in_bag,  # 在袋外的样本数
    random_state  # 随机数生成器对象
):
    """Create a random sample mask where ``n_total_in_bag`` elements are set.

    Parameters
    ----------
    n_total_samples : int
        The length of the resulting mask.

    n_total_in_bag : int
        The number of elements in the sample mask which are set to 1.

    random_state : RandomState
        A numpy ``RandomState`` object.

    Returns
    -------
    sample_mask : np.ndarray, shape=[n_total_samples]
        An ndarray where ``n_total_in_bag`` elements are set to ``True``
        the others are ``False``.
    """
    cdef float64_t[::1] rand = random_state.uniform(size=n_total_samples)  # 随机生成0到1之间的浮点数
    cdef uint8_t[::1] sample_mask = np_zeros((n_total_samples,), dtype=bool)  # 初始化一个布尔类型的数组作为样本掩码

    cdef intp_t n_bagged = 0  # 已放入袋内的样本计数
    cdef intp_t i = 0  # 循环变量

    for i in range(n_total_samples):  # 遍历所有样本
        if rand[i] * (n_total_samples - i) < (n_total_in_bag - n_bagged):
            sample_mask[i] = 1  # 将当前样本置为True，表示放入袋内
            n_bagged += 1  # 更新袋内样本计数

    return sample_mask.base  # 返回样本掩码的基础数组
```