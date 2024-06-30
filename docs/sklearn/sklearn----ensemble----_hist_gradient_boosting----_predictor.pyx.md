# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\_predictor.pyx`

```
# 作者: Nicolas Hug

# 导入必要的模块和库
from cython.parallel import prange  # 导入并行处理的prange函数
from libc.math cimport isnan  # 导入isnan函数，用于检查是否为NaN
import numpy as np  # 导入NumPy库

# 导入Cython类型和函数定义
from ...utils._typedefs cimport intp_t  # 导入整数类型定义intp_t
from .common cimport X_DTYPE_C  # 导入X数据类型C
from .common cimport Y_DTYPE_C  # 导入Y数据类型C
from .common import Y_DTYPE  # 导入Y数据类型
from .common cimport X_BINNED_DTYPE_C  # 导入X经过分箱后的数据类型C
from .common cimport BITSET_INNER_DTYPE_C  # 导入位集内部数据类型C
from .common cimport node_struct  # 导入节点结构体定义
from ._bitset cimport in_bitset_2d_memoryview  # 导入Cython版的位集查找函数


def _predict_from_raw_data(  # 从原始数据预测，原始数据为非分箱数据
        const node_struct [:] nodes,  # 节点结构体数组
        const X_DTYPE_C [:, :] numeric_data,  # 数值数据的C数组
        const BITSET_INNER_DTYPE_C [:, ::1] raw_left_cat_bitsets,  # 原始左侧分类位集的C数组
        const BITSET_INNER_DTYPE_C [:, ::1] known_cat_bitsets,  # 已知分类位集的C数组
        const unsigned int [::1] f_idx_map,  # 特征索引映射的C数组
        int n_threads,  # 线程数
        Y_DTYPE_C [:] out):  # 输出数组的C数组

    cdef:
        int i  # 循环变量i

    # 使用并行循环prange处理每行数据
    for i in prange(numeric_data.shape[0], schedule='static', nogil=True,
                    num_threads=n_threads):
        # 对每行数据调用_predict_one_from_raw_data进行预测，并将结果存入输出数组out中
        out[i] = _predict_one_from_raw_data(
            nodes, numeric_data, raw_left_cat_bitsets,
            known_cat_bitsets,
            f_idx_map, i)


cdef inline Y_DTYPE_C _predict_one_from_raw_data(
        const node_struct [:] nodes,  # 节点结构体数组
        const X_DTYPE_C [:, :] numeric_data,  # 数值数据的C数组
        const BITSET_INNER_DTYPE_C [:, ::1] raw_left_cat_bitsets,  # 原始左侧分类位集的C数组
        const BITSET_INNER_DTYPE_C [:, ::1] known_cat_bitsets,  # 已知分类位集的C数组
        const unsigned int [::1] f_idx_map,  # 特征索引映射的C数组
        const int row) noexcept nogil:  # 行索引，无异常处理且无全局解锁

    # 需要传递整个数组和行索引才能使prange正常工作
    # 参考Cython问题 #2798

    cdef:
        node_struct node = nodes[0]  # 初始化节点为第一个节点
        unsigned int node_idx = 0  # 初始化节点索引为0
        X_DTYPE_C data_val  # 数据值

    while True:  # 开始循环
        if node.is_leaf:  # 如果节点是叶节点
            return node.value  # 直接返回节点值

        data_val = numeric_data[row, node.feature_idx]  # 获取当前节点对应特征的数据值

        if isnan(data_val):  # 如果数据值为NaN
            if node.missing_go_to_left:  # 如果缺失值处理方式为向左
                node_idx = node.left  # 转向左子节点
            else:
                node_idx = node.right  # 否则转向右子节点
        elif node.is_categorical:  # 如果节点是分类特征
            if data_val < 0:  # 如果数据值小于0
                # 数据值不在接受范围内，视为缺失值处理
                node_idx = node.left if node.missing_go_to_left else node.right
            elif in_bitset_2d_memoryview(
                    raw_left_cat_bitsets,
                    <X_BINNED_DTYPE_C>data_val,
                    node.bitset_idx):  # 在原始左侧分类位集中找到数据值
                node_idx = node.left  # 转向左子节点
            elif in_bitset_2d_memoryview(
                    known_cat_bitsets,
                    <X_BINNED_DTYPE_C>data_val,
                    f_idx_map[node.feature_idx]):  # 在已知分类位集中找到数据值
                node_idx = node.right  # 转向右子节点
            else:
                # 未知的分类视为缺失值处理
                node_idx = node.left if node.missing_go_to_left else node.right
        else:  # 如果节点是数值特征
            if data_val <= node.num_threshold:  # 如果数据值小于等于节点阈值
                node_idx = node.left  # 转向左子节点
            else:
                node_idx = node.right  # 否则转向右子节点
        node = nodes[node_idx]  # 更新当前节点为新的节点
# 针对给定的节点和二进制化数据进行预测，将结果存储在输出数组中
def _predict_from_binned_data(
        node_struct [:] nodes,  # 节点结构数组，表示预测树的节点
        const X_BINNED_DTYPE_C [:, :] binned_data,  # 二进制化的输入数据数组
        BITSET_INNER_DTYPE_C [:, :] binned_left_cat_bitsets,  # 左子树分类特征的位集数组
        const unsigned char missing_values_bin_idx,  # 缺失值的二进制化索引
        int n_threads,  # 使用的线程数
        Y_DTYPE_C [:] out  # 存储预测结果的输出数组
):

    cdef:
        int i  # 循环变量

    # 使用并行循环处理每个输入数据行，静态分配，无全局解锁，使用指定的线程数
    for i in prange(binned_data.shape[0], schedule='static', nogil=True,
                    num_threads=n_threads):
        # 调用内联函数进行单个数据点的预测，并将结果存储在输出数组中
        out[i] = _predict_one_from_binned_data(nodes,
                                               binned_data,
                                               binned_left_cat_bitsets, i,
                                               missing_values_bin_idx)


cdef inline Y_DTYPE_C _predict_one_from_binned_data(
        node_struct [:] nodes,  # 节点结构数组，表示预测树的节点
        const X_BINNED_DTYPE_C [:, :] binned_data,  # 二进制化的输入数据数组
        const BITSET_INNER_DTYPE_C [:, :] binned_left_cat_bitsets,  # 左子树分类特征的位集数组
        const int row,  # 当前处理的数据行索引
        const unsigned char missing_values_bin_idx  # 缺失值的二进制化索引
) noexcept nogil:
    # 需要传递整个数组和行索引，否则 prange 无法正常工作。参见 Cython 问题 #2798

    cdef:
        node_struct node = nodes[0]  # 初始化为根节点
        unsigned int node_idx = 0  # 当前节点索引
        X_BINNED_DTYPE_C data_val  # 当前数据点的特征值

    # 进行树的遍历，直到找到叶子节点为止
    while True:
        if node.is_leaf:
            return node.value  # 返回叶子节点的值作为预测结果

        # 获取当前数据点在当前节点分裂特征上的特征值
        data_val = binned_data[row, node.feature_idx]

        # 根据特征值与节点的分裂方式决定下一步向左还是向右遍历
        if data_val == missing_values_bin_idx:
            if node.missing_go_to_left:
                node_idx = node.left
            else:
                node_idx = node.right
        elif node.is_categorical:
            if in_bitset_2d_memoryview(
                    binned_left_cat_bitsets,
                    data_val,
                    node.bitset_idx):
                node_idx = node.left
            else:
                node_idx = node.right
        else:
            if data_val <= node.bin_threshold:
                node_idx = node.left
            else:
                node_idx = node.right

        # 更新当前节点为下一个节点
        node = nodes[node_idx]


def _compute_partial_dependence(
    node_struct [:] nodes,  # 节点结构数组，表示预测树的节点
    const X_DTYPE_C [:, ::1] X,  # 输入数据数组，每行是一个数据样本
    const intp_t [:] target_features,  # 目标特征的索引数组
    Y_DTYPE_C [:] out  # 存储偏依赖结果的输出数组
):
    """Partial dependence of the response on the ``target_features`` set.

    For each sample in ``X`` a tree traversal is performed.
    Each traversal starts from the root with weight 1.0.

    At each non-leaf node that splits on a target feature, either
    the left child or the right child is visited based on the feature
    value of the current sample, and the weight is not modified.
    At each non-leaf node that splits on a complementary feature,
    both children are visited and the weight is multiplied by the fraction
    of training samples which went to each child.

    At each leaf, the value of the node is multiplied by the current
    weight (weights sum to 1 for all visited terminal nodes).

    Parameters
    ----------
    nodes : view on array of PREDICTOR_RECORD_DTYPE, shape (n_nodes)
        The array representing the predictor tree.
   `
    X : view on 2d ndarray, shape (n_samples, n_target_features)
        要评估偏依赖的网格点的二维数组视图，形状为 (n_samples, n_target_features)
    target_features : view on 1d ndarray of intp_t, shape (n_target_features)
        要评估偏依赖的目标特征集合的一维数组视图，形状为 (n_target_features)
    out : view on 1d ndarray, shape (n_samples)
        每个网格点上偏依赖函数的值的一维数组视图，形状为 (n_samples)
    """
    # 定义未签名整数变量，表示当前节点索引
    cdef:
        unsigned int current_node_idx
        # 创建节点索引堆栈的视图，形状与节点数组的行数相同
        unsigned int [:] node_idx_stack = np.zeros(shape=nodes.shape[0],
                                                   dtype=np.uint32)
        # 创建权重堆栈的视图，形状与节点数组的行数相同，类型为 Y_DTYPE_C
        Y_DTYPE_C [::1] weight_stack = np.zeros(shape=nodes.shape[0],
                                                dtype=Y_DTYPE)
        # 当前节点结构体的指针，用于避免复制属性
        node_struct * current_node

        # 未签名整数样本索引
        unsigned int sample_idx
        # intp_t 类型的特征索引
        intp_t feature_idx
        # 未签名整数堆栈大小
        unsigned stack_size
        # 当前左侧样本分数的 Y_DTYPE_C 类型
        Y_DTYPE_C left_sample_frac
        # 当前权重的 Y_DTYPE_C 类型
        Y_DTYPE_C current_weight
        # 仅用于健全性检查的总权重 Y_DTYPE_C 类型
        Y_DTYPE_C total_weight
        # 是否是目标特征的布尔值
        bint is_target_feature
    for sample_idx in range(X.shape[0]):
        # 初始化当前样本的堆栈
        stack_size = 1
        node_idx_stack[0] = 0  # 根节点索引为0
        weight_stack[0] = 1  # 所有样本都在根节点
        total_weight = 0

        while stack_size > 0:

            # 弹出堆栈顶部的元素
            stack_size -= 1
            current_node_idx = node_idx_stack[stack_size]
            current_node = &nodes[current_node_idx]

            if current_node.is_leaf:
                # 如果当前节点是叶子节点，累加加权后的节点值到输出
                out[sample_idx] += (weight_stack[stack_size] *
                                    current_node.value)
                total_weight += weight_stack[stack_size]
            else:
                # 判断分裂特征是否是目标特征
                is_target_feature = False
                for feature_idx in range(target_features.shape[0]):
                    if target_features[feature_idx] == current_node.feature_idx:
                        is_target_feature = True
                        break

                if is_target_feature:
                    # 如果是目标特征，则根据阈值将左右子节点推入堆栈
                    if X[sample_idx, feature_idx] <= current_node.num_threshold:
                        node_idx_stack[stack_size] = current_node.left
                    else:
                        node_idx_stack[stack_size] = current_node.right
                    stack_size += 1
                else:
                    # 如果不是目标特征，则将左右子节点同时推入堆栈，并按照样本比例给予权重
                    # 推入左子节点
                    node_idx_stack[stack_size] = current_node.left
                    left_sample_frac = (
                        <Y_DTYPE_C> nodes[current_node.left].count /
                        current_node.count)
                    current_weight = weight_stack[stack_size]
                    weight_stack[stack_size] = current_weight * left_sample_frac
                    stack_size += 1

                    # 推入右子节点
                    node_idx_stack[stack_size] = current_node.right
                    weight_stack[stack_size] = (
                        current_weight * (1 - left_sample_frac))
                    stack_size += 1

        # 对总权重进行检查，应该接近1.0
        if not (0.999 < total_weight < 1.001):
            raise ValueError("Total weight should be 1.0 but was %.9f" % total_weight)
```