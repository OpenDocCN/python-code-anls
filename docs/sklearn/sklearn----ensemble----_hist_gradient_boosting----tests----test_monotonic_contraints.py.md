# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_monotonic_contraints.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

from sklearn.ensemble import (  # 从scikit-learn的ensemble模块中导入以下分类器和回归器
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.ensemble._hist_gradient_boosting.common import (  # 从scikit-learn的ensemble模块中导入一些共享的常量和类
    G_H_DTYPE,
    X_BINNED_DTYPE,
    MonotonicConstraint,
)
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower  # 导入TreeGrower类
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder  # 导入HistogramBuilder类
from sklearn.ensemble._hist_gradient_boosting.splitting import (  # 从splitting模块中导入以下函数和类
    Splitter,
    compute_node_value,
)
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads  # 导入_openmp_effective_n_threads函数
from sklearn.utils._testing import _convert_container  # 导入_convert_container函数

n_threads = _openmp_effective_n_threads()  # 获取有效的OpenMP线程数


def is_increasing(a):
    """判断数组a是否严格递增"""
    return (np.diff(a) >= 0.0).all()


def is_decreasing(a):
    """判断数组a是否严格递减"""
    return (np.diff(a) <= 0.0).all()


def assert_leaves_values_monotonic(predictor, monotonic_cst):
    """确保叶节点值满足单调性约束"""
    nodes = predictor.nodes

    def get_leaves_values():
        """从左到右获取叶节点值"""
        values = []

        def depth_first_collect_leaf_values(node_idx):
            node = nodes[node_idx]
            if node["is_leaf"]:
                values.append(node["value"])
                return
            depth_first_collect_leaf_values(node["left"])
            depth_first_collect_leaf_values(node["right"])

        depth_first_collect_leaf_values(0)  # 从根节点（索引0）开始深度优先收集叶节点值
        return values

    values = get_leaves_values()

    if monotonic_cst == MonotonicConstraint.NO_CST:
        # 没有单调性约束，叶节点值可以既递增也递减
        assert not is_increasing(values) and not is_decreasing(values)
    elif monotonic_cst == MonotonicConstraint.POS:
        # 必须递增
        assert is_increasing(values)
    else:  # monotonic_cst == MonotonicConstraint.NEG
        # 必须递减
        assert is_decreasing(values)


def assert_children_values_monotonic(predictor, monotonic_cst):
    """确保子节点值满足单调性约束"""
    nodes = predictor.nodes
    left_lower = []
    left_greater = []
    for node in nodes:
        if node["is_leaf"]:
            continue

        left_idx = node["left"]
        right_idx = node["right"]

        if nodes[left_idx]["value"] < nodes[right_idx]["value"]:
            left_lower.append(node)
        elif nodes[left_idx]["value"] > nodes[right_idx]["value"]:
            left_greater.append(node)
    # 如果没有单调约束（MonotonicConstraint.NO_CST），则确保左边界小于右边界且左边界大于右边界均为真
    if monotonic_cst == MonotonicConstraint.NO_CST:
        assert left_lower and left_greater
    # 如果单调约束为正（MonotonicConstraint.POS），则确保左边界小于右边界为真，左边界大于右边界为假
    elif monotonic_cst == MonotonicConstraint.POS:
        assert left_lower and not left_greater
    # 否则（单调约束为负，MonotonicConstraint.NEG），确保左边界小于右边界为假，左边界大于右边界为真
    else:  # NEG
        assert not left_lower and left_greater
# 确保节点的子节点数值在该节点与其兄弟节点（如果有单调性约束）之间的中间值范围内
# 作为额外的检查，我们还验证兄弟节点的值是否正确排序，这与 assert_children_values_monotonic 稍微重复（但是
# 这个检查是在 grower 节点上进行的，而 assert_children_values_monotonic 是在 predictor 节点上进行的）

def assert_children_values_bounded(grower, monotonic_cst):
    if monotonic_cst == MonotonicConstraint.NO_CST:
        return

    # 递归地检查节点的子节点数值
    def recursively_check_children_node_values(node, right_sibling=None):
        # 如果节点是叶子节点，则直接返回
        if node.is_leaf:
            return
        # 如果有右兄弟节点，则计算当前节点值与右兄弟节点值之间的中间值
        if right_sibling is not None:
            middle = (node.value + right_sibling.value) / 2
            # 根据单调性约束进行断言
            if monotonic_cst == MonotonicConstraint.POS:
                # 如果是正单调性，则断言左子节点值 <= 右子节点值 <= 中间值
                assert node.left_child.value <= node.right_child.value <= middle
                # 如果右兄弟节点不是叶子节点，则继续断言右兄弟节点的左右子节点值也符合顺序要求
                if not right_sibling.is_leaf:
                    assert middle <= right_sibling.left_child.value <= right_sibling.right_child.value
            else:  # NEG
                # 如果是负单调性，则断言左子节点值 >= 右子节点值 >= 中间值
                assert node.left_child.value >= node.right_child.value >= middle
                # 如果右兄弟节点不是叶子节点，则继续断言右兄弟节点的左右子节点值也符合顺序要求
                if not right_sibling.is_leaf:
                    assert middle >= right_sibling.left_child.value >= right_sibling.right_child.value

        # 递归检查左子节点与右子节点
        recursively_check_children_node_values(node.left_child, right_sibling=node.right_child)
        recursively_check_children_node_values(node.right_child)

    # 从 grower 的根节点开始递归检查
    recursively_check_children_node_values(grower.root)


# 使用 pytest 的参数化功能，对节点数值进行测试
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "monotonic_cst",
    (
        MonotonicConstraint.NO_CST,
        MonotonicConstraint.POS,
        MonotonicConstraint.NEG,
    ),
)
def test_nodes_values(monotonic_cst, seed):
    # 构建一个只有一个特征的单一树，并确保节点数值符合单调性约束

    # 考虑以下具有单调正约束的树，我们应该有：
    #
    #       root
    #      /    \
    #     5     10    # 中间值 = 7.5
    #    / \   / \
    #   a  b  c  d
    #
    # a <= b 且 c <= d  (assert_children_values_monotonic)
    # a, b <= 中间值 <= c, d (assert_children_values_bounded)
    # a <= b <= c <= d (assert_leaves_values_monotonic)
    #
    # 最后一个是其他条件的必然结果，但检查一下也无妨

    rng = np.random.RandomState(seed)
    n_samples = 1000
    n_features = 1
    X_binned = rng.randint(0, 255, size=(n_samples, n_features), dtype=np.uint8)
    X_binned = np.asfortranarray(X_binned)

    gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    # 使用TreeGrower类初始化一个树生长器对象，传入特征X_binned、梯度gradients、
    # Hessian矩阵hessians、单调性常量monotonic_cst和收缩率shrinkage=0.1
    grower = TreeGrower(
        X_binned, gradients, hessians, monotonic_cst=[monotonic_cst], shrinkage=0.1
    )
    
    # 调用grow()方法执行树的生长过程
    grower.grow()

    # 对于每一个已完成的叶子节点，将其值除以收缩率shrinkage，
    # 以便在比较测试中能够与未收缩的节点值进行正确的比较
    for leave in grower.finalized_leaves:
        leave.value /= grower.shrinkage

    # 使用grower.make_predictor()方法创建一个预测器对象predictor，
    # 传入未定义的binning_thresholds，因为我们不会使用predict方法
    predictor = grower.make_predictor(
        binning_thresholds=np.zeros((X_binned.shape[1], X_binned.max() + 1))
    )

    # 在树生长器对象(grower)上执行节点值的单调性检查
    assert_children_values_bounded(grower, monotonic_cst)
    
    # 在预测器对象(predictor)上执行节点值的单调性检查
    assert_children_values_monotonic(predictor, monotonic_cst)
    
    # 在预测器对象(predictor)上执行叶子节点值的单调性检查
    assert_leaves_values_monotonic(predictor, monotonic_cst)
@pytest.mark.parametrize("use_feature_names", (True, False))
def test_predictions(global_random_seed, use_feature_names):
    """
    测试预测功能，验证模型对于非类别特征的正负约束是否有效。
    测试方法源自lightgbm的test_monotone_constraint()，灵感来自于https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
    """

    rng = np.random.RandomState(global_random_seed)

    n_samples = 1000
    f_0 = rng.rand(n_samples)  # 与 y 正相关的特征
    f_1 = rng.rand(n_samples)  # 与 y 负相关的特征

    # 额外的类别特征，与 y 无关，用于检验单调性约束的正确性重映射，参见问题 #28898
    f_a = rng.randint(low=0, high=9, size=n_samples)
    f_b = rng.randint(low=0, high=9, size=n_samples)
    f_c = rng.randint(low=0, high=9, size=n_samples)

    X = np.c_[f_a, f_0, f_b, f_1, f_c]
    columns_name = ["f_a", "f_0", "f_b", "f_1", "f_c"]
    constructor_name = "dataframe" if use_feature_names else "array"
    X = _convert_container(X, constructor_name, columns_name=columns_name)

    noise = rng.normal(loc=0.0, scale=0.01, size=n_samples)
    y = 5 * f_0 + np.sin(10 * np.pi * f_0) - 5 * f_1 - np.cos(10 * np.pi * f_1) + noise

    if use_feature_names:
        monotonic_cst = {"f_0": +1, "f_1": -1}
        categorical_features = ["f_a", "f_b", "f_c"]
    else:
        monotonic_cst = [0, +1, 0, -1, 0]
        categorical_features = [0, 2, 4]

    gbdt = HistGradientBoostingRegressor(
        monotonic_cst=monotonic_cst, categorical_features=categorical_features
    )
    gbdt.fit(X, y)

    linspace = np.linspace(0, 1, 100)
    sin = np.sin(linspace)
    constant = np.full_like(linspace, fill_value=0.5)

    # 现在我们断言预测结果是否正确遵守约束条件，对每个特征进行测试。
    # 在测试一个特征时，需要将其他特征设为常数，因为单调性约束仅适用于“其他条件不变”的情况：
    # 对第一个特征的约束仅意味着 x0 < x0' => f(x0, x1) < f(x0', x1)，而 x1 保持不变。
    # 约束并不保证 x0 < x0' => f(x0, x1) < f(x0', x1')

    # 第一个非类别特征（正约束）
    # 断言当 f_0 全部增加时，预测值 pred 是否全都增加
    X = np.c_[constant, linspace, constant, constant, constant]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert is_increasing(pred)
    # 断言预测值是否实际上遵循 f_0 的变化
    X = np.c_[constant, sin, constant, constant, constant]
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    pred = gbdt.predict(X)
    assert np.all((np.diff(pred) >= 0) == (np.diff(sin) >= 0))

    # 第二个非类别特征（负约束）
    # 断言：当 f_1 是完全增加时，预测值 pred 应完全减少
    X = np.c_[constant, constant, constant, linspace, constant]
    # 将 X 转换为指定构造函数名和列名的数据容器
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    # 使用 GBDT 模型对 X 进行预测
    pred = gbdt.predict(X)
    # 断言：预测值 pred 应当是递减的
    assert is_decreasing(pred)
    
    # 断言：预测值 pred 应当与 f_1 的反向变化一致
    X = np.c_[constant, constant, constant, sin, constant]
    # 将 X 转换为指定构造函数名和列名的数据容器
    X = _convert_container(X, constructor_name, columns_name=columns_name)
    # 使用 GBDT 模型对 X 进行预测
    pred = gbdt.predict(X)
    # 断言：预测值 pred 的差分应与 sin 的差分关系一致
    assert ((np.diff(pred) <= 0) == (np.diff(sin) >= 0)).all()
# 定义一个测试函数，用于测试输入错误情况
def test_bounded_value_min_gain_to_split():
    # 设定一个正则化参数
    l2_regularization = 0
    # 设置分裂时的最小 Hessian 约束
    min_hessian_to_split = 0
    # 设置叶子节点的最小样本数
    min_samples_leaf = 1
    # 设定总样本数和分箱数
    n_bins = n_samples = 5
    # 创建一个 numpy 数组 X_binned，包含从 0 到 n_samples-1 的整数，reshape 成列数为 1 的二维数组，并转换为指定的数据类型 X_BINNED_DTYPE
    X_binned = np.arange(n_samples).reshape(-1, 1).astype(X_BINNED_DTYPE)
    # 创建一个 numpy 数组 sample_indices，包含从 0 到 n_samples-1 的整数，数据类型为 np.uint32
    sample_indices = np.arange(n_samples, dtype=np.uint32)
    # 创建一个 numpy 数组 all_hessians，包含 n_samples 个元素，每个元素为 1，数据类型为 G_H_DTYPE
    all_hessians = np.ones(n_samples, dtype=G_H_DTYPE)
    # 创建一个 numpy 数组 all_gradients，包含指定的整数序列，数据类型为 G_H_DTYPE
    all_gradients = np.array([1, 1, 100, 1, 1], dtype=G_H_DTYPE)
    # 计算 all_gradients 数组的总和
    sum_gradients = all_gradients.sum()
    # 计算 all_hessians 数组的总和
    sum_hessians = all_hessians.sum()
    # 初始化 hessians_are_constant 变量为 False
    hessians_are_constant = False

    # 使用 HistogramBuilder 类创建 builder 对象，传入相关参数初始化
    builder = HistogramBuilder(
        X_binned, n_bins, all_gradients, all_hessians, hessians_are_constant, n_threads
    )
    # 创建一个 numpy 数组 n_bins_non_missing，包含 X_binned.shape[1] 个元素，每个元素为 n_bins - 1，数据类型为 np.uint32
    n_bins_non_missing = np.array([n_bins - 1] * X_binned.shape[1], dtype=np.uint32)
    # 创建一个 numpy 数组 has_missing_values，包含 X_binned.shape[1] 个元素，每个元素为 False，数据类型为 np.uint8
    has_missing_values = np.array([False] * X_binned.shape[1], dtype=np.uint8)
    # 创建一个 numpy 数组 monotonic_cst，包含 X_binned.shape[1] 个元素，每个元素为 MonotonicConstraint.NO_CST，数据类型为 np.int8
    monotonic_cst = np.array(
        [MonotonicConstraint.NO_CST] * X_binned.shape[1], dtype=np.int8
    )
    # 创建一个 numpy 数组 is_categorical，和 monotonic_cst 相同形状的数组，元素初始化为 0，数据类型为 np.uint8
    is_categorical = np.zeros_like(monotonic_cst, dtype=np.uint8)
    # 设置 missing_values_bin_idx 为 n_bins - 1
    missing_values_bin_idx = n_bins - 1
    # 初始化 children_lower_bound 和 children_upper_bound 分别为负无穷和正无穷
    children_lower_bound, children_upper_bound = -np.inf, np.inf

    # 设置 min_gain_to_split 为 2000
    min_gain_to_split = 2000
    # 使用 Splitter 类创建 splitter 对象，传入相关参数初始化
    splitter = Splitter(
        X_binned,
        n_bins_non_missing,
        missing_values_bin_idx,
        has_missing_values,
        is_categorical,
        monotonic_cst,
        l2_regularization,
        min_hessian_to_split,
        min_samples_leaf,
        min_gain_to_split,
        hessians_are_constant,
    )

    # 调用 builder 对象的 compute_histograms_brute 方法计算直方图
    histograms = builder.compute_histograms_brute(sample_indices)

    # 计算节点值，传入相关参数并赋值给 value
    value = compute_node_value(
        sum_gradients,
        sum_hessians,
        current_lower_bound,
        current_upper_bound,
        l2_regularization,
    )
    # 断言 value 大致等于 -104 / 5
    assert value == pytest.approx(-104 / 5)

    # 使用 splitter 对象的 find_node_split 方法查找节点的分裂信息，传入相关参数并赋值给 split_info
    split_info = splitter.find_node_split(
        n_samples,
        histograms,
        sum_gradients,
        sum_hessians,
        value,
        lower_bound=children_lower_bound,
        upper_bound=children_upper_bound,
    )
    # 断言 split_info 的 gain 等于 -1，说明未能满足 min_gain_to_split 的限制条件

    # 重新设置 current_lower_bound 和 current_upper_bound 分别为 -10 和正无穷
    current_lower_bound, current_upper_bound = -10, np.inf
    # 重新计算节点值，传入相关参数并赋值给 value
    value = compute_node_value(
        sum_gradients,
        sum_hessians,
        current_lower_bound,
        current_upper_bound,
        l2_regularization,
    )
    # 断言 value 等于 -10
    assert value == -10

    # 再次使用 splitter 对象的 find_node_split 方法查找节点的分裂信息，传入相关参数并赋值给 split_info
    split_info = splitter.find_node_split(
        n_samples,
        histograms,
        sum_gradients,
        sum_hessians,
        value,
        lower_bound=children_lower_bound,
        upper_bound=children_upper_bound,
    )
    # 断言，用于检查分裂信息的增益是否大于指定的最小分裂增益阈值
    assert split_info.gain > min_gain_to_split
```