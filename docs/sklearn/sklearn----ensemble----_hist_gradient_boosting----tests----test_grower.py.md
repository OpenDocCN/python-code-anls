# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_grower.py`

```
# 导入必要的库
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

# 导入需要测试的模块和函数
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import (
    G_H_DTYPE,
    X_BINNED_DTYPE,
    X_BITSET_INNER_DTYPE,
    X_DTYPE,
    Y_DTYPE,
)
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

# 获取当前可用的线程数
n_threads = _openmp_effective_n_threads()

# 函数：生成训练数据
def _make_training_data(n_bins=256, constant_hessian=True):
    # 设置随机种子
    rng = np.random.RandomState(42)
    # 样本数
    n_samples = 10000

    # 生成已经分箱的测试数据，用于测试生长器代码而不依赖于分箱逻辑
    X_binned = rng.randint(0, n_bins - 1, size=(n_samples, 2), dtype=X_BINNED_DTYPE)
    X_binned = np.asfortranarray(X_binned)

    # 定义真实的决策函数
    def true_decision_function(input_features):
        """Ground truth decision function

        This is a very simple yet asymmetric decision tree. Therefore the
        grower code should have no trouble recovering the decision function
        from 10000 training samples.
        """
        if input_features[0] <= n_bins // 2:
            return -1
        else:
            return -1 if input_features[1] <= n_bins // 3 else 1

    # 目标变量：根据真实决策函数生成目标值
    target = np.array([true_decision_function(x) for x in X_binned], dtype=Y_DTYPE)

    # 假设应用于初始模型的平方损失，该模型始终预测0（针对此测试固定为零）：
    all_gradients = target.astype(G_H_DTYPE)
    shape_hessians = 1 if constant_hessian else all_gradients.shape
    all_hessians = np.ones(shape=shape_hessians, dtype=G_H_DTYPE)

    return X_binned, all_gradients, all_hessians


# 函数：检查节点的子节点一致性
def _check_children_consistency(parent, left, right):
    # 确保样本正确地分配给父节点的两个子节点
    assert parent.left_child is left
    assert parent.right_child is right

    # 检查所有样本是否被正确分配到左右子节点
    assert len(left.sample_indices) + len(right.sample_indices) == len(
        parent.sample_indices
    )

    # 检查所有样本的集合是否等于父节点样本的集合
    assert set(left.sample_indices).union(set(right.sample_indices)) == set(
        parent.sample_indices
    )

    # 确保样本要么被分配到左子节点，要么被分配到右子节点，不会同时被两个节点接受
    assert set(left.sample_indices).intersection(set(right.sample_indices)) == set()


# 参数化测试：测试不同的参数组合
@pytest.mark.parametrize(
    "n_bins, constant_hessian, stopping_param, shrinkage",
    [
        (11, True, "min_gain_to_split", 0.5),
        (11, False, "min_gain_to_split", 1.0),
        (11, True, "max_leaf_nodes", 1.0),
        (11, False, "max_leaf_nodes", 0.1),
        (42, True, "max_leaf_nodes", 0.01),
        (42, False, "max_leaf_nodes", 1.0),
        (256, True, "min_gain_to_split", 1.0),
        (256, True, "max_leaf_nodes", 0.1),
    ],
)
# 定义一个函数用于测试决策树生长过程，根据给定的参数设置生成训练数据
def test_grow_tree(n_bins, constant_hessian, stopping_param, shrinkage):
    # 调用内部函数生成训练数据：包括特征数据 X_binned，所有样本的梯度 all_gradients，所有样本的 Hessian 矩阵 all_hessians
    X_binned, all_gradients, all_hessians = _make_training_data(
        n_bins=n_bins, constant_hessian=constant_hessian
    )
    # 获取样本数目
    n_samples = X_binned.shape[0]

    # 根据停止参数设置，决定生长树的停止条件
    if stopping_param == "max_leaf_nodes":
        stopping_param = {"max_leaf_nodes": 3}
    else:
        stopping_param = {"min_gain_to_split": 0.01}

    # 创建 TreeGrower 对象，用于生长决策树
    grower = TreeGrower(
        X_binned,
        all_gradients,
        all_hessians,
        n_bins=n_bins,
        shrinkage=shrinkage,
        min_samples_leaf=1,
        **stopping_param,
    )

    # 断言：根节点还未分裂，但已经评估出最佳分裂条件
    assert grower.root.left_child is None
    assert grower.root.right_child is None

    # 获取根节点的分裂信息
    root_split = grower.root.split_info
    assert root_split.feature_idx == 0
    assert root_split.bin_idx == n_bins // 2
    assert len(grower.splittable_nodes) == 1

    # 调用 split_next() 方法进行下一步的分裂操作，计算每个新子节点的最佳分裂
    left_node, right_node = grower.split_next()

    # 断言：所有训练样本已被分裂到两个节点中，大约50%/50%
    _check_children_consistency(grower.root, left_node, right_node)
    assert len(left_node.sample_indices) > 0.4 * n_samples
    assert len(left_node.sample_indices) < 0.6 * n_samples

    # 如果 min_gain_to_split 大于 0，则左子节点过于纯净，无法获得足够的增益再次分裂
    if grower.min_gain_to_split > 0:
        assert left_node.split_info.gain < grower.min_gain_to_split
        assert left_node in grower.finalized_leaves

    # 右子节点仍可进一步分裂，这次在特征 #1 上进行分裂
    split_info = right_node.split_info
    assert split_info.gain > 1.0
    assert split_info.feature_idx == 1
    assert split_info.bin_idx == n_bins // 3
    assert right_node.left_child is None
    assert right_node.right_child is None

    # 断言：右分裂还未应用，现在应用它：
    assert len(grower.splittable_nodes) == 1
    right_left_node, right_right_node = grower.split_next()
    _check_children_consistency(right_node, right_left_node, right_right_node)

    # 断言：左子节点样本数大约为总样本数的 10%-20%
    assert len(right_left_node.sample_indices) > 0.1 * n_samples
    assert len(right_left_node.sample_indices) < 0.2 * n_samples

    # 断言：右子节点样本数大约为总样本数的 20%-40%
    assert len(right_right_node.sample_indices) > 0.2 * n_samples
    assert len(right_right_node.sample_indices) < 0.4 * n_samples

    # 所有叶子节点都已纯净，无法再进行更多分裂
    assert not grower.splittable_nodes

    # 应用缩减系数（shrinkage）到叶子节点的值
    grower._apply_shrinkage()

    # 检查叶子节点的值是否正确
    assert grower.root.left_child.value == approx(shrinkage)
    assert grower.root.right_child.left_child.value == approx(shrinkage)
    assert grower.root.right_child.right_child.value == approx(-shrinkage, rel=1e-3)


# 测试函数：从已生成的 TreeGrower 中构建预测器
def test_predictor_from_grower():
    # 使用 256 个 bin 的玩具数据集构建一个决策树，以提取预测器
    n_bins = 256
    # 使用 `_make_training_data` 函数生成训练数据的分箱数据、所有梯度和海森矩阵
    X_binned, all_gradients, all_hessians = _make_training_data(n_bins=n_bins)
    
    # 使用 `TreeGrower` 类构建决策树生长器对象
    grower = TreeGrower(
        X_binned,                # 分箱后的训练数据
        all_gradients,           # 所有样本的梯度
        all_hessians,            # 所有样本的海森矩阵
        n_bins=n_bins,           # 分箱数目
        shrinkage=1.0,           # 收缩率
        max_leaf_nodes=3,        # 最大叶子节点数
        min_samples_leaf=5,      # 每个叶子节点的最小样本数
    )
    
    # 执行决策树的生长过程
    grower.grow()
    
    # 断言决策树生长器对象中节点数目为5（2个决策节点 + 3个叶子节点）
    assert grower.n_nodes == 5  # (2 decision nodes + 3 leaves)

    # 检查节点结构能否被转换为预测器对象，以便进行大规模预测
    # 传递未定义的分箱阈值，因为我们将不会使用预测功能
    predictor = grower.make_predictor(
        binning_thresholds=np.zeros((X_binned.shape[1], n_bins))
    )
    
    # 断言预测器对象中节点数目为5
    assert predictor.nodes.shape[0] == 5
    
    # 断言预测器对象中叶子节点的数量为3
    assert predictor.nodes["is_leaf"].sum() == 3

    # 探测树的每个叶子节点的一些预测结果
    # 每组3个样本对应于 `_make_training_data` 中的一个条件
    input_data = np.array(
        [
            [0, 0],
            [42, 99],
            [128, 254],
            [129, 0],
            [129, 85],
            [254, 85],
            [129, 86],
            [129, 254],
            [242, 100],
        ],
        dtype=np.uint8,
    )
    
    # 缺失值的分箱索引为 `n_bins - 1`
    predictions = predictor.predict_binned(
        input_data, missing_values_bin_idx, n_threads
    )
    
    # 期望的目标结果
    expected_targets = [1, 1, 1, 1, 1, 1, -1, -1, -1]
    
    # 断言预测结果与期望目标非常接近
    assert np.allclose(predictions, expected_targets)

    # 检查训练集是否可以完全恢复：
    predictions = predictor.predict_binned(X_binned, missing_values_bin_idx, n_threads)
    
    # 断言预测结果与所有样本的梯度相近
    assert np.allclose(predictions, -all_gradients)
@pytest.mark.parametrize(
    "n_samples, min_samples_leaf, n_bins, constant_hessian, noise",
    [
        (11, 10, 7, True, 0),  # 参数化测试：设置不同的测试参数
        (13, 10, 42, False, 0),  # 参数化测试：设置不同的测试参数
        (56, 10, 255, True, 0.1),  # 参数化测试：设置不同的测试参数
        (101, 3, 7, True, 0),  # 参数化测试：设置不同的测试参数
        (200, 42, 42, False, 0),  # 参数化测试：设置不同的测试参数
        (300, 55, 255, True, 0.1),  # 参数化测试：设置不同的测试参数
        (300, 301, 255, True, 0.1),  # 参数化测试：设置不同的测试参数
    ],
)
def test_min_samples_leaf(n_samples, min_samples_leaf, n_bins, constant_hessian, noise):
    rng = np.random.RandomState(seed=0)
    # data = linear target, 3 features, 1 irrelevant.
    X = rng.normal(size=(n_samples, 3))  # 生成服从正态分布的数据作为特征矩阵
    y = X[:, 0] - X[:, 1]  # 根据特征生成目标变量
    if noise:
        y_scale = y.std()  # 计算目标变量的标准差
        y += rng.normal(scale=noise, size=n_samples) * y_scale  # 添加噪声到目标变量中
    mapper = _BinMapper(n_bins=n_bins)  # 初始化数据的分箱映射器
    X = mapper.fit_transform(X)  # 对特征矩阵进行分箱处理

    all_gradients = y.astype(G_H_DTYPE)  # 将目标变量转换为指定类型
    shape_hessian = 1 if constant_hessian else all_gradients.shape  # 根据常数 hessian 参数确定 hessian 的形状
    all_hessians = np.ones(shape=shape_hessian, dtype=G_H_DTYPE)  # 创建全为 1 的 hessian 矩阵
    grower = TreeGrower(
        X,
        all_gradients,
        all_hessians,
        n_bins=n_bins,
        shrinkage=1.0,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=n_samples,
    )  # 初始化树的生长器对象
    grower.grow()  # 执行树的生长过程
    predictor = grower.make_predictor(binning_thresholds=mapper.bin_thresholds_)  # 创建预测器对象

    if n_samples >= min_samples_leaf:
        for node in predictor.nodes:
            if node["is_leaf"]:
                assert node["count"] >= min_samples_leaf  # 检查叶子节点中的样本数是否符合最小要求
    else:
        assert predictor.nodes.shape[0] == 1  # 如果样本数小于最小叶子样本数要求，预测器应只有一个节点
        assert predictor.nodes[0]["is_leaf"]  # 确保预测器的唯一节点是叶子节点
        assert predictor.nodes[0]["count"] == n_samples  # 确保预测器的唯一节点的样本数与输入的样本数相同


@pytest.mark.parametrize("n_samples, min_samples_leaf", [(99, 50), (100, 50)])
def test_min_samples_leaf_root(n_samples, min_samples_leaf):
    # 确保如果样本数不至少是最小叶子样本数的两倍，根节点不会分裂
    rng = np.random.RandomState(seed=0)

    n_bins = 256

    # data = linear target, 3 features, 1 irrelevant.
    X = rng.normal(size=(n_samples, 3))  # 生成服从正态分布的数据作为特征矩阵
    y = X[:, 0] - X[:, 1]  # 根据特征生成目标变量
    mapper = _BinMapper(n_bins=n_bins)  # 初始化数据的分箱映射器
    X = mapper.fit_transform(X)  # 对特征矩阵进行分箱处理

    all_gradients = y.astype(G_H_DTYPE)  # 将目标变量转换为指定类型
    all_hessians = np.ones(shape=1, dtype=G_H_DTYPE)  # 创建全为 1 的 hessian 矩阵
    grower = TreeGrower(
        X,
        all_gradients,
        all_hessians,
        n_bins=n_bins,
        shrinkage=1.0,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=n_samples,
    )  # 初始化树的生长器对象
    grower.grow()  # 执行树的生长过程
    if n_samples >= min_samples_leaf * 2:
        assert len(grower.finalized_leaves) >= 2  # 确保至少有两个最终的叶子节点
    else:
        assert len(grower.finalized_leaves) == 1  # 如果样本数不足两倍的最小叶子样本数要求，应该只有一个最终的叶子节点


def assert_is_stump(grower):
    # 断言当 max_depth=1 时，确保生成的是决策树桩
    for leaf in (grower.root.left_child, grower.root.right_child):
        assert leaf.left_child is None  # 确保左子节点为空
        assert leaf.right_child is None  # 确保右子节点为空


@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_max_depth(max_depth):
    # 确保 max_depth 参数按预期工作
    rng = np.random.RandomState(seed=0)

    n_bins = 256
    n_samples = 1000
    # 生成一个具有指定形状（n_samples行，3列）的服从标准正态分布的随机数矩阵，作为特征矩阵 X
    X = rng.normal(size=(n_samples, 3))
    # 从特征矩阵 X 中选择第一列和第二列的值进行运算，作为目标向量 y
    y = X[:, 0] - X[:, 1]
    # 初始化一个 _BinMapper 对象，用于将特征矩阵 X 进行映射和转换
    mapper = _BinMapper(n_bins=n_bins)
    # 使用 _BinMapper 对象对特征矩阵 X 进行映射和转换，更新 X
    X = mapper.fit_transform(X)

    # 将目标向量 y 转换为 G_H_DTYPE 类型的数组，作为所有叶子节点的梯度
    all_gradients = y.astype(G_H_DTYPE)
    # 创建一个形状为 (1,)、数据类型为 G_H_DTYPE 的数组，作为所有叶子节点的 Hessian 矩阵
    all_hessians = np.ones(shape=1, dtype=G_H_DTYPE)
    # 初始化一个 TreeGrower 对象，用于构建和生长决策树
    grower = TreeGrower(X, all_gradients, all_hessians, max_depth=max_depth)
    # 执行决策树的生长过程
    grower.grow()

    # 计算所有最终叶子节点的深度，并取最大深度
    depth = max(leaf.depth for leaf in grower.finalized_leaves)
    # 断言最大深度与设定的最大深度 max_depth 相等
    assert depth == max_depth

    # 如果 max_depth 等于 1，断言当前决策树为一棵单节点树（stump）
    if max_depth == 1:
        assert_is_stump(grower)
# 定`
def test_input_validation():
    # 调用私有方法生成训练数据，返回经过分箱处理的特征、所有梯度和所有海森矩阵
    X_binned, all_gradients, all_hessians = _make_training_data()

    # 将 X_binned 转换为 float32 类型
    X_binned_float = X_binned.astype(np.float32)
    # 检查 X_binned 的类型，如果不是 uint8 类型，则预期抛出 NotImplementedError 异常
    with pytest.raises(NotImplementedError, match="X_binned must be of type uint8"):
        TreeGrower(X_binned_float, all_gradients, all_hessians)

    # 将 X_binned 转换为 Fortran 连续数组
    X_binned_C_array = np.ascontiguousarray(X_binned)
    # 检查 X_binned 是否为 Fortran 连续数组，如果不是，预期抛出 ValueError 异常
    with pytest.raises(
        ValueError, match="X_binned should be passed as Fortran contiguous array"
    ):
        TreeGrower(X_binned_C_array, all_gradients, all_hessians)


def test_init_parameters_validation():
    # 调用私有方法生成训练数据，返回经过分箱处理的特征、所有梯度和所有海森矩阵
    X_binned, all_gradients, all_hessians = _make_training_data()
    # 检查 min_gain_to_split 参数是否大于 0，若小于等于 0，预期抛出 ValueError 异常
    with pytest.raises(ValueError, match="min_gain_to_split=-1 must be positive"):
        TreeGrower(X_binned, all_gradients, all_hessians, min_gain_to_split=-1)

    # 检查 min_hessian_to_split 参数是否大于 0，若小于等于 0，预期抛出 ValueError 异常
    with pytest.raises(ValueError, match="min_hessian_to_split=-1 must be positive"):
        TreeGrower(X_binned, all_gradients, all_hessians, min_hessian_to_split=-1)


def test_missing_value_predict_only():
    # 确保即使在训练数据中未遇到缺失值的情况下，预测时也支持缺失值：缺失值会被分配到样本最多的子节点

    # 初始化随机数生成器，设置随机种子
    rng = np.random.RandomState(0)
    n_samples = 100
    # 生成指定范围内的随机整数，作为分箱特征
    X_binned = rng.randint(0, 256, size=(n_samples, 1), dtype=np.uint8)
    # 转换为 Fortran 连续数组
    X_binned = np.asfortranarray(X_binned)

    # 生成随机梯度数据，服从正态分布
    gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
    # 创建海森矩阵，填充为 1
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)

    # 初始化 TreeGrower 对象，设置 min_samples_leaf 为 5，并指定无缺失值
    grower = TreeGrower(
        X_binned, gradients, hessians, min_samples_leaf=5, has_missing_values=False
    )
    # 执行生长操作
    grower.grow()

    # 创建预测器，传入 binning_thresholds 设置为空数组，因为预测不使用该参数
    predictor = grower.make_predictor(
        binning_thresholds=np.zeros((X_binned.shape[1], X_binned.max() + 1))
    )

    # 从根节点开始，沿着样本数量最多的子节点遍历，直到达到叶节点，这是缺失值应该经过的路径
    node = predictor.nodes[0]
    while not node["is_leaf"]:
        left = predictor.nodes[node["left"]]
        right = predictor.nodes[node["right"]]
        node = left if left["count"] > right["count"] else right

    # 记录主路径的预测值
    prediction_main_path = node["value"]

    # 创建测试数据 X_test，所有特征值为 NaN，并确保所有预测结果等于主路径的预测值
    all_nans = np.full(shape=(n_samples, 1), fill_value=np.nan)
    # 初始化已知类别的位集，空数组
    known_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    # 初始化特征索引映射，空数组
    f_idx_map = np.zeros(0, dtype=np.uint32)

    # 执行预测，传入测试数据和必要参数
    y_pred = predictor.predict(all_nans, known_cat_bitsets, f_idx_map, n_threads)
    # 检查所有预测值是否与主路径的预测值相等
    assert np.all(y_pred == prediction_main_path)


def test_split_on_nan_with_infinite_values():
    # 确保在存在正无穷值的样本情况下，分割时对 NaN 的处理得当（我们将阈值设为正无穷，因此此测试确保没有引入边界情况错误）。
    # 需要使用私有 API 以便测试更多细节
    # predict_binned().

    # 创建一个包含特定值和缺失值的一维数组，并将其重塑为列向量
    X = np.array([0, 1, np.inf, np.nan, np.nan]).reshape(-1, 1)
    # 梯度值将强制在NaN情况下进行分割
    gradients = np.array([0, 0, 0, 100, 100], dtype=G_H_DTYPE)
    # 全部使用单位矩阵
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)

    # 创建一个 _BinMapper 对象
    bin_mapper = _BinMapper()
    # 对输入数据 X 进行拟合和转换
    X_binned = bin_mapper.fit_transform(X)

    # 非缺失值的箱数
    n_bins_non_missing = 3
    # 存在缺失值
    has_missing_values = True
    # 创建 TreeGrower 对象
    grower = TreeGrower(
        X_binned,
        gradients,
        hessians,
        n_bins_non_missing=n_bins_non_missing,
        has_missing_values=has_missing_values,
        min_samples_leaf=1,
        n_threads=n_threads,
    )

    # 执行生长操作
    grower.grow()

    # 创建预测器对象，并指定分箱的阈值
    predictor = grower.make_predictor(binning_thresholds=bin_mapper.bin_thresholds_)

    # 确认预测器对NaN进行了正确的分割
    assert predictor.nodes[0]["num_threshold"] == np.inf
    assert predictor.nodes[0]["bin_threshold"] == n_bins_non_missing - 1

    # 获取已知类别的位集合和特征索引映射
    known_cat_bitsets, f_idx_map = bin_mapper.make_known_categories_bitsets()

    # 确保特定的 +inf 样本被映射到左子节点
    # 注意，LightGBM 在这里会“失败”，会将 +inf 样本分配给右子节点，尽管实际上是“在NaN上分割”的情况
    # 进行预测
    predictions = predictor.predict(X, known_cat_bitsets, f_idx_map, n_threads)
    # 使用分箱数据进行预测
    predictions_binned = predictor.predict_binned(
        X_binned,
        missing_values_bin_idx=bin_mapper.missing_values_bin_idx_,
        n_threads=n_threads,
    )
    # 使用 np.testing.assert_allclose 确保预测结果与梯度的负数非常接近
    np.testing.assert_allclose(predictions, -gradients)
    np.testing.assert_allclose(predictions_binned, -gradients)
# 定义一个测试函数，用于测试分类变量在生长预测树时的行为
def test_grow_tree_categories():
    # 创建一个二维数组 X_binned，包含22个元素，其中前21个元素交替为0和1，最后一个元素为1，使用 X_BINNED_DTYPE 类型
    X_binned = np.array([[0, 1] * 11 + [1]], dtype=X_BINNED_DTYPE).T
    # 将 X_binned 转换为 Fortran 顺序数组
    X_binned = np.asfortranarray(X_binned)

    # 创建一个包含22个元素的数组 all_gradients，其中前21个元素为10和1交替，最后一个元素为1，使用 G_H_DTYPE 类型
    all_gradients = np.array([10, 1] * 11 + [1], dtype=G_H_DTYPE)
    # 创建一个包含1个元素的全为1的数组 all_hessians，使用 G_H_DTYPE 类型
    all_hessians = np.ones(1, dtype=G_H_DTYPE)
    # 创建一个包含1个元素的全为1的数组 is_categorical，使用 np.uint8 类型
    is_categorical = np.ones(1, dtype=np.uint8)

    # 使用 TreeGrower 类初始化一个 grower 对象，传入必要的参数
    grower = TreeGrower(
        X_binned,
        all_gradients,
        all_hessians,
        n_bins=4,
        shrinkage=1.0,
        min_samples_leaf=1,
        is_categorical=is_categorical,
        n_threads=n_threads,
    )
    # 调用 grower 对象的 grow 方法，开始生长树
    grower.grow()
    # 断言 grower 的节点数量为3
    assert grower.n_nodes == 3

    # 创建一个包含一个数组的列表 categories，数组中包含两个元素，使用 X_DTYPE 类型
    categories = [np.array([4, 9], dtype=X_DTYPE)]
    # 使用 grower 对象的 make_predictor 方法，传入 binning_thresholds 参数创建一个 predictor 对象
    predictor = grower.make_predictor(binning_thresholds=categories)
    # 获取 predictor 对象的根节点
    root = predictor.nodes[0]
    # 断言根节点的 count 属性为23
    assert root["count"] == 23
    # 断言根节点的 depth 属性为0
    assert root["depth"] == 0
    # 断言根节点是分类节点
    assert root["is_categorical"]

    # 获取根节点的左右子节点
    left, right = predictor.nodes[root["left"]], predictor.nodes[root["right"]]

    # 简单验证，确保左子节点的样本数大于等于右子节点的样本数
    assert left["count"] >= right["count"]

    # 检查 binned 类别值（1）是否正确
    expected_binned_cat_bitset = [2**1] + [0] * 7
    binned_cat_bitset = predictor.binned_left_cat_bitsets
    assert_array_equal(binned_cat_bitset[0], expected_binned_cat_bitset)

    # 检查 raw 类别值（9）是否正确
    expected_raw_cat_bitsets = [2**9] + [0] * 7
    raw_cat_bitsets = predictor.raw_left_cat_bitsets
    assert_array_equal(raw_cat_bitsets[0], expected_raw_cat_bitsets)

    # 注意，由于训练过程中没有缺失值，缺失值不会出现在位集中。
    # 但我们期望缺失值会被分到最大的子节点（即左子节点）。
    # 左子节点具有值 -1 = 负梯度。
    assert root["missing_go_to_left"]

    # 确保预测时缺失值被映射到左子节点
    prediction_binned = predictor.predict_binned(
        np.asarray([[6]]).astype(X_BINNED_DTYPE),
        missing_values_bin_idx=6,
        n_threads=n_threads,
    )
    # 断言预测结果接近于 -1，即负梯度
    assert_allclose(prediction_binned, [-1])

    # 确保在预测时原始的缺失值被映射到左子节点
    known_cat_bitsets = np.zeros((1, 8), dtype=np.uint32)  # 无论如何被忽略
    f_idx_map = np.array([0], dtype=np.uint32)
    prediction = predictor.predict(
        np.array([[np.nan]]), known_cat_bitsets, f_idx_map, n_threads
    )
    # 断言预测结果接近于 -1，即负梯度
    assert_allclose(prediction, [-1])


# 使用 pytest 的参数化装饰器对以下三个参数进行组合测试：min_samples_leaf, n_unique_categories, target
@pytest.mark.parametrize("min_samples_leaf", (1, 20))
@pytest.mark.parametrize("n_unique_categories", (2, 10, 100))
@pytest.mark.parametrize("target", ("binary", "random", "equal"))
def test_ohe_equivalence(min_samples_leaf, n_unique_categories, target):
    # 确保当树的深度足够时，原生的分类分割等同于使用独热编码（OHE）

    # 创建一个随机数生成器 rng，种子为0
    rng = np.random.RandomState(0)
    # 设定样本数为10000
    n_samples = 10_000
    # 使用随机数生成器rng生成一个二维数组X_binned，每个元素均为0到n_unique_categories之间的随机整数，数据类型为uint8
    X_binned = rng.randint(0, n_unique_categories, size=(n_samples, 1), dtype=np.uint8)

    # 使用OneHotEncoder将X_binned进行独热编码，得到稠密的编码结果X_ohe，并将其转换为Fortran风格的数组，数据类型为uint8
    X_ohe = OneHotEncoder(sparse_output=False).fit_transform(X_binned)
    X_ohe = np.asfortranarray(X_ohe).astype(np.uint8)

    # 根据目标类型选择对应的梯度计算方式：如果目标是"equal"，则将X_binned扁平化作为梯度；如果是"binary"，则取X_binned取余2作为梯度；否则使用随机数生成器rng生成的随机数作为梯度
    if target == "equal":
        gradients = X_binned.reshape(-1)
    elif target == "binary":
        gradients = (X_binned % 2).reshape(-1)
    else:
        gradients = rng.randn(n_samples)
    # 将梯度数据类型转换为指定的G_H_DTYPE类型
    gradients = gradients.astype(G_H_DTYPE)

    # 创建全为1的Hessian矩阵，数据类型为G_H_DTYPE
    hessians = np.ones(shape=1, dtype=G_H_DTYPE)

    # 定义TreeGrower的参数，用于树的生长过程
    grower_params = {
        "min_samples_leaf": min_samples_leaf,
        "max_depth": None,
        "max_leaf_nodes": None,
    }

    # 使用TreeGrower对X_binned数据进行树的生长，传入梯度和Hessian矩阵，以及是否为分类变量的标志
    grower = TreeGrower(
        X_binned, gradients, hessians, is_categorical=[True], **grower_params
    )
    grower.grow()  # 执行树的生长过程

    # 创建基于grower对象的预测器，传入空的binning_thresholds因为不会使用predict()方法
    predictor = grower.make_predictor(
        binning_thresholds=np.zeros((1, n_unique_categories))
    )
    # 使用predict_binned方法对X_binned进行预测，设置255为缺失值的索引，使用n_threads线程数进行并行预测
    preds = predictor.predict_binned(
        X_binned, missing_values_bin_idx=255, n_threads=n_threads
    )

    # 对X_ohe数据进行类似的树生长过程
    grower_ohe = TreeGrower(X_ohe, gradients, hessians, **grower_params)
    grower_ohe.grow()  # 执行树的生长过程

    # 创建基于grower_ohe对象的预测器，传入binning_thresholds矩阵作为阈值
    predictor_ohe = grower_ohe.make_predictor(
        binning_thresholds=np.zeros((X_ohe.shape[1], n_unique_categories))
    )
    # 使用predict_binned方法对X_ohe进行预测，设置255为缺失值的索引，使用n_threads线程数进行并行预测
    preds_ohe = predictor_ohe.predict_binned(
        X_ohe, missing_values_bin_idx=255, n_threads=n_threads
    )

    # 断言：确保普通编码(predictor)的树的最大深度小于等于独热编码(predictor_ohe)的树的最大深度
    assert predictor.get_max_depth() <= predictor_ohe.get_max_depth()

    # 如果目标为"binary"且n_unique_categories大于2，则断言普通编码的树的最大深度必须小于独热编码的树的最大深度
    if target == "binary" and n_unique_categories > 2:
        assert predictor.get_max_depth() < predictor_ohe.get_max_depth()

    # 使用np.testing.assert_allclose函数断言普通编码(predicts)和独热编码(predicts_ohe)的预测结果近似相等
    np.testing.assert_allclose(preds, preds_ohe)
def test_grower_interaction_constraints():
    """Check that grower respects interaction constraints."""
    # 定义特征数量
    n_features = 6
    # 定义交互约束集合
    interaction_cst = [{0, 1}, {1, 2}, {3, 4, 5}]
    # 定义样本数量
    n_samples = 10
    # 定义分箱数
    n_bins = 6
    # 初始化根节点的特征分裂列表为空
    root_feature_splits = []

    def get_all_children(node):
        # 递归函数，获取给定节点的所有子节点
        res = []
        if node.is_leaf:
            return res
        for n in [node.left_child, node.right_child]:
            res.append(n)
            res.extend(get_all_children(n))
        return res

    # 遍历不同的随机种子
    for seed in range(20):
        rng = np.random.RandomState(seed)

        # 生成随机整数矩阵 X_binned，表示分箱后的特征矩阵
        X_binned = rng.randint(
            0, n_bins - 1, size=(n_samples, n_features), dtype=X_BINNED_DTYPE
        )
        # 将 X_binned 转换为 Fortran 风格数组
        X_binned = np.asfortranarray(X_binned)
        # 生成随机梯度数据
        gradients = rng.normal(size=n_samples).astype(G_H_DTYPE)
        # 生成全为1的随机 Hessian 数据
        hessians = np.ones(shape=1, dtype=G_H_DTYPE)

        # 创建 TreeGrower 对象
        grower = TreeGrower(
            X_binned,
            gradients,
            hessians,
            n_bins=n_bins,
            min_samples_leaf=1,
            interaction_cst=interaction_cst,
            n_threads=n_threads,
        )
        # 执行树的生长过程
        grower.grow()

        # 获取根节点的特征索引
        root_feature_idx = grower.root.split_info.feature_idx
        root_feature_splits.append(root_feature_idx)

        # 定义特征索引到约束集合的映射字典
        feature_idx_to_constraint_set = {
            0: {0, 1},
            1: {0, 1, 2},
            2: {1, 2},
            3: {3, 4, 5},
            4: {3, 4, 5},
            5: {3, 4, 5},
        }

        # 获取根节点的约束集合
        root_constraint_set = feature_idx_to_constraint_set[root_feature_idx]
        # 检查根节点的子节点的允许特征必须等于根节点的约束集合
        for node in (grower.root.left_child, grower.root.right_child):
            assert_array_equal(node.allowed_features, list(root_constraint_set))
        # 对于所有节点（非叶子节点），确保每个节点使用其父节点特征的子集
        for node in get_all_children(grower.root):
            if node.is_leaf:
                continue
            parent_interaction_cst_indices = set(node.interaction_cst_indices)
            right_interactions_cst_indices = set(
                node.right_child.interaction_cst_indices
            )
            left_interactions_cst_indices = set(node.left_child.interaction_cst_indices)

            # 断言右子节点的交互约束索引是父节点的交互约束索引的子集
            assert right_interactions_cst_indices.issubset(
                parent_interaction_cst_indices
            )
            # 断言左子节点的交互约束索引是父节点的交互约束索引的子集
            assert left_interactions_cst_indices.issubset(
                parent_interaction_cst_indices
            )
            # 断言当前节点的分裂特征索引存在于根节点的约束集合中
            assert node.split_info.feature_idx in root_constraint_set

    # 确保根节点使用的特征分裂包含了所有特征
    assert (
        len(set(root_feature_splits))
        == len(set().union(*interaction_cst))
        == n_features
    )
```