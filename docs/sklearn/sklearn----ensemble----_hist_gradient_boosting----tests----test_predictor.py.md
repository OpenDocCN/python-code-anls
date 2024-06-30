# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\test_predictor.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于编写和运行测试
from numpy.testing import assert_allclose  # 导入 NumPy 的测试工具 assert_allclose

from sklearn.datasets import make_regression  # 导入 make_regression 函数，用于生成回归数据集
from sklearn.ensemble._hist_gradient_boosting._bitset import (
    set_bitset_memoryview,
    set_raw_bitset_from_binned_bitset,
)  # 导入位集相关的函数

from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper  # 导入 _BinMapper 类，用于特征分箱映射
from sklearn.ensemble._hist_gradient_boosting.common import (
    ALMOST_INF,
    G_H_DTYPE,
    PREDICTOR_RECORD_DTYPE,
    X_BINNED_DTYPE,
    X_BITSET_INNER_DTYPE,
    X_DTYPE,
)  # 从 common 模块导入常量和数据类型定义

from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower  # 导入 TreeGrower 类，用于构建树模型
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor  # 导入 TreePredictor 类，用于树模型预测

from sklearn.metrics import r2_score  # 导入 r2_score 函数，用于计算 R^2 分数
from sklearn.model_selection import train_test_split  # 导入 train_test_split 函数，用于数据集划分

from sklearn.utils._openmp_helpers import _openmp_effective_n_threads  # 导入并行计算相关的函数

n_threads = _openmp_effective_n_threads()  # 获取有效的 OpenMP 线程数


@pytest.mark.parametrize("n_bins", [200, 256])
def test_regression_dataset(n_bins):
    X, y = make_regression(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )  # 生成回归数据集 X 和目标值 y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # 划分训练集和测试集

    mapper = _BinMapper(n_bins=n_bins, random_state=42)  # 初始化 _BinMapper 对象，用于特征分箱映射
    X_train_binned = mapper.fit_transform(X_train)  # 对训练集特征进行分箱映射

    # Init gradients and hessians to that of least squares loss
    gradients = -y_train.astype(G_H_DTYPE)  # 初始化梯度为最小二乘损失函数的负梯度
    hessians = np.ones(1, dtype=G_H_DTYPE)  # 初始化 Hessian 矩阵为全1

    min_samples_leaf = 10  # 叶子节点的最小样本数
    max_leaf_nodes = 30  # 树的最大叶子节点数
    grower = TreeGrower(
        X_train_binned,
        gradients,
        hessians,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        n_bins=n_bins,
        n_bins_non_missing=mapper.n_bins_non_missing_,
    )  # 初始化 TreeGrower 对象，用于训练树模型
    grower.grow()  # 训练树模型

    predictor = grower.make_predictor(binning_thresholds=mapper.bin_thresholds_)  # 创建 TreePredictor 对象

    known_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)  # 初始化已知类别位集
    f_idx_map = np.zeros(0, dtype=np.uint32)  # 初始化特征索引映射表

    y_pred_train = predictor.predict(X_train, known_cat_bitsets, f_idx_map, n_threads)  # 在训练集上进行预测
    assert r2_score(y_train, y_pred_train) > 0.82  # 断言预测结果的 R^2 分数大于 0.82

    y_pred_test = predictor.predict(X_test, known_cat_bitsets, f_idx_map, n_threads)  # 在测试集上进行预测
    assert r2_score(y_test, y_pred_test) > 0.67  # 断言预测结果的 R^2 分数大于 0.67


@pytest.mark.parametrize(
    "num_threshold, expected_predictions",
    [
        (-np.inf, [0, 1, 1, 1]),
        (10, [0, 0, 1, 1]),
        (20, [0, 0, 0, 1]),
        (ALMOST_INF, [0, 0, 0, 1]),
        (np.inf, [0, 0, 0, 0]),
    ],
)
def test_infinite_values_and_thresholds(num_threshold, expected_predictions):
    # 确保处理无穷大值和无穷大阈值的情况
    # 特别地，如果一个值是 +inf 并且阈值是 ALMOST_INF，样本应该进入右子节点。
    # 如果阈值是 inf（在 NaN 上分割），+inf 的样本将进入左子节点。

    X = np.array([-np.inf, 10, 20, np.inf]).reshape(-1, 1)  # 创建包含无穷大值的特征矩阵
    nodes = np.zeros(3, dtype=PREDICTOR_RECORD_DTYPE)  # 初始化节点数组，用于构建树结构

    # 我们只构建一个简单的树，包含一个根节点和两个子节点
    # 父节点
    # 将树的根节点的左子节点设为1
    nodes[0]["left"] = 1
    # 将树的根节点的右子节点设为2
    nodes[0]["right"] = 2
    # 将树的根节点的特征索引设为0
    nodes[0]["feature_idx"] = 0
    # 将树的根节点的数值阈值设为给定的阈值
    nodes[0]["num_threshold"] = num_threshold

    # 左子节点为叶节点
    nodes[1]["is_leaf"] = True
    # 左子节点的预测值设为0
    nodes[1]["value"] = 0

    # 右子节点为叶节点
    nodes[2]["is_leaf"] = True
    # 右子节点的预测值设为1
    nodes[2]["value"] = 1

    # 初始化一个空的二维数组，用于存储分箱后的分类特征位集合
    binned_cat_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    # 初始化一个空的二维数组，用于存储原始分类特征位集合
    raw_categorical_bitsets = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    # 初始化一个空的二维数组，用于存储已知分类特征位集合
    known_cat_bitset = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    # 初始化一个空的一维数组，用于存储特征索引映射关系
    f_idx_map = np.zeros(0, dtype=np.uint32)

    # 创建一个树预测器对象，用于进行预测
    predictor = TreePredictor(nodes, binned_cat_bitsets, raw_categorical_bitsets)
    # 使用预测器进行预测，返回预测结果
    predictions = predictor.predict(X, known_cat_bitset, f_idx_map, n_threads)

    # 断言所有预测结果与期望的预测结果相等
    assert np.all(predictions == expected_predictions)
# 使用 pytest.mark.parametrize 装饰器为 test_categorical_predictor 函数参数化测试用例
@pytest.mark.parametrize(
    "bins_go_left, expected_predictions",
    [
        ([0, 3, 4, 6], [1, 0, 0, 1, 1, 0]),  # 第一组测试参数和期望输出
        ([0, 1, 2, 6], [1, 1, 1, 0, 0, 0]),  # 第二组测试参数和期望输出
        ([3, 5, 6], [0, 0, 0, 1, 0, 1]),  # 第三组测试参数和期望输出
    ],
)
def test_categorical_predictor(bins_go_left, expected_predictions):
    # Test predictor outputs are correct with categorical features

    # 创建 X_binned 数组，包含一个单独的观测值，每个值是一个分箱的分类特征
    X_binned = np.array([[0, 1, 2, 3, 4, 5]], dtype=X_BINNED_DTYPE).T
    # 创建 categories 数组，包含分箱后的类别
    categories = np.array([2, 5, 6, 8, 10, 15], dtype=X_DTYPE)

    # 将 bins_go_left 转换为 X_BINNED_DTYPE 类型的 numpy 数组
    bins_go_left = np.array(bins_go_left, dtype=X_BINNED_DTYPE)

    # 构建一个简单的树结构，包含一个根节点和两个子节点
    # 父节点
    nodes = np.zeros(3, dtype=PREDICTOR_RECORD_DTYPE)
    nodes[0]["left"] = 1
    nodes[0]["right"] = 2
    nodes[0]["feature_idx"] = 0
    nodes[0]["is_categorical"] = True
    nodes[0]["missing_go_to_left"] = True

    # 左子节点
    nodes[1]["is_leaf"] = True
    nodes[1]["value"] = 1

    # 右子节点
    nodes[2]["is_leaf"] = True
    nodes[2]["value"] = 0

    # 初始化一个大小为 (1, 8) 的零矩阵，用于存储分箱后的分类特征的位集
    binned_cat_bitsets = np.zeros((1, 8), dtype=X_BITSET_INNER_DTYPE)
    raw_categorical_bitsets = np.zeros((1, 8), dtype=X_BITSET_INNER_DTYPE)

    # 对于每个 bins_go_left 中的值，调用 set_bitset_memoryview 函数设置位集
    for go_left in bins_go_left:
        set_bitset_memoryview(binned_cat_bitsets[0], go_left)

    # 根据分箱后的位集和 categories 创建原始分类特征的位集
    set_raw_bitset_from_binned_bitset(
        raw_categorical_bitsets[0], binned_cat_bitsets[0], categories
    )

    # 使用 TreePredictor 类构建预测器对象
    predictor = TreePredictor(nodes, binned_cat_bitsets, raw_categorical_bitsets)

    # 检查分箱数据是否能够正确预测输出
    prediction_binned = predictor.predict_binned(
        X_binned, missing_values_bin_idx=6, n_threads=n_threads
    )
    assert_allclose(prediction_binned, expected_predictions)

    # 手动构建位集
    known_cat_bitsets = np.zeros((1, 8), dtype=np.uint32)
    known_cat_bitsets[0, 0] = np.sum(2**categories, dtype=np.uint32)
    f_idx_map = np.array([0], dtype=np.uint32)

    # 检查未分箱数据的预测结果
    predictions = predictor.predict(
        categories.reshape(-1, 1), known_cat_bitsets, f_idx_map, n_threads
    )
    assert_allclose(predictions, expected_predictions)

    # 检查缺失值为 6 时是否会左转
    X_binned_missing = np.array([[6]], dtype=X_BINNED_DTYPE).T
    predictions = predictor.predict_binned(
        X_binned_missing, missing_values_bin_idx=6, n_threads=n_threads
    )
    assert_allclose(predictions, [1])

    # 当存在缺失值和未知值时均会左转
    predictions = predictor.predict(
        np.array([[np.nan, 17]], dtype=X_DTYPE).T,
        known_cat_bitsets,
        f_idx_map,
        n_threads,
    )
    assert_allclose(predictions, [1, 1])
```