# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_iforest.py`

```
"""
Testing for Isolation Forest algorithm (sklearn.ensemble.iforest).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入警告模块
import warnings
# 导入单元测试所需的模块
from unittest.mock import Mock, patch

# 导入科学计算库
import numpy as np
# 导入pytest用于参数化测试
import pytest

# 导入数据集加载函数和Isolation Forest算法
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.ensemble import IsolationForest
# 导入Isolation Forest的内部函数_average_path_length
from sklearn.ensemble._iforest import _average_path_length
# 导入评估指标ROC AUC的计算函数
from sklearn.metrics import roc_auc_score
# 导入数据集划分函数和参数网格生成函数
from sklearn.model_selection import ParameterGrid, train_test_split
# 导入随机状态检查函数
from sklearn.utils import check_random_state
# 导入用于单元测试的辅助函数
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
# 导入用于兼容性修复的函数和数据结构
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS

# 加载iris和diabetes数据集
iris = load_iris()
diabetes = load_diabetes()


def test_iforest(global_random_seed):
    """Check Isolation Forest for various parameter settings."""
    # 定义训练和测试数据集
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    # 定义参数网格
    grid = ParameterGrid(
        {"n_estimators": [3], "max_samples": [0.5, 1.0, 3], "bootstrap": [True, False]}
    )

    # 忽略警告进行测试
    with ignore_warnings():
        # 遍历参数网格
        for params in grid:
            # 使用当前参数训练Isolation Forest模型并预测
            IsolationForest(random_state=global_random_seed, **params).fit(
                X_train
            ).predict(X_test)


@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_iforest_sparse(global_random_seed, sparse_container):
    """Check IForest for various parameter settings on sparse input."""
    # 设置随机种子
    rng = check_random_state(global_random_seed)
    # 划分稀疏数据集的训练和测试集
    X_train, X_test = train_test_split(diabetes.data[:50], random_state=rng)
    # 定义参数网格
    grid = ParameterGrid({"max_samples": [0.5, 1.0], "bootstrap": [True, False]})

    # 将训练和测试集转换为稀疏格式
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)

    # 遍历参数网格
    for params in grid:
        # 训练和预测稀疏格式数据
        sparse_classifier = IsolationForest(
            n_estimators=10, random_state=global_random_seed, **params
        ).fit(X_train_sparse)
        sparse_results = sparse_classifier.predict(X_test_sparse)

        # 训练和预测密集格式数据
        dense_classifier = IsolationForest(
            n_estimators=10, random_state=global_random_seed, **params
        ).fit(X_train)
        dense_results = dense_classifier.predict(X_test)

        # 断言稀疏和密集预测结果一致
        assert_array_equal(sparse_results, dense_results)


def test_iforest_error():
    """Test that it gives proper exception on deficient input."""
    # 使用iris数据集
    X = iris.data

    # 数据集样本数小于256时，设置max_samples > n_samples应该产生警告
    warn_msg = "max_samples will be set to n_samples for estimation"
    with pytest.warns(UserWarning, match=warn_msg):
        # 测试Isolation Forest在不足输入时的行为
        IsolationForest(max_samples=1000).fit(X)
    # 使用 `warnings` 模块捕获 UserWarning 异常
    with warnings.catch_warnings():
        # 设置警告过滤器，将 UserWarning 转换为错误
        warnings.simplefilter("error", UserWarning)
        # 使用 IsolationForest 模型拟合数据 X
        IsolationForest(max_samples="auto").fit(X)

    # 使用 `warnings` 模块捕获 UserWarning 异常
    with warnings.catch_warnings():
        # 设置警告过滤器，将 UserWarning 转换为错误
        warnings.simplefilter("error", UserWarning)
        # 使用 IsolationForest 模型拟合数据 X，设置 max_samples 为 np.int64(2)
        IsolationForest(max_samples=np.int64(2)).fit(X)

    # 使用 `pytest` 断言捕获 ValueError 异常
    with pytest.raises(ValueError):
        # 使用 IsolationForest 模型拟合数据 X，并预测 X[:, 1:]，验证是否抛出 ValueError
        IsolationForest().fit(X).predict(X[:, 1:])
def test_recalculate_max_depth():
    """Check max_depth recalculation when max_samples is reset to n_samples"""
    # 使用 iris 数据集中的特征作为输入数据 X
    X = iris.data
    # 使用默认参数训练 IsolationForest 模型
    clf = IsolationForest().fit(X)
    # 遍历模型中的每棵树
    for est in clf.estimators_:
        # 断言每棵树的最大深度等于 ceil(log2(X.shape[0]))
        assert est.max_depth == int(np.ceil(np.log2(X.shape[0])))


def test_max_samples_attribute():
    # 使用 iris 数据集中的特征作为输入数据 X
    X = iris.data
    # 使用默认参数训练 IsolationForest 模型
    clf = IsolationForest().fit(X)
    # 断言模型的 max_samples_ 属性等于输入数据的样本数
    assert clf.max_samples_ == X.shape[0]

    # 使用 max_samples=500 参数重新训练模型
    clf = IsolationForest(max_samples=500)
    warn_msg = "max_samples will be set to n_samples for estimation"
    # 断言警告信息是否正确匹配
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X)
    # 再次断言模型的 max_samples_ 属性等于输入数据的样本数
    assert clf.max_samples_ == X.shape[0]

    # 使用 max_samples=0.4 参数训练模型
    clf = IsolationForest(max_samples=0.4).fit(X)
    # 断言模型的 max_samples_ 属性等于 0.4 乘以输入数据的样本数
    assert clf.max_samples_ == 0.4 * X.shape[0]


def test_iforest_parallel_regression(global_random_seed):
    """Check parallel regression."""
    # 使用全局随机种子初始化随机数生成器 rng
    rng = check_random_state(global_random_seed)

    # 将 diabetes 数据集拆分为训练集和测试集
    X_train, X_test = train_test_split(diabetes.data, random_state=rng)

    # 使用 n_jobs=3 参数训练 IsolationForest 模型
    ensemble = IsolationForest(n_jobs=3, random_state=global_random_seed).fit(X_train)

    # 设置 n_jobs=1 参数重新预测，并断言结果数组几乎相等
    ensemble.set_params(n_jobs=1)
    y1 = ensemble.predict(X_test)
    # 设置 n_jobs=2 参数重新预测，并断言结果数组几乎相等
    ensemble.set_params(n_jobs=2)
    y2 = ensemble.predict(X_test)
    # 断言两次预测结果数组几乎相等
    assert_array_almost_equal(y1, y2)

    # 使用 n_jobs=1 参数重新训练 IsolationForest 模型
    ensemble = IsolationForest(n_jobs=1, random_state=global_random_seed).fit(X_train)

    # 预测测试集数据，并断言结果数组几乎相等
    y3 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y3)


def test_iforest_performance(global_random_seed):
    """Test Isolation Forest performs well"""

    # 生成训练集和测试集数据
    rng = check_random_state(global_random_seed)
    X = 0.3 * rng.randn(600, 2)
    X = rng.permutation(np.vstack((X + 2, X - 2)))
    X_train = X[:1000]

    # 生成一些异常的新观测数据
    X_outliers = rng.uniform(low=-1, high=1, size=(200, 2))
    X_test = np.vstack((X[1000:], X_outliers))
    y_test = np.array([0] * 200 + [1] * 200)

    # 使用 max_samples=100 参数训练 IsolationForest 模型
    clf = IsolationForest(max_samples=100, random_state=rng).fit(X_train)

    # 预测异常程度得分，ROC AUC 分数应大于 0.98
    y_pred = -clf.decision_function(X_test)

    # 断言 ROC AUC 分数大于 0.98
    assert roc_auc_score(y_test, y_pred) > 0.98


@pytest.mark.parametrize("contamination", [0.25, "auto"])
def test_iforest_works(contamination, global_random_seed):
    # 创建一个玩具样本 X（最后两个样本是异常值）
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [7, 4], [-5, 9]]

    # 测试 IsolationForest 模型
    clf = IsolationForest(random_state=global_random_seed, contamination=contamination)
    clf.fit(X)
    decision_func = -clf.decision_function(X)
    pred = clf.predict(X)
    # 断言检测到的异常值：
    assert np.min(decision_func[-2:]) > np.max(decision_func[:-2])
    assert_array_equal(pred, 6 * [1] + 2 * [-1])


def test_max_samples_consistency():
    # 确保 iforest 和 BaseBagging 中验证的 max_samples 参数相同
    X = iris.data
    clf = IsolationForest().fit(X)
    # 断言模型的 max_samples_ 属性等于模型的 _max_samples 属性
    assert clf.max_samples_ == clf._max_samples
def test_iforest_subsampled_features():
    # 测试非回归性问题 #5732，在预测时失败。
    # 设置随机数生成器
    rng = check_random_state(0)
    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data[:50], diabetes.target[:50], random_state=rng
    )
    # 初始化孤立森林分类器，最大特征数设置为0.8
    clf = IsolationForest(max_features=0.8)
    # 在训练集上拟合分类器
    clf.fit(X_train, y_train)
    # 在测试集上进行预测
    clf.predict(X_test)


def test_iforest_average_path_length():
    # 测试非回归性问题 #8549，该问题使用了错误的平均路径长度公式，严格适用于整数情况
    # 当输入值 <= 2 时，更新以检查平均路径长度（问题 #11839）
    result_one = 2.0 * (np.log(4.0) + np.euler_gamma) - 2.0 * 4.0 / 5.0
    result_two = 2.0 * (np.log(998.0) + np.euler_gamma) - 2.0 * 998.0 / 999.0
    # 检查当输入为 [0], [1], [2] 时的平均路径长度是否接近预期值
    assert_allclose(_average_path_length([0]), [0.0])
    assert_allclose(_average_path_length([1]), [0.0])
    assert_allclose(_average_path_length([2]), [1.0])
    # 检查当输入为 [5], [999] 时的平均路径长度是否接近预期值 result_one 和 result_two
    assert_allclose(_average_path_length([5]), [result_one])
    assert_allclose(_average_path_length([999]), [result_two])
    # 检查当输入为 [1, 2, 5, 999] 的数组时，平均路径长度是否接近预期值
    assert_allclose(
        _average_path_length(np.array([1, 2, 5, 999])),
        [0.0, 1.0, result_one, result_two],
    )
    # 验证 _average_path_length 函数的单调递增性质
    avg_path_length = _average_path_length(np.arange(5))
    assert_array_equal(avg_path_length, np.sort(avg_path_length))


def test_score_samples():
    # 设置训练数据集
    X_train = [[1, 1], [1, 2], [2, 1]]
    # 使用孤立森林拟合两个分类器 clf1 和 clf2
    clf1 = IsolationForest(contamination=0.1).fit(X_train)
    clf2 = IsolationForest().fit(X_train)
    # 验证 clf1 和 clf2 对于 [[2.0, 2.0]] 的样本得分是否相等
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]),
        clf1.decision_function([[2.0, 2.0]]) + clf1.offset_,
    )
    assert_array_equal(
        clf2.score_samples([[2.0, 2.0]]),
        clf2.decision_function([[2.0, 2.0]]) + clf2.offset_,
    )
    # 再次验证 clf1 和 clf2 对于 [[2.0, 2.0]] 的样本得分是否相等
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]), clf2.score_samples([[2.0, 2.0]])
    )


def test_iforest_warm_start():
    """测试向 iForest 迭代添加 iTrees"""

    rng = check_random_state(0)
    X = rng.randn(20, 2)

    # 初次拟合前 10 棵树
    clf = IsolationForest(
        n_estimators=10, max_samples=20, random_state=rng, warm_start=True
    )
    clf.fit(X)
    # 记录第一棵树
    tree_1 = clf.estimators_[0]
    # 再次拟合另外 10 棵树
    clf.set_params(n_estimators=20)
    clf.fit(X)
    # 预期有 20 棵拟合好的树，且不覆盖之前的树
    assert len(clf.estimators_) == 20
    assert clf.estimators_[0] is tree_1


# 用 mock 的方式测试 get_chunk_n_rows，实际测试多个 chunk 的情况（这里每个 chunk 有 3 行）:
@patch(
    "sklearn.ensemble._iforest.get_chunk_n_rows",
    side_effect=Mock(**{"return_value": 3}),
)
@pytest.mark.parametrize("contamination, n_predict_calls", [(0.25, 3), ("auto", 2)])
def test_iforest_chunks_works1(
    mocked_get_chunk, contamination, n_predict_calls, global_random_seed
):
    # 测试孤立森林在多个 chunk 的工作情况，同时也测试了 contamination 和全局随机种子
    test_iforest_works(contamination, global_random_seed)
    assert mocked_get_chunk.call_count == n_predict_calls
# 使用 pytest 的 patch 装饰器模拟 sklearn.ensemble._iforest.get_chunk_n_rows 函数，使其在测试期间返回固定值 10
@patch(
    "sklearn.ensemble._iforest.get_chunk_n_rows",
    side_effect=Mock(**{"return_value": 10}),
)
# 使用 pytest 的 parametrize 装饰器定义多个参数化测试用例
@pytest.mark.parametrize("contamination, n_predict_calls", [(0.25, 3), ("auto", 2)])
# 定义测试函数 test_iforest_chunks_works2，接受参数 mocked_get_chunk, contamination, n_predict_calls, global_random_seed
def test_iforest_chunks_works2(
    mocked_get_chunk, contamination, n_predict_calls, global_random_seed
):
    # 调用 test_iforest_works 函数，验证 Isolation Forest 的功能
    test_iforest_works(contamination, global_random_seed)
    # 断言 mock 对象 mocked_get_chunk 被调用的次数符合预期的 n_predict_calls
    assert mocked_get_chunk.call_count == n_predict_calls


# 定义测试函数 test_iforest_with_uniform_data，验证 Isolation Forest 在均匀数据集上的预测行为
def test_iforest_with_uniform_data():
    """Test whether iforest predicts inliers when using uniform data"""

    # 创建一个全为1的二维数组 X
    X = np.ones((100, 10))
    # 初始化 Isolation Forest 模型
    iforest = IsolationForest()
    # 在 X 上拟合模型
    iforest.fit(X)

    # 创建随机数生成器 rng
    rng = np.random.RandomState(0)

    # 断言模型预测所有样本为正常值（1）
    assert all(iforest.predict(X) == 1)
    # 断言模型预测随机正态分布样本为正常值（1）
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    # 断言模型预测 X 加1的结果为正常值（1）
    assert all(iforest.predict(X + 1) == 1)
    # 断言模型预测 X 减1的结果为正常值（1）
    assert all(iforest.predict(X - 1) == 1)

    # 创建一个包含相同值的重复数组 X
    X = np.repeat(rng.randn(1, 10), 100, 0)
    # 初始化 Isolation Forest 模型
    iforest = IsolationForest()
    # 在 X 上拟合模型
    iforest.fit(X)

    # 断言模型预测所有样本为正常值（1）
    assert all(iforest.predict(X) == 1)
    # 断言模型预测随机正态分布样本为正常值（1）
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    # 断言模型预测全为1的数组为正常值（1）
    assert all(iforest.predict(np.ones((100, 10))) == 1)

    # 创建一个单行数组 X
    X = rng.randn(1, 10)
    # 初始化 Isolation Forest 模型
    iforest = IsolationForest()
    # 在 X 上拟合模型
    iforest.fit(X)

    # 断言模型预测单行数组 X 为正常值（1）
    assert all(iforest.predict(X) == 1)
    # 断言模型预测随机正态分布样本为正常值（1）
    assert all(iforest.predict(rng.randn(100, 10)) == 1)
    # 断言模型预测全为1的数组为正常值（1）
    assert all(iforest.predict(np.ones((100, 10))) == 1)


# 使用 parametrize 装饰器定义多个参数化测试用例，验证 Isolation Forest 在不同稀疏数据容器上的行为
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_iforest_with_n_jobs_does_not_segfault(csc_container):
    """Check that Isolation Forest does not segfault with n_jobs=2

    Non-regression test for #23252
    """
    # 创建具有 85000 个样本和 100 个特征的分类数据集 X
    X, _ = make_classification(n_samples=85_000, n_features=100, random_state=0)
    # 使用给定的稀疏容器 csc_container 包装 X
    X = csc_container(X)
    # 在 X 上拟合 Isolation Forest 模型，设置 n_jobs=2
    IsolationForest(n_estimators=10, max_samples=256, n_jobs=2).fit(X)


# 定义测试函数 test_iforest_preserve_feature_names，验证在 contamination 不为 "auto" 时，特征名是否得以保留
def test_iforest_preserve_feature_names():
    """Check that feature names are preserved when contamination is not "auto".

    Feature names are required for consistency checks during scoring.

    Non-regression test for Issue #25844
    """
    # 导入 pandas 库，如果导入失败则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建随机数生成器 rng
    rng = np.random.RandomState(0)

    # 创建一个包含随机正态分布值的 DataFrame X，有一个名为 "a" 的特征列
    X = pd.DataFrame(data=rng.randn(4), columns=["a"])
    # 初始化 Isolation Forest 模型，设置 random_state=0 和 contamination=0.05
    model = IsolationForest(random_state=0, contamination=0.05)

    # 使用警告过滤器捕获 UserWarning 类型的警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # 在 X 上拟合模型
        model.fit(X)


# 使用 parametrize 装饰器定义多个参数化测试用例，验证 Isolation Forest 在稀疏矩阵输入和浮点数污染值下的行为
@pytest.mark.parametrize("sparse_container", CSC_CONTAINERS + CSR_CONTAINERS)
def test_iforest_sparse_input_float_contamination(sparse_container):
    """Check that `IsolationForest` accepts sparse matrix input and float value for
    contamination.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27626
    """
    # 创建一个包含 50 个样本和 4 个特征的分类数据集 X
    X, _ = make_classification(n_samples=50, n_features=4, random_state=0)
    # 使用给定的稀疏容器 sparse_container 包装 X
    X = sparse_container(X)
    # 对 X 进行排序
    X.sort_indices()
    # 设置 n_estimators=5, contamination=0.1 和 random_state=0，拟合 Isolation Forest 模型
    iforest = IsolationForest(
        n_estimators=5, contamination=0.1, random_state=0
    ).fit(X)
    # 使用 Isolation Forest 模型计算输入数据集 X 的异常分数（decision function）
    X_decision = iforest.decision_function(X)
    # 断言：计算异常分数小于 0 的样本数占总样本数的比例，应该接近预期的异常比例 contamination
    assert (X_decision < 0).sum() / X.shape[0] == pytest.approx(contamination)
```