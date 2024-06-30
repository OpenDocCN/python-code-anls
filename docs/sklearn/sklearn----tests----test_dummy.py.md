# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_dummy.py`

```
# 导入必要的库
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例
import scipy.sparse as sp  # 导入SciPy稀疏矩阵库

# 导入机器学习相关模块
from sklearn.base import clone  # 导入clone函数，用于复制估计器
from sklearn.dummy import DummyClassifier, DummyRegressor  # 导入Dummy分类器和回归器
from sklearn.exceptions import NotFittedError  # 导入NotFittedError异常类
from sklearn.utils._testing import (  # 导入测试相关函数和装饰器
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)
from sklearn.utils.fixes import CSC_CONTAINERS  # 导入用于修复的常量
from sklearn.utils.stats import _weighted_percentile  # 导入加权百分位数计算函数


@ignore_warnings  # 忽略警告装饰器
def _check_predict_proba(clf, X, y):
    # 预测类别概率
    proba = clf.predict_proba(X)
    # 计算对数类别概率，可能会出现除以零的情况
    log_proba = clf.predict_log_proba(X)

    # 将y转换为至少一维数组
    y = np.atleast_1d(y)
    if y.ndim == 1:
        y = np.reshape(y, (-1, 1))

    # 获取输出维度和样本数
    n_outputs = y.shape[1]
    n_samples = len(X)

    if n_outputs == 1:
        proba = [proba]
        log_proba = [log_proba]

    # 检查每个输出维度的预测结果
    for k in range(n_outputs):
        assert proba[k].shape[0] == n_samples
        assert proba[k].shape[1] == len(np.unique(y[:, k]))
        assert_array_almost_equal(proba[k].sum(axis=1), np.ones(len(X)))
        # 计算对数概率应与预测对数概率接近，可能会出现除以零的情况
        assert_array_almost_equal(np.log(proba[k]), log_proba[k])


def _check_behavior_2d(clf):
    # 1维情况
    X = np.array([[0], [0], [0], [0]])  # 被忽略的数据
    y = np.array([1, 2, 1, 1])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape

    # 2维情况
    y = np.array([[1, 0], [2, 0], [1, 0], [1, 3]])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape


def _check_behavior_2d_for_constant(clf):
    # 仅适用于2维情况
    X = np.array([[0], [0], [0], [0]])  # 被忽略的数据
    y = np.array([[1, 0, 5, 4, 3], [2, 0, 1, 2, 5], [1, 0, 4, 5, 2], [1, 3, 3, 2, 0]])
    est = clone(clf)
    est.fit(X, y)
    y_pred = est.predict(X)
    assert y.shape == y_pred.shape


def _check_equality_regressor(statistic, y_learn, y_pred_learn, y_test, y_pred_test):
    # 断言回归器预测结果与统计量近似相等
    assert_array_almost_equal(np.tile(statistic, (y_learn.shape[0], 1)), y_pred_learn)
    assert_array_almost_equal(np.tile(statistic, (y_test.shape[0], 1)), y_pred_test)


def test_feature_names_in_and_n_features_in_(global_random_seed, n_samples=10):
    # 导入pytest并跳过如果不可用
    pd = pytest.importorskip("pandas")

    # 设置随机数种子
    random_state = np.random.RandomState(seed=global_random_seed)

    # 创建一个DataFrame对象X和一个随机数组y
    X = pd.DataFrame([[0]] * n_samples, columns=["feature_1"])
    y = random_state.rand(n_samples)

    # 使用DummyRegressor拟合数据并检查特性
    est = DummyRegressor().fit(X, y)
    assert hasattr(est, "feature_names_in_")
    assert hasattr(est, "n_features_in_")

    # 使用DummyClassifier拟合数据并检查特性
    est = DummyClassifier().fit(X, y)
    assert hasattr(est, "feature_names_in_")
    assert hasattr(est, "n_features_in_")


def test_most_frequent_and_prior_strategy():
    X = [[0], [0], [0], [0]]  # 被忽略的数据
    y = [1, 2, 1, 1]
    # 遍历两种策略："most_frequent" 和 "prior"
    for strategy in ("most_frequent", "prior"):
        # 使用 DummyClassifier 创建分类器对象，设置策略和随机种子
        clf = DummyClassifier(strategy=strategy, random_state=0)
        # 使用训练数据 X 和标签 y 进行分类器训练
        clf.fit(X, y)
        # 断言预测结果与全为 1 的数组相等
        assert_array_equal(clf.predict(X), np.ones(len(X)))
        # 检查分类器的预测概率是否符合预期
        _check_predict_proba(clf, X, y)
    
        # 如果策略是 "prior"
        if strategy == "prior":
            # 断言预测 X[0] 的概率与分类器的类先验概率相近
            assert_array_almost_equal(
                clf.predict_proba([X[0]]), clf.class_prior_.reshape((1, -1))
            )
        else:  # 对于 "most_frequent" 策略
            # 断言预测 X[0] 的概率是否大于类先验概率的 0.5
            assert_array_almost_equal(
                clf.predict_proba([X[0]]), clf.class_prior_.reshape((1, -1)) > 0.5
            )
# 定义测试函数，用于验证在二维列向量 y 下使用最频繁和先验策略的 DummyClassifier 的表现
def test_most_frequent_and_prior_strategy_with_2d_column_y():
    # 在 https://github.com/scikit-learn/scikit-learn/pull/13545 中添加的非回归测试
    X = [[0], [0], [0], [0]]  # 输入特征 X，包含四个样本
    y_1d = [1, 2, 1, 1]  # 一维标签 y
    y_2d = [[1], [2], [1], [1]]  # 二维标签 y

    # 遍历两种策略："most_frequent" 和 "prior"
    for strategy in ("most_frequent", "prior"):
        # 创建 DummyClassifier 对象，分别基于一维和二维标签 y
        clf_1d = DummyClassifier(strategy=strategy, random_state=0)
        clf_2d = DummyClassifier(strategy=strategy, random_state=0)

        # 使用 X 和对应的标签训练模型
        clf_1d.fit(X, y_1d)
        clf_2d.fit(X, y_2d)

        # 验证两个模型在预测相同输入 X 上的结果是否相等
        assert_array_equal(clf_1d.predict(X), clf_2d.predict(X))


# 定义测试函数，验证在多输出情况下使用最频繁和先验策略的 DummyClassifier 的表现
def test_most_frequent_and_prior_strategy_multioutput():
    X = [[0], [0], [0], [0]]  # 输入特征 X，包含四个样本，实际上被忽略
    y = np.array([[1, 0], [2, 0], [1, 0], [1, 3]])  # 多输出标签 y

    n_samples = len(X)  # 样本数

    # 遍历两种策略："prior" 和 "most_frequent"
    for strategy in ("prior", "most_frequent"):
        # 创建 DummyClassifier 对象，使用指定策略
        clf = DummyClassifier(strategy=strategy, random_state=0)
        clf.fit(X, y)  # 使用 X 和多输出标签 y 训练模型

        # 验证模型在相同输入 X 上的预测结果是否符合预期
        assert_array_equal(
            clf.predict(X),
            np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))]),
        )

        # 检查模型预测概率的正确性
        _check_predict_proba(clf, X, y)

        # 检查模型在二维输出数据上的行为
        _check_behavior_2d(clf)


# 定义测试函数，验证使用分层策略的 DummyClassifier 的表现
def test_stratified_strategy(global_random_seed):
    X = [[0]] * 5  # 输入特征 X，包含五个样本，实际上被忽略
    y = [1, 2, 1, 1, 2]  # 标签 y

    # 创建 DummyClassifier 对象，使用分层策略和全局随机种子
    clf = DummyClassifier(strategy="stratified", random_state=global_random_seed)
    clf.fit(X, y)  # 使用 X 和标签 y 训练模型

    # 创建包含 500 个样本的输入特征 X
    X = [[0]] * 500
    y_pred = clf.predict(X)  # 预测结果

    # 计算预测结果中各类别的比例并验证其准确性
    p = np.bincount(y_pred) / float(len(X))
    assert_almost_equal(p[1], 3.0 / 5, decimal=1)
    assert_almost_equal(p[2], 2.0 / 5, decimal=1)

    # 检查模型预测概率的正确性
    _check_predict_proba(clf, X, y)


# 定义测试函数，验证在多输出情况下使用分层策略的 DummyClassifier 的表现
def test_stratified_strategy_multioutput(global_random_seed):
    X = [[0]] * 5  # 输入特征 X，包含五个样本，实际上被忽略
    y = np.array([[2, 1], [2, 2], [1, 1], [1, 2], [1, 1]])  # 多输出标签 y

    # 创建 DummyClassifier 对象，使用分层策略和全局随机种子
    clf = DummyClassifier(strategy="stratified", random_state=global_random_seed)
    clf.fit(X, y)  # 使用 X 和多输出标签 y 训练模型

    # 创建包含 500 个样本的输入特征 X
    X = [[0]] * 500
    y_pred = clf.predict(X)  # 预测结果

    # 遍历多输出标签的每个维度
    for k in range(y.shape[1]):
        # 计算预测结果中各类别的比例并验证其准确性
        p = np.bincount(y_pred[:, k]) / float(len(X))
        assert_almost_equal(p[1], 3.0 / 5, decimal=1)
        assert_almost_equal(p[2], 2.0 / 5, decimal=1)

        # 检查模型预测概率的正确性
        _check_predict_proba(clf, X, y)

    # 检查模型在二维输出数据上的行为
    _check_behavior_2d(clf)


# 定义测试函数，验证使用均匀策略的 DummyClassifier 的表现
def test_uniform_strategy(global_random_seed):
    X = [[0]] * 4  # 输入特征 X，包含四个样本，实际上被忽略
    y = [1, 2, 1, 1]  # 标签 y

    # 创建 DummyClassifier 对象，使用均匀策略和全局随机种子
    clf = DummyClassifier(strategy="uniform", random_state=global_random_seed)
    clf.fit(X, y)  # 使用 X 和标签 y 训练模型

    # 创建包含 500 个样本的输入特征 X
    X = [[0]] * 500
    y_pred = clf.predict(X)  # 预测结果

    # 计算预测结果中各类别的比例并验证其准确性
    p = np.bincount(y_pred) / float(len(X))
    assert_almost_equal(p[1], 0.5, decimal=1)
    assert_almost_equal(p[2], 0.5, decimal=1)

    # 检查模型预测概率的正确性
    _check_predict_proba(clf, X, y)


# 定义测试函数，验证在多输出情况下使用均匀策略的 DummyClassifier 的表现
def test_uniform_strategy_multioutput(global_random_seed):
    X = [[0]] * 4  # 输入特征 X，包含四个样本，实际上被忽略
    y = np.array([[2, 1], [2, 2], [1, 2], [1, 1]])  # 多输出标签 y

    # 创建 DummyClassifier 对象，使用均匀策略和全局随机种子
    clf = DummyClassifier(strategy="uniform", random_state=global_random_seed)
    clf.fit(X, y)  # 使用 X 和多输出标签 y 训练模型

    # 创建包含 500 个样本的输入特征 X
    X = [[0]] * 500
    y_pred = clf.predict(X)  # 预测结果
    # 对于每一列的预测结果进行处理
    for k in range(y.shape[1]):
        # 计算预测值 y_pred 中第 k 列中每个类别的频率分布，并将其归一化为浮点数
        p = np.bincount(y_pred[:, k]) / float(len(X))
        # 断言预测结果中类别为 1 的频率接近 0.5，精确度为小数点后一位
        assert_almost_equal(p[1], 0.5, decimal=1)
        # 断言预测结果中类别为 2 的频率接近 0.5，精确度为小数点后一位
        assert_almost_equal(p[2], 0.5, decimal=1)
        # 检查分类器的预测概率是否符合预期
        _check_predict_proba(clf, X, y)
    
    # 检查分类器的二维行为是否符合预期
    _check_behavior_2d(clf)
# 测试使用字符串标签进行分类
def test_string_labels():
    # 创建包含单个特征值的列表，复制5次，形成二维列表 X
    X = [[0]] * 5
    # 创建目标标签 y，包含城市名字符串
    y = ["paris", "paris", "tokyo", "amsterdam", "berlin"]
    # 创建一个使用"most_frequent"策略的DummyClassifier分类器
    clf = DummyClassifier(strategy="most_frequent")
    # 使用 X 和 y 进行分类器的拟合
    clf.fit(X, y)
    # 断言预测结果与 ["paris"] * 5 相等
    assert_array_equal(clf.predict(X), ["paris"] * 5)


# 使用不同参数进行参数化测试
@pytest.mark.parametrize(
    "y,y_test",
    [
        # 第一组参数: y=[2, 1, 1, 1], y_test=[2, 2, 1, 1]
        ([2, 1, 1, 1], [2, 2, 1, 1]),
        # 第二组参数: y=[[2, 2], [1, 1], [1, 1], [1, 1]], y_test=[[2, 2], [2, 2], [1, 1], [1, 1]]
        (
            np.array([[2, 2], [1, 1], [1, 1], [1, 1]]),
            np.array([[2, 2], [2, 2], [1, 1], [1, 1]]),
        ),
    ],
)
# 测试带有 None 输入的分类器得分
def test_classifier_score_with_None(y, y_test):
    # 创建一个使用"most_frequent"策略的DummyClassifier分类器
    clf = DummyClassifier(strategy="most_frequent")
    # 使用 None 和 y 进行分类器的拟合
    clf.fit(None, y)
    # 断言使用 None 和 y_test 作为输入的分类器得分为 0.5
    assert clf.score(None, y_test) == 0.5


# 使用不同策略参数进行参数化测试
@pytest.mark.parametrize(
    "strategy", ["stratified", "most_frequent", "prior", "uniform", "constant"]
)
# 测试分类器预测与输入 X 无关
def test_classifier_prediction_independent_of_X(strategy, global_random_seed):
    # 创建目标标签 y
    y = [0, 2, 1, 1]
    # 创建包含单个特征值的列表 X1，复制4次
    X1 = [[0]] * 4
    # 创建一个使用指定策略和全局随机种子的DummyClassifier分类器
    clf1 = DummyClassifier(
        strategy=strategy, random_state=global_random_seed, constant=0
    )
    # 使用 X1 和 y 进行分类器的拟合
    clf1.fit(X1, y)
    # 获取对 X1 的预测结果
    predictions1 = clf1.predict(X1)

    # 创建包含单个特征值的列表 X2，复制4次
    X2 = [[1]] * 4
    # 创建另一个使用相同策略和全局随机种子的DummyClassifier分类器
    clf2 = DummyClassifier(
        strategy=strategy, random_state=global_random_seed, constant=0
    )
    # 使用 X2 和 y 进行分类器的拟合
    clf2.fit(X2, y)
    # 获取对 X2 的预测结果
    predictions2 = clf2.predict(X2)

    # 断言两次预测结果相等
    assert_array_equal(predictions1, predictions2)


# 测试使用"mean"策略的DummyRegressor回归器
def test_mean_strategy_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态对象
    random_state = np.random.RandomState(seed=global_random_seed)

    # 创建包含单个特征值的列表 X，复制4次
    X = [[0]] * 4  # 被忽略
    # 生成随机正态分布的目标值 y
    y = random_state.randn(4)

    # 创建一个使用"mean"策略的DummyRegressor回归器
    reg = DummyRegressor()
    # 使用 X 和 y 进行回归器的拟合
    reg.fit(X, y)
    # 断言预测结果与 y 均值组成的列表相等
    assert_array_equal(reg.predict(X), [np.mean(y)] * len(X))


# 测试使用"median"策略的DummyRegressor回归器
def test_median_strategy_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态对象
    random_state = np.random.RandomState(seed=global_random_seed)

    # 创建包含单个特征值的列表 X，复制5次
    X = [[0]] * 5  # 被忽略
    # 生成随机正态分布的目标值 y
    y = random_state.randn(5)

    # 创建一个使用"median"策略的DummyRegressor回归器
    reg = DummyRegressor(strategy="median")
    # 使用 X 和 y 进行回归器的拟合
    reg.fit(X, y)
    # 断言预测结果与 y 中位数组成的列表相等
    assert_array_equal(reg.predict(X), [np.median(y)] * len(X))


# 测试DummyRegressor在未拟合状态下是否抛出NotFittedError异常
def test_regressor_exceptions():
    # 创建一个未拟合的DummyRegressor回归器
    reg = DummyRegressor()
    # 断言调用未拟合回归器的predict方法会抛出NotFittedError异常
    with pytest.raises(NotFittedError):
        reg.predict([])


# 测试使用"mean"策略的多输出DummyRegressor回归器
def test_mean_strategy_multioutput_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态对象
    random_state = np.random.RandomState(seed=global_random_seed)

    # 创建随机正态分布的输入特征和目标值
    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    # 计算 y_learn 每列的均值，形成一个行向量
    mean = np.mean(y_learn, axis=0).reshape((1, -1))

    # 创建随机正态分布的测试输入特征和目标值
    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # 创建一个DummyRegressor回归器
    est = DummyRegressor()
    # 使用 X_learn 和 y_learn 进行回归器的拟合
    est.fit(X_learn, y_learn)
    # 获取对 X_learn 和 X_test 的预测结果
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    # 检查预测结果与期望值的一致性
    _check_equality_regressor(mean, y_learn, y_pred_learn, y_test, y_pred_test)
    _check_behavior_2d(est)


# 测试使用"median"策略的多输出DummyRegressor回归器
def test_median_strategy_multioutput_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态对象
    random_state = np.random.RandomState(seed=global_random_seed)

    # 创建随机正态分布的输入特征和目标值
    X_learn = random_state.randn(10, 10)
    y_learn = random_state.randn(10, 5)

    # 计算 y_learn 每列的中位数，形成一个行向量
    median = np.median(y_learn, axis=0).reshape((1, -1))

    # 创建随机正态分布的测试输入特征和目标值
    X_test = random_state.randn(20, 10)
    y_test = random_state.randn(20, 5)

    # 创建一个使用"median"策略的DummyRegressor回归器
    est = DummyRegressor(strategy="median")
    # 使用 X_learn 和 y_learn 进行回归器的拟合
    est.fit(X_learn, y_learn)
    # 获取对 X_learn 和 X_test 的预测结果
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    # 检查预测
    # 创建一个虚拟的回归器对象，使用中位数策略进行拟合
    est = DummyRegressor(strategy="median")
    # 使用学习数据集 X_learn 和 y_learn 来拟合回归器
    est.fit(X_learn, y_learn)
    # 对学习数据集 X_learn 进行预测，得到预测结果 y_pred_learn
    y_pred_learn = est.predict(X_learn)
    # 对测试数据集 X_test 进行预测，得到预测结果 y_pred_test
    y_pred_test = est.predict(X_test)

    # 检查回归器预测的结果与中位数的一致性，比较预测结果 y_pred_learn 和 y_pred_test
    _check_equality_regressor(median, y_learn, y_pred_learn, y_test, y_pred_test)
    # 检查回归器在二维数据上的行为，传入回归器对象 est 进行检查
    _check_behavior_2d(est)
# 测试量化策略的回归器函数，使用全局随机种子来初始化随机状态
def test_quantile_strategy_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态对象
    random_state = np.random.RandomState(seed=global_random_seed)

    # 创建一个包含5个相同元素的列表作为特征向量 X，但在该测试中被忽略
    X = [[0]] * 5  # ignored
    # 生成一个服从标准正态分布的长度为5的随机向量作为目标值 y
    y = random_state.randn(5)

    # 创建一个使用中位数作为预测值的虚拟回归器
    reg = DummyRegressor(strategy="quantile", quantile=0.5)
    reg.fit(X, y)
    # 断言预测结果与 y 中位数的数组相等
    assert_array_equal(reg.predict(X), [np.median(y)] * len(X))

    # 创建一个使用最小值作为预测值的虚拟回归器
    reg = DummyRegressor(strategy="quantile", quantile=0)
    reg.fit(X, y)
    # 断言预测结果与 y 最小值的数组相等
    assert_array_equal(reg.predict(X), [np.min(y)] * len(X))

    # 创建一个使用最大值作为预测值的虚拟回归器
    reg = DummyRegressor(strategy="quantile", quantile=1)
    reg.fit(X, y)
    # 断言预测结果与 y 最大值的数组相等
    assert_array_equal(reg.predict(X), [np.max(y)] * len(X))

    # 创建一个使用百分位数(30%)作为预测值的虚拟回归器
    reg = DummyRegressor(strategy="quantile", quantile=0.3)
    reg.fit(X, y)
    # 断言预测结果与 y 第30百分位数的数组相等
    assert_array_equal(reg.predict(X), [np.percentile(y, q=30)] * len(X))


# 测试多输出情况下的量化策略回归器函数，使用全局随机种子初始化随机状态
def test_quantile_strategy_multioutput_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态对象
    random_state = np.random.RandomState(seed=global_random_seed)

    # 生成一个形状为(10, 10)的随机特征矩阵 X_learn
    X_learn = random_state.randn(10, 10)
    # 生成一个形状为(10, 5)的随机目标矩阵 y_learn
    y_learn = random_state.randn(10, 5)

    # 计算 y_learn 每列的中位数，并将结果重塑为形状为(1, 5)的矩阵 median
    median = np.median(y_learn, axis=0).reshape((1, -1))
    # 计算 y_learn 每列的80%分位数，并将结果重塑为形状为(1, 5)的矩阵 quantile_values
    quantile_values = np.percentile(y_learn, axis=0, q=80).reshape((1, -1))

    # 生成一个形状为(20, 10)的随机特征矩阵 X_test
    X_test = random_state.randn(20, 10)
    # 生成一个形状为(20, 5)的随机目标矩阵 y_test
    y_test = random_state.randn(20, 5)

    # 创建一个使用中位数作为预测值的虚拟回归器
    est = DummyRegressor(strategy="quantile", quantile=0.5)
    est.fit(X_learn, y_learn)
    # 计算训练集和测试集的预测值
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    # 检查预测值和实际值的一致性
    _check_equality_regressor(median, y_learn, y_pred_learn, y_test, y_pred_test)
    # 检查回归器的二维行为
    _check_behavior_2d(est)

    # 创建一个使用80%分位数作为预测值的虚拟回归器
    est = DummyRegressor(strategy="quantile", quantile=0.8)
    est.fit(X_learn, y_learn)
    # 计算训练集和测试集的预测值
    y_pred_learn = est.predict(X_learn)
    y_pred_test = est.predict(X_test)

    # 检查预测值和实际值的一致性
    _check_equality_regressor(
        quantile_values, y_learn, y_pred_learn, y_test, y_pred_test
    )
    # 检查回归器的二维行为
    _check_behavior_2d(est)


# 测试无效量化策略的函数
def test_quantile_invalid():
    # 创建一个包含5个相同元素的列表作为特征向量 X，但在该测试中被忽略
    X = [[0]] * 5  # ignored
    # 创建一个包含5个相同元素的列表作为目标值 y，但在该测试中被忽略
    y = [0] * 5  # ignored

    # 创建一个使用空量化参数的虚拟回归器
    est = DummyRegressor(strategy="quantile", quantile=None)
    # 定义一个错误消息，说明在使用策略'quantile'时必须指定所需的量化值
    err_msg = (
        "When using `strategy='quantile', you have to specify the desired quantile"
    )
    # 断言调用此回归器时会抛出值错误，并匹配预期的错误消息
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


# 测试在空训练集情况下的量化策略回归器函数
def test_quantile_strategy_empty_train():
    # 创建一个使用量化值为0.4的虚拟回归器
    est = DummyRegressor(strategy="quantile", quantile=0.4)
    # 断言在空训练集情况下会抛出索引错误
    with pytest.raises(IndexError):
        est.fit([], [])


# 测试常数策略的回归器函数，使用全局随机种子来初始化随机状态
def test_constant_strategy_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态对象
    random_state = np.random.RandomState(seed=global_random_seed)

    # 创建一个包含5个相同元素的列表作为特征向量 X，但在该测试中被忽略
    X = [[0]] * 5  # ignored
    # 生成一个服从标准正态分布的长度为5的随机向量作为目标值 y
    y = random_state.randn(5)

    # 创建一个使用常数43作为预测值的虚拟回归器
    reg = DummyRegressor(strategy="constant", constant=[43])
    reg.fit(X, y)
    # 断言预测结果与常数43的数组相等
    assert_array_equal(reg.predict(X), [43] * len(X))

    # 创建一个使用常数43作为预测值的虚拟回归器（常数作为单个值输入）
    reg = DummyRegressor(strategy="constant", constant=43)
    reg.fit(X, y)
    # 断言预测结果与常数43的数组相等
    assert_array_equal(reg.predict(X), [43] * len(X))

    # 对于＃22478的非回归测试
    # 断言常数不是一个NumPy数组的实例
    assert not isinstance(reg.constant, np.ndarray)


# 测试多输出情况下的常数策略回归器函数，使用全局随机种子初始化随机状态
def test_constant_strategy_multioutput_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态对象
    random_state = np.random.RandomState(seed=global_random_seed)
    # 使用随机状态生成一个 10x10 的二维数组作为学习特征矩阵
    X_learn = random_state.randn(10, 10)
    # 使用随机状态生成一个 10x5 的二维数组作为学习目标值
    y_learn = random_state.randn(10, 5)

    # 使用随机状态生成一个包含5个随机常数的一维数组，用于测试
    constants = random_state.randn(5)

    # 使用随机状态生成一个 20x10 的二维数组作为测试特征矩阵
    X_test = random_state.randn(20, 10)
    # 使用随机状态生成一个 20x5 的二维数组作为测试目标值
    y_test = random_state.randn(20, 5)

    # 创建一个 DummyRegressor 模型，使用常数策略，并设置常数值为 constants
    est = DummyRegressor(strategy="constant", constant=constants)
    # 使用学习数据集 (X_learn, y_learn) 训练模型
    est.fit(X_learn, y_learn)
    # 对学习数据集进行预测
    y_pred_learn = est.predict(X_learn)
    # 对测试数据集进行预测
    y_pred_test = est.predict(X_test)

    # 检查回归器的预测结果是否与给定的常数值 constants 相符，以及预测结果的一致性
    _check_equality_regressor(constants, y_learn, y_pred_learn, y_test, y_pred_test)
    # 检查 DummyRegressor 在处理二维数据时的行为
    _check_behavior_2d_for_constant(est)
# 测试 DummyRegressor 类的 'mean' 策略，即用平均值作为预测常数
def test_y_mean_attribute_regressor():
    # 创建一个包含 5 个相同数据点的特征矩阵
    X = [[0]] * 5
    # 创建目标向量 y
    y = [1, 2, 4, 6, 8]
    # 使用 DummyRegressor 模型，指定策略为 'mean'
    est = DummyRegressor(strategy="mean")
    # 拟合模型
    est.fit(X, y)

    # 断言拟合后的常数属性等于目标向量 y 的平均值
    assert est.constant_ == np.mean(y)


# 测试 DummyRegressor 类的 'constant' 策略，即常数目标值未指定时的异常处理
def test_constants_not_specified_regressor():
    # 创建一个包含 5 个相同数据点的特征矩阵
    X = [[0]] * 5
    # 创建目标向量 y
    y = [1, 2, 4, 6, 8]

    # 使用 DummyRegressor 模型，指定策略为 'constant'，但未指定常数值
    est = DummyRegressor(strategy="constant")
    # 定义预期的错误信息
    err_msg = "Constant target value has to be specified"
    # 使用 pytest 断言捕获 ValueError，并匹配错误信息
    with pytest.raises(TypeError, match=err_msg):
        est.fit(X, y)


# 测试 DummyRegressor 类的 'constant' 策略，多输出时的异常处理
def test_constant_size_multioutput_regressor(global_random_seed):
    # 使用全局随机种子创建随机状态生成器
    random_state = np.random.RandomState(seed=global_random_seed)
    # 创建随机特征矩阵 X（大小为 10x10）
    X = random_state.randn(10, 10)
    # 创建随机目标矩阵 y（大小为 10x5）
    y = random_state.randn(10, 5)

    # 使用 DummyRegressor 模型，指定策略为 'constant'，并指定常数值为 [1, 2, 3, 4]
    est = DummyRegressor(strategy="constant", constant=[1, 2, 3, 4])
    # 定义预期的错误信息
    err_msg = r"Constant target value should have shape \(5, 1\)."
    # 使用 pytest 断言捕获 ValueError，并匹配错误信息
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


# 测试 DummyClassifier 类的 'constant' 策略，单输出情况
def test_constant_strategy():
    # 创建一个特征矩阵 X，其中的数据点在本测试中被忽略
    X = [[0], [0], [0], [0]]  # ignored
    # 创建单输出目标向量 y
    y = [2, 1, 2, 2]

    # 使用 DummyClassifier 模型，指定策略为 'constant'，随机状态为 0，常数值为 1
    clf = DummyClassifier(strategy="constant", random_state=0, constant=1)
    # 拟合模型
    clf.fit(X, y)
    # 断言预测结果与 np.ones(len(X)) 相等
    assert_array_equal(clf.predict(X), np.ones(len(X)))
    # 调用辅助函数检查预测概率的正确性
    _check_predict_proba(clf, X, y)

    # 创建一个特征矩阵 X，其中的数据点在本测试中被忽略
    X = [[0], [0], [0], [0]]  # ignored
    # 创建单输出目标向量 y
    y = ["two", "one", "two", "two"]
    # 使用 DummyClassifier 模型，指定策略为 'constant'，随机状态为 0，常数值为 "one"
    clf = DummyClassifier(strategy="constant", random_state=0, constant="one")
    # 拟合模型
    clf.fit(X, y)
    # 断言预测结果与 np.array(["one"] * 4) 相等
    assert_array_equal(clf.predict(X), np.array(["one"] * 4))
    # 调用辅助函数检查预测概率的正确性
    _check_predict_proba(clf, X, y)


# 测试 DummyClassifier 类的 'constant' 策略，多输出情况
def test_constant_strategy_multioutput():
    # 创建一个特征矩阵 X，其中的数据点在本测试中被忽略
    X = [[0], [0], [0], [0]]  # ignored
    # 创建多输出目标矩阵 y
    y = np.array([[2, 3], [1, 3], [2, 3], [2, 0]])

    # 计算样本数
    n_samples = len(X)

    # 使用 DummyClassifier 模型，指定策略为 'constant'，随机状态为 0，常数值为 [1, 0]
    clf = DummyClassifier(strategy="constant", random_state=0, constant=[1, 0])
    # 拟合模型
    clf.fit(X, y)
    # 断言预测结果与 np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))]) 相等
    assert_array_equal(
        clf.predict(X), np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    )
    # 调用辅助函数检查预测概率的正确性
    _check_predict_proba(clf, X, y)


# 使用 pytest.mark.parametrize 进行参数化测试，测试 DummyClassifier 类的 'constant' 策略异常情况
@pytest.mark.parametrize(
    "y, params, err_msg",
    [
        ([2, 1, 2, 2], {"random_state": 0}, "Constant.*has to be specified"),
        ([2, 1, 2, 2], {"constant": [2, 0]}, "Constant.*should have shape"),
        (
            np.transpose([[2, 1, 2, 2], [2, 1, 2, 2]]),
            {"constant": 2},
            "Constant.*should have shape",
        ),
        (
            [2, 1, 2, 2],
            {"constant": "my-constant"},
            "constant=my-constant.*Possible values.*\\[1, 2]",
        ),
        (
            np.transpose([[2, 1, 2, 2], [2, 1, 2, 2]]),
            {"constant": [2, "unknown"]},
            "constant=\\[2, 'unknown'].*Possible values.*\\[1, 2]",
        ),
    ],
    ids=[
        "no-constant",
        "too-many-constant",
        "not-enough-output",
        "single-output",
        "multi-output",
    ],
)
def test_constant_strategy_exceptions(y, params, err_msg):
    # 创建一个特征矩阵 X，其中的数据点在本测试中被忽略
    X = [[0], [0], [0], [0]]

    # 使用 DummyClassifier 模型，指定策略为 'constant'，传入参数 params
    clf = DummyClassifier(strategy="constant", **params)
    # 使用 pytest 断言捕获 ValueError，并匹配错误信息 err_msg
    with pytest.raises(ValueError, match=err_msg):
        clf.fit(X, y)


# 测试分类器在使用样本权重时的行为
def test_classification_sample_weight():
    # 创建一个特征矩阵 X，其中包含数据点 [[0], [0], [1]]（数据点不影响本测试）
    X = [[0], [0], [1]]
    # 定义标签 y，这是一个包含三个元素的列表，表示三个样本的分类标签
    y = [0, 1, 0]
    
    # 定义样本权重 sample_weight，也是一个包含三个元素的列表，对应 y 中每个样本的权重值
    sample_weight = [0.1, 1.0, 0.1]
    
    # 使用 DummyClassifier 创建分类器 clf，采用随机策略（stratified），并对模型进行拟合
    # X 是假设已定义的特征数据，此处未给出
    clf = DummyClassifier(strategy="stratified").fit(X, y, sample_weight)
    
    # 使用 assert_array_almost_equal 函数断言 clf 的 class_prior_ 属性，确保其值接近于指定的数组
    assert_array_almost_equal(clf.class_prior_, [0.2 / 1.2, 1.0 / 1.2])
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_constant_strategy_sparse_target(csc_container):
    X = [[0]] * 5  # ignored
    # 使用传入的容器 csc_container 将数组转换为稀疏矩阵
    y = csc_container(np.array([[0, 1], [4, 0], [1, 1], [1, 4], [1, 1]]))

    n_samples = len(X)

    # 创建一个 DummyClassifier 实例，采用常数策略 [1, 0]，设置随机种子为 0
    clf = DummyClassifier(strategy="constant", random_state=0, constant=[1, 0])
    # 使用 X 和 y 进行拟合
    clf.fit(X, y)
    # 对 X 进行预测
    y_pred = clf.predict(X)
    # 断言 y_pred 是稀疏矩阵
    assert sp.issparse(y_pred)
    # 断言预测结果与期望结果一致
    assert_array_equal(
        y_pred.toarray(), np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    )


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_uniform_strategy_sparse_target_warning(global_random_seed, csc_container):
    X = [[0]] * 5  # ignored
    # 使用传入的容器 csc_container 将数组转换为稀疏矩阵
    y = csc_container(np.array([[2, 1], [2, 2], [1, 4], [4, 2], [1, 1]]))

    # 创建一个 DummyClassifier 实例，采用均匀策略，设置随机种子为 global_random_seed
    clf = DummyClassifier(strategy="uniform", random_state=global_random_seed)
    # 断言在使用均匀策略时会发出 UserWarning，提醒可能不节省内存
    with pytest.warns(UserWarning, match="the uniform strategy would not save memory"):
        clf.fit(X, y)

    X = [[0]] * 500
    # 对 X 进行预测
    y_pred = clf.predict(X)

    # 对每个类别 k 计算预测结果的比例
    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        # 断言比例接近 1/3，精度为小数点后一位
        assert_almost_equal(p[1], 1 / 3, decimal=1)
        assert_almost_equal(p[2], 1 / 3, decimal=1)
        assert_almost_equal(p[4], 1 / 3, decimal=1)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_stratified_strategy_sparse_target(global_random_seed, csc_container):
    X = [[0]] * 5  # ignored
    # 使用传入的容器 csc_container 将数组转换为稀疏矩阵
    y = csc_container(np.array([[4, 1], [0, 0], [1, 1], [1, 4], [1, 1]]))

    # 创建一个 DummyClassifier 实例，采用分层策略，设置随机种子为 global_random_seed
    clf = DummyClassifier(strategy="stratified", random_state=global_random_seed)
    # 对 X 和 y 进行拟合
    clf.fit(X, y)

    X = [[0]] * 500
    # 对 X 进行预测
    y_pred = clf.predict(X)
    # 断言 y_pred 是稀疏矩阵
    assert sp.issparse(y_pred)
    # 将 y_pred 转换为数组
    y_pred = y_pred.toarray()

    # 对每个类别 k 计算预测结果的比例
    for k in range(y.shape[1]):
        p = np.bincount(y_pred[:, k]) / float(len(X))
        # 断言比例接近期望值，精度为小数点后一位
        assert_almost_equal(p[1], 3.0 / 5, decimal=1)
        assert_almost_equal(p[0], 1.0 / 5, decimal=1)
        assert_almost_equal(p[4], 1.0 / 5, decimal=1)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_most_frequent_and_prior_strategy_sparse_target(csc_container):
    X = [[0]] * 5  # ignored
    # 使用传入的容器 csc_container 将数组转换为稀疏矩阵
    y = csc_container(np.array([[1, 0], [1, 3], [4, 0], [0, 1], [1, 0]]))

    n_samples = len(X)
    # 期望的 y 结果
    y_expected = np.hstack([np.ones((n_samples, 1)), np.zeros((n_samples, 1))])
    for strategy in ("most_frequent", "prior"):
        # 创建一个 DummyClassifier 实例，采用最频繁策略或先验策略
        clf = DummyClassifier(strategy=strategy, random_state=0)
        # 对 X 和 y 进行拟合
        clf.fit(X, y)

        # 对 X 进行预测
        y_pred = clf.predict(X)
        # 断言 y_pred 是稀疏矩阵
        assert sp.issparse(y_pred)
        # 断言预测结果与期望结果一致
        assert_array_equal(y_pred.toarray(), y_expected)


def test_dummy_regressor_sample_weight(global_random_seed, n_samples=10):
    random_state = np.random.RandomState(seed=global_random_seed)

    X = [[0]] * n_samples
    # 随机生成 y 和样本权重
    y = random_state.rand(n_samples)
    sample_weight = random_state.rand(n_samples)

    # 创建一个 DummyRegressor 实例，采用均值策略，拟合时使用样本权重
    est = DummyRegressor(strategy="mean").fit(X, y, sample_weight)
    # 断言常数属性等于带权重的 y 的加权平均值
    assert est.constant_ == np.average(y, weights=sample_weight)
    # 使用 DummyRegressor 创建一个估算器对象，采用中位数策略拟合数据
    est = DummyRegressor(strategy="median").fit(X, y, sample_weight)
    # 断言估算器对象的 constant_ 属性等于使用加权样本权重计算的 y 数据的中位数
    assert est.constant_ == _weighted_percentile(y, sample_weight, 50.0)

    # 使用 DummyRegressor 创建一个估算器对象，采用分位数为 0.95 的策略拟合数据
    est = DummyRegressor(strategy="quantile", quantile=0.95).fit(X, y, sample_weight)
    # 断言估算器对象的 constant_ 属性等于使用加权样本权重计算的 y 数据的 95th 分位数
    assert est.constant_ == _weighted_percentile(y, sample_weight, 95.0)
# 测试使用 DummyRegressor 在 3D 数组上的表现
def test_dummy_regressor_on_3D_array():
    # 创建一个包含字符串的 3D 数组作为输入特征 X
    X = np.array([[["foo"]], [["bar"]], [["baz"]]])
    # 创建目标变量 y，全为整数 2
    y = np.array([2, 2, 2])
    # 期望的目标变量 y_expected，与 y 相同
    y_expected = np.array([2, 2, 2])
    # 初始化 DummyRegressor 模型
    cls = DummyRegressor()
    # 使用 X 和 y 拟合 DummyRegressor 模型
    cls.fit(X, y)
    # 对 X 进行预测
    y_pred = cls.predict(X)
    # 断言预测结果 y_pred 与期望值 y_expected 相等
    assert_array_equal(y_pred, y_expected)


# 测试使用 DummyClassifier 在 3D 数组上的表现
def test_dummy_classifier_on_3D_array():
    # 创建一个包含字符串的 3D 数组作为输入特征 X
    X = np.array([[["foo"]], [["bar"]], [["baz"]]])
    # 创建目标变量 y，为整数列表 [2, 2, 2]
    y = [2, 2, 2]
    # 期望的目标变量 y_expected，与 y 相同
    y_expected = [2, 2, 2]
    # 期望的类别概率 y_proba_expected，全为列表 [[1], [1], [1]]
    y_proba_expected = [[1], [1], [1]]
    # 初始化 DummyClassifier 模型，策略为 "stratified"
    cls = DummyClassifier(strategy="stratified")
    # 使用 X 和 y 拟合 DummyClassifier 模型
    cls.fit(X, y)
    # 对 X 进行分类预测
    y_pred = cls.predict(X)
    # 对 X 进行概率预测
    y_pred_proba = cls.predict_proba(X)
    # 断言分类预测结果 y_pred 与期望值 y_expected 相等
    assert_array_equal(y_pred, y_expected)
    # 断言概率预测结果 y_pred_proba 与期望值 y_proba_expected 相等
    assert_array_equal(y_pred_proba, y_proba_expected)


# 测试 DummyRegressor 在返回标准差时的行为
def test_dummy_regressor_return_std():
    # 创建一个包含三个相同元素的列表 X
    X = [[0]] * 3  # ignored
    # 创建目标变量 y，全为整数 2
    y = np.array([2, 2, 2])
    # 期望的标准差 y_std_expected，全为 0 的数组
    y_std_expected = np.array([0, 0, 0])
    # 初始化 DummyRegressor 模型
    cls = DummyRegressor()
    # 使用 X 和 y 拟合 DummyRegressor 模型，并返回预测结果和标准差
    y_pred_list = cls.predict(X, return_std=True)
    # 断言当 return_std 为 True 时，预测结果列表长度为 2
    assert len(y_pred_list) == 2
    # 断言预测结果列表的第二个元素应全为 0
    assert_array_equal(y_pred_list[1], y_std_expected)


# 使用参数化测试验证 DummyRegressor 的 score 方法对 None 的行为
@pytest.mark.parametrize(
    "y,y_test",
    [
        # y 和 y_test 参数化为不同的输入
        ([1, 1, 1, 2], [1.25] * 4),
        (np.array([[2, 2], [1, 1], [1, 1], [1, 1]]), [[1.25, 1.25]] * 4),
    ],
)
def test_regressor_score_with_None(y, y_test):
    # 初始化 DummyRegressor 模型
    reg = DummyRegressor()
    # 使用 None 和 y 拟合 DummyRegressor 模型
    reg.fit(None, y)
    # 断言使用 None 和 y_test 作为参数时，score 方法返回 1.0
    assert reg.score(None, y_test) == 1.0


# 使用参数化测试验证 DummyRegressor 在不同策略下的预测独立性
@pytest.mark.parametrize("strategy", ["mean", "median", "quantile", "constant"])
def test_regressor_prediction_independent_of_X(strategy):
    # 创建目标变量 y
    y = [0, 2, 1, 1]
    # 创建 X1，全为包含单个元素 [0] 的列表，作为第一个数据集
    X1 = [[0]] * 4
    # 初始化 DummyRegressor 模型 reg1，使用指定策略和常数值
    reg1 = DummyRegressor(strategy=strategy, constant=0, quantile=0.7)
    # 使用 X1 和 y 拟合 reg1 模型
    reg1.fit(X1, y)
    # 对 X1 进行预测
    predictions1 = reg1.predict(X1)

    # 创建 X2，全为包含单个元素 [1] 的列表，作为第二个数据集
    X2 = [[1]] * 4
    # 初始化 DummyRegressor 模型 reg2，使用相同的策略和常数值
    reg2 = DummyRegressor(strategy=strategy, constant=0, quantile=0.7)
    # 使用 X2 和 y 拟合 reg2 模型
    reg2.fit(X2, y)
    # 对 X2 进行预测
    predictions2 = reg2.predict(X2)

    # 断言两个数据集的预测结果 predictions1 和 predictions2 相等
    assert_array_equal(predictions1, predictions2)


# 使用参数化测试验证 DummyClassifier 模型输出概率的数据类型
@pytest.mark.parametrize(
    "strategy", ["stratified", "most_frequent", "prior", "uniform", "constant"]
)
def test_dtype_of_classifier_probas(strategy):
    # 创建目标变量 y
    y = [0, 2, 1, 1]
    # 创建一个全为 0 的数组 X
    X = np.zeros(4)
    # 初始化 DummyClassifier 模型，使用指定策略和随机种子
    model = DummyClassifier(strategy=strategy, random_state=0, constant=0)
    # 使用 X 和 y 拟合 DummyClassifier 模型，并进行概率预测
    probas = model.fit(X, y).predict_proba(X)

    # 断言概率预测结果 probas 的数据类型为 np.float64
    assert probas.dtype == np.float64
```