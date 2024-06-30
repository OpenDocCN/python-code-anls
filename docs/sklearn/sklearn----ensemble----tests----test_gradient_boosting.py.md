# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_gradient_boosting.py`

```
"""
Testing for the gradient boosting module (sklearn.ensemble.gradient_boosting).
"""

# 导入必要的库
import re  # 导入正则表达式库
import warnings  # 导入警告处理模块

import numpy as np  # 导入数值计算库numpy
import pytest  # 导入测试框架pytest
from numpy.testing import assert_allclose  # 导入numpy的数组比较函数

from sklearn import datasets  # 导入sklearn中的数据集模块
from sklearn.base import clone  # 导入模型克隆函数
from sklearn.datasets import make_classification, make_regression  # 导入分类和回归数据生成函数
from sklearn.dummy import DummyClassifier, DummyRegressor  # 导入虚拟分类器和回归器
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # 导入梯度提升分类器和回归器
from sklearn.ensemble._gb import _safe_divide  # 导入内部函数_safe_divide
from sklearn.ensemble._gradient_boosting import predict_stages  # 导入预测阶段函数
from sklearn.exceptions import DataConversionWarning, NotFittedError  # 导入数据转换警告和未拟合异常
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.metrics import mean_squared_error  # 导入均方误差指标
from sklearn.model_selection import train_test_split  # 导入数据集分割函数
from sklearn.pipeline import make_pipeline  # 导入创建管道的函数
from sklearn.preprocessing import scale  # 导入数据标准化函数
from sklearn.svm import NuSVR  # 导入NuSVR支持向量回归器
from sklearn.utils import check_random_state  # 导入随机状态检查函数
from sklearn.utils._mocking import NoSampleWeightWrapper  # 导入模拟无样本权重的包装器
from sklearn.utils._param_validation import InvalidParameterError  # 导入无效参数异常
from sklearn.utils._testing import (  # 导入测试相关工具
    assert_array_almost_equal,
    assert_array_equal,
    skip_if_32bit,
)
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS  # 导入数据容器修复模块

GRADIENT_BOOSTING_ESTIMATORS = [GradientBoostingClassifier, GradientBoostingRegressor]  # 梯度提升估计器列表

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]  # 简单样本特征
y = [-1, -1, -1, 1, 1, 1]  # 简单样本标签
T = [[-1, -1], [2, 2], [3, 2]]  # 测试集特征
true_result = [-1, 1, 1]  # 测试集的真实结果

# also make regression dataset
X_reg, y_reg = make_regression(
    n_samples=100, n_features=4, n_informative=8, noise=10, random_state=7
)
y_reg = scale(y_reg)  # 生成回归数据集并对标签进行标准化处理

rng = np.random.RandomState(0)  # 创建随机状态生成器对象
# also load the iris dataset
# and randomly permute it
iris = datasets.load_iris()  # 加载鸢尾花数据集
perm = rng.permutation(iris.target.size)  # 随机排列鸢尾花数据集的标签
iris.data = iris.data[perm]  # 根据排列重新排序数据
iris.target = iris.target[perm]  # 根据排列重新排序标签


def test_exponential_n_classes_gt_2():
    """Test exponential loss raises for n_classes > 2."""
    clf = GradientBoostingClassifier(loss="exponential")  # 创建梯度提升分类器对象，使用指数损失函数
    msg = "loss='exponential' is only suitable for a binary classification"
    with pytest.raises(ValueError, match=msg):  # 断言捕获值错误，检测是否引发指定消息的异常
        clf.fit(iris.data, iris.target)  # 对鸢尾花数据集进行拟合


def test_raise_if_init_has_no_predict_proba():
    """Test raise if init_ has no predict_proba method."""
    clf = GradientBoostingClassifier(init=GradientBoostingRegressor)  # 创建梯度提升分类器对象，使用回归器作为初始化器
    msg = (
        "The 'init' parameter of GradientBoostingClassifier must be a str among "
        "{'zero'}, None or an object implementing 'fit' and 'predict_proba'."
    )
    with pytest.raises(ValueError, match=msg):  # 断言捕获值错误，检测是否引发指定消息的异常
        clf.fit(X, y)  # 对简单样本数据集进行拟合


@pytest.mark.parametrize("loss", ("log_loss", "exponential"))
def test_classification_toy(loss, global_random_seed):
    # Check classification on a toy dataset.
    clf = GradientBoostingClassifier(
        loss=loss, n_estimators=10, random_state=global_random_seed
    )  # 创建梯度提升分类器对象，指定损失函数和基础估计器数量

    with pytest.raises(ValueError):
        clf.predict(T)  # 断言捕获值错误，检测是否引发异常

    clf.fit(X, y)  # 对简单样本数据集进行拟合
    # 断言分类器 clf 对测试集 T 的预测结果与真实结果 true_result 相等
    assert_array_equal(clf.predict(T), true_result)
    
    # 断言分类器 clf 的 estimators_ 列表长度为 10
    assert 10 == len(clf.estimators_)
    
    # 计算训练过程中的对数损失的递减情况，存储在 log_loss_decrease 变量中
    log_loss_decrease = clf.train_score_[:-1] - clf.train_score_[1:]
    
    # 断言 log_loss_decrease 中存在至少一个值大于等于 0.0
    assert np.any(log_loss_decrease >= 0.0)
    
    # 对输入数据 X 应用分类器 clf，得到叶子节点索引，存储在 leaves 变量中
    leaves = clf.apply(X)
    
    # 断言 leaves 的形状为 (6, 10, 1)
    assert leaves.shape == (6, 10, 1)
# 使用pytest装饰器标记此函数为参数化测试，参数为loss，可以取"log_loss"或"exponential"
@pytest.mark.parametrize("loss", ("log_loss", "exponential"))
def test_classification_synthetic(loss, global_random_seed):
    # 在合成数据集上测试GradientBoostingClassifier，该数据集由Hastie等人在ESLII中的Figure 10.9使用
    # 注意Figure 10.9重用了Figure 10.2生成的数据集，应该有2,000个训练数据点和10,000个测试数据点
    # 这里故意使用一个较小的变体以加快测试速度，但结论仍然相同，尽管数据集较小
    X, y = datasets.make_hastie_10_2(n_samples=2000, random_state=global_random_seed)

    split_idx = 500
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 增加树的数量应该降低测试错误
    common_params = {
        "max_depth": 1,
        "learning_rate": 1.0,
        "loss": loss,
        "random_state": global_random_seed,
    }
    # 创建拥有10个估计器的GradientBoostingClassifier对象，应用公共参数
    gbrt_10_stumps = GradientBoostingClassifier(n_estimators=10, **common_params)
    gbrt_10_stumps.fit(X_train, y_train)

    # 创建拥有50个估计器的GradientBoostingClassifier对象，应用公共参数
    gbrt_50_stumps = GradientBoostingClassifier(n_estimators=50, **common_params)
    gbrt_50_stumps.fit(X_train, y_train)

    # 断言：10个估计器的模型的测试分数应小于50个估计器的模型的测试分数
    assert gbrt_10_stumps.score(X_test, y_test) < gbrt_50_stumps.score(X_test, y_test)

    # 决策桩（stumps）更适合这个数据集，特别是在具有大量估计器的情况下
    common_params = {
        "n_estimators": 200,
        "learning_rate": 1.0,
        "loss": loss,
        "random_state": global_random_seed,
    }
    # 创建一个最大深度为1的GradientBoostingClassifier对象，应用公共参数
    gbrt_stumps = GradientBoostingClassifier(max_depth=1, **common_params)
    gbrt_stumps.fit(X_train, y_train)

    # 创建一个最大叶子节点数为10的GradientBoostingClassifier对象，应用公共参数
    gbrt_10_nodes = GradientBoostingClassifier(max_leaf_nodes=10, **common_params)
    gbrt_10_nodes.fit(X_train, y_train)

    # 断言：深度为1的模型的测试分数应大于最大叶子节点数为10的模型的测试分数
    assert gbrt_stumps.score(X_test, y_test) > gbrt_10_nodes.score(X_test, y_test)


# 使用pytest装饰器标记此函数为参数化测试，参数为loss和subsample，分别可以取"squared_error", "absolute_error", "huber"和(1.0, 0.5)
@pytest.mark.parametrize("loss", ("squared_error", "absolute_error", "huber"))
@pytest.mark.parametrize("subsample", (1.0, 0.5))
def test_regression_dataset(loss, subsample, global_random_seed):
    # 在回归数据集上使用最小二乘法和最小绝对偏差检查一致性
    ones = np.ones(len(y_reg))
    last_y_pred = None
    # 对于三种不同的样本权重进行迭代：无权重、全为1、全为2的权重数组
    for sample_weight in [None, ones, 2 * ones]:
        # 设置了学习率（learning_rate）、树的最大深度（max_depth）和基学习器的数量（n_estimators）
        # 以获得在训练集上达到低均方误差（MSE）的准确模型，并保持资源使用在可接受范围内
        reg = GradientBoostingRegressor(
            n_estimators=30,
            loss=loss,
            max_depth=4,
            subsample=subsample,
            min_samples_split=2,
            random_state=global_random_seed,
            learning_rate=0.5,
        )

        # 使用梯度提升回归模型拟合数据，可选择性地传入样本权重
        reg.fit(X_reg, y_reg, sample_weight=sample_weight)
        
        # 获取每个样本对应的叶子节点索引
        leaves = reg.apply(X_reg)
        # 断言叶子节点索引的形状应为 (100, 30)
        assert leaves.shape == (100, 30)

        # 对测试数据进行预测
        y_pred = reg.predict(X_reg)
        # 计算预测值与实际标签之间的均方误差（MSE）
        mse = mean_squared_error(y_reg, y_pred)
        # 断言均方误差应小于 0.05
        assert mse < 0.05

        if last_y_pred is not None:
            # FIXME: 暂时跳过此测试。这是因为带权重和不带权重的梯度提升回归树在使用 `DummyRegressor` 进行初始化时，
            # 中位数的计算实现不同。未来我们应确保两种实现应该是相同的。详见 PR #17377。
            # assert_allclose(last_y_pred, y_pred)
            pass

        # 更新上一次的预测结果
        last_y_pred = y_pred
@pytest.mark.parametrize("subsample", (1.0, 0.5))
@pytest.mark.parametrize("sample_weight", (None, 1))
# 定义测试函数，使用参数化测试来测试不同的subsample和sample_weight组合
def test_iris(subsample, sample_weight, global_random_seed):
    if sample_weight == 1:
        # 如果sample_weight为1，将其设为与iris目标长度相同的全1数组
        sample_weight = np.ones(len(iris.target))
    # 检查在数据集iris上的一致性
    clf = GradientBoostingClassifier(
        n_estimators=100,
        loss="log_loss",
        random_state=global_random_seed,
        subsample=subsample,
    )
    # 使用梯度提升分类器拟合数据
    clf.fit(iris.data, iris.target, sample_weight=sample_weight)
    # 计算分类器在训练数据上的得分
    score = clf.score(iris.data, iris.target)
    # 断言分类器的得分大于0.9
    assert score > 0.9

    # 计算叶子节点索引
    leaves = clf.apply(iris.data)
    # 断言叶子节点的形状为(150, 100, 3)
    assert leaves.shape == (150, 100, 3)


def test_regression_synthetic(global_random_seed):
    # 在Leo Breiman的`Bagging Predictors?`中使用合成回归数据集进行测试
    random_state = check_random_state(global_random_seed)
    regression_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "min_samples_split": 2,
        "learning_rate": 0.1,
        "loss": "squared_error",
        "random_state": global_random_seed,
    }

    # Friedman1数据集
    X, y = datasets.make_friedman1(n_samples=1200, random_state=random_state, noise=1.0)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    # 计算均方误差
    mse = mean_squared_error(y_test, clf.predict(X_test))
    # 断言均方误差小于6.5
    assert mse < 6.5

    # Friedman2数据集
    X, y = datasets.make_friedman2(n_samples=1200, random_state=random_state)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    # 计算均方误差
    mse = mean_squared_error(y_test, clf.predict(X_test))
    # 断言均方误差小于2500.0
    assert mse < 2500.0

    # Friedman3数据集
    X, y = datasets.make_friedman3(n_samples=1200, random_state=random_state)
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    clf = GradientBoostingRegressor(**regression_params)
    clf.fit(X_train, y_train)
    # 计算均方误差
    mse = mean_squared_error(y_test, clf.predict(X_test))
    # 断言均方误差小于0.025
    assert mse < 0.025


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (GradientBoostingRegressor, X_reg, y_reg),
        (GradientBoostingClassifier, iris.data, iris.target),
    ],
)
# 测试梯度提升模型是否公开了属性feature_importances_
def test_feature_importances(GradientBoosting, X, y):
    gbdt = GradientBoosting()
    # 断言梯度提升模型没有feature_importances_属性
    assert not hasattr(gbdt, "feature_importances_")
    # 拟合模型
    gbdt.fit(X, y)
    # 断言梯度提升模型现在有feature_importances_属性
    assert hasattr(gbdt, "feature_importances_")


def test_probability_log(global_random_seed):
    # 预测概率
    clf = GradientBoostingClassifier(n_estimators=100, random_state=global_random_seed)

    # 使用pytest.raises检查是否会抛出值错误异常
    with pytest.raises(ValueError):
        clf.predict_proba(T)

    # 拟合数据
    clf.fit(X, y)
    # 断言预测结果与真实结果数组相等
    assert_array_equal(clf.predict(T), true_result)

    # 检查预测概率是否在[0, 1]之间
    # 使用分类器 clf 对测试数据集 T 进行预测，返回每个样本属于各个类别的概率
    y_proba = clf.predict_proba(T)

    # 断言所有预测概率值大于等于0且小于等于1，确保概率值的合法性
    assert np.all(y_proba >= 0.0)
    assert np.all(y_proba <= 1.0)

    # 从预测的概率中推导出最终的类别预测值
    # y_proba.argmax(axis=1) 返回每行中最大值的索引，即预测概率最高的类别索引
    y_pred = clf.classes_.take(y_proba.argmax(axis=1), axis=0)

    # 断言预测的结果与真实结果数组 true_result 相等，验证预测的准确性
    assert_array_equal(y_pred, true_result)
def test_single_class_with_sample_weight():
    # 定义样本权重列表，标识只有最后三个样本有权重
    sample_weight = [0, 0, 0, 1, 1, 1]
    # 初始化梯度提升分类器，设置随机种子
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    # 错误信息，用于验证是否抛出 ValueError 异常
    msg = (
        "y contains 1 class after sample_weight trimmed classes with "
        "zero weights, while a minimum of 2 classes are required."
    )
    # 断言抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match=msg):
        # 使用样本权重训练分类器
        clf.fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_check_inputs_predict_stages(csc_container):
    # 检查 predict_stages 是否在 X 类型不支持时抛出错误
    x, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 转换成稀疏 CSC 格式
    x_sparse_csc = csc_container(x)
    # 初始化梯度提升分类器，设置随机种子
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    # 使用 x, y 训练分类器
    clf.fit(x, y)
    # 初始化 score 数组
    score = np.zeros((y.shape)).reshape(-1, 1)
    # 错误信息，验证是否抛出 ValueError 异常
    err_msg = "When X is a sparse matrix, a CSR format is expected"
    with pytest.raises(ValueError, match=err_msg):
        # 预测每个阶段的分数
        predict_stages(clf.estimators_, x_sparse_csc, clf.learning_rate, score)
    # 转换成 Fortran 顺序的数组
    x_fortran = np.asfortranarray(x)
    # 断言抛出 ValueError 异常，并匹配特定的错误信息
    with pytest.raises(ValueError, match="X should be C-ordered np.ndarray"):
        # 预测每个阶段的分数
        predict_stages(clf.estimators_, x_fortran, clf.learning_rate, score)


def test_max_feature_regression(global_random_seed):
    # 测试随机状态是否正确设置
    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=global_random_seed)

    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    # 初始化梯度提升分类器，设置各种参数
    gbrt = GradientBoostingClassifier(
        n_estimators=100,
        min_samples_split=5,
        max_depth=2,
        learning_rate=0.1,
        max_features=2,
        random_state=global_random_seed,
    )
    # 使用训练集训练分类器
    gbrt.fit(X_train, y_train)
    # 计算对数损失
    log_loss = gbrt._loss(y_test, gbrt.decision_function(X_test))
    # 断言对数损失小于 0.5
    assert log_loss < 0.5, "GB failed with deviance %.4f" % log_loss


def test_feature_importance_regression(
    fetch_california_housing_fxt, global_random_seed
):
    """Test that Gini importance is calculated correctly.

    This test follows the example from [1]_ (pg. 373).

    .. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements
       of statistical learning. New York: Springer series in statistics.
    """
    # 获取加利福尼亚房屋数据集
    california = fetch_california_housing_fxt()
    X, y = california.data, california.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=global_random_seed
    )

    # 初始化梯度提升回归器，设置各种参数
    reg = GradientBoostingRegressor(
        loss="huber",
        learning_rate=0.1,
        max_leaf_nodes=6,
        n_estimators=100,
        random_state=global_random_seed,
    )
    # 使用训练集训练回归器
    reg.fit(X_train, y_train)
    # 根据特征重要性排序特征索引
    sorted_idx = np.argsort(reg.feature_importances_)[::-1]
    sorted_features = [california.feature_names[s] for s in sorted_idx]

    # 断言最重要的特征是收入中位数
    assert sorted_features[0] == "MedInc"
    # 断言语句，用于确保集合 sorted_features[1:4] 中的元素与 {"Longitude", "AveOccup", "Latitude"} 相同
    assert set(sorted_features[1:4]) == {"Longitude", "AveOccup", "Latitude"}
    # 断言成功则说明 sorted_features 列表的第 1 至第 3 个元素确实包含 "Longitude", "AveOccup", "Latitude" 三个特征
    # 如果断言失败则会抛出 AssertionError 异常
def test_max_features():
    # Test if max features is set properly for floats and str.
    # 生成一个具有10个特征的二元分类数据集，共12000个样本，固定随机种子以确保可重复性
    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
    # 获取数据集的特征数
    _, n_features = X.shape

    # 将数据集划分为训练集（前2000个样本）
    X_train = X[:2000]
    y_train = y[:2000]

    # 使用GradientBoostingClassifier，设置max_features为None，创建分类器实例
    gbrt = GradientBoostingClassifier(n_estimators=1, max_features=None)
    # 在训练集上拟合分类器
    gbrt.fit(X_train, y_train)
    # 断言分类器实例的max_features_属性是否等于数据集的特征数
    assert gbrt.max_features_ == n_features

    # 使用GradientBoostingRegressor，设置max_features为None，创建回归器实例
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=None)
    # 在训练集上拟合回归器
    gbrt.fit(X_train, y_train)
    # 断言回归器实例的max_features_属性是否等于数据集的特征数
    assert gbrt.max_features_ == n_features

    # 使用GradientBoostingRegressor，设置max_features为0.3（特征数的30%），创建回归器实例
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=0.3)
    # 在训练集上拟合回归器
    gbrt.fit(X_train, y_train)
    # 断言回归器实例的max_features_属性是否等于特征数的30%（向下取整）
    assert gbrt.max_features_ == int(n_features * 0.3)

    # 使用GradientBoostingRegressor，设置max_features为"sqrt"，创建回归器实例
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features="sqrt")
    # 在训练集上拟合回归器
    gbrt.fit(X_train, y_train)
    # 断言回归器实例的max_features_属性是否等于特征数的平方根（向下取整）
    assert gbrt.max_features_ == int(np.sqrt(n_features))

    # 使用GradientBoostingRegressor，设置max_features为"log2"，创建回归器实例
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features="log2")
    # 在训练集上拟合回归器
    gbrt.fit(X_train, y_train)
    # 断言回归器实例的max_features_属性是否等于特征数的以2为底的对数（向下取整）
    assert gbrt.max_features_ == int(np.log2(n_features))

    # 使用GradientBoostingRegressor，设置max_features为0.01 / 特征数，创建回归器实例
    gbrt = GradientBoostingRegressor(n_estimators=1, max_features=0.01 / X.shape[1])
    # 在训练集上拟合回归器
    gbrt.fit(X_train, y_train)
    # 断言回归器实例的max_features_属性是否等于1（当计算结果小于1时，取1）
    assert gbrt.max_features_ == 1


def test_staged_predict():
    # Test whether staged decision function eventually gives
    # the same prediction.
    # 生成一个具有1个特征的回归数据集，共1200个样本，固定随机种子以确保可重复性，并引入噪声
    X, y = datasets.make_friedman1(n_samples=1200, random_state=1, noise=1.0)
    # 将数据集划分为训练集（前200个样本）和测试集（后1000个样本）
    X_train, y_train = X[:200], y[:200]
    X_test = X[200:]
    # 创建GradientBoostingRegressor分类器实例
    clf = GradientBoostingRegressor()
    # 测试未拟合的分类器是否会抛出ValueError异常
    with pytest.raises(ValueError):
        np.fromiter(clf.staged_predict(X_test), dtype=np.float64)

    # 在训练集上拟合分类器
    clf.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = clf.predict(X_test)

    # 遍历分类器的staged_predict函数输出，断言每个预测结果的形状与y_pred相同
    for y in clf.staged_predict(X_test):
        assert y.shape == y_pred.shape

    # 断言最终预测结果与y_pred几乎相等
    assert_array_almost_equal(y_pred, y)


def test_staged_predict_proba():
    # Test whether staged predict proba eventually gives
    # the same prediction.
    # 生成一个具有10个特征的二元分类数据集，共1200个样本，固定随机种子以确保可重复性
    X, y = datasets.make_hastie_10_2(n_samples=1200, random_state=1)
    # 将数据集划分为训练集（前200个样本）和测试集（后1000个样本）
    X_train, y_train = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]
    # 创建GradientBoostingClassifier分类器实例，设置n_estimators为20
    clf = GradientBoostingClassifier(n_estimators=20)
    # 测试未拟合的分类器是否会抛出NotFittedError异常
    with pytest.raises(NotFittedError):
        np.fromiter(clf.staged_predict_proba(X_test), dtype=np.float64)

    # 在训练集上拟合分类器
    clf.fit(X_train, y_train)

    # 遍历分类器的staged_predict函数输出，断言每个预测结果的形状与y_test相同
    for y_pred in clf.staged_predict(X_test):
        assert y_test.shape == y_pred.shape

    # 断言最终预测结果与predict函数结果相等
    assert_array_equal(clf.predict(X_test), y_pred)

    # 遍历分类器的staged_predict_proba函数输出，断言每个预测结果的形状和列数是否符合预期
    for staged_proba in clf.staged_predict_proba(X_test):
        assert y_test.shape[0] == staged_proba.shape[0]
        assert 2 == staged_proba.shape[1]

    # 断言最终预测结果与predict_proba函数结果几乎相等
    assert_array_almost_equal(clf.predict_proba(X_test), staged_proba)
    # 使用全局随机种子创建随机数生成器对象
    rng = np.random.RandomState(global_random_seed)
    # 创建一个 10x3 的随机数数组 X
    X = rng.uniform(size=(10, 3))
    # 根据 X 的第一列创建一个整数数组 y，并确保不预测零值
    y = (4 * X[:, 0]).astype(int) + 1
    # 创建一个 Estimator 的实例
    estimator = Estimator()
    # 使用 X 和 y 来拟合估算器
    estimator.fit(X, y)
    # 针对每个函数（"predict", "decision_function", "predict_proba"），进行如下操作
    for func in ["predict", "decision_function", "predict_proba"]:
        # 获取估算器对象的阶段函数（staged_func），如 staged_predict、staged_decision_function、staged_predict_proba
        staged_func = getattr(estimator, "staged_" + func, None)
        # 如果 staged_func 为 None，则跳过当前循环，因为该回归器没有对应的 staged 函数
        if staged_func is None:
            continue
        # 使用警告记录来捕获可能的警告信息
        with warnings.catch_warnings(record=True):
            # 获取 staged_func(X) 的结果，并将其转换为列表形式
            staged_result = list(staged_func(X))
        # 将 staged_result 的第二个元素的所有元素设置为零
        staged_result[1][:] = 0
        # 断言 staged_result 的第一个元素中没有零值
        assert np.all(staged_result[0] != 0)
# Check model serialization.
def test_serialization():
    # 创建一个梯度提升分类器，设定参数：100棵树，随机种子为1
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    # 使用训练数据 X 和标签 y 进行模型训练
    clf.fit(X, y)
    # 断言预测结果与真实结果相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言模型的 estimators_ 属性中包含 100 棵树
    assert 100 == len(clf.estimators_)

    try:
        import cPickle as pickle  # 尝试导入 cPickle，Python 2 中的 pickle 实现
    except ImportError:
        import pickle  # 如果导入失败，则使用 Python 3 的 pickle

    # 序列化分类器对象 clf，使用 pickle 的 HIGHEST_PROTOCOL 协议
    serialized_clf = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
    # 清空 clf 对象
    clf = None
    # 反序列化之前序列化的对象 serialized_clf
    clf = pickle.loads(serialized_clf)
    # 断言反序列化后的对象预测结果与真实结果相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言反序列化后的对象的 estimators_ 属性中包含 100 棵树
    assert 100 == len(clf.estimators_)


# Check if we can fit even though all targets are equal.
def test_degenerate_targets():
    # 创建一个梯度提升分类器，设定参数：100棵树，随机种子为1
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    # 应该会抛出 ValueError 异常，因为所有的目标值都相等
    with pytest.raises(ValueError):
        clf.fit(X, np.ones(len(X)))

    # 创建一个梯度提升回归器，设定参数：100棵树，随机种子为1
    clf = GradientBoostingRegressor(n_estimators=100, random_state=1)
    # 使用所有目标值都为1的数据进行训练
    clf.fit(X, np.ones(len(X)))
    # 预测随机数生成的两个数据点，并断言预测结果与真实结果相等
    clf.predict([rng.rand(2)])
    assert_array_equal(np.ones((1,), dtype=np.float64), clf.predict([rng.rand(2)]))


# Check if quantile loss with alpha=0.5 equals absolute_error.
def test_quantile_loss(global_random_seed):
    # 创建一个梯度提升回归器，使用分位数损失，alpha=0.5，设定参数：100棵树，深度为4，全局随机种子
    clf_quantile = GradientBoostingRegressor(
        n_estimators=100,
        loss="quantile",
        max_depth=4,
        alpha=0.5,
        random_state=global_random_seed,
    )

    # 使用回归数据 X_reg 和标签 y_reg 进行模型训练
    clf_quantile.fit(X_reg, y_reg)
    # 使用训练好的模型进行预测
    y_quantile = clf_quantile.predict(X_reg)

    # 创建一个梯度提升回归器，使用绝对误差损失，设定参数：100棵树，深度为4，全局随机种子
    clf_ae = GradientBoostingRegressor(
        n_estimators=100,
        loss="absolute_error",
        max_depth=4,
        random_state=global_random_seed,
    )

    # 使用回归数据 X_reg 和标签 y_reg 进行模型训练
    clf_ae.fit(X_reg, y_reg)
    # 使用训练好的模型进行预测
    y_ae = clf_ae.predict(X_reg)
    # 断言使用分位数损失和绝对误差损失训练的模型预测结果非常接近
    assert_allclose(y_quantile, y_ae)


# Test with non-integer class labels.
def test_symbol_labels():
    # 创建一个梯度提升分类器，设定参数：100棵树，随机种子为1
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    # 将标签 y 转换为字符串类型的列表
    symbol_y = list(map(str, y))

    # 使用训练数据 X 和字符串类型的标签进行模型训练
    clf.fit(X, symbol_y)
    # 断言预测结果与真实结果相等（字符串类型）
    assert_array_equal(clf.predict(T), list(map(str, true_result)))
    # 断言模型的 estimators_ 属性中包含 100 棵树
    assert 100 == len(clf.estimators_)


# Test with float class labels.
def test_float_class_labels():
    # 创建一个梯度提升分类器，设定参数：100棵树，随机种子为1
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    # 将标签 y 转换为 float32 类型的数组
    float_y = np.asarray(y, dtype=np.float32)

    # 使用训练数据 X 和 float32 类型的标签进行模型训练
    clf.fit(X, float_y)
    # 断言预测结果与真实结果相等（float32 类型）
    assert_array_equal(clf.predict(T), np.asarray(true_result, dtype=np.float32))
    # 断言模型的 estimators_ 属性中包含 100 棵树
    assert 100 == len(clf.estimators_)


# Test with float class labels.
def test_shape_y():
    # 创建一个梯度提升分类器，设定参数：100棵树，随机种子为1
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)

    # 将标签 y 转换为 int32 类型的数组，并增加一个维度
    y_ = np.asarray(y, dtype=np.int32)
    y_ = y_[:, np.newaxis]

    # 使用警告 pytest.warns 检测数据转换警告信息
    warn_msg = (
        "A column-vector y was passed when a 1d array was expected. "
        "Please change the shape of y to \\(n_samples, \\), for "
        "example using ravel()."
    )
    with pytest.warns(DataConversionWarning, match=warn_msg):
        # 使用警告抛出的情况下，训练模型 clf 使用数据 X 和改变形状后的标签 y_
        clf.fit(X, y_)
    # 验证分类器 clf 对输入数据 T 的预测结果是否与 true_result 数组完全一致
    assert_array_equal(clf.predict(T), true_result)
    
    # 确保分类器 clf 中包含的估计器（estimators）数量为 100
    assert 100 == len(clf.estimators_)
def test_mem_layout():
    # Test with different memory layouts of X and y

    # 使用 np.asfortranarray 将 X 转换为列优先存储的数组
    X_ = np.asfortranarray(X)
    # 创建梯度提升分类器对象，设置参数并训练模型
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(X_, y)
    # 断言预测结果与真实结果相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言模型的 estimators_ 数量为 100
    assert 100 == len(clf.estimators_)

    # 使用 np.ascontiguousarray 将 X 转换为行优先存储的数组
    X_ = np.ascontiguousarray(X)
    # 创建梯度提升分类器对象，设置参数并训练模型
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(X_, y)
    # 断言预测结果与真实结果相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言模型的 estimators_ 数量为 100
    assert 100 == len(clf.estimators_)

    # 将 y 转换为 int32 类型的数组
    y_ = np.asarray(y, dtype=np.int32)
    # 使用 np.ascontiguousarray 将 y 转换为行优先存储的数组
    y_ = np.ascontiguousarray(y_)
    # 创建梯度提升分类器对象，设置参数并训练模型
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(X, y_)
    # 断言预测结果与真实结果相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言模型的 estimators_ 数量为 100
    assert 100 == len(clf.estimators_)

    # 将 y 转换为 int32 类型的数组
    y_ = np.asarray(y, dtype=np.int32)
    # 使用 np.asfortranarray 将 y 转换为列优先存储的数组
    y_ = np.asfortranarray(y_)
    # 创建梯度提升分类器对象，设置参数并训练模型
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    clf.fit(X, y_)
    # 断言预测结果与真实结果相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言模型的 estimators_ 数量为 100
    assert 100 == len(clf.estimators_)


@pytest.mark.parametrize("GradientBoostingEstimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_oob_improvement(GradientBoostingEstimator):
    # Test if oob improvement has correct shape and regression test.

    # 创建指定类型的梯度提升估计器对象，设置参数并训练模型
    estimator = GradientBoostingEstimator(
        n_estimators=100, random_state=1, subsample=0.5
    )
    estimator.fit(X, y)
    # 断言 oob_improvement_ 的形状为 (100,)
    assert estimator.oob_improvement_.shape[0] == 100
    # 硬编码的回归测试 - 如果 oob 计算修改，需要更改
    assert_array_almost_equal(
        estimator.oob_improvement_[:5],
        np.array([0.19, 0.15, 0.12, -0.11, 0.11]),
        decimal=2,
    )


@pytest.mark.parametrize("GradientBoostingEstimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_oob_scores(GradientBoostingEstimator):
    # Test if oob scores has correct shape and regression test.

    # 创建样本数据 X, y
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 创建指定类型的梯度提升估计器对象，设置参数并训练模型
    estimator = GradientBoostingEstimator(
        n_estimators=100, random_state=1, subsample=0.5
    )
    estimator.fit(X, y)
    # 断言 oob_scores_ 的形状为 (100,)
    assert estimator.oob_scores_.shape[0] == 100
    # 断言最后一个 oob_score_ 与 oob_scores_ 的最后一个元素近似相等
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)

    # 创建指定类型的梯度提升估计器对象，设置参数并训练模型
    estimator = GradientBoostingEstimator(
        n_estimators=100,
        random_state=1,
        subsample=0.5,
        n_iter_no_change=5,
    )
    estimator.fit(X, y)
    # 断言 oob_scores_ 的形状小于 (100,)
    assert estimator.oob_scores_.shape[0] < 100
    # 断言最后一个 oob_score_ 与 oob_scores_ 的最后一个元素近似相等
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)


@pytest.mark.parametrize(
    "GradientBoostingEstimator, oob_attribute",
    [
        (GradientBoostingClassifier, "oob_improvement_"),
        (GradientBoostingClassifier, "oob_scores_"),
        (GradientBoostingClassifier, "oob_score_"),
        (GradientBoostingRegressor, "oob_improvement_"),
        (GradientBoostingRegressor, "oob_scores_"),
        (GradientBoostingRegressor, "oob_score_"),
    ],
)
def test_oob_attributes_error(GradientBoostingEstimator, oob_attribute):
    """
    Check that we raise an AttributeError when the OOB statistics were not computed.
    """
    # 使用 make_hastie_10_2 数据集生成器创建数据集 X 和标签 y，共 100 个样本，随机种子为 1
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    
    # 创建梯度提升估计器对象，设定参数：100 个基础估计器，随机种子为 1，子样本比例为 1.0
    estimator = GradientBoostingEstimator(
        n_estimators=100,
        random_state=1,
        subsample=1.0,
    )
    
    # 使用梯度提升估计器拟合数据集 X 和标签 y
    estimator.fit(X, y)
    
    # 使用 pytest 的 raises 方法检查是否抛出 AttributeError 异常
    with pytest.raises(AttributeError):
        # 尝试访问未计算的 out-of-bag (OOB) 统计属性
        estimator.oob_attribute
def test_oob_multilcass_iris():
    # Check OOB improvement on multi-class dataset.
    # 创建一个GradientBoostingClassifier对象，设置参数并初始化
    estimator = GradientBoostingClassifier(
        n_estimators=100, loss="log_loss", random_state=1, subsample=0.5
    )
    # 使用iris数据集进行训练
    estimator.fit(iris.data, iris.target)
    # 计算模型在训练数据上的准确率得分
    score = estimator.score(iris.data, iris.target)
    # 断言模型的准确率得分大于0.9
    assert score > 0.9
    # 断言模型的OOB改善数组的长度与估计器中设置的估计器数量相同
    assert estimator.oob_improvement_.shape[0] == estimator.n_estimators
    # 断言模型的OOB评分数组的长度与估计器中设置的估计器数量相同
    assert estimator.oob_scores_.shape[0] == estimator.n_estimators
    # 断言模型的最后一个OOB评分与使用pytest.approx函数比较，与OOB得分相似
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)

    # 创建另一个GradientBoostingClassifier对象，设置参数并初始化
    estimator = GradientBoostingClassifier(
        n_estimators=100,
        loss="log_loss",
        random_state=1,
        subsample=0.5,
        n_iter_no_change=5,
    )
    # 使用iris数据集进行训练
    estimator.fit(iris.data, iris.target)
    # 计算模型在训练数据上的准确率得分
    score = estimator.score(iris.data, iris.target)
    # 断言模型的OOB改善数组的长度小于估计器中设置的估计器数量
    assert estimator.oob_improvement_.shape[0] < estimator.n_estimators
    # 断言模型的OOB评分数组的长度小于估计器中设置的估计器数量
    assert estimator.oob_scores_.shape[0] < estimator.n_estimators
    # 断言模型的最后一个OOB评分与使用pytest.approx函数比较，与OOB得分相似
    assert estimator.oob_scores_[-1] == pytest.approx(estimator.oob_score_)

    # 固定的回归测试 - 如果OOB计算发生修改，则需要更改
    # FIXME: 下面的片段在32位系统上的结果不一致
    # assert_array_almost_equal(estimator.oob_improvement_[:5],
    #                           np.array([12.68, 10.45, 8.18, 6.43, 5.13]),
    #                           decimal=2)


def test_verbose_output():
    # Check verbose=1 does not cause error.
    import sys
    from io import StringIO

    # 保存标准输出流，并将其重定向到StringIO对象
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    # 创建一个GradientBoostingClassifier对象，设置参数并初始化，启用详细输出
    clf = GradientBoostingClassifier(
        n_estimators=100, random_state=1, verbose=1, subsample=0.8
    )
    # 使用X和y数据进行训练
    clf.fit(X, y)
    # 获取详细输出内容
    verbose_output = sys.stdout
    # 恢复原始的标准输出流
    sys.stdout = old_stdout

    # 检查输出内容是否符合预期
    verbose_output.seek(0)
    header = verbose_output.readline().rstrip()
    # 包含OOB情况的真实标题
    true_header = " ".join(["%10s"] + ["%16s"] * 3) % (
        "Iter",
        "Train Loss",
        "OOB Improve",
        "Remaining Time",
    )
    # 断言详细输出的标题与真实标题相同
    assert true_header == header

    # 统计详细输出中的行数
    n_lines = sum(1 for l in verbose_output.readlines())
    # 预期输出包含10行（1-10）和9行（20-100）
    assert 10 + 9 == n_lines


def test_more_verbose_output():
    # Check verbose=2 does not cause error.
    import sys
    from io import StringIO

    # 保存标准输出流，并将其重定向到StringIO对象
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    # 创建一个GradientBoostingClassifier对象，设置参数并初始化，启用更详细的输出
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1, verbose=2)
    # 使用X和y数据进行训练
    clf.fit(X, y)
    # 获取详细输出内容
    verbose_output = sys.stdout
    # 恢复原始的标准输出流
    sys.stdout = old_stdout

    # 检查输出内容是否符合预期
    verbose_output.seek(0)
    header = verbose_output.readline().rstrip()
    # 不包含OOB情况的真实标题
    true_header = " ".join(["%10s"] + ["%16s"] * 2) % (
        "Iter",
        "Train Loss",
        "Remaining Time",
    )
    # 断言详细输出的标题与真实标题相同
    assert true_header == header

    # 统计详细输出中的行数
    n_lines = sum(1 for l in verbose_output.readlines())
    # 预期输出应为100行，因为n_estimators==100
    assert 100 == n_lines


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start(Cls, global_random_seed):
    # This test checks the warm start behavior of the classifiers.
    # 这个测试用例检查分类器的热启动行为。
    # 创建一个数据集 X 和标签 y，使用 make_hastie_10_2 函数生成
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=global_random_seed)
    
    # 创建一个分类器 est，设定参数包括：200 个估计器，最大深度为 1，随机种子与全局随机种子相同
    est = Cls(n_estimators=200, max_depth=1, random_state=global_random_seed)
    
    # 使用数据集 X 和标签 y 对 est 进行拟合
    est.fit(X, y)

    # 创建一个具有热启动功能的分类器 est_ws，设定参数包括：100 个估计器，最大深度为 1，热启动为真，随机种子与全局随机种子相同
    est_ws = Cls(
        n_estimators=100, max_depth=1, warm_start=True, random_state=global_random_seed
    )
    
    # 使用数据集 X 和标签 y 对 est_ws 进行拟合
    est_ws.fit(X, y)
    
    # 设置 est_ws 的估计器数量为 200，继续使用数据集 X 和标签 y 进行拟合
    est_ws.set_params(n_estimators=200)
    est_ws.fit(X, y)

    # 如果分类器 Cls 是 GradientBoostingRegressor 类型，则断言 est_ws 预测结果与 est 相同
    if Cls is GradientBoostingRegressor:
        assert_allclose(est_ws.predict(X), est.predict(X))
    else:
        # 否则，断言 est_ws 的预测结果与 est 的预测结果相等
        # 由于随机状态被保留，因此 predict_proba 方法的结果也必须相同
        assert_array_equal(est_ws.predict(X), est.predict(X))
        assert_allclose(est_ws.predict_proba(X), est.predict_proba(X))
# 使用 pytest 的参数化装饰器，依次测试梯度提升算法的不同实现类
@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_n_estimators(Cls, global_random_seed):
    # 测试是否温暖启动等于拟合 - 设置 n_estimators
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=global_random_seed)
    # 初始化梯度提升算法对象，设置参数
    est = Cls(n_estimators=300, max_depth=1, random_state=global_random_seed)
    est.fit(X, y)

    # 使用温暖启动模式初始化另一个对象，设置不同的 n_estimators，并拟合数据
    est_ws = Cls(
        n_estimators=100, max_depth=1, warm_start=True, random_state=global_random_seed
    )
    est_ws.fit(X, y)
    # 修改 n_estimators 参数，并重新拟合数据
    est_ws.set_params(n_estimators=300)
    est_ws.fit(X, y)

    # 断言预测结果是否接近
    assert_allclose(est_ws.predict(X), est.predict(X))


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_max_depth(Cls):
    # 测试是否能在集成中拟合不同深度的树
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 初始化梯度提升算法对象，设置参数，并开启温暖启动模式
    est = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est.fit(X, y)
    # 修改参数 n_estimators 和 max_depth，并重新拟合数据
    est.set_params(n_estimators=110, max_depth=2)
    est.fit(X, y)

    # 断言最后 10 个树的深度是否为 2
    assert est.estimators_[0, 0].max_depth == 1
    for i in range(1, 11):
        assert est.estimators_[-i, 0].max_depth == 2


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_clear(Cls):
    # 测试拟合是否清除状态
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 初始化梯度提升算法对象，设置参数，并拟合数据
    est = Cls(n_estimators=100, max_depth=1)
    est.fit(X, y)

    # 使用温暖启动模式初始化另一个对象，并拟合数据
    est_2 = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est_2.fit(X, y)  # 初始化状态
    # 关闭温暖启动模式，并重新拟合数据，清除旧状态，结果应与 est 相同
    est_2.set_params(warm_start=False)
    est_2.fit(X, y)

    # 断言预测结果是否几乎相等
    assert_array_almost_equal(est_2.predict(X), est.predict(X))


@pytest.mark.parametrize("GradientBoosting", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_state_oob_scores(GradientBoosting):
    """
    检查在 `warm_start` 模式下是否清除了 OOB 分数的状态。
    """
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    n_estimators = 100
    # 初始化梯度提升算法对象，设置参数，并开启温暖启动模式
    estimator = GradientBoosting(
        n_estimators=n_estimators,
        max_depth=1,
        subsample=0.5,
        warm_start=True,
        random_state=1,
    )
    estimator.fit(X, y)
    oob_scores, oob_score = estimator.oob_scores_, estimator.oob_score_
    
    # 断言 OOB 分数的长度与 n_estimators 是否相等
    assert len(oob_scores) == n_estimators
    assert oob_scores[-1] == pytest.approx(oob_score)

    n_more_estimators = 200
    # 设置更多的 n_estimators，并重新拟合数据
    estimator.set_params(n_estimators=n_more_estimators).fit(X, y)
    assert len(estimator.oob_scores_) == n_more_estimators
    assert_allclose(estimator.oob_scores_[:n_estimators], oob_scores)

    # 关闭温暖启动模式，并重新拟合数据，断言状态是否已更新
    estimator.set_params(n_estimators=n_estimators, warm_start=False).fit(X, y)
    assert estimator.oob_scores_ is not oob_scores
    assert estimator.oob_score_ is not oob_score
    assert_allclose(estimator.oob_scores_, oob_scores)
    assert estimator.oob_score_ == pytest.approx(oob_score)
    assert oob_scores[-1] == pytest.approx(oob_score)
@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_smaller_n_estimators(Cls):
    # 测试在使用较小的 n_estimators 进行热启动时是否会引发错误
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 创建梯度提升模型实例，设置 n_estimators=100, max_depth=1，并启用热启动
    est = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est.fit(X, y)
    # 将模型的 n_estimators 参数设置为 99
    est.set_params(n_estimators=99)
    # 使用相同数据再次拟合模型，预期会引发 ValueError 异常
    with pytest.raises(ValueError):
        est.fit(X, y)


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_equal_n_estimators(Cls):
    # 测试在相等的 n_estimators 下进行热启动是否不会产生变化
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 创建梯度提升模型实例，设置 n_estimators=100, max_depth=1
    est = Cls(n_estimators=100, max_depth=1)
    est.fit(X, y)

    # 克隆第一个模型实例
    est2 = clone(est)
    # 设置第二个模型实例的参数，保持 n_estimators 相等，并启用热启动
    est2.set_params(n_estimators=est.n_estimators, warm_start=True)
    est2.fit(X, y)

    # 断言预测结果几乎相等
    assert_array_almost_equal(est2.predict(X), est.predict(X))


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_oob_switch(Cls):
    # 测试在热启动过程中是否可以打开 oob（袋外估计）功能
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 创建梯度提升模型实例，设置 n_estimators=100, max_depth=1，并启用热启动
    est = Cls(n_estimators=100, max_depth=1, warm_start=True)
    est.fit(X, y)
    # 调整模型参数，增加 n_estimators 和 subsample 参数
    est.set_params(n_estimators=110, subsample=0.5)
    est.fit(X, y)

    # 断言 oob_improvement_ 和 oob_scores_ 的前 100 个值为零
    assert_array_equal(est.oob_improvement_[:100], np.zeros(100))
    assert_array_equal(est.oob_scores_[:100], np.zeros(100))

    # 断言 oob_improvement_ 和 oob_scores_ 的最后 10 个值不为零
    assert (est.oob_improvement_[-10:] != 0.0).all()
    assert (est.oob_scores_[-10:] != 0.0).all()

    # 断言 oob_scores_ 的最后一个值与 oob_score_ 函数的值近似相等
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_oob(Cls):
    # 测试热启动时的 oob 是否等于普通拟合的结果
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 创建常规的梯度提升模型实例，设置 n_estimators=200, max_depth=1, subsample=0.5
    est = Cls(n_estimators=200, max_depth=1, subsample=0.5, random_state=1)
    est.fit(X, y)

    # 创建热启动的梯度提升模型实例，设置 n_estimators=100, max_depth=1, subsample=0.5，并启用热启动
    est_ws = Cls(
        n_estimators=100, max_depth=1, subsample=0.5, random_state=1, warm_start=True
    )
    est_ws.fit(X, y)
    # 设置模型的 n_estimators 参数为 200，继续拟合
    est_ws.set_params(n_estimators=200)
    est_ws.fit(X, y)

    # 断言前 100 个值的 oob_improvement_ 和 oob_scores_ 几乎相等
    assert_array_almost_equal(est_ws.oob_improvement_[:100], est.oob_improvement_[:100])
    assert_array_almost_equal(est_ws.oob_scores_[:100], est.oob_scores_[:100])
    # 断言普通模型的 oob_score_ 函数值与 est.oob_scores_ 的最后一个值近似相等
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)
    # 断言热启动模型的 oob_score_ 函数值与 est_ws.oob_scores_ 的最后一个值近似相等
    assert est_ws.oob_scores_[-1] == pytest.approx(est_ws.oob_score_)


@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
def test_warm_start_sparse(Cls, sparse_container):
    # 测试所有稀疏矩阵类型是否都被支持
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 创建稠密数据的梯度提升模型实例，设置 n_estimators=100, max_depth=1, subsample=0.5
    est_dense = Cls(
        n_estimators=100, max_depth=1, subsample=0.5, random_state=1, warm_start=True
    )
    est_dense.fit(X, y)
    est_dense.predict(X)
    # 将模型的 n_estimators 参数设置为 200，继续拟合
    est_dense.set_params(n_estimators=200)
    est_dense.fit(X, y)
    y_pred_dense = est_dense.predict(X)
    # 将稀疏矩阵 X 转换为稀疏容器（假设 sparse_container 是一个函数）
    X_sparse = sparse_container(X)

    # 使用指定参数初始化一个分类器实例 Cls
    est_sparse = Cls(
        n_estimators=100,
        max_depth=1,
        subsample=0.5,
        random_state=1,
        warm_start=True,
    )

    # 使用稀疏矩阵 X_sparse 和目标值 y 进行训练
    est_sparse.fit(X_sparse, y)

    # 使用训练好的 est_sparse 模型对输入数据 X 进行预测
    est_sparse.predict(X)

    # 修改 est_sparse 模型的 n_estimators 参数为 200
    est_sparse.set_params(n_estimators=200)

    # 使用更新后的 est_sparse 模型再次对 X_sparse 和 y 进行训练
    est_sparse.fit(X_sparse, y)

    # 使用训练好的 est_sparse 模型对输入数据 X 进行预测，并保存预测结果
    y_pred_sparse = est_sparse.predict(X)

    # 检查密集模型 est_dense 和稀疏模型 est_sparse 的前 100 个 out-of-bag 改进是否近似相等
    assert_array_almost_equal(
        est_dense.oob_improvement_[:100], est_sparse.oob_improvement_[:100]
    )

    # 检查密集模型 est_dense 的最后一个 out-of-bag 分数是否与其总体 out-of-bag 分数的近似相等
    assert est_dense.oob_scores_[-1] == pytest.approx(est_dense.oob_score_)

    # 检查密集模型 est_dense 和稀疏模型 est_sparse 的前 100 个 out-of-bag 分数是否近似相等
    assert_array_almost_equal(est_dense.oob_scores_[:100], est_sparse.oob_scores_[:100])

    # 检查稀疏模型 est_sparse 的最后一个 out-of-bag 分数是否与其总体 out-of-bag 分数的近似相等
    assert est_sparse.oob_scores_[-1] == pytest.approx(est_sparse.oob_score_)

    # 检查密集模型 est_dense 和稀疏模型 est_sparse 的预测结果是否近似相等
    assert_array_almost_equal(y_pred_dense, y_pred_sparse)
# 使用参数化测试对所有的梯度提升估计器执行单元测试
@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_warm_start_fortran(Cls, global_random_seed):
    # 测试以Fortran顺序提供X是否与以C顺序提供X给出相同结果
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=global_random_seed)
    # 创建C顺序的估计器实例
    est_c = Cls(n_estimators=1, random_state=global_random_seed, warm_start=True)
    # 创建Fortran顺序的估计器实例
    est_fortran = Cls(n_estimators=1, random_state=global_random_seed, warm_start=True)

    # 使用C顺序的X拟合估计器
    est_c.fit(X, y)
    # 修改估计器参数后再次使用C顺序的X拟合
    est_c.set_params(n_estimators=11)
    est_c.fit(X, y)

    # 将X转换为Fortran顺序
    X_fortran = np.asfortranarray(X)
    # 使用Fortran顺序的X拟合估计器
    est_fortran.fit(X_fortran, y)
    # 修改估计器参数后再次使用Fortran顺序的X拟合
    est_fortran.set_params(n_estimators=11)
    est_fortran.fit(X_fortran, y)

    # 断言两种顺序下的预测结果近似相等
    assert_allclose(est_c.predict(X), est_fortran.predict(X))


# 返回第10次迭代时为True的早停监视器函数
def early_stopping_monitor(i, est, locals):
    if i == 9:
        return True
    else:
        return False


# 使用参数化测试检验监视器返回值的工作情况
@pytest.mark.parametrize("Cls", GRADIENT_BOOSTING_ESTIMATORS)
def test_monitor_early_stopping(Cls):
    # 测试监视器返回值的有效性
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)

    # 创建梯度提升估计器实例，并设置最大深度和随机子样本
    est = Cls(n_estimators=20, max_depth=1, random_state=1, subsample=0.5)
    # 使用早停监视器拟合估计器
    est.fit(X, y, monitor=early_stopping_monitor)
    # 断言估计器的属性在早停止后保持不变
    assert est.n_estimators == 20
    assert est.estimators_.shape[0] == 10
    assert est.train_score_.shape[0] == 10
    assert est.oob_improvement_.shape[0] == 10
    assert est.oob_scores_.shape[0] == 10
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)

    # 尝试重新拟合估计器
    est.set_params(n_estimators=30)
    est.fit(X, y)
    assert est.n_estimators == 30
    assert est.estimators_.shape[0] == 30
    assert est.train_score_.shape[0] == 30
    assert est.oob_improvement_.shape[0] == 30
    assert est.oob_scores_.shape[0] == 30
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)

    # 使用启动热启动的梯度提升估计器实例
    est = Cls(
        n_estimators=20, max_depth=1, random_state=1, subsample=0.5, warm_start=True
    )
    # 使用早停监视器拟合估计器
    est.fit(X, y, monitor=early_stopping_monitor)
    assert est.n_estimators == 20
    assert est.estimators_.shape[0] == 10
    assert est.train_score_.shape[0] == 10
    assert est.oob_improvement_.shape[0] == 10
    assert est.oob_scores_.shape[0] == 10
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)

    # 尝试关闭热启动重新拟合估计器
    est.set_params(n_estimators=30, warm_start=False)
    est.fit(X, y)
    assert est.n_estimators == 30
    assert est.train_score_.shape[0] == 30
    assert est.estimators_.shape[0] == 30
    assert est.oob_improvement_.shape[0] == 30
    assert est.oob_scores_.shape[0] == 30
    assert est.oob_scores_[-1] == pytest.approx(est.oob_score_)


# 测试带有最大深度+1叶子节点的贪婪树完全分类
def test_complete_classification():
    from sklearn.tree._tree import TREE_LEAF

    # 创建用于完全分类测试的数据集
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    k = 4
    # 创建一个梯度提升分类器对象，设定参数如下：
    # - 使用 20 个基础决策树作为基础估计器
    # - 不设定树的最大深度
    # - 设定随机种子为 1
    # - 设定每棵树的最大叶子节点数为 k + 1
    est = GradientBoostingClassifier(
        n_estimators=20, max_depth=None, random_state=1, max_leaf_nodes=k + 1
    )
    # 使用训练数据 X 和标签 y 来训练梯度提升分类器
    est.fit(X, y)

    # 获取梯度提升分类器的第一棵基础决策树的决策树对象
    tree = est.estimators_[0, 0].tree_
    # 断言第一棵树的最大深度等于 k
    assert tree.max_depth == k
    # 断言第一棵树中叶子节点的数量等于 k + 1
    assert tree.children_left[tree.children_left == TREE_LEAF].shape[0] == k + 1
def test_complete_regression():
    # Test greedy trees with max_depth + 1 leafs.
    # 导入所需的模块和常量
    from sklearn.tree._tree import TREE_LEAF

    # 设定叶子节点数目
    k = 4

    # 创建梯度提升回归器对象
    est = GradientBoostingRegressor(
        n_estimators=20, max_depth=None, random_state=1, max_leaf_nodes=k + 1
    )
    # 使用回归数据拟合回归器
    est.fit(X_reg, y_reg)

    # 获取最后一个回归器的决策树对象
    tree = est.estimators_[-1, 0].tree_
    # 断言叶子节点数与预期的 k + 1 相等
    assert tree.children_left[tree.children_left == TREE_LEAF].shape[0] == k + 1


def test_zero_estimator_reg(global_random_seed):
    # Test if init='zero' works for regression by checking that it is better
    # than a simple baseline.

    # 创建均值虚拟回归器作为基准
    baseline = DummyRegressor(strategy="mean").fit(X_reg, y_reg)
    # 计算基准均方误差
    mse_baseline = mean_squared_error(baseline.predict(X_reg), y_reg)

    # 创建零初始化的梯度提升回归器
    est = GradientBoostingRegressor(
        n_estimators=5,
        max_depth=1,
        random_state=global_random_seed,
        init="zero",
        learning_rate=0.5,
    )
    # 使用回归数据拟合零初始化的回归器
    est.fit(X_reg, y_reg)
    # 计算梯度提升回归器的均方误差
    y_pred = est.predict(X_reg)
    mse_gbdt = mean_squared_error(y_reg, y_pred)
    # 断言梯度提升回归器的均方误差比基准好
    assert mse_gbdt < mse_baseline


def test_zero_estimator_clf(global_random_seed):
    # Test if init='zero' works for classification.
    # 获取鸢尾花数据集的特征和标签
    X = iris.data
    y = np.array(iris.target)

    # 创建零初始化的梯度提升分类器
    est = GradientBoostingClassifier(
        n_estimators=20, max_depth=1, random_state=global_random_seed, init="zero"
    )
    # 使用数据拟合分类器
    est.fit(X, y)

    # 断言分类器的预测准确率高于0.96
    assert est.score(X, y) > 0.96

    # 将多类问题转换为二元分类问题
    mask = y != 0
    y[mask] = 1
    y[~mask] = 0
    # 创建零初始化的梯度提升分类器
    est = GradientBoostingClassifier(
        n_estimators=20, max_depth=1, random_state=global_random_seed, init="zero"
    )
    # 使用数据拟合分类器
    est.fit(X, y)
    # 断言分类器的预测准确率高于0.96
    assert est.score(X, y) > 0.96


@pytest.mark.parametrize("GBEstimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_max_leaf_nodes_max_depth(GBEstimator):
    # Test precedence of max_leaf_nodes over max_depth.
    # 生成 10x2 的哈斯蒂数据集
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)

    # 设定叶子节点数目
    k = 4

    # 创建梯度提升估计器对象，测试 max_leaf_nodes 参数优先于 max_depth 参数
    est = GBEstimator(max_depth=1, max_leaf_nodes=k).fit(X, y)
    tree = est.estimators_[0, 0].tree_
    # 断言决策树的最大深度为 1
    assert tree.max_depth == 1

    # 创建梯度提升估计器对象，测试未指定 max_leaf_nodes 参数时的行为
    est = GBEstimator(max_depth=1).fit(X, y)
    tree = est.estimators_[0, 0].tree_
    # 断言决策树的最大深度为 1
    assert tree.max_depth == 1


@pytest.mark.parametrize("GBEstimator", GRADIENT_BOOSTING_ESTIMATORS)
def test_min_impurity_decrease(GBEstimator):
    # Test if min_impurity_decrease parameter is correctly passed on to trees.
    # 生成 10x2 的哈斯蒂数据集
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)

    # 创建梯度提升估计器对象，设置 min_impurity_decrease 参数
    est = GBEstimator(min_impurity_decrease=0.1)
    est.fit(X, y)
    for tree in est.estimators_.flat:
        # 检查参数是否正确传递给决策树对象
        assert tree.min_impurity_decrease == 0.1


def test_warm_start_wo_nestimators_change():
    # Test if warm_start does nothing if n_estimators is not changed.
    # Regression test for #3513.

    # 创建梯度提升分类器对象，启用 warm_start 功能
    clf = GradientBoostingClassifier(n_estimators=10, warm_start=True)
    # 使用数据拟合分类器
    clf.fit([[0, 1], [2, 3]], [0, 1])
    # 断言分类器的基础估计器数量为 10
    assert clf.estimators_.shape[0] == 10
    # 再次使用相同数据拟合分类器
    clf.fit([[0, 1], [2, 3]], [0, 1])
    # 断言分类器的基础估计器数量仍为 10
    assert clf.estimators_.shape[0] == 10
    # 一个包含损失函数名称和对应数值的元组列表
    ("loss", "value"),
    # 一个包含多个损失函数及其参数的列表
    [
        # 均方误差损失函数，参数为0.5
        ("squared_error", 0.5),
        # 绝对误差损失函数，参数为0.0
        ("absolute_error", 0.0),
        # Huber损失函数，参数为0.5
        ("huber", 0.5),
        # 分位数损失函数，参数为0.5
        ("quantile", 0.5),
    ],
# 定义了一个测试函数，用于测试非均匀权重的情况下的回归模型的边缘案例
def test_non_uniform_weights_toy_edge_case_reg(loss, value):
    # 输入特征矩阵 X，包括四个样本，每个样本有两个特征
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    # 标签 y，共四个样本
    y = [0, 0, 1, 0]
    # 设置样本权重，忽略前两个样本的训练，因此权重为 [0, 0, 1, 1]
    sample_weight = [0, 0, 1, 1]
    # 创建梯度提升回归器对象，学习率为 1.0，使用 2 个基础估算器，指定损失函数
    gb = GradientBoostingRegressor(learning_rate=1.0, n_estimators=2, loss=loss)
    # 使用样本 X, y 和权重 sample_weight 训练梯度提升回归器
    gb.fit(X, y, sample_weight=sample_weight)
    # 断言预测结果的第一个元素大于等于给定的 value
    assert gb.predict([[1, 0]])[0] >= value


# 定义了一个测试函数，用于测试非均匀权重的情况下的分类模型的边缘案例
def test_non_uniform_weights_toy_edge_case_clf():
    # 输入特征矩阵 X，包括四个样本，每个样本有两个特征
    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    # 标签 y，共四个样本
    y = [0, 0, 1, 0]
    # 设置样本权重，忽略前两个样本的训练，因此权重为 [0, 0, 1, 1]
    sample_weight = [0, 0, 1, 1]
    # 遍历损失函数列表 ["log_loss", "exponential"]
    for loss in ("log_loss", "exponential"):
        # 创建梯度提升分类器对象，使用 5 个基础估算器，指定损失函数
        gb = GradientBoostingClassifier(n_estimators=5, loss=loss)
        # 使用样本 X, y 和权重 sample_weight 训练梯度提升分类器
        gb.fit(X, y, sample_weight=sample_weight)
        # 断言预测结果为 [1]
        assert_array_equal(gb.predict([[1, 0]]), [1])


# 装饰器函数，用于标记需要在 32 位系统上跳过的测试
@skip_if_32bit
# 参数化测试函数，测试稀疏输入的情况
@pytest.mark.parametrize(
    # 参数化的估算器类，包括分类器和回归器
    "EstimatorClass", (GradientBoostingClassifier, GradientBoostingRegressor)
)
# 参数化测试函数，测试不同的稀疏矩阵容器
@pytest.mark.parametrize(
    # 稀疏容器的参数化列表，包括 COO、CSC 和 CSR 格式的稀疏矩阵
    "sparse_container", COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
# 测试函数，用于测试稀疏输入的梯度提升模型的行为
def test_sparse_input(EstimatorClass, sparse_container):
    # 生成多标签分类的人工数据集，随机种子为 0，样本数为 50，特征数为 1，类别数为 20
    y, X = datasets.make_multilabel_classification(
        random_state=0, n_samples=50, n_features=1, n_classes=20
    )
    # 只保留 y 的第一列作为标签
    y = y[:, 0]
    # 将输入 X 转换成稀疏矩阵形式
    X_sparse = sparse_container(X)

    # 创建深度为 2 的梯度提升估算器，最小不纯度减少为 1e-7 的密集版对象
    dense = EstimatorClass(
        n_estimators=10, random_state=0, max_depth=2, min_impurity_decrease=1e-7
    ).fit(X, y)
    # 创建深度为 2 的梯度提升估算器，最小不纯度减少为 1e-7 的稀疏版对象
    sparse = EstimatorClass(
        n_estimators=10, random_state=0, max_depth=2, min_impurity_decrease=1e-7
    ).fit(X_sparse, y)

    # 断言稠密和稀疏版的预测应用结果近似相等
    assert_array_almost_equal(sparse.apply(X), dense.apply(X))
    # 断言稠密和稀疏版的预测结果近似相等
    assert_array_almost_equal(sparse.predict(X), dense.predict(X))
    # 断言稠密和稀疏版的特征重要性近似相等
    assert_array_almost_equal(sparse.feature_importances_, dense.feature_importances_)

    # 断言稀疏版和稠密版输入的预测结果近似相等
    assert_array_almost_equal(sparse.predict(X_sparse), dense.predict(X))
    # 断言稠密版和稀疏版输入的预测结果近似相等
    assert_array_almost_equal(dense.predict(X_sparse), sparse.predict(X))

    # 如果估算器类是梯度提升分类器，进一步断言预测概率和对数预测概率近似相等
    if issubclass(EstimatorClass, GradientBoostingClassifier):
        assert_array_almost_equal(sparse.predict_proba(X), dense.predict_proba(X))
        assert_array_almost_equal(
            sparse.predict_log_proba(X), dense.predict_log_proba(X)
        )

        # 断言稀疏版和稠密版输入的决策函数近似相等
        assert_array_almost_equal(
            sparse.decision_function(X_sparse), sparse.decision_function(X)
        )
        # 断言稠密版和稀疏版输入的决策函数近似相等
        assert_array_almost_equal(
            dense.decision_function(X_sparse), sparse.decision_function(X)
        )
        # 断言稀疏版和稠密版输入的阶段性决策函数近似相等
        for res_sparse, res in zip(
            sparse.staged_decision_function(X_sparse),
            sparse.staged_decision_function(X),
        ):
            assert_array_almost_equal(res_sparse, res)


# 参数化测试函数，测试梯度提升估算器的早停功能
@pytest.mark.parametrize(
    # 参数化的梯度提升估算器类，包括分类器和回归器
    "GradientBoostingEstimator", [GradientBoostingClassifier, GradientBoostingRegressor]
)
# 测试函数，用于测试梯度提升模型的早停功能
def test_gradient_boosting_early_stopping(GradientBoostingEstimator):
    # 检查早停是否按预期工作，即通过经验检查当容忍度减少时训练的估算器数量是否增加
    # 使用 make_classification 生成具有1000个样本的分类数据集 X 和对应的标签 y
    X, y = make_classification(n_samples=1000, random_state=0)
    
    # 设置梯度提升算法中的基学习器数量为 1000
    n_estimators = 1000
    
    # 创建一个梯度提升估计器对象 gb_large_tol，设置参数如下：
    #   - 基学习器数量为 n_estimators
    #   - 在连续 n_iter_no_change 次迭代中，验证集分数没有改善则停止训练
    #   - 学习率为 0.1
    #   - 每棵树的最大深度为 3
    #   - 随机数种子为 42
    #   - 公差为 0.1
    gb_large_tol = GradientBoostingEstimator(
        n_estimators=n_estimators,
        n_iter_no_change=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        tol=1e-1,
    )
    
    # 创建另一个梯度提升估计器对象 gb_small_tol，设置参数如下：
    #   - 基学习器数量为 n_estimators
    #   - 在连续 n_iter_no_change 次迭代中，验证集分数没有改善则停止训练
    #   - 学习率为 0.1
    #   - 每棵树的最大深度为 3
    #   - 随机数种子为 42
    #   - 公差为 0.001
    gb_small_tol = GradientBoostingEstimator(
        n_estimators=n_estimators,
        n_iter_no_change=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        tol=1e-3,
    )
    
    # 将数据集 X 和 y 随机分割成训练集和测试集，随机种子为 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # 使用 gb_large_tol 拟合训练集数据
    gb_large_tol.fit(X_train, y_train)
    
    # 使用 gb_small_tol 拟合训练集数据
    gb_small_tol.fit(X_train, y_train)
    
    # 断言 gb_large_tol 的基学习器数量比 gb_small_tol 小且都小于 n_estimators
    assert gb_large_tol.n_estimators_ < gb_small_tol.n_estimators_ < n_estimators
    
    # 断言 gb_large_tol 在测试集上的预测准确率大于 0.7
    assert gb_large_tol.score(X_test, y_test) > 0.7
    
    # 断言 gb_small_tol 在测试集上的预测准确率大于 0.7
    assert gb_small_tol.score(X_test, y_test) > 0.7
def test_gradient_boosting_without_early_stopping():
    # When early stopping is not used, the number of trained estimators
    # must be the one specified.
    
    # 生成一个具有1000个样本的分类数据集，用于训练
    X, y = make_classification(n_samples=1000, random_state=0)

    # 创建一个梯度提升分类器对象，设定参数并训练模型
    gbc = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42
    )
    gbc.fit(X, y)

    # 创建一个梯度提升回归器对象，设定参数并训练模型
    gbr = GradientBoostingRegressor(
        n_estimators=30, learning_rate=0.1, max_depth=3, random_state=42
    )
    gbr.fit(X, y)

    # 断言所训练的估计器数目符合设定的值
    assert gbc.n_estimators_ == 50
    assert gbr.n_estimators_ == 30


def test_gradient_boosting_validation_fraction():
    # 生成一个具有1000个样本的分类数据集，用于训练
    X, y = make_classification(n_samples=1000, random_state=0)

    # 创建一个梯度提升分类器对象，设定参数，包括验证集比例，并生成克隆对象进行参数设置
    gbc = GradientBoostingClassifier(
        n_estimators=100,
        n_iter_no_change=10,
        validation_fraction=0.1,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    gbc2 = clone(gbc).set_params(validation_fraction=0.3)
    gbc3 = clone(gbc).set_params(n_iter_no_change=20)

    # 创建一个梯度提升回归器对象，设定参数，包括验证集比例，并生成克隆对象进行参数设置
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        n_iter_no_change=10,
        learning_rate=0.1,
        max_depth=3,
        validation_fraction=0.1,
        random_state=42,
    )
    gbr2 = clone(gbr).set_params(validation_fraction=0.3)
    gbr3 = clone(gbr).set_params(n_iter_no_change=20)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 检查验证集比例参数是否影响结果
    gbc.fit(X_train, y_train)
    gbc2.fit(X_train, y_train)
    assert gbc.n_estimators_ != gbc2.n_estimators_

    gbr.fit(X_train, y_train)
    gbr2.fit(X_train, y_train)
    assert gbr.n_estimators_ != gbr2.n_estimators_

    # 检查当 n_iter_no_change 参数增加时，n_estimators_ 是否单调增加
    gbc3.fit(X_train, y_train)
    gbr3.fit(X_train, y_train)
    assert gbr.n_estimators_ < gbr3.n_estimators_
    assert gbc.n_estimators_ < gbc3.n_estimators_


def test_early_stopping_stratified():
    # 确保早停策略的数据分割是分层的
    X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [0, 0, 0, 1]

    # 创建一个梯度提升分类器对象，设定早停参数，并使用断言检查是否引发预期的 ValueError
    gbc = GradientBoostingClassifier(n_iter_no_change=5)
    with pytest.raises(
        ValueError, match="The least populated class in y has only 1 member"
    ):
        gbc.fit(X, y)


def _make_multiclass():
    return make_classification(n_classes=3, n_clusters_per_class=1)


@pytest.mark.parametrize(
    "gb, dataset_maker, init_estimator",
    [
        (GradientBoostingClassifier, make_classification, DummyClassifier),
        (GradientBoostingClassifier, _make_multiclass, DummyClassifier),
        (GradientBoostingRegressor, make_regression, DummyRegressor),
    ],
    ids=["binary classification", "multiclass classification", "regression"],
)
def test_gradient_boosting_with_init(
    gb, dataset_maker, init_estimator, global_random_seed
):
    # 检查 GradientBoostingRegressor 在使用 sklearn 初始化时的工作情况
    # estimator.
    # Check that an error is raised if trying to fit with sample weight but
    # initial estimator does not support sample weight

    # 生成数据集 X 和标签 y
    X, y = dataset_maker()
    # 创建一个长度为 100 的随机样本权重数组
    sample_weight = np.random.RandomState(global_random_seed).rand(100)

    # 使用支持样本权重的初始估算器初始化 GradientBoostingRegressor
    init_est = init_estimator()
    gb(init=init_est).fit(X, y, sample_weight=sample_weight)

    # 使用不支持样本权重的包装器初始化 GradientBoostingRegressor
    init_est = NoSampleWeightWrapper(init_estimator())
    gb(init=init_est).fit(X, y)  # ok no sample weights
    # 使用 pytest 来断言在使用不支持样本权重的估算器时是否会引发 ValueError
    with pytest.raises(ValueError, match="estimator.*does not support sample weights"):
        gb(init=init_est).fit(X, y, sample_weight=sample_weight)
def test_gradient_boosting_with_init_pipeline():
    # Check that the init estimator can be a pipeline (see issue #13466)

    # 创建一个随机生成的回归数据集 X, y
    X, y = make_regression(random_state=0)
    # 使用线性回归作为初始估计器创建一个管道
    init = make_pipeline(LinearRegression())
    # 使用初始估计器初始化梯度提升回归模型
    gb = GradientBoostingRegressor(init=init)
    # 在数据集上拟合梯度提升回归模型
    gb.fit(X, y)  # pipeline without sample_weight works fine

    with pytest.raises(
        ValueError,
        match="The initial estimator Pipeline does not support sample weights",
    ):
        # 确保管道在使用样本权重时会引发 ValueError
        gb.fit(X, y, sample_weight=np.ones(X.shape[0]))

    # Passing sample_weight to a pipeline raises a ValueError. This test makes
    # sure we make the distinction between ValueError raised by a pipeline that
    # was passed sample_weight, and a InvalidParameterError raised by a regular
    # estimator whose input checking failed.
    invalid_nu = 1.5
    err_msg = (
        "The 'nu' parameter of NuSVR must be a float in the"
        f" range (0.0, 1.0]. Got {invalid_nu} instead."
    )
    with pytest.raises(InvalidParameterError, match=re.escape(err_msg)):
        # 确保 NuSVR 在支持样本权重时不会引发错误
        init = NuSVR(gamma="auto", nu=invalid_nu)
        gb = GradientBoostingRegressor(init=init)
        gb.fit(X, y, sample_weight=np.ones(X.shape[0]))


def test_early_stopping_n_classes():
    # when doing early stopping (_, , y_train, _ = train_test_split(X, y))
    # there might be classes in y that are missing in y_train. As the init
    # estimator will be trained on y_train, we need to raise an error if this
    # happens.

    # 创建一个有少数负类样本的分类问题数据集 X, y
    X = [[1]] * 10
    y = [0, 0] + [1] * 8  # only 2 negative class over 10 samples
    # 初始化梯度提升分类器，设置早停参数，验证集占比为 0.8
    gb = GradientBoostingClassifier(
        n_iter_no_change=5, random_state=0, validation_fraction=0.8
    )
    with pytest.raises(
        ValueError, match="The training data after the early stopping split"
    ):
        # 确保在早停时，如果训练集中缺少某些类别，会引发 ValueError
        gb.fit(X, y)

    # No error if we let training data be big enough
    gb = GradientBoostingClassifier(
        n_iter_no_change=5, random_state=0, validation_fraction=0.4
    )


def test_gbr_degenerate_feature_importances():
    # growing an ensemble of single node trees. See #13620
    # 创建一个全零特征矩阵 X 和全一目标向量 y
    X = np.zeros((10, 10))
    y = np.ones((10,))
    # 训练梯度提升回归模型
    gbr = GradientBoostingRegressor().fit(X, y)
    # 断言特征重要性全为零
    assert_array_equal(gbr.feature_importances_, np.zeros(10, dtype=np.float64))


def test_huber_vs_mean_and_median():
    """Check that huber lies between absolute and squared error."""
    n_rep = 100
    n_samples = 10
    y = np.tile(np.arange(n_samples), n_rep)
    x1 = np.minimum(y, n_samples / 2)
    x2 = np.minimum(-y, -n_samples / 2)
    X = np.c_[x1, x2]

    rng = np.random.RandomState(42)
    # We want an asymmetric distribution.
    y = y + rng.exponential(scale=1, size=y.shape)

    # 训练使用不同损失函数的梯度提升回归模型
    gbt_absolute_error = GradientBoostingRegressor(loss="absolute_error").fit(X, y)
    gbt_huber = GradientBoostingRegressor(loss="huber").fit(X, y)
    gbt_squared_error = GradientBoostingRegressor().fit(X, y)

    gbt_huber_predictions = gbt_huber.predict(X)
    # 断言：验证所有的 GBT 绝对误差预测结果都小于等于 GBT Huber 预测结果
    assert np.all(gbt_absolute_error.predict(X) <= gbt_huber_predictions)
    
    # 断言：验证所有的 GBT Huber 预测结果都小于等于 GBT 平方误差预测结果
    assert np.all(gbt_huber_predictions <= gbt_squared_error.predict(X))
# 定义一个测试函数，用于测试 _safe_divide 函数处理除以零的情况
def test_safe_divide():
    """Test that _safe_divide handles division by zero."""
    # 捕获警告，并将所有警告转换为异常
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 断言调用 _safe_divide 函数，分别对 np.float64(1e300) 和 0 进行除法，期望触发异常
        assert _safe_divide(np.float64(1e300), 0) == 0
        # 断言调用 _safe_divide 函数，对 np.float64(0.0) 和 np.float64(0.0) 进行除法，期望结果为 0
        assert _safe_divide(np.float64(0.0), np.float64(0.0)) == 0
    # 再次捕获警告，并期望捕获 RuntimeWarning 类型的警告，且匹配字符串 "overflow"
    with pytest.warns(RuntimeWarning, match="overflow"):
        # 调用 _safe_divide 函数，对 np.float64(1e300) 和 1e-10 进行除法
        _safe_divide(np.float64(1e300), 1e-10)


# 定义一个测试函数，用于测试 squared error GBT 在简单数据集上的向后兼容性
def test_squared_error_exact_backward_compat():
    """Test squared error GBT backward compat on a simple dataset.

    The results to compare against are taken from scikit-learn v1.2.0.
    """
    # 设置样本数量
    n_samples = 10
    # 创建目标变量 y，为长度为 n_samples 的数组，元素值从 0 到 n_samples-1
    y = np.arange(n_samples)
    # 创建特征变量 x1，为 y 和 n_samples / 2 之间较小的值组成的数组
    x1 = np.minimum(y, n_samples / 2)
    # 创建特征变量 x2，为 -y 和 -n_samples / 2 之间较小的值组成的数组
    x2 = np.minimum(-y, -n_samples / 2)
    # 合并 x1 和 x2 成为特征矩阵 X，使用 np.c_ 进行列合并
    X = np.c_[x1, x2]
    # 使用 squared_error 损失函数和 100 个估计器构建梯度提升回归模型 gbt，并拟合数据
    gbt = GradientBoostingRegressor(loss="squared_error", n_estimators=100).fit(X, y)

    # 创建预测结果 pred_result 数组
    pred_result = np.array(
        [
            1.39245726e-04,
            1.00010468e00,
            2.00007043e00,
            3.00004051e00,
            4.00000802e00,
            4.99998972e00,
            5.99996312e00,
            6.99993395e00,
            7.99989372e00,
            8.99985660e00,
        ]
    )
    # 使用 assert_allclose 断言 gbt 模型对 X 的预测结果与 pred_result 数组在给定的相对和绝对误差容差范围内相等
    assert_allclose(gbt.predict(X), pred_result, rtol=1e-8)

    # 创建训练分数 train_score 数组
    train_score = np.array(
        [
            4.87246390e-08,
            3.95590036e-08,
            3.21267865e-08,
            2.60970300e-08,
            2.11820178e-08,
            1.71995782e-08,
            1.39695549e-08,
            1.13391770e-08,
            9.19931587e-09,
            7.47000575e-09,
        ]
    )
    # 使用 assert_allclose 断言 gbt 模型的训练分数后 10 项与 train_score 数组在给定的相对误差和绝对误差容差范围内相等
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-8)

    # 使用样本权重 sample_weights，再次构建 squared_error 损失函数和 100 个估计器的梯度提升回归模型 gbt，并拟合数据
    sample_weights = np.tile([1, 10], n_samples // 2)
    gbt = GradientBoostingRegressor(loss="squared_error", n_estimators=100).fit(
        X, y, sample_weight=sample_weights
    )

    # 重新定义预测结果 pred_result 数组
    pred_result = np.array(
        [
            1.52391462e-04,
            1.00011168e00,
            2.00007724e00,
            3.00004638e00,
            4.00001302e00,
            4.99999873e00,
            5.99997093e00,
            6.99994329e00,
            7.99991290e00,
            8.99988727e00,
        ]
    )
    # 使用 assert_allclose 断言 gbt 模型对 X 的预测结果与 pred_result 数组在给定的相对和绝对误差容差范围内相等
    assert_allclose(gbt.predict(X), pred_result, rtol=1e-6, atol=1e-5)

    # 重新定义训练分数 train_score 数组
    train_score = np.array(
        [
            4.12445296e-08,
            3.34418322e-08,
            2.71151383e-08,
            2.19782469e-08,
            1.78173649e-08,
            1.44461976e-08,
            1.17120123e-08,
            9.49485678e-09,
            7.69772505e-09,
            6.24155316e-09,
        ]
    )
    # 使用 assert_allclose 断言 gbt 模型的训练分数后 10 项与 train_score 数组在给定的相对误差和绝对误差容差范围内相等
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-3, atol=1e-11)


# 根据条件装饰器 skip_if_32bit，定义一个测试函数，用于测试 huber GBT 在简单数据集上的向后兼容性
def test_huber_exact_backward_compat():
    """Test huber GBT backward compat on a simple dataset.

    The results to compare against are taken from scikit-learn v1.2.0.
    """
    # 设置样本数量
    n_samples = 10
    # 创建目标变量 y，为长度为 n_samples 的数组，元素值从 0 到 n_samples-1
    y = np.arange(n_samples)
    # 创建特征变量 x1，为 y 和 n_samples / 2 之间较小的值组成的数组
    x1 = np.minimum(y, n_samples / 2)
    # 使用 NumPy 的最小函数计算数组 -y 和 -n_samples / 2 元素对应位置的最小值，并存储在 x2 中
    x2 = np.minimum(-y, -n_samples / 2)
    
    # 使用 NumPy 的列连接函数 np.c_ 将 x1 和 x2 组合成一个特征矩阵 X
    X = np.c_[x1, x2]
    
    # 使用梯度提升回归器 GradientBoostingRegressor，使用 Huber 损失函数，设置为 100 个估计器，alpha 为 0.8，拟合数据 X 和目标 y
    gbt = GradientBoostingRegressor(loss="huber", n_estimators=100, alpha=0.8).fit(X, y)
    
    # 使用 assert_allclose 函数验证 gbt._loss.closs.delta 的值与预期值 0.0001655688041282133 的接近程度
    assert_allclose(gbt._loss.closs.delta, 0.0001655688041282133)
    
    # 创建预期的预测结果数组 pred_result，并使用 assert_allclose 函数验证 gbt.predict(X) 的输出与 pred_result 的接近程度，相对容差设为 1e-8
    pred_result = np.array(
        [
            1.48120765e-04,
            9.99949174e-01,
            2.00116957e00,
            2.99986716e00,
            4.00012064e00,
            5.00002462e00,
            5.99998898e00,
            6.99692549e00,
            8.00006356e00,
            8.99985099e00,
        ]
    )
    assert_allclose(gbt.predict(X), pred_result, rtol=1e-8)
    
    # 创建预期的训练得分数组 train_score，并使用 assert_allclose 函数验证 gbt.train_score_ 的最后十个元素与 train_score 的接近程度，相对容差设为 1e-8
    train_score = np.array(
        [
            2.59484709e-07,
            2.19165900e-07,
            1.89644782e-07,
            1.64556454e-07,
            1.38705110e-07,
            1.20373736e-07,
            1.04746082e-07,
            9.13835687e-08,
            8.20245756e-08,
            7.17122188e-08,
        ]
    )
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-8)
def test_binomial_error_exact_backward_compat():
    """Test binary log_loss GBT backward compat on a simple dataset.

    The results to compare against are taken from scikit-learn v1.2.0.
    """
    # 创建包含 10 个样本的一维数组 y，值为 [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    n_samples = 10
    y = np.arange(n_samples) % 2
    # 创建 x1 和 x2，分别为 [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5] 和 [0, -5.0, 0, -5.0, 0, -5.0, 0, -5.0, 0, -5.0]
    x1 = np.minimum(y, n_samples / 2)
    x2 = np.minimum(-y, -n_samples / 2)
    # 将 x1 和 x2 合并成一个二维数组 X
    X = np.c_[x1, x2]
    # 使用 GradientBoostingClassifier 对象 gbt，基于 log_loss 损失函数和 100 个估计器，拟合数据 X, y
    gbt = GradientBoostingClassifier(loss="log_loss", n_estimators=100).fit(X, y)

    # 预测结果数组 pred_result，与预期结果比较，rtol 设置为 1e-8
    pred_result = np.array(
        [
            [9.99978098e-01, 2.19017313e-05],
            [2.19017313e-05, 9.99978098e-01],
            [9.99978098e-01, 2.19017313e-05],
            [2.19017313e-05, 9.99978098e-01],
            [9.99978098e-01, 2.19017313e-05],
            [2.19017313e-05, 9.99978098e-01],
            [9.99978098e-01, 2.19017313e-05],
            [2.19017313e-05, 9.99978098e-01],
            [9.99978098e-01, 2.19017313e-05],
            [2.19017313e-05, 9.99978098e-01],
        ]
    )
    # 使用 assert_allclose 函数比较 gbt 预测的概率值与预期的 pred_result，rtol 设置为 1e-8
    assert_allclose(gbt.predict_proba(X), pred_result, rtol=1e-8)

    # 训练得分数组 train_score，与预期结果比较，rtol 设置为 1e-8
    train_score = np.array(
        [
            1.07742210e-04,
            9.74889078e-05,
            8.82113863e-05,
            7.98167784e-05,
            7.22210566e-05,
            6.53481907e-05,
            5.91293869e-05,
            5.35023988e-05,
            4.84109045e-05,
            4.38039423e-05,
        ]
    )
    # 使用 assert_allclose 函数比较 gbt 的训练得分数组与预期的 train_score，rtol 设置为 1e-8
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-8)


def test_multinomial_error_exact_backward_compat():
    """Test multiclass log_loss GBT backward compat on a simple dataset.

    The results to compare against are taken from scikit-learn v1.2.0.
    """
    # 创建包含 10 个样本的一维数组 y，值为 [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    n_samples = 10
    y = np.arange(n_samples) % 4
    # 创建 x1 和 x2，分别为 [0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5] 和 [0, -5.0, 0, -5.0, 0, -5.0, 0, -5.0, 0, -5.0]
    x1 = np.minimum(y, n_samples / 2)
    x2 = np.minimum(-y, -n_samples / 2)
    # 将 x1 和 x2 合并成一个二维数组 X
    X = np.c_[x1, x2]
    # 使用 GradientBoostingClassifier 对象 gbt，基于 log_loss 损失函数和 100 个估计器，拟合数据 X, y
    gbt = GradientBoostingClassifier(loss="log_loss", n_estimators=100).fit(X, y)

    # 预测结果数组 pred_result，与预期结果比较，rtol 设置为 1e-8
    pred_result = np.array(
        [
            [9.99999727e-01, 1.11956255e-07, 8.04921671e-08, 8.04921668e-08],
            [1.11956254e-07, 9.99999727e-01, 8.04921671e-08, 8.04921668e-08],
            [1.19417637e-07, 1.19417637e-07, 9.99999675e-01, 8.60526098e-08],
            [1.19417637e-07, 1.19417637e-07, 8.60526088e-08, 9.99999675e-01],
            [9.99999727e-01, 1.11956255e-07, 8.04921671e-08, 8.04921668e-08],
            [1.11956254e-07, 9.99999727e-01, 8.04921671e-08, 8.04921668e-08],
            [1.19417637e-07, 1.19417637e-07, 9.99999675e-01, 8.60526098e-08],
            [1.19417637e-07, 1.19417637e-07, 8.60526088e-08, 9.99999675e-01],
            [9.99999727e-01, 1.11956255e-07, 8.04921671e-08, 8.04921668e-08],
            [1.11956254e-07, 9.99999727e-01, 8.04921671e-08, 8.04921668e-08],
        ]
    )
    # 使用 assert_allclose 函数比较 gbt 预测的概率值与预期的 pred_result，rtol 设置为 1e-8
    assert_allclose(gbt.predict_proba(X), pred_result, rtol=1e-8)
    # 定义一个 NumPy 数组，存储了模型训练过程中最后 10 轮的训练得分数据
    train_score = np.array(
        [
            1.13300150e-06,
            9.75183397e-07,
            8.39348103e-07,
            7.22433588e-07,
            6.21804338e-07,
            5.35191943e-07,
            4.60643966e-07,
            3.96479930e-07,
            3.41253434e-07,
            2.93719550e-07,
        ]
    )
    # 使用 NumPy 的 assert_allclose 函数来比较模型最后 10 轮的训练得分和预期的训练得分 train_score
    # rtol=1e-8 表示相对误差的阈值，用于确定两个数组之间的近似相等性
    assert_allclose(gbt.train_score_[-10:], train_score, rtol=1e-8)
# 定义一个测试函数，用于验证 _update_terminal_regions 函数的分母不为零。

"""Test _update_terminal_regions denominator is not zero.
For instance for log loss based binary classification, the line search step might
become nan/inf as denominator = hessian = prob * (1 - prob) and prob = 0 or 1 can
happen.
Here, we create a situation were this happens (at least with roughly 80%) based
on the random seed.
"""
# 使用 make_hastie_10_2 函数生成一个包含 100 个样本的数据集 X 和对应的标签 y
X, y = datasets.make_hastie_10_2(n_samples=100, random_state=20)

# 定义一个参数字典 params，包括梯度提升树分类器的各种参数设置
params = {
    "learning_rate": 1.0,
    "subsample": 0.5,
    "n_estimators": 100,
    "max_leaf_nodes": 4,
    "max_depth": None,
    "random_state": global_random_seed,
    "min_samples_leaf": 2,
}

# 使用参数字典 params 创建一个梯度提升树分类器 clf
clf = GradientBoostingClassifier(**params)

# 使用带有警告处理器的上下文管理器，捕获运行时警告，避免其打印到控制台
with warnings.catch_warnings():
    warnings.simplefilter("error")
    # 使用训练集 X 和标签 y 对分类器进行训练
    clf.fit(X, y)
```