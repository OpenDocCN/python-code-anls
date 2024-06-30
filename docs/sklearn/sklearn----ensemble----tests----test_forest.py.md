# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_forest.py`

```
"""
Testing for the forest module (sklearn.ensemble.forest).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 导入所需的库和模块
import itertools  # 提供迭代工具的函数
import math  # 数学函数库
import pickle  # 序列化和反序列化 Python 对象
from collections import defaultdict  # 默认字典实现
from functools import partial  # 创建 partial function
from itertools import combinations, product  # 组合和笛卡尔积生成函数
from typing import Any, Dict  # 类型提示相关
from unittest.mock import patch  # 用于模拟对象的测试库

import joblib  # 用于并行执行的库
import numpy as np  # 数组操作库
import pytest  # Python 测试工具
from scipy.special import comb  # 组合数函数

import sklearn  # scikit-learn 机器学习库
from sklearn import clone, datasets  # 数据集和模型克隆函数
from sklearn.datasets import make_classification, make_hastie_10_2  # 创建分类和回归数据集的函数
from sklearn.decomposition import TruncatedSVD  # 截断奇异值分解
from sklearn.dummy import DummyRegressor  # 用于基准评估的虚拟回归器
from sklearn.ensemble import (  # 集成学习算法
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RandomTreesEmbedding,
)
from sklearn.ensemble._forest import (  # 内部森林算法函数
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
)
from sklearn.exceptions import NotFittedError  # 未拟合模型异常
from sklearn.metrics import (  # 性能评估指标函数
    explained_variance_score,
    f1_score,
    mean_poisson_deviance,
    mean_squared_error,
)
from sklearn.model_selection import (  # 模型选择和评估工具
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import LinearSVC  # 线性支持向量分类器
from sklearn.tree._classes import SPARSE_SPLITTERS  # 稀疏数据分割器
from sklearn.utils._testing import (  # 测试工具函数
    _convert_container,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
    skip_if_no_parallel,
)
from sklearn.utils.fixes import (  # 兼容性修复函数
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
)
from sklearn.utils.multiclass import type_of_target  # 多类别标签类型推断函数
from sklearn.utils.parallel import Parallel  # 并行执行工具
from sklearn.utils.validation import check_random_state  # 随机状态检查函数

# toy sample
# 简单的示例数据集
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# Larger classification sample used for testing feature importances
# 更大的分类数据集，用于测试特征重要性
X_large, y_large = datasets.make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    shuffle=False,
    random_state=0,
)

# also load the iris dataset
# and randomly permute it
# 加载鸢尾花数据集，并随机排列
iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# Make regression dataset
# 创建回归数据集
X_reg, y_reg = datasets.make_regression(n_samples=500, n_features=10, random_state=1)

# also make a hastie_10_2 dataset
# 创建 hastie_10_2 数据集
hastie_X, hastie_y = datasets.make_hastie_10_2(n_samples=20, random_state=1)
hastie_X = hastie_X.astype(np.float32)

# Get the default backend in joblib to test parallelism and interaction with
# different backends
# 获取 joblib 的默认后端以测试并行性和与不同后端的交互
DEFAULT_JOBLIB_BACKEND = joblib.parallel.get_active_backend()[0].__class__

FOREST_CLASSIFIERS = {
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}

FOREST_REGRESSORS = {
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "RandomForestRegressor": RandomForestRegressor,


注释：


# 将字符串 "RandomForestRegressor" 映射到 RandomForestRegressor 类


这行代码将字符串 "RandomForestRegressor" 关联到 `RandomForestRegressor` 类。在 Python 中，这种映射通常用于配置或注册机制，允许通过字符串动态引用类或函数。
}

# 定义一个字典，包含了名为 "RandomTreesEmbedding" 的键，对应的值是 RandomTreesEmbedding 类
FOREST_TRANSFORMERS = {
    "RandomTreesEmbedding": RandomTreesEmbedding,
}

# 定义一个空的字典 FOREST_ESTIMATORS，键和值的类型为任意类型
FOREST_ESTIMATORS: Dict[str, Any] = dict()
# 将 FOREST_CLASSIFIERS 的内容更新到 FOREST_ESTIMATORS 中
FOREST_ESTIMATORS.update(FOREST_CLASSIFIERS)
# 将 FOREST_REGRESSORS 的内容更新到 FOREST_ESTIMATORS 中
FOREST_ESTIMATORS.update(FOREST_REGRESSORS)
# 将 FOREST_TRANSFORMERS 的内容更新到 FOREST_ESTIMATORS 中
FOREST_ESTIMATORS.update(FOREST_TRANSFORMERS)

# 定义一个字典 FOREST_CLASSIFIERS_REGRESSORS，初始值等于 FOREST_CLASSIFIERS 的副本
FOREST_CLASSIFIERS_REGRESSORS: Dict[str, Any] = FOREST_CLASSIFIERS.copy()
# 将 FOREST_REGRESSORS 的内容更新到 FOREST_CLASSIFIERS_REGRESSORS 中
FOREST_CLASSIFIERS_REGRESSORS.update(FOREST_REGRESSORS)


# 使用 pytest 框架的 parametrize 装饰器，传入 FOREST_CLASSIFIERS 中的每个分类器名称作为参数
def test_classification_toy(name):
    """Check classification on a toy dataset."""
    # 根据分类器名称获取对应的分类器类
    ForestClassifier = FOREST_CLASSIFIERS[name]

    # 初始化一个分类器对象 clf，设定参数并拟合数据
    clf = ForestClassifier(n_estimators=10, random_state=1)
    clf.fit(X, y)
    # 断言预测结果与真实结果数组相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言分类器的估计器数量为 10
    assert 10 == len(clf)

    # 使用不同的参数初始化另一个分类器对象 clf，设定参数并拟合数据
    clf = ForestClassifier(n_estimators=10, max_features=1, random_state=1)
    clf.fit(X, y)
    # 断言预测结果与真实结果数组相等
    assert_array_equal(clf.predict(T), true_result)
    # 断言分类器的估计器数量为 10
    assert 10 == len(clf)

    # 进行 apply 方法的测试
    leaf_indices = clf.apply(X)
    # 断言返回的叶子节点索引数组形状为 (样本数, 估计器数量)
    assert leaf_indices.shape == (len(X), clf.n_estimators)


# 使用 pytest 框架的 parametrize 装饰器，传入 FOREST_CLASSIFIERS 中的每个分类器名称和两种不同的准则（criterion）作为参数
def test_iris_criterion(name, criterion):
    # Check consistency on dataset iris.
    # 根据分类器名称获取对应的分类器类
    ForestClassifier = FOREST_CLASSIFIERS[name]

    # 初始化一个分类器对象 clf，设定参数并拟合 iris 数据集
    clf = ForestClassifier(n_estimators=10, criterion=criterion, random_state=1)
    clf.fit(iris.data, iris.target)
    # 计算并记录分类器的得分
    score = clf.score(iris.data, iris.target)
    # 断言得分大于 0.9
    assert score > 0.9, "Failed with criterion %s and score = %f" % (criterion, score)

    # 使用不同的参数初始化另一个分类器对象 clf，设定参数并拟合 iris 数据集
    clf = ForestClassifier(
        n_estimators=10, criterion=criterion, max_features=2, random_state=1
    )
    clf.fit(iris.data, iris.target)
    # 计算并记录分类器的得分
    score = clf.score(iris.data, iris.target)
    # 断言得分大于 0.5
    assert score > 0.5, "Failed with criterion %s and score = %f" % (criterion, score)


# 使用 pytest 框架的 parametrize 装饰器，传入 FOREST_REGRESSORS 中的每个回归器名称和三种不同的准则（criterion）作为参数
def test_regression_criterion(name, criterion):
    # Check consistency on regression dataset.
    # 根据回归器名称获取对应的回归器类
    ForestRegressor = FOREST_REGRESSORS[name]

    # 初始化一个回归器对象 reg，设定参数并拟合 X_reg, y_reg 数据集
    reg = ForestRegressor(n_estimators=5, criterion=criterion, random_state=1)
    reg.fit(X_reg, y_reg)
    # 计算并记录回归器的得分
    score = reg.score(X_reg, y_reg)
    # 断言得分大于 0.93
    assert (
        score > 0.93
    ), "Failed with max_features=None, criterion %s and score = %f" % (
        criterion,
        score,
    )

    # 使用不同的参数初始化另一个回归器对象 reg，设定参数并拟合 X_reg, y_reg 数据集
    reg = ForestRegressor(
        n_estimators=5, criterion=criterion, max_features=6, random_state=1
    )
    reg.fit(X_reg, y_reg)
    # 计算并记录回归器的得分
    score = reg.score(X_reg, y_reg)
    # 断言得分大于 0.92
    assert score > 0.92, "Failed with max_features=6, criterion %s and score = %f" % (
        criterion,
        score,
    )


# 测试随机森林在泊松准则下的表现优于 MSE 准则，用于泊松目标的情况
def test_poisson_vs_mse():
    """Test that random forest with poisson criterion performs better than
    mse for a poisson target.

    There is a similar test for DecisionTreeRegressor.
    """
    # 初始化一个随机数生成器 rng
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 500, 500, 10
    # 使用 make_low_rank_matrix 函数生成低秩矩阵 X，其样本数为 n_train + n_test，特征数为 n_features，随机数种子为 rng
    X = datasets.make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )
    # 创建一个对数线性泊松模型，并对 coef 进行缩放，因为在模型中它将被指数化
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    # 使用泊松分布生成 y 值，参数 lambda 为 np.exp(X @ coef)
    y = rng.poisson(lam=np.exp(X @ coef))
    # 将数据集 X 和 y 拆分为训练集和测试集，测试集大小为 n_test，随机数种子为 rng
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=rng
    )
    # 通过设置 min_samples_split=10 防止过拟合
    forest_poi = RandomForestRegressor(
        criterion="poisson", min_samples_leaf=10, max_features="sqrt", random_state=rng
    )
    forest_mse = RandomForestRegressor(
        criterion="squared_error",
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=rng,
    )

    # 使用训练集训练泊松准则的随机森林模型和均方误差准则的随机森林模型
    forest_poi.fit(X_train, y_train)
    forest_mse.fit(X_train, y_train)
    # 使用 DummyRegressor 拟合训练集，策略为"mean"
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)

    # 对于训练集和测试集，计算泊松模型的平均泊松偏差（metric_poi）、均方误差模型的平均泊松偏差（metric_mse）以及 Dummy 模型的平均泊松偏差（metric_dummy）
    for X, y, data_name in [(X_train, y_train, "train"), (X_test, y_test, "test")]:
        metric_poi = mean_poisson_deviance(y, forest_poi.predict(X))
        # 由于 squared_error 模型可能产生非正预测值，这里进行裁剪
        # 如果对应的 y 为 0，则泊松偏差会过于优秀
        # 如果我们抽取更多样本，最终会得到 y > 0，泊松偏差将变得不稳定
        # 因此，我们不裁剪到类似 1e-15 这样微小的值，而是裁剪到 1e-6，这相当于对非正预测施加一个小惩罚
        metric_mse = mean_poisson_deviance(
            y, np.clip(forest_mse.predict(X), 1e-6, None)
        )
        metric_dummy = mean_poisson_deviance(y, dummy.predict(X))
        # 对于测试集，验证泊松模型的指标是否小于均方误差模型的指标
        if data_name == "test":
            assert metric_poi < metric_mse
        # 验证泊松模型的指标是否小于 Dummy 模型指标的80%
        assert metric_poi < 0.8 * metric_dummy
# 使用 pytest 的 mark.parametrize 装饰器来多次运行此测试函数，参数为 "poisson" 和 "squared_error"
@pytest.mark.parametrize("criterion", ("poisson", "squared_error"))
def test_balance_property_random_forest(criterion):
    """ 
    Test that sum(y_pred)==sum(y_true) on the training set.
    """
    # 设置随机数种子为 42
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 500, 500, 10
    # 生成低秩矩阵 X，用于训练和测试，随机种子为 rng
    X = datasets.make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )

    # 生成随机系数，用于创建泊松分布的 y 值
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))

    # 创建随机森林回归器对象
    reg = RandomForestRegressor(
        criterion=criterion, n_estimators=10, bootstrap=False, random_state=rng
    )
    # 使用 X, y 进行拟合
    reg.fit(X, y)

    # 断言预测值的总和等于真实值的总和，使用 pytest 的 approx 函数进行近似匹配
    assert np.sum(reg.predict(X)) == pytest.approx(np.sum(y))


@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_regressor_attributes(name):
    # 回归模型不应该有 classes_ 属性。
    r = FOREST_REGRESSORS[name](random_state=0)
    assert not hasattr(r, "classes_")
    assert not hasattr(r, "n_classes_")

    r.fit([[1, 2, 3], [4, 5, 6]], [1, 2])
    assert not hasattr(r, "classes_")
    assert not hasattr(r, "n_classes_")


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_probability(name):
    # 预测概率值。
    ForestClassifier = FOREST_CLASSIFIERS[name]
    # 忽略除法错误，创建分类器对象 clf
    with np.errstate(divide="ignore"):
        clf = ForestClassifier(
            n_estimators=10, random_state=1, max_features=1, max_depth=1
        )
        # 使用 iris 数据集进行拟合
        clf.fit(iris.data, iris.target)
        # 断言预测的概率之和应该等于每行的总和为 1
        assert_array_almost_equal(
            np.sum(clf.predict_proba(iris.data), axis=1), np.ones(iris.data.shape[0])
        )
        # 断言预测的概率值与对数概率的指数应该近似相等
        assert_array_almost_equal(
            clf.predict_proba(iris.data), np.exp(clf.predict_log_proba(iris.data))
        )


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize(
    "name, criterion",
    itertools.chain(
        product(FOREST_CLASSIFIERS, ["gini", "log_loss"]),
        product(FOREST_REGRESSORS, ["squared_error", "friedman_mse", "absolute_error"]),
    ),
)
def test_importances(dtype, name, criterion):
    tolerance = 0.01
    if name in FOREST_REGRESSORS and criterion == "absolute_error":
        tolerance = 0.05

    # 将 X_large 和 y_large 转换为指定的数据类型 dtype
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)

    # 根据名称和标准创建随机森林估计器对象
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=10, criterion=criterion, random_state=0)
    # 使用 X, y 进行拟合
    est.fit(X, y)
    importances = est.feature_importances_

    # 随机森林估计器可以检测数据集中重要的特征数量
    n_important = np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    assert n_important == 3
    assert np.all(importances[:3] > 0.1)

    # 使用并行计算检查特征重要性
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)

    # 使用样本权重检查
    # 使用随机状态 0 来生成与 X 长度相同的随机权重数组
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    
    # 创建一个包含 10 个估算器的随机森林估算器对象，设置随机状态为 0，使用指定的分裂准则
    est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
    
    # 使用给定的权重数组来拟合随机森林模型
    est.fit(X, y, sample_weight=sample_weight)
    
    # 获取特征重要性指标
    importances = est.feature_importances_
    
    # 断言所有特征重要性指标均大于等于 0
    assert np.all(importances >= 0.0)

    # 针对不同的比例进行权重缩放，遍历 [0.5, 100] 这两个比例
    for scale in [0.5, 100]:
        # 创建一个包含 10 个估算器的随机森林估算器对象，设置随机状态为 0，使用指定的分裂准则
        est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
        
        # 使用缩放后的权重数组来拟合随机森林模型
        est.fit(X, y, sample_weight=scale * sample_weight)
        
        # 获取缩放后的特征重要性指标
        importances_bis = est.feature_importances_
        
        # 断言两次计算的特征重要性指标的平均差小于给定的容差 tolerance
        assert np.abs(importances - importances_bis).mean() < tolerance
# 定义一个测试函数，用于验证完全随机树的变量重要性是否收敛到其理论值。
def test_importances_asymptotic():

    # 定义二项式系数函数，用于计算组合数C(n, k)
    def binomial(k, n):
        return 0 if k < 0 or k > n else comb(int(n), int(k), exact=True)

    # 定义信息熵函数，用于计算样本集合的信息熵
    def entropy(samples):
        n_samples = len(samples)
        entropy = 0.0

        # 统计每个类别出现的次数，并计算其概率和信息熵
        for count in np.bincount(samples):
            p = 1.0 * count / n_samples
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    # 定义MDI变量重要性函数，用于计算给定X_m下的变量重要性
    def mdi_importance(X_m, X, y):
        n_samples, n_features = X.shape

        # 获取所有特征的索引列表
        features = list(range(n_features))
        # 剔除当前特征X_m，得到其他特征的索引列表
        features.pop(X_m)
        # 获取每个特征可能的取值列表
        values = [np.unique(X[:, i]) for i in range(n_features)]

        imp = 0.0

        # 遍历特征组合的大小k
        for k in range(n_features):
            # 计算每个大小为k的特征组合的权重系数
            coef = 1.0 / (binomial(k, n_features) * (n_features - k))

            # 遍历所有大小为k的特征组合B
            for B in combinations(features, k):
                # 遍历组合B中每个特征的可能取值b
                for b in product(*[values[B[j]] for j in range(k)]):
                    # 创建一个布尔掩码，用于选择与当前特征组合B的取值b匹配的样本
                    mask_b = np.ones(n_samples, dtype=bool)

                    for j in range(k):
                        mask_b &= X[:, B[j]] == b[j]

                    # 根据掩码选择对应的样本集合和标签
                    X_, y_ = X[mask_b, :], y[mask_b]
                    n_samples_b = len(X_)

                    if n_samples_b > 0:
                        children = []

                        # 针对当前特征X_m的每个可能取值xi，构建子集合
                        for xi in values[X_m]:
                            mask_xi = X_[:, X_m] == xi
                            children.append(y_[mask_xi])

                        # 计算MDI变量重要性的增益
                        imp += (
                            coef
                            * (1.0 * n_samples_b / n_samples)  # P(B=b)
                            * (
                                entropy(y_)
                                - sum(
                                    [
                                        entropy(c) * len(c) / n_samples_b
                                        for c in children
                                    ]
                                )
                            )
                        )

        return imp

    # 准备数据集
    data = np.array(
        [
            [0, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 2],
            [1, 0, 1, 1, 0, 1, 1, 3],
            [0, 1, 1, 1, 0, 1, 0, 4],
            [1, 1, 0, 1, 0, 1, 1, 5],
            [1, 1, 0, 1, 1, 1, 1, 6],
            [1, 0, 1, 0, 0, 1, 0, 7],
            [1, 1, 1, 1, 1, 1, 1, 8],
            [1, 1, 1, 1, 0, 1, 1, 9],
            [1, 1, 1, 0, 1, 1, 1, 0],
        ]
    )

    # 将数据集分割为特征X和标签y
    X, y = np.array(data[:, :7], dtype=bool), data[:, 7]
    n_features = X.shape[1]

    # 计算真实的变量重要性
    true_importances = np.zeros(n_features)

    for i in range(n_features):
        true_importances[i] = mdi_importance(i, X, y)

    # 使用完全随机树估算变量重要性
    # 使用 ExtraTreesClassifier 构建分类器，设置参数如下：
    # n_estimators=500 表示使用500棵树来构建随机森林
    # max_features=1 表示每棵树中使用的最大特征数为1
    # criterion="log_loss" 使用对数损失作为判断标准
    # random_state=0 设置随机种子以确保结果的可重复性
    clf = ExtraTreesClassifier(
        n_estimators=500, max_features=1, criterion="log_loss", random_state=0
    ).fit(X, y)

    # 计算每棵树的特征重要性并求和，得到平均特征重要性
    importances = (
        sum(
            tree.tree_.compute_feature_importances(normalize=False)
            for tree in clf.estimators_
        )
        / clf.n_estimators
    )

    # 检查正确性：使用信息熵计算目标变量 y 的熵值，并与特征重要性之和进行比较
    assert_almost_equal(entropy(y), sum(importances))
    
    # 检查真实特征重要性与计算得到的特征重要性的平均值之差的绝对值的平均值是否小于0.01
    assert np.abs(true_importances - importances).mean() < 0.01
# 使用 pytest.mark.parametrize 装饰器，对 test_unfitted_feature_importances 函数进行参数化测试
@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_unfitted_feature_importances(name):
    # 构造错误消息，指示未拟合的估算器实例，要求在使用估算器之前调用 'fit' 方法进行拟合
    err_msg = (
        "This {} instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator.".format(name)
    )
    # 使用 pytest.raises 断言捕获 NotFittedError 异常，并验证异常消息与 err_msg 匹配
    with pytest.raises(NotFittedError, match=err_msg):
        # 获取对应估算器的 feature_importances_ 属性，预期引发 NotFittedError 异常
        getattr(FOREST_ESTIMATORS[name](), "feature_importances_")


# 使用 pytest.mark.parametrize 装饰器，对 test_forest_classifier_oob 函数进行参数化测试
@pytest.mark.parametrize("ForestClassifier", FOREST_CLASSIFIERS.values())
@pytest.mark.parametrize("X_type", ["array", "sparse_csr", "sparse_csc"])
@pytest.mark.parametrize(
    "X, y, lower_bound_accuracy",
    [
        (
            *datasets.make_classification(n_samples=300, n_classes=2, random_state=0),
            0.9,
        ),
        (
            *datasets.make_classification(
                n_samples=1000, n_classes=3, n_informative=6, random_state=0
            ),
            0.65,
        ),
        (
            iris.data,
            iris.target * 2 + 1,
            0.65,
        ),
        (
            *datasets.make_multilabel_classification(n_samples=300, random_state=0),
            0.18,
        ),
    ],
)
@pytest.mark.parametrize("oob_score", [True, partial(f1_score, average="micro")])
def test_forest_classifier_oob(
    ForestClassifier, X, y, X_type, lower_bound_accuracy, oob_score
):
    """Check that OOB score is close to score on a test set."""
    # 根据 X_type 将 X 转换为适当的数据容器类型
    X = _convert_container(X, constructor_name=X_type)
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=0,
    )
    # 创建随机森林分类器实例
    classifier = ForestClassifier(
        n_estimators=40,
        bootstrap=True,
        oob_score=oob_score,
        random_state=0,
    )

    # 断言分类器不具有属性 "oob_score_" 和 "oob_decision_function_"
    assert not hasattr(classifier, "oob_score_")
    assert not hasattr(classifier, "oob_decision_function_")

    # 使用训练集拟合分类器
    classifier.fit(X_train, y_train)
    
    # 如果 oob_score 是可调用对象，则计算 oob_score
    if callable(oob_score):
        test_score = oob_score(y_test, classifier.predict(X_test))
    else:
        # 否则，计算分类器在测试集上的得分
        test_score = classifier.score(X_test, y_test)
        
    # 断言分类器的 oob_score_ 大于等于预定义的 lower_bound_accuracy
    assert classifier.oob_score_ >= lower_bound_accuracy

    # 断言测试集得分与 oob_score_ 的差异不超过 0.1
    assert abs(test_score - classifier.oob_score_) <= 0.1

    # 断言分类器具有属性 "oob_score_" 和 "oob_decision_function_"，但没有属性 "oob_prediction_"
    assert hasattr(classifier, "oob_score_")
    assert not hasattr(classifier, "oob_prediction_")
    assert hasattr(classifier, "oob_decision_function_")

    # 预期的 oob_decision_function_ 的形状，根据目标 y 的维度确定
    if y.ndim == 1:
        expected_shape = (X_train.shape[0], len(set(y)))
    else:
        expected_shape = (X_train.shape[0], len(set(y[:, 0])), y.shape[1])
    assert classifier.oob_decision_function_.shape == expected_shape


# 使用 pytest.mark.parametrize 装饰器，对 test_forest_regressor_oob 函数进行参数化测试
@pytest.mark.parametrize("ForestRegressor", FOREST_REGRESSORS.values())
@pytest.mark.parametrize("X_type", ["array", "sparse_csr", "sparse_csc"])
@pytest.mark.parametrize(
    "X, y, lower_bound_r2",
    # 创建两个回归数据集和对应的相关系数
    [
        # 第一个数据集：使用 make_regression 函数生成，包括 500 个样本、10个特征、1个目标值，随机种子为 0
        (
            *datasets.make_regression(
                n_samples=500, n_features=10, n_targets=1, random_state=0
            ),
            # 相关系数为 0.7
            0.7,
        ),
        # 第二个数据集：使用 make_regression 函数生成，包括 500 个样本、10个特征、2个目标值，随机种子为 0
        (
            *datasets.make_regression(
                n_samples=500, n_features=10, n_targets=2, random_state=0
            ),
            # 相关系数为 0.55
            0.55,
        ),
    ],
# 使用 pytest 的参数化装饰器，为测试函数提供多组参数化输入
@pytest.mark.parametrize("oob_score", [True, explained_variance_score])
def test_forest_regressor_oob(ForestRegressor, X, y, X_type, lower_bound_r2, oob_score):
    """检查基于森林的回归器是否提供接近测试集得分的OOB分数。"""
    # 将输入数据 X 转换为指定类型的容器
    X = _convert_container(X, constructor_name=X_type)
    # 将数据集拆分为训练集和测试集，测试集占总数据集的50%
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=0,
    )
    # 创建具有指定参数的森林回归器对象
    regressor = ForestRegressor(
        n_estimators=50,
        bootstrap=True,
        oob_score=oob_score,
        random_state=0,
    )

    # 断言：确保回归器对象不具有属性 "oob_score_" 和 "oob_prediction_"
    assert not hasattr(regressor, "oob_score_")
    assert not hasattr(regressor, "oob_prediction_")

    # 使用训练集拟合回归器
    regressor.fit(X_train, y_train)

    # 如果 oob_score 是可调用对象，则计算使用 oob_score 函数；否则使用 regressor.score 计算测试集得分
    if callable(oob_score):
        test_score = oob_score(y_test, regressor.predict(X_test))
    else:
        test_score = regressor.score(X_test, y_test)
        # 断言：确保 oob_score_ 大于等于预先定义的下限 lower_bound_r2
        assert regressor.oob_score_ >= lower_bound_r2

    # 断言：确保测试集得分与 oob_score_ 之间的差异小于等于 0.1
    assert abs(test_score - regressor.oob_score_) <= 0.1

    # 断言：确保回归器对象现在具有属性 "oob_score_" 和 "oob_prediction_"
    assert hasattr(regressor, "oob_score_")
    assert hasattr(regressor, "oob_prediction_")
    # 断言：确保回归器对象不具有属性 "oob_decision_function_"
    assert not hasattr(regressor, "oob_decision_function_")

    # 如果目标值 y 的维度为 1，则期望 oob_prediction_ 的形状为 (训练集样本数,)
    # 否则期望形状为 (训练集样本数, y 的维度)
    if y.ndim == 1:
        expected_shape = (X_train.shape[0],)
    else:
        expected_shape = (X_train.shape[0], y.ndim)
    # 断言：确保 oob_prediction_ 的形状符合预期
    assert regressor.oob_prediction_.shape == expected_shape


# 使用 pytest 的参数化装饰器，为测试函数提供多组参数化输入
@pytest.mark.parametrize("ForestEstimator", FOREST_CLASSIFIERS_REGRESSORS.values())
def test_forest_oob_warning(ForestEstimator):
    """检查当估计器数量不足且OOB估计将不准确时是否会引发警告。"""
    # 创建具有指定参数的森林估计器对象
    estimator = ForestEstimator(
        n_estimators=1,
        oob_score=True,
        bootstrap=True,
        random_state=0,
    )
    # 使用 pytest 的 warn 断言，检查是否引发 UserWarning，且消息包含 "Some inputs do not have OOB scores"
    with pytest.warns(UserWarning, match="Some inputs do not have OOB scores"):
        estimator.fit(iris.data, iris.target)


# 使用 pytest 的参数化装饰器，为测试函数提供多组参数化输入
@pytest.mark.parametrize("ForestEstimator", FOREST_CLASSIFIERS_REGRESSORS.values())
def test_forest_oob_score_requires_bootstrap(ForestEstimator):
    """检查如果请求 OOB 分数但未激活 bootstrapping 是否会引发错误。"""
    X = iris.data
    y = iris.target
    # 定义错误消息字符串
    err_msg = "Out of bag estimation only available if bootstrap=True"
    # 创建具有指定参数的森林估计器对象
    estimator = ForestEstimator(oob_score=True, bootstrap=False)
    # 使用 pytest 的 raises 断言，检查是否引发 ValueError，且消息匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        estimator.fit(X, y)


# 使用 pytest 的参数化装饰器，为测试函数提供多组参数化输入
@pytest.mark.parametrize("ForestClassifier", FOREST_CLASSIFIERS.values())
def test_classifier_error_oob_score_multiclass_multioutput(ForestClassifier):
    """检查请求使用多类多输出分类目标时是否会引发错误。"""
    # 创建随机数生成器
    rng = np.random.RandomState(42)
    X = iris.data
    # 创建多类多输出的随机目标值 y
    y = rng.randint(low=0, high=5, size=(iris.data.shape[0], 2))
    # 确定目标值 y 的类型
    y_type = type_of_target(y)
    # 断言：确保目标值 y 的类型为 "multiclass-multioutput"
    assert y_type == "multiclass-multioutput"
    # 创建具有指定参数的森林分类器对象
    estimator = ForestClassifier(oob_score=True, bootstrap=True)
    # 定义错误消息字符串
    err_msg = "The type of target cannot be used to compute OOB estimates"
    # 使用 pytest 的 raises 断言，检查是否引发 ValueError，且消息匹配 err_msg
    with pytest.raises(ValueError, match=err_msg):
        estimator.fit(X, y)
    # 使用 pytest 的上下文管理器检查是否抛出指定类型的异常，并验证异常消息是否匹配 err_msg 参数
    with pytest.raises(ValueError, match=err_msg):
        # 调用 estimator 对象的 fit 方法，传入训练数据 X 和标签数据 y
        estimator.fit(X, y)
# 使用 pytest 模块的 parametrize 装饰器，为 FOREST_REGRESSORS 字典中的每个回归器创建测试
@pytest.mark.parametrize("ForestRegressor", FOREST_REGRESSORS.values())
def test_forest_multioutput_integral_regression_target(ForestRegressor):
    """Check that multioutput regression with integral values is not interpreted
    as a multiclass-multioutput target and OOB score can be computed.
    """
    # 创建随机数生成器
    rng = np.random.RandomState(42)
    # 使用鸢尾花数据集的特征作为输入特征 X
    X = iris.data
    # 创建一个形状为 (样本数, 2) 的随机整数目标值 y
    y = rng.randint(low=0, high=10, size=(iris.data.shape[0], 2))
    # 使用给定的回归器创建估计器对象，并设置参数
    estimator = ForestRegressor(
        n_estimators=30, oob_score=True, bootstrap=True, random_state=0
    )
    # 对估计器进行拟合
    estimator.fit(X, y)

    # 计算每棵树的 bootstrap 样本数目
    n_samples_bootstrap = _get_n_samples_bootstrap(len(X), estimator.max_samples)
    # 设置测试样本数目为总样本数的四分之一
    n_samples_test = X.shape[0] // 4
    # 创建一个形状为 (n_samples_test, 2) 的零矩阵，用于存储 out-of-bag 预测值
    oob_pred = np.zeros([n_samples_test, 2])
    
    # 对前 n_samples_test 个样本进行循环
    for sample_idx, sample in enumerate(X[:n_samples_test]):
        # 初始化 out-of-bag 样本数目为 0，以及零数组用于存储 out-of-bag 预测
        n_samples_oob = 0
        oob_pred_sample = np.zeros(2)
        
        # 对估计器中的每棵树进行循环
        for tree in estimator.estimators_:
            # 生成未抽样索引，用于计算 out-of-bag
            oob_unsampled_indices = _generate_unsampled_indices(
                tree.random_state, len(X), n_samples_bootstrap
            )
            # 如果当前样本索引在 out-of-bag 未抽样索引中
            if sample_idx in oob_unsampled_indices:
                # 增加 out-of-bag 样本数目计数
                n_samples_oob += 1
                # 对当前样本进行预测，并将预测结果加到 out-of-bag 预测样本中
                oob_pred_sample += tree.predict(sample.reshape(1, -1)).squeeze()
        
        # 计算每个 out-of-bag 样本的平均预测值，并存储到 oob_pred 中
        oob_pred[sample_idx] = oob_pred_sample / n_samples_oob
    
    # 断言 out-of-bag 预测值与估计器的 oob_prediction_ 在前 n_samples_test 个样本上的接近程度
    assert_allclose(oob_pred, estimator.oob_prediction_[:n_samples_test])


# 使用 pytest 模块的 parametrize 装饰器，为 oob_score 参数创建测试
@pytest.mark.parametrize("oob_score", [True, False])
def test_random_trees_embedding_raise_error_oob(oob_score):
    # 断言在创建 RandomTreesEmbedding 对象时，使用不支持的 oob_score 参数会引发 TypeError 异常
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        RandomTreesEmbedding(oob_score=oob_score)
    # 断言在调用 RandomTreesEmbedding()._set_oob_score_and_attributes 方法时，会引发 NotImplementedError 异常
    with pytest.raises(NotImplementedError, match="OOB score not supported"):
        RandomTreesEmbedding()._set_oob_score_and_attributes(X, y)


# 使用 pytest 模块的 parametrize 装饰器，为 FOREST_CLASSIFIERS 字典中的每个分类器创建测试
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_gridsearch(name):
    # 检查可以对基础树进行网格搜索
    forest = FOREST_CLASSIFIERS[name]()
    # 创建一个 GridSearchCV 对象，对 n_estimators 和 max_depth 参数进行网格搜索
    clf = GridSearchCV(forest, {"n_estimators": (1, 2), "max_depth": (1, 2)})
    # 对鸢尾花数据集进行拟合
    clf.fit(iris.data, iris.target)


# 使用 pytest 模块的 parametrize 装饰器，为 FOREST_CLASSIFIERS_REGRESSORS 字典中的每个分类器或回归器创建测试
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_parallel(name):
    """Check parallel computations in classification"""
    # 根据分类器或回归器的名称选择合适的输入数据 X 和目标数据 y
    if name in FOREST_CLASSIFIERS:
        X = iris.data
        y = iris.target
    elif name in FOREST_REGRESSORS:
        X = X_reg
        y = y_reg

    # 根据名称获取相应的估计器类
    ForestEstimator = FOREST_ESTIMATORS[name]
    # 创建估计器对象，设置参数为 n_estimators=10, n_jobs=3, random_state=0
    forest = ForestEstimator(n_estimators=10, n_jobs=3, random_state=0)

    # 对估计器对象进行拟合
    forest.fit(X, y)
    # 断言估计器对象的长度为 10
    assert len(forest) == 10

    # 将 n_jobs 参数分别设置为 1 和 2，并分别预测数据 X，然后断言两次预测结果的接近程度
    forest.set_params(n_jobs=1)
    y1 = forest.predict(X)
    forest.set_params(n_jobs=2)
    y2 = forest.predict(X)
    assert_array_almost_equal(y1, y2, 3)


# 使用 pytest 模块的 parametrize 装饰器，为 FOREST_CLASSIFIERS_REGRESSORS 字典中的每个分类器或回归器创建测试
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_pickle(name):
    # 检查估计器对象的可序列化性
    # 根据分类器或回归器的名称选择合适的输入数据 X 和目标数据 y
    if name in FOREST_CLASSIFIERS:
        X = iris.data[::2]
        y = iris.target[::2]
    elif name in FOREST_REGRESSORS:
        X = X_reg[::2]
        y = y_reg[::2]

    # 根据名称获取相应的估计器类
    ForestEstimator = FOREST_ESTIMATORS[name]
    # 创建一个随机森林估计器对象，设置随机种子为0
    obj = ForestEstimator(random_state=0)
    # 使用输入数据 X 和标签 y 对对象进行拟合
    obj.fit(X, y)
    # 计算对象在输入数据 X 和标签 y 上的得分
    score = obj.score(X, y)
    # 将对象序列化为字节流对象
    pickle_object = pickle.dumps(obj)

    # 从序列化的字节流对象中加载对象
    obj2 = pickle.loads(pickle_object)
    # 断言加载的对象类型与原对象类型相同
    assert type(obj2) == obj.__class__
    # 计算加载的对象在输入数据 X 和标签 y 上的得分
    score2 = obj2.score(X, y)
    # 断言原对象和加载的对象在相同数据上得分相同
    assert score == score2
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_multioutput(name):
    # Check estimators on multi-output problems.

    X_train = [
        [-2, -1],   # 定义训练集特征 X_train
        [-1, -1],
        [-1, -2],
        [1, 1],
        [1, 2],
        [2, 1],
        [-2, 1],
        [-1, 1],
        [-1, 2],
        [2, -1],
        [1, -1],
        [1, -2],
    ]
    y_train = [
        [-1, 0],    # 定义训练集标签 y_train
        [-1, 0],
        [-1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [-1, 2],
        [-1, 2],
        [-1, 2],
        [1, 3],
        [1, 3],
        [1, 3],
    ]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]   # 定义测试集特征 X_test
    y_test = [[-1, 0], [1, 1], [-1, 2], [1, 3]]     # 定义测试集标签 y_test

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)   # 初始化分类器或回归器 est
    y_pred = est.fit(X_train, y_train).predict(X_test)   # 训练模型并预测测试集标签

    assert_array_almost_equal(y_pred, y_test)   # 断言预测结果与测试集标签 y_test 几乎相等

    if name in FOREST_CLASSIFIERS:
        with np.errstate(divide="ignore"):
            proba = est.predict_proba(X_test)   # 预测测试集的概率
            assert len(proba) == 2   # 断言概率的长度为 2
            assert proba[0].shape == (4, 2)   # 断言第一个类别的概率形状为 (4, 2)
            assert proba[1].shape == (4, 4)   # 断言第二个类别的概率形状为 (4, 4)

            log_proba = est.predict_log_proba(X_test)   # 预测测试集的对数概率
            assert len(log_proba) == 2   # 断言对数概率的长度为 2
            assert log_proba[0].shape == (4, 2)   # 断言第一个类别的对数概率形状为 (4, 2)
            assert log_proba[1].shape == (4, 4)   # 断言第二个类别的对数概率形状为 (4, 4)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_multioutput_string(name):
    # Check estimators on multi-output problems with string outputs.

    X_train = [
        [-2, -1],   # 定义训练集特征 X_train
        [-1, -1],
        [-1, -2],
        [1, 1],
        [1, 2],
        [2, 1],
        [-2, 1],
        [-1, 1],
        [-1, 2],
        [2, -1],
        [1, -1],
        [1, -2],
    ]
    y_train = [
        ["red", "blue"],    # 定义训练集标签 y_train
        ["red", "blue"],
        ["red", "blue"],
        ["green", "green"],
        ["green", "green"],
        ["green", "green"],
        ["red", "purple"],
        ["red", "purple"],
        ["red", "purple"],
        ["green", "yellow"],
        ["green", "yellow"],
        ["green", "yellow"],
    ]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]   # 定义测试集特征 X_test
    y_test = [
        ["red", "blue"],    # 定义测试集标签 y_test
        ["green", "green"],
        ["red", "purple"],
        ["green", "yellow"],
    ]

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)   # 初始化分类器 est
    y_pred = est.fit(X_train, y_train).predict(X_test)   # 训练模型并预测测试集标签

    assert_array_equal(y_pred, y_test)   # 断言预测结果与测试集标签 y_test 相等

    with np.errstate(divide="ignore"):
        proba = est.predict_proba(X_test)   # 预测测试集的概率
        assert len(proba) == 2   # 断言概率的长度为 2
        assert proba[0].shape == (4, 2)   # 断言第一个类别的概率形状为 (4, 2)
        assert proba[1].shape == (4, 4)   # 断言第二个类别的概率形状为 (4, 4)

        log_proba = est.predict_log_proba(X_test)   # 预测测试集的对数概率
        assert len(log_proba) == 2   # 断言对数概率的长度为 2
        assert log_proba[0].shape == (4, 2)   # 断言第一个类别的对数概率形状为 (4, 2)
        assert log_proba[1].shape == (4, 4)   # 断言第二个类别的对数概率形状为 (4, 4)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_classes_shape(name):
    # Test that n_classes_ and classes_ have proper shape.
    ForestClassifier = FOREST_CLASSIFIERS[name]   # 获取对应分类器的类对象 ForestClassifier
    # 使用随机种子为0来初始化一个随机森林分类器，并用数据集 X 和标签 y 进行训练
    clf = ForestClassifier(random_state=0).fit(X, y)

    # 断言分类器 clf 的类别数为 2
    assert clf.n_classes_ == 2
    # 断言分类器 clf 的类别列表与预期的 [-1, 1] 相等
    assert_array_equal(clf.classes_, [-1, 1])

    # 对于多输出分类，创建一个新的标签 _y，它是原始标签 y 和 y 的每个元素乘以2组成的数组的转置
    _y = np.vstack((y, np.array(y) * 2)).T
    # 使用相同的随机种子为0来初始化另一个随机森林分类器，并用数据集 X 和 _y 进行训练
    clf = ForestClassifier(random_state=0).fit(X, _y)

    # 断言分类器 clf 的每个输出的类别数与预期的 [2, 2] 相等
    assert_array_equal(clf.n_classes_, [2, 2])
    # 断言分类器 clf 的每个输出的类别列表与预期的 [[-1, 1], [-2, 2]] 相等
    assert_array_equal(clf.classes_, [[-1, 1], [-2, 2]])
def test_random_trees_dense_type():
    # Test that the `sparse_output` parameter of RandomTreesEmbedding
    # works by returning a dense array.

    # Create the RTE with sparse=False
    hasher = RandomTreesEmbedding(n_estimators=10, sparse_output=False)
    # Generate a synthetic dataset of circles
    X, y = datasets.make_circles(factor=0.5)
    # Transform the data using RandomTreesEmbedding
    X_transformed = hasher.fit_transform(X)

    # Assert that type is ndarray, not scipy.sparse.csr_matrix
    assert isinstance(X_transformed, np.ndarray)


def test_random_trees_dense_equal():
    # Test that the `sparse_output` parameter of RandomTreesEmbedding
    # works by returning the same array for both argument values.

    # Create the RTEs with different sparse_output settings
    hasher_dense = RandomTreesEmbedding(
        n_estimators=10, sparse_output=False, random_state=0
    )
    hasher_sparse = RandomTreesEmbedding(
        n_estimators=10, sparse_output=True, random_state=0
    )
    # Generate a synthetic dataset of circles
    X, y = datasets.make_circles(factor=0.5)
    # Transform the data using both RTEs
    X_transformed_dense = hasher_dense.fit_transform(X)
    X_transformed_sparse = hasher_sparse.fit_transform(X)

    # Assert that dense and sparse hashers have the same array.
    assert_array_equal(X_transformed_sparse.toarray(), X_transformed_dense)


# Ignore warnings from switching to more power iterations in randomized_svd
@ignore_warnings
def test_random_hasher():
    # test random forest hashing on circles dataset
    # make sure that it is linearly separable.
    # even after projected to two SVD dimensions
    # Note: Not all random_states produce perfect results.
    hasher = RandomTreesEmbedding(n_estimators=30, random_state=1)
    # Generate a synthetic dataset of circles
    X, y = datasets.make_circles(factor=0.5)
    # Transform the data using RandomTreesEmbedding
    X_transformed = hasher.fit_transform(X)

    # test fit and transform:
    hasher = RandomTreesEmbedding(n_estimators=30, random_state=1)
    assert_array_equal(hasher.fit(X).transform(X).toarray(), X_transformed.toarray())

    # one leaf active per data point per forest
    assert X_transformed.shape[0] == X.shape[0]
    assert_array_equal(X_transformed.sum(axis=1), hasher.n_estimators)
    svd = TruncatedSVD(n_components=2)
    X_reduced = svd.fit_transform(X_transformed)
    linear_clf = LinearSVC()
    linear_clf.fit(X_reduced, y)
    assert linear_clf.score(X_reduced, y) == 1.0


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_random_hasher_sparse_data(csc_container):
    # Generate a synthetic dataset for multilabel classification
    X, y = datasets.make_multilabel_classification(random_state=0)
    hasher = RandomTreesEmbedding(n_estimators=30, random_state=1)
    # Transform the data using RandomTreesEmbedding
    X_transformed = hasher.fit_transform(X)
    X_transformed_sparse = hasher.fit_transform(csc_container(X))
    assert_array_equal(X_transformed_sparse.toarray(), X_transformed.toarray())


def test_parallel_train():
    # Initialize random number generator for reproducibility
    rng = check_random_state(12321)
    n_samples, n_features = 80, 30
    # Generate synthetic training data
    X_train = rng.randn(n_samples, n_features)
    y_train = rng.randint(0, 2, n_samples)
    # 创建一个包含多个随机森林分类器的列表，每个分类器使用不同的并行工作数（n_jobs）
    clfs = [
        RandomForestClassifier(n_estimators=20, n_jobs=n_jobs, random_state=12345).fit(
            X_train, y_train
        )
        for n_jobs in [1, 2, 3, 8, 16, 32]
    ]

    # 使用随机数生成器生成测试数据集 X_test，其形状为 (n_samples, n_features)
    X_test = rng.randn(n_samples, n_features)
    
    # 对每个分类器 clf 在 X_test 上进行预测，得到预测概率列表 probas
    probas = [clf.predict_proba(X_test) for clf in clfs]
    
    # 对 probas 中的每对相邻预测概率进行检查，确保它们几乎相等（精度检查）
    for proba1, proba2 in zip(probas, probas[1:]):
        assert_array_almost_equal(proba1, proba2)
def test_distribution():
    rng = check_random_state(12321)

    # 生成一个包含1000行、1列的数组，每个元素为0到3之间的随机整数
    X = rng.randint(0, 4, size=(1000, 1))
    # 生成一个包含1000个随机数的数组，每个数均匀分布在[0, 1)之间
    y = rng.rand(1000)
    n_trees = 500

    # 使用ExtraTreesRegressor构建回归器，设置500棵树，并用(X, y)训练
    reg = ExtraTreesRegressor(n_estimators=n_trees, random_state=42).fit(X, y)

    # 使用defaultdict初始化uniques字典，用于存储不同树结构的出现次数
    uniques = defaultdict(int)
    # 遍历每棵树
    for tree in reg.estimators_:
        # 将树的特征和阈值连接成字符串表示
        tree = "".join(
            ("%d,%d/" % (f, int(t)) if f >= 0 else "-")
            for f, t in zip(tree.tree_.feature, tree.tree_.threshold)
        )
        # 统计每种树结构出现的次数
        uniques[tree] += 1

    # 对uniques字典按照出现次数比例排序，存储为列表
    uniques = sorted([(1.0 * count / n_trees, tree) for tree, count in uniques.items()])

    # 对单变量问题进行验证：X_0有4个等概率值时，有5种构建随机树的方式
    # 最紧凑的一种 (0,1/0,0/--0,2/--) 的概率约为1/3，其他4种概率约为1/6
    assert len(uniques) == 5
    assert 0.20 > uniques[0][0]  # 大致估计为1/6
    assert 0.20 > uniques[1][0]
    assert 0.20 > uniques[2][0]
    assert 0.20 > uniques[3][0]
    assert uniques[4][0] > 0.3
    assert uniques[4][1] == "0,1/0,0/--0,2/--"

    # 生成一个包含1000行、2列的空数组
    X = np.empty((1000, 2))
    # 第一列随机取0或1
    X[:, 0] = np.random.randint(0, 2, 1000)
    # 第二列随机取0到2之间的整数
    X[:, 1] = np.random.randint(0, 3, 1000)
    y = rng.rand(1000)

    # 使用ExtraTreesRegressor构建回归器，设置max_features=1，并用(X, y)训练
    reg = ExtraTreesRegressor(max_features=1, random_state=1).fit(X, y)

    # 初始化uniques字典，用于存储不同树结构的出现次数
    uniques = defaultdict(int)
    # 遍历每棵树
    for tree in reg.estimators_:
        # 将树的特征和阈值连接成字符串表示
        tree = "".join(
            ("%d,%d/" % (f, int(t)) if f >= 0 else "-")
            for f, t in zip(tree.tree_.feature, tree.tree_.threshold)
        )
        # 统计每种树结构出现的次数
        uniques[tree] += 1

    # 将uniques字典转换为列表，包含出现次数和树结构
    uniques = [(count, tree) for tree, count in uniques.items()]
    assert len(uniques) == 8


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_max_leaf_nodes_max_depth(name):
    X, y = hastie_X, hastie_y

    # 测试max_leaf_nodes对max_depth的优先级
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(
        max_depth=1, max_leaf_nodes=4, n_estimators=1, random_state=0
    ).fit(X, y)
    assert est.estimators_[0].get_depth() == 1

    est = ForestEstimator(max_depth=1, n_estimators=1, random_state=0).fit(X, y)
    assert est.estimators_[0].get_depth() == 1


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_min_samples_split(name):
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]

    # 使用min_samples_split=10构建ForestEstimator，并检查节点样本数是否大于阈值
    est = ForestEstimator(min_samples_split=10, n_estimators=1, random_state=0)
    est.fit(X, y)
    node_idx = est.estimators_[0].tree_.children_left != -1
    node_samples = est.estimators_[0].tree_.n_node_samples[node_idx]

    assert np.min(node_samples) > len(X) * 0.5 - 1, "Failed with {0}".format(name)

    # 使用min_samples_split=0.5构建ForestEstimator，并检查节点样本数是否大于阈值
    est = ForestEstimator(min_samples_split=0.5, n_estimators=1, random_state=0)
    est.fit(X, y)
    node_idx = est.estimators_[0].tree_.children_left != -1
    node_samples = est.estimators_[0].tree_.n_node_samples[node_idx]
    # 断言语句：确保节点样本数的最小值大于样本集 X 的一半减一
    assert np.min(node_samples) > len(X) * 0.5 - 1, "Failed with {0}".format(name)
# 使用 pytest 框架的 parametrize 装饰器，为 FOREST_ESTIMATORS 中的每个模型名称执行下面的测试函数
@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_min_samples_leaf(name):
    # 准备数据集 X, y，这里使用 hastie_X 和 hastie_y
    X, y = hastie_X, hastie_y

    # 获取当前循环的模型类 ForestEstimator
    ForestEstimator = FOREST_ESTIMATORS[name]

    # 创建模型对象 est，设置 min_samples_leaf=5, n_estimators=1, random_state=0 并拟合数据
    est = ForestEstimator(min_samples_leaf=5, n_estimators=1, random_state=0)
    est.fit(X, y)
    
    # 获取第一个基本估计器的叶子节点索引
    out = est.estimators_[0].tree_.apply(X)
    
    # 统计每个节点的出现次数
    node_counts = np.bincount(out)
    
    # 选择所有非零节点（即叶子节点）
    leaf_count = node_counts[node_counts != 0]
    
    # 断言叶子节点的最小数量大于4
    assert np.min(leaf_count) > 4, "Failed with {0}".format(name)

    # 以 min_samples_leaf=0.25 重新创建 est，并进行拟合
    est = ForestEstimator(min_samples_leaf=0.25, n_estimators=1, random_state=0)
    est.fit(X, y)
    
    # 获取第一个基本估计器的叶子节点索引
    out = est.estimators_[0].tree_.apply(X)
    
    # 统计每个节点的出现次数
    node_counts = np.bincount(out)
    
    # 选择所有非零节点（即叶子节点）
    leaf_count = node_counts[node_counts != 0]
    
    # 断言叶子节点的最小数量大于 len(X) * 0.25 - 1
    assert np.min(leaf_count) > len(X) * 0.25 - 1, "Failed with {0}".format(name)


# 使用 pytest 框架的 parametrize 装饰器，为 FOREST_ESTIMATORS 中的每个模型名称执行下面的测试函数
@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_min_weight_fraction_leaf(name):
    # 准备数据集 X, y，这里使用 hastie_X 和 hastie_y
    X, y = hastie_X, hastie_y

    # 获取当前循环的模型类 ForestEstimator
    ForestEstimator = FOREST_ESTIMATORS[name]
    
    # 创建随机数生成器 rng
    rng = np.random.RandomState(0)
    
    # 生成随机样本权重
    weights = rng.rand(X.shape[0])
    
    # 计算总权重
    total_weight = np.sum(weights)

    # 对于 np.linspace(0, 0.5, 6) 中的每个 frac 值
    for frac in np.linspace(0, 0.5, 6):
        # 创建模型对象 est，设置 min_weight_fraction_leaf=frac, n_estimators=1, random_state=0 并拟合数据
        est = ForestEstimator(
            min_weight_fraction_leaf=frac, n_estimators=1, random_state=0
        )
        
        # 如果模型名中包含 "RandomForest"，则设置 bootstrap=False
        if "RandomForest" in name:
            est.bootstrap = False
        
        # 使用样本权重拟合数据
        est.fit(X, y, sample_weight=weights)
        
        # 获取第一个基本估计器的叶子节点索引
        out = est.estimators_[0].tree_.apply(X)
        
        # 根据权重统计每个节点的出现次数
        node_weights = np.bincount(out, weights=weights)
        
        # 选择所有非零节点（即叶子节点）
        leaf_weights = node_weights[node_weights != 0]
        
        # 断言叶子节点的最小权重大于等于 total_weight * est.min_weight_fraction_leaf
        assert (
            np.min(leaf_weights) >= total_weight * est.min_weight_fraction_leaf
        ), "Failed with {0} min_weight_fraction_leaf={1}".format(
            name, est.min_weight_fraction_leaf
        )


# 使用 pytest 框架的 parametrize 装饰器，为 FOREST_ESTIMATORS 中的每个模型名称和 COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS 的每个稀疏容器执行下面的测试函数
@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
def test_sparse_input(name, sparse_container):
    # 使用 make_multilabel_classification 生成多标签分类的数据集 X, y
    X, y = datasets.make_multilabel_classification(random_state=0, n_samples=50)

    # 获取当前循环的模型类 ForestEstimator
    ForestEstimator = FOREST_ESTIMATORS[name]

    # 创建深度为 2 的稠密模型 dense，并拟合数据 X, y
    dense = ForestEstimator(random_state=0, max_depth=2).fit(X, y)
    
    # 创建深度为 2 的稀疏模型 sparse，并拟合经过 sparse_container 处理的数据 X, y
    sparse = ForestEstimator(random_state=0, max_depth=2).fit(sparse_container(X), y)

    # 断言稀疏模型的 apply 方法的结果与稠密模型相同
    assert_array_almost_equal(sparse.apply(X), dense.apply(X))

    # 如果模型名在 FOREST_CLASSIFIERS 或 FOREST_REGRESSORS 中
    if name in FOREST_CLASSIFIERS or name in FOREST_REGRESSORS:
        # 断言稀疏模型的 predict 方法的结果与稠密模型相同
        assert_array_almost_equal(sparse.predict(X), dense.predict(X))
        
        # 断言稀疏模型的 feature_importances_ 属性与稠密模型相同
        assert_array_almost_equal(
            sparse.feature_importances_, dense.feature_importances_
        )
    # 如果当前的分类器名称在 FOREST_CLASSIFIERS 列表中
    if name in FOREST_CLASSIFIERS:
        # 断言稀疏模型和密集模型对给定数据 X 的预测概率是否几乎相等
        assert_array_almost_equal(sparse.predict_proba(X), dense.predict_proba(X))
        # 断言稀疏模型和密集模型对给定数据 X 的预测对数概率是否几乎相等
        assert_array_almost_equal(
            sparse.predict_log_proba(X), dense.predict_log_proba(X)
        )

    # 如果当前的变换器名称在 FOREST_TRANSFORMERS 列表中
    if name in FOREST_TRANSFORMERS:
        # 断言稀疏模型对给定数据 X 的转换结果（转换为稀疏矩阵后）和密集模型的转换结果（转换为稀疏矩阵后）是否几乎相等
        assert_array_almost_equal(
            sparse.transform(X).toarray(), dense.transform(X).toarray()
        )
        # 断言稀疏模型对给定数据 X 的拟合转换结果（拟合并转换为稀疏矩阵后）和密集模型的拟合转换结果（拟合并转换为稀疏矩阵后）是否几乎相等
        assert_array_almost_equal(
            sparse.fit_transform(X).toarray(), dense.fit_transform(X).toarray()
        )
# 使用 pytest 的参数化装饰器，对 FOREST_CLASSIFIERS_REGRESSORS 中的每个模型和 np.float64、np.float32 两种数据类型分别执行测试
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
@pytest.mark.parametrize("dtype", (np.float64, np.float32))
def test_memory_layout(name, dtype):
    # 使用指定模型名字实例化一个估算器对象，设置随机种子为0，禁用自举采样（bootstrap=False）
    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)

    # 对不同的内存布局进行测试
    # Dense Arrays
    for container, kwargs in (
        (np.asarray, {}),  # 默认情况
        (np.asarray, {"order": "C"}),  # C顺序（行主序）
        (np.asarray, {"order": "F"}),  # F顺序（列主序）
        (np.ascontiguousarray, {}),  # 连续内存布局
    ):
        # 根据容器类型和参数创建数据数组 X，并指定数据类型为 dtype
        X = container(iris.data, dtype=dtype, **kwargs)
        y = iris.target
        # 断言拟合后的估算器在 X 上的预测结果与 y 相近
        assert_array_almost_equal(est.fit(X, y).predict(X), y)

    # Sparse Arrays（如果模型支持稀疏数据）
    if est.estimator.splitter in SPARSE_SPLITTERS:
        for sparse_container in COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS:
            X = sparse_container(iris.data, dtype=dtype)
            y = iris.target
            assert_array_almost_equal(est.fit(X, y).predict(X), y)

    # Strided Arrays
    X = np.asarray(iris.data[::3], dtype=dtype)
    y = iris.target[::3]
    assert_array_almost_equal(est.fit(X, y).predict(X), y)


# 使用 pytest 的参数化装饰器，对 FOREST_ESTIMATORS 中的每个分类器模型执行测试
@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_1d_input(name):
    # 取第一列作为 1 维输入
    X = iris.data[:, 0]
    # 将 1 维输入转换为 2 维输入
    X_2d = iris.data[:, 0].reshape((-1, 1))
    y = iris.target

    with ignore_warnings():
        # 获取当前模型的估算器类
        ForestEstimator = FOREST_ESTIMATORS[name]
        # 期望引发 ValueError 异常，因为输入维度不符合要求
        with pytest.raises(ValueError):
            ForestEstimator(n_estimators=1, random_state=0).fit(X, y)

        # 使用 2 维输入拟合模型
        est = ForestEstimator(random_state=0)
        est.fit(X_2d, y)

        # 对于分类器和回归器模型，期望引发 ValueError 异常，因为预测输入维度不符合要求
        if name in FOREST_CLASSIFIERS or name in FOREST_REGRESSORS:
            with pytest.raises(ValueError):
                est.predict(X)


# 使用 pytest 的参数化装饰器，对 FOREST_CLASSIFIERS 中的每个分类器模型执行测试
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_class_weights(name):
    # 检查 class_weight 参数是否像 sample_weights 一样影响模型行为
    ForestClassifier = FOREST_CLASSIFIERS[name]

    # 使用默认设置（未设置 class_weight），期望结果无变化
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target)

    # 使用 'balanced' 权重，期望结果无变化
    clf2 = ForestClassifier(class_weight="balanced", random_state=0)
    clf2.fit(iris.data, iris.target)
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # 创建多输出问题（使用三个相同的 Iris 数据集）
    iris_multi = np.vstack((iris.target, iris.target, iris.target)).T

    # 创建用户定义的权重，预期在多输出上能够平衡各类别
    clf3 = ForestClassifier(
        class_weight=[
            {0: 2.0, 1: 2.0, 2: 1.0},
            {0: 2.0, 1: 1.0, 2: 2.0},
            {0: 1.0, 1: 2.0, 2: 2.0},
        ],
        random_state=0,
    )
    clf3.fit(iris.data, iris_multi)
    assert_almost_equal(clf2.feature_importances_, clf3.feature_importances_)

    # 使用 'balanced' 权重在多输出问题上，期望结果无变化
    clf4 = ForestClassifier(class_weight="balanced", random_state=0)
    clf4.fit(iris.data, iris_multi)
    # 检查 clf3 和 clf4 的特征重要性是否几乎相等
    assert_almost_equal(clf3.feature_importances_, clf4.feature_importances_)

    # 增大类别 1 的重要性，并与用户定义的权重进行比较
    sample_weight = np.ones(iris.target.shape)
    sample_weight[iris.target == 1] *= 100
    class_weight = {0: 1.0, 1: 100.0, 2: 1.0}

    # 创建一个没有使用类别权重的随机森林分类器 clf1
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight)

    # 创建一个使用了用户定义类别权重的随机森林分类器 clf2
    clf2 = ForestClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target)

    # 检查 clf1 和 clf2 的特征重要性是否几乎相等
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)

    # 检查样本权重和类别权重是否是乘法关系
    clf1 = ForestClassifier(random_state=0)
    clf1.fit(iris.data, iris.target, sample_weight**2)

    clf2 = ForestClassifier(class_weight=class_weight, random_state=0)
    clf2.fit(iris.data, iris.target, sample_weight)

    # 检查 clf1 和 clf2 的特征重要性是否几乎相等
    assert_almost_equal(clf1.feature_importances_, clf2.feature_importances_)
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_class_weight_balanced_and_bootstrap_multi_output(name):
    # 测试多输出情况下 class_weight 是否有效
    ForestClassifier = FOREST_CLASSIFIERS[name]
    _y = np.vstack((y, np.array(y) * 2)).T
    # 使用 balanced class_weight 创建分类器实例，并训练模型
    clf = ForestClassifier(class_weight="balanced", random_state=0)
    clf.fit(X, _y)
    # 使用指定的 class_weight 列表创建分类器实例，并训练模型
    clf = ForestClassifier(
        class_weight=[{-1: 0.5, 1: 1.0}, {-2: 1.0, 2: 1.0}], random_state=0
    )
    clf.fit(X, _y)
    # 使用 balanced_subsample class_weight 创建分类器实例，并训练模型（烟雾测试）
    clf = ForestClassifier(class_weight="balanced_subsample", random_state=0)
    clf.fit(X, _y)


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_class_weight_errors(name):
    # 测试 class_weight 是否在预期情况下引发错误和警告
    ForestClassifier = FOREST_CLASSIFIERS[name]
    _y = np.vstack((y, np.array(y) * 2)).T

    # 使用 warm_start 和 balanced class_weight 创建分类器实例，并训练模型
    clf = ForestClassifier(class_weight="balanced", warm_start=True, random_state=0)
    clf.fit(X, y)

    # 验证警告信息是否匹配，并引发 UserWarning
    warn_msg = (
        "Warm-start fitting without increasing n_estimators does not fit new trees."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X, _y)

    # 使用长度不正确的 class_weight 列表创建分类器实例，并引发 ValueError
    clf = ForestClassifier(class_weight=[{-1: 0.5, 1: 1.0}], random_state=0)
    with pytest.raises(ValueError):
        clf.fit(X, _y)


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_warm_start(name):
    # 测试使用 warm start 逐步训练是否产生正确大小的森林以及与普通训练相同的结果
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    est_ws = None
    for n_estimators in [5, 10]:
        if est_ws is None:
            est_ws = ForestEstimator(
                n_estimators=n_estimators, random_state=42, warm_start=True
            )
        else:
            est_ws.set_params(n_estimators=n_estimators)
        est_ws.fit(X, y)
        assert len(est_ws) == n_estimators

    # 创建不使用 warm start 的分类器实例，并训练模型
    est_no_ws = ForestEstimator(n_estimators=10, random_state=42, warm_start=False)
    est_no_ws.fit(X, y)

    # 检查使用 warm start 和不使用 warm start 时生成的随机状态是否相同
    assert set([tree.random_state for tree in est_ws]) == set(
        [tree.random_state for tree in est_no_ws]
    )

    # 检查 apply 方法的输出是否相等，用于验证分类器结果
    assert_array_equal(
        est_ws.apply(X), est_no_ws.apply(X), err_msg="Failed with {0}".format(name)
    )


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_warm_start_clear(name):
    # 测试在 warm_start==False 时，fit 方法是否清除状态并生成新的森林
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    # 创建不使用 warm start 的分类器实例，并训练模型
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=False, random_state=1)
    est.fit(X, y)

    # 创建使用 warm start 的分类器实例，并训练模型
    est_2 = ForestEstimator(
        n_estimators=5, max_depth=1, warm_start=True, random_state=2
    )
    est_2.fit(X, y)  # 初始化状态
    est_2.set_params(warm_start=False, random_state=1)
    est_2.fit(X, y)  # 清除旧状态，并等同于 est 的结果
    # 断言检查：验证 est_2 对象应用到 X 上的结果几乎等于 est 对象应用到 X 上的结果
    assert_array_almost_equal(est_2.apply(X), est.apply(X))
# 使用 pytest.mark.parametrize 装饰器，依次传入 FOREST_ESTIMATORS 中的每个模型名称进行测试
@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_warm_start_smaller_n_estimators(name):
    # 测试当 warm_start=True 时，第二次拟合时减小 n_estimators 是否会引发错误
    X, y = hastie_X, hastie_y
    # 根据模型名称获取对应的模型类
    ForestEstimator = FOREST_ESTIMATORS[name]
    # 初始化一个模型对象，设定 n_estimators=5, max_depth=1，并启用 warm_start
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=True)
    # 第一次拟合模型
    est.fit(X, y)
    # 修改 n_estimators 参数为 4
    est.set_params(n_estimators=4)
    # 使用 pytest.raises 检查是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        est.fit(X, y)


@pytest.mark.parametrize("name", FOREST_ESTIMATORS)
def test_warm_start_equal_n_estimators(name):
    # 测试当 warm_start=True 且 n_estimators 相等时，是否什么都不做并返回相同的森林，同时会引发警告
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    # 初始化第一个模型对象，设定 n_estimators=5, max_depth=3, warm_start=True, random_state=1
    est = ForestEstimator(n_estimators=5, max_depth=3, warm_start=True, random_state=1)
    est.fit(X, y)

    # 初始化第二个模型对象，设定相同的参数
    est_2 = ForestEstimator(
        n_estimators=5, max_depth=3, warm_start=True, random_state=1
    )
    est_2.fit(X, y)
    # 现在 est_2 应该等于 est.

    # 修改 est_2 的 random_state 参数
    est_2.set_params(random_state=2)
    # 设置警告信息的匹配文本
    warn_msg = (
        "Warm-start fitting without increasing n_estimators does not fit new trees."
    )
    # 使用 pytest.warns 检查是否引发 UserWarning，并匹配 warn_msg
    with pytest.warns(UserWarning, match=warn_msg):
        est_2.fit(X, y)
    # 如果我们重新拟合了树，应该会得到不同的森林，因为我们改变了随机状态。
    # 检查 est 和 est_2 的应用结果是否相等
    assert_array_equal(est.apply(X), est_2.apply(X))


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_warm_start_oob(name):
    # 测试当设置 warm_start=True 时，是否在需要时计算 oob 得分
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    # 初始化第一个模型对象，设定 n_estimators=15, max_depth=3, warm_start=False, random_state=1, bootstrap=True, oob_score=True
    est = ForestEstimator(
        n_estimators=15,
        max_depth=3,
        warm_start=False,
        random_state=1,
        bootstrap=True,
        oob_score=True,
    )
    est.fit(X, y)

    # 初始化第二个模型对象，设定 n_estimators=5, max_depth=3, warm_start=False, random_state=1, bootstrap=True, oob_score=False
    est_2 = ForestEstimator(
        n_estimators=5,
        max_depth=3,
        warm_start=False,
        random_state=1,
        bootstrap=True,
        oob_score=False,
    )
    est_2.fit(X, y)

    # 修改 est_2 的参数，设置 warm_start=True, oob_score=True, n_estimators=15
    est_2.set_params(warm_start=True, oob_score=True, n_estimators=15)
    est_2.fit(X, y)

    # 检查 est_2 是否具有属性 "oob_score_"
    assert hasattr(est_2, "oob_score_")
    # 检查 est 和 est_2 的 oob_score_ 是否相等
    assert est.oob_score_ == est_2.oob_score_

    # 测试即使不需要训练额外的树，也要计算 oob_score
    # 初始化第三个模型对象，设定 n_estimators=15, max_depth=3, warm_start=True, random_state=1, bootstrap=True, oob_score=False
    est_3 = ForestEstimator(
        n_estimators=15,
        max_depth=3,
        warm_start=True,
        random_state=1,
        bootstrap=True,
        oob_score=False,
    )
    est_3.fit(X, y)
    # 检查 est_3 是否没有属性 "oob_score_"
    assert not hasattr(est_3, "oob_score_")

    # 修改 est_3 的参数，设置 oob_score=True，并忽略警告进行拟合
    ignore_warnings(est_3.fit)(X, y)
    # 检查 est 和 est_3 的 oob_score_ 是否相等
    assert est.oob_score_ == est_3.oob_score_


@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_oob_not_computed_twice(name):
    # 检查当 warm_start=True 时，不会计算两次 oob_score
    X, y = hastie_X, hastie_y
    ForestEstimator = FOREST_ESTIMATORS[name]
    # 创建一个名为 est 的 ForestEstimator 对象，设置参数包括：10 个估计器、启用热启动、使用自助法进行采样、计算 out-of-bag 分数
    est = ForestEstimator(
        n_estimators=10, warm_start=True, bootstrap=True, oob_score=True
    )

    # 使用 patch.object 方法替换 est 对象中的 _set_oob_score_and_attributes 方法，并使用 wraps 参数保留原始方法的包装
    with patch.object(
        est, "_set_oob_score_and_attributes", wraps=est._set_oob_score_and_attributes
    ) as mock_set_oob_score_and_attributes:
        # 调用 est 对象的 fit 方法，对给定的特征数据 X 和目标数据 y 进行拟合
        est.fit(X, y)

        # 使用 pytest.warns 检查是否会发出 UserWarning，且匹配指定的字符串 "Warm-start fitting without increasing"
        with pytest.warns(UserWarning, match="Warm-start fitting without increasing"):
            # 再次调用 est 对象的 fit 方法，对相同的特征数据 X 和目标数据 y 进行拟合
            est.fit(X, y)

        # 断言 mock_set_oob_score_and_attributes 方法仅被调用一次
        mock_set_oob_score_and_attributes.assert_called_once()
# 定义一个测试函数，用于测试数据类型转换是否正确
def test_dtype_convert(n_classes=15):
    # 创建一个随机森林分类器对象，设置随机种子，禁用自举采样
    classifier = RandomForestClassifier(random_state=0, bootstrap=False)

    # 创建一个 n_classes × n_classes 的对角矩阵，作为特征矩阵 X
    X = np.eye(n_classes)
    # 创建一个包含 n_classes 个字母的列表，作为目标标签 y
    y = [ch for ch in "ABCDEFGHIJKLMNOPQRSTU"[:n_classes]]

    # 使用 X 和 y 进行训练，并对 X 进行预测
    result = classifier.fit(X, y).predict(X)
    # 断言分类器的类别属性与 y 相等
    assert_array_equal(classifier.classes_, y)
    # 断言预测结果与 y 相等
    assert_array_equal(result, y)


# 使用 pytest 的参数化装饰器，测试不同的森林估计器
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_decision_path(name):
    # 获取数据集 X 和标签 y
    X, y = hastie_X, hastie_y
    # 获取对应名称的森林估计器类
    ForestEstimator = FOREST_ESTIMATORS[name]
    # 创建该类的估计器对象，设置参数并进行拟合
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=False, random_state=1)
    est.fit(X, y)
    # 获取决策路径指示器和节点指针
    indicator, n_nodes_ptr = est.decision_path(X)

    # 断言指示器的形状与节点指针的最后一个元素相等
    assert indicator.shape[1] == n_nodes_ptr[-1]
    # 断言指示器的行数与样本数相等
    assert indicator.shape[0] == n_samples
    # 断言节点指针的差分与每个估计器的树的节点数相等
    assert_array_equal(
        np.diff(n_nodes_ptr), [e.tree_.node_count for e in est.estimators_]
    )

    # 断言叶子节点索引的正确性
    leaves = est.apply(X)
    for est_id in range(leaves.shape[1]):
        leave_indicator = [
            indicator[i, n_nodes_ptr[est_id] + j]
            for i, j in enumerate(leaves[:, est_id])
        ]
        assert_array_almost_equal(leave_indicator, np.ones(shape=n_samples))


# 测试最小不纯度减少参数是否被正确传递
def test_min_impurity_decrease():
    # 创建 Hastie 数据集，包含 100 个样本
    X, y = datasets.make_hastie_10_2(n_samples=100, random_state=1)
    # 定义所有的森林估计器类
    all_estimators = [
        RandomForestClassifier,
        RandomForestRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
    ]

    # 遍历所有估计器类
    for Estimator in all_estimators:
        # 创建具有指定最小不纯度减少参数的估计器对象
        est = Estimator(min_impurity_decrease=0.1)
        # 使用数据集 X 和 y 进行拟合
        est.fit(X, y)
        # 遍历每棵树并断言其最小不纯度减少参数为 0.1
        for tree in est.estimators_:
            assert tree.min_impurity_decrease == 0.1


# 测试泊松回归中 y 值的正性检查
def test_poisson_y_positive_check():
    # 创建一个泊松回归的随机森林回归器
    est = RandomForestRegressor(criterion="poisson")
    # 创建一个 3×3 的零矩阵作为特征矩阵 X
    X = np.zeros((3, 3))

    # 定义两种不合法的 y 值情况
    y = [-1, 1, 3]
    err_msg = (
        r"Some value\(s\) of y are negative which is "
        r"not allowed for Poisson regression."
    )
    # 使用 pytest 的断言检查预期异常信息是否被触发
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)

    y = [0, 0, 0]
    err_msg = (
        r"Sum of y is not strictly positive which "
        r"is necessary for Poisson regression."
    )
    # 使用 pytest 的断言检查预期异常信息是否被触发
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


# 自定义后台并注册为并行后台测试
# mypy 错误：变量 "DEFAULT_JOBLIB_BACKEND" 不是有效类型
class MyBackend(DEFAULT_JOBLIB_BACKEND):  # type: ignore
    def __init__(self, *args, **kwargs):
        self.count = 0
        super().__init__(*args, **kwargs)

    # 开始调用时计数增加
    def start_call(self):
        self.count += 1
        return super().start_call()


# 将自定义后台注册为并行后台
joblib.register_parallel_backend("testing", MyBackend)


# 测试后台是否被正确地尊重
@skip_if_no_parallel
def test_backend_respected():
    # 创建包含 10 棵树和 2 个作业的随机森林分类器
    clf = RandomForestClassifier(n_estimators=10, n_jobs=2)

    # 使用自定义的并行后台进行测试
    with joblib.parallel_backend("testing") as (ba, n_jobs):
        clf.fit(X, y)

    # 断言至少有一个调用被执行
    assert ba.count > 0
    # 使用 joblib.parallel_backend() 来设置并行计算的背景，此处使用了名为 "testing" 的背景
    with joblib.parallel_backend("testing") as (ba, _):
        # 在设置的并行计算背景下，调用 clf 对象的 predict_proba 方法进行预测概率计算
        clf.predict_proba(X)

    # 断言语句，用于验证在并行计算背景中没有任何任务被执行
    assert ba.count == 0
# 定义一个测试函数，用于验证随机森林分类器的特征重要性之和是否接近1
def test_forest_feature_importances_sum():
    # 生成一个具有指定特征数量和随机种子的分类数据集
    X, y = make_classification(
        n_samples=15, n_informative=3, random_state=1, n_classes=3
    )
    # 构建随机森林分类器，设定最小叶子节点样本数和随机种子，并进行拟合
    clf = RandomForestClassifier(
        min_samples_leaf=5, random_state=42, n_estimators=200
    ).fit(X, y)
    # 断言随机森林特征重要性之和是否接近1
    assert math.isclose(1, clf.feature_importances_.sum(), abs_tol=1e-7)


# 定义一个测试函数，验证随机森林回归器的特征重要性是否为全零向量
def test_forest_degenerate_feature_importances():
    # 创建一个全零的数据集
    X = np.zeros((10, 10))
    y = np.ones((10,))
    # 构建随机森林回归器，设定树的数量，并进行拟合
    gbr = RandomForestRegressor(n_estimators=10).fit(X, y)
    # 断言随机森林特征重要性是否为全零向量
    assert_array_equal(gbr.feature_importances_, np.zeros(10, dtype=np.float64))


# 使用pytest的参数化装饰器，对随机森林分类器和回归器进行参数化测试
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_max_samples_bootstrap(name):
    # 创建一个指定算法名称的分类器或回归器实例，设定bootstrap=False和max_samples=0.5
    est = FOREST_CLASSIFIERS_REGRESSORS[name](bootstrap=False, max_samples=0.5)
    # 定义错误消息字符串
    err_msg = (
        r"`max_sample` cannot be set if `bootstrap=False`. "
        r"Either switch to `bootstrap=True` or set "
        r"`max_sample=None`."
    )
    # 使用pytest的断言来验证是否抛出指定错误类型和错误消息
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


# 使用pytest的参数化装饰器，对随机森林分类器和回归器进行参数化测试
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS_REGRESSORS)
def test_large_max_samples_exception(name):
    # 创建一个指定算法名称的分类器或回归器实例，设定bootstrap=True和max_samples=int(1e9)
    est = FOREST_CLASSIFIERS_REGRESSORS[name](bootstrap=True, max_samples=int(1e9))
    # 定义匹配字符串
    match = "`max_samples` must be <= n_samples=6 but got value 1000000000"
    # 使用pytest的断言来验证是否抛出指定错误类型和匹配的错误消息
    with pytest.raises(ValueError, match=match):
        est.fit(X, y)


# 使用pytest的参数化装饰器，对随机森林回归器进行参数化测试
@pytest.mark.parametrize("name", FOREST_REGRESSORS)
def test_max_samples_boundary_regressors(name):
    # 划分回归数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, train_size=0.7, test_size=0.3, random_state=0
    )
    # 创建两个不同`max_samples`参数的回归器实例并拟合数据，计算均方误差
    ms_1_model = FOREST_REGRESSORS[name](
        bootstrap=True, max_samples=1.0, random_state=0
    )
    ms_1_predict = ms_1_model.fit(X_train, y_train).predict(X_test)

    ms_None_model = FOREST_REGRESSORS[name](
        bootstrap=True, max_samples=None, random_state=0
    )
    ms_None_predict = ms_None_model.fit(X_train, y_train).predict(X_test)

    # 使用pytest的断言验证两种`max_samples`参数下的均方误差是否接近
    ms_1_ms = mean_squared_error(ms_1_predict, y_test)
    ms_None_ms = mean_squared_error(ms_None_predict, y_test)
    assert ms_1_ms == pytest.approx(ms_None_ms)


# 使用pytest的参数化装饰器，对随机森林分类器进行参数化测试
@pytest.mark.parametrize("name", FOREST_CLASSIFIERS)
def test_max_samples_boundary_classifiers(name):
    # 划分分类数据集为训练集和测试集
    X_train, X_test, y_train, _ = train_test_split(
        X_large, y_large, random_state=0, stratify=y_large
    )
    # 创建两个不同`max_samples`参数的分类器实例并拟合数据，计算预测概率
    ms_1_model = FOREST_CLASSIFIERS[name](
        bootstrap=True, max_samples=1.0, random_state=0
    )
    ms_1_proba = ms_1_model.fit(X_train, y_train).predict_proba(X_test)

    ms_None_model = FOREST_CLASSIFIERS[name](
        bootstrap=True, max_samples=None, random_state=0
    )
    ms_None_proba = ms_None_model.fit(X_train, y_train).predict_proba(X_test)

    # 使用numpy的断言验证两种`max_samples`参数下的预测概率是否接近
    np.testing.assert_allclose(ms_1_proba, ms_None_proba)


# 使用pytest的参数化装饰器，对压缩稀疏行矩阵数据结构进行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_forest_y_sparse(csr_container):
    # 创建一个压缩稀疏行矩阵的示例数据
    X = [[1, 2, 3]]
    # 创建一个稀疏矩阵容器 `y`，其中包含一个 CSR 格式的稀疏矩阵，内容为 [[4, 5, 6]]
    y = csr_container([[4, 5, 6]])
    
    # 初始化一个随机森林分类器 `est`
    est = RandomForestClassifier()
    
    # 定义一条错误消息，用于匹配异常信息
    msg = "sparse multilabel-indicator for y is not supported."
    
    # 使用 `pytest.raises` 上下文管理器来捕获预期的 ValueError 异常，并匹配错误消息 `msg`
    with pytest.raises(ValueError, match=msg):
        # 调用随机森林分类器的 `fit` 方法，尝试拟合输入数据 `X` 和目标变量 `y`
        est.fit(X, y)
@pytest.mark.parametrize("ForestClass", [RandomForestClassifier, RandomForestRegressor])
def test_little_tree_with_small_max_samples(ForestClass):
    # 使用 pytest 的参数化功能，分别测试 RandomForestClassifier 和 RandomForestRegressor

    rng = np.random.RandomState(1)
    # 创建一个伪随机数生成器实例，种子为1

    X = rng.randn(10000, 2)
    # 生成一个形状为 (10000, 2) 的随机数组 X

    y = rng.randn(10000) > 0
    # 生成一个长度为 10000 的随机布尔数组 y

    # First fit with no restriction on max samples
    # 第一次拟合，不限制 max samples
    est1 = ForestClass(
        n_estimators=1,
        random_state=rng,
        max_samples=None,
    )

    # Second fit with max samples restricted to just 2
    # 第二次拟合，限制 max samples 为 2
    est2 = ForestClass(
        n_estimators=1,
        random_state=rng,
        max_samples=2,
    )

    est1.fit(X, y)
    # 使用 X 和 y 拟合第一个估计器

    est2.fit(X, y)
    # 使用 X 和 y 拟合第二个估计器

    tree1 = est1.estimators_[0].tree_
    # 获取第一个估计器的第一个树的树结构

    tree2 = est2.estimators_[0].tree_
    # 获取第二个估计器的第一个树的树结构

    msg = "Tree without `max_samples` restriction should have more nodes"
    # 消息：没有限制 `max_samples` 的树应该有更多的节点
    assert tree1.node_count > tree2.node_count, msg
    # 断言：第一个树的节点数量应该大于第二个树的节点数量


@pytest.mark.parametrize("Forest", FOREST_REGRESSORS)
def test_mse_criterion_object_segfault_smoke_test(Forest):
    # 这是一个烟雾测试，确保在使用可变标准时不会在并发线程中导致段错误
    # 非回归测试：https://github.com/scikit-learn/scikit-learn/issues/12623

    from sklearn.tree._criterion import MSE
    # 导入 MSE 标准

    y = y_reg.reshape(-1, 1)
    # 将 y_reg 重塑为列向量

    n_samples, n_outputs = y.shape
    # 获取 y 的样本数和输出数

    mse_criterion = MSE(n_outputs, n_samples)
    # 使用 MSE 标准初始化一个对象

    est = FOREST_REGRESSORS[Forest](n_estimators=2, n_jobs=2, criterion=mse_criterion)
    # 使用给定的回归器类型创建一个估计器对象，设置估计器数量和并行工作数，并使用 mse_criterion 标准

    est.fit(X_reg, y)
    # 使用 X_reg 和 y 拟合估计器


def test_random_trees_embedding_feature_names_out():
    """Check feature names out for Random Trees Embedding."""
    # 检查 Random Trees Embedding 的特征输出名称

    random_state = np.random.RandomState(0)
    # 创建一个伪随机数生成器实例，种子为 0

    X = np.abs(random_state.randn(100, 4))
    # 生成一个形状为 (100, 4) 的随机数组 X，取绝对值

    hasher = RandomTreesEmbedding(
        n_estimators=2, max_depth=2, sparse_output=False, random_state=0
    ).fit(X)
    # 使用 RandomTreesEmbedding 初始化一个对象，并拟合 X 数据

    names = hasher.get_feature_names_out()
    # 获取特征输出的名称

    expected_names = [
        f"randomtreesembedding_{tree}_{leaf}"
        # 预期的特征名称列表，格式为 "randomtreesembedding_tree_leaf"
        for tree, leaf in [
            (0, 2),
            (0, 3),
            (0, 5),
            (0, 6),
            (1, 2),
            (1, 3),
            (1, 5),
            (1, 6),
        ]
    ]

    assert_array_equal(expected_names, names)
    # 断言：预期的名称列表应与实际获取的名称列表相等


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_read_only_buffer(csr_container, monkeypatch):
    # 测试只读稀疏数据上的 RandomForestClassifier 的工作性能

    # 非回归测试：https://github.com/scikit-learn/scikit-learn/issues/25333
    monkeypatch.setattr(
        sklearn.ensemble._forest,
        "Parallel",
        partial(Parallel, max_nbytes=100),
    )
    # 使用 monkeypatch 修改 sklearn.ensemble._forest 中的 "Parallel" 属性为一个部分应用的 Parallel 对象，设置最大字节为 100

    rng = np.random.RandomState(seed=0)
    # 创建一个伪随机数生成器实例，种子为 0

    X, y = make_classification(n_samples=100, n_features=200, random_state=rng)
    # 生成一个分类数据集 X 和对应的标签 y，使用给定的随机数生成器实例

    X = csr_container(X, copy=True)
    # 将 X 转换为给定的 csr_container 类型的数据结构，进行复制以确保只读性

    clf = RandomForestClassifier(n_jobs=2, random_state=rng)
    # 使用给定的随机数生成器实例创建一个 RandomForestClassifier 对象，设置并行工作数为 2

    cross_val_score(clf, X, y, cv=2)
    # 使用交叉验证评估分类器在 X 和 y 上的性能


@pytest.mark.parametrize("class_weight", ["balanced_subsample", None])
# 定义测试函数，检查当样本数过低时是否将其四舍五入为一。

# 非回归测试，针对 gh-24037。
def test_round_samples_to_one_when_samples_too_low(class_weight):
    # 加载 Wine 数据集，返回特征 X 和目标 y
    X, y = datasets.load_wine(return_X_y=True)
    
    # 创建随机森林分类器对象
    forest = RandomForestClassifier(
        n_estimators=10,  # 使用 10 个基学习器
        max_samples=1e-4,  # 设置最大样本数为 0.0001
        class_weight=class_weight,  # 设置类别权重
        random_state=0  # 设置随机种子
    )
    
    # 对模型进行拟合
    forest.fit(X, y)


# 使用参数化测试装饰器，为 seed 参数传入值 [None, 1]
# 为 bootstrap 参数传入值 [True, False]
# 使用 FOREST_CLASSIFIERS_REGRESSORS 中的值对 ForestClass 参数进行参数化
@pytest.mark.parametrize("seed", [None, 1])
@pytest.mark.parametrize("bootstrap", [True, False])
@pytest.mark.parametrize("ForestClass", FOREST_CLASSIFIERS_REGRESSORS.values())
def test_estimators_samples(ForestClass, bootstrap, seed):
    # 使用 make_hastie_10_2 生成数据集，包含 200 个样本，设置随机种子为 1
    X, y = make_hastie_10_2(n_samples=200, random_state=1)

    # 根据 bootstrap 参数设置 max_samples 的值
    if bootstrap:
        max_samples = 0.5
    else:
        max_samples = None
    
    # 创建 ForestClass 类型的模型对象
    est = ForestClass(
        n_estimators=10,  # 使用 10 个基学习器
        max_samples=max_samples,  # 设置最大样本数
        max_features=0.5,  # 设置最大特征数
        random_state=seed,  # 设置随机种子
        bootstrap=bootstrap,  # 是否启用 bootstrap
    )
    
    # 对模型进行拟合
    est.fit(X, y)

    # 复制模型中的 estimators_samples_ 属性值
    estimators_samples = est.estimators_samples_.copy()

    # 测试多次调用结果是否一致
    assert_array_equal(estimators_samples, est.estimators_samples_)
    # 获取模型中的 estimators 属性值
    estimators = est.estimators_

    # 断言 estimators_samples 类型为 list
    assert isinstance(estimators_samples, list)
    # 断言 estimators_samples 长度与 estimators 相同
    assert len(estimators_samples) == len(estimators)
    # 断言 estimators_samples 中第一个元素的数据类型为 np.int32
    assert estimators_samples[0].dtype == np.int32

    # 遍历每个基学习器
    for i in range(len(estimators)):
        if bootstrap:
            # 断言 bootstrap 模式下，estimators_samples[i] 的长度为样本数的一半
            assert len(estimators_samples[i]) == len(X) // 2

            # 断言 bootstrap 应该是一种有放回的重新抽样
            assert len(np.unique(estimators_samples[i])) < len(estimators_samples[i])
        else:
            # 断言非 bootstrap 模式下，estimators_samples[i] 的集合长度应等于样本数
            assert len(set(estimators_samples[i])) == len(X)

    # 获取指定索引的基学习器的样本索引
    estimator_index = 0
    estimator_samples = estimators_samples[estimator_index]
    estimator = estimators[estimator_index]

    # 使用 estimator_samples 提取训练集
    X_train = X[estimator_samples]
    y_train = y[estimator_samples]

    # 获取原始树的值
    orig_tree_values = estimator.tree_.value
    # 克隆基学习器对象
    estimator = clone(estimator)
    # 对克隆后的基学习器对象进行拟合
    estimator.fit(X_train, y_train)
    # 获取新树的值
    new_tree_values = estimator.tree_.value
    # 断言原始树的值与新树的值在接近误差范围内相等
    assert_allclose(orig_tree_values, new_tree_values)


# 使用参数化测试装饰器，为 make_data 和 Forest 参数传入值
# 分别为 datasets.make_regression、RandomForestRegressor
# 和 datasets.make_classification、RandomForestClassifier
@pytest.mark.parametrize(
    "make_data, Forest",
    [
        (datasets.make_regression, RandomForestRegressor),
        (datasets.make_classification, RandomForestClassifier),
    ],
)
def test_missing_values_is_resilient(make_data, Forest):
    # 设置随机种子为 0
    rng = np.random.RandomState(0)
    # 定义样本数和特征数
    n_samples, n_features = 1000, 10
    # 使用 make_data 生成数据集
    X, y = make_data(n_samples=n_samples, n_features=n_features, random_state=rng)

    # 创建带有缺失值的数据集
    X_missing = X.copy()
    X_missing[rng.choice([False, True], size=X.shape, p=[0.95, 0.05])] = np.nan
    # 断言 X_missing 中存在 NaN 值
    assert np.isnan(X_missing).any()
    # 将原始数据集按照指定随机种子划分为训练集和测试集，包括带有缺失值的特征和对应的目标值
    X_missing_train, X_missing_test, y_train, y_test = train_test_split(
        X_missing, y, random_state=0
    )

    # 使用带有缺失值的特征训练随机森林模型
    forest_with_missing = Forest(random_state=rng, n_estimators=50)
    forest_with_missing.fit(X_missing_train, y_train)
    # 计算在测试集上带有缺失值的模型得分
    score_with_missing = forest_with_missing.score(X_missing_test, y_test)

    # 将原始数据集按照指定随机种子划分为训练集和测试集，不包括缺失值的特征和对应的目标值
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 使用不带缺失值的特征训练另一个随机森林模型
    forest = Forest(random_state=rng, n_estimators=50)
    forest.fit(X_train, y_train)
    # 计算在测试集上不带缺失值的模型得分
    score_without_missing = forest.score(X_test, y_test)

    # 断言带有缺失值的模型得分至少达到不带缺失值模型得分的80%
    assert score_with_missing >= 0.80 * score_without_missing
# 使用 pytest.mark.parametrize 装饰器来定义一个参数化测试函数，测试随机森林分类器和回归器
@pytest.mark.parametrize("Forest", [RandomForestClassifier, RandomForestRegressor])
def test_missing_value_is_predictive(Forest):
    """Check that the forest learns when missing values are only present for
    a predictive feature."""

    # 设定随机数生成器
    rng = np.random.RandomState(0)
    n_samples = 300

    # 生成非预测特征的随机数据
    X_non_predictive = rng.standard_normal(size=(n_samples, 10))
    y = rng.randint(0, high=2, size=n_samples)

    # 创建一个带有预测性特征的数据集，使用 y 并加入一些噪音
    X_random_mask = rng.choice([False, True], size=n_samples, p=[0.95, 0.05])
    y_mask = y.astype(bool)
    y_mask[X_random_mask] = ~y_mask[X_random_mask]

    predictive_feature = rng.standard_normal(size=n_samples)
    predictive_feature[y_mask] = np.nan
    assert np.isnan(predictive_feature).any()

    # 复制 X_non_predictive，并在第5列加入预测特征
    X_predictive = X_non_predictive.copy()
    X_predictive[:, 5] = predictive_feature

    # 使用 train_test_split 将数据集划分为训练集和测试集
    (
        X_predictive_train,
        X_predictive_test,
        X_non_predictive_train,
        X_non_predictive_test,
        y_train,
        y_test,
    ) = train_test_split(X_predictive, X_non_predictive, y, random_state=0)

    # 使用随机森林分类器或回归器拟合预测性特征数据集和非预测性特征数据集
    forest_predictive = Forest(random_state=0).fit(X_predictive_train, y_train)
    forest_non_predictive = Forest(random_state=0).fit(X_non_predictive_train, y_train)

    # 计算预测性测试集的得分
    predictive_test_score = forest_predictive.score(X_predictive_test, y_test)

    # 断言预测性测试得分至少为0.75
    assert predictive_test_score >= 0.75
    # 断言预测性测试得分至少不低于非预测性测试得分
    assert predictive_test_score >= forest_non_predictive.score(
        X_non_predictive_test, y_test
    )


# 定义测试函数，测试在存在缺失值时不支持的评估标准是否会引发错误
def test_non_supported_criterion_raises_error_with_missing_values():
    """Raise error for unsupported criterion when there are missing values."""

    # 创建带有缺失值的数据集 X 和目标值 y
    X = np.array([[0, 1, 2], [np.nan, 0, 2.0]])
    y = [0.5, 1.0]

    # 使用 criterion="absolute_error" 创建一个随机森林回归器对象
    forest = RandomForestRegressor(criterion="absolute_error")

    # 准备匹配的错误信息
    msg = "RandomForestRegressor does not accept missing values"

    # 使用 pytest 的 pytest.raises 断言来检查是否引发 ValueError，并且错误信息匹配预期的错误信息
    with pytest.raises(ValueError, match=msg):
        forest.fit(X, y)
```