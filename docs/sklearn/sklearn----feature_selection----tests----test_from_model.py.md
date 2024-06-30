# `D:\src\scipysrc\scikit-learn\sklearn\feature_selection\tests\test_from_model.py`

```
# 导入必要的模块和库
import re  # 导入正则表达式模块
import warnings  # 导入警告模块
from unittest.mock import Mock  # 从单元测试模块中导入 Mock 类

import numpy as np  # 导入 NumPy 库并重命名为 np
import pytest  # 导入 pytest 测试框架

from sklearn import datasets  # 导入 sklearn 中的 datasets 模块
from sklearn.base import BaseEstimator  # 从 sklearn 的 base 模块中导入 BaseEstimator 类
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression  # 导入交叉分解模型
from sklearn.datasets import make_friedman1  # 从 sklearn 的 datasets 模块中导入 make_friedman1 数据生成函数
from sklearn.decomposition import PCA  # 导入 PCA 主成分分析模块
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # 导入集成模型
from sklearn.exceptions import NotFittedError  # 导入未拟合错误类
from sklearn.feature_selection import SelectFromModel  # 导入基于模型的特征选择类
from sklearn.linear_model import (  # 导入线性模型
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    PassiveAggressiveClassifier,
    SGDClassifier,
)
from sklearn.pipeline import make_pipeline  # 导入管道构建函数
from sklearn.svm import LinearSVC  # 导入支持向量机模型中的线性支持向量分类器
from sklearn.utils._testing import (  # 导入测试工具函数
    MinimalClassifier,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    skip_if_32bit,
)


class NaNTag(BaseEstimator):
    def _more_tags(self):
        return {"allow_nan": True}  # 返回包含 "allow_nan" 为 True 的字典的方法


class NoNaNTag(BaseEstimator):
    def _more_tags(self):
        return {"allow_nan": False}  # 返回包含 "allow_nan" 为 False 的字典的方法


class NaNTagRandomForest(RandomForestClassifier):
    def _more_tags(self):
        return {"allow_nan": True}  # 返回包含 "allow_nan" 为 True 的字典的方法


iris = datasets.load_iris()  # 加载鸢尾花数据集
data, y = iris.data, iris.target  # 将数据集特征和目标分配给 data 和 y
rng = np.random.RandomState(0)  # 使用种子 0 初始化随机数生成器


def test_invalid_input():
    # 测试无效输入的函数
    clf = SGDClassifier(
        alpha=0.1, max_iter=10, shuffle=True, random_state=None, tol=None
    )  # 创建随机梯度下降分类器对象 clf
    for threshold in ["gobbledigook", ".5 * gobbledigook"]:
        # 循环遍历不同的阈值
        model = SelectFromModel(clf, threshold=threshold)  # 使用给定的阈值创建特征选择模型
        model.fit(data, y)  # 在数据上拟合模型
        with pytest.raises(ValueError):
            model.transform(data)  # 使用模型转换数据并验证是否引发 ValueError 异常


def test_input_estimator_unchanged():
    # 测试 SelectFromModel 是否适用于估计器的克隆版本
    est = RandomForestClassifier()  # 创建随机森林分类器对象 est
    transformer = SelectFromModel(estimator=est)  # 使用给定的估计器创建特征选择模型
    transformer.fit(data, y)  # 在数据上拟合模型
    assert transformer.estimator is est  # 验证模型的估计器属性是否为原始的估计器对象


@pytest.mark.parametrize(
    "max_features, err_type, err_msg",
    [  # 使用 pytest 的参数化标记定义多个测试参数
        (
            data.shape[1] + 1,
            ValueError,
            "max_features ==",  # 验证 max_features 是否为预期的异常类型和消息
        ),
        (
            lambda X: 1.5,
            TypeError,
            "max_features must be an instance of int, not float.",  # 验证 max_features 是否为预期的异常类型和消息
        ),
        (
            lambda X: data.shape[1] + 1,
            ValueError,
            "max_features ==",  # 验证 max_features 是否为预期的异常类型和消息
        ),
        (
            lambda X: -1,
            ValueError,
            "max_features ==",  # 验证 max_features 是否为预期的异常类型和消息
        ),
    ],
)
def test_max_features_error(max_features, err_type, err_msg):
    err_msg = re.escape(err_msg)  # 对错误消息进行转义以匹配正则表达式
    clf = RandomForestClassifier(n_estimators=5, random_state=0)  # 创建随机森林分类器对象 clf

    transformer = SelectFromModel(
        estimator=clf, max_features=max_features, threshold=-np.inf
    )  # 使用给定的参数创建特征选择模型
    with pytest.raises(err_type, match=err_msg):
        transformer.fit(data, y)  # 在数据上拟合模型并验证是否引发特定类型和消息的异常


@pytest.mark.parametrize("max_features", [0, 2, data.shape[1], None])
def test_inferred_max_features_integer(max_features):
    # 测试推断的 max_features 是否为整数
    """检查整数型 max_features_ 和输出形状。"""
    # 创建一个随机森林分类器，包含 5 棵树，使用固定的随机种子
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    # 创建一个 SelectFromModel 转换器，使用上面创建的分类器作为评估器，
    # max_features 参数用于指定最大特征数，threshold 参数设置为负无穷大
    transformer = SelectFromModel(
        estimator=clf, max_features=max_features, threshold=-np.inf
    )
    # 对输入数据进行转换，并返回转换后的数据 X_trans
    X_trans = transformer.fit_transform(data, y)
    
    # 如果 max_features 参数不为 None，则进行以下断言：
    if max_features is not None:
        # 断言转换器的 max_features_ 属性等于 max_features
        assert transformer.max_features_ == max_features
        # 断言转换后的数据 X_trans 的特征数等于转换器的 max_features_
        assert X_trans.shape[1] == transformer.max_features_
    else:
        # 如果 max_features 参数为 None，则进行以下断言：
        # 断言转换器没有 max_features_ 属性
        assert not hasattr(transformer, "max_features_")
        # 断言转换后的数据 X_trans 的特征数等于原始数据的特征数
        assert X_trans.shape[1] == data.shape[1]
@pytest.mark.parametrize(
    "max_features",
    [lambda X: 1, lambda X: X.shape[1], lambda X: min(X.shape[1], 10000)],
)
def test_inferred_max_features_callable(max_features):
    """Check max_features_ and output shape for callable max_features."""
    # 创建随机森林分类器对象
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    # 创建SelectFromModel对象，使用随机森林分类器作为评估器，可调用的max_features作为参数
    transformer = SelectFromModel(
        estimator=clf, max_features=max_features, threshold=-np.inf
    )
    # 对数据进行转换并拟合模型
    X_trans = transformer.fit_transform(data, y)
    # 断言max_features_属性与max_features函数在数据上的返回值相等
    assert transformer.max_features_ == max_features(data)
    # 断言转换后的数据形状的特征数与transformer的max_features_属性相等


@pytest.mark.parametrize("max_features", [lambda X: round(len(X[0]) / 2), 2])
def test_max_features_array_like(max_features):
    # 创建输入数据X和标签y
    X = [
        [0.87, -1.34, 0.31],
        [-2.79, -0.02, -0.85],
        [-1.34, -0.48, -2.55],
        [1.92, 1.48, 0.65],
    ]
    y = [0, 1, 0, 1]

    # 创建随机森林分类器对象
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    # 创建SelectFromModel对象，使用随机森林分类器作为评估器，max_features作为参数
    transformer = SelectFromModel(
        estimator=clf, max_features=max_features, threshold=-np.inf
    )
    # 对数据进行转换并拟合模型
    X_trans = transformer.fit_transform(X, y)
    # 断言转换后的数据形状的特征数与transformer的max_features_属性相等


@pytest.mark.parametrize(
    "max_features",
    [lambda X: min(X.shape[1], 10000), lambda X: X.shape[1], lambda X: 1],
)
def test_max_features_callable_data(max_features):
    """Tests that the callable passed to `fit` is called on X."""
    # 创建随机森林分类器对象
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    # 创建Mock对象用于模拟max_features的行为
    m = Mock(side_effect=max_features)
    # 创建SelectFromModel对象，使用随机森林分类器作为评估器，max_features作为参数
    transformer = SelectFromModel(estimator=clf, max_features=m, threshold=-np.inf)
    # 对数据进行转换并拟合模型
    transformer.fit_transform(data, y)
    # 断言Mock对象的调用参数与数据相等


class FixedImportanceEstimator(BaseEstimator):
    def __init__(self, importances):
        self.importances = importances

    def fit(self, X, y=None):
        self.feature_importances_ = np.array(self.importances)


def test_max_features():
    # Test max_features parameter using various values
    # 生成分类数据集X和标签y
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )
    # 获取特征数作为max_features参数
    max_features = X.shape[1]
    # 创建随机森林分类器对象
    est = RandomForestClassifier(n_estimators=50, random_state=0)

    # 创建SelectFromModel对象，使用随机森林分类器作为评估器，不设置max_features参数
    transformer1 = SelectFromModel(estimator=est, threshold=-np.inf)
    # 创建SelectFromModel对象，使用随机森林分类器作为评估器，max_features参数为特征数
    transformer2 = SelectFromModel(
        estimator=est, max_features=max_features, threshold=-np.inf
    )
    # 对数据进行转换并拟合模型
    X_new1 = transformer1.fit_transform(X, y)
    X_new2 = transformer2.fit_transform(X, y)
    # 断言两个转换结果的数据数组近似相等

    # 创建SelectFromModel对象，使用Lasso回归作为评估器
    transformer1 = SelectFromModel(estimator=Lasso(alpha=0.025, random_state=42))
    # 对数据进行转换并拟合模型
    X_new1 = transformer1.fit_transform(X, y)
    # 获取特征重要性分数的绝对值
    scores1 = np.abs(transformer1.estimator_.coef_)
    # 对重要性分数进行排序，获取候选特征索引
    candidate_indices1 = np.argsort(-scores1, kind="mergesort")
    # 对于每个特征数从1到X_new1.shape[1]的范围进行迭代
    for n_features in range(1, X_new1.shape[1] + 1):
        # 使用 Lasso 回归作为评估器，选择最重要的特征
        transformer2 = SelectFromModel(
            estimator=Lasso(alpha=0.025, random_state=42),
            max_features=n_features,
            threshold=-np.inf,
        )
        # 从输入数据 X 中选择最重要的特征，并转换数据集
        X_new2 = transformer2.fit_transform(X, y)
        # 获取选择模型的系数的绝对值
        scores2 = np.abs(transformer2.estimator_.coef_)
        # 按特征重要性排序特征的索引
        candidate_indices2 = np.argsort(-scores2, kind="mergesort")
        # 断言所选的特征在两个变换器中是相同的
        assert_allclose(
            X[:, candidate_indices1[:n_features]], X[:, candidate_indices2[:n_features]]
        )
    # 断言第一个和第二个变换器的估计器具有相同的系数
    assert_allclose(transformer1.estimator_.coef_, transformer2.estimator_.coef_)
# 测试 max_features 是否能够在特征重要性相同时进行选择
def test_max_features_tiebreak():
    # 生成一个分类数据集 X 和对应的标签 y
    X, y = datasets.make_classification(
        n_samples=1000,  # 样本数为 1000
        n_features=10,   # 特征数为 10
        n_informative=3, # 有信息特征数为 3
        n_redundant=0,   # 无冗余特征
        n_repeated=0,    # 无重复特征
        shuffle=False,   # 不打乱数据
        random_state=0,  # 随机种子设为 0
    )
    max_features = X.shape[1]  # 最大特征数为 X 的列数

    # 模拟特征重要性
    feature_importances = np.array([4, 4, 4, 4, 3, 3, 3, 2, 2, 1])
    # 从 1 到 max_features 遍历
    for n_features in range(1, max_features + 1):
        # 使用自定义的 FixedImportanceEstimator 和当前的 n_features 创建特征选择器
        transformer = SelectFromModel(
            FixedImportanceEstimator(feature_importances),
            max_features=n_features,  # 最大特征数设置为 n_features
            threshold=-np.inf,         # 阈值设为负无穷
        )
        # 对数据进行拟合转换
        X_new = transformer.fit_transform(X, y)
        # 获取选择的特征索引
        selected_feature_indices = np.where(transformer._get_support_mask())[0]
        # 断言所选特征索引与期望的范围一致
        assert_array_equal(selected_feature_indices, np.arange(n_features))
        # 断言转换后的数据形状中特征数与 n_features 相同
        assert X_new.shape[1] == n_features


# 测试阈值和最大特征数同时使用的情况
def test_threshold_and_max_features():
    # 生成一个分类数据集 X 和对应的标签 y
    X, y = datasets.make_classification(
        n_samples=1000,  # 样本数为 1000
        n_features=10,   # 特征数为 10
        n_informative=3, # 有信息特征数为 3
        n_redundant=0,   # 无冗余特征
        n_repeated=0,    # 无重复特征
        shuffle=False,   # 不打乱数据
        random_state=0,  # 随机种子设为 0
    )
    est = RandomForestClassifier(n_estimators=50, random_state=0)

    # 创建特征选择器 transformer1，设定 max_features=3, threshold=-np.inf
    transformer1 = SelectFromModel(estimator=est, max_features=3, threshold=-np.inf)
    # 对数据进行拟合转换
    X_new1 = transformer1.fit_transform(X, y)

    # 创建特征选择器 transformer2，设定 threshold=0.04
    transformer2 = SelectFromModel(estimator=est, threshold=0.04)
    # 对数据进行拟合转换
    X_new2 = transformer2.fit_transform(X, y)

    # 创建特征选择器 transformer3，设定 max_features=3, threshold=0.04
    transformer3 = SelectFromModel(estimator=est, max_features=3, threshold=0.04)
    # 对数据进行拟合转换
    X_new3 = transformer3.fit_transform(X, y)
    # 断言 X_new3 的特征数等于 X_new1 和 X_new2 中较小的特征数
    assert X_new3.shape[1] == min(X_new1.shape[1], X_new2.shape[1])
    # 对输入数据的特征索引进行转换，断言转换后的数据与原始数据 X 中相应的列相似
    selected_indices = transformer3.transform(np.arange(X.shape[1])[np.newaxis, :])
    assert_allclose(X_new3, X[:, selected_indices[0]])


# 测试样本权重是否传递到基础估算器
def test_sample_weight():
    # 生成一个分类数据集 X 和对应的标签 y
    X, y = datasets.make_classification(
        n_samples=100,    # 样本数为 100
        n_features=10,    # 特征数为 10
        n_informative=3,  # 有信息特征数为 3
        n_redundant=0,    # 无冗余特征
        n_repeated=0,     # 无重复特征
        shuffle=False,    # 不打乱数据
        random_state=0,   # 随机种子设为 0
    )

    # 检查使用样本权重时的情况
    # 创建一个与 y 形状相同的数组，所有元素初始化为 1
    sample_weight = np.ones(y.shape)
    
    # 将 y 等于 1 的位置对应的 sample_weight 元素乘以 100
    sample_weight[y == 1] *= 100
    
    # 使用 LogisticRegression 进行分类器的初始化，设置随机种子为 0，且不包含截距
    est = LogisticRegression(random_state=0, fit_intercept=False)
    
    # 使用 SelectFromModel 对象进行特征选择的初始化，使用 est 作为评估器
    transformer = SelectFromModel(estimator=est)
    
    # 使用无权重的方式拟合 transformer 对象，进行特征选择
    transformer.fit(X, y, sample_weight=None)
    
    # 获取特征选择后的掩码（mask）
    mask = transformer._get_support_mask()
    
    # 使用带权重的方式重新拟合 transformer 对象，进行特征选择
    transformer.fit(X, y, sample_weight=sample_weight)
    
    # 获取带权重特征选择后的掩码（mask）
    weighted_mask = transformer._get_support_mask()
    
    # 断言带权重特征选择后的掩码与无权重时不完全相同
    assert not np.all(weighted_mask == mask)
    
    # 使用进一步加权的方式重新拟合 transformer 对象，进行特征选择
    transformer.fit(X, y, sample_weight=3 * sample_weight)
    
    # 获取再次加权特征选择后的掩码（mask）
    reweighted_mask = transformer._get_support_mask()
    
    # 断言带进一步加权特征选择后的掩码与原始带权重时完全相同
    assert np.all(weighted_mask == reweighted_mask)
@pytest.mark.parametrize(
    "estimator",
    [
        Lasso(alpha=0.1, random_state=42),  # 创建 Lasso 模型实例，设置 alpha 和随机种子
        LassoCV(random_state=42),  # 创建 LassoCV 模型实例，设置随机种子
        ElasticNet(l1_ratio=1, random_state=42),  # 创建 ElasticNet 模型实例，设置 l1_ratio 和随机种子
        ElasticNetCV(l1_ratio=[1], random_state=42),  # 创建 ElasticNetCV 模型实例，设置 l1_ratio 和随机种子
    ],
)
def test_coef_default_threshold(estimator):
    X, y = datasets.make_classification(
        n_samples=100,  # 样本数量
        n_features=10,  # 特征数量
        n_informative=3,  # 信息特征数量
        n_redundant=0,  # 冗余特征数量
        n_repeated=0,  # 重复特征数量
        shuffle=False,  # 不打乱数据
        random_state=0,  # 随机种子
    )

    # 对于 Lasso 和相关模型，默认阈值为 1e-5
    transformer = SelectFromModel(estimator=estimator)
    transformer.fit(X, y)  # 使用模型拟合数据
    X_new = transformer.transform(X)  # 对数据进行转换
    mask = np.abs(transformer.estimator_.coef_) > 1e-5  # 根据阈值生成特征掩码
    assert_array_almost_equal(X_new, X[:, mask])  # 断言转换后的数据与预期的数据一致


@skip_if_32bit
def test_2d_coef():
    X, y = datasets.make_classification(
        n_samples=1000,  # 样本数量
        n_features=10,  # 特征数量
        n_informative=3,  # 信息特征数量
        n_redundant=0,  # 冗余特征数量
        n_repeated=0,  # 重复特征数量
        shuffle=False,  # 不打乱数据
        random_state=0,  # 随机种子
        n_classes=4,  # 类别数量
    )

    est = LogisticRegression()  # 创建逻辑回归模型实例
    for threshold, func in zip(["mean", "median"], [np.mean, np.median]):
        for order in [1, 2, np.inf]:
            # 对于多类问题，使用 SelectFromModel 进行拟合
            transformer = SelectFromModel(
                estimator=LogisticRegression(), threshold=threshold, norm_order=order
            )
            transformer.fit(X, y)  # 使用模型拟合数据
            assert hasattr(transformer.estimator_, "coef_")  # 断言模型具有 coef_ 属性
            X_new = transformer.transform(X)  # 对数据进行转换
            assert X_new.shape[1] < X.shape[1]  # 断言转换后的特征数小于原始特征数

            # 手动检查是否正确执行了范数计算
            est.fit(X, y)  # 使用逻辑回归模型拟合数据
            importances = np.linalg.norm(est.coef_, axis=0, ord=order)  # 计算特征重要性的范数
            feature_mask = importances > func(importances)  # 基于给定函数计算特征掩码
            assert_array_almost_equal(X_new, X[:, feature_mask])  # 断言转换后的数据与预期的数据一致


def test_partial_fit():
    est = PassiveAggressiveClassifier(  # 创建 PassiveAggressiveClassifier 模型实例
        random_state=0, shuffle=False, max_iter=5, tol=None
    )
    transformer = SelectFromModel(estimator=est)  # 使用模型创建 SelectFromModel 实例
    transformer.partial_fit(data, y, classes=np.unique(y))  # 部分拟合模型
    old_model = transformer.estimator_  # 获取拟合后的模型
    transformer.partial_fit(data, y, classes=np.unique(y))  # 再次部分拟合模型
    new_model = transformer.estimator_  # 获取更新后的模型
    assert old_model is new_model  # 断言两个模型对象相同

    X_transform = transformer.transform(data)  # 对数据进行转换
    transformer.fit(np.vstack((data, data)), np.concatenate((y, y)))  # 使用堆叠后的数据进行拟合
    assert_array_almost_equal(X_transform, transformer.transform(data))  # 断言转换后的数据与预期的数据一致

    # 检查如果 est 没有 partial_fit 方法，SelectFromModel 也不应有
    transformer = SelectFromModel(estimator=RandomForestClassifier())  # 使用随机森林模型创建 SelectFromModel 实例
    assert not hasattr(transformer, "partial_fit")  # 断言 SelectFromModel 实例没有 partial_fit 方法


def test_calling_fit_reinitializes():
    est = LinearSVC(random_state=0)  # 创建 LinearSVC 模型实例
    transformer = SelectFromModel(estimator=est)  # 使用模型创建 SelectFromModel 实例
    transformer.fit(data, y)  # 使用数据拟合模型
    transformer.set_params(estimator__C=100)  # 设置模型参数 C 为 100
    transformer.fit(data, y)  # 重新使用数据拟合模型
    assert transformer.estimator_.C == 100  # 断言模型的参数 C 已更新为 100


def test_prefit():
    # 此测试用例未完整提供，无法添加注释
    # Test all possible combinations of the prefit parameter.

    # Passing a prefit parameter with the selected model
    # and fitting an unfit model with prefit=False should give the same results.
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, random_state=0, tol=None)
    # Initialize SelectFromModel with the classifier
    model = SelectFromModel(clf)
    # Fit the model with data
    model.fit(data, y)
    # Transform the original data based on the fitted model
    X_transform = model.transform(data)
    # Fit the classifier directly with data
    clf.fit(data, y)
    # Initialize SelectFromModel with the pre-fitted classifier
    model = SelectFromModel(clf, prefit=True)
    # Assert that the transformed data from both methods are almost equal
    assert_array_almost_equal(model.transform(data), X_transform)
    # Fit the model again with data
    model.fit(data, y)
    # Assert that the estimator of the model is not the same object as the classifier
    assert model.estimator_ is not clf

    # Check that the model is rewritten if prefit=False and a fitted model is
    # passed
    model = SelectFromModel(clf, prefit=False)
    # Fit the model with data
    model.fit(data, y)
    # Assert that the transformed data from both methods are almost equal
    assert_array_almost_equal(model.transform(data), X_transform)

    # Check that passing an unfitted estimator with `prefit=True` raises a
    # `ValueError`
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, random_state=0, tol=None)
    model = SelectFromModel(clf, prefit=True)
    # Define the error message expected from the `pytest.raises`
    err_msg = "When `prefit=True`, `estimator` is expected to be a fitted estimator."
    # Check that calling `fit`, `partial_fit`, or `transform` raises `NotFittedError`
    with pytest.raises(NotFittedError, match=err_msg):
        model.fit(data, y)
    with pytest.raises(NotFittedError, match=err_msg):
        model.partial_fit(data, y)
    with pytest.raises(NotFittedError, match=err_msg):
        model.transform(data)

    # Check that the internal parameters of a pre-fitted model are not changed
    # when calling `fit` or `partial_fit` with `prefit=True`
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, tol=None).fit(data, y)
    model = SelectFromModel(clf, prefit=True)
    # Fit the model with data
    model.fit(data, y)
    # Assert that the coefficients of the estimator and the classifier are almost equal
    assert_allclose(model.estimator_.coef_, clf.coef_)
    # Partially fit the model with data
    model.partial_fit(data, y)
    # Assert that the coefficients of the estimator and the classifier are almost equal
    assert_allclose(model.estimator_.coef_, clf.coef_)
def test_prefit_max_features():
    """检查`prefit`和`max_features`之间的交互作用。"""
    # case 1: 如果没有调用`fit`来验证属性，则在`transform`时应该引发错误
    estimator = RandomForestClassifier(n_estimators=5, random_state=0)
    estimator.fit(data, y)
    model = SelectFromModel(estimator, prefit=True, max_features=lambda X: X.shape[1])

    err_msg = (
        "当`prefit=True`且`max_features`是可调用对象时，在调用`transform`之前调用`fit`。"
    )
    with pytest.raises(NotFittedError, match=err_msg):
        model.transform(data)

    # case 2: `max_features`未经验证且与整数不同
    # FIXME: 我们无法在transform时验证属性的上限，如果我们打算强制属性具有这样的上限，应该强制调用`fit`。
    max_features = 2.5
    model.set_params(max_features=max_features)
    with pytest.raises(ValueError, match="`max_features`必须是整数"):
        model.transform(data)


def test_prefit_get_feature_names_out():
    """检查`prefit`和特征名称之间的交互作用。"""
    clf = RandomForestClassifier(n_estimators=2, random_state=0)
    clf.fit(data, y)
    model = SelectFromModel(clf, prefit=True, max_features=1)

    name = type(model).__name__
    err_msg = (
        f"此{name}实例尚未拟合。在使用此估计器之前，请调用带有适当参数的'fit'。"
    )
    with pytest.raises(NotFittedError, match=err_msg):
        model.get_feature_names_out()

    model.fit(data, y)
    feature_names = model.get_feature_names_out()
    assert feature_names == ["x3"]


def test_threshold_string():
    est = RandomForestClassifier(n_estimators=50, random_state=0)
    model = SelectFromModel(est, threshold="0.5*mean")
    model.fit(data, y)
    X_transform = model.transform(data)

    # 直接从估计器中计算阈值。
    est.fit(data, y)
    threshold = 0.5 * np.mean(est.feature_importances_)
    mask = est.feature_importances_ > threshold
    assert_array_almost_equal(X_transform, data[:, mask])


def test_threshold_without_refitting():
    # 测试可以在不重新拟合模型的情况下设置阈值。
    clf = SGDClassifier(alpha=0.1, max_iter=10, shuffle=True, random_state=0, tol=None)
    model = SelectFromModel(clf, threshold="0.1 * mean")
    model.fit(data, y)
    X_transform = model.transform(data)

    # 设置更高的阈值以过滤更多特征。
    model.threshold = "1.0 * mean"
    assert X_transform.shape[1] > model.transform(data).shape[1]


def test_fit_accepts_nan_inf():
    # 测试`fit`不检查np.inf和np.nan值。
    clf = HistGradientBoostingClassifier(random_state=0)

    model = SelectFromModel(estimator=clf)

    nan_data = data.copy()
    nan_data[0] = np.nan
    nan_data[1] = np.inf

    model.fit(data, y)


def test_transform_accepts_nan_inf():
    # 测试转换是否检查 np.inf 和 np.nan 值。
    clf = NaNTagRandomForest(n_estimators=100, random_state=0)
    # 复制数据以防修改原始数据
    nan_data = data.copy()
    
    # 创建 SelectFromModel 对象，使用给定的 NaNTagRandomForest 评估器
    model = SelectFromModel(estimator=clf)
    # 使用数据 nan_data 和目标变量 y 来拟合模型
    model.fit(nan_data, y)
    
    # 在数据中人为地引入 NaN 和 Inf 值
    nan_data[0] = np.nan
    nan_data[1] = np.inf
    
    # 对 nan_data 进行转换，检查模型的选择特征
    model.transform(nan_data)
# 测试函数，验证 allow_nan_estimator 的行为是否正确设置为允许 NaN
def test_allow_nan_tag_comes_from_estimator():
    # 创建一个 NaNTag 的实例
    allow_nan_est = NaNTag()
    # 使用 SelectFromModel 包装 allow_nan_est 模型
    model = SelectFromModel(estimator=allow_nan_est)
    # 断言模型的标签中 allow_nan 的值为 True
    assert model._get_tags()["allow_nan"] is True

    # 创建一个 NoNaNTag 的实例
    no_nan_est = NoNaNTag()
    # 使用 SelectFromModel 包装 no_nan_est 模型
    model = SelectFromModel(estimator=no_nan_est)
    # 断言模型的标签中 allow_nan 的值为 False
    assert model._get_tags()["allow_nan"] is False


# 返回 PCA 估计器的特征重要性，即其解释的方差绝对值
def _pca_importances(pca_estimator):
    return np.abs(pca_estimator.explained_variance_)


# 参数化测试，验证 importance_getter 在不同情况下的正确性
@pytest.mark.parametrize(
    "estimator, importance_getter",
    [
        (
            # 创建一个包含 PCA 和 LogisticRegression 的流水线
            make_pipeline(PCA(random_state=0), LogisticRegression()),
            "named_steps.logisticregression.coef_",
        ),
        # 单独使用 PCA，并指定 _pca_importances 作为 importance_getter
        (PCA(random_state=0), _pca_importances),
    ],
)
def test_importance_getter(estimator, importance_getter):
    # 使用 SelectFromModel 进行特征选择
    selector = SelectFromModel(
        estimator, threshold="mean", importance_getter=importance_getter
    )
    # 对数据进行拟合
    selector.fit(data, y)
    # 断言经过选择后的数据形状的第二个维度为 1
    assert selector.transform(data).shape[1] == 1


# 参数化测试，验证 PLS 估计器在 SelectFromModel 下的行为
def test_select_from_model_pls(PLSEstimator):
    """Check the behaviour of SelectFromModel with PLS estimators.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/12410
    """
    # 创建 Friedman1 数据集
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    # 使用指定的 PLS 估计器创建模型流水线
    estimator = PLSEstimator(n_components=1)
    model = make_pipeline(SelectFromModel(estimator), estimator).fit(X, y)
    # 断言模型的得分大于 0.5
    assert model.score(X, y) > 0.5


# 测试 SelectFromModel 在不支持 feature_names_in_ 的估计器下的行为
def test_estimator_does_not_support_feature_names():
    """SelectFromModel works with estimators that do not support feature_names_in_.

    Non-regression test for #21949.
    """
    # 确保导入 pandas 库
    pytest.importorskip("pandas")
    # 加载鸢尾花数据集
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    all_feature_names = set(X.columns)

    # 定义 importance_getter 函数，返回特征的序号
    def importance_getter(estimator):
        return np.arange(X.shape[1])

    # 使用 MinimalClassifier 创建 SelectFromModel 实例
    selector = SelectFromModel(
        MinimalClassifier(), importance_getter=importance_getter
    ).fit(X, y)

    # 断言选择器自己学习到的特征名与原数据集的列名相同
    assert_array_equal(selector.feature_names_in_, X.columns)

    # 获取选择后的特征名集合
    feature_names_out = set(selector.get_feature_names_out())
    # 断言选择后的特征名集合是原数据集所有列名的子集
    assert feature_names_out < all_feature_names

    # 捕获警告，检查选择器在转换数据时的行为
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        selector.transform(X.iloc[1:3])


# 参数化测试，验证 SelectFromModel 在部分拟合时验证 max_features 的行为
@pytest.mark.parametrize(
    "error, err_msg, max_features",
    (
        [ValueError, "max_features == 10, must be <= 4", 10],
        [ValueError, "max_features == 5, must be <= 4", lambda x: x.shape[1] + 1],
    ),
)
def test_partial_fit_validate_max_features(error, err_msg, max_features):
    """Test that partial_fit from SelectFromModel validates `max_features`."""
    # 创建二分类的随机数据集
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=4,
        random_state=0,
    )

    # 断言部分拟合时 SelectFromModel 对 max_features 进行验证
    with pytest.raises(error, match=err_msg):
        SelectFromModel(
            estimator=SGDClassifier(), max_features=max_features
        ).partial_fit(X, y, classes=[0, 1])
# 使用 pytest.mark.parametrize 装饰器为 test_partial_fit_validate_feature_names 函数添加参数化测试，测试参数 as_frame 分别为 True 和 False 时的情况
@pytest.mark.parametrize("as_frame", [True, False])
def test_partial_fit_validate_feature_names(as_frame):
    """Test that partial_fit from SelectFromModel validates `feature_names_in_`."""
    # 导入 pandas 库，如果未安装则跳过测试
    pytest.importorskip("pandas")
    # 加载鸢尾花数据集，根据 as_frame 参数确定是否返回 DataFrame，返回特征 X 和标签 y
    X, y = datasets.load_iris(as_frame=as_frame, return_X_y=True)

    # 使用 SGDClassifier 作为估计器，通过 partial_fit 进行部分拟合
    selector = SelectFromModel(estimator=SGDClassifier(), max_features=4).partial_fit(
        X, y, classes=[0, 1, 2]
    )
    # 如果 as_frame 为 True，则断言 selector 的 feature_names_in_ 属性与 X 的列名相等
    if as_frame:
        assert_array_equal(selector.feature_names_in_, X.columns)
    else:
        # 如果 as_frame 为 False，则断言 selector 没有 feature_names_in_ 属性
        assert not hasattr(selector, "feature_names_in_")


# 测试函数 test_from_model_estimator_attribute_error
def test_from_model_estimator_attribute_error():
    """Check that we raise the proper AttributeError when the estimator
    does not implement the `partial_fit` method, which is decorated with
    `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    """
    # 创建一个 SelectFromModel 对象，使用 LinearRegression 作为估计器
    # LinearRegression 类不实现 'partial_fit' 方法，应当引发 AttributeError
    from_model = SelectFromModel(estimator=LinearRegression())

    # 预期的外部异常消息
    outer_msg = "This 'SelectFromModel' has no attribute 'partial_fit'"
    # 预期的内部异常消息
    inner_msg = "'LinearRegression' object has no attribute 'partial_fit'"
    
    # 使用 pytest.raises 断言捕获 AttributeError 异常，并验证异常信息
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        # 对 from_model 对象进行拟合，并尝试调用 partial_fit 方法
        from_model.fit(data, y).partial_fit(data)
    # 断言 exec_info.value 的原因是 AttributeError 类型的异常
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言 inner_msg 存在于 exec_info.value 的原因中
    assert inner_msg in str(exec_info.value.__cause__)
```