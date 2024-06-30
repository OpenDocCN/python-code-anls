# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_bagging.py`

```
"""
Testing for the bagging ensemble module (sklearn.ensemble.bagging).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
from itertools import cycle, product  # 导入 cycle 和 product 函数

import joblib  # 导入 joblib 库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

import sklearn  # 导入 scikit-learn 库
from sklearn.base import BaseEstimator  # 导入 BaseEstimator 类
from sklearn.datasets import load_diabetes, load_iris, make_hastie_10_2  # 导入数据集加载函数
from sklearn.dummy import DummyClassifier, DummyRegressor  # 导入 Dummy 分类器和回归器
from sklearn.ensemble import (  # 导入集成学习模型
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest  # 导入特征选择函数
from sklearn.linear_model import LogisticRegression, Perceptron  # 导入线性模型
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split  # 导入交叉验证函数和参数网格
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # 导入 K 近邻分类器和回归器
from sklearn.pipeline import make_pipeline  # 导入管道函数
from sklearn.preprocessing import FunctionTransformer, scale  # 导入数据预处理函数
from sklearn.random_projection import SparseRandomProjection  # 导入稀疏随机投影
from sklearn.svm import SVC, SVR  # 导入支持向量机模型
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 导入决策树分类器和回归器
from sklearn.utils import check_random_state  # 导入随机状态检查函数
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal  # 导入测试函数
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS  # 导入稀疏矩阵格式容器修复函数

rng = check_random_state(0)  # 设置随机数生成器的种子

# also load the iris dataset
# and randomly permute it
iris = load_iris()  # 加载鸢尾花数据集
perm = rng.permutation(iris.target.size)  # 随机排列数据集的目标值索引
iris.data = iris.data[perm]  # 根据随机排列重新排列数据集的特征
iris.target = iris.target[perm]  # 根据随机排列重新排列数据集的目标值

# also load the diabetes dataset
# and randomly permute it
diabetes = load_diabetes()  # 加载糖尿病数据集
perm = rng.permutation(diabetes.target.size)  # 随机排列数据集的目标值索引
diabetes.data = diabetes.data[perm]  # 根据随机排列重新排列数据集的特征
diabetes.target = diabetes.target[perm]  # 根据随机排列重新排列数据集的目标值


def test_classification():
    # Check classification for various parameter settings.
    rng = check_random_state(0)  # 设置随机数生成器的种子
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )  # 将鸢尾花数据集划分为训练集和测试集
    grid = ParameterGrid(
        {
            "max_samples": [0.5, 1.0],
            "max_features": [1, 4],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
        }
    )  # 创建参数网格进行参数组合
    estimators = [
        None,
        DummyClassifier(),
        Perceptron(max_iter=20),
        DecisionTreeClassifier(max_depth=2),
        KNeighborsClassifier(),
        SVC(),
    ]  # 创建不同的基学习器列表
    # Try different parameter settings with different base classifiers without
    # doing the full cartesian product to keep the test durations low.
    for params, estimator in zip(grid, cycle(estimators)):
        BaggingClassifier(
            estimator=estimator,
            random_state=rng,
            n_estimators=2,
            **params,
        ).fit(X_train, y_train).predict(X_test)


@pytest.mark.parametrize(
    "sparse_container, params, method",
    # 调用 product 函数生成参数的笛卡尔积，这里是两个列表的笛卡尔积
    product(
        CSR_CONTAINERS + CSC_CONTAINERS,  # 第一个参数是 CSR_CONTAINERS 和 CSC_CONTAINERS 列表的组合
        [
            {  # 第二个参数是包含多个字典的列表，每个字典代表一组参数配置
                "max_samples": 0.5,  # 每个字典包含的参数：最大样本数为 0.5
                "max_features": 2,   # 最大特征数为 2
                "bootstrap": True,   # 使用 bootstrap 方法
                "bootstrap_features": True,  # 使用 bootstrap 特征
            },
            {
                "max_samples": 1.0,  # 另一组参数：最大样本数为 1.0
                "max_features": 4,   # 最大特征数为 4
                "bootstrap": True,   # 使用 bootstrap 方法
                "bootstrap_features": True,  # 使用 bootstrap 特征
            },
            {  # 下一组参数：最大特征数为 2，不使用 bootstrap
                "max_features": 2,
                "bootstrap": False,
                "bootstrap_features": True,
            },
            {  # 最后一组参数：最大样本数为 0.5，使用 bootstrap，但不使用 bootstrap 特征
                "max_samples": 0.5,
                "bootstrap": True,
                "bootstrap_features": False,
            },
        ],
        # 第三个参数是一个包含字符串的列表，代表要调用的函数名称
        ["predict", "predict_proba", "predict_log_proba", "decision_function"],
    ),
def test_sparse_classification(sparse_container, params, method):
    # 检查稀疏输入的各种参数设置下的分类效果。

    class CustomSVC(SVC):
        """记录训练集特性的SVC变体"""

        def fit(self, X, y):
            # 调用父类的fit方法来训练模型
            super().fit(X, y)
            # 记录训练集的数据类型
            self.data_type_ = type(X)
            return self

    rng = check_random_state(0)
    # 将数据进行标准化，并随机划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        scale(iris.data), iris.target, random_state=rng
    )

    # 将训练集和测试集转换为稀疏格式
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    # 在稀疏格式上训练分类器
    sparse_classifier = BaggingClassifier(
        estimator=CustomSVC(kernel="linear", decision_function_shape="ovr"),
        random_state=1,
        **params,
    ).fit(X_train_sparse, y_train)
    # 使用指定方法对稀疏格式的测试集进行预测
    sparse_results = getattr(sparse_classifier, method)(X_test_sparse)

    # 在密集格式上训练分类器
    dense_classifier = BaggingClassifier(
        estimator=CustomSVC(kernel="linear", decision_function_shape="ovr"),
        random_state=1,
        **params,
    ).fit(X_train, y_train)
    # 使用指定方法对密集格式的测试集进行预测
    dense_results = getattr(dense_classifier, method)(X_test)
    # 检查稀疏格式和密集格式的预测结果是否近似相等
    assert_array_almost_equal(sparse_results, dense_results)

    # 获取稀疏分类器中使用的数据类型
    sparse_type = type(X_train_sparse)
    # 获取所有基础估算器中记录的数据类型
    types = [i.data_type_ for i in sparse_classifier.estimators_]

    # 断言所有基础估算器中记录的数据类型与稀疏格式一致
    assert all([t == sparse_type for t in types])


def test_regression():
    # 检查各种参数设置下的回归效果。
    rng = check_random_state(0)
    # 将数据随机划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data[:50], diabetes.target[:50], random_state=rng
    )
    # 定义参数网格
    grid = ParameterGrid(
        {
            "max_samples": [0.5, 1.0],
            "max_features": [0.5, 1.0],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
        }
    )

    # 针对每种估算器类型和每组参数进行循环
    for estimator in [
        None,
        DummyRegressor(),
        DecisionTreeRegressor(),
        KNeighborsRegressor(),
        SVR(),
    ]:
        for params in grid:
            # 使用BaggingRegressor进行拟合和预测
            BaggingRegressor(estimator=estimator, random_state=rng, **params).fit(
                X_train, y_train
            ).predict(X_test)


@pytest.mark.parametrize("sparse_container", CSR_CONTAINERS + CSC_CONTAINERS)
def test_sparse_regression(sparse_container):
    # 检查稀疏输入的各种参数设置下的回归效果。
    rng = check_random_state(0)
    # 将数据随机划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data[:50], diabetes.target[:50], random_state=rng
    )

    class CustomSVR(SVR):
        """记录训练集特性的SVR变体"""

        def fit(self, X, y):
            # 调用父类的fit方法来训练模型
            super().fit(X, y)
            # 记录训练集的数据类型
            self.data_type_ = type(X)
            return self
    # 定义一组参数集合，用于不同配置的 BaggingRegressor
    parameter_sets = [
        {
            "max_samples": 0.5,                   # 每个基础估计器中抽取的样本比例
            "max_features": 2,                    # 每个基础估计器使用的特征数量
            "bootstrap": True,                    # 是否进行自助采样
            "bootstrap_features": True,           # 是否对特征进行自助采样
        },
        {
            "max_samples": 1.0,                   # 每个基础估计器中抽取的样本比例
            "max_features": 4,                    # 每个基础估计器使用的特征数量
            "bootstrap": True,                    # 是否进行自助采样
            "bootstrap_features": True,           # 是否对特征进行自助采样
        },
        {
            "max_features": 2,                    # 每个基础估计器使用的特征数量
            "bootstrap": False,                   # 不进行样本自助采样
            "bootstrap_features": True,           # 是否对特征进行自助采样
        },
        {
            "max_samples": 0.5,                   # 每个基础估计器中抽取的样本比例
            "bootstrap": True,                    # 是否进行自助采样
            "bootstrap_features": False,          # 不对特征进行自助采样
        },
    ]

    # 将训练数据转换为稀疏矩阵格式
    X_train_sparse = sparse_container(X_train)
    # 将测试数据转换为稀疏矩阵格式
    X_test_sparse = sparse_container(X_test)

    # 对每组参数进行循环
    for params in parameter_sets:
        # 在稀疏格式上训练 BaggingRegressor
        sparse_classifier = BaggingRegressor(
            estimator=CustomSVR(),               # 使用自定义支持向量回归器作为基础估计器
            random_state=1,                      # 随机数种子设为1
            **params                             # 使用当前参数集合
        ).fit(X_train_sparse, y_train)
        
        # 在稀疏格式上进行预测
        sparse_results = sparse_classifier.predict(X_test_sparse)

        # 在稠密格式上训练 BaggingRegressor 并预测
        dense_results = (
            BaggingRegressor(
                estimator=CustomSVR(),            # 使用自定义支持向量回归器作为基础估计器
                random_state=1,                   # 随机数种子设为1
                **params                          # 使用当前参数集合
            ).fit(X_train, y_train)
            .predict(X_test)
        )

        # 获取稀疏数据的类型
        sparse_type = type(X_train_sparse)
        # 获取每个基础估计器的数据类型
        types = [i.data_type_ for i in sparse_classifier.estimators_]

        # 断言稀疏结果与稠密结果几乎相等
        assert_array_almost_equal(sparse_results, dense_results)
        # 断言所有基础估计器的数据类型与稀疏数据类型相同
        assert all([t == sparse_type for t in types])
        # 断言稀疏结果与稠密结果几乎相等
        assert_array_almost_equal(sparse_results, dense_results)
class DummySizeEstimator(BaseEstimator):
    # 自定义的大小估算器类，继承自BaseEstimator

    def fit(self, X, y):
        # 拟合方法，计算训练样本的大小和哈希值
        self.training_size_ = X.shape[0]
        self.training_hash_ = joblib.hash(X)

    def predict(self, X):
        # 预测方法，返回与输入样本数相同长度的全1数组
        return np.ones(X.shape[0])


def test_bootstrap_samples():
    # 测试用例：验证使用自举采样生成的集成估算器是否不完全匹配基估算器
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    estimator = DecisionTreeRegressor().fit(X_train, y_train)

    # 不使用自举采样时，所有树在训练集上均为完美匹配
    ensemble = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        max_samples=1.0,
        bootstrap=False,
        random_state=rng,
    ).fit(X_train, y_train)

    assert estimator.score(X_train, y_train) == ensemble.score(X_train, y_train)

    # 使用自举采样时，树在训练集上不再完美匹配
    ensemble = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        max_samples=1.0,
        bootstrap=True,
        random_state=rng,
    ).fit(X_train, y_train)

    assert estimator.score(X_train, y_train) > ensemble.score(X_train, y_train)

    # 检查每次采样是否对应完整的自举重采样
    # 每次自举重采样的大小应与输入数据相同，但数据应不同（通过数据的哈希值来验证）
    ensemble = BaggingRegressor(estimator=DummySizeEstimator(), bootstrap=True).fit(
        X_train, y_train
    )
    training_hash = []
    for estimator in ensemble.estimators_:
        assert estimator.training_size_ == X_train.shape[0]
        training_hash.append(estimator.training_hash_)
    assert len(set(training_hash)) == len(training_hash)


def test_bootstrap_features():
    # 测试用例：验证使用自举特征是否可能生成重复特征
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    ensemble = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        max_features=1.0,
        bootstrap_features=False,
        random_state=rng,
    ).fit(X_train, y_train)

    for features in ensemble.estimators_features_:
        assert diabetes.data.shape[1] == np.unique(features).shape[0]

    ensemble = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        max_features=1.0,
        bootstrap_features=True,
        random_state=rng,
    ).fit(X_train, y_train)

    for features in ensemble.estimators_features_:
        assert diabetes.data.shape[1] > np.unique(features).shape[0]


def test_probability():
    # 测试用例：预测概率
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )
    # 设置 numpy 的错误状态，在计算时忽略除零和无效值的警告
    with np.errstate(divide="ignore", invalid="ignore"):
        # 正常情况下的集成学习，使用决策树作为基础分类器
        ensemble = BaggingClassifier(
            estimator=DecisionTreeClassifier(), random_state=rng
        ).fit(X_train, y_train)

        # 断言预测概率之和近似为全1数组
        assert_array_almost_equal(
            np.sum(ensemble.predict_proba(X_test), axis=1), np.ones(len(X_test))
        )

        # 断言预测概率和预测对数概率的指数近似相等
        assert_array_almost_equal(
            ensemble.predict_proba(X_test), np.exp(ensemble.predict_log_proba(X_test))
        )

        # 特殊情况，部分类别缺失的集成学习，使用逻辑回归作为基础分类器
        ensemble = BaggingClassifier(
            estimator=LogisticRegression(), random_state=rng, max_samples=5
        ).fit(X_train, y_train)

        # 断言预测概率之和近似为全1数组
        assert_array_almost_equal(
            np.sum(ensemble.predict_proba(X_test), axis=1), np.ones(len(X_test))
        )

        # 断言预测概率和预测对数概率的指数近似相等
        assert_array_almost_equal(
            ensemble.predict_proba(X_test), np.exp(ensemble.predict_log_proba(X_test))
        )
def test_oob_score_classification():
    # Check that oob prediction is a good estimation of the generalization
    # error.

    # 设置随机数生成器，用于确定训练和测试集的分割方式
    rng = check_random_state(0)
    
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )

    # 对于每个分类器，分别进行袋装法分类器的测试
    for estimator in [DecisionTreeClassifier(), SVC()]:
        # 创建一个 Bagging 分类器，使用给定的基础估计器
        clf = BaggingClassifier(
            estimator=estimator,
            n_estimators=100,
            bootstrap=True,
            oob_score=True,
            random_state=rng,
        ).fit(X_train, y_train)

        # 测试分类器的得分
        test_score = clf.score(X_test, y_test)

        # 断言测试得分与 oob 得分的差距小于 0.1
        assert abs(test_score - clf.oob_score_) < 0.1

        # 测试使用较少估计器的情况
        warn_msg = (
            "Some inputs do not have OOB scores. This probably means too few "
            "estimators were used to compute any reliable oob estimates."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            # 创建一个只包含一个估计器的 Bagging 分类器
            clf = BaggingClassifier(
                estimator=estimator,
                n_estimators=1,
                bootstrap=True,
                oob_score=True,
                random_state=rng,
            )
            clf.fit(X_train, y_train)


def test_oob_score_regression():
    # Check that oob prediction is a good estimation of the generalization
    # error.

    # 设置随机数生成器，用于确定训练和测试集的分割方式
    rng = check_random_state(0)
    
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    # 创建一个 Bagging 回归器，使用决策树回归器作为基础估计器
    clf = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=50,
        bootstrap=True,
        oob_score=True,
        random_state=rng,
    ).fit(X_train, y_train)

    # 测试回归器的得分
    test_score = clf.score(X_test, y_test)

    # 断言测试得分与 oob 得分的差距小于 0.1
    assert abs(test_score - clf.oob_score_) < 0.1

    # 测试使用较少估计器的情况
    warn_msg = (
        "Some inputs do not have OOB scores. This probably means too few "
        "estimators were used to compute any reliable oob estimates."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        # 创建一个只包含一个估计器的 Bagging 回归器
        regr = BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            n_estimators=1,
            bootstrap=True,
            oob_score=True,
            random_state=rng,
        )
        regr.fit(X_train, y_train)


def test_single_estimator():
    # Check singleton ensembles.

    # 设置随机数生成器，用于确定训练和测试集的分割方式
    rng = check_random_state(0)
    
    # 将数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    # 创建一个只包含一个估计器的 Bagging 回归器，不使用自助法和自助属性抽样
    clf1 = BaggingRegressor(
        estimator=KNeighborsRegressor(),
        n_estimators=1,
        bootstrap=False,
        bootstrap_features=False,
        random_state=rng,
    ).fit(X_train, y_train)

    # 创建一个普通的 K 近邻回归器
    clf2 = KNeighborsRegressor().fit(X_train, y_train)

    # 断言两个回归器的预测结果几乎相等
    assert_array_almost_equal(clf1.predict(X_test), clf2.predict(X_test))


def test_error():
    # Test support of decision_function

    # 获取鸢尾花数据集的特征和目标变量
    X, y = iris.data, iris.target
    
    # 创建一个基于决策树的分类器
    base = DecisionTreeClassifier()
    # 断言语句，用于检查 BaggingClassifier 对象是否没有 "decision_function" 属性
    assert not hasattr(BaggingClassifier(base).fit(X, y), "decision_function")
def test_parallel_classification():
    # Check parallel classification.

    # 使用 train_test_split 函数划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0
    )

    # 创建一个 BaggingClassifier 集成模型，使用决策树分类器作为基础估计器，启用 3 个工作进程并拟合模型
    ensemble = BaggingClassifier(
        DecisionTreeClassifier(), n_jobs=3, random_state=0
    ).fit(X_train, y_train)

    # 使用拟合好的模型进行预测，返回类别概率
    y1 = ensemble.predict_proba(X_test)

    # 将模型的工作进程数设置为 1，再次预测类别概率
    ensemble.set_params(n_jobs=1)
    y2 = ensemble.predict_proba(X_test)

    # 检查两次预测的类别概率是否几乎相等
    assert_array_almost_equal(y1, y2)

    # 使用单工作进程的 BaggingClassifier 拟合模型，再次预测类别概率
    ensemble = BaggingClassifier(
        DecisionTreeClassifier(), n_jobs=1, random_state=0
    ).fit(X_train, y_train)
    y3 = ensemble.predict_proba(X_test)

    # 检查不同设置下的类别概率预测是否几乎相等
    assert_array_almost_equal(y1, y3)

    # 创建一个使用 One-vs-Rest 方式决策函数的 BaggingClassifier 模型，启用 3 个工作进程拟合模型
    ensemble = BaggingClassifier(
        SVC(decision_function_shape="ovr"), n_jobs=3, random_state=0
    ).fit(X_train, y_train)

    # 预测测试集的决策函数值
    decisions1 = ensemble.decision_function(X_test)

    # 将模型的工作进程数设置为 1，再次预测决策函数值
    ensemble.set_params(n_jobs=1)
    decisions2 = ensemble.decision_function(X_test)

    # 检查两次预测的决策函数值是否几乎相等
    assert_array_almost_equal(decisions1, decisions2)

    # 使用单工作进程的 BaggingClassifier 拟合模型，再次预测决策函数值
    ensemble = BaggingClassifier(
        SVC(decision_function_shape="ovr"), n_jobs=1, random_state=0
    ).fit(X_train, y_train)
    decisions3 = ensemble.decision_function(X_test)

    # 检查不同设置下的决策函数值预测是否几乎相等
    assert_array_almost_equal(decisions1, decisions3)


def test_parallel_regression():
    # Check parallel regression.

    # 使用 check_random_state 函数创建随机数生成器
    rng = check_random_state(0)

    # 使用 train_test_split 函数划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )

    # 创建一个 BaggingRegressor 集成模型，使用决策树回归器作为基础估计器，启用 3 个工作进程并拟合模型
    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=3, random_state=0).fit(
        X_train, y_train
    )

    # 将模型的工作进程数设置为 1，再次预测目标变量
    ensemble.set_params(n_jobs=1)
    y1 = ensemble.predict(X_test)

    # 将模型的工作进程数设置为 2，再次预测目标变量
    ensemble.set_params(n_jobs=2)
    y2 = ensemble.predict(X_test)

    # 检查两次预测的目标变量是否几乎相等
    assert_array_almost_equal(y1, y2)

    # 使用单工作进程的 BaggingRegressor 拟合模型，再次预测目标变量
    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=1, random_state=0).fit(
        X_train, y_train
    )
    y3 = ensemble.predict(X_test)

    # 检查不同设置下的目标变量预测是否几乎相等
    assert_array_almost_equal(y1, y3)


def test_gridsearch():
    # Check that bagging ensembles can be grid-searched.

    # 将 iris 数据集转换为二元分类任务
    X, y = iris.data, iris.target
    y[y == 2] = 1

    # 定义参数网格
    parameters = {"n_estimators": (1, 2), "estimator__C": (1, 2)}

    # 使用 GridSearchCV 对 BaggingClassifier 进行网格搜索，使用 roc_auc 作为评分标准
    GridSearchCV(BaggingClassifier(SVC()), parameters, scoring="roc_auc").fit(X, y)


def test_estimator():
    # Check estimator and its default values.

    # 使用 check_random_state 函数创建随机数生成器
    rng = check_random_state(0)

    # 分类任务
    # 使用 train_test_split 函数划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )

    # 创建一个 BaggingClassifier 集成模型，使用默认估计器，启用 3 个工作进程并拟合模型
    ensemble = BaggingClassifier(None, n_jobs=3, random_state=0).fit(X_train, y_train)

    # 检查模型的基础估计器是否为 DecisionTreeClassifier
    assert isinstance(ensemble.estimator_, DecisionTreeClassifier)

    # 创建一个 BaggingClassifier 集成模型，使用决策树分类器作为基础估计器，启用 3 个工作进程并拟合模型
    ensemble = BaggingClassifier(
        DecisionTreeClassifier(), n_jobs=3, random_state=0
    ).fit(X_train, y_train)

    # 检查模型的基础估计器是否为 DecisionTreeClassifier
    assert isinstance(ensemble.estimator_, DecisionTreeClassifier)
    # 使用 Bagging 方法创建一个分类器，基础模型为 Perceptron，使用 3 个工作进程并设置随机种子为 0 进行训练
    ensemble = BaggingClassifier(Perceptron(), n_jobs=3, random_state=0).fit(
        X_train, y_train
    )
    
    # 断言确保 ensemble 的基础模型是 Perceptron 类型的对象
    assert isinstance(ensemble.estimator_, Perceptron)
    
    # 将数据集按照一定的随机方式划分为训练集和测试集，用于后续的回归任务
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=rng
    )
    
    # 使用 Bagging 方法创建一个回归器，基础模型为 None（默认为 DecisionTreeRegressor），使用 3 个工作进程并设置随机种子为 0 进行训练
    ensemble = BaggingRegressor(None, n_jobs=3, random_state=0).fit(X_train, y_train)
    
    # 断言确保 ensemble 的基础模型是 DecisionTreeRegressor 类型的对象
    assert isinstance(ensemble.estimator_, DecisionTreeRegressor)
    
    # 使用 Bagging 方法创建一个回归器，基础模型为 DecisionTreeRegressor，使用 3 个工作进程并设置随机种子为 0 进行训练
    ensemble = BaggingRegressor(DecisionTreeRegressor(), n_jobs=3, random_state=0).fit(
        X_train, y_train
    )
    
    # 断言确保 ensemble 的基础模型是 DecisionTreeRegressor 类型的对象
    assert isinstance(ensemble.estimator_, DecisionTreeRegressor)
    
    # 使用 Bagging 方法创建一个回归器，基础模型为 SVR，使用 3 个工作进程并设置随机种子为 0 进行训练
    ensemble = BaggingRegressor(SVR(), n_jobs=3, random_state=0).fit(X_train, y_train)
    
    # 断言确保 ensemble 的基础模型是 SVR 类型的对象
    assert isinstance(ensemble.estimator_, SVR)
def test_bagging_with_pipeline():
    # 创建一个 BaggingClassifier，内部使用 SelectKBest 和 DecisionTreeClassifier 组成的 pipeline
    estimator = BaggingClassifier(
        make_pipeline(SelectKBest(k=1), DecisionTreeClassifier()), max_features=2
    )
    # 使用 iris 数据集进行拟合
    estimator.fit(iris.data, iris.target)
    # 断言 estimator 的第一个步骤（SelectKBest）的最后一个步骤（DecisionTreeClassifier）的 random_state 是整数类型
    assert isinstance(estimator[0].steps[-1][1].random_state, int)


class DummyZeroEstimator(BaseEstimator):
    def fit(self, X, y):
        # 设置 DummyZeroEstimator 的 classes_ 属性为 y 的唯一值
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # 返回一个数组，内容为 classes_ 数组的长度，全为整数 0
        return self.classes_[np.zeros(X.shape[0], dtype=int)]


def test_bagging_sample_weight_unsupported_but_passed():
    # 使用 DummyZeroEstimator 创建一个 BaggingClassifier
    estimator = BaggingClassifier(DummyZeroEstimator())
    rng = check_random_state(0)

    # 进行拟合和预测，忽略 sample_weight 参数
    estimator.fit(iris.data, iris.target).predict(iris.data)
    # 使用 pytest 断言捕获 ValueError 异常
    with pytest.raises(ValueError):
        # 重新拟合，传入一个随机生成的 sample_weight 参数
        estimator.fit(
            iris.data,
            iris.target,
            sample_weight=rng.randint(10, size=(iris.data.shape[0])),
        )


def test_warm_start(random_state=42):
    # 测试增量拟合 warm start 是否能正确产生预期数量的森林，并与正常拟合结果一致
    X, y = make_hastie_10_2(n_samples=20, random_state=1)

    clf_ws = None
    for n_estimators in [5, 10]:
        if clf_ws is None:
            # 创建一个 warm start 的 BaggingClassifier
            clf_ws = BaggingClassifier(
                n_estimators=n_estimators, random_state=random_state, warm_start=True
            )
        else:
            # 调整 n_estimators 参数
            clf_ws.set_params(n_estimators=n_estimators)
        # 拟合数据
        clf_ws.fit(X, y)
        # 断言 clf_ws 中的分类器数量是否等于 n_estimators
        assert len(clf_ws) == n_estimators

    # 创建一个非 warm start 的 BaggingClassifier
    clf_no_ws = BaggingClassifier(
        n_estimators=10, random_state=random_state, warm_start=False
    )
    clf_no_ws.fit(X, y)

    # 断言两个 BaggingClassifier 的随机状态集合是否相等
    assert set([tree.random_state for tree in clf_ws]) == set(
        [tree.random_state for tree in clf_no_ws]
    )


def test_warm_start_smaller_n_estimators():
    # 测试 warm start 的情况下，第二次拟合时如果 n_estimators 减小是否会引发错误
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    # 创建一个 warm start 的 BaggingClassifier
    clf = BaggingClassifier(n_estimators=5, warm_start=True)
    clf.fit(X, y)
    # 调整 n_estimators 参数，应当引发 ValueError 异常
    clf.set_params(n_estimators=4)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_warm_start_equal_n_estimators():
    # 测试当不增加 n_estimators 的情况下进行 warm start 是否不产生变化
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    # 创建一个 warm start 的 BaggingClassifier
    clf = BaggingClassifier(n_estimators=5, warm_start=True, random_state=83)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # 修改 X 的值为无意义的数值，不应改变预测结果
    X_train += 1.0

    # 使用 pytest.warns 检查是否产生 UserWarning
    warn_msg = "Warm-start fitting without increasing n_estimators does not"
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X_train, y_train)
    # 断言预测结果与之前一致
    assert_array_equal(y_pred, clf.predict(X_test))


def test_warm_start_equivalence():
    # 测试 warm start 下，使用 5+5 个分类器的 BaggingClassifier 是否等效于一个使用 10 个分类器的 BaggingClassifier
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    # 使用 train_test_split 函数将数据集 X 和标签 y 分割成训练集和测试集，使用固定的随机种子 43
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)
    
    # 创建一个 BaggingClassifier 分类器对象 clf_ws，包含 5 个基分类器，允许 warm_start 模式，并设置随机种子为 3141
    clf_ws = BaggingClassifier(n_estimators=5, warm_start=True, random_state=3141)
    # 使用训练集 (X_train, y_train) 对 clf_ws 进行训练
    clf_ws.fit(X_train, y_train)
    # 修改 clf_ws 的参数，设置基分类器数量为 10，继续在原有的基础上训练
    clf_ws.set_params(n_estimators=10)
    # 使用扩展后的 clf_ws 对测试集 X_test 进行预测
    y1 = clf_ws.predict(X_test)
    
    # 创建一个新的 BaggingClassifier 分类器对象 clf，包含 10 个基分类器，禁用 warm_start 模式，并设置随机种子为 3141
    clf = BaggingClassifier(n_estimators=10, warm_start=False, random_state=3141)
    # 使用训练集 (X_train, y_train) 对 clf 进行训练
    clf.fit(X_train, y_train)
    # 使用 clf 对测试集 X_test 进行预测
    y2 = clf.predict(X_test)
    
    # 使用 assert_array_almost_equal 函数断言 y1 和 y2 数组几乎完全相等，以检验分类器的一致性
    assert_array_almost_equal(y1, y2)
# 测试在使用 oob_score 和 warm_start 同时时会失败
def test_warm_start_with_oob_score_fails():
    # 生成一个小数据集 X, y
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    # 创建一个 Bagging 分类器，设置了 warm_start 和 oob_score
    clf = BaggingClassifier(n_estimators=5, warm_start=True, oob_score=True)
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        clf.fit(X, y)


# 测试在 warm_start 模式下是否成功移除 oob_score
def test_oob_score_removed_on_warm_start():
    # 生成一个较大的数据集 X, y
    X, y = make_hastie_10_2(n_samples=100, random_state=1)

    # 创建一个 Bagging 分类器，设置了 oob_score
    clf = BaggingClassifier(n_estimators=5, oob_score=True)
    clf.fit(X, y)

    # 修改 clf 的参数，启用 warm_start 模式，并移除 oob_score
    clf.set_params(warm_start=True, oob_score=False, n_estimators=10)
    clf.fit(X, y)

    # 使用 pytest 检查是否会抛出 AttributeError 异常
    with pytest.raises(AttributeError):
        getattr(clf, "oob_score_")


# 测试确保在固定 random_state、estimator 和训练数据的情况下，OOB 分数在两次拟合时是一致的
def test_oob_score_consistency():
    # 生成一个较大的数据集 X, y
    X, y = make_hastie_10_2(n_samples=200, random_state=1)
    # 创建一个 Bagging 分类器，使用 KNeighborsClassifier 作为基础估计器，并设置 oob_score
    bagging = BaggingClassifier(
        KNeighborsClassifier(),
        max_samples=0.5,
        max_features=0.5,
        oob_score=True,
        random_state=1,
    )
    # 使用 assert 检查两次拟合的 OOB 分数是否一致
    assert bagging.fit(X, y).oob_score_ == bagging.fit(X, y).oob_score_


# 测试确保 estimators_samples_ 的格式正确，并且在 fit 时生成的结果可以在后续时间中通过对象属性复现
def test_estimators_samples():
    # 生成一个较大的数据集 X, y
    X, y = make_hastie_10_2(n_samples=200, random_state=1)
    # 创建一个 Bagging 分类器，使用 LogisticRegression 作为基础估计器，并设置 max_samples, max_features
    # 同时关闭 bootstrap
    bagging = BaggingClassifier(
        LogisticRegression(),
        max_samples=0.5,
        max_features=0.5,
        random_state=1,
        bootstrap=False,
    )
    bagging.fit(X, y)

    # 获取相关属性
    estimators_samples = bagging.estimators_samples_
    estimators_features = bagging.estimators_features_
    estimators = bagging.estimators_

    # 使用 assert 检查 estimators_samples 的格式是否正确
    assert len(estimators_samples) == len(estimators)
    assert len(estimators_samples[0]) == len(X) // 2
    assert estimators_samples[0].dtype.kind == "i"

    # 重新拟合单个估计器以测试采样的一致性
    estimator_index = 0
    estimator_samples = estimators_samples[estimator_index]
    estimator_features = estimators_features[estimator_index]
    estimator = estimators[estimator_index]

    X_train = (X[estimator_samples])[:, estimator_features]
    y_train = y[estimator_samples]

    orig_coefs = estimator.coef_
    estimator.fit(X_train, y_train)
    new_coefs = estimator.coef_

    # 使用 assert_array_almost_equal 检查拟合前后的系数是否一致
    assert_array_almost_equal(orig_coefs, new_coefs)


# 这个测试是一个回归测试，用于检查在具有随机步骤（例如 SparseRandomProjection）和给定随机状态时，
# 使用 fit 时生成的结果是否可以在后续时间中通过对象属性复现。
# 查看 issue #9524 进行详细讨论。
def test_estimators_samples_deterministic():
    iris = load_iris()
    X, y = iris.data, iris.target

    # 创建一个基础 pipeline，包含 SparseRandomProjection 和 LogisticRegression
    base_pipeline = make_pipeline(
        SparseRandomProjection(n_components=2), LogisticRegression()
    )
    # 使用 Bagging 方法构建分类器 clf，基分类器为 base_pipeline，每个基分类器使用的样本数为总样本数的 50%，随机种子为 0
    clf = BaggingClassifier(estimator=base_pipeline, max_samples=0.5, random_state=0)
    # 使用 clf 对数据集 X, y 进行拟合
    clf.fit(X, y)
    # 获取 BaggingClassifier 中第一个基分类器的最后一步的系数，并复制到 pipeline_estimator_coef 中
    pipeline_estimator_coef = clf.estimators_[0].steps[-1][1].coef_.copy()

    # 获取 BaggingClassifier 中第一个基分类器的引用
    estimator = clf.estimators_[0]
    # 获取 BaggingClassifier 中第一个基分类器使用的样本索引
    estimator_sample = clf.estimators_samples_[0]
    # 获取 BaggingClassifier 中第一个基分类器使用的特征索引
    estimator_feature = clf.estimators_features_[0]

    # 根据基分类器的样本索引和特征索引，从原始数据集 X 中获取训练数据 X_train
    X_train = (X[estimator_sample])[:, estimator_feature]
    # 根据基分类器的样本索引，从原始标签数据 y 中获取训练标签 y_train
    y_train = y[estimator_sample]

    # 使用基分类器 estimator 对训练数据 X_train 和标签 y_train 进行拟合
    estimator.fit(X_train, y_train)
    # 断言基分类器最后一步的系数与之前复制的 pipeline_estimator_coef 相等
    assert_array_equal(estimator.steps[-1][1].coef_, pipeline_estimator_coef)
# 确保验证的 max_samples 和原始的 max_samples 在用户提供有效整数 max_samples 的情况下是相同的
def test_max_samples_consistency():
    # 设定最大样本数为 100
    max_samples = 100
    # 创建数据集 X 和标签 y，确保样本数是 max_samples 的两倍
    X, y = make_hastie_10_2(n_samples=2 * max_samples, random_state=1)
    # 使用 KNeighborsClassifier 作为基分类器，创建 BaggingClassifier 对象
    bagging = BaggingClassifier(
        KNeighborsClassifier(),
        max_samples=max_samples,
        max_features=0.5,
        random_state=1,
    )
    # 在数据集上训练 BaggingClassifier
    bagging.fit(X, y)
    # 断言 BaggingClassifier 内部的 _max_samples 等于预期的 max_samples
    assert bagging._max_samples == max_samples


# 确保当标签发生变化时，oob_score 不会改变
def test_set_oob_score_label_encoding():
    # 设置随机种子
    random_state = 5
    # 创建样本数据 X 和三种不同的标签 Y1, Y2, Y3
    X = [[-1], [0], [1]] * 5
    Y1 = ["A", "B", "C"] * 5
    Y2 = [-1, 0, 1] * 5
    Y3 = [0, 1, 2] * 5
    # 分别使用不同的标签训练 BaggingClassifier，并获取其 oob_score
    x1 = (
        BaggingClassifier(oob_score=True, random_state=random_state)
        .fit(X, Y1)
        .oob_score_
    )
    x2 = (
        BaggingClassifier(oob_score=True, random_state=random_state)
        .fit(X, Y2)
        .oob_score_
    )
    x3 = (
        BaggingClassifier(oob_score=True, random_state=random_state)
        .fit(X, Y3)
        .oob_score_
    )
    # 断言三种不同标签下的 oob_score 是相同的
    assert [x1, x2] == [x3, x3]


# 替换 X 中的缺失值和无穷值为 0，并返回替换后的 X
def replace(X):
    X = X.astype("float", copy=True)
    X[~np.isfinite(X)] = 0
    return X


# 检查 BaggingRegressor 是否可以接受具有缺失数据/无穷数据的 X
def test_bagging_regressor_with_missing_inputs():
    # 创建包含缺失数据和无穷数据的样本 X
    X = np.array(
        [
            [1, 3, 5],
            [2, None, 6],
            [2, np.nan, 6],
            [2, np.inf, 6],
            [2, -np.inf, 6],
        ]
    )
    # 创建不同形状的标签 y
    y_values = [
        np.array([2, 3, 3, 3, 3]),
        np.array(
            [
                [2, 1, 9],
                [3, 6, 8],
                [3, 6, 8],
                [3, 6, 8],
                [3, 6, 8],
            ]
        ),
    ]
    # 对每种形状的 y 进行测试
    for y in y_values:
        # 创建决策树回归器作为基本回归器
        regressor = DecisionTreeRegressor()
        # 创建包含数据预处理和回归器的 Pipeline
        pipeline = make_pipeline(FunctionTransformer(replace), regressor)
        # 训练 Pipeline 并预测结果
        pipeline.fit(X, y).predict(X)
        # 创建 BaggingRegressor 对象
        bagging_regressor = BaggingRegressor(pipeline)
        # 训练 BaggingRegressor 并预测结果
        y_hat = bagging_regressor.fit(X, y).predict(X)
        # 断言预测结果 y_hat 的形状与原始标签 y 的形状相同
        assert y.shape == y_hat.shape

        # 验证包装回归器是否能引发异常
        regressor = DecisionTreeRegressor()
        pipeline = make_pipeline(regressor)
        # 使用 Pipeline 拟合数据并预测
        with pytest.raises(ValueError):
            pipeline.fit(X, y)
        # 创建 BaggingRegressor 对象并验证是否能引发异常
        bagging_regressor = BaggingRegressor(pipeline)
        with pytest.raises(ValueError):
            bagging_regressor.fit(X, y)


# 检查 BaggingClassifier 是否可以接受具有缺失数据/无穷数据的 X
def test_bagging_classifier_with_missing_inputs():
    # 创建包含缺失数据和无穷数据的样本 X
    X = np.array(
        [
            [1, 3, 5],
            [2, None, 6],
            [2, np.nan, 6],
            [2, np.inf, 6],
            [2, -np.inf, 6],
        ]
    )
    # 创建标签 y
    y = np.array([3, 6, 6, 6, 6])
    # 创建决策树分类器作为基本分类器
    classifier = DecisionTreeClassifier()
    # 创建包含数据预处理和分类器的 Pipeline
    pipeline = make_pipeline(FunctionTransformer(replace), classifier)
    # 使用管道模型拟合数据 X 和标签 y，并对 X 进行预测
    pipeline.fit(X, y).predict(X)

    # 创建一个 Bagging 分类器，使用给定的 pipeline 作为基本分类器
    bagging_classifier = BaggingClassifier(pipeline)

    # 使用 Bagging 分类器拟合数据 X 和标签 y
    bagging_classifier.fit(X, y)

    # 使用拟合好的 Bagging 分类器对 X 进行预测
    y_hat = bagging_classifier.predict(X)

    # 断言预测结果 y_hat 的形状与原始标签 y 的形状相同
    assert y.shape == y_hat.shape

    # 使用 Bagging 分类器对 X 计算预测的对数概率
    bagging_classifier.predict_log_proba(X)

    # 使用 Bagging 分类器对 X 计算预测的概率
    bagging_classifier.predict_proba(X)

    # 验证包装分类器是否能够引发异常
    classifier = DecisionTreeClassifier()
    # 创建一个 pipeline 包含决策树分类器
    pipeline = make_pipeline(classifier)
    # 使用 pytest 验证异常是否能够被正确引发
    with pytest.raises(ValueError):
        # 尝试使用 pipeline 对数据 X 和标签 y 进行拟合
        pipeline.fit(X, y)

    # 创建一个 Bagging 分类器，使用上面定义的 pipeline 作为基础分类器
    bagging_classifier = BaggingClassifier(pipeline)

    # 使用 pytest 验证异常是否能够被正确引发
    with pytest.raises(ValueError):
        # 尝试使用 bagging_classifier 对数据 X 和标签 y 进行拟合
        bagging_classifier.fit(X, y)
# 测试 BaggingClassifier 是否能接受低分数的 max_features

# 创建输入特征 X 和标签 y 的 NumPy 数组
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 0])

# 使用 LogisticRegression 作为基础估计器，创建一个 BaggingClassifier 实例，
# 设置 max_features 为 0.3，随机种子为 1
bagging = BaggingClassifier(LogisticRegression(), max_features=0.3, random_state=1)

# 使用输入数据 X 和标签 y 对 BaggingClassifier 进行训练
bagging.fit(X, y)


# 检查 BaggingClassifier 是否能正确生成样本索引
# 这是一个非回归测试，用于检查以下问题的修复：
# https://github.com/scikit-learn/scikit-learn/issues/16436

# 创建一个随机数生成器 rng
rng = np.random.RandomState(0)

# 创建一个形状为 (13, 4) 的随机数数组 X
X = rng.randn(13, 4)

# 创建一个包含 0 到 12 的数组作为标签 y
y = np.arange(13)

# 定义一个自定义的估计器 MyEstimator，它在拟合时存储 y 索引信息
class MyEstimator(DecisionTreeRegressor):
    """An estimator which stores y indices information at fit."""

    def fit(self, X, y):
        # 将 y 索引信息存储在 _sample_indices 属性中
        self._sample_indices = y

# 创建一个 BaggingRegressor 实例，使用 MyEstimator 作为基础估计器，估计器数量为 1，随机种子为 0
clf = BaggingRegressor(estimator=MyEstimator(), n_estimators=1, random_state=0)

# 使用输入数据 X 和标签 y 对 BaggingRegressor 进行训练
clf.fit(X, y)

# 断言检查第一个估计器的 _sample_indices 属性是否与 estimators_samples_ 中的第一个元素相等
assert_array_equal(clf.estimators_[0]._sample_indices, clf.estimators_samples_[0])


# 使用参数化测试来验证 Bagging 模型是否继承了 allow_nan 标签

@pytest.mark.parametrize(
    "bagging, expected_allow_nan",
    [
        (BaggingClassifier(HistGradientBoostingClassifier(max_iter=1)), True),
        (BaggingRegressor(HistGradientBoostingRegressor(max_iter=1)), True),
        (BaggingClassifier(LogisticRegression()), False),
        (BaggingRegressor(SVR()), False),
    ],
)
def test_bagging_allow_nan_tag(bagging, expected_allow_nan):
    """Check that bagging inherits allow_nan tag."""
    # 断言检查 bagging 模型的 allow_nan 标签是否等于预期值 expected_allow_nan
    assert bagging._get_tags()["allow_nan"] == expected_allow_nan


# 使用参数化测试来验证 Bagging 模型是否支持非默认估计器的元数据路由

@pytest.mark.parametrize(
    "model",
    [
        BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=1), n_estimators=1
        ),
        BaggingRegressor(
            estimator=RandomForestRegressor(n_estimators=1), n_estimators=1
        ),
    ],
)
def test_bagging_with_metadata_routing(model):
    """Make sure that metadata routing works with non-default estimator."""
    # 在启用元数据路由的上下文中，使用 iris 数据集对模型进行拟合
    with sklearn.config_context(enable_metadata_routing=True):
        model.fit(iris.data, iris.target)


# 使用参数化测试来验证 Bagging 模型是否能使用不支持元数据路由的估计器

@pytest.mark.parametrize(
    "model",
    [
        BaggingClassifier(
            estimator=AdaBoostClassifier(n_estimators=1, algorithm="SAMME"),
            n_estimators=1,
        ),
        BaggingRegressor(estimator=AdaBoostRegressor(n_estimators=1), n_estimators=1),
    ],
)
def test_bagging_without_support_metadata_routing(model):
    """Make sure that we still can use an estimator that does not implement the
    metadata routing."""
    # 使用 iris 数据集对模型进行拟合
    model.fit(iris.data, iris.target)
```