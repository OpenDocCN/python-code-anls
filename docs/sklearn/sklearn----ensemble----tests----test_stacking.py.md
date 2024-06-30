# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\tests\test_stacking.py`

```
"""Test the stacking classifier and regressor."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re  # 导入正则表达式模块
from unittest.mock import Mock  # 导入 Mock 类

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架
from numpy.testing import assert_array_equal  # 导入 NumPy 测试工具
from scipy import sparse  # 导入 SciPy 稀疏矩阵支持

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone  # 导入 scikit-learn 的基类和混合类
from sklearn.datasets import (  # 导入 scikit-learn 的数据集
    load_breast_cancer,
    load_diabetes,
    load_iris,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.dummy import DummyClassifier, DummyRegressor  # 导入 scikit-learn 的虚拟分类器和回归器
from sklearn.ensemble import (  # 导入 scikit-learn 的集成模型
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.exceptions import ConvergenceWarning, NotFittedError  # 导入 scikit-learn 的异常类
from sklearn.linear_model import (  # 导入 scikit-learn 的线性模型
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split  # 导入 scikit-learn 的模型选择函数和分割器
from sklearn.neighbors import KNeighborsClassifier  # 导入 scikit-learn 的 K 近邻分类器
from sklearn.neural_network import MLPClassifier  # 导入 scikit-learn 的多层感知器分类器
from sklearn.preprocessing import scale  # 导入 scikit-learn 的数据预处理函数
from sklearn.svm import SVC, LinearSVC, LinearSVR  # 导入 scikit-learn 的 SVM 相关类
from sklearn.tests.metadata_routing_common import (  # 导入 scikit-learn 测试中的元数据路由通用函数
    ConsumingClassifier,
    ConsumingRegressor,
    _Registry,
    check_recorded_metadata,
)
from sklearn.utils._mocking import CheckingClassifier  # 导入 scikit-learn 的检查分类器
from sklearn.utils._testing import (  # 导入 scikit-learn 的测试工具
    assert_allclose,
    assert_allclose_dense_sparse,
    ignore_warnings,
)
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS  # 导入 scikit-learn 的修复功能

diabetes = load_diabetes()  # 载入糖尿病数据集
X_diabetes, y_diabetes = diabetes.data, diabetes.target  # 获取糖尿病数据集的特征和目标值
iris = load_iris()  # 载入鸢尾花数据集
X_iris, y_iris = iris.data, iris.target  # 获取鸢尾花数据集的特征和目标值
X_multilabel, y_multilabel = make_multilabel_classification(  # 生成多标签分类数据集
    n_classes=3, random_state=42
)
X_binary, y_binary = make_classification(  # 生成二分类数据集
    n_classes=2, random_state=42
)

@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器进行多组测试参数
    "cv", [3, StratifiedKFold(n_splits=3, shuffle=True, random_state=42)]
)
@pytest.mark.parametrize(
    "final_estimator", [None, RandomForestClassifier(random_state=42)]
)
@pytest.mark.parametrize("passthrough", [False, True])
def test_stacking_classifier_iris(cv, final_estimator, passthrough):
    # 对数据进行预处理，避免收敛警告而不使用管道
    X_train, X_test, y_train, y_test = train_test_split(
        scale(X_iris), y_iris, stratify=y_iris, random_state=42
    )
    estimators = [("lr", LogisticRegression()), ("svc", LinearSVC())]  # 定义基本估算器列表
    clf = StackingClassifier(  # 创建堆叠分类器对象
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
    )
    clf.fit(X_train, y_train)  # 拟合堆叠分类器
    clf.predict(X_test)  # 对测试集进行预测
    clf.predict_proba(X_test)  # 对测试集进行概率预测
    assert clf.score(X_test, y_test) > 0.8  # 断言分类器的预测准确率大于0.8

    X_trans = clf.transform(X_test)  # 对测试集进行转换
    expected_column_count = 10 if passthrough else 6  # 根据是否透传，确定期望的列数
    assert X_trans.shape[1] == expected_column_count  # 断言转换后的特征数是否符合期望
    if passthrough:
        assert_allclose(X_test, X_trans[:, -4:])  # 如果透传，则断言转换后的特征与原始特征接近
    # 设置分类器 clf 的 lr 参数为 "drop"
    clf.set_params(lr="drop")
    # 使用训练集 X_train 和标签 y_train 训练分类器 clf
    clf.fit(X_train, y_train)
    # 对测试集 X_test 进行预测
    clf.predict(X_test)
    # 对测试集 X_test 进行概率预测
    clf.predict_proba(X_test)
    # 如果 final_estimator 为 None，则使用分类器 clf 的 decision_function 方法
    if final_estimator is None:
        # 调用分类器 clf 的 decision_function 方法，生成决策函数的值
        clf.decision_function(X_test)

    # 对测试集 X_test 进行转换，生成转换后的特征矩阵 X_trans
    X_trans = clf.transform(X_test)
    # 如果 passthrough 为真，预期的列数应为 7；否则为 3
    expected_column_count_drop = 7 if passthrough else 3
    # 断言转换后的特征矩阵 X_trans 的列数与预期列数相等
    assert X_trans.shape[1] == expected_column_count_drop
    # 如果 passthrough 为真，断言 X_test 与 X_trans 的后 4 列（passthrough 后的原始数据）近似相等
    if passthrough:
        assert_allclose(X_test, X_trans[:, -4:])
def test_stacking_classifier_drop_column_binary_classification():
    # 检查在二分类中是否会删除列
    # 加载乳腺癌数据集，并返回特征(X)和目标(y)
    X, y = load_breast_cancer(return_X_y=True)
    # 将数据集划分为训练集和测试集，使用数据标准化，并保持类别比例，随机种子为42
    X_train, X_test, y_train, _ = train_test_split(
        scale(X), y, stratify=y, random_state=42
    )

    # 定义两个分类器，分别是逻辑回归和随机森林
    estimators = [
        ("lr", LogisticRegression()),
        ("rf", RandomForestClassifier(random_state=42)),
    ]
    # 创建堆叠分类器对象
    clf = StackingClassifier(estimators=estimators, cv=3)

    # 使用训练集拟合堆叠分类器
    clf.fit(X_train, y_train)
    # 对测试集进行变换
    X_trans = clf.transform(X_test)
    # 断言变换后的特征列数为2
    assert X_trans.shape[1] == 2

    # 重新定义分类器列表，包括逻辑回归和线性支持向量机（LinearSVC）
    estimators = [("lr", LogisticRegression()), ("svc", LinearSVC())]
    # 设置堆叠分类器的参数为上述分类器列表
    clf.set_params(estimators=estimators)

    # 再次使用训练集拟合堆叠分类器
    clf.fit(X_train, y_train)
    # 再次对测试集进行变换
    X_trans = clf.transform(X_test)
    # 再次断言变换后的特征列数为2
    assert X_trans.shape[1] == 2


def test_stacking_classifier_drop_estimator():
    # 为了避免收敛警告，在没有使用管道的情况下预先缩放数据，以供后续断言使用
    # 划分预处理后的鸢尾花数据集为训练集和测试集，随机种子为42
    X_train, X_test, y_train, _ = train_test_split(
        scale(X_iris), y_iris, stratify=y_iris, random_state=42
    )
    # 定义估计器列表，包括逻辑回归和线性支持向量机（LinearSVC）
    estimators = [("lr", "drop"), ("svc", LinearSVC(random_state=0))]
    # 定义随机森林分类器对象
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    # 创建堆叠分类器对象，包含一个估计器：线性支持向量机（LinearSVC）
    clf = StackingClassifier(
        estimators=[("svc", LinearSVC(random_state=0))],
        final_estimator=rf,
        cv=5,
    )
    # 创建另一个堆叠分类器对象，包含两个估计器：逻辑回归和线性支持向量机（LinearSVC）
    clf_drop = StackingClassifier(estimators=estimators, final_estimator=rf, cv=5)

    # 使用训练集拟合堆叠分类器和带有“drop”的堆叠分类器
    clf.fit(X_train, y_train)
    clf_drop.fit(X_train, y_train)
    # 断言两个分类器预测结果的近似性
    assert_allclose(clf.predict(X_test), clf_drop.predict(X_test))
    # 断言两个分类器预测概率的近似性
    assert_allclose(clf.predict_proba(X_test), clf_drop.predict_proba(X_test))
    # 断言两个分类器变换结果的近似性
    assert_allclose(clf.transform(X_test), clf_drop.transform(X_test))


def test_stacking_regressor_drop_estimator():
    # 为了避免收敛警告，在没有使用管道的情况下预先缩放数据，以供后续断言使用
    # 划分预处理后的糖尿病数据集为训练集和测试集，随机种子为42
    X_train, X_test, y_train, _ = train_test_split(
        scale(X_diabetes), y_diabetes, random_state=42
    )
    # 定义估计器列表，包括线性回归和线性支持向量回归（LinearSVR）
    estimators = [("lr", "drop"), ("svr", LinearSVR(random_state=0))]
    # 定义随机森林回归器对象
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    # 创建堆叠回归器对象，包含一个估计器：线性支持向量回归（LinearSVR）
    reg = StackingRegressor(
        estimators=[("svr", LinearSVR(random_state=0))],
        final_estimator=rf,
        cv=5,
    )
    # 创建另一个堆叠回归器对象，包含两个估计器：线性回归和线性支持向量回归（LinearSVR）
    reg_drop = StackingRegressor(estimators=estimators, final_estimator=rf, cv=5)

    # 使用训练集拟合堆叠回归器和带有“drop”的堆叠回归器
    reg.fit(X_train, y_train)
    reg_drop.fit(X_train, y_train)
    # 断言两个回归器预测结果的近似性
    assert_allclose(reg.predict(X_test), reg_drop.predict(X_test))
    # 断言两个回归器变换结果的近似性
    assert_allclose(reg.transform(X_test), reg_drop.transform(X_test))


@pytest.mark.parametrize("cv", [3, KFold(n_splits=3, shuffle=True, random_state=42)])
@pytest.mark.parametrize(
    "final_estimator, predict_params",
    [
        (None, {}),
        (RandomForestRegressor(random_state=42), {}),
        (DummyRegressor(), {"return_std": True}),
    ],
)
@pytest.mark.parametrize("passthrough", [False, True])
def test_stacking_regressor_diabetes(cv, final_estimator, predict_params, passthrough):
    # 使用参数化测试，分别测试是否启用 passthrough 功能
    # 预先缩放数据以避免收敛警告，而不使用管道
    X_train, X_test, y_train, _ = train_test_split(
        scale(X_diabetes), y_diabetes, random_state=42
    )
    # 定义回归器的列表，包含线性回归和线性支持向量回归
    estimators = [("lr", LinearRegression()), ("svr", LinearSVR())]
    # 创建堆叠回归器对象，指定回归器列表、最终估算器、交叉验证次数和是否启用 passthrough
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
    )
    # 使用训练数据拟合堆叠回归器
    reg.fit(X_train, y_train)
    # 对测试数据进行预测
    result = reg.predict(X_test, **predict_params)
    # 根据是否有预测参数来确定预期结果长度
    expected_result_length = 2 if predict_params else 1
    if predict_params:
        # 验证预测结果的长度是否符合预期
        assert len(result) == expected_result_length

    # 对测试数据进行变换
    X_trans = reg.transform(X_test)
    # 根据是否启用 passthrough 来确定预期的列数
    expected_column_count = 12 if passthrough else 2
    # 验证变换后的数据列数是否符合预期
    assert X_trans.shape[1] == expected_column_count
    if passthrough:
        # 如果启用 passthrough，验证变换后的数据是否与原始数据的后 10 列相似
        assert_allclose(X_test, X_trans[:, -10:])

    # 设置堆叠回归器中线性回归器的参数为 "drop"
    reg.set_params(lr="drop")
    # 重新使用训练数据拟合堆叠回归器
    reg.fit(X_train, y_train)
    # 对测试数据进行预测
    reg.predict(X_test)

    # 再次对测试数据进行变换
    X_trans = reg.transform(X_test)
    # 根据是否启用 passthrough 来确定在线性回归器被移除后预期的列数
    expected_column_count_drop = 11 if passthrough else 1
    # 验证变换后的数据列数是否符合预期
    assert X_trans.shape[1] == expected_column_count_drop
    if passthrough:
        # 如果启用 passthrough，验证变换后的数据是否与原始数据的后 10 列相似
        assert_allclose(X_test, X_trans[:, -10:])


@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
def test_stacking_regressor_sparse_passthrough(sparse_container):
    # 检查稀疏 X 矩阵上的 passthrough 行为
    X_train, X_test, y_train, _ = train_test_split(
        sparse_container(scale(X_diabetes)), y_diabetes, random_state=42
    )
    # 定义回归器的列表，包含线性回归和线性支持向量回归
    estimators = [("lr", LinearRegression()), ("svr", LinearSVR())]
    # 创建堆叠回归器对象，指定回归器列表、最终估算器（随机森林）、交叉验证次数和是否启用 passthrough
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    clf = StackingRegressor(
        estimators=estimators, final_estimator=rf, cv=5, passthrough=True
    )
    # 使用训练数据拟合堆叠回归器
    clf.fit(X_train, y_train)
    # 对测试数据进行变换
    X_trans = clf.transform(X_test)
    # 验证变换后的数据是否与原始数据的后 10 列相似
    assert_allclose_dense_sparse(X_test, X_trans[:, -10:])
    # 验证变换后的数据是否为稀疏矩阵
    assert sparse.issparse(X_trans)
    # 验证测试数据的格式是否与变换后的数据相同
    assert X_test.format == X_trans.format


@pytest.mark.parametrize(
    "sparse_container", COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS
)
def test_stacking_classifier_sparse_passthrough(sparse_container):
    # 检查稀疏 X 矩阵上的 passthrough 行为
    X_train, X_test, y_train, _ = train_test_split(
        sparse_container(scale(X_iris)), y_iris, random_state=42
    )
    # 定义分类器的列表，包含逻辑回归和线性支持向量机
    estimators = [("lr", LogisticRegression()), ("svc", LinearSVC())]
    # 创建堆叠分类器对象，指定分类器列表、最终估算器（随机森林）、交叉验证次数和是否启用 passthrough
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf = StackingClassifier(
        estimators=estimators, final_estimator=rf, cv=5, passthrough=True
    )
    # 使用训练数据拟合堆叠分类器
    clf.fit(X_train, y_train)
    # 对测试数据进行变换
    X_trans = clf.transform(X_test)
    # 验证变换后的数据是否与原始数据的后 4 列相似
    assert_allclose_dense_sparse(X_test, X_trans[:, -4:])
    # 验证变换后的数据是否为稀疏矩阵
    assert sparse.issparse(X_trans)
    # 验证测试数据的格式是否与变换后的数据相同
    assert X_test.format == X_trans.format


def test_stacking_classifier_drop_binary_prob():
    # 待实现，目前为空的测试函数，用于测试分类器在丢弃二元概率时的行为
    # 检查分类器是否会为二分类问题丢弃其中一个概率列

    # 仅选择前两类数据
    X_, y_ = scale(X_iris[:100]), y_iris[:100]

    # 定义多个基础分类器
    estimators = [("lr", LogisticRegression()), ("rf", RandomForestClassifier())]

    # 创建堆叠分类器对象
    clf = StackingClassifier(estimators=estimators)

    # 使用选定的数据拟合堆叠分类器
    clf.fit(X_, y_)

    # 生成元特征集合
    X_meta = clf.transform(X_)

    # 断言生成的元特征集合中包含的特征数量为2
    assert X_meta.shape[1] == 2
# 定义一个不带权重的回归器类，继承自RegressorMixin和BaseEstimator
class NoWeightRegressor(RegressorMixin, BaseEstimator):
    
    # 定义fit方法，用于训练模型
    def fit(self, X, y):
        # 创建一个DummyRegressor实例作为基础回归器
        self.reg = DummyRegressor()
        # 调用DummyRegressor的fit方法进行模型拟合
        return self.reg.fit(X, y)

    # 定义predict方法，用于预测
    def predict(self, X):
        # 返回一个全为1的预测结果数组，长度为输入X的样本数
        return np.ones(X.shape[0])


# 定义一个不带权重的分类器类，继承自ClassifierMixin和BaseEstimator
class NoWeightClassifier(ClassifierMixin, BaseEstimator):
    
    # 定义fit方法，用于训练模型
    def fit(self, X, y):
        # 创建一个DummyClassifier实例作为基础分类器，采用"stratified"策略
        self.clf = DummyClassifier(strategy="stratified")
        # 调用DummyClassifier的fit方法进行模型拟合
        return self.clf.fit(X, y)


# 定义测试用例，使用pytest.mark.parametrize装饰器进行参数化测试
@pytest.mark.parametrize(
    "y, params, type_err, msg_err",
    [
        # 第一个测试参数组合：使用y_iris作为标签，空的estimators列表，期望抛出ValueError异常，消息为"Invalid 'estimators' attribute,"
        (y_iris, {"estimators": []}, ValueError, "Invalid 'estimators' attribute,"),
        
        # 第二个测试参数组合：使用y_iris作为标签，包含LogisticRegression和SVC模型的estimators列表，
        # 使用"predict_proba"作为stack_method，期望抛出ValueError异常，消息为"does not implement the method predict_proba"
        (
            y_iris,
            {
                "estimators": [
                    ("lr", LogisticRegression()),
                    ("svm", SVC(max_iter=50_000)),
                ],
                "stack_method": "predict_proba",
            },
            ValueError,
            "does not implement the method predict_proba",
        ),
        
        # 第三个测试参数组合：使用y_iris作为标签，包含LogisticRegression和NoWeightClassifier模型的estimators列表，
        # 期望抛出TypeError异常，消息为"does not support sample weight"
        (
            y_iris,
            {
                "estimators": [
                    ("lr", LogisticRegression()),
                    ("cor", NoWeightClassifier()),
                ]
            },
            TypeError,
            "does not support sample weight",
        ),
        
        # 第四个测试参数组合：使用y_iris作为标签，包含LogisticRegression和LinearSVC模型的estimators列表，
        # 以及NoWeightClassifier模型作为final_estimator，期望抛出TypeError异常，消息为"does not support sample weight"
        (
            y_iris,
            {
                "estimators": [
                    ("lr", LogisticRegression()),
                    ("cor", LinearSVC(max_iter=50_000)),
                ],
                "final_estimator": NoWeightClassifier(),
            },
            TypeError,
            "does not support sample weight",
        ),
    ],
)
def test_stacking_classifier_error(y, params, type_err, msg_err):
    # 使用pytest.raises检测是否抛出预期异常，并匹配预期异常消息
    with pytest.raises(type_err, match=msg_err):
        # 创建StackingClassifier实例，传入参数params和cv=3
        clf = StackingClassifier(**params, cv=3)
        # 调用StackingClassifier的fit方法进行拟合，传入标准化后的特征X_iris，标签y，以及全为1的样本权重数组
        clf.fit(scale(X_iris), y, sample_weight=np.ones(X_iris.shape[0]))


# 定义测试用例，使用pytest.mark.parametrize装饰器进行参数化测试
@pytest.mark.parametrize(
    "y, params, type_err, msg_err",
    [
        # 第一个测试参数组合：使用y_diabetes作为标签，空的estimators列表，期望抛出ValueError异常，消息为"Invalid 'estimators' attribute,"
        (y_diabetes, {"estimators": []}, ValueError, "Invalid 'estimators' attribute,"),
        
        # 第二个测试参数组合：使用y_diabetes作为标签，包含LinearRegression和NoWeightRegressor模型的estimators列表，
        # 期望抛出TypeError异常，消息为"does not support sample weight"
        (
            y_diabetes,
            {"estimators": [("lr", LinearRegression()), ("cor", NoWeightRegressor())]},
            TypeError,
            "does not support sample weight",
        ),
        
        # 第三个测试参数组合：使用y_diabetes作为标签，包含LinearRegression和LinearSVR模型的estimators列表，
        # 以及NoWeightRegressor模型作为final_estimator，期望抛出TypeError异常，消息为"does not support sample weight"
        (
            y_diabetes,
            {
                "estimators": [
                    ("lr", LinearRegression()),
                    ("cor", LinearSVR()),
                ],
                "final_estimator": NoWeightRegressor(),
            },
            TypeError,
            "does not support sample weight",
        ),
    ],
)
def test_stacking_regressor_error(y, params, type_err, msg_err):
    # 使用pytest.raises检测是否抛出预期异常，并匹配预期异常消息
    with pytest.raises(type_err, match=msg_err):
        # 创建StackingRegressor实例，传入参数params和cv=3
        reg = StackingRegressor(**params, cv=3)
        # 调用StackingRegressor的fit方法进行拟合，传入标准化后的特征X_diabetes，标签y，以及全为1的样本权重数组
        reg.fit(scale(X_diabetes), y, sample_weight=np.ones(X_diabetes.shape[0]))
    [
        (  # 第一个元组开始
            StackingClassifier(  # 创建一个堆叠分类器
                estimators=[  # 定义基础分类器的列表
                    ("lr", LogisticRegression(random_state=0)),  # 逻辑回归分类器
                    ("svm", LinearSVC(random_state=0)),  # 线性支持向量分类器
                ]
            ),  # 堆叠分类器对象创建完毕
            X_iris[:100],  # 用于训练的特征数据（取前100个样本）
            y_iris[:100],  # 用于训练的目标数据（取前100个样本的标签）
        ),  # 第一个元组结束，用于堆叠分类器的训练数据
        (  # 第二个元组开始
            StackingRegressor(  # 创建一个堆叠回归器
                estimators=[  # 定义基础回归器的列表
                    ("lr", LinearRegression()),  # 线性回归器
                    ("svm", LinearSVR(random_state=0)),  # 线性支持向量回归器
                ]
            ),  # 堆叠回归器对象创建完毕
            X_diabetes,  # 用于训练的糖尿病数据集的特征数据
            y_diabetes,  # 用于训练的糖尿病数据集的目标数据
        ),  # 第二个元组结束，用于堆叠回归器的训练数据
    ],
    ids=["StackingClassifier", "StackingRegressor"],  # 每个元组的标识，分别对应堆叠分类器和堆叠回归器
# 定义测试函数，用于验证设置随机状态后 CV 结果一致性
def test_stacking_randomness(estimator, X, y):
    # 克隆传入的估计器对象
    estimator_full = clone(estimator)
    # 设置估计器的参数，包括随机状态的 KFold CV 对象
    estimator_full.set_params(
        cv=KFold(shuffle=True, random_state=np.random.RandomState(0))
    )

    # 克隆另一个估计器对象
    estimator_drop = clone(estimator)
    # 设置估计器的参数，标记一个估计器不使用
    estimator_drop.set_params(lr="drop")
    # 设置估计器的参数，包括随机状态的 KFold CV 对象
    estimator_drop.set_params(
        cv=KFold(shuffle=True, random_state=np.random.RandomState(0))
    )

    # 断言两个估计器的变换结果在指定精度下相等
    assert_allclose(
        estimator_full.fit(X, y).transform(X)[:, 1:],
        estimator_drop.fit(X, y).transform(X),
    )


# 定义测试函数，验证默认 CV 中的类分层
def test_stacking_classifier_stratify_default():
    # 创建堆叠分类器对象
    clf = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=10_000)),
            ("svm", LinearSVC(max_iter=10_000)),
        ]
    )
    # 由于 iris 数据集未经过洗牌，简单的 k-fold 在训练过程中无法包含所有 3 个类别
    clf.fit(X_iris, y_iris)


# 使用参数化测试装饰器，定义堆叠模型和数据的组合
@pytest.mark.parametrize(
    "stacker, X, y",
    [
        (
            StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression()),
                    ("svm", LinearSVC(random_state=42)),
                ],
                final_estimator=LogisticRegression(),
                cv=KFold(shuffle=True, random_state=42),
            ),
            *load_breast_cancer(return_X_y=True),
        ),
        (
            StackingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("svm", LinearSVR(random_state=42)),
                ],
                final_estimator=LinearRegression(),
                cv=KFold(shuffle=True, random_state=42),
            ),
            X_diabetes,
            y_diabetes,
        ),
    ],
    ids=["StackingClassifier", "StackingRegressor"],
)
# 定义测试函数，验证样本权重对拟合结果的影响
def test_stacking_with_sample_weight(stacker, X, y):
    # 计算样本数量的一半
    n_half_samples = len(y) // 2
    # 创建总体样本权重数组，前一半样本权重为 0.1，后一半为 0.9
    total_sample_weight = np.array(
        [0.1] * n_half_samples + [0.9] * (len(y) - n_half_samples)
    )
    # 使用 train_test_split 函数划分训练集和测试集，同时获取样本权重
    X_train, X_test, y_train, _, sample_weight_train, _ = train_test_split(
        X, y, total_sample_weight, random_state=42
    )

    # 忽略收敛警告后，拟合堆叠模型到训练数据
    with ignore_warnings(category=ConvergenceWarning):
        stacker.fit(X_train, y_train)
    y_pred_no_weight = stacker.predict(X_test)

    # 忽略收敛警告后，使用单位权重拟合堆叠模型到训练数据
    with ignore_warnings(category=ConvergenceWarning):
        stacker.fit(X_train, y_train, sample_weight=np.ones(y_train.shape))
    y_pred_unit_weight = stacker.predict(X_test)

    # 断言不使用样本权重和使用单位权重拟合的预测结果在指定精度下相等
    assert_allclose(y_pred_no_weight, y_pred_unit_weight)

    # 忽略收敛警告后，使用样本权重拟合堆叠模型到训练数据
    with ignore_warnings(category=ConvergenceWarning):
        stacker.fit(X_train, y_train, sample_weight=sample_weight_train)
    y_pred_biased = stacker.predict(X_test)
    # 断言，验证无权重预测值和有偏预测值的绝对差的总和大于0
    assert np.abs(y_pred_no_weight - y_pred_biased).sum() > 0
# 定义测试函数，用于验证在 fit 方法的所有调用中是否传递了 sample_weight 参数
def test_stacking_classifier_sample_weight_fit_param():
    # 创建一个 StackingClassifier 对象，其中包含一个名为 "lr" 的估计器，
    # 使用 CheckingClassifier 作为其估计器，期望传递 sample_weight 参数
    # 同时设置最终估计器 final_estimator 也期望传递 sample_weight 参数
    stacker = StackingClassifier(
        estimators=[("lr", CheckingClassifier(expected_sample_weight=True))],
        final_estimator=CheckingClassifier(expected_sample_weight=True),
    )
    # 对 StackingClassifier 对象进行拟合，传递样本权重为所有样本的权重均为 1
    stacker.fit(X_iris, y_iris, sample_weight=np.ones(X_iris.shape[0]))


# 使用 pytest 的标记，忽略特定的警告类型，这里是 sklearn 的 ConvergenceWarning
# 使用参数化装饰器，设置多组参数来测试不同情况下的 StackingClassifier 或 StackingRegressor
@pytest.mark.parametrize(
    "stacker, X, y",
    [
        (
            # 第一个参数化测试组：StackingClassifier 包含 LogisticRegression 和 LinearSVC 估计器
            StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression()),
                    ("svm", LinearSVC(random_state=42)),
                ],
                final_estimator=LogisticRegression(),
            ),
            # 载入乳腺癌数据集并返回特征 X 和标签 y
            *load_breast_cancer(return_X_y=True),
        ),
        (
            # 第二个参数化测试组：StackingRegressor 包含 LinearRegression 和 LinearSVR 估计器
            StackingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("svm", LinearSVR(random_state=42)),
                ],
                final_estimator=LinearRegression(),
            ),
            # 使用糖尿病数据集的特征 X_diabetes 和标签 y_diabetes
            X_diabetes,
            y_diabetes,
        ),
    ],
    # 为不同的测试组设置标识符，便于识别测试结果
    ids=["StackingClassifier", "StackingRegressor"],
)
def test_stacking_cv_influence(stacker, X, y):
    # 检查堆叠方法是否影响最终估计器的拟合，但不影响基础估计器的拟合
    # 注意：由于我们不关心收敛问题，所以捕获 ConvergenceWarning 警告
    stacker_cv_3 = clone(stacker)
    stacker_cv_5 = clone(stacker)

    # 分别设置交叉验证折数为 3 和 5
    stacker_cv_3.set_params(cv=3)
    stacker_cv_5.set_params(cv=5)

    # 对数据进行拟合
    stacker_cv_3.fit(X, y)
    stacker_cv_5.fit(X, y)

    # 基础估计器应当是相同的
    for est_cv_3, est_cv_5 in zip(stacker_cv_3.estimators_, stacker_cv_5.estimators_):
        assert_allclose(est_cv_3.coef_, est_cv_5.coef_)

    # 最终估计器应当是不同的
    with pytest.raises(AssertionError, match="Not equal"):
        assert_allclose(
            stacker_cv_3.final_estimator_.coef_, stacker_cv_5.final_estimator_.coef_
        )


# 使用参数化装饰器，设置多组参数来测试不同情况下的 StackingClassifier 或 StackingRegressor
@pytest.mark.parametrize(
    "Stacker, Estimator, stack_method, final_estimator, X, y",
    [
        (
            # 第一个参数化测试组：StackingClassifier 使用 DummyClassifier 作为基础估计器，
            # 使用 predict_proba 作为堆叠方法，最终估计器为 LogisticRegression
            StackingClassifier,
            DummyClassifier,
            "predict_proba",
            LogisticRegression(random_state=42),
            X_iris,
            y_iris,
        ),
        (
            # 第二个参数化测试组：StackingRegressor 使用 DummyRegressor 作为基础估计器，
            # 使用 predict 作为堆叠方法，最终估计器为 LinearRegression
            StackingRegressor,
            DummyRegressor,
            "predict",
            LinearRegression(),
            X_diabetes,
            y_diabetes,
        ),
    ],
)
def test_stacking_prefit(Stacker, Estimator, stack_method, final_estimator, X, y):
    """检查在 `cv='prefit'` 时堆叠方法的行为"""
    # 将数据集分成两部分用于训练和测试
    X_train1, X_train2, y_train1, y_train2 = train_test_split(
        X, y, random_state=42, test_size=0.5
    )
    # 使用 DummyClassifier 或 DummyRegressor 分别对两个部分进行拟合
    estimators = [
        ("d0", Estimator().fit(X_train1, y_train1)),
        ("d1", Estimator().fit(X_train1, y_train1)),
    ]
    # 为后续断言做准备：模拟 `fit` 和 `stack_method` 方法，以便稍后进行断言验证
    for _, estimator in estimators:
        # 使用 Mock 对象替换 `fit` 方法
        estimator.fit = Mock(name="fit")
        # 获取当前估算器对象的 `stack_method` 方法，并创建其 Mock 对象
        stack_func = getattr(estimator, stack_method)
        predict_method_mocked = Mock(side_effect=stack_func)
        # 由于 Mock 对象不会提供 `__name__` 属性，而 Python 方法会有，因此需要在 `_get_response_method` 中使用它
        predict_method_mocked.__name__ = stack_method
        # 将 Mock 对象设置为估算器对象的 `stack_method`
        setattr(estimator, stack_method, predict_method_mocked)

    # 创建 Stacker 对象，使用给定的估算器列表和预先拟合的交叉验证方法
    stacker = Stacker(
        estimators=estimators, cv="prefit", final_estimator=final_estimator
    )
    # 对 Stacker 对象进行拟合操作，使用指定的训练数据 X_train2 和标签数据 y_train2
    stacker.fit(X_train2, y_train2)

    # 断言 Stacker 对象的 estimators_ 属性与估算器列表中的估算器一致
    assert stacker.estimators_ == [estimator for _, estimator in estimators]
    # 断言每个估算器的 fit 方法没有被调用
    assert all(estimator.fit.call_count == 0 for estimator in stacker.estimators_)

    # 断言每个估算器的 `stack_method` 方法被正确调用，传入了 X_train2 参数
    for estimator in stacker.estimators_:
        stack_func_mock = getattr(estimator, stack_method)
        stack_func_mock.assert_called_with(X_train2)
# 使用 pytest.mark.parametrize 装饰器定义多个参数化测试用例
@pytest.mark.parametrize(
    "stacker, X, y",
    [
        # 第一个参数化测试用例，测试 StackingClassifier
        (
            StackingClassifier(
                estimators=[("lr", LogisticRegression()), ("svm", SVC())],
                cv="prefit",
            ),
            X_iris,
            y_iris,
        ),
        # 第二个参数化测试用例，测试 StackingRegressor
        (
            StackingRegressor(
                estimators=[
                    ("lr", LinearRegression()),
                    ("svm", LinearSVR()),
                ],
                cv="prefit",
            ),
            X_diabetes,
            y_diabetes,
        ),
    ],
)
def test_stacking_prefit_error(stacker, X, y):
    # 检查当 cv="prefit" 时，如果基本估计器没有拟合，则引发 NotFittedError
    with pytest.raises(NotFittedError):
        stacker.fit(X, y)


# 使用 pytest.mark.parametrize 装饰器定义另一组参数化测试用例
@pytest.mark.parametrize(
    "make_dataset, Stacking, Estimator",
    [
        # 第一个参数化测试用例，使用 make_classification 和 StackingClassifier
        (make_classification, StackingClassifier, LogisticRegression),
        # 第二个参数化测试用例，使用 make_regression 和 StackingRegressor
        (make_regression, StackingRegressor, LinearRegression),
    ],
)
def test_stacking_without_n_features_in(make_dataset, Stacking, Estimator):
    # Stacking 支持没有 `n_features_in_` 属性的估计器。这是对 issue #17353 的回归测试

    # 自定义的估计器类，继承自 Estimator，但没有 n_features_in_ 属性
    class MyEstimator(Estimator):
        """Estimator without n_features_in_"""

        def fit(self, X, y):
            super().fit(X, y)
            del self.n_features_in_

    # 生成数据集
    X, y = make_dataset(random_state=0, n_samples=100)
    # 创建 Stacking 对象，使用 MyEstimator 作为估计器
    stacker = Stacking(estimators=[("lr", MyEstimator())])

    # 预期的错误消息
    msg = f"{Stacking.__name__} object has no attribute n_features_in_"
    # 使用 pytest 检查是否引发 AttributeError，并匹配错误消息
    with pytest.raises(AttributeError, match=msg):
        stacker.n_features_in_

    # 不应引发错误
    stacker.fit(X, y)

    # 再次检查另一种错误消息
    msg = "'MyEstimator' object has no attribute 'n_features_in_'"
    # 使用 pytest 检查是否引发 AttributeError，并匹配错误消息
    with pytest.raises(AttributeError, match=msg):
        stacker.n_features_in_


# 使用 pytest.mark.parametrize 装饰器定义参数化测试用例
@pytest.mark.parametrize(
    "estimator",
    [
        # MLPClassifier 对象，用于多标签分类，输出每个输出的正类概率的二维数组
        MLPClassifier(random_state=42),
        # RandomForestClassifier 对象，用于多标签分类，输出包含每个输出的每个类别概率的二维数组的列表
        RandomForestClassifier(random_state=42),
    ],
    ids=["MLPClassifier", "RandomForestClassifier"],  # 每个参数化测试用例的标识
)
def test_stacking_classifier_multilabel_predict_proba(estimator):
    """检查多标签分类情况下的 `predict_proba` 方法在 StackingClassifier 中的行为。

    估计器的输出数组不一致，需要确保我们处理所有情况。
    """
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_multilabel, y_multilabel, stratify=y_multilabel, random_state=42
    )
    n_outputs = 3  # 输出的数量

    # 定义估计器列表
    estimators = [("est", estimator)]
    # 创建 StackingClassifier 对象，使用 KNeighborsClassifier 作为最终估计器，使用 'predict_proba' 堆叠方法
    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=KNeighborsClassifier(),
        stack_method="predict_proba",
    ).fit(X_train, y_train)

    # 转换测试集
    X_trans = stacker.transform(X_test)
    # 断言转换后的形状与预期相符
    assert X_trans.shape == (X_test.shape[0], n_outputs)
    # 断言：检查在转换后的特征矩阵 X_trans 的每一行之和是否接近于 1.0。
    # 这里的 np.isclose 函数用于检查浮点数是否在一个容差范围内相等。
    assert not any(np.isclose(X_trans.sum(axis=1), 1.0))

    # 使用 stacker 模型对测试集 X_test 进行预测，保存预测结果到 y_pred 变量中
    y_pred = stacker.predict(X_test)
    # 断言：检查预测结果 y_pred 的形状是否与测试集标签 y_test 的形状相同。
    assert y_pred.shape == y_test.shape
def test_stacking_classifier_multilabel_decision_function():
    """Check the behaviour for the multilabel classification case and the
    `decision_function` stacking method. Only `RidgeClassifier` supports this
    case.
    """
    # 划分训练集和测试集，保持类别比例，设置随机种子
    X_train, X_test, y_train, y_test = train_test_split(
        X_multilabel, y_multilabel, stratify=y_multilabel, random_state=42
    )
    # 输出的类别数目
    n_outputs = 3

    # 使用 RidgeClassifier 作为基础估计器
    estimators = [("est", RidgeClassifier())]
    # 构建堆叠分类器，最终估计器为 KNeighborsClassifier，使用 decision_function 作为堆叠方法
    stacker = StackingClassifier(
        estimators=estimators,
        final_estimator=KNeighborsClassifier(),
        stack_method="decision_function",
    ).fit(X_train, y_train)

    # 对测试集进行转换，检查转换后的形状是否符合预期
    X_trans = stacker.transform(X_test)
    assert X_trans.shape == (X_test.shape[0], n_outputs)

    # 对测试集进行预测，检查预测结果的形状是否与 y_test 相同
    y_pred = stacker.predict(X_test)
    assert y_pred.shape == y_test.shape


@pytest.mark.parametrize("stack_method", ["auto", "predict"])
@pytest.mark.parametrize("passthrough", [False, True])
def test_stacking_classifier_multilabel_auto_predict(stack_method, passthrough):
    """Check the behaviour for the multilabel classification case for stack methods
    supported for all estimators or automatically picked up.
    """
    # 划分训练集和测试集，保持类别比例，设置随机种子
    X_train, X_test, y_train, y_test = train_test_split(
        X_multilabel, y_multilabel, stratify=y_multilabel, random_state=42
    )
    # 复制 y_train，用于后续比较是否改变
    y_train_before_fit = y_train.copy()
    # 输出的类别数目
    n_outputs = 3

    # 多个基础估计器列表
    estimators = [
        ("mlp", MLPClassifier(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("ridge", RidgeClassifier()),
    ]
    # 最终估计器为 KNeighborsClassifier
    final_estimator = KNeighborsClassifier()

    # 构建堆叠分类器，根据参数决定使用的堆叠方法
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        passthrough=passthrough,
        stack_method=stack_method,
    ).fit(X_train, y_train)

    # 确保 `y_train` 在拟合过程中没有改变
    assert_array_equal(y_train_before_fit, y_train)

    # 对测试集进行预测，检查预测结果的形状是否与 y_test 相同
    y_pred = clf.predict(X_test)
    assert y_pred.shape == y_test.shape

    # 根据堆叠方法是否为 "auto"，设置期望的堆叠方法列表
    if stack_method == "auto":
        expected_stack_methods = ["predict_proba", "predict_proba", "decision_function"]
    else:
        expected_stack_methods = ["predict"] * len(estimators)
    # 检查实际使用的堆叠方法是否符合预期
    assert clf.stack_method_ == expected_stack_methods

    # 计算转换后的特征数目
    n_features_X_trans = n_outputs * len(estimators)
    if passthrough:
        n_features_X_trans += X_train.shape[1]
    # 对测试集进行转换，检查转换后的形状是否符合预期
    X_trans = clf.transform(X_test)
    assert X_trans.shape == (X_test.shape[0], n_features_X_trans)

    # 检查类别数组是否按预期设置
    assert_array_equal(clf.classes_, [np.array([0, 1])] * n_outputs)


@pytest.mark.parametrize(
    "stacker, feature_names, X, y, expected_names",
    [
        (  # 创建第一个堆叠分类器
            StackingClassifier(  # 使用堆叠分类器
                estimators=[  # 设定基础分类器列表
                    ("lr", LogisticRegression(random_state=0)),  # 逻辑回归分类器，随机状态为0
                    ("svm", LinearSVC(random_state=0)),  # 线性支持向量机分类器，随机状态为0
                ]
            ),
            iris.feature_names,  # 使用鸢尾花数据集的特征名
            X_iris,  # 鸢尾花数据集的特征数据
            y_iris,  # 鸢尾花数据集的标签数据
            [  # 为分类器命名结果列
                "stackingclassifier_lr0",  # 逻辑回归分类器的第一个结果
                "stackingclassifier_lr1",  # 逻辑回归分类器的第二个结果
                "stackingclassifier_lr2",  # 逻辑回归分类器的第三个结果
                "stackingclassifier_svm0",  # 支持向量机分类器的第一个结果
                "stackingclassifier_svm1",  # 支持向量机分类器的第二个结果
                "stackingclassifier_svm2",  # 支持向量机分类器的第三个结果
            ],
        ),
        (  # 创建第二个堆叠分类器
            StackingClassifier(  # 使用堆叠分类器
                estimators=[  # 设定基础分类器列表
                    ("lr", LogisticRegression(random_state=0)),  # 逻辑回归分类器，随机状态为0
                    ("other", "drop"),  # 被丢弃的占位符
                    ("svm", LinearSVC(random_state=0)),  # 线性支持向量机分类器，随机状态为0
                ]
            ),
            iris.feature_names,  # 使用鸢尾花数据集的特征名
            X_iris[:100],  # 鸢尾花数据集的前100个样本的特征数据
            y_iris[:100],  # 鸢尾花数据集的前100个样本的标签数据（仅保留类别0和1）
            [  # 为分类器命名结果列
                "stackingclassifier_lr",  # 逻辑回归分类器的结果
                "stackingclassifier_svm",  # 支持向量机分类器的结果
            ],
        ),
        (  # 创建堆叠回归器
            StackingRegressor(  # 使用堆叠回归器
                estimators=[  # 设定基础回归器列表
                    ("lr", LinearRegression()),  # 线性回归器
                    ("svm", LinearSVR(random_state=0)),  # 线性支持向量回归器，随机状态为0
                ]
            ),
            diabetes.feature_names,  # 使用糖尿病数据集的特征名
            X_diabetes,  # 糖尿病数据集的特征数据
            y_diabetes,  # 糖尿病数据集的标签数据
            [  # 为回归器命名结果列
                "stackingregressor_lr",  # 线性回归器的结果
                "stackingregressor_svm",  # 支持向量回归器的结果
            ],
        ),
    ],
    ids=[  # 每个分类器/回归器对应的标识符
        "StackingClassifier_multiclass",  # 多类别堆叠分类器
        "StackingClassifier_binary",  # 二元堆叠分类器
        "StackingRegressor",  # 堆叠回归器
    ],
# 参数化测试：测试函数 `test_get_feature_names_out` 被多次调用，每次传入不同的 `passthrough` 值（True 或 False）
@pytest.mark.parametrize("passthrough", [True, False])
def test_get_feature_names_out(
    stacker, feature_names, X, y, expected_names, passthrough
):
    """Check get_feature_names_out works for stacking."""

    # 设置 `stacker` 对象的 `passthrough` 参数
    stacker.set_params(passthrough=passthrough)
    # 使用缩放后的数据 `X` 和标签 `y` 来训练 `stacker`
    stacker.fit(scale(X), y)

    if passthrough:
        # 如果 `passthrough` 是 True，则期望的特征名应包含原特征名和期望的特征名
        expected_names = np.concatenate((expected_names, feature_names))

    # 获取 `get_feature_names_out` 方法返回的特征名
    names_out = stacker.get_feature_names_out(feature_names)
    # 断言返回的特征名与期望的特征名一致
    assert_array_equal(names_out, expected_names)


# 测试函数：测试在 `StackingClassifier` 中使用回归器作为第一层是否有效
def test_stacking_classifier_base_regressor():
    """Check that a regressor can be used as the first layer in `StackingClassifier`."""
    # 将鸢尾花数据集分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        scale(X_iris), y_iris, stratify=y_iris, random_state=42
    )
    # 创建包含 Ridge 回归器的 `StackingClassifier` 对象
    clf = StackingClassifier(estimators=[("ridge", Ridge())])
    # 使用训练集训练 `clf`
    clf.fit(X_train, y_train)
    # 对测试集进行预测
    clf.predict(X_test)
    # 对测试集进行预测概率
    clf.predict_proba(X_test)
    # 断言在测试集上的准确率大于 0.8
    assert clf.score(X_test, y_test) > 0.8


# 测试函数：测试当最终估算器不实现 `decision_function` 方法时是否会引发正确的 AttributeError
def test_stacking_final_estimator_attribute_error():
    """Check that we raise the proper AttributeError when the final estimator
    does not implement the `decision_function` method, which is decorated with
    `available_if`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28108
    """
    # 创建分类数据集
    X, y = make_classification(random_state=42)

    estimators = [
        ("lr", LogisticRegression()),
        ("rf", RandomForestClassifier(n_estimators=2, random_state=42)),
    ]
    # 创建不实现 'decision_function' 方法的 `RandomForestClassifier` 实例
    final_estimator = RandomForestClassifier(n_estimators=2, random_state=42)
    # 创建 `StackingClassifier` 对象，使用上述估算器和最终估算器
    clf = StackingClassifier(
        estimators=estimators, final_estimator=final_estimator, cv=3
    )

    outer_msg = "This 'StackingClassifier' has no attribute 'decision_function'"
    inner_msg = "'RandomForestClassifier' object has no attribute 'decision_function'"
    # 使用 `pytest.raises` 断言在调用 `decision_function` 方法时会引发 AttributeError
    with pytest.raises(AttributeError, match=outer_msg) as exec_info:
        clf.fit(X, y).decision_function(X)
    # 断言引发的异常原因是 AttributeError
    assert isinstance(exec_info.value.__cause__, AttributeError)
    # 断言异常消息中包含正确的内部消息
    assert inner_msg in str(exec_info.value.__cause__)


# Metadata Routing Tests
# ======================


# 参数化测试：测试当 `enable_metadata_routing=False` 时，传递元数据会引发 ValueError
@pytest.mark.parametrize(
    "Estimator, Child",
    [
        (StackingClassifier, ConsumingClassifier),
        (StackingRegressor, ConsumingRegressor),
    ],
)
def test_routing_passed_metadata_not_supported(Estimator, Child):
    """Test that the right error message is raised when metadata is passed while
    not supported when `enable_metadata_routing=False`."""

    with pytest.raises(
        ValueError, match="is only supported if enable_metadata_routing=True"
    ):
        # 创建 `Estimator` 对象，传入包含 `Child` 实例的元组，并尝试拟合数据，传递了元数据 "a"
        Estimator(["clf", Child()]).fit(
            X_iris, y_iris, sample_weight=[1, 1, 1, 1, 1], metadata="a"
        )


# 参数化测试：测试使用 `enable_slep006` 标记，确保 SLEP006 规范下的行为
@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize(
    "Estimator, Child",
    [
        (
    [
        # 元组1: 包含 StackingClassifier 和 ConsumingClassifier 类
        (StackingClassifier, ConsumingClassifier),
        # 元组2: 包含 StackingRegressor 和 ConsumingRegressor 类
        (StackingRegressor, ConsumingRegressor),
    ],
def test_get_metadata_routing_without_fit(Estimator, Child):
    # 测试在调用 fit 之前调用 metadata_routing() 不会引发异常
    est = Estimator([("sub_est", Child())])
    est.get_metadata_routing()


@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize(
    "Estimator, Child",
    [
        (StackingClassifier, ConsumingClassifier),
        (StackingRegressor, ConsumingRegressor),
    ],
)
@pytest.mark.parametrize(
    "prop, prop_value", [("sample_weight", np.ones(X_iris.shape[0])), ("metadata", "a")]
)
def test_metadata_routing_for_stacking_estimators(Estimator, Child, prop, prop_value):
    """Test that metadata is routed correctly for Stacking*."""
    
    # 创建 StackingEstimator 对象，包含多个子估计器和一个最终估计器
    est = Estimator(
        [
            (
                "sub_est1",
                Child(registry=_Registry()).set_fit_request(**{prop: True}),
            ),
            (
                "sub_est2",
                Child(registry=_Registry()).set_fit_request(**{prop: True}),
            ),
        ],
        final_estimator=Child(registry=_Registry()).set_predict_request(**{prop: True}),
    )

    # 调用 fit 和 fit_transform 方法
    est.fit(X_iris, y_iris, **{prop: prop_value})
    est.fit_transform(X_iris, y_iris, **{prop: prop_value})

    # 调用 predict 方法
    est.predict(X_iris, **{prop: prop_value})

    # 遍历子估计器列表，验证每个子估计器的注册情况和记录的元数据
    for estimator in est.estimators:
        # 访问估计器对象的注册表
        registry = estimator[1].registry
        assert len(registry)
        for sub_est in registry:
            # 检查记录的元数据是否符合预期
            check_recorded_metadata(
                obj=sub_est,
                method="fit",
                parent="fit",
                split_params=(prop),
                **{prop: prop_value},
            )
    
    # 访问最终估计器的注册表
    registry = est.final_estimator_.registry
    assert len(registry)
    # 检查记录的元数据是否符合预期
    check_recorded_metadata(
        obj=registry[-1],
        method="predict",
        parent="predict",
        split_params=(prop),
        **{prop: prop_value},
    )


@pytest.mark.usefixtures("enable_slep006")
@pytest.mark.parametrize(
    "Estimator, Child",
    [
        (StackingClassifier, ConsumingClassifier),
        (StackingRegressor, ConsumingRegressor),
    ],
)
def test_metadata_routing_error_for_stacking_estimators(Estimator, Child):
    """Test that the right error is raised when metadata is not requested."""
    
    # 准备 sample_weight 和 metadata
    sample_weight, metadata = np.ones(X_iris.shape[0]), "a"

    # 创建 StackingEstimator 对象，包含单个子估计器
    est = Estimator([("sub_est", Child())])

    # 准备错误消息
    error_message = (
        "[sample_weight, metadata] are passed but are not explicitly set as requested"
        f" or not requested for {Child.__name__}.fit"
    )

    # 断言会抛出 ValueError 异常，且异常消息符合预期
    with pytest.raises(ValueError, match=re.escape(error_message)):
        est.fit(X_iris, y_iris, sample_weight=sample_weight, metadata=metadata)

# End of Metadata Routing Tests
# =============================
```