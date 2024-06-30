# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\tests\test_mlp.py`

```
"""
Testing for Multi-layer Perceptron module (sklearn.neural_network)
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re  # 导入正则表达式模块
import sys  # 导入系统相关的功能模块
import warnings  # 导入警告处理模块
from io import StringIO  # 导入字符串I/O模块

import joblib  # 导入持久化模块
import numpy as np  # 导入数值计算模块numpy
import pytest  # 导入单元测试框架pytest
from numpy.testing import (  # 从numpy.testing中导入断言方法
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

from sklearn.datasets import (  # 从sklearn.datasets中导入数据集相关模块
    load_digits,
    load_iris,
    make_multilabel_classification,
    make_regression,
)
from sklearn.exceptions import ConvergenceWarning  # 导入收敛警告异常类
from sklearn.metrics import roc_auc_score  # 导入ROC AUC评估指标
from sklearn.neural_network import MLPClassifier, MLPRegressor  # 导入多层感知器分类和回归模型
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, scale  # 导入数据预处理相关模块
from sklearn.utils._testing import ignore_warnings  # 导入用于忽略警告的测试工具
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入用于稀疏矩阵容器的修复模块

ACTIVATION_TYPES = ["identity", "logistic", "tanh", "relu"]  # 激活函数类型列表

X_digits, y_digits = load_digits(n_class=3, return_X_y=True)  # 加载手写数字数据集

X_digits_multi = MinMaxScaler().fit_transform(X_digits[:200])  # 对手写数字数据集进行最小-最大标准化处理
y_digits_multi = y_digits[:200]

X_digits, y_digits = load_digits(n_class=2, return_X_y=True)  # 重新加载手写数字数据集，设置为二分类

X_digits_binary = MinMaxScaler().fit_transform(X_digits[:200])  # 对二分类手写数字数据集进行最小-最大标准化处理
y_digits_binary = y_digits[:200]

classification_datasets = [  # 分类任务数据集列表
    (X_digits_multi, y_digits_multi),
    (X_digits_binary, y_digits_binary),
]

X_reg, y_reg = make_regression(  # 生成回归任务数据集
    n_samples=200, n_features=10, bias=20.0, noise=100.0, random_state=7
)
y_reg = scale(y_reg)  # 对回归目标变量进行标准化处理
regression_datasets = [(X_reg, y_reg)]

iris = load_iris()  # 加载鸢尾花数据集

X_iris = iris.data  # 鸢尾花特征数据
y_iris = iris.target  # 鸢尾花目标数据


def test_alpha():
    # Test that larger alpha yields weights closer to zero
    X = X_digits_binary[:100]  # 使用前100个样本进行测试
    y = y_digits_binary[:100]

    alpha_vectors = []  # 存储不同alpha值下的权重绝对值和

    alpha_values = np.arange(2)  # 设置alpha值为0和1
    absolute_sum = lambda x: np.sum(np.abs(x))  # 定义计算绝对值和的函数

    for alpha in alpha_values:
        mlp = MLPClassifier(hidden_layer_sizes=10, alpha=alpha, random_state=1)  # 创建MLP分类器对象
        with ignore_warnings(category=ConvergenceWarning):  # 忽略收敛警告
            mlp.fit(X, y)  # 拟合模型
        alpha_vectors.append(
            np.array([absolute_sum(mlp.coefs_[0]), absolute_sum(mlp.coefs_[1])])
        )  # 计算并存储第一和第二层权重的绝对值和

    for i in range(len(alpha_values) - 1):
        assert (alpha_vectors[i] > alpha_vectors[i + 1]).all()  # 断言：较大的alpha值对应的权重绝对值和更接近零


def test_fit():
    # Test that the algorithm solution is equal to a worked out example.
    X = np.array([[0.6, 0.8, 0.7]])  # 输入特征数据
    y = np.array([0])  # 目标数据为0
    mlp = MLPClassifier(
        solver="sgd",  # 使用随机梯度下降算法求解
        learning_rate_init=0.1,  # 学习率初始化为0.1
        alpha=0.1,  # 正则化参数alpha为0.1
        activation="logistic",  # 使用逻辑斯蒂激活函数
        random_state=1,  # 设置随机种子为1
        max_iter=1,  # 最大迭代次数为1
        hidden_layer_sizes=2,  # 隐藏层神经元数量为2
        momentum=0,  # 动量参数为0
    )
    # set weights
    mlp.coefs_ = [0] * 2  # 初始化权重数组
    mlp.intercepts_ = [0] * 2  # 初始化偏置数组
    mlp.n_outputs_ = 1  # 输出层节点数量为1
    mlp.coefs_[0] = np.array([[0.1, 0.2], [0.3, 0.1], [0.5, 0]])  # 设置第一层权重
    mlp.coefs_[1] = np.array([[0.1], [0.2]])  # 设置输出层权重
    mlp.intercepts_[0] = np.array([0.1, 0.1])  # 设置第一层偏置
    mlp.intercepts_[1] = np.array([1.0])  # 设置输出层偏置
    mlp._coef_grads = [] * 2  # 初始化权重梯度数组
    mlp._intercept_grads = [] * 2  # 初始化偏置梯度数组
    mlp.n_features_in_ = 3  # 输入特征维度为3

    # Initialize parameters
    # 初始化 MLP 模型的迭代次数为 0
    mlp.n_iter_ = 0
    # 设置 MLP 模型的学习率为 0.1
    mlp.learning_rate_ = 0.1

    # 计算 MLP 模型的层数，并设置为 3
    mlp.n_layers_ = 3

    # 预先分配梯度矩阵的空间，根据层数 - 1 进行初始化
    mlp._coef_grads = [0] * (mlp.n_layers_ - 1)
    mlp._intercept_grads = [0] * (mlp.n_layers_ - 1)

    # 设置输出层的激活函数为 logistic
    mlp.out_activation_ = "logistic"
    # 初始化 MLP 模型的迭代次数为 0
    mlp.t_ = 0
    # 初始化 MLP 模型的最佳损失为无穷大
    mlp.best_loss_ = np.inf
    # 初始化损失曲线为空列表
    mlp.loss_curve_ = []
    # 初始化没有改善的次数为 0
    mlp._no_improvement_count = 0

    # 初始化偏置的速度矩阵，根据模型的偏置进行初始化
    mlp._intercept_velocity = [
        np.zeros_like(intercepts) for intercepts in mlp.intercepts_
    ]
    # 初始化系数的速度矩阵，根据模型的系数进行初始化
    mlp._coef_velocity = [np.zeros_like(coefs) for coefs in mlp.coefs_]

    # 使用部分拟合方法对模型进行一次迭代更新，传入训练数据 X 和标签 y，以及类别列表
    mlp.partial_fit(X, y, classes=[0, 1])

    # 断言模型的第一个系数矩阵与预期值的近似相等，精度为小数点后 3 位
    assert_almost_equal(
        mlp.coefs_[0],
        np.array([[0.098, 0.195756], [0.2956664, 0.096008], [0.4939998, -0.002244]]),
        decimal=3,
    )
    # 断言模型的第二个系数矩阵与预期值的近似相等，精度为小数点后 3 位
    assert_almost_equal(mlp.coefs_[1], np.array([[0.04706], [0.154089]]), decimal=3)
    # 断言模型的第一个偏置向量与预期值的近似相等，精度为小数点后 3 位
    assert_almost_equal(mlp.intercepts_[0], np.array([0.098333, 0.09626]), decimal=3)
    # 断言模型的第二个偏置值与预期值的近似相等，精度为小数点后 3 位
    assert_almost_equal(mlp.intercepts_[1], np.array(0.9235), decimal=3)
    # 进行断言以验证神经网络模型在给定输入数据集 X 上的预测概率
    assert_almost_equal(mlp.predict_proba(X)[0, 1], 0.739, decimal=3)
def test_gradient():
    # Test gradient.

    # This makes sure that the activation functions and their derivatives
    # are correct. The numerical and analytical computation of the gradient
    # should be close.
    for n_labels in [2, 3]:
        n_samples = 5
        n_features = 10
        random_state = np.random.RandomState(seed=42)
        X = random_state.rand(n_samples, n_features)
        y = 1 + np.mod(np.arange(n_samples) + 1, n_labels)
        Y = LabelBinarizer().fit_transform(y)

        for activation in ACTIVATION_TYPES:
            # Initialize MLP classifier with specified parameters
            mlp = MLPClassifier(
                activation=activation,
                hidden_layer_sizes=10,
                solver="lbfgs",
                alpha=1e-5,
                learning_rate_init=0.2,
                max_iter=1,
                random_state=1,
            )
            # Fit MLP classifier to the data
            mlp.fit(X, y)

            # Flatten the coefficients and intercepts into a single vector
            theta = np.hstack([l.ravel() for l in mlp.coefs_ + mlp.intercepts_])

            # Determine the units in each layer of the MLP
            layer_units = [X.shape[1]] + [mlp.hidden_layer_sizes] + [mlp.n_outputs_]

            activations = []
            deltas = []
            coef_grads = []
            intercept_grads = []

            activations.append(X)
            # Compute activations, deltas, coefficient gradients, and intercept gradients for each layer
            for i in range(mlp.n_layers_ - 1):
                activations.append(np.empty((X.shape[0], layer_units[i + 1])))
                deltas.append(np.empty((X.shape[0], layer_units[i + 1])))

                fan_in = layer_units[i]
                fan_out = layer_units[i + 1]
                coef_grads.append(np.empty((fan_in, fan_out)))
                intercept_grads.append(np.empty(fan_out))

            # analytically compute the gradients using the loss gradient function
            def loss_grad_fun(t):
                return mlp._loss_grad_lbfgs(
                    t, X, Y, activations, deltas, coef_grads, intercept_grads
                )

            [value, grad] = loss_grad_fun(theta)
            numgrad = np.zeros(np.size(theta))
            n = np.size(theta, 0)
            E = np.eye(n)
            epsilon = 1e-5
            # numerically compute the gradients using finite differences
            for i in range(n):
                dtheta = E[:, i] * epsilon
                numgrad[i] = (
                    loss_grad_fun(theta + dtheta)[0] - loss_grad_fun(theta - dtheta)[0]
                ) / (epsilon * 2.0)
            # Assert that numerical and analytical gradients are almost equal
            assert_almost_equal(numgrad, grad)


@pytest.mark.parametrize("X,y", classification_datasets)
def test_lbfgs_classification(X, y):
    # Test lbfgs on classification.
    # It should achieve a score higher than 0.95 for the binary and multi-class
    # versions of the digits dataset.
    X_train = X[:150]
    y_train = y[:150]
    X_test = X[150:]
    expected_shape_dtype = (X_test.shape[0], y_train.dtype.kind)
    for activation in ACTIVATION_TYPES:
        # 对每种激活函数类型循环迭代，ACTIVATION_TYPES 是包含多种激活函数的列表
        mlp = MLPClassifier(
            solver="lbfgs",
            hidden_layer_sizes=50,
            max_iter=150,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        # 创建一个多层感知器分类器对象（MLPClassifier）
        mlp.fit(X_train, y_train)
        # 使用训练数据（X_train, y_train）来训练多层感知器分类器
        y_predict = mlp.predict(X_test)
        # 对测试数据（X_test）进行预测，并将结果保存到 y_predict 变量中
        assert mlp.score(X_train, y_train) > 0.95
        # 断言训练集上的准确率（score）大于 0.95
        assert (y_predict.shape[0], y_predict.dtype.kind) == expected_shape_dtype
        # 断言预测结果 y_predict 的形状和数据类型与预期的形状和数据类型（expected_shape_dtype）相符
@pytest.mark.parametrize("X,y", regression_datasets)
def test_lbfgs_regression(X, y):
    # 对回归数据集使用 lbfgs 求解器进行测试
    for activation in ACTIVATION_TYPES:
        # 创建 MLPRegressor 多层感知机回归模型
        mlp = MLPRegressor(
            solver="lbfgs",
            hidden_layer_sizes=50,
            max_iter=150,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        # 使用数据 X, y 进行模型训练
        mlp.fit(X, y)
        if activation == "identity":
            # 对于恒等激活函数，确保模型得分大于0.80
            assert mlp.score(X, y) > 0.80
        else:
            # 非线性模型应该比线性模型表现更好
            assert mlp.score(X, y) > 0.98


@pytest.mark.parametrize("X,y", classification_datasets)
def test_lbfgs_classification_maxfun(X, y):
    # 测试 lbfgs 参数 max_fun
    # 它独立限制 lbfgs 的迭代次数
    max_fun = 10
    # 分类测试
    for activation in ACTIVATION_TYPES:
        # 创建 MLPClassifier 多层感知机分类模型
        mlp = MLPClassifier(
            solver="lbfgs",
            hidden_layer_sizes=50,
            max_iter=150,
            max_fun=max_fun,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        # 使用 pytest 捕获收敛警告
        with pytest.warns(ConvergenceWarning):
            mlp.fit(X, y)
            # 确保迭代次数不超过 max_fun
            assert max_fun >= mlp.n_iter_


@pytest.mark.parametrize("X,y", regression_datasets)
def test_lbfgs_regression_maxfun(X, y):
    # 测试 lbfgs 参数 max_fun
    # 它独立限制 lbfgs 的迭代次数
    max_fun = 10
    # 回归测试
    for activation in ACTIVATION_TYPES:
        # 创建 MLPRegressor 多层感知机回归模型
        mlp = MLPRegressor(
            solver="lbfgs",
            hidden_layer_sizes=50,
            tol=0.0,
            max_iter=150,
            max_fun=max_fun,
            shuffle=True,
            random_state=1,
            activation=activation,
        )
        # 使用 pytest 捕获收敛警告
        with pytest.warns(ConvergenceWarning):
            mlp.fit(X, y)
            # 确保迭代次数不超过 max_fun
            assert max_fun >= mlp.n_iter_


def test_learning_rate_warmstart():
    # 测试 warm_start 是否重用过去的解决方案
    X = [[3, 2], [1, 6], [5, 6], [-2, -4]]
    y = [1, 1, 1, 0]
    for learning_rate in ["invscaling", "constant"]:
        # 创建 MLPClassifier 多层感知机分类模型，使用随机梯度下降 solver
        mlp = MLPClassifier(
            solver="sgd",
            hidden_layer_sizes=4,
            learning_rate=learning_rate,
            max_iter=1,
            power_t=0.25,
            warm_start=True,
        )
        # 忽略收敛警告
        with ignore_warnings(category=ConvergenceWarning):
            mlp.fit(X, y)
            prev_eta = mlp._optimizer.learning_rate
            mlp.fit(X, y)
            post_eta = mlp._optimizer.learning_rate

        if learning_rate == "constant":
            # 对于常数学习率，确保前后学习率相同
            assert prev_eta == post_eta
        elif learning_rate == "invscaling":
            # 对于 invscaling 学习率，确保学习率计算正确
            assert mlp.learning_rate_init / pow(8 + 1, mlp.power_t) == post_eta


def test_multilabel_classification():
    # 测试多标签分类是否按预期工作
    # 测试 fit 方法
    # 使用 make_multilabel_classification 生成一个多标签分类的数据集 X, y
    X, y = make_multilabel_classification(
        n_samples=50, random_state=0, return_indicator=True
    )
    
    # 创建一个多层感知机分类器（MLPClassifier）对象 mlp
    mlp = MLPClassifier(
        solver="lbfgs",  # 指定求解器为 "lbfgs"，适用于小数据集
        hidden_layer_sizes=50,  # 设定隐藏层大小为 50
        alpha=1e-5,  # 正则化参数设定为 1e-5
        max_iter=150,  # 最大迭代次数设定为 150
        random_state=0,  # 随机种子设定为 0，保证可复现性
        activation="logistic",  # 激活函数设定为逻辑斯蒂（logistic）
        learning_rate_init=0.2,  # 初始学习率设定为 0.2
    )
    
    # 使用 mlp 对象拟合数据集 X, y
    mlp.fit(X, y)
    
    # 断言模型在训练集 X, y 上的准确率大于 0.97
    assert mlp.score(X, y) > 0.97

    # 测试 partial_fit 方法
    mlp = MLPClassifier(
        solver="sgd",  # 指定求解器为 "sgd"，随机梯度下降
        hidden_layer_sizes=50,  # 设定隐藏层大小为 50
        max_iter=150,  # 最大迭代次数设定为 150
        random_state=0,  # 随机种子设定为 0，保证可复现性
        activation="logistic",  # 激活函数设定为逻辑斯蒂（logistic）
        alpha=1e-5,  # 正则化参数设定为 1e-5
        learning_rate_init=0.2,  # 初始学习率设定为 0.2
    )
    
    # 使用 partial_fit 方法进行多次迭代更新模型，指定类别为 [0, 1, 2, 3, 4]
    for i in range(100):
        mlp.partial_fit(X, y, classes=[0, 1, 2, 3, 4])
    
    # 断言模型在训练集 X, y 上的准确率大于 0.9
    assert mlp.score(X, y) > 0.9

    # 确保在默认情况下（多标签分类），早停机制仍然有效，因为数据拆分是分层的
    mlp = MLPClassifier(early_stopping=True)  # 创建开启了早停机制的 MLPClassifier 对象
    mlp.fit(X, y).predict(X)  # 拟合数据集 X, y 并预测结果
# 测试多输出回归是否按预期工作
def test_multioutput_regression():
    # 生成包含 200 个样本和 5 个目标的回归数据集
    X, y = make_regression(n_samples=200, n_targets=5)
    # 创建一个多层感知机回归器对象，使用 'lbfgs' 求解器，50 个隐藏层单元，最大迭代次数为 200，随机状态为 1
    mlp = MLPRegressor(
        solver="lbfgs", hidden_layer_sizes=50, max_iter=200, random_state=1
    )
    # 使用数据集 X, y 进行拟合
    mlp.fit(X, y)
    # 断言模型在训练数据 X, y 上的评分大于 0.9
    assert mlp.score(X, y) > 0.9


# 测试 partial_fit 方法在分类器类别不匹配时是否引发错误
def test_partial_fit_classes_error():
    # 定义输入特征 X 和类别标签 y
    X = [[3, 2]]
    y = [0]
    # 创建一个多层感知机分类器对象，使用 'sgd' 求解器
    clf = MLPClassifier(solver="sgd")
    # 使用部分拟合方法进行模型训练，传入类别参数 [0, 1]
    clf.partial_fit(X, y, classes=[0, 1])
    # 使用部分拟合方法传入不匹配的类别参数 [1, 2]，断言应引发 ValueError 异常
    with pytest.raises(ValueError):
        clf.partial_fit(X, y, classes=[1, 2])


# 测试分类问题中 partial_fit 方法的效果
def test_partial_fit_classification():
    # 对于每个分类数据集 (X, y)，分别进行测试
    for X, y in classification_datasets:
        # 创建一个多层感知机分类器对象，使用 'sgd' 求解器，最大迭代次数为 100，随机状态为 1，容忍度为 0，学习率初始值为 0.2，正则化参数为 1e-5
        mlp = MLPClassifier(
            solver="sgd",
            max_iter=100,
            random_state=1,
            tol=0,
            alpha=1e-5,
            learning_rate_init=0.2,
        )

        # 忽略收敛警告进行模型拟合
        with ignore_warnings(category=ConvergenceWarning):
            mlp.fit(X, y)
        
        # 获取第一次预测结果
        pred1 = mlp.predict(X)
        
        # 重新创建一个多层感知机分类器对象，使用 'sgd' 求解器，随机状态为 1，正则化参数为 1e-5，学习率初始值为 0.2
        mlp = MLPClassifier(
            solver="sgd", random_state=1, alpha=1e-5, learning_rate_init=0.2
        )
        
        # 使用部分拟合方法进行多次模型更新
        for i in range(100):
            mlp.partial_fit(X, y, classes=np.unique(y))
        
        # 获取第二次预测结果
        pred2 = mlp.predict(X)
        
        # 断言两次预测结果应完全相等
        assert_array_equal(pred1, pred2)
        
        # 断言模型在数据集 (X, y) 上的评分大于 0.95
        assert mlp.score(X, y) > 0.95


# 针对未见过的类别进行 partial_fit 方法的测试
def test_partial_fit_unseen_classes():
    # 针对 bug 6994 的非回归测试
    # 测试 partial_fit 方法在标签错误时的处理

    # 创建一个多层感知机分类器对象，随机状态为 0
    clf = MLPClassifier(random_state=0)
    
    # 使用部分拟合方法传入部分数据 [[1], [2], [3]] 和标签 ["a", "b", "c"]，类别参数为 ["a", "b", "c", "d"]
    clf.partial_fit([[1], [2], [3]], ["a", "b", "c"], classes=["a", "b", "c", "d"])
    
    # 再次使用部分拟合方法传入数据 [[4]] 和标签 ["d"]
    clf.partial_fit([[4]], ["d"])
    
    # 断言模型在数据集 [[1], [2], [3], [4]] 和标签 ["a", "b", "c", "d"] 上的评分大于 0
    assert clf.score([[1], [2], [3], [4]], ["a", "b", "c", "d"]) > 0


# 测试 partial_fit 方法在回归问题上的效果
def test_partial_fit_regression():
    # 对于回归数据集 (X_reg, y_reg)，分别进行测试
    X = X_reg
    y = y_reg

    # 分别测试动量参数为 0 和 0.9 时的效果
    for momentum in [0, 0.9]:
        # 创建一个多层感知机回归器对象，使用 'sgd' 求解器，最大迭代次数为 100，激活函数为 'relu'，随机状态为 1，学习率初始值为 0.01，批量大小为 X 的样本数，动量参数为当前循环的 momentum 值
        mlp = MLPRegressor(
            solver="sgd",
            max_iter=100,
            activation="relu",
            random_state=1,
            learning_rate_init=0.01,
            batch_size=X.shape[0],
            momentum=momentum,
        )
        
        # 捕获警告信息
        with warnings.catch_warnings(record=True):
            # 捕获收敛警告
            mlp.fit(X, y)
        
        # 获取第一次预测结果
        pred1 = mlp.predict(X)
        
        # 重新创建一个多层感知机回归器对象，使用 'sgd' 求解器，激活函数为 'relu'，随机状态为 1，学习率初始值为 0.01，批量大小为 X 的样本数，动量参数为当前循环的 momentum 值
        mlp = MLPRegressor(
            solver="sgd",
            activation="relu",
            learning_rate_init=0.01,
            random_state=1,
            batch_size=X.shape[0],
            momentum=momentum,
        )
        
        # 使用部分拟合方法进行多次模型更新
        for i in range(100):
            mlp.partial_fit(X, y)
        
        # 获取第二次预测结果
        pred2 = mlp.predict(X)
        
        # 断言两次预测结果应在数值上非常接近
        assert_allclose(pred1, pred2)
        
        # 计算模型在数据集 (X, y) 上的评分
        score = mlp.score(X, y)
        
        # 断言模型的评分大于 0.65
        assert score > 0.65


# 测试 partial_fit 方法的错误处理
def test_partial_fit_errors():
    # 定义输入特征 X 和类别标签 y
    X = [[3, 2], [1, 6]]
    y = [1, 0]

    # 没有传入类别参数的情况下使用部分拟合方法
    # 使用 pytest 来验证是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        # 创建一个 MLPClassifier 实例，使用 solver 参数为 "sgd"，然后调用 partial_fit 方法
        MLPClassifier(solver="sgd").partial_fit(X, y, classes=[2])

    # 断言 MLPClassifier 使用 "lbfgs" solver 不支持 partial_fit 方法
    assert not hasattr(MLPClassifier(solver="lbfgs"), "partial_fit")
# 检查 MLPRegressor 在处理非有限参数值时是否引发 ValueError 异常
def test_nonfinite_params():
    # 创建随机数生成器，种子为0
    rng = np.random.RandomState(0)
    # 设置样本数量
    n_samples = 10
    # 获取 np.float64 的最大值
    fmax = np.finfo(np.float64).max
    # 生成一个大小为 (n_samples, 2) 的随机数组，元素范围在 0 到 fmax 之间
    X = fmax * rng.uniform(size=(n_samples, 2))
    # 生成一个大小为 n_samples 的标准正态分布随机数组
    y = rng.standard_normal(size=n_samples)

    # 初始化 MLPRegressor 对象
    clf = MLPRegressor()
    # 定义错误消息内容
    msg = (
        "Solver produced non-finite parameter weights. The input data may contain large"
        " values and need to be preprocessed."
    )
    # 使用 pytest 来检查是否会引发 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, y)


# 测试二分类情况下 predict_proba 的预期行为
def test_predict_proba_binary():
    # 使用前50个样本
    X = X_digits_binary[:50]
    y = y_digits_binary[:50]

    # 初始化 MLPClassifier 对象，设置隐藏层大小为 5，激活函数为 logistic，随机种子为 1
    clf = MLPClassifier(hidden_layer_sizes=5, activation="logistic", random_state=1)
    # 忽略收敛警告进行拟合
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    # 获取预测概率和对数概率
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)

    # 确定样本数量和类别数
    (n_samples, n_classes) = y.shape[0], 2

    # 找到每个样本的最大概率类别
    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    # 断言预测概率的形状是否正确
    assert y_proba.shape == (n_samples, n_classes)
    # 断言最大概率类别相等
    assert_array_equal(proba_max, proba_log_max)
    # 断言对数概率和自然对数预测概率之间的接近程度
    assert_allclose(y_log_proba, np.log(y_proba))

    # 断言 ROC AUC 分数等于 1.0
    assert roc_auc_score(y, y_proba[:, 1]) == 1.0


# 测试多分类情况下 predict_proba 的预期行为
def test_predict_proba_multiclass():
    # 使用前10个样本
    X = X_digits_multi[:10]
    y = y_digits_multi[:10]

    # 初始化 MLPClassifier 对象，设置隐藏层大小为 5
    clf = MLPClassifier(hidden_layer_sizes=5)
    # 忽略收敛警告进行拟合
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    # 获取预测概率和对数概率
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)

    # 确定样本数量和类别数
    (n_samples, n_classes) = y.shape[0], np.unique(y).size

    # 找到每个样本的最大概率类别
    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    # 断言预测概率的形状是否正确
    assert y_proba.shape == (n_samples, n_classes)
    # 断言最大概率类别相等
    assert_array_equal(proba_max, proba_log_max)
    # 断言对数概率和自然对数预测概率之间的接近程度
    assert_allclose(y_log_proba, np.log(y_proba))


# 测试多标签情况下 predict_proba 的预期行为
def test_predict_proba_multilabel():
    # 生成多标签分类数据集，返回指示器矩阵
    X, Y = make_multilabel_classification(
        n_samples=50, random_state=0, return_indicator=True
    )
    # 获取样本数量和类别数
    n_samples, n_classes = Y.shape

    # 初始化 MLPClassifier 对象，设置 solver="lbfgs"，隐藏层大小为 30，随机种子为 0
    clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=30, random_state=0)
    # 使用数据进行拟合
    clf.fit(X, Y)
    # 获取预测概率
    y_proba = clf.predict_proba(X)

    # 断言预测概率的形状是否正确
    assert y_proba.shape == (n_samples, n_classes)
    # 断言预测概率是否大于 0.5
    assert_array_equal(y_proba > 0.5, Y)

    # 获取对数预测概率
    y_log_proba = clf.predict_log_proba(X)
    # 找到每个样本的最大概率类别
    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    # 断言概率和的误差是否大于 1e-10
    assert (y_proba.sum(1) - 1).dot(y_proba.sum(1) - 1) > 1e-10
    # 断言最大概率类别相等
    assert_array_equal(proba_max, proba_log_max)
    # 断言对数概率和自然对数预测概率之间的接近程度
    assert_allclose(y_log_proba, np.log(y_proba))


# 测试 shuffle 参数对训练过程的影响
def test_shuffle():
    # 生成回归数据集，样本数为 50，特征数为 5，目标数为 1，随机种子为 0
    # 使用循环测试是否洗牌对结果的影响，分别测试 shuffle=True 和 shuffle=False 两种情况
    for shuffle in [True, False]:
        # 创建一个MLPRegressor模型实例mlp1，设置参数并指定是否洗牌
        mlp1 = MLPRegressor(
            hidden_layer_sizes=1,
            max_iter=1,
            batch_size=1,
            random_state=0,
            shuffle=shuffle,
        )
        # 创建另一个MLPRegressor模型实例mlp2，设置相同参数但洗牌设置相反
        mlp2 = MLPRegressor(
            hidden_layer_sizes=1,
            max_iter=1,
            batch_size=1,
            random_state=0,
            shuffle=shuffle,
        )
        # 使用数据集X和目标变量y拟合mlp1模型
        mlp1.fit(X, y)
        # 使用相同的数据集X和目标变量y拟合mlp2模型
        mlp2.fit(X, y)

        # 断言两个模型的第一个隐藏层的权重系数是否完全相同
        assert np.array_equal(mlp1.coefs_[0], mlp2.coefs_[0])

    # 测试 shuffle=True 和 shuffle=False 时，模型的第一个隐藏层的权重系数是否不完全相同
    mlp1 = MLPRegressor(
        hidden_layer_sizes=1, max_iter=1, batch_size=1, random_state=0, shuffle=True
    )
    mlp2 = MLPRegressor(
        hidden_layer_sizes=1, max_iter=1, batch_size=1, random_state=0, shuffle=False
    )
    # 使用数据集X和目标变量y拟合mlp1模型
    mlp1.fit(X, y)
    # 使用相同的数据集X和目标变量y拟合mlp2模型
    mlp2.fit(X, y)

    # 断言两个模型的第一个隐藏层的权重系数是否不完全相同
    assert not np.array_equal(mlp1.coefs_[0], mlp2.coefs_[0])
# 使用参数化测试，对不同的CSR容器进行稀疏矩阵测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_matrices(csr_container):
    # 测试稀疏矩阵和密集矩阵输入是否产生相同的结果
    X = X_digits_binary[:50]
    y = y_digits_binary[:50]
    # 将输入转换为稀疏矩阵
    X_sparse = csr_container(X)
    # 创建多层感知器分类器
    mlp = MLPClassifier(solver="lbfgs", hidden_layer_sizes=15, random_state=1)
    # 使用密集矩阵拟合模型
    mlp.fit(X, y)
    # 预测密集矩阵的结果
    pred1 = mlp.predict(X)
    # 使用稀疏矩阵拟合模型
    mlp.fit(X_sparse, y)
    # 预测稀疏矩阵的结果
    pred2 = mlp.predict(X_sparse)
    # 断言预测结果几乎相等
    assert_almost_equal(pred1, pred2)
    # 再次预测密集矩阵的结果
    pred1 = mlp.predict(X)
    # 再次预测稀疏矩阵的结果
    pred2 = mlp.predict(X_sparse)
    # 断言预测结果数组完全相等
    assert_array_equal(pred1, pred2)


def test_tolerance():
    # 测试容忍度
    # 容忍度应在求解器收敛时强制退出循环
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    # 创建MLP分类器，设置容忍度为0.5，最大迭代次数为3000，求解器为"sgd"
    clf = MLPClassifier(tol=0.5, max_iter=3000, solver="sgd")
    # 使用输入数据拟合模型
    clf.fit(X, y)
    # 断言最大迭代次数大于实际迭代次数
    assert clf.max_iter > clf.n_iter_


def test_verbose_sgd():
    # 测试详细输出
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    # 创建SGD求解器的MLP分类器，设置最大迭代次数为2，详细输出为10，隐藏层大小为2
    clf = MLPClassifier(solver="sgd", max_iter=2, verbose=10, hidden_layer_sizes=2)
    old_stdout = sys.stdout
    sys.stdout = output = StringIO()

    # 忽略收敛警告，使用输入数据拟合模型
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
    # 部分拟合输入数据
    clf.partial_fit(X, y)

    sys.stdout = old_stdout
    # 断言输出中包含"Iteration"
    assert "Iteration" in output.getvalue()


@pytest.mark.parametrize("MLPEstimator", [MLPClassifier, MLPRegressor])
def test_early_stopping(MLPEstimator):
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]
    tol = 0.2
    # 创建MLP估计器，设置容忍度、最大迭代次数、求解器为"sgd"，启用早期停止
    mlp_estimator = MLPEstimator(
        tol=tol, max_iter=3000, solver="sgd", early_stopping=True
    )
    # 使用输入数据拟合模型
    mlp_estimator.fit(X, y)
    # 断言最大迭代次数大于实际迭代次数
    assert mlp_estimator.max_iter > mlp_estimator.n_iter_

    # 断言最佳损失值为None
    assert mlp_estimator.best_loss_ is None
    # 断言验证分数列表的类型为list
    assert isinstance(mlp_estimator.validation_scores_, list)

    valid_scores = mlp_estimator.validation_scores_
    best_valid_score = mlp_estimator.best_validation_score_
    # 断言最大验证分数等于最佳验证分数
    assert max(valid_scores) == best_valid_score
    # 断言最佳验证分数加上容忍度大于倒数第二个验证分数
    assert best_valid_score + tol > valid_scores[-2]
    # 断言最佳验证分数加上容忍度大于最后一个验证分数
    assert best_valid_score + tol > valid_scores[-1]

    # 当早期停止为False时，检查属性`validation_scores_`和`best_validation_score_`是否为None
    mlp_estimator = MLPEstimator(
        tol=tol, max_iter=3000, solver="sgd", early_stopping=False
    )
    # 使用输入数据拟合模型
    mlp_estimator.fit(X, y)
    # 断言验证分数列表为None
    assert mlp_estimator.validation_scores_ is None
    # 断言最佳验证分数为None
    assert mlp_estimator.best_validation_score_ is None
    # 断言最佳损失值不为None
    assert mlp_estimator.best_loss_ is not None


def test_adaptive_learning_rate():
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    # 创建MLP分类器，设置容忍度为0.5，最大迭代次数为3000，求解器为"sgd"，学习率为自适应
    clf = MLPClassifier(tol=0.5, max_iter=3000, solver="sgd", learning_rate="adaptive")
    # 使用输入数据拟合模型
    clf.fit(X, y)
    # 断言最大迭代次数大于实际迭代次数
    assert clf.max_iter > clf.n_iter_
    # 断言优化器学习率小于1e-6


@ignore_warnings(category=RuntimeWarning)
def test_warm_start():
    X = X_iris
    y = y_iris

    y_2classes = np.array([0] * 75 + [1] * 75)
    y_3classes = np.array([0] * 40 + [1] * 40 + [2] * 70)
    y_3classes_alt = np.array([0] * 50 + [1] * 50 + [3] * 50)
    # 创建一个包含 4 个类的标签数组，每个类分别包含不同数量的元素
    y_4classes = np.array([0] * 37 + [1] * 37 + [2] * 38 + [3] * 38)
    # 创建一个包含 5 个类的标签数组，每个类分别包含相同数量的元素
    y_5classes = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 30 + [4] * 30)

    # 创建一个多层感知器分类器对象，使用 lbfgs 求解器，设置隐藏层大小为 2，启用 warm_start 参数
    clf = MLPClassifier(hidden_layer_sizes=2, solver="lbfgs", warm_start=True).fit(X, y)
    # 使用训练数据 X 和标签 y 训练分类器
    clf.fit(X, y)
    # 使用训练数据 X 和具有 3 类的标签 y_3classes 继续训练分类器
    clf.fit(X, y_3classes)

    # 对于标签数组 y_i 中的每一个 y_i
    for y_i in (y_2classes, y_3classes_alt, y_4classes, y_5classes):
        # 创建一个多层感知器分类器对象，使用 lbfgs 求解器，设置隐藏层大小为 2，启用 warm_start 参数
        clf = MLPClassifier(hidden_layer_sizes=2, solver="lbfgs", warm_start=True).fit(
            X, y
        )
        # 准备错误消息，指示 warm_start 只能用于具有与上一次 fit 调用中相同类别的标签 y
        message = (
            "warm_start can only be used where `y` has the same "
            "classes as in the previous call to fit."
            " Previously got [0 1 2], `y` has %s" % np.unique(y_i)
        )
        # 使用 pytest 检测是否会抛出 ValueError，并验证错误消息是否包含预期的内容
        with pytest.raises(ValueError, match=re.escape(message)):
            # 继续使用标签数组 y_i 训练分类器，期望抛出特定的 ValueError
            clf.fit(X, y_i)
@pytest.mark.parametrize("MLPEstimator", [MLPClassifier, MLPRegressor])
# 使用 pytest.mark.parametrize 装饰器定义一个参数化测试，测试的参数是 MLPClassifier 和 MLPRegressor 这两个类
def test_warm_start_full_iteration(MLPEstimator):
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/16812
    # 检查 MLP 估计器在使用热启动时是否能完成 `max_iter`
    X, y = X_iris, y_iris
    max_iter = 3
    # 创建一个 MLP 估计器对象 clf，设置隐藏层大小为 2，使用 SGD 求解器，启用热启动，最大迭代次数为 max_iter
    clf = MLPEstimator(
        hidden_layer_sizes=2, solver="sgd", warm_start=True, max_iter=max_iter
    )
    # 使用数据 X, y 进行拟合
    clf.fit(X, y)
    # 断言检查最大迭代次数是否等于 clf 实际迭代次数
    assert max_iter == clf.n_iter_
    # 再次使用相同的数据进行拟合
    clf.fit(X, y)
    # 断言检查最大迭代次数是否等于 clf 实际迭代次数
    assert max_iter == clf.n_iter_


def test_n_iter_no_change():
    # test n_iter_no_change using binary data set
    # the classifying fitting process is not prone to loss curve fluctuations
    # 使用二进制数据集 X_digits_binary 和 y_digits_binary 的前 100 条数据
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]
    tol = 0.01
    max_iter = 3000

    # test multiple n_iter_no_change
    # 测试多个 n_iter_no_change 值
    for n_iter_no_change in [2, 5, 10, 50, 100]:
        # 创建一个 MLPClassifier 对象 clf，设置容忍度为 tol，最大迭代次数为 max_iter，
        # 使用 SGD 求解器，并指定 n_iter_no_change 参数
        clf = MLPClassifier(
            tol=tol, max_iter=max_iter, solver="sgd", n_iter_no_change=n_iter_no_change
        )
        # 使用数据 X, y 进行拟合
        clf.fit(X, y)

        # validate n_iter_no_change
        # 验证 _no_improvement_count 属性是否等于 n_iter_no_change + 1
        assert clf._no_improvement_count == n_iter_no_change + 1
        # 断言检查实际迭代次数是否小于最大迭代次数
        assert max_iter > clf.n_iter_


@ignore_warnings(category=ConvergenceWarning)
# 使用 ignore_warnings 装饰器忽略收敛警告类 ConvergenceWarning
def test_n_iter_no_change_inf():
    # test n_iter_no_change using binary data set
    # the fitting process should go to max_iter iterations
    # 使用二进制数据集 X_digits_binary 和 y_digits_binary 的前 100 条数据
    X = X_digits_binary[:100]
    y = y_digits_binary[:100]

    # set a ridiculous tolerance
    # this should always trigger _update_no_improvement_count()
    # 设置一个极大的容忍度，应该总是触发 _update_no_improvement_count() 方法
    tol = 1e9

    # fit
    # 将 n_iter_no_change 设置为正无穷大
    n_iter_no_change = np.inf
    max_iter = 3000
    # 创建一个 MLPClassifier 对象 clf，设置容忍度为 tol，最大迭代次数为 max_iter，
    # 使用 SGD 求解器，并指定 n_iter_no_change 参数
    clf = MLPClassifier(
        tol=tol, max_iter=max_iter, solver="sgd", n_iter_no_change=n_iter_no_change
    )
    # 使用数据 X, y 进行拟合
    clf.fit(X, y)

    # validate n_iter_no_change doesn't cause early stopping
    # 断言检查实际迭代次数是否等于最大迭代次数
    assert clf.n_iter_ == max_iter

    # validate _update_no_improvement_count() was always triggered
    # 断言检查 _no_improvement_count 属性是否等于 clf.n_iter_ - 1
    assert clf._no_improvement_count == clf.n_iter_ - 1


def test_early_stopping_stratified():
    # Make sure data splitting for early stopping is stratified
    # 确保早停止的数据分割是分层的
    X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y = [0, 0, 0, 1]

    # 创建一个 MLPClassifier 对象 mlp，启用早停止
    mlp = MLPClassifier(early_stopping=True)
    # 使用 pytest.raises 断言捕获 ValueError 异常，异常信息匹配特定文本
    with pytest.raises(
        ValueError, match="The least populated class in y has only 1 member"
    ):
        # 使用数据 X, y 进行拟合
        mlp.fit(X, y)


def test_mlp_classifier_dtypes_casting():
    # Compare predictions for different dtypes
    # 创建一个 MLPClassifier 对象 mlp_64，设置 alpha 值、隐藏层大小、随机种子和最大迭代次数
    mlp_64 = MLPClassifier(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    # 使用数据 X_digits 的前 300 条和 y_digits 的前 300 条进行拟合
    mlp_64.fit(X_digits[:300], y_digits[:300])
    # 进行预测
    pred_64 = mlp_64.predict(X_digits[300:])
    proba_64 = mlp_64.predict_proba(X_digits[300:])

    # 创建一个 MLPClassifier 对象 mlp_32，设置 alpha 值、隐藏层大小、随机种子和最大迭代次数
    mlp_32 = MLPClassifier(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    # 使用数据 X_digits 的前 300 条转换为 np.float32 类型和 y_digits 的前 300 条进行拟合
    mlp_32.fit(X_digits[:300].astype(np.float32), y_digits[:300])
    # 进行预测
    pred_32 = mlp_32.predict(X_digits[300:].astype(np.float32))
    proba_32 = mlp_32.predict_proba(X_digits[300:].astype(np.float32))
    # 检查预测结果数组 `pred_64` 和 `pred_32` 是否完全相等
    assert_array_equal(pred_64, pred_32)
    
    # 检查概率预测数组 `proba_64` 和 `proba_32` 是否在相对误差容差 `1e-02` 内全部接近
    assert_allclose(proba_64, proba_32, rtol=1e-02)
def test_mlp_regressor_dtypes_casting():
    # 创建一个包含64位精度的MLP回归器对象
    mlp_64 = MLPRegressor(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    # 使用前300个样本拟合MLP模型
    mlp_64.fit(X_digits[:300], y_digits[:300])
    # 对剩余的样本进行预测
    pred_64 = mlp_64.predict(X_digits[300:])

    # 创建一个包含32位精度的MLP回归器对象
    mlp_32 = MLPRegressor(
        alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50
    )
    # 将前300个样本转换为32位精度并拟合MLP模型
    mlp_32.fit(X_digits[:300].astype(np.float32), y_digits[:300])
    # 对剩余的样本进行32位精度预测
    pred_32 = mlp_32.predict(X_digits[300:].astype(np.float32))

    # 断言两个精度下的预测结果接近
    assert_allclose(pred_64, pred_32, rtol=1e-04)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("Estimator", [MLPClassifier, MLPRegressor])
def test_mlp_param_dtypes(dtype, Estimator):
    # 检查网络参数和预测结果是否使用了输入的数据类型
    X, y = X_digits.astype(dtype), y_digits
    # 创建指定数据类型的MLP分类器或回归器对象
    mlp = Estimator(alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=50)
    # 使用前300个样本拟合MLP模型
    mlp.fit(X[:300], y[:300])
    # 对剩余的样本进行预测
    pred = mlp.predict(X[300:])

    # 断言所有隐含层的截距参数使用了指定的数据类型
    assert all([intercept.dtype == dtype for intercept in mlp.intercepts_])

    # 断言所有权重参数使用了指定的数据类型
    assert all([coef.dtype == dtype for coef in mlp.coefs_])

    if Estimator == MLPRegressor:
        # 如果是回归器，断言预测结果使用了指定的数据类型
        assert pred.dtype == dtype


def test_mlp_loading_from_joblib_partial_fit(tmp_path):
    """从MLP加载并使用partial_fit更新权重。非回归测试#19626."""
    # 创建一个预先训练好的MLP回归器对象
    pre_trained_estimator = MLPRegressor(
        hidden_layer_sizes=(42,), random_state=42, learning_rate_init=0.01, max_iter=200
    )
    features, target = [[2]], [4]

    # 在x=2，y=4上进行拟合
    pre_trained_estimator.fit(features, target)

    # 存储和加载模型
    pickled_file = tmp_path / "mlp.pkl"
    joblib.dump(pre_trained_estimator, pickled_file)
    load_estimator = joblib.load(pickled_file)

    # 对点x=2，y=1进行更多次数的训练
    fine_tune_features, fine_tune_target = [[2]], [1]

    for _ in range(200):
        load_estimator.partial_fit(fine_tune_features, fine_tune_target)

    # 微调后的模型学习了新的目标值
    predicted_value = load_estimator.predict(fine_tune_features)
    assert_allclose(predicted_value, fine_tune_target, rtol=1e-4)


@pytest.mark.parametrize("Estimator", [MLPClassifier, MLPRegressor])
def test_preserve_feature_names(Estimator):
    """检查启用早停时是否保留特征名称。

    在评分期间，需要特征名称进行一致性检查。

    非回归测试gh-24846
    """
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)

    # 创建包含两列的DataFrame作为特征和一个列的Series作为目标
    X = pd.DataFrame(data=rng.randn(10, 2), columns=["colname_a", "colname_b"])
    y = pd.Series(data=np.full(10, 1), name="colname_y")

    model = Estimator(early_stopping=True, validation_fraction=0.2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # 使用Pandas数据拟合模型
        model.fit(X, y)


@pytest.mark.parametrize("MLPEstimator", [MLPClassifier, MLPRegressor])
def test_mlp_warm_start_with_early_stopping(MLPEstimator):
    """Check that early stopping works with warm start."""
    # 创建一个多层感知器估计器对象，设置最大迭代次数为10，随机种子为0，启用热启动和早停机制
    mlp = MLPEstimator(
        max_iter=10, random_state=0, warm_start=True, early_stopping=True
    )
    # 使用鸢尾花数据集进行训练
    mlp.fit(X_iris, y_iris)
    # 获取训练过程中的验证分数列表的长度
    n_validation_scores = len(mlp.validation_scores_)
    # 修改最大迭代次数为20，但保持现有的模型参数不变（热启动）
    mlp.set_params(max_iter=20)
    # 再次使用相同数据集进行训练
    mlp.fit(X_iris, y_iris)
    # 断言：验证分数列表的长度应该大于之前的长度，以确认早停机制起作用
    assert len(mlp.validation_scores_) > n_validation_scores
# 参数化测试，对于不同的MLP估计器（分类器或回归器），以及不同的求解器进行测试
@pytest.mark.parametrize("MLPEstimator", [MLPClassifier, MLPRegressor])
@pytest.mark.parametrize("solver", ["sgd", "adam", "lbfgs"])
def test_mlp_warm_start_no_convergence(MLPEstimator, solver):
    """检查在启用热启动时，当达到`max_iter`时停止迭代。

    非回归测试，针对：
    https://github.com/scikit-learn/scikit-learn/issues/24764
    """
    # 创建MLP估计器对象，设置参数
    model = MLPEstimator(
        solver=solver,
        warm_start=True,
        early_stopping=False,
        max_iter=10,
        n_iter_no_change=np.inf,
        random_state=0,
    )

    # 检测是否发出收敛警告
    with pytest.warns(ConvergenceWarning):
        model.fit(X_iris, y_iris)
    # 断言模型的迭代次数是否为10
    assert model.n_iter_ == 10

    # 修改参数，增加迭代次数为20
    model.set_params(max_iter=20)
    # 再次检测是否发出收敛警告
    with pytest.warns(ConvergenceWarning):
        model.fit(X_iris, y_iris)
    # 断言模型的迭代次数是否为20
    assert model.n_iter_ == 20


# 参数化测试，对于不同的MLP估计器（分类器或回归器）进行测试
@pytest.mark.parametrize("MLPEstimator", [MLPClassifier, MLPRegressor])
def test_mlp_partial_fit_after_fit(MLPEstimator):
    """检查在设置early_stopping=True后，partial_fit在fit后不会失败。

    非回归测试，针对gh-25693。
    """
    # 创建MLP估计器对象，设置参数，执行fit操作
    mlp = MLPEstimator(early_stopping=True, random_state=0).fit(X_iris, y_iris)

    # 出现错误信息的断言，验证partial_fit在early_stopping=True时是否不支持
    msg = "partial_fit does not support early_stopping=True"
    with pytest.raises(ValueError, match=msg):
        mlp.partial_fit(X_iris, y_iris)
```