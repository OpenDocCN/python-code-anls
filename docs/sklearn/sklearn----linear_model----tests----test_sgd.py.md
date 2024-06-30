# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_sgd.py`

```
# 导入所需的模块和类
import pickle  # 用于序列化和反序列化 Python 对象
from unittest.mock import Mock  # 用于创建模拟对象

import joblib  # 用于保存和加载模型
import numpy as np  # 用于数值计算的核心库
import pytest  # 用于编写和运行测试的框架
import scipy.sparse as sp  # 用于稀疏矩阵的科学计算库

from sklearn import datasets, linear_model, metrics  # 加载数据集和机器学习模型、评估指标
from sklearn.base import clone, is_classifier  # 用于克隆和判断是否是分类器的基类
from sklearn.exceptions import ConvergenceWarning  # 用于收敛警告的异常类
from sklearn.kernel_approximation import Nystroem  # 用于核近似的类
from sklearn.linear_model import _sgd_fast as sgd_fast  # 私有的快速梯度下降优化器
from sklearn.linear_model import _stochastic_gradient  # 随机梯度下降的私有实现
from sklearn.model_selection import (  # 用于模型选择和评估的类和函数
    RandomizedSearchCV,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import make_pipeline  # 用于创建管道的函数
from sklearn.preprocessing import (  # 用于数据预处理的函数
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    scale,
)
from sklearn.svm import OneClassSVM  # 用于支持向量机的类
from sklearn.utils._testing import (  # 用于测试的辅助函数
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    ignore_warnings,
)


def _update_kwargs(kwargs):
    # 如果参数字典中没有 "random_state" 键，设置其默认值为 42
    if "random_state" not in kwargs:
        kwargs["random_state"] = 42

    # 如果参数字典中没有 "tol" 键，设置其默认值为 None
    if "tol" not in kwargs:
        kwargs["tol"] = None
    # 如果参数字典中没有 "max_iter" 键，设置其默认值为 5
    if "max_iter" not in kwargs:
        kwargs["max_iter"] = 5


class _SparseSGDClassifier(linear_model.SGDClassifier):
    def fit(self, X, y, *args, **kw):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return super().fit(X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return super().partial_fit(X, y, *args, **kw)

    def decision_function(self, X):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return super().decision_function(X)

    def predict_proba(self, X):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return super().predict_proba(X)


class _SparseSGDRegressor(linear_model.SGDRegressor):
    def fit(self, X, y, *args, **kw):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.fit(self, X, y, *args, **kw)

    def partial_fit(self, X, y, *args, **kw):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.partial_fit(self, X, y, *args, **kw)

    def decision_function(self, X, *args, **kw):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return linear_model.SGDRegressor.decision_function(self, X, *args, **kw)


class _SparseSGDOneClassSVM(linear_model.SGDOneClassSVM):
    def fit(self, X, *args, **kw):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.fit(self, X, *args, **kw)

    def partial_fit(self, X, *args, **kw):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.partial_fit(self, X, *args, **kw)

    def decision_function(self, X, *args, **kw):
        # 将输入数据 X 转换为稀疏矩阵格式
        X = sp.csr_matrix(X)
        return linear_model.SGDOneClassSVM.decision_function(self, X, *args, **kw)


def SGDClassifier(**kwargs):
    # 更新参数 kwargs，确保包含必要的默认值
    _update_kwargs(kwargs)
    return linear_model.SGDClassifier(**kwargs)


def SGDRegressor(**kwargs):
    # 更新参数 kwargs，确保包含必要的默认值
    _update_kwargs(kwargs)
    return linear_model.SGDRegressor(**kwargs)


def SGDOneClassSVM(**kwargs):
    # 更新参数 kwargs，确保包含必要的默认值
    _update_kwargs(kwargs)
    return linear_model.SGDOneClassSVM(**kwargs)
# 调用 _update_kwargs 函数，更新传入参数 kwargs
def SparseSGDClassifier(**kwargs):
    _update_kwargs(kwargs)
    # 返回 _SparseSGDClassifier 类的实例，使用传入的关键字参数 kwargs
    return _SparseSGDClassifier(**kwargs)


# 调用 _update_kwargs 函数，更新传入参数 kwargs
def SparseSGDRegressor(**kwargs):
    _update_kwargs(kwargs)
    # 返回 _SparseSGDRegressor 类的实例，使用传入的关键字参数 kwargs
    return _SparseSGDRegressor(**kwargs)


# 调用 _update_kwargs 函数，更新传入参数 kwargs
def SparseSGDOneClassSVM(**kwargs):
    _update_kwargs(kwargs)
    # 返回 _SparseSGDOneClassSVM 类的实例，使用传入的关键字参数 kwargs
    return _SparseSGDOneClassSVM(**kwargs)


# Test Data

# 测试样本 1
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y = [1, 1, 1, 2, 2, 2]
T = np.array([[-1, -1], [2, 2], [3, 2]])
true_result = [1, 2, 2]

# 测试样本 2；使用字符串作为类标签
X2 = np.array(
    [
        [-1, 1],
        [-0.75, 0.5],
        [-1.5, 1.5],
        [1, 1],
        [0.75, 0.5],
        [1.5, 1.5],
        [-1, -1],
        [0, -0.5],
        [1, -1],
    ]
)
Y2 = ["one"] * 3 + ["two"] * 3 + ["three"] * 3
T2 = np.array([[-1.5, 0.5], [1, 2], [0, -2]])
true_result2 = ["one", "two", "three"]

# 测试样本 3
X3 = np.array(
    [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)
Y3 = np.array([1, 1, 1, 1, 2, 2, 2, 2])

# 测试样本 4 - 两个相对冗余的特征组
X4 = np.array(
    [
        [1, 0.9, 0.8, 0, 0, 0],
        [1, 0.84, 0.98, 0, 0, 0],
        [1, 0.96, 0.88, 0, 0, 0],
        [1, 0.91, 0.99, 0, 0, 0],
        [0, 0, 0, 0.89, 0.91, 1],
        [0, 0, 0, 0.79, 0.84, 1],
        [0, 0, 0, 0.91, 0.95, 1],
        [0, 0, 0, 0.93, 1, 1],
    ]
)
Y4 = np.array([1, 1, 1, 1, 2, 2, 2, 2])

# 使用 datasets 模块加载 iris 数据集
iris = datasets.load_iris()

# 测试样本 5 - 将测试样本 1 作为二元分类问题
X5 = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y5 = [1, 1, 1, 2, 2, 2]
true_result5 = [0, 1, 1]


###############################################################################
# Common Test Case to classification and regression


# 简单的 ASGD 实现，用于测试
# 使用平方损失来计算梯度
def asgd(klass, X, y, eta, alpha, weight_init=None, intercept_init=0.0):
    # 如果未提供 weight_init，则将权重初始化为全零数组
    if weight_init is None:
        weights = np.zeros(X.shape[1])
    else:
        weights = weight_init

    # 初始化平均权重和截距
    average_weights = np.zeros(X.shape[1])
    intercept = intercept_init
    average_intercept = 0.0
    decay = 1.0

    # 对于 SparseSGDClassifier 和 SparseSGDRegressor，设置固定的衰减率为 0.01
    if klass in (SparseSGDClassifier, SparseSGDRegressor):
        decay = 0.01

    # 遍历输入数据 X 中的每一个条目
    for i, entry in enumerate(X):
        # 计算预测值 p
        p = np.dot(entry, weights)
        p += intercept
        # 计算梯度
        gradient = p - y[i]
        # 更新权重
        weights *= 1.0 - (eta * alpha)
        weights += -(eta * gradient * entry)
        # 更新截距，应用衰减
        intercept += -(eta * gradient) * decay

        # 更新平均权重
        average_weights *= i
        average_weights += weights
        average_weights /= i + 1.0

        # 更新平均截距
        average_intercept *= i
        average_intercept += intercept
        average_intercept /= i + 1.0

    # 返回平均权重和平均截距
    return average_weights, average_intercept


# 内部函数，用于测试热启动
def _test_warm_start(klass, X, Y, lr):
    # 创建一个具有指定参数的分类器对象 clf，用于训练数据 X 和标签 Y
    clf = klass(alpha=0.01, eta0=0.01, shuffle=False, learning_rate=lr)
    # 使用训练数据 X 和标签 Y 对分类器进行训练
    clf.fit(X, Y)

    # 创建另一个具有不同 alpha 参数的分类器对象 clf2，但使用了与 clf 相同的初始化参数
    clf2 = klass(alpha=0.001, eta0=0.01, shuffle=False, learning_rate=lr)
    # 使用训练数据 X 和标签 Y 对 clf2 进行训练，并以 clf 的权重和截距作为初始值
    clf2.fit(X, Y, coef_init=clf.coef_.copy(), intercept_init=clf.intercept_.copy())

    # 创建具有 warm_start 参数的分类器对象 clf3，用于测试隐式热启动
    clf3 = klass(
        alpha=0.01, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr
    )
    # 使用训练数据 X 和标签 Y 对 clf3 进行训练
    clf3.fit(X, Y)

    # 断言 clf3 和 clf 在同一时刻迭代的次数 t_
    assert clf3.t_ == clf.t_
    # 断言 clf3 的权重 coef_ 与 clf 的权重 coef_ 几乎相等
    assert_array_almost_equal(clf3.coef_, clf.coef_)

    # 修改 clf3 的 alpha 参数为 0.001
    clf3.set_params(alpha=0.001)
    # 使用训练数据 X 和标签 Y 对 clf3 进行重新训练
    clf3.fit(X, Y)

    # 断言 clf3 和 clf2 在同一时刻迭代的次数 t_
    assert clf3.t_ == clf2.t_
    # 断言 clf3 的权重 coef_ 与 clf2 的权重 coef_ 几乎相等
    assert_array_almost_equal(clf3.coef_, clf2.coef_)
@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_warm_start(klass, lr):
    _test_warm_start(klass, X, Y, lr)

第一个测试函数：参数化测试热启动功能。对于给定的分类器类（klass）和学习率类型（lr），调用内部函数 `_test_warm_start` 进行测试。


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_input_format(klass):
    # 输入格式测试。
    clf = klass(alpha=0.01, shuffle=False)
    # 使用给定的参数初始化分类器
    clf.fit(X, Y)
    Y_ = np.array(Y)[:, np.newaxis]
    # 将 Y 转换为二维数组
    Y_ = np.c_[Y_, Y_]
    # 尝试使用错误的格式重新拟合分类器，应该触发 ValueError
    with pytest.raises(ValueError):
        clf.fit(X, Y_)

第二个测试函数：测试输入数据的格式。使用给定的分类器类（klass）和固定的 alpha 值初始化分类器，对其进行拟合操作，然后测试异常情况下的数据格式。


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_clone(klass):
    # 测试克隆是否正常工作。
    clf = klass(alpha=0.01, penalty="l1")
    # 使用克隆函数创建分类器的副本
    clf = clone(clf)
    clf.set_params(penalty="l2")
    clf.fit(X, Y)

    clf2 = klass(alpha=0.01, penalty="l2")
    clf2.fit(X, Y)

    assert_array_equal(clf.coef_, clf2.coef_)

第三个测试函数：测试分类器对象的克隆功能。使用给定的分类器类（klass）和特定的参数初始化分类器，然后通过克隆函数 `clone` 创建分类器的副本，验证其参数设置和拟合后的结果是否一致。


@pytest.mark.parametrize(
    "klass",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_plain_has_no_average_attr(klass):
    clf = klass(average=True, eta0=0.01)
    clf.fit(X, Y)

    assert hasattr(clf, "_average_coef")
    assert hasattr(clf, "_average_intercept")
    assert hasattr(clf, "_standard_intercept")
    assert hasattr(clf, "_standard_coef")

    clf = klass()
    clf.fit(X, Y)

    assert not hasattr(clf, "_average_coef")
    assert not hasattr(clf, "_average_intercept")
    assert not hasattr(clf, "_standard_intercept")
    assert not hasattr(clf, "_standard_coef")

第四个测试函数：测试是否具有平均属性。对于给定的分类器类（klass），分别测试在设置了平均参数和未设置平均参数的情况下，分类器是否具有特定的属性。


@pytest.mark.parametrize(
    "klass",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_late_onset_averaging_not_reached(klass):
    clf1 = klass(average=600)
    clf2 = klass()
    for _ in range(100):
        if is_classifier(clf1):
            clf1.partial_fit(X, Y, classes=np.unique(Y))
            clf2.partial_fit(X, Y, classes=np.unique(Y))
        else:
            clf1.partial_fit(X, Y)
            clf2.partial_fit(X, Y)

    assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=16)
    if klass in [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]:
        assert_almost_equal(clf1.intercept_, clf2.intercept_, decimal=16)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        assert_allclose(clf1.offset_, clf2.offset_)

第五个测试函数：测试延迟启动平均未达到的情况。对于给定的分类器类（klass），创建两个分类器实例并使用部分拟合方法多次拟合数据，最后验证其权重（coef_）是否接近。


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_late_onset_averaging_reached(klass):
    eta0 = 0.001
    alpha = 0.0001
    Y_encode = np.array(Y)
    Y_encode[Y_encode == 1] = -1.0
    Y_encode[Y_encode == 2] = 1.0

最后一个测试函数：测试延迟启动平均已达到的情况。对于给定的分类器类（klass）和特定的参数，创建一个已编码的目标数组 Y_encode，用于后续的测试。
    # 使用给定的类（klass）初始化第一个分类器（clf1）
    clf1 = klass(
        average=7,
        learning_rate="constant",
        loss="squared_error",
        eta0=eta0,
        alpha=alpha,
        max_iter=2,
        shuffle=False,
    )

    # 使用给定的类（klass）初始化第二个分类器（clf2）
    clf2 = klass(
        average=False,
        learning_rate="constant",
        loss="squared_error",
        eta0=eta0,
        alpha=alpha,
        max_iter=1,
        shuffle=False,
    )

    # 使用训练数据 X 和编码后的标签数据 Y_encode 对 clf1 进行训练
    clf1.fit(X, Y_encode)
    
    # 使用训练数据 X 和编码后的标签数据 Y_encode 对 clf2 进行训练
    clf2.fit(X, Y_encode)

    # 使用平均随机梯度下降算法（asgd）计算平均权重和平均截距
    average_weights, average_intercept = asgd(
        klass,
        X,
        Y_encode,
        eta0,
        alpha,
        # 初始权重使用 clf2 的系数展平后的值
        weight_init=clf2.coef_.ravel(),
        # 初始截距使用 clf2 的截距值
        intercept_init=clf2.intercept_,
    )

    # 断言：比较 clf1 训练后的权重和计算得到的平均权重，精确到小数点后 16 位
    assert_array_almost_equal(clf1.coef_.ravel(), average_weights.ravel(), decimal=16)
    
    # 断言：比较 clf1 训练后的截距和计算得到的平均截距，精确到小数点后 16 位
    assert_almost_equal(clf1.intercept_, average_intercept, decimal=16)
# 使用 pytest.mark.parametrize 装饰器来为 test_early_stopping 函数参数化测试用例，参数 klass 分别为 SGDClassifier、SparseSGDClassifier、SGDRegressor、SparseSGDRegressor 类
@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_early_stopping(klass):
    # 从鸢尾花数据集 iris 中选择数据和目标，其中目标值大于 0 的样本
    X = iris.data[iris.target > 0]
    Y = iris.target[iris.target > 0]
    # 对 early_stopping 参数分别设置为 True 和 False 进行测试
    for early_stopping in [True, False]:
        max_iter = 1000
        # 使用给定的 klass 创建分类器或回归器对象，设置 early_stopping、tol 和 max_iter 参数，使用 X, Y 数据进行拟合
        clf = klass(early_stopping=early_stopping, tol=1e-3, max_iter=max_iter).fit(
            X, Y
        )
        # 断言实际迭代次数 clf.n_iter_ 小于设定的最大迭代次数 max_iter
        assert clf.n_iter_ < max_iter


# 类似地，对 test_adaptive_longer_than_constant 函数进行参数化测试
@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_adaptive_longer_than_constant(klass):
    # 创建 learning_rate 为 adaptive 的分类器或回归器对象 clf1，使用 iris 数据集进行拟合
    clf1 = klass(learning_rate="adaptive", eta0=0.01, tol=1e-3, max_iter=100)
    clf1.fit(iris.data, iris.target)
    # 创建 learning_rate 为 constant 的分类器或回归器对象 clf2，使用 iris 数据集进行拟合
    clf2 = klass(learning_rate="constant", eta0=0.01, tol=1e-3, max_iter=100)
    clf2.fit(iris.data, iris.target)
    # 断言 clf1 的迭代次数 clf1.n_iter_ 大于 clf2 的迭代次数 clf2.n_iter_
    assert clf1.n_iter_ > clf2.n_iter_


# 继续对 test_validation_set_not_used_for_training 函数进行参数化测试
@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_validation_set_not_used_for_training(klass):
    X, Y = iris.data, iris.target
    validation_fraction = 0.4
    seed = 42
    shuffle = False
    max_iter = 10
    # 创建具有 early_stopping 和 validation_fraction 参数的分类器或回归器对象 clf1
    clf1 = klass(
        early_stopping=True,
        random_state=np.random.RandomState(seed),
        validation_fraction=validation_fraction,
        learning_rate="constant",
        eta0=0.01,
        tol=None,
        max_iter=max_iter,
        shuffle=shuffle,
    )
    clf1.fit(X, Y)
    # 断言 clf1 的迭代次数 clf1.n_iter_ 等于设定的最大迭代次数 max_iter
    assert clf1.n_iter_ == max_iter

    # 创建不具有 early_stopping 参数的分类器或回归器对象 clf2
    clf2 = klass(
        early_stopping=False,
        random_state=np.random.RandomState(seed),
        learning_rate="constant",
        eta0=0.01,
        tol=None,
        max_iter=max_iter,
        shuffle=shuffle,
    )

    # 根据分类器类型创建交叉验证对象 cv，用于分割数据集
    if is_classifier(clf2):
        cv = StratifiedShuffleSplit(test_size=validation_fraction, random_state=seed)
    else:
        cv = ShuffleSplit(test_size=validation_fraction, random_state=seed)
    # 获取训练集和验证集的索引
    idx_train, idx_val = next(cv.split(X, Y))
    idx_train = np.sort(idx_train)  # 移除洗牌操作
    # 使用训练集数据拟合 clf2 分类器或回归器
    clf2.fit(X[idx_train], Y[idx_train])
    # 断言 clf2 的迭代次数 clf2.n_iter_ 等于设定的最大迭代次数 max_iter
    assert clf2.n_iter_ == max_iter

    # 断言 clf1 和 clf2 的权重 coef_ 数组相等
    assert_array_equal(clf1.coef_, clf2.coef_)


# 最后，对 test_n_iter_no_change 函数进行参数化测试
@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_n_iter_no_change(klass):
    X, Y = iris.data, iris.target
    # 测试当 early_stopping 参数为 True 和 False 时，n_iter_ 随 n_iter_no_change 的增加而单调增加
    for early_stopping in [True, False]:
        # 创建包含不同 n_iter_no_change 参数的分类器或回归器对象，使用 X, Y 数据进行拟合，并获取其 n_iter_ 属性
        n_iter_list = [
            klass(
                early_stopping=early_stopping,
                n_iter_no_change=n_iter_no_change,
                tol=1e-4,
                max_iter=1000,
            )
            .fit(X, Y)
            .n_iter_
            for n_iter_no_change in [2, 3, 10]
        ]
        # 断言 n_iter_list 中的迭代次数列表为非降序排序
        assert_array_equal(n_iter_list, sorted(n_iter_list))
    # 创建一个分类器对象 clf，设置 early_stopping 为 True，并且指定 validation_fraction 为 0.99
    clf = klass(early_stopping=True, validation_fraction=0.99)
    
    # 使用 pytest 框架验证预期会引发 ValueError 异常
    with pytest.raises(ValueError):
        # 调用分类器对象的 fit 方法，尝试用 X3 和 Y3 数据进行训练
        clf.fit(X3, Y3)
###############################################################################
# Classification Test Case

# 使用 pytest 提供的参数化功能，分别对 SGDClassifier 和 SparseSGDClassifier 进行测试
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_clf(klass):
    # 检查 SGD 在不同损失函数下的分类结果

    # 循环遍历不同的损失函数
    for loss in ("hinge", "squared_hinge", "log_loss", "modified_huber"):
        # 使用给定的参数创建分类器对象
        clf = klass(
            penalty="l2",
            alpha=0.01,
            fit_intercept=True,
            loss=loss,
            max_iter=10,
            shuffle=True,
        )
        # 对分类器进行拟合
        clf.fit(X, Y)
        # 断言预测结果与真实结果相等
        assert_array_equal(clf.predict(T), true_result)


# 使用 pytest 提供的参数化功能，对多个分类器进行测试
@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM]
)
def test_provide_coef(klass):
    """检查 `coef_init` 的形状是否被正确验证。"""
    # 使用 pytest 的断言，确保当 `coef_init` 与数据集不匹配时会抛出 ValueError 异常
    with pytest.raises(ValueError, match="Provided coef_init does not match dataset"):
        klass().fit(X, Y, coef_init=np.zeros((3,)))


# 使用 pytest 提供的参数化功能，对多个分类器及其参数进行测试
@pytest.mark.parametrize(
    "klass, fit_params",
    [
        (SGDClassifier, {"intercept_init": np.zeros((3,))}),
        (SparseSGDClassifier, {"intercept_init": np.zeros((3,))}),
        (SGDOneClassSVM, {"offset_init": np.zeros((3,))}),
        (SparseSGDOneClassSVM, {"offset_init": np.zeros((3,))}),
    ],
)
def test_set_intercept_offset(klass, fit_params):
    """检查 `intercept_init` 或 `offset_init` 是否被正确验证。"""
    # 创建分类器对象
    sgd_estimator = klass()
    # 使用 pytest 的断言，确保当 `intercept_init` 或 `offset_init` 与数据集不匹配时会抛出 ValueError 异常
    with pytest.raises(ValueError, match="does not match dataset"):
        sgd_estimator.fit(X, Y, **fit_params)


# 使用 pytest 提供的参数化功能，对多个分类器进行测试
@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDRegressor, SparseSGDRegressor]
)
def test_sgd_early_stopping_with_partial_fit(klass):
    """检查 `partial_fit` 方法与 `early_stopping` 参数结合使用时是否会抛出错误。"""
    # 错误消息
    err_msg = "early_stopping should be False with partial_fit"
    # 使用 pytest 的断言，确保当 `early_stopping=True` 时会抛出 ValueError 异常
    with pytest.raises(ValueError, match=err_msg):
        klass(early_stopping=True).partial_fit(X, Y)


# 使用 pytest 提供的参数化功能，对多个分类器及其参数进行测试
@pytest.mark.parametrize(
    "klass, fit_params",
    [
        (SGDClassifier, {"intercept_init": 0}),
        (SparseSGDClassifier, {"intercept_init": 0}),
        (SGDOneClassSVM, {"offset_init": 0}),
        (SparseSGDOneClassSVM, {"offset_init": 0}),
    ],
)
def test_set_intercept_offset_binary(klass, fit_params):
    """检查在二元分类问题中，能否将标量传递给 `intercept_init` 或 `offset_init`。"""
    # 对给定的分类器使用标量参数进行拟合
    klass().fit(X5, Y5, **fit_params)


# 使用 pytest 提供的参数化功能，对 SGDClassifier 和 SparseSGDClassifier 进行测试
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_average_binary_computed_correctly(klass):
    # 检查 SGDClassifier 是否正确计算了平均权重

    # 设置参数
    eta = 0.1
    alpha = 2.0
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    # 使用给定的分类器类别初始化分类器对象
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    # 计算简单线性函数的输出，不考虑噪声
    y = np.dot(X, w)
    y = np.sign(y)

    # 使用训练数据 X 和目标值 y 来拟合分类器
    clf.fit(X, y)

    # 调用自定义函数 asgd 计算平均权重和平均截距
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)

    # 将平均权重重塑为二维数组
    average_weights = average_weights.reshape(1, -1)

    # 断言分类器的权重与计算得到的平均权重接近（精度到小数点后14位）
    assert_array_almost_equal(clf.coef_, average_weights, decimal=14)

    # 断言分类器的截距与计算得到的平均截距接近（精度到小数点后14位）
    assert_almost_equal(clf.intercept_, average_intercept, decimal=14)
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_set_intercept_to_intercept(klass):
    # 检查 warm start 下的 intercept_ 形状一致性
    # 创建一个分类器实例并拟合数据集 X5, Y5
    clf = klass().fit(X5, Y5)
    # 使用拟合得到的 intercept_ 进行另一次拟合
    klass().fit(X5, Y5, intercept_init=clf.intercept_)
    # 创建另一个分类器实例并拟合数据集 X, Y
    clf = klass().fit(X, Y)
    # 使用拟合得到的 intercept_ 进行另一次拟合
    klass().fit(X, Y, intercept_init=clf.intercept_)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_at_least_two_labels(klass):
    # 目标变量至少包含两个标签
    # 创建一个分类器实例
    clf = klass(alpha=0.01, max_iter=20)
    # 断言在给定数据集 X2 和全为1的目标变量时会触发 ValueError 异常
    with pytest.raises(ValueError):
        clf.fit(X2, np.ones(9))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_partial_fit_weight_class_balanced(klass):
    # 不支持使用 class_weight='balanced' 进行 partial_fit
    # 定义用于匹配异常信息的正则表达式
    regex = (
        r"class_weight 'balanced' is not supported for "
        r"partial_fit\. In order to use 'balanced' weights, "
        r"use compute_class_weight\('balanced', classes=classes, y=y\). "
        r"In place of y you can use a large enough sample "
        r"of the full training set target to properly "
        r"estimate the class frequency distributions\. "
        r"Pass the resulting weights as the class_weight "
        r"parameter\."
    )
    # 断言在使用 class_weight='balanced' 时会触发 ValueError 异常，并匹配预期的异常信息
    with pytest.raises(ValueError, match=regex):
        klass(class_weight="balanced").partial_fit(X, Y, classes=np.unique(Y))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass(klass):
    # 多类别分类测试案例
    # 创建一个分类器实例并拟合数据集 X2, Y2
    clf = klass(alpha=0.01, max_iter=20).fit(X2, Y2)
    # 断言分类器的 coef_ 属性形状为 (3, 2)
    assert clf.coef_.shape == (3, 2)
    # 断言分类器的 intercept_ 属性形状为 (3,)
    assert clf.intercept_.shape == (3,)
    # 断言对单个样本 [[0, 0]] 的决策函数输出形状为 (1, 3)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    # 对测试集 T2 进行预测并断言预测结果与真实结果 true_result2 相等
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_average(klass):
    eta = 0.001
    alpha = 0.01
    # 多类别平均测试案例
    # 创建一个具有特定参数的分类器实例
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    np_Y2 = np.array(Y2)
    # 使用 partial_fit 进行拟合
    clf.fit(X2, np_Y2)
    # 计算目标变量的唯一类别
    classes = np.unique(np_Y2)

    for i, cl in enumerate(classes):
        # 根据类别构建目标变量
        y_i = np.ones(np_Y2.shape[0])
        y_i[np_Y2 != cl] = -1
        # 调用 asgd 函数计算平均系数和截距
        average_coef, average_intercept = asgd(klass, X2, y_i, eta, alpha)
        # 断言平均系数与分类器实例中的 coef_ 相等
        assert_array_almost_equal(average_coef, clf.coef_[i], decimal=16)
        # 断言平均截距与分类器实例中的 intercept_ 相等
        assert_almost_equal(average_intercept, clf.intercept_[i], decimal=16)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_with_init_coef(klass):
    # 多类别测试案例
    # 创建一个具有特定参数的分类器实例
    clf = klass(alpha=0.01, max_iter=20)
    # 使用给定的 coef_init 和 intercept_init 参数进行拟合
    clf.fit(X2, Y2, coef_init=np.zeros((3, 2)), intercept_init=np.zeros(3))
    # 断言分类器的 coef_ 属性形状为 (3, 2)
    assert clf.coef_.shape == (3, 2)
    # 断言分类器的 intercept_ 属性形状为 (3,)
    assert clf.intercept_.shape, (3,)
    # 使用分类器 clf 对输入数据 T2 进行预测
    pred = clf.predict(T2)
    # 断言预测结果 pred 与真实结果 true_result2 相等，若不相等则抛出异常
    assert_array_equal(pred, true_result2)
# 使用 pytest 的 parametrize 装饰器来为 SGDClassifier 和 SparseSGDClassifier 类型的测试函数创建参数化测试
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_multiclass_njobs(klass):
    # 多类别测试用例，支持多核处理
    clf = klass(alpha=0.01, max_iter=20, n_jobs=2).fit(X2, Y2)
    # 断言分类器的系数形状为 (3, 2)
    assert clf.coef_.shape == (3, 2)
    # 断言分类器的截距形状为 (3,)
    assert clf.intercept_.shape == (3,)
    # 断言分类器的决策函数预测形状为 (1, 3)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    # 预测分类结果并断言其与真实结果 true_result2 相等
    pred = clf.predict(T2)
    assert_array_equal(pred, true_result2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_set_coef_multiclass(klass):
    # 检查多类别问题中 coef_init 和 intercept_init 的形状
    # 当提供的 coef_ 不匹配数据集时，应引发 ValueError
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, coef_init=np.zeros((2, 2)))

    # 当提供的 coef_ 匹配数据集时
    clf = klass().fit(X2, Y2, coef_init=np.zeros((3, 2)))

    # 当提供的 intercept_ 不匹配数据集时，应引发 ValueError
    clf = klass()
    with pytest.raises(ValueError):
        clf.fit(X2, Y2, intercept_init=np.zeros((1,)))

    # 当提供的 intercept_ 匹配数据集时
    clf = klass().fit(X2, Y2, intercept_init=np.zeros((3,)))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_predict_proba_method_access(klass):
    # 检查 SGDClassifier 的 predict_proba 和 predict_log_proba 方法的访问性
    # 或者引发适当的错误消息
    # 对于不同的损失函数，验证是否具有这些属性或者是否正确引发 AttributeError
    for loss in linear_model.SGDClassifier.loss_functions:
        clf = SGDClassifier(loss=loss)
        if loss in ("log_loss", "modified_huber"):
            assert hasattr(clf, "predict_proba")
            assert hasattr(clf, "predict_log_proba")
        else:
            inner_msg = "probability estimates are not available for loss={!r}".format(
                loss
            )
            assert not hasattr(clf, "predict_proba")
            assert not hasattr(clf, "predict_log_proba")
            with pytest.raises(
                AttributeError, match="has no attribute 'predict_proba'"
            ) as exec_info:
                clf.predict_proba

            assert isinstance(exec_info.value.__cause__, AttributeError)
            assert inner_msg in str(exec_info.value.__cause__)

            with pytest.raises(
                AttributeError, match="has no attribute 'predict_log_proba'"
            ) as exec_info:
                clf.predict_log_proba
            assert isinstance(exec_info.value.__cause__, AttributeError)
            assert inner_msg in str(exec_info.value.__cause__)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_proba(klass):
    # 检查 SGDClassifier 的 predict_proba 方法

    # Hinge 损失不允许条件概率估计。
    # 我们不能在这里使用工厂函数，因为它无论如何都会定义 predict_proba。
    clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=10, tol=None).fit(X, Y)
    # 检查分类器 clf 是否缺少 predict_proba 和 predict_log_proba 方法
    assert not hasattr(clf, "predict_proba")
    assert not hasattr(clf, "predict_log_proba")

    # 对于 log_loss 和 modified_huber 损失函数，可以输出概率估计
    # 二分类情况
    for loss in ["log_loss", "modified_huber"]:
        # 使用指定的损失函数、alpha 值和最大迭代次数创建分类器 clf
        clf = klass(loss=loss, alpha=0.01, max_iter=10)
        # 使用训练数据 X 和标签 Y 进行训练
        clf.fit(X, Y)
        # 对新样本 [[3, 2]] 进行概率预测
        p = clf.predict_proba([[3, 2]])
        # 确保预测的第一个类别的概率大于 0.5
        assert p[0, 1] > 0.5
        # 对新样本 [[-1, -1]] 进行概率预测
        p = clf.predict_proba([[-1, -1]])
        # 确保预测的第一个类别的概率小于 0.5
        assert p[0, 1] < 0.5

        # 如果 predict_proba 输出为 0，则会出现 "RuntimeWarning: divide by zero encountered in log"，在此处避免该警告
        with np.errstate(divide="ignore"):
            # 对新样本 [[3, 2]] 进行对数概率预测
            p = clf.predict_log_proba([[3, 2]])
            # 确保第一个类别的对数概率大于第二个类别的对数概率
            assert p[0, 1] > p[0, 0]
            # 对新样本 [[-1, -1]] 进行对数概率预测
            p = clf.predict_log_proba([[-1, -1]])
            # 确保第一个类别的对数概率小于第二个类别的对数概率

    # 对于 log_loss 多分类问题，进行概率估计
    clf = klass(loss="log_loss", alpha=0.01, max_iter=10).fit(X2, Y2)

    # 计算决策函数的值
    d = clf.decision_function([[0.1, -0.1], [0.3, 0.2]])
    # 对新样本 [[0.1, -0.1], [0.3, 0.2]] 进行概率预测
    p = clf.predict_proba([[0.1, -0.1], [0.3, 0.2]])
    # 确保预测的类别是决策函数输出的类别中概率最大的类别
    assert_array_equal(np.argmax(p, axis=1), np.argmax(d, axis=1))
    # 确保概率总和为 1
    assert_almost_equal(p[0].sum(), 1)
    # 确保所有类别的概率大于等于 0
    assert np.all(p[0] >= 0)

    # 对新样本 [[-1, -1]] 进行概率预测
    p = clf.predict_proba([[-1, -1]])
    # 计算决策函数的值
    d = clf.decision_function([[-1, -1]])
    # 确保预测的类别顺序与决策函数输出的类别顺序相同
    assert_array_equal(np.argsort(p[0]), np.argsort(d[0]))

    # 对新样本 [[3, 2]] 进行对数概率预测
    lp = clf.predict_log_proba([[3, 2]])
    # 对新样本 [[3, 2]] 进行概率预测
    p = clf.predict_proba([[3, 2]])
    # 确保对数概率与概率的自然对数相等
    assert_array_almost_equal(np.log(p), lp)

    # 对新样本 [[-1, -1]] 进行对数概率预测
    lp = clf.predict_log_proba([[-1, -1]])
    # 对新样本 [[-1, -1]] 进行概率预测
    p = clf.predict_proba([[-1, -1]])
    # 确保对数概率与概率的自然对数相等

    # Modified Huber 损失函数的多分类概率估计；需要单独测试，因为硬性的零/一概率可能会破坏决策函数输出中的排序
    clf = klass(loss="modified_huber", alpha=0.01, max_iter=10)
    # 使用训练数据 X2 和标签 Y2 进行训练
    clf.fit(X2, Y2)
    # 计算决策函数的值
    d = clf.decision_function([[3, 2]])
    # 对新样本 [[3, 2]] 进行概率预测
    p = clf.predict_proba([[3, 2]])
    # 确保预测的类别是决策函数输出的类别中概率最大的类别
    if klass != SparseSGDClassifier:
        assert np.argmax(d, axis=1) == np.argmax(p, axis=1)
    else:  # 在稀疏测试案例中不成立的情况下（为什么？）
        assert np.argmin(d, axis=1) == np.argmin(p, axis=1)

    # 以下样本的决策函数值小于 -1，这会导致天真的归一化失败（参见 SGDClassifier.predict_proba 中的注释）
    x = X.mean(axis=0)
    # 计算决策函数的值
    d = clf.decision_function([x])
    # 如果所有决策函数值都小于 -1
    if np.all(d < -1):
        # 对样本 x 进行概率预测
        p = clf.predict_proba([x])
        # 确保概率预测值为 [1/3.0, 1/3.0, 1/3.0]
        assert_array_almost_equal(p[0], [1 / 3.0] * 3)
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sgd_l1(klass):
    # 使用pytest的@parametrize装饰器，对SGDClassifier和SparseSGDClassifier进行参数化测试

    # 测试L1正则化
    n = len(X4)
    rng = np.random.RandomState(13)
    idx = np.arange(n)
    rng.shuffle(idx)

    X = X4[idx, :]
    Y = Y4[idx]

    # 初始化分类器对象
    clf = klass(
        penalty="l1",
        alpha=0.2,
        fit_intercept=False,
        max_iter=2000,
        tol=None,
        shuffle=False,
    )
    # 使用数据X, Y进行训练
    clf.fit(X, Y)

    # 检查第一个系数的非零元素，预期为全零
    assert_array_equal(clf.coef_[0, 1:-1], np.zeros((4,)))

    # 使用训练后的模型对X进行预测
    pred = clf.predict(X)
    assert_array_equal(pred, Y)

    # 测试在稠密输入上进行稀疏化
    clf.sparsify()
    # 检查coef_是否为稀疏矩阵
    assert sp.issparse(clf.coef_)

    # 再次使用训练后的模型对X进行预测
    pred = clf.predict(X)
    assert_array_equal(pred, Y)

    # 对带有稀疏coef_的模型进行pickle和unpickle
    clf = pickle.loads(pickle.dumps(clf))
    # 再次检查coef_是否为稀疏矩阵
    assert sp.issparse(clf.coef_)

    # 最后使用训练后的模型对X进行预测
    pred = clf.predict(X)
    assert_array_equal(pred, Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_class_weights(klass):
    # 测试类别权重

    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    # 不使用类别权重进行训练
    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False, class_weight=None)
    clf.fit(X, y)
    # 检查预测结果是否符合预期
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # 给类别1分配一个较小的权重
    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False, class_weight={1: 0.001})
    clf.fit(X, y)

    # 现在超平面应该顺时针旋转，对于这一点的预测结果应该变化
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_equal_class_weight(klass):
    # 测试是否相等的类别权重近似等于没有类别权重

    X = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y = [0, 0, 1, 1]

    # 不使用类别权重进行训练
    clf = klass(alpha=0.1, max_iter=1000, class_weight=None)
    clf.fit(X, y)

    X = [[1, 0], [0, 1]]
    y = [0, 1]

    # 使用相等的类别权重进行训练
    clf_weighted = klass(alpha=0.1, max_iter=1000, class_weight={0: 0.5, 1: 0.5})
    clf_weighted.fit(X, y)

    # 应该是相似的，由于学习率的调整可能有一些小的差异
    assert_almost_equal(clf.coef_, clf_weighted.coef_, decimal=2)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_wrong_class_weight_label(klass):
    # 测试由于不存在的类别标签导致的ValueError异常

    clf = klass(alpha=0.1, max_iter=1000, class_weight={0: 0.5})
    with pytest.raises(ValueError):
        clf.fit(X, Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_weights_multiplied(klass):
    # 测试类别权重和样本权重的乘法关系

    class_weights = {1: 0.6, 2: 0.3}
    rng = np.random.RandomState(0)
    sample_weights = rng.random_sample(Y4.shape[0])
    multiplied_together = np.copy(sample_weights)
    multiplied_together[Y4 == 1] *= class_weights[1]
    multiplied_together[Y4 == 2] *= class_weights[2]
    # 创建一个分类器 clf1，使用给定的参数：alpha=0.1, max_iter=20, class_weight=class_weights
    clf1 = klass(alpha=0.1, max_iter=20, class_weight=class_weights)
    # 创建另一个分类器 clf2，使用相同的参数，但没有指定 class_weight
    clf2 = klass(alpha=0.1, max_iter=20)

    # 使用 clf1 对数据 X4 和标签 Y4 进行拟合，使用给定的样本权重 sample_weights
    clf1.fit(X4, Y4, sample_weight=sample_weights)
    # 使用 clf2 对数据 X4 和标签 Y4 进行拟合，使用给定的权重 multiplied_together
    clf2.fit(X4, Y4, sample_weight=multiplied_together)

    # 断言 clf1 和 clf2 的系数（coef_）在数值上的近似相等
    assert_almost_equal(clf1.coef_, clf2.coef_)
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_balanced_weight(klass):
    # 测试在不平衡数据上使用类权重"""
    
    # 在默认情况下相当平衡的鸢尾花数据集上计算参考指标
    X, y = iris.data, iris.target
    X = scale(X)
    idx = np.arange(X.shape[0])
    rng = np.random.RandomState(6)
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    # 使用不带类权重的分类器拟合数据
    clf = klass(alpha=0.0001, max_iter=1000, class_weight=None, shuffle=False).fit(X, y)
    f1 = metrics.f1_score(y, clf.predict(X), average="weighted")
    assert_almost_equal(f1, 0.96, decimal=1)

    # 使用平衡类权重的分类器做相同的预测
    clf_balanced = klass(
        alpha=0.0001, max_iter=1000, class_weight="balanced", shuffle=False
    ).fit(X, y)
    f1 = metrics.f1_score(y, clf_balanced.predict(X), average="weighted")
    assert_almost_equal(f1, 0.96, decimal=1)

    # 确保在平衡的情况下使用“balanced”不会改变任何东西
    assert_array_almost_equal(clf.coef_, clf_balanced.coef_, 6)

    # 使用鸢尾花数据创建一个非常不平衡的数据集
    X_0 = X[y == 0, :]
    y_0 = y[y == 0]

    X_imbalanced = np.vstack([X] + [X_0] * 10)
    y_imbalanced = np.concatenate([y] + [y_0] * 10)

    # 在不带类权重信息的不平衡数据上拟合模型
    clf = klass(max_iter=1000, class_weight=None, shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average="weighted") < 0.96

    # 使用平衡的类权重拟合模型
    clf = klass(max_iter=1000, class_weight="balanced", shuffle=False)
    clf.fit(X_imbalanced, y_imbalanced)
    y_pred = clf.predict(X)
    assert metrics.f1_score(y, y_pred, average="weighted") > 0.96


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_sample_weights(klass):
    # 测试样本权重
    
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = [1, 1, 1, -1, -1]

    # 使用给定的参数拟合分类器
    clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    clf.fit(X, y)
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # 给类别1的样本赋予小权重
    clf.fit(X, y, sample_weight=[0.001] * 3 + [1] * 2)

    # 现在超平面应该顺时针旋转，并且在这一点的预测应该发生变化
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


@pytest.mark.parametrize(
    "klass", [SGDClassifier, SparseSGDClassifier, SGDOneClassSVM, SparseSGDOneClassSVM]
)
def test_wrong_sample_weights(klass):
    # 测试如果样本权重具有错误的形状，则引发 ValueError
    
    if klass in [SGDClassifier, SparseSGDClassifier]:
        clf = klass(alpha=0.1, max_iter=1000, fit_intercept=False)
    elif klass in [SGDOneClassSVM, SparseSGDOneClassSVM]:
        clf = klass(nu=0.1, max_iter=1000, fit_intercept=False)
    # 提供的样本权重太长
    # 使用 pytest 模块验证是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 clf 对象的 fit 方法，传入 X, Y 和样本权重 np.arange(7)
        clf.fit(X, Y, sample_weight=np.arange(7))
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
# 参数化测试，针对不同的分类器类（SGDClassifier 和 SparseSGDClassifier）执行测试
def test_partial_fit_exception(klass):
    clf = klass(alpha=0.01)
    # 初始化一个分类器实例，设置学习率为 0.01
    # classes was not specified
    # 预期会引发 ValueError 异常，因为没有指定 classes 参数
    with pytest.raises(ValueError):
        clf.partial_fit(X3, Y3)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
# 参数化测试，针对不同的分类器类（SGDClassifier 和 SparseSGDClassifier）执行测试
def test_partial_fit_binary(klass):
    third = X.shape[0] // 3
    # 将数据集划分为三分之一
    clf = klass(alpha=0.01)
    # 初始化一个分类器实例，设置学习率为 0.01
    classes = np.unique(Y)
    # 获取类别的唯一值

    clf.partial_fit(X[:third], Y[:third], classes=classes)
    # 使用部分数据进行拟合，指定类别信息
    assert clf.coef_.shape == (1, X.shape[1])
    # 断言分类器的 coef_ 属性的形状为 (1, 特征数)
    assert clf.intercept_.shape == (1,)
    # 断言分类器的 intercept_ 属性的形状为 (1,)
    assert clf.decision_function([[0, 0]]).shape == (1,)
    # 断言分类器在给定输入 [[0, 0]] 时的决策函数输出形状为 (1,)
    id1 = id(clf.coef_.data)
    # 获取分类器 coef_ 属性的数据标识

    clf.partial_fit(X[third:], Y[third:])
    # 使用剩余数据进行继续拟合
    id2 = id(clf.coef_.data)
    # 获取分类器 coef_ 属性的数据标识
    # check that coef_ haven't been re-allocated
    # 检查 coef_ 属性是否未重新分配
    assert id1, id2
    # 断言 id1 和 id2 的值相等

    y_pred = clf.predict(T)
    # 使用测试数据 T 进行预测
    assert_array_equal(y_pred, true_result)
    # 断言预测结果与真实结果相等


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
# 参数化测试，针对不同的分类器类（SGDClassifier 和 SparseSGDClassifier）执行测试
def test_partial_fit_multiclass(klass):
    third = X2.shape[0] // 3
    # 将数据集 X2 划分为三分之一
    clf = klass(alpha=0.01)
    # 初始化一个分类器实例，设置学习率为 0.01
    classes = np.unique(Y2)
    # 获取类别的唯一值

    clf.partial_fit(X2[:third], Y2[:third], classes=classes)
    # 使用部分数据进行多类别拟合，指定类别信息
    assert clf.coef_.shape == (3, X2.shape[1])
    # 断言分类器的 coef_ 属性的形状为 (3, 特征数)
    assert clf.intercept_.shape == (3,)
    # 断言分类器的 intercept_ 属性的形状为 (3,)
    assert clf.decision_function([[0, 0]]).shape == (1, 3)
    # 断言分类器在给定输入 [[0, 0]] 时的决策函数输出形状为 (1, 3)
    id1 = id(clf.coef_.data)
    # 获取分类器 coef_ 属性的数据标识

    clf.partial_fit(X2[third:], Y2[third:])
    # 使用剩余数据进行继续拟合
    id2 = id(clf.coef_.data)
    # 获取分类器 coef_ 属性的数据标识
    # check that coef_ haven't been re-allocated
    # 检查 coef_ 属性是否未重新分配
    assert id1, id2
    # 断言 id1 和 id2 的值相等


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
# 参数化测试，针对不同的分类器类（SGDClassifier 和 SparseSGDClassifier）执行测试
def test_partial_fit_multiclass_average(klass):
    third = X2.shape[0] // 3
    # 将数据集 X2 划分为三分之一
    clf = klass(alpha=0.01, average=X2.shape[0])
    # 初始化一个分类器实例，设置学习率为 0.01，平均值参数为数据集 X2 的行数
    classes = np.unique(Y2)
    # 获取类别的唯一值

    clf.partial_fit(X2[:third], Y2[:third], classes=classes)
    # 使用部分数据进行多类别拟合，指定类别信息
    assert clf.coef_.shape == (3, X2.shape[1])
    # 断言分类器的 coef_ 属性的形状为 (3, 特征数)
    assert clf.intercept_.shape == (3,)
    # 断言分类器的 intercept_ 属性的形状为 (3,)

    clf.partial_fit(X2[third:], Y2[third:])
    # 使用剩余数据进行继续拟合
    assert clf.coef_.shape == (3, X2.shape[1])
    # 断言分类器的 coef_ 属性的形状为 (3, 特征数)
    assert clf.intercept_.shape == (3,)
    # 断言分类器的 intercept_ 属性的形状为 (3,)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
# 参数化测试，针对不同的分类器类（SGDClassifier 和 SparseSGDClassifier）执行测试
def test_fit_then_partial_fit(klass):
    # Partial_fit should work after initial fit in the multiclass case.
    # 在多类别情况下，初始拟合后应该可以使用 partial_fit 进行继续拟合。
    # Non-regression test for #2496; fit would previously produce a
    # Fortran-ordered coef_ that subsequent partial_fit couldn't handle.
    # 非回归测试 #2496；先前的拟合会生成 Fortran 排序的 coef_，而后续的 partial_fit 无法处理。
    clf = klass()
    # 初始化一个分类器实例
    clf.fit(X2, Y2)
    # 使用完整数据集进行拟合
    clf.partial_fit(X2, Y2)
    # 在已完成初始拟合后，使用 partial_fit 进行拟合，这里不应该抛出异常


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
# 参数化测试，针对不同的分类器类和学习率参数进行测试
def test_partial_fit_equal_fit_classif(klass, lr):
    # 遍历三元组 (X_, Y_, T_) 和 (X2, Y2, T2)
    for X_, Y_, T_ in ((X, Y, T), (X2, Y2, T2)):
        # 使用给定的参数初始化分类器对象
        clf = klass(alpha=0.01, eta0=0.01, max_iter=2, learning_rate=lr, shuffle=False)
        # 使用数据 X_ 和标签 Y_ 进行拟合
        clf.fit(X_, Y_)
        # 对测试数据 T_ 进行预测，得到预测的决策函数值
        y_pred = clf.decision_function(T_)
        # 获取分类器的当前时间步 t
        t = clf.t_

        # 获取 Y_ 中唯一的类别值
        classes = np.unique(Y_)
        # 使用给定的参数重新初始化分类器对象
        clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
        # 在两个时间步上进行部分拟合，使用先前提取的类别
        for i in range(2):
            clf.partial_fit(X_, Y_, classes=classes)
        # 对测试数据 T_ 再次进行预测，得到另一组预测的决策函数值
        y_pred2 = clf.decision_function(T_)

        # 断言分类器的当前时间步仍然是 t
        assert clf.t_ == t
        # 检查两次预测结果的决策函数值是否在小数点后两位精度内相等
        assert_array_almost_equal(y_pred, y_pred2, decimal=2)
# 使用 pytest 的参数化装饰器标记测试函数，对给定的分类器类进行测试
@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_regression_losses(klass):
    # 设置一个确定的随机种子，确保结果的可重复性
    random_state = np.random.RandomState(1)
    
    # 使用给定的参数初始化分类器对象 clf，loss 函数为 epsilon_insensitive
    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.1,
        loss="epsilon_insensitive",
        random_state=random_state,
    )
    # 使用数据 X, Y 进行拟合
    clf.fit(X, Y)
    # 断言预测的准确率为 1.0
    assert 1.0 == np.mean(clf.predict(X) == Y)

    # 使用相同的分类器类和不同的 loss 函数 squared_epsilon_insensitive 初始化 clf
    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.1,
        loss="squared_epsilon_insensitive",
        random_state=random_state,
    )
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

    # 使用分类器类和 loss 函数为 huber 初始化 clf
    clf = klass(alpha=0.01, loss="huber", random_state=random_state)
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)

    # 再次使用分类器类，设定不同的参数和 loss 函数为 squared_error 初始化 clf
    clf = klass(
        alpha=0.01,
        learning_rate="constant",
        eta0=0.01,
        loss="squared_error",
        random_state=random_state,
    )
    clf.fit(X, Y)
    assert 1.0 == np.mean(clf.predict(X) == Y)


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_warm_start_multiclass(klass):
    # 调用 _test_warm_start 函数，测试多类别情况下的热启动
    _test_warm_start(klass, X2, Y2, "optimal")


@pytest.mark.parametrize("klass", [SGDClassifier, SparseSGDClassifier])
def test_multiple_fit(klass):
    # 测试多次调用 fit 函数，输入数据形状不同的情况
    # 初始化分类器 clf，设定 alpha 和 shuffle 参数
    clf = klass(alpha=0.01, shuffle=False)
    clf.fit(X, Y)
    # 断言 clf 对象具有 coef_ 属性

    # 非回归测试：尝试使用不同的标签集合重新拟合分类器 clf
    y = [["ham", "spam"][i] for i in LabelEncoder().fit_transform(Y)]
    clf.fit(X[:, :-1], y)


###############################################################################
# Regression Test Case


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_reg(klass):
    # 检查 SGD 回归是否能给出任何结果
    # 初始化回归器 clf，设定 alpha、max_iter 和 fit_intercept 参数
    clf = klass(alpha=0.1, max_iter=2, fit_intercept=False)
    clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    assert clf.coef_[0] == clf.coef_[1]


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_averaged_computed_correctly(klass):
    # 测试平均回归器是否正确计算
    eta = 0.001
    alpha = 0.01
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)

    # 创建一个没有噪声的简单线性函数
    y = np.dot(X, w)

    # 初始化回归器 clf，设定多个参数
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    # 使用数据 X, y 进行拟合
    clf.fit(X, y)
    # 调用 asgd 函数计算平均权重和截距
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)

    # 断言 clf 的 coef_ 和 intercept_ 与计算得到的平均值几乎相等
    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_sgd_averaged_partial_fit(klass):
    # 设置学习率 eta 和正则化参数 alpha
    eta = 0.001
    alpha = 0.01
    # 设置样本数量和特征数量
    n_samples = 20
    n_features = 10
    # 使用随机数种子创建随机数生成器 rng
    rng = np.random.RandomState(0)
    # 生成服从正态分布的样本数据 X
    X = rng.normal(size=(n_samples, n_features))
    # 生成服从正态分布的权重向量 w
    w = rng.normal(size=n_features)

    # 计算目标值 y，为 X 和 w 的点积（不包含噪声）
    y = np.dot(X, w)

    # 创建分类器对象 clf，使用给定的参数进行初始化
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    # 使用半批量更新方式进行模型的部分拟合（使用前半部分数据）
    clf.partial_fit(X[: int(n_samples / 2)][:], y[: int(n_samples / 2)])
    # 继续使用半批量更新方式进行模型的部分拟合（使用后半部分数据）
    clf.partial_fit(X[int(n_samples / 2) :][:], y[int(n_samples / 2) :])

    # 调用 asgd 函数计算平均权重和平均截距
    average_weights, average_intercept = asgd(klass, X, y, eta, alpha)

    # 断言分类器的权重 coef_ 接近于计算得到的平均权重 average_weights
    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    # 断言分类器的截距 intercept_ 的第一个元素接近于计算得到的平均截距 average_intercept
    assert_almost_equal(clf.intercept_[0], average_intercept, decimal=16)
@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
# 使用 pytest 的 parametrize 装饰器，将 klass 参数化为 SGDRegressor 和 SparseSGDRegressor 两个类
def test_average_sparse(klass):
    # 测试函数：检查在数据中含有 0 的情况下的平均权重

    eta = 0.001
    alpha = 0.01
    clf = klass(
        loss="squared_error",
        learning_rate="constant",
        eta0=eta,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,
        average=True,
        shuffle=False,
    )

    n_samples = Y3.shape[0]

    clf.partial_fit(X3[: int(n_samples / 2)][:], Y3[: int(n_samples / 2)])
    # 使用部分拟合对数据的前半部分进行拟合
    clf.partial_fit(X3[int(n_samples / 2) :][:], Y3[int(n_samples / 2) :])
    # 使用部分拟合对数据的后半部分进行拟合
    average_weights, average_intercept = asgd(klass, X3, Y3, eta, alpha)
    # 调用 asgd 函数计算平均权重和平均截距

    assert_array_almost_equal(clf.coef_, average_weights, decimal=16)
    # 断言：验证 clf 对象的权重与计算得到的平均权重接近
    assert_almost_equal(clf.intercept_, average_intercept, decimal=16)
    # 断言：验证 clf 对象的截距与计算得到的平均截距接近


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
# 使用 pytest 的 parametrize 装饰器，将 klass 参数化为 SGDRegressor 和 SparseSGDRegressor 两个类
def test_sgd_least_squares_fit(klass):
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(loss="squared_error", alpha=0.1, max_iter=20, fit_intercept=False)
    # 使用 klass 类初始化一个回归器对象，指定损失函数、正则化参数、最大迭代次数和是否拟合截距
    clf.fit(X, y)
    # 使用数据 X 和标签 y 拟合回归器对象
    score = clf.score(X, y)
    # 计算并存储回归器对象在给定数据集上的得分
    assert score > 0.99
    # 断言：验证回归器对象在数据集上的得分是否大于0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf = klass(loss="squared_error", alpha=0.1, max_iter=20, fit_intercept=False)
    # 使用 klass 类初始化一个回归器对象，指定损失函数、正则化参数、最大迭代次数和是否拟合截距
    clf.fit(X, y)
    # 使用数据 X 和带噪声的标签 y 拟合回归器对象
    score = clf.score(X, y)
    # 计算并存储回归器对象在给定数据集上的得分
    assert score > 0.5
    # 断言：验证回归器对象在数据集上的得分是否大于0.5


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
# 使用 pytest 的 parametrize 装饰器，将 klass 参数化为 SGDRegressor 和 SparseSGDRegressor 两个类
def test_sgd_epsilon_insensitive(klass):
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(
        loss="epsilon_insensitive",
        epsilon=0.01,
        alpha=0.1,
        max_iter=20,
        fit_intercept=False,
    )
    # 使用 klass 类初始化一个回归器对象，指定损失函数、epsilon、正则化参数、最大迭代次数和是否拟合截距
    clf.fit(X, y)
    # 使用数据 X 和标签 y 拟合回归器对象
    score = clf.score(X, y)
    # 计算并存储回归器对象在给定数据集上的得分
    assert score > 0.99
    # 断言：验证回归器对象在数据集上的得分是否大于0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()

    clf = klass(
        loss="epsilon_insensitive",
        epsilon=0.01,
        alpha=0.1,
        max_iter=20,
        fit_intercept=False,
    )
    # 使用 klass 类初始化一个回归器对象，指定损失函数、epsilon、正则化参数、最大迭代次数和是否拟合截距
    clf.fit(X, y)
    # 使用数据 X 和带噪声的标签 y 拟合回归器对象
    score = clf.score(X, y)
    # 计算并存储回归器对象在给定数据集上的得分
    assert score > 0.5
    # 断言：验证回归器对象在数据集上的得分是否大于0.5


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
# 使用 pytest 的 parametrize 装饰器，将 klass 参数化为 SGDRegressor 和 SparseSGDRegressor 两个类
def test_sgd_huber_fit(klass):
    xmin, xmax = -5, 5
    n_samples = 100
    rng = np.random.RandomState(0)
    X = np.linspace(xmin, xmax, n_samples).reshape(n_samples, 1)

    # simple linear function without noise
    y = 0.5 * X.ravel()

    clf = klass(loss="huber", epsilon=0.1, alpha=0.1, max_iter=20, fit_intercept=False)
    # 使用 klass 类初始化一个回归器对象，指定损失函数、huber 参数、epsilon、正则化参数、最大迭代次数和是否拟合截距
    clf.fit(X, y)
    # 使用数据 X 和标签 y 拟合回归器对象
    score = clf.score(X, y)
    # 计算并存储回归器对象在给定数据集上的得分
    assert score > 0.99
    # 断言：验证回归器对象在数据集上的得分是否大于0.99

    # simple linear function with noise
    y = 0.5 * X.ravel() + rng.randn(n_samples, 1).ravel()
    # 使用指定参数初始化一个 Huber 回归模型，设置损失函数为 "huber"，容忍度为 0.1，正则化项参数为 0.1，最大迭代次数为 20，不拟合截距项
    clf = klass(loss="huber", epsilon=0.1, alpha=0.1, max_iter=20, fit_intercept=False)
    
    # 使用输入数据 X 和目标数据 y 来训练分类器 clf
    clf.fit(X, y)
    
    # 计算分类器在训练数据 X 上的得分，并将结果保存在变量 score 中
    score = clf.score(X, y)
    
    # 断言分类器在训练数据上的得分 score 大于 0.5
    assert score > 0.5
@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_elasticnet_convergence(klass):
    # 检查 SGD 输出是否与坐标下降一致

    n_samples, n_features = 1000, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    # 生成基准线性模型，用于从 X 生成 y，并且如果正则化参数设置为 0.0，模型应该收敛到这个基准模型
    ground_truth_coef = rng.randn(n_features)
    y = np.dot(X, ground_truth_coef)

    # XXX: alpha = 0.1 似乎导致收敛问题
    for alpha in [0.01, 0.001]:
        for l1_ratio in [0.5, 0.8, 1.0]:
            cd = linear_model.ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False
            )
            cd.fit(X, y)
            sgd = klass(
                penalty="elasticnet",
                max_iter=50,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=False,
            )
            sgd.fit(X, y)
            err_msg = (
                "cd and sgd did not converge to comparable "
                "results for alpha=%f and l1_ratio=%f" % (alpha, l1_ratio)
            )
            assert_almost_equal(cd.coef_, sgd.coef_, decimal=2, err_msg=err_msg)


@ignore_warnings
@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_partial_fit(klass):
    third = X.shape[0] // 3
    clf = klass(alpha=0.01)

    clf.partial_fit(X[:third], Y[:third])
    assert clf.coef_.shape == (X.shape[1],)
    assert clf.intercept_.shape == (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    id1 = id(clf.coef_.data)

    clf.partial_fit(X[third:], Y[third:])
    id2 = id(clf.coef_.data)
    # 检查 coef_ 是否没有重新分配内存
    assert id1, id2


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
def test_partial_fit_equal_fit(klass, lr):
    clf = klass(alpha=0.01, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X, Y)
    y_pred = clf.predict(T)
    t = clf.t_

    clf = klass(alpha=0.01, eta0=0.01, learning_rate=lr, shuffle=False)
    for i in range(2):
        clf.partial_fit(X, Y)
    y_pred2 = clf.predict(T)

    assert clf.t_ == t
    assert_array_almost_equal(y_pred, y_pred2, decimal=2)


@pytest.mark.parametrize("klass", [SGDRegressor, SparseSGDRegressor])
def test_loss_function_epsilon(klass):
    clf = klass(epsilon=0.9)
    clf.set_params(epsilon=0.1)
    assert clf.loss_functions["huber"][1] == 0.1


###############################################################################
# SGD One Class SVM Test Case


# 一个简单的 ASGD 实现，用于测试 SGDOneClassSVM
def asgd_oneclass(klass, X, eta, nu, coef_init=None, offset_init=0.0):
    if coef_init is None:
        coef = np.zeros(X.shape[1])
    else:
        coef = coef_init

    average_coef = np.zeros(X.shape[1])
    # 初始化偏移量为初始偏移量
    offset = offset_init
    # 计算截距，初始为 1 减去偏移量
    intercept = 1 - offset
    # 平均截距初始值为 0.0
    average_intercept = 0.0
    # 初始化衰减率为 1.0
    decay = 1.0

    # 如果类别为 SparseSGDOneClassSVM，则设置衰减率为 0.01
    if klass == SparseSGDOneClassSVM:
        decay = 0.01

    # 遍历输入数据 X 中的每个条目
    for i, entry in enumerate(X):
        # 计算 p，为条目 entry 与系数 coef 的点积
        p = np.dot(entry, coef)
        # 将截距加到 p 上
        p += intercept
        # 如果 p 小于等于 1.0，则设置梯度为 -1，否则为 0
        if p <= 1.0:
            gradient = -1
        else:
            gradient = 0
        # 更新系数 coef，使用学习率 eta、正则化参数 nu 和梯度 gradient 计算
        coef *= max(0, 1.0 - (eta * nu / 2))
        coef += -(eta * gradient * entry)
        # 更新截距 intercept，使用学习率 eta、正则化参数 nu 和梯度 gradient 计算，并乘以衰减率 decay
        intercept += -(eta * (nu + gradient)) * decay

        # 更新平均系数 average_coef，计算累积平均值
        average_coef *= i
        average_coef += coef
        average_coef /= i + 1.0

        # 更新平均截距 average_intercept，计算累积平均值
        average_intercept *= i
        average_intercept += intercept
        average_intercept /= i + 1.0

    # 返回累积平均系数和 1 减去累积平均截距
    return average_coef, 1 - average_intercept
@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
# 使用 pytest 的 parametrize 标记，为测试函数提供不同的参数化输入 klass，可以是 SGDOneClassSVM 或 SparseSGDOneClassSVM
def _test_warm_start_oneclass(klass, X, lr):
    # 测试显式和隐式的热启动功能

    # 创建一个分类器 clf，使用给定的参数实例化 klass 类，然后在数据集 X 上进行拟合
    clf = klass(nu=0.5, eta0=0.01, shuffle=False, learning_rate=lr)
    clf.fit(X)

    # 创建另一个分类器 clf2，再次使用 klass 类实例化，但使用前一个分类器 clf 的权重和偏置作为初始值进行拟合
    clf2 = klass(nu=0.1, eta0=0.01, shuffle=False, learning_rate=lr)
    clf2.fit(X, coef_init=clf.coef_.copy(), offset_init=clf.offset_.copy())

    # 使用隐式热启动方式创建分类器 clf3，使用相同的学习率 lr，和初始 nu=0.5 在数据集 X 上进行拟合
    clf3 = klass(nu=0.5, eta0=0.01, shuffle=False, warm_start=True, learning_rate=lr)
    clf3.fit(X)

    # 断言隐式热启动后的分类器 clf3 和初始的 clf 在相同迭代次数下（t_）具有相同的状态
    assert clf3.t_ == clf.t_

    # 断言隐式热启动后的分类器 clf3 的权重（coef_）与初始的 clf 非常接近
    assert_allclose(clf3.coef_, clf.coef_)

    # 修改 clf3 的参数 nu=0.1，并在数据集 X 上重新进行拟合
    clf3.set_params(nu=0.1)
    clf3.fit(X)

    # 断言 nu=0.1 的分类器 clf3 和初始化的 clf2 在相同迭代次数下（t_）具有相同的状态
    assert clf3.t_ == clf2.t_

    # 断言 nu=0.1 的分类器 clf3 的权重（coef_）与初始化的 clf2 非常接近
    assert_allclose(clf3.coef_, clf2.coef_)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
# 使用 pytest 的 parametrize 标记，为测试函数提供两个参数化输入 klass 和 lr，分别测试不同的学习率
def test_warm_start_oneclass(klass, lr):
    # 调用内部函数 _test_warm_start_oneclass，使用给定的 klass 和 lr 参数
    _test_warm_start_oneclass(klass, X, lr)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
# 使用 pytest 的 parametrize 标记，为测试函数提供不同的参数化输入 klass，可以是 SGDOneClassSVM 或 SparseSGDOneClassSVM
def test_clone_oneclass(klass):
    # 测试克隆功能是否正常运行

    # 创建分类器 clf，使用给定的 nu=0.5 实例化 klass 类
    clf = klass(nu=0.5)
    
    # 使用 clone 函数克隆分类器 clf，并将 nu 参数修改为 0.1
    clf = clone(clf)
    clf.set_params(nu=0.1)
    
    # 在数据集 X 上进行拟合
    clf.fit(X)

    # 创建另一个分类器 clf2，使用 nu=0.1 实例化 klass 类，并在数据集 X 上进行拟合
    clf2 = klass(nu=0.1)
    clf2.fit(X)

    # 断言克隆后的分类器 clf 的权重（coef_）与分类器 clf2 的权重非常接近
    assert_array_equal(clf.coef_, clf2.coef_)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
# 使用 pytest 的 parametrize 标记，为测试函数提供不同的参数化输入 klass，可以是 SGDOneClassSVM 或 SparseSGDOneClassSVM
def test_partial_fit_oneclass(klass):
    # 测试部分拟合功能

    # 计算数据集 X 的三分之一长度
    third = X.shape[0] // 3
    
    # 使用给定的 nu=0.1 实例化 klass 类创建分类器 clf
    clf = klass(nu=0.1)

    # 在数据集 X 的前 third 部分进行部分拟合
    clf.partial_fit(X[:third])

    # 断言分类器 clf 的权重（coef_）的形状为 (X 的特征数,)
    assert clf.coef_.shape == (X.shape[1],)
    
    # 断言分类器 clf 的偏置（offset_）的形状为 (1,)
    assert clf.offset_.shape == (1,)
    
    # 断言预测一个新数据 [[0, 0]] 后的结果形状为 (1,)
    assert clf.predict([[0, 0]]).shape == (1,)
    
    # 记录部分拟合后的 coef_ 引用
    previous_coefs = clf.coef_

    # 在数据集 X 的后 two-thirds 部分进行部分拟合
    clf.partial_fit(X[third:])

    # 断言部分拟合后的 coef_ 仍然是之前的引用 previous_coefs
    assert clf.coef_ is previous_coefs

    # 如果特征数与之前数据不匹配，会引发 ValueError
    with pytest.raises(ValueError):
        clf.partial_fit(X[:, 1])


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
@pytest.mark.parametrize("lr", ["constant", "optimal", "invscaling", "adaptive"])
# 使用 pytest 的 parametrize 标记，为测试函数提供两个参数化输入 klass 和 lr，分别测试不同的学习率
def test_partial_fit_equal_fit_oneclass(klass, lr):
    # 测试部分拟合与完全拟合的等效性

    # 使用给定的参数实例化 klass 类创建分类器 clf，设置 max_iter=2，并在数据集 X 上进行完全拟合
    clf = klass(nu=0.05, max_iter=2, eta0=0.01, learning_rate=lr, shuffle=False)
    clf.fit(X)

    # 获取完全拟合后的决策函数结果 y_scores、迭代次数 t、权重 coef_ 和偏置 offset_
    y_scores = clf.decision_function(T)
    t = clf.t_
    coef = clf.coef_
    offset = clf.offset_

    # 使用给定的 nu=0.05 和 learning_rate=lr 实例化 klass 类创建分类器 clf，并进行两次部分拟合
    clf = klass(nu=0.05, eta0=0.01, max_iter=1, learning_rate=lr, shuffle=False)
    for _ in range(2):
        clf.partial_fit(X)

    # 获取部分拟合后的决策函数结果 y_scores2
    y_scores2 = clf.decision_function(T)

    # 断言部分拟合和完全拟合的分类器在相同迭代次数下 t 相等
    assert clf.t_ == t

    # 断言部分拟合和完全拟合的决策函数结果 y_scores 和 y_scores2 非常接近
    assert_allclose(y_scores, y_scores2)

    # 断言部分拟合和完全拟合的权重 coef 和偏置 offset 非常接近
    assert_allclose(clf.coef_, coef)
    assert_allclose(clf.offset_, offset)


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
# 使用 pytest 的 parametrize 标记，为测试函数提供不同的参数化输入 klass，可以是 SGDOneClassSVM 或 SparseSGDOneClassSVM
def test_late_onset_averaging_reached_oneclass(klass):
    # 测试延迟启动平均

    # 设置学习率 eta0 和 nu 参数
    eta0 = 0.001
    nu = 0.05

    # 进行两次训练集的遍历，
    # 使用给定参数初始化一个分类器 clf1，设置参数包括平均、学习率、初始学习率 eta0、nu、最大迭代次数和不进行数据洗牌
    clf1 = klass(
        average=7, learning_rate="constant", eta0=eta0, nu=nu, max_iter=2, shuffle=False
    )
    
    # 使用给定参数初始化另一个分类器 clf2，设置参数包括不进行平均、学习率、初始学习率 eta0、nu、最大迭代次数为1和不进行数据洗牌
    # 这里的 klass 可能是一个类或者函数，用来创建分类器实例
    clf2 = klass(
        average=False,
        learning_rate="constant",
        eta0=eta0,
        nu=nu,
        max_iter=1,
        shuffle=False,
    )

    # 使用分类器 clf1 对训练集 X 进行拟合
    clf1.fit(X)
    
    # 使用分类器 clf2 对训练集 X 进行拟合
    clf2.fit(X)

    # 从 clf2 的解开始，使用 asgd_oneclass 函数计算平均值，并与 clf1 的解进行比较
    # asgd_oneclass 可能是一个函数，用于执行随机梯度下降平均计算
    average_coef, average_offset = asgd_oneclass(
        klass, X, eta0, nu, coef_init=clf2.coef_.ravel(), offset_init=clf2.offset_
    )

    # 断言 clf1 的系数（展平后）与计算得到的平均系数相近
    assert_allclose(clf1.coef_.ravel(), average_coef.ravel())
    
    # 断言 clf1 的偏移与计算得到的平均偏移相近
    assert_allclose(clf1.offset_, average_offset)
@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_averaged_computed_correctly_oneclass(klass):
    # 测试平均化的 SGD One-Class SVM 是否与朴素实现一致
    eta = 0.001  # 学习率参数 eta
    nu = 0.05    # nu 参数
    n_samples = 20  # 样本数
    n_features = 10  # 特征数
    rng = np.random.RandomState(0)  # 创建随机数生成器对象
    X = rng.normal(size=(n_samples, n_features))  # 生成服从正态分布的随机数据

    clf = klass(
        learning_rate="constant",  # 使用常数学习率
        eta0=eta,  # 初始学习率
        nu=nu,  # nu 参数
        fit_intercept=True,  # 拟合截距
        max_iter=1,  # 最大迭代次数
        average=True,  # 开启平均化
        shuffle=False,  # 不进行数据洗牌
    )

    clf.fit(X)  # 拟合模型
    average_coef, average_offset = asgd_oneclass(klass, X, eta, nu)  # 计算平均系数和平均偏移量

    assert_allclose(clf.coef_, average_coef)  # 断言检查模型的系数是否与平均系数接近
    assert_allclose(clf.offset_, average_offset)  # 断言检查模型的偏移量是否与平均偏移量接近


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_sgd_averaged_partial_fit_oneclass(klass):
    # 测试部分拟合是否产生与完全拟合相同的平均值
    eta = 0.001  # 学习率参数 eta
    nu = 0.05    # nu 参数
    n_samples = 20  # 样本数
    n_features = 10  # 特征数
    rng = np.random.RandomState(0)  # 创建随机数生成器对象
    X = rng.normal(size=(n_samples, n_features))  # 生成服从正态分布的随机数据

    clf = klass(
        learning_rate="constant",  # 使用常数学习率
        eta0=eta,  # 初始学习率
        nu=nu,  # nu 参数
        fit_intercept=True,  # 拟合截距
        max_iter=1,  # 最大迭代次数
        average=True,  # 开启平均化
        shuffle=False,  # 不进行数据洗牌
    )

    clf.partial_fit(X[: int(n_samples / 2)][:])  # 部分拟合模型
    clf.partial_fit(X[int(n_samples / 2) :][:])  # 继续部分拟合模型
    average_coef, average_offset = asgd_oneclass(klass, X, eta, nu)  # 计算平均系数和平均偏移量

    assert_allclose(clf.coef_, average_coef)  # 断言检查模型的系数是否与平均系数接近
    assert_allclose(clf.offset_, average_offset)  # 断言检查模型的偏移量是否与平均偏移量接近


@pytest.mark.parametrize("klass", [SGDOneClassSVM, SparseSGDOneClassSVM])
def test_average_sparse_oneclass(klass):
    # 检查在含有0的数据上的平均系数
    eta = 0.001  # 学习率参数 eta
    nu = 0.01    # nu 参数
    clf = klass(
        learning_rate="constant",  # 使用常数学习率
        eta0=eta,  # 初始学习率
        nu=nu,  # nu 参数
        fit_intercept=True,  # 拟合截距
        max_iter=1,  # 最大迭代次数
        average=True,  # 开启平均化
        shuffle=False,  # 不进行数据洗牌
    )

    n_samples = X3.shape[0]  # 获取样本数

    clf.partial_fit(X3[: int(n_samples / 2)])  # 部分拟合模型
    clf.partial_fit(X3[int(n_samples / 2) :])  # 继续部分拟合模型
    average_coef, average_offset = asgd_oneclass(klass, X3, eta, nu)  # 计算平均系数和平均偏移量

    assert_allclose(clf.coef_, average_coef)  # 断言检查模型的系数是否与平均系数接近
    assert_allclose(clf.offset_, average_offset)  # 断言检查模型的偏移量是否与平均偏移量接近


def test_sgd_oneclass():
    # 测试在一个简单数据集上的拟合、决策函数、预测和样本分数
    X_train = np.array([[-2, -1], [-1, -1], [1, 1]])  # 训练数据集
    X_test = np.array([[0.5, -2], [2, 2]])  # 测试数据集
    clf = SGDOneClassSVM(
        nu=0.5, eta0=1, learning_rate="constant", shuffle=False, max_iter=1
    )  # 创建 SGD One-Class SVM 模型对象
    clf.fit(X_train)  # 拟合模型

    assert_allclose(clf.coef_, np.array([-0.125, 0.4375]))  # 断言检查模型的系数是否与预期值接近
    assert clf.offset_[0] == -0.5  # 断言检查模型的偏移量是否与预期值一致

    scores = clf.score_samples(X_test)  # 计算测试数据集的样本分数
    assert_allclose(scores, np.array([-0.9375, 0.625]))  # 断言检查样本分数是否与预期值接近

    dec = clf.score_samples(X_test) - clf.offset_  # 计算修正后的决策函数值
    assert_allclose(clf.decision_function(X_test), dec)  # 断言检查决策函数值是否与预期值接近

    pred = clf.predict(X_test)  # 预测测试数据集的标签
    assert_array_equal(pred, np.array([-1, 1]))  # 断言检查预测结果是否与预期值一致


def test_ocsvm_vs_sgdocsvm():
    # 检查 SGD One-Class SVM 是否对核化数据给出良好的近似
    # 设置 One-Class SVM 的参数
    nu = 0.05    # 控制支持向量的比例
    gamma = 2.0  # 核函数的系数

    # 生成训练和测试数据集
    random_state = 42
    rng = np.random.RandomState(random_state)
    X = 0.3 * rng.randn(500, 2)    # 生成500个数据点，每个点有两个特征，符合正态分布
    X_train = np.r_[X + 2, X - 2] # 构造训练数据，分别平移生成两组数据
    X = 0.3 * rng.randn(100, 2)
    X_test = np.r_[X + 2, X - 2]  # 构造测试数据，分别平移生成两组数据

    # 初始化 One-Class SVM 模型
    clf = OneClassSVM(gamma=gamma, kernel="rbf", nu=nu)
    clf.fit(X_train)    # 在训练数据上拟合模型
    y_pred_ocsvm = clf.predict(X_test)    # 预测测试数据的标签
    dec_ocsvm = clf.decision_function(X_test).reshape(1, -1)    # 计算测试数据的决策函数值

    # 使用 SGDOneClassSVM 结合核近似
    max_iter = 15
    transform = Nystroem(gamma=gamma, random_state=random_state)    # 核近似转换器
    clf_sgd = SGDOneClassSVM(
        nu=nu,
        shuffle=True,
        fit_intercept=True,
        max_iter=max_iter,
        random_state=random_state,
        tol=None,
    )
    pipe_sgd = make_pipeline(transform, clf_sgd)    # 构建管道，将核近似和SGD模型组合
    pipe_sgd.fit(X_train)    # 在训练数据上拟合管道
    y_pred_sgdocsvm = pipe_sgd.predict(X_test)    # 预测测试数据的标签
    dec_sgdocsvm = pipe_sgd.decision_function(X_test).reshape(1, -1)    # 计算测试数据的决策函数值

    # 断言预测标签的匹配度至少为0.99
    assert np.mean(y_pred_sgdocsvm == y_pred_ocsvm) >= 0.99
    # 计算两种方法的决策函数值的相关系数，并断言相关系数至少为0.9
    corrcoef = np.corrcoef(np.concatenate((dec_ocsvm, dec_sgdocsvm)))[0, 1]
    assert corrcoef >= 0.9
# 定义测试函数，用于验证 l1_ratio 参数在 SGDClassifier 中的影响
def test_l1_ratio():
    # 使用 make_classification 生成具有指定特征的分类数据集 X 和标签 y
    X, y = datasets.make_classification(
        n_samples=1000, n_features=100, n_informative=20, random_state=1234
    )

    # 测试 l1_ratio 接近 1 时，elasticnet 是否与纯 l1 相同
    est_en = SGDClassifier(
        alpha=0.001,
        penalty="elasticnet",
        tol=None,
        max_iter=6,
        l1_ratio=0.9999999999,
        random_state=42,
    ).fit(X, y)
    est_l1 = SGDClassifier(
        alpha=0.001, penalty="l1", max_iter=6, random_state=42, tol=None
    ).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l1.coef_)

    # 测试 l1_ratio 接近 0 时，elasticnet 是否与纯 l2 相同
    est_en = SGDClassifier(
        alpha=0.001,
        penalty="elasticnet",
        tol=None,
        max_iter=6,
        l1_ratio=0.0000000001,
        random_state=42,
    ).fit(X, y)
    est_l2 = SGDClassifier(
        alpha=0.001, penalty="l2", max_iter=6, random_state=42, tol=None
    ).fit(X, y)
    assert_array_almost_equal(est_en.coef_, est_l2.coef_)


# 定义测试函数，用于验证数值稳定性在数值下溢或上溢情况下的表现
def test_underflow_or_overlow():
    # 在 numpy 中设置错误状态全部抛出异常
    with np.errstate(all="raise"):
        # 使用随机数生成器生成具有极大未缩放特征的异常数据
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 10

        X = rng.normal(size=(n_samples, n_features))
        X[:, :2] *= 1e300
        assert np.isfinite(X).all()

        # 使用 MinMaxScaler 对数据进行缩放，以避免数值不稳定性
        X_scaled = MinMaxScaler().fit_transform(X)
        assert np.isfinite(X_scaled).all()

        # 在缩放后的数据上定义一个基准真值
        ground_truth = rng.normal(size=n_features)
        y = (np.dot(X_scaled, ground_truth) > 0.0).astype(np.int32)
        assert_array_equal(np.unique(y), [0, 1])

        # 创建一个 SGDClassifier 模型，用于测试在缩放数据上的稳定性
        model = SGDClassifier(alpha=0.1, loss="squared_hinge", max_iter=500)

        # 烟雾测试：模型在缩放数据上稳定
        model.fit(X_scaled, y)
        assert np.isfinite(model.coef_).all()

        # 模型在未缩放数据上数值不稳定
        msg_regxp = (
            r"Floating-point under-/overflow occurred at epoch #.*"
            " Scaling input data with StandardScaler or MinMaxScaler"
            " might help."
        )
        # 使用 pytest 检查是否引发了数值异常的异常
        with pytest.raises(ValueError, match=msg_regxp):
            model.fit(X, y)


# 定义测试函数，用于验证在大梯度问题上的数值稳定性
def test_numerical_stability_large_gradient():
    # 针对数值稳定性的非回归测试案例，测试在 scaled 问题上的稳定性
    model = SGDClassifier(
        loss="squared_hinge",
        max_iter=10,
        shuffle=True,
        penalty="elasticnet",
        l1_ratio=0.3,
        alpha=0.01,
        eta0=0.001,
        random_state=0,
        tol=None,
    )
    # 在 numpy 中设置错误状态全部抛出异常
    with np.errstate(all="raise"):
        # 使用 iris 数据集训练模型，检验其数值稳定性
        model.fit(iris.data, iris.target)
    # 断言检查模型的系数是否全部是有限数值
    assert np.isfinite(model.coef_).all()
@pytest.mark.parametrize("penalty", ["l2", "l1", "elasticnet"])
def test_large_regularization(penalty):
    # 对于由于大正则化参数引起的数值稳定性问题的非回归测试
    model = SGDClassifier(
        alpha=1e5,
        learning_rate="constant",
        eta0=0.1,
        penalty=penalty,
        shuffle=False,
        tol=None,
        max_iter=6,
    )
    # 设置数值异常状态为抛出
    with np.errstate(all="raise"):
        model.fit(iris.data, iris.target)
    # 断言模型的系数接近零向量
    assert_array_almost_equal(model.coef_, np.zeros_like(model.coef_))


def test_tol_parameter():
    # 测试 tol 参数的行为是否符合预期
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target == 1

    # 当 tol 为 None 时，迭代次数应该等于 max_iter
    max_iter = 42
    model_0 = SGDClassifier(tol=None, random_state=0, max_iter=max_iter)
    model_0.fit(X, y)
    assert max_iter == model_0.n_iter_

    # 如果 tol 不为 None，迭代次数应该少于 max_iter
    max_iter = 2000
    model_1 = SGDClassifier(tol=0, random_state=0, max_iter=max_iter)
    model_1.fit(X, y)
    assert max_iter > model_1.n_iter_
    assert model_1.n_iter_ > 5

    # 较大的 tol 应该导致较少的迭代次数
    model_2 = SGDClassifier(tol=0.1, random_state=0, max_iter=max_iter)
    model_2.fit(X, y)
    assert model_1.n_iter_ > model_2.n_iter_
    assert model_2.n_iter_ > 3

    # 严格的容差和小的 max_iter 应该触发警告
    model_3 = SGDClassifier(max_iter=3, tol=1e-3, random_state=0)
    warning_message = (
        "Maximum number of iteration reached before "
        "convergence. Consider increasing max_iter to "
        "improve the fit."
    )
    # 使用 pytest 的 warns 方法检查是否出现 ConvergenceWarning
    with pytest.warns(ConvergenceWarning, match=warning_message):
        model_3.fit(X, y)
    assert model_3.n_iter_ == 3


def _test_loss_common(loss_function, cases):
    # 测试不同的损失函数
    # cases 是一个元组列表，包含 (p, y, expected_loss, expected_dloss)
    for p, y, expected_loss, expected_dloss in cases:
        # 断言计算得到的损失函数值接近预期值
        assert_almost_equal(loss_function.py_loss(p, y), expected_loss)
        # 断言计算得到的损失函数导数接近预期值
        assert_almost_equal(loss_function.py_dloss(p, y), expected_dloss)


def test_loss_hinge():
    # 测试 Hinge 损失函数（hinge / 感知器）
    # hinge
    loss = sgd_fast.Hinge(1.0)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.1, 1.0, 0.0, 0.0),
        (-2.0, -1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, -1.0),
        (-1.0, -1.0, 0.0, 1.0),
        (0.5, 1.0, 0.5, -1.0),
        (2.0, -1.0, 3.0, 1.0),
        (-0.5, -1.0, 0.5, 1.0),
        (0.0, 1.0, 1, -1.0),
    ]
    # 调用 _test_loss_common 进行 Hinge 损失函数的测试
    _test_loss_common(loss, cases)

    # perceptron
    loss = sgd_fast.Hinge(0.0)
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (1.0, 1.0, 0.0, 0.0),
        (-0.1, -1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, -1.0),
        (0.0, -1.0, 0.0, 1.0),
        (0.5, -1.0, 0.5, 1.0),
        (2.0, -1.0, 2.0, 1.0),
        (-0.5, 1.0, 0.5, -1.0),
        (-1.0, 1.0, 1.0, -1.0),
    ]
    # 再次调用 _test_loss_common 进行感知器损失函数的测试
    _test_loss_common(loss, cases)
    ]
    # 调用 _test_loss_common 函数，并传入 loss 和 cases 作为参数进行测试
    _test_loss_common(loss, cases)
def test_gradient_squared_hinge():
    # Test SquaredHinge
    # 创建一个 SquaredHinge 的损失函数对象，参数为 1.0
    loss = sgd_fast.SquaredHinge(1.0)
    # 定义测试用例，每个元组包含 (预测值p, 真实标签y, 预期损失expected_loss, 预期梯度expected_dloss)
    cases = [
        (1.0, 1.0, 0.0, 0.0),
        (-2.0, -1.0, 0.0, 0.0),
        (1.0, -1.0, 4.0, 4.0),
        (-1.0, 1.0, 4.0, -4.0),
        (0.5, 1.0, 0.25, -1.0),
        (0.5, -1.0, 2.25, 3.0),
    ]
    # 调用通用的测试损失函数的函数，传入损失对象和测试用例
    _test_loss_common(loss, cases)


def test_loss_log():
    # Test Log (logistic loss)
    # 创建一个 Log (logistic loss) 的损失函数对象
    loss = sgd_fast.Log()
    # 定义测试用例，每个元组包含 (预测值p, 真实标签y, 预期损失expected_loss, 预期梯度expected_dloss)
    cases = [
        (1.0, 1.0, np.log(1.0 + np.exp(-1.0)), -1.0 / (np.exp(1.0) + 1.0)),
        (1.0, -1.0, np.log(1.0 + np.exp(1.0)), 1.0 / (np.exp(-1.0) + 1.0)),
        (-1.0, -1.0, np.log(1.0 + np.exp(-1.0)), 1.0 / (np.exp(1.0) + 1.0)),
        (-1.0, 1.0, np.log(1.0 + np.exp(1.0)), -1.0 / (np.exp(-1.0) + 1.0)),
        (0.0, 1.0, np.log(2), -0.5),
        (0.0, -1.0, np.log(2), 0.5),
        (17.9, -1.0, 17.9, 1.0),
        (-17.9, 1.0, 17.9, -1.0),
    ]
    # 调用通用的测试损失函数的函数，传入损失对象和测试用例
    _test_loss_common(loss, cases)
    # 使用 assert_almost_equal 函数验证特定情况下的梯度和损失值
    assert_almost_equal(loss.py_dloss(18.1, 1.0), np.exp(-18.1) * -1.0, 16)
    assert_almost_equal(loss.py_loss(18.1, 1.0), np.exp(-18.1), 16)
    assert_almost_equal(loss.py_dloss(-18.1, -1.0), np.exp(-18.1) * 1.0, 16)
    assert_almost_equal(loss.py_loss(-18.1, 1.0), 18.1, 16)


def test_loss_squared_loss():
    # Test SquaredLoss
    # 创建一个 SquaredLoss 的损失函数对象
    loss = sgd_fast.SquaredLoss()
    # 定义测试用例，每个元组包含 (预测值p, 真实标签y, 预期损失expected_loss, 预期梯度expected_dloss)
    cases = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.5, 1.0),
        (0.5, -1.0, 1.125, 1.5),
        (-2.5, 2.0, 10.125, -4.5),
    ]
    # 调用通用的测试损失函数的函数，传入损失对象和测试用例
    _test_loss_common(loss, cases)


def test_loss_huber():
    # Test Huber
    # 创建一个 Huber 的损失函数对象，delta 参数为 0.1
    loss = sgd_fast.Huber(0.1)
    # 定义测试用例，每个元组包含 (预测值p, 真实标签y, 预期损失expected_loss, 预期梯度expected_dloss)
    cases = [
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.005, 0.1),
        (0.0, 0.1, 0.005, -0.1),
        (3.95, 4.0, 0.00125, -0.05),
        (5.0, 2.0, 0.295, 0.1),
        (-1.0, 5.0, 0.595, -0.1),
    ]
    # 调用通用的测试损失函数的函数，传入损失对象和测试用例
    _test_loss_common(loss, cases)


def test_loss_modified_huber():
    # Test ModifiedHuber
    # 创建一个 ModifiedHuber 的损失函数对象
    loss = sgd_fast.ModifiedHuber()
    # 定义测试用例，每个元组包含 (预测值p, 真实标签y, 预期损失expected_loss, 预期梯度expected_dloss)
    cases = [
        (1.0, 1.0, 0.0, 0.0),
        (-1.0, -1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, -2.0),
        (-1.0, 1.0, 4.0, -4.0),
        (0.5, -1.0, 2.25, 3.0),
        (-2.0, 1.0, 8, -4.0),
        (-3.0, 1.0, 12, -4.0),
    ]
    # 调用通用的测试损失函数的函数，传入损失对象和测试用例
    _test_loss_common(loss, cases)


def test_loss_epsilon_insensitive():
    # Test EpsilonInsensitive
    # 创建一个 EpsilonInsensitive 的损失函数对象，epsilon 参数为 0.1
    loss = sgd_fast.EpsilonInsensitive(0.1)
    # 定义测试用例，每个元组包含 (预测值p, 真实标签y, 预期损失expected_loss, 预期梯度expected_dloss)
    cases = [
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0, 0.0),
        (-2.05, -2.0, 0.0, 0.0),
        (3.05, 3.0, 0.0, 0.0),
        (2.2, 2.0, 0.1, 1.0),
        (2.0, -1.0, 2.9, 1.0),
        (2.0, 2.2, 0.1, -1.0),
        (-2.0, 1.0, 2.9, -1.0),
    ]
    # 调用通用的测试损失函数的函数，传入损失对象和测试用例
    _test_loss_common(loss, cases)
    # Test SquaredEpsilonInsensitive
    # 创建一个 SquaredEpsilonInsensitive 的损失函数对象，设置容差为 0.1
    loss = sgd_fast.SquaredEpsilonInsensitive(0.1)
    # 定义测试用例列表
    cases = [
        # (p, y, expected_loss, expected_dloss)
        (0.0, 0.0, 0.0, 0.0),    # 测试点：p 和 y 都为 0，期望损失和损失导数都为 0
        (0.1, 0.0, 0.0, 0.0),    # 测试点：p 为 0.1，y 为 0，期望损失和损失导数都为 0
        (-2.05, -2.0, 0.0, 0.0), # 测试点：p 为 -2.05，y 为 -2.0，期望损失和损失导数都为 0
        (3.05, 3.0, 0.0, 0.0),   # 测试点：p 为 3.05，y 为 3.0，期望损失和损失导数都为 0
        (2.2, 2.0, 0.01, 0.2),   # 测试点：p 为 2.2，y 为 2.0，期望损失为 0.01，期望损失导数为 0.2
        (2.0, -1.0, 8.41, 5.8),  # 测试点：p 为 2.0，y 为 -1.0，期望损失为 8.41，期望损失导数为 5.8
        (2.0, 2.2, 0.01, -0.2),  # 测试点：p 为 2.0，y 为 2.2，期望损失为 0.01，期望损失导数为 -0.2
        (-2.0, 1.0, 8.41, -5.8), # 测试点：p 为 -2.0，y 为 1.0，期望损失为 8.41，期望损失导数为 -5.8
    ]
    # 使用通用的损失函数测试函数对损失函数对象和测试用例进行测试
    _test_loss_common(loss, cases)
def test_multi_thread_multi_class_and_early_stopping():
    # 这是一个非回归测试，用于检查早停止内部属性与基于线程的并行性之间的不良交互作用。

    # 创建一个SGDClassifier分类器对象，配置了alpha、tolerance、最大迭代次数、早停止设置、
    # 迭代无变化次数、随机种子和线程数等参数。
    clf = SGDClassifier(
        alpha=1e-3,
        tol=1e-3,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=100,
        random_state=0,
        n_jobs=2,
    )

    # 使用iris数据集训练分类器
    clf.fit(iris.data, iris.target)

    # 断言确保迭代次数大于迭代无变化次数
    assert clf.n_iter_ > clf.n_iter_no_change

    # 断言确保迭代次数小于迭代无变化次数加上一个阈值（20）
    assert clf.n_iter_ < clf.n_iter_no_change + 20

    # 断言确保分类器在iris数据集上的得分高于0.8
    assert clf.score(iris.data, iris.target) > 0.8


def test_multi_core_gridsearch_and_early_stopping():
    # 这是一个非回归测试，用于检查早停止内部属性与基于进程的多核并行性之间的不良交互作用。

    # 定义参数网格，包含alpha和迭代无变化次数的不同取值
    param_grid = {
        "alpha": np.logspace(-4, 4, 9),
        "n_iter_no_change": [5, 10, 50],
    }

    # 创建一个SGDClassifier分类器对象，配置了tolerance、最大迭代次数、早停止设置和随机种子
    clf = SGDClassifier(tol=1e-2, max_iter=1000, early_stopping=True, random_state=0)

    # 使用随机化搜索进行参数优化
    search = RandomizedSearchCV(clf, param_grid, n_iter=5, n_jobs=2, random_state=0)
    search.fit(iris.data, iris.target)

    # 断言确保搜索得到的最佳得分高于0.8
    assert search.best_score_ > 0.8


@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_SGDClassifier_fit_for_all_backends(backend):
    # 这是一个非回归测试。在多类别情况下，SGDClassifier.fit 使用joblib.Parallel
    # 按一对多方式拟合每个类别。每个OvA步骤会就地更新估计器的coef_属性。
    # 内部，SGDClassifier 使用require='sharedmem'调用Parallel。该测试确保
    # 即使用户请求不提供sharedmem语义的后端，SGDClassifier.fit 也能正常工作。

    # 在lokj或multiprocessing后端下，如果SGDClassifier.fit被调用，会使用memmap。
    # 在这种情况下，clf.coef_的就地修改将导致尝试写入只读内存映射缓冲区时导致段错误。

    random_state = np.random.RandomState(42)

    # 创建一个具有50000个特征和20个类别的分类问题。使用lokj或multiprocessing时，
    # 这会导致clf.coef_超过joblib和lokj（2018/11/1的阈值为1MB）中使用memmap的阈值。
    X = sp.random(500, 2000, density=0.02, format="csr", random_state=random_state)
    y = random_state.choice(20, 500)

    # 使用顺序方式拟合SGD分类器
    clf_sequential = SGDClassifier(max_iter=1000, n_jobs=1, random_state=42)
    clf_sequential.fit(X, y)

    # 使用指定的后端并行方式拟合SGDClassifier，并确保系数等于顺序拟合得到的系数
    clf_parallel = SGDClassifier(max_iter=1000, n_jobs=4, random_state=42)
    with joblib.parallel_backend(backend=backend):
        clf_parallel.fit(X, y)

    # 断言确保拟合得到的系数几乎相等
    assert_array_almost_equal(clf_sequential.coef_, clf_parallel.coef_)
@pytest.mark.parametrize(
    "Estimator", [linear_model.SGDClassifier, linear_model.SGDRegressor]
)
# 定义测试函数，参数化两个不同的 Estimator：SGDClassifier 和 SGDRegressor
def test_sgd_random_state(Estimator, global_random_seed):
    # Train the same model on the same data without converging and check that we
    # get reproducible results by fixing the random seed.
    # 在相同数据上训练相同模型，不收敛，并通过固定随机种子检查结果是否可复现。

    if Estimator == linear_model.SGDRegressor:
        # 如果 Estimator 是 SGDRegressor
        X, y = datasets.make_regression(random_state=global_random_seed)
    else:
        # 否则（Estimator 是 SGDClassifier）
        X, y = datasets.make_classification(random_state=global_random_seed)
    
    # Fitting twice a model with the same hyper-parameters on the same training
    # set with the same seed leads to the same results deterministically.
    # 使用相同超参数和种子在相同训练集上训练两次模型将确定性地得到相同结果。

    est = Estimator(random_state=global_random_seed, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_same_seed_a = est.fit(X, y).coef_
        assert est.n_iter_ == 1

    est = Estimator(random_state=global_random_seed, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_same_seed_b = est.fit(X, y).coef_
        assert est.n_iter_ == 1

    assert_allclose(coef_same_seed_a, coef_same_seed_b)

    # Fitting twice a model with the same hyper-parameters on the same training
    # set but with different random seed leads to different results after one
    # epoch because of the random shuffling of the dataset.
    # 在相同训练集上使用相同超参数但不同随机种子训练两次模型，由于数据集随机打乱，
    # 一轮迭代后会得到不同结果。

    est = Estimator(random_state=global_random_seed + 1, max_iter=1)
    with pytest.warns(ConvergenceWarning):
        coef_other_seed = est.fit(X, y).coef_
        assert est.n_iter_ == 1

    assert np.abs(coef_same_seed_a - coef_other_seed).max() > 1.0


def test_validation_mask_correctly_subsets(monkeypatch):
    """Test that data passed to validation callback correctly subsets.

    Non-regression test for #23255.
    """
    # 测试验证回调函数中传递的数据正确地进行子集选择。
    # 非回归测试 #23255。

    X, Y = iris.data, iris.target
    n_samples = X.shape[0]
    validation_fraction = 0.2
    clf = linear_model.SGDClassifier(
        early_stopping=True,
        tol=1e-3,
        max_iter=1000,
        validation_fraction=validation_fraction,
    )

    mock = Mock(side_effect=_stochastic_gradient._ValidationScoreCallback)
    monkeypatch.setattr(_stochastic_gradient, "_ValidationScoreCallback", mock)
    clf.fit(X, Y)

    X_val, y_val = mock.call_args[0][1:3]
    assert X_val.shape[0] == int(n_samples * validation_fraction)
    assert y_val.shape[0] == int(n_samples * validation_fraction)


def test_sgd_error_on_zero_validation_weight():
    # Test that SGDClassifier raises error when all the validation samples
    # have zero sample_weight. Non-regression test for #17229.
    # 测试 SGDClassifier 在所有验证样本的 sample_weight 都为零时是否引发错误。
    # 非回归测试 #17229。

    X, Y = iris.data, iris.target
    sample_weight = np.zeros_like(Y)
    validation_fraction = 0.4

    clf = linear_model.SGDClassifier(
        early_stopping=True, validation_fraction=validation_fraction, random_state=0
    )

    error_message = (
        "The sample weights for validation set are all zero, consider using a"
        " different random state."
    )
    # 使用 pytest 来验证是否抛出 ValueError 异常，并检查异常消息是否匹配指定的错误消息
    with pytest.raises(ValueError, match=error_message):
        # 调用分类器的 fit 方法，传入训练数据 X、标签 Y，以及样本权重 sample_weight
        clf.fit(X, Y, sample_weight=sample_weight)
# 使用 pytest 模块的 mark.parametrize 装饰器，定义一个测试函数 test_sgd_verbose，参数为 Estimator
@pytest.mark.parametrize("Estimator", [SGDClassifier, SGDRegressor])
def test_sgd_verbose(Estimator):
    """non-regression test for gh #25249"""
    # 创建一个 Estimator 实例，verbose 参数设置为 1，对数据 X, Y 进行拟合操作
    Estimator(verbose=1).fit(X, Y)


# 使用 pytest 模块的 mark.parametrize 装饰器，定义一个测试函数 test_sgd_dtype_match，参数为 SGDEstimator 和 data_type
@pytest.mark.parametrize(
    "SGDEstimator",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
@pytest.mark.parametrize("data_type", (np.float32, np.float64))
def test_sgd_dtype_match(SGDEstimator, data_type):
    # 将数据 X 转换为指定的 data_type 类型
    _X = X.astype(data_type)
    # 将数据 Y 转换为指定的 data_type 类型
    _Y = np.array(Y, dtype=data_type)
    # 创建一个 SGDEstimator 实例
    sgd_model = SGDEstimator()
    # 使用转换后的数据 _X, _Y 进行拟合操作
    sgd_model.fit(_X, _Y)
    # 断言拟合后的模型参数 coef_ 的数据类型与 data_type 一致
    assert sgd_model.coef_.dtype == data_type


# 使用 pytest 模块的 mark.parametrize 装饰器，定义一个测试函数 test_sgd_numerical_consistency，参数为 SGDEstimator
@pytest.mark.parametrize(
    "SGDEstimator",
    [
        SGDClassifier,
        SparseSGDClassifier,
        SGDRegressor,
        SparseSGDRegressor,
        SGDOneClassSVM,
        SparseSGDOneClassSVM,
    ],
)
def test_sgd_numerical_consistency(SGDEstimator):
    # 将数据 X 转换为 np.float64 类型
    X_64 = X.astype(dtype=np.float64)
    # 将数据 Y 转换为 np.float64 类型
    Y_64 = np.array(Y, dtype=np.float64)

    # 将数据 X 转换为 np.float32 类型
    X_32 = X.astype(dtype=np.float32)
    # 将数据 Y 转换为 np.float32 类型
    Y_32 = np.array(Y, dtype=np.float32)

    # 创建 SGDEstimator 实例，指定最大迭代次数为 20
    sgd_64 = SGDEstimator(max_iter=20)
    # 使用 np.float64 类型的数据进行拟合
    sgd_64.fit(X_64, Y_64)

    # 创建 SGDEstimator 实例，指定最大迭代次数为 20
    sgd_32 = SGDEstimator(max_iter=20)
    # 使用 np.float32 类型的数据进行拟合
    sgd_32.fit(X_32, Y_32)

    # 断言两个不同数据类型下拟合得到的模型参数 coef_ 在数值上一致
    assert_allclose(sgd_64.coef_, sgd_32.coef_)


# 使用 pytest 模块的 mark.parametrize 装饰器，定义一个测试函数 test_passive_aggressive_deprecated_average，参数为 Estimator
# TODO(1.7): remove
@pytest.mark.parametrize("Estimator", [SGDClassifier, SGDRegressor, SGDOneClassSVM])
def test_passive_aggressive_deprecated_average(Estimator):
    # 创建一个 Estimator 实例，设置 average 参数为 0
    est = Estimator(average=0)
    # 使用 pytest 的 warn 断言，期望捕获 FutureWarning，其中包含字符串 "average=0"
    with pytest.warns(FutureWarning, match="average=0"):
        # 对数据 X, Y 进行拟合操作
        est.fit(X, Y)
```