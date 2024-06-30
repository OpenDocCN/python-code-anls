# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_passive_aggressive.py`

```
# 导入必要的库
import numpy as np
import pytest

# 从sklearn库中导入需要使用的类和函数
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS

# 加载鸢尾花数据集
iris = load_iris()

# 初始化随机数生成器
random_state = check_random_state(12)

# 对样本索引进行随机重排
indices = np.arange(iris.data.shape[0])
random_state.shuffle(indices)

# 根据随机重排后的索引重新排列数据集
X = iris.data[indices]
y = iris.target[indices]

# 定义自定义的PassiveAggressive分类器类
class MyPassiveAggressive(ClassifierMixin):
    def __init__(
        self,
        C=1.0,
        epsilon=0.01,
        loss="hinge",
        fit_intercept=True,
        n_iter=1,
        random_state=None,
    ):
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter

    # 拟合方法，用于训练分类器
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)  # 初始化权重向量
        self.b = 0.0  # 初始化截距

        # 迭代训练过程
        for t in range(self.n_iter):
            for i in range(n_samples):
                p = self.project(X[i])  # 计算投影
                if self.loss in ("hinge", "squared_hinge"):
                    loss = max(1 - y[i] * p, 0)  # 计算损失
                else:
                    loss = max(np.abs(p - y[i]) - self.epsilon, 0)  # 计算损失

                sqnorm = np.dot(X[i], X[i])  # 计算样本向量的平方范数

                # 根据损失函数类型计算步长
                if self.loss in ("hinge", "epsilon_insensitive"):
                    step = min(self.C, loss / sqnorm)
                elif self.loss in ("squared_hinge", "squared_epsilon_insensitive"):
                    step = loss / (sqnorm + 1.0 / (2 * self.C))

                if self.loss in ("hinge", "squared_hinge"):
                    step *= y[i]
                else:
                    step *= np.sign(y[i] - p)

                # 更新权重向量和截距
                self.w += step * X[i]
                if self.fit_intercept:
                    self.b += step

    # 投影方法，用于计算投影
    def project(self, X):
        return np.dot(X, self.w) + self.b


# 使用pytest的参数化测试功能，对PassiveAggressiveClassifier进行测试
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
def test_classifier_accuracy(csr_container, fit_intercept, average):
    data = csr_container(X) if csr_container is not None else X
    # 初始化PassiveAggressiveClassifier对象
    clf = PassiveAggressiveClassifier(
        C=1.0,
        max_iter=30,
        fit_intercept=fit_intercept,
        random_state=1,
        average=average,
        tol=None,
    )
    clf.fit(data, y)  # 使用数据训练分类器
    score = clf.score(data, y)  # 计算分类器在训练集上的得分
    assert score > 0.79  # 断言分类器的得分大于0.79
    if average:
        assert hasattr(clf, "_average_coef")  # 断言分类器对象具有"_average_coef"属性
        assert hasattr(clf, "_average_intercept")  # 断言分类器对象具有"_average_intercept"属性
        assert hasattr(clf, "_standard_intercept")  # 断言分类器对象具有"_standard_intercept"属性
        assert hasattr(clf, "_standard_coef")  # 断言分类器对象具有"_standard_coef"属性


# 再次使用pytest的参数化测试功能，对PassiveAggressiveClassifier进行测试
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
def test_classifier_partial_fit(csr_container, average):
    # 获取唯一的类别数组
    classes = np.unique(y)
    # 根据是否使用稀疏矩阵容器，选择性地转换数据集
    data = csr_container(X) if csr_container is not None else X
    # 初始化被动攻击者分类器，设置随机种子和参数
    clf = PassiveAggressiveClassifier(random_state=0, average=average, max_iter=5)
    # 部分拟合分类器，迭代30次
    for t in range(30):
        clf.partial_fit(data, y, classes)
    # 计算分类器在训练数据上的准确率
    score = clf.score(data, y)
    # 断言准确率高于0.79
    assert score > 0.79
    # 如果选择了平均模式
    if average:
        # 断言分类器包含平均系数
        assert hasattr(clf, "_average_coef")
        # 断言分类器包含平均截距
        assert hasattr(clf, "_average_intercept")
        # 断言分类器包含标准截距
        assert hasattr(clf, "_standard_intercept")
        # 断言分类器包含标准系数
        assert hasattr(clf, "_standard_coef")


def test_classifier_refit():
    # 分类器可以在不同的标签和特征上重新训练。
    clf = PassiveAggressiveClassifier(max_iter=5).fit(X, y)
    # 断言分类器的类别数组等于y中的唯一值
    assert_array_equal(clf.classes_, np.unique(y))

    # 使用部分特征重新训练分类器
    clf.fit(X[:, :-1], iris.target_names[y])
    # 断言分类器的类别数组等于鸢尾花数据集的目标名
    assert_array_equal(clf.classes_, iris.target_names)


@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
@pytest.mark.parametrize("loss", ("hinge", "squared_hinge"))
def test_classifier_correctness(loss, csr_container):
    # 复制y数组并将非1类标签设为-1
    y_bin = y.copy()
    y_bin[y != 1] = -1

    # 使用自定义的被动攻击者分类器进行拟合
    clf1 = MyPassiveAggressive(loss=loss, n_iter=2)
    clf1.fit(X, y_bin)

    # 根据是否使用稀疏矩阵容器，选择性地转换数据集
    data = csr_container(X) if csr_container is not None else X
    # 使用sklearn的被动攻击者分类器进行拟合
    clf2 = PassiveAggressiveClassifier(loss=loss, max_iter=2, shuffle=False, tol=None)
    clf2.fit(data, y_bin)

    # 断言两个分类器的权重向量近似相等
    assert_array_almost_equal(clf1.w, clf2.coef_.ravel(), decimal=2)


@pytest.mark.parametrize(
    "response_method", ["predict_proba", "predict_log_proba", "transform"]
)
def test_classifier_undefined_methods(response_method):
    # 初始化被动攻击者分类器
    clf = PassiveAggressiveClassifier(max_iter=100)
    # 使用pytest断言捕获AttributeError异常
    with pytest.raises(AttributeError):
        getattr(clf, response_method)


def test_class_weights():
    # 测试类别权重
    X2 = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y2 = [1, 1, 1, -1, -1]

    # 初始化被动攻击者分类器，设置参数和随机种子
    clf = PassiveAggressiveClassifier(
        C=0.1, max_iter=100, class_weight=None, random_state=100
    )
    clf.fit(X2, y2)
    # 断言对[[0.2, -1.0]]点的预测结果为1
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([1]))

    # 给类别1分配一个小权重
    clf = PassiveAggressiveClassifier(
        C=0.1, max_iter=100, class_weight={1: 0.001}, random_state=100
    )
    clf.fit(X2, y2)

    # 此时超平面应该顺时针旋转，对该点的预测结果应该改变
    assert_array_equal(clf.predict([[0.2, -1.0]]), np.array([-1]))


def test_partial_fit_weight_class_balanced():
    # 不支持使用class_weight='balanced'进行部分拟合
    clf = PassiveAggressiveClassifier(class_weight="balanced", max_iter=100)
    with pytest.raises(ValueError):
        clf.partial_fit(X, y, classes=np.unique(y))


def test_equal_class_weight():
    X2 = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y2 = [0, 0, 1, 1]
    # 初始化被动攻击者分类器，设置参数和无效的class_weight
    clf = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight=None)
    clf.fit(X2, y2)

    # 已经是平衡的，所以"balanced"权重不应该产生影响
    # 创建一个PassiveAggressiveClassifier分类器，使用平衡的类权重（balanced）
    clf_balanced = PassiveAggressiveClassifier(C=0.1, tol=None, class_weight="balanced")
    # 使用数据集X2和标签y2来训练平衡类权重的分类器
    clf_balanced.fit(X2, y2)
    
    # 创建一个PassiveAggressiveClassifier分类器，使用自定义的类权重
    # 类0的权重为0.5，类1的权重为0.5
    clf_weighted = PassiveAggressiveClassifier(
        C=0.1, tol=None, class_weight={0: 0.5, 1: 0.5}
    )
    # 使用数据集X2和标签y2来训练自定义类权重的分类器
    clf_weighted.fit(X2, y2)
    
    # 由于学习率调度的影响，这两个分类器的coef_属性应该在一定范围内相似
    assert_almost_equal(clf.coef_, clf_weighted.coef_, decimal=2)
    # 断言：平衡类权重的分类器与未加权的分类器的coef_属性应该在一定范围内相似
    assert_almost_equal(clf.coef_, clf_balanced.coef_, decimal=2)
# 测试分类器在使用错误的类权重标签时是否会引发 ValueError 异常
def test_wrong_class_weight_label():
    # 创建包含样本数据和对应标签的 NumPy 数组
    X2 = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y2 = [1, 1, 1, -1, -1]

    # 初始化被测分类器对象，设定类权重为 {0: 0.5}，最大迭代次数为 100
    clf = PassiveAggressiveClassifier(class_weight={0: 0.5}, max_iter=100)
    # 使用 pytest 检查是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        clf.fit(X2, y2)


# 使用参数化测试验证回归器的均方误差是否小于 1.7
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
def test_regressor_mse(csr_container, fit_intercept, average):
    # 创建二元分类标签
    y_bin = y.copy()
    y_bin[y != 1] = -1

    # 根据是否存在 csr_container 来选择数据的容器类型
    data = csr_container(X) if csr_container is not None else X
    # 初始化被测回归器对象，设定参数，包括是否拟合截距、随机状态、是否进行平均处理、最大迭代次数等
    reg = PassiveAggressiveRegressor(
        C=1.0,
        fit_intercept=fit_intercept,
        random_state=0,
        average=average,
        max_iter=5,
    )
    # 对数据进行拟合
    reg.fit(data, y_bin)
    # 进行预测
    pred = reg.predict(data)
    # 验证均方误差是否小于 1.7
    assert np.mean((pred - y_bin) ** 2) < 1.7
    # 如果进行了平均处理，验证是否存在相关属性
    if average:
        assert hasattr(reg, "_average_coef")
        assert hasattr(reg, "_average_intercept")
        assert hasattr(reg, "_standard_intercept")
        assert hasattr(reg, "_standard_coef")


# 使用参数化测试验证回归器的部分拟合功能
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
def test_regressor_partial_fit(csr_container, average):
    # 创建二元分类标签
    y_bin = y.copy()
    y_bin[y != 1] = -1

    # 根据是否存在 csr_container 来选择数据的容器类型
    data = csr_container(X) if csr_container is not None else X
    # 初始化被测回归器对象，设定随机状态、是否进行平均处理、最大迭代次数等
    reg = PassiveAggressiveRegressor(random_state=0, average=average, max_iter=100)
    # 进行多次部分拟合
    for t in range(50):
        reg.partial_fit(data, y_bin)
    # 进行预测
    pred = reg.predict(data)
    # 验证均方误差是否小于 1.7
    assert np.mean((pred - y_bin) ** 2) < 1.7
    # 如果进行了平均处理，验证是否存在相关属性
    if average:
        assert hasattr(reg, "_average_coef")
        assert hasattr(reg, "_average_intercept")
        assert hasattr(reg, "_standard_intercept")
        assert hasattr(reg, "_standard_coef")


# 使用参数化测试验证回归器在不同容器和损失函数下的正确性
@pytest.mark.parametrize("csr_container", [None, *CSR_CONTAINERS])
@pytest.mark.parametrize("loss", ("epsilon_insensitive", "squared_epsilon_insensitive"))
def test_regressor_correctness(loss, csr_container):
    # 创建二元分类标签
    y_bin = y.copy()
    y_bin[y != 1] = -1

    # 使用自定义的 PassiveAggressive 类进行拟合
    reg1 = MyPassiveAggressive(loss=loss, n_iter=2)
    reg1.fit(X, y_bin)

    # 根据是否存在 csr_container 来选择数据的容器类型
    data = csr_container(X) if csr_container is not None else X
    # 初始化被测回归器对象，设定参数，包括公差、损失函数、最大迭代次数、是否洗牌等
    reg2 = PassiveAggressiveRegressor(tol=None, loss=loss, max_iter=2, shuffle=False)
    # 对数据进行拟合
    reg2.fit(data, y_bin)

    # 验证两个回归器的权重向量是否几乎相等
    assert_array_almost_equal(reg1.w, reg2.coef_.ravel(), decimal=2)


# 测试回归器是否能捕获转换方法未定义的属性错误
def test_regressor_undefined_methods():
    # 初始化被测回归器对象，设定最大迭代次数
    reg = PassiveAggressiveRegressor(max_iter=100)
    # 使用 pytest 检查是否会引发 AttributeError 异常
    with pytest.raises(AttributeError):
        reg.transform(X)


# 使用参数化测试验证被弃用的 average 参数是否会引发警告
@pytest.mark.parametrize(
    "Estimator", [PassiveAggressiveClassifier, PassiveAggressiveRegressor]
)
def test_passive_aggressive_deprecated_average(Estimator):
    # 初始化被测估计器对象，设定 average 参数为 0
    est = Estimator(average=0)
    # 使用 pytest 检查是否会引发 FutureWarning 并包含 "average=0" 的警告
    with pytest.warns(FutureWarning, match="average=0"):
        est.fit(X, y)
```