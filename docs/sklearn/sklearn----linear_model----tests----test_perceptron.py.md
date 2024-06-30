# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\tests\test_perceptron.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from sklearn.datasets import load_iris  # 从 scikit-learn 中导入 load_iris 数据集
from sklearn.linear_model import Perceptron  # 从 scikit-learn 中导入感知机模型
from sklearn.utils import check_random_state  # 从 scikit-learn 中导入随机状态检查函数
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal  # 导入测试函数用于比较数组
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入稀疏矩阵容器相关的修复

iris = load_iris()  # 载入 iris 数据集
random_state = check_random_state(12)  # 使用随机状态检查函数创建随机状态对象
indices = np.arange(iris.data.shape[0])  # 创建数据集索引数组
random_state.shuffle(indices)  # 打乱数据集索引顺序
X = iris.data[indices]  # 根据打乱后的索引重新排序特征数据
y = iris.target[indices]  # 根据打乱后的索引重新排序目标数据


class MyPerceptron:
    def __init__(self, n_iter=1):
        self.n_iter = n_iter  # 初始化感知机的迭代次数

    def fit(self, X, y):
        n_samples, n_features = X.shape  # 获取样本数和特征数
        self.w = np.zeros(n_features, dtype=np.float64)  # 初始化权重向量为零向量
        self.b = 0.0  # 初始化偏置为零

        for t in range(self.n_iter):  # 迭代感知机的次数
            for i in range(n_samples):  # 遍历每个样本
                if self.predict(X[i])[0] != y[i]:  # 如果预测错误
                    self.w += y[i] * X[i]  # 更新权重向量
                    self.b += y[i]  # 更新偏置

    def project(self, X):
        return np.dot(X, self.w) + self.b  # 返回投影结果

    def predict(self, X):
        X = np.atleast_2d(X)  # 将输入数据至少视为二维数组
        return np.sign(self.project(X))  # 返回投影结果的符号


@pytest.mark.parametrize("container", CSR_CONTAINERS + [np.array])
def test_perceptron_accuracy(container):
    data = container(X)  # 使用不同的容器封装数据
    clf = Perceptron(max_iter=100, tol=None, shuffle=False)  # 创建感知机分类器对象
    clf.fit(data, y)  # 训练分类器
    score = clf.score(data, y)  # 计算分类准确率
    assert score > 0.7  # 断言分类准确率大于 0.7


def test_perceptron_correctness():
    y_bin = y.copy()  # 拷贝目标数据
    y_bin[y != 1] = -1  # 将非1类标签设置为-1

    clf1 = MyPerceptron(n_iter=2)  # 创建自定义的感知机对象
    clf1.fit(X, y_bin)  # 训练自定义感知机

    clf2 = Perceptron(max_iter=2, shuffle=False, tol=None)  # 创建标准感知机对象
    clf2.fit(X, y_bin)  # 训练标准感知机

    assert_array_almost_equal(clf1.w, clf2.coef_.ravel())  # 断言两个模型的权重数组几乎相等


def test_undefined_methods():
    clf = Perceptron(max_iter=100)  # 创建感知机分类器对象
    for meth in ("predict_proba", "predict_log_proba"):  # 遍历未定义的预测方法
        with pytest.raises(AttributeError):  # 断言捕获属性错误异常
            getattr(clf, meth)  # 获取指定属性


def test_perceptron_l1_ratio():
    """Check that `l1_ratio` has an impact when `penalty='elasticnet'`"""
    clf1 = Perceptron(l1_ratio=0, penalty="elasticnet")  # 创建弹性网络正则化参数为0的感知机对象
    clf1.fit(X, y)  # 训练感知机

    clf2 = Perceptron(l1_ratio=0.15, penalty="elasticnet")  # 创建弹性网络正则化参数为0.15的感知机对象
    clf2.fit(X, y)  # 训练感知机

    assert clf1.score(X, y) != clf2.score(X, y)  # 断言两个模型的分类准确率不相等

    # 检查弹性网络的边界，根据 `l1_ratio` 值应对应于 L1 或 L2 正则化
    clf_l1 = Perceptron(penalty="l1").fit(X, y)  # 使用 L1 正则化训练感知机
    clf_elasticnet = Perceptron(l1_ratio=1, penalty="elasticnet").fit(X, y)  # 使用弹性网络正则化和 l1_ratio=1 训练感知机
    assert_allclose(clf_l1.coef_, clf_elasticnet.coef_)  # 断言两个模型的权重数组几乎相等

    clf_l2 = Perceptron(penalty="l2").fit(X, y)  # 使用 L2 正则化训练感知机
    clf_elasticnet = Perceptron(l1_ratio=0, penalty="elasticnet").fit(X, y)  # 使用弹性网络正则化和 l1_ratio=0 训练感知机
    assert_allclose(clf_l2.coef_, clf_elasticnet.coef_)  # 断言两个模型的权重数组几乎相等
```