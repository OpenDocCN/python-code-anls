# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_kernel_ridge.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from sklearn.datasets import make_regression  # 导入 make_regression 函数，用于生成回归模拟数据集
from sklearn.kernel_ridge import KernelRidge  # 导入 KernelRidge 类，用于核岭回归
from sklearn.linear_model import Ridge  # 导入 Ridge 类，用于岭回归
from sklearn.metrics.pairwise import pairwise_kernels  # 导入 pairwise_kernels 函数，用于计算核函数
from sklearn.utils._testing import assert_array_almost_equal, ignore_warnings  # 导入测试相关的函数和装饰器
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS  # 导入容器修复函数

X, y = make_regression(n_features=10, random_state=0)  # 生成具有10个特征的回归模拟数据集 X 和对应的目标值 y
Y = np.array([y, y]).T  # 构造一个多输出的目标矩阵 Y

def test_kernel_ridge():
    # 使用 Ridge 模型进行岭回归预测
    pred = Ridge(alpha=1, fit_intercept=False).fit(X, y).predict(X)
    # 使用 KernelRidge 模型进行核岭回归预测
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X, y).predict(X)
    assert_array_almost_equal(pred, pred2)  # 断言预测结果应当相似

@pytest.mark.parametrize("sparse_container", [*CSR_CONTAINERS, *CSC_CONTAINERS])
def test_kernel_ridge_sparse(sparse_container):
    X_sparse = sparse_container(X)  # 将数据 X 转换为稀疏格式
    # 使用 Ridge 模型进行岭回归预测（处理稀疏数据）
    pred = (
        Ridge(alpha=1, fit_intercept=False, solver="cholesky")
        .fit(X_sparse, y)
        .predict(X_sparse)
    )
    # 使用 KernelRidge 模型进行核岭回归预测（处理稀疏数据）
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X_sparse, y).predict(X_sparse)
    assert_array_almost_equal(pred, pred2)  # 断言预测结果应当相似

def test_kernel_ridge_singular_kernel():
    # 当 alpha=0 时，计算双重系数会导致 LinAlgError，从而回退到 lstsq 求解器。这里进行测试。
    pred = Ridge(alpha=0, fit_intercept=False).fit(X, y).predict(X)
    kr = KernelRidge(kernel="linear", alpha=0)
    ignore_warnings(kr.fit)(X, y)  # 忽略警告，执行拟合操作
    pred2 = kr.predict(X)
    assert_array_almost_equal(pred, pred2)  # 断言预测结果应当相似

def test_kernel_ridge_precomputed():
    for kernel in ["linear", "rbf", "poly", "cosine"]:
        K = pairwise_kernels(X, X, metric=kernel)  # 计算核矩阵 K
        # 使用 KernelRidge 模型进行预测（核预计算）
        pred = KernelRidge(kernel=kernel).fit(X, y).predict(X)
        pred2 = KernelRidge(kernel="precomputed").fit(K, y).predict(K)
        assert_array_almost_equal(pred, pred2)  # 断言预测结果应当相似

def test_kernel_ridge_precomputed_kernel_unchanged():
    K = np.dot(X, X.T)  # 计算核矩阵 K
    K2 = K.copy()
    KernelRidge(kernel="precomputed").fit(K, y)  # 使用核预计算模型拟合
    assert_array_almost_equal(K, K2)  # 断言核矩阵 K 未被改变

def test_kernel_ridge_sample_weights():
    K = np.dot(X, X.T)  # 计算预计算核
    sw = np.random.RandomState(0).rand(X.shape[0])  # 创建样本权重

    # 使用 Ridge 模型进行带样本权重的岭回归预测
    pred = Ridge(alpha=1, fit_intercept=False).fit(X, y, sample_weight=sw).predict(X)
    # 使用 KernelRidge 模型进行带样本权重的核岭回归预测
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X, y, sample_weight=sw).predict(X)
    # 使用 KernelRidge 模型进行带样本权重和预计算核的核岭回归预测
    pred3 = (
        KernelRidge(kernel="precomputed", alpha=1)
        .fit(K, y, sample_weight=sw)
        .predict(K)
    )
    assert_array_almost_equal(pred, pred2)  # 断言预测结果应当相似
    assert_array_almost_equal(pred, pred3)  # 断言预测结果应当相似

def test_kernel_ridge_multi_output():
    pred = Ridge(alpha=1, fit_intercept=False).fit(X, Y).predict(X)
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X, Y).predict(X)
    assert_array_almost_equal(pred, pred2)  # 断言预测结果应当相似

    pred3 = KernelRidge(kernel="linear", alpha=1).fit(X, y).predict(X)
    pred3 = np.array([pred3, pred3]).T
    assert_array_almost_equal(pred2, pred3)  # 断言预测结果应当相似
```