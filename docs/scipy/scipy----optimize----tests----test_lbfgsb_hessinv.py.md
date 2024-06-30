# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_lbfgsb_hessinv.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_allclose  # 导入 NumPy 测试模块中的 assert_allclose 函数
import scipy.linalg  # 导入 SciPy 线性代数模块
from scipy.optimize import minimize  # 导入 SciPy 优化模块中的 minimize 函数


def test_1():
    def f(x):
        return x**4, 4*x**3  # 定义函数 f(x)，返回 x^4 和其导数 4x^3

    for gtol in [1e-8, 1e-12, 1e-20]:  # 循环不同的 gtol 值
        for maxcor in range(20, 35):  # 循环不同的 maxcor 值，范围从 20 到 34
            result = minimize(fun=f, jac=True, method='L-BFGS-B', x0=20,
                options={'gtol': gtol, 'maxcor': maxcor})  # 调用 minimize 函数进行优化

            H1 = result.hess_inv(np.array([1])).reshape(1,1)  # 计算 Hessian 逆的值，并重塑为 1x1 的数组
            H2 = result.hess_inv.todense()  # 获取 Hessian 逆的密集表示

            assert_allclose(H1, H2)  # 使用 assert_allclose 检查 H1 和 H2 是否近似相等


def test_2():
    H0 = [[3, 0], [1, 2]]  # 定义一个矩阵 H0

    def f(x):
        return np.dot(x, np.dot(scipy.linalg.inv(H0), x))  # 定义函数 f(x)，计算 x^T * H0^(-1) * x

    result1 = minimize(fun=f, method='L-BFGS-B', x0=[10, 20])  # 调用 minimize 函数进行优化
    result2 = minimize(fun=f, method='BFGS', x0=[10, 20])  # 使用不同的优化方法进行优化

    H1 = result1.hess_inv.todense()  # 获取 result1 的 Hessian 逆的密集表示

    H2 = np.vstack((
        result1.hess_inv(np.array([1, 0])),  # 计算 Hessian 逆在 [1, 0] 处的值
        result1.hess_inv(np.array([0, 1]))))  # 计算 Hessian 逆在 [0, 1] 处的值

    assert_allclose(
        result1.hess_inv(np.array([1, 0]).reshape(2,1)).reshape(-1),
        result1.hess_inv(np.array([1, 0])))  # 使用 assert_allclose 检查两种方式计算的结果是否相等
    assert_allclose(H1, H2)  # 使用 assert_allclose 检查 H1 和 H2 是否近似相等
    assert_allclose(H1, result2.hess_inv, rtol=1e-2, atol=0.03)  # 使用 assert_allclose 检查 H1 和 result2 的 Hessian 逆是否近似相等
```