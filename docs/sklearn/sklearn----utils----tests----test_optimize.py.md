# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_optimize.py`

```
# 导入必要的库
import numpy as np
import pytest
from scipy.optimize import fmin_ncg

# 导入所需的异常类和工具函数
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.optimize import _newton_cg


# 定义单元测试函数，测试 newton_cg 函数的正确性
def test_newton_cg():
    # 设置随机数生成器
    rng = np.random.RandomState(0)
    # 创建一个随机的 10x10 的矩阵 A
    A = rng.normal(size=(10, 10))
    # 初始值 x0 全为 1 的向量
    x0 = np.ones(10)

    # 定义函数 func(x)，计算 Ax 的平方的一半
    def func(x):
        Ax = A.dot(x)
        return 0.5 * (Ax).dot(Ax)

    # 定义梯度函数 grad(x)，计算 A.T.dot(A.dot(x))
    def grad(x):
        return A.T.dot(A.dot(x))

    # 定义黑塞矩阵向量积函数 hess(x, p)，计算 p.dot(A.T.dot(A.dot(x)))
    def hess(x, p):
        return p.dot(A.T.dot(A.dot(x.all())))

    # 定义梯度和黑塞矩阵向量积函数 grad_hess(x)
    def grad_hess(x):
        return grad(x), lambda x: A.T.dot(A.dot(x))

    # 断言 _newton_cg 和 scipy 中 fmin_ncg 的结果近似相等
    assert_array_almost_equal(
        _newton_cg(grad_hess, func, grad, x0, tol=1e-10)[0],
        fmin_ncg(f=func, x0=x0, fprime=grad, fhess_p=hess),
    )


# 使用 pytest 的参数化测试功能，测试 newton_cg 函数的不同 verbose 模式下的输出
@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_newton_cg_verbosity(capsys, verbose):
    """Test the std output of verbose newton_cg solver."""
    # 创建单位矩阵 A 和向量 b
    A = np.eye(2)
    b = np.array([1, 2], dtype=float)

    # 调用 _newton_cg 函数，并捕获输出
    _newton_cg(
        grad_hess=lambda x: (A @ x - b, lambda z: A @ z),
        func=lambda x: 0.5 * x @ A @ x - b @ x,
        grad=lambda x: A @ x - b,
        x0=np.zeros(A.shape[0]),
        verbose=verbose,
    )  # returns array([1., 2])
    captured = capsys.readouterr()

    # 根据 verbose 的值进行断言
    if verbose == 0:
        assert captured.out == ""
    else:
        # 断言输出中包含特定信息
        msg = [
            "Newton-CG iter = 1",
            "Check Convergence",
            "max |gradient|",
            "Solver did converge at loss = ",
        ]
        for m in msg:
            assert m in captured.out

    if verbose >= 2:
        # 断言输出中包含更详细的信息
        msg = [
            "Inner CG solver iteration 1 stopped with",
            "sum(|residuals|) <= tol",
            "Line Search",
            "try line search wolfe1",
            "wolfe1 line search was successful",
        ]
        for m in msg:
            assert m in captured.out
```