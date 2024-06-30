# `D:\src\scipysrc\scipy\scipy\optimize\tests\test__root.py`

```
"""
Unit tests for optimization routines from _root.py.
"""
# 导入必要的测试工具和库
from numpy.testing import assert_, assert_equal
import pytest
from pytest import raises as assert_raises, warns as assert_warns
import numpy as np

# 导入 scipy.optimize 中的 root 函数
from scipy.optimize import root

# 定义测试类 TestRoot，用于测试 root 函数的不同参数和方法
class TestRoot:

    # 测试 tol 参数是否有效
    def test_tol_parameter(self):
        # 定义一个测试函数 func，返回方程组的解
        def func(z):
            x, y = z
            return np.array([x**3 - 1, y**3 - 1])

        # 定义 func 的雅可比矩阵函数 dfunc
        def dfunc(z):
            x, y = z
            return np.array([[3*x**2, 0], [0, 3*y**2]])

        # 遍历不同的求解方法
        for method in ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson',
                       'diagbroyden', 'krylov']:
            # 对于不收敛的方法，跳过测试
            if method in ('linearmixing', 'excitingmixing'):
                continue

            # 根据方法选择是否使用雅可比矩阵
            if method in ('hybr', 'lm'):
                jac = dfunc
            else:
                jac = None

            # 使用不同的 tol 值进行求解，比较结果
            sol1 = root(func, [1.1,1.1], jac=jac, tol=1e-4, method=method)
            sol2 = root(func, [1.1,1.1], jac=jac, tol=0.5, method=method)
            msg = f"{method}: {func(sol1.x)} vs. {func(sol2.x)}"
            # 断言求解成功
            assert_(sol1.success, msg)
            assert_(sol2.success, msg)
            # 断言使用更小 tol 的解具有更小的函数值
            assert_(abs(func(sol1.x)).max() < abs(func(sol2.x)).max(),
                    msg)

    # 测试 tol_norm 参数
    def test_tol_norm(self):

        # 定义一个简单的范数函数 norm
        def norm(x):
            return abs(x[0])

        # 遍历不同的求解方法，测试 tol_norm 参数
        for method in ['excitingmixing',
                       'diagbroyden',
                       'linearmixing',
                       'anderson',
                       'broyden1',
                       'broyden2',
                       'krylov']:
            # 调用 root 函数测试 tol_norm 参数是否有效
            root(np.zeros_like, np.zeros(2), method=method,
                options={"tol_norm": norm})

    # 测试 minimize_scalar_coerce_args_param 参数
    def test_minimize_scalar_coerce_args_param(self):
        # github issue #3503
        # 定义一个包含默认参数的测试函数 func
        def func(z, f=1):
            x, y = z
            return np.array([x**3 - 1, y**3 - f])

        # 调用 root 函数测试 args 参数
        root(func, [1.1, 1.1], args=1.5)

    # 测试 f_size 参数
    def test_f_size(self):
        # gh8320
        # 创建一个类 fun，用于测试返回数组大小变化时的错误处理
        class fun:
            def __init__(self):
                self.count = 0

            def __call__(self, x):
                self.count += 1

                if not (self.count % 5):
                    ret = x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0
                else:
                    ret = ([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0,
                           0.5 * (x[1] - x[0]) ** 3 + x[1]])

                return ret

        # 创建 fun 类的实例 F
        F = fun()
        # 使用 assert_raises 断言函数调用时引发 ValueError
        with assert_raises(ValueError):
            root(F, [0.1, 0.0], method='lm')
    # 定义一个测试函数，用于测试 GitHub 问题 #10370 报告的问题
    def test_gh_10370(self):
        # 定义一个函数 `fun`，接受 `x` 和 `ignored` 参数，返回一个列表
        def fun(x, ignored):
            return [3*x[0] - 0.25*x[1]**2 + 10, 0.1*x[0]**2 + 5*x[1] - 2]

        # 定义一个函数 `grad`，接受 `x` 和 `ignored` 参数，返回一个二维列表
        def grad(x, ignored):
            return [[3, 0.5 * x[1]], [0.2 * x[0], 5]]

        # 定义一个函数 `fun_grad`，调用 `fun` 和 `grad` 函数，返回它们的结果
        def fun_grad(x, ignored):
            return fun(x, ignored), grad(x, ignored)

        # 初始化 `x0` 为一个包含两个零元素的 NumPy 数组
        x0 = np.zeros(2)

        # 调用 `root` 函数，使用 `fun` 函数和初始值 `x0`，并传入参数 `(1,)` 和方法 `'krylov'`
        ref = root(fun, x0, args=(1,), method='krylov')

        # 设置警告信息字符串
        message = 'Method krylov does not use the jacobian'

        # 使用 `assert_warns` 检查是否有 RuntimeWarning 警告，并匹配 `message` 字符串
        with assert_warns(RuntimeWarning, match=message):
            # 第一次调用 `root` 函数，使用 `fun` 函数、初始值 `x0` 和参数 `(1,)`，并使用 `jac=grad` 参数
            res1 = root(fun, x0, args=(1,), method='krylov', jac=grad)
        
        # 使用 `assert_warns` 检查是否有 RuntimeWarning 警告，并匹配 `message` 字符串
        with assert_warns(RuntimeWarning, match=message):
            # 第二次调用 `root` 函数，使用 `fun_grad` 函数、初始值 `x0` 和参数 `(1,)`，并使用 `jac=True` 参数
            res2 = root(fun_grad, x0, args=(1,), method='krylov', jac=True)

        # 使用 `assert_equal` 检查 `res1.x` 和 `ref.x` 是否相等
        assert_equal(res1.x, ref.x)
        # 使用 `assert_equal` 检查 `res2.x` 和 `ref.x` 是否相等
        assert_equal(res2.x, ref.x)
        # 使用 `assert` 检查 `res1.success`、`res2.success` 和 `ref.success` 是否都为 True
        assert res1.success is res2.success is ref.success is True
    
    # 使用 `pytest.mark.parametrize` 装饰器定义一个参数化测试函数，测试不同的方法
    @pytest.mark.parametrize("method", ["hybr", "lm", "broyden1", "broyden2",
                                        "anderson", "linearmixing",
                                        "diagbroyden", "excitingmixing",
                                        "krylov", "df-sane"])
    # 定义一个测试函数，接受 `method` 参数
    def test_method_in_result(self, method):
        # 定义一个简单的函数 `func`，接受 `x` 参数，返回 `x - 1`
        def func(x):
            return x - 1
        
        # 调用 `root` 函数，使用 `func` 函数、初始值 `x0=[1]` 和指定的 `method` 参数
        res = root(func, x0=[1], method=method)

        # 使用 `assert` 检查 `res.method` 是否等于当前测试的 `method`
        assert res.method == method
```