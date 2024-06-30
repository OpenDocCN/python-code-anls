# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_linesearch.py`

```
"""
Tests for line search routines
"""
# 从 numpy.testing 导入多个断言方法，用于测试
from numpy.testing import (assert_equal, assert_array_almost_equal,
                           assert_array_almost_equal_nulp, assert_warns,
                           suppress_warnings)
# 导入 scipy.optimize._linesearch 模块，并捕获 LineSearchWarning 异常
import scipy.optimize._linesearch as ls
from scipy.optimize._linesearch import LineSearchWarning
# 导入 numpy 库并使用 np 别名
import numpy as np


def assert_wolfe(s, phi, derphi, c1=1e-4, c2=0.9, err_msg=""):
    """
    Check that strong Wolfe conditions apply
    
    检查强 Wolfe 条件是否成立
    """
    # 计算函数值和导数值
    phi1 = phi(s)
    phi0 = phi(0)
    derphi0 = derphi(0)
    derphi1 = derphi(s)
    # 构建消息字符串
    msg = (f"s = {s}; phi(0) = {phi0}; phi(s) = {phi1}; phi'(0) = {derphi0};"
           f" phi'(s) = {derphi1}; {err_msg}")
    # 断言 Wolfe 条件1是否满足
    assert phi1 <= phi0 + c1*s*derphi0, "Wolfe 1 failed: " + msg
    # 断言 Wolfe 条件2是否满足
    assert abs(derphi1) <= abs(c2*derphi0), "Wolfe 2 failed: " + msg


def assert_armijo(s, phi, c1=1e-4, err_msg=""):
    """
    Check that Armijo condition applies
    
    检查 Armijo 条件是否成立
    """
    # 计算函数值
    phi1 = phi(s)
    phi0 = phi(0)
    # 构建消息字符串
    msg = f"s = {s}; phi(0) = {phi0}; phi(s) = {phi1}; {err_msg}"
    # 断言 Armijo 条件是否满足
    assert phi1 <= (1 - c1*s)*phi0, msg


def assert_line_wolfe(x, p, s, f, fprime, **kw):
    # 断言强 Wolfe 条件是否满足
    assert_wolfe(s, phi=lambda sp: f(x + p*sp),
                 derphi=lambda sp: np.dot(fprime(x + p*sp), p), **kw)


def assert_line_armijo(x, p, s, f, **kw):
    # 断言 Armijo 条件是否满足
    assert_armijo(s, phi=lambda sp: f(x + p*sp), **kw)


def assert_fp_equal(x, y, err_msg="", nulp=50):
    """Assert two arrays are equal, up to some floating-point rounding error
    
    断言两个数组在浮点舍入误差范围内相等
    """
    try:
        # 调用 assert_array_almost_equal_nulp 函数进行比较
        assert_array_almost_equal_nulp(x, y, nulp)
    except AssertionError as e:
        # 如果断言失败，抛出带有详细错误消息的 AssertionError
        raise AssertionError(f"{e}\n{err_msg}") from e


class TestLineSearch:
    # -- scalar functions; must have dphi(0.) < 0
    def _scalar_func_1(self, s):  # skip name check
        # 增加函数调用计数器
        self.fcount += 1
        # 定义标量函数及其导数
        p = -s - s**3 + s**4
        dp = -1 - 3*s**2 + 4*s**3
        return p, dp

    def _scalar_func_2(self, s):  # skip name check
        # 增加函数调用计数器
        self.fcount += 1
        # 定义标量函数及其导数
        p = np.exp(-4*s) + s**2
        dp = -4*np.exp(-4*s) + 2*s
        return p, dp

    def _scalar_func_3(self, s):  # skip name check
        # 增加函数调用计数器
        self.fcount += 1
        # 定义标量函数及其导数
        p = -np.sin(10*s)
        dp = -10*np.cos(10*s)
        return p, dp

    # -- n-d functions

    def _line_func_1(self, x):  # skip name check
        # 增加函数调用计数器
        self.fcount += 1
        # 定义 n 维函数及其梯度
        f = np.dot(x, x)
        df = 2*x
        return f, df

    def _line_func_2(self, x):  # skip name check
        # 增加函数调用计数器
        self.fcount += 1
        # 定义 n 维函数及其梯度
        f = np.dot(x, np.dot(self.A, x)) + 1
        df = np.dot(self.A + self.A.T, x)
        return f, df

    # --
    # 设置测试方法的初始化状态
    def setup_method(self):
        # 初始化空列表，用于存储标量函数
        self.scalar_funcs = []
        # 初始化空列表，用于存储线性函数
        self.line_funcs = []
        # 设置变量 N 的初始值为 20
        self.N = 20
        # 初始化计数器 fcount 为 0
        self.fcount = 0

        # 定义一个内部函数 bind_index，用于绑定函数和索引
        def bind_index(func, idx):
            # 返回一个新的函数，该函数调用原始函数 func，并返回其结果的第 idx 个元素
            return lambda *a, **kw: func(*a, **kw)[idx]

        # 遍历当前对象的所有属性名称
        for name in sorted(dir(self)):
            # 如果属性名以 '_scalar_func_' 开头
            if name.startswith('_scalar_func_'):
                # 获取该属性的值（函数）
                value = getattr(self, name)
                # 将函数名和其绑定了 0 和 1 索引的新函数添加到标量函数列表中
                self.scalar_funcs.append(
                    (name, bind_index(value, 0), bind_index(value, 1)))
            # 如果属性名以 '_line_func_' 开头
            elif name.startswith('_line_func_'):
                # 获取该属性的值（函数）
                value = getattr(self, name)
                # 将函数名和其绑定了 0 和 1 索引的新函数添加到线性函数列表中
                self.line_funcs.append(
                    (name, bind_index(value, 0), bind_index(value, 1)))

        # 设置随机种子为 1234
        np.random.seed(1234)
        # 初始化一个 N*N 的随机数组
        self.A = np.random.randn(self.N, self.N)

    # 标量迭代器方法
    def scalar_iter(self):
        # 遍历标量函数列表中的每个函数及其索引
        for name, phi, derphi in self.scalar_funcs:
            # 对于每个函数，随机生成三个旧的初始值
            for old_phi0 in np.random.randn(3):
                # 生成器返回函数名、函数及其导数函数、旧的初始值
                yield name, phi, derphi, old_phi0

    # 线性迭代器方法
    def line_iter(self):
        # 遍历线性函数列表中的每个函数及其索引
        for name, f, fprime in self.line_funcs:
            k = 0
            # 循环直到 k 达到 9 次
            while k < 9:
                # 生成一个 N 维的随机数组 x 和 p
                x = np.random.randn(self.N)
                p = np.random.randn(self.N)
                # 如果 p 与 fprime(x) 的点积大于等于 0
                if np.dot(p, fprime(x)) >= 0:
                    # 继续循环以选择下降方向
                    continue
                k += 1
                # 生成一个随机的旧函数值 old_fv
                old_fv = float(np.random.randn())
                # 生成器返回函数名、函数及其导数函数、随机数组 x 和 p、旧的函数值
                yield name, f, fprime, x, p, old_fv

    # -- 通用标量搜索方法

    # 测试 Wolfe 算法的第一种标量搜索方法
    def test_scalar_search_wolfe1(self):
        c = 0
        # 遍历标量迭代器中的每个函数及其参数
        for name, phi, derphi, old_phi0 in self.scalar_iter():
            c += 1
            # 调用 Wolfe 算法的第一种标量搜索方法
            s, phi1, phi0 = ls.scalar_search_wolfe1(phi, derphi, phi(0),
                                                    old_phi0, derphi(0))
            # 断言初始函数值与搜索后的函数值相等
            assert_fp_equal(phi0, phi(0), name)
            # 断言搜索到的新函数值与 phi(s) 相等
            assert_fp_equal(phi1, phi(s), name)
            # 断言 Wolfe 条件成立
            assert_wolfe(s, phi, derphi, err_msg=name)

        # 断言至少进行了 3 次迭代
        assert c > 3  # check that the iterator really works...

    # 测试 Wolfe 算法的第二种标量搜索方法
    def test_scalar_search_wolfe2(self):
        # 遍历标量迭代器中的每个函数及其参数
        for name, phi, derphi, old_phi0 in self.scalar_iter():
            # 调用 Wolfe 算法的第二种标量搜索方法
            s, phi1, phi0, derphi1 = ls.scalar_search_wolfe2(
                phi, derphi, phi(0), old_phi0, derphi(0))
            # 断言初始函数值与搜索后的函数值相等
            assert_fp_equal(phi0, phi(0), name)
            # 断言搜索到的新函数值与 phi(s) 相等
            assert_fp_equal(phi1, phi(s), name)
            # 如果导数函数的值不为空，断言搜索到的新导数值与 derphi(s) 相等
            if derphi1 is not None:
                assert_fp_equal(derphi1, derphi(s), name)
            # 断言 Wolfe 条件成立
            assert_wolfe(s, phi, derphi, err_msg=f"{name} {old_phi0:g}")

    # 测试 Wolfe 算法的第二种标量搜索方法，带有低 amax 参数
    def test_scalar_search_wolfe2_with_low_amax(self):
        # 定义一个简单的标量函数 phi(alpha)
        def phi(alpha):
            return (alpha - 5) ** 2

        # 定义其导数函数 derphi(alpha)
        def derphi(alpha):
            return 2 * (alpha - 5)

        # 调用 Wolfe 算法的第二种标量搜索方法，限制 amax 为 0.001
        alpha_star, _, _, derphi_star = ls.scalar_search_wolfe2(phi, derphi, amax=0.001)
        # 断言未收敛时 alpha_star 和 derphi_star 为 None
        assert alpha_star is None  # Not converged
        assert derphi_star is None  # Not converged
    def test_scalar_search_wolfe2_regression(self):
        # Regression test for gh-12157
        # This phi has its minimum at alpha=4/3 ~ 1.333.
        # 定义一个函数 phi(alpha)，根据 alpha 的不同取值返回不同的数值
        def phi(alpha):
            if alpha < 1:
                return - 3*np.pi/2 * (alpha - 1)
            else:
                return np.cos(3*np.pi/2 * alpha - np.pi)

        # 定义函数 derphi(alpha)，根据 alpha 的不同取值返回不同的导数值
        def derphi(alpha):
            if alpha < 1:
                return - 3*np.pi/2
            else:
                return - 3*np.pi/2 * np.sin(3*np.pi/2 * alpha - np.pi)

        # 调用 ls.scalar_search_wolfe2 函数获取最小值 s
        s, _, _, _ = ls.scalar_search_wolfe2(phi, derphi)
        # 断言 s 的值小于 1.5
        assert s < 1.5

    def test_scalar_search_armijo(self):
        # 遍历 self.scalar_iter() 返回的迭代器，依次处理每个迭代元组 (name, phi, derphi, old_phi0)
        for name, phi, derphi, old_phi0 in self.scalar_iter():
            # 调用 ls.scalar_search_armijo 函数，获取最小值 s 和 phi(s)
            s, phi1 = ls.scalar_search_armijo(phi, phi(0), derphi(0))
            # 断言 phi1 等于 phi(s)，带上 name 进行错误消息提示
            assert_fp_equal(phi1, phi(s), name)
            # 断言 armijo 函数的返回值满足条件，带上错误消息和 old_phi0 值
            assert_armijo(s, phi, err_msg=f"{name} {old_phi0:g}")

    # -- Generic line searches

    def test_line_search_wolfe1(self):
        # 初始化变量 c 和 smax
        c = 0
        smax = 100
        # 遍历 self.line_iter() 返回的迭代器，依次处理每个迭代元组 (name, f, fprime, x, p, old_f)
        for name, f, fprime, x, p, old_f in self.line_iter():
            # 计算 f0 和 g0 的初始值
            f0 = f(x)
            g0 = fprime(x)
            # 初始化 self.fcount 为 0
            self.fcount = 0
            # 调用 ls.line_search_wolfe1 函数，获取最小值 s 和相关统计信息
            s, fc, gc, fv, ofv, gv = ls.line_search_wolfe1(f, fprime, x, p,
                                                           g0, f0, old_f,
                                                           amax=smax)
            # 断言 self.fcount 等于 fc+gc
            assert_equal(self.fcount, fc+gc)
            # 断言 ofv 等于 f(x)
            assert_fp_equal(ofv, f(x))
            # 如果 s 为 None，则继续下一轮迭代
            if s is None:
                continue
            # 断言 fv 等于 f(x + s*p)
            assert_fp_equal(fv, f(x + s*p))
            # 断言 gv 等于 fprime(x + s*p)，精度为 14 位小数
            assert_array_almost_equal(gv, fprime(x + s*p), decimal=14)
            # 如果 s 小于 smax，则增加 c 的计数
            if s < smax:
                c += 1
                # 断言 line_wolfe 函数返回的条件满足，带上错误消息 name
                assert_line_wolfe(x, p, s, f, fprime, err_msg=name)

        # 断言 c 大于 3，验证迭代器确实有效
        assert c > 3  # check that the iterator really works...
    # 定义测试函数 test_line_search_wolfe2
    def test_line_search_wolfe2(self):
        # 初始化计数器 c
        c = 0
        # 设置最大步长 smax
        smax = 512
        # 迭代 self.line_iter() 生成器产生的每组参数 name, f, fprime, x, p, old_f
        for name, f, fprime, x, p, old_f in self.line_iter():
            # 计算函数值 f0 和梯度值 g0
            f0 = f(x)
            g0 = fprime(x)
            # 重置 self.fcount 计数器
            self.fcount = 0
            # 捕获特定警告并忽略，设置警告过滤条件
            with suppress_warnings() as sup:
                sup.filter(LineSearchWarning,
                           "The line search algorithm could not find a solution")
                sup.filter(LineSearchWarning,
                           "The line search algorithm did not converge")
                # 调用 ls.line_search_wolfe2 执行线搜索算法
                s, fc, gc, fv, ofv, gv = ls.line_search_wolfe2(f, fprime, x, p,
                                                               g0, f0, old_f,
                                                               amax=smax)
            # 断言 self.fcount 与 fc+gc 的相等性
            assert_equal(self.fcount, fc+gc)
            # 断言 ofv 与 f(x) 的近似相等性
            assert_fp_equal(ofv, f(x))
            # 断言 fv 与 f(x + s*p) 的近似相等性
            assert_fp_equal(fv, f(x + s*p))
            # 如果 gv 不为 None，则断言 gv 与 fprime(x + s*p) 的近似相等性
            if gv is not None:
                assert_array_almost_equal(gv, fprime(x + s*p), decimal=14)
            # 如果 s 小于 smax，则增加计数器 c，并断言满足线搜索 Wolfe 条件
            if s < smax:
                c += 1
                assert_line_wolfe(x, p, s, f, fprime, err_msg=name)
        # 断言 c 大于 3，验证迭代器确实有效
        assert c > 3

    # 定义测试函数 test_line_search_wolfe2_bounds
    def test_line_search_wolfe2_bounds(self):
        # 为 GitHub 问题 #7475 设置测试函数
        # 定义函数 f(x) 和其导数 fp(x)
        def f(x):
            return np.dot(x, x)
        def fp(x):
            return 2 * x
        # 设置方向向量 p
        p = np.array([1, 0])
        # 设置起始点 x 和参数 c2
        x = -60 * p
        c2 = 0.5

        # 调用 ls.line_search_wolfe2 执行线搜索算法，返回最优步长 s
        s, _, _, _, _, _ = ls.line_search_wolfe2(f, fp, x, p, amax=30, c2=c2)
        # 断言满足 Wolfe 条件的线搜索
        assert_line_wolfe(x, p, s, f, fp)

        # 在警告条件下调用 ls.line_search_wolfe2 执行线搜索算法
        s, _, _, _, _, _ = assert_warns(LineSearchWarning,
                                        ls.line_search_wolfe2, f, fp, x, p,
                                        amax=29, c2=c2)
        # 断言未找到最优步长 s
        assert s is None

        # 在限制迭代次数条件下调用 ls.line_search_wolfe2 执行线搜索算法
        assert_warns(LineSearchWarning, ls.line_search_wolfe2, f, fp, x, p,
                     c2=c2, maxiter=5)

    # 定义测试函数 test_line_search_armijo
    def test_line_search_armijo(self):
        # 初始化计数器 c
        c = 0
        # 迭代 self.line_iter() 生成器产生的每组参数 name, f, fprime, x, p, old_f
        for name, f, fprime, x, p, old_f in self.line_iter():
            # 计算函数值 f0 和梯度值 g0
            f0 = f(x)
            g0 = fprime(x)
            # 重置 self.fcount 计数器
            self.fcount = 0
            # 调用 ls.line_search_armijo 执行 Armijo 线搜索算法
            s, fc, fv = ls.line_search_armijo(f, x, p, g0, f0)
            # 计数器 c 加 1
            c += 1
            # 断言 self.fcount 与 fc 的相等性
            assert_equal(self.fcount, fc)
            # 断言 fv 与 f(x + s*p) 的近似相等性
            assert_fp_equal(fv, f(x + s*p))
            # 断言满足 Armijo 条件的线搜索
            assert_line_armijo(x, p, s, f, err_msg=name)
        # 断言 c 大于等于 9，验证迭代器确实有效
        assert c >= 9
    def test_armijo_terminate_1(self):
        # Armijo should evaluate the function only once if the trial step
        # is already suitable

        # 计数器，用于记录函数评估次数
        count = [0]

        # 定义函数 phi(s)，用于 Armijo 搜索中评估目标函数
        def phi(s):
            # 每次调用 phi(s)，增加计数器
            count[0] += 1
            return -s + 0.01*s**2
        
        # 执行 Armijo 线搜索，查找合适的步长 s
        s, phi1 = ls.scalar_search_armijo(phi, phi(0), -1, alpha0=1)

        # 断言找到的步长 s 为 1
        assert_equal(s, 1)
        # 断言函数 phi 被评估了两次
        assert_equal(count[0], 2)
        # 断言 Armijo 条件满足
        assert_armijo(s, phi)

    def test_wolfe_terminate(self):
        # wolfe1 and wolfe2 should also evaluate the function only a few
        # times if the trial step is already suitable

        # 定义函数 phi(s)，用于 Wolfe 线搜索中评估目标函数
        def phi(s):
            # 每次调用 phi(s)，增加计数器
            count[0] += 1
            return -s + 0.05*s**2

        # 定义函数 derphi(s)，用于 Wolfe 线搜索中评估目标函数的导数
        def derphi(s):
            # 每次调用 derphi(s)，增加计数器
            count[0] += 1
            return -1 + 0.05*2*s

        # 遍历 Wolfe 线搜索的两种方法：wolfe1 和 wolfe2
        for func in [ls.scalar_search_wolfe1, ls.scalar_search_wolfe2]:
            # 每个方法开始时，重置计数器
            count = [0]
            # 执行 Wolfe 线搜索
            r = func(phi, derphi, phi(0), None, derphi(0))
            # 断言搜索结果不为空
            assert r[0] is not None, (r, func)
            # 断言函数 phi 和 derphi 的总评估次数不超过预期次数
            assert count[0] <= 2 + 2, (count, func)
            # 断言 Wolfe 条件满足
            assert_wolfe(r[0], phi, derphi, err_msg=str(func))
```