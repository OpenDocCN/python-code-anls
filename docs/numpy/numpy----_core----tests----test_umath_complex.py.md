# `.\numpy\numpy\_core\tests\test_umath_complex.py`

```
# 导入系统相关的模块
import sys
# 导入平台相关的模块
import platform
# 导入 pytest 测试框架
import pytest

# 导入 NumPy 库，并使用 np 别名
import numpy as np
# 直接导入 C 扩展模块，因为 _arg 未通过 umath 导出
import numpy._core._multiarray_umath as ncu
# 从 NumPy 测试模块中导入断言函数
from numpy.testing import (
    assert_raises, assert_equal, assert_array_equal, assert_almost_equal, assert_array_max_ulp
    )

# TODO: 分支切割（使用 Pauli 代码）
# TODO: conj '对称性'
# TODO: 浮点数处理单元（FPU）异常

# 在 Windows 平台上，许多复杂函数的结果与 C99 标准不符。参见票号 1574。
# 在 Solaris 平台上也是如此（票号 1642），以及 PowerPC 上的 OS X。
# FIXME: 当我们需要完全的 C99 兼容性时，这可能会改变
# 忽略 NumPy 的所有错误状态
with np.errstate(all='ignore'):
    # 检查复杂函数在特定条件下的表现是否不稳定
    functions_seem_flaky = ((np.exp(complex(np.inf, 0)).imag != 0)
                            or (np.log(complex(ncu.NZERO, 0)).imag != np.pi))
# 根据平台和复杂函数的稳定性，决定是否标记复杂数测试为失败
xfail_complex_tests = (not sys.platform.startswith('linux') or functions_seem_flaky)

# TODO: 当生成器函数被移除时，可以将此标记为 xfail
# 根据条件决定是否跳过测试，原因是复杂函数支持不足
platform_skip = pytest.mark.skipif(xfail_complex_tests,
                                   reason="Inadequate C99 complex support")


class TestCexp:
    def test_simple(self):
        # 定义一个用于检查复数值的函数
        check = check_complex_value
        # 定义一个函数变量 f，用于表示 np.exp 函数
        f = np.exp

        # 使用 check 函数检查 np.exp 的结果
        check(f, 1, 0, np.exp(1), 0, False)
        check(f, 0, 1, np.cos(1), np.sin(1), False)

        # 计算参考值 ref，并使用 check 函数检查 np.exp 的结果
        ref = np.exp(1) * complex(np.cos(1), np.sin(1))
        check(f, 1, 1, ref.real, ref.imag, False)

    # 使用 platform_skip 标记来决定是否跳过测试
    @platform_skip
    def test_special_values(self):
        # 根据 C99: Section G 6.3.1，测试特殊值的复数指数函数的行为

        # 设置检查函数为 check_complex_value
        check = check_complex_value
        # 设置函数 f 为 numpy 的指数函数 np.exp

        # 测试 cexp(+-0 + 0i)，预期结果为 1 + 0i
        check(f, ncu.PZERO, 0, 1, 0, False)
        check(f, ncu.NZERO, 0, 1, 0, False)

        # 测试 cexp(x + infi)，对于有限的 x，预期结果为 nan + nani，并引发无效的 FPU 异常
        check(f,  1, np.inf, np.nan, np.nan)
        check(f, -1, np.inf, np.nan, np.nan)
        check(f,  0, np.inf, np.nan, np.nan)

        # 测试 cexp(inf + 0i)，预期结果为 inf + 0i
        check(f,  np.inf, 0, np.inf, 0)

        # 测试 cexp(-inf + yi)，对于有限的 y，预期结果为 +0 + (cos(y) + i sin(y))
        check(f,  -np.inf, 1, ncu.PZERO, ncu.PZERO)
        check(f,  -np.inf, 0.75 * np.pi, ncu.NZERO, ncu.PZERO)

        # 测试 cexp(inf + yi)，对于有限的 y，预期结果为 +inf + (cos(y) + i sin(y))
        check(f,  np.inf, 1, np.inf, np.inf)
        check(f,  np.inf, 0.75 * np.pi, -np.inf, np.inf)

        # 测试 cexp(-inf + inf i)，预期结果为 +-0 +- 0i（符号未指定）
        def _check_ninf_inf(dummy):
            msgform = "cexp(-inf, inf) is (%f, %f), expected (+-0, +-0)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.inf)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_inf(None)

        # 测试 cexp(inf + inf i)，预期结果为 +-inf + NaNi，并引发无效的 FPU 异常
        def _check_inf_inf(dummy):
            msgform = "cexp(inf, inf) is (%f, %f), expected (+-inf, nan)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.inf)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_inf_inf(None)

        # 测试 cexp(-inf + nan i)，预期结果为 +-0 +- 0i
        def _check_ninf_nan(dummy):
            msgform = "cexp(-inf, nan) is (%f, %f), expected (+-0, +-0)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.nan)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_nan(None)

        # 测试 cexp(inf + nan i)，预期结果为 +-inf + nan
        def _check_inf_nan(dummy):
            msgform = "cexp(-inf, nan) is (%f, %f), expected (+-inf, nan)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.nan)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_inf_nan(None)

        # 测试 cexp(nan + yi)，对于 y != 0，预期结果为 nan + nani（可选：引发无效的 FPU 异常）
        check(f, np.nan, 1, np.nan, np.nan)
        check(f, np.nan, -1, np.nan, np.nan)

        check(f, np.nan,  np.inf, np.nan, np.nan)
        check(f, np.nan, -np.inf, np.nan, np.nan)

        # 测试 cexp(nan + nani)，预期结果为 nan + nani
        check(f, np.nan, np.nan, np.nan, np.nan)
    # TODO This can be xfail when the generator functions are got rid of.
    # 标记这个测试为跳过状态，并附带原因说明
    @pytest.mark.skip(reason="cexp(nan + 0I) is wrong on most platforms")
    # 定义一个测试函数，用于测试特殊值处理
    def test_special_values2(self):
        # XXX: most implementations get it wrong here (including glibc <= 2.10)
        # XXX: 大多数实现在这里处理错误（包括 glibc <= 2.10）
        # cexp(nan + 0i) 的结果应该是 nan + 0i
        # 定义一个函数别名用于检查复数值
        check = check_complex_value
        # 使用 NumPy 的指数函数
        f = np.exp

        # 调用检查函数，验证指数函数对于 nan + 0i 的处理是否正确
        check(f, np.nan, 0, np.nan, 0)
class TestClog:
    # 测试对复数数组的对数函数的正确性
    def test_simple(self):
        # 创建复数数组
        x = np.array([1+0j, 1+2j])
        # 计算绝对值的对数和幅角的复数对数
        y_r = np.log(np.abs(x)) + 1j * np.angle(x)
        # 计算复数数组的对数
        y = np.log(x)
        # 断言两种计算结果的近似相等性
        assert_almost_equal(y, y_r)

    @platform_skip
    # 根据平台跳过测试，条件是机器架构为 "armv5tel"，原因见 GitHub issue 413
    @pytest.mark.skipif(platform.machine() == "armv5tel", reason="See gh-413.")
class TestCsqrt:

    def test_simple(self):
        # 测试平方根函数对实数1的正确性
        # sqrt(1)
        check_complex_value(np.sqrt, 1, 0, 1, 0)

        # 测试平方根函数对虚数单位1i的正确性
        # sqrt(1i)
        rres = 0.5*np.sqrt(2)
        ires = rres
        check_complex_value(np.sqrt, 0, 1, rres, ires, False)

        # 测试平方根函数对负一的正确性
        # sqrt(-1)
        check_complex_value(np.sqrt, -1, 0, 0, 1)

    def test_simple_conjugate(self):
        # 测试共轭平方根函数对复数1+1i的正确性
        ref = np.conj(np.sqrt(complex(1, 1)))

        def f(z):
            return np.sqrt(np.conj(z))

        check_complex_value(f, 1, 1, ref.real, ref.imag, False)

    #def test_branch_cut(self):
    #    _check_branch_cut(f, -1, 0, 1, -1)

    @platform_skip
    # 根据平台跳过测试
    def test_special_values(self):
        # C99: Sec G 6.4.2

        check = check_complex_value
        f = np.sqrt

        # 测试特殊值，参考 C99 标准的 Section G 6.4.2
        # csqrt(+-0 + 0i) 结果为 0 + 0i
        check(f, ncu.PZERO, 0, 0, 0)
        check(f, ncu.NZERO, 0, 0, 0)

        # csqrt(x + infi) 结果为 inf + infi，对任何 x（包括 NaN）
        check(f,  1, np.inf, np.inf, np.inf)
        check(f, -1, np.inf, np.inf, np.inf)

        check(f, ncu.PZERO, np.inf, np.inf, np.inf)
        check(f, ncu.NZERO, np.inf, np.inf, np.inf)
        check(f,   np.inf, np.inf, np.inf, np.inf)
        check(f,  -np.inf, np.inf, np.inf, np.inf)
        check(f,  -np.nan, np.inf, np.inf, np.inf)

        # csqrt(x + nani) 结果为 nan + nani，对任何有限 x
        check(f,  1, np.nan, np.nan, np.nan)
        check(f, -1, np.nan, np.nan, np.nan)
        check(f,  0, np.nan, np.nan, np.nan)

        # csqrt(-inf + yi) 结果为 +0 + infi，对任何有限的 y > 0
        check(f, -np.inf, 1, ncu.PZERO, np.inf)

        # csqrt(inf + yi) 结果为 +inf + 0i，对任何有限的 y > 0
        check(f, np.inf, 1, np.inf, ncu.PZERO)

        # csqrt(-inf + nani) 结果为 nan +- infi（+i infi 都是有效的）
        def _check_ninf_nan(dummy):
            msgform = "csqrt(-inf, nan) is (%f, %f), expected (nan, +-inf)"
            z = np.sqrt(np.array(complex(-np.inf, np.nan)))
            #Fixme: ugly workaround for isinf bug.
            with np.errstate(invalid='ignore'):
                if not (np.isnan(z.real) and np.isinf(z.imag)):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_nan(None)

        # csqrt(+inf + nani) 结果为 inf + nani
        check(f, np.inf, np.nan, np.inf, np.nan)

        # csqrt(nan + yi) 结果为 nan + nani，对任何有限的 y（无限在 x + nani 处理）
        check(f, np.nan,       0, np.nan, np.nan)
        check(f, np.nan,       1, np.nan, np.nan)
        check(f, np.nan,  np.nan, np.nan, np.nan)

        # XXX: 检查 conj(csqrt(z)) == csqrt(conj(z)) 是否成立（需要先修复分支切割）
        # 需要先修复分支切割

class TestCpow:
    # 设置测试方法的准备阶段，用于忽略无效值错误
    def setup_method(self):
        # 保存当前的错误处理设置，并设置忽略无效值错误
        self.olderr = np.seterr(invalid='ignore')

    # 设置测试方法的清理阶段，恢复之前保存的错误处理设置
    def teardown_method(self):
        np.seterr(**self.olderr)

    # 测试简单情况下的复数幂运算
    def test_simple(self):
        # 定义输入数组 x 包含复数
        x = np.array([1+1j, 0+2j, 1+2j, np.inf, np.nan])
        # 计算 x 的平方
        y_r = x ** 2
        # 使用 np.power 函数计算 x 的平方
        y = np.power(x, 2)
        # 断言 y 和 y_r 几乎相等
        assert_almost_equal(y, y_r)

    # 测试标量和复数之间的幂运算
    def test_scalar(self):
        # 定义输入数组 x 和 y 包含标量和复数
        x = np.array([1, 1j, 2, 2.5+.37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5+1.5j, -0.5+1.5j, 2, 3])
        # 构建索引列表 lx
        lx = list(range(len(x)))

        # 预期的复数幂运算结果，由于问题 bpo-44698，需要硬编码这些值
        p_r = [
            1+0j,
            0.20787957635076193+0j,
            0.35812203996480685+0.6097119028618724j,
            0.12659112128185032+0.48847676699581527j,
            complex(np.inf, np.nan),
            complex(np.nan, np.nan),
        ]

        # 计算实际结果 n_r
        n_r = [x[i] ** y[i] for i in lx]
        # 遍历 lx 进行断言，确保 n_r[i] 与预期值 p_r[i] 几乎相等
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)

    # 测试数组之间的复数幂运算
    def test_array(self):
        # 定义输入数组 x 和 y 包含标量和复数
        x = np.array([1, 1j, 2, 2.5+.37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5+1.5j, -0.5+1.5j, 2, 3])
        # 构建索引列表 lx
        lx = list(range(len(x)))

        # 预期的复数幂运算结果，由于问题 bpo-44698，需要硬编码这些值
        p_r = [
            1+0j,
            0.20787957635076193+0j,
            0.35812203996480685+0.6097119028618724j,
            0.12659112128185032+0.48847676699581527j,
            complex(np.inf, np.nan),
            complex(np.nan, np.nan),
        ]

        # 计算实际结果 n_r
        n_r = x ** y
        # 遍历 lx 进行断言，确保 n_r[i] 与预期值 p_r[i] 几乎相等
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)
# 定义一个测试类 TestCabs，用于测试复数的绝对值函数 np.abs
class TestCabs:
    # 在每个测试方法执行前设置错误处理方式
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    # 在每个测试方法执行后恢复原始的错误处理方式
    def teardown_method(self):
        np.seterr(**self.olderr)

    # 测试 np.abs 函数对简单输入的计算结果是否正确
    def test_simple(self):
        # 定义输入数组 x 和期望输出数组 y_r
        x = np.array([1+1j, 0+2j, 1+2j, np.inf, np.nan])
        y_r = np.array([np.sqrt(2.), 2, np.sqrt(5), np.inf, np.nan])
        # 计算 np.abs(x) 的实际输出 y
        y = np.abs(x)
        # 断言计算结果与期望输出 y_r 几乎相等
        assert_almost_equal(y, y_r)

    # 测试 np.abs 函数对特定边界条件的处理
    def test_fabs(self):
        # 测试 np.abs(x +- 0j) 是否等于 np.abs(x)，这符合 C99 标准中对 cabs 的规定
        x = np.array([1+0j], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        # 测试 np.abs 函数对复数的特定情况处理
        x = np.array([complex(1, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(np.inf, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(np.nan, ncu.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

    # 测试 np.abs 函数在处理无穷大和 NaN 时的行为是否符合预期
    def test_cabs_inf_nan(self):
        x, y = [], []

        # 对于 cabs(+-nan + nani)，预期返回 nan
        x.append(np.nan)
        y.append(np.nan)
        check_real_value(np.abs,  np.nan, np.nan, np.nan)

        x.append(np.nan)
        y.append(-np.nan)
        check_real_value(np.abs, -np.nan, np.nan, np.nan)

        # 根据 C99 标准，如果实部或虚部中有一个是 inf，另一个是 nan，则 cabs 应返回 inf
        x.append(np.inf)
        y.append(np.nan)
        check_real_value(np.abs,  np.inf, np.nan, np.inf)

        x.append(-np.inf)
        y.append(np.nan)
        check_real_value(np.abs, -np.inf, np.nan, np.inf)

        # 对于 cabs(conj(z)) == conj(cabs(z))，即共轭复数的绝对值等于绝对值的共轭
        def f(a):
            return np.abs(np.conj(a))

        def g(a, b):
            return np.abs(complex(a, b))

        # 创建输入数组 xa，并对每对输入进行测试
        xa = np.array(x, dtype=complex)
        assert len(xa) == len(x) == len(y)
        for xi, yi in zip(x, y):
            ref = g(xi, yi)
            check_real_value(f, xi, yi, ref)

# 定义一个测试类 TestCarg，用于测试复数的辐角函数 ncu._arg
class TestCarg:
    # 测试 ncu._arg 函数对简单输入的计算结果是否正确
    def test_simple(self):
        check_real_value(ncu._arg, 1, 0, 0, False)
        check_real_value(ncu._arg, 0, 1, 0.5*np.pi, False)

        check_real_value(ncu._arg, 1, 1, 0.25*np.pi, False)
        check_real_value(ncu._arg, ncu.PZERO, ncu.PZERO, ncu.PZERO)

    # TODO 当生成器函数被移除时，这个测试应该标记为 xfail
    @pytest.mark.skip(
        reason="Complex arithmetic with signed zero fails on most platforms")
    def test_zero(self):
        # 测试 carg(-0 +- 0i)，返回值应为正负π
        check_real_value(ncu._arg, ncu.NZERO, ncu.PZERO,  np.pi, False)
        # 测试 carg(-0 - 0i)，返回值应为负π
        check_real_value(ncu._arg, ncu.NZERO, ncu.NZERO, -np.pi, False)

        # 测试 carg(+0 +- 0i)，返回值应为正负0
        check_real_value(ncu._arg, ncu.PZERO, ncu.PZERO, ncu.PZERO)
        # 测试 carg(+0 - 0i)，返回值应为正0
        check_real_value(ncu._arg, ncu.PZERO, ncu.NZERO, ncu.NZERO)

        # 测试 carg(x +- 0i) 当 x > 0 时，返回值应为正0
        check_real_value(ncu._arg, 1, ncu.PZERO, ncu.PZERO, False)
        # 测试 carg(x - 0i) 当 x > 0 时，返回值应为负0
        check_real_value(ncu._arg, 1, ncu.NZERO, ncu.NZERO, False)

        # 测试 carg(x +- 0i) 当 x < 0 时，返回值应为正负π
        check_real_value(ncu._arg, -1, ncu.PZERO,  np.pi, False)
        # 测试 carg(x - 0i) 当 x < 0 时，返回值应为负π
        check_real_value(ncu._arg, -1, ncu.NZERO, -np.pi, False)

        # 测试 carg(+- 0 + yi) 当 y > 0 时，返回值应为 π/2
        check_real_value(ncu._arg, ncu.PZERO, 1, 0.5 * np.pi, False)
        # 测试 carg(+- 0 + yi) 当 y > 0 时，返回值应为 π/2
        check_real_value(ncu._arg, ncu.NZERO, 1, 0.5 * np.pi, False)

        # 测试 carg(+- 0 + yi) 当 y < 0 时，返回值应为 -π/2
        check_real_value(ncu._arg, ncu.PZERO, -1, 0.5 * np.pi, False)
        # 测试 carg(+- 0 + yi) 当 y < 0 时，返回值应为 -π/2
        check_real_value(ncu._arg, ncu.NZERO, -1, -0.5 * np.pi, False)

    #def test_branch_cuts(self):
    #    _check_branch_cut(ncu._arg, -1, 1j, -1, 1)

    def test_special_values(self):
        # 测试 carg(-np.inf + yi) 当 y > 0 时，返回值应为 π
        check_real_value(ncu._arg, -np.inf,  1,  np.pi, False)
        # 测试 carg(-np.inf - yi) 当 y < 0 时，返回值应为 -π
        check_real_value(ncu._arg, -np.inf, -1, -np.pi, False)

        # 测试 carg(np.inf + yi) 当 y > 0 时，返回值应为 0
        check_real_value(ncu._arg, np.inf,  1, ncu.PZERO, False)
        # 测试 carg(np.inf - yi) 当 y < 0 时，返回值应为 -0
        check_real_value(ncu._arg, np.inf, -1, ncu.NZERO, False)

        # 测试 carg(x + np.infi) 当 x > 0 时，返回值应为 π/2
        check_real_value(ncu._arg, 1,  np.inf,  0.5 * np.pi, False)
        # 测试 carg(x - np.infi) 当 x < 0 时，返回值应为 -π/2
        check_real_value(ncu._arg, 1, -np.inf, -0.5 * np.pi, False)

        # 测试 carg(-np.inf + np.infi) 返回值应为 3π/4
        check_real_value(ncu._arg, -np.inf,  np.inf,  0.75 * np.pi, False)
        # 测试 carg(-np.inf - np.infi) 返回值应为 -3π/4
        check_real_value(ncu._arg, -np.inf, -np.inf, -0.75 * np.pi, False)

        # 测试 carg(np.inf + np.infi) 返回值应为 π/4
        check_real_value(ncu._arg, np.inf,  np.inf,  0.25 * np.pi, False)
        # 测试 carg(np.inf - np.infi) 返回值应为 -π/4
        check_real_value(ncu._arg, np.inf, -np.inf, -0.25 * np.pi, False)

        # 测试 carg(x + yi) 如果 x 或 y 是 nan，返回值应为 nan
        check_real_value(ncu._arg, np.nan,      0, np.nan, False)
        check_real_value(ncu._arg,      0, np.nan, np.nan, False)

        check_real_value(ncu._arg, np.nan, np.inf, np.nan, False)
        check_real_value(ncu._arg, np.inf, np.nan, np.nan, False)
# 定义函数，用于检查函数在复数输入时的输出值是否符合预期
def check_real_value(f, x1, y1, x, exact=True):
    # 创建一个包含单个复数的 NumPy 数组
    z1 = np.array([complex(x1, y1)])
    # 如果 exact 为 True，使用 assert_equal 断言确保函数 f 返回值等于 x
    if exact:
        assert_equal(f(z1), x)
    # 如果 exact 不为 True，使用 assert_almost_equal 断言确保函数 f 返回值接近于 x
    else:
        assert_almost_equal(f(z1), x)


# 定义函数，用于检查函数在复数输入时的输出值是否符合预期
def check_complex_value(f, x1, y1, x2, y2, exact=True):
    # 创建一个包含单个复数的 NumPy 数组
    z1 = np.array([complex(x1, y1)])
    # 创建目标复数 z2
    z2 = complex(x2, y2)
    # 忽略无效操作的错误，执行以下代码块
    with np.errstate(invalid='ignore'):
        # 如果 exact 为 True，使用 assert_equal 断言确保函数 f 返回值等于 z2
        if exact:
            assert_equal(f(z1), z2)
        # 如果 exact 不为 True，使用 assert_almost_equal 断言确保函数 f 返回值接近于 z2
        else:
            assert_almost_equal(f(z1), z2)


# 定义测试类 TestSpecialComplexAVX
class TestSpecialComplexAVX:
    # 使用 pytest.mark.parametrize 装饰器设置参数化测试
    @pytest.mark.parametrize("stride", [-4,-2,-1,1,2,4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    # 定义测试方法 test_array，用于测试复数数组处理
    def test_array(self, stride, astype):
        # 创建包含特定复数值的 NumPy 数组 arr
        arr = np.array([complex(np.nan , np.nan),
                        complex(np.nan , np.inf),
                        complex(np.inf , np.nan),
                        complex(np.inf , np.inf),
                        complex(0.     , np.inf),
                        complex(np.inf , 0.),
                        complex(0.     , 0.),
                        complex(0.     , np.nan),
                        complex(np.nan , 0.)], dtype=astype)
        # 创建目标数组 abs_true，存储 arr 对应的绝对值
        abs_true = np.array([np.nan, np.inf, np.inf, np.inf, np.inf, np.inf, 0., np.nan, np.nan], dtype=arr.real.dtype)
        # 创建目标数组 sq_true，存储 arr 对应的平方值
        sq_true = np.array([complex(np.nan,  np.nan),
                            complex(np.nan,  np.nan),
                            complex(np.nan,  np.nan),
                            complex(np.nan,  np.inf),
                            complex(-np.inf, np.nan),
                            complex(np.inf,  np.nan),
                            complex(0.,     0.),
                            complex(np.nan, np.nan),
                            complex(np.nan, np.nan)], dtype=astype)
        # 忽略无效操作的错误，执行以下代码块
        with np.errstate(invalid='ignore'):
            # 使用 assert_equal 断言确保 np.abs(arr[::stride]) 与 abs_true[::stride] 相等
            assert_equal(np.abs(arr[::stride]), abs_true[::stride])
            # 使用 assert_equal 断言确保 np.square(arr[::stride]) 与 sq_true[::stride] 相等


# 定义测试类 TestComplexAbsoluteAVX
class TestComplexAbsoluteAVX:
    # 使用 pytest.mark.parametrize 装饰器设置参数化测试
    @pytest.mark.parametrize("arraysize", [1,2,3,4,5,6,7,8,9,10,11,13,15,17,18,19])
    @pytest.mark.parametrize("stride", [-4,-3,-2,-1,1,2,3,4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    # 定义测试方法 test_array，用于测试复数数组的绝对值计算
    def test_array(self, arraysize, stride, astype):
        # 创建包含全为 1 的指定大小和数据类型的 NumPy 数组 arr
        arr = np.ones(arraysize, dtype=astype)
        # 创建目标数组 abs_true，存储 arr 的绝对值
        abs_true = np.ones(arraysize, dtype=arr.real.dtype)
        # 使用 assert_equal 断言确保 np.abs(arr[::stride]) 与 abs_true[::stride] 相等


# 测试用例从 https://github.com/numpy/numpy/issues/16660 中直接引用
# 定义测试类 TestComplexAbsoluteMixedDTypes
class TestComplexAbsoluteMixedDTypes:
    # 使用 pytest.mark.parametrize 装饰器设置参数化测试
    @pytest.mark.parametrize("stride", [-4,-3,-2,-1,1,2,3,4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("func", ['abs', 'square', 'conjugate'])
    # 定义一个测试函数，用于测试数组操作
    def test_array(self, stride, astype, func):
        # 定义一个结构化数据类型，包含各字段的名称和数据类型
        dtype = [('template_id', '<i8'), ('bank_chisq','<f4'),
                 ('bank_chisq_dof','<i8'), ('chisq', '<f4'), ('chisq_dof','<i8'),
                 ('cont_chisq', '<f4'), ('psd_var_val', '<f4'), ('sg_chisq','<f4'),
                 ('mycomplex', astype), ('time_index', '<i8')]
        # 创建一个 numpy 数组，使用指定的数据类型，包含多行数据
        vec = np.array([
               (0, 0., 0, -31.666483, 200, 0., 0.,  1.      ,  3.0+4.0j   ,  613090),
               (1, 0., 0, 260.91525 ,  42, 0., 0.,  1.      ,  5.0+12.0j  ,  787315),
               (1, 0., 0,  52.15155 ,  42, 0., 0.,  1.      ,  8.0+15.0j  ,  806641),
               (1, 0., 0,  52.430195,  42, 0., 0.,  1.      ,  7.0+24.0j  , 1363540),
               (2, 0., 0, 304.43646 ,  58, 0., 0.,  1.      ,  20.0+21.0j ,  787323),
               (3, 0., 0, 299.42108 ,  52, 0., 0.,  1.      ,  12.0+35.0j ,  787332),
               (4, 0., 0,  39.4836  ,  28, 0., 0.,  9.182192,  9.0+40.0j  ,  787304),
               (4, 0., 0,  76.83787 ,  28, 0., 0.,  1.      ,  28.0+45.0j, 1321869),
               (5, 0., 0, 143.26366 ,  24, 0., 0., 10.996129,  11.0+60.0j ,  787299)], dtype=dtype)
        # 获取指定的 numpy 函数，通过字符串 func 动态获取
        myfunc = getattr(np, func)
        # 提取数组中 'mycomplex' 列的数据
        a = vec['mycomplex']
        # 对提取的数据进行函数操作，使用指定的步长 stride
        g = myfunc(a[::stride])

        # 复制 'mycomplex' 列的数据到新的数组 b
        b = vec['mycomplex'].copy()
        # 对复制的数据进行函数操作，同样使用指定的步长 stride
        h = myfunc(b[::stride])

        # 断言操作的结果，比较两个数组的实部和虚部的最大ULP（单位最后一位）
        assert_array_max_ulp(h.real, g.real, 1)
        assert_array_max_ulp(h.imag, g.imag, 1)
```