# `D:\src\scipysrc\scipy\scipy\integrate\tests\test_quadpack.py`

```
# 导入必要的模块和库
import sys  # 导入系统模块
import math  # 导入数学函数模块
import numpy as np  # 导入 NumPy 库，并重命名为 np
from numpy import sqrt, cos, sin, arctan, exp, log, pi  # 从 NumPy 导入特定函数
from numpy.testing import (assert_,  # 导入 NumPy 测试模块中的函数
        assert_allclose, assert_array_less, assert_almost_equal)
import pytest  # 导入 pytest 测试框架

# 导入 SciPy 中的积分和特殊函数模块
from scipy.integrate import quad, dblquad, tplquad, nquad
from scipy.special import erf, erfc
from scipy._lib._ccallback import LowLevelCallable

# 导入 ctypes 库和相关函数
import ctypes
import ctypes.util
from scipy._lib._ccallback_c import sine_ctypes

# 导入测试模块
import scipy.integrate._test_multivariate as clib_test


# 自定义函数，用于测试积分的精度
def assert_quad(value_and_err, tabled_value, error_tolerance=1.5e-8):
    value, err = value_and_err
    # 断言计算的值接近预期值
    assert_allclose(value, tabled_value, atol=err, rtol=0)
    if error_tolerance is not None:
        # 断言误差在可接受范围内
        assert_array_less(err, error_tolerance)


# 获取 C 库中的测试函数
def get_clib_test_routine(name, restype, *argtypes):
    ptr = getattr(clib_test, name)
    return ctypes.cast(ptr, ctypes.CFUNCTYPE(restype, *argtypes))


# 测试类，用于测试 ctypes 库中的函数
class TestCtypesQuad:
    def setup_method(self):
        # 根据平台选择相应的数学库文件
        if sys.platform == 'win32':
            files = ['api-ms-win-crt-math-l1-1-0.dll']
        elif sys.platform == 'darwin':
            files = ['libm.dylib']
        else:
            files = ['libm.so', 'libm.so.6']

        for file in files:
            try:
                # 尝试加载库文件
                self.lib = ctypes.CDLL(file)
                break
            except OSError:
                pass
        else:
            # 如果加载失败，跳过测试
            pytest.skip("Ctypes can't import libm.so")

        # 设置函数的返回类型和参数类型
        restype = ctypes.c_double
        argtypes = (ctypes.c_double,)
        for name in ['sin', 'cos', 'tan']:
            func = getattr(self.lib, name)
            func.restype = restype
            func.argtypes = argtypes

    def test_typical(self):
        # 测试 ctypes 库中的三角函数与标准库中函数的积分结果
        assert_quad(quad(self.lib.sin, 0, 5), quad(math.sin, 0, 5)[0])
        assert_quad(quad(self.lib.cos, 0, 5), quad(math.cos, 0, 5)[0])
        assert_quad(quad(self.lib.tan, 0, 1), quad(math.tan, 0, 1)[0])

    def test_ctypes_sine(self):
        # 测试 ctypes 库中的低级回调函数的积分
        quad(LowLevelCallable(sine_ctypes), 0, 1)
    # 定义测试函数 test_ctypes_variants，用于测试不同的 ctypes 变体

    # 获取名为 '_sin_0' 的 C 库测试函数，接受一个双精度浮点数参数，返回一个双精度浮点数，第三个参数为 void 指针
    sin_0 = get_clib_test_routine('_sin_0', ctypes.c_double,
                                  ctypes.c_double, ctypes.c_void_p)

    # 获取名为 '_sin_1' 的 C 库测试函数，接受一个双精度浮点数和一个整数参数，返回一个双精度浮点数，第三个参数为指向双精度浮点数的指针，第四个参数为 void 指针
    sin_1 = get_clib_test_routine('_sin_1', ctypes.c_double,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                  ctypes.c_void_p)

    # 获取名为 '_sin_2' 的 C 库测试函数，接受一个双精度浮点数参数，返回一个双精度浮点数
    sin_2 = get_clib_test_routine('_sin_2', ctypes.c_double,
                                  ctypes.c_double)

    # 获取名为 '_sin_3' 的 C 库测试函数，接受一个双精度浮点数和一个整数参数，返回一个双精度浮点数，第三个参数为指向双精度浮点数的指针
    sin_3 = get_clib_test_routine('_sin_3', ctypes.c_double,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # 获取名为 '_sin_3' 的 C 库测试函数，接受一个双精度浮点数和一个整数参数，返回一个双精度浮点数
    sin_4 = get_clib_test_routine('_sin_3', ctypes.c_double,
                                  ctypes.c_int, ctypes.c_double)

    # 将所有测试函数的签名存入列表 all_sigs
    all_sigs = [sin_0, sin_1, sin_2, sin_3, sin_4]

    # 将传统签名的测试函数存入列表 legacy_sigs
    legacy_sigs = [sin_2, sin_4]

    # 将仅传统签名的测试函数存入列表 legacy_only_sigs
    legacy_only_sigs = [sin_4]

    # 使用 LowLevelCallable 尝试新签名的函数
    for j, func in enumerate(all_sigs):
        callback = LowLevelCallable(func)
        # 如果 func 是仅传统签名的函数，则应该引发 ValueError 异常
        if func in legacy_only_sigs:
            pytest.raises(ValueError, quad, callback, 0, pi)
        else:
            # 否则，对于新签名的函数，使用 quad 函数计算积分并断言结果接近 2.0
            assert_allclose(quad(callback, 0, pi)[0], 2.0)

    # 对于仅传统签名的函数，使用普通的 ctypes 对象进行测试
    for j, func in enumerate(legacy_sigs):
        if func in legacy_sigs:
            # 对于传统签名的函数，使用 quad 函数计算积分并断言结果接近 2.0
            assert_allclose(quad(func, 0, pi)[0], 2.0)
        else:
            # 对于非传统签名的函数，应该引发 ValueError 异常
            pytest.raises(ValueError, quad, func, 0, pi)
class TestMultivariateCtypesQuad:
    # 定义测试类 TestMultivariateCtypesQuad
    def setup_method(self):
        # 设置每个测试方法的准备工作
        restype = ctypes.c_double
        argtypes = (ctypes.c_int, ctypes.c_double)
        # 定义返回类型为双精度浮点数和参数类型为整数和双精度浮点数的元组
        for name in ['_multivariate_typical', '_multivariate_indefinite',
                     '_multivariate_sin']:
            # 遍历函数名称列表
            func = get_clib_test_routine(name, restype, *argtypes)
            # 调用函数获取具有特定名称和类型的函数对象
            setattr(self, name, func)
            # 将获取的函数对象设置为当前类的属性

    def test_typical(self):
        # 测试典型情况下带有两个额外参数的函数
        assert_quad(quad(self._multivariate_typical, 0, pi, (2, 1.8)),
                    0.30614353532540296487)

    def test_indefinite(self):
        # 测试无限积分上限的情况 --- 欧拉常数
        assert_quad(quad(self._multivariate_indefinite, 0, np.inf),
                    0.577215664901532860606512)

    def test_threadsafety(self):
        # 确保多变量 ctypes 是线程安全的
        def threadsafety(y):
            return y + quad(self._multivariate_sin, 0, 1)[0]
        assert_quad(quad(threadsafety, 0, 1), 0.9596976941318602)


class TestQuad:
    # 定义测试类 TestQuad
    def test_typical(self):
        # 测试典型情况下带有两个额外参数的函数
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        assert_quad(quad(myfunc, 0, pi, (2, 1.8)), 0.30614353532540296487)

    def test_indefinite(self):
        # 测试无限积分上限的情况 --- 欧拉常数
        def myfunc(x):           # Euler's constant integrand
            return -exp(-x)*log(x)
        assert_quad(quad(myfunc, 0, np.inf), 0.577215664901532860606512)

    def test_singular(self):
        # 测试积分区域中的奇点
        def myfunc(x):
            if 0 < x < 2.5:
                return sin(x)
            elif 2.5 <= x <= 5.0:
                return exp(-x)
            else:
                return 0.0

        assert_quad(quad(myfunc, 0, 10, points=[2.5, 5.0]),
                    1 - cos(2.5) + exp(-2.5) - exp(-5.0))

    def test_sine_weighted_finite(self):
        # 测试正弦加权积分（有限上限）
        def myfunc(x, a):
            return exp(a*(x-1))

        ome = 2.0**3.4
        assert_quad(quad(myfunc, 0, 1, args=20, weight='sin', wvar=ome),
                    (20*sin(ome)-ome*cos(ome)+ome*exp(-20))/(20**2 + ome**2))

    def test_sine_weighted_infinite(self):
        # 测试正弦加权积分（无限上限）
        def myfunc(x, a):
            return exp(-x*a)

        a = 4.0
        ome = 3.0
        assert_quad(quad(myfunc, 0, np.inf, args=a, weight='sin', wvar=ome),
                    ome/(a**2 + ome**2))

    def test_cosine_weighted_infinite(self):
        # 测试余弦加权积分（负无限上限）
        def myfunc(x, a):
            return exp(x*a)

        a = 2.5
        ome = 2.3
        assert_quad(quad(myfunc, -np.inf, 0, args=a, weight='cos', wvar=ome),
                    a/(a**2 + ome**2))
    def test_algebraic_log_weight(self):
        # 6) Algebraic-logarithmic weight.
        # 定义一个函数 myfunc，计算公式为 1/(1+x+2**(-a))
        def myfunc(x, a):
            return 1/(1+x+2**(-a))

        # 设置参数 a 的值为 1.5
        a = 1.5
        # 使用 assert_quad 函数验证 quad 函数对 myfunc 的积分结果是否接近 pi/sqrt((1+2**(-a))**2 - 1)
        assert_quad(quad(myfunc, -1, 1, args=a, weight='alg',
                         wvar=(-0.5, -0.5)),
                    pi/sqrt((1+2**(-a))**2 - 1))

    def test_cauchypv_weight(self):
        # 7) Cauchy prinicpal value weighting w(x) = 1/(x-c)
        # 定义一个函数 myfunc，计算公式为 2.0**(-a)/((x-1)**2+4.0**(-a))
        def myfunc(x, a):
            return 2.0**(-a)/((x-1)**2+4.0**(-a))

        # 设置参数 a 的值为 0.4
        a = 0.4
        # 计算表格化数值 tabledValue，用于后续对比
        tabledValue = ((2.0**(-0.4)*log(1.5) -
                        2.0**(-1.4)*log((4.0**(-a)+16) / (4.0**(-a)+1)) -
                        arctan(2.0**(a+2)) -
                        arctan(2.0**a)) /
                       (4.0**(-a) + 1))
        # 使用 assert_quad 函数验证 quad 函数对 myfunc 的积分结果是否接近 tabledValue，设定误差容限为 1.9e-8
        assert_quad(quad(myfunc, 0, 5, args=0.4, weight='cauchy', wvar=2.0),
                    tabledValue, error_tolerance=1.9e-8)

    def test_b_less_than_a(self):
        # 定义函数 f，计算公式为 p * np.exp(-q*x)
        def f(x, p, q):
            return p * np.exp(-q*x)

        # 对两个积分进行计算和验证，确保第一个积分值等于第二个积分值的相反数，容忍度为两个误差的最大值
        val_1, err_1 = quad(f, 0, np.inf, args=(2, 3))
        val_2, err_2 = quad(f, np.inf, 0, args=(2, 3))
        assert_allclose(val_1, -val_2, atol=max(err_1, err_2))

    def test_b_less_than_a_2(self):
        # 定义函数 f，计算公式为 np.exp(-x**2 / 2 / s) / np.sqrt(2.*s)
        def f(x, s):
            return np.exp(-x**2 / 2 / s) / np.sqrt(2.*s)

        # 对两个积分进行计算和验证，确保第一个积分值等于第二个积分值的相反数，容忍度为两个误差的最大值
        val_1, err_1 = quad(f, -np.inf, np.inf, args=(2,))
        val_2, err_2 = quad(f, np.inf, -np.inf, args=(2,))
        assert_allclose(val_1, -val_2, atol=max(err_1, err_2))

    def test_b_less_than_a_3(self):
        # 定义一个简单的函数 f，始终返回 1.0
        def f(x):
            return 1.0

        # 使用 assert_quad 函数验证 quad 函数对 f 的积分结果是否接近零，设定权重为 'alg'，权重变量为 (0, 0)
        val_1, err_1 = quad(f, 0, 1, weight='alg', wvar=(0, 0))
        val_2, err_2 = quad(f, 1, 0, weight='alg', wvar=(0, 0))
        assert_allclose(val_1, -val_2, atol=max(err_1, err_2))

    def test_b_less_than_a_full_output(self):
        # 定义一个简单的函数 f，始终返回 1.0
        def f(x):
            return 1.0

        # 使用 assert_quad 函数验证 quad 函数对 f 的积分结果是否接近零，设定权重为 'alg'，权重变量为 (0, 0)，并输出完整的输出结果
        res_1 = quad(f, 0, 1, weight='alg', wvar=(0, 0), full_output=True)
        res_2 = quad(f, 1, 0, weight='alg', wvar=(0, 0), full_output=True)
        # 计算误差为两个结果的最大误差
        err = max(res_1[1], res_2[1])
        assert_allclose(res_1[0], -res_2[0], atol=err)

    def test_double_integral(self):
        # 8) Double Integral test
        # 定义一个函数 simpfunc，参数顺序为 y, x，计算公式为 x+y
        def simpfunc(y, x):       # Note order of arguments.
            return x+y

        # 设置积分区间 a, b，并使用 assert_quad 函数验证 dblquad 对 simpfunc 的双重积分结果是否接近 5/6.0 * (b**3.0-a**3.0)
        a, b = 1.0, 2.0
        assert_quad(dblquad(simpfunc, a, b, lambda x: x, lambda x: 2*x),
                    5/6.0 * (b**3.0-a**3.0))

    def test_double_integral2(self):
        # 定义函数 func，计算公式为 x0 + x1 + t0 + t1
        def func(x0, x1, t0, t1):
            return x0 + x1 + t0 + t1
        # 定义函数 g，计算公式为 x
        def g(x):
            return x
        # 定义函数 h，计算公式为 2 * x
        def h(x):
            return 2 * x
        # 设置参数 args 的值为 1, 2，并使用 assert_quad 函数验证 dblquad 对 func 的双重积分结果是否接近 35./6 + 9*.5
        args = 1, 2
        assert_quad(dblquad(func, 1, 2, g, h, args=args), 35./6 + 9*.5)

    def test_double_integral3(self):
        # 定义函数 func，计算公式为 x0 + x1 + 1 + 2
        def func(x0, x1):
            return x0 + x1 + 1 + 2
        # 使用 assert_quad 函数验证 dblquad 对 func 的双重积分结果是否接近 6.0
        assert_quad(dblquad(func, 1, 2, 1, 2), 6.0)
    ):
        # The Gaussian Integral.
        def f(x, y):
            return np.exp(-x ** 2 - y ** 2)

        assert_quad(
            dblquad(f, x_lower, x_upper, y_lower, y_upper),
            expected,
            error_tolerance=3e-8
        )

    def test_triple_integral(self):
        # 9) Triple Integral test
        def simpfunc(z, y, x, t):      # Note order of arguments.
            return (x+y+z)*t

        a, b = 1.0, 2.0
        assert_quad(tplquad(simpfunc, a, b,
                            lambda x: x, lambda x: 2*x,
                            lambda x, y: x - y, lambda x, y: x + y,
                            (2.,)),
                     2*8/3.0 * (b**4.0 - a**4.0))

    @pytest.mark.xslow
    )
    def test_triple_integral_improper(
            self,
            x_lower,
            x_upper,
            y_lower,
            y_upper,
            z_lower,
            z_upper,
            expected
    ):
        # The Gaussian Integral.
        def f(x, y, z):
            return np.exp(-x ** 2 - y ** 2 - z ** 2)

        assert_quad(
            tplquad(f, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper),
            expected,
            error_tolerance=6e-8
        )

    def test_complex(self):
        # Define a complex function.
        def tfunc(x):
            return np.exp(1j*x)

        # Assert that the integral of the complex function from 0 to pi/2 is approximately 1+1j.
        assert np.allclose(
                    quad(tfunc, 0, np.pi/2, complex_func=True)[0],
                    1+1j)

        # Consider a divergent case to force quadpack to return an error message.
        # Compare the output against what is returned by explicit integration of the parts.
        kwargs = {'a': 0, 'b': np.inf, 'full_output': True,
                  'weight': 'cos', 'wvar': 1}
        res_c = quad(tfunc, complex_func=True, **kwargs)
        res_r = quad(lambda x: np.real(np.exp(1j*x)),
                     complex_func=False,
                     **kwargs)
        res_i = quad(lambda x: np.imag(np.exp(1j*x)),
                     complex_func=False,
                     **kwargs)

        # Assert equality between the complex result and the sum of real and imaginary parts.
        np.testing.assert_equal(res_c[0], res_r[0] + 1j*res_i[0])
        np.testing.assert_equal(res_c[1], res_r[1] + 1j*res_i[1])

        # Additional assertions about the structure of the returned results.
        assert len(res_c[2]['real']) == len(res_r[2:]) == 3
        assert res_c[2]['real'][2] == res_r[4]
        assert res_c[2]['real'][1] == res_r[3]
        assert res_c[2]['real'][0]['lst'] == res_r[2]['lst']

        assert len(res_c[2]['imag']) == len(res_i[2:]) == 1
        assert res_c[2]['imag'][0]['lst'] == res_i[2]['lst']
# 定义一个测试类 TestNQuad
class TestNQuad:
    
    # 用 pytest 的标记将此测试标记为失败较慢的测试，允许失败最多 5 次
    @pytest.mark.fail_slow(5)
    # 定义一个测试方法 test_fixed_limits
    def test_fixed_limits(self):
        
        # 定义一个函数 func1，接受四个参数 x0, x1, x2, x3，计算并返回一个复杂的数值
        def func1(x0, x1, x2, x3):
            val = (x0**2 + x1*x2 - x3**3 + np.sin(x0) +
                   (1 if (x0 - 0.2*x3 - 0.5 - 0.25*x1 > 0) else 0))
            return val
        
        # 定义一个函数 opts_basic，接受任意数量的参数，返回一个包含 'points' 键的字典
        def opts_basic(*args):
            return {'points': [0.2*args[2] + 0.5 + 0.25*args[0]]}
        
        # 使用 nquad 函数进行多维积分，调用 func1，指定各维度的积分范围和选项，返回完整输出
        res = nquad(func1, [[0, 1], [-1, 1], [.13, .8], [-.15, 1]],
                    opts=[opts_basic, {}, {}, {}], full_output=True)
        
        # 断言积分结果的前几项与预期值接近
        assert_quad(res[:-1], 1.5267454070738635)
        
        # 断言积分操作使用的评估次数大于 0 且小于 4e5
        assert_(res[-1]['neval'] > 0 and res[-1]['neval'] < 4e5)
    
    # 用 pytest 的标记将此测试标记为失败较慢的测试，允许失败最多 5 次
    @pytest.mark.fail_slow(5)
    # 定义一个测试方法 test_variable_limits
    def test_variable_limits(self):
        
        # 定义一个函数 func2，接受六个参数 x0, x1, x2, x3, t0, t1，计算并返回一个复杂的数值
        def func2(x0, x1, x2, x3, t0, t1):
            val = (x0*x1*x3**2 + np.sin(x2) + 1 +
                   (1 if x0 + t1*x1 - t0 > 0 else 0))
            return val
        
        # 定义四个限制函数 lim0, lim1, lim2, lim3，每个函数接受一定数量的参数并返回一个限制范围列表
        def lim0(x1, x2, x3, t0, t1):
            return [scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) - 1,
                    scale * (x1**2 + x2 + np.cos(x3)*t0*t1 + 1) + 1]
        
        def lim1(x2, x3, t0, t1):
            return [scale * (t0*x2 + t1*x3) - 1,
                    scale * (t0*x2 + t1*x3) + 1]
        
        def lim2(x3, t0, t1):
            return [scale * (x3 + t0**2*t1**3) - 1,
                    scale * (x3 + t0**2*t1**3) + 1]
        
        def lim3(t0, t1):
            return [scale * (t0 + t1) - 1, scale * (t0 + t1) + 1]
        
        # 定义四个选项函数 opts0, opts1, opts2, opts3，每个函数接受一定数量的参数并返回一个选项字典
        def opts0(x1, x2, x3, t0, t1):
            return {'points': [t0 - t1*x1]}
        
        def opts1(x2, x3, t0, t1):
            return {}
        
        def opts2(x3, t0, t1):
            return {}
        
        def opts3(t0, t1):
            return {}
        
        # 使用 nquad 函数进行多维积分，调用 func2，指定各维度的限制函数和选项函数，返回积分结果
        res = nquad(func2, [lim0, lim1, lim2, lim3], args=(0, 0),
                    opts=[opts0, opts1, opts2, opts3])
        
        # 断言积分结果与预期值接近
        assert_quad(res, 25.066666666666663)
    
    # 定义一个测试方法 test_square_separate_ranges_and_opts
    def test_square_separate_ranges_and_opts(self):
        
        # 定义一个简单的函数 f，接受两个参数 y 和 x，始终返回 1.0
        def f(y, x):
            return 1.0
        
        # 使用 nquad 函数进行二维积分，调用函数 f，指定分别给定的范围和选项字典，断言结果为 4.0
        assert_quad(nquad(f, [[-1, 1], [-1, 1]], opts=[{}, {}]), 4.0)
    
    # 定义一个测试方法 test_square_aliased_ranges_and_opts
    def test_square_aliased_ranges_and_opts(self):
        
        # 定义一个简单的函数 f，接受两个参数 y 和 x，始终返回 1.0
        def f(y, x):
            return 1.0
        
        # 定义范围列表 r 和选项字典 opt，使其指向相同对象
        r = [-1, 1]
        opt = {}
        
        # 使用 nquad 函数进行二维积分，调用函数 f，指定使用别名的范围和选项字典，断言结果为 4.0
        assert_quad(nquad(f, [r, r], opts=[opt, opt]), 4.0)
    
    # 定义一个测试方法 test_square_separate_fn_ranges_and_opts
    def test_square_separate_fn_ranges_and_opts(self):
        
        # 定义一个简单的函数 f，接受两个参数 y 和 x，始终返回 1.0
        def f(y, x):
            return 1.0
        
        # 定义四个函数 fn_range0, fn_range1, fn_opt0, fn_opt1，每个函数返回一个范围或选项字典
        def fn_range0(*args):
            return (-1, 1)
        
        def fn_range1(*args):
            return (-1, 1)
        
        def fn_opt0(*args):
            return {}
        
        def fn_opt1(*args):
            return {}
        
        # 定义范围列表 ranges 和选项列表 opts，分别引用四个函数
        ranges = [fn_range0, fn_range1]
        opts = [fn_opt0, fn_opt1]
        
        # 使用 nquad 函数进行二维积分，调用函数 f，指定使用函数引用的范围和选项函数，断言结果为 4.0
        assert_quad(nquad(f, ranges, opts=opts), 4.0)
    
    # 定义一个测试方法 test_square_aliased_fn_ranges_and_opts
    def test_square_aliased_fn_ranges_and_opts(self):
        
        # 定义一个简单的函数 f，接受两个参数 y 和 x，始终返回 1.0
        def f(y, x):
            return 1.0
        
        # 定义函数 fn_range 和 fn_opt，分别返回相同的范围和选项字典
        def fn_range(*args):
            return (-1, 1)
        
        def fn_opt(*args):
            return {}
        
        # 定义范围列表 ranges 和选项列表 opts，每个列表包含两次对应的函数引用
        ranges = [fn_range, fn_range]
        opts = [fn_opt, fn_opt]
        
        # 使用 nquad 函数进行二维积分，调用函数 f，指定使用函数引用的范围和选项函数，断言结果为 4.0
        assert_quad(nquad(f, ranges, opts=opts), 4.0)
    # 定义测试函数 test_matching_quad，用于测试 quad 和 nquad 函数的匹配性
    def test_matching_quad(self):
        # 定义函数 func(x)，计算 x^2 + 1 的结果
        def func(x):
            return x**2 + 1

        # 使用 quad 函数计算 func 在区间 [0, 4] 上的积分值和误差
        res, reserr = quad(func, 0, 4)
        # 使用 nquad 函数计算 func 在区间 [0, 4] 上的积分值和误差
        res2, reserr2 = nquad(func, ranges=[[0, 4]])
        # 断言 quad 和 nquad 计算的积分值几乎相等
        assert_almost_equal(res, res2)
        # 断言 quad 和 nquad 计算的误差几乎相等
        assert_almost_equal(reserr, reserr2)

    # 定义测试函数 test_matching_dblquad，用于测试 dblquad 和 nquad 函数的匹配性
    def test_matching_dblquad(self):
        # 定义函数 func2d(x0, x1)，计算 x0^2 + x1^3 - x0 * x1 + 1 的结果
        def func2d(x0, x1):
            return x0**2 + x1**3 - x0 * x1 + 1

        # 使用 dblquad 函数计算 func2d 在区间 [-2, 2] × [-3, 3] 上的二重积分值和误差
        res, reserr = dblquad(func2d, -2, 2, lambda x: -3, lambda x: 3)
        # 使用 nquad 函数计算 func2d 在区间 [-3, 3] × [-2, 2] 上的二重积分值和误差
        res2, reserr2 = nquad(func2d, [[-3, 3], (-2, 2)])
        # 断言 dblquad 和 nquad 计算的二重积分值几乎相等
        assert_almost_equal(res, res2)
        # 断言 dblquad 和 nquad 计算的误差几乎相等
        assert_almost_equal(reserr, reserr2)

    # 定义测试函数 test_matching_tplquad，用于测试 tplquad 和 nquad 函数的匹配性
    def test_matching_tplquad(self):
        # 定义函数 func3d(x0, x1, x2, c0, c1)，计算 x0^2 + c0 * x1^3 - x0 * x1 + 1 + c1 * sin(x2) 的结果
        def func3d(x0, x1, x2, c0, c1):
            return x0**2 + c0 * x1**3 - x0 * x1 + 1 + c1 * np.sin(x2)

        # 使用 tplquad 函数计算 func3d 在立体区域 [-1, 2] × [-2, 2] × [-π, π] 上的三重积分值
        res = tplquad(func3d, -1, 2, lambda x: -2, lambda x: 2,
                      lambda x, y: -np.pi, lambda x, y: np.pi,
                      args=(2, 3))
        # 使用 nquad 函数计算 func3d 在区间 [-π, π] × [-2, 2] × [-1, 2] 上的三重积分值
        res2 = nquad(func3d, [[-np.pi, np.pi], [-2, 2], (-1, 2)], args=(2, 3))
        # 断言 tplquad 和 nquad 计算的三重积分值几乎相等
        assert_almost_equal(res, res2)

    # 定义测试函数 test_dict_as_opts，测试 nquad 函数接受字典形式的参数选项
    def test_dict_as_opts(self):
        # 尝试使用 nquad 函数计算 lambda 函数 x * y 在区间 [0, 1] × [0, 1] 上的积分，并传入选项 {'epsrel': 0.0001}
        try:
            nquad(lambda x, y: x * y, [[0, 1], [0, 1]], opts={'epsrel': 0.0001})
        # 捕获 TypeError 异常，如果 nquad 函数不接受 opts 参数，则断言失败
        except TypeError:
            assert False
```