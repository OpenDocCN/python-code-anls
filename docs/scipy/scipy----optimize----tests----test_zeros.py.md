# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_zeros.py`

```
import pytest  # 导入 pytest 库，用于单元测试

from functools import lru_cache  # 导入 lru_cache 装饰器，用于函数结果的缓存

from numpy.testing import (assert_warns, assert_,  # 导入 numpy.testing 模块的断言函数
                           assert_allclose,
                           assert_equal,
                           assert_array_equal,
                           suppress_warnings)
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos  # 导入 NumPy 中的数学函数

from scipy import optimize  # 导入 SciPy 库的 optimize 模块，用于数值优化
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,  # 导入 optimize 模块的部分函数和类
                            OptimizeResult)

from scipy._lib._util import getfullargspec_no_self as _getfullargspec  # 导入辅助函数

# Import testing parameters
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions  # 导入测试参数

TOL = 4*np.finfo(float).eps  # 定义容差 TOL，使用 float 类型的机器精度

_FLOAT_EPS = finfo(float).eps  # 获取 float 类型的机器精度，并赋值给 _FLOAT_EPS

bracket_methods = [zeros.bisect, zeros.ridder, zeros.brentq, zeros.brenth,  # 定义一组区间搜索方法
                   zeros.toms748]
gradient_methods = [zeros.newton]  # 定义一组梯度搜索方法
all_methods = bracket_methods + gradient_methods  # 将所有搜索方法合并成一个列表

# A few test functions used frequently:
# # A simple quadratic, (x-1)^2 - 1
def f1(x):
    return x ** 2 - 2 * x - 1  # 定义简单的二次方程函数 f1(x)


def f1_1(x):
    return 2 * x - 2  # 定义 f1(x) 的一阶导数函数 f1_1(x)


def f1_2(x):
    return 2.0 + 0 * x  # 定义 f1(x) 的二阶导数函数 f1_2(x)


def f1_and_p_and_pp(x):
    return f1(x), f1_1(x), f1_2(x)  # 返回 f1(x)、f1_1(x) 和 f1_2(x) 的元组

# Simple transcendental function
def f2(x):
    return exp(x) - cos(x)  # 定义简单的超越函数 f2(x)


def f2_1(x):
    return exp(x) + sin(x)  # 定义 f2(x) 的一阶导数函数 f2_1(x)


def f2_2(x):
    return exp(x) + cos(x)  # 定义 f2(x) 的二阶导数函数 f2_2(x)


# lru cached function
@lru_cache
def f_lrucached(x):
    return x  # 使用 lru_cache 装饰器缓存函数 f_lrucached(x)，以提高性能

class TestScalarRootFinders:
    # Basic tests for all scalar root finders

    xtol = 4 * np.finfo(float).eps  # 设置误差容限 xtol
    rtol = 4 * np.finfo(float).eps  # 设置相对容限 rtol

    def _run_one_test(self, tc, method, sig_args_keys=None,  # 定义单个测试运行函数 _run_one_test
                      sig_kwargs_keys=None, **kwargs):
        method_args = []
        for k in sig_args_keys or []:
            if k not in tc:
                # If a,b not present use x0, x1. Similarly for f and func
                k = {'a': 'x0', 'b': 'x1', 'func': 'f'}.get(k, k)
            method_args.append(tc[k])

        method_kwargs = dict(**kwargs)
        method_kwargs.update({'full_output': True, 'disp': False})
        for k in sig_kwargs_keys or []:
            method_kwargs[k] = tc[k]

        root = tc.get('root')
        func_args = tc.get('args', ())

        try:
            r, rr = method(*method_args, args=func_args, **method_kwargs)
            return root, rr, tc
        except Exception:
            return root, zeros.RootResults(nan, -1, -1, zeros._EVALUEERR, method), tc
    def run_tests(self, tests, method, name, known_fail=None, **kwargs):
        r"""Run test-cases using the specified method and the supplied signature.

        Extract the arguments for the method call from the test case
        dictionary using the supplied keys for the method's signature."""
        # 定义一个方法用于运行测试用例，接受多个参数，包括测试用例集合、方法、方法名称、已知失败的测试用例列表等

        # The methods have one of two base signatures:
        # (f, a, b, **kwargs)  # newton
        # (func, x0, **kwargs)  # bisect/brentq/...
        # 方法有两种基本签名之一：

        # FullArgSpec with args, varargs, varkw, defaults, ...
        sig = _getfullargspec(method)
        # 获取方法的完整参数规范

        assert_(not sig.kwonlyargs)
        # 确保方法没有仅限关键字参数

        nDefaults = len(sig.defaults)
        nRequired = len(sig.args) - nDefaults
        sig_args_keys = sig.args[:nRequired]
        # 确定方法签名中的必需参数

        sig_kwargs_keys = []
        if name in ['secant', 'newton', 'halley']:
            # 根据方法名称确定是否需要额外的关键字参数
            if name in ['newton', 'halley']:
                sig_kwargs_keys.append('fprime')
                # 如果方法是牛顿法或哈雷法，则需要 'fprime' 参数
                if name in ['halley']:
                    sig_kwargs_keys.append('fprime2')
                    # 如果方法是哈雷法，则还需要 'fprime2' 参数
            kwargs['tol'] = self.xtol
            # 设置默认的公差参数 'tol' 为对象的 xtol 属性
        else:
            kwargs['xtol'] = self.xtol
            kwargs['rtol'] = self.rtol
            # 对于其他方法，设置 'xtol' 和 'rtol' 为对象的 xtol 和 rtol 属性

        results = [list(self._run_one_test(
            tc, method, sig_args_keys=sig_args_keys,
            sig_kwargs_keys=sig_kwargs_keys, **kwargs)) for tc in tests]
        # 对每个测试用例运行单个测试方法，将结果存储在 results 列表中

        known_fail = known_fail or []
        # 如果没有已知失败的测试用例，则为空列表
        notcvgd = [elt for elt in results if not elt[1].converged]
        # 获取未收敛的测试结果列表

        notcvgd = [elt for elt in notcvgd if elt[-1]['ID'] not in known_fail]
        # 排除已知失败的测试用例后，再次筛选未收敛的测试结果列表

        notcvged_IDS = [elt[-1]['ID'] for elt in notcvgd]
        # 获取未收敛测试用例的 ID 列表

        assert_equal([len(notcvged_IDS), notcvged_IDS], [0, []])
        # 确保未收敛的测试用例数为 0，即所有测试用例都应该收敛

        # The usable xtol and rtol depend on the test
        tols = {'xtol': self.xtol, 'rtol': self.rtol}
        tols.update(**kwargs)
        # 更新公差参数字典，包括对象的 xtol 和 rtol 属性

        rtol = tols['rtol']
        atol = tols.get('tol', tols['xtol'])
        # 获取实际使用的相对误差和绝对误差

        cvgd = [elt for elt in results if elt[1].converged]
        # 获取收敛的测试结果列表

        approx = [elt[1].root for elt in cvgd]
        correct = [elt[0] for elt in cvgd]
        # 提取近似根和正确根的列表

        # See if the root matches the reference value
        notclose = [[a] + elt for a, c, elt in zip(approx, correct, cvgd) if
                    not isclose(a, c, rtol=rtol, atol=atol)
                    and elt[-1]['ID'] not in known_fail]
        # 检查近似根是否与参考值匹配，如果不匹配则存储相应的信息

        # If not, evaluate the function and see if is 0 at the purported root
        fvs = [tc['f'](aroot, *tc.get('args', tuple()))
               for aroot, c, fullout, tc in notclose]
        # 对于不匹配的近似根，计算函数值并检查是否在所述根处为零

        notclose = [[fv] + elt for fv, elt in zip(fvs, notclose) if fv != 0]
        # 如果函数值不为零，则存储相应信息

        assert_equal([notclose, len(notclose)], [[], 0])
        # 确保所有不匹配的近似根都有函数值为零

        method_from_result = [result[1].method for result in results]
        expected_method = [name for _ in results]
        # 提取每个结果的方法和期望的方法名称

        assert_equal(method_from_result, expected_method)
        # 确保每个结果的方法与期望的方法名称一致
    def run_collection(self, collection, method, name, smoothness=None,
                       known_fail=None, **kwargs):
        r"""Run a collection of tests using the specified method.

        The name is used to determine some optional arguments."""
        # 获取测试集合中的测试用例列表，根据需要平滑处理
        tests = get_tests(collection, smoothness=smoothness)
        # 调用实例方法运行测试用例列表中的测试，使用指定的方法和名称，并传递额外的参数
        self.run_tests(tests, method, name, known_fail=known_fail, **kwargs)
class TestBracketMethods(TestScalarRootFinders):
    # 继承自 TestScalarRootFinders 类，用于测试括号法求根的方法

    @pytest.mark.parametrize('method', bracket_methods)
    @pytest.mark.parametrize('function', tstutils_functions)
    def test_basic_root_scalar(self, method, function):
        # 测试使用 `root_scalar` 调用括号法求根的方法解决简单问题集合，
        # 每个问题在 `x=1` 处有一个根。检查收敛状态和找到的根。
        a, b = .5, sqrt(3)

        r = root_scalar(function, method=method.__name__, bracket=[a, b], x0=a,
                        xtol=self.xtol, rtol=self.rtol)
        # 断言根据所设置的容差值，根已经收敛到 1.0
        assert r.converged
        assert_allclose(r.root, 1.0, atol=self.xtol, rtol=self.rtol)
        # 断言使用的方法与预期相符
        assert r.method == method.__name__

    @pytest.mark.parametrize('method', bracket_methods)
    @pytest.mark.parametrize('function', tstutils_functions)
    def test_basic_individual(self, method, function):
        # 测试单独使用括号法求根的方法解决简单问题集合，
        # 每个问题在 `x=1` 处有一个根。检查收敛状态和找到的根。
        a, b = .5, sqrt(3)
        # 调用具体的括号法求根函数，获取根和结果对象
        root, r = method(function, a, b, xtol=self.xtol, rtol=self.rtol,
                         full_output=True)

        # 断言根据所设置的容差值，根已经收敛到 1.0
        assert r.converged
        assert_allclose(root, 1.0, atol=self.xtol, rtol=self.rtol)

    @pytest.mark.parametrize('method', bracket_methods)
    def test_aps_collection(self, method):
        # 运行 'aps' 集合测试，使用指定的括号法求根函数和方法名
        self.run_collection('aps', method, method.__name__, smoothness=1)

    @pytest.mark.parametrize('method', [zeros.bisect, zeros.ridder,
                                        zeros.toms748])
    def test_chandrupatla_collection(self, method):
        known_fail = {'fun7.4'} if method == zeros.ridder else {}
        # 运行 'chandrupatla' 集合测试，使用指定的方法和方法名，
        # 可能包含已知的失败案例
        self.run_collection('chandrupatla', method, method.__name__,
                            known_fail=known_fail)

    @pytest.mark.parametrize('method', bracket_methods)
    def test_lru_cached_individual(self, method):
        # 检查 https://github.com/scipy/scipy/issues/10846 是否已修复，
        # 当传递一个被 `@lru_cache` 缓存的函数给 `root_scalar` 时会失败
        a, b = -1, 1
        root, r = method(f_lrucached, a, b, full_output=True)
        assert r.converged
        assert_allclose(root, 0)


class TestNewton(TestScalarRootFinders):
    def test_newton_collections(self):
        known_fail = ['aps.13.00']
        known_fail += ['aps.12.05', 'aps.12.17']  # 在 Windows Py27 下失败
        for collection in ['aps', 'complex']:
            # 运行 'aps' 和 'complex' 集合测试，使用牛顿法求根函数和方法名，
            # 设置平滑度和已知的失败案例
            self.run_collection(collection, zeros.newton, 'newton',
                                smoothness=2, known_fail=known_fail)
    def test_halley_collections(self):
        # 预先定义一组已知失败的测试集合
        known_fail = ['aps.12.06', 'aps.12.07', 'aps.12.08', 'aps.12.09',
                      'aps.12.10', 'aps.12.11', 'aps.12.12', 'aps.12.13',
                      'aps.12.14', 'aps.12.15', 'aps.12.16', 'aps.12.17',
                      'aps.12.18', 'aps.13.00']
        # 遍历不同的数据集合进行测试，运行自定义的集合测试方法
        for collection in ['aps', 'complex']:
            self.run_collection(collection, zeros.newton, 'halley',
                                smoothness=2, known_fail=known_fail)

    def test_newton(self):
        # 遍历函数和它们的一阶和二阶导数对，测试牛顿法求根
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            x = zeros.newton(f, 3, tol=1e-6)
            assert_allclose(f(x), 0, atol=1e-6)
            x = zeros.newton(f, 3, x1=5, tol=1e-6)  # 使用初始点 x0 和 x1 进行割线法
            assert_allclose(f(x), 0, atol=1e-6)
            x = zeros.newton(f, 3, fprime=f_1, tol=1e-6)   # 使用给定的一阶导数进行牛顿法
            assert_allclose(f(x), 0, atol=1e-6)
            x = zeros.newton(f, 3, fprime=f_1, fprime2=f_2, tol=1e-6)  # 使用给定的一阶和二阶导数进行 Halley 法
            assert_allclose(f(x), 0, atol=1e-6)

    def test_newton_by_name(self):
        r"""通过 root_scalar() 调用牛顿法"""
        # 遍历函数和它们的一阶和二阶导数对，通过名称调用牛顿法
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='newton', x0=3, fprime=f_1, xtol=1e-6)
            assert_allclose(f(r.root), 0, atol=1e-6)
        # 再次遍历函数和它们的一阶和二阶导数对，通过名称调用牛顿法，但不指定一阶导数
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='newton', x0=3, xtol=1e-6)  # 不指定一阶导数
            assert_allclose(f(r.root), 0, atol=1e-6)

    def test_secant_by_name(self):
        r"""通过 root_scalar() 调用割线法"""
        # 遍历函数和它们的一阶和二阶导数对，通过名称调用割线法
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='secant', x0=3, x1=2, xtol=1e-6)
            assert_allclose(f(r.root), 0, atol=1e-6)
            r = root_scalar(f, method='secant', x0=3, x1=5, xtol=1e-6)
            assert_allclose(f(r.root), 0, atol=1e-6)
        # 再次遍历函数和它们的一阶和二阶导数对，通过名称调用割线法，但不指定割线法的第二个初始点 x1
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='secant', x0=3, xtol=1e-6)  # 不指定 x1
            assert_allclose(f(r.root), 0, atol=1e-6)

    def test_halley_by_name(self):
        r"""通过 root_scalar() 调用 Halley 法"""
        # 遍历函数和它们的一阶和二阶导数对，通过名称调用 Halley 法
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='halley', x0=3,
                            fprime=f_1, fprime2=f_2, xtol=1e-6)
            assert_allclose(f(r.root), 0, atol=1e-6)

    def test_root_scalar_fail(self):
        # 测试未能指定 Halley 法所需的二阶导数 fprime2 的情况
        message = 'fprime2 must be specified for halley'
        with pytest.raises(ValueError, match=message):
            root_scalar(f1, method='halley', fprime=f1_1, x0=3, xtol=1e-6)  # 没有指定 fprime2
        # 测试未能指定 Halley 法所需的一阶导数 fprime 的情况
        message = 'fprime must be specified for halley'
        with pytest.raises(ValueError, match=message):
            root_scalar(f1, method='halley', fprime2=f1_2, x0=3, xtol=1e-6)  # 没有指定 fprime
    def test_array_newton(self):
        """test newton with array"""

        # 定义函数 f1，接受参数 x 和可变参数 a，计算并返回复杂表达式的值
        def f1(x, *a):
            b = a[0] + x * a[3]  # 计算变量 b 的值
            return a[1] - a[2] * (np.exp(b / a[5]) - 1.0) - b / a[4] - x  # 返回复杂表达式的计算结果

        # 定义函数 f1_1，接受参数 x 和可变参数 a，计算并返回复杂表达式的导数
        def f1_1(x, *a):
            b = a[3] / a[5]  # 计算变量 b 的值
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b - a[3] / a[4] - 1  # 返回复杂表达式的导数的计算结果

        # 定义函数 f1_2，接受参数 x 和可变参数 a，计算并返回复杂表达式的二阶导数
        def f1_2(x, *a):
            b = a[3] / a[5]  # 计算变量 b 的值
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b**2  # 返回复杂表达式的二阶导数的计算结果

        # 定义数组 a0，包含十个浮点数
        a0 = np.array([
            5.32725221, 5.48673747, 5.49539973,
            5.36387202, 4.80237316, 1.43764452,
            5.23063958, 5.46094772, 5.50512718,
            5.42046290
        ])
        
        # 定义数组 a1，包含十个浮点数，通过 sin 函数和常数进行计算
        a1 = (np.sin(range(10)) + 1.0) * 7.0
        
        # 定义元组 args，包含参数 a0, a1, 1e-09, 0.004, 10, 0.27456
        args = (a0, a1, 1e-09, 0.004, 10, 0.27456)
        
        # 定义列表 x0，包含十个浮点数 7.0
        x0 = [7.0] * 10
        
        # 使用 zeros 模块中的 newton 函数进行牛顿法求解，计算结果赋给变量 x
        x = zeros.newton(f1, x0, f1_1, args)
        
        # 期望的结果，包含十个浮点数
        x_expected = (
            6.17264965, 11.7702805, 12.2219954,
            7.11017681, 1.18151293, 0.143707955,
            4.31928228, 10.5419107, 12.7552490,
            8.91225749
        )
        
        # 断言 x 与 x_expected 的值接近
        assert_allclose(x, x_expected)
        
        # 测试 Halley 方法
        x = zeros.newton(f1, x0, f1_1, args, fprime2=f1_2)
        
        # 断言 x 与 x_expected 的值接近
        assert_allclose(x, x_expected)
        
        # 测试割线法
        x = zeros.newton(f1, x0, args=args)
        
        # 断言 x 与 x_expected 的值接近
        assert_allclose(x, x_expected)

    def test_array_newton_complex(self):
        # 定义函数 f，计算并返回 x + 1 + 1j 的值
        def f(x):
            return x + 1+1j
        
        # 定义函数 fprime，计算并返回 1.0
        def fprime(x):
            return 1.0
        
        # 定义长度为 4 的复数数组 t
        t = np.full(4, 1j)
        
        # 使用 zeros 模块中的 newton 函数进行牛顿法求解，计算结果赋给变量 x
        x = zeros.newton(f, t, fprime=fprime)
        
        # 断言 f(x) 的值接近 0
        assert_allclose(f(x), 0.)

        # 即使 x0 不是复数，也应该能正常工作
        t = np.ones(4)
        x = zeros.newton(f, t, fprime=fprime)
        assert_allclose(f(x), 0.)

        x = zeros.newton(f, t)
        assert_allclose(f(x), 0.)

    def test_array_secant_active_zero_der(self):
        """test secant doesn't continue to iterate zero derivatives"""
        # 使用 zeros 模块中的 newton 函数进行牛顿法求解，计算结果赋给变量 x
        x = zeros.newton(lambda x, *a: x*x - a[0], x0=[4.123, 5],
                         args=[np.array([17, 25])])
        
        # 断言 x 的值与期望的结果接近
        assert_allclose(x, (4.123105625617661, 5.0))

    def test_array_newton_integers(self):
        # 测试带有浮点数的割线法
        x = zeros.newton(lambda y, z: z - y ** 2, [4.0] * 2,
                         args=([15.0, 17.0],))
        
        # 断言 x 的值与期望的结果接近
        assert_allclose(x, (3.872983346207417, 4.123105625617661))
        
        # 测试整数转换为浮点数
        x = zeros.newton(lambda y, z: z - y ** 2, [4] * 2, args=([15, 17],))
        
        # 断言 x 的值与期望的结果接近
        assert_allclose(x, (3.872983346207417, 4.123105625617661))
    def test_array_newton_zero_der_failures(self):
        # 测试导数为零时的警告
        assert_warns(RuntimeWarning, zeros.newton,
                     lambda y: y**2 - 2, [0., 0.], lambda y: 2 * y)
        # 测试失败和零导数
        with pytest.warns(RuntimeWarning):
            # 使用牛顿法求解方程 lambda y: y**2 - 2 = 0，初始点为 [0., 0.]，导数函数为 lambda y: 2*y，并输出详细信息
            results = zeros.newton(lambda y: y**2 - 2, [0., 0.],
                                   lambda y: 2*y, full_output=True)
            # 断言结果的根接近于 0
            assert_allclose(results.root, 0)
            # 断言结果的零导数数组中的所有值为真
            assert results.zero_der.all()
            # 断言结果的收敛状态中没有任何一个为真
            assert not results.converged.any()

    def test_newton_combined(self):
        # 定义函数 f1(x) = x^2 - 2x - 1
        def f1(x):
            return x ** 2 - 2 * x - 1
        # 定义函数 f1_1(x) = 2x - 2
        def f1_1(x):
            return 2 * x - 2
        # 定义函数 f1_2(x) = 2.0
        def f1_2(x):
            return 2.0 + 0 * x

        # 定义函数 f1_and_p_and_pp(x)，返回 f1(x)，f1_1(x)，f1_2(x)
        def f1_and_p_and_pp(x):
            return x**2 - 2*x-1, 2*x-2, 2.0

        # 使用牛顿法求解 f1(x)=0，初始点为 3，导数函数为 f1_1
        sol0 = root_scalar(f1, method='newton', x0=3, fprime=f1_1)
        # 使用牛顿法求解 f1_and_p_and_pp(x)=0，初始点为 3，导数函数为 True（自动计算）
        sol = root_scalar(f1_and_p_and_pp, method='newton', x0=3, fprime=True)
        # 断言两种方法得到的根接近
        assert_allclose(sol0.root, sol.root, atol=1e-8)
        # 断言两种方法的函数调用次数满足特定关系
        assert_equal(2*sol.function_calls, sol0.function_calls)

        # 使用 Halley 方法求解 f1(x)=0，初始点为 3，导数函数为 f1_1，二阶导数函数为 f1_2
        sol0 = root_scalar(f1, method='halley', x0=3, fprime=f1_1, fprime2=f1_2)
        # 使用 Halley 方法求解 f1_and_p_and_pp(x)=0，初始点为 3，二阶导数函数为 True（自动计算）
        sol = root_scalar(f1_and_p_and_pp, method='halley', x0=3, fprime2=True)
        # 断言两种方法得到的根接近
        assert_allclose(sol0.root, sol.root, atol=1e-8)
        # 断言两种方法的函数调用次数满足特定关系
        assert_equal(3*sol.function_calls, sol0.function_calls)

    def test_newton_full_output(self, capsys):
        # 测试 full_output 功能，包括收敛和不收敛两种情况
        # 使用简单的多项式，避免平台依赖（如指数和三角函数）导致迭代次数变化

        x0 = 3
        # 预期的迭代次数和函数调用次数
        expected_counts = [(6, 7), (5, 10), (3, 9)]

        for derivs in range(3):
            kwargs = {'tol': 1e-6, 'full_output': True, }
            # 根据 derivs 的不同值，选择导数函数 f1_1 和 f1_2
            for k, v in [['fprime', f1_1], ['fprime2', f1_2]][:derivs]:
                kwargs[k] = v

            # 使用牛顿法求解 f1(x)=0，初始点为 x0，不显示过程中间信息，使用指定的参数
            x, r = zeros.newton(f1, x0, disp=False, **kwargs)
            # 断言结果已经收敛
            assert_(r.converged)
            # 断言结果的根等于 r.root
            assert_equal(x, r.root)
            # 断言迭代次数和函数调用次数符合预期
            assert_equal((r.iterations, r.function_calls), expected_counts[derivs])
            if derivs == 0:
                assert r.function_calls <= r.iterations + 1
            else:
                assert_equal(r.function_calls, (derivs + 1) * r.iterations)

            # 重复上述过程，减少一次迭代以强制收敛失败
            iters = r.iterations - 1
            # 使用牛顿法求解 f1(x)=0，初始点为 x0，最大迭代次数为 iters，不显示过程中间信息，使用指定的参数
            x, r = zeros.newton(f1, x0, maxiter=iters, disp=False, **kwargs)
            # 断言结果未收敛
            assert_(not r.converged)
            # 断言结果的根等于 r.root
            assert_equal(x, r.root)
            # 断言迭代次数符合预期
            assert_equal(r.iterations, iters)

            if derivs == 1:
                # 检查是否引发正确的异常，并验证消息的开头部分
                msg = 'Failed to converge after %d iterations, value is .*' % (iters)
                with pytest.raises(RuntimeError, match=msg):
                    # 使用牛顿法求解 f1(x)=0，初始点为 x0，最大迭代次数为 iters，显示过程中间信息，使用指定的参数
                    x, r = zeros.newton(f1, x0, maxiter=iters, disp=True, **kwargs)
    # 测试函数：检查在导数为零时是否会触发 RuntimeWarning
    def test_deriv_zero_warning(self):
        # 定义一个简单的二次函数
        def func(x):
            return x ** 2 - 2.0
        # 定义该函数的导数
        def dfunc(x):
            return 2 * x
        # 断言：调用 newton 函数时，希望捕获 RuntimeWarning
        assert_warns(RuntimeWarning, zeros.newton, func, 0.0, dfunc, disp=False)
        # 使用 pytest 断言：调用 newton 函数时，期望抛出 RuntimeError，并匹配特定消息
        with pytest.raises(RuntimeError, match='Derivative was zero'):
            zeros.newton(func, 0.0, dfunc)

    # 测试函数：检查 newton 函数是否会修改初始值 x0
    def test_newton_does_not_modify_x0(self):
        # https://github.com/scipy/scipy/issues/9964
        x0 = np.array([0.1, 3])
        x0_copy = x0.copy()  # 复制 x0 以便后续比较相等性
        # 调用 newton 函数，不期望修改 x0 的值
        newton(np.sin, x0, np.cos)
        # 使用 assert_array_equal 断言：确保调用后 x0 的值没有改变
        assert_array_equal(x0, x0_copy)

    # 测试函数：检查 root_scalar 在默认参数设置下的行为
    def test_gh17570_defaults(self):
        # 先前，当未指定 fprime 时，root_scalar 默认使用割线法。当未指定 x1 时，割线法失败。
        # 检查当没有指定 fprime 时，默认方法是 newton；当指定了 x1 时，默认方法是割线法。
        res_newton_default = root_scalar(f1, method='newton', x0=3, xtol=1e-6)
        res_secant_default = root_scalar(f1, method='secant', x0=3, x1=2,
                                         xtol=1e-6)
        # `newton` 方法在指定了 `x1` 和 `x2` 时使用割线法
        res_secant = newton(f1, x0=3, x1=2, tol=1e-6, full_output=True)[1]

        # 断言：确保所有方法找到了根
        assert_allclose(f1(res_newton_default.root), 0, atol=1e-6)
        assert res_newton_default.root.shape == tuple()
        assert_allclose(f1(res_secant_default.root), 0, atol=1e-6)
        assert res_secant_default.root.shape == tuple()
        assert_allclose(f1(res_secant.root), 0, atol=1e-6)
        assert res_secant.root.shape == tuple()

        # 断言：检查默认设置是否正确
        assert (res_secant_default.root
                == res_secant.root
                != res_newton_default.iterations)
        assert (res_secant_default.iterations
                == res_secant_default.function_calls - 1  # 对于割线法成立
                == res_secant.iterations
                != res_newton_default.iterations
                == res_newton_default.function_calls/2)  # newton 方法的二点差分

    # 测试函数：使用参数化测试检查 root_scalar 的行为
    @pytest.mark.parametrize('kwargs', [dict(), {'method': 'newton'}])
    def test_args_gh19090(self, kwargs):
        # 定义一个带有参数的函数
        def f(x, a, b):
            assert a == 3
            assert b == 1
            return (x ** a - b)

        # 调用 optimize.root_scalar 函数，使用给定的 kwargs 参数
        res = optimize.root_scalar(f, x0=3, args=(3, 1), **kwargs)
        # 断言：确保计算收敛
        assert res.converged
        assert_allclose(res.root, 1)

    # 参数化测试：使用不同的方法参数来测试 root_scalar 函数的行为
    @pytest.mark.parametrize('method', ['secant', 'newton'])
    # 定义测试函数，用于测试修复整数输入问题的代码是否有效
    def test_int_x0_gh19280(self, method):
        # 原本 `newton` 函数确保只有浮点数传递给可调用对象。这个行为不小心在 gh-17669 中改变了。
        # 现在需要检查它是否已经修复回来。
        
        # 定义测试函数，计算给定整数的负幂，验证整数输入是否可以正确处理
        def f(x):
            # 一个整数的负幂操作会失败
            return x**-2 - 2
        
        # 使用优化模块的 root_scalar 函数计算函数 f 的根，起始点 x0=1，使用指定的方法
        res = optimize.root_scalar(f, x0=1, method=method)
        
        # 断言根据所选方法函数是否收敛
        assert res.converged
        
        # 断言所得根的绝对值是否接近预期的数值 2 的平方根的倒数
        assert_allclose(abs(res.root), 2**-0.5)
        
        # 断言所得根的数据类型是否为 np.float64 类型
        assert res.root.dtype == np.dtype(np.float64)
def test_gh_5555():
    # 设置根的初始猜测值
    root = 0.1

    # 定义一个函数 f(x)，返回 x 减去 root 的结果
    def f(x):
        return x - root

    # 设置要测试的方法列表
    methods = [zeros.bisect, zeros.ridder]

    # 设置容差值
    xtol = rtol = TOL

    # 对每种方法进行迭代测试
    for method in methods:
        # 使用当前方法求解方程 f(x)=0，给定初始范围 [-1e8, 1e7]
        res = method(f, -1e8, 1e7, xtol=xtol, rtol=rtol)
        
        # 断言求解结果与预期根的接近程度
        assert_allclose(root, res, atol=xtol, rtol=rtol,
                        err_msg='method %s' % method.__name__)


def test_gh_5557():
    # 展示在 5557 中的更改之前，brentq 和 brenth 可能只能达到 2*(xtol + rtol*|res|) 的容差
    # 函数 f 在 (0, -0.1), (0.5, -0.1), 和 (1, 0.4) 线性插值。
    # 重要的是 |f(0)| < |f(1)|（这样 brent 可以把 0 作为初始猜测）、|f(0)| < atol（这样 brent 接受 0 作为根）、
    # 以及 f 的确切根离 0 大于 atol（这样 brent 不能达到期望的容差）。
    def f(x):
        if x < 0.5:
            return -0.1
        else:
            return x - 0.6

    # 设置容差值和相对误差
    atol = 0.51
    rtol = 4 * _FLOAT_EPS

    # 设置要测试的方法列表
    methods = [zeros.brentq, zeros.brenth]

    # 对每种方法进行迭代测试
    for method in methods:
        # 使用当前方法求解方程 f(x)=0，给定初始范围 [0, 1]
        res = method(f, 0, 1, xtol=atol, rtol=rtol)
        
        # 断言求解结果与预期根的接近程度
        assert_allclose(0.6, res, atol=atol, rtol=rtol)


def test_brent_underflow_in_root_bracketing():
    # 测试如果一个区间 [a,b] 包围函数的零点，通过检查 f(a)*f(b) < 0 是否可靠，当乘积下溢/溢出时不可靠（见 issue# 13737）
    
    # 下溢场景和溢出场景的例子
    underflow_scenario = (-450.0, -350.0, -400.0)
    overflow_scenario = (350.0, 450.0, 400.0)

    # 对于每个场景，使用 brenth 和 brentq 方法进行测试
    for a, b, root in [underflow_scenario, overflow_scenario]:
        # 计算 c 的值，用于生成函数 lambda x: np.exp(x)-c
        c = np.exp(root)
        
        # 对每种方法进行迭代测试
        for method in [zeros.brenth, zeros.brentq]:
            # 使用当前方法求解方程 np.exp(x)-c=0，给定初始范围 [a, b]
            res = method(lambda x: np.exp(x)-c, a, b)
            
            # 断言求解结果与预期根的接近程度
            assert_allclose(root, res)


class TestRootResults:
    # 初始化一个 RootResults 对象，包含根、迭代次数、函数调用次数、标志和方法名等属性
    r = zeros.RootResults(root=1.0, iterations=44, function_calls=46, flag=0,
                          method="newton")

    # 测试 RootResults 对象的字符串表示是否符合预期
    def test_repr(self):
        expected_repr = ("      converged: True\n           flag: converged"
                         "\n function_calls: 46\n     iterations: 44\n"
                         "           root: 1.0\n         method: newton")
        assert_equal(repr(self.r), expected_repr)

    # 测试 RootResults 对象是否是 OptimizeResult 的实例
    def test_type(self):
        assert isinstance(self.r, OptimizeResult)


def test_complex_halley():
    """测试 Halley 方法是否适用于复数根"""
    
    # 定义一个二次多项式函数 f(x, *a)
    def f(x, *a):
        return a[0] * x**2 + a[1] * x + a[2]

    # 定义 f 的一阶导数函数 f_1(x, *a)
    def f_1(x, *a):
        return 2 * a[0] * x + a[1]

    # 定义 f 的二阶导数函数 f_2(x, *a)
    def f_2(x, *a):
        retval = 2 * a[0]
        try:
            size = len(x)
        except TypeError:
            return retval
        else:
            return [retval] * size

    # 设置初始猜测值 z 和多项式系数 coeffs
    z = complex(1.0, 2.0)
    coeffs = (2.0, 3.0, 4.0)

    # 使用 Newton 方法求解复数根，给定初始猜测值 z 和多项式系数 coeffs
    y = zeros.newton(f, z, args=coeffs, fprime=f_1, fprime2=f_2, tol=1e-6)
    
    # 断言求解结果 f(y, *coeffs) 是否接近 0
    assert_allclose(f(y, *coeffs), 0, atol=1e-6)

    # 将 z 复制 10 次，测试多个初始猜测值的情况
    z = [z] * 10

    # 使用 Newton 方法求解复数根，给定初始猜测值 z 和多项式系数 coeffs
    y = zeros.newton(f, z, args=coeffs, fprime=f_1, fprime2=f_2, tol=1e-6)
    # 使用 assert_allclose 函数验证函数 f(y, *coeffs) 的返回值是否接近 0，允许的绝对误差为 1e-6
    assert_allclose(f(y, *coeffs), 0, atol=1e-6)
# 测试带有非零 dp（变化量）但无限牛顿步骤的割线法
def test_zero_der_nz_dp(capsys):
    """Test secant method with a non-zero dp, but an infinite newton step"""

    # 选择一个对称的函数，并选择一个点，使得在 dx 的情况下，割线的斜率为零
    # 例如 f = (x - 100)**2，在 x = 100 处有一个根，且围绕 x = 100 对称
    # 我们需要选择一个非常大的数字，以确保它一直成立

    # 计算一个小的数值，用于计算 p0，确保割线的斜率为零
    dx = np.finfo(float).eps ** 0.33

    # 根据割线的斜率为零的条件计算 p0
    p0 = (200.0 - dx) / (2.0 + dx)

    # 使用 suppress_warnings 上下文管理器来过滤 RuntimeWarning
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "RMS of")

        # 使用 Newton 方法求解方程 (y - 100.0)**2 = 0，初始值 x0 是一个包含 10 个 p0 的列表
        x = zeros.newton(lambda y: (y - 100.0)**2, x0=[p0] * 10)

    # 断言结果 x 应接近 [100, 100, ..., 100]，即列表中包含 10 个 100
    assert_allclose(x, [100] * 10)

    # 测试标量情况
    p0 = (2.0 - 1e-4) / (2.0 + 1e-4)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "Tolerance of")

        # 使用 Newton 方法求解方程 (y - 1.0)**2 = 0，初始值 x0 是一个标量 p0，禁用显示信息
        x = zeros.newton(lambda y: (y - 1.0) ** 2, x0=p0, disp=False)

    # 断言结果 x 应接近 1
    assert_allclose(x, 1)

    # 使用 pytest 的断言检查是否会引发 RuntimeError 异常，匹配消息 'Tolerance of'
    with pytest.raises(RuntimeError, match='Tolerance of'):
        # 使用 Newton 方法求解方程 (y - 1.0)**2 = 0，初始值 x0 是一个标量 p0，启用显示信息
        x = zeros.newton(lambda y: (y - 1.0) ** 2, x0=p0, disp=True)

    p0 = (-2.0 + 1e-4) / (2.0 + 1e-4)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "Tolerance of")

        # 使用 Newton 方法求解方程 (y + 1.0)**2 = 0，初始值 x0 是一个标量 p0，禁用显示信息
        x = zeros.newton(lambda y: (y + 1.0) ** 2, x0=p0, disp=False)

    # 断言结果 x 应接近 -1
    assert_allclose(x, -1)

    # 使用 pytest 的断言检查是否会引发 RuntimeError 异常，匹配消息 'Tolerance of'
    with pytest.raises(RuntimeError, match='Tolerance of'):
        # 使用 Newton 方法求解方程 (y + 1.0)**2 = 0，初始值 x0 是一个标量 p0，启用显示信息
        x = zeros.newton(lambda y: (y + 1.0) ** 2, x0=p0, disp=True)


# 测试数组输入情况下 Newton 方法的失败
def test_array_newton_failures():
    """Test that array newton fails as expected"""

    # 定义一些管道流体的属性
    diameter = 0.10  # 管道直径 [m]
    roughness = 0.00015  # 管道表面粗糙度 [m]
    rho = 988.1  # 流体密度 [kg/m**3]
    mu = 5.4790e-04  # 流体动力粘度 [Pa*s]
    u = 2.488  # 流速 [m/s]

    # 计算雷诺数 Reynolds number
    reynolds_number = rho * u * diameter / mu

    # 定义 Colebrook 方程
    def colebrook_eqn(darcy_friction, re, dia):
        return (1 / np.sqrt(darcy_friction) +
                2 * np.log10(roughness / 3.7 / dia +
                             2.51 / re / np.sqrt(darcy_friction)))

    # 使用 pytest 的 warn 上下文管理器检查是否有 RuntimeWarning 警告
    with pytest.warns(RuntimeWarning):

        # 使用 Newton 方法求解 Colebrook 方程，初始值 x0 是一个包含多个数值的列表，最大迭代次数为 2
        result = zeros.newton(
            colebrook_eqn, x0=[0.01, 0.2, 0.02223, 0.3], maxiter=2,
            args=[reynolds_number, diameter], full_output=True
        )

        # 断言结果 result 的收敛情况，期望有至少一个未收敛
        assert not result.converged.all()

    # 使用 pytest 的 raises 断言检查是否会引发 RuntimeError 异常
    with pytest.raises(RuntimeError):

        # 使用 Newton 方法再次求解 Colebrook 方程，初始值 x0 是包含两个相同数值的列表，最大迭代次数为 2
        result = zeros.newton(
            colebrook_eqn, x0=[0.01] * 2, maxiter=2,
            args=[reynolds_number, diameter], full_output=True
        )


# 这个测试不应该引发 RuntimeWarning
def test_gh8904_zeroder_at_root_fails():
    """Test that Newton or Halley don't warn if zero derivative at root"""

    # 一个在其根处具有零导数的函数
    def f_zeroder_root(x):
        return x**3 - x**2

    # 应该能够使用割线法处理这类函数
    # 使用牛顿法找到函数 f_zeroder_root 的根，并将结果赋给 r
    r = zeros.newton(f_zeroder_root, x0=0)
    # 断言 r 的值接近于 0，使用绝对容差 atol 和相对容差 rtol
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    
    # 再次使用数组作为初始值进行测试
    r = zeros.newton(f_zeroder_root, x0=[0]*10)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

    # 定义函数 fder，表示 f_zeroder_root 的一阶导数
    def fder(x):
        return 3 * x**2 - 2 * x

    # 定义函数 fder2，表示 f_zeroder_root 的二阶导数
    def fder2(x):
        return 6*x - 2

    # 使用牛顿法，并指定一阶导数 fprime=fder
    r = zeros.newton(f_zeroder_root, x0=0, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    
    # 使用牛顿法，并同时指定一阶导数 fprime=fder 和二阶导数 fprime2=fder2
    r = zeros.newton(f_zeroder_root, x0=0, fprime=fder,
                     fprime2=fder2)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    
    # 再次使用数组作为初始值进行测试，同时指定一阶导数 fprime=fder
    r = zeros.newton(f_zeroder_root, x0=[0]*10, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    
    # 再次使用数组作为初始值进行测试，同时指定一阶导数 fprime=fder 和二阶导数 fprime2=fder2
    r = zeros.newton(f_zeroder_root, x0=[0]*10, fprime=fder,
                     fprime2=fder2)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

    # 测试在一阶导数为零时是否能正常找到根，避免引发 RuntimeWarning
    r = zeros.newton(f_zeroder_root, x0=0.5, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    
    # 再次使用数组作为初始值进行测试，在一阶导数为零时是否能正常找到根，避免引发 RuntimeWarning
    r = zeros.newton(f_zeroder_root, x0=[0.5]*10, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    # Halley 方法不适用此条件，因此不进行测试
def test_gh_8881():
    r"""Test that Halley's method realizes that the 2nd order adjustment
    is too big and drops off to the 1st order adjustment."""
    n = 9  # 设置常数 n 的值为 9

    def f(x):
        return power(x, 1.0/n) - power(n, 1.0/n)  # 定义函数 f(x)，计算 x 的 n 次方根减去 n 的 n 次方根

    def fp(x):
        return power(x, (1.0-n)/n)/n  # 定义函数 fp(x)，计算 f(x) 的导数

    def fpp(x):
        return power(x, (1.0-2*n)/n) * (1.0/n) * (1.0-n)/n  # 定义函数 fpp(x)，计算 f(x) 的二阶导数

    x0 = 0.1  # 设置初始猜测值 x0 为 0.1
    # 根据 Newton-Raphson 方法，使用 f 和 fp 进行迭代求根
    rt, r = newton(f, x0, fprime=fp, full_output=True)
    assert r.converged  # 断言求根过程已收敛

    # 在 Issue 8881/PR 8882 之前，Halley 方法可能会导致 x 向错误的方向移动
    # 检查现在是否成功
    rt, r = newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
    assert r.converged  # 断言求根过程已收敛


def test_gh_9608_preserve_array_shape():
    """
    Test that shape is preserved for array inputs even if fprime or fprime2 is
    scalar
    """
    def f(x):
        return x**2  # 定义函数 f(x)，计算 x 的平方

    def fp(x):
        return 2 * x  # 定义函数 fp(x)，计算 f(x) 的导数

    def fpp(x):
        return 2  # 定义函数 fpp(x)，计算 f(x) 的二阶导数

    x0 = np.array([-2], dtype=np.float32)  # 创建包含一个元素的浮点数数组 x0
    rt, r = newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
    assert r.converged  # 断言求根过程已收敛

    x0_array = np.array([-2, -3], dtype=np.float32)  # 创建包含两个元素的浮点数数组 x0_array
    # 下面的调用应该失败
    with pytest.raises(IndexError):
        result = zeros.newton(
            f, x0_array, fprime=fp, fprime2=fpp, full_output=True
        )

    def fpp_array(x):
        return np.full(np.shape(x), 2, dtype=np.float32)  # 返回形状与 x 相同的元素都为 2 的数组

    result = zeros.newton(
        f, x0_array, fprime=fp, fprime2=fpp_array, full_output=True
    )
    assert result.converged.all()  # 断言所有求根过程均已收敛


@pytest.mark.parametrize(
    "maximum_iterations,flag_expected",
    [(10, zeros.CONVERR), (100, zeros.CONVERGED)])
def test_gh9254_flag_if_maxiter_exceeded(maximum_iterations, flag_expected):
    """
    Test that if the maximum iterations is exceeded that the flag is not
    converged.
    """
    result = zeros.brentq(
        lambda x: ((1.2*x - 2.3)*x + 3.4)*x - 4.5,
        -30, 30, (), 1e-6, 1e-6, maximum_iterations,
        full_output=True, disp=False)
    assert result[1].flag == flag_expected  # 断言结果标志与预期标志相同
    if flag_expected == zeros.CONVERR:
        # 因为超过了最大迭代次数而未收敛
        assert result[1].iterations == maximum_iterations  # 断言迭代次数等于最大迭代次数
    elif flag_expected == zeros.CONVERGED:
        # 在最大迭代次数之前已收敛
        assert result[1].iterations < maximum_iterations  # 断言迭代次数小于最大迭代次数


def test_gh9551_raise_error_if_disp_true():
    """Test that if disp is true then zero derivative raises RuntimeError"""

    def f(x):
        return x*x + 1  # 定义函数 f(x)，计算 x 的平方加 1

    def f_p(x):
        return 2*x  # 定义函数 f_p(x)，计算 f(x) 的导数

    assert_warns(RuntimeWarning, zeros.newton, f, 1.0, f_p, disp=False)  # 断言在 disp=False 时产生 RuntimeWarning
    with pytest.raises(
            RuntimeError,
            match=r'^Derivative was zero\. Failed to converge after \d+ iterations, '
                  r'value is [+-]?\d*\.\d+\.$'):
        zeros.newton(f, 1.0, f_p)  # 断言在没有导数的情况下产生 RuntimeError
    root = zeros.newton(f, complex(10.0, 10.0), f_p)  # 使用复数作为初始值进行求根
    # 使用 assert_allclose 函数检查变量 root 的值是否等于复数 0.0 + 1.0j。
    assert_allclose(root, complex(0.0, 1.0))
@pytest.mark.parametrize('solver_name',
                         ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
def test_gh3089_8394(solver_name):
    # gh-3089 and gh-8394 reported that bracketing solvers returned incorrect
    # results when they encountered NaNs. Check that this is resolved.

    # 定义一个函数 f(x)，始终返回 NaN
    def f(x):
        return np.nan

    # 从 zeros 模块中获取特定的求解器函数
    solver = getattr(zeros, solver_name)
    
    # 使用 pytest 来断言调用 solver 函数时会抛出 ValueError 异常，并检查异常信息是否包含特定字符串
    with pytest.raises(ValueError, match="The function value at x..."):
        solver(f, 0, 1)


@pytest.mark.parametrize('method',
                         ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
def test_gh18171(method):
    # gh-3089 and gh-8394 reported that bracketing solvers returned incorrect
    # results when they encountered NaNs. Check that `root_scalar` returns
    # normally but indicates that convergence was unsuccessful. See gh-18171.

    # 定义一个函数 f(x)，每次调用增加计数 f._count，并始终返回 NaN
    def f(x):
        f._count += 1
        return np.nan
    f._count = 0

    # 使用 root_scalar 函数来求解 f(x)=NaN 的根，使用给定的方法和初始区间
    res = root_scalar(f, bracket=(0, 1), method=method)
    
    # 断言求解结果表明未收敛
    assert res.converged is False
    # 断言结果标志以特定字符串开头，指示出现函数值错误
    assert res.flag.startswith("The function value at x")
    # 断言函数调用次数与 f._count 相等
    assert res.function_calls == f._count
    # 断言根在结果标志中出现过
    assert str(res.root) in res.flag


@pytest.mark.parametrize('solver_name',
                         ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
@pytest.mark.parametrize('rs_interface', [True, False])
def test_function_calls(solver_name, rs_interface):
    # There do not appear to be checks that the bracketing solvers report the
    # correct number of function evaluations. Check that this is the case.
    
    # 根据 rs_interface 决定使用不同的求解器函数
    solver = ((lambda f, a, b, **kwargs: root_scalar(f, bracket=(a, b)))
              if rs_interface else getattr(zeros, solver_name))

    # 定义一个函数 f(x)，每次调用增加计数 f.calls，并返回 x^2 - 1
    def f(x):
        f.calls += 1
        return x**2 - 1
    f.calls = 0

    # 使用求解器函数 solver 来求解 f(x)=0 在区间 [0, 10] 内的根，并获取完整的输出信息
    res = solver(f, 0, 10, full_output=True)

    # 根据 rs_interface 判断如何断言函数调用次数与 f.calls 相等
    if rs_interface:
        assert res.function_calls == f.calls
    else:
        assert res[1].function_calls == f.calls


def test_gh_14486_converged_false():
    """Test that zero slope with secant method results in a converged=False"""
    
    # 定义一个函数 lhs(x)，返回 x * np.exp(-x*x) - 0.07
    def lhs(x):
        return x * np.exp(-x*x) - 0.07

    # 使用 root_scalar 函数来求解 lhs(x)=0 使用 secant 方法，给定初始点 x0=-0.15, x1=1.0
    with pytest.warns(RuntimeWarning, match='Tolerance of'):
        res = root_scalar(lhs, method='secant', x0=-0.15, x1=1.0)
    # 断言结果未收敛
    assert not res.converged
    # 断言结果标志为 'convergence error'
    assert res.flag == 'convergence error'

    # 使用 newton 函数来求解 lhs(x)=0，给定初始点 x0=-0.15, x1=1.0，但不显示输出，获取完整输出信息后取第二个元素
    with pytest.warns(RuntimeWarning, match='Tolerance of'):
        res = newton(lhs, x0=-0.15, x1=1.0, disp=False, full_output=True)[1]
    # 断言结果未收敛
    assert not res.converged
    # 断言结果标志为 'convergence error'


@pytest.mark.parametrize('solver_name',
                         ['brentq', 'brenth', 'bisect', 'ridder', 'toms748'])
@pytest.mark.parametrize('rs_interface', [True, False])
def test_gh5584(solver_name, rs_interface):
    # gh-5584 reported that an underflow can cause sign checks in the algorithm
    # to fail. Check that this is resolved.
    
    # 根据 rs_interface 决定使用不同的求解器函数
    solver = ((lambda f, a, b, **kwargs: root_scalar(f, bracket=(a, b)))
              if rs_interface else getattr(zeros, solver_name))
    # 定义一个函数 f(x)，返回结果为 1e-200*x
    def f(x):
        return 1e-200*x

    # 当函数 f 在指定区间 [-0.5, -0.4] 内的解不存在时，应抛出 ValueError 异常，匹配错误信息为 '...must have different signs'
    with pytest.raises(ValueError, match='...must have different signs'):
        # 调用 solver 函数尝试解方程，期望抛出 ValueError 异常
        solver(f, -0.5, -0.4, full_output=True)

    # 当函数 f 在区间 [-0.5, 0.4] 内的解存在且符号不同时，成功解方程
    res = solver(f, -0.5, 0.4, full_output=True)
    # 若使用 RS 接口，则结果为 res，否则取 res 的第二个元素
    res = res if rs_interface else res[1]
    # 断言解收敛
    assert res.converged
    # 断言解的根接近于 0，允许的绝对误差为 1e-8
    assert_allclose(res.root, 0, atol=1e-8)

    # 当函数 f 在区间 [-0.5, -0.0] 内，其中一侧的值为负零时，成功解方程
    res = solver(f, -0.5, float('-0.0'), full_output=True)
    # 若使用 RS 接口，则结果为 res，否则取 res 的第二个元素
    res = res if rs_interface else res[1]
    # 断言解收敛
    assert res.converged
    # 断言解的根接近于 0，允许的绝对误差为 1e-8
    assert_allclose(res.root, 0, atol=1e-8)
# 测试 GitHub 问题 #13407
def test_gh13407():
    # 定义一个函数 f(x)，计算 x^3 - 2*x - 5 的值
    def f(x):
        return x**3 - 2*x - 5

    # 设置 xtol 为极小值，eps 为浮点数类型的机器精度
    xtol = 1e-300
    eps = np.finfo(float).eps

    # 使用 `zeros.toms748` 求解函数 f 的根，指定 xtol 和 rtol=1*eps
    x1 = zeros.toms748(f, 1e-10, 1e10, xtol=xtol, rtol=1*eps)
    # 计算在 x1 处函数 f 的值
    f1 = f(x1)

    # 使用 `zeros.toms748` 求解函数 f 的根，指定 xtol 和 rtol=4*eps
    x4 = zeros.toms748(f, 1e-10, 1e10, xtol=xtol, rtol=4*eps)
    # 计算在 x4 处函数 f 的值
    f4 = f(x4)

    # 断言：rtol=eps 时得到的函数值 f1 应小于 rtol=4*eps 时得到的函数值 f4
    assert f1 < f4

    # 使用旧式语法获取完全相同的错误消息
    message = fr"rtol too small \({eps/2:g} < {eps:g}\)"
    # 断言：使用 rtol=eps/2 时，应该引发 ValueError 错误，并匹配特定的错误消息
    with pytest.raises(ValueError, match=message):
        zeros.toms748(f, 1e-10, 1e10, xtol=xtol, rtol=eps/2)


# 测试 GitHub 问题 #10103
def test_newton_complex_gh10103():
    # 定义一个函数 f(z)，计算 z - 1 的值
    def f(z):
        return z - 1

    # 使用 `newton` 函数求解 f 的根，起始点为 1+1j
    res = newton(f, 1+1j)
    # 断言：求解得到的根 res 应接近于 1，允许的误差为 1e-12
    assert_allclose(res, 1, atol=1e-12)

    # 使用 `root_scalar` 函数通过割线法求解 f 的根，指定起始点和终止点
    res = root_scalar(f, x0=1+1j, x1=2+1.5j, method='secant')
    # 断言：求解得到的根的值应接近于 1，允许的误差为 1e-12
    assert_allclose(res.root, 1, atol=1e-12)


# 使用所有方法参数化测试最大迭代次数整数检查，测试 GitHub 问题 #10236
@pytest.mark.parametrize('method', all_methods)
def test_maxiter_int_check_gh10236(method):
    # 定义一个错误消息，用于检查当 `maxiter` 不是整数时的异常情况
    message = "'float' object cannot be interpreted as an integer"
    # 断言：使用非整数 maxiter 时，应该引发 TypeError 错误，并匹配特定的错误消息
    with pytest.raises(TypeError, match=message):
        method(f1, 0.0, 1.0, maxiter=72.45)
```