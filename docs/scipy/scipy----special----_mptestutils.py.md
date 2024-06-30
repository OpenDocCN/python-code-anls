# `D:\src\scipysrc\scipy\scipy\special\_mptestutils.py`

```
# 导入标准库中的模块
import os
import sys
import time
# 导入 itertools 模块中的 zip_longest 函数
from itertools import zip_longest

# 导入第三方库 numpy，并只导入其 np 别名
import numpy as np
# 从 numpy.testing 模块中导入 assert_ 函数
from numpy.testing import assert_
# 导入 pytest 测试框架
import pytest

# 从 scipy.special._testutils 模块中导入 assert_func_equal 函数
from scipy.special._testutils import assert_func_equal

# 尝试导入 mpmath 库，如果导入失败则忽略
try:
    import mpmath
except ImportError:
    pass


# ------------------------------------------------------------------------------
# 用于与 mpmath 进行系统测试的工具函数和类
# ------------------------------------------------------------------------------

class Arg:
    """生成一组实数轴上的数字，重点关注 'interesting' 区域，并覆盖所有数量级。

    """

    def __init__(self, a=-np.inf, b=np.inf, inclusive_a=True, inclusive_b=True):
        # 如果 a 大于 b，抛出数值错误异常
        if a > b:
            raise ValueError("a should be less than or equal to b")
        # 如果 a 是负无穷大，则设定为负无穷大的一半
        if a == -np.inf:
            a = -0.5*np.finfo(float).max
        # 如果 b 是正无穷大，则设定为正无穷大的一半
        if b == np.inf:
            b = 0.5*np.finfo(float).max
        # 设置实例变量 a 和 b
        self.a, self.b = a, b

        # 设置是否包含 a 和 b 的标志
        self.inclusive_a, self.inclusive_b = inclusive_a, inclusive_b
    # 定义一个方法 `_positive_values`，该方法是一个对象方法，带有三个参数 `a`, `b`, `n`
    def _positive_values(self, a, b, n):
        # 如果参数 `a` 小于 0，则抛出值错误异常，要求 `a` 必须是正数
        if a < 0:
            raise ValueError("a should be positive")

        # 尝试将点的一半放入从 `a` 到 10 的线性空间中，另一半放入对数空间中。
        if n % 2 == 0:
            # 如果 `n` 是偶数，则对数空间点数为 `n//2`，线性空间点数也为 `n//2`
            nlogpts = n//2
            nlinpts = nlogpts
        else:
            # 如果 `n` 是奇数，则对数空间点数为 `n//2`，线性空间点数为 `n//2 + 1`
            nlogpts = n//2
            nlinpts = nlogpts + 1

        if a >= 10:
            # 如果 `a` 大于等于 10，则超出了线性空间范围；直接返回一个对数空间
            pts = np.logspace(np.log10(a), np.log10(b), n)
        elif a > 0 and b < 10:
            # 如果 `a` 大于 0 且 `b` 小于 10，则超出了对数空间范围；直接返回一个线性空间
            pts = np.linspace(a, b, n)
        elif a > 0:
            # 如果 `a` 大于 0，则返回一个从 `a` 到 10 的线性空间和从 10 到 `b` 的对数空间
            linpts = np.linspace(a, 10, nlinpts, endpoint=False)
            logpts = np.logspace(1, np.log10(b), nlogpts)
            pts = np.hstack((linpts, logpts))
        elif a == 0 and b <= 10:
            # 如果 `a` 等于 0 且 `b` 小于等于 10，则返回一个从 0 到 `b` 的线性空间和一个从极小正数到 `b` 的对数空间
            linpts = np.linspace(0, b, nlinpts)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts = np.logspace(-30, right, nlogpts, endpoint=False)
            pts = np.hstack((logpts, linpts))
        else:
            # 如果 `a` 在 0 到 10 之间，则返回一个从 0 到 10 的线性空间，从极小正数到 `b` 的对数空间，以及从 10 到 `b` 的对数空间
            if nlogpts % 2 == 0:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1
            else:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1 + 1
            linpts = np.linspace(0, 10, nlinpts, endpoint=False)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts1 = np.logspace(-30, right, nlogpts1, endpoint=False)
            logpts2 = np.logspace(1, np.log10(b), nlogpts2)
            pts = np.hstack((logpts1, linpts, logpts2))

        # 返回排序后的结果数组 `pts`
        return np.sort(pts)
    def values(self, n):
        """Return an array containing n numbers."""
        # 从对象的属性中获取 a 和 b 的值
        a, b = self.a, self.b
        # 如果 a 等于 b，则返回包含 n 个零的数组
        if a == b:
            return np.zeros(n)

        # 如果 inclusive_a 为 False，则增加 n 的计数
        if not self.inclusive_a:
            n += 1
        # 如果 inclusive_b 为 False，则增加 n 的计数
        if not self.inclusive_b:
            n += 1

        # 如果 n 是偶数，则将其分成两半
        if n % 2 == 0:
            n1 = n//2
            n2 = n1
        else:
            n1 = n//2
            n2 = n1 + 1

        # 根据 a 的正负值调用不同的方法来获取正数的点集和负数的点集
        if a >= 0:
            pospts = self._positive_values(a, b, n)
            negpts = []
        elif b <= 0:
            pospts = []
            negpts = -self._positive_values(-b, -a, n)
        else:
            pospts = self._positive_values(0, b, n1)
            negpts = -self._positive_values(0, -a, n2 + 1)
            # 不希望零值出现两次，因此去除负数点集中的第一个零
            negpts = negpts[1:]
        # 将负数点集逆序排列并与正数点集连接起来
        pts = np.hstack((negpts[::-1], pospts))

        # 如果 inclusive_a 为 False，则去除结果数组中的第一个点
        if not self.inclusive_a:
            pts = pts[1:]
        # 如果 inclusive_b 为 False，则去除结果数组中的最后一个点
        if not self.inclusive_b:
            pts = pts[:-1]
        # 返回最终结果数组
        return pts
class FixedArg:
    # 初始化固定参数对象，将传入的值转换为 NumPy 数组
    def __init__(self, values):
        self._values = np.asarray(values)

    # 返回固定参数对象的值
    def values(self, n):
        return self._values


class ComplexArg:
    # 初始化复杂参数对象，如果未指定，默认设置为复数范围的极限值
    def __init__(self, a=complex(-np.inf, -np.inf), b=complex(np.inf, np.inf)):
        # 初始化实部和虚部参数对象
        self.real = Arg(a.real, b.real)
        self.imag = Arg(a.imag, b.imag)

    # 根据给定的 n 返回复杂参数对象的值
    def values(self, n):
        # 计算复杂参数对象的实部和虚部的值
        m = int(np.floor(np.sqrt(n)))
        x = self.real.values(m)
        y = self.imag.values(m + 1)
        # 将实部和虚部的组合形成复数数组，并展平为一维数组返回
        return (x[:,None] + 1j*y[None,:]).ravel()


class IntArg:
    # 初始化整数参数对象，如果未指定，默认设置为 -1000 到 1000 的范围
    def __init__(self, a=-1000, b=1000):
        self.a = a
        self.b = b

    # 根据给定的 n 返回整数参数对象的值
    def values(self, n):
        # 使用 Arg 类生成从 a 到 b 范围内的整数值，通过取最大值和一定规则生成的整数数组
        v1 = Arg(self.a, self.b).values(max(1 + n//2, n-5)).astype(int)
        # 创建从 -5 到 5 的整数序列
        v2 = np.arange(-5, 5)
        # 合并并去重两个数组的值，确保在范围 [a, b) 内
        v = np.unique(np.r_[v1, v2])
        v = v[(v >= self.a) & (v < self.b)]
        return v


def get_args(argspec, n):
    # 如果 argspec 是 ndarray 类型，则直接复制该数组
    if isinstance(argspec, np.ndarray):
        args = argspec.copy()
    else:
        # 否则，计算每个参数规范的权重系数，并根据这些系数生成相应数量的参数值
        nargs = len(argspec)
        ms = np.asarray(
            [1.5 if isinstance(spec, ComplexArg) else 1.0 for spec in argspec]
        )
        ms = (n**(ms/sum(ms))).astype(int) + 1

        args = [spec.values(m) for spec, m in zip(argspec, ms)]
        args = np.array(np.broadcast_arrays(*np.ix_(*args))).reshape(nargs, -1).T

    return args


class MpmathData:
    # 初始化 MpmathData 类，用于存储测试数据和参数
    def __init__(self, scipy_func, mpmath_func, arg_spec, name=None,
                 dps=None, prec=None, n=None, rtol=1e-7, atol=1e-300,
                 ignore_inf_sign=False, distinguish_nan_and_inf=True,
                 nan_ok=True, param_filter=None):

        # 根据环境变量设定 n 的默认值，用于控制测试点的数量
        if n is None:
            try:
                is_xslow = int(os.environ.get('SCIPY_XSLOW', '0'))
            except ValueError:
                is_xslow = False

            n = 5000 if is_xslow else 500

        # 初始化各种属性
        self.scipy_func = scipy_func
        self.mpmath_func = mpmath_func
        self.arg_spec = arg_spec
        self.dps = dps
        self.prec = prec
        self.n = n
        self.rtol = rtol
        self.atol = atol
        self.ignore_inf_sign = ignore_inf_sign
        self.nan_ok = nan_ok

        # 判断参数规范是否为复数类型
        if isinstance(self.arg_spec, np.ndarray):
            self.is_complex = np.issubdtype(self.arg_spec.dtype, np.complexfloating)
        else:
            self.is_complex = any(
                [isinstance(arg, ComplexArg) for arg in self.arg_spec]
            )

        # 初始化其他属性
        self.ignore_inf_sign = ignore_inf_sign
        self.distinguish_nan_and_inf = distinguish_nan_and_inf

        # 根据函数对象获取名称，如果未指定则使用函数的内置名称
        if not name or name == '<lambda>':
            name = getattr(scipy_func, '__name__', None)
        if not name or name == '<lambda>':
            name = getattr(mpmath_func, '__name__', None)
        self.name = name
        self.param_filter = param_filter
    # 定义一个方法用于检查
    def check(self):
        # 设定随机种子为1234，以确保结果可重复性
        np.random.seed(1234)

        # 生成参数值的数组
        argarr = get_args(self.arg_spec, self.n)

        # 检查前保存当前的 mpmath 精度设置
        old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
        try:
            # 如果 self.dps 不为 None，则使用指定的精度值；否则默认使用 20
            if self.dps is not None:
                dps_list = [self.dps]
            else:
                dps_list = [20]
            
            # 如果 self.prec 不为 None，则设置 mpmath 的精度为指定值
            if self.prec is not None:
                mpmath.mp.prec = self.prec

            # 根据输入参数的类型进行适当的 mpmath 类型转换，以提高精度
            if np.issubdtype(argarr.dtype, np.complexfloating):
                # 如果参数类型为复数，使用 mpc2complex 函数进行类型转换
                pytype = mpc2complex

                def mptype(x):
                    return mpmath.mpc(complex(x))
            else:
                # 如果参数类型为实数，使用 mpf2float 函数进行类型转换
                def mptype(x):
                    return mpmath.mpf(float(x))

                def pytype(x):
                    # 对于输出类型为实数的情况，如果虚部绝对值超过实部的一定比例，则返回 NaN
                    if abs(x.imag) > 1e-16*(1 + abs(x.real)):
                        return np.nan
                    else:
                        return mpf2float(x.real)

            # 尝试不同的 dps 值，直到找到一个符合条件的或者所有都不符合
            for j, dps in enumerate(dps_list):
                mpmath.mp.dps = dps

                try:
                    # 断言调用 scipy_func 与 mpmath_func 经过转换后的结果相等
                    assert_func_equal(
                        self.scipy_func,
                        lambda *a: pytype(self.mpmath_func(*map(mptype, a))),
                        argarr,
                        vectorized=False,
                        rtol=self.rtol,
                        atol=self.atol,
                        ignore_inf_sign=self.ignore_inf_sign,
                        distinguish_nan_and_inf=self.distinguish_nan_and_inf,
                        nan_ok=self.nan_ok,
                        param_filter=self.param_filter
                    )
                    # 断言通过后跳出循环
                    break
                except AssertionError:
                    # 如果当前 dps 值是最后一个，重新抛出异常
                    if j >= len(dps_list)-1:
                        tp, value, tb = sys.exc_info()
                        if value.__traceback__ is not tb:
                            raise value.with_traceback(tb)
                        raise value
        finally:
            # 恢复原来的 mpmath 精度设置
            mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec

    # 定义对象的字符串表示形式
    def __repr__(self):
        # 如果对象是复数类型，返回带有 "(complex)" 的字符串表示
        if self.is_complex:
            return f"<MpmathData: {self.name} (complex)>"
        else:
            # 否则返回简单的对象名称
            return f"<MpmathData: {self.name}>"
# 定义一个函数 assert_mpmath_equal，接受任意数量位置参数和关键字参数，创建一个 MpmathData 对象并进行检查
def assert_mpmath_equal(*a, **kw):
    d = MpmathData(*a, **kw)
    d.check()

# 定义一个装饰器函数 nonfunctional_tooslow，接受一个函数作为参数，返回一个 pytest 的跳过标记，表示测试尚未可用（速度太慢），需要进一步完善
def nonfunctional_tooslow(func):
    return pytest.mark.skip(
        reason="    Test not yet functional (too slow), needs more work."
    )(func)


# ------------------------------------------------------------------------------
# 处理 mpmath 特性的工具函数
# ------------------------------------------------------------------------------

# 定义函数 mpf2float，将 mpmath 的 mpf 类型转换为最接近的浮点数
def mpf2float(x):
    """
    Convert an mpf to the nearest floating point number. Just using
    float directly doesn't work because of results like this:

    with mp.workdps(50):
        float(mpf("0.99999999999999999")) = 0.9999999999999999

    """
    return float(mpmath.nstr(x, 17, min_fixed=0, max_fixed=0))


# 定义函数 mpc2complex，将 mpmath 的 mpc 类型转换为 Python 的复数类型
def mpc2complex(x):
    return complex(mpf2float(x.real), mpf2float(x.imag))


# 定义函数 trace_args，装饰器函数用于跟踪函数调用的参数和返回值
def trace_args(func):
    def tofloat(x):
        if isinstance(x, mpmath.mpc):
            return complex(x)
        else:
            return float(x)

    def wrap(*a, **kw):
        sys.stderr.write(f"{tuple(map(tofloat, a))!r}: ")
        sys.stderr.flush()
        try:
            r = func(*a, **kw)
            sys.stderr.write("-> %r" % r)
        finally:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return r
    return wrap


# 尝试导入 signal 模块，检查系统是否为 POSIX 系统
try:
    import signal
    POSIX = ('setitimer' in dir(signal))
except ImportError:
    POSIX = False


# 定义一个超时异常类 TimeoutError
class TimeoutError(Exception):
    pass


# 定义装饰器函数 time_limited，设置纯 Python 函数的执行超时限制
def time_limited(timeout=0.5, return_val=np.nan, use_sigalrm=True):
    """
    Decorator for setting a timeout for pure-Python functions.

    If the function does not return within `timeout` seconds, the
    value `return_val` is returned instead.

    On POSIX this uses SIGALRM by default. On non-POSIX, settrace is
    used. Do not use this with threads: the SIGALRM implementation
    does probably not work well. The settrace implementation only
    traces the current thread.

    The settrace implementation slows down execution speed. Slowdown
    by a factor around 10 is probably typical.
    """
    # 如果是 POSIX 系统且 use_sigalrm 为 True，使用 SIGALRM 实现超时
    if POSIX and use_sigalrm:
        def sigalrm_handler(signum, frame):
            raise TimeoutError()

        def deco(func):
            def wrap(*a, **kw):
                old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    return return_val
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrap
        return deco
    else:
        # 定义一个装饰器函数，接受一个函数作为参数
        def deco(func):
            # 定义一个包装函数，用于实现超时控制
            def wrap(*a, **kw):
                # 记录函数开始时间
                start_time = time.time()

                # 定义一个追踪函数，用于检测运行时间是否超过设定的超时时间
                def trace(frame, event, arg):
                    if time.time() - start_time > timeout:
                        # 如果超过超时时间，抛出 TimeoutError 异常
                        raise TimeoutError()
                    return trace
                
                # 设置 Python 解释器的追踪函数
                sys.settrace(trace)
                try:
                    # 执行被装饰的函数，传入参数
                    return func(*a, **kw)
                except TimeoutError:
                    # 捕获超时异常后，取消追踪函数，并返回指定的返回值
                    sys.settrace(None)
                    return return_val
                finally:
                    # 最终取消追踪函数
                    sys.settrace(None)
            
            # 返回包装函数，即装饰后的函数
            return wrap
    # 返回装饰器函数
    return deco
# 装饰器函数：如果被装饰的函数抛出异常，则返回 np.nan
def exception_to_nan(func):
    def wrap(*a, **kw):
        try:
            return func(*a, **kw)
        except Exception:
            return np.nan
    return wrap


# 装饰器函数：如果被装饰的函数返回无穷大 (inf)，则返回 np.nan
def inf_to_nan(func):
    def wrap(*a, **kw):
        v = func(*a, **kw)
        if not np.isfinite(v):
            return np.nan
        return v
    return wrap


# 比较两个列表中的 mpmath.mpf 或 mpmath.mpc 对象，可以使用更高精度比较（超过双精度浮点数）
def mp_assert_allclose(res, std, atol=0, rtol=1e-17):
    failures = []
    for k, (resval, stdval) in enumerate(zip_longest(res, std)):
        # 检查输入的 res 和 std 的长度是否相等
        if resval is None or stdval is None:
            raise ValueError('Lengths of inputs res and std are not equal.')
        # 计算相对误差，并与指定的容差 atol + rtol*|stdval| 进行比较
        if mpmath.fabs(resval - stdval) > atol + rtol*mpmath.fabs(stdval):
            failures.append((k, resval, stdval))

    nfail = len(failures)
    # 如果存在失败的比较，则生成错误消息
    if nfail > 0:
        ndigits = int(abs(np.log10(rtol)))
        msg = [""]
        msg.append(f"Bad results ({nfail} out of {k + 1}) for the following points:")
        for k, resval, stdval in failures:
            # 将 resval 和 stdval 转换为字符串表示，限定精度为 ndigits
            resrep = mpmath.nstr(resval, ndigits, min_fixed=0, max_fixed=0)
            stdrep = mpmath.nstr(stdval, ndigits, min_fixed=0, max_fixed=0)
            # 计算相对差异
            if stdval == 0:
                rdiff = "inf"
            else:
                rdiff = mpmath.fabs((resval - stdval)/stdval)
                rdiff = mpmath.nstr(rdiff, 3)
            # 将详细信息添加到错误消息中
            msg.append(f"{k}: {resrep} != {stdrep} (rdiff {rdiff})")
        # 如果有失败的比较，触发断言错误，输出错误消息
        assert_(False, "\n".join(msg))
```