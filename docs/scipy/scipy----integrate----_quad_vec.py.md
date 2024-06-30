# `D:\src\scipysrc\scipy\scipy\integrate\_quad_vec.py`

```
import sys  # 导入sys模块，用于访问系统相关的功能
import copy  # 导入copy模块，用于对象的浅拷贝和深拷贝操作
import heapq  # 导入heapq模块，提供堆队列算法的实现
import collections  # 导入collections模块，包含了额外的数据结构实现
import functools  # 导入functools模块，提供函数式编程的支持
import warnings  # 导入warnings模块，用于处理警告信息的控制

import numpy as np  # 导入NumPy库，用于数值计算

from scipy._lib._util import MapWrapper, _FunctionWrapper  # 从scipy._lib._util模块导入特定对象

class LRUDict(collections.OrderedDict):  # 定义LRU缓存字典类，继承自collections.OrderedDict
    def __init__(self, max_size):  # 初始化方法，指定最大缓存大小
        self.__max_size = max_size  # 设置最大缓存大小的私有属性

    def __setitem__(self, key, value):  # 自定义__setitem__方法，用于设置键值对
        existing_key = (key in self)  # 检查是否存在相同的键
        super().__setitem__(key, value)  # 调用父类(OrderedDict)的__setitem__方法设置键值对
        if existing_key:  # 如果已存在相同的键
            self.move_to_end(key)  # 将该键移动到字典的尾部（最近访问的位置）
        elif len(self) > self.__max_size:  # 如果字典超过了最大缓存大小
            self.popitem(last=False)  # 弹出字典中第一个插入的元素（最旧的元素）

class SemiInfiniteFunc:  # 定义半无限函数类
    """
    Argument transform from (start, +-oo) to (0, 1)
    """
    def __init__(self, func, start, infty):  # 初始化方法，接收函数、起始点和无穷远处的符号
        self._func = func  # 将传入的函数保存为属性
        self._start = start  # 设置起始点属性
        self._sgn = -1 if infty < 0 else 1  # 根据无穷远处的符号确定方向

        # Overflow threshold for the 1/t**2 factor
        self._tmin = sys.float_info.min**0.5  # 设置1/t^2 因子的溢出阈值

    def get_t(self, x):  # 定义方法，将参数x映射到t
        z = self._sgn * (x - self._start) + 1  # 计算变换后的参数z
        if z == 0:  # 如果z为零，表示点不在范围内
            return np.inf  # 返回无穷大
        return 1 / z  # 否则返回1/z

    def __call__(self, t):  # 定义__call__方法，允许对象像函数一样被调用
        if t < self._tmin:  # 如果参数t小于溢出阈值
            return 0.0  # 返回0.0
        else:
            x = self._start + self._sgn * (1 - t) / t  # 计算反向变换后的x
            f = self._func(x)  # 计算函数在x处的值
            return self._sgn * (f / t) / t  # 返回变换后的值

class DoubleInfiniteFunc:  # 定义双无限函数类
    """
    Argument transform from (-oo, oo) to (-1, 1)
    """
    def __init__(self, func):  # 初始化方法，接收一个函数作为参数
        self._func = func  # 将传入的函数保存为属性

        # Overflow threshold for the 1/t**2 factor
        self._tmin = sys.float_info.min**0.5  # 设置1/t^2 因子的溢出阈值

    def get_t(self, x):  # 定义方法，将参数x映射到t
        s = -1 if x < 0 else 1  # 确定符号s
        return s / (abs(x) + 1)  # 计算映射后的t值

    def __call__(self, t):  # 定义__call__方法，允许对象像函数一样被调用
        if abs(t) < self._tmin:  # 如果参数t的绝对值小于溢出阈值
            return 0.0  # 返回0.0
        else:
            x = (1 - abs(t)) / t  # 计算反向变换后的x
            f = self._func(x)  # 计算函数在x处的值
            return (f / t) / t  # 返回变换后的值

def _max_norm(x):  # 定义函数，计算向量x的最大范数
    return np.amax(abs(x))  # 返回x的绝对值的最大值

def _get_sizeof(obj):  # 定义函数，获取对象obj的大小
    try:
        return sys.getsizeof(obj)  # 尝试获取对象的大小
    except TypeError:
        # occurs on pypy
        if hasattr(obj, '__sizeof__'):  # 如果对象有__sizeof__方法
            return int(obj.__sizeof__())  # 返回对象的__sizeof__方法返回值的整数形式
        return 64  # 否则返回默认大小64

class _Bunch:  # 定义_Bunch类，实现一种简单的批量属性赋值方式
    def __init__(self, **kwargs):  # 初始化方法，接收任意关键字参数
        self.__keys = kwargs.keys()  # 将所有关键字的名称保存为私有属性
        self.__dict__.update(**kwargs)  # 使用关键字参数更新对象的属性字典

    def __repr__(self):  # 定义__repr__方法，返回对象的字符串表示
        return "_Bunch({})".format(", ".join(f"{k}={repr(self.__dict__[k])}"
                                             for k in self.__keys))  # 返回对象的格式化字符串表示

def quad_vec(f, a, b, epsabs=1e-200, epsrel=1e-8, norm='2', cache_size=100e6,
             limit=10000, workers=1, points=None, quadrature=None, full_output=False,
             *, args=()):  # 定义quad_vec函数，实现向量值函数的自适应积分
    r"""Adaptive integration of a vector-valued function.

    Parameters
    ----------
    f : callable
        Vector-valued function f(x) to integrate.
    a : float
        Initial point.
    b : float
        Final point.
    epsabs : float, optional
        Absolute tolerance.
    epsrel : float, optional
        Relative tolerance.
    # norm : {'max', '2'}, optional
    #     Vector norm to use for error estimation.
    norm : {'max', '2'}, optional
        # 用于误差估计的向量范数。
    
    # cache_size : int, optional
    #     Number of bytes to use for memoization.
    cache_size : int, optional
        # 用于记忆化的字节大小。
    
    # limit : float or int, optional
    #     An upper bound on the number of subintervals used in the adaptive
    #     algorithm.
    limit : float or int, optional
        # 自适应算法中使用的子区间数量的上限。
    
    # workers : int or map-like callable, optional
    #     If `workers` is an integer, part of the computation is done in
    #     parallel subdivided to this many tasks (using
    #     :class:`python:multiprocessing.pool.Pool`).
    #     Supply `-1` to use all cores available to the Process.
    #     Alternatively, supply a map-like callable, such as
    #     :meth:`python:multiprocessing.pool.Pool.map` for evaluating the
    #     population in parallel.
    #     This evaluation is carried out as ``workers(func, iterable)``.
    workers : int or map-like callable, optional
        # 如果 `workers` 是整数，则部分计算并行化为这么多任务（使用
        # :class:`python:multiprocessing.pool.Pool`）。
        # 提供 `-1` 表示使用所有可用于进程的核心。
        # 或者，提供类似映射的可调用对象，例如
        # :meth:`python:multiprocessing.pool.Pool.map` 以并行评估群体。
        # 此评估执行为 ``workers(func, iterable)``。
    
    # points : list, optional
    #     List of additional breakpoints.
    points : list, optional
        # 额外的断点列表。
    
    # quadrature : {'gk21', 'gk15', 'trapezoid'}, optional
    #     Quadrature rule to use on subintervals.
    #     Options: 'gk21' (Gauss-Kronrod 21-point rule),
    #     'gk15' (Gauss-Kronrod 15-point rule),
    #     'trapezoid' (composite trapezoid rule).
    #     Default: 'gk21' for finite intervals and 'gk15' for (semi-)infinite
    quadrature : {'gk21', 'gk15', 'trapezoid'}, optional
        # 在子区间上使用的数值积分规则。
        # 选项：'gk21'（高斯-克朗罗德21点法则），
        # 'gk15'（高斯-克朗罗德15点法则），
        # 'trapezoid'（复合梯形法则）。
        # 默认为有限区间使用 'gk21'，（半）无限区间使用 'gk15'。
    
    # full_output : bool, optional
    #     Return an additional ``info`` dictionary.
    full_output : bool, optional
        # 返回额外的 ``info`` 字典。
    
    # args : tuple, optional
    #     Extra arguments to pass to function, if any.
    args : tuple, optional
        # 如果有的话，传递给函数的额外参数。
        
        # .. versionadded:: 1.8.0
        # .. versionadded:: 1.8.0
    
    # Returns
    # -------
    # res : {float, array-like}
    #     Estimate for the result
    # err : float
    #     Error estimate for the result in the given norm
    # info : dict
    #     Returned only when ``full_output=True``.
    #     Info dictionary. Is an object with the attributes:
    #
    #         success : bool
    #             Whether integration reached target precision.
    #         status : int
    #             Indicator for convergence, success (0),
    #             failure (1), and failure due to rounding error (2).
    #         neval : int
    #             Number of function evaluations.
    #         intervals : ndarray, shape (num_intervals, 2)
    #             Start and end points of subdivision intervals.
    #         integrals : ndarray, shape (num_intervals, ...)
    #             Integral for each interval.
    #             Note that at most ``cache_size`` values are recorded,
    #             and the array may contains *nan* for missing items.
    #         errors : ndarray, shape (num_intervals,)
    #             Estimated integration error for each interval.
    #
    # Notes
    # -----
    # The algorithm mainly follows the implementation of QUADPACK's
    # DQAG* algorithms, implementing global error control and adaptive
    # subdivision.
    #
    # The algorithm here has some differences to the QUADPACK approach:
    #
    # Instead of subdividing one interval at a time, the algorithm
    # subdivides N intervals with largest errors at once. This enables
    # (partial) parallelization of the integration.
    #
    # The logic of subdividing "next largest" intervals first is then
    #
    #     .. versionadded:: 1.8.0
    #     .. versionadded:: 1.8.0
    # 将参数 a 和 b 转换为浮点数
    a = float(a)
    b = float(b)

    # 如果有额外的参数 args，则将其封装成元组（如果尚未是元组的话）
    if args:
        if not isinstance(args, tuple):
            args = (args,)

        # 创建一个函数包装器，以便能够使用 map 和 Pool.map
        f = _FunctionWrapper(f, args)

    # 使用简单的转换处理无穷区间的积分
    kwargs = dict(epsabs=epsabs,
                  epsrel=epsrel,
                  norm=norm,
                  cache_size=cache_size,
                  limit=limit,
                  workers=workers,
                  points=points,
                  quadrature='gk15' if quadrature is None else quadrature,
                  full_output=full_output)
    
    # 如果 a 有限而 b 为无穷，则创建半无限函数对象 f2，并进行积分计算
    if np.isfinite(a) and np.isinf(b):
        f2 = SemiInfiniteFunc(f, start=a, infty=b)
        # 如果指定了 points，则将其转换为半无限函数对象 f2 的时间点
        if points is not None:
            kwargs['points'] = tuple(f2.get_t(xp) for xp in points)
        return quad_vec(f2, 0, 1, **kwargs)
    
    # 如果 b 有限而 a 为无穷，则创建半无限函数对象 f2，并进行积分计算
    elif np.isfinite(b) and np.isinf(a):
        f2 = SemiInfiniteFunc(f, start=b, infty=a)
        # 如果指定了 points，则将其转换为半无限函数对象 f2 的时间点
        if points is not None:
            kwargs['points'] = tuple(f2.get_t(xp) for xp in points)
        # 对积分结果进行修正，返回负值
        res = quad_vec(f2, 0, 1, **kwargs)
        return (-res[0],) + res[1:]
    elif np.isinf(a) and np.isinf(b):
        # 如果 a 和 b 都是无穷大，则确定符号为 -1（如果 b 小于 a）或者 1（其他情况）
        sgn = -1 if b < a else 1

        # 创建一个 DoubleInfiniteFunc 对象 f2，用于处理双无穷积分的函数 f
        f2 = DoubleInfiniteFunc(f)

        # 如果给定了 points，则将其转换为对应于 f2 在各点处的 t 值，作为 kwargs 的 'points' 参数
        if points is not None:
            kwargs['points'] = (0,) + tuple(f2.get_t(xp) for xp in points)
        else:
            kwargs['points'] = (0,)

        # 根据 a 和 b 的关系选择积分方法，并计算积分结果
        if a != b:
            res = quad_vec(f2, -1, 1, **kwargs)
        else:
            res = quad_vec(f2, 1, 1, **kwargs)

        # 返回积分结果，第一个元素乘以 sgn 表示积分的符号
        return (res[0]*sgn,) + res[1:]

    elif not (np.isfinite(a) and np.isfinite(b)):
        # 如果 a 或者 b 不是有限值，则抛出 ValueError 异常
        raise ValueError(f"invalid integration bounds a={a}, b={b}")

    # 定义不同范数计算函数的字典
    norm_funcs = {
        None: _max_norm,       # 默认为最大范数
        'max': _max_norm,      # 'max' 也使用最大范数
        '2': np.linalg.norm    # '2' 使用 numpy 中的 2-范数
    }

    # 根据传入的 norm 参数选择相应的范数计算函数
    if callable(norm):
        norm_func = norm
    else:
        norm_func = norm_funcs[norm]

    # 并行计算的线程数
    parallel_count = 128
    # 初始积分区间的最小间隔数
    min_intervals = 2

    try:
        # 根据 quadrature 参数选择相应的积分方法函数
        _quadrature = {None: _quadrature_gk21,
                       'gk21': _quadrature_gk21,
                       'gk15': _quadrature_gk15,
                       'trapz': _quadrature_trapezoid,  # 'trapz' 的别名，向后兼容性
                       'trapezoid': _quadrature_trapezoid}[quadrature]
    except KeyError as e:
        # 如果 quadrature 参数未知，则抛出 ValueError 异常
        raise ValueError(f"unknown quadrature {quadrature!r}") from e

    # 如果 quadrature 为 "trapz"，则发出关于其弃用的警告信息
    if quadrature == "trapz":
        msg = ("`quadrature='trapz'` is deprecated in favour of "
               "`quadrature='trapezoid' and will raise an error from SciPy 1.16.0 "
               "onwards.")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    # 初始积分区间的设置
    if points is None:
        initial_intervals = [(a, b)]
    else:
        prev = a
        initial_intervals = []
        for p in sorted(points):
            p = float(p)
            # 如果点 p 不在 (a, b) 范围内或者与前一个点相同，则跳过
            if not (a < p < b) or p == prev:
                continue
            initial_intervals.append((prev, p))
            prev = p
        initial_intervals.append((prev, b))

    # 初始化全局积分结果、全局误差、舍入误差、区间缓存和区间列表
    global_integral = None
    global_error = None
    rounding_error = None
    interval_cache = None
    intervals = []
    neval = 0

    # 遍历初始积分区间
    for x1, x2 in initial_intervals:
        # 使用指定的积分方法计算区间 (x1, x2) 上的积分结果、误差和舍入误差
        ig, err, rnd = _quadrature(x1, x2, f, norm_func)
        # 增加积分函数的评估次数到 neval 中
        neval += _quadrature.num_eval

        if global_integral is None:
            if isinstance(ig, (float, complex)):
                # 对标量值进行特殊化处理，如果使用的是最大范数或 2-范数的话
                if norm_func in (_max_norm, np.linalg.norm):
                    norm_func = abs

            # 初始化全局积分结果、全局误差和舍入误差
            global_integral = ig
            global_error = float(err)
            rounding_error = float(rnd)

            # 根据缓存大小设置区间缓存
            cache_count = cache_size // _get_sizeof(ig)
            interval_cache = LRUDict(cache_count)
        else:
            # 累加全局积分结果、全局误差和舍入误差
            global_integral += ig
            global_error += err
            rounding_error += rnd

        # 将 (x1, x2) 区间的积分结果 ig 存入区间缓存中
        interval_cache[(x1, x2)] = copy.copy(ig)
        # 将 (-err, x1, x2) 的元组添加到区间列表中，用于后续的堆排序
        intervals.append((-err, x1, x2))

    # 对区间列表进行堆排序
    heapq.heapify(intervals)

    # 定义收敛状态常量
    CONVERGED = 0
    NOT_CONVERGED = 1
    ROUNDING_ERROR = 2
    NOT_A_NUMBER = 3
    # 定义状态信息字典，将不同状态与对应信息文本关联起来
    status_msg = {
        CONVERGED: "Target precision reached.",
        NOT_CONVERGED: "Target precision not reached.",
        ROUNDING_ERROR: "Target precision could not be reached due to rounding error.",
        NOT_A_NUMBER: "Non-finite values encountered."
    }

    # 使用 MapWrapper 并行处理任务
    with MapWrapper(workers) as mapwrapper:
        # 初始化迭代器状态为 NOT_CONVERGED
        ier = NOT_CONVERGED

        # 当仍有未处理的区间且未达到限制时继续循环
        while intervals and len(intervals) < limit:
            # 计算容差值，选择具有最大误差的区间进行细分
            tol = max(epsabs, epsrel * norm_func(global_integral))

            # 准备要处理的区间列表和误差总和
            to_process = []
            err_sum = 0

            # 遍历并选择要并行处理的区间
            for j in range(parallel_count):
                if not intervals:
                    break

                if j > 0 and err_sum > global_error - tol / 8:
                    # 避免不必要的并行分割
                    break

                # 弹出堆中具有最大负误差的区间
                interval = heapq.heappop(intervals)

                neg_old_err, a, b = interval
                old_int = interval_cache.pop((a, b), None)
                # 将区间信息和处理函数添加到处理列表中
                to_process.append(
                    ((-neg_old_err, a, b, old_int), f, norm_func, _quadrature)
                )
                err_sum += -neg_old_err

            # 并行细分区间
            for parts in mapwrapper(_subdivide_interval, to_process):
                dint, derr, dround_err, subint, dneval = parts
                # 更新统计数据
                neval += dneval
                global_integral += dint
                global_error += derr
                rounding_error += dround_err
                # 将细分后的子区间重新推入堆中
                for x in subint:
                    x1, x2, ig, err = x
                    interval_cache[(x1, x2)] = ig
                    heapq.heappush(intervals, (-err, x1, x2))

            # 终止条件检查
            if len(intervals) >= min_intervals:
                tol = max(epsabs, epsrel * norm_func(global_integral))
                # 如果全局误差小于容差的八分之一，认为收敛
                if global_error < tol / 8:
                    ier = CONVERGED
                    break
                # 如果全局误差小于四舍五入误差，认为由于舍入误差未达到目标精度
                if global_error < rounding_error:
                    ier = ROUNDING_ERROR
                    break

            # 如果全局误差或舍入误差包含非有限值，认为遇到非数值
            if not (np.isfinite(global_error) and np.isfinite(rounding_error)):
                ier = NOT_A_NUMBER
                break

    # 计算最终的积分结果和误差
    res = global_integral
    err = global_error + rounding_error

    # 如果需要完整输出，则生成详细信息
    if full_output:
        res_arr = np.asarray(res)
        dummy = np.full(res_arr.shape, np.nan, dtype=res_arr.dtype)
        # 从区间列表中获取积分值和误差信息
        integrals = np.array([interval_cache.get((z[1], z[2]), dummy)
                              for z in intervals], dtype=res_arr.dtype)
        errors = np.array([-z[0] for z in intervals])
        intervals = np.array([[z[1], z[2]] for z in intervals])

        # 组装详细信息对象
        info = _Bunch(neval=neval,
                      success=(ier == CONVERGED),
                      status=ier,
                      message=status_msg[ier],
                      intervals=intervals,
                      integrals=integrals,
                      errors=errors)
        return (res, err, info)
    else:
        # 如果条件不满足，则返回元组 (res, err)
        return (res, err)
def _subdivide_interval(args):
    interval, f, norm_func, _quadrature = args
    old_err, a, b, old_int = interval

    c = 0.5 * (a + b)

    # Left-hand side
    # 如果 _quadrature 对象有 cache_size 属性且大于 0，则使用 functools.lru_cache 对函数 f 进行缓存
    if getattr(_quadrature, 'cache_size', 0) > 0:
        f = functools.lru_cache(_quadrature.cache_size)(f)

    # 计算左半部分积分
    s1, err1, round1 = _quadrature(a, c, f, norm_func)
    dneval = _quadrature.num_eval
    # 计算右半部分积分
    s2, err2, round2 = _quadrature(c, b, f, norm_func)
    dneval += _quadrature.num_eval
    # 如果之前没有计算整个区间的积分，则计算整个区间的积分并增加评估次数
    if old_int is None:
        old_int, _, _ = _quadrature(a, b, f, norm_func)
        dneval += _quadrature.num_eval

    # 如果 _quadrature 对象有 cache_size 属性且大于 0，则更新 dneval 为缓存未命中次数
    if getattr(_quadrature, 'cache_size', 0) > 0:
        dneval = f.cache_info().misses

    # 计算增量
    dint = s1 + s2 - old_int
    derr = err1 + err2 - old_err
    dround_err = round1 + round2

    # 子区间信息
    subintervals = ((a, c, s1, err1), (c, b, s2, err2))
    return dint, derr, dround_err, subintervals, dneval


def _quadrature_trapezoid(x1, x2, f, norm_func):
    """
    Composite trapezoid quadrature
    """
    x3 = 0.5*(x1 + x2)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    # 使用复合梯形公式计算积分和误差
    s2 = 0.25 * (x2 - x1) * (f1 + 2*f3 + f2)

    round_err = 0.25 * abs(x2 - x1) * (float(norm_func(f1))
                                       + 2*float(norm_func(f3))
                                       + float(norm_func(f2))) * 2e-16

    s1 = 0.5 * (x2 - x1) * (f1 + f2)
    err = 1/3 * float(norm_func(s1 - s2))
    return s2, err, round_err


_quadrature_trapezoid.cache_size = 3 * 3
_quadrature_trapezoid.num_eval = 3


def _quadrature_gk(a, b, f, norm_func, x, w, v):
    """
    Generic Gauss-Kronrod quadrature
    """

    fv = [0.0]*len(x)

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)

    # Gauss-Kronrod 方法
    s_k = 0.0
    s_k_abs = 0.0
    for i in range(len(x)):
        ff = f(c + h*x[i])
        fv[i] = ff

        vv = v[i]

        # 计算积分值 s_k 和 |f(x)| 的积分值 s_k_abs
        s_k += vv * ff
        s_k_abs += vv * abs(ff)

    # Gauss 方法
    s_g = 0.0
    for i in range(len(w)):
        s_g += w[i] * fv[2*i + 1]

    # 计算 |f(x) - y0| 的积分值 s_k_dabs
    s_k_dabs = 0.0
    y0 = s_k / 2.0
    for i in range(len(x)):
        s_k_dabs += v[i] * abs(fv[i] - y0)

    # 使用与 quadpack 类似的误差估计
    err = float(norm_func((s_k - s_g) * h))
    dabs = float(norm_func(s_k_dabs * h))
    if dabs != 0 and err != 0:
        err = dabs * min(1.0, (200 * err / dabs)**1.5)

    eps = sys.float_info.epsilon
    round_err = float(norm_func(50 * eps * h * s_k_abs))

    if round_err > sys.float_info.min:
        err = max(err, round_err)

    return h * s_k, err, round_err


def _quadrature_gk21(a, b, f, norm_func):
    """
    Gauss-Kronrod 21 quadrature with error estimate
    """
    # Gauss-Kronrod 21 点
    # 省略部分，未完待续
    x = (0.995657163025808080735527280689003,
         0.973906528517171720077964012084452,
         0.930157491355708226001207180059508,
         0.865063366688984510732096688423493,
         0.780817726586416897063717578345042,
         0.679409568299024406234327365114874,
         0.562757134668604683339000099272694,
         0.433395394129247190799265943165784,
         0.294392862701460198131126603103866,
         0.148874338981631210884826001129720,
         0,
         -0.148874338981631210884826001129720,
         -0.294392862701460198131126603103866,
         -0.433395394129247190799265943165784,
         -0.562757134668604683339000099272694,
         -0.679409568299024406234327365114874,
         -0.780817726586416897063717578345042,
         -0.865063366688984510732096688423493,
         -0.930157491355708226001207180059508,
         -0.973906528517171720077964012084452,
         -0.995657163025808080735527280689003)

    # 10-point weights
    w = (0.066671344308688137593568809893332,
         0.149451349150580593145776339657697,
         0.219086362515982043995534934228163,
         0.269266719309996355091226921569469,
         0.295524224714752870173892994651338,
         0.295524224714752870173892994651338,
         0.269266719309996355091226921569469,
         0.219086362515982043995534934228163,
         0.149451349150580593145776339657697,
         0.066671344308688137593568809893332)

    # 21-point weights
    v = (0.011694638867371874278064396062192,
         0.032558162307964727478818972459390,
         0.054755896574351996031381300244580,
         0.075039674810919952767043140916190,
         0.093125454583697605535065465083366,
         0.109387158802297641899210590325805,
         0.123491976262065851077958109831074,
         0.134709217311473325928054001771707,
         0.142775938577060080797094273138717,
         0.147739104901338491374841515972068,
         0.149445554002916905664936468389821,
         0.147739104901338491374841515972068,
         0.142775938577060080797094273138717,
         0.134709217311473325928054001771707,
         0.123491976262065851077958109831074,
         0.109387158802297641899210590325805,
         0.093125454583697605535065465083366,
         0.075039674810919952767043140916190,
         0.054755896574351996031381300244580,
         0.032558162307964727478818972459390,
         0.011694638867371874278064396062192)

    # 调用某个函数进行数值积分计算，返回计算结果
    return _quadrature_gk(a, b, f, norm_func, x, w, v)
# 设置 Gauss-Kronrod 21 点求积法的评估次数为 21
_quadrature_gk21.num_eval = 21

# 定义 Gauss-Kronrod 15 点求积法，带误差估计
def _quadrature_gk15(a, b, f, norm_func):
    """
    Gauss-Kronrod 15 quadrature with error estimate
    """
    # Gauss-Kronrod 点的位置
    x = (0.991455371120812639206854697526329,
         0.949107912342758524526189684047851,
         0.864864423359769072789712788640926,
         0.741531185599394439863864773280788,
         0.586087235467691130294144838258730,
         0.405845151377397166906606412076961,
         0.207784955007898467600689403773245,
         0.000000000000000000000000000000000,
         -0.207784955007898467600689403773245,
         -0.405845151377397166906606412076961,
         -0.586087235467691130294144838258730,
         -0.741531185599394439863864773280788,
         -0.864864423359769072789712788640926,
         -0.949107912342758524526189684047851,
         -0.991455371120812639206854697526329)

    # 7 点求积的权重
    w = (0.129484966168869693270611432679082,
         0.279705391489276667901467771423780,
         0.381830050505118944950369775488975,
         0.417959183673469387755102040816327,
         0.381830050505118944950369775488975,
         0.279705391489276667901467771423780,
         0.129484966168869693270611432679082)

    # 15 点求积的权重
    v = (0.022935322010529224963732008058970,
         0.063092092629978553290700663189204,
         0.104790010322250183839876322541518,
         0.140653259715525918745189590510238,
         0.169004726639267902826583426598550,
         0.190350578064785409913256402421014,
         0.204432940075298892414161999234649,
         0.209482141084727828012999174891714,
         0.204432940075298892414161999234649,
         0.190350578064785409913256402421014,
         0.169004726639267902826583426598550,
         0.140653259715525918745189590510238,
         0.104790010322250183839876322541518,
         0.063092092629978553290700663189204,
         0.022935322010529224963732008058970)

    # 调用内部的 Gauss-Kronrod 求积函数 _quadrature_gk 进行计算
    return _quadrature_gk(a, b, f, norm_func, x, w, v)

# 设置 Gauss-Kronrod 15 点求积法的评估次数为 15
_quadrature_gk15.num_eval = 15
```