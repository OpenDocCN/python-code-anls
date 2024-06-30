# `D:\src\scipysrc\scipy\scipy\optimize\_tstutils.py`

```
# Parameters used in test and benchmark methods.
# Collections of test cases suitable for testing 1-D root-finders

# 'original': The original benchmarking functions.
# Real-valued functions of real-valued inputs on an interval with a zero.
# f1, .., f3 are continuous and infinitely differentiable
# f4 has a left- and right- discontinuity at the root
# f5 has a root at 1 replacing a 1st order pole
# f6 is randomly positive on one side of the root,
# randomly negative on the other.
# f4 - f6 are not continuous at the root.
'original': [
    {'f': f1, 'fprime': f1_fp, 'fprime2': f1_fpp},
    {'f': f2, 'fprime': f2_fp, 'fprime2': f2_fpp},
    {'f': f3, 'fprime': f3_fp, 'fprime2': f3_fpp},
],

# 'aps': The test problems in the 1995 paper
# TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions"
# by Alefeld, Potra and Shi. Real-valued functions of
# real-valued inputs on an interval with a zero.
# Suitable for methods which start with an enclosing interval, and
# derivatives up to 2nd order.
'aps': [],

# 'complex': Some complex-valued functions of complex-valued inputs.
# No enclosing bracket is provided.
# Suitable for methods which use one or more starting values, and
# derivatives up to 2nd order.
'complex': [],

# The test cases are provided as a list of dictionaries. The dictionary
# keys will be a subset of:
# ["f", "fprime", "fprime2", "args", "bracket", "smoothness",
# "a", "b", "x0", "x1", "root", "ID"]

# Sources:
# [1] Alefeld, G. E. and Potra, F. A. and Shi, Yixun,
# "Algorithm 748: Enclosing Zeros of Continuous Functions",
# ACM Trans. Math. Softw. Volume 221(1995)
# doi = {10.1145/210089.210111},
# [2] Chandrupatla, Tirupathi R. "A new hybrid quadratic/bisection algorithm
# for finding the zero of a nonlinear function without using derivatives."
# Advances in Engineering Software 28.3 (1997): 145-149.

from random import random

import numpy as np

from scipy.optimize import _zeros_py as cc
from scipy._lib._array_api import array_namespace

# "description" refers to the original functions
description = """
f2 is a symmetric parabola, x**2 - 1
f3 is a quartic polynomial with large hump in interval
f4 is step function with a discontinuity at 1
f5 is a hyperbola with vertical asymptote at 1
f6 has random values positive to left of 1, negative to right

Of course, these are not real problems. They just test how the
'good' solvers behave in bad circumstances where bisection is
really the best. A good solver should not be much worse than
bisection in such circumstance, while being faster for smooth
monotone sorts of functions.
"""


def f1(x):
    r"""f1 is a quadratic with roots at 0 and 1"""
    return x * (x - 1.)


def f1_fp(x):
    return 2 * x - 1


def f1_fpp(x):
    return 2


def f2(x):
    r"""f2 is a symmetric parabola, x**2 - 1"""
    return x**2 - 1


def f2_fp(x):
    return 2 * x


def f2_fpp(x):
    return 2


def f3(x):
    r"""A quartic with roots at 0, 1, 2 and 3"""
    return x * (x - 1.) * (x - 2.) * (x - 3.)  # x**4 - 6x**3 + 11x**2 - 6x


def f3_fp(x):
    return 4 * x**3 - 18 * x**2 + 22 * x - 6


def f3_fpp(x):
    return 12 * x**2 - 36 * x + 22
    # 计算并返回一个二次方程的结果，形式为 12 * x^2 - 36 * x + 22
    return 12 * x**2 - 36 * x + 22
# 定义函数 f4(x)，表示一个分段线性函数，在 x=1 处左右不连续，是其根
def f4(x):
    # 如果 x 大于 1，返回 1.0 + 0.1 * x
    if x > 1:
        return 1.0 + .1 * x
    # 如果 x 小于 1，返回 -1.0 + 0.1 * x
    if x < 1:
        return -1.0 + .1 * x
    # 如果 x 等于 1，返回 0
    return 0


# 定义函数 f5(x)，表示一个双曲线，在 x=1 处有极点，但被替换为 0，在根处不连续
def f5(x):
    # 如果 x 不等于 1，返回 1.0 / (1. - x)
    if x != 1:
        return 1.0 / (1. - x)
    # 如果 x 等于 1，返回 0
    return 0


# f6(x) 返回随机值。没有记忆化处理，对相同的 x 调用两次会返回不同的值，因此是一个“具有随机值”的函数，而不是“随机值的函数”
_f6_cache = {}
# 定义函数 f6(x)，通过缓存机制实现记忆化
def f6(x):
    # 尝试从缓存中获取 x 对应的值
    v = _f6_cache.get(x, None)
    # 如果缓存中没有 x 对应的值
    if v is None:
        # 如果 x 大于 1，v 赋值为随机数
        if x > 1:
            v = random()
        # 如果 x 小于 1，v 赋值为负随机数
        elif x < 1:
            v = -random()
        # 如果 x 等于 1，v 赋值为 0
        else:
            v = 0
        # 将计算结果存入缓存
        _f6_cache[x] = v
    # 返回计算结果
    return v


# 每个原始测试案例包含：
# - 一个函数及其两个导数，
# - 额外参数，
# - 包含根的区间，
# - 在该区间上的可微性顺序，
# - 对于不需要区间的方法的起始值，
# - 根（在区间内），
# - 测试案例的标识符
# 原始测试案例的键列表
_ORIGINAL_TESTS_KEYS = [
    "f", "fprime", "fprime2", "args", "bracket", "smoothness", "x0", "root", "ID"
]
# 原始测试案例的列表，每个测试案例是一个列表，与键列表一一对应
_ORIGINAL_TESTS = [
    [f1, f1_fp, f1_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.01.00"],
    [f2, f2_fp, f2_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.02.00"],
    [f3, f3_fp, f3_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.03.00"],
    [f4, None, None, (), [0.5, np.sqrt(3)], -1, 0.6, 1.0, "original.04.00"],
    [f5, None, None, (), [0.5, np.sqrt(3)], -1, 0.6, 1.0, "original.05.00"],
    [f6, None, None, (), [0.5, np.sqrt(3)], -np.inf, 0.6, 1.0, "original.05.00"]
]
# 将原始测试案例列表转换为字典列表，每个字典对应一个测试案例
_ORIGINAL_TESTS_DICTS = [
    dict(zip(_ORIGINAL_TESTS_KEYS, testcase)) for testcase in _ORIGINAL_TESTS
]

# ##################
# "APS" 测试案例
# 函数及其出现在[1]中的测试案例


# 定义函数 aps01_f(x)，表示三角函数和多项式的简单和
def aps01_f(x):
    return np.sin(x) - x / 2


# 定义函数 aps01_fp(x)，表示 aps01_f(x) 的导数
def aps01_fp(x):
    return np.cos(x) - 1.0 / 2


# 定义函数 aps01_fpp(x)，表示 aps01_f(x) 的二阶导数
def aps01_fpp(x):
    return -np.sin(x)


# 定义函数 aps02_f(x)，表示在 x=n**2 处有极点，一阶和二阶导数在根处也接近于 0
def aps02_f(x):
    ii = np.arange(1, 21)
    return -2 * np.sum((2 * ii - 5)**2 / (x - ii**2)**3)


# 定义函数 aps02_fp(x)，表示 aps02_f(x) 的导数
def aps02_fp(x):
    ii = np.arange(1, 21)
    return 6 * np.sum((2 * ii - 5)**2 / (x - ii**2)**4)


# 定义函数 aps02_fpp(x)，表示 aps02_f(x) 的二阶导数
def aps02_fpp(x):
    ii = np.arange(1, 21)
    return 24 * np.sum((2 * ii - 5)**2 / (x - ii**2)**5)


# 定义函数 aps03_f(x, a, b)，表示在根处变化迅速的函数
def aps03_f(x, a, b):
    return a * x * np.exp(b * x)


# 定义函数 aps03_fp(x, a, b)，表示 aps03_f(x, a, b) 的导数
def aps03_fp(x, a, b):
    return a * (b * x + 1) * np.exp(b * x)


# 定义函数 aps03_fpp(x, a, b)，表示 aps03_f(x, a, b) 的二阶导数
def aps03_fpp(x, a, b):
    return a * (b * (b * x + 1) + b) * np.exp(b * x)


# 定义函数 aps04_f(x, n, a)，表示中等程度的多项式函数
def aps04_f(x, n, a):
    return x**n - a


# 定义函数 aps04_fp(x, n, a)，表示 aps04_f(x, n, a) 的导数
def aps04_fp(x, n, a):
    return n * x**(n - 1)


# 定义函数 aps04_fpp(x, n, a)，表示 aps04_f(x, n, a) 的二阶导数
def aps04_fpp(x, n, a):
    return n * (n - 1) * x**(n - 2)


# 定义函数 aps05_f(x)，待补充
def aps05_f(x):
    pass
    # 定义一个简单的三角函数
    return np.sin(x) - 1.0 / 2
def aps05_fp(x):
    # 返回 x 的余弦值
    return np.cos(x)


def aps05_fpp(x):
    # 返回 x 的负正弦值
    return -np.sin(x)


def aps06_f(x, n):
    r"""在 x=0 处从 -1 急剧变化到 1 的指数函数"""
    return 2 * x * np.exp(-n) - 2 * np.exp(-n * x) + 1


def aps06_fp(x, n):
    # 返回对于指数函数的一阶导数
    return 2 * np.exp(-n) + 2 * n * np.exp(-n * x)


def aps06_fpp(x, n):
    # 返回对于指数函数的二阶导数
    return -2 * n * n * np.exp(-n * x)


def aps07_f(x, n):
    r"""高度可调的倒置抛物线"""
    return (1 + (1 - n)**2) * x - (1 - n * x)**2


def aps07_fp(x, n):
    # 返回倒置抛物线的一阶导数
    return (1 + (1 - n)**2) + 2 * n * (1 - n * x)


def aps07_fpp(x, n):
    # 返回倒置抛物线的二阶导数
    return -2 * n * n


def aps08_f(x, n):
    r"""n 次多项式"""
    return x * x - (1 - x)**n


def aps08_fp(x, n):
    # 返回 n 次多项式的一阶导数
    return 2 * x + n * (1 - x)**(n - 1)


def aps08_fpp(x, n):
    # 返回 n 次多项式的二阶导数
    return 2 - n * (n - 1) * (1 - x)**(n - 2)


def aps09_f(x, n):
    r"""高度可调的倒置四次方程"""
    return (1 + (1 - n)**4) * x - (1 - n * x)**4


def aps09_fp(x, n):
    # 返回倒置四次方程的一阶导数
    return (1 + (1 - n)**4) + 4 * n * (1 - n * x)**3


def aps09_fpp(x, n):
    # 返回倒置四次方程的二阶导数
    return -12 * n * (1 - n * x)**2


def aps10_f(x, n):
    r"""指数函数加多项式"""
    return np.exp(-n * x) * (x - 1) + x**n


def aps10_fp(x, n):
    # 返回指数函数加多项式的一阶导数
    return np.exp(-n * x) * (-n * (x - 1) + 1) + n * x**(n - 1)


def aps10_fpp(x, n):
    # 返回指数函数加多项式的二阶导数
    return (np.exp(-n * x) * (-n * (-n * (x - 1) + 1) + -n * x)
            + n * (n - 1) * x**(n - 2))


def aps11_f(x, n):
    r"""有零点为 x=1/n 和极点为 x=0 的有理函数"""
    return (n * x - 1) / ((n - 1) * x)


def aps11_fp(x, n):
    # 返回有理函数的一阶导数
    return 1 / (n - 1) / x**2


def aps11_fpp(x, n):
    # 返回有理函数的二阶导数
    return -2 / (n - 1) / x**3


def aps12_f(x, n):
    r"""x 的 n 次根，有零点为 x=n"""
    return np.power(x, 1.0 / n) - np.power(n, 1.0 / n)


def aps12_fp(x, n):
    # 返回 x 的 n 次根的一阶导数
    return np.power(x, (1.0 - n) / n) / n


def aps12_fpp(x, n):
    # 返回 x 的 n 次根的二阶导数
    return np.power(x, (1.0 - 2 * n) / n) * (1.0 / n) * (1.0 - n) / n


_MAX_EXPABLE = np.log(np.finfo(float).max)


def aps13_f(x):
    r"""在根处所有导数都为 0 的函数"""
    if x == 0:
        return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return x / np.exp(y)


def aps13_fp(x):
    if x == 0:
        return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return (1 + 2 / x**2) / np.exp(y)


def aps13_fpp(x):
    if x == 0:
        return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return 2 * (2 - x**2) / x**5 / np.exp(y)


def aps14_f(x, n):
    r"""对于负 x 值返回 0，对于正 x 值使用三角函数和线性函数"""
    if x <= 0:
        return -n / 20.0
    return n / 20.0 * (x / 1.5 + np.sin(x) - 1)


def aps14_fp(x, n):
    if x <= 0:
        return 0
    return n / 20.0 * (1.0 / 1.5 + np.cos(x))


def aps14_fpp(x, n):
    if x <= 0:
        return 0
    return -n / 20.0 * (np.sin(x))


def aps15_f(x, n):
    r"""分段线性函数，在区间 [0, 0.002/(1+n)] 之外为常数"""
    if x < 0:
        return -0.859
    # 如果 x 大于 2 * 1e-3 / (1 + n)，则返回 np.e - 1.859
    if x > 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    # 否则，计算并返回 np.exp((n + 1) * x / 2 * 1000) - 1.859
    return np.exp((n + 1) * x / 2 * 1000) - 1.859
# 定义函数 aps15_fp，计算给定参数 x 和 n 下的函数值
def aps15_fp(x, n):
    # 检查 x 是否在指定范围内，如果不在则返回一个默认值 np.e - 1.859
    if not 0 <= x <= 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    # 如果 x 在范围内，根据给定公式计算函数值并返回
    return np.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000

# 定义函数 aps15_fpp，计算给定参数 x 和 n 下的函数值，此函数与 aps15_fp 的区别在于返回值计算方式不同
def aps15_fpp(x, n):
    # 检查 x 是否在指定范围内，如果不在则返回一个默认值 np.e - 1.859
    if not 0 <= x <= 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    # 如果 x 在范围内，根据给定公式计算函数值并返回，这里有两处相同的部分 (n + 1) / 2 * 1000
    return np.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000 * (n + 1) / 2 * 1000

# APS 测试案例，每个案例包含以下信息：
# - 函数及其两个导数函数
# - 额外参数
# - 用于寻找根的区间范围
# - 函数在此区间上的可导性
# - 对于不需要区间范围的方法的起始值 x0
# - 根的值
# - 案例标识符
#
# 算法 748 是一种包含区间的算法，因此每个测试案例都提供了一个区间范围
# 牛顿和 Halley 方法需要一个起始点 x0，该点通常选择在区间的中间位置，除非这样做会使问题过于简单。
_APS_TESTS_KEYS = [
    "f", "fprime", "fprime2", "args", "bracket", "smoothness", "x0", "root", "ID"
]

# APS 测试案例列表，每个列表项包含一个 APS 测试案例的详细信息
_APS_TESTS = [
    [aps01_f, aps01_fp, aps01_fpp, (), [np.pi / 2, np.pi], np.inf,
     3, 1.89549426703398094e+00, "aps.01.00"],
    [aps02_f, aps02_fp, aps02_fpp, (), [1 + 1e-9, 4 - 1e-9], np.inf,
     2, 3.02291534727305677e+00, "aps.02.00"],
    [aps02_f, aps02_fp, aps02_fpp, (), [4 + 1e-9, 9 - 1e-9], np.inf,
     5, 6.68375356080807848e+00, "aps.02.01"],
    [aps02_f, aps02_fp, aps02_fpp, (), [9 + 1e-9, 16 - 1e-9], np.inf,
     10, 1.12387016550022114e+01, "aps.02.02"],
    [aps02_f, aps02_fp, aps02_fpp, (), [16 + 1e-9, 25 - 1e-9], np.inf,
     17, 1.96760000806234103e+01, "aps.02.03"],
    [aps02_f, aps02_fp, aps02_fpp, (), [25 + 1e-9, 36 - 1e-9], np.inf,
     26, 2.98282273265047557e+01, "aps.02.04"],
    [aps02_f, aps02_fp, aps02_fpp, (), [36 + 1e-9, 49 - 1e-9], np.inf,
     37, 4.19061161952894139e+01, "aps.02.05"],
    [aps02_f, aps02_fp, aps02_fpp, (), [49 + 1e-9, 64 - 1e-9], np.inf,
     50, 5.59535958001430913e+01, "aps.02.06"],
    [aps02_f, aps02_fp, aps02_fpp, (), [64 + 1e-9, 81 - 1e-9], np.inf,
     65, 7.19856655865877997e+01, "aps.02.07"],
    [aps02_f, aps02_fp, aps02_fpp, (), [81 + 1e-9, 100 - 1e-9], np.inf,
     82, 9.00088685391666701e+01, "aps.02.08"],
    [aps02_f, aps02_fp, aps02_fpp, (), [100 + 1e-9, 121 - 1e-9], np.inf,
     101, 1.10026532748330197e+02, "aps.02.09"],
    [aps03_f, aps03_fp, aps03_fpp, (-40, -1), [-9, 31], np.inf,
     -2, 0, "aps.03.00"],
    [aps03_f, aps03_fp, aps03_fpp, (-100, -2), [-9, 31], np.inf,
     -2, 0, "aps.03.01"],
    [aps03_f, aps03_fp, aps03_fpp, (-200, -3), [-9, 31], np.inf,
     -2, 0, "aps.03.02"],
    [aps04_f, aps04_fp, aps04_fpp, (4, 0.2), [0, 5], np.inf,
     2.5, 6.68740304976422006e-01, "aps.04.00"],
    [aps04_f, aps04_fp, aps04_fpp, (6, 0.2), [0, 5], np.inf,
     2.5, 7.64724491331730039e-01, "aps.04.01"],
    [aps04_f, aps04_fp, aps04_fpp, (8, 0.2), [0, 5], np.inf,
     2.5, 8.17765433957942545e-01, "aps.04.02"],
    # 创建包含多个元素的列表，每个元素是一个包含参数的列表，表示不同的数据集合
    [
        # 第一个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (10, 0.2), [0, 5], np.inf,
         2.5, 8.51339922520784609e-01, "aps.04.03"],
        # 第二个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (12, 0.2), [0, 5], np.inf,
         2.5, 8.74485272221167897e-01, "aps.04.04"],
        # 第三个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (4, 1), [0, 5], np.inf,
         2.5, 1, "aps.04.05"],
        # 第四个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (6, 1), [0, 5], np.inf,
         2.5, 1, "aps.04.06"],
        # 第五个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (8, 1), [0, 5], np.inf,
         2.5, 1, "aps.04.07"],
        # 第六个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (10, 1), [0, 5], np.inf,
         2.5, 1, "aps.04.08"],
        # 第七个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (12, 1), [0, 5], np.inf,
         2.5, 1, "aps.04.09"],
        # 第八个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (8, 1), [-0.95, 4.05], np.inf,
         1.5, 1, "aps.04.10"],
        # 第九个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (10, 1), [-0.95, 4.05], np.inf,
         1.5, 1, "aps.04.11"],
        # 第十个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (12, 1), [-0.95, 4.05], np.inf,
         1.5, 1, "aps.04.12"],
        # 第十一个数据集合
        [aps04_f, aps04_fp, aps04_fpp, (14, 1), [-0.95, 4.05], np.inf,
         1.5, 1, "aps.04.13"],
        # 第十二个数据集合
        [aps05_f, aps05_fp, aps05_fpp, (), [0, 1.5], np.inf,
         1.3, np.pi / 6, "aps.05.00"],
        # 第十三个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (1,), [0, 1], np.inf,
         0.5, 4.22477709641236709e-01, "aps.06.00"],
        # 第十四个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (2,), [0, 1], np.inf,
         0.5, 3.06699410483203705e-01, "aps.06.01"],
        # 第十五个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (3,), [0, 1], np.inf,
         0.5, 2.23705457654662959e-01, "aps.06.02"],
        # 第十六个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (4,), [0, 1], np.inf,
         0.5, 1.71719147519508369e-01, "aps.06.03"],
        # 第十七个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (5,), [0, 1], np.inf,
         0.4, 1.38257155056824066e-01, "aps.06.04"],
        # 第十八个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (20,), [0, 1], np.inf,
         0.1, 3.46573590208538521e-02, "aps.06.05"],
        # 第十九个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (40,), [0, 1], np.inf,
         5e-02, 1.73286795139986315e-02, "aps.06.06"],
        # 第二十个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (60,), [0, 1], np.inf,
         1.0 / 30, 1.15524530093324210e-02, "aps.06.07"],
        # 第二十一个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (80,), [0, 1], np.inf,
         2.5e-02, 8.66433975699931573e-03, "aps.06.08"],
        # 第二十二个数据集合
        [aps06_f, aps06_fp, aps06_fpp, (100,), [0, 1], np.inf,
         2e-02, 6.93147180559945415e-03, "aps.06.09"],
        # 第二十三个数据集合
        [aps07_f, aps07_fp, aps07_fpp, (5,), [0, 1], np.inf,
         0.4, 3.84025518406218985e-02, "aps.07.00"],
        # 第二十四个数据集合
        [aps07_f, aps07_fp, aps07_fpp, (10,), [0, 1], np.inf,
         0.4, 9.90000999800049949e-03, "aps.07.01"],
        # 第二十五个数据集合
        [aps07_f, aps07_fp, aps07_fpp, (20,), [0, 1], np.inf,
         0.4, 2.49375003906201174e-03, "aps.07.02"],
        # 第二十六个数据集合
        [aps08_f, aps08_fp, aps08_fpp, (2,), [0, 1], np.inf,
         0.9, 0.5, "aps.08.00"],
        # 第二十七个数据集合
        [aps08_f, aps08_fp, aps08_fpp, (5,), [0, 1], np.inf,
         0.9, 3.45954815848242059e-01, "aps.08.01"],
        # 第二十八个数据集合
        [aps08_f, aps08_fp, aps08_fpp, (10,), [0, 1], np.inf,
         0.9, 2.45122333753307220e-01, "aps.08.02"],
        # 第二十九个数据集合
        [aps08_f, aps08_fp, aps08_fpp, (15,), [0, 1], np.inf,
         0.9, 1.95547623536565629e-01, "aps.08.03"],
    ]
    # 定义一个包含多个参数的列表，每个元素都是一个参数组合的元组
    [aps08_f, aps08_fp, aps08_fpp, (20,), [0, 1], np.inf,
     0.9, 1.64920957276440960e-01, "aps.08.04"],
    # 同上，定义另一个参数组合的列表
    [aps09_f, aps09_fp, aps09_fpp, (1,), [0, 1], np.inf,
     0.5, 2.75508040999484394e-01, "aps.09.00"],
    # 同上，定义另一个参数组合的列表
    [aps09_f, aps09_fp, aps09_fpp, (2,), [0, 1], np.inf,
     0.5, 1.37754020499742197e-01, "aps.09.01"],
    # 同上，定义另一个参数组合的列表
    [aps09_f, aps09_fp, aps09_fpp, (4,), [0, 1], np.inf,
     0.5, 1.03052837781564422e-02, "aps.09.02"],
    # 同上，定义另一个参数组合的列表
    [aps09_f, aps09_fp, aps09_fpp, (5,), [0, 1], np.inf,
     0.5, 3.61710817890406339e-03, "aps.09.03"],
    # 同上，定义另一个参数组合的列表
    [aps09_f, aps09_fp, aps09_fpp, (8,), [0, 1], np.inf,
     0.5, 4.10872918496395375e-04, "aps.09.04"],
    # 同上，定义另一个参数组合的列表
    [aps09_f, aps09_fp, aps09_fpp, (15,), [0, 1], np.inf,
     0.5, 2.59895758929076292e-05, "aps.09.05"],
    # 同上，定义另一个参数组合的列表
    [aps09_f, aps09_fp, aps09_fpp, (20,), [0, 1], np.inf,
     0.5, 7.66859512218533719e-06, "aps.09.06"],
    # 同上，定义另一个参数组合的列表
    [aps10_f, aps10_fp, aps10_fpp, (1,), [0, 1], np.inf,
     0.9, 4.01058137541547011e-01, "aps.10.00"],
    # 同上，定义另一个参数组合的列表
    [aps10_f, aps10_fp, aps10_fpp, (5,), [0, 1], np.inf,
     0.9, 5.16153518757933583e-01, "aps.10.01"],
    # 同上，定义另一个参数组合的列表
    [aps10_f, aps10_fp, aps10_fpp, (10,), [0, 1], np.inf,
     0.9, 5.39522226908415781e-01, "aps.10.02"],
    # 同上，定义另一个参数组合的列表
    [aps10_f, aps10_fp, aps10_fpp, (15,), [0, 1], np.inf,
     0.9, 5.48182294340655241e-01, "aps.10.03"],
    # 同上，定义另一个参数组合的列表
    [aps10_f, aps10_fp, aps10_fpp, (20,), [0, 1], np.inf,
     0.9, 5.52704666678487833e-01, "aps.10.04"],
    # 同上，定义另一个参数组合的列表
    [aps11_f, aps11_fp, aps11_fpp, (2,), [0.01, 1], np.inf,
     1e-02, 1.0 / 2, "aps.11.00"],
    # 同上，定义另一个参数组合的列表
    [aps11_f, aps11_fp, aps11_fpp, (5,), [0.01, 1], np.inf,
     1e-02, 1.0 / 5, "aps.11.01"],
    # 同上，定义另一个参数组合的列表
    [aps11_f, aps11_fp, aps11_fpp, (15,), [0.01, 1], np.inf,
     1e-02, 1.0 / 15, "aps.11.02"],
    # 同上，定义另一个参数组合的列表
    [aps11_f, aps11_fp, aps11_fpp, (20,), [0.01, 1], np.inf,
     1e-02, 1.0 / 20, "aps.11.03"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (2,), [1, 100], np.inf,
     1.1, 2, "aps.12.00"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (3,), [1, 100], np.inf,
     1.1, 3, "aps.12.01"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (4,), [1, 100], np.inf,
     1.1, 4, "aps.12.02"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (5,), [1, 100], np.inf,
     1.1, 5, "aps.12.03"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (6,), [1, 100], np.inf,
     1.1, 6, "aps.12.04"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (7,), [1, 100], np.inf,
     1.1, 7, "aps.12.05"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (9,), [1, 100], np.inf,
     1.1, 9, "aps.12.06"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (11,), [1, 100], np.inf,
     1.1, 11, "aps.12.07"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (13,), [1, 100], np.inf,
     1.1, 13, "aps.12.08"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (15,), [1, 100], np.inf,
     1.1, 15, "aps.12.09"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (17,), [1, 100], np.inf,
     1.1, 17, "aps.12.10"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (19,), [1, 100], np.inf,
     1.1, 19, "aps.12.11"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (21,), [1, 100], np.inf,
     1.1, 21, "aps.12.12"],
    # 同上，定义另一个参数组合的列表
    [aps12_f, aps12_fp, aps12_fpp, (23,), [1,
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps12_f, aps12_fp, aps12_fpp, (25,), [1, 100], np.inf,
     1.1, 25, "aps.12.14"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps12_f, aps12_fp, aps12_fpp, (27,), [1, 100], np.inf,
     1.1, 27, "aps.12.15"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps12_f, aps12_fp, aps12_fpp, (29,), [1, 100], np.inf,
     1.1, 29, "aps.12.16"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps12_f, aps12_fp, aps12_fpp, (31,), [1, 100], np.inf,
     1.1, 31, "aps.12.17"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps12_f, aps12_fp, aps12_fpp, (33,), [1, 100], np.inf,
     1.1, 33, "aps.12.18"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps13_f, aps13_fp, aps13_fpp, (), [-1, 4], np.inf,
     1.5, 0, "aps.13.00"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (1,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.00"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (2,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.01"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (3,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.02"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (4,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.03"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (5,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.04"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (6,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.05"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (7,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.06"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (8,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.07"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (9,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.08"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (10,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.09"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (11,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.10"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (12,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.11"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (13,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.12"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (14,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.13"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (15,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.14"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (16,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.15"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (17,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.16"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (18,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.17"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (19,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.18"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (20,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.19"],
    # 创建包含不同数据类型的列表，每个子列表表示一个数据集
    [aps14_f, aps14_fp, aps14_fpp, (21,), [-1000, np.pi / 2], 0,
     1, 6.23806518961612433e-01, "aps.14.20"],
    # 创建包含不同数据类型的列表，
    # 创建一个列表，包含多个子列表，每个子列表表示一个数据集
    [
        # 子列表1
        [aps14_f, aps14_fp, aps14_fpp, (23,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.22"],
        # 子列表2
        [aps14_f, aps14_fp, aps14_fpp, (24,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.23"],
        # 子列表3
        [aps14_f, aps14_fp, aps14_fpp, (25,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.24"],
        # 子列表4
        [aps14_f, aps14_fp, aps14_fpp, (26,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.25"],
        # 子列表5
        [aps14_f, aps14_fp, aps14_fpp, (27,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.26"],
        # 子列表6
        [aps14_f, aps14_fp, aps14_fpp, (28,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.27"],
        # 子列表7
        [aps14_f, aps14_fp, aps14_fpp, (29,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.28"],
        # 子列表8
        [aps14_f, aps14_fp, aps14_fpp, (30,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.29"],
        # 子列表9
        [aps14_f, aps14_fp, aps14_fpp, (31,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.30"],
        # 子列表10
        [aps14_f, aps14_fp, aps14_fpp, (32,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.31"],
        # 子列表11
        [aps14_f, aps14_fp, aps14_fpp, (33,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.32"],
        # 子列表12
        [aps14_f, aps14_fp, aps14_fpp, (34,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.33"],
        # 子列表13
        [aps14_f, aps14_fp, aps14_fpp, (35,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.34"],
        # 子列表14
        [aps14_f, aps14_fp, aps14_fpp, (36,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.35"],
        # 子列表15
        [aps14_f, aps14_fp, aps14_fpp, (37,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.36"],
        # 子列表16
        [aps14_f, aps14_fp, aps14_fpp, (38,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.37"],
        # 子列表17
        [aps14_f, aps14_fp, aps14_fpp, (39,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.38"],
        # 子列表18
        [aps14_f, aps14_fp, aps14_fpp, (40,), [-1000, np.pi / 2], 0,
         1, 6.23806518961612433e-01, "aps.14.39"],
        # 子列表19
        [aps15_f, aps15_fp, aps15_fpp, (20,), [-1000, 1e-4], 0,
         -2, 5.90513055942197166e-05, "aps.15.00"],
        # 子列表20
        [aps15_f, aps15_fp, aps15_fpp, (21,), [-1000, 1e-4], 0,
         -2, 5.63671553399369967e-05, "aps.15.01"],
        # 子列表21
        [aps15_f, aps15_fp, aps15_fpp, (22,), [-1000, 1e-4], 0,
         -2, 5.39164094555919196e-05, "aps.15.02"],
        # 子列表22
        [aps15_f, aps15_fp, aps15_fpp, (23,), [-1000, 1e-4], 0,
         -2, 5.16698923949422470e-05, "aps.15.03"],
        # 子列表23
        [aps15_f, aps15_fp, aps15_fpp, (24,), [-1000, 1e-4], 0,
         -2, 4.96030966991445609e-05, "aps.15.04"],
        # 子列表24
        [aps15_f, aps15_fp, aps15_fpp, (25,), [-1000, 1e-4], 0,
         -2, 4.76952852876389951e-05, "aps.15.05"],
        # 子列表25
        [aps15_f, aps15_fp, aps15_fpp, (26,), [-1000, 1e-4], 0,
         -2, 4.59287932399486662e-05, "aps.15.06"],
        # 子列表26
        [aps15_f, aps15_fp, aps15_fpp, (27,), [-1000, 1e-4], 0,
         -2, 4.42884791956647841e-05, "aps.15.07"],
        # 子列表27
        [aps15_f, aps15_fp, aps15_fpp, (28,), [-1000, 1e-4], 0,
         -2, 4.27612902578832391e-05, "aps.15.08"],
    ]
    # 创建包含多个列表的大列表，每个小列表代表一个数据项
    [
        # 第一个数据项
        [aps15_f, aps15_fp, aps15_fpp, (29,), [-1000, 1e-4], 0,
         -2, 4.13359139159538030e-05, "aps.15.09"],
        # 第二个数据项
        [aps15_f, aps15_fp, aps15_fpp, (30,), [-1000, 1e-4], 0,
         -2, 4.00024973380198076e-05, "aps.15.10"],
        # 第三个数据项
        [aps15_f, aps15_fp, aps15_fpp, (31,), [-1000, 1e-4], 0,
         -2, 3.87524192962066869e-05, "aps.15.11"],
        # 第四个数据项
        [aps15_f, aps15_fp, aps15_fpp, (32,), [-1000, 1e-4], 0,
         -2, 3.75781035599579910e-05, "aps.15.12"],
        # 第五个数据项
        [aps15_f, aps15_fp, aps15_fpp, (33,), [-1000, 1e-4], 0,
         -2, 3.64728652199592355e-05, "aps.15.13"],
        # 第六个数据项
        [aps15_f, aps15_fp, aps15_fpp, (34,), [-1000, 1e-4], 0,
         -2, 3.54307833565318273e-05, "aps.15.14"],
        # 第七个数据项
        [aps15_f, aps15_fp, aps15_fpp, (35,), [-1000, 1e-4], 0,
         -2, 3.44465949299614980e-05, "aps.15.15"],
        # 第八个数据项
        [aps15_f, aps15_fp, aps15_fpp, (36,), [-1000, 1e-4], 0,
         -2, 3.35156058778003705e-05, "aps.15.16"],
        # 第九个数据项
        [aps15_f, aps15_fp, aps15_fpp, (37,), [-1000, 1e-4], 0,
         -2, 3.26336162494372125e-05, "aps.15.17"],
        # 第十个数据项
        [aps15_f, aps15_fp, aps15_fpp, (38,), [-1000, 1e-4], 0,
         -2, 3.17968568584260013e-05, "aps.15.18"],
        # 第十一个数据项
        [aps15_f, aps15_fp, aps15_fpp, (39,), [-1000, 1e-4], 0,
         -2, 3.10019354369653455e-05, "aps.15.19"],
        # 第十二个数据项
        [aps15_f, aps15_fp, aps15_fpp, (40,), [-1000, 1e-4], 0,
         -2, 3.02457906702100968e-05, "aps.15.20"],
        # 第十三个数据项
        [aps15_f, aps15_fp, aps15_fpp, (100,), [-1000, 1e-4], 0,
         -2, 1.22779942324615231e-05, "aps.15.21"],
        # 第十四个数据项
        [aps15_f, aps15_fp, aps15_fpp, (200,), [-1000, 1e-4], 0,
         -2, 6.16953939044086617e-06, "aps.15.22"],
        # 第十五个数据项
        [aps15_f, aps15_fp, aps15_fpp, (300,), [-1000, 1e-4], 0,
         -2, 4.11985852982928163e-06, "aps.15.23"],
        # 第十六个数据项
        [aps15_f, aps15_fp, aps15_fpp, (400,), [-1000, 1e-4], 0,
         -2, 3.09246238772721682e-06, "aps.15.24"],
        # 第十七个数据项
        [aps15_f, aps15_fp, aps15_fpp, (500,), [-1000, 1e-4], 0,
         -2, 2.47520442610501789e-06, "aps.15.25"],
        # 第十八个数据项
        [aps15_f, aps15_fp, aps15_fpp, (600,), [-1000, 1e-4], 0,
         -2, 2.06335676785127107e-06, "aps.15.26"],
        # 第十九个数据项
        [aps15_f, aps15_fp, aps15_fpp, (700,), [-1000, 1e-4], 0,
         -2, 1.76901200781542651e-06, "aps.15.27"],
        # 第二十个数据项
        [aps15_f, aps15_fp, aps15_fpp, (800,), [-1000, 1e-4], 0,
         -2, 1.54816156988591016e-06, "aps.15.28"],
        # 第二十一个数据项
        [aps15_f, aps15_fp, aps15_fpp, (900,), [-1000, 1e-4], 0,
         -2, 1.37633453660223511e-06, "aps.15.29"],
        # 第二十二个数据项
        [aps15_f, aps15_fp, aps15_fpp, (1000,), [-1000, 1e-4], 0,
         -2, 1.23883857889971403e-06, "aps.15.30"]
    ]
# 将两个列表中的测试用例转换为字典格式，每个字典包含复杂测试用例的函数及其属性
_COMPLEX_TESTS_DICTS = [
    dict(zip(_COMPLEX_TESTS_KEYS, testcase)) for testcase in _COMPLEX_TESTS
]

# 将字典列表_TESTS_DICTS中的每个字典添加'a'和'b'键，这些键的值来自'd'字典中的'bracket'键对应的列表
def _add_a_b(tests):
    r"""Add "a" and "b" keys to each test from the "bracket" value"""
    for d in tests:
        for k, v in zip(['a', 'b'], d.get('bracket', [])):
            d[k] = v

# 为原始测试和APS测试分别调用_add_a_b函数，给每个测试用例添加'a'和'b'键
_add_a_b(_ORIGINAL_TESTS_DICTS)
_add_a_b(_APS_TESTS_DICTS)
_add_a_b(_COMPLEX_TESTS_DICTS)

# 返回指定集合的测试用例数组，每个测试用例都是一个字典，包含特定子集的键
def get_tests(collection='original', smoothness=None):
    r"""Return the requested collection of test cases, as an array of dicts with subset-specific keys
    
    Allowed values of collection:
    'original': The original benchmarking functions.
         Real-valued functions of real-valued inputs on an interval with a zero.
         f1, .., f3 are continuous and infinitely differentiable
         f4 has a single discontinuity at the root
         f5 has a root at 1 replacing a 1st order pole
         f6 is randomly positive on one side of the root, randomly negative on the other
    """
    'aps': The test problems in the TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions"
         paper by Alefeld, Potra and Shi. Real-valued functions of
         real-valued inputs on an interval with a zero.
         Suitable for methods which start with an enclosing interval, and
         derivatives up to 2nd order.
    'complex': Some complex-valued functions of complex-valued inputs.
         No enclosing bracket is provided.
         Suitable for methods which use one or more starting values, and
         derivatives up to 2nd order.

    The dictionary keys will be a subset of
    ["f", "fprime", "fprime2", "args", "bracket", "a", b", "smoothness", "x0", "x1", "root", "ID"]
    """  # noqa: E501
# Backwards compatibility
methods = [cc.bisect, cc.ridder, cc.brenth, cc.brentq]
mstrings = ['cc.bisect', 'cc.ridder', 'cc.brenth', 'cc.brentq']
functions = [f2, f3, f4, f5, f6]
fstrings = ['f2', 'f3', 'f4', 'f5', 'f6']

#   ##################
#   "Chandrupatla" test cases
#   Functions and test cases that appear in [2]

def fun1(x):
    return x**3 - 2*x - 5
fun1.root = 2.0945514815423265  # additional precision using mpmath.findroot

# Define fun2 with a simple mathematical expression
def fun2(x):
    return 1 - 1/x**2
fun2.root = 1  # Set the root value for fun2

# Define fun3 with a cubic function
def fun3(x):
    return (x-3)**3
fun3.root = 3  # Set the root value for fun3

# Define fun4 with a quintic function
def fun4(x):
    return 6*(x-2)**5
fun4.root = 2  # Set the root value for fun4

# Define fun5 with a ninth degree polynomial
def fun5(x):
    return x**9
fun5.root = 0  # Set the root value for fun5

# Define fun6 with a nineteenth degree polynomial
def fun6(x):
    return x**19
fun6.root = 0  # Set the root value for fun6

# Define fun7 with a conditional function involving array_namespace
def fun7(x):
    xp = array_namespace(x)
    return 0 if xp.abs(x) < 3.8e-4 else x*xp.exp(-x**(-2))
fun7.root = 0  # Set the root value for fun7

# Define fun8 with a complex mathematical expression involving array_namespace
def fun8(x):
    xp = array_namespace(x)
    xi = 0.61489
    return -(3062*(1-xi)*xp.exp(-x))/(xi + (1-xi)*xp.exp(-x)) - 1013 + 1628/x
fun8.root = 1.0375360332870405  # Set the root value for fun8

# Define fun9 with an exponential and polynomial expression
def fun9(x):
    xp = array_namespace(x)
    return xp.exp(x) - 2 - 0.01/x**2 + .000002/x**3
fun9.root = 0.7032048403631358  # Set the root value for fun9

# Each "chandropatla" test case has
# - a function,
# - two starting values x0 and x1
# - the root
# - the number of function evaluations required by Chandrupatla's algorithm
# - an Identifier of the test case
#
# Chandrupatla's is a bracketing algorithm, so a bracketing interval was
# provided in [2] for each test case. No special support for testing with
# secant/Newton/Halley is provided.

_CHANDRUPATLA_TESTS_KEYS = ["f", "bracket", "root", "nfeval", "ID"]
_CHANDRUPATLA_TESTS = [
    [fun1, [2, 3], fun1.root, 7],
    [fun1, [1, 10], fun1.root, 11],
    [fun1, [1, 100], fun1.root, 14],
    [fun1, [-1e4, 1e4], fun1.root, 23],
    [fun1, [-1e10, 1e10], fun1.root, 43],
    [fun2, [0.5, 1.51], fun2.root, 8],
    [fun2, [1e-4, 1e4], fun2.root, 22],
    [fun2, [1e-6, 1e6], fun2.root, 28],
    [fun2, [1e-10, 1e10], fun2.root, 41],
    [fun2, [1e-12, 1e12], fun2.root, 48],
    [fun3, [0, 5], fun3.root, 21],
    [fun3, [-10, 10], fun3.root, 23],
    [fun3, [-1e4, 1e4], fun3.root, 36],
    [fun3, [-1e6, 1e6], fun3.root, 45],
    [fun3, [-1e10, 1e10], fun3.root, 55],
    [fun4, [0, 5], fun4.root, 21],
    [fun4, [-10, 10], fun4.root, 23],
    [fun4, [-1e4, 1e4], fun4.root, 33],
    [fun4, [-1e6, 1e6], fun4.root, 43],
    [fun4, [-1e10, 1e10], fun4.root, 54],
    [fun5, [-1, 4], fun5.root, 21],
    [fun5, [-2, 5], fun5.root, 22],
    [fun5, [-1, 10], fun5.root, 23],
    [fun5, [-5, 50], fun5.root, 25],
    [fun5, [-10, 100], fun5.root, 26],
    [fun6, [-1., 4.], fun6.root, 21],
    [fun6, [-2., 5.], fun6.root, 22],
    [fun6, [-1., 10.], fun6.root, 23],
    [fun6, [-5., 50.], fun6.root, 25],
    [fun6, [-10., 100.], fun6.root, 26],
    [fun7, [-1, 4], fun7.root, 8],
    [fun7, [-2, 5], fun7.root, 8],
    [fun7, [-1, 10], fun7.root, 11],
    [fun7, [-5, 50], fun7.root, 18],
    [fun7, [-10, 100], fun7.root, 19],
    [fun8, [2e-4, 2], fun8.root, 9],
    [fun8, [2e-4, 3], fun8.root, 10],
    # 调用函数fun8，并传递参数列表[2e-4, 3]以及fun8.root作为第三个参数，结果放在索引为10的位置

    [fun8, [2e-4, 9], fun8.root, 11],
    # 调用函数fun8，并传递参数列表[2e-4, 9]以及fun8.root作为第三个参数，结果放在索引为11的位置

    [fun8, [2e-4, 27], fun8.root, 12],
    # 调用函数fun8，并传递参数列表[2e-4, 27]以及fun8.root作为第三个参数，结果放在索引为12的位置

    [fun8, [2e-4, 81], fun8.root, 14],
    # 调用函数fun8，并传递参数列表[2e-4, 81]以及fun8.root作为第三个参数，结果放在索引为14的位置

    [fun9, [2e-4, 1], fun9.root, 7],
    # 调用函数fun9，并传递参数列表[2e-4, 1]以及fun9.root作为第三个参数，结果放在索引为7的位置

    [fun9, [2e-4, 3], fun9.root, 8],
    # 调用函数fun9，并传递参数列表[2e-4, 3]以及fun9.root作为第三个参数，结果放在索引为8的位置

    [fun9, [2e-4, 9], fun9.root, 10],
    # 调用函数fun9，并传递参数列表[2e-4, 9]以及fun9.root作为第三个参数，结果放在索引为10的位置

    [fun9, [2e-4, 27], fun9.root, 11],
    # 调用函数fun9，并传递参数列表[2e-4, 27]以及fun9.root作为第三个参数，结果放在索引为11的位置

    [fun9, [2e-4, 81], fun9.root, 13],
    # 调用函数fun9，并传递参数列表[2e-4, 81]以及fun9.root作为第三个参数，结果放在索引为13的位置
# 将每个测试用例的名称扩展为包含测试运行编号的格式，并构成列表_CHANDRUPATLA_TESTS
_CHANDRUPATLA_TESTS = [test + [f'{test[0].__name__}.{i%5+1}']
                       for i, test in enumerate(_CHANDRUPATLA_TESTS)]

# 将_CHANDRUPATLA_TESTS中的每个测试用例转换为字典，键由_CHANDRUPATLA_TESTS_KEYS指定
_CHANDRUPATLA_TESTS_DICTS = [dict(zip(_CHANDRUPATLA_TESTS_KEYS, testcase))
                             for testcase in _CHANDRUPATLA_TESTS]

# 将生成的测试用例字典列表_CHANDRUPATLA_TESTS_DICTS传递给函数_add_a_b进行处理
_add_a_b(_CHANDRUPATLA_TESTS_DICTS)
```