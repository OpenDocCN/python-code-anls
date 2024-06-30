# `D:\src\scipysrc\scipy\scipy\special\tests\test_mpmath.py`

```
"""
Test SciPy functions versus mpmath, if available.

"""
# 导入所需的库和模块
import numpy as np  # 导入NumPy库，用于数值计算
from numpy.testing import assert_, assert_allclose  # 导入测试函数
from numpy import pi  # 导入π常数
import pytest  # 导入pytest，用于测试框架
import itertools  # 导入itertools，用于迭代工具

from scipy._lib import _pep440  # 导入_scipy._lib._pep440模块

import scipy.special as sc  # 导入SciPy的special模块
from scipy.special._testutils import (  # 导入测试工具函数
    MissingModule, check_version, FuncData,
    assert_func_equal)
from scipy.special._mptestutils import (  # 导入测试工具函数
    Arg, FixedArg, ComplexArg, IntArg, assert_mpmath_equal,
    nonfunctional_tooslow, trace_args, time_limited, exception_to_nan,
    inf_to_nan)
from scipy.special._ufuncs import (  # 导入特殊函数
    _sinpi, _cospi, _lgam1p, _lanczos_sum_expg_scaled, _log1pmx,
    _igam_fac)

try:
    import mpmath  # 尝试导入mpmath库
except ImportError:
    mpmath = MissingModule('mpmath')  # 如果导入失败，创建一个MissingModule对象


# ------------------------------------------------------------------------------
# expi
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.10')
def test_expi_complex():
    # 准备数据集
    dataset = []
    for r in np.logspace(-99, 2, 10):
        for p in np.linspace(0, 2*np.pi, 30):
            z = r*np.exp(1j*p)
            dataset.append((z, complex(mpmath.ei(z))))
    dataset = np.array(dataset, dtype=np.cdouble)

    # 使用FuncData对象检查sc.expi函数
    FuncData(sc.expi, dataset, 0, 1).check()


# ------------------------------------------------------------------------------
# expn
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
def test_expn_large_n():
    # 测试n较大时的指数积分函数expn
    dataset = []
    for n in [50, 51]:
        for x in np.logspace(0, 4, 200):
            with mpmath.workdps(100):
                dataset.append((n, x, float(mpmath.expint(n, x))))
    dataset = np.asarray(dataset)

    # 使用FuncData对象检查sc.expn函数
    FuncData(sc.expn, dataset, (0, 1), 2, rtol=1e-13).check()

# ------------------------------------------------------------------------------
# hyp0f1
# ------------------------------------------------------------------------------


@check_version(mpmath, '0.19')
def test_hyp0f1_gh5764():
    # 对hyp0f1函数进行小规模系统测试
    dataset = []
    axis = [-99.5, -9.5, -0.5, 0.5, 9.5, 99.5]
    for v in axis:
        for x in axis:
            for y in axis:
                z = x + 1j*y
                # 使用高精度计算模式进行计算
                with mpmath.workdps(120):
                    res = complex(mpmath.hyp0f1(v, z))
                dataset.append((v, z, res))
    dataset = np.array(dataset)

    # 使用FuncData对象检查sc.hyp0f1函数
    FuncData(lambda v, z: sc.hyp0f1(v.real, z), dataset, (0, 1), 2,
             rtol=1e-13).check()


@check_version(mpmath, '0.19')
def test_hyp0f1_gh_1609():
    # 这是对gh-1609的回归测试
    vv = np.linspace(150, 180, 21)
    af = sc.hyp0f1(vv, 0.5)
    mf = np.array([mpmath.hyp0f1(v, 0.5) for v in vv])
    # 使用断言检查数组 af 和 mf.astype(float) 是否在给定的相对误差容限范围内全部接近
    assert_allclose(af, mf.astype(float), rtol=1e-12)
# ------------------------------------------------------------------------------
# hyperu
# ------------------------------------------------------------------------------

# 根据 mpmath 版本检查装饰器，确保至少是 1.1.0 版本
@check_version(mpmath, '1.1.0')
# 定义测试函数，测试 mpmath 中 hyperu 函数在接近 0 处的行为
def test_hyperu_around_0():
    dataset = []
    # 遍历 DLMF 13.2.14-15 中的测试点
    for n in np.arange(-5, 5):
        for b in np.linspace(-5, 5, 20):
            a = -n
            # 将测试点添加到数据集中，包括参数 a, b, 0 以及计算的 hyperu(a, b, 0) 的结果
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
            a = -n + b - 1
            # 添加另一组测试点到数据集中
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
    # 遍历 DLMF 13.2.16-22 中的测试点
    for a in [-10.5, -1.5, -0.5, 0, 0.5, 1, 10]:
        for b in [-1.0, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
            # 添加测试点到数据集中，包括参数 a, b, 0 以及计算的 hyperu(a, b, 0) 的结果
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
    # 将数据集转换为 NumPy 数组
    dataset = np.array(dataset)

    # 使用 FuncData 类检查 hyperu 函数在数据集上的行为，验证结果的相对误差和绝对误差
    FuncData(sc.hyperu, dataset, (0, 1, 2), 3, rtol=1e-15, atol=5e-13).check()


# ------------------------------------------------------------------------------
# hyp2f1
# ------------------------------------------------------------------------------

# 根据 mpmath 版本检查装饰器，确保至少是 1.0.0 版本
@check_version(mpmath, '1.0.0')
# 定义测试函数，测试 mpmath 中 hyp2f1 函数在特定奇点的行为
def test_hyp2f1_strange_points():
    # 定义特定的测试点列表
    pts = [
        (2, -1, -1, 0.7),  # 预期结果: 2.4
        (2, -2, -2, 0.7),  # 预期结果: 3.87
    ]
    # 将所有可能的参数组合添加到测试点列表中
    pts += list(itertools.product([2, 1, -0.7, -1000], repeat=4))
    # 筛选出符合条件的测试点，保留 b == c 且 b 是负整数的情况
    pts = [
        (a, b, c, x) for a, b, c, x in pts
        if b == c and round(b) == b and b < 0 and b != -1000
    ]
    # 定义关键字参数
    kw = dict(eliminate=True)
    # 构建数据集，包括每个测试点及其对应的 hyp2f1 函数计算结果
    dataset = [p + (float(mpmath.hyp2f1(*p, **kw)),) for p in pts]
    # 将数据集转换为 NumPy 数组，使用双精度浮点数类型
    dataset = np.array(dataset, dtype=np.float64)

    # 使用 FuncData 类检查 hyp2f1 函数在数据集上的行为，验证结果的相对误差
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()


# 根据 mpmath 版本检查装饰器，确保至少是 0.13 版本
@check_version(mpmath, '0.13')
# 定义测试函数，测试 mpmath 中 hyp2f1 函数在一些实数点上的行为
def test_hyp2f1_real_some_points():
    # 定义特定的测试点列表
    pts = [
        (1, 2, 3, 0),
        (1./3, 2./3, 5./6, 27./32),
        (1./4, 1./2, 3./4, 80./81),
        (2,-2, -3, 3),
        (2, -3, -2, 3),
        (2, -1.5, -1.5, 3),
        (1, 2, 3, 0),
        (0.7235, -1, -5, 0.3),
        (0.25, 1./3, 2, 0.999),
        (0.25, 1./3, 2, -1),
        (2, 3, 5, 0.99),
        (3./2, -0.5, 3, 0.99),
        (2, 2.5, -3.25, 0.999),
        (-8, 18.016500331508873, 10.805295997850628, 0.90875647507000001),
        (-10, 900, -10.5, 0.99),
        (-10, 900, 10.5, 0.99),
        (-1, 2, 1, 1.0),
        (-1, 2, 1, -1.0),
        (-3, 13, 5, 1.0),
        (-3, 13, 5, -1.0),
        (0.5, 1 - 270.5, 1.5, 0.999**2),  # 来自问题 1561
    ]
    # 构建数据集，包括每个测试点及其对应的 hyp2f1 函数计算结果
    dataset = [p + (float(mpmath.hyp2f1(*p)),) for p in pts]
    # 将数据集转换为 NumPy 数组，使用双精度浮点数类型
    dataset = np.array(dataset, dtype=np.float64)

    # 使用 FuncData 类检查 hyp2f1 函数在数据集上的行为，验证结果的相对误差
    with np.errstate(invalid='ignore'):
        FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()


# 根据 mpmath 版本检查装饰器，确保至少是 0.14 版本
@check_version(mpmath, '0.14')
# 定义测试函数，测试 mpmath 中 hyp2f1 函数在一些特定点上的行为
def test_hyp2f1_some_points_2():
    # 定义特定的测试点列表
    pts = [
        (112, (51,10), (-9,10), -0.99999),
        (10,-900,10.5,0.99),
        (10,-900,-10.5,0.99),
    ]
    # 定义一个函数 fev，用于处理参数 x
    def fev(x):
        # 如果 x 是元组类型，则计算其浮点数除法结果
        if isinstance(x, tuple):
            return float(x[0]) / x[1]
        else:
            return x

    # 使用 fev 函数映射列表 pts 中的每个元组 p，并为每个 p 计算 hyp2f1 函数的结果，形成新的元组
    dataset = [tuple(map(fev, p)) + (float(mpmath.hyp2f1(*p)),) for p in pts]
    
    # 将 dataset 转换为 numpy 数组，并指定数据类型为 np.float64
    dataset = np.array(dataset, dtype=np.float64)

    # 创建 FuncData 对象，使用 sc.hyp2f1 函数，传入 dataset 的列索引 (0, 1, 2, 3)，期望结果个数为 4
    # 设定相对容差 rtol=1e-10 进行检查
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()
# 根据 mpmath 库的版本要求，装饰器检查函数版本是否符合 '0.13'，如果符合则执行该函数
@check_version(mpmath, '0.13')
def test_hyp2f1_real_some():
    dataset = []
    # 遍历给定的参数范围，生成测试数据集
    for a in [-10, -5, -1.8, 1.8, 5, 10]:
        for b in [-2.5, -1, 1, 7.4]:
            for c in [-9, -1.8, 5, 20.4]:
                for z in [-10, -1.01, -0.99, 0, 0.6, 0.95, 1.5, 10]:
                    try:
                        # 计算超几何函数 2F1 的值并转换为浮点数
                        v = float(mpmath.hyp2f1(a, b, c, z))
                    except Exception:
                        continue
                    # 将计算结果和参数组合成元组，添加到数据集中
                    dataset.append((a, b, c, z, v))
    # 将数据集转换为 NumPy 数组，并指定数据类型为 float64
    dataset = np.array(dataset, dtype=np.float64)

    # 忽略无效值错误，并使用 FuncData 类检查超几何函数的测试结果
    with np.errstate(invalid='ignore'):
        FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-9,
                 ignore_inf_sign=True).check()


# 根据 mpmath 库的版本要求，装饰器检查函数版本是否符合 '0.12'，如果符合则执行该函数，且标记为慢速测试
@check_version(mpmath, '0.12')
@pytest.mark.slow
def test_hyp2f1_real_random():
    npoints = 500
    # 创建一个空的二维 NumPy 数组，用于存储随机生成的测试数据集
    dataset = np.zeros((npoints, 5), np.float64)

    # 设置随机数种子，并生成具有 Pareto 分布的随机数填充数据集的前三列
    np.random.seed(1234)
    dataset[:, 0] = np.random.pareto(1.5, npoints)
    dataset[:, 1] = np.random.pareto(1.5, npoints)
    dataset[:, 2] = np.random.pareto(1.5, npoints)
    dataset[:, 3] = 2*np.random.rand(npoints) - 1

    # 对数据集的前三列进行随机符号翻转
    dataset[:, 0] *= (-1)**np.random.randint(2, npoints)
    dataset[:, 1] *= (-1)**np.random.randint(2, npoints)
    dataset[:, 2] *= (-1)**np.random.randint(2, npoints)

    # 遍历数据集的每一行，计算超几何函数 2F1 的值并存储在第四列
    for ds in dataset:
        if mpmath.__version__ < '0.14':
            # 当 mpmath 版本小于 '0.14' 时，对于 c 远小于 a 或 b 的情况进行修正
            if abs(ds[:2]).max() > abs(ds[2]):
                ds[2] = abs(ds[:2]).max()
        # 计算超几何函数 2F1 的值并存储在第五列
        ds[4] = float(mpmath.hyp2f1(*tuple(ds[:4])))

    # 使用 FuncData 类检查超几何函数的测试结果，指定测试的函数和参数
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-9).check()


# ------------------------------------------------------------------------------
# erf (complex)
# ------------------------------------------------------------------------------

# 根据 mpmath 库的版本要求，装饰器检查函数版本是否符合 '0.14'，如果符合则执行该函数
@check_version(mpmath, '0.14')
def test_erf_complex():
    # 需要增加 mpmath 的精度以执行此测试
    old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
    try:
        mpmath.mp.dps = 70
        # 生成二维网格以创建复杂参数点集合
        x1, y1 = np.meshgrid(np.linspace(-10, 1, 31), np.linspace(-10, 1, 11))
        x2, y2 = np.meshgrid(np.logspace(-80, .8, 31), np.logspace(-80, .8, 11))
        points = np.r_[x1.ravel(),x2.ravel()] + 1j*np.r_[y1.ravel(), y2.ravel()]

        # 使用 assert_func_equal 函数检查 mpmath 库计算的 erf 和 erfc 函数在复杂点集上的正确性
        assert_func_equal(sc.erf, lambda x: complex(mpmath.erf(x)), points,
                          vectorized=False, rtol=1e-13)
        assert_func_equal(sc.erfc, lambda x: complex(mpmath.erfc(x)), points,
                          vectorized=False, rtol=1e-13)
    finally:
        # 恢复 mpmath 的默认精度设置
        mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec


# ------------------------------------------------------------------------------
# lpmv
# ------------------------------------------------------------------------------

# 根据 mpmath 库的版本要求，装饰器检查函数版本是否符合 '0.15'，如果符合则执行该函数
@check_version(mpmath, '0.15')
def test_lpmv():
    # 创建空列表以存储测试数据点
    pts = []
    # 循环遍历给定的浮点数列表，依次为每个 x 扩展多个点到 pts 列表中
    for x in [-0.99, -0.557, 1e-6, 0.132, 1]:
        # 将下列点坐标依次添加到 pts 列表中，每个点坐标包含 (x, y, z)
        pts.extend([
            (1, 1, x),
            (1, -1, x),
            (-1, 1, x),
            (-1, -2, x),
            (1, 1.7, x),
            (1, -1.7, x),
            (-1, 1.7, x),
            (-1, -2.7, x),
            (1, 10, x),
            (1, 11, x),
            (3, 8, x),
            (5, 11, x),
            (-3, 8, x),
            (-5, 11, x),
            (3, -8, x),
            (5, -11, x),
            (-3, -8, x),
            (-5, -11, x),
            (3, 8.3, x),
            (5, 11.3, x),
            (-3, 8.3, x),
            (-5, 11.3, x),
            (3, -8.3, x),
            (5, -11.3, x),
            (-3, -8.3, x),
            (-5, -11.3, x),
        ])

    # 定义一个函数 mplegenp，根据给定的参数 nu, mu, x 返回对应的结果
    def mplegenp(nu, mu, x):
        # 如果 mu 是整数且 x 等于 1，则根据条件判断返回特定值
        if mu == int(mu) and x == 1:
            # 特定情况下返回 1，这是为了修正 mpmath 0.17 的错误行为
            if mu == 0:
                return 1
            else:
                return 0
        # 否则调用 mpmath 库的 legenp 函数，返回其结果
        return mpmath.legenp(nu, mu, x)

    # 对 pts 中的每个点调用 mplegenp 函数，形成一个新的列表 dataset
    dataset = [p + (mplegenp(p[1], p[0], p[2]),) for p in pts]
    # 将 dataset 转换为 numpy 数组，数据类型为 np.float64
    dataset = np.array(dataset, dtype=np.float64)

    # 定义一个函数 evf，根据给定的参数 mu, nu, x 返回对应的结果
    def evf(mu, nu, x):
        # 调用 scipy 库的 lpmv 函数，返回其结果
        return sc.lpmv(mu.astype(int), nu, x)

    # 忽略 numpy 中的无效操作错误，并使用 FuncData 类对 evf 函数和 dataset 数据执行检查
    with np.errstate(invalid='ignore'):
        FuncData(evf, dataset, (0,1,2), 3, rtol=1e-10, atol=1e-14).check()
# ------------------------------------------------------------------------------
# beta
# ------------------------------------------------------------------------------

# 装饰器，检查所需的 mpmath 版本是否至少为 0.15
@check_version(mpmath, '0.15')
# 定义测试函数 test_beta
def test_beta():
    # 设定随机种子，确保结果可重现
    np.random.seed(1234)

    # 生成数组 b，包含多个区间的对数空间、线性空间和负数
    b = np.r_[np.logspace(-200, 200, 4),
              np.logspace(-10, 10, 4),
              np.logspace(-1, 1, 4),
              np.arange(-10, 11, 1),
              np.arange(-10, 11, 1) + 0.5,
              -1, -2.3, -3, -100.3, -10003.4]
    # 将 b 赋值给 a，保持相同的数值
    a = b

    # 创建 ab 数组，通过广播机制生成所有可能的组合，并按指定形状重新排列
    ab = np.array(np.broadcast_arrays(a[:,None], b[None,:])).reshape(2, -1).T

    # 保存 mpmath 当前的设置精度和位数
    old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
    try:
        # 将 mpmath 的小数位数设置为 400
        mpmath.mp.dps = 400

        # 断言 sc.beta 函数与 mpmath.beta 函数在 ab 数组上的结果近似相等，
        # vectorized=False 表示不进行向量化，rtol=1e-10 表示相对误差容忍度为 1e-10
        assert_func_equal(sc.beta,
                          lambda a, b: float(mpmath.beta(a, b)),
                          ab,
                          vectorized=False,
                          rtol=1e-10,
                          ignore_inf_sign=True)

        # 断言 sc.betaln 函数与 mpmath.beta 的自然对数的绝对值的对数在 ab 数组上的结果近似相等，
        # vectorized=False 表示不进行向量化，rtol=1e-10 表示相对误差容忍度为 1e-10
        assert_func_equal(
            sc.betaln,
            lambda a, b: float(mpmath.log(abs(mpmath.beta(a, b)))),
            ab,
            vectorized=False,
            rtol=1e-10)
    finally:
        # 恢复 mpmath 原来的设置精度和位数
        mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec


# ------------------------------------------------------------------------------
# loggamma
# ------------------------------------------------------------------------------

# 定义 loggamma_taylor_transition 函数，检查使用 Taylor 级数和递推关系切换时精度的变化
@check_version(mpmath, '0.19')
def test_loggamma_taylor_transition():
    # 确保从使用 Taylor 级数到递推关系时没有明显的精度跳跃

    # 定义 r 数组，包含 LOGGAMMA_TAYLOR_RADIUS 值上下浮动的数值
    r = LOGGAMMA_TAYLOR_RADIUS + np.array([-0.1, -0.01, 0, 0.01, 0.1])
    # 在 [0, 2π] 区间内均匀分布 20 个点作为 theta 数组
    theta = np.linspace(0, 2*np.pi, 20)
    # 创建 r 和 theta 的网格
    r, theta = np.meshgrid(r, theta)
    # 计算 dz，dz 是 r * exp(1j*theta) 的复数数组
    dz = r*np.exp(1j*theta)
    # 将 1 + dz 和 2 + dz 合并成 z 数组，并展平为一维
    z = np.r_[1 + dz, 2 + dz].flatten()

    # 创建 dataset，包含 z 值和 mpmath.loggamma(z) 的复数结果对
    dataset = [(z0, complex(mpmath.loggamma(z0))) for z0 in z]
    dataset = np.array(dataset)

    # 使用 FuncData 类的 check 方法检查 sc.loggamma 函数在 dataset 上的表现，
    # rtol=5e-14 表示相对误差容忍度为 5e-14
    FuncData(sc.loggamma, dataset, 0, 1, rtol=5e-14).check()


# 定义 test_loggamma_taylor 函数，测试在 z = 1, 2 周围的 loggamma 函数的表现
@check_version(mpmath, '0.19')
def test_loggamma_taylor():
    # 在接近 z = 1, 2 的零点周围进行测试

    # 生成 r 数组，在很小的范围内均匀分布 10 个点
    r = np.logspace(-16, np.log10(LOGGAMMA_TAYLOR_RADIUS), 10)
    # 在 [0, 2π] 区间内均匀分布 20 个点作为 theta 数组
    theta = np.linspace(0, 2*np.pi, 20)
    # 创建 r 和 theta 的网格
    r, theta = np.meshgrid(r, theta)
    # 计算 dz，dz 是 r * exp(1j*theta) 的复数数组
    dz = r*np.exp(1j*theta)
    # 将 1 + dz 和 2 + dz 合并成 z 数组，并展平为一维
    z = np.r_[1 + dz, 2 + dz].flatten()

    # 创建 dataset，包含 z 值和 mpmath.loggamma(z) 的复数结果对
    dataset = [(z0, complex(mpmath.loggamma(z0))) for z0 in z]
    dataset = np.array(dataset)

    # 使用 FuncData 类的 check 方法检查 sc.loggamma 函数在 dataset 上的表现，
    # rtol=5e-14 表示相对误差容忍度为 5e-14
    FuncData(sc.loggamma, dataset, 0, 1, rtol=5e-14).check()


# ------------------------------------------------------------------------------
# rgamma
# ------------------------------------------------------------------------------

# 定义 test_rgamma_zeros 函数，测试在 z = 0, -1, -2, ..., -169 零点周围的 rgamma 函数的表现
@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_rgamma_zeros():
    # 在 z = 0, -1, -2, ..., -169 的零点周围进行测试（-169 之后的值即使在零点附近也超出浮点数范围）

    # 不能使用太多点，否则测试会花费很长时间
    # 创建 dx 数组，包含在不同范围内均匀分布的数值
    dx = np.r_[-np.logspace(-1, -13, 3), 0, np.logspace(-13, -1, 3)]
    # 复制 dx 数组，创建 dy 数组，两者内容相同
    dy = dx.copy()
    # 根据 dx 和 dy 数组创建网格
    dx, dy = np.meshgrid(dx, dy)
    # 使用 dx 和 dy 创建复数数组 dz
    dz = dx + 1j*dy
    # 创建一个从 0 到 -170 的递减数组，reshape 成 1x1x170 的形状
    zeros = np.arange(0, -170, -1).reshape(1, 1, -1)
    # 将 dz 和 zeros 组合成一个新的数组 z，并展平为一维数组
    z = (zeros + np.dstack((dz,)*zeros.size)).flatten()
    # 设置 mpmath 的工作精度为 100 位小数
    with mpmath.workdps(100):
        # 使用 z 中的每个 z0 计算复数的 gamma 函数值，并存储为 (z0, gamma(z0)) 的元组
        dataset = [(z0, complex(mpmath.rgamma(z0))) for z0 in z]

    # 将 dataset 转换为 numpy 数组
    dataset = np.array(dataset)
    # 使用 FuncData 类的方法检查 gamma 函数的值
    FuncData(sc.rgamma, dataset, 0, 1, rtol=1e-12).check()
# ------------------------------------------------------------------------------
# digamma
# ------------------------------------------------------------------------------

# 根据 mpmath 的版本要求装饰器，用于检查函数是否满足版本要求
@check_version(mpmath, '0.19')
# 标记为慢速测试，通常意味着测试可能需要更长时间运行
@pytest.mark.slow
def test_digamma_roots():
    # 测试 digamma 函数的特殊根
    # 找到 digamma 函数在 1.5 处的根
    root = mpmath.findroot(mpmath.digamma, 1.5)
    roots = [float(root)]
    # 找到 digamma 函数在 -0.5 处的根
    root = mpmath.findroot(mpmath.digamma, -0.5)
    roots.append(float(root))
    roots = np.array(roots)

    # 如果超出 0.24 的半径，mpmath 将需要很长时间才能运行
    dx = np.r_[-0.24, -np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10), 0.24]
    dy = dx.copy()
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j*dy
    z = (roots + np.dstack((dz,)*roots.size)).flatten()
    # 设置工作精度为 30 位小数
    with mpmath.workdps(30):
        # 创建数据集，包含每个 z0 及其对应的 digamma(z0)
        dataset = [(z0, complex(mpmath.digamma(z0))) for z0 in z]

    dataset = np.array(dataset)
    # 使用 FuncData 类对 digamma 函数的数据集进行检查，期望精度为 1e-14
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-14).check()


@check_version(mpmath, '0.19')
def test_digamma_negreal():
    # 测试负实轴附近的 digamma 函数
    # 注意：不要在 TestSystematic 中进行此测试，因为需要调整点以避免 mpmath 运行时间过长

    # 将 digamma 函数异常值转换为 NaN
    digamma = exception_to_nan(mpmath.digamma)

    # 生成负对数空间内的 x 值
    x = -np.logspace(300, -30, 100)
    # 生成包含小范围内和大范围内的 y 值
    y = np.r_[-np.logspace(0, -3, 5), 0, np.logspace(-3, 0, 5)]
    x, y = np.meshgrid(x, y)
    z = (x + 1j*y).flatten()

    # 设置工作精度为 40 位小数
    with mpmath.workdps(40):
        # 创建数据集，包含每个 z0 及其对应的 digamma(z0)
        dataset = [(z0, complex(digamma(z0))) for z0 in z]
    dataset = np.asarray(dataset)

    # 使用 FuncData 类对 digamma 函数的数据集进行检查，期望精度为 1e-13
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-13).check()


@check_version(mpmath, '0.19')
def test_digamma_boundary():
    # 检查当从渐近级数切换到反射公式时，精度是否存在跃变

    # 生成负对数空间内的 x 值
    x = -np.logspace(300, -30, 100)
    # 定义特定的 y 值
    y = np.array([-6.1, -5.9, 5.9, 6.1])
    x, y = np.meshgrid(x, y)
    z = (x + 1j*y).flatten()

    # 设置工作精度为 30 位小数
    with mpmath.workdps(30):
        # 创建数据集，包含每个 z0 及其对应的 digamma(z0)
        dataset = [(z0, complex(mpmath.digamma(z0))) for z0 in z]
    dataset = np.asarray(dataset)

    # 使用 FuncData 类对 digamma 函数的数据集进行检查，期望精度为 1e-13
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-13).check()


# ------------------------------------------------------------------------------
# gammainc
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
# 标记为慢速测试，通常意味着测试可能需要更长时间运行
@pytest.mark.slow
def test_gammainc_boundary():
    # 测试到渐近级数的过渡

    small = 20
    # 创建包含多个 a 值的数组
    a = np.linspace(0.5*small, 2*small, 50)
    x = a.copy()
    a, x = np.meshgrid(a, x)
    a, x = a.flatten(), x.flatten()
    # 设置工作精度为 100 位小数
    with mpmath.workdps(100):
        # 创建数据集，包含每个 (a0, x0) 及其对应的 gammainc(a0, b=x0, regularized=True)
        dataset = [(a0, x0, float(mpmath.gammainc(a0, b=x0, regularized=True)))
                   for a0, x0 in zip(a, x)]
    dataset = np.array(dataset)

    # 使用 FuncData 类对 gammainc 函数的数据集进行检查，期望精度为 1e-12
    FuncData(sc.gammainc, dataset, (0, 1), 2, rtol=1e-12).check()


# ------------------------------------------------------------------------------
# spence
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
@pytest.mark.slow
# 定义一个慢速测试标记，用于标识测试的执行时间较长
def test_spence_circle():
    # 对于斯宾斯函数最棘手的区域是在圆周 |z - 1| = 1，因此需要仔细测试该区域。
    # 定义斯宾斯函数，使用 mpmath 中的 polylog 函数计算
    def spence(z):
        return complex(mpmath.polylog(2, 1 - z))

    # 在区间 [0.5, 1.5] 内生成均匀间隔的值
    r = np.linspace(0.5, 1.5)
    # 在区间 [0, 2*pi] 内生成均匀间隔的角度值
    theta = np.linspace(0, 2*pi)
    # 构造复平面上的点 z = 1 + r * exp(i*theta)，并展平为一维数组
    z = (1 + np.outer(r, np.exp(1j*theta))).flatten()
    # 生成数据集，包含每个 z0 和 spence(z0) 的复数值
    dataset = np.asarray([(z0, spence(z0)) for z0 in z])

    # 使用 FuncData 类对 spence 函数进行测试，检查结果的准确性
    FuncData(sc.spence, dataset, 0, 1, rtol=1e-14).check()


# ------------------------------------------------------------------------------
# sinpi and cospi
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
# 检查 mpmath 版本是否符合要求，如果不符合则跳过测试
def test_sinpi_zeros():
    # 获取浮点数类型的机器精度
    eps = np.finfo(float).eps
    # 在指数分布的对数和线性空间中生成一组 dx
    dx = np.r_[-np.logspace(0, -13, 3), 0, np.logspace(-13, 0, 3)]
    # 将 dx 复制给 dy
    dy = dx.copy()
    # 在 dx 和 dy 的网格上生成复数 dz
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j*dy
    # 创建一维数组，包含在给定范围内的复数值 z0
    zeros = np.arange(-100, 100, 1).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,)*zeros.size)).flatten()
    # 生成数据集，包含每个 z0 和 sinpi(z0) 的复数值
    dataset = np.asarray([(z0, complex(mpmath.sinpi(z0)))
                          for z0 in z])
    # 使用 FuncData 类对 _sinpi 函数进行测试，检查结果的准确性
    FuncData(_sinpi, dataset, 0, 1, rtol=2*eps).check()


@check_version(mpmath, '0.19')
# 检查 mpmath 版本是否符合要求，如果不符合则跳过测试
def test_cospi_zeros():
    # 获取浮点数类型的机器精度
    eps = np.finfo(float).eps
    # 在指数分布的对数和线性空间中生成一组 dx
    dx = np.r_[-np.logspace(0, -13, 3), 0, np.logspace(-13, 0, 3)]
    # 将 dx 复制给 dy
    dy = dx.copy()
    # 在 dx 和 dy 的网格上生成复数 dz
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j*dy
    # 创建一维数组，包含在给定范围内的复数值 z0
    zeros = (np.arange(-100, 100, 1) + 0.5).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,)*zeros.size)).flatten()
    # 生成数据集，包含每个 z0 和 cospi(z0) 的复数值
    dataset = np.asarray([(z0, complex(mpmath.cospi(z0)))
                          for z0 in z])

    # 使用 FuncData 类对 _cospi 函数进行测试，检查结果的准确性
    FuncData(_cospi, dataset, 0, 1, rtol=2*eps).check()


# ------------------------------------------------------------------------------
# ellipj
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
# 检查 mpmath 版本是否符合要求，如果不符合则跳过测试
def test_dn_quarter_period():
    # 定义 dn(u, m) 函数，使用 sc.ellipj 计算 Jacobian椭圆函数的 dn 分量
    def dn(u, m):
        return sc.ellipj(u, m)[2]

    # 定义 mpmath_dn(u, m) 函数，使用 mpmath 库计算 Jacobian椭圆函数的 dn 分量
    def mpmath_dn(u, m):
        return float(mpmath.ellipfun("dn", u=u, m=m))

    # 在 [0, 1] 范围内生成均匀间隔的 m 值
    m = np.linspace(0, 1, 20)
    # 在指数分布的对数空间中生成一组 du
    du = np.r_[-np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10)]
    dataset = []
    # 生成数据集，包含每个 (p, m0, mpmath_dn(p, m0)) 的值对
    for m0 in m:
        u0 = float(mpmath.ellipk(m0))
        for du0 in du:
            p = u0 + du0
            dataset.append((p, m0, mpmath_dn(p, m0)))
    dataset = np.asarray(dataset)

    # 使用 FuncData 类对 dn 函数进行测试，检查结果的准确性
    FuncData(dn, dataset, (0, 1), 2, rtol=1e-10).check()


# ------------------------------------------------------------------------------
# Wright Omega
# ------------------------------------------------------------------------------

def _mpmath_wrightomega(z, dps):
    # 使用指定的精度 dps 运行 mpmath 计算
    with mpmath.workdps(dps):
        # 将 z 转换为 mpc 类型的复数
        z = mpmath.mpc(z)
        # 计算 unwind 值
        unwind = mpmath.ceil((z.imag - mpmath.pi)/(2*mpmath.pi))
        # 计算 Wright Omega 函数的值
        res = mpmath.lambertw(mpmath.exp(z), unwind)
    return res


@pytest.mark.slow
# 定义一个慢速测试标记，用于标识测试的执行时间较长
@check_version(mpmath, '0.19')
# 检查 mpmath 版本是否符合要求，如果不符合则跳过测试
def test_wrightomega_branch():
    # 在指数分布的对数空间中生成一组负数 x
    x = -np.logspace(10, 0, 25)
    # 创建一个临界点列表，用于 Wright Omega 函数
    picut_above = [np.nextafter(np.pi, np.inf)]
    picut_below = [np.nextafter(np.pi, -np.inf)]
    # 初始化一个列表，包含一个接近 -π 的值，用作上限
    npicut_above = [np.nextafter(-np.pi, np.inf)]
    # 初始化一个列表，包含一个接近 -π 的值，用作下限
    npicut_below = [np.nextafter(-np.pi, -np.inf)]
    
    # 循环执行50次，生成一系列接近 -π 的值，作为上限和下限
    for i in range(50):
        # 将上一个上限值的下一个值添加到上限列表中
        picut_above.append(np.nextafter(picut_above[-1], np.inf))
        # 将上一个下限值的下一个值添加到下限列表中
        picut_below.append(np.nextafter(picut_below[-1], -np.inf))
        # 将上一个上限值的下一个值添加到上限列表中
        npicut_above.append(np.nextafter(npicut_above[-1], np.inf))
        # 将上一个下限值的下一个值添加到下限列表中
        npicut_below.append(np.nextafter(npicut_below[-1], -np.inf))
    
    # 将上限和下限列表连接成一个一维数组
    y = np.hstack((picut_above, picut_below, npicut_above, npicut_below))
    
    # 创建一个二维网格，x 是原始数组，y 是连接后的数组
    x, y = np.meshgrid(x, y)
    
    # 将 x 和 y 转换成一个一维数组，并转换为复数 z
    z = (x + 1j*y).flatten()
    
    # 创建一个数据集，包含 z0 和对应的特殊函数值的复数对
    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25)))
                          for z0 in z])
    
    # 使用特殊函数检查数据集，FuncData 是一个自定义类，使用特定的参数进行初始化
    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-8).check()
@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_region1():
    # This region gets less coverage in the TestSystematic test
    # 创建 x 轴的均匀分布数组，范围从 -2 到 1
    x = np.linspace(-2, 1)
    # 创建 y 轴的均匀分布数组，范围从 1 到 2π
    y = np.linspace(1, 2*np.pi)
    # 创建二维网格
    x, y = np.meshgrid(x, y)
    # 将二维网格展平为一维数组
    z = (x + 1j*y).flatten()

    # 创建数据集，包含每个 z0 及其对应的 Wright Omega 函数计算结果，放入复数数组
    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25)))
                          for z0 in z])

    # 使用 FuncData 类对 wrightomega 函数的数据进行检查，设定相对误差容忍度为 1e-15
    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-15).check()


@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_region2():
    # This region gets less coverage in the TestSystematic test
    # 创建 x 轴的均匀分布数组，范围从 -2 到 1
    x = np.linspace(-2, 1)
    # 创建 y 轴的均匀分布数组，范围从 -2π 到 -1
    y = np.linspace(-2*np.pi, -1)
    # 创建二维网格
    x, y = np.meshgrid(x, y)
    # 将二维网格展平为一维数组
    z = (x + 1j*y).flatten()

    # 创建数据集，包含每个 z0 及其对应的 Wright Omega 函数计算结果，放入复数数组
    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25)))
                          for z0 in z])

    # 使用 FuncData 类对 wrightomega 函数的数据进行检查，设定相对误差容忍度为 1e-15
    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-15).check()


# ------------------------------------------------------------------------------
# lambertw
# ------------------------------------------------------------------------------

@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_lambertw_smallz():
    # 创建 x 和 y 轴的均匀分布数组，范围从 -1 到 1，25 个点
    x, y = np.linspace(-1, 1, 25), np.linspace(-1, 1, 25)
    # 创建二维网格
    x, y = np.meshgrid(x, y)
    # 将二维网格展平为一维数组
    z = (x + 1j*y).flatten()

    # 创建数据集，包含每个 z0 及其对应的 Lambert W 函数计算结果，放入复数数组
    dataset = np.asarray([(z0, complex(mpmath.lambertw(z0)))
                          for z0 in z])

    # 使用 FuncData 类对 lambertw 函数的数据进行检查，设定相对误差容忍度为 1e-13
    FuncData(sc.lambertw, dataset, 0, 1, rtol=1e-13).check()


# ------------------------------------------------------------------------------
# Systematic tests
# ------------------------------------------------------------------------------

HYPERKW = dict(maxprec=200, maxterms=200)


@pytest.mark.slow
@check_version(mpmath, '0.17')
class TestSystematic:

    def test_airyai(self):
        # 振荡函数，限制范围
        assert_mpmath_equal(lambda z: sc.airy(z)[0],
                            mpmath.airyai,
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)
        # 振荡函数，限制范围
        assert_mpmath_equal(lambda z: sc.airy(z)[0],
                            mpmath.airyai,
                            [Arg(-1e3, 1e3)])

    def test_airyai_complex(self):
        # 检查复数参数下的 airyai 函数
        assert_mpmath_equal(lambda z: sc.airy(z)[0],
                            mpmath.airyai,
                            [ComplexArg()])

    def test_airyai_prime(self):
        # 振荡函数，限制范围
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z:
                            mpmath.airyai(z, derivative=1),
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)
        # 振荡函数，限制范围
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z:
                            mpmath.airyai(z, derivative=1),
                            [Arg(-1e3, 1e3)])

    def test_airyai_prime_complex(self):
        # 检查复数参数下的 airyai 函数的导数
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z:
                            mpmath.airyai(z, derivative=1),
                            [ComplexArg()])
    # 定义测试函数 test_airybi，用于验证 sc.airy(z)[2] 和 mpmath.airybi(z) 的相等性
    def test_airybi(self):
        # 断言 sc.airy(z)[2] 和 mpmath.airybi(z) 相等，验证在 Arg(-1e8, 1e8) 范围内的值
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z:
                            mpmath.airybi(z),
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)
        # 断言 sc.airy(z)[2] 和 mpmath.airybi(z) 相等，验证在 Arg(-1e3, 1e3) 范围内的值
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z:
                            mpmath.airybi(z),
                            [Arg(-1e3, 1e3)])

    # 定义测试函数 test_airybi_complex，用于验证复数参数下 sc.airy(z)[2] 和 mpmath.airybi(z) 的相等性
    def test_airybi_complex(self):
        # 断言 sc.airy(z)[2] 和 mpmath.airybi(z) 相等，验证在复数参数范围内
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z:
                            mpmath.airybi(z),
                            [ComplexArg()])

    # 定义测试函数 test_airybi_prime，用于验证 sc.airy(z)[3] 和 mpmath.airybi(z, derivative=1) 的相等性
    def test_airybi_prime(self):
        # 断言 sc.airy(z)[3] 和 mpmath.airybi(z, derivative=1) 相等，验证在 Arg(-1e8, 1e8) 范围内的值
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z:
                            mpmath.airybi(z, derivative=1),
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)
        # 断言 sc.airy(z)[3] 和 mpmath.airybi(z, derivative=1) 相等，验证在 Arg(-1e3, 1e3) 范围内的值
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z:
                            mpmath.airybi(z, derivative=1),
                            [Arg(-1e3, 1e3)])

    # 定义测试函数 test_airybi_prime_complex，用于验证复数参数下 sc.airy(z)[3] 和 mpmath.airybi(z, derivative=1) 的相等性
    def test_airybi_prime_complex(self):
        # 断言 sc.airy(z)[3] 和 mpmath.airybi(z, derivative=1) 相等，验证在复数参数范围内
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z:
                            mpmath.airybi(z, derivative=1),
                            [ComplexArg()])

    # 定义测试函数 test_bei，用于验证 sc.bei 和 mpmath.bei(0, z, **HYPERKW) 的相等性
    def test_bei(self):
        # 断言 sc.bei 和 mpmath.bei(0, z, **HYPERKW) 相等，验证在 Arg(-1e3, 1e3) 范围内的值
        assert_mpmath_equal(sc.bei,
                            exception_to_nan(lambda z: mpmath.bei(0, z, **HYPERKW)),
                            [Arg(-1e3, 1e3)])

    # 定义测试函数 test_ber，用于验证 sc.ber 和 mpmath.ber(0, z, **HYPERKW) 的相等性
    def test_ber(self):
        # 断言 sc.ber 和 mpmath.ber(0, z, **HYPERKW) 相等，验证在 Arg(-1e3, 1e3) 范围内的值
        assert_mpmath_equal(sc.ber,
                            exception_to_nan(lambda z: mpmath.ber(0, z, **HYPERKW)),
                            [Arg(-1e3, 1e3)])

    # 定义测试函数 test_bernoulli，用于验证 sc.bernoulli 和 mpmath.bernoulli 的相等性
    def test_bernoulli(self):
        # 断言 sc.bernoulli(n) 和 float(mpmath.bernoulli(n)) 相等，验证在 IntArg(0, 13000) 范围内的值
        assert_mpmath_equal(lambda n: sc.bernoulli(int(n))[int(n)],
                            lambda n: float(mpmath.bernoulli(int(n))),
                            [IntArg(0, 13000)],
                            rtol=1e-9, n=13000)

    # 定义测试函数 test_besseli，用于验证 sc.iv 和 mpmath.besseli 的相等性
    def test_besseli(self):
        # 断言 sc.iv 和 mpmath.besseli 相等，验证在 Arg(-1e100, 1e100) 和 Arg() 范围内的值，设置 atol=1e-270
        assert_mpmath_equal(
            sc.iv,
            exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)),
            [Arg(-1e100, 1e100), Arg()],
            atol=1e-270,
        )

    # 定义测试函数 test_besseli_complex，用于验证复数参数下 sc.iv 和 mpmath.besseli 的相等性
    def test_besseli_complex(self):
        # 断言 sc.iv(v.real, z) 和 mpmath.besseli(v, z, **HYPERKW) 相等，验证在 Arg(-1e100, 1e100) 和 ComplexArg() 范围内的值
        assert_mpmath_equal(
            lambda v, z: sc.iv(v.real, z),
            exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)),
            [Arg(-1e100, 1e100), ComplexArg()],
        )

    # 定义测试函数 test_besselj，用于验证 sc.jv 和 mpmath.besselj 的相等性
    def test_besselj(self):
        # 断言 sc.jv 和 mpmath.besselj 相等，验证在 Arg(-1e100, 1e100) 和 Arg(-1e3, 1e3) 范围内的值，设置 ignore_inf_sign=True
        assert_mpmath_equal(
            sc.jv,
            exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)),
            [Arg(-1e100, 1e100), Arg(-1e3, 1e3)],
            ignore_inf_sign=True,
        )

        # 断言 sc.jv 和 mpmath.besselj 相等，验证在 Arg(-1e100, 1e100) 和 Arg(-1e8, 1e8) 范围内的值，设置 ignore_inf_sign=True，rtol=1e-5
        # 由于振荡导致大参数下的精度损失
        assert_mpmath_equal(
            sc.jv,
            exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)),
            [Arg(-1e100, 1e100), Arg(-1e8, 1e8)],
            ignore_inf_sign=True,
            rtol=1e-5,
        )
    # 定义测试函数 test_besselj_complex，用于测试复数参数的贝塞尔函数 J_v(z)
    def test_besselj_complex(self):
        # 断言 sc.jv(v.real, z) 等于 mpmath.besselj(v, z, **HYPERKW)，检查它们的相等性
        assert_mpmath_equal(
            lambda v, z: sc.jv(v.real, z),
            exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)),
            [Arg(), ComplexArg()]  # 使用 Arg() 和 ComplexArg() 作为参数进行测试
        )

    # 定义测试函数 test_besselk，用于测试贝塞尔函数 K_v(z)
    def test_besselk(self):
        # 断言 sc.kv 等于 mpmath.besselk，并检查其相等性
        assert_mpmath_equal(
            sc.kv,
            mpmath.besselk,
            [Arg(-200, 200), Arg(0, np.inf)],  # 使用指定范围的 Arg() 作为参数进行测试
            nan_ok=False,  # 不允许结果为 NaN
            rtol=1e-12,    # 相对误差容忍度为 1e-12
        )

    # 定义测试函数 test_besselk_int，用于测试整数参数的贝塞尔函数 K_v(z)
    def test_besselk_int(self):
        # 断言 sc.kn 等于 mpmath.besselk，并检查其相等性
        assert_mpmath_equal(
            sc.kn,
            mpmath.besselk,
            [IntArg(-200, 200), Arg(0, np.inf)],  # 使用 IntArg(-200, 200) 和 Arg(0, np.inf) 作为参数进行测试
            nan_ok=False,  # 不允许结果为 NaN
            rtol=1e-12,    # 相对误差容忍度为 1e-12
        )

    # 定义测试函数 test_besselk_complex，用于测试复数参数的贝塞尔函数 K_v(z)
    def test_besselk_complex(self):
        # 断言 sc.kv(v.real, z) 等于 mpmath.besselk(v, z, **HYPERKW)，检查它们的相等性
        assert_mpmath_equal(
            lambda v, z: sc.kv(v.real, z),
            exception_to_nan(lambda v, z: mpmath.besselk(v, z, **HYPERKW)),
            [Arg(-1e100, 1e100), ComplexArg()],  # 使用 Arg(-1e100, 1e100) 和 ComplexArg() 作为参数进行测试
        )

    # 定义测试函数 test_bessely，用于测试贝塞尔函数 Y_v(z)
    def test_bessely(self):
        # 定义本地函数 mpbessely，用于计算 mpmath.bessely(v, x, **HYPERKW) 的结果
        def mpbessely(v, x):
            r = float(mpmath.bessely(v, x, **HYPERKW))  # 计算 mpmath.bessely(v, x, **HYPERKW) 的浮点数结果
            if abs(r) > 1e305:
                # 若结果绝对值超过 1e305，则将其处理为无穷大，稍早地溢出到无穷大是可以接受的
                r = np.inf * np.sign(r)
            if abs(r) == 0 and x == 0:
                # 如果结果绝对值为零且 x 等于零，则从 mpmath 得到的结果无效，此处 x=0 是一个发散点
                return np.nan
            return r

        # 断言 sc.yv 等于 exception_to_nan(mpbessely)，检查它们的相等性
        assert_mpmath_equal(
            sc.yv,
            exception_to_nan(mpbessely),
            [Arg(-1e100, 1e100), Arg(-1e8, 1e8)],  # 使用指定范围的 Arg() 作为参数进行测试
            n=5000,  # 迭代次数为 5000
        )

    # 定义测试函数 test_bessely_complex，用于测试复数参数的贝塞尔函数 Y_v(z)
    def test_bessely_complex(self):
        # 定义本地函数 mpbessely，用于计算 mpmath.bessely(v, x, **HYPERKW) 的复数结果
        def mpbessely(v, x):
            r = complex(mpmath.bessely(v, x, **HYPERKW))  # 计算 mpmath.bessely(v, x, **HYPERKW) 的复数结果
            if abs(r) > 1e305:
                # 若结果绝对值超过 1e305，则将其处理为无穷大，稍早地溢出到无穷大是可以接受的
                with np.errstate(invalid='ignore'):
                    r = np.inf * np.sign(r)
            return r

        # 断言 sc.yv(v.real, z) 等于 exception_to_nan(mpbessely)，检查它们的相等性
        assert_mpmath_equal(
            lambda v, z: sc.yv(v.real, z),
            exception_to_nan(mpbessely),
            [Arg(), ComplexArg()],  # 使用 Arg() 和 ComplexArg() 作为参数进行测试
            n=15000,  # 迭代次数为 15000
        )

    # 定义测试函数 test_bessely_int，用于测试整数参数的贝塞尔函数 Y_v(z)
    def test_bessely_int(self):
        # 定义本地函数 mpbessely，用于计算 mpmath.bessely(v, x) 的浮点数结果
        def mpbessely(v, x):
            r = float(mpmath.bessely(v, x))  # 计算 mpmath.bessely(v, x) 的浮点数结果
            if abs(r) == 0 and x == 0:
                # 如果结果绝对值为零且 x 等于零，则从 mpmath 得到的结果无效，此处 x=0 是一个发散点
                return np.nan
            return r

        # 断言 lambda v, z: sc.yn(int(v), z) 等于 exception_to_nan(mpbessely)，检查它们的相等性
        assert_mpmath_equal(
            lambda v, z: sc.yn(int(v), z),
            exception_to_nan(mpbessely),
            [IntArg(-1000, 1000), Arg(-1e8, 1e8)],  # 使用 IntArg(-1000, 1000) 和 Arg(-1e8, 1e8) 作为参数进行测试
        )
    def test_beta(self):
        # 存储不合格点的列表
        bad_points = []

        # 定义 beta 函数
        def beta(a, b, nonzero=False):
            # 如果 a 或 b 小于 -1e12，则由于精度丢失，返回 NaN
            if a < -1e12 or b < -1e12:
                # 函数在这里仅在整数点定义，但由于精度损失，这在数值上是不明确定义的。不要在此比较值。
                return np.nan
            # 如果 a 或 b 小于 0 并且 a+b 的绝对值除以 1 等于 0，则接近函数的零点
            if (a < 0 or b < 0) and (abs(float(a + b)) % 1) == 0:
                # 接近函数零点处：mpmath 和 scipy 不会在此处进行相同的舍入，因此需要使用绝对容差运行测试
                if nonzero:
                    # 将不合格点添加到列表中
                    bad_points.append((float(a), float(b)))
                    return np.nan
            # 调用 mpmath 库中的 beta 函数计算结果
            return mpmath.beta(a, b)

        # 断言 mpmath 中的 beta 函数与自定义的 beta 函数相等
        assert_mpmath_equal(
            sc.beta,
            lambda a, b: beta(a, b, nonzero=True),
            [Arg(), Arg()],
            dps=400,
            ignore_inf_sign=True,
        )

        # 断言 mpmath 中的 beta 函数与自定义的 beta 函数相等，使用不合格点数组
        assert_mpmath_equal(
            sc.beta,
            beta,
            np.array(bad_points),
            dps=400,
            atol=1e-11,
        )

    def test_betainc(self):
        # 断言 mpmath 中的 betainc 函数与时间限制的异常处理函数的结果相等
        assert_mpmath_equal(
            sc.betainc,
            time_limited()(
                exception_to_nan(
                    lambda a, b, x: mpmath.betainc(a, b, 0, x, regularized=True)
                )
            ),
            [Arg(), Arg(), Arg()],
        )

    def test_betaincc(self):
        # 断言 mpmath 中的 betaincc 函数与时间限制的异常处理函数的结果相等
        assert_mpmath_equal(
            sc.betaincc,
            time_limited()(
                exception_to_nan(
                    lambda a, b, x: mpmath.betainc(a, b, x, 1, regularized=True)
                )
            ),
            [Arg(), Arg(), Arg()],
            dps=400,
        )

    def test_binom(self):
        # 存储不合格点的列表
        bad_points = []

        # 定义二项式函数
        def binomial(n, k, nonzero=False):
            # 如果 k 的绝对值大于 1e8*(abs(n) + 1)，则由于函数在这个区域快速振荡，数值上是不明确定义的。不在此比较值。
            if abs(k) > 1e8*(abs(n) + 1):
                return np.nan
            # 如果 n 小于 k 并且 float(n-k) - np.round(float(n-k)) 的绝对值小于 1e-15，则接近函数的零点
            if n < k and abs(float(n-k) - np.round(float(n-k))) < 1e-15:
                # 接近函数零点处：mpmath 和 scipy 不会在此处进行相同的舍入，因此需要使用绝对容差运行测试
                if nonzero:
                    # 将不合格点添加到列表中
                    bad_points.append((float(n), float(k)))
                    return np.nan
            # 调用 mpmath 库中的二项式函数计算结果
            return mpmath.binomial(n, k)

        # 断言 mpmath 中的二项式函数与自定义的二项式函数相等
        assert_mpmath_equal(
            sc.binom,
            lambda n, k: binomial(n, k, nonzero=True),
            [Arg(), Arg()],
            dps=400,
        )

        # 断言 mpmath 中的二项式函数与自定义的二项式函数相等，使用不合格点数组
        assert_mpmath_equal(
            sc.binom,
            binomial,
            np.array(bad_points),
            dps=400,
            atol=1e-14,
        )
    # 定义一个测试函数，用于验证 eval_chebyt 函数的整数参数版本
    def test_chebyt_int(self):
        # 使用 assert_mpmath_equal 函数比较 eval_chebyt 的结果和 mpmath 库中 chebyt 函数的结果
        assert_mpmath_equal(
            lambda n, x: sc.eval_chebyt(int(n), x),  # 调用 eval_chebyt 函数
            exception_to_nan(lambda n, x: mpmath.chebyt(n, x, **HYPERKW)),  # 调用 mpmath 库中的 chebyt 函数
            [IntArg(), Arg()],  # 提供参数类型的列表
            dps=50,  # 设定计算精度为 50 位小数点
        )

    # 使用 pytest.mark.xfail 装饰器标记的测试函数，表示预期有一些情况会失败
    @pytest.mark.xfail(run=False, reason="some cases in hyp2f1 not fully accurate")
    def test_chebyt(self):
        # 使用 assert_mpmath_equal 函数比较 eval_chebyt 函数和 time_limited 包装的 mpmath 库中 chebyt 函数的结果
        assert_mpmath_equal(
            sc.eval_chebyt,  # 调用 eval_chebyt 函数
            lambda n, x: time_limited()(exception_to_nan(mpmath.chebyt))(n, x, **HYPERKW),  # 限时执行的 mpmath 库中的 chebyt 函数
            [Arg(-101, 101), Arg()],  # 提供参数类型的列表，限定 n 的范围在 -101 到 101 之间
            n=10000,  # 指定 n 的具体值为 10000
        )

    # 定义一个测试函数，用于验证 eval_chebyu 函数的整数参数版本
    def test_chebyu_int(self):
        # 使用 assert_mpmath_equal 函数比较 eval_chebyu 函数的结果和 mpmath 库中 chebyu 函数的结果
        assert_mpmath_equal(
            lambda n, x: sc.eval_chebyu(int(n), x),  # 调用 eval_chebyu 函数
            exception_to_nan(lambda n, x: mpmath.chebyu(n, x, **HYPERKW)),  # 调用 mpmath 库中的 chebyu 函数
            [IntArg(), Arg()],  # 提供参数类型的列表
            dps=50,  # 设定计算精度为 50 位小数点
        )

    # 使用 pytest.mark.xfail 装饰器标记的测试函数，表示预期有一些情况会失败
    @pytest.mark.xfail(run=False, reason="some cases in hyp2f1 not fully accurate")
    def test_chebyu(self):
        # 使用 assert_mpmath_equal 函数比较 eval_chebyu 函数和 time_limited 包装的 mpmath 库中 chebyu 函数的结果
        assert_mpmath_equal(
            sc.eval_chebyu,  # 调用 eval_chebyu 函数
            lambda n, x: time_limited()(exception_to_nan(mpmath.chebyu))(n, x, **HYPERKW),  # 限时执行的 mpmath 库中的 chebyu 函数
            [Arg(-101, 101), Arg()],  # 提供参数类型的列表，限定 n 的范围在 -101 到 101 之间
        )

    # 定义一个测试函数，用于验证 digamma 函数
    def test_digamma(self):
        # 使用 assert_mpmath_equal 函数比较 sc.digamma 函数和 exception_to_nan 包装的 mpmath 库中 digamma 函数的结果
        assert_mpmath_equal(
            sc.digamma,  # 调用 sc.digamma 函数
            exception_to_nan(mpmath.digamma),  # 调用 mpmath 库中的 digamma 函数
            [Arg()],  # 提供参数类型的列表
            rtol=1e-12,  # 相对误差容忍度设定为 1e-12
            dps=50,  # 设定计算精度为 50 位小数点
        )

    # 定义一个测试函数，用于验证 chi 函数
    def test_chi(self):
        # 内部定义 chi 函数，并使用 assert_mpmath_equal 函数比较 chi 函数和 mpmath 库中 chi 函数的结果
        def chi(x):
            return sc.shichi(x)[1]  # 调用 sc.shichi 函数的第二个返回值作为 chi 函数的结果
        assert_mpmath_equal(chi, mpmath.chi, [Arg()])  # 比较 chi 函数和 mpmath 库中 chi 函数的结果
        # 检查渐近级数的交叉点
        assert_mpmath_equal(chi, mpmath.chi, [FixedArg([88 - 1e-9, 88, 88 + 1e-9])])  # 比较 chi 函数和 mpmath 库中 chi 函数在固定参数范围内的结果

    # 定义一个测试函数，用于验证 chi 函数在复数参数下的行为
    def test_chi_complex(self):
        # 内部定义 chi 函数，并使用 assert_mpmath_equal 函数比较 chi 函数和 mpmath 库中 chi 函数的结果
        def chi(z):
            return sc.shichi(z)[1]  # 调用 sc.shichi 函数的第二个返回值作为 chi 函数的结果
        # chi 函数在 Im[z] -> +- inf 时振荡，所以限定参数范围
        assert_mpmath_equal(
            chi,  # 调用 chi 函数
            mpmath.chi,  # 调用 mpmath 库中 chi 函数
            [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))],  # 提供复数参数类型的列表
            rtol=1e-12,  # 相对误差容忍度设定为 1e-12
        )

    # 定义一个测试函数，用于验证 ci 函数
    def test_ci(self):
        # 内部定义 ci 函数，并使用 assert_mpmath_equal 函数比较 ci 函数和 mpmath 库中 ci 函数的结果
        def ci(x):
            return sc.sici(x)[1]  # 调用 sc.sici 函数的第二个返回值作为 ci 函数的结果
        # ci 函数是振荡函数，所以限定参数范围
        assert_mpmath_equal(ci, mpmath.ci, [Arg(-1e8, 1e8)])  # 比较 ci 函数和 mpmath 库中 ci 函数的结果

    # 定义一个测试函数，用于验证 ci 函数在复数参数下的行为
    def test_ci_complex(self):
        # 内部定义 ci 函数，并使用 assert_mpmath_equal 函数比较 ci 函数和 mpmath 库中 ci 函数的结果
        def ci(z):
            return sc.sici(z)[1]  # 调用 sc.sici 函数的第二个返回值作为 ci 函数的结果
        # ci 函数在 Re[z] -> +- inf 时振荡，所以限定参数范围
        assert_mpmath_equal(
            ci,  # 调用 ci 函数
            mpmath.ci,  # 调用 mpmath 库中 ci 函数
            [ComplexArg(complex(-1e8, -np.inf), complex(1e8, np.inf))],  # 提供复数参数类型的列表
            rtol=1e-8,  # 相对误差容忍度设定为 1e-8
        )

    # 定义一个测试函数，用于验证 cospi 函数
    def test_cospi(self):
        eps = np.finfo(float).eps  # 计算浮点数的机器精度
        assert_mpmath_equal(_cospi, mpmath.cospi, [Arg()], nan_ok=False, rtol=2*eps)  # 比较 _cospi 和 mpmath 库中 cospi 函数的结果，指定相对误差容忍度和不允许 NaN 值
    def test_digamma_complex(self):
        # 复数参数的对数伽玛函数测试，因为mpmath可能会挂起。参见test_digamma_negreal函数在负实轴上的测试。
        # 参数过滤器函数，根据复数z的实部小于0且虚部绝对值小于1.12返回False，否则返回True。
        def param_filter(z):
            return np.where((z.real < 0) & (np.abs(z.imag) < 1.12), False, True)

        # 断言：scipy库的digamma函数应与mpmath库的digamma函数在复数参数上近似相等
        assert_mpmath_equal(
            sc.digamma,
            exception_to_nan(mpmath.digamma),
            [ComplexArg()],
            rtol=1e-13,
            dps=40,
            param_filter=param_filter
        )

    def test_e1(self):
        # 断言：scipy库的exp1函数应与mpmath库的e1函数在实数参数上近似相等
        assert_mpmath_equal(
            sc.exp1,
            mpmath.e1,
            [Arg()],
            rtol=1e-14,
        )

    def test_e1_complex(self):
        # exp1函数在复平面上当Im[z]趋近正负无穷时振荡，因此限制范围
        assert_mpmath_equal(
            sc.exp1,
            mpmath.e1,
            [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))],
            rtol=1e-11,
        )

        # 检查交叉区域
        assert_mpmath_equal(
            sc.exp1,
            mpmath.e1,
            (np.linspace(-50, 50, 171)[:, None]
             + np.r_[0, np.logspace(-3, 2, 61), -np.logspace(-3, 2, 11)]*1j).ravel(),
            rtol=1e-11,
        )
        assert_mpmath_equal(
            sc.exp1,
            mpmath.e1,
            (np.linspace(-50, -35, 10000) + 0j),
            rtol=1e-11,
        )

    def test_exprel(self):
        # 断言：scipy库的exprel函数应与lambda表达式在实数参数上近似相等，lambda表达式处理特殊情况x=0
        assert_mpmath_equal(
            sc.exprel,
            lambda x: mpmath.expm1(x)/x if x != 0 else mpmath.mpf('1.0'),
            [Arg(a=-np.log(np.finfo(np.float64).max),
                 b=np.log(np.finfo(np.float64).max))],
        )
        # 断言：scipy库的exprel函数应与lambda表达式在给定实数参数数组上近似相等
        assert_mpmath_equal(
            sc.exprel,
            lambda x: mpmath.expm1(x)/x if x != 0 else mpmath.mpf('1.0'),
            np.array([1e-12, 1e-24, 0, 1e12, 1e24, np.inf]),
            rtol=1e-11,
        )
        # 断言：scipy库的exprel函数在正无穷处返回正无穷
        assert_(np.isinf(sc.exprel(np.inf)))
        # 断言：scipy库的exprel函数在负无穷处返回0
        assert_(sc.exprel(-np.inf) == 0)

    def test_expm1_complex(self):
        # expm1函数在复平面上当Im[z]趋近正负无穷时振荡，因此限制范围以避免精度损失
        assert_mpmath_equal(
            sc.expm1,
            mpmath.expm1,
            [ComplexArg(complex(-np.inf, -1e7), complex(np.inf, 1e7))],
        )

    def test_log1p_complex(self):
        # 断言：scipy库的log1p函数应与lambda表达式在复数参数上近似相等
        assert_mpmath_equal(
            sc.log1p,
            lambda x: mpmath.log(x+1),
            [ComplexArg()],
            dps=60,
        )

    def test_log1pmx(self):
        # 断言：_log1pmx函数应与lambda表达式在实数参数上近似相等
        assert_mpmath_equal(
            _log1pmx,
            lambda x: mpmath.log(x + 1) - x,
            [Arg()],
            dps=60,
            rtol=1e-14,
        )

    def test_ei(self):
        # 断言：scipy库的expi函数应与mpmath库的ei函数在实数参数上近似相等
        assert_mpmath_equal(sc.expi, mpmath.ei, [Arg()], rtol=1e-11)

    def test_ei_complex(self):
        # expi函数在复平面上当Im[z]趋近正负无穷时振荡，因此限制范围
        assert_mpmath_equal(
            sc.expi,
            mpmath.ei,
            [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))],
            rtol=1e-9,
        )
    # 测试scipy中的ellipe函数是否等同于mpmath中的ellipe函数，使用单个参数1.0进行比较
    def test_ellipe(self):
        assert_mpmath_equal(sc.ellipe, mpmath.ellipe, [Arg(b=1.0)])

    # 测试scipy中的ellipeinc函数是否等同于mpmath中的ellipe函数，使用参数范围在-1e3到1e3之间及单个参数1.0进行比较
    def test_ellipeinc(self):
        assert_mpmath_equal(sc.ellipeinc, mpmath.ellipe, [Arg(-1e3, 1e3), Arg(b=1.0)])

    # 测试scipy中的ellipeinc函数是否等同于mpmath中的ellipe函数，使用默认参数进行比较
    def test_ellipeinc_largephi(self):
        assert_mpmath_equal(sc.ellipeinc, mpmath.ellipe, [Arg(), Arg()])

    # 测试scipy中的ellipf函数是否等同于mpmath中的ellipf函数，使用参数范围在-1e3到1e3之间及默认参数进行比较
    def test_ellipf(self):
        assert_mpmath_equal(sc.ellipkinc, mpmath.ellipf, [Arg(-1e3, 1e3), Arg()])

    # 测试scipy中的ellipf函数是否等同于mpmath中的ellipf函数，使用默认参数进行比较
    def test_ellipf_largephi(self):
        assert_mpmath_equal(sc.ellipkinc, mpmath.ellipf, [Arg(), Arg()])

    # 测试scipy中的ellipk函数是否等同于mpmath中的ellipk函数，使用单个参数1.0进行比较
    def test_ellipk(self):
        assert_mpmath_equal(sc.ellipk, mpmath.ellipk, [Arg(b=1.0)])
        # 测试scipy中的ellipkm1函数是否等同于mpmath中使用1-m为参数的ellipk函数，使用参数0.0进行比较，设置精度为400
        assert_mpmath_equal(
            sc.ellipkm1,
            lambda m: mpmath.ellipk(1 - m),
            [Arg(a=0.0)],
            dps=400,
        )

    # 测试scipy中的ellipkinc函数是否等同于mpmath中的ellippi函数的0阶函数，使用参数范围在-1e3到1e3之间及单个参数1.0进行比较，忽略无穷大的符号
    def test_ellipkinc(self):
        def ellipkinc(phi, m):
            return mpmath.ellippi(0, phi, m)
        assert_mpmath_equal(
            sc.ellipkinc,
            ellipkinc,
            [Arg(-1e3, 1e3), Arg(b=1.0)],
            ignore_inf_sign=True,
        )

    # 测试scipy中的ellipkinc函数是否等同于mpmath中的ellippi函数的0阶函数，使用默认参数及单个参数1.0进行比较，忽略无穷大的符号
    def test_ellipkinc_largephi(self):
        def ellipkinc(phi, m):
            return mpmath.ellippi(0, phi, m)
        assert_mpmath_equal(
            sc.ellipkinc,
            ellipkinc,
            [Arg(), Arg(b=1.0)],
            ignore_inf_sign=True,
        )

    # 测试scipy中的ellipfun函数的sn子函数是否等同于mpmath中的ellipfun函数的sn子函数，使用参数范围在-1e6到1e6之间及参数a为0, b为1进行比较，设置相对容差为1e-8
    def test_ellipfun_sn(self):
        def sn(u, m):
            # mpmath在u=0时不能得到零--修正此问题
            if u == 0:
                return 0
            else:
                return mpmath.ellipfun("sn", u=u, m=m)

        # 振荡函数 --- 限制第一个参数的范围；在这里的精度损失是一个预期的数值特征，而非实际错误
        assert_mpmath_equal(
            lambda u, m: sc.ellipj(u, m)[0],
            sn,
            [Arg(-1e6, 1e6), Arg(a=0, b=1)],
            rtol=1e-8,
        )

    # 测试scipy中的ellipfun函数的cn子函数是否等同于mpmath中的ellipfun函数的cn子函数，使用参数范围在-1e6到1e6之间及参数a为0, b为1进行比较，设置相对容差为1e-8
    def test_ellipfun_cn(self):
        # 参见ellipfun_sn中的注释
        assert_mpmath_equal(
            lambda u, m: sc.ellipj(u, m)[1],
            lambda u, m: mpmath.ellipfun("cn", u=u, m=m),
            [Arg(-1e6, 1e6), Arg(a=0, b=1)],
            rtol=1e-8,
        )

    # 测试scipy中的ellipfun函数的dn子函数是否等同于mpmath中的ellipfun函数的dn子函数，使用参数范围在-1e6到1e6之间及参数a为0, b为1进行比较，设置相对容差为1e-8
    def test_ellipfun_dn(self):
        # 参见ellipfun_sn中的注释
        assert_mpmath_equal(
            lambda u, m: sc.ellipj(u, m)[2],
            lambda u, m: mpmath.ellipfun("dn", u=u, m=m),
            [Arg(-1e6, 1e6), Arg(a=0, b=1)],
            rtol=1e-8,
        )

    # 测试scipy中的erf函数是否等同于mpmath中的erf函数，使用单个参数进行比较
    def test_erf(self):
        assert_mpmath_equal(sc.erf, lambda z: mpmath.erf(z), [Arg()])

    # 测试scipy中的erf函数是否等同于mpmath中的erf函数，使用复数参数进行比较，生成200个测试案例
    def test_erf_complex(self):
        assert_mpmath_equal(sc.erf, lambda z: mpmath.erf(z), [ComplexArg()], n=200)

    # 测试scipy中的erfc函数是否等同于mpmath中的erfc函数，使用单个参数进行比较，设置相对容差为1e-13
    def test_erfc(self):
        assert_mpmath_equal(
            sc.erfc,
            exception_to_nan(lambda z: mpmath.erfc(z)),
            [Arg()],
            rtol=1e-13,
        )
    # 测试 sc.erfc 函数是否等效于 mpmath.erfc 函数，对复数参数进行测试
    def test_erfc_complex(self):
        assert_mpmath_equal(
            sc.erfc,
            exception_to_nan(lambda z: mpmath.erfc(z)),
            [ComplexArg()],  # 使用复数参数 ComplexArg 进行测试
            n=200,  # 进行 200 次测试
        )

    # 测试 sc.erfi 函数是否等效于 mpmath.erfi 函数，对普通参数进行测试
    def test_erfi(self):
        assert_mpmath_equal(sc.erfi, mpmath.erfi, [Arg()], n=200)  # 使用普通参数 Arg 进行 200 次测试

    # 测试 sc.erfi 函数是否等效于 mpmath.erfi 函数，对复数参数进行测试
    def test_erfi_complex(self):
        assert_mpmath_equal(sc.erfi, mpmath.erfi, [ComplexArg()], n=200)  # 使用复数参数 ComplexArg 进行 200 次测试

    # 测试 sc.ndtr 函数是否等效于 mpmath.ncdf 函数，对普通参数进行测试
    def test_ndtr(self):
        assert_mpmath_equal(
            sc.ndtr,
            exception_to_nan(lambda z: mpmath.ncdf(z)),
            [Arg()],  # 使用普通参数 Arg 进行测试
            n=200,  # 进行 200 次测试
        )

    # 测试 sc.ndtr 函数是否等效于 lambda 函数，对复数参数进行测试
    def test_ndtr_complex(self):
        assert_mpmath_equal(
            sc.ndtr,
            lambda z: mpmath.erfc(-z/np.sqrt(2.))/2.,  # 使用 lambda 函数进行测试
            [ComplexArg(a=complex(-10000, -10000), b=complex(10000, 10000))],  # 使用复数参数进行测试
            n=400,  # 进行 400 次测试
        )

    # 测试 sc.log_ndtr 函数是否等效于 mpmath.log(mpmath.ncdf(z))，对普通参数进行测试
    def test_log_ndtr(self):
        assert_mpmath_equal(
            sc.log_ndtr,
            exception_to_nan(lambda z: mpmath.log(mpmath.ncdf(z))),
            [Arg()],  # 使用普通参数 Arg 进行测试
            n=600,  # 进行 600 次测试
            dps=300,  # 设定精度为 300
            rtol=1e-13,  # 相对误差为 1e-13
        )

    # 测试 sc.log_ndtr 函数是否等效于 mpmath.log(mpmath.erfc(-z/np.sqrt(2.))/2.)，对复数参数进行测试
    def test_log_ndtr_complex(self):
        assert_mpmath_equal(
            sc.log_ndtr,
            exception_to_nan(lambda z: mpmath.log(mpmath.erfc(-z/np.sqrt(2.))/2.)),
            [ComplexArg(a=complex(-10000, -100), b=complex(10000, 100))],  # 使用复数参数进行测试
            n=200,  # 进行 200 次测试
            dps=300,  # 设定精度为 300
        )

    # 测试 sc.euler 函数返回结果是否等效于 mpmath.eulernum 函数，对整数参数进行测试
    def test_eulernum(self):
        assert_mpmath_equal(
            lambda n: sc.euler(n)[-1],  # 返回 sc.euler(n) 的最后一个结果
            mpmath.eulernum,  # 对比的标准是 mpmath.eulernum
            [IntArg(1, 10000)],  # 使用整数参数 IntArg 进行测试，范围为 1 到 10000
            n=10000,  # 进行 10000 次测试
        )

    # 测试 sc.expn 函数是否等效于 mpmath.expint 函数，对整数和非负实数参数进行测试
    def test_expint(self):
        assert_mpmath_equal(
            sc.expn,
            mpmath.expint,
            [IntArg(0, 200), Arg(0, np.inf)],  # 使用整数参数和非负实数参数进行测试
            rtol=1e-13,  # 相对误差为 1e-13
            dps=160,  # 设定精度为 160
        )

    # 测试自定义的 fresnels 函数是否等效于 mpmath.fresnels 函数，对普通参数进行测试
    def test_fresnels(self):
        def fresnels(x):
            return sc.fresnel(x)[0]
        assert_mpmath_equal(fresnels, mpmath.fresnels, [Arg()])  # 使用普通参数 Arg 进行测试

    # 测试自定义的 fresnelc 函数是否等效于 mpmath.fresnelc 函数，对普通参数进行测试
    def test_fresnelc(self):
        def fresnelc(x):
            return sc.fresnel(x)[1]
        assert_mpmath_equal(fresnelc, mpmath.fresnelc, [Arg()])  # 使用普通参数 Arg 进行测试

    # 测试 sc.gamma 函数是否等效于 mpmath.gamma 函数，对普通参数进行测试
    def test_gamma(self):
        assert_mpmath_equal(sc.gamma, exception_to_nan(mpmath.gamma), [Arg()])  # 使用普通参数 Arg 进行测试

    # 测试 sc.gamma 函数是否等效于 mpmath.gamma 函数，对复数参数进行测试
    def test_gamma_complex(self):
        assert_mpmath_equal(
            sc.gamma,
            exception_to_nan(mpmath.gamma),
            [ComplexArg()],  # 使用复数参数 ComplexArg 进行测试
            rtol=5e-13,  # 相对误差为 5e-13
        )

    # 测试 sc.gammainc 函数是否等效于 lambda 函数，对两个参数进行测试
    def test_gammainc(self):
        # 更大的参数在 test_data.py:test_local 中进行测试
        assert_mpmath_equal(
            sc.gammainc,
            lambda z, b: mpmath.gammainc(z, b=b, regularized=True),  # 使用 lambda 函数进行测试
            [Arg(0, 1e4, inclusive_a=False), Arg(0, 1e4)],  # 使用两个参数进行测试
            nan_ok=False,  # 不允许出现 NaN
            rtol=1e-11,  # 相对误差为 1e-11
        )
    # 定义一个测试函数，用于测试 sc.gammaincc 函数的正确性
    def test_gammaincc(self):
        # 较大的参数在 test_data.py:test_local 中进行测试
        assert_mpmath_equal(
            sc.gammaincc,  # 断言 sc.gammaincc 函数与 mpmath.gammainc 函数一致
            lambda z, a: mpmath.gammainc(z, a=a, regularized=True),  # 使用 mpmath.gammainc 函数进行比较
            [Arg(0, 1e4, inclusive_a=False), Arg(0, 1e4)],  # 参数范围 [0, 10000] 进行测试
            nan_ok=False,  # 不允许返回 NaN
            rtol=1e-11,  # 相对误差容忍度设置为 1e-11
        )

    # 定义一个测试函数，用于测试 sc.gammaln 函数的正确性
    def test_gammaln(self):
        # 实部为 loggamma 的结果是 log(|gamma(z)|)
        def f(z):
            return mpmath.loggamma(z).real  # 计算 mpmath.loggamma(z) 的实部

        assert_mpmath_equal(sc.gammaln, exception_to_nan(f), [Arg()])  # 断言 sc.gammaln 函数的输出与 f 函数一致

    # 使用 pytest.mark.xfail 标记的测试函数，预期运行失败时不执行
    @pytest.mark.xfail(run=False)
    def test_gegenbauer(self):
        assert_mpmath_equal(
            sc.eval_gegenbauer,  # 断言 sc.eval_gegenbauer 函数与 mpmath.gegenbauer 函数一致
            exception_to_nan(mpmath.gegenbauer),  # 使用 mpmath.gegenbauer 函数进行比较
            [Arg(-1e3, 1e3), Arg(), Arg()],  # 参数范围 [-1000, 1000] 进行测试
        )

    # 定义一个测试函数，用于测试 sc_gegenbauer 函数的正确性
    def test_gegenbauer_int(self):
        # 重新定义函数以处理数值和 mpmath 的问题
        def gegenbauer(n, a, x):
            # 在大的 `a` 值时避免溢出（mpmath 需要更大的 dps 来正确处理这些情况，因此跳过这个区域）
            if abs(a) > 1e100:
                return np.nan

            # 正确处理 n=0 和 n=1 的情况；mpmath 0.17 版本并不总是正确处理这些情况
            if n == 0:
                r = 1.0
            elif n == 1:
                r = 2*a*x
            else:
                r = mpmath.gegenbauer(n, a, x)

            # mpmath 0.17 版本在某些情况下会给出错误的结果（虚假的零），因此通过扰动结果来计算值
            if float(r) == 0 and a < -1 and float(a) == int(float(a)):
                r = mpmath.gegenbauer(n, a + mpmath.mpf('1e-50'), x)
                if abs(r) < mpmath.mpf('1e-50'):
                    r = mpmath.mpf('0.0')

            # scipy 与 mpmath 在溢出阈值上存在差异
            if abs(r) > 1e270:
                return np.inf
            return r

        # 定义一个与 sc_gegenbauer 函数对应的函数，用于进行 sc_gegenbauer 函数与 gegenbauer 函数的比较
        def sc_gegenbauer(n, a, x):
            r = sc.eval_gegenbauer(int(n), a, x)
            # scipy 与 mpmath 在溢出阈值上存在差异
            if abs(r) > 1e270:
                return np.inf
            return r

        assert_mpmath_equal(
            sc_gegenbauer,  # 断言 sc_gegenbauer 函数与 gegenbauer 函数一致
            exception_to_nan(gegenbauer),  # 使用 gegenbauer 函数进行比较
            [IntArg(0, 100), Arg(-1e9, 1e9), Arg()],  # 参数范围 [0, 100], [-1e9, 1e9], 和所有可能的 x 值进行测试
            n=40000, dps=100, ignore_inf_sign=True, rtol=1e-6,  # 额外参数设定：n=40000, dps=100, 忽略无穷大符号，相对误差容忍度设定为 1e-6
        )

        # 检查在小 x 展开时的情况
        assert_mpmath_equal(
            sc_gegenbauer,  # 断言 sc_gegenbauer 函数与 gegenbauer 函数一致
            exception_to_nan(gegenbauer),  # 使用 gegenbauer 函数进行比较
            [IntArg(0, 100), Arg(), FixedArg(np.logspace(-30, -4, 30))],  # 参数范围 [0, 100] 和固定范围的小 x 值进行测试
            dps=100, ignore_inf_sign=True,  # 额外参数设定：dps=100, 忽略无穷大符号
        )

    # 使用 pytest.mark.xfail 标记的测试函数，预期运行失败时不执行
    @pytest.mark.xfail(run=False)
    def test_gegenbauer_complex(self):
        assert_mpmath_equal(
            lambda n, a, x: sc.eval_gegenbauer(int(n), a.real, x),  # 断言 lambda 函数与 mpmath.gegenbauer 函数一致
            exception_to_nan(mpmath.gegenbauer),  # 使用 mpmath.gegenbauer 函数进行比较
            [IntArg(0, 100), Arg(), ComplexArg()],  # 参数范围 [0, 100] 和复数 x 进行测试
        )

    # 标记为 nonfunctional_tooslow 的测试，因功能缓慢而不执行
    # 定义测试函数 test_gegenbauer_complex_general，用于测试 scipy 中的 Gegenbauer 函数在复数情况下的一般性表现
    def test_gegenbauer_complex_general(self):
        # 断言 mpmath 中的 Gegenbauer 函数与 scipy 中的评估结果相等，处理异常为 NaN
        assert_mpmath_equal(
            lambda n, a, x: sc.eval_gegenbauer(n.real, a.real, x),
            exception_to_nan(mpmath.gegenbauer),
            [Arg(-1e3, 1e3), Arg(), ComplexArg()],
        )

    # 定义测试函数 test_hankel1，用于测试 scipy 中的第一类汉克尔函数的表现
    def test_hankel1(self):
        # 断言 scipy 中的 hankel1 函数与 mpmath 中的 hankel1 函数评估结果相等，处理异常为 NaN
        assert_mpmath_equal(
            sc.hankel1,
            exception_to_nan(lambda v, x: mpmath.hankel1(v, x, **HYPERKW)),
            [Arg(-1e20, 1e20), Arg()],
        )

    # 定义测试函数 test_hankel2，用于测试 scipy 中的第二类汉克尔函数的表现
    def test_hankel2(self):
        # 断言 scipy 中的 hankel2 函数与 mpmath 中的 hankel2 函数评估结果相等，处理异常为 NaN
        assert_mpmath_equal(
            sc.hankel2,
            exception_to_nan(lambda v, x: mpmath.hankel2(v, x, **HYPERKW)),
            [Arg(-1e20, 1e20), Arg()],
        )

    # 标记为预期失败的测试函数 test_hermite，用于测试 scipy 中的 Hermite 函数
    @pytest.mark.xfail(run=False, reason="issues at intermediately large orders")
    def test_hermite(self):
        # 断言 scipy 中的 eval_hermite 函数与 mpmath 中的 Hermite 函数评估结果相等，处理异常为 NaN
        assert_mpmath_equal(
            lambda n, x: sc.eval_hermite(int(n), x),
            exception_to_nan(mpmath.hermite),
            [IntArg(0, 10000), Arg()],
        )

    # 定义测试函数 test_hyp0f1，用于测试 scipy 中的超几何函数 0F1 的表现
    def test_hyp0f1(self):
        # 设定关键字参数 KW，用于提高 mpmath 中 hyp0f1 函数的计算精度
        KW = dict(maxprec=400, maxterms=1500)
        # 断言 scipy 中的 hyp0f1 函数与 mpmath 中的 hyp0f1 函数评估结果相等，处理异常为 NaN
        assert_mpmath_equal(
            sc.hyp0f1,
            lambda a, x: mpmath.hyp0f1(a, x, **KW),
            [Arg(-1e7, 1e7), Arg(0, 1e5)],
            n=5000,
        )
        # 注意事项：第二个参数 ("z") 的取值范围从下限开始受限，因为中间计算可能会溢出。
        # 可以通过为大量阶数的贝塞尔 J 函数实现渐近展开来解决此问题（类似于这里实现的贝塞尔 I 函数）。

    # 定义测试函数 test_hyp0f1_complex，用于测试 scipy 中的复数情况下的超几何函数 0F1 的表现
    def test_hyp0f1_complex(self):
        # 断言 scipy 中的 hyp0f1 函数与 mpmath 中的 hyp0f1 函数在复数情况下的评估结果相等，处理异常为 NaN
        assert_mpmath_equal(
            lambda a, z: sc.hyp0f1(a.real, z),
            exception_to_nan(lambda a, x: mpmath.hyp0f1(a, x, **HYPERKW)),
            [Arg(-10, 10), ComplexArg(complex(-120, -120), complex(120, 120))],
        )
        # 注意事项：第一个参数 ("v") 的取值范围受限于中间计算中的溢出。可以通过为大阶数的贝塞尔函数实现渐近展开来解决。

    # 定义测试函数 test_hyp1f1，用于测试 scipy 中的超几何函数 1F1 的表现
    def test_hyp1f1(self):
        # 定义辅助函数 mpmath_hyp1f1，用于调用 mpmath 中的 hyp1f1 函数处理可能出现的零除错误
        def mpmath_hyp1f1(a, b, x):
            try:
                return mpmath.hyp1f1(a, b, x)
            except ZeroDivisionError:
                return np.inf

        # 断言 scipy 中的 hyp1f1 函数与 mpmath 中的 hyp1f1 函数评估结果相等，处理异常为 NaN
        assert_mpmath_equal(
            sc.hyp1f1,
            mpmath_hyp1f1,
            [Arg(-50, 50), Arg(1, 50, inclusive_a=False), Arg(-50, 50)],
            n=500,
            nan_ok=False,
        )

    # 标记为预期失败的测试函数 test_hyp1f1_complex，用于测试 scipy 中复数情况下的超几何函数 1F1 的表现
    @pytest.mark.xfail(run=False)
    def test_hyp1f1_complex(self):
        # 断言 scipy 中的 hyp1f1 函数与 mpmath 中的 hyp1f1 函数在复数情况下的评估结果相等，处理异常为 NaN
        assert_mpmath_equal(
            inf_to_nan(lambda a, b, x: sc.hyp1f1(a.real, b.real, x)),
            exception_to_nan(lambda a, b, x: mpmath.hyp1f1(a, b, x, **HYPERKW)),
            [Arg(-1e3, 1e3), Arg(-1e3, 1e3), ComplexArg()],
            n=2000,
        )

    # 标记为非功能性和过慢的测试函数
    @nonfunctional_tooslow
    def test_hyp2f1_complex(self):
        # 测试 SciPy 的 hyp2f1 函数在性能和精度上存在问题
        # 使用 assert_mpmath_equal 函数比较 SciPy 和 mpmath 中的 hyp2f1 函数的结果
        assert_mpmath_equal(
            lambda a, b, c, x: sc.hyp2f1(a.real, b.real, c.real, x),
            exception_to_nan(lambda a, b, c, x: mpmath.hyp2f1(a, b, c, x, **HYPERKW)),
            [Arg(-1e2, 1e2), Arg(-1e2, 1e2), Arg(-1e2, 1e2), ComplexArg()],
            n=10,
        )

    @pytest.mark.xfail(run=False)
    def test_hyperu(self):
        # 标记此测试为预期失败，不执行
        assert_mpmath_equal(
            sc.hyperu,
            exception_to_nan(lambda a, b, x: mpmath.hyperu(a, b, x, **HYPERKW)),
            [Arg(), Arg(), Arg()],
        )

    @pytest.mark.xfail_on_32bit("mpmath issue gh-342: "
                                "unsupported operand mpz, long for pow")
    def test_igam_fac(self):
        # 定义一个函数 mp_igam_fac，计算逆伽玛函数的系数
        def mp_igam_fac(a, x):
            return mpmath.power(x, a)*mpmath.exp(-x)/mpmath.gamma(a)

        # 使用 assert_mpmath_equal 比较两个函数的结果
        assert_mpmath_equal(
            _igam_fac,
            mp_igam_fac,
            [Arg(0, 1e14, inclusive_a=False), Arg(0, 1e14)],
            rtol=1e-10,
        )

    def test_j0(self):
        # 对于大参数，贝塞尔函数 j0(x) 的近似为 cos(x + phi)/sqrt(x)
        # 在大参数下，cosine 函数的相位会失去精度
        # 我们仅比较到 1e8 = 1e15 * 1e-7，这是数值上的预期行为
        assert_mpmath_equal(sc.j0, mpmath.j0, [Arg(-1e3, 1e3)])
        assert_mpmath_equal(sc.j0, mpmath.j0, [Arg(-1e8, 1e8)], rtol=1e-5)

    def test_j1(self):
        # 参见 test_j0 中的注释
        assert_mpmath_equal(sc.j1, mpmath.j1, [Arg(-1e3, 1e3)])
        assert_mpmath_equal(sc.j1, mpmath.j1, [Arg(-1e8, 1e8)], rtol=1e-5)

    @pytest.mark.xfail(run=False)
    def test_jacobi(self):
        # 使用 assert_mpmath_equal 比较 SciPy 中 eval_jacobi 函数和 mpmath 中 jacobi 函数的结果
        assert_mpmath_equal(
            sc.eval_jacobi,
            exception_to_nan(lambda a, b, c, x: mpmath.jacobi(a, b, c, x, **HYPERKW)),
            [Arg(), Arg(), Arg(), Arg()],
        )
        # 同样使用 assert_mpmath_equal 比较 eval_jacobi 函数，但在 n 参数上强制转换为整数
        assert_mpmath_equal(
            lambda n, b, c, x: sc.eval_jacobi(int(n), b, c, x),
            exception_to_nan(lambda a, b, c, x: mpmath.jacobi(a, b, c, x, **HYPERKW)),
            [IntArg(), Arg(), Arg(), Arg()],
        )

    def test_jacobi_int(self):
        # 重新定义函数以处理数值和 mpmath 的问题
        def jacobi(n, a, b, x):
            # mpmath 在 n=0 时并不总是处理得很正确
            if n == 0:
                return 1.0
            return mpmath.jacobi(n, a, b, x)
        # 使用 assert_mpmath_equal 比较 eval_jacobi 函数和经过异常处理的 jacobi 函数的结果
        assert_mpmath_equal(
            lambda n, a, b, x: sc.eval_jacobi(int(n), a, b, x),
            lambda n, a, b, x: exception_to_nan(jacobi)(n, a, b, x, **HYPERKW),
            [IntArg(), Arg(), Arg(), Arg()],
            n=20000,
            dps=50,
        )
    def test_kei(self):
        # 定义函数 kei(x)，计算 mpmath 库中的 Kelvin 函数 kei(x)
        def kei(x):
            if x == 0:
                # 如果 x 等于 0，处理 mpmath 在 x=0 时的问题
                return -pi/4
            return exception_to_nan(mpmath.kei)(0, x, **HYPERKW)
        # 使用 assert_mpmath_equal 函数验证 sc.kei 和 kei 的值是否相等，参数为区间 [-1e30, 1e30]，计算精度为 1000
        assert_mpmath_equal(sc.kei, kei, [Arg(-1e30, 1e30)], n=1000)

    def test_ker(self):
        # 使用 lambda 表达式处理 mpmath 库中的 Kelvin 函数 ker(x)，并处理异常为 NaN
        assert_mpmath_equal(
            sc.ker,
            exception_to_nan(lambda x: mpmath.ker(0, x, **HYPERKW)),
            [Arg(-1e30, 1e30)],
            n=1000,
        )

    @nonfunctional_tooslow
    def test_laguerre(self):
        # 使用 trace_args 函数验证 sc.eval_laguerre 的输出和 mpmath 库中的 Laguerre 多项式函数 laguerre(n, x, **HYPERKW) 的输出是否相等
        assert_mpmath_equal(
            trace_args(sc.eval_laguerre),
            lambda n, x: exception_to_nan(mpmath.laguerre)(n, x, **HYPERKW),
            [Arg(), Arg()],
        )

    def test_laguerre_int(self):
        # 验证 sc.eval_laguerre(int(n), x) 和 mpmath 库中的 Laguerre 多项式函数 laguerre(n, x, **HYPERKW) 的输出是否相等，其中 n 为整数，精度为 20000
        assert_mpmath_equal(
            lambda n, x: sc.eval_laguerre(int(n), x),
            lambda n, x: exception_to_nan(mpmath.laguerre)(n, x, **HYPERKW),
            [IntArg(), Arg()],
            n=20000,
        )

    @pytest.mark.xfail_on_32bit("see gh-3551 for bad points")
    def test_lambertw_real(self):
        # 验证 sc.lambertw(x, int(k.real)) 和 mpmath 库中的 Lambert W 函数 lambertw(x, int(k.real)) 的输出是否相等，其中 x 为复数，k 的实部为整数，相对误差为 1e-13，不接受 NaN 值
        assert_mpmath_equal(
            lambda x, k: sc.lambertw(x, int(k.real)),
            lambda x, k: mpmath.lambertw(x, int(k.real)),
            [ComplexArg(-np.inf, np.inf), IntArg(0, 10)],
            rtol=1e-13, nan_ok=False,
        )

    def test_lanczos_sum_expg_scaled(self):
        maxgamma = 171.624376956302725
        e = np.exp(1)
        g = 6.024680040776729583740234375

        # 定义 gamma(x) 函数，计算 mpmath 库中的 Gamma 函数 gamma(x)
        def gamma(x):
            with np.errstate(over='ignore'):
                fac = ((x + g - 0.5)/e)**(x - 0.5)
                if fac != np.inf:
                    res = fac*_lanczos_sum_expg_scaled(x)
                else:
                    fac = ((x + g - 0.5)/e)**(0.5*(x - 0.5))
                    res = fac*_lanczos_sum_expg_scaled(x)
                    res *= fac
            return res

        # 验证 gamma(x) 和 mpmath 库中的 Gamma 函数 gamma(x) 的输出是否相等，其中 x 的取值范围为 (0, maxgamma]，相对误差为 1e-13
        assert_mpmath_equal(
            gamma,
            mpmath.gamma,
            [Arg(0, maxgamma, inclusive_a=False)],
            rtol=1e-13,
        )

    @nonfunctional_tooslow
    def test_legendre(self):
        # 验证 sc.eval_legendre 和 mpmath 库中的 Legendre 多项式函数 legendre(n, x, **HYPERKW) 的输出是否相等
        assert_mpmath_equal(sc.eval_legendre, mpmath.legendre, [Arg(), Arg()])

    def test_legendre_int(self):
        # 验证 sc.eval_legendre(int(n), x) 和 mpmath 库中的 Legendre 多项式函数 legendre(n, x, **HYPERKW) 的输出是否相等，其中 n 为整数，精度为 20000
        assert_mpmath_equal(
            lambda n, x: sc.eval_legendre(int(n), x),
            lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW),
            [IntArg(), Arg()],
            n=20000,
        )

        # 验证小 x 范围内 sc.eval_legendre(int(n), x) 和 mpmath 库中的 Legendre 多项式函数 legendre(n, x, **HYPERKW) 的输出是否相等，其中 n 为整数，x 在 [1e-30, 1e-4] 之间
        assert_mpmath_equal(
            lambda n, x: sc.eval_legendre(int(n), x),
            lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW),
            [IntArg(), FixedArg(np.logspace(-30, -4, 20))],
        )
    def test_legenp(self):
        # 定义一个内部函数 lpnm，计算球面调和函数 P_{n}^{m}(z) 的值
        def lpnm(n, m, z):
            try:
                # 调用 scipy 的 lpmn 函数计算球面调和函数 P_{n}^{m}(z)，并取最后一个元素
                v = sc.lpmn(m, n, z)[0][-1,-1]
            except ValueError:
                # 如果计算出错，则返回 NaN
                return np.nan
            # 如果计算结果绝对值大于 1e306，则将其调整为正无穷或负无穷
            if abs(v) > 1e306:
                # harmonize overflow to inf
                v = np.inf * np.sign(v.real)
            return v

        # 定义另一个内部函数 lpnm_2，计算球面调和函数 P_{n}^{m}(z) 的值
        def lpnm_2(n, m, z):
            # 调用 scipy 的 lpmv 函数计算球面调和函数 P_{n}^{m}(z)
            v = sc.lpmv(m, n, z)
            # 如果计算结果绝对值大于 1e306，则将其调整为正无穷或负无穷
            if abs(v) > 1e306:
                # harmonize overflow to inf
                v = np.inf * np.sign(v.real)
            return v

        # 定义函数 legenp，计算勒让德函数 P_{n}^{m}(z) 的值
        def legenp(n, m, z):
            # 处理特殊情况：当 z 等于 1 或 -1 且 n 是整数时，使用 mpmath 进行计算
            if (z == 1 or z == -1) and int(n) == n:
                # 如果 m 等于 0，则根据 n 的正负返回相应值
                if m == 0:
                    if n < 0:
                        n = -n - 1
                    return mpmath.power(mpmath.sign(z), n)
                else:
                    return 0

            # 当 z 的绝对值小于 1e-15 时，返回 NaN
            if abs(z) < 1e-15:
                # mpmath 在这里性能较差
                return np.nan

            # 根据 z 的绝对值大小选择使用 type=2 或 type=3 进行计算
            typ = 2 if abs(z) < 1 else 3
            # 调用 exception_to_nan 函数处理 mpmath 的 legenp 函数，返回计算结果
            v = exception_to_nan(mpmath.legenp)(n, m, z, type=typ)

            # 如果计算结果绝对值大于 1e306，则将其调整为正无穷或负无穷
            if abs(v) > 1e306:
                # harmonize overflow to inf
                v = mpmath.inf * mpmath.sign(v.real)

            return v

        # 使用 assert_mpmath_equal 函数验证 lpnm 和 legenp 在指定范围内的一致性
        assert_mpmath_equal(lpnm, legenp, [IntArg(-100, 100), IntArg(-100, 100), Arg()])

        # 使用 assert_mpmath_equal 函数验证 lpnm_2 和 legenp 在指定范围内的一致性，设置容差为 1e-10
        assert_mpmath_equal(
            lpnm_2,
            legenp,
            [IntArg(-100, 100), Arg(-100, 100), Arg(-1, 1)],
            atol=1e-10,
        )

    def test_legenp_complex_2(self):
        # 定义内部函数 clpnm，计算复数域球面调和函数 P_{n}^{m}(z) 的值
        def clpnm(n, m, z):
            try:
                # 调用 scipy 的 clpmn 函数计算复数域球面调和函数 P_{n}^{m}(z)，并取最后一个元素
                return sc.clpmn(m.real, n.real, z, type=2)[0][-1,-1]
            except ValueError:
                # 如果计算出错，则返回 NaN
                return np.nan

        # 定义函数 legenp，计算复数域勒让德函数 P_{n}^{m}(z) 的值
        def legenp(n, m, z):
            # 当 z 的绝对值小于 1e-15 时，返回 NaN
            if abs(z) < 1e-15:
                # mpmath 在这里性能较差
                return np.nan
            # 调用 exception_to_nan 函数处理 mpmath 的 legenp 函数，返回计算结果
            return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=2)

        # 创建复数数组 z，用于验证 clpnm 和 legenp 在复数域上的一致性
        x = np.array([-2, -0.99, -0.5, 0, 1e-5, 0.5, 0.99, 20, 2e3])
        y = np.array([-1e3, -0.5, 0.5, 1.3])
        z = (x[:,None] + 1j*y[None,:]).ravel()

        # 使用 assert_mpmath_equal 函数验证 clpnm 和 legenp 在复数域上的一致性，设置相对容差为 1e-6，计算次数为 500 次
        assert_mpmath_equal(
            clpnm,
            legenp,
            [FixedArg([-2, -1, 0, 1, 2, 10]),
             FixedArg([-2, -1, 0, 1, 2, 10]),
             FixedArg(z)],
            rtol=1e-6,
            n=500,
        )
    def test_legenp_complex_3(self):
        def clpnm(n, m, z):
            try:
                # 调用 sc.clpmn 计算勒让德多项式的值
                return sc.clpmn(m.real, n.real, z, type=3)[0][-1,-1]
            except ValueError:
                # 如果出现 ValueError，则返回 NaN
                return np.nan

        def legenp(n, m, z):
            if abs(z) < 1e-15:
                # 如果 z 的绝对值小于 1e-15，则返回 NaN
                # 这里提到 mpmath 在这里性能较差
                return np.nan
            # 调用 exception_to_nan(mpmath.legenp) 计算第三类勒让德函数的值
            return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=3)

        # 准备测试数据
        x = np.array([-2, -0.99, -0.5, 0, 1e-5, 0.5, 0.99, 20, 2e3])
        y = np.array([-1e3, -0.5, 0.5, 1.3])
        z = (x[:,None] + 1j*y[None,:]).ravel()

        # 调用 assert_mpmath_equal 进行断言
        assert_mpmath_equal(
            clpnm,
            legenp,
            [FixedArg([-2, -1, 0, 1, 2, 10]),
             FixedArg([-2, -1, 0, 1, 2, 10]),
             FixedArg(z)],
            rtol=1e-6,
            n=500,
        )

    @pytest.mark.xfail(run=False, reason="apparently picks wrong function at |z| > 1")
    def test_legenq(self):
        def lqnm(n, m, z):
            # 调用 sc.lqmn 计算第二类勒让德函数的值
            return sc.lqmn(m, n, z)[0][-1,-1]

        def legenq(n, m, z):
            if abs(z) < 1e-15:
                # 如果 z 的绝对值小于 1e-15，则返回 NaN
                # 这里提到 mpmath 在这里性能较差
                return np.nan
            # 调用 exception_to_nan(mpmath.legenq) 计算第二类勒让德函数的值
            return exception_to_nan(mpmath.legenq)(n, m, z, type=2)

        # 调用 assert_mpmath_equal 进行断言
        assert_mpmath_equal(
            lqnm,
            legenq,
            [IntArg(0, 100), IntArg(0, 100), Arg()],
        )

    @nonfunctional_tooslow
    def test_legenq_complex(self):
        def lqnm(n, m, z):
            # 调用 sc.lqmn 计算第二类勒让德函数的值
            return sc.lqmn(int(m.real), int(n.real), z)[0][-1,-1]

        def legenq(n, m, z):
            if abs(z) < 1e-15:
                # 如果 z 的绝对值小于 1e-15，则返回 NaN
                # 这里提到 mpmath 在这里性能较差
                return np.nan
            # 调用 exception_to_nan(mpmath.legenq) 计算第二类勒让德函数的值
            return exception_to_nan(mpmath.legenq)(int(n.real), int(m.real), z, type=2)

        # 调用 assert_mpmath_equal 进行断言
        assert_mpmath_equal(
            lqnm,
            legenq,
            [IntArg(0, 100), IntArg(0, 100), ComplexArg()],
            n=100,
        )

    def test_lgam1p(self):
        def param_filter(x):
            # 过滤掉极点
            return np.where((np.floor(x) == x) & (x <= 0), False, True)

        def mp_lgam1p(z):
            # loggamma 的实部是 log(|gamma(z)|)
            return mpmath.loggamma(1 + z).real

        # 调用 assert_mpmath_equal 进行断言
        assert_mpmath_equal(
            _lgam1p,
            mp_lgam1p,
            [Arg()],
            rtol=1e-13,
            dps=100,
            param_filter=param_filter,
        )

    def test_loggamma(self):
        def mpmath_loggamma(z):
            try:
                # 调用 mpmath.loggamma 计算对数伽玛函数
                res = mpmath.loggamma(z)
            except ValueError:
                # 如果出现 ValueError，则返回复数 NaN
                res = complex(np.nan, np.nan)
            return res

        # 调用 assert_mpmath_equal 进行断言
        assert_mpmath_equal(
            sc.loggamma,
            mpmath_loggamma,
            [ComplexArg()],
            nan_ok=False,
            distinguish_nan_and_inf=False,
            rtol=5e-14,
        )

    @pytest.mark.xfail(run=False)
    # 定义测试函数 test_pcfd，用于测试自定义函数 pcfd
    def test_pcfd(self):
        # 定义内部函数 pcfd，调用 sc.pbdv 返回值的第一个元素
        def pcfd(v, x):
            return sc.pbdv(v, x)[0]
        # 使用 assert_mpmath_equal 断言 pcfd 函数与 mpmath.pcfd 函数结果相等
        assert_mpmath_equal(
            pcfd,
            # 将 mpmath.pcfd 函数嵌套在异常处理函数中并使用 HYPERKW 关键字参数调用
            exception_to_nan(lambda v, x: mpmath.pcfd(v, x, **HYPERKW)),
            # 期望的参数列表，使用 Arg 类表示任意数值参数
            [Arg(), Arg()],
        )

    # 带有标记 xfail 的测试函数 test_pcfv
    @pytest.mark.xfail(run=False, reason="it's not the same as the mpmath function --- "
                                         "maybe different definition?")
    # 定义测试函数 test_pcfv
    def test_pcfv(self):
        # 定义内部函数 pcfv，调用 sc.pbvv 返回值的第一个元素
        def pcfv(v, x):
            return sc.pbvv(v, x)[0]
        # 使用 assert_mpmath_equal 断言 pcfv 函数与 mpmath.pcfv 函数结果相等
        assert_mpmath_equal(
            pcfv,
            # 使用时间限制函数包装 mpmath.pcfv，并在异常处理中将结果转换为 NaN
            lambda v, x: time_limited()(exception_to_nan(mpmath.pcfv))(v, x, **HYPERKW),
            # 期望的参数列表，使用 Arg 类表示任意数值参数
            [Arg(), Arg()],
            # 设定测试迭代次数为 1000
            n=1000,
        )

    # 定义测试函数 test_pcfw
    def test_pcfw(self):
        # 定义内部函数 pcfw，调用 sc.pbwa 返回值的第一个元素
        def pcfw(a, x):
            return sc.pbwa(a, x)[0]

        # 定义内部函数 dpcfw，调用 sc.pbwa 返回值的第二个元素
        def dpcfw(a, x):
            return sc.pbwa(a, x)[1]

        # 定义内部函数 mpmath_dpcfw，使用 mpmath.diff 计算 mpmath.pcfw 的偏导数
        def mpmath_dpcfw(a, x):
            return mpmath.diff(mpmath.pcfw, (a, x), (0, 1))

        # 断言 pcfw 函数与 mpmath.pcfw 函数在给定范围内的结果相等
        assert_mpmath_equal(
            pcfw,
            mpmath.pcfw,
            # 期望的参数列表，使用 Arg 类表示参数范围为 [-5, 5]
            [Arg(-5, 5), Arg(-5, 5)],
            # 相对误差容限为 2e-8
            rtol=2e-8,
            # 设定测试迭代次数为 100
            n=100,
        )

        # 断言 dpcfw 函数与 mpmath_dpcfw 函数在给定范围内的结果相等
        assert_mpmath_equal(
            dpcfw,
            mpmath_dpcfw,
            # 期望的参数列表，使用 Arg 类表示参数范围为 [-5, 5]
            [Arg(-5, 5), Arg(-5, 5)],
            # 相对误差容限为 2e-9
            rtol=2e-9,
            # 设定测试迭代次数为 100
            n=100,
        )

    # 带有标记 xfail 的测试函数 test_polygamma
    @pytest.mark.xfail(reason="issues at large arguments (atol OK, rtol not) "
                              "and <eps-close to z=0")
    # 定义测试函数 test_polygamma
    def test_polygamma(self):
        # 断言 sc.polygamma 函数与 mpmath.polygamma 函数在给定范围内的结果相等
        assert_mpmath_equal(
            sc.polygamma,
            # 使用时间限制函数包装 mpmath.polygamma，并在异常处理中将结果转换为 NaN
            time_limited()(exception_to_nan(mpmath.polygamma)),
            # 期望的参数列表，IntArg 类表示整数参数范围为 [0, 1000]
            [IntArg(0, 1000), Arg()],
        )

    # 定义测试函数 test_rgamma
    def test_rgamma(self):
        # 断言 sc.rgamma 函数与 mpmath.rgamma 函数在给定范围内的结果相等
        assert_mpmath_equal(
            sc.rgamma,
            mpmath.rgamma,
            # 期望的参数列表，使用 Arg 类表示参数范围为 [-8000, ∞)
            [Arg(-8000, np.inf)],
            # 设定测试迭代次数为 5000
            n=5000,
            # 不允许 NaN 结果，忽略正负无穷的符号
            nan_ok=False,
            ignore_inf_sign=True,
        )

    # 定义测试函数 test_rgamma_complex
    def test_rgamma_complex(self):
        # 断言 sc.rgamma 函数与经异常处理后的 mpmath.rgamma 函数在复数参数范围内的结果相等
        assert_mpmath_equal(
            sc.rgamma,
            exception_to_nan(mpmath.rgamma),
            # 期望的参数列表，ComplexArg 类表示复数参数范围
            [ComplexArg()],
            # 相对误差容限为 5e-13
            rtol=5e-13,
        )

    # 带有标记 xfail 的测试函数，包含原因说明
    @pytest.mark.xfail(reason=("see gh-3551 for bad points on 32 bit "
                               "systems and gh-8095 for another bad "
                               "point"))
    def test_rf(self):
        # 如果 mpmath 版本符合 PEP 440 标准的 "1.0.0" 及以上版本
        if _pep440.parse(mpmath.__version__) >= _pep440.Version("1.0.0"):
            # 不需要任何解决方案
            mppoch = mpmath.rf
        else:
            # 定义一个新的函数 mppoch，处理双精度结果恰好是非正整数的情况
            def mppoch(a, m):
                # 如果双精度浮点数的结果恰好是非正整数，但对应的扩展精度 mpf 浮点数不是
                if float(a + m) == int(a + m) and float(a + m) <= 0:
                    a = mpmath.mpf(a)
                    m = int(a + m) - a
                return mpmath.rf(a, m)

        # 断言两个函数 sc.poch 和 mppoch 相等，使用参数 [Arg(), Arg()] 进行比较，精度为 400 位小数点
        assert_mpmath_equal(sc.poch, mppoch, [Arg(), Arg()], dps=400)

    def test_sinpi(self):
        # 获取浮点数的机器精度
        eps = np.finfo(float).eps
        # 断言两个函数 _sinpi 和 mpmath.sinpi 相等，使用参数 [Arg()] 进行比较，不允许返回 NaN，相对容差为机器精度的两倍
        assert_mpmath_equal(
            _sinpi,
            mpmath.sinpi,
            [Arg()],
            nan_ok=False,
            rtol=2*eps,
        )

    def test_sinpi_complex(self):
        # 断言两个函数 _sinpi 和 mpmath.sinpi 相等，使用参数 [ComplexArg()] 进行比较，不允许返回 NaN，相对容差为 2e-14
        assert_mpmath_equal(
            _sinpi,
            mpmath.sinpi,
            [ComplexArg()],
            nan_ok=False,
            rtol=2e-14,
        )

    def test_shi(self):
        # 定义一个新的函数 shi，返回 sc.shichi(x) 的第一个元素
        def shi(x):
            return sc.shichi(x)[0]
        # 断言两个函数 shi 和 mpmath.shi 相等，使用参数 [Arg()] 进行比较
        assert_mpmath_equal(shi, mpmath.shi, [Arg()])
        # 检查渐近级数的交叉点
        assert_mpmath_equal(shi, mpmath.shi, [FixedArg([88 - 1e-9, 88, 88 + 1e-9])])

    def test_shi_complex(self):
        # 定义一个新的函数 shi，返回 sc.shichi(z) 的第一个元素
        def shi(z):
            return sc.shichi(z)[0]
        # 断言两个函数 shi 和 mpmath.shi 相等，使用参数 [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))] 进行比较，相对容差为 1e-12
        assert_mpmath_equal(
            shi,
            mpmath.shi,
            [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))],
            rtol=1e-12,
        )

    def test_si(self):
        # 定义一个新的函数 si，返回 sc.sici(x) 的第一个元素
        def si(x):
            return sc.sici(x)[0]
        # 断言两个函数 si 和 mpmath.si 相等，使用参数 [Arg()] 进行比较
        assert_mpmath_equal(si, mpmath.si, [Arg()])

    def test_si_complex(self):
        # 定义一个新的函数 si，返回 sc.sici(z) 的第一个元素
        def si(z):
            return sc.sici(z)[0]
        # 断言两个函数 si 和 mpmath.si 相等，使用参数 [ComplexArg(complex(-1e8, -np.inf), complex(1e8, np.inf))] 进行比较，相对容差为 1e-12
        assert_mpmath_equal(
            si,
            mpmath.si,
            [ComplexArg(complex(-1e8, -np.inf), complex(1e8, np.inf))],
            rtol=1e-12,
        )

    def test_spence(self):
        # 定义一个新的函数 dilog，返回 mpmath.polylog(2, 1 - x)
        def dilog(x):
            return mpmath.polylog(2, 1 - x)
        # 断言两个函数 sc.spence 和 exception_to_nan(dilog) 相等，使用参数 [Arg(0, np.inf)] 进行比较，相对容差为 1e-14
        assert_mpmath_equal(
            sc.spence,
            exception_to_nan(dilog),
            [Arg(0, np.inf)],
            rtol=1e-14,
        )

    def test_spence_complex(self):
        # 定义一个新的函数 dilog，返回 mpmath.polylog(2, 1 - z)
        def dilog(z):
            return mpmath.polylog(2, 1 - z)
        # 断言两个函数 sc.spence 和 exception_to_nan(dilog) 相等，使用参数 [ComplexArg()] 进行比较，相对容差为 1e-14
        assert_mpmath_equal(
            sc.spence,
            exception_to_nan(dilog),
            [ComplexArg()],
            rtol=1e-14,
        )
    # 定义测试函数 test_spherharm，用于测试球谐函数的计算
    def test_spherharm(self):
        # 定义球谐函数 spherharm，计算给定 l, m, theta, phi 参数下的球谐函数值
        def spherharm(l, m, theta, phi):
            # 如果 m 大于 l，则返回 NaN
            if m > l:
                return np.nan
            # 调用 scipy 的 sph_harm 函数计算球谐函数值
            return sc.sph_harm(m, l, phi, theta)
        # 使用自定义函数 assert_mpmath_equal 进行测试
        assert_mpmath_equal(
            spherharm,  # 测试的函数对象
            mpmath.spherharm,  # 对应的 mpmath 函数对象
            [IntArg(0, 100), IntArg(0, 100), Arg(a=0, b=pi), Arg(a=0, b=2*pi)],  # 参数范围
            atol=1e-8,  # 公差
            n=6000,  # 迭代次数
            dps=150,  # 十进制位数
        )

    # 定义测试函数 test_struveh，用于测试斯特劳维函数的计算
    def test_struveh(self):
        # 使用 assert_mpmath_equal 进行测试
        assert_mpmath_equal(
            sc.struve,  # 测试的函数对象
            exception_to_nan(mpmath.struveh),  # 对应的 mpmath 函数对象
            [Arg(-1e4, 1e4), Arg(0, 1e4)],  # 参数范围
            rtol=5e-10,  # 相对公差
        )

    # 定义测试函数 test_struvel，用于测试负斯特劳维函数的计算
    def test_struvel(self):
        # 定义 mp_struvel 函数，处理负斯特劳维函数的特殊情况
        def mp_struvel(v, z):
            # 当 v 小于 0 并且 z 小于 -v 且 abs(v) 大于 1000 时
            if v < 0 and z < -v and abs(v) > 1000:
                # 需要更高的精度以获得正确的结果
                old_dps = mpmath.mp.dps
                try:
                    mpmath.mp.dps = 300
                    return mpmath.struvel(v, z)
                finally:
                    mpmath.mp.dps = old_dps
            # 否则调用普通的 mpmath.struvel 函数计算负斯特劳维函数值
            return mpmath.struvel(v, z)

        # 使用 assert_mpmath_equal 进行测试
        assert_mpmath_equal(
            sc.modstruve,  # 测试的函数对象
            exception_to_nan(mp_struvel),  # 对应的负斯特劳维函数对象
            [Arg(-1e4, 1e4), Arg(0, 1e4)],  # 参数范围
            rtol=5e-10,  # 相对公差
            ignore_inf_sign=True,  # 忽略无穷符号
        )

    # 定义测试函数 test_wrightomega_real，用于测试赖特欧米伽函数的实数版本的计算
    def test_wrightomega_real(self):
        # 定义 mpmath_wrightomega_real 函数，计算赖特欧米伽函数的实数版本
        def mpmath_wrightomega_real(x):
            return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))

        # 对于 x < -1000，赖特欧米伽函数精确为 0
        # 对于 x > 1e21，赖特欧米伽函数精确为 x
        # 使用 assert_mpmath_equal 进行测试
        assert_mpmath_equal(
            sc.wrightomega,  # 测试的函数对象
            mpmath_wrightomega_real,  # 对应的赖特欧米伽函数对象
            [Arg(-1000, 1e21)],  # 参数范围
            rtol=5e-15,  # 相对公差
            atol=0,  # 绝对公差
            nan_ok=False,  # 不接受 NaN
        )

    # 定义测试函数 test_wrightomega，用于测试赖特欧米伽函数的计算
    def test_wrightomega(self):
        # 使用 assert_mpmath_equal 进行测试
        assert_mpmath_equal(
            sc.wrightomega,  # 测试的函数对象
            lambda z: _mpmath_wrightomega(z, 25),  # 匿名函数，计算赖特欧米伽函数
            [ComplexArg()],  # 复数参数范围
            rtol=1e-14,  # 相对公差
            nan_ok=False,  # 不接受 NaN
        )

    # 定义测试函数 test_hurwitz_zeta，用于测试休尔维兹莱塞塔函数的计算
    def test_hurwitz_zeta(self):
        # 使用 assert_mpmath_equal 进行测试
        assert_mpmath_equal(
            sc.zeta,  # 测试的函数对象
            exception_to_nan(mpmath.zeta),  # 对应的 mpmath 函数对象
            [Arg(a=1, b=1e10, inclusive_a=False), Arg(a=0, inclusive_a=False)],  # 参数范围
        )

    # 定义测试函数 test_riemann_zeta，用于测试黎曼莱塞塔函数的计算
    def test_riemann_zeta(self):
        # 使用 assert_mpmath_equal 进行测试
        assert_mpmath_equal(
            sc.zeta,  # 测试的函数对象
            lambda x: mpmath.zeta(x) if x != 1 else mpmath.inf,  # 黎曼莱塞塔函数的计算方式
            [Arg(-100, 100)],  # 参数范围
            nan_ok=False,  # 不接受 NaN
            rtol=5e-13,  # 相对公差
        )

    # 定义测试函数 test_zetac，用于测试 zeta 函数的补函数 zetac 的计算
    def test_zetac(self):
        # 使用 assert_mpmath_equal 进行测试
        assert_mpmath_equal(
            sc.zetac,  # 测试的函数对象
            lambda x: mpmath.zeta(x) - 1 if x != 1 else mpmath.inf,  # zeta 函数的补函数计算方式
            [Arg(-100, 100)],  # 参数范围
            nan_ok=False,  # 不接受 NaN
            dps=45,  # 十进制位数
            rtol=5e-13,  # 相对公差
        )
    def test_boxcox(self):
        # 定义一个支持 Box-Cox 转换的函数，使用 mpmath 库处理高精度数学运算
        def mp_boxcox(x, lmbda):
            # 将输入参数转换为高精度浮点数
            x = mpmath.mp.mpf(x)
            lmbda = mpmath.mp.mpf(lmbda)
            # 如果 lambda 参数为 0，则返回 x 的自然对数
            if lmbda == 0:
                return mpmath.mp.log(x)
            else:
                # 否则，返回 Box-Cox 转换后的结果
                return mpmath.mp.powm1(x, lmbda) / lmbda

        # 使用 assert_mpmath_equal 函数断言 mpmath 版本的 boxcox 函数和 mp_boxcox 的结果相等
        assert_mpmath_equal(
            sc.boxcox,  # sc.boxcox 是被测试的 scipy 版本的 Box-Cox 转换函数
            exception_to_nan(mp_boxcox),  # 使用 exception_to_nan 处理异常情况，返回结果
            [Arg(a=0, inclusive_a=False), Arg()],  # 参数列表，包括一个范围限制和一个无限制的参数
            n=200,  # 迭代次数
            dps=60,  # 精度设置为 60 位小数
            rtol=1e-13,  # 相对误差容差
        )

    def test_boxcox1p(self):
        # 定义一个支持 Box-Cox1p 转换的函数，使用 mpmath 库处理高精度数学运算
        def mp_boxcox1p(x, lmbda):
            # 将输入参数转换为高精度浮点数
            x = mpmath.mp.mpf(x)
            lmbda = mpmath.mp.mpf(lmbda)
            one = mpmath.mp.mpf(1)
            # 如果 lambda 参数为 0，则返回 1 + x 的自然对数
            if lmbda == 0:
                return mpmath.mp.log(one + x)
            else:
                # 否则，返回 Box-Cox1p 转换后的结果
                return mpmath.mp.powm1(one + x, lmbda) / lmbda

        # 使用 assert_mpmath_equal 函数断言 mpmath 版本的 boxcox1p 函数和 mp_boxcox1p 的结果相等
        assert_mpmath_equal(
            sc.boxcox1p,  # sc.boxcox1p 是被测试的 scipy 版本的 Box-Cox1p 转换函数
            exception_to_nan(mp_boxcox1p),  # 使用 exception_to_nan 处理异常情况，返回结果
            [Arg(a=-1, inclusive_a=False), Arg()],  # 参数列表，包括一个范围限制和一个无限制的参数
            n=200,  # 迭代次数
            dps=60,  # 精度设置为 60 位小数
            rtol=1e-13,  # 相对误差容差
        )

    def test_spherical_jn(self):
        # 定义一个支持球 Bessel 函数 J_n(z) 的函数，使用 mpmath 库处理高精度数学运算
        def mp_spherical_jn(n, z):
            # 将输入参数转换为 mpmath 对象
            arg = mpmath.mpmathify(z)
            # 计算球 Bessel 函数 J_n(z) 的值
            out = (mpmath.besselj(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            # 如果参数 z 的虚部为 0，则返回实部
            if arg.imag == 0:
                return out.real
            else:
                return out

        # 使用 assert_mpmath_equal 函数断言 mpmath 版本的 spherical_jn 函数和 mp_spherical_jn 的结果相等
        assert_mpmath_equal(
            lambda n, z: sc.spherical_jn(int(n), z),  # sc.spherical_jn 是被测试的 scipy 版本的球 Bessel 函数 J_n(z)
            exception_to_nan(mp_spherical_jn),  # 使用 exception_to_nan 处理异常情况，返回结果
            [IntArg(0, 200), Arg(-1e8, 1e8)],  # 参数列表，包括一个整数范围和一个实数范围的参数
            dps=300,  # 精度设置为 300 位小数
        )

    def test_spherical_jn_complex(self):
        # 定义一个支持复数参数的球 Bessel 函数 J_n(z) 的函数，使用 mpmath 库处理高精度数学运算
        def mp_spherical_jn(n, z):
            # 将输入参数转换为 mpmath 对象
            arg = mpmath.mpmathify(z)
            # 计算球 Bessel 函数 J_n(z) 的值
            out = (mpmath.besselj(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            # 如果参数 z 的虚部为 0，则返回实部
            if arg.imag == 0:
                return out.real
            else:
                return out

        # 使用 assert_mpmath_equal 函数断言 mpmath 版本的 spherical_jn 函数和 mp_spherical_jn 的结果相等
        assert_mpmath_equal(
            lambda n, z: sc.spherical_jn(int(n.real), z),  # sc.spherical_jn 是被测试的 scipy 版本的球 Bessel 函数 J_n(z)
            exception_to_nan(mp_spherical_jn),  # 使用 exception_to_nan 处理异常情况，返回结果
            [IntArg(0, 200), ComplexArg()],  # 参数列表，包括一个整数范围和一个复数参数的范围
        )

    def test_spherical_yn(self):
        # 定义一个支持球 Bessel 函数 Y_n(z) 的函数，使用 mpmath 库处理高精度数学运算
        def mp_spherical_yn(n, z):
            # 将输入参数转换为 mpmath 对象
            arg = mpmath.mpmathify(z)
            # 计算球 Bessel 函数 Y_n(z) 的值
            out = (mpmath.bessely(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            # 如果参数 z 的虚部为 0，则返回实部
            if arg.imag == 0:
                return out.real
            else:
                return out

        # 使用 assert_mpmath_equal 函数断言 mpmath 版本的 spherical_yn 函数和 mp_spherical_yn 的结果相等
        assert_mpmath_equal(
            lambda n, z: sc.spherical_yn(int(n), z),  # sc.spherical_yn 是被测试的 scipy 版本的球 Bessel 函数 Y_n(z)
            exception_to_nan(mp_spherical_yn),  # 使用 exception_to_nan 处理异常情况，返回结果
            [IntArg(0, 200), Arg(-1e10, 1e10)],  # 参数列表，包括一个整数范围和一个实数范围的参数
            dps=100,  # 精度设置为 100 位小数
        )
    def test_spherical_yn_complex(self):
        # 定义一个函数 mp_spherical_yn，计算复数情况下的球面贝塞尔函数 Y_n(z)
        def mp_spherical_yn(n, z):
            # 将 z 转换为 mpmath 数字
            arg = mpmath.mpmathify(z)
            # 计算球面贝塞尔函数 Y_n(z) 的复数形式
            out = (mpmath.bessely(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            # 如果 z 是实数，则返回实部；否则返回整个结果
            if arg.imag == 0:
                return out.real
            else:
                return out

        # 使用 assert_mpmath_equal 函数断言球面贝塞尔函数 Y_n(z) 的计算结果与 mp_spherical_yn 函数的结果相等
        assert_mpmath_equal(
            lambda n, z: sc.spherical_yn(int(n.real), z),
            # 将 mp_spherical_yn 函数包装为异常处理函数，处理可能的异常情况
            exception_to_nan(mp_spherical_yn),
            # 参数列表，包括 n 和 z 的取值范围
            [IntArg(0, 200), ComplexArg()],
        )

    def test_spherical_in(self):
        # 定义一个函数 mp_spherical_in，计算球面贝塞尔函数 I_n(z)
        def mp_spherical_in(n, z):
            # 将 z 转换为 mpmath 数字
            arg = mpmath.mpmathify(z)
            # 计算球面贝塞尔函数 I_n(z) 的复数形式
            out = (mpmath.besseli(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            # 如果 z 是实数，则返回实部；否则返回整个结果
            if arg.imag == 0:
                return out.real
            else:
                return out

        # 使用 assert_mpmath_equal 函数断言球面贝塞尔函数 I_n(z) 的计算结果与 mp_spherical_in 函数的结果相等
        assert_mpmath_equal(
            lambda n, z: sc.spherical_in(int(n), z),
            # 将 mp_spherical_in 函数包装为异常处理函数，处理可能的异常情况
            exception_to_nan(mp_spherical_in),
            # 参数列表，包括 n 和 z 的取值范围
            [IntArg(0, 200), Arg()],
            # 设置计算精度和绝对容差
            dps=200,
            atol=10**(-278),
        )

    def test_spherical_in_complex(self):
        # 定义一个函数 mp_spherical_in，计算复数情况下的球面贝塞尔函数 I_n(z)
        def mp_spherical_in(n, z):
            # 将 z 转换为 mpmath 数字
            arg = mpmath.mpmathify(z)
            # 计算球面贝塞尔函数 I_n(z) 的复数形式
            out = (mpmath.besseli(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            # 如果 z 是实数，则返回实部；否则返回整个结果
            if arg.imag == 0:
                return out.real
            else:
                return out

        # 使用 assert_mpmath_equal 函数断言球面贝塞尔函数 I_n(z) 的计算结果与 mp_spherical_in 函数的结果相等
        assert_mpmath_equal(
            lambda n, z: sc.spherical_in(int(n.real), z),
            # 将 mp_spherical_in 函数包装为异常处理函数，处理可能的异常情况
            exception_to_nan(mp_spherical_in),
            # 参数列表，包括 n 和 z 的取值范围
            [IntArg(0, 200), ComplexArg()],
        )

    def test_spherical_kn(self):
        # 定义一个函数 mp_spherical_kn，计算球面贝塞尔函数 K_n(z)
        def mp_spherical_kn(n, z):
            # 计算球面贝塞尔函数 K_n(z) 的复数形式
            out = (mpmath.besselk(n + mpmath.mpf(1)/2, z) *
                   mpmath.sqrt(mpmath.pi/(2*mpmath.mpmathify(z))))
            # 如果 z 是实数，则返回实部；否则返回整个结果
            if mpmath.mpmathify(z).imag == 0:
                return out.real
            else:
                return out

        # 使用 assert_mpmath_equal 函数断言球面贝塞尔函数 K_n(z) 的计算结果与 mp_spherical_kn 函数的结果相等
        assert_mpmath_equal(
            lambda n, z: sc.spherical_kn(int(n), z),
            # 将 mp_spherical_kn 函数包装为异常处理函数，处理可能的异常情况
            exception_to_nan(mp_spherical_kn),
            # 参数列表，包括 n 和 z 的取值范围
            [IntArg(0, 150), Arg()],
            # 设置计算精度
            dps=100,
        )

    @pytest.mark.xfail(run=False,
                       reason="Accuracy issues near z = -1 inherited from kv.")
    def test_spherical_kn_complex(self):
        # 定义一个函数 mp_spherical_kn，计算复数情况下的球面贝塞尔函数 K_n(z)
        def mp_spherical_kn(n, z):
            # 将 z 转换为 mpmath 数字
            arg = mpmath.mpmathify(z)
            # 计算球面贝塞尔函数 K_n(z) 的复数形式
            out = (mpmath.besselk(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            # 如果 z 是实数，则返回实部；否则返回整个结果
            if arg.imag == 0:
                return out.real
            else:
                return out

        # 使用 assert_mpmath_equal 函数断言球面贝塞尔函数 K_n(z) 的计算结果与 mp_spherical_kn 函数的结果相等
        assert_mpmath_equal(
            lambda n, z: sc.spherical_kn(int(n.real), z),
            # 将 mp_spherical_kn 函数包装为异常处理函数，处理可能的异常情况
            exception_to_nan(mp_spherical_kn),
            # 参数列表，包括 n 和 z 的取值范围
            [IntArg(0, 200), ComplexArg()],
            # 设置计算精度
            dps=200,
        )
```