# `D:\src\scipysrc\scipy\scipy\integrate\tests\test__quad_vec.py`

```
# 导入 pytest 模块，用于测试
import pytest

# 导入 numpy 库，并导入 assert_allclose 函数用于比较数值是否接近
import numpy as np
from numpy.testing import assert_allclose

# 导入 scipy 库中的 quad_vec 函数，用于向量化积分
from scipy.integrate import quad_vec

# 导入 multiprocessing.dummy 库中的 Pool 类，用于创建多线程池
from multiprocessing.dummy import Pool

# 使用 pytest 的 parametrize 装饰器，定义了一个参数化测试的标记 quadrature_params
quadrature_params = pytest.mark.parametrize(
    'quadrature', [None, "gk15", "gk21", "trapezoid"])


# 定义一个参数化测试函数，用于测试 quad_vec 函数的简单情况
@quadrature_params
def test_quad_vec_simple(quadrature):
    # 定义一个 numpy 数组 n，包含从 0 到 9 的整数
    n = np.arange(10)
    
    # 定义一个函数 f(x)，返回 x 的 n 次方
    def f(x):
        return x ** n
    
    # 循环遍历不同的 epsabs 值
    for epsabs in [0.1, 1e-3, 1e-6]:
        # 如果 quadrature 是 'trapezoid' 并且 epsabs 小于 1e-4，则跳过当前循环
        if quadrature == 'trapezoid' and epsabs < 1e-4:
            # slow: skip
            continue
        
        # 定义一个 kwargs 字典，包含 epsabs 和 quadrature 参数
        kwargs = dict(epsabs=epsabs, quadrature=quadrature)
        
        # 计算精确值，exact 是 2^(n+1)/(n+1)
        exact = 2**(n+1)/(n + 1)
        
        # 调用 quad_vec 函数进行积分计算，norm='max'，并使用 assert_allclose 进行断言
        res, err = quad_vec(f, 0, 2, norm='max', **kwargs)
        assert_allclose(res, exact, rtol=0, atol=epsabs)
        
        # 再次调用 quad_vec 函数，norm='2'，使用 np.linalg.norm 进行断言
        res, err = quad_vec(f, 0, 2, norm='2', **kwargs)
        assert np.linalg.norm(res - exact) < epsabs
        
        # 再次调用 quad_vec 函数，norm='max'，并指定 points=(0.5, 1.0)，使用 assert_allclose 进行断言
        res, err = quad_vec(f, 0, 2, norm='max', points=(0.5, 1.0), **kwargs)
        assert_allclose(res, exact, rtol=0, atol=epsabs)
        
        # 再次调用 quad_vec 函数，norm='max'，并设置 epsrel=1e-8、full_output=True、limit=10000，使用 assert_allclose 进行断言
        res, err, *rest = quad_vec(f, 0, 2, norm='max',
                                   epsrel=1e-8,
                                   full_output=True,
                                   limit=10000,
                                   **kwargs)
        assert_allclose(res, exact, rtol=0, atol=epsabs)


# 定义另一个参数化测试函数，用于测试 quad_vec 函数的无穷限情况
@quadrature_params
def test_quad_vec_simple_inf(quadrature):
    # 定义一个函数 f(x)，返回 1 / (1 + x^2)
    def f(x):
        return 1 / (1 + np.float64(x) ** 2)

    # 循环遍历不同的 epsabs 值
    for epsabs in [0.1, 1e-3, 1e-6]:
        # 如果 quadrature 是 'trapezoid' 并且 epsabs 小于 1e-4，则跳过当前循环
        if quadrature == 'trapezoid' and epsabs < 1e-4:
            # slow: skip
            continue
        
        # 定义一个 kwargs 字典，包含 norm='max'、epsabs 和 quadrature 参数
        kwargs = dict(norm='max', epsabs=epsabs, quadrature=quadrature)
        
        # 调用 quad_vec 函数进行积分计算，从 0 到无穷大，使用 assert_allclose 进行断言
        res, err = quad_vec(f, 0, np.inf, **kwargs)
        assert_allclose(res, np.pi/2, rtol=0, atol=max(epsabs, err))
        
        # 再次调用 quad_vec 函数进行积分计算，从 0 到负无穷，使用 assert_allclose 进行断言
        res, err = quad_vec(f, 0, -np.inf, **kwargs)
        assert_allclose(res, -np.pi/2, rtol=0, atol=max(epsabs, err))
        
        # 其余类似的测试情况，均调用 quad_vec 函数进行积分计算，并使用 assert_allclose 进行断言
        res, err = quad_vec(f, -np.inf, 0, **kwargs)
        assert_allclose(res, np.pi/2, rtol=0, atol=max(epsabs, err))
        
        res, err = quad_vec(f, np.inf, 0, **kwargs)
        assert_allclose(res, -np.pi/2, rtol=0, atol=max(epsabs, err))
        
        res, err = quad_vec(f, -np.inf, np.inf, **kwargs)
        assert_allclose(res, np.pi, rtol=0, atol=max(epsabs, err))
        
        res, err = quad_vec(f, np.inf, -np.inf, **kwargs)
        assert_allclose(res, -np.pi, rtol=0, atol=max(epsabs, err))
        
        res, err = quad_vec(f, np.inf, np.inf, **kwargs)
        assert_allclose(res, 0, rtol=0, atol=max(epsabs, err))
        
        res, err = quad_vec(f, -np.inf, -np.inf, **kwargs)
        assert_allclose(res, 0, rtol=0, atol=max(epsabs, err))
        
        res, err = quad_vec(f, 0, np.inf, points=(1.0, 2.0), **kwargs)
        assert_allclose(res, np.pi/2, rtol=0, atol=max(epsabs, err))
    
    # 定义另一个函数 f(x)，返回 np.sin(x + 2) / (1 + x^2)
    def f(x):
        return np.sin(x + 2) / (1 + x ** 2)
    
    # 定义精确值 exact
    exact = np.pi / np.e * np.sin(2)
    # 设置 epsabs 值
    epsabs = 1e-5
    # 调用 quad_vec 函数，对函数 f 在区间负无穷到正无穷上进行数值积分
    # 返回结果 res（积分结果）、err（估计的积分误差）、info（包含有关积分过程的详细信息）
    res, err, info = quad_vec(f, -np.inf, np.inf, limit=1000, norm='max', epsabs=epsabs,
                              quadrature=quadrature, full_output=True)
    
    # 使用断言确保积分信息状态为 1（成功完成积分）
    assert info.status == 1
    
    # 使用断言确保计算得到的积分结果 res 与精确值 exact 很接近，相对误差不超过 0，绝对误差不超过 epsabs 和 1.5 * err 中的较大者
    assert_allclose(res, exact, rtol=0, atol=max(epsabs, 1.5 * err))
def test_quad_vec_args():
    # 定义测试函数 f(x, a)，计算 x * (x + a) * [0, 1, 2]
    def f(x, a):
        return x * (x + a) * np.arange(3)
    
    # 设定参数 a 为 2
    a = 2
    # 精确结果为 [0, 4/3, 8/3]
    exact = np.array([0, 4/3, 8/3])

    # 调用 quad_vec 计算积分结果 res 和误差 err
    res, err = quad_vec(f, 0, 1, args=(a,))
    # 使用 assert_allclose 检查 res 是否接近 exact，容忍度设置为绝对误差 1e-4
    assert_allclose(res, exact, rtol=0, atol=1e-4)


def _lorenzian(x):
    # 定义洛伦兹函数
    return 1 / (1 + x**2)


@pytest.mark.fail_slow(10)
def test_quad_vec_pool():
    # 将函数 _lorenzian 赋值给 f
    f = _lorenzian
    # 调用 quad_vec 计算 f 在区间 (-∞, ∞) 的积分结果 res 和误差 err
    res, err = quad_vec(f, -np.inf, np.inf, norm='max', epsabs=1e-4, workers=4)
    # 使用 assert_allclose 检查 res 是否接近 π，容忍度设置为绝对误差 1e-4
    assert_allclose(res, np.pi, rtol=0, atol=1e-4)

    # 使用 Pool 创建一个进程池，设定最大进程数为 10
    with Pool(10) as pool:
        # 定义函数 f(x) = 1 / (1 + x ** 2)
        def f(x):
            return 1 / (1 + x ** 2)
        # 调用 quad_vec 计算 f 在区间 (-∞, ∞) 的积分结果 res 和忽略的误差 err
        res, _ = quad_vec(f, -np.inf, np.inf, norm='max', epsabs=1e-4, workers=pool.map)
        # 使用 assert_allclose 检查 res 是否接近 π，容忍度设置为绝对误差 1e-4
        assert_allclose(res, np.pi, rtol=0, atol=1e-4)


def _func_with_args(x, a):
    # 定义测试函数 f(x, a)，计算 x * (x + a) * [0, 1, 2]
    return x * (x + a) * np.arange(3)


@pytest.mark.fail_slow(10)
@pytest.mark.parametrize('extra_args', [2, (2,)])
@pytest.mark.parametrize('workers', [1, 10])
def test_quad_vec_pool_args(extra_args, workers):
    # 将函数 _func_with_args 赋值给 f
    f = _func_with_args
    # 精确结果为 [0, 4/3, 8/3]
    exact = np.array([0, 4/3, 8/3])

    # 调用 quad_vec 计算积分结果 res 和误差 err，使用额外参数 extra_args 和进程数 workers
    res, err = quad_vec(f, 0, 1, args=extra_args, workers=workers)
    # 使用 assert_allclose 检查 res 是否接近 exact，容忍度设置为绝对误差 1e-4
    assert_allclose(res, exact, rtol=0, atol=1e-4)

    # 使用 Pool 创建一个进程池，设定最大进程数为 workers
    with Pool(workers) as pool:
        # 调用 quad_vec 计算积分结果 res 和忽略的误差 err，使用额外参数 extra_args 和 pool.map
        res, err = quad_vec(f, 0, 1, args=extra_args, workers=pool.map)
        # 使用 assert_allclose 检查 res 是否接近 exact，容忍度设置为绝对误差 1e-4
        assert_allclose(res, exact, rtol=0, atol=1e-4)


@quadrature_params
def test_num_eval(quadrature):
    # 定义测试函数 f(x)，计算 x**5，并记录函数调用次数到 count[0]
    def f(x):
        count[0] += 1
        return x**5

    # 初始化函数调用次数 count 为 [0]
    count = [0]
    # 调用 quad_vec 计算 f 在区间 (0, 1) 的积分结果 res，并检查评估次数是否与 count[0] 相等
    res = quad_vec(f, 0, 1, norm='max', full_output=True, quadrature=quadrature)
    assert res[2].neval == count[0]


def test_info():
    # 定义返回形状为 (3, 2, 1) 的全 1 数组的函数 f(x)
    def f(x):
        return np.ones((3, 2, 1))

    # 调用 quad_vec 计算 f 在区间 (0, 1) 的积分结果 res、误差 err 和信息 info
    res, err, info = quad_vec(f, 0, 1, norm='max', full_output=True)

    # 检查 info 对象中的属性值是否符合预期
    assert info.success is True
    assert info.status == 0
    assert info.message == 'Target precision reached.'
    assert info.neval > 0
    assert info.intervals.shape[1] == 2
    assert info.integrals.shape == (info.intervals.shape[0], 3, 2, 1)
    assert info.errors.shape == (info.intervals.shape[0],)


def test_nan_inf():
    # 返回 NaN 的函数 f_nan(x)
    def f_nan(x):
        return np.nan

    # 返回无穷大的函数 f_inf(x)
    def f_inf(x):
        return np.inf if x < 0.1 else 1/x

    # 调用 quad_vec 计算 f_nan 在区间 (0, 1) 的积分结果 res、误差 err 和信息 info
    res, err, info = quad_vec(f_nan, 0, 1, full_output=True)
    # 检查 info 对象中的状态是否为 3（无效输入）
    assert info.status == 3

    # 调用 quad_vec 计算 f_inf 在区间 (0, 1) 的积分结果 res、误差 err 和信息 info
    res, err, info = quad_vec(f_inf, 0, 1, full_output=True)
    # 检查 info 对象中的状态是否为 3（无效输入）
    assert info.status == 3


@pytest.mark.parametrize('a,b', [(0, 1), (0, np.inf), (np.inf, 0),
                                 (-np.inf, np.inf), (np.inf, -np.inf)])
def test_points(a, b):
    # 检查初始区间分割是否根据 points 进行，通过检查连续的 15 个点函数评估是否位于 points 之间
    points = (0, 0.25, 0.5, 0.75, 1.0)
    points += tuple(-x for x in points)

    quadrature_points = 15
    interval_sets = []
    count = 0

    # 定义函数 f(x)，记录每个函数调用的 x 值到 interval_sets 中
    def f(x):
        nonlocal count

        if count % quadrature_points == 0:
            interval_sets.append(set())

        count += 1
        interval_sets[-1].add(float(x))
        return 0.0
    # 使用 `quad_vec` 函数计算在指定的区间 `[a, b]` 上的积分值，使用 `gk15` 15点 Gauss-Kronrod 积分法进行数值积分
    quad_vec(f, a, b, points=points, quadrature='gk15', limit=0)
    
    # 检查所有的点集是否位于单个 `points` 区间内
    for p in interval_sets:
        # 对每个点集 `p` 进行排序，并在已排序的 `points` 中搜索其位置 `j`
        j = np.searchsorted(sorted(points), tuple(p))
        # 使用断言确保所有的位置 `j` 都与第一个位置 `j[0]` 相同
        assert np.all(j == j[0])
# 定义一个测试函数，用于测试在使用 trapz 拟合时是否会出现弃用警告
def test_trapz_deprecation():
    # 使用 pytest 提供的上下文管理器，捕获对 `quadrature='trapz'` 的弃用调用警告
    with pytest.deprecated_call(match="`quadrature='trapz'`"):
        # 调用 quad_vec 函数，对函数 lambda x: x 在区间 [0, 1] 上进行数值积分，
        # 使用 trapz 方法进行积分计算
        quad_vec(lambda x: x, 0, 1, quadrature="trapz")
```