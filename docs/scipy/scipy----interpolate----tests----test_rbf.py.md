# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_rbf.py`

```
# Created by John Travers, Robert Hetland, 2007
""" Test functions for rbf module """

import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
                           assert_almost_equal)
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf

# 定义可用的插值函数类型
FUNCTIONS = ('multiquadric', 'inverse multiquadric', 'gaussian',
             'cubic', 'quintic', 'thin-plate', 'linear')

# 检查一维 Rbf 插值函数是否通过节点插值
def check_rbf1d_interpolation(function):
    # 生成均匀分布的一维输入数据
    x = linspace(0, 10, 9)
    # 根据输入数据生成正弦函数值作为目标输出
    y = sin(x)
    # 创建 Rbf 插值对象
    rbf = Rbf(x, y, function=function)
    # 对输入数据进行插值
    yi = rbf(x)
    # 检查插值结果是否与目标输出接近
    assert_array_almost_equal(y, yi)
    # 验证插值函数在第一个节点处的表现
    assert_almost_equal(rbf(float(x[0])), y[0])

# 检查二维 Rbf 插值函数是否通过节点插值
def check_rbf2d_interpolation(function):
    # 生成随机分布的二维输入数据
    x = random.rand(50, 1) * 4 - 2
    y = random.rand(50, 1) * 4 - 2
    # 根据输入数据生成复杂的目标输出
    z = x * exp(-x ** 2 - 1j * y ** 2)
    # 创建 Rbf 插值对象
    rbf = Rbf(x, y, z, epsilon=2, function=function)
    # 对输入数据进行插值
    zi = rbf(x, y)
    # 调整插值结果的形状以与目标输出一致
    zi.shape = x.shape
    # 检查插值结果是否与目标输出接近
    assert_array_almost_equal(z, zi)

# 检查三维 Rbf 插值函数是否通过节点插值
def check_rbf3d_interpolation(function):
    # 生成随机分布的三维输入数据
    x = random.rand(50, 1) * 4 - 2
    y = random.rand(50, 1) * 4 - 2
    z = random.rand(50, 1) * 4 - 2
    # 根据输入数据生成复杂的目标输出
    d = x * exp(-x ** 2 - y ** 2)
    # 创建 Rbf 插值对象
    rbf = Rbf(x, y, z, d, epsilon=2, function=function)
    # 对输入数据进行插值
    di = rbf(x, y, z)
    # 调整插值结果的形状以与目标输出一致
    di.shape = x.shape
    # 检查插值结果是否与目标输出接近
    assert_array_almost_equal(di, d)

# 测试所有定义的插值函数类型的插值结果是否正确
def test_rbf_interpolation():
    for function in FUNCTIONS:
        check_rbf1d_interpolation(function)
        check_rbf2d_interpolation(function)
        check_rbf3d_interpolation(function)

# 检查二维多输出 Rbf 插值函数是否通过节点插值
def check_2drbf1d_interpolation(function):
    # 生成均匀分布的一维输入数据
    x = linspace(0, 10, 9)
    # 根据输入数据生成正弦和余弦函数值作为目标输出
    y0 = sin(x)
    y1 = cos(x)
    y = np.vstack([y0, y1]).T
    # 创建 Rbf 插值对象
    rbf = Rbf(x, y, function=function, mode='N-D')
    # 对输入数据进行插值
    yi = rbf(x)
    # 检查插值结果是否与目标输出接近
    assert_array_almost_equal(y, yi)
    # 验证插值函数在第一个节点处的表现
    assert_almost_equal(rbf(float(x[0])), y[0])

# 检查二维多输出 Rbf 插值函数是否通过节点插值
def check_2drbf2d_interpolation(function):
    # 生成随机分布的二维输入数据
    x = random.rand(50, ) * 4 - 2
    y = random.rand(50, ) * 4 - 2
    # 根据输入数据生成复杂的目标输出
    z0 = x * exp(-x ** 2 - 1j * y ** 2)
    z1 = y * exp(-y ** 2 - 1j * x ** 2)
    z = np.vstack([z0, z1]).T
    # 创建 Rbf 插值对象
    rbf = Rbf(x, y, z, epsilon=2, function=function, mode='N-D')
    # 对输入数据进行插值
    zi = rbf(x, y)
    # 调整插值结果的形状以与目标输出一致
    zi.shape = z.shape
    # 检查插值结果是否与目标输出接近
    assert_array_almost_equal(z, zi)

# 检查二维多输出 Rbf 插值函数是否通过节点插值
def check_2drbf3d_interpolation(function):
    # 生成随机分布的三维输入数据
    x = random.rand(50, ) * 4 - 2
    y = random.rand(50, ) * 4 - 2
    z = random.rand(50, ) * 4 - 2
    # 根据输入数据生成复杂的目标输出
    d0 = x * exp(-x ** 2 - y ** 2)
    d1 = y * exp(-y ** 2 - x ** 2)
    d = np.vstack([d0, d1]).T
    # 创建 Rbf 插值对象
    rbf = Rbf(x, y, z, d, epsilon=2, function=function, mode='N-D')
    # 对输入数据进行插值
    di = rbf(x, y, z)
    # 调整插值结果的形状以与目标输出一致
    di.shape = d.shape
    # 检查插值结果是否与目标输出接近
    assert_array_almost_equal(di, d)

# 测试所有定义的插值函数类型的二维多输出插值结果是否正确
def test_2drbf_interpolation():
    # 对于 FUNCTIONS 列表中的每个元素 function，依次调用以下三个函数进行检查
    for function in FUNCTIONS:
        # 调用函数 check_2drbf1d_interpolation，检查 function 的一维径向基函数插值
        check_2drbf1d_interpolation(function)
        # 调用函数 check_2drbf2d_interpolation，检查 function 的二维径向基函数插值
        check_2drbf2d_interpolation(function)
        # 调用函数 check_2drbf3d_interpolation，检查 function 的三维径向基函数插值
        check_2drbf3d_interpolation(function)
# 检查 Rbf 函数在远离节点处是否能很好地逼近平滑函数
def check_rbf1d_regularity(function, atol):
    # 定义输入数据 x，包含 9 个点，范围在 [0, 10] 内
    x = linspace(0, 10, 9)
    # 计算对应的正弦值作为目标输出 y
    y = sin(x)
    # 使用 Rbf 函数基于给定的 function 进行插值
    rbf = Rbf(x, y, function=function)
    # 在 [0, 10] 范围内生成 100 个点作为插值的目标点 xi
    xi = linspace(0, 10, 100)
    # 计算 Rbf 插值得到的 yi
    yi = rbf(xi)
    # 生成包含最大绝对差值的消息字符串
    msg = "abs-diff: %f" % abs(yi - sin(xi)).max()
    # 使用 allclose 函数检查 yi 是否在给定的绝对容差 atol 内与 sin(xi) 相近，否则抛出消息 msg
    assert_(allclose(yi, sin(xi), atol=atol), msg)


# 测试 Rbf 函数的平滑性
def test_rbf_regularity():
    # 定义不同函数的容差
    tolerances = {
        'multiquadric': 0.1,
        'inverse multiquadric': 0.15,
        'gaussian': 0.15,
        'cubic': 0.15,
        'quintic': 0.1,
        'thin-plate': 0.1,
        'linear': 0.2
    }
    # 对每个函数调用 check_rbf1d_regularity 进行测试
    for function in FUNCTIONS:
        check_rbf1d_regularity(function, tolerances.get(function, 1e-2))


# 检查 2-D Rbf 函数在远离节点处是否能很好地逼近平滑函数
def check_2drbf1d_regularity(function, atol):
    # 定义输入数据 x，包含 9 个点，范围在 [0, 10] 内
    x = linspace(0, 10, 9)
    # 计算对应的正弦和余弦值作为目标输出 y
    y0 = sin(x)
    y1 = cos(x)
    y = np.vstack([y0, y1]).T
    # 使用 Rbf 函数基于给定的 function 进行插值，指定 mode='N-D' 表示多维插值
    rbf = Rbf(x, y, function=function, mode='N-D')
    # 在 [0, 10] 范围内生成 100 个点作为插值的目标点 xi
    xi = linspace(0, 10, 100)
    # 计算 Rbf 插值得到的 yi
    yi = rbf(xi)
    # 生成包含最大绝对差值的消息字符串
    msg = "abs-diff: %f" % abs(yi - np.vstack([sin(xi), cos(xi)]).T).max()
    # 使用 allclose 函数检查 yi 是否在给定的绝对容差 atol 内与 np.vstack([sin(xi), cos(xi)]).T 相近，否则抛出消息 msg
    assert_(allclose(yi, np.vstack([sin(xi), cos(xi)]).T, atol=atol), msg)


# 测试 2-D Rbf 函数的平滑性
def test_2drbf_regularity():
    # 定义不同函数的容差
    tolerances = {
        'multiquadric': 0.1,
        'inverse multiquadric': 0.15,
        'gaussian': 0.15,
        'cubic': 0.15,
        'quintic': 0.1,
        'thin-plate': 0.15,
        'linear': 0.2
    }
    # 对每个函数调用 check_2drbf1d_regularity 进行测试
    for function in FUNCTIONS:
        check_2drbf1d_regularity(function, tolerances.get(function, 1e-2))


# 检查带有默认 epsilon 的 Rbf 函数是否存在过冲现象
def check_rbf1d_stability(function):
    # 生成一些数据（使用固定的随机种子以保证结果确定性）
    np.random.seed(1234)
    # 定义输入数据 x，包含 50 个点，范围在 [0, 10] 内
    x = np.linspace(0, 10, 50)
    # 生成随机扰动数据 z
    z = x + 4.0 * np.random.randn(len(x))

    # 使用 Rbf 函数基于给定的 function 进行插值
    rbf = Rbf(x, z, function=function)
    # 在 [0, 10] 范围内生成 1000 个点作为插值的目标点 xi
    xi = np.linspace(0, 10, 1000)
    # 计算 Rbf 插值得到的 yi
    yi = rbf(xi)

    # 减去线性趋势并确保没有尖峰
    assert_(np.abs(yi-xi).max() / np.abs(z-x).max() < 1.1)


# 测试 Rbf 函数的稳定性
def test_rbf_stability():
    # 对每个函数调用 check_rbf1d_stability 进行测试
    for function in FUNCTIONS:
        check_rbf1d_stability(function)


# 检查默认构造函数能够正确构造 Rbf 类，并进行平滑性测试
def test_default_construction():
    # 定义输入数据 x，包含 9 个点，范围在 [0, 10] 内
    x = linspace(0,10,9)
    # 计算对应的正弦值作为目标输出 y
    y = sin(x)
    # 使用默认多孔径基函数构造 Rbf 类
    rbf = Rbf(x, y)
    # 计算 Rbf 插值得到的 yi
    yi = rbf(x)
    # 使用 assert_array_almost_equal 函数检查 y 和 yi 的接近程度
    assert_array_almost_equal(y, yi)


# 检查能够使用 callable 的函数作为 Rbf 类的 function 参数构造 Rbf 类
def test_function_is_callable():
    # 定义输入数据 x，包含 9 个点，范围在 [0, 10] 内
    x = linspace(0,10,9)
    # 计算对应的正弦值作为目标输出 y
    y = sin(x)
    # 定义一个简单的可调用函数 linfunc
    def linfunc(x):
        return x
    # 使用 linfunc 函数构造 Rbf 类
    rbf = Rbf(x, y, function=linfunc)
    # 计算 Rbf 插值得到的 yi
    yi = rbf(x)
    # 使用 assert_array_almost_equal 函数检查 y 和 yi 的接近程度
    assert_array_almost_equal(y, yi)


# 检查能够使用带有两个参数的 callable 函数作为 Rbf 类的 function 参数构造 Rbf 类
def test_two_arg_function_is_callable():
    # 定义一个带有两个参数的函数 _func
    def _func(self, r):
        return self.epsilon + r

    # 定义输入数据 x，包含 9 个点，范围在 [0, 10] 内
    x = linspace(0,10,9)
    # 计算对应的正弦值作为目标输出 y
    y = sin(x)
    # 使用 Rbf 类创建一个径向基函数插值对象，传入 x, y 数据和函数类型 _func
    rbf = Rbf(x, y, function=_func)
    # 调用径向基函数插值对象 rbf 对象的 __call__ 方法，计算插值后的 y 值
    yi = rbf(x)
    # 使用 assert_array_almost_equal 函数检查计算得到的 yi 与原始数据 y 在精度上的近似性
    assert_array_almost_equal(y, yi)
# 定义测试函数，测试当 epsilon 参数为 None 时的情况
def test_rbf_epsilon_none():
    # 生成一个包含 9 个元素的等间距数组，范围从 0 到 10
    x = linspace(0, 10, 9)
    # 计算 x 中每个元素的正弦值，生成对应的 y 值数组
    y = sin(x)
    # 创建 Rbf 对象，使用 x 和 y 数组作为输入数据，epsilon 参数设置为 None
    Rbf(x, y, epsilon=None)

# 定义测试函数，测试在一维中共线点不会因为 epsilon = 0 而导致错误
def test_rbf_epsilon_none_collinear():
    # 检查在一维中共线的点不会因为 epsilon = 0 而导致错误
    # 定义 x, y, z 三个数组，分别表示三个坐标轴上的值
    x = [1, 2, 3]
    y = [4, 4, 4]
    z = [5, 6, 7]
    # 创建 Rbf 对象，使用 x, y, z 作为输入数据，epsilon 参数设置为 None
    rbf = Rbf(x, y, z, epsilon=None)
    # 断言检查 rbf 对象的 epsilon 值大于 0
    assert_(rbf.epsilon > 0)
```