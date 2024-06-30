# `D:\src\scipysrc\scipy\scipy\signal\tests\test_savitzky_golay.py`

```
import pytest  # 导入 pytest 库，用于编写和运行测试
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import (assert_allclose, assert_equal,
                           assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal)  # 导入 NumPy 测试模块中的断言函数

from scipy.ndimage import convolve1d  # 导入 SciPy 库中的图像处理模块中的一维卷积函数

from scipy.signal import savgol_coeffs, savgol_filter  # 导入 SciPy 库中的信号处理模块中的 Savitzky-Golay 滤波相关函数
from scipy.signal._savitzky_golay import _polyder  # 导入 SciPy 库中的信号处理模块中的 Savitzky-Golay 滤波的多项式求导函数


def check_polyder(p, m, expected):
    """检查 _polyder 函数的输出是否符合预期。

    Args:
        p: 多项式系数数组
        m: 求导阶数
        expected: 预期的导数结果数组
    """
    dp = _polyder(p, m)  # 调用 _polyder 函数，计算多项式 p 的 m 阶导数
    assert_array_equal(dp, expected)  # 使用断言函数验证 dp 是否等于预期的导数结果数组


def test_polyder():
    """测试 _polyder 函数在不同输入情况下的输出是否正确。"""
    cases = [
        ([5], 0, [5]),  # 测试单项式 [5] 求 0 阶导数的结果是否为 [5]
        ([5], 1, [0]),  # 测试单项式 [5] 求 1 阶导数的结果是否为 [0]
        ([3, 2, 1], 0, [3, 2, 1]),  # 测试多项式 [3, 2, 1] 求 0 阶导数的结果是否为 [3, 2, 1]
        ([3, 2, 1], 1, [6, 2]),  # 测试多项式 [3, 2, 1] 求 1 阶导数的结果是否为 [6, 2]
        ([3, 2, 1], 2, [6]),  # 测试多项式 [3, 2, 1] 求 2 阶导数的结果是否为 [6]
        ([3, 2, 1], 3, [0]),  # 测试多项式 [3, 2, 1] 求 3 阶导数的结果是否为 [0]
        ([[3, 2, 1], [5, 6, 7]], 0, [[3, 2, 1], [5, 6, 7]]),  # 测试多维数组的情况
        ([[3, 2, 1], [5, 6, 7]], 1, [[6, 2], [10, 6]]),
        ([[3, 2, 1], [5, 6, 7]], 2, [[6], [10]]),
        ([[3, 2, 1], [5, 6, 7]], 3, [[0], [0]]),
    ]
    for p, m, expected in cases:
        check_polyder(np.array(p).T, m, np.array(expected).T)  # 调用 check_polyder 函数进行断言


#--------------------------------------------------------------------
# savgol_coeffs tests
#--------------------------------------------------------------------

def alt_sg_coeffs(window_length, polyorder, pos):
    """这是一个替代实现的 Savitzky-Golay 系数计算函数。

    使用 numpy.polyfit 和 numpy.polyval。结果应该与 savgol_coeffs() 的结果等价，
    但这个实现较慢。

    window_length 应为奇数。

    Args:
        window_length: 窗口长度
        polyorder: 多项式阶数
        pos: 中心位置，如果为 None，则默认为 window_length // 2

    Returns:
        h: 计算得到的 SG 系数数组
    """
    if pos is None:
        pos = window_length // 2  # 如果 pos 为 None，则将其设置为窗口长度的一半
    t = np.arange(window_length)  # 创建一个长度为 window_length 的数组 t
    unit = (t == pos).astype(int)  # 创建一个单位脉冲函数
    h = np.polyval(np.polyfit(t, unit, polyorder), t)  # 使用多项式拟合和求值计算 SG 系数
    return h


def test_sg_coeffs_trivial():
    """测试 savgol_coeffs 函数在简单情况下的输出是否正确。"""
    h = savgol_coeffs(1, 0)  # 测试窗口长度为 1，多项式阶数为 0 的情况
    assert_allclose(h, [1])  # 使用断言函数验证 h 是否等于预期值 [1]

    h = savgol_coeffs(3, 2)  # 测试窗口长度为 3，多项式阶数为 2 的情况
    assert_allclose(h, [0, 1, 0], atol=1e-10)  # 使用断言函数验证 h 是否等于预期值 [0, 1, 0]

    h = savgol_coeffs(5, 4)  # 测试窗口长度为 5，多项式阶数为 4 的情况
    assert_allclose(h, [0, 0, 1, 0, 0], atol=1e-10)  # 使用断言函数验证 h 是否等于预期值 [0, 0, 1, 0, 0]

    h = savgol_coeffs(5, 4, pos=1)  # 测试窗口长度为 5，多项式阶数为 4，中心位置为 1 的情况
    assert_allclose(h, [0, 0, 0, 1, 0], atol=1e-10)  # 使用断言函数验证 h 是否等于预期值 [0, 0, 0, 1, 0]

    h = savgol_coeffs(5, 4, pos=1, use='dot')  # 测试使用 dot 方法的情况
    assert_allclose(h, [0, 1, 0, 0, 0], atol=1e-10)  # 使用断言函数验证 h 是否等于预期值 [0, 1, 0, 0, 0]


def compare_coeffs_to_alt(window_length, order):
    """比较 savgol_coeffs 和 alt_sg_coeffs 函数的输出是否一致。

    Args:
        window_length: 窗口长度
        order: 多项式阶数
    """
    for pos in [None] + list(range(window_length)):
        h1 = savgol_coeffs(window_length, order, pos=pos, use='dot')  # 调用 savgol_coeffs 函数计算 SG 系数
        h2 = alt_sg_coeffs(window_length, order, pos=pos)  # 调用 alt_sg_coeffs 函数计算 SG 系数
        assert_allclose(h1, h2, atol=1e-10,  # 使用断言函数验证 h1 和 h2 是否在误差范围内一致
                        err_msg=("window_length = %d, order = %d, pos = %s" %
                                 (window_length, order, pos)))


def test_sg_coeffs_compare():
    """比较 savgol_coeffs() 和 alt_sg_coeffs() 函数的输出是否一致。"""
    for window_length in range(1, 8, 2):
        for order in range(window_length):
            compare_coeffs_to_alt(window_length, order)


def test_sg_coeffs_exact():
    # To be continued, as this part of the code was not provided in the request.
    pass
    # 定义多项式的阶数为4
    polyorder = 4
    # 定义窗口长度为9
    window_length = 9
    # 计算窗口长度的一半
    halflen = window_length // 2
    
    # 在区间[0, 21]上生成43个等间距的数值作为x的取值
    x = np.linspace(0, 21, 43)
    # 计算x中相邻两个数值的差值
    delta = x[1] - x[0]
    
    # 根据公式生成一个三次多项式作为y的取值
    y = 0.5 * x ** 3 - x
    
    # 使用指定的窗口长度和多项式阶数，计算Savitzky-Golay滤波器的系数
    h = savgol_coeffs(window_length, polyorder)
    # 对y应用卷积运算，得到滤波后的结果y0
    y0 = convolve1d(y, h)
    # 断言滤波后的结果在边界除去半个窗口长度的范围内与原始y相等
    assert_allclose(y0[halflen:-halflen], y[halflen:-halflen])
    
    # 对同样的输入数据进行验证，此时计算y的一阶导数dy作为精确结果
    dy = 1.5 * x ** 2 - 1
    # 使用指定的窗口长度、多项式阶数和导数阶数，以及x中相邻点的间距delta，计算滤波器系数
    h = savgol_coeffs(window_length, polyorder, deriv=1, delta=delta)
    # 对y应用卷积运算，得到一阶导数的结果y1
    y1 = convolve1d(y, h)
    # 断言滤波后的一阶导数结果在边界除去半个窗口长度的范围内与dy相等
    assert_allclose(y1[halflen:-halflen], dy[halflen:-halflen])
    
    # 对同样的输入数据进行验证，此时计算y的二阶导数d2y作为精确结果
    d2y = 3.0 * x
    # 使用指定的窗口长度、多项式阶数和导数阶数，以及x中相邻点的间距delta，计算滤波器系数
    h = savgol_coeffs(window_length, polyorder, deriv=2, delta=delta)
    # 对y应用卷积运算，得到二阶导数的结果y2
    y2 = convolve1d(y, h)
    # 断言滤波后的二阶导数结果在边界除去半个窗口长度的范围内与d2y相等
    assert_allclose(y2[halflen:-halflen], d2y[halflen:-halflen])
# 定义用于测试 savgol_coeffs 函数的函数 test_sg_coeffs_deriv
def test_sg_coeffs_deriv():
    # 采样的数据 x 是一个抛物线的样本数据，因此使用 savgol_coeffs 和阶数大于等于 2 的多项式应该给出精确的结果。
    i = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
    x = i ** 2 / 4  # 计算 x 的值
    dx = i / 2  # 计算 dx 的值
    d2x = np.full_like(i, 0.5)  # 创建一个与 i 相同大小的数组，并填充为 0.5
    for pos in range(x.size):
        # 调用 savgol_coeffs 函数，计算 coeffs0，并断言其与 x[pos] 的点积接近于 x[pos]，允许误差为 1e-10
        coeffs0 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot')
        assert_allclose(coeffs0.dot(x), x[pos], atol=1e-10)
        # 调用 savgol_coeffs 函数，计算 coeffs1，并断言其与 dx[pos] 的点积接近于 dx[pos]，允许误差为 1e-10
        coeffs1 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot', deriv=1)
        assert_allclose(coeffs1.dot(x), dx[pos], atol=1e-10)
        # 调用 savgol_coeffs 函数，计算 coeffs2，并断言其与 d2x[pos] 的点积接近于 d2x[pos]，允许误差为 1e-10
        coeffs2 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot', deriv=2)
        assert_allclose(coeffs2.dot(x), d2x[pos], atol=1e-10)


# 定义用于测试 savgol_coeffs 函数的函数 test_sg_coeffs_deriv_gt_polyorder
def test_sg_coeffs_deriv_gt_polyorder():
    """
    如果 deriv 大于 polyorder，系数应全部为 0。
    这是一个回归测试，用于检查 savgol_coeffs(5, polyorder=1, deriv=2) 等是否会引发错误。
    """
    # 调用 savgol_coeffs 函数，计算 coeffs，并断言其与全为零的数组相等
    coeffs = savgol_coeffs(5, polyorder=1, deriv=2)
    assert_array_equal(coeffs, np.zeros(5))
    # 调用 savgol_coeffs 函数，计算 coeffs，并断言其与全为零的数组相等
    coeffs = savgol_coeffs(7, polyorder=4, deriv=6)
    assert_array_equal(coeffs, np.zeros(7))


# 定义用于测试 savgol_coeffs 函数的函数 test_sg_coeffs_large
def test_sg_coeffs_large():
    # 测试当 window_length 和 polyorder 较大时，返回的系数数组是否对称。
    # 目的在于确保没有潜在的数值溢出。
    coeffs0 = savgol_coeffs(31, 9)
    assert_array_almost_equal(coeffs0, coeffs0[::-1])
    coeffs1 = savgol_coeffs(31, 9, deriv=1)
    assert_array_almost_equal(coeffs1, -coeffs1[::-1])


# --------------------------------------------------------------------
# savgol_coeffs tests for even window length
# --------------------------------------------------------------------


# 定义用于测试 savgol_coeffs 函数在窗口长度为偶数时的函数 test_sg_coeffs_even_window_length
def test_sg_coeffs_even_window_length():
    # 简单情况 - deriv=0, polyorder=0, 1
    window_lengths = [4, 6, 8, 10, 12, 14, 16]
    for length in window_lengths:
        # 调用 savgol_coeffs 函数，计算 h_p_d，并断言其与 1/length 接近
        h_p_d = savgol_coeffs(length, 0, 0)
        assert_allclose(h_p_d, 1/length)

    # 使用闭式表达式验证
    # deriv=1, polyorder=1, 2
    def h_p_d_closed_form_1(k, m):
        return 6*(k - 0.5)/((2*m + 1)*m*(2*m - 1))

    # deriv=2, polyorder=2
    def h_p_d_closed_form_2(k, m):
        numer = 15*(-4*m**2 + 1 + 12*(k - 0.5)**2)
        denom = 4*(2*m + 1)*(m + 1)*m*(m - 1)*(2*m - 1)
        return numer/denom
    # 对于每个窗口长度，进行以下操作
    for length in window_lengths:
        # 计算中点位置
        m = length//2
        
        # 使用闭合形式函数计算期望输出，倒序排列
        expected_output = [h_p_d_closed_form_1(k, m)
                           for k in range(-m + 1, m + 1)][::-1]
        
        # 调用 savgol_coeffs 函数计算实际输出，并进行断言检查是否接近期望输出
        actual_output = savgol_coeffs(length, 1, 1)
        assert_allclose(expected_output, actual_output)
        
        # 调用 savgol_coeffs 函数计算实际输出，并进行断言检查是否接近期望输出
        actual_output = savgol_coeffs(length, 2, 1)
        assert_allclose(expected_output, actual_output)
        
        # 使用闭合形式函数计算期望输出，倒序排列
        expected_output = [h_p_d_closed_form_2(k, m)
                           for k in range(-m + 1, m + 1)][::-1]
        
        # 调用 savgol_coeffs 函数计算实际输出，并进行断言检查是否接近期望输出
        actual_output = savgol_coeffs(length, 2, 2)
        assert_allclose(expected_output, actual_output)
        
        # 调用 savgol_coeffs 函数计算实际输出，并进行断言检查是否接近期望输出
        actual_output = savgol_coeffs(length, 3, 2)
        assert_allclose(expected_output, actual_output)
#--------------------------------------------------------------------
# savgol_filter tests
#--------------------------------------------------------------------

# 定义测试函数，用于测试 savgol_filter 函数的一些极端情况
def test_sg_filter_trivial():
    """ Test some trivial edge cases for savgol_filter()."""
    # 创建包含单个元素的数组 x
    x = np.array([1.0])
    # 对 x 应用 savgol_filter，期望输出和输入相同
    y = savgol_filter(x, 1, 0)
    # 断言 y 的值等于 [1.0]
    assert_equal(y, [1.0])

    # 对单个值的情况进行测试，窗口长度为 3，多项式阶数为 1，
    # 期望 y 的输出值为 [1.0]，这是 (-1,0), (0,3), (1,0) 直线拟合的平均值
    x = np.array([3.0])
    y = savgol_filter(x, 3, 1, mode='constant')
    # 断言 y 的值近似于 [1.0]，精度为小数点后 15 位
    assert_almost_equal(y, [1.0], decimal=15)

    # 对单个值的情况进行测试，使用 mode='nearest'
    x = np.array([3.0])
    y = savgol_filter(x, 3, 1, mode='nearest')
    # 断言 y 的值近似于 [3.0]，精度为小数点后 15 位
    assert_almost_equal(y, [3.0], decimal=15)

    # 对包含三个相同值的数组进行测试，使用 mode='wrap'
    x = np.array([1.0] * 3)
    y = savgol_filter(x, 3, 1, mode='wrap')
    # 断言 y 的值近似于 [1.0, 1.0, 1.0]，精度为小数点后 15 位
    assert_almost_equal(y, [1.0, 1.0, 1.0], decimal=15)


def test_sg_filter_basic():
    # 一些 savgol_filter 的基本测试用例
    x = np.array([1.0, 2.0, 1.0])
    y = savgol_filter(x, 3, 1, mode='constant')
    # 断言 y 的输出近似于 [1.0, 4.0 / 3, 1.0]
    assert_allclose(y, [1.0, 4.0 / 3, 1.0])

    y = savgol_filter(x, 3, 1, mode='mirror')
    # 断言 y 的输出近似于 [5.0 / 3, 4.0 / 3, 5.0 / 3]
    assert_allclose(y, [5.0 / 3, 4.0 / 3, 5.0 / 3])

    y = savgol_filter(x, 3, 1, mode='wrap')
    # 断言 y 的输出近似于 [4.0 / 3, 4.0 / 3, 4.0 / 3]
    assert_allclose(y, [4.0 / 3, 4.0 / 3, 4.0 / 3])


def test_sg_filter_2d():
    # 对 savgol_filter 进行二维数组的测试
    x = np.array([[1.0, 2.0, 1.0],
                  [2.0, 4.0, 2.0]])
    expected = np.array([[1.0, 4.0 / 3, 1.0],
                         [2.0, 8.0 / 3, 2.0]])
    y = savgol_filter(x, 3, 1, mode='constant')
    # 断言 y 的输出近似于 expected
    assert_allclose(y, expected)

    y = savgol_filter(x.T, 3, 1, mode='constant', axis=0)
    # 断言 y 的输出近似于 expected 的转置
    assert_allclose(y, expected.T)


def test_sg_filter_interp_edges():
    # 另一个测试低次多项式数据的例子，使用 mode='interp'，
    # 预期 savgol_filter 应该精确匹配整个数据集的精确解，包括边缘。
    t = np.linspace(-5, 5, 21)
    delta = t[1] - t[0]
    # 多项式测试数据
    x = np.array([t,
                  3 * t ** 2,
                  t ** 3 - t])
    dx = np.array([np.ones_like(t),
                   6 * t,
                   3 * t ** 2 - 1.0])
    d2x = np.array([np.zeros_like(t),
                    np.full_like(t, 6),
                    6 * t])

    window_length = 7

    y = savgol_filter(x, window_length, 3, axis=-1, mode='interp')
    # 断言 y 的输出近似于 x，允许误差为 1e-12
    assert_allclose(y, x, atol=1e-12)

    y1 = savgol_filter(x, window_length, 3, axis=-1, mode='interp',
                       deriv=1, delta=delta)
    # 断言 y1 的输出近似于 dx，允许误差为 1e-12
    assert_allclose(y1, dx, atol=1e-12)

    y2 = savgol_filter(x, window_length, 3, axis=-1, mode='interp',
                       deriv=2, delta=delta)
    # 断言 y2 的输出近似于 d2x，允许误差为 1e-12
    assert_allclose(y2, d2x, atol=1e-12)

    # 转置所有数据，并使用 axis=0 再次进行测试
    x = x.T
    dx = dx.T
    d2x = d2x.T

    y = savgol_filter(x, window_length, 3, axis=0, mode='interp')
    # 断言 y 的输出近似于 x，允许误差为 1e-12
    assert_allclose(y, x, atol=1e-12)
    # 使用 Savitzky-Golay 滤波器对输入数据 x 进行一阶导数的平滑处理
    y1 = savgol_filter(x, window_length, 3, axis=0, mode='interp',
                       deriv=1, delta=delta)
    # 使用 assert_allclose 函数验证 y1 是否与预期的一阶导数 dx 接近，允许误差为 1e-12
    
    # 使用 Savitzky-Golay 滤波器对输入数据 x 进行二阶导数的平滑处理
    y2 = savgol_filter(x, window_length, 3, axis=0, mode='interp',
                       deriv=2, delta=delta)
    # 使用 assert_allclose 函数验证 y2 是否与预期的二阶导数 d2x 接近，允许误差为 1e-12
def test_sg_filter_interp_edges_3d():
    # 测试在3维数组上使用 mode='interp' 模式。
    t = np.linspace(-5, 5, 21)  # 在[-5, 5]区间内生成21个等间距的数作为时间点 t
    delta = t[1] - t[0]  # 计算时间步长 delta
    x1 = np.array([t, -t])  # 创建第一个数组 x1，包含 t 和 -t
    x2 = np.array([t ** 2, 3 * t ** 2 + 5])  # 创建第二个数组 x2，包含 t 的平方和 3*t 的平方加上5
    x3 = np.array([t ** 3, 2 * t ** 3 + t ** 2 - 0.5 * t])  # 创建第三个数组 x3，包含 t 的立方和 2*t 的立方加上 t 的平方减去0.5*t
    dx1 = np.array([np.ones_like(t), -np.ones_like(t)])  # 创建第一个导数数组 dx1，包含 t 长度的1数组和-t 长度的1数组
    dx2 = np.array([2 * t, 6 * t])  # 创建第二个导数数组 dx2，包含 2*t 和 6*t
    dx3 = np.array([3 * t ** 2, 6 * t ** 2 + 2 * t - 0.5])  # 创建第三个导数数组 dx3，包含 3*t的平方和 6*t的平方加2*t减0.5

    # z 的形状为 (3, 2, 21)
    z = np.array([x1, x2, x3])  # 创建一个3维数组 z，包含 x1, x2, x3
    dz = np.array([dx1, dx2, dx3])  # 创建一个3维数组 dz，包含 dx1, dx2, dx3

    y = savgol_filter(z, 7, 3, axis=-1, mode='interp', delta=delta)  # 对 z 应用 Savitzky-Golay 滤波器，使用 mode='interp' 模式，保持 delta 为时间步长
    assert_allclose(y, z, atol=1e-10)  # 断言 y 应当与 z 接近，绝对误差小于1e-10

    dy = savgol_filter(z, 7, 3, axis=-1, mode='interp', deriv=1, delta=delta)  # 对 z 应用 Savitzky-Golay 滤波器，使用 mode='interp' 模式计算一阶导数，保持 delta 为时间步长
    assert_allclose(dy, dz, atol=1e-10)  # 断言 dy 应当与 dz 接近，绝对误差小于1e-10

    # z 的形状为 (3, 21, 2)
    z = np.array([x1.T, x2.T, x3.T])  # 创建一个3维数组 z，包含 x1, x2, x3 的转置
    dz = np.array([dx1.T, dx2.T, dx3.T])  # 创建一个3维数组 dz，包含 dx1, dx2, dx3 的转置

    y = savgol_filter(z, 7, 3, axis=1, mode='interp', delta=delta)  # 对 z 应用 Savitzky-Golay 滤波器，使用 mode='interp' 模式，沿第1轴进行滤波，保持 delta 为时间步长
    assert_allclose(y, z, atol=1e-10)  # 断言 y 应当与 z 接近，绝对误差小于1e-10

    dy = savgol_filter(z, 7, 3, axis=1, mode='interp', deriv=1, delta=delta)  # 对 z 应用 Savitzky-Golay 滤波器，使用 mode='interp' 模式计算一阶导数，沿第1轴进行滤波，保持 delta 为时间步长
    assert_allclose(dy, dz, atol=1e-10)  # 断言 dy 应当与 dz 接近，绝对误差小于1e-10

    # z 的形状为 (21, 3, 2)
    z = z.swapaxes(0, 1).copy()  # 将 z 的第0轴和第1轴交换，然后进行拷贝操作
    dz = dz.swapaxes(0, 1).copy()  # 将 dz 的第0轴和第1轴交换，然后进行拷贝操作

    y = savgol_filter(z, 7, 3, axis=0, mode='interp', delta=delta)  # 对 z 应用 Savitzky-Golay 滤波器，使用 mode='interp' 模式，沿第0轴进行滤波，保持 delta 为时间步长
    assert_allclose(y, z, atol=1e-10)  # 断言 y 应当与 z 接近，绝对误差小于1e-10

    dy = savgol_filter(z, 7, 3, axis=0, mode='interp', deriv=1, delta=delta)  # 对 z 应用 Savitzky-Golay 滤波器，使用 mode='interp' 模式计算一阶导数，沿第0轴进行滤波，保持 delta 为时间步长
    assert_allclose(dy, dz, atol=1e-10)  # 断言 dy 应当与 dz 接近，绝对误差小于1e-10


def test_sg_filter_valid_window_length_3d():
    """Tests that the window_length check is using the correct axis."""
    
    x = np.ones((10, 20, 30))  # 创建一个形状为 (10, 20, 30) 的全1数组 x

    savgol_filter(x, window_length=29, polyorder=3, mode='interp')  # 对 x 应用 Savitzky-Golay 滤波器，使用 mode='interp' 模式，window_length 设为29

    with pytest.raises(ValueError, match='window_length must be less than'):
        # 当 window_length 大于 x.shape[-1] 时应该抛出 ValueError 异常
        savgol_filter(x, window_length=31, polyorder=3, mode='interp')

    savgol_filter(x, window_length=9, polyorder=3, axis=0, mode='interp')  # 对 x 应用 Savitzky-Golay 滤波器，使用 mode='interp' 模式，沿第0轴进行滤波，window_length 设为9

    with pytest.raises(ValueError, match='window_length must be less than'):
        # 当 window_length 大于 x.shape[0] 时应该抛出 ValueError 异常
        savgol_filter(x, window_length=11, polyorder=3, axis=0, mode='interp')
```