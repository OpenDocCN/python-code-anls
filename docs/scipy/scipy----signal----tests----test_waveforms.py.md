# `D:\src\scipysrc\scipy\scipy\signal\tests\test_waveforms.py`

```
# 导入 NumPy 库，并使用别名 np
import numpy as np
# 从 numpy.testing 模块中导入多个断言函数，用于测试数值计算结果的准确性
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_, assert_allclose, assert_array_equal)
# 从 pytest 模块中导入 raises 函数并重命名为 assert_raises
from pytest import raises as assert_raises

# 导入 scipy.signal._waveforms 模块中的 waveforms 对象
import scipy.signal._waveforms as waveforms

# 这些 chirp_* 函数是由 chirp() 返回信号的瞬时频率函数

# 定义线性扫频信号的瞬时频率函数
def chirp_linear(t, f0, f1, t1):
    f = f0 + (f1 - f0) * t / t1
    return f

# 定义二次扫频信号的瞬时频率函数
def chirp_quadratic(t, f0, f1, t1, vertex_zero=True):
    if vertex_zero:
        f = f0 + (f1 - f0) * t**2 / t1**2
    else:
        f = f1 - (f1 - f0) * (t1 - t)**2 / t1**2
    return f

# 定义几何扫频信号的瞬时频率函数
def chirp_geometric(t, f0, f1, t1):
    f = f0 * (f1/f0)**(t/t1)
    return f

# 定义双曲线扫频信号的瞬时频率函数
def chirp_hyperbolic(t, f0, f1, t1):
    f = f0*f1*t1 / ((f0 - f1)*t + f1*t1)
    return f

# 计算频率
def compute_frequency(t, theta):
    """
    计算 theta'(t)/(2*pi)，其中 theta'(t) 是 theta(t) 的导数。
    """
    # 假设 theta 和 t 是 1-D NumPy 数组
    # 假设 t 是均匀间隔的
    dt = t[1] - t[0]
    f = np.diff(theta)/(2*np.pi) / dt
    tf = 0.5*(t[1:] + t[:-1])
    return tf, f

# 定义测试类 TestChirp
class TestChirp:

    # 测试线性扫频信号在 t=0 时的返回值
    def test_linear_at_zero(self):
        w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='linear')
        assert_almost_equal(w, 1.0)

    # 测试线性扫频信号的频率计算准确性（第一种情况）
    def test_linear_freq_01(self):
        method = 'linear'
        f0 = 1.0
        f1 = 2.0
        t1 = 1.0
        t = np.linspace(0, t1, 100)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_linear(tf, f0, f1, t1)))
        assert_(abserr < 1e-6)

    # 测试线性扫频信号的频率计算准确性（第二种情况）
    def test_linear_freq_02(self):
        method = 'linear'
        f0 = 200.0
        f1 = 100.0
        t1 = 10.0
        t = np.linspace(0, t1, 100)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_linear(tf, f0, f1, t1)))
        assert_(abserr < 1e-6)

    # 测试二次扫频信号在 t=0 时的返回值
    def test_quadratic_at_zero(self):
        w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='quadratic')
        assert_almost_equal(w, 1.0)

    # 测试二次扫频信号在 t=0 时的返回值（第二种情况）
    def test_quadratic_at_zero2(self):
        w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='quadratic',
                            vertex_zero=False)
        assert_almost_equal(w, 1.0)

    # 测试二次扫频信号的频率计算准确性（第一种情况）
    def test_quadratic_freq_01(self):
        method = 'quadratic'
        f0 = 1.0
        f1 = 2.0
        t1 = 1.0
        t = np.linspace(0, t1, 2000)
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        tf, f = compute_frequency(t, phase)
        abserr = np.max(np.abs(f - chirp_quadratic(tf, f0, f1, t1)))
        assert_(abserr < 1e-6)
    # 定义测试函数：测试二次频率调制方法（quadratic）
    def test_quadratic_freq_02(self):
        # 设置方法名为'quadratic'
        method = 'quadratic'
        # 设定起始频率为20.0 Hz
        f0 = 20.0
        # 设定结束频率为10.0 Hz
        f1 = 10.0
        # 设定持续时间为10.0秒
        t1 = 10.0
        # 在时间范围[0, t1]内生成2000个均匀分布的时间点
        t = np.linspace(0, t1, 2000)
        # 调用_waveforms模块中的_chirp_phase函数，计算相位
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        # 计算频率和时间
        tf, f = compute_frequency(t, phase)
        # 计算与二次调制期望值的频率误差的最大绝对值
        abserr = np.max(np.abs(f - chirp_quadratic(tf, f0, f1, t1)))
        # 断言：频率误差应小于1e-6
        assert_(abserr < 1e-6)

    # 定义测试函数：测试对数频率调制方法（logarithmic）在时间为0时的情况
    def test_logarithmic_at_zero(self):
        # 调用waveforms模块的chirp函数，生成对数调制波形，设定参数t=0, f0=1.0, f1=2.0, t1=1.0, method='logarithmic'
        w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='logarithmic')
        # 断言：生成的波形w应接近1.0
        assert_almost_equal(w, 1.0)

    # 定义测试函数：测试对数频率调制方法（logarithmic），频率变化从1.0 Hz到2.0 Hz，时间持续10.0秒
    def test_logarithmic_freq_01(self):
        # 设置方法名为'logarithmic'
        method = 'logarithmic'
        # 设定起始频率为1.0 Hz
        f0 = 1.0
        # 设定结束频率为2.0 Hz
        f1 = 2.0
        # 设定持续时间为10.0秒
        t1 = 1.0
        # 在时间范围[0, t1]内生成10000个均匀分布的时间点
        t = np.linspace(0, t1, 10000)
        # 调用_waveforms模块中的_chirp_phase函数，计算相位
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        # 计算频率和时间
        tf, f = compute_frequency(t, phase)
        # 计算与几何对数调制期望值的频率误差的最大绝对值
        abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
        # 断言：频率误差应小于1e-6
        assert_(abserr < 1e-6)

    # 定义测试函数：测试对数频率调制方法（logarithmic），频率变化从200.0 Hz到100.0 Hz，时间持续10.0秒
    def test_logarithmic_freq_02(self):
        # 设置方法名为'logarithmic'
        method = 'logarithmic'
        # 设定起始频率为200.0 Hz
        f0 = 200.0
        # 设定结束频率为100.0 Hz
        f1 = 100.0
        # 设定持续时间为10.0秒
        t1 = 10.0
        # 在时间范围[0, t1]内生成10000个均匀分布的时间点
        t = np.linspace(0, t1, 10000)
        # 调用_waveforms模块中的_chirp_phase函数，计算相位
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        # 计算频率和时间
        tf, f = compute_frequency(t, phase)
        # 计算与几何对数调制期望值的频率误差的最大绝对值
        abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
        # 断言：频率误差应小于1e-6
        assert_(abserr < 1e-6)

    # 定义测试函数：测试对数频率调制方法（logarithmic），起始频率和结束频率都为100.0 Hz，时间持续10.0秒
    def test_logarithmic_freq_03(self):
        # 设置方法名为'logarithmic'
        method = 'logarithmic'
        # 设定起始频率为100.0 Hz
        f0 = 100.0
        # 设定结束频率为100.0 Hz
        f1 = 100.0
        # 设定持续时间为10.0秒
        t1 = 10.0
        # 在时间范围[0, t1]内生成10000个均匀分布的时间点
        t = np.linspace(0, t1, 10000)
        # 调用_waveforms模块中的_chirp_phase函数，计算相位
        phase = waveforms._chirp_phase(t, f0, t1, f1, method)
        # 计算频率和时间
        tf, f = compute_frequency(t, phase)
        # 计算与几何对数调制期望值的频率误差的最大绝对值
        abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
        # 断言：频率误差应小于1e-6
        assert_(abserr < 1e-6)

    # 定义测试函数：测试双曲线频率调制方法（hyperbolic）在时间为0时的情况
    def test_hyperbolic_at_zero(self):
        # 调用waveforms模块的chirp函数，生成双曲线调制波形，设定参数t=0, f0=10.0, f1=1.0, t1=1.0, method='hyperbolic'
        w = waveforms.chirp(t=0, f0=10.0, f1=1.0, t1=1.0, method='hyperbolic')
        # 断言：生成的波形w应接近1.0
        assert_almost_equal(w, 1.0)

    # 定义测试函数：测试双曲线频率调制方法（hyperbolic），在时间持续1.0秒内测试多个起始和结束频率对
    def test_hyperbolic_freq_01(self):
        # 设置方法名为'hyperbolic'
        method = 'hyperbolic'
        # 设定持续时间为1.0秒
        t1 = 1.0
        # 在时间范围[0, t1]内生成10000个均匀分布的时间点
        t = np.linspace(0, t1, 10000)
        # 不同起始和结束频率的测试用例列表
        cases = [[10.0, 1.0],
                 [1.0, 10.0],
                 [-10.0, -1.0],
                 [-1.0, -10.0]]
        # 遍历每个测试用例
        for f0, f1 in cases:
            # 调用_waveforms模块中的_chirp_phase函数，计算相位
            phase = waveforms._chirp_phase(t, f0, t1, f1, method)
            # 计算频率和时间
            tf, f = compute_frequency(t, phase)
            # 计算与双曲线调制期望值的频率误差
            expected = chirp_hyperbolic(tf, f0, f1, t1)
            # 断言：计算得到的频率f应接近期望值expected
            assert
    # 测试函数：测试 chirp 函数对整数输入的处理
    def test_integer_t1(self):
        # 设置初始参数
        f0 = 10.0
        f1 = 20.0
        t = np.linspace(-1, 1, 11)
        # 使用浮点数 t1 调用 chirp 函数，并记录浮点数结果
        t1 = 3.0
        float_result = waveforms.chirp(t, f0, t1, f1)
        # 使用整数 t1 调用 chirp 函数，并记录整数结果
        t1 = 3
        int_result = waveforms.chirp(t, f0, t1, f1)
        # 断言整数输入 t1=3 的 chirp 函数结果与浮点数输入结果相等
        err_msg = "Integer input 't1=3' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)

    # 测试函数：测试 chirp 函数对整数输入的处理
    def test_integer_f0(self):
        # 设置初始参数
        f1 = 20.0
        t1 = 3.0
        t = np.linspace(-1, 1, 11)
        # 使用浮点数 f0 调用 chirp 函数，并记录浮点数结果
        f0 = 10.0
        float_result = waveforms.chirp(t, f0, t1, f1)
        # 使用整数 f0 调用 chirp 函数，并记录整数结果
        f0 = 10
        int_result = waveforms.chirp(t, f0, t1, f1)
        # 断言整数输入 f0=10 的 chirp 函数结果与浮点数输入结果相等
        err_msg = "Integer input 'f0=10' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)

    # 测试函数：测试 chirp 函数对整数输入的处理
    def test_integer_f1(self):
        # 设置初始参数
        f0 = 10.0
        t1 = 3.0
        t = np.linspace(-1, 1, 11)
        # 使用浮点数 f1 调用 chirp 函数，并记录浮点数结果
        f1 = 20.0
        float_result = waveforms.chirp(t, f0, t1, f1)
        # 使用整数 f1 调用 chirp 函数，并记录整数结果
        f1 = 20
        int_result = waveforms.chirp(t, f0, t1, f1)
        # 断言整数输入 f1=20 的 chirp 函数结果与浮点数输入结果相等
        err_msg = "Integer input 'f1=20' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)

    # 测试函数：测试 chirp 函数对整数输入的处理
    def test_integer_all(self):
        # 设置初始参数
        f0 = 10
        t1 = 3
        f1 = 20
        t = np.linspace(-1, 1, 11)
        # 使用浮点数 f0, t1, f1 调用 chirp 函数，并记录浮点数结果
        float_result = waveforms.chirp(t, float(f0), float(t1), float(f1))
        # 使用整数 f0, t1, f1 调用 chirp 函数，并记录整数结果
        int_result = waveforms.chirp(t, f0, t1, f1)
        # 断言整数输入 f0=10, t1=3, f1=20 的 chirp 函数结果与浮点数输入结果相等
        err_msg = "Integer input 'f0=10, t1=3, f1=20' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)
class TestSweepPoly:

    def test_sweep_poly_quad1(self):
        # 创建一个二次多项式 p(x) = x^2 + 1
        p = np.poly1d([1.0, 0.0, 1.0])
        # 在时间范围 [0, 3.0] 内生成 10000 个均匀分布的点
        t = np.linspace(0, 3.0, 10000)
        # 计算扫频信号的相位
        phase = waveforms._sweep_poly_phase(t, p)
        # 根据时间和相位计算频率和频率响应
        tf, f = compute_frequency(t, phase)
        # 计算期望的频率响应
        expected = p(tf)
        # 计算实际频率响应与期望之间的最大绝对误差
        abserr = np.max(np.abs(f - expected))
        # 断言最大误差小于 1e-6
        assert_(abserr < 1e-6)

    def test_sweep_poly_const(self):
        # 创建一个常数多项式 p(x) = 2.0
        p = np.poly1d(2.0)
        t = np.linspace(0, 3.0, 10000)
        phase = waveforms._sweep_poly_phase(t, p)
        tf, f = compute_frequency(t, phase)
        expected = p(tf)
        abserr = np.max(np.abs(f - expected))
        assert_(abserr < 1e-6)

    def test_sweep_poly_linear(self):
        # 创建一个一次多项式 p(x) = -x + 10
        p = np.poly1d([-1.0, 10.0])
        t = np.linspace(0, 3.0, 10000)
        phase = waveforms._sweep_poly_phase(t, p)
        tf, f = compute_frequency(t, phase)
        expected = p(tf)
        abserr = np.max(np.abs(f - expected))
        assert_(abserr < 1e-6)

    def test_sweep_poly_quad2(self):
        # 创建一个二次多项式 p(x) = x^2 - 2
        p = np.poly1d([1.0, 0.0, -2.0])
        t = np.linspace(0, 3.0, 10000)
        phase = waveforms._sweep_poly_phase(t, p)
        tf, f = compute_frequency(t, phase)
        expected = p(tf)
        abserr = np.max(np.abs(f - expected))
        assert_(abserr < 1e-6)

    def test_sweep_poly_cubic(self):
        # 创建一个三次多项式 p(x) = 2x^3 + x^2 - 2
        p = np.poly1d([2.0, 1.0, 0.0, -2.0])
        t = np.linspace(0, 2.0, 10000)
        phase = waveforms._sweep_poly_phase(t, p)
        tf, f = compute_frequency(t, phase)
        expected = p(tf)
        abserr = np.max(np.abs(f - expected))
        assert_(abserr < 1e-6)

    def test_sweep_poly_cubic2(self):
        """使用系数数组代替 poly1d 对象。"""
        p = np.array([2.0, 1.0, 0.0, -2.0])
        t = np.linspace(0, 2.0, 10000)
        phase = waveforms._sweep_poly_phase(t, p)
        tf, f = compute_frequency(t, phase)
        expected = np.poly1d(p)(tf)
        abserr = np.max(np.abs(f - expected))
        assert_(abserr < 1e-6)

    def test_sweep_poly_cubic3(self):
        """使用系数列表代替 poly1d 对象。"""
        p = [2.0, 1.0, 0.0, -2.0]
        t = np.linspace(0, 2.0, 10000)
        phase = waveforms._sweep_poly_phase(t, p)
        tf, f = compute_frequency(t, phase)
        expected = np.poly1d(p)(tf)
        abserr = np.max(np.abs(f - expected))
        assert_(abserr < 1e-6)


class TestGaussPulse:

    def test_integer_fc(self):
        # 测试整数类型的截止频率 fc=1000.0 和 fc=1000 的结果是否一致
        float_result = waveforms.gausspulse('cutoff', fc=1000.0)
        int_result = waveforms.gausspulse('cutoff', fc=1000)
        err_msg = "Integer input 'fc=1000' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)

    def test_integer_bw(self):
        # 测试整数类型的带宽 bw=1.0 和 bw=1 的结果是否一致
        float_result = waveforms.gausspulse('cutoff', bw=1.0)
        int_result = waveforms.gausspulse('cutoff', bw=1)
        err_msg = "Integer input 'bw=1' gives wrong result"
        assert_equal(int_result, float_result, err_msg=err_msg)
    # 定义测试函数，用于测试 gausspulse 函数对整数参数的处理
    def test_integer_bwr(self):
        # 调用 gausspulse 函数，使用浮点数参数 'bwr=-6.0'
        float_result = waveforms.gausspulse('cutoff', bwr=-6.0)
        # 再次调用 gausspulse 函数，使用整数参数 'bwr=-6'
        int_result = waveforms.gausspulse('cutoff', bwr=-6)
        # 错误消息字符串，用于在断言失败时提供更详细的信息
        err_msg = "Integer input 'bwr=-6' gives wrong result"
        # 断言整数参数和浮点数参数调用的结果相等，否则输出错误消息
        assert_equal(int_result, float_result, err_msg=err_msg)

    # 定义测试函数，用于测试 gausspulse 函数对整数参数的处理
    def test_integer_tpr(self):
        # 调用 gausspulse 函数，使用浮点数参数 'tpr=-60.0'
        float_result = waveforms.gausspulse('cutoff', tpr=-60.0)
        # 再次调用 gausspulse 函数，使用整数参数 'tpr=-60'
        int_result = waveforms.gausspulse('cutoff', tpr=-60)
        # 错误消息字符串，用于在断言失败时提供更详细的信息
        err_msg = "Integer input 'tpr=-60' gives wrong result"
        # 断言整数参数和浮点数参数调用的结果相等，否则输出错误消息
        assert_equal(int_result, float_result, err_msg=err_msg)
# 定义一个测试类 TestUnitImpulse，用于测试 unit_impulse 函数的各种输入情况

class TestUnitImpulse:

    # 测试当没有索引参数时的 unit_impulse 函数的输出
    def test_no_index(self):
        # 断言调用 unit_impulse(7) 的结果等于预期的 [1, 0, 0, 0, 0, 0, 0]
        assert_array_equal(waveforms.unit_impulse(7), [1, 0, 0, 0, 0, 0, 0])
        # 断言调用 unit_impulse((3, 3)) 的结果等于预期的二维数组
        assert_array_equal(waveforms.unit_impulse((3, 3)),
                           [[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    # 测试带有索引参数时的 unit_impulse 函数的输出
    def test_index(self):
        # 断言调用 unit_impulse(10, 3) 的结果等于预期的 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        assert_array_equal(waveforms.unit_impulse(10, 3),
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        # 断言调用 unit_impulse((3, 3), (1, 1)) 的结果等于预期的二维数组
        assert_array_equal(waveforms.unit_impulse((3, 3), (1, 1)),
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        # 测试广播功能
        # 调用 unit_impulse((4, 4), 2) 并断言其结果与预期的二维数组相等
        imp = waveforms.unit_impulse((4, 4), 2)
        assert_array_equal(imp, np.array([[0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 0]]))

    # 测试 'mid' 索引参数时的 unit_impulse 函数的输出
    def test_mid(self):
        # 断言调用 unit_impulse((3, 3), 'mid') 的结果等于预期的二维数组
        assert_array_equal(waveforms.unit_impulse((3, 3), 'mid'),
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        # 断言调用 unit_impulse(9, 'mid') 的结果等于预期的 [0, 0, 0, 0, 1, 0, 0, 0, 0]
        assert_array_equal(waveforms.unit_impulse(9, 'mid'),
                           [0, 0, 0, 0, 1, 0, 0, 0, 0])

    # 测试不同 dtype 参数时的 unit_impulse 函数的输出
    def test_dtype(self):
        # 调用 unit_impulse(7) 并断言其返回结果的数据类型是浮点数
        imp = waveforms.unit_impulse(7)
        assert_(np.issubdtype(imp.dtype, np.floating))

        # 调用 unit_impulse(5, 3, dtype=int) 并断言其返回结果的数据类型是整数
        imp = waveforms.unit_impulse(5, 3, dtype=int)
        assert_(np.issubdtype(imp.dtype, np.integer))

        # 调用 unit_impulse((5, 2), (3, 1), dtype=complex) 并断言其返回结果的数据类型是复数
        imp = waveforms.unit_impulse((5, 2), (3, 1), dtype=complex)
        assert_(np.issubdtype(imp.dtype, np.complexfloating))
```