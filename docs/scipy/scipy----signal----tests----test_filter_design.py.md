# `D:\src\scipysrc\scipy\scipy\signal\tests\test_filter_design.py`

```
import warnings
# 导入警告模块

from scipy._lib import _pep440
# 从 scipy 库中导入 _pep440 模块

import numpy as np
# 导入 NumPy 库，并使用 np 别名

from numpy.testing import (assert_array_almost_equal,
                           assert_array_almost_equal_nulp,
                           assert_array_equal, assert_array_less,
                           assert_equal, assert_,
                           assert_allclose, assert_warns, suppress_warnings)
# 从 NumPy 的测试模块中导入多个断言函数

import pytest
# 导入 pytest 测试框架

from pytest import raises as assert_raises
# 从 pytest 中导入 raises 函数，并使用 assert_raises 别名

from numpy import array, spacing, sin, pi, sort, sqrt
# 从 NumPy 中导入 array, spacing, sin, pi, sort, sqrt 函数

from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
                          buttap, butter, buttord, cheb1ap, cheb1ord, cheb2ap,
                          cheb2ord, cheby1, cheby2, ellip, ellipap, ellipord,
                          firwin, freqs_zpk, freqs, freqz, freqz_zpk,
                          gammatone, group_delay, iircomb, iirdesign, iirfilter,
                          iirnotch, iirpeak, lp2bp, lp2bs, lp2hp, lp2lp, normalize,
                          medfilt, order_filter,
                          sos2tf, sos2zpk, sosfreqz, tf2sos, tf2zpk, zpk2sos,
                          zpk2tf, bilinear_zpk, lp2lp_zpk, lp2hp_zpk, lp2bp_zpk,
                          lp2bs_zpk)
# 从 scipy.signal 中导入多个信号处理函数和类

from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
                                        _bessel_poly, _bessel_zeros)
# 从 scipy.signal._filter_design 中导入特定的函数和类

try:
    import mpmath
except ImportError:
    mpmath = None
# 尝试导入 mpmath 库，如果导入失败，则将 mpmath 设为 None


def mpmath_check(min_ver):
    return pytest.mark.skipif(
        mpmath is None
        or _pep440.parse(mpmath.__version__) < _pep440.Version(min_ver),
        reason=f"mpmath version >= {min_ver} required",
    )
# 定义一个函数 mpmath_check，用于检查 mpmath 版本是否符合要求，返回 pytest.mark.skipif 标记

class TestCplxPair:
    # 定义测试类 TestCplxPair

    def test_trivial_input(self):
        assert_equal(_cplxpair([]).size, 0)
        # 断言空输入的复数对数组大小为 0
        assert_equal(_cplxpair(1), 1)
        # 断言输入为 1 时，复数对为 1

    def test_output_order(self):
        assert_allclose(_cplxpair([1+1j, 1-1j]), [1-1j, 1+1j])
        # 断言复数对 [1+1j, 1-1j] 的排序结果符合预期

        a = [1+1j, 1+1j, 1, 1-1j, 1-1j, 2]
        b = [1-1j, 1+1j, 1-1j, 1+1j, 1, 2]
        assert_allclose(_cplxpair(a), b)
        # 断言复数对列表 a 的排序结果符合列表 b

        # points spaced around the unit circle
        z = np.exp(2j*pi*array([4, 3, 5, 2, 6, 1, 0])/7)
        z1 = np.copy(z)
        np.random.shuffle(z)
        assert_allclose(_cplxpair(z), z1)
        np.random.shuffle(z)
        assert_allclose(_cplxpair(z), z1)
        np.random.shuffle(z)
        assert_allclose(_cplxpair(z), z1)
        # 对单位圆周围的点进行测试，断言复数对排序后保持不变

        # Should be able to pair up all the conjugates
        x = np.random.rand(10000) + 1j * np.random.rand(10000)
        y = x.conj()
        z = np.random.rand(10000)
        x = np.concatenate((x, y, z))
        np.random.shuffle(x)
        c = _cplxpair(x)

        # Every other element of head should be conjugates:
        assert_allclose(c[0:20000:2], np.conj(c[1:20000:2]))
        # 每个复数对的头部应该是共轭的
        # Real parts of head should be in sorted order:
        assert_allclose(c[0:20000:2].real, np.sort(c[0:20000:2].real))
        # 头部的实部应按排序顺序排列
        # Tail should be sorted real numbers:
        assert_allclose(c[20000:], np.sort(c[20000:]))
        # 尾部应该是排序后的实数
    # 定义测试函数，验证_cplxpair函数对实数和整数输入的处理是否正确
    def test_real_integer_input(self):
        # 断言_cplxpair函数处理[2, 0, 1]应返回[0, 1, 2]
        assert_array_equal(_cplxpair([2, 0, 1]), [0, 1, 2])

    # 定义测试函数，验证_cplxpair函数对容差(tol)参数的处理是否正确
    def test_tolerances(self):
        # 计算机epsilon，即最小浮点数间的距离
        eps = spacing(1)
        # 断言_cplxpair函数处理[1j, -1j, 1+1j*eps]应返回[-1j, 1j, 1+1j*eps]，容差为2*eps
        assert_allclose(_cplxpair([1j, -1j, 1+1j*eps], tol=2*eps),
                        [-1j, 1j, 1+1j*eps])

        # 断言_cplxpair函数处理[-eps+1j, +eps-1j]应返回[-1j, +1j]，排序接近0的情况
        assert_allclose(_cplxpair([-eps+1j, +eps-1j]), [-1j, +1j])
        # 断言_cplxpair函数处理[+eps+1j, -eps-1j]应返回[-1j, +1j]，排序接近0的情况
        assert_allclose(_cplxpair([+eps+1j, -eps-1j]), [-1j, +1j])
        # 断言_cplxpair函数处理[+1j, -1j]应返回[-1j, +1j]
        assert_allclose(_cplxpair([+1j, -1j]), [-1j, +1j])

    # 定义测试函数，验证_cplxpair函数对不匹配共轭数的处理是否正确
    def test_unmatched_conjugates(self):
        # 断言_cplxpair函数处理[1+3j, 1-3j, 1+2j]应引发ValueError异常，因为1+2j无匹配项
        assert_raises(ValueError, _cplxpair, [1+3j, 1-3j, 1+2j])

        # 断言_cplxpair函数处理[1+3j, 1-3j, 1+2j, 1-3j]应引发ValueError异常，因为1+2j和1-3j无匹配项
        assert_raises(ValueError, _cplxpair, [1+3j, 1-3j, 1+2j, 1-3j])

        # 断言_cplxpair函数处理[1+3j, 1-3j, 1+3j]应引发ValueError异常，因为1+3j无匹配项
        assert_raises(ValueError, _cplxpair, [1+3j, 1-3j, 1+3j])

        # 断言_cplxpair函数处理[4+5j, 4+5j]应引发ValueError异常，因为不是共轭数
        assert_raises(ValueError, _cplxpair, [4+5j, 4+5j])
        # 断言_cplxpair函数处理[1-7j, 1-7j]应引发ValueError异常，因为不是共轭数
        assert_raises(ValueError, _cplxpair, [1-7j, 1-7j])

        # 断言_cplxpair函数处理[1+3j]应引发ValueError异常，因为没有成对出现的共轭数
        assert_raises(ValueError, _cplxpair, [1+3j])
        # 断言_cplxpair函数处理[1-3j]应引发ValueError异常，因为没有成对出现的共轭数
        assert_raises(ValueError, _cplxpair, [1-3j])
class TestCplxReal:

    def test_trivial_input(self):
        # 测试空列表输入时的情况
        assert_equal(_cplxreal([]), ([], []))
        # 测试输入为单个实数时的情况
        assert_equal(_cplxreal(1), ([], [1]))

    def test_output_order(self):
        # 测试复数根与实数根的顺序输出
        zc, zr = _cplxreal(np.roots(array([1, 0, 0, 1])))
        assert_allclose(np.append(zc, zr), [1/2 + 1j*sin(pi/3), -1])

        eps = spacing(1)

        a = [0+1j, 0-1j, eps + 1j, eps - 1j, -eps + 1j, -eps - 1j,
             1, 4, 2, 3, 0, 0,
             2+3j, 2-3j,
             1-eps + 1j, 1+2j, 1-2j, 1+eps - 1j,  # sorts out of order
             3+1j, 3+1j, 3+1j, 3-1j, 3-1j, 3-1j,
             2-3j, 2+3j]
        zc, zr = _cplxreal(a)
        # 验证排序后的复数部分
        assert_allclose(zc, [1j, 1j, 1j, 1+1j, 1+2j, 2+3j, 2+3j, 3+1j, 3+1j,
                             3+1j])
        # 验证排序后的实数部分
        assert_allclose(zr, [0, 0, 1, 2, 3, 4])

        z = array([1-eps + 1j, 1+2j, 1-2j, 1+eps - 1j, 1+eps+3j, 1-2*eps-3j,
                   0+1j, 0-1j, 2+4j, 2-4j, 2+3j, 2-3j, 3+7j, 3-7j, 4-eps+1j,
                   4+eps-2j, 4-1j, 4-eps+2j])

        zc, zr = _cplxreal(z)
        # 验证排序后的复数部分
        assert_allclose(zc, [1j, 1+1j, 1+2j, 1+3j, 2+3j, 2+4j, 3+7j, 4+1j,
                             4+2j])
        # 验证排序后的实数部分
        assert_equal(zr, [])

    def test_unmatched_conjugates(self):
        # 测试未匹配的共轭根情况
        # 1+2j 无法找到匹配的共轭根
        assert_raises(ValueError, _cplxreal, [1+3j, 1-3j, 1+2j])

        # 1+2j 和 1-3j 都无法找到匹配的共轭根
        assert_raises(ValueError, _cplxreal, [1+3j, 1-3j, 1+2j, 1-3j])

        # 1+3j 无法找到匹配的共轭根
        assert_raises(ValueError, _cplxreal, [1+3j, 1-3j, 1+3j])

        # 没有成对的根
        assert_raises(ValueError, _cplxreal, [1+3j])
        assert_raises(ValueError, _cplxreal, [1-3j])

    def test_real_integer_input(self):
        # 测试实数输入的情况
        zc, zr = _cplxreal([2, 0, 1, 4])
        assert_array_equal(zc, [])
        assert_array_equal(zr, [0, 1, 2, 4])


class TestTf2zpk:

    @pytest.mark.parametrize('dt', (np.float64, np.complex128))
    def test_simple(self, dt):
        z_r = np.array([0.5, -0.5])
        p_r = np.array([1.j / np.sqrt(2), -1.j / np.sqrt(2)])
        # 对零点和极点进行排序，以避免因排序问题而导致测试失败
        z_r.sort()
        p_r.sort()
        b = np.poly(z_r).astype(dt)
        a = np.poly(p_r).astype(dt)

        z, p, k = tf2zpk(b, a)
        z.sort()
        # 极点 `p` 的实部接近于0.0，因此按虚部排序
        p = p[np.argsort(p.imag)]

        assert_array_almost_equal(z, z_r)
        assert_array_almost_equal(p, p_r)
        assert_array_almost_equal(k, 1.)
        assert k.dtype == dt

    def test_bad_filter(self):
        # 用于回归测试的案例：更好地处理糟糕条件下的滤波器系数
        with suppress_warnings():
            warnings.simplefilter("error", BadCoefficients)
            assert_raises(BadCoefficients, tf2zpk, [1e-15], [1.0, 1.0])
    # 定义一个测试函数，用于测试恒等传输函数的功能
    def test_identity(self):
        """Test the identity transfer function."""
        # 初始化零点和极点列表为空
        z = []
        p = []
        # 设定增益因子为1.0
        k = 1.
        # 调用 zpk2tf 函数，将零点、极点、增益转换为传递函数的系数
        b, a = zpk2tf(z, p, k)
        # 设定预期的结果为单元素的数组 [1.0]
        b_r = np.array([1.])  # desired result
        a_r = np.array([1.])  # desired result
        # 检查返回值 b 是否与预期结果 b_r 相等
        assert_array_equal(b, b_r)
        # 检查返回值 b 的类型是否为 numpy 数组
        assert_(isinstance(b, np.ndarray))
        # 检查返回值 a 是否与预期结果 a_r 相等
        assert_array_equal(a, a_r)
        # 检查返回值 a 的类型是否为 numpy 数组
        assert_(isinstance(a, np.ndarray))
class TestSos2Zpk:

    def test_basic(self):
        # 定义一个二阶系统的状态空间表示
        sos = [[1, 0, 1, 1, 0, -0.81],
               [1, 0, 0, 1, 0, +0.49]]
        # 将二阶系统的状态空间表示转换为零极点增益表示
        z, p, k = sos2zpk(sos)
        # 预期的零点
        z2 = [1j, -1j, 0, 0]
        # 预期的极点
        p2 = [0.9, -0.9, 0.7j, -0.7j]
        # 预期的增益
        k2 = 1
        # 检查计算得到的零点与预期是否接近
        assert_array_almost_equal(sort(z), sort(z2), decimal=4)
        # 检查计算得到的极点与预期是否接近
        assert_array_almost_equal(sort(p), sort(p2), decimal=4)
        # 检查计算得到的增益与预期是否相等
        assert_array_almost_equal(k, k2)

        # 另一个二阶系统的状态空间表示
        sos = [[1.00000, +0.61803, 1.0000, 1.00000, +0.60515, 0.95873],
               [1.00000, -1.61803, 1.0000, 1.00000, -1.58430, 0.95873],
               [1.00000, +1.00000, 0.0000, 1.00000, +0.97915, 0.00000]]
        # 将另一个二阶系统的状态空间表示转换为零极点增益表示
        z, p, k = sos2zpk(sos)
        # 预期的零点
        z2 = [-0.3090 + 0.9511j, -0.3090 - 0.9511j, 0.8090 + 0.5878j,
              0.8090 - 0.5878j, -1.0000 + 0.0000j, 0]
        # 预期的极点
        p2 = [-0.3026 + 0.9312j, -0.3026 - 0.9312j, 0.7922 + 0.5755j,
              0.7922 - 0.5755j, -0.9791 + 0.0000j, 0]
        # 预期的增益
        k2 = 1
        # 检查计算得到的零点与预期是否接近
        assert_array_almost_equal(sort(z), sort(z2), decimal=4)
        # 检查计算得到的极点与预期是否接近
        assert_array_almost_equal(sort(p), sort(p2), decimal=4)

        # 另一个二阶系统的状态空间表示（使用numpy数组）
        sos = array([[1, 2, 3, 1, 0.2, 0.3],
                     [4, 5, 6, 1, 0.4, 0.5]])
        # 预期的零点
        z = array([-1 - 1.41421356237310j, -1 + 1.41421356237310j,
                  -0.625 - 1.05326872164704j, -0.625 + 1.05326872164704j])
        # 预期的极点
        p = array([-0.2 - 0.678232998312527j, -0.2 + 0.678232998312527j,
                  -0.1 - 0.538516480713450j, -0.1 + 0.538516480713450j])
        # 预期的增益
        k = 4
        # 将状态空间表示转换为零极点增益表示
        z2, p2, k2 = sos2zpk(sos)
        # 检查计算得到的零点与预期是否接近
        assert_allclose(_cplxpair(z2), z)
        # 检查计算得到的极点与预期是否接近
        assert_allclose(_cplxpair(p2), p)
        # 检查计算得到的增益与预期是否相等
        assert_allclose(k2, k)

    def test_fewer_zeros(self):
        """Test not the expected number of p/z (effectively at origin)."""
        # 使用butterworth滤波器生成一个SOS表示
        sos = butter(3, 0.1, output='sos')
        # 将状态空间表示转换为零极点增益表示
        z, p, k = sos2zpk(sos)
        # 断言零点的数量是否为预期值
        assert len(z) == 4
        # 断言极点的数量是否为预期值
        assert len(p) == 4

        # 使用带通Butterworth滤波器生成一个SOS表示
        sos = butter(12, [5., 30.], 'bandpass', fs=1200., analog=False,
                     output='sos')
        # 将状态空间表示转换为零极点增益表示，带有警告
        with pytest.warns(BadCoefficients, match='Badly conditioned'):
            z, p, k = sos2zpk(sos)
        # 断言零点的数量是否为预期值
        assert len(z) == 24
        # 断言极点的数量是否为预期值
        assert len(p) == 24


class TestSos2Tf:

    def test_basic(self):
        # 定义一个二阶系统的状态空间表示
        sos = [[1, 1, 1, 1, 0, -1],
               [-2, 3, 1, 1, 10, 1]]
        # 将状态空间表示转换为传递函数表示的分子和分母系数
        b, a = sos2tf(sos)
        # 检查计算得到的分子系数与预期是否接近
        assert_array_almost_equal(b, [-2, 1, 2, 4, 1])
        # 检查计算得到的分母系数与预期是否接近
        assert_array_almost_equal(a, [1, 10, 0, -10, -1])


class TestTf2Sos:

    def test_basic(self):
        # 定义一个传递函数的分子和分母系数
        num = [2, 16, 44, 56, 32]
        den = [3, 3, -15, 18, -12]
        # 将传递函数表示转换为状态空间表示（SOS形式）
        sos = tf2sos(num, den)
        # 预期的SOS表示
        sos2 = [[0.6667, 4.0000, 5.3333, 1.0000, +2.0000, -4.0000],
                [1.0000, 2.0000, 2.0000, 1.0000, -1.0000, +1.0000]]
        # 检查计算得到的SOS表示与预期是否接近
        assert_array_almost_equal(sos, sos2, decimal=4)

        # 另一个传递函数的分子和分母系数
        b = [1, -3, 11, -27, 18]
        a = [16, 12, 2, -4, -1]
        # 将传递函数表示转换为状态空间表示（SOS形式）
        sos = tf2sos(b, a)
        # 预期的SOS表示
        sos2 = [[0.0625, -0.1875, 0.1250, 1.0000, -0.2500, -0.1250],
                [1.0000, +0.0000, 9.0000, 1.0000, +1.0000, +0.5000]]
        # assert_array_almost_equal(sos, sos2, decimal=4)
    # 使用 pytest.mark.parametrize 装饰器标记测试参数化，定义多组参数进行测试
    @pytest.mark.parametrize('b, a, analog, sos',
                             [([1], [1], False, [[1., 0., 0., 1., 0., 0.]]),  # 第一组参数：b=[1], a=[1], analog=False, sos=[[1., 0., 0., 1., 0., 0.]]
                              ([1], [1], True, [[0., 0., 1., 0., 0., 1.]]),   # 第二组参数：b=[1], a=[1], analog=True, sos=[[0., 0., 1., 0., 0., 1.]]
                              ([1], [1., 0., -1.01, 0, 0.01], False,          # 第三组参数：b=[1], a=[1., 0., -1.01, 0, 0.01], analog=False,
                               [[1., 0., 0., 1., 0., -0.01],                    # sos=[[1., 0., 0., 1., 0., -0.01],
                                [1., 0., 0., 1., 0., -1]]),                     #      [1., 0., 0., 1., 0., -1]]
                              ([1], [1., 0., -1.01, 0, 0.01], True,             # 第四组参数：b=[1], a=[1., 0., -1.01, 0, 0.01], analog=True,
                               [[0., 0., 1., 1., 0., -1],                        # sos=[[0., 0., 1., 1., 0., -1],
                                [0., 0., 1., 1., 0., -0.01]])])                 #      [0., 0., 1., 1., 0., -0.01]])
    # 定义测试方法 test_analog，参数为 b, a, analog, sos
    def test_analog(self, b, a, analog, sos):
        # 调用 tf2sos 函数，传入参数 b, a, analog=analog，获取返回值 sos2
        sos2 = tf2sos(b, a, analog=analog)
        # 使用 assert_array_almost_equal 断言，比较 sos 和 sos2 的近似相等性，精确度为小数点后 4 位
        assert_array_almost_equal(sos, sos2, decimal=4)
# 定义一个测试类 TestZpk2Sos，用于测试 zpk2sos 函数的输出是否符合预期
class TestZpk2Sos:

    # 使用 pytest.mark.parametrize 装饰器，指定参数 dt 为 'fdgFDG' 的每个字符
    # 使用 pytest.mark.parametrize 装饰器，指定参数 pairing 和 analog 的组合
    @pytest.mark.parametrize('dt', 'fdgFDG')
    @pytest.mark.parametrize('pairing, analog',
                             [('nearest', False),  # pairing 为 'nearest'，analog 为 False
                              ('keep_odd', False),  # pairing 为 'keep_odd'，analog 为 False
                              ('minimal', False),   # pairing 为 'minimal'，analog 为 False
                              ('minimal', True)])   # pairing 为 'minimal'，analog 为 True
    def test_dtypes(self, dt, pairing, analog):
        # 创建一个长度为 2 的数组 z，元素为 [-1, -1]，并将其类型转换为 dt 指定的类型
        z = np.array([-1, -1]).astype(dt)
        # 将 dt 转换为大写，用于指定极点必须是复数
        ct = dt.upper()
        # 创建一个长度为 2 的数组 p，元素为 [0.57149 + 0.29360j, 0.57149 - 0.29360j]，并将其类型转换为 ct 指定的类型
        p = np.array([0.57149 + 0.29360j, 0.57149 - 0.29360j]).astype(ct)
        # 创建一个包含一个元素的数组 k，元素为 1，并将其类型转换为 dt 指定的类型
        k = np.array(1).astype(dt)
        # 调用 zpk2sos 函数，生成二阶节流的描述符，将结果赋给变量 sos
        sos = zpk2sos(z, p, k, pairing=pairing, analog=analog)
        # 创建一个期望的二阶节流的描述符 sos2，其内容与 Octave 和 MATLAB 的输出相符
        sos2 = [[1, 2, 1, 1, -1.14298, 0.41280]]
        # 断言 sos 与 sos2 数组近似相等，精确度为小数点后四位
        assert_array_almost_equal(sos, sos2, decimal=4)

    # 使用 pytest.mark.parametrize 装饰器，指定参数 pairing 和 sos 的组合
    # 每个组合对应不同的二阶节流的描述符 sos
    @pytest.mark.parametrize('pairing, sos',
                             [('nearest', np.array([[1., 1., 0.5, 1., -0.75, 0.],
                                                    [1., 1., 0., 1., -1.6, 0.65]])),  # pairing 为 'nearest'，对应 sos
                              ('keep_odd', np.array([[1., 1., 0, 1., -0.75, 0.],
                                                     [1., 1., 0.5, 1., -1.6, 0.65]])),  # pairing 为 'keep_odd'，对应 sos
                              ('minimal', np.array([[0., 1., 1., 0., 1., -0.75],
                                                    [1., 1., 0.5, 1., -1.6, 0.65]]))])  # pairing 为 'minimal'，对应 sos
    def test_pairing(self, pairing, sos):
        # 创建一个长度为 3 的数组 z1，元素为 [-1, -0.5-0.5j, -0.5+0.5j]
        z1 = np.array([-1, -0.5-0.5j, -0.5+0.5j])
        # 创建一个长度为 3 的数组 p1，元素为 [0.75, 0.8+0.1j, 0.8-0.1j]
        p1 = np.array([0.75, 0.8+0.1j, 0.8-0.1j])
        # 调用 zpk2sos 函数，生成二阶节流的描述符 sos2，使用参数 pairing 和默认值的 analog
        sos2 = zpk2sos(z1, p1, 1, pairing=pairing)
        # 断言 sos2 与预期的 sos 数组近似相等，精确度为小数点后四位
        assert_array_almost_equal(sos, sos2, decimal=4)

    # 使用 pytest.mark.parametrize 装饰器，指定参数 p 和 sos_dt 的组合
    # 每个组合对应不同的二阶节流的描述符 sos_dt
    @pytest.mark.parametrize('p, sos_dt',
                             [([-1, 1, -0.1, 0.1],
                               [[0., 0., 1., 1., 0., -0.01],
                                [0., 0., 1., 1., 0., -1]]),  # p 为 [-1, 1, -0.1, 0.1]，对应 sos_dt
                              ([-0.7071+0.7071j, -0.7071-0.7071j, -0.1j, 0.1j],
                               [[0., 0., 1., 1., 0., 0.01],
                                [0., 0., 1., 1., 1.4142, 1.]])])  # p 为 [-0.7071+0.7071j, -0.7071-0.7071j, -0.1j, 0.1j]，对应 sos_dt
    def test_analog(self, p, sos_dt):
        # 调用 zpk2sos 函数，生成二阶节流的描述符 sos2_dt，使用参数 'minimal' 和 analog=False
        sos2_dt = zpk2sos([], p, 1, pairing='minimal', analog=False)
        # 调用 zpk2sos 函数，生成二阶节流的描述符 sos2_ct，使用参数 'minimal' 和 analog=True
        sos2_ct = zpk2sos([], p, 1, pairing='minimal', analog=True)
        # 断言 sos_dt 与 sos2_dt 数组近似相等，精确度为小数点后四位
        assert_array_almost_equal(sos_dt, sos2_dt, decimal=4)
        # 断言 sos_dt 的逆序与 sos2_ct 数组近似相等，精确度为小数点后四位
        assert_array_almost_equal(sos_dt[::-1], sos2_ct, decimal=4)
    # 定义测试函数 test_bad_args(self)，用于测试 zpk2sos 函数在接收到不合法参数时是否能正确抛出异常

        # 使用 pytest 的上下文管理器，检查调用 zpk2sos 函数时是否抛出 ValueError 异常，并匹配错误消息中包含 'pairing must be one of' 的字符串
        with pytest.raises(ValueError, match=r'pairing must be one of'):
            zpk2sos([1], [2], 1, pairing='no_such_pairing')

        # 使用 pytest 的上下文管理器，检查调用 zpk2sos 函数时是否抛出 ValueError 异常，并匹配错误消息中包含 'pairing must be "minimal"' 的字符串
        with pytest.raises(ValueError, match=r'.*pairing must be "minimal"'):
            zpk2sos([1], [2], 1, pairing='keep_odd', analog=True)

        # 使用 pytest 的上下文管理器，检查调用 zpk2sos 函数时是否抛出 ValueError 异常，并匹配错误消息中包含 'must have len(p)>=len(z)' 的字符串
        with pytest.raises(ValueError,
                           match=r'.*must have len\(p\)>=len\(z\)'):
            zpk2sos([1, 1], [2], 1, analog=True)

        # 使用 pytest 的上下文管理器，检查调用 zpk2sos 函数时是否抛出 ValueError 异常，并匹配错误消息中包含 'k must be real' 的字符串
        with pytest.raises(ValueError, match=r'k must be real'):
            zpk2sos([1], [2], k=1j)
class TestFreqs:

    def test_basic(self):
        _, h = freqs([1.0], [1.0], worN=8)
        assert_array_almost_equal(h, np.ones(8))

    def test_output(self):
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # 定义角频率数组
        w = [0.1, 1, 10, 100]
        num = [1]
        den = [1, 1]
        # 计算频率响应
        w, H = freqs(num, den, worN=w)
        # 计算复数角频率
        s = w * 1j
        expected = 1 / (s + 1)
        # 检查实部和虚部是否接近预期值
        assert_array_almost_equal(H.real, expected.real)
        assert_array_almost_equal(H.imag, expected.imag)

    def test_freq_range(self):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # 预期的频率范围是从 0.01 到 10
        num = [1]
        den = [1, 1]
        n = 10
        expected_w = np.logspace(-2, 1, n)
        # 计算频率响应
        w, H = freqs(num, den, worN=n)
        # 检查计算得到的频率数组是否与预期的频率范围接近
        assert_array_almost_equal(w, expected_w)

    def test_plot(self):

        def plot(w, h):
            assert_array_almost_equal(h, np.ones(8))

        # 检查是否能捕获到除零错误
        assert_raises(ZeroDivisionError, freqs, [1.0], [1.0], worN=8,
                      plot=lambda w, h: 1 / 0)
        # 调用频率响应计算函数，并使用自定义绘图函数进行检查
        freqs([1.0], [1.0], worN=8, plot=plot)

    def test_backward_compat(self):
        # For backward compatibility, test if None act as a wrapper for default
        # 测试 None 是否作为默认值的包装器以确保向后兼容性
        w1, h1 = freqs([1.0], [1.0])
        w2, h2 = freqs([1.0], [1.0], None)
        # 检查两次调用返回的频率数组和响应是否几乎相等
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(h1, h2)

    def test_w_or_N_types(self):
        # Measure at 8 equally-spaced points
        # 在等间距点测量频率响应
        for N in (8, np.int8(8), np.int16(8), np.int32(8), np.int64(8),
                  np.array(8)):
            # 计算频率响应
            w, h = freqs([1.0], [1.0], worN=N)
            # 检查频率数组长度是否为8，以及频率响应是否为全1数组
            assert_equal(len(w), 8)
            assert_array_almost_equal(h, np.ones(8))

        # Measure at frequency 8 rad/sec
        # 在频率为8 rad/s处测量频率响应
        for w in (8.0, 8.0+0j):
            # 计算频率响应
            w_out, h = freqs([1.0], [1.0], worN=w)
            # 检查输出的频率值和频率响应是否与预期接近
            assert_array_almost_equal(w_out, [8])
            assert_array_almost_equal(h, [1])


class TestFreqs_zpk:

    def test_basic(self):
        _, h = freqs_zpk([1.0], [1.0], [1.0], worN=8)
        assert_array_almost_equal(h, np.ones(8))

    def test_output(self):
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # 定义角频率数组
        w = [0.1, 1, 10, 100]
        z = []
        p = [-1]
        k = 1
        # 计算频率响应
        w, H = freqs_zpk(z, p, k, worN=w)
        # 计算复数角频率
        s = w * 1j
        expected = 1 / (s + 1)
        # 检查实部和虚部是否接近预期值
        assert_array_almost_equal(H.real, expected.real)
        assert_array_almost_equal(H.imag, expected.imag)

    def test_freq_range(self):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # 预期的频率范围是从 0.01 到 10
        z = []
        p = [-1]
        k = 1
        n = 10
        expected_w = np.logspace(-2, 1, n)
        # 计算频率响应
        w, H = freqs_zpk(z, p, k, worN=n)
        # 检查计算得到的频率数组是否与预期的频率范围接近
        assert_array_almost_equal(w, expected_w)
    # 定义测试函数，用于验证频率响应函数的正确性
    def test_vs_freqs(self):
        # 使用 Chebyshev Type I 滤波器设计函数生成分子和分母系数
        b, a = cheby1(4, 5, 100, analog=True, output='ba')
        # 使用 Chebyshev Type I 滤波器设计函数生成零点、极点和增益
        z, p, k = cheby1(4, 5, 100, analog=True, output='zpk')

        # 计算频率响应，返回频率和幅度响应
        w1, h1 = freqs(b, a)
        # 计算频率响应，给定零点、极点和增益，返回频率和幅度响应
        w2, h2 = freqs_zpk(z, p, k)
        
        # 断言两组频率响应的频率部分近似相等
        assert_allclose(w1, w2)
        # 断言两组频率响应的幅度部分近似相等，允许的相对误差为 1e-6
        assert_allclose(h1, h2, rtol=1e-6)

    # 定义测试函数，用于验证向后兼容性
    def test_backward_compat(self):
        # 用于向后兼容性测试，检查 None 是否作为默认值的包装器
        w1, h1 = freqs_zpk([1.0], [1.0], [1.0])
        w2, h2 = freqs_zpk([1.0], [1.0], [1.0], None)
        
        # 断言两组频率响应的频率部分几乎相等
        assert_array_almost_equal(w1, w2)
        # 断言两组频率响应的幅度部分几乎相等
        assert_array_almost_equal(h1, h2)

    # 定义测试函数，用于验证 worN 参数的类型
    def test_w_or_N_types(self):
        # 在 8 个等间距点上测量频率响应
        for N in (8, np.int8(8), np.int16(8), np.int32(8), np.int64(8),
                  np.array(8)):
            # 使用给定的 N 值计算频率响应，返回频率和幅度响应
            w, h = freqs_zpk([], [], 1, worN=N)
            # 断言返回的频率数组长度为 8
            assert_equal(len(w), 8)
            # 断言幅度响应数组近似为全 1
            assert_array_almost_equal(h, np.ones(8))

        # 在频率为 8 rad/sec 处测量频率响应
        for w in (8.0, 8.0+0j):
            # 使用给定的 w 值计算频率响应，返回频率和幅度响应
            w_out, h = freqs_zpk([], [], 1, worN=w)
            # 断言返回的频率数组近似为 [8]
            assert_array_almost_equal(w_out, [8])
            # 断言幅度响应数组近似为 [1]
            assert_array_almost_equal(h, [1])
class TestFreqz:

    def test_ticket1441(self):
        """Regression test for ticket 1441."""
        # 设置测试参数 N 为 100000
        N = 100000
        # 调用 freqz 函数，返回频率和响应
        w, h = freqz([1.0], worN=N)
        # 断言频率数组 w 的形状为 (N,)
        assert_equal(w.shape, (N,))

    def test_basic(self):
        # 调用 freqz 函数，worN 设置为 8，返回频率和响应
        w, h = freqz([1.0], worN=8)
        # 断言频率数组 w 的值近似于 np.pi * np.arange(8) / 8.
        assert_array_almost_equal(w, np.pi * np.arange(8) / 8.)
        # 断言响应数组 h 的值近似于 np.ones(8)
        assert_array_almost_equal(h, np.ones(8))
        
        # 调用 freqz 函数，worN 设置为 9，返回频率和响应
        w, h = freqz([1.0], worN=9)
        # 断言频率数组 w 的值近似于 np.pi * np.arange(9) / 9.
        assert_array_almost_equal(w, np.pi * np.arange(9) / 9.)
        # 断言响应数组 h 的值近似于 np.ones(9)
        assert_array_almost_equal(h, np.ones(9))

        # 迭代测试用例中的不同参数
        for a in [1, np.ones(2)]:
            # 调用 freqz 函数，worN 设置为 0，返回频率和响应
            w, h = freqz(np.ones(2), a, worN=0)
            # 断言频率数组 w 的形状为 (0,)
            assert_equal(w.shape, (0,))
            # 断言响应数组 h 的形状为 (0,)
            assert_equal(h.shape, (0,))
            # 断言响应数组 h 的数据类型为 complex128
            assert_equal(h.dtype, np.dtype('complex128'))

        # 生成时间向量 t
        t = np.linspace(0, 1, 4, endpoint=False)
        # 迭代测试用例中的不同参数组合
        for b, a, h_whole in zip(
                ([1., 0, 0, 0], np.sin(2 * np.pi * t)),
                ([1., 0, 0, 0], [0.5, 0, 0, 0]),
                ([1., 1., 1., 1.], [0, -4j, 0, 4j])):
            # 调用 freqz 函数，worN 设置为 4，返回频率和响应
            w, h = freqz(b, a, worN=4, whole=True)
            # 期望的频率数组 w
            expected_w = np.linspace(0, 2 * np.pi, 4, endpoint=False)
            # 断言频率数组 w 的值近似于 expected_w
            assert_array_almost_equal(w, expected_w)
            # 断言响应数组 h 的值近似于预期的 h_whole
            assert_array_almost_equal(h, h_whole)
            # 同时检查整数类型支持
            # 调用 freqz 函数，worN 设置为 np.int32(4)，返回频率和响应
            w, h = freqz(b, a, worN=np.int32(4), whole=True)
            # 断言频率数组 w 的值近似于 expected_w
            assert_array_almost_equal(w, expected_w)
            # 断言响应数组 h 的值近似于预期的 h_whole
            assert_array_almost_equal(h, h_whole)
            # 调用 freqz 函数，worN 设置为 w，返回频率和响应
            w, h = freqz(b, a, worN=w, whole=True)
            # 断言频率数组 w 的值近似于 expected_w
            assert_array_almost_equal(w, expected_w)
            # 断言响应数组 h 的值近似于预期的 h_whole
            assert_array_almost_equal(h, h_whole)

    def test_basic_whole(self):
        # 调用 freqz 函数，worN 设置为 8，whole 设置为 True，返回频率和响应
        w, h = freqz([1.0], worN=8, whole=True)
        # 断言频率数组 w 的值近似于 2 * np.pi * np.arange(8.0) / 8
        assert_array_almost_equal(w, 2 * np.pi * np.arange(8.0) / 8)
        # 断言响应数组 h 的值近似于 np.ones(8)
        assert_array_almost_equal(h, np.ones(8))

    def test_plot(self):

        def plot(w, h):
            # 断言频率数组 w 的值近似于 np.pi * np.arange(8.0) / 8
            assert_array_almost_equal(w, np.pi * np.arange(8.0) / 8)
            # 断言响应数组 h 的值近似于 np.ones(8)
            assert_array_almost_equal(h, np.ones(8))

        # 断言会引发 ZeroDivisionError 异常，因为 plot 函数尝试执行 1 / 0
        assert_raises(ZeroDivisionError, freqz, [1.0], worN=8,
                      plot=lambda w, h: 1 / 0)
        # 调用 freqz 函数，worN 设置为 8，调用 plot 函数进行断言检查
        freqz([1.0], worN=8, plot=plot)
    def test_broadcasting1(self):
        # 测试广播功能，其中 worN 可以是整数或1维数组，
        # b 和 a 是n维数组。
        np.random.seed(123)
        # 创建一个3x5x1的随机数组 b
        b = np.random.rand(3, 5, 1)
        # 创建一个2x1的随机数组 a
        a = np.random.rand(2, 1)
        for whole in [False, True]:
            # 测试 worN 分别为整数（用于 FFT 的快速值和非快速值）、
            # 1维数组和空数组的情况。
            for worN in [16, 17, np.linspace(0, 1, 10), np.array([])]:
                # 调用 freqz 函数计算频率响应 w 和 h
                w, h = freqz(b, a, worN=worN, whole=whole)
                for k in range(b.shape[1]):
                    # 选择 b 的第 k 列第 0 行作为 bk
                    bk = b[:, k, 0]
                    # 选择 a 的第 0 列作为 ak
                    ak = a[:, 0]
                    # 再次调用 freqz 函数计算频率响应 ww 和 hh
                    ww, hh = freqz(bk, ak, worN=worN, whole=whole)
                    # 断言 ww 和 w 的值近似相等
                    assert_allclose(ww, w)
                    # 断言 hh 和 h[k] 的值近似相等
                    assert_allclose(hh, h[k])

    def test_broadcasting2(self):
        # 测试广播功能，其中 worN 可以是整数或1维数组，
        # b 是n维数组，而 a 使用默认值。
        np.random.seed(123)
        # 创建一个3x5x1的随机数组 b
        b = np.random.rand(3, 5, 1)
        for whole in [False, True]:
            for worN in [16, 17, np.linspace(0, 1, 10)]:
                # 调用 freqz 函数计算频率响应 w 和 h
                w, h = freqz(b, worN=worN, whole=whole)
                for k in range(b.shape[1]):
                    # 选择 b 的第 k 列第 0 行作为 bk
                    bk = b[:, k, 0]
                    # 再次调用 freqz 函数计算频率响应 ww 和 hh
                    ww, hh = freqz(bk, worN=worN, whole=whole)
                    # 断言 ww 和 w 的值近似相等
                    assert_allclose(ww, w)
                    # 断言 hh 和 h[k] 的值近似相等
                    assert_allclose(hh, h[k])

    def test_broadcasting3(self):
        # 测试广播功能，其中 b.shape[-1] 与 worN 的长度相同，
        # 而 a 使用默认值。
        np.random.seed(123)
        N = 16
        # 创建一个3xN的随机数组 b
        b = np.random.rand(3, N)
        for whole in [False, True]:
            for worN in [N, np.linspace(0, 1, N)]:
                # 调用 freqz 函数计算频率响应 w 和 h
                w, h = freqz(b, worN=worN, whole=whole)
                # 断言 w 的长度等于 N
                assert_equal(w.size, N)
                for k in range(N):
                    # 选择 b 的第 k 列作为 bk
                    bk = b[:, k]
                    # 再次调用 freqz 函数计算频率响应 ww 和 hh
                    ww, hh = freqz(bk, worN=w[k], whole=whole)
                    # 断言 ww 和 w[k] 的值近似相等
                    assert_allclose(ww, w[k])
                    # 断言 hh 和 h[k] 的值近似相等
                    assert_allclose(hh, h[k])

    def test_broadcasting4(self):
        # 测试广播功能，其中 worN 是一个2维数组。
        np.random.seed(123)
        # 创建一个4x2x1x1的随机数组 b
        b = np.random.rand(4, 2, 1, 1)
        # 创建一个5x2x1x1的随机数组 a
        a = np.random.rand(5, 2, 1, 1)
        for whole in [False, True]:
            for worN in [np.random.rand(6, 7), np.empty((6, 0))]:
                # 调用 freqz 函数计算频率响应 w 和 h
                w, h = freqz(b, a, worN=worN, whole=whole)
                # 断言 w 和 worN 的值近似相等
                assert_allclose(w, worN, rtol=1e-14)
                # 断言 h 的形状为 (2, ) + worN 的形状
                assert_equal(h.shape, (2,) + worN.shape)
                for k in range(2):
                    # 选择 b 的第 k 列第 0 行第 0 列第 0 行作为输入 bk 和 ak
                    ww, hh = freqz(b[:, k, 0, 0], a[:, k, 0, 0],
                                   worN=worN.ravel(),
                                   whole=whole)
                    # 断言 ww 和 worN.ravel() 的值近似相等
                    assert_allclose(ww, worN.ravel(), rtol=1e-14)
                    # 断言 hh 和 h[k, :, :] 的展平值近似相等
                    assert_allclose(hh, h[k, :, :].ravel())
    # 定义测试函数，用于测试向后兼容性，检查 None 是否作为默认参数的包装
    def test_backward_compat(self):
        # 调用 freqz 函数，计算频率响应 w1, h1，使用默认参数 fs=1
        w1, h1 = freqz([1.0], 1)
        # 再次调用 freqz 函数，计算频率响应 w2, h2，明确指定 fs=None
        w2, h2 = freqz([1.0], 1, None)
        # 断言 w1 与 w2 几乎相等
        assert_array_almost_equal(w1, w2)
        # 断言 h1 与 h2 几乎相等
        assert_array_almost_equal(h1, h2)

    # 定义测试函数，用于测试参数 fs 的不同设置
    def test_fs_param(self):
        # 设置采样频率 fs
        fs = 900
        # 设置传递函数的分子系数 b
        b = [0.039479155677484369, 0.11843746703245311, 0.11843746703245311,
             0.039479155677484369]
        # 设置传递函数的分母系数 a
        a = [1.0, -1.3199152021838287, 0.80341991081938424,
             -0.16767146321568049]

        # 测试 N=None, whole=False 的情况
        w1, h1 = freqz(b, a, fs=fs)
        w2, h2 = freqz(b, a)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs/2, 512, endpoint=False))

        # 测试 N=None, whole=True 的情况
        w1, h1 = freqz(b, a, whole=True, fs=fs)
        w2, h2 = freqz(b, a, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 512, endpoint=False))

        # 测试 N=5, whole=False 的情况
        w1, h1 = freqz(b, a, 5, fs=fs)
        w2, h2 = freqz(b, a, 5)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs/2, 5, endpoint=False))

        # 测试 N=5, whole=True 的情况
        w1, h1 = freqz(b, a, 5, whole=True, fs=fs)
        w2, h2 = freqz(b, a, 5, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 5, endpoint=False))

        # 测试 w 是数组的情况
        for w in ([123], (123,), np.array([123]), (50, 123, 230),
                  np.array([50, 123, 230])):
            w1, h1 = freqz(b, a, w, fs=fs)
            w2, h2 = freqz(b, a, 2*pi*np.array(w)/fs)
            assert_allclose(h1, h2)
            assert_allclose(w, w1)

    # 定义测试函数，测试参数 worN 的不同类型
    def test_w_or_N_types(self):
        # 分别测试在 7 (polyval) 或 8 (fft) 等间隔点进行测量
        for N in (7, np.int8(7), np.int16(7), np.int32(7), np.int64(7),
                  np.array(7),
                  8, np.int8(8), np.int16(8), np.int32(8), np.int64(8),
                  np.array(8)):
            # 调用 freqz 函数，测量频率响应 w, h
            w, h = freqz([1.0], worN=N)
            # 断言 w 与期望的频率值几乎相等
            assert_array_almost_equal(w, np.pi * np.arange(N) / N)
            # 断言 h 为全为 1 的数组
            assert_array_almost_equal(h, np.ones(N))

            # 调用 freqz 函数，测量频率响应 w, h，同时指定 fs=100
            w, h = freqz([1.0], worN=N, fs=100)
            # 断言 w 与期望的频率范围几乎相等
            assert_array_almost_equal(w, np.linspace(0, 50, N, endpoint=False))
            # 断言 h 为全为 1 的数组
            assert_array_almost_equal(h, np.ones(N))

        # 测量频率为 8 Hz 的情况
        for w in (8.0, 8.0+0j):
            # 只有在指定 fs 的情况下才有意义
            w_out, h = freqz([1.0], worN=w, fs=100)
            # 断言测量的频率 w_out 与期望的频率值 [8] 几乎相等
            assert_array_almost_equal(w_out, [8])
            # 断言 h 为全为 1 的数组
            assert_array_almost_equal(h, [1])
    #`
    # 定义一个测试方法，用于测试频率响应函数的包含奈奎斯特频率的功能
    def test_nyquist(self):
        # 调用 freqz 函数计算频率响应
        w, h = freqz([1.0], worN=8, include_nyquist=True)
        # 断言频率数组 w 的近似值是否与预期相符
        assert_array_almost_equal(w, np.pi * np.arange(8) / 7.)
        # 断言频率响应数组 h 的近似值是否为全 1 数组
        assert_array_almost_equal(h, np.ones(8))
        
        # 再次调用 freqz 函数计算不同参数下的频率响应
        w, h = freqz([1.0], worN=9, include_nyquist=True)
        assert_array_almost_equal(w, np.pi * np.arange(9) / 8.)
        assert_array_almost_equal(h, np.ones(9))

        # 针对不同的输入 a 进行频率响应计算
        for a in [1, np.ones(2)]:
            w, h = freqz(np.ones(2), a, worN=0, include_nyquist=True)
            # 断言返回的空频率数组形状是否为 (0,)
            assert_equal(w.shape, (0,))
            # 断言返回的空频率响应数组形状是否为 (0,)
            assert_equal(h.shape, (0,))
            # 断言返回的频率响应数组的数据类型是否为 complex128
            assert_equal(h.dtype, np.dtype('complex128'))

        # 调用 freqz 函数计算整个频率范围内的频率响应
        w1, h1 = freqz([1.0], worN=8, whole=True, include_nyquist=True)
        w2, h2 = freqz([1.0], worN=8, whole=True, include_nyquist=False)
        # 断言两种计算方式得到的频率数组是否近似相等
        assert_array_almost_equal(w1, w2)
        # 断言两种计算方式得到的频率响应数组是否近似相等
        assert_array_almost_equal(h1, h2)

    # 标记测试用例与 GitHub 上的特定问题相关联
    # https://github.com/scipy/scipy/issues/17289
    # https://github.com/scipy/scipy/issues/15273
    @pytest.mark.parametrize('whole,nyquist,worN',
                             [(False, False, 32),
                              (False, True, 32),
                              (True, False, 32),
                              (True, True, 32),
                              (False, False, 257),
                              (False, True, 257),
                              (True, False, 257),
                              (True, True, 257)])
    # 定义一个测试方法，用于验证与特定问题相关的频率响应计算
    def test_17289(self, whole, nyquist, worN):
        # 定义一个简单的信号
        d = [0, 1]
        # 调用 freqz 函数计算不同参数下的频率响应
        w, Drfft = freqz(d, worN=32, whole=whole, include_nyquist=nyquist)
        # 调用 freqz 函数计算相同参数下的频率响应
        _, Dpoly = freqz(d, worN=w)
        # 断言两种计算方式得到的频率响应数组是否非常接近
        assert_allclose(Drfft, Dpoly)

    # 定义一个测试方法，用于验证频率响应函数对采样率参数的验证
    def test_fs_validation(self):
        # 使用 pytest 断言抛出 ValueError 异常，匹配指定的错误消息
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            freqz([1.0], fs=np.array([10, 20]))

        # 使用 pytest 断言抛出 ValueError 异常，匹配指定的错误消息
        with pytest.raises(ValueError, match="Sampling.*be none."):
            freqz([1.0], fs=None)
class TestSOSFreqz:

    def test_sosfreqz_basic(self):
        # 比较 freqz 和 sosfreqz 对低阶 Butterworth 滤波器的结果

        N = 500  # 设置频率响应的采样点数

        # 使用 butter 函数生成 4 阶低通 Butterworth 滤波器的系数
        b, a = butter(4, 0.2)
        # 使用 butter 函数生成 4 阶低通 Butterworth 滤波器的 second-order section (SOS) 表示
        sos = butter(4, 0.2, output='sos')
        # 计算频率响应 w 和 h
        w, h = freqz(b, a, worN=N)
        # 计算 SOS 表示的频率响应 w2 和 h2
        w2, h2 = sosfreqz(sos, worN=N)
        # 断言频率响应的频率数组 w2 应该与 w 相等
        assert_equal(w2, w)
        # 断言频率响应的幅度响应 h2 应该与 h 在给定的相对误差和绝对误差范围内非常接近
        assert_allclose(h2, h, rtol=1e-10, atol=1e-14)

        # 使用 ellip 函数生成 3 阶带通 Elliptic 滤波器的系数
        b, a = ellip(3, 1, 30, (0.2, 0.3), btype='bandpass')
        # 使用 ellip 函数生成 3 阶带通 Elliptic 滤波器的 SOS 表示
        sos = ellip(3, 1, 30, (0.2, 0.3), btype='bandpass', output='sos')
        # 计算频率响应 w 和 h
        w, h = freqz(b, a, worN=N)
        # 计算 SOS 表示的频率响应 w2 和 h2
        w2, h2 = sosfreqz(sos, worN=N)
        # 断言频率响应的频率数组 w2 应该与 w 相等
        assert_equal(w2, w)
        # 断言频率响应的幅度响应 h2 应该与 h 在给定的相对误差和绝对误差范围内非常接近
        assert_allclose(h2, h, rtol=1e-10, atol=1e-14)
        
        # 断言：SOS 表示必须至少包含一个 section，否则应该引发 ValueError 异常
        assert_raises(ValueError, sosfreqz, sos[:0])
    def test_sosfrez_design(self):
        # 对不同滤波器类型的 sosfreqz 输出与预期值进行比较测试

        # 使用 cheb2ord 函数计算 Chebyshev Type II 滤波器的阶数 N 和归一化截止频率 Wn
        N, Wn = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        # 使用 cheby2 函数生成 Chebyshev Type II 滤波器的 second-order section (SOS) 表示
        sos = cheby2(N, 60, Wn, 'stop', output='sos')
        # 使用 sosfreqz 函数计算频率响应 w 和幅度响应 h
        w, h = sosfreqz(sos)
        # 取幅度响应 h 的绝对值
        h = np.abs(h)
        # 将频率响应 w 归一化到 π
        w /= np.pi
        # 断言频率范围在 0.1 以下的幅度响应为 0 dB，容许误差为 3.01 dB
        assert_allclose(20 * np.log10(h[w <= 0.1]), 0, atol=3.01)
        # 断言频率范围在 0.6 以上的幅度响应为 0 dB，容许误差为 3.01 dB
        assert_allclose(20 * np.log10(h[w >= 0.6]), 0., atol=3.01)
        # 断言频率范围在 0.2 到 0.5 之间的幅度响应为 0，容许误差为 1e-3，相当于 -60 dB
        assert_allclose(h[(w >= 0.2) & (w <= 0.5)], 0., atol=1e-3)

        # 使用 cheb2ord 函数计算更高通阻滤波器的阶数 N 和归一化截止频率 Wn
        N, Wn = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 150)
        # 使用 cheby2 函数生成更高通阻滤波器的 second-order section (SOS) 表示
        sos = cheby2(N, 150, Wn, 'stop', output='sos')
        # 使用 sosfreqz 函数计算频率响应 w 和幅度响应 h
        w, h = sosfreqz(sos)
        # 计算幅度响应 h 的分贝值
        dB = 20*np.log10(np.abs(h))
        # 将频率响应 w 归一化到 π
        w /= np.pi
        # 断言频率范围在 0.1 以下的分贝响应为 0，容许误差为 3.01 dB
        assert_allclose(dB[w <= 0.1], 0, atol=3.01)
        # 断言频率范围在 0.6 以上的分贝响应为 0，容许误差为 3.01 dB
        assert_allclose(dB[w >= 0.6], 0., atol=3.01)
        # 断言频率范围在 0.2 到 0.5 之间的分贝响应小于 -149.9 dB
        assert_array_less(dB[(w >= 0.2) & (w <= 0.5)], -149.9)

        # 使用 cheb1ord 函数计算 Chebyshev Type I 滤波器的阶数 N 和归一化截止频率 Wn
        N, Wn = cheb1ord(0.2, 0.3, 3, 40)
        # 使用 cheby1 函数生成 Chebyshev Type I 滤波器的 second-order section (SOS) 表示
        sos = cheby1(N, 3, Wn, 'low', output='sos')
        # 使用 sosfreqz 函数计算频率响应 w 和幅度响应 h
        w, h = sosfreqz(sos)
        # 取幅度响应 h 的绝对值
        h = np.abs(h)
        # 将频率响应 w 归一化到 π
        w /= np.pi
        # 断言频率范围在 0.2 以下的幅度响应为 0 dB，容许误差为 3.01 dB
        assert_allclose(20 * np.log10(h[w <= 0.2]), 0, atol=3.01)
        # 断言频率范围在 0.3 以上的幅度响应为 0，容许误差为 1e-2，相当于 -40 dB
        assert_allclose(h[w >= 0.3], 0., atol=1e-2)

        # 使用 cheb1ord 函数计算更高通阻滤波器的阶数 N 和归一化截止频率 Wn
        N, Wn = cheb1ord(0.2, 0.3, 1, 150)
        # 使用 cheby1 函数生成更高通阻滤波器的 second-order section (SOS) 表示
        sos = cheby1(N, 1, Wn, 'low', output='sos')
        # 使用 sosfreqz 函数计算频率响应 w 和幅度响应 h
        w, h = sosfreqz(sos)
        # 计算幅度响应 h 的分贝值
        dB = 20*np.log10(np.abs(h))
        # 将频率响应 w 归一化到 π
        w /= np.pi
        # 断言频率范围在 0.2 以下的分贝响应为 0，容许误差为 1.01 dB
        assert_allclose(dB[w <= 0.2], 0, atol=1.01)
        # 断言频率范围在 0.3 以上的分贝响应小于 -149.9 dB
        assert_array_less(dB[w >= 0.3], -149.9)

        # 使用 ellipord 函数计算 Elliptic 滤波器的阶数 N 和归一化截止频率 Wn
        N, Wn = ellipord(0.3, 0.2, 3, 60)
        # 使用 ellip 函数生成 Elliptic 滤波器的 second-order section (SOS) 表示
        sos = ellip(N, 0.3, 60, Wn, 'high', output='sos')
        # 使用 sosfreqz 函数计算频率响应 w 和幅度响应 h
        w, h = sosfreqz(sos)
        # 取幅度响应 h 的绝对值
        h = np.abs(h)
        # 将频率响应 w 归一化到 π
        w /= np.pi
        # 断言频率范围在 0.3 以上的幅度响应为 0 dB，容许误差为 3.01 dB
        assert_allclose(20 * np.log10(h[w >= 0.3]), 0, atol=3.01)
        # 断言频率范围在 0.1 以下的幅度响应为 0，容许误差为 1.5e-3，相当于 -60 dB（近似）
        assert_allclose(h[w <= 0.1], 0., atol=1.5e-3)

        # 使用 buttord 函数计算 Butterworth 滤波器的阶数 N 和归一化截止频率 Wn
        N, Wn = buttord([0.2, 0.5], [0.14, 0.6], 3, 40)
        # 使用 butter 函数生成 Butterworth 滤波器的 second-order section (SOS) 表示
        sos = butter(N, Wn, 'band', output='sos')
        # 使用 sosfreqz 函数计算频率响应 w 和幅度响应
    # 定义测试函数 test_sosfreqz_design_ellip，用于测试 ellipord、ellip 和 sosfreqz 函数的设计
    def test_sosfreqz_design_ellip(self):
        # 计算 Elliptic 滤波器的阶数 N 和归一化频率 Wn
        N, Wn = ellipord(0.3, 0.1, 3, 60)
        # 根据计算得到的 N、Wn 设计 Elliptic 滤波器并返回二阶节矩阵
        sos = ellip(N, 0.3, 60, Wn, 'high', output='sos')
        # 计算滤波器的频率响应
        w, h = sosfreqz(sos)
        # 取频率响应的幅值
        h = np.abs(h)
        # 将频率轴 w 归一化到 [0, 1]
        w /= np.pi
        # 断言在频率大于等于 0.3 时，20*log10(h) 接近 0，允许误差为 3.01 dB
        assert_allclose(20 * np.log10(h[w >= 0.3]), 0, atol=3.01)
        # 断言在频率小于等于 0.1 时，h 接近 0，允许误差为 1.5e-3
        assert_allclose(h[w <= 0.1], 0., atol=1.5e-3)  # <= -60 dB (approx)

        # 重新计算不同参数下的 Elliptic 滤波器的阶数 N 和归一化频率 Wn
        N, Wn = ellipord(0.3, 0.2, .5, 150)
        # 根据新的 N、Wn 设计 Elliptic 滤波器并返回二阶节矩阵
        sos = ellip(N, .5, 150, Wn, 'high', output='sos')
        # 计算滤波器的频率响应
        w, h = sosfreqz(sos)
        # 计算频率响应的 dB 值
        dB = 20*np.log10(np.maximum(np.abs(h), 1e-10))
        # 将频率轴 w 归一化到 [0, 1]
        w /= np.pi
        # 断言在频率大于等于 0.3 时，dB 接近 0，允许误差为 0.55 dB
        assert_allclose(dB[w >= 0.3], 0, atol=.55)
        # 允许在 -150 dB 的上界上有一些数值上的余量，因此检查 dB[w <= 0.2] 是否小于或几乎等于 -150
        assert dB[w <= 0.2].max() < -150*(1 - 1e-12)

    @mpmath_check("0.10")
    # 标记为 mpmath_check 的测试函数，要求 mpmath 版本至少为 "0.10"
    def test_sos_freqz_against_mp(self):
        # 比较将高阶 Butterworth 滤波器应用于 sosfreqz 函数得到的结果与使用 mpmath 计算的结果
        # （对于如此高阶的滤波器，signal.freqz 函数表现不佳）
        from . import mpsig
        # 设置 Butterworth 滤波器的阶数和归一化频率
        N = 500
        order = 25
        Wn = 0.15
        # 设置 mpmath 的工作精度为 80 位小数
        with mpmath.workdps(80):
            # 使用 mpmath 计算 Butterworth 滤波器的零点、极点和增益
            z_mp, p_mp, k_mp = mpsig.butter_lp(order, Wn)
            # 使用 mpmath 计算频率响应
            w_mp, h_mp = mpsig.zpkfreqz(z_mp, p_mp, k_mp, N)
        # 将 mpmath 计算得到的频率和频率响应转换为 NumPy 数组
        w_mp = np.array([float(x) for x in w_mp])
        h_mp = np.array([complex(x) for x in h_mp])

        # 使用 Butterworth 函数设计滤波器，并使用 sosfreqz 函数计算其频率响应
        sos = butter(order, Wn, output='sos')
        w, h = sosfreqz(sos, worN=N)
        # 断言频率响应 w 和 h 与 mpmath 计算的结果 w_mp 和 h_mp 接近，允许误差为 1e-12 和 1e-14
        assert_allclose(w, w_mp, rtol=1e-12, atol=1e-14)
        assert_allclose(h, h_mp, rtol=1e-12, atol=1e-14)
    def test_fs_param(self):
        # 设定采样频率为900 Hz
        fs = 900
        # Second Order Sections (SOS) coefficients
        sos = [[0.03934683014103762, 0.07869366028207524, 0.03934683014103762,
                1.0, -0.37256600288916636, 0.0],
               [1.0, 1.0, 0.0, 1.0, -0.9495739996946778, 0.45125966317124144]]

        # N = None, whole=False，计算频率响应
        w1, h1 = sosfreqz(sos, fs=fs)
        w2, h2 = sosfreqz(sos)
        # 断言两种计算方式得到的频率响应在数值上相近
        assert_allclose(h1, h2)
        # 断言频率轴的计算结果与预期一致
        assert_allclose(w1, np.linspace(0, fs/2, 512, endpoint=False))

        # N = None, whole=True，计算整个频率响应
        w1, h1 = sosfreqz(sos, whole=True, fs=fs)
        w2, h2 = sosfreqz(sos, whole=True)
        # 断言两种计算方式得到的整个频率响应在数值上相近，容忍度为1e-27
        assert_allclose(h1, h2, atol=1e-27)
        # 断言整个频率轴的计算结果与预期一致
        assert_allclose(w1, np.linspace(0, fs, 512, endpoint=False))

        # N = 5, whole=False，计算截断的频率响应
        w1, h1 = sosfreqz(sos, 5, fs=fs)
        w2, h2 = sosfreqz(sos, 5)
        # 断言两种计算方式得到的截断频率响应在数值上相近
        assert_allclose(h1, h2)
        # 断言截断频率轴的计算结果与预期一致
        assert_allclose(w1, np.linspace(0, fs/2, 5, endpoint=False))

        # N = 5, whole=True，计算截断的整个频率响应
        w1, h1 = sosfreqz(sos, 5, whole=True, fs=fs)
        w2, h2 = sosfreqz(sos, 5, whole=True)
        # 断言两种计算方式得到的截断整个频率响应在数值上相近
        assert_allclose(h1, h2)
        # 断言截断整个频率轴的计算结果与预期一致
        assert_allclose(w1, np.linspace(0, fs, 5, endpoint=False))

        # w 是一个数组的情况，计算指定频率响应
        for w in ([123], (123,), np.array([123]), (50, 123, 230),
                  np.array([50, 123, 230])):
            w1, h1 = sosfreqz(sos, w, fs=fs)
            w2, h2 = sosfreqz(sos, 2*pi*np.array(w)/fs)
            # 断言两种计算方式得到的指定频率响应在数值上相近
            assert_allclose(h1, h2)
            # 断言频率轴的计算结果与预期一致
            assert_allclose(w, w1)

    def test_w_or_N_types(self):
        # 测量在7或8个等间隔点上的频率响应
        for N in (7, np.int8(7), np.int16(7), np.int32(7), np.int64(7),
                  np.array(7),
                  8, np.int8(8), np.int16(8), np.int32(8), np.int64(8),
                  np.array(8)):

            # 使用指定的 N 值计算频率响应
            w, h = sosfreqz([1, 0, 0, 1, 0, 0], worN=N)
            # 断言计算得到的频率轴与预期的几乎相等
            assert_array_almost_equal(w, np.pi * np.arange(N) / N)
            # 断言计算得到的频率响应与预期的几乎相等
            assert_array_almost_equal(h, np.ones(N))

            # 使用指定的 N 值和采样频率计算频率响应
            w, h = sosfreqz([1, 0, 0, 1, 0, 0], worN=N, fs=100)
            # 断言计算得到的频率轴与预期的几乎相等
            assert_array_almost_equal(w, np.linspace(0, 50, N, endpoint=False))
            # 断言计算得到的频率响应与预期的几乎相等
            assert_array_almost_equal(h, np.ones(N))

        # 测量在频率为8 Hz 处的频率响应
        for w in (8.0, 8.0+0j):
            # 仅在指定了采样频率时才有意义
            w_out, h = sosfreqz([1, 0, 0, 1, 0, 0], worN=w, fs=100)
            # 断言计算得到的频率轴与预期的几乎相等
            assert_array_almost_equal(w_out, [8])
            # 断言计算得到的频率响应与预期的几乎相等
            assert_array_almost_equal(h, [1])

    def test_fs_validation(self):
        # 使用 Butterworth 滤波器系数创建 SOS 形式
        sos = butter(4, 0.2, output='sos')
        # 断言当采样频率参数为非单一标量时引发 ValueError 异常
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            sosfreqz(sos, fs=np.array([10, 20]))
class TestFreqz_zpk:

    def test_ticket1441(self):
        """Regression test for ticket 1441."""
        # 因为之前 freqz 使用 arange 而不是 linspace，
        # 当 N 很大时，可能会返回比请求的点数多一个。
        N = 100000
        w, h = freqz_zpk([0.5], [0.5], 1.0, worN=N)
        assert_equal(w.shape, (N,))

    def test_basic(self):
        # 测试基本情况，计算频率响应
        w, h = freqz_zpk([0.5], [0.5], 1.0, worN=8)
        assert_array_almost_equal(w, np.pi * np.arange(8.0) / 8)
        assert_array_almost_equal(h, np.ones(8))

    def test_basic_whole(self):
        # 测试整个频率范围的情况
        w, h = freqz_zpk([0.5], [0.5], 1.0, worN=8, whole=True)
        assert_array_almost_equal(w, 2 * np.pi * np.arange(8.0) / 8)
        assert_array_almost_equal(h, np.ones(8))

    def test_vs_freqz(self):
        # 与 freqz 函数的结果进行比较
        b, a = cheby1(4, 5, 0.5, analog=False, output='ba')
        z, p, k = cheby1(4, 5, 0.5, analog=False, output='zpk')

        w1, h1 = freqz(b, a)
        w2, h2 = freqz_zpk(z, p, k)
        assert_allclose(w1, w2)
        assert_allclose(h1, h2, rtol=1e-6)

    def test_backward_compat(self):
        # 测试向后兼容性，检查 None 是否作为默认参数的包装器
        w1, h1 = freqz_zpk([0.5], [0.5], 1.0)
        w2, h2 = freqz_zpk([0.5], [0.5], 1.0, None)
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(h1, h2)

    def test_fs_param(self):
        fs = 900
        z = [-1, -1, -1]
        p = [0.4747869998473389+0.4752230717749344j, 0.37256600288916636,
             0.4747869998473389-0.4752230717749344j]
        k = 0.03934683014103762

        # N = None, whole=False
        # 测试参数 fs 对频率轴的影响
        w1, h1 = freqz_zpk(z, p, k, whole=False, fs=fs)
        w2, h2 = freqz_zpk(z, p, k, whole=False)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs/2, 512, endpoint=False))

        # N = None, whole=True
        w1, h1 = freqz_zpk(z, p, k, whole=True, fs=fs)
        w2, h2 = freqz_zpk(z, p, k, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 512, endpoint=False))

        # N = 5, whole=False
        w1, h1 = freqz_zpk(z, p, k, 5, fs=fs)
        w2, h2 = freqz_zpk(z, p, k, 5)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs/2, 5, endpoint=False))

        # N = 5, whole=True
        w1, h1 = freqz_zpk(z, p, k, 5, whole=True, fs=fs)
        w2, h2 = freqz_zpk(z, p, k, 5, whole=True)
        assert_allclose(h1, h2)
        assert_allclose(w1, np.linspace(0, fs, 5, endpoint=False))

        # w is an array_like
        for w in ([123], (123,), np.array([123]), (50, 123, 230),
                  np.array([50, 123, 230])):
            w1, h1 = freqz_zpk(z, p, k, w, fs=fs)
            w2, h2 = freqz_zpk(z, p, k, 2*pi*np.array(w)/fs)
            assert_allclose(h1, h2)
            assert_allclose(w, w1)
    # 定义测试函数，用于测试频率响应函数在不同输入类型下的行为
    def test_w_or_N_types(self):
        # 在8个等间距点进行测量
        for N in (8, np.int8(8), np.int16(8), np.int32(8), np.int64(8),
                  np.array(8)):
            # 调用频率响应函数，返回频率和响应
            w, h = freqz_zpk([], [], 1, worN=N)
            # 断言频率数组接近于等差序列
            assert_array_almost_equal(w, np.pi * np.arange(8) / 8.)
            # 断言响应数组为全1
            assert_array_almost_equal(h, np.ones(8))

            # 调用频率响应函数，指定采样频率为100 Hz，返回频率和响应
            w, h = freqz_zpk([], [], 1, worN=N, fs=100)
            # 断言频率数组接近于线性等间距序列
            assert_array_almost_equal(w, np.linspace(0, 50, 8, endpoint=False))
            # 断言响应数组为全1
            assert_array_almost_equal(h, np.ones(8))

        # 在频率为8 Hz时进行测量
        for w in (8.0, 8.0+0j):
            # 只有在指定了采样频率时才有意义
            w_out, h = freqz_zpk([], [], 1, worN=w, fs=100)
            # 断言输出频率为8 Hz
            assert_array_almost_equal(w_out, [8])
            # 断言响应为1
            assert_array_almost_equal(h, [1])

    # 定义测试函数，用于验证采样频率参数的有效性
    def test_fs_validation(self):
        # 使用 pytest 断言捕获到采样频率不合法的异常，异常信息包含指定文本
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            freqz_zpk([1.0], [1.0], [1.0], fs=np.array([10, 20]))

        # 使用 pytest 断言捕获到采样频率为 None 的异常，异常信息包含指定文本
        with pytest.raises(ValueError, match="Sampling.*be none."):
            freqz_zpk([1.0], [1.0], [1.0], fs=None)
class TestNormalize:

    def test_allclose(self):
        """Test for false positive on allclose in normalize() in
        filter_design.py"""
        # 测试 signal.normalize 中 allclose 的误报情况，并检查与 MATLAB 输出的匹配性

        # 这些系数是从 MATLAB 中执行 `[b,a] = cheby1(8, 0.5, 0.048)'` 后返回的。
        # 每个系数至少有 15 个有效数字，因此测试误差应在 1e-13 左右（如果不同平台有不同的舍入误差，可以稍作调整）。
        b_matlab = np.array([2.150733144728282e-11, 1.720586515782626e-10,
                             6.022052805239190e-10, 1.204410561047838e-09,
                             1.505513201309798e-09, 1.204410561047838e-09,
                             6.022052805239190e-10, 1.720586515782626e-10,
                             2.150733144728282e-11])
        a_matlab = np.array([1.000000000000000e+00, -7.782402035027959e+00,
                             2.654354569747454e+01, -5.182182531666387e+01,
                             6.334127355102684e+01, -4.963358186631157e+01,
                             2.434862182949389e+01, -6.836925348604676e+00,
                             8.412934944449140e-01])

        # 这是经过 signal.iirfilter 中等效步骤处理后传递给 signal.normalize 的输入
        b_norm_in = np.array([1.5543135865293012e-06, 1.2434508692234413e-05,
                              4.3520780422820447e-05, 8.7041560845640893e-05,
                              1.0880195105705122e-04, 8.7041560845640975e-05,
                              4.3520780422820447e-05, 1.2434508692234413e-05,
                              1.5543135865293012e-06])
        a_norm_in = np.array([7.2269025909127173e+04, -5.6242661430467968e+05,
                              1.9182761917308895e+06, -3.7451128364682454e+06,
                              4.5776121393762771e+06, -3.5869706138592605e+06,
                              1.7596511818472347e+06, -4.9409793515707983e+05,
                              6.0799461347219651e+04])

        # 调用 normalize 函数，获取输出的 b 和 a
        b_output, a_output = normalize(b_norm_in, a_norm_in)

        # 对 b 和 a 进行准确度比较。b 的测试使用 decimal=14，a 的测试使用 decimal=13。
        # 为了一致性起见，两者都设定为 decimal=13。如果在其他平台上出现问题，可以适度放宽这个限制。
        assert_array_almost_equal(b_matlab, b_output, decimal=13)
        assert_array_almost_equal(a_matlab, a_output, decimal=13)
    def test_errors(self):
        """Test the error cases."""
        # 测试除法中的特殊情况：分母全为零
        assert_raises(ValueError, normalize, [1, 2], 0)

        # 测试除法中的特殊情况：分母不是一维数组
        assert_raises(ValueError, normalize, [1, 2], [[1]])

        # 测试除法中的特殊情况：分子维度过多
        assert_raises(ValueError, normalize, [[[1, 2]]], 1)
class TestLp2lp:

    def test_basic(self):
        # 设定低通滤波器的分子和分母系数
        b = [1]
        a = [1, np.sqrt(2), 1]
        # 进行低通到低通滤波器转换
        b_lp, a_lp = lp2lp(b, a, 0.38574256627112119)
        # 检验转换后的分子系数是否准确
        assert_array_almost_equal(b_lp, [0.1488], decimal=4)
        # 检验转换后的分母系数是否准确
        assert_array_almost_equal(a_lp, [1, 0.5455, 0.1488], decimal=4)


class TestLp2hp:

    def test_basic(self):
        # 设定低通滤波器的分子和分母系数
        b = [0.25059432325190018]
        a = [1, 0.59724041654134863, 0.92834805757524175, 0.25059432325190018]
        # 进行低通到高通滤波器转换
        b_hp, a_hp = lp2hp(b, a, 2*np.pi*5000)
        # 检验转换后的分子系数是否准确
        assert_allclose(b_hp, [1, 0, 0, 0])
        # 检验转换后的分母系数是否准确
        assert_allclose(a_hp, [1, 1.1638e5, 2.3522e9, 1.2373e14], rtol=1e-4)


class TestLp2bp:

    def test_basic(self):
        # 设定低通滤波器的分子和分母系数
        b = [1]
        a = [1, 2, 2, 1]
        # 进行低通到带通滤波器转换
        b_bp, a_bp = lp2bp(b, a, 2*np.pi*4000, 2*np.pi*2000)
        # 检验转换后的分子系数是否准确
        assert_allclose(b_bp, [1.9844e12, 0, 0, 0], rtol=1e-6)
        # 检验转换后的分母系数是否准确
        assert_allclose(a_bp, [1, 2.5133e4, 2.2108e9, 3.3735e13,
                               1.3965e18, 1.0028e22, 2.5202e26], rtol=1e-4)


class TestLp2bs:

    def test_basic(self):
        # 设定低通滤波器的分子和分母系数
        b = [1]
        a = [1, 1]
        # 进行低通到带阻滤波器转换
        b_bs, a_bs = lp2bs(b, a, 0.41722257286366754, 0.18460575326152251)
        # 检验转换后的分子系数是否准确
        assert_array_almost_equal(b_bs, [1, 0, 0.17407], decimal=5)
        # 检验转换后的分母系数是否准确
        assert_array_almost_equal(a_bs, [1, 0.18461, 0.17407], decimal=5)


class TestBilinear:

    def test_basic(self):
        # 设定连续时间滤波器的分子和分母系数
        b = [0.14879732743343033]
        a = [1, 0.54552236880522209, 0.14879732743343033]
        # 进行双线性变换
        b_z, a_z = bilinear(b, a, 0.5)
        # 检验变换后的分子系数是否准确
        assert_array_almost_equal(b_z, [0.087821, 0.17564, 0.087821],
                                  decimal=5)
        # 检验变换后的分母系数是否准确
        assert_array_almost_equal(a_z, [1, -1.0048, 0.35606], decimal=4)

        # 再次设定连续时间滤波器的分子和分母系数
        b = [1, 0, 0.17407467530697837]
        a = [1, 0.18460575326152251, 0.17407467530697837]
        # 进行双线性变换
        b_z, a_z = bilinear(b, a, 0.5)
        # 检验变换后的分子系数是否准确
        assert_array_almost_equal(b_z, [0.86413, -1.2158, 0.86413],
                                  decimal=4)
        # 检验变换后的分母系数是否准确
        assert_array_almost_equal(a_z, [1, -1.2158, 0.72826],
                                  decimal=4)

    def test_fs_validation(self):
        # 设定连续时间滤波器的分子和分母系数
        b = [0.14879732743343033]
        a = [1, 0.54552236880522209, 0.14879732743343033]
        # 检验是否引发异常，因为采样频率应为单一标量
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            bilinear(b, a, fs=np.array([10, 20]))

        # 检验是否引发异常，因为采样频率不能为 None
        with pytest.raises(ValueError, match="Sampling.*be none"):
            bilinear(b, a, fs=None)


class TestLp2lp_zpk:

    def test_basic(self):
        # 设定连续时间滤波器的零点、极点和增益
        z = []
        p = [(-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2)]
        k = 1
        # 进行零极点增益到零极点增益的转换
        z_lp, p_lp, k_lp = lp2lp_zpk(z, p, k, 5)
        # 检验转换后的零点是否准确
        assert_array_equal(z_lp, [])
        # 检验转换后的极点是否准确
        assert_allclose(sort(p_lp), sort(p)*5)
        # 检验转换后的增益是否准确
        assert_allclose(k_lp, 25)

        # 再次设定连续时间滤波器的零点、极点和增益
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3
        # 进行零极点增益到零极点增益的转换
        z_lp, p_lp, k_lp = lp2lp_zpk(z, p, k, 20)
        # 检验转换后的零点是否准确
        assert_allclose(sort(z_lp), sort([-40j, +40j]))
        # 检验转换后的极点是否准确
        assert_allclose(sort(p_lp), sort([-15, -10-10j, -10+10j]))
        # 检验转换后的增益是否准确
        assert_allclose(k_lp, 60)
    # 定义一个单元测试方法，用于测试频率采样验证
    def test_fs_validation(self):
        # 定义复数列表 z，包含两个复数
        z = [-2j, +2j]
        # 定义极点列表 p，包含三个复数
        p = [-0.75, -0.5 - 0.5j, -0.5 + 0.5j]
        # 定义增益 k，为整数 3

        # 使用 pytest 检查是否会引发 ValueError 异常，并且异常消息匹配 "Sampling.*single scalar"
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            # 调用 bilinear_zpk 函数，传入 z, p, k 作为参数，同时指定 fs 参数为 np.array([10, 20])
            bilinear_zpk(z, p, k, fs=np.array([10, 20]))

        # 使用 pytest 检查是否会引发 ValueError 异常，并且异常消息匹配 "Sampling.*be none"
        with pytest.raises(ValueError, match="Sampling.*be none"):
            # 调用 bilinear_zpk 函数，传入 z, p, k 作为参数，同时指定 fs 参数为 None
            bilinear_zpk(z, p, k, fs=None)
class TestLp2hp_zpk:

    def test_basic(self):
        # 初始化零点、极点和增益
        z = []
        p = [(-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2)]
        k = 1

        # 转换为高通滤波器的零点、极点和增益
        z_hp, p_hp, k_hp = lp2hp_zpk(z, p, k, 5)
        # 断言结果与期望相等
        assert_array_equal(z_hp, [0, 0])
        # 断言排序后的结果非常接近
        assert_allclose(sort(p_hp), sort(p)*5)
        # 断言增益与期望相等
        assert_allclose(k_hp, 1)

        # 更新零点、极点和增益
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3
        # 转换为高通滤波器的零点、极点和增益
        z_hp, p_hp, k_hp = lp2hp_zpk(z, p, k, 6)
        # 断言排序后的结果非常接近
        assert_allclose(sort(z_hp), sort([-3j, 0, +3j]))
        # 断言排序后的结果非常接近
        assert_allclose(sort(p_hp), sort([-8, -6-6j, -6+6j]))
        # 断言增益与期望相等
        assert_allclose(k_hp, 32)


class TestLp2bp_zpk:

    def test_basic(self):
        # 初始化零点、极点和增益
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3

        # 转换为带通滤波器的零点、极点和增益
        z_bp, p_bp, k_bp = lp2bp_zpk(z, p, k, 15, 8)
        # 断言排序后的结果非常接近
        assert_allclose(sort(z_bp), sort([-25j, -9j, 0, +9j, +25j]))
        # 断言排序后的结果非常接近
        assert_allclose(sort(p_bp), sort([-3 + 6j*sqrt(6),
                                          -3 - 6j*sqrt(6),
                                          +2j+sqrt(-8j-225)-2,
                                          -2j+sqrt(+8j-225)-2,
                                          +2j-sqrt(-8j-225)-2,
                                          -2j-sqrt(+8j-225)-2, ]))
        # 断言增益与期望相等
        assert_allclose(k_bp, 24)


class TestLp2bs_zpk:

    def test_basic(self):
        # 初始化零点、极点和增益
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3

        # 转换为带阻滤波器的零点、极点和增益
        z_bs, p_bs, k_bs = lp2bs_zpk(z, p, k, 35, 12)
        # 断言排序后的结果非常接近
        assert_allclose(sort(z_bs), sort([+35j, -35j,
                                          +3j+sqrt(1234)*1j,
                                          -3j+sqrt(1234)*1j,
                                          +3j-sqrt(1234)*1j,
                                          -3j-sqrt(1234)*1j]))
        # 断言排序后的结果非常接近
        assert_allclose(sort(p_bs), sort([+3j*sqrt(129) - 8,
                                          -3j*sqrt(129) - 8,
                                          (-6 + 6j) - sqrt(-1225 - 72j),
                                          (-6 - 6j) - sqrt(-1225 + 72j),
                                          (-6 + 6j) + sqrt(-1225 - 72j),
                                          (-6 - 6j) + sqrt(-1225 + 72j), ]))
        # 断言增益与期望相等
        assert_allclose(k_bs, 32)


class TestBilinear_zpk:

    def test_basic(self):
        # 初始化零点、极点和增益
        z = [-2j, +2j]
        p = [-0.75, -0.5-0.5j, -0.5+0.5j]
        k = 3

        # 双线性变换为数字域的零点、极点和增益
        z_d, p_d, k_d = bilinear_zpk(z, p, k, 10)
        # 断言排序后的结果非常接近
        assert_allclose(sort(z_d), sort([(20-2j)/(20+2j), (20+2j)/(20-2j),
                                         -1]))
        # 断言排序后的结果非常接近
        assert_allclose(sort(p_d), sort([77/83,
                                         (1j/2 + 39/2) / (41/2 - 1j/2),
                                         (39/2 - 1j/2) / (1j/2 + 41/2), ]))
        # 断言增益与期望相等
        assert_allclose(k_d, 9696/69803)


class TestPrototypeType:
    pass  # 没有需要注释的代码
    def test_output_type(self):
        # 定义测试函数，验证函数原型的输出类型是否一致，应输出数组而非列表
        # 参考：https://github.com/scipy/scipy/pull/441
        for func in (buttap,        # 遍历函数列表，包括 buttap
                     besselap,      # besselap 函数
                     lambda N: cheb1ap(N, 1),    # 使用 lambda 表达式定义的 cheb1ap 函数
                     lambda N: cheb2ap(N, 20),   # 使用 lambda 表达式定义的 cheb2ap 函数
                     lambda N: ellipap(N, 1, 20)):   # 使用 lambda 表达式定义的 ellipap 函数
            for N in range(7):   # 遍历 N 从 0 到 6 的范围
                z, p, k = func(N)   # 调用 func 函数，并获取返回的 z, p, k 值
                assert_(isinstance(z, np.ndarray))   # 断言 z 是 numpy 数组
                assert_(isinstance(p, np.ndarray))   # 断言 p 是 numpy 数组
def dB(x):
    # 返回以分贝为单位的幅度值，避免除零警告
    # （并处理一些“不是较小有序”的错误，当 -inf 出现时）
    return 20 * np.log10(np.maximum(np.abs(x), np.finfo(np.float64).tiny))


class TestButtord:

    def test_lowpass(self):
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60
        # 计算巴特沃斯滤波器的阶数 N 和截止频率 Wn
        N, Wn = buttord(wp, ws, rp, rs, False)
        # 根据 N 和 Wn 设计低通巴特沃斯滤波器
        b, a = butter(N, Wn, 'lowpass', False)
        # 计算频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 断言频率响应在指定频率范围内的衰减不超过指定值
        assert_array_less(-rp, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs)

        # 断言计算得到的阶数 N 符合预期值
        assert_equal(N, 16)
        # 断言计算得到的截止频率 Wn 符合预期值
        assert_allclose(Wn, 2.0002776782743284e-01, rtol=1e-15)

    def test_highpass(self):
        wp = 0.3
        ws = 0.2
        rp = 3
        rs = 70
        # 计算巴特沃斯滤波器的阶数 N 和截止频率 Wn
        N, Wn = buttord(wp, ws, rp, rs, False)
        # 根据 N 和 Wn 设计高通巴特沃斯滤波器
        b, a = butter(N, Wn, 'highpass', False)
        # 计算频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 断言频率响应在指定频率范围内的衰减不超过指定值
        assert_array_less(-rp, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs)

        # 断言计算得到的阶数 N 符合预期值
        assert_equal(N, 18)
        # 断言计算得到的截止频率 Wn 符合预期值
        assert_allclose(Wn, 2.9996603079132672e-01, rtol=1e-15)

    def test_bandpass(self):
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        rp = 3
        rs = 80
        # 计算巴特沃斯滤波器的阶数 N 和截止频率 Wn
        N, Wn = buttord(wp, ws, rp, rs, False)
        # 根据 N 和 Wn 设计带通巴特沃斯滤波器
        b, a = butter(N, Wn, 'bandpass', False)
        # 计算频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 断言频率响应在指定频率范围内的衰减不超过指定值
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)

        # 断言计算得到的阶数 N 符合预期值
        assert_equal(N, 18)
        # 断言计算得到的截止频率 Wn 符合预期值
        assert_allclose(Wn, [1.9998742411409134e-01, 5.0002139595676276e-01],
                        rtol=1e-15)

    def test_bandstop(self):
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        rp = 3
        rs = 90
        # 计算巴特沃斯滤波器的阶数 N 和截止频率 Wn
        N, Wn = buttord(wp, ws, rp, rs, False)
        # 根据 N 和 Wn 设计带阻巴特沃斯滤波器
        b, a = butter(N, Wn, 'bandstop', False)
        # 计算频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 断言频率响应在指定频率范围内的衰减不超过指定值
        assert_array_less(-rp,
                          dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]),
                          -rs)

        # 断言计算得到的阶数 N 符合预期值
        assert_equal(N, 20)
        # 断言计算得到的截止频率 Wn 符合预期值
        assert_allclose(Wn, [1.4759432329294042e-01, 5.9997365985276407e-01],
                        rtol=1e-6)

    def test_analog(self):
        wp = 200
        ws = 600
        rp = 3
        rs = 60
        # 计算模拟域下的巴特沃斯滤波器的阶数 N 和截止频率 Wn
        N, Wn = buttord(wp, ws, rp, rs, True)
        # 根据 N 和 Wn 设计模拟域下的低通巴特沃斯滤波器
        b, a = butter(N, Wn, 'lowpass', True)
        # 计算模拟域下的频率响应
        w, h = freqs(b, a)
        # 断言模拟域下频率响应在指定频率范围内的衰减不超过指定值
        assert_array_less(-rp, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs)

        # 断言计算得到的阶数 N 符合预期值
        assert_equal(N, 7)
        # 断言计算得到的截止频率 Wn 符合预期值
        assert_allclose(Wn, 2.0006785355671877e+02, rtol=1e-15)

        # 在模拟域下进一步验证阶数和截止频率的计算
        n, Wn = buttord(1, 550/450, 1, 26, analog=True)
        assert_equal(n, 19)
        assert_allclose(Wn, 1.0361980524629517, rtol=1e-15)

        # 验证特定参数情况下的阶数计算
        assert_equal(buttord(1, 1.2, 1, 80, analog=True)[0], 55)
    def test_fs_param(self):
        # 设定通带边界频率和阻带边界频率的列表
        wp = [4410, 11025]
        ws = [2205, 13230]
        # 设定通带最大允许衰减（dB）
        rp = 3
        # 设定阻带最小要求衰减（dB）
        rs = 80
        # 设定采样频率（Hz）
        fs = 44100
        # 计算巴特沃斯滤波器的最佳阶数和截止频率
        N, Wn = buttord(wp, ws, rp, rs, False, fs=fs)
        # 根据计算得到的阶数和截止频率设计巴特沃斯滤波器
        b, a = butter(N, Wn, 'bandpass', False, fs=fs)
        # 计算频率响应曲线
        w, h = freqz(b, a, fs=fs)
        # 断言通带内的最小衰减满足要求
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        # 断言阻带外的最大衰减满足要求
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)

        # 断言计算得到的阶数符合预期值
        assert_equal(N, 18)
        # 断言计算得到的截止频率符合预期值
        assert_allclose(Wn, [4409.722701715714, 11025.47178084662],
                        rtol=1e-15)

    def test_invalid_input(self):
        # 测试输入参数无效的情况下是否会引发 ValueError 异常
        with pytest.raises(ValueError) as exc_info:
            buttord([20, 50], [14, 60], 3, 2)
        assert "gpass should be smaller than gstop" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            buttord([20, 50], [14, 60], -1, 2)
        assert "gpass should be larger than 0.0" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            buttord([20, 50], [14, 60], 1, -2)
        assert "gstop should be larger than 0.0" in str(exc_info.value)

    def test_runtime_warnings(self):
        # 测试在运行时是否会生成 RuntimeWarning 警告信息
        msg = "Order is zero.*|divide by zero encountered"
        with pytest.warns(RuntimeWarning, match=msg):
            buttord(0.0, 1.0, 3, 60)

    def test_ellip_butter(self):
        # 测试椭圆滤波器与已知版本的输出是否一致
        # 用于比较的值是在过去某个 scipy 版本（例如 1.9.1）生成的
        n, wn = buttord([0.1, 0.6], [0.2, 0.5], 3, 60)
        assert n == 14

    def test_fs_validation(self):
        # 测试采样频率参数的有效性验证
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60

        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            buttord(wp, ws, rp, rs, False, fs=np.array([10, 20]))
class TestCheb1ord:

    def test_lowpass(self):
        wp = 0.2  # 低通滤波器的通带端频率
        ws = 0.3  # 低通滤波器的阻带端频率
        rp = 3    # 通带最大允许波纹的增益（dB）
        rs = 60   # 阻带最小要求衰减（dB）
        # 计算 Chebyshev Type I 滤波器的阶数 N 和归一化频率 Wn
        N, Wn = cheb1ord(wp, ws, rp, rs, False)
        # 根据给定的 N、Wn、滤波器类型（低通）、模拟滤波器标志位，设计 Chebyshev Type I 低通滤波器
        b, a = cheby1(N, rp, Wn, 'low', False)
        # 计算频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 检查频率响应在指定频率范围内的幅度响应是否满足要求
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)

        # 断言检查设计的滤波器阶数是否为 8
        assert_equal(N, 8)
        # 断言检查设计的归一化频率是否接近于 0.2
        assert_allclose(Wn, 0.2, rtol=1e-15)

    def test_highpass(self):
        wp = 0.3  # 高通滤波器的通带端频率
        ws = 0.2  # 高通滤波器的阻带端频率
        rp = 3    # 通带最大允许波纹的增益（dB）
        rs = 70   # 阻带最小要求衰减（dB）
        # 计算 Chebyshev Type I 滤波器的阶数 N 和归一化频率 Wn
        N, Wn = cheb1ord(wp, ws, rp, rs, False)
        # 根据给定的 N、Wn、滤波器类型（高通）、模拟滤波器标志位，设计 Chebyshev Type I 高通滤波器
        b, a = cheby1(N, rp, Wn, 'high', False)
        # 计算频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 检查频率响应在指定频率范围内的幅度响应是否满足要求
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)

        # 断言检查设计的滤波器阶数是否为 9
        assert_equal(N, 9)
        # 断言检查设计的归一化频率是否接近于 0.3
        assert_allclose(Wn, 0.3, rtol=1e-15)

    def test_bandpass(self):
        wp = [0.2, 0.5]  # 带通滤波器的通带端频率范围
        ws = [0.1, 0.6]  # 带通滤波器的阻带端频率范围
        rp = 3           # 通带最大允许波纹的增益（dB）
        rs = 80          # 阻带最小要求衰减（dB）
        # 计算 Chebyshev Type I 滤波器的阶数 N 和归一化频率 Wn
        N, Wn = cheb1ord(wp, ws, rp, rs, False)
        # 根据给定的 N、Wn、滤波器类型（带通）、模拟滤波器标志位，设计 Chebyshev Type I 带通滤波器
        b, a = cheby1(N, rp, Wn, 'band', False)
        # 计算频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 检查频率响应在指定频率范围内的幅度响应是否满足要求
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)

        # 断言检查设计的滤波器阶数是否为 9
        assert_equal(N, 9)
        # 断言检查设计的归一化频率范围是否接近于 [0.2, 0.5]
        assert_allclose(Wn, [0.2, 0.5], rtol=1e-15)

    def test_bandstop(self):
        wp = [0.1, 0.6]  # 带阻滤波器的通带端频率范围
        ws = [0.2, 0.5]  # 带阻滤波器的阻带端频率范围
        rp = 3           # 通带最大允许波纹的增益（dB）
        rs = 90          # 阻带最小要求衰减（dB）
        # 计算 Chebyshev Type I 滤波器的阶数 N 和归一化频率 Wn
        N, Wn = cheb1ord(wp, ws, rp, rs, False)
        # 根据给定的 N、Wn、滤波器类型（带阻）、模拟滤波器标志位，设计 Chebyshev Type I 带阻滤波器
        b, a = cheby1(N, rp, Wn, 'stop', False)
        # 计算频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 检查频率响应在指定频率范围内的幅度响应是否满足要求
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]),
                          -rs + 0.1)

        # 断言检查设计的滤波器阶数是否为 10
        assert_equal(N, 10)
        # 断言检查设计的归一化频率范围是否接近于 [0.14758232569947785, 0.6]
        assert_allclose(Wn, [0.14758232569947785, 0.6], rtol=1e-5)

    def test_analog(self):
        wp = 700   # 模拟滤波器的通带端频率
        ws = 100   # 模拟滤波器的阻带端频率
        rp = 3     # 通带最大允许波纹的增益（dB）
        rs = 70    # 阻带最小要求衰减（dB）
        # 计算 Chebyshev Type I 模拟滤波器的阶数 N 和归一化频率 Wn
        N, Wn = cheb1ord(wp, ws, rp, rs, True)
        # 根据给定的 N、Wn、滤波器类型（高通）、模拟滤波器标志位，设计 Chebyshev Type I 模拟高通滤波器
        b, a = cheby1(N, rp, Wn, 'high', True)
        # 计算频率响应
        w, h = freqs(b, a
    # 定义一个测试方法来测试无效输入的情况
    def test_invalid_input(self):
        # 检查当 gpass 大于 gstop 时，是否会引发 ValueError 异常
        with pytest.raises(ValueError) as exc_info:
            cheb1ord(0.2, 0.3, 3, 2)
        # 断言异常信息中包含特定的错误信息
        assert "gpass should be smaller than gstop" in str(exc_info.value)

        # 检查当 gpass 小于等于 0.0 时，是否会引发 ValueError 异常
        with pytest.raises(ValueError) as exc_info:
            cheb1ord(0.2, 0.3, -1, 2)
        # 断言异常信息中包含特定的错误信息
        assert "gpass should be larger than 0.0" in str(exc_info.value)

        # 检查当 gstop 小于等于 0.0 时，是否会引发 ValueError 异常
        with pytest.raises(ValueError) as exc_info:
            cheb1ord(0.2, 0.3, 1, -2)
        # 断言异常信息中包含特定的错误信息
        assert "gstop should be larger than 0.0" in str(exc_info.value)

    # 定义一个测试方法来验证 ellip 和 cheb1 函数的输出
    def test_ellip_cheb1(self):
        # 该测试的目的是与过去某些已知版本的 scipy 输出进行比较
        # 使用 scipy 1.9.1 生成待比较的值（尽管这个特定版本没有特殊含义）
        # 调用 cheb1ord 函数计算 n 和 wn
        n, wn = cheb1ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        # 断言 n 的值为 7
        assert n == 7

        # 调用 cheb2ord 函数计算 n2 和 w2
        n2, w2 = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        # 断言 wn 和 w2 不完全相等
        assert not (wn == w2).all()

    # 定义一个测试方法来验证频率采样 fs 的有效性
    def test_fs_validation(self):
        # 设置测试参数
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 60

        # 使用 pytest 检查当 fs 是一个包含多个元素的数组时，是否会引发 ValueError 异常
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            cheb1ord(wp, ws, rp, rs, False, fs=np.array([10, 20]))
class TestCheb2ord:

    def test_lowpass(self):
        wp = 0.2  # 低通滤波器的通带端点
        ws = 0.3  # 低通滤波器的阻带端点
        rp = 3    # 通带最大允许波纹（dB）
        rs = 60   # 阻带最小衰减（dB）
        N, Wn = cheb2ord(wp, ws, rp, rs, False)  # 计算切比雪夫二型滤波器的阶数和归一化频率
        b, a = cheby2(N, rs, Wn, 'lp', False)    # 设计低通切比雪夫二型滤波器
        w, h = freqz(b, a)                       # 计算频率响应
        w /= np.pi                              # 归一化频率
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))  # 断言通带内频率响应在给定范围内
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)  # 断言阻带内频率响应在给定范围内

        assert_equal(N, 8)                           # 断言计算得到的阶数正确
        assert_allclose(Wn, 0.28647639976553163, rtol=1e-15)  # 断言计算得到的归一化频率正确

    def test_highpass(self):
        wp = 0.3  # 高通滤波器的通带端点
        ws = 0.2  # 高通滤波器的阻带端点
        rp = 3    # 通带最大允许波纹（dB）
        rs = 70   # 阻带最小衰减（dB）
        N, Wn = cheb2ord(wp, ws, rp, rs, False)  # 计算切比雪夫二型滤波器的阶数和归一化频率
        b, a = cheby2(N, rs, Wn, 'hp', False)    # 设计高通切比雪夫二型滤波器
        w, h = freqz(b, a)                       # 计算频率响应
        w /= np.pi                              # 归一化频率
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))  # 断言通带内频率响应在给定范围内
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)  # 断言阻带内频率响应在给定范围内

        assert_equal(N, 9)                           # 断言计算得到的阶数正确
        assert_allclose(Wn, 0.20697492182903282, rtol=1e-15)  # 断言计算得到的归一化频率正确

    def test_bandpass(self):
        wp = [0.2, 0.5]  # 带通滤波器的通带端点
        ws = [0.1, 0.6]  # 带通滤波器的阻带端点
        rp = 3           # 通带最大允许波纹（dB）
        rs = 80          # 阻带最小衰减（dB）
        N, Wn = cheb2ord(wp, ws, rp, rs, False)  # 计算切比雪夫二型滤波器的阶数和归一化频率
        b, a = cheby2(N, rs, Wn, 'bp', False)    # 设计带通切比雪夫二型滤波器
        w, h = freqz(b, a)                       # 计算频率响应
        w /= np.pi                              # 归一化频率
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))  # 断言带通内频率响应在给定范围内
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)  # 断言阻带内频率响应在给定范围内

        assert_equal(N, 9)                           # 断言计算得到的阶数正确
        assert_allclose(Wn, [0.14876937565923479, 0.59748447842351482],
                        rtol=1e-15)  # 断言计算得到的归一化频率正确

    def test_bandstop(self):
        wp = [0.1, 0.6]  # 带阻滤波器的通带端点
        ws = [0.2, 0.5]  # 带阻滤波器的阻带端点
        rp = 3           # 通带最大允许波纹（dB）
        rs = 90          # 阻带最小衰减（dB）
        N, Wn = cheb2ord(wp, ws, rp, rs, False)  # 计算切比雪夫二型滤波器的阶数和归一化频率
        b, a = cheby2(N, rs, Wn, 'bs', False)    # 设计带阻切比雪夫二型滤波器
        w, h = freqz(b, a)                       # 计算频率响应
        w /= np.pi                              # 归一化频率
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))  # 断言带阻内频率响应在给定范围内
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]),
                          -rs + 0.1)  # 断言阻带内频率响应在给定范围内

        assert_equal(N, 10)                          # 断言计算得到的阶数正确
        assert_allclose(Wn, [0.19926249974781743, 0.50125246585567362],
                        rtol=1e-6)  # 断言计算得到的归一化频率正确

    def test_analog(self):
        wp = [20, 50]  # 模拟带通滤波器的通带端点
        ws = [10, 60]  # 模拟带通滤波器的阻带端点
        rp = 3         # 通带最大允许波纹（dB）
        rs = 80        # 阻带最小衰减（dB）
        N, Wn = cheb2ord(wp, ws, rp, rs, True)   # 计算模拟切比雪夫二型滤波器的阶数和归一化频率
        b, a = cheby2(N, rs, Wn, 'bp', True)     # 设计模拟带通切比雪夫二型滤波器
        w, h = freqs(b, a)                      # 计算模拟频率响应
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))  # 断言带通内频率响应在给定范围内
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)  # 断言阻带内频率响应在给定范围内

        assert_equal(N, 11)  # 断言计算得到的阶数正确
        assert_allclose(Wn, [1.673740595370124e+01, 5.974641487254268e+01],
    def test_fs_param(self):
        # 设定通带端频率为 150 Hz
        wp = 150
        # 设定阻带端频率为 100 Hz
        ws = 100
        # 设定通带最大衰减为 3 dB
        rp = 3
        # 设定阻带最小衰减为 70 dB
        rs = 70
        # 设定采样频率为 1000 Hz
        fs = 1000
        
        # 使用 Chebyshev Type II 方法计算滤波器的阶数 N 和归一化截止频率 Wn
        N, Wn = cheb2ord(wp, ws, rp, rs, False, fs=fs)
        # 使用计算得到的阶数 N、归一化截止频率 Wn，设计 Chebyshev Type II 高通滤波器
        b, a = cheby2(N, rs, Wn, 'hp', False, fs=fs)
        # 计算滤波器的频率响应 w, h
        w, h = freqz(b, a, fs=fs)
        
        # 断言滤波器在通带端的衰减不超过 -rp - 0.1 dB
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        # 断言滤波器在阻带端的衰减不超过 -rs + 0.1 dB
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)
        
        # 断言计算得到的阶数 N 为 9
        assert_equal(N, 9)
        # 断言计算得到的归一化截止频率 Wn 的值接近于 103.4874609145164，相对误差小于等于 1e-15
        assert_allclose(Wn, 103.4874609145164, rtol=1e-15)

    def test_invalid_input(self):
        # 测试当 gpass 大于 gstop 时，是否抛出 ValueError 异常
        with pytest.raises(ValueError) as exc_info:
            cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 2)
        assert "gpass should be smaller than gstop" in str(exc_info.value)

        # 测试当 gpass 小于等于 0 时，是否抛出 ValueError 异常
        with pytest.raises(ValueError) as exc_info:
            cheb2ord([0.1, 0.6], [0.2, 0.5], -1, 2)
        assert "gpass should be larger than 0.0" in str(exc_info.value)

        # 测试当 gstop 小于等于 0 时，是否抛出 ValueError 异常
        with pytest.raises(ValueError) as exc_info:
            cheb2ord([0.1, 0.6], [0.2, 0.5], 1, -2)
        assert "gstop should be larger than 0.0" in str(exc_info.value)

    def test_ellip_cheb2(self):
        # 用于测试的目的是比较当前输出与过去 scipy 版本的已知输出值
        # 使用 Chebyshev Type II 方法计算滤波器的阶数 n 和归一化截止频率 wn
        n, wn = cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        # 断言计算得到的阶数 n 为 7
        assert n == 7

        # 使用 Chebyshev Type I 方法计算滤波器的阶数 n1 和归一化截止频率 w1
        n1, w1 = cheb1ord([0.1, 0.6], [0.2, 0.5], 3, 60)
        # 断言归一化截止频率 wn 与 w1 不完全相等
        assert not (wn == w1).all()

    def test_fs_validation(self):
        # 设定通带端频率为 0.2
        wp = 0.2
        # 设定阻带端频率为 0.3
        ws = 0.3
        # 设定通带最大衰减为 3 dB
        rp = 3
        # 设定阻带最小衰减为 60 dB
        rs = 60

        # 测试当采样频率为数组时，是否抛出 ValueError 异常
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            cheb2ord(wp, ws, rp, rs, False, fs=np.array([10, 20]))
class TestEllipord:

    def test_lowpass(self):
        # 设定通带边界频率和阻带边界频率
        wp = 0.2
        ws = 0.3
        # 设定通带最大衰减（dB）和阻带最小衰减（dB）
        rp = 3
        rs = 60
        # 计算椭圆滤波器的阶数 N 和归一化频率 Wn
        N, Wn = ellipord(wp, ws, rp, rs, False)
        # 根据 N、Wn、滤波器类型生成滤波器系数 b 和 a
        b, a = ellip(N, rp, rs, Wn, 'lp', False)
        # 计算滤波器的频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 断言通带内的衰减至少为 -rp - 0.1 dB
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))
        # 断言阻带内的衰减至少为 -rs + 0.1 dB
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)

        # 断言计算得到的滤波器阶数 N 符合预期值
        assert_equal(N, 5)
        # 断言计算得到的归一化频率 Wn 与预期值 0.2 接近
        assert_allclose(Wn, 0.2, rtol=1e-15)

    def test_lowpass_1000dB(self):
        # 失败时，ellipord 和 ellipap 中未使用 ellipkm1 函数
        wp = 0.2
        ws = 0.3
        rp = 3
        rs = 1000
        # 计算椭圆滤波器的阶数 N 和归一化频率 Wn
        N, Wn = ellipord(wp, ws, rp, rs, False)
        # 根据 N、Wn、滤波器类型生成二阶节级联形式的滤波器系数 sos
        sos = ellip(N, rp, rs, Wn, 'lp', False, output='sos')
        # 计算二阶节级联形式滤波器的频率响应
        w, h = sosfreqz(sos)
        w /= np.pi
        # 断言通带内的衰减至少为 -rp - 0.1 dB
        assert_array_less(-rp - 0.1, dB(h[w <= wp]))
        # 断言阻带内的衰减至少为 -rs + 0.1 dB
        assert_array_less(dB(h[ws <= w]), -rs + 0.1)

    def test_highpass(self):
        # 设定通带边界频率和阻带边界频率
        wp = 0.3
        ws = 0.2
        # 设定通带最大衰减（dB）和阻带最小衰减（dB）
        rp = 3
        rs = 70
        # 计算椭圆滤波器的阶数 N 和归一化频率 Wn
        N, Wn = ellipord(wp, ws, rp, rs, False)
        # 根据 N、Wn、滤波器类型生成滤波器系数 b 和 a
        b, a = ellip(N, rp, rs, Wn, 'hp', False)
        # 计算滤波器的频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 断言通带外的衰减至少为 -rp - 0.1 dB
        assert_array_less(-rp - 0.1, dB(h[wp <= w]))
        # 断言阻带内的衰减至少为 -rs + 0.1 dB
        assert_array_less(dB(h[w <= ws]), -rs + 0.1)

        # 断言计算得到的滤波器阶数 N 符合预期值
        assert_equal(N, 6)
        # 断言计算得到的归一化频率 Wn 与预期值 0.3 接近
        assert_allclose(Wn, 0.3, rtol=1e-15)

    def test_bandpass(self):
        # 设定通带边界频率和阻带边界频率
        wp = [0.2, 0.5]
        ws = [0.1, 0.6]
        # 设定通带最大衰减（dB）和阻带最小衰减（dB）
        rp = 3
        rs = 80
        # 计算椭圆滤波器的阶数 N 和归一化频率 Wn
        N, Wn = ellipord(wp, ws, rp, rs, False)
        # 根据 N、Wn、滤波器类型生成滤波器系数 b 和 a
        b, a = ellip(N, rp, rs, Wn, 'bp', False)
        # 计算滤波器的频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 断言通带内的衰减至少为 -rp - 0.1 dB
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_and(wp[0] <= w, w <= wp[1])]))
        # 断言阻带内的衰减至少为 -rs + 0.1 dB
        assert_array_less(dB(h[np.logical_or(w <= ws[0], ws[1] <= w)]),
                          -rs + 0.1)

        # 断言计算得到的滤波器阶数 N 符合预期值
        assert_equal(N, 6)
        # 断言计算得到的归一化频率 Wn 与预期值 [0.2, 0.5] 接近
        assert_allclose(Wn, [0.2, 0.5], rtol=1e-15)

    def test_bandstop(self):
        # 设定通带边界频率和阻带边界频率
        wp = [0.1, 0.6]
        ws = [0.2, 0.5]
        # 设定通带最大衰减（dB）和阻带最小衰减（dB）
        rp = 3
        rs = 90
        # 计算椭圆滤波器的阶数 N 和归一化频率 Wn
        N, Wn = ellipord(wp, ws, rp, rs, False)
        # 根据 N、Wn、滤波器类型生成滤波器系数 b 和 a
        b, a = ellip(N, rp, rs, Wn, 'bs', False)
        # 计算滤波器的频率响应
        w, h = freqz(b, a)
        w /= np.pi
        # 断言通带外的衰减至少为 -rp - 0.1 dB
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        # 断言阻带内的衰减至少为 -rs + 0.1 dB
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]),
                          -rs + 0.1)

        # 断言计算得到的滤波器阶数 N 符合预期值
        assert_equal(N, 7)
        # 断言计算得到的归一化频率 Wn 与预期值 [0.14758232794342988, 0.6] 接近
        assert_allclose(Wn, [0.14758232794342988, 0.6], rtol=1e-5)

    def test_analog(self):
        # 设定通带边界频率和阻带边界频率
        wp = [1000, 6000]
        ws
    def test_fs_param(self):
        # 设定通带边界频率和阻带边界频率
        wp = [400, 2400]
        ws = [800, 2000]
        # 设定通带最大衰减和阻带最小衰减
        rp = 3
        rs = 90
        # 设定采样频率
        fs = 8000
        # 计算椭圆滤波器的阶数 N 和归一化频率 Wn
        N, Wn = ellipord(wp, ws, rp, rs, False, fs=fs)
        # 根据 N, Wn 设计椭圆滤波器，返回分子 b 和分母 a
        b, a = ellip(N, rp, rs, Wn, 'bs', False, fs=fs)
        # 计算频率响应 w, h
        w, h = freqz(b, a, fs=fs)
        # 断言通带内衰减不低于 -rp - 0.1 dB
        assert_array_less(-rp - 0.1,
                          dB(h[np.logical_or(w <= wp[0], wp[1] <= w)]))
        # 断言阻带内衰减不高于 -rs + 0.1 dB
        assert_array_less(dB(h[np.logical_and(ws[0] <= w, w <= ws[1])]),
                          -rs + 0.1)

        # 断言计算得到的阶数 N 为 7
        assert_equal(N, 7)
        # 断言归一化频率 Wn 与预期值接近
        assert_allclose(Wn, [590.3293117737195, 2400], rtol=1e-5)

    def test_invalid_input(self):
        # 测试异常情况：通带和阻带交换，应该引发 ValueError
        with pytest.raises(ValueError) as exc_info:
            ellipord(0.2, 0.5, 3, 2)
        assert "gpass should be smaller than gstop" in str(exc_info.value)

        # 测试异常情况：通带最大衰减为负数，应该引发 ValueError
        with pytest.raises(ValueError) as exc_info:
            ellipord(0.2, 0.5, -1, 2)
        assert "gpass should be larger than 0.0" in str(exc_info.value)

        # 测试异常情况：阻带最小衰减为负数，应该引发 ValueError
        with pytest.raises(ValueError) as exc_info:
            ellipord(0.2, 0.5, 1, -2)
        assert "gstop should be larger than 0.0" in str(exc_info.value)

    def test_ellip_butter(self):
        # 测试椭圆和 Butterworth 滤波器的比较，确认与早期 scipy 版本的输出一致性
        n, wn = ellipord([0.1, 0.6], [0.2, 0.5], 3, 60)
        # 断言计算得到的阶数 n 为 5
        assert n == 5

    def test_fs_validation(self):
        # 设定通带边界频率和阻带边界频率为单一标量
        wp = 0.2
        ws = 0.3
        # 设定通带最大衰减和阻带最小衰减
        rp = 3
        rs = 60

        # 测试异常情况：采样频率以数组形式给出，应该引发 ValueError
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            ellipord(wp, ws, rp, rs, False, fs=np.array([10, 20]))
class TestBessel:

    def test_degenerate(self):
        # 遍历 'delay', 'phase', 'mag' 三种归一化方式
        for norm in ('delay', 'phase', 'mag'):
            # 0阶滤波器仅为直通滤波器
            b, a = bessel(0, 1, analog=True, norm=norm)
            assert_array_equal(b, [1])
            assert_array_equal(a, [1])

            # 1阶滤波器在所有类型下相同
            b, a = bessel(1, 1, analog=True, norm=norm)
            assert_allclose(b, [1], rtol=1e-15)
            assert_allclose(a, [1, 1], rtol=1e-15)

            # 获取1阶滤波器的零点、极点和增益，输出格式为零极点增益（ZPK）
            z, p, k = bessel(1, 0.3, analog=True, output='zpk', norm=norm)
            assert_array_equal(z, [])
            assert_allclose(p, [-0.3], rtol=1e-14)
            assert_allclose(k, 0.3, rtol=1e-14)

    def test_norm_phase(self):
        # 测试不同阶数和频率，验证它们在w0处的相位正确性
        for N in (1, 2, 3, 4, 5, 51, 72):
            for w0 in (1, 100):
                # 获取归一化为相位的贝塞尔滤波器系数
                b, a = bessel(N, w0, analog=True, norm='phase')
                w = np.linspace(0, w0, 100)
                w, h = freqs(b, a, w)
                phase = np.unwrap(np.angle(h))
                assert_allclose(phase[[0, -1]], (0, -N*pi/4), rtol=1e-1)

    def test_norm_mag(self):
        # 测试不同阶数和频率，验证它们在w0处的幅度正确性
        for N in (1, 2, 3, 4, 5, 51, 72):
            for w0 in (1, 100):
                # 获取归一化为幅度的贝塞尔滤波器系数
                b, a = bessel(N, w0, analog=True, norm='mag')
                w = (0, w0)
                w, h = freqs(b, a, w)
                mag = abs(h)
                assert_allclose(mag, (1, 1/np.sqrt(2)))

    def test_norm_delay(self):
        # 测试不同阶数和频率，验证它们在直流（DC）处的延迟正确性
        for N in (1, 2, 3, 4, 5, 51, 72):
            for w0 in (1, 100):
                # 获取归一化为延迟的贝塞尔滤波器系数
                b, a = bessel(N, w0, analog=True, norm='delay')
                w = np.linspace(0, 10*w0, 1000)
                w, h = freqs(b, a, w)
                delay = -np.diff(np.unwrap(np.angle(h)))/np.diff(w)
                assert_allclose(delay[0], 1/w0, rtol=1e-4)
    # 定义测试函数，用于测试_norm_factor函数
    def test_norm_factor(self):
        # 预设一组mpmath_values，包含不同N值对应的参考结果
        mpmath_values = {
            1: 1, 2: 1.361654128716130520, 3: 1.755672368681210649,
            4: 2.113917674904215843, 5: 2.427410702152628137,
            6: 2.703395061202921876, 7: 2.951722147038722771,
            8: 3.179617237510651330, 9: 3.391693138911660101,
            10: 3.590980594569163482, 11: 3.779607416439620092,
            12: 3.959150821144285315, 13: 4.130825499383535980,
            14: 4.295593409533637564, 15: 4.454233021624377494,
            16: 4.607385465472647917, 17: 4.755586548961147727,
            18: 4.899289677284488007, 19: 5.038882681488207605,
            20: 5.174700441742707423, 21: 5.307034531360917274,
            22: 5.436140703250035999, 23: 5.562244783787878196,
            24: 5.685547371295963521, 25: 5.806227623775418541,
            50: 8.268963160013226298, 51: 8.352374541546012058,
        }
        # 遍历mpmath_values中的每个N值
        for N in mpmath_values:
            # 调用besselap函数获取返回值z, p, k
            z, p, k = besselap(N, 'delay')
            # 断言计算得到的_norm_factor(p, k)与mpmath_values[N]接近，相对误差小于1e-13
            assert_allclose(mpmath_values[N], _norm_factor(p, k), rtol=1e-13)

    # 定义测试函数，用于测试_bessel_poly函数
    def test_bessel_poly(self):
        # 断言计算得到的_bessel_poly(5)与预期结果相等
        assert_array_equal(_bessel_poly(5), [945, 945, 420, 105, 15, 1])
        # 断言计算得到的_bessel_poly(4, True)与预期结果相等
        assert_array_equal(_bessel_poly(4, True), [1, 10, 45, 105, 105])

    # 定义测试函数，用于测试_bessel_zeros函数
    def test_bessel_zeros(self):
        # 断言计算得到的_bessel_zeros(0)为空列表
        assert_array_equal(_bessel_zeros(0), [])

    # 定义测试函数，用于测试异常情况
    def test_invalid(self):
        # 断言调用besselap函数抛出ValueError异常，当参数为5和'nonsense'时
        assert_raises(ValueError, besselap, 5, 'nonsense')
        # 断言调用besselap函数抛出ValueError异常，当参数为-5时
        assert_raises(ValueError, besselap, -5)
        # 断言调用besselap函数抛出ValueError异常，当参数为3.2时
        assert_raises(ValueError, besselap, 3.2)
        # 断言调用_bessel_poly函数抛出ValueError异常，当参数为-3时
        assert_raises(ValueError, _bessel_poly, -3)
        # 断言调用_bessel_poly函数抛出ValueError异常，当参数为3.3时
        assert_raises(ValueError, _bessel_poly, 3.3)

    # 标记测试为fail_slow，设置最大重试次数为10
    @pytest.mark.fail_slow(10)
    # 定义测试函数，用于测试fs_param参数
    def test_fs_param(self):
        # 遍历norm和fs的组合
        for norm in ('phase', 'mag', 'delay'):
            # 遍历不同fs值
            for fs in (900, 900.1, 1234.567):
                # 遍历不同N值
                for N in (0, 1, 2, 3, 10):
                    # 遍历不同fc值
                    for fc in (100, 100.1, 432.12345):
                        # 遍历不同btype值
                        for btype in ('lp', 'hp'):
                            # 调用bessel函数计算ba1
                            ba1 = bessel(N, fc, btype, norm=norm, fs=fs)
                            # 调用bessel函数计算ba2，将fc转换为相对于fs的归一化频率
                            ba2 = bessel(N, fc/(fs/2), btype, norm=norm)
                            # 断言ba1与ba2的所有元素接近
                            assert_allclose(ba1, ba2)
                    # 遍历不同(fc1, fc2)元组值
                    for fc in ((100, 200), (100.1, 200.2), (321.123, 432.123)):
                        # 遍历不同btype值
                        for btype in ('bp', 'bs'):
                            # 调用bessel函数计算ba1
                            ba1 = bessel(N, fc, btype, norm=norm, fs=fs)
                            # 将(fc1, fc2)转换为相对于fs的归一化频率序列
                            fcnorm = [f/(fs/2) for f in fc]
                            # 调用bessel函数计算ba2
                            ba2 = bessel(N, fcnorm, btype, norm=norm)
                            # 断言ba1与ba2的所有元素接近
                            assert_allclose(ba1, ba2)
class TestButter:

    def test_degenerate(self):
        # 0-order filter is just a passthrough
        b, a = butter(0, 1, analog=True)
        assert_array_equal(b, [1])
        assert_array_equal(a, [1])

        # 1-order filter is same for all types
        b, a = butter(1, 1, analog=True)
        assert_array_almost_equal(b, [1])
        assert_array_almost_equal(a, [1, 1])

        # Calculate zeros, poles, and gain for a 1st order Butterworth filter
        z, p, k = butter(1, 0.3, output='zpk')
        assert_array_equal(z, [-1])
        assert_allclose(p, [3.249196962329063e-01], rtol=1e-14)
        assert_allclose(k, 3.375401518835469e-01, rtol=1e-14)

    def test_basic(self):
        # analog s-plane
        # Test Butterworth filters for various orders (N) in the analog domain
        for N in range(25):
            wn = 0.01
            # Compute zeros, poles, and gain for low-pass Butterworth filter in analog domain
            z, p, k = butter(N, wn, 'low', analog=True, output='zpk')
            assert_array_almost_equal([], z)
            assert_(len(p) == N)
            # All poles should be at distance wn from origin
            assert_array_almost_equal(wn, abs(p))
            assert_(all(np.real(p) <= 0))  # No poles in right half of S-plane
            assert_array_almost_equal(wn**N, k)

        # digital z-plane
        # Test Butterworth filters for various orders (N) in the digital domain
        for N in range(25):
            wn = 0.01
            # Compute zeros, poles, and gain for high-pass Butterworth filter in digital domain
            z, p, k = butter(N, wn, 'high', analog=False, output='zpk')
            assert_array_equal(np.ones(N), z)  # All zeros exactly at DC
            assert_(all(np.abs(p) <= 1))  # No poles outside unit circle

        # Test predefined Butterworth filters in the analog domain
        b1, a1 = butter(2, 1, analog=True)
        assert_array_almost_equal(b1, [1])
        assert_array_almost_equal(a1, [1, np.sqrt(2), 1])

        b2, a2 = butter(5, 1, analog=True)
        assert_array_almost_equal(b2, [1])
        assert_array_almost_equal(a2, [1, 3.2361, 5.2361,
                                       5.2361, 3.2361, 1], decimal=4)

        b3, a3 = butter(10, 1, analog=True)
        assert_array_almost_equal(b3, [1])
        assert_array_almost_equal(a3, [1, 6.3925, 20.4317, 42.8021, 64.8824,
                                       74.2334, 64.8824, 42.8021, 20.4317,
                                       6.3925, 1], decimal=4)

        # Test a specific Butterworth filter in the analog domain
        b2, a2 = butter(19, 1.0441379169150726, analog=True)
        assert_array_almost_equal(b2, [2.2720], decimal=4)
        assert_array_almost_equal(a2, 1.0e+004 * np.array([
                        0.0001, 0.0013, 0.0080, 0.0335, 0.1045, 0.2570,
                        0.5164, 0.8669, 1.2338, 1.5010, 1.5672, 1.4044,
                        1.0759, 0.6986, 0.3791, 0.1681, 0.0588, 0.0153,
                        0.0026, 0.0002]), decimal=0)

        # Test Butterworth filters in the digital domain
        b, a = butter(5, 0.4)
        assert_array_almost_equal(b, [0.0219, 0.1097, 0.2194,
                                      0.2194, 0.1097, 0.0219], decimal=4)
        assert_array_almost_equal(a, [1.0000, -0.9853, 0.9738,
                                      -0.3864, 0.1112, -0.0113], decimal=4)
    # 定义测试方法 test_bandpass(self)，用于测试带通滤波器设计函数的输出
    def test_bandpass(self):
        # 调用 butter 函数设计一个带通滤波器，并返回其零点、极点和增益
        z, p, k = butter(8, [0.25, 0.33], 'band', output='zpk')
        # 期望的零点序列
        z2 = [1, 1, 1, 1, 1, 1, 1, 1,
              -1, -1, -1, -1, -1, -1, -1, -1]
        # 期望的极点序列
        p2 = [
            4.979909925436156e-01 + 8.367609424799387e-01j,
            4.979909925436156e-01 - 8.367609424799387e-01j,
            4.913338722555539e-01 + 7.866774509868817e-01j,
            4.913338722555539e-01 - 7.866774509868817e-01j,
            5.035229361778706e-01 + 7.401147376726750e-01j,
            5.035229361778706e-01 - 7.401147376726750e-01j,
            5.307617160406101e-01 + 7.029184459442954e-01j,
            5.307617160406101e-01 - 7.029184459442954e-01j,
            5.680556159453138e-01 + 6.788228792952775e-01j,
            5.680556159453138e-01 - 6.788228792952775e-01j,
            6.100962560818854e-01 + 6.693849403338664e-01j,
            6.100962560818854e-01 - 6.693849403338664e-01j,
            6.904694312740631e-01 + 6.930501690145245e-01j,
            6.904694312740631e-01 - 6.930501690145245e-01j,
            6.521767004237027e-01 + 6.744414640183752e-01j,
            6.521767004237027e-01 - 6.744414640183752e-01j,
            ]
        # 期望的增益值
        k2 = 3.398854055800844e-08
        # 断言实际输出的零点序列与期望相等
        assert_array_equal(z, z2)
        # 断言实际输出的极点序列与期望相等（按虚部排序）
        assert_allclose(sorted(p, key=np.imag),
                        sorted(p2, key=np.imag), rtol=1e-13)
        # 断言实际输出的增益值与期望相等
        assert_allclose(k, k2, rtol=1e-13)

        # bandpass 模拟
        # 调用 butter 函数设计一个模拟带通滤波器，并返回其零点、极点和增益
        z, p, k = butter(4, [90.5, 110.5], 'bp', analog=True, output='zpk')
        # 期望的零点序列全为零
        z2 = np.zeros(4)
        # 期望的极点序列
        p2 = [
            -4.179137760733086e+00 + 1.095935899082837e+02j,
            -4.179137760733086e+00 - 1.095935899082837e+02j,
            -9.593598668443835e+00 + 1.034745398029734e+02j,
            -9.593598668443835e+00 - 1.034745398029734e+02j,
            -8.883991981781929e+00 + 9.582087115567160e+01j,
            -8.883991981781929e+00 - 9.582087115567160e+01j,
            -3.474530886568715e+00 + 9.111599925805801e+01j,
            -3.474530886568715e+00 - 9.111599925805801e+01j,
            ]
        # 期望的增益值
        k2 = 1.600000000000001e+05
        # 断言实际输出的零点序列与期望相等
        assert_array_equal(z, z2)
        # 断言实际输出的极点序列与期望相等（按虚部排序）
        assert_allclose(sorted(p, key=np.imag), sorted(p2, key=np.imag))
        # 断言实际输出的增益值与期望相等
        assert_allclose(k, k2, rtol=1e-15)
    # 定义一个测试方法，用于测试带阻滤波器的设计
    def test_bandstop(self):
        # 使用 butter 函数设计一个阶数为 7 的带阻滤波器，指定频率范围为 [0.45, 0.56]
        # 输出 z, p, k 是该滤波器的零点、极点和增益
        z, p, k = butter(7, [0.45, 0.56], 'stop', output='zpk')
        
        # 预期的零点列表 z2，其中包含了带阻滤波器的零点值
        z2 = [-1.594474531383421e-02 + 9.998728744679880e-01j,
              -1.594474531383421e-02 - 9.998728744679880e-01j,
              -1.594474531383421e-02 + 9.998728744679880e-01j,
              -1.594474531383421e-02 - 9.998728744679880e-01j,
              -1.594474531383421e-02 + 9.998728744679880e-01j,
              -1.594474531383421e-02 - 9.998728744679880e-01j,
              -1.594474531383421e-02 + 9.998728744679880e-01j,
              -1.594474531383421e-02 - 9.998728744679880e-01j,
              -1.594474531383421e-02 + 9.998728744679880e-01j,
              -1.594474531383421e-02 - 9.998728744679880e-01j,
              -1.594474531383421e-02 + 9.998728744679880e-01j,
              -1.594474531383421e-02 - 9.998728744679880e-01j,
              -1.594474531383421e-02 + 9.998728744679880e-01j,
              -1.594474531383421e-02 - 9.998728744679880e-01j]
        
        # 预期的极点列表 p2，其中包含了带阻滤波器的极点值
        p2 = [-1.766850742887729e-01 + 9.466951258673900e-01j,
              -1.766850742887729e-01 - 9.466951258673900e-01j,
               1.467897662432886e-01 + 9.515917126462422e-01j,
               1.467897662432886e-01 - 9.515917126462422e-01j,
              -1.370083529426906e-01 + 8.880376681273993e-01j,
              -1.370083529426906e-01 - 8.880376681273993e-01j,
               1.086774544701390e-01 + 8.915240810704319e-01j,
               1.086774544701390e-01 - 8.915240810704319e-01j,
              -7.982704457700891e-02 + 8.506056315273435e-01j,
              -7.982704457700891e-02 - 8.506056315273435e-01j,
               5.238812787110331e-02 + 8.524011102699969e-01j,
               5.238812787110331e-02 - 8.524011102699969e-01j,
              -1.357545000491310e-02 + 8.382287744986582e-01j,
              -1.357545000491310e-02 - 8.382287744986582e-01j]
        
        # 预期的增益 k2，为带阻滤波器的增益值
        k2 = 4.577122512960063e-01
        
        # 使用 assert_allclose 检查计算得到的零点 z 是否与预期值 z2 排序后相似
        assert_allclose(sorted(z, key=np.imag), sorted(z2, key=np.imag))
        
        # 使用 assert_allclose 检查计算得到的极点 p 是否与预期值 p2 排序后相似
        assert_allclose(sorted(p, key=np.imag), sorted(p2, key=np.imag))
        
        # 使用 assert_allclose 检查计算得到的增益 k 是否与预期值 k2 相似，相对容差设为 1e-14
        assert_allclose(k, k2, rtol=1e-14)

    # 定义一个测试方法，用于测试巴特沃斯滤波器的 b 和 a 系数输出
    def test_ba_output(self):
        # 使用 butter 函数设计一个阶数为 4 的带通滤波器，指定频率范围为 [100, 300]，模拟模式为 True
        b, a = butter(4, [100, 300], 'bandpass', analog=True)
        
        # 预期的 b 系数列表 b2，其中包含了带通滤波器的 b 系数
        b2 = [1.6e+09, 0, 0, 0, 0]
        
        # 预期的 a 系数列表 a2，其中包含了带通滤波器的 a 系数
        a2 = [1.000000000000000e+00, 5.226251859505511e+02,
              2.565685424949238e+05, 6.794127417357160e+07,
              1.519411254969542e+10, 2.038238225207147e+12,
              2.309116882454312e+14, 1.411088002066486e+16,
              8.099999999999991e+17]
        
        # 使用 assert_allclose 检查计算得到的 b 系数 b 是否与预期值 b2 相似，相对容差设为 1e-14
        assert_allclose(b, b2, rtol=1e-14)
        
        # 使用 assert_allclose 检查计算得到的 a 系数 a 是否与预期值 a2 相似，相对容差设为 1e-14
        assert_allclose(a, a2, rtol=1e-14)
    # 定义一个测试方法，用于测试 butter 函数的参数组合
    def test_fs_param(self):
        # 循环遍历不同的采样频率 fs
        for fs in (900, 900.1, 1234.567):
            # 循环遍历不同的滤波器阶数 N
            for N in (0, 1, 2, 3, 10):
                # 循环遍历不同的截止频率 fc
                for fc in (100, 100.1, 432.12345):
                    # 循环遍历不同的滤波器类型 btype（低通 'lp' 或高通 'hp'）
                    for btype in ('lp', 'hp'):
                        # 调用 butter 函数，生成滤波器系数 ba1
                        ba1 = butter(N, fc, btype, fs=fs)
                        # 使用 fc/(fs/2) 计算归一化频率后，再调用 butter 函数，生成滤波器系数 ba2
                        ba2 = butter(N, fc/(fs/2), btype)
                        # 使用 assert_allclose 函数断言 ba1 和 ba2 在数值上的接近
                        assert_allclose(ba1, ba2)
                
                # 继续循环遍历不同的带通和带阻滤波器的截止频率组合
                for fc in ((100, 200), (100.1, 200.2), (321.123, 432.123)):
                    # 循环遍历不同的滤波器类型 btype（带通 'bp' 或带阻 'bs'）
                    for btype in ('bp', 'bs'):
                        # 调用 butter 函数，生成滤波器系数 ba1
                        ba1 = butter(N, fc, btype, fs=fs)
                        # 使用列表、元组或数组的形式，计算归一化频率后，再调用 butter 函数，生成滤波器系数 ba2
                        for seq in (list, tuple, array):
                            fcnorm = seq([f/(fs/2) for f in fc])
                            ba2 = butter(N, fcnorm, btype)
                            # 使用 assert_allclose 函数断言 ba1 和 ba2 在数值上的接近
                            assert_allclose(ba1, ba2)
class TestCheby1:
    
    def test_degenerate(self):
        # 0-order filter is just a passthrough
        # Even-order filters have DC gain of -rp dB
        # 调用 cheby1 函数生成零阶滤波器，模拟模式下，增益为 -rp dB
        b, a = cheby1(0, 10*np.log10(2), 1, analog=True)
        # 断言 b 数组近似等于 [1/√2]
        assert_array_almost_equal(b, [1/np.sqrt(2)])
        # 断言 a 数组等于 [1]
        assert_array_equal(a, [1])

        # 1-order filter is same for all types
        # 调用 cheby1 函数生成一阶滤波器，模拟模式下
        b, a = cheby1(1, 10*np.log10(2), 1, analog=True)
        # 断言 b 数组近似等于 [1]
        assert_array_almost_equal(b, [1])
        # 断言 a 数组近似等于 [1, 1]
        assert_array_almost_equal(a, [1, 1])

        # 调用 cheby1 函数生成一阶带通滤波器，返回零极点增益信息
        z, p, k = cheby1(1, 0.1, 0.3, output='zpk')
        # 断言 z 数组等于 [-1]
        assert_array_equal(z, [-1])
        # 断言 p 数组各项接近 [-5.390126972799615e-01]
        assert_allclose(p, [-5.390126972799615e-01], rtol=1e-14)
        # 断言 k 值接近 7.695063486399808e-01
        assert_allclose(k, 7.695063486399808e-01, rtol=1e-14)

    def test_bandpass(self):
        # 调用 cheby1 函数生成八阶带通滤波器，指定频带范围和类型，返回零极点增益信息
        z, p, k = cheby1(8, 1, [0.3, 0.4], 'bp', output='zpk')
        # 预期的零点数组
        z2 = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
        # 预期的极点数组
        p2 = [3.077784854851463e-01 + 9.453307017592942e-01j,
              3.077784854851463e-01 - 9.453307017592942e-01j,
              3.280567400654425e-01 + 9.272377218689016e-01j,
              3.280567400654425e-01 - 9.272377218689016e-01j,
              3.677912763284301e-01 + 9.038008865279966e-01j,
              3.677912763284301e-01 - 9.038008865279966e-01j,
              4.194425632520948e-01 + 8.769407159656157e-01j,
              4.194425632520948e-01 - 8.769407159656157e-01j,
              4.740921994669189e-01 + 8.496508528630974e-01j,
              4.740921994669189e-01 - 8.496508528630974e-01j,
              5.234866481897429e-01 + 8.259608422808477e-01j,
              5.234866481897429e-01 - 8.259608422808477e-01j,
              5.844717632289875e-01 + 8.052901363500210e-01j,
              5.844717632289875e-01 - 8.052901363500210e-01j,
              5.615189063336070e-01 + 8.100667803850766e-01j,
              5.615189063336070e-01 - 8.100667803850766e-01j]
        # 预期的增益值
        k2 = 5.007028718074307e-09
        # 断言 z 数组等于预期的 z2
        assert_array_equal(z, z2)
        # 断言 p 数组按虚部排序后与预期的 p2 接近
        assert_allclose(sorted(p, key=np.imag),
                        sorted(p2, key=np.imag), rtol=1e-13)
        # 断言 k 值接近预期的 k2
        assert_allclose(k, k2, rtol=1e-13)
    # 定义一个名为 test_bandstop 的测试方法，使用 self 参数表示它是一个类方法（针对测试类的实例）
    def test_bandstop(self):
        # 调用 cheby1 函数，返回 Chebyshev I 滤波器的零点、极点和增益
        z, p, k = cheby1(7, 1, [0.5, 0.6], 'stop', output='zpk')
        
        # 指定预期的零点 z2，其中包含了多个复数值的列表
        z2 = [-1.583844403245361e-01 + 9.873775210440450e-01j,
              -1.583844403245361e-01 - 9.873775210440450e-01j,
              -1.583844403245361e-01 + 9.873775210440450e-01j,
              -1.583844403245361e-01 - 9.873775210440450e-01j,
              -1.583844403245361e-01 + 9.873775210440450e-01j,
              -1.583844403245361e-01 - 9.873775210440450e-01j,
              -1.583844403245361e-01 + 9.873775210440450e-01j,
              -1.583844403245361e-01 - 9.873775210440450e-01j,
              -1.583844403245361e-01 + 9.873775210440450e-01j,
              -1.583844403245361e-01 - 9.873775210440450e-01j,
              -1.583844403245361e-01 + 9.873775210440450e-01j,
              -1.583844403245361e-01 - 9.873775210440450e-01j,
              -1.583844403245361e-01 + 9.873775210440450e-01j,
              -1.583844403245361e-01 - 9.873775210440450e-01j]
        
        # 指定预期的极点 p2，其中包含了多个复数值的列表
        p2 = [-8.942974551472813e-02 + 3.482480481185926e-01j,
              -8.942974551472813e-02 - 3.482480481185926e-01j,
               1.293775154041798e-01 + 8.753499858081858e-01j,
               1.293775154041798e-01 - 8.753499858081858e-01j,
               3.399741945062013e-02 + 9.690316022705607e-01j,
               3.399741945062013e-02 - 9.690316022705607e-01j,
               4.167225522796539e-04 + 9.927338161087488e-01j,
               4.167225522796539e-04 - 9.927338161087488e-01j,
              -3.912966549550960e-01 + 8.046122859255742e-01j,
              -3.912966549550960e-01 - 8.046122859255742e-01j,
              -3.307805547127368e-01 + 9.133455018206508e-01j,
              -3.307805547127368e-01 - 9.133455018206508e-01j,
              -3.072658345097743e-01 + 9.443589759799366e-01j,
              -3.072658345097743e-01 - 9.443589759799366e-01j]
        
        # 指定预期的增益 k2
        k2 = 3.619438310405028e-01
        
        # 使用 assert_allclose 函数比较 z 和 z2 的排序后的值，使用 np.imag 作为排序关键字，rtol 设置相对误差容忍度
        assert_allclose(sorted(z, key=np.imag),
                        sorted(z2, key=np.imag), rtol=1e-13)
        
        # 使用 assert_allclose 函数比较 p 和 p2 的排序后的值，使用 np.imag 作为排序关键字，rtol 设置相对误差容忍度
        assert_allclose(sorted(p, key=np.imag),
                        sorted(p2, key=np.imag), rtol=1e-13)
        
        # 使用 assert_allclose 函数比较 k 和 k2 的值，rtol 设置相对误差容忍度，atol 设置绝对误差容忍度
        assert_allclose(k, k2, rtol=0, atol=5e-16)
    # 定义一个测试方法，用于验证 cheby1 函数的输出是否正确
    def test_ba_output(self):
        # 使用 cheby1 函数生成 Chebyshev Type I 滤波器的系数 b 和 a
        # 在模拟域（analog=True）下，生成一个阻带滤波器
        b, a = cheby1(5, 0.9, [210, 310], 'stop', analog=True)
        
        # 预期的系数 b2，这里列出了其逐项的值
        b2 = [1.000000000000006e+00, 0,
              3.255000000000020e+05, 0,
              4.238010000000026e+10, 0,
              2.758944510000017e+15, 0,
              8.980364380050052e+19, 0,
              1.169243442282517e+24
              ]
        
        # 预期的系数 a2，这里列出了其逐项的值
        a2 = [1.000000000000000e+00, 4.630555945694342e+02,
              4.039266454794788e+05, 1.338060988610237e+08,
              5.844333551294591e+10, 1.357346371637638e+13,
              3.804661141892782e+15, 5.670715850340080e+17,
              1.114411200988328e+20, 8.316815934908471e+21,
              1.169243442282517e+24
              ]
        
        # 使用 assert_allclose 函数断言实际计算出的 b 与预期的 b2 非常接近
        assert_allclose(b, b2, rtol=1e-14)
        
        # 使用 assert_allclose 函数断言实际计算出的 a 与预期的 a2 非常接近
        assert_allclose(a, a2, rtol=1e-14)

    # 定义一个测试方法，用于验证 cheby1 函数在不同参数下的一致性
    def test_fs_param(self):
        # 分别测试不同的采样频率 fs
        for fs in (900, 900.1, 1234.567):
            # 分别测试不同的阶数 N
            for N in (0, 1, 2, 3, 10):
                # 分别测试不同的截止频率 fc
                for fc in (100, 100.1, 432.12345):
                    # 分别测试不同的滤波器类型 btype
                    for btype in ('lp', 'hp'):
                        # 使用 cheby1 函数计算得到滤波器系数 ba1
                        ba1 = cheby1(N, 1, fc, btype, fs=fs)
                        # 使用 cheby1 函数计算得到归一化频率下的滤波器系数 ba2
                        ba2 = cheby1(N, 1, fc/(fs/2), btype)
                        # 使用 assert_allclose 函数断言 ba1 与 ba2 非常接近
                        assert_allclose(ba1, ba2)
                
                # 分别测试不同的带通频率范围 fc
                for fc in ((100, 200), (100.1, 200.2), (321.123, 432.123)):
                    # 分别测试不同的带通滤波器类型 btype
                    for btype in ('bp', 'bs'):
                        # 使用 cheby1 函数计算得到滤波器系数 ba1
                        ba1 = cheby1(N, 1, fc, btype, fs=fs)
                        # 将带通频率范围归一化，使用 cheby1 函数计算得到 ba2
                        for seq in (list, tuple, array):
                            fcnorm = seq([f/(fs/2) for f in fc])
                            ba2 = cheby1(N, 1, fcnorm, btype)
                            # 使用 assert_allclose 函数断言 ba1 与 ba2 非常接近
                            assert_allclose(ba1, ba2)
class TestCheby2:

    def test_degenerate(self):
        # 0-order filter is just a passthrough
        # 声明一个零阶滤波器，仅仅是一个传递函数
        # Stopband ripple factor doesn't matter
        # 阻带波动因子不影响结果
        b, a = cheby2(0, 123.456, 1, analog=True)
        assert_array_equal(b, [1])
        assert_array_equal(a, [1])

        # 1-order filter is same for all types
        # 一阶滤波器对所有类型来说都是相同的
        b, a = cheby2(1, 10*np.log10(2), 1, analog=True)
        assert_array_almost_equal(b, [1])
        assert_array_almost_equal(a, [1, 1])

        # Compute zeros, poles, and gain for a 1st-order Chebyshev type II filter
        # 计算第一阶切比雪夫II型滤波器的零点、极点和增益
        z, p, k = cheby2(1, 50, 0.3, output='zpk')
        assert_array_equal(z, [-1])
        assert_allclose(p, [9.967826460175649e-01], rtol=1e-14)
        assert_allclose(k, 1.608676991217512e-03, rtol=1e-14)

    def test_basic(self):
        # Test low-pass Chebyshev type II filters
        # 测试低通切比雪夫II型滤波器
        for N in range(25):
            wn = 0.01
            z, p, k = cheby2(N, 40, wn, 'low', analog=True, output='zpk')
            assert_(len(p) == N)
            assert_(all(np.real(p) <= 0))  # No poles in right half of S-plane
            # 在S平面的右半部分没有极点

        # Test high-pass Chebyshev type II filters
        # 测试高通切比雪夫II型滤波器
        for N in range(25):
            wn = 0.01
            z, p, k = cheby2(N, 40, wn, 'high', analog=False, output='zpk')
            assert_(all(np.abs(p) <= 1))  # No poles outside unit circle
            # 单位圆外没有极点

        # Compute coefficients for a band-stop Chebyshev type II filter
        # 计算带阻切比雪夫II型滤波器的系数
        B, A = cheby2(18, 100, 0.5)
        assert_array_almost_equal(B, [
            0.00167583914216, 0.01249479541868, 0.05282702120282,
            0.15939804265706, 0.37690207631117, 0.73227013789108,
            1.20191856962356, 1.69522872823393, 2.07598674519837,
            2.21972389625291, 2.07598674519838, 1.69522872823395,
            1.20191856962359, 0.73227013789110, 0.37690207631118,
            0.15939804265707, 0.05282702120282, 0.01249479541868,
            0.00167583914216], decimal=13)
        assert_array_almost_equal(A, [
            1.00000000000000, -0.27631970006174, 3.19751214254060,
            -0.15685969461355, 4.13926117356269, 0.60689917820044,
            2.95082770636540, 0.89016501910416, 1.32135245849798,
            0.51502467236824, 0.38906643866660, 0.15367372690642,
            0.07255803834919, 0.02422454070134, 0.00756108751837,
            0.00179848550988, 0.00033713574499, 0.00004258794833,
            0.00000281030149], decimal=13)
    # 定义一个测试函数，用于测试 cheby2 函数的带通滤波器功能
    def test_bandpass(self):
        # 调用 cheby2 函数生成一个带通滤波器的零点 z、极点 p 和增益 k
        z, p, k = cheby2(9, 40, [0.07, 0.2], 'pass', output='zpk')
        
        # 预期的零点列表 z2
        z2 = [-9.999999999999999e-01,
               3.676588029658514e-01 + 9.299607543341383e-01j,
               3.676588029658514e-01 - 9.299607543341383e-01j,
               7.009689684982283e-01 + 7.131917730894889e-01j,
               7.009689684982283e-01 - 7.131917730894889e-01j,
               7.815697973765858e-01 + 6.238178033919218e-01j,
               7.815697973765858e-01 - 6.238178033919218e-01j,
               8.063793628819866e-01 + 5.913986160941200e-01j,
               8.063793628819866e-01 - 5.913986160941200e-01j,
               1.000000000000001e+00,
               9.944493019920448e-01 + 1.052168511576739e-01j,
               9.944493019920448e-01 - 1.052168511576739e-01j,
               9.854674703367308e-01 + 1.698642543566085e-01j,
               9.854674703367308e-01 - 1.698642543566085e-01j,
               9.762751735919308e-01 + 2.165335665157851e-01j,
               9.762751735919308e-01 - 2.165335665157851e-01j,
               9.792277171575134e-01 + 2.027636011479496e-01j,
               9.792277171575134e-01 - 2.027636011479496e-01j]
        
        # 预期的极点列表 p2
        p2 = [8.143803410489621e-01 + 5.411056063397541e-01j,
              8.143803410489621e-01 - 5.411056063397541e-01j,
              7.650769827887418e-01 + 5.195412242095543e-01j,
              7.650769827887418e-01 - 5.195412242095543e-01j,
              6.096241204063443e-01 + 3.568440484659796e-01j,
              6.096241204063443e-01 - 3.568440484659796e-01j,
              6.918192770246239e-01 + 4.770463577106911e-01j,
              6.918192770246239e-01 - 4.770463577106911e-01j,
              6.986241085779207e-01 + 1.146512226180060e-01j,
              6.986241085779207e-01 - 1.146512226180060e-01j,
              8.654645923909734e-01 + 1.604208797063147e-01j,
              8.654645923909734e-01 - 1.604208797063147e-01j,
              9.164831670444591e-01 + 1.969181049384918e-01j,
              9.164831670444591e-01 - 1.969181049384918e-01j,
              9.630425777594550e-01 + 2.317513360702271e-01j,
              9.630425777594550e-01 - 2.317513360702271e-01j,
              9.438104703725529e-01 + 2.193509900269860e-01j,
              9.438104703725529e-01 - 2.193509900269860e-01j]
        
        # 预期的增益 k2
        k2 = 9.345352824659604e-03
        
        # 使用 numpy 的 assert_allclose 函数，检查 z 和 z2 是否近似相等（按角度排序，相对误差容差为 1e-13）
        assert_allclose(sorted(z, key=np.angle),
                        sorted(z2, key=np.angle), rtol=1e-13)
        
        # 使用 numpy 的 assert_allclose 函数，检查 p 和 p2 是否近似相等（按角度排序，相对误差容差为 1e-13）
        assert_allclose(sorted(p, key=np.angle),
                        sorted(p2, key=np.angle), rtol=1e-13)
        
        # 使用 numpy 的 assert_allclose 函数，检查 k 和 k2 是否近似相等（相对误差容差为 1e-11）
        assert_allclose(k, k2, rtol=1e-11)
    def test_bandstop(self):
        # 使用 cheby2 函数生成 Chebyshev Type II 带阻滤波器的零点（z）、极点（p）和增益（k）
        z, p, k = cheby2(6, 55, [0.1, 0.9], 'stop', output='zpk')
        
        # 预期的零点列表
        z2 = [6.230544895101009e-01 + 7.821784343111114e-01j,
              6.230544895101009e-01 - 7.821784343111114e-01j,
              9.086608545660115e-01 + 4.175349702471991e-01j,
              9.086608545660115e-01 - 4.175349702471991e-01j,
              9.478129721465802e-01 + 3.188268649763867e-01j,
              9.478129721465802e-01 - 3.188268649763867e-01j,
              -6.230544895100982e-01 + 7.821784343111109e-01j,
              -6.230544895100982e-01 - 7.821784343111109e-01j,
              -9.086608545660116e-01 + 4.175349702472088e-01j,
              -9.086608545660116e-01 - 4.175349702472088e-01j,
              -9.478129721465784e-01 + 3.188268649763897e-01j,
              -9.478129721465784e-01 - 3.188268649763897e-01j]
        
        # 预期的极点列表
        p2 = [-9.464094036167638e-01 + 1.720048695084344e-01j,
              -9.464094036167638e-01 - 1.720048695084344e-01j,
              -8.715844103386737e-01 + 1.370665039509297e-01j,
              -8.715844103386737e-01 - 1.370665039509297e-01j,
              -8.078751204586425e-01 + 5.729329866682983e-02j,
              -8.078751204586425e-01 - 5.729329866682983e-02j,
               9.464094036167665e-01 + 1.720048695084332e-01j,
               9.464094036167665e-01 - 1.720048695084332e-01j,
               8.078751204586447e-01 + 5.729329866683007e-02j,
               8.078751204586447e-01 - 5.729329866683007e-02j,
               8.715844103386721e-01 + 1.370665039509331e-01j,
               8.715844103386721e-01 - 1.370665039509331e-01j]
        
        # 预期的增益
        k2 = 2.917823332763358e-03
        
        # 检查计算得到的零点 z 与预期 z2 是否接近（按角度排序）
        assert_allclose(sorted(z, key=np.angle),
                        sorted(z2, key=np.angle), rtol=1e-13)
        
        # 检查计算得到的极点 p 与预期 p2 是否接近（按角度排序）
        assert_allclose(sorted(p, key=np.angle),
                        sorted(p2, key=np.angle), rtol=1e-13)
        
        # 检查计算得到的增益 k 与预期 k2 是否接近
        assert_allclose(k, k2, rtol=1e-11)

    def test_ba_output(self):
        # 使用 cheby2 函数生成 Chebyshev Type II 带阻滤波器的系数 b 和 a
        b, a = cheby2(5, 20, [2010, 2100], 'stop', True)
        
        # 预期的系数 b 列表
        b2 = [1.000000000000000e+00, 0,  # Matlab: 6.683253076978249e-12,
              2.111512500000000e+07, 0,  # Matlab: 1.134325604589552e-04,
              1.782966433781250e+14, 0,  # Matlab: 7.216787944356781e+02,
              7.525901316990656e+20, 0,  # Matlab: 2.039829265789886e+09,
              1.587960565565748e+27, 0,  # Matlab: 2.161236218626134e+15,
              1.339913493808585e+33]
        
        # 预期的系数 a 列表
        a2 = [1.000000000000000e+00, 1.849550755473371e+02,
              2.113222918998538e+07, 3.125114149732283e+09,
              1.785133457155609e+14, 1.979158697776348e+16,
              7.535048322653831e+20, 5.567966191263037e+22,
              1.589246884221346e+27, 5.871210648525566e+28,
              1.339913493808590e+33]
        
        # 检查计算得到的系数 b 与预期 b2 是否接近
        assert_allclose(b, b2, rtol=1e-14)
        
        # 检查计算得到的系数 a 与预期 a2 是否接近
        assert_allclose(a, a2, rtol=1e-14)
    # 定义测试函数，用于测试 cheby2 函数的参数组合
    def test_fs_param(self):
        # 针对不同的采样率 fs 进行循环测试
        for fs in (900, 900.1, 1234.567):
            # 针对不同的阶数 N 进行循环测试
            for N in (0, 1, 2, 3, 10):
                # 针对不同的截止频率 fc 进行循环测试
                for fc in (100, 100.1, 432.12345):
                    # 针对滤波器类型 'lp'（低通）和 'hp'（高通）进行循环测试
                    for btype in ('lp', 'hp'):
                        # 使用 cheby2 函数计算巴特沃斯滤波器系数 ba1
                        ba1 = cheby2(N, 20, fc, btype, fs=fs)
                        # 使用 cheby2 函数计算标准化后的截止频率（fs/2）的比例，再计算 ba2
                        ba2 = cheby2(N, 20, fc/(fs/2), btype)
                        # 断言 ba1 和 ba2 在数值上接近
                        assert_allclose(ba1, ba2)
                
                # 针对带通和带阻滤波器类型 'bp'（带通）和 'bs'（带阻）进行循环测试
                for fc in ((100, 200), (100.1, 200.2), (321.123, 432.123)):
                    for btype in ('bp', 'bs'):
                        # 使用 cheby2 函数计算巴特沃斯滤波器系数 ba1
                        ba1 = cheby2(N, 20, fc, btype, fs=fs)
                        # 使用列表、元组或数组来存储标准化后的截止频率比例，再计算 ba2
                        for seq in (list, tuple, array):
                            fcnorm = seq([f/(fs/2) for f in fc])
                            ba2 = cheby2(N, 20, fcnorm, btype)
                            # 断言 ba1 和 ba2 在数值上接近
                            assert_allclose(ba1, ba2)
class TestEllip:

    def test_degenerate(self):
        # 0-order filter is just a passthrough
        # Even-order filters have DC gain of -rp dB
        # Stopband ripple factor doesn't matter
        
        # 调用 ellip 函数生成 0 阶滤波器的系数 b 和 a，模拟模式为模拟信号
        b, a = ellip(0, 10*np.log10(2), 123.456, 1, analog=True)
        assert_array_almost_equal(b, [1/np.sqrt(2)])
        assert_array_equal(a, [1])

        # 1-order filter is same for all types
        
        # 调用 ellip 函数生成 1 阶滤波器的系数 b 和 a，模拟模式为模拟信号
        b, a = ellip(1, 10*np.log10(2), 1, 1, analog=True)
        assert_array_almost_equal(b, [1])
        assert_array_almost_equal(a, [1, 1])

        # 获取 ellip 函数返回的零点 z、极点 p 和增益 k，输出为 'zpk'
        z, p, k = ellip(1, 1, 55, 0.3, output='zpk')
        assert_allclose(z, [-9.999999999999998e-01], rtol=1e-14)
        assert_allclose(p, [-6.660721153525525e-04], rtol=1e-10)
        assert_allclose(k, 5.003330360576763e-01, rtol=1e-14)

    def test_basic(self):
        for N in range(25):
            wn = 0.01
            # 调用 ellip 函数生成 N 阶低通滤波器的系数 b 和 a，数字模拟模式，输出 'zpk'
            z, p, k = ellip(N, 1, 40, wn, 'low', analog=True, output='zpk')
            assert_(len(p) == N)
            assert_(all(np.real(p) <= 0))  # No poles in right half of S-plane

        for N in range(25):
            wn = 0.01
            # 调用 ellip 函数生成 N 阶高通滤波器的系数 b 和 a，数字模拟模式，输出 'zpk'
            z, p, k = ellip(N, 1, 40, wn, 'high', analog=False, output='zpk')
            assert_(all(np.abs(p) <= 1))  # No poles outside unit circle

        # 调用 ellip 函数生成 5 阶低通滤波器的系数 b 和 a，模拟模式为模拟信号
        b3, a3 = ellip(5, 3, 26, 1, analog=True)
        assert_array_almost_equal(b3, [0.1420, 0, 0.3764, 0,
                                       0.2409], decimal=4)
        assert_array_almost_equal(a3, [1, 0.5686, 1.8061, 0.8017, 0.8012,
                                       0.2409], decimal=4)

        # 调用 ellip 函数生成 3 阶带阻滤波器的系数 b 和 a，截止频率为 0.4 和 0.7
        b, a = ellip(3, 1, 60, [0.4, 0.7], 'stop')
        assert_array_almost_equal(b, [0.3310, 0.3469, 1.1042, 0.7044, 1.1042,
                                      0.3469, 0.3310], decimal=4)
        assert_array_almost_equal(a, [1.0000, 0.6973, 1.1441, 0.5878, 0.7323,
                                      0.1131, -0.0060], decimal=4)
    # 定义一个测试方法 test_bandstop
    def test_bandstop(self):
        # 使用 ellip 函数生成一个 Elliptic 滤波器的零点（z）、极点（p）和增益（k）
        z, p, k = ellip(8, 1, 65, [0.2, 0.4], 'stop', output='zpk')
        
        # 手动设置预期的零点列表 z2
        z2 = [3.528578094286510e-01 + 9.356769561794296e-01j,
              3.528578094286510e-01 - 9.356769561794296e-01j,
              3.769716042264783e-01 + 9.262248159096587e-01j,
              3.769716042264783e-01 - 9.262248159096587e-01j,
              4.406101783111199e-01 + 8.976985411420985e-01j,
              4.406101783111199e-01 - 8.976985411420985e-01j,
              5.539386470258847e-01 + 8.325574907062760e-01j,
              5.539386470258847e-01 - 8.325574907062760e-01j,
              6.748464963023645e-01 + 7.379581332490555e-01j,
              6.748464963023645e-01 - 7.379581332490555e-01j,
              7.489887970285254e-01 + 6.625826604475596e-01j,
              7.489887970285254e-01 - 6.625826604475596e-01j,
              7.913118471618432e-01 + 6.114127579150699e-01j,
              7.913118471618432e-01 - 6.114127579150699e-01j,
              7.806804740916381e-01 + 6.249303940216475e-01j,
              7.806804740916381e-01 - 6.249303940216475e-01j]
        
        # 手动设置预期的极点列表 p2
        p2 = [-1.025299146693730e-01 + 5.662682444754943e-01j,
              -1.025299146693730e-01 - 5.662682444754943e-01j,
               1.698463595163031e-01 + 8.926678667070186e-01j,
               1.698463595163031e-01 - 8.926678667070186e-01j,
               2.750532687820631e-01 + 9.351020170094005e-01j,
               2.750532687820631e-01 - 9.351020170094005e-01j,
               3.070095178909486e-01 + 9.457373499553291e-01j,
               3.070095178909486e-01 - 9.457373499553291e-01j,
               7.695332312152288e-01 + 2.792567212705257e-01j,
               7.695332312152288e-01 - 2.792567212705257e-01j,
               8.083818999225620e-01 + 4.990723496863960e-01j,
               8.083818999225620e-01 - 4.990723496863960e-01j,
               8.066158014414928e-01 + 5.649811440393374e-01j,
               8.066158014414928e-01 - 5.649811440393374e-01j,
               8.062787978834571e-01 + 5.855780880424964e-01j,
               8.062787978834571e-01 - 5.855780880424964e-01j]
        
        # 手动设置预期的增益 k2
        k2 = 2.068622545291259e-01
        
        # 使用 assert_allclose 检查计算得到的零点 z 和预期的零点列表 z2 是否在给定的相对误差范围内匹配
        assert_allclose(sorted(z, key=np.angle),
                        sorted(z2, key=np.angle), rtol=1e-6)
        
        # 使用 assert_allclose 检查计算得到的极点 p 和预期的极点列表 p2 是否在给定的相对误差范围内匹配
        assert_allclose(sorted(p, key=np.angle),
                        sorted(p2, key=np.angle), rtol=1e-5)
        
        # 使用 assert_allclose 检查计算得到的增益 k 和预期的增益 k2 是否在给定的相对误差范围内匹配
        assert_allclose(k, k2, rtol=1e-5)
    # 定义测试方法，用于验证带有特定频率响应的Elliptic滤波器设计
    def test_ba_output(self):
        # 调用ellip函数生成Elliptic滤波器的系数b和a，指定阻带频率和其他参数
        b, a = ellip(5, 1, 40, [201, 240], 'stop', True)
        
        # 期望的滤波器系数b的列表，包括实部和虚部，用于与计算结果比较
        b2 = [
             1.000000000000000e+00, 0,  # Matlab: 1.743506051190569e-13,
             2.426561778314366e+05, 0,  # Matlab: 3.459426536825722e-08,
             2.348218683400168e+10, 0,  # Matlab: 2.559179747299313e-03,
             1.132780692872241e+15, 0,  # Matlab: 8.363229375535731e+01,
             2.724038554089566e+19, 0,  # Matlab: 1.018700994113120e+06,
             2.612380874940186e+23
             ]
        
        # 期望的滤波器系数a的列表，包括实部和虚部，用于与计算结果比较
        a2 = [
             1.000000000000000e+00, 1.337266601804649e+02,
             2.486725353510667e+05, 2.628059713728125e+07,
             2.436169536928770e+10, 1.913554568577315e+12,
             1.175208184614438e+15, 6.115751452473410e+16,
             2.791577695211466e+19, 7.241811142725384e+20,
             2.612380874940182e+23
             ]
        
        # 使用assert_allclose函数断言实际计算得到的b与期望值b2在相对误差容忍度下非常接近
        assert_allclose(b, b2, rtol=1e-6)
        
        # 使用assert_allclose函数断言实际计算得到的a与期望值a2在相对误差容忍度下非常接近
        assert_allclose(a, a2, rtol=1e-4)

    # 定义测试方法，用于验证Elliptic滤波器在不同采样频率下的参数计算
    def test_fs_param(self):
        # 遍历不同的采样频率fs
        for fs in (900, 900.1, 1234.567):
            # 遍历不同阶数N
            for N in (0, 1, 2, 3, 10):
                # 遍历不同的截止频率fc
                for fc in (100, 100.1, 432.12345):
                    # 遍历滤波器类型btype（低通或高通）
                    for btype in ('lp', 'hp'):
                        # 调用ellip函数生成滤波器系数ba1，使用指定的fs、N、20dB衰减、fc和btype
                        ba1 = ellip(N, 1, 20, fc, btype, fs=fs)
                        
                        # 根据fs计算归一化的截止频率fc/(fs/2)，再调用ellip函数生成滤波器系数ba2
                        ba2 = ellip(N, 1, 20, fc/(fs/2), btype)
                        
                        # 使用assert_allclose函数断言ba1与ba2在相对误差容忍度下非常接近
                        assert_allclose(ba1, ba2)
                
                # 遍历带通或带阻滤波器的中心频率对(fc_low, fc_high)
                for fc in ((100, 200), (100.1, 200.2), (321.123, 432.123)):
                    # 遍历带通或带阻滤波器类型btype（带通或带阻）
                    for btype in ('bp', 'bs'):
                        # 调用ellip函数生成滤波器系数ba1，使用指定的fs、N、20dB衰减、(fc_low, fc_high)和btype
                        ba1 = ellip(N, 1, 20, fc, btype, fs=fs)
                        
                        # 根据fs计算归一化的截止频率列表[fc_low/(fs/2), fc_high/(fs/2)]，再调用ellip函数生成滤波器系数ba2
                        fcnorm = [f/(fs/2) for f in fc]
                        ba2 = ellip(N, 1, 20, fcnorm, btype)
                        
                        # 使用assert_allclose函数断言ba1与ba2在相对误差容忍度下非常接近
                        assert_allclose(ba1, ba2)

    # 定义测试方法，用于验证Elliptic滤波器对采样频率的有效性验证
    def test_fs_validation(self):
        # 使用pytest.raises断言捕获采样频率为数组时的ValueError异常，匹配特定的错误信息
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            iirnotch(0.06, 30, fs=np.array([10, 20]))
        
        # 使用pytest.raises断言捕获采样频率为None时的ValueError异常，匹配特定的错误信息
        with pytest.raises(ValueError, match="Sampling.*be none"):
            iirnotch(0.06, 30, fs=None)
def test_sos_consistency():
    # Consistency checks of output='sos' for the specialized IIR filter
    # design functions.
    # 定义一组需要进行一致性检查的 IIR 滤波器设计函数和参数
    design_funcs = [(bessel, (0.1,)),
                    (butter, (0.1,)),
                    (cheby1, (45.0, 0.1)),
                    (cheby2, (0.087, 0.1)),
                    (ellip, (0.087, 45, 0.1))]
    
    # 遍历每个设计函数及其参数
    for func, args in design_funcs:
        name = func.__name__
        
        # 使用设计函数计算 b, a 系数，output='ba'
        b, a = func(2, *args, output='ba')
        # 使用设计函数计算 SOS 形式的滤波器系数，output='sos'
        sos = func(2, *args, output='sos')
        # 断言检查 SOS 形式的滤波器系数是否与期望值一致
        assert_allclose(sos, [np.hstack((b, a))], err_msg="%s(2,...)" % name)

        # 使用设计函数计算 zpk 形式的滤波器系数，output='zpk'
        zpk = func(3, *args, output='zpk')
        # 将 zpk 形式的滤波器系数转换为 SOS 形式
        sos = func(3, *args, output='sos')
        # 断言检查 SOS 形式的滤波器系数是否与转换后的值一致
        assert_allclose(sos, zpk2sos(*zpk), err_msg="%s(3,...)" % name)

        # 使用设计函数计算 zpk 形式的滤波器系数，output='zpk'
        zpk = func(4, *args, output='zpk')
        # 将 zpk 形式的滤波器系数转换为 SOS 形式
        sos = func(4, *args, output='sos')
        # 断言检查 SOS 形式的滤波器系数是否与转换后的值一致
        assert_allclose(sos, zpk2sos(*zpk), err_msg="%s(4,...)" % name)


class TestIIRNotch:

    def test_ba_output(self):
        # Compare coefficients with Matlab ones
        # for the equivalent input:
        # 使用 iirnotch 函数计算 w0=0.06, Q=30 的 b, a 系数
        b, a = iirnotch(0.06, 30)
        # Matlab 中的预期 b 系数
        b2 = [
             9.9686824e-01, -1.9584219e+00,
             9.9686824e-01
             ]
        # Matlab 中的预期 a 系数
        a2 = [
             1.0000000e+00, -1.9584219e+00,
             9.9373647e-01
             ]

        # 断言检查计算得到的 b 是否与 Matlab 中的值接近
        assert_allclose(b, b2, rtol=1e-8)
        # 断言检查计算得到的 a 是否与 Matlab 中的值接近
        assert_allclose(a, a2, rtol=1e-8)

    def test_frequency_response(self):
        # Get filter coefficients
        # 使用 iirnotch 函数计算 w0=0.3, Q=30 的 b, a 系数
        b, a = iirnotch(0.3, 30)

        # Get frequency response
        # 获取频率响应函数 w, h
        w, h = freqz(b, a, 1000)

        # Pick 5 points
        # 选择五个频率点
        p = [200,  # w0 = 0.200
             295,  # w0 = 0.295
             300,  # w0 = 0.300
             305,  # w0 = 0.305
             400]  # w0 = 0.400

        # Get frequency response correspondent to each of those points
        # 获取每个频率点对应的频率响应
        hp = h[p]

        # Check if the frequency response fulfill the specifications:
        # hp[0] and hp[4]  correspond to frequencies distant from
        # w0 = 0.3 and should be close to 1
        # 检查频率响应是否满足规范：
        # hp[0] 和 hp[4] 对应远离 w0 = 0.3 的频率，应接近于 1
        assert_allclose(abs(hp[0]), 1, rtol=1e-2)
        assert_allclose(abs(hp[4]), 1, rtol=1e-2)

        # hp[1] and hp[3] correspond to frequencies approximately
        # on the edges of the passband and should be close to -3dB
        # hp[1] 和 hp[3] 对应通带边缘的频率，应接近于 -3dB
        assert_allclose(abs(hp[1]), 1/np.sqrt(2), rtol=1e-2)
        assert_allclose(abs(hp[3]), 1/np.sqrt(2), rtol=1e-2)

        # hp[2] correspond to the frequency that should be removed
        # the frequency response should be very close to 0
        # hp[2] 对应需要被移除的频率，频率响应应接近于 0
        assert_allclose(abs(hp[2]), 0, atol=1e-10)

    def test_errors(self):
        # Exception should be raised if w0 > 1 or w0 <0
        # 如果 w0 > 1 或 w0 < 0 应该引发异常
        assert_raises(ValueError, iirnotch, w0=2, Q=30)
        assert_raises(ValueError, iirnotch, w0=-1, Q=30)

        # Exception should be raised if any of the parameters
        # are not float (or cannot be converted to one)
        # 如果任何参数不是 float 类型（或无法转换为 float）应该引发异常
        assert_raises(ValueError, iirnotch, w0="blabla", Q=30)
        assert_raises(TypeError, iirnotch, w0=-1, Q=[1, 2, 3])
    # 定义一个测试方法，用于测试参数 fs 的情况

        # 调用 iirnotch 函数获取带阻滤波器的系数 b 和 a
        b, a = iirnotch(1500, 30, fs=10000)

        # 调用 freqz 函数获取频率响应 w 和 h
        # 使用参数 fs=10000，采样频率为 10000 Hz
        w, h = freqz(b, a, 1000, fs=10000)

        # 选取五个频率点
        p = [200,  # w0 = 1000
             295,  # w0 = 1475
             300,  # w0 = 1500
             305,  # w0 = 1525
             400]  # w0 = 2000

        # 获取每个选定频率点对应的频率响应值
        hp = h[p]

        # 检查频率响应是否符合规范：

        # hp[0] 和 hp[4] 对应距离 w0 = 1500 的频率，应接近于 1
        assert_allclose(abs(hp[0]), 1, rtol=1e-2)
        assert_allclose(abs(hp[4]), 1, rtol=1e-2)

        # hp[1] 和 hp[3] 对应通带边缘附近的频率，应接近于 -3dB
        assert_allclose(abs(hp[1]), 1/np.sqrt(2), rtol=1e-2)
        assert_allclose(abs(hp[3]), 1/np.sqrt(2), rtol=1e-2)

        # hp[2] 对应应该被抑制的频率，其频率响应应该非常接近于 0
        assert_allclose(abs(hp[2]), 0, atol=1e-10)
class TestIIRPeak:

    def test_ba_output(self):
        # 比较系数与 Matlab 的等价输入
        # 对应的 iirpeak 函数调用，获取 b 和 a 系数
        b, a = iirpeak(0.06, 30)
        
        # 预定义 Matlab 的系数值
        b2 = [
             3.131764229e-03, 0,
             -3.131764229e-03
             ]
        a2 = [
             1.0000000e+00, -1.958421917e+00,
             9.9373647e-01
             ]
        
        # 使用 assert_allclose 函数检查 b 和 b2 是否非常接近
        assert_allclose(b, b2, rtol=1e-8)
        
        # 使用 assert_allclose 函数检查 a 和 a2 是否非常接近
        assert_allclose(a, a2, rtol=1e-8)

    def test_frequency_response(self):
        # 获取滤波器系数
        b, a = iirpeak(0.3, 30)

        # 获取频率响应
        w, h = freqz(b, a, 1000)

        # 选择 5 个频率点
        p = [30,  # w0 = 0.030
             295,  # w0 = 0.295
             300,  # w0 = 0.300
             305,  # w0 = 0.305
             800]  # w0 = 0.800

        # 获取每个频率点对应的频率响应
        hp = h[p]

        # 检查频率响应是否符合规范：
        # hp[0] 和 hp[4] 对应于远离 w0 = 0.3 的频率，应该接近于 0
        assert_allclose(abs(hp[0]), 0, atol=1e-2)
        assert_allclose(abs(hp[4]), 0, atol=1e-2)

        # hp[1] 和 hp[3] 对应于通带边缘的频率，应该接近于 10**(-3/20)
        assert_allclose(abs(hp[1]), 1/np.sqrt(2), rtol=1e-2)
        assert_allclose(abs(hp[3]), 1/np.sqrt(2), rtol=1e-2)

        # hp[2] 对应于应该保留的频率，频率响应应该非常接近于 1
        assert_allclose(abs(hp[2]), 1, rtol=1e-10)

    def test_errors(self):
        # 如果 w0 > 1 或 w0 < 0，应该引发 ValueError 异常
        assert_raises(ValueError, iirpeak, w0=2, Q=30)
        assert_raises(ValueError, iirpeak, w0=-1, Q=30)

        # 如果任何参数不是浮点数（或不能转换为浮点数），应该引发 ValueError 异常
        assert_raises(ValueError, iirpeak, w0="blabla", Q=30)
        assert_raises(TypeError, iirpeak, w0=-1, Q=[1, 2, 3])
    # 定义一个测试方法，用于测试带有频率参数的滤波器设计

    # 获取 IIR 峰值滤波器的滤波器系数
    b, a = iirpeak(1200, 30, fs=8000)

    # 获取频率响应
    w, h = freqz(b, a, 1000, fs=8000)

    # 选择5个频率点
    p = [30,    # w0 = 120
         295,   # w0 = 1180
         300,   # w0 = 1200
         305,   # w0 = 1220
         800]   # w0 = 3200

    # 获取每个频率点对应的频率响应
    hp = h[p]

    # 检查频率响应是否符合规格要求：

    # hp[0] 和 hp[4] 对应的频率远离 w0 = 1200，应该接近于0
    assert_allclose(abs(hp[0]), 0, atol=1e-2)
    assert_allclose(abs(hp[4]), 0, atol=1e-2)

    # hp[1] 和 hp[3] 对应的频率大致在通带的边缘，应该接近于 10**(-3/20)
    assert_allclose(abs(hp[1]), 1/np.sqrt(2), rtol=1e-2)
    assert_allclose(abs(hp[3]), 1/np.sqrt(2), rtol=1e-2)

    # hp[2] 对应应该保留的频率，频率响应应该非常接近于1
    assert_allclose(abs(hp[2]), 1, rtol=1e-10)
# 定义一个测试类 TestIIRComb，用于测试 IIRComb 滤波器的功能
class TestIIRComb:
    
    # 测试错误输入情况
    def test_invalid_input(self):
        # 设定采样率 fs 为 1000 Hz
        fs = 1000
        
        # 对于每组参数 args，在以下情况下会引发 ValueError 异常
        # w0 <= 0 或者 w0 >= fs / 2
        for args in [(-fs, 30), (0, 35), (fs / 2, 40), (fs, 35)]:
            with pytest.raises(ValueError, match='w0 must be between '):
                iircomb(*args, fs=fs)

        # fs 不能被 w0 整除
        for args in [(120, 30), (157, 35)]:
            with pytest.raises(ValueError, match='fs must be divisible '):
                iircomb(*args, fs=fs)

        # 测试特定情况下的异常情况
        # https://github.com/scipy/scipy/issues/14043#issuecomment-1107349140
        # 以前，fs=44100，w0=49.999 会被拒绝，但是现在，fs=2，w0=49.999/int(44100/2) 也被拒绝了。
        with pytest.raises(ValueError, match='fs must be divisible '):
            iircomb(w0=49.999/int(44100/2), Q=30)

        with pytest.raises(ValueError, match='fs must be divisible '):
            iircomb(w0=49.999, Q=30, fs=44100)

        # 滤波器类型不是 notch 或 peak
        for args in [(0.2, 30, 'natch'), (0.5, 35, 'comb')]:
            with pytest.raises(ValueError, match='ftype must be '):
                iircomb(*args)

    # 验证滤波器的频率响应是否在截止频率处有一个 notch
    @pytest.mark.parametrize('ftype', ('notch', 'peak'))
    def test_frequency_response(self, ftype):
        # 创建一个在 1000 Hz 处的 notch 或 peak comb 滤波器
        b, a = iircomb(1000, 30, ftype=ftype, fs=10000)

        # 计算频率响应
        freqs, response = freqz(b, a, 1000, fs=10000)

        # 使用 argrelextrema 找到 notch
        comb_points = argrelextrema(abs(response), np.less)[0]

        # 验证第一个 notch 是否位于 1000 Hz 处
        comb1 = comb_points[0]
        assert_allclose(freqs[comb1], 1000)

    # 验证 pass_zero 参数
    @pytest.mark.parametrize('ftype,pass_zero,peak,notch',
                             [('peak', True, 123.45, 61.725),
                              ('peak', False, 61.725, 123.45),
                              ('peak', None, 61.725, 123.45),
                              ('notch', None, 61.725, 123.45),
                              ('notch', True, 123.45, 61.725),
                              ('notch', False, 61.725, 123.45)])
    def test_pass_zero(self, ftype, pass_zero, peak, notch):
        # 创建一个 notching 或 peaking comb 滤波器
        b, a = iircomb(123.45, 30, ftype=ftype, fs=1234.5, pass_zero=pass_zero)

        # 计算频率响应
        freqs, response = freqz(b, a, [peak, notch], fs=1234.5)

        # 验证期望的 notch 是否为 notch，peak 是否为 peak
        assert abs(response[0]) > 0.99
        assert abs(response[1]) < 1e-10

    # 所有内置的 IIR 滤波器都是实数，所以应该有完全对称的极点和零点。
    # ba 表示法（使用 numpy.poly）将是完全实数，而不是具有可以忽略的虚部。
    # 验证 IIR 滤波器的对称性
    def test_iir_symmetry(self):
        # 调用 iircomb 函数获取 IIR 组合滤波器的系数 b 和 a
        b, a = iircomb(400, 30, fs=24000)
        # 将 b, a 转换为零极点形式
        z, p, k = tf2zpk(b, a)
        # 断言零点 z 和其共轭的排序结果相同
        assert_array_equal(sorted(z), sorted(z.conj()))
        # 断言极点 p 和其共轭的排序结果相同
        assert_array_equal(sorted(p), sorted(p.conj()))
        # 断言增益 k 是实数
        assert_equal(k, np.real(k))

        # 断言 b 的数据类型是 np.floating 的子类
        assert issubclass(b.dtype.type, np.floating)
        # 断言 a 的数据类型是 np.floating 的子类
        assert issubclass(a.dtype.type, np.floating)

    # 使用 MATLAB 的 iircomb 函数验证滤波器系数
    def test_ba_output(self):
        # 调用 iircomb 函数创建陷波滤波器 b_notch 和 a_notch
        b_notch, a_notch = iircomb(60, 35, ftype='notch', fs=600)
        # 期望的 b_notch2 和 a_notch2 系数
        b_notch2 = [0.957020174408697, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, -0.957020174408697]
        a_notch2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, -0.914040348817395]
        # 断言 b_notch 和 b_notch2 的所有元素近似相等
        assert_allclose(b_notch, b_notch2)
        # 断言 a_notch 和 a_notch2 的所有元素近似相等
        assert_allclose(a_notch, a_notch2)

        # 调用 iircomb 函数创建峰值滤波器 b_peak 和 a_peak
        b_peak, a_peak = iircomb(60, 35, ftype='peak', fs=600)
        # 期望的 b_peak2 和 a_peak2 系数
        b_peak2 = [0.0429798255913026, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, -0.0429798255913026]
        a_peak2 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.914040348817395]
        # 断言 b_peak 和 b_peak2 的所有元素近似相等
        assert_allclose(b_peak, b_peak2)
        # 断言 a_peak 和 a_peak2 的所有元素近似相等
        assert_allclose(a_peak, a_peak2)

    # 验证 https://github.com/scipy/scipy/issues/14043 的问题是否已修复
    def test_nearest_divisor(self):
        # 创建一个陷波组合滤波器 b 和 a
        b, a = iircomb(50/int(44100/2), 50.0, ftype='notch')

        # 计算在 22 kHz 上一个上层谐波的频率响应
        freqs, response = freqz(b, a, [22000], fs=44100)

        # 在修复 bug 前，这应该产生 N = 881，所以 22 kHz 是约 0 dB。
        # 现在 N = 882 正确，22 kHz 应该是一个陷波，<-220 dB
        assert abs(response[0]) < 1e-10

    # 验证采样率参数的有效性
    def test_fs_validation(self):
        # 使用 pytest 断言捕获到 ValueError 异常，匹配特定消息
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            # 调用 iircomb 函数，传入无效的采样率参数 np.array([10, 20])
            iircomb(1000, 30, fs=np.array([10, 20]))

        # 使用 pytest 断言捕获到 ValueError 异常，匹配特定消息
        with pytest.raises(ValueError, match="Sampling.*be none"):
            # 调用 iircomb 函数，传入 None 作为采样率参数
            iircomb(1000, 30, fs=None)
class TestIIRDesign:

    def test_fs_validation(self):
        # 断言应该抛出 ValueError 异常，且异常信息匹配指定的正则表达式
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            # 调用 iirfilter 函数，传入无效参数，期望抛出异常
            iirfilter(1, 1, btype="low", fs=np.array([10, 20]))


class TestIIRFilter:

    def test_symmetry(self):
        # 所有内置 IIR 滤波器都是实数，因此应具有完全对称的极点和零点。
        # 因此 ba 表示（使用 numpy.poly）将是纯实数，而不是具有微不足道的虚部。
        for N in np.arange(1, 26):
            for ftype in ('butter', 'bessel', 'cheby1', 'cheby2', 'ellip'):
                # 获取零点、极点和增益，要求输出为零极点形式
                z, p, k = iirfilter(N, 1.1, 1, 20, 'low', analog=True,
                                    ftype=ftype, output='zpk')
                # 断言排序后的零点与其共轭的排序后零点相等
                assert_array_equal(sorted(z), sorted(z.conj()))
                # 断言排序后的极点与其共轭的排序后极点相等
                assert_array_equal(sorted(p), sorted(p.conj()))
                # 断言增益为实数
                assert_equal(k, np.real(k))

                # 获取传递函数系数 b 和 a，要求输出为系数形式
                b, a = iirfilter(N, 1.1, 1, 20, 'low', analog=True,
                                 ftype=ftype, output='ba')
                # 断言 b 的数据类型为浮点数
                assert_(issubclass(b.dtype.type, np.floating))
                # 断言 a 的数据类型为浮点数
                assert_(issubclass(a.dtype.type, np.floating))

    def test_int_inputs(self):
        # 使用整数频率参数和大 N 不应导致 numpy 整数溢出为负数
        # 获取增益 k，要求输出为零极点形式，验证其与预期值的接近程度
        k = iirfilter(24, 100, btype='low', analog=True, ftype='bessel',
                      output='zpk')[2]
        k2 = 9.999999999999989e+47
        assert_allclose(k, k2)
        # 如果指定了 fs，则 Wn 的归一化使得 0 <= Wn <= 1 不应导致整数溢出
        # 以下行不应引发异常
        iirfilter(20, [1000000000, 1100000000], btype='bp', 
                  analog=False, fs=6250000000)

    def test_invalid_wn_size(self):
        # 对于低通和高通，需要一个 Wn；对于带通和阻带，需要两个 Wn
        # 断言调用 iirfilter 函数时会抛出 ValueError 异常
        assert_raises(ValueError, iirfilter, 1, [0.1, 0.9], btype='low')
        assert_raises(ValueError, iirfilter, 1, [0.2, 0.5], btype='high')
        assert_raises(ValueError, iirfilter, 1, 0.2, btype='bp')
        assert_raises(ValueError, iirfilter, 1, 400, btype='bs', analog=True)
    def test_invalid_wn_range(self):
        # 检查数字滤波器的频率范围，应满足 0 <= Wn <= 1
        assert_raises(ValueError, iirfilter, 1, 2, btype='low')  # 检查低通滤波器的频率范围异常情况
        assert_raises(ValueError, iirfilter, 1, [0.5, 1], btype='band')  # 检查带通滤波器的频率范围异常情况
        assert_raises(ValueError, iirfilter, 1, [0., 0.5], btype='band')  # 检查带通滤波器的频率范围异常情况
        assert_raises(ValueError, iirfilter, 1, -1, btype='high')  # 检查高通滤波器的频率范围异常情况
        assert_raises(ValueError, iirfilter, 1, [1, 2], btype='band')  # 检查带通滤波器的频率范围异常情况
        assert_raises(ValueError, iirfilter, 1, [10, 20], btype='stop')  # 检查阻带滤波器的频率范围异常情况

        # 当使用模拟滤波器且关键频率非正时应引发 ValueError 异常
        with pytest.raises(ValueError, match="must be greater than 0"):
            iirfilter(2, 0, btype='low', analog=True)  # 检查模拟低通滤波器关键频率异常情况
        with pytest.raises(ValueError, match="must be greater than 0"):
            iirfilter(2, -1, btype='low', analog=True)  # 检查模拟低通滤波器关键频率异常情况
        with pytest.raises(ValueError, match="must be greater than 0"):
            iirfilter(2, [0, 100], analog=True)  # 检查模拟带通滤波器关键频率异常情况
        with pytest.raises(ValueError, match="must be greater than 0"):
            iirfilter(2, [-1, 100], analog=True)  # 检查模拟带通滤波器关键频率异常情况
        with pytest.raises(ValueError, match="must be greater than 0"):
            iirfilter(2, [10, 0], analog=True)  # 检查模拟带通滤波器关键频率异常情况
        with pytest.raises(ValueError, match="must be greater than 0"):
            iirfilter(2, [10, -1], analog=True)  # 检查模拟带通滤波器关键频率异常情况
class TestGroupDelay:
    # 定义测试类 TestGroupDelay，用于测试 group_delay 函数的各种情况

    def test_identity_filter(self):
        # 测试恒等滤波器情况下的 group_delay 函数
        w, gd = group_delay((1, 1))
        # 断言频率数组 w 几乎等于 pi 乘以 512 等分后的数组
        assert_array_almost_equal(w, pi * np.arange(512) / 512)
        # 断言组延迟数组 gd 是 512 个零组成的数组
        assert_array_almost_equal(gd, np.zeros(512))
        # 再次调用 group_delay 函数，测试参数设置为 whole=True 的情况
        w, gd = group_delay((1, 1), whole=True)
        # 断言频率数组 w 几乎等于 2*pi 乘以 512 等分后的数组
        assert_array_almost_equal(w, 2 * pi * np.arange(512) / 512)
        # 断言组延迟数组 gd 是 512 个零组成的数组
        assert_array_almost_equal(gd, np.zeros(512))

    def test_fir(self):
        # 测试线性相位 FIR 滤波器的情况，并检查组延迟是否恒定
        N = 100
        # 设计长度为 N+1 的 FIR 窗口
        b = firwin(N + 1, 0.1)
        # 调用 group_delay 函数计算频率数组 w 和组延迟 gd
        w, gd = group_delay((b, 1))
        # 断言组延迟 gd 几乎等于 0.5 * N
        assert_allclose(gd, 0.5 * N)

    def test_iir(self):
        # 测试 Butterworth 滤波器的情况，并将组延迟与 MATLAB 结果进行比较
        b, a = butter(4, 0.1)
        # 在频率范围内生成 10 个点的频率数组 w
        w = np.linspace(0, pi, num=10, endpoint=False)
        # 调用 group_delay 函数计算频率数组 w 和组延迟 gd
        w, gd = group_delay((b, a), w=w)
        # MATLAB 计算的组延迟结果
        matlab_gd = np.array([8.249313898506037, 11.958947880907104,
                              2.452325615326005, 1.048918665702008,
                              0.611382575635897, 0.418293269460578,
                              0.317932917836572, 0.261371844762525,
                              0.229038045801298, 0.212185774208521])
        # 断言计算得到的组延迟 gd 与 MATLAB 计算的结果相近
        assert_array_almost_equal(gd, matlab_gd)

    def test_singular(self):
        # 测试在单位圆上存在零点和极点的滤波器，并检查是否在这些频率上引发警告
        z1 = np.exp(1j * 0.1 * pi)
        z2 = np.exp(1j * 0.25 * pi)
        p1 = np.exp(1j * 0.5 * pi)
        p2 = np.exp(1j * 0.8 * pi)
        # 计算滤波器系数 b 和 a
        b = np.convolve([1, -z1], [1, -z2])
        a = np.convolve([1, -p1], [1, -p2])
        # 在指定频率点生成频率数组 w
        w = np.array([0.1 * pi, 0.25 * pi, -0.5 * pi, -0.8 * pi])
        # 调用 group_delay 函数，并断言会引发 UserWarning 警告
        w, gd = assert_warns(UserWarning, group_delay, (b, a), w=w)

    def test_backward_compat(self):
        # 为了向后兼容性，测试当参数为 None 时是否作为默认值处理
        w1, gd1 = group_delay((1, 1))
        w2, gd2 = group_delay((1, 1), None)
        # 断言两种调用方式得到的频率数组 w 和组延迟 gd 几乎相等
        assert_array_almost_equal(w1, w2)
        assert_array_almost_equal(gd1, gd2)

    def test_fs_param(self):
        # 测试指定采样率情况下的 Butterworth 滤波器，并将组延迟与归一化频率结果进行比较
        b, a = butter(4, 4800, fs=96000)
        # 在指定频率范围内生成频率数组 w
        w = np.linspace(0, 96000/2, num=10, endpoint=False)
        # 调用 group_delay 函数计算频率数组 w 和组延迟 gd
        w, gd = group_delay((b, a), w=w, fs=96000)
        # 归一化组延迟结果
        norm_gd = np.array([8.249313898506037, 11.958947880907104,
                            2.452325615326005, 1.048918665702008,
                            0.611382575635897, 0.418293269460578,
                            0.317932917836572, 0.261371844762525,
                            0.229038045801298, 0.212185774208521])
        # 断言计算得到的组延迟 gd 与归一化结果相近
        assert_array_almost_equal(gd, norm_gd)
    def test_w_or_N_types(self):
        # Measure at 8 equally-spaced points
        # 在8个等间距点上进行测量
        for N in (8, np.int8(8), np.int16(8), np.int32(8), np.int64(8),
                  np.array(8)):
            # 调用 group_delay 函数计算组延迟
            w, gd = group_delay((1, 1), N)
            # 断言频率数组 w 与预期的 pi * np.arange(8) / 8 几乎相等
            assert_array_almost_equal(w, pi * np.arange(8) / 8)
            # 断言组延迟数组 gd 与预期的 np.zeros(8) 几乎相等
            assert_array_almost_equal(gd, np.zeros(8))

        # Measure at frequency 8 rad/sec
        # 在频率为8 rad/sec处进行测量
        for w in (8.0, 8.0+0j):
            # 调用 group_delay 函数计算组延迟
            w_out, gd = group_delay((1, 1), w)
            # 断言频率数组 w_out 与预期的 [8] 几乎相等
            assert_array_almost_equal(w_out, [8])
            # 断言组延迟数组 gd 与预期的 [0] 几乎相等
            assert_array_almost_equal(gd, [0])

    def test_complex_coef(self):
        # gh-19586: handle complex coef TFs
        # 处理复数系数的传递函数
        #
        # for g(z) = (alpha*z+1)/(1+conjugate(alpha)), group delay is
        # given by function below.
        # 对于传递函数 g(z) = (alpha*z+1)/(1+conjugate(alpha))，下面的函数给出了组延迟。
        #
        # def gd_expr(w, alpha):
        #     num = 1j*(abs(alpha)**2-1)*np.exp(1j*w)
        #     den = (alpha*np.exp(1j*w)+1)*(np.exp(1j*w)+np.conj(alpha))
        #     return -np.imag(num/den)

        # arbitrary non-real alpha
        # 任意非实数 alpha
        alpha = -0.6143077933232609+0.3355978770229421j
        # 8 points from from -pi to pi
        # 从 -pi 到 pi 的8个点
        wref = np.array([-3.141592653589793 ,
                         -2.356194490192345 ,
                         -1.5707963267948966,
                         -0.7853981633974483,
                         0.                ,
                         0.7853981633974483,
                         1.5707963267948966,
                         2.356194490192345 ])
        gdref =  array([0.18759548150354619,
                        0.17999770352712252,
                        0.23598047471879877,
                        0.46539443069907194,
                        1.9511492420564165 ,
                        3.478129975138865  ,
                        0.6228594960517333 ,
                        0.27067831839471224])
        b = [alpha,1]
        a = [1, np.conjugate(alpha)]
        # 调用 group_delay 函数计算组延迟，并且使用 nulp=16 进行断言
        gdtest = group_delay((b,a), wref)[1]
        # 需要 nulp=14 适应 macOS arm64 平台的构建；对于其他平台，增加了 2 以增强鲁棒性。
        assert_array_almost_equal_nulp(gdtest, gdref, nulp=16)

    def test_fs_validation(self):
        # 使用 pytest 来断言异常：Sampling 需要是单个标量
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            group_delay((1, 1), fs=np.array([10, 20]))

        # 使用 pytest 来断言异常：Sampling 不能是 None
        with pytest.raises(ValueError, match="Sampling.*be none"):
            group_delay((1, 1), fs=None)
class TestGammatone:
    # 测试错误输入情况。

    def test_invalid_input(self):
        # 设置采样频率为 16000
        fs = 16000
        # 对于不同的参数组合进行循环测试
        for args in [(-fs, 'iir'), (0, 'fir'), (fs / 2, 'iir'), (fs, 'fir')]:
            # 使用 pytest 断言异常并匹配特定消息
            with pytest.raises(ValueError, match='The frequency must be '):
                gammatone(*args, fs=fs)

        # 滤波器类型不是 'fir' 或 'iir'
        for args in [(440, 'fie'), (220, 'it')]:
            # 使用 pytest 断言异常并匹配特定消息
            with pytest.raises(ValueError, match='ftype must be '):
                gammatone(*args, fs=fs)

        # 对于 FIR 滤波器，阶数应该在大于 0 且不超过 24
        for args in [(440, 'fir', -50), (220, 'fir', 0), (110, 'fir', 25),
                     (55, 'fir', 50)]:
            # 使用 pytest 断言异常并匹配特定消息
            with pytest.raises(ValueError, match='Invalid order: '):
                gammatone(*args, numtaps=None, fs=fs)

    # 验证滤波器在截止频率处的频率响应是否接近 1。
    def test_frequency_response(self):
        # 设置采样频率为 16000
        fs = 16000
        # 循环测试不同类型的滤波器
        ftypes = ['fir', 'iir']
        for ftype in ftypes:
            # 创建以 1000 Hz 为中心的 gammatone 滤波器
            b, a = gammatone(1000, ftype, fs=fs)

            # 计算频率响应
            freqs, response = freqz(b, a)

            # 确定响应的峰值幅度和对应的频率
            response_max = np.max(np.abs(response))
            freq_hz = freqs[np.argmax(np.abs(response))] / ((2 * np.pi) / fs)

            # 断言峰值幅度为 1，频率为 1000 Hz
            assert_allclose(response_max, 1, rtol=1e-2)
            assert_allclose(freq_hz, 1000, rtol=1e-2)

    # 所有内置 IIR 滤波器都是实数，因此它们应该具有完全对称的极点和零点。
    # ba 表示（使用 numpy.poly）应该是纯实数，而不是具有可忽略虚部的形式。
    def test_iir_symmetry(self):
        # 创建一个以 440 Hz 为中心的 IIR gammatone 滤波器
        b, a = gammatone(440, 'iir', fs=24000)
        # 将滤波器转换为极点零点增益形式
        z, p, k = tf2zpk(b, a)
        # 断言极点按排序后应该是对称的
        assert_array_equal(sorted(z), sorted(z.conj()))
        # 断言零点按排序后应该是对称的
        assert_array_equal(sorted(p), sorted(p.conj()))
        # 断言增益是实数
        assert_equal(k, np.real(k))

        # 断言 b 和 a 的数据类型是浮点型
        assert_(issubclass(b.dtype.type, np.floating))
        assert_(issubclass(a.dtype.type, np.floating))

    # 使用论文中的 Mathematica 实现验证 FIR 滤波器系数
    def test_fir_ba_output(self):
        # 创建一个以 15 Hz 为中心的 FIR gammatone 滤波器
        b, _ = gammatone(15, 'fir', fs=1000)
        # 论文中给出的系数
        b2 = [0.0, 2.2608075649884e-04,
              1.5077903981357e-03, 4.2033687753998e-03,
              8.1508962726503e-03, 1.2890059089154e-02,
              1.7833890391666e-02, 2.2392613558564e-02,
              2.6055195863104e-02, 2.8435872863284e-02,
              2.9293319149544e-02, 2.852976858014e-02,
              2.6176557156294e-02, 2.2371510270395e-02,
              1.7332485267759e-02]
        # 断言系数应该接近
        assert_allclose(b, b2)

    # 使用论文中的 MATLAB 实现验证 IIR 滤波器系数
    # 定义一个测试函数，用于测试 IIR 滤波器设计函数的输出
    def test_iir_ba_output(self):
        # 调用 gammatone 函数，生成 440Hz 的 IIR 滤波器系数 b 和 a
        b, a = gammatone(440, 'iir', fs=16000)
        # 预期的滤波器系数 b2，这是一个包含浮点数的列表
        b2 = [1.31494461367464e-06, -5.03391196645395e-06,
              7.00649426000897e-06, -4.18951968419854e-06,
              9.02614910412011e-07]
        # 预期的滤波器系数 a2，这是一个包含浮点数的列表
        a2 = [1.0, -7.65646235454218,
              25.7584699322366, -49.7319214483238,
              60.2667361289181, -46.9399590980486,
              22.9474798808461, -6.43799381299034,
              0.793651554625368]
        # 使用 assert_allclose 函数检查计算得到的 b 是否与预期的 b2 接近
        assert_allclose(b, b2)
        # 使用 assert_allclose 函数检查计算得到的 a 是否与预期的 a2 接近
        assert_allclose(a, a2)
    
    # 定义另一个测试函数，用于测试输入参数 fs 的验证
    def test_fs_validation(self):
        # 使用 pytest.raises 来检查是否会抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            # 调用 gammatone 函数，传入一个 np.array([10, 20]) 作为 fs 参数
            gammatone(440, 'iir', fs=np.array([10, 20]))
class TestOrderFilter:
    def test_doc_example(self):
        # 创建一个 5x5 的 NumPy 数组，其中包含 0 到 24 的整数，然后重新形状为 5x5
        x = np.arange(25).reshape(5, 5)
        # 创建一个 3x3 的单位矩阵作为滤波器的领域
        domain = np.identity(3)

        # 期望的输出是一个包含特定值的 5x5 的 NumPy 数组
        expected = np.array(
            [[0., 0., 0., 0., 0.],
             [0., 0., 1., 2., 0.],
             [0., 5., 6., 7., 0.],
             [0., 10., 11., 12., 0.],
             [0., 0., 0., 0., 0.]],
        )
        # 断言 order_filter 函数对 x 应用领域 domain 和 order 0 的结果等于期望值
        assert_allclose(order_filter(x, domain, 0), expected)

        # 期望的输出是一个包含特定值的 5x5 的 NumPy 数组
        expected = np.array(
            [[6., 7., 8., 9., 4.],
             [11., 12., 13., 14., 9.],
             [16., 17., 18., 19., 14.],
             [21., 22., 23., 24., 19.],
             [20., 21., 22., 23., 24.]],
        )
        # 断言 order_filter 函数对 x 应用领域 domain 和 order 2 的结果等于期望值
        assert_allclose(order_filter(x, domain, 2), expected)

        # 期望的输出是一个包含特定值的 5x5 的 NumPy 数组
        expected = np.array(
            [[0, 1, 2, 3, 0],
             [5, 6, 7, 8, 3],
             [10, 11, 12, 13, 8],
             [15, 16, 17, 18, 13],
             [0, 15, 16, 17, 18]],
        )
        # 断言 order_filter 函数对 x 应用领域 domain 和 order 1 的结果等于期望值
        assert_allclose(order_filter(x, domain, 1), expected)

    def test_medfilt_order_filter(self):
        # 创建一个 5x5 的 NumPy 数组，其中包含 0 到 24 的整数，然后重新形状为 5x5
        x = np.arange(25).reshape(5, 5)

        # 期望的输出是一个包含特定值的 5x5 的 NumPy 数组
        expected = np.array(
            [[0, 1, 2, 3, 0],
             [1, 6, 7, 8, 4],
             [6, 11, 12, 13, 9],
             [11, 16, 17, 18, 14],
             [0, 16, 17, 18, 0]],
        )
        # 断言 medfilt 函数对 x 应用窗口大小 3 的结果等于期望值
        assert_allclose(medfilt(x, 3), expected)

        # 断言 order_filter 函数对 x 应用 3x3 的全 1 领域和 order 4 的结果等于期望值
        assert_allclose(
            order_filter(x, np.ones((3, 3)), 4),
            expected
        )

    def test_order_filter_asymmetric(self):
        # 创建一个 5x5 的 NumPy 数组，其中包含 0 到 24 的整数，然后重新形状为 5x5
        x = np.arange(25).reshape(5, 5)
        # 创建一个自定义的 3x3 NumPy 数组作为不对称的领域
        domain = np.array(
            [[1, 1, 0],
             [0, 1, 0],
             [0, 0, 0]],
        )

        # 期望的输出是一个包含特定值的 5x5 的 NumPy 数组
        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 2, 3],
             [0, 5, 6, 7, 8],
             [0, 10, 11, 12, 13],
             [0, 15, 16, 17, 18]]
        )
        # 断言 order_filter 函数对 x 应用领域 domain 和 order 0 的结果等于期望值
        assert_allclose(order_filter(x, domain, 0), expected)

        # 期望的输出是一个包含特定值的 5x5 的 NumPy 数组
        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 1, 2, 3, 4],
             [5, 6, 7, 8, 9],
             [10, 11, 12, 13, 14],
             [15, 16, 17, 18, 19]]
        )
        # 断言 order_filter 函数对 x 应用领域 domain 和 order 1 的结果等于期望值
        assert_allclose(order_filter(x, domain, 1), expected)
```