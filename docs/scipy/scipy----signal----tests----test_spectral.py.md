# `D:\src\scipysrc\scipy\scipy\signal\tests\test_spectral.py`

```
# 导入 sys 模块，用于访问系统相关功能
import sys

# 导入 numpy 库，并将其重命名为 np
import numpy as np

# 从 numpy.testing 模块中导入多个断言函数，用于进行测试时的断言判断
from numpy.testing import (assert_, assert_approx_equal,
                           assert_allclose, assert_array_equal, assert_equal,
                           assert_array_almost_equal_nulp, suppress_warnings)

# 导入 pytest 库
import pytest

# 从 pytest 模块中导入 raises 函数，并将其重命名为 assert_raises
from pytest import raises as assert_raises

# 导入 scipy 库中的 signal 模块
from scipy import signal

# 从 scipy.fft 模块中导入 fftfreq, rfftfreq, fft, irfft 函数
from scipy.fft import fftfreq, rfftfreq, fft, irfft

# 从 scipy.integrate 模块中导入 trapezoid 函数
from scipy.integrate import trapezoid

# 导入 scipy.signal 模块中的多个函数：periodogram, welch, lombscargle, coherence,
# spectrogram, check_COLA, check_NOLA
from scipy.signal import (periodogram, welch, lombscargle, coherence,
                          spectrogram, check_COLA, check_NOLA)

# 从 scipy.signal.windows 模块中导入 hann 窗函数
from scipy.signal.windows import hann

# 导入 scipy.signal._spectral_py 模块中的 _spectral_helper 函数
from scipy.signal._spectral_py import _spectral_helper

# 从 scipy.signal.tests._scipy_spectral_test_shim 模块中导入 stft_compare, istft_compare, csd_compare 函数
# 并将其重命名为 stft, istft, csd
from scipy.signal.tests._scipy_spectral_test_shim import stft_compare as stft
from scipy.signal.tests._scipy_spectral_test_shim import istft_compare as istft
from scipy.signal.tests._scipy_spectral_test_shim import csd_compare as csd


class TestPeriodogram:
    # 测试函数：test_real_onesided_even
    def test_real_onesided_even(self):
        # 创建长度为 16 的零数组，并将第一个元素设为 1
        x = np.zeros(16)
        x[0] = 1
        # 调用 periodogram 函数计算信号 x 的功率谱
        f, p = periodogram(x)
        # 断言频率 f 等于从 0 到 0.5 等间隔采样的数组
        assert_allclose(f, np.linspace(0, 0.5, 9))
        # 创建期望功率谱数组 q，首尾特殊处理，然后除以 8
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        # 断言计算得到的功率谱 p 等于期望功率谱 q
        assert_allclose(p, q)

    # 测试函数：test_real_onesided_odd
    def test_real_onesided_odd(self):
        # 创建长度为 15 的零数组，并将第一个元素设为 1
        x = np.zeros(15)
        x[0] = 1
        # 调用 periodogram 函数计算信号 x 的功率谱
        f, p = periodogram(x)
        # 断言频率 f 等于 np.arange(8.0)/15.0
        assert_allclose(f, np.arange(8.0)/15.0)
        # 创建期望功率谱数组 q，首位特殊处理，然后乘以 2/15
        q = np.ones(8)
        q[0] = 0
        q *= 2.0/15.0
        # 断言计算得到的功率谱 p 等于期望功率谱 q，设置容忍度为 1e-15
        assert_allclose(p, q, atol=1e-15)

    # 测试函数：test_real_twosided
    def test_real_twosided(self):
        # 创建长度为 16 的零数组，并将第一个元素设为 1
        x = np.zeros(16)
        x[0] = 1
        # 调用 periodogram 函数计算信号 x 的功率谱，设置 return_onesided=False 表示计算双边频谱
        f, p = periodogram(x, return_onesided=False)
        # 断言频率 f 等于 fftfreq(16, 1.0)
        assert_allclose(f, fftfreq(16, 1.0))
        # 创建期望功率谱数组 q，每个值设为 1/16.0，首位特殊处理
        q = np.full(16, 1/16.0)
        q[0] = 0
        # 断言计算得到的功率谱 p 等于期望功率谱 q
        assert_allclose(p, q)

    # 测试函数：test_real_spectrum
    def test_real_spectrum(self):
        # 创建长度为 16 的零数组，并将第一个元素设为 1
        x = np.zeros(16)
        x[0] = 1
        # 调用 periodogram 函数计算信号 x 的功率谱，设置 scaling='spectrum' 表示计算谱密度
        f, p = periodogram(x, scaling='spectrum')
        # 调用 periodogram 函数计算信号 x 的功率谱，设置 scaling='density' 表示计算密度谱
        g, q = periodogram(x, scaling='density')
        # 断言频率 f 等于从 0 到 0.5 等间隔采样的数组
        assert_allclose(f, np.linspace(0, 0.5, 9))
        # 断言计算得到的功率谱 p 等于计算得到的密度谱 q 除以 16.0
        assert_allclose(p, q/16.0)

    # 测试函数：test_integer_even
    def test_integer_even(self):
        # 创建长度为 16 的零数组，并将第一个元素设为 1，数据类型为整型
        x = np.zeros(16, dtype=int)
        x[0] = 1
        # 调用 periodogram 函数计算信号 x 的功率谱
        f, p = periodogram(x)
        # 断言频率 f 等于从 0 到 0.5 等间隔采样的数组
        assert_allclose(f, np.linspace(0, 0.5, 9))
        # 创建期望功率谱数组 q，首尾特殊处理，然后除以 8
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        # 断言计算得到的功率谱 p 等于期望功率谱 q
        assert_allclose(p, q)

    # 测试函数：test_integer_odd
    def test_integer_odd(self):
        # 创建长度为 15 的零数组，并将第一个元素设为 1，数据类型为整型
        x = np.zeros(15, dtype=int)
        x[0] = 1
        # 调用 periodogram 函数计算信号 x 的功率谱
        f, p = periodogram(x)
        # 断言频率 f 等于 np.arange(8.0)/15.0
        assert_allclose(f, np.arange(8.0)/15.0)
        # 创建期望功率谱数组 q，首位特殊处理，然后乘以 2/15
        q = np.ones(8)
        q[0] = 0
        q *= 2.0/15.0
        # 断言计算得到的功率谱 p 等于期望功率谱 q，设置容忍度为 1e-15
        assert_allclose(p, q, atol=1e-15)

    # 测试函数：test_integer_twosided
    def test_integer_twosided(self):
        # 创建长度为 16 的零数组，并将第一个元素设为 1，数据类型为整型
        x = np.zeros(16, dtype=int)
        x[0]
    # 测试复数数组的周期图计算
    def test_complex(self):
        # 创建一个复数类型的全零数组
        x = np.zeros(16, np.complex128)
        # 在数组的第一个元素位置设置一个复数值
        x[0] = 1.0 + 2.0j
        # 计算数组的周期图，不限定为单边频谱
        f, p = periodogram(x, return_onesided=False)
        # 断言频率数组与使用 fftfreq 函数生成的数组相似
        assert_allclose(f, fftfreq(16, 1.0))
        # 创建一个全为 5/16 的数组，并在第一个位置置零
        q = np.full(16, 5.0/16.0)
        q[0] = 0
        # 断言周期图与预期数组 p 相似
        assert_allclose(p, q)

    # 测试在未知的缩放选项下是否会引发 ValueError
    def test_unk_scaling(self):
        # 断言期望引发 ValueError 的情况
        assert_raises(ValueError, periodogram, np.zeros(4, np.complex128),
                      scaling='foo')

    # 根据系统位数跳过测试，出于对某些 32 位系统的容差问题
    @pytest.mark.skipif(
        sys.maxsize <= 2**32,
        reason="On some 32-bit tolerance issue"
    )
    def test_nd_axis_m1(self):
        # 创建一个浮点类型的全零数组，并将其重新形状为 (2, 1, 10)
        x = np.zeros(20, dtype=np.float64)
        x = x.reshape((2, 1, 10))
        # 在数组的第一个平面的所有切片上设置值为 1.0
        x[:,:,0] = 1.0
        # 计算数组的周期图
        f, p = periodogram(x)
        # 断言周期图的形状为 (2, 1, 6)
        assert_array_equal(p.shape, (2, 1, 6))
        # 断言第一个平面与第二个平面的周期图在 60 个最低单位位置上几乎相等
        assert_array_almost_equal_nulp(p[0,0,:], p[1,0,:], 60)
        # 计算第一个平面的周期图
        f0, p0 = periodogram(x[0,0,:])
        # 断言第一个平面的周期图与第二个平面的一致性在 60 个最低单位位置上几乎相等
        assert_array_almost_equal_nulp(p0[np.newaxis,:], p[1,:], 60)

    # 根据系统位数跳过测试，出于对某些 32 位系统的容差问题
    @pytest.mark.skipif(
        sys.maxsize <= 2**32,
        reason="On some 32-bit tolerance issue"
    )
    def test_nd_axis_0(self):
        # 创建一个浮点类型的全零数组，并将其重新形状为 (10, 2, 1)
        x = np.zeros(20, dtype=np.float64)
        x = x.reshape((10, 2, 1))
        # 在数组的第一个轴上的所有切片设置值为 1.0
        x[0,:,:] = 1.0
        # 计算数组在第一个轴上的周期图
        f, p = periodogram(x, axis=0)
        # 断言周期图的形状为 (6, 2, 1)
        assert_array_equal(p.shape, (6, 2, 1))
        # 断言第一个轴上第一个和第二个切片的周期图在 60 个最低单位位置上几乎相等
        assert_array_almost_equal_nulp(p[:,0,0], p[:,1,0], 60)
        # 计算第一个轴上第一个切片的周期图
        f0, p0 = periodogram(x[:,0,0])
        # 断言第一个轴上第一个切片的周期图与第二个切片的一致性在 60 个最低单位位置上几乎相等
        assert_array_almost_equal_nulp(p0, p[:,1,0])

    # 测试使用外部窗口函数
    def test_window_external(self):
        # 创建一个全零数组，并将其第一个元素设置为 1
        x = np.zeros(16)
        x[0] = 1
        # 计算数组的周期图，使用 'hann' 窗口长度为 10
        f, p = periodogram(x, 10, 'hann')
        # 获取 'hann' 窗口函数
        win = signal.get_window('hann', 16)
        # 计算使用 'hann' 窗口的数组的周期图
        fe, pe = periodogram(x, 10, win)
        # 断言周期图 p 与使用 'hann' 窗口的 pe 几乎相等
        assert_array_almost_equal_nulp(p, pe)
        # 断言频率数组 f 与使用 'hann' 窗口的 fe 几乎相等
        assert_array_almost_equal_nulp(f, fe)
        # 获取长度为 32 的 'hann' 窗口函数
        win_err = signal.get_window('hann', 32)
        # 断言期望引发 ValueError，因为窗口比信号长
        assert_raises(ValueError, periodogram, x, 10, win_err)

    # 测试填充后的 FFT
    def test_padded_fft(self):
        # 创建一个全零数组，并将其第一个元素设置为 1
        x = np.zeros(16)
        x[0] = 1
        # 计算数组的周期图
        f, p = periodogram(x)
        # 计算填充后的 FFT
        fp, pp = periodogram(x, nfft=32)
        # 断言频率数组 f 与填充后的 fp 的相等性（每隔一个取值）
        assert_allclose(f, fp[::2])
        # 断言周期图 p 与填充后的 pp 的相等性（每隔一个取值）
        assert_allclose(p, pp[::2])
        # 断言填充后的 pp 的形状为 (17,)
        assert_array_equal(pp.shape, (17,))

    # 测试空输入情况
    def test_empty_input(self):
        # 计算空数组的周期图
        f, p = periodogram([])
        # 断言空数组的频率数组形状为 (0,)
        assert_array_equal(f.shape, (0,))
        # 断言空数组的周期图形状为 (0,)
        assert_array_equal(p.shape, (0,))
        # 对不同形状的空数组进行循环测试
        for shape in [(0,), (3,0), (0,5,2)]:
            f, p = periodogram(np.empty(shape))
            # 断言空数组的周期图形状与数组形状相同
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

    # 测试空输入情况下的其它轴
    def test_empty_input_other_axis(self):
        # 对不同形状的空数组进行循环测试，指定轴为 1
        for shape in [(3,0), (0,5,2)]:
            f, p = periodogram(np.empty(shape), axis=1)
            # 断言空数组的频率数组形状与数组形状相同
            assert_array_equal(f.shape, shape)
            # 断言空数组的周期图形状与数组形状相同
            assert_array_equal(p.shape, shape)

    # 测试短 nfft 的情况
    def test_short_nfft(self):
        # 创建一个全零数组，并将其第一个元素设置为 1
        x = np.zeros(18)
        x[0] = 1
        # 计算数组的周期图，指定 nfft 为 16
        f, p = periodogram(x, nfft=16)
        # 断言频率数组 f 与期望的线性空间值相似
        assert_allclose(f, np.linspace(0, 0.5, 9))
        # 创建一个期望的周期图数组
        q = np.ones(9)
        q[0] = 0
    # 测试函数：验证使用指定长度的 FFT 进行周期图计算时的正确性
    def test_nfft_is_xshape(self):
        # 创建一个长度为 16 的零向量 x，并将第一个元素设置为 1
        x = np.zeros(16)
        x[0] = 1
        # 调用 periodogram 函数计算周期图 f 和功率谱 p，指定 FFT 长度为 16
        f, p = periodogram(x, nfft=16)
        # 验证频率向量 f 是否与预期的线性空间匹配
        assert_allclose(f, np.linspace(0, 0.5, 9))
        # 创建一个预期的功率谱向量 q，用于验证 p 的正确性
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        # 验证计算得到的功率谱 p 是否与预期的 q 向量匹配
        assert_allclose(p, q)

    # 测试函数：验证使用默认参数计算单边实数频谱的功率谱
    def test_real_onesided_even_32(self):
        # 创建一个长度为 16 的零向量 x，并将第一个元素设置为 1，数据类型为单精度浮点数
        x = np.zeros(16, 'f')
        x[0] = 1
        # 调用 periodogram 函数计算周期图 f 和功率谱 p，使用默认的 FFT 长度
        f, p = periodogram(x)
        # 验证频率向量 f 是否与预期的线性空间匹配
        assert_allclose(f, np.linspace(0, 0.5, 9))
        # 创建一个预期的功率谱向量 q，用于验证 p 的正确性
        q = np.ones(9, 'f')
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        # 验证计算得到的功率谱 p 是否与预期的 q 向量匹配，并且验证 p 的数据类型
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)

    # 测试函数：验证使用默认参数计算单边实数频谱的功率谱，输入向量长度为奇数
    def test_real_onesided_odd_32(self):
        # 创建一个长度为 15 的零向量 x，并将第一个元素设置为 1，数据类型为单精度浮点数
        x = np.zeros(15, 'f')
        x[0] = 1
        # 调用 periodogram 函数计算周期图 f 和功率谱 p，使用默认的 FFT 长度
        f, p = periodogram(x)
        # 验证频率向量 f 是否与预期的线性空间匹配
        assert_allclose(f, np.arange(8.0)/15.0)
        # 创建一个预期的功率谱向量 q，用于验证 p 的正确性
        q = np.ones(8, 'f')
        q[0] = 0
        # 计算预期功率谱 q 的值，并验证 p 的正确性（使用给定的绝对误差容忍度）
        q *= 2.0/15.0
        assert_allclose(p, q, atol=1e-7)
        # 验证 p 的数据类型是否与 q 的数据类型匹配
        assert_(p.dtype == q.dtype)

    # 测试函数：验证使用返回双边频谱的 FFT 计算的功率谱
    def test_real_twosided_32(self):
        # 创建一个长度为 16 的零向量 x，并将第一个元素设置为 1，数据类型为单精度浮点数
        x = np.zeros(16, 'f')
        x[0] = 1
        # 调用 periodogram 函数计算周期图 f 和功率谱 p，指定返回双边频谱
        f, p = periodogram(x, return_onesided=False)
        # 验证频率向量 f 是否与预期的频率数组匹配
        assert_allclose(f, fftfreq(16, 1.0))
        # 创建一个预期的功率谱向量 q，用于验证 p 的正确性
        q = np.full(16, 1/16.0, 'f')
        q[0] = 0
        # 验证计算得到的功率谱 p 是否与预期的 q 向量匹配，并且验证 p 的数据类型
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)

    # 测试函数：验证使用复数输入计算功率谱
    def test_complex_32(self):
        # 创建一个长度为 16 的零向量 x，并将第一个元素设置为复数 1.0 + 2.0j，数据类型为单精度复数
        x = np.zeros(16, 'F')
        x[0] = 1.0 + 2.0j
        # 调用 periodogram 函数计算周期图 f 和功率谱 p，指定返回双边频谱
        f, p = periodogram(x, return_onesided=False)
        # 验证频率向量 f 是否与预期的频率数组匹配
        assert_allclose(f, fftfreq(16, 1.0))
        # 创建一个预期的功率谱向量 q，用于验证 p 的正确性
        q = np.full(16, 5.0/16.0, 'f')
        q[0] = 0
        # 验证计算得到的功率谱 p 是否与预期的 q 向量匹配，并且验证 p 的数据类型
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)

    # 测试函数：验证当窗口长度短于输入长度时，期望引发 ValueError 异常
    def test_shorter_window_error(self):
        # 创建一个长度为 16 的零向量 x，并将第一个元素设置为 1
        x = np.zeros(16)
        x[0] = 1
        # 获取 'hann' 窗口的长度为 10
        win = signal.get_window('hann', 10)
        # 定义预期的错误消息
        expected_msg = ('the size of the window must be the same size '
                        'of the input on the specified axis')
        # 使用 assert_raises 验证期望的 ValueError 异常是否被正确引发，并且错误消息匹配预期消息
        with assert_raises(ValueError, match=expected_msg):
            periodogram(x, window=win)
class TestWelch:
    # 定义测试类 TestWelch

    def test_real_onesided_even(self):
        # 测试函数：测试实数信号，单边谱，长度为偶数
        x = np.zeros(16)
        # 创建长度为16的零数组
        x[0] = 1
        # 在数组第一个位置赋值为1
        x[8] = 1
        # 在数组第九个位置赋值为1
        f, p = welch(x, nperseg=8)
        # 使用 Welch 方法计算信号 x 的功率谱密度估计值 f 和功率谱 p
        assert_allclose(f, np.linspace(0, 0.5, 5))
        # 断言：f 应当接近等差数列，范围从 0 到 0.5，共5个点
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111])
        # 期望的功率谱密度估计值
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言：计算得到的功率谱 p 应当接近期望值 q

    def test_real_onesided_odd(self):
        # 测试函数：测试实数信号，单边谱，长度为奇数
        x = np.zeros(16)
        # 创建长度为16的零数组
        x[0] = 1
        # 在数组第一个位置赋值为1
        x[8] = 1
        # 在数组第九个位置赋值为1
        f, p = welch(x, nperseg=9)
        # 使用 Welch 方法计算信号 x 的功率谱密度估计值 f 和功率谱 p
        assert_allclose(f, np.arange(5.0)/9.0)
        # 断言：f 应当接近等差数列，范围从 0 到 1/2，共5个点
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113,
                      0.17072113])
        # 期望的功率谱密度估计值
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言：计算得到的功率谱 p 应当接近期望值 q

    def test_real_twosided(self):
        # 测试函数：测试实数信号，双边谱
        x = np.zeros(16)
        # 创建长度为16的零数组
        x[0] = 1
        # 在数组第一个位置赋值为1
        x[8] = 1
        # 在数组第九个位置赋值为1
        f, p = welch(x, nperseg=8, return_onesided=False)
        # 使用 Welch 方法计算信号 x 的功率谱密度估计值 f 和功率谱 p（双边）
        assert_allclose(f, fftfreq(8, 1.0))
        # 断言：f 应当接近频率数组，长度为8，采样频率为1.0
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.07638889])
        # 期望的功率谱密度估计值
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言：计算得到的功率谱 p 应当接近期望值 q

    def test_real_spectrum(self):
        # 测试函数：测试实数信号，使用 'spectrum' 缩放
        x = np.zeros(16)
        # 创建长度为16的零数组
        x[0] = 1
        # 在数组第一个位置赋值为1
        x[8] = 1
        # 在数组第九个位置赋值为1
        f, p = welch(x, nperseg=8, scaling='spectrum')
        # 使用 Welch 方法计算信号 x 的功率谱密度估计值 f 和功率谱 p（使用 'spectrum' 缩放）
        assert_allclose(f, np.linspace(0, 0.5, 5))
        # 断言：f 应当接近等差数列，范围从 0 到 0.5，共5个点
        q = np.array([0.015625, 0.02864583, 0.04166667, 0.04166667,
                      0.02083333])
        # 期望的功率谱密度估计值
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言：计算得到的功率谱 p 应当接近期望值 q

    def test_integer_onesided_even(self):
        # 测试函数：测试整数信号，单边谱，长度为偶数
        x = np.zeros(16, dtype=int)
        # 创建长度为16的零数组，数据类型为整数
        x[0] = 1
        # 在数组第一个位置赋值为1
        x[8] = 1
        # 在数组第九个位置赋值为1
        f, p = welch(x, nperseg=8)
        # 使用 Welch 方法计算信号 x 的功率谱密度估计值 f 和功率谱 p
        assert_allclose(f, np.linspace(0, 0.5, 5))
        # 断言：f 应当接近等差数列，范围从 0 到 0.5，共5个点
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111])
        # 期望的功率谱密度估计值
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言：计算得到的功率谱 p 应当接近期望值 q

    def test_integer_onesided_odd(self):
        # 测试函数：测试整数信号，单边谱，长度为奇数
        x = np.zeros(16, dtype=int)
        # 创建长度为16的零数组，数据类型为整数
        x[0] = 1
        # 在数组第一个位置赋值为1
        x[8] = 1
        # 在数组第九个位置赋值为1
        f, p = welch(x, nperseg=9)
        # 使用 Welch 方法计算信号 x 的功率谱密度估计值 f 和功率谱 p
        assert_allclose(f, np.arange(5.0)/9.0)
        # 断言：f 应当接近等差数列，范围从 0 到 1/2，共5个点
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113,
                      0.17072113])
        # 期望的功率谱密度估计值
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言：计算得到的功率谱 p 应当接近期望值 q

    def test_integer_twosided(self):
        # 测试函数：测试整数信号，双边谱
        x = np.zeros(16, dtype=int)
        # 创建长度为16的零数组，数据类型为整数
        x[0] = 1
        # 在数组第一个位置赋值为1
        x[8] = 1
        # 在数组第九个位置赋值为1
        f, p = welch(x, nperseg=8, return_onesided=False)
        # 使用 Welch 方法计算信号 x 的功率谱密度估计值
    # 测试异常情况：使用无效的缩放参数 'foo' 调用 welch 函数，预期会引发 ValueError 异常
    def test_unk_scaling(self):
        assert_raises(ValueError, welch, np.zeros(4, np.complex128),
                      scaling='foo', nperseg=4)

    # 测试线性趋势去除的效果：生成一个浮点数数组 x，对其进行 Welch 方法频谱估计，并应用线性趋势去除
    # 预期频谱 p 应该接近全零数组，允许误差为 1e-15
    def test_detrend_linear(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f, p = welch(x, nperseg=10, detrend='linear')
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    # 测试不进行趋势去除的情况下 Welch 方法的效果：生成一个浮点数数组 x，分别使用 detrend=False
    # 和自定义的 detrend=lambda x: x 参数进行频谱估计，预期结果 f1 和 f2 应该非常接近，允许误差为 1e-15
    def test_no_detrending(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f1, p1 = welch(x, nperseg=10, detrend=False)
        f2, p2 = welch(x, nperseg=10, detrend=lambda x: x)
        assert_allclose(f1, f2, atol=1e-15)
        assert_allclose(p1, p2, atol=1e-15)

    # 测试使用外部定义的线性趋势去除函数的效果：生成一个浮点数数组 x，对其进行 Welch 方法频谱估计
    # 使用自定义的 detrend=lambda seg: signal.detrend(seg, type='l') 参数进行趋势去除
    # 预期频谱 p 应该接近全零数组，允许误差为 1e-15
    def test_detrend_external(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f, p = welch(x, nperseg=10,
                     detrend=lambda seg: signal.detrend(seg, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    # 测试在多维数组中使用外部定义的线性趋势去除函数的效果：生成一个浮点数数组 x，reshape 成 (2, 2, 10)
    # 对其进行 Welch 方法频谱估计，使用自定义的 detrend=lambda seg: signal.detrend(seg, type='l') 参数进行趋势去除
    # 预期频谱 p 应该接近全零数组，允许误差为 1e-15
    def test_detrend_external_nd_m1(self):
        x = np.arange(40, dtype=np.float64) + 0.04
        x = x.reshape((2, 2, 10))
        f, p = welch(x, nperseg=10,
                     detrend=lambda seg: signal.detrend(seg, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    # 测试在多维数组中使用外部定义的线性趋势去除函数的效果：生成一个浮点数数组 x，reshape 成 (2, 1, 10)
    # 并对其进行轴移操作，对第 0 轴使用自定义的 detrend=lambda seg: signal.detrend(seg, axis=0, type='l') 参数进行趋势去除
    # 预期频谱 p 应该接近全零数组，允许误差为 1e-15
    def test_detrend_external_nd_0(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2, 1, 10))
        x = np.moveaxis(x, 2, 0)
        f, p = welch(x, nperseg=10, axis=0,
                     detrend=lambda seg: signal.detrend(seg, axis=0, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    # 测试在多维数组中的特定轴（-1）上进行 Welch 方法频谱估计的效果：生成一个浮点数数组 x，reshape 成 (2, 1, 10)
    # 预期频谱 p 的形状应该是 (2, 1, 6)，并且第一个和第二个数组的频谱应该非常接近，允许误差为 1e-13
    # 进一步测试对第一个子数组进行频谱估计，预期结果应该与第二个子数组的频谱非常接近，允许误差为 1e-13
    def test_nd_axis_m1(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2, 1, 10))
        f, p = welch(x, nperseg=10)
        assert_array_equal(p.shape, (2, 1, 6))
        assert_allclose(p[0, 0, :], p[1, 0, :], atol=1e-13, rtol=1e-13)
        f0, p0 = welch(x[0, 0, :], nperseg=10)
        assert_allclose(p0[np.newaxis, :], p[1, :], atol=1e-13, rtol=1e-13)

    # 测试在多维数组中的特定轴（0）上进行 Welch 方法频谱估计的效果：生成一个浮点数数组 x，reshape 成 (10, 2, 1)
    # 预期频谱 p 的形状应该是 (6, 2, 1)，并且第一列和第二列的频谱应该非常接近，允许误差为 1e-13
    # 进一步测试对第一个列进行频谱估计，预期结果应该与第二个列的频谱非常接近，允许误差为 1e-13
    def test_nd_axis_0(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((10, 2, 1))
        f, p = welch(x, nperseg=10, axis=0)
        assert_array_equal(p.shape, (6, 2, 1))
        assert_allclose(p[:, 0, 0], p[:, 1, 0], atol=1e-13, rtol=1e-13)
        f0, p0 = welch(x[:, 0, 0], nperseg=10)
        assert_allclose(p0, p[:, 1, 0], atol=1e-13, rtol=1e-13)
    def test_window_external(self):
        # 创建一个长度为16的零数组
        x = np.zeros(16)
        # 设置数组第一个和第九个元素为1
        x[0] = 1
        x[8] = 1
        # 使用Welch方法计算信号的功率谱密度估计值
        f, p = welch(x, 10, 'hann', nperseg=8)
        # 获取'Hann'窗口函数并赋值给变量win
        win = signal.get_window('hann', 8)
        # 使用Welch方法计算信号的功率谱密度估计值，指定窗口函数为win
        fe, pe = welch(x, 10, win, nperseg=None)
        # 断言两组功率谱密度估计值在数值上几乎相等
        assert_array_almost_equal_nulp(p, pe)
        assert_array_almost_equal_nulp(f, fe)
        # 断言fe的形状为(5,)，因为使用win的长度作为nperseg
        assert_array_equal(fe.shape, (5,))
        # 断言pe的形状为(5,)
        assert_array_equal(pe.shape, (5,))
        # 断言当nperseg=4时会引发ValueError异常，因为nperseg与win的长度不同
        assert_raises(ValueError, welch, x,
                      10, win, nperseg=4)
        # 获取长度为32的'Hann'窗口函数并赋值给win_err
        win_err = signal.get_window('hann', 32)
        # 断言当使用win_err作为窗口函数时会引发ValueError异常，因为窗口长度大于信号长度
        assert_raises(ValueError, welch, x,
                      10, win_err, nperseg=None)

    def test_empty_input(self):
        # 对空输入进行功率谱密度估计，返回空数组
        f, p = welch([])
        # 断言f的形状为(0,)
        assert_array_equal(f.shape, (0,))
        # 断言p的形状为(0,)
        assert_array_equal(p.shape, (0,))
        # 对多种空形状的输入进行功率谱密度估计，返回相同形状的空数组
        for shape in [(0,), (3, 0), (0, 5, 2)]:
            f, p = welch(np.empty(shape))
            # 断言f的形状与shape相同
            assert_array_equal(f.shape, shape)
            # 断言p的形状与shape相同
            assert_array_equal(p.shape, shape)

    def test_empty_input_other_axis(self):
        # 对指定轴上的空输入进行功率谱密度估计，返回相同形状的空数组
        for shape in [(3, 0), (0, 5, 2)]:
            f, p = welch(np.empty(shape), axis=1)
            # 断言f的形状与shape相同
            assert_array_equal(f.shape, shape)
            # 断言p的形状与shape相同
            assert_array_equal(p.shape, shape)

    def test_short_data(self):
        # 创建一个长度为8的零数组，设置第一个元素为1
        x = np.zeros(8)
        x[0] = 1
        # 使用Welch方法计算信号的功率谱密度估计值，对于使用字符串表示的窗口函数，当输入信号长度小于nperseg时会产生UserWarning，设置nperseg为x.shape[-1]
        with suppress_warnings() as sup:
            msg = "nperseg = 256 is greater than input length  = 8, using nperseg = 8"
            sup.filter(UserWarning, msg)
            # 使用默认nperseg进行功率谱密度估计
            f, p = welch(x, window='hann')
            # 使用用户指定的nperseg进行功率谱密度估计
            f1, p1 = welch(x, window='hann', nperseg=256)
        # 使用有效的nperseg进行功率谱密度估计
        f2, p2 = welch(x, nperseg=8)
        # 断言所有结果的频率f与功率谱密度p在数值上几乎相等
        assert_allclose(f, f2)
        assert_allclose(p, p2)
        assert_allclose(f1, f2)
        assert_allclose(p1, p2)

    def test_window_long_or_nd(self):
        # 断言当窗口函数长度大于信号长度时会引发ValueError异常
        assert_raises(ValueError, welch, np.zeros(4), 1, np.array([1, 1, 1, 1, 1]))
        # 断言当窗口函数为多维数组时会引发ValueError异常
        assert_raises(ValueError, welch, np.zeros(4), 1,
                      np.arange(6).reshape((2, 3)))

    def test_nondefault_noverlap(self):
        # 创建一个长度为64的零数组，每隔8个元素设置一个为1
        x = np.zeros(64)
        x[::8] = 1
        # 使用Welch方法计算信号的功率谱密度估计值，指定nperseg和noverlap参数
        f, p = welch(x, nperseg=16, noverlap=4)
        # 断言功率谱密度估计结果与预期值在数值上几乎相等，设置绝对误差tolerance为1e-12
        q = np.array([0, 1./12., 1./3., 1./5., 1./3., 1./5., 1./3., 1./5.,
                      1./6.])
        assert_allclose(p, q, atol=1e-12)

    def test_bad_noverlap(self):
        # 断言当指定的noverlap不合法时会引发ValueError异常
        assert_raises(ValueError, welch, np.zeros(4), 1, 'hann', 2, 7)

    def test_nfft_too_short(self):
        # 断言当指定的nfft参数值过小时会引发ValueError异常
        assert_raises(ValueError, welch, np.ones(12), nfft=3, nperseg=4)
    # 测试实数信号的单边频谱估计（样本数为偶数）
    def test_real_onesided_even_32(self):
        # 创建一个长度为16的单精度浮点数组，并初始化为0
        x = np.zeros(16, 'f')
        # 设置数组的第一个和第九个元素为1
        x[0] = 1
        x[8] = 1
        # 使用 Welch 方法计算信号的功率谱密度估计值
        f, p = welch(x, nperseg=8)
        # 断言频率轴上的值与预期值在数值上相近
        assert_allclose(f, np.linspace(0, 0.5, 5))
        # 预期功率谱密度与实际计算的功率谱密度在数值上相近
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言功率谱密度的数据类型与预期的数据类型一致
        assert_(p.dtype == q.dtype)

    # 测试实数信号的单边频谱估计（样本数为奇数）
    def test_real_onesided_odd_32(self):
        # 创建一个长度为16的单精度浮点数组，并初始化为0
        x = np.zeros(16, 'f')
        # 设置数组的第一个和第九个元素为1
        x[0] = 1
        x[8] = 1
        # 使用 Welch 方法计算信号的功率谱密度估计值
        f, p = welch(x, nperseg=9)
        # 断言频率轴上的值与预期值在数值上相近
        assert_allclose(f, np.arange(5.0)/9.0)
        # 预期功率谱密度与实际计算的功率谱密度在数值上相近
        q = np.array([0.12477458, 0.23430935, 0.17072113, 0.17072116,
                      0.17072113], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言功率谱密度的数据类型与预期的数据类型一致
        assert_(p.dtype == q.dtype)

    # 测试实数信号的双边频谱估计
    def test_real_twosided_32(self):
        # 创建一个长度为16的单精度浮点数组，并初始化为0
        x = np.zeros(16, 'f')
        # 设置数组的第一个和第九个元素为1
        x[0] = 1
        x[8] = 1
        # 使用 Welch 方法计算信号的功率谱密度估计值（双边频谱）
        f, p = welch(x, nperseg=8, return_onesided=False)
        # 断言频率轴上的值与预期值在数值上相近
        assert_allclose(f, fftfreq(8, 1.0))
        # 预期功率谱密度与实际计算的功率谱密度在数值上相近
        q = np.array([0.08333333, 0.07638889, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.11111111,
                      0.07638889], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言功率谱密度的数据类型与预期的数据类型一致
        assert_(p.dtype == q.dtype)

    # 测试复数信号的双边频谱估计
    def test_complex_32(self):
        # 创建一个长度为16的单精度复数数组，并初始化为0
        x = np.zeros(16, 'F')
        # 设置数组的第一个和第九个元素为1+2j
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        # 使用 Welch 方法计算信号的功率谱密度估计值（双边频谱）
        f, p = welch(x, nperseg=8, return_onesided=False)
        # 断言频率轴上的值与预期值在数值上相近
        assert_allclose(f, fftfreq(8, 1.0))
        # 预期功率谱密度与实际计算的功率谱密度在数值上相近
        q = np.array([0.41666666, 0.38194442, 0.55555552, 0.55555552,
                      0.55555558, 0.55555552, 0.55555552, 0.38194442], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言功率谱密度的数据类型与预期的数据类型一致，如果不一致则输出错误信息
        assert_(p.dtype == q.dtype,
                f'dtype mismatch, {p.dtype}, {q.dtype}')

    # 测试填充后的频谱值
    def test_padded_freqs(self):
        # 创建一个长度为12的零数组
        x = np.zeros(12)

        # 设置 FFT 使用的点数为24
        nfft = 24
        # 计算 FFT 的频率轴并保留非负频率部分
        f = fftfreq(nfft, 1.0)[:nfft//2+1]
        # 将频率轴的最后一个频率值取反
        f[-1] *= -1
        # 使用 Welch 方法计算信号的功率谱密度估计值（样本数为5，使用24点 FFT）
        fodd, _ = welch(x, nperseg=5, nfft=nfft)
        # 使用 Welch 方法计算信号的功率谱密度估计值（样本数为6，使用24点 FFT）
        feven, _ = welch(x, nperseg=6, nfft=nfft)
        # 断言频率轴上的值与预期值在数值上相近
        assert_allclose(f, fodd)
        assert_allclose(f, feven)

        # 设置 FFT 使用的点数为25
        nfft = 25
        # 计算 FFT 的频率轴并保留非负频率部分
        f = fftfreq(nfft, 1.0)[:(nfft + 1)//2]
        # 使用 Welch 方法计算信号的功率谱密度估计值（样本数为5，使用25点 FFT）
        fodd, _ = welch(x, nperseg=5, nfft=nfft)
        # 使用 Welch 方法计算信号的功率谱密度估计值（样本数为6，使用25点 FFT）
        feven, _ = welch(x, nperseg=6, nfft=nfft)
        # 断言频率轴上的值与预期值在数值上相近
        assert_allclose(f, fodd)
        assert_allclose(f, feven)
    def test_window_correction(self):
        A = 20  # 设置振幅为20
        fs = 1e4  # 设置采样频率为10,000 Hz
        nperseg = int(fs//10)  # 计算每段长度为采样频率的十分之一的整数值
        fsig = 300  # 设置信号频率为300 Hz
        ii = int(fsig*nperseg//fs)  # 计算信号频率在频谱中的索引

        tt = np.arange(fs)/fs  # 生成从0到1秒的时间序列
        x = A*np.sin(2*np.pi*fsig*tt)  # 生成频率为fsig的正弦信号

        for window in ['hann', 'bartlett', ('tukey', 0.1), 'flattop']:
            _, p_spec = welch(x, fs=fs, nperseg=nperseg, window=window,
                              scaling='spectrum')  # 计算信号的功率谱密度
            freq, p_dens = welch(x, fs=fs, nperseg=nperseg, window=window,
                                 scaling='density')  # 计算信号的功率谱密度

            # 检查频谱中信号频率处的峰值高度，用于'spectrum'模式
            assert_allclose(p_spec[ii], A**2/2.0)
            # 检查密度谱的积分平均值是否等于期望值，用于'density'模式
            assert_allclose(np.sqrt(trapezoid(p_dens, freq)), A*np.sqrt(2)/2,
                            rtol=1e-3)

    def test_axis_rolling(self):
        np.random.seed(1234)

        x_flat = np.random.randn(1024)  # 生成1024个正态分布随机数
        _, p_flat = welch(x_flat)  # 计算平均功率谱

        for a in range(3):
            newshape = [1,]*3
            newshape[a] = -1
            x = x_flat.reshape(newshape)  # 改变数组形状为指定维度

            _, p_plus = welch(x, axis=a)  # 计算指定正轴索引的功率谱
            _, p_minus = welch(x, axis=a-x.ndim)  # 计算指定负轴索引的功率谱

            assert_equal(p_flat, p_plus.squeeze(), err_msg=a)
            assert_equal(p_flat, p_minus.squeeze(), err_msg=a-x.ndim)

    def test_average(self):
        x = np.zeros(16)  # 创建长度为16的零数组
        x[0] = 1
        x[8] = 1  # 在数组中设置两个非零值
        f, p = welch(x, nperseg=8, average='median')  # 计算中值平均的功率谱
        assert_allclose(f, np.linspace(0, 0.5, 5))  # 检查频率数组是否符合预期
        q = np.array([.1, .05, 0., 1.54074396e-33, 0.])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)  # 检查功率谱是否与期望值接近

        assert_raises(ValueError, welch, x, nperseg=8,
                      average='unrecognised-average')  # 测试不支持的平均模式的异常处理
class TestCSD:
    # 测试函数，验证在 x 比 y 短时的 csd 函数行为
    def test_pad_shorter_x(self):
        # 创建长度为 8 的全零数组 x
        x = np.zeros(8)
        # 创建长度为 12 的全零数组 y
        y = np.zeros(12)

        # 生成从 0 到 0.5 的 7 个等间距数列作为频率
        f = np.linspace(0, 0.5, 7)
        # 创建复数数组 c，长度为 7
        c = np.zeros(7, dtype=np.complex128)
        # 调用 csd 函数计算 x 和 y 的交叉功率谱密度
        f1, c1 = csd(x, y, nperseg=12)

        # 断言 f1 与预期的 f 数组近似相等
        assert_allclose(f, f1)
        # 断言 c1 与预期的 c 数组近似相等
        assert_allclose(c, c1)

    # 测试函数，验证在 y 比 x 短时的 csd 函数行为
    def test_pad_shorter_y(self):
        # 创建长度为 12 的全零数组 x
        x = np.zeros(12)
        # 创建长度为 8 的全零数组 y
        y = np.zeros(8)

        # 生成从 0 到 0.5 的 7 个等间距数列作为频率
        f = np.linspace(0, 0.5, 7)
        # 创建复数数组 c，长度为 7
        c = np.zeros(7, dtype=np.complex128)
        # 调用 csd 函数计算 x 和 y 的交叉功率谱密度
        f1, c1 = csd(x, y, nperseg=12)

        # 断言 f1 与预期的 f 数组近似相等
        assert_allclose(f, f1)
        # 断言 c1 与预期的 c 数组近似相等
        assert_allclose(c, c1)

    # 测试函数，验证实数信号的单边功率谱密度估计（长度为偶数）
    def test_real_onesided_even(self):
        # 创建长度为 16 的全零整数数组 x
        x = np.zeros(16)
        # 设置数组 x 的前两个元素为 1
        x[0] = 1
        x[8] = 1
        # 调用 csd 函数计算 x 与自身的交叉功率谱密度
        f, p = csd(x, x, nperseg=8)
        # 断言 f 与预期的频率数列近似相等
        assert_allclose(f, np.linspace(0, 0.5, 5))
        # 预期的功率密度数组 q
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222, 0.11111111])
        # 断言 p 与预期的 q 数组近似相等，允许误差范围为 1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    # 测试函数，验证实数信号的单边功率谱密度估计（长度为奇数）
    def test_real_onesided_odd(self):
        # 创建长度为 16 的全零整数数组 x
        x = np.zeros(16)
        # 设置数组 x 的前两个元素为 1
        x[0] = 1
        x[8] = 1
        # 调用 csd 函数计算 x 与自身的交叉功率谱密度，窗口长度为 9
        f, p = csd(x, x, nperseg=9)
        # 断言 f 与预期的频率数列近似相等
        assert_allclose(f, np.arange(5.0) / 9.0)
        # 预期的功率密度数组 q
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113, 0.17072113])
        # 断言 p 与预期的 q 数组近似相等，允许误差范围为 1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    # 测试函数，验证实数信号的双边功率谱密度估计
    def test_real_twosided(self):
        # 创建长度为 16 的全零整数数组 x
        x = np.zeros(16)
        # 设置数组 x 的前两个元素为 1
        x[0] = 1
        x[8] = 1
        # 调用 csd 函数计算 x 与自身的交叉功率谱密度，窗口长度为 8，双边估计
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        # 断言 f 与预期的频率数列近似相等，使用 fftfreq 生成
        assert_allclose(f, fftfreq(8, 1.0))
        # 预期的功率密度数组 q
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.07638889])
        # 断言 p 与预期的 q 数组近似相等，允许误差范围为 1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    # 测试函数，验证实数信号的频谱功率密度估计
    def test_real_spectrum(self):
        # 创建长度为 16 的全零整数数组 x
        x = np.zeros(16)
        # 设置数组 x 的前两个元素为 1
        x[0] = 1
        x[8] = 1
        # 调用 csd 函数计算 x 与自身的交叉功率谱密度，窗口长度为 8，使用 spectrum scaling
        f, p = csd(x, x, nperseg=8, scaling='spectrum')
        # 断言 f 与预期的频率数列近似相等
        assert_allclose(f, np.linspace(0, 0.5, 5))
        # 预期的功率密度数组 q
        q = np.array([0.015625, 0.02864583, 0.04166667, 0.04166667, 0.02083333])
        # 断言 p 与预期的 q 数组近似相等，允许误差范围为 1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    # 测试函数，验证整数信号的单边功率谱密度估计（长度为偶数）
    def test_integer_onesided_even(self):
        # 创建长度为 16 的全零整数数组 x
        x = np.zeros(16, dtype=int)
        # 设置数组 x 的前两个元素为 1
        x[0] = 1
        x[8] = 1
        # 调用 csd 函数计算 x 与自身的交叉功率谱密度，窗口长度为 8
        f, p = csd(x, x, nperseg=8)
        # 断言 f 与预期的频率数列近似相等
        assert_allclose(f, np.linspace(0, 0.5, 5))
        # 预期的功率密度数组 q
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222, 0.11111111])
        # 断言 p 与预期的 q 数组近似相等，允许误差范围为 1e-
    # 测试整数输入的双边交叉谱密度计算
    def test_integer_twosided(self):
        # 创建一个长度为16的整数数组，初始值全为0
        x = np.zeros(16, dtype=int)
        # 设置数组的第一个和第九个元素为1
        x[0] = 1
        x[8] = 1
        # 计算输入信号x自身的交叉谱密度，nperseg设置为8，返回双边结果
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        # 验证频率结果f与FFT频率值的一致性
        assert_allclose(f, fftfreq(8, 1.0))
        # 预期的功率谱密度q
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.07638889])
        # 验证计算得到的功率谱密度p与预期值q的一致性，允许误差范围在1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    # 测试复数输入的双边交叉谱密度计算
    def test_complex(self):
        # 创建一个长度为16的复数数组，初始值全为0
        x = np.zeros(16, np.complex128)
        # 设置数组的第一个和第九个元素为复数值1.0 + 2.0j
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        # 计算输入信号x自身的交叉谱密度，nperseg设置为8，返回双边结果
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        # 验证频率结果f与FFT频率值的一致性
        assert_allclose(f, fftfreq(8, 1.0))
        # 预期的功率谱密度q
        q = np.array([0.41666667, 0.38194444, 0.55555556, 0.55555556,
                      0.55555556, 0.55555556, 0.55555556, 0.38194444])
        # 验证计算得到的功率谱密度p与预期值q的一致性，允许误差范围在1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    # 测试未知缩放参数的异常处理
    def test_unk_scaling(self):
        # 断言调用csd函数时，传入未知的scaling参数值'foo'会引发ValueError异常
        assert_raises(ValueError, csd, np.zeros(4, np.complex128),
                      np.ones(4, np.complex128), scaling='foo', nperseg=4)

    # 测试线性去趋势处理的功率谱密度计算
    def test_detrend_linear(self):
        # 创建一个长度为10的浮点数数组，值从0.04开始递增
        x = np.arange(10, dtype=np.float64) + 0.04
        # 计算输入信号x自身的交叉谱密度，nperseg设置为10，使用线性去趋势处理
        f, p = csd(x, x, nperseg=10, detrend='linear')
        # 验证计算得到的功率谱密度p全为零，允许误差范围在1e-15
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    # 测试不进行去趋势处理的功率谱密度计算
    def test_no_detrending(self):
        # 创建一个长度为10的浮点数数组，值从0.04开始递增
        x = np.arange(10, dtype=np.float64) + 0.04
        # 计算输入信号x自身的交叉谱密度，nperseg设置为10，不进行去趋势处理
        f1, p1 = csd(x, x, nperseg=10, detrend=False)
        # 使用自定义的去趋势函数计算功率谱密度
        f2, p2 = csd(x, x, nperseg=10, detrend=lambda x: x)
        # 验证两种方法得到的频率f和功率谱密度p的一致性，允许误差范围在1e-15
        assert_allclose(f1, f2, atol=1e-15)
        assert_allclose(p1, p2, atol=1e-15)

    # 测试使用外部去趋势函数进行功率谱密度计算
    def test_detrend_external(self):
        # 创建一个长度为10的浮点数数组，值从0.04开始递增
        x = np.arange(10, dtype=np.float64) + 0.04
        # 计算输入信号x自身的交叉谱密度，nperseg设置为10，使用外部函数进行线性去趋势处理
        f, p = csd(x, x, nperseg=10,
                   detrend=lambda seg: signal.detrend(seg, type='l'))
        # 验证计算得到的功率谱密度p全为零，允许误差范围在1e-15
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    # 测试使用外部去趋势函数进行多维输入信号的功率谱密度计算
    def test_detrend_external_nd_m1(self):
        # 创建一个长度为40的浮点数数组，值从0.04开始递增，然后将其reshape为(2, 2, 10)
        x = np.arange(40, dtype=np.float64) + 0.04
        x = x.reshape((2, 2, 10))
        # 计算多维输入信号x自身的交叉谱密度，nperseg设置为10，使用外部函数进行线性去趋势处理
        f, p = csd(x, x, nperseg=10,
                   detrend=lambda seg: signal.detrend(seg, type='l'))
        # 验证计算得到的功率谱密度p全为零，允许误差范围在1e-15
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    # 测试使用外部去趋势函数进行多维输入信号的功率谱密度计算，对轴0进行处理
    def test_detrend_external_nd_0(self):
        # 创建一个长度为20的浮点数数组，值从0.04开始递增，然后将其reshape为(2, 1, 10)，并将轴2移动到轴0位置
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2, 1, 10))
        x = np.moveaxis(x, 2, 0)
        # 计算多维输入信号x自身的交叉谱密度，nperseg设置为10，对轴0使用外部函数进行线性去趋势处理
        f, p = csd(x, x, nperseg=10, axis=0,
                   detrend=lambda seg: signal.detrend(seg, axis=0, type='l'))
        # 验证计算得到的功率谱密度p全为零，允许误差范围在1e-15
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    # 测试多维输入信号的功率谱密度计算，对轴-1进行处理
    def test_nd_axis_m1(self):
        # 创建一个长度为20的浮点数数组，值从0.04开始递增，然后将其reshape为(2,
    # 定义测试方法，测试 csd 函数在 axis=0 上的表现
    def test_nd_axis_0(self):
        # 创建一个包含20个浮点数的一维数组，并加上0.04
        x = np.arange(20, dtype=np.float64) + 0.04
        # 将一维数组重新形状为 (10, 2, 1) 的三维数组
        x = x.reshape((10, 2, 1))
        # 调用 csd 函数计算交叉谱密度，设置 nperseg=10，axis=0
        f, p = csd(x, x, nperseg=10, axis=0)
        # 断言结果数组 p 的形状为 (6, 2, 1)
        assert_array_equal(p.shape, (6, 2, 1))
        # 断言 p[:,0,0] 和 p[:,1,0] 的每个元素在给定的误差范围内相等
        assert_allclose(p[:,0,0], p[:,1,0], atol=1e-13, rtol=1e-13)
        # 调用 csd 函数计算两个一维数组的交叉谱密度，设置 nperseg=10
        f0, p0 = csd(x[:,0,0], x[:,0,0], nperseg=10)
        # 断言 p0 和 p[:,1,0] 的每个元素在给定的误差范围内相等
        assert_allclose(p0, p[:,1,0], atol=1e-13, rtol=1e-13)

    # 定义测试方法，测试 csd 函数在使用外部窗口时的表现
    def test_window_external(self):
        # 创建一个长度为 16 的零数组
        x = np.zeros(16)
        # 将数组中的第一个和第九个元素设置为 1
        x[0] = 1
        x[8] = 1
        # 调用 csd 函数计算交叉谱密度，指定窗口为 'hann'，窗口长度为 8
        f, p = csd(x, x, 10, 'hann', 8)
        # 获取 'hann' 窗口的数组表示
        win = signal.get_window('hann', 8)
        # 调用 csd 函数计算交叉谱密度，使用 'hann' 窗口，nperseg=None
        fe, pe = csd(x, x, 10, win, nperseg=None)
        # 断言两个交叉谱密度数组 p 和 pe 的每个元素都在机器精度误差内相等
        assert_array_almost_equal_nulp(p, pe)
        # 断言两个频率数组 f 和 fe 的每个元素都在机器精度误差内相等
        assert_array_almost_equal_nulp(f, fe)
        # 断言 fe 的形状为 (5,)，因为 nperseg 使用了窗口的长度
        assert_array_equal(fe.shape, (5,))
        # 断言 pe 的形状为 (5,)
        assert_array_equal(pe.shape, (5,))
        # 断言调用 csd 函数时抛出 ValueError 异常，因为 nperseg 与 win.shape[-1] 不匹配
        assert_raises(ValueError, csd, x, x, 10, win, nperseg=256)
        # 获取长度为 32 的 'hann' 窗口的数组表示
        win_err = signal.get_window('hann', 32)
        # 断言调用 csd 函数时抛出 ValueError 异常，因为窗口长度大于信号长度
        assert_raises(ValueError, csd, x, x, 10, win_err, nperseg=None)

    # 定义测试方法，测试 csd 函数在空输入情况下的表现
    def test_empty_input(self):
        # 调用 csd 函数计算交叉谱密度，输入一个空列表和一个长度为 10 的零数组
        f, p = csd([], np.zeros(10))
        # 断言频率数组 f 的形状为 (0,)
        assert_array_equal(f.shape, (0,))
        # 断言交叉谱密度数组 p 的形状为 (0,)
        assert_array_equal(p.shape, (0,))

        # 调用 csd 函数计算交叉谱密度，输入一个长度为 10 的零数组和一个空列表
        f, p = csd(np.zeros(10), [])
        # 断言频率数组 f 的形状为 (0,)
        assert_array_equal(f.shape, (0,))
        # 断言交叉谱密度数组 p 的形状为 (0,)

        # 对于各种形状的空数组输入，依次调用 csd 函数计算交叉谱密度
        for shape in [(0,), (3,0), (0,5,2)]:
            f, p = csd(np.empty(shape), np.empty(shape))
            # 断言频率数组 f 的形状与输入的空数组形状相同
            assert_array_equal(f.shape, shape)
            # 断言交叉谱密度数组 p 的形状与输入的空数组形状相同

        # 调用 csd 函数计算交叉谱密度，输入一个长度为 10 的一数组和一个形状为 (5,0) 的空数组
        f, p = csd(np.ones(10), np.empty((5,0)))
        # 断言频率数组 f 的形状为 (5,0)
        assert_array_equal(f.shape, (5,0))
        # 断言交叉谱密度数组 p 的形状为 (5,0)

        # 调用 csd 函数计算交叉谱密度，输入一个形状为 (5,0) 的空数组和一个长度为 10 的一数组
        f, p = csd(np.empty((5,0)), np.ones(10))
        # 断言频率数组 f 的形状为 (5,0)
        assert_array_equal(f.shape, (5,0))
        # 断言交叉谱密度数组 p 的形状为 (5,0)

    # 定义测试方法，测试 csd 函数在空输入情况下的表现（使用其他轴）
    def test_empty_input_other_axis(self):
        # 对于各种形状的空数组输入，依次调用 csd 函数计算交叉谱密度，指定 axis=1
        for shape in [(3,0), (0,5,2)]:
            f, p = csd(np.empty(shape), np.empty(shape), axis=1)
            # 断言频率数组 f 的形状与输入的空数组形状相同
            assert_array_equal(f.shape, shape)
            # 断言交叉谱密度数组 p 的形状与输入的空数组形状相同

        # 调用 csd 函数计算交叉谱密度，输入形状为 (10,10,3) 的空数组和形状为 (10,0,1) 的零数组，指定 axis=1
        f, p = csd(np.empty((10,10,3)), np.zeros((10,0,1)), axis=1)
        # 断言频率数组 f 的形状为 (10,0,3)
        assert_array_equal(f.shape, (10,0,3))
        # 断言交叉谱密度数组 p 的形状为 (10,0,3)

        # 调用 csd 函数计算交叉谱密度，输入形状为 (10,0,1) 的空数组和形状为 (10,10,3) 的零数组，指定 axis=1
        f, p = csd(np.empty((10,0,1)), np.zeros((10,10,3)), axis=1)
        # 断言频率数组 f 的形状为 (10,0,3)
        assert_array_equal(f.shape, (10,0,3))
        # 断言交叉谱密度数组 p 的形状为 (10,0,3)
    def test_short_data(self):
        x = np.zeros(8)  # 创建一个长度为8的全零数组x
        x[0] = 1  # 将数组x的第一个元素设置为1

        # 使用suppress_warnings上下文管理器抑制特定警告消息
        with suppress_warnings() as sup:
            # 设置过滤器，过滤特定消息类型的UserWarning，消息内容为指定的msg字符串
            msg = "nperseg = 256 is greater than input length  = 8, using nperseg = 8"
            sup.filter(UserWarning, msg)
            # 调用csd函数计算信号x和自身的交叉功率谱密度，使用'Hann'窗口，默认nperseg
            f, p = csd(x, x, window='hann')  # 使用默认的nperseg
            # 调用csd函数计算信号x和自身的交叉功率谱密度，使用'Hann'窗口，指定nperseg为256
            f1, p1 = csd(x, x, window='hann', nperseg=256)  # 用户指定的nperseg
        # 调用csd函数计算信号x和自身的交叉功率谱密度，指定nperseg为8
        f2, p2 = csd(x, x, nperseg=8)  # 有效的nperseg，不会产生警告
        # 断言所有返回的频率数组f与f2相近
        assert_allclose(f, f2)
        # 断言所有返回的功率谱密度数组p与p2相近
        assert_allclose(p, p2)
        # 断言所有返回的频率数组f1与f2相近
        assert_allclose(f1, f2)
        # 断言所有返回的功率谱密度数组p1与p2相近
        assert_allclose(p1, p2)

    def test_window_long_or_nd(self):
        # 断言调用csd函数时传入长度为4的全零数组和长度为4的全1数组，设置nperseg为1，传入的数组是长度为5的一维数组
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1,
                      np.array([1,1,1,1,1]))
        # 断言调用csd函数时传入长度为4的全零数组和长度为4的全1数组，设置nperseg为1，传入的数组是形状为(2,3)的二维数组
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1,
                      np.arange(6).reshape((2,3)))

    def test_nondefault_noverlap(self):
        x = np.zeros(64)  # 创建一个长度为64的全零数组x
        x[::8] = 1  # 将数组x每隔8个元素设为1
        # 调用csd函数计算信号x和自身的交叉功率谱密度，设置nperseg为16，noverlap为4
        f, p = csd(x, x, nperseg=16, noverlap=4)
        q = np.array([0, 1./12., 1./3., 1./5., 1./3., 1./5., 1./3., 1./5.,
                      1./6.])  # 预期的功率谱密度数组
        # 断言所有返回的功率谱密度数组p与预期数组q相近，允许误差为1e-12
        assert_allclose(p, q, atol=1e-12)

    def test_bad_noverlap(self):
        # 断言调用csd函数时传入长度为4的全零数组和长度为4的全1数组，设置nperseg为1，指定的窗口类型为'Hann'，noverlap设置为2和7都会引发ValueError
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1, 'hann',
                      2, 7)

    def test_nfft_too_short(self):
        # 断言调用csd函数时传入全1的长度为12的数组和全零的长度为12的数组，设置nfft为3，nperseg为4会引发ValueError
        assert_raises(ValueError, csd, np.ones(12), np.zeros(12), nfft=3,
                      nperseg=4)

    def test_real_onesided_even_32(self):
        x = np.zeros(16, 'f')  # 创建一个长度为16的全零数组x，数据类型为单精度浮点数
        x[0] = 1  # 将数组x的第一个元素设为1
        # 调用csd函数计算信号x和自身的交叉功率谱密度，设置nperseg为8
        f, p = csd(x, x, nperseg=8)
        # 断言所有返回的频率数组f与期望的线性空间数组相近，从0到0.5，共5个点
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111], 'f')  # 预期的功率谱密度数组
        # 断言所有返回的功率谱密度数组p与预期数组q相近，允许的相对误差和绝对误差均为1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言功率谱密度数组p的数据类型与q相同
        assert_(p.dtype == q.dtype)

    def test_real_onesided_odd_32(self):
        x = np.zeros(16, 'f')  # 创建一个长度为16的全零数组x，数据类型为单精度浮点数
        x[0] = 1  # 将数组x的第一个元素设为1
        # 调用csd函数计算信号x和自身的交叉功率谱密度，设置nperseg为9
        f, p = csd(x, x, nperseg=9)
        # 断言所有返回的频率数组f与期望的等间隔数组相近，从0到1，共5个点
        assert_allclose(f, np.arange(5.0)/9.0)
        q = np.array([0.12477458, 0.23430935, 0.17072113, 0.17072116,
                      0.17072113], 'f')  # 预期的功率谱密度数组
        # 断言所有返回的功率谱密度数组p与预期数组q相近，允许的相对误差和绝对误差均为1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言功率谱密度数组p的数据类型与q相同
        assert_(p.dtype == q.dtype)

    def test_real_twosided_32(self):
        x = np.zeros(16, 'f')  # 创建一个长度为16的全零数组x，数据类型为单精度浮点数
        x[0] = 1  # 将数组x的第一个元素设为1
        # 调用csd函数计算信号x和自身的交叉功率谱密度，设置nperseg为8，return_onesided为False
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        # 断言所有返回的频率数组f与期望的频率数组相近，使用fftfreq函数生成
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.11111111,
                      0.07638889], 'f')  # 预期的功率谱密度数组
        # 断言所有返回的功率
    # 定义一个名为 test_complex_32 的测试方法
    def test_complex_32(self):
        # 创建一个复数类型的零数组，长度为 16
        x = np.zeros(16, 'F')
        # 在数组中设置第一个和第九个元素为复数值 1.0 + 2.0j
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        # 计算交叉功率谱密度，返回频率和功率密度
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        # 断言频率 f 等于使用 fftfreq 函数生成的频率数组，采样间隔为 1.0
        assert_allclose(f, fftfreq(8, 1.0))
        # 定义预期的功率谱密度数组 q
        q = np.array([0.41666666, 0.38194442, 0.55555552, 0.55555552,
                      0.55555558, 0.55555552, 0.55555552, 0.38194442], 'f')
        # 断言计算得到的功率密度 p 与预期的 q 接近，允许误差范围为 1e-7
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        # 断言 p 和 q 的数据类型相同
        assert_(p.dtype == q.dtype,
                f'dtype mismatch, {p.dtype}, {q.dtype}')

    # 定义一个名为 test_padded_freqs 的测试方法
    def test_padded_freqs(self):
        # 创建长度为 12 的零数组和全为 1 的数组
        x = np.zeros(12)
        y = np.ones(12)

        # 设置 FFT 的长度为 24，生成频率数组 f
        nfft = 24
        f = fftfreq(nfft, 1.0)[:nfft//2+1]
        f[-1] *= -1
        # 计算交叉功率谱密度，返回频率 fodd 和未使用的结果
        fodd, _ = csd(x, y, nperseg=5, nfft=nfft)
        # 计算交叉功率谱密度，返回频率 feven 和未使用的结果
        feven, _ = csd(x, y, nperseg=6, nfft=nfft)
        # 断言频率 f 与 fodd 和 feven 接近
        assert_allclose(f, fodd)
        assert_allclose(f, feven)

        # 设置 FFT 的长度为 25，生成频率数组 f
        nfft = 25
        f = fftfreq(nfft, 1.0)[:(nfft + 1)//2]
        # 计算交叉功率谱密度，返回频率 fodd 和未使用的结果
        fodd, _ = csd(x, y, nperseg=5, nfft=nfft)
        # 计算交叉功率谱密度，返回频率 feven 和未使用的结果
        feven, _ = csd(x, y, nperseg=6, nfft=nfft)
        # 断言频率 f 与 fodd 和 feven 接近
        assert_allclose(f, fodd)
        assert_allclose(f, feven)

    # 定义一个名为 test_copied_data 的测试方法
    def test_copied_data(self):
        # 创建一个包含 64 个随机数的数组 x，复制一份给数组 y
        x = np.random.randn(64)
        y = x.copy()

        # 计算自相关功率谱密度，返回未使用的结果和 p_same
        _, p_same = csd(x, x, nperseg=8, average='mean',
                        return_onesided=False)
        # 计算交叉功率谱密度，返回未使用的结果和 p_copied
        _, p_copied = csd(x, y, nperseg=8, average='mean',
                          return_onesided=False)
        # 断言 p_same 和 p_copied 的值接近
        assert_allclose(p_same, p_copied)

        # 计算自相关功率谱密度，返回未使用的结果和 p_same
        _, p_same = csd(x, x, nperseg=8, average='median',
                        return_onesided=False)
        # 计算交叉功率谱密度，返回未使用的结果和 p_copied
        _, p_copied = csd(x, y, nperseg=8, average='median',
                          return_onesided=False)
        # 断言 p_same 和 p_copied 的值接近
        assert_allclose(p_same, p_copied)
class TestCoherence:
    # 定义测试类 TestCoherence，用于测试 coherence 函数的一致性
    def test_identical_input(self):
        # 测试相同输入条件下的 coherence 函数
        x = np.random.randn(20)
        y = np.copy(x)  # 创建 x 的副本 y，确保 y is x 返回 False

        f = np.linspace(0, 0.5, 6)
        C = np.ones(6)
        f1, C1 = coherence(x, y, nperseg=10)

        assert_allclose(f, f1)  # 断言频率结果的一致性
        assert_allclose(C, C1)  # 断言 coerence 结果的一致性

    def test_phase_shifted_input(self):
        # 测试相位偏移输入条件下的 coherence 函数
        x = np.random.randn(20)
        y = -x  # 创建相位相反的输入 y

        f = np.linspace(0, 0.5, 6)
        C = np.ones(6)
        f1, C1 = coherence(x, y, nperseg=10)

        assert_allclose(f, f1)  # 断言频率结果的一致性
        assert_allclose(C, C1)  # 断言 coerence 结果的一致性


class TestSpectrogram:
    # 定义测试类 TestSpectrogram，用于测试 spectrogram 函数的功能
    def test_average_all_segments(self):
        # 测试对所有段均值的计算
        x = np.random.randn(1024)

        fs = 1.0
        window = ('tukey', 0.25)
        nperseg = 16
        noverlap = 2

        f, _, P = spectrogram(x, fs, window, nperseg, noverlap)
        fw, Pw = welch(x, fs, window, nperseg, noverlap)

        assert_allclose(f, fw)  # 断言频率结果的一致性
        assert_allclose(np.mean(P, axis=-1), Pw)  # 断言功率谱结果的一致性

    def test_window_external(self):
        # 测试使用外部窗口的情况
        x = np.random.randn(1024)

        fs = 1.0
        window = ('tukey', 0.25)
        nperseg = 16
        noverlap = 2

        f, _, P = spectrogram(x, fs, window, nperseg, noverlap)

        win = signal.get_window(('tukey', 0.25), 16)
        fe, _, Pe = spectrogram(x, fs, win, nperseg=None, noverlap=2)

        assert_array_equal(fe.shape, (9,))  # 断言频率数组形状的一致性
        assert_array_equal(Pe.shape, (9,73))  # 断言功率谱数组形状的一致性

        assert_raises(ValueError, spectrogram, x, fs, win, nperseg=8)
        # 断言当 nperseg != win.shape[-1] 时会引发 ValueError

        win_err = signal.get_window(('tukey', 0.25), 2048)
        assert_raises(ValueError, spectrogram, x, fs, win_err, nperseg=None)
        # 断言当窗口长度超过信号长度时会引发 ValueError

    def test_short_data(self):
        # 测试短数据的情况
        x = np.random.randn(1024)
        fs = 1.0

        # 使用字符串类型窗口时，输入信号长度 < nperseg 会产生 UserWarning
        # 此时 nperseg 被设置为 x.shape[-1]
        f, _, p = spectrogram(x, fs, window=('tukey', 0.25))  # 默认 nperseg
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "nperseg = 1025 is greater than input length  = 1024, "
                       "using nperseg = 1024",)
            f1, _, p1 = spectrogram(x, fs, window=('tukey', 0.25), nperseg=1025)
            # 用户指定 nperseg 时的测试
        f2, _, p2 = spectrogram(x, fs, nperseg=256)  # 默认 nperseg 的测试
        f3, _, p3 = spectrogram(x, fs, nperseg=1024)  # 用户指定 nperseg 的测试

        assert_allclose(f, f2)  # 断言频率结果的一致性
        assert_allclose(p, p2)  # 断言功率谱结果的一致性
        assert_allclose(f1, f3)  # 断言频率结果的一致性
        assert_allclose(p1, p3)  # 断言功率谱结果的一致性
    def test_frequency(self):
        """Test if frequency location of peak corresponds to frequency of
        generated input signal.
        """

        # Input parameters
        ampl = 2.  # 振幅设定为2
        w = 1.  # 角频率设定为1
        phi = 0.5 * np.pi  # 初相位设定为π/2
        nin = 100  # 输入信号的时间步数设定为100
        nout = 1000  # 计算周期图的频率点数设定为1000
        p = 0.7  # 选择的时间点占比设定为0.7

        # Randomly select a fraction of an array with timesteps
        np.random.seed(2353425)  # 使用固定种子以确保结果可重复
        r = np.random.rand(nin)  # 生成长度为nin的随机数组r
        t = np.linspace(0.01*np.pi, 10.*np.pi, nin)[r >= p]  # 在给定时间范围内随机选择符合条件的时间点数组t

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi)  # 根据设定的振幅、角频率和初相位生成正弦波信号x

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout)  # 定义计算周期图的频率数组f

        # Calculate Lomb-Scargle periodogram
        P = lombscargle(t, x, f)  # 计算Lomb-Scargle周期图P

        # Check if difference between found frequency maximum and input
        # frequency is less than accuracy
        delta = f[1] - f[0]  # 计算频率步长delta
        assert_(w - f[np.argmax(P)] < (delta/2.))  # 断言找到的频率最大值与输入频率w之间的差异小于步长的一半

    def test_amplitude(self):
        # Test if height of peak in normalized Lomb-Scargle periodogram
        # corresponds to amplitude of the generated input signal.

        # Input parameters (same as in test_frequency)
        ampl = 2.  # 振幅设定为2
        w = 1.  # 角频率设定为1
        phi = 0.5 * np.pi  # 初相位设定为π/2
        nin = 100  # 输入信号的时间步数设定为100
        nout = 1000  # 计算周期图的频率点数设定为1000
        p = 0.7  # 选择的时间点占比设定为0.7

        # Randomly select a fraction of an array with timesteps
        np.random.seed(2353425)  # 使用相同的固定种子确保结果一致
        r = np.random.rand(nin)  # 生成长度为nin的随机数组r
        t = np.linspace(0.01*np.pi, 10.*np.pi, nin)[r >= p]  # 在给定时间范围内随机选择符合条件的时间点数组t

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi)  # 根据设定的振幅、角频率和初相位生成正弦波信号x

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout)  # 定义计算周期图的频率数组f

        # Calculate Lomb-Scargle periodogram
        pgram = lombscargle(t, x, f)  # 计算Lomb-Scargle周期图pgram

        # Normalize
        pgram = np.sqrt(4 * pgram / t.shape[0])  # 对周期图进行归一化处理

        # Check if difference between found frequency maximum and input
        # frequency is less than accuracy
        assert_approx_equal(np.max(pgram), ampl, significant=2)  # 断言归一化后的周期图的峰值与输入振幅ampl之间的近似相等性
    def test_precenter(self):
        # Test if precenter gives the same result as manually precentering.

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select
        offset = 0.15  # Offset to be subtracted in pre-centering

        # Randomly select a fraction of an array with timesteps
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01*np.pi, 10.*np.pi, nin)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi) + offset

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout)

        # Calculate Lomb-Scargle periodogram with pre-centering
        pgram = lombscargle(t, x, f, precenter=True)
        # Calculate Lomb-Scargle periodogram without pre-centering
        pgram2 = lombscargle(t, x - x.mean(), f, precenter=False)

        # check if centering worked
        assert_allclose(pgram, pgram2)

    def test_normalize(self):
        # Test normalize option of Lomb-Scarge.

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select

        # Randomly select a fraction of an array with timesteps
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01*np.pi, 10.*np.pi, nin)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi)

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout)

        # Calculate Lomb-Scargle periodogram without normalization
        pgram = lombscargle(t, x, f)
        # Calculate Lomb-Scargle periodogram with normalization
        pgram2 = lombscargle(t, x, f, normalize=True)

        # check if normalization works as expected
        assert_allclose(pgram * 2 / np.dot(x, x), pgram2)
        assert_approx_equal(np.max(pgram2), 1.0, significant=2)

    def test_wrong_shape(self):
        # Test case where input arrays have incompatible shapes.

        t = np.linspace(0, 1, 1)
        x = np.linspace(0, 1, 2)
        f = np.linspace(0, 1, 3)
        # Expecting a ValueError due to shape mismatch
        assert_raises(ValueError, lombscargle, t, x, f)

    def test_zero_division(self):
        # Test case where division by zero is expected.

        t = np.zeros(1)
        x = np.zeros(1)
        f = np.zeros(1)
        # Expecting a ZeroDivisionError due to all zeros in inputs
        assert_raises(ZeroDivisionError, lombscargle, t, x, f)

    def test_lombscargle_atan_vs_atan2(self):
        # Test case for an issue related to arctangent calculation method.

        # Generate input data
        t = np.linspace(0, 10, 1000, endpoint=False)
        x = np.sin(4*t)
        f = np.linspace(0, 50, 500, endpoint=False) + 0.1
        # Call Lomb-Scargle with adjusted frequencies
        lombscargle(t, x, f*2*np.pi)
class TestSTFT:
    # 测试检查 COLA 条件的方法
    def test_check_COLA(self):
        # 不同设置的测试参数
        settings = [
                    ('boxcar', 10, 0),
                    ('boxcar', 10, 9),
                    ('bartlett', 51, 26),
                    ('hann', 256, 128),
                    ('hann', 256, 192),
                    ('blackman', 300, 200),
                    (('tukey', 0.5), 256, 64),
                    ('hann', 256, 255),
                    ]
        
        # 遍历不同设置进行测试
        for setting in settings:
            msg = '{}, {}, {}'.format(*setting)
            # 断言 COLA 条件为真
            assert_equal(True, check_COLA(*setting), err_msg=msg)

    # 测试检查 NOLA 条件的方法
    def test_check_NOLA(self):
        # 通过的测试设置
        settings_pass = [
                    ('boxcar', 10, 0),
                    ('boxcar', 10, 9),
                    ('boxcar', 10, 7),
                    ('bartlett', 51, 26),
                    ('bartlett', 51, 10),
                    ('hann', 256, 128),
                    ('hann', 256, 192),
                    ('hann', 256, 37),
                    ('blackman', 300, 200),
                    ('blackman', 300, 123),
                    (('tukey', 0.5), 256, 64),
                    (('tukey', 0.5), 256, 38),
                    ('hann', 256, 255),
                    ('hann', 256, 39),
                    ]
        
        # 遍历通过的测试设置进行测试
        for setting in settings_pass:
            msg = '{}, {}, {}'.format(*setting)
            # 断言 NOLA 条件为真
            assert_equal(True, check_NOLA(*setting), err_msg=msg)

        # 不通过的测试设置
        w_fail = np.ones(16)
        w_fail[::2] = 0
        settings_fail = [
                    (w_fail, len(w_fail), len(w_fail) // 2),
                    ('hann', 64, 0),
        ]
        
        # 遍历不通过的测试设置进行测试
        for setting in settings_fail:
            msg = '{}, {}, {}'.format(*setting)
            # 断言 NOLA 条件为假
            assert_equal(False, check_NOLA(*setting), err_msg=msg)

    # 测试计算所有片段平均值的方法
    def test_average_all_segments(self):
        np.random.seed(1234)
        x = np.random.randn(1024)

        fs = 1.0
        window = 'hann'
        nperseg = 16
        noverlap = 8

        # 使用 STFT 计算频谱
        f, _, Z = stft(x, fs, window, nperseg, noverlap, padded=False,
                       return_onesided=False, boundary=None)
        # 使用 Welch 方法计算频谱
        fw, Pw = welch(x, fs, window, nperseg, noverlap, return_onesided=False,
                       scaling='spectrum', detrend=False)

        # 断言频谱和功率谱接近
        assert_allclose(f, fw)
        assert_allclose(np.mean(np.abs(Z)**2, axis=-1), Pw)
    def test_permute_axes(self):
        # 设置随机数种子，确保结果可重复
        np.random.seed(1234)
        # 生成长度为 1024 的随机数组
        x = np.random.randn(1024)

        # 设置信号的采样频率
        fs = 1.0
        # 设置窗口函数类型为汉宁窗
        window = 'hann'
        # 设置每个段的长度为 16
        nperseg = 16
        # 设置重叠的长度为 8
        noverlap = 8

        # 对信号进行短时傅里叶变换（STFT），返回频率 f1、时间 t1 和 STFT 结果 Z1
        f1, t1, Z1 = stft(x, fs, window, nperseg, noverlap)
        # 对重塑后的信号进行短时傅里叶变换，返回频率 f2、时间 t2 和 STFT 结果 Z2
        f2, t2, Z2 = stft(x.reshape((-1, 1, 1)), fs, window, nperseg, noverlap,
                          axis=0)

        # 对 Z1 进行逆短时傅里叶变换（ISTFT），返回时间 t3 和恢复信号 x1
        t3, x1 = istft(Z1, fs, window, nperseg, noverlap)
        # 对 Z2 转置后进行逆短时傅里叶变换，返回时间 t4 和恢复信号 x2
        t4, x2 = istft(Z2.T, fs, window, nperseg, noverlap, time_axis=0,
                       freq_axis=-1)

        # 检查频率 f1 和 f2 是否相等
        assert_allclose(f1, f2)
        # 检查时间 t1 和 t2 是否相等
        assert_allclose(t1, t2)
        # 检查时间 t3 和 t4 是否相等
        assert_allclose(t3, t4)
        # 检查 STFT 结果 Z1 的第一个频率维度与 Z2 的第一个频率维度是否相等
        assert_allclose(Z1, Z2[:, 0, 0, :])
        # 检查恢复信号 x1 和 x2 的第一个频率维度是否相等
        assert_allclose(x1, x2[:, 0, 0])

    @pytest.mark.parametrize('scaling', ['spectrum', 'psd'])
    def test_roundtrip_real(self, scaling):
        # 设置随机数种子，确保结果可重复
        np.random.seed(1234)

        # 定义多组设置进行测试
        settings = [
                    ('boxcar', 100, 10, 0),           # 测试无重叠
                    ('boxcar', 100, 10, 9),           # 测试高重叠
                    ('bartlett', 101, 51, 26),        # 测试奇数 nperseg
                    ('hann', 1024, 256, 128),         # 测试默认设置
                    (('tukey', 0.5), 1152, 256, 64),  # 测试 Tukey 窗
                    ('hann', 1024, 256, 255),         # 测试重叠的汉宁窗
                    ]

        # 遍历每组设置进行测试
        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10*np.random.randn(t.size)

            # 对信号进行短时傅里叶变换（STFT），返回频率 t、时间 xr 和 STFT 结果 zz
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=False,
                            scaling=scaling)

            # 对 STFT 结果 zz 进行逆短时傅里叶变换（ISTFT），返回时间 tr 和恢复信号 xr
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                           window=window, scaling=scaling)

            msg = f'{window}, {noverlap}'
            # 检查时间 t 和 tr 是否相等
            assert_allclose(t, tr, err_msg=msg)
            # 检查信号 x 和 xr 是否相等
            assert_allclose(x, xr, err_msg=msg)

    def test_roundtrip_not_nola(self):
        # 设置随机数种子，确保结果可重复
        np.random.seed(1234)

        # 创建一个不符合 NOLA 条件的窗口函数
        w_fail = np.ones(16)
        w_fail[::2] = 0
        settings = [
                    (w_fail, 256, len(w_fail), len(w_fail) // 2),
                    ('hann', 256, 64, 0),
        ]

        # 遍历每组设置进行测试
        for window, N, nperseg, noverlap in settings:
            msg = f'{window}, {N}, {nperseg}, {noverlap}'
            # 检查窗口函数是否符合 NOLA 条件
            assert not check_NOLA(window, nperseg, noverlap), msg

            t = np.arange(N)
            x = 10 * np.random.randn(t.size)

            # 对信号进行短时傅里叶变换（STFT），返回频率 t、时间 xr 和 STFT 结果 zz
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=True,
                            boundary='zeros')
            # 捕获 NOLA 警告，并进行逆短时傅里叶变换（ISTFT）
            with pytest.warns(UserWarning, match='NOLA'):
                tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                               window=window, boundary=True)

            # 检查时间 t 和 tr 是否接近相等
            assert np.allclose(t, tr[:len(t)]), msg
            # 检查信号 x 和 xr 是否不相等
            assert not np.allclose(x, xr[:len(x)]), msg
    # 定义一个测试函数，用于测试信号处理过程中的循环遍历、反变换的正确性
    def test_roundtrip_nola_not_cola(self):
        # 设置随机数种子，确保可重复性
        np.random.seed(1234)

        # 设置不同的信号处理参数组合
        settings = [
                    ('boxcar', 100, 10, 3),           # 使用boxcar窗口，N=100，nperseg=10，noverlap=3
                    ('bartlett', 101, 51, 37),        # 使用bartlett窗口，N=101，nperseg=51，noverlap=37
                    ('hann', 1024, 256, 127),         # 使用hann窗口，N=1024，nperseg=256，noverlap=127
                    (('tukey', 0.5), 1152, 256, 14),  # 使用tukey窗口，alpha=0.5，N=1152，nperseg=256，noverlap=14
                    ('hann', 1024, 256, 5),           # 使用hann窗口，N=1024，nperseg=256，noverlap=5
                    ]

        # 对每组设置进行遍历
        for window, N, nperseg, noverlap in settings:
            # 根据当前设置生成消息字符串
            msg = f'{window}, {nperseg}, {noverlap}'
            # 断言当前窗口是否满足NOLA条件
            assert check_NOLA(window, nperseg, noverlap), msg
            # 断言当前窗口不满足COLA条件
            assert not check_COLA(window, nperseg, noverlap), msg

            # 生成时间向量t
            t = np.arange(N)
            # 生成随机信号x
            x = 10 * np.random.randn(t.size)

            # 进行短时傅里叶变换STFT，得到频谱zz
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=True,
                            boundary='zeros')

            # 对频谱进行反变换ISTFT，得到重构的时间序列tr和xr
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                           window=window, boundary=True)

            # 生成消息字符串
            msg = f'{window}, {noverlap}'
            # 断言重构的时间序列tr与原始时间向量t的接近程度
            assert_allclose(t, tr[:len(t)], err_msg=msg)
            # 断言重构的信号xr与原始信号x的接近程度
            assert_allclose(x, xr[:len(x)], err_msg=msg)

    # 定义一个测试函数，用于测试浮点数类型为float32时的信号处理过程
    def test_roundtrip_float32(self):
        # 设置随机数种子，确保可重复性
        np.random.seed(1234)

        # 设置一个信号处理参数组合
        settings = [('hann', 1024, 256, 128)]

        # 对设置的参数组合进行遍历
        for window, N, nperseg, noverlap in settings:
            # 生成时间向量t
            t = np.arange(N)
            # 生成随机信号x
            x = 10*np.random.randn(t.size)
            # 将信号x转换为float32类型
            x = x.astype(np.float32)

            # 进行短时傅里叶变换STFT，得到频谱zz
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=False)

            # 对频谱进行反变换ISTFT，得到重构的时间序列tr和xr
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                           window=window)

            # 生成消息字符串
            msg = f'{window}, {noverlap}'
            # 断言重构的时间序列tr与原始时间向量t的接近程度
            assert_allclose(t, t, err_msg=msg)
            # 断言重构的信号xr与原始信号x的接近程度，设置容忍的相对误差和绝对误差
            assert_allclose(x, xr, err_msg=msg, rtol=1e-4, atol=1e-5)
            # 断言重构的信号xr的数据类型为float32
            assert_(x.dtype == xr.dtype)

    # 使用pytest的参数化装饰器，定义一个参数化的测试用例，用于测试频谱和功率谱的缩放选项
    @pytest.mark.parametrize('scaling', ['spectrum', 'psd'])
    # 定义一个测试函数，用于测试复杂情况下的信号处理函数的回路
    def test_roundtrip_complex(self, scaling):
        # 设定随机种子以保证结果的可重复性
        np.random.seed(1234)

        # 设置不同的信号处理参数组合进行测试
        settings = [
                    ('boxcar', 100, 10, 0),           # 测试无重叠情况
                    ('boxcar', 100, 10, 9),           # 测试高重叠情况
                    ('bartlett', 101, 51, 26),        # 测试奇数长度分段
                    ('hann', 1024, 256, 128),         # 测试默认参数
                    (('tukey', 0.5), 1152, 256, 64),  # 测试 Tukey 窗口
                    ('hann', 1024, 256, 255),         # 测试重叠的 hann 窗口
                    ]

        # 遍历不同的参数设置进行测试
        for window, N, nperseg, noverlap in settings:
            # 创建时间序列
            t = np.arange(N)
            # 创建复数随机信号
            x = 10*np.random.randn(t.size) + 10j*np.random.randn(t.size)

            # 进行短时傅里叶变换
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=False,
                            return_onesided=False, scaling=scaling)

            # 进行逆短时傅里叶变换
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                           window=window, input_onesided=False,
                           scaling=scaling)

            # 准备错误消息
            msg = f'{window}, {nperseg}, {noverlap}'
            # 断言实际结果与期望结果之间的近似性
            assert_allclose(t, tr, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg)

        # 检查当设置为单侧返回时是否切换到双侧返回
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "Input data is complex, switching to return_onesided=False")
            # 执行短时傅里叶变换
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=False,
                            return_onesided=True, scaling=scaling)

        # 再次进行逆短时傅里叶变换
        tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                       window=window, input_onesided=False, scaling=scaling)

        # 准备错误消息
        msg = f'{window}, {nperseg}, {noverlap}'
        # 断言实际结果与期望结果之间的近似性
        assert_allclose(t, tr, err_msg=msg)
        assert_allclose(x, xr, err_msg=msg)
    # 定义一个测试方法，用于测试边界扩展的情况
    def test_roundtrip_boundary_extension(self):
        np.random.seed(1234)

        # 使用 boxcar 窗口进行测试，因为窗口中的所有值都是 1，所以可以完全恢复没有边界扩展的情况
        # 设置测试参数，包括窗口类型、信号长度、每段长度和重叠部分长度
        settings = [
                    ('boxcar', 100, 10, 0),           # 测试无重叠
                    ('boxcar', 100, 10, 9),           # 测试高重叠
                    ]

        # 遍历所有设置
        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)  # 创建时间轴
            x = 10*np.random.randn(t.size)  # 生成随机信号

            # 计算短时傅里叶变换（STFT）
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                           window=window, detrend=None, padded=True,
                           boundary=None)

            # 计算逆短时傅里叶变换（ISTFT），设置边界为 False
            _, xr = istft(zz, noverlap=noverlap, window=window, boundary=False)

            # 针对不同的边界类型（'even', 'odd', 'constant', 'zeros'）进行测试
            for boundary in ['even', 'odd', 'constant', 'zeros']:
                # 计算带有边界扩展的短时傅里叶变换
                _, _, zz_ext = stft(x, nperseg=nperseg, noverlap=noverlap,
                                window=window, detrend=None, padded=True,
                                boundary=boundary)

                # 计算带有边界扩展的逆短时傅里叶变换，设置边界为 True
                _, xr_ext = istft(zz_ext, noverlap=noverlap, window=window,
                                boundary=True)

                msg = f'{window}, {noverlap}, {boundary}'
                # 断言结果与原始信号的接近程度
                assert_allclose(x, xr, err_msg=msg)
                assert_allclose(x, xr_ext, err_msg=msg)

    # 定义一个测试方法，用于测试带有填充信号的情况
    def test_roundtrip_padded_signal(self):
        np.random.seed(1234)

        # 设置测试参数，包括窗口类型、信号长度、每段长度和重叠部分长度
        settings = [
                    ('boxcar', 101, 10, 0),
                    ('hann', 1000, 256, 128),
                    ]

        # 遍历所有设置
        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)  # 创建时间轴
            x = 10*np.random.randn(t.size)  # 生成随机信号

            # 计算带有填充信号的短时傅里叶变换（STFT）
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=True)

            # 计算逆短时傅里叶变换（ISTFT）
            tr, xr = istft(zz, noverlap=noverlap, window=window)

            msg = f'{window}, {noverlap}'
            # 断言时间轴与信号的接近程度，考虑可能的末尾零填充
            assert_allclose(t, tr[:t.size], err_msg=msg)
            assert_allclose(x, xr[:x.size], err_msg=msg)
    # 定义一个测试方法，用于测试带填充的 FFT 的往返过程
    def test_roundtrip_padded_FFT(self):
        # 设定随机种子
        np.random.seed(1234)

        # 设置不同的参数组合
        settings = [
                    ('hann', 1024, 256, 128, 512),           # 窗口类型为 'hann'，FFT 大小为 1024，段长为 256，重叠为 128，FFT 点数为 512
                    ('hann', 1024, 256, 128, 501),           # 窗口类型为 'hann'，FFT 大小为 1024，段长为 256，重叠为 128，FFT 点数为 501
                    ('boxcar', 100, 10, 0, 33),              # 窗口类型为 'boxcar'，FFT 大小为 100，段长为 10，重叠为 0，FFT 点数为 33
                    (('tukey', 0.5), 1152, 256, 64, 1024),    # 窗口类型为 'tukey'，α=0.5，FFT 大小为 1152，段长为 256，重叠为 64，FFT 点数为 1024
                    ]

        # 遍历设置中的每个参数组合
        for window, N, nperseg, noverlap, nfft in settings:
            # 生成时间序列 t
            t = np.arange(N)
            # 生成具有高斯分布的随机信号 x
            x = 10*np.random.randn(t.size)
            # 生成复数信号 xc，为 x 乘上 exp(1j * π/4)
            xc = x*np.exp(1j*np.pi/4)

            # 对实部信号进行短时傅里叶变换 (STFT)
            _, _, z = stft(x, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                            window=window, detrend=None, padded=True)

            # 对复数信号进行短时傅里叶变换 (STFT)，return_onesided 设为 False
            _, _, zc = stft(xc, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                            window=window, detrend=None, padded=True,
                            return_onesided=False)

            # 对实部信号的短时逆傅里叶变换 (ISTFT)
            tr, xr = istft(z, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                           window=window)

            # 对复数信号的短时逆傅里叶变换 (ISTFT)，input_onesided 设为 False
            tr, xcr = istft(zc, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                            window=window, input_onesided=False)

            # 断言实部信号的时间序列与逆变换结果的时间序列相等
            msg = f'{window}, {noverlap}'
            assert_allclose(t, tr, err_msg=msg)
            # 断言实部信号与逆变换结果的信号相等
            assert_allclose(x, xr, err_msg=msg)
            # 断言复数信号与逆变换结果的信号相等
            assert_allclose(xc, xcr, err_msg=msg)

    # 定义一个测试方法，用于测试轴滚动
    def test_axis_rolling(self):
        # 设定随机种子
        np.random.seed(1234)

        # 生成长度为 1024 的随机信号 x_flat
        x_flat = np.random.randn(1024)
        # 对 x_flat 进行短时傅里叶变换 (STFT)，得到频谱 z_flat
        _, _, z_flat = stft(x_flat)

        # 对于每一个轴 a，进行测试
        for a in range(3):
            # 创建一个新的形状数组 newshape，其中除了第 a 轴，其它轴都是 1
            newshape = [1,]*3
            newshape[a] = -1
            x = x_flat.reshape(newshape)

            # 对 x 进行短时傅里叶变换 (STFT)，指定轴 a 为正索引
            _, _, z_plus = stft(x, axis=a)  # Positive axis index
            # 对 x 进行短时傅里叶变换 (STFT)，指定轴 a 为负索引
            _, _, z_minus = stft(x, axis=a-x.ndim)  # Negative axis index

            # 断言 z_flat 与 z_plus 去除多余维度后形状相同
            assert_equal(z_flat, z_plus.squeeze(), err_msg=a)
            # 断言 z_flat 与 z_minus 去除多余维度后形状相同
            assert_equal(z_flat, z_minus.squeeze(), err_msg=a-x.ndim)

        # z_flat 的形状为 [n_freq, n_time]

        # 测试与转置结果的比较
        _, x_transpose_m = istft(z_flat.T, time_axis=-2, freq_axis=-1)
        _, x_transpose_p = istft(z_flat.T, time_axis=0, freq_axis=1)

        # 断言 x_flat 与 istft 转置结果 x_transpose_m 相等
        assert_allclose(x_flat, x_transpose_m, err_msg='istft transpose minus')
        # 断言 x_flat 与 istft 转置结果 x_transpose_p 相等
        assert_allclose(x_flat, x_transpose_p, err_msg='istft transpose plus')
class TestSampledSpectralRepresentations:
    """Check energy/power relations from `Spectral Analysis` section in the user guide.

    A 32 sample cosine signal is used to compare the numerical to the expected results
    stated in :ref:`tutorial_SpectralAnalysis` in
    file ``doc/source/tutorial/signal.rst``
    """

    n: int = 32  #: number of samples
    T: float = 1/16  #: sampling interval
    a_ref: float = 3  #: amplitude of reference
    l_a: int = 3  #: index in fft for defining frequency of test signal

    x_ref: np.ndarray  #: reference signal
    X_ref: np.ndarray  #: two-sided FFT of x_ref
    E_ref: float  #: energy of signal
    P_ref: float  #: power of signal

    def setup_method(self):
        """Create Cosine signal with amplitude a from spectrum. """

        # Generate frequency values for FFT
        f = rfftfreq(self.n, self.T)

        # Initialize FFT spectrum
        X_ref = np.zeros_like(f)

        # Set the amplitude of the signal at a specific index
        self.l_a = 3
        X_ref[self.l_a] = self.a_ref/2 * self.n  # set amplitude

        # Compute inverse FFT to obtain time-domain signal
        self.x_ref = irfft(X_ref)

        # Compute FFT of the time-domain signal
        self.X_ref = fft(self.x_ref)

        # Calculate energy of the signal using a closed form expression
        self.E_ref = self.tau * self.a_ref**2 / 2  # energy of signal

        # Calculate power of the signal
        self.P_ref = self.a_ref**2 / 2  # power of signal

    @property
    def tau(self) -> float:
        """Duration of signal. """
        return self.n * self.T

    @property
    def delta_f(self) -> float:
        """Bin width """
        return 1 / (self.n * self.T)

    def test_reference_signal(self):
        """Test energy and power formulas. """

        # Verify that the amplitude of the signal is approximately a_ref
        assert_allclose(2*self.a_ref, np.ptp(self.x_ref), rtol=0.1)

        # Verify that the energy expression for the sampled signal matches the expected E_ref
        assert_allclose(self.T * sum(self.x_ref ** 2), self.E_ref)

        # Verify that spectral energy and power formulas are correct:
        sum_X_ref_squared = sum(self.X_ref.real**2 + self.X_ref.imag**2)

        # Check spectral energy formula
        assert_allclose(self.T/self.n * sum_X_ref_squared, self.E_ref)

        # Check spectral power formula
        assert_allclose(1/self.n**2 * sum_X_ref_squared, self.P_ref)
    def test_windowed_DFT(self):
        """验证窗口化离散傅里叶变换的频谱表示。

        此外，验证 `periodogram` 和 `welch` 的缩放。
        """
        w = hann(self.n, sym=False)  # 使用汉宁窗口函数生成长度为 self.n 的窗口

        # 计算窗口的幅度和均方根
        c_amp, c_rms = abs(sum(w)), np.sqrt(sum(w.real**2 + w.imag**2))

        Xw = fft(self.x_ref*w)  # 计算未归一化的窗口化离散傅里叶变换

        # 验证 *频谱* 峰值的一致性:
        assert_allclose(self.tau * Xw[self.l_a] / c_amp, self.a_ref * self.tau / 2)
        # 验证 *幅度谱* 峰值的一致性:
        assert_allclose(Xw[self.l_a] / c_amp, self.a_ref/2)

        # 验证频谱能量密度等于信号的能量:
        X_ESD = self.tau * self.T * abs(Xw / c_rms)**2  # 能量谱密度
        X_PSD = self.T * abs(Xw / c_rms)**2  # 功率谱密度
        assert_allclose(self.delta_f * sum(X_ESD), self.E_ref)
        assert_allclose(self.delta_f * sum(X_PSD), self.P_ref)

        # 验证 periodogram 的缩放:
        kw = dict(fs=1/self.T, window=w, detrend=False, return_onesided=False)
        _, P_mag = periodogram(self.x_ref, scaling='spectrum', **kw)
        _, P_psd = periodogram(self.x_ref, scaling='density', **kw)

        # 验证 periodogram 计算的是平方幅度谱:
        float_res = np.finfo(P_mag.dtype).resolution
        assert_allclose(P_mag, abs(Xw/c_amp)**2, atol=float_res*max(P_mag))
        # 验证 periodogram 计算的是功率谱密度:
        assert_allclose(P_psd, X_PSD, atol=float_res*max(P_psd))

        # 确保 welch 的缩放与 periodogram 相同:
        kw = dict(nperseg=len(self.x_ref), noverlap=0, **kw)
        assert_allclose(welch(self.x_ref, scaling='spectrum', **kw)[1], P_mag,
                        atol=float_res*max(P_mag))
        assert_allclose(welch(self.x_ref, scaling='density', **kw)[1], P_psd,
                        atol=float_res*max(P_psd))
```