# `D:\src\scipysrc\scipy\scipy\signal\tests\test_fir_filter_design.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_equal, assert_,
                           assert_allclose, assert_warns)  # 导入 NumPy 测试模块中的断言函数
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 函数，并重命名为 assert_raises
import pytest  # 导入 pytest 测试框架

from scipy.fft import fft  # 导入 scipy 中 FFT 相关函数
from scipy.special import sinc  # 导入 scipy 中 sinc 函数
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
    firwin, firwin2, freqz, remez, firls, minimum_phase  # 导入 scipy 中的信号处理函数

# 定义测试函数 test_kaiser_beta
def test_kaiser_beta():
    b = kaiser_beta(58.7)  # 计算 Kaiser 窗口的 beta 值
    assert_almost_equal(b, 0.1102 * 50.0)  # 断言 beta 值接近预期值
    b = kaiser_beta(22.0)  # 计算 Kaiser 窗口的 beta 值
    assert_almost_equal(b, 0.5842 + 0.07886)  # 断言 beta 值接近预期值
    b = kaiser_beta(21.0)  # 计算 Kaiser 窗口的 beta 值
    assert_equal(b, 0.0)  # 断言 beta 值为 0
    b = kaiser_beta(10.0)  # 计算 Kaiser 窗口的 beta 值
    assert_equal(b, 0.0)  # 断言 beta 值为 0

# 定义测试函数 test_kaiser_atten
def test_kaiser_atten():
    a = kaiser_atten(1, 1.0)  # 计算 Kaiser 窗口的衰减值
    assert_equal(a, 7.95)  # 断言衰减值为预期值
    a = kaiser_atten(2, 1/np.pi)  # 计算 Kaiser 窗口的衰减值
    assert_equal(a, 2.285 + 7.95)  # 断言衰减值接近预期值

# 定义测试函数 test_kaiserord
def test_kaiserord():
    assert_raises(ValueError, kaiserord, 1.0, 1.0)  # 断言调用 kaiserord 函数时会抛出 ValueError 异常
    numtaps, beta = kaiserord(2.285 + 7.95 - 0.001, 1/np.pi)  # 计算 Kaiser 窗口的阶数和 beta 值
    assert_equal((numtaps, beta), (2, 0.0))  # 断言阶数和 beta 值符合预期

# 定义测试类 TestFirwin
class TestFirwin:

    # 定义检查滤波器响应的方法
    def check_response(self, h, expected_response, tol=.05):
        N = len(h)
        alpha = 0.5 * (N-1)
        m = np.arange(0,N) - alpha   # taps 的时间索引
        for freq, expected in expected_response:
            actual = abs(np.sum(h*np.exp(-1.j*np.pi*m*freq)))  # 计算实际响应
            mse = abs(actual-expected)**2  # 计算均方误差
            assert_(mse < tol, f'response not as expected, mse={mse:g} > {tol:g}')  # 断言均方误差小于容忍值

    # 定义测试滤波器响应的方法
    def test_response(self):
        N = 51
        f = .5
        h = firwin(N, f)  # 生成长度为 N 的 FIR 窗口滤波器
        self.check_response(h, [(.25,1), (.75,0)])  # 检查低通滤波器响应

        h = firwin(N+1, f, window='nuttall')  # 使用 Nuttall 窗口生成滤波器
        self.check_response(h, [(.25,1), (.75,0)])  # 检查滤波器响应

        h = firwin(N+2, f, pass_zero=False)  # 生成高通滤波器
        self.check_response(h, [(.25,0), (.75,1)])  # 检查滤波器响应

        f1, f2, f3, f4 = .2, .4, .6, .8
        h = firwin(N+3, [f1, f2], pass_zero=False)  # 生成带通滤波器
        self.check_response(h, [(.1,0), (.3,1), (.5,0)])  # 检查滤波器响应

        h = firwin(N+4, [f1, f2])  # 生成带阻滤波器
        self.check_response(h, [(.1,1), (.3,0), (.5,1)])  # 检查滤波器响应

        h = firwin(N+5, [f1, f2, f3, f4], pass_zero=False, scale=False)  # 生成多带通滤波器
        self.check_response(h, [(.1,0), (.3,1), (.5,0), (.7,1), (.9,0)])  # 检查滤波器响应

        h = firwin(N+6, [f1, f2, f3, f4])  # 生成多带阻滤波器
        self.check_response(h, [(.1,1), (.3,0), (.5,1), (.7,0), (.9,1)])  # 检查滤波器响应

        h = firwin(N+7, 0.1, width=.03)  # 生成低通滤波器
        self.check_response(h, [(.05,1), (.75,0)])  # 检查滤波器响应

        h = firwin(N+8, 0.1, pass_zero=False)  # 生成高通滤波器
        self.check_response(h, [(.05,0), (.75,1)])  # 检查滤波器响应
    def mse(self, h, bands):
        """Compute mean squared error versus ideal response across frequency
        band.
          h -- coefficients  # 输入参数 h 是滤波器的系数
          bands -- list of (left, right) tuples relative to 1==Nyquist of
            passbands  # 输入参数 bands 是频率带的列表，每个元素是左右频率边界的元组，相对于 Nyquist 频率的比例

        # 计算频率响应和单位频率向量
        w, H = freqz(h, worN=1024)
        f = w/np.pi
        
        # 创建一个长度为 w 的布尔数组，表示通过带指示器
        passIndicator = np.zeros(len(w), bool)
        
        # 遍历每个频率带，设置通过带指示器的相应元素为 True
        for left, right in bands:
            passIndicator |= (f >= left) & (f < right)
        
        # 创建理想的频率响应向量 Hideal，通过带对应的元素为 1，其他为 0
        Hideal = np.where(passIndicator, 1, 0)
        
        # 计算均方误差（MSE），衡量实际频率响应与理想响应之间的差异
        mse = np.mean(abs(abs(H)-Hideal)**2)
        
        # 返回计算得到的均方误差
        return mse

    def test_scaling(self):
        """
        For one lowpass, bandpass, and highpass example filter, this test
        checks two things:
          - the mean squared error over the frequency domain of the unscaled
            filter is smaller than the scaled filter (true for rectangular
            window)
          - the response of the scaled filter is exactly unity at the center
            of the first passband
        """
        N = 11
        
        # 不同滤波器的测试用例
        cases = [
            ([.5], True, (0, 1)),   # 低通滤波器，期望的响应是在第一个通带中心处为单位响应
            ([0.2, .6], False, (.4, 1)),   # 带通滤波器，期望的响应是在第一个通带中心处为单位响应
            ([.5], False, (1, 1)),   # 高通滤波器，期望的响应是在第一个通带中心处为单位响应
        ]
        
        # 遍历每个测试用例
        for cutoff, pass_zero, expected_response in cases:
            # 获取未经缩放的滤波器响应 h
            h = firwin(N, cutoff, scale=False, pass_zero=pass_zero, window='ones')
            
            # 获取经过缩放的滤波器响应 hs
            hs = firwin(N, cutoff, scale=True, pass_zero=pass_zero, window='ones')
            
            # 如果滤波器只有一个截止频率，根据类型设置截止频率
            if len(cutoff) == 1:
                if pass_zero:
                    cutoff = [0] + cutoff
                else:
                    cutoff = cutoff + [1]
            
            # 断言未经缩放的滤波器的均方误差小于经过缩放的滤波器的均方误差
            assert_(self.mse(h, [cutoff]) < self.mse(hs, [cutoff]),
                'least squares violation')  # 如果条件不成立，输出最小二乘误差违规信息
            
            # 检查经过缩放的滤波器响应是否在第一个通带中心处确实为单位响应
            self.check_response(hs, [expected_response], 1e-12)

    def test_fs_validation(self):
        # 使用 pytest 断言，检查当采样频率为单个标量时是否引发值错误异常
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            firwin(51, .5, fs=np.array([10, 20]))
class TestFirWinMore:
    """Different author, different style, different tests..."""

    def test_lowpass(self):
        # 设定带宽为0.04
        width = 0.04
        # 使用 kaiserord 函数计算滤波器的阶数和贝塔值
        ntaps, beta = kaiserord(120, width)
        # 构建参数字典，指定截止频率为0.5，窗口函数为 kaiser，贝塔值为计算得到的 beta，不进行缩放
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        # 使用 firwin 函数生成低通滤波器的滤波器系数
        taps = firwin(ntaps, **kwargs)

        # 检查滤波器系数的对称性
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # 在几个已知的样本频率处检查增益，预期结果为接近0或1
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        # 生成带通类型的滤波器系数字符串
        taps_str = firwin(ntaps, pass_zero='lowpass', **kwargs)
        # 检查两种生成方式的滤波器系数是否相等
        assert_allclose(taps, taps_str)

    def test_highpass(self):
        # 设定带宽为0.04
        width = 0.04
        # 使用 kaiserord 函数计算滤波器的阶数和贝塔值
        ntaps, beta = kaiserord(120, width)

        # 确保滤波器的阶数是奇数
        ntaps |= 1

        # 构建参数字典，指定截止频率为0.5，窗口函数为 kaiser，贝塔值为计算得到的 beta，不进行缩放
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        # 使用 firwin 函数生成高通滤波器的滤波器系数
        taps = firwin(ntaps, pass_zero=False, **kwargs)

        # 检查滤波器系数的对称性
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # 在几个已知的样本频率处检查增益，预期结果为接近0或1
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

        # 生成高通类型的滤波器系数字符串
        taps_str = firwin(ntaps, pass_zero='highpass', **kwargs)
        # 检查两种生成方式的滤波器系数是否相等
        assert_allclose(taps, taps_str)

    def test_bandpass(self):
        # 设定带宽为0.04
        width = 0.04
        # 使用 kaiserord 函数计算滤波器的阶数和贝塔值
        ntaps, beta = kaiserord(120, width)
        # 构建参数字典，指定截止频率为[0.3, 0.7]，窗口函数为 kaiser，贝塔值为计算得到的 beta，不进行缩放
        kwargs = dict(cutoff=[0.3, 0.7], window=('kaiser', beta), scale=False)
        # 使用 firwin 函数生成带通滤波器的滤波器系数
        taps = firwin(ntaps, pass_zero=False, **kwargs)

        # 检查滤波器系数的对称性
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # 在几个已知的样本频率处检查增益，预期结果为接近0或1
        freq_samples = np.array([0.0, 0.2, 0.3-width/2, 0.3+width/2, 0.5,
                                0.7-width/2, 0.7+width/2, 0.8, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        # 生成带通类型的滤波器系数字符串
        taps_str = firwin(ntaps, pass_zero='bandpass', **kwargs)
        # 检查两种生成方式的滤波器系数是否相等
        assert_allclose(taps, taps_str)
    def test_bad_cutoff(self):
        """Test that invalid cutoff argument raises ValueError."""
        # cutoff values must be greater than 0 and less than 1.
        assert_raises(ValueError, firwin, 99, -0.5)
        assert_raises(ValueError, firwin, 99, 1.5)
        # Don't allow 0 or 1 in cutoff.
        assert_raises(ValueError, firwin, 99, [0, 0.5])
        assert_raises(ValueError, firwin, 99, [0.5, 1])
        # cutoff values must be strictly increasing.
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.2])
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.5])
        # Must have at least one cutoff value.
        assert_raises(ValueError, firwin, 99, [])
        # 2D array not allowed.
        assert_raises(ValueError, firwin, 99, [[0.1, 0.2],[0.3, 0.4]])
        # cutoff values must be less than nyq.
        assert_raises(ValueError, firwin, 99, 50.0, fs=80)
        assert_raises(ValueError, firwin, 99, [10, 20, 30], fs=50)



# 定义测试函数，用于检验 firwin 函数在不合法截止频率参数时是否会引发 ValueError 异常
def test_bad_cutoff(self):
    """Test that invalid cutoff argument raises ValueError."""
    # 截止频率值必须大于 0 且小于 1
    assert_raises(ValueError, firwin, 99, -0.5)
    assert_raises(ValueError, firwin, 99, 1.5)
    # 不允许截止频率值为 0 或 1
    assert_raises(ValueError, firwin, 99, [0, 0.5])
    assert_raises(ValueError, firwin, 99, [0.5, 1])
    # 截止频率值必须严格递增
    assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.2])
    assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.5])
    # 至少需要一个截止频率值
    assert_raises(ValueError, firwin, 99, [])
    # 不允许使用二维数组作为截止频率值
    assert_raises(ValueError, firwin, 99, [[0.1, 0.2],[0.3, 0.4]])
    # 截止频率值必须小于 nyq（奈奎斯特频率）
    assert_raises(ValueError, firwin, 99, 50.0, fs=80)
    assert_raises(ValueError, firwin, 99, [10, 20, 30], fs=50)


这段代码是用于测试 `firwin` 函数在不合法的截止频率参数情况下是否会正确地引发 `ValueError` 异常，并对异常情况进行了详细的断言和验证。
    # 定义测试函数，用于验证当尝试使用偶数个 taps 创建高通滤波器时是否会引发 ValueError 异常
    def test_even_highpass_raises_value_error(self):
        """Test that attempt to create a highpass filter with an even number
        of taps raises a ValueError exception."""
        # 断言调用 firwin 函数时传递参数 40, 0.5, pass_zero=False 会引发 ValueError 异常
        assert_raises(ValueError, firwin, 40, 0.5, pass_zero=False)
        # 断言调用 firwin 函数时传递参数 40, [.25, 0.5] 会引发 ValueError 异常
        assert_raises(ValueError, firwin, 40, [.25, 0.5])

    # 定义测试函数，测试 pass_zero 参数不合法的情况
    def test_bad_pass_zero(self):
        """Test degenerate pass_zero cases."""
        # 使用 assert_raises 验证当 pass_zero 参数为 'foo' 时，调用 firwin 函数会引发 ValueError 异常
        with assert_raises(ValueError, match='pass_zero must be'):
            firwin(41, 0.5, pass_zero='foo')
        # 使用 assert_raises 验证当 pass_zero 参数为 1.0 时，调用 firwin 函数会引发 TypeError 异常
        with assert_raises(TypeError, match='cannot be interpreted'):
            firwin(41, 0.5, pass_zero=1.)
        # 遍历 pass_zero 参数为 'lowpass' 和 'highpass' 时，验证调用 firwin 函数是否会引发 ValueError 异常
        for pass_zero in ('lowpass', 'highpass'):
            with assert_raises(ValueError, match='cutoff must have one'):
                firwin(41, [0.5, 0.6], pass_zero=pass_zero)
        # 遍历 pass_zero 参数为 'bandpass' 和 'bandstop' 时，验证调用 firwin 函数是否会引发 ValueError 异常
        for pass_zero in ('bandpass', 'bandstop'):
            with assert_raises(ValueError, match='must have at least two'):
                firwin(41, [0.5], pass_zero=pass_zero)

    # 定义测试函数，测试 fs 参数的验证
    def test_fs_validation(self):
        # 使用 pytest.raises 验证当 fs 参数为 np.array([10, 20]) 时，调用 firwin2 函数会引发 ValueError 异常
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            firwin2(51, .5, 1, fs=np.array([10, 20]))
class TestFirwin2:

    def test_invalid_args(self):
        # 当 `freq` 和 `gain` 的长度不同时，引发值错误异常
        with assert_raises(ValueError, match='must be of same length'):
            firwin2(50, [0, 0.5, 1], [0.0, 1.0])
        
        # 当 `nfreqs` 小于 `ntaps` 时，引发值错误异常
        with assert_raises(ValueError, match='ntaps must be less than nfreqs'):
            firwin2(50, [0, 0.5, 1], [0.0, 1.0, 1.0], nfreqs=33)
        
        # 当 `freq` 中出现非递增的值时，引发值错误异常
        with assert_raises(ValueError, match='must be nondecreasing'):
            firwin2(50, [0, 0.5, 0.4, 1.0], [0, .25, .5, 1.0])
        
        # 当 `freq` 中某个值重复超过两次时，引发值错误异常
        with assert_raises(ValueError, match='must not occur more than twice'):
            firwin2(50, [0, .1, .1, .1, 1.0], [0.0, 0.5, 0.75, 1.0, 1.0])
        
        # 当 `freq` 不以 0.0 开头时，引发值错误异常
        with assert_raises(ValueError, match='start with 0'):
            firwin2(50, [0.5, 1.0], [0.0, 1.0])
        
        # 当 `freq` 不以 fs/2 结尾时，引发值错误异常
        with assert_raises(ValueError, match='end with fs/2'):
            firwin2(50, [0.0, 0.5], [0.0, 1.0])
        
        # 当 `freq` 中出现重复的值 0.0 时，引发值错误异常
        with assert_raises(ValueError, match='0 must not be repeated'):
            firwin2(50, [0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        
        # 当 `freq` 中出现重复的值 fs/2 时，引发值错误异常
        with assert_raises(ValueError, match='fs/2 must not be repeated'):
            firwin2(50, [0.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
        
        # 当 `freq` 中某个值与其周围数值过于接近时，引发值错误异常
        with assert_raises(ValueError, match='cannot contain numbers that are too close'):
            firwin2(50, [0.0, 0.5 - np.finfo(float).eps * 0.5, 0.5, 0.5, 1.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0])

        # 当 Type II 滤波器的情况下，Nyquist 频率处的增益不为零时，引发值错误异常
        with assert_raises(ValueError, match='Type II filter'):
            firwin2(16, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0])

        # 当 Type III 滤波器的情况下，Nyquist 频率和零频率处的增益不为零时，引发值错误异常
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], antisymmetric=True)

        # 当 Type IV 滤波器的情况下，零频率处的增益不为零时，引发值错误异常
        with assert_raises(ValueError, match='Type IV filter'):
            firwin2(16, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)
    def test01(self):
        width = 0.04
        beta = 12.0
        ntaps = 400
        # 定义频率和增益的列表，描述滤波器的频率响应特性
        freq = [0.0, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0]
        # 使用 firwin2 函数设计 FIR 滤波器，基于给定的频率和增益特性
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        # 定义用于计算频率响应的频率样本点
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2,
                                                        0.75, 1.0-width/2])
        # 计算滤波器的频率响应
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        # 断言滤波器的频率响应与预期值的近似性
        assert_array_almost_equal(np.abs(response),
                        [1.0, 1.0, 1.0, 1.0-width, 0.5, width], decimal=5)

    def test02(self):
        width = 0.04
        beta = 12.0
        ntaps = 401
        # 定义频率和增益的列表，描述滤波器的频率响应特性
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [0.0, 0.0, 1.0, 1.0]
        # 使用 firwin2 函数设计 FIR 滤波器，基于给定的频率和增益特性
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        # 定义用于计算频率响应的频率样本点
        freq_samples = np.array([0.0, 0.25, 0.5-width, 0.5+width, 0.75, 1.0])
        # 计算滤波器的频率响应
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        # 断言滤波器的频率响应与预期值的近似性
        assert_array_almost_equal(np.abs(response),
                                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

    def test03(self):
        width = 0.02
        # 使用 kaiserord 函数计算滤波器的阶数和贝塔值
        ntaps, beta = kaiserord(120, width)
        # 确保滤波器阶数为奇数，以确保 Nyquist 频率处有正增益
        ntaps = int(ntaps) | 1
        # 定义频率和增益的列表，描述滤波器的频率响应特性
        freq = [0.0, 0.4, 0.4, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        # 使用 firwin2 函数设计 FIR 滤波器，基于给定的频率和增益特性
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        # 定义用于计算频率响应的频率样本点
        freq_samples = np.array([0.0, 0.4-width, 0.4+width, 0.45,
                                    0.5-width, 0.5+width, 0.75, 1.0])
        # 计算滤波器的频率响应
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        # 断言滤波器的频率响应与预期值的近似性
        assert_array_almost_equal(np.abs(response),
                    [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

    def test04(self):
        """Test firwin2 when window=None."""
        ntaps = 5
        # 定义频率和增益的列表，描述滤波器的频率响应特性
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0]
        # 使用 firwin2 函数设计 FIR 滤波器，基于给定的频率和增益特性，窗口函数为 None
        taps = firwin2(ntaps, freq, gain, window=None, nfreqs=8193)
        # 计算理想低通滤波器的冲激响应
        alpha = 0.5 * (ntaps - 1)
        m = np.arange(0, ntaps) - alpha
        h = 0.5 * sinc(0.5 * m)
        # 断言滤波器的冲激响应与预期值的近似性
        assert_array_almost_equal(h, taps)

    def test05(self):
        """Test firwin2 for calculating Type IV filters"""
        ntaps = 1500
        # 定义频率和增益的列表，描述滤波器的频率响应特性
        freq = [0.0, 1.0]
        gain = [0.0, 1.0]
        # 使用 firwin2 函数设计 FIR 滤波器，基于给定的频率和增益特性，选择对称性
        taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        # 断言滤波器的前一半系数与后一半系数的反向相等性
        assert_array_almost_equal(taps[: ntaps // 2], -taps[ntaps // 2:][::-1])

        # 计算滤波器的频率响应
        freqs, response = freqz(taps, worN=2048)
        # 断言滤波器的频率响应与预期值的近似性
        assert_array_almost_equal(abs(response), freqs / np.pi, decimal=4)
    def test06(self):
        """Test firwin2 for calculating Type III filters"""
        ntaps = 1501  # 定义滤波器的长度为1501个点

        freq = [0.0, 0.5, 0.55, 1.0]  # 定义滤波器通频带的频率点
        gain = [0.0, 0.5, 0.0, 0.0]   # 定义每个频率点对应的增益
        taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)  # 使用 firwin2 函数生成滤波器系数
        assert_equal(taps[ntaps // 2], 0.0)  # 断言滤波器中间点的系数为0
        assert_array_almost_equal(taps[: ntaps // 2], -taps[ntaps // 2 + 1:][::-1])  # 断言滤波器左右对称性

        freqs, response1 = freqz(taps, worN=2048)  # 计算滤波器的频率响应
        response2 = np.interp(freqs / np.pi, freq, gain)  # 使用线性插值得到期望的频率响应
        assert_array_almost_equal(abs(response1), response2, decimal=3)  # 断言实际频率响应与期望频率响应的一致性

    def test_fs_nyq(self):
        taps1 = firwin2(80, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])  # 生成滤波器系数，指定频率和增益
        taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], fs=120.0)  # 指定采样率生成另一个滤波器系数
        assert_array_almost_equal(taps1, taps2)  # 断言两组滤波器系数的一致性

    def test_tuple(self):
        taps1 = firwin2(150, (0.0, 0.5, 0.5, 1.0), (1.0, 1.0, 0.0, 0.0))  # 使用元组形式的频率和增益生成滤波器系数
        taps2 = firwin2(150, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])  # 使用列表形式的频率和增益生成滤波器系数
        assert_array_almost_equal(taps1, taps2)  # 断言两组滤波器系数的一致性

    def test_input_modyfication(self):
        freq1 = np.array([0.0, 0.5, 0.5, 1.0])  # 创建一个频率数组
        freq2 = np.array(freq1)  # 复制一份频率数组
        firwin2(80, freq1, [1.0, 1.0, 0.0, 0.0])  # 调用 firwin2 函数生成滤波器系数，修改 freq1 的内容
        assert_equal(freq1, freq2)  # 断言修改后的 freq1 与原始的 freq2 相同
class TestRemez:
    
    def test_bad_args(self):
        # 断言：调用 remez 函数应该引发 ValueError 异常，因为 type 参数设置为 'pooka' 是无效的
        assert_raises(ValueError, remez, 11, [0.1, 0.4], [1], type='pooka')

    def test_hilbert(self):
        N = 11  # 滤波器的阶数
        a = 0.1  # 过渡带的宽度

        # 设计一个单位增益的 Hilbert 带通滤波器，从 w 到 0.5-w
        h = remez(11, [a, 0.5-a], [1], type='hilbert')

        # 确保滤波器具有正确数量的阶数
        assert_(len(h) == N, "Number of Taps")

        # 确保它是 III 型（反对称的系数）
        assert_array_almost_equal(h[:(N-1)//2], -h[:-(N-1)//2-1:-1])

        # 由于请求的响应是对称的，所有偶数系数应该是零（或者在这种情况下非常小）
        assert_((abs(h[1::2]) < 1e-15).all(), "Even Coefficients Equal Zero")

        # 现在检查频率响应
        w, H = freqz(h, 1)
        f = w/2/np.pi
        Hmag = abs(H)

        # 应该在 0 和 pi 处有一个零点（在这种情况下接近零）
        assert_((Hmag[[0, -1]] < 0.02).all(), "Zero at zero and pi")

        # 检查通带接近于单位增益
        idx = np.logical_and(f > a, f < 0.5-a)
        assert_((abs(Hmag[idx] - 1) < 0.015).all(), "Pass Band Close To Unity")

    def test_compare(self):
        # 测试与 MATLAB 的比较
        k = [0.024590270518440, -0.041314581814658, -0.075943803756711,
             -0.003530911231040, 0.193140296954975, 0.373400753484939,
             0.373400753484939, 0.193140296954975, -0.003530911231040,
             -0.075943803756711, -0.041314581814658, 0.024590270518440]
        # 测试 remez 函数生成的滤波器系数与预期的系数 k 的近似度
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.)
        assert_allclose(h, k)

        h = [-0.038976016082299, 0.018704846485491, -0.014644062687875,
             0.002879152556419, 0.016849978528150, -0.043276706138248,
             0.073641298245579, -0.103908158578635, 0.129770906801075,
             -0.147163447297124, 0.153302248456347, -0.147163447297124,
             0.129770906801075, -0.103908158578635, 0.073641298245579,
             -0.043276706138248, 0.016849978528150, 0.002879152556419,
             -0.014644062687875, 0.018704846485491, -0.038976016082299]
        # 测试 remez 函数生成的滤波器系数与预期的系数 h 的近似度
        assert_allclose(remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.), h)

    def test_fs_validation(self):
        # 使用 pytest 断言：应该引发 ValueError 异常，错误信息包含 "Sampling" 并且 fs 参数是一个数组而不是单一的标量
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            remez(11, .1, 1, fs=np.array([10, 20]))

class TestFirls:
    # 定义一个测试函数，用于测试 firls 函数在不良参数下是否能正确抛出 ValueError 异常
    def test_bad_args(self):
        # 测试当 numtaps 是偶数时，是否会抛出 ValueError 异常
        assert_raises(ValueError, firls, 10, [0.1, 0.2], [0, 0])
        # 测试当 bands 列表长度为奇数时，是否会抛出 ValueError 异常
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.4], [0, 0, 0])
        # 测试当 bands 列表长度与 desired 列表长度不一致时，是否会抛出 ValueError 异常
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.4], [0, 0, 0])
        # 测试当 bands 列表不是单调递增或递减时，是否会抛出 ValueError 异常
        assert_raises(ValueError, firls, 11, [0.2, 0.1], [0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.3], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.3, 0.4, 0.1, 0.2], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.1, 0.3, 0.2, 0.4], [0] * 4)
        # 测试当 desired 列表中包含负数时，是否会抛出 ValueError 异常
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [-1, 1])
        # 测试当 weight 列表长度与 pairs 列表长度不一致时，是否会抛出 ValueError 异常
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], weight=[1, 2])
        # 测试当 weight 列表中包含负数时，是否会抛出 ValueError 异常
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], weight=[-1])

    # 定义一个测试函数，用于测试 firls 函数的输出是否符合预期
    def test_firls(self):
        N = 11  # 滤波器的 taps 数量
        a = 0.1  # 过渡带的宽度

        # 设计一个半波段对称低通滤波器
        h = firls(11, [0, a, 0.5-a, 0.5], [1, 1, 0, 0], fs=1.0)

        # 确保滤波器的 taps 数量正确
        assert_equal(len(h), N)

        # 确保滤波器是对称的
        midx = (N-1) // 2
        assert_array_almost_equal(h[:midx], h[:-midx-1:-1])

        # 确保中间的 tap 值为 0.5
        assert_almost_equal(h[midx], 0.5)

        # 对于半波段对称滤波器，奇数位系数（除了中间的）应该接近于零
        hodd = np.hstack((h[1:midx:2], h[-midx+1::2]))
        assert_array_almost_equal(hodd, 0)

        # 现在检查频率响应
        w, H = freqz(h, 1)
        f = w/2/np.pi
        Hmag = np.abs(H)

        # 检查通带是否接近于单位增益
        idx = np.logical_and(f > 0, f < a)
        assert_array_almost_equal(Hmag[idx], 1, decimal=3)

        # 检查阻带是否接近于零增益
        idx = np.logical_and(f > 0.5-a, f < 0.5)
        assert_array_almost_equal(Hmag[idx], 0, decimal=3)
    # 定义一个测试函数，用于测试 firls 函数的输出是否与预期值接近

    def test_compare(self):
        # 与 OCTAVE 输出进行比较
        taps = firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], weight=[1, 2])
        # >> taps = firls(8, [0 0.5 0.55 1], [1 1 0 0], [1, 2]);
        known_taps = [-6.26930101730182e-04, -1.03354450635036e-01,
                      -9.81576747564301e-03, 3.17271686090449e-01,
                      5.11409425599933e-01, 3.17271686090449e-01,
                      -9.81576747564301e-03, -1.03354450635036e-01,
                      -6.26930101730182e-04]
        assert_allclose(taps, known_taps)

        # 与 MATLAB 输出进行比较
        taps = firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], weight=[1, 2])
        # >> taps = firls(10, [0 0.5 0.5 1], [1 1 0 0], [1, 2]);
        known_taps = [
            0.058545300496815, -0.014233383714318, -0.104688258464392,
            0.012403323025279, 0.317930861136062, 0.488047220029700,
            0.317930861136062, 0.012403323025279, -0.104688258464392,
            -0.014233383714318, 0.058545300496815]
        assert_allclose(taps, known_taps)

        # 带有线性变化的比较
        taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], fs=20)
        # >> taps = firls(6, [0, 0.1, 0.2, 0.3, 0.4, 0.5], [1, 0, 0, 1, 1, 0])
        known_taps = [
            1.156090832768218, -4.1385894727395849, 7.5288619164321826,
            -8.5530572592947856, 7.5288619164321826, -4.1385894727395849,
            1.156090832768218]
        assert_allclose(taps, known_taps)

    def test_rank_deficient(self):
        # solve() 可能会警告（有时），此处不使用 match 参数
        x = firls(21, [0, 0.1, 0.9, 1], [1, 1, 0, 0])
        w, h = freqz(x, fs=2.)
        assert_allclose(np.abs(h[:2]), 1., atol=1e-5)
        assert_allclose(np.abs(h[-2:]), 0., atol=1e-6)
        # 切换到 pinvh（对于更长的滤波器，容差可以更高，但使用较短的滤波器在计算速度上更快，
        # 思想是相同的）
        x = firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        w, h = freqz(x, fs=2.)
        mask = w < 0.01
        assert mask.sum() > 3
        assert_allclose(np.abs(h[mask]), 1., atol=1e-4)
        mask = w > 0.99
        assert mask.sum() > 3
        assert_allclose(np.abs(h[mask]), 0., atol=1e-4)

    def test_fs_validation(self):
        # 使用 pytest 检查是否会引发值错误，并确保错误信息匹配指定模式
        with pytest.raises(ValueError, match="Sampling.*single scalar"):
            firls(11, .1, 1, fs=np.array([10, 20]))
class TestMinimumPhase:

    def test_bad_args(self):
        # not enough taps
        # 测试参数不足情况下的异常抛出
        assert_raises(ValueError, minimum_phase, [1.])
        assert_raises(ValueError, minimum_phase, [1., 1.])
        assert_raises(ValueError, minimum_phase, np.full(10, 1j))
        assert_raises(ValueError, minimum_phase, 'foo')
        assert_raises(ValueError, minimum_phase, np.ones(10), n_fft=8)
        assert_raises(ValueError, minimum_phase, np.ones(10), method='foo')
        assert_warns(RuntimeWarning, minimum_phase, np.arange(3))
        # 使用 pytest 模块测试抛出特定异常并给出匹配字符串的警告
        with pytest.raises(ValueError, match="is only supported when"):
            minimum_phase(np.ones(3), method='hilbert', half=False)

    def test_homomorphic(self):
        # check that it can recover frequency responses of arbitrary
        # linear-phase filters
        # 检查它是否可以恢复任意线性相位滤波器的频率响应

        # for some cases we can get the actual filter back
        # 一些情况下，我们可以获得实际的滤波器
        h = [1, -1]
        h_new = minimum_phase(np.convolve(h, h[::-1]))
        assert_allclose(h_new, h, rtol=0.05)

        # but in general we only guarantee we get the magnitude back
        # 但通常我们只保证我们能恢复幅度响应
        rng = np.random.RandomState(0)
        for n in (2, 3, 10, 11, 15, 16, 17, 20, 21, 100, 101):
            h = rng.randn(n)
            h_linear = np.convolve(h, h[::-1])
            h_new = minimum_phase(h_linear)
            assert_allclose(np.abs(fft(h_new)), np.abs(fft(h)), rtol=1e-4)
            h_new = minimum_phase(h_linear, half=False)
            assert len(h_linear) == len(h_new)
            assert_allclose(np.abs(fft(h_new)), np.abs(fft(h_linear)), rtol=1e-4)

    def test_hilbert(self):
        # compare to MATLAB output of reference implementation
        # 与 MATLAB 参考实现输出进行比较

        # f=[0 0.3 0.5 1];
        # a=[1 1 0 0];
        # h=remez(11,f,a);
        # 定义一个滤波器 h，并使用 remez 函数生成其系数
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.)
        k = [0.349585548646686, 0.373552164395447, 0.326082685363438,
             0.077152207480935, -0.129943946349364, -0.059355880509749]
        # 使用最小相位函数对 h 进行处理，使用 'hilbert' 方法，并进行近似值检查
        m = minimum_phase(h, 'hilbert')
        assert_allclose(m, k, rtol=5e-3)

        # f=[0 0.8 0.9 1];
        # a=[0 0 1 1];
        # h=remez(20,f,a);
        # 定义另一个滤波器 h，并使用 remez 函数生成其系数
        h = remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.)
        k = [0.232486803906329, -0.133551833687071, 0.151871456867244,
             -0.157957283165866, 0.151739294892963, -0.129293146705090,
             0.100787844523204, -0.065832656741252, 0.035361328741024,
             -0.014977068692269, -0.158416139047557]
        # 使用最小相位函数对 h 进行处理，使用 'hilbert' 方法和额外的 n_fft 参数，并进行近似值检查
        m = minimum_phase(h, 'hilbert', n_fft=2**19)
        assert_allclose(m, k, rtol=2e-3)
```