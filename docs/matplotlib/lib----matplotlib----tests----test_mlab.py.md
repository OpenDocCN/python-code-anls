# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_mlab.py`

```
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_equal, assert_array_almost_equal_nulp)
import numpy as np
import pytest

from matplotlib import mlab  # 导入 mlab 模块


def test_window():  # 定义单元测试函数 test_window
    np.random.seed(0)  # 设置随机种子为 0
    n = 1000  # 定义变量 n 为 1000
    rand = np.random.standard_normal(n) + 100  # 生成标准正态分布随机数并加上 100
    ones = np.ones(n)  # 生成元素全为 1 的数组

    assert_array_equal(mlab.window_none(ones), ones)  # 断言调用 mlab.window_none 函数结果与 ones 相等
    assert_array_equal(mlab.window_none(rand), rand)  # 断言调用 mlab.window_none 函数结果与 rand 相等
    assert_array_equal(np.hanning(len(rand)) * rand, mlab.window_hanning(rand))  # 断言汉宁窗函数应用于 rand 后结果与调用 mlab.window_hanning 函数结果相等
    assert_array_equal(np.hanning(len(ones)), mlab.window_hanning(ones))  # 断言汉宁窗函数应用于 ones 后结果与调用 mlab.window_hanning 函数结果相等


class TestDetrend:  # 定义测试类 TestDetrend
    def setup_method(self):  # 设置测试方法的初始化
        np.random.seed(0)  # 设置随机种子为 0
        n = 1000  # 定义变量 n 为 1000
        x = np.linspace(0., 100, n)  # 生成从 0 到 100 的 n 个等间隔的数

        self.sig_zeros = np.zeros(n)  # 生成元素全为 0 的数组 self.sig_zeros

        self.sig_off = self.sig_zeros + 100.  # self.sig_off 是 self.sig_zeros 加上 100
        self.sig_slope = np.linspace(-10., 90., n)  # self.sig_slope 是从 -10 到 90 的 n 个等间隔的数
        self.sig_slope_mean = x - x.mean()  # self.sig_slope_mean 是 x 减去 x 的平均值

        self.sig_base = (
            np.random.standard_normal(n) + np.sin(x*2*np.pi/(n/100)))  # self.sig_base 是标准正态分布随机数加上 sin 函数的结果
        self.sig_base -= self.sig_base.mean()  # self.sig_base 减去其均值

    def allclose(self, *args):  # 定义辅助方法 allclose，用于比较是否全部接近
        assert_allclose(*args, atol=1e-8)  # 断言所有参数接近，绝对容差为 1e-8

    def test_detrend_none(self):  # 定义测试去趋势的方法 test_detrend_none
        assert mlab.detrend_none(0.) == 0.  # 断言 mlab.detrend_none(0.) 的结果为 0
        assert mlab.detrend_none(0., axis=1) == 0.  # 断言 mlab.detrend_none(0., axis=1) 的结果为 0
        assert mlab.detrend(0., key="none") == 0.  # 断言 mlab.detrend(0., key="none") 的结果为 0
        assert mlab.detrend(0., key=mlab.detrend_none) == 0.  # 断言 mlab.detrend(0., key=mlab.detrend_none) 的结果为 0

        for sig in [
                5.5, self.sig_off, self.sig_slope, self.sig_base,
                (self.sig_base + self.sig_slope + self.sig_off).tolist(),
                np.vstack([self.sig_base,  # 2D case.
                           self.sig_base + self.sig_off,
                           self.sig_base + self.sig_slope,
                           self.sig_base + self.sig_off + self.sig_slope]),
                np.vstack([self.sig_base,  # 2D transposed case.
                           self.sig_base + self.sig_off,
                           self.sig_base + self.sig_slope,
                           self.sig_base + self.sig_off + self.sig_slope]).T,
        ]:
            if isinstance(sig, np.ndarray):  # 如果 sig 是 numpy 数组
                assert_array_equal(mlab.detrend_none(sig), sig)  # 断言 mlab.detrend_none(sig) 的结果与 sig 相等
            else:
                assert mlab.detrend_none(sig) == sig  # 断言 mlab.detrend_none(sig) 的结果与 sig 相等

    def test_detrend_mean(self):  # 定义测试去均值趋势的方法 test_detrend_mean
        for sig in [0., 5.5]:  # 对于 sig 在列表 [0., 5.5] 中的每个值
            assert mlab.detrend_mean(sig) == 0.  # 断言 mlab.detrend_mean(sig) 的结果为 0
            assert mlab.detrend(sig, key="mean") == 0.  # 断言 mlab.detrend(sig, key="mean") 的结果为 0
            assert mlab.detrend(sig, key=mlab.detrend_mean) == 0.  # 断言 mlab.detrend(sig, key=mlab.detrend_mean) 的结果为 0
        
        # 1D.
        self.allclose(mlab.detrend_mean(self.sig_zeros), self.sig_zeros)  # 断言 mlab.detrend_mean(self.sig_zeros) 的结果与 self.sig_zeros 接近
        self.allclose(mlab.detrend_mean(self.sig_base), self.sig_base)  # 断言 mlab.detrend_mean(self.sig_base) 的结果与 self.sig_base 接近
        self.allclose(mlab.detrend_mean(self.sig_base + self.sig_off),
                      self.sig_base)  # 断言 mlab.detrend_mean(self.sig_base + self.sig_off) 的结果与 self.sig_base 接近
        self.allclose(mlab.detrend_mean(self.sig_base + self.sig_slope),
                      self.sig_base + self.sig_slope_mean)  # 断言 mlab.detrend_mean(self.sig_base + self.sig_slope) 的结果与 self.sig_base + self.sig_slope_mean 接近
        self.allclose(
            mlab.detrend_mean(self.sig_base + self.sig_slope + self.sig_off),
            self.sig_base + self.sig_slope_mean)  # 断言 mlab.detrend_mean(self.sig_base + self.sig_slope + self.sig_off) 的结果与 self.sig_base + self.sig_slope_mean 接近
    # 定义单元测试函数，用于测试 detrend_mean 函数在不同输入情况下的表现
    def test_detrend_mean_1d_base_slope_off_list_andor_axis0(self):
        # 将基础信号、斜率信号和偏移信号合并成一个输入信号
        input = self.sig_base + self.sig_slope + self.sig_off
        # 计算期望的目标信号，为基础信号加上斜率信号的平均值
        target = self.sig_base + self.sig_slope_mean
        # 检查使用 axis=0 参数进行均值去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend_mean(input, axis=0), target)
        # 将输入信号转换为列表并检查去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend_mean(input.tolist()), target)
        # 将输入信号转换为列表，并使用 axis=0 参数进行均值去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend_mean(input.tolist(), axis=0), target)
    
    # 定义单元测试函数，用于测试 detrend_mean 函数在二维输入情况下的表现
    def test_detrend_mean_2d(self):
        # 构建二维输入信号，第一行为偏移信号，第二行为基础信号加偏移信号
        input = np.vstack([self.sig_off,
                           self.sig_base + self.sig_off])
        # 构建期望的目标信号，第一行为全零信号，第二行为基础信号
        target = np.vstack([self.sig_zeros,
                            self.sig_base])
        # 检查使用默认 axis=0 参数进行均值去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend_mean(input), target)
        # 检查使用 axis=None 参数进行全局均值去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend_mean(input, axis=None), target)
        # 检查将转置后的输入信号，使用 axis=None 参数进行全局均值去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend_mean(input.T, axis=None).T, target)
        # 检查使用默认参数进行均值去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend(input), target)
        # 检查使用 axis=None 参数进行全局去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend(input, axis=None), target)
        # 检查将转置后的输入信号，使用 key="constant" 和 axis=None 参数进行全局去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend(input.T, key="constant", axis=None), target.T)
    
        # 构建另一种二维输入信号，包含四行：基础信号、基础信号加偏移信号、基础信号加斜率信号、基础信号加偏移和斜率信号
        input = np.vstack([self.sig_base,
                           self.sig_base + self.sig_off,
                           self.sig_base + self.sig_slope,
                           self.sig_base + self.sig_off + self.sig_slope])
        # 构建期望的目标信号，包含四行：基础信号、基础信号、基础信号加上斜率信号的平均值、基础信号加上斜率信号的平均值
        target = np.vstack([self.sig_base,
                            self.sig_base,
                            self.sig_base + self.sig_slope_mean,
                            self.sig_base + self.sig_slope_mean])
        # 检查将转置后的输入信号，使用 axis=0 参数进行均值去趋势后的输出是否与期望的目标信号转置后的结果相近
        self.allclose(mlab.detrend_mean(input.T, axis=0), target.T)
        # 检查使用 axis=1 参数进行均值去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend_mean(input, axis=1), target)
        # 检查使用 axis=-1 参数进行均值去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend_mean(input, axis=-1), target)
        # 检查使用 key="default" 和 axis=1 参数进行默认去趋势后的输出是否与期望的目标信号相近
        self.allclose(mlab.detrend(input, key="default", axis=1), target)
        # 检查将转置后的输入信号，使用 key="mean" 和 axis=0 参数进行均值去趋势后的输出是否与期望的目标信号转置后的结果相近
        self.allclose(mlab.detrend(input.T, key="mean", axis=0), target.T)
        # 检查将转置后的输入信号，使用 key=mlab.detrend_mean 和 axis=0 参数进行均值去趋势后的输出是否与期望的目标信号转置后的结果相近
        self.allclose(mlab.detrend(input.T, key=mlab.detrend_mean, axis=0), target.T)
    
    # 定义单元测试函数，用于测试在输入错误情况下是否能抛出 ValueError 异常
    def test_detrend_ValueError(self):
        for signal, kwargs in [
                # 测试将一维斜率信号作为输入，并传入无效参数 key="spam"
                (self.sig_slope[np.newaxis], {"key": "spam"}),
                # 测试将一维斜率信号作为输入，并传入无效参数 key=5
                (self.sig_slope[np.newaxis], {"key": 5}),
                # 测试将标量 5.5 作为输入，并传入无效参数 axis=0
                (5.5, {"axis": 0}),
                # 测试将一维斜率信号作为输入，并传入无效参数 axis=1
                (self.sig_slope, {"axis": 1}),
                # 测试将一维斜率信号作为输入，并传入无效参数 axis=2
                (self.sig_slope[np.newaxis], {"axis": 2}),
        ]:
            # 使用 pytest 检查调用 detrend 函数时是否抛出 ValueError 异常
            with pytest.raises(ValueError):
                mlab.detrend(signal, **kwargs)
    
    # 定义单元测试函数，用于测试 detrend_mean 函数在输入错误情况下是否能抛出 ValueError 异常
    def test_detrend_mean_ValueError(self):
        for signal, kwargs in [
                # 测试将标量 5.5 作为输入，并传入无效参数 axis=0
                (5.5, {"axis": 0}),
                # 测试将一维斜率信号作为输入，并传入无效参数 axis=1
                (self.sig_slope, {"axis": 1}),
                # 测试将一维斜率信号作为输入，并传入无效参数 axis=2
                (self.sig_slope[np.newaxis], {"axis": 2}),
        ]:
            # 使用 pytest 检查调用 detrend_mean 函数时是否抛出 ValueError 异常
            with pytest.raises(ValueError):
                mlab.detrend_mean(signal, **kwargs)
    # 定义测试函数，用于测试线性去趋势的功能
    def test_detrend_linear(self):
        # 0D 情况下，输入为0时返回0
        assert mlab.detrend_linear(0.) == 0.
        # 输入为非零常数时，线性去趋势后应返回0
        assert mlab.detrend_linear(5.5) == 0.
        # 使用"linear"关键字调用detrend函数，等效于调用detrend_linear，输入5.5时应返回0
        assert mlab.detrend(5.5, key="linear") == 0.
        # 使用mlab.detrend_linear函数作为关键字调用detrend函数，输入5.5时应返回0
        assert mlab.detrend(5.5, key=mlab.detrend_linear) == 0.
        
        # 对于1D信号，测试以下几种情况
        for sig in [
                self.sig_off,                      # 偏移信号
                self.sig_slope,                    # 斜坡信号
                self.sig_slope + self.sig_off,     # 斜坡加偏移信号
        ]:
            # 检查线性去趋势后的信号是否与预期的全零信号相近
            self.allclose(mlab.detrend_linear(sig), self.sig_zeros)

    # 测试1D信号线性去趋势的函数，输入为信号加斜坡和偏移
    def test_detrend_str_linear_1d(self):
        input = self.sig_slope + self.sig_off
        target = self.sig_zeros
        # 使用"linear"关键字调用detrend函数，期望输出与target相近
        self.allclose(mlab.detrend(input, key="linear"), target)
        # 使用mlab.detrend_linear函数作为关键字调用detrend函数，期望输出与target相近
        self.allclose(mlab.detrend(input, key=mlab.detrend_linear), target)
        # 将1D输入转换为列表并调用mlab.detrend_linear，期望输出与target相近
        self.allclose(mlab.detrend_linear(input.tolist()), target)

    # 测试2D信号线性去趋势的函数
    def test_detrend_linear_2d(self):
        input = np.vstack([self.sig_off,         # 偏移信号
                           self.sig_slope,       # 斜坡信号
                           self.sig_slope + self.sig_off])   # 斜坡加偏移信号
        target = np.vstack([self.sig_zeros,       # 全零信号
                            self.sig_zeros,       # 全零信号
                            self.sig_zeros])      # 全零信号
        
        # 对输入信号按列进行线性去趋势，期望输出与target转置相近
        self.allclose(
            mlab.detrend(input.T, key="linear", axis=0), target.T)
        # 对输入信号按列使用mlab.detrend_linear函数进行线性去趋势，期望输出与target转置相近
        self.allclose(
            mlab.detrend(input.T, key=mlab.detrend_linear, axis=0), target.T)
        # 对输入信号按行进行线性去趋势，期望输出与target相近
        self.allclose(
            mlab.detrend(input, key="linear", axis=1), target)
        # 对输入信号按行使用mlab.detrend_linear函数进行线性去趋势，期望输出与target相近
        self.allclose(
            mlab.detrend(input, key=mlab.detrend_linear, axis=1), target)
        
        # 测试当输入为行向量时是否引发值错误异常
        with pytest.raises(ValueError):
            mlab.detrend_linear(self.sig_slope[np.newaxis])
# 使用 pytest.mark.parametrize 装饰器为 TestSpectral 类定义参数化测试，测试用例涵盖不同条件下的输入参数组合
@pytest.mark.parametrize('iscomplex', [False, True],
                         ids=['real', 'complex'], scope='class')
@pytest.mark.parametrize('sides', ['onesided', 'twosided', 'default'],
                         scope='class')
@pytest.mark.parametrize(
    'fstims,len_x,NFFT_density,nover_density,pad_to_density,pad_to_spectrum',
    [
        ([], None, -1, -1, -1, -1),  # 无信号数据的情况
        ([4], None, -1, -1, -1, -1),  # 单个信号数据
        ([4, 5, 10], None, -1, -1, -1, -1),  # 多个信号数据
        ([], None, None, -1, -1, None),  # 无信号数据，无NFFT和部分pad_to情况
        ([], None, -1, -1, None, None),  # 无信号数据，无pad_to情况
        ([], None, None, -1, None, None),  # 无信号数据，无NFFT和pad_to情况
        ([], 1024, 512, -1, -1, 128),  # 信号数据，指定len_x、NFFT_density和pad_to_spectrum参数
        ([], 256, -1, -1, 33, 257),  # 信号数据，指定len_x、pad_to_density和pad_to_spectrum参数
        ([], 255, 33, -1, -1, None),  # 信号数据，指定len_x和NFFT_density参数
        ([], 256, 128, -1, 256, 256),  # 信号数据，指定len_x、NFFT_density、nover_density和pad_to_spectrum参数
        ([], None, -1, 32, -1, -1),  # 无信号数据，指定nover_density参数
    ],
    ids=[
        'nosig',
        'Fs4',
        'FsAll',
        'nosig_noNFFT',
        'nosig_nopad_to',
        'nosig_noNFFT_no_pad_to',
        'nosig_trim',
        'nosig_odd',
        'nosig_oddlen',
        'nosig_stretch',
        'nosig_overlap',
    ],
    scope='class')
class TestSpectral:
    # 使用 pytest.fixture 装饰器定义一个在整个 TestSpectral 类执行前自动调用的前置条件
    @pytest.fixture(scope='class', autouse=True)
    def check_freqs(self, vals, targfreqs, resfreqs, fstims):
        # 断言结果频率数组中的最小值索引为0
        assert resfreqs.argmin() == 0
        # 断言结果频率数组中的最大值索引为数组长度减1
        assert resfreqs.argmax() == len(resfreqs)-1
        # 使用 numpy.testing.assert_allclose 函数比较结果频率数组和目标频率数组，在给定的绝对容差范围内相等
        assert_allclose(resfreqs, targfreqs, atol=1e-06)
        # 遍历每个输入的刺激频率，进行以下断言
        for fstim in fstims:
            # 找到与当前刺激频率最接近的结果频率的索引
            i = np.abs(resfreqs - fstim).argmin()
            # 断言当前刺激频率对应的值大于其前两个索引位置的值
            assert vals[i] > vals[i+2]
            # 断言当前刺激频率对应的值大于其后两个索引位置的值
            assert vals[i] > vals[i-2]

    # 定义一个方法用于检查最大频率
    def check_maxfreq(self, spec, fsp, fstims):
        # 如果没有刺激频率数据，则跳过测试
        if len(fstims) == 0:
            return

        # 如果是双边频谱，针对每一边进行测试
        if fsp.min() < 0:
            # 取频谱的绝对值
            fspa = np.abs(fsp)
            # 找到最接近零点的索引
            zeroind = fspa.argmin()
            # 递归调用 check_maxfreq 方法，对每一边的频谱进行测试
            self.check_maxfreq(spec[:zeroind], fspa[:zeroind], fstims)
            self.check_maxfreq(spec[zeroind:], fspa[zeroind:], fstims)
            return

        # 复制刺激频率列表和频谱数组
        fstimst = fstims[:]
        spect = spec.copy()

        # 遍历每个峰值，并确保其是最大峰值
        while fstimst:
            # 找到频谱中的最大值索引
            maxind = spect.argmax()
            # 获取对应的频率值
            maxfreq = fsp[maxind]
            # 使用 numpy.testing.assert_almost_equal 函数比较最大频率值与当前刺激频率列表的最后一个频率值，在给定的精度下相等
            assert_almost_equal(maxfreq, fstimst[-1])
            # 删除已验证的刺激频率
            del fstimst[-1]
            # 将最大值索引周围的值置为0，以便下一次迭代找到下一个最大峰值
            spect[maxind-5:maxind+5] = 0
    def test_spectral_helper_raises(self):
        # 通过循环测试多种参数组合，验证 _spectral_helper 方法在不同错误条件下是否能正确抛出 ValueError 异常。
        # 这些条件包括：复杂模式下要求 x 等于 y、幅度模式、角度模式、相位模式、错误的模式、错误的 sides 参数、noverlap 大于 NFFT、noverlap 等于 NFFT、窗口长度不等于 NFFT。
        for kwargs in [  # Various error conditions:
            {"y": self.y+1, "mode": "complex"},  # Modes requiring ``x is y``.
            {"y": self.y+1, "mode": "magnitude"},
            {"y": self.y+1, "mode": "angle"},
            {"y": self.y+1, "mode": "phase"},
            {"mode": "spam"},  # Bad mode.
            {"y": self.y, "sides": "eggs"},  # Bad sides.
            {"y": self.y, "NFFT": 10, "noverlap": 20},  # noverlap > NFFT.
            {"NFFT": 10, "noverlap": 10},  # noverlap == NFFT.
            {"y": self.y, "NFFT": 10,
             "window": np.ones(9)},  # len(win) != NFFT.
        ]:
            with pytest.raises(ValueError):
                mlab._spectral_helper(x=self.y, **kwargs)

    @pytest.mark.parametrize('mode', ['default', 'psd'])
    def test_single_spectrum_helper_unsupported_modes(self, mode):
        # 测试 _single_spectrum_helper 方法在不支持的模式下是否能正确抛出 ValueError 异常。
        with pytest.raises(ValueError):
            mlab._single_spectrum_helper(x=self.y, mode=mode)

    @pytest.mark.parametrize("mode, case", [
        ("psd", "density"),
        ("magnitude", "specgram"),
        ("magnitude", "spectrum"),
    ])
    def test_spectral_helper_psd(self, mode, case):
        # 测试 _spectral_helper 方法在不同模式下的输出结果是否正确，通过对比频率和时间轴的精度来验证。
        freqs = getattr(self, f"freqs_{case}")
        spec, fsp, t = mlab._spectral_helper(
            x=self.y, y=self.y,
            NFFT=getattr(self, f"NFFT_{case}"),
            Fs=self.Fs,
            noverlap=getattr(self, f"nover_{case}"),
            pad_to=getattr(self, f"pad_to_{case}"),
            sides=self.sides,
            mode=mode)

        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, getattr(self, f"t_{case}"), atol=1e-06)
        assert spec.shape[0] == freqs.shape[0]
        assert spec.shape[1] == getattr(self, f"t_{case}").shape[0]

    def test_csd(self):
        # 测试 mlab.csd 方法的输出结果是否与预期的频率数组相匹配。
        freqs = self.freqs_density
        spec, fsp = mlab.csd(x=self.y, y=self.y+1,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=self.nover_density,
                             pad_to=self.pad_to_density,
                             sides=self.sides)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert spec.shape == freqs.shape

    def test_csd_padding(self):
        """Test zero padding of csd()."""
        # 测试 mlab.csd 方法在进行零填充时的结果，用于验证不同 NFFT 下的功率谱密度积分是否正确。
        if self.NFFT_density is None:  # for derived classes
            return
        sargs = dict(x=self.y, y=self.y+1, Fs=self.Fs, window=mlab.window_none,
                     sides=self.sides)

        spec0, _ = mlab.csd(NFFT=self.NFFT_density, **sargs)
        spec1, _ = mlab.csd(NFFT=self.NFFT_density*2, **sargs)
        assert_almost_equal(np.sum(np.conjugate(spec0)*spec0).real,
                            np.sum(np.conjugate(spec1/2)*spec1/2).real)
    # 定义一个测试方法，用于测试功率谱密度的计算
    def test_psd(self):
        # 从测试类中获取频率密度数据
        freqs = self.freqs_density
        # 调用mlab.psd函数计算信号的功率谱密度
        spec, fsp = mlab.psd(x=self.y,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=self.nover_density,
                             pad_to=self.pad_to_density,
                             sides=self.sides)
        # 断言计算得到的功率谱密度的形状与预期的频率密度数据形状相同
        assert spec.shape == freqs.shape
        # 调用辅助方法验证频率数据的正确性
        self.check_freqs(spec, freqs, fsp, self.fstims)

    # 使用pytest的参数化标记定义多组参数化测试用例，用于测试不同的数据生成方法和去趋势方法
    @pytest.mark.parametrize(
        'make_data, detrend',
        [(np.zeros, mlab.detrend_mean), (np.zeros, 'mean'),
         (np.arange, mlab.detrend_linear), (np.arange, 'linear')])
    # 定义一个测试方法，用于测试功率谱密度计算时的去趋势效果
    def test_psd_detrend(self, make_data, detrend):
        # 如果NFFT_density参数为None，直接返回，不进行测试
        if self.NFFT_density is None:
            return
        # 生成指定长度的测试数据
        ydata = make_data(self.NFFT_density)
        ydata1 = ydata + 5
        ydata2 = ydata + 3.3
        # 创建一个2行的数据矩阵，每行包含ydata1和ydata2
        ydata = np.vstack([ydata1, ydata2])
        # 将数据矩阵重复20次
        ydata = np.tile(ydata, (20, 1))
        # 将矩阵转置并展平成一维数组
        ydatab = ydata.T.flatten()
        ydata = ydata.flatten()
        # 创建一个与ydata相同长度的全零数组作为对照数据
        ycontrol = np.zeros_like(ydata)
        # 调用mlab.psd函数计算信号的功率谱密度，不使用去趋势
        spec_g, fsp_g = mlab.psd(x=ydata,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=detrend)
        # 调用mlab.psd函数计算信号的功率谱密度，不使用去趋势
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=detrend)
        # 调用mlab.psd函数计算信号的功率谱密度，不使用去趋势
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides)
        # 断言不同条件下的频谱频率轴数据一致
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        # 断言不同条件下的功率谱密度数据近似相等，允许误差为1e-08
        assert_allclose(spec_g, spec_c, atol=1e-08)
        # 使用pytest断言，验证应该不会近乎相等的条件
        with pytest.raises(AssertionError):
            assert_allclose(spec_b, spec_c, atol=1e-08)
    # 定义一个测试方法，用于测试 PSD（Power Spectral Density）计算中的汉宁窗口效果
    def test_psd_window_hanning(self):
        # 如果未设置 NFFT_density，则直接返回，不执行测试
        if self.NFFT_density is None:
            return
        
        # 生成长度为 NFFT_density 的等差数列作为测试数据
        ydata = np.arange(self.NFFT_density)
        ydata1 = ydata + 5
        ydata2 = ydata + 3.3
        
        # 使用 mlab.window_hanning 函数生成汉宁窗口的系数
        windowVals = mlab.window_hanning(np.ones_like(ydata1))
        
        # 分别将 ydata1 和 ydata2 乘以汉宁窗口系数，作为控制数据
        ycontrol1 = ydata1 * windowVals
        ycontrol2 = mlab.window_hanning(ydata2)
        
        # 将 ydata1 和 ydata2 组合成二维数组
        ydata = np.vstack([ydata1, ydata2])
        
        # 将 ycontrol1 和 ycontrol2 组合成二维数组作为控制数据
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        
        # 将 ydata 和 ycontrol 扩展为 20 行，保持原始形状不变
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        
        # 将 ydata 转置并展平为一维数组
        ydatab = ydata.T.flatten()
        ydataf = ydata.flatten()
        ycontrol = ycontrol.flatten()
        
        # 计算 ydataf 的功率谱密度和频谱
        spec_g, fsp_g = mlab.psd(x=ydataf,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=mlab.window_hanning)
        
        # 计算 ydatab 的功率谱密度和频谱
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=mlab.window_hanning)
        
        # 计算 ycontrol 的功率谱密度和频谱，使用无窗口函数
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=mlab.window_none)
        
        # 根据汉宁窗口的效果对 spec_c 进行缩放修正
        spec_c *= len(ycontrol1) / (windowVals ** 2).sum()
        
        # 断言函数返回的频谱结果 fsp_g 和 fsp_c 应该相等
        assert_array_equal(fsp_g, fsp_c)
        
        # 断言函数返回的频谱结果 fsp_b 和 fsp_c 应该相等
        assert_array_equal(fsp_b, fsp_c)
        
        # 使用绝对容差 1e-08 断言 spec_g 和 spec_c 应该非常接近
        assert_allclose(spec_g, spec_c, atol=1e-08)
        
        # 断言 spec_b 和 spec_c 不应该非常接近，引发断言错误异常
        # （即 spec_b 和 spec_c 不应该通过绝对容差 1e-08 判定为接近）
        with pytest.raises(AssertionError):
            assert_allclose(spec_b, spec_c, atol=1e-08)
    # 定义一个测试函数，用于测试 PSD 窗口处理函数的线性去趋势效果
    def test_psd_window_hanning_detrend_linear(self):
        # 如果 NFFT_density 为 None，则直接返回，不进行测试
        if self.NFFT_density is None:
            return
        # 生成一个长度为 NFFT_density 的等差数列作为测试数据
        ydata = np.arange(self.NFFT_density)
        # 创建一个与 ydata 同样长度的全零数组作为控制数据
        ycontrol = np.zeros(self.NFFT_density)
        # 根据 ydata 生成两个稍微偏移的测试数据
        ydata1 = ydata+5
        ydata2 = ydata+3.3
        # 复制 ycontrol 作为两个控制数据
        ycontrol1 = ycontrol
        ycontrol2 = ycontrol
        # 使用 mlab.window_hanning 函数生成一个 Hanning 窗口并应用于 ycontrol1
        windowVals = mlab.window_hanning(np.ones_like(ycontrol1))
        ycontrol1 = ycontrol1 * windowVals
        # 将 Hanning 窗口直接应用于 ycontrol2
        ycontrol2 = mlab.window_hanning(ycontrol2)
        # 将两组数据堆叠成二维数组，重复 20 次
        ydata = np.vstack([ydata1, ydata2])
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        # 将二维数组展平成一维数组
        ydatab = ydata.T.flatten()
        ydataf = ydata.flatten()
        ycontrol = ycontrol.flatten()
        # 计算 ydataf 的 PSD（功率谱密度）和频谱
        spec_g, fsp_g = mlab.psd(x=ydataf,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=mlab.detrend_linear,
                                 window=mlab.window_hanning)
        # 计算 ydatab 的 PSD 和频谱
        spec_b, fsp_b = mlab.psd(x=ydatab,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 detrend=mlab.detrend_linear,
                                 window=mlab.window_hanning)
        # 计算 ycontrol 的 PSD 和频谱，使用无窗函数 mlab.window_none
        spec_c, fsp_c = mlab.psd(x=ycontrol,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=mlab.window_none)
        # 根据窗口的归一化因子调整 spec_c 的值
        spec_c *= len(ycontrol1)/(windowVals**2).sum()
        # 断言 fsp_g 与 fsp_c 相等
        assert_array_equal(fsp_g, fsp_c)
        # 断言 fsp_b 与 fsp_c 相等
        assert_array_equal(fsp_b, fsp_c)
        # 断言 spec_g 与 spec_c 在一定误差范围内相近
        assert_allclose(spec_g, spec_c, atol=1e-08)
        # 以下两组数据 spec_b 和 spec_c 不应该非常接近，否则抛出 AssertionError
        with pytest.raises(AssertionError):
            assert_allclose(spec_b, spec_c, atol=1e-08)
    def test_psd_window_flattop(self):
        # flattop window
        # 从 https://github.com/scipy/scipy/blob/v1.10.0/scipy/signal/windows/_windows.py#L562-L622 改编
        # 定义 flattop 窗口系数
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        # 生成频率轴
        fac = np.linspace(-np.pi, np.pi, self.NFFT_density_real)
        # 初始化窗口数组
        win = np.zeros(self.NFFT_density_real)
        # 计算 flattop 窗口
        for k in range(len(a)):
            win += a[k] * np.cos(k * fac)

        # 计算信号的功率谱密度
        spec, fsp = mlab.psd(x=self.y,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=0,
                             sides=self.sides,
                             window=win,
                             scale_by_freq=False)
        # 使用同一窗口计算备选功率谱密度
        spec_a, fsp_a = mlab.psd(x=self.y,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=0,
                                 sides=self.sides,
                                 window=win)
        # 断言两个功率谱的关系
        assert_allclose(spec*win.sum()**2,
                        spec_a*self.Fs*(win**2).sum(),
                        atol=1e-08)

    def test_psd_windowarray(self):
        # 使用自定义窗口数组计算功率谱密度
        freqs = self.freqs_density
        spec, fsp = mlab.psd(x=self.y,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=self.nover_density,
                             pad_to=self.pad_to_density,
                             sides=self.sides,
                             window=np.ones(self.NFFT_density_real))
        # 断言频率轴的正确性
        assert_allclose(fsp, freqs, atol=1e-06)
        # 断言功率谱的形状符合预期
        assert spec.shape == freqs.shape
    # 定义一个测试方法，测试在给定参数下的 PSD（功率谱密度）计算结果是否正确
    def test_psd_windowarray_scale_by_freq(self):
        # 使用 Hanning 窗口函数生成长度为 self.NFFT_density_real 的窗口
        win = mlab.window_hanning(np.ones(self.NFFT_density_real))

        # 计算信号 x 的 PSD 和频谱
        spec, fsp = mlab.psd(x=self.y,
                             NFFT=self.NFFT_density,
                             Fs=self.Fs,
                             noverlap=self.nover_density,
                             pad_to=self.pad_to_density,
                             sides=self.sides,
                             window=mlab.window_hanning)
        
        # 计算使用频率缩放的 PSD 和频谱
        spec_s, fsp_s = mlab.psd(x=self.y,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=self.nover_density,
                                 pad_to=self.pad_to_density,
                                 sides=self.sides,
                                 window=mlab.window_hanning,
                                 scale_by_freq=True)
        
        # 计算不使用频率缩放的 PSD 和频谱
        spec_n, fsp_n = mlab.psd(x=self.y,
                                 NFFT=self.NFFT_density,
                                 Fs=self.Fs,
                                 noverlap=self.nover_density,
                                 pad_to=self.pad_to_density,
                                 sides=self.sides,
                                 window=mlab.window_hanning,
                                 scale_by_freq=False)
        
        # 断言频谱 fsp_s 和 fsp 相等
        assert_array_equal(fsp, fsp_s)
        # 断言频谱 fsp_n 和 fsp 相等
        assert_array_equal(fsp, fsp_n)
        # 断言 PSD spec_s 和 spec 相等
        assert_array_equal(spec, spec_s)
        
        # 使用 allclose 断言 spec_s 与预期值的相对误差在给定的容差内
        assert_allclose(spec_s*(win**2).sum(),
                        spec_n/self.Fs*win.sum()**2,
                        atol=1e-08)

    # 使用 pytest 的参数化装饰器，测试 mlabs 模块中不同谱类型的计算
    @pytest.mark.parametrize(
        "kind", ["complex", "magnitude", "angle", "phase"])
    def test_spectrum(self, kind):
        # 获取频率数组
        freqs = self.freqs_spectrum
        
        # 根据 kind 参数调用对应的频谱计算函数
        spec, fsp = getattr(mlab, f"{kind}_spectrum")(
            x=self.y,
            Fs=self.Fs, sides=self.sides, pad_to=self.pad_to_spectrum)
        
        # 使用 allclose 断言计算出的频谱 fsp 与预期频率数组 freqs 相似
        assert_allclose(fsp, freqs, atol=1e-06)
        
        # 断言 spec 的形状与 freqs 相同
        assert spec.shape == freqs.shape
        
        # 如果 kind 是 "magnitude"，则进一步检查最大频率和频率范围
        if kind == "magnitude":
            self.check_maxfreq(spec, fsp, self.fstims)
            self.check_freqs(spec, freqs, fsp, self.fstims)

    # 使用 pytest 的参数化装饰器，测试 mlabs 模块中不同模式下的频谱计算
    @pytest.mark.parametrize(
        'kwargs',
        [{}, {'mode': 'default'}, {'mode': 'psd'}, {'mode': 'magnitude'},
         {'mode': 'complex'}, {'mode': 'angle'}, {'mode': 'phase'}])
    # 测试特定的频谱图功能
    def test_specgram(self, kwargs):
        # 从对象属性中获取频率信息
        freqs = self.freqs_specgram
        # 使用给定参数计算频谱图
        spec, fsp, t = mlab.specgram(x=self.y,
                                     NFFT=self.NFFT_specgram,
                                     Fs=self.Fs,
                                     noverlap=self.nover_specgram,
                                     pad_to=self.pad_to_specgram,
                                     sides=self.sides,
                                     **kwargs)
        # 如果参数中包含'mode'，且其值为'complex'，则取频谱图的绝对值
        if kwargs.get('mode') == 'complex':
            spec = np.abs(spec)
        # 计算频谱图在频率轴上的均值
        specm = np.mean(spec, axis=1)

        # 断言频谱轴上的频率和预期频率非常接近
        assert_allclose(fsp, freqs, atol=1e-06)
        # 断言时间轴和预期时间轴非常接近
        assert_allclose(t, self.t_specgram, atol=1e-06)

        # 断言频谱图的形状与预期一致
        assert spec.shape[0] == freqs.shape[0]
        assert spec.shape[1] == self.t_specgram.shape[0]

        # 如果'mode'不在['complex', 'angle', 'phase']中，则进行进一步的断言
        if kwargs.get('mode') not in ['complex', 'angle', 'phase']:
            # 如果频谱图的最大值非零，断言在时间轴上的最大变化与最大值的比率非常接近零
            if np.abs(spec.max()) != 0:
                assert_allclose(
                    np.diff(spec, axis=1).max() / np.abs(spec.max()), 0,
                    atol=1e-02)
        
        # 如果'mode'不在['angle', 'phase']中，则调用self.check_freqs方法验证频率信息
        if kwargs.get('mode') not in ['angle', 'phase']:
            self.check_freqs(specm, freqs, fsp, self.fstims)

    # 测试只计算一个时间片段时是否会发出警告
    def test_specgram_warn_only1seg(self):
        """Warning should be raised if len(x) <= NFFT."""
        # 使用pytest验证是否会发出UserWarning，且警告消息包含"Only one segment is calculated"
        with pytest.warns(UserWarning, match="Only one segment is calculated"):
            mlab.specgram(x=self.y, NFFT=len(self.y), Fs=self.Fs)

    # 测试PSD和CSD是否相等
    def test_psd_csd_equal(self):
        # 计算信号的功率谱密度和交叉功率谱密度
        Pxx, freqsxx = mlab.psd(x=self.y,
                                NFFT=self.NFFT_density,
                                Fs=self.Fs,
                                noverlap=self.nover_density,
                                pad_to=self.pad_to_density,
                                sides=self.sides)
        Pxy, freqsxy = mlab.csd(x=self.y, y=self.y,
                                NFFT=self.NFFT_density,
                                Fs=self.Fs,
                                noverlap=self.nover_density,
                                pad_to=self.pad_to_density,
                                sides=self.sides)
        # 断言两个功率谱密度数组几乎相等
        assert_array_almost_equal_nulp(Pxx, Pxy)
        # 断言两个频率数组完全相等
        assert_array_equal(freqsxx, freqsxy)

    # 使用pytest的参数化测试，参数为'mode'，值为["default", "psd"]
    @pytest.mark.parametrize("mode", ["default", "psd"])
    def test_specgram_auto_default_psd_equal(self, mode):
        """
        Test that mlab.specgram without mode and with mode 'default' and 'psd'
        are all the same.
        """
        # 调用 mlab.specgram 函数，使用默认参数计算谱图和频率和时间
        speca, freqspeca, ta = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides)
        # 再次调用 mlab.specgram 函数，使用 'default' 或 'psd' 模式计算谱图和频率和时间
        specb, freqspecb, tb = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode=mode)
        # 断言两次调用返回的谱图数据、频率、时间相等
        assert_array_equal(speca, specb)
        assert_array_equal(freqspeca, freqspecb)
        assert_array_equal(ta, tb)

    @pytest.mark.parametrize(
        "mode, conv", [
            ("magnitude", np.abs),
            ("angle", np.angle),
            ("phase", lambda x: np.unwrap(np.angle(x), axis=0))
        ])
    def test_specgram_complex_equivalent(self, mode, conv):
        # 调用 mlab.specgram 函数，使用 'complex' 模式计算复数谱图及其频率和时间
        specc, freqspecc, tc = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode='complex')
        # 再次调用 mlab.specgram 函数，使用指定的 mode 计算谱图及其频率和时间
        specm, freqspecm, tm = mlab.specgram(x=self.y,
                                             NFFT=self.NFFT_specgram,
                                             Fs=self.Fs,
                                             noverlap=self.nover_specgram,
                                             pad_to=self.pad_to_specgram,
                                             sides=self.sides,
                                             mode=mode)

        # 断言两次调用返回的频率、时间数据相等
        assert_array_equal(freqspecc, freqspecm)
        assert_array_equal(tc, tm)
        # 断言使用指定转换函数后的复数谱图与指定模式下的谱图数据在允许误差范围内相等
        assert_allclose(conv(specc), specm, atol=1e-06)
    # 定义测试方法，用于测试 PSD（功率谱密度）计算函数的窗口加权数组相等性
    def test_psd_windowarray_equal(self):
        # 使用 Hanning 窗口函数生成与数据数组长度相同的窗口数组
        win = mlab.window_hanning(np.ones(self.NFFT_density_real))
        # 计算第一个信号的 PSD（功率谱密度）及其频谱
        speca, fspa = mlab.psd(x=self.y,
                               NFFT=self.NFFT_density,
                               Fs=self.Fs,
                               noverlap=self.nover_density,
                               pad_to=self.pad_to_density,
                               sides=self.sides,
                               window=win)
        # 计算第二个信号的 PSD（功率谱密度）及其频谱，使用相同参数但未指定窗口
        specb, fspb = mlab.psd(x=self.y,
                               NFFT=self.NFFT_density,
                               Fs=self.Fs,
                               noverlap=self.nover_density,
                               pad_to=self.pad_to_density,
                               sides=self.sides)
        # 断言频谱数组 fspa 和 fspb 相等
        assert_array_equal(fspa, fspb)
        # 断言 PSD 结果数组 speca 和 specb 在指定的精度范围内近似相等
        assert_allclose(speca, specb, atol=1e-08)
# extra test for cohere...
# 定义一个额外的测试函数，用于测试cohere函数

def test_cohere():
    # 设置数据点数量为1024，并设置随机种子
    N = 1024
    np.random.seed(19680801)
    # 生成服从标准正态分布的随机数序列x
    x = np.random.randn(N)
    # 将序列x向右移动20个位置，模拟相位偏移
    y = np.roll(x, 20)
    # 对序列y进行高频率滤波（滚动平均）
    y = np.convolve(y, np.ones(20) / 20., mode='same')
    # 计算信号的相干谱密度cohsq和频率f
    cohsq, f = mlab.cohere(x, y, NFFT=256, Fs=2, noverlap=128)
    # 断言：相干谱密度cohsq的均值接近于0.837，允许的误差为1.e-3
    assert_allclose(np.mean(cohsq), 0.837, atol=1.e-3)
    # 断言：相干谱密度cohsq的均值是实数
    assert np.isreal(np.mean(cohsq))


# *****************************************************************
# These Tests were taken from SCIPY with some minor modifications
# this can be retrieved from:
# https://github.com/scipy/scipy/blob/master/scipy/stats/tests/test_kdeoth.py
# *****************************************************************

class TestGaussianKDE:

    def test_kde_integer_input(self):
        """Regression test for #1181."""
        # 创建一个长度为5的整数数组x1
        x1 = np.arange(5)
        # 使用x1初始化一个高斯核密度估计对象kde
        kde = mlab.GaussianKDE(x1)
        # 预期的输出结果y_expected
        y_expected = [0.13480721, 0.18222869, 0.19514935, 0.18222869,
                      0.13480721]
        # 断言：kde对x1的估计结果与y_expected数组几乎相等，精度为小数点后6位
        np.testing.assert_array_almost_equal(kde(x1), y_expected, decimal=6)

    def test_gaussian_kde_covariance_caching(self):
        # 创建一个包含浮点数的数组x1
        x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
        # 在[-10, 10]范围内均匀分布的5个点作为xs
        xs = np.linspace(-10, 10, num=5)
        # 期望的输出值y_expected，来源于Scipy 0.10版本之前的高斯核密度估计结果
        y_expected = [0.02463386, 0.04689208, 0.05395444, 0.05337754,
                      0.01664475]
        # 使用默认的带宽参数'scott'初始化一个高斯核密度估计对象kde2
        kde2 = mlab.GaussianKDE(x1, 'scott')
        # 计算kde2在xs上的密度估计值y2
        y2 = kde2(xs)
        # 断言：y2与y_expected几乎相等，精度为小数点后7位
        np.testing.assert_array_almost_equal(y_expected, y2, decimal=7)

    def test_kde_bandwidth_method(self):
        # 设定随机种子
        np.random.seed(8765678)
        # 生成包含50个随机数的数组xn
        n_basesample = 50
        xn = np.random.randn(n_basesample)

        # 默认带宽方法
        gkde = mlab.GaussianKDE(xn)
        # 使用'scott'带宽方法初始化一个高斯核密度估计对象gkde2
        gkde2 = mlab.GaussianKDE(xn, 'scott')
        # 使用gkde的带宽参数初始化一个高斯核密度估计对象gkde3
        gkde3 = mlab.GaussianKDE(xn, bw_method=gkde.factor)

        # 在[-7, 7]范围内均匀分布的51个点作为xs
        xs = np.linspace(-7, 7, 51)
        # 计算gkde在xs上的概率密度估计值kdepdf
        kdepdf = gkde.evaluate(xs)
        # 计算gkde2在xs上的概率密度估计值kdepdf2
        kdepdf2 = gkde2.evaluate(xs)
        # 断言：kdepdf和kdepdf2完全相等
        assert kdepdf.all() == kdepdf2.all()
        # 计算gkde3在xs上的概率密度估计值kdepdf3
        kdepdf3 = gkde3.evaluate(xs)
        # 断言：kdepdf和kdepdf3完全相等
        assert kdepdf.all() == kdepdf3.all()


class TestGaussianKDECustom:
    def test_no_data(self):
        """Pass no data into the GaussianKDE class."""
        # 测试：不传入数据时，初始化高斯核密度估计对象应引发ValueError异常
        with pytest.raises(ValueError):
            mlab.GaussianKDE([])

    def test_single_dataset_element(self):
        """Pass a single dataset element into the GaussianKDE class."""
        # 测试：传入单个数据元素时，初始化高斯核密度估计对象应引发ValueError异常
        with pytest.raises(ValueError):
            mlab.GaussianKDE([42])

    def test_silverman_multidim_dataset(self):
        """Test silverman's for a multi-dimensional array."""
        # 创建一个二维数组x1
        x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # 测试：传入多维数组时，使用'silverman'带宽方法应引发np.linalg.LinAlgError异常
        with pytest.raises(np.linalg.LinAlgError):
            mlab.GaussianKDE(x1, "silverman")
    # 测试 silverman 方法对单维度列表的输出
    def test_silverman_singledim_dataset(self):
        """Test silverman's output for a single dimension list."""
        # 创建包含数据的 NumPy 数组
        x1 = np.array([-7, -5, 1, 4, 5])
        # 使用 silverman 方法创建 GaussianKDE 对象
        mygauss = mlab.GaussianKDE(x1, "silverman")
        # 预期的输出值
        y_expected = 0.76770389927475502
        # 断言计算的协方差因子近似等于预期值
        assert_almost_equal(mygauss.covariance_factor(), y_expected, 7)

    # 测试 scott 方法对多维度数组的输出
    def test_scott_multidim_dataset(self):
        """Test scott's output for a multi-dimensional array."""
        # 创建包含数据的 NumPy 二维数组
        x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # 用 pytest 检查是否会引发 np.linalg.LinAlgError 异常
        with pytest.raises(np.linalg.LinAlgError):
            mlab.GaussianKDE(x1, "scott")

    # 测试 scott 方法对单维度数组的输出
    def test_scott_singledim_dataset(self):
        """Test scott's output a single-dimensional array."""
        # 创建包含数据的 NumPy 数组
        x1 = np.array([-7, -5, 1, 4, 5])
        # 使用 scott 方法创建 GaussianKDE 对象
        mygauss = mlab.GaussianKDE(x1, "scott")
        # 预期的输出值
        y_expected = 0.72477966367769553
        # 断言计算的协方差因子近似等于预期值
        assert_almost_equal(mygauss.covariance_factor(), y_expected, 7)

    # 测试空数组的情况下 scalar 方法的协方差因子
    def test_scalar_empty_dataset(self):
        """Test the scalar's cov factor for an empty array."""
        # 使用 pytest 检查是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            mlab.GaussianKDE([], bw_method=5)

    # 测试 scalar 方法的协方差因子
    def test_scalar_covariance_dataset(self):
        """Test a scalar's cov factor."""
        # 设定随机种子以便重现结果
        np.random.seed(8765678)
        n_basesample = 50
        # 创建多维度数据列表
        multidim_data = [np.random.randn(n_basesample) for i in range(5)]
        # 使用 scalar 方法创建 GaussianKDE 对象
        kde = mlab.GaussianKDE(multidim_data, bw_method=0.5)
        # 断言计算的协方差因子等于指定的值
        assert kde.covariance_factor() == 0.5

    # 测试 callable 方法的协方差因子，对多维度数组
    def test_callable_covariance_dataset(self):
        """Test the callable's cov factor for a multi-dimensional array."""
        # 设定随机种子以便重现结果
        np.random.seed(8765678)
        n_basesample = 50
        # 创建多维度数据列表
        multidim_data = [np.random.randn(n_basesample) for i in range(5)]

        # 定义一个返回固定值的可调用函数
        def callable_fun(x):
            return 0.55
        # 使用 callable 方法创建 GaussianKDE 对象
        kde = mlab.GaussianKDE(multidim_data, bw_method=callable_fun)
        # 断言计算的协方差因子等于指定的值
        assert kde.covariance_factor() == 0.55

    # 测试 callable 方法的协方差因子，对单维度数组
    def test_callable_singledim_dataset(self):
        """Test the callable's cov factor for a single-dimensional array."""
        # 设定随机种子以便重现结果
        np.random.seed(8765678)
        n_basesample = 50
        # 创建单维度数据数组
        multidim_data = np.random.randn(n_basesample)
        # 使用 silverman 方法创建 GaussianKDE 对象
        kde = mlab.GaussianKDE(multidim_data, bw_method='silverman')
        # 预期的输出值
        y_expected = 0.48438841363348911
        # 断言计算的协方差因子近似等于预期值
        assert_almost_equal(kde.covariance_factor(), y_expected, 7)

    # 测试当 bw_method 参数无效时是否会抛出 ValueError 异常
    def test_wrong_bw_method(self):
        """Test the error message that should be called when bw is invalid."""
        # 设定随机种子以便重现结果
        np.random.seed(8765678)
        n_basesample = 50
        # 创建随机数据数组
        data = np.random.randn(n_basesample)
        # 使用 pytest 检查是否会引发 ValueError 异常
        with pytest.raises(ValueError):
            mlab.GaussianKDE(data, bw_method="invalid")
class TestGaussianKDEEvaluate:

    def test_evaluate_diff_dim(self):
        """
        Test the evaluate method when the dim's of dataset and points have
        different dimensions.
        """
        # 创建一个包含 [3, 5, 7, 9] 的 NumPy 数组
        x1 = np.arange(3, 10, 2)
        # 创建一个 GaussianKDE 对象，以 x1 作为数据集
        kde = mlab.GaussianKDE(x1)
        # 创建一个包含 [3, 5, 7, 9, 11] 的 NumPy 数组
        x2 = np.arange(3, 12, 2)
        # 预期的 y 值列表
        y_expected = [
            0.08797252, 0.11774109, 0.11774109, 0.08797252, 0.0370153
        ]
        # 使用 GaussianKDE 对象计算 x2 对应的密度估计
        y = kde.evaluate(x2)
        # 使用 NumPy 断言近似相等，精度为 7 位小数
        np.testing.assert_array_almost_equal(y, y_expected, 7)

    def test_evaluate_inv_dim(self):
        """
        Invert the dimensions; i.e., for a dataset of dimension 1 [3, 2, 4],
        the points should have a dimension of 3 [[3], [2], [4]].
        """
        # 设定随机数种子，确保结果可重现
        np.random.seed(8765678)
        # 设定基本样本数量
        n_basesample = 50
        # 生成服从标准正态分布的随机数据
        multidim_data = np.random.randn(n_basesample)
        # 创建一个 GaussianKDE 对象，以 multidim_data 作为数据集
        kde = mlab.GaussianKDE(multidim_data)
        # 创建一个包含 [[1], [2], [3]] 的 NumPy 数组
        x2 = [[1], [2], [3]]
        # 使用 pytest 断言应该抛出 ValueError 异常
        with pytest.raises(ValueError):
            kde.evaluate(x2)

    def test_evaluate_dim_and_num(self):
        """Tests if evaluated against a one by one array"""
        # 创建一个包含 [3, 5, 7, 9] 的 NumPy 数组
        x1 = np.arange(3, 10, 2)
        # 创建一个包含 [3] 的 NumPy 数组
        x2 = np.array([3])
        # 创建一个 GaussianKDE 对象，以 x1 作为数据集
        kde = mlab.GaussianKDE(x1)
        # 预期的 y 值列表
        y_expected = [0.08797252]
        # 使用 GaussianKDE 对象计算 x2 对应的密度估计
        y = kde.evaluate(x2)
        # 使用 NumPy 断言近似相等，精度为 7 位小数
        np.testing.assert_array_almost_equal(y, y_expected, 7)

    def test_evaluate_point_dim_not_one(self):
        # 创建一个包含 [3, 5, 7, 9] 的 NumPy 数组
        x1 = np.arange(3, 10, 2)
        # 创建一个包含两个相同的 [array([3, 5, 7, 9]), array([3, 5, 7, 9])] 的列表
        x2 = [np.arange(3, 10, 2), np.arange(3, 10, 2)]
        # 创建一个 GaussianKDE 对象，以 x1 作为数据集
        kde = mlab.GaussianKDE(x1)
        # 使用 pytest 断言应该抛出 ValueError 异常
        with pytest.raises(ValueError):
            kde.evaluate(x2)

    def test_evaluate_equal_dim_and_num_lt(self):
        # 创建一个包含 [3, 5, 7, 9] 的 NumPy 数组
        x1 = np.arange(3, 10, 2)
        # 创建一个包含 [3, 5, 7] 的 NumPy 数组
        x2 = np.arange(3, 8, 2)
        # 创建一个 GaussianKDE 对象，以 x1 作为数据集
        kde = mlab.GaussianKDE(x1)
        # 预期的 y 值列表
        y_expected = [0.08797252, 0.11774109, 0.11774109]
        # 使用 GaussianKDE 对象计算 x2 对应的密度估计
        y = kde.evaluate(x2)
        # 使用 NumPy 断言近似相等，精度为 7 位小数
        np.testing.assert_array_almost_equal(y, y_expected, 7)


def test_psd_onesided_norm():
    # 创建一个包含 [0, 1, 2, 3, 1, 2, 1] 的 NumPy 数组
    u = np.array([0, 1, 2, 3, 1, 2, 1])
    # 设定时间步长
    dt = 1.0
    # 计算 u 的功率谱密度
    Su = np.abs(np.fft.fft(u) * dt)**2 / (dt * u.size)
    # 使用 mlab.psd 计算 u 的单侧功率谱密度
    P, f = mlab.psd(u, NFFT=u.size, Fs=1/dt, window=mlab.window_none,
                    detrend=mlab.detrend_none, noverlap=0, pad_to=None,
                    scale_by_freq=None,
                    sides='onesided')
    # 创建单侧功率谱密度的预期值数组
    Su_1side = np.append([Su[0]], Su[1:4] + Su[4:][::-1])
    # 使用 assert_allclose 断言 P 与 Su_1side 近似相等，容差为 1e-06
    assert_allclose(P, Su_1side, atol=1e-06)


def test_psd_oversampling():
    """Test the case len(x) < NFFT for psd()."""
    # 创建一个包含 [0, 1, 2, 3, 1, 2, 1] 的 NumPy 数组
    u = np.array([0, 1, 2, 3, 1, 2, 1])
    # 设定时间步长
    dt = 1.0
    # 计算 u 的功率谱密度
    Su = np.abs(np.fft.fft(u) * dt)**2 / (dt * u.size)
    # 使用 mlab.psd 计算 u 的单侧功率谱密度，NFFT 设置为 u.size*2
    P, f = mlab.psd(u, NFFT=u.size*2, Fs=1/dt, window=mlab.window_none,
                    detrend=mlab.detrend_none, noverlap=0, pad_to=None,
                    scale_by_freq=None,
                    sides='onesided')
    # 创建单侧功率谱密度的预期值数组
    Su_1side = np.append([Su[0]], Su[1:4] + Su[4:][::-1])
    # 使用 assert_almost_equal 断言 P 的总和与 Su_1side 的总和近似相等
    assert_almost_equal(np.sum(P), np.sum(Su_1side))  # same energy
```