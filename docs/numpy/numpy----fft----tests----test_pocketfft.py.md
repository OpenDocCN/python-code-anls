# `.\numpy\numpy\fft\tests\test_pocketfft.py`

```py
# 导入必要的库
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库用于单元测试
from numpy.random import random  # 导入NumPy的随机数生成函数random
from numpy.testing import (  # 导入NumPy测试模块中的断言函数
        assert_array_equal, assert_raises, assert_allclose, IS_WASM
        )
import threading  # 导入线程模块
import queue  # 导入队列模块


def fft1(x):
    # 计算输入数组的长度
    L = len(x)
    # 计算相位信息，用于FFT计算
    phase = -2j * np.pi * (np.arange(L) / L)
    # 创建相位矩阵，形状为(L, 1)
    phase = np.arange(L).reshape(-1, 1) * phase
    # 执行FFT计算并返回结果
    return np.sum(x*np.exp(phase), axis=1)


class TestFFTShift:

    def test_fft_n(self):
        # 测试在指定情况下是否会引发值错误
        assert_raises(ValueError, np.fft.fft, [1, 2, 3], 0)


class TestFFT1D:

    def test_identity(self):
        # 测试FFT和逆FFT的身份运算是否精确
        maxlen = 512
        # 生成随机复数数组
        x = random(maxlen) + 1j*random(maxlen)
        xr = random(maxlen)
        for i in range(1, maxlen):
            # 检查FFT和逆FFT的精确度
            assert_allclose(np.fft.ifft(np.fft.fft(x[0:i])), x[0:i],
                            atol=1e-12)
            assert_allclose(np.fft.irfft(np.fft.rfft(xr[0:i]), i),
                            xr[0:i], atol=1e-12)

    @pytest.mark.parametrize("dtype", [np.single, np.double, np.longdouble])
    def test_identity_long_short(self, dtype):
        # 使用不同数据类型进行测试，包括单精度、双精度和长双精度
        maxlen = 16
        # 计算数值精度容忍度
        atol = 4 * np.spacing(np.array(1., dtype=dtype))
        # 生成随机复数数组，指定数据类型
        x = random(maxlen).astype(dtype) + 1j*random(maxlen).astype(dtype)
        xx = np.concatenate([x, np.zeros_like(x)])
        xr = random(maxlen).astype(dtype)
        xxr = np.concatenate([xr, np.zeros_like(xr)])
        for i in range(1, maxlen*2):
            # 执行FFT和逆FFT，并检查结果精确度
            check_c = np.fft.ifft(np.fft.fft(x, n=i), n=i)
            assert check_c.real.dtype == dtype
            assert_allclose(check_c, xx[0:i], atol=atol, rtol=0)
            check_r = np.fft.irfft(np.fft.rfft(xr, n=i), n=i)
            assert check_r.dtype == dtype
            assert_allclose(check_r, xxr[0:i], atol=atol, rtol=0)

    @pytest.mark.parametrize("dtype", [np.single, np.double, np.longdouble])
    # 测试函数，用于验证 FFT 和 IFFT 的逆向操作
    def test_identity_long_short_reversed(self, dtype):
        # 也测试以相反顺序明确给出点数的情况。
        maxlen = 16
        # 定义绝对容差
        atol = 5 * np.spacing(np.array(1., dtype=dtype))
        # 创建复数数组 x，包含随机实部和虚部
        x = random(maxlen).astype(dtype) + 1j*random(maxlen).astype(dtype)
        # 将 x 与其翻转拼接为 xx
        xx = np.concatenate([x, np.zeros_like(x)])
        # 遍历不同点数的范围
        for i in range(1, maxlen*2):
            # 使用 FFT 和 IFFT 进行交叉验证
            check_via_c = np.fft.fft(np.fft.ifft(x, n=i), n=i)
            # 断言验证结果的数据类型与 x 的数据类型相同
            assert check_via_c.dtype == x.dtype
            # 断言验证 FFT 和 IFFT 操作后的结果近似相等
            assert_allclose(check_via_c, xx[0:i], atol=atol, rtol=0)
            # 对于 irfft，如果点数是偶数，则无法恢复第一个元素的虚部
            # 或者最后一个元素的虚部。因此在比较时将其设为 0。
            y = x.copy()
            n = i // 2 + 1
            y.imag[0] = 0
            if i % 2 == 0:
                y.imag[n-1:] = 0
            yy = np.concatenate([y, np.zeros_like(y)])
            # 使用 FFT 和 IFFT 进行反向交叉验证
            check_via_r = np.fft.rfft(np.fft.irfft(x, n=i), n=i)
            # 断言验证结果的数据类型与 x 的数据类型相同
            assert check_via_r.dtype == x.dtype
            # 断言验证 rfft 和 irfft 操作后的结果近似相等
            assert_allclose(check_via_r, yy[0:n], atol=atol, rtol=0)

    # 测试 FFT 函数的不同用法
    def test_fft(self):
        # 创建复数数组 x，包含随机实部和虚部
        x = random(30) + 1j*random(30)
        # 断言验证 fft1(x) 和 np.fft.fft(x) 的结果近似相等
        assert_allclose(fft1(x), np.fft.fft(x), atol=1e-6)
        # 断言验证 fft1(x) 和 np.fft.fft(x, norm="backward") 的结果近似相等
        assert_allclose(fft1(x), np.fft.fft(x, norm="backward"), atol=1e-6)
        # 断言验证 fft1(x) / np.sqrt(30) 和 np.fft.fft(x, norm="ortho") 的结果近似相等
        assert_allclose(fft1(x) / np.sqrt(30),
                        np.fft.fft(x, norm="ortho"), atol=1e-6)
        # 断言验证 fft1(x) / 30. 和 np.fft.fft(x, norm="forward") 的结果近似相等
        assert_allclose(fft1(x) / 30.,
                        np.fft.fft(x, norm="forward"), atol=1e-6)

    # 测试 FFT 函数的 out 参数
    @pytest.mark.parametrize("axis", (0, 1))
    @pytest.mark.parametrize("dtype", (complex, float))
    @pytest.mark.parametrize("transpose", (True, False))
    def test_fft_out_argument(self, dtype, transpose, axis):
        # 定义类似于 np.zeros_like 的函数
        def zeros_like(x):
            if transpose:
                return np.zeros_like(x.T).T
            else:
                return np.zeros_like(x)

        # 仅测试 out 参数
        if dtype is complex:
            y = random((10, 20)) + 1j*random((10, 20))
            fft, ifft = np.fft.fft, np.fft.ifft
        else:
            y = random((10, 20))
            fft, ifft = np.fft.rfft, np.fft.irfft

        # 期望的 FFT 结果
        expected = fft(y, axis=axis)
        # 创建与 expected 相同形状的零数组
        out = zeros_like(expected)
        # 使用 out 参数进行 FFT 操作
        result = fft(y, out=out, axis=axis)
        # 断言验证 result 和 out 是同一对象
        assert result is out
        # 断言验证 result 和 expected 的值近似相等
        assert_array_equal(result, expected)

        # 期望的 IFFT 结果
        expected2 = ifft(expected, axis=axis)
        # 根据数据类型创建与 expected2 相同形状的零数组
        out2 = out if dtype is complex else zeros_like(expected2)
        # 使用 out 参数进行 IFFT 操作
        result2 = ifft(out, out=out2, axis=axis)
        # 断言验证 result2 和 out2 是同一对象
        assert result2 is out2
        # 断言验证 result2 和 expected2 的值近似相等
        assert_array_equal(result2, expected2)
    # 测试原地 FFT（快速傅里叶变换）的各种组合情况
    def test_fft_inplace_out(self, axis):
        # 创建一个大小为 (20, 20) 的复数随机数组 y
        y = random((20, 20)) + 1j * random((20, 20))
        
        # 完全原地计算 FFT
        y1 = y.copy()
        expected1 = np.fft.fft(y1, axis=axis)  # 计算预期的 FFT
        result1 = np.fft.fft(y1, axis=axis, out=y1)  # 原地计算 FFT，并将结果存入 y1
        assert result1 is y1  # 确保原地操作成功
        assert_array_equal(result1, expected1)  # 确保结果与预期一致
        
        # 部分数组原地计算 FFT；其余部分不变
        y2 = y.copy()
        out2 = y2[:10] if axis == 0 else y2[:, :10]
        expected2 = np.fft.fft(y2, n=10, axis=axis)  # 计算预期的 FFT
        result2 = np.fft.fft(y2, n=10, axis=axis, out=out2)  # 原地计算 FFT，并将结果存入 out2
        assert result2 is out2  # 确保原地操作成功
        assert_array_equal(result2, expected2)  # 确保结果与预期一致
        if axis == 0:
            assert_array_equal(y2[10:], y[10:])  # 确保其余部分未被修改
        else:
            assert_array_equal(y2[:, 10:], y[:, 10:])  # 确保其余部分未被修改
        
        # 原地计算另一部分数组的 FFT
        y3 = y.copy()
        y3_sel = y3[5:] if axis == 0 else y3[:, 5:]
        out3 = y3[5:15] if axis == 0 else y3[:, 5:15]
        expected3 = np.fft.fft(y3_sel, n=10, axis=axis)  # 计算预期的 FFT
        result3 = np.fft.fft(y3_sel, n=10, axis=axis, out=out3)  # 原地计算 FFT，并将结果存入 out3
        assert result3 is out3  # 确保原地操作成功
        assert_array_equal(result3, expected3)  # 确保结果与预期一致
        if axis == 0:
            assert_array_equal(y3[:5], y[:5])  # 确保其余部分未被修改
            assert_array_equal(y3[15:], y[15:])  # 确保其余部分未被修改
        else:
            assert_array_equal(y3[:, :5], y[:, :5])  # 确保其余部分未被修改
            assert_array_equal(y3[:, 15:], y[:, 15:])  # 确保其余部分未被修改
        
        # 原地计算 FFT，其中 n > 输入的数组长度；其余部分不变
        y4 = y.copy()
        y4_sel = y4[:10] if axis == 0 else y4[:, :10]
        out4 = y4[:15] if axis == 0 else y4[:, :15]
        expected4 = np.fft.fft(y4_sel, n=15, axis=axis)  # 计算预期的 FFT
        result4 = np.fft.fft(y4_sel, n=15, axis=axis, out=out4)  # 原地计算 FFT，并将结果存入 out4
        assert result4 is out4  # 确保原地操作成功
        assert_array_equal(result4, expected4)  # 确保结果与预期一致
        if axis == 0:
            assert_array_equal(y4[15:], y[15:])  # 确保其余部分未被修改
        else:
            assert_array_equal(y4[:, 15:], y[:, 15:])  # 确保其余部分未被修改
        
        # 在转置操作中进行覆写
        y5 = y.copy()
        out5 = y5.T
        result5 = np.fft.fft(y5, axis=axis, out=out5)  # 原地计算 FFT，并将结果存入 out5
        assert result5 is out5  # 确保原地操作成功
        assert_array_equal(result5, expected1)  # 确保结果与预期一致
        
        # 反向步长操作
        y6 = y.copy()
        out6 = y6[::-1] if axis == 0 else y6[:, ::-1]
        result6 = np.fft.fft(y6, axis=axis, out=out6)  # 原地计算 FFT，并将结果存入 out6
        assert result6 is out6  # 确保原地操作成功
        assert_array_equal(result6, expected1)  # 确保结果与预期一致
    # 定义测试函数，测试逆快速傅里叶变换（IFFT）功能
    def test_ifft(self, norm):
        # 生成一个包含随机复数的长度为30的数组
        x = random(30) + 1j*random(30)
        # 断言：对 x 进行 FFT 后再进行 IFFT 应当近似于 x 自身，使用指定的归一化方式
        assert_allclose(
            x, np.fft.ifft(np.fft.fft(x, norm=norm), norm=norm),
            atol=1e-6)
        # 确保会得到预期的错误消息
        with pytest.raises(ValueError,
                           match='Invalid number of FFT data points'):
            # 对一个空列表进行 IFFT 操作，应当引发 ValueError 异常
            np.fft.ifft([], norm=norm)

    # 定义测试函数，测试二维快速傅里叶变换（FFT2）功能
    def test_fft2(self):
        # 生成一个大小为 (30, 20) 的二维随机复数数组
        x = random((30, 20)) + 1j*random((30, 20))
        # 断言：对 x 进行两次 FFT 操作（分别在不同的轴上），结果应当与直接使用 FFT2 函数结果近似
        assert_allclose(np.fft.fft(np.fft.fft(x, axis=1), axis=0),
                        np.fft.fft2(x), atol=1e-6)
        # 断言：使用 FFT2 函数的默认归一化方式应当与指定 "backward" 归一化方式的 FFT2 函数结果近似
        assert_allclose(np.fft.fft2(x),
                        np.fft.fft2(x, norm="backward"), atol=1e-6)
        # 断言：使用 "ortho" 归一化方式对 FFT2 的结果进行归一化应当近似于原结果除以 sqrt(30 * 20)
        assert_allclose(np.fft.fft2(x) / np.sqrt(30 * 20),
                        np.fft.fft2(x, norm="ortho"), atol=1e-6)
        # 断言：使用 "forward" 归一化方式对 FFT2 的结果进行归一化应当近似于原结果除以 (30. * 20.)
        assert_allclose(np.fft.fft2(x) / (30. * 20.),
                        np.fft.fft2(x, norm="forward"), atol=1e-6)

    # 定义测试函数，测试二维逆快速傅里叶变换（IFFT2）功能
    def test_ifft2(self):
        # 生成一个大小为 (30, 20) 的二维随机复数数组
        x = random((30, 20)) + 1j*random((30, 20))
        # 断言：对 x 进行两次 IFFT 操作（分别在不同的轴上），结果应当与直接使用 IFFT2 函数结果近似
        assert_allclose(np.fft.ifft(np.fft.ifft(x, axis=1), axis=0),
                        np.fft.ifft2(x), atol=1e-6)
        # 断言：使用 IFFT2 函数的默认归一化方式应当与指定 "backward" 归一化方式的 IFFT2 函数结果近似
        assert_allclose(np.fft.ifft2(x),
                        np.fft.ifft2(x, norm="backward"), atol=1e-6)
        # 断言：使用 "ortho" 归一化方式对 IFFT2 的结果进行归一化应当近似于原结果乘以 sqrt(30 * 20)
        assert_allclose(np.fft.ifft2(x) * np.sqrt(30 * 20),
                        np.fft.ifft2(x, norm="ortho"), atol=1e-6)
        # 断言：使用 "forward" 归一化方式对 IFFT2 的结果进行归一化应当近似于原结果乘以 (30. * 20.)
        assert_allclose(np.fft.ifft2(x) * (30. * 20.),
                        np.fft.ifft2(x, norm="forward"), atol=1e-6)

    # 定义测试函数，测试多维快速傅里叶变换（FFTN）功能
    def test_fftn(self):
        # 生成一个大小为 (30, 20, 10) 的三维随机复数数组
        x = random((30, 20, 10)) + 1j*random((30, 20, 10))
        # 断言：对 x 进行三次 FFT 操作（分别在不同的轴上），结果应当与直接使用 FFTN 函数结果近似
        assert_allclose(
            np.fft.fft(np.fft.fft(np.fft.fft(x, axis=2), axis=1), axis=0),
            np.fft.fftn(x), atol=1e-6)
        # 断言：使用 FFTN 函数的默认归一化方式应当与指定 "backward" 归一化方式的 FFTN 函数结果近似
        assert_allclose(np.fft.fftn(x),
                        np.fft.fftn(x, norm="backward"), atol=1e-6)
        # 断言：使用 "ortho" 归一化方式对 FFTN 的结果进行归一化应当近似于原结果除以 sqrt(30 * 20 * 10)
        assert_allclose(np.fft.fftn(x) / np.sqrt(30 * 20 * 10),
                        np.fft.fftn(x, norm="ortho"), atol=1e-6)
        # 断言：使用 "forward" 归一化方式对 FFTN 的结果进行归一化应当近似于原结果除以 (30. * 20. * 10.)
        assert_allclose(np.fft.fftn(x) / (30. * 20. * 10.),
                        np.fft.fftn(x, norm="forward"), atol=1e-6)

    # 定义测试函数，测试多维逆快速傅里叶变换（IFFTN）功能
    def test_ifftn(self):
        # 生成一个大小为 (30, 20, 10) 的三维随机复数数组
        x = random((30, 20, 10)) + 1j*random((30, 20, 10))
        # 断言：对 x 进行三次 IFFT 操作（分别在不同的轴上），结果应当与直接使用 IFFTN 函数结果近似
        assert_allclose(
            np.fft.ifft(np.fft.ifft(np.fft.ifft(x, axis=2), axis=1), axis=0),
            np.fft.ifftn(x), atol=1e-6)
        # 断言：使用 IFFTN 函数的默认归一化方式应当与指定 "backward" 归一化方式的 IFFTN 函数结果近似
        assert_allclose(np.fft.ifftn(x),
                        np.fft.ifftn(x, norm="backward"), atol=1e-6)
        # 断言：使用 "ortho" 归一化方式对 IFFTN 的结果进行归一化应当近似于原结果乘以 sqrt(30 * 20 * 10)
        assert_allclose(np.fft.ifftn(x) * np.sqrt(30 * 20 * 10),
                        np.fft.ifftn(x, norm="ortho"), atol=1e-6)
        # 断言：使用 "forward" 归一化方式对 IFFTN 的结果进行归一化应当近似于原结果乘以 (30. * 20. * 10.)
        assert_allclose(np.fft.ifftn(x) * (30. * 20. * 10.),
                        np.fft.ifftn(x, norm="forward"), atol=1e-6)
    # 定义测试函数 test_rfft，用于测试 np.fft.rfft 函数的不同参数组合
    def test_rfft(self):
        # 创建长度为 30 的随机数组 x
        x = random(30)
        # 遍历 n 的两种可能取值：x.size 和 2*x.size
        for n in [x.size, 2*x.size]:
            # 遍历 norm 的四种可能取值：None, 'backward', 'ortho', 'forward'
            for norm in [None, 'backward', 'ortho', 'forward']:
                # 断言 np.fft.fft(x, n=n, norm=norm) 的前半部分与 np.fft.rfft(x, n=n, norm=norm) 的结果相近
                assert_allclose(
                    np.fft.fft(x, n=n, norm=norm)[:(n//2 + 1)],
                    np.fft.rfft(x, n=n, norm=norm), atol=1e-6)
            # 断言 np.fft.rfft(x, n=n) 与 np.fft.rfft(x, n=n, norm="backward") 的结果相近
            assert_allclose(
                np.fft.rfft(x, n=n),
                np.fft.rfft(x, n=n, norm="backward"), atol=1e-6)
            # 断言 np.fft.rfft(x, n=n) / np.sqrt(n) 与 np.fft.rfft(x, n=n, norm="ortho") 的结果相近
            assert_allclose(
                np.fft.rfft(x, n=n) / np.sqrt(n),
                np.fft.rfft(x, n=n, norm="ortho"), atol=1e-6)
            # 断言 np.fft.rfft(x, n=n) / n 与 np.fft.rfft(x, n=n, norm="forward") 的结果相近
            assert_allclose(
                np.fft.rfft(x, n=n) / n,
                np.fft.rfft(x, n=n, norm="forward"), atol=1e-6)

    # 定义测试函数 test_rfft_even，用于测试 np.fft.rfft 对于偶数长度输入的行为
    def test_rfft_even(self):
        # 创建长度为 8 的 numpy 数组 x
        x = np.arange(8)
        n = 4
        # 计算 np.fft.rfft(x, n) 的结果，并与 np.fft.fft(x[:n])[:n//2 + 1] 的结果进行比较
        y = np.fft.rfft(x, n)
        assert_allclose(y, np.fft.fft(x[:n])[:n//2 + 1], rtol=1e-14)

    # 定义测试函数 test_rfft_odd，用于测试 np.fft.rfft 对于奇数长度输入的行为
    def test_rfft_odd(self):
        # 创建长度为 5 的 numpy 数组 x
        x = np.array([1, 0, 2, 3, -3])
        # 计算 np.fft.rfft(x) 的结果，并与 np.fft.fft(x)[:3] 的结果进行比较
        y = np.fft.rfft(x)
        assert_allclose(y, np.fft.fft(x)[:3], rtol=1e-14)

    # 定义测试函数 test_irfft，用于测试 np.fft.irfft 函数的不同参数组合
    def test_irfft(self):
        # 创建长度为 30 的随机数组 x
        x = random(30)
        # 断言 np.fft.irfft(np.fft.rfft(x)) 与 x 的结果相近
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x)), atol=1e-6)
        # 断言 np.fft.irfft(np.fft.rfft(x, norm="backward"), norm="backward") 与 x 的结果相近
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm="backward"),
                        norm="backward"), atol=1e-6)
        # 断言 np.fft.irfft(np.fft.rfft(x, norm="ortho"), norm="ortho") 与 x 的结果相近
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm="ortho"),
                        norm="ortho"), atol=1e-6)
        # 断言 np.fft.irfft(np.fft.rfft(x, norm="forward"), norm="forward") 与 x 的结果相近
        assert_allclose(x, np.fft.irfft(np.fft.rfft(x, norm="forward"),
                        norm="forward"), atol=1e-6)

    # 定义测试函数 test_rfft2，用于测试 np.fft.rfft2 函数的不同参数组合
    def test_rfft2(self):
        # 创建大小为 (30, 20) 的随机二维数组 x
        x = random((30, 20))
        # 断言 np.fft.fft2(x)[:, :11] 与 np.fft.rfft2(x) 的结果相近
        assert_allclose(np.fft.fft2(x)[:, :11], np.fft.rfft2(x), atol=1e-6)
        # 断言 np.fft.rfft2(x) 与 np.fft.rfft2(x, norm="backward") 的结果相近
        assert_allclose(np.fft.rfft2(x),
                        np.fft.rfft2(x, norm="backward"), atol=1e-6)
        # 断言 np.fft.rfft2(x) / np.sqrt(30 * 20) 与 np.fft.rfft2(x, norm="ortho") 的结果相近
        assert_allclose(np.fft.rfft2(x) / np.sqrt(30 * 20),
                        np.fft.rfft2(x, norm="ortho"), atol=1e-6)
        # 断言 np.fft.rfft2(x) / (30. * 20.) 与 np.fft.rfft2(x, norm="forward") 的结果相近
        assert_allclose(np.fft.rfft2(x) / (30. * 20.),
                        np.fft.rfft2(x, norm="forward"), atol=1e-6)

    # 定义测试函数 test_irfft2，用于测试 np.fft.irfft2 函数的不同参数组合
    def test_irfft2(self):
        # 创建大小为 (30, 20) 的随机二维数组 x
        x = random((30, 20))
        # 断言 np.fft.irfft2(np.fft.rfft2(x)) 与 x 的结果相近
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x)), atol=1e-6)
        # 断言 np.fft.irfft2(np.fft.rfft2(x, norm="backward"), norm="backward") 与 x 的结果相近
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm="backward"),
                        norm="backward"), atol=1e-6)
        # 断言 np.fft.irfft2(np.fft.rfft2(x, norm="ortho"), norm="ortho") 与 x 的结果相近
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm="ortho"),
                        norm="ortho"), atol=1e-6)
        # 断言 np.fft.irfft2(np.fft.rfft2(x, norm="forward"), norm="forward") 与 x 的结果相近
        assert_allclose(x, np.fft.irfft2(np.fft.rfft2(x, norm="forward"),
                        norm="forward"), atol=1e-6)
    # 定义测试函数 test_rfftn，用于测试 np.fft.rfftn 函数的功能
    def test_rfftn(self):
        # 创建一个形状为 (30, 20, 10) 的随机数组 x
        x = random((30, 20, 10))
        # 断言 np.fft.fftn(x) 的前三个维度切片与 np.fft.rfftn(x) 相近，允许的绝对误差为 1e-6
        assert_allclose(np.fft.fftn(x)[:, :, :6], np.fft.rfftn(x), atol=1e-6)
        # 断言 np.fft.rfftn(x) 与 np.fft.rfftn(x, norm="backward") 相近，允许的绝对误差为 1e-6
        assert_allclose(np.fft.rfftn(x),
                        np.fft.rfftn(x, norm="backward"), atol=1e-6)
        # 断言 np.fft.rfftn(x) 除以 np.sqrt(30 * 20 * 10) 与 np.fft.rfftn(x, norm="ortho") 相近，允许的绝对误差为 1e-6
        assert_allclose(np.fft.rfftn(x) / np.sqrt(30 * 20 * 10),
                        np.fft.rfftn(x, norm="ortho"), atol=1e-6)
        # 断言 np.fft.rfftn(x) 除以 (30. * 20. * 10.) 与 np.fft.rfftn(x, norm="forward") 相近，允许的绝对误差为 1e-6
        assert_allclose(np.fft.rfftn(x) / (30. * 20. * 10.),
                        np.fft.rfftn(x, norm="forward"), atol=1e-6)

    # 定义测试函数 test_irfftn，用于测试 np.fft.irfftn 函数的功能
    def test_irfftn(self):
        # 创建一个形状为 (30, 20, 10) 的随机数组 x
        x = random((30, 20, 10))
        # 断言 x 与 np.fft.irfftn(np.fft.rfftn(x)) 相近，允许的绝对误差为 1e-6
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x)), atol=1e-6)
        # 断言 x 与 np.fft.irfftn(np.fft.rfftn(x, norm="backward"), norm="backward") 相近，允许的绝对误差为 1e-6
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm="backward"),
                        norm="backward"), atol=1e-6)
        # 断言 x 与 np.fft.irfftn(np.fft.rfftn(x, norm="ortho"), norm="ortho") 相近，允许的绝对误差为 1e-6
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm="ortho"),
                        norm="ortho"), atol=1e-6)
        # 断言 x 与 np.fft.irfftn(np.fft.rfftn(x, norm="forward"), norm="forward") 相近，允许的绝对误差为 1e-6
        assert_allclose(x, np.fft.irfftn(np.fft.rfftn(x, norm="forward"),
                        norm="forward"), atol=1e-6)

    # 定义测试函数 test_hfft，用于测试 np.fft.hfft 函数的功能
    def test_hfft(self):
        # 创建长度为 14 的随机复数数组 x
        x = random(14) + 1j * random(14)
        # 构造 Hermite 对称序列 x_herm
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        # 断言 np.fft.fft(x) 与 np.fft.hfft(x_herm) 相近，允许的绝对误差为 1e-6
        assert_allclose(np.fft.fft(x), np.fft.hfft(x_herm), atol=1e-6)
        # 断言 np.fft.hfft(x_herm) 与 np.fft.hfft(x_herm, norm="backward") 相近，允许的绝对误差为 1e-6
        assert_allclose(np.fft.hfft(x_herm),
                        np.fft.hfft(x_herm, norm="backward"), atol=1e-6)
        # 断言 np.fft.hfft(x_herm) 除以 np.sqrt(30) 与 np.fft.hfft(x_herm, norm="ortho") 相近，允许的绝对误差为 1e-6
        assert_allclose(np.fft.hfft(x_herm) / np.sqrt(30),
                        np.fft.hfft(x_herm, norm="ortho"), atol=1e-6)
        # 断言 np.fft.hfft(x_herm) 除以 30. 与 np.fft.hfft(x_herm, norm="forward") 相近，允许的绝对误差为 1e-6
        assert_allclose(np.fft.hfft(x_herm) / 30.,
                        np.fft.hfft(x_herm, norm="forward"), atol=1e-6)

    # 定义测试函数 test_ihfft，用于测试 np.fft.ihfft 函数的功能
    def test_ihfft(self):
        # 创建长度为 14 的随机复数数组 x
        x = random(14) + 1j * random(14)
        # 构造 Hermite 对称序列 x_herm
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        # 断言 x_herm 与 np.fft.ihfft(np.fft.hfft(x_herm)) 相近，允许的绝对误差为 1e-6
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm)), atol=1e-6)
        # 断言 x_herm 与 np.fft.ihfft(np.fft.hfft(x_herm, norm="backward"), norm="backward") 相近，允许的绝对误差为 1e-6
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm,
                        norm="backward"), norm="backward"), atol=1e-6)
        # 断言 x_herm 与 np.fft.ihfft(np.fft.hfft(x_herm, norm="ortho"), norm="ortho") 相近，允许的绝对误差为 1e-6
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm,
                        norm="ortho"), norm="ortho"), atol=1e-6)
        # 断言 x_herm 与 np.fft.ihfft(np.fft.hfft(x_herm, norm="forward"), norm="forward") 相近，允许的绝对误差为 1e-6
        assert_allclose(x_herm, np.fft.ihfft(np.fft.hfft(x_herm,
                        norm="forward"), norm="forward"), atol=1e-6)

    # 使用 pytest 的参数化装饰器，定义测试函数 test_axes，测试多种 FFT 相关函数在不同轴上的转置操作
    @pytest.mark.parametrize("op", [np.fft.fftn, np.fft.ifftn,
                                    np.fft.rfftn, np.fft.irfftn])
    def test_axes(self, op):
        # 创建一个形状为 (30, 20, 10) 的随机数组 x
        x = random((30, 20, 10))
        # 定义多种轴置换方案
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        # 遍历每种轴置换方案
        for a in axes:
            # 对 np.transpose(x, a) 进行 op 操作
            op_tr = op(np.transpose(x, a))
            # 对 x 在轴 a 上进行 op 操作，并再次进行轴置换为 a
            tr_op = np.transpose(op(x, axes=a), a)
            # 断言 op_tr 与 tr_op 相近，允许的绝对误差为 1e-6
            assert_allclose(op_tr, tr_op, atol=1e-
    # 定义测试方法，用于测试带有 s 参数的操作函数 op
    def test_s_negative_1(self, op):
        # 创建一个 10x10 的二维数组，包含 0 到 99 的整数
        x = np.arange(100).reshape(10, 10)
        # 断言调用 op 函数后返回的数组形状为 (10, 5)
        # s=(-1, 5) 表示沿着第一个轴使用整个输入数组
        assert op(x, s=(-1, 5), axes=(0, 1)).shape == (10, 5)

    # 使用 pytest 的参数化装饰器，测试带有 s 参数和警告的操作函数 op
    @pytest.mark.parametrize("op", [np.fft.fftn, np.fft.ifftn,
                                    np.fft.rfftn, np.fft.irfftn])
    def test_s_axes_none(self, op):
        # 创建一个 10x10 的二维数组，包含 0 到 99 的整数
        x = np.arange(100).reshape(10, 10)
        # 期望捕获警告，警告信息应包含 "`axes` should not be `None` if `s`"
        with pytest.warns(match='`axes` should not be `None` if `s`'):
            # 调用 op 函数，传入 s=(-1, 5)
            op(x, s=(-1, 5))

    # 使用 pytest 的参数化装饰器，测试带有 s 参数和警告的 2D 操作函数 op
    @pytest.mark.parametrize("op", [np.fft.fft2, np.fft.ifft2])
    def test_s_axes_none_2D(self, op):
        # 创建一个 10x10 的二维数组，包含 0 到 99 的整数
        x = np.arange(100).reshape(10, 10)
        # 期望捕获警告，警告信息应包含 "`axes` should not be `None` if `s`"
        with pytest.warns(match='`axes` should not be `None` if `s`'):
            # 调用 op 函数，传入 s=(-1, 5)，axes=None
            op(x, s=(-1, 5), axes=None)

    # 使用 pytest 的参数化装饰器，测试带有 s 参数和包含 None 的操作函数 op
    @pytest.mark.parametrize("op", [np.fft.fftn, np.fft.ifftn,
                                    np.fft.rfftn, np.fft.irfftn,
                                    np.fft.fft2, np.fft.ifft2])
    def test_s_contains_none(self, op):
        # 创建一个形状为 (30, 20, 10) 的随机数组
        x = random((30, 20, 10))
        # 期望捕获警告，警告信息应包含 "array containing `None` values to `s`"
        with pytest.warns(match='array containing `None` values to `s`'):
            # 调用 op 函数，传入 s=(10, None, 10)，axes=(0, 1, 2)
            op(x, s=(10, None, 10), axes=(0, 1, 2))

    # 定义测试方法，验证一维正向和反向变换是否保持范数不变
    def test_all_1d_norm_preserving(self):
        # 创建一个包含 30 个随机元素的一维数组
        x = random(30)
        # 计算原始数组 x 的范数
        x_norm = np.linalg.norm(x)
        # 将数组长度倍增为 n，并测试所有的正向和反向变换
        # func_pairs 列表包含了正向和反向变换函数对
        func_pairs = [(np.fft.fft, np.fft.ifft),
                      (np.fft.rfft, np.fft.irfft),
                      # hfft: 指定第一个函数取 x.size 个样本以便与上述 x_norm 进行比较
                      (np.fft.ihfft, np.fft.hfft),
                      ]
        for forw, back in func_pairs:
            for n in [x.size, 2*x.size]:
                for norm in [None, 'backward', 'ortho', 'forward']:
                    # 执行正向变换
                    tmp = forw(x, n=n, norm=norm)
                    # 执行反向变换
                    tmp = back(tmp, n=n, norm=norm)
                    # 断言经过变换后的范数与原始 x 的范数相近，误差限定为 1e-6
                    assert_allclose(x_norm,
                                    np.linalg.norm(tmp), atol=1e-6)

    # 使用 pytest 的参数化装饰器，测试不同参数 axes 和 dtype 的操作
    @pytest.mark.parametrize("axes", [(0, 1), (0, 2), None])
    @pytest.mark.parametrize("dtype", (complex, float))
    @pytest.mark.parametrize("transpose", (True, False))
    # 定义一个内部函数 `zeros_like`，根据输入数组 `x` 的形状返回一个零填充的数组，如果 `transpose` 为真，则对 `x` 进行转置操作
    def zeros_like(x):
        if transpose:
            return np.zeros_like(x.T).T
        else:
            return np.zeros_like(x)

    # 根据参数 `dtype` 的类型（复数或实数），选择不同的随机数组 `x`，并分别设置 FFT 和 IFFT 函数
    if dtype is complex:
        x = random((10, 5, 6)) + 1j*random((10, 5, 6))
        fft, ifft = np.fft.fftn, np.fft.ifftn
    else:
        x = random((10, 5, 6))
        fft, ifft = np.fft.rfftn, np.fft.irfftn

    # 计算预期的 FFT 结果
    expected = fft(x, axes=axes)
    # 创建一个与 `expected` 形状相同的零填充数组 `out`
    out = zeros_like(expected)
    # 在指定输出数组 `out` 的情况下计算 FFT，结果保存在 `result` 中
    result = fft(x, out=out, axes=axes)
    # 断言 `result` 与 `out` 是同一个对象
    assert result is out
    # 断言 `result` 与 `expected` 相等
    assert_array_equal(result, expected)

    # 计算预期的 IFFT 结果
    expected2 = ifft(expected, axes=axes)
    # 如果 `dtype` 是复数，则 `out2` 使用之前计算的 `out`；否则，创建一个与 `expected2` 形状相同的零填充数组 `out2`
    out2 = out if dtype is complex else zeros_like(expected2)
    # 在指定输出数组 `out2` 的情况下计算 IFFT，结果保存在 `result2` 中
    result2 = ifft(out, out=out2, axes=axes)
    # 断言 `result2` 与 `out2` 是同一个对象
    assert result2 is out2
    # 断言 `result2` 与 `expected2` 相等
    assert_array_equal(result2, expected2)

@pytest.mark.parametrize("fft", [np.fft.fftn, np.fft.ifftn, np.fft.rfftn])
def test_fftn_out_and_s_interaction(self, fft):
    # 对于带有 `s` 参数的情况，由于形状可能变化，通常不能传递 `out` 参数。
    if fft is np.fft.rfftn:
        x = random((10, 5, 6))
    else:
        x = random((10, 5, 6)) + 1j*random((10, 5, 6))
    # 使用 `pytest.raises` 断言，在指定 `out` 参数的情况下会引发 ValueError 异常，并且异常信息包含 "has wrong shape"
    with pytest.raises(ValueError, match="has wrong shape"):
        fft(x, out=np.zeros_like(x), s=(3, 3, 3), axes=(0, 1, 2))
    # 除了第一个轴（即 `axes` 的最后一个轴）之外，对于指定了 `s` 参数的情况，计算预期的 FFT 结果
    s = (10, 5, 5)
    expected = fft(x, s=s, axes=(0, 1, 2))
    # 创建一个与 `expected` 形状相同的零填充数组 `out`
    out = np.zeros_like(expected)
    # 在指定输出数组 `out` 的情况下计算 FFT，结果保存在 `result` 中
    result = fft(x, s=s, axes=(0, 1, 2), out=out)
    # 断言 `result` 与 `out` 是同一个对象
    assert result is out
    # 断言 `result` 与 `expected` 相等
    assert_array_equal(result, expected)

@pytest.mark.parametrize("s", [(9, 5, 5), (3, 3, 3)])
def test_irfftn_out_and_s_interaction(self, s):
    # 对于 `irfftn`，输出是实数，因此不能用于中间步骤，应始终有效。
    x = random((9, 5, 6, 2)) + 1j*random((9, 5, 6, 2))
    # 计算预期的 `irfftn` 结果
    expected = np.fft.irfftn(x, s=s, axes=(0, 1, 2))
    # 创建一个与 `expected` 形状相同的零填充数组 `out`
    out = np.zeros_like(expected)
    # 在指定输出数组 `out` 的情况下计算 `irfftn`，结果保存在 `result` 中
    result = np.fft.irfftn(x, s=s, axes=(0, 1, 2), out=out)
    # 断言 `result` 与 `out` 是同一个对象
    assert result is out
    # 断言 `result` 与 `expected` 相等
    assert_array_equal(result, expected)
@pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("order", ["F", 'non-contiguous'])
@pytest.mark.parametrize(
        "fft",
        [np.fft.fft, np.fft.fft2, np.fft.fftn,
         np.fft.ifft, np.fft.ifft2, np.fft.ifftn])
def test_fft_with_order(dtype, order, fft):
    # 检查对于C、Fortran和非连续数组，FFT/IFFT是否产生相同结果
    rng = np.random.RandomState(42)
    X = rng.rand(8, 7, 13).astype(dtype, copy=False)
    # 根据 pull/14178 中的讨论设置容差值
    _tol = 8.0 * np.sqrt(np.log2(X.size)) * np.finfo(X.dtype).eps
    if order == 'F':
        Y = np.asfortranarray(X)
    else:
        # 创建一个非连续数组
        Y = X[::-1]
        X = np.ascontiguousarray(X[::-1])

    if fft.__name__.endswith('fft'):
        # 对于 fft 函数，沿着三个轴进行循环
        for axis in range(3):
            X_res = fft(X, axis=axis)
            Y_res = fft(Y, axis=axis)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    elif fft.__name__.endswith(('fft2', 'fftn')):
        # 对于 fft2 和 fftn 函数，使用不同的轴组合
        axes = [(0, 1), (1, 2), (0, 2)]
        if fft.__name__.endswith('fftn'):
            axes.extend([(0,), (1,), (2,), None])
        for ax in axes:
            X_res = fft(X, axes=ax)
            Y_res = fft(Y, axes=ax)
            assert_allclose(X_res, Y_res, atol=_tol, rtol=_tol)
    else:
        # 抛出数值错误异常
        raise ValueError()


@pytest.mark.skipif(IS_WASM, reason="Cannot start thread")
class TestFFTThreadSafe:
    threads = 16
    input_shape = (800, 200)

    def _test_mtsame(self, func, *args):
        def worker(args, q):
            q.put(func(*args))

        q = queue.Queue()
        expected = func(*args)

        # 启动一组线程同时调用同一函数
        t = [threading.Thread(target=worker, args=(args, q))
             for i in range(self.threads)]
        [x.start() for x in t]

        [x.join() for x in t]
        # 确保所有线程返回正确的值
        for i in range(self.threads):
            assert_array_equal(q.get(timeout=5), expected,
                'Function returned wrong value in multithreaded context')

    def test_fft(self):
        a = np.ones(self.input_shape) * 1+0j
        self._test_mtsame(np.fft.fft, a)

    def test_ifft(self):
        a = np.ones(self.input_shape) * 1+0j
        self._test_mtsame(np.fft.ifft, a)

    def test_rfft(self):
        a = np.ones(self.input_shape)
        self._test_mtsame(np.fft.rfft, a)

    def test_irfft(self):
        a = np.ones(self.input_shape) * 1+0j
        self._test_mtsame(np.fft.irfft, a)


def test_irfft_with_n_1_regression():
    # gh-25661 的回归测试
    x = np.arange(10)
    np.fft.irfft(x, n=1)
    np.fft.hfft(x, n=1)
    np.fft.irfft(np.array([0], complex), n=10)


def test_irfft_with_n_large_regression():
    # gh-25679 的回归测试
    x = np.arange(5) * (1 + 1j)
    result = np.fft.hfft(x, n=10)
    # 定义预期的 NumPy 数组，包含期望的浮点数值
    expected = np.array([20., 9.91628173, -11.8819096, 7.1048486,
                         -6.62459848, 4., -3.37540152, -0.16057669,
                         1.8819096, -20.86055364])
    # 使用 assert_allclose 函数检查 result 数组与预期数组 expected 是否在允许误差范围内相等
    assert_allclose(result, expected)
# 使用 pytest 的 parametrize 装饰器为 test_fft_with_integer_or_bool_input 函数参数化，fft 参数为四种不同的 FFT 函数
@pytest.mark.parametrize("fft", [
    np.fft.fft, np.fft.ifft, np.fft.rfft, np.fft.irfft
])
# 使用 pytest 的 parametrize 装饰器为 test_fft_with_integer_or_bool_input 函数参数化，data 参数为三种不同的输入数据类型和值
@pytest.mark.parametrize("data", [
    np.array([False, True, False]),   # 布尔类型数组作为输入数据
    np.arange(10, dtype=np.uint8),    # 无符号 8 位整数数组作为输入数据
    np.arange(5, dtype=np.int16),     # 有符号 16 位整数数组作为输入数据
])
def test_fft_with_integer_or_bool_input(data, fft):
    # Regression test for gh-25819
    # 对于给定的数据 data，使用指定的 FFT 函数 fft 进行变换
    result = fft(data)
    # 将输入数据 data 转换为浮点数类型，用于比较期望结果
    float_data = data.astype(np.result_type(data, 1.))
    # 使用 fft 函数对转换后的浮点数数据进行变换，得到期望结果
    expected = fft(float_data)
    # 检查变换后的结果是否一致
    assert_array_equal(result, expected)
```