# `D:\src\scipysrc\scipy\scipy\fft\tests\test_basic.py`

```
# 导入必要的模块和库
import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises
import scipy.fft as fft
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import (
    array_namespace, size, xp_assert_close, xp_assert_equal
)

# 将 array_api_compatible 和 skip_xp_backends 添加到 pytest 的标记中
pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]

# 定义 skip_xp_backends 标记，用于跳过不兼容的后端
skip_xp_backends = pytest.mark.skip_xp_backends

# 根据 FFT 函数和数组接口选择期望的输入数据类型
def get_expected_input_dtype(func, xp):
    if func in [fft.fft, fft.fftn, fft.fft2,
                fft.ifft, fft.ifftn, fft.ifft2,
                fft.hfft, fft.hfftn, fft.hfft2,
                fft.irfft, fft.irfftn, fft.irfft2]:
        dtype = xp.complex128
    elif func in [fft.rfft, fft.rfftn, fft.rfft2,
                  fft.ihfft, fft.ihfftn, fft.ihfft2]:
        dtype = xp.float64
    else:
        raise ValueError(f'Unknown FFT function: {func}')

    return dtype

# 定义一维 FFT 函数
def fft1(x):
    L = len(x)
    phase = -2j*np.pi*(np.arange(L)/float(L))
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x*np.exp(phase), axis=1)

# 定义一维 FFT 测试类
class TestFFT1D:

    # 测试一维 FFT 的恒等性
    def test_identity(self, xp):
        maxlen = 512
        x = xp.asarray(random(maxlen) + 1j*random(maxlen))
        xr = xp.asarray(random(maxlen))
        # 检查一些 2 的幂次方和一些质数
        for i in [1, 2, 16, 128, 512, 53, 149, 281, 397]:
            xp_assert_close(fft.ifft(fft.fft(x[0:i])), x[0:i])
            xp_assert_close(fft.irfft(fft.rfft(xr[0:i]), i), xr[0:i])
    
    # 详尽测试一维 FFT 的恒等性，跳过特定的后端
    @skip_xp_backends(np_only=True, reasons=['significant overhead for some backends'])
    def test_identity_extensive(self, xp):
        maxlen = 512
        x = xp.asarray(random(maxlen) + 1j*random(maxlen))
        xr = xp.asarray(random(maxlen))
        for i in range(1, maxlen):
            xp_assert_close(fft.ifft(fft.fft(x[0:i])), x[0:i])
            xp_assert_close(fft.irfft(fft.rfft(xr[0:i]), i), xr[0:i])

    # 测试一维 FFT
    def test_fft(self, xp):
        x = random(30) + 1j*random(30)
        expect = xp.asarray(fft1(x))
        x = xp.asarray(x)
        xp_assert_close(fft.fft(x), expect)
        xp_assert_close(fft.fft(x, norm="backward"), expect)
        xp_assert_close(fft.fft(x, norm="ortho"),
                        expect / xp.sqrt(xp.asarray(30, dtype=xp.float64)),)
        xp_assert_close(fft.fft(x, norm="forward"), expect / 30)

    # 测试一维 FFT，跳过特定的后端
    @skip_xp_backends(np_only=True, reasons=['some backends allow `n=0`'])
    def test_fft_n(self, xp):
        x = xp.asarray([1, 2, 3], dtype=xp.complex128)
        assert_raises(ValueError, fft.fft, x, 0)
    # 定义一个测试函数，用于测试逆快速傅里叶变换（ifft）功能
    def test_ifft(self, xp):
        # 生成一个包含随机复数的数组，并将其转换为xp（可能是NumPy或类似的库）支持的数组类型
        x = xp.asarray(random(30) + 1j*random(30))
        # 断言逆傅里叶变换应用于傅里叶变换的结果，结果应该接近于原始输入数组x
        xp_assert_close(fft.ifft(fft.fft(x)), x)
        # 遍历不同的归一化模式（"backward", "ortho", "forward"），测试对应的傅里叶变换和逆变换
        for norm in ["backward", "ortho", "forward"]:
            xp_assert_close(fft.ifft(fft.fft(x, norm=norm), norm=norm), x)

    # 定义一个测试函数，用于测试二维快速傅里叶变换（fft2）功能
    def test_fft2(self, xp):
        # 生成一个随机的二维复数数组，并将其转换为xp支持的数组类型
        x = xp.asarray(random((30, 20)) + 1j*random((30, 20)))
        # 计算期望的二维快速傅里叶变换结果
        expect = fft.fft(fft.fft(x, axis=1), axis=0)
        # 断言二维快速傅里叶变换应用于x的结果应该接近于期望的结果
        xp_assert_close(fft.fft2(x), expect)
        # 测试使用不同归一化模式的fft2函数
        xp_assert_close(fft.fft2(x, norm="backward"), expect)
        xp_assert_close(fft.fft2(x, norm="ortho"),
                        expect / xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64)))
        xp_assert_close(fft.fft2(x, norm="forward"), expect / (30 * 20))

    # 定义一个测试函数，用于测试二维逆快速傅里叶变换（ifft2）功能
    def test_ifft2(self, xp):
        # 生成一个随机的二维复数数组，并将其转换为xp支持的数组类型
        x = xp.asarray(random((30, 20)) + 1j*random((30, 20)))
        # 计算期望的二维逆快速傅里叶变换结果
        expect = fft.ifft(fft.ifft(x, axis=1), axis=0)
        # 断言二维逆快速傅里叶变换应用于x的结果应该接近于期望的结果
        xp_assert_close(fft.ifft2(x), expect)
        # 测试使用不同归一化模式的ifft2函数
        xp_assert_close(fft.ifft2(x, norm="backward"), expect)
        xp_assert_close(fft.ifft2(x, norm="ortho"),
                        expect * xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64)))
        xp_assert_close(fft.ifft2(x, norm="forward"), expect * (30 * 20))

    # 定义一个测试函数，用于测试多维快速傅里叶变换（fftn）功能
    def test_fftn(self, xp):
        # 生成一个随机的三维复数数组，并将其转换为xp支持的数组类型
        x = xp.asarray(random((30, 20, 10)) + 1j*random((30, 20, 10)))
        # 计算期望的多维快速傅里叶变换结果
        expect = fft.fft(fft.fft(fft.fft(x, axis=2), axis=1), axis=0)
        # 断言多维快速傅里叶变换应用于x的结果应该接近于期望的结果
        xp_assert_close(fft.fftn(x), expect)
        # 测试使用不同归一化模式的fftn函数
        xp_assert_close(fft.fftn(x, norm="backward"), expect)
        xp_assert_close(fft.fftn(x, norm="ortho"),
                        expect / xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64)))
        xp_assert_close(fft.fftn(x, norm="forward"), expect / (30 * 20 * 10))

    # 定义一个测试函数，用于测试多维逆快速傅里叶变换（ifftn）功能
    def test_ifftn(self, xp):
        # 生成一个随机的三维复数数组，并将其转换为xp支持的数组类型
        x = xp.asarray(random((30, 20, 10)) + 1j*random((30, 20, 10)))
        # 计算期望的多维逆快速傅里叶变换结果
        expect = fft.ifft(fft.ifft(fft.ifft(x, axis=2), axis=1), axis=0)
        # 断言多维逆快速傅里叶变换应用于x的结果应该接近于期望的结果，允许一定的相对误差
        xp_assert_close(fft.ifftn(x), expect, rtol=1e-7)
        # 测试使用不同归一化模式的ifftn函数
        xp_assert_close(fft.ifftn(x, norm="backward"), expect, rtol=1e-7)
        xp_assert_close(
            fft.ifftn(x, norm="ortho"),
            fft.ifftn(x) * xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64))
        )
        xp_assert_close(fft.ifftn(x, norm="forward"),
                        expect * (30 * 20 * 10),
                        rtol=1e-7)

    # 定义一个测试函数，用于测试实数输入的快速傅里叶变换（rfft）功能
    def test_rfft(self, xp):
        # 生成一个随机的实数数组，并将其转换为xp支持的数组类型
        x = xp.asarray(random(29), dtype=xp.float64)
        # 对不同大小（size(x), 2*size(x)）和归一化模式（None, "backward", "ortho", "forward"）进行测试
        for n in [size(x), 2*size(x)]:
            for norm in [None, "backward", "ortho", "forward"]:
                # 断言rfft函数应用于x的结果应该接近于对应的fft函数的结果（取前n//2 + 1个元素）
                xp_assert_close(fft.rfft(x, n=n, norm=norm),
                                fft.fft(xp.asarray(x, dtype=xp.complex128),
                                        n=n, norm=norm)[:(n//2 + 1)])
            # 测试使用归一化模式为"ortho"时的rfft函数
            xp_assert_close(
                fft.rfft(x, n=n, norm="ortho"),
                fft.rfft(x, n=n) / xp.sqrt(xp.asarray(n, dtype=xp.float64))
            )
    # 定义测试函数 test_irfft，接受一个 xp 参数作为不同库（如 NumPy、CuPy）的数组对象
    def test_irfft(self, xp):
        # 生成一个包含 30 个随机数的数组，并将其转换为 xp 数组
        x = xp.asarray(random(30))
        # 断言 fft.irfft(fft.rfft(x)) 的结果与 x 近似相等
        xp_assert_close(fft.irfft(fft.rfft(x)), x)
        
        # 遍历不同的归一化选项：backward、ortho、forward
        for norm in ["backward", "ortho", "forward"]:
            # 断言使用指定归一化参数 norm 的 fft.irfft(fft.rfft(x, norm=norm)) 的结果与 x 近似相等
            xp_assert_close(fft.irfft(fft.rfft(x, norm=norm), norm=norm), x)

    # 定义测试函数 test_rfft2，接受一个 xp 参数作为不同库（如 NumPy、CuPy）的数组对象
    def test_rfft2(self, xp):
        # 生成一个形状为 (30, 20) 的随机浮点数数组，并将其转换为 xp 数组
        x = xp.asarray(random((30, 20)), dtype=xp.float64)
        # 生成期望的 fft.fft2(x) 的结果，并对其取前 11 列
        expect = fft.fft2(xp.asarray(x, dtype=xp.complex128))[:, :11]
        # 断言 fft.rfft2(x) 的结果与预期的 expect 近似相等
        xp_assert_close(fft.rfft2(x), expect)
        # 断言使用 norm="backward" 归一化参数的 fft.rfft2(x) 的结果与预期的 expect 近似相等
        xp_assert_close(fft.rfft2(x, norm="backward"), expect)
        # 断言使用 norm="ortho" 归一化参数的 fft.rfft2(x) 的结果与预期的 expect 除以归一化因子后近似相等
        xp_assert_close(fft.rfft2(x, norm="ortho"),
                        expect / xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64)))
        # 断言使用 norm="forward" 归一化参数的 fft.rfft2(x) 的结果与预期的 expect 除以归一化因子后近似相等
        xp_assert_close(fft.rfft2(x, norm="forward"), expect / (30 * 20))

    # 定义测试函数 test_irfft2，接受一个 xp 参数作为不同库（如 NumPy、CuPy）的数组对象
    def test_irfft2(self, xp):
        # 生成一个形状为 (30, 20) 的随机数组，并将其转换为 xp 数组
        x = xp.asarray(random((30, 20)))
        # 断言 fft.irfft2(fft.rfft2(x)) 的结果与 x 近似相等
        xp_assert_close(fft.irfft2(fft.rfft2(x)), x)
        
        # 遍历不同的归一化选项：backward、ortho、forward
        for norm in ["backward", "ortho", "forward"]:
            # 断言使用指定归一化参数 norm 的 fft.irfft2(fft.rfft2(x, norm=norm)) 的结果与 x 近似相等
            xp_assert_close(fft.irfft2(fft.rfft2(x, norm=norm), norm=norm), x)

    # 定义测试函数 test_rfftn，接受一个 xp 参数作为不同库（如 NumPy、CuPy）的数组对象
    def test_rfftn(self, xp):
        # 生成一个形状为 (30, 20, 10) 的双精度浮点数数组，并将其转换为 xp 数组
        x = xp.asarray(random((30, 20, 10)), dtype=xp.float64)
        # 生成期望的 fft.fftn(x) 的结果，并对其取前 6 列
        expect = fft.fftn(xp.asarray(x, dtype=xp.complex128))[:, :, :6]
        # 断言 fft.rfftn(x) 的结果与预期的 expect 近似相等
        xp_assert_close(fft.rfftn(x), expect)
        # 断言使用 norm="backward" 归一化参数的 fft.rfftn(x) 的结果与预期的 expect 近似相等
        xp_assert_close(fft.rfftn(x, norm="backward"), expect)
        # 断言使用 norm="ortho" 归一化参数的 fft.rfftn(x) 的结果与预期的 expect 除以归一化因子后近似相等
        xp_assert_close(fft.rfftn(x, norm="ortho"),
                        expect / xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64)))
        # 断言使用 norm="forward" 归一化参数的 fft.rfftn(x) 的结果与预期的 expect 除以归一化因子后近似相等
        xp_assert_close(fft.rfftn(x, norm="forward"), expect / (30 * 20 * 10))

    # 定义测试函数 test_irfftn，接受一个 xp 参数作为不同库（如 NumPy、CuPy）的数组对象
    def test_irfftn(self, xp):
        # 生成一个形状为 (30, 20, 10) 的随机数组，并将其转换为 xp 数组
        x = xp.asarray(random((30, 20, 10)))
        # 断言 fft.irfftn(fft.rfftn(x)) 的结果与 x 近似相等
        xp_assert_close(fft.irfftn(fft.rfftn(x)), x)
        
        # 遍历不同的归一化选项：backward、ortho、forward
        for norm in ["backward", "ortho", "forward"]:
            # 断言使用指定归一化参数 norm 的 fft.irfftn(fft.rfftn(x, norm=norm)) 的结果与 x 近似相等
            xp_assert_close(fft.irfftn(fft.rfftn(x, norm=norm), norm=norm), x)

    # 定义测试函数 test_hfft，接受一个 xp 参数作为不同库（如 NumPy、CuPy）的数组对象
    def test_hfft(self, xp):
        # 生成一个包含 14 个随机复数的数组
        x = random(14) + 1j * random(14)
        # 构造 Hermite 对称数组 x_herm
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        # 将构造的数组转换为 xp 数组
        x = xp.asarray(x)
        x_herm = xp.asarray(x_herm)
        # 生成期望的 fft.fft(x) 的实部作为期望值
        expect = xp.real(fft.fft(x))
        # 断言 fft.hfft(x_herm) 的结果与期望的 expect 近似相等
        xp_assert_close(fft.hfft(x_herm), expect)
        # 断言使用 norm="backward" 归一化参数的 fft.hfft(x_herm) 的结果与期望的 expect 近似相等
        xp_assert_close(fft.hfft(x_herm, norm="backward"), expect)
        # 断言使用 norm="ortho" 归一化参数的 fft.hfft(x_herm) 的结果与期望的 expect 除以归一化因子后近似相等
        xp_assert_close(fft.hfft(x_herm, norm="ortho"),
                        expect / xp.sqrt(xp.asarray(30, dtype=xp.float64)))
        # 断言使用 norm="forward" 归一化参数的 fft.hfft(x_herm) 的结果与期望的 expect 除以归一化因子后近似相等
        xp_assert_close(fft.hfft(x_herm, norm="forward"), expect / 30)

    # 定义测试函数 test_ihfft，接受一个 xp 参数作为不同库（如 NumPy、CuPy）的数组对象
    def test_ihfft(self, xp):
        # 生成一个包含 14 个随机复数的数组
        x = random(14) + 1j * random(14)
        # 构造 Hermite 对称数组 x_herm
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        # 将构
    # 定义测试函数，测试 fft.hfft2 函数
    def test_hfft2(self, xp):
        # 创建一个随机数组并转换为 xp 数组
        x = xp.asarray(random((30, 20)))
        # 断言 fft.ihfft2(fft.hfft2(x)) 结果与 x 接近
        xp_assert_close(fft.hfft2(fft.ihfft2(x)), x)
        # 遍历不同的归一化选项进行测试
        for norm in ["backward", "ortho", "forward"]:
            # 断言带有不同归一化选项的 fft.ihfft2(fft.hfft2(x, norm=norm), norm=norm) 结果与 x 接近
            xp_assert_close(fft.hfft2(fft.ihfft2(x, norm=norm), norm=norm), x)

    # 定义测试函数，测试 fft.ihfft2 函数
    def test_ihfft2(self, xp):
        # 创建一个随机数组并转换为 xp 数组，指定数据类型为 xp.float64
        x = xp.asarray(random((30, 20)), dtype=xp.float64)
        # 期望的结果是使用 fft.ifft2 转换的 x 的部分结果
        expect = fft.ifft2(xp.asarray(x, dtype=xp.complex128))[:, :11]
        # 断言 fft.ihfft2(x) 结果与 expect 接近
        xp_assert_close(fft.ihfft2(x), expect)
        # 断言带有 "backward" 归一化选项的 fft.ihfft2(x, norm="backward") 结果与 expect 接近
        xp_assert_close(fft.ihfft2(x, norm="backward"), expect)
        # 断言带有 "ortho" 归一化选项的 fft.ihfft2(x, norm="ortho") 结果与 expect 乘以归一化系数接近
        xp_assert_close(
            fft.ihfft2(x, norm="ortho"),
            expect * xp.sqrt(xp.asarray(30 * 20, dtype=xp.float64))
        )
        # 断言带有 "forward" 归一化选项的 fft.ihfft2(x, norm="forward") 结果与 expect 乘以 (30 * 20) 接近
        xp_assert_close(fft.ihfft2(x, norm="forward"), expect * (30 * 20))

    # 定义测试函数，测试 fft.hfftn 函数
    def test_hfftn(self, xp):
        # 创建一个三维随机数组并转换为 xp 数组
        x = xp.asarray(random((30, 20, 10)))
        # 断言 fft.hfftn(fft.ihfftn(x)) 结果与 x 接近
        xp_assert_close(fft.hfftn(fft.ihfftn(x)), x)
        # 遍历不同的归一化选项进行测试
        for norm in ["backward", "ortho", "forward"]:
            # 断言带有不同归一化选项的 fft.hfftn(fft.ihfftn(x, norm=norm), norm=norm) 结果与 x 接近
            xp_assert_close(fft.hfftn(fft.ihfftn(x, norm=norm), norm=norm), x)

    # 定义测试函数，测试 fft.ihfftn 函数
    def test_ihfftn(self, xp):
        # 创建一个三维随机数组并转换为 xp 数组，指定数据类型为 xp.float64
        x = xp.asarray(random((30, 20, 10)), dtype=xp.float64)
        # 期望的结果是使用 fft.ifftn 转换的 x 的部分结果
        expect = fft.ifftn(xp.asarray(x, dtype=xp.complex128))[:, :, :6]
        # 断言 expect 与 fft.ihfftn(x) 结果接近
        xp_assert_close(expect, fft.ihfftn(x))
        # 断言 expect 与带有 "backward" 归一化选项的 fft.ihfftn(x, norm="backward") 结果接近
        xp_assert_close(expect, fft.ihfftn(x, norm="backward"))
        # 断言带有 "ortho" 归一化选项的 fft.ihfftn(x, norm="ortho") 结果与 expect 乘以归一化系数接近
        xp_assert_close(
            fft.ihfftn(x, norm="ortho"),
            expect * xp.sqrt(xp.asarray(30 * 20 * 10, dtype=xp.float64))
        )
        # 断言带有 "forward" 归一化选项的 fft.ihfftn(x, norm="forward") 结果与 expect 乘以 (30 * 20 * 10) 接近
        xp_assert_close(fft.ihfftn(x, norm="forward"), expect * (30 * 20 * 10))

    # 定义内部函数，检查轴的顺序
    def _check_axes(self, op, xp):
        # 获取操作函数期望的输入数据类型
        dtype = get_expected_input_dtype(op, xp)
        # 创建一个三维随机数组并转换为指定数据类型的 xp 数组
        x = xp.asarray(random((30, 20, 10)), dtype=dtype)
        # 定义不同的轴顺序
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        # 获取数组的操作命名空间
        xp_test = array_namespace(x)
        # 遍历每个轴顺序
        for a in axes:
            # 对 x 执行轴顺序重排并进行操作 op，将结果保存为 op_tr
            op_tr = op(xp_test.permute_dims(x, axes=a))
            # 对 x 执行操作 op，并对结果进行轴顺序重排，保存为 tr_op
            tr_op = xp_test.permute_dims(op(x, axes=a), axes=a)
            # 断言 op_tr 与 tr_op 的结果接近
            xp_assert_close(op_tr, tr_op)

    # 使用 pytest.mark.parametrize 标注的参数化测试函数，对 fft.fftn、fft.ifftn、fft.rfftn、fft.irfftn 进行测试
    @pytest.mark.parametrize("op", [fft.fftn, fft.ifftn, fft.rfftn, fft.irfftn])
    def test_axes_standard(self, op, xp):
        # 调用内部函数 _check_axes 进行参数化测试
        self._check_axes(op, xp)

    # 使用 pytest.mark.parametrize 标注的参数化测试函数，对 fft.hfftn、fft.ihfftn 进行测试
    @pytest.mark.parametrize("op", [fft.hfftn, fft.ihfftn])
    def test_axes_non_standard(self, op, xp):
        # 调用内部函数 _check_axes 进行参数化测试
        self._check_axes(op, xp)

    # 使用 pytest.mark.parametrize 标注的参数化测试函数，对 fft.fftn、fft.ifftn、fft.rfftn、fft.irfftn 进行测试
    @pytest.mark.parametrize("op", [fft.fftn, fft.ifftn,
                                    fft.rfftn, fft.irfftn])
    # 测试函数，验证在标准形状下，使用给定的操作和数组库操作
    def test_axes_subset_with_shape_standard(self, op, xp):
        # 获取期望的输入数据类型
        dtype = get_expected_input_dtype(op, xp)
        # 创建一个随机数组，形状为 (16, 8, 4)，指定数据类型
        x = xp.asarray(random((16, 8, 4)), dtype=dtype)
        # 定义轴的组合
        axes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        # 创建数组命名空间
        xp_test = array_namespace(x)
        # 遍历每个轴组合
        for a in axes:
            # 在前两个轴上形状不同的数组
            shape = tuple([2*x.shape[ax] if ax in a[:2] else x.shape[ax]
                           for ax in range(x.ndim)])
            # 对前两个轴进行变换
            op_tr = op(xp_test.permute_dims(x, axes=a), s=shape[:2], axes=(0, 1))
            tr_op = xp_test.permute_dims(op(x, s=shape[:2], axes=a[:2]), axes=a)
            # 断言操作结果的近似性
            xp_assert_close(op_tr, tr_op)

    @pytest.mark.parametrize("op", [fft.fft2, fft.ifft2,
                                    fft.rfft2, fft.irfft2,
                                    fft.hfft2, fft.ihfft2,
                                    fft.hfftn, fft.ihfftn])
    # 测试函数，验证在非标准形状下，使用给定的操作和数组库操作
    def test_axes_subset_with_shape_non_standard(self, op, xp):
        # 获取期望的输入数据类型
        dtype = get_expected_input_dtype(op, xp)
        # 创建一个随机数组，形状为 (16, 8, 4)，指定数据类型
        x = xp.asarray(random((16, 8, 4)), dtype=dtype)
        # 定义轴的组合
        axes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        # 创建数组命名空间
        xp_test = array_namespace(x)
        # 遍历每个轴组合
        for a in axes:
            # 在前两个轴上形状不同的数组
            shape = tuple([2*x.shape[ax] if ax in a[:2] else x.shape[ax]
                           for ax in range(x.ndim)])
            # 对前两个轴进行变换
            op_tr = op(xp_test.permute_dims(x, axes=a), s=shape[:2], axes=(0, 1))
            tr_op = xp_test.permute_dims(op(x, s=shape[:2], axes=a[:2]), axes=a)
            # 断言操作结果的近似性
            xp_assert_close(op_tr, tr_op)

    # 测试函数，验证所有一维规范保持不变
    def test_all_1d_norm_preserving(self, xp):
        # 验证往返变换是否保持规范
        x = xp.asarray(random(30), dtype=xp.float64)
        # 创建数组命名空间
        xp_test = array_namespace(x)
        # 计算向量范数
        x_norm = xp_test.linalg.vector_norm(x)
        # 定义变量 n，为数组大小的两倍
        n = size(x) * 2
        # 函数对列表，包含进行前向和后向变换的函数
        func_pairs = [(fft.rfft, fft.irfft),
                      (fft.ihfft, fft.hfft),
                      (fft.fft, fft.ifft)]
        # 遍历函数对
        for forw, back in func_pairs:
            if forw == fft.fft:
                # 如果是 FFT 函数，将数组类型转换为复数型
                x = xp.asarray(x, dtype=xp.complex128)
                # 重新计算向量范数
                x_norm = xp_test.linalg.vector_norm(x)
            # 对每个 n 进行循环测试
            for n in [size(x), 2*size(x)]:
                # 对每种规范进行测试
                for norm in ['backward', 'ortho', 'forward']:
                    # 前向变换
                    tmp = forw(x, n=n, norm=norm)
                    # 后向变换
                    tmp = back(tmp, n=n, norm=norm)
                    # 断言变换后的向量范数与原始向量范数的近似性
                    xp_assert_close(xp_test.linalg.vector_norm(tmp), x_norm)

    # 使用 np_only=True 跳过对特定的后端的测试，参数化为不同的数据类型
    @skip_xp_backends(np_only=True)
    @pytest.mark.parametrize("dtype", [np.float16, np.longdouble])
    # 定义一个测试函数，测试对于非标准数据类型的情况
    def test_dtypes_nonstandard(self, dtype):
        # 创建一个长度为30的随机数组，并将其转换为指定的数据类型
        x = random(30).astype(dtype)
        # 根据给定的数据类型选择输出数据类型的映射关系
        out_dtypes = {np.float16: np.complex64, np.longdouble: np.clongdouble}
        # 将数组 x 转换为复数类型，使用映射表中的数据类型
        x_complex = x.astype(out_dtypes[dtype])

        # 对数组进行快速傅里叶变换和反变换，并进行比较
        res_fft = fft.ifft(fft.fft(x))
        res_rfft = fft.irfft(fft.rfft(x))
        # 对数组进行埃尔米特傅里叶变换和反变换，并指定长度
        res_hfft = fft.hfft(fft.ihfft(x), x.shape[0])
        
        # 检查数值结果和精确的数据类型匹配
        assert_array_almost_equal(res_fft, x_complex)
        assert_array_almost_equal(res_rfft, x)
        assert_array_almost_equal(res_hfft, x)
        assert res_fft.dtype == x_complex.dtype
        assert res_rfft.dtype == np.result_type(np.float32, x.dtype)
        assert res_hfft.dtype == np.result_type(np.float32, x.dtype)

    # 使用参数化装饰器标记，定义一个测试函数，测试对于实数类型的情况
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_dtypes_real(self, dtype, xp):
        # 使用 xp 的特定数据类型创建一个长度为30的随机数组
        x = xp.asarray(random(30), dtype=getattr(xp, dtype))

        # 对数组进行实数快速反傅里叶变换和埃尔米特反傅里叶变换，并进行比较
        res_rfft = fft.irfft(fft.rfft(x))
        res_hfft = fft.hfft(fft.ihfft(x), x.shape[0])
        
        # 检查数值结果和精确的数据类型匹配
        xp_assert_close(res_rfft, x)
        xp_assert_close(res_hfft, x)

    # 使用参数化装饰器标记，定义一个测试函数，测试对于复数类型的情况
    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    def test_dtypes_complex(self, dtype, xp):
        # 使用 xp 的特定数据类型创建一个长度为30的随机数组
        x = xp.asarray(random(30), dtype=getattr(xp, dtype))

        # 对数组进行复数快速傅里叶变换和反变换，并进行比较
        res_fft = fft.ifft(fft.fft(x))
        
        # 检查数值结果和精确的数据类型匹配
        xp_assert_close(res_fft, x)
# 装饰器：跳过不支持 NumPy 的后端，仅在参数 np_only=True 时生效
@skip_xp_backends(np_only=True)
# 使用 pytest.mark.parametrize 装饰器，对以下参数进行参数化测试：dtype
@pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.longdouble,
         np.complex64, np.complex128, np.clongdouble])
# 使用 pytest.mark.parametrize 装饰器，对以下参数进行参数化测试：order
@pytest.mark.parametrize("order", ["F", 'non-contiguous'])
# 使用 pytest.mark.parametrize 装饰器，对以下参数进行参数化测试：fft
@pytest.mark.parametrize(
        "fft",
        [fft.fft, fft.fft2, fft.fftn,
         fft.ifft, fft.ifft2, fft.ifftn])
def test_fft_with_order(dtype, order, fft):
    # 检查 FFT/IFFT 是否能够在 C、Fortran 和非连续数组上产生相同的结果
    rng = np.random.RandomState(42)
    # 创建一个具有指定 dtype 的随机数组 X
    X = rng.rand(8, 7, 13).astype(dtype, copy=False)
    if order == 'F':
        # 如果 order 为 'F'，则创建一个 Fortran 风格的数组 Y
        Y = np.asfortranarray(X)
    else:
        # 否则，创建一个非连续的数组 Y，并将 X 转换为连续数组
        Y = X[::-1]
        X = np.ascontiguousarray(X[::-1])

    # 根据 fft 函数的名称结尾调用相应的 FFT 或 IFFT 函数
    if fft.__name__.endswith('fft'):
        # 对于每一个轴，计算 X 和 Y 的 FFT 结果，并断言它们几乎相等
        for axis in range(3):
            X_res = fft(X, axis=axis)
            Y_res = fft(Y, axis=axis)
            assert_array_almost_equal(X_res, Y_res)
    elif fft.__name__.endswith(('fft2', 'fftn')):
        # 对于 fft2 和 fftn，定义要处理的轴，并计算 X 和 Y 的 FFT 结果
        axes = [(0, 1), (1, 2), (0, 2)]
        if fft.__name__.endswith('fftn'):
            axes.extend([(0,), (1,), (2,), None])
        for ax in axes:
            X_res = fft(X, axes=ax)
            Y_res = fft(Y, axes=ax)
            assert_array_almost_equal(X_res, Y_res)
    else:
        # 如果不是预期的 FFT 函数，则抛出 ValueError 异常
        raise ValueError


# 装饰器：跳过不支持 CPU 的后端，仅在参数 cpu_only=True 时生效
@skip_xp_backends(cpu_only=True)
class TestFFTThreadSafe:
    threads = 16
    input_shape = (800, 200)

    def _test_mtsame(self, func, *args, xp=None):
        def worker(args, q):
            q.put(func(*args))

        q = queue.Queue()
        expected = func(*args)

        # 启动一组线程来同时调用相同的函数
        t = [threading.Thread(target=worker, args=(args, q))
             for i in range(self.threads)]
        [x.start() for x in t]

        [x.join() for x in t]

        # 确保所有线程都返回了正确的值
        for i in range(self.threads):
            xp_assert_equal(
                q.get(timeout=5), expected,
                err_msg='Function returned wrong value in multithreaded context'
            )

    def test_fft(self, xp):
        # 创建一个全为复数 1 的数组 a，并调用 _test_mtsame 方法测试 fft 函数
        a = xp.ones(self.input_shape, dtype=xp.complex128)
        self._test_mtsame(fft.fft, a, xp=xp)

    def test_ifft(self, xp):
        # 创建一个全为 1+0j 的数组 a，并调用 _test_mtsame 方法测试 ifft 函数
        a = xp.full(self.input_shape, 1+0j)
        self._test_mtsame(fft.ifft, a, xp=xp)

    def test_rfft(self, xp):
        # 创建一个全为 1 的数组 a，并调用 _test_mtsame 方法测试 rfft 函数
        a = xp.ones(self.input_shape)
        self._test_mtsame(fft.rfft, a, xp=xp)

    def test_irfft(self, xp):
        # 创建一个全为 1+0j 的数组 a，并调用 _test_mtsame 方法测试 irfft 函数
        a = xp.full(self.input_shape, 1+0j)
        self._test_mtsame(fft.irfft, a, xp=xp)

    def test_hfft(self, xp):
        # 创建一个全为复数 1 的数组 a，并调用 _test_mtsame 方法测试 hfft 函数
        a = xp.ones(self.input_shape, dtype=xp.complex64)
        self._test_mtsame(fft.hfft, a, xp=xp)

    def test_ihfft(self, xp):
        # 创建一个全为 1 的数组 a，并调用 _test_mtsame 方法测试 ihfft 函数
        a = xp.ones(self.input_shape)
        self._test_mtsame(fft.ihfft, a, xp=xp)


# 装饰器：跳过不支持 NumPy 的后端，仅在参数 np_only=True 时生效
@skip_xp_backends(np_only=True)
# 使用 pytest.mark.parametrize 装饰器，对以下参数进行参数化测试：func
@pytest.mark.parametrize("func", [fft.fft, fft.ifft, fft.rfft, fft.irfft])
def test_multiprocess(func):
    # 测试多进程情况下 FFT 和 IFFT 函数的行为
    # 在多进程环境中测试 FFT 是否在 fork 后仍然正常工作 (gh-10422)
    
    # 使用 multiprocessing.Pool 创建一个具有两个进程的进程池
    with multiprocessing.Pool(2) as p:
        # 使用 map 函数将 func 应用到包含四个长度为 100 的全 1 数组的列表上，并收集结果
        res = p.map(func, [np.ones(100) for _ in range(4)])
    
    # 期望的结果是将 func 应用到一个长度为 100 的全 1 数组上得到的结果
    expect = func(np.ones(100))
    
    # 遍历多进程操作后得到的结果列表
    for x in res:
        # 使用 assert_allclose 断言函数检查每个结果 x 是否与期望的结果 expect 接近
        assert_allclose(x, expect)
class TestIRFFTN:

    def test_not_last_axis_success(self, xp):
        # 生成两个形状为 (2, 16, 8, 32) 的随机实部和虚部数组
        ar, ai = np.random.random((2, 16, 8, 32))
        # 将实部和虚部数组合成复数数组
        a = ar + 1j * ai
        # 将数组转换为 xp （可以是 numpy 或 cupy）的数组
        a = xp.asarray(a)

        # 定义需要进行傅里叶反变换的轴
        axes = (-2,)

        # 调用傅里叶反变换函数，确保不会引发错误
        fft.irfftn(a, axes=axes)


@pytest.mark.parametrize("func", [fft.fft, fft.ifft, fft.rfft, fft.irfft,
                                  fft.fftn, fft.ifftn,
                                  fft.rfftn, fft.irfftn, fft.hfft, fft.ihfft])
def test_non_standard_params(func, xp):
    # 根据函数类型确定要使用的数据类型
    if func in [fft.rfft, fft.rfftn, fft.ihfft]:
        dtype = xp.float64
    else:
        dtype = xp.complex128

    # 如果 xp 不是 numpy，创建一个包含 [1, 2, 3] 的数组，指定数据类型为 dtype
    if xp.__name__ != 'numpy':
        x = xp.asarray([1, 2, 3], dtype=dtype)
        # 调用函数 func(x)，确保不会引发异常
        func(x)
        # 断言调用 func(x, workers=2) 时会引发 ValueError 异常
        assert_raises(ValueError, func, x, workers=2)
        # 'plan' 参数目前未被测试，因为 SciPy 目前没有使用它，但如果将来使用，应该进行测试
        # 但如果将来使用，应该进行测试
```