# `.\pytorch\test\test_spectral_ops.py`

```
# Owner(s): ["module: fft"]

import torch  # 导入PyTorch库
import unittest  # 导入unittest用于单元测试
import math  # 导入数学函数
from contextlib import contextmanager  # 导入上下文管理器
from itertools import product  # 导入product函数用于迭代器
import itertools  # 导入itertools模块
import doctest  # 导入doctest模块用于文档测试
import inspect  # 导入inspect模块用于获取对象信息

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_NUMPY, TEST_LIBROSA, TEST_MKL, first_sample, TEST_WITH_ROCM,
     make_tensor, skipIfTorchDynamo)  # 导入测试工具函数和标志
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, onlyNativeDeviceTypes,
     skipCPUIfNoFFT, deviceCountAtLeast, onlyCUDA, OpDTypes, skipIf, toleranceOverride, tol)  # 导入设备相关测试函数
from torch.testing._internal.common_methods_invocations import (
    spectral_funcs, SpectralFuncType)  # 导入频谱函数和类型

from torch.testing._internal.common_cuda import SM53OrLater  # 导入CUDA相关测试
from torch._prims_common import corresponding_complex_dtype  # 导入复数数据类型匹配函数

from typing import Optional, List  # 导入类型提示

from packaging import version  # 导入版本控制模块


if TEST_NUMPY:
    import numpy as np  # 如果测试标志为True，导入numpy库


if TEST_LIBROSA:
    import librosa  # 如果测试标志为True，导入librosa库

has_scipy_fft = False  # 初始化是否有scipy.fft标志为False
try:
    import scipy.fft  # 尝试导入scipy.fft模块
    has_scipy_fft = True  # 如果导入成功，将标志设置为True
except ModuleNotFoundError:
    pass  # 如果导入失败，忽略该错误

# 根据numpy和scipy的版本选择参考的归一化模式
REFERENCE_NORM_MODES = (
    (None, "forward", "backward", "ortho")  # 若numpy版本 >= 1.20.0且scipy版本 >= 1.6.0，选择四种模式
    if version.parse(np.__version__) >= version.parse('1.20.0') and (
        not has_scipy_fft or version.parse(scipy.__version__) >= version.parse('1.6.0'))
    else (None, "ortho"))  # 否则选择两种模式

def _complex_stft(x, *args, **kwargs):
    # 分别对实部和虚部进行变换
    stft_real = torch.stft(x.real, *args, **kwargs, return_complex=True, onesided=False)
    stft_imag = torch.stft(x.imag, *args, **kwargs, return_complex=True, onesided=False)
    return stft_real + 1j * stft_imag  # 返回复数的STFT结果

def _hermitian_conj(x, dim):
    """返回沿单个维度的共轭转置

    H(x)[i] = conj(x[-i])
    """
    out = torch.empty_like(x)  # 创建一个和输入张量x相同大小的空张量out
    mid = (x.size(dim) - 1) // 2  # 计算维度dim的中间索引
    idx = [slice(None)] * out.dim()  # 创建索引列表，长度为out的维数
    idx_center = list(idx)  # 复制索引列表
    idx_center[dim] = 0  # 设置中心索引为0
    out[idx] = x[idx]  # 复制x的数据到out

    idx_neg = list(idx)  # 复制索引列表
    idx_neg[dim] = slice(-mid, None)  # 设置负索引范围
    idx_pos = idx  # 复制索引列表
    idx_pos[dim] = slice(1, mid + 1)  # 设置正索引范围

    out[idx_pos] = x[idx_neg].flip(dim)  # 对负索引部分进行翻转赋值给正索引部分
    out[idx_neg] = x[idx_pos].flip(dim)  # 对正索引部分进行翻转赋值给负索引部分
    if (2 * mid + 1 < x.size(dim)):
        idx[dim] = mid + 1  # 设置中心索引
        out[idx] = x[idx]  # 复制中心索引数据到out
    return out.conj()  # 返回out的共轭

def _complex_istft(x, *args, **kwargs):
    # 将复数STFT分解为Hermitian（实部FFT）和anti-Hermitian（虚部FFT）
    n_fft = x.size(-2)  # 获取FFT长度
    slc = (Ellipsis, slice(None, n_fft // 2 + 1), slice(None))  # 设置切片范围

    hconj = _hermitian_conj(x, dim=-2)  # 沿着dim=-2维度计算共轭转置
    x_hermitian = (x + hconj) / 2  # 计算Hermitian部分
    x_antihermitian = (x - hconj) / 2  # 计算anti-Hermitian部分
    istft_real = torch.istft(x_hermitian[slc], *args, **kwargs, onesided=True)  # 计算实部逆STFT
    istft_imag = torch.istft(-1j * x_antihermitian[slc], *args, **kwargs, onesided=True)  # 计算虚部逆STFT
    return torch.complex(istft_real, istft_imag)  # 返回复数结果

def _stft_reference(x, hop_length, window):
    r"""参考STFT实现

    这只实现了STFT的定义，而不是torch.stft的全部功能：
    ```
    """
    计算短时傅里叶变换（STFT）的频谱表示。
    
    :param x: 输入信号的一维张量
    :param window: 窗函数的一维张量
    :param hop_length: 帧移（采样点数），整数
    :return: STFT的频谱表示，二维张量
    
    STFT的计算公式为 X(m, omega) = sum_n x[n] * w[n - m] * exp(-j*n*omega)
    
    其中:
    - x: 输入信号
    - w: 窗函数
    - m: 第m个窗口（帧）的索引
    - omega: 频率角频率
    - X(m, omega): 在第m个窗口下频谱的表示
    """
    
    n_fft = window.numel()  # 窗函数长度，即FFT的长度
    X = torch.empty((n_fft, (x.numel() - n_fft + hop_length) // hop_length),
                    device=x.device, dtype=torch.cdouble)  # 初始化存储STFT结果的张量
    
    # 遍历每个窗口（帧），计算其STFT
    for m in range(X.size(1)):
        start = m * hop_length  # 计算当前帧的起始位置
        if start + n_fft > x.numel():
            slc = torch.empty(n_fft, device=x.device, dtype=x.dtype)  # 创建一个空张量用于存储当前帧数据
            tmp = x[start:]  # 截取剩余部分的信号数据
            slc[:tmp.numel()] = tmp  # 将截取到的信号数据复制到空张量中
        else:
            slc = x[start: start + n_fft]  # 直接取出当前帧的信号数据
    
        X[:, m] = torch.fft.fft(slc * window)  # 计算当前帧的FFT，并存储到结果张量中
    
    return X  # 返回计算得到的STFT频谱表示
# 定义一个辅助函数用于跳过 FFT 相关测试
def skip_helper_for_fft(device, dtype):
    # 获取设备类型
    device_type = torch.device(device).type
    # 如果数据类型不是 torch.half 或 torch.complex32，则直接返回
    if dtype not in (torch.half, torch.complex32):
        return

    # 如果设备类型是 'cpu'，则抛出跳过测试的异常，因为 half 和 complex32 不支持在 CPU 上运行
    if device_type == 'cpu':
        raise unittest.SkipTest("half and complex32 are not supported on CPU")
    # 如果不满足 SM53OrLater 条件，抛出跳过测试的异常，因为 half 和 complex32 只支持 CUDA 设备且 SM > 53
    if not SM53OrLater:
        raise unittest.SkipTest("half and complex32 are only supported on CUDA device with SM>53")


# 在 torch.fft 命名空间中测试傅里叶分析相关函数
class TestFFT(TestCase):
    exact_dtype = True

    # 测试一维频谱函数的引用实现
    @onlyNativeDeviceTypes
    @ops([op for op in spectral_funcs if op.ndimensional == SpectralFuncType.OneD],
         allowed_dtypes=(torch.float, torch.cfloat))
    def test_reference_1d(self, device, dtype, op):
        # 如果没有引用实现，则跳过测试
        if op.ref is None:
            raise unittest.SkipTest("No reference implementation")

        # 参考实现的规范化模式
        norm_modes = REFERENCE_NORM_MODES
        # 测试参数列表
        test_args = [
            *product(
                # input
                (torch.randn(67, device=device, dtype=dtype),
                 torch.randn(80, device=device, dtype=dtype),
                 torch.randn(12, 14, device=device, dtype=dtype),
                 torch.randn(9, 6, 3, device=device, dtype=dtype)),
                # n
                (None, 50, 6),
                # dim
                (-1, 0),
                # norm
                norm_modes
            ),
            # 测试转换多维张量的中间维度
            *product(
                (torch.randn(4, 5, 6, 7, device=device, dtype=dtype),),
                (None,),
                (1, 2, -2,),
                norm_modes
            )
        ]

        # 遍历测试参数
        for iargs in test_args:
            args = list(iargs)
            # 获取输入张量
            input = args[0]
            args = args[1:]

            # 获取预期结果
            expected = op.ref(input.cpu().numpy(), *args)
            # 确定是否精确匹配数据类型
            exact_dtype = dtype in (torch.double, torch.complex128)
            # 获取实际结果
            actual = op(input, *args)
            # 断言实际结果与预期结果相等
            self.assertEqual(actual, expected, exact_dtype=exact_dtype)

    # 如果没有 FFT 支持则跳过 CPU 测试
    @skipCPUIfNoFFT
    @onlyNativeDeviceTypes
    @toleranceOverride({
        torch.half : tol(1e-2, 1e-2),
        torch.chalf : tol(1e-2, 1e-2),
    })
    @dtypes(torch.half, torch.float, torch.double, torch.complex32, torch.complex64, torch.complex128)
    # 定义测试函数，用于测试 FFT 双向转换的一致性
    def test_fft_round_trip(self, device, dtype):
        # 调用辅助函数，根据设备和数据类型跳过 FFT 测试
        skip_helper_for_fft(device, dtype)

        # 检验 ifft(fft(x)) 是否保持恒等的测试
        if dtype not in (torch.half, torch.complex32):
            # 对于非半精度和复数半精度，设置测试参数列表
            test_args = list(product(
                # input
                (torch.randn(67, device=device, dtype=dtype),
                 torch.randn(80, device=device, dtype=dtype),
                 torch.randn(12, 14, device=device, dtype=dtype),
                 torch.randn(9, 6, 3, device=device, dtype=dtype)),
                # dim
                (-1, 0),
                # norm
                (None, "forward", "backward", "ortho")
            ))
        else:
            # 对于半精度和复数半精度，cuFFT 支持 2 的幂作为输入
            test_args = list(product(
                # input
                (torch.randn(64, device=device, dtype=dtype),
                 torch.randn(128, device=device, dtype=dtype),
                 torch.randn(4, 16, device=device, dtype=dtype),
                 torch.randn(8, 6, 2, device=device, dtype=dtype)),
                # dim
                (-1, 0),
                # norm
                (None, "forward", "backward", "ortho")
            ))

        # 定义 FFT 函数列表
        fft_functions = [(torch.fft.fft, torch.fft.ifft)]

        # 如果数据类型不是复数类型，添加实部输入的 FFT 函数
        if not dtype.is_complex:
            # 注意：使用 ihfft 作为 "forward" 变换，避免需要生成真实的半复数输入
            fft_functions += [(torch.fft.rfft, torch.fft.irfft),
                              (torch.fft.ihfft, torch.fft.hfft)]

        # 遍历 FFT 函数组合
        for forward, backward in fft_functions:
            for x, dim, norm in test_args:
                kwargs = {
                    'n': x.size(dim),  # FFT 的数据点数
                    'dim': dim,         # FFT 的维度
                    'norm': norm,       # FFT 的归一化方式
                }

                # 执行 ifft(fft(x)) 得到逆变换后的结果 y
                y = backward(forward(x, **kwargs), **kwargs)

                # 对于半精度输入和复数32位的输出，需要手动提升输入 x 的类型为 complex32
                if x.dtype is torch.half and y.dtype is torch.complex32:
                    x = x.to(torch.complex32)

                # 对于实部输入，ifft(fft(x)) 将转换为复数类型，进行精确的类型匹配断言
                self.assertEqual(x, y, exact_dtype=(
                    forward != torch.fft.fft or x.is_complex()))

    # 注意：对于空输入，NumPy 会抛出 ValueError 异常
    @onlyNativeDeviceTypes
    @ops(spectral_funcs, allowed_dtypes=(torch.half, torch.float, torch.complex32, torch.cfloat))
    # 定义测试空输入的 FFT 操作
    def test_empty_fft(self, device, dtype, op):
        # 创建空的张量 t，用于测试空输入情况
        t = torch.empty(1, 0, device=device, dtype=dtype)
        match = r"Invalid number of data points \([-\d]*\) specified"

        # 使用断言检查运行时异常是否包含匹配的错误信息
        with self.assertRaisesRegex(RuntimeError, match):
            op(t)

    @onlyNativeDeviceTypes
    # 测试空的逆傅立叶变换函数调用
    def test_empty_ifft(self, device):
        # 创建一个空的复数张量
        t = torch.empty(2, 1, device=device, dtype=torch.complex64)
        # 匹配的异常消息模式
        match = r"Invalid number of data points \([-\d]*\) specified"

        # 对于一组逆傅立叶变换函数，逐个进行异常断言
        for f in [torch.fft.irfft, torch.fft.irfft2, torch.fft.irfftn,
                  torch.fft.hfft, torch.fft.hfft2, torch.fft.hfftn]:
            # 使用断言确保调用这些函数会抛出指定的运行时异常
            with self.assertRaisesRegex(RuntimeError, match):
                f(t)

    # 仅对原生设备类型进行测试
    @onlyNativeDeviceTypes
    # 测试傅立叶变换中的无效数据类型
    def test_fft_invalid_dtypes(self, device):
        # 创建一个复数张量，用于测试
        t = torch.randn(64, device=device, dtype=torch.complex128)

        # 使用断言确保调用这些函数会抛出指定的运行时异常
        with self.assertRaisesRegex(RuntimeError, "rfft expects a real input tensor"):
            torch.fft.rfft(t)

        with self.assertRaisesRegex(RuntimeError, "rfftn expects a real-valued input tensor"):
            torch.fft.rfftn(t)

        with self.assertRaisesRegex(RuntimeError, "ihfft expects a real input tensor"):
            torch.fft.ihfft(t)

    # 如果没有 FFT 支持则跳过测试
    @skipCPUIfNoFFT
    # 仅对原生设备类型进行测试
    @onlyNativeDeviceTypes
    # 测试傅立叶变换类型提升
    @dtypes(torch.int8, torch.half, torch.float, torch.double,
            torch.complex32, torch.complex64, torch.complex128)
    def test_fft_type_promotion(self, device, dtype):
        # 调用辅助函数帮助跳过 FFT
        skip_helper_for_fft(device, dtype)

        # 根据数据类型创建相应的测试张量
        if dtype.is_complex or dtype.is_floating_point:
            t = torch.randn(64, device=device, dtype=dtype)
        else:
            t = torch.randint(-2, 2, (64,), device=device, dtype=dtype)

        # 映射不同数据类型到其提升后的复数类型
        PROMOTION_MAP = {
            torch.int8: torch.complex64,
            torch.half: torch.complex32,
            torch.float: torch.complex64,
            torch.double: torch.complex128,
            torch.complex32: torch.complex32,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
        }
        
        # 对测试张量进行傅立叶变换，并断言结果数据类型符合预期
        T = torch.fft.fft(t)
        self.assertEqual(T.dtype, PROMOTION_MAP[dtype])

        # 映射不同数据类型到其实数到复数的提升后的数据类型
        PROMOTION_MAP_C2R = {
            torch.int8: torch.float,
            torch.half: torch.half,
            torch.float: torch.float,
            torch.double: torch.double,
            torch.complex32: torch.half,
            torch.complex64: torch.float,
            torch.complex128: torch.double,
        }

        # 根据数据类型选择不同的输入张量，并断言逆傅立叶变换的结果数据类型符合预期
        if dtype in (torch.half, torch.complex32):
            # cuFFT 仅支持半精度和复杂半精度的大小为 2 的幂次输入
            # 注意：使用 hfft 和默认参数，其中输出大小 n=2*(输入大小-1)，确保逻辑傅立叶变换大小为2的幂次方。
            x = torch.randn(65, device=device, dtype=dtype)
            R = torch.fft.hfft(x)
        else:
            R = torch.fft.hfft(t)
        self.assertEqual(R.dtype, PROMOTION_MAP_C2R[dtype])

        # 如果数据类型不是复数，则映射到实数到复数的提升后的数据类型
        if not dtype.is_complex:
            PROMOTION_MAP_R2C = {
                torch.int8: torch.complex64,
                torch.half: torch.complex32,
                torch.float: torch.complex64,
                torch.double: torch.complex128,
            }
            # 对测试张量进行实数到复数的傅立叶变换，并断言结果数据类型符合预期
            C = torch.fft.rfft(t)
            self.assertEqual(C.dtype, PROMOTION_MAP_R2C[dtype])
    # 使用装饰器限制该测试方法只能在本地设备上运行
    @onlyNativeDeviceTypes
    # 使用装饰器定义操作的特性，包括谱函数和不支持的数据类型
    @ops(spectral_funcs, dtypes=OpDTypes.unsupported,
         allowed_dtypes=[torch.half, torch.bfloat16])
    # 定义测试 FFT 对半精度和 bfloat16 数据类型的错误情况
    def test_fft_half_and_bfloat16_errors(self, device, dtype, op):
        # TODO: Remove torch.half error when complex32 is fully implemented
        # 获取第一个样本作为测试输入
        sample = first_sample(self, op.sample_inputs(device, dtype))
        # 获取设备类型（CPU 或 CUDA）
        device_type = torch.device(device).type
        # 默认错误消息
        default_msg = "Unsupported dtype"
        # 根据条件设置错误消息
        if dtype is torch.half and device_type == 'cuda' and TEST_WITH_ROCM:
            err_msg = default_msg
        elif dtype is torch.half and device_type == 'cuda' and not SM53OrLater:
            err_msg = "cuFFT doesn't support signals of half type with compute capability less than SM_53"
        else:
            err_msg = default_msg
        # 断言运行时异常应包含特定错误消息
        with self.assertRaisesRegex(RuntimeError, err_msg):
            op(sample.input, *sample.args, **sample.kwargs)

    # 使用装饰器限制该测试方法只能在本地设备上运行
    @onlyNativeDeviceTypes
    # 使用装饰器定义操作的特性，包括谱函数和允许的数据类型
    @ops(spectral_funcs, allowed_dtypes=(torch.half, torch.chalf))
    # 定义测试 FFT 对半精度和复半精度非二次幂维度的错误情况
    def test_fft_half_and_chalf_not_power_of_two_error(self, device, dtype, op):
        # 创建指定维度的张量
        t = make_tensor(13, 13, device=device, dtype=dtype)
        # 错误消息：cuFFT 仅支持尺寸为二次幂的维度
        err_msg = "cuFFT only supports dimensions whose sizes are powers of two"
        # 断言运行时异常应包含特定错误消息
        with self.assertRaisesRegex(RuntimeError, err_msg):
            op(t)

        # 根据操作的维度类型设置关键字参数
        if op.ndimensional in (SpectralFuncType.ND, SpectralFuncType.TwoD):
            kwargs = {'s': (12, 12)}
        else:
            kwargs = {'n': 12}

        # 断言运行时异常应包含特定错误消息
        with self.assertRaisesRegex(RuntimeError, err_msg):
            op(t, **kwargs)

    # 定义测试多维 FFT 的参考实现情况
    @onlyNativeDeviceTypes
    # 如果没有安装 NumPy，则跳过该测试
    @unittest.skipIf(not TEST_NUMPY, 'NumPy not found')
    # 使用装饰器定义操作的特性，包括谱函数和支持的数据类型
    @ops([op for op in spectral_funcs if op.ndimensional == SpectralFuncType.ND],
         allowed_dtypes=(torch.cfloat, torch.cdouble))
    def test_reference_nd(self, device, dtype, op):
        # 如果没有参考实现，则跳过该测试
        if op.ref is None:
            raise unittest.SkipTest("No reference implementation")

        # 参考实现的规范化模式
        norm_modes = REFERENCE_NORM_MODES

        # 定义变换描述列表：输入维度、s 参数、dim 参数
        transform_desc = [
            *product(range(2, 5), (None,), (None, (0,), (0, -1))),
            *product(range(2, 5), (None, (4, 10)), (None,)),
            (6, None, None),
            (5, None, (1, 3, 4)),
            (3, None, (1,)),
            (1, None, (0,)),
            (4, (10, 10), None),
            (4, (10, 10), (0, 1))
        ]

        # 遍历变换描述
        for input_ndim, s, dim in transform_desc:
            # 创建指定设备和数据类型的随机张量
            shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
            input = torch.randn(*shape, device=device, dtype=dtype)

            # 遍历规范化模式
            for norm in norm_modes:
                # 获取预期输出（使用 NumPy）
                expected = op.ref(input.cpu().numpy(), s, dim, norm)
                # 确定精确数据类型
                exact_dtype = dtype in (torch.double, torch.complex128)
                # 获取实际输出
                actual = op(input, s, dim, norm)
                # 断言实际输出等于预期输出
                self.assertEqual(actual, expected, exact_dtype=exact_dtype)

    # 使用装饰器跳过没有 FFT 支持的 CPU 平台
    @skipCPUIfNoFFT
    # 使用装饰器限制该测试方法只能在本地设备上运行
    @onlyNativeDeviceTypes
    @toleranceOverride({
        torch.half : tol(1e-2, 1e-2),
        torch.chalf : tol(1e-2, 1e-2),
    })
    @dtypes(torch.half, torch.float, torch.double,
            torch.complex32, torch.complex64, torch.complex128)
    def test_fftn_round_trip(self, device, dtype):
        # 覆盖默认的容差值，针对 torch.half 和 torch.chalf 类型设置容差为 1e-2
        skip_helper_for_fft(device, dtype)

        norm_modes = (None, "forward", "backward", "ortho")

        # 定义变换描述列表，包含不同的输入维度和维度标识
        transform_desc = [
            *product(range(2, 5), (None, (0,), (0, -1))),
            (7, None),
            (5, (1, 3, 4)),
            (3, (1,)),
            (1, 0),
        ]

        # 定义 FFT 相关函数对
        fft_functions = [(torch.fft.fftn, torch.fft.ifftn)]

        # 仅适用于实数的函数
        if not dtype.is_complex:
            # 使用 ihfftn 作为 "forward" 变换以避免生成真实的半复数输入
            fft_functions += [(torch.fft.rfftn, torch.fft.irfftn),
                              (torch.fft.ihfftn, torch.fft.hfftn)]

        # 遍历变换描述列表
        for input_ndim, dim in transform_desc:
            if dtype in (torch.half, torch.complex32):
                # 对于 torch.half 和 torch.complex32，cuFFT 支持 2 的幂次方作为输入形状
                shape = itertools.islice(itertools.cycle((2, 4, 8)), input_ndim)
            else:
                # 对于其他数据类型，使用 4 到 8 的范围作为输入形状
                shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
            x = torch.randn(*shape, device=device, dtype=dtype)

            # 遍历 FFT 函数对和正常化模式
            for (forward, backward), norm in product(fft_functions, norm_modes):
                if isinstance(dim, tuple):
                    s = [x.size(d) for d in dim]
                else:
                    s = x.size() if dim is None else x.size(dim)

                kwargs = {'s': s, 'dim': dim, 'norm': norm}
                y = backward(forward(x, **kwargs), **kwargs)

                # 对于实数输入，ifftn(fftn(x)) 将转换为复数
                if x.dtype is torch.half and y.dtype is torch.chalf:
                    # 由于类型提升目前不能与 complex32 一起工作，手动将 `x` 提升为 complex32
                    self.assertEqual(x.to(torch.chalf), y)
                else:
                    # 断言 x 和 y 相等，考虑到是否完全匹配数据类型
                    self.assertEqual(x, y, exact_dtype=(
                        forward != torch.fft.fftn or x.is_complex()))

    @onlyNativeDeviceTypes
    @ops([op for op in spectral_funcs if op.ndimensional == SpectralFuncType.ND],
         allowed_dtypes=[torch.float, torch.cfloat])
    # 定义一个测试函数，测试在非法参数情况下的 FFT 变换操作
    def test_fftn_invalid(self, device, dtype, op):
        # 创建一个形状为 (10, 10, 10) 的随机张量 a，指定设备和数据类型
        a = torch.rand(10, 10, 10, device=device, dtype=dtype)
        # 设置错误信息，用于异常断言
        errMsg = "dims must be unique"
        # 检查在指定维度上重复的维度参数是否会触发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, errMsg):
            op(a, dim=(0, 1, 0))

        # 检查负数索引和正数索引混用是否会触发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, errMsg):
            op(a, dim=(2, -1))

        # 检查维度参数和形状参数长度不匹配是否会触发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "dim and shape .* same length"):
            op(a, s=(1,), dim=(0, 1))

        # 检查超出张量维度范围的维度参数是否会触发 IndexError 异常
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            op(a, dim=(3,))

        # 检查张量维度不符合预期的形状参数是否会触发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "tensor only has 3 dimensions"):
            op(a, s=(10, 10, 10, 10))

    @skipCPUIfNoFFT
    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 定义一个测试函数，测试 FFT 变换在不同情况下的正确性
    def test_fftn_noop_transform(self, device, dtype):
        # 跳过不支持 FFT 的设备和数据类型的辅助函数
        skip_helper_for_fft(device, dtype)
        # 设置预期的输出数据类型
        RESULT_TYPE = {
            torch.half: torch.chalf,
            torch.float: torch.cfloat,
            torch.double: torch.cdouble,
        }

        # 遍历不同的 FFT 操作函数
        for op in [
            torch.fft.fftn,
            torch.fft.ifftn,
            torch.fft.fft2,
            torch.fft.ifft2,
        ]:
            # 创建一个形状为 (10, 10) 的输入张量 inp
            inp = make_tensor((10, 10), device=device, dtype=dtype)
            # 执行 FFT 操作，使用空维度列表 dim=[]
            out = torch.fft.fftn(inp, dim=[])

            # 获取预期的数据类型，如果当前数据类型不在 RESULT_TYPE 中，则保持不变
            expect_dtype = RESULT_TYPE.get(inp.dtype, inp.dtype)
            # 将输入张量 inp 转换为预期的数据类型 expect_dtype
            expect = inp.to(expect_dtype)
            # 断言预期输出和实际输出是否相等
            self.assertEqual(expect, out)

    @skipCPUIfNoFFT
    @onlyNativeDeviceTypes
    @toleranceOverride({
        torch.half : tol(1e-2, 1e-2),
    })
    @dtypes(torch.half, torch.float, torch.double)
    # 定义一个测试函数，测试半傅立叶变换的正确性
    def test_hfftn(self, device, dtype):
        # 跳过不支持 FFT 的设备和数据类型的辅助函数
        skip_helper_for_fft(device, dtype)

        # 定义变换描述列表，包含不同的输入维度和维度参数
        transform_desc = [
            *product(range(2, 5), (None, (0,), (0, -1))),
            (6, None),
            (5, (1, 3, 4)),
            (3, (1,)),
            (1, (0,)),
            (4, (0, 1))
        ]

        # 遍历变换描述列表
        for input_ndim, dim in transform_desc:
            # 如果数据类型是 torch.half，使用循环生成固定形状为 (2, 4, 8) 的形状
            if dtype is torch.half:
                shape = tuple(itertools.islice(itertools.cycle((2, 4, 8)), input_ndim))
            else:
                # 否则使用循环生成范围为 (4, 9) 的形状
                shape = tuple(itertools.islice(itertools.cycle(range(4, 9)), input_ndim))
            # 创建预期的随机张量 expect，指定设备和数据类型
            expect = torch.randn(*shape, device=device, dtype=dtype)
            # 执行逆 FFT 变换，指定维度参数 dim 和归一化参数 norm="ortho"
            input = torch.fft.ifftn(expect, dim=dim, norm="ortho")

            # 获取最后一个维度和其大小
            lastdim = actual_dims[-1]
            lastdim_size = input.size(lastdim) // 2 + 1
            # 创建切片索引，截取逆 FFT 变换结果的部分数据
            idx = [slice(None)] * input_ndim
            idx[lastdim] = slice(0, lastdim_size)
            input = input[idx]

            # 获取实际数据的形状 s，根据实际维度 actual_dims
            s = [shape[dim] for dim in actual_dims]
            # 执行半傅立叶变换，指定形状参数 s、维度参数 dim 和归一化参数 norm="ortho"
            actual = torch.fft.hfftn(input, s=s, dim=dim, norm="ortho")

            # 断言预期输出和实际输出是否相等
            self.assertEqual(expect, actual)

    @skipCPUIfNoFFT
    @onlyNativeDeviceTypes
    # 使用 toleranceOverride 装饰器，设置 torch.half 数据类型的容差为 1e-2
    @toleranceOverride({
        torch.half : tol(1e-2, 1e-2),
    })
    # 使用 dtypes 装饰器，声明测试函数接受 torch.half、torch.float 和 torch.double 数据类型
    @dtypes(torch.half, torch.float, torch.double)
    # 定义测试函数 test_ihfftn，接受 self, device, dtype 作为参数
    def test_ihfftn(self, device, dtype):
        # 调用 skip_helper_for_fft 函数，跳过 FFT 相关的辅助函数
        skip_helper_for_fft(device, dtype)

        # 定义变换描述列表，包含各种输入维度和维度参数的组合
        transform_desc = [
            *product(range(2, 5), (None, (0,), (0, -1))),
            (6, None),
            (5, (1, 3, 4)),
            (3, (1,)),
            (1, (0,)),
            (4, (0, 1))
        ]

        # 遍历 transform_desc 中的每个元素，依次为 input_ndim 和 dim
        for input_ndim, dim in transform_desc:
            # 根据 dtype 类型选择不同的 shape
            if dtype is torch.half:
                shape = tuple(itertools.islice(itertools.cycle((2, 4, 8)), input_ndim))
            else:
                shape = tuple(itertools.islice(itertools.cycle(range(4, 9)), input_ndim))

            # 生成随机输入 tensor，使用给定的 shape、device 和 dtype
            input = torch.randn(*shape, device=device, dtype=dtype)
            # 计算预期输出，使用 torch.fft.ifftn 函数进行逆傅里叶变换
            expect = torch.fft.ifftn(input, dim=dim, norm="ortho")

            # 截取半对称分量
            lastdim = -1 if dim is None else dim[-1]
            lastdim_size = expect.size(lastdim) // 2 + 1
            idx = [slice(None)] * input_ndim
            idx[lastdim] = slice(0, lastdim_size)
            expect = expect[idx]

            # 计算实际输出，使用 torch.fft.ihfftn 函数进行逆高维傅里叶变换
            actual = torch.fft.ihfftn(input, dim=dim, norm="ortho")
            # 断言预期输出和实际输出相等
            self.assertEqual(expect, actual)


    # 2d-fft tests

    # 注意：2d 变换仅是 n 维变换的简单包装，因此不需要详尽测试。

    # 使用 skipCPUIfNoFFT 装饰器，如果没有 FFT 支持则跳过测试
    @skipCPUIfNoFFT
    # 使用 onlyNativeDeviceTypes 装饰器，仅在原生设备类型上运行测试
    @onlyNativeDeviceTypes
    # 使用 dtypes 装饰器，声明测试函数接受 torch.double 和 torch.complex128 数据类型
    @dtypes(torch.double, torch.complex128)
    # 定义一个测试函数，用于测试 FFT 和逆 FFT 相关函数在不同输入条件下的输出
    def test_fft2_numpy(self, device, dtype):
        # 使用的规范化模式
        norm_modes = REFERENCE_NORM_MODES

        # 定义变换描述列表，包含不同的输入维度和可选的形状参数
        transform_desc = [
            *product(range(2, 5), (None, (4, 10))),
        ]

        # 支持的 FFT 函数列表
        fft_functions = ['fft2', 'ifft2', 'irfft2', 'hfft2']
        # 如果数据类型是浮点数，还添加实数 FFT 函数
        if dtype.is_floating_point:
            fft_functions += ['rfft2', 'ihfft2']

        # 遍历变换描述列表
        for input_ndim, s in transform_desc:
            # 使用 itertools.cycle 创建一个循环的维度形状迭代器
            shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
            # 生成随机输入张量
            input = torch.randn(*shape, device=device, dtype=dtype)
            # 遍历 FFT 函数和规范化模式的组合
            for fname, norm in product(fft_functions, norm_modes):
                # 获取 torch 中对应的 FFT 函数
                torch_fn = getattr(torch.fft, fname)
                # 如果函数名中包含 'hfft'，并且没有安装 scipy，则跳过该函数的测试
                if "hfft" in fname:
                    if not has_scipy_fft:
                        continue  # 需要 scipy 才能进行比较
                    # 否则获取 scipy 中对应的 FFT 函数
                    numpy_fn = getattr(scipy.fft, fname)
                else:
                    # 获取 numpy 中对应的 FFT 函数
                    numpy_fn = getattr(np.fft, fname)

                # 定义一个函数，用于在 torch 中进行 FFT 操作，支持可选的维度和规范化模式
                def fn(t: torch.Tensor, s: Optional[List[int]], dim: List[int] = (-2, -1), norm: Optional[str] = None):
                    return torch_fn(t, s, dim, norm)

                # 创建包含 torch 函数和其 JIT 脚本版本的元组
                torch_fns = (torch_fn, torch.jit.script(fn))

                # 使用默认维度进行一次测试
                input_np = input.cpu().numpy()
                expected = numpy_fn(input_np, s, norm=norm)
                # 对每个 torch 函数执行 FFT 操作，并断言结果与 numpy 的预期结果相同
                for fn in torch_fns:
                    actual = fn(input, s, norm=norm)
                    self.assertEqual(actual, expected)

                # 使用显式指定的维度进行一次测试
                dim = (1, 0)
                expected = numpy_fn(input_np, s, dim, norm)
                # 对每个 torch 函数执行 FFT 操作，并断言结果与 numpy 的预期结果相同
                for fn in torch_fns:
                    actual = fn(input, s, dim, norm)
                    self.assertEqual(actual, expected)

    # 装饰器：如果当前环境不支持 FFT，跳过测试
    @skipCPUIfNoFFT
    # 装饰器：仅在本地原生设备类型上运行测试
    @onlyNativeDeviceTypes
    # 装饰器：指定数据类型为浮点数或复数 64 位
    @dtypes(torch.float, torch.complex64)
    # 定义测试函数，用于测试 Torch 的 FFT 相关函数的等效性
    def test_fft2_fftn_equivalence(self, device, dtype):
        # 定义归一化模式列表
        norm_modes = (None, "forward", "backward", "ortho")

        # 定义变换描述列表，包括输入维度、s 参数和dim 参数的组合
        transform_desc = [
            *product(range(2, 5), (None, (4, 10)), (None, (1, 0))),
            (3, None, (0, 2)),
        ]

        # 定义 FFT 相关函数列表
        fft_functions = ['fft', 'ifft', 'irfft', 'hfft']
        # 如果数据类型是浮点型，则添加实部函数到 FFT 函数列表中
        if dtype.is_floating_point:
            fft_functions += ['rfft', 'ihfft']

        # 遍历变换描述列表
        for input_ndim, s, dim in transform_desc:
            # 使用 itertools 生成指定长度的形状
            shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
            # 生成指定形状的随机张量 x
            x = torch.randn(*shape, device=device, dtype=dtype)

            # 遍历 FFT 函数列表和归一化模式列表的组合
            for func, norm in product(fft_functions, norm_modes):
                # 获取对应的 2D 和 nD FFT 函数
                f2d = getattr(torch.fft, func + '2')
                fnd = getattr(torch.fft, func + 'n')

                # 设置 FFT 函数的参数字典
                kwargs = {'s': s, 'norm': norm}

                # 如果 dim 不为 None，则将 dim 加入参数字典，计算期望结果
                if dim is not None:
                    kwargs['dim'] = dim
                    expect = fnd(x, **kwargs)
                else:
                    # 否则，在 dim=(-2, -1) 的维度上计算期望结果
                    expect = fnd(x, dim=(-2, -1), **kwargs)

                # 计算实际结果
                actual = f2d(x, **kwargs)

                # 断言实际结果等于期望结果
                self.assertEqual(actual, expect)

    # 根据条件跳过 CPU 上没有 FFT 支持的测试
    @skipCPUIfNoFFT
    # 只在本地设备类型上运行的装饰器
    @onlyNativeDeviceTypes
    # 定义测试 FFT2 函数的无效输入情况
    def test_fft2_invalid(self, device):
        # 生成指定设备上形状为 (10, 10, 10) 的随机张量 a
        a = torch.rand(10, 10, 10, device=device)
        # 定义 FFT2 函数列表
        fft_funcs = (torch.fft.fft2, torch.fft.ifft2,
                     torch.fft.rfft2, torch.fft.irfft2)

        # 遍历 FFT2 函数列表
        for func in fft_funcs:
            # 使用断言检查在重复维度情况下是否抛出 RuntimeError
            with self.assertRaisesRegex(RuntimeError, "dims must be unique"):
                func(a, dim=(0, 0))

            # 使用断言检查在非唯一维度情况下是否抛出 RuntimeError
            with self.assertRaisesRegex(RuntimeError, "dims must be unique"):
                func(a, dim=(2, -1))

            # 使用断言检查在维度和形状长度不同情况下是否抛出 RuntimeError
            with self.assertRaisesRegex(RuntimeError, "dim and shape .* same length"):
                func(a, s=(1,))

            # 使用断言检查维度超出范围时是否抛出 IndexError
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                func(a, dim=(2, 3))

        # 创建复数张量 c，并使用断言检查输入为复数时是否抛出 RuntimeError
        c = torch.complex(a, a)
        with self.assertRaisesRegex(RuntimeError, "rfftn expects a real-valued input"):
            torch.fft.rfft2(c)

    # 辅助函数

    # 根据条件跳过 CPU 上没有 FFT 支持的测试
    @skipCPUIfNoFFT
    # 只在本地设备类型上运行的装饰器
    @onlyNativeDeviceTypes
    # 如果没有安装 NumPy，则跳过该测试
    @unittest.skipIf(not TEST_NUMPY, 'NumPy not found')
    # 定义测试 Torch FFT 函数和 NumPy FFT 函数的结果等效性
    @dtypes(torch.float, torch.double)
    def test_fftfreq_numpy(self, device, dtype):
        # 定义测试参数列表，包括 n 和 d 的组合
        test_args = [
            *product(
                # n 的取值范围为 1 到 19
                range(1, 20),
                # d 的取值为 None 或者 10.0
                (None, 10.0),
            )
        ]

        # 定义 FFT 函数列表
        functions = ['fftfreq', 'rfftfreq']

        # 遍历 FFT 函数列表
        for fname in functions:
            # 获取 Torch FFT 函数和 NumPy FFT 函数
            torch_fn = getattr(torch.fft, fname)
            numpy_fn = getattr(np.fft, fname)

            # 遍历测试参数列表
            for n, d in test_args:
                args = (n,) if d is None else (n, d)
                # 计算 NumPy 的期望结果
                expected = numpy_fn(*args)
                # 计算 Torch 的实际结果
                actual = torch_fn(*args, device=device, dtype=dtype)
                # 断言 Torch 的实际结果等于 NumPy 的期望结果，允许数据类型不精确匹配
                self.assertEqual(actual, expected, exact_dtype=False)
    # 定义测试函数，测试 torch.fft.fftfreq 和 torch.fft.rfftfreq 函数的输出
    def test_fftfreq_out(self, device, dtype):
        # 对于每个函数 torch.fft.fftfreq 和 torch.fft.rfftfreq
        for func in (torch.fft.fftfreq, torch.fft.rfftfreq):
            # 使用指定的参数调用函数，获取期望的输出
            expect = func(n=100, d=.5, device=device, dtype=dtype)
            # 创建一个空的张量，用于实际的输出
            actual = torch.empty((), device=device, dtype=dtype)
            # 断言调用函数时会发出 UserWarning，提醒输出张量将被重新调整大小
            with self.assertWarnsRegex(UserWarning, "out tensor will be resized"):
                func(n=100, d=.5, out=actual)
            # 断言实际输出与期望输出相等
            self.assertEqual(actual, expect)

    # 装饰器：如果没有 FFT 支持则跳过测试
    # 装饰器：仅在本地设备类型下运行测试
    # 装饰器：如果未安装 NumPy，则跳过测试
    # 装饰器：指定测试数据类型为 torch.float, torch.double, torch.complex64, torch.complex128
    def test_fftshift_numpy(self, device, dtype):
        # 定义测试参数列表，包括不同的形状和维度
        test_args = [
            # shape, dim
            *product(((11,), (12,)), (None, 0, -1)),
            *product(((4, 5), (6, 6)), (None, 0, (-1,))),
            *product(((1, 1, 4, 6, 7, 2),), (None, (3, 4))),
        ]

        # 定义需要测试的函数列表
        functions = ['fftshift', 'ifftshift']

        # 遍历测试参数
        for shape, dim in test_args:
            # 创建随机输入张量，指定设备和数据类型
            input = torch.rand(*shape, device=device, dtype=dtype)
            # 将输入张量转换为 NumPy 数组
            input_np = input.cpu().numpy()

            # 遍历需要测试的函数名称
            for fname in functions:
                # 获取 torch.fft 和 np.fft 对应的函数
                torch_fn = getattr(torch.fft, fname)
                numpy_fn = getattr(np.fft, fname)

                # 使用 NumPy 函数计算期望输出
                expected = numpy_fn(input_np, axes=dim)
                # 使用 torch.fft 函数计算实际输出
                actual = torch_fn(input, dim=dim)
                # 断言实际输出与期望输出相等
                self.assertEqual(actual, expected)

    # 装饰器：如果没有 FFT 支持则跳过测试
    # 装饰器：仅在本地设备类型下运行测试
    # 装饰器：如果未安装 NumPy，则跳过测试
    # 装饰器：指定测试数据类型为 torch.float, torch.double
    def test_fftshift_frequencies(self, device, dtype):
        # 遍历测试范围内的 n 值
        for n in range(10, 15):
            # 创建排序后的 FFT 频率张量
            sorted_fft_freqs = torch.arange(-(n // 2), n - (n // 2),
                                            device=device, dtype=dtype)
            # 使用 torch.fft.fftfreq 函数计算 FFT 频率
            x = torch.fft.fftfreq(n, d=1 / n, device=device, dtype=dtype)

            # 测试 fftshift 是否对 fftfreq 输出进行了排序
            shifted = torch.fft.fftshift(x)
            self.assertEqual(shifted, shifted.sort().values)
            self.assertEqual(sorted_fft_freqs, shifted)

            # 测试 ifftshift 是否是 fftshift 的逆操作
            self.assertEqual(x, torch.fft.ifftshift(shifted))

    # Legacy fft tests
    # 定义一个测试函数，用于测试 FFT 和 IFFT 的功能
    def _test_fft_ifft_rfft_irfft(self, device, dtype):
        # 根据给定的数据类型获取对应的复数类型
        complex_dtype = corresponding_complex_dtype(dtype)

        # 测试复数输入的情况
        def _test_complex(sizes, signal_ndim, prepro_fn=lambda x: x):
            # 生成符合指定大小的复数张量
            x = prepro_fn(torch.randn(*sizes, dtype=complex_dtype, device=device))
            # 确定信号的维度范围
            dim = tuple(range(-signal_ndim, 0))
            # 对于正交或非正交情况，执行 FFT 和 IFFT
            for norm in ('ortho', None):
                res = torch.fft.fftn(x, dim=dim, norm=norm)
                rec = torch.fft.ifftn(res, dim=dim, norm=norm)
                # 断言 FFT 和 IFFT 的结果是否与原始输入一致
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='fft and ifft')
                res = torch.fft.ifftn(x, dim=dim, norm=norm)
                rec = torch.fft.fftn(res, dim=dim, norm=norm)
                # 断言 IFFT 和 FFT 的结果是否与原始输入一致
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='ifft and fft')

        # 测试实数输入的情况
        def _test_real(sizes, signal_ndim, prepro_fn=lambda x: x):
            # 生成符合指定大小的实数张量
            x = prepro_fn(torch.randn(*sizes, dtype=dtype, device=device))
            # 确定信号的元素数量和尺寸
            signal_numel = 1
            signal_sizes = x.size()[-signal_ndim:]
            dim = tuple(range(-signal_ndim, 0))
            # 对于正交或非正交情况，执行 RFFT 和 IRFFT
            for norm in (None, 'ortho'):
                res = torch.fft.rfftn(x, dim=dim, norm=norm)
                rec = torch.fft.irfftn(res, s=signal_sizes, dim=dim, norm=norm)
                # 断言 RFFT 和 IRFFT 的结果是否与原始输入一致
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='rfft and irfft')
                res = torch.fft.fftn(x, dim=dim, norm=norm)
                rec = torch.fft.ifftn(res, dim=dim, norm=norm)
                # 将实数张量转换为复数张量，执行 FFT 和 IFFT
                x_complex = torch.complex(x, torch.zeros_like(x))
                self.assertEqual(x_complex, rec, atol=1e-8, rtol=0, msg='fft and ifft (from real)')

        # 测试连续情况
        _test_real((100,), 1)
        _test_real((10, 1, 10, 100), 1)
        _test_real((100, 100), 2)
        _test_real((2, 2, 5, 80, 60), 2)
        _test_real((50, 40, 70), 3)
        _test_real((30, 1, 50, 25, 20), 3)

        _test_complex((100,), 1)
        _test_complex((100, 100), 1)
        _test_complex((100, 100), 2)
        _test_complex((1, 20, 80, 60), 2)
        _test_complex((50, 40, 70), 3)
        _test_complex((6, 5, 50, 25, 20), 3)

        # 测试非连续情况
        _test_real((165,), 1, lambda x: x.narrow(0, 25, 100))  # 输入张量未对齐到复数类型
        _test_real((100, 100, 3), 1, lambda x: x[:, :, 0])
        _test_real((100, 100), 2, lambda x: x.t())
        _test_real((20, 100, 10, 10), 2, lambda x: x.view(20, 100, 100)[:, :60])
        _test_real((65, 80, 115), 3, lambda x: x[10:60, 13:53, 10:80])
        _test_real((30, 20, 50, 25), 3, lambda x: x.transpose(1, 2).transpose(2, 3))

        _test_complex((100,), 1, lambda x: x.expand(100, 100))
        _test_complex((20, 90, 110), 2, lambda x: x[:, 5:85].narrow(2, 5, 100))
        _test_complex((40, 60, 3, 80), 3, lambda x: x.transpose(2, 0).select(0, 2)[5:55, :, 10:])
        _test_complex((30, 55, 50, 22), 3, lambda x: x[:, 3:53, 15:40, 1:21)]

    @skipCPUIfNoFFT
    @onlyNativeDeviceTypes
    @dtypes(torch.double)
    # 调用内部方法 _test_fft_ifft_rfft_irfft，测试 FFT、IFFT、RFFT 和 IRFFT 函数的功能
    def test_fft_ifft_rfft_irfft(self, device, dtype):
        self._test_fft_ifft_rfft_irfft(device, dtype)

    # 设置测试条件：至少有一个 CUDA 设备、数据类型为双精度浮点数或复数的 CUDA 设备
    @deviceCountAtLeast(1)
    @onlyCUDA
    @dtypes(torch.double)
    @onlyCUDA
    @dtypes(torch.cfloat, torch.cdouble)
    def test_cufft_context(self, device, dtype):
        # 用随机生成的数据 x 初始化张量，设备为指定的 CUDA 设备，数据类型为指定的 dtype，并且需要梯度计算
        x = torch.randn(32, dtype=dtype, device=device, requires_grad=True)
        # 创建与 x 形状相同的零张量 dout，设备和数据类型与 x 相同
        dout = torch.zeros(32, dtype=dtype, device=device)

        # 计算 x 的 FFT 后再进行逆 FFT，用于测试是否能够还原数据
        out = torch.fft.ifft(torch.fft.fft(x))
        # 对 out 执行反向传播，使用 dout 作为梯度，保留计算图以供多次反向传播使用
        out.backward(dout, retain_graph=True)

        # 计算 dout 的 FFT，然后再进行逆 FFT，用于比较是否与 x 的梯度匹配
        dx = torch.fft.fft(torch.fft.ifft(dout))

        # 断言：x 的梯度减去 dx 的绝对值最大值应该为 0，用于验证梯度计算的准确性
        self.assertTrue((x.grad - dx).abs().max() == 0)
        # 断言：x 的梯度减去 x 自身的绝对值最大值不应为 0，用于验证梯度不是完全还原的
        self.assertFalse((x.grad - x).abs().max() == 0)

    # 在 ROCm 平台上，使用 Python 2.7 可通过，但在 Python 3.6 下会失败
    @skipIfTorchDynamo("cannot set WRITEABLE flag to True of this array")
    # 跳过不支持 FFT 的 CPU 设备的测试
    @skipCPUIfNoFFT
    # 只在原生设备类型上运行测试
    @onlyNativeDeviceTypes
    @dtypes(torch.double)
    @skipIfTorchDynamo("double")
    # 跳过不支持 FFT 的 CPU 设备的测试
    @skipCPUIfNoFFT
    # 只在原生设备类型上运行测试
    @onlyNativeDeviceTypes
    @dtypes(torch.double)
    # 定义一个测试方法，用于验证自定义的逆短时傅里叶变换（ISTFT）实现与Librosa库的对比
    def test_istft_against_librosa(self, device, dtype):
        # 如果没有安装Librosa库，则跳过测试并抛出跳过测试的异常
        if not TEST_LIBROSA:
            raise unittest.SkipTest('librosa not found')

        # 定义一个使用Librosa库执行ISTFT的辅助函数
        def librosa_istft(x, n_fft, hop_length, win_length, window, length, center):
            # 如果未提供窗口参数，则使用长度为n_fft或win_length的全1窗口
            if window is None:
                window = np.ones(n_fft if win_length is None else win_length)
            else:
                # 否则，将窗口转换为NumPy数组
                window = window.cpu().numpy()

            # 调用Librosa库的ISTFT函数，返回反变换结果
            return librosa.istft(x.cpu().numpy(), n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length, length=length, window=window, center=center)

        # 定义一个测试函数，用于执行ISTFT的测试
        def _test(size, n_fft, hop_length=None, win_length=None, win_sizes=None,
                  length=None, center=True):
            # 生成指定大小和数据类型的随机Tensor数据x
            x = torch.randn(size, dtype=dtype, device=device)
            # 如果指定了窗口大小，生成相应大小和数据类型的随机Tensor窗口数据；否则置为None
            if win_sizes is not None:
                window = torch.randn(*win_sizes, dtype=dtype, device=device)
            else:
                window = None

            # 对输入Tensor x 执行STFT，返回频谱数据x_stft
            x_stft = x.stft(n_fft, hop_length, win_length, window, center=center,
                            onesided=True, return_complex=True)

            # 使用Librosa库执行ISTFT，作为参考结果ref_result
            ref_result = librosa_istft(x_stft, n_fft, hop_length, win_length,
                                       window, length, center)
            # 调用自定义的ISTFT方法，返回计算结果result
            result = x_stft.istft(n_fft, hop_length, win_length, window,
                                  length=length, center=center)
            # 断言自定义ISTFT的结果与Librosa库的结果一致
            self.assertEqual(result, ref_result)

        # 对center参数进行True和False两种情况的测试
        for center in [True, False]:
            _test(10, 7, center=center)
            _test(4000, 1024, center=center)
            _test(4000, 1024, center=center, length=4000)

            _test(10, 7, 2, center=center)
            _test(4000, 1024, 512, center=center)
            _test(4000, 1024, 512, center=center, length=4000)

            _test(10, 7, 2, win_sizes=(7,), center=center)
            _test(4000, 1024, 512, win_sizes=(1024,), center=center)
            _test(4000, 1024, 512, win_sizes=(1024,), center=center, length=4000)
    # 定义一个测试方法，用于测试复杂的短时傅里叶变换（STFT）往返过程
    def test_complex_stft_roundtrip(self, device, dtype):
        # 准备测试参数，通过 product 函数生成所有可能的参数组合
        test_args = list(product(
            # input
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype),
             torch.randn(12, 60, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (None, 10),
            # center
            (True,),
            # pad_mode
            ("constant", "reflect", "circular"),
            # normalized
            (True, False),
            # onesided
            (True, False) if not dtype.is_complex else (False,),
        ))

        # 对于每组参数进行测试
        for args in test_args:
            # 解包参数
            x, n_fft, hop_length, center, pad_mode, normalized, onesided = args
            # 构建通用参数字典
            common_kwargs = {
                'n_fft': n_fft, 'hop_length': hop_length, 'center': center,
                'normalized': normalized, 'onesided': onesided,
            }

            # 使用函数式接口进行短时傅里叶变换
            x_stft = torch.stft(x, pad_mode=pad_mode, return_complex=True, **common_kwargs)
            # 对变换结果进行逆短时傅里叶变换，返回复杂数结果
            x_roundtrip = torch.istft(x_stft, return_complex=dtype.is_complex,
                                      length=x.size(-1), **common_kwargs)
            # 断言逆变换结果与原始输入相等
            self.assertEqual(x_roundtrip, x)

            # 使用张量方法接口进行短时傅里叶变换
            x_stft = x.stft(pad_mode=pad_mode, return_complex=True, **common_kwargs)
            # 对变换结果进行逆短时傅里叶变换，返回复杂数结果
            x_roundtrip = torch.istft(x_stft, return_complex=dtype.is_complex,
                                      length=x.size(-1), **common_kwargs)
            # 断言逆变换结果与原始输入相等
            self.assertEqual(x_roundtrip, x)

    # 应用装饰器，仅在原生设备类型上运行测试
    @onlyNativeDeviceTypes
    # 如果没有 FFT 支持，则跳过 CPU 测试
    @skipCPUIfNoFFT
    # 定义测试类型为双精度浮点数和复双精度复数
    @dtypes(torch.double, torch.cdouble)
    # 测试带有复杂窗口的短时傅里叶变换的往返过程
    def test_stft_roundtrip_complex_window(self, device, dtype):
        # 构建测试参数的笛卡尔积
        test_args = list(product(
            # 输入信号
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype),
             torch.randn(12, 60, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (None, 10),
            # pad_mode
            ("constant", "reflect", "replicate", "circular"),
            # normalized
            (True, False),
        ))

        # 遍历每组参数进行测试
        for args in test_args:
            x, n_fft, hop_length, pad_mode, normalized = args
            # 创建随机复数窗口
            window = torch.rand(n_fft, device=device, dtype=torch.cdouble)
            # 进行短时傅里叶变换
            x_stft = torch.stft(
                x, n_fft=n_fft, hop_length=hop_length, window=window,
                center=True, pad_mode=pad_mode, normalized=normalized)
            # 断言输出的数据类型为复数类型
            self.assertEqual(x_stft.dtype, torch.cdouble)
            # 断言输出的时间窗口大小与 n_fft 相同（非单边）
            self.assertEqual(x_stft.size(-2), n_fft)

            # 进行逆短时傅里叶变换
            x_roundtrip = torch.istft(
                x_stft, n_fft=n_fft, hop_length=hop_length, window=window,
                center=True, normalized=normalized, length=x.size(-1),
                return_complex=True)
            # 再次断言输出的数据类型为复数类型
            self.assertEqual(x_stft.dtype, torch.cdouble)

            # 如果数据类型不是复数类型，进一步断言逆变换的实部和虚部
            if not dtype.is_complex:
                self.assertEqual(x_roundtrip.imag, torch.zeros_like(x_roundtrip.imag),
                                 atol=1e-6, rtol=0)
                self.assertEqual(x_roundtrip.real, x)
            else:
                # 否则，直接断言逆变换的结果与原始输入相同
                self.assertEqual(x_roundtrip, x)


    # 仅在有 FFT 支持的情况下跳过 CPU 测试
    @skipCPUIfNoFFT
    # 使用复数类型作为输入数据类型
    @dtypes(torch.cdouble)
    # 测试复数类型的短时傅里叶变换定义
    def test_complex_stft_definition(self, device, dtype):
        # 构建测试参数的笛卡尔积
        test_args = list(product(
            # 输入信号
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (10, 15)
        ))

        # 遍历每组参数进行测试
        for args in test_args:
            # 创建随机复数窗口
            window = torch.randn(args[1], device=device, dtype=dtype)
            # 调用内部参考函数计算预期输出
            expected = _stft_reference(args[0], args[2], window)
            # 调用 Torch 的短时傅里叶变换计算实际输出
            actual = torch.stft(*args, window=window, center=False)
            # 断言实际输出与预期输出相等
            self.assertEqual(actual, expected)


    # 仅在本地设备类型下进行测试
    @onlyNativeDeviceTypes
    # 仅在有 FFT 支持的情况下跳过 CPU 测试
    @skipCPUIfNoFFT
    # 使用复数类型作为输入数据类型
    @dtypes(torch.cdouble)
    # 定义一个测试函数，用于测试复杂的短时傅里叶变换 (STFT) 是否与实际值等效
    def test_complex_stft_real_equiv(self, device, dtype):
        # 准备测试参数的组合列表
        test_args = list(product(
            # input
            (torch.rand(600, device=device, dtype=dtype),   # 随机生成600个元素的张量
             torch.rand(807, device=device, dtype=dtype),   # 随机生成807个元素的张量
             torch.rand(14, 50, device=device, dtype=dtype),  # 随机生成大小为[14, 50]的张量
             torch.rand(6, 51, device=device, dtype=dtype)),  # 随机生成大小为[6, 51]的张量
            # n_fft
            (50, 27),  # 用于STFT的FFT窗口大小
            # hop_length
            (None, 10),  # STFT中的跳跃长度
            # win_length
            (None, 20),  # STFT中的窗口长度
            # center
            (False, True),  # STFT中是否居中
            # pad_mode
            ("constant", "reflect", "circular"),  # STFT中的填充模式
            # normalized
            (True, False),  # STFT中是否归一化
        ))

        # 遍历测试参数组合
        for args in test_args:
            # 解包参数
            x, n_fft, hop_length, win_length, center, pad_mode, normalized = args
            # 调用自定义函数计算期望值的复杂STFT
            expected = _complex_stft(x, n_fft, hop_length=hop_length,
                                     win_length=win_length, pad_mode=pad_mode,
                                     center=center, normalized=normalized)
            # 调用PyTorch中的STFT计算实际值
            actual = torch.stft(x, n_fft, hop_length=hop_length,
                                win_length=win_length, pad_mode=pad_mode,
                                center=center, normalized=normalized)
            # 使用单元测试断言检查期望值与实际值是否相等
            self.assertEqual(expected, actual)

    # 用于跳过没有FFT支持的CPU测试的装饰器
    @skipCPUIfNoFFT
    # 用于测试复杂的逆短时傅里叶变换 (ISTFT) 是否与实际值等效的函数
    @dtypes(torch.cdouble)
    def test_complex_istft_real_equiv(self, device, dtype):
        # 准备测试参数的组合列表
        test_args = list(product(
            # input
            (torch.rand(40, 20, device=device, dtype=dtype),   # 随机生成大小为[40, 20]的张量
             torch.rand(25, 1, device=device, dtype=dtype),    # 随机生成大小为[25, 1]的张量
             torch.rand(4, 20, 10, device=device, dtype=dtype)),  # 随机生成大小为[4, 20, 10]的张量
            # hop_length
            (None, 10),  # ISTFT中的跳跃长度
            # center
            (False, True),  # ISTFT中是否居中
            # normalized
            (True, False),  # ISTFT中是否归一化
        ))

        # 遍历测试参数组合
        for args in test_args:
            # 解包参数
            x, hop_length, center, normalized = args
            # 获取FFT窗口大小
            n_fft = x.size(-2)
            # 调用自定义函数计算期望值的复杂ISTFT
            expected = _complex_istft(x, n_fft, hop_length=hop_length,
                                      center=center, normalized=normalized)
            # 调用PyTorch中的ISTFT计算实际值
            actual = torch.istft(x, n_fft, hop_length=hop_length,
                                 center=center, normalized=normalized,
                                 return_complex=True)
            # 使用单元测试断言检查期望值与实际值是否相等
            self.assertEqual(expected, actual)

    # 用于跳过没有FFT支持的CPU测试的装饰器
    @skipCPUIfNoFFT
    # 定义一个测试方法，用于测试 stft 函数在复杂输入时不能使用 onesided 参数
    def test_complex_stft_onesided(self, device):
        # 使用 product 函数生成所有可能的数据类型组合
        for x_dtype, window_dtype in product((torch.double, torch.cdouble), repeat=2):
            # 创建随机张量 x 和 window，指定设备和数据类型
            x = torch.rand(100, device=device, dtype=x_dtype)
            window = torch.rand(10, device=device, dtype=window_dtype)

            # 如果 x 或 window 的数据类型是复数
            if x_dtype.is_complex or window_dtype.is_complex:
                # 使用 assertRaisesRegex 检查是否引发 RuntimeError，并匹配 'complex' 字符串
                with self.assertRaisesRegex(RuntimeError, 'complex'):
                    # 调用 x 的 stft 方法，期望引发异常，因为不能使用 onesided 参数
                    x.stft(10, window=window, pad_mode='constant', onesided=True)
            else:
                # 调用 x 的 stft 方法，期望返回复数结果 y
                y = x.stft(10, window=window, pad_mode='constant', onesided=True,
                           return_complex=True)
                # 断言 y 的数据类型为 torch.cdouble
                self.assertEqual(y.dtype, torch.cdouble)
                # 断言 y 的大小为 (6, 51)
                self.assertEqual(y.size(), (6, 51))

        # 创建一个复数类型的张量 x
        x = torch.rand(100, device=device, dtype=torch.cdouble)
        # 使用 assertRaisesRegex 检查是否引发 RuntimeError，并匹配 'complex' 字符串
        with self.assertRaisesRegex(RuntimeError, 'complex'):
            # 调用 x 的 stft 方法，期望引发异常，因为不能使用 onesided 参数
            x.stft(10, pad_mode='constant', onesided=True)

    # stft 目前警告需要使用 return_complex 参数，正在编写升级器
    @onlyNativeDeviceTypes
    @skipCPUIfNoFFT
    def test_stft_requires_complex(self, device):
        # 创建一个长度为 100 的随机张量 x
        x = torch.rand(100)
        # 使用 assertRaisesRegex 检查是否引发 RuntimeError，并匹配 'stft requires the return_complex parameter' 字符串
        with self.assertRaisesRegex(RuntimeError, 'stft requires the return_complex parameter'):
            # 调用 x 的 stft 方法，期望引发异常，因为没有指定 return_complex 参数
            y = x.stft(10, pad_mode='constant')

    # stft 和 istft 目前警告如果未提供 window
    @onlyNativeDeviceTypes
    @skipCPUIfNoFFT
    def test_stft_requires_window(self, device):
        # 创建一个长度为 100 的随机张量 x
        x = torch.rand(100)
        # 使用 assertWarnsOnceRegex 检查是否警告 UserWarning，并匹配 "A window was not provided" 字符串
        with self.assertWarnsOnceRegex(UserWarning, "A window was not provided"):
            # 调用 x 的 stft 方法，期望警告因为没有提供 window 参数
            y = x.stft(10, pad_mode='constant', return_complex=True)

    @onlyNativeDeviceTypes
    @skipCPUIfNoFFT
    def test_istft_requires_window(self, device):
        # 创建一个形状为 (51, 5) 的随机复数张量 stft
        stft = torch.rand((51, 5), dtype=torch.cdouble)
        # 使用 assertWarnsOnceRegex 检查是否警告 UserWarning，并匹配 "A window was not provided" 字符串
        with self.assertWarnsOnceRegex(UserWarning, "A window was not provided"):
            # 调用 torch 的 istft 方法，期望警告因为没有提供 window 参数
            x = torch.istft(stft, n_fft=100, length=100)

    @skipCPUIfNoFFT
    def test_fft_input_modification(self, device):
        # FFT 函数不应修改它们的输入 (gh-34551)

        # 创建一个形状为 (2, 2, 2) 的张量 signal，填充为全 1
        signal = torch.ones((2, 2, 2), device=device)
        # 创建 signal 的一个克隆 signal_copy
        signal_copy = signal.clone()
        # 对 signal 进行 FFT 变换，计算其频谱 spectrum，指定在最后两个维度上进行变换
        spectrum = torch.fft.fftn(signal, dim=(-2, -1))
        # 断言 signal 与 signal_copy 在克隆前后不变
        self.assertEqual(signal, signal_copy)

        # 创建 spectrum 的一个克隆 spectrum_copy
        spectrum_copy = spectrum.clone()
        # 对 spectrum 进行逆 FFT 变换，计算其时域信号
        _ = torch.fft.ifftn(spectrum, dim=(-2, -1))
        # 断言 spectrum 与 spectrum_copy 在克隆前后不变
        self.assertEqual(spectrum, spectrum_copy)

        # 对 signal 进行实部为正的快速傅里叶变换，计算其频谱 half_spectrum
        half_spectrum = torch.fft.rfftn(signal, dim=(-2, -1))
        # 断言 signal 与 signal_copy 在克隆前后不变
        self.assertEqual(signal, signal_copy)

        # 创建 half_spectrum 的一个克隆 half_spectrum_copy
        half_spectrum_copy = half_spectrum.clone()
        # 对 half_spectrum 进行逆实部为正的快速傅里叶变换，计算其时域信号
        _ = torch.fft.irfftn(half_spectrum_copy, s=(2, 2), dim=(-2, -1))
        # 断言 half_spectrum 与 half_spectrum_copy 在克隆前后不变
        self.assertEqual(half_spectrum, half_spectrum_copy)
    # 定义一个测试函数，用于验证 FFT 操作的重复性
    def test_fft_plan_repeatable(self, device):
        # 回归测试 gh-58724 和 gh-63152
        # 分别对不同长度的随机复数张量进行 FFT 及其克隆的 FFT 操作
        for n in [2048, 3199, 5999]:
            # 创建随机复数张量 a，指定设备和数据类型为复数64位浮点数
            a = torch.randn(n, device=device, dtype=torch.complex64)
            # 执行 FFT 操作
            res1 = torch.fft.fftn(a)
            # 克隆张量并执行 FFT 操作
            res2 = torch.fft.fftn(a.clone())
            # 断言两次 FFT 的结果应该相等
            self.assertEqual(res1, res2)

            # 创建随机张量 a，指定设备和数据类型为双精度浮点数
            a = torch.randn(n, device=device, dtype=torch.float64)
            # 执行实数部分的 FFT 操作
            res1 = torch.fft.rfft(a)
            # 克隆张量并执行实数部分的 FFT 操作
            res2 = torch.fft.rfft(a.clone())
            # 断言两次 FFT 的结果应该相等
            self.assertEqual(res1, res2)

    # 用于测试逆短时傅里叶变换的简单情况，期望恢复原始信号
    @onlyNativeDeviceTypes
    @skipCPUIfNoFFT
    @dtypes(torch.double)
    def test_istft_round_trip_simple_cases(self, device, dtype):
        """stft -> istft should recover the original signale"""
        # 定义一个内部测试函数，验证输入信号的短时傅里叶变换和逆变换
        def _test(input, n_fft, length):
            # 执行短时傅里叶变换，返回复数结果
            stft = torch.stft(input, n_fft=n_fft, return_complex=True)
            # 执行逆短时傅里叶变换，指定变换长度
            inverse = torch.istft(stft, n_fft=n_fft, length=length)
            # 断言逆变换结果与原始输入信号应该相等
            self.assertEqual(input, inverse, exact_dtype=True)

        # 对全为 1 的张量进行测试，设备和数据类型由外部传入
        _test(torch.ones(4, dtype=dtype, device=device), 4, 4)
        # 对全为 0 的张量进行测试，设备和数据类型由外部传入
        _test(torch.zeros(4, dtype=dtype, device=device), 4, 4)

    # 仅对本地设备类型进行测试，跳过无 FFT 支持的 CPU 测试
    @onlyNativeDeviceTypes
    @skipCPUIfNoFFT
    @dtypes(torch.double)
    def test_istft_round_trip_various_params(self, device, dtype):
        """stft -> istft should recover the original signal"""

        def _test_istft_is_inverse_of_stft(stft_kwargs):
            # 为每个数据大小生成随机音频信号，并进行 stft/istft 操作，验证是否可以重建信号
            data_sizes = [(2, 20), (3, 15), (4, 10)]
            num_trials = 100
            istft_kwargs = stft_kwargs.copy()
            # 删除 istft 需要的 pad_mode 参数
            del istft_kwargs['pad_mode']
            for sizes in data_sizes:
                for i in range(num_trials):
                    # 生成指定大小和类型的随机张量作为原始信号
                    original = torch.randn(*sizes, dtype=dtype, device=device)
                    # 对原始信号进行短时傅里叶变换（STFT），返回复数形式的结果
                    stft = torch.stft(original, return_complex=True, **stft_kwargs)
                    # 对STFT结果进行逆短时傅里叶变换（ISTFT），指定长度为原始信号长度
                    inversed = torch.istft(stft, length=original.size(1), **istft_kwargs)
                    # 使用断言验证 ISTFT 后的信号与原始信号在一定容差内相等
                    self.assertEqual(
                        inversed, original, msg='istft comparison against original',
                        atol=7e-6, rtol=0, exact_dtype=True)

        patterns = [
            # hann_window, centered, normalized, onesided
            {
                'n_fft': 12,
                'hop_length': 4,
                'win_length': 12,
                'window': torch.hann_window(12, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'reflect',
                'normalized': True,
                'onesided': True,
            },
            # hann_window, centered, not normalized, not onesided
            {
                'n_fft': 12,
                'hop_length': 2,
                'win_length': 8,
                'window': torch.hann_window(8, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'reflect',
                'normalized': False,
                'onesided': False,
            },
            # hamming_window, centered, normalized, not onesided
            {
                'n_fft': 15,
                'hop_length': 3,
                'win_length': 11,
                'window': torch.hamming_window(11, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'constant',
                'normalized': True,
                'onesided': False,
            },
            # hamming_window, centered, not normalized, onesided
            # window same size as n_fft
            {
                'n_fft': 5,
                'hop_length': 2,
                'win_length': 5,
                'window': torch.hamming_window(5, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'constant',
                'normalized': False,
                'onesided': True,
            },
        ]
        # 对每个 pattern 调用 _test_istft_is_inverse_of_stft 进行测试
        for i, pattern in enumerate(patterns):
            _test_istft_is_inverse_of_stft(pattern)

    @onlyNativeDeviceTypes
    @skipCPUIfNoFFT
    @dtypes(torch.double)
    @onlyNativeDeviceTypes
    def test_istft_throws(self, device):
        """istft should throw exception for invalid parameters"""
        # 创建一个大小为 (3, 5, 2) 的零张量 stft，设备类型由参数 device 指定
        stft = torch.zeros((3, 5, 2), device=device)
        # 当窗口大小为 1 但跳跃长度为 20 时会有间隙，导致错误抛出
        self.assertRaises(
            RuntimeError, torch.istft, stft, n_fft=4,
            hop_length=20, win_length=1, window=torch.ones(1))
        # 创建一个大小为 4 的零张量 invalid_window，设备类型由参数 device 指定
        invalid_window = torch.zeros(4, device=device)
        # 使用全零窗口无法满足 NOLA (Non-zero Overlap Add)
        self.assertRaises(
            RuntimeError, torch.istft, stft, n_fft=4, win_length=4, window=invalid_window)
        # 输入张量不可为空
        self.assertRaises(RuntimeError, torch.istft, torch.zeros((3, 0, 2)), 2)
        self.assertRaises(RuntimeError, torch.istft, torch.zeros((0, 3, 2)), 2)

    @skipIfTorchDynamo("Failed running call_function")
    @onlyNativeDeviceTypes
    @skipCPUIfNoFFT
    @dtypes(torch.double)
    def test_istft_of_sine(self, device, dtype):
        complex_dtype = corresponding_complex_dtype(dtype)

        def _test(amplitude, L, n):
            # 使用设备和数据类型创建长度为 2*L+1 的序列 x
            x = torch.arange(2 * L + 1, device=device, dtype=dtype)
            # 创建原始信号 amplitude*sin(2*pi/L*n*x)
            original = amplitude * torch.sin(2 * math.pi / L * x * n)
            # 创建复数类型的零张量 stft，设备和数据类型由 complex_dtype 指定
            stft = torch.zeros((L // 2 + 1, 2), device=device, dtype=complex_dtype)
            # 计算 stft 中的最大值，用于设置虚部
            stft_largest_val = (amplitude * L) / 2.0
            if n < stft.size(0):
                # 设置 stft[n] 的虚部为负的最大值
                stft[n].imag = torch.tensor(-stft_largest_val, dtype=dtype)

            if 0 <= L - n < stft.size(0):
                # 关于 L // 2 对称设置 stft[L - n] 的虚部为正的最大值
                stft[L - n].imag = torch.tensor(stft_largest_val, dtype=dtype)

            # 使用 istft 函数进行逆短时傅里叶变换
            inverse = torch.istft(
                stft, L, hop_length=L, win_length=L,
                window=torch.ones(L, device=device, dtype=dtype), center=False, normalized=False)
            # 由于振幅缩放，存在较大的误差
            original = original[..., :inverse.size(-1)]
            # 断言逆变换结果与原始信号相等，允许的绝对误差为 1e-3，相对误差为 0
            self.assertEqual(inverse, original, atol=1e-3, rtol=0)

        # 使用不同的参数调用 _test 函数
        _test(amplitude=123, L=5, n=1)
        _test(amplitude=150, L=5, n=2)
        _test(amplitude=111, L=5, n=3)
        _test(amplitude=160, L=7, n=4)
        _test(amplitude=145, L=8, n=5)
        _test(amplitude=80, L=9, n=6)
        _test(amplitude=99, L=10, n=7)

    @onlyNativeDeviceTypes
    @skipCPUIfNoFFT
    @dtypes(torch.double)
    # 定义测试函数，用于测试逆短时傅里叶变换（ISTFT）的线性性
    def test_istft_linearity(self, device, dtype):
        num_trials = 100  # 设定测试的次数

        # 根据给定的数据类型获取对应的复数数据类型
        complex_dtype = corresponding_complex_dtype(dtype)

        # 内部测试函数，针对特定数据大小和参数进行测试
        def _test(data_size, kwargs):
            # 执行多次测试
            for i in range(num_trials):
                # 生成随机的复数张量 tensor1 和 tensor2
                tensor1 = torch.randn(data_size, device=device, dtype=complex_dtype)
                tensor2 = torch.randn(data_size, device=device, dtype=complex_dtype)
                # 随机生成两个标量 a 和 b
                a, b = torch.rand(2, dtype=dtype, device=device)

                # 使用 ISTFT 方法进行重构，使用给定的参数 kwargs
                istft1 = tensor1.istft(**kwargs)
                istft2 = tensor2.istft(**kwargs)

                # 构造线性组合 a * ISTFT(tensor1) + b * ISTFT(tensor2)
                istft = a * istft1 + b * istft2

                # 使用 torch 自带的 ISTFT 函数进行重构，作为预期结果
                estimate = torch.istft(a * tensor1 + b * tensor2, **kwargs)

                # 断言两种重构结果应该非常接近，给定允许的误差
                self.assertEqual(istft, estimate, atol=1e-5, rtol=0)

        # 定义不同的测试模式，包括不同的窗口函数和参数设置
        patterns = [
            # hann_window, centered, normalized, onesided
            (
                (2, 7, 7),
                {
                    'n_fft': 12,
                    'window': torch.hann_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': True,
                    'onesided': True,
                },
            ),
            # hann_window, centered, not normalized, not onesided
            (
                (2, 12, 7),
                {
                    'n_fft': 12,
                    'window': torch.hann_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': False,
                    'onesided': False,
                },
            ),
            # hamming_window, centered, normalized, not onesided
            (
                (2, 12, 7),
                {
                    'n_fft': 12,
                    'window': torch.hamming_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': True,
                    'onesided': False,
                },
            ),
            # hamming_window, not centered, not normalized, onesided
            (
                (2, 7, 3),
                {
                    'n_fft': 12,
                    'window': torch.hamming_window(12, device=device, dtype=dtype),
                    'center': False,
                    'normalized': False,
                    'onesided': True,
                },
            )
        ]

        # 对每种模式依次执行测试
        for data_size, kwargs in patterns:
            _test(data_size, kwargs)

    # 标记仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 如果没有 FFT 支持则跳过测试
    @skipCPUIfNoFFT
    # 定义测试函数，用于批量测试 istft 函数在不同输入条件下的行为
    def test_batch_istft(self, device):
        # 创建原始输入张量，包含复数类型数据
        original = torch.tensor([
            [4., 4., 4., 4., 4.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ], device=device, dtype=torch.complex64)

        # 复制原始张量以创建单一和多重复制的输入张量
        single = original.repeat(1, 1, 1)
        multi = original.repeat(4, 1, 1)

        # 调用 istft 函数对各种输入进行逆短时傅里叶变换（istft），指定参数 n_fft 和 length 都为 4
        i_original = torch.istft(original, n_fft=4, length=4)
        i_single = torch.istft(single, n_fft=4, length=4)
        i_multi = torch.istft(multi, n_fft=4, length=4)

        # 断言单一复制和多重复制的 istft 结果与原始的 istft 结果一致，使用指定的容差值
        self.assertEqual(i_original.repeat(1, 1), i_single, atol=1e-6, rtol=0, exact_dtype=True)
        self.assertEqual(i_original.repeat(4, 1), i_multi, atol=1e-6, rtol=0, exact_dtype=True)

    # 使用装饰器 @onlyCUDA 和 @skipIf 条件测试 MKL 是否存在，仅在 CUDA 下执行此测试
    @onlyCUDA
    @skipIf(not TEST_MKL, "Test requires MKL")
    # 测试 stft 和 istft 函数的设备兼容性
    def test_stft_window_device(self, device):
        # 创建随机复数张量 x 和窗口张量 window，均使用指定的设备
        x = torch.randn(1000, dtype=torch.complex64)
        window = torch.randn(100, dtype=torch.complex64)

        # 断言 stft 函数在输入张量 x 和窗口张量 window 不在同一设备时会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "stft input and window must be on the same device"):
            torch.stft(x, n_fft=100, window=window.to(device))

        with self.assertRaisesRegex(RuntimeError, "stft input and window must be on the same device"):
            torch.stft(x.to(device), n_fft=100, window=window)

        # 对输入张量 x 应用 stft 函数，使用给定的 n_fft 和 window 参数创建频谱表示 X
        X = torch.stft(x, n_fft=100, window=window)

        # 断言 istft 函数在频谱表示 X 和窗口张量 window 不在同一设备时会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "istft input and window must be on the same device"):
            torch.istft(X, n_fft=100, window=window.to(device))

        with self.assertRaisesRegex(RuntimeError, "istft input and window must be on the same device"):
            torch.istft(x.to(device), n_fft=100, window=window)
# FFTDocTestFinder 类，用于查找给定对象（通常是 torch.fft 模块）中的文档测试用例
class FFTDocTestFinder:
    '''The default doctest finder doesn't like that function.__module__ doesn't
    match torch.fft. It assumes the functions are leaked imports.
    '''
    
    # 初始化方法，创建一个 doctest 解析器实例
    def __init__(self):
        self.parser = doctest.DocTestParser()

    # 查找并返回给定对象中的文档测试用例
    def find(self, obj, name=None, module=None, globs=None, extraglobs=None):
        # 存储找到的文档测试用例列表
        doctests = []
        
        # 确定对象的模块名称
        modname = name if name is not None else obj.__name__
        # 如果没有指定全局变量字典，设置为空字典
        globs = {} if globs is None else globs

        # 遍历对象的所有公开方法名
        for fname in obj.__all__:
            # 获取方法对象
            func = getattr(obj, fname)
            # 如果方法是可调用的
            if inspect.isroutine(func):
                # 构建方法的全限定名
                qualname = modname + '.' + fname
                # 获取方法的文档字符串
                docstring = inspect.getdoc(func)
                # 如果方法没有文档字符串，继续下一个方法
                if docstring is None:
                    continue

                # 解析方法的文档字符串，生成 doctest 示例列表
                examples = self.parser.get_doctest(
                    docstring, globs=globs, name=fname, filename=None, lineno=None)
                # 将解析得到的 doctest 示例列表添加到结果列表中
                doctests.append(examples)

        # 返回找到的所有文档测试用例列表
        return doctests


# TestFFTDocExamples 类，用于承载 FFT 模块文档测试的测试用例
class TestFFTDocExamples(TestCase):
    pass


# 生成单个文档测试用例的测试方法
def generate_doc_test(doc_test):
    def test(self, device):
        # 断言测试设备为 'cpu'
        self.assertEqual(device, 'cpu')
        # 创建一个 doctest 运行器
        runner = doctest.DocTestRunner()
        # 运行文档测试用例
        runner.run(doc_test)

        # 如果有测试失败，则打印摘要信息并标记测试失败
        if runner.failures != 0:
            runner.summarize()
            self.fail('Doctest failed')

    # 动态为 TestFFTDocExamples 类添加测试方法，方法名以 'test_' 开头并使用文档测试的名称
    setattr(TestFFTDocExamples, 'test_' + doc_test.name, skipCPUIfNoFFT(test))


# 使用 FFTDocTestFinder 查找 torch.fft 模块中的文档测试用例，并生成对应的测试方法
for doc_test in FFTDocTestFinder().find(torch.fft, globs=dict(torch=torch)):
    generate_doc_test(doc_test)


# 实例化 TestFFT 类的设备类型测试
instantiate_device_type_tests(TestFFT, globals())

# 为 TestFFTDocExamples 类实例化设备类型测试，仅适用于 'cpu' 设备
instantiate_device_type_tests(TestFFTDocExamples, globals(), only_for='cpu')

# 如果脚本作为主程序运行，则执行测试
if __name__ == '__main__':
    run_tests()
```