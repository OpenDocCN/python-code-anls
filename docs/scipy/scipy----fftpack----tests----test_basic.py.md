# `D:\src\scipysrc\scipy\scipy\fftpack\tests\test_basic.py`

```
# Created by Pearu Peterson, September 2002

# 从 numpy.testing 模块中导入多个断言函数，用于测试
from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
                           assert_array_almost_equal_nulp, assert_array_less)
# 导入 pytest 库及其 raises 函数作为 assert_raises 的别名
import pytest
from pytest import raises as assert_raises
# 从 scipy.fftpack 模块中导入多个 FFT 相关函数
from scipy.fftpack import ifft, fft, fftn, ifftn, rfft, irfft, fft2

# 从 numpy 模块中导入多个函数和类
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
                   swapaxes, double, cdouble)
# 导入 numpy 模块并用 np 别名表示
import numpy as np
import numpy.fft
# 从 numpy.random 模块中导入 rand 函数
from numpy.random import rand

# 定义 "large" 复合数的 FFT 支持的大小
LARGE_COMPOSITE_SIZES = [
    2**13,
    2**5 * 3**5,
    2**3 * 3**3 * 5**2,
]
# 定义 "small" 复合数的大小
SMALL_COMPOSITE_SIZES = [
    2,
    2*3*5,
    2*2*3*3,
]
# 定义大质数的大小
LARGE_PRIME_SIZES = [
    2011
]
# 定义小质数的大小
SMALL_PRIME_SIZES = [
    29
]

# 定义一个辅助函数，用于测试，检查两个向量在给定容差下的接近程度
def _assert_close_in_norm(x, y, rtol, size, rdt):
    err_msg = f"size: {size}  rdt: {rdt}"
    # 使用 np.linalg.norm 函数检查两个向量的范数差是否小于给定的相对容差 rtol
    assert_array_less(np.linalg.norm(x - y), rtol*np.linalg.norm(x), err_msg)

# 定义一个生成指定大小随机数数组的函数
def random(size):
    return rand(*size)

# 实现直接计算一维离散傅里叶变换的函数
def direct_dft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = -arange(n)*(2j*pi/n)
    # 循环计算 DFT 公式
    for i in range(n):
        y[i] = dot(exp(i*w), x)
    return y

# 实现直接计算一维逆离散傅里叶变换的函数
def direct_idft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = arange(n)*(2j*pi/n)
    # 循环计算 IDFT 公式
    for i in range(n):
        y[i] = dot(exp(i*w), x)/n
    return y

# 实现直接计算多维离散傅里叶变换的函数
def direct_dftn(x):
    x = asarray(x)
    for axis in range(len(x.shape)):
        x = fft(x, axis=axis)
    return x

# 实现直接计算多维逆离散傅里叶变换的函数
def direct_idftn(x):
    x = asarray(x)
    for axis in range(len(x.shape)):
        x = ifft(x, axis=axis)
    return x

# 实现直接计算一维实数傅里叶变换的函数
def direct_rdft(x):
    x = asarray(x)
    n = len(x)
    w = -arange(n)*(2j*pi/n)
    r = zeros(n, dtype=double)
    # 循环计算实数傅里叶变换的公式
    for i in range(n//2+1):
        y = dot(exp(i*w), x)
        if i:
            r[2*i-1] = y.real
            if 2*i < n:
                r[2*i] = y.imag
        else:
            r[0] = y.real
    return r

# 实现直接计算一维实数逆离散傅里叶变换的函数
def direct_irdft(x):
    x = asarray(x)
    n = len(x)
    x1 = zeros(n, dtype=cdouble)
    # 循环计算一维逆实数傅里叶变换的公式
    for i in range(n//2+1):
        if i:
            if 2*i < n:
                x1[i] = x[2*i-1] + 1j*x[2*i]
                x1[n-i] = x[2*i-1] - 1j*x[2*i]
            else:
                x1[i] = x[2*i-1]
        else:
            x1[0] = x[0]
    return direct_idft(x1).real

# 定义一个测试 FFT 的基类
class _TestFFTBase:
    def setup_method(self):
        self.cdt = None
        self.rdt = None
        np.random.seed(1234)

    # 定义测试 FFT 定义的方法
    def test_definition(self):
        # 创建一个复数数组 x 进行测试
        x = np.array([1,2,3,4+1j,1,2,3,4+2j], dtype=self.cdt)
        # 调用 fft 函数计算傅里叶变换 y
        y = fft(x)
        # 断言 y 的数据类型与预期相同
        assert_equal(y.dtype, self.cdt)
        # 使用 assert_array_almost_equal 函数检查 y 和直接计算的结果 y1 是否接近
        y1 = direct_dft(x)
        assert_array_almost_equal(y, y1)
        # 再次创建一个复数数组 x 进行测试
        x = np.array([1,2,3,4+0j,5], dtype=self.cdt)
        # 使用 assert_array_almost_equal 检查 fft 函数与 direct_dft 函数的结果是否接近
        assert_array_almost_equal(fft(x), direct_dft(x))
    # 定义测试函数，用于测试 FFT 函数在给定参数情况下的输出结果
    def test_n_argument_real(self):
        # 创建实数类型的 NumPy 数组 x1 和 x2，使用 self.rdt 指定数据类型
        x1 = np.array([1,2,3,4], dtype=self.rdt)
        x2 = np.array([1,2,3,4], dtype=self.rdt)
        # 对数组 [x1, x2] 进行 FFT 变换，指定 n=4
        y = fft([x1, x2], n=4)
        # 断言 FFT 结果的数据类型为 self.cdt
        assert_equal(y.dtype, self.cdt)
        # 断言 FFT 结果的形状为 (2, 4)
        assert_equal(y.shape, (2, 4))
        # 断言 FFT 结果的第一个元素与直接计算 x1 的 DFT 结果接近
        assert_array_almost_equal(y[0], direct_dft(x1))
        # 断言 FFT 结果的第二个元素与直接计算 x2 的 DFT 结果接近
        assert_array_almost_equal(y[1], direct_dft(x2))

    # 定义复数输入情况下的测试函数
    def _test_n_argument_complex(self):
        # 创建复数类型的 NumPy 数组 x1 和 x2，使用 self.cdt 指定数据类型
        x1 = np.array([1, 2, 3, 4+1j], dtype=self.cdt)
        x2 = np.array([1, 2, 3, 4+1j], dtype=self.cdt)
        # 对数组 [x1, x2] 进行 FFT 变换，指定 n=4
        y = fft([x1, x2], n=4)
        # 断言 FFT 结果的数据类型为 self.cdt
        assert_equal(y.dtype, self.cdt)
        # 断言 FFT 结果的形状为 (2, 4)
        assert_equal(y.shape, (2, 4))
        # 断言 FFT 结果的第一个元素与直接计算 x1 的 DFT 结果接近
        assert_array_almost_equal(y[0], direct_dft(x1))
        # 断言 FFT 结果的第二个元素与直接计算 x2 的 DFT 结果接近
        assert_array_almost_equal(y[1], direct_dft(x2))

    # 定义测试函数，用于测试 FFT 函数在不合法大小输入情况下是否会引发 ValueError 异常
    def test_invalid_sizes(self):
        # 断言调用 FFT 函数时传递空列表会引发 ValueError 异常
        assert_raises(ValueError, fft, [])
        # 断言调用 FFT 函数时传递 [[1,1],[2,2]] 这样的列表会引发 ValueError 异常
        assert_raises(ValueError, fft, [[1, 1], [2, 2]], -5)
class TestDoubleFFT(_TestFFTBase):
    # 定义一个测试类 TestDoubleFFT，继承自 _TestFFTBase 类
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.cdt = np.complex128
        self.rdt = np.float64


class TestSingleFFT(_TestFFTBase):
    # 定义一个测试类 TestSingleFFT，继承自 _TestFFTBase 类
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.cdt = np.complex64
        self.rdt = np.float32

    # 原因字符串，描述了单精度 FFT 实现部分禁用的原因
    reason = ("single-precision FFT implementation is partially disabled, "
              "until accuracy issues with large prime powers are resolved")

    # 标记为预期失败的测试方法，以注释形式存在
    @pytest.mark.xfail(run=False, reason=reason)
    def test_notice(self):
        # 测试方法暂无实际实现
        pass


class TestFloat16FFT:
    # 定义一个测试类 TestFloat16FFT

    def test_1_argument_real(self):
        # 测试方法，测试一维实数数组的 FFT
        x1 = np.array([1, 2, 3, 4], dtype=np.float16)
        y = fft(x1, n=4)
        assert_equal(y.dtype, np.complex64)
        assert_equal(y.shape, (4, ))
        assert_array_almost_equal(y, direct_dft(x1.astype(np.float32)))

    def test_n_argument_real(self):
        # 测试方法，测试多个一维实数数组的 FFT
        x1 = np.array([1, 2, 3, 4], dtype=np.float16)
        x2 = np.array([1, 2, 3, 4], dtype=np.float16)
        y = fft([x1, x2], n=4)
        assert_equal(y.dtype, np.complex64)
        assert_equal(y.shape, (2, 4))
        assert_array_almost_equal(y[0], direct_dft(x1.astype(np.float32)))
        assert_array_almost_equal(y[1], direct_dft(x2.astype(np.float32)))


class _TestIFFTBase:
    # 定义一个基类 _TestIFFTBase

    def setup_method(self):
        # 设置测试方法的初始化操作
        np.random.seed(1234)

    def test_definition(self):
        # 测试方法，测试复数数组的逆 FFT
        x = np.array([1,2,3,4+1j,1,2,3,4+2j], self.cdt)
        y = ifft(x)
        y1 = direct_idft(x)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(y,y1)

        x = np.array([1,2,3,4+0j,5], self.cdt)
        assert_array_almost_equal(ifft(x),direct_idft(x))

    def test_definition_real(self):
        # 测试方法，测试实数数组的逆 FFT
        x = np.array([1,2,3,4,1,2,3,4], self.rdt)
        y = ifft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_idft(x)
        assert_array_almost_equal(y,y1)

        x = np.array([1,2,3,4,5], dtype=self.rdt)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(ifft(x),direct_idft(x))

    def test_random_complex(self):
        # 测试方法，测试随机复数数组的 FFT 和逆 FFT
        for size in [1,51,111,100,200,64,128,256,1024]:
            x = np.random.random([size]).astype(self.cdt)
            x = x + 1j * np.random.random([size]).astype(self.cdt)
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)

    def test_random_real(self):
        # 测试方法，测试随机实数数组的 FFT 和逆 FFT
        for size in [1,51,111,100,200,64,128,256,1024]:
            x = np.random.random([size]).astype(self.rdt)
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)
    # 测试函数，用于验证精度是否满足对于素数大小和非素数大小输入的基本要求
    def test_size_accuracy(self):
        # 如果数据类型为 np.float32，则设置相对误差容差
        if self.rdt == np.float32:
            rtol = 1e-5
        # 如果数据类型为 np.float64，则设置相对误差容差
        elif self.rdt == np.float64:
            rtol = 1e-10

        # 遍历大的组合数和大的素数数目大小
        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            # 设置随机数种子为 1234
            np.random.seed(1234)
            # 创建长度为 size 的随机数组 x，并转换为指定数据类型 self.rdt
            x = np.random.rand(size).astype(self.rdt)
            # 对 x 进行傅里叶逆变换（IFFT），然后再进行傅里叶变换（FFT），并对结果进行检验
            y = ifft(fft(x))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)
            # 对 x 进行傅里叶变换（FFT），然后再进行傅里叶逆变换（IFFT），并对结果进行检验
            y = fft(ifft(x))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)

            # 创建包含虚数部分的长度为 size 的随机数组 x，并转换为复数数据类型 self.cdt
            x = (x + 1j*np.random.rand(size)).astype(self.cdt)
            # 对 x 进行傅里叶逆变换（IFFT），然后再进行傅里叶变换（FFT），并对结果进行检验
            y = ifft(fft(x))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)
            # 对 x 进行傅里叶变换（FFT），然后再进行傅里叶逆变换（IFFT），并对结果进行检验
            y = fft(ifft(x))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)

    # 测试函数，用于验证输入数据大小为无效值时是否会引发 ValueError 异常
    def test_invalid_sizes(self):
        # 断言调用 ifft 函数传入空列表会引发 ValueError 异常
        assert_raises(ValueError, ifft, [])
        # 断言调用 ifft 函数传入二维列表及负数参数会引发 ValueError 异常
        assert_raises(ValueError, ifft, [[1,1],[2,2]], -5)
class TestDoubleIFFT(_TestIFFTBase):
    # TestDoubleIFFT 类，测试双精度的逆快速傅里叶变换
    def setup_method(self):
        # 设置方法，在每个测试方法执行前运行
        self.cdt = np.complex128  # 设置复数类型为 np.complex128
        self.rdt = np.float64     # 设置实数类型为 np.float64


class TestSingleIFFT(_TestIFFTBase):
    # TestSingleIFFT 类，测试单精度的逆快速傅里叶变换
    def setup_method(self):
        # 设置方法，在每个测试方法执行前运行
        self.cdt = np.complex64   # 设置复数类型为 np.complex64
        self.rdt = np.float32     # 设置实数类型为 np.float32


class _TestRFFTBase:
    # _TestRFFTBase 类，用于测试快速傅里叶变换的基础类
    def setup_method(self):
        # 设置方法，在每个测试方法执行前运行
        np.random.seed(1234)  # 设置随机种子为 1234

    def test_definition(self):
        # 测试定义方法，验证快速傅里叶变换的定义
        for t in [[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4, 5]]:
            x = np.array(t, dtype=self.rdt)  # 使用指定的实数类型创建数组 x
            y = rfft(x)                      # 对 x 进行实数快速傅里叶变换
            y1 = direct_rdft(x)              # 使用直接实数傅里叶变换计算结果 y1
            assert_array_almost_equal(y, y1)  # 断言 y 和 y1 几乎相等
            assert_equal(y.dtype, self.rdt)   # 断言 y 的数据类型与设定的实数类型一致

    def test_invalid_sizes(self):
        # 测试无效尺寸的情况
        assert_raises(ValueError, rfft, [])         # 断言对空数组进行 rfft 抛出 ValueError 异常
        assert_raises(ValueError, rfft, [[1,1],[2,2]], -5)  # 断言对多维数组进行 rfft 抛出 ValueError 异常

    # See gh-5790
    class MockSeries:
        # MockSeries 类，模拟一个数据序列
        def __init__(self, data):
            self.data = np.asarray(data)  # 将输入数据转换为 ndarray 并保存在 data 属性中

        def __getattr__(self, item):
            try:
                return getattr(self.data, item)  # 尝试从 data 属性中获取指定的属性
            except AttributeError as e:
                raise AttributeError("'MockSeries' object "
                                      f"has no attribute '{item}'") from e

    def test_non_ndarray_with_dtype(self):
        # 测试非 ndarray 对象并指定数据类型的情况
        x = np.array([1., 2., 3., 4., 5.])  # 创建一个 ndarray 对象 x
        xs = _TestRFFTBase.MockSeries(x)   # 使用 MockSeries 类创建一个对象 xs

        expected = [1, 2, 3, 4, 5]         # 期望的结果列表
        rfft(xs)                           # 对 MockSeries 对象 xs 进行 rfft 操作

        # 数据不应该被改写
        assert_equal(x, expected)          # 断言 x 的值等于期望值 expected
        assert_equal(xs.data, expected)    # 断言 xs.data 的值等于期望值 expected

    def test_complex_input(self):
        # 测试复数输入的情况
        assert_raises(TypeError, rfft, np.arange(4, dtype=np.complex64))  # 断言对复数数组进行 rfft 抛出 TypeError 异常


class TestRFFTDouble(_TestRFFTBase):
    # TestRFFTDouble 类，测试双精度的实数快速傅里叶变换
    def setup_method(self):
        # 设置方法，在每个测试方法执行前运行
        self.cdt = np.complex128  # 设置复数类型为 np.complex128
        self.rdt = np.float64     # 设置实数类型为 np.float64


class TestRFFTSingle(_TestRFFTBase):
    # TestRFFTSingle 类，测试单精度的实数快速傅里叶变换
    def setup_method(self):
        # 设置方法，在每个测试方法执行前运行
        self.cdt = np.complex64   # 设置复数类型为 np.complex64
        self.rdt = np.float32     # 设置实数类型为 np.float32


class _TestIRFFTBase:
    # _TestIRFFTBase 类，用于测试逆快速傅里叶变换的基础类
    def setup_method(self):
        # 设置方法，在每个测试方法执行前运行
        np.random.seed(1234)  # 设置随机种子为 1234

    def test_definition(self):
        # 测试定义方法，验证逆快速傅里叶变换的定义
        x1 = [1,2,3,4,1,2,3,4]            # 输入序列 x1
        x1_1 = [1,2+3j,4+1j,2+3j,4,2-3j,4-1j,2-3j]  # 对应的预期结果序列 x1_1
        x2 = [1,2,3,4,1,2,3,4,5]          # 输入序列 x2
        x2_1 = [1,2+3j,4+1j,2+3j,4+5j,4-5j,2-3j,4-1j,2-3j]  # 对应的预期结果序列 x2_1

        def _test(x, xr):
            # 内部测试函数，对给定输入 x 进行测试
            y = irfft(np.array(x, dtype=self.rdt))  # 对 x 进行逆实数快速傅里叶变换
            y1 = direct_irdft(x)                   # 使用直接逆实数傅里叶变换计算结果 y1
            assert_equal(y.dtype, self.rdt)         # 断言 y 的数据类型与设定的实数类型一致
            assert_array_almost_equal(y, y1, decimal=self.ndec)  # 断言 y 和 y1 几乎相等，给定小数精度 self.ndec
            assert_array_almost_equal(y, ifft(xr), decimal=self.ndec)  # 断言 y 和使用 ifft 计算的结果 xr 几乎相等

        _test(x1, x1_1)  # 对 x1 进行测试
        _test(x2, x2_1)  # 对 x2 进行测试
    # 测试随机生成的实数数组的逆傅里叶变换和傅里叶变换的正确性
    def test_random_real(self):
        # 针对不同大小的数组进行测试
        for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
            # 生成指定大小的随机实数数组，并转换为指定的数据类型
            x = random([size]).astype(self.rdt)
            # 对 x 进行实数逆傅里叶变换，再进行实数傅里叶变换得到 y1
            y1 = irfft(rfft(x))
            # 对 x 先进行实数傅里叶变换，再进行实数逆傅里叶变换得到 y2
            y2 = rfft(irfft(x))
            # 断言 y1 和 y2 的数据类型与预期相符
            assert_equal(y1.dtype, self.rdt)
            assert_equal(y2.dtype, self.rdt)
            # 断言 y1 和 x 的值在给定精度下（decimal=self.ndec）近似相等
            assert_array_almost_equal(y1, x, decimal=self.ndec,
                                      err_msg="size=%d" % size)
            # 断言 y2 和 x 的值在给定精度下（decimal=self.ndec）近似相等
            assert_array_almost_equal(y2, x, decimal=self.ndec,
                                      err_msg="size=%d" % size)

    # 测试对于大的复合数和素数大小的输入的精度检查
    def test_size_accuracy(self):
        # 根据数据类型选择不同的相对误差限制
        if self.rdt == np.float32:
            rtol = 1e-5
        elif self.rdt == np.float64:
            rtol = 1e-10

        # 遍历大的复合数和素数大小的输入
        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            # 使用种子初始化随机数生成器
            np.random.seed(1234)
            # 生成指定大小的随机实数数组，并转换为指定的数据类型
            x = np.random.rand(size).astype(self.rdt)
            # 对 x 进行实数逆傅里叶变换，再进行实数傅里叶变换得到 y
            y = irfft(rfft(x))
            # 检查 x 和 y 的接近程度，使用给定的相对误差限制
            _assert_close_in_norm(x, y, rtol, size, self.rdt)
            # 对 x 进行实数傅里叶变换，再进行实数逆傅里叶变换得到 y
            y = rfft(irfft(x))
            # 检查 x 和 y 的接近程度，使用给定的相对误差限制
            _assert_close_in_norm(x, y, rtol, size, self.rdt)

    # 测试不合法大小的输入情况是否能触发 ValueError 异常
    def test_invalid_sizes(self):
        # 断言调用 irfft 函数时传入空列表能触发 ValueError 异常
        assert_raises(ValueError, irfft, [])
        # 断言调用 irfft 函数时传入不合法的二维数组能触发 ValueError 异常
        assert_raises(ValueError, irfft, [[1, 1], [2, 2]], -5)

    # 测试复数输入是否能触发 TypeError 异常
    def test_complex_input(self):
        # 断言调用 irfft 函数时传入复数数组能触发 TypeError 异常
        assert_raises(TypeError, irfft, np.arange(4, dtype=np.complex64))
# 定义一个测试类 TestIRFFTDouble，继承自 _TestIRFFTBase
class TestIRFFTDouble(_TestIRFFTBase):
    # 设置测试方法的初始化操作
    def setup_method(self):
        # 设置复数数据类型为 np.complex128
        self.cdt = np.complex128
        # 设置实数数据类型为 np.float64
        self.rdt = np.float64
        # 设置保留有效数字的位数为 14
        self.ndec = 14


# 定义一个测试类 TestIRFFTSingle，继承自 _TestIRFFTBase
class TestIRFFTSingle(_TestIRFFTBase):
    # 设置测试方法的初始化操作
    def setup_method(self):
        # 设置复数数据类型为 np.complex64
        self.cdt = np.complex64
        # 设置实数数据类型为 np.float32
        self.rdt = np.float32
        # 设置保留有效数字的位数为 5
        self.ndec = 5


# 定义一个测试类 Testfft2
class Testfft2:
    # 设置测试方法的初始化操作
    def setup_method(self):
        # 使用随机种子 1234 初始化 NumPy 的随机数生成器
        np.random.seed(1234)

    # 定义一个测试方法 test_regression_244
    def test_regression_244(self):
        """FFT returns wrong result with axes parameter."""
        # 当 axes 参数和 shape 参数同时使用时，fftn（以及 fft2）曾经会出现问题
        x = numpy.ones((4, 4, 2))
        # 使用 fft2 函数计算 x 的二维快速傅里叶变换，指定 shape=(8, 8)，axes=(-3, -2)
        y = fft2(x, shape=(8, 8), axes=(-3, -2))
        # 使用 np.fft.fftn 函数计算 x 的 n 维快速傅里叶变换，指定 s=(8, 8)，axes=(-3, -2)
        y_r = numpy.fft.fftn(x, s=(8, 8), axes=(-3, -2))
        # 断言 y 和 y_r 的值几乎相等
        assert_array_almost_equal(y, y_r)

    # 定义一个测试方法 test_invalid_sizes
    def test_invalid_sizes(self):
        # 断言 ValueError 异常会被抛出，当传入空列表 [[]] 时调用 fft2 函数
        assert_raises(ValueError, fft2, [[]])
        # 断言 ValueError 异常会被抛出，当传入形状为 [[1, 1], [2, 2]]，shape=(4, -3) 时调用 fft2 函数
        assert_raises(ValueError, fft2, [[1, 1], [2, 2]], (4, -3))


# 定义一个测试类 TestFftnSingle
class TestFftnSingle:
    # 设置测试方法的初始化操作
    def setup_method(self):
        # 使用随机种子 1234 初始化 NumPy 的随机数生成器
        np.random.seed(1234)

    # 定义一个测试方法 test_definition
    def test_definition(self):
        x = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        # 对 x 转换为 np.float32 类型后进行 n 维快速傅里叶变换
        y = fftn(np.array(x, np.float32))
        # 断言 y 的数据类型为 np.complex64，带有 "double precision output with single precision" 的错误信息
        assert_(y.dtype == np.complex64,
                msg="double precision output with single precision")

        # 使用 np.fft.fftn 函数计算 x 的 n 维快速傅里叶变换，并转换为 np.complex64 类型
        y_r = np.array(fftn(x), np.complex64)
        # 断言 y 和 y_r 的数值误差在可接受的范围内
        assert_array_almost_equal_nulp(y, y_r)

    # 使用 pytest.mark.parametrize 标记的参数化测试方法，参数为 SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES
    @pytest.mark.parametrize('size', SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
    def test_size_accuracy_small(self, size):
        # 创建一个大小为 (size, size) 的随机复数矩阵 x
        x = np.random.rand(size, size) + 1j*np.random.rand(size, size)
        # 对 x 的实部转换为 np.float32 类型后进行 n 维快速傅里叶变换
        y1 = fftn(x.real.astype(np.float32))
        # 对 x 的实部转换为 np.float64 类型后进行 n 维快速傅里叶变换，并转换为 np.complex64 类型
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        # 断言 y1 的数据类型为 np.complex64
        assert_equal(y1.dtype, np.complex64)
        # 断言 y1 和 y2 的数值误差在 2000 个最低单位内
        assert_array_almost_equal_nulp(y1, y2, 2000)

    # 使用 pytest.mark.parametrize 标记的参数化测试方法，参数为 LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES
    @pytest.mark.parametrize('size', LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES)
    def test_size_accuracy_large(self, size):
        # 创建一个大小为 (size, 3) 的随机复数矩阵 x
        x = np.random.rand(size, 3) + 1j*np.random.rand(size, 3)
        # 对 x 的实部转换为 np.float32 类型后进行 n 维快速傅里叶变换
        y1 = fftn(x.real.astype(np.float32))
        # 对 x 的实部转换为 np.float64 类型后进行 n 维快速傅里叶变换，并转换为 np.complex64 类型
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        # 断言 y1 的数据类型为 np.complex64
        assert_equal(y1.dtype, np.complex64)
        # 断言 y1 和 y2 的数值误差在 2000 个最低单位内
        assert_array_almost_equal_nulp(y1, y2, 2000)

    # 定义一个测试方法 test_definition_float16
    def test_definition_float16(self):
        x = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        # 对 x 转换为 np.float16 类型后进行 n 维快速傅里叶变换
        y = fftn(np.array(x, np.float16))
        # 断言 y 的数据类型为 np.complex64
        assert_equal(y.dtype, np.complex64)
        # 使用 np.fft.fftn 函数计算 x 的 n 维快速傅里叶变换，并转换为 np.complex64 类型
        y_r = np.array(fftn(x), np.complex64)
        # 断言 y 和 y_r 的数值误差在可接受的范围内
        assert_array_almost_equal_nulp(y, y_r)

    # 使用 pytest.mark.parametrize 标记的参数化测试方法，参数为 SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES
    @pytest.mark.parametrize('size', SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
    def test_float16_input_small(self, size):
        # 创建一个大小为 (size, size) 的随机复数矩阵 x
        x = np.random.rand(size, size) + 1j*np.random.rand(size, size)
        # 对 x 的实部转换为 np.float16 类型后进行 n 维快速傅里叶变换
        y1 = fftn(x.real.astype(np.float16))
        # 对 x 的实部转换为 np.float64 类型后进行 n 维快速傅里叶变换，并转换为 np.complex64 类型
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        # 断言 y1 的数据类型为 np.complex64
        assert_equal(y1.dtype, np.complex64)
        # 断言 y1 和 y2 的数值误差在 5e5 个最低单位内
        assert_array_almost_equal_nulp(y1, y2, 5e5)
    # 使用 pytest 的 parametrize 装饰器，指定参数化测试的参数为 LARGE_COMPOSITE_SIZES 和 LARGE_PRIME_SIZES 中的元素
    @pytest.mark.parametrize('size', LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES)
    # 定义一个测试函数，测试输入为复数形式的随机数组 x，其中大小为 size x 3
    def test_float16_input_large(self, size):
        # 创建一个大小为 size x 3 的随机复数数组 x
        x = np.random.rand(size, 3) + 1j*np.random.rand(size, 3)
        # 对 x 的实部进行 np.float16 类型的傅里叶变换，得到结果 y1
        y1 = fftn(x.real.astype(np.float16))
        # 对 x 的实部进行 np.float64 类型的傅里叶变换，再将结果转换为 np.complex64 类型，得到结果 y2
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        # 断言 y1 的数据类型为 np.complex64
        assert_equal(y1.dtype, np.complex64)
        # 断言 y1 和 y2 的值在误差范围内几乎相等，误差范围为 2e6 个单位的最小单位处的误差
        assert_array_almost_equal_nulp(y1, y2, 2e6)
# 定义一个测试类 TestFftn，用于测试 fftn 函数的各种情况
class TestFftn:
    # 在每个测试方法运行前，设置随机种子为 1234
    def setup_method(self):
        np.random.seed(1234)

    # 测试 fftn 函数的基本定义和使用
    def test_definition(self):
        # 测试对一个小矩阵进行 FFT 变换，并与直接 DFT 变换结果比较
        x = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        y = fftn(x)
        assert_array_almost_equal(y, direct_dftn(x))

        # 对一个随机生成的大矩阵进行 FFT 变换，并与直接 DFT 变换结果比较
        x = random((20, 26))
        assert_array_almost_equal(fftn(x), direct_dftn(x))

        # 对一个随机生成的更大的四维矩阵进行 FFT 变换，并与直接 DFT 变换结果比较
        x = random((5, 4, 3, 20))
        assert_array_almost_equal(fftn(x), direct_dftn(x))

    # 测试 fftn 函数的 shape 参数
    def test_shape_argument(self):
        # 测试在指定 shape 的情况下进行 FFT 变换，并与预期结果比较
        small_x = [[1, 2, 3],
                   [4, 5, 6]]
        large_x1 = [[1, 2, 3, 0],
                    [4, 5, 6, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]]

        y = fftn(small_x, shape=(4, 4))
        assert_array_almost_equal(y, fftn(large_x1))

        y = fftn(small_x, shape=(3, 4))
        assert_array_almost_equal(y, fftn(large_x1[:-1]))

    # 测试 fftn 函数的 shape 和 axes 参数
    def test_shape_axes_argument(self):
        # 测试在指定 shape 和 axes 的情况下进行 FFT 变换，并与预期结果比较
        small_x = [[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]
        large_x1 = array([[1, 2, 3, 0],
                          [4, 5, 6, 0],
                          [7, 8, 9, 0],
                          [0, 0, 0, 0]])
        y = fftn(small_x, shape=(4, 4), axes=(-2, -1))
        assert_array_almost_equal(y, fftn(large_x1))
        y = fftn(small_x, shape=(4, 4), axes=(-1, -2))

        assert_array_almost_equal(y, swapaxes(
            fftn(swapaxes(large_x1, -1, -2)), -1, -2))

    # 测试 fftn 函数的 axes 和 shape 参数的其他情况
    def test_shape_axes_argument2(self):
        # 改变最后一个轴的形状
        x = numpy.random.random((10, 5, 3, 7))
        y = fftn(x, axes=(-1,), shape=(8,))
        assert_array_almost_equal(y, fft(x, axis=-1, n=8))

        # 改变一个非最后一个轴的形状
        x = numpy.random.random((10, 5, 3, 7))
        y = fftn(x, axes=(-2,), shape=(8,))
        assert_array_almost_equal(y, fft(x, axis=-2, n=8))

        # 改变多个轴的形状，参考 issue #244，其中 shape 和 axes 被混淆
        x = numpy.random.random((4, 4, 2))
        y = fftn(x, axes=(-3, -2), shape=(8, 8))
        assert_array_almost_equal(y,
                                  numpy.fft.fftn(x, axes=(-3, -2), s=(8, 8)))

    # 测试 fftn 函数在传入非法大小时的行为
    def test_shape_argument_more(self):
        x = zeros((4, 4, 2))
        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            fftn(x, shape=(8, 8, 2, 1))

    # 测试 fftn 函数在传入无效大小时的行为
    def test_invalid_sizes(self):
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[1, 0\]\) specified"):
            fftn([[]])

        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[4, -3\]\) specified"):
            fftn([[1, 1], [2, 2]], (4, -3))


# 定义一个测试类 TestIfftn，目前没有具体实现内容，只设置了随机种子
class TestIfftn:
    dtype = None
    cdtype = None

    def setup_method(self):
        np.random.seed(1234)
    # 使用 pytest 的参数化装饰器，定义测试方法参数的多组值
    @pytest.mark.parametrize('dtype,cdtype,maxnlp',
                             [(np.float64, np.complex128, 2000),
                              (np.float32, np.complex64, 3500)])
    # 定义测试函数，测试傅立叶逆变换函数 ifftn 的功能
    def test_definition(self, dtype, cdtype, maxnlp):
        # 创建一个 numpy 数组 x，指定数据类型为 dtype
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=dtype)
        # 对 x 进行多维傅立叶逆变换，得到 y
        y = ifftn(x)
        # 断言 y 的数据类型为 cdtype
        assert_equal(y.dtype, cdtype)
        # 断言 ifftn(x) 与 direct_idftn(x) 的数值近似误差不超过 maxnlp
        assert_array_almost_equal_nulp(y, direct_idftn(x), maxnlp)

        # 创建一个随机 numpy 数组 x，形状为 (20, 26)
        x = random((20, 26))
        # 断言 ifftn(x) 与 direct_idftn(x) 的数值近似误差不超过 maxnlp
        assert_array_almost_equal_nulp(ifftn(x), direct_idftn(x), maxnlp)

        # 创建一个随机 numpy 数组 x，形状为 (5, 4, 3, 20)
        x = random((5, 4, 3, 20))
        # 断言 ifftn(x) 与 direct_idftn(x) 的数值近似误差不超过 maxnlp
        assert_array_almost_equal_nulp(ifftn(x), direct_idftn(x), maxnlp)

    # 使用 pytest 的参数化装饰器，定义测试方法参数的多组值
    @pytest.mark.parametrize('maxnlp', [2000, 3500])
    # 使用 pytest 的参数化装饰器，定义测试方法参数的多组值
    @pytest.mark.parametrize('size', [1, 2, 51, 32, 64, 92])
    # 定义测试函数，测试随机复数输入的情况
    def test_random_complex(self, maxnlp, size):
        # 创建一个随机复数数组 x，形状为 (size, size)
        x = random([size, size]) + 1j * random([size, size])
        # 断言 ifftn(fftn(x)) 与 x 的数值近似误差不超过 maxnlp
        assert_array_almost_equal_nulp(ifftn(fftn(x)), x, maxnlp)
        # 断言 fftn(ifftn(x)) 与 x 的数值近似误差不超过 maxnlp
        assert_array_almost_equal_nulp(fftn(ifftn(x)), x, maxnlp)

    # 定义测试函数，测试无效大小的输入情况
    def test_invalid_sizes(self):
        # 使用 assert_raises 检测 ValueError 异常，并匹配指定的错误消息
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[1, 0\]\) specified"):
            # 调用 ifftn 函数并传入无效的空数组 [[]]，预期抛出异常
            ifftn([[]])

        # 使用 assert_raises 检测 ValueError 异常，并匹配指定的错误消息
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[4, -3\]\) specified"):
            # 调用 ifftn 函数并传入无效的数组 [[1, 1], [2, 2]]，预期抛出异常
            ifftn([[1, 1], [2, 2]], (4, -3))
class FakeArray:
    # 初始化方法，接受一个数据对象作为参数
    def __init__(self, data):
        # 将输入的数据对象存储在实例变量 _data 中
        self._data = data
        # 将数据对象的 __array_interface__ 属性复制给实例的 __array_interface__
        self.__array_interface__ = data.__array_interface__


class FakeArray2:
    # 初始化方法，接受一个数据对象作为参数
    def __init__(self, data):
        # 将输入的数据对象存储在实例变量 _data 中
        self._data = data

    # __array__ 方法，返回实例的数据对象 _data
    def __array__(self, dtype=None, copy=None):
        return self._data


class TestOverwrite:
    """Check input overwrite behavior of the FFT functions."""

    # 支持的实数数据类型
    real_dtypes = (np.float32, np.float64)
    # 支持的所有数据类型，包括实数和复数
    dtypes = real_dtypes + (np.complex64, np.complex128)
    # FFT 处理的尺寸
    fftsizes = [8, 16, 32]

    # 检查函数，验证输入参数的覆写行为
    def _check(self, x, routine, fftsize, axis, overwrite_x):
        # 复制输入数据 x 到 x2
        x2 = x.copy()
        # 对于三种不同的伪数组类型，执行 FFT 函数
        for fake in [lambda x: x, FakeArray, FakeArray2]:
            routine(fake(x2), fftsize, axis, overwrite_x=overwrite_x)

            # 构建函数调用的签名字符串
            sig = "{}({}{!r}, {!r}, axis={!r}, overwrite_x={!r})".format(
                routine.__name__, x.dtype, x.shape, fftsize, axis, overwrite_x)
            # 如果不允许覆写，断言 x2 与 x 相等，否则抛出错误信息
            if not overwrite_x:
                assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)

    # 检查一维输入数据的函数
    def _check_1d(self, routine, dtype, shape, axis, overwritable_dtypes,
                  fftsize, overwrite_x):
        # 使用种子生成随机数据
        np.random.seed(1234)
        # 根据数据类型生成相应的随机数据
        if np.issubdtype(dtype, np.complexfloating):
            data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
        else:
            data = np.random.randn(*shape)
        data = data.astype(dtype)

        # 调用 _check 方法验证数据处理函数的覆写行为
        self._check(data, routine, fftsize, axis,
                    overwrite_x=overwrite_x)

    # 参数化测试 FFT 和 IFFT 函数
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('fftsize', fftsizes)
    @pytest.mark.parametrize('overwrite_x', [True, False])
    @pytest.mark.parametrize('shape,axes', [((16,), -1),
                                            ((16, 2), 0),
                                            ((2, 16), 1)])
    def test_fft_ifft(self, dtype, fftsize, overwrite_x, shape, axes):
        overwritable = (np.complex128, np.complex64)
        self._check_1d(fft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)
        self._check_1d(ifft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)

    # 参数化测试实数 FFT 和 IRFFT 函数
    @pytest.mark.parametrize('dtype', real_dtypes)
    @pytest.mark.parametrize('fftsize', fftsizes)
    @pytest.mark.parametrize('overwrite_x', [True, False])
    @pytest.mark.parametrize('shape,axes', [((16,), -1),
                                            ((16, 2), 0),
                                            ((2, 16), 1)])
    def test_rfft_irfft(self, dtype, fftsize, overwrite_x, shape, axes):
        overwritable = self.real_dtypes
        self._check_1d(irfft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)
        self._check_1d(rfft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)
    # 定义一个方法，用于检查单个 FFT / IFFT 操作的正确性
    def _check_nd_one(self, routine, dtype, shape, axes, overwritable_dtypes,
                      overwrite_x):
        # 设定随机种子以确保可重复性
        np.random.seed(1234)
        
        # 根据数据类型生成相应形状的随机数据
        if np.issubdtype(dtype, np.complexfloating):
            # 如果数据类型是复数类型，则生成复数随机数据
            data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
        else:
            # 否则生成实数随机数据
            data = np.random.randn(*shape)
        
        # 将数据类型转换为指定的 dtype 类型
        data = data.astype(dtype)

        # 定义一个递归生成 FFT 形状的迭代器
        def fftshape_iter(shp):
            if len(shp) <= 0:
                yield ()
            else:
                for j in (shp[0]//2, shp[0], shp[0]*2):
                    for rest in fftshape_iter(shp[1:]):
                        yield (j,) + rest

        # 如果未指定 axes，则使用整个数据形状
        if axes is None:
            part_shape = shape
        else:
            # 否则根据指定的 axes 提取数据部分形状
            part_shape = tuple(np.take(shape, axes))

        # 遍历 FFT 形状的各种可能组合
        for fftshape in fftshape_iter(part_shape):
            # 对数据执行 _check 方法，用于检查 FFT / IFFT 操作的正确性
            self._check(data, routine, fftshape, axes,
                        overwrite_x=overwrite_x)
            # 如果数据维度大于 1，则对数据的转置执行 _check 方法，进一步检查
            if data.ndim > 1:
                self._check(data.T, routine, fftshape, axes,
                            overwrite_x=overwrite_x)

    # 使用 pytest 的参数化装饰器，为 dtype、overwrite_x、shape 和 axes 参数进行组合测试
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('overwrite_x', [True, False])
    @pytest.mark.parametrize('shape,axes', [((16,), None),
                                            ((16,), (0,)),
                                            ((16, 2), (0,)),
                                            ((2, 16), (1,)),
                                            ((8, 16), None),
                                            ((8, 16), (0, 1)),
                                            ((8, 16, 2), (0, 1)),
                                            ((8, 16, 2), (1, 2)),
                                            ((8, 16, 2), (0,)),
                                            ((8, 16, 2), (1,)),
                                            ((8, 16, 2), (2,)),
                                            ((8, 16, 2), None),
                                            ((8, 16, 2), (0, 1, 2))])
    # 定义测试 FFTN / IFFTN 操作的方法，通过不同参数组合进行测试
    def test_fftn_ifftn(self, dtype, overwrite_x, shape, axes):
        # 定义可重写的数据类型为复数类型
        overwritable = (np.complex128, np.complex64)
        
        # 调用 _check_nd_one 方法，测试 FFTN 操作
        self._check_nd_one(fftn, dtype, shape, axes, overwritable,
                           overwrite_x)
        
        # 调用 _check_nd_one 方法，测试 IFFTN 操作
        self._check_nd_one(ifftn, dtype, shape, axes, overwritable,
                           overwrite_x)
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 func，可传入 fftn、ifftn、fft2 函数
@pytest.mark.parametrize('func', [fftn, ifftn, fft2])
# 定义测试函数 test_shape_axes_ndarray(func)，测试 fftn 和 ifftn 函数在 NumPy 数组的形状和轴参数上的表现
# 这是对 gh-13342 的回归测试
def test_shape_axes_ndarray(func):
    # 创建一个 10x10 的随机数 NumPy 数组 a
    a = np.random.rand(10, 10)

    # 期望值为调用 func(a, shape=(5, 5)) 的结果
    expect = func(a, shape=(5, 5))
    # 实际值为调用 func(a, shape=np.array([5, 5])) 的结果
    actual = func(a, shape=np.array([5, 5]))
    # 使用 assert_equal 函数比较期望值和实际值是否相等
    assert_equal(expect, actual)

    # 期望值为调用 func(a, axes=(-1,)) 的结果
    expect = func(a, axes=(-1,))
    # 实际值为调用 func(a, axes=np.array([-1,])) 的结果
    actual = func(a, axes=np.array([-1,]))
    # 使用 assert_equal 函数比较期望值和实际值是否相等
    assert_equal(expect, actual)

    # 期望值为调用 func(a, shape=(4, 7), axes=(1, 0)) 的结果
    expect = func(a, shape=(4, 7), axes=(1, 0))
    # 实际值为调用 func(a, shape=np.array([4, 7]), axes=np.array([1, 0])) 的结果
    actual = func(a, shape=np.array([4, 7]), axes=np.array([1, 0]))
    # 使用 assert_equal 函数比较期望值和实际值是否相等
    assert_equal(expect, actual)
```