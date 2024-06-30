# `D:\src\scipysrc\scipy\scipy\fft\_pocketfft\tests\test_basic.py`

```
# Created by Pearu Peterson, September 2002

# 从 numpy.testing 模块导入多个断言函数，用于测试
from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
                           assert_array_almost_equal_nulp, assert_array_less,
                           assert_allclose)
# 导入 pytest 模块及其 raises 函数别名
import pytest
from pytest import raises as assert_raises
# 从 scipy.fft._pocketfft 模块导入多个 FFT 相关函数
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
                                  rfft, irfft, rfftn, irfftn,
                                  hfft, ihfft, hfftn, ihfftn)

# 导入多个函数和常量从 numpy 模块
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
                   swapaxes, cdouble)
import numpy as np
import numpy.fft
# 从 numpy.random 模块导入 rand 函数
from numpy.random import rand

# 定义大型复合数的列表，这些数被 FFT._PYPOCKETFFT 支持
LARGE_COMPOSITE_SIZES = [
    2**13,
    2**5 * 3**5,
    2**3 * 3**3 * 5**2,
]
# 定义小型复合数的列表
SMALL_COMPOSITE_SIZES = [
    2,
    2*3*5,
    2*2*3*3,
]
# 定义大型质数的列表
LARGE_PRIME_SIZES = [
    2011
]
# 定义小型质数的列表
SMALL_PRIME_SIZES = [
    29
]

# 定义一个辅助函数，用于测试，比较两个向量的范数
def _assert_close_in_norm(x, y, rtol, size, rdt):
    err_msg = f"size: {size}  rdt: {rdt}"
    assert_array_less(np.linalg.norm(x - y), rtol*np.linalg.norm(x), err_msg)

# 定义一个生成指定大小随机数组的函数
def random(size):
    return rand(*size)

# 定义一个函数，返回输入数组的字节顺序交换后的结果
def swap_byteorder(arr):
    """Returns the same array with swapped byteorder"""
    dtype = arr.dtype.newbyteorder('S')
    return arr.astype(dtype)

# 定义直接实现离散傅里叶变换（DFT）的函数
def direct_dft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = -arange(n)*(2j*pi/n)
    for i in range(n):
        y[i] = dot(exp(i*w), x)
    return y

# 定义直接实现逆离散傅里叶变换（IDFT）的函数
def direct_idft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = arange(n)*(2j*pi/n)
    for i in range(n):
        y[i] = dot(exp(i*w), x)/n
    return y

# 定义直接实现多维离散傅里叶变换（DFTN）的函数
def direct_dftn(x):
    x = asarray(x)
    for axis in range(x.ndim):
        x = fft(x, axis=axis)
    return x

# 定义直接实现多维逆离散傅里叶变换（IDFTN）的函数
def direct_idftn(x):
    x = asarray(x)
    for axis in range(x.ndim):
        x = ifft(x, axis=axis)
    return x

# 定义直接实现实数输入的离散傅里叶变换（RDFT）的函数
def direct_rdft(x):
    x = asarray(x)
    n = len(x)
    w = -arange(n)*(2j*pi/n)
    y = zeros(n//2+1, dtype=cdouble)
    for i in range(n//2+1):
        y[i] = dot(exp(i*w), x)
    return y

# 定义直接实现实数输入的逆离散傅里叶变换（IRDFT）的函数
def direct_irdft(x, n):
    x = asarray(x)
    x1 = zeros(n, dtype=cdouble)
    for i in range(n//2+1):
        x1[i] = x[i]
        if i > 0 and 2*i < n:
            x1[n-i] = np.conj(x[i])
    return direct_idft(x1).real

# 定义直接实现多维实数输入的实数离散傅里叶变换（RDFTN）的函数
def direct_rdftn(x):
    return fftn(rfft(x), axes=range(x.ndim - 1))

# 定义 _TestFFTBase 类，用于测试 FFT 功能的基本类
class _TestFFTBase:
    def setup_method(self):
        self.cdt = None
        self.rdt = None
        np.random.seed(1234)

    # 定义测试离散傅里叶变换定义的方法
    def test_definition(self):
        x = np.array([1,2,3,4+1j,1,2,3,4+2j], dtype=self.cdt)
        y = fft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_dft(x)
        assert_array_almost_equal(y,y1)
        x = np.array([1,2,3,4+0j,5], dtype=self.cdt)
        assert_array_almost_equal(fft(x),direct_dft(x))
    # 定义测试函数，用于测试带有 n 参数的 FFT 函数的实际情况
    def test_n_argument_real(self):
        # 创建一个包含 [1,2,3,4] 的 NumPy 数组 x1，数据类型为 self.rdt
        x1 = np.array([1,2,3,4], dtype=self.rdt)
        # 创建另一个包含 [1,2,3,4] 的 NumPy 数组 x2，数据类型也为 self.rdt
        x2 = np.array([1,2,3,4], dtype=self.rdt)
        # 对 x1 和 x2 应用 FFT 函数，指定 n=4
        y = fft([x1,x2],n=4)
        # 断言 y 的数据类型等于 self.cdt
        assert_equal(y.dtype, self.cdt)
        # 断言 y 的形状为 (2,4)
        assert_equal(y.shape,(2,4))
        # 断言 y 的第一个元素近似等于直接计算 x1 的 DFT
        assert_array_almost_equal(y[0],direct_dft(x1))
        # 断言 y 的第二个元素近似等于直接计算 x2 的 DFT
        assert_array_almost_equal(y[1],direct_dft(x2))
    
    # 定义测试函数，用于测试带有 n 参数的 FFT 函数的复数情况
    def _test_n_argument_complex(self):
        # 创建一个包含 [1,2,3,4+1j] 的复数 NumPy 数组 x1，数据类型为 self.cdt
        x1 = np.array([1,2,3,4+1j], dtype=self.cdt)
        # 创建另一个包含 [1,2,3,4+1j] 的复数 NumPy 数组 x2，数据类型也为 self.cdt
        x2 = np.array([1,2,3,4+1j], dtype=self.cdt)
        # 对 x1 和 x2 应用 FFT 函数，指定 n=4
        y = fft([x1,x2],n=4)
        # 断言 y 的数据类型等于 self.cdt
        assert_equal(y.dtype, self.cdt)
        # 断言 y 的形状为 (2,4)
        assert_equal(y.shape,(2,4))
        # 断言 y 的第一个元素近似等于直接计算 x1 的 DFT
        assert_array_almost_equal(y[0],direct_dft(x1))
        # 断言 y 的第二个元素近似等于直接计算 x2 的 DFT
        assert_array_almost_equal(y[1],direct_dft(x2))
    
    # 定义测试函数，用于测试 DJBFFT 算法在不同大小输入上的输出
    def test_djbfft(self):
        # 对于范围在 [2, 13] 内的整数 i
        for i in range(2,14):
            # 计算 n = 2^i
            n = 2**i
            # 创建长度为 n 的 NumPy 数组 x
            x = np.arange(n)
            # 应用 FFT 函数到复数形式的 x
            y = fft(x.astype(complex))
            # 使用 NumPy 自带的 FFT 函数计算 x 的 FFT
            y2 = numpy.fft.fft(x)
            # 断言 y 和 y2 的结果近似相等
            assert_array_almost_equal(y,y2)
            # 对于实数形式的 x，再次应用 FFT 函数
            y = fft(x)
            # 断言 y 和 y2 的结果近似相等
            assert_array_almost_equal(y,y2)
    
    # 定义测试函数，用于测试 FFT 函数在无效大小输入时是否引发 ValueError 异常
    def test_invalid_sizes(self):
        # 断言调用 FFT 函数时传入空列表会引发 ValueError 异常
        assert_raises(ValueError, fft, [])
        # 断言调用 FFT 函数时传入包含两个元素的列表 [[1,1],[2,2]] 且 n 值为 -5 会引发 ValueError 异常
        assert_raises(ValueError, fft, [[1,1],[2,2]], -5)
class TestLongDoubleFFT(_TestFFTBase):
    # 定义测试类 TestLongDoubleFFT，继承自 _TestFFTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.cdt = np.clongdouble
        self.rdt = np.longdouble


class TestDoubleFFT(_TestFFTBase):
    # 定义测试类 TestDoubleFFT，继承自 _TestFFTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.cdt = np.cdouble
        self.rdt = np.float64


class TestSingleFFT(_TestFFTBase):
    # 定义测试类 TestSingleFFT，继承自 _TestFFTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.cdt = np.complex64
        self.rdt = np.float32


class TestFloat16FFT:
    # 定义测试类 TestFloat16FFT

    def test_1_argument_real(self):
        # 测试方法：使用实数数组进行 FFT 的单参数测试
        x1 = np.array([1, 2, 3, 4], dtype=np.float16)
        y = fft(x1, n=4)
        assert_equal(y.dtype, np.complex64)
        assert_equal(y.shape, (4, ))
        assert_array_almost_equal(y, direct_dft(x1.astype(np.float32)))

    def test_n_argument_real(self):
        # 测试方法：使用实数数组进行 FFT 的多参数测试
        x1 = np.array([1, 2, 3, 4], dtype=np.float16)
        x2 = np.array([1, 2, 3, 4], dtype=np.float16)
        y = fft([x1, x2], n=4)
        assert_equal(y.dtype, np.complex64)
        assert_equal(y.shape, (2, 4))
        assert_array_almost_equal(y[0], direct_dft(x1.astype(np.float32)))
        assert_array_almost_equal(y[1], direct_dft(x2.astype(np.float32)))


class _TestIFFTBase:
    # 定义测试基类 _TestIFFTBase

    def setup_method(self):
        # 设置测试方法的初始化操作
        np.random.seed(1234)

    def test_definition(self):
        # 测试方法：定义复数数组进行 IFFT 的测试
        x = np.array([1, 2, 3, 4+1j, 1, 2, 3, 4+2j], self.cdt)
        y = ifft(x)
        y1 = direct_idft(x)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(y, y1)

        x = np.array([1, 2, 3, 4+0j, 5], self.cdt)
        assert_array_almost_equal(ifft(x), direct_idft(x))

    def test_definition_real(self):
        # 测试方法：定义实数数组进行 IFFT 的测试
        x = np.array([1, 2, 3, 4, 1, 2, 3, 4], self.rdt)
        y = ifft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_idft(x)
        assert_array_almost_equal(y, y1)

        x = np.array([1, 2, 3, 4, 5], dtype=self.rdt)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(ifft(x), direct_idft(x))

    def test_djbfft(self):
        # 测试方法：使用不同大小的数组进行 FFT 和 IFFT 的测试
        for i in range(2, 14):
            n = 2**i
            x = np.arange(n)
            y = ifft(x.astype(self.cdt))
            y2 = numpy.fft.ifft(x.astype(self.cdt))
            assert_allclose(y, y2, rtol=self.rtol, atol=self.atol)
            y = ifft(x)
            assert_allclose(y, y2, rtol=self.rtol, atol=self.atol)

    def test_random_complex(self):
        # 测试方法：使用随机复数数组进行 FFT 和 IFFT 的测试
        for size in [1, 51, 111, 100, 200, 64, 128, 256, 1024]:
            x = np.random.random([size]).astype(self.cdt)
            x = np.random.random([size]).astype(self.cdt) + 1j * x
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)
    # 定义一个测试函数，用于验证随机实数输入的傅里叶变换与逆变换的准确性
    def test_random_real(self):
        # 遍历不同的输入大小
        for size in [1,51,111,100,200,64,128,256,1024]:
            # 生成指定大小的随机实数数组，并转换为指定类型 self.rdt
            x = random([size]).astype(self.rdt)
            # 对 x 进行傅里叶变换后再进行逆变换，得到 y1
            y1 = ifft(fft(x))
            # 对 x 进行逆变换后再进行傅里叶变换，得到 y2
            y2 = fft(ifft(x))
            # 断言 y1 的数据类型为 self.cdt
            assert_equal(y1.dtype, self.cdt)
            # 断言 y2 的数据类型为 self.cdt
            assert_equal(y2.dtype, self.cdt)
            # 断言 y1 与原始输入 x 几乎相等
            assert_array_almost_equal(y1, x)
            # 断言 y2 与原始输入 x 几乎相等
            assert_array_almost_equal(y2, x)

    # 定义一个测试函数，用于验证不同大小输入的精度
    def test_size_accuracy(self):
        # 对于大型复合大小和大型素数大小的输入进行基本检查
        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            # 设置随机数种子
            np.random.seed(1234)
            # 生成指定大小的随机浮点数数组，并转换为指定类型 self.rdt
            x = np.random.rand(size).astype(self.rdt)
            # 对 x 进行傅里叶变换后再进行逆变换，得到 y
            y = ifft(fft(x))
            # 使用自定义函数检查 x 和 y 在规范下的接近程度
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            # 对 x 进行逆变换后再进行傅里叶变换，得到 y
            y = fft(ifft(x))
            # 使用自定义函数检查 x 和 y 在规范下的接近程度
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)

            # 生成包含虚部的复数数组，并转换为指定类型 self.cdt
            x = (x + 1j*np.random.rand(size)).astype(self.cdt)
            # 对 x 进行傅里叶变换后再进行逆变换，得到 y
            y = ifft(fft(x))
            # 使用自定义函数检查 x 和 y 在规范下的接近程度
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            # 对 x 进行逆变换后再进行傅里叶变换，得到 y
            y = fft(ifft(x))
            # 使用自定义函数检查 x 和 y 在规范下的接近程度
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)

    # 定义一个测试函数，用于验证无效输入大小时是否引发异常
    def test_invalid_sizes(self):
        # 断言空列表输入时会引发 ValueError 异常
        assert_raises(ValueError, ifft, [])
        # 断言非法输入矩阵时会引发 ValueError 异常
        assert_raises(ValueError, ifft, [[1,1],[2,2]], -5)
# 如果 np.longdouble 和 np.float64 是同一个类型，则跳过测试，给出跳过的原因
@pytest.mark.skipif(np.longdouble is np.float64,
                    reason="Long double is aliased to double")
# 定义一个测试类 TestLongDoubleIFFT，继承自 _TestIFFTBase
class TestLongDoubleIFFT(_TestIFFTBase):
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置复数类型为 np.clongdouble
        self.cdt = np.clongdouble
        # 设置实数类型为 np.longdouble
        self.rdt = np.longdouble
        # 设置相对容差为 1e-10
        self.rtol = 1e-10
        # 设置绝对容差为 1e-10


# 定义一个测试类 TestDoubleIFFT，继承自 _TestIFFTBase
class TestDoubleIFFT(_TestIFFTBase):
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置复数类型为 np.complex128
        self.cdt = np.complex128
        # 设置实数类型为 np.float64
        self.rdt = np.float64
        # 设置相对容差为 1e-10
        self.rtol = 1e-10
        # 设置绝对容差为 1e-10


# 定义一个测试类 TestSingleIFFT，继承自 _TestIFFTBase
class TestSingleIFFT(_TestIFFTBase):
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置复数类型为 np.complex64
        self.cdt = np.complex64
        # 设置实数类型为 np.float32
        self.rdt = np.float32
        # 设置相对容差为 1e-5
        self.rtol = 1e-5
        # 设置绝对容差为 1e-4


# 定义一个基类 _TestRFFTBase
class _TestRFFTBase:
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置随机种子为 1234
        np.random.seed(1234)

    # 定义测试方法 test_definition
    def test_definition(self):
        # 遍历测试集合
        for t in [[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4, 5]]:
            # 创建 numpy 数组 x，数据类型为 self.rdt
            x = np.array(t, dtype=self.rdt)
            # 计算 x 的快速傅里叶变换
            y = rfft(x)
            # 计算 x 的直接实数快速傅里叶变换
            y1 = direct_rdft(x)
            # 断言 y 和 y1 数组几乎相等
            assert_array_almost_equal(y, y1)
            # 断言 y 的数据类型与 self.cdt 相等
            assert_equal(y.dtype, self.cdt)

    # 定义测试方法 test_djbfft
    def test_djbfft(self):
        # 遍历指数从 2 到 13
        for i in range(2, 14):
            # 计算 n = 2 的 i 次方
            n = 2**i
            # 创建长度为 n 的等差数组 x
            x = np.arange(n)
            # 计算 x 的 numpy 快速傅里叶变换
            y1 = np.fft.rfft(x)
            # 计算 x 的快速傅里叶变换
            y = rfft(x)
            # 断言 y 和 y1 数组几乎相等
            assert_array_almost_equal(y, y1)

    # 定义测试方法 test_invalid_sizes
    def test_invalid_sizes(self):
        # 断言 rfft 函数对空数组抛出 ValueError 异常
        assert_raises(ValueError, rfft, [])
        # 断言 rfft 函数对非二维数组抛出 ValueError 异常
        assert_raises(ValueError, rfft, [[1, 1], [2, 2]], -5)

    # 定义测试方法 test_complex_input
    def test_complex_input(self):
        # 创建长度为 10 的零数组 x，数据类型为 self.cdt
        x = np.zeros(10, dtype=self.cdt)
        # 断言 rfft 函数对 x 抛出 TypeError 异常，异常信息匹配 "x must be a real sequence"
        with assert_raises(TypeError, match="x must be a real sequence"):
            rfft(x)

    # 定义一个内部类 MockSeries，用于模拟序列
    # 参见 gh-5790
    class MockSeries:
        # 定义初始化方法，接受数据并转换为 numpy 数组
        def __init__(self, data):
            self.data = np.asarray(data)

        # 定义 __getattr__ 方法，用于访问属性
        def __getattr__(self, item):
            try:
                # 尝试获取 self.data 的属性
                return getattr(self.data, item)
            except AttributeError as e:
                # 若属性不存在，则抛出带有属性名的 AttributeError 异常
                raise AttributeError("'MockSeries' object "
                                     f"has no attribute '{item}'") from e

    # 定义测试方法 test_non_ndarray_with_dtype
    def test_non_ndarray_with_dtype(self):
        # 创建长度为 5 的浮点数组 x
        x = np.array([1., 2., 3., 4., 5.])
        # 使用 MockSeries 类将 x 转换为 MockSeries 对象 xs
        xs = _TestRFFTBase.MockSeries(x)

        # 期望的结果列表
        expected = [1, 2, 3, 4, 5]
        # 对 xs 进行快速傅里叶变换
        rfft(xs)

        # 断言数据 x 没有被改写
        assert_equal(x, expected)
        # 断言 MockSeries 对象 xs 的 data 属性与期望结果相等
        assert_equal(xs.data, expected)

# 如果 np.longdouble 和 np.float64 是同一个类型，则跳过测试，给出跳过的原因
@pytest.mark.skipif(np.longdouble is np.float64,
                    reason="Long double is aliased to double")
# 定义一个测试类 TestRFFTLongDouble，继承自 _TestRFFTBase
class TestRFFTLongDouble(_TestRFFTBase):
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置复数类型为 np.clongdouble
        self.cdt = np.clongdouble
        # 设置实数类型为 np.longdouble
        self.rdt = np.longdouble


# 定义一个测试类 TestRFFTDouble，继承自 _TestRFFTBase
class TestRFFTDouble(_TestRFFTBase):
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置复数类型为 np.complex128
        self.cdt = np.complex128
        # 设置实数类型为 np.float64
        self.rdt = np.float64


# 定义一个测试类 TestRFFTSingle，继承自 _TestRFFTBase
class TestRFFTSingle(_TestRFFTBase):
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置复数类型为 np.complex64
        self.cdt = np.complex64
        # 设置实数类型为 np.float32
        self.rdt = np.float32


# 定义一个基类 _TestIRFFTBase
class _TestIRFFTBase:
    # 设置测试方法的初始化
    def setup_method(self):
        # 设置随机种子为 1234
        np.random.seed(1234)
    def test_definition(self):
        # 定义测试函数 test_definition，用于测试不同输入的反向快速傅里叶变换的结果

        x1 = [1,2+3j,4+1j,1+2j,3+4j]
        # 定义复数列表 x1，包含了五个复数元素

        x1_1 = [1,2+3j,4+1j,2+3j,4,2-3j,4-1j,2-3j]
        # 定义复数列表 x1_1，包含了八个复数元素

        x1 = x1_1[:5]
        # 变量 x1 被赋值为 x1_1 列表的前五个元素

        x2_1 = [1,2+3j,4+1j,2+3j,4+5j,4-5j,2-3j,4-1j,2-3j]
        # 定义复数列表 x2_1，包含了九个复数元素

        x2 = x2_1[:5]
        # 变量 x2 被赋值为 x2_1 列表的前五个元素

        def _test(x, xr):
            # 定义内部测试函数 _test，接受两个参数 x 和 xr

            y = irfft(np.array(x, dtype=self.cdt), n=len(xr))
            # 计算 x 的反向快速傅里叶变换，并将结果存储在 y 中

            y1 = direct_irdft(x, len(xr))
            # 调用 direct_irdft 函数，计算 x 的直接逆离散傅里叶变换

            assert_equal(y.dtype, self.rdt)
            # 断言 y 的数据类型与 self.rdt 相等

            assert_array_almost_equal(y,y1, decimal=self.ndec)
            # 断言 y 和 y1 数组在给定的小数精度 self.ndec 下几乎相等

            assert_array_almost_equal(y,ifft(xr), decimal=self.ndec)
            # 断言 y 和 ifft(xr) 数组在给定的小数精度 self.ndec 下几乎相等

        _test(x1, x1_1)
        # 调用 _test 函数，传入 x1 和 x1_1 进行测试

        _test(x2, x2_1)
        # 调用 _test 函数，传入 x2 和 x2_1 进行测试

    def test_djbfft(self):
        # 定义测试函数 test_djbfft，用于测试不同大小的复数输入的反向快速傅里叶变换结果

        for i in range(2,14):
            # 迭代 i 从 2 到 13

            n = 2**i
            # 计算 n 为 2 的 i 次幂

            x = np.arange(-1, n, 2) + 1j * np.arange(0, n+1, 2)
            # 创建复数数组 x，包含了从 -1 开始，以步长 2 的实部和从 0 开始，以步长 2 的虚部

            x[0] = 0
            # 将 x 的第一个元素设为 0

            if n % 2 == 0:
                x[-1] = np.real(x[-1])
                # 如果 n 是偶数，将 x 的最后一个元素的虚部替换为其实部

            y1 = np.fft.irfft(x)
            # 使用 numpy 库中的 irfft 函数计算 x 的反向快速傅里叶变换

            y = irfft(x)
            # 调用 irfft 函数，计算 x 的反向快速傅里叶变换

            assert_array_almost_equal(y,y1)
            # 断言 y 和 y1 数组在几乎相等

    def test_random_real(self):
        # 定义测试函数 test_random_real，用于测试不同大小的随机实数输入的反向快速傅里叶变换结果

        for size in [1,51,111,100,200,64,128,256,1024]:
            # 迭代 size 取值分别为 1, 51, 111, 100, 200, 64, 128, 256, 1024

            x = random([size]).astype(self.rdt)
            # 生成大小为 size 的随机实数数组 x，并转换为指定的数据类型 self.rdt

            y1 = irfft(rfft(x), n=size)
            # 对 x 进行快速傅里叶变换，然后再进行反向快速傅里叶变换，结果存储在 y1 中

            y2 = rfft(irfft(x, n=(size*2-1)))
            # 对 x 进行反向快速傅里叶变换，然后再进行快速傅里叶变换，结果存储在 y2 中

            assert_equal(y1.dtype, self.rdt)
            # 断言 y1 的数据类型与 self.rdt 相等

            assert_equal(y2.dtype, self.cdt)
            # 断言 y2 的数据类型与 self.cdt 相等

            assert_array_almost_equal(y1, x, decimal=self.ndec,
                                       err_msg="size=%d" % size)
            # 断言 y1 和 x 数组在给定的小数精度 self.ndec 下几乎相等，如果不等则输出错误信息

            assert_array_almost_equal(y2, x, decimal=self.ndec,
                                       err_msg="size=%d" % size)
            # 断言 y2 和 x 数组在给定的小数精度 self.ndec 下几乎相等，如果不等则输出错误信息

    def test_size_accuracy(self):
        # 定义测试函数 test_size_accuracy，用于测试大数和素数大小输入的反向快速傅里叶变换的精度

        # 根据数据类型 self.rdt 来设定 rtol 的值
        if self.rdt == np.float32:
            rtol = 1e-5
        elif self.rdt == np.float64:
            rtol = 1e-10

        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            # 迭代 size 分别为 LARGE_COMPOSITE_SIZES 和 LARGE_PRIME_SIZES 中的值

            np.random.seed(1234)
            # 设定随机数种子为 1234

            x = np.random.rand(size).astype(self.rdt)
            # 生成大小为 size 的随机实数数组 x，并转换为指定的数据类型 self.rdt

            y = irfft(rfft(x), len(x))
            # 对 x 进行快速傅里叶变换，然后再进行反向快速傅里叶变换，结果存储在 y 中

            _assert_close_in_norm(x, y, rtol, size, self.rdt)
            # 调用 _assert_close_in_norm 函数，断言 x 和 y 的标准化近似误差在给定的容差范围内

            y = rfft(irfft(x, 2 * len(x) - 1))
            # 对 x 进行反向快速傅里叶变换，然后再进行快速傅里叶变换，结果存储在 y 中

            _assert_close_in_norm(x, y, rtol, size, self.rdt)
            # 调用 _assert_close_in_norm 函数，断言 x 和 y 的标准化近似误差在给定的容差范围内

    def test_invalid_sizes(self):
        # 定义测试函数 test_invalid_sizes，用于测试对于无效大小输入的处理是否引发 ValueError 异常

        assert_raises(ValueError, irfft, [])
        # 断言调用 irfft 函数时传入空列表会引发 ValueError 异常

        assert_raises(ValueError, irfft, [[1,1],[2,2]], -5)
        # 断言调用 irfft 函数时传入二维列表和负数会引发 ValueError 异常
# self.ndec is bogus; we should have a assert_array_approx_equal for number of
# significant digits

# 使用 pytest.mark.skipif 装饰器，指定条件为 np.longdouble 等于 np.float64 时跳过测试，
# 理由是长双精度浮点数别名为双精度浮点数
@pytest.mark.skipif(np.longdouble is np.float64,
                    reason="Long double is aliased to double")
# 定义 TestIRFFTLongDouble 类，继承自 _TestIRFFTBase 类
class TestIRFFTLongDouble(_TestIRFFTBase):
    # 初始化方法 setup_method
    def setup_method(self):
        self.cdt = np.complex128  # 设置 self.cdt 为 np.complex128
        self.rdt = np.float64      # 设置 self.rdt 为 np.float64
        self.ndec = 14             # 设置 self.ndec 为 14


# 定义 TestIRFFTDouble 类，继承自 _TestIRFFTBase 类
class TestIRFFTDouble(_TestIRFFTBase):
    # 初始化方法 setup_method
    def setup_method(self):
        self.cdt = np.complex128  # 设置 self.cdt 为 np.complex128
        self.rdt = np.float64      # 设置 self.rdt 为 np.float64
        self.ndec = 14             # 设置 self.ndec 为 14


# 定义 TestIRFFTSingle 类，继承自 _TestIRFFTBase 类
class TestIRFFTSingle(_TestIRFFTBase):
    # 初始化方法 setup_method
    def setup_method(self):
        self.cdt = np.complex64  # 设置 self.cdt 为 np.complex64
        self.rdt = np.float32    # 设置 self.rdt 为 np.float32
        self.ndec = 5            # 设置 self.ndec 为 5


# 定义 TestFftnSingle 类
class TestFftnSingle:
    # 初始化方法 setup_method
    def setup_method(self):
        np.random.seed(1234)  # 设定随机种子为 1234

    # 定义 test_definition 方法
    def test_definition(self):
        x = [[1, 2, 3],           # 定义二维列表 x
             [4, 5, 6],
             [7, 8, 9]]
        y = fftn(np.array(x, np.float32))  # 对 x 进行快速傅里叶变换，存入 y
        assert_(y.dtype == np.complex64,  # 断言 y 的数据类型为 np.complex64
                msg="double precision output with single precision")

        y_r = np.array(fftn(x), np.complex64)  # 对 x 进行快速傅里叶变换，结果存入 y_r
        assert_array_almost_equal_nulp(y, y_r)  # 断言 y 和 y_r 的近似相等性

    # 使用 pytest.mark.parametrize 装饰器，参数为 SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES
    def test_size_accuracy_small(self, size):
        x = np.random.rand(size, size) + 1j*np.random.rand(size, size)  # 生成大小为 size*size 的随机复数矩阵 x
        y1 = fftn(x.real.astype(np.float32))  # 对 x 的实部转换为 np.float32 类型后进行快速傅里叶变换，结果存入 y1
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)  # 对 x 的实部转换为 np.float64 类型后进行快速傅里叶变换，并转换为 np.complex64 类型，结果存入 y2

        assert_equal(y1.dtype, np.complex64)  # 断言 y1 的数据类型为 np.complex64
        assert_array_almost_equal_nulp(y1, y2, 2000)  # 断言 y1 和 y2 的近似相等性，允许误差为 2000 个单位最低单位

    # 使用 pytest.mark.parametrize 装饰器，参数为 LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES
    def test_size_accuracy_large(self, size):
        x = np.random.rand(size, 3) + 1j*np.random.rand(size, 3)  # 生成大小为 size*3 的随机复数矩阵 x
        y1 = fftn(x.real.astype(np.float32))  # 对 x 的实部转换为 np.float32 类型后进行快速傅里叶变换，结果存入 y1
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)  # 对 x 的实部转换为 np.float64 类型后进行快速傅里叶变换，并转换为 np.complex64 类型，结果存入 y2

        assert_equal(y1.dtype, np.complex64)  # 断言 y1 的数据类型为 np.complex64
        assert_array_almost_equal_nulp(y1, y2, 2000)  # 断言 y1 和 y2 的近似相等性，允许误差为 2000 个单位最低单位

    # 定义 test_definition_float16 方法
    def test_definition_float16(self):
        x = [[1, 2, 3],           # 定义二维列表 x
             [4, 5, 6],
             [7, 8, 9]]
        y = fftn(np.array(x, np.float16))  # 对 x 进行快速傅里叶变换，存入 y
        assert_equal(y.dtype, np.complex64)  # 断言 y 的数据类型为 np.complex64
        y_r = np.array(fftn(x), np.complex64)  # 对 x 进行快速傅里叶变换，结果存入 y_r
        assert_array_almost_equal_nulp(y, y_r)  # 断言 y 和 y_r 的近似相等性

    # 使用 pytest.mark.parametrize 装饰器，参数为 SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES
    def test_float16_input_small(self, size):
        x = np.random.rand(size, size) + 1j*np.random.rand(size, size)  # 生成大小为 size*size 的随机复数矩阵 x
        y1 = fftn(x.real.astype(np.float16))  # 对 x 的实部转换为 np.float16 类型后进行快速傅里叶变换，结果存入 y1
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)  # 对 x 的实部转换为 np.float64 类型后进行快速傅里叶变换，并转换为 np.complex64 类型，结果存入 y2

        assert_equal(y1.dtype, np.complex64)  # 断言 y1 的数据类型为 np.complex64
        assert_array_almost_equal_nulp(y1, y2, 5e5)  # 断言 y1 和 y2 的近似相等性，允许误差为 5e5 个单位最低单位

    # 使用 pytest.mark.parametrize 装饰器，参数为 LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES
    def test_float16_input_large(self, size):
    # 定义一个测试方法，用于测试对大型输入的复数 FFT 运算
    def test_float16_input_large(self, size):
        # 生成一个大小为 size x 3 的随机复数数组，实部为 np.float16 类型，虚部为 np.float64 类型
        x = np.random.rand(size, 3) + 1j*np.random.rand(size, 3)
        
        # 对 x 的实部进行 FFT 运算，结果为 np.float16 类型的复数数组
        y1 = fftn(x.real.astype(np.float16))
        
        # 对 x 的实部进行 FFT 运算，结果为 np.float64 类型的复数数组，然后转换为 np.complex64 类型
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)
        
        # 断言 y1 的数据类型为 np.complex64
        assert_equal(y1.dtype, np.complex64)
        
        # 检查 y1 和 y2 之间的数值近似误差是否在 2e6 以内
        assert_array_almost_equal_nulp(y1, y2, 2e6)
class TestFftn:
    # 在每个测试方法执行前种下随机种子，确保可重复性
    def setup_method(self):
        np.random.seed(1234)

    # 测试 fftn 函数的基本定义
    def test_definition(self):
        # 测试对小矩阵 x 的 FFT 和直接 DFT 的近似一致性
        x = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        y = fftn(x)
        assert_array_almost_equal(y, direct_dftn(x))

        # 测试对随机生成的大矩阵 x 的 FFT 和直接 DFT 的近似一致性
        x = random((20, 26))
        assert_array_almost_equal(fftn(x), direct_dftn(x))

        # 测试对多维随机生成的大矩阵 x 的 FFT 和直接 DFT 的近似一致性
        x = random((5, 4, 3, 20))
        assert_array_almost_equal(fftn(x), direct_dftn(x))

    # 测试 fftn 函数的 shape 参数
    def test_shape_argument(self):
        # 测试对小矩阵 small_x 使用不同 shape 的 FFT
        small_x = [[1, 2, 3],
                   [4, 5, 6]]
        large_x1 = [[1, 2, 3, 0],
                    [4, 5, 6, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]]
        y = fftn(small_x, s=(4, 4))
        assert_array_almost_equal(y, fftn(large_x1))

        # 再次测试对 small_x 使用不同 shape 的 FFT
        y = fftn(small_x, s=(3, 4))
        assert_array_almost_equal(y, fftn(large_x1[:-1]))

    # 测试 fftn 函数的 shape 和 axes 参数
    def test_shape_axes_argument(self):
        # 测试对 small_x 使用不同 shape 和 axes 的 FFT
        small_x = [[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]
        large_x1 = array([[1, 2, 3, 0],
                          [4, 5, 6, 0],
                          [7, 8, 9, 0],
                          [0, 0, 0, 0]])
        y = fftn(small_x, s=(4, 4), axes=(-2, -1))
        assert_array_almost_equal(y, fftn(large_x1))

        # 再次测试对 small_x 使用不同 shape 和不同排列 axes 的 FFT
        y = fftn(small_x, s=(4, 4), axes=(-1, -2))
        assert_array_almost_equal(y, swapaxes(
            fftn(swapaxes(large_x1, -1, -2)), -1, -2))

    # 测试 fftn 函数的 axes 参数在不同情况下的应用
    def test_shape_axes_argument2(self):
        # 改变最后一个轴的形状
        x = numpy.random.random((10, 5, 3, 7))
        y = fftn(x, axes=(-1,), s=(8,))
        assert_array_almost_equal(y, fft(x, axis=-1, n=8))

        # 改变非最后一个轴的形状
        x = numpy.random.random((10, 5, 3, 7))
        y = fftn(x, axes=(-2,), s=(8,))
        assert_array_almost_equal(y, fft(x, axis=-2, n=8))

        # 改变多个轴的形状：参考 #244，其中混淆了形状和轴
        x = numpy.random.random((4, 4, 2))
        y = fftn(x, axes=(-3, -2), s=(8, 8))
        assert_array_almost_equal(y,
                                  numpy.fft.fftn(x, axes=(-3, -2), s=(8, 8)))

    # 测试 fftn 函数在提供无效形状时是否抛出异常
    def test_shape_argument_more(self):
        x = zeros((4, 4, 2))
        with assert_raises(ValueError,
                           match="shape requires more axes than are present"):
            fftn(x, s=(8, 8, 2, 1))

    # 测试 fftn 函数在提供无效大小时是否抛出异常
    def test_invalid_sizes(self):
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[1, 0\]\) specified"):
            fftn([[]])

        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[4, -3\]\) specified"):
            fftn([[1, 1], [2, 2]], (4, -3))

    # 测试在不提供 axes 参数时 fftn 函数的行为
    def test_no_axes(self):
        x = numpy.random.random((2, 2, 2))
        assert_allclose(fftn(x, axes=[]), x, atol=1e-7)
    def test_regression_244(self):
        """测试回归问题244。

        FFT 在使用轴参数时返回错误结果。
        """
        # fftn（以及因此 fft2）在同时使用轴和形状时曾经出现故障
        x = numpy.ones((4, 4, 2))
        # 对 x 进行 FFT 变换，指定新的形状为 (8, 8)，指定轴为 (-3, -2)
        y = fftn(x, s=(8, 8), axes=(-3, -2))
        # 使用 numpy.fft.fftn 函数对比验证结果
        y_r = numpy.fft.fftn(x, s=(8, 8), axes=(-3, -2))
        # 使用 assert_allclose 函数检查 y 和 y_r 是否接近
        assert_allclose(y, y_r)
# 定义名为 TestIfftn 的测试类
class TestIfftn:
    # 类变量，用于存储数据类型和复数数据类型
    dtype = None
    cdtype = None

    # 设置测试方法的初始化方法
    def setup_method(self):
        # 设置随机数种子为 1234
        np.random.seed(1234)

    # 使用参数化装饰器定义测试方法，参数为 dtype, cdtype, maxnlp 的多组取值
    def test_definition(self, dtype, cdtype, maxnlp):
        # 创建一个二维数组 x，指定数据类型为 dtype
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=dtype)
        # 对 x 进行逆离散傅里叶变换
        y = ifftn(x)
        # 断言 y 的数据类型为 cdtype
        assert_equal(y.dtype, cdtype)
        # 使用 almost_equal_nulp 函数检查 y 和 direct_idftn(x) 的近似程度
        assert_array_almost_equal_nulp(y, direct_idftn(x), maxnlp)

        # 创建一个随机数组 x，并使用 ifftn 函数进行逆离散傅里叶变换，再与 direct_idftn(x) 进行比较
        x = random((20, 26))
        assert_array_almost_equal_nulp(ifftn(x), direct_idftn(x), maxnlp)

        # 创建一个随机数组 x，并使用 ifftn 函数进行逆离散傅里叶变换，再与 direct_idftn(x) 进行比较
        x = random((5, 4, 3, 20))
        assert_array_almost_equal_nulp(ifftn(x), direct_idftn(x), maxnlp)

    # 使用参数化装饰器定义测试方法，参数为 maxnlp 的多组取值
    # size 参数在另一参数化装饰器中定义，会与 size 参数化装饰器共同形成多组测试
    def test_random_complex(self, maxnlp, size):
        # 创建一个复数随机数组 x，并使用 ifftn 函数进行逆离散傅里叶变换，再与 x 进行比较
        x = random([size, size]) + 1j * random([size, size])
        assert_array_almost_equal_nulp(ifftn(fftn(x)), x, maxnlp)
        assert_array_almost_equal_nulp(fftn(ifftn(x)), x, maxnlp)

    # 测试无效尺寸情况
    def test_invalid_sizes(self):
        # 使用 assert_raises 函数检测 ValueError 异常，验证 ifftn([[]]) 的无效输入
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[1, 0\]\) specified"):
            ifftn([[]])

        # 使用 assert_raises 函数检测 ValueError 异常，验证 ifftn([[1, 1], [2, 2]], (4, -3)) 的无效输入
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[4, -3\]\) specified"):
            ifftn([[1, 1], [2, 2]], (4, -3))

    # 测试无轴情况
    def test_no_axes(self):
        # 创建一个形状为 (2,2,2) 的随机数组 x，并使用 ifftn 函数进行逆离散傅里叶变换，再与 x 进行比较
        x = numpy.random.random((2, 2, 2))
        assert_allclose(ifftn(x, axes=[]), x, atol=1e-7)


# 定义名为 TestRfftn 的测试类
class TestRfftn:
    # 类变量，用于存储数据类型和复数数据类型
    dtype = None
    cdtype = None

    # 设置测试方法的初始化方法
    def setup_method(self):
        # 设置随机数种子为 1234
        np.random.seed(1234)

    # 使用参数化装饰器定义测试方法，参数为 dtype, cdtype, maxnlp 的多组取值
    def test_definition(self, dtype, cdtype, maxnlp):
        # 创建一个二维数组 x，指定数据类型为 dtype
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=dtype)
        # 对 x 进行实部快速傅里叶变换
        y = rfftn(x)
        # 断言 y 的数据类型为 cdtype
        assert_equal(y.dtype, cdtype)
        # 使用 almost_equal_nulp 函数检查 y 和 direct_rdftn(x) 的近似程度
        assert_array_almost_equal_nulp(y, direct_rdftn(x), maxnlp)

        # 创建一个随机数组 x，并使用 rfftn 函数进行实部快速傅里叶变换，再与 direct_rdftn(x) 进行比较
        x = random((20, 26))
        assert_array_almost_equal_nulp(rfftn(x), direct_rdftn(x), maxnlp)

        # 创建一个随机数组 x，并使用 rfftn 函数进行实部快速傅里叶变换，再与 direct_rdftn(x) 进行比较
        x = random((5, 4, 3, 20))
        assert_array_almost_equal_nulp(rfftn(x), direct_rdftn(x), maxnlp)

    # 使用参数化装饰器定义测试方法，参数为 size 的多组取值
    def test_random(self, size):
        # 创建一个二维数组 x，其中元素为随机数
        x = random([size, size])
        # 使用 irfftn 函数进行逆实部快速傅里叶变换，并与 x 进行比较
        assert_allclose(irfftn(rfftn(x), x.shape), x, atol=1e-10)
    # 定义一个测试方法，用于测试给定的函数对于无效大小的输入是否能正确引发异常
    def test_invalid_sizes(self, func):
        # 使用断言上下文管理器，检查传入空列表 [[]] 是否会引发 ValueError 异常，并验证异常消息
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[1, 0\]\) specified"):
            func([[]])

        # 使用断言上下文管理器，检查传入无效大小的列表 [[1, 1], [2, 2]] 是否会引发 ValueError 异常，并验证异常消息
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[4, -3\]\) specified"):
            func([[1, 1], [2, 2]], (4, -3))

    # 使用 pytest 的参数化装饰器，为 rfftn 和 irfftn 函数分别执行相同的测试用例
    @pytest.mark.parametrize('func', [rfftn, irfftn])
    # 定义一个测试方法，用于测试在没有指定轴的情况下，函数是否能正确引发 ValueError 异常
    def test_no_axes(self, func):
        # 使用断言上下文管理器，检查传入空数组 [] 是否会引发 ValueError 异常，并验证异常消息
        with assert_raises(ValueError,
                           match="at least 1 axis must be transformed"):
            func([], axes=[])

    # 定义一个测试方法，用于测试对于复数输入是否能正确引发 TypeError 异常
    def test_complex_input(self):
        # 使用断言上下文管理器，检查传入复数数组 np.zeros(10, dtype=np.complex64) 是否会引发 TypeError 异常，并验证异常消息
        with assert_raises(TypeError, match="x must be a real sequence"):
            rfftn(np.zeros(10, dtype=np.complex64))
class FakeArray:
    # 初始化方法，接受一个参数 data，并将其保存在实例变量 _data 中
    def __init__(self, data):
        self._data = data
        # 将 data 对象的 __array_interface__ 属性赋值给实例的 __array_interface__ 属性
        self.__array_interface__ = data.__array_interface__


class FakeArray2:
    # 初始化方法，接受一个参数 data，并将其保存在实例变量 _data 中
    def __init__(self, data):
        self._data = data

    # 返回实例变量 _data，即 data 参数
    def __array__(self, dtype=None, copy=None):
        return self._data

# TODO: Is this test actually valuable? The behavior it's testing shouldn't be
# relied upon by users except for overwrite_x = False
class TestOverwrite:
    """Check input overwrite behavior of the FFT functions."""

    # 支持的实数数据类型列表
    real_dtypes = [np.float32, np.float64, np.longdouble]
    # 支持的数据类型列表，包括实数和复数
    dtypes = real_dtypes + [np.complex64, np.complex128, np.clongdouble]
    # FFT 大小的列表
    fftsizes = [8, 16, 32]

    # 检查函数，用于测试 FFT 函数的输入覆盖行为
    def _check(self, x, routine, fftsize, axis, overwrite_x, should_overwrite):
        # 复制 x，并将结果保存在 x2 中
        x2 = x.copy()
        # 遍历三种假数据类型 FakeArray, FakeArray2 和 lambda x: x
        for fake in [lambda x: x, FakeArray, FakeArray2]:
            # 调用 FFT 函数（routine），传入假数据类型的实例 x2 以及其他参数
            routine(fake(x2), fftsize, axis, overwrite_x=overwrite_x)

            # 构建包含函数调用信息的字符串 sig
            sig = "{}({}{!r}, {!r}, axis={!r}, overwrite_x={!r})".format(
                routine.__name__, x.dtype, x.shape, fftsize, axis, overwrite_x)
            # 如果不应该发生覆盖，断言 x2 等于 x，错误消息为 %s 中的意外覆盖
            if not should_overwrite:
                assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)

    # 一维数组检查函数
    def _check_1d(self, routine, dtype, shape, axis, overwritable_dtypes,
                  fftsize, overwrite_x):
        # 设定随机种子
        np.random.seed(1234)
        # 根据数据类型生成随机数据
        if np.issubdtype(dtype, np.complexfloating):
            data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
        else:
            data = np.random.randn(*shape)
        data = data.astype(dtype)

        # 是否应该覆盖的条件
        should_overwrite = (overwrite_x
                            and dtype in overwritable_dtypes
                            and fftsize <= shape[axis])
        # 调用 _check 方法，传入随机生成的数据和其他参数
        self._check(data, routine, fftsize, axis,
                    overwrite_x=overwrite_x,
                    should_overwrite=should_overwrite)

    # 使用参数化测试框架标记的 FFT 和 IFFT 测试方法
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('fftsize', fftsizes)
    @pytest.mark.parametrize('overwrite_x', [True, False])
    @pytest.mark.parametrize('shape,axes', [((16,), -1),
                                            ((16, 2), 0),
                                            ((2, 16), 1)])
    def test_fft_ifft(self, dtype, fftsize, overwrite_x, shape, axes):
        overwritable = (np.clongdouble, np.complex128, np.complex64)
        # 调用 _check_1d 方法，测试 FFT 和 IFFT 函数
        self._check_1d(fft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)
        self._check_1d(ifft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)

    # 使用参数化测试框架标记的实数数据类型的 FFT 和 IFFT 测试方法
    @pytest.mark.parametrize('dtype', real_dtypes)
    @pytest.mark.parametrize('fftsize', fftsizes)
    @pytest.mark.parametrize('overwrite_x', [True, False])
    @pytest.mark.parametrize('shape,axes', [((16,), -1),
                                            ((16, 2), 0),
                                            ((2, 16), 1)])
    # 定义一个测试函数，用于测试实部傅里叶变换及其逆变换
    def test_rfft_irfft(self, dtype, fftsize, overwrite_x, shape, axes):
        # 获取支持实数类型的数据类型列表
        overwritable = self.real_dtypes
        # 调用内部函数检查一维逆傅里叶变换结果
        self._check_1d(irfft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)
        # 调用内部函数检查一维傅里叶变换结果
        self._check_1d(rfft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)

    # 内部方法，用于检查一个多维数组的傅里叶变换的正确性
    def _check_nd_one(self, routine, dtype, shape, axes, overwritable_dtypes,
                      overwrite_x):
        # 设置随机种子以确保每次生成的随机数据一致
        np.random.seed(1234)
        # 如果数据类型是复数类型，则生成实部和虚部都是随机的复数数组
        if np.issubdtype(dtype, np.complexfloating):
            data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
        else:
            # 否则生成实数类型的随机数组
            data = np.random.randn(*shape)
        # 将生成的随机数组转换为指定的数据类型
        data = data.astype(dtype)

        # 生成一个迭代器，用于遍历各种傅里叶变换形状
        def fftshape_iter(shp):
            if len(shp) <= 0:
                yield ()
            else:
                # 对于每个维度，生成该维度大小的一半、大小和两倍的三种变换形状
                for j in (shp[0]//2, shp[0], shp[0]*2):
                    for rest in fftshape_iter(shp[1:]):
                        yield (j,) + rest

        # 根据指定的轴返回数据的部分形状
        def part_shape(shape, axes):
            if axes is None:
                return shape
            else:
                return tuple(np.take(shape, axes))

        # 确定是否应该重写输入数据
        def should_overwrite(data, shape, axes):
            s = part_shape(data.shape, axes)
            return (overwrite_x and
                    np.prod(shape) <= np.prod(s)
                    and dtype in overwritable_dtypes)

        # 对每种可能的傅里叶变换形状进行测试
        for fftshape in fftshape_iter(part_shape(shape, axes)):
            # 调用内部函数检查傅里叶变换结果的正确性
            self._check(data, routine, fftshape, axes,
                        overwrite_x=overwrite_x,
                        should_overwrite=should_overwrite(data, fftshape, axes))
            # 如果数据的维度大于1，还需检查Fortran顺序的情况
            if data.ndim > 1:
                # 检查Fortran顺序的傅里叶变换结果的正确性
                self._check(data.T, routine, fftshape, axes,
                            overwrite_x=overwrite_x,
                            should_overwrite=should_overwrite(
                                data.T, fftshape, axes))

    # 使用pytest的参数化标记，对不同的数据类型、是否覆盖输入数据以及不同的形状与轴组合进行参数化测试
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
    # 定义一个测试方法，用于测试 FFTN 和 IFFTN 函数
    def test_fftn_ifftn(self, dtype, overwrite_x, shape, axes):
        # 定义一个列表，包含可以被覆写的数据类型
        overwritable = (np.clongdouble, np.complex128, np.complex64)
        # 调用 _check_nd_one 方法，用于检查 fftn 函数的参数和行为
        self._check_nd_one(fftn, dtype, shape, axes, overwritable,
                           overwrite_x)
        # 调用 _check_nd_one 方法，用于检查 ifftn 函数的参数和行为
        self._check_nd_one(ifftn, dtype, shape, axes, overwritable,
                           overwrite_x)
# 使用 pytest 的 mark.parametrize 装饰器，参数化函数列表 [fft, ifft, fftn, ifftn, rfft, irfft, rfftn, irfftn]
@pytest.mark.parametrize('func', [fft, ifft, fftn, ifftn,
                                 rfft, irfft, rfftn, irfftn])
def test_invalid_norm(func):
    # 创建一个长度为 10 的浮点数数组
    x = np.arange(10, dtype=float)
    # 使用 assert_raises 上下文管理器，检查是否抛出 ValueError 异常
    with assert_raises(ValueError,
                       match='Invalid norm value \'o\', should be'
                             ' "backward", "ortho" or "forward"'):
        # 调用 func 函数，传入数组 x 和非法的 norm 参数 'o'
        func(x, norm='o')


# 使用 pytest 的 mark.parametrize 装饰器，参数化函数列表 [fft, ifft, fftn, ifftn, irfft, irfftn, hfft, hfftn]
@pytest.mark.parametrize('func', [fft, ifft, fftn, ifftn,
                                   irfft, irfftn, hfft, hfftn])
def test_swapped_byte_order_complex(func):
    # 创建一个随机数生成器，种子为 1234
    rng = np.random.RandomState(1234)
    # 创建一个长度为 10 的复数数组，实部和虚部均为随机数
    x = rng.rand(10) + 1j * rng.rand(10)
    # 断言调用 swap_byteorder 函数后，func 函数对原始数组 x 和字节顺序交换后的数组的输出近似相等
    assert_allclose(func(swap_byteorder(x)), func(x))


# 使用 pytest 的 mark.parametrize 装饰器，参数化函数列表 [ihfft, ihfftn, rfft, rfftn]
@pytest.mark.parametrize('func', [ihfft, ihfftn, rfft, rfftn])
def test_swapped_byte_order_real(func):
    # 创建一个随机数生成器，种子为 1234
    rng = np.random.RandomState(1234)
    # 创建一个长度为 10 的实数数组
    x = rng.rand(10)
    # 断言调用 swap_byteorder 函数后，func 函数对原始数组 x 和字节顺序交换后的数组的输出近似相等
    assert_allclose(func(swap_byteorder(x)), func(x))
```