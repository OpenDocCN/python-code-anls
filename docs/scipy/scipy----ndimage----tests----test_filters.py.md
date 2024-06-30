# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_filters.py`

```
# 导入必要的库和模块，包括函数式编程工具、迭代工具、数学函数、numpy 库及其测试相关模块等
''' Some tests for filters '''
import functools
import itertools
import math
import numpy as np

# 导入 numpy 的测试工具函数和断言方法
from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_almost_equal,
                           assert_array_equal, assert_almost_equal,
                           suppress_warnings, assert_)
# 导入 pytest 测试框架及其异常处理方法
import pytest
from pytest import raises as assert_raises

# 导入 scipy 的图像处理模块及其部分滤波器函数
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d

# 导入当前包中的类型定义及相关类型声明
from . import types, float_types, complex_types


# 定义一个计算两个数组平方和平方根的函数
def sumsq(a, b):
    return math.sqrt(((a - b)**2).sum())


# 定义一个用于执行复数相关（或卷积）操作的工具函数
def _complex_correlate(array, kernel, real_dtype, convolve=False,
                       mode="reflect", cval=0, ):
    """Utility to perform a reference complex-valued convolutions.

    When convolve==False, correlation is performed instead
    """
    # 将输入的数组和卷积核转换为 numpy 数组
    array = np.asarray(array)
    kernel = np.asarray(kernel)
    # 检查数组和卷积核是否为复数类型
    complex_array = array.dtype.kind == 'c'
    complex_kernel = kernel.dtype.kind == 'c'
    
    # 根据数组维度选择合适的 scipy.ndimage 函数
    if array.ndim == 1:
        func = ndimage.convolve1d if convolve else ndimage.correlate1d
    else:
        func = ndimage.convolve if convolve else ndimage.correlate
    
    # 如果是相关操作，需对卷积核进行共轭处理
    if not convolve:
        kernel = kernel.conj()
    
    # 根据数组和卷积核的复杂性执行相关或卷积操作，并根据情况添加虚部
    if complex_array and complex_kernel:
        # 使用给定的实部和虚部值进行相关或卷积操作
        output = (
            func(array.real, kernel.real, output=real_dtype,
                 mode=mode, cval=np.real(cval)) -
            func(array.imag, kernel.imag, output=real_dtype,
                 mode=mode, cval=np.imag(cval)) +
            1j * func(array.imag, kernel.real, output=real_dtype,
                      mode=mode, cval=np.imag(cval)) +
            1j * func(array.real, kernel.imag, output=real_dtype,
                      mode=mode, cval=np.real(cval))
        )
    elif complex_array:
        # 只有数组是复数时，执行相关或卷积操作
        output = (
            func(array.real, kernel, output=real_dtype, mode=mode,
                 cval=np.real(cval)) +
            1j * func(array.imag, kernel, output=real_dtype, mode=mode,
                      cval=np.imag(cval))
        )
    elif complex_kernel:
        # 只有卷积核是复数时，执行相关或卷积操作
        output = (
            func(array, kernel.real, output=real_dtype, mode=mode, cval=cval) +
            1j * func(array, kernel.imag, output=real_dtype, mode=mode,
                      cval=cval)
        )
    return output


# 定义一个生成滤波器函数、有效关键字参数和导致大小不匹配的关键字-值对的函数
def _cases_axes_tuple_length_mismatch():
    # 使用高斯滤波器作为基础滤波器函数
    filter_func = ndimage.gaussian_filter
    kwargs = dict(radius=3, mode='constant', sigma=1.0, order=0)
    
    # 遍历关键字参数，生成导致大小不匹配的情况
    for key, val in kwargs.items():
        yield filter_func, kwargs, key, val
    
    # 使用均匀、最小和最大滤波器作为基础滤波器函数
    filter_funcs = [ndimage.uniform_filter, ndimage.minimum_filter,
                    ndimage.maximum_filter]
    kwargs = dict(size=3, mode='constant', origin=0)
    # 对于每个过滤函数列表中的过滤函数，依次进行操作
    for filter_func in filter_funcs:
        # 遍历关键字参数字典中的每个键值对
        for key, val in kwargs.items():
            # 使用生成器 yield 返回每个过滤函数、关键字参数字典本身、当前处理的键和值
            yield filter_func, kwargs, key, val
class TestNdimageFilters:

    def _validate_complex(self, array, kernel, type2, mode='reflect', cval=0):
        # utility for validating complex-valued correlations
        
        # Determine the real part dtype of type2 and assign it to real_dtype
        real_dtype = np.asarray([], dtype=type2).real.dtype
        
        # Compute the expected complex correlation using _complex_correlate function
        expected = _complex_correlate(
            array, kernel, real_dtype, convolve=False, mode=mode, cval=cval
        )

        # Depending on the dimensionality of array, define correlate and convolve functions
        if array.ndim == 1:
            correlate = functools.partial(ndimage.correlate1d, axis=-1,
                                          mode=mode, cval=cval)
            convolve = functools.partial(ndimage.convolve1d, axis=-1,
                                         mode=mode, cval=cval)
        else:
            correlate = functools.partial(ndimage.correlate, mode=mode,
                                          cval=cval)
            convolve = functools.partial(ndimage.convolve, mode=mode,
                                         cval=cval)

        # Test correlate function output dtype
        output = correlate(array, kernel, output=type2)
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)

        # Test correlate with pre-allocated output
        output = np.zeros_like(array, dtype=type2)
        correlate(array, kernel, output=output)
        assert_array_almost_equal(expected, output)

        # Test convolve function output dtype
        output = convolve(array, kernel, output=type2)
        expected = _complex_correlate(
            array, kernel, real_dtype, convolve=True, mode=mode, cval=cval,
        )
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)

        # Test convolve with pre-allocated output
        convolve(array, kernel, output=output)
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)

        # Issue a warning if the output dtype is not complex
        with pytest.warns(UserWarning,
                          match="promoting specified output dtype to complex"):
            correlate(array, kernel, output=real_dtype)

        with pytest.warns(UserWarning,
                          match="promoting specified output dtype to complex"):
            convolve(array, kernel, output=real_dtype)

        # Raise a RuntimeError if output array is provided but not complex-valued
        output_real = np.zeros_like(array, dtype=real_dtype)
        with assert_raises(RuntimeError):
            correlate(array, kernel, output=output_real)

        with assert_raises(RuntimeError):
            convolve(array, kernel, output=output_real)
    # 定义测试函数 test_correlate01，用于测试 ndimage 库中的相关函数
    def test_correlate01(self):
        # 创建包含两个元素的 NumPy 数组
        array = np.array([1, 2])
        # 创建包含一个元素的 NumPy 数组作为权重
        weights = np.array([2])
        # 预期的输出结果
        expected = [2, 4]

        # 对数组 array 使用权重 weights 进行相关操作，返回结果存储在 output 中
        output = ndimage.correlate(array, weights)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

        # 对数组 array 使用权重 weights 进行卷积操作，返回结果存储在 output 中
        output = ndimage.convolve(array, weights)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

        # 对数组 array 使用权重 weights 进行一维相关操作，返回结果存储在 output 中
        output = ndimage.correlate1d(array, weights)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

        # 对数组 array 使用权重 weights 进行一维卷积操作，返回结果存储在 output 中
        output = ndimage.convolve1d(array, weights)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

    # 定义测试函数 test_correlate01_overlap，用于测试带有重叠的一维相关操作
    def test_correlate01_overlap(self):
        # 创建一个 16x16 的二维数组
        array = np.arange(256).reshape(16, 16)
        # 创建包含一个元素的 NumPy 数组作为权重
        weights = np.array([2])
        # 预期的输出结果是原始数组的每个元素乘以2
        expected = 2 * array

        # 对数组 array 使用权重 weights 进行一维相关操作，将结果直接存储回 array
        ndimage.correlate1d(array, weights, output=array)
        # 断言数组 array 的内容与预期结果几乎相等
        assert_array_almost_equal(array, expected)

    # 定义测试函数 test_correlate02，用于测试另一组数组和核的相关操作
    def test_correlate02(self):
        # 创建包含三个元素的 NumPy 数组
        array = np.array([1, 2, 3])
        # 创建包含一个元素的 NumPy 数组作为核
        kernel = np.array([1])

        # 对数组 array 使用核 kernel 进行相关操作，返回结果存储在 output 中
        output = ndimage.correlate(array, kernel)
        # 断言输出结果与数组 array 几乎相等
        assert_array_almost_equal(array, output)

        # 对数组 array 使用核 kernel 进行卷积操作，返回结果存储在 output 中
        output = ndimage.convolve(array, kernel)
        # 断言输出结果与数组 array 几乎相等
        assert_array_almost_equal(array, output)

        # 对数组 array 使用核 kernel 进行一维相关操作，返回结果存储在 output 中
        output = ndimage.correlate1d(array, kernel)
        # 断言输出结果与数组 array 几乎相等
        assert_array_almost_equal(array, output)

        # 对数组 array 使用核 kernel 进行一维卷积操作，返回结果存储在 output 中
        output = ndimage.convolve1d(array, kernel)
        # 断言输出结果与数组 array 几乎相等
        assert_array_almost_equal(array, output)

    # 定义测试函数 test_correlate03，用于测试另一组数组和权重的相关操作
    def test_correlate03(self):
        # 创建包含一个元素的 NumPy 数组
        array = np.array([1])
        # 创建包含两个元素的 NumPy 数组作为权重
        weights = np.array([1, 1])
        # 预期的输出结果
        expected = [2]

        # 对数组 array 使用权重 weights 进行相关操作，返回结果存储在 output 中
        output = ndimage.correlate(array, weights)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

        # 对数组 array 使用权重 weights 进行卷积操作，返回结果存储在 output 中
        output = ndimage.convolve(array, weights)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

        # 对数组 array 使用权重 weights 进行一维相关操作，返回结果存储在 output 中
        output = ndimage.correlate1d(array, weights)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

        # 对数组 array 使用权重 weights 进行一维卷积操作，返回结果存储在 output 中
        output = ndimage.convolve1d(array, weights)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

    # 定义测试函数 test_correlate04，用于测试另一组数组、权重和预期结果的相关操作
    def test_correlate04(self):
        # 创建包含两个元素的 NumPy 数组
        array = np.array([1, 2])
        # 创建包含两个元素的预期相关结果数组
        tcor = [2, 3]
        # 创建包含两个元素的预期卷积结果数组
        tcov = [3, 4]
        # 创建包含两个元素的 NumPy 数组作为权重
        weights = np.array([1, 1])

        # 对数组 array 使用权重 weights 进行相关操作，返回结果存储在 output 中
        output = ndimage.correlate(array, weights)
        # 断言输出结果与预期相关结果 tcor 几乎相等
        assert_array_almost_equal(output, tcor)
        
        # 对数组 array 使用权重 weights 进行卷积操作，返回结果存储在 output 中
        output = ndimage.convolve(array, weights)
        # 断言输出结果与预期卷积结果 tcov 几乎相等
        assert_array_almost_equal(output, tcov)

        # 对数组 array 使用权重 weights 进行一维相关操作，返回结果存储在 output 中
        output = ndimage.correlate1d(array, weights)
        # 断言输出结果与预期相关结果 tcor 几乎相等
        assert_array_almost_equal(output, tcor)

        # 对数组 array 使用权重 weights 进行一维卷积操作，返回结果存储在 output 中
        output = ndimage.convolve1d(array, weights)
        # 断言输出结果与预期卷积结果 tcov 几乎相等
        assert_array_almost_equal(output, tcov)

    # 定义测试函数 test_correlate05，用于测试另一组数组、核和预期结果的相关操作
    def test_correlate05(self):
        # 创建包含三个元素的 NumPy 数组
        array = np.array([1, 2, 3])
        # 创建包含三个元素的预期相关结果数组
        tcor = [2, 3, 5]
        # 创建包含三个元素的预期卷积结果数组
        tcov = [3, 5, 6]
        # 创建包含两个元素的 NumPy 数组作为核
        kernel = np.array([1, 1])

        # 对数组 array 使用核 kernel 进行相关操作，返回结果存储在 output 中
        output = ndimage.correlate(array, kernel)
        # 断言输出结果与预期相关结果 tcor 几乎相等
        assert_array_almost_equal(tcor, output)

        # 对数组 array 使用核 kernel 进行卷积操作，返回结果存储在 output 中
        output = ndimage.convolve(array, kernel)
        # 断言输出结果与预期卷积结果 tcov 几乎相等
        assert_array_almost_equal(tcov, output)

        # 对数组 array 使用核 kernel 进行一维相关操作，返回结果存
    # 定义测试方法 test_correlate06，用于测试 ndimage 模块中的相关和卷积函数
    def test_correlate06(self):
        # 创建一个包含元素 [1, 2, 3] 的 NumPy 数组
        array = np.array([1, 2, 3])
        # 预期的相关输出结果
        tcor = [9, 14, 17]
        # 预期的卷积输出结果
        tcov = [7, 10, 15]
        # 创建一个包含元素 [1, 2, 3] 的权重数组
        weights = np.array([1, 2, 3])
        # 对输入数组和权重数组进行相关操作，输出结果保存到 output
        output = ndimage.correlate(array, weights)
        # 断言相关输出结果与预期相近
        assert_array_almost_equal(output, tcor)
        # 对输入数组和权重数组进行卷积操作，输出结果保存到 output
        output = ndimage.convolve(array, weights)
        # 断言卷积输出结果与预期相近
        assert_array_almost_equal(output, tcov)
        # 对输入数组和权重数组进行一维相关操作，输出结果保存到 output
        output = ndimage.correlate1d(array, weights)
        # 断言一维相关输出结果与预期相近
        assert_array_almost_equal(output, tcor)
        # 对输入数组和权重数组进行一维卷积操作，输出结果保存到 output
        output = ndimage.convolve1d(array, weights)
        # 断言一维卷积输出结果与预期相近
        assert_array_almost_equal(output, tcov)

    # 定义测试方法 test_correlate07，用于测试 ndimage 模块中的相关和卷积函数
    def test_correlate07(self):
        # 创建一个包含元素 [1, 2, 3] 的 NumPy 数组
        array = np.array([1, 2, 3])
        # 预期的输出结果
        expected = [5, 8, 11]
        # 创建一个包含元素 [1, 2, 1] 的权重数组
        weights = np.array([1, 2, 1])
        # 对输入数组和权重数组进行相关操作，输出结果保存到 output
        output = ndimage.correlate(array, weights)
        # 断言相关输出结果与预期相近
        assert_array_almost_equal(output, expected)
        # 对输入数组和权重数组进行卷积操作，输出结果保存到 output
        output = ndimage.convolve(array, weights)
        # 断言卷积输出结果与预期相近
        assert_array_almost_equal(output, expected)
        # 对输入数组和权重数组进行一维相关操作，输出结果保存到 output
        output = ndimage.correlate1d(array, weights)
        # 断言一维相关输出结果与预期相近
        assert_array_almost_equal(output, expected)
        # 对输入数组和权重数组进行一维卷积操作，输出结果保存到 output
        output = ndimage.convolve1d(array, weights)
        # 断言一维卷积输出结果与预期相近
        assert_array_almost_equal(output, expected)

    # 定义测试方法 test_correlate08，用于测试 ndimage 模块中的相关和卷积函数
    def test_correlate08(self):
        # 创建一个包含元素 [1, 2, 3] 的 NumPy 数组
        array = np.array([1, 2, 3])
        # 预期的相关输出结果
        tcor = [1, 2, 5]
        # 预期的卷积输出结果
        tcov = [3, 6, 7]
        # 创建一个包含元素 [1, 2, -1] 的权重数组
        weights = np.array([1, 2, -1])
        # 对输入数组和权重数组进行相关操作，输出结果保存到 output
        output = ndimage.correlate(array, weights)
        # 断言相关输出结果与预期相近
        assert_array_almost_equal(output, tcor)
        # 对输入数组和权重数组进行卷积操作，输出结果保存到 output
        output = ndimage.convolve(array, weights)
        # 断言卷积输出结果与预期相近
        assert_array_almost_equal(output, tcov)
        # 对输入数组和权重数组进行一维相关操作，输出结果保存到 output
        output = ndimage.correlate1d(array, weights)
        # 断言一维相关输出结果与预期相近
        assert_array_almost_equal(output, tcor)
        # 对输入数组和权重数组进行一维卷积操作，输出结果保存到 output
        output = ndimage.convolve1d(array, weights)
        # 断言一维卷积输出结果与预期相近
        assert_array_almost_equal(output, tcov)

    # 定义测试方法 test_correlate09，用于测试 ndimage 模块中的相关和卷积函数
    def test_correlate09(self):
        # 创建一个空列表 array
        array = []
        # 创建一个包含元素 [1, 1] 的权重数组 kernel
        kernel = np.array([1, 1])
        # 对空列表 array 和权重数组 kernel 进行相关操作，输出结果保存到 output
        output = ndimage.correlate(array, kernel)
        # 断言 array 和 output 结果相近
        assert_array_almost_equal(array, output)
        # 对空列表 array 和权重数组 kernel 进行卷积操作，输出结果保存到 output
        output = ndimage.convolve(array, kernel)
        # 断言 array 和 output 结果相近
        assert_array_almost_equal(array, output)
        # 对空列表 array 和权重数组 kernel 进行一维相关操作，输出结果保存到 output
        output = ndimage.correlate1d(array, kernel)
        # 断言 array 和 output 结果相近
        assert_array_almost_equal(array, output)
        # 对空列表 array 和权重数组 kernel 进行一维卷积操作，输出结果保存到 output
        output = ndimage.convolve1d(array, kernel)
        # 断言 array 和 output 结果相近
        assert_array_almost_equal(array, output)

    # 定义测试方法 test_correlate10，用于测试 ndimage 模块中的相关和卷积函数
    def test_correlate10(self):
        # 创建一个包含一个空列表 [[]] 的 NumPy 数组 array
        array = [[]]
        # 创建一个包含元素 [[1, 1]] 的权重数组 kernel
        kernel = np.array([[1, 1]])
        # 对二维空数组 array 和权重数组 kernel 进行相关操作，输出结果保存到 output
        output = ndimage.correlate(array, kernel)
        # 断言 array 和 output 结果相近
        assert_array_almost_equal(array, output)
        # 对二维空数组 array 和权重数组 kernel 进行卷积操作，输出结果保存到 output
        output = ndimage.convolve(array, kernel)
        # 断言 array 和 output 结果相近
        assert_array_almost_equal(array, output)

    # 定义测试方法 test_correlate11，用于测试 ndimage 模块中的相关和卷积函数
    def test_correlate11(self):
        # 创建一个包含两行三列的二维 NumPy 数组 array
        array = np.array([[1, 2, 3],
                          [4, 5, 6]])
        # 创建一个包含两行一列的二维权重数组 kernel
        kernel = np.array([[1, 1],
                           [1, 1]])
        # 对二维数组 array 和权重数组 kernel 进行相关操作，输出结果保存到 output
        output = ndimage.correlate(array, kernel)
        # 断言 array 和 output 结果相近
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
        # 对二维数组 array 和权重数组 kernel 进行卷积操作，输出结果保存到 output
        output = ndimage.convolve(array, kernel)
        # 断言 array 和 output 结果相近
        assert_array_almost_equal([[12, 16, 18], [18, 22, 24]], output)
    # 定义一个测试方法，用于测试 ndimage.correlate 和 ndimage.convolve 函数
    def test_correlate12(self):
        # 创建一个 2x3 的 NumPy 数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6]])
        # 创建一个 2x2 的 NumPy 数组作为卷积核
        kernel = np.array([[1, 0],
                           [0, 1]])
        # 对数组进行卷积操作，输出结果存储在 output 中
        output = ndimage.correlate(array, kernel)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        # 对数组进行相关操作，输出结果存储在 output 中
        output = ndimage.convolve(array, kernel)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)

    # 使用 pytest 的参数化功能，定义一个测试方法，测试不同数据类型的相关和卷积操作
    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_kernel', types)
    def test_correlate13(self, dtype_array, dtype_kernel):
        # 创建一个 2x2 的 NumPy 数组作为卷积核
        kernel = np.array([[1, 0],
                           [0, 1]])
        # 创建一个指定数据类型的 2x3 NumPy 数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        # 对数组进行相关操作，指定输出的数据类型为 dtype_kernel，结果存储在 output 中
        output = ndimage.correlate(array, kernel, output=dtype_kernel)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        # 断言 output 的数据类型是否与指定的 dtype_kernel 一致
        assert_equal(output.dtype.type, dtype_kernel)

        # 对数组进行卷积操作，指定输出的数据类型为 dtype_kernel，结果存储在 output 中
        output = ndimage.convolve(array, kernel, output=dtype_kernel)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        # 断言 output 的数据类型是否与指定的 dtype_kernel 一致
        assert_equal(output.dtype.type, dtype_kernel)

    # 使用 pytest 的参数化功能，定义一个测试方法，测试不同数据类型的相关和卷积操作
    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate14(self, dtype_array, dtype_output):
        # 创建一个 2x2 的 NumPy 数组作为卷积核
        kernel = np.array([[1, 0],
                           [0, 1]])
        # 创建一个指定数据类型的 2x3 NumPy 数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        # 创建一个与 array 相同形状和指定数据类型的全零数组作为输出
        output = np.zeros(array.shape, dtype_output)
        # 对数组进行相关操作，输出结果存储在预先创建的 output 中
        ndimage.correlate(array, kernel, output=output)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        # 断言 output 的数据类型是否与指定的 dtype_output 一致
        assert_equal(output.dtype.type, dtype_output)

        # 对数组进行卷积操作，输出结果存储在预先创建的 output 中
        ndimage.convolve(array, kernel, output=output)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        # 断言 output 的数据类型是否与指定的 dtype_output 一致
        assert_equal(output.dtype.type, dtype_output)

    # 使用 pytest 的参数化功能，定义一个测试方法，测试不同数据类型的相关和卷积操作
    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate15(self, dtype_array):
        # 创建一个 2x2 的 NumPy 数组作为卷积核
        kernel = np.array([[1, 0],
                           [0, 1]])
        # 创建一个指定数据类型的 2x3 NumPy 数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        # 对数组进行相关操作，指定输出的数据类型为 np.float32，结果存储在 output 中
        output = ndimage.correlate(array, kernel, output=np.float32)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        # 断言 output 的数据类型是否为 np.float32
        assert_equal(output.dtype.type, np.float32)

        # 对数组进行卷积操作，指定输出的数据类型为 np.float32，结果存储在 output 中
        output = ndimage.convolve(array, kernel, output=np.float32)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        # 断言 output 的数据类型是否为 np.float32
        assert_equal(output.dtype.type, np.float32)

    # 使用 pytest 的参数化功能，定义一个测试方法，测试不同数据类型的相关和卷积操作
    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate16(self, dtype_array):
        # 创建一个 2x2 的 NumPy 数组作为卷积核
        kernel = np.array([[1, 0],
                           [0, 1]])
        # 创建一个指定数据类型的 2x3 NumPy 数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        # 对数组进行相关操作，指定输出的数据类型为 np.float64，结果存储在 output 中
        output = ndimage.correlate(array, kernel, output=np.float64)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        # 断言 output 的数据类型是否为 np.float64
        assert_equal(output.dtype.type, np.float64)

        # 对数组进行卷积操作，指定输出的数据类型为 np.float64，结果存储在 output 中
        output = ndimage.convolve(array, kernel, output=np.float64)
        # 断言 output 是否与期望的数组几乎相等
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        # 断言 output 的数据类型是否为 np.float64
        assert_equal(output.dtype.type, np.float64)
    # 使用参数 dtype_array 来测试函数 test_correlate16，传入一个特定的 NumPy 数组类型
    def test_correlate16(self, dtype_array):
        # 定义一个 2x2 的卷积核 kernel
        kernel = np.array([[0.5, 0],
                           [0, 0.5]])
        # 创建一个 2x3 的数组 array，使用传入的 dtype_array 类型
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype_array)
        # 对数组 array 进行二维相关操作，结果存储在 output 中，输出类型为 np.float32
        output = ndimage.correlate(array, kernel, output=np.float32)
        # 断言 output 与预期的值近似相等
        assert_array_almost_equal([[1, 1.5, 2.5], [2.5, 3, 4]], output)
        # 断言 output 的数据类型为 np.float32

        # 对数组 array 进行二维卷积操作，结果存储在 output 中，输出类型为 np.float32
        output = ndimage.convolve(array, kernel, output=np.float32)
        # 断言 output 与预期的值近似相等
        assert_array_almost_equal([[3, 4, 4.5], [4.5, 5.5, 6]], output)
        # 断言 output 的数据类型为 np.float32

    # 无参数的测试函数 test_correlate17
    def test_correlate17(self):
        # 创建一个长度为 3 的一维数组 array
        array = np.array([1, 2, 3])
        # 预期的二维相关操作的结果 tcor
        tcor = [3, 5, 6]
        # 预期的二维卷积操作的结果 tcov
        tcov = [2, 3, 5]
        # 定义一个一维卷积核 kernel
        kernel = np.array([1, 1])
        # 对数组 array 进行一维相关操作，结果存储在 output 中，起始位置为 -1
        output = ndimage.correlate(array, kernel, origin=-1)
        # 断言 output 与预期的 tcor 近似相等
        assert_array_almost_equal(tcor, output)
        # 对数组 array 进行一维卷积操作，结果存储在 output 中，起始位置为 -1
        output = ndimage.convolve(array, kernel, origin=-1)
        # 断言 output 与预期的 tcov 近似相等
        assert_array_almost_equal(tcov, output)
        # 对数组 array 进行一维相关操作，结果存储在 output 中，起始位置为 -1
        output = ndimage.correlate1d(array, kernel, origin=-1)
        # 断言 output 与预期的 tcor 近似相等
        assert_array_almost_equal(tcor, output)
        # 对数组 array 进行一维卷积操作，结果存储在 output 中，起始位置为 -1
        output = ndimage.convolve1d(array, kernel, origin=-1)
        # 断言 output 与预期的 tcov 近似相等
        assert_array_almost_equal(tcov, output)

    # 使用参数 dtype_array 来测试函数 test_correlate18，传入一个特定的 NumPy 数组类型
    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate18(self, dtype_array):
        # 定义一个 2x2 的卷积核 kernel
        kernel = np.array([[1, 0],
                           [0, 1]])
        # 创建一个 2x3 的数组 array，使用传入的 dtype_array 类型
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        # 对数组 array 进行二维相关操作，结果存储在 output 中，输出类型为 np.float32，边界模式为 'nearest'，起始位置为 -1
        output = ndimage.correlate(array, kernel,
                                   output=np.float32,
                                   mode='nearest', origin=-1)
        # 断言 output 与预期的值近似相等
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        # 断言 output 的数据类型为 np.float32

        # 对数组 array 进行二维卷积操作，结果存储在 output 中，输出类型为 np.float32，边界模式为 'nearest'，起始位置为 -1
        output = ndimage.convolve(array, kernel,
                                  output=np.float32,
                                  mode='nearest', origin=-1)
        # 断言 output 与预期的值近似相等
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        # 断言 output 的数据类型为 np.float32

    # 测试不支持的边界模式组合是否会引发 RuntimeError 的测试函数 test_correlate_mode_sequence
    def test_correlate_mode_sequence(self):
        # 定义一个 2x2 的全一卷积核 kernel
        kernel = np.ones((2, 2))
        # 创建一个 3x3 的全一数组 array，数据类型为 float
        array = np.ones((3, 3), float)
        # 使用断言检测在指定的模式序列 'nearest', 'reflect' 下是否会引发 RuntimeError
        with assert_raises(RuntimeError):
            ndimage.correlate(array, kernel, mode=['nearest', 'reflect'])
        with assert_raises(RuntimeError):
            ndimage.convolve(array, kernel, mode=['nearest', 'reflect'])
    # 定义一个测试方法，用于测试 ndimage.correlate 和 ndimage.convolve 函数
    def test_correlate19(self, dtype_array):
        # 创建一个 2x2 的二维 NumPy 数组作为卷积核
        kernel = np.array([[1, 0],
                           [0, 1]])
        # 根据传入的 dtype_array 参数创建一个 2x3 的二维 NumPy 数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_array)
        # 使用 ndimage.correlate 函数对数组 array 进行卷积操作，指定输出类型为 np.float32
        output = ndimage.correlate(array, kernel,
                                   output=np.float32,
                                   mode='nearest', origin=[-1, 0])
        # 断言卷积的结果与预期结果接近
        assert_array_almost_equal([[5, 6, 8], [8, 9, 11]], output)
        # 断言输出数组的数据类型为 np.float32

        # 使用 ndimage.convolve 函数对数组 array 进行卷积操作，指定输出类型为 np.float32
        output = ndimage.convolve(array, kernel,
                                  output=np.float32,
                                  mode='nearest', origin=[-1, 0])
        # 断言卷积的结果与预期结果接近
        assert_array_almost_equal([[3, 5, 6], [6, 8, 9]], output)
        # 断言输出数组的数据类型为 np.float32

    # 使用 pytest 的参数化装饰器，对 test_correlate20 方法进行多组测试
    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate20(self, dtype_array, dtype_output):
        # 定义一个权重数组
        weights = np.array([1, 2, 1])
        # 定义预期的输出结果
        expected = [[5, 10, 15], [7, 14, 21]]
        # 根据传入的 dtype_array 参数创建一个 2x3 的二维 NumPy 数组
        array = np.array([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        # 创建一个指定 dtype_output 类型的全零数组作为输出
        output = np.zeros((2, 3), dtype_output)
        # 使用 ndimage.correlate1d 函数对数组 array 进行一维卷积操作，指定在 axis=0 方向上输出到 output
        ndimage.correlate1d(array, weights, axis=0, output=output)
        # 断言输出结果与预期结果接近
        assert_array_almost_equal(output, expected)
        # 使用 ndimage.convolve1d 函数对数组 array 进行一维卷积操作，指定在 axis=0 方向上输出到 output
        ndimage.convolve1d(array, weights, axis=0, output=output)
        # 断言输出结果与预期结果接近
        assert_array_almost_equal(output, expected)

    # 定义一个测试方法，用于测试 ndimage.correlate1d 和 ndimage.convolve1d 函数
    def test_correlate21(self):
        # 根据给定数组创建一个 2x3 的二维 NumPy 数组
        array = np.array([[1, 2, 3],
                          [2, 4, 6]])
        # 定义预期的输出结果
        expected = [[5, 10, 15], [7, 14, 21]]
        # 定义一个权重数组
        weights = np.array([1, 2, 1])
        # 使用 ndimage.correlate1d 函数对数组 array 进行一维卷积操作，axis=0 方向
        output = ndimage.correlate1d(array, weights, axis=0)
        # 断言输出结果与预期结果接近
        assert_array_almost_equal(output, expected)
        # 使用 ndimage.convolve1d 函数对数组 array 进行一维卷积操作，axis=0 方向
        output = ndimage.convolve1d(array, weights, axis=0)
        # 断言输出结果与预期结果接近
        assert_array_almost_equal(output, expected)

    # 使用 pytest 的参数化装饰器，对 test_correlate22 方法进行多组测试
    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate22(self, dtype_array, dtype_output):
        # 定义一个权重数组
        weights = np.array([1, 2, 1])
        # 定义预期的输出结果
        expected = [[6, 12, 18], [6, 12, 18]]
        # 根据传入的 dtype_array 参数创建一个 2x3 的二维 NumPy 数组
        array = np.array([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        # 创建一个指定 dtype_output 类型的全零数组作为输出
        output = np.zeros((2, 3), dtype_output)
        # 使用 ndimage.correlate1d 函数对数组 array 进行一维卷积操作，指定在 axis=0 方向上输出到 output，并使用 'wrap' 模式
        ndimage.correlate1d(array, weights, axis=0,
                            mode='wrap', output=output)
        # 断言输出结果与预期结果接近
        assert_array_almost_equal(output, expected)
        # 使用 ndimage.convolve1d 函数对数组 array 进行一维卷积操作，指定在 axis=0 方向上输出到 output，并使用 'wrap' 模式
        ndimage.convolve1d(array, weights, axis=0,
                           mode='wrap', output=output)
        # 断言输出结果与预期结果接近
        assert_array_almost_equal(output, expected)
    # 定义一个测试方法，用于测试 ndimage.correlate1d 和 ndimage.convolve1d 函数的行为
    def test_correlate23(self, dtype_array, dtype_output):
        # 定义权重数组
        weights = np.array([1, 2, 1])
        # 预期的输出结果
        expected = [[5, 10, 15], [7, 14, 21]]
        # 创建输入数组
        array = np.array([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        # 创建用于存储输出结果的数组，初始化为全零
        output = np.zeros((2, 3), dtype_output)
        # 对输入数组进行一维相关操作，结果存储到 output 中
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)
        # 对输入数组进行一维卷积操作，结果存储到 output 中
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output)
        # 断言输出结果与预期结果几乎相等
        assert_array_almost_equal(output, expected)

    # 使用参数化测试来测试不同数据类型的输入和输出
    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate24(self, dtype_array, dtype_output):
        # 定义权重数组
        weights = np.array([1, 2, 1])
        # 预期的相关结果
        tcor = [[7, 14, 21], [8, 16, 24]]
        # 预期的卷积结果
        tcov = [[4, 8, 12], [5, 10, 15]]
        # 创建输入数组
        array = np.array([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        # 创建用于存储输出结果的数组，初始化为全零
        output = np.zeros((2, 3), dtype_output)
        # 对输入数组进行一维相关操作，结果存储到 output 中，使用自定义的 origin 参数
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output, origin=-1)
        # 断言输出结果与预期相关结果几乎相等
        assert_array_almost_equal(output, tcor)
        # 对输入数组进行一维卷积操作，结果存储到 output 中，使用自定义的 origin 参数
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output, origin=-1)
        # 断言输出结果与预期卷积结果几乎相等
        assert_array_almost_equal(output, tcov)

    # 使用参数化测试来测试不同数据类型的输入和输出
    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate25(self, dtype_array, dtype_output):
        # 定义权重数组
        weights = np.array([1, 2, 1])
        # 预期的相关结果
        tcor = [[4, 8, 12], [5, 10, 15]]
        # 预期的卷积结果
        tcov = [[7, 14, 21], [8, 16, 24]]
        # 创建输入数组
        array = np.array([[1, 2, 3],
                          [2, 4, 6]], dtype_array)
        # 创建用于存储输出结果的数组，初始化为全零
        output = np.zeros((2, 3), dtype_output)
        # 对输入数组进行一维相关操作，结果存储到 output 中，使用自定义的 origin 参数
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output, origin=1)
        # 断言输出结果与预期相关结果几乎相等
        assert_array_almost_equal(output, tcor)
        # 对输入数组进行一维卷积操作，结果存储到 output 中，使用自定义的 origin 参数
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output, origin=1)
        # 断言输出结果与预期卷积结果几乎相等
        assert_array_almost_equal(output, tcov)

    # 测试针对 gh-11661 问题的修复（长度为1的信号的镜像扩展）
    def test_correlate26(self):
        # 对长度为1的全一数组进行卷积操作，使用镜像模式
        y = ndimage.convolve1d(np.ones(1), np.ones(5), mode='mirror')
        # 断言输出结果与预期结果相等，应该得到一个长度为5的全5数组
        assert_array_equal(y, np.array(5.))

        # 对长度为1的全一数组进行相关操作，使用镜像模式
        y = ndimage.correlate1d(np.ones(1), np.ones(5), mode='mirror')
        # 断言输出结果与预期结果相等，应该得到一个长度为5的全5数组
        assert_array_equal(y, np.array(5.))

    # 使用参数化测试来测试复杂数据类型的核和输入，以及复杂数据类型的输出
    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_kernel(self, dtype_input, dtype_kernel,
                                      dtype_output):
        # 定义复杂类型的核数组
        kernel = np.array([[1, 0],
                           [0, 1 + 1j]], dtype_kernel)
        # 创建输入数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_input)
        # 调用私有方法 _validate_complex，对输入数组和核进行复杂数据类型的验证
        self._validate_complex(array, kernel, dtype_output)
    # 使用 pytest 的参数化功能，对以下参数进行组合测试：complex_types 中的 dtype_kernel，
    # types 中的 dtype_input，complex_types 中的 dtype_output，以及 'grid-constant' 和 'constant' 两种模式
    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    @pytest.mark.parametrize('mode', ['grid-constant', 'constant'])
    def test_correlate_complex_kernel_cval(self, dtype_input, dtype_kernel,
                                           dtype_output, mode):
        # 测试复数输入时使用非零 cval 的情况
        # 同时验证 mode 为 'grid-constant' 时不会发生段错误（segfault）
        
        # 创建复数类型的 kernel，包含实部和虚部
        kernel = np.array([[1, 0],
                           [0, 1 + 1j]], dtype_kernel)
        
        # 创建复数类型的 array，作为输入
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_input)
        
        # 调用 _validate_complex 方法，验证复数输入的相关操作，包括指定的 mode 和 cval
        self._validate_complex(array, kernel, dtype_output, mode=mode,
                               cval=5.0)

    # 使用 pytest 的参数化功能，对以下参数进行组合测试：complex_types 中的 dtype_kernel 和 types 中的 dtype_input
    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    def test_correlate_complex_kernel_invalid_cval(self, dtype_input,
                                                   dtype_kernel):
        # 不能在实数图像上使用复数 cval
        # 创建复数类型的 kernel，包含实部和虚部
        kernel = np.array([[1, 0],
                           [0, 1 + 1j]], dtype_kernel)
        
        # 创建指定类型的 array，作为输入
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype_input)
        
        # 针对 ndimage 模块的一系列函数进行循环测试，验证在 'constant' 模式下不能使用复数 cval
        for func in [ndimage.convolve, ndimage.correlate, ndimage.convolve1d,
                     ndimage.correlate1d]:
            with pytest.raises(ValueError):
                func(array, kernel, mode='constant', cval=5.0 + 1.0j,
                     output=np.complex64)

    # 使用 pytest 的参数化功能，对以下参数进行组合测试：complex_types 中的 dtype_kernel，
    # types 中的 dtype_input，以及 complex_types 中的 dtype_output
    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_kernel(self, dtype_input, dtype_kernel,
                                        dtype_output):
        # 创建复数类型的 1D kernel，包含实部和虚部
        kernel = np.array([1, 1 + 1j], dtype_kernel)
        
        # 创建复数类型的 1D array，作为输入
        array = np.array([1, 2, 3, 4, 5, 6], dtype_input)
        
        # 调用 _validate_complex 方法，验证复数输入的相关操作，不指定 mode 和 cval
        self._validate_complex(array, kernel, dtype_output)

    # 使用 pytest 的参数化功能，对以下参数进行组合测试：complex_types 中的 dtype_kernel，
    # types 中的 dtype_input，以及 complex_types 中的 dtype_output
    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_kernel_cval(self, dtype_input, dtype_kernel,
                                             dtype_output):
        # 创建复数类型的 1D kernel，包含实部和虚部
        kernel = np.array([1, 1 + 1j], dtype_kernel)
        
        # 创建复数类型的 1D array，作为输入
        array = np.array([1, 2, 3, 4, 5, 6], dtype_input)
        
        # 调用 _validate_complex 方法，验证复数输入的相关操作，指定 mode='constant' 和 cval=5.0
        self._validate_complex(array, kernel, dtype_output, mode='constant',
                               cval=5.0)

    # 使用 pytest 的参数化功能，对以下参数进行组合测试：types 中的 dtype_kernel，
    # complex_types 中的 dtype_input 和 dtype_output
    # 定义测试函数，用于测试复杂输入的相关性计算
    def test_correlate_complex_input(self, dtype_input, dtype_kernel,
                                     dtype_output):
        # 创建指定数据类型和内容的二维数组作为卷积核
        kernel = np.array([[1, 0],
                           [0, 1]], dtype_kernel)
        # 创建指定数据类型和内容的二维复数数组作为输入数组
        array = np.array([[1, 2j, 3],
                          [1 + 4j, 5, 6j]], dtype_input)
        # 调用验证复数数组的方法，验证相关性计算
        self._validate_complex(array, kernel, dtype_output)

    # 使用参数化装饰器标记的测试函数，测试复杂输入的相关性计算
    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input(self, dtype_input, dtype_kernel,
                                       dtype_output):
        # 创建指定数据类型和内容的一维数组作为卷积核
        kernel = np.array([1, 0, 1], dtype_kernel)
        # 创建指定数据类型和内容的一维复数数组作为输入数组
        array = np.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype_input)
        # 调用验证复数数组的方法，验证相关性计算
        self._validate_complex(array, kernel, dtype_output)

    # 使用参数化装饰器标记的测试函数，测试带常量值的复杂输入的相关性计算
    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_cval(self, dtype_input, dtype_kernel,
                                            dtype_output):
        # 创建指定数据类型和内容的一维数组作为卷积核
        kernel = np.array([1, 0, 1], dtype_kernel)
        # 创建指定数据类型和内容的一维复数数组作为输入数组
        array = np.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype_input)
        # 调用验证复数数组的方法，验证相关性计算，设置模式为常量，常量值为复数
        self._validate_complex(array, kernel, dtype_output, mode='constant',
                               cval=5 - 3j)

    # 使用参数化装饰器标记的测试函数，测试复杂输入和复杂卷积核的相关性计算
    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input_and_kernel(self, dtype, dtype_output):
        # 创建指定数据类型和内容的二维数组作为复杂卷积核
        kernel = np.array([[1, 0],
                           [0, 1 + 1j]], dtype)
        # 创建指定数据类型和内容的二维复数数组作为输入数组
        array = np.array([[1, 2j, 3],
                          [1 + 4j, 5, 6j]], dtype)
        # 调用验证复数数组的方法，验证相关性计算
        self._validate_complex(array, kernel, dtype_output)

    # 使用参数化装饰器标记的测试函数，测试带常量值的复杂输入和复杂卷积核的相关性计算
    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input_and_kernel_cval(self, dtype,
                                                     dtype_output):
        # 创建指定数据类型和内容的二维数组作为复杂卷积核
        kernel = np.array([[1, 0],
                           [0, 1 + 1j]], dtype)
        # 创建指定数据类型和内容的二维数组作为输入数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype)
        # 调用验证复数数组的方法，验证相关性计算，设置模式为常量，常量值为复数
        self._validate_complex(array, kernel, dtype_output, mode='constant',
                               cval=5.0 + 2.0j)

    # 使用参数化装饰器标记的测试函数，测试复杂输入和复杂卷积核的一维相关性计算
    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_and_kernel(self, dtype, dtype_output):
        # 创建指定数据类型和内容的一维数组作为复杂卷积核
        kernel = np.array([1, 1 + 1j], dtype)
        # 创建指定数据类型和内容的一维复数数组作为输入数组
        array = np.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
        # 调用验证复数数组的方法，验证相关性计算
        self._validate_complex(array, kernel, dtype_output)
    # 测试一维复杂输入和内核的相关性，使用指定的数据类型和输出数据类型
    def test_correlate1d_complex_input_and_kernel_cval(self, dtype,
                                                       dtype_output):
        # 创建复杂数内核数组，包含实部和虚部
        kernel = np.array([1, 1 + 1j], dtype)
        # 创建复杂数输入数组，包含实部和虚部
        array = np.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
        # 使用指定的模式和常数值验证复杂数数组的有效性
        self._validate_complex(array, kernel, dtype_output, mode='constant',
                               cval=5.0 + 2.0j)

    # 测试高斯滤波器，标准差为0
    def test_gauss01(self):
        # 创建输入二维数组，数据类型为32位浮点数
        input = np.array([[1, 2, 3],
                          [2, 4, 6]], np.float32)
        # 对输入数组进行标准差为0的高斯滤波
        output = ndimage.gaussian_filter(input, 0)
        # 断言输出与输入几乎相等
        assert_array_almost_equal(output, input)

    # 测试高斯滤波器，标准差为1.0
    def test_gauss02(self):
        # 创建输入二维数组，数据类型为32位浮点数
        input = np.array([[1, 2, 3],
                          [2, 4, 6]], np.float32)
        # 对输入数组进行标准差为1.0的高斯滤波
        output = ndimage.gaussian_filter(input, 1.0)
        # 断言输入和输出的数据类型相等
        assert_equal(input.dtype, output.dtype)
        # 断言输入和输出的形状相等
        assert_equal(input.shape, output.shape)

    # 测试高斯滤波器，标准差为[1.0, 1.0]，并验证精度和形状
    def test_gauss03(self):
        # 创建单精度浮点数的一维数组，元素从0到9999
        input = np.arange(100 * 100).astype(np.float32)
        input.shape = (100, 100)
        # 对输入数组进行标准差为[1.0, 1.0]的高斯滤波
        output = ndimage.gaussian_filter(input, [1.0, 1.0])

        # 断言输入和输出的数据类型相等
        assert_equal(input.dtype, output.dtype)
        # 断言输入和输出的形状相等
        assert_equal(input.shape, output.shape)

        # 使用单精度浮点数计算输入和输出数组的总和，并验证小数点后0位的精度
        assert_almost_equal(output.sum(dtype='d'), input.sum(dtype='d'),
                            decimal=0)
        # 断言输入和输出数组的平方和大于1.0
        assert_(sumsq(input, output) > 1.0)

    # 测试高斯滤波器，标准差为[1.0, 1.0]，输出数据类型为64位浮点数
    def test_gauss04(self):
        # 创建单精度浮点数的二维数组，元素从0到9999
        input = np.arange(100 * 100).astype(np.float32)
        input.shape = (100, 100)
        # 输出数据类型为64位浮点数
        otype = np.float64
        # 对输入数组进行标准差为[1.0, 1.0]的高斯滤波，输出数据类型为64位浮点数
        output = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        # 断言输出的数据类型为64位浮点数
        assert_equal(output.dtype.type, np.float64)
        # 断言输入和输出的形状相等
        assert_equal(input.shape, output.shape)
        # 断言输入和输出数组的平方和大于1.0
        assert_(sumsq(input, output) > 1.0)

    # 测试高斯滤波器，标准差为[1.0, 1.0]，阶数为1，输出数据类型为64位浮点数
    def test_gauss05(self):
        # 创建单精度浮点数的二维数组，元素从0到9999
        input = np.arange(100 * 100).astype(np.float32)
        input.shape = (100, 100)
        # 输出数据类型为64位浮点数
        otype = np.float64
        # 对输入数组进行标准差为[1.0, 1.0]、阶数为1的高斯滤波，输出数据类型为64位浮点数
        output = ndimage.gaussian_filter(input, [1.0, 1.0],
                                         order=1, output=otype)
        # 断言输出的数据类型为64位浮点数
        assert_equal(output.dtype.type, np.float64)
        # 断言输入和输出的形状相等
        assert_equal(input.shape, output.shape)
        # 断言输入和输出数组的平方和大于1.0
        assert_(sumsq(input, output) > 1.0)

    # 测试高斯滤波器，标准差为[1.0, 1.0]，输出数据类型为64位浮点数
    def test_gauss06(self):
        # 创建单精度浮点数的二维数组，元素从0到9999
        input = np.arange(100 * 100).astype(np.float32)
        input.shape = (100, 100)
        # 输出数据类型为64位浮点数
        otype = np.float64
        # 对输入数组进行标准差为[1.0, 1.0]的高斯滤波，输出数据类型为64位浮点数
        output1 = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        # 对输入数组进行标准差为1.0的高斯滤波，输出数据类型为64位浮点数
        output2 = ndimage.gaussian_filter(input, 1.0, output=otype)
        # 断言两个输出数组几乎相等
        assert_array_almost_equal(output1, output2)

    # 测试高斯滤波器，标准差为1.0，验证内存重叠的情况
    def test_gauss_memory_overlap(self):
        # 创建单精度浮点数的二维数组，元素从0到9999
        input = np.arange(100 * 100).astype(np.float32)
        input.shape = (100, 100)
        # 对输入数组进行标准差为1.0的高斯滤波，输出结果存储在原始输入数组中
        output1 = ndimage.gaussian_filter(input, 1.0)
        # 再次对输入数组进行标准差为1.0的高斯滤波，输出结果存储在原始输入数组中
        ndimage.gaussian_filter(input, 1.0, output=input)
        # 断言两次滤波结果几乎相等
        assert_array_almost_equal(output1, input)
    @pytest.mark.parametrize(('filter_func', 'extra_args', 'size0', 'size'),
                             [(ndimage.gaussian_filter, (), 0, 1.0),
                              (ndimage.uniform_filter, (), 1, 3),
                              (ndimage.minimum_filter, (), 1, 3),
                              (ndimage.maximum_filter, (), 1, 3),
                              (ndimage.median_filter, (), 1, 3),
                              (ndimage.rank_filter, (1,), 1, 3),
                              (ndimage.percentile_filter, (40,), 1, 3)])

```    
    @pytest.mark.parametrize(
        'axes',
        tuple(itertools.combinations(range(-3, 3), 1))
        + tuple(itertools.combinations(range(-3, 3), 2))
        + ((0, 1, 2),))

```    
    def test_filter_axes(self, filter_func, extra_args, size0, size, axes):
        # 创建一个测试方法，用于参数化测试不同的滤波函数和参数组合
        array = np.arange(6 * 8 * 12, dtype=np.float64).reshape(6, 8, 12)
        # 创建一个 6x8x12 的浮点数数组作为测试数据
        axes = np.array(axes)
        # 将 axes 转换为 numpy 数组

        if len(set(axes % array.ndim)) != len(axes):
            # 如果 axes 中存在重复的轴，则抛出 ValueError 错误
            with pytest.raises(ValueError, match="axes must be unique"):
                filter_func(array, *extra_args, size, axes=axes)
            return
        # 使用给定的滤波函数、参数和轴执行滤波操作
        output = filter_func(array, *extra_args, size, axes=axes)

        # 预期的输出应当等同于在未滤波的轴上应用 size=1.0/size0=0 的效果
        all_sizes = (size if ax in (axes % array.ndim) else size0
                     for ax in range(array.ndim))
        expected = filter_func(array, *extra_args, all_sizes)
        # 断言输出结果与预期结果的接近程度
        assert_allclose(output, expected)

    kwargs_gauss = dict(radius=[4, 2, 3], order=[0, 1, 2],
                        mode=['reflect', 'nearest', 'constant'])
    kwargs_other = dict(origin=(-1, 0, 1),
                        mode=['reflect', 'nearest', 'constant'])
    kwargs_rank = dict(origin=(-1, 0, 1))

    @pytest.mark.parametrize("filter_func, size0, size, kwargs",
                             [(ndimage.gaussian_filter, 0, 1.0, kwargs_gauss),
                              (ndimage.uniform_filter, 1, 3, kwargs_other),
                              (ndimage.maximum_filter, 1, 3, kwargs_other),
                              (ndimage.minimum_filter, 1, 3, kwargs_other),
                              (ndimage.median_filter, 1, 3, kwargs_rank),
                              (ndimage.rank_filter, 1, 3, kwargs_rank),
                              (ndimage.percentile_filter, 1, 3, kwargs_rank)])

```    
    @pytest.mark.parametrize('axes', itertools.combinations(range(-3, 3), 2))
    # 定义一个测试方法，用于测试滤波器函数在指定条件下的行为
    def test_filter_axes_kwargs(self, filter_func, size0, size, kwargs, axes):
        # 创建一个 3D 数组，用于测试，大小为 6x8x12，数据类型为 float64
        array = np.arange(6 * 8 * 12, dtype=np.float64).reshape(6, 8, 12)

        # 将传入的 kwargs 字典中的值转换为 NumPy 数组
        kwargs = {key: np.array(val) for key, val in kwargs.items()}
        # 将传入的 axes 元组转换为 NumPy 数组
        axes = np.array(axes)
        # 计算 axes 数组的大小
        n_axes = axes.size

        # 根据 filter_func 的类型确定额外的参数
        if filter_func == ndimage.rank_filter:
            args = (2,)  # (rank,)
        elif filter_func == ndimage.percentile_filter:
            args = (30,)  # (percentile,)
        else:
            args = ()

        # 创建一个新的 kwargs 字典，仅包含在 axes 中指定的轴的值
        reduced_kwargs = {key: val[axes] for key, val in kwargs.items()}
        # 检查 axes 是否有重复的轴，若有则抛出 ValueError 异常
        if len(set(axes % array.ndim)) != len(axes):
            with pytest.raises(ValueError, match="axes must be unique"):
                # 调用滤波器函数，传入相关参数和减少后的 kwargs
                filter_func(array, *args, [size]*n_axes, axes=axes,
                            **reduced_kwargs)
            return

        # 调用滤波器函数，传入相关参数和减少后的 kwargs，得到输出结果
        output = filter_func(array, *args, [size]*n_axes, axes=axes,
                             **reduced_kwargs)

        # 检查输出结果与预期结果的接近程度
        # 预期结果基于原始参数 kwargs 和 size 的组合
        size_3d = np.full(array.ndim, fill_value=size0)
        size_3d[axes] = size
        if 'origin' in kwargs:
            # 如果 kwargs 中包含 origin 参数，设置 origin 值为 0 的轴
            origin = np.array([0, 0, 0])
            origin[axes] = reduced_kwargs['origin']
            kwargs['origin'] = origin
        expected = filter_func(array, *args, size_3d, **kwargs)
        assert_allclose(output, expected)

    # 使用 pytest 的参数化装饰器，对不同的滤波器函数和参数进行测试
    @pytest.mark.parametrize("filter_func, kwargs",
                             [(ndimage.minimum_filter, {}),
                              (ndimage.maximum_filter, {}),
                              (ndimage.median_filter, {}),
                              (ndimage.rank_filter, {"rank": 1}),
                              (ndimage.percentile_filter, {"percentile": 30})])
    # 定义测试方法，测试滤波器函数对权重、子集、轴和原点的影响
    def test_filter_weights_subset_axes_origins(self, filter_func, kwargs):
        # 指定测试用的轴和原点
        axes = (-2, -1)
        origins = (0, 1)
        # 创建一个 3D 数组，用于测试，大小为 6x8x12，数据类型为 float64
        array = np.arange(6 * 8 * 12, dtype=np.float64).reshape(6, 8, 12)
        # 将传入的轴元组转换为 NumPy 数组
        axes = np.array(axes)

        # 创建一个指定形状和数据类型的全 True 数组
        footprint = np.ones((3, 5), dtype=bool)
        # 将 footprint 的特定位置设为 False，使其变为非可分离
        footprint[0, 1] = 0

        # 调用滤波器函数，传入相关参数，包括权重、轴和原点
        output = filter_func(
            array, footprint=footprint, axes=axes, origin=origins, **kwargs)

        # 调用滤波器函数，传入相关参数，但将原点设置为 0
        output0 = filter_func(
            array, footprint=footprint, axes=axes, origin=0, **kwargs)

        # 检查输出与输出0在最后一个轴上的原点偏移是否相等
        np.testing.assert_array_equal(output[:, :, 1:], output0[:, :, :-1])
    @pytest.mark.parametrize(
        'filter_func, args',
        [(ndimage.gaussian_filter, (1.0,)),      # 使用高斯滤波器，参数为 (sigma,)
         (ndimage.uniform_filter, (3,)),         # 使用均匀滤波器，参数为 (size,)
         (ndimage.minimum_filter, (3,)),         # 使用最小值滤波器，参数为 (size,)
         (ndimage.maximum_filter, (3,)),         # 使用最大值滤波器，参数为 (size,)
         (ndimage.median_filter, (3,)),          # 使用中值滤波器，参数为 (size,)
         (ndimage.rank_filter, (2, 3)),          # 使用排位滤波器，参数为 (rank, size)
         (ndimage.percentile_filter, (30, 3))])  # 使用百分位滤波器，参数为 (percentile, size)
    @pytest.mark.parametrize(
        'axes', [(1.5,), (0, 1, 2, 3), (3,), (-4,)]
    )
    def test_filter_invalid_axes(self, filter_func, args, axes):
        # 创建一个形状为 (6, 8, 12) 的浮点数数组
        array = np.arange(6 * 8 * 12, dtype=np.float64).reshape(6, 8, 12)
        # 如果 axes 中存在 float 类型的元素，则错误类型为 TypeError
        if any(isinstance(ax, float) for ax in axes):
            error_class = TypeError
            match = "cannot be interpreted as an integer"
        else:
            error_class = ValueError
            match = "out of range"
        # 使用 pytest 检测是否会抛出特定错误类和匹配的错误信息
        with pytest.raises(error_class, match=match):
            filter_func(array, *args, axes=axes)

    @pytest.mark.parametrize(
        'filter_func, kwargs',
        [(ndimage.minimum_filter, {}),
         (ndimage.maximum_filter, {}),
         (ndimage.median_filter, {}),
         (ndimage.rank_filter, dict(rank=3)),
         (ndimage.percentile_filter, dict(percentile=30))])
    @pytest.mark.parametrize(
        'axes', [(0, ), (1, 2), (0, 1, 2)]
    )
    @pytest.mark.parametrize('separable_footprint', [False, True])
    def test_filter_invalid_footprint_ndim(self, filter_func, kwargs, axes,
                                           separable_footprint):
        # 创建一个形状为 (6, 8, 12) 的浮点数数组
        array = np.arange(6 * 8 * 12, dtype=np.float64).reshape(6, 8, 12)
        # 创建一个维度比 axes 多一的足迹数组
        footprint = np.ones((3,) * (len(axes) + 1))
        # 如果 separable_footprint 为 False，则将足迹数组的第一个元素设为 0
        if not separable_footprint:
            footprint[(0,) * footprint.ndim] = 0
        # 根据不同的条件设定匹配的错误信息
        if (filter_func in [ndimage.minimum_filter, ndimage.maximum_filter]
            and separable_footprint):
            match = "sequence argument must have length equal to input rank"
        else:
            match = "footprint array has incorrect shape"
        # 使用 pytest 检测是否会抛出 RuntimeError 错误类和匹配的错误信息
        with pytest.raises(RuntimeError, match=match):
            filter_func(array, **kwargs, footprint=footprint, axes=axes)

    @pytest.mark.parametrize('n_mismatch', [1, 3])
    @pytest.mark.parametrize('filter_func, kwargs, key, val',
                             _cases_axes_tuple_length_mismatch())
    def test_filter_tuple_length_mismatch(self, n_mismatch, filter_func,
                                          kwargs, key, val):
        # 测试当 kwargs 的大小不匹配时是否会引发预期的 RuntimeError
        # 创建一个三维数组，总共 576 个元素，类型为 np.float64
        array = np.arange(6 * 8 * 12, dtype=np.float64).reshape(6, 8, 12)
        # 将 kwargs 中的 axes 参数设置为 (0, 1)
        kwargs = dict(**kwargs, axes=(0, 1))
        # 设置 kwargs 中的 key 参数为一个长度为 n_mismatch 的元组，其中每个元素都是 val
        kwargs[key] = (val,) * n_mismatch
        # 准备一个错误消息，用于断言是否引发 RuntimeError 异常
        err_msg = "sequence argument must have length equal to input rank"
        # 使用 pytest 的断言来检查是否引发了 RuntimeError 异常，并且异常消息匹配 err_msg
        with pytest.raises(RuntimeError, match=err_msg):
            filter_func(array, **kwargs)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt01(self, dtype):
        # 创建一个二维数组，类型由参数 dtype 指定
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 在第一维度上使用 [-1.0, 0.0, 1.0] 进行一维卷积操作
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        # 在第二维度上使用 [1.0, 1.0, 1.0] 进行一维卷积操作
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 1)
        # 使用 Prewitt 算子在第一维度上进行边缘检测
        output = ndimage.prewitt(array, 0)
        # 断言 t 与 output 数组几乎相等
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt02(self, dtype):
        # 创建一个二维数组，类型由参数 dtype 指定
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 在第一维度上使用 [-1.0, 0.0, 1.0] 进行一维卷积操作
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        # 在第二维度上使用 [1.0, 1.0, 1.0] 进行一维卷积操作
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 1)
        # 创建一个与 array 形状相同的全零数组，类型由参数 dtype 指定
        output = np.zeros(array.shape, dtype)
        # 使用 Prewitt 算子在第一维度上进行边缘检测，并将结果存储到 output 中
        ndimage.prewitt(array, 0, output)
        # 断言 t 与 output 数组几乎相等
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt03(self, dtype):
        # 创建一个二维数组，类型由参数 dtype 指定
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 在第二维度上使用 [-1.0, 0.0, 1.0] 进行一维卷积操作
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
        # 在第一维度上使用 [1.0, 1.0, 1.0] 进行一维卷积操作
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 0)
        # 使用 Prewitt 算子在第二维度上进行边缘检测
        output = ndimage.prewitt(array, 1)
        # 断言 t 与 output 数组几乎相等
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt04(self, dtype):
        # 创建一个二维数组，类型由参数 dtype 指定
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 使用 Prewitt 算子在第一维度上进行边缘检测
        t = ndimage.prewitt(array, -1)
        # 使用 Prewitt 算子在第二维度上进行边缘检测
        output = ndimage.prewitt(array, 1)
        # 断言 t 与 output 数组几乎相等
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel01(self, dtype):
        # 创建一个二维数组，类型由参数 dtype 指定
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 在第一维度上使用 [-1.0, 0.0, 1.0] 进行一维卷积操作
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        # 在第二维度上使用 [1.0, 2.0, 1.0] 进行一维卷积操作
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 1)
        # 使用 Sobel 算子在第一维度上进行边缘检测
        output = ndimage.sobel(array, 0)
        # 断言 t 与 output 数组几乎相等
        assert_array_almost_equal(t, output)
    # 使用 pytest 框架的测试方法，测试 ndimage 模块的 sobel 算子功能
    def test_sobel02(self, dtype):
        # 创建一个指定数据类型的 NumPy 数组
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 在第一个维度上应用一维相关操作，使用 [-1.0, 0.0, 1.0] 作为核
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        # 在第二个维度上应用一维相关操作，使用 [1.0, 2.0, 1.0] 作为核
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 1)
        # 创建一个与 array 形状相同的零数组，指定数据类型
        output = np.zeros(array.shape, dtype)
        # 使用 sobel 算子计算 array 的梯度，并将结果存储在 output 中
        ndimage.sobel(array, 0, output)
        # 断言 t 与 output 的值几乎相等
        assert_array_almost_equal(t, output)

    # 使用 pytest 的参数化功能，测试不同数据类型的 sobel 算子功能
    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel03(self, dtype):
        # 创建一个指定数据类型的 NumPy 数组
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 在第二个维度上应用一维相关操作，使用 [-1.0, 0.0, 1.0] 作为核
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
        # 在第一个维度上应用一维相关操作，使用 [1.0, 2.0, 1.0] 作为核
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 0)
        # 创建一个与 array 形状相同的零数组，指定数据类型
        output = np.zeros(array.shape, dtype)
        # 使用 sobel 算子计算 array 的梯度，并将结果存储在 output 中
        output = ndimage.sobel(array, 1)
        # 断言 t 与 output 的值几乎相等
        assert_array_almost_equal(t, output)

    # 使用 pytest 的参数化功能，测试不同数据类型的 sobel 算子功能
    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel04(self, dtype):
        # 创建一个指定数据类型的 NumPy 数组
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 使用 sobel 算子计算 array 的梯度，并将结果存储在 t 中
        t = ndimage.sobel(array, -1)
        # 使用 sobel 算子计算 array 的梯度，并将结果存储在 output 中
        output = ndimage.sobel(array, 1)
        # 断言 t 与 output 的值几乎相等
        assert_array_almost_equal(t, output)

    # 使用 pytest 的参数化功能，测试不同数据类型的 laplace 算子功能
    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_laplace01(self, dtype):
        # 创建一个指定数据类型的 NumPy 数组，并乘以 100
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        # 在第一个维度上应用一维相关操作，使用 [1, -2, 1] 作为核
        tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
        # 在第二个维度上应用一维相关操作，使用 [1, -2, 1] 作为核
        tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
        # 使用 laplace 算子计算 array 的拉普拉斯算子，并将结果存储在 output 中
        output = ndimage.laplace(array)
        # 断言 tmp1 加 tmp2 与 output 的值几乎相等
        assert_array_almost_equal(tmp1 + tmp2, output)

    # 使用 pytest 的参数化功能，测试不同数据类型的 laplace 算子功能
    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_laplace02(self, dtype):
        # 创建一个指定数据类型的 NumPy 数组，并乘以 100
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        # 在第一个维度上应用一维相关操作，使用 [1, -2, 1] 作为核
        tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
        # 在第二个维度上应用一维相关操作，使用 [1, -2, 1] 作为核
        tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
        # 创建一个与 array 形状相同的零数组，指定数据类型
        output = np.zeros(array.shape, dtype)
        # 使用 laplace 算子计算 array 的拉普拉斯算子，并将结果存储在 output 中
        ndimage.laplace(array, output=output)
        # 断言 tmp1 加 tmp2 与 output 的值几乎相等
        assert_array_almost_equal(tmp1 + tmp2, output)

    # 使用 pytest 的参数化功能，测试不同数据类型的 gaussian_laplace 算子功能
    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    def test_gaussian_laplace01(self, dtype):
        # 创建一个指定数据类型的 NumPy 数组，并乘以 100
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        # 在第一个维度上应用二维高斯滤波，sigma=1.0，方向为 [2, 0]
        tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
        # 在第二个维度上应用二维高斯滤波，sigma=1.0，方向为 [0, 2]
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
        # 使用 gaussian_laplace 算子计算 array 的高斯拉普拉斯算子，并将结果存储在 output 中
        output = ndimage.gaussian_laplace(array, 1.0)
        # 断言 tmp1 加 tmp2 与 output 的值几乎相等
        assert_array_almost_equal(tmp1 + tmp2, output)
    # 使用 pytest 的参数化装饰器，对不同的 dtype 进行参数化测试
    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    # 定义测试函数 test_gaussian_laplace02，测试 ndimage 中的高斯拉普拉斯滤波器
    def test_gaussian_laplace02(self, dtype):
        # 创建一个 numpy 数组 array，根据给定的 dtype，乘以 100
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        # 对 array 应用高斯滤波器，sigma=1.0，方向为 [2, 0]，存入 tmp1
        tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
        # 对 array 应用高斯滤波器，sigma=1.0，方向为 [0, 2]，存入 tmp2
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
        # 创建一个与 array 形状相同的零数组，dtype 由参数 dtype 决定，用于存储输出
        output = np.zeros(array.shape, dtype)
        # 对 array 应用高斯拉普拉斯滤波器，sigma=1.0，输出存入 output
        ndimage.gaussian_laplace(array, 1.0, output)
        # 检查 tmp1 + tmp2 是否与 output 数组几乎相等
        assert_array_almost_equal(tmp1 + tmp2, output)

    # 使用 pytest 的参数化装饰器，对复合类型进行参数化测试
    @pytest.mark.parametrize('dtype', types + complex_types)
    # 定义测试函数 test_generic_laplace01，测试 ndimage 中的通用拉普拉斯滤波器
    def test_generic_laplace01(self, dtype):
        # 定义函数 derivative2，用于计算 input 的二阶导数
        def derivative2(input, axis, output, mode, cval, a, b):
            # 设置高斯滤波器的 sigma 参数
            sigma = [a, b / 2.0]
            # 将 input 转换为 numpy 数组
            input = np.asarray(input)
            # 设置各维度的导数阶数
            order = [0] * input.ndim
            order[axis] = 2
            # 返回应用高斯滤波器后的结果
            return ndimage.gaussian_filter(input, sigma, order,
                                           output, mode, cval)
        # 创建一个 numpy 数组 array，根据给定的 dtype
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 创建一个与 array 形状相同的零数组，dtype 由参数 dtype 决定，用于存储输出
        output = np.zeros(array.shape, dtype)
        # 对 array 应用通用拉普拉斯滤波器，输出存入 tmp
        tmp = ndimage.generic_laplace(array, derivative2,
                                      extra_arguments=(1.0,),
                                      extra_keywords={'b': 2.0})
        # 对 array 应用高斯拉普拉斯滤波器，sigma=1.0，输出存入 output
        ndimage.gaussian_laplace(array, 1.0, output)
        # 检查 tmp 是否与 output 数组几乎相等
        assert_array_almost_equal(tmp, output)

    # 使用 pytest 的参数化装饰器，对不同的 dtype 进行参数化测试
    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    # 定义测试函数 test_gaussian_gradient_magnitude01，测试 ndimage 中的高斯梯度幅值滤波器
    def test_gaussian_gradient_magnitude01(self, dtype):
        # 创建一个 numpy 数组 array，根据给定的 dtype，乘以 100
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        # 对 array 应用高斯滤波器，sigma=1.0，方向为 [1, 0]，存入 tmp1
        tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
        # 对 array 应用高斯滤波器，sigma=1.0，方向为 [0, 1]，存入 tmp2
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
        # 计算 array 的高斯梯度幅值，sigma=1.0，输出存入 output
        output = ndimage.gaussian_gradient_magnitude(array, 1.0)
        # 计算预期的幅值，由 tmp1 和 tmp2 计算得出，并转换为指定的 dtype
        expected = tmp1 * tmp1 + tmp2 * tmp2
        expected = np.sqrt(expected).astype(dtype)
        # 检查 expected 是否与 output 数组几乎相等
        assert_array_almost_equal(expected, output)

    # 使用 pytest 的参数化装饰器，对不同的 dtype 进行参数化测试
    @pytest.mark.parametrize('dtype',
                             [np.int32, np.float32, np.float64,
                              np.complex64, np.complex128])
    # 定义测试函数 test_gaussian_gradient_magnitude02，测试 ndimage 中的高斯梯度幅值滤波器
    def test_gaussian_gradient_magnitude02(self, dtype):
        # 创建一个 numpy 数组 array，根据给定的 dtype，乘以 100
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype) * 100
        # 对 array 应用高斯滤波器，sigma=1.0，方向为 [1, 0]，存入 tmp1
        tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
        # 对 array 应用高斯滤波器，sigma=1.0，方向为 [0, 1]，存入 tmp2
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
        # 创建一个与 array 形状相同的零数组，dtype 由参数 dtype 决定，用于存储输出
        output = np.zeros(array.shape, dtype)
        # 计算 array 的高斯梯度幅值，sigma=1.0，输出存入 output
        ndimage.gaussian_gradient_magnitude(array, 1.0, output)
        # 计算预期的幅值，由 tmp1 和 tmp2 计算得出，并转换为指定的 dtype
        expected = tmp1 * tmp1 + tmp2 * tmp2
        expected = np.sqrt(expected).astype(dtype)
        # 检查 expected 是否与 output 数组几乎相等
        assert_array_almost_equal(expected, output)
    def test_generic_gradient_magnitude01(self):
        # 创建一个二维数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], np.float64)

        # 定义一个计算导数的函数 derivative，接受输入、轴、输出、模式、常量值、a 和 b 作为参数
        def derivative(input, axis, output, mode, cval, a, b):
            # 设置高斯滤波器的 sigma 值
            sigma = [a, b / 2.0]
            # 将输入转换为 NumPy 数组
            input = np.asarray(input)
            # 设置导数的阶数
            order = [0] * input.ndim
            order[axis] = 1
            # 调用 ndimage 库中的高斯滤波函数计算导数
            return ndimage.gaussian_filter(input, sigma, order,
                                           output, mode, cval)

        # 计算数组 array 的高斯梯度幅值
        tmp1 = ndimage.gaussian_gradient_magnitude(array, 1.0)
        # 使用自定义的导数函数 derivative 计算数组 array 的通用梯度幅值
        tmp2 = ndimage.generic_gradient_magnitude(
            array, derivative, extra_arguments=(1.0,),
            extra_keywords={'b': 2.0})
        # 断言两者的值几乎相等
        assert_array_almost_equal(tmp1, tmp2)

    def test_uniform01(self):
        # 创建一个一维数组作为测试数据
        array = np.array([2, 4, 6])
        # 定义滤波器的大小
        size = 2
        # 对数组 array 应用一维均匀滤波器
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        # 断言滤波后的输出与预期的近似相等
        assert_array_almost_equal([3, 5, 6], output)

    def test_uniform01_complex(self):
        # 创建一个复数类型的一维数组作为测试数据
        array = np.array([2 + 1j, 4 + 2j, 6 + 3j], dtype=np.complex128)
        # 定义滤波器的大小
        size = 2
        # 对复数数组 array 应用一维均匀滤波器
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        # 断言滤波后的实部输出与预期的近似相等
        assert_array_almost_equal([3, 5, 6], output.real)
        # 断言滤波后的虚部输出与预期的近似相等
        assert_array_almost_equal([1.5, 2.5, 3], output.imag)

    def test_uniform02(self):
        # 创建一个一维数组作为测试数据
        array = np.array([1, 2, 3])
        # 定义滤波器的形状
        filter_shape = [0]
        # 对数组 array 应用零维均匀滤波器
        output = ndimage.uniform_filter(array, filter_shape)
        # 断言滤波后的输出与原始数组 array 的值相等
        assert_array_almost_equal(array, output)

    def test_uniform03(self):
        # 创建一个一维数组作为测试数据
        array = np.array([1, 2, 3])
        # 定义滤波器的形状
        filter_shape = [1]
        # 对数组 array 应用一维均匀滤波器
        output = ndimage.uniform_filter(array, filter_shape)
        # 断言滤波后的输出与原始数组 array 的值相等
        assert_array_almost_equal(array, output)

    def test_uniform04(self):
        # 创建一个一维数组作为测试数据
        array = np.array([2, 4, 6])
        # 定义滤波器的形状
        filter_shape = [2]
        # 对数组 array 应用一维均匀滤波器
        output = ndimage.uniform_filter(array, filter_shape)
        # 断言滤波后的输出与预期的近似相等
        assert_array_almost_equal([2, 3, 5], output)

    def test_uniform05(self):
        # 创建一个空数组作为测试数据
        array = []
        # 定义滤波器的形状
        filter_shape = [1]
        # 对空数组 array 应用一维均匀滤波器
        output = ndimage.uniform_filter(array, filter_shape)
        # 断言滤波后的输出与预期的近似相等
        assert_array_almost_equal([], output)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_uniform06(self, dtype_array, dtype_output):
        # 定义滤波器的形状
        filter_shape = [2, 2]
        # 创建一个二维数组作为测试数据
        array = np.array([[4, 8, 12],
                          [16, 20, 24]], dtype_array)
        # 对数组 array 应用二维均匀滤波器
        output = ndimage.uniform_filter(
            array, filter_shape, output=dtype_output)
        # 断言滤波后的输出与预期的近似相等
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
        # 断言滤波后输出的数据类型与预期的输出类型相同
        assert_equal(output.dtype.type, dtype_output)

    @pytest.mark.parametrize('dtype_array', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    # 定义一个测试函数，用于测试 ndimage.uniform_filter 的复杂情况
    def test_uniform06_complex(self, dtype_array, dtype_output):
        # 定义滤波器形状为 [2, 2]
        filter_shape = [2, 2]
        # 创建一个复数类型的 NumPy 数组，作为输入数据
        array = np.array([[4, 8 + 5j, 12],
                          [16, 20, 24]], dtype_array)
        # 调用 ndimage.uniform_filter 进行均匀滤波，指定输出类型为 dtype_output
        output = ndimage.uniform_filter(
            array, filter_shape, output=dtype_output)
        # 断言输出的实部近似等于 [[4, 6, 10], [10, 12, 16]]
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output.real)
        # 断言输出的数据类型等于 dtype_output
        assert_equal(output.dtype.type, dtype_output)

    # 定义一个测试函数，测试 ndimage.minimum_filter 的基本用法
    def test_minimum_filter01(self):
        # 创建一个一维 NumPy 数组作为输入数据
        array = np.array([1, 2, 3, 4, 5])
        # 定义滤波器形状为 [2]
        filter_shape = np.array([2])
        # 调用 ndimage.minimum_filter 进行最小值滤波
        output = ndimage.minimum_filter(array, filter_shape)
        # 断言输出近似等于 [1, 1, 2, 3, 4]
        assert_array_almost_equal([1, 1, 2, 3, 4], output)

    # 定义一系列测试函数，测试不同滤波器形状下的 ndimage.minimum_filter 行为
    def test_minimum_filter02(self):
        # 同上，不同的滤波器形状 [3]
        ...

    def test_minimum_filter03(self):
        # 同上，不同的输入数组和滤波器形状 [2]
        ...

    def test_minimum_filter04(self):
        # 同上，不同的输入数组和滤波器形状 [3]
        ...

    def test_minimum_filter05(self):
        # 测试二维数组和二维滤波器形状 [2, 3] 下的最小值滤波
        ...

    def test_minimum_filter05_overlap(self):
        # 测试在重叠模式下应用二维数组和二维滤波器形状 [2, 3] 的最小值滤波
        ...

    def test_minimum_filter06(self):
        # 测试使用自定义足迹（footprint）进行最小值滤波，并验证分离足迹可以使用模式序列
        ...
    # 定义测试函数 test_minimum_filter07，用于测试 ndimage 库中的最小值滤波器函数
    def test_minimum_filter07(self):
        # 创建一个 NumPy 数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个 2D 结构元素（足迹）
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 调用 ndimage 库中的最小值滤波器函数，并存储输出结果
        output = ndimage.minimum_filter(array, footprint=footprint)
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1]], output)
        # 使用断言确保在特定条件下会引发 RuntimeError 异常
        with assert_raises(RuntimeError):
            ndimage.minimum_filter(array, footprint=footprint,
                                   mode=['reflect', 'constant'])

    # 定义测试函数 test_minimum_filter08，用于测试 ndimage 库中的最小值滤波器函数的 origin 参数
    def test_minimum_filter08(self):
        # 创建一个 NumPy 数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个 2D 结构元素（足迹）
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 调用 ndimage 库中的最小值滤波器函数，使用 origin 参数设置为 -1，并存储输出结果
        output = ndimage.minimum_filter(array, footprint=footprint, origin=-1)
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal([[3, 1, 3, 1, 1],
                                   [5, 3, 3, 1, 1],
                                   [3, 3, 1, 1, 1]], output)

    # 定义测试函数 test_minimum_filter09，用于测试 ndimage 库中的最小值滤波器函数的 origin 参数
    def test_minimum_filter09(self):
        # 创建一个 NumPy 数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个 2D 结构元素（足迹）
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 调用 ndimage 库中的最小值滤波器函数，使用 origin 参数设置为 [-1, 0]，并存储输出结果
        output = ndimage.minimum_filter(array, footprint=footprint,
                                        origin=[-1, 0])
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal([[2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1],
                                   [5, 3, 3, 1, 1]], output)

    # 定义测试函数 test_maximum_filter01，用于测试 ndimage 库中的最大值滤波器函数
    def test_maximum_filter01(self):
        # 创建一个 NumPy 数组作为测试数据
        array = np.array([1, 2, 3, 4, 5])
        # 定义一个 1D 结构元素（滤波器形状）
        filter_shape = np.array([2])
        # 调用 ndimage 库中的最大值滤波器函数，并存储输出结果
        output = ndimage.maximum_filter(array, filter_shape)
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal([1, 2, 3, 4, 5], output)

    # 定义测试函数 test_maximum_filter02，用于测试 ndimage 库中的最大值滤波器函数
    def test_maximum_filter02(self):
        # 创建一个 NumPy 数组作为测试数据
        array = np.array([1, 2, 3, 4, 5])
        # 定义一个 1D 结构元素（滤波器形状）
        filter_shape = np.array([3])
        # 调用 ndimage 库中的最大值滤波器函数，并存储输出结果
        output = ndimage.maximum_filter(array, filter_shape)
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal([2, 3, 4, 5, 5], output)

    # 定义测试函数 test_maximum_filter03，用于测试 ndimage 库中的最大值滤波器函数
    def test_maximum_filter03(self):
        # 创建一个 NumPy 数组作为测试数据
        array = np.array([3, 2, 5, 1, 4])
        # 定义一个 1D 结构元素（滤波器形状）
        filter_shape = np.array([2])
        # 调用 ndimage 库中的最大值滤波器函数，并存储输出结果
        output = ndimage.maximum_filter(array, filter_shape)
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal([3, 3, 5, 5, 4], output)

    # 定义测试函数 test_maximum_filter04，用于测试 ndimage 库中的最大值滤波器函数
    def test_maximum_filter04(self):
        # 创建一个 NumPy 数组作为测试数据
        array = np.array([3, 2, 5, 1, 4])
        # 定义一个 1D 结构元素（滤波器形状）
        filter_shape = np.array([3])
        # 调用 ndimage 库中的最大值滤波器函数，并存储输出结果
        output = ndimage.maximum_filter(array, filter_shape)
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal([3, 5, 5, 5, 4], output)

    # 定义测试函数 test_maximum_filter05，用于测试 ndimage 库中的最大值滤波器函数
    def test_maximum_filter05(self):
        # 创建一个 NumPy 2D 数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个 2D 结构元素（滤波器形状）
        filter_shape = np.array([2, 3])
        # 调用 ndimage 库中的最大值滤波器函数，并存储输出结果
        output = ndimage.maximum_filter(array, filter_shape)
        # 断言输出结果与预期结果的近似性
        assert_array_almost_equal([[3, 5, 5, 5, 4],
                                   [7, 9, 9, 9, 5],
                                   [8, 9, 9, 9, 7]], output)
    def test_maximum_filter06(self):
        # 创建一个测试用例，测试 ndimage.maximum_filter 函数的功能
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个 2x3 的脚印（footprint）
        footprint = [[1, 1, 1], [1, 1, 1]]
        # 对 array 应用最大值滤波器，使用指定的脚印（footprint）
        output = ndimage.maximum_filter(array, footprint=footprint)
        # 断言输出数组与期望结果的近似相等性
        assert_array_almost_equal([[3, 5, 5, 5, 4],
                                   [7, 9, 9, 9, 5],
                                   [8, 9, 9, 9, 7]], output)
        # 使用可分离脚印应该允许模式序列
        output2 = ndimage.maximum_filter(array, footprint=footprint,
                                         mode=['reflect', 'reflect'])
        # 断言两个输出数组的近似相等性
        assert_array_almost_equal(output2, output)

    def test_maximum_filter07(self):
        # 创建另一个测试用例，测试非可分离脚印情况下的最大值滤波器
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个 2x3 的脚印（footprint）
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 对 array 应用最大值滤波器，使用指定的脚印（footprint）
        output = ndimage.maximum_filter(array, footprint=footprint)
        # 断言输出数组与期望结果的近似相等性
        assert_array_almost_equal([[3, 5, 5, 5, 4],
                                   [7, 7, 9, 9, 5],
                                   [7, 9, 8, 9, 7]], output)
        # 非可分离脚印不应该允许模式序列
        with assert_raises(RuntimeError):
            ndimage.maximum_filter(array, footprint=footprint,
                                   mode=['reflect', 'reflect'])

    def test_maximum_filter08(self):
        # 创建另一个测试用例，测试带有 origin 参数的最大值滤波器
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个 2x3 的脚印（footprint）
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 对 array 应用最大值滤波器，使用指定的脚印（footprint）和 origin 参数
        output = ndimage.maximum_filter(array, footprint=footprint, origin=-1)
        # 断言输出数组与期望结果的近似相等性
        assert_array_almost_equal([[7, 9, 9, 5, 5],
                                   [9, 8, 9, 7, 5],
                                   [8, 8, 7, 7, 7]], output)

    def test_maximum_filter09(self):
        # 创建另一个测试用例，测试带有 origin 参数的最大值滤波器（origin 是一个数组）
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 定义一个 2x3 的脚印（footprint）
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 对 array 应用最大值滤波器，使用指定的脚印（footprint）和 origin 参数
        output = ndimage.maximum_filter(array, footprint=footprint,
                                        origin=[-1, 0])
        # 断言输出数组与期望结果的近似相等性
        assert_array_almost_equal([[7, 7, 9, 9, 5],
                                   [7, 9, 8, 9, 7],
                                   [8, 8, 8, 7, 7]], output)

    @pytest.mark.parametrize(
        'axes', tuple(itertools.combinations(range(-3, 3), 2))
    )
    @pytest.mark.parametrize(
        'filter_func, kwargs',
        [(ndimage.minimum_filter, {}),
         (ndimage.maximum_filter, {}),
         (ndimage.median_filter, {}),
         (ndimage.rank_filter, dict(rank=3)),
         (ndimage.percentile_filter, dict(percentile=60))]
    )
    # 定义一个测试方法，用于测试在非可分离轴上的最大最小滤波器
    def test_minmax_nonseparable_axes(self, filter_func, axes, kwargs):
        # 创建一个形状为 (6, 8, 12) 的浮点数类型的数组
        array = np.arange(6 * 8 * 12, dtype=np.float32).reshape(6, 8, 12)
        # 使用二维三角形状的足迹，因为它是非可分离的
        footprint = np.tri(5)
        # 将传入的轴参数转换为 NumPy 数组
        axes = np.array(axes)

        # 如果轴参数的模数集合长度不等于轴参数长度，则抛出 ValueError 异常
        if len(set(axes % array.ndim)) != len(axes):
            with pytest.raises(ValueError):
                # 调用滤波函数，期望引发异常
                filter_func(array, footprint=footprint, axes=axes, **kwargs)
            return
        
        # 调用滤波函数，生成输出结果
        output = filter_func(array, footprint=footprint, axes=axes, **kwargs)

        # 计算缺失的轴，这个轴是在 0 到 2 范围内与轴参数取模不重复的值
        missing_axis = tuple(set(range(3)) - set(axes % array.ndim))[0]
        # 在缺失轴上扩展足迹数组为三维
        footprint_3d = np.expand_dims(footprint, missing_axis)
        # 使用三维足迹数组调用滤波函数，生成预期结果
        expected = filter_func(array, footprint=footprint_3d, **kwargs)
        # 断言输出结果与预期结果的接近程度
        assert_allclose(output, expected)

    # 定义一个测试方法，用于测试排名滤波器在数组上的功能
    def test_rank01(self):
        # 创建一个包含五个元素的一维数组
        array = np.array([1, 2, 3, 4, 5])
        # 使用排名滤波器进行滤波，期望输出与输入接近
        output = ndimage.rank_filter(array, 1, size=2)
        assert_array_almost_equal(array, output)
        # 使用百分位滤波器进行滤波，期望输出与输入接近
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal(array, output)
        # 使用中值滤波器进行滤波，期望输出与输入接近
        output = ndimage.median_filter(array, 2)
        assert_array_almost_equal(array, output)

    # 定义一个测试方法，测试排名滤波器在不同参数下的表现
    def test_rank02(self):
        # 创建一个包含五个元素的一维数组
        array = np.array([1, 2, 3, 4, 5])
        # 使用排名滤波器进行滤波，期望输出与输入接近
        output = ndimage.rank_filter(array, 1, size=[3])
        assert_array_almost_equal(array, output)
        # 使用百分位滤波器进行滤波，期望输出与输入接近
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(array, output)
        # 使用中值滤波器进行滤波，期望输出与输入接近
        output = ndimage.median_filter(array, (3,))
        assert_array_almost_equal(array, output)

    # 定义一个测试方法，测试排名滤波器在不同参数下的表现
    def test_rank03(self):
        # 创建一个包含五个元素的一维数组
        array = np.array([3, 2, 5, 1, 4])
        # 使用排名滤波器进行滤波，期望输出与指定数组接近
        output = ndimage.rank_filter(array, 1, size=[2])
        assert_array_almost_equal([3, 3, 5, 5, 4], output)
        # 使用百分位滤波器进行滤波，期望输出与指定数组接近
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal([3, 3, 5, 5, 4], output)

    # 定义一个测试方法，测试排名滤波器在不同参数下的表现
    def test_rank04(self):
        # 创建一个包含五个元素的一维数组
        array = np.array([3, 2, 5, 1, 4])
        # 期望输出数组
        expected = [3, 3, 2, 4, 4]
        # 使用排名滤波器进行滤波，期望输出与指定数组接近
        output = ndimage.rank_filter(array, 1, size=3)
        assert_array_almost_equal(expected, output)
        # 使用百分位滤波器进行滤波，期望输出与指定数组接近
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(expected, output)
        # 使用中值滤波器进行滤波，期望输出与指定数组接近
        output = ndimage.median_filter(array, size=3)
        assert_array_almost_equal(expected, output)

    # 定义一个测试方法，测试排名滤波器在不同参数下的表现
    def test_rank05(self):
        # 创建一个包含五个元素的一维数组
        array = np.array([3, 2, 5, 1, 4])
        # 期望输出数组
        expected = [3, 3, 2, 4, 4]
        # 使用排名滤波器进行滤波，期望输出与指定数组接近
        output = ndimage.rank_filter(array, -2, size=3)
        assert_array_almost_equal(expected, output)
    def test_rank06(self):
        # 创建一个二维 NumPy 数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]])
        # 期望的输出结果，用于测试断言
        expected = [[2, 2, 1, 1, 1],
                    [3, 3, 2, 1, 1],
                    [5, 5, 3, 3, 1]]
        # 使用 ndimage 库中的 rank_filter 对数组进行秩滤波处理
        output = ndimage.rank_filter(array, 1, size=[2, 3])
        # 断言输出结果与期望结果几乎相等
        assert_array_almost_equal(expected, output)
        # 使用 ndimage 库中的 percentile_filter 对数组进行百分位滤波处理
        output = ndimage.percentile_filter(array, 17, size=(2, 3))
        # 再次断言输出结果与期望结果几乎相等

    def test_rank06_overlap(self):
        # 创建一个二维 NumPy 数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]])
        # 备份原始数组，以便在覆盖输出的情况下进行比较
        array_copy = array.copy()
        # 期望的输出结果，用于测试断言
        expected = [[2, 2, 1, 1, 1],
                    [3, 3, 2, 1, 1],
                    [5, 5, 3, 3, 1]]
        # 使用 ndimage 库中的 rank_filter 对数组进行秩滤波处理，并将结果输出到原数组中
        ndimage.rank_filter(array, 1, size=[2, 3], output=array)
        # 断言输出结果与期望结果几乎相等
        assert_array_almost_equal(expected, array)

        # 使用 ndimage 库中的 percentile_filter 对备份数组进行百分位滤波处理，并将结果输出到备份数组中
        ndimage.percentile_filter(array_copy, 17, size=(2, 3), output=array_copy)
        # 再次断言输出结果与期望结果几乎相等

    def test_rank07(self):
        # 创建一个二维 NumPy 数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]])
        # 期望的输出结果，用于测试断言
        expected = [[3, 5, 5, 5, 4],
                    [5, 5, 7, 5, 4],
                    [6, 8, 8, 7, 5]]
        # 使用 ndimage 库中的 rank_filter 对数组进行秩滤波处理，负秩值将被解释为最大值减去该值
        output = ndimage.rank_filter(array, -2, size=[2, 3])
        # 断言输出结果与期望结果几乎相等

    def test_rank08(self):
        # 创建一个二维 NumPy 数组作为测试数据
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]])
        # 期望的输出结果，用于测试断言
        expected = [[3, 3, 2, 4, 4],
                    [5, 5, 5, 4, 4],
                    [5, 6, 7, 5, 5]]
        # 使用 ndimage 库中的 percentile_filter 对数组进行百分位滤波处理
        output = ndimage.percentile_filter(array, 50.0, size=(2, 3))
        # 断言输出结果与期望结果几乎相等
        output = ndimage.rank_filter(array, 3, size=(2, 3))
        # 再次断言输出结果与期望结果几乎相等
        output = ndimage.median_filter(array, size=(2, 3))
        # 三种滤波方法的输出结果均与期望结果几乎相等，用于测试断言

        # 非可分离模式: 不允许模式序列
        with assert_raises(RuntimeError):
            # 在非可分离模式下，使用 percentile_filter 抛出运行时错误
            ndimage.percentile_filter(array, 50.0, size=(2, 3),
                                      mode=['reflect', 'constant'])
        with assert_raises(RuntimeError):
            # 在非可分离模式下，使用 rank_filter 抛出运行时错误
            ndimage.rank_filter(array, 3, size=(2, 3), mode=['reflect']*2)
        with assert_raises(RuntimeError):
            # 在非可分离模式下，使用 median_filter 抛出运行时错误
            ndimage.median_filter(array, size=(2, 3), mode=['reflect']*2)

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 框架测试 rank09 方法，传入数据类型参数 dtype
    def test_rank09(self, dtype):
        # 预期输出的二维数组
        expected = [[3, 3, 2, 4, 4],
                    [3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5]]
        # 定义一个足迹（footprint）数组
        footprint = [[1, 0, 1], [0, 1, 0]]
        # 创建一个 numpy 数组，根据给定的数据类型 dtype
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 使用 ndimage 库的 rank_filter 方法进行秩滤波处理，指定秩为1和足迹数组
        output = ndimage.rank_filter(array, 1, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)
        # 使用 ndimage 库的 percentile_filter 方法进行百分位滤波处理，指定百分位为35和足迹数组
        output = ndimage.percentile_filter(array, 35, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)

    # 使用 pytest 框架测试 rank10 方法
    def test_rank10(self):
        # 创建一个 numpy 数组
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 预期输出的二维数组
        expected = [[2, 2, 1, 1, 1],
                    [2, 3, 1, 3, 1],
                    [5, 5, 3, 3, 1]]
        # 定义一个足迹（footprint）数组
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 使用 ndimage 库的 rank_filter 方法进行秩滤波处理，指定秩为0和足迹数组
        output = ndimage.rank_filter(array, 0, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)
        # 使用 ndimage 库的 percentile_filter 方法进行百分位滤波处理，指定百分位为0.0和足迹数组
        output = ndimage.percentile_filter(array, 0.0, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)

    # 使用 pytest 框架测试 rank11 方法
    def test_rank11(self):
        # 创建一个 numpy 数组
        array = np.array([[3, 2, 5, 1, 4],
                          [7, 6, 9, 3, 5],
                          [5, 8, 3, 7, 1]])
        # 预期输出的二维数组
        expected = [[3, 5, 5, 5, 4],
                    [7, 7, 9, 9, 5],
                    [7, 9, 8, 9, 7]]
        # 定义一个足迹（footprint）数组
        footprint = [[1, 0, 1], [1, 1, 0]]
        # 使用 ndimage 库的 rank_filter 方法进行秩滤波处理，指定秩为-1和足迹数组
        output = ndimage.rank_filter(array, -1, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)
        # 使用 ndimage 库的 percentile_filter 方法进行百分位滤波处理，指定百分位为100.0和足迹数组
        output = ndimage.percentile_filter(array, 100.0, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)

    # 使用 pytest 框架测试 rank12 方法，使用 @pytest.mark.parametrize 标注测试方法，传入数据类型参数 dtype
    @pytest.mark.parametrize('dtype', types)
    def test_rank12(self, dtype):
        # 预期输出的二维数组
        expected = [[3, 3, 2, 4, 4],
                    [3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5]]
        # 定义一个足迹（footprint）数组
        footprint = [[1, 0, 1], [0, 1, 0]]
        # 创建一个 numpy 数组，根据给定的数据类型 dtype
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 使用 ndimage 库的 rank_filter 方法进行秩滤波处理，指定秩为1和足迹数组
        output = ndimage.rank_filter(array, 1, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)
        # 使用 ndimage 库的 percentile_filter 方法进行百分位滤波处理，指定百分位为50.0和足迹数组
        output = ndimage.percentile_filter(array, 50.0, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)
        # 使用 ndimage 库的 median_filter 方法进行中值滤波处理，指定足迹数组
        output = ndimage.median_filter(array, footprint=footprint)
        # 断言预期输出和实际输出的数组几乎相等
        assert_array_almost_equal(expected, output)

    # 使用 pytest 框架测试 rank12 方法，使用 @pytest.mark.parametrize 标注测试方法，传入数据类型参数 dtype
    # 定义一个测试方法，用于测试 rank_filter 函数的功能，针对特定的数据类型 dtype
    def test_rank13(self, dtype):
        # 期望的输出结果
        expected = [[5, 2, 5, 1, 1],
                    [5, 8, 3, 5, 5],
                    [6, 6, 5, 5, 5]]
        # 定义一个足迹（footprint）数组，用于指定滤波器的形状
        footprint = [[1, 0, 1], [0, 1, 0]]
        # 创建一个 numpy 数组，作为输入数据，使用指定的数据类型 dtype
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 使用 ndimage.rank_filter 函数对数组进行排序滤波，指定滤波器大小为 1，使用指定的足迹
        # origin 参数设定为 -1，表示起始点的位置
        output = ndimage.rank_filter(array, 1, footprint=footprint,
                                     origin=-1)
        # 断言输出结果与期望结果的近似性
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    # 使用 pytest 的参数化装饰器，针对不同的数据类型 dtype 执行相同的测试
    def test_rank14(self, dtype):
        expected = [[3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5],
                    [5, 6, 6, 5, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 使用 ndimage.rank_filter 函数进行排序滤波，指定滤波器大小为 1，使用指定的足迹
        # origin 参数设定为 [-1, 0]，表示起始点的位置分别在两个轴上的偏移
        output = ndimage.rank_filter(array, 1, footprint=footprint,
                                     origin=[-1, 0])
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank15(self, dtype):
        expected = [[2, 3, 1, 4, 1],
                    [5, 3, 7, 1, 1],
                    [5, 5, 3, 3, 3]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = np.array([[3, 2, 5, 1, 4],
                          [5, 8, 3, 7, 1],
                          [5, 6, 9, 3, 5]], dtype)
        # 使用 ndimage.rank_filter 函数进行排序滤波，指定滤波器大小为 0，使用指定的足迹
        # origin 参数设定为 [-1, 0]，表示起始点的位置分别在两个轴上的偏移
        output = ndimage.rank_filter(array, 0, footprint=footprint,
                                     origin=[-1, 0])
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_generic_filter1d01(self, dtype):
        weights = np.array([1.1, 2.2, 3.3])

        def _filter_func(input, output, fltr, total):
            # 将过滤器 fltr 归一化为总和为 total
            fltr = fltr / total
            # 对输入数据 input 进行一维通用滤波操作，计算输出到 output
            for ii in range(input.shape[0] - 2):
                output[ii] = input[ii] * fltr[0]
                output[ii] += input[ii + 1] * fltr[1]
                output[ii] += input[ii + 2] * fltr[2]
        # 创建一个 numpy 数组作为输入数据，使用指定的数据类型 dtype
        a = np.arange(12, dtype=dtype)
        a.shape = (3, 4)
        # 使用 ndimage.correlate1d 函数进行一维卷积，使用归一化后的权重数组
        # origin 参数设定为 -1，表示起始点的位置
        r1 = ndimage.correlate1d(a, weights / weights.sum(), 0, origin=-1)
        # 使用 ndimage.generic_filter1d 函数进行一维通用滤波操作，传递自定义的滤波函数 _filter_func
        # axis 参数设定为 0，表示沿着第一个轴进行滤波
        # origin 参数设定为 -1，表示起始点的位置
        # extra_arguments 传递额外的参数 weights
        # extra_keywords 传递额外的关键字参数 {'total': weights.sum()}
        r2 = ndimage.generic_filter1d(
            a, _filter_func, 3, axis=0, origin=-1,
            extra_arguments=(weights,),
            extra_keywords={'total': weights.sum()})
        # 断言两种滤波方法得到的结果近似相等
        assert_array_almost_equal(r1, r2)
    # 定义一个测试方法，用于测试通用滤波器的功能，传入数据类型参数 dtype
    def test_generic_filter01(self, dtype):
        # 定义一个二维数组作为滤波器
        filter_ = np.array([[1.0, 2.0], [3.0, 4.0]])
        # 定义一个二维数组作为印迹（footprint）
        footprint = np.array([[1, 0], [0, 1]])
        # 定义一个包含权重值的数组
        cf = np.array([1., 4.])

        # 定义一个局部函数 _filter_func，用于计算给定缓冲区的加权和
        def _filter_func(buffer, weights, total=1.0):
            # 将权重数组标准化为总和为1.0的形式
            weights = cf / total
            # 返回缓冲区与权重数组的加权和
            return (buffer * weights).sum()

        # 创建一个包含12个元素的数组，并将其形状改为3行4列
        a = np.arange(12, dtype=dtype)
        a.shape = (3, 4)
        # 对数组 a 进行与 filter_ * footprint 相关的ND数组卷积计算，结果存储在 r1 中
        r1 = ndimage.correlate(a, filter_ * footprint)
        
        # 如果数据类型在 float_types 中，则将 r1 除以 5；否则将其整除 5
        if dtype in float_types:
            r1 /= 5
        else:
            r1 //= 5
        
        # 使用通用滤波器对数组 a 进行处理，使用 _filter_func 函数作为滤波器函数，
        # footprint 作为印迹，cf 作为额外参数传入，{'total': cf.sum()} 作为额外的关键字参数
        r2 = ndimage.generic_filter(
            a, _filter_func, footprint=footprint, extra_arguments=(cf,),
            extra_keywords={'total': cf.sum()})
        
        # 断言 r1 与 r2 的值近似相等
        assert_array_almost_equal(r1, r2)

        # 使用 assert_raises 检查通用滤波器在使用 mode 序列时是否会引发 RuntimeError 异常
        with assert_raises(RuntimeError):
            r2 = ndimage.generic_filter(
                a, _filter_func, mode=['reflect', 'reflect'],
                footprint=footprint, extra_arguments=(cf,),
                extra_keywords={'total': cf.sum()})

    # 使用 pytest.mark.parametrize 对 test_extend01 方法进行参数化测试
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1, 1, 2]),
         ('wrap', [3, 1, 2]),
         ('reflect', [1, 1, 2]),
         ('mirror', [2, 1, 2]),
         ('constant', [0, 1, 2])]
    )
    # 定义一个测试方法，用于测试 ndimage.correlate1d 方法在不同模式下的扩展行为
    def test_extend01(self, mode, expected_value):
        # 定义一个一维数组作为输入数据
        array = np.array([1, 2, 3])
        # 定义一个一维数组作为权重
        weights = np.array([1, 0])
        # 调用 ndimage.correlate1d 方法，对输入数组 array 进行一维卷积计算，
        # 使用 mode 参数指定边界条件，cval 参数指定填充值
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        # 断言计算结果与预期值 expected_value 相等
        assert_array_equal(output, expected_value)

    # 使用 pytest.mark.parametrize 对 test_extend02 方法进行参数化测试
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1, 1, 1]),
         ('wrap', [3, 1, 2]),
         ('reflect', [3, 3, 2]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    # 定义一个测试方法，用于测试 ndimage.correlate1d 方法在不同模式下的扩展行为
    def test_extend02(self, mode, expected_value):
        # 定义一个一维数组作为输入数据
        array = np.array([1, 2, 3])
        # 定义一个一维数组作为权重
        weights = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        # 调用 ndimage.correlate1d 方法，对输入数组 array 进行一维卷积计算，
        # 使用 mode 参数指定边界条件，cval 参数指定填充值
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        # 断言计算结果与预期值 expected_value 相等
        assert_array_equal(output, expected_value)

    # 使用 pytest.mark.parametrize 对 test_extend03 方法进行参数化测试
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [2, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 3, 3]),
         ('mirror', [2, 3, 2]),
         ('constant', [2, 3, 0])]
    )
    # 定义一个测试方法，用于测试 ndimage.correlate1d 方法在不同模式下的扩展行为
    def test_extend03(self, mode, expected_value):
        # 定义一个一维数组作为输入数据
        array = np.array([1, 2, 3])
        # 定义一个一维数组作为权重
        weights = np.array([0, 0, 1])
        # 调用 ndimage.correlate1d 方法，对输入数组 array 进行一维卷积计算，
        # 使用 mode 参数指定边界条件，cval 参数指定填充值
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        # 断言计算结果与预期值 expected_value 相等
        assert_array_equal(output, expected_value)

    # 使用 pytest.mark.parametrize 对 test_extend04 方法进行参数化测试
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    # 定义一个测试方法，用于测试 ndimage.correlate1d 方法在不同模式下的扩展行为
    def test_extend04(self, mode, expected_value):
        # 定义一个一维数组作为输入数据
        array = np.array([1, 2, 3])
        # 定义一个一维数组作为权重
        weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        # 调用 ndimage.correlate1d 方法，对输入数组 array 进行一维卷积计算，
        # 使用 mode 参数指定边界条件，cval 参数指定填充值
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        # 断言计算结果与预期值 expected_value 相等
        assert_array_equal(output, expected_value)
    # 使用 pytest 的 parametrize 装饰器，为 test_extend05 方法参数化测试
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]),
         ('wrap', [[9, 7, 8], [3, 1, 2], [6, 4, 5]]),
         ('reflect', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]),
         ('mirror', [[5, 4, 5], [2, 1, 2], [5, 4, 5]]),
         ('constant', [[0, 0, 0], [0, 1, 2], [0, 4, 5]])]
    )
    # 定义测试方法 test_extend05，参数 mode 和 expected_value 由 parametrize 提供
    def test_extend05(self, mode, expected_value):
        # 创建一个 3x3 的 NumPy 数组
        array = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        # 创建一个 2x2 的 NumPy 权重数组
        weights = np.array([[1, 0], [0, 0]])
        # 使用 ndimage 模块中的 correlate 函数对 array 应用权重 weights，设置模式为 mode，边界值为 0
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        # 断言 output 和 expected_value 相等
        assert_array_equal(output, expected_value)

    # 同上述 test_extend05 方法相同的注释模式
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]),
         ('wrap', [[5, 6, 4], [8, 9, 7], [2, 3, 1]]),
         ('reflect', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]),
         ('mirror', [[5, 6, 5], [8, 9, 8], [5, 6, 5]]),
         ('constant', [[5, 6, 0], [8, 9, 0], [0, 0, 0]])]
    )
    def test_extend06(self, mode, expected_value):
        array = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
        weights = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    # 同上述 test_extend05 方法相同的注释模式
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend07(self, mode, expected_value):
        array = np.array([1, 2, 3])
        weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    # 同上述 test_extend05 方法相同的注释模式
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[3], [3], [3]]),
         ('wrap', [[2], [3], [1]]),
         ('reflect', [[2], [1], [1]]),
         ('mirror', [[1], [2], [3]]),
         ('constant', [[0], [0], [0]])]
    )
    def test_extend08(self, mode, expected_value):
        array = np.array([[1], [2], [3]])
        weights = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    # 同上述 test_extend05 方法相同的注释模式
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend09(self, mode, expected_value):
        array = np.array([1, 2, 3])
        weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)
    # 使用 pytest 的参数化装饰器标记这个测试方法，测试不同模式下的扩展功能
    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[3], [3], [3]]),  # 最近邻插值模式的预期输出
         ('wrap', [[2], [3], [1]]),      # 包裹模式的预期输出
         ('reflect', [[2], [1], [1]]),   # 反射模式的预期输出
         ('mirror', [[1], [2], [3]]),    # 镜像模式的预期输出
         ('constant', [[0], [0], [0]])]  # 常数扩展模式的预期输出
    )
    # 定义一个测试方法，测试对数组进行权重卷积时的扩展行为
    def test_extend10(self, mode, expected_value):
        # 创建一个包含数组 [[1], [2], [3]] 的 NumPy 数组
        array = np.array([[1], [2], [3]])
        # 创建一个权重数组，用于卷积，包含了9个0和1个1
        weights = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        # 使用 SciPy 的 ndimage 模块进行卷积操作，设置边界模式为 mode，未指定值的填充为 0
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        # 断言输出结果与预期结果相等
        assert_array_equal(output, expected_value)
def test_ticket_701():
    # Test generic filter sizes
    # 创建一个2x2的NumPy数组
    arr = np.arange(4).reshape((2, 2))
    # 定义一个函数，返回数组中的最小值
    def func(x):
        return np.min(x)
    # 使用ndimage的generic_filter函数对数组arr进行通用过滤，指定过滤器大小为(1, 1)
    res = ndimage.generic_filter(arr, func, size=(1, 1))
    # 如果ticket 701未修复，则以下代码会引发错误
    # 用过滤器大小为1对数组arr进行通用过滤
    res2 = ndimage.generic_filter(arr, func, size=1)
    # 断言res和res2相等
    assert_equal(res, res2)


def test_gh_5430():
    # At least one of these raises an error unless gh-5430 is
    # fixed. In py2k an int is implemented using a C long, so
    # which one fails depends on your system. In py3k there is only
    # one arbitrary precision integer type, so both should fail.
    # 使用np.int32创建整数sigma
    sigma = np.int32(1)
    # 调用_nd_support._normalize_sequence函数，对sigma进行归一化处理
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    # 断言out与[sigma]相等
    assert_equal(out, [sigma])
    # 使用np.int64创建整数sigma
    sigma = np.int64(1)
    # 再次调用_normalize_sequence函数，对sigma进行归一化处理
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    # 断言out与[sigma]相等
    assert_equal(out, [sigma])
    # 这个例子之前可以正常工作，确保现在仍然可以正常工作
    sigma = 1
    # 再次调用_normalize_sequence函数，对sigma进行归一化处理
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    # 断言out与[sigma]相等
    assert_equal(out, [sigma])
    # 这个例子之前可以正常工作，确保现在仍然可以正常工作
    sigma = [1, 1]
    # 再次调用_normalize_sequence函数，对sigma进行归一化处理
    out = ndimage._ni_support._normalize_sequence(sigma, 2)
    # 断言out与sigma相等
    assert_equal(out, sigma)
    # 还包括原始的示例，以确保我们修复了这个问题
    # 创建一个256x256的随机正态分布数组
    x = np.random.normal(size=(256, 256))
    # 创建一个与x相同形状的全零数组perlin
    perlin = np.zeros_like(x)
    # 循环处理，对perlin数组进行高斯滤波
    for i in 2**np.arange(6):
        perlin += ndimage.gaussian_filter(x, i, mode="wrap") * i**2
    # 这也修复了gh-4106，展示原始示例现在可以运行
    # 使用np.int64创建整数x
    x = np.int64(21)
    # 再次调用_normalize_sequence函数，对x进行归一化处理
    ndimage._ni_support._normalize_sequence(x, 0)


def test_gaussian_kernel1d():
    # Check order inputs to Gaussians
    # 设置高斯核的半径和标准差
    radius = 10
    sigma = 2
    sigma2 = sigma * sigma
    # 创建一个从-radius到radius的浮点数数组x
    x = np.arange(-radius, radius + 1, dtype=np.double)
    # 计算一维高斯核phi_x
    phi_x = np.exp(-0.5 * x * x / sigma2)
    phi_x /= phi_x.sum()
    # 使用assert_allclose函数检查phi_x与_gaussian_kernel1d(sigma, 0, radius)的结果是否接近
    assert_allclose(phi_x, _gaussian_kernel1d(sigma, 0, radius))
    # 使用assert_allclose函数检查-phi_x * x / sigma2与_gaussian_kernel1d(sigma, 1, radius)的结果是否接近
    assert_allclose(-phi_x * x / sigma2, _gaussian_kernel1d(sigma, 1, radius))
    # 使用assert_allclose函数检查phi_x * (x * x / sigma2 - 1) / sigma2与_gaussian_kernel1d(sigma, 2, radius)的结果是否接近
    assert_allclose(phi_x * (x * x / sigma2 - 1) / sigma2,
                    _gaussian_kernel1d(sigma, 2, radius))
    # 使用assert_allclose函数检查phi_x * (3 - x * x / sigma2) * x / (sigma2 * sigma2)与_gaussian_kernel1d(sigma, 3, radius)的结果是否接近
    assert_allclose(phi_x * (3 - x * x / sigma2) * x / (sigma2 * sigma2),
                    _gaussian_kernel1d(sigma, 3, radius))


def test_orders_gauss():
    # Check order inputs to Gaussians
    # 创建一个长度为1的零数组arr
    arr = np.zeros((1,))
    # 使用ndimage的gaussian_filter函数，检查order输入
    assert_equal(0, ndimage.gaussian_filter(arr, 1, order=0))
    assert_equal(0, ndimage.gaussian_filter(arr, 1, order=3))
    # 使用assert_raises函数检查传入无效值时是否引发ValueError异常
    assert_raises(ValueError, ndimage.gaussian_filter, arr, 1, -1)
    # 使用ndimage的gaussian_filter1d函数，检查order输入
    assert_equal(0, ndimage.gaussian_filter1d(arr, 1, axis=-1, order=0))
    assert_equal(0, ndimage.gaussian_filter1d(arr, 1, axis=-1, order=3))
    # 使用assert_raises函数检查传入无效值时是否引发ValueError异常
    assert_raises(ValueError, ndimage.gaussian_filter1d, arr, 1, -1, -1)


def test_valid_origins():
    """Regression test for #1311."""
    # 定义一个函数，返回数组的均值
    def func(x):
        return np.mean(x)
    # 创建一个浮点数类型的数组data
    data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    # 使用assert_raises函数检查传入无效值时是否引发ValueError异常
    assert_raises(ValueError, ndimage.generic_filter, data, func, size=3,
                  origin=2)
    # 断言调用指定的函数和参数会引发 ValueError 异常，这里是 ndimage.generic_filter1d 函数
    # 使用特定的 data、func 参数，并设置 filter_size=3、origin=2
    assert_raises(ValueError, ndimage.generic_filter1d, data, func,
                  filter_size=3, origin=2)
    
    # 断言调用指定的函数和参数会引发 ValueError 异常，这里是 ndimage.percentile_filter 函数
    # 使用特定的 data、0.2 参数，并设置 size=3、origin=2
    assert_raises(ValueError, ndimage.percentile_filter, data, 0.2, size=3,
                  origin=2)
    
    # 遍历一个包含多个滤波器函数的列表，依次对 data 进行滤波处理
    for filter in [ndimage.uniform_filter, ndimage.minimum_filter,
                   ndimage.maximum_filter, ndimage.maximum_filter1d,
                   ndimage.median_filter, ndimage.minimum_filter1d]:
        # 这里测试 origin=-1 时调用 filter 函数是否正常
        list(filter(data, 3, origin=-1))
        # 这里测试 origin=1 时调用 filter 函数是否正常
        list(filter(data, 3, origin=1))
        # 断言调用 filter 函数并设置 origin=2 时会引发 ValueError 异常
        assert_raises(ValueError, filter, data, 3, origin=2)
# 定义一个测试函数，用于回归测试 GitHub 问题 #822
def test_bad_convolve_and_correlate_origins():
    """Regression test for gh-822."""
    # 在修复 GitHub 问题 #822 之前，以下情况会导致许多系统出现段错误或其他崩溃
    # 测试 ndimage.correlate1d 函数是否会引发 ValueError 异常，
    # 传入参数是两个列表和一个指定的起始位置 origin=2
    assert_raises(ValueError, ndimage.correlate1d,
                  [0, 1, 2, 3, 4, 5], [1, 1, 2, 0], origin=2)
    # 测试 ndimage.correlate 函数是否会引发 ValueError 异常，
    # 传入参数是两个列表和一个指定的起始位置 origin=[2]
    assert_raises(ValueError, ndimage.correlate,
                  [0, 1, 2, 3, 4, 5], [0, 1, 2], origin=[2])
    # 测试 ndimage.correlate 函数是否会引发 ValueError 异常，
    # 传入参数是两个二维数组和一个指定的起始位置 origin=[0, 1]
    assert_raises(ValueError, ndimage.correlate,
                  np.ones((3, 5)), np.ones((2, 2)), origin=[0, 1])

    # 测试 ndimage.convolve1d 函数是否会引发 ValueError 异常，
    # 传入参数是一个一维数组、一个滤波器和一个指定的起始位置 origin=-2
    assert_raises(ValueError, ndimage.convolve1d,
                  np.arange(10), np.ones(3), origin=-2)
    # 测试 ndimage.convolve 函数是否会引发 ValueError 异常，
    # 传入参数是一个一维数组、一个滤波器和一个指定的起始位置 origin=[-2]
    assert_raises(ValueError, ndimage.convolve,
                  np.arange(10), np.ones(3), origin=[-2])
    # 测试 ndimage.convolve 函数是否会引发 ValueError 异常，
    # 传入参数是一个二维数组、一个滤波器和一个指定的起始位置 origin=[0, -2]
    assert_raises(ValueError, ndimage.convolve,
                  np.ones((3, 5)), np.ones((2, 2)), origin=[0, -2])


def test_multiple_modes():
    # 测试多种模式下滤波器对不同维度的应用是否与单一模式下的结果相同
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    mode1 = 'reflect'
    mode2 = ['reflect', 'reflect']

    # 比较使用单一模式 mode1 和多模式 mode2 的高斯滤波器结果是否相等
    assert_equal(ndimage.gaussian_filter(arr, 1, mode=mode1),
                 ndimage.gaussian_filter(arr, 1, mode=mode2))
    # 比较使用单一模式 mode1 和多模式 mode2 的 Prewitt 滤波器结果是否相等
    assert_equal(ndimage.prewitt(arr, mode=mode1),
                 ndimage.prewitt(arr, mode=mode2))
    # 比较使用单一模式 mode1 和多模式 mode2 的 Sobel 滤波器结果是否相等
    assert_equal(ndimage.sobel(arr, mode=mode1),
                 ndimage.sobel(arr, mode=mode2))
    # 比较使用单一模式 mode1 和多模式 mode2 的 Laplace 滤波器结果是否相等
    assert_equal(ndimage.laplace(arr, mode=mode1),
                 ndimage.laplace(arr, mode=mode2))
    # 比较使用单一模式 mode1 和多模式 mode2 的 Gaussian-Laplace 滤波器结果是否相等
    assert_equal(ndimage.gaussian_laplace(arr, 1, mode=mode1),
                 ndimage.gaussian_laplace(arr, 1, mode=mode2))
    # 比较使用单一模式 mode1 和多模式 mode2 的最大值滤波器结果是否相等
    assert_equal(ndimage.maximum_filter(arr, size=5, mode=mode1),
                 ndimage.maximum_filter(arr, size=5, mode=mode2))
    # 比较使用单一模式 mode1 和多模式 mode2 的最小值滤波器结果是否相等
    assert_equal(ndimage.minimum_filter(arr, size=5, mode=mode1),
                 ndimage.minimum_filter(arr, size=5, mode=mode2))
    # 比较使用单一模式 mode1 和多模式 mode2 的高斯梯度幅值滤波器结果是否相等
    assert_equal(ndimage.gaussian_gradient_magnitude(arr, 1, mode=mode1),
                 ndimage.gaussian_gradient_magnitude(arr, 1, mode=mode2))
    # 比较使用单一模式 mode1 和多模式 mode2 的均匀滤波器结果是否相等
    assert_equal(ndimage.uniform_filter(arr, 5, mode=mode1),
                 ndimage.uniform_filter(arr, 5, mode=mode2))


def test_multiple_modes_sequentially():
    # 测试多种模式下滤波器是否与按顺序应用不同模式的滤波器结果相同
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    modes = ['reflect', 'wrap']

    # 期望的结果是先在 axis=0 上使用 modes[0] 模式的高斯滤波器，然后在 axis=1 上使用 modes[1] 模式
    expected = ndimage.gaussian_filter1d(arr, 1, axis=0, mode=modes[0])
    expected = ndimage.gaussian_filter1d(expected, 1, axis=1, mode=modes[1])
    # 比较使用 modes 模式和期望结果的高斯滤波器结果是否相等
    assert_equal(expected,
                 ndimage.gaussian_filter(arr, 1, mode=modes))
    # 对数组 expected 应用一维均匀滤波器，沿指定轴（axis=1）进行平滑处理，使用 modes[1] 模式
    expected = ndimage.uniform_filter1d(expected, 5, axis=1, mode=modes[1])
    # 使用 ndimage 中的均匀滤波器对数组 arr 进行平滑处理，窗口大小为 5，使用指定模式 modes
    assert_equal(expected,
                 ndimage.uniform_filter(arr, 5, mode=modes))

    # 对数组 arr 沿 axis=0 轴应用最大值滤波器，窗口大小为 5，使用 modes[0] 模式
    expected = ndimage.maximum_filter1d(arr, size=5, axis=0, mode=modes[0])
    # 对上一步操作后的结果 expected，沿 axis=1 轴再次应用最大值滤波器，窗口大小为 5，使用 modes[1] 模式
    expected = ndimage.maximum_filter1d(expected, size=5, axis=1,
                                        mode=modes[1])
    # 使用 ndimage 中的最大值滤波器对数组 arr 进行处理，窗口大小为 5，使用指定模式 modes
    assert_equal(expected,
                 ndimage.maximum_filter(arr, size=5, mode=modes))

    # 对数组 arr 沿 axis=0 轴应用最小值滤波器，窗口大小为 5，使用 modes[0] 模式
    expected = ndimage.minimum_filter1d(arr, size=5, axis=0, mode=modes[0])
    # 对上一步操作后的结果 expected，沿 axis=1 轴再次应用最小值滤波器，窗口大小为 5，使用 modes[1] 模式
    expected = ndimage.minimum_filter1d(expected, size=5, axis=1,
                                        mode=modes[1])
    # 使用 ndimage 中的最小值滤波器对数组 arr 进行处理，窗口大小为 5，使用指定模式 modes
    assert_equal(expected,
                 ndimage.minimum_filter(arr, size=5, mode=modes))
def test_multiple_modes_prewitt():
    # 测试多种外推模式下的 Prewitt 滤波器
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = np.array([[1., -3., 2.],
                         [1., -2., 1.],
                         [1., -1., 0.]])

    modes = ['reflect', 'wrap']

    # 断言预期输出与使用指定模式计算的 Prewitt 滤波结果相等
    assert_equal(expected,
                 ndimage.prewitt(arr, mode=modes))


def test_multiple_modes_sobel():
    # 测试多种外推模式下的 Sobel 滤波器
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = np.array([[1., -4., 3.],
                         [2., -3., 1.],
                         [1., -1., 0.]])

    modes = ['reflect', 'wrap']

    # 断言预期输出与使用指定模式计算的 Sobel 滤波结果相等
    assert_equal(expected,
                 ndimage.sobel(arr, mode=modes))


def test_multiple_modes_laplace():
    # 测试多种外推模式下的 Laplace 滤波器
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = np.array([[-2., 2., 1.],
                         [-2., -3., 2.],
                         [1., 1., 0.]])

    modes = ['reflect', 'wrap']

    # 断言预期输出与使用指定模式计算的 Laplace 滤波结果相等
    assert_equal(expected,
                 ndimage.laplace(arr, mode=modes))


def test_multiple_modes_gaussian_laplace():
    # 测试多种外推模式下的 Gaussian-Laplace 滤波器
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = np.array([[-0.28438687, 0.01559809, 0.19773499],
                         [-0.36630503, -0.20069774, 0.07483620],
                         [0.15849176, 0.18495566, 0.21934094]])

    modes = ['reflect', 'wrap']

    # 断言预期输出与使用指定模式计算的 Gaussian-Laplace 滤波结果相等
    assert_almost_equal(expected,
                        ndimage.gaussian_laplace(arr, 1, mode=modes))


def test_multiple_modes_gaussian_gradient_magnitude():
    # 测试多种外推模式下的 Gaussian 梯度幅值滤波器
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = np.array([[0.04928965, 0.09745625, 0.06405368],
                         [0.23056905, 0.14025305, 0.04550846],
                         [0.19894369, 0.14950060, 0.06796850]])

    modes = ['reflect', 'wrap']

    # 计算并获取使用指定模式计算的 Gaussian 梯度幅值滤波结果
    calculated = ndimage.gaussian_gradient_magnitude(arr, 1, mode=modes)

    # 断言预期输出与计算结果近似相等
    assert_almost_equal(expected, calculated)


def test_multiple_modes_uniform():
    # 测试多种外推模式下的均匀滤波器
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = np.array([[0.32, 0.40, 0.48],
                         [0.20, 0.28, 0.32],
                         [0.28, 0.32, 0.40]])

    modes = ['reflect', 'wrap']

    # 断言预期输出与使用指定模式计算的均匀滤波结果相等
    assert_almost_equal(expected,
                        ndimage.uniform_filter(arr, 5, mode=modes))
    # 测试高斯滤波器在不同宽度下的截断效果。
    # 这些测试仅检查结果具有预期数量的非零元素。

    # 创建一个大小为100x100的全零数组
    arr = np.zeros((100, 100), float)
    # 将数组中心位置的一个元素设为1，其余仍为0
    arr[50, 50] = 1
    # 对 arr 应用高斯滤波器，截断参数为2，统计大于0的元素个数
    num_nonzeros_2 = (ndimage.gaussian_filter(arr, 5, truncate=2) > 0).sum()
    # 断言：num_nonzeros_2 应等于 21^2
    assert_equal(num_nonzeros_2, 21**2)
    # 对 arr 应用高斯滤波器，截断参数为5，统计大于0的元素个数
    num_nonzeros_5 = (ndimage.gaussian_filter(arr, 5, truncate=5) > 0).sum()
    # 断言：num_nonzeros_5 应等于 51^2
    assert_equal(num_nonzeros_5, 51**2)

    # 测试当 sigma 是一个序列时的截断效果。
    # 对 arr 应用高斯滤波器，sigma 参数为 [0.5, 2.5]，截断参数为3.5
    f = ndimage.gaussian_filter(arr, [0.5, 2.5], truncate=3.5)
    # 找出大于0的位置
    fpos = f > 0
    # 沿着第一维度（列）统计大于0的元素个数
    n0 = fpos.any(axis=0).sum()
    # 断言：n0 应该等于 2*int(2.5*3.5 + 0.5) + 1
    assert_equal(n0, 19)
    # 沿着第二维度（行）统计大于0的元素个数
    n1 = fpos.any(axis=1).sum()
    # 断言：n1 应该等于 2*int(0.5*3.5 + 0.5) + 1
    assert_equal(n1, 5)

    # 测试 gaussian_filter1d
    # 创建一个大小为51的全零数组
    x = np.zeros(51)
    # 将数组中心位置的一个元素设为1，其余仍为0
    x[25] = 1
    # 对 x 应用一维高斯滤波器，sigma 参数为2，截断参数为3.5
    f = ndimage.gaussian_filter1d(x, sigma=2, truncate=3.5)
    # 统计大于0的元素个数
    n = (f > 0).sum()
    # 断言：n 应等于 15
    assert_equal(n, 15)

    # 测试 gaussian_laplace
    # 对 x 应用高斯拉普拉斯滤波器，sigma 参数为2，截断参数为3.5
    y = ndimage.gaussian_laplace(x, sigma=2, truncate=3.5)
    # 找出非零元素的索引
    nonzero_indices = np.nonzero(y != 0)[0]
    # 计算非零元素的范围长度加1
    n = np.ptp(nonzero_indices) + 1
    # 断言：n 应等于 15
    assert_equal(n, 15)

    # 测试 gaussian_gradient_magnitude
    # 对 x 应用高斯梯度幅值滤波器，sigma 参数为2，截断参数为3.5
    y = ndimage.gaussian_gradient_magnitude(x, sigma=2, truncate=3.5)
    # 找出非零元素的索引
    nonzero_indices = np.nonzero(y != 0)[0]
    # 计算非零元素的范围长度加1
    n = np.ptp(nonzero_indices) + 1
    # 断言：n 应等于 15
    assert_equal(n, 15)
# 定义测试函数，用于验证带有半径参数的高斯滤波器与相应截断参数的滤波器产生相同结果。

def test_gaussian_radius():
    # 创建一个长度为7的零向量，并将中间位置设为1，用于测试一维高斯滤波器
    x = np.zeros(7)
    x[3] = 1
    # 使用 sigma=2 和 truncate=1.5 参数对 x 进行一维高斯滤波
    f1 = ndimage.gaussian_filter1d(x, sigma=2, truncate=1.5)
    # 使用 sigma=2 和 radius=3 参数对 x 进行一维高斯滤波
    f2 = ndimage.gaussian_filter1d(x, sigma=2, radius=3)
    # 断言两次滤波的结果应该相等
    assert_equal(f1, f2)

    # 创建一个 9x9 的零矩阵，并将中心位置设为1，用于测试二维高斯滤波器
    a = np.zeros((9, 9))
    a[4, 4] = 1
    # 使用 sigma=0.5 和 truncate=3.5 参数对 a 进行二维高斯滤波
    f1 = ndimage.gaussian_filter(a, sigma=0.5, truncate=3.5)
    # 使用 sigma=0.5 和 radius=2 参数对 a 进行二维高斯滤波
    f2 = ndimage.gaussian_filter(a, sigma=0.5, radius=2)
    # 断言两次滤波的结果应该相等
    assert_equal(f1, f2)

    # 创建一个 50x50 的零矩阵，并将中心位置设为1，用于测试多尺度的二维高斯滤波器
    a = np.zeros((50, 50))
    a[25, 25] = 1
    # 使用 sigma=[0.5, 2.5] 和 truncate=3.5 参数对 a 进行多尺度的二维高斯滤波
    f1 = ndimage.gaussian_filter(a, sigma=[0.5, 2.5], truncate=3.5)
    # 使用 sigma=[0.5, 2.5] 和 radius=[2, 9] 参数对 a 进行多尺度的二维高斯滤波
    f2 = ndimage.gaussian_filter(a, sigma=[0.5, 2.5], radius=[2, 9])
    # 断言两次滤波的结果应该相等
    assert_equal(f1, f2)


def test_gaussian_radius_invalid():
    # 测试当 radius 参数为负数或小数时是否会引发 ValueError 异常

    # 断言当 radius 参数为负数时会引发 ValueError 异常
    with assert_raises(ValueError):
        ndimage.gaussian_filter1d(np.zeros(8), sigma=1, radius=-1)
    # 断言当 radius 参数为小数时会引发 ValueError 异常
    with assert_raises(ValueError):
        ndimage.gaussian_filter1d(np.zeros(8), sigma=1, radius=1.1)


class TestThreading:
    # 定义用于多线程测试的类

    def check_func_thread(self, n, fun, args, out):
        # 创建多线程并执行指定函数 fun

        # 生成 n 个线程，每个线程都调用 fun 函数，传入相应的参数和输出位置 out[x]
        thrds = [Thread(target=fun, args=args, kwargs={'output': out[x]})
                 for x in range(n)]
        # 启动所有线程
        [t.start() for t in thrds]
        # 等待所有线程执行完毕
        [t.join() for t in thrds]

    def check_func_serial(self, n, fun, args, out):
        # 创建串行执行的函数

        # 依次执行 n 次函数调用 fun(*args, output=out[i])
        for i in range(n):
            fun(*args, output=out[i])

    def test_correlate1d(self):
        # 测试一维相关操作

        d = np.random.randn(5000)  # 生成一个长度为5000的随机数列
        os = np.empty((4, d.size))  # 创建一个存储结果的数组
        ot = np.empty_like(os)      # 创建一个与 os 相同大小的空数组
        k = np.arange(5)            # 创建一个长度为5的数组
        # 使用串行方式对 d 和 k 执行一维相关操作，并将结果存入 os
        self.check_func_serial(4, ndimage.correlate1d, (d, k), os)
        # 使用多线程方式对 d 和 k 执行一维相关操作，并将结果存入 ot
        self.check_func_thread(4, ndimage.correlate1d, (d, k), ot)
        # 断言串行和多线程的操作结果应该相等
        assert_array_equal(os, ot)

    def test_correlate(self):
        # 测试二维相关操作

        d = np.random.randn(500, 500)  # 生成一个大小为500x500的随机矩阵
        k = np.random.randn(10, 10)    # 生成一个大小为10x10的随机矩阵
        os = np.empty([4] + list(d.shape))  # 创建一个存储结果的数组
        ot = np.empty_like(os)              # 创建一个与 os 相同大小的空数组
        # 使用串行方式对 d 和 k 执行二维相关操作，并将结果存入 os
        self.check_func_serial(4, ndimage.correlate, (d, k), os)
        # 使用多线程方式对 d 和 k 执行二维相关操作，并将结果存入 ot
        self.check_func_thread(4, ndimage.correlate, (d, k), ot)
        # 断言串行和多线程的操作结果应该相等
        assert_array_equal(os, ot)

    def test_median_filter(self):
        # 测试中值滤波器

        d = np.random.randn(500, 500)  # 生成一个大小为500x500的随机矩阵
        os = np.empty([4] + list(d.shape))  # 创建一个存储结果的数组
        ot = np.empty_like(os)              # 创建一个与 os 相同大小的空数组
        # 使用串行方式对 d 执行中值滤波，并将结果存入 os
        self.check_func_serial(4, ndimage.median_filter, (d, 3), os)
        # 使用多线程方式对 d 执行中值滤波，并将结果存入 ot
        self.check_func_thread(4, ndimage.median_filter, (d, 3), ot)
        # 断言串行和多线程的操作结果应该相等
        assert_array_equal(os, ot)

    def test_uniform_filter1d(self):
        # 测试一维均匀滤波器

        d = np.random.randn(5000)  # 生成一个长度为5000的随机数列
        os = np.empty((4, d.size))  # 创建一个存储结果的数组
        ot = np.empty_like(os)      # 创建一个与 os 相同大小的空数组
        # 使用串行方式对 d 执行一维均匀滤波，并将结果存入 os
        self.check_func_serial(4, ndimage.uniform_filter1d, (d, 5), os)
        # 使用多线程方式对 d 执行一维均匀滤波，并将结果存入 ot
        self.check_func_thread(4, ndimage.uniform_filter1d, (d, 5), ot)
        # 断言串行和多线程的操作结果应该相等
        assert_array_equal(os, ot)
    # 定义一个测试函数，用于测试最大最小滤波器的功能
    def test_minmax_filter(self):
        # 生成一个形状为 (500, 500) 的随机数数组
        d = np.random.randn(500, 500)
        # 创建一个形状为 [4, 500, 500] 的空数组 os
        os = np.empty([4] + list(d.shape))
        # 根据 os 的形状创建一个同样形状的空数组 ot
        ot = np.empty_like(os)
        # 使用单线程调用 ndimage 库的最大值滤波函数，并将结果存储在 os 中
        self.check_func_serial(4, ndimage.maximum_filter, (d, 3), os)
        # 使用多线程调用 ndimage 库的最大值滤波函数，并将结果存储在 ot 中
        self.check_func_thread(4, ndimage.maximum_filter, (d, 3), ot)
        # 断言 os 和 ot 数组内容是否完全相等
        assert_array_equal(os, ot)
        # 使用单线程调用 ndimage 库的最小值滤波函数，并将结果存储在 os 中
        self.check_func_serial(4, ndimage.minimum_filter, (d, 3), os)
        # 使用多线程调用 ndimage 库的最小值滤波函数，并将结果存储在 ot 中
        self.check_func_thread(4, ndimage.minimum_filter, (d, 3), ot)
        # 断言 os 和 ot 数组内容是否完全相等
        assert_array_equal(os, ot)
# 定义测试函数 test_minmaximum_filter1d，用于测试 ndimage 库中的 minimum_filter1d 和 maximum_filter1d 函数
def test_minmaximum_filter1d():
    # Regression gh-3898
    # 创建长度为 10 的数组 in_，包含 0 到 9 的整数
    in_ = np.arange(10)
    # 对数组 in_ 应用长度为 1 的 minimum_filter1d 函数
    out = ndimage.minimum_filter1d(in_, 1)
    # 断言函数处理后的输出与输入相等
    assert_equal(in_, out)
    # 对数组 in_ 应用长度为 1 的 maximum_filter1d 函数
    out = ndimage.maximum_filter1d(in_, 1)
    # 断言函数处理后的输出与输入相等
    assert_equal(in_, out)
    
    # Test reflect
    # 对数组 in_ 应用长度为 5 的 reflect 模式的 minimum_filter1d 函数
    out = ndimage.minimum_filter1d(in_, 5, mode='reflect')
    # 断言函数处理后的输出与预期列表相等
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
    # 对数组 in_ 应用长度为 5 的 reflect 模式的 maximum_filter1d 函数
    out = ndimage.maximum_filter1d(in_, 5, mode='reflect')
    # 断言函数处理后的输出与预期列表相等
    assert_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
    
    # Test constant
    # 对数组 in_ 应用长度为 5 的 constant 模式的 minimum_filter1d 函数，cval 设置为 -1
    out = ndimage.minimum_filter1d(in_, 5, mode='constant', cval=-1)
    # 断言函数处理后的输出与预期列表相等
    assert_equal([-1, -1, 0, 1, 2, 3, 4, 5, -1, -1], out)
    # 对数组 in_ 应用长度为 5 的 constant 模式的 maximum_filter1d 函数，cval 设置为 10
    out = ndimage.maximum_filter1d(in_, 5, mode='constant', cval=10)
    # 断言函数处理后的输出与预期列表相等
    assert_equal([10, 10, 4, 5, 6, 7, 8, 9, 10, 10], out)
    
    # Test nearest
    # 对数组 in_ 应用长度为 5 的 nearest 模式的 minimum_filter1d 函数
    out = ndimage.minimum_filter1d(in_, 5, mode='nearest')
    # 断言函数处理后的输出与预期列表相等
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
    # 对数组 in_ 应用长度为 5 的 nearest 模式的 maximum_filter1d 函数
    out = ndimage.maximum_filter1d(in_, 5, mode='nearest')
    # 断言函数处理后的输出与预期列表相等
    assert_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
    
    # Test wrap
    # 对数组 in_ 应用长度为 5 的 wrap 模式的 minimum_filter1d 函数
    out = ndimage.minimum_filter1d(in_, 5, mode='wrap')
    # 断言函数处理后的输出与预期列表相等
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 0, 0], out)
    # 对数组 in_ 应用长度为 5 的 wrap 模式的 maximum_filter1d 函数
    out = ndimage.maximum_filter1d(in_, 5, mode='wrap')
    # 断言函数处理后的输出与预期列表相等
    assert_equal([9, 9, 4, 5, 6, 7, 8, 9, 9, 9], out)


# 定义测试函数 test_uniform_filter1d_roundoff_errors，用于测试 ndimage 库中的 uniform_filter1d 函数
def test_uniform_filter1d_roundoff_errors():
    # gh-6930
    # 创建长度为 27 的数组 in_，包含 0, 1, 0 这个序列重复 9 次
    in_ = np.repeat([0, 1, 0], [9, 9, 9])
    # 对数组 in_ 应用 filter_size 从 3 到 9 的 uniform_filter1d 函数
    for filter_size in range(3, 10):
        out = ndimage.uniform_filter1d(in_, filter_size)
        # 断言函数处理后的输出的总和等于 10 减去 filter_size
        assert_equal(out.sum(), 10 - filter_size)


# 定义测试函数 test_footprint_all_zeros，用于测试 ndimage 库中的 maximum_filter 函数
def test_footprint_all_zeros():
    # regression test for gh-6876: footprint of all zeros segfaults
    # 创建一个大小为 (100, 100) 的随机整数数组 arr
    arr = np.random.randint(0, 100, (100, 100))
    # 创建一个 3x3 全为 False 的二维数组 kernel
    kernel = np.zeros((3, 3), bool)
    # 断言调用 maximum_filter 函数时，使用全零的 kernel 会抛出 ValueError 异常
    with assert_raises(ValueError):
        ndimage.maximum_filter(arr, footprint=kernel)


# 定义测试函数 test_gaussian_filter，用于测试 ndimage 库中的 gaussian_filter 函数
def test_gaussian_filter():
    # Test gaussian filter with np.float16
    # gh-8207
    # 创建一个包含单个元素 1 的 np.float16 类型的数组 data
    data = np.array([1], dtype=np.float16)
    sigma = 1.0
    # 断言调用 gaussian_filter 函数时，使用 np.float16 类型的数组会抛出 RuntimeError 异常
    with assert_raises(RuntimeError):
        ndimage.gaussian_filter(data, sigma)


# 定义测试函数 test_rank_filter_noninteger_rank，用于测试 ndimage 库中的 rank_filter 函数
def test_rank_filter_noninteger_rank():
    # regression test for issue 9388: ValueError for
    # non integer rank when performing rank_filter
    # 创建一个大小为 (10, 20, 30) 的随机浮点数数组 arr
    arr = np.random.random((10, 20, 30))
    # 断言调用 rank_filter 函数时，使用非整数 rank 会抛出 TypeError 异常
    assert_raises(TypeError, ndimage.rank_filter, arr, 0.5,
                  footprint=np.ones((1, 1, 10), dtype=bool))


# 定义测试函数 test_size_footprint_both_set，用于测试 ndimage 库中的 rank_filter 函数
def test_size_footprint_both_set():
    # test for input validation, expect user warning when
    # size and footprint is set
    # 使用 suppress_warnings 上下文管理器
    with suppress_warnings() as sup:
        # 过滤掉用户警告 "ignoring size because footprint is set"
        sup.filter(UserWarning,
                   "ignoring size because footprint is set")
        # 创建一个大小为 (10, 20, 30) 的随
    # 使用断言检查两个数组 `ref` 和 `t` 是否几乎相等
    assert_array_almost_equal(ref, t)
```