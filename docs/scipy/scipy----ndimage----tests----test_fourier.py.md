# `D:\src\scipysrc\scipy\scipy\ndimage\tests\test_fourier.py`

```
import numpy as np
from numpy import fft
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_equal)

import pytest

from scipy import ndimage


class TestNdimageFourier:

    @pytest.mark.parametrize('shape', [(32, 16), (31, 15), (1, 10)])
    @pytest.mark.parametrize('dtype, dec', [(np.float32, 6), (np.float64, 14)])
    def test_fourier_gaussian_real01(self, shape, dtype, dec):
        a = np.zeros(shape, dtype)  # 创建一个指定形状和数据类型的全零数组a
        a[0, 0] = 1.0  # 将数组a的第一个元素设为1.0
        a = fft.rfft(a, shape[0], 0)  # 对a进行实部的快速傅里叶变换，沿第0轴进行操作
        a = fft.fft(a, shape[1], 1)   # 对a进行复数的快速傅里叶变换，沿第1轴进行操作
        a = ndimage.fourier_gaussian(a, [5.0, 2.5], shape[0], 0)  # 对a进行傅里叶高斯滤波
        a = fft.ifft(a, shape[1], 1)  # 对a进行复数的逆快速傅里叶变换，沿第1轴进行操作
        a = fft.irfft(a, shape[0], 0)  # 对a进行实部的逆快速傅里叶变换，沿第0轴进行操作
        assert_almost_equal(ndimage.sum(a), 1, decimal=dec)  # 断言a的总和近似为1，精确度为dec

    @pytest.mark.parametrize('shape', [(32, 16), (31, 15)])
    @pytest.mark.parametrize('dtype, dec', [(np.complex64, 6), (np.complex128, 14)])
    def test_fourier_gaussian_complex01(self, shape, dtype, dec):
        a = np.zeros(shape, dtype)  # 创建一个指定形状和数据类型的全零数组a
        a[0, 0] = 1.0  # 将数组a的第一个元素设为1.0
        a = fft.fft(a, shape[0], 0)  # 对a进行复数的快速傅里叶变换，沿第0轴进行操作
        a = fft.fft(a, shape[1], 1)  # 对a进行复数的快速傅里叶变换，沿第1轴进行操作
        a = ndimage.fourier_gaussian(a, [5.0, 2.5], -1, 0)  # 对a进行傅里叶高斯滤波
        a = fft.ifft(a, shape[1], 1)  # 对a进行复数的逆快速傅里叶变换，沿第1轴进行操作
        a = fft.ifft(a, shape[0], 0)  # 对a进行复数的逆快速傅里叶变换，沿第0轴进行操作
        assert_almost_equal(ndimage.sum(a.real), 1.0, decimal=dec)  # 断言a的实部的总和近似为1.0，精确度为dec

    @pytest.mark.parametrize('shape', [(32, 16), (31, 15), (1, 10)])
    @pytest.mark.parametrize('dtype, dec', [(np.float32, 6), (np.float64, 14)])
    def test_fourier_uniform_real01(self, shape, dtype, dec):
        a = np.zeros(shape, dtype)  # 创建一个指定形状和数据类型的全零数组a
        a[0, 0] = 1.0  # 将数组a的第一个元素设为1.0
        a = fft.rfft(a, shape[0], 0)  # 对a进行实部的快速傅里叶变换，沿第0轴进行操作
        a = fft.fft(a, shape[1], 1)   # 对a进行复数的快速傅里叶变换，沿第1轴进行操作
        a = ndimage.fourier_uniform(a, [5.0, 2.5], shape[0], 0)  # 对a进行傅里叶均匀滤波
        a = fft.ifft(a, shape[1], 1)  # 对a进行复数的逆快速傅里叶变换，沿第1轴进行操作
        a = fft.irfft(a, shape[0], 0)  # 对a进行实部的逆快速傅里叶变换，沿第0轴进行操作
        assert_almost_equal(ndimage.sum(a), 1.0, decimal=dec)  # 断言a的总和近似为1.0，精确度为dec

    @pytest.mark.parametrize('shape', [(32, 16), (31, 15)])
    @pytest.mark.parametrize('dtype, dec', [(np.complex64, 6), (np.complex128, 14)])
    def test_fourier_uniform_complex01(self, shape, dtype, dec):
        a = np.zeros(shape, dtype)  # 创建一个指定形状和数据类型的全零数组a
        a[0, 0] = 1.0  # 将数组a的第一个元素设为1.0
        a = fft.fft(a, shape[0], 0)  # 对a进行复数的快速傅里叶变换，沿第0轴进行操作
        a = fft.fft(a, shape[1], 1)  # 对a进行复数的快速傅里叶变换，沿第1轴进行操作
        a = ndimage.fourier_uniform(a, [5.0, 2.5], -1, 0)  # 对a进行傅里叶均匀滤波
        a = fft.ifft(a, shape[1], 1)  # 对a进行复数的逆快速傅里叶变换，沿第1轴进行操作
        a = fft.ifft(a, shape[0], 0)  # 对a进行复数的逆快速傅里叶变换，沿第0轴进行操作
        assert_almost_equal(ndimage.sum(a.real), 1.0, decimal=dec)  # 断言a的实部的总和近似为1.0，精确度为dec
    # 定义一个测试函数，用于测试实数数据的傅里叶变换和平移操作
    def test_fourier_shift_real01(self, shape, dtype, dec):
        # 创建一个期望的数据数组，其数值为从0到数组元素总数的序列，数据类型为指定的dtype
        expected = np.arange(shape[0] * shape[1], dtype=dtype)
        # 将一维数组转换为指定形状的二维数组
        expected.shape = shape
        # 对期望数据进行傅里叶变换，返回实数数据的傅里叶变换结果
        a = fft.rfft(expected, shape[0], 0)
        # 对傅里叶变换结果再进行傅里叶变换，沿指定轴进行操作
        a = fft.fft(a, shape[1], 1)
        # 对二维数据进行傅里叶平移操作，向右下方向移动一个单位
        a = ndimage.fourier_shift(a, [1, 1], shape[0], 0)
        # 对平移后的数据进行逆傅里叶变换
        a = fft.ifft(a, shape[1], 1)
        # 对逆傅里叶变换结果再进行逆实数傅里叶变换
        a = fft.irfft(a, shape[0], 0)
        # 断言函数的输出与期望数据的减去最后一行和最后一列的结果的近似程度，精度为dec
        assert_array_almost_equal(a[1:, 1:], expected[:-1, :-1], decimal=dec)
        # 断言函数的虚部与全零数组的近似程度，精度为dec
        assert_array_almost_equal(a.imag, np.zeros(shape), decimal=dec)

    # 为测试函数指定不同的形状和数据类型，用于测试复数数据的傅里叶变换和平移操作
    @pytest.mark.parametrize('shape', [(32, 16), (31, 15)])
    @pytest.mark.parametrize('dtype, dec', [(np.complex64, 4), (np.complex128, 11)])
    def test_fourier_shift_complex01(self, shape, dtype, dec):
        # 创建一个期望的复数数据数组，其数值为从0到数组元素总数的序列，数据类型为指定的dtype
        expected = np.arange(shape[0] * shape[1], dtype=dtype)
        # 将一维数组转换为指定形状的二维数组
        expected.shape = shape
        # 对期望数据进行傅里叶变换，返回复数数据的傅里叶变换结果
        a = fft.fft(expected, shape[0], 0)
        # 对傅里叶变换结果再进行傅里叶变换，沿指定轴进行操作
        a = fft.fft(a, shape[1], 1)
        # 对二维数据进行傅里叶平移操作，向右下方向移动一个单位
        a = ndimage.fourier_shift(a, [1, 1], -1, 0)
        # 对平移后的数据进行逆傅里叶变换
        a = fft.ifft(a, shape[1], 1)
        # 对逆傅里叶变换结果再进行逆傅里叶变换
        a = fft.ifft(a, shape[0], 0)
        # 断言函数的实部的输出与期望数据的减去最后一行和最后一列的结果的近似程度，精度为dec
        assert_array_almost_equal(a.real[1:, 1:], expected[:-1, :-1], decimal=dec)
        # 断言函数的虚部与全零数组的近似程度，精度为dec
        assert_array_almost_equal(a.imag, np.zeros(shape), decimal=dec)

    # 定义一个测试函数，用于测试实数数据的傅里叶变换和椭球操作
    @pytest.mark.parametrize('shape', [(32, 16), (31, 15), (1, 10)])
    @pytest.mark.parametrize('dtype, dec', [(np.float32, 5), (np.float64, 14)])
    def test_fourier_ellipsoid_real01(self, shape, dtype, dec):
        # 创建一个形状为shape，数据类型为dtype的全零数组
        a = np.zeros(shape, dtype)
        # 将数组中第一个元素设为1.0
        a[0, 0] = 1.0
        # 对数据进行实数傅里叶变换
        a = fft.rfft(a, shape[0], 0)
        # 对傅里叶变换结果再进行傅里叶变换，沿指定轴进行操作
        a = fft.fft(a, shape[1], 1)
        # 对二维数据进行傅里叶椭球操作，以指定的半径进行形状调整
        a = ndimage.fourier_ellipsoid(a, [5.0, 2.5], shape[0], 0)
        # 对椭球操作后的数据进行逆傅里叶变换
        a = fft.ifft(a, shape[1], 1)
        # 对逆傅里叶变换结果再进行逆实数傅里叶变换
        a = fft.irfft(a, shape[0], 0)
        # 断言函数对数组的元素求和结果与1.0的近似程度，精度为dec
        assert_almost_equal(ndimage.sum(a), 1.0, decimal=dec)

    # 为测试函数指定不同的形状和数据类型，用于测试复数数据的傅里叶变换和椭球操作
    @pytest.mark.parametrize('shape', [(32, 16), (31, 15)])
    @pytest.mark.parametrize('dtype, dec', [(np.complex64, 5), (np.complex128, 14)])
    def test_fourier_ellipsoid_complex01(self, shape, dtype, dec):
        # 创建一个形状为shape，数据类型为dtype的全零数组
        a = np.zeros(shape, dtype)
        # 将数组中第一个元素设为1.0
        a[0, 0] = 1.0
        # 对数据进行傅里叶变换
        a = fft.fft(a, shape[0], 0)
        # 对傅里叶变换结果再进行傅里叶变换，沿指定轴进行操作
        a = fft.fft(a, shape[1], 1)
        # 对二维数据进行傅里叶椭球操作，以指定的半径进行形状调整
        a = ndimage.fourier_ellipsoid(a, [5.0, 2.5], -1, 0)
        # 对椭球操作后的数据进行逆傅里叶变换
        a = fft.ifft(a, shape[1], 1)
        # 对逆傅里叶变换结果再进行逆傅里叶变换
        a = fft.ifft(a, shape[0], 0)
        # 断言函数对实部的输出的元素求和结果与1.0的近似程度，精度为dec
        assert_almost_equal(ndimage.sum(a.real), 1.0, decimal=dec)

    # 定义一个测试函数，用于测试对于维度大于3的数组进行傅里叶椭球操作时是否抛出NotImplementedError异常
    def test_fourier_ellipsoid_unimplemented_ndim(self):
        # 创建一个维度为4的全一数组，数据类型为复数128位浮点数
        x = np.ones((4, 6, 8, 10), dtype=np.complex128)
        # 使用pytest的断言，验证执行函数时是否会抛出NotImplementedError异常
        with pytest.raises(NotImplementedError):
            # 调用ndimage.fourier_ellipsoid函数，对维度大于3的数组进行椭球操作
            ndimage.fourier_ellipsoid(x, 3)
    # 定义测试方法，用于测试一维复杂椭球的傅里叶变换结果是否与均匀分布的傅里叶变换结果相同
    def test_fourier_ellipsoid_1d_complex(self):
        # 对于形状分别为(32,)和(31,)的数组进行测试
        for shape in [(32, ), (31, )]:
            # 遍历数据类型为复数的 np.complex64 和 np.complex128
            for type_, dec in zip([np.complex64, np.complex128], [5, 14]):
                # 创建一个指定形状和数据类型的全1数组
                x = np.ones(shape, dtype=type_)
                # 计算椭球的傅里叶变换，参数分别为：x数组，椭球的半径，旋转角度，倾斜角度
                a = ndimage.fourier_ellipsoid(x, 5, -1, 0)
                # 计算均匀分布的傅里叶变换，参数同上
                b = ndimage.fourier_uniform(x, 5, -1, 0)
                # 断言两个数组几乎相等，允许的最大差值由 decimal 参数指定
                assert_array_almost_equal(a, b, decimal=dec)

    # 使用 pytest 的参数化装饰器指定多组参数来测试零长度维度的情况
    @pytest.mark.parametrize('shape', [(0, ), (0, 10), (10, 0)])
    # 同时参数化数据类型为 np.float32, np.float64, np.complex64, np.complex128
    @pytest.mark.parametrize('dtype', [np.float32, np.float64,
                                       np.complex64, np.complex128])
    # 参数化测试函数，分别为椭球的傅里叶变换、高斯函数的傅里叶变换和均匀分布的傅里叶变换
    @pytest.mark.parametrize('test_func',
                             [ndimage.fourier_ellipsoid,
                              ndimage.fourier_gaussian,
                              ndimage.fourier_uniform])
    # 测试函数：测试零长度维度的情况
    def test_fourier_zero_length_dims(self, shape, dtype, test_func):
        # 创建一个指定形状和数据类型的全1数组
        a = np.ones(shape, dtype)
        # 调用指定的傅里叶变换函数，参数为全1数组 a 和半径为 3
        b = test_func(a, 3)
        # 断言数组 a 和 b 相等
        assert_equal(a, b)
```