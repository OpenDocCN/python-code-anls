# `D:\src\scipysrc\scipy\scipy\fftpack\tests\test_real_transforms.py`

```
# 导入必要的模块和函数
from os.path import join, dirname  # 从os.path模块导入join和dirname函数

import numpy as np  # 导入NumPy库，使用别名np
from numpy.testing import assert_array_almost_equal, assert_equal  # 导入NumPy的测试函数
import pytest  # 导入pytest库
from pytest import raises as assert_raises  # 从pytest库导入raises函数并重命名为assert_raises

from scipy.fftpack._realtransforms import (  # 从SciPy的fftpack模块导入一系列变换函数
    dct, idct, dst, idst, dctn, idctn, dstn, idstn)

# Matlab参考数据
MDATA = np.load(join(dirname(__file__), 'test.npz'))  # 加载Matlab测试数据
X = [MDATA['x%d' % i] for i in range(8)]  # 创建X列表，包含MDATA中的x0到x7数据
Y = [MDATA['y%d' % i] for i in range(8)]  # 创建Y列表，包含MDATA中的y0到y7数据

# FFTW参考数据：数据组织如下
#    * SIZES是一个包含所有可用大小的数组
#    * 对于每种类型（1, 2, 3, 4）和每个大小，数组dct_type_size包含应用于输入np.linspace(0, size-1, size)的DCT的输出
FFTWDATA_DOUBLE = np.load(join(dirname(__file__), 'fftw_double_ref.npz'))  # 加载FFTW双精度参考数据
FFTWDATA_SINGLE = np.load(join(dirname(__file__), 'fftw_single_ref.npz'))  # 加载FFTW单精度参考数据
FFTWDATA_SIZES = FFTWDATA_DOUBLE['sizes']  # 获取FFTW双精度参考数据中的sizes数组


def fftw_dct_ref(type, size, dt):
    """计算DCT的参考值用于测试."""
    x = np.linspace(0, size-1, size).astype(dt)  # 生成一个0到size-1的等差数列，并转换为指定数据类型
    dt = np.result_type(np.float32, dt)  # 获取给定数据类型的结果类型
    if dt == np.float64:
        data = FFTWDATA_DOUBLE  # 如果数据类型是双精度浮点数，使用双精度参考数据
    elif dt == np.float32:
        data = FFTWDATA_SINGLE  # 如果数据类型是单精度浮点数，使用单精度参考数据
    else:
        raise ValueError()  # 如果数据类型不是浮点数，抛出值错误异常
    y = (data['dct_%d_%d' % (type, size)]).astype(dt)  # 获取对应于给定类型和大小的DCT参考值，并转换为指定数据类型
    return x, y, dt  # 返回生成的输入数组x，参考输出数组y，以及数据类型dt


def fftw_dst_ref(type, size, dt):
    """计算DST的参考值用于测试."""
    x = np.linspace(0, size-1, size).astype(dt)  # 生成一个0到size-1的等差数列，并转换为指定数据类型
    dt = np.result_type(np.float32, dt)  # 获取给定数据类型的结果类型
    if dt == np.float64:
        data = FFTWDATA_DOUBLE  # 如果数据类型是双精度浮点数，使用双精度参考数据
    elif dt == np.float32:
        data = FFTWDATA_SINGLE  # 如果数据类型是单精度浮点数，使用单精度参考数据
    else:
        raise ValueError()  # 如果数据类型不是浮点数，抛出值错误异常
    y = (data['dst_%d_%d' % (type, size)]).astype(dt)  # 获取对应于给定类型和大小的DST参考值，并转换为指定数据类型
    return x, y, dt  # 返回生成的输入数组x，参考输出数组y，以及数据类型dt


def dct_2d_ref(x, **kwargs):
    """计算用于测试dct2的参考值."""
    x = np.array(x, copy=True)  # 深拷贝输入数组x
    for row in range(x.shape[0]):  # 对于输入数组的每一行
        x[row, :] = dct(x[row, :], **kwargs)  # 对行应用一维DCT变换
    for col in range(x.shape[1]):  # 对于输入数组的每一列
        x[:, col] = dct(x[:, col], **kwargs)  # 对列应用一维DCT变换
    return x  # 返回应用二维DCT变换后的数组


def idct_2d_ref(x, **kwargs):
    """计算用于测试idct2的参考值."""
    x = np.array(x, copy=True)  # 深拷贝输入数组x
    for row in range(x.shape[0]):  # 对于输入数组的每一行
        x[row, :] = idct(x[row, :], **kwargs)  # 对行应用一维IDCT变换
    for col in range(x.shape[1]):  # 对于输入数组的每一列
        x[:, col] = idct(x[:, col], **kwargs)  # 对列应用一维IDCT变换
    return x  # 返回应用二维IDCT变换后的数组


def dst_2d_ref(x, **kwargs):
    """计算用于测试dst2的参考值."""
    x = np.array(x, copy=True)  # 深拷贝输入数组x
    for row in range(x.shape[0]):  # 对于输入数组的每一行
        x[row, :] = dst(x[row, :], **kwargs)  # 对行应用一维DST变换
    for col in range(x.shape[1]):  # 对于输入数组的每一列
        x[:, col] = dst(x[:, col], **kwargs)  # 对列应用一维DST变换
    return x  # 返回应用二维DST变换后的数组


def idst_2d_ref(x, **kwargs):
    """计算用于测试idst2的参考值."""
    x = np.array(x, copy=True)  # 深拷贝输入数组x
    for row in range(x.shape[0]):  # 对于输入数组的每一行
        x[row, :] = idst(x[row, :], **kwargs)  # 对行应用一维IDST变换
    for col in range(x.shape[1]):  # 对于输入数组的每一列
        x[:, col] = idst(x[:, col], **kwargs)  # 对列应用一维IDST变换
    return x  # 返回应用二维IDST变换后的数组


def naive_dct1(x, norm=None):
    """计算DCT-I的文本书籍定义版本."""
    x = np.array(x, copy=True)  # 深拷贝输入数组x
    N = len(x)  # 获取数组x的长度
    M = N-1  # 计算M为N-1
    y = np.zeros(N)  # 创建一个长度为N的零数组y
    m0, m = 1, 2  # 初始化m0和m为1和2
    # 如果规范为'ortho'，则进行正交变换的系数初始化
    if norm == 'ortho':
        m0 = np.sqrt(1.0/M)  # 初始化 m0 为 1/M 的平方根
        m = np.sqrt(2.0/M)   # 初始化 m 为 2/M 的平方根
    
    # 对每个输出元素 k 进行离散余弦变换
    for k in range(N):
        # 对输入信号 x 的每个元素 n 进行加权和的计算
        for n in range(1, N-1):
            y[k] += m * x[n] * np.cos(np.pi * n * k / M)  # 加权和的一部分
        y[k] += m0 * x[0]  # 加上对第一个输入元素的加权和
        y[k] += m0 * x[N-1] * (1 if k % 2 == 0 else -1)  # 加上对最后一个输入元素的加权和（根据 k 的奇偶性决定正负）
    
    # 如果规范为'ortho'，进行正交变换后的归一化处理
    if norm == 'ortho':
        y[0] *= 1 / np.sqrt(2)    # 归一化第一个输出元素
        y[N-1] *= 1 / np.sqrt(2)  # 归一化最后一个输出元素
    
    # 返回离散余弦变换后的输出向量
    return y
# 计算DST-I的经典定义版本
def naive_dst1(x, norm=None):
    # 创建输入数组的副本
    x = np.array(x, copy=True)
    # 获取数组长度
    N = len(x)
    # M为N+1
    M = N + 1
    # 初始化输出数组
    y = np.zeros(N)
    # 双重循环计算DST-I公式
    for k in range(N):
        for n in range(N):
            y[k] += 2 * x[n] * np.sin(np.pi * (n + 1.0) * (k + 1.0) / M)
    # 如果启用正交归一化
    if norm == 'ortho':
        y *= np.sqrt(0.5 / M)
    return y


# 计算DCT-IV的经典定义版本
def naive_dct4(x, norm=None):
    # 创建输入数组的副本
    x = np.array(x, copy=True)
    # 获取数组长度
    N = len(x)
    # 初始化输出数组
    y = np.zeros(N)
    # 双重循环计算DCT-IV公式
    for k in range(N):
        for n in range(N):
            y[k] += x[n] * np.cos(np.pi * (n + 0.5) * (k + 0.5) / (N))
    # 如果启用正交归一化
    if norm == 'ortho':
        y *= np.sqrt(2.0 / N)
    else:
        y *= 2
    return y


# 计算DST-IV的经典定义版本
def naive_dst4(x, norm=None):
    # 创建输入数组的副本
    x = np.array(x, copy=True)
    # 获取数组长度
    N = len(x)
    # 初始化输出数组
    y = np.zeros(N)
    # 双重循环计算DST-IV公式
    for k in range(N):
        for n in range(N):
            y[k] += x[n] * np.sin(np.pi * (n + 0.5) * (k + 0.5) / (N))
    # 如果启用正交归一化
    if norm == 'ortho':
        y *= np.sqrt(2.0 / N)
    else:
        y *= 2
    return y


class TestComplex:
    # 测试复数类型的DCT
    def test_dct_complex64(self):
        # 计算虚数数组的DCT，并验证结果
        y = dct(1j * np.arange(5, dtype=np.complex64))
        x = 1j * dct(np.arange(5))
        assert_array_almost_equal(x, y)

    # 测试复数类型的DCT
    def test_dct_complex(self):
        # 计算虚数数组的DCT，并验证结果
        y = dct(np.arange(5) * 1j)
        x = 1j * dct(np.arange(5))
        assert_array_almost_equal(x, y)

    # 测试复数类型的IDCT
    def test_idct_complex(self):
        # 计算虚数数组的IDCT，并验证结果
        y = idct(np.arange(5) * 1j)
        x = 1j * idct(np.arange(5))
        assert_array_almost_equal(x, y)

    # 测试复数类型的DST
    def test_dst_complex64(self):
        # 计算虚数数组的DST，并验证结果
        y = dst(np.arange(5, dtype=np.complex64) * 1j)
        x = 1j * dst(np.arange(5))
        assert_array_almost_equal(x, y)

    # 测试复数类型的DST
    def test_dst_complex(self):
        # 计算虚数数组的DST，并验证结果
        y = dst(np.arange(5) * 1j)
        x = 1j * dst(np.arange(5))
        assert_array_almost_equal(x, y)

    # 测试复数类型的IDST
    def test_idst_complex(self):
        # 计算虚数数组的IDST，并验证结果
        y = idst(np.arange(5) * 1j)
        x = 1j * idst(np.arange(5))
        assert_array_almost_equal(x, y)


class _TestDCTBase:
    # 设置测试前的准备工作
    def setup_method(self):
        # 初始化变量
        self.rdt = None
        self.dec = 14
        self.type = None

    # 测试DCT的定义
    def test_definition(self):
        # 对每个FFTWDATA_SIZES中的大小进行测试
        for i in FFTWDATA_SIZES:
            # 使用fftw_dct_ref参考函数计算DCT，并验证结果
            x, yr, dt = fftw_dct_ref(self.type, i, self.rdt)
            y = dct(x, type=self.type)
            # 验证输出类型是否正确
            assert_equal(y.dtype, dt)
            # 由于fft使用更好的算法导致误差传播不同，这里除以np.max(y)来修正，应使用assert_array_approx_equal
            assert_array_almost_equal(y / np.max(y), yr / np.max(y), decimal=self.dec,
                                      err_msg="Size %d failed" % i)
    # 定义一个测试方法，用于测试离散余弦变换函数在不同情况下的表现
    def test_axis(self):
        # 设置变量 nt 为 2，表示测试两组数据
        nt = 2
        # 对于列表中的每个元素 i，依次执行以下操作
        for i in [7, 8, 9, 16, 32, 64]:
            # 生成一个大小为 (nt, i) 的随机正态分布数组 x
            x = np.random.randn(nt, i)
            # 对数组 x 进行离散余弦变换，类型由 self.type 决定，返回结果为 y
            y = dct(x, type=self.type)
            # 对于每个 j 在范围 [0, nt) 内，执行以下断言操作
            for j in range(nt):
                # 断言 y[j] 与 dct(x[j], type=self.type) 的元素近似相等，
                # 其精度由 self.dec 决定
                assert_array_almost_equal(y[j], dct(x[j], type=self.type),
                                          decimal=self.dec)

            # 将数组 x 进行转置操作
            x = x.T
            # 对转置后的数组 x 进行离散余弦变换，沿轴 0 进行变换，类型为 self.type
            y = dct(x, axis=0, type=self.type)
            # 对于每个 j 在范围 [0, nt) 内，执行以下断言操作
            for j in range(nt):
                # 断言 y[:,j] 与 dct(x[:,j], type=self.type) 的元素近似相等，
                # 其精度由 self.dec 决定
                assert_array_almost_equal(y[:,j], dct(x[:,j], type=self.type),
                                          decimal=self.dec)
class _TestDCTIBase(_TestDCTBase):
    # 定义一个测试类，继承自 _TestDCTBase 类
    def test_definition_ortho(self):
        # 测试正交模式
        dt = np.result_type(np.float32, self.rdt)
        # 获取结果数据类型
        for xr in X:
            # 对于每个 xr 在 X 中
            x = np.array(xr, dtype=self.rdt)
            # 将 xr 转换为指定数据类型的 NumPy 数组
            y = dct(x, norm='ortho', type=1)
            # 计算正交类型 1 的离散余弦变换
            y2 = naive_dct1(x, norm='ortho')
            # 调用自定义的正交类型 1 算法进行变换
            assert_equal(y.dtype, dt)
            # 断言 y 的数据类型与 dt 相等
            assert_array_almost_equal(y / np.max(y), y2 / np.max(y), decimal=self.dec)
            # 断言 y 与 y2 的归一化结果几乎相等，精度为 self.dec

class _TestDCTIIBase(_TestDCTBase):
    # 定义一个测试类，继承自 _TestDCTBase 类
    def test_definition_matlab(self):
        # 测试与 MATLAB 的对应性（正交模式）
        dt = np.result_type(np.float32, self.rdt)
        # 获取结果数据类型
        for xr, yr in zip(X, Y):
            # 对于每个 xr, yr 在 X, Y 中
            x = np.array(xr, dtype=dt)
            # 将 xr 转换为指定数据类型的 NumPy 数组
            y = dct(x, norm="ortho", type=2)
            # 计算正交类型 2 的离散余弦变换
            assert_equal(y.dtype, dt)
            # 断言 y 的数据类型与 dt 相等
            assert_array_almost_equal(y, yr, decimal=self.dec)
            # 断言 y 与 yr 的结果几乎相等，精度为 self.dec

class _TestDCTIIIBase(_TestDCTBase):
    # 定义一个测试类，继承自 _TestDCTBase 类
    def test_definition_ortho(self):
        # 测试正交模式
        dt = np.result_type(np.float32, self.rdt)
        # 获取结果数据类型
        for xr in X:
            # 对于每个 xr 在 X 中
            x = np.array(xr, dtype=self.rdt)
            # 将 xr 转换为指定数据类型的 NumPy 数组
            y = dct(x, norm='ortho', type=2)
            # 计算正交类型 2 的离散余弦变换
            xi = dct(y, norm="ortho", type=3)
            # 对 y 进行逆变换，计算正交类型 3 的离散余弦逆变换
            assert_equal(xi.dtype, dt)
            # 断言 xi 的数据类型与 dt 相等
            assert_array_almost_equal(xi, x, decimal=self.dec)
            # 断言 xi 与 x 的结果几乎相等，精度为 self.dec

class _TestDCTIVBase(_TestDCTBase):
    # 定义一个测试类，继承自 _TestDCTBase 类
    def test_definition_ortho(self):
        # 测试正交模式
        dt = np.result_type(np.float32, self.rdt)
        # 获取结果数据类型
        for xr in X:
            # 对于每个 xr 在 X 中
            x = np.array(xr, dtype=self.rdt)
            # 将 xr 转换为指定数据类型的 NumPy 数组
            y = dct(x, norm='ortho', type=4)
            # 计算正交类型 4 的离散余弦变换
            y2 = naive_dct4(x, norm='ortho')
            # 调用自定义的正交类型 4 算法进行变换
            assert_equal(y.dtype, dt)
            # 断言 y 的数据类型与 dt 相等
            assert_array_almost_equal(y / np.max(y), y2 / np.max(y), decimal=self.dec)
            # 断言 y 与 y2 的归一化结果几乎相等，精度为 self.dec

class TestDCTIDouble(_TestDCTIBase):
    # 定义一个测试类，继承自 _TestDCTIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = np.float64
        # 设置数据类型为双精度浮点数
        self.dec = 10
        # 设置断言的精度为 10
        self.type = 1
        # 设置类型为 1

class TestDCTIFloat(_TestDCTIBase):
    # 定义一个测试类，继承自 _TestDCTIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = np.float32
        # 设置数据类型为单精度浮点数
        self.dec = 4
        # 设置断言的精度为 4
        self.type = 1
        # 设置类型为 1

class TestDCTIInt(_TestDCTIBase):
    # 定义一个测试类，继承自 _TestDCTIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = int
        # 设置数据类型为整数
        self.dec = 5
        # 设置断言的精度为 5
        self.type = 1
        # 设置类型为 1

class TestDCTIIDouble(_TestDCTIIBase):
    # 定义一个测试类，继承自 _TestDCTIIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = np.float64
        # 设置数据类型为双精度浮点数
        self.dec = 10
        # 设置断言的精度为 10
        self.type = 2
        # 设置类型为 2

class TestDCTIIFloat(_TestDCTIIBase):
    # 定义一个测试类，继承自 _TestDCTIIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = np.float32
        # 设置数据类型为单精度浮点数
        self.dec = 5
        # 设置断言的精度为 5
        self.type = 2
        # 设置类型为 2

class TestDCTIIInt(_TestDCTIIBase):
    # 定义一个测试类，继承自 _TestDCTIIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = int
        # 设置数据类型为整数
        self.dec = 5
        # 设置断言的精度为 5
        self.type = 2
        # 设置类型为 2

class TestDCTIIIDouble(_TestDCTIIIBase):
    # 定义一个测试类，继承自 _TestDCTIIIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = np.float64
        # 设置数据类型为双精度浮点数
        self.dec = 14
        # 设置断言的精度为 14
        self.type = 3
        # 设置类型为 3

class TestDCTIIIFloat(_TestDCTIIIBase):
    # 定义一个测试类，继承自 _TestDCTIIIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = np.float32
        # 设置数据类型为单精度浮点数
        self.dec = 5
        # 设置断言的精度为 5
        self.type = 3
        # 设置类型为 3

class TestDCTIIIInt(_TestDCTIIIBase):
    # 定义一个测试类，继承自 _TestDCTIIIBase 类
    def setup_method(self):
        # 初始化测试方法
        self.rdt = int
        # 设置数据类型为整数
        self.dec = 5
        # 设置断言的精度为 5
        self.type = 3
        # 设置类型为 3

class TestDCTIVDouble(_TestDCTIVBase):
    # 定义一个测试类，继承自 _TestDCTIVBase 类
    # 这个类没有定义特定的测试方法，继承了 _TestDCTIVBase 的
    # 设置测试方法的初始化操作
    def setup_method(self):
        # 设置 self.rdt 为 np.float64 类型
        self.rdt = np.float64
        # 设置 self.dec 为 12
        self.dec = 12
        # 设置 self.type 为 3
        self.type = 3
class TestDCTIVFloat(_TestDCTIVBase):
    # 定义测试类 TestDCTIVFloat，继承自 _TestDCTIVBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float32  # 设置 self.rdt 为 np.float32 类型
        self.dec = 5  # 设置 self.dec 为 5
        self.type = 3  # 设置 self.type 为 3


class TestDCTIVInt(_TestDCTIVBase):
    # 定义测试类 TestDCTIVInt，继承自 _TestDCTIVBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = int  # 设置 self.rdt 为 int 类型
        self.dec = 5  # 设置 self.dec 为 5
        self.type = 3  # 设置 self.type 为 3


class _TestIDCTBase:
    # 定义基类 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = None  # 初始化 self.rdt 为 None
        self.dec = 14  # 初始化 self.dec 为 14
        self.type = None  # 初始化 self.type 为 None

    def test_definition(self):
        # 定义测试方法 test_definition
        for i in FFTWDATA_SIZES:
            # 迭代 FFTWDATA_SIZES 中的每个元素，作为变量 i
            xr, yr, dt = fftw_dct_ref(self.type, i, self.rdt)
            # 调用 fftw_dct_ref 函数，并将返回的结果分别赋给 xr, yr, dt 变量
            x = idct(yr, type=self.type)
            # 调用 idct 函数，传入 yr 和 self.type 作为参数，并将结果赋给 x
            if self.type == 1:
                # 如果 self.type 等于 1
                x /= 2 * (i-1)
                # 对 x 进行除以 2 * (i-1) 的操作
            else:
                # 否则
                x /= 2 * i
                # 对 x 进行除以 2 * i 的操作
            assert_equal(x.dtype, dt)
            # 使用 assert_equal 断言 x 的数据类型等于 dt
            # XXX: we divide by np.max(y) because the tests fail otherwise. We
            # should really use something like assert_array_almost_equal. The
            # difference is due to fftw using a better algorithm w.r.t error
            # propagation compared to the ones from fftpack.
            assert_array_almost_equal(x / np.max(x), xr / np.max(x), decimal=self.dec,
                    err_msg="Size %d failed" % i)
            # 使用 assert_array_almost_equal 断言 x / np.max(x) 与 xr / np.max(x) 几乎相等，
            # 设置 decimal 参数为 self.dec，设置错误消息为 "Size %d failed" % i


class TestIDCTIDouble(_TestIDCTBase):
    # 定义测试类 TestIDCTIDouble，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float64  # 设置 self.rdt 为 np.float64 类型
        self.dec = 10  # 设置 self.dec 为 10
        self.type = 1  # 设置 self.type 为 1


class TestIDCTIFloat(_TestIDCTBase):
    # 定义测试类 TestIDCTIFloat，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float32  # 设置 self.rdt 为 np.float32 类型
        self.dec = 4  # 设置 self.dec 为 4
        self.type = 1  # 设置 self.type 为 1


class TestIDCTIInt(_TestIDCTBase):
    # 定义测试类 TestIDCTIInt，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = int  # 设置 self.rdt 为 int 类型
        self.dec = 4  # 设置 self.dec 为 4
        self.type = 1  # 设置 self.type 为 1


class TestIDCTIIDouble(_TestIDCTBase):
    # 定义测试类 TestIDCTIIDouble，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float64  # 设置 self.rdt 为 np.float64 类型
        self.dec = 10  # 设置 self.dec 为 10
        self.type = 2  # 设置 self.type 为 2


class TestIDCTIIFloat(_TestIDCTBase):
    # 定义测试类 TestIDCTIIFloat，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float32  # 设置 self.rdt 为 np.float32 类型
        self.dec = 5  # 设置 self.dec 为 5
        self.type = 2  # 设置 self.type 为 2


class TestIDCTIIInt(_TestIDCTBase):
    # 定义测试类 TestIDCTIIInt，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = int  # 设置 self.rdt 为 int 类型
        self.dec = 5  # 设置 self.dec 为 5
        self.type = 2  # 设置 self.type 为 2


class TestIDCTIIIDouble(_TestIDCTBase):
    # 定义测试类 TestIDCTIIIDouble，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float64  # 设置 self.rdt 为 np.float64 类型
        self.dec = 14  # 设置 self.dec 为 14
        self.type = 3  # 设置 self.type 为 3


class TestIDCTIIIFloat(_TestIDCTBase):
    # 定义测试类 TestIDCTIIIFloat，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float32  # 设置 self.rdt 为 np.float32 类型
        self.dec = 5  # 设置 self.dec 为 5
        self.type = 3  # 设置 self.type 为 3


class TestIDCTIIIInt(_TestIDCTBase):
    # 定义测试类 TestIDCTIIIInt，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = int  # 设置 self.rdt 为 int 类型
        self.dec = 5  # 设置 self.dec 为 5
        self.type = 3  # 设置 self.type 为 3


class TestIDCTIVDouble(_TestIDCTBase):
    # 定义测试类 TestIDCTIVDouble，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float64  # 设置 self.rdt 为 np.float64 类型
        self.dec = 12  # 设置 self.dec 为 12
        self.type = 4  # 设置 self.type 为 4


class TestIDCTIVFloat(_TestIDCTBase):
    # 定义测试类 TestIDCTIVFloat，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = np.float32  # 设置 self.rdt 为 np.float32 类型
        self.dec = 5  # 设置 self.dec 为 5
        self.type = 4  # 设置 self.type 为 4


class TestIDCTIVInt(_TestIDCTBase):
    # 定义测试类 TestIDCTIVInt，继承自 _TestIDCTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = int  # 设置 self.rdt 为 int 类型
        self.dec = 5  # 设置 self.dec 为 5
        self.type = 4  # 设置 self.type 为 4


class _TestDSTBase:
    # 定义基类 _TestDSTBase
    def setup_method(self):
        # 设置测试方法的初始化操作
        self.rdt = None  # 初始化 self.rdt 为 None，表示数据类型未定义
        self.dec = None  # 初始化 self.dec 为 None，表示精度未定义
        self.type = None  # 初始化 self.type 为 None，表示 DST 类型未定义
    # 在测试定义方法中进行循环，遍历 FFTWDATA_SIZES 中的每个大小
    def test_definition(self):
        # 对于每个大小 i，调用 fftw_dst_ref 方法获取 xr, yr, dt 三个返回值
        for i in FFTWDATA_SIZES:
            xr, yr, dt = fftw_dst_ref(self.type, i, self.rdt)
            # 调用 dst 方法计算 y，使用 self.type 作为参数
            y = dst(xr, type=self.type)
            # 断言 y 的数据类型与预期的 dt 相等
            assert_equal(y.dtype, dt)
            # XXX: 我们通过 np.max(y) 进行除法是因为如果不这样做，测试会失败。
            # 我们应该真正使用类似 assert_array_approx_equal 的方法。
            # 差异是由于 fftw 使用更好的算法来处理误差传播，相对于 fftpack 中的算法。
            assert_array_almost_equal(y / np.max(y), yr / np.max(y), decimal=self.dec,
                    err_msg="Size %d failed" % i)
class _TestDSTIBase(_TestDSTBase):
    def test_definition_ortho(self):
        # Test orthornomal mode.
        # 计算结果数据类型，使用 np.float32 和 self.rdt 中更宽松的类型
        dt = np.result_type(np.float32, self.rdt)
        # 对 X 中的每个元素 xr 进行循环
        for xr in X:
            # 将 xr 转换为指定数据类型 self.rdt 的 NumPy 数组 x
            x = np.array(xr, dtype=self.rdt)
            # 计算 x 的离散正弦变换（DST），使用正交模式（'ortho'），类型为 1
            y = dst(x, norm='ortho', type=1)
            # 使用自定义方法 naive_dst1 计算 x 的类型 1 的离散正弦变换，使用正交模式（'ortho'）
            y2 = naive_dst1(x, norm='ortho')
            # 断言 y 的数据类型与 dt 相等
            assert_equal(y.dtype, dt)
            # 断言 y 和 y2 的数值近似相等，精度为 self.dec
            assert_array_almost_equal(y / np.max(y), y2 / np.max(y), decimal=self.dec)

class _TestDSTIVBase(_TestDSTBase):
    def test_definition_ortho(self):
        # Test orthornomal mode.
        # 计算结果数据类型，使用 np.float32 和 self.rdt 中更宽松的类型
        dt = np.result_type(np.float32, self.rdt)
        # 对 X 中的每个元素 xr 进行循环
        for xr in X:
            # 将 xr 转换为指定数据类型 self.rdt 的 NumPy 数组 x
            x = np.array(xr, dtype=self.rdt)
            # 计算 x 的离散正弦变换（DST），使用正交模式（'ortho'），类型为 4
            y = dst(x, norm='ortho', type=4)
            # 使用自定义方法 naive_dst4 计算 x 的类型 4 的离散正弦变换，使用正交模式（'ortho'）
            y2 = naive_dst4(x, norm='ortho')
            # 断言 y 的数据类型与 dt 相等
            assert_equal(y.dtype, dt)
            # 断言 y 和 y2 的数值近似相等，精度为 self.dec
            assert_array_almost_equal(y, y2, decimal=self.dec)

class TestDSTIDouble(_TestDSTIBase):
    def setup_method(self):
        # 设置测试环境，使用 np.float64 作为数据类型
        self.rdt = np.float64
        # 设置数值比较的小数点精度为 12
        self.dec = 12
        # 设置离散正弦变换的类型为 1
        self.type = 1

class TestDSTIFloat(_TestDSTIBase):
    def setup_method(self):
        # 设置测试环境，使用 np.float32 作为数据类型
        self.rdt = np.float32
        # 设置数值比较的小数点精度为 4
        self.dec = 4
        # 设置离散正弦变换的类型为 1
        self.type = 1

class TestDSTIInt(_TestDSTIBase):
    def setup_method(self):
        # 设置测试环境，使用 int 作为数据类型
        self.rdt = int
        # 设置数值比较的小数点精度为 5
        self.dec = 5
        # 设置离散正弦变换的类型为 1
        self.type = 1

class TestDSTIIDouble(_TestDSTBase):
    def setup_method(self):
        # 设置测试环境，使用 np.float64 作为数据类型
        self.rdt = np.float64
        # 设置数值比较的小数点精度为 14
        self.dec = 14
        # 设置离散正弦变换的类型为 2
        self.type = 2

class TestDSTIIFloat(_TestDSTBase):
    def setup_method(self):
        # 设置测试环境，使用 np.float32 作为数据类型
        self.rdt = np.float32
        # 设置数值比较的小数点精度为 6
        self.dec = 6
        # 设置离散正弦变换的类型为 2
        self.type = 2

class TestDSTIIInt(_TestDSTBase):
    def setup_method(self):
        # 设置测试环境，使用 int 作为数据类型
        self.rdt = int
        # 设置数值比较的小数点精度为 6
        self.dec = 6
        # 设置离散正弦变换的类型为 2
        self.type = 2

class TestDSTIIIDouble(_TestDSTBase):
    def setup_method(self):
        # 设置测试环境，使用 np.float64 作为数据类型
        self.rdt = np.float64
        # 设置数值比较的小数点精度为 14
        self.dec = 14
        # 设置离散正弦变换的类型为 3
        self.type = 3

class TestDSTIIIFloat(_TestDSTBase):
    def setup_method(self):
        # 设置测试环境，使用 np.float32 作为数据类型
        self.rdt = np.float32
        # 设置数值比较的小数点精度为 7
        self.dec = 7
        # 设置离散正弦变换的类型为 3
        self.type = 3

class TestDSTIIIInt(_TestDSTBase):
    def setup_method(self):
        # 设置测试环境，使用 int 作为数据类型
        self.rdt = int
        # 设置数值比较的小数点精度为 7
        self.dec = 7
        # 设置离散正弦变换的类型为 3
        self.type = 3

class TestDSTIVDouble(_TestDSTIVBase):
    def setup_method(self):
        # 设置测试环境，使用 np.float64 作为数据类型
        self.rdt = np.float64
        # 设置数值比较的小数点精度为 12
        self.dec = 12
        # 设置离散正弦变换的类型为 4
        self.type = 4

class TestDSTIVFloat(_TestDSTIVBase):
    def setup_method(self):
        # 设置测试环境，使用 np.float32 作为数据类型
        self.rdt = np.float32
        # 设置数值比较的小数点精度为 4
        self.dec = 4
        # 设置离散正弦变换的类型为 4
        self.type = 4

class TestDSTIVInt(_TestDSTIVBase):
    def setup_method(self):
        # 设置测试环境，使用 int 作为数据类型
        self.rdt = int
        # 设置数值比较的小数点精度为 5
        self.dec = 5
        # 设置离散正弦变换的类型为 4
        self.type = 4

class _TestIDSTBase:
    def setup_method(self):
        # 设置初始化方法，分别为类型、数据类型和小数点精度设置默认值
        self.rdt = None
        self.dec = None
        self.type = None
    # 对每个 FFTWDATA_SIZES 中的大小进行测试
    def test_definition(self):
        # 遍历 FFTWDATA_SIZES 中的大小
        for i in FFTWDATA_SIZES:
            # 使用 fftw_dst_ref 函数获取参考结果 xr, yr, dt
            xr, yr, dt = fftw_dst_ref(self.type, i, self.rdt)
            # 使用 idst 函数计算逆离散正弦变换得到 x
            x = idst(yr, type=self.type)
            # 如果 self.type 为 1，则将 x 除以 2 * (i+1)
            if self.type == 1:
                x /= 2 * (i+1)
            else:
                # 否则将 x 除以 2 * i
                x /= 2 * i
            # 断言 x 的数据类型与 dt 相同
            assert_equal(x.dtype, dt)
            # XXX: 由于测试失败，我们通过除以 np.max(x) 来规范化 x。实际上应该使用类似 assert_array_approx_equal 的方法。
            # 这种差异是由于 fftw 使用更好的算法来处理误差传播，相对于 fftpack 中的算法。
            assert_array_almost_equal(x / np.max(x), xr / np.max(x), decimal=self.dec,
                    err_msg="Size %d failed" % i)
class TestID
    # 定义一个测试方法，用于测试 idct 函数
    def test_idct(self):
        # 遍历指定的实数数据类型列表
        for dtype in self.real_dtypes:
            # 调用 _check_1d 方法，检查 idct 函数在指定数据类型、形状 (16,) 和轴向 -1 下的行为
            self._check_1d(idct, dtype, (16,), -1)
            # 调用 _check_1d 方法，检查 idct 函数在指定数据类型、形状 (16, 2) 和轴向 0 下的行为
            self._check_1d(idct, dtype, (16, 2), 0)
            # 调用 _check_1d 方法，检查 idct 函数在指定数据类型、形状 (2, 16) 和轴向 1 下的行为
            self._check_1d(idct, dtype, (2, 16), 1)
    
    # 定义一个测试方法，用于测试 dst 函数
    def test_dst(self):
        # 遍历指定的实数数据类型列表
        for dtype in self.real_dtypes:
            # 调用 _check_1d 方法，检查 dst 函数在指定数据类型、形状 (16,) 和轴向 -1 下的行为
            self._check_1d(dst, dtype, (16,), -1)
            # 调用 _check_1d 方法，检查 dst 函数在指定数据类型、形状 (16, 2) 和轴向 0 下的行为
            self._check_1d(dst, dtype, (16, 2), 0)
            # 调用 _check_1d 方法，检查 dst 函数在指定数据类型、形状 (2, 16) 和轴向 1 下的行为
            self._check_1d(dst, dtype, (2, 16), 1)
    
    # 定义一个测试方法，用于测试 idst 函数
    def test_idst(self):
        # 遍历指定的实数数据类型列表
        for dtype in self.real_dtypes:
            # 调用 _check_1d 方法，检查 idst 函数在指定数据类型、形状 (16,) 和轴向 -1 下的行为
            self._check_1d(idst, dtype, (16,), -1)
            # 调用 _check_1d 方法，检查 idst 函数在指定数据类型、形状 (16, 2) 和轴向 0 下的行为
            self._check_1d(idst, dtype, (16, 2), 0)
            # 调用 _check_1d 方法，检查 idst 函数在指定数据类型、形状 (2, 16) 和轴向 1 下的行为
            self._check_1d(idst, dtype, (2, 16), 1)
# 定义一个测试类 Test_DCTN_IDCTN，用于测试离散余弦变换（DCT）和逆变换（IDCT）的功能
class Test_DCTN_IDCTN:
    # 设置变量 dec，表示精度为14
    dec = 14
    # 定义 dct_type 列表，包含四种可能的变换类型
    dct_type = [1, 2, 3, 4]
    # 定义 norms 列表，包含两种正交归一化类型
    norms = [None, 'ortho']
    # 创建一个随机状态对象 rstate，种子为1234
    rstate = np.random.RandomState(1234)
    # 设置 shape 变量为元组 (32, 16)，表示数据的形状为 32 行 16 列
    shape = (32, 16)
    # 使用 rstate 生成符合正态分布的随机数据，形状由 shape 变量指定
    data = rstate.randn(*shape)

    # 使用 pytest 的 parametrize 标记为测试函数 test_axes_round_trip 添加参数化测试
    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    # 参数化 axes 参数，测试多种不同的维度参数组合
    @pytest.mark.parametrize('axes', [None,
                                      1, (1,), [1],
                                      0, (0,), [0],
                                      (0, 1), [0, 1],
                                      (-2, -1), [-2, -1]])
    # 参数化 dct_type 参数，测试多种不同的变换类型
    @pytest.mark.parametrize('dct_type', dct_type)
    # 参数化 norm 参数，测试正交归一化
    @pytest.mark.parametrize('norm', ['ortho'])
    # 定义测试函数 test_axes_round_trip，用于测试变换与逆变换的一轮操作
    def test_axes_round_trip(self, fforward, finverse, axes, dct_type, norm):
        # 对 self.data 进行正向变换
        tmp = fforward(self.data, type=dct_type, axes=axes, norm=norm)
        # 对正向变换的结果 tmp 进行逆向变换
        tmp = finverse(tmp, type=dct_type, axes=axes, norm=norm)
        # 断言逆向变换后的结果与原始数据 self.data 相近
        assert_array_almost_equal(self.data, tmp, decimal=12)

    # 使用 pytest 的 parametrize 标记为测试函数 test_dctn_vs_2d_reference 添加参数化测试
    @pytest.mark.parametrize('fforward,fforward_ref', [(dctn, dct_2d_ref),
                                                       (dstn, dst_2d_ref)])
    # 参数化 dct_type 参数，测试多种不同的变换类型
    @pytest.mark.parametrize('dct_type', dct_type)
    # 参数化 norm 参数，测试 norms 列表中的两种正交归一化类型
    @pytest.mark.parametrize('norm', norms)
    # 定义测试函数 test_dctn_vs_2d_reference，测试实现的变换与参考实现的二维变换结果的一致性
    def test_dctn_vs_2d_reference(self, fforward, fforward_ref,
                                  dct_type, norm):
        # 对 self.data 进行测试变换
        y1 = fforward(self.data, type=dct_type, axes=None, norm=norm)
        # 调用参考实现的二维变换函数计算结果 y2
        y2 = fforward_ref(self.data, type=dct_type, norm=norm)
        # 断言两种变换结果的近似程度
        assert_array_almost_equal(y1, y2, decimal=11)

    # 使用 pytest 的 parametrize 标记为测试函数 test_idctn_vs_2d_reference 添加参数化测试
    @pytest.mark.parametrize('finverse,finverse_ref', [(idctn, idct_2d_ref),
                                                       (idstn, idst_2d_ref)])
    # 参数化 dct_type 参数，测试多种不同的变换类型
    @pytest.mark.parametrize('dct_type', dct_type)
    # 参数化 norm 参数，测试 norms 列表中的两种正交归一化类型和 None
    @pytest.mark.parametrize('norm', [None, 'ortho'])
    # 定义测试函数 test_idctn_vs_2d_reference，测试实现的逆变换与参考实现的二维逆变换结果的一致性
    def test_idctn_vs_2d_reference(self, finverse, finverse_ref,
                                   dct_type, norm):
        # 对 self.data 进行测试变换，得到变换后的数据 fdata
        fdata = dctn(self.data, type=dct_type, norm=norm)
        # 对变换后的数据 fdata 进行逆变换
        y1 = finverse(fdata, type=dct_type, norm=norm)
        # 调用参考实现的二维逆变换函数计算结果 y2
        y2 = finverse_ref(fdata, type=dct_type, norm=norm)
        # 断言两种逆变换结果的近似程度
        assert_array_almost_equal(y1, y2, decimal=11)

    # 使用 pytest 的 parametrize 标记为测试函数 test_axes_round_trip 添加参数化测试
    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    # 测试函数，用于验证在给定不匹配的 axes 和 shape 参数时是否引发 ValueError 异常
    def test_axes_and_shape(self, fforward, finverse):
        # 使用 assert_raises 上下文管理器验证调用 fforward 函数时是否会抛出 ValueError 异常，
        # 并且异常消息应包含指定的错误信息
        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            # 调用 fforward 函数，传入 self.data 作为数据参数，shape 参数为 self.data 的第一个维度大小，
            # axes 参数为元组 (0, 1)
            fforward(self.data, shape=self.data.shape[0], axes=(0, 1))

        # 使用 assert_raises 上下文管理器验证调用 fforward 函数时是否会抛出 ValueError 异常，
        # 并且异常消息应包含指定的错误信息
        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            # 调用 fforward 函数，传入 self.data 作为数据参数，shape 参数为 self.data 的第一个维度大小，
            # axes 参数为 None
            fforward(self.data, shape=self.data.shape[0], axes=None)

        # 使用 assert_raises 上下文管理器验证调用 fforward 函数时是否会抛出 ValueError 异常，
        # 并且异常消息应包含指定的错误信息
        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            # 调用 fforward 函数，传入 self.data 作为数据参数，shape 参数为 self.data 的维度元组，
            # axes 参数为整数 0
            fforward(self.data, shape=self.data.shape, axes=0)

    # 使用 pytest 的参数化装饰器，测试函数的 shape 是否符合预期
    @pytest.mark.parametrize('fforward', [dctn, dstn])
    def test_shape(self, fforward):
        # 调用 fforward 函数，传入 self.data 作为数据参数，shape 参数为 (128, 128)，axes 参数为 None
        tmp = fforward(self.data, shape=(128, 128), axes=None)
        # 断言 tmp 的形状应为 (128, 128)
        assert_equal(tmp.shape, (128, 128))

    # 使用 pytest 的参数化装饰器，测试函数在不同的 axes 参数下 shape 是否为 None 的情况
    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    # 使用 pytest 的参数化装饰器，参数 axes 接受不同的值进行测试
    @pytest.mark.parametrize('axes', [1, (1,), [1],
                                      0, (0,), [0]])
    def test_shape_is_none_with_axes(self, fforward, finverse, axes):
        # 调用 fforward 函数，传入 self.data 作为数据参数，shape 参数为 None，
        # axes 参数为传入的 axes 值，norm 参数为 'ortho'
        tmp = fforward(self.data, shape=None, axes=axes, norm='ortho')
        # 调用 finverse 函数，传入 tmp 作为数据参数，shape 参数为 None，
        # axes 参数为传入的 axes 值，norm 参数为 'ortho'
        tmp = finverse(tmp, shape=None, axes=axes, norm='ortho')
        # 断言 tmp 与 self.data 几乎相等，精度为 self.dec
        assert_array_almost_equal(self.data, tmp, decimal=self.dec)
```