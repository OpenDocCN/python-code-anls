# `D:\src\scipysrc\scipy\scipy\fft\_pocketfft\tests\test_real_transforms.py`

```
from os.path import join, dirname  # 导入路径拼接和目录名称函数
from typing import Callable, Union  # 导入类型提示相关的模块

import numpy as np  # 导入 NumPy 库
from numpy.testing import (  # 导入 NumPy 测试模块中的断言函数
    assert_array_almost_equal, assert_equal, assert_allclose)
import pytest  # 导入 Pytest 测试框架
from pytest import raises as assert_raises  # 导入 Pytest 的 raises 函数用作别名

from scipy.fft._pocketfft.realtransforms import (  # 导入 SciPy 中的 FFT 实现模块中的变换函数
    dct, idct, dst, idst, dctn, idctn, dstn, idstn)

fftpack_test_dir = join(dirname(__file__), '..', '..', '..', 'fftpack', 'tests')  # 设置 FFT 测试目录路径

MDATA_COUNT = 8  # 定义 Matlab 数据集个数
FFTWDATA_COUNT = 14  # 定义 FFTW 数据集个数

def is_longdouble_binary_compatible():  # 检查长双精度是否与二进制兼容的函数
    try:
        one = np.frombuffer(  # 从二进制数据创建 NumPy 数组，用于比较
            b'\x00\x00\x00\x00\x00\x00\x00\x80\xff\x3f\x00\x00\x00\x00\x00\x00',
            dtype='<f16')
        return one == np.longdouble(1.)  # 检查是否等于长双精度的 1.0
    except TypeError:
        return False  # 如果出现类型错误，则返回 False


@pytest.fixture(scope="module")
def reference_data():
    # 加载 Matlab 参考数据
    MDATA = np.load(join(fftpack_test_dir, 'test.npz'))
    X = [MDATA['x%d' % i] for i in range(MDATA_COUNT)]  # 获取 MDATA 中 x 数据集
    Y = [MDATA['y%d' % i] for i in range(MDATA_COUNT)]  # 获取 MDATA 中 y 数据集

    # 加载 FFTW 参考数据：数据组织如下
    #    * SIZES 是一个包含所有可用大小的数组
    #    * 对于每种类型（1、2、3、4）和每个大小，数组 dct_type_size 包含应用于输入 np.linspace(0, size-1, size) 的 DCT 的输出
    FFTWDATA_DOUBLE = np.load(join(fftpack_test_dir, 'fftw_double_ref.npz'))
    FFTWDATA_SINGLE = np.load(join(fftpack_test_dir, 'fftw_single_ref.npz'))
    FFTWDATA_SIZES = FFTWDATA_DOUBLE['sizes']  # 获取 FFTW 双精度数据的大小信息
    assert len(FFTWDATA_SIZES) == FFTWDATA_COUNT  # 断言 FFTW 数据大小与预期的数量一致

    if is_longdouble_binary_compatible():
        FFTWDATA_LONGDOUBLE = np.load(
            join(fftpack_test_dir, 'fftw_longdouble_ref.npz'))
    else:
        FFTWDATA_LONGDOUBLE = {k: v.astype(np.longdouble)
                               for k,v in FFTWDATA_DOUBLE.items()}  # 将 FFTW 双精度数据转换为长双精度

    ref = {
        'FFTWDATA_LONGDOUBLE': FFTWDATA_LONGDOUBLE,
        'FFTWDATA_DOUBLE': FFTWDATA_DOUBLE,
        'FFTWDATA_SINGLE': FFTWDATA_SINGLE,
        'FFTWDATA_SIZES': FFTWDATA_SIZES,
        'X': X,
        'Y': Y
    }

    yield ref  # 返回参考数据字典作为生成器的值

    if is_longdouble_binary_compatible():
        FFTWDATA_LONGDOUBLE.close()  # 关闭 FFTW 长双精度数据文件
    FFTWDATA_SINGLE.close()  # 关闭 FFTW 单精度数据文件
    FFTWDATA_DOUBLE.close()  # 关闭 FFTW 双精度数据文件
    MDATA.close()  # 关闭 Matlab 数据文件


@pytest.fixture(params=range(FFTWDATA_COUNT))
def fftwdata_size(request, reference_data):
    return reference_data['FFTWDATA_SIZES'][request.param]  # 返回 FFTW 数据大小的参数化 fixture


@pytest.fixture(params=range(MDATA_COUNT))
def mdata_x(request, reference_data):
    return reference_data['X'][request.param]  # 返回 Matlab 数据集 X 的参数化 fixture


@pytest.fixture(params=range(MDATA_COUNT))
def mdata_xy(request, reference_data):
    y = reference_data['Y'][request.param]  # 获取 Matlab 数据集 Y 的参数化 fixture
    x = reference_data['X'][request.param]  # 获取 Matlab 数据集 X 的参数化 fixture
    return x, y  # 返回 X 和 Y 的元组作为 fixture


def fftw_dct_ref(type, size, dt, reference_data):
    x = np.linspace(0, size-1, size).astype(dt)  # 创建一个类型为 dt 的长度为 size 的线性空间数组 x
    dt = np.result_type(np.float32, dt)  # 确定浮点数类型
    if dt == np.float64:
        data = reference_data['FFTWDATA_DOUBLE']  # 如果是双精度浮点数，使用 FFTW 双精度数据
    elif dt == np.float32:
        data = reference_data['FFTWDATA_SINGLE']  # 如果是单精度浮点数，使用 FFTW 单精度数据
    # 如果数据类型是 np.longdouble，从参考数据中获取相应的数据
    elif dt == np.longdouble:
        data = reference_data['FFTWDATA_LONGDOUBLE']
    # 如果以上条件都不满足，引发 ValueError 异常
    else:
        raise ValueError()
    # 根据给定的 type 和 size 构造字符串 'dct_%d_%d'，并从 data 中取出对应的值
    y = (data['dct_%d_%d' % (type, size)]).astype(dt)
    # 返回 x, y 以及数据类型 dt
    return x, y, dt
# 定义一个函数 fftw_dst_ref，用于生成傅立叶变换的离散正弦变换（DST）的参考数据
def fftw_dst_ref(type, size, dt, reference_data):
    # 创建一个从0到size-1的等间距数组，并转换其数据类型为指定的 dt
    x = np.linspace(0, size-1, size).astype(dt)
    # 确定 dt 的类型，选择相应的参考数据
    dt = np.result_type(np.float32, dt)
    if dt == np.float64:
        data = reference_data['FFTWDATA_DOUBLE']
    elif dt == np.float32:
        data = reference_data['FFTWDATA_SINGLE']
    elif dt == np.longdouble:
        data = reference_data['FFTWDATA_LONGDOUBLE']
    else:
        raise ValueError()
    # 从参考数据中选择适当的 DST 数据，根据 type 和 size 组合成键进行选择
    y = (data['dst_%d_%d' % (type, size)]).astype(dt)
    # 返回生成的 x、y 数组以及数据类型 dt
    return x, y, dt


# 定义一个函数 ref_2d，用于从一维变换计算二维参考数据
def ref_2d(func, x, **kwargs):
    """Calculate 2-D reference data from a 1d transform"""
    # 深复制输入的数组 x
    x = np.array(x, copy=True)
    # 对数组 x 的每一行应用 func 函数，传入额外的关键字参数 kwargs
    for row in range(x.shape[0]):
        x[row, :] = func(x[row, :], **kwargs)
    # 对数组 x 的每一列应用 func 函数，传入额外的关键字参数 kwargs
    for col in range(x.shape[1]):
        x[:, col] = func(x[:, col], **kwargs)
    # 返回处理后的二维数组 x
    return x


# 定义一个函数 naive_dct1，计算离散余弦变换（DCT-I）的文本书籍定义版本
def naive_dct1(x, norm=None):
    """Calculate textbook definition version of DCT-I."""
    # 深复制输入的数组 x
    x = np.array(x, copy=True)
    # 获取数组 x 的长度 N
    N = len(x)
    M = N-1
    # 创建一个长度为 N 的全零数组 y
    y = np.zeros(N)
    m0, m = 1, 2
    # 根据正交参数 norm 调整 m0 和 m 的值
    if norm == 'ortho':
        m0 = np.sqrt(1.0/M)
        m = np.sqrt(2.0/M)
    # 使用双重循环计算 DCT-I
    for k in range(N):
        for n in range(1, N-1):
            y[k] += m*x[n]*np.cos(np.pi*n*k/M)
        y[k] += m0 * x[0]
        y[k] += m0 * x[N-1] * (1 if k % 2 == 0 else -1)
    # 根据正交参数 norm 调整 y[0] 和 y[N-1] 的值
    if norm == 'ortho':
        y[0] *= 1/np.sqrt(2)
        y[N-1] *= 1/np.sqrt(2)
    # 返回计算得到的 DCT-I 结果数组 y
    return y


# 定义一个函数 naive_dst1，计算离散正弦变换（DST-I）的文本书籍定义版本
def naive_dst1(x, norm=None):
    """Calculate textbook definition version of DST-I."""
    # 深复制输入的数组 x
    x = np.array(x, copy=True)
    # 获取数组 x 的长度 N
    N = len(x)
    M = N+1
    # 创建一个长度为 N 的全零数组 y
    y = np.zeros(N)
    # 使用双重循环计算 DST-I
    for k in range(N):
        for n in range(N):
            y[k] += 2*x[n]*np.sin(np.pi*(n+1.0)*(k+1.0)/M)
    # 根据正交参数 norm 调整 y 的值
    if norm == 'ortho':
        y *= np.sqrt(0.5/M)
    # 返回计算得到的 DST-I 结果数组 y
    return y


# 定义一个函数 naive_dct4，计算离散余弦变换（DCT-IV）的文本书籍定义版本
def naive_dct4(x, norm=None):
    """Calculate textbook definition version of DCT-IV."""
    # 深复制输入的数组 x
    x = np.array(x, copy=True)
    # 获取数组 x 的长度 N
    N = len(x)
    # 创建一个长度为 N 的全零数组 y
    y = np.zeros(N)
    # 使用双重循环计算 DCT-IV
    for k in range(N):
        for n in range(N):
            y[k] += x[n]*np.cos(np.pi*(n+0.5)*(k+0.5)/(N))
    # 根据正交参数 norm 调整 y 的值
    if norm == 'ortho':
        y *= np.sqrt(2.0/N)
    else:
        y *= 2
    # 返回计算得到的 DCT-IV 结果数组 y
    return y


# 定义一个函数 naive_dst4，计算离散正弦变换（DST-IV）的文本书籍定义版本
def naive_dst4(x, norm=None):
    """Calculate textbook definition version of DST-IV."""
    # 深复制输入的数组 x
    x = np.array(x, copy=True)
    # 获取数组 x 的长度 N
    N = len(x)
    # 创建一个长度为 N 的全零数组 y
    y = np.zeros(N)
    # 使用双重循环计算 DST-IV
    for k in range(N):
        for n in range(N):
            y[k] += x[n]*np.sin(np.pi*(n+0.5)*(k+0.5)/(N))
    # 根据正交参数 norm 调整 y 的值
    if norm == 'ortho':
        y *= np.sqrt(2.0/N)
    else:
        y *= 2
    # 返回计算得到的 DST-IV 结果数组 y
    return y


# 使用 pytest 的参数化测试装饰器进行测试，参数为复数类型和变换类型
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128, np.clongdouble])
@pytest.mark.parametrize('transform', [dct, dst, idct, idst])
def test_complex(transform, dtype):
    # 对复数类型进行变换测试，预期结果为近似相等
    y = transform(1j*np.arange(5, dtype=dtype))
    x = 1j*transform(np.arange(5))
    assert_array_almost_equal(x, y)


# 定义一个类型映射 DecMapType，将 (变换函数, 数据类型, 变换类型) 映射到十进制数值的字典
DecMapType = dict[
    tuple[Callable[..., np.ndarray], Union[type[np.floating], type[int]], int],
    int,
]

# 创建一个 dec_map 字典，将 DCT 和 DST 变换函数、数据类型和变换类型映射到十进制数值
dec_map: DecMapType = {
    # DCT
    (dct, np.float64, 1): 13,
    (dct, np.float32, 1): 6,

    (dct, np.float64, 2): 14,
    (dct, np.float32, 2): 5,

    # DST
    (dst, np.float64, 1): 13,
    (dst, np.float32, 1): 6,

    (dst, np.float64, 2): 14,
    (dst, np.float32, 2): 5,
}
    # 定义字典，键为元组，元组的第一个元素是函数名，第二个元素是 NumPy 数据类型，第三个元素是整数
    (dct, np.float64, 3): 14,
    (dct, np.float32, 3): 5,

    (dct, np.float64, 4): 13,
    (dct, np.float32, 4): 6,

    # IDCT (逆离散余弦变换)
    (idct, np.float64, 1): 14,
    (idct, np.float32, 1): 6,

    (idct, np.float64, 2): 14,
    (idct, np.float32, 2): 5,

    (idct, np.float64, 3): 14,
    (idct, np.float32, 3): 5,

    (idct, np.float64, 4): 14,
    (idct, np.float32, 4): 6,

    # DST (离散正弦变换)
    (dst, np.float64, 1): 13,
    (dst, np.float32, 1): 6,

    (dst, np.float64, 2): 14,
    (dst, np.float32, 2): 6,

    (dst, np.float64, 3): 14,
    (dst, np.float32, 3): 7,

    (dst, np.float64, 4): 13,
    (dst, np.float32, 4): 5,

    # IDST (逆离散正弦变换)
    (idst, np.float64, 1): 14,
    (idst, np.float32, 1): 6,

    (idst, np.float64, 2): 14,
    (idst, np.float32, 2): 6,

    (idst, np.float64, 3): 14,
    (idst, np.float32, 3): 6,

    (idst, np.float64, 4): 14,
    (idst, np.float32, 4): 6,
}

# 遍历 dec_map 的副本，复制出每一项进行迭代
for k,v in dec_map.copy().items():
    # 如果 k 的第二个元素是 np.float64 类型
    if k[1] == np.float64:
        # 将键为 (k[0], np.longdouble, k[2]) 的条目设置为 v
        dec_map[(k[0], np.longdouble, k[2])] = v
    # 如果 k 的第二个元素是 np.float32 类型
    elif k[1] == np.float32:
        # 将键为 (k[0], int, k[2]) 的条目设置为 v
        dec_map[(k[0], int, k[2])] = v


# 使用参数化装饰器为 TestDCT 类添加多组参数化测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
class TestDCT:
    # 定义测试函数 test_definition，接受 rdt, type, fftwdata_size, reference_data 参数
    def test_definition(self, rdt, type, fftwdata_size, reference_data):
        # 调用 fftw_dct_ref 函数获取 x, yr, dt
        x, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt, reference_data)
        # 调用 dct 函数计算 y
        y = dct(x, type=type)
        # 断言 y 的数据类型为 dt
        assert_equal(y.dtype, dt)
        # 获取对应于 (dct, rdt, type) 键的 dec 值
        dec = dec_map[(dct, rdt, type)]
        # 断言 y 与 yr 的近似程度
        assert_allclose(y, yr, rtol=0., atol=np.max(yr)*10**(-dec))

    # 使用参数化装饰器为 test_axis 方法添加多组参数化测试
    @pytest.mark.parametrize('size', [7, 8, 9, 16, 32, 64])
    def test_axis(self, rdt, type, size):
        # 设置 nt 为 2
        nt = 2
        # 获取对应于 (dct, rdt, type) 键的 dec 值
        dec = dec_map[(dct, rdt, type)]
        # 创建随机数组 x
        x = np.random.randn(nt, size)
        # 计算 y = dct(x, type=type)
        y = dct(x, type=type)
        # 对每个 j 进行断言，验证 y[j] 与 dct(x[j], type=type) 的近似程度
        for j in range(nt):
            assert_array_almost_equal(y[j], dct(x[j], type=type),
                                      decimal=dec)

        # 将 x 转置
        x = x.T
        # 计算 y = dct(x, axis=0, type=type)
        y = dct(x, axis=0, type=type)
        # 对每个 j 进行断言，验证 y[:,j] 与 dct(x[:,j], type=type) 的近似程度
        for j in range(nt):
            assert_array_almost_equal(y[:,j], dct(x[:,j], type=type),
                                      decimal=dec)


# 使用参数化装饰器为 test_dct1_definition_ortho 函数添加多组参数化测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dct1_definition_ortho(rdt, mdata_x):
    # 测试正交模式
    dec = dec_map[(dct, rdt, 1)]
    # 创建数组 x，数据类型为 rdt
    x = np.array(mdata_x, dtype=rdt)
    # 计算 y = dct(x, norm='ortho', type=1)
    dt = np.result_type(np.float32, rdt)
    y = dct(x, norm='ortho', type=1)
    y2 = naive_dct1(x, norm='ortho')
    # 断言 y 的数据类型为 dt
    assert_equal(y.dtype, dt)
    # 断言 y 与 y2 的近似程度
    assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))


# 使用参数化装饰器为 test_dct2_definition_matlab 函数添加多组参数化测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dct2_definition_matlab(mdata_xy, rdt):
    # 测试与 MATLAB 对应性（正交模式）
    dt = np.result_type(np.float32, rdt)
    # 创建数组 x，数据类型为 dt
    x = np.array(mdata_xy[0], dtype=dt)

    yr = mdata_xy[1]
    # 计算 y = dct(x, norm="ortho", type=2)
    y = dct(x, norm="ortho", type=2)
    dec = dec_map[(dct, rdt, 2)]
    # 断言 y 的数据类型为 dt
    assert_equal(y.dtype, dt)
    # 断言 y 与 yr 的近似程度
    assert_array_almost_equal(y, yr, decimal=dec)


# 使用参数化装饰器为 test_dct3_definition_ortho 函数添加多组参数化测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dct3_definition_ortho(mdata_x, rdt):
    # 测试正交模式
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    # 计算 y = dct(x, norm='ortho', type=2)
    y = dct(x, norm='ortho', type=2)
    # 计算 xi = dct(y, norm="ortho", type=3)
    xi = dct(y, norm="ortho", type=3)
    dec = dec_map[(dct, rdt, 3)]
    # 断言 xi 的数据类型为 dt
    assert_equal(xi.dtype, dt)
    # 断言 xi 与 x 的近似程度
    assert_array_almost_equal(xi, x, decimal=dec)


# 使用参数化装饰器为 test_dct4_definition_ortho 函数添加多组参数化测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dct4_definition_ortho(mdata_x, rdt):
    # 测试正交模式
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    # 计算 y = dct(x, norm='ortho', type=4)
    y = dct(x, norm='ortho', type=4)
    y2 = naive_dct4(x, norm='ortho')
    dec = dec_map[(dct, rdt, 4)]
    # 断言 y 的数据类型为 dt
    assert_equal(y.dtype, dt)
    # 断言 y 与 y2 的近似程度
    assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))
# 参数化测试，针对不同的数据类型和变量类型进行测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_idct_definition(fftwdata_size, rdt, type, reference_data):
    # 调用 fftw_dct_ref 函数获取参考数据
    xr, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt, reference_data)
    # 调用 idct 函数进行逆离散余弦变换
    x = idct(yr, type=type)
    # 获取相应函数的小数位数精度
    dec = dec_map[(idct, rdt, type)]
    # 断言结果的数据类型与预期一致
    assert_equal(x.dtype, dt)
    # 断言结果与参考数据的接近程度
    assert_allclose(x, xr, rtol=0., atol=np.max(xr)*10**(-dec))


# 参数化测试，针对不同的数据类型和变量类型进行测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_definition(fftwdata_size, rdt, type, reference_data):
    # 调用 fftw_dst_ref 函数获取参考数据
    xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt, reference_data)
    # 调用 dst 函数进行离散正弦变换
    y = dst(xr, type=type)
    # 获取相应函数的小数位数精度
    dec = dec_map[(dst, rdt, type)]
    # 断言结果的数据类型与预期一致
    assert_equal(y.dtype, dt)
    # 断言结果与参考数据的接近程度
    assert_allclose(y, yr, rtol=0., atol=np.max(yr)*10**(-dec))


# 参数化测试，针对不同的数据类型进行正交模式下的变换测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dst1_definition_ortho(rdt, mdata_x):
    # Test orthornomal mode.
    # 获取相应函数的小数位数精度
    dec = dec_map[(dst, rdt, 1)]
    # 根据给定的数据类型创建 NumPy 数组
    x = np.array(mdata_x, dtype=rdt)
    # 推断结果的数据类型
    dt = np.result_type(np.float32, rdt)
    # 调用 dst 函数进行离散正弦变换，采用正交模式
    y = dst(x, norm='ortho', type=1)
    # 调用 naive_dst1 函数进行简单的离散正弦变换，采用正交模式
    y2 = naive_dst1(x, norm='ortho')
    # 断言结果的数据类型与预期一致
    assert_equal(y.dtype, dt)
    # 断言结果与参考数据的接近程度
    assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))


# 参数化测试，针对不同的数据类型进行正交模式下的变换测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dst4_definition_ortho(rdt, mdata_x):
    # Test orthornomal mode.
    # 获取相应函数的小数位数精度
    dec = dec_map[(dst, rdt, 4)]
    # 根据给定的数据类型创建 NumPy 数组
    x = np.array(mdata_x, dtype=rdt)
    # 推断结果的数据类型
    dt = np.result_type(np.float32, rdt)
    # 调用 dst 函数进行离散正弦变换，采用正交模式
    y = dst(x, norm='ortho', type=4)
    # 调用 naive_dst4 函数进行简单的离散正弦变换，采用正交模式
    y2 = naive_dst4(x, norm='ortho')
    # 断言结果的数据类型与预期一致
    assert_equal(y.dtype, dt)
    # 断言结果与参考数据的接近程度
    assert_array_almost_equal(y, y2, decimal=dec)


# 参数化测试，针对不同的数据类型和变量类型进行测试
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_idst_definition(fftwdata_size, rdt, type, reference_data):
    # 调用 fftw_dst_ref 函数获取参考数据
    xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt, reference_data)
    # 调用 idst 函数进行逆离散正弦变换
    x = idst(yr, type=type)
    # 获取相应函数的小数位数精度
    dec = dec_map[(idst, rdt, type)]
    # 断言结果的数据类型与预期一致
    assert_equal(x.dtype, dt)
    # 断言结果与参考数据的接近程度
    assert_allclose(x, xr, rtol=0., atol=np.max(xr)*10**(-dec))


# 参数化测试，针对不同的例程和数据类型进行测试
@pytest.mark.parametrize('routine', [dct, dst, idct, idst])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize('shape, axis', [
    ((16,), -1), ((16, 2), 0), ((2, 16), 1)
])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
@pytest.mark.parametrize('overwrite_x', [True, False])
@pytest.mark.parametrize('norm', [None, 'ortho'])
def test_overwrite(routine, dtype, shape, axis, type, norm, overwrite_x):
    # Check input overwrite behavior
    np.random.seed(1234)
    # 根据数据类型创建随机数数组
    if np.issubdtype(dtype, np.complexfloating):
        x = np.random.randn(*shape) + 1j*np.random.randn(*shape)
    else:
        x = np.random.randn(*shape)
    x = x.astype(dtype)
    # 复制输入数组
    x2 = x.copy()
    # 调用指定的例程进行变换，并检查是否支持原地操作
    routine(x2, type, None, axis, norm, overwrite_x=overwrite_x)
    # 根据指定的例程和参数，生成一个描述性的签名字符串
    sig = "{}({}{!r}, {!r}, axis={!r}, overwrite_x={!r})".format(
        routine.__name__, x.dtype, x.shape, None, axis, overwrite_x)
    # 如果不允许在 x 上进行覆盖操作，则进行断言检查
    if not overwrite_x:
        # 检查 x2 和 x 是否相等，如果不相等则输出错误信息，指明发生了意外的覆盖操作
        assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)
class Test_DCTN_IDCTN:
    # 初始化测试类的属性
    dec = 14  # 设置一个整数属性 dec
    dct_type = [1, 2, 3, 4]  # 定义一个列表属性 dct_type，包含几种 DCT 类型
    norms = [None, 'backward', 'ortho', 'forward']  # 定义一个列表属性 norms，包含几种正交化选项
    rstate = np.random.RandomState(1234)  # 创建一个指定种子的随机状态对象 rstate
    shape = (32, 16)  # 定义一个元组属性 shape，表示数据的形状
    data = rstate.randn(*shape)  # 用 rstate 生成符合指定形状的随机数据，并赋值给属性 data

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    @pytest.mark.parametrize('axes', [None,
                                      1, (1,), [1],
                                      0, (0,), [0],
                                      (0, 1), [0, 1],
                                      (-2, -1), [-2, -1]])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', ['ortho'])
    def test_axes_round_trip(self, fforward, finverse, axes, dct_type, norm):
        # 测试 DCT 和 IDCT 转换的正确性
        tmp = fforward(self.data, type=dct_type, axes=axes, norm=norm)  # 执行正变换
        tmp = finverse(tmp, type=dct_type, axes=axes, norm=norm)  # 执行逆变换
        assert_array_almost_equal(self.data, tmp, decimal=12)  # 断言转换前后数据的准确性

    @pytest.mark.parametrize('funcn,func', [(dctn, dct), (dstn, dst)])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', norms)
    def test_dctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        # 测试 DCTN 和参考实现在二维情况下的一致性
        y1 = funcn(self.data, type=dct_type, axes=None, norm=norm)  # 执行 DCTN
        y2 = ref_2d(func, self.data, type=dct_type, norm=norm)  # 使用参考实现计算
        assert_array_almost_equal(y1, y2, decimal=11)  # 断言两者的结果几乎相等

    @pytest.mark.parametrize('funcn,func', [(idctn, idct), (idstn, idst)])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', norms)
    def test_idctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        # 测试 IDCTN 和参考实现在二维情况下的一致性
        fdata = dctn(self.data, type=dct_type, norm=norm)  # 执行 DCT 变换
        y1 = funcn(fdata, type=dct_type, norm=norm)  # 执行 IDCTN 变换
        y2 = ref_2d(func, fdata, type=dct_type, norm=norm)  # 使用参考实现计算
        assert_array_almost_equal(y1, y2, decimal=11)  # 断言两者的结果几乎相等

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    def test_axes_and_shape(self, fforward, finverse):
        # 测试在给定情况下，axes 和 shape 参数是否具有相同的长度
        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            fforward(self.data, s=self.data.shape[0], axes=(0, 1))  # 触发值错误异常

        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            fforward(self.data, s=self.data.shape, axes=0)  # 触发值错误异常

    @pytest.mark.parametrize('fforward', [dctn, dstn])
    def test_shape(self, fforward):
        # 测试变换后的数据形状
        tmp = fforward(self.data, s=(128, 128), axes=None)  # 执行变换
        assert_equal(tmp.shape, (128, 128))  # 断言结果的形状是否符合预期

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    @pytest.mark.parametrize('axes', [1, (1,), [1],
                                      0, (0,), [0]])
    def test_axes(self, fforward, finverse):
        # 测试变换的轴参数
        # 此处应该添加具体的测试代码，该部分注释中遗漏了这一点
    # 定义测试函数，验证给定的正向和反向变换函数在指定轴上的行为
    def test_shape_is_none_with_axes(self, fforward, finverse, axes):
        # 使用正向变换函数对数据进行变换，其中参数 s=None 表示无缩放，norm='ortho' 表示正交归一化
        tmp = fforward(self.data, s=None, axes=axes, norm='ortho')
        # 使用反向变换函数对变换后的数据进行逆变换
        tmp = finverse(tmp, s=None, axes=axes, norm='ortho')
        # 断言逆变换后的数据与原始数据几乎相等，使用给定的小数精度进行比较
        assert_array_almost_equal(self.data, tmp, decimal=self.dec)
# 使用 pytest 的 parametrize 装饰器，为以下函数生成多个参数化测试实例
@pytest.mark.parametrize('func', [dct, dctn, idct, idctn,
                                  dst, dstn, idst, idstn])
# 定义测试函数 test_swapped_byte_order，参数为 func，即测试的函数之一
def test_swapped_byte_order(func):
    # 创建随机数生成器对象 rng，种子为 1234
    rng = np.random.RandomState(1234)
    # 生成一个包含 10 个随机浮点数的数组 x
    x = rng.rand(10)
    # 创建 x 的副本，并将其数据类型转换为指定的字节顺序 'S'
    swapped_dt = x.dtype.newbyteorder('S')
    # 断言 func 对 x 转换为指定字节顺序的数组和对原始数组 x 的计算结果近似相等
    assert_allclose(func(x.astype(swapped_dt)), func(x))
```