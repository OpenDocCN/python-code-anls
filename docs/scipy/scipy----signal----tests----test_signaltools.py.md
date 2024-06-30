# `D:\src\scipysrc\scipy\scipy\signal\tests\test_signaltools.py`

```
# 导入 sys 模块，用于系统相关操作
import sys

# 导入 ThreadPoolExecutor 和 as_completed 方法，用于并发任务处理
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入 Decimal 类，支持高精度浮点运算
from decimal import Decimal

# 导入 product 函数，用于生成迭代器的笛卡尔积
from itertools import product

# 导入 gcd 函数，用于计算最大公约数
from math import gcd

# 导入 pytest 模块及其 raises 方法，用于测试断言
import pytest
from pytest import raises as assert_raises

# 导入 numpy.testing 模块的各种断言方法，用于数组测试
from numpy.testing import (
    assert_equal,
    assert_almost_equal, assert_array_equal, assert_array_almost_equal,
    assert_allclose, assert_, assert_array_less,
    suppress_warnings)

# 导入 array 和 arange 函数，以及 numpy 模块，用于数组操作
from numpy import array, arange
import numpy as np

# 导入 scipy.fft 模块的 fft 函数，用于快速傅里叶变换
from scipy.fft import fft

# 导入 scipy.ndimage 模块的 correlate1d 函数，用于一维相关操作
from scipy.ndimage import correlate1d

# 导入 scipy.optimize 模块的 fmin 和 linear_sum_assignment 函数，用于优化和线性求解
from scipy.optimize import fmin, linear_sum_assignment

# 导入 scipy.signal 模块，用于信号处理
from scipy import signal

# 导入 scipy.signal 模块中的一些函数，包括相关、卷积、滤波等
from scipy.signal import (
    correlate, correlate2d, correlation_lags, convolve, convolve2d,
    fftconvolve, oaconvolve, choose_conv_method,
    hilbert, hilbert2, lfilter, lfilter_zi, filtfilt, butter, zpk2tf, zpk2sos,
    invres, invresz, vectorstrength, lfiltic, tf2sos, sosfilt, sosfiltfilt,
    sosfilt_zi, tf2zpk, BadCoefficients, detrend, unique_roots, residue,
    residuez)

# 导入 scipy.signal.windows 模块的 hann 函数，用于汉宁窗的生成
from scipy.signal.windows import hann

# 导入 scipy.signal._signaltools 模块中的部分函数，用于信号处理工具
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
                                      _group_poles)

# 导入 scipy.signal._upfirdn 模块中的 _upfirdn_modes 函数，用于上采样和下采样
from scipy.signal._upfirdn import _upfirdn_modes

# 导入 scipy._lib 模块中的 _testutils，用于测试工具
from scipy._lib import _testutils

# 导入 scipy._lib._util 模块中的 ComplexWarning, np_long, np_ulong，用于复杂数警告和数据类型处理
from scipy._lib._util import ComplexWarning, np_long, np_ulong


class _TestConvolve:

    # 测试 convolve 函数的基本功能
    def test_basic(self):
        a = [3, 4, 5, 6, 5, 4]
        b = [1, 2, 3]
        c = convolve(a, b)
        assert_array_equal(c, array([3, 10, 22, 28, 32, 32, 23, 12]))

    # 测试 convolve 函数在 mode="same" 模式下的功能
    def test_same(self):
        a = [3, 4, 5]
        b = [1, 2, 3, 4]
        c = convolve(a, b, mode="same")
        assert_array_equal(c, array([10, 22, 34]))

    # 测试 convolve 函数在 mode="same" 模式下，输入数组长度相等的功能
    def test_same_eq(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        c = convolve(a, b, mode="same")
        assert_array_equal(c, array([10, 22, 22]))

    # 测试 convolve 函数处理复数数组的功能
    def test_complex(self):
        x = array([1 + 1j, 2 + 1j, 3 + 1j])
        y = array([1 + 1j, 2 + 1j])
        z = convolve(x, y)
        assert_array_equal(z, array([2j, 2 + 6j, 5 + 8j, 5 + 5j]))

    # 测试 convolve 函数处理零维数组的功能
    def test_zero_rank(self):
        a = 1289
        b = 4567
        c = convolve(a, b)
        assert_equal(c, a * b)

    # 测试 convolve 函数处理可广播的数组的功能
    def test_broadcastable(self):
        a = np.arange(27).reshape(3, 3, 3)
        b = np.arange(3)
        for i in range(3):
            b_shape = [1]*3
            b_shape[i] = 3
            x = convolve(a, b.reshape(b_shape), method='direct')
            y = convolve(a, b.reshape(b_shape), method='fft')
            assert_allclose(x, y)

    # 测试 convolve 函数处理单个元素数组的功能
    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        c = convolve(a, b)
        assert_equal(c, a * b)

    # 测试 convolve 函数处理二维数组的功能
    def test_2d_arrays(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve(a, b)
        d = array([[2, 7, 16, 17, 12],
                   [10, 30, 62, 58, 38],
                   [12, 31, 58, 49, 30]])
        assert_array_equal(c, d)
    # 定义一个测试函数，用于测试输入交换的情况
    def test_input_swapping(self):
        # 创建一个形状为 (2, 2, 2) 的小数组，包含 0 到 7 的整数
        small = arange(8).reshape(2, 2, 2)
        # 创建一个形状为 (3, 3, 3) 的大数组，每个元素为虚数，包含 0 到 26 的整数
        big = 1j * arange(27).reshape(3, 3, 3)
        # 大数组的每个元素加上反向排列的 0 到 26 的整数，形成复数数组
        big += arange(27)[::-1].reshape(3, 3, 3)

        # 创建一个期望的输出数组，包含复数，形状为 (4, 4, 4)
        out_array = array(
            [[[0 + 0j, 26 + 0j, 25 + 1j, 24 + 2j],
              [52 + 0j, 151 + 5j, 145 + 11j, 93 + 11j],
              [46 + 6j, 133 + 23j, 127 + 29j, 81 + 23j],
              [40 + 12j, 98 + 32j, 93 + 37j, 54 + 24j]],

             [[104 + 0j, 247 + 13j, 237 + 23j, 135 + 21j],
              [282 + 30j, 632 + 96j, 604 + 124j, 330 + 86j],
              [246 + 66j, 548 + 180j, 520 + 208j, 282 + 134j],
              [142 + 66j, 307 + 161j, 289 + 179j, 153 + 107j]],

             [[68 + 36j, 157 + 103j, 147 + 113j, 81 + 75j],
              [174 + 138j, 380 + 348j, 352 + 376j, 186 + 230j],
              [138 + 174j, 296 + 432j, 268 + 460j, 138 + 278j],
              [70 + 138j, 145 + 323j, 127 + 341j, 63 + 197j]],

             [[32 + 72j, 68 + 166j, 59 + 175j, 30 + 100j],
              [68 + 192j, 139 + 433j, 117 + 455j, 57 + 255j],
              [38 + 222j, 73 + 499j, 51 + 521j, 21 + 291j],
              [12 + 144j, 20 + 318j, 7 + 331j, 0 + 182j]]])

        # 断言不同模式下使用 convolve 函数计算得到的结果与期望的输出数组相等
        assert_array_equal(convolve(small, big, 'full'), out_array)
        assert_array_equal(convolve(big, small, 'full'), out_array)
        assert_array_equal(convolve(small, big, 'same'),
                           out_array[1:3, 1:3, 1:3])
        assert_array_equal(convolve(big, small, 'same'),
                           out_array[0:3, 0:3, 0:3])
        assert_array_equal(convolve(small, big, 'valid'),
                           out_array[1:3, 1:3, 1:3])
        assert_array_equal(convolve(big, small, 'valid'),
                           out_array[1:3, 1:3, 1:3])

    # 定义一个测试函数，用于测试无效参数的情况
    def test_invalid_params(self):
        # 创建两个简单的列表 a 和 b
        a = [3, 4, 5]
        b = [1, 2, 3]
        # 断言使用 convolve 函数时，期望抛出 ValueError 异常，模式为 'spam'
        assert_raises(ValueError, convolve, a, b, mode='spam')
        # 断言使用 convolve 函数时，期望抛出 ValueError 异常，模式为 'eggs'，方法为 'fft'
        assert_raises(ValueError, convolve, a, b, mode='eggs', method='fft')
        # 断言使用 convolve 函数时，期望抛出 ValueError 异常，模式为 'ham'，方法为 'direct'
        assert_raises(ValueError, convolve, a, b, mode='ham', method='direct')
        # 断言使用 convolve 函数时，期望抛出 ValueError 异常，模式为 'full'，方法为 'bacon'
        assert_raises(ValueError, convolve, a, b, mode='full', method='bacon')
        # 断言使用 convolve 函数时，期望抛出 ValueError 异常，模式为 'same'，方法为 'bacon'
        assert_raises(ValueError, convolve, a, b, mode='same', method='bacon')
class TestConvolve(_TestConvolve):

    def test_valid_mode2(self):
        # See gh-5897
        # 定义数组 a 和 b 以及预期的输出 expected
        a = [1, 2, 3, 6, 5, 3]
        b = [2, 3, 4, 5, 3, 4, 2, 2, 1]
        expected = [70, 78, 73, 65]

        # 对数组 a 和 b 进行卷积运算，使用 'valid' 模式
        out = convolve(a, b, 'valid')
        # 断言输出结果 out 与预期结果 expected 相等
        assert_array_equal(out, expected)

        # 对数组 b 和 a 进行卷积运算，使用 'valid' 模式
        out = convolve(b, a, 'valid')
        # 断言输出结果 out 与预期结果 expected 相等
        assert_array_equal(out, expected)

        # 定义复数数组 a 和 b 以及预期的输出 expected
        a = [1 + 5j, 2 - 1j, 3 + 0j]
        b = [2 - 3j, 1 + 0j]
        expected = [2 - 3j, 8 - 10j]

        # 对复数数组 a 和 b 进行卷积运算，使用 'valid' 模式
        out = convolve(a, b, 'valid')
        # 断言输出结果 out 与预期结果 expected 相等
        assert_array_equal(out, expected)

        # 对复数数组 b 和 a 进行卷积运算，使用 'valid' 模式
        out = convolve(b, a, 'valid')
        # 断言输出结果 out 与预期结果 expected 相等
        assert_array_equal(out, expected)

    def test_same_mode(self):
        # 定义数组 a 和 b
        a = [1, 2, 3, 3, 1, 2]
        b = [1, 4, 3, 4, 5, 6, 7, 4, 3, 2, 1, 1, 3]
        # 对数组 a 和 b 进行卷积运算，使用 'same' 模式
        c = convolve(a, b, 'same')
        # 定义预期输出 d
        d = array([57, 61, 63, 57, 45, 36])
        # 断言输出结果 c 与预期结果 d 相等
        assert_array_equal(c, d)

    def test_invalid_shapes(self):
        # "invalid" 指的是两个数组的维度不能完全覆盖对方的维度，预期会抛出 ValueError
        # 定义数组 a 和 b，具有不同的形状
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        # 断言在 'valid' 模式下对 a 和 b 进行卷积会抛出 ValueError
        assert_raises(ValueError, convolve, *(a, b), **{'mode': 'valid'})
        # 断言在 'valid' 模式下对 b 和 a 进行卷积会抛出 ValueError
        assert_raises(ValueError, convolve, *(b, a), **{'mode': 'valid'})
    # 定义一个测试函数，用于测试卷积方法，参数 n 是测试数组的大小，默认为 100
    def test_convolve_method(self, n=100):
        # 手动编码数据结构，而不是使用即将移除的 np.sctypes 上的自定义过滤器
        types = {'uint16', 'uint64', 'int64', 'int32',
                 'complex128', 'float64', 'float16',
                 'complex64', 'float32', 'int16',
                 'uint8', 'uint32', 'int8', 'bool'}
        # 生成所有可能的参数组合，包括类型对 (t1, t2) 和模式 mode
        args = [(t1, t2, mode) for t1 in types for t2 in types
                               for mode in ['valid', 'full', 'same']]

        # 这些是随机数组，这意味着通过卷积两个 np.ones 数组进行测试要更强大
        np.random.seed(42)
        # 定义不同类型的随机数组
        array_types = {'i': np.random.choice([0, 1], size=n),
                       'f': np.random.randn(n)}
        # 'b' 和 'u' 类型的数组与 'i' 相同
        array_types['b'] = array_types['u'] = array_types['i']
        # 'c' 类型的数组是 'f' 类型的实部加上虚部（虚部为实部的一半）
        array_types['c'] = array_types['f'] + 0.5j * array_types['f']

        # 遍历所有参数组合
        for t1, t2, mode in args:
            # 根据 t1 和 t2 类型选择相应的数组 x1 和 x2，并转换类型
            x1 = array_types[np.dtype(t1).kind].astype(t1)
            x2 = array_types[np.dtype(t2).kind].astype(t2)

            # 对 x1 和 x2 进行卷积，使用 'fft' 和 'direct' 两种方法，并存储结果
            results = {key: convolve(x1, x2, method=key, mode=mode)
                       for key in ['fft', 'direct']}

            # 断言使用 'fft' 方法和 'direct' 方法得到的结果数据类型相同
            assert_equal(results['fft'].dtype, results['direct'].dtype)

            # 如果 t1 和 t2 都是 'bool' 类型，则使用 'direct' 方法
            if 'bool' in t1 and 'bool' in t2:
                assert_equal(choose_conv_method(x1, x2), 'direct')
                continue

            # 根据实验结果设置 np.allclose 的参数，以确保测试通过
            if any([t in {'complex64', 'float32'} for t in [t1, t2]]):
                kwargs = {'rtol': 1.0e-4, 'atol': 1e-6}
            elif 'float16' in [t1, t2]:
                kwargs = {'rtol': 1e-3, 'atol': 1e-3}
            else:
                kwargs = {'rtol': 1e-5, 'atol': 1e-8}

            # 使用 np.allclose 断言 'fft' 和 'direct' 方法的结果接近
            assert_allclose(results['fft'], results['direct'], **kwargs)

    # 定义一个测试函数，用于测试大输入时的卷积方法
    def test_convolve_method_large_input(self):
        # 对于不同的 n 值，测试卷积两个大整数时 'fft' 和 'direct' 方法的结果
        for n in [10, 20, 50, 51, 52, 53, 54, 60, 62]:
            z = np.array([2**n], dtype=np.int64)
            fft = convolve(z, z, method='fft')
            direct = convolve(z, z, method='direct')

            # 当 n < 50 时，确保 'fft' 和 'direct' 结果相等，并且等于 2 的 2*n 次方
            if n < 50:
                assert_equal(fft, direct)
                assert_equal(fft, 2**(2*n))
                assert_equal(direct, 2**(2*n))
    def test_mismatched_dims(self):
        # 定义测试方法：检查输入数组的维度是否匹配

        # 断言：使用直接方法进行卷积时，如果输入数组维度不匹配，应该抛出 ValueError 异常
        assert_raises(ValueError, convolve, [1], 2, method='direct')
        
        # 断言：使用直接方法进行卷积时，如果输入数组维度不匹配，应该抛出 ValueError 异常
        assert_raises(ValueError, convolve, 1, [2], method='direct')
        
        # 断言：使用 FFT 方法进行卷积时，如果输入数组维度不匹配，应该抛出 ValueError 异常
        assert_raises(ValueError, convolve, [1], 2, method='fft')
        
        # 断言：使用 FFT 方法进行卷积时，如果输入数组维度不匹配，应该抛出 ValueError 异常
        assert_raises(ValueError, convolve, 1, [2], method='fft')
        
        # 断言：如果输入数组维度不匹配，应该抛出 ValueError 异常
        assert_raises(ValueError, convolve, [1], [[2]])
        
        # 断言：如果输入数组维度不匹配，应该抛出 ValueError 异常
        assert_raises(ValueError, convolve, [3], 2)
    # 定义一个测试类 `_TestConvolve2d`，用于测试二维卷积函数
class _TestConvolve2d:

    # 测试处理二维数组的情况
    def test_2d_arrays(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        # 预期的卷积结果
        d = array([[2, 7, 16, 17, 12],
                   [10, 30, 62, 58, 38],
                   [12, 31, 58, 49, 30]])
        # 调用 convolve2d 函数进行卷积计算
        e = convolve2d(a, b)
        # 断言计算结果与预期结果相等
        assert_array_equal(e, d)

    # 测试有效模式（'valid' mode）
    def test_valid_mode(self):
        e = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        f = [[1, 2, 3], [3, 4, 5]]
        # 预期的卷积结果
        h = array([[62, 80, 98, 116, 134]])

        # 调用 convolve2d 函数，使用 'valid' 模式进行卷积计算
        g = convolve2d(e, f, 'valid')
        # 断言计算结果与预期结果相等
        assert_array_equal(g, h)

        # See gh-5897
        # 再次调用 convolve2d 函数，变换参数顺序，使用 'valid' 模式进行卷积计算
        g = convolve2d(f, e, 'valid')
        # 断言计算结果与预期结果相等
        assert_array_equal(g, h)

    # 测试有效模式（'valid' mode），处理复数
    def test_valid_mode_complx(self):
        e = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        f = np.array([[1, 2, 3], [3, 4, 5]], dtype=complex) + 1j
        # 预期的卷积结果
        h = array([[62.+24.j, 80.+30.j, 98.+36.j, 116.+42.j, 134.+48.j]])

        # 调用 convolve2d 函数，使用 'valid' 模式进行卷积计算
        g = convolve2d(e, f, 'valid')
        # 断言计算结果与预期结果相等（允许小的数值差异）
        assert_array_almost_equal(g, h)

        # See gh-5897
        # 再次调用 convolve2d 函数，变换参数顺序，使用 'valid' 模式进行卷积计算
        g = convolve2d(f, e, 'valid')
        # 断言计算结果与预期结果相等
        assert_array_equal(g, h)

    # 测试使用 fillvalue 参数
    def test_fillvalue(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        fillval = 1
        # 使用 fillvalue 参数，计算 'full' 模式的卷积
        c = convolve2d(a, b, 'full', 'fill', fillval)
        # 预期的卷积结果
        d = array([[24, 26, 31, 34, 32],
                   [28, 40, 62, 64, 52],
                   [32, 46, 67, 62, 48]])
        # 断言计算结果与预期结果相等
        assert_array_equal(c, d)

    # 测试 fillvalue 参数引发的错误情况
    def test_fillvalue_errors(self):
        # 预期的错误消息
        msg = "could not cast `fillvalue` directly to the output "
        # 使用 assert_raises 检查 ValueError 异常，匹配指定的错误消息
        with np.testing.suppress_warnings() as sup:
            sup.filter(ComplexWarning, "Casting complex values")
            with assert_raises(ValueError, match=msg):
                convolve2d([[1]], [[1, 2]], fillvalue=1j)

        # 预期的错误消息
        msg = "`fillvalue` must be scalar or an array with "
        # 使用 assert_raises 检查 ValueError 异常，匹配指定的错误消息
        with assert_raises(ValueError, match=msg):
            convolve2d([[1]], [[1, 2]], fillvalue=[1, 2])

    # 测试 fillvalue 参数为空引发的错误情况
    def test_fillvalue_empty(self):
        # 检查 fillvalue 参数为空时是否引发 ValueError 异常
        assert_raises(ValueError, convolve2d, [[1]], [[1, 2]],
                      fillvalue=[])

    # 测试边界模式为 'wrap' 的情况
    def test_wrap_boundary(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        # 使用 'wrap' 边界模式，计算 'full' 模式的卷积
        c = convolve2d(a, b, 'full', 'wrap')
        # 预期的卷积结果
        d = array([[80, 80, 74, 80, 80],
                   [68, 68, 62, 68, 68],
                   [80, 80, 74, 80, 80]])
        # 断言计算结果与预期结果相等
        assert_array_equal(c, d)

    # 测试边界模式为 'symm' 的情况
    def test_sym_boundary(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        # 使用 'symm' 边界模式，计算 'full' 模式的卷积
        c = convolve2d(a, b, 'full', 'symm')
        # 预期的卷积结果
        d = array([[34, 30, 44, 62, 66],
                   [52, 48, 62, 80, 84],
                   [82, 78, 92, 110, 114]])
        # 断言计算结果与预期结果相等
        assert_array_equal(c, d)

    # 参数化测试：使用 parametrize 标记，测试 convolve2d 和 correlate2d 函数
    @pytest.mark.parametrize('func', [convolve2d, correlate2d])
    @pytest.mark.parametrize('boundary, expected',
                             [('symm', [[37.0, 42.0, 44.0, 45.0]]),
                              ('wrap', [[43.0, 44.0, 42.0, 39.0]])])
    def test_same_with_boundary(self, func, boundary, expected):
        # 测试在使用 "symm" 和 "wrap" 边界处理方式时，使用一个"长"卷积核。
        # 卷积核的尺寸要求图像值被多次扩展以处理所请求的边界处理方法。
        # 这是对 gh-8684 和 gh-8814 的回归测试。
        image = np.array([[2.0, -1.0, 3.0, 4.0]])
        kernel = np.ones((1, 21))
        result = func(image, kernel, mode='same', boundary=boundary)
        # 预期结果是手动计算的。因为卷积核全为1，预期 convolve2d 和 correlate2d 将得到相同的结果。
        assert_array_equal(result, expected)

    def test_boundary_extension_same(self):
        # gh-12686 的回归测试。
        # 使用适当的参数调用 ndimage.convolve 来创建预期的结果。
        import scipy.ndimage as ndi
        a = np.arange(1, 10*3+1, dtype=float).reshape(10, 3)
        b = np.arange(1, 10*10+1, dtype=float).reshape(10, 10)
        c = convolve2d(a, b, mode='same', boundary='wrap')
        assert_array_equal(c, ndi.convolve(a, b, mode='wrap', origin=(-1, -1)))

    def test_boundary_extension_full(self):
        # gh-12686 的回归测试。
        # 使用适当的参数调用 ndimage.convolve 来创建预期的结果。
        import scipy.ndimage as ndi
        a = np.arange(1, 3*3+1, dtype=float).reshape(3, 3)
        b = np.arange(1, 6*6+1, dtype=float).reshape(6, 6)
        c = convolve2d(a, b, mode='full', boundary='wrap')
        apad = np.pad(a, ((3, 3), (3, 3)), 'wrap')
        assert_array_equal(c, ndi.convolve(apad, b, mode='wrap')[:-1, :-1])

    def test_invalid_shapes(self):
        # "无效" 意味着没有一个数组的维度至少与另一个数组的对应维度一样大。
        # 这种设置应该引发 ValueError。
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        assert_raises(ValueError, convolve2d, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, convolve2d, *(b, a), **{'mode': 'valid'})
class TestConvolve2d(_TestConvolve2d):  # 定义一个测试类，继承自_TestConvolve2d类

    def test_same_mode(self):  # 测试函数：验证'same'模式下的卷积运算
        e = [[1, 2, 3], [3, 4, 5]]  # 输入矩阵e
        f = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]  # 卷积核f
        g = convolve2d(e, f, 'same')  # 进行'same'模式下的二维卷积运算
        h = array([[22, 28, 34],  # 期望的卷积结果h
                   [80, 98, 116]])
        assert_array_equal(g, h)  # 断言卷积结果g与期望结果h相等

    def test_valid_mode2(self):  # 测试函数：验证'valid'模式下的卷积运算
        # See gh-5897
        e = [[1, 2, 3], [3, 4, 5]]  # 输入矩阵e
        f = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]  # 卷积核f
        expected = [[62, 80, 98, 116, 134]]  # 期望的卷积结果expected

        out = convolve2d(e, f, 'valid')  # 进行'valid'模式下的二维卷积运算
        assert_array_equal(out, expected)  # 断言卷积结果out与期望结果expected相等

        out = convolve2d(f, e, 'valid')  # 交换输入顺序进行'valid'模式下的二维卷积运算
        assert_array_equal(out, expected)  # 断言卷积结果out与期望结果expected相等

        e = [[1 + 1j, 2 - 3j], [3 + 1j, 4 + 0j]]  # 复数输入矩阵e
        f = [[2 - 1j, 3 + 2j, 4 + 0j], [4 - 0j, 5 + 1j, 6 - 3j]]  # 复数卷积核f
        expected = [[27 - 1j, 46. + 2j]]  # 复数期望的卷积结果expected

        out = convolve2d(e, f, 'valid')  # 进行'valid'模式下的二维复数卷积运算
        assert_array_equal(out, expected)  # 断言卷积结果out与期望结果expected相等

        # See gh-5897
        out = convolve2d(f, e, 'valid')  # 交换输入顺序进行'valid'模式下的二维复数卷积运算
        assert_array_equal(out, expected)  # 断言卷积结果out与期望结果expected相等

    def test_consistency_convolve_funcs(self):  # 测试函数：比较np.convolve、signal.convolve、signal.convolve2d的一致性
        a = np.arange(5)  # 创建一个长度为5的数组a
        b = np.array([3.2, 1.4, 3])  # 创建数组b
        for mode in ['full', 'valid', 'same']:  # 遍历模式列表
            assert_almost_equal(np.convolve(a, b, mode=mode),  # 断言np.convolve与signal.convolve的结果近似相等
                                signal.convolve(a, b, mode=mode))
            assert_almost_equal(np.squeeze(
                signal.convolve2d([a], [b], mode=mode)),
                signal.convolve(a, b, mode=mode))

    def test_invalid_dims(self):  # 测试函数：验证不合法维度输入时的异常处理
        assert_raises(ValueError, convolve2d, 3, 4)  # 断言当输入维度为标量时抛出ValueError异常
        assert_raises(ValueError, convolve2d, [3], [4])  # 断言当输入维度为一维数组时抛出ValueError异常
        assert_raises(ValueError, convolve2d, [[[3]]], [[[4]]])  # 断言当输入维度为三维数组时抛出ValueError异常

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit("Can't create large array for test")  # 标记：在32位系统上不支持创建大数组进行测试
    def test_large_array(self):  # 测试函数：验证大数组情况下的卷积运算
        # Test indexing doesn't overflow an int (gh-10761)
        n = 2**31 // (1000 * np.int64().itemsize)  # 计算n的值，确保索引不会溢出整型范围
        _testutils.check_free_memory(2 * n * 1001 * np.int64().itemsize / 1e6)  # 检查空闲内存是否足够进行大数组测试

        # Create a chequered pattern of 1s and 0s
        a = np.zeros(1001 * n, dtype=np.int64)  # 创建一个长为1001*n的零数组a
        a[::2] = 1  # 设置间隔为2的元素为1，形成棋盘格模式
        a = np.lib.stride_tricks.as_strided(a, shape=(n, 1000), strides=(8008, 8))  # 利用步幅技巧创建新的数组a

        count = signal.convolve2d(a, [[1, 1]])  # 对数组a进行二维卷积计算
        fails = np.where(count > 1)  # 找出卷积结果中大于1的位置
        assert fails[0].size == 0  # 断言找不到大于1的位置

class TestFFTConvolve:

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])  # 参数化测试：测试不同的axes参数值
    def test_real(self, axes):  # 测试函数：验证实数情况下的FFT卷积运算
        a = array([1, 2, 3])  # 输入数组a
        expected = array([1, 4, 10, 12, 9.])  # 期望的卷积结果expected

        if axes == '':  # 如果axes为空字符串
            out = fftconvolve(a, a)  # 进行无轴的FFT卷积运算
        else:
            out = fftconvolve(a, a, axes=axes)  # 根据指定的axes参数进行FFT卷积运算

        assert_array_almost_equal(out, expected)  # 断言FFT卷积结果out与期望结果expected近似相等

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])  # 参数化测试：测试不同的axes参数值
    # 定义一个测试方法，用于测试 fftconvolve 函数在实数轴上的行为
    def test_real_axes(self, axes):
        # 创建一个包含整数的一维数组
        a = array([1, 2, 3])
        # 创建一个预期的一维数组，包含浮点数
        expected = array([1, 4, 10, 12, 9.])

        # 在第一个维度上复制数组 a，结果是一个二维数组
        a = np.tile(a, [2, 1])
        # 在第一个维度上复制数组 expected，结果是一个二维数组
        expected = np.tile(expected, [2, 1])

        # 对数组 a 进行自相关运算，根据参数 axes 的指定进行计算
        out = fftconvolve(a, a, axes=axes)
        # 检查计算结果 out 是否与预期 expected 接近
        assert_array_almost_equal(out, expected)

    # 使用 pytest 的参数化装饰器，定义一个测试方法，用于测试 fftconvolve 函数在复数轴上的行为
    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_complex(self, axes):
        # 创建一个包含复数的一维数组
        a = array([1 + 1j, 2 + 2j, 3 + 3j])
        # 创建一个预期的一维数组，包含复数
        expected = array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

        # 根据参数 axes 的值，选择性地调用 fftconvolve 函数
        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        # 检查计算结果 out 是否与预期 expected 接近
        assert_array_almost_equal(out, expected)

    # 使用 pytest 的参数化装饰器，定义一个测试方法，用于测试 fftconvolve 函数在复数轴上的行为，其中 axes 是列表
    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_complex_axes(self, axes):
        # 创建一个包含复数的一维数组
        a = array([1 + 1j, 2 + 2j, 3 + 3j])
        # 创建一个预期的一维数组，包含复数
        expected = array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

        # 在第一个维度上复制数组 a，结果是一个二维数组
        a = np.tile(a, [2, 1])
        # 在第一个维度上复制数组 expected，结果是一个二维数组
        expected = np.tile(expected, [2, 1])

        # 对数组 a 进行自相关运算，根据参数 axes 的指定进行计算
        out = fftconvolve(a, a, axes=axes)
        # 检查计算结果 out 是否与预期 expected 接近
        assert_array_almost_equal(out, expected)

    # 使用 pytest 的参数化装饰器，定义一个测试方法，用于测试 fftconvolve 函数在二维实数数组上的行为，其中 axes 是字符串或列表
    @pytest.mark.parametrize('axes', ['',
                                      None,
                                      [0, 1],
                                      [1, 0],
                                      [0, -1],
                                      [-1, 0],
                                      [-2, 1],
                                      [1, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_real_same(self, axes):
        # 创建一个二维数组
        a = array([[1, 2, 3],
                   [4, 5, 6]])
        # 创建一个预期的二维数组
        expected = array([[1, 4, 10, 12, 9],
                          [8, 26, 56, 54, 36],
                          [16, 40, 73, 60, 36]])

        # 根据参数 axes 的值，选择性地调用 fftconvolve 函数
        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        # 检查计算结果 out 是否与预期 expected 接近
        assert_array_almost_equal(out, expected)

    # 使用 pytest 的参数化装饰器，定义一个测试方法，用于测试 fftconvolve 函数在二维实数数组上的行为，其中 axes 是列表
    @pytest.mark.parametrize('axes', [[1, 2],
                                      [2, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-2, 2],
                                      [2, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_real_same_axes(self, axes):
        # 创建一个二维数组
        a = array([[1, 2, 3],
                   [4, 5, 6]])
        # 创建一个预期的二维数组
        expected = array([[1, 4, 10, 12, 9],
                          [8, 26, 56, 54, 36],
                          [16, 40, 73, 60, 36]])

        # 在第一个维度上复制数组 a，结果是一个三维数组
        a = np.tile(a, [2, 1, 1])
        # 在第一个维度上复制数组 expected，结果是一个三维数组
        expected = np.tile(expected, [2, 1, 1])

        # 对数组 a 进行自相关运算，根据参数 axes 的指定进行计算
        out = fftconvolve(a, a, axes=axes)
        # 检查计算结果 out 是否与预期 expected 接近
        assert_array_almost_equal(out, expected)
    @pytest.mark.parametrize('axes', ['',
                                      None,
                                      [0, 1],
                                      [1, 0],
                                      [0, -1],
                                      [-1, 0],
                                      [-2, 1],
                                      [1, -2],
                                      [-2, -1],
                                      [-1, -2]])


    # 使用 pytest 的参数化装饰器，为 test_2d_complex_same 方法创建多个测试用例
    def test_2d_complex_same(self, axes):
        # 创建一个复数数组 a
        a = array([[1 + 2j, 3 + 4j, 5 + 6j],
                   [2 + 1j, 4 + 3j, 6 + 5j]])
        # 创建预期的结果数组 expected
        expected = array([
            [-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
            [10j, 44j, 118j, 156j, 122j],
            [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]
            ])

        # 根据不同的 axes 参数调用 fftconvolve 函数
        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)

        # 断言 out 和预期结果 expected 几乎相等
        assert_array_almost_equal(out, expected)


    @pytest.mark.parametrize('axes', [[1, 2],
                                      [2, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-2, 2],
                                      [2, -2],
                                      [-2, -1],
                                      [-1, -2]])


    # 使用 pytest 的参数化装饰器，为 test_2d_complex_same_axes 方法创建多个测试用例
    def test_2d_complex_same_axes(self, axes):
        # 创建一个复数数组 a
        a = array([[1 + 2j, 3 + 4j, 5 + 6j],
                   [2 + 1j, 4 + 3j, 6 + 5j]])
        # 创建预期的结果数组 expected
        expected = array([
            [-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
            [10j, 44j, 118j, 156j, 122j],
            [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]
            ])

        # 将数组 a 和 expected 在第一个维度上复制一次
        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])

        # 根据不同的 axes 参数调用 fftconvolve 函数
        out = fftconvolve(a, a, axes=axes)

        # 断言 out 和预期结果 expected 几乎相等
        assert_array_almost_equal(out, expected)


    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])


    # 使用 pytest 的参数化装饰器，为 test_real_same_mode 方法创建多个测试用例
    def test_real_same_mode(self, axes):
        # 创建两个实数数组 a 和 b
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        # 创建预期的结果数组 expected_1 和 expected_2
        expected_1 = array([35., 41., 47.])
        expected_2 = array([9., 20., 25., 35., 41., 47., 39., 28., 2.])

        # 根据不同的 axes 参数调用 fftconvolve 函数，并指定 mode='same'
        if axes == '':
            out = fftconvolve(a, b, 'same')
        else:
            out = fftconvolve(a, b, 'same', axes=axes)
        # 断言 out 和预期结果 expected_1 几乎相等
        assert_array_almost_equal(out, expected_1)

        # 根据不同的 axes 参数调用 fftconvolve 函数，并指定 mode='same'
        if axes == '':
            out = fftconvolve(b, a, 'same')
        else:
            out = fftconvolve(b, a, 'same', axes=axes)
        # 断言 out 和预期结果 expected_2 几乎相等
        assert_array_almost_equal(out, expected_2)


    @pytest.mark.parametrize('axes', [1, -1, [1], [-1]])


    # 使用 pytest 的参数化装饰器，为下一个方法创建多个测试用例
    pass
    def test_real_same_mode_axes(self, axes):
        # 定义输入数组 a 和 b
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        # 定义期望的输出结果 expected_1 和 expected_2
        expected_1 = array([35., 41., 47.])
        expected_2 = array([9., 20., 25., 35., 41., 47., 39., 28., 2.])

        # 对数组 a 和 b 进行复制以扩展维度
        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected_1 = np.tile(expected_1, [2, 1])
        expected_2 = np.tile(expected_2, [2, 1])

        # 使用 fftconvolve 函数计算卷积，保持输出数组大小相同
        out = fftconvolve(a, b, 'same', axes=axes)
        assert_array_almost_equal(out, expected_1)

        out = fftconvolve(b, a, 'same', axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_valid_mode_real(self, axes):
        # See gh-5897
        # 定义输入数组 a 和 b
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        # 定义期望的输出结果 expected
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        # 根据不同的 axes 参数调用 fftconvolve 函数计算卷积，保持输出数组大小相同
        if axes == '':
            out = fftconvolve(a, b, 'valid')
        else:
            out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        if axes == '':
            out = fftconvolve(b, a, 'valid')
        else:
            out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1]])
    def test_valid_mode_real_axes(self, axes):
        # See gh-5897
        # 定义输入数组 a 和 b
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        # 定义期望的输出结果 expected
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        # 对输入数组进行复制以扩展维度
        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        # 使用 fftconvolve 函数计算卷积，保持输出数组大小相同，指定 axes 参数
        out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_valid_mode_complex(self, axes):
        # 定义输入数组 a 和 b（包含复数）
        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        # 定义期望的输出结果 expected（包含复数）
        expected = array([45. + 12.j, 30. + 23.j, 48 + 32.j])

        # 根据不同的 axes 参数调用 fftconvolve 函数计算卷积，保持输出数组大小相同
        if axes == '':
            out = fftconvolve(a, b, 'valid')
        else:
            out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        if axes == '':
            out = fftconvolve(b, a, 'valid')
        else:
            out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_valid_mode_complex_axes(self, axes):
        # 定义输入数组 a 和 b（包含复数）
        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        # 定义期望的输出结果 expected（包含复数）
        expected = array([45. + 12.j, 30. + 23.j, 48 + 32.j])

        # 对输入数组进行复制以扩展维度
        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        # 使用 fftconvolve 函数计算卷积，保持输出数组大小相同，指定 axes 参数
        out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)
    def test_valid_mode_ignore_nonaxes(self):
        # 根据 GitHub 问题 #5897 进行测试
        a = array([3, 2, 1])  # 创建包含元素 3, 2, 1 的数组 a
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])  # 创建包含元素 3, 3, 5, ... 的数组 b
        expected = array([24., 31., 41., 43., 49., 25., 12.])  # 创建预期结果数组 expected

        a = np.tile(a, [2, 1])  # 在 a 上进行 2x1 的瓦片复制
        b = np.tile(b, [1, 1])  # 在 b 上进行 1x1 的瓦片复制
        expected = np.tile(expected, [2, 1])  # 在 expected 上进行 2x1 的瓦片复制

        out = fftconvolve(a, b, 'valid', axes=1)  # 使用 fftconvolve 进行有效卷积，指定轴为 1
        assert_array_almost_equal(out, expected)  # 断言输出 out 与预期结果 expected 几乎相等

    def test_empty(self):
        # 回归测试 #1745：处理长度为 0 的输入时不崩溃
        assert_(fftconvolve([], []).size == 0)  # 确保空数组卷积后的大小为 0
        assert_(fftconvolve([5, 6], []).size == 0)  # 确保数组 [5, 6] 和空数组卷积后的大小为 0
        assert_(fftconvolve([], [7]).size == 0)  # 确保空数组和数组 [7] 卷积后的大小为 0

    def test_zero_rank(self):
        a = array(4967)  # 创建一个标量数组 a
        b = array(3920)  # 创建一个标量数组 b
        out = fftconvolve(a, b)  # 对标量数组 a 和 b 进行卷积
        assert_equal(out, a * b)  # 断言输出 out 等于标量数组 a 和 b 的乘积

    def test_single_element(self):
        a = array([4967])  # 创建包含单个元素 4967 的数组 a
        b = array([3920])  # 创建包含单个元素 3920 的数组 b
        out = fftconvolve(a, b)  # 对数组 a 和 b 进行卷积
        assert_equal(out, a * b)  # 断言输出 out 等于数组 a 和 b 的乘积

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_random_data(self, axes):
        np.random.seed(1234)  # 使用种子 1234 初始化随机数生成器
        a = np.random.rand(1233) + 1j * np.random.rand(1233)  # 创建包含随机复数的数组 a
        b = np.random.rand(1321) + 1j * np.random.rand(1321)  # 创建包含随机复数的数组 b
        expected = np.convolve(a, b, 'full')  # 计算完全卷积的预期结果

        if axes == '':
            out = fftconvolve(a, b, 'full')  # 如果 axes 为空字符串，则计算完全卷积
        else:
            out = fftconvolve(a, b, 'full', axes=axes)  # 否则按指定的轴计算完全卷积
        assert_(np.allclose(out, expected, rtol=1e-10))  # 断言输出 out 与预期结果 expected 在给定容差下全部相近

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_random_data_axes(self, axes):
        np.random.seed(1234)  # 使用种子 1234 初始化随机数生成器
        a = np.random.rand(1233) + 1j * np.random.rand(1233)  # 创建包含随机复数的数组 a
        b = np.random.rand(1321) + 1j * np.random.rand(1321)  # 创建包含随机复数的数组 b
        expected = np.convolve(a, b, 'full')  # 计算完全卷积的预期结果

        a = np.tile(a, [2, 1])  # 在 a 上进行 2x1 的瓦片复制
        b = np.tile(b, [2, 1])  # 在 b 上进行 2x1 的瓦片复制
        expected = np.tile(expected, [2, 1])  # 在 expected 上进行 2x1 的瓦片复制

        out = fftconvolve(a, b, 'full', axes=axes)  # 按指定的轴进行完全卷积
        assert_(np.allclose(out, expected, rtol=1e-10))  # 断言输出 out 与预期结果 expected 在给定容差下全部相近

    @pytest.mark.parametrize('axes', [[1, 4],
                                      [4, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-4, 4],
                                      [4, -4],
                                      [-4, -1],
                                      [-1, -4]])
    # 定义一个测试函数，用于测试多维数组的卷积操作
    def test_random_data_multidim_axes(self, axes):
        # 定义两个数组的形状
        a_shape, b_shape = (123, 22), (132, 11)
        # 设置随机数种子，确保结果可重复
        np.random.seed(1234)
        # 生成复数随机数组 a 和 b
        a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
        b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
        # 计算期望的卷积结果
        expected = convolve2d(a, b, 'full')

        # 在 a 和 expected 数组的第三和第四个维度上添加额外的维度
        a = a[:, :, None, None, None]
        b = b[:, :, None, None, None]
        expected = expected[:, :, None, None, None]

        # 移动数组 a、b 和 expected 的轴，使得第一维度和第三维度交换位置，第二维度和第五维度交换位置
        a = np.moveaxis(a.swapaxes(0, 2), 1, 4)
        b = np.moveaxis(b.swapaxes(0, 2), 1, 4)
        expected = np.moveaxis(expected.swapaxes(0, 2), 1, 4)

        # 在 a 和 b 的第二和第四个维度上分别复制数组，以进行广播测试
        a = np.tile(a, [2, 1, 3, 1, 1])
        b = np.tile(b, [2, 1, 1, 4, 1])
        expected = np.tile(expected, [2, 1, 3, 4, 1])

        # 使用 fftconvolve 函数计算卷积，指定 axes 参数来控制卷积计算的维度
        out = fftconvolve(a, b, 'full', axes=axes)
        # 断言计算得到的结果与期望值在给定的相对和绝对容差下相等
        assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    # 用于测试多个不同大小的输入数组进行卷积计算
    @pytest.mark.slow
    @pytest.mark.parametrize(
        'n',
        list(range(1, 100)) +
        list(range(1000, 1500)) +
        np.random.RandomState(1234).randint(1001, 10000, 5).tolist())
    def test_many_sizes(self, n):
        # 生成大小为 n 的随机复数数组 a 和 b
        a = np.random.rand(n) + 1j * np.random.rand(n)
        b = np.random.rand(n) + 1j * np.random.rand(n)
        # 计算期望的卷积结果
        expected = np.convolve(a, b, 'full')

        # 使用 fftconvolve 函数计算卷积，对比计算结果与期望值
        out = fftconvolve(a, b, 'full')
        assert_allclose(out, expected, atol=1e-10)

        # 在指定的轴上进行卷积计算，再次对比计算结果与期望值
        out = fftconvolve(a, b, 'full', axes=[0])
        assert_allclose(out, expected, atol=1e-10)

    # 测试 FFT 卷积在包含 NaN 和 Inf 值的信号上的行为
    def test_fft_nan(self):
        n = 1000
        rng = np.random.default_rng(43876432987)
        # 生成包含 NaN 和 Inf 值的信号
        sig_nan = rng.standard_normal(n)

        # 分别设置信号中的一个值为 NaN 和 Inf，并计算其与滤波系数的卷积
        for val in [np.nan, np.inf]:
            sig_nan[100] = val
            coeffs = signal.firwin(200, 0.2)

            # 断言在使用 FFT 方法计算卷积时会发出 RuntimeWarning 警告，匹配特定的警告消息
            msg = "Use of fft convolution.*|invalid value encountered.*"
            with pytest.warns(RuntimeWarning, match=msg):
                signal.convolve(sig_nan, coeffs, mode='same', method='fft')
# 定义一个函数 fftconvolve_err，用于抛出 RuntimeError 异常，指示退回到 fftconvolve 函数
def fftconvolve_err(*args, **kwargs):
    raise RuntimeError('Fell back to fftconvolve')

# 定义函数 gen_oa_shapes，生成给定大小列表的所有可能的形状组合，使得它们的差的绝对值大于3
def gen_oa_shapes(sizes):
    return [(a, b) for a, b in product(sizes, repeat=2)
            if abs(a - b) > 3]

# 定义函数 gen_oa_shapes_2d，生成二维形状的所有可能组合，包括两个 gen_oa_shapes 的组合和不同的卷积模式
def gen_oa_shapes_2d(sizes):
    # 生成两个 gen_oa_shapes 的结果列表
    shapes0 = gen_oa_shapes(sizes)
    shapes1 = gen_oa_shapes(sizes)
    # 将两个结果列表进行组合，形成所有可能的形状组合
    shapes = [ishapes0+ishapes1 for ishapes0, ishapes1 in
              zip(shapes0, shapes1)]

    # 定义卷积的模式列表
    modes = ['full', 'valid', 'same']
    # 返回形状和模式的所有可能组合
    return [ishapes+(imode,) for ishapes, imode in product(shapes, modes)
            if imode != 'valid' or
            (ishapes[0] > ishapes[1] and ishapes[2] > ishapes[3]) or
            (ishapes[0] < ishapes[1] and ishapes[2] < ishapes[3])]

# 定义函数 gen_oa_shapes_eq，生成给定大小列表的所有可能的形状组合，其中第一个大小大于等于第二个大小
def gen_oa_shapes_eq(sizes):
    return [(a, b) for a, b in product(sizes, repeat=2)
            if a >= b]

# 定义一个测试类 TestOAConvolve
class TestOAConvolve:
    # 标记测试为慢速测试
    @pytest.mark.slow()
    # 参数化测试用例，使用 gen_oa_shapes_eq 生成的形状对
    @pytest.mark.parametrize('shape_a_0, shape_b_0',
                             gen_oa_shapes_eq(list(range(100)) +
                                              list(range(100, 1000, 23)))
                             )
    # 定义测试方法 test_real_manylens，测试 oaconvolve 的输出是否与 fftconvolve 的期望输出几乎相等
    def test_real_manylens(self, shape_a_0, shape_b_0):
        # 生成随机数组 a 和 b，形状分别为 shape_a_0 和 shape_b_0
        a = np.random.rand(shape_a_0)
        b = np.random.rand(shape_b_0)

        # 计算 fftconvolve 的期望输出
        expected = fftconvolve(a, b)
        # 调用 oaconvolve 计算输出
        out = oaconvolve(a, b)

        # 断言输出数组 out 几乎等于期望输出 expected
        assert_array_almost_equal(out, expected)

    # 参数化测试用例，使用 gen_oa_shapes 生成的形状对和 is_complex 参数
    @pytest.mark.parametrize('shape_a_0, shape_b_0',
                             gen_oa_shapes([50, 47, 6, 4, 1]))
    # 参数化测试用例，使用 is_complex 参数
    @pytest.mark.parametrize('is_complex', [True, False])
    # 参数化测试用例，使用 mode 参数
    @pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
    # 定义测试方法 test_1d_noaxes，测试 oaconvolve 的输出是否与 fftconvolve 的期望输出几乎相等
    # 同时模拟复杂的测试条件，如修改 fftconvolve 的行为
    def test_1d_noaxes(self, shape_a_0, shape_b_0,
                       is_complex, mode, monkeypatch):
        # 生成随机数组 a 和 b，形状分别为 shape_a_0 和 shape_b_0
        a = np.random.rand(shape_a_0)
        b = np.random.rand(shape_b_0)
        # 如果 is_complex 为 True，则使数组 a 和 b 变为复数数组
        if is_complex:
            a = a + 1j*np.random.rand(shape_a_0)
            b = b + 1j*np.random.rand(shape_b_0)

        # 计算 fftconvolve 的期望输出
        expected = fftconvolve(a, b, mode=mode)

        # 修改 fftconvolve 的行为为抛出异常
        monkeypatch.setattr(signal._signaltools, 'fftconvolve',
                            fftconvolve_err)
        # 调用 oaconvolve 计算输出
        out = oaconvolve(a, b, mode=mode)

        # 断言输出数组 out 几乎等于期望输出 expected
        assert_array_almost_equal(out, expected)

    # 参数化测试用例，使用 axes 参数
    @pytest.mark.parametrize('axes', [0, 1])
    # 参数化测试用例，使用 gen_oa_shapes 生成的形状对
    @pytest.mark.parametrize('shape_a_0, shape_b_0',
                             gen_oa_shapes([50, 47, 6, 4]))
    # 参数化测试用例，使用 shape_a_extra 和 shape_b_extra 参数
    @pytest.mark.parametrize('shape_a_extra', [1, 3])
    @pytest.mark.parametrize('shape_b_extra', [1, 3])
    # 参数化测试用例，使用 is_complex 参数
    @pytest.mark.parametrize('is_complex', [True, False])
    # 参数化测试用例，使用 mode 参数
    @pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
    # 定义测试方法，用于测试一维轴向的情况
    def test_1d_axes(self, axes, shape_a_0, shape_b_0,
                     shape_a_extra, shape_b_extra,
                     is_complex, mode, monkeypatch):
        # 创建包含两个元素的轴列表，每个元素为 shape_a_extra 的值
        ax_a = [shape_a_extra]*2
        # 创建包含两个元素的轴列表，每个元素为 shape_b_extra 的值
        ax_b = [shape_b_extra]*2
        # 将第 axes 个轴设置为 shape_a_0
        ax_a[axes] = shape_a_0
        # 将第 axes 个轴设置为 shape_b_0
        ax_b[axes] = shape_b_0

        # 随机生成形状为 ax_a 的数组 a
        a = np.random.rand(*ax_a)
        # 随机生成形状为 ax_b 的数组 b
        b = np.random.rand(*ax_b)
        # 如果 is_complex 为 True，将 a 和 b 转换为复数形式
        if is_complex:
            a = a + 1j*np.random.rand(*ax_a)
            b = b + 1j*np.random.rand(*ax_b)

        # 计算预期结果，使用 fftconvolve 函数进行卷积计算
        expected = fftconvolve(a, b, mode=mode, axes=axes)

        # 使用 monkeypatch 修改 signal._signaltools 中的 fftconvolve 函数为 fftconvolve_err
        monkeypatch.setattr(signal._signaltools, 'fftconvolve',
                            fftconvolve_err)
        # 调用 oaconvolve 函数进行卷积计算
        out = oaconvolve(a, b, mode=mode, axes=axes)

        # 断言输出 out 与预期结果 expected 几乎相等
        assert_array_almost_equal(out, expected)

    # 使用 pytest 参数化装饰器，定义测试方法，测试二维情况下没有轴向的情况
    @pytest.mark.parametrize('shape_a_0, shape_b_0, '
                             'shape_a_1, shape_b_1, mode',
                             gen_oa_shapes_2d([50, 47, 6, 4]))
    @pytest.mark.parametrize('is_complex', [True, False])
    def test_2d_noaxes(self, shape_a_0, shape_b_0,
                       shape_a_1, shape_b_1, mode,
                       is_complex, monkeypatch):
        # 随机生成形状为 (shape_a_0, shape_a_1) 的数组 a
        a = np.random.rand(shape_a_0, shape_a_1)
        # 随机生成形状为 (shape_b_0, shape_b_1) 的数组 b
        b = np.random.rand(shape_b_0, shape_b_1)
        # 如果 is_complex 为 True，将 a 和 b 转换为复数形式
        if is_complex:
            a = a + 1j*np.random.rand(shape_a_0, shape_a_1)
            b = b + 1j*np.random.rand(shape_b_0, shape_b_1)

        # 计算预期结果，使用 fftconvolve 函数进行卷积计算
        expected = fftconvolve(a, b, mode=mode)

        # 使用 monkeypatch 修改 signal._signaltools 中的 fftconvolve 函数为 fftconvolve_err
        monkeypatch.setattr(signal._signaltools, 'fftconvolve',
                            fftconvolve_err)
        # 调用 oaconvolve 函数进行卷积计算
        out = oaconvolve(a, b, mode=mode)

        # 断言输出 out 与预期结果 expected 几乎相等
        assert_array_almost_equal(out, expected)

    # 使用 pytest 参数化装饰器，定义测试方法，测试二维情况下指定轴向的情况
    @pytest.mark.parametrize('axes', [[0, 1], [0, 2], [1, 2]])
    @pytest.mark.parametrize('shape_a_0, shape_b_0, '
                             'shape_a_1, shape_b_1, mode',
                             gen_oa_shapes_2d([50, 47, 6, 4]))
    @pytest.mark.parametrize('shape_a_extra', [1, 3])
    @pytest.mark.parametrize('shape_b_extra', [1, 3])
    @pytest.mark.parametrize('is_complex', [True, False])
    def test_2d_axes(self, axes, shape_a_0, shape_b_0,
                     shape_a_1, shape_b_1, mode,
                     shape_a_extra, shape_b_extra,
                     is_complex, monkeypatch):
        # 创建包含三个元素的轴列表，每个元素为 shape_a_extra 的值
        ax_a = [shape_a_extra]*3
        # 创建包含三个元素的轴列表，每个元素为 shape_b_extra 的值
        ax_b = [shape_b_extra]*3
        # 将 axes 列表中的第一个轴设置为 shape_a_0
        ax_a[axes[0]] = shape_a_0
        # 将 axes 列表中的第一个轴设置为 shape_b_0
        ax_b[axes[0]] = shape_b_0
        # 将 axes 列表中的第二个轴设置为 shape_a_1
        ax_a[axes[1]] = shape_a_1
        # 将 axes 列表中的第二个轴设置为 shape_b_1
        ax_b[axes[1]] = shape_b_1

        # 随机生成形状为 ax_a 的数组 a
        a = np.random.rand(*ax_a)
        # 随机生成形状为 ax_b 的数组 b
        b = np.random.rand(*ax_b)
        # 如果 is_complex 为 True，将 a 和 b 转换为复数形式
        if is_complex:
            a = a + 1j*np.random.rand(*ax_a)
            b = b + 1j*np.random.rand(*ax_b)

        # 计算预期结果，使用 fftconvolve 函数进行卷积计算，指定轴向为 axes
        expected = fftconvolve(a, b, mode=mode, axes=axes)

        # 使用 monkeypatch 修改 signal._signaltools 中的 fftconvolve 函数为 fftconvolve_err
        monkeypatch.setattr(signal._signaltools, 'fftconvolve',
                            fftconvolve_err)
        # 调用 oaconvolve 函数进行卷积计算，指定轴向为 axes
        out = oaconvolve(a, b, mode=mode, axes=axes)

        # 断言输出 out 与预期结果 expected 几乎相等
        assert_array_almost_equal(out, expected)
    # 定义测试函数，用于测试 oaconvolve 函数在特定条件下的行为
    def test_empty(self):
        # 断言：测试空列表与空列表的卷积结果的长度是否为 0
        assert_(oaconvolve([], []).size == 0)
        # 断言：测试一个空列表与非空列表的卷积结果的长度是否为 0
        assert_(oaconvolve([5, 6], []).size == 0)
        # 断言：测试一个非空列表与一个空列表的卷积结果的长度是否为 0
        assert_(oaconvolve([], [7]).size == 0)

    # 定义测试函数，测试输入数组为零秩数组时 oaconvolve 函数的行为
    def test_zero_rank(self):
        # 创建零秩数组 a 和 b
        a = array(4967)
        b = array(3920)
        # 对 a 和 b 进行卷积
        out = oaconvolve(a, b)
        # 断言：卷积结果应该等于 a 乘以 b
        assert_equal(out, a * b)

    # 定义测试函数，测试输入数组为单个元素数组时 oaconvolve 函数的行为
    def test_single_element(self):
        # 创建单个元素数组 a 和 b
        a = array([4967])
        b = array([3920])
        # 对 a 和 b 进行卷积
        out = oaconvolve(a, b)
        # 断言：卷积结果应该等于 a 乘以 b
        assert_equal(out, a * b)
# 定义测试类 TestAllFreqConvolves，用于测试频率卷积相关功能
class TestAllFreqConvolves:

    # 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_invalid_shapes，传入两种不同的卷积函数 fftconvolve 和 oaconvolve
    @pytest.mark.parametrize('convapproach',
                             [fftconvolve, oaconvolve])
    # 定义测试函数 test_invalid_shapes，测试不合法的数组形状情况
    def test_invalid_shapes(self, convapproach):
        # 创建数组 a，形状为 (2, 3)，元素为 1 到 6 的整数
        a = np.arange(1, 7).reshape((2, 3))
        # 创建数组 b，形状为 (3, 2)，元素为 -6 到 -1 的整数
        b = np.arange(-6, 0).reshape((3, 2))
        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError,
                           match="For 'valid' mode, one must be at least "
                           "as large as the other in every dimension"):
            # 调用卷积函数 convapproach，使用 'valid' 模式计算卷积
            convapproach(a, b, mode='valid')

    # 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_invalid_shapes_axes，传入两种不同的卷积函数 fftconvolve 和 oaconvolve
    @pytest.mark.parametrize('convapproach',
                             [fftconvolve, oaconvolve])
    # 定义测试函数 test_invalid_shapes_axes，测试不合法的数组形状和轴参数情况
    def test_invalid_shapes_axes(self, convapproach):
        # 创建数组 a，形状为 [5, 6, 2, 1]，元素全为 0
        a = np.zeros([5, 6, 2, 1])
        # 创建数组 b，形状为 [5, 6, 3, 1]，元素全为 0
        b = np.zeros([5, 6, 3, 1])
        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError,
                           match=r"incompatible shapes for in1 and in2:"
                           r" \(5L?, 6L?, 2L?, 1L?\) and"
                           r" \(5L?, 6L?, 3L?, 1L?\)"):
            # 调用卷积函数 convapproach，指定轴参数为 [0, 1]，计算卷积
            convapproach(a, b, axes=[0, 1])

    # 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_mismatched_dims，传入不匹配维度的数组对 a 和 b，以及两种卷积函数 fftconvolve 和 oaconvolve
    @pytest.mark.parametrize('a,b',
                             [([1], 2),
                              (1, [2]),
                              ([3], [[2]])])
    @pytest.mark.parametrize('convapproach',
                             [fftconvolve, oaconvolve])
    # 定义测试函数 test_mismatched_dims，测试维度不匹配的情况
    def test_mismatched_dims(self, a, b, convapproach):
        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError,
                           match="in1 and in2 should have the same"
                           " dimensionality"):
            # 调用卷积函数 convapproach，计算卷积
            convapproach(a, b)

    # 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_invalid_flags，传入两种卷积函数 fftconvolve 和 oaconvolve
    @pytest.mark.parametrize('convapproach',
                             [fftconvolve, oaconvolve])
    # 定义测试函数 test_invalid_flags，测试不合法的标志参数情况
    def test_invalid_flags(self, convapproach):
        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError,
                           match="acceptable mode flags are 'valid',"
                           " 'same', or 'full'"):
            # 调用卷积函数 convapproach，使用未知的模式 'chips' 计算卷积
            convapproach([1], [2], mode='chips')

        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError,
                           match="when provided, axes cannot be empty"):
            # 调用卷积函数 convapproach，传入空的轴参数 axes=[]
            convapproach([1], [2], axes=[])

        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError, match="axes must be a scalar or "
                           "iterable of integers"):
            # 调用卷积函数 convapproach，传入非法的轴参数 axes=[[1, 2], [3, 4]]
            convapproach([1], [2], axes=[[1, 2], [3, 4]])

        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError, match="axes must be a scalar or "
                           "iterable of integers"):
            # 调用卷积函数 convapproach，传入非法的轴参数 axes=[1., 2., 3., 4.]
            convapproach([1], [2], axes=[1., 2., 3., 4.])

        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError,
                           match="axes exceeds dimensionality of input"):
            # 调用卷积函数 convapproach，传入超出输入维度的轴参数 axes=[1]
            convapproach([1], [2], axes=[1])

        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError,
                           match="axes exceeds dimensionality of input"):
            # 调用卷积函数 convapproach，传入超出输入维度的轴参数 axes=[-2]
            convapproach([1], [2], axes=[-2])

        # 使用 assert_raises 断言异常，期望捕获 ValueError，并匹配特定错误信息字符串
        with assert_raises(ValueError,
                           match="all axes must be unique"):
            # 调用卷积函数 convapproach，传入重复的轴参数 axes=[0, 0]
            convapproach([1], [2], axes=[0, 0])

    # 使用 pytest.mark.parametrize 装饰器，参数化测试函数 test_mismatched_dims，传入两种不同的 dtype，测试长双精度和复长双精度
    @pytest.mark.parametrize('dtype', [np.longdouble, np.clongdouble])
    # 定义一个测试方法，用于测试指定数据类型的输入
    def test_longdtype_input(self, dtype):
        # 生成一个随机数组，形状为 (27, 27)，数据类型为参数指定的 dtype
        x = np.random.random((27, 27)).astype(dtype)
        # 生成另一个随机数组，形状为 (4, 4)，数据类型为参数指定的 dtype
        y = np.random.random((4, 4)).astype(dtype)
        
        # 如果数据类型是复数类型，则对 x 添加虚部 0.1j，对 y 减去虚部 0.1j
        if np.iscomplexobj(dtype()):
            x += .1j
            y -= .1j
        
        # 使用 FFT 进行卷积运算，计算 x 和 y 的卷积结果
        res = fftconvolve(x, y)
        
        # 使用直接方法进行卷积运算，然后使用 assert 检查两种方法的结果是否接近
        assert_allclose(res, convolve(x, y, method='direct'))
        
        # 使用 assert 检查卷积结果的数据类型是否与输入的数据类型一致
        assert res.dtype == dtype
class TestMedFilt:
    # 输入矩阵示例
    IN = [[50, 50, 50, 50, 50, 92, 18, 27, 65, 46],
          [50, 50, 50, 50, 50, 0, 72, 77, 68, 66],
          [50, 50, 50, 50, 50, 46, 47, 19, 64, 77],
          [50, 50, 50, 50, 50, 42, 15, 29, 95, 35],
          [50, 50, 50, 50, 50, 46, 34, 9, 21, 66],
          [70, 97, 28, 68, 78, 77, 61, 58, 71, 42],
          [64, 53, 44, 29, 68, 32, 19, 68, 24, 84],
          [3, 33, 53, 67, 1, 78, 74, 55, 12, 83],
          [7, 11, 46, 70, 60, 47, 24, 43, 61, 26],
          [32, 61, 88, 7, 39, 4, 92, 64, 45, 61]]

    # 预期的中值滤波后输出矩阵示例
    OUT = [[0, 50, 50, 50, 42, 15, 15, 18, 27, 0],
           [0, 50, 50, 50, 50, 42, 19, 21, 29, 0],
           [50, 50, 50, 50, 50, 47, 34, 34, 46, 35],
           [50, 50, 50, 50, 50, 50, 42, 47, 64, 42],
           [50, 50, 50, 50, 50, 50, 46, 55, 64, 35],
           [33, 50, 50, 50, 50, 47, 46, 43, 55, 26],
           [32, 50, 50, 50, 50, 47, 46, 45, 55, 26],
           [7, 46, 50, 50, 47, 46, 46, 43, 45, 21],
           [0, 32, 33, 39, 32, 32, 43, 43, 43, 0],
           [0, 7, 11, 7, 4, 4, 19, 19, 24, 0]]

    # 中值滤波器的核大小
    KERNEL_SIZE = [7,3]

    # 基本测试：验证中值滤波函数的基本功能
    def test_basic(self):
        # 对输入矩阵进行一维中值滤波
        d = signal.medfilt(self.IN, self.KERNEL_SIZE)
        # 对输入矩阵进行二维中值滤波
        e = signal.medfilt2d(np.array(self.IN, float), self.KERNEL_SIZE)
        # 断言：一维滤波结果与预期输出一致
        assert_array_equal(d, self.OUT)
        # 断言：一维滤波结果与二维滤波结果一致
        assert_array_equal(d, e)

    # 参数化测试：验证中值滤波函数对不同数据类型的支持
    @pytest.mark.parametrize('dtype', [np.ubyte, np.byte, np.ushort, np.short,
                                       np_ulong, np_long, np.ulonglong, np.ulonglong,
                                       np.float32, np.float64])
    def test_types(self, dtype):
        # 将输入矩阵转换为指定数据类型
        in_typed = np.array(self.IN, dtype=dtype)
        # 断言：一维滤波后的数据类型符合预期
        assert_equal(signal.medfilt(in_typed).dtype, dtype)
        # 断言：二维滤波后的数据类型符合预期
        assert_equal(signal.medfilt2d(in_typed).dtype, dtype)

    # 参数化测试：验证中值滤波函数对不支持的数据类型的处理
    @pytest.mark.parametrize('dtype', [np.bool_, np.complex64, np.complex128,
                                       np.clongdouble, np.float16, np.object_,
                                       "float96", "float128"])
    def test_invalid_dtypes(self, dtype):
        # 如果平台不支持float96或float128类型，则跳过这些测试
        if (dtype in ["float96", "float128"]
                and np.finfo(np.longdouble).dtype != dtype):
            pytest.skip(f"Platform does not support {dtype}")
        
        # 将输入矩阵转换为指定数据类型
        in_typed = np.array(self.IN, dtype=dtype)
        # 断言：对于不支持的数据类型，调用一维滤波函数会抛出 ValueError 异常
        with pytest.raises(ValueError, match="not supported"):
            signal.medfilt(in_typed)

        # 断言：对于不支持的数据类型，调用二维滤波函数会抛出 ValueError 异常
        with pytest.raises(ValueError, match="not supported"):
            signal.medfilt2d(in_typed)

    # 单元测试：验证中值滤波函数在处理 None 输入时不会发生段错误
    def test_none(self):
        # 检查对于 dtype=object 类型输入，调用中值滤波函数会抛出 ValueError 异常
        msg = "dtype=object is not supported by medfilt"
        with assert_raises(ValueError, match=msg):
            signal.medfilt(None)
    def test_odd_strides(self):
        # 避免因为可能有奇怪步幅的连续 numpy 数组而导致的回归问题。下面的步幅值
        # 如果被使用会将我们带入错误的内存位置（但实际上它不需要被使用）。
        dummy = np.arange(10, dtype=np.float64)
        a = dummy[5:6]  # 从 dummy 数组中切出一个切片 a
        a.strides = 16  # 设置步幅为 16，这是一个错误的步幅设置
        assert_(signal.medfilt(a, 1) == 5.)  # 断言使用 medfilt 函数对 a 进行中值滤波应该得到 5. 的结果

    @pytest.mark.parametrize("dtype", [np.ubyte, np.float32, np.float64])
    def test_medfilt2d_parallel(self, dtype):
        in_typed = np.array(self.IN, dtype=dtype)  # 根据给定的数据类型创建输入数组
        expected = np.array(self.OUT, dtype=dtype)  # 根据给定的数据类型创建期望的输出数组

        # 用于简化索引计算。
        assert in_typed.shape == expected.shape  # 断言输入数组和期望数组具有相同的形状

        # 将计算分为四个块进行。M1 和 N1 是第一个输出块的维度。
        # 我们必须扩展输入的一半核大小以计算完整的输出块。
        M1 = expected.shape[0] // 2
        N1 = expected.shape[1] // 2
        offM = self.KERNEL_SIZE[0] // 2 + 1
        offN = self.KERNEL_SIZE[1] // 2 + 1

        def apply(chunk):
            # in = 要使用的 in_typed 的切片。
            # sel = 要裁剪以匹配正确区域的输出的切片。
            # out = 要存储在输出数组中的切片。
            M, N = chunk
            if M == 0:
                Min = slice(0, M1 + offM)
                Msel = slice(0, -offM)
                Mout = slice(0, M1)
            else:
                Min = slice(M1 - offM, None)
                Msel = slice(offM, None)
                Mout = slice(M1, None)
            if N == 0:
                Nin = slice(0, N1 + offN)
                Nsel = slice(0, -offN)
                Nout = slice(0, N1)
            else:
                Nin = slice(N1 - offN, None)
                Nsel = slice(offN, None)
                Nout = slice(N1, None)

            # 执行计算，但不要在线程中写入输出。
            chunk_data = in_typed[Min, Nin]
            med = signal.medfilt2d(chunk_data, self.KERNEL_SIZE)
            return med[Msel, Nsel], Mout, Nout

        # 将每个块分配给不同的线程。
        output = np.zeros_like(expected)
        with ThreadPoolExecutor(max_workers=4) as pool:
            chunks = {(0, 0), (0, 1), (1, 0), (1, 1)}
            futures = {pool.submit(apply, chunk) for chunk in chunks}

            # 每当结果到达时将其存储在输出中。
            for future in as_completed(futures):
                data, Mslice, Nslice = future.result()
                output[Mslice, Nslice] = data

        assert_array_equal(output, expected)
class TestWiener:

    def test_basic(self):
        # 创建一个二维数组 g，表示输入信号
        g = array([[5, 6, 4, 3],
                   [3, 5, 6, 2],
                   [2, 3, 5, 6],
                   [1, 6, 9, 7]], 'd')
        # 创建一个二维数组 h，表示预期输出信号
        h = array([[2.16374269, 3.2222222222, 2.8888888889, 1.6666666667],
                   [2.666666667, 4.33333333333, 4.44444444444, 2.8888888888],
                   [2.222222222, 4.4444444444, 5.4444444444, 4.801066874837],
                   [1.33333333333, 3.92735042735, 6.0712560386, 5.0404040404]])
        # 使用 wiener 函数对输入信号 g 进行滤波，并断言输出结果与 h 相近
        assert_array_almost_equal(signal.wiener(g), h, decimal=6)
        # 使用 wiener 函数对输入信号 g 进行滤波（指定参数 mysize=3），并断言输出结果与 h 相近
        assert_array_almost_equal(signal.wiener(g, mysize=3), h, decimal=6)


padtype_options = ["mean", "median", "minimum", "maximum", "line"]
# 将 _upfirdn_modes 合并到 padtype_options 列表中
padtype_options += _upfirdn_modes


class TestResample:
    def test_basic(self):
        # 一些基本测试

        # 用于问题 #3603 的回归测试。
        # 窗口的形状必须等于 sig 的行数
        sig = np.arange(128)
        num = 256
        # 获取 Kaiser 窗口，长度为 160
        win = signal.get_window(('kaiser', 8.0), 160)
        # 断言使用该窗口对 sig 进行重采样会引发 ValueError 异常
        assert_raises(ValueError, signal.resample, sig, num, window=win)

        # 其他异常情况的测试
        assert_raises(ValueError, signal.resample_poly, sig, 'yo', 1)
        assert_raises(ValueError, signal.resample_poly, sig, 1, 0)
        assert_raises(ValueError, signal.resample_poly, sig, 2, 1, padtype='')
        assert_raises(ValueError, signal.resample_poly, sig, 2, 1,
                      padtype='mean', cval=10)

        # 问题 #6505 的测试 - 当 axis ≠ 0 时，应不修改窗口的形状
        sig2 = np.tile(np.arange(160), (2, 1))
        signal.resample(sig2, num, axis=-1, window=win)
        assert_(win.shape == (160,))

    @pytest.mark.parametrize('window', (None, 'hamming'))
    @pytest.mark.parametrize('N', (20, 19))
    @pytest.mark.parametrize('num', (100, 101, 10, 11))
    def test_rfft(self, N, num, window):
        # 确保使用 rfft 加速后的结果与正常使用 fft 的结果相同

        # 创建长度为 N 的信号 x
        x = np.linspace(0, 10, N, endpoint=False)
        # 创建信号 y，使用 cos 函数生成
        y = np.cos(-x**2/6.0)
        # 使用 resample 函数对信号 y 进行重采样，指定窗口函数为 window
        assert_allclose(signal.resample(y, num, window=window),
                        signal.resample(y + 0j, num, window=window).real)

        # 创建一个包含两个信号的数组 y
        y = np.array([np.cos(-x**2/6.0), np.sin(-x**2/6.0)])
        y_complex = y + 0j
        # 对信号 y 进行重采样，指定 axis=1 和窗口函数为 window
        assert_allclose(
            signal.resample(y, num, axis=1, window=window),
            signal.resample(y_complex, num, axis=1, window=window).real,
            atol=1e-9)

    def test_input_domain(self):
        # 测试频域和时域两种输入域模式产生相同结果

        tsig = np.arange(256) + 0j
        fsig = fft(tsig)
        num = 256
        # 断言使用频域输入和时域输入两种方式对信号进行重采样结果相近
        assert_allclose(
            signal.resample(fsig, num, domain='freq'),
            signal.resample(tsig, num, domain='time'),
            atol=1e-9)

    @pytest.mark.parametrize('nx', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('ny', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('dtype', ('float', 'complex'))
    def test_dc(self, nx, ny, dtype):
        # 创建一个长度为 nx 的数组 x，元素全为 1，指定数据类型为 dtype
        x = np.array([1] * nx, dtype)
        # 使用 signal 模块中的 resample 函数对数组 x 进行重采样，目标长度为 ny
        y = signal.resample(x, ny)
        # 断言重采样后的数组 y 的值与元素全为 1、长度为 ny 的数组一致
        assert_allclose(y, [1] * ny)

    @pytest.mark.parametrize('padtype', padtype_options)
    def test_mutable_window(self, padtype):
        # 测试可变窗口未被修改
        impulse = np.zeros(3)
        # 创建一个长度为 2 的随机数数组 window，保存其副本至 window_orig
        window = np.random.RandomState(0).randn(2)
        window_orig = window.copy()
        # 使用 signal 模块中的 resample_poly 函数进行多项式重采样，指定窗口为 window，填充类型为 padtype
        signal.resample_poly(impulse, 5, 1, window=window, padtype=padtype)
        # 断言重采样后的窗口数组 window 与其副本 window_orig 相等
        assert_array_equal(window, window_orig)

    @pytest.mark.parametrize('padtype', padtype_options)
    def test_output_float32(self, padtype):
        # 测试 float32 类型的输入能得到 float32 类型的输出
        x = np.arange(10, dtype=np.float32)
        h = np.array([1, 1, 1], dtype=np.float32)
        # 使用 signal 模块中的 resample_poly 函数对数组 x 进行多项式重采样，输出数据类型保持为 float32，窗口为 h，填充类型为 padtype
        y = signal.resample_poly(x, 1, 2, window=h, padtype=padtype)
        # 断言重采样后的数组 y 的数据类型为 np.float32
        assert y.dtype == np.float32

    @pytest.mark.parametrize('padtype', padtype_options)
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_output_match_dtype(self, padtype, dtype):
        # 测试 x 数组的数据类型在重采样后得到保持不变，修复问题 #14733
        x = np.arange(10, dtype=dtype)
        # 使用 signal 模块中的 resample_poly 函数对数组 x 进行多项式重采样，填充类型为 padtype
        y = signal.resample_poly(x, 1, 2, padtype=padtype)
        # 断言重采样后的数组 y 的数据类型与原始数组 x 的数据类型相同
        assert y.dtype == x.dtype

    @pytest.mark.parametrize(
        "method, ext, padtype",
        [("fft", False, None)]
        + list(
            product(
                ["polyphase"], [False, True], padtype_options,
            )
        ),
    )
    def test_poly_vs_filtfilt(self):
        # 检查当 up=1.0 时，resample_poly 是否能得到与 filtfilt + 切片操作相同的结果
        random_state = np.random.RandomState(17)
        try_types = (int, np.float32, np.complex64, float, complex)
        size = 10000
        down_factors = [2, 11, 79]

        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * random_state.randn(size)

            # resample_poly 假设信号外部为零，而 filtfilt 只能使用常数填充。使它们等价：
            x[0] = 0
            x[-1] = 0

            for down in down_factors:
                # 使用 signal 模块中的 firwin 函数生成长度为 31 的低通滤波器 h
                h = signal.firwin(31, 1. / down, window='hamming')
                # 使用 filtfilt 函数进行前后滤波，得到 yf，然后每 down 个样本取一个
                yf = filtfilt(h, 1.0, x, padtype='constant')[::down]

                # 需要将卷积版本的滤波器传递给 resample_poly，因为 filtfilt 做前向和后向滤波，而 resample_poly 只进行前向
                hc = convolve(h, h[::-1])
                # 使用 signal 模块中的 resample_poly 函数对数组 x 进行多项式重采样，上采样因子为 1，下采样因子为 down，窗口为 hc
                y = signal.resample_poly(x, 1, down, window=hc)
                # 断言重采样后的数组 yf 与 y 在给定的容差范围内（atol 和 rtol）相等
                assert_allclose(yf, y, atol=1e-7, rtol=1e-7)
    # 定义名为 test_correlate1d 的测试方法，用于测试 correlate1d 函数的功能
    def test_correlate1d(self):
        # 外层循环，遍历 down 取值为 2 和 4
        for down in [2, 4]:
            # 中层循环，从 1 到 39，步长为 down 的范围，作为 nx 的取值
            for nx in range(1, 40, down):
                # 内层循环，遍历 nweights 取值为 32 和 33
                for nweights in (32, 33):
                    # 生成长度为 nx 的随机数组 x
                    x = np.random.random((nx,))
                    # 生成长度为 nweights 的随机数组 weights
                    weights = np.random.random((nweights,))
                    # 使用 correlate1d 函数计算 x 和 weights 的卷积，结果存入 y_g
                    y_g = correlate1d(x, weights[::-1], mode='constant')
                    # 使用 signal.resample_poly 函数对 x 进行重采样，结果存入 y_s
                    y_s = signal.resample_poly(
                        x, up=1, down=down, window=weights)
                    # 使用 assert_allclose 断言函数检查 y_g 和 y_s 的近似程度
                    assert_allclose(y_g[::down], y_s)
class TestCSpline1DEval:

    def test_basic(self):
        # 定义一个简单的数值数组作为测试数据
        y = array([1, 2, 3, 4, 3, 2, 1, 2, 3.0])
        # 生成与 y 等长的索引数组
        x = arange(len(y))
        # 计算相邻索引之间的距离
        dx = x[1] - x[0]
        # 对 y 应用 1D Cubic Spline 插值
        cj = signal.cspline1d(y)

        # 生成更密集的 x2 数组用于插值
        x2 = arange(len(y) * 10.0) / 10.0
        # 使用 cspline1d_eval 函数对已计算的 cubic spline 进行评估
        y2 = signal.cspline1d_eval(cj, x2, dx=dx, x0=x[0])

        # 确保插值后的值位于节点上
        assert_array_almost_equal(y2[::10], y, decimal=5)

    def test_complex(self):
        # 创建一个平滑变化的复杂信号用于插值
        x = np.arange(2)
        y = np.zeros(x.shape, dtype=np.complex64)
        T = 10.0
        f = 1.0 / T
        y = np.exp(2.0J * np.pi * f * x)

        # 计算复杂信号的 cubic spline 变换
        cy = signal.cspline1d(y)

        # 确定新的测试 x 值并进行插值
        xnew = np.array([0.5])
        ynew = signal.cspline1d_eval(cy, xnew)

        # 确保插值后的数据类型与原始信号一致
        assert_equal(ynew.dtype, y.dtype)


class TestOrderFilt:

    def test_basic(self):
        # 测试 order_filter 函数基本功能
        assert_array_equal(signal.order_filter([1, 2, 3], [1, 0, 1], 1),
                           [2, 3, 2])


class _TestLinearFilter:

    def generate(self, shape):
        # 生成一个按顺序排列的二维数组
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        return self.convert_dtype(x)

    def convert_dtype(self, arr):
        # 将输入数组 arr 转换为指定数据类型
        if self.dtype == np.dtype('O'):
            arr = np.asarray(arr)
            out = np.empty(arr.shape, self.dtype)
            iter = np.nditer([arr, out], ['refs_ok','zerosize_ok'],
                        [['readonly'],['writeonly']])
            for x, y in iter:
                y[...] = self.type(x[()])
            return out
        else:
            return np.asarray(arr, dtype=self.dtype)

    def test_rank_1_IIR(self):
        # 生成一个线性滤波器的测试数据并测试 IIR 滤波器
        x = self.generate((6,))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, -0.5])
        y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.])
        assert_array_almost_equal(lfilter(b, a, x), y_r)

    def test_rank_1_FIR(self):
        # 生成一个线性滤波器的测试数据并测试 FIR 滤波器
        x = self.generate((6,))
        b = self.convert_dtype([1, 1])
        a = self.convert_dtype([1])
        y_r = self.convert_dtype([0, 1, 3, 5, 7, 9.])
        assert_array_almost_equal(lfilter(b, a, x), y_r)

    def test_rank_1_IIR_init_cond(self):
        # 生成一个线性滤波器的测试数据并测试带有初始条件的 IIR 滤波器
        x = self.generate((6,))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([0.5, -0.5])
        zi = self.convert_dtype([1, 2])
        y_r = self.convert_dtype([1, 5, 9, 13, 17, 21])
        zf_r = self.convert_dtype([13, -10])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, y_r)
        assert_array_almost_equal(zf, zf_r)
    # 定义测试函数，用于测试一阶有限脉冲响应（FIR）滤波器的初始化条件
    def test_rank_1_FIR_init_cond(self):
        # 生成长度为6的随机输入信号
        x = self.generate((6,))
        # 将系数向量b转换为指定数据类型
        b = self.convert_dtype([1, 1, 1])
        # 将系数向量a转换为指定数据类型，这里是一个一阶FIR滤波器，所以只有一个元素1
        a = self.convert_dtype([1])
        # 设置初始状态向量zi，将其转换为指定数据类型
        zi = self.convert_dtype([1, 1])
        # 预期输出的信号y_r，用于验证滤波器输出结果的准确性
        y_r = self.convert_dtype([1, 2, 3, 6, 9, 12.])
        # 预期的最终状态zf_r，用于验证滤波器最终状态的准确性
        zf_r = self.convert_dtype([9, 5])
        # 使用lfilter函数进行滤波计算，得到输出信号y和最终状态zf
        y, zf = lfilter(b, a, x, zi=zi)
        # 断言输出信号y与预期结果y_r几乎相等
        assert_array_almost_equal(y, y_r)
        # 断言最终状态zf与预期结果zf_r几乎相等
        assert_array_almost_equal(zf, zf_r)

    # 定义测试函数，用于测试二阶无穷脉冲响应（IIR）滤波器在axis=0上的操作
    def test_rank_2_IIR_axis_0(self):
        # 生成形状为(4, 3)的随机输入信号
        x = self.generate((4, 3))
        # 将系数向量b转换为指定数据类型
        b = self.convert_dtype([1, -1])
        # 将系数向量a转换为指定数据类型
        a = self.convert_dtype([0.5, 0.5])
        # 预期输出的信号y_r2_a0，用于验证滤波器输出结果的准确性
        y_r2_a0 = self.convert_dtype([[0, 2, 4], [6, 4, 2], [0, 2, 4],
                                      [6, 4, 2]])
        # 使用lfilter函数进行滤波计算，指定在axis=0上操作，得到输出信号y
        y = lfilter(b, a, x, axis=0)
        # 断言输出信号y与预期结果y_r2_a0几乎相等
        assert_array_almost_equal(y_r2_a0, y)

    # 定义测试函数，用于测试二阶无穷脉冲响应（IIR）滤波器在axis=1上的操作
    def test_rank_2_IIR_axis_1(self):
        # 生成形状为(4, 3)的随机输入信号
        x = self.generate((4, 3))
        # 将系数向量b转换为指定数据类型
        b = self.convert_dtype([1, -1])
        # 将系数向量a转换为指定数据类型
        a = self.convert_dtype([0.5, 0.5])
        # 预期输出的信号y_r2_a1，用于验证滤波器输出结果的准确性
        y_r2_a1 = self.convert_dtype([[0, 2, 0], [6, -4, 6], [12, -10, 12],
                            [18, -16, 18]])
        # 使用lfilter函数进行滤波计算，指定在axis=1上操作，得到输出信号y
        y = lfilter(b, a, x, axis=1)
        # 断言输出信号y与预期结果y_r2_a1几乎相等
        assert_array_almost_equal(y_r2_a1, y)

    # 定义测试函数，用于测试二阶无穷脉冲响应（IIR）滤波器在axis=0上，且带有初始条件的操作
    def test_rank_2_IIR_axis_0_init_cond(self):
        # 生成形状为(4, 3)的随机输入信号
        x = self.generate((4, 3))
        # 将系数向量b转换为指定数据类型
        b = self.convert_dtype([1, -1])
        # 将系数向量a转换为指定数据类型
        a = self.convert_dtype([0.5, 0.5])
        # 设置初始状态向量zi，将其转换为指定数据类型，这里是一个列向量
        zi = self.convert_dtype(np.ones((4,1)))
        # 预期输出的信号y_r2_a0_1，用于验证滤波器输出结果的准确性
        y_r2_a0_1 = self.convert_dtype([[1, 1, 1], [7, -5, 7], [13, -11, 13],
                              [19, -17, 19]])
        # 预期的最终状态zf_r，用于验证滤波器最终状态的准确性，这里是一个列向量
        zf_r = self.convert_dtype([-5, -17, -29, -41])[:, np.newaxis]
        # 使用lfilter函数进行滤波计算，指定在axis=1上操作和初始状态zi，得到输出信号y和最终状态zf
        y, zf = lfilter(b, a, x, axis=1, zi=zi)
        # 断言输出信号y与预期结果y_r2_a0_1几乎相等
        assert_array_almost_equal(y_r2_a0_1, y)
        # 断言最终状态zf与预期结果zf_r几乎相等
        assert_array_almost_equal(zf, zf_r)

    # 定义测试函数，用于测试二阶无穷脉冲响应（IIR）滤波器在axis=1上，且带有初始条件的操作
    def test_rank_2_IIR_axis_1_init_cond(self):
        # 生成形状为(4, 3)的随机输入信号
        x = self.generate((4,3))
        # 将系数向量b转换为指定数据类型
        b = self.convert_dtype([1, -1])
        # 将系数向量a转换为指定数据类型
        a = self.convert_dtype([0.5, 0.5])
        # 设置初始状态向量zi，将其转换为指定数据类型，这里是一个行向量
        zi = self.convert_dtype(np.ones((1,3)))
        # 预期输出的信号y_r2_a0_0，用于验证滤波器输出结果的准确性
        y_r2_a0_0 = self.convert_dtype([[1, 3, 5], [5, 3, 1],
                                        [1, 3, 5], [5, 3, 1]])
        # 预期的最终状态zf_r，用于验证滤波器最终状态的准确性，这里是一个行向量
        zf_r = self.convert_dtype([[-23, -23, -23]])
        # 使用lfilter函数进行滤波计算，指定在axis=0上操作和初始状态zi，得到输出信号y和最终状态zf
        y, zf = lfilter(b, a, x, axis=0, zi=zi)
        # 断言输出信号y与预期结果y_r2_a0_0几乎相等
        assert_array_almost_equal(y_r2_a0_0, y)
        # 断言最终状态zf与预期结果zf_r几乎相等
        assert_array_almost_equal(zf, zf_r)

    # 定义测试函数，用于测试三阶无穷脉冲响应（IIR）滤波器的操作
    def test_rank_3_IIR(self):
        # 生成形状为(4, 3
    # 测试初始化条件的三阶IIR滤波器
    def test_rank_3_IIR_init_cond(self):
        # 生成一个形状为(4, 3, 2)的测试数据
        x = self.generate((4, 3, 2))
        # 将系数向量 [1, -1] 转换为适当的数据类型
        b = self.convert_dtype([1, -1])
        # 将系数向量 [0.5, 0.5] 转换为适当的数据类型
        a = self.convert_dtype([0.5, 0.5])

        # 对数据的每一个维度执行以下操作
        for axis in range(x.ndim):
            # 复制当前数据形状，并将当前轴的长度设为1
            zi_shape = list(x.shape)
            zi_shape[axis] = 1
            # 创建一个全为1的数组作为初始条件 zi
            zi = self.convert_dtype(np.ones(zi_shape))
            # 创建一个包含单个元素 [1] 的数组作为初始条件 zi1
            zi1 = self.convert_dtype([1])
            # 调用 lfilter 函数进行滤波，返回滤波后的输出 y 和最终状态 zf
            y, zf = lfilter(b, a, x, axis, zi)
            
            # 定义一个函数 lf0，用于在当前轴上应用 lfilter 并返回第一个输出
            def lf0(w):
                return lfilter(b, a, w, zi=zi1)[0]
            
            # 定义一个函数 lf1，用于在当前轴上应用 lfilter 并返回第二个输出
            def lf1(w):
                return lfilter(b, a, w, zi=zi1)[1]
            
            # 使用 np.apply_along_axis 在当前轴上应用 lf0 函数，得到参考输出 y_r 和最终状态 zf_r
            y_r = np.apply_along_axis(lf0, axis, x)
            zf_r = np.apply_along_axis(lf1, axis, x)
            
            # 断言滤波后的输出 y 和参考输出 y_r 的准确性
            assert_array_almost_equal(y, y_r)
            # 断言最终状态 zf 和参考最终状态 zf_r 的准确性
            assert_array_almost_equal(zf, zf_r)

    # 测试三阶FIR滤波器
    def test_rank_3_FIR(self):
        # 生成一个形状为(4, 3, 2)的测试数据
        x = self.generate((4, 3, 2))
        # 将系数向量 [1, 0, -1] 转换为适当的数据类型
        b = self.convert_dtype([1, 0, -1])
        # 将系数向量 [1] 转换为适当的数据类型
        a = self.convert_dtype([1])

        # 对数据的每一个维度执行以下操作
        for axis in range(x.ndim):
            # 调用 lfilter 函数进行滤波，返回滤波后的输出 y
            y = lfilter(b, a, x, axis)
            # 使用 np.apply_along_axis 在当前轴上应用 lambda 函数，得到参考输出 y_r
            y_r = np.apply_along_axis(lambda w: lfilter(b, a, w), axis, x)
            # 断言滤波后的输出 y 和参考输出 y_r 的准确性
            assert_array_almost_equal(y, y_r)

    # 测试带初始化条件的三阶FIR滤波器
    def test_rank_3_FIR_init_cond(self):
        # 生成一个形状为(4, 3, 2)的测试数据
        x = self.generate((4, 3, 2))
        # 将系数向量 [1, 0, -1] 转换为适当的数据类型
        b = self.convert_dtype([1, 0, -1])
        # 将系数向量 [1] 转换为适当的数据类型
        a = self.convert_dtype([1])

        # 对数据的每一个维度执行以下操作
        for axis in range(x.ndim):
            # 复制当前数据形状，并将当前轴的长度设为2
            zi_shape = list(x.shape)
            zi_shape[axis] = 2
            # 创建一个全为1的数组作为初始条件 zi
            zi = self.convert_dtype(np.ones(zi_shape))
            # 创建一个包含两个元素 [1, 1] 的数组作为初始条件 zi1
            zi1 = self.convert_dtype([1, 1])
            # 调用 lfilter 函数进行滤波，返回滤波后的输出 y 和最终状态 zf
            y, zf = lfilter(b, a, x, axis, zi)
            
            # 定义一个函数 lf0，用于在当前轴上应用 lfilter 并返回第一个输出
            def lf0(w):
                return lfilter(b, a, w, zi=zi1)[0]
            
            # 定义一个函数 lf1，用于在当前轴上应用 lfilter 并返回第二个输出
            def lf1(w):
                return lfilter(b, a, w, zi=zi1)[1]
            
            # 使用 np.apply_along_axis 在当前轴上应用 lf0 函数，得到参考输出 y_r 和最终状态 zf_r
            y_r = np.apply_along_axis(lf0, axis, x)
            zf_r = np.apply_along_axis(lf1, axis, x)
            
            # 断言滤波后的输出 y 和参考输出 y_r 的准确性
            assert_array_almost_equal(y, y_r)
            # 断言最终状态 zf 和参考最终状态 zf_r 的准确性
            assert_array_almost_equal(zf, zf_r)

    # 测试 lfilter 函数在伪广播情况下的表现
    def test_zi_pseudobroadcast(self):
        # 生成一个形状为(4, 5, 20)的测试数据
        x = self.generate((4, 5, 20))
        # 使用 signal.butter 生成一对滤波器的系数向量 b 和 a
        b, a = signal.butter(8, 0.2, output='ba')
        # 将系数向量 b 转换为适当的数据类型
        b = self.convert_dtype(b)
        # 将系数向量 a 转换为适当的数据类型
        a = self.convert_dtype(a)
        # 计算初始条件 zi 的大小，即 b 的长度减1
        zi_size = b.shape[0] - 1

        # 创建一个形状为(4, 5, zi_size)的全为1的数组作为完整初始化条件 zi_full
        zi_full = self.convert_dtype(np.ones((4, 5, zi_size)))
        # 创建一个形状为(1, 1, zi_size)的全为1的数组作为单一初始化条件 zi_sing
        zi_sing = self.convert_dtype(np.ones((1, 1, zi_size)))

        # 使用完整初始化条件 zi_full 调用 lfilter 函数进行滤波，返回滤波后的输出 y_full 和最终状态 zf_full
        y_full, zf_full = lfilter(b, a, x, zi=zi_full)
        # 使用单一初始化条件 zi_sing 调用 lfilter 函数进行滤波，返回滤波后的输出 y_sing 和最终状态 zf_sing
        y_sing, zf_sing = lfilter(b, a, x, zi=zi_sing)

        # 断言使用单一初始化条件和完整初始化条件的输出 y_sing 和 y_full 的准确性
        assert_array_almost_equal(y_sing, y_full)
        # 断言使用单一初始化条件和完整初始化条件的最终状态 zf_sing 和 zf_full 的准确性
        assert_array_almost_equal(zf_full, zf_sing)

        # 断言 lfilter 函数在不合适的初始化条件下会引发 ValueError 异常
        assert_raises(ValueError, lfilter, b, a, x, -1, np.ones(zi_size))

    # 测试 lfilter 函数中系数 a 可以是标量的情况
    def test_scalar_a(self):
        # 生成一个长度为6的测试数据
        x = self.generate(6)
        # 将系数向量 [
    # 测试 lfilter 函数在特定情况下的行为：不会广播 (不会在前面填充1)，但如果 x 和 zi 的维度相同，则会进行单例扩展。
    def test_zi_some_singleton_dims(self):
        # 创建一个形状为 (3,2,5) 的全零数组，并转换为 'l' 类型
        x = self.convert_dtype(np.zeros((3,2,5), 'l'))
        # 创建一个形状为 (5,) 的全一数组，并转换为 'l' 类型
        b = self.convert_dtype(np.ones(5, 'l'))
        # 创建一个包含 [1, 0, 0] 的数组，并转换为 'l' 类型
        a = self.convert_dtype(np.array([1,0,0]))
        # 创建一个形状为 (3,1,4) 的全一数组，并转换为 'l' 类型
        zi = np.ones((3,1,4), 'l')
        # 修改 zi 的第二个维度，使其值乘以 2
        zi[1,:,:] *= 2
        # 修改 zi 的第三个维度，使其值乘以 3
        zi[2,:,:] *= 3
        # 将修改后的 zi 转换为 'l' 类型
        zi = self.convert_dtype(zi)

        # 创建一个形状为 (3,2,4) 的全零数组，并转换为 'l' 类型，作为预期的 zf
        zf_expected = self.convert_dtype(np.zeros((3,2,4), 'l'))
        # 创建一个形状为 (3,2,5) 的全零数组，作为预期的 y
        y_expected = np.zeros((3,2,5), 'l')
        # 将 y_expected 的前三个维度的前四个元素赋值为 [[[1]], [[2]], [[3]]]
        y_expected[:,:,:4] = [[[1]], [[2]], [[3]]]
        # 将 y_expected 转换为 'l' 类型
        y_expected = self.convert_dtype(y_expected)

        # 使用 lfilter 函数计算 IIR 滤波的结果 y_iir 和最终状态 zf_iir
        y_iir, zf_iir = lfilter(b, a, x, -1, zi)
        # 断言 y_iir 的值接近于预期的 y_expected
        assert_array_almost_equal(y_iir, y_expected)
        # 断言 zf_iir 的值接近于预期的 zf_expected
        assert_array_almost_equal(zf_iir, zf_expected)

        # 使用 lfilter 函数计算 FIR 滤波的结果 y_fir 和最终状态 zf_fir
        y_fir, zf_fir = lfilter(b, a[0], x, -1, zi)
        # 断言 y_fir 的值接近于预期的 y_expected
        assert_array_almost_equal(y_fir, y_expected)
        # 断言 zf_fir 的值接近于预期的 zf_expected
        assert_array_almost_equal(zf_fir, zf_expected)

    # 检查当输入的 b, a, x, axis, zi 参数大小不合适时，是否会引发 ValueError 异常
    def base_bad_size_zi(self, b, a, x, axis, zi):
        # 将输入的 b 转换为指定类型
        b = self.convert_dtype(b)
        # 将输入的 a 转换为指定类型
        a = self.convert_dtype(a)
        # 将输入的 x 转换为指定类型
        x = self.convert_dtype(x)
        # 将输入的 zi 转换为指定类型
        zi = self.convert_dtype(zi)
        # 断言调用 lfilter 函数时，使用不合适大小的 zi 参数会引发 ValueError 异常
        assert_raises(ValueError, lfilter, b, a, x, axis, zi)

    # 对空数组 zi 进行回归测试，验证是否能正确处理空数组输入的情况
    def test_empty_zi(self):
        # 生成一个形状为 (5,) 的数组 x
        x = self.generate((5,))
        # 创建一个包含 [1] 的数组 a，并转换为指定类型
        a = self.convert_dtype([1])
        # 创建一个包含 [1] 的数组 b，并转换为指定类型
        b = self.convert_dtype([1])
        # 创建一个空数组作为 zi，并转换为指定类型
        zi = self.convert_dtype([])
        # 使用 lfilter 函数计算结果 y 和最终状态 zf，传入空数组作为 zi
        y, zf = lfilter(b, a, x, zi=zi)
        # 断言 y 的值接近于 x
        assert_array_almost_equal(y, x)
        # 断言 zf 的数据类型与预期的 self.dtype 相同
        assert_equal(zf.dtype, self.dtype)
        # 断言 zf 的大小为 0
        assert_equal(zf.size, 0)

    # 对 lfiltic 函数处理不良初始条件的回归测试，验证是否能正确处理不良初始条件
    def test_lfiltic_bad_zi(self):
        # 创建一个包含 [1] 的数组 a，并转换为指定类型
        a = self.convert_dtype([1])
        # 创建一个包含 [1] 的数组 b，并转换为指定类型
        b = self.convert_dtype([1])
        # 使用 lfiltic 函数计算不同初始条件下的 zi
        zi = lfiltic(b, a, [1., 0])
        zi_1 = lfiltic(b, a, [1, 0])
        zi_2 = lfiltic(b, a, [True, False])
        # 断言不同条件下得到的 zi 相等
        assert_array_equal(zi, zi_1)
        assert_array_equal(zi, zi_2)

    # 对于 x 短于 b 的情况下进行 FIR 滤波的回归测试，验证是否能正确处理这种情况
    def test_short_x_FIR(self):
        # 创建一个包含 [1] 的数组 a，并转换为指定类型
        a = self.convert_dtype([1])
        # 创建一个包含 [1, 0, -1] 的数组 b，并转换为指定类型
        b = self.convert_dtype([1, 0, -1])
        # 创建一个包含 [2, 7] 的数组 zi，并转换为指定类型
        zi = self.convert_dtype([2, 7])
        # 创建一个包含 [72] 的数组 x，并转换为指定类型
        x = self.convert_dtype([72])
        # 创建一个包含 [74] 的数组 ye，并转换为指定类型
        ye = self.convert_dtype([74])
        # 创建一个包含 [7, -72] 的数组 zfe，并转换为指定类型
        zfe = self.convert_dtype([7, -72])
        # 使用 lfilter 函数计算结果 y 和最终状态 zf，传入非空 zi
        y, zf = lfilter(b, a, x, zi=zi)
        # 断言 y 的值接近于 ye
        assert_array_almost_equal(y, ye)
        # 断言 zf 的值接近于 zfe
        assert_array_almost_equal(zf, zfe)
    # 测试短信号与 IIR 滤波器的回归测试
    def test_short_x_IIR(self):
        # 根据 issue #5116 进行回归测试
        # 当 x 比 b 短，但 zi 不为 None 时会失败
        # 初始化滤波器系数 a, b，以及初始状态 zi 和输入信号 x
        a = self.convert_dtype([1, 1])
        b = self.convert_dtype([1, 0, -1])
        zi = self.convert_dtype([2, 7])
        x = self.convert_dtype([72])
        # 预期输出的信号 y 和最终状态 zf
        ye = self.convert_dtype([74])
        zfe = self.convert_dtype([-67, -72])
        # 应用 lfilter 函数进行滤波
        y, zf = lfilter(b, a, x, zi=zi)
        # 断言输出信号 y 和预期信号 ye 相近
        assert_array_almost_equal(y, ye)
        # 断言最终状态 zf 和预期状态 zfe 相等
        assert_array_almost_equal(zf, zfe)

    # 测试不修改 a 和 b 的 IIR 滤波器
    def test_do_not_modify_a_b_IIR(self):
        # 生成长度为 6 的输入信号 x
        x = self.generate((6,))
        # 初始化滤波器系数 b，保存初始值 b0
        b = self.convert_dtype([1, -1])
        b0 = b.copy()
        # 初始化滤波器系数 a，保存初始值 a0
        a = self.convert_dtype([0.5, -0.5])
        a0 = a.copy()
        # 预期输出的信号 y_r
        y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.])
        # 应用 lfilter 函数进行滤波
        y_f = lfilter(b, a, x)
        # 断言输出信号 y_f 和预期信号 y_r 相近
        assert_array_almost_equal(y_f, y_r)
        # 断言滤波器系数 b 没有被修改
        assert_equal(b, b0)
        # 断言滤波器系数 a 没有被修改
        assert_equal(a, a0)

    # 测试不修改 a 和 b 的 FIR 滤波器
    def test_do_not_modify_a_b_FIR(self):
        # 生成长度为 6 的输入信号 x
        x = self.generate((6,))
        # 初始化滤波器系数 b，保存初始值 b0
        b = self.convert_dtype([1, 0, 1])
        b0 = b.copy()
        # 初始化滤波器系数 a，保存初始值 a0
        a = self.convert_dtype([2])
        a0 = a.copy()
        # 预期输出的信号 y_r
        y_r = self.convert_dtype([0, 0.5, 1, 2, 3, 4.])
        # 应用 lfilter 函数进行滤波
        y_f = lfilter(b, a, x)
        # 断言输出信号 y_f 和预期信号 y_r 相近
        assert_array_almost_equal(y_f, y_r)
        # 断言滤波器系数 b 没有被修改
        assert_equal(b, b0)
        # 断言滤波器系数 a 没有被修改
        assert_equal(a, a0)

    # 使用标量输入参数进行测试
    @pytest.mark.parametrize("a", [1.0, [1.0], np.array(1.0)])
    @pytest.mark.parametrize("b", [1.0, [1.0], np.array(1.0)])
    def test_scalar_input(self, a, b):
        # 生成长度为 10 的随机数据
        data = np.random.randn(10)
        # 断言使用标准滤波器和用户提供的系数 a, b 的输出近似相等
        assert_allclose(
            lfilter(np.array([1.0]), np.array([1.0]), data),
            lfilter(b, a, data))
class TestLinearFilterFloat32(_TestLinearFilter):
    # 定义测试类 TestLinearFilterFloat32，继承自 _TestLinearFilter
    dtype = np.dtype('f')  # 设置 dtype 为 32 位浮点数类型


class TestLinearFilterFloat64(_TestLinearFilter):
    # 定义测试类 TestLinearFilterFloat64，继承自 _TestLinearFilter
    dtype = np.dtype('d')  # 设置 dtype 为 64 位浮点数类型


class TestLinearFilterFloatExtended(_TestLinearFilter):
    # 定义测试类 TestLinearFilterFloatExtended，继承自 _TestLinearFilter
    dtype = np.dtype('g')  # 设置 dtype 为扩展精度浮点数类型


class TestLinearFilterComplex64(_TestLinearFilter):
    # 定义测试类 TestLinearFilterComplex64，继承自 _TestLinearFilter
    dtype = np.dtype('F')  # 设置 dtype 为 64 位复数类型


class TestLinearFilterComplex128(_TestLinearFilter):
    # 定义测试类 TestLinearFilterComplex128，继承自 _TestLinearFilter
    dtype = np.dtype('D')  # 设置 dtype 为 128 位复数类型


class TestLinearFilterComplexExtended(_TestLinearFilter):
    # 定义测试类 TestLinearFilterComplexExtended，继承自 _TestLinearFilter
    dtype = np.dtype('G')  # 设置 dtype 为扩展精度复数类型


class TestLinearFilterDecimal(_TestLinearFilter):
    # 定义测试类 TestLinearFilterDecimal，继承自 _TestLinearFilter
    dtype = np.dtype('O')  # 设置 dtype 为对象类型

    def type(self, x):
        # 返回 x 的十进制表示，作为 Decimal 类型
        return Decimal(str(x))


class TestLinearFilterObject(_TestLinearFilter):
    # 定义测试类 TestLinearFilterObject，继承自 _TestLinearFilter
    dtype = np.dtype('O')  # 设置 dtype 为对象类型
    type = float  # 定义 type 为浮点数类型


def test_lfilter_bad_object():
    # 测试 lfilter：包含非数值对象的对象数组会引发 TypeError。
    # 用于检验 GitHub 问题票号 #1452。
    if hasattr(sys, 'abiflags') and 'd' in sys.abiflags:
        pytest.skip('test is flaky when run with python3-dbg')  # 在使用 python3-dbg 运行时跳过测试
    assert_raises(TypeError, lfilter, [1.0], [1.0], [1.0, None, 2.0])  # 断言会引发 TypeError
    assert_raises(TypeError, lfilter, [1.0], [None], [1.0, 2.0, 3.0])  # 断言会引发 TypeError
    assert_raises(TypeError, lfilter, [None], [1.0], [1.0, 2.0, 3.0])  # 断言会引发 TypeError


def test_lfilter_notimplemented_input():
    # 测试 lfilter：不应崩溃，用于 GitHub 问题 #7991
    assert_raises(NotImplementedError, lfilter, [2,3], [4,5], [1,2,3,4,5])


@pytest.mark.parametrize('dt', [np.ubyte, np.byte, np.ushort, np.short,
                                np_ulong, np_long, np.ulonglong, np.ulonglong,
                                np.float32, np.float64, np.longdouble,
                                Decimal])
class TestCorrelateReal:
    def _setup_rank1(self, dt):
        # 设置数组 a 和 b，类型为 dt
        a = np.linspace(0, 3, 4).astype(dt)
        b = np.linspace(1, 2, 2).astype(dt)

        # 设置 y_r 数组，类型为 dt
        y_r = np.array([0, 2, 5, 8, 3]).astype(dt)
        return a, b, y_r

    def equal_tolerance(self, res_dt):
        # 默认的关键字值
        decimal = 6
        try:
            dt_info = np.finfo(res_dt)
            if hasattr(dt_info, 'resolution'):
                decimal = int(-0.5*np.log10(dt_info.resolution))
        except Exception:
            pass
        return decimal

    def equal_tolerance_fft(self, res_dt):
        # FFT 实现会将 longdouble 类型参数转换为 double 类型，因此不期望有更高的精度，参见 GitHub 问题 #9520
        if res_dt == np.longdouble:
            return self.equal_tolerance(np.float64)
        else:
            return self.equal_tolerance(res_dt)
    # 定义一个测试方法，接受参数 self 和 dt
    def test_method(self, dt):
        # 如果 dt 的类型为 Decimal
        if dt == Decimal:
            # 调用 choose_conv_method 函数，传入 Decimal 类型的参数，期望返回 'direct'
            method = choose_conv_method([Decimal(4)], [Decimal(3)])
            # 断言方法返回值与 'direct' 相等
            assert_equal(method, 'direct')
        else:
            # 否则，调用 _setup_rank3 方法，获取返回的 a, b, y_r
            a, b, y_r = self._setup_rank3(dt)
            # 使用 correlate 函数计算 a 和 b 的相关性，方法为 'fft'，结果保存在 y_fft 中
            y_fft = correlate(a, b, method='fft')
            # 使用 correlate 函数计算 a 和 b 的相关性，方法为 'direct'，结果保存在 y_direct 中
            y_direct = correlate(a, b, method='direct')

            # 断言 y_r 和 y_fft 数组几乎相等，使用 self.equal_tolerance_fft(y_fft.dtype) 作为精度
            assert_array_almost_equal(y_r,
                                      y_fft,
                                      decimal=self.equal_tolerance_fft(y_fft.dtype),)
            # 断言 y_r 和 y_direct 数组几乎相等，使用 self.equal_tolerance(y_direct.dtype) 作为精度
            assert_array_almost_equal(y_r,
                                      y_direct,
                                      decimal=self.equal_tolerance(y_direct.dtype),)
            # 断言 y_fft 的数据类型与 dt 相等
            assert_equal(y_fft.dtype, dt)
            # 断言 y_direct 的数据类型与 dt 相等
            assert_equal(y_direct.dtype, dt)

    # 定义一个测试方法，接受参数 self 和 dt
    def test_rank1_valid(self, dt):
        # 调用 _setup_rank1 方法，获取返回的 a, b, y_r
        a, b, y_r = self._setup_rank1(dt)
        # 使用 correlate 函数计算 a 和 b 的有效相关性，结果保存在 y 中
        y = correlate(a, b, 'valid')
        # 断言 y 数组几乎等于 y_r 的子数组 y_r[1:4]
        assert_array_almost_equal(y, y_r[1:4])
        # 断言 y 的数据类型与 dt 相等
        assert_equal(y.dtype, dt)

        # 见 gh-5897
        # 使用 correlate 函数计算 b 和 a 的有效相关性，结果保存在 y 中
        y = correlate(b, a, 'valid')
        # 断言 y 数组几乎等于 y_r 的子数组 y_r[1:4] 的逆序
        assert_array_almost_equal(y, y_r[1:4][::-1])
        # 断言 y 的数据类型与 dt 相等
        assert_equal(y.dtype, dt)

    # 定义一个测试方法，接受参数 self 和 dt
    def test_rank1_same(self, dt):
        # 调用 _setup_rank1 方法，获取返回的 a, b, y_r
        a, b, y_r = self._setup_rank1(dt)
        # 使用 correlate 函数计算 a 和 b 的相同尺寸相关性，结果保存在 y 中
        y = correlate(a, b, 'same')
        # 断言 y 数组几乎等于 y_r 的子数组 y_r[:-1]
        assert_array_almost_equal(y, y_r[:-1])
        # 断言 y 的数据类型与 dt 相等
        assert_equal(y.dtype, dt)

    # 定义一个测试方法，接受参数 self 和 dt
    def test_rank1_full(self, dt):
        # 调用 _setup_rank1 方法，获取返回的 a, b, y_r
        a, b, y_r = self._setup_rank1(dt)
        # 使用 correlate 函数计算 a 和 b 的完整尺寸相关性，结果保存在 y 中
        y = correlate(a, b, 'full')
        # 断言 y 数组几乎等于 y_r
        assert_array_almost_equal(y, y_r)
        # 断言 y 的数据类型与 dt 相等
        assert_equal(y.dtype, dt)
    # 在类中定义一个私有方法，用于设置三维数组 a, b 和目标结果 y_r
    def _setup_rank3(self, dt):
        # 创建一个形状为 (2, 4, 5) 的三维数组 a，元素为从0到39的等间距数列，使用列优先顺序（Fortran顺序），并将其类型转换为 dt 指定的类型
        a = np.linspace(0, 39, 40).reshape((2, 4, 5), order='F').astype(
            dt)
        # 创建一个形状为 (2, 3, 4) 的三维数组 b，元素为从0到23的等间距数列，使用列优先顺序，并将其类型转换为 dt 指定的类型
        b = np.linspace(0, 23, 24).reshape((2, 3, 4), order='F').astype(
            dt)

        # 创建一个预先设定好的目标结果 y_r，包含一个三维数组，元素为浮点数，具体数值直接定义在数组中，同时将其类型转换为 dt 指定的类型
        y_r = np.array([[[0., 184., 504., 912., 1360., 888., 472., 160.],
                        [46., 432., 1062., 1840., 2672., 1698., 864., 266.],
                        [134., 736., 1662., 2768., 3920., 2418., 1168., 314.],
                        [260., 952., 1932., 3056., 4208., 2580., 1240., 332.],
                        [202., 664., 1290., 1984., 2688., 1590., 712., 150.],
                        [114., 344., 642., 960., 1280., 726., 296., 38.]],

                       [[23., 400., 1035., 1832., 2696., 1737., 904., 293.],
                        [134., 920., 2166., 3680., 5280., 3306., 1640., 474.],
                        [325., 1544., 3369., 5512., 7720., 4683., 2192., 535.],
                        [571., 1964., 3891., 6064., 8272., 4989., 2324., 565.],
                        [434., 1360., 2586., 3920., 5264., 3054., 1312., 230.],
                        [241., 700., 1281., 1888., 2496., 1383., 532., 39.]],

                       [[22., 214., 528., 916., 1332., 846., 430., 132.],
                        [86., 484., 1098., 1832., 2600., 1602., 772., 206.],
                        [188., 802., 1698., 2732., 3788., 2256., 1018., 218.],
                        [308., 1006., 1950., 2996., 4052., 2400., 1078., 230.],
                        [230., 692., 1290., 1928., 2568., 1458., 596., 78.],
                        [126., 354., 636., 924., 1212., 654., 234., 0.]]],
                      dtype=np.float64).astype(dt)

        # 返回创建的数组 a, b, y_r 作为函数的结果
        return a, b, y_r

    # 测试三维卷积的有效性（valid 模式）
    def test_rank3_valid(self, dt):
        # 调用私有方法 _setup_rank3 获取设置好的数组 a, b, y_r
        a, b, y_r = self._setup_rank3(dt)
        # 使用 scipy.signal.correlate 对数组 a 和 b 进行有效卷积，得到结果 y
        y = correlate(a, b, "valid")
        # 断言 y 的值与预期的部分 y_r 值相近
        assert_array_almost_equal(y, y_r[1:2, 2:4, 3:5])
        # 断言 y 的数据类型与输入的 dt 相同
        assert_equal(y.dtype, dt)

        # 针对 gh-5897 的特定情况，再次进行卷积运算
        y = correlate(b, a, "valid")
        # 断言 y 的值与预期的部分 y_r 倒序的值相近
        assert_array_almost_equal(y, y_r[1:2, 2:4, 3:5][::-1, ::-1, ::-1])
        # 断言 y 的数据类型与输入的 dt 相同
        assert_equal(y.dtype, dt)

    # 测试三维卷积的相同填充（same 模式）
    def test_rank3_same(self, dt):
        # 调用私有方法 _setup_rank3 获取设置好的数组 a, b, y_r
        a, b, y_r = self._setup_rank3(dt)
        # 使用 scipy.signal.correlate 对数组 a 和 b 进行相同填充卷积，得到结果 y
        y = correlate(a, b, "same")
        # 断言 y 的值与预期的部分 y_r 值相近
        assert_array_almost_equal(y, y_r[0:-1, 1:-1, 1:-2])
        # 断言 y 的数据类型与输入的 dt 相同
        assert_equal(y.dtype, dt)

    # 测试三维卷积的全填充（full 模式）
    def test_rank3_all(self, dt):
        # 调用私有方法 _setup_rank3 获取设置好的数组 a, b, y_r
        a, b, y_r = self._setup_rank3(dt)
        # 使用 scipy.signal.correlate 对数组 a 和 b 进行全填充卷积，得到结果 y
        y = correlate(a, b)
        # 断言 y 的值与预期的 y_r 完全相同
        assert_array_almost_equal(y, y_r)
        # 断言 y 的数据类型与输入的 dt 相同
        assert_equal(y.dtype, dt)
class TestCorrelate:
    # Tests that don't depend on dtype

    def test_invalid_shapes(self):
        # 检查无效的形状，即一个数组的维度不能完全大于另一个数组对应的维度。
        # 这种设置应该引发 ValueError 异常。
        a = np.arange(1, 7).reshape((2, 3))  # 创建一个2x3的数组a
        b = np.arange(-6, 0).reshape((3, 2))  # 创建一个3x2的数组b

        assert_raises(ValueError, correlate, *(a, b), **{'mode': 'valid'})  # 检查a和b的相关性，期望抛出异常
        assert_raises(ValueError, correlate, *(b, a), **{'mode': 'valid'})  # 检查b和a的相关性，期望抛出异常

    def test_invalid_params(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        assert_raises(ValueError, correlate, a, b, mode='spam')  # 使用无效的模式'spam'调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, a, b, mode='eggs', method='fft')  # 使用无效的参数调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, a, b, mode='ham', method='direct')  # 使用无效的参数调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, a, b, mode='full', method='bacon')  # 使用无效的参数调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, a, b, mode='same', method='bacon')  # 使用无效的参数调用correlate函数，期望抛出异常

    def test_mismatched_dims(self):
        # 输入数组应该具有相同数量的维度
        assert_raises(ValueError, correlate, [1], 2, method='direct')  # 使用维度不匹配的参数调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, 1, [2], method='direct')  # 使用维度不匹配的参数调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, [1], 2, method='fft')  # 使用维度不匹配的参数调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, 1, [2], method='fft')  # 使用维度不匹配的参数调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, [1], [[2]])  # 使用维度不匹配的参数调用correlate函数，期望抛出异常
        assert_raises(ValueError, correlate, [3], 2)  # 使用维度不匹配的参数调用correlate函数，期望抛出异常

    def test_numpy_fastpath(self):
        a = [1, 2, 3]
        b = [4, 5]
        assert_allclose(correlate(a, b, mode='same'), [5, 14, 23])  # 检查使用给定模式的数组a和b的相关性，期望结果接近给定的数组

        a = [1, 2, 3]
        b = [4, 5, 6]
        assert_allclose(correlate(a, b, mode='same'), [17, 32, 23])  # 检查使用给定模式的数组a和b的相关性，期望结果接近给定的数组
        assert_allclose(correlate(a, b, mode='full'), [6, 17, 32, 23, 12])  # 检查使用给定模式的数组a和b的相关性，期望结果接近给定的数组
        assert_allclose(correlate(a, b, mode='valid'), [32])  # 检查使用给定模式的数组a和b的相关性，期望结果接近给定的数组

@pytest.mark.parametrize("mode", ["valid", "same", "full"])
@pytest.mark.parametrize("behind", [True, False])
@pytest.mark.parametrize("input_size", [100, 101, 1000, 1001, 10000, 10001])
def test_correlation_lags(mode, behind, input_size):
    # 生成随机数据
    rng = np.random.RandomState(0)
    in1 = rng.standard_normal(input_size)  # 使用给定大小生成标准正态分布的随机数据
    offset = int(input_size/10)
    # 生成要与之相关的数组的偏移版本
    if behind:
        # y在x的后面
        in2 = np.concatenate([rng.standard_normal(offset), in1])  # 创建一个与in1相关的偏移版本
        expected = -offset  # 预期的偏移值为-offset
    else:
        # y在x的前面
        in2 = in1[offset:]  # 创建一个与in1相关的偏移版本
        expected = offset  # 预期的偏移值为offset
    # 交叉相关，返回滞后信息
    correlation = correlate(in1, in2, mode=mode)  # 计算in1和in2的相关性
    lags = correlation_lags(in1.size, in2.size, mode=mode)  # 获取相关性滞后信息
    # 确定峰值
    lag_index = np.argmax(correlation)  # 找到相关性数组中的最大值索引
    # 检查预期结果
    assert_equal(lags[lag_index], expected)  # 检查相关性滞后数组中的值是否符合预期
    # 相关性和滞后的形状应该匹配
    assert_equal(lags.shape, correlation.shape)  # 检查相关性和滞后数组的形状是否匹配
@pytest.mark.parametrize('dt', [np.csingle, np.cdouble, np.clongdouble])
class TestCorrelateComplex:
    # 用于比较结果时所使用的十进制精度。
    # 这个值将作为 assert_array_almost_equal() 的 'decimal' 关键字参数传递。
    # 由于 correlate 可能选择使用 FFT 方法，在内部将 longdouble 转换为 double，因此不要期望 longdouble 比 double 有更好的精度（参见 gh-9520）。

    def decimal(self, dt):
        if dt == np.clongdouble:
            dt = np.cdouble
        return int(2 * np.finfo(dt).precision / 3)

    def _setup_rank1(self, dt, mode):
        np.random.seed(9)
        # 创建一个长度为 10 的随机实数数组，并将其类型转换为指定的 dt 类型，然后加上长度为 10 的随机虚数数组
        a = np.random.randn(10).astype(dt)
        a += 1j * np.random.randn(10).astype(dt)
        # 创建一个长度为 8 的随机实数数组，并将其类型转换为指定的 dt 类型，然后加上长度为 8 的随机虚数数组
        b = np.random.randn(8).astype(dt)
        b += 1j * np.random.randn(8).astype(dt)

        # 计算相关性的实部并转换为指定的 dt 类型
        y_r = (correlate(a.real, b.real, mode=mode) +
               correlate(a.imag, b.imag, mode=mode)).astype(dt)
        # 计算相关性的虚部并转换为指定的 dt 类型
        y_r += 1j * (-correlate(a.real, b.imag, mode=mode) +
                     correlate(a.imag, b.real, mode=mode))
        return a, b, y_r

    def test_rank1_valid(self, dt):
        # 设置测试数据和预期结果
        a, b, y_r = self._setup_rank1(dt, 'valid')
        # 计算相关性并比较结果，使用预先定义的精度
        y = correlate(a, b, 'valid')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

        # 见 gh-5897
        # 对调参数顺序再次计算相关性并比较结果，使用预先定义的精度
        y = correlate(b, a, 'valid')
        assert_array_almost_equal(y, y_r[::-1].conj(), decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_same(self, dt):
        # 设置测试数据和预期结果
        a, b, y_r = self._setup_rank1(dt, 'same')
        # 计算相关性并比较结果，使用预先定义的精度
        y = correlate(a, b, 'same')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_full(self, dt):
        # 设置测试数据和预期结果
        a, b, y_r = self._setup_rank1(dt, 'full')
        # 计算相关性并比较结果，使用预先定义的精度
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_swap_full(self, dt):
        # 创建两个复数数组，计算它们的相关性，并断言结果是否符合预期
        d = np.array([0.+0.j, 1.+1.j, 2.+2.j], dtype=dt)
        k = np.array([1.+3.j, 2.+4.j, 3.+5.j, 4.+6.j], dtype=dt)
        y = correlate(d, k)
        assert_equal(y, [0.+0.j, 10.-2.j, 28.-6.j, 22.-6.j, 16.-6.j, 8.-4.j])

    def test_swap_same(self, dt):
        # 创建两个复数列表，计算它们的相关性，并断言结果是否符合预期
        d = [0.+0.j, 1.+1.j, 2.+2.j]
        k = [1.+3.j, 2.+4.j, 3.+5.j, 4.+6.j]
        y = correlate(d, k, mode="same")
        assert_equal(y, [10.-2.j, 28.-6.j, 22.-6.j])

    def test_rank3(self, dt):
        # 创建两个三维数组，计算它们的相关性，并断言结果是否符合预期
        a = np.random.randn(10, 8, 6).astype(dt)
        a += 1j * np.random.randn(10, 8, 6).astype(dt)
        b = np.random.randn(8, 6, 4).astype(dt)
        b += 1j * np.random.randn(8, 6, 4).astype(dt)

        # 计算相关性的实部和虚部，并转换为指定的 dt 类型
        y_r = (correlate(a.real, b.real)
               + correlate(a.imag, b.imag)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag) + correlate(a.imag, b.real))

        y = correlate(a, b, 'full')
        # 比较计算结果和预期结果，使用稍微减小的精度以容忍数值计算误差
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        assert_equal(y.dtype, dt)
    # 定义一个测试方法，用于测试一维信号相关性
    def test_rank0(self, dt):
        # 生成一个实部和虚部都是随机数的复数数组，并指定数据类型为 dt
        a = np.array(np.random.randn()).astype(dt)
        a += 1j * np.array(np.random.randn()).astype(dt)
        # 生成另一个实部和虚部都是随机数的复数数组，并指定数据类型为 dt
        b = np.array(np.random.randn()).astype(dt)
        b += 1j * np.array(np.random.randn()).astype(dt)

        # 计算实部和虚部的相关性，结果的数据类型为 dt
        y_r = (correlate(a.real, b.real)
               + correlate(a.imag, b.imag)).astype(dt)
        # 计算交叉相关性，结果包括实部和虚部，数据类型为 dt
        y_r += 1j * np.array(-correlate(a.real, b.imag) +
                             correlate(a.imag, b.real))

        # 计算完全相关性，并进行断言验证其与预期结果 y_r 接近
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        # 断言 y 的数据类型与指定的 dt 相符
        assert_equal(y.dtype, dt)

        # 验证一些特殊情况的相关性计算是否正确
        assert_equal(correlate([1], [2j]), correlate(1, 2j))
        assert_equal(correlate([2j], [3j]), correlate(2j, 3j))
        assert_equal(correlate([3j], [4]), correlate(3j, 4))
class TestCorrelate2d:

    def test_consistency_correlate_funcs(self):
        # 比较 np.correlate, signal.correlate, signal.correlate2d 的一致性
        a = np.arange(5)  # 创建长度为 5 的数组 a
        b = np.array([3.2, 1.4, 3])  # 创建数组 b 包含三个元素
        for mode in ['full', 'valid', 'same']:  # 遍历三种模式：'full', 'valid', 'same'
            # 断言 np.correlate 和 signal.correlate 的结果近似相等，使用给定的模式
            assert_almost_equal(np.correlate(a, b, mode=mode),
                                signal.correlate(a, b, mode=mode))
            # 断言 signal.correlate2d 的结果与 signal.correlate 函数近似相等，使用给定的模式
            assert_almost_equal(np.squeeze(signal.correlate2d([a], [b], mode=mode)),
                                signal.correlate(a, b, mode=mode))

            # 修复 gh-5897 问题
            if mode == 'valid':
                # 断言 np.correlate(b, a, mode='valid') 和 signal.correlate(b, a, mode='valid') 的结果近似相等
                assert_almost_equal(np.correlate(b, a, mode=mode),
                                    signal.correlate(b, a, mode=mode))
                # 断言 signal.correlate2d([b], [a], mode='valid') 和 signal.correlate(b, a, mode='valid') 的结果近似相等
                assert_almost_equal(np.squeeze(signal.correlate2d([b], [a], mode=mode)),
                                    signal.correlate(b, a, mode=mode))

    def test_invalid_shapes(self):
        # "invalid" 指的是没有一个数组的维度至少与另一个数组的相应维度一样大。
        # 这种设置应该引发 ValueError 异常。
        a = np.arange(1, 7).reshape((2, 3))  # 创建形状为 (2, 3) 的数组 a
        b = np.arange(-6, 0).reshape((3, 2))  # 创建形状为 (3, 2) 的数组 b

        # 断言 signal.correlate2d(a, b, mode='valid') 抛出 ValueError 异常
        assert_raises(ValueError, signal.correlate2d, *(a, b), **{'mode': 'valid'})
        # 断言 signal.correlate2d(b, a, mode='valid') 抛出 ValueError 异常
        assert_raises(ValueError, signal.correlate2d, *(b, a), **{'mode': 'valid'})

    def test_complex_input(self):
        # 断言 signal.correlate2d([[1]], [[2j]]) 的结果等于 -2j
        assert_equal(signal.correlate2d([[1]], [[2j]]), -2j)
        # 断言 signal.correlate2d([[2j]], [[3j]]) 的结果等于 6
        assert_equal(signal.correlate2d([[2j]], [[3j]]), 6)
        # 断言 signal.correlate2d([[3j]], [[4]]) 的结果等于 12j
        assert_equal(signal.correlate2d([[3j]], [[4]]), 12j)


class TestLFilterZI:

    def test_basic(self):
        a = np.array([1.0, -1.0, 0.5])  # 创建数组 a
        b = np.array([1.0, 0.0, 2.0])   # 创建数组 b
        zi_expected = np.array([5.0, -1.0])  # 创建预期的初始条件数组 zi_expected
        zi = lfilter_zi(b, a)  # 调用 lfilter_zi 函数计算初始条件 zi
        # 断言计算得到的 zi 与预期的 zi_expected 近似相等
        assert_array_almost_equal(zi, zi_expected)

    def test_scale_invariance(self):
        # 回归测试。当 a[0] 非零时，存在一个错误，即 b 没有被正确重新缩放。
        b = np.array([2, 8, 5])  # 创建数组 b
        a = np.array([1, 1, 8])  # 创建数组 a
        zi1 = lfilter_zi(b, a)  # 计算第一个初始条件 zi1
        zi2 = lfilter_zi(2*b, 2*a)  # 计算第二个初始条件 zi2
        # 断言两个初始条件 zi1 和 zi2 近乎相等
        assert_allclose(zi2, zi1, rtol=1e-12)

    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_types(self, dtype):
        b = np.zeros((8), dtype=dtype)  # 创建指定数据类型的全零数组 b
        a = np.array([1], dtype=dtype)  # 创建指定数据类型的数组 a
        # 断言 signal.lfilter_zi(b, a) 返回的实部数据类型与指定数据类型 dtype 相同
        assert_equal(np.real(signal.lfilter_zi(b, a)).dtype, dtype)


class TestFiltFilt:
    filtfilt_kind = 'tf'
    ```python`
        def filtfilt(self, zpk, x, axis=-1, padtype='odd', padlen=None,
                     method='pad', irlen=None):
            # 根据 self.filtfilt_kind 的值选择不同的滤波器类型进行滤波处理
            if self.filtfilt_kind == 'tf':
                # 将 zpk 参数转换为传统滤波器的系数形式
                b, a = zpk2tf(*zpk)
                # 调用 scipy 的 filtfilt 函数进行前向后向滤波处理，并返回结果
                return filtfilt(b, a, x, axis, padtype, padlen, method, irlen)
            elif self.filtfilt_kind == 'sos':
                # 将 zpk 参数转换为二阶节序列（SOS）形式的滤波器系数
                sos = zpk2sos(*zpk)
                # 调用 scipy 的 sosfiltfilt 函数进行前向后向滤波处理，并返回结果
                return sosfiltfilt(sos, x, axis, padtype, padlen)
    
        def test_basic(self):
            # 使用 tf2zpk 函数生成传统滤波器的零极点信息
            zpk = tf2zpk([1, 2, 3], [1, 2, 3])
            # 调用 self.filtfilt 函数对输入信号进行滤波处理
            out = self.filtfilt(zpk, np.arange(12))
            # 使用 assert_allclose 检查滤波处理结果是否与期望值接近
            assert_allclose(out, arange(12), atol=5.28e-11)
    
        def test_sine(self):
            rate = 2000
            t = np.linspace(0, 1.0, rate + 1)
            # 生成一个包含低频和高频信号的复合信号
            xlow = np.sin(5 * 2 * np.pi * t)
            xhigh = np.sin(250 * 2 * np.pi * t)
            x = xlow + xhigh
    
            # 使用 butter 函数生成二阶巴特沃斯滤波器的零极点信息
            zpk = butter(8, 0.125, output='zpk')
            # 计算巴特沃斯滤波器的主极点的幅值
            r = np.abs(zpk[1]).max()
            eps = 1e-5
            # 计算信号在初始阶段衰减到给定因子 eps 所需的步数估计值
            n = int(np.ceil(np.log(eps) / np.log(r)))
    
            # 使用 self.filtfilt 函数对输入信号进行滤波处理，padlen 指定为 n
            y = self.filtfilt(zpk, x, padlen=n)
            # 断言滤波后的信号 y 应接近于原始低频信号 xlow
            err = np.abs(y - xlow).max()
            assert_(err < 1e-4)
    
            # 在二维情况下进行相同的滤波处理
            x2d = np.vstack([xlow, xlow + xhigh])
            y2d = self.filtfilt(zpk, x2d, padlen=n, axis=1)
            # 断言二维信号滤波处理后形状不变，并且结果应接近于原始低频信号 xlow
            assert_equal(y2d.shape, x2d.shape)
            err = np.abs(y2d - xlow).max()
            assert_(err < 1e-4)
    
            # 使用先前的结果检查 axis 关键字的使用
            y2dt = self.filtfilt(zpk, x2d.T, padlen=n, axis=0)
            # 断言转置后的二维信号滤波结果与原始结果一致
            assert_equal(y2d, y2dt.T)
    
        def test_axis(self):
            # 在三维数组上测试 'axis' 关键字的使用
            x = np.arange(10.0 * 11.0 * 12.0).reshape(10, 11, 12)
            # 使用 butter 函数生成三阶巴特沃斯滤波器的零极点信息
            zpk = butter(3, 0.125, output='zpk')
            # 分别在不同轴向上应用滤波器，验证结果
            y0 = self.filtfilt(zpk, x, padlen=0, axis=0)
            y1 = self.filtfilt(zpk, np.swapaxes(x, 0, 1), padlen=0, axis=1)
            assert_array_equal(y0, np.swapaxes(y1, 0, 1))
            y2 = self.filtfilt(zpk, np.swapaxes(x, 0, 2), padlen=0, axis=2)
            assert_array_equal(y0, np.swapaxes(y2, 0, 2))
    
        def test_acoeff(self):
            # 只有在 self.filtfilt_kind 为 'tf' 时才需要进行 'a' 系数的测试
            if self.filtfilt_kind != 'tf':
                return
            # 使用 signal.filtfilt 函数测试 'a' 系数作为单个数值时的滤波效果
            out = signal.filtfilt([.5, .5], 1, np.arange(10))
            assert_allclose(out, np.arange(10), rtol=1e-14, atol=1e-14)
    # 定义名为 test_gust_simple 的测试方法
    def test_gust_simple(self):
        # 如果 filtfilt_kind 不等于 'tf'，跳过测试并显示相应信息
        if self.filtfilt_kind != 'tf':
            pytest.skip('gust only implemented for TF systems')
        
        # 定义输入数组 x，长度为 2，这种情况下的精确解通过手动计算得出
        x = np.array([1.0, 2.0])
        # 定义滤波器的分子系数 b
        b = np.array([0.5])
        # 定义滤波器的分母系数 a
        a = np.array([1.0, -0.5])
        
        # 调用 _filtfilt_gust 函数进行双向滤波计算，并返回结果 y 和状态变量 z1, z2
        y, z1, z2 = _filtfilt_gust(b, a, x)
        
        # 使用 assert_allclose 断言函数检查 z1 和 z2 的计算结果是否接近预期值
        assert_allclose([z1[0], z2[0]],
                        [0.3*x[0] + 0.2*x[1], 0.2*x[0] + 0.3*x[1]])
        
        # 使用 assert_allclose 断言函数检查 y 的计算结果是否接近预期值
        assert_allclose(y, [z1[0] + 0.25*z2[0] + 0.25*x[0] + 0.125*x[1],
                            0.25*z1[0] + z2[0] + 0.125*x[0] + 0.25*x[1]])

    # 定义名为 test_gust_scalars 的测试方法
    def test_gust_scalars(self):
        # 如果 filtfilt_kind 不等于 'tf'，跳过测试并显示相应信息
        if self.filtfilt_kind != 'tf':
            pytest.skip('gust only implemented for TF systems')
        
        # 定义输入数组 x，从 0 到 11
        x = np.arange(12)
        # 定义滤波器的分子系数 b，为标量 3.0
        b = 3.0
        # 定义滤波器的分母系数 a，为标量 2.0
        a = 2.0
        
        # 调用 filtfilt 函数进行双向滤波计算，使用 "gust" 方法，并返回结果 y
        y = filtfilt(b, a, x, method="gust")
        
        # 计算预期结果，即 (b/a)**2 * x
        expected = (b/a)**2 * x
        
        # 使用 assert_allclose 断言函数检查 y 的计算结果是否接近预期值
        assert_allclose(y, expected)
class TestSOSFiltFilt(TestFiltFilt):
    filtfilt_kind = 'sos'

    def test_equivalence(self):
        """Test equivalence between sosfiltfilt and filtfilt"""
        # 生成一个长度为1000的随机数组
        x = np.random.RandomState(0).randn(1000)
        # 循环测试阶数从1到5的情况
        for order in range(1, 6):
            # 使用Butterworth滤波器设计，生成传递函数(z, p, k)
            zpk = signal.butter(order, 0.35, output='zpk')
            # 将传递函数转换为传递函数分子和分母系数
            b, a = zpk2tf(*zpk)
            # 将传递函数转换为二阶段系统表示
            sos = zpk2sos(*zpk)
            # 使用传统的filtfilt函数进行前向后向滤波
            y = filtfilt(b, a, x)
            # 使用sosfiltfilt函数进行前向后向滤波
            y_sos = sosfiltfilt(sos, x)
            # 检查两种方法的输出是否等价
            assert_allclose(y, y_sos, atol=1e-12, err_msg='order=%s' % order)


def filtfilt_gust_opt(b, a, x):
    """
    An alternative implementation of filtfilt with Gustafsson edges.

    This function computes the same result as
    `scipy.signal._signaltools._filtfilt_gust`, but only 1-d arrays
    are accepted.  The problem is solved using `fmin` from `scipy.optimize`.
    `_filtfilt_gust` is significantly faster than this implementation.
    """
    def filtfilt_gust_opt_func(ics, b, a, x):
        """Objective function used in filtfilt_gust_opt."""
        # 计算前向和后向滤波的初始条件长度
        m = max(len(a), len(b)) - 1
        # 前向滤波器的初始条件
        z0f = ics[:m]
        # 后向滤波器的初始条件
        z0b = ics[m:]
        # 前向传播数据的滤波过程，并返回结果
        y_f = lfilter(b, a, x, zi=z0f)[0]
        # 反向传播前向结果的数据，并进行反向滤波过程，并返回结果
        y_fb = lfilter(b, a, y_f[::-1], zi=z0b)[0][::-1]

        # 反向传播数据的滤波过程，并返回结果
        y_b = lfilter(b, a, x[::-1], zi=z0b)[0][::-1]
        # 前向传播反向结果的数据，并进行前向滤波过程，并返回结果
        y_bf = lfilter(b, a, y_b, zi=z0f)[0]
        # 计算前向后向滤波结果之间的平方差值
        value = np.sum((y_fb - y_bf)**2)
        return value

    # 计算前向和后向滤波的初始条件长度
    m = max(len(a), len(b)) - 1
    # 计算初始条件，使用数据x的均值作为初始条件的一部分
    zi = lfilter_zi(b, a)
    ics = np.concatenate((x[:m].mean()*zi, x[-m:].mean()*zi))
    # 使用优化方法fmin进行优化，找到最小值
    result = fmin(filtfilt_gust_opt_func, ics, args=(b, a, x),
                  xtol=1e-10, ftol=1e-12,
                  maxfun=10000, maxiter=10000,
                  full_output=True, disp=False)
    opt, fopt, niter, funcalls, warnflag = result
    if warnflag > 0:
        # 如果优化方法出现问题，抛出运行时错误
        raise RuntimeError("minimization failed in filtfilt_gust_opt: "
                           "warnflag=%d" % warnflag)
    z0f = opt[:m]
    z0b = opt[m:]

    # 使用计算出的初始条件应用前向后向滤波
    y_b = lfilter(b, a, x[::-1], zi=z0b)[0][::-1]
    y = lfilter(b, a, y_b, zi=z0f)[0]

    return y, z0f, z0b


def check_filtfilt_gust(b, a, shape, axis, irlen=None):
    # 生成需要进行滤波的数据x
    np.random.seed(123)
    x = np.random.randn(*shape)

    # 对x应用filtfilt进行滤波，这是要检查的主要计算
    y = filtfilt(b, a, x, axis=axis, method="gust", irlen=irlen)

    # 调用私有函数以便测试初始条件
    yg, zg1, zg2 = _filtfilt_gust(b, a, x, axis=axis, irlen=irlen)

    # filtfilt_gust_opt是一个独立的实现，给出了预期的结果，
    # 但它只处理1-D数组，因此需要一些循环和重塑操作来创建预期的输出数组。
    xx = np.swapaxes(x, axis, -1)
    out_shape = xx.shape[:-1]
    yo = np.empty_like(xx)
    m = max(len(a), len(b)) - 1
    zo1 = np.empty(out_shape + (m,))
    zo2 = np.empty(out_shape + (m,))
    # 遍历每个可能的索引组合，使用 filtfit_gust_opt 函数处理
    for indx in product(*[range(d) for d in out_shape]):
        # 对当前索引组合应用 filtfit_gust_opt 函数，将结果分别存储到 yo, zo1, zo2 中
        yo[indx], zo1[indx], zo2[indx] = filtfilt_gust_opt(b, a, xx[indx])
    
    # 将数组 yo 按指定轴与最后一个轴进行交换
    yo = np.swapaxes(yo, -1, axis)
    # 将数组 zo1 按指定轴与最后一个轴进行交换
    zo1 = np.swapaxes(zo1, -1, axis)
    # 将数组 zo2 按指定轴与最后一个轴进行交换
    zo2 = np.swapaxes(zo2, -1, axis)

    # 使用 assert_allclose 函数验证 y 和 yo 的接近程度
    assert_allclose(y, yo, rtol=1e-8, atol=1e-9)
    # 使用 assert_allclose 函数验证 yg 和 yo 的接近程度
    assert_allclose(yg, yo, rtol=1e-8, atol=1e-9)
    # 使用 assert_allclose 函数验证 zg1 和 zo1 的接近程度
    assert_allclose(zg1, zo1, rtol=1e-8, atol=1e-9)
    # 使用 assert_allclose 函数验证 zg2 和 zo2 的接近程度
    assert_allclose(zg2, zo2, rtol=1e-8, atol=1e-9)
@pytest.mark.fail_slow(10)
# 使用 pytest 的 mark 功能，标记该测试为失败较慢的测试，超过10秒将视为失败
def test_choose_conv_method():
    # 遍历卷积模式和维度参数的组合
    for mode in ['valid', 'same', 'full']:
        for ndim in [1, 2]:
            n, k, true_method = 8, 6, 'direct'
            # 生成随机的输入信号 x 和滤波器 h
            x = np.random.randn(*((n,) * ndim))
            h = np.random.randn(*((k,) * ndim))

            # 调用 choose_conv_method 函数选择卷积方法，并断言选择的方法与预期相符
            method = choose_conv_method(x, h, mode=mode)
            assert_equal(method, true_method)

            # 调用 choose_conv_method 函数，同时测量执行时间
            method_try, times = choose_conv_method(x, h, mode=mode, measure=True)
            # 断言尝试选择的方法为 'fft' 或 'direct'，并且 times 是一个字典
            assert_(method_try in {'fft', 'direct'})
            assert_(isinstance(times, dict))
            # 断言 'fft' 和 'direct' 两个键在 times 字典中
            assert_('fft' in times.keys() and 'direct' in times.keys())

        n = 10
        # 遍历不支持 FFT 的复数数据类型，如果 np 模块有对应的数据类型，则进行测试
        for not_fft_conv_supp in ["complex256", "complex192"]:
            if hasattr(np, not_fft_conv_supp):
                x = np.ones(n, dtype=not_fft_conv_supp)
                h = x.copy()
                # 断言对于给定的数据类型，choose_conv_method 返回 'direct'
                assert_equal(choose_conv_method(x, h, mode=mode), 'direct')

        # 测试一个超大整数的输入情况，期望 choose_conv_method 返回 'direct'
        x = np.array([2**51], dtype=np.int64)
        h = x.copy()
        assert_equal(choose_conv_method(x, h, mode=mode), 'direct')

        # 使用 Decimal 类型的输入测试，期望 choose_conv_method 返回 'direct'
        x = [Decimal(3), Decimal(2)]
        h = [Decimal(1), Decimal(4)]
        assert_equal(choose_conv_method(x, h, mode=mode), 'direct')


@pytest.mark.fail_slow(10)
# 使用 pytest 的 mark 功能，标记该测试为失败较慢的测试，超过10秒将视为失败
def test_filtfilt_gust():
    # 设计一个椭圆滤波器
    z, p, k = signal.ellip(3, 0.01, 120, 0.0875, output='zpk')

    # 计算滤波器的近似脉冲响应长度
    eps = 1e-10
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))

    np.random.seed(123)

    # 将 zpk 形式转换为传递函数形式
    b, a = zpk2tf(z, p, k)
    for irlen in [None, approx_impulse_len]:
        signal_len = 5 * approx_impulse_len

        # 测试一维情况下的 filtffilt_gust 函数
        check_filtfilt_gust(b, a, (signal_len,), 0, irlen)

        # 测试三维情况下的 filtffilt_gust 函数，测试每个轴
        for axis in range(3):
            shape = [2, 2, 2]
            shape[axis] = signal_len
            check_filtfilt_gust(b, a, shape, axis, irlen)

    # 测试长度小于 2*approx_impulse_len 的情况
    # 在这种情况下，filtfilt_gust 应该与 irlen=None 的情况行为相同
    length = 2*approx_impulse_len - 50
    check_filtfilt_gust(b, a, (length,), 0, approx_impulse_len)


class TestDecimate:
    # 测试 signal.decimate 函数的错误参数情况
    def test_bad_args(self):
        x = np.arange(12)
        assert_raises(TypeError, signal.decimate, x, q=0.5, n=1)
        assert_raises(TypeError, signal.decimate, x, q=2, n=0.5)

    # 测试 signal.decimate 函数的基本 IIR 滤波情况
    def test_basic_IIR(self):
        x = np.arange(12)
        y = signal.decimate(x, 2, n=1, ftype='iir', zero_phase=False).round()
        assert_array_equal(y, x[::2])

    # 测试 signal.decimate 函数的基本 FIR 滤波情况
    def test_basic_FIR(self):
        x = np.arange(12)
        y = signal.decimate(x, 2, n=1, ftype='fir', zero_phase=False).round()
        assert_array_equal(y, x[::2])
    # 定义测试函数，用于验证信号处理库中的 downsample 功能是否正常
    def test_shape(self):
        # 标记为 ticket #1480 的回归测试
        z = np.zeros((30, 30))
        # 对 z 进行沿指定轴的二次降采样，不使用零相位
        d0 = signal.decimate(z, 2, axis=0, zero_phase=False)
        # 断言降采样后的形状是否符合预期
        assert_equal(d0.shape, (15, 30))
        d1 = signal.decimate(z, 2, axis=1, zero_phase=False)
        # 断言降采样后的形状是否符合预期
        assert_equal(d1.shape, (30, 15))

    # 零相位 FIR 滤波器相位移测试
    def test_phaseshift_FIR(self):
        with suppress_warnings() as sup:
            # 过滤特定警告
            sup.filter(BadCoefficients, "Badly conditioned filter")
            # 调用内部方法进行零相位 FIR 滤波器相位测试
            self._test_phaseshift(method='fir', zero_phase=False)

    # 零相位 FIR 滤波器零相位测试
    def test_zero_phase_FIR(self):
        with suppress_warnings() as sup:
            # 过滤特定警告
            sup.filter(BadCoefficients, "Badly conditioned filter")
            # 调用内部方法进行零相位 FIR 滤波器零相位测试
            self._test_phaseshift(method='fir', zero_phase=True)

    # IIR 滤波器相位移测试
    def test_phaseshift_IIR(self):
        # 调用内部方法进行 IIR 滤波器相位测试，不使用零相位
        self._test_phaseshift(method='iir', zero_phase=False)

    # IIR 滤波器零相位测试
    def test_zero_phase_IIR(self):
        # 调用内部方法进行 IIR 滤波器零相位测试
        self._test_phaseshift(method='iir', zero_phase=True)

    # 内部方法，用于测试不同滤波器和相位模式下的相位移效果
    def _test_phaseshift(self, method, zero_phase):
        rate = 120
        rates_to = [15, 20, 30, 40]  # q = 8, 6, 4, 3

        t_tot = 100  # 需要让抗混叠滤波器稳定
        t = np.arange(rate*t_tot+1) / float(rate)

        # 在 0.8*nyquist 处设置正弦波，使用窗口函数以避免边缘伪影
        freqs = np.array(rates_to) * 0.8 / 2
        d = (np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * t)
             * signal.windows.tukey(t.size, 0.1))

        for rate_to in rates_to:
            q = rate // rate_to
            t_to = np.arange(rate_to*t_tot+1) / float(rate_to)
            d_tos = (np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * t_to)
                     * signal.windows.tukey(t_to.size, 0.1))

            # 设置下采样滤波器，匹配 v0.17 版本的默认设置
            if method == 'fir':
                n = 30
                # 创建 FIR 滤波器系统
                system = signal.dlti(signal.firwin(n + 1, 1. / q,
                                                   window='hamming'), 1.)
            elif method == 'iir':
                n = 8
                wc = 0.8*np.pi/q
                # 创建 IIR 滤波器系统，使用 Chebyshev I 滤波器设计方法
                system = signal.dlti(*signal.cheby1(n, 0.05, wc/np.pi))

            # 计算预期的相位响应，作为单位复数向量
            if zero_phase is False:
                # 计算频率响应
                _, h_resps = signal.freqz(system.num, system.den,
                                          freqs/rate*2*np.pi)
                h_resps /= np.abs(h_resps)
            else:
                # 零相位情况下，相位响应为 1
                h_resps = np.ones_like(freqs)

            # 对实部进行降采样，使用指定的滤波器类型和相位模式
            y_resamps = signal.decimate(d.real, q, n, ftype=system,
                                        zero_phase=zero_phase)

            # 从复内积中获取相位，类似于 CSD
            h_resamps = np.sum(d_tos.conj() * y_resamps, axis=-1)
            h_resamps /= np.abs(h_resamps)
            subnyq = freqs < 0.5*rate_to

            # 复向量应该对齐，只比较 Nyquist 以下的部分
            assert_allclose(np.angle(h_resps.conj()*h_resamps)[subnyq], 0,
                            atol=1e-3, rtol=1e-3)
    def test_auto_n(self):
        # 测试我们选择的 n 值是否合理（取决于下采样因子）
        sfreq = 100.
        n = 1000
        t = np.arange(n) / sfreq
        # 对于降采样（>= 15），会发生混叠现象
        x = np.sqrt(2. / n) * np.sin(2 * np.pi * (sfreq / 30.) * t)
        assert_allclose(np.linalg.norm(x), 1., rtol=1e-3)
        x_out = signal.decimate(x, 30, ftype='fir')
        assert_array_less(np.linalg.norm(x_out), 0.01)

    def test_long_float32(self):
        # 回归测试：gh-15072。使用 32 位浮点数和 lfilter 或 filtfilt 会导致数值不稳定
        x = signal.decimate(np.ones(10_000, dtype=np.float32), 10)
        assert not any(np.isnan(x))

    def test_float16_upcast(self):
        # float16 必须向上转换为 float64
        x = signal.decimate(np.ones(100, dtype=np.float16), 10)
        assert x.dtype.type == np.float64

    def test_complex_iir_dlti(self):
        # 回归测试：gh-17845
        # 滤波器的中心频率 [Hz]
        fcentre = 50
        # 滤波器通带宽度 [Hz]
        fwidth = 5
        # 采样率 [Hz]
        fs = 1e3

        z, p, k = signal.butter(2, 2*np.pi*fwidth/2, output='zpk', fs=fs)
        z = z.astype(complex) * np.exp(2j * np.pi * fcentre/fs)
        p = p.astype(complex) * np.exp(2j * np.pi * fcentre/fs)
        system = signal.dlti(z, p, k)

        t = np.arange(200) / fs

        # 输入信号
        u = (np.exp(2j * np.pi * fcentre * t)
             + 0.5 * np.exp(-2j * np.pi * fcentre * t))

        # 非零相位下采样
        ynzp = signal.decimate(u, 2, ftype=system, zero_phase=False)
        ynzpref = signal.lfilter(*signal.zpk2tf(z, p, k),
                                 u)[::2]

        assert_equal(ynzp, ynzpref)

        # 零相位下采样
        yzp = signal.decimate(u, 2, ftype=system, zero_phase=True)
        yzpref = signal.filtfilt(*signal.zpk2tf(z, p, k),
                                 u)[::2]

        assert_allclose(yzp, yzpref, rtol=1e-10, atol=1e-13)
    # 定义一个测试方法，用于测试复杂 FIR 数字线性时不变系统（DLTI）
    def test_complex_fir_dlti(self):
        # 中心频率 [Hz]
        fcentre = 50
        # 滤波器通带宽度 [Hz]
        fwidth = 5
        # 采样率 [Hz]
        fs = 1e3
        # 滤波器系数个数
        numtaps = 20

        # 创建一个关于0Hz的FIR滤波器
        bbase = signal.firwin(numtaps, fwidth/2, fs=fs)

        # 将其旋转到所需的频率
        zbase = np.roots(bbase)
        zrot = zbase * np.exp(2j * np.pi * fcentre/fs)
        # 在约50Hz处创建FIR滤波器，保持通带增益为0dB
        bz = bbase[0] * np.poly(zrot)

        # 创建一个DLTI系统对象
        system = signal.dlti(bz, 1)

        # 生成时间向量
        t = np.arange(200) / fs

        # 输入信号
        u = (np.exp(2j * np.pi * fcentre * t)
             + 0.5 * np.exp(-2j * np.pi * fcentre * t))

        # 非零相位的抽样
        ynzp = signal.decimate(u, 2, ftype=system, zero_phase=False)
        # 使用 upfirdn 函数进行抽样处理
        ynzpref = signal.upfirdn(bz, u, up=1, down=2)[:100]

        # 断言非零相位抽样结果与 upfirdn 函数处理结果相等
        assert_equal(ynzp, ynzpref)

        # 零相位的抽样
        yzp = signal.decimate(u, 2, ftype=system, zero_phase=True)
        # 使用 resample_poly 函数进行抽样处理
        yzpref = signal.resample_poly(u, 1, 2, window=bz)

        # 断言零相位抽样结果与 resample_poly 函数处理结果相等
        assert_equal(yzp, yzpref)
class TestHilbert:

    def test_bad_args(self):
        # 创建包含一个复数的 NumPy 数组
        x = np.array([1.0 + 0.0j])
        # 断言调用 hilbert 函数时应该抛出 ValueError 异常
        assert_raises(ValueError, hilbert, x)
        
        # 创建一个包含浮点数的 NumPy 数组
        x = np.arange(8.0)
        # 断言调用 hilbert 函数时应该抛出 ValueError 异常，且 N 参数为 0
        assert_raises(ValueError, hilbert, x, N=0)

    def test_hilbert_theoretical(self):
        # test cases by Ariel Rokem
        # 设置精度值
        decimal = 14

        # 设置常数 pi
        pi = np.pi
        # 在 [0, 2*pi) 范围内生成均匀间隔的数值数组
        t = np.arange(0, 2 * pi, pi / 256)
        # 计算四个正弦和余弦函数
        a0 = np.sin(t)
        a1 = np.cos(t)
        a2 = np.sin(2 * t)
        a3 = np.cos(2 * t)
        # 将这些函数堆叠成一个二维数组
        a = np.vstack([a0, a1, a2, a3])

        # 计算希尔伯特变换
        h = hilbert(a)
        # 计算希尔伯特变换的绝对值
        h_abs = np.abs(h)
        # 计算希尔伯特变换的相位角度
        h_angle = np.angle(h)
        # 计算希尔伯特变换的实部
        h_real = np.real(h)

        # 断言实部应该与原始信号几乎相等
        assert_almost_equal(h_real, a, decimal)
        # 断言绝对值应该几乎全为 1
        assert_almost_equal(h_abs, np.ones(a.shape), decimal)
        # 对于“慢”正弦函数，相位角度应在第一个 256 个数据点内从 -pi/2 到 pi/2 变化
        assert_almost_equal(h_angle[0, :256],
                            np.arange(-pi / 2, pi / 2, pi / 256),
                            decimal)
        # 对于“慢”余弦函数，相位角度应在同一间隔内从 0 到 pi 变化
        assert_almost_equal(
            h_angle[1, :256], np.arange(0, pi, pi / 256), decimal)
        # 对于“快”正弦函数，相位角度应在一半的时间内完成相同的变化
        assert_almost_equal(h_angle[2, :128],
                            np.arange(-pi / 2, pi / 2, pi / 128),
                            decimal)
        # 对于“快”余弦函数，相位角度应在同样的间隔内从 0 到 pi 变化
        assert_almost_equal(
            h_angle[3, :128], np.arange(0, pi, pi / 128), decimal)

        # 断言希尔伯特变换的虚部与原始信号的正弦函数几乎相等
        assert_almost_equal(h[1].imag, a0, decimal)
    # 定义测试函数 test_hilbert_axisN，用于测试 hilbert 函数的 axis 和 N 参数
    def test_hilbert_axisN(self):
        # 创建一个 3x6 的数组 a，包含元素从 0 到 17
        a = np.arange(18).reshape(3, 6)
        
        # 测试 axis 参数，对数组 a 进行 Hilbert 变换，沿着最后一个轴（-1）进行变换
        aa = hilbert(a, axis=-1)
        # 断言：对数组 a 的转置进行 Hilbert 变换，沿着第一个轴（0）进行变换后，应与 aa 的转置相等
        assert_equal(hilbert(a.T, axis=0), aa.T)
        
        # 测试 1 维数组的 Hilbert 变换
        assert_almost_equal(hilbert(a[0]), aa[0], 14)

        # 测试 N 参数为 20，沿着最后一个轴进行 Hilbert 变换
        aan = hilbert(a, N=20, axis=-1)
        # 断言：aan 的形状应为 [3, 20]
        assert_equal(aan.shape, [3, 20])
        # 断言：对数组 a 的转置进行 Hilbert 变换，N 参数为 20，沿着第一个轴进行变换后，应该是形状为 [20, 3]
        assert_equal(hilbert(a.T, N=20, axis=0).shape, [20, 3])
        
        # 下面的测试是一个回归测试，用于检查数值是否合理
        # 注意事项：这些数值不一定是正确的，仅用于回归测试
        a0hilb = np.array([0.000000000000000e+00 - 1.72015830311905j,
                           1.000000000000000e+00 - 2.047794505137069j,
                           1.999999999999999e+00 - 2.244055555687583j,
                           3.000000000000000e+00 - 1.262750302935009j,
                           4.000000000000000e+00 - 1.066489252384493j,
                           5.000000000000000e+00 + 2.918022706971047j,
                           8.881784197001253e-17 + 3.845658908989067j,
                          -9.444121133484362e-17 + 0.985044202202061j,
                          -1.776356839400251e-16 + 1.332257797702019j,
                          -3.996802888650564e-16 + 0.501905089898885j,
                           1.332267629550188e-16 + 0.668696078880782j,
                          -1.192678053963799e-16 + 0.235487067862679j,
                          -1.776356839400251e-16 + 0.286439612812121j,
                           3.108624468950438e-16 + 0.031676888064907j,
                           1.332267629550188e-16 - 0.019275656884536j,
                          -2.360035624836702e-16 - 0.1652588660287j,
                           0.000000000000000e+00 - 0.332049855010597j,
                           3.552713678800501e-16 - 0.403810179797771j,
                           8.881784197001253e-17 - 0.751023775297729j,
                           9.444121133484362e-17 - 0.79252210110103j])
        # 断言：aan 的第一个元素应与 a0hilb 中的元素近似相等，精度为 14 位小数，用于 N 参数的回归测试
        assert_almost_equal(aan[0], a0hilb, 14, 'N regression')

    # 使用 pytest 的参数化装饰器，测试不同的数据类型（dtype）对 hilbert 函数的影响
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_hilbert_types(self, dtype):
        # 创建一个元素全为 0 的长度为 8 的数组，数据类型由参数 dtype 指定
        in_typed = np.zeros(8, dtype=dtype)
        # 断言：对输入数组进行 Hilbert 变换后，得到的实部的数据类型应与输入的 dtype 相同
        assert_equal(np.real(signal.hilbert(in_typed)).dtype, dtype)
class TestHilbert2:
    # 测试处理不良参数情况的方法
    def test_bad_args(self):
        # x 必须是实数
        x = np.array([[1.0 + 0.0j]])
        assert_raises(ValueError, hilbert2, x)

        # x 必须是二维数组
        x = np.arange(24).reshape(2, 3, 4)
        assert_raises(ValueError, hilbert2, x)

        # N 的值不合法
        x = np.arange(16).reshape(4, 4)
        assert_raises(ValueError, hilbert2, x, N=0)
        assert_raises(ValueError, hilbert2, x, N=(2, 0))
        assert_raises(ValueError, hilbert2, x, N=(2,))

    # 测试 hilbert2 函数支持的数据类型
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_hilbert2_types(self, dtype):
        # 创建一个指定数据类型的零矩阵
        in_typed = np.zeros((2, 32), dtype=dtype)
        # 断言经过 hilbert2 处理后的实部数据类型与输入数据类型一致
        assert_equal(np.real(signal.hilbert2(in_typed)).dtype, dtype)


class TestPartialFractionExpansion:
    # 断言实部与极点近似相等的静态方法
    @staticmethod
    def assert_rp_almost_equal(r, p, r_true, p_true, decimal=7):
        # 将 r_true 和 p_true 转换为 numpy 数组
        r_true = np.asarray(r_true)
        p_true = np.asarray(p_true)

        # 计算 r 和 p 的距离
        distance = np.hypot(abs(p[:, None] - p_true),
                            abs(r[:, None] - r_true))

        # 使用匈牙利算法找到最优的行列匹配
        rows, cols = linear_sum_assignment(distance)
        # 断言匹配后的 p 和 r 与真实值 p_true 和 r_true 近似相等
        assert_almost_equal(p[rows], p_true[cols], decimal=decimal)
        assert_almost_equal(r[rows], r_true[cols], decimal=decimal)

    # 测试计算因子的方法
    def test_compute_factors(self):
        # 调用 _compute_factors 方法计算因子和多项式
        factors, poly = _compute_factors([1, 2, 3], [3, 2, 1])
        assert_equal(len(factors), 3)
        assert_almost_equal(factors[0], np.poly([2, 2, 3]))
        assert_almost_equal(factors[1], np.poly([1, 1, 1, 3]))
        assert_almost_equal(factors[2], np.poly([1, 1, 1, 2, 2]))
        assert_almost_equal(poly, np.poly([1, 1, 1, 2, 2, 3]))

        # 调用 _compute_factors 方法计算因子和多项式（包括幂指数）
        factors, poly = _compute_factors([1, 2, 3], [3, 2, 1],
                                         include_powers=True)
        assert_equal(len(factors), 6)
        assert_almost_equal(factors[0], np.poly([1, 1, 2, 2, 3]))
        assert_almost_equal(factors[1], np.poly([1, 2, 2, 3]))
        assert_almost_equal(factors[2], np.poly([2, 2, 3]))
        assert_almost_equal(factors[3], np.poly([1, 1, 1, 2, 3]))
        assert_almost_equal(factors[4], np.poly([1, 1, 1, 3]))
        assert_almost_equal(factors[5], np.poly([1, 1, 1, 2, 2]))
        assert_almost_equal(poly, np.poly([1, 1, 1, 2, 2, 3]))

    # 测试分组极点的方法
    def test_group_poles(self):
        # 调用 _group_poles 方法分组极点
        unique, multiplicity = _group_poles(
            [1.0, 1.001, 1.003, 2.0, 2.003, 3.0], 0.1, 'min')
        assert_equal(unique, [1.0, 2.0, 3.0])
        assert_equal(multiplicity, [3, 2, 1])
    # 定义一个测试方法，用于测试 residue 函数的通用情况
    def test_residue_general(self):
        # 使用 residue 函数计算给定分子和分母的残差(r), 极点(p)和增益(k)
        # 来自 issue #4464 的测试，注意 SciPy 中的极点按绝对值增加的顺序排列，与 MATLAB 相反
        r, p, k = residue([5, 3, -2, 7], [-4, 0, 8, 3])
        # 断言残差的近似值
        assert_almost_equal(r, [1.3320, -0.6653, -1.4167], decimal=4)
        # 断言极点的近似值
        assert_almost_equal(p, [-0.4093, -1.1644, 1.5737], decimal=4)
        # 断言增益的近似值
        assert_almost_equal(k, [-1.2500], decimal=4)

        # 对另一组输入进行测试
        r, p, k = residue([-4, 8], [1, 6, 8])
        assert_almost_equal(r, [8, -12])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)

        # 对更多输入进行类似的测试
        r, p, k = residue([4, 1], [1, -1, -2])
        assert_almost_equal(r, [1, 3])
        assert_almost_equal(p, [-1, 2])
        assert_equal(k.size, 0)

        # 更多的输入测试
        r, p, k = residue([4, 3], [2, -3.4, 1.98, -0.406])
        self.assert_rp_almost_equal(
            r, p, [-18.125 - 13.125j, -18.125 + 13.125j, 36.25],
            [0.5 - 0.2j, 0.5 + 0.2j, 0.7])
        assert_equal(k.size, 0)

        # 更多的输入测试
        r, p, k = residue([2, 1], [1, 5, 8, 4])
        self.assert_rp_almost_equal(r, p, [-1, 1, 3], [-1, -2, -2])
        assert_equal(k.size, 0)

        # 更多的输入测试
        r, p, k = residue([3, -1.1, 0.88, -2.396, 1.348],
                          [1, -0.7, -0.14, 0.048])
        assert_almost_equal(r, [-3, 4, 1])
        assert_almost_equal(p, [0.2, -0.3, 0.8])
        assert_almost_equal(k, [3, 1])

        # 更多的输入测试
        r, p, k = residue([1], [1, 2, -3])
        assert_almost_equal(r, [0.25, -0.25])
        assert_almost_equal(p, [1, -3])
        assert_equal(k.size, 0)

        # 更多的输入测试
        r, p, k = residue([1, 0, -5], [1, 0, 0, 0, -1])
        self.assert_rp_almost_equal(r, p,
                                    [1, 1.5j, -1.5j, -1], [-1, -1j, 1j, 1])
        assert_equal(k.size, 0)

        # 更多的输入测试
        r, p, k = residue([3, 8, 6], [1, 3, 3, 1])
        self.assert_rp_almost_equal(r, p, [1, 2, 3], [-1, -1, -1])
        assert_equal(k.size, 0)

        # 更多的输入测试
        r, p, k = residue([3, -1], [1, -3, 2])
        assert_almost_equal(r, [-2, 5])
        assert_almost_equal(p, [1, 2])
        assert_equal(k.size, 0)

        # 更多的输入测试
        r, p, k = residue([2, 3, -1], [1, -3, 2])
        assert_almost_equal(r, [-4, 13])
        assert_almost_equal(p, [1, 2])
        assert_almost_equal(k, [2])

        # 更多的输入测试
        r, p, k = residue([7, 2, 3, -1], [1, -3, 2])
        assert_almost_equal(r, [-11, 69])
        assert_almost_equal(p, [1, 2])
        assert_almost_equal(k, [7, 23])

        # 最后一组输入测试
        r, p, k = residue([2, 3, -1], [1, -3, 4, -2])
        self.assert_rp_almost_equal(r, p, [4, -1 + 3.5j, -1 - 3.5j],
                                    [1, 1 - 1j, 1 + 1j])
        assert_almost_equal(k.size, 0)
    def test_residue_leading_zeros(self):
        # 测试函数：test_residue_leading_zeros
        # 前导零在分子或分母中不影响结果。
        r0, p0, k0 = residue([5, 3, -2, 7], [-4, 0, 8, 3])
        r1, p1, k1 = residue([0, 5, 3, -2, 7], [-4, 0, 8, 3])
        r2, p2, k2 = residue([5, 3, -2, 7], [0, -4, 0, 8, 3])
        r3, p3, k3 = residue([0, 0, 5, 3, -2, 7], [0, 0, 0, -4, 0, 8, 3])
        # 断言近似相等的结果
        assert_almost_equal(r0, r1)
        assert_almost_equal(r0, r2)
        assert_almost_equal(r0, r3)
        assert_almost_equal(p0, p1)
        assert_almost_equal(p0, p2)
        assert_almost_equal(p0, p3)
        assert_almost_equal(k0, k1)
        assert_almost_equal(k0, k2)
        assert_almost_equal(k0, k3)

    def test_resiude_degenerate(self):
        # 测试函数：test_resiude_degenerate
        # 多个零分子和分母的测试。
        r, p, k = residue([0, 0], [1, 6, 8])
        assert_almost_equal(r, [0, 0])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)

        r, p, k = residue(0, 1)
        assert_equal(r.size, 0)
        assert_equal(p.size, 0)
        assert_equal(k.size, 0)

        # 使用 pytest 检查预期的 ValueError 异常
        with pytest.raises(ValueError, match="Denominator `a` is zero."):
            residue(1, 0)

    def test_residuez_trailing_zeros(self):
        # 测试函数：test_residuez_trailing_zeros
        # 尾随零在分子或分母中不影响结果。
        r0, p0, k0 = residuez([5, 3, -2, 7], [-4, 0, 8, 3])
        r1, p1, k1 = residuez([5, 3, -2, 7, 0], [-4, 0, 8, 3])
        r2, p2, k2 = residuez([5, 3, -2, 7], [-4, 0, 8, 3, 0])
        r3, p3, k3 = residuez([5, 3, -2, 7, 0, 0], [-4, 0, 8, 3, 0, 0, 0])
        # 断言近似相等的结果
        assert_almost_equal(r0, r1)
        assert_almost_equal(r0, r2)
        assert_almost_equal(r0, r3)
        assert_almost_equal(p0, p1)
        assert_almost_equal(p0, p2)
        assert_almost_equal(p0, p3)
        assert_almost_equal(k0, k1)
        assert_almost_equal(k0, k2)
        assert_almost_equal(k0, k3)

    def test_residuez_degenerate(self):
        # 测试函数：test_residuez_degenerate
        r, p, k = residuez([0, 0], [1, 6, 8])
        assert_almost_equal(r, [0, 0])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)

        r, p, k = residuez(0, 1)
        assert_equal(r.size, 0)
        assert_equal(p.size, 0)
        assert_equal(k.size, 0)

        # 使用 pytest 检查预期的 ValueError 异常
        with pytest.raises(ValueError, match="Denominator `a` is zero."):
            residuez(1, 0)

        # 使用 pytest 检查预期的 ValueError 异常
        with pytest.raises(ValueError,
                           match="First coefficient of determinant `a` must "
                                 "be non-zero."):
            residuez(1, [0, 1, 2, 3])
    # 测试不同类型的逆系统函数对唯一根的反转情况
    def test_inverse_unique_roots_different_rtypes(self):
        # 受 GitHub 问题编号 2496 的启发而设计的测试用例
        r = [3 / 10, -1 / 6, -2 / 15]  # 分子多项式的系数列表
        p = [0, -2, -5]  # 分母多项式的系数列表
        k = []  # 系统的直接通道增益列表
        b_expected = [0, 1, 3]  # 预期的分子多项式系数
        a_expected = [1, 7, 10, 0]  # 预期的分母多项式系数

        # 对于这个例子，默认的容差值下，rtype 参数不影响结果
        for rtype in ('avg', 'mean', 'min', 'minimum', 'max', 'maximum'):
            # 调用 invres 函数计算逆系统的分子和分母多项式系数
            b, a = invres(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected)  # 断言分子多项式系数符合预期
            assert_allclose(a, a_expected)  # 断言分母多项式系数符合预期

            # 调用 invresz 函数计算逆系统的分子和分母多项式系数（考虑零点）
            b, a = invresz(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected)  # 断言分子多项式系数符合预期
            assert_allclose(a, a_expected)  # 断言分母多项式系数符合预期

    # 测试不同类型的逆系统函数对重复根的反转情况
    def test_inverse_repeated_roots_different_rtypes(self):
        r = [3 / 20, -7 / 36, -1 / 6, 2 / 45]  # 分子多项式的系数列表
        p = [0, -2, -2, -5]  # 分母多项式的系数列表
        k = []  # 系统的直接通道增益列表
        b_expected = [0, 0, 1, 3]  # 预期的分子多项式系数
        b_expected_z = [-1/6, -2/3, 11/6, 3]  # 考虑零点后的预期分子多项式系数
        a_expected = [1, 9, 24, 20, 0]  # 预期的分母多项式系数

        # 对于这个例子，默认的容差值下，rtype 参数不影响结果
        for rtype in ('avg', 'mean', 'min', 'minimum', 'max', 'maximum'):
            # 调用 invres 函数计算逆系统的分子和分母多项式系数
            b, a = invres(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected, atol=1e-14)  # 断言分子多项式系数符合预期
            assert_allclose(a, a_expected)  # 断言分母多项式系数符合预期

            # 调用 invresz 函数计算逆系统的分子和分母多项式系数（考虑零点）
            b, a = invresz(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected_z, atol=1e-14)  # 断言分子多项式系数符合预期
            assert_allclose(a, a_expected)  # 断言分母多项式系数符合预期

    # 测试传入无效的 rtype 参数时的异常情况
    def test_inverse_bad_rtype(self):
        r = [3 / 20, -7 / 36, -1 / 6, 2 / 45]  # 分子多项式的系数列表
        p = [0, -2, -2, -5]  # 分母多项式的系数列表
        k = []  # 系统的直接通道增益列表
        with pytest.raises(ValueError, match="`rtype` must be one of"):
            invres(r, p, k, rtype='median')  # 检查调用 invres 函数时的异常情况
        with pytest.raises(ValueError, match="`rtype` must be one of"):
            invresz(r, p, k, rtype='median')  # 检查调用 invresz 函数时的异常情况

    # 对于 gh-4646 中问题的回归测试
    def test_invresz_one_coefficient_bug(self):
        r = [1]  # 分子多项式的系数列表
        p = [2]  # 分母多项式的系数列表
        k = [0]  # 系统的直接通道增益列表
        b, a = invresz(r, p, k)  # 计算逆系统的分子和分母多项式系数
        assert_allclose(b, [1.0])  # 断言分子多项式系数符合预期
        assert_allclose(a, [1.0, -2.0])  # 断言分母多项式系数符合预期

    # 测试 invres 函数的不同情况
    def test_invres(self):
        b, a = invres([1], [1], [])  # 计算逆系统的分子和分母多项式系数
        assert_almost_equal(b, [1])  # 断言分子多项式系数符合预期
        assert_almost_equal(a, [1, -1])  # 断言分母多项式系数符合预期

        b, a = invres([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], [])  # 计算逆系统的分子和分母多项式系数
        assert_almost_equal(b, [3.5 - 4j, -8.5 + 0.25j, 3.5 + 3.25j])  # 断言分子多项式系数符合预期
        assert_almost_equal(a, [1, -2 - 1.5j, 0.5 + 2j, 0.5 - 0.5j])  # 断言分母多项式系数符合预期

        b, a = invres([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3])  # 计算逆系统的分子和分母多项式系数
        assert_almost_equal(b, [1, -1 - 1j, 1 - 2j, 0.5 - 3j, 10])  # 断言分子多项式系数符合预期
        assert_almost_equal(a, [1, -3 - 1j, 4])  # 断言分母多项式系数符合预期

        b, a = invres([-1, 2, 1j, 3 - 1j, 4, -2],
                      [-1, 2 - 1j, 2 - 1j, 3, 3, 3], [])  # 计算逆系统的分子和分母多项式系数
        assert_almost_equal(b, [4 - 1j, -28 + 16j, 40 - 62j, 100 + 24j,
                                -292 + 219j, 192 - 268j])  # 断言分子多项式系数符合预期
        assert_almost_equal(a, [1, -12 + 2j, 53 - 20j, -96 + 68j, 27 - 72j,
                                108 - 54j, -81 + 108j])  # 断言分母多项式系数符合预期

        b, a = invres([-1, 1j], [1, 1], [1, 2])  # 计算逆系统的分子和分母多项式系数
    # 定义测试方法 test_invresz，用于测试 invresz 函数的返回值是否符合预期
    def test_invresz(self):
        # 调用 invresz 函数，传入参数 [1], [1], []，获取返回的 b 和 a
        b, a = invresz([1], [1], [])
        # 断言 b 的值与期望的 [1] 几乎相等
        assert_almost_equal(b, [1])
        # 断言 a 的值与期望的 [1, -1] 几乎相等
        assert_almost_equal(a, [1, -1])

        # 再次调用 invresz 函数，传入不同的参数，测试返回值是否符合预期
        b, a = invresz([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], [])
        # 断言 b 的值与期望的 [3.5 - 4j, -8.5 + 0.25j, 3.5 + 3.25j] 几乎相等
        assert_almost_equal(b, [3.5 - 4j, -8.5 + 0.25j, 3.5 + 3.25j])
        # 断言 a 的值与期望的 [1, -2 - 1.5j, 0.5 + 2j, 0.5 - 0.5j] 几乎相等
        assert_almost_equal(a, [1, -2 - 1.5j, 0.5 + 2j, 0.5 - 0.5j])

        # 继续调用 invresz 函数，传入不同的参数，测试返回值是否符合预期
        b, a = invresz([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3])
        # 断言 b 的值与期望的 [2.5, -3 - 1j, 1 - 2j, -1 - 3j, 12] 几乎相等
        assert_almost_equal(b, [2.5, -3 - 1j, 1 - 2j, -1 - 3j, 12])
        # 断言 a 的值与期望的 [1, -3 - 1j, 4] 几乎相等
        assert_almost_equal(a, [1, -3 - 1j, 4])

        # 再次调用 invresz 函数，传入不同的参数，测试返回值是否符合预期
        b, a = invresz([-1, 2, 1j, 3 - 1j, 4, -2],
                       [-1, 2 - 1j, 2 - 1j, 3, 3, 3], [])
        # 断言 b 的值与期望的 [6, -50 + 11j, 100 - 72j, 80 + 58j, -354 + 228j, 234 - 297j] 几乎相等
        assert_almost_equal(b, [6, -50 + 11j, 100 - 72j, 80 + 58j, -354 + 228j, 234 - 297j])
        # 断言 a 的值与期望的 [1, -12 + 2j, 53 - 20j, -96 + 68j, 27 - 72j, 108 - 54j, -81 + 108j] 几乎相等
        assert_almost_equal(a, [1, -12 + 2j, 53 - 20j, -96 + 68j, 27 - 72j, 108 - 54j, -81 + 108j])

        # 继续调用 invresz 函数，传入不同的参数，测试返回值是否符合预期
        b, a = invresz([-1, 1j], [1, 1], [1, 2])
        # 断言 b 的值与期望的 [1j, 1, -3, 2] 几乎相等
        assert_almost_equal(b, [1j, 1, -3, 2])
        # 断言 a 的值与期望的 [1, -2, 1] 几乎相等
        assert_almost_equal(a, [1, -2, 1])

    # 定义测试方法 test_inverse_scalar_arguments，用于测试 invres 和 invresz 函数在传入标量参数时的行为
    def test_inverse_scalar_arguments(self):
        # 调用 invres 函数，传入标量参数 1, 1, 1，获取返回的 b 和 a
        b, a = invres(1, 1, 1)
        # 断言 b 的值与期望的 [1, 0] 几乎相等
        assert_almost_equal(b, [1, 0])
        # 断言 a 的值与期望的 [1, -1] 几乎相等
        assert_almost_equal(a, [1, -1])

        # 调用 invresz 函数，传入标量参数 1, 1, 1，获取返回的 b 和 a
        b, a = invresz(1, 1, 1)
        # 断言 b 的值与期望的 [2, -1] 几乎相等
        assert_almost_equal(b, [2, -1])
        # 断言 a 的值与期望的 [1, -1] 几乎相等
        assert_almost_equal(a, [1, -1])
class TestVectorstrength:

    # 定义测试函数，测试单一1维周期
    def test_single_1dperiod(self):
        # 准备事件数组
        events = np.array([.5])
        # 设置周期
        period = 5.
        # 目标强度和相位
        targ_strength = 1.
        targ_phase = .1

        # 调用被测试函数，计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度应为0
        assert_equal(strength.ndim, 0)
        # 断言相位的维度应为0
        assert_equal(phase.ndim, 0)
        # 断言强度的近似值与目标强度一致
        assert_almost_equal(strength, targ_strength)
        # 断言相位的近似值与目标相位的2π倍一致
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    # 定义测试函数，测试单一2维周期
    def test_single_2dperiod(self):
        events = np.array([.5])
        # 设置多个周期
        period = [1, 2, 5.]
        # 目标强度和相位
        targ_strength = [1.] * 3
        targ_phase = np.array([.5, .25, .1])

        # 调用被测试函数，计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度应为1
        assert_equal(strength.ndim, 1)
        # 断言相位的维度应为1
        assert_equal(phase.ndim, 1)
        # 断言强度数组的近似值与目标强度数组一致
        assert_array_almost_equal(strength, targ_strength)
        # 断言相位数组的近似值与目标相位数组的2π倍一致
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    # 定义测试函数，测试相等的1维周期
    def test_equal_1dperiod(self):
        events = np.array([.25, .25, .25, .25, .25, .25])
        # 设置周期
        period = 2
        # 目标强度和相位
        targ_strength = 1.
        targ_phase = .125

        # 调用被测试函数，计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度应为0
        assert_equal(strength.ndim, 0)
        # 断言相位的维度应为0
        assert_equal(phase.ndim, 0)
        # 断言强度的近似值与目标强度一致
        assert_almost_equal(strength, targ_strength)
        # 断言相位的近似值与目标相位的2π倍一致
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    # 定义测试函数，测试相等的2维周期
    def test_equal_2dperiod(self):
        events = np.array([.25, .25, .25, .25, .25, .25])
        # 设置多个周期
        period = [1, 2, ]
        # 目标强度和相位
        targ_strength = [1.] * 2
        targ_phase = np.array([.25, .125])

        # 调用被测试函数，计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度应为1
        assert_equal(strength.ndim, 1)
        # 断言相位的维度应为1
        assert_equal(phase.ndim, 1)
        # 断言强度数组的近似值与目标强度数组一致
        assert_almost_equal(strength, targ_strength)
        # 断言相位数组的近似值与目标相位数组的2π倍一致
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    # 定义测试函数，测试间隔的1维周期
    def test_spaced_1dperiod(self):
        events = np.array([.1, 1.1, 2.1, 4.1, 10.1])
        # 设置周期
        period = 1
        # 目标强度和相位
        targ_strength = 1.
        targ_phase = .1

        # 调用被测试函数，计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度应为0
        assert_equal(strength.ndim, 0)
        # 断言相位的维度应为0
        assert_equal(phase.ndim, 0)
        # 断言强度的近似值与目标强度一致
        assert_almost_equal(strength, targ_strength)
        # 断言相位的近似值与目标相位的2π倍一致
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    # 定义测试函数，测试间隔的2维周期
    def test_spaced_2dperiod(self):
        events = np.array([.1, 1.1, 2.1, 4.1, 10.1])
        # 设置多个周期
        period = [1, .5]
        # 目标强度和相位
        targ_strength = [1.] * 2
        targ_phase = np.array([.1, .2])

        # 调用被测试函数，计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度应为1
        assert_equal(strength.ndim, 1)
        # 断言相位的维度应为1
        assert_equal(phase.ndim, 1)
        # 断言强度数组的近似值与目标强度数组一致
        assert_almost_equal(strength, targ_strength)
        # 断言相位数组的近似值与目标相位数组的2π倍一致
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    # 定义测试函数，测试部分的1维周期
    def test_partial_1dperiod(self):
        events = np.array([.25, .5, .75])
        # 设置周期
        period = 1
        # 目标强度和相位
        targ_strength = 1. / 3.
        targ_phase = .5

        # 调用被测试函数，计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度应为0
        assert_equal(strength.ndim, 0)
        # 断言相位的维度应为0
        assert_equal(phase.ndim, 0)
        # 断言强度的近似值与目标强度一致
        assert_almost_equal(strength, targ_strength)
        # 断言相位的近似值与目标相位的2π倍一致
        assert_almost_equal(phase, 2 * np.pi * targ_phase)
    # 测试函数，用于测试 vectorstrength 函数在处理部分二维周期时的行为
    def test_partial_2dperiod(self):
        # 创建事件数组，包含三个时间点
        events = np.array([.25, .5, .75])
        # 设定周期为一维数组，长度为4
        period = [1., 1., 1., 1.]
        # 目标强度为每个周期的事件数量的倒数
        targ_strength = [1. / 3.] * 4
        # 目标相位为二维数组，每个元素乘以2π
        targ_phase = np.array([.5, .5, .5, .5])

        # 调用 vectorstrength 函数计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度为1维
        assert_equal(strength.ndim, 1)
        # 断言相位的维度为1维
        assert_equal(phase.ndim, 1)
        # 断言强度近似等于目标强度
        assert_almost_equal(strength, targ_strength)
        # 断言相位近似等于目标相位乘以2π
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    # 测试函数，用于测试 vectorstrength 函数在处理一维周期的相反情况时的行为
    def test_opposite_1dperiod(self):
        # 创建事件数组，包含四个时间点
        events = np.array([0, .25, .5, .75])
        # 设定周期为标量值1
        period = 1.
        # 目标强度为0
        targ_strength = 0

        # 调用 vectorstrength 函数计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度为0维
        assert_equal(strength.ndim, 0)
        # 断言相位的维度为0维
        assert_equal(phase.ndim, 0)
        # 断言强度近似等于目标强度
        assert_almost_equal(strength, targ_strength)

    # 测试函数，用于测试 vectorstrength 函数在处理二维周期的相反情况时的行为
    def test_opposite_2dperiod(self):
        # 创建事件数组，包含四个时间点
        events = np.array([0, .25, .5, .75])
        # 设定周期为包含10个1的一维数组
        period = [1.] * 10
        # 目标强度为包含10个0的一维数组
        targ_strength = [0.] * 10

        # 调用 vectorstrength 函数计算强度和相位
        strength, phase = vectorstrength(events, period)

        # 断言强度的维度为1维
        assert_equal(strength.ndim, 1)
        # 断言相位的维度为1维
        assert_equal(phase.ndim, 1)
        # 断言强度近似等于目标强度
        assert_almost_equal(strength, targ_strength)

    # 测试函数，用于测试 vectorstrength 函数在处理二维事件数组时是否引发 ValueError 异常
    def test_2d_events_ValueError(self):
        # 创建二维事件数组
        events = np.array([[1, 2]])
        # 设定周期为标量值1
        period = 1.
        # 断言调用 vectorstrength 函数时会引发 ValueError 异常
        assert_raises(ValueError, vectorstrength, events, period)

    # 测试函数，用于测试 vectorstrength 函数在处理二维周期数组时是否引发 ValueError 异常
    def test_2d_period_ValueError(self):
        # 创建事件数组，包含一个标量值
        events = 1.
        # 设定周期为包含一个二维数组的标量
        period = np.array([[1]])
        # 断言调用 vectorstrength 函数时会引发 ValueError 异常
        assert_raises(ValueError, vectorstrength, events, period)

    # 测试函数，用于测试 vectorstrength 函数在处理周期为0时是否引发 ValueError 异常
    def test_zero_period_ValueError(self):
        # 创建事件数组，包含一个标量值
        events = 1.
        # 设定周期为0
        period = 0
        # 断言调用 vectorstrength 函数时会引发 ValueError 异常
        assert_raises(ValueError, vectorstrength, events, period)

    # 测试函数，用于测试 vectorstrength 函数在处理负周期时是否引发 ValueError 异常
    def test_negative_period_ValueError(self):
        # 创建事件数组，包含一个标量值
        events = 1.
        # 设定周期为负数
        period = -1
        # 断言调用 vectorstrength 函数时会引发 ValueError 异常
        assert_raises(ValueError, vectorstrength, events, period)
# 定义一个函数，用于比较两个数组的元素是否接近，同时支持对象数组的数据类型转换
def assert_allclose_cast(actual, desired, rtol=1e-7, atol=0):
    """Wrap assert_allclose while casting object arrays."""
    # 如果 actual 是对象数组，则将其转换为第一个元素的数据类型
    if actual.dtype.kind == 'O':
        dtype = np.array(actual.flat[0]).dtype
        actual, desired = actual.astype(dtype), desired.astype(dtype)
    # 使用 assert_allclose 函数比较两个数组的接近程度
    assert_allclose(actual, desired, rtol, atol)


# 使用 pytest.mark.parametrize 标记，为函数 func 参数化测试
@pytest.mark.parametrize('func', (sosfilt, lfilter))
def test_nonnumeric_dtypes(func):
    # 定义 Decimal 类型的数组
    x = [Decimal(1), Decimal(2), Decimal(3)]
    b = [Decimal(1), Decimal(2), Decimal(3)]
    a = [Decimal(1), Decimal(2), Decimal(3)]
    # 将数组 x 转换为 NumPy 数组
    x = np.array(x)
    # 断言 x 的数据类型为对象数组
    assert x.dtype.kind == 'O'
    # 将 x 转换为 float 类型后，使用 lfilter 函数计算 desired 数组
    desired = lfilter(np.array(b, float), np.array(a, float), x.astype(float))
    # 根据 func 的不同，选择 sosfilt 或 lfilter 函数计算 actual 数组
    if func is sosfilt:
        actual = sosfilt([b + a], x)
    else:
        actual = lfilter(b, a, x)
    # 断言 actual 数组中的所有元素都是 Decimal 类型
    assert all(isinstance(x, Decimal) for x in actual)
    # 使用 assert_allclose_cast 函数比较 actual 和 desired 数组的接近程度
    assert_allclose_cast(actual.astype(float), desired.astype(float))
    # 处理特殊情况
    if func is lfilter:
        args = [1., 1.]
    else:
        args = [tf2sos(1., 1.)]
    # 使用 pytest.raises 检查是否抛出 ValueError 异常，异常消息要求包含 'must be at least 1-D'
    with pytest.raises(ValueError, match='must be at least 1-D'):
        func(*args, x=1.)


# 使用 pytest.mark.parametrize 标记，为 dt 参数化测试类 TestSOSFilt
@pytest.mark.parametrize('dt', 'fdFD')
class TestSOSFilt:

    # 从 _TestLinearFilter 中获取 test_rank* 测试用例
    def test_rank1(self, dt):
        # 创建一个 dt 类型的线性空间数组 x
        x = np.linspace(0, 5, 6).astype(dt)
        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, -0.5]).astype(dt)

        # 测试简单 IIR 滤波器
        y_r = np.array([0, 2, 4, 6, 8, 10.]).astype(dt)
        sos = tf2sos(b, a)
        # 断言使用 sosfilt 函数对 x 进行滤波后结果与 y_r 接近
        assert_array_almost_equal(sosfilt(tf2sos(b, a), x), y_r)

        # 测试简单 FIR 滤波器
        b = np.array([1, 1]).astype(dt)
        # 注意：相对于 TestLinear... 添加了零极点:
        a = np.array([1, 0]).astype(dt)
        y_r = np.array([0, 1, 3, 5, 7, 9.]).astype(dt)
        # 断言使用 sosfilt 函数对 x 进行滤波后结果与 y_r 接近
        assert_array_almost_equal(sosfilt(tf2sos(b, a), x), y_r)

        # 额外的情况
        b = [1, 1, 0]
        a = [1, 0, 0]
        x = np.ones(8)
        sos = np.concatenate((b, a))
        sos.shape = (1, 6)
        # 使用 sosfilt 函数进行滤波
        y = sosfilt(sos, x)
        # 断言 y 与预期结果接近
        assert_allclose(y, [1, 2, 2, 2, 2, 2, 2, 2])

    # 添加关于 dt 参数化测试的 test_rank2 测试用例
    def test_rank2(self, dt):
        shape = (4, 3)
        # 创建一个 dt 类型的线性空间数组 x
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        x = x.astype(dt)

        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, 0.5]).astype(dt)

        y_r2_a0 = np.array([[0, 2, 4], [6, 4, 2], [0, 2, 4], [6, 4, 2]],
                           dtype=dt)

        y_r2_a1 = np.array([[0, 2, 0], [6, -4, 6], [12, -10, 12],
                            [18, -16, 18]], dtype=dt)

        # 使用 sosfilt 函数对 x 进行轴向滤波，与预期结果 y_r2_a0 和 y_r2_a1 比较接近
        y = sosfilt(tf2sos(b, a), x, axis=0)
        assert_array_almost_equal(y_r2_a0, y)

        y = sosfilt(tf2sos(b, a), x, axis=1)
        assert_array_almost_equal(y_r2_a1, y)
    # 定义一个测试函数，用于测试三维数组的滤波操作
    def test_rank3(self, dt):
        # 定义一个形状为 (4, 3, 2) 的数组
        shape = (4, 3, 2)
        # 生成一个线性空间，从 0 到 shape 元素总数减 1，然后重新形状为 shape
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)

        # 定义两个系数向量 b 和 a，并将其类型转换为指定的数据类型 dt
        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, 0.5]).astype(dt)

        # 使用 tf2sos 函数将 b 和 a 转换为二阶段二阶段序列 (Second-Order-Sections, SOS)，然后进行 sosfilt 滤波
        y = sosfilt(tf2sos(b, a), x)

        # 循环遍历数组 x 的前两个维度，对每个元素进行 lfilter 滤波，并使用 assert_array_almost_equal 进行断言
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                assert_array_almost_equal(y[i, j], lfilter(b, a, x[i, j]))

    # 定义一个测试函数，用于测试滤波器的初始条件
    def test_initial_conditions(self, dt):
        # 使用 butter 函数生成两个二阶低通滤波器的系数 b 和 a
        b1, a1 = signal.butter(2, 0.25, 'low')
        b2, a2 = signal.butter(2, 0.75, 'low')
        b3, a3 = signal.butter(2, 0.75, 'low')
        
        # 将三个滤波器的系数 b 和 a 进行卷积，得到总的 b 和 a
        b = np.convolve(np.convolve(b1, b2), b3)
        a = np.convolve(np.convolve(a1, a2), a3)
        
        # 将 b1, b2, b3 和 a1, a2, a3 组成 SOS (Second-Order-Sections) 格式
        sos = np.array((np.r_[b1, a1], np.r_[b2, a2], np.r_[b3, a3]))

        # 生成一个长度为 50 的随机数组，并将其类型转换为 dt
        x = np.random.rand(50).astype(dt)

        # 使用 lfilter 函数进行滤波，验证预期输出 y_true 与 lfilter 的输出结果 y_true 的接近程度
        y_true, zi = lfilter(b, a, x[:20], zi=np.zeros(6))
        y_true = np.r_[y_true, lfilter(b, a, x[20:], zi=zi)[0]]
        assert_allclose_cast(y_true, lfilter(b, a, x))

        # 使用 sosfilt 函数进行滤波，验证预期输出 y_true 与 y_sos 的接近程度
        y_sos, zi = sosfilt(sos, x[:20], zi=np.zeros((3, 2)))
        y_sos = np.r_[y_sos, sosfilt(sos, x[20:], zi=zi)[0]]
        assert_allclose_cast(y_true, y_sos)

        # 使用步函数生成输入数据 x，然后使用 sosfilt 函数进行滤波，验证输出结果 y 和初始条件 zf 的接近程度
        zi = sosfilt_zi(sos)
        x = np.ones(8, dt)
        y, zf = sosfilt(sos, x, zi=zi)
        assert_allclose_cast(y, np.ones(8))
        assert_allclose_cast(zf, zi)

        # 改变数组 x 的形状为三维，然后验证传入的 zi 与形状匹配时是否引发 ValueError 异常
        x.shape = (1, 1) + x.shape  # 3D
        assert_raises(ValueError, sosfilt, sos, x, zi=zi)
        
        # 复制一份 zi，改变其形状以验证是否引发 ValueError 异常
        zi_nd = zi.copy()
        zi_nd.shape = (zi.shape[0], 1, 1, zi.shape[-1])
        assert_raises(ValueError, sosfilt, sos, x,
                      zi=zi_nd[:, :, :, [0, 1, 1]])
        
        # 使用改变形状后的 zi 进行滤波，验证输出结果 y 和 zf 的接近程度
        y, zf = sosfilt(sos, x, zi=zi_nd)
        assert_allclose_cast(y[0, 0], np.ones(8))
        assert_allclose_cast(zf[:, 0, 0, :], zi)
    # 测试在三维输入的第一个轴上应用 sosfilt 时 zi 参数的使用情况。

    # 创建随机整数数组 x，形状为 (2, 15, 3)，类型为 dt
    x = np.random.RandomState(159).randint(0, 5, size=(2, 15, 3))
    x = x.astype(dt)

    # 设计一个 ZPK 格式的滤波器，并将其转换为 SOS 形式
    zpk = signal.butter(6, 0.35, output='zpk')
    sos = zpk2sos(*zpk)
    nsections = sos.shape[0]

    # 指定滤波器作用的轴
    axis = 1

    # 初始化条件，全部为零
    shp = list(x.shape)
    shp[axis] = 2
    shp = [nsections] + shp
    z0 = np.zeros(shp)

    # 对 x 应用滤波器
    yf, zf = sosfilt(sos, x, axis=axis, zi=z0)

    # 分两阶段对 x 应用滤波器
    y1, z1 = sosfilt(sos, x[:, :5, :], axis=axis, zi=z0)
    y2, z2 = sosfilt(sos, x[:, 5:, :], axis=axis, zi=z1)

    # y 应等于 yf，z2 应等于 zf
    y = np.concatenate((y1, y2), axis=axis)
    assert_allclose_cast(y, yf, rtol=1e-10, atol=1e-13)
    assert_allclose_cast(z2, zf, rtol=1e-10, atol=1e-13)

    # 尝试使用 "step" 初始条件
    zi = sosfilt_zi(sos)
    zi.shape = [nsections, 1, 2, 1]
    zi = zi * x[:, 0:1, :]
    y = sosfilt(sos, x, axis=axis, zi=zi)[0]

    # 与 TF 形式进行比较
    b, a = zpk2tf(*zpk)
    zi = lfilter_zi(b, a)
    zi.shape = [1, zi.size, 1]
    zi = zi * x[:, 0:1, :]
    y_tf = lfilter(b, a, x, axis=axis, zi=zi)[0]
    assert_allclose_cast(y, y_tf, rtol=1e-10, atol=1e-13)


# 测试当 zi 的形状不正确时的行为
def test_bad_zi_shape(self, dt):
    # 在使用参数值之前检查 zi 的形状，因此使用 np.empty 创建参数是可以接受的。

    # 创建一个形状为 (3, 15, 3) 的空数组 x，类型为 dt
    x = np.empty((3, 15, 3), dt)
    
    # 创建一个形状为 (4, 3, 3, 2) 的空数组 zi，正确的形状应为 (4, 3, 2, 3)
    zi = np.empty((4, 3, 3, 2))

    # 断言应该引发 ValueError，匹配错误信息 'should be all ones'
    with pytest.raises(ValueError, match='should be all ones'):
        sosfilt(sos, x, zi=zi, axis=1)

    # 修改数组 sos 的一列为 1.0
    sos[:, 3] = 1.

    # 断言应该引发 ValueError，匹配错误信息 'Invalid zi shape'
    with pytest.raises(ValueError, match='Invalid zi shape'):
        sosfilt(sos, x, zi=zi, axis=1)


# 测试 sosfilt_zi 函数的行为
def test_sosfilt_zi(self, dt):
    # 获取使用 butter 方法创建的 sos 滤波器
    sos = signal.butter(6, 0.2, output='sos')

    # 获取 sosfilt_zi 函数生成的初始状态 zi
    zi = sosfilt_zi(sos)

    # 对全为 1 的数组应用滤波器，得到输出 y 和最终状态 zf
    y, zf = sosfilt(sos, np.ones(40, dt), zi=zi)
    assert_allclose_cast(zf, zi, rtol=1e-13)

    # 预期该滤波器的阶跃响应稳态值
    ss = np.prod(sos[:, :3].sum(axis=-1) / sos[:, 3:].sum(axis=-1))
    assert_allclose_cast(y, ss, rtol=1e-13)

    # 将 zi 作为类数组
    _, zf = sosfilt(sos, np.ones(40, dt), zi=zi.tolist())
    assert_allclose_cast(zf, zi, rtol=1e-13)
class TestDeconvolve:
    
    def test_basic(self):
        # From docstring example
        # 定义原始信号、脉冲响应和记录信号
        original = [0, 1, 0, 0, 1, 1, 0, 0]
        impulse_response = [2, 1]
        recorded = [0, 2, 1, 0, 2, 3, 1, 0, 0]
        # 对记录信号进行反卷积，返回恢复的信号和余数
        recovered, remainder = signal.deconvolve(recorded, impulse_response)
        # 断言恢复的信号与原始信号几乎相等
        assert_allclose(recovered, original)
    
    def test_n_dimensional_signal(self):
        # 对于多维信号，测试异常情况：信号必须是一维的
        recorded = [[0, 0], [0, 0]]
        impulse_response = [0, 0]
        with pytest.raises(ValueError, match="signal must be 1-D."):
            quotient, remainder = signal.deconvolve(recorded, impulse_response)
    
    def test_n_dimensional_divisor(self):
        # 对于多维除数，测试异常情况：除数必须是一维的
        recorded = [0, 0]
        impulse_response = [[0, 0], [0, 0]]
        with pytest.raises(ValueError, match="divisor must be 1-D."):
            quotient, remainder = signal.deconvolve(recorded, impulse_response)


class TestDetrend:

    def test_basic(self):
        # 基本测试：去趋势化一维数组
        detrended = detrend(array([1, 2, 3]))
        detrended_exact = array([0, 0, 0])
        # 断言去趋势化后的结果与期望的完全相等
        assert_array_almost_equal(detrended, detrended_exact)

    def test_copy(self):
        # 复制和就地操作的比较
        x = array([1, 1.2, 1.5, 1.6, 2.4])
        # 非就地操作的去趋势化
        copy_array = detrend(x, overwrite_data=False)
        # 就地操作的去趋势化
        inplace = detrend(x, overwrite_data=True)
        # 断言两种操作的结果几乎相等
        assert_array_almost_equal(copy_array, inplace)

    @pytest.mark.parametrize('kind', ['linear', 'constant'])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    def test_axis(self, axis, kind):
        # 测试不同类型和轴向的去趋势化操作
        data = np.arange(5*6*7).reshape(5, 6, 7)
        # 执行去趋势化操作
        detrended = detrend(data, type=kind, axis=axis)
        # 断言去趋势化后的形状与原始数据相同
        assert detrended.shape == data.shape

    def test_bp(self):
        # 测试带断点的线性去趋势化
        data = [0, 1, 2] + [5, 0, -5, -10]
        # 执行带断点的线性去趋势化
        detrended = detrend(data, type='linear', bp=3)
        # 断言去趋势化后的结果几乎为0
        assert_allclose(detrended, 0, atol=1e-14)

        # 对于多维数据和指定轴的测试
        data = np.asarray(data)[None, :, None]
        detrended = detrend(data, type="linear", bp=3, axis=1)
        assert_allclose(detrended, 0, atol=1e-14)

        # 断点索引大于轴的形状：应引发异常
        with assert_raises(ValueError):
            detrend(data, type="linear", bp=3)

    @pytest.mark.parametrize('bp', [np.array([0, 2]), [0, 2]])
    def test_detrend_array_bp(self, bp):
        # 用于回归测试的断点数组测试
        # regression test for https://github.com/scipy/scipy/issues/18675
        rng = np.random.RandomState(12345)
        x = rng.rand(10)
        # 断点数组 bp 的测试
        res = detrend(x, bp=bp)
        res_scipy_191 = np.array([-4.44089210e-16, -2.22044605e-16,
            -1.11128506e-01, -1.69470553e-01,  1.14710683e-01,  6.35468419e-02,
            3.53533144e-01, -3.67877935e-02, -2.00417675e-02, -1.94362049e-01])
        # 断言结果与预期非常接近
        assert_allclose(res, res_scipy_191, atol=1e-14)


class TestUniqueRoots:
    
    def test_real_no_repeat(self):
        # 测试实数根且无重复的情况
        p = [-1.0, -0.5, 0.3, 1.2, 10.0]
        unique, multiplicity = unique_roots(p)
        # 断言唯一根和多重性与输入几乎相等
        assert_almost_equal(unique, p, decimal=15)
        assert_equal(multiplicity, np.ones(len(p)))
    def test_real_repeat(self):
        # 定义测试用例中的多项式系数列表
        p = [-1.0, -0.95, -0.89, -0.8, 0.5, 1.0, 1.05]

        # 测试 unique_roots 函数，rtype='min' 模式下的唯一根和重数
        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='min')
        # 断言唯一根的近似值
        assert_almost_equal(unique, [-1.0, -0.89, 0.5, 1.0], decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [2, 2, 1, 2])

        # 测试 rtype='max' 模式下的唯一根和重数
        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='max')
        # 断言唯一根的近似值
        assert_almost_equal(unique, [-0.95, -0.8, 0.5, 1.05], decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [2, 2, 1, 2])

        # 测试 rtype='avg' 模式下的唯一根和重数
        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='avg')
        # 断言唯一根的近似值
        assert_almost_equal(unique, [-0.975, -0.845, 0.5, 1.025], decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [2, 2, 1, 2])

    def test_complex_no_repeat(self):
        # 定义测试用例中的复数根列表
        p = [-1.0, 1.0j, 0.5 + 0.5j, -1.0 - 1.0j, 3.0 + 2.0j]
        # 测试 unique_roots 函数，不指定参数时的唯一根和重数
        unique, multiplicity = unique_roots(p)
        # 断言唯一根的近似值
        assert_almost_equal(unique, p, decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, np.ones(len(p)))

    def test_complex_repeat(self):
        # 定义测试用例中的复数根列表
        p = [-1.0, -1.0 + 0.05j, -0.95 + 0.15j, -0.90 + 0.15j, 0.0,
             0.5 + 0.5j, 0.45 + 0.55j]

        # 测试 rtype='min' 模式下的唯一根和重数
        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='min')
        # 断言唯一根的近似值
        assert_almost_equal(unique, [-1.0, -0.95 + 0.15j, 0.0, 0.45 + 0.55j],
                            decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [2, 2, 1, 2])

        # 测试 rtype='max' 模式下的唯一根和重数
        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='max')
        # 断言唯一根的近似值
        assert_almost_equal(unique,
                            [-1.0 + 0.05j, -0.90 + 0.15j, 0.0, 0.5 + 0.5j],
                            decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [2, 2, 1, 2])

        # 测试 rtype='avg' 模式下的唯一根和重数
        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='avg')
        # 断言唯一根的近似值
        assert_almost_equal(
            unique, [-1.0 + 0.025j, -0.925 + 0.15j, 0.0, 0.475 + 0.525j],
            decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [2, 2, 1, 2])

    def test_gh_4915(self):
        # 计算具有给定系数的多项式的根
        p = np.roots(np.convolve(np.ones(5), np.ones(5)))
        # 预期的真实根
        true_roots = [-(-1)**(1/5), (-1)**(4/5), -(-1)**(3/5), (-1)**(2/5)]

        # 测试 unique_roots 函数的返回值
        unique, multiplicity = unique_roots(p)
        # 对唯一根进行排序
        unique = np.sort(unique)

        # 断言排序后的唯一根近似等于真实根
        assert_almost_equal(np.sort(unique), true_roots, decimal=7)
        # 断言根的重数
        assert_equal(multiplicity, [2, 2, 2, 2])

    def test_complex_roots_extra(self):
        # 测试具有复数根的情况
        unique, multiplicity = unique_roots([1.0, 1.0j, 1.0])
        # 断言唯一根的近似值
        assert_almost_equal(unique, [1.0, 1.0j], decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [2, 1])

        # 测试具有浮点数根的情况，并指定公差参数
        unique, multiplicity = unique_roots([1, 1 + 2e-9, 1e-9 + 1j], tol=0.1)
        # 断言唯一根的近似值
        assert_almost_equal(unique, [1.0, 1e-9 + 1.0j], decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [2, 1])

    def test_single_unique_root(self):
        # 生成一个具有随机复数根的多项式
        p = np.random.rand(100) + 1j * np.random.rand(100)
        # 测试 unique_roots 函数，期望返回单一唯一根和其重数为100
        unique, multiplicity = unique_roots(p, 2)
        # 断言唯一根的近似值
        assert_almost_equal(unique, [np.min(p)], decimal=15)
        # 断言根的重数
        assert_equal(multiplicity, [100])
```