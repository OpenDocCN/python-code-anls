# `D:\src\scipysrc\scipy\scipy\signal\tests\test_upfirdn.py`

```
# 导入必要的库
import numpy as np
from itertools import product  # 导入 itertools 库中的 product 函数

from numpy.testing import assert_equal, assert_allclose  # 从 numpy.testing 中导入断言函数 assert_equal 和 assert_allclose
from pytest import raises as assert_raises  # 从 pytest 中导入 raises 函数并重命名为 assert_raises
import pytest  # 导入 pytest 库

from scipy.signal import upfirdn, firwin  # 从 scipy.signal 中导入 upfirdn 和 firwin 函数
from scipy.signal._upfirdn import _output_len, _upfirdn_modes  # 导入 scipy.signal._upfirdn 中的 _output_len 和 _upfirdn_modes 函数
from scipy.signal._upfirdn_apply import _pad_test  # 导入 scipy.signal._upfirdn_apply 中的 _pad_test 函数


class UpFIRDnCase:
    """Test _UpFIRDn object"""
    def __init__(self, up, down, h, x_dtype):
        # 初始化 UpFIRDnCase 类的实例
        self.up = up  # 设置上采样因子
        self.down = down  # 设置下采样因子
        self.h = np.atleast_1d(h)  # 确保 h 至少是一维数组
        self.x_dtype = x_dtype  # 设置输入信号的数据类型
        self.rng = np.random.RandomState(17)  # 使用种子为 17 初始化随机数生成器
    # 调用对象的特殊方法，用于执行对象的实例
    def __call__(self):
        # 调用 scrub 方法，传入一个包含单个元素的数组，数据类型为 self.x_dtype
        self.scrub(np.ones(1, self.x_dtype))
        # 调用 scrub 方法，传入一个包含十个元素的数组，数据类型为 self.x_dtype，全为 1
        self.scrub(np.ones(10, self.x_dtype))  # ones
        # 生成一个包含十个随机数的数组 x，数据类型为 self.x_dtype
        x = self.rng.randn(10).astype(self.x_dtype)
        # 如果数据类型是复数类型，则添加虚部，虚部随机生成
        if self.x_dtype in (np.complex64, np.complex128):
            x += 1j * self.rng.randn(10)
        # 调用 scrub 方法，传入数组 x
        self.scrub(x)
        # 调用 scrub 方法，传入一个包含十个元素的等差数列数组，数据类型为 self.x_dtype
        self.scrub(np.arange(10).astype(self.x_dtype))
        # 生成一个三维数组 x，形状为 (2, 3, 5)，数据类型为 self.x_dtype，内容为随机数
        size = (2, 3, 5)
        x = self.rng.randn(*size).astype(self.x_dtype)
        # 如果数据类型是复数类型，则添加虚部，虚部随机生成
        if self.x_dtype in (np.complex64, np.complex128):
            x += 1j * self.rng.randn(*size)
        # 遍历数组 x 的每个轴，依次调用 scrub 方法
        for axis in range(len(size)):
            self.scrub(x, axis=axis)
        # 对数组 x 进行切片操作，然后再次遍历每个轴，依次调用 scrub 方法
        x = x[:, ::2, 1::3].T
        for axis in range(len(size)):
            self.scrub(x, axis=axis)

    # 定义一个方法 scrub，用于处理输入数组 x，进行特定的处理操作
    def scrub(self, x, axis=-1):
        # 对数组 x 进行一维卷积操作，使用自定义函数 upfirdn_naive
        yr = np.apply_along_axis(upfirdn_naive, axis, x,
                                 self.h, self.up, self.down)
        # 计算预期的输出长度
        want_len = _output_len(len(self.h), x.shape[axis], self.up, self.down)
        # 断言 yr 的特定轴的长度符合预期长度
        assert yr.shape[axis] == want_len
        # 对数组 x 进行一维卷积操作，使用 upfirdn 函数
        y = upfirdn(self.h, x, self.up, self.down, axis=axis)
        # 断言 y 的特定轴的长度符合预期长度
        assert y.shape[axis] == want_len
        # 断言 y 和 yr 的形状完全相同
        assert y.shape == yr.shape
        # 检查 self.h 和 x 的数据类型，根据类型进行断言
        dtypes = (self.h.dtype, x.dtype)
        if all(d == np.complex64 for d in dtypes):
            assert_equal(y.dtype, np.complex64)
        elif np.complex64 in dtypes and np.float32 in dtypes:
            assert_equal(y.dtype, np.complex64)
        elif all(d == np.float32 for d in dtypes):
            assert_equal(y.dtype, np.float32)
        elif np.complex128 in dtypes or np.complex64 in dtypes:
            assert_equal(y.dtype, np.complex128)
        else:
            assert_equal(y.dtype, np.float64)
        # 断言 yr 和 y 的所有元素在容忍范围内相等
        assert_allclose(yr, y)
# 定义一个元组，包含用于输入验证的数据类型，包括整数、32位浮点数、64位复数、普通浮点数和复数
_UPFIRDN_TYPES = (int, np.float32, np.complex64, float, complex)

# 定义一个测试类 TestUpfirdn
class TestUpfirdn:

    # 测试无效输入情况
    def test_valid_input(self):
        # 断言：当上采样或下采样因子小于1时，抛出 ValueError 异常
        assert_raises(ValueError, upfirdn, [1], [1], 1, 0)
        # 断言：当滤波器系数 h 的维度不为1时，抛出 ValueError 异常
        assert_raises(ValueError, upfirdn, [], [1], 1, 1)
        # 断言：当滤波器系数 h 不是一维数组时，抛出 ValueError 异常
        assert_raises(ValueError, upfirdn, [[1]], [1], 1, 1)

    # 参数化测试：测试单元素情况
    @pytest.mark.parametrize('len_h', [1, 2, 3, 4, 5])
    @pytest.mark.parametrize('len_x', [1, 2, 3, 4, 5])
    def test_singleton(self, len_h, len_x):
        # gh-9844: 测试产生预期输出的长度情况
        h = np.zeros(len_h)
        h[len_h // 2] = 1.  # 将 h 设为单位冲击响应
        x = np.ones(len_x)
        y = upfirdn(h, x, 1, 1)
        want = np.pad(x, (len_h // 2, (len_h - 1) // 2), 'constant')
        assert_allclose(y, want)

    # 测试：移位 x 是否会改变值
    def test_shift_x(self):
        # gh-9844: 移位后的 x 是否会改变值
        y = upfirdn([1, 1], [1.], 1, 1)
        assert_allclose(y, [1, 1])  # 在问题中为 [0, 1]
        y = upfirdn([1, 1], [0., 1.], 1, 1)
        assert_allclose(y, [0, 1, 1])

    # 参数化测试：测试长度和因子对
    @pytest.mark.parametrize('len_h, len_x, up, down, expected', [
        (2, 2, 5, 2, [1, 0, 0, 0]),
        (2, 3, 6, 3, [1, 0, 1, 0, 1]),
        (2, 4, 4, 3, [1, 0, 0, 0, 1]),
        (3, 2, 6, 2, [1, 0, 0, 1, 0]),
        (4, 11, 3, 5, [1, 0, 0, 1, 0, 0, 1]),
    ])
    def test_length_factors(self, len_h, len_x, up, down, expected):
        # gh-9844: 测试奇异因子
        h = np.zeros(len_h)
        h[0] = 1.
        x = np.ones(len_x)
        y = upfirdn(h, x, up, down)
        assert_allclose(y, expected)

    # 参数化测试：测试与 MATLAB 预期结果比较的下采样情况
    @pytest.mark.parametrize('down, want_len', [
        (2, 5015),
        (11, 912),
        (79, 127),
    ])
    def test_vs_convolve(self, down, want_len):
        # 检查当 up=1.0 时，是否与 convolve + slicing 的结果相同
        random_state = np.random.RandomState(17)
        # 尝试的数据类型
        try_types = (int, np.float32, np.complex64, float, complex)
        size = 10000

        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * random_state.randn(size)

            h = firwin(31, 1. / down, window='hamming')
            yl = upfirdn_naive(x, h, 1, down)
            y = upfirdn(h, x, up=1, down=down)
            assert y.shape == (want_len,)
            assert yl.shape[0] == y.shape[0]
            assert_allclose(yl, y, atol=1e-7, rtol=1e-7)

    # 参数化测试：测试与 Naive 实现的 delta 情况比较
    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('h', (1., 1j))
    @pytest.mark.parametrize('up, down', [(1, 1), (2, 2), (3, 2), (2, 3)])
    def test_vs_naive_delta(self, x_dtype, h, up, down):
        UpFIRDnCase(up, down, h, x_dtype)()

    # 参数化测试：测试各种数据类型的 x
    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    # 使用 pytest 的 parametrize 装饰器，为 h_dtype 参数化测试
    @pytest.mark.parametrize('h_dtype', _UPFIRDN_TYPES)
    # 使用 pytest 的 parametrize 装饰器，为 p_max 和 q_max 参数化测试
    @pytest.mark.parametrize('p_max, q_max',
                             list(product((10, 100), (10, 100))))
    # 定义一个测试方法，用于测试不同参数下的算法与 Naive 算法的对比
    def test_vs_naive(self, x_dtype, h_dtype, p_max, q_max):
        # 生成随机测试数据集，每组数据包含多个测试用例
        tests = self._random_factors(p_max, q_max, h_dtype, x_dtype)
        # 遍历每个测试用例并执行
        for test in tests:
            test()

    # 生成随机测试数据集的私有方法
    def _random_factors(self, p_max, q_max, h_dtype, x_dtype):
        # 每种参数组合生成的测试用例数量
        n_rep = 3
        # FIR 系数的最大长度
        longest_h = 25
        # 初始化随机数生成器
        random_state = np.random.RandomState(17)
        tests = []

        # 生成 n_rep 组测试用例
        for _ in range(n_rep):
            # 随机化上采样/下采样因子
            p_add = q_max if p_max > q_max else 1
            q_add = p_max if q_max > p_max else 1
            p = random_state.randint(p_max) + p_add
            q = random_state.randint(q_max) + q_add

            # 生成随机 FIR 系数
            len_h = random_state.randint(longest_h) + 1
            h = np.atleast_1d(random_state.randint(len_h))
            h = h.astype(h_dtype)
            if h_dtype == complex:
                h += 1j * random_state.randint(len_h)

            # 创建 UpFIRDnCase 测试实例并加入测试集合
            tests.append(UpFIRDnCase(p, q, h, x_dtype))

        return tests

    # 使用 pytest 的 parametrize 装饰器，为 mode 参数化测试
    @pytest.mark.parametrize('mode', _upfirdn_modes)
    # 测试不同扩展模式下的功能
    def test_extensions(self, mode):
        """Test vs. manually computed results for modes not in numpy's pad."""
        # 输入数组 x
        x = np.array([1, 2, 3, 1], dtype=float)
        # 前后填充的长度
        npre, npost = 6, 6
        # 调用 _pad_test 函数计算填充后的结果 y
        y = _pad_test(x, npre=npre, npost=npost, mode=mode)
        
        # 根据不同的 mode 设置预期结果 y_expected
        if mode == 'antisymmetric':
            y_expected = np.asarray(
                [3, 1, -1, -3, -2, -1, 1, 2, 3, 1, -1, -3, -2, -1, 1, 2])
        elif mode == 'antireflect':
            y_expected = np.asarray(
                [1, 2, 3, 1, -1, 0, 1, 2, 3, 1, -1, 0, 1, 2, 3, 1])
        elif mode == 'smooth':
            y_expected = np.asarray(
                [-5, -4, -3, -2, -1, 0, 1, 2, 3, 1, -1, -3, -5, -7, -9, -11])
        elif mode == "line":
            # 计算线性模式下的预期输出
            lin_slope = (x[-1] - x[0]) / (len(x) - 1)
            left = x[0] + np.arange(-npre, 0, 1) * lin_slope
            right = x[-1] + np.arange(1, npost + 1) * lin_slope
            y_expected = np.concatenate((left, x, right))
        else:
            # 对于其他 mode 使用 numpy 的 pad 函数计算预期结果 y_expected
            y_expected = np.pad(x, (npre, npost), mode=mode)
        
        # 断言实际计算结果 y 与预期结果 y_expected 接近
        assert_allclose(y, y_expected)

    # 使用 pytest 的 parametrize 装饰器，为 size, h_len, mode, dtype 参数化测试
    @pytest.mark.parametrize(
        'size, h_len, mode, dtype',
        product(
            [8],
            [4, 5, 26],  # include cases with h_len > 2*size
            _upfirdn_modes,
            [np.float32, np.float64, np.complex64, np.complex128],
        )
    )
    # 定义一个测试方法，用于测试信号处理函数的不同模式
    def test_modes(self, size, h_len, mode, dtype):
        # 创建一个特定随机种子的随机数生成器对象
        random_state = np.random.RandomState(5)
        # 生成指定大小的随机数列，并转换为指定的数据类型
        x = random_state.randn(size).astype(dtype)
        # 如果数据类型是复数类型，则生成虚部并添加到实部中
        if dtype in (np.complex64, np.complex128):
            x += 1j * random_state.randn(size)
        # 生成一个从1开始到h_len结束的数组，数据类型与x的实部相同
        h = np.arange(1, 1 + h_len, dtype=x.real.dtype)

        # 使用给定的模式进行信号处理
        y = upfirdn(h, x, up=1, down=1, mode=mode)
        # 预期结果：对输入进行填充，使用零填充进行滤波，然后裁剪结果
        npad = h_len - 1
        # 如果模式在['antisymmetric', 'antireflect', 'smooth', 'line']中，则使用 _pad_test 函数，因为这些模式不被 np.pad 支持
        if mode in ['antisymmetric', 'antireflect', 'smooth', 'line']:
            xpad = _pad_test(x, npre=npad, npost=npad, mode=mode)
        else:
            # 否则，使用 np.pad 函数进行填充
            xpad = np.pad(x, npad, mode=mode)
        # 对填充后的信号再次使用给定的模式进行信号处理
        ypad = upfirdn(h, xpad, up=1, down=1, mode='constant')
        # 获取预期的输出，裁剪掉填充的部分
        y_expected = ypad[npad:-npad]

        # 设置绝对误差和相对误差的阈值
        atol = rtol = np.finfo(dtype).eps * 1e2
        # 使用 assert_allclose 函数断言 y 和 y_expected 在给定的误差范围内相等
        assert_allclose(y, y_expected, atol=atol, rtol=rtol)
def test_output_len_long_input():
    # 用于回归测试 gh-17375。在Windows上，当输入非常大时，
    # 应该远远低于64位整数的能力范围，但由于Cython 0.29.32中的一个 bug，
    # 会导致32位溢出。
    len_h = 1001  # 设置 len_h 的值为 1001
    in_len = 10**8  # 设置 in_len 的值为 100,000,000
    up = 320  # 设置 up 的值为 320
    down = 441  # 设置 down 的值为 441
    # 调用 _output_len 函数计算输出长度
    out_len = _output_len(len_h, in_len, up, down)
    # 预期值是通过手工计算以下公式得出的：
    #   (((in_len - 1) * up + len_h) - 1) // down + 1
    assert out_len == 72562360  # 断言输出长度等于 72562360
```