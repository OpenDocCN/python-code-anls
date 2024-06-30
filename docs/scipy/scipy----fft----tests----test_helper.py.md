# `D:\src\scipysrc\scipy\scipy\fft\tests\test_helper.py`

```
"""Includes test functions for fftpack.helper module

Copied from fftpack.helper by Pearu Peterson, October 2005
Modified for Array API, 2023

"""
# 导入需要的模块和函数
from scipy.fft._helper import next_fast_len, prev_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import (
    xp_assert_close, get_xp_devices, device, array_namespace
)
from scipy import fft

# 声明 pytestmark 列表，用于标记兼容 Array API 并跳过特定后端的测试
pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]
# 标记用于跳过特定后端的测试的 pytest 装饰器
skip_xp_backends = pytest.mark.skip_xp_backends

# 预定义 5-smooth 数组
_5_smooth_numbers = [
    2, 3, 4, 5, 6, 8, 9, 10,
    2 * 3 * 5,
    2**3 * 3**5,
    2**3 * 3**3 * 5**2,
]

# 测试函数：测试 next_fast_len 函数
def test_next_fast_len():
    # 遍历预定义的 5-smooth 数组，验证 next_fast_len 函数的输出
    for n in _5_smooth_numbers:
        assert_equal(next_fast_len(n), n)


# 辅助函数：验证 x 是否为 n-smooth
def _assert_n_smooth(x, n):
    x_orig = x
    # 如果 n 小于 2，则断言失败
    if n < 2:
        assert False

    # 循环直到 x 不再整除 2
    while True:
        q, r = divmod(x, 2)
        if r != 0:
            break
        x = q

    # 从 3 到 n 的奇数因子，直到 x 等于 1
    for d in range(3, n+1, 2):
        while True:
            q, r = divmod(x, d)
            if r != 0:
                break
            x = q

    # 断言 x 等于 1，否则输出错误信息
    assert x == 1, \
           f'x={x_orig} is not {n}-smooth, remainder={x}'


# 使用装饰器标记仅在 NumPy 环境下运行的测试类
@skip_xp_backends(np_only=True)
class TestNextFastLen:

    # 测试方法：验证 next_fast_len 函数在不同输入下的表现
    def test_next_fast_len(self):
        # 设定随机数种子
        np.random.seed(1234)

        # 定义生成器函数 nums，产生范围从 1 到 1000 的数及一个大数
        def nums():
            yield from range(1, 1000)
            yield 2**5 * 3**5 * 4**5 + 1

        # 遍历 nums 生成的数值
        for n in nums():
            # 计算 next_fast_len(n) 并断言其为 11-smooth
            m = next_fast_len(n)
            _assert_n_smooth(m, 11)
            # 断言 next_fast_len(n, False) 等于 m
            assert m == next_fast_len(n, False)

            # 计算 next_fast_len(n, True) 并断言其为 5-smooth
            m = next_fast_len(n, True)
            _assert_n_smooth(m, 5)

    # 测试方法：验证 next_fast_len 函数对 NumPy 整数的处理
    def test_np_integers(self):
        # 定义整数类型数组
        ITYPES = [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]
        # 遍历整数类型数组
        for ityp in ITYPES:
            x = ityp(12345)
            # 计算 next_fast_len(x) 并断言其与 next_fast_len(int(x)) 相等
            testN = next_fast_len(x)
            assert_equal(testN, next_fast_len(int(x)))

    # 测试方法：验证 next_fast_len 函数在小数值下的表现
    def testnext_fast_len_small(self):
        # 预定义的哈明数字典
        hams = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8, 8: 8, 14: 15, 15: 15,
            16: 16, 17: 18, 1021: 1024, 1536: 1536, 51200000: 51200000
        }
        # 遍历哈明数字典，验证 next_fast_len 函数的输出
        for x, y in hams.items():
            assert_equal(next_fast_len(x, True), y)

    # 标记测试为预期失败，如果系统最大整数小于 2**32，输出错误信息
    @pytest.mark.xfail(sys.maxsize < 2**32,
                       reason="Hamming Numbers too large for 32-bit",
                       raises=ValueError, strict=True)
    # 定义测试函数 testnext_fast_len_big，用于测试 next_fast_len 函数的处理能力
    def testnext_fast_len_big(self):
        # 定义一个字典 hams，包含一系列键值对，用于测试 next_fast_len 函数的不同输入
        hams = {
            510183360: 510183360, 510183360 + 1: 512000000,
            511000000: 512000000,
            854296875: 854296875, 854296875 + 1: 859963392,
            196608000000: 196608000000, 196608000000 + 1: 196830000000,
            8789062500000: 8789062500000, 8789062500000 + 1: 8796093022208,
            206391214080000: 206391214080000,
            206391214080000 + 1: 206624260800000,
            470184984576000: 470184984576000,
            470184984576000 + 1: 470715894135000,
            7222041363087360: 7222041363087360,
            7222041363087360 + 1: 7230196133913600,
            # power of 5    5**23
            11920928955078125: 11920928955078125,
            11920928955078125 - 1: 11920928955078125,
            # power of 3    3**34
            16677181699666569: 16677181699666569,
            16677181699666569 - 1: 16677181699666569,
            # power of 2   2**54
            18014398509481984: 18014398509481984,
            18014398509481984 - 1: 18014398509481984,
            # above this, int(ceil(n)) == int(ceil(n+1))
            19200000000000000: 19200000000000000,
            19200000000000000 + 1: 19221679687500000,
            288230376151711744: 288230376151711744,
            288230376151711744 + 1: 288325195312500000,
            288325195312500000 - 1: 288325195312500000,
            288325195312500000: 288325195312500000,
            288325195312500000 + 1: 288555831593533440,
        }
        # 遍历字典 hams 中的每一对键值对，对 next_fast_len 函数的返回值进行断言比较
        for x, y in hams.items():
            assert_equal(next_fast_len(x, True), y)

    # 定义测试函数 test_keyword_args，测试 next_fast_len 函数的关键字参数处理能力
    def test_keyword_args(self):
        # 断言调用 next_fast_len 函数，以参数 11 和 real=True，返回值为 12
        assert next_fast_len(11, real=True) == 12
        # 断言调用 next_fast_len 函数，以关键字参数 target=7 和 real=False，返回值为 7
        assert next_fast_len(target=7, real=False) == 7
# 装饰器，指定在测试中跳过特定的 XP 后端（仅限 NumPy）
@skip_xp_backends(np_only=True)
# 定义一个测试类 TestPrevFastLen
class TestPrevFastLen:

    # 定义测试方法 test_prev_fast_len
    def test_prev_fast_len(self):
        # 设定随机种子为1234
        np.random.seed(1234)

        # 定义一个生成器函数 nums，生成器产生从1到999的数字以及一个大数
        def nums():
            yield from range(1, 1000)
            yield 2**5 * 3**5 * 4**5 + 1

        # 对生成器 nums 生成的每个数字进行迭代
        for n in nums():
            # 调用 prev_fast_len 函数计算给定数字 n 的最大 n-smooth 数，并断言其符合条件
            m = prev_fast_len(n)
            _assert_n_smooth(m, 11)
            # 断言调用 prev_fast_len 函数的两种调用方式返回相同的结果
            assert m == prev_fast_len(n, False)

            # 调用 prev_fast_len 函数计算给定数字 n 的最大 n-smooth 实数，并断言其符合条件
            m = prev_fast_len(n, True)
            _assert_n_smooth(m, 5)

    # 定义测试方法 test_np_integers
    def test_np_integers(self):
        # 定义一个包含 NumPy 整数类型的列表
        ITYPES = [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64]
        # 对列表中的每种整数类型进行迭代
        for ityp in ITYPES:
            # 创建一个指定类型的整数 x
            x = ityp(12345)
            # 调用 prev_fast_len 函数计算整数 x 的最大 n-smooth 数，并断言其相等
            testN = prev_fast_len(x)
            assert_equal(testN, prev_fast_len(int(x)))

            # 调用 prev_fast_len 函数计算整数 x 的最大 n-smooth 实数，并断言其相等
            testN = prev_fast_len(x, real=True)
            assert_equal(testN, prev_fast_len(int(x), real=True))

    # 定义测试方法 testprev_fast_len_small
    def testprev_fast_len_small(self):
        # 定义一个字典 hams，包含一组数字及其对应的最大 n-smooth 数
        hams = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 8, 14: 12, 15: 15,
            16: 16, 17: 16, 1021: 1000, 1536: 1536, 51200000: 51200000
        }
        # 对字典 hams 中的每对键值对进行迭代
        for x, y in hams.items():
            # 断言调用 prev_fast_len 函数计算给定数字 x 的最大 n-smooth 实数并等于 y
            assert_equal(prev_fast_len(x, True), y)

        # 定义另一个字典 hams，包含一组数字及其对应的最大 n-smooth 数
        hams = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
            11: 11, 12: 12, 13: 12, 14: 14, 15: 15, 16: 16, 17: 16, 18: 18,
            19: 18, 20: 20, 21: 21, 22: 22, 120: 120, 121: 121, 122: 121,
            1021: 1008, 1536: 1536, 51200000: 51200000
        }
        # 对字典 hams 中的每对键值对进行迭代
        for x, y in hams.items():
            # 断言调用 prev_fast_len 函数计算给定数字 x 的最大 n-smooth 整数并等于 y
            assert_equal(prev_fast_len(x, False), y)

    # 使用 pytest 的标记，标记为预期失败，如果系统的最大整数大小小于 2 的 32 次方，则抛出 ValueError 异常
    @pytest.mark.xfail(sys.maxsize < 2**32,
                       reason="Hamming Numbers too large for 32-bit",
                       raises=ValueError, strict=True)
    def testprev_fast_len_big(self):
        hams = {
            # 定义测试用例字典，每个键值对包含输入值和期望输出值
            510183360: 510183360,  # 输入值为 510183360，期望输出值为 510183360
            510183360 + 1: 510183360,  # 输入值为 510183361，期望输出值为 510183360
            510183360 - 1: 509607936,  # 输入值为 510183359，期望输出值为 509607936
            511000000: 510183360,  # 输入值为 511000000，期望输出值为 510183360
            511000000 + 1: 510183360,  # 输入值为 511000001，期望输出值为 510183360
            511000000 - 1: 510183360,  # 输入值为 510999999，期望输出值为 510183360
            854296875: 854296875,  # 输入值为 854296875，期望输出值为 854296875
            854296875 + 1: 854296875,  # 输入值为 854296876，期望输出值为 854296875
            854296875 - 1: 850305600,  # 输入值为 854296874，期望输出值为 850305600
            196608000000: 196608000000,  # 输入值为 196608000000，期望输出值为 196608000000
            196608000000 + 1: 196608000000,  # 输入值为 196608000001，期望输出值为 196608000000
            196608000000 - 1: 195910410240,  # 输入值为 196607999999，期望输出值为 195910410240
            8789062500000: 8789062500000,  # 输入值为 8789062500000，期望输出值为 8789062500000
            8789062500000 + 1: 8789062500000,  # 输入值为 8789062500001，期望输出值为 8789062500000
            8789062500000 - 1: 8748000000000,  # 输入值为 8789062499999，期望输出值为 8748000000000
            206391214080000: 206391214080000,  # 输入值为 206391214080000，期望输出值为 206391214080000
            206391214080000 + 1: 206391214080000,  # 输入值为 206391214080001，期望输出值为 206391214080000
            206391214080000 - 1: 206158430208000,  # 输入值为 206391214079999，期望输出值为 206158430208000
            470184984576000: 470184984576000,  # 输入值为 470184984576000，期望输出值为 470184984576000
            470184984576000 + 1: 470184984576000,  # 输入值为 470184984576001，期望输出值为 470184984576000
            470184984576000 - 1: 469654673817600,  # 输入值为 470184984575999，期望输出值为 469654673817600
            7222041363087360: 7222041363087360,  # 输入值为 7222041363087360，期望输出值为 7222041363087360
            7222041363087360 + 1: 7222041363087360,  # 输入值为 7222041363087361，期望输出值为 7222041363087360
            7222041363087360 - 1: 7213895789838336,  # 输入值为 7222041363087359，期望输出值为 7213895789838336
            11920928955078125: 11920928955078125,  # 输入值为 11920928955078125，期望输出值为 11920928955078125
            11920928955078125 + 1: 11920928955078125,  # 输入值为 11920928955078126，期望输出值为 11920928955078125
            11920928955078125 - 1: 11901557422080000,  # 输入值为 11920928955078124，期望输出值为 11901557422080000
            16677181699666569: 16677181699666569,  # 输入值为 16677181699666569，期望输出值为 16677181699666569
            16677181699666569 + 1: 16677181699666569,  # 输入值为 16677181699666570，期望输出值为 16677181699666569
            16677181699666569 - 1: 16607531250000000,  # 输入值为 16677181699666568，期望输出值为 16607531250000000
            18014398509481984: 18014398509481984,  # 输入值为 18014398509481984，期望输出值为 18014398509481984
            18014398509481984 + 1: 18014398509481984,  # 输入值为 18014398509481985，期望输出值为 18014398509481984
            18014398509481984 - 1: 18000000000000000,  # 输入值为 18014398509481983，期望输出值为 18000000000000000
            19200000000000000: 19200000000000000,  # 输入值为 19200000000000000，期望输出值为 19200000000000000
            19200000000000000 + 1: 19200000000000000,  # 输入值为 19200000000000001，期望输出值为 19200000000000000
            19200000000000000 - 1: 19131876000000000,  # 输入值为 19199999999999999，期望输出值为 19131876000000000
            288230376151711744: 288230376151711744,  # 输入值为 288230376151711744，期望输出值为 288230376151711744
            288230376151711744 + 1: 288230376151711744,  # 输入值为 288230376151711745，期望输出值为 288230376151711744
            288230376151711744 - 1: 288000000000000000,  # 输入值为 288230376151711743，期望输出值为 288000000000000000
        }
        # 对每个测试用例进行断言验证
        for x, y in hams.items():
            assert_equal(prev_fast_len(x, True), y)
    # 定义一个测试方法，用于测试 prev_fast_len 函数的关键字参数使用情况
    def test_keyword_args(self):
        # 断言：调用 prev_fast_len 函数，传入参数 11 和 real=True，期望返回 10
        assert prev_fast_len(11, real=True) == 10
        # 断言：调用 prev_fast_len 函数，传入参数 target=7 和 real=False，期望返回 7
        assert prev_fast_len(target=7, real=False) == 7
# 定义一个装饰器，用于跳过特定的 XP 后端，只使用 CPU
@skip_xp_backends(cpu_only=True)
# 定义一个测试类，用于测试 _init_nd_shape_and_axes 函数的不同情况
class Test_init_nd_shape_and_axes:

    # 测试在 Python 中创建的 0 维数组的默认情况
    def test_py_0d_defaults(self, xp):
        # 创建一个包含单个元素 4 的数组
        x = xp.asarray(4)
        # 未指定形状和轴
        shape = None
        axes = None

        # 预期的形状和轴
        shape_expected = ()
        axes_expected = []

        # 调用 _init_nd_shape_and_axes 函数进行初始化
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        # 断言实际输出与预期输出相符
        assert shape_res == shape_expected
        assert axes_res == axes_expected

    # 测试在 XP 后端中创建的 0 维数组的默认情况
    def test_xp_0d_defaults(self, xp):
        # 创建一个包含单个浮点数 7.0 的数组
        x = xp.asarray(7.)
        shape = None
        axes = None

        shape_expected = ()
        axes_expected = []

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    # 测试在 Python 中创建的 1 维数组的默认情况
    def test_py_1d_defaults(self, xp):
        # 创建一个包含元素 [1, 2, 3] 的数组
        x = xp.asarray([1, 2, 3])
        shape = None
        axes = None

        shape_expected = (3,)
        axes_expected = [0]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    # 测试在 XP 后端中创建的 1 维数组的默认情况
    def test_xp_1d_defaults(self, xp):
        # 创建一个从 0 到 1 步长为 0.1 的数组，共 10 个元素
        x = xp.arange(0, 1, .1)
        shape = None
        axes = None

        shape_expected = (10,)
        axes_expected = [0]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    # 测试在 Python 中创建的 2 维数组的默认情况
    def test_py_2d_defaults(self, xp):
        # 创建一个包含两个行向量的数组
        x = xp.asarray([[1, 2, 3, 4],
                        [5, 6, 7, 8]])
        shape = None
        axes = None

        shape_expected = (2, 4)
        axes_expected = [0, 1]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    # 测试在 XP 后端中创建的 2 维数组的默认情况
    def test_xp_2d_defaults(self, xp):
        # 创建一个从 0 到 1 步长为 0.1 的数组，然后将其重塑为 (5, 2) 形状
        x = xp.arange(0, 1, .1)
        x = xp.reshape(x, (5, 2))
        shape = None
        axes = None

        shape_expected = (5, 2)
        axes_expected = [0, 1]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    # 测试在 XP 后端中创建的 5 维数组的默认情况
    def test_xp_5d_defaults(self, xp):
        # 创建一个所有元素为零的 5 维数组
        x = xp.zeros([6, 2, 5, 3, 4])
        shape = None
        axes = None

        shape_expected = (6, 2, 5, 3, 4)
        axes_expected = [0, 1, 2, 3, 4]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected

    # 测试在 XP 后端中创建的 5 维数组，指定形状的情况
    def test_xp_5d_set_shape(self, xp):
        # 创建一个所有元素为零的 5 维数组
        x = xp.zeros([6, 2, 5, 3, 4])
        # 指定形状为 [10, -1, -1, 1, 4]
        shape = [10, -1, -1, 1, 4]
        axes = None

        shape_expected = (10, 2, 5, 1, 4)
        axes_expected = [0, 1, 2, 3, 4]

        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)

        assert shape_res == shape_expected
        assert axes_res == axes_expected
    # 测试设置多维数组的形状和轴参数，对应于第一个测试用例
    def test_xp_5d_set_axes(self, xp):
        # 创建一个形状为 [6, 2, 5, 3, 4] 的全零多维数组 x
        x = xp.zeros([6, 2, 5, 3, 4])
        # 初始 shape 为 None
        shape = None
        # 设定轴的顺序为 [4, 1, 2]
        axes = [4, 1, 2]
    
        # 预期的形状
        shape_expected = (4, 2, 5)
        # 预期的轴顺序
        axes_expected = [4, 1, 2]
    
        # 调用 _init_nd_shape_and_axes 函数，返回实际的形状和轴顺序
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
    
        # 断言实际的形状与预期的形状相同
        assert shape_res == shape_expected
        # 断言实际的轴顺序与预期的轴顺序相同
        assert axes_res == axes_expected
    
    
    # 测试设置多维数组的形状和轴参数，对应于第二个测试用例
    def test_xp_5d_set_shape_axes(self, xp):
        # 创建一个形状为 [6, 2, 5, 3, 4] 的全零多维数组 x
        x = xp.zeros([6, 2, 5, 3, 4])
        # 设定形状为 [10, -1, 2]
        shape = [10, -1, 2]
        # 设定轴的顺序为 [1, 0, 3]
        axes = [1, 0, 3]
    
        # 预期的形状
        shape_expected = (10, 6, 2)
        # 预期的轴顺序
        axes_expected = [1, 0, 3]
    
        # 调用 _init_nd_shape_and_axes 函数，返回实际的形状和轴顺序
        shape_res, axes_res = _init_nd_shape_and_axes(x, shape, axes)
    
        # 断言实际的形状与预期的形状相同
        assert shape_res == shape_expected
        # 断言实际的轴顺序与预期的轴顺序相同
        assert axes_res == axes_expected
    
    
    # 测试初始化多维数组的形状和轴参数，对应于第三个测试用例
    def test_shape_axes_subset(self, xp):
        # 创建一个形状为 (2, 3, 4, 5) 的全零多维数组 x
        x = xp.zeros((2, 3, 4, 5))
        # 初始化形状为 (5, 5, 5)，轴参数为 None
        shape, axes = _init_nd_shape_and_axes(x, shape=(5, 5, 5), axes=None)
    
        # 断言实际的形状与预期的形状相同
        assert shape == (5, 5, 5)
        # 断言实际的轴顺序为 [1, 2, 3]（因为没有指定轴参数，保持默认顺序）
        assert axes == [1, 2, 3]
    # 定义一个测试方法，用于测试 _init_nd_shape_and_axes 函数在不同错误情况下的行为
    def test_errors(self, xp):
        # 创建一个包含一个元素的零数组
        x = xp.zeros(1)
        # 测试当 axes 参数既非标量也非整数迭代器时，应引发 ValueError 异常
        with assert_raises(ValueError, match="axes must be a scalar or "
                           "iterable of integers"):
            _init_nd_shape_and_axes(x, shape=None, axes=[[1, 2], [3, 4]])

        # 测试当 axes 参数包含非整数时，应引发 ValueError 异常
        with assert_raises(ValueError, match="axes must be a scalar or "
                           "iterable of integers"):
            _init_nd_shape_and_axes(x, shape=None, axes=[1., 2., 3., 4.])

        # 测试当 axes 参数的长度超过输入数组的维度时，应引发 ValueError 异常
        with assert_raises(ValueError,
                           match="axes exceeds dimensionality of input"):
            _init_nd_shape_and_axes(x, shape=None, axes=[1])

        # 测试当 axes 参数包含负数时，应引发 ValueError 异常
        with assert_raises(ValueError,
                           match="axes exceeds dimensionality of input"):
            _init_nd_shape_and_axes(x, shape=None, axes=[-2])

        # 测试当 axes 参数中存在重复的轴时，应引发 ValueError 异常
        with assert_raises(ValueError,
                           match="all axes must be unique"):
            _init_nd_shape_and_axes(x, shape=None, axes=[0, 0])

        # 测试当 shape 参数既非标量也非整数迭代器时，应引发 ValueError 异常
        with assert_raises(ValueError, match="shape must be a scalar or "
                           "iterable of integers"):
            _init_nd_shape_and_axes(x, shape=[[1, 2], [3, 4]], axes=None)

        # 测试当 shape 参数包含非整数时，应引发 ValueError 异常
        with assert_raises(ValueError, match="shape must be a scalar or "
                           "iterable of integers"):
            _init_nd_shape_and_axes(x, shape=[1., 2., 3., 4.], axes=None)

        # 测试当 axes 和 shape 参数的长度不一致时，应引发 ValueError 异常
        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            _init_nd_shape_and_axes(xp.zeros([1, 1, 1, 1]),
                                    shape=[1, 2, 3], axes=[1])

        # 测试当 shape 参数为零时，应引发 ValueError 异常
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[0\]\) specified"):
            _init_nd_shape_and_axes(x, shape=[0], axes=None)

        # 测试当 shape 参数为负数时，应引发 ValueError 异常
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[-2\]\) specified"):
            _init_nd_shape_and_axes(x, shape=-2, axes=None)
class TestFFTShift:

    def test_definition(self, xp):
        # 创建一个 numpy 数组 x，包含指定的数据序列
        x = xp.asarray([0., 1, 2, 3, 4, -4, -3, -2, -1])
        # 创建一个 numpy 数组 y，包含指定的数据序列
        y = xp.asarray([-4., -3, -2, -1, 0, 1, 2, 3, 4])
        # 调用 xp_assert_close 函数，验证 fft.fftshift 函数对 x 的操作结果接近于 y
        xp_assert_close(fft.fftshift(x), y)
        # 调用 xp_assert_close 函数，验证 fft.ifftshift 函数对 y 的操作结果接近于 x
        xp_assert_close(fft.ifftshift(y), x)
        
        # 修改 x 数组内容并重复上述验证
        x = xp.asarray([0., 1, 2, 3, 4, -5, -4, -3, -2, -1])
        y = xp.asarray([-5., -4, -3, -2, -1, 0, 1, 2, 3, 4])
        xp_assert_close(fft.fftshift(x), y)
        xp_assert_close(fft.ifftshift(y), x)

    def test_inverse(self, xp):
        # 随机生成长度为 n 的 numpy 数组 x，对其进行 fftshift 和 ifftshift 操作，验证逆操作
        for n in [1, 4, 9, 100, 211]:
            x = xp.asarray(np.random.random((n,)))
            xp_assert_close(fft.ifftshift(fft.fftshift(x)), x)

    def test_axes_keyword(self, xp):
        # 创建一个 2D 的 numpy 数组 freqs 和对应的 shifted 数组，对 fftshift 和 ifftshift 函数进行测试
        freqs = xp.asarray([[0., 1, 2], [3, 4, -4], [-3, -2, -1]])
        shifted = xp.asarray([[-1., -3, -2], [2, 0, 1], [-4, 3, 4]])
        
        # 测试 fftshift 函数在指定轴上的操作
        xp_assert_close(fft.fftshift(freqs, axes=(0, 1)), shifted)
        xp_assert_close(fft.fftshift(freqs, axes=0), fft.fftshift(freqs, axes=(0,)))
        
        # 测试 ifftshift 函数在指定轴上的操作
        xp_assert_close(fft.ifftshift(shifted, axes=(0, 1)), freqs)
        xp_assert_close(fft.ifftshift(shifted, axes=0),
                        fft.ifftshift(shifted, axes=(0,)))
        
        # 验证 fftshift 和 ifftshift 在默认轴上的操作
        xp_assert_close(fft.fftshift(freqs), shifted)
        xp_assert_close(fft.ifftshift(shifted), freqs)
    
    def test_uneven_dims(self, xp):
        """ Test 2D input, which has uneven dimension sizes """
        # 创建一个不均匀维度的 2D numpy 数组 freqs
        freqs = xp.asarray([
            [0, 1],
            [2, 3],
            [4, 5]
        ], dtype=xp.float64)

        # 在维度 0 上进行 fftshift 操作，并验证结果
        shift_dim0 = xp.asarray([
            [4, 5],
            [0, 1],
            [2, 3]
        ], dtype=xp.float64)
        xp_assert_close(fft.fftshift(freqs, axes=0), shift_dim0)
        xp_assert_close(fft.ifftshift(shift_dim0, axes=0), freqs)
        xp_assert_close(fft.fftshift(freqs, axes=(0,)), shift_dim0)
        xp_assert_close(fft.ifftshift(shift_dim0, axes=[0]), freqs)

        # 在维度 1 上进行 fftshift 操作，并验证结果
        shift_dim1 = xp.asarray([
            [1, 0],
            [3, 2],
            [5, 4]
        ], dtype=xp.float64)
        xp_assert_close(fft.fftshift(freqs, axes=1), shift_dim1)
        xp_assert_close(fft.ifftshift(shift_dim1, axes=1), freqs)

        # 在两个维度上同时进行 fftshift 操作，并验证结果
        shift_dim_both = xp.asarray([
            [5, 4],
            [1, 0],
            [3, 2]
        ], dtype=xp.float64)
        xp_assert_close(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
        xp_assert_close(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        xp_assert_close(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
        xp_assert_close(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)

        # 在所有维度上进行 fftshift 操作，并验证结果
        xp_assert_close(fft.fftshift(freqs, axes=None), shift_dim_both)
        xp_assert_close(fft.ifftshift(shift_dim_both, axes=None), freqs)
        xp_assert_close(fft.fftshift(freqs), shift_dim_both)
        xp_assert_close(fft.ifftshift(shift_dim_both), freqs)
@skip_xp_backends("cupy", "jax.numpy",
                  reasons=["CuPy has not implemented the `device` param",
                           "JAX has not implemented the `device` param"])
# 装饰器：跳过使用 CuPy 和 jax.numpy 后端的测试，给出跳过的原因列表

class TestFFTFreq:
    # FFTFreq 测试类定义

    def test_definition(self, xp):
        # 定义测试函数 test_definition，接受 xp 参数作为后端库
        x = xp.asarray([0, 1, 2, 3, 4, -4, -3, -2, -1], dtype=xp.float64)
        # 创建 xp 数组 x，包含指定数据和数据类型

        x2 = xp.asarray([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=xp.float64)
        # 创建另一个 xp 数组 x2，包含指定数据和数据类型

        # default dtype varies across backends
        # 默认的数据类型在不同后端中有所不同

        y = 9 * fft.fftfreq(9, xp=xp)
        # 计算 FFT 频率，乘以 9
        xp_assert_close(y, x, check_dtype=False, check_namespace=True)
        # 使用自定义函数 xp_assert_close 检查 y 和 x 的接近程度，忽略数据类型检查，检查命名空间

        y = 9 * xp.pi * fft.fftfreq(9, xp.pi, xp=xp)
        # 计算带有 pi 的 FFT 频率，乘以 9
        xp_assert_close(y, x, check_dtype=False)
        # 使用自定义函数 xp_assert_close 检查 y 和 x 的接近程度，忽略数据类型检查

        y = 10 * fft.fftfreq(10, xp=xp)
        # 计算 FFT 频率，乘以 10
        xp_assert_close(y, x2, check_dtype=False)
        # 使用自定义函数 xp_assert_close 检查 y 和 x2 的接近程度，忽略数据类型检查

        y = 10 * xp.pi * fft.fftfreq(10, xp.pi, xp=xp)
        # 计算带有 pi 的 FFT 频率，乘以 10
        xp_assert_close(y, x2, check_dtype=False)
        # 使用自定义函数 xp_assert_close 检查 y 和 x2 的接近程度，忽略数据类型检查

    def test_device(self, xp):
        # 定义测试函数 test_device，接受 xp 参数作为后端库
        xp_test = array_namespace(xp.empty(0))
        # 创建测试用的 xp 数组命名空间
        devices = get_xp_devices(xp)
        # 获取后端库 xp 的设备列表
        for d in devices:
            # 遍历设备列表
            y = fft.fftfreq(9, xp=xp, device=d)
            # 在指定设备上计算 FFT 频率
            x = xp_test.empty(0, device=d)
            # 在指定设备上创建空的 xp 数组
            assert device(y) == device(x)
            # 断言 y 和 x 的设备相同

@skip_xp_backends("cupy", "jax.numpy",
                  reasons=["CuPy has not implemented the `device` param",
                           "JAX has not implemented the `device` param"])
# 装饰器：跳过使用 CuPy 和 jax.numpy 后端的测试，给出跳过的原因列表

class TestRFFTFreq:
    # RFFTFreq 测试类定义

    def test_definition(self, xp):
        # 定义测试函数 test_definition，接受 xp 参数作为后端库
        x = xp.asarray([0, 1, 2, 3, 4], dtype=xp.float64)
        # 创建 xp 数组 x，包含指定数据和数据类型

        x2 = xp.asarray([0, 1, 2, 3, 4, 5], dtype=xp.float64)
        # 创建另一个 xp 数组 x2，包含指定数据和数据类型

        # default dtype varies across backends
        # 默认的数据类型在不同后端中有所不同

        y = 9 * fft.rfftfreq(9, xp=xp)
        # 计算 rFFT 频率，乘以 9
        xp_assert_close(y, x, check_dtype=False, check_namespace=True)
        # 使用自定义函数 xp_assert_close 检查 y 和 x 的接近程度，忽略数据类型检查，检查命名空间

        y = 9 * xp.pi * fft.rfftfreq(9, xp.pi, xp=xp)
        # 计算带有 pi 的 rFFT 频率，乘以 9
        xp_assert_close(y, x, check_dtype=False)
        # 使用自定义函数 xp_assert_close 检查 y 和 x 的接近程度，忽略数据类型检查

        y = 10 * fft.rfftfreq(10, xp=xp)
        # 计算 rFFT 频率，乘以 10
        xp_assert_close(y, x2, check_dtype=False)
        # 使用自定义函数 xp_assert_close 检查 y 和 x2 的接近程度，忽略数据类型检查

        y = 10 * xp.pi * fft.rfftfreq(10, xp.pi, xp=xp)
        # 计算带有 pi 的 rFFT 频率，乘以 10
        xp_assert_close(y, x2, check_dtype=False)
        # 使用自定义函数 xp_assert_close 检查 y 和 x2 的接近程度，忽略数据类型检查

    def test_device(self, xp):
        # 定义测试函数 test_device，接受 xp 参数作为后端库
        xp_test = array_namespace(xp.empty(0))
        # 创建测试用的 xp 数组命名空间
        devices = get_xp_devices(xp)
        # 获取后端库 xp 的设备列表
        for d in devices:
            # 遍历设备列表
            y = fft.rfftfreq(9, xp=xp, device=d)
            # 在指定设备上计算 rFFT 频率
            x = xp_test.empty(0, device=d)
            # 在指定设备上创建空的 xp 数组
            assert device(y) == device(x)
            # 断言 y 和 x 的设备相同
```