# `.\numpy\numpy\lib\tests\test_arraypad.py`

```
# 导入 pytest 模块，用于测试
import pytest

# 导入 numpy 库，并分别导入测试函数 assert_array_equal, assert_allclose, assert_equal
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
# 从 numpy.lib._arraypad_impl 模块中导入 _as_pairs 函数
from numpy.lib._arraypad_impl import _as_pairs

# 定义数值类型的元组，包括整数、无符号整数、浮点数和复数
_numeric_dtypes = (
    np._core.sctypes["uint"]
    + np._core.sctypes["int"]
    + np._core.sctypes["float"]
    + np._core.sctypes["complex"]
)

# 定义所有填充模式的字典
_all_modes = {
    'constant': {'constant_values': 0},
    'edge': {},
    'linear_ramp': {'end_values': 0},
    'maximum': {'stat_length': None},
    'mean': {'stat_length': None},
    'median': {'stat_length': None},
    'minimum': {'stat_length': None},
    'reflect': {'reflect_type': 'even'},
    'symmetric': {'reflect_type': 'even'},
    'wrap': {},
    'empty': {}
}


# 定义 TestAsPairs 测试类
class TestAsPairs:
    # 定义测试单个值的方法
    def test_single_value(self):
        """Test casting for a single value."""
        # 预期结果为一个包含 10 个 [[3, 3]] 的 numpy 数组
        expected = np.array([[3, 3]] * 10)
        # 对于不同的输入 x，执行 _as_pairs 函数，并断言结果与预期相等
        for x in (3, [3], [[3]]):
            result = _as_pairs(x, 10)
            assert_equal(result, expected)
        # 使用 dtype=object 测试
        obj = object()
        assert_equal(
            _as_pairs(obj, 10),
            np.array([[obj, obj]] * 10)
        )

    # 定义测试两个不同值的方法
    def test_two_values(self):
        """Test proper casting for two different values."""
        # 在第一维度广播数字的情况下的预期结果
        expected = np.array([[3, 4]] * 10)
        # 对于不同的输入 x，执行 _as_pairs 函数，并断言结果与预期相等
        for x in ([3, 4], [[3, 4]]):
            result = _as_pairs(x, 10)
            assert_equal(result, expected)
        # 使用 dtype=object 测试
        obj = object()
        assert_equal(
            _as_pairs(["a", obj], 10),
            np.array([["a", obj]] * 10)
        )

        # 在第二个/最后一个维度广播数字的情况下的预期结果
        assert_equal(
            _as_pairs([[3], [4]], 2),
            np.array([[3, 3], [4, 4]])
        )
        # 使用 dtype=object 测试
        assert_equal(
            _as_pairs([["a"], [obj]], 2),
            np.array([["a", "a"], [obj, obj]])
        )

    # 定义测试 None 的方法
    def test_with_none(self):
        # 预期结果为 ((None, None), (None, None), (None, None))
        expected = ((None, None), (None, None), (None, None))
        # 测试当 as_index=False 时，_as_pairs 函数的返回结果与预期相等
        assert_equal(
            _as_pairs(None, 3, as_index=False),
            expected
        )
        # 测试当 as_index=True 时，_as_pairs 函数的返回结果与预期相等
        assert_equal(
            _as_pairs(None, 3, as_index=True),
            expected
        )

    # 定义测试直接通过的方法
    def test_pass_through(self):
        """Test if `x` already matching desired output are passed through."""
        # 预期结果为一个 6x2 的 numpy 数组，其值为从 0 到 11 的连续数列
        expected = np.arange(12).reshape((6, 2))
        # 测试当输入已经符合预期输出时，_as_pairs 函数的返回结果与预期相等
        assert_equal(
            _as_pairs(expected, 6),
            expected
        )
    # 定义测试函数 test_as_index，用于测试参数 `as_index=True` 的结果
    def test_as_index(self):
        """Test results if `as_index=True`."""
        # 断言调用 _as_pairs 函数，验证返回结果是否符合预期
        assert_equal(
            _as_pairs([2.6, 3.3], 10, as_index=True),
            np.array([[3, 3]] * 10, dtype=np.intp)
        )
        # 再次断言调用 _as_pairs 函数，验证不同输入的返回结果是否符合预期
        assert_equal(
            _as_pairs([2.6, 4.49], 10, as_index=True),
            np.array([[3, 4]] * 10, dtype=np.intp)
        )
        # 对于一系列输入 x，检查是否引发 ValueError 异常，且异常消息匹配 "negative values"
        for x in (-3, [-3], [[-3]], [-3, 4], [3, -4], [[-3, 4]], [[4, -3]],
                  [[1, 2]] * 9 + [[1, -2]]):
            with pytest.raises(ValueError, match="negative values"):
                _as_pairs(x, 10, as_index=True)

    # 定义测试函数 test_exceptions，用于确保捕获到错误的使用情况
    def test_exceptions(self):
        """Ensure faulty usage is discovered."""
        # 使用 pytest 检查是否引发 ValueError 异常，且异常消息匹配 "more dimensions than allowed"
        with pytest.raises(ValueError, match="more dimensions than allowed"):
            _as_pairs([[[3]]], 10)
        # 使用 pytest 检查是否引发 ValueError 异常，且异常消息匹配 "could not be broadcast"
        with pytest.raises(ValueError, match="could not be broadcast"):
            _as_pairs([[1, 2], [3, 4]], 3)
        # 使用 pytest 检查是否引发 ValueError 异常，且异常消息匹配 "could not be broadcast"
        with pytest.raises(ValueError, match="could not be broadcast"):
            _as_pairs(np.ones((2, 3)), 3)
class TestConditionalShortcuts:
    # 使用 pytest.mark.parametrize 装饰器，为 test_zero_padding_shortcuts 方法添加参数化测试，参数为 _all_modes 字典的所有键
    @pytest.mark.parametrize("mode", _all_modes.keys())
    # 定义 test_zero_padding_shortcuts 方法，接受参数 mode
    def test_zero_padding_shortcuts(self, mode):
        # 创建一个 4x5x6 的数组 test，包含从 0 到 119 的整数
        test = np.arange(120).reshape(4, 5, 6)
        # 创建一个与 test 形状相同的 pad_amt 列表，元素为 (0, 0)
        pad_amt = [(0, 0) for _ in test.shape]
        # 断言 np.pad 方法对 test 应用 pad_amt 和 mode 参数后与原始 test 数组相等
        assert_array_equal(test, np.pad(test, pad_amt, mode=mode))

    # 使用 pytest.mark.parametrize 装饰器，为 test_shallow_statistic_range 方法添加参数化测试，参数为 ['maximum', 'mean', 'median', 'minimum']
    @pytest.mark.parametrize("mode", ['maximum', 'mean', 'median', 'minimum',])
    # 定义 test_shallow_statistic_range 方法，接受参数 mode
    def test_shallow_statistic_range(self, mode):
        # 创建一个 4x5x6 的数组 test，包含从 0 到 119 的整数
        test = np.arange(120).reshape(4, 5, 6)
        # 创建一个与 test 形状相同的 pad_amt 列表，元素为 (1, 1)
        pad_amt = [(1, 1) for _ in test.shape]
        # 断言两次 np.pad 方法应用相同的 pad_amt，但一个 mode 为 'edge'，另一个 mode 为参数化测试中的 mode
        assert_array_equal(np.pad(test, pad_amt, mode='edge'),
                           np.pad(test, pad_amt, mode=mode, stat_length=1))

    # 使用 pytest.mark.parametrize 装饰器，为 test_clip_statistic_range 方法添加参数化测试，参数为 ['maximum', 'mean', 'median', 'minimum']
    @pytest.mark.parametrize("mode", ['maximum', 'mean', 'median', 'minimum',])
    # 定义 test_clip_statistic_range 方法，接受参数 mode
    def test_clip_statistic_range(self, mode):
        # 创建一个 5x6 的数组 test，包含从 0 到 29 的整数
        test = np.arange(30).reshape(5, 6)
        # 创建一个与 test 形状相同的 pad_amt 列表，元素为 (3, 3)
        pad_amt = [(3, 3) for _ in test.shape]
        # 断言两次 np.pad 方法应用相同的 pad_amt 和 mode 参数后结果相等
        assert_array_equal(np.pad(test, pad_amt, mode=mode),
                           np.pad(test, pad_amt, mode=mode, stat_length=30))


class TestStatistic:
    # 定义 test_check_mean_stat_length 方法
    def test_check_mean_stat_length(self):
        # 创建一个包含 0 到 99 的浮点数数组 a，长度为 100
        a = np.arange(100).astype('f')
        # 对数组 a 进行 np.pad 操作，使用 pad 参数 ((25, 20))，mode 参数为 'mean'，stat_length 参数为 ((2, 3))
        a = np.pad(a, ((25, 20), ), 'mean', stat_length=((2, 3), ))
        # 创建一个期望的结果数组 b，包含预先计算好的数值
        b = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
             0.5, 0.5, 0.5, 0.5, 0.5,

             0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
             10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
             20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
             40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
             50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
             60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
             70., 71., 72., 73., 74., 75., 76., 77., 78., 79.,
             80., 81., 82., 83., 84., 85., 86., 87., 88., 89.,
             90., 91., 92., 93., 94., 95., 96., 97., 98., 99.,

             98., 98., 98., 98., 98., 98., 98., 98., 98., 98.,
             98., 98., 98., 98., 98., 98., 98., 98., 98., 98.
             ])
        # 断言数组 a 与 b 相等
        assert_array_equal(a, b)
    # 定义测试函数，验证 np.pad 使用 'maximum' 模式时的行为
    def test_check_maximum_1(self):
        # 创建长度为 100 的 NumPy 数组，元素为 0 到 99
        a = np.arange(100)
        # 在数组两侧分别填充 25 和 20 个元素，填充模式为 'maximum'
        a = np.pad(a, (25, 20), 'maximum')
        # 预期的结果数组，与 np.pad 的 'maximum' 填充模式结果对比
        b = np.array(
            [99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
            )
        # 使用 assert_array_equal 函数检查数组 a 是否与数组 b 相等
        assert_array_equal(a, b)

    # 定义测试函数，验证另一组 np.pad 使用 'maximum' 模式时的行为
    def test_check_maximum_2(self):
        # 创建长度为 100 的 NumPy 数组，元素为 1 到 100
        a = np.arange(100) + 1
        # 在数组两侧分别填充 25 和 20 个元素，填充模式为 'maximum'
        a = np.pad(a, (25, 20), 'maximum')
        # 预期的结果数组，与 np.pad 的 'maximum' 填充模式结果对比
        b = np.array(
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100,

             1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
             61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
             71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
             91, 92, 93, 94, 95, 96, 97, 98, 99, 100,

             100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
            )
        # 使用 assert_array_equal 函数检查数组 a 是否与数组 b 相等
        assert_array_equal(a, b)

    # 定义测试函数，验证 np.pad 使用 'maximum' 模式时，带有 stat_length 参数的行为
    def test_check_maximum_stat_length(self):
        # 创建长度为 100 的 NumPy 数组，元素为 1 到 100
        a = np.arange(100) + 1
        # 在数组两侧分别填充 25 和 20 个元素，填充模式为 'maximum'，指定 stat_length 为 10
        a = np.pad(a, (25, 20), 'maximum', stat_length=10)
        # 预期的结果数组，与 np.pad 的 'maximum' 填充模式结果对比
        b = np.array(
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10,

             1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
             61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
             71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
             91, 92, 93, 94, 95, 96, 97, 98, 99, 100,

             100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
            )
        # 使用 assert_array_equal 函数检查数组 a 是否与数组 b 相等
        assert_array_equal(a, b)
    def test_check_minimum_1(self):
        # 创建一个长度为 100 的 NumPy 数组，内容为 0 到 99
        a = np.arange(100)
        # 在数组的两侧各填充 25 个 0，20 个 0，填充方式为 'minimum'
        a = np.pad(a, (25, 20), 'minimum')
        # 预期的数组 b，按照填充方式 'minimum' 填充
        b = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
        # 断言数组 a 和数组 b 相等
        assert_array_equal(a, b)

    def test_check_minimum_2(self):
        # 创建一个长度为 100 的 NumPy 数组，内容为 2 到 101
        a = np.arange(100) + 2
        # 在数组的两侧各填充 25 个 2，20 个 2，填充方式为 'minimum'
        a = np.pad(a, (25, 20), 'minimum')
        # 预期的数组 b，按照填充方式 'minimum' 填充
        b = np.array(
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2,

             2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
             22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
             42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
             52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
             62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
             72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
             82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
             92, 93, 94, 95, 96, 97, 98, 99, 100, 101,

             2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            )
        # 断言数组 a 和数组 b 相等
        assert_array_equal(a, b)

    def test_check_minimum_stat_length(self):
        # 创建一个长度为 100 的 NumPy 数组，内容为 1 到 100
        a = np.arange(100) + 1
        # 在数组的两侧各填充 25 个 1，20 个 1，填充方式为 'minimum'，指定 stat_length 参数为 10
        a = np.pad(a, (25, 20), 'minimum', stat_length=10)
        # 预期的数组 b，按照填充方式 'minimum' 填充，stat_length 为 10
        b = np.array(
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
              1,  1,  1,  1,  1,

              1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
             61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
             71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
             91, 92, 93, 94, 95, 96, 97, 98, 99, 100,

             91, 91, 91, 91, 91, 91, 91, 91, 91, 91,
             91, 91, 91, 91, 91, 91, 91, 91, 91, 91]
            )
        # 断言数组 a 和数组 b 相等
        assert_array_equal(a, b)
    # 定义一个测试函数，用于验证 np.pad 函数在 'median' 模式下的行为
    def test_check_median(self):
        # 创建一个包含 0 到 99 的浮点数数组，并将其类型设置为 'f'
        a = np.arange(100).astype('f')
        # 在数组的两侧各填充 25 个 'median' 值，形成一个新的数组 a
        a = np.pad(a, (25, 20), 'median')
        # 预期的结果数组 b，包含了填充后的期望值
        b = np.array(
            [49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5,

             0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
             10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
             20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
             40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
             50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
             60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
             70., 71., 72., 73., 74., 75., 76., 77., 78., 79.,
             80., 81., 82., 83., 84., 85., 86., 87., 88., 89.,
             90., 91., 92., 93., 94., 95., 96., 97., 98., 99.,

             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5]
            )
        # 断言填充后的数组 a 与预期的数组 b 相等
        assert_array_equal(a, b)

    # 定义第二个测试函数，验证在二维数组上使用 'median' 模式的 np.pad 行为
    def test_check_median_01(self):
        # 创建一个二维数组 a
        a = np.array([[3, 1, 4], [4, 5, 9], [9, 8, 2]])
        # 在数组的周围用 'median' 模式填充宽度为 1，形成新的数组 a
        a = np.pad(a, 1, 'median')
        # 预期的结果数组 b，包含了填充后的期望值
        b = np.array(
            [[4, 4, 5, 4, 4],

             [3, 3, 1, 4, 3],
             [5, 4, 5, 9, 5],
             [8, 9, 8, 2, 8],

             [4, 4, 5, 4, 4]]
            )
        # 断言填充后的数组 a 与预期的数组 b 相等
        assert_array_equal(a, b)

    # 定义第三个测试函数，验证在二维数组的转置上使用 'median' 模式的 np.pad 行为
    def test_check_median_02(self):
        # 创建一个二维数组 a
        a = np.array([[3, 1, 4], [4, 5, 9], [9, 8, 2]])
        # 将数组 a 进行转置后，在其周围用 'median' 模式填充宽度为 1，再转置回来得到新的数组 a
        a = np.pad(a.T, 1, 'median').T
        # 预期的结果数组 b，包含了填充后的期望值
        b = np.array(
            [[5, 4, 5, 4, 5],

             [3, 3, 1, 4, 3],
             [5, 4, 5, 9, 5],
             [8, 9, 8, 2, 8],

             [5, 4, 5, 4, 5]]
            )
        # 断言填充后的数组 a 与预期的数组 b 相等
        assert_array_equal(a, b)
    def test_check_median_stat_length(self):
        # 创建一个长度为100的浮点数数组，值为0到99，数据类型为单精度浮点数
        a = np.arange(100).astype('f')
        # 将索引为1的元素设置为2.0
        a[1] = 2.
        # 将索引为97的元素设置为96.0
        a[97] = 96.
        # 使用中位数方式填充数组a，使用统计长度为(3, 5)，填充宽度为25和20
        a = np.pad(a, (25, 20), 'median', stat_length=(3, 5))
        # 创建预期结果数组b
        b = np.array(
            [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
              2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
              2.,  2.,  2.,  2.,  2.,

              0.,  2.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
             10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
             20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
             40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
             50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
             60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
             70., 71., 72., 73., 74., 75., 76., 77., 78., 79.,
             80., 81., 82., 83., 84., 85., 86., 87., 88., 89.,
             90., 91., 92., 93., 94., 95., 96., 96., 98., 99.,

             96., 96., 96., 96., 96., 96., 96., 96., 96., 96.,
             96., 96., 96., 96., 96., 96., 96., 96., 96., 96.]
            )
        # 断言a与b相等
        assert_array_equal(a, b)

    def test_check_mean_shape_one(self):
        # 创建一个包含一个子列表[[4, 5, 6]]的列表a
        a = [[4, 5, 6]]
        # 使用均值方式填充数组a，使用统计长度为2，填充宽度为5和7
        a = np.pad(a, (5, 7), 'mean', stat_length=2)
        # 创建预期结果数组b
        b = np.array(
            [[4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],

             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],

             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6],
             [4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6]]
            )
        # 断言a与b相等
        assert_array_equal(a, b)
    # 定义一个测试函数，用于测试 np.pad 函数以 'mean' 模式填充数组的结果是否正确
    def test_check_mean_2(self):
        # 创建一个包含100个浮点数的 NumPy 数组，从0到99
        a = np.arange(100).astype('f')
        # 使用 'mean' 模式，在数组两端分别填充25个和20个均值
        a = np.pad(a, (25, 20), 'mean')
        # 创建预期结果数组 b，包含特定的填充结果
        b = np.array(
            [49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5,

             0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
             10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
             20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
             30., 31., 32., 33., 34., 35., 36., 37., 38., 39.,
             40., 41., 42., 43., 44., 45., 46., 47., 48., 49.,
             50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
             60., 61., 62., 63., 64., 65., 66., 67., 68., 69.,
             70., 71., 72., 73., 74., 75., 76., 77., 78., 79.,
             80., 81., 82., 83., 84., 85., 86., 87., 88., 89.,
             90., 91., 92., 93., 94., 95., 96., 97., 98., 99.,

             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5,
             49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5, 49.5]
            )
        # 断言填充后的数组 a 是否与预期结果 b 相等
        assert_array_equal(a, b)

    # 使用 pytest 的 parametrize 装饰器，为 mode 参数分别传入 "mean", "median", "minimum", "maximum" 进行测试
    @pytest.mark.parametrize("mode", [
        "mean",
        "median",
        "minimum",
        "maximum"
    ])
    # 定义一个测试函数，验证在特定模式下，np.pad 函数填充后数组的首尾值是否相等
    def test_same_prepend_append(self, mode):
        """ Test that appended and prepended values are equal """
        # 创建两个数组，一个包含整数数组，另一个包含小数数组，以触发浮点数精度问题，主要针对 mode=='mean'
        a = np.array([-1, 2, -1]) + np.array([0, 1e-12, 0], dtype=np.float64)
        # 使用指定的填充模式，在数组两端各填充一个值
        a = np.pad(a, (1, 1), mode)
        # 断言填充后数组的第一个值是否等于最后一个值
        assert_equal(a[0], a[-1])

    # 使用 pytest 的 parametrize 装饰器，为 mode 和 stat_length 参数分别传入多组值进行测试
    @pytest.mark.parametrize("mode", ["mean", "median", "minimum", "maximum"])
    @pytest.mark.parametrize(
        "stat_length", [-2, (-2,), (3, -1), ((5, 2), (-2, 3)), ((-4,), (2,))]
    )
    # 定义一个测试函数，验证当 stat_length 包含负值时，np.pad 函数是否会引发 ValueError 异常
    def test_check_negative_stat_length(self, mode, stat_length):
        # 创建一个二维数组，形状为 (6, 5)，包含从0到29的整数
        arr = np.arange(30).reshape((6, 5))
        # 匹配错误信息，用于验证是否抛出预期的 ValueError 异常
        match = "index can't contain negative values"
        # 使用 pytest 的 raises 函数检查是否抛出 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match=match):
            # 调用 np.pad 函数，在数组两端各填充2个值，使用指定的填充模式和 stat_length 参数
            np.pad(arr, 2, mode, stat_length=stat_length)

    # 定义一个测试函数，验证在简单场景下，使用 'mean' 模式填充后的数组是否与预期结果相等
    def test_simple_stat_length(self):
        # 创建一个包含0到29的整数的一维数组 a
        a = np.arange(30)
        # 将一维数组 a 重塑为二维数组，形状为 (6, 5)
        a = np.reshape(a, (6, 5))
        # 使用 'mean' 模式，在数组的行和列两端各填充不同数量的均值
        a = np.pad(a, ((2, 3), (3, 2)), mode='mean', stat_length=(3,))
        # 创建预期结果数组 b，包含特定的填充结果
        b = np.array(
            [[6, 6, 6, 5, 6, 7, 8, 9, 8, 8],
             [6, 6, 6, 5, 6, 7, 8, 9, 8, 8],

             [1, 1, 1, 0, 1, 2, 3, 4, 3, 3],
             [6, 6, 6, 5, 6, 7, 8, 9, 8, 8],
             [11, 11, 11, 10, 11, 12, 13, 14, 13, 13],
             [16, 16, 16, 15, 16, 17, 18, 19, 18, 18],
             [21, 21, 21, 20, 21, 22, 23, 24, 23, 23],
             [26, 26, 26, 25, 26, 27, 28, 29, 28, 28],

             [21, 21, 21, 20, 21, 22, 23, 24, 23, 23],
             [21, 21, 21, 20, 21, 22, 23, 24, 23, 23],
             [21, 21, 21, 20, 21, 22, 23, 24, 23, 23]]
            )
        # 断言填充后的数组 a 是否与预期结果 b 相等
        assert_array_equal(a, b)
    # 使用 pytest 的标记忽略指定的 RuntimeWarning：Mean of empty slice
    # 使用 pytest 的标记忽略指定的 RuntimeWarning：invalid value encountered in scalar divide
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in( scalar)? divide:RuntimeWarning"
    )
    
    # 使用 pytest 的参数化功能，分别测试 "mean" 和 "median" 两种模式下的函数
    def test_zero_stat_length_valid(self, mode):
        # 创建一个长度为 5 的数组，用指定的填充模式和 stat_length=0 进行填充
        arr = np.pad([1., 2.], (1, 2), mode, stat_length=0)
        # 期望的结果数组
        expected = np.array([np.nan, 1., 2., np.nan, np.nan])
        # 断言填充后的数组与期望的数组相等
        assert_equal(arr, expected)
    
    # 使用 pytest 的参数化功能，分别测试 "minimum" 和 "maximum" 两种模式下的函数
    def test_zero_stat_length_invalid(self, mode):
        # 预期的错误信息
        match = "stat_length of 0 yields no value for padding"
        
        # 使用 pytest 的上下文管理器，检查在 stat_length=0 时是否会引发 ValueError 异常
        with pytest.raises(ValueError, match=match):
            np.pad([1., 2.], 0, mode, stat_length=0)
        with pytest.raises(ValueError, match=match):
            np.pad([1., 2.], 0, mode, stat_length=(1, 0))
        with pytest.raises(ValueError, match=match):
            np.pad([1., 2.], 1, mode, stat_length=0)
        with pytest.raises(ValueError, match=match):
            np.pad([1., 2.], 1, mode, stat_length=(1, 0))
class TestConstant:
    # 定义测试类 TestConstant

    def test_check_constant(self):
        # 定义测试方法 test_check_constant
        a = np.arange(100)
        # 创建长度为 100 的 NumPy 数组 a，包含从 0 到 99 的整数
        a = np.pad(a, (25, 20), 'constant', constant_values=(10, 20))
        # 使用常数填充模式，将数组 a 在两侧分别填充 25 和 20 个常数，左边填充值为 10，右边填充值为 20
        b = np.array(
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
             10, 10, 10, 10, 10,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
             20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        )
        # 创建期望结果数组 b，与 a 相同的填充模式
        assert_array_equal(a, b)
        # 断言数组 a 与数组 b 相等

    def test_check_constant_zeros(self):
        # 定义测试方法 test_check_constant_zeros
        a = np.arange(100)
        # 创建长度为 100 的 NumPy 数组 a，包含从 0 到 99 的整数
        a = np.pad(a, (25, 20), 'constant')
        # 使用常数填充模式，将数组 a 在两侧分别填充 25 和 20 个零
        b = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        # 创建期望结果数组 b，与 a 相同的填充模式
        assert_array_equal(a, b)
        # 断言数组 a 与数组 b 相等

    def test_check_constant_float(self):
        # 定义测试方法 test_check_constant_float
        # 如果输入数组为整数，但常数值为浮点数，填充后的数组仍保持整数类型
        arr = np.arange(30).reshape(5, 6)
        # 创建形状为 (5, 6) 的二维数组 arr，包含从 0 到 29 的整数
        test = np.pad(arr, (1, 2), mode='constant',
                   constant_values=1.1)
        # 使用常数填充模式，将数组 arr 在上下各填充 1 和 2 行/列，填充值为 1.1
        expected = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1],

             [1, 0, 1, 2, 3, 4, 5, 1, 1],
             [1, 6, 7, 8, 9, 10, 11, 1, 1],
             [1, 12, 13, 14, 15, 16, 17, 1, 1],
             [1, 18, 19, 20, 21, 22, 23, 1, 1],
             [1, 24, 25, 26, 27, 28, 29, 1, 1],

             [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )
        # 创建期望结果数组 expected，与 test 相同的填充模式
        assert_allclose(test, expected)
        # 断言 test 与 expected 数组近似相等
    def test_check_constant_float2(self):
        # 如果输入数组是浮点数，并且常数值也是浮点数，则填充后的数组保持浮点数类型，这里保留了浮点数常数
        arr = np.arange(30).reshape(5, 6)  # 创建一个5x6的数组，元素为0到29
        arr_float = arr.astype(np.float64)  # 将数组类型转换为float64
        test = np.pad(arr_float, ((1, 2), (1, 2)), mode='constant',
                      constant_values=1.1)  # 使用常数值1.1在指定模式下填充数组
        expected = np.array(
            [[  1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1],

             [  1.1,   0. ,   1. ,   2. ,   3. ,   4. ,   5. ,   1.1,   1.1],
             [  1.1,   6. ,   7. ,   8. ,   9. ,  10. ,  11. ,   1.1,   1.1],
             [  1.1,  12. ,  13. ,  14. ,  15. ,  16. ,  17. ,   1.1,   1.1],
             [  1.1,  18. ,  19. ,  20. ,  21. ,  22. ,  23. ,   1.1,   1.1],
             [  1.1,  24. ,  25. ,  26. ,  27. ,  28. ,  29. ,   1.1,   1.1],

             [  1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1],
             [  1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1,   1.1]]
            )
        assert_allclose(test, expected)  # 断言填充后的数组与预期结果的接近程度

    def test_check_constant_float3(self):
        a = np.arange(100, dtype=float)  # 创建一个包含100个元素的浮点数数组
        a = np.pad(a, (25, 20), 'constant', constant_values=(-1.1, -1.2))  # 使用常数值在指定模式下填充数组
        b = np.array(
            [-1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1,
             -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1, -1.1,
             -1.1, -1.1, -1.1, -1.1, -1.1,

             0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2,
             -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2]
            )
        assert_allclose(a, b)  # 断言填充后的数组与预期结果的接近程度

    def test_check_constant_odd_pad_amount(self):
        arr = np.arange(30).reshape(5, 6)  # 创建一个5x6的数组，元素为0到29
        test = np.pad(arr, ((1,), (2,)), mode='constant',
                      constant_values=3)  # 使用常数值在指定模式下填充数组
        expected = np.array(
            [[ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3],

             [ 3,  3,  0,  1,  2,  3,  4,  5,  3,  3],
             [ 3,  3,  6,  7,  8,  9, 10, 11,  3,  3],
             [ 3,  3, 12, 13, 14, 15, 16, 17,  3,  3],
             [ 3,  3, 18, 19, 20, 21, 22, 23,  3,  3],
             [ 3,  3, 24, 25, 26, 27, 28, 29,  3,  3],

             [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3]]
            )
        assert_allclose(test, expected)  # 断言填充后的数组与预期结果的接近程度
    def test_check_constant_pad_2d(self):
        arr = np.arange(4).reshape(2, 2)
        # 使用 np.pad 函数对 arr 进行二维常数填充
        test = np.pad(arr, ((1, 2), (1, 3)), mode='constant',
                          constant_values=((1, 2), (3, 4)))
        expected = np.array(
            [[3, 1, 1, 4, 4, 4],
             [3, 0, 1, 4, 4, 4],
             [3, 2, 3, 4, 4, 4],
             [3, 2, 2, 4, 4, 4],
             [3, 2, 2, 4, 4, 4]]
        )
        # 断言测试结果与预期结果相等
        assert_allclose(test, expected)

    def test_check_large_integers(self):
        # 设置 uint64 类型的最大值
        uint64_max = 2 ** 64 - 1
        arr = np.full(5, uint64_max, dtype=np.uint64)
        # 使用常数填充模式对 arr 进行填充
        test = np.pad(arr, 1, mode="constant", constant_values=arr.min())
        expected = np.full(7, uint64_max, dtype=np.uint64)
        # 断言测试结果与预期结果相等
        assert_array_equal(test, expected)

        # 设置 int64 类型的最大值
        int64_max = 2 ** 63 - 1
        arr = np.full(5, int64_max, dtype=np.int64)
        # 使用常数填充模式对 arr 进行填充
        test = np.pad(arr, 1, mode="constant", constant_values=arr.min())
        expected = np.full(7, int64_max, dtype=np.int64)
        # 断言测试结果与预期结果相等
        assert_array_equal(test, expected)

    def test_check_object_array(self):
        arr = np.empty(1, dtype=object)
        obj_a = object()
        arr[0] = obj_a
        obj_b = object()
        obj_c = object()
        # 使用常数填充模式对对象数组 arr 进行填充
        arr = np.pad(arr, pad_width=1, mode='constant',
                     constant_values=(obj_b, obj_c))

        expected = np.empty((3,), dtype=object)
        expected[0] = obj_b
        expected[1] = obj_a
        expected[2] = obj_c

        # 断言测试结果与预期结果相等
        assert_array_equal(arr, expected)

    def test_pad_empty_dimension(self):
        arr = np.zeros((3, 0, 2))
        # 使用常数填充模式对 arr 进行填充，填充宽度分别为 [(0,), (2,), (1,)]
        result = np.pad(arr, [(0,), (2,), (1,)], mode="constant")
        # 断言结果的形状与预期形状相等
        assert result.shape == (3, 4, 4)
# 定义一个测试类 TestLinearRamp，用于测试 np.pad 函数的线性填充模式

class TestLinearRamp:

    # 测试简单的一维数组线性填充
    def test_check_simple(self):
        # 创建一个包含100个浮点数的一维数组，并进行线性填充
        a = np.arange(100).astype('f')
        a = np.pad(a, (25, 20), 'linear_ramp', end_values=(4, 5))
        
        # 预期的填充结果数组 b
        b = np.array(
            [4.00, 3.84, 3.68, 3.52, 3.36, 3.20, 3.04, 2.88, 2.72, 2.56,
             2.40, 2.24, 2.08, 1.92, 1.76, 1.60, 1.44, 1.28, 1.12, 0.96,
             0.80, 0.64, 0.48, 0.32, 0.16,
             0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00,
             10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
             20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
             30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
             40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
             50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0,
             60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
             70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
             80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0,
             90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,

             94.3, 89.6, 84.9, 80.2, 75.5, 70.8, 66.1, 61.4, 56.7, 52.0,
             47.3, 42.6, 37.9, 33.2, 28.5, 23.8, 19.1, 14.4, 9.7, 5.]
            )
        
        # 使用 numpy 的 assert_allclose 函数比较数组 a 和预期的数组 b 是否在误差范围内相等
        assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    # 测试二维数组的线性填充
    def test_check_2d(self):
        # 创建一个包含20个浮点数的二维数组，并进行线性填充
        arr = np.arange(20).reshape(4, 5).astype(np.float64)
        test = np.pad(arr, (2, 2), mode='linear_ramp', end_values=(0, 0))
        
        # 预期的填充结果数组 expected
        expected = np.array(
            [[0.,   0.,   0.,   0.,   0.,   0.,   0.,    0.,   0.],
             [0.,   0.,   0.,  0.5,   1.,  1.5,   2.,    1.,   0.],
             [0.,   0.,   0.,   1.,   2.,   3.,   4.,    2.,   0.],
             [0.,  2.5,   5.,   6.,   7.,   8.,   9.,   4.5,   0.],
             [0.,   5.,  10.,  11.,  12.,  13.,  14.,    7.,   0.],
             [0.,  7.5,  15.,  16.,  17.,  18.,  19.,   9.5,   0.],
             [0., 3.75,  7.5,   8.,  8.5,   9.,  9.5,  4.75,   0.],
             [0.,   0.,   0.,   0.,   0.,   0.,   0.,    0.,   0.]])
        
        # 使用 numpy 的 assert_allclose 函数比较数组 test 和预期的数组 expected 是否在误差范围内相等
        assert_allclose(test, expected)

    # 测试包含对象数组的线性填充，标记为预期失败的测试
    @pytest.mark.xfail(exceptions=(AssertionError,))
    def test_object_array(self):
        from fractions import Fraction
        # 创建包含分数对象的数组，并进行线性填充
        arr = np.array([Fraction(1, 2), Fraction(-1, 2)])
        actual = np.pad(arr, (2, 3), mode='linear_ramp', end_values=0)
        
        # 预期的填充结果数组 expected，使用分数对象
        expected = np.array([
            Fraction( 0, 12),
            Fraction( 3, 12),
            Fraction( 6, 12),
            Fraction(-6, 12),
            Fraction(-4, 12),
            Fraction(-2, 12),
            Fraction(-0, 12),
        ])
        
        # 使用 numpy 的 assert_equal 函数比较数组 actual 和预期的数组 expected 是否完全相等
        assert_equal(actual, expected)
    def test_end_values(self):
        """Ensure that end values are exact."""
        # 创建一个二维数组，元素为10个1，再在边缘填充223行123列，使用线性渐变填充模式
        a = np.pad(np.ones(10).reshape(2, 5), (223, 123), mode="linear_ramp")
        # 断言：第一列的所有元素应该为0
        assert_equal(a[:, 0], 0.)
        # 断言：最后一列的所有元素应该为0
        assert_equal(a[:, -1], 0.)
        # 断言：第一行的所有元素应该为0
        assert_equal(a[0, :], 0.)
        # 断言：最后一行的所有元素应该为0
        assert_equal(a[-1, :], 0.)

    @pytest.mark.parametrize("dtype", _numeric_dtypes)
    def test_negative_difference(self, dtype):
        """
        Check correct behavior of unsigned dtypes if there is a negative
        difference between the edge to pad and `end_values`. Check both cases
        to be independent of implementation. Test behavior for all other dtypes
        in case dtype casting interferes with complex dtypes. See gh-14191.
        """
        # 创建一个包含单个元素3的numpy数组，指定数据类型为dtype
        x = np.array([3], dtype=dtype)
        # 使用线性渐变填充模式，向两侧各填充3个元素，末值为0
        result = np.pad(x, 3, mode="linear_ramp", end_values=0)
        # 期望得到的结果数组，数据类型与输入数组相同
        expected = np.array([0, 1, 2, 3, 2, 1, 0], dtype=dtype)
        # 断言：填充后的结果应该与期望结果相同
        assert_equal(result, expected)

        # 创建一个包含单个元素0的numpy数组，指定数据类型为dtype
        x = np.array([0], dtype=dtype)
        # 使用线性渐变填充模式，向两侧各填充3个元素，末值为3
        result = np.pad(x, 3, mode="linear_ramp", end_values=3)
        # 期望得到的结果数组，数据类型与输入数组相同
        expected = np.array([3, 2, 1, 0, 1, 2, 3], dtype=dtype)
        # 断言：填充后的结果应该与期望结果相同
        assert_equal(result, expected)
# 定义一个名为 TestReflect 的测试类
class TestReflect:
    
    # 定义一个测试方法 test_check_simple
    def test_check_simple(self):
        # 创建一个长度为 100 的 NumPy 数组 a，其值为 0 到 99
        a = np.arange(100)
        # 在数组 a 的两侧各填充 25 个元素，填充方式为反射（reflect）
        a = np.pad(a, (25, 20), 'reflect')
        # 创建期望的结果数组 b，用于与填充后的数组 a 进行比较
        b = np.array(
            [25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
             15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
             5, 4, 3, 2, 1,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             98, 97, 96, 95, 94, 93, 92, 91, 90, 89,
             88, 87, 86, 85, 84, 83, 82, 81, 80, 79]
            )
        # 使用 NumPy 的断言函数检查数组 a 和 b 是否相等
        assert_array_equal(a, b)

    # 定义一个测试方法 test_check_odd_method
    def test_check_odd_method(self):
        # 创建一个长度为 100 的 NumPy 数组 a，其值为 0 到 99
        a = np.arange(100)
        # 在数组 a 的两侧各填充 25 个元素，填充方式为反射（reflect），反射类型为 'odd'
        a = np.pad(a, (25, 20), 'reflect', reflect_type='odd')
        # 创建期望的结果数组 b，用于与填充后的数组 a 进行比较
        b = np.array(
            [-25, -24, -23, -22, -21, -20, -19, -18, -17, -16,
             -15, -14, -13, -12, -11, -10, -9, -8, -7, -6,
             -5, -4, -3, -2, -1,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
             110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
            )
        # 使用 NumPy 的断言函数检查数组 a 和 b 是否相等
        assert_array_equal(a, b)

    # 定义一个测试方法 test_check_large_pad
    def test_check_large_pad(self):
        # 创建一个包含两个子列表的 Python 列表 a
        a = [[4, 5, 6], [6, 7, 8]]
        # 在列表 a 的每个子列表的上下各填充 5 个元素，左右各填充 7 个元素，填充方式为反射（reflect）
        a = np.pad(a, (5, 7), 'reflect')
        # 创建期望的结果数组 b，用于与填充后的数组 a 进行比较
        b = np.array(
            [[7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],

             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],

             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7, 8, 7, 6, 7],
             [5, 4, 5, 6, 5, 4, 5, 6, 5,
    # 定义测试函数 test_check_shape
    def test_check_shape(self):
        # 创建一个二维数组 a，包含一个子数组 [4, 5, 6]
        a = [[4, 5, 6]]
        # 使用 np.pad 函数对数组 a 进行填充，边界填充模式为 'reflect'，左右填充各 5 个，上下填充各 7 个
        a = np.pad(a, (5, 7), 'reflect')
        # 创建一个二维数组 b，包含多个子数组，每个子数组均包含相同的元素 [5, 4, 5, 6]，共有 13 行
        b = np.array(
            [[5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],

             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],

             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5],
             [5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5, 6, 5, 4, 5]]
            )
        # 断言数组 a 与数组 b 相等
        assert_array_equal(a, b)

    # 定义测试函数 test_check_01
    def test_check_01(self):
        # 使用 np.pad 函数对数组 [1, 2, 3] 进行填充，边界填充模式为 'reflect'，左右均填充 2 个
        a = np.pad([1, 2, 3], 2, 'reflect')
        # 创建一个数组 b，包含元素 [3, 2, 1, 2, 3, 2, 1]
        b = np.array([3, 2, 1, 2, 3, 2, 1])
        # 断言数组 a 与数组 b 相等
        assert_array_equal(a, b)

    # 定义测试函数 test_check_02
    def test_check_02(self):
        # 使用 np.pad 函数对数组 [1, 2, 3] 进行填充，边界填充模式为 'reflect'，左右均填充 3 个
        a = np.pad([1, 2, 3], 3, 'reflect')
        # 创建一个数组 b，包含元素 [2, 3, 2, 1, 2, 3, 2, 1, 2]
        b = np.array([2, 3, 2, 1, 2, 3, 2, 1, 2])
        # 断言数组 a 与数组 b 相等
        assert_array_equal(a, b)

    # 定义测试函数 test_check_03
    def test_check_03(self):
        # 使用 np.pad 函数对数组 [1, 2, 3] 进行填充，边界填充模式为 'reflect'，左右均填充 4 个
        a = np.pad([1, 2, 3], 4, 'reflect')
        # 创建一个数组 b，包含元素 [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3]
        b = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])
        # 断言数组 a 与数组 b 相等
        assert_array_equal(a, b)
    
    # 定义测试函数 test_check_04
    def test_check_04(self):
        # 使用 np.pad 函数对数组 [1, 2, 3] 进行填充，边界填充模式为 'reflect'，左侧填充 1 个，右侧填充 10 个
        a = np.pad([1, 2, 3], [1, 10], 'reflect')
        # 创建一个数组 b，包含元素 [2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1]
        b = np.array([2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1])
        # 断言数组 a 与数组 b 相等
        assert_array_equal(a, b)
    
    # 定义测试函数 test_check_05
    def test_check_05(self):
        # 使用 np.pad 函数对数组 [1, 2, 3, 4] 进行填充，边界填充模式为 'reflect'，左侧填充 45 个，右侧填充 10 个
        a = np.pad([1, 2, 3, 4], [45, 10], 'reflect')
        # 创建一个数组 b，包含多行元素，以 'reflect' 模式填充得到的结果
        b = np.array(
            [4, 3, 2, 1, 2, 3, 4, 3, 2, 1,
             2, 3, 4, 3, 2, 1, 2, 3, 4, 3,
             2, 1, 2, 3, 4, 3, 2, 1, 2, 3,
             4, 3, 2, 1, 2, 3, 4, 3, 2, 1,
             2, 3, 4, 3, 2, 1, 2, 3, 4, 3,
             2, 1, 2, 3, 4, 3, 2, 1, 2])
        # 断言数组 a 与数组 b 相等
        assert_array_equal(a, b)
    
    # 定义测试函数 test_check_06
    def test_check_06(self):
        # 使用 np.pad 函数对数组 [1, 2, 3, 4] 进行填充，边界填充模式为 'symmetric'，左侧填充 15 个，右侧填充 2 个
        a = np.pad([1, 2, 3, 4], [15, 2], 'symmetric')
        # 创建
class TestEmptyArray:
    """Check how padding behaves on arrays with an empty dimension."""

    @pytest.mark.parametrize(
        # Keep parametrization ordered, otherwise pytest-xdist might believe
        # that different tests were collected during parallelization
        "mode", sorted(_all_modes.keys() - {"constant", "empty"})
    )
    def test_pad_empty_dimension(self, mode):
        # 定义匹配字符串，用于验证错误消息
        match = ("can't extend empty axis 0 using modes other than 'constant' "
                 "or 'empty'")
        # 测试空数组在不同模式下填充是否会引发 ValueError 异常
        with pytest.raises(ValueError, match=match):
            np.pad([], 4, mode=mode)
        with pytest.raises(ValueError, match=match):
            np.pad(np.ndarray(0), 4, mode=mode)
        with pytest.raises(ValueError, match=match):
            np.pad(np.zeros((0, 3)), ((1,), (0,)), mode=mode)

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_pad_non_empty_dimension(self, mode):
        # 测试非空数组在不同模式下填充后形状是否符合预期
        result = np.pad(np.ones((2, 0, 2)), ((3,), (0,), (1,)), mode=mode)
        assert result.shape == (8, 0, 4)


class TestSymmetric:
    def test_check_simple(self):
        # 创建长度为 100 的一维数组 a
        a = np.arange(100)
        # 对数组 a 进行对称填充，左右各填充 25 个元素，总共填充 45 个元素
        a = np.pad(a, (25, 20), 'symmetric')
        # 创建预期结果数组 b，用于对比
        b = np.array(
            [24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
             14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
             4, 3, 2, 1, 0,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             99, 98, 97, 96, 95, 94, 93, 92, 91, 90,
             89, 88, 87, 86, 85, 84, 83, 82, 81, 80]
            )
        # 断言填充后的数组 a 是否与预期数组 b 相等
        assert_array_equal(a, b)

    def test_check_odd_method(self):
        # 创建长度为 100 的一维数组 a
        a = np.arange(100)
        # 对数组 a 进行对称填充，左右各填充 25 个元素，总共填充 45 个元素，使用奇对称
        a = np.pad(a, (25, 20), 'symmetric', reflect_type='odd')
        # 创建预期结果数组 b，用于对比
        b = np.array(
            [-24, -23, -22, -21, -20, -19, -18, -17, -16, -15,
             -14, -13, -12, -11, -10, -9, -8, -7, -6, -5,
             -4, -3, -2, -1, 0,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
             109, 110, 111, 112, 113, 114, 115, 116, 117, 118]
            )
        # 断言填充后的数组 a 是否与预期数组 b 相等
        assert_array_equal(a, b)
    # 定义测试函数 test_check_large_pad，用于测试 np.pad 函数对数组的填充操作
    def test_check_large_pad(self):
        # 创建一个二维列表 a
        a = [[4, 5, 6], [6, 7, 8]]
        # 使用 np.pad 函数对数组 a 进行对称填充，填充边界分别为 (5, 7)
        a = np.pad(a, (5, 7), 'symmetric')
        # 创建期望的填充后的数组 b
        b = np.array(
            [[5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],

             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],

             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [7, 8, 8, 7, 6, 6, 7, 8, 8, 7, 6, 6, 7, 8, 8],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6]]
        )
        # 断言填充后的数组 a 等于预期的数组 b
        assert_array_equal(a, b)

    # 定义测试函数 test_check_large_pad_odd，用于测试 np.pad 函数对数组的奇数对称填充操作
    def test_check_large_pad_odd(self):
        # 创建一个二维列表 a
        a = [[4, 5, 6], [6, 7, 8]]
        # 使用 np.pad 函数对数组 a 进行奇数对称填充，填充边界分别为 (5, 7)，反射类型为 'odd'
        a = np.pad(a, (5, 7), 'symmetric', reflect_type='odd')
        # 创建期望的填充后的数组 b
        b = np.array(
            [[-3, -2, -2, -1,  0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6],
             [-3, -2, -2, -1,  0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6],
             [-1,  0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8],
             [-1,  0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8],
             [ 1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9, 10, 10],

             [ 1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9, 10, 10],
             [ 3,  4,  4,  5,  6,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12],

             [ 3,  4,  4,  5,  6,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12],
             [ 5,  6,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12, 13, 14, 14],
             [ 5,  ```python
             [ 5,  6,  6,  7,  8,  8,  9, 10, 10, 11, 12, 12, 13, 14, 14],
             [ 7,  8,  8,  9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16],
             [ 7,  8,  8,  9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16],
             [ 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18],
             [ 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18]]
        )
        # 断言填充后的数组 a 等于预期的数组 b
        assert_array_equal(a, b)
    # 定义测试函数 test_check_shape，用于验证 np.pad 函数的对称填充功能
    def test_check_shape(self):
        # 创建二维数组 a，并对其进行对称填充，填充宽度分别为 5 和 7
        a = [[4, 5, 6]]
        a = np.pad(a, (5, 7), 'symmetric')
        # 创建期望的填充后的二维数组 b
        b = np.array(
            [[5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],

             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],

             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6],
             [5, 6, 6, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5, 6, 6]]
            )
        # 断言 a 和 b 是否相等
        assert_array_equal(a, b)

    # 定义测试函数 test_check_01，用于验证 np.pad 函数在一维数组上的对称填充功能
    def test_check_01(self):
        # 对一维数组 [1, 2, 3] 进行对称填充，填充宽度为 2
        a = np.pad([1, 2, 3], 2, 'symmetric')
        # 创建期望的填充后的数组 b
        b = np.array([2, 1, 1, 2, 3, 3, 2])
        # 断言 a 和 b 是否相等
        assert_array_equal(a, b)

    # 定义测试函数 test_check_02，用于验证 np.pad 函数在一维数组上的对称填充功能
    def test_check_02(self):
        # 对一维数组 [1, 2, 3] 进行对称填充，填充宽度为 3
        a = np.pad([1, 2, 3], 3, 'symmetric')
        # 创建期望的填充后的数组 b
        b = np.array([3, 2, 1, 1, 2, 3, 3, 2, 1])
        # 断言 a 和 b 是否相等
        assert_array_equal(a, b)

    # 定义测试函数 test_check_03，用于验证 np.pad 函数在一维数组上的对称填充功能
    def test_check_03(self):
        # 对一维数组 [1, 2, 3] 进行对称填充，填充宽度为 6
        a = np.pad([1, 2, 3], 6, 'symmetric')
        # 创建期望的填充后的数组 b
        b = np.array([1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3])
        # 断言 a 和 b 是否相等
        assert_array_equal(a, b)
class TestWrap:
    def test_check_simple(self):
        # 创建一个包含100个元素的 numpy 数组，值从 0 到 99
        a = np.arange(100)
        # 使用 'wrap' 模式对数组进行填充，左右各填充25个元素，结果数组长度为120
        a = np.pad(a, (25, 20), 'wrap')
        # 创建预期结果的 numpy 数组 b，展示了 'wrap' 模式填充后的期望值
        b = np.array(
            [75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
             85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
             95, 96, 97, 98, 99,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
             80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
             90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            )
        # 使用 numpy 的 assert_array_equal 函数验证 a 是否等于 b
        assert_array_equal(a, b)

    def test_check_01(self):
        # 对一个长度为3的列表使用 'wrap' 模式填充，左右各填充3个元素
        a = np.pad([1, 2, 3], 3, 'wrap')
        # 创建预期结果的 numpy 数组 b，展示了 'wrap' 模式填充后的期望值
        b = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        # 使用 numpy 的 assert_array_equal 函数验证 a 是否等于 b
        assert_array_equal(a, b)

    def test_check_02(self):
        # 对一个长度为3的列表使用 'wrap' 模式填充，左右各填充4个元素
        a = np.pad([1, 2, 3], 4, 'wrap')
        # 创建预期结果的 numpy 数组 b，展示了 'wrap' 模式填充后的期望值
        b = np.array([3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        # 使用 numpy 的 assert_array_equal 函数验证 a 是否等于 b
        assert_array_equal(a, b)

    def test_pad_with_zero(self):
        # 创建一个3x5的全1数组
        a = np.ones((3, 5))
        # 使用 'wrap' 模式对数组进行填充，上下各填充0行，左右各填充5列
        b = np.pad(a, (0, 5), mode="wrap")
        # 使用 numpy 的 assert_array_equal 函数验证 a 与 b 的前几行前几列是否相等
        assert_array_equal(a, b[:-5, :-5])

    def test_repeated_wrapping(self):
        """
        如果被包裹区域长度超过原始数组，检查在每个边上单独进行包裹的情况。
        """
        # 创建一个长度为5的数组
        a = np.arange(5)
        # 使用 'wrap' 模式对数组进行填充，左边填充12个元素，右边填充0个元素
        b = np.pad(a, (12, 0), mode="wrap")
        # 使用 numpy 的 assert_array_equal 函数验证 a 与 b 的结果是否相等
        assert_array_equal(np.r_[a, a, a, a][3:], b)

        # 创建一个长度为5的数组
        a = np.arange(5)
        # 使用 'wrap' 模式对数组进行填充，左边填充0个元素，右边填充12个元素
        b = np.pad(a, (0, 12), mode="wrap")
        # 使用 numpy 的 assert_array_equal 函数验证 a 与 b 的结果是否相等
        assert_array_equal(np.r_[a, a, a, a][:-3], b)
    
    def test_repeated_wrapping_multiple_origin(self):
        """
        如果填充宽度大于原始数组，则断言 'wrap' 只以原始区域的倍数填充。
        """
        # 创建一个2x2的数组
        a = np.arange(4).reshape(2, 2)
        # 使用 'wrap' 模式对数组进行填充，行上下分别填充1和3个原始区域，列左右分别填充3和1个原始区域
        a = np.pad(a, [(1, 3), (3, 1)], mode='wrap')
        # 创建预期结果的 numpy 数组 b，展示了 'wrap' 模式填充后的期望值
        b = np.array(
            [[3, 2, 3, 2, 3, 2],
             [1, 0, 1, 0, 1, 0],
             [3, 2, 3, 2, 3, 2],
             [1, 0, 1, 0, 1, 0],
             [3, 2, 3, 2, 3, 2],
             [1, 0, 1, 0, 1, 0]]
        )
        # 使用 numpy 的 assert_array_equal 函数验证 a 是否等于 b
        assert_array_equal(a, b)


class TestEdge:
    def test_check_simple(self):
        # 创建一个长度为12的数组，并将其重塑为4x3的数组
        a = np.arange(12)
        a = np.reshape(a, (4, 3))
        # 使用 'edge' 模式对数组进行填充，行上下各填充2行，列左右各填充3列
        a = np.pad(a, ((2, 3), (3, 2)), 'edge')
        # 创建预期结果的 numpy 数组 b，展示了 'edge' 模式填充后的期望值
        b = np.array(
            [[0, 0, 0, 0, 1, 2, 2, 2],
             [0, 0, 0, 0, 1, 2, 2, 2],

             [0, 0, 0, 0, 1, 2, 2, 2],
             [3, 3, 3, 3, 4, 5, 5, 5],
             [6, 6, 6, 6, 7, 8, 8, 8],
             [9, 9, 9, 9, 10, 11, 11, 11],

             [9, 9, 9, 9, 10, 11, 11, 11],
             [9, 9, 9, 9, 10, 11, 11, 11],
             [9, 9, 9, 9, 10, 11, 11, 11]]
        )
        # 使用 numpy 的 assert_array_equal 函数验证 a 是否等于 b
        assert_array_equal(a, b)
    # 定义一个测试方法，用于验证 pad_width 为 ((1, 2),) 的情况
    def test_check_width_shape_1_2(self):
        # 检查 pad_width 为 ((1, 2),) 的情况
        # 这是对问题 gh-7808 的回归测试
        # 创建一个包含整数的 NumPy 数组
        a = np.array([1, 2, 3])
        # 对数组进行边缘填充，使用 'edge' 策略
        padded = np.pad(a, ((1, 2),), 'edge')
        # 预期的填充结果
        expected = np.array([1, 1, 2, 3, 3, 3])
        # 断言填充后的结果与预期结果相等
        assert_array_equal(padded, expected)

        # 创建一个包含整数的二维 NumPy 数组
        a = np.array([[1, 2, 3], [4, 5, 6]])
        # 对数组进行边缘填充，使用 ((1, 2),) 和 'edge' 策略
        padded = np.pad(a, ((1, 2),), 'edge')
        # 预期的填充结果
        expected = np.pad(a, ((1, 2), (1, 2)), 'edge')
        # 断言填充后的结果与预期结果相等
        assert_array_equal(padded, expected)

        # 创建一个包含整数的三维 NumPy 数组
        a = np.arange(24).reshape(2, 3, 4)
        # 对数组进行边缘填充，使用 ((1, 2),) 和 'edge' 策略
        padded = np.pad(a, ((1, 2),), 'edge')
        # 预期的填充结果
        expected = np.pad(a, ((1, 2), (1, 2), (1, 2)), 'edge')
        # 断言填充后的结果与预期结果相等
        assert_array_equal(padded, expected)
class TestEmpty:
    # 定义测试方法 test_simple
    def test_simple(self):
        # 创建一个 4x6 的数组，其中元素为 0 到 23
        arr = np.arange(24).reshape(4, 6)
        # 对数组进行空填充，上方填充2行，下方填充3行，左侧填充3列，右侧填充1列
        result = np.pad(arr, [(2, 3), (3, 1)], mode="empty")
        # 断言填充后数组的形状为 (9, 10)
        assert result.shape == (9, 10)
        # 断言原始数组和填充后数组的切片（去除填充部分）相等
        assert_equal(arr, result[2:-3, 3:-1])

    # 定义测试方法 test_pad_empty_dimension
    def test_pad_empty_dimension(self):
        # 创建一个维度为 (3, 0, 2) 的全零数组
        arr = np.zeros((3, 0, 2))
        # 对数组进行空填充，第一个维度不填充，第二个维度上下各填充2列，第三个维度左右各填充1列
        result = np.pad(arr, [(0,), (2,), (1,)], mode="empty")
        # 断言填充后数组的形状为 (3, 4, 4)
        assert result.shape == (3, 4, 4)


def test_legacy_vector_functionality():
    # 定义内部函数 _padwithtens，用于向向量两端填充10
    def _padwithtens(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 10
        vector[-pad_width[1]:] = 10

    # 创建一个 2x3 的数组
    a = np.arange(6).reshape(2, 3)
    # 使用自定义的 _padwithtens 函数对数组进行填充，上下各填充2行，左右各填充2列
    a = np.pad(a, 2, _padwithtens)
    # 创建期望的结果数组 b
    b = np.array(
        [[10, 10, 10, 10, 10, 10, 10],
         [10, 10, 10, 10, 10, 10, 10],

         [10, 10,  0,  1,  2, 10, 10],
         [10, 10,  3,  4,  5, 10, 10],

         [10, 10, 10, 10, 10, 10, 10],
         [10, 10, 10, 10, 10, 10, 10]]
        )
    # 断言填充后的数组 a 与期望的数组 b 相等
    assert_array_equal(a, b)


def test_unicode_mode():
    # 使用常量填充数组，向数组 [1] 的两端各填充2个0
    a = np.pad([1], 2, mode='constant')
    # 创建期望的结果数组 b
    b = np.array([0, 0, 1, 0, 0])
    # 断言填充后的数组 a 与期望的数组 b 相等
    assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["edge", "symmetric", "reflect", "wrap"])
def test_object_input(mode):
    # 创建一个 4x3 的数组，元素均为 None
    a = np.full((4, 3), fill_value=None)
    # 设定填充量为 ((2, 3), (3, 2))
    pad_amt = ((2, 3), (3, 2))
    # 创建期望的结果数组 b
    b = np.full((9, 8), fill_value=None)
    # 断言使用不同模式填充后的数组与期望的数组 b 相等
    assert_array_equal(np.pad(a, pad_amt, mode=mode), b)


class TestPadWidth:
    @pytest.mark.parametrize("pad_width", [
        # 参数化测试用例：pad_width 参数为非法形状时的异常情况
        (4, 5, 6, 7),
        ((1,), (2,), (3,)),
        ((1, 2), (3, 4), (5, 6)),
        ((3, 4, 5), (0, 1, 2)),
    ])
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_misshaped_pad_width(self, pad_width, mode):
        # 创建一个 6x5 的数组
        arr = np.arange(30).reshape((6, 5))
        # 期望捕获到的异常信息
        match = "operands could not be broadcast together"
        # 使用 pytest 断言捕获异常，并验证异常信息
        with pytest.raises(ValueError, match=match):
            np.pad(arr, pad_width, mode)

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_misshaped_pad_width_2(self, mode):
        # 创建一个 6x5 的数组
        arr = np.arange(30).reshape((6, 5))
        # 期望捕获到的异常信息
        match = ("input operand has more dimensions than allowed by the axis "
                 "remapping")
        # 使用 pytest 断言捕获异常，并验证异常信息
        with pytest.raises(ValueError, match=match):
            np.pad(arr, (((3,), (4,), (5,)), ((0,), (1,), (2,))), mode)

    @pytest.mark.parametrize(
        "pad_width", [-2, (-2,), (3, -1), ((5, 2), (-2, 3)), ((-4,), (2,))])
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_negative_pad_width(self, pad_width, mode):
        # 创建一个 6x5 的数组
        arr = np.arange(30).reshape((6, 5))
        # 期望捕获到的异常信息
        match = "index can't contain negative values"
        # 使用 pytest 断言捕获异常，并验证异常信息
        with pytest.raises(ValueError, match=match):
            np.pad(arr, pad_width, mode)

    @pytest.mark.parametrize("pad_width, dtype", [
        ("3", None),
        ("word", None),
        (None, None),
        (object(), None),
        (3.4, None),
        (((2, 3, 4), (3, 2)), object),
        (complex(1, -1), None),
        (((-2.1, 3), (3, 2)), None),
    ])
    # 使用 pytest 的 parametrize 装饰器，参数化测试方法，测试的模式为所有可能的填充模式
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_bad_type(self, pad_width, dtype, mode):
        # 创建一个 6x5 的 NumPy 数组，内容为 0 到 29
        arr = np.arange(30).reshape((6, 5))
        # 错误匹配字符串，用于检查异常信息是否包含该字符串
        match = "`pad_width` must be of integral type."
        if dtype is not None:
            # 当 dtype 不为空时，避免 DeprecationWarning，期望抛出 TypeError 异常并匹配特定错误信息
            with pytest.raises(TypeError, match=match):
                # 使用指定的 dtype 转换 pad_width，并调用 np.pad 方法
                np.pad(arr, np.array(pad_width, dtype=dtype), mode)
        else:
            # 当 dtype 为空时，期望抛出 TypeError 异常并匹配特定错误信息
            with pytest.raises(TypeError, match=match):
                # 直接使用 pad_width 调用 np.pad 方法
                np.pad(arr, pad_width, mode)
            with pytest.raises(TypeError, match=match):
                # 将 pad_width 转换为 ndarray 后调用 np.pad 方法
                np.pad(arr, np.array(pad_width), mode)
    
    # 测试 pad_width 参数为 ndarray 的情况
    def test_pad_width_as_ndarray(self):
        # 创建一个长度为 12 的连续数组 a，并将其重塑为 4x3 的数组
        a = np.arange(12)
        a = np.reshape(a, (4, 3))
        # 使用 ndarray 格式的 pad_width 进行边缘填充 'edge'
        a = np.pad(a, np.array(((2, 3), (3, 2))), 'edge')
        # 预期的填充后结果数组 b
        b = np.array(
            [[0,  0,  0,    0,  1,  2,    2,  2],
             [0,  0,  0,    0,  1,  2,    2,  2],
             [0,  0,  0,    0,  1,  2,    2,  2],
             [3,  3,  3,    3,  4,  5,    5,  5],
             [6,  6,  6,    6,  7,  8,    8,  8],
             [9,  9,  9,    9, 10, 11,   11, 11],
             [9,  9,  9,    9, 10, 11,   11, 11],
             [9,  9,  9,    9, 10, 11,   11, 11],
             [9,  9,  9,    9, 10, 11,   11, 11]]
        )
        # 断言填充后的数组 a 与预期结果数组 b 相等
        assert_array_equal(a, b)
    
    # 使用 pytest 的 parametrize 装饰器，参数化 pad_width 和 mode 进行测试
    @pytest.mark.parametrize("pad_width", [0, (0, 0), ((0, 0), (0, 0))])
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_zero_pad_width(self, pad_width, mode):
        # 创建一个 6x5 的 NumPy 数组，内容为 0 到 29
        arr = np.arange(30).reshape(6, 5)
        # 断言调用 np.pad 方法后的结果数组与原数组 arr 相等
        assert_array_equal(arr, np.pad(arr, pad_width, mode=mode))
# 使用 pytest.mark.parametrize 装饰器，为 test_kwargs 函数添加参数化测试，参数为 _all_modes 字典的键
@pytest.mark.parametrize("mode", _all_modes.keys())
# 定义测试函数 test_kwargs，用于测试 pad 函数在给定 mode 下的行为
def test_kwargs(mode):
    # 获取当前 mode 对应的允许的关键字参数
    allowed = _all_modes[mode]
    # 创建一个空字典，用于存储所有不允许的关键字参数
    not_allowed = {}
    # 遍历 _all_modes 字典的所有值
    for kwargs in _all_modes.values():
        # 如果当前遍历到的值不等于当前 mode 对应的允许参数，则将其添加到 not_allowed 字典中
        if kwargs != allowed:
            not_allowed.update(kwargs)
    # 测试允许的关键字参数是否能够通过
    np.pad([1, 2, 3], 1, mode, **allowed)
    # 测试其他模式下不允许的关键字参数是否会引发错误
    for key, value in not_allowed.items():
        match = "unsupported keyword arguments for mode '{}'".format(mode)
        # 使用 pytest.raises 检查是否引发 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match=match):
            np.pad([1, 2, 3], 1, mode, **{key: value})


# 定义测试函数 test_constant_zero_default
def test_constant_zero_default():
    # 创建一个包含两个元素的 NumPy 数组 arr
    arr = np.array([1, 1])
    # 验证调用 np.pad 函数对 arr 进行默认常数填充时的结果
    assert_array_equal(np.pad(arr, 2), [0, 0, 1, 1, 0, 0])


# 使用 pytest.mark.parametrize 装饰器，为 test_unsupported_mode 函数添加参数化测试，参数为不支持的 mode 值
@pytest.mark.parametrize("mode", [1, "const", object(), None, True, False])
# 定义测试函数 test_unsupported_mode，用于测试 pad 函数在不支持的 mode 下是否会引发 ValueError 异常
def test_unsupported_mode(mode):
    # 创建匹配错误信息的字符串
    match = "mode '{}' is not supported".format(mode)
    # 使用 pytest.raises 检查是否引发 ValueError 异常，并匹配特定错误信息
    with pytest.raises(ValueError, match=match):
        np.pad([1, 2, 3], 4, mode=mode)


# 使用 pytest.mark.parametrize 装饰器，为 test_non_contiguous_array 函数添加参数化测试，参数为 _all_modes 字典的键
@pytest.mark.parametrize("mode", _all_modes.keys())
# 定义测试函数 test_non_contiguous_array，用于测试 pad 函数对非连续数组的填充行为
def test_non_contiguous_array(mode):
    # 创建一个非连续的 NumPy 数组 arr
    arr = np.arange(24).reshape(4, 6)[::2, ::2]
    # 对 arr 进行填充操作，保存结果到 result
    result = np.pad(arr, (2, 3), mode)
    # 验证填充后结果的形状是否为 (7, 8)
    assert result.shape == (7, 8)
    # 验证填充后结果中核心部分是否与原始 arr 的核心部分相等
    assert_equal(result[2:-3, 2:-3], arr)


# 使用 pytest.mark.parametrize 装饰器，为 test_memory_layout_persistence 函数添加参数化测试，参数为 _all_modes 字典的键
@pytest.mark.parametrize("mode", _all_modes.keys())
# 定义测试函数 test_memory_layout_persistence，用于测试 pad 函数对内存布局的保持性
def test_memory_layout_persistence(mode):
    # 创建一个形状为 (5, 10) 的全为 1 的 NumPy 数组 x，指定内存布局为 'C'（行优先）
    x = np.ones((5, 10), order='C')
    # 验证对 x 进行填充操作后结果的内存布局是否仍然为行优先
    assert np.pad(x, 5, mode).flags["C_CONTIGUOUS"]
    # 创建一个形状为 (5, 10) 的全为 1 的 NumPy 数组 x，指定内存布局为 'F'（列优先）
    x = np.ones((5, 10), order='F')
    # 验证对 x 进行填充操作后结果的内存布局是否仍然为列优先
    assert np.pad(x, 5, mode).flags["F_CONTIGUOUS"]


# 使用 pytest.mark.parametrize 装饰器，为 test_dtype_persistence 函数添加参数化测试，参数为 _numeric_dtypes 和 _all_modes 字典的键
@pytest.mark.parametrize("dtype", _numeric_dtypes)
@pytest.mark.parametrize("mode", _all_modes.keys())
# 定义测试函数 test_dtype_persistence，用于测试 pad 函数对数据类型的保持性
def test_dtype_persistence(dtype, mode):
    # 创建一个全为 0 的形状为 (3, 2, 1)，数据类型为 dtype 的 NumPy 数组 arr
    arr = np.zeros((3, 2, 1), dtype=dtype)
    # 对 arr 进行填充操作，保存结果到 result
    result = np.pad(arr, 1, mode=mode)
    # 验证填充后结果的数据类型是否与原始 arr 的数据类型相同
    assert result.dtype == dtype
```