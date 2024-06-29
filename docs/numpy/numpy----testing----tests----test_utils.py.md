# `.\numpy\numpy\testing\tests\test_utils.py`

```
import warnings
import sys
import os
import itertools
import pytest
import weakref
import re

import numpy as np
import numpy._core._multiarray_umath as ncu
from numpy.testing import (
    assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_array_less, build_err_msg,
    assert_raises, assert_warns, assert_no_warnings, assert_allclose,
    assert_approx_equal, assert_array_almost_equal_nulp, assert_array_max_ulp,
    clear_and_catch_warnings, suppress_warnings, assert_string_equal, assert_,
    tempdir, temppath, assert_no_gc_cycles, HAS_REFCOUNT
)

# 定义一个名为 _GenericTest 的基类，用于测试数组相等性
class _GenericTest:

    # 辅助方法，测试两个对象是否相等
    def _test_equal(self, a, b):
        self._assert_func(a, b)

    # 辅助方法，测试两个对象是否不相等
    def _test_not_equal(self, a, b):
        with assert_raises(AssertionError):
            self._assert_func(a, b)

    # 测试两个 rank 1 数组是否相等
    def test_array_rank1_eq(self):
        """Test two equal array of rank 1 are found equal."""
        a = np.array([1, 2])
        b = np.array([1, 2])

        self._test_equal(a, b)

    # 测试两个不同的 rank 1 数组是否不相等
    def test_array_rank1_noteq(self):
        """Test two different array of rank 1 are found not equal."""
        a = np.array([1, 2])
        b = np.array([2, 2])

        self._test_not_equal(a, b)

    # 测试两个 rank 2 数组是否相等
    def test_array_rank2_eq(self):
        """Test two equal array of rank 2 are found equal."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4]])

        self._test_equal(a, b)

    # 测试两个形状不同的数组是否不相等
    def test_array_diffshape(self):
        """Test two arrays with different shapes are found not equal."""
        a = np.array([1, 2])
        b = np.array([[1, 2], [1, 2]])

        self._test_not_equal(a, b)

    # 测试对象数组是否正确处理
    def test_objarray(self):
        """Test object arrays."""
        a = np.array([1, 1], dtype=object)
        self._test_equal(a, 1)

    # 测试类似数组的情况
    def test_array_likes(self):
        self._test_equal([1, 2, 3], (1, 2, 3))


# 定义一个测试类 TestArrayEqual，继承自 _GenericTest
class TestArrayEqual(_GenericTest):

    # 在每个测试方法执行前设置 assert_func 属性为 assert_array_equal
    def setup_method(self):
        self._assert_func = assert_array_equal

    # 测试不同类型的 rank 1 数组是否相等
    def test_generic_rank1(self):
        """Test rank 1 array for all dtypes."""
        def foo(t):
            # 创建一个指定类型 t 的空数组，填充为 1
            a = np.empty(2, t)
            a.fill(1)
            # 复制数组 a 到 b 和 c
            b = a.copy()
            c = a.copy()
            # 数组 c 填充为 0
            c.fill(0)
            # 测试 a 和 b 是否相等
            self._test_equal(a, b)
            # 测试 c 和 b 是否不相等
            self._test_not_equal(c, b)

        # 测试数值类型和对象类型
        for t in '?bhilqpBHILQPfdgFDG':
            foo(t)

        # 测试字符串类型
        for t in ['S1', 'U1']:
            foo(t)
    def test_0_ndim_array(self):
        # 创建包含一个大整数的 NumPy 数组 x
        x = np.array(473963742225900817127911193656584771)
        # 创建包含一个较小整数的 NumPy 数组 y
        y = np.array(18535119325151578301457182298393896)

        # 使用 pytest 检查是否会抛出 AssertionError 异常
        with pytest.raises(AssertionError) as exc_info:
            # 调用 self._assert_func 进行断言
            self._assert_func(x, y)
        # 获取异常信息的字符串表示
        msg = str(exc_info.value)
        # 断言异常信息中包含特定的字符串
        assert_('Mismatched elements: 1 / 1 (100%)\n'
                in msg)

        # 将 y 赋值为 x，以便进行下一次断言
        y = x
        # 调用 self._assert_func 进行断言
        self._assert_func(x, y)

        # 创建包含一个浮点数的 NumPy 数组 x
        x = np.array(4395065348745.5643764887869876)
        # 创建包含整数 0 的 NumPy 数组 y
        y = np.array(0)
        # 预期的异常消息，包含详细的差异信息
        expected_msg = ('Mismatched elements: 1 / 1 (100%)\n'
                        'Max absolute difference among violations: '
                        '4.39506535e+12\n'
                        'Max relative difference among violations: inf\n')
        # 使用 pytest 检查是否会抛出 AssertionError 异常，并匹配预期的异常消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            # 调用 self._assert_func 进行断言
            self._assert_func(x, y)

        # 将 x 赋值为 y，以便进行下一次断言
        x = y
        # 调用 self._assert_func 进行断言
        self._assert_func(x, y)

    def test_generic_rank3(self):
        """Test rank 3 array for all dtypes."""
        def foo(t):
            # 创建一个空的 NumPy 数组 a，形状为 (4, 2, 3)，数据类型为 t，并填充为 1
            a = np.empty((4, 2, 3), t)
            a.fill(1)
            # 创建数组 b，复制自数组 a
            b = a.copy()
            # 创建数组 c，复制自数组 a，并填充为 0
            c = a.copy()
            c.fill(0)
            # 调用 self._test_equal 检查数组 a 和 b 是否相等
            self._test_equal(a, b)
            # 调用 self._test_not_equal 检查数组 c 和 b 是否不相等

        # 测试所有数字类型和对象类型
        for t in '?bhilqpBHILQPfdgFDG':
            foo(t)

        # 测试字符串类型
        for t in ['S1', 'U1']:
            foo(t)

    def test_nan_array(self):
        """Test arrays with nan values in them."""
        # 创建包含 NaN 值的 NumPy 数组 a
        a = np.array([1, 2, np.nan])
        # 创建包含 NaN 值的 NumPy 数组 b，与 a 相同
        b = np.array([1, 2, np.nan])

        # 调用 self._test_equal 检查数组 a 和 b 是否相等
        self._test_equal(a, b)

        # 创建不包含 NaN 值的 NumPy 数组 c
        c = np.array([1, 2, 3])
        # 调用 self._test_not_equal 检查数组 c 和 b 是否不相等
        self._test_not_equal(c, b)

    def test_string_arrays(self):
        """Test two arrays with different shapes are found not equal."""
        # 创建字符串数组 a
        a = np.array(['floupi', 'floupa'])
        # 创建字符串数组 b，与 a 相同
        b = np.array(['floupi', 'floupa'])

        # 调用 self._test_equal 检查数组 a 和 b 是否相等

        # 创建具有不同形状的字符串数组 c
        c = np.array(['floupipi', 'floupa'])
        # 调用 self._test_not_equal 检查数组 c 和 b 是否不相等

    def test_recarrays(self):
        """Test record arrays."""
        # 创建一个空的记录数组 a，包含两个字段 'floupi' 和 'floupa'
        a = np.empty(2, [('floupi', float), ('floupa', float)])
        a['floupi'] = [1, 2]
        a['floupa'] = [1, 2]
        # 复制记录数组 a 到数组 b
        b = a.copy()

        # 调用 self._test_equal 检查数组 a 和 b 是否相等

        # 创建一个空的记录数组 c，包含三个字段 'floupipi'、'floupi' 和 'floupa'
        c = np.empty(2, [('floupipi', float),
                         ('floupi', float), ('floupa', float)])
        c['floupipi'] = a['floupi'].copy()
        c['floupa'] = a['floupa'].copy()
        # 使用 pytest 检查是否会抛出 TypeError 异常
        with pytest.raises(TypeError):
            # 调用 self._test_not_equal 检查数组 c 和 b 是否不相等

    def test_masked_nan_inf(self):
        # Regression test for gh-11121
        # 创建包含掩码和 NaN 值的掩码数组 a
        a = np.ma.MaskedArray([3., 4., 6.5], mask=[False, True, False])
        # 创建包含 NaN 值的 NumPy 数组 b
        b = np.array([3., np.nan, 6.5])
        # 调用 self._test_equal 检查数组 a 和 b 是否相等
        self._test_equal(a, b)
        # 调用 self._test_equal 检查数组 b 和 a 是否相等
        self._test_equal(b, a)

        # 创建包含掩码和无穷大值的掩码数组 a
        a = np.ma.MaskedArray([3., 4., 6.5], mask=[True, False, False])
        # 创建包含无穷大值的 NumPy 数组 b
        b = np.array([np.inf, 4., 6.5])
        # 调用 self._test_equal 检查数组 a 和 b 是否相等
        self._test_equal(a, b)
        # 调用 self._test_equal 检查数组 b 和 a 是否相等
        self._test_equal(b, a)
    def test_subclass_that_overrides_eq(self):
        # 定义一个测试子类，覆盖了 __eq__ 方法来自定义相等比较行为
        # 这种子类不依赖于能够存储布尔值（例如 astropy Quantity 无法有效使用布尔值）
        # 参见 GitHub 问题 gh-8452。
        class MyArray(np.ndarray):
            def __eq__(self, other):
                return bool(np.equal(self, other).all())

            def __ne__(self, other):
                return not self == other

        # 创建两个 MyArray 的实例
        a = np.array([1., 2.]).view(MyArray)
        b = np.array([2., 3.]).view(MyArray)

        # 断言 a == a 的返回类型是布尔值
        assert_(type(a == a), bool)
        # 断言 a 等于自己
        assert_(a == a)
        # 断言 a 不等于 b
        assert_(a != b)

        # 调用测试函数，测试相等性
        self._test_equal(a, a)
        # 调用测试函数，测试不相等性
        self._test_not_equal(a, b)
        # 调用测试函数，测试不相等性（反向）
        self._test_not_equal(b, a)

        # 准备预期的错误消息
        expected_msg = ('Mismatched elements: 1 / 2 (50%)\n'
                        'Max absolute difference among violations: 1.\n'
                        'Max relative difference among violations: 0.5')
        # 使用 pytest 检查是否会抛出 AssertionError，并匹配预期的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._test_equal(a, b)

        # 创建另一个 MyArray 的实例 c
        c = np.array([0., 2.9]).view(MyArray)
        # 准备另一个预期的错误消息
        expected_msg = ('Mismatched elements: 1 / 2 (50%)\n'
                        'Max absolute difference among violations: 2.\n'
                        'Max relative difference among violations: inf')
        # 使用 pytest 检查是否会抛出 AssertionError，并匹配预期的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._test_equal(b, c)

    def test_subclass_that_does_not_implement_npall(self):
        # 定义一个不实现 np.all 方法的测试子类
        class MyArray(np.ndarray):
            def __array_function__(self, *args, **kwargs):
                return NotImplemented

        # 创建两个 MyArray 的实例
        a = np.array([1., 2.]).view(MyArray)
        b = np.array([2., 3.]).view(MyArray)

        # 使用 assert_raises 检查是否会抛出 TypeError
        with assert_raises(TypeError):
            np.all(a)

        # 调用测试函数，测试相等性
        self._test_equal(a, a)
        # 调用测试函数，测试不相等性
        self._test_not_equal(a, b)
        # 调用测试函数，测试不相等性（反向）
        self._test_not_equal(b, a)

    def test_suppress_overflow_warnings(self):
        # 基于问题 #18992 进行测试，验证是否会抑制溢出警告
        with pytest.raises(AssertionError):
            with np.errstate(all="raise"):
                np.testing.assert_array_equal(
                    np.array([1, 2, 3], np.float32),
                    np.array([1, 1e-40, 3], np.float32))

    def test_array_vs_scalar_is_equal(self):
        """测试当所有值相等时，比较数组和标量的相等性。"""
        # 创建数组 a 和标量 b
        a = np.array([1., 1., 1.])
        b = 1.

        # 调用测试函数，测试相等性
        self._test_equal(a, b)
    def test_array_vs_array_not_equal(self):
        """Test comparing an array with a scalar when not all values equal."""
        # 创建一个包含整数的 NumPy 数组
        a = np.array([34986, 545676, 439655, 563766])
        # 创建另一个包含整数和一个零的 NumPy 数组
        b = np.array([34986, 545676, 439655, 0])

        # 预期的错误信息，描述了不匹配元素的数量和最大差异
        expected_msg = ('Mismatched elements: 1 / 4 (25%)\n'
                        'Max absolute difference among violations: 563766\n'
                        'Max relative difference among violations: inf')
        # 使用 pytest 来验证预期的 AssertionError，并匹配指定的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(a, b)

        # 创建一个包含浮点数和一个整数的 NumPy 数组
        a = np.array([34986, 545676, 439655.2, 563766])
        # 更新预期的错误信息，描述了不匹配元素的数量和最大差异
        expected_msg = ('Mismatched elements: 2 / 4 (50%)\n'
                        'Max absolute difference among violations: '
                        '563766.\n'
                        'Max relative difference among violations: '
                        '4.54902139e-07')
        # 使用 pytest 来验证预期的 AssertionError，并匹配指定的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(a, b)

    def test_array_vs_scalar_strict(self):
        """Test comparing an array with a scalar with strict option."""
        # 创建一个包含浮点数的 NumPy 数组
        a = np.array([1., 1., 1.])
        # 创建一个浮点数标量
        b = 1.

        # 使用 pytest 来验证预期的 AssertionError
        with pytest.raises(AssertionError):
            self._assert_func(a, b, strict=True)

    def test_array_vs_array_strict(self):
        """Test comparing two arrays with strict option."""
        # 创建两个相同的包含浮点数的 NumPy 数组
        a = np.array([1., 1., 1.])
        b = np.array([1., 1., 1.])

        # 调用 _assert_func 函数来比较这两个数组，使用 strict=True 选项
        self._assert_func(a, b, strict=True)

    def test_array_vs_float_array_strict(self):
        """Test comparing two arrays with strict option."""
        # 创建一个包含整数的 NumPy 数组
        a = np.array([1, 1, 1])
        # 创建一个包含浮点数的 NumPy 数组
        b = np.array([1., 1., 1.])

        # 使用 pytest 来验证预期的 AssertionError
        with pytest.raises(AssertionError):
            self._assert_func(a, b, strict=True)
class TestBuildErrorMessage:

    def test_build_err_msg_defaults(self):
        # 创建两个 NumPy 数组作为比较的输入数据
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        # 错误消息字符串
        err_msg = 'There is a mismatch'

        # 调用 build_err_msg 函数生成错误消息，包含详细信息
        a = build_err_msg([x, y], err_msg)
        # 期望的错误消息，包含详细的比较结果
        b = ('\nItems are not equal: There is a mismatch\n ACTUAL: array(['
             '1.00001, 2.00002, 3.00003])\n DESIRED: array([1.00002, '
             '2.00003, 3.00004])')
        # 使用 assert_equal 断言 a 和 b 相等
        assert_equal(a, b)

    def test_build_err_msg_no_verbose(self):
        # 创建两个 NumPy 数组作为比较的输入数据
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        # 错误消息字符串
        err_msg = 'There is a mismatch'

        # 调用 build_err_msg 函数生成简单错误消息，不包含详细信息
        a = build_err_msg([x, y], err_msg, verbose=False)
        # 期望的简单错误消息
        b = '\nItems are not equal: There is a mismatch'
        # 使用 assert_equal 断言 a 和 b 相等
        assert_equal(a, b)

    def test_build_err_msg_custom_names(self):
        # 创建两个 NumPy 数组作为比较的输入数据
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        # 错误消息字符串
        err_msg = 'There is a mismatch'

        # 调用 build_err_msg 函数生成带有自定义命名的错误消息
        a = build_err_msg([x, y], err_msg, names=('FOO', 'BAR'))
        # 期望的带有自定义命名的错误消息，包含详细的比较结果
        b = ('\nItems are not equal: There is a mismatch\n FOO: array(['
             '1.00001, 2.00002, 3.00003])\n BAR: array([1.00002, 2.00003, '
             '3.00004])')
        # 使用 assert_equal 断言 a 和 b 相等
        assert_equal(a, b)

    def test_build_err_msg_custom_precision(self):
        # 创建两个 NumPy 数组作为比较的输入数据，其中一个有更高的精度
        x = np.array([1.000000001, 2.00002, 3.00003])
        y = np.array([1.000000002, 2.00003, 3.00004])
        # 错误消息字符串
        err_msg = 'There is a mismatch'

        # 调用 build_err_msg 函数生成带有自定义精度的错误消息
        a = build_err_msg([x, y], err_msg, precision=10)
        # 期望的带有自定义精度的错误消息，包含详细的比较结果
        b = ('\nItems are not equal: There is a mismatch\n ACTUAL: array(['
             '1.000000001, 2.00002    , 3.00003    ])\n DESIRED: array(['
             '1.000000002, 2.00003    , 3.00004    ])')
        # 使用 assert_equal 断言 a 和 b 相等
        assert_equal(a, b)


class TestEqual(TestArrayEqual):

    def setup_method(self):
        # 设置测试中使用的断言函数为 assert_equal
        self._assert_func = assert_equal

    def test_nan_items(self):
        # 测试 NaN 值的相等性断言
        self._assert_func(np.nan, np.nan)
        self._assert_func([np.nan], [np.nan])
        # 测试 NaN 值的不等性断言
        self._test_not_equal(np.nan, [np.nan])
        self._test_not_equal(np.nan, 1)

    def test_inf_items(self):
        # 测试 Infinity 值的相等性断言
        self._assert_func(np.inf, np.inf)
        self._assert_func([np.inf], [np.inf])
        # 测试 Infinity 值的不等性断言
        self._test_not_equal(np.inf, [np.inf])

    def test_datetime(self):
        # 测试日期时间对象的相等性断言
        self._test_equal(
            np.datetime64("2017-01-01", "s"),
            np.datetime64("2017-01-01", "s")
        )
        self._test_equal(
            np.datetime64("2017-01-01", "s"),
            np.datetime64("2017-01-01", "m")
        )

        # gh-10081
        # 测试日期时间对象的不等性断言
        self._test_not_equal(
            np.datetime64("2017-01-01", "s"),
            np.datetime64("2017-01-02", "s")
        )
        self._test_not_equal(
            np.datetime64("2017-01-01", "s"),
            np.datetime64("2017-01-02", "m")
        )
    # 定义测试函数，验证非有效日期时间对象
    def test_nat_items(self):
        # 创建不是日期时间对象的 NaT（Not a Time）
        nadt_no_unit = np.datetime64("NaT")
        # 创建秒精度的 NaT
        nadt_s = np.datetime64("NaT", "s")
        # 创建纳秒精度的 NaT
        nadt_d = np.datetime64("NaT", "ns")
        
        # 创建不是时间差对象的 NaT（Not a Timedelta）
        natd_no_unit = np.timedelta64("NaT")
        # 创建秒精度的 NaT
        natd_s = np.timedelta64("NaT", "s")
        # 创建纳秒精度的 NaT
        natd_d = np.timedelta64("NaT", "ns")
        
        # 构建日期时间对象和时间差对象列表
        dts = [nadt_no_unit, nadt_s, nadt_d]
        tds = [natd_no_unit, natd_s, natd_d]
        
        # 遍历日期时间对象列表进行各种断言和不相等测试
        for a, b in itertools.product(dts, dts):
            self._assert_func(a, b)
            self._assert_func([a], [b])
            self._test_not_equal([a], b)
        
        # 遍历时间差对象列表进行各种断言和不相等测试
        for a, b in itertools.product(tds, tds):
            self._assert_func(a, b)
            self._assert_func([a], [b])
            self._test_not_equal([a], b)
        
        # 遍历日期时间对象和时间差对象列表进行不相等测试
        for a, b in itertools.product(tds, dts):
            self._test_not_equal(a, b)
            self._test_not_equal(a, [b])
            self._test_not_equal([a], [b])
            self._test_not_equal([a], np.datetime64("2017-01-01", "s"))
            self._test_not_equal([b], np.datetime64("2017-01-01", "s"))
            self._test_not_equal([a], np.timedelta64(123, "s"))
            self._test_not_equal([b], np.timedelta64(123, "s"))

    # 验证非数值对象
    def test_non_numeric(self):
        self._assert_func('ab', 'ab')
        self._test_not_equal('ab', 'abb')

    # 验证复杂对象
    def test_complex_item(self):
        self._assert_func(complex(1, 2), complex(1, 2))
        self._assert_func(complex(1, np.nan), complex(1, np.nan))
        self._test_not_equal(complex(1, np.nan), complex(1, 2))
        self._test_not_equal(complex(np.nan, 1), complex(1, np.nan))
        self._test_not_equal(complex(np.nan, np.inf), complex(np.nan, 2))

    # 验证负零
    def test_negative_zero(self):
        self._test_not_equal(ncu.PZERO, ncu.NZERO)

    # 验证复杂对象数组
    def test_complex(self):
        x = np.array([complex(1, 2), complex(1, np.nan)])
        y = np.array([complex(1, 2), complex(1, 2)])
        self._assert_func(x, x)
        self._test_not_equal(x, y)

    # 验证对象数组
    def test_object(self):
        # gh-12942
        import datetime
        a = np.array([datetime.datetime(2000, 1, 1),
                      datetime.datetime(2000, 1, 2)])
        self._test_not_equal(a, a[::-1])
class TestArrayAlmostEqual(_GenericTest):

    def setup_method(self):
        # 设置断言函数为 assert_array_almost_equal
        self._assert_func = assert_array_almost_equal

    def test_closeness(self):
        # 在漫长的时间过程中，我们发现
        #     `abs(x - y) < 1.5 * 10**(-decimal)`
        # 取代了先前记录的
        #     `abs(x - y) < 0.5 * 10**(-decimal)`
        # 因此，这个检查旨在保留错误。

        # 测试标量
        expected_msg = ('Mismatched elements: 1 / 1 (100%)\n'
                        'Max absolute difference among violations: 1.5\n'
                        'Max relative difference among violations: inf')
        # 使用 pytest 的断言来确保抛出 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(1.5, 0.0, decimal=0)

        # 测试数组
        self._assert_func([1.499999], [0.0], decimal=0)

        expected_msg = ('Mismatched elements: 1 / 1 (100%)\n'
                        'Max absolute difference among violations: 1.5\n'
                        'Max relative difference among violations: inf')
        # 再次使用 pytest 的断言来确保抛出 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func([1.5], [0.0], decimal=0)

        a = [1.4999999, 0.00003]
        b = [1.49999991, 0]
        expected_msg = ('Mismatched elements: 1 / 2 (50%)\n'
                        'Max absolute difference among violations: 3.e-05\n'
                        'Max relative difference among violations: inf')
        # 使用 pytest 的断言来确保抛出 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(a, b, decimal=7)

        expected_msg = ('Mismatched elements: 1 / 2 (50%)\n'
                        'Max absolute difference among violations: 3.e-05\n'
                        'Max relative difference among violations: 1.')
        # 再次使用 pytest 的断言来确保抛出 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(b, a, decimal=7)

    def test_simple(self):
        x = np.array([1234.2222])
        y = np.array([1234.2223])

        # 使用断言函数进行比较，精度为 3
        self._assert_func(x, y, decimal=3)
        # 使用断言函数进行比较，精度为 4
        self._assert_func(x, y, decimal=4)

        expected_msg = ('Mismatched elements: 1 / 1 (100%)\n'
                        'Max absolute difference among violations: '
                        '1.e-04\n'
                        'Max relative difference among violations: '
                        '8.10226812e-08')
        # 使用 pytest 的断言来确保抛出 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y, decimal=5)
    # 定义测试函数，用于比较数组和标量的断言
    def test_array_vs_scalar(self):
        # 创建数组 a 和标量 b
        a = [5498.42354, 849.54345, 0.00]
        b = 5498.42354
        # 定义预期的错误消息，包含预期的差异统计信息
        expected_msg = ('Mismatched elements: 2 / 3 (66.7%)\n'
                        'Max absolute difference among violations: '
                        '5498.42354\n'
                        'Max relative difference among violations: 1.')
        # 使用 pytest 断言检查调用 self._assert_func(a, b, decimal=9) 是否会引发 AssertionError，并匹配预期的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(a, b, decimal=9)

        # 更新预期的错误消息，改变 b 和 a 的位置进行比较
        expected_msg = ('Mismatched elements: 2 / 3 (66.7%)\n'
                        'Max absolute difference among violations: '
                        '5498.42354\n'
                        'Max relative difference among violations: 5.4722099')
        # 使用 pytest 断言检查调用 self._assert_func(b, a, decimal=9) 是否会引发 AssertionError，并匹配预期的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(b, a, decimal=9)

        # 更新数组 a 和预期的错误消息，调整精度 decimal=7
        a = [5498.42354, 0.00]
        expected_msg = ('Mismatched elements: 1 / 2 (50%)\n'
                        'Max absolute difference among violations: '
                        '5498.42354\n'
                        'Max relative difference among violations: inf')
        # 使用 pytest 断言检查调用 self._assert_func(b, a, decimal=7) 是否会引发 AssertionError，并匹配预期的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(b, a, decimal=7)

        # 更新标量 b 和预期的错误消息，调整精度 decimal=7
        b = 0
        expected_msg = ('Mismatched elements: 1 / 2 (50%)\n'
                        'Max absolute difference among violations: '
                        '5498.42354\n'
                        'Max relative difference among violations: inf')
        # 使用 pytest 断言检查调用 self._assert_func(a, b, decimal=7) 是否会引发 AssertionError，并匹配预期的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(a, b, decimal=7)

    # 测试处理 NaN 值的情况
    def test_nan(self):
        # 创建包含 NaN 的数组 anan 和包含数值 1 的数组 aone
        anan = np.array([np.nan])
        aone = np.array([1])
        # 断言调用 self._assert_func(anan, anan) 没有引发异常
        self._assert_func(anan, anan)
        # 使用 assert_raises 检查调用 self._assert_func(anan, aone) 是否会引发 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func(anan, aone))
        # 使用 assert_raises 检查调用 self._assert_func(anan, ainf) 是否会引发 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func(anan, ainf))
        # 使用 assert_raises 检查调用 self._assert_func(ainf, anan) 是否会引发 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func(ainf, anan))

    # 测试处理无穷大值的情况
    def test_inf(self):
        # 创建数组 a 和其副本 b，修改 a 中的一个元素为 np.inf
        a = np.array([[1., 2.], [3., 4.]])
        b = a.copy()
        a[0, 0] = np.inf
        # 使用 assert_raises 检查调用 self._assert_func(a, b) 是否会引发 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func(a, b))
        # 将 b 中相同位置的元素修改为 -np.inf
        b[0, 0] = -np.inf
        # 使用 assert_raises 检查调用 self._assert_func(a, b) 是否会引发 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func(a, b))
    # 定义一个测试方法，用于测试特定的子类行为
    def test_subclass(self):
        # 创建一个普通的 NumPy 数组 a
        a = np.array([[1., 2.], [3., 4.]])
        # 创建一个带掩码的 NumPy 掩码数组 b
        b = np.ma.masked_array([[1., 2.], [0., 4.]],
                               [[False, False], [True, False]])
        # 断言 a 和 b 通过自定义的断言函数 _assert_func 相等
        self._assert_func(a, b)
        # 断言 b 和 a 通过自定义的断言函数 _assert_func 相等
        self._assert_func(b, a)
        # 断言 b 和 b 通过自定义的断言函数 _assert_func 相等
        self._assert_func(b, b)

        # 测试完全掩码的情况（参见 gh-11123）
        # 创建一个完全掩码的 NumPy 掩码数组 a
        a = np.ma.MaskedArray(3.5, mask=True)
        # 创建一个普通的 NumPy 数组 b
        b = np.array([3., 4., 6.5])
        # 断言 a 和 b 通过自定义的测试相等函数 _test_equal 相等
        self._test_equal(a, b)
        # 断言 b 和 a 通过自定义的测试相等函数 _test_equal 相等
        self._test_equal(b, a)
        # 创建一个未定义掩码的掩码数组 a
        a = np.ma.masked
        # 创建一个普通的 NumPy 数组 b
        b = np.array([3., 4., 6.5])
        # 断言 a 和 b 通过自定义的测试相等函数 _test_equal 相等
        self._test_equal(a, b)
        # 断言 b 和 a 通过自定义的测试相等函数 _test_equal 相等
        self._test_equal(b, a)
        # 创建一个带部分掩码的 NumPy 掩码数组 a
        a = np.ma.MaskedArray([3., 4., 6.5], mask=[True, True, True])
        # 创建一个普通的 NumPy 数组 b
        b = np.array([1., 2., 3.])
        # 断言 a 和 b 通过自定义的测试相等函数 _test_equal 相等
        self._test_equal(a, b)
        # 断言 b 和 a 通过自定义的测试相等函数 _test_equal 相等
        self._test_equal(b, a)
        # 创建一个带部分掩码的 NumPy 掩码数组 a
        a = np.ma.MaskedArray([3., 4., 6.5], mask=[True, True, True])
        # 创建一个普通的 NumPy 数组 b
        b = np.array(1.)
        # 断言 a 和 b 通过自定义的测试相等函数 _test_equal 相等
        self._test_equal(a, b)
        # 断言 b 和 a 通过自定义的测试相等函数 _test_equal 相等
        self._test_equal(b, a)

    # 定义另一个测试方法，测试无法用布尔值存储的子类行为
    def test_subclass_2(self):
        # 尽管我们无法保证测试函数总是适用于子类，测试理想情况下应该仅依赖于子类具有比较运算符，
        # 而不是依赖它们能够存储布尔值（例如 astropy Quantity 无法有用地执行此操作）。参见 gh-8452。
        class MyArray(np.ndarray):
            def __eq__(self, other):
                return super().__eq__(other).view(np.ndarray)

            def __lt__(self, other):
                return super().__lt__(other).view(np.ndarray)

            def all(self, *args, **kwargs):
                return all(self)

        # 创建一个视图为 MyArray 的普通 NumPy 数组 a
        a = np.array([1., 2.]).view(MyArray)
        # 断言 a 和 a 通过自定义的断言函数 _assert_func 相等
        self._assert_func(a, a)

        # 创建一个视图为 MyArray 的布尔值数组 z
        z = np.array([True, True]).view(MyArray)
        # 调用 all 方法，这里并未使用其返回值
        all(z)
        # 创建一个视图为 MyArray 的普通 NumPy 数组 b
        b = np.array([1., 202]).view(MyArray)
        # 创建预期的错误消息
        expected_msg = ('Mismatched elements: 1 / 2 (50%)\n'
                        'Max absolute difference among violations: 200.\n'
                        'Max relative difference among violations: 0.99009')
        # 使用 pytest 断言捕获 AssertionError 异常，并匹配预期消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(a, b)

    # 定义另一个测试方法，测试无法用布尔值存储的子类行为
    def test_subclass_that_cannot_be_bool(self):
        # 尽管我们无法保证测试函数总是适用于子类，测试理想情况下应该仅依赖于子类具有比较运算符，
        # 而不是依赖它们能够存储布尔值（例如 astropy Quantity 无法有用地执行此操作）。参见 gh-8452。
        class MyArray(np.ndarray):
            def __eq__(self, other):
                return super().__eq__(other).view(np.ndarray)

            def __lt__(self, other):
                return super().__lt__(other).view(np.ndarray)

            def all(self, *args, **kwargs):
                raise NotImplementedError

        # 创建一个视图为 MyArray 的普通 NumPy 数组 a
        a = np.array([1., 2.]).view(MyArray)
        # 断言 a 和 a 通过自定义的断言函数 _assert_func 相等
        self._assert_func(a, a)
class TestAlmostEqual(_GenericTest):

    def setup_method(self):
        # 设置断言函数为 assert_almost_equal
        self._assert_func = assert_almost_equal

    def test_closeness(self):
        # 测试近似性
        # 注意，随着时间推移，我们将
        #     `abs(x - y) < 1.5 * 10**(-decimal)`
        # 替换了之前文档中的
        #     `abs(x - y) < 0.5 * 10**(-decimal)`
        # 因此这个检查用来保留错误行为。

        # 测试标量值
        self._assert_func(1.499999, 0.0, decimal=0)
        # 断言会抛出 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func(1.5, 0.0, decimal=0))

        # 测试数组
        self._assert_func([1.499999], [0.0], decimal=0)
        # 断言会抛出 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func([1.5], [0.0], decimal=0))

    def test_nan_item(self):
        # 测试 NaN 值
        self._assert_func(np.nan, np.nan)
        # 断言会抛出 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func(np.nan, 1))
        assert_raises(AssertionError,
                      lambda: self._assert_func(np.nan, np.inf))
        assert_raises(AssertionError,
                      lambda: self._assert_func(np.inf, np.nan))

    def test_inf_item(self):
        # 测试无穷大值
        self._assert_func(np.inf, np.inf)
        self._assert_func(-np.inf, -np.inf)
        # 断言会抛出 AssertionError
        assert_raises(AssertionError,
                      lambda: self._assert_func(np.inf, 1))
        assert_raises(AssertionError,
                      lambda: self._assert_func(-np.inf, np.inf))

    def test_simple_item(self):
        # 测试简单值的不相等情况
        self._test_not_equal(1, 2)

    def test_complex_item(self):
        # 测试复数值
        self._assert_func(complex(1, 2), complex(1, 2))
        self._assert_func(complex(1, np.nan), complex(1, np.nan))
        self._assert_func(complex(np.inf, np.nan), complex(np.inf, np.nan))
        self._test_not_equal(complex(1, np.nan), complex(1, 2))
        self._test_not_equal(complex(np.nan, 1), complex(1, np.nan))
        self._test_not_equal(complex(np.nan, np.inf), complex(np.nan, 2))

    def test_complex(self):
        x = np.array([complex(1, 2), complex(1, np.nan)])
        z = np.array([complex(1, 2), complex(np.nan, 1)])
        y = np.array([complex(1, 2), complex(1, 2)])
        self._assert_func(x, x)
        self._test_not_equal(x, y)
        self._test_not_equal(x, z)
    # 定义一个测试方法，用于检查错误消息的格式化是否正确，特别是针对十进制值进行检查。
    # 还要检查包含 inf 或 nan 的输入时的错误消息 (gh12200)。
    def test_error_message(self):
        """Check the message is formatted correctly for the decimal value.
           Also check the message when input includes inf or nan (gh12200)"""

        # 创建两个 NumPy 数组，包含浮点数值，用于测试
        x = np.array([1.00000000001, 2.00000000002, 3.00003])
        y = np.array([1.00000000002, 2.00000000003, 3.00004])

        # 测试带有不同小数位数的情况
        expected_msg = ('Mismatched elements: 3 / 3 (100%)\n'
                        'Max absolute difference among violations: 1.e-05\n'
                        'Max relative difference among violations: '
                        '3.33328889e-06\n'
                        ' ACTUAL: array([1.00000000001, '
                        '2.00000000002, '
                        '3.00003      ])\n'
                        ' DESIRED: array([1.00000000002, 2.00000000003, '
                        '3.00004      ])')

        # 使用 pytest 的断言，验证是否抛出预期的 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y, decimal=12)

        # 使用默认的小数位数测试，只有第三个元素不同。注意，这里仅检查数组本身的格式化。
        expected_msg = ('Mismatched elements: 1 / 3 (33.3%)\n'
                        'Max absolute difference among violations: 1.e-05\n'
                        'Max relative difference among violations: '
                        '3.33328889e-06\n'
                        ' ACTUAL: array([1.     , 2.     , 3.00003])\n'
                        ' DESIRED: array([1.     , 2.     , 3.00004])')

        # 使用 pytest 的断言，验证是否抛出预期的 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)

        # 检查包含 inf 输入时的错误消息
        x = np.array([np.inf, 0])
        y = np.array([np.inf, 1])

        expected_msg = ('Mismatched elements: 1 / 2 (50%)\n'
                        'Max absolute difference among violations: 1.\n'
                        'Max relative difference among violations: 1.\n'
                        ' ACTUAL: array([inf,  0.])\n'
                        ' DESIRED: array([inf,  1.])')

        # 使用 pytest 的断言，验证是否抛出预期的 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)

        # 检查除以零时的错误消息
        x = np.array([1, 2])
        y = np.array([0, 0])

        expected_msg = ('Mismatched elements: 2 / 2 (100%)\n'
                        'Max absolute difference among violations: 2\n'
                        'Max relative difference among violations: inf')

        # 使用 pytest 的断言，验证是否抛出预期的 AssertionError，并匹配预期的消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)
    def test_error_message_2(self):
        """检查消息格式是否正确"""
        """当 x 或 y 中至少有一个是标量时。"""
        # 设置变量 x 为整数 2
        x = 2
        # 设置变量 y 为包含 20 个值为 1 的数组
        y = np.ones(20)
        # 设置期望的错误消息，用于匹配断言错误
        expected_msg = ('Mismatched elements: 20 / 20 (100%)\n'
                        'Max absolute difference among violations: 1.\n'
                        'Max relative difference among violations: 1.')
        # 使用 pytest 断言预期的 AssertionError，并匹配预期的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)

        # 将变量 y 设置为整数 2
        y = 2
        # 将变量 x 设置为包含 20 个值为 1 的数组
        x = np.ones(20)
        # 更新期望的错误消息，用于匹配断言错误
        expected_msg = ('Mismatched elements: 20 / 20 (100%)\n'
                        'Max absolute difference among violations: 1.\n'
                        'Max relative difference among violations: 0.5')
        # 使用 pytest 断言预期的 AssertionError，并匹配预期的错误消息
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)

    def test_subclass_that_cannot_be_bool(self):
        # 虽然我们无法保证测试函数始终适用于所有子类，
        # 但测试应该仅依赖于子类具有比较运算符，而不是依赖于它们能够存储布尔值
        # （例如，astropy Quantity 不能有用地执行此操作）。参见 gh-8452。
        # 定义一个自定义子类 MyArray，继承自 np.ndarray
        class MyArray(np.ndarray):
            # 重载等于运算符，返回 np.ndarray 视图
            def __eq__(self, other):
                return super().__eq__(other).view(np.ndarray)

            # 重载小于运算符，返回 np.ndarray 视图
            def __lt__(self, other):
                return super().__lt__(other).view(np.ndarray)

            # 定义 all 方法，抛出 NotImplementedError
            def all(self, *args, **kwargs):
                raise NotImplementedError

        # 创建一个包含 [1., 2.] 的 np.ndarray，并视图转换为 MyArray 类型
        a = np.array([1., 2.]).view(MyArray)
        # 使用自定义断言函数 _assert_func 对 a 进行断言
        self._assert_func(a, a)
class TestApproxEqual:

    # 设置测试方法的初始化，将 assert_approx_equal 函数赋给 self._assert_func
    def setup_method(self):
        self._assert_func = assert_approx_equal

    # 测试比较简单的零维数组情况
    def test_simple_0d_arrays(self):
        x = np.array(1234.22)   # 创建包含单个元素的 NumPy 数组 x
        y = np.array(1234.23)   # 创建包含单个元素的 NumPy 数组 y

        # 使用 self._assert_func 断言 x 和 y 相近，有效位数为 5 和 6
        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        
        # 使用 lambda 函数和 assert_raises 检查 self._assert_func 报错情况
        assert_raises(AssertionError,
                      lambda: self._assert_func(x, y, significant=7))

    # 测试比较简单的标量情况
    def test_simple_items(self):
        x = 1234.22   # 创建标量 x
        y = 1234.23   # 创建标量 y

        # 使用 self._assert_func 断言 x 和 y 相近，有效位数为 4、5 和 6
        self._assert_func(x, y, significant=4)
        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        
        # 使用 lambda 函数和 assert_raises 检查 self._assert_func 报错情况
        assert_raises(AssertionError,
                      lambda: self._assert_func(x, y, significant=7))

    # 测试包含 NaN 的数组情况
    def test_nan_array(self):
        anan = np.array(np.nan)   # 创建包含 NaN 的 NumPy 数组 anan
        aone = np.array(1)        # 创建包含单个元素的 NumPy 数组 aone
        ainf = np.array(np.inf)   # 创建包含 Inf 的 NumPy 数组 ainf
        
        # 使用 self._assert_func 断言 anan 和 anan 相等
        self._assert_func(anan, anan)
        
        # 使用 lambda 函数和 assert_raises 检查 self._assert_func 报错情况
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))

    # 测试包含 NaN 的标量情况
    def test_nan_items(self):
        anan = np.array(np.nan)   # 创建包含 NaN 的 NumPy 数组 anan
        aone = np.array(1)        # 创建包含单个元素的 NumPy 数组 aone
        ainf = np.array(np.inf)   # 创建包含 Inf 的 NumPy 数组 ainf
        
        # 使用 self._assert_func 断言 anan 和 anan 相等
        self._assert_func(anan, anan)
        
        # 使用 lambda 函数和 assert_raises 检查 self._assert_func 报错情况
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))


class TestArrayAssertLess:

    # 设置测试方法的初始化，将 assert_array_less 函数赋给 self._assert_func
    def setup_method(self):
        self._assert_func = assert_array_less

    # 测试比较简单的数组情况
    def test_simple_arrays(self):
        x = np.array([1.1, 2.2])   # 创建包含多个元素的 NumPy 数组 x
        y = np.array([1.2, 2.3])   # 创建包含多个元素的 NumPy 数组 y

        # 使用 self._assert_func 断言 x 小于 y
        self._assert_func(x, y)
        
        # 使用 lambda 函数和 assert_raises 检查 self._assert_func 报错情况
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        y = np.array([1.0, 2.3])   # 创建包含多个元素的 NumPy 数组 y

        # 使用 lambda 函数和 assert_raises 检查 self._assert_func 报错情况
        assert_raises(AssertionError, lambda: self._assert_func(x, y))
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        a = np.array([1, 3, 6, 20])   # 创建包含多个元素的 NumPy 数组 a
        b = np.array([2, 4, 6, 8])    # 创建包含多个元素的 NumPy 数组 b

        # 设置预期的错误信息
        expected_msg = ('Mismatched elements: 2 / 4 (50%)\n'
                        'Max absolute difference among violations: 12\n'
                        'Max relative difference among violations: 1.5')
        
        # 使用 pytest.raises 和 match=re.escape(expected_msg) 断言 self._assert_func 抛出错误
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(a, b)
    # 定义一个测试函数，用于测试二维数组的相等性断言
    def test_rank2(self):
        # 创建两个二维 NumPy 数组作为测试数据
        x = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([[1.2, 2.3], [3.4, 4.5]])

        # 调用自定义的断言函数，验证两个数组是否相等
        self._assert_func(x, y)

        # 设置预期的错误消息，用于验证断言失败时的异常内容
        expected_msg = ('Mismatched elements: 4 / 4 (100%)\n'
                        'Max absolute difference among violations: 0.1\n'
                        'Max relative difference among violations: 0.09090909')
        
        # 使用 pytest 来验证调换参数顺序时的断言失败
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(y, x)

        # 修改 y 的值，使得与 x 不相等
        y = np.array([[1.0, 2.3], [3.4, 4.5]])

        # 使用 lambda 函数和 assert_raises 来验证断言失败
        assert_raises(AssertionError, lambda: self._assert_func(x, y))
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

    # 定义另一个测试函数，用于测试三维数组的相等性断言
    def test_rank3(self):
        # 创建两个全为 1 的三维 NumPy 数组作为测试数据
        x = np.ones(shape=(2, 2, 2))
        y = np.ones(shape=(2, 2, 2)) + 1

        # 调用自定义的断言函数，验证两个数组是否相等
        self._assert_func(x, y)

        # 使用 lambda 函数和 assert_raises 来验证断言失败
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        # 修改 y 的一个元素，使得与 x 不相等
        y[0, 0, 0] = 0

        # 设置预期的错误消息，用于验证断言失败时的异常内容
        expected_msg = ('Mismatched elements: 1 / 8 (12.5%)\n'
                        'Max absolute difference among violations: 1.\n'
                        'Max relative difference among violations: inf')
        
        # 使用 pytest 来验证特定条件下的断言失败
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)

        # 使用 lambda 函数和 assert_raises 来验证断言失败
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

    # 定义第三个测试函数，用于测试简单数值和数组的相等性断言
    def test_simple_items(self):
        # 创建两个简单的数值作为测试数据
        x = 1.1
        y = 2.2

        # 调用自定义的断言函数，验证两个数值是否相等
        self._assert_func(x, y)

        # 设置预期的错误消息，用于验证断言失败时的异常内容
        expected_msg = ('Mismatched elements: 1 / 1 (100%)\n'
                        'Max absolute difference among violations: 1.1\n'
                        'Max relative difference among violations: 1.')
        
        # 使用 pytest 来验证调换参数顺序时的断言失败
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(y, x)

        # 修改 y 的值，使得与 x 不相等
        y = np.array([2.2, 3.3])

        # 调用自定义的断言函数，验证数值与数组的相等性
        self._assert_func(x, y)

        # 使用 lambda 函数和 assert_raises 来验证断言失败
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        # 修改 y 的值，使得与 x 不相等
        y = np.array([1.0, 3.3])

        # 使用 lambda 函数和 assert_raises 来验证断言失败
        assert_raises(AssertionError, lambda: self._assert_func(x, y))
    # 定义一个测试函数，用于测试简单的数值和数组
    def test_simple_items_and_array(self):
        # 创建一个包含浮点数的 NumPy 数组
        x = np.array([[621.345454, 390.5436, 43.54657, 626.4535],
                      [54.54, 627.3399, 13., 405.5435],
                      [543.545, 8.34, 91.543, 333.3]])
        # 设置一个浮点数作为预期值 y，并调用 _assert_func 进行断言
        y = 627.34
        self._assert_func(x, y)

        # 更改预期值 y 为另一个浮点数，并再次调用 _assert_func 进行断言
        y = 8.339999
        self._assert_func(y, x)

        # 修改数组 x 为另一个包含浮点数的 NumPy 数组
        x = np.array([[3.4536, 2390.5436, 435.54657, 324525.4535],
                      [5449.54, 999090.54, 130303.54, 405.5435],
                      [543.545, 8.34, 91.543, 999090.53999]])
        # 设置预期值 y 为一个浮点数，并预计会引发 AssertionError
        y = 999090.54

        # 定义预期的错误消息，包含在 pytest 的断言中进行验证
        expected_msg = ('Mismatched elements: 1 / 12 (8.33%)\n'
                        'Max absolute difference among violations: 0.\n'
                        'Max relative difference among violations: 0.')
        # 使用 pytest.raises 检查是否抛出预期的 AssertionError，验证 _assert_func 的行为
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)

        # 定义另一个预期的错误消息，包含在 pytest 的断言中进行验证
        expected_msg = ('Mismatched elements: 12 / 12 (100%)\n'
                        'Max absolute difference among violations: '
                        '999087.0864\n'
                        'Max relative difference among violations: '
                        '289288.5934676')
        # 再次使用 pytest.raises 检查是否抛出预期的 AssertionError，验证 _assert_func 的行为
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(y, x)

    # 定义另一个测试函数，用于测试包含零值的情况
    def test_zeroes(self):
        # 创建一个包含浮点数的 NumPy 数组 x 和一个浮点数 y
        x = np.array([546456., 0, 15.455])
        y = np.array(87654.)

        # 定义预期的错误消息，包含在 pytest 的断言中进行验证
        expected_msg = ('Mismatched elements: 1 / 3 (33.3%)\n'
                        'Max absolute difference among violations: 458802.\n'
                        'Max relative difference among violations: 5.23423917')
        # 使用 pytest.raises 检查是否抛出预期的 AssertionError，验证 _assert_func 的行为
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)

        # 定义另一个预期的错误消息，包含在 pytest 的断言中进行验证
        expected_msg = ('Mismatched elements: 2 / 3 (66.7%)\n'
                        'Max absolute difference among violations: 87654.\n'
                        'Max relative difference among violations: '
                        '5670.5626011')
        # 再次使用 pytest.raises 检查是否抛出预期的 AssertionError，验证 _assert_func 的行为
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(y, x)

        # 修改预期值 y 为零
        y = 0

        # 定义预期的错误消息，包含在 pytest 的断言中进行验证
        expected_msg = ('Mismatched elements: 3 / 3 (100%)\n'
                        'Max absolute difference among violations: 546456.\n'
                        'Max relative difference among violations: inf')
        # 使用 pytest.raises 检查是否抛出预期的 AssertionError，验证 _assert_func 的行为
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(x, y)

        # 定义另一个预期的错误消息，包含在 pytest 的断言中进行验证
        expected_msg = ('Mismatched elements: 1 / 3 (33.3%)\n'
                        'Max absolute difference among violations: 0.\n'
                        'Max relative difference among violations: inf')
        # 再次使用 pytest.raises 检查是否抛出预期的 AssertionError，验证 _assert_func 的行为
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            self._assert_func(y, x)
    # 测试处理 NaN 值的情况，使用 NumPy 数组进行比较
    def test_nan_noncompare(self):
        # 创建包含 NaN 的 NumPy 数组
        anan = np.array(np.nan)
        # 创建包含整数 1 的 NumPy 数组
        aone = np.array(1)
        # 创建包含正无穷大的 NumPy 数组
        ainf = np.array(np.inf)
        
        # 调用自定义断言函数，验证相等性
        self._assert_func(anan, anan)
        # 预期抛出断言错误，因为整数数组和 NaN 数组不相等
        assert_raises(AssertionError, lambda: self._assert_func(aone, anan))
        # 预期抛出断言错误，因为 NaN 数组和整数数组不相等
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        # 预期抛出断言错误，因为 NaN 数组和正无穷大数组不相等
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        # 预期抛出断言错误，因为正无穷大数组和 NaN 数组不相等
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))
    
    # 测试处理包含 NaN 值的 NumPy 数组的情况
    def test_nan_noncompare_array(self):
        # 创建包含浮点数的 NumPy 数组
        x = np.array([1.1, 2.2, 3.3])
        # 创建包含 NaN 的 NumPy 数组
        anan = np.array(np.nan)
        
        # 预期抛出断言错误，因为数组 x 和 NaN 数组不相等
        assert_raises(AssertionError, lambda: self._assert_func(x, anan))
        # 预期抛出断言错误，因为 NaN 数组和数组 x 不相等
        assert_raises(AssertionError, lambda: self._assert_func(anan, x))
        
        # 修改数组 x，包含 NaN 值
        x = np.array([1.1, 2.2, np.nan])
        
        # 预期抛出断言错误，因为数组 x 和 NaN 数组不相等
        assert_raises(AssertionError, lambda: self._assert_func(x, anan))
        # 预期抛出断言错误，因为 NaN 数组和数组 x 不相等
        assert_raises(AssertionError, lambda: self._assert_func(anan, x))
        
        # 创建另一个包含 NaN 值的 NumPy 数组
        y = np.array([1.0, 2.0, np.nan])
        
        # 调用自定义断言函数，验证两个包含 NaN 值的数组相等
        self._assert_func(y, x)
        # 预期抛出断言错误，因为数组 x 和 y 不相等
        assert_raises(AssertionError, lambda: self._assert_func(x, y))
    
    # 测试处理无穷大值的情况，使用 NumPy 数组进行比较
    def test_inf_compare(self):
        # 创建包含整数 1 的 NumPy 数组
        aone = np.array(1)
        # 创建包含正无穷大的 NumPy 数组
        ainf = np.array(np.inf)
        
        # 调用自定义断言函数，验证 aone 和 ainf 相等
        self._assert_func(aone, ainf)
        # 调用自定义断言函数，验证 -ainf 和 aone 相等
        self._assert_func(-ainf, aone)
        # 调用自定义断言函数，验证 -ainf 和 ainf 相等
        self._assert_func(-ainf, ainf)
        # 预期抛出断言错误，因为 ainf 和 aone 不相等
        assert_raises(AssertionError, lambda: self._assert_func(ainf, aone))
        # 预期抛出断言错误，因为 aone 和 -ainf 不相等
        assert_raises(AssertionError, lambda: self._assert_func(aone, -ainf))
        # 预期抛出断言错误，因为 ainf 和 ainf 不相等
        assert_raises(AssertionError, lambda: self._assert_func(ainf, ainf))
        # 预期抛出断言错误，因为 ainf 和 -ainf 不相等
        assert_raises(AssertionError, lambda: self._assert_func(ainf, -ainf))
        # 预期抛出断言错误，因为 -ainf 和 -ainf 不相等
        assert_raises(AssertionError, lambda: self._assert_func(-ainf, -ainf))
    
    # 测试处理包含无穷大值的 NumPy 数组的情况
    def test_inf_compare_array(self):
        # 创建包含浮点数的 NumPy 数组
        x = np.array([1.1, 2.2, np.inf])
        # 创建包含正无穷大的 NumPy 数组
        ainf = np.array(np.inf)
        
        # 预期抛出断言错误，因为数组 x 和 ainf 不相等
        assert_raises(AssertionError, lambda: self._assert_func(x, ainf))
        # 预期抛出断言错误，因为 ainf 和数组 x 不相等
        assert_raises(AssertionError, lambda: self._assert_func(ainf, x))
        # 预期抛出断言错误，因为数组 x 和 -ainf 不相等
        assert_raises(AssertionError, lambda: self._assert_func(x, -ainf))
        # 预期抛出断言错误，因为 -x 和 -ainf 不相等
        assert_raises(AssertionError, lambda: self._assert_func(-x, -ainf))
        # 预期抛出断言错误，因为 -ainf 和 -x 不相等
        assert_raises(AssertionError, lambda: self._assert_func(-ainf, -x))
        # 调用自定义断言函数，验证 -ainf 和数组 x 相等
        self._assert_func(-ainf, x)
    
    # 测试严格模式（strict）的行为
    def test_strict(self):
        """Test the behavior of the `strict` option."""
        # 创建包含三个元素的零数组
        x = np.zeros(3)
        # 创建包含单个元素 1.0 的数组
        y = np.ones(())
        
        # 调用自定义断言函数，验证 x 和 y 相等
        self._assert_func(x, y)
        # 预期抛出断言错误，因为严格模式下，x 和 y 不相等
        with pytest.raises(AssertionError):
            self._assert_func(x, y, strict=True)
        
        # 将数组 y 广播至与 x 相同的形状
        y = np.broadcast_to(y, x.shape)
        
        # 调用自定义断言函数，验证 x 和 y 相等
        self._assert_func(x, y)
        # 预期抛出断言错误，因为严格模式下，x 和 y 的类型不同
        with pytest.raises(AssertionError):
            self._assert_func(x, y.astype(np.float32), strict=True)
    # 定义一个名为 TestWarns 的测试类
class TestWarns:

    # 定义一个测试方法 test_warn，用于测试警告功能
    def test_warn(self):
        # 定义内部函数 f，其中发出一个警告 "yo"，然后返回 3
        def f():
            warnings.warn("yo")
            return 3
        
        # 备份当前警告过滤器
        before_filters = sys.modules['warnings'].filters[:]
        # 使用 assert_warns 检查 f 函数是否会发出 UserWarning，并返回其返回值
        assert_equal(assert_warns(UserWarning, f), 3)
        # 获取执行 assert_warns 后的警告过滤器状态
        after_filters = sys.modules['warnings'].filters

        # 使用 assert_raises 检查 assert_no_warnings 是否会抛出 AssertionError
        assert_raises(AssertionError, assert_no_warnings, f)
        # 使用 assert_no_warnings 检查 lambda 函数是否不会发出警告
        assert_equal(assert_no_warnings(lambda x: x, 1), 1)

        # 检查警告状态是否未发生变化
        assert_equal(before_filters, after_filters,
                     "assert_warns does not preserve warnings state")

    # 定义另一个测试方法 test_context_manager，测试上下文管理器的警告功能
    def test_context_manager(self):
        # 备份当前警告过滤器
        before_filters = sys.modules['warnings'].filters[:]
        # 使用 assert_warns 上下文管理器检查代码块是否会发出 UserWarning
        with assert_warns(UserWarning):
            warnings.warn("yo")
        # 获取执行 assert_warns 后的警告过滤器状态
        after_filters = sys.modules['warnings'].filters

        # 定义一个内部函数 no_warnings，使用 assert_no_warnings 上下文管理器检查代码块不会发出警告
        def no_warnings():
            with assert_no_warnings():
                warnings.warn("yo")

        # 使用 assert_raises 检查 no_warnings 是否会抛出 AssertionError
        assert_raises(AssertionError, no_warnings)
        # 检查警告状态是否未发生变化
        assert_equal(before_filters, after_filters,
                     "assert_warns does not preserve warnings state")

    # 定义另一个测试方法 test_args，测试带参数的函数警告功能
    def test_args(self):
        # 定义函数 f，带有默认参数 a=0, b=1，发出警告 "yo"，并返回 a + b 的值
        def f(a=0, b=1):
            warnings.warn("yo")
            return a + b

        # 使用 assert_warns 检查 f 函数是否会发出 UserWarning，并返回其正确的返回值 20
        assert assert_warns(UserWarning, f, b=20) == 20

        # 使用 pytest.raises 检查使用错误参数调用 assert_warns 是否会抛出 RuntimeError
        with pytest.raises(RuntimeError) as exc:
            # assert_warns 无法进行正则表达式匹配，应使用 pytest.warns
            with assert_warns(UserWarning, match="A"):
                warnings.warn("B", UserWarning)
        assert "assert_warns" in str(exc)
        assert "pytest.warns" in str(exc)

        # 使用 pytest.raises 检查使用错误参数调用 assert_warns 是否会抛出 RuntimeError
        with pytest.raises(RuntimeError) as exc:
            # assert_warns 无法进行正则表达式匹配，应使用 pytest.warns
            with assert_warns(UserWarning, wrong="A"):
                warnings.warn("B", UserWarning)
        assert "assert_warns" in str(exc)
        assert "pytest.warns" not in str(exc)

    # 定义测试方法 test_warn_wrong_warning，测试错误的警告类型
    def test_warn_wrong_warning(self):
        # 定义函数 f，发出一个带有 DeprecationWarning 的警告 "yo"
        def f():
            warnings.warn("yo", DeprecationWarning)

        # 设置一个标志变量 failed
        failed = False
        # 使用 warnings.catch_warnings 上下文管理器捕获警告
        with warnings.catch_warnings():
            # 将 DeprecationWarning 设置为错误，以便触发异常
            warnings.simplefilter("error", DeprecationWarning)
            try:
                # 使用 assert_warns 检查 f 函数是否会发出 DeprecationWarning
                assert_warns(UserWarning, f)
                # 如果未触发预期的 DeprecationWarning，将 failed 设为 True
                failed = True
            except DeprecationWarning:
                pass

        # 如果 failed 为 True，抛出 AssertionError，表示捕获了错误的警告类型
        if failed:
            raise AssertionError("wrong warning caught by assert_warn")
    def test_simple(self):
        # 设置变量 x 和 y 分别为 0.001 和 1e-9
        x = 1e-3
        y = 1e-9

        # 断言 x 和 y 的绝对误差不超过 1，应该通过测试
        assert_allclose(x, y, atol=1)
        # 断言 x 和 y 的绝对误差超过 1，预期抛出 AssertionError
        assert_raises(AssertionError, assert_allclose, x, y)

        # 预期的错误消息，包含了详细的误差信息
        expected_msg = ('Mismatched elements: 1 / 1 (100%)\n'
                        'Max absolute difference among violations: 0.001\n'
                        'Max relative difference among violations: 999999.')
        # 断言 x 和 y 的误差匹配预期错误消息，预期抛出 AssertionError
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            assert_allclose(x, y)

        # 设置 z 为 0
        z = 0
        # 预期的错误消息，包含了详细的误差信息
        expected_msg = ('Mismatched elements: 1 / 1 (100%)\n'
                        'Max absolute difference among violations: 1.e-09\n'
                        'Max relative difference among violations: inf')
        # 断言 y 和 z 的误差匹配预期错误消息，预期抛出 AssertionError
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            assert_allclose(y, z)

        # 预期的错误消息，包含了详细的误差信息
        expected_msg = ('Mismatched elements: 1 / 1 (100%)\n'
                        'Max absolute difference among violations: 1.e-09\n'
                        'Max relative difference among violations: 1.')
        # 断言 z 和 y 的误差匹配预期错误消息，预期抛出 AssertionError
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            assert_allclose(z, y)

        # 创建数组 a 和 b
        a = np.array([x, y, x, y])
        b = np.array([x, y, x, x])

        # 断言 a 和 b 的绝对误差不超过 1，应该通过测试
        assert_allclose(a, b, atol=1)
        # 断言 a 和 b 的绝对误差超过 1，预期抛出 AssertionError
        assert_raises(AssertionError, assert_allclose, a, b)

        # 修改 b 的最后一个元素，使其与 a 的最后一个元素略有偏差
        b[-1] = y * (1 + 1e-8)
        # 断言 a 和修改后的 b 的误差在默认容差内，应该通过测试
        assert_allclose(a, b)
        # 断言 a 和修改后的 b 的误差超过指定相对容差，预期抛出 AssertionError
        assert_raises(AssertionError, assert_allclose, a, b, rtol=1e-9)

        # 断言 6 和 10 的相对误差不超过 50%，应该通过测试
        assert_allclose(6, 10, rtol=0.5)
        # 断言 10 和 6 的相对误差超过 50%，预期抛出 AssertionError
        assert_raises(AssertionError, assert_allclose, 10, 6, rtol=0.5)

        # 重新设置 b 和 c 数组
        b = np.array([x, y, x, x])
        c = np.array([x, y, x, z])

        # 预期的错误消息，包含了详细的误差信息
        expected_msg = ('Mismatched elements: 1 / 4 (25%)\n'
                        'Max absolute difference among violations: 0.001\n'
                        'Max relative difference among violations: inf')
        # 断言 b 和 c 的误差匹配预期错误消息，预期抛出 AssertionError
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            assert_allclose(b, c)

        # 预期的错误消息，包含了详细的误差信息
        expected_msg = ('Mismatched elements: 1 / 4 (25%)\n'
                        'Max absolute difference among violations: 0.001\n'
                        'Max relative difference among violations: 1.')
        # 断言 c 和 b 的误差匹配预期错误消息，预期抛出 AssertionError
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            assert_allclose(c, b)
    def test_equal_nan(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        # 应该不会引发异常：
        assert_allclose(a, b, equal_nan=True)

    def test_not_equal_nan(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        # 应该引发 AssertionError：
        assert_raises(AssertionError, assert_allclose, a, b, equal_nan=False)

    def test_equal_nan_default(self):
        # 确保 equal_nan 的默认行为保持不变。 (这些函数都在底层使用 assert_array_compare。)
        # 以下都不应该引发异常。
        a = np.array([np.nan])
        b = np.array([np.nan])
        assert_array_equal(a, b)
        assert_array_almost_equal(a, b)
        assert_array_less(a, b)
        assert_allclose(a, b)

    def test_report_max_relative_error(self):
        a = np.array([0, 1])
        b = np.array([0, 2])

        expected_msg = 'Max relative difference among violations: 0.5'
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            assert_allclose(a, b)

    def test_timedelta(self):
        # 见 gh-18286
        a = np.array([[1, 2, 3, "NaT"]], dtype="m8[ns]")
        assert_allclose(a, a)

    def test_error_message_unsigned(self):
        """检查溢出可能发生时消息的格式是否正确 (gh21768)"""
        # 确保在以下情况下测试潜在的溢出：
        #        x - y
        # 和
        #        y - x
        x = np.asarray([0, 1, 8], dtype='uint8')
        y = np.asarray([4, 4, 4], dtype='uint8')
        expected_msg = 'Max absolute difference among violations: 4'
        with pytest.raises(AssertionError, match=re.escape(expected_msg)):
            assert_allclose(x, y, atol=3)

    def test_strict(self):
        """测试 `strict` 选项的行为。"""
        x = np.ones(3)
        y = np.ones(())
        assert_allclose(x, y)
        with pytest.raises(AssertionError):
            assert_allclose(x, y, strict=True)
        assert_allclose(x, x)
        with pytest.raises(AssertionError):
            assert_allclose(x, x.astype(np.float32), strict=True)
class TestArrayAlmostEqualNulp:

    def test_float64_pass(self):
        # 定义单位最小精度的数量
        nulp = 5
        # 生成一个包含50个元素的浮点数数组，范围从-20到20
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        # 对数组中的每个元素取10的幂
        x = 10**x
        # 在数组的前后添加相反数形成一个新的数组
        x = np.r_[-x, x]

        # 计算浮点数类型的机器精度
        eps = np.finfo(x.dtype).eps
        # 构造与数组x相加后差异在几个单位最小精度内的数组y
        y = x + x*eps*nulp/2.
        # 断言数组x与数组y在指定单位最小精度内几乎相等
        assert_array_almost_equal_nulp(x, y, nulp)

        # 构造与数组x相减后差异在几个单位最小精度内的数组y
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        # 断言数组x与数组y在指定单位最小精度内几乎相等
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float64_fail(self):
        # 定义单位最小精度的数量
        nulp = 5
        # 生成一个包含50个元素的浮点数数组，范围从-20到20
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        # 对数组中的每个元素取10的幂
        x = 10**x
        # 在数组的前后添加相反数形成一个新的数组
        x = np.r_[-x, x]

        # 计算浮点数类型的机器精度
        eps = np.finfo(x.dtype).eps
        # 构造与数组x相加后超出几个单位最小精度内的数组y
        y = x + x*eps*nulp*2.
        # 断言抛出AssertionError异常，因为数组x与数组y不在指定的单位最小精度内几乎相等
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

        # 构造与数组x相减后超出几个单位最小精度内的数组y
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        # 断言抛出AssertionError异常，因为数组x与数组y不在指定的单位最小精度内几乎相等
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

    def test_float64_ignore_nan(self):
        # 忽略各种NaN之间的单位最小精度差异
        # 注意在MIPS中，可能会颠倒安静NaN和信号NaN
        # 所以我们使用内建版本作为基础
        offset = np.uint64(0xffffffff)
        # 将NaN转换为64位浮点数，并获取其对应的64位整数表示
        nan1_i64 = np.array(np.nan, dtype=np.float64).view(np.uint64)
        # 对第一个NaN的整数表示进行位异或操作，得到另一个NaN的整数表示
        nan2_i64 = nan1_i64 ^ offset  # 在MIPS上，NaN的有效载荷是全1
        # 将64位整数表示转换回64位浮点数
        nan1_f64 = nan1_i64.view(np.float64)
        nan2_f64 = nan2_i64.view(np.float64)
        # 断言两个NaN在最大单位最小精度内的差异为0
        assert_array_max_ulp(nan1_f64, nan2_f64, 0)

    def test_float32_pass(self):
        # 定义单位最小精度的数量
        nulp = 5
        # 生成一个包含50个元素的浮点数数组，范围从-20到20
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        # 对数组中的每个元素取10的幂
        x = 10**x
        # 在数组的前后添加相反数形成一个新的数组
        x = np.r_[-x, x]

        # 计算浮点数类型的机器精度
        eps = np.finfo(x.dtype).eps
        # 构造与数组x相加后差异在几个单位最小精度内的数组y
        y = x + x*eps*nulp/2.
        # 断言数组x与数组y在指定单位最小精度内几乎相等
        assert_array_almost_equal_nulp(x, y, nulp)

        # 构造与数组x相减后差异在几个单位最小精度内的数组y
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        # 断言数组x与数组y在指定单位最小精度内几乎相等
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float32_fail(self):
        # 定义单位最小精度的数量
        nulp = 5
        # 生成一个包含50个元素的浮点数数组，范围从-20到20
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        # 对数组中的每个元素取10的幂
        x = 10**x
        # 在数组的前后添加相反数形成一个新的数组
        x = np.r_[-x, x]

        # 计算浮点数类型的机器精度
        eps = np.finfo(x.dtype).eps
        # 构造与数组x相加后超出几个单位最小精度内的数组y
        y = x + x*eps*nulp*2.
        # 断言抛出AssertionError异常，因为数组x与数组y不在指定的单位最小精度内几乎相等
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

        # 构造与数组x相减后超出几个单位最小精度内的数组y
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        # 断言抛出AssertionError异常，因为数组x与数组y不在指定的单位最小精度内几乎相等
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)
    def test_float32_ignore_nan(self):
        # Ignore ULP differences between various NAN's
        # Note that MIPS may reverse quiet and signaling nans
        # so we use the builtin version as a base.
        
        # 定义偏移量为 0xffff，用于处理不同类型的 NaN 时的ULP差异
        offset = np.uint32(0xffff)
        
        # 创建一个 np.float32 类型的数组，视图转换为 np.uint32 类型
        nan1_i32 = np.array(np.nan, dtype=np.float32).view(np.uint32)
        
        # 对第一个 NaN 使用偏移量，MIPS架构下 NaN 的负载位为全 1
        nan2_i32 = nan1_i32 ^ offset  # nan payload on MIPS is all ones.
        
        # 将处理后的 NaN 转换回 np.float32 类型
        nan1_f32 = nan1_i32.view(np.float32)
        nan2_f32 = nan2_i32.view(np.float32)
        
        # 检查两个浮点数数组的最大 ULP（单位最小精度差异）
        assert_array_max_ulp(nan1_f32, nan2_f32, 0)

    def test_float16_pass(self):
        # 定义 NULP 的值为 5
        nulp = 5
        
        # 生成一个 np.float16 类型的数组 x，从 -4 到 4 之间等间距取 10 个数
        x = np.linspace(-4, 4, 10, dtype=np.float16)
        
        # 对数组 x 中的每个元素取 10 的 x 次方
        x = 10**x
        
        # 将数组 x 扩展为一个更大的数组，包含其相反数
        x = np.r_[-x, x]

        # 获取 x 数组的精度范围
        eps = np.finfo(x.dtype).eps
        
        # 计算浮点数数组 y，考虑 x 的 NULP 误差
        y = x + x*eps*nulp/2.
        
        # 检查两个浮点数数组 x 和 y 的 NULP 误差是否小于等于 nulp
        assert_array_almost_equal_nulp(x, y, nulp)

        # 获取 x 数组的负精度范围
        epsneg = np.finfo(x.dtype).epsneg
        
        # 计算浮点数数组 y，考虑 x 的负 NULP 误差
        y = x - x*epsneg*nulp/2.
        
        # 检查两个浮点数数组 x 和 y 的 NULP 误差是否小于等于 nulp
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float16_fail(self):
        # 定义 NULP 的值为 5
        nulp = 5
        
        # 生成一个 np.float16 类型的数组 x，从 -4 到 4 之间等间距取 10 个数
        x = np.linspace(-4, 4, 10, dtype=np.float16)
        
        # 对数组 x 中的每个元素取 10 的 x 次方
        x = 10**x
        
        # 将数组 x 扩展为一个更大的数组，包含其相反数
        x = np.r_[-x, x]

        # 获取 x 数组的精度范围
        eps = np.finfo(x.dtype).eps
        
        # 计算浮点数数组 y，考虑 x 的 NULP 误差的两倍
        y = x + x*eps*nulp*2.
        
        # 断言抛出异常，检查两个浮点数数组 x 和 y 的 NULP 误差是否小于等于 nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

        # 获取 x 数组的负精度范围
        epsneg = np.finfo(x.dtype).epsneg
        
        # 计算浮点数数组 y，考虑 x 的负 NULP 误差的两倍
        y = x - x*epsneg*nulp*2.
        
        # 断言抛出异常，检查两个浮点数数组 x 和 y 的 NULP 误差是否小于等于 nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

    def test_float16_ignore_nan(self):
        # Ignore ULP differences between various NAN's
        # Note that MIPS may reverse quiet and signaling nans
        # so we use the builtin version as a base.
        
        # 定义偏移量为 0xff，用于处理不同类型的 NaN 时的ULP差异
        offset = np.uint16(0xff)
        
        # 创建一个 np.float16 类型的数组，视图转换为 np.uint16 类型
        nan1_i16 = np.array(np.nan, dtype=np.float16).view(np.uint16)
        
        # 对第一个 NaN 使用偏移量，MIPS架构下 NaN 的负载位为全 1
        nan2_i16 = nan1_i16 ^ offset  # nan payload on MIPS is all ones.
        
        # 将处理后的 NaN 转换回 np.float16 类型
        nan1_f16 = nan1_i16.view(np.float16)
        nan2_f16 = nan2_i16.view(np.float16)
        
        # 检查两个浮点数数组的最大 ULP（单位最小精度差异）
        assert_array_max_ulp(nan1_f16, nan2_f16, 0)

    def test_complex128_pass(self):
        # 定义 NULP 的值为 5
        nulp = 5
        
        # 生成一个 np.float64 类型的数组 x，从 -20 到 20 之间等间距取 50 个数
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        
        # 对数组 x 中的每个元素取 10 的 x 次方
        x = 10**x
        
        # 将数组 x 扩展为一个更大的数组，包含其相反数
        x = np.r_[-x, x]
        
        # 将数组 x 转换为复数数组 xi
        xi = x + x*1j

        # 获取 x 数组的精度范围
        eps = np.finfo(x.dtype).eps
        
        # 计算复数数组 y，考虑 x 的 NULP 误差
        y = x + x*eps*nulp/2.
        
        # 检查两个复数数组 xi 和 x + y*1j 的 NULP 误差是否小于等于 nulp
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        
        # 检查两个复数数组 xi 和 y + x*1j 的 NULP 误差是否小于等于 nulp
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        
        # 由于实部和虚部都发生变化，测试条件至少要小于 sqrt(2) 的因子
        y = x + x*eps*nulp/4.
        
        # 检查两个复数数组 xi 和 y + y*1j 的 NULP 误差是否小于等于 nulp
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

        # 获取 x 数组的负精度范围
        epsneg = np.finfo(x.dtype).epsneg
        
        # 计算复数数组 y，考虑 x 的负 NULP 误差
        y = x - x*epsneg*nulp/2.
        
        # 检查两个复数数组 xi 和 x + y*1j 的 NULP 误差是否小于等于 nulp
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        
        # 检查两个复数数组 xi 和 y + x*1j 的 NULP 误差是否小于等于 nulp
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        
        y = x - x*epsneg*nulp/4.
        
        # 检查两个复数数组 xi 和 y + y*1j 的 NULP 误差是否小于等于 nulp
        assert_array_almost_equal_nulp(xi, y
    # 定义一个测试函数，用于测试复数类型为 complex128 的情况，预期测试失败
    def test_complex128_fail(self):
        # 设置 Nulp 值为 5
        nulp = 5
        # 生成一个包含 50 个元素的浮点数数组，范围从 -20 到 20
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        # 将数组 x 中的元素转换为 10 的指数值
        x = 10**x
        # 将数组 x 与其负值拼接形成新数组
        x = np.r_[-x, x]
        # 创建一个复数数组 xi，由 x 与 x 乘以虚数单位 1j 构成
        xi = x + x*1j

        # 获取数组 x 的数据类型的机器精度
        eps = np.finfo(x.dtype).eps
        # 计算 y，将 x 与 x 乘以 eps 和 nulp*2 后相加
        y = x + x*eps*nulp*2.
        # 断言 xi 与 x + y*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, x + y*1j, nulp)
        # 断言 xi 与 y + x*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + x*1j, nulp)
        
        # 调整 y 的计算，以确保测试条件至少要比 sqrt(2) 小，因为实部和虚部都在变化
        y = x + x*eps*nulp
        # 断言 xi 与 y + y*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + y*1j, nulp)

        # 获取数组 x 的数据类型的负机器精度
        epsneg = np.finfo(x.dtype).epsneg
        # 计算 y，将 x 与 x 乘以 epsneg 和 nulp*2 后相减
        y = x - x*epsneg*nulp*2.
        # 断言 xi 与 x + y*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, x + y*1j, nulp)
        # 断言 xi 与 y + x*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + x*1j, nulp)
        
        # 调整 y 的计算，以确保测试条件至少要比 sqrt(2) 小，因为实部和虚部都在变化
        y = x - x*epsneg*nulp
        # 断言 xi 与 y + y*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + y*1j, nulp)

    # 定义一个测试函数，用于测试复数类型为 complex64 的情况，预期测试通过
    def test_complex64_pass(self):
        # 设置 Nulp 值为 5
        nulp = 5
        # 生成一个包含 50 个元素的浮点数数组，范围从 -20 到 20，数据类型为 np.float32
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        # 将数组 x 中的元素转换为 10 的指数值
        x = 10**x
        # 将数组 x 与其负值拼接形成新数组
        x = np.r_[-x, x]
        # 创建一个复数数组 xi，由 x 与 x 乘以虚数单位 1j 构成
        xi = x + x*1j

        # 获取数组 x 的数据类型的机器精度
        eps = np.finfo(x.dtype).eps
        # 计算 y，将 x 与 x 乘以 eps 和 nulp/2 后相加
        y = x + x*eps*nulp/2.
        # 断言 xi 与 x + y*1j 几乎相等，nulp 个单位
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        # 断言 xi 与 y + x*1j 几乎相等，nulp 个单位
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        
        # 调整 y 的计算，以确保测试通过
        y = x + x*eps*nulp/4.
        # 断言 xi 与 y + y*1j 几乎相等，nulp 个单位
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

        # 获取数组 x 的数据类型的负机器精度
        epsneg = np.finfo(x.dtype).epsneg
        # 计算 y，将 x 与 x 乘以 epsneg 和 nulp/2 后相减
        y = x - x*epsneg*nulp/2.
        # 断言 xi 与 x + y*1j 几乎相等，nulp 个单位
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        # 断言 xi 与 y + x*1j 几乎相等，nulp 个单位
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        
        # 调整 y 的计算，以确保测试通过
        y = x - x*epsneg*nulp/4.
        # 断言 xi 与 y + y*1j 几乎相等，nulp 个单位
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

    # 定义一个测试函数，用于测试复数类型为 complex64 的情况，预期测试失败
    def test_complex64_fail(self):
        # 设置 Nulp 值为 5
        nulp = 5
        # 生成一个包含 50 个元素的浮点数数组，范围从 -20 到 20，数据类型为 np.float32
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        # 将数组 x 中的元素转换为 10 的指数值
        x = 10**x
        # 将数组 x 与其负值拼接形成新数组
        x = np.r_[-x, x]
        # 创建一个复数数组 xi，由 x 与 x 乘以虚数单位 1j 构成
        xi = x + x*1j

        # 获取数组 x 的数据类型的机器精度
        eps = np.finfo(x.dtype).eps
        # 计算 y，将 x 与 x 乘以 eps 和 nulp*2 后相加
        y = x + x*eps*nulp*2.
        # 断言 xi 与 x + y*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, x + y*1j, nulp)
        # 断言 xi 与 y + x*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + x*1j, nulp)
        
        # 调整 y 的计算，以确保测试条件至少要比 sqrt(2) 小，因为实部和虚部都在变化
        y = x + x*eps*nulp
        # 断言 xi 与 y + y*1j 几乎相等，nulp 个单位
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + y*1j, nulp)

        # 获取数组 x 的数据类型的负机器精度
        epsneg = np.finfo(x.dtype).epsneg
        # 计算 y，将 x 与 x 乘以 epsneg 和 nulp*2 后相减
        y = x - x*epsneg*nulp*2.
        #
class TestULP:

    def test_equal(self):
        # 生成一个包含10个随机数的数组
        x = np.random.randn(10)
        # 断言数组x与自身的最大ULP误差不超过0
        assert_array_max_ulp(x, x, maxulp=0)

    def test_single(self):
        # 生成一个包含10个单精度浮点数的数组，初始值为1，并添加小的随机浮动
        x = np.ones(10).astype(np.float32)
        x += 0.01 * np.random.randn(10).astype(np.float32)
        # 获取单精度浮点数的机器精度eps
        eps = np.finfo(np.float32).eps
        # 断言数组x与x加上eps之后的最大ULP误差不超过20
        assert_array_max_ulp(x, x+eps, maxulp=20)

    def test_double(self):
        # 生成一个包含10个双精度浮点数的数组，初始值为1，并添加小的随机浮动
        x = np.ones(10).astype(np.float64)
        x += 0.01 * np.random.randn(10).astype(np.float64)
        # 获取双精度浮点数的机器精度eps
        eps = np.finfo(np.float64).eps
        # 断言数组x与x加上eps之后的最大ULP误差不超过200
        assert_array_max_ulp(x, x+eps, maxulp=200)

    def test_inf(self):
        # 分别对单精度和双精度浮点数进行测试
        for dt in [np.float32, np.float64]:
            # 生成一个包含正无穷大的数组和一个包含浮点数dt类型最大值的数组
            inf = np.array([np.inf]).astype(dt)
            big = np.array([np.finfo(dt).max])
            # 断言inf数组与big数组的最大ULP误差不超过200
            assert_array_max_ulp(inf, big, maxulp=200)

    def test_nan(self):
        # 测试nan与各种值之间的ULP误差
        for dt in [np.float32, np.float64]:
            if dt == np.float32:
                maxulp = 1e6
            else:
                maxulp = 1e12
            # 生成包含正无穷大、nan、最大值、最小非负数、零、负零的数组
            inf = np.array([np.inf]).astype(dt)
            nan = np.array([np.nan]).astype(dt)
            big = np.array([np.finfo(dt).max])
            tiny = np.array([np.finfo(dt).tiny])
            zero = np.array([0.0]).astype(dt)
            nzero = np.array([-0.0]).astype(dt)
            # 断言nan与inf、big、tiny、zero、nzero数组的最大ULP误差不超过设定的maxulp
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, inf,
                                                       maxulp=maxulp))
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, big,
                                                       maxulp=maxulp))
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, tiny,
                                                       maxulp=maxulp))
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, zero,
                                                       maxulp=maxulp))
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, nzero,
                                                       maxulp=maxulp))


class TestStringEqual:
    def test_simple(self):
        # 断言两个字符串相等
        assert_string_equal("hello", "hello")
        assert_string_equal("hello\nmultiline", "hello\nmultiline")

        # 使用pytest断言异常类型为AssertionError，并检查异常信息是否符合预期
        with pytest.raises(AssertionError) as exc_info:
            assert_string_equal("foo\nbar", "hello\nbar")
        msg = str(exc_info.value)
        assert_equal(msg, "Differences in strings:\n- foo\n+ hello")

        # 使用assert_raises断言两个字符串不相等会触发AssertionError异常
        assert_raises(AssertionError,
                      lambda: assert_string_equal("foo", "hello"))
    # 定义一个测试方法，用于测试正则表达式的功能
    def test_regex(self):
        # 断言两个字符串相等，如果不相等则抛出 AssertionError
        assert_string_equal("a+*b", "a+*b")

        # 断言 lambda 表达式抛出指定的异常类型（AssertionError），如果不抛出则测试失败
        assert_raises(AssertionError,
                      lambda: assert_string_equal("aaa", "a+b"))
# 检查模块是否具有 __warningregistry__ 属性，获取其中的警告信息字典
try:
    mod_warns = mod.__warningregistry__
except AttributeError:
    # 如果模块缺少 __warningregistry__ 属性，则说明没有警告发生；
    # 这种情况可能发生在并行测试场景中，在串行测试场景中，初始警告（因此属性）总是首先创建的
    mod_warns = {}

# 获取当前警告的数量
num_warns = len(mod_warns)

# 如果警告字典中有 'version' 键，Python 3 会向注册表中添加一个 'version' 条目，不计入警告数量
if 'version' in mod_warns:
    num_warns -= 1

# 断言当前警告的数量与期望的数量 n_in_context 相等
assert_equal(num_warns, n_in_context)


def test_warn_len_equal_call_scenarios():
    # assert_warn_len_equal 在不同情境下被调用，取决于串行 vs 并行测试场景；
    # 本测试旨在探索两种代码路径，并检查是否有断言未被捕获

    # 并行测试场景 -- 尚未发出警告
    class mod:
        pass

    mod_inst = mod()

    assert_warn_len_equal(mod=mod_inst,
                          n_in_context=0)

    # 串行测试场景 -- 应该存在 __warningregistry__ 属性
    class mod:
        def __init__(self):
            self.__warningregistry__ = {'warning1': 1,
                                        'warning2': 2}

    mod_inst = mod()
    assert_warn_len_equal(mod=mod_inst,
                          n_in_context=2)


def _get_fresh_mod():
    # 获取当前模块，并清空警告注册表
    my_mod = sys.modules[__name__]
    try:
        my_mod.__warningregistry__.clear()
    except AttributeError:
        # 除非模块中曾经引发过警告，否则不会有 __warningregistry__
        pass
    return my_mod


def test_clear_and_catch_warnings():
    # 模块的初始状态，没有警告
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})
    
    # 在指定模块下，清除并捕获警告
    with clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_equal(my_mod.__warningregistry__, {})
    
    # 在未指定模块的情况下，不在上下文期间清除警告
    # catch_warnings 对 'ignore' 不会做出记录
    with clear_and_catch_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)

    # 手动向注册表中添加两个警告
    my_mod.__warningregistry__ = {'warning1': 1,
                                  'warning2': 2}

    # 确认指定模块保留旧警告，并且不添加新警告
    with clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter('ignore')
        warnings.warn('Another warning')
    assert_warn_len_equal(my_mod, 2)

    # 另一个警告，没有指定模块，它会清理注册表
    # 使用 clear_and_catch_warnings() 上下文管理器，用于捕获和清除警告
    with clear_and_catch_warnings():
        # 设置警告过滤器，忽略所有警告
        warnings.simplefilter('ignore')
        # 发出一个自定义警告消息 'Another warning'
        warnings.warn('Another warning')
    
    # 断言语句，验证 my_mod 的警告数量是否等于 0
    assert_warn_len_equal(my_mod, 0)
def test_suppress_warnings_module():
    # 初始状态下，模块没有警告
    my_mod = _get_fresh_mod()
    # 断言确保模块的 __warningregistry__ 属性为空字典
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})

    def warn_other_module():
        # apply_along_axis 在 Python 中实现；stacklevel=2 表示
        # 我们会在其模块内部执行，而不是在我们的模块内部。
        def warn(arr):
            # 发出警告 "Some warning 2"，stacklevel=2 表示警告来自于调用栈的上一层
            warnings.warn("Some warning 2", stacklevel=2)
            return arr
        np.apply_along_axis(warn, 0, [0])

    # 测试基于模块的警告抑制：
    assert_warn_len_equal(my_mod, 0)
    with suppress_warnings() as sup:
        sup.record(UserWarning)
        # 抑制来自其他模块的警告（可能以 .pyc 结尾），
        # 如果 apply_along_axis 被移动，这里需要修改。
        sup.filter(module=np.lib._shape_base_impl)
        warnings.warn("Some warning")
        warn_other_module()
    # 检查抑制是否正确测试了文件（本模块被过滤）
    assert_equal(len(sup.log), 1)
    assert_equal(sup.log[0].message.args[0], "Some warning")
    assert_warn_len_equal(my_mod, 0)
    sup = suppress_warnings()
    # 如果 apply_along_axis 被移动，这里需要修改：
    sup.filter(module=my_mod)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    # 并且测试重复是否有效：
    sup.filter(module=my_mod)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)

    # 没有指定模块时
    with suppress_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)


def test_suppress_warnings_type():
    # 初始状态下，模块没有警告
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})

    # 测试基于类型的警告抑制：
    with suppress_warnings() as sup:
        sup.filter(UserWarning)
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    sup = suppress_warnings()
    sup.filter(UserWarning)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    # 并且测试重复是否有效：
    sup.filter(module=my_mod)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)

    # 没有指定模块时
    with suppress_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)


def test_suppress_warnings_decorate_no_record():
    sup = suppress_warnings()
    sup.filter(UserWarning)

    @sup
    def warn(category):
        warnings.warn('Some warning', category)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn(UserWarning)  # 应该被抑制
        warn(RuntimeWarning)
        assert_equal(len(w), 1)


def test_suppress_warnings_record():
    sup = suppress_warnings()
    log1 = sup.record()
    # 使用上下文管理器 `sup` 来捕获警告信息和操作
    with sup:
        # 记录一条警告消息到 `log2`
        log2 = sup.record(message='Some other warning 2')
        # 过滤掉特定消息为 'Some warning' 的警告
        sup.filter(message='Some warning')
        # 发出一条 'Some warning' 的警告
        warnings.warn('Some warning')
        # 发出一条 'Some other warning' 的警告
        warnings.warn('Some other warning')
        # 再次发出一条 'Some other warning 2' 的警告
        warnings.warn('Some other warning 2')

        # 断言捕获的警告日志条数为 2
        assert_equal(len(sup.log), 2)
        # 断言 `log1` 的长度为 1
        assert_equal(len(log1), 1)
        # 断言 `log2` 的长度为 1
        assert_equal(len(log2), 1)
        # 断言 `log2` 中的第一条消息的参数为 'Some other warning 2'
        assert_equal(log2[0].message.args[0], 'Some other warning 2')

    # 再次使用相同的上下文 `sup` 来测试警告是否被正确捕获:
    with sup:
        # 记录一条 'Some other warning 2' 的警告消息到 `log2`
        log2 = sup.record(message='Some other warning 2')
        # 过滤掉特定消息为 'Some warning' 的警告
        sup.filter(message='Some warning')
        # 发出一条 'Some warning' 的警告
        warnings.warn('Some warning')
        # 发出一条 'Some other warning' 的警告
        warnings.warn('Some other warning')
        # 再次发出一条 'Some other warning 2' 的警告
        warnings.warn('Some other warning 2')

        # 断言捕获的警告日志条数为 2
        assert_equal(len(sup.log), 2)
        # 断言 `log1` 的长度为 1
        assert_equal(len(log1), 1)
        # 断言 `log2` 的长度为 1
        assert_equal(len(log2), 1)
        # 断言 `log2` 中的第一条消息的参数为 'Some other warning 2'
        assert_equal(log2[0].message.args[0], 'Some other warning 2')

    # 测试嵌套情况下的警告捕获:
    with suppress_warnings() as sup:
        # 记录当前上下文中的警告
        sup.record()
        # 嵌套使用另一个 `suppress_warnings` 上下文 `sup2`
        with suppress_warnings() as sup2:
            # 记录一条 'Some warning' 的警告消息到 `sup2` 的日志中
            sup2.record(message='Some warning')
            # 发出一条 'Some warning' 的警告
            warnings.warn('Some warning')
            # 发出一条 'Some other warning' 的警告
            warnings.warn('Some other warning')
            # 断言 `sup2` 日志中的条目数量为 1
            assert_equal(len(sup2.log), 1)
        # 断言 `sup` 日志中的条目数量为 1
        assert_equal(len(sup.log), 1)
# 定义一个测试函数，用于测试警告抑制和转发
def test_suppress_warnings_forwarding():
    # 定义一个内部函数，用于发出警告
    def warn_other_module():
        # warn 函数发出警告信息："Some warning"，stacklevel=2 表示警告位置在其模块内部
        def warn(arr):
            warnings.warn("Some warning", stacklevel=2)
            return arr
        # 应用 np.apply_along_axis 函数，对数组应用 warn 函数
        np.apply_along_axis(warn, 0, [0])

    # 使用 suppress_warnings 上下文管理器，捕获并记录警告
    with suppress_warnings() as sup:
        # 记录当前的警告状态
        sup.record()
        # 在 "always" 模式下抑制警告
        with suppress_warnings("always"):
            # 循环发出两次警告信息："Some warning"
            for i in range(2):
                warnings.warn("Some warning")
        # 断言捕获到的警告数量为 2
        assert_equal(len(sup.log), 2)

    with suppress_warnings() as sup:
        sup.record()
        # 在 "location" 模式下抑制警告
        with suppress_warnings("location"):
            # 循环发出两次警告信息："Some warning"
            for i in range(2):
                warnings.warn("Some warning")
                # 再次发出一次警告信息："Some warning"
                warnings.warn("Some warning")
        # 断言捕获到的警告数量为 2
        assert_equal(len(sup.log), 2)

    with suppress_warnings() as sup:
        sup.record()
        # 在 "module" 模式下抑制警告
        with suppress_warnings("module"):
            # 循环发出两次警告信息："Some warning"
            for i in range(2):
                warnings.warn("Some warning")
                # 再次发出一次警告信息："Some warning"
                warnings.warn("Some warning")
                # 调用外部函数发出警告
                warn_other_module()
        # 断言捕获到的警告数量为 2
        assert_equal(len(sup.log), 2)

    with suppress_warnings() as sup:
        sup.record()
        # 在 "once" 模式下抑制警告
        with suppress_warnings("once"):
            # 循环发出两次警告信息："Some warning" 和 "Some other warning"
            for i in range(2):
                warnings.warn("Some warning")
                warnings.warn("Some other warning")
                # 调用外部函数发出警告
                warn_other_module()
        # 断言捕获到的警告数量为 2
        assert_equal(len(sup.log), 2)


# 测试临时目录管理器的功能
def test_tempdir():
    # 在 tempdir 上下文中创建临时目录 tdir
    with tempdir() as tdir:
        # 在 tdir 中创建临时文件 'tmp'
        fpath = os.path.join(tdir, 'tmp')
        with open(fpath, 'w'):
            pass
    # 断言 tdir 不再存在
    assert_(not os.path.isdir(tdir))

    raised = False
    try:
        with tempdir() as tdir:
            # 在 tempdir 上下文中抛出 ValueError 异常
            raise ValueError()
    except ValueError:
        raised = True
    # 断言捕获到异常，并且 tdir 不再存在
    assert_(raised)
    assert_(not os.path.isdir(tdir))


# 测试临时文件路径管理器的功能
def test_temppath():
    # 在 temppath 上下文中创建临时文件路径 fpath
    with temppath() as fpath:
        # 在 fpath 中创建临时文件
        with open(fpath, 'w'):
            pass
    # 断言 fpath 不再存在
    assert_(not os.path.isfile(fpath))

    raised = False
    try:
        with temppath() as fpath:
            # 在 temppath 上下文中抛出 ValueError 异常
            raise ValueError()
    except ValueError:
        raised = True
    # 断言捕获到异常，并且 fpath 不再存在
    assert_(raised)
    assert_(not os.path.isfile(fpath))


# 自定义的警告清除和捕获上下文管理器
class my_cacw(clear_and_catch_warnings):

    # 指定要处理的模块，这里处理当前模块
    class_modules = (sys.modules[__name__],)


# 测试自定义警告清除和捕获上下文管理器的继承行为
def test_clear_and_catch_warnings_inherit():
    # 获取一个新的模块对象
    my_mod = _get_fresh_mod()
    # 在 my_cacw 上下文中使用警告清除和捕获功能
    with my_cacw():
        # 忽略所有警告
        warnings.simplefilter('ignore')
        # 发出一条警告信息："Some warning"
        warnings.warn('Some warning')
    # 断言当前模块的警告注册表为空字典
    assert_equal(my_mod.__warningregistry__, {})


# 使用 pytest 的条件标记，如果没有 refcount 功能则跳过该测试
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
class TestAssertNoGcCycles:
    """ Test assert_no_gc_cycles """

    def test_passes(self):
        # 内部函数，创建一个无循环引用的对象
        def no_cycle():
            b = []
            b.append([])
            return b

        # 使用 assert_no_gc_cycles 上下文管理器，测试无循环引用的情况
        with assert_no_gc_cycles():
            no_cycle()

        # 使用 assert_no_gc_cycles 函数，直接测试无循环引用的情况
        assert_no_gc_cycles(no_cycle)
    def test_asserts(self):
        # 定义一个函数，创建一个包含自身引用的循环列表
        def make_cycle():
            a = []
            a.append(a)
            a.append(a)
            return a

        # 测试断言：期望抛出 AssertionError
        with assert_raises(AssertionError):
            # 断言在不产生垃圾循环的情况下执行 make_cycle 函数
            with assert_no_gc_cycles():
                make_cycle()

        # 另一种测试断言：期望抛出 AssertionError
        with assert_raises(AssertionError):
            # 断言在不产生垃圾循环的情况下执行 make_cycle 函数
            assert_no_gc_cycles(make_cycle)

    @pytest.mark.slow
    def test_fails(self):
        """
        Test that in cases where the garbage cannot be collected, we raise an
        error, instead of hanging forever trying to clear it.
        """
        
        # 定义一个带有 __del__ 方法的类，创建并释放引用循环的对象
        class ReferenceCycleInDel:
            """
            An object that not only contains a reference cycle, but creates new
            cycles whenever it's garbage-collected and its __del__ runs
            """
            make_cycle = True

            def __init__(self):
                self.cycle = self

            def __del__(self):
                # 断开当前循环，以便可以释放 self 对象
                self.cycle = None

                if ReferenceCycleInDel.make_cycle:
                    # 但创建一个新的循环，使得垃圾收集器需要更多工作
                    ReferenceCycleInDel()

        try:
            # 创建一个对 ReferenceCycleInDel 对象的弱引用
            w = weakref.ref(ReferenceCycleInDel())
            try:
                with assert_raises(RuntimeError):
                    # 尝试在无基准空闲垃圾的情况下执行 assert_no_gc_cycles
                    assert_no_gc_cycles(lambda: None)
            except AssertionError:
                # 如果垃圾收集器尝试释放我们的对象，才需要进行上述测试
                if w() is not None:
                    pytest.skip("GC does not call __del__ on cyclic objects")
                    raise

        finally:
            # 确保停止创建引用循环
            ReferenceCycleInDel.make_cycle = False
# 使用 pytest.mark.parametrize 装饰器，参数化测试函数 assert_func，以便多次运行测试用例
@pytest.mark.parametrize('assert_func', [assert_array_equal,
                                         assert_array_almost_equal])
def test_xy_rename(assert_func):
    # 测试关键字 `x` 和 `y` 是否已重命名为 `actual` 和 `desired`，这些测试和 `_rename_parameter` 装饰器在 NumPy 2.2.0 发布之前可以移除
    assert_func(1, 1)
    assert_func(actual=1, desired=1)

    # 断言消息字符串，指示数组不相等的情况
    assert_message = "Arrays are not..."
    # 使用 pytest.raises 检查断言错误，确保抛出 Assertion Error，并匹配特定消息
    with pytest.raises(AssertionError, match=assert_message):
        assert_func(1, 2)
    with pytest.raises(AssertionError, match=assert_message):
        assert_func(actual=1, desired=2)

    # 警告消息字符串，指示关键字参数的使用已不推荐
    dep_message = 'Use of keyword argument...'
    # 使用 pytest.warns 检查是否抛出 DeprecationWarning，并匹配特定的警告消息
    with pytest.warns(DeprecationWarning, match=dep_message):
        assert_func(x=1, desired=1)
    with pytest.warns(DeprecationWarning, match=dep_message):
        assert_func(1, y=1)

    # 类型错误消息字符串，指示函数调用中参数重复赋值
    type_message = '...got multiple values for argument'
    # 显式使用换行以支持 Python 3.9，同时检查是否抛出 DeprecationWarning 和 TypeError，并匹配特定的警告和错误消息
    with pytest.warns(DeprecationWarning, match=dep_message), \
          pytest.raises(TypeError, match=type_message):
        assert_func(1, x=1)
        assert_func(1, 2, y=2)
```