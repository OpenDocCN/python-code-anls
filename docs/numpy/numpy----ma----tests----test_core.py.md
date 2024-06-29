# `.\numpy\numpy\ma\tests\test_core.py`

```
"""
Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
"""
__author__ = "Pierre GF Gerard-Marchant"

import sys  # 导入sys模块，用于系统相关操作
import warnings  # 导入warnings模块，用于处理警告
import copy  # 导入copy模块，用于对象复制
import operator  # 导入operator模块，用于操作符函数
import itertools  # 导入itertools模块，用于迭代工具函数
import textwrap  # 导入textwrap模块，用于文本包装和填充
import pickle  # 导入pickle模块，用于对象序列化和反序列化
from functools import reduce  # 导入reduce函数，用于累积计算

import pytest  # 导入pytest模块，用于编写和运行测试用例

import numpy as np  # 导入NumPy库，用于科学计算
import numpy.ma.core  # 导入NumPy的MaskedArray核心模块
import numpy._core.fromnumeric as fromnumeric  # 导入NumPy的fromnumeric模块
import numpy._core.umath as umath  # 导入NumPy的umath模块
from numpy.exceptions import AxisError  # 导入NumPy的AxisError异常
from numpy.testing import (  # 导入NumPy的测试工具函数
    assert_raises, assert_warns, suppress_warnings, IS_WASM
    )
from numpy.testing._private.utils import requires_memory  # 导入测试工具中的requires_memory函数
from numpy import ndarray  # 导入NumPy的ndarray类
from numpy._utils import asbytes  # 导入NumPy的asbytes函数
from numpy.ma.testutils import (  # 导入NumPy的测试工具函数
    assert_, assert_array_equal, assert_equal, assert_almost_equal,
    assert_equal_records, fail_if_equal, assert_not_equal,
    assert_mask_equal
    )
from numpy.ma.core import (  # 导入NumPy的MaskedArray核心函数
    MAError, MaskError, MaskType, MaskedArray, abs, absolute, add, all,
    allclose, allequal, alltrue, angle, anom, arange, arccos, arccosh, arctan2,
    arcsin, arctan, argsort, array, asarray, choose, concatenate,
    conjugate, cos, cosh, count, default_fill_value, diag, divide, doc_note,
    empty, empty_like, equal, exp, flatten_mask, filled, fix_invalid,
    flatten_structured_array, fromflex, getmask, getmaskarray, greater,
    greater_equal, identity, inner, isMaskedArray, less, less_equal, log,
    log10, make_mask, make_mask_descr, mask_or, masked, masked_array,
    masked_equal, masked_greater, masked_greater_equal, masked_inside,
    masked_less, masked_less_equal, masked_not_equal, masked_outside,
    masked_print_option, masked_values, masked_where, max, maximum,
    maximum_fill_value, min, minimum, minimum_fill_value, mod, multiply,
    mvoid, nomask, not_equal, ones, ones_like, outer, power, product, put,
    putmask, ravel, repeat, reshape, resize, shape, sin, sinh, sometrue, sort,
    sqrt, subtract, sum, take, tan, tanh, transpose, where, zeros, zeros_like,
    )

pi = np.pi  # 将NumPy中的pi赋值给变量pi


suppress_copy_mask_on_assignment = suppress_warnings()  # 创建警告抑制器对象
suppress_copy_mask_on_assignment.filter(
    numpy.ma.core.MaskedArrayFutureWarning,
    "setting an item on a masked array which has a shared mask will not copy")  # 设置警告过滤器，抑制特定警告消息


# For parametrized numeric testing
num_dts = [np.dtype(dt_) for dt_ in '?bhilqBHILQefdgFD']  # 创建包含多种数据类型的列表
num_ids = [dt_.char for dt_ in num_dts]  # 根据数据类型列表创建对应的字符列表


class TestMaskedArray:  # 定义测试MaskedArray的基类

    # Base test class for MaskedArrays.
"""
    def setup_method(self):
        # 定义基础数据。
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        z = np.array([-.5, 0., .5, .8])
        zm = masked_array(z, mask=[0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        xm.set_fill_value(1e+20)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)

    def test_basicattributes(self):
        # 测试一些基本数组属性。
        a = array([1, 3, 2])
        b = array([1, 3, 2], mask=[1, 0, 1])
        assert_equal(a.ndim, 1)
        assert_equal(b.ndim, 1)
        assert_equal(a.size, 3)
        assert_equal(b.size, 3)
        assert_equal(a.shape, (3,))
        assert_equal(b.shape, (3,))

    def test_basic0d(self):
        # 检查屏蔽标量的情况。
        x = masked_array(0)
        assert_equal(str(x), '0')
        x = masked_array(0, mask=True)
        assert_equal(str(x), str(masked_print_option))
        x = masked_array(0, mask=False)
        assert_equal(str(x), '0')
        x = array(0, mask=1)
        assert_(x.filled().dtype is x._data.dtype)

    def test_basic1d(self):
        # 在一维中测试基本数组的创建和属性。
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_(not isMaskedArray(x))
        assert_(isMaskedArray(xm))
        assert_((xm - ym).filled(0).any())
        fail_if_equal(xm.mask.astype(int), ym.mask.astype(int))
        s = x.shape
        assert_equal(np.shape(xm), s)
        assert_equal(xm.shape, s)
        assert_equal(xm.dtype, x.dtype)
        assert_equal(zm.dtype, z.dtype)
        assert_equal(xm.size, reduce(lambda x, y:x * y, s))
        assert_equal(count(xm), len(m1) - reduce(lambda x, y:x + y, m1))
        assert_array_equal(xm, xf)
        assert_array_equal(filled(xm, 1.e20), xf)
        assert_array_equal(x, xm)

    def test_basic2d(self):
        # 在二维中测试基本数组的创建和属性。
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        for s in [(4, 3), (6, 2)]:
            x.shape = s
            y.shape = s
            xm.shape = s
            ym.shape = s
            xf.shape = s

            assert_(not isMaskedArray(x))
            assert_(isMaskedArray(xm))
            assert_equal(shape(xm), s)
            assert_equal(xm.shape, s)
            assert_equal(xm.size, reduce(lambda x, y:x * y, s))
            assert_equal(count(xm), len(m1) - reduce(lambda x, y:x + y, m1))
            assert_equal(xm, xf)
            assert_equal(filled(xm, 1.e20), xf)
            assert_equal(x, xm)
    def test_concatenate_basic(self):
        # Tests concatenations.
        # 从 self.d 中获取变量 (x, y, a10, m1, m2, xm, ym, z, zm, xf)
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # 基本的串联操作
        assert_equal(np.concatenate((x, y)), concatenate((xm, ym)))
        assert_equal(np.concatenate((x, y)), concatenate((x, y)))
        assert_equal(np.concatenate((x, y)), concatenate((xm, y)))
        assert_equal(np.concatenate((x, y, x)), concatenate((x, ym, x)))

    def test_concatenate_alongaxis(self):
        # Tests concatenations.
        # 从 self.d 中获取变量 (x, y, a10, m1, m2, xm, ym, z, zm, xf)
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # 沿轴进行串联操作
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = s
        assert_equal(xm.mask, np.reshape(m1, s))
        assert_equal(ym.mask, np.reshape(m2, s))
        xmym = concatenate((xm, ym), 1)
        assert_equal(np.concatenate((x, y), 1), xmym)
        assert_equal(np.concatenate((xm.mask, ym.mask), 1), xmym._mask)

        x = zeros(2)
        y = array(ones(2), mask=[False, True])
        z = concatenate((x, y))
        assert_array_equal(z, [0, 0, 1, 1])
        assert_array_equal(z.mask, [False, False, False, True])
        z = concatenate((y, x))
        assert_array_equal(z, [1, 1, 0, 0])
        assert_array_equal(z.mask, [False, True, False, False])

    def test_concatenate_flexible(self):
        # Tests the concatenation on flexible arrays.
        # 创建一个包含随机数和索引的 masked_array
        data = masked_array(list(zip(np.random.rand(10),
                                     np.arange(10))),
                            dtype=[('a', float), ('b', int)])

        test = concatenate([data[:5], data[5:]])
        assert_equal_records(test, data)

    def test_creation_ndmin(self):
        # Check the use of ndmin
        # 使用 ndmin 参数创建数组 x
        x = array([1, 2, 3], mask=[1, 0, 0], ndmin=2)
        assert_equal(x.shape, (1, 3))
        assert_equal(x._data, [[1, 2, 3]])
        assert_equal(x._mask, [[1, 0, 0]])

    def test_creation_ndmin_from_maskedarray(self):
        # Make sure we're not losing the original mask w/ ndmin
        # 确保使用 ndmin 不会丢失原始的掩码信息
        x = array([1, 2, 3])
        x[-1] = masked
        xx = array(x, ndmin=2, dtype=float)
        assert_equal(x.shape, x._mask.shape)
        assert_equal(xx.shape, xx._mask.shape)
    def test_creation_maskcreation(self):
        # Tests how masks are initialized at the creation of Maskedarrays.
        # 创建一个包含24个浮点数的数组
        data = arange(24, dtype=float)
        # 将索引为3、6、15的位置设为masked（掩码值）
        data[[3, 6, 15]] = masked
        # 创建一个新的 MaskedArray 对象 dma_1
        dma_1 = MaskedArray(data)
        # 断言 dma_1 的掩码与 data 的掩码相同
        assert_equal(dma_1.mask, data.mask)
        # 通过已有的 MaskedArray 对象 dma_1 创建新的对象 dma_2
        dma_2 = MaskedArray(dma_1)
        # 断言 dma_2 的掩码与 dma_1 的掩码相同
        assert_equal(dma_2.mask, dma_1.mask)
        # 使用给定的掩码创建新的 MaskedArray 对象 dma_3
        dma_3 = MaskedArray(dma_1, mask=[1, 0, 0, 0] * 6)
        # 断言 dma_3 的掩码与 dma_1 的掩码不相等
        fail_if_equal(dma_3.mask, dma_1.mask)

        # 创建一个具有掩码的数组 x，所有元素均为 True
        x = array([1, 2, 3], mask=True)
        # 断言 x 的掩码为 [True, True, True]
        assert_equal(x._mask, [True, True, True])
        # 创建一个没有掩码的数组 x，所有元素均为 False
        x = array([1, 2, 3], mask=False)
        # 断言 x 的掩码为 [False, False, False]
        assert_equal(x._mask, [False, False, False])
        # 创建数组 y，使用 x 的掩码但不复制数据
        y = array([1, 2, 3], mask=x._mask, copy=False)
        # 断言 x 的掩码和 y 的掩码共享内存
        assert_(np.may_share_memory(x.mask, y.mask))
        # 创建数组 y，使用 x 的掩码并复制数据
        y = array([1, 2, 3], mask=x._mask, copy=True)
        # 断言 x 的掩码和 y 的掩码不共享内存
        assert_(not np.may_share_memory(x.mask, y.mask))
        # 创建一个没有掩码的数组 x
        x = array([1, 2, 3], mask=None)
        # 断言 x 的掩码为 [False, False, False]
        assert_equal(x._mask, [False, False, False])

    def test_masked_singleton_array_creation_warns(self):
        # The first works, but should not (ideally), there may be no way
        # to solve this, however, as long as `np.ma.masked` is an ndarray.
        np.array(np.ma.masked)
        # 断言试图创建一个包含 float 和 np.ma.masked 的数组会触发 UserWarning
        with pytest.warns(UserWarning):
            # 尝试创建一个浮点数数组，其中包含 `float(np.ma.masked)`
            # 未来可能会定义这种行为为无效行为！
            np.array([3., np.ma.masked])

    def test_creation_with_list_of_maskedarrays(self):
        # Tests creating a masked array from a list of masked arrays.
        # 使用包含掩码数组的列表创建 MaskedArray 对象 x
        x = array(np.arange(5), mask=[1, 0, 0, 0, 0])
        # 创建一个包含 x 和 x 的反向数组的数组 data
        data = array((x, x[::-1]))
        # 断言 data 的值与预期相同
        assert_equal(data, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        # 断言 data 的掩码与预期相同
        assert_equal(data._mask, [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

        # 将 x 的掩码设为 nomask
        x.mask = nomask
        # 重新创建数组 data
        data = array((x, x[::-1]))
        # 断言 data 的值与预期相同
        assert_equal(data, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        # 断言 data 的掩码为 nomask
        assert_(data.mask is nomask)
    def test_creation_with_list_of_maskedarrays_no_bool_cast(self):
        # 测试修复gh-18551中的回归问题
        masked_str = np.ma.masked_array(['a', 'b'], mask=[True, False])
        normal_int = np.arange(2)
        res = np.ma.asarray([masked_str, normal_int], dtype="U21")
        assert_array_equal(res.mask, [[True, False], [False, False]])

        # 上述问题仅由一长串异常链导致失败，尝试使用不能始终转换为布尔值的对象数组进行测试:
        class NotBool():
            def __bool__(self):
                raise ValueError("not a bool!")
        masked_obj = np.ma.masked_array([NotBool(), 'b'], mask=[True, False])
        # 检查NotBool实际上是否像我们期望的那样失败:
        with pytest.raises(ValueError, match="not a bool!"):
            np.asarray([masked_obj], dtype=bool)

        res = np.ma.asarray([masked_obj, normal_int])
        assert_array_equal(res.mask, [[True, False], [False, False]])

    def test_creation_from_ndarray_with_padding(self):
        x = np.array([('A', 0)], dtype={'names':['f0','f1'],
                                        'formats':['S4','i8'],
                                        'offsets':[0,8]})
        array(x)  # 以前由于x.dtype.descr中的'V'填充字段而失败

    def test_unknown_keyword_parameter(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MaskedArray([1, 2, 3], maks=[0, 1, 0])  # `mask`拼写错误。

    def test_asarray(self):
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        xm.fill_value = -9999
        xm._hardmask = True
        xmm = asarray(xm)
        assert_equal(xmm._data, xm._data)
        assert_equal(xmm._mask, xm._mask)
        assert_equal(xmm.fill_value, xm.fill_value)
        assert_equal(xmm._hardmask, xm._hardmask)

    def test_asarray_default_order(self):
        # 参见问题#6646
        m = np.eye(3).T
        assert_(not m.flags.c_contiguous)

        new_m = asarray(m)
        assert_(new_m.flags.c_contiguous)

    def test_asarray_enforce_order(self):
        # 参见问题#6646
        m = np.eye(3).T
        assert_(not m.flags.c_contiguous)

        new_m = asarray(m, order='C')
        assert_(new_m.flags.c_contiguous)

    def test_fix_invalid(self):
        # 检查fix_invalid函数
        with np.errstate(invalid='ignore'):
            data = masked_array([np.nan, 0., 1.], mask=[0, 0, 1])
            data_fixed = fix_invalid(data)
            assert_equal(data_fixed._data, [data.fill_value, 0., 1.])
            assert_equal(data_fixed._mask, [1., 0., 1.])

    def test_maskedelement(self):
        # 测试掩码元素
        x = arange(6)
        x[1] = masked
        assert_(str(masked) == '--')
        assert_(x[1] is masked)
        assert_equal(filled(x[1], 0), 0)
    def test_set_element_as_object(self):
        # Tests setting elements with object
        # 创建一个长度为1的空对象数组 `a`
        a = empty(1, dtype=object)
        # 创建一个元组 `x`
        x = (1, 2, 3, 4, 5)
        # 将元组 `x` 赋值给数组 `a` 的第一个元素
        a[0] = x
        # 断言数组 `a` 的第一个元素与 `x` 相等
        assert_equal(a[0], x)
        # 断言数组 `a` 的第一个元素与 `x` 是同一个对象
        assert_(a[0] is x)

        import datetime
        # 获取当前日期时间对象 `dt`
        dt = datetime.datetime.now()
        # 将日期时间对象 `dt` 赋值给数组 `a` 的第一个元素
        a[0] = dt
        # 断言数组 `a` 的第一个元素与 `dt` 是同一个对象
        assert_(a[0] is dt)

    def test_indexing(self):
        # Tests conversions and indexing
        # 创建一个包含整数的 NumPy 数组 `x1`
        x1 = np.array([1, 2, 4, 3])
        # 使用 `x1` 和 mask 参数创建一个掩码数组 `x2`
        x2 = array(x1, mask=[1, 0, 0, 0])
        # 使用 `x1` 创建一个普通的 NumPy 数组 `x3`
        x3 = array(x1, mask=[0, 1, 0, 1])
        # 创建一个普通的 NumPy 数组 `x4`
        x4 = array(x1)
        # 测试转换为字符串
        str(x2)  # raises?
        repr(x2)  # raises?
        # 断言对 `x1` 排序后的结果与不带结束参数的 `sort(x2)` 相等
        assert_equal(np.sort(x1), sort(x2, endwith=False))
        # 索引测试
        # 断言 `x2[1]` 的类型与 `x1[1]` 的类型相同
        assert_(type(x2[1]) is type(x1[1]))
        # 断言 `x2[1]` 的值与 `x1[1]` 的值相等
        assert_(x1[1] == x2[1])
        # 断言 `x2[0]` 是掩码值 `masked`
        assert_(x2[0] is masked)
        # 断言 `x1[2]` 的值与 `x2[2]` 的值相等
        assert_equal(x1[2], x2[2])
        # 断言切片操作的结果相等
        assert_equal(x1[2:5], x2[2:5])
        assert_equal(x1[:], x2[:])
        assert_equal(x1[1:], x3[1:])
        # 修改 `x1[2]` 的值为 9
        x1[2] = 9
        x2[2] = 9
        # 断言 `x1` 和 `x2` 的值相等
        assert_equal(x1, x2)
        # 将 `x1[1:3]` 的值修改为 99
        x1[1:3] = 99
        x2[1:3] = 99
        # 断言 `x1` 和 `x2` 的值相等
        assert_equal(x1, x2)
        # 将 `x2[1]` 的值设置为掩码值 `masked`
        x2[1] = masked
        # 断言 `x1` 和 `x2` 的值相等
        assert_equal(x1, x2)
        # 将 `x2[1:3]` 的值设置为掩码值 `masked`
        x2[1:3] = masked
        # 断言 `x1` 和 `x2` 的值相等
        assert_equal(x1, x2)
        # 将 `x2[:]` 的值设置为 `x1`
        x2[:] = x1
        # 将 `x2[1]` 的值设置为掩码值 `masked`
        x2[1] = masked
        # 断言 `x2` 的掩码与数组 [0, 1, 0, 0] 相等
        assert_(allequal(getmask(x2), array([0, 1, 0, 0])))
        # 将 `x3[:]` 的值设置为带有掩码的数组
        x3[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        # 断言 `x3` 的掩码与数组 [0, 1, 1, 0] 相等
        assert_(allequal(getmask(x3), array([0, 1, 1, 0])))
        # 将 `x4[:]` 的值设置为带有掩码的数组
        x4[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        # 断言 `x4` 与数组 [1, 2, 3, 4] 相等
        assert_(allequal(x4, array([1, 2, 3, 4])))
        # 创建一个浮点数数组 `x1`
        x1 = np.arange(5) * 1.0
        # 使用 `masked_values` 函数将 `x1` 中的值为 3.0 的元素设置为掩码
        x2 = masked_values(x1, 3.0)
        # 断言 `x1` 和 `x2` 相等
        assert_equal(x1, x2)
        # 断言 `x2` 的掩码数组与数组 [0, 0, 0, 1, 0] 相等
        assert_(allequal(array([0, 0, 0, 1, 0], MaskType), x2.mask))
        # 断言 `x2` 的填充值为 3.0
        assert_equal(3.0, x2.fill_value)
        # 创建一个包含不同类型对象的数组 `x1`
        x1 = array([1, 'hello', 2, 3], object)
        x2 = np.array([1, 'hello', 2, 3], object)
        # 获取 `x1` 和 `x2` 中索引为 1 的元素
        s1 = x1[1]
        s2 = x2[1]
        # 断言 `s1` 和 `s2` 的类型相同且值相等
        assert_equal(type(s2), str)
        assert_equal(type(s1), str)
        assert_equal(s1, s2)
        # 断言 `x1[1:1]` 的形状为 (0,)
        assert_(x1[1:1].shape == (0,))

    def test_setitem_no_warning(self):
        # Setitem shouldn't warn, because the assignment might be masked
        # and warning for a masked assignment is weird (see gh-23000)
        # (When the value is masked, otherwise a warning would be acceptable
        # but is not given currently.)
        # 创建一个 6x10 的掩码数组 `x`
        x = np.ma.arange(60).reshape((6, 10))
        # 创建索引 `index`
        index = (slice(1, 5, 2), [7, 5])
        # 创建一个掩码数组 `value`
        value = np.ma.masked_all((2, 2))
        value._data[...] = np.inf  # not a valid integer...
        # 使用索引 `index` 将 `value` 赋值给数组 `x`
        x[index] = value
        # 将数组 `x` 的所有元素设置为掩码值 `masked`
        x[...] = np.ma.masked
        # 创建一个包含 3.0 的浮点数数组 `x`
        x = np.ma.arange(3., dtype=np.float32)
        # 创建一个带有掩码的数组 `value`
        value = np.ma.array([2e234, 1, 1], mask=[True, False, False])
        # 使用数组 `value` 将 `x` 的所有元素设置为对应的值
        x[...] = value
        # 使用数组 `value` 将 `x` 的索引 [0, 1, 2] 处的元素设置为对应的值
        x[[0, 1, 2]] = value
    def test_copy(self):
        # Tests of some subtle points of copying and sizing.

        # 创建一个列表 n
        n = [0, 0, 1, 0, 0]
        # 调用 make_mask 函数，生成一个掩码数组 m
        m = make_mask(n)
        # 再次调用 make_mask 函数，生成 m 的副本 m2
        m2 = make_mask(m)
        # 断言 m 和 m2 是同一个对象
        assert_(m is m2)
        # 使用 copy=True 参数再次调用 make_mask 函数，生成 m 的另一个副本 m3
        m3 = make_mask(m, copy=True)
        # 断言 m 和 m3 不是同一个对象
        assert_(m is not m3)

        # 创建一个数组 x1，包含 [0, 1, 2, 3, 4]
        x1 = np.arange(5)
        # 创建一个掩码数组 y1，使用 x1 和 m
        y1 = array(x1, mask=m)
        # 断言 y1 的数据与 x1 的数据接口相同
        assert_equal(y1._data.__array_interface__, x1.__array_interface__)
        # 断言 y1 的数据与 x1 的数据完全相等
        assert_(allequal(x1, y1.data))
        # 断言 y1 的掩码与 m 的掩码接口相同
        assert_equal(y1._mask.__array_interface__, m.__array_interface__)

        # 创建 y1 的一个副本 y1a
        y1a = array(y1)
        # 默认情况下，掩码数组的副本不会被复制（参考 gh-10318）
        assert_(y1a._data.__array_interface__ ==
                        y1._data.__array_interface__)
        assert_(y1a._mask.__array_interface__ ==
                        y1._mask.__array_interface__)

        # 使用 m3 创建数组 y2
        y2 = array(x1, mask=m3)
        # 断言 y2 的数据与 x1 的数据接口相同
        assert_(y2._data.__array_interface__ == x1.__array_interface__)
        # 断言 y2 的掩码与 m3 的掩码接口相同
        assert_(y2._mask.__array_interface__ == m3.__array_interface__)
        # 断言 y2 的索引 2 是 masked
        assert_(y2[2] is masked)
        # 修改 y2 的索引 2 为 9
        y2[2] = 9
        # 断言 y2 的索引 2 不再是 masked
        assert_(y2[2] is not masked)
        # 再次断言 y2 的掩码与 m3 的掩码接口相同
        assert_(y2._mask.__array_interface__ == m3.__array_interface__)
        # 断言 y2 的所有掩码元素都为 0
        assert_(allequal(y2.mask, 0))

        # 使用 m 创建数组 y2a，同时将 copy 参数设置为 1
        y2a = array(x1, mask=m, copy=1)
        # 断言 y2a 的数据与 x1 的数据接口不同
        assert_(y2a._data.__array_interface__ != x1.__array_interface__)
        # 断言 y2a 的掩码与 m 的掩码接口不同
        assert_(y2a._mask.__array_interface__ != m.__array_interface__)
        # 断言 y2a 的索引 2 是 masked
        assert_(y2a[2] is masked)
        # 修改 y2a 的索引 2 为 9
        y2a[2] = 9
        # 断言 y2a 的索引 2 不再是 masked
        assert_(y2a[2] is not masked)
        # 再次断言 y2a 的掩码与 m 的掩码接口不同
        assert_(y2a._mask.__array_interface__ != m.__array_interface__)
        # 断言 y2a 的所有掩码元素都为 0
        assert_(allequal(y2a.mask, 0))

        # 使用 x1 * 1.0 创建数组 y3，同时使用 m 作为掩码
        y3 = array(x1 * 1.0, mask=m)
        # 断言 y3 的填充数据类型与 (x1 * 1.0) 的数据类型相同
        assert_(filled(y3).dtype is (x1 * 1.0).dtype)

        # 创建数组 x4，包含 [0, 1, 2, 3]，并将索引 2 的元素设置为 masked
        x4 = arange(4)
        x4[2] = masked
        # 调整 x4 的大小为 (8,)，并将结果保存在 y4 中
        y4 = resize(x4, (8,))
        # 断言 y4 与 [0, 1, 2, 3, 0, 1, 2, 3] 相等
        assert_equal(concatenate([x4, x4]), y4)
        # 断言 y4 的掩码为 [0, 0, 1, 0, 0, 0, 1, 0]
        assert_equal(getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0])
        # 将 x4 沿轴 0 重复 [2, 2, 2, 2]，结果保存在 y5 中
        y5 = repeat(x4, (2, 2, 2, 2), axis=0)
        # 断言 y5 与 [0, 0, 1, 1, 2, 2, 3, 3] 相等
        assert_equal(y5, [0, 0, 1, 1, 2, 2, 3, 3])
        # 将 x4 沿轴 0 重复 2 次，结果保存在 y6 中
        y6 = repeat(x4, 2, axis=0)
        # 断言 y5 与 y6 相等
        assert_equal(y5, y6)
        # 使用 x4.repeat((2, 2, 2, 2), axis=0) 重复 x4，结果保存在 y7 中
        y7 = x4.repeat((2, 2, 2, 2), axis=0)
        # 断言 y5 与 y7 相等
        assert_equal(y5, y7)
        # 使用 x4.repeat(2, 0) 重复 x4，结果保存在 y8 中
        y8 = x4.repeat(2, 0)
        # 断言 y5 与 y8 相等
        assert_equal(y5, y8)

        # 创建 x4 的副本 y9
        y9 = x4.copy()
        # 断言 y9 的数据与 x4 的数据相等
        assert_equal(y9._data, x4._data)
        # 断言 y9 的掩码与 x4 的掩码相等
        assert_equal(y9._mask, x4._mask)

        # 创建一个 masked_array x，数据为 [1, 2, 3]，掩码为 [0, 1, 0]
        x = masked_array([1, 2, 3], mask=[0, 1, 0])
        # 默认情况下，复制操作是 False
        y = masked_array(x)
        # 断言 y 的数据接口与 x 的数据接口相同
        assert_equal(y._data.ctypes.data, x._data.ctypes.data)
        # 断言 y 的掩码接口与 x 的掩码接口相同
        assert_equal(y._mask.ctypes.data, x._mask.ctypes.data)
        # 使用 copy=True 创建 y 的副本
        y = masked_array(x, copy=True)
        # 断言 y 的数据接口与 x 的数据接口不同
        assert_not_equal(y._data.ctypes.data, x._data.ctypes.data)
        # 断言 y 的掩码接口与 x 的掩码接口不同
        assert_not_equal(y._mask.ctypes.data, x._mask.ctypes.data)

    def test_copy_0d(self):
        # gh-9430
        # 创建一个 0 维的 masked array x，数据为 43，掩码为 True
        x = np.ma.array(43, mask=True)
        # 创建 x 的副本 xc
        xc = x.copy()
        # 断言 xc 的掩码为 True
        assert_equal(xc.mask, True)
    # 测试复制是否适用于 Python 内置对象（问题 #8019）
    def test_copy_on_python_builtins(self):
        # 断言：确保 np.ma.copy 对于列表 [1, 2, 3] 返回的是 MaskedArray
        assert_(isMaskedArray(np.ma.copy([1,2,3])))
        # 断言：确保 np.ma.copy 对于元组 (1, 2, 3) 返回的是 MaskedArray
        assert_(isMaskedArray(np.ma.copy((1,2,3))))

    # 测试 copy 方法是否是不可变的，GitHub 问题 #5247
    def test_copy_immutable(self):
        # 创建一个包含 [1, 2, 3] 的 MaskedArray 对象 a
        a = np.ma.array([1, 2, 3])
        # 创建一个包含 [4, 5, 6] 的 MaskedArray 对象 b
        b = np.ma.array([4, 5, 6])
        # 将 a 的 copy 方法保存到 a_copy_method 变量中
        a_copy_method = a.copy
        # 调用 b 的 copy 方法，但未保存结果，没有实际效果
        b.copy
        # 断言：调用 a_copy_method 方法后，返回的结果应与 [1, 2, 3] 相等
        assert_equal(a_copy_method(), [1, 2, 3])

    # 测试深拷贝
    def test_deepcopy(self):
        # 导入深拷贝函数 deepcopy
        from copy import deepcopy
        # 创建一个数组 a，包含 [0, 1, 2]，并指定其中的 [False, True, False] 为掩码
        a = array([0, 1, 2], mask=[False, True, False])
        # 对 a 进行深拷贝，得到 copied
        copied = deepcopy(a)
        # 断言：copied 的掩码应该与 a 的掩码相等
        assert_equal(copied.mask, a.mask)
        # 断言：确保 a 的掩码对象与 copied 的掩码对象不同
        assert_not_equal(id(a._mask), id(copied._mask))

        # 修改 copied 中的第二个元素为 1
        copied[1] = 1
        # 断言：修改后，copied 的掩码应该变为 [0, 0, 0]
        assert_equal(copied.mask, [0, 0, 0])
        # 断言：原始数组 a 的掩码不应受到影响，仍为 [0, 1, 0]
        assert_equal(a.mask, [0, 1, 0])

        # 再次进行深拷贝操作
        copied = deepcopy(a)
        # 断言：再次确保 copied 的掩码与 a 的掩码相等
        assert_equal(copied.mask, a.mask)
        # 修改 copied 的掩码的第二个元素为 False
        copied.mask[1] = False
        # 断言：修改后，copied 的掩码应该变为 [0, 0, 0]
        assert_equal(copied.mask, [0, 0, 0])
        # 断言：原始数组 a 的掩码不应受到影响，仍为 [0, 1, 0]

    # 测试格式化功能
    def test_format(self):
        # 创建一个数组 a，包含 [0, 1, 2]，并指定其中的 [False, True, False] 为掩码
        a = array([0, 1, 2], mask=[False, True, False])
        # 断言：格式化数组 a 应该得到 "[0 -- 2]"
        assert_equal(format(a), "[0 -- 2]")
        # 断言：格式化未定义的变量 masked 应该得到 "--"
        assert_equal(format(masked), "--")
        # 断言：使用空格式化字符串格式化 masked 应该得到 "--"
        assert_equal(format(masked, ""), "--")

        # 以下部分来自 PR #15410，可能在未来解决
        # assert_equal(format(masked, " >5"), "   --")
        # assert_equal(format(masked, " <5"), "--   ")

        # 断言：使用未来版本警告（FutureWarning）格式化 MaskedElement masked
        with assert_warns(FutureWarning):
            with_format_string = format(masked, " >5")
        # 断言：使用格式化字符串 " >5" 格式化 masked 应该得到 "--"
        assert_equal(with_format_string, "--")
    # 定义一个测试方法，用于验证字符串表示的准确性
    def test_str_repr(self):
        # 创建一个带有掩码的数组a，其中第二个元素被掩盖
        a = array([0, 1, 2], mask=[False, True, False])
        # 断言字符串表示的结果是否符合预期
        assert_equal(str(a), '[0 -- 2]')
        # 断言使用repr函数的结果是否符合预期
        assert_equal(
            repr(a),
            textwrap.dedent('''\
            masked_array(data=[0, --, 2],
                         mask=[False,  True, False],
                   fill_value=999999)''')
        )

        # 创建一个带有部分掩盖的数组a，展示掩盖部分和数据的字符串表示
        a = np.ma.arange(2000)
        a[1:50] = np.ma.masked
        # 断言repr函数的结果是否符合预期
        assert_equal(
            repr(a),
            textwrap.dedent('''\
            masked_array(data=[0, --, --, ..., 1997, 1998, 1999],
                         mask=[False,  True,  True, ..., False, False, False],
                   fill_value=999999)''')
        )

        # 创建一个一维数组a，展示其正确对齐的字符串表示
        a = np.ma.arange(20)
        # 断言repr函数的结果是否符合预期
        assert_equal(
            repr(a),
            textwrap.dedent('''\
            masked_array(data=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                               14, 15, 16, 17, 18, 19],
                         mask=False,
                   fill_value=999999)''')
        )

        # 创建一个二维数组a，展示其正确的字符串表示和掩盖部分
        a = array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        a[1,1] = np.ma.masked
        # 使用repr函数验证结果是否符合预期
        assert_equal(
            repr(a),
            textwrap.dedent(f'''\
            masked_array(
              data=[[1, 2, 3],
                    [4, --, 6]],
              mask=[[False, False, False],
                    [False,  True, False]],
              fill_value={np.array(999999)[()]!r},
              dtype=int8)''')
        )

        # 创建一个行向量的数组a，验证其字符串表示
        assert_equal(
            repr(a[:1]),
            textwrap.dedent(f'''\
            masked_array(data=[[1, 2, 3]],
                         mask=[[False, False, False]],
                   fill_value={np.array(999999)[()]!r},
                        dtype=int8)''')
        )

        # 创建一个整型数据类型的数组a，验证其字符串表示
        assert_equal(
            repr(a.astype(int)),
            textwrap.dedent('''\
            masked_array(
              data=[[1, 2, 3],
                    [4, --, 6]],
              mask=[[False, False, False],
                    [False,  True, False]],
              fill_value=999999)''')
        )
    # 定义一个测试方法，用于测试字符串表示的兼容性
    def test_str_repr_legacy(self):
        # 备份旧的打印选项
        oldopts = np.get_printoptions()
        # 设置打印选项以支持旧版本（'1.13'）的打印格式
        np.set_printoptions(legacy='1.13')
        try:
            # 创建一个带有掩码的数组
            a = array([0, 1, 2], mask=[False, True, False])
            # 断言数组的字符串表示符合预期
            assert_equal(str(a), '[0 -- 2]')
            # 断言数组的详细表示符合预期
            assert_equal(repr(a), 'masked_array(data = [0 -- 2],\n'
                                  '             mask = [False  True False],\n'
                                  '       fill_value = 999999)\n')

            # 创建一个带有掩码的 ma 数组
            a = np.ma.arange(2000)
            a[1:50] = np.ma.masked
            # 断言数组的详细表示符合预期
            assert_equal(
                repr(a),
                'masked_array(data = [0 -- -- ..., 1997 1998 1999],\n'
                '             mask = [False  True  True ..., False False False],\n'
                '       fill_value = 999999)\n'
            )
        finally:
            # 恢复旧的打印选项
            np.set_printoptions(**oldopts)

    # 定义一个测试方法，用于测试零维 unicode 字符串的处理
    def test_0d_unicode(self):
        # 创建一个 unicode 字符串
        u = 'caf\xe9'
        # 获取 unicode 字符串的类型
        utype = type(u)

        # 创建一个不带掩码的 ma 数组
        arr_nomask = np.ma.array(u)
        # 创建一个带有掩码的 ma 数组
        arr_masked = np.ma.array(u, mask=True)

        # 断言不带掩码的数组经过类型转换后仍然等于原始 unicode 字符串
        assert_equal(utype(arr_nomask), u)
        # 断言带有掩码的数组经过类型转换后等于特定占位符 '--'
        assert_equal(utype(arr_masked), '--')

    # 定义一个测试方法，用于测试对象的序列化与反序列化
    def test_pickling(self):
        # 测试不同数据类型的数组
        for dtype in (int, float, str, object):
            # 创建一个包含 0 到 9 的数组，并转换为指定数据类型
            a = arange(10).astype(dtype)
            # 设置填充值为 999
            a.fill_value = 999

            # 定义不同的掩码情况
            masks = ([0, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # 部分掩码
                     True,                            # 全部掩码
                     False)                           # 无掩码

            # 遍历不同的 pickle 协议版本
            for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
                for mask in masks:
                    # 设置数组的掩码
                    a.mask = mask
                    # 序列化并反序列化数组
                    a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
                    # 断言反序列化后的数组掩码与原数组一致
                    assert_equal(a_pickled._mask, a._mask)
                    # 断言反序列化后的数组数据部分与原数组一致
                    assert_equal(a_pickled._data, a._data)
                    # 根据数据类型断言填充值是否正确
                    if dtype in (object, int):
                        assert_equal(a_pickled.fill_value, 999)
                    else:
                        assert_equal(a_pickled.fill_value, dtype(999))
                    # 断言反序列化后的数组掩码与预期一致
                    assert_array_equal(a_pickled.mask, mask)

    # 定义一个测试方法，用于测试子类化 ndarray 后的序列化与反序列化
    def test_pickling_subbaseclass(self):
        # 创建一个具有字段 'x' 和 'y' 的结构化数组，并转换为记录数组
        x = np.array([(1.0, 2), (3.0, 4)],
                     dtype=[('x', float), ('y', int)]).view(np.recarray)
        # 创建一个带有掩码的 ma 数组
        a = masked_array(x, mask=[(True, False), (False, True)])
        # 遍历不同的 pickle 协议版本
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 序列化并反序列化 ma 数组
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            # 断言反序列化后的数组掩码与原数组一致
            assert_equal(a_pickled._mask, a._mask)
            # 断言反序列化后的数组与原数组一致
            assert_equal(a_pickled, a)
            # 断言反序列化后的数据类型为 np.recarray
            assert_(isinstance(a_pickled._data, np.recarray))
    # 测试序列化 MaskedConstant 对象的功能
    def test_pickling_maskedconstant(self):
        # 创建一个 MaskedConstant 对象
        mc = np.ma.masked
        # 使用不同的协议版本进行序列化和反序列化测试
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 将 mc 对象序列化为字节流，然后反序列化为新的对象 mc_pickled
            mc_pickled = pickle.loads(pickle.dumps(mc, protocol=proto))
            # 检查反序列化后的对象的基类是否与原对象一致
            assert_equal(mc_pickled._baseclass, mc._baseclass)
            # 检查反序列化后的对象的掩码是否与原对象一致
            assert_equal(mc_pickled._mask, mc._mask)
            # 检查反序列化后的对象的数据是否与原对象一致
            assert_equal(mc_pickled._data, mc._data)

    # 测试结构化数组的序列化和反序列化功能
    def test_pickling_wstructured(self):
        # 创建一个结构化数组 a
        a = array([(1, 1.), (2, 2.)], mask=[(0, 0), (0, 1)],
                  dtype=[('a', int), ('b', float)])
        # 使用不同的协议版本进行序列化和反序列化测试
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 将数组 a 序列化为字节流，然后反序列化为新的数组 a_pickled
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            # 检查反序列化后的数组的掩码是否与原数组一致
            assert_equal(a_pickled._mask, a._mask)
            # 检查反序列化后的数组是否与原数组一致
            assert_equal(a_pickled, a)

    # 测试 F_CONTIGUOUS 数组的序列化和反序列化功能
    def test_pickling_keepalignment(self):
        # 创建一个二维数组 a，并转置为数组 b
        a = arange(10)
        a.shape = (-1, 2)
        b = a.T
        # 使用不同的协议版本进行序列化和反序列化测试
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 将数组 b 序列化为字节流，然后反序列化为新的数组 test
            test = pickle.loads(pickle.dumps(b, protocol=proto))
            # 检查反序列化后的数组是否与原数组 b 一致
            assert_equal(test, b)

    # 测试 Maskedarrays 的单个元素下标功能
    def test_single_element_subscript(self):
        # 创建一个普通数组 a 和一个带掩码的数组 b
        a = array([1, 3, 2])
        b = array([1, 3, 2], mask=[1, 0, 1])
        # 检查普通数组 a 的第一个元素的形状是否为空元组
        assert_equal(a[0].shape, ())
        # 检查带掩码数组 b 的第一个元素的形状是否为空元组
        assert_equal(b[0].shape, ())
        # 检查带掩码数组 b 的第二个元素的形状是否为单个元素的元组
        assert_equal(b[1].shape, ())

    # 测试与 Python 交互的一些通信问题
    def test_topython(self):
        # 检查将数组转换为整数是否成功
        assert_equal(1, int(array(1)))
        # 检查将数组转换为浮点数是否成功
        assert_equal(1.0, float(array(1)))
        # 检查多维数组转换为整数是否成功
        assert_equal(1, int(array([[[1]]])))
        # 检查多维数组转换为浮点数是否成功
        assert_equal(1.0, float(array([[1]])))
        # 检查预期的 TypeError 是否被触发，尝试将数组转换为浮点数
        assert_raises(TypeError, float, array([1, 1]))

        # 使用 suppress_warnings 上下文管理器来忽略特定的警告
        with suppress_warnings() as sup:
            # 设置过滤器以忽略特定类型的警告
            sup.filter(UserWarning, 'Warning: converting a masked element')
            # 检查带掩码数组的某些元素是否为 NaN
            assert_(np.isnan(float(array([1], mask=[1]))))

            # 创建一个带掩码数组 a
            a = array([1, 2, 3], mask=[1, 0, 0])
            # 检查预期的 TypeError 是否被触发，尝试将带掩码数组 a 转换为浮点数
            assert_raises(TypeError, lambda: float(a))
            # 检查带掩码数组 a 的最后一个元素是否成功转换为浮点数，并且其值为 3.0
            assert_equal(float(a[-1]), 3.)
            # 检查带掩码数组 a 的第一个元素是否为 NaN
            assert_(np.isnan(float(a[0])))
        # 检查预期的 TypeError 是否被触发，尝试将带掩码数组 a 转换为整数
        assert_raises(TypeError, int, a)
        # 检查带掩码数组 a 的最后一个元素是否成功转换为整数，并且其值为 3
        assert_equal(int(a[-1]), 3)
        # 检查预期的 MAError 是否被触发，尝试将带掩码数组 a 的第一个元素转换为整数
        assert_raises(MAError, lambda:int(a[0]))
    def test_oddfeatures_1(self):
        # 测试一些奇怪的特性
        x = arange(20)  # 创建一个包含 0 到 19 的数组
        x = x.reshape(4, 5)  # 将数组重新形状为 4x5
        x.flat[5] = 12  # 将扁平化后索引为 5 的位置设置为 12
        assert_(x[1, 0] == 12)  # 断言数组位置 (1, 0) 的值为 12
        z = x + 10j * x  # 创建一个复数数组 z
        assert_equal(z.real, x)  # 断言 z 的实部与 x 相等
        assert_equal(z.imag, 10 * x)  # 断言 z 的虚部是 x 的 10 倍
        assert_equal((z * conjugate(z)).real, 101 * x * x)  # 断言 z 与其共轭乘积的实部为 101 * x^2
        z.imag[...] = 0.0  # 将 z 的虚部全部设置为 0.0

        x = arange(10)  # 创建一个包含 0 到 9 的数组
        x[3] = masked  # 将数组中索引为 3 的位置设置为 masked
        assert_(str(x[3]) == str(masked))  # 断言数组中索引为 3 的位置的字符串表示与 masked 相等
        c = x >= 8  # 创建一个布尔掩码数组，表示 x 中大于等于 8 的位置
        assert_(count(where(c, masked, masked)) == 0)  # 断言掩码数组中使用 where 函数得到 masked 的数量为 0
        assert_(shape(where(c, masked, masked)) == c.shape)  # 断言 where 函数的输出形状与掩码数组相同

        z = masked_where(c, x)  # 使用掩码数组 c 来创建一个掩码数组 z
        assert_(z.dtype is x.dtype)  # 断言 z 的数据类型与 x 相同
        assert_(z[3] is masked)  # 断言 z 中索引为 3 的位置为 masked
        assert_(z[4] is not masked)  # 断言 z 中索引为 4 的位置不是 masked
        assert_(z[7] is not masked)  # 断言 z 中索引为 7 的位置不是 masked
        assert_(z[8] is masked)  # 断言 z 中索引为 8 的位置为 masked
        assert_(z[9] is masked)  # 断言 z 中索引为 9 的位置为 masked
        assert_equal(x, z)  # 断言数组 x 与 z 相等

    def test_oddfeatures_2(self):
        # 测试更多特性
        x = array([1., 2., 3., 4., 5.])  # 创建一个包含浮点数的数组
        c = array([1, 1, 1, 0, 0])  # 创建一个布尔数组
        x[2] = masked  # 将数组中索引为 2 的位置设置为 masked
        z = where(c, x, -x)  # 根据布尔数组 c，选择 x 或者 -x 组成新的数组 z
        assert_equal(z, [1., 2., 0., -4., -5])  # 断言 z 与预期结果相等
        c[0] = masked  # 将布尔数组 c 中的第一个元素设置为 masked
        z = where(c, x, -x)  # 再次根据布尔数组 c 选择 x 或者 -x 组成新的数组 z
        assert_equal(z, [1., 2., 0., -4., -5])  # 断言 z 与预期结果相等
        assert_(z[0] is masked)  # 断言 z 中索引为 0 的位置为 masked
        assert_(z[1] is not masked)  # 断言 z 中索引为 1 的位置不是 masked
        assert_(z[2] is masked)  # 断言 z 中索引为 2 的位置为 masked

    @suppress_copy_mask_on_assignment
    def test_oddfeatures_3(self):
        # 测试一些通用特性
        atest = array([10], mask=True)  # 创建一个带有掩码的数组 atest
        btest = array([20])  # 创建一个普通数组 btest
        idx = atest.mask  # 获取 atest 的掩码
        atest[idx] = btest[idx]  # 使用 btest 中相应的掩码部分填充 atest
        assert_equal(atest, [20])  # 断言 atest 与预期结果相等

    def test_filled_with_object_dtype(self):
        a = np.ma.masked_all(1, dtype='O')  # 创建一个填充了 masked 值的对象类型的数组 a
        assert_equal(a.filled('x')[0], 'x')  # 断言填充 a 后的第一个元素为 'x'

    def test_filled_with_flexible_dtype(self):
        # 测试使用灵活数据类型填充
        flexi = array([(1, 1, 1)], dtype=[('i', int), ('s', '|S8'), ('f', float)])  # 创建一个灵活数据类型的结构化数组
        flexi[0] = masked  # 将数组中的第一个元素设置为 masked
        assert_equal(flexi.filled(),
                     np.array([(default_fill_value(0),
                                default_fill_value('0'),
                                default_fill_value(0.),)], dtype=flexi.dtype))  # 断言填充 flexi 后的结果与预期结果相等
        flexi[0] = masked  # 再次将数组中的第一个元素设置为 masked
        assert_equal(flexi.filled(1),
                     np.array([(1, '1', 1.)], dtype=flexi.dtype))  # 断言使用给定值填充 flexi 后的结果与预期结果相等

    def test_filled_with_mvoid(self):
        # 测试使用 mvoid 填充
        ndtype = [('a', int), ('b', float)]  # 定义一个结构化数据类型
        a = mvoid((1, 2.), mask=[(0, 1)], dtype=ndtype)  # 创建一个带有掩码的 mvoid 数组 a
        # 使用默认值进行填充
        test = a.filled()
        assert_equal(tuple(test), (1, default_fill_value(1.)))  # 断言填充后的结果与预期结果相等
        # 使用显式的填充值
        test = a.filled((-1, -1))
        assert_equal(tuple(test), (1, -1))  # 断言填充后的结果与预期结果相等
        # 使用预定义的填充值
        a.fill_value = (-999, -999)
        assert_equal(tuple(a.filled()), (1, -999))  # 断言填充后的结果与预期结果相等
    def test_filled_with_nested_dtype(self):
        # Test filled w/ nested dtype
        # 定义一个嵌套的数据类型结构
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        # 创建一个结构化数组a，包含两行数据，每行数据包括一个整数和一个
    # 定义测试方法：测试使用复杂数据类型的掩码数组打印功能
    def test_fancy_printoptions(self):
        # 创建一个复杂的数据类型（结构化 dtype）
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        # 创建一个掩码数组（masked array），包含特定的数据和掩码
        test = array([(1, (2, 3.0)), (4, (5, 6.0))],
                     mask=[(1, (0, 1)), (0, (1, 0))],
                     dtype=fancydtype)
        # 预期的输出结果
        control = "[(--, (2, --)) (4, (--, 6.0))]"
        # 断言测试结果与预期结果相等
        assert_equal(str(test), control)

        # 测试零维数组和多维数据类型
        t_2d0 = masked_array(data=(0, [[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0]],
                                  0.0),
                             mask=(False, [[True, False, True],
                                           [False, False, True]],
                                   False),
                             dtype="int, (2,3)float, float")
        # 预期的输出结果
        control = "(0, [[--, 0.0, --], [0.0, 0.0, --]], 0.0)"
        # 断言测试结果与预期结果相等
        assert_equal(str(t_2d0), control)

    # 定义测试方法：测试 flatten_structured_array 函数
    def test_flatten_structured_array(self):
        # 测试在普通数组上的 flatten_structured_array 函数
        # 定义结构化数据类型
        ndtype = [('a', int), ('b', float)]
        # 创建普通数组
        a = np.array([(1, 1), (2, 2)], dtype=ndtype)
        # 调用 flatten_structured_array 函数进行测试
        test = flatten_structured_array(a)
        # 预期的输出结果
        control = np.array([[1., 1.], [2., 2.]], dtype=float)
        # 断言测试结果与预期结果相等
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)

        # 测试在掩码数组上的 flatten_structured_array 函数
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
        test = flatten_structured_array(a)
        control = array([[1., 1.], [2., 2.]],
                        mask=[[0, 1], [1, 0]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
        assert_equal(test.mask, control.mask)

        # 测试在具有嵌套结构的掩码数组上的 flatten_structured_array 函数
        ndtype = [('a', int), ('b', [('ba', int), ('bb', float)])]
        a = array([(1, (1, 1.1)), (2, (2, 2.2))],
                  mask=[(0, (1, 0)), (1, (0, 1))], dtype=ndtype)
        test = flatten_structured_array(a)
        control = array([[1., 1., 1.1], [2., 2., 2.2]],
                        mask=[[0, 1, 0], [1, 0, 1]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
        assert_equal(test.mask, control.mask)

        # 测试保持初始形状的 flatten_structured_array 函数
        ndtype = [('a', int), ('b', float)]
        a = np.array([[(1, 1), ], [(2, 2), ]], dtype=ndtype)
        test = flatten_structured_array(a)
        control = np.array([[[1., 1.], ], [[2., 2.], ]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
    def test_void0d(self):
        # 测试创建 mvoid 对象
        ndtype = [('a', int), ('b', int)]
        # 创建一个 NumPy 数组，并选择第一个元素
        a = np.array([(1, 2,)], dtype=ndtype)[0]
        # 使用数组元素创建 mvoid 对象
        f = mvoid(a)
        # 断言 f 是 mvoid 类型的实例
        assert_(isinstance(f, mvoid))

        # 创建带遮罩的 masked_array 对象
        a = masked_array([(1, 2)], mask=[(1, 0)], dtype=ndtype)[0]
        # 断言 a 是 mvoid 类型的实例
        assert_(isinstance(a, mvoid))

        # 创建带遮罩的 masked_array 对象，并选取数据和遮罩
        a = masked_array([(1, 2), (1, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
        # 使用数据和遮罩创建 mvoid 对象
        f = mvoid(a._data[0], a._mask[0])
        # 断言 f 是 mvoid 类型的实例
        assert_(isinstance(f, mvoid))

    def test_mvoid_getitem(self):
        # 测试 mvoid.__getitem__ 方法
        ndtype = [('a', int), ('b', int)]
        # 创建带遮罩的 masked_array 对象
        a = masked_array([(1, 2,), (3, 4)], mask=[(0, 0), (1, 0)],
                         dtype=ndtype)
        # 测试无遮罩时的 __getitem__ 方法
        f = a[0]
        # 断言 f 是 mvoid 类型的实例
        assert_(isinstance(f, mvoid))
        # 断言获取元素值和字段 'a' 的值
        assert_equal((f[0], f['a']), (1, 1))
        # 断言获取字段 'b' 的值
        assert_equal(f['b'], 2)
        # 测试有遮罩时的 __getitem__ 方法
        f = a[1]
        # 断言 f 是 mvoid 类型的实例
        assert_(isinstance(f, mvoid))
        # 断言遮罩元素和字段 'a' 的值是遮罩状态
        assert_(f[0] is masked)
        assert_(f['a'] is masked)
        # 断言获取字段 'b' 的值
        assert_equal(f[1], 4)

        # 测试特殊的数据类型
        A = masked_array(data=[([0,1],)],
                         mask=[([True, False],)],
                         dtype=[("A", ">i2", (2,))])
        # 断言获取字段 "A" 的值
        assert_equal(A[0]["A"], A["A"][0])
        # 断言获取字段 "A" 的值，并验证遮罩状态
        assert_equal(A[0]["A"], masked_array(data=[0, 1],
                         mask=[True, False], dtype=">i2"))

    def test_mvoid_iter(self):
        # 测试 __getitem__ 上的迭代
        ndtype = [('a', int), ('b', int)]
        # 创建带遮罩的 masked_array 对象
        a = masked_array([(1, 2,), (3, 4)], mask=[(0, 0), (1, 0)],
                         dtype=ndtype)
        # 测试无遮罩时的迭代
        assert_equal(list(a[0]), [1, 2])
        # 测试有遮罩时的迭代
        assert_equal(list(a[1]), [masked, 4])

    def test_mvoid_print(self):
        # 测试打印 mvoid 对象
        mx = array([(1, 1), (2, 2)], dtype=[('a', int), ('b', int)])
        # 断言打印结果
        assert_equal(str(mx[0]), "(1, 1)")
        # 修改 mx['b'][0] 为遮罩状态
        mx['b'][0] = masked
        # 保存初始的打印选项
        ini_display = masked_print_option._display
        # 设置新的打印选项
        masked_print_option.set_display("-X-")
        try:
            # 断言打印结果
            assert_equal(str(mx[0]), "(1, -X-)")
            assert_equal(repr(mx[0]), "(1, -X-)")
        finally:
            # 恢复初始的打印选项
            masked_print_option.set_display(ini_display)

        # 同时检查是否存在对象数据类型（参见 gh-7493）
        mx = array([(1,), (2,)], dtype=[('a', 'O')])
        # 断言打印结果
        assert_equal(str(mx[0]), "(1,)")
    def test_mvoid_multidim_print(self):

        # regression test for gh-6019
        # 创建一个多维的掩码数组 t_ma，用于测试回归问题 gh-6019
        t_ma = masked_array(data = [([1, 2, 3],)],
                            mask = [([False, True, False],)],
                            fill_value = ([999999, 999999, 999999],),
                            dtype = [('a', '<i4', (3,))])
        # 断言 t_ma[0] 的字符串表示符合预期
        assert_(str(t_ma[0]) == "([1, --, 3],)")
        # 断言 t_ma[0] 的详细表示符合预期
        assert_(repr(t_ma[0]) == "([1, --, 3],)")

        # additional tests with structured arrays
        # 使用结构化数组 t_2d 进行额外测试
        t_2d = masked_array(data = [([[1, 2], [3,4]],)],
                            mask = [([[False, True], [True, False]],)],
                            dtype = [('a', '<i4', (2,2))])
        # 断言 t_2d[0] 的字符串表示符合预期
        assert_(str(t_2d[0]) == "([[1, --], [--, 4]],)")
        # 断言 t_2d[0] 的详细表示符合预期
        assert_(repr(t_2d[0]) == "([[1, --], [--, 4]],)")

        # 创建一个零维的掩码数组 t_0d
        t_0d = masked_array(data = [(1,2)],
                            mask = [(True,False)],
                            dtype = [('a', '<i4'), ('b', '<i4')])
        # 断言 t_0d[0] 的字符串表示符合预期
        assert_(str(t_0d[0]) == "(--, 2)")
        # 断言 t_0d[0] 的详细表示符合预期
        assert_(repr(t_0d[0]) == "(--, 2)")

        # 使用带数组的对象进行测试
        t_2d = masked_array(data = [([[1, 2], [3,4]], 1)],
                            mask = [([[False, True], [True, False]], False)],
                            dtype = [('a', '<i4', (2,2)), ('b', float)])
        # 断言 t_2d[0] 的字符串表示符合预期
        assert_(str(t_2d[0]) == "([[1, --], [--, 4]], 1.0)")
        # 断言 t_2d[0] 的详细表示符合预期
        assert_(repr(t_2d[0]) == "([[1, --], [--, 4]], 1.0)")

        # 创建一个包含对象数组的掩码数组 t_ne 进行测试
        t_ne = masked_array(data=[(1, (1, 1))],
                            mask=[(True, (True, False))],
                            dtype = [('a', '<i4'), ('b', 'i4,i4')])
        # 断言 t_ne[0] 的字符串表示符合预期
        assert_(str(t_ne[0]) == "(--, (--, 1))")
        # 断言 t_ne[0] 的详细表示符合预期
        assert_(repr(t_ne[0]) == "(--, (--, 1))")

    def test_object_with_array(self):
        # 创建两个掩码数组 mx1 和 mx2，用于对象数组测试
        mx1 = masked_array([1.], mask=[True])
        mx2 = masked_array([1., 2.])
        # 创建一个对象数组 mx，包含上述两个掩码数组，并进行相关断言
        mx = masked_array([mx1, mx2], mask=[False, True], dtype=object)
        assert_(mx[0] is mx1)
        assert_(mx[1] is not mx2)
        assert_(np.all(mx[1].data == mx2.data))
        assert_(np.all(mx[1].mask))
        # 检查返回的视图是否正确
        mx[1].data[0] = 0.
        assert_(mx2[0] == 0.)
class TestMaskedArrayArithmetic:
    # MaskedArrays 的基础测试类。

    def setup_method(self):
        # 设置测试数据。
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        
        # 创建带掩码的 MaskedArray 对象
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        
        z = np.array([-.5, 0., .5, .8])
        zm = masked_array(z, mask=[0, 1, 0, 0])
        
        # 根据掩码设置填充值
        xf = np.where(m1, 1e+20, x)
        xm.set_fill_value(1e+20)
        
        # 设置测试数据的元组
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)
        
        # 保存当前的错误状态
        self.err_status = np.geterr()
        
        # 忽略除零和无效操作的错误
        np.seterr(divide='ignore', invalid='ignore')

    def teardown_method(self):
        # 恢复之前保存的错误状态
        np.seterr(**self.err_status)

    def test_basic_arithmetic(self):
        # 基本算术运算的测试。
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        
        a2d = array([[1, 2], [0, 4]])
        a2dm = masked_array(a2d, [[0, 0], [1, 0]])
        
        # 测试矩阵乘法
        assert_equal(a2d * a2d, a2d * a2dm)
        
        # 测试矩阵加法
        assert_equal(a2d + a2d, a2d + a2dm)
        
        # 测试矩阵减法
        assert_equal(a2d - a2d, a2d - a2dm)
        
        for s in [(12,), (4, 3), (2, 6)]:
            x = x.reshape(s)
            y = y.reshape(s)
            xm = xm.reshape(s)
            ym = ym.reshape(s)
            xf = xf.reshape(s)
            
            # 测试取反
            assert_equal(-x, -xm)
            
            # 测试加法
            assert_equal(x + y, xm + ym)
            
            # 测试减法
            assert_equal(x - y, xm - ym)
            
            # 测试乘法
            assert_equal(x * y, xm * ym)
            
            # 测试除法
            assert_equal(x / y, xm / ym)
            
            # 测试加常数
            assert_equal(a10 + y, a10 + ym)
            
            # 测试减常数
            assert_equal(a10 - y, a10 - ym)
            
            # 测试乘常数
            assert_equal(a10 * y, a10 * ym)
            
            # 测试除常数
            assert_equal(a10 / y, a10 / ym)
            
            # 测试矩阵加常数
            assert_equal(x + a10, xm + a10)
            
            # 测试矩阵减常数
            assert_equal(x - a10, xm - a10)
            
            # 测试矩阵乘常数
            assert_equal(x * a10, xm * a10)
            
            # 测试矩阵除常数
            assert_equal(x / a10, xm / a10)
            
            # 测试矩阵平方
            assert_equal(x ** 2, xm ** 2)
            
            # 测试绝对值后的操作
            assert_equal(abs(x) ** 2.5, abs(xm) ** 2.5)
            
            # 测试矩阵幂运算
            assert_equal(x ** y, xm ** ym)
            
            # 使用 numpy 函数测试加法
            assert_equal(np.add(x, y), add(xm, ym))
            
            # 使用 numpy 函数测试减法
            assert_equal(np.subtract(x, y), subtract(xm, ym))
            
            # 使用 numpy 函数测试乘法
            assert_equal(np.multiply(x, y), multiply(xm, ym))
            
            # 使用 numpy 函数测试除法
            assert_equal(np.divide(x, y), divide(xm, ym))

    def test_divide_on_different_shapes(self):
        x = arange(6, dtype=float)
        x.shape = (2, 3)
        y = arange(3, dtype=float)

        # 测试二维数组除法
        z = x / y
        assert_equal(z, [[-1., 1., 1.], [-1., 4., 2.5]])
        assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])

        # 测试二维数组除法，使用 y 的行广播
        z = x / y[None,:]
        assert_equal(z, [[-1., 1., 1.], [-1., 4., 2.5]])
        assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])

        y = arange(2, dtype=float)
        
        # 测试二维数组除法，使用 y 的列广播
        z = x / y[:, None]
        assert_equal(z, [[-1., -1., -1.], [3., 4., 5.]])
        assert_equal(z.mask, [[1, 1, 1], [0, 0, 0]])
    def test_mixed_arithmetic(self):
        # Tests mixed arithmetic.
        # 创建一个 NumPy 数组 na，包含一个整数 1
        na = np.array([1])
        # 创建一个 NumPy 数组 ma，使用 array 函数创建，包含一个整数 1
        ma = array([1])
        # 断言 na + ma 的结果是 MaskedArray 类型
        assert_(isinstance(na + ma, MaskedArray))
        # 断言 ma + na 的结果是 MaskedArray 类型
        assert_(isinstance(ma + na, MaskedArray))

    def test_limits_arithmetic(self):
        # Tests limits arithmetic.
        # 计算 float 类型的最小正数
        tiny = np.finfo(float).tiny
        # 创建一个 NumPy 数组 a，包含三个元素：tiny、1/tiny、0
        a = array([tiny, 1. / tiny, 0.])
        # 断言 a / 2 的掩码数组为 [0, 0, 0]
        assert_equal(getmaskarray(a / 2), [0, 0, 0])
        # 断言 2 / a 的掩码数组为 [1, 0, 1]
        assert_equal(getmaskarray(2 / a), [1, 0, 1])

    def test_masked_singleton_arithmetic(self):
        # Tests some scalar arithmetic on MaskedArrays.
        # Masked singleton should remain masked no matter what
        # 创建一个带有掩码的标量 MaskedArray xm，值为 0，掩码为 1
        xm = array(0, mask=1)
        # 断言 (1 / array(0)).mask 是 True
        assert_((1 / array(0)).mask)
        # 断言 (1 + xm).mask 是 True
        assert_((1 + xm).mask)
        # 断言 (-xm).mask 是 True
        assert_((-xm).mask)
        # 断言 maximum(xm, xm).mask 是 True
        assert_(maximum(xm, xm).mask)
        # 断言 minimum(xm, xm).mask 是 True
        assert_(minimum(xm, xm).mask)

    def test_masked_singleton_equality(self):
        # Tests (in)equality on masked singleton
        # 创建一个带有掩码的数组 a，包含元素 [1, 2, 3]，掩码为 [1, 1, 0]
        a = array([1, 2, 3], mask=[1, 1, 0])
        # 断言 a[0] == 0 是 masked
        assert_((a[0] == 0) is masked)
        # 断言 a[0] != 0 是 masked
        assert_((a[0] != 0) is masked)
        # 断言 (a[-1] == 0) 的结果是 False
        assert_equal((a[-1] == 0), False)
        # 断言 (a[-1] != 0) 的结果是 True
        assert_equal((a[-1] != 0), True)

    def test_arithmetic_with_masked_singleton(self):
        # Checks that there's no collapsing to masked
        # 创建一个 MaskedArray x，包含元素 [1, 2]
        x = masked_array([1, 2])
        # 将 x 与 masked 进行乘法运算，得到 y
        y = x * masked
        # 断言 y 的形状与 x 相同
        assert_equal(y.shape, x.shape)
        # 断言 y 的掩码数组为 [True, True]
        assert_equal(y._mask, [True, True])
        # 将 x[0] 与 masked 进行乘法运算，得到 y
        y = x[0] * masked
        # 断言 y 是 masked
        assert_(y is masked)
        # 将 x 与 masked 进行加法运算，得到 y
        y = x + masked
        # 断言 y 的形状与 x 相同
        assert_equal(y.shape, x.shape)
        # 断言 y 的掩码数组为 [True, True]
        assert_equal(y._mask, [True, True])

    def test_arithmetic_with_masked_singleton_on_1d_singleton(self):
        # Check that we're not losing the shape of a singleton
        # 创建一个 MaskedArray x，包含元素 [1, ]，形状为单维数组
        x = masked_array([1, ])
        # 将 x 与 masked 进行加法运算，得到 y
        y = x + masked
        # 断言 y 的形状与 x 相同
        assert_equal(y.shape, x.shape)
        # 断言 y 的掩码数组为 [True, ]
        assert_equal(y.mask, [True, ])

    def test_scalar_arithmetic(self):
        # Tests scalar arithmetic.
        # 创建一个标量 MaskedArray x，值为 0，无掩码
        x = array(0, mask=0)
        # 断言 x.filled().ctypes.data 与 x.ctypes.data 相等
        assert_equal(x.filled().ctypes.data, x.ctypes.data)
        # 创建一个数组 xm，包含元素 (0, 0)，在除零操作时生成掩码
        xm = array((0, 0)) / 0.
        # 断言 xm 的形状为 (2,)
        assert_equal(xm.shape, (2,))
        # 断言 xm 的掩码为 [1, 1]
        assert_equal(xm.mask, [1, 1])
    # 定义测试基本函数的方法
    def test_basic_ufuncs(self):
        # 测试各种函数，如 sin，cos 等
        # 获取参数值
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # 断言cos函数在参数x和xm上的结果相等
        assert_equal(np.cos(x), cos(xm))
        # 断言cosh函数在参数x和xm上的结果相等
        assert_equal(np.cosh(x), cosh(xm))
        # 断言sin函数在参数x和xm上的结果相等
        assert_equal(np.sin(x), sin(xm))
        # 断言sinh函数在参数x和xm上的结果相等
        assert_equal(np.sinh(x), sinh(xm))
        # 断言tan函数在参数x和xm上的结果相等
        assert_equal(np.tan(x), tan(xm))
        # 断言tanh函数在参数x和xm上的结果相等
        assert_equal(np.tanh(x), tanh(xm))
        # 断言sqrt函数在参数x和xm上的结果相等
        assert_equal(np.sqrt(abs(x)), sqrt(xm))
        # 断言log函数在参数x和xm上的结果相等
        assert_equal(np.log(abs(x)), log(xm))
        # 断言log10函数在参数x和xm上的结果相等
        assert_equal(np.log10(abs(x)), log10(xm))
        # 断言exp函数在参数x和xm上的结果相等
        assert_equal(np.exp(x), exp(xm))
        # 断言arcsin函数在参数z和zm上的结果相等
        assert_equal(np.arcsin(z), arcsin(zm))
        # 断言arccos函数在参数z和zm上的结果相等
        assert_equal(np.arccos(z), arccos(zm))
        # 断言arctan函数在参数z和zm上的结果相等
        assert_equal(np.arctan(z), arctan(zm))
        # 断言arctan2函数在参数x, y和xm, ym上的结果相等
        assert_equal(np.arctan2(x, y), arctan2(xm, ym))
        # 断言absolute函数在参数x和xm上的结果相等
        assert_equal(np.absolute(x), absolute(xm))
        # 断言angle函数在参数x+1j*y和xm+1j*ym上的结果相等
        assert_equal(np.angle(x + 1j*y), angle(xm + 1j*ym))
        # 断言带有deg参数的angle函数在参数x+1j*y和xm+1j*ym上的结果相等
        assert_equal(np.angle(x + 1j*y, deg=True), angle(xm + 1j*ym, deg=True))
        # 断言equal函数在参数x和y和xm, ym上的结果相等
        assert_equal(np.equal(x, y), equal(xm, ym))
        # 断言not_equal函数在参数x和y和xm, ym上的结果相等
        assert_equal(np.not_equal(x, y), not_equal(xm, ym))
        # 断言less函数在参数x和y和xm, ym上的结果相等
        assert_equal(np.less(x, y), less(xm, ym))
        # 断言greater函数在参数x和y和xm, ym上的结果相等
        assert_equal(np.greater(x, y), greater(xm, ym))
        # 断言less_equal函数在参数x和y和xm, ym上的结果相等
        assert_equal(np.less_equal(x, y), less_equal(xm, ym))
        # 断言greater_equal函数在参数x和y和xm, ym上的结果相等
        assert_equal(np.greater_equal(x, y), greater_equal(xm, ym))
        # 断言conjugate函数在参数x和xm上的结果相等
        assert_equal(np.conjugate(x), conjugate(xm))

    # 测试count_func方法
    def test_count_func(self):
        # 测试count函数
        # 断言count(1)的结果为1
        assert_equal(1, count(1))
        # 断言array(1, mask=[1])的结果为0
        assert_equal(0, array(1, mask=[1]))

        # 创建带有mask的数组
        ott = array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        # 断言执行count函数的结果为3
        res = count(ott)
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)

        # 将数组形状改变为(2,2)
        ott = ott.reshape((2, 2))
        # 断言执行count函数的结果为3
        res = count(ott)
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)
        # 在axis=0上执行count函数
        res = count(ott, 0)
        assert_(isinstance(res, ndarray))
        assert_equal([1, 2], res)
        assert_(getmask(res) is nomask)

        # 创建不带mask的数组
        ott = array([0., 1., 2., 3.])
        # 在axis=0上执行count函数
        res = count(ott, 0)
        assert_(isinstance(res, ndarray))
        assert_(res.dtype.type is np.intp)
        # 期待抛出AxisError异常
        assert_raises(AxisError, ott.count, axis=1)

    # 测试count_on_python_builtins方法
    def test_count_on_python_builtins(self):
        # 测试count函数对python内置函数的工作情况
        assert_equal(3, count([1,2,3]))
        assert_equal(2, count((1,2))
    def test_minmax_func(self):
        # Tests minimum and maximum functions.

        # 解包元组 self.d 中的变量
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d

        # 如果 x 已经是扁平化的则不影响最大值函数
        xr = np.ravel(x)
        xmr = np.ravel(xm)

        # 由于数据的精心选择，以下断言为真
        assert_equal(max(xr), np.maximum.reduce(xmr))
        assert_equal(min(xr), np.minimum.reduce(xmr))

        # 断言以下语句为真，因为数据的精心选择
        assert_equal(np.minimum([1, 2, 3], [4, 0, 9]), [1, 0, 3])
        assert_equal(np.maximum([1, 2, 3], [4, 0, 9]), [4, 2, 9])

        # 创建数组 x 和 y
        x = np.arange(5)
        y = np.arange(5) - 2

        # 在 x 中标记第三个元素为 masked
        x[3] = np.ma.masked
        # 在 y 中标记第一个元素为 masked
        y[0] = np.ma.masked

        # 断言 minimum 函数的输出
        assert_equal(np.minimum(x, y), np.where(np.less(x, y), x, y))
        # 断言 maximum 函数的输出
        assert_equal(np.maximum(x, y), np.where(np.greater(x, y), x, y))

        # 断言最小值 reduce 函数的输出
        assert_(np.minimum.reduce(x) == 0)
        # 断言最大值 reduce 函数的输出
        assert_(np.maximum.reduce(x) == 4)

        # 创建一个形状为 (2, 2) 的数组 x
        x = np.arange(4).reshape(2, 2)
        # 在数组 x 的最后一个元素位置标记为 masked
        x[-1, -1] = np.ma.masked

        # 断言最大值 reduce 函数在无轴情况下的输出
        assert_equal(np.maximum.reduce(x, axis=None), 2)

    def test_minimummaximum_func(self):
        # Tests the minimum and maximum functions on arrays with specified shapes.

        # 创建一个全为 1 的 2x2 数组 a
        a = np.ones((2, 2))

        # 使用 minimum 函数计算 a 与 a 的最小值
        aminimum = np.minimum(a, a)
        assert_(isinstance(aminimum, np.ma.MaskedArray))
        assert_equal(aminimum, np.minimum(a, a))

        # 使用 minimum.outer 函数计算 a 与 a 的最小值
        aminimum = np.minimum.outer(a, a)
        assert_(isinstance(aminimum, np.ma.MaskedArray))
        assert_equal(aminimum, np.minimum.outer(a, a))

        # 使用 maximum 函数计算 a 与 a 的最大值
        amaximum = np.maximum(a, a)
        assert_(isinstance(amaximum, np.ma.MaskedArray))
        assert_equal(amaximum, np.maximum(a, a))

        # 使用 maximum.outer 函数计算 a 与 a 的最大值
        amaximum = np.maximum.outer(a, a)
        assert_(isinstance(amaximum, np.ma.MaskedArray))
        assert_equal(amaximum, np.maximum.outer(a, a))

    def test_minmax_reduce(self):
        # Test np.min/maximum.reduce on array with full False mask

        # 创建一个带有全 False mask 的数组 a
        a = np.ma.array([1, 2, 3], mask=[False, False, False])

        # 使用 np.maximum.reduce 函数
        b = np.maximum.reduce(a)
        assert_equal(b, 3)

    def test_minmax_funcs_with_output(self):
        # Tests the min/max functions with explicit outputs

        # 创建一个随机的 mask
        mask = np.random.rand(12).round()
        # 创建一个带有 mask 的随机数组 xm
        xm = np.ma.array(np.random.uniform(0, 10, 12), mask=mask)
        xm.shape = (3, 4)

        # 遍历 'min' 和 'max' 两个函数名
        for funcname in ('min', 'max'):
            # 获取对应的 np 版本和 ma 版本的函数
            npfunc = getattr(np, funcname)
            mafunc = getattr(np.ma.core, funcname)

            # 初始化一个 int 类型的输出数组 nout
            nout = np.empty((4,), dtype=int)
            try:
                # 尝试使用 np 版本的函数，设置输出为 nout
                result = npfunc(xm, axis=0, out=nout)
            except np.ma.MaskError:
                pass
            # 初始化一个 float 类型的输出数组 nout
            nout = np.empty((4,), dtype=float)
            # 使用 np 版本的函数，设置输出为 nout
            result = npfunc(xm, axis=0, out=nout)
            # 断言结果与 nout 是同一个对象
            assert_(result is nout)

            # 初始化一个填充了 -999 的输出数组 nout
            nout.fill(-999)
            # 使用 ma 版本的函数，设置输出为 nout
            result = mafunc(xm, axis=0, out=nout)
            # 断言结果与 nout 是同一个对象
            assert_(result is nout)
    def test_minmax_methods(self):
        # 测试最大值和最小值方法的附加用例
        # 解构元组 self.d 中的第六个元素 xm
        (_, _, _, _, _, xm, _, _, _, _) = self.d
        # 调整 xm 的形状为一维数组
        xm.shape = (xm.size,)
        # 断言 xm 的最大值为 10
        assert_equal(xm.max(), 10)
        # 断言 xm[0] 的最大值是掩码值
        assert_(xm[0].max() is masked)
        # 断言 xm[0] 在轴 0 上的最大值是掩码值
        assert_(xm[0].max(0) is masked)
        # 断言 xm[0] 在轴 -1 上的最大值是掩码值
        assert_(xm[0].max(-1) is masked)
        # 断言 xm 的最小值为 -10.0
        assert_equal(xm.min(), -10.)
        # 断言 xm[0] 的最小值是掩码值
        assert_(xm[0].min() is masked)
        # 断言 xm[0] 在轴 0 上的最小值是掩码值
        assert_(xm[0].min(0) is masked)
        # 断言 xm[0] 在轴 -1 上的最小值是掩码值
        assert_(xm[0].min(-1) is masked)
        # 断言 xm 的峰-峰值为 20.0
        assert_equal(xm.ptp(), 20.)
        # 断言 xm[0] 的峰-峰值是掩码值
        assert_(xm[0].ptp() is masked)
        # 断言 xm[0] 在轴 0 上的峰-峰值是掩码值
        assert_(xm[0].ptp(0) is masked)
        # 断言 xm[0] 在轴 -1 上的峰-峰值是掩码值
        assert_(xm[0].ptp(-1) is masked)

        # 创建一个带掩码的数组 x
        x = array([1, 2, 3], mask=True)
        # 断言 x 的最小值是掩码值
        assert_(x.min() is masked)
        # 断言 x 的最大值是掩码值
        assert_(x.max() is masked)
        # 断言 x 的峰-峰值是掩码值
        assert_(x.ptp() is masked)

    def test_minmax_dtypes(self):
        # 针对非标准浮点数和复数类型的最大值和最小值进行附加测试
        # 创建数组 x
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        # 定义浮点数常量
        a10 = 10.
        an10 = -10.0
        # 创建掩码数组 m1
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        # 使用 m1 创建掩码数组 xm
        xm = masked_array(x, mask=m1)
        # 设置 xm 的填充值
        xm.set_fill_value(1e+20)
        # 定义浮点数数据类型列表 float_dtypes
        float_dtypes = [np.float16, np.float32, np.float64, np.longdouble,
                        np.complex64, np.complex128, np.clongdouble]
        # 遍历浮点数数据类型列表
        for float_dtype in float_dtypes:
            # 断言使用指定数据类型创建的 xm 的最大值与浮点数常量 a10 相等
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).max(),
                         float_dtype(a10))
            # 断言使用指定数据类型创建的 xm 的最小值与浮点数常量 an10 相等
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).min(),
                         float_dtype(an10))

        # 断言 xm 的最小值与 an10 相等
        assert_equal(xm.min(), an10)
        # 断言 xm 的最大值与 a10 相等
        assert_equal(xm.max(), a10)

        # 仅针对非复数类型进行测试
        for float_dtype in float_dtypes[:4]:
            # 断言使用指定数据类型创建的 xm 的最大值与浮点数常量 a10 相等
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).max(),
                         float_dtype(a10))
            # 断言使用指定数据类型创建的 xm 的最小值与浮点数常量 an10 相等
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).min(),
                         float_dtype(an10))

        # 仅针对复数类型进行测试
        for float_dtype in float_dtypes[-3:]:
            # 创建掩码数组 ym
            ym = masked_array([1e20+1j, 1e20-2j, 1e20-1j], mask=[0, 1, 0],
                              dtype=float_dtype)
            # 断言 ym 的最小值与复数常量 1e20-1j 相等
            assert_equal(ym.min(), float_dtype(1e20-1j))
            # 断言 ym 的最大值与复数常量 1e20+1j 相等
            assert_equal(ym.max(), float_dtype(1e20+1j))

            # 创建掩码数组 zm
            zm = masked_array([np.inf+2j, np.inf+3j, -np.inf-1j], mask=[0, 1, 0],
                              dtype=float_dtype)
            # 断言 zm 的最小值与复数常量 -np.inf-1j 相等
            assert_equal(zm.min(), float_dtype(-np.inf-1j))
            # 断言 zm 的最大值与复数常量 np.inf+2j 相等
            assert_equal(zm.max(), float_dtype(np.inf+2j))

            # 创建掩码数组 cmax
            cmax = np.inf - 1j * np.finfo(np.float64).max
            # 断言创建的掩码数组的最大值为 -cmax
            assert masked_array([-cmax, 0], mask=[0, 1]).max() == -cmax
            # 断言创建的掩码数组的最小值为 cmax
            assert masked_array([cmax, 0], mask=[0, 1]).min() == cmax
    def test_addsumprod(self):
        # Tests add, sum, product.
        # 从元组 self.d 中解包得到变量 x, y, a10, m1, m2, xm, ym, z, zm, xf
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # 断言 np.add.reduce(x) 和 add.reduce(x) 的结果相等
        assert_equal(np.add.reduce(x), add.reduce(x))
        # 断言 np.add.accumulate(x) 和 add.accumulate(x) 的结果相等
        assert_equal(np.add.accumulate(x), add.accumulate(x))
        # 断言 sum(array(4), axis=0) 等于 4
        assert_equal(4, sum(array(4), axis=0))
        # 断言 sum(array(4), axis=0) 等于 4
        assert_equal(4, sum(array(4), axis=0))
        # 断言 np.sum(x, axis=0) 和 sum(x, axis=0) 的结果相等
        assert_equal(np.sum(x, axis=0), sum(x, axis=0))
        # 断言 np.sum(filled(xm, 0), axis=0) 和 sum(xm, axis=0) 的结果相等
        assert_equal(np.sum(filled(xm, 0), axis=0), sum(xm, axis=0))
        # 断言 np.sum(x, 0) 和 sum(x, 0) 的结果相等
        assert_equal(np.sum(x, 0), sum(x, 0))
        # 断言 np.prod(x, axis=0) 和 product(x, axis=0) 的结果相等
        assert_equal(np.prod(x, axis=0), product(x, axis=0))
        # 断言 np.prod(x, 0) 和 product(x, 0) 的结果相等
        assert_equal(np.prod(x, 0), product(x, 0))
        # 断言 np.prod(filled(xm, 1), axis=0) 和 product(xm, axis=0) 的结果相等
        assert_equal(np.prod(filled(xm, 1), axis=0), product(xm, axis=0))
        
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = s
        # 如果 s 的长度大于 1
        if len(s) > 1:
            # 断言 np.concatenate((x, y), 1) 和 concatenate((xm, ym), 1) 的结果相等
            assert_equal(np.concatenate((x, y), 1), concatenate((xm, ym), 1))
            # 断言 np.add.reduce(x, 1) 和 add.reduce(x, 1) 的结果相等
            assert_equal(np.add.reduce(x, 1), add.reduce(x, 1))
            # 断言 np.sum(x, 1) 和 sum(x, 1) 的结果相等
            assert_equal(np.sum(x, 1), sum(x, 1))
            # 断言 np.prod(x, 1) 和 product(x, 1) 的结果相等
            assert_equal(np.prod(x, 1), product(x, 1))

    def test_binops_d2D(self):
        # Test binary operations on 2D data
        # 创建 array a 和 b
        a = array([[1.], [2.], [3.]], mask=[[False], [True], [True]])
        b = array([[2., 3.], [4., 5.], [6., 7.]])

        # 测试 a * b 的结果是否等于 control
        test = a * b
        control = array([[2., 3.], [2., 2.], [3., 3.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        # 测试 b * a 的结果是否等于 control
        test = b * a
        control = array([[2., 3.], [4., 5.], [6., 7.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        # 重新定义 a 和 b
        a = array([[1.], [2.], [3.]])
        b = array([[2., 3.], [4., 5.], [6., 7.]],
                  mask=[[0, 0], [0, 0], [0, 1]])
        
        # 测试 a * b 的结果是否等于 control
        test = a * b
        control = array([[2, 3], [8, 10], [18, 3]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        # 测试 b * a 的结果是否等于 control
        test = b * a
        control = array([[2, 3], [8, 10], [18, 7]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)
    def test_domained_binops_d2D(self):
        # Test domained binary operations on 2D data
        
        # 创建一个带掩码的二维数组 `a`
        a = array([[1.], [2.], [3.]], mask=[[False], [True], [True]])
        # 创建一个二维数组 `b`
        b = array([[2., 3.], [4., 5.], [6., 7.]])
        
        # 执行 `a` 与 `b` 的除法运算，并将结果保存在 `test` 中
        test = a / b
        # 创建控制数组 `control` 以验证计算结果
        control = array([[1. / 2., 1. / 3.], [2., 2.], [3., 3.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        # 断言 `test` 与 `control` 相等
        assert_equal(test, control)
        # 断言 `test` 的数据部分与 `control` 的数据部分相等
        assert_equal(test.data, control.data)
        # 断言 `test` 的掩码部分与 `control` 的掩码部分相等
        assert_equal(test.mask, control.mask)

        # 执行 `b` 与 `a` 的除法运算，并将结果保存在 `test` 中
        test = b / a
        # 创建控制数组 `control` 以验证计算结果
        control = array([[2. / 1., 3. / 1.], [4., 5.], [6., 7.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        # 断言 `test` 与 `control` 相等
        assert_equal(test, control)
        # 断言 `test` 的数据部分与 `control` 的数据部分相等
        assert_equal(test.data, control.data)
        # 断言 `test` 的掩码部分与 `control` 的掩码部分相等
        assert_equal(test.mask, control.mask)

        # 创建一个不带掩码的二维数组 `a`
        a = array([[1.], [2.], [3.]])
        # 创建一个带掩码的二维数组 `b`
        b = array([[2., 3.], [4., 5.], [6., 7.]],
                  mask=[[0, 0], [0, 0], [0, 1]])
        # 执行 `a` 与 `b` 的除法运算，并将结果保存在 `test` 中
        test = a / b
        # 创建控制数组 `control` 以验证计算结果
        control = array([[1. / 2, 1. / 3], [2. / 4, 2. / 5], [3. / 6, 3]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        # 断言 `test` 与 `control` 相等
        assert_equal(test, control)
        # 断言 `test` 的数据部分与 `control` 的数据部分相等
        assert_equal(test.data, control.data)
        # 断言 `test` 的掩码部分与 `control` 的掩码部分相等
        assert_equal(test.mask, control.mask)

        # 执行 `b` 与 `a` 的除法运算，并将结果保存在 `test` 中
        test = b / a
        # 创建控制数组 `control` 以验证计算结果
        control = array([[2 / 1., 3 / 1.], [4 / 2., 5 / 2.], [6 / 3., 7]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        # 断言 `test` 与 `control` 相等
        assert_equal(test, control)
        # 断言 `test` 的数据部分与 `control` 的数据部分相等
        assert_equal(test.data, control.data)
        # 断言 `test` 的掩码部分与 `control` 的掩码部分相等
        assert_equal(test.mask, control.mask)
    def test_TakeTransposeInnerOuter(self):
        # Test of take, transpose, inner, outer products
        x = arange(24)  # 创建一个长度为24的一维数组x，内容为0到23的连续整数
        y = np.arange(24)  # 创建一个长度为24的一维数组y，内容同样为0到23的连续整数
        x[5:6] = masked  # 将数组x中索引为5的元素设置为masked（未定义的值）
        x = x.reshape(2, 3, 4)  # 将数组x重新形状为一个3维数组，形状为(2, 3, 4)
        y = y.reshape(2, 3, 4)  # 将数组y也重新形状为一个3维数组，形状同样为(2, 3, 4)
        assert_equal(np.transpose(y, (2, 0, 1)), transpose(x, (2, 0, 1)))
        # 断言两个数组的转置结果相等，分别使用指定的轴顺序进行转置
        assert_equal(np.take(y, (2, 0, 1), 1), take(x, (2, 0, 1), 1))
        # 断言两个数组按指定轴和索引取值后的结果相等
        assert_equal(np.inner(filled(x, 0), filled(y, 0)),
                     inner(x, y))
        # 断言两个数组的内积（inner product）结果相等，使用填充函数将masked值填充为0
        assert_equal(np.outer(filled(x, 0), filled(y, 0)),
                     outer(x, y))
        # 断言两个数组的外积（outer product）结果相等，同样使用填充函数将masked值填充为0
        y = array(['abc', 1, 'def', 2, 3], object)
        y[2] = masked  # 将数组y中索引为2的元素设置为masked
        t = take(y, [0, 3, 4])  # 使用指定索引从数组y中取值形成新数组t
        assert_(t[0] == 'abc')  # 断言t的第一个元素为'abc'
        assert_(t[1] == 2)  # 断言t的第二个元素为2
        assert_(t[2] == 3)  # 断言t的第三个元素为3

    def test_imag_real(self):
        # Check complex
        xx = array([1 + 10j, 20 + 2j], mask=[1, 0])  # 创建一个复数数组xx，其中包含有一个masked值
        assert_equal(xx.imag, [10, 2])  # 断言xx数组的虚部与给定列表相等
        assert_equal(xx.imag.filled(), [1e+20, 2])
        # 断言xx数组的虚部使用填充方法后与给定列表相等（填充masked值为默认值1e+20）
        assert_equal(xx.imag.dtype, xx._data.imag.dtype)
        # 断言xx数组的虚部数据类型与其底层数据的虚部数据类型相同
        assert_equal(xx.real, [1, 20])  # 断言xx数组的实部与给定列表相等
        assert_equal(xx.real.filled(), [1e+20, 20])
        # 断言xx数组的实部使用填充方法后与给定列表相等（填充masked值为默认值1e+20）
        assert_equal(xx.real.dtype, xx._data.real.dtype)
        # 断言xx数组的实部数据类型与其底层数据的实部数据类型相同

    def test_methods_with_output(self):
        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked
        # 将数组xm的列和行的部分元素设置为masked

        funclist = ('sum', 'prod', 'var', 'std', 'max', 'min', 'ptp', 'mean',)
        # 定义要测试的数组方法列表

        for funcname in funclist:
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)
            # 获取numpy库和数组xm中的对应方法

            # A ndarray as explicit input
            output = np.empty(4, dtype=float)  # 创建一个长度为4的空浮点型数组output
            output.fill(-9999)  # 将数组output填充为-9999
            result = npfunc(xm, axis=0, out=output)
            # 使用numpy方法计算数组xm的函数，指定轴为0，输出结果到数组output
            assert_(result is output)  # 断言计算结果与output是同一个对象
            assert_equal(result, xmmeth(axis=0, out=output))
            # 断言使用数组xm的方法计算结果与使用numpy方法计算结果相等

            output = empty(4, dtype=int)  # 创建一个长度为4的空整型数组output
            result = xmmeth(axis=0, out=output)
            # 使用数组xm的方法计算结果，指定轴为0，输出结果到数组output
            assert_(result is output)  # 断言计算结果与output是同一个对象
            assert_(output[0] is masked)
            # 断言output的第一个元素是masked
    def test_eq_on_structured(self):
        # 测试结构化数组的相等性
        ndtype = [('A', int), ('B', int)]
        # 创建结构化数组a，包括数据和掩码
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (0, 0)], dtype=ndtype)

        # 测试a与自身的相等性
        test = (a == a)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        # 测试a与a的第一个元素的相等性
        test = (a == a[0])
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        # 创建结构化数组b，包括数据和掩码
        b = array([(1, 1), (2, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)

        # 测试a与b的相等性
        test = (a == b)
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        # 测试a的第一个元素与b的相等性
        test = (a[0] == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        # 创建结构化数组b，包括数据和掩码
        b = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)

        # 测试a与b的相等性
        test = (a == b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        # 复杂的数据类型，二维数组
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        a = array([[(1, (1, 1)), (2, (2, 2))],
                   [(3, (3, 3)), (4, (4, 4))]],
                  mask=[[(0, (1, 0)), (0, (0, 1))],
                        [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)

        # 测试a的第一个元素与整个数组a的相等性
        test = (a[0, 0] == a)
        assert_equal(test.data, [[True, False], [False, False]])
        assert_equal(test.mask, [[False, False], [False, True]])
        assert_(test.fill_value == True)
    def test_ne_on_structured(self):
        # Test the equality of structured arrays

        # 定义结构化数组的数据类型
        ndtype = [('A', int), ('B', int)]

        # 创建一个结构化数组 `a`，带有数据和遮罩
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (0, 0)], dtype=ndtype)

        # 测试 `a` 是否不等于自身，结果存储在 `test` 中
        test = (a != a)
        assert_equal(test.data, [False, False])  # 断言 `test.data` 的预期值
        assert_equal(test.mask, [False, False])  # 断言 `test.mask` 的预期值
        assert_(test.fill_value == True)  # 断言 `test.fill_value` 的预期值

        # 测试 `a` 是否不等于其第一个元素，结果存储在 `test` 中
        test = (a != a[0])
        assert_equal(test.data, [False, True])  # 断言 `test.data` 的预期值
        assert_equal(test.mask, [False, False])  # 断言 `test.mask` 的预期值
        assert_(test.fill_value == True)  # 断言 `test.fill_value` 的预期值

        # 创建另一个结构化数组 `b`，带有数据和遮罩
        b = array([(1, 1), (2, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)

        # 测试 `a` 是否不等于 `b`，结果存储在 `test` 中
        test = (a != b)
        assert_equal(test.data, [True, False])  # 断言 `test.data` 的预期值
        assert_equal(test.mask, [True, False])  # 断言 `test.mask` 的预期值
        assert_(test.fill_value == True)  # 断言 `test.fill_value` 的预期值

        # 测试 `a` 的第一个元素是否不等于 `b`，结果存储在 `test` 中
        test = (a[0] != b)
        assert_equal(test.data, [True, True])  # 断言 `test.data` 的预期值
        assert_equal(test.mask, [True, False])  # 断言 `test.mask` 的预期值
        assert_(test.fill_value == True)  # 断言 `test.fill_value` 的预期值

        # 修改 `b` 的遮罩
        b = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)

        # 再次测试 `a` 是否不等于 `b`，结果存储在 `test` 中
        test = (a != b)
        assert_equal(test.data, [False, False])  # 断言 `test.data` 的预期值
        assert_equal(test.mask, [False, False])  # 断言 `test.mask` 的预期值
        assert_(test.fill_value == True)  # 断言 `test.fill_value` 的预期值

        # 复杂数据类型测试，二维数组 `a` 的某个元素是否不等于整个数组 `a`
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        a = array([[(1, (1, 1)), (2, (2, 2))],
                   [(3, (3, 3)), (4, (4, 4))]],
                  mask=[[(0, (1, 0)), (0, (0, 1))],
                        [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)
        test = (a[0, 0] != a)
        assert_equal(test.data, [[False, True], [True, True]])  # 断言 `test.data` 的预期值
        assert_equal(test.mask, [[False, False], [False, True]])  # 断言 `test.mask` 的预期值
        assert_(test.fill_value == True)  # 断言 `test.fill_value` 的预期值
    def test_eq_ne_structured_extra(self):
        # 确保简单示例对称且合理。
        # 来自 https://github.com/numpy/numpy/pull/8590#discussion_r101126465
        # 定义一个包含两个整数字段的数据类型
        dt = np.dtype('i4,i4')
        # 遍历不同的 m1 示例
        for m1 in (mvoid((1, 2), mask=(0, 0), dtype=dt),
                   mvoid((1, 2), mask=(0, 1), dtype=dt),
                   mvoid((1, 2), mask=(1, 0), dtype=dt),
                   mvoid((1, 2), mask=(1, 1), dtype=dt)):
            # 将 m1 转换为 MaskedArray
            ma1 = m1.view(MaskedArray)
            # 获取 ma1 的视图 '2i4'
            r1 = ma1.view('2i4')
            # 再次遍历不同的 m2 示例
            for m2 in (np.array((1, 1), dtype=dt),
                       mvoid((1, 1), dtype=dt),
                       mvoid((1, 0), mask=(0, 1), dtype=dt),
                       mvoid((3, 2), mask=(0, 1), dtype=dt)):
                # 将 m2 转换为 MaskedArray
                ma2 = m2.view(MaskedArray)
                # 获取 ma2 的视图 '2i4'
                r2 = ma2.view('2i4')
                # 预期 r1 与 r2 的全等性
                eq_expected = (r1 == r2).all()
                # 断言 m1 与 m2 是否相等，期望结果是 eq_expected
                assert_equal(m1 == m2, eq_expected)
                # 断言 m2 与 m1 是否相等，期望结果是 eq_expected
                assert_equal(m2 == m1, eq_expected)
                # 断言 ma1 与 m2 是否相等，期望结果是 eq_expected
                assert_equal(ma1 == m2, eq_expected)
                # 断言 m1 与 ma2 是否相等，期望结果是 eq_expected
                assert_equal(m1 == ma2, eq_expected)
                # 断言 ma1 与 ma2 是否相等，期望结果是 eq_expected
                assert_equal(ma1 == ma2, eq_expected)
                # 按元素检查是否相等
                el_by_el = [m1[name] == m2[name] for name in dt.names]
                # 断言按元素比较的结果是否与 eq_expected 相等
                assert_equal(array(el_by_el, dtype=bool).all(), eq_expected)
                # 预期 r1 与 r2 的不等性
                ne_expected = (r1 != r2).any()
                # 断言 m1 与 m2 是否不相等，期望结果是 ne_expected
                assert_equal(m1 != m2, ne_expected)
                # 断言 m2 与 m1 是否不相等，期望结果是 ne_expected
                assert_equal(m2 != m1, ne_expected)
                # 断言 ma1 与 m2 是否不相等，期望结果是 ne_expected
                assert_equal(ma1 != m2, ne_expected)
                # 断言 m1 与 ma2 是否不相等，期望结果是 ne_expected
                assert_equal(m1 != ma2, ne_expected)
                # 断言 ma1 与 ma2 是否不相等，期望结果是 ne_expected
                assert_equal(ma1 != ma2, ne_expected)
                # 按元素检查是否不相等
                el_by_el = [m1[name] != m2[name] for name in dt.names]
                # 断言按元素比较的结果是否与 ne_expected 相等
                assert_equal(array(el_by_el, dtype=bool).any(), ne_expected)

    @pytest.mark.parametrize('dt', ['S', 'U'])
    @pytest.mark.parametrize('fill', [None, 'A'])
    def test_eq_for_strings(self, dt, fill):
        # 测试结构化数组的相等性
        # 创建一个包含字符串的数组 a，指定数据类型和填充值
        a = array(['a', 'b'], dtype=dt, mask=[0, 1], fill_value=fill)

        # 测试 a 是否等于自身
        test = (a == a)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 测试 a 是否等于其第一个元素
        test = (a == a[0])
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 创建另一个结构化数组 b，指定数据类型和掩码
        b = array(['a', 'b'], dtype=dt, mask=[1, 0], fill_value=fill)

        # 测试 a 是否等于 b
        test = (a == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        # 测试 a 的第一个元素是否等于 b
        test = (a[0] == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        # 测试 b 的第一个元素是否等于 a 的第一个元素
        test = (b == a[0])
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)
    # 使用 pytest 的参数化标记，测试字符串类型的结构化数组的不等式
    @pytest.mark.parametrize('dt', ['S', 'U'])
    @pytest.mark.parametrize('fill', [None, 'A'])
    def test_ne_for_strings(self, dt, fill):
        # 创建一个结构化数组 a，包含字符串 'a' 和 'b'
        a = array(['a', 'b'], dtype=dt, mask=[0, 1], fill_value=fill)

        # 测试 a 与自身的不等性
        test = (a != a)
        assert_equal(test.data, [False, False])  # 断言数据部分是否正确
        assert_equal(test.mask, [False, True])   # 断言掩码部分是否正确
        assert_(test.fill_value == True)         # 断言填充值是否为 True

        # 测试 a 与其第一个元素的不等性
        test = (a != a[0])
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 创建另一个结构化数组 b，与 a 的数据类型相同，但掩码不同
        b = array(['a', 'b'], dtype=dt, mask=[1, 0], fill_value=fill)

        # 测试 a 与 b 的不等性
        test = (a != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        # 测试 a 的第一个元素与 b 的不等性
        test = (a[0] != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        # 测试 b 与 a 的第一个元素的不等性
        test = (b != a[0])
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    # 使用 pytest 的参数化标记，测试数值类型的结构化数组的等式
    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    def test_eq_for_numeric(self, dt1, dt2, fill):
        # 创建一个结构化数组 a，包含数字 0 和 1
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)

        # 测试 a 与自身的等性
        test = (a == a)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 测试 a 与其第一个元素的等性
        test = (a == a[0])
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 创建另一个结构化数组 b，与 a 的数据类型可能不同，但掩码可能不同
        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)

        # 测试 a 与 b 的等性
        test = (a == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        # 测试 a 的第一个元素与 b 的等性
        test = (a[0] == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        # 测试 b 与 a 的第一个元素的等性
        test = (b == a[0])
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    # 使用 pytest 的参数化标记和运算符，测试带有未掩码的结构化数组的广播等式
    @pytest.mark.parametrize("op", [operator.eq, operator.lt])
    def test_eq_broadcast_with_unmasked(self, op):
        # 创建一个未掩码的结构化数组 a
        a = array([0, 1], mask=[0, 1])
        # 创建一个形状为 (5, 2) 的数组 b，其中的内容是 0 到 9 的数字
        b = np.arange(10).reshape(5, 2)
        # 使用给定的操作符 op 进行 a 和 b 的比较操作
        result = op(a, b)
        # 断言结果的掩码形状与 b 的形状相同
        assert_(result.mask.shape == b.shape)
        # 断言结果的掩码与 a 的掩码或运算后是否一致
        assert_equal(result.mask, np.zeros(b.shape, bool) | a.mask)

    # 使用 pytest 的参数化标记和运算符，测试带有广播操作的等式
    @pytest.mark.parametrize("op", [operator.eq, operator.gt])
    def test_comp_no_mask_not_broadcast(self, op):
        # 用于回归测试 MaskedArray.nonzero 的失败 doctest
        # 在 gh-24556 之后出现的问题。
        # 创建一个普通的二维数组 a
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # 使用给定的操作符 op 对数组 a 和标量 3 进行比较
        result = op(a, 3)
        # 断言 result 的 mask 属性为空形状
        assert_(not result.mask.shape)
        # 断言 result 的 mask 属性为 nomask
        assert_(result.mask is nomask)

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    def test_ne_for_numeric(self, dt1, dt2, fill):
        # 测试结构化数组的不等性操作
        # 创建一个结构化数组 a，包括数据、掩码和填充值
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)

        # 测试 a 与自身的不等性
        test = (a != a)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 测试 a 与其第一个元素的不等性
        test = (a != a[0])
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 创建另一个结构化数组 b，包括数据、掩码和填充值
        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)
        # 测试 a 与 b 的不等性
        test = (a != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        # 测试 a[0] 与 b 的不等性
        test = (a[0] != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        # 测试 b 与 a[0] 的不等性
        test = (b != a[0])
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    @pytest.mark.parametrize('op',
            [operator.le, operator.lt, operator.ge, operator.gt])
    def test_comparisons_for_numeric(self, op, dt1, dt2, fill):
        # 测试结构化数组的比较操作（小于等于、小于、大于等于、大于）
        # 创建一个结构化数组 a，包括数据、掩码和填充值
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)

        # 使用指定的操作符 op 对 a 和 a 进行比较
        test = op(a, a)
        assert_equal(test.data, op(a._data, a._data))
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 使用指定的操作符 op 对 a 和 a 的第一个元素进行比较
        test = op(a, a[0])
        assert_equal(test.data, op(a._data, a._data[0]))
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        # 创建另一个结构化数组 b，包括数据、掩码和填充值
        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)
        # 使用指定的操作符 op 对 a 和 b 进行比较
        test = op(a, b)
        assert_equal(test.data, op(a._data, b._data))
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        # 使用指定的操作符 op 对 a 的第一个元素和 b 进行比较
        test = op(a[0], b)
        assert_equal(test.data, op(a._data[0], b._data))
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        # 使用指定的操作符 op 对 b 和 a 的第一个元素进行比较
        test = op(b, a[0])
        assert_equal(test.data, op(b._data, a._data[0]))
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('op',
            [operator.le, operator.lt, operator.ge, operator.gt])
    @pytest.mark.parametrize('fill', [None, "N/A"])
    # 使用 pytest 的 parametrize 装饰器，为测试方法 test_comparisons_strings 参数 fill 注入多组测试数据
    def test_comparisons_strings(self, op, fill):
        # 见 gh-21770，字符串（以及其他一些情况）的掩码传播存在问题，因此在这里显式测试字符串。
        # 原则上只有 == 和 != 可能需要特殊处理...
        
        # 创建一个具有部分掩码的 masked_array 对象 ma1 和 ma2，分别使用不同的字符串和掩码创建
        ma1 = masked_array(["a", "b", "cde"], mask=[0, 1, 0], fill_value=fill)
        ma2 = masked_array(["cde", "b", "a"], mask=[0, 1, 0], fill_value=fill)
        
        # 断言操作 op 应用于 ma1 和 ma2 的结果与应用于它们数据部分的结果相等
        assert_equal(op(ma1, ma2)._data, op(ma1._data, ma2._data))

    def test_eq_with_None(self):
        # 实际上，不应该与 None 进行比较，但仍然检查它们。请注意，pep8 将标记这些测试。
        # 对于数组已经有弃用警告，当弃用实施时，这些测试将失败（并且必须相应修改）。

        # 在部分掩码情况下进行测试
        with suppress_warnings() as sup:
            sup.filter(FutureWarning, "Comparison to `None`")
            # 创建一个具有 None 元素和掩码的 array 对象 a
            a = array([None, 1], mask=[0, 1])
            # 断言 a 与 None 的比较结果为 array([True, False])，掩码保持不变
            assert_equal(a == None, array([True, False], mask=[0, 1]))
            # 断言 a 的数据部分与 None 的比较结果为 [True, False]
            assert_equal(a.data == None, [True, False])
            # 断言 a 与 None 的不等比较结果为 array([False, True])，掩码保持不变
            assert_equal(a != None, array([False, True], mask=[0, 1]))
            
            # 在无掩码情况下创建 array 对象 a
            a = array([None, 1], mask=False)
            # 断言 a 与 None 的比较结果为 [True, False]
            assert_equal(a == None, [True, False])
            # 断言 a 与 None 的不等比较结果为 [False, True]
            assert_equal(a != None, [False, True])
            
            # 在完全掩码情况下创建 array 对象 a
            a = array([None, 2], mask=True)
            # 断言 a 与 None 的比较结果为 array([False, True], mask=True)
            assert_equal(a == None, array([False, True], mask=True))
            # 断言 a 与 None 的不等比较结果为 array([True, False], mask=True)
            assert_equal(a != None, array([True, False], mask=True))
            
            # 完全掩码的情况下，即使与 None 比较也应返回 "masked"
            a = masked
            # 断言 a 与 None 的比较结果为 masked
            assert_equal(a == None, masked)

    def test_eq_with_scalar(self):
        # 创建一个标量值为 1 的 array 对象 a
        a = array(1)
        # 断言 a 与 1 的比较结果为 True
        assert_equal(a == 1, True)
        # 断言 a 与 0 的比较结果为 False
        assert_equal(a == 0, False)
        # 断言 a 与 1 的不等比较结果为 False
        assert_equal(a != 1, False)
        # 断言 a 与 0 的不等比较结果为 True
        assert_equal(a != 0, True)
        
        # 创建一个标量值为 1 的 array 对象 b，使用掩码
        b = array(1, mask=True)
        # 断言 b 与 0 的比较结果为 masked
        assert_equal(b == 0, masked)
        # 断言 b 与 1 的比较结果为 masked
        assert_equal(b == 1, masked)
        # 断言 b 与 0 的不等比较结果为 masked
        assert_equal(b != 0, masked)
        # 断言 b 与 1 的不等比较结果为 masked
        assert_equal(b != 1, masked)

    def test_eq_different_dimensions(self):
        # 创建一个包含掩码的 array 对象 m1
        m1 = array([1, 1], mask=[0, 1])
        # 针对不同维度的数组 m2 进行比较测试
        for m2 in (array([[0, 1], [1, 2]]),
                   np.array([[0, 1], [1, 2]])):
            # 测试 m1 与 m2 的比较结果，应保持数据部分的 True 和 False，掩码部分保持一致
            test = (m1 == m2)
            assert_equal(test.data, [[False, False],
                                     [True, False]])
            assert_equal(test.mask, [[False, True],
                                     [False, True]])
    def test_numpyarithmetic(self):
        # 定义一个测试函数，用于测试 numpy 的算术运算和掩码操作的行为

        # 创建一个带掩码的数组 a，其中包含数据 [-1, 0, 1, 2, 3] 和掩码 [0, 0, 0, 0, 1]
        a = masked_array([-1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1])

        # 创建一个控制数组 control，包含数据 [NaN, NaN, 0, ln(2), -1] 和掩码 [1, 1, 0, 0, 1]
        control = masked_array([np.nan, np.nan, 0, np.log(2), -1],
                               mask=[1, 1, 0, 0, 1])

        # 使用 numpy 的 log 函数计算数组 a 的对数，存储结果到 test 变量
        test = log(a)

        # 断言 test 的值等于 control 的值
        assert_equal(test, control)

        # 断言 test 的掩码（mask）与 control 的掩码相同
        assert_equal(test.mask, control.mask)

        # 断言数组 a 的掩码保持不变，仍为 [0, 0, 0, 0, 1]
        assert_equal(a.mask, [0, 0, 0, 0, 1])

        # 使用 numpy 的 np.log 函数计算数组 a 的对数，存储结果到 test 变量
        test = np.log(a)

        # 断言 test 的值等于 control 的值
        assert_equal(test, control)

        # 断言 test 的掩码（mask）与 control 的掩码相同
        assert_equal(test.mask, control.mask)

        # 断言数组 a 的掩码保持不变，仍为 [0, 0, 0, 0, 1]
        assert_equal(a.mask, [0, 0, 0, 0, 1])
    # 定义一个测试类 TestMaskedArrayAttributes，用于测试掩码数组的属性和方法
    class TestMaskedArrayAttributes:

        # 测试 keep_mask 参数的功能
        def test_keepmask(self):
            # 创建一个掩码数组 x，其中第一个元素被掩盖
            x = masked_array([1, 2, 3], mask=[1, 0, 0])
            # 使用 x 创建一个新的掩码数组 mx
            mx = masked_array(x)
            # 断言 mx 的掩码与 x 的掩码相同
            assert_equal(mx.mask, x.mask)
            # 创建一个新的掩码数组 mx，指定新的掩码和 keep_mask=False
            mx = masked_array(x, mask=[0, 1, 0], keep_mask=False)
            # 断言 mx 的掩码是指定的新掩码
            assert_equal(mx.mask, [0, 1, 0])
            # 创建一个新的掩码数组 mx，指定新的掩码和 keep_mask=True
            mx = masked_array(x, mask=[0, 1, 0], keep_mask=True)
            # 断言 mx 的掩码是指定的新掩码，并且保持了 x 的原始掩码
            assert_equal(mx.mask, [1, 1, 0])
            # 默认情况下 keep_mask=True
            mx = masked_array(x, mask=[0, 1, 0])
            # 断言 mx 的掩码是指定的新掩码，并且保持了 x 的原始掩码
            assert_equal(mx.mask, [1, 1, 0])

        # 测试 hard_mask 参数的功能
        def test_hardmask(self):
            # 创建一个普通数组 d
            d = arange(5)
            # 创建一个掩码 n
            n = [0, 0, 0, 1, 1]
            # 根据掩码 n 创建掩码 m
            m = make_mask(n)
            # 创建一个带有硬掩码的数组 xh
            xh = array(d, mask=m, hard_mask=True)
            # 创建一个不带硬掩码的数组 xs，并进行复制以避免更新原始数组 d
            xs = array(d, mask=m, hard_mask=False, copy=True)
            # 修改 xh 和 xs 的部分元素值
            xh[[1, 4]] = [10, 40]
            xs[[1, 4]] = [10, 40]
            # 断言 xh 和 xs 的数据部分被正确修改
            assert_equal(xh._data, [0, 10, 2, 3, 4])
            assert_equal(xs._data, [0, 10, 2, 3, 40])
            # 断言 xs 的掩码部分被正确修改
            assert_equal(xs.mask, [0, 0, 0, 1, 0])
            # 断言 xh 使用了硬掩码
            assert_(xh._hardmask)
            # 断言 xs 没有使用硬掩码
            assert_(not xs._hardmask)
            # 修改 xh 和 xs 的部分元素值
            xh[1:4] = [10, 20, 30]
            xs[1:4] = [10, 20, 30]
            # 断言 xh 和 xs 的数据部分被正确修改
            assert_equal(xh._data, [0, 10, 20, 3, 4])
            assert_equal(xs._data, [0, 10, 20, 30, 40])
            # 断言 xs 的掩码为无掩码状态
            assert_equal(xs.mask, nomask)
            # 将 xh 和 xs 的第一个元素设置为掩码
            xh[0] = masked
            xs[0] = masked
            # 断言 xh 和 xs 的掩码部分被正确修改
            assert_equal(xh.mask, [1, 0, 0, 1, 1])
            assert_equal(xs.mask, [1, 0, 0, 0, 0])
            # 将 xh 和 xs 所有元素设置为相同的值
            xh[:] = 1
            xs[:] = 1
            # 断言 xh 和 xs 的数据部分被正确修改
            assert_equal(xh._data, [0, 1, 1, 3, 4])
            assert_equal(xs._data, [1, 1, 1, 1, 1])
            # 断言 xh 和 xs 的掩码部分被正确修改
            assert_equal(xh.mask, [1, 0, 0, 1, 1])
            assert_equal(xs.mask, nomask)
            # 将 xh 的掩码由硬掩码切换为软掩码
            xh.soften_mask()
            # 修改 xh 的所有元素为 0 到 4
            xh[:] = arange(5)
            # 断言 xh 的数据部分被正确修改
            assert_equal(xh._data, [0, 1, 2, 3, 4])
            # 断言 xh 的掩码为无掩码状态
            assert_equal(xh.mask, nomask)
            # 将 xh 的掩码由软掩码切换为硬掩码
            xh.harden_mask()
            # 将 xh 小于 3 的元素设置为掩码
            xh[xh < 3] = masked
            # 断言 xh 的数据部分被正确修改
            assert_equal(xh._data, [0, 1, 2, 3, 4])
            # 断言 xh 的掩码部分被正确修改
            assert_equal(xh._mask, [1, 1, 1, 0, 0])
            # 将 xh 大于 1 的元素设置为 5
            xh[filled(xh > 1, False)] = 5
            # 断言 xh 的数据部分被正确修改
            assert_equal(xh._data, [0, 1, 2, 5, 5])
            # 断言 xh 的掩码部分被正确修改
            assert_equal(xh._mask, [1, 1, 1, 0, 0])

            # 创建一个二维数组 xh，指定硬掩码
            xh = array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]], hard_mask=True)
            # 修改 xh 的第一行为 [0, 0]
            xh[0] = 0
            # 断言 xh 的数据部分被正确修改
            assert_equal(xh._data, [[1, 0], [3, 4]])
            # 断言 xh 的掩码部分被正确修改
            assert_equal(xh._mask, [[1, 0], [0, 0]])
            # 修改 xh 的最后一个元素为 5
            xh[-1, -1] = 5
            # 断言 xh 的数据部分被正确修改
            assert_equal(xh._data, [[1, 0], [3, 5]])
            # 断言 xh 的掩码部分被正确修改
            assert_equal(xh._mask, [[1, 0], [0, 0]])
            # 将 xh 中小于 5 的元素设置为 2
            xh[filled(xh < 5, False)] = 2
            # 断言 xh 的数据部分被正确修改
            assert_equal(xh._data, [[1, 2], [2, 5]])
            # 断言 xh 的掩码部分被正确修改
            assert_equal(xh._mask, [[1, 0], [0, 0]])

        # 另一个测试硬掩码的功能
        def test_hardmask_again(self):
            # 创建一个普通数组 d
            d = arange(5)
            # 创建一个掩码 n
            n = [0, 0, 0, 1, 1]
            # 根据掩码 n 创建掩码 m
            m = make_mask(n)
            # 创建一个带有硬掩码的数组 xh
            xh = array(d, mask=m, hard_mask=True)
            # 将 xh 的最后两个元素设置为 999
            xh[4:5] = 999
            # 将 xh 的第
    def test_hardmask_oncemore_yay(self):
        # OK, yet another test of hardmask
        # Make sure that harden_mask/soften_mask//unshare_mask returns self

        # 创建一个带有掩码的数组，第一个元素被掩盖
        a = array([1, 2, 3], mask=[1, 0, 0])

        # 测试 harden_mask 方法，确保返回自身
        b = a.harden_mask()
        assert_equal(a, b)

        # 修改 b 的第一个元素，验证 a 和 b 仍然相等
        b[0] = 0
        assert_equal(a, b)

        # 验证 b 是否与特定掩码的数组相等
        assert_equal(b, array([1, 2, 3], mask=[1, 0, 0]))

        # 使用 soften_mask 方法创建新数组 a，修改第一个元素
        a = b.soften_mask()
        a[0] = 0

        # 验证 soften_mask 方法修改了掩码，且 a 与 b 相等
        assert_equal(a, b)

        # 验证 b 是否与新的特定掩码的数组相等
        assert_equal(b, array([0, 2, 3], mask=[0, 0, 0]))

    def test_smallmask(self):
        # Checks the behaviour of _smallmask

        # 创建一个简单的数组 a
        a = arange(10)

        # 将第一个元素设为 masked
        a[1] = masked

        # 将第一个元素设为 1，验证掩码是否为 nomask
        a[1] = 1
        assert_equal(a._mask, nomask)

        # 创建另一个简单的数组 a
        a = arange(10)

        # 关闭 _smallmask，将第一个元素设为 masked
        a._smallmask = False
        a[1] = masked

        # 将第一个元素设为 1，验证掩码是否为全零数组
        a[1] = 1
        assert_equal(a._mask, zeros(10))

    def test_shrink_mask(self):
        # Tests .shrink_mask()

        # 创建一个数组 a，没有掩码
        a = array([1, 2, 3], mask=[0, 0, 0])

        # 调用 shrink_mask 方法，验证 a 是否等于 b，且掩码为 nomask
        b = a.shrink_mask()
        assert_equal(a, b)
        assert_equal(a.mask, nomask)

        # 创建一个结构化数组 a，带有掩码的记录
        a = np.ma.array([(1, 2.0)], [('a', int), ('b', float)])

        # 复制数组 a 到 b，调用 shrink_mask 方法，验证掩码是否不变
        b = a.copy()
        a.shrink_mask()
        assert_equal(a.mask, b.mask)

    def test_flat(self):
        # Test that flat can return all types of items [#4585, #4615]
        # test 2-D record array
        # ... on structured array w/ masked records

        # 创建一个二维结构化数组 x，包含掩码的记录
        x = array([[(1, 1.1, 'one'), (2, 2.2, 'two'), (3, 3.3, 'thr')],
                   [(4, 4.4, 'fou'), (5, 5.5, 'fiv'), (6, 6.6, 'six')]],
                  dtype=[('a', int), ('b', float), ('c', '|S8')])

        # 设置某些记录的特定元素为 masked
        x['a'][0, 1] = masked
        x['b'][1, 0] = masked
        x['c'][0, 2] = masked
        x[-1, -1] = masked

        # 获取数组 x 的 flat 属性
        xflat = x.flat

        # 验证 flat 是否按预期返回元素
        assert_equal(xflat[0], x[0, 0])
        assert_equal(xflat[1], x[0, 1])
        assert_equal(xflat[2], x[0, 2])
        assert_equal(xflat[:3], x[0])
        assert_equal(xflat[3], x[1, 0])
        assert_equal(xflat[4], x[1, 1])
        assert_equal(xflat[5], x[1, 2])
        assert_equal(xflat[3:], x[1])
        assert_equal(xflat[-1], x[-1, -1])

        # 通过迭代验证 flat 是否正确返回所有元素
        i = 0
        j = 0
        for xf in xflat:
            assert_equal(xf, x[j, i])
            i += 1
            if i >= x.shape[-1]:
                i = 0
                j += 1
    def test_assign_dtype(self):
        # 检查当 dtype 被改变时，掩码的 dtype 是否被更新
        a = np.zeros(4, dtype='f4,i4')

        # 创建一个掩码数组，基于数组 a
        m = np.ma.array(a)
        # 将掩码数组 m 的 dtype 设置为 'f4'
        m.dtype = np.dtype('f4')
        # 返回 m 的字符串表示形式
        repr(m)  # raises?
        # 断言 m 的 dtype 为 'f4'
        assert_equal(m.dtype, np.dtype('f4'))

        # 检查不允许修改会导致掩码形状发生过大变化的 dtype
        def assign():
            m = np.ma.array(a)
            m.dtype = np.dtype('f8')
        # 断言调用 assign 函数会引发 ValueError 异常
        assert_raises(ValueError, assign)

        # 创建一个基于数组 a 视图的 MaskedArray，dtype 设置为 'f4'
        b = a.view(dtype='f4', type=np.ma.MaskedArray)  # raises?
        # 断言 b 的 dtype 为 'f4'
        assert_equal(b.dtype, np.dtype('f4'))

        # 检查 nomask 是否被保留
        a = np.zeros(4, dtype='f4')
        m = np.ma.array(a)
        # 将掩码数组 m 的 dtype 设置为 'f4,i4'
        m.dtype = np.dtype('f4,i4')
        # 断言 m 的 dtype 为 'f4,i4'
        assert_equal(m.dtype, np.dtype('f4,i4'))
        # 断言 m 的掩码 _mask 为 np.ma.nomask
        assert_equal(m._mask, np.ma.nomask)
class TestFillingValues:
    # 定义一个测试类 TestFillingValues

    def test_check_on_scalar(self):
        # 定义测试方法 test_check_on_scalar，用于测试 _check_fill_value 函数的行为

        # 将 np.ma.core._check_fill_value 赋值给局部变量 _check_fill_value
        _check_fill_value = np.ma.core._check_fill_value

        # 测试当 _check_fill_value 函数以整数类型调用时，填充值应为 0
        fval = _check_fill_value(0, int)
        assert_equal(fval, 0)

        # 测试当 _check_fill_value 函数以 None 和整数类型调用时，填充值应为默认的整数填充值
        fval = _check_fill_value(None, int)
        assert_equal(fval, default_fill_value(0))

        # 测试当 _check_fill_value 函数以整数类型和字节字符串 "|S3" 调用时，填充值应为 b"0"
        fval = _check_fill_value(0, "|S3")
        assert_equal(fval, b"0")

        # 测试当 _check_fill_value 函数以 None 和字节字符串 "|S3" 调用时，填充值应为默认的字节字符串填充值
        fval = _check_fill_value(None, "|S3")
        assert_equal(fval, default_fill_value(b"camelot!"))

        # 测试当 _check_fill_value 函数以超出整数范围的值和整数类型调用时，应引发 TypeError 异常
        assert_raises(TypeError, _check_fill_value, 1e+20, int)

        # 测试当 _check_fill_value 函数以字符串和整数类型调用时，应引发 TypeError 异常
        assert_raises(TypeError, _check_fill_value, 'stuff', int)
    # 定义一个测试方法，用于检查 _check_fill_value 函数在记录上的表现
    def test_check_on_fields(self):
        # 获取 _check_fill_value 函数的引用
        _check_fill_value = np.ma.core._check_fill_value
        # 定义一个结构化数据类型
        ndtype = [('a', int), ('b', float), ('c', "|S3")]
        # 对列表进行检查，应返回一个单一记录数组
        fval = _check_fill_value([-999, -12345678.9, "???"], ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b"???"])
        
        # 对 None 进行检查，应返回默认值
        fval = _check_fill_value(None, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [default_fill_value(0),
                                   default_fill_value(0.),
                                   asbytes(default_fill_value("0"))])
        
        # 使用结构化类型作为填充值应当有效
        fill_val = np.array((-999, -12345678.9, "???"), dtype=ndtype)
        fval = _check_fill_value(fill_val, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b"???"])

        # 使用具有不同类型的灵活类型作为填充值也应当有效
        # 在 1.5 之前和 1.13 之后的行为中，按位置匹配结构化类型
        fill_val = np.array((-999, -12345678.9, "???"),
                            dtype=[("A", int), ("B", float), ("C", "|S3")])
        fval = _check_fill_value(fill_val, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b"???"])

        # 使用对象数组作为填充值也应当有效
        fill_val = np.ndarray(shape=(1,), dtype=object)
        fill_val[0] = (-999, -12345678.9, b"???")
        fval = _check_fill_value(fill_val, object)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b"???"])
        
        # 注意：这个测试没有正确运行，因为"fill_value"而不是"fill_val"被赋值。
        # 好的写法应该导致测试失败。
        
        # 使用仅有一个字段的灵活类型也应当有效
        ndtype = [("a", int)]
        fval = _check_fill_value(-999999999, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), (-999999999,))
    def test_fillvalue_conversion(self):
        # Tests the behavior of fill_value during conversion
        # 填充值在转换过程中的行为测试
        # We had a tailored comment to make sure special attributes are
        # properly dealt with
        # 我们加入了定制的注释以确保特殊属性被正确处理
        # 创建一个数组 `a`，包含字节字符串 '3', '4', '5'
        a = array([b'3', b'4', b'5'])
        # 更新数组 `a` 的特殊属性 '_optinfo' 的注释字段为 "updated!"
        a._optinfo.update({'comment':"updated!"})

        # 通过整型类型创建数组 `b`，并验证其数据与预期一致
        b = array(a, dtype=int)
        assert_equal(b._data, [3, 4, 5])
        # 验证 `b` 的填充值与默认填充值函数返回的值一致
        assert_equal(b.fill_value, default_fill_value(0))

        # 通过浮点型类型创建数组 `b`，并验证其数据与预期一致
        b = array(a, dtype=float)
        assert_equal(b._data, [3, 4, 5])
        # 验证 `b` 的填充值与默认填充值函数返回的值一致
        assert_equal(b.fill_value, default_fill_value(0.))

        # 将数组 `a` 转换为整型，创建数组 `b`，并验证其数据与预期一致
        b = a.astype(int)
        assert_equal(b._data, [3, 4, 5])
        # 验证 `b` 的填充值与默认填充值函数返回的值一致
        assert_equal(b.fill_value, default_fill_value(0))
        # 验证 `b` 的特殊属性 '_optinfo' 的注释字段为 "updated!"
        assert_equal(b._optinfo['comment'], "updated!")

        # 将数组 `a` 转换为结构化数组，包含一个名为 'a' 的字段
        b = a.astype([('a', '|S3')])
        # 验证结构化数组字段 'a' 的数据与原数组 `a` 的数据一致
        assert_equal(b['a']._data, a._data)
        # 验证结构化数组字段 'a' 的填充值与原数组 `a` 的填充值一致
        assert_equal(b['a'].fill_value, a.fill_value)

    def test_default_fill_value(self):
        # check all calling conventions
        # 检查所有的调用约定
        # 调用默认填充值函数，返回浮点数类型的填充值 `f1`
        f1 = default_fill_value(1.)
        # 调用默认填充值函数，传入 `np.array(1.)`，返回填充值 `f2`
        f2 = default_fill_value(np.array(1.))
        # 调用默认填充值函数，传入 `np.array(1.).dtype`，返回填充值 `f3`
        f3 = default_fill_value(np.array(1.).dtype)
        # 验证三个填充值 `f1`, `f2`, `f3` 相等
        assert_equal(f1, f2)
        assert_equal(f1, f3)

    def test_default_fill_value_structured(self):
        # 创建一个包含结构的数组 `fields`
        fields = array([(1, 1, 1)],
                      dtype=[('i', int), ('s', '|S8'), ('f', float)])

        # 调用默认填充值函数，传入结构化数组 `fields`，返回填充值 `f1`
        f1 = default_fill_value(fields)
        # 调用默认填充值函数，传入结构化数组 `fields.dtype`，返回填充值 `f2`
        f2 = default_fill_value(fields.dtype)
        # 创建预期的结构化数组 `expected`
        expected = np.array((default_fill_value(0),
                             default_fill_value('0'),
                             default_fill_value(0.)), dtype=fields.dtype)
        # 验证填充值 `f1` 与预期结构化数组 `expected` 相等
        assert_equal(f1, expected)
        # 验证填充值 `f2` 与预期结构化数组 `expected` 相等
        assert_equal(f2, expected)

    def test_default_fill_value_void(self):
        # 创建一个虚类型的数据类型 `dt`
        dt = np.dtype([('v', 'V7')])
        # 调用默认填充值函数，传入数据类型 `dt`，返回填充值 `f`
        f = default_fill_value(dt)
        # 验证填充值 `f` 的 'v' 字段与 `dt['v']` 的默认填充值相等
        assert_equal(f['v'], np.array(default_fill_value(dt['v']), dt['v']))

    def test_fillvalue(self):
        # Yet more fun with the fill_value
        # 填充值的更多有趣用法
        # 创建一个带填充值的遮蔽数组 `data`
        data = masked_array([1, 2, 3], fill_value=-999)
        # 创建 `series`，包含 `data` 的部分索引，并验证其填充值与 `data` 的填充值相等
        series = data[[0, 2, 1]]
        assert_equal(series._fill_value, data._fill_value)

        # 创建一个结构化数组 `x`，包含整数和字符串字段
        mtype = [('f', float), ('s', '|S3')]
        x = array([(1, 'a'), (2, 'b'), (pi, 'pi')], dtype=mtype)
        # 设置数组 `x` 的填充值为 999，并验证各字段的填充值与预期一致
        x.fill_value = 999
        assert_equal(x.fill_value.item(), [999., b'999'])
        assert_equal(x['f'].fill_value, 999)
        assert_equal(x['s'].fill_value, b'999')

        # 再次设置数组 `x` 的填充值为 (9, '???')，并验证各字段的填充值与预期一致
        x.fill_value = (9, '???')
        assert_equal(x.fill_value.item(), (9, b'???'))
        assert_equal(x['f'].fill_value, 9)
        assert_equal(x['s'].fill_value, b'???')

        # 创建一个数组 `x`，包含整数和浮点数，并设置其填充值为 999，验证填充值及其数据类型
        x = array([1, 2, 3.1])
        x.fill_value = 999
        assert_equal(np.asarray(x.fill_value).dtype, float)
        assert_equal(x.fill_value, 999.)
        assert_equal(x._fill_value, np.array(999.))
    def test_subarray_fillvalue(self):
        # 测试子数组的填充值
        # 设置一个包含多个字段的数组
        fields = array([(1, 1, 1)],
                      dtype=[('i', int), ('s', '|S8'), ('f', float)])
        # 使用 suppress_warnings 上下文管理器，过滤掉未来警告
        with suppress_warnings() as sup:
            # 过滤掉未来警告 "Numpy has detected"
            sup.filter(FutureWarning, "Numpy has detected")
            # 提取字段 'i' 和 'f' 构成子数组
            subfields = fields[['i', 'f']]
            # 断言子数组的填充值是 (999999, 1.e+20)
            assert_equal(tuple(subfields.fill_value), (999999, 1.e+20))
            # 测试比较操作不会引发异常：
            subfields[1:] == subfields[:-1]

    def test_fillvalue_exotic_dtype(self):
        # 测试更多的奇异灵活数据类型
        _check_fill_value = np.ma.core._check_fill_value
        # 定义一个复杂的数据类型
        ndtype = [('i', int), ('s', '|S8'), ('f', float)]
        # 创建一个控制数组，使用默认填充值
        control = np.array((default_fill_value(0),
                            default_fill_value('0'),
                            default_fill_value(0.),),
                           dtype=ndtype)
        # 断言 _check_fill_value 函数返回的结果符合控制数组
        assert_equal(_check_fill_value(None, ndtype), control)
        # 形状不应该影响结果
        ndtype = [('f0', float, (2, 2))]
        control = np.array((default_fill_value(0.),),
                           dtype=[('f0', float)]).astype(ndtype)
        # 断言 _check_fill_value 函数返回的结果符合控制数组
        assert_equal(_check_fill_value(None, ndtype), control)
        control = np.array((0,), dtype=[('f0', float)]).astype(ndtype)
        # 断言 _check_fill_value 函数返回的结果符合控制数组
        assert_equal(_check_fill_value(0, ndtype), control)

        ndtype = np.dtype("int, (2,3)float, float")
        control = np.array((default_fill_value(0),
                            default_fill_value(0.),
                            default_fill_value(0.),),
                           dtype="int, float, float").astype(ndtype)
        # 断言 _check_fill_value 函数返回的结果符合控制数组
        test = _check_fill_value(None, ndtype)
        assert_equal(test, control)
        control = np.array((0, 0, 0), dtype="int, float, float").astype(ndtype)
        # 断言 _check_fill_value 函数返回的结果符合控制数组
        assert_equal(_check_fill_value(0, ndtype), control)
        # 但是在索引时，填充值应变为标量而不是元组
        # 参见问题 #6723
        M = masked_array(control)
        # 断言 M["f1"].fill_value 的维度为 0
        assert_equal(M["f1"].fill_value.ndim, 0)

    def test_fillvalue_datetime_timedelta(self):
        # 测试 datetime64 和 timedelta64 类型的默认填充值
        # 参见问题 #4476，原本会返回 '?'，可能在其他地方引发错误

        for timecode in ("as", "fs", "ps", "ns", "us", "ms", "s", "m",
                         "h", "D", "W", "M", "Y"):
            # 创建控制值
            control = numpy.datetime64("NaT", timecode)
            # 调用 default_fill_value 函数测试结果
            test = default_fill_value(numpy.dtype("<M8[" + timecode + "]"))
            # 使用 np.testing.assert_equal 断言测试结果与控制值相等
            np.testing.assert_equal(test, control)

            control = numpy.timedelta64("NaT", timecode)
            # 调用 default_fill_value 函数测试结果
            test = default_fill_value(numpy.dtype("<m8[" + timecode + "]"))
            # 使用 np.testing.assert_equal 断言测试结果与控制值相等
            np.testing.assert_equal(test, control)
    def test_extremum_fill_value(self):
        # Tests extremum fill values for flexible type.
        # 创建一个结构化数组 `a`，包含整数和嵌套元组的结构
        a = array([(1, (2, 3)), (4, (5, 6))],
                  dtype=[('A', int), ('B', [('BA', int), ('BB', int)])])
        # 获取 `a` 的填充值
        test = a.fill_value
        # 断言填充值的数据类型与 `a` 的数据类型相同
        assert_equal(test.dtype, a.dtype)
        # 断言填充值的字段 'A' 等于字段 'A' 的默认填充值
        assert_equal(test['A'], default_fill_value(a['A']))
        # 断言填充值的字段 'B' 的子字段 'BA' 等于字段 'B' 的子字段 'BA' 的默认填充值
        assert_equal(test['B']['BA'], default_fill_value(a['B']['BA']))
        # 断言填充值的字段 'B' 的子字段 'BB' 等于字段 'B' 的子字段 'BB' 的默认填充值
        assert_equal(test['B']['BB'], default_fill_value(a['B']['BB']))

        # 获取 `a` 的最小填充值
        test = minimum_fill_value(a)
        # 断言最小填充值的数据类型与 `a` 的数据类型相同
        assert_equal(test.dtype, a.dtype)
        # 断言最小填充值的第一个元素等于字段 'A' 的最小填充值
        assert_equal(test[0], minimum_fill_value(a['A']))
        # 断言最小填充值的第二个元素的第一个子字段等于字段 'B' 的子字段 'BA' 的最小填充值
        assert_equal(test[1][0], minimum_fill_value(a['B']['BA']))
        # 断言最小填充值的第二个元素的第二个子字段等于字段 'B' 的子字段 'BB' 的最小填充值
        assert_equal(test[1][1], minimum_fill_value(a['B']['BB']))
        # 断言最小填充值的第二个元素等于字段 'B' 的最小填充值
        assert_equal(test[1], minimum_fill_value(a['B']))

        # 获取 `a` 的最大填充值
        test = maximum_fill_value(a)
        # 断言最大填充值的数据类型与 `a` 的数据类型相同
        assert_equal(test.dtype, a.dtype)
        # 断言最大填充值的第一个元素等于字段 'A' 的最大填充值
        assert_equal(test[0], maximum_fill_value(a['A']))
        # 断言最大填充值的第二个元素的第一个子字段等于字段 'B' 的子字段 'BA' 的最大填充值
        assert_equal(test[1][0], maximum_fill_value(a['B']['BA']))
        # 断言最大填充值的第二个元素的第二个子字段等于字段 'B' 的子字段 'BB' 的最大填充值
        assert_equal(test[1][1], maximum_fill_value(a['B']['BB']))
        # 断言最大填充值的第二个元素等于字段 'B' 的最大填充值

    def test_extremum_fill_value_subdtype(self):
        # 创建一个具有子数据类型的结构化数组 `a`
        a = array(([2, 3, 4],), dtype=[('value', np.int8, 3)])

        # 获取 `a` 的最小填充值
        test = minimum_fill_value(a)
        # 断言最小填充值的数据类型与 `a` 的数据类型相同
        assert_equal(test.dtype, a.dtype)
        # 断言最小填充值的第一个元素等于 `a['value']` 的最小填充值
        assert_equal(test[0], np.full(3, minimum_fill_value(a['value'])))

        # 获取 `a` 的最大填充值
        test = maximum_fill_value(a)
        # 断言最大填充值的数据类型与 `a` 的数据类型相同
        assert_equal(test.dtype, a.dtype)
        # 断言最大填充值的第一个元素等于 `a['value']` 的最大填充值
        assert_equal(test[0], np.full(3, maximum_fill_value(a['value'])))

    def test_fillvalue_individual_fields(self):
        # 测试在各个字段上设置填充值
        ndtype = [('a', int), ('b', int)]
        # 使用显式填充值创建结构化数组 `a`
        a = array(list(zip([1, 2, 3], [4, 5, 6])),
                  fill_value=(-999, -999), dtype=ndtype)
        # 获取字段 'a' 的视图 `aa`
        aa = a['a']
        # 设置字段 'a' 的填充值为 10
        aa.set_fill_value(10)
        # 断言字段 'a' 的内部填充值等于数组 [10]
        assert_equal(aa._fill_value, np.array(10))
        # 断言结构化数组 `a` 的整体填充值为 (10, -999)
        assert_equal(tuple(a.fill_value), (10, -999))
        # 修改结构化数组 `a` 中字段 'b' 的填充值为 -10
        a.fill_value['b'] = -10
        # 断言结构化数组 `a` 的整体填充值为 (10, -10)
        assert_equal(tuple(a.fill_value), (10, -10))
        # 使用隐式填充值创建结构化数组 `t`
        t = array(list(zip([1, 2, 3], [4, 5, 6])), dtype=ndtype)
        # 获取字段 'a' 的视图 `tt`
        tt = t['a']
        # 设置字段 'a' 的填充值为 10
        tt.set_fill_value(10)
        # 断言字段 'a' 的内部填充值等于数组 [10]
        assert_equal(tt._fill_value, np.array(10))
        # 断言结构化数组 `t` 的整体填充值为 (10, 默认填充值(0))
        assert_equal(tuple(t.fill_value), (10, default_fill_value(0)))

    def test_fillvalue_implicit_structured_array(self):
        # 检查结构化数组总是定义填充值
        ndtype = ('b', float)
        adtype = ('a', float)
        # 创建具有 NaN 填充值的结构化数组 `a`
        a = array([(1.,), (2.,)], mask=[(False,), (False,)],
                  fill_value=(np.nan,), dtype=np.dtype([adtype]))
        # 创建结构相同的空结构化数组 `b`
        b = empty(a.shape, dtype=[adtype, ndtype])
        # 将 `a` 的字段 'a' 赋给 `b` 的字段 'a'
        b['a'] = a['a']
        # 将 `a` 的字段 'a' 的填充值设置给 `b` 的字段 'a'
        b['a'].set_fill_value(a['a'].fill_value)
        # 获取 `b` 的内部填充值
        f = b._fill_value[()]
        # 断言 `b` 的字段 'a' 的填充值是 NaN
        assert_(np.isnan(f[0]))
        # 断言 `b` 的字段 'b' 的填充值是默认填充值(1.)
        assert_equal(f[-1], default_fill_value(1.))
    def test_fillvalue_as_arguments(self):
        # 测试向 empty/ones/zeros 函数添加 fill_value 参数的效果

        # 使用 fill_value 参数创建长度为 3 的空数组 a，并验证其 fill_value 属性为 999.。
        a = empty(3, fill_value=999.)
        assert_equal(a.fill_value, 999.)

        # 使用 fill_value 和 dtype 参数创建长度为 3 的值为 1 的数组 a，并验证其 fill_value 属性为 999.。
        a = ones(3, fill_value=999., dtype=float)
        assert_equal(a.fill_value, 999.)

        # 使用 fill_value 和 dtype 参数创建长度为 3 的值为 0 的复数数组 a，并验证其 fill_value 属性为 0.。
        a = zeros(3, fill_value=0., dtype=complex)
        assert_equal(a.fill_value, 0.)

        # 使用 fill_value 和 dtype 参数创建单位矩阵的复数版本 a，并验证其 fill_value 属性为 0.。
        a = identity(3, fill_value=0., dtype=complex)
        assert_equal(a.fill_value, 0.)

    def test_shape_argument(self):
        # 测试 shape 参数的传入方式
        # GH 问题号 6106

        # 使用 shape 参数创建长度为 3 的空数组 a，并验证其 shape 属性为 (3,)。
        a = empty(shape=(3, ))
        assert_equal(a.shape, (3, ))

        # 使用 shape 和 dtype 参数创建长度为 3 的值为 1 的数组 a，并验证其 shape 属性为 (3,)。
        a = ones(shape=(3, ), dtype=float)
        assert_equal(a.shape, (3, ))

        # 使用 shape 和 dtype 参数创建长度为 3 的值为 0 的复数数组 a，并验证其 shape 属性为 (3,)。
        a = zeros(shape=(3, ), dtype=complex)
        assert_equal(a.shape, (3, ))

    def test_fillvalue_in_view(self):
        # 测试视图中 fill_value 的行为

        # 创建初始的带填充值为 1 的掩码数组 x
        x = array([1, 2, 3], fill_value=1, dtype=np.int64)

        # 检查默认情况下视图 y 会保留 fill_value 属性
        y = x.view()
        assert_(y.fill_value == 1)

        # 指定 dtype 为 MaskedArray，检查视图 y 会保留 fill_value 属性
        y = x.view(MaskedArray)
        assert_(y.fill_value == 1)

        # 使用 type=MaskedArray，检查视图 y 会保留 fill_value 属性
        y = x.view(type=MaskedArray)
        assert_(y.fill_value == 1)

        # 如果传入不带 _fill_value 属性的 ndarray 子类，确保代码不会崩溃
        y = x.view(np.ndarray)
        y = x.view(type=np.ndarray)

        # 使用 view 创建 MaskedArray 的视图，并覆盖 fill_value 为 2，验证 fill_value 属性为 2
        y = x.view(MaskedArray, fill_value=2)
        assert_(y.fill_value == 2)

        # 使用 view 创建 MaskedArray 的视图，使用 type=，覆盖 fill_value 为 2，验证 fill_value 属性为 2
        y = x.view(type=MaskedArray, fill_value=2)
        assert_(y.fill_value == 2)

        # 检查当传入 dtype 参数但未传入 fill_value 参数时，fill_value 是否会重置为默认值
        # 这是因为有些情况下可以安全地转换 fill_value，例如从 int32 数组视图到 int64，但在其他情况下则不行
        y = x.view(dtype=np.int32)
        assert_(y.fill_value == 999999)

    def test_fillvalue_bytes_or_str(self):
        # 测试结构化 dtype 包含 bytes 或 str 时 fill_value 的预期行为
        # 参见问题编号 #7259

        # 使用结构化 dtype 创建长度为 3 的数组 a，并验证其字段 "f0" 的 fill_value 属性符合预期
        a = empty(shape=(3, ), dtype="(2,)3S,(2,)3U")
        assert_equal(a["f0"].fill_value, default_fill_value(b"spam"))
        
        # 验证结构化 dtype 中字段 "f1" 的 fill_value 属性符合预期
        assert_equal(a["f1"].fill_value, default_fill_value("eggs"))
class TestUfuncs:
    # 对 MaskedArrays 应用 ufuncs 进行测试的测试类

    def setup_method(self):
        # 设置测试方法的基础数据定义
        self.d = (array([1.0, 0, -1, pi / 2] * 2, mask=[0, 1] + [0] * 6),
                  array([1.0, 0, -1, pi / 2] * 2, mask=[1, 0] + [0] * 6),)
        # 保存当前的错误状态并设置忽略除法和无效值错误
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def teardown_method(self):
        # 恢复之前保存的错误状态
        np.seterr(**self.err_status)

    def test_testUfuncRegression(self):
        # 测试 MaskedArrays 上的新 ufuncs
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
                  'sin', 'cos', 'tan',
                  'arcsin', 'arccos', 'arctan',
                  'sinh', 'cosh', 'tanh',
                  'arcsinh',
                  'arccosh',
                  'arctanh',
                  'absolute', 'fabs', 'negative',
                  'floor', 'ceil',
                  'logical_not',
                  'add', 'subtract', 'multiply',
                  'divide', 'true_divide', 'floor_divide',
                  'remainder', 'fmod', 'hypot', 'arctan2',
                  'equal', 'not_equal', 'less_equal', 'greater_equal',
                  'less', 'greater',
                  'logical_and', 'logical_or', 'logical_xor',
                  ]:
            try:
                # 尝试从 umath 模块获取 ufunc
                uf = getattr(umath, f)
            except AttributeError:
                # 如果在 umath 模块中找不到，则从 fromnumeric 模块获取
                uf = getattr(fromnumeric, f)
            # 获取对应的 MaskedArray 中的 ufunc
            mf = getattr(numpy.ma.core, f)
            # 准备参数
            args = self.d[:uf.nin]
            # 调用普通 ufunc 和 MaskedArray 中的对应 ufunc
            ur = uf(*args)
            mr = mf(*args)
            # 断言两者结果的填充后是否相等
            assert_equal(ur.filled(0), mr.filled(0), f)
            # 断言两者的掩码是否相等
            assert_mask_equal(ur.mask, mr.mask, err_msg=f)

    def test_reduce(self):
        # 测试 MaskedArrays 上的 reduce 操作
        a = self.d[0]
        # 断言在指定轴上不全为真
        assert_(not alltrue(a, axis=0))
        # 断言在指定轴上至少有一个为真
        assert_(sometrue(a, axis=0))
        # 断言对指定轴上的数据求和是否等于0
        assert_equal(sum(a[:3], axis=0), 0)
        # 断言对指定轴上的数据求积是否等于0
        assert_equal(product(a, axis=0), 0)
        # 断言对数组所有元素求和是否等于 pi
        assert_equal(add.reduce(a), pi)

    def test_minmax(self):
        # 测试 MaskedArrays 上的极值操作
        a = arange(1, 13).reshape(3, 4)
        amask = masked_where(a < 5, a)
        # 断言掩码后的最大值与原始数组的最大值相等
        assert_equal(amask.max(), a.max())
        # 断言掩码后的最小值与给定值相等
        assert_equal(amask.min(), 5)
        # 断言掩码后在指定轴上的最大值与原始数组相等
        assert_equal(amask.max(0), a.max(0))
        # 断言掩码后在指定轴上的最小值与给定数组相等
        assert_equal(amask.min(0), [5, 6, 7, 8])
        # 断言掩码后在指定轴上的第一项最大值的掩码为真
        assert_(amask.max(1)[0].mask)
        # 断言掩码后在指定轴上的第一项最小值的掩码为真
        assert_(amask.min(1)[0].mask)

    def test_ndarray_mask(self):
        # 检查结果的掩码是 ndarray 而不是 MaskedArray
        a = masked_array([-1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1])
        test = np.sqrt(a)
        control = masked_array([-1, 0, 1, np.sqrt(2), -1],
                               mask=[1, 0, 0, 0, 1])
        # 断言结果与控制组相等
        assert_equal(test, control)
        # 断言结果的掩码与控制组的掩码相等
        assert_equal(test.mask, control.mask)
        # 断言结果的掩码不是 MaskedArray 类型
        assert_(not isinstance(test.mask, MaskedArray))
    def test_treatment_of_NotImplemented(self):
        # 检查在适当的位置返回 NotImplemented

        # 创建一个带有掩码的数组
        a = masked_array([1., 2.], mask=[1, 0])

        # 确保使用非法类型时会引发 TypeError
        assert_raises(TypeError, operator.mul, a, "abc")
        assert_raises(TypeError, operator.truediv, a, "abc")

        # 定义一个带有特定 __array_priority__ 的类
        class MyClass:
            __array_priority__ = a.__array_priority__ + 1

            # 自定义乘法运算符
            def __mul__(self, other):
                return "My mul"

            # 自定义反向乘法运算符
            def __rmul__(self, other):
                return "My rmul"

        me = MyClass()

        # 验证自定义乘法运算符是否按预期工作
        assert_(me * a == "My mul")
        assert_(a * me == "My rmul")

        # 确保尊重 __array_priority__
        class MyClass2:
            __array_priority__ = 100

            # 自定义乘法运算符
            def __mul__(self, other):
                return "Me2mul"

            # 自定义反向乘法运算符
            def __rmul__(self, other):
                return "Me2rmul"

            # 自定义反向除法运算符
            def __rdiv__(self, other):
                return "Me2rdiv"

            __rtruediv__ = __rdiv__

        me_too = MyClass2()

        # 验证 NotImplemented 的返回
        assert_(a.__mul__(me_too) is NotImplemented)

        # 验证 multiply.outer 的结果
        assert_(all(multiply.outer(a, me_too) == "Me2rmul"))

        # 验证 truediv 的 NotImplemented
        assert_(a.__truediv__(me_too) is NotImplemented)

        # 验证自定义乘法和反向乘法运算符是否按预期工作
        assert_(me_too * a == "Me2mul")
        assert_(a * me_too == "Me2rmul")

        # 验证自定义除法运算符是否按预期工作
        assert_(a / me_too == "Me2rdiv")

    def test_no_masked_nan_warnings(self):
        # 检查掩码位置的 NaN 不会导致 ufunc 警告

        # 创建一个带有 NaN 和掩码的掩码数组
        m = np.ma.array([0.5, np.nan], mask=[0,1])

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            # 测试一元和二元 ufuncs
            exp(m)
            add(m, 1)
            m > 0

            # 测试不同的一元函数
            sqrt(m)
            log(m)
            tan(m)
            arcsin(m)
            arccos(m)
            arccosh(m)

            # 测试二元函数
            divide(m, 2)

            # 确保 allclose 使用 ma ufuncs，避免警告
            allclose(m, 0.5)

    def test_masked_array_underflow(self):
        # 检查掩码数组的下溢处理

        # 创建一个普通数组
        x = np.arange(0, 3, 0.1)

        # 创建一个带有掩码的掩码数组
        X = np.ma.array(x)

        with np.errstate(under="raise"):
            # 执行除法操作
            X2 = X/2.0

            # 使用 np.testing.assert_array_equal 确保结果符合预期
            np.testing.assert_array_equal(X2, x/2)
# 定义一个测试类，用于测试 MaskedArray 的原地算术运算
class TestMaskedArrayInPlaceArithmetic:

    # 设置每个测试方法的初始状态
    def setup_method(self):
        # 创建三个包含整数序列的数组
        x = arange(10)
        y = arange(10)
        xm = arange(10)
        # 将第三个数组的第二个元素标记为掩码（masked）
        xm[2] = masked
        # 分别存储整数数据、浮点数数据、其它类型数据和无符号8位整数数据
        self.intdata = (x, y, xm)
        self.floatdata = (x.astype(float), y.astype(float), xm.astype(float))
        self.othertypes = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        self.othertypes = [np.dtype(_).type for _ in self.othertypes]
        self.uint8data = (
            x.astype(np.uint8),
            y.astype(np.uint8),
            xm.astype(np.uint8)
        )

    # 测试原地加法操作
    def test_inplace_addition_scalar(self):
        # 获取整数数据和掩码数据
        (x, y, xm) = self.intdata
        # 将掩码数组的第二个元素标记为掩码
        xm[2] = masked
        # 对数组 x 和 xm 进行原地加法运算
        x += 1
        assert_equal(x, y + 1)
        xm += 1
        assert_equal(xm, y + 1)

        # 获取浮点数数据
        (x, _, xm) = self.floatdata
        # 记录初始数据的内存地址
        id1 = x.data.ctypes.data
        # 对 x 进行原地加法运算
        x += 1.
        # 断言内存地址未改变
        assert_(id1 == x.data.ctypes.data)
        assert_equal(x, y + 1.)

    # 测试原地数组加法操作
    def test_inplace_addition_array(self):
        # 获取整数数据和掩码数据
        (x, y, xm) = self.intdata
        # 获取掩码
        m = xm.mask
        # 创建另一个整数数组 a
        a = arange(10, dtype=np.int16)
        a[-1] = masked
        # 对数组 x 和 xm 进行原地加法运算
        x += a
        xm += a
        assert_equal(x, y + a)
        assert_equal(xm, y + a)
        # 断言掩码数组等于原掩码数组和新数组 a 的掩码的逻辑或
        assert_equal(xm.mask, mask_or(m, a.mask))

    # 测试原地减法操作
    def test_inplace_subtraction_scalar(self):
        # 获取整数数据和掩码数据
        (x, y, xm) = self.intdata
        # 对数组 x 和 xm 进行原地减法运算
        x -= 1
        assert_equal(x, y - 1)
        xm -= 1
        assert_equal(xm, y - 1)

    # 测试原地数组减法操作
    def test_inplace_subtraction_array(self):
        # 获取浮点数数据和掩码数据
        (x, y, xm) = self.floatdata
        # 获取掩码
        m = xm.mask
        # 创建另一个浮点数数组 a
        a = arange(10, dtype=float)
        a[-1] = masked
        # 对数组 x 和 xm 进行原地减法运算
        x -= a
        xm -= a
        assert_equal(x, y - a)
        assert_equal(xm, y - a)
        # 断言掩码数组等于原掩码数组和新数组 a 的掩码的逻辑或
        assert_equal(xm.mask, mask_or(m, a.mask))

    # 测试原地乘法操作
    def test_inplace_multiplication_scalar(self):
        # 获取浮点数数据
        (x, y, xm) = self.floatdata
        # 对数组 x 和 xm 进行原地乘法运算
        x *= 2.0
        assert_equal(x, y * 2)
        xm *= 2.0
        assert_equal(xm, y * 2)

    # 测试原地数组乘法操作
    def test_inplace_multiplication_array(self):
        # 获取浮点数数据和掩码数据
        (x, y, xm) = self.floatdata
        # 获取掩码
        m = xm.mask
        # 创建另一个浮点数数组 a
        a = arange(10, dtype=float)
        a[-1] = masked
        # 对数组 x 和 xm 进行原地乘法运算
        x *= a
        xm *= a
        assert_equal(x, y * a)
        assert_equal(xm, y * a)
        # 断言掩码数组等于原掩码数组和新数组 a 的掩码的逻辑或
        assert_equal(xm.mask, mask_or(m, a.mask))

    # 测试原地整数除法操作
    def test_inplace_division_scalar_int(self):
        # 获取整数数据和掩码数据
        (x, y, xm) = self.intdata
        # 对数组 x 和 xm 进行整数除法运算
        x //= 2
        assert_equal(x, y)
        xm //= 2
        assert_equal(xm, y)

    # 测试原地浮点数除法操作
    def test_inplace_division_scalar_float(self):
        # 获取浮点数数据
        (x, y, xm) = self.floatdata
        # 对数组 x 和 xm 进行浮点数除法运算
        x /= 2.0
        assert_equal(x, y / 2.0)
        xm /= arange(10)
        assert_equal(xm, ones((10,)))
    def test_inplace_division_array_float(self):
        # Test of inplace division
        # 解包测试数据元组
        (x, y, xm) = self.floatdata
        # 获取掩码
        m = xm.mask
        # 创建一个浮点数类型的数组，包含从0到9的元素
        a = arange(10, dtype=float)
        # 将数组中最后一个元素设置为掩码
        a[-1] = masked
        # 对数组a进行原地除法操作，修改数组x
        x /= a
        # 对数组xm进行原地除法操作，修改数组xm
        xm /= a
        # 断言x的值等于y除以a的结果
        assert_equal(x, y / a)
        # 断言xm的值等于y除以a的结果
        assert_equal(xm, y / a)
        # 断言xm的掩码等于m、a的掩码或者a等于0的掩码的逻辑或结果
        assert_equal(xm.mask, mask_or(mask_or(m, a.mask), (a == 0)))

    def test_inplace_division_misc(self):

        x = [1., 1., 1., -2., pi / 2., 4., 5., -10., 10., 1., 2., 3.]
        y = [5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.]
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        # 创建带掩码的数组xm和ym
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)

        # 对xm和ym进行除法操作，得到结果数组z
        z = xm / ym
        # 断言z的掩码等于指定的列表
        assert_equal(z._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        # 断言z的数据部分等于指定的列表
        assert_equal(z._data,
                     [1., 1., 1., -1., -pi / 2., 4., 5., 1., 1., 1., 2., 3.])

        # 复制xm的副本，并对其进行原地除法操作，修改xm自身
        xm = xm.copy()
        xm /= ym
        # 断言xm的掩码等于指定的列表
        assert_equal(xm._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        # 断言z的数据部分等于指定的列表
        assert_equal(z._data,
                     [1., 1., 1., -1., -pi / 2., 4., 5., 1., 1., 1., 2., 3.])

    def test_datafriendly_add(self):
        # Test keeping data w/ (inplace) addition
        x = array([1, 2, 3], mask=[0, 0, 1])
        # 使用标量进行加法操作
        xx = x + 1
        # 断言加法后的数据部分与预期相符
        assert_equal(xx.data, [2, 3, 3])
        # 断言加法后的掩码部分与预期相符
        assert_equal(xx.mask, [0, 0, 1])
        # 使用标量进行原地加法操作
        x += 1
        # 断言原地加法后的数据部分与预期相符
        assert_equal(x.data, [2, 3, 3])
        # 断言原地加法后的掩码部分与预期相符
        assert_equal(x.mask, [0, 0, 1])
        # 使用数组进行加法操作
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x + array([1, 2, 3], mask=[1, 0, 0])
        # 断言加法后的数据部分与预期相符
        assert_equal(xx.data, [1, 4, 3])
        # 断言加法后的掩码部分与预期相符
        assert_equal(xx.mask, [1, 0, 1])
        # 使用数组进行原地加法操作
        x = array([1, 2, 3], mask=[0, 0, 1])
        x += array([1, 2, 3], mask=[1, 0, 0])
        # 断言原地加法后的数据部分与预期相符
        assert_equal(x.data, [1, 4, 3])
        # 断言原地加法后的掩码部分与预期相符
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_sub(self):
        # Test keeping data w/ (inplace) subtraction
        # 使用标量进行减法操作
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x - 1
        # 断言减法后的数据部分与预期相符
        assert_equal(xx.data, [0, 1, 3])
        # 断言减法后的掩码部分与预期相符
        assert_equal(xx.mask, [0, 0, 1])
        # 使用标量进行原地减法操作
        x = array([1, 2, 3], mask=[0, 0, 1])
        x -= 1
        # 断言原地减法后的数据部分与预期相符
        assert_equal(x.data, [0, 1, 3])
        # 断言原地减法后的掩码部分与预期相符
        assert_equal(x.mask, [0, 0, 1])
        # 使用数组进行减法操作
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x - array([1, 2, 3], mask=[1, 0, 0])
        # 断言减法后的数据部分与预期相符
        assert_equal(xx.data, [1, 0, 3])
        # 断言减法后的掩码部分与预期相符
        assert_equal(xx.mask, [1, 0, 1])
        # 使用数组进行原地减法操作
        x = array([1, 2, 3], mask=[0, 0, 1])
        x -= array([1, 2, 3], mask=[1, 0, 0])
        # 断言原地减法后的数据部分与预期相符
        assert_equal(x.data, [1, 0, 3])
        # 断言原地减法后的掩码部分与预期相符
        assert_equal(x.mask, [1, 0, 1])
    def test_datafriendly_mul(self):
        # Test keeping data w/ (inplace) multiplication

        # Test mul w/ scalar
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x * 2
        assert_equal(xx.data, [2, 4, 3])
        assert_equal(xx.mask, [0, 0, 1])

        # Test imul w/ scalar
        x = array([1, 2, 3], mask=[0, 0, 1])
        x *= 2
        assert_equal(x.data, [2, 4, 3])
        assert_equal(x.mask, [0, 0, 1])

        # Test mul w/ array
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x * array([10, 20, 30], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 40, 3])
        assert_equal(xx.mask, [1, 0, 1])

        # Test imul w/ array
        x = array([1, 2, 3], mask=[0, 0, 1])
        x *= array([10, 20, 30], mask=[1, 0, 0])
        assert_equal(x.data, [1, 40, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_div(self):
        # Test keeping data w/ (inplace) division

        # Test div on scalar
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x / 2.
        assert_equal(xx.data, [1 / 2., 2 / 2., 3])
        assert_equal(xx.mask, [0, 0, 1])

        # Test idiv on scalar
        x = array([1., 2., 3.], mask=[0, 0, 1])
        x /= 2.
        assert_equal(x.data, [1 / 2., 2 / 2., 3])
        assert_equal(x.mask, [0, 0, 1])

        # Test div on array
        x = array([1., 2., 3.], mask=[0, 0, 1])
        xx = x / array([10., 20., 30.], mask=[1, 0, 0])
        assert_equal(xx.data, [1., 2. / 20., 3.])
        assert_equal(xx.mask, [1, 0, 1])

        # Test idiv on array
        x = array([1., 2., 3.], mask=[0, 0, 1])
        x /= array([10., 20., 30.], mask=[1, 0, 0])
        assert_equal(x.data, [1., 2 / 20., 3.])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_pow(self):
        # Test keeping data w/ (inplace) power

        # Test pow on scalar
        x = array([1., 2., 3.], mask=[0, 0, 1])
        xx = x ** 2.5
        assert_equal(xx.data, [1., 2. ** 2.5, 3.])
        assert_equal(xx.mask, [0, 0, 1])

        # Test ipow on scalar
        x **= 2.5
        assert_equal(x.data, [1., 2. ** 2.5, 3])
        assert_equal(x.mask, [0, 0, 1])

    def test_datafriendly_add_arrays(self):
        # Test addition of arrays with mask handling

        # Test case where mask is not affecting result
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 0])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        # Test case where mask affects some elements
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        assert_equal(a.mask, [[0, 1], [0, 1]])
    def test_datafriendly_sub_arrays(self):
        # 定义一个二维数组 a
        a = array([[1, 1], [3, 3]])
        # 定义一个一维数组 b，并指定其掩码（mask）
        b = array([1, 1], mask=[0, 0])
        # 对数组 a 中的每个元素减去数组 b 对应位置的元素
        a -= b
        # 断言数组 a 的结果与预期相符
        assert_equal(a, [[0, 0], [2, 2]])
        # 如果数组 a 使用了掩码，断言其掩码与预期相符
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        # 重新定义数组 a 和 b
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        # 对数组 a 中的每个元素减去数组 b 对应位置的元素
        a -= b
        # 断言数组 a 的结果与预期相符
        assert_equal(a, [[0, 0], [2, 2]])
        # 断言数组 a 的掩码与预期相符
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_datafriendly_mul_arrays(self):
        # 定义一个二维数组 a
        a = array([[1, 1], [3, 3]])
        # 定义一个一维数组 b，并指定其掩码（mask）
        b = array([1, 1], mask=[0, 0])
        # 对数组 a 中的每个元素乘以数组 b 对应位置的元素
        a *= b
        # 断言数组 a 的结果与预期相符
        assert_equal(a, [[1, 1], [3, 3]])
        # 如果数组 a 使用了掩码，断言其掩码与预期相符
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        # 重新定义数组 a 和 b
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        # 对数组 a 中的每个元素乘以数组 b 对应位置的元素
        a *= b
        # 断言数组 a 的结果与预期相符
        assert_equal(a, [[1, 1], [3, 3]])
        # 断言数组 a 的掩码与预期相符
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_inplace_addition_scalar_type(self):
        # 测试原地加法操作
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 从 self.uint8data 中获取不同类型的数据
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                # 将 xm 中索引为 2 的元素设为 masked
                xm[2] = masked
                # 对数组 x 中的每个元素加上类型 t 的整数 1
                x += t(1)
                # 断言数组 x 的结果与数组 y 加上类型 t 的整数 1 的结果相符
                assert_equal(x, y + t(1))
                # 对数组 xm 中的每个元素加上类型 t 的整数 1
                xm += t(1)
                # 断言数组 xm 的结果与数组 y 加上类型 t 的整数 1 的结果相符
                assert_equal(xm, y + t(1))

    def test_inplace_addition_array_type(self):
        # 测试原地加法操作
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 从 self.uint8data 中获取不同类型的数据
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                # 获取 xm 的掩码
                m = xm.mask
                # 创建一个类型为 t 的数组 a，长度为 10
                a = arange(10, dtype=t)
                # 将数组 a 中最后一个元素设为 masked
                a[-1] = masked
                # 对数组 x 中的每个元素加上数组 a 中对应位置的元素
                x += a
                # 对数组 xm 中的每个元素加上数组 a 中对应位置的元素
                xm += a
                # 断言数组 x 的结果与数组 y 加上数组 a 的结果相符
                assert_equal(x, y + a)
                # 断言数组 xm 的结果与数组 y 加上数组 a 的结果相符
                assert_equal(xm, y + a)
                # 断言数组 xm 的掩码与 m 或 a 的掩码的结果相符
                assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_subtraction_scalar_type(self):
        # 测试原地减法操作
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 从 self.uint8data 中获取不同类型的数据
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                # 对数组 x 中的每个元素减去类型 t 的整数 1
                x -= t(1)
                # 断言数组 x 的结果与数组 y 减去类型 t 的整数 1 的结果相符
                assert_equal(x, y - t(1))
                # 对数组 xm 中的每个元素减去类型 t 的整数 1
                xm -= t(1)
                # 断言数组 xm 的结果与数组 y 减去类型 t 的整数 1 的结果相符
                assert_equal(xm, y - t(1))

    def test_inplace_subtraction_array_type(self):
        # 测试原地减法操作
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 从 self.uint8data 中获取不同类型的数据
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                # 获取 xm 的掩码
                m = xm.mask
                # 创建一个类型为 t 的数组 a，长度为 10
                a = arange(10, dtype=t)
                # 将数组 a 中最后一个元素设为 masked
                a[-1] = masked
                # 对数组 x 中的每个元素减去数组 a 中对应位置的元素
                x -= a
                # 对数组 xm 中的每个元素减去数组 a 中对应位置的元素
                xm -= a
                # 断言数组 x 的结果与数组 y 减去数组 a 的结果相符
                assert_equal(x, y - a)
                # 断言数组 xm 的结果与数组 y 减去数组 a 的结果相符
                assert_equal(xm, y - a)
                # 断言数组 xm 的掩码与 m 或 a 的掩码的结果相符
                assert_equal(xm.mask, mask_or(m, a.mask))
    def test_inplace_multiplication_scalar_type(self):
        # 测试原地乘法
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 将每个数据转换为指定类型，并执行原地乘法操作
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x *= t(2)  # 原地乘以标量 t(2)
                assert_equal(x, y * t(2))  # 断言结果是否与预期相同
                xm *= t(2)  # 原地乘以标量 t(2)
                assert_equal(xm, y * t(2))  # 断言结果是否与预期相同

    def test_inplace_multiplication_array_type(self):
        # 测试原地乘法
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 将每个数据转换为指定类型，并执行原地乘法操作
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                x *= a  # 原地乘以数组 a
                xm *= a  # 原地乘以数组 a
                assert_equal(x, y * a)  # 断言结果是否与预期相同
                assert_equal(xm, y * a)  # 断言结果是否与预期相同
                assert_equal(xm.mask, mask_or(m, a.mask))  # 断言结果是否与预期相同

    def test_inplace_floor_division_scalar_type(self):
        # 测试原地整除
        # 检查不支持类型时是否会抛出 TypeError
        unsupported = {np.dtype(t).type for t in np.typecodes["Complex"]}
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 将每个数据转换为指定类型，并执行原地整除操作
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x = arange(10, dtype=t) * t(2)
                xm = arange(10, dtype=t) * t(2)
                xm[2] = masked
                try:
                    x //= t(2)  # 原地整除以标量 t(2)
                    xm //= t(2)  # 原地整除以标量 t(2)
                    assert_equal(x, y)  # 断言结果是否与预期相同
                    assert_equal(xm, y)  # 断言结果是否与预期相同
                except TypeError:
                    msg = f"Supported type {t} throwing TypeError"
                    assert t in unsupported, msg  # 断言是否符合不支持类型的预期行为

    def test_inplace_floor_division_array_type(self):
        # 测试原地整除
        # 检查不支持类型时是否会抛出 TypeError
        unsupported = {np.dtype(t).type for t in np.typecodes["Complex"]}
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 将每个数据转换为指定类型，并执行原地整除操作
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                try:
                    x //= a  # 原地整除以数组 a
                    xm //= a  # 原地整除以数组 a
                    assert_equal(x, y // a)  # 断言结果是否与预期相同
                    assert_equal(xm, y // a)  # 断言结果是否与预期相同
                    assert_equal(
                        xm.mask,
                        mask_or(mask_or(m, a.mask), (a == t(0)))
                    )  # 断言结果是否与预期相同
                except TypeError:
                    msg = f"Supported type {t} throwing TypeError"
                    assert t in unsupported, msg  # 断言是否符合不支持类型的预期行为
    # 定义测试方法，用于测试就地除法
    def test_inplace_division_scalar_type(self):
        # 对其他类型进行就地除法的测试
        for t in self.othertypes:
            # 使用suppress_warnings上下文管理器，记录UserWarning
            with suppress_warnings() as sup:
                sup.record(UserWarning)

                # 将每个元素转换为指定类型，并赋值给变量x, y, xm
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                # 创建长度为10的数组x，并将其乘以t(2)赋值给x
                x = arange(10, dtype=t) * t(2)
                # 创建长度为10的数组xm，并将其乘以t(2)赋值给xm
                xm = arange(10, dtype=t) * t(2)
                # 将数组xm的第二个元素设为masked

                # 可能会出现DeprecationWarning或TypeError
                #
                # 这是事实是真除法，需要将其转换为浮点数进行计算，然后再转换回原始类型。只会对整数类型才会引发警告或错误。
                # 是否是错误还是警告取决于转换规则的严格程度。
                #
                # 将以相同方式处理
                try:
                    # 对x进行就地除以t(2)的操作
                    x /= t(2)
                    # 断言x与y相等
                    assert_equal(x, y)
                except (DeprecationWarning, TypeError) as e:
                    # 若出现DeprecationWarning或TypeError，则发出警告
                    warnings.warn(str(e), stacklevel=1)
                try:
                    # 对xm进行就地除以t(2)的操作
                    xm /= t(2)
                    # 断言xm与y相等
                    assert_equal(xm, y)
                except (DeprecationWarning, TypeError) as e:
                    # 若出现DeprecationWarning或TypeError，则发出警告
                    warnings.warn(str(e), stacklevel=1)

                # 如果t是np.integer的子类
                if issubclass(t, np.integer):
                    # 断言sup.log的长度为2，若不相等则抛出异常信息
                    assert_equal(len(sup.log), 2, f'Failed on type={t}.')
                else:
                    # 断言sup.log的长度为0，若不相等则抛出异常信息
                    assert_equal(len(sup.log), 0, f'Failed on type={t}.')
    # 测试数组的原地除法操作
    def test_inplace_division_array_type(self):
        # 遍历不同的数据类型进行测试
        for t in self.othertypes:
            # 使用 suppress_warnings 上下文管理器，记录 UserWarning
            with suppress_warnings() as sup:
                sup.record(UserWarning)
                # 将数据转换为指定类型 t，并解包为 x, y, xm 三个变量
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                # 获取 xm 的掩码
                m = xm.mask
                # 创建一个 dtype 为 t 的 0 到 9 的数组
                a = arange(10, dtype=t)
                # 将数组 a 的最后一个元素设置为 masked

                # 可能会引发 DeprecationWarning 或 TypeError
                #
                # 这是因为这是真正的除法操作，需要将数据转换为浮点数进行计算，
                # 然后再转换回原始类型。这只会在处理整数时引发警告或错误，
                # 具体是警告还是错误取决于转换规则的严格程度。
                #
                # 将以相同方式处理。
                try:
                    x /= a
                    assert_equal(x, y / a)
                except (DeprecationWarning, TypeError) as e:
                    # 发出警告，并指定 stacklevel 为 1
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= a
                    assert_equal(xm, y / a)
                    assert_equal(
                        xm.mask,
                        mask_or(mask_or(m, a.mask), (a == t(0)))
                    )
                except (DeprecationWarning, TypeError) as e:
                    # 发出警告，并指定 stacklevel 为 1
                    warnings.warn(str(e), stacklevel=1)

                # 如果 t 是 np.integer 的子类，断言 sup.log 长度为 2
                # 否则断言 sup.log 长度为 0
                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, f'Failed on type={t}.')
                else:
                    assert_equal(len(sup.log), 0, f'Failed on type={t}.')

    # 测试保持数据的原地幂运算
    def test_inplace_pow_type(self):
        # 遍历不同的数据类型进行测试
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # 测试标量的幂运算
                x = array([1, 2, 3], mask=[0, 0, 1], dtype=t)
                xx = x ** t(2)
                xx_r = array([1, 2 ** 2, 3], mask=[0, 0, 1], dtype=t)
                # 断言 xx 的数据和掩码与预期结果相等
                assert_equal(xx.data, xx_r.data)
                assert_equal(xx.mask, xx_r.mask)
                # 测试原地标量的幂运算
                x **= t(2)
                # 断言 x 的数据和掩码与预期结果相等
                assert_equal(x.data, xx_r.data)
                assert_equal(x.mask, xx_r.mask)
# Test class for miscellaneous MaskedArrays methods.
class TestMaskedArrayMethods:
    
    def setup_method(self):
        # Base data definition.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        # Create masked arrays based on data and mask arrays
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        # Create another set of masked arrays for comparison
        m2x = array(data=x, mask=m2)
        m2X = array(data=X, mask=m2.reshape(X.shape))
        m2XX = array(data=XX, mask=m2.reshape(XX.shape))
        # Store all created arrays in instance variable d
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_generic_methods(self):
        # Tests some MaskedArray methods.
        a = array([1, 3, 2])
        # Assert equality of results from MaskedArray methods and their underlying data counterparts
        assert_equal(a.any(), a._data.any())
        assert_equal(a.all(), a._data.all())
        assert_equal(a.argmax(), a._data.argmax())
        assert_equal(a.argmin(), a._data.argmin())
        assert_equal(a.choose(0, 1, 2, 3, 4), a._data.choose(0, 1, 2, 3, 4))
        assert_equal(a.compress([1, 0, 1]), a._data.compress([1, 0, 1]))
        assert_equal(a.conj(), a._data.conj())
        assert_equal(a.conjugate(), a._data.conjugate())

        m = array([[1, 2], [3, 4]])
        assert_equal(m.diagonal(), m._data.diagonal())
        assert_equal(a.sum(), a._data.sum())
        assert_equal(a.take([1, 2]), a._data.take([1, 2]))
        assert_equal(m.transpose(), m._data.transpose())
    def test_allclose(self):
        # Tests allclose on arrays

        # 创建一个包含10个随机数的NumPy数组a
        a = np.random.rand(10)
        # 将b定义为a加上一个非常小的随机数（避免完全相等）
        b = a + np.random.rand(10) * 1e-8
        # 断言a和b在允许误差范围内相等
        assert_(allclose(a, b))

        # 将a的第一个元素设置为无穷大
        a[0] = np.inf
        # 断言a和b不相等，因为a包含无穷大
        assert_(not allclose(a, b))
        # 将b的第一个元素也设置为无穷大
        b[0] = np.inf
        # 断言a和b在允许误差范围内相等
        assert_(allclose(a, b))

        # 将a转换为带掩码的数组
        a = masked_array(a)
        # 将a的最后一个元素设置为掩码值
        a[-1] = masked
        # 断言a和b在允许误差范围内相等，同时考虑掩码值
        assert_(allclose(a, b, masked_equal=True))
        # 断言a和b不相等，不考虑掩码值
        assert_(not allclose(a, b, masked_equal=False))

        # 将a的每个元素乘以一个非常小的数，并将第一个元素设为0
        a *= 1e-8
        a[0] = 0
        # 断言a和0在允许误差范围内相等，同时考虑掩码值
        assert_(allclose(a, 0, masked_equal=True))

        # 测试函数对MIN_INT整型数组的处理
        a = masked_array([np.iinfo(np.int_).min], dtype=np.int_)
        # 断言a和自身在允许误差范围内相等
        assert_(allclose(a, a))

    def test_allclose_timedelta(self):
        # Allclose目前对timedelta64类型有效，只要atol是整数或timedelta64类型
        a = np.array([[1, 2, 3, 4]], dtype="m8[ns]")
        # 断言a和自身在tolerance为0时相等
        assert allclose(a, a, atol=0)
        # 断言a和自身在tolerance为1纳秒时相等
        assert allclose(a, a, atol=np.timedelta64(1, "ns"))

    def test_allany(self):
        # 检查any/all方法和函数的使用

        # 创建一个包含浮点数的NumPy数组x和一个布尔掩码数组m
        x = np.array([[0.13, 0.26, 0.90],
                      [0.28, 0.33, 0.63],
                      [0.31, 0.87, 0.70]])
        m = np.array([[True, False, False],
                      [False, False, False],
                      [True, True, False]], dtype=np.bool)
        # 将x定义为带掩码的数组mx
        mx = masked_array(x, mask=m)
        # 创建mx的大于0.5的布尔数组mxbig和小于0.5的布尔数组mxsmall
        mxbig = (mx > 0.5)
        mxsmall = (mx < 0.5)

        # 断言mxbig中不是所有元素都为真
        assert_(not mxbig.all())
        # 断言mxbig中至少有一个元素为真
        assert_(mxbig.any())
        # 断言mxbig沿着0轴的all()结果为[False, False, True]
        assert_equal(mxbig.all(0), [False, False, True])
        # 断言mxbig沿着1轴的all()结果为[False, False, True]
        assert_equal(mxbig.all(1), [False, False, True])
        # 断言mxbig沿着0轴的any()结果为[False, False, True]
        assert_equal(mxbig.any(0), [False, False, True])
        # 断言mxbig沿着1轴的any()结果为[True, True, True]
        assert_equal(mxbig.any(1), [True, True, True])

        # 断言mxsmall中不是所有元素都为真
        assert_(not mxsmall.all())
        # 断言mxsmall中至少有一个元素为真
        assert_(mxsmall.any())
        # 断言mxsmall沿着0轴的all()结果为[True, True, False]
        assert_equal(mxsmall.all(0), [True, True, False])
        # 断言mxsmall沿着1轴的all()结果为[False, False, False]
        assert_equal(mxsmall.all(1), [False, False, False])
        # 断言mxsmall沿着0轴的any()结果为[True, True, False]
        assert_equal(mxsmall.any(0), [True, True, False])
        # 断言mxsmall沿着1轴的any()结果为[True, True, False]
        assert_equal(mxsmall.any(1), [True, True, False])

    def test_allany_oddities(self):
        # 一些与all和any相关的有趣情况的测试

        # 创建一个空的布尔类型数组store
        store = empty((), dtype=bool)
        # 创建一个包含掩码的数组full
        full = array([1, 2, 3], mask=True)

        # 断言full中的所有元素都是掩码
        assert_(full.all() is masked)
        # 将full的all()结果存储到store中
        full.all(out=store)
        # 断言store为真
        assert_(store)
        # 断言store的掩码属性为真
        assert_(store._mask, True)
        # 断言store不是掩码
        assert_(store is not masked)

        # 再次创建一个空的布尔类型数组store
        store = empty((), dtype=bool)
        # 断言full中至少有一个元素不是掩码
        assert_(full.any() is masked)
        # 将full的any()结果存储到store中
        full.any(out=store)
        # 断言store为假
        assert_(not store)
        # 断言store的掩码属性为真
        assert_(store._mask, True)
        # 断言store不是掩码
        assert_(store is not masked)
    def test_argmax_argmin(self):
        # 测试在 MaskedArrays 上的 argmin 和 argmax 函数

        # 从元组 self.d 中解包得到各个变量
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d

        # 断言 mx 的最小值索引为 35
        assert_equal(mx.argmin(), 35)
        # 断言 mX 的最小值索引为 35
        assert_equal(mX.argmin(), 35)
        # 断言 m2x 的最小值索引为 4
        assert_equal(m2x.argmin(), 4)
        # 断言 m2X 的最小值索引为 4
        assert_equal(m2X.argmin(), 4)
        # 断言 mx 的最大值索引为 28
        assert_equal(mx.argmax(), 28)
        # 断言 mX 的最大值索引为 28
        assert_equal(mX.argmax(), 28)
        # 断言 m2x 的最大值索引为 31
        assert_equal(m2x.argmax(), 31)
        # 断言 m2X 的最大值索引为 31
        assert_equal(m2X.argmax(), 31)

        # 断言 mX 按列（axis=0）的最小值索引数组为 [2, 2, 2, 5, 0, 5]
        assert_equal(mX.argmin(0), [2, 2, 2, 5, 0, 5])
        # 断言 m2X 按列（axis=0）的最小值索引数组为 [2, 2, 4, 5, 0, 4]
        assert_equal(m2X.argmin(0), [2, 2, 4, 5, 0, 4])
        # 断言 mX 按列（axis=0）的最大值索引数组为 [0, 5, 0, 5, 4, 0]
        assert_equal(mX.argmax(0), [0, 5, 0, 5, 4, 0])
        # 断言 m2X 按列（axis=0）的最大值索引数组为 [5, 5, 0, 5, 1, 0]
        assert_equal(m2X.argmax(0), [5, 5, 0, 5, 1, 0])

        # 断言 mX 按行（axis=1）的最小值索引数组为 [4, 1, 0, 0, 5, 5]
        assert_equal(mX.argmin(1), [4, 1, 0, 0, 5, 5])
        # 断言 m2X 按行（axis=1）的最小值索引数组为 [4, 4, 0, 0, 5, 3]
        assert_equal(m2X.argmin(1), [4, 4, 0, 0, 5, 3])
        # 断言 mX 按行（axis=1）的最大值索引数组为 [2, 4, 1, 1, 4, 1]
        assert_equal(mX.argmax(1), [2, 4, 1, 1, 4, 1])
        # 断言 m2X 按行（axis=1）的最大值索引数组为 [2, 4, 1, 1, 1, 1]
        assert_equal(m2X.argmax(1), [2, 4, 1, 1, 1, 1])

    def test_clip(self):
        # 测试 MaskedArrays 上的 clip 函数

        # 创建数组 x 和掩码数组 m
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        # 创建 MaskedArray 对象 mx
        mx = array(x, mask=m)
        # 对 mx 应用 clip(2, 8)
        clipped = mx.clip(2, 8)
        # 断言 clipped 的掩码与 mx 的掩码相等
        assert_equal(clipped.mask, mx.mask)
        # 断言 clipped 的数据部分与 x 应用 clip(2, 8) 后的结果相等
        assert_equal(clipped._data, x.clip(2, 8))
        # 断言 clipped 的数据部分与 mx 的数据部分应用 clip(2, 8) 后的结果相等
        assert_equal(clipped._data, mx._data.clip(2, 8))

    def test_clip_out(self):
        # 测试 MaskedArray 上的 clip 函数的 out 参数

        # 创建长度为 10 的一维数组 a
        a = np.arange(10)
        # 创建一个 MaskedArray 对象 m，使用 a 作为数据，[0, 1] * 5 作为掩码
        m = np.ma.MaskedArray(a, mask=[0, 1] * 5)
        # 将 m 按照 clip(0, 5) 的结果写回 m
        m.clip(0, 5, out=m)
        # 断言 m 的掩码部分与预期结果 [0, 1] * 5 相等
        assert_equal(m.mask, [0, 1] * 5)

    def test_compress(self):
        # 测试 MaskedArray 上的 compress 函数

        # 创建填充值为 9999 的 masked_array 对象 a
        a = masked_array([1., 2., 3., 4., 5.], fill_value=9999)
        # 创建条件数组 condition
        condition = (a > 1.5) & (a < 3.5)
        # 断言 a 按照 condition 压缩后的结果为 [2., 3.]
        assert_equal(a.compress(condition), [2., 3.])

        # 修改 a 的部分数据为 masked
        a[[2, 3]] = masked
        # 再次按照 condition 压缩得到结果 b
        b = a.compress(condition)
        # 断言 b 的数据部分与预期结果 [2., 3.] 相等
        assert_equal(b._data, [2., 3.])
        # 断言 b 的掩码部分与预期结果 [0, 1] 相等
        assert_equal(b._mask, [0, 1])
        # 断言 b 的填充值与 a 的填充值相等
        assert_equal(b.fill_value, 9999)
        # 断言 b 与 a[condition] 的结果相等
        assert_equal(b, a[condition])

        # 创建二维 masked_array 对象 a
        a = masked_array([[10, 20, 30], [40, 50, 60]],
                         mask=[[0, 0, 1], [1, 0, 0]])
        # 按照 a.ravel() >= 22 的条件压缩得到结果 b
        b = a.compress(a.ravel() >= 22)
        # 断言 b 的数据部分与预期结果 [30, 40, 50, 60] 相等
        assert_equal(b._data, [30, 40, 50, 60])
        # 断言 b 的掩码部分与预期结果 [1, 1, 0, 0] 相等
        assert_equal(b._
    def test_compressed(self):
        # Tests compressed method of masked arrays
        # 创建一个包含四个元素的数组，其中没有被屏蔽的值为 [1, 2, 3, 4]
        a = array([1, 2, 3, 4], mask=[0, 0, 0, 0])
        # 使用 compressed 方法生成一个没有屏蔽值的新数组 b，并断言其与原数组 a 相等
        b = a.compressed()
        assert_equal(b, a)
        # 将数组 a 的第一个元素设置为 masked（屏蔽），再次使用 compressed 方法生成新数组 b，并断言其值为 [2, 3, 4]
        a[0] = masked
        b = a.compressed()
        assert_equal(b, [2, 3, 4])

    def test_empty(self):
        # Tests empty and empty_like functions for masked arrays
        # 定义一个包含三个字段的数据类型结构
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        # 创建一个 masked_array 对象 a，包含三个元组，每个元组都包含一个整数、一个浮点数和一个字符串
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')],
                         dtype=datatype)
        # 断言 a 的 fill_value 属性的长度等于数据类型结构的长度
        assert_equal(len(a.fill_value.item()), len(datatype))

        # 使用 empty_like 函数创建一个与 a 相同形状的新数组 b，并断言其形状与 a 相等，fill_value 与 a 的 fill_value 相等
        b = empty_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # 使用 empty 函数创建一个与 a 相同长度和数据类型的新数组 b，并断言其形状与 a 相等，fill_value 与 a 的 fill_value 相等
        b = empty(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # 检查 empty_like 对 mask 的处理，创建一个包含 [1, 2, 3] 的 masked_array 对象 a，其中第二个元素被屏蔽
        a = masked_array([1, 2, 3], mask=[False, True, False])
        # 使用 empty_like 创建新数组 b，断言 a 和 b 的 mask 不共享内存
        b = empty_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        # 将 b 转换为 masked_array 类型，断言 a 和 b 的 mask 共享内存
        b = a.view(masked_array)
        assert_(np.may_share_memory(a.mask, b.mask))

    def test_zeros(self):
        # Tests zeros and zeros_like functions for masked arrays
        # 定义一个包含三个字段的数据类型结构
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        # 创建一个 masked_array 对象 a，包含三个元组，每个元组都包含一个整数、一个浮点数和一个字符串
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')],
                         dtype=datatype)
        # 断言 a 的 fill_value 属性的长度等于数据类型结构的长度
        assert_equal(len(a.fill_value.item()), len(datatype))

        # 使用 zeros 函数创建一个长度与 a 相同且数据类型为 datatype 的新数组 b，并断言其形状与 a 相等，fill_value 与 a 的 fill_value 相等
        b = zeros(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # 使用 zeros_like 函数创建一个与 a 相同形状的新数组 b，并断言其形状与 a 相等，fill_value 与 a 的 fill_value 相等
        b = zeros_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # 检查 zeros_like 对 mask 的处理，创建一个包含 [1, 2, 3] 的 masked_array 对象 a，其中第二个元素被屏蔽
        a = masked_array([1, 2, 3], mask=[False, True, False])
        # 使用 zeros_like 创建新数组 b，断言 a 和 b 的 mask 不共享内存
        b = zeros_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        # 将 b 转换为普通数组类型，断言 a 和 b 的 mask 共享内存
        b = a.view()
        assert_(np.may_share_memory(a.mask, b.mask))

    def test_ones(self):
        # Tests ones and ones_like functions for masked arrays
        # 定义一个包含三个字段的数据类型结构
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        # 创建一个 masked_array 对象 a，包含三个元组，每个元组都包含一个整数、一个浮点数和一个字符串
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')],
                         dtype=datatype)
        # 断言 a 的 fill_value 属性的长度等于数据类型结构的长度
        assert_equal(len(a.fill_value.item()), len(datatype))

        # 使用 ones 函数创建一个长度与 a 相同且数据类型为 datatype 的新数组 b，并断言其形状与 a 相等，fill_value 与 a 的 fill_value 相等
        b = ones(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # 使用 ones_like 函数创建一个与 a 相同形状的新数组 b，并断言其形状与 a 相等，fill_value 与 a 的 fill_value 相等
        b = ones_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # 检查 ones_like 对 mask 的处理，创建一个包含 [1, 2, 3] 的 masked_array 对象 a，其中第二个元素被屏蔽
        a = masked_array([1, 2, 3], mask=[False, True, False])
        # 使用 ones_like 创建新数组 b，断言 a 和 b 的 mask 不共享内存
        b = ones_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        # 将 b 转换为普通数组类型，断言 a 和 b 的 mask 共享内存
        b = a.view()
        assert_(np.may_share_memory(a.mask, b.mask))

    @suppress_copy_mask_on_assignment
    def test_put(self):
        # 测试 put 方法
        # 创建一个包含数字 0 到 4 的数组
        d = arange(5)
        # 指定每个元素的分组情况，用于创建掩码
        n = [0, 0, 0, 1, 1]
        # 根据给定的分组情况创建掩码
        m = make_mask(n)
        # 创建一个带掩码的数组
        x = array(d, mask=m)
        # 确保索引为 3 的元素被掩盖
        assert_(x[3] is masked)
        # 确保索引为 4 的元素被掩盖
        assert_(x[4] is masked)
        # 将索引为 1 和 4 的元素分别设置为 10 和 40
        x[[1, 4]] = [10, 40]
        # 确保索引为 3 的元素仍然被掩盖
        assert_(x[3] is masked)
        # 确保索引为 4 的元素不再被掩盖
        assert_(x[4] is not masked)
        # 确保数组 x 的值与期望值相等
        assert_equal(x, [0, 10, 2, -1, 40])

        # 创建一个带掩码的数组
        x = masked_array(arange(10), mask=[1, 0, 0, 0, 0] * 2)
        # 指定要设置的索引
        i = [0, 2, 4, 6]
        # 使用 put 方法设置指定索引处的值
        x.put(i, [6, 4, 2, 0])
        # 确保数组 x 的值与期望值相等
        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9, ]))
        # 确保数组 x 的掩码与期望值相等
        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        # 使用 put 方法设置带有掩码的数组 x 的指定索引处的值
        x.put(i, masked_array([0, 2, 4, 6], [1, 0, 1, 0]))
        # 确保数组 x 的值与期望值相等
        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        # 确保数组 x 的掩码与期望值相等
        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

        # 创建一个带掩码的数组
        x = masked_array(arange(10), mask=[1, 0, 0, 0, 0] * 2)
        # 使用 put 函数设置指定索引处的值
        put(x, i, [6, 4, 2, 0])
        # 确保数组 x 的值与期望值相等
        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9, ]))
        # 确保数组 x 的掩码与期望值相等
        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        # 使用 put 函数设置带有掩码的数组 x 的指定索引处的值
        put(x, i, masked_array([0, 2, 4, 6], [1, 0, 1, 0]))
        # 确保数组 x 的值与期望值相等
        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        # 确保数组 x 的掩码与期望值相等
        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

    def test_put_nomask(self):
        # 测试在不带掩码数组上使用 put 方法
        # 创建一个全部为 0 的数组
        x = zeros(10)
        # 创建一个带有部分掩码的数组
        z = array([3., -1.], mask=[False, True])

        # 使用 put 方法设置指定索引处的值
        x.put([1, 2], z)
        # 确保索引为 0 的元素不被掩盖
        assert_(x[0] is not masked)
        # 确保索引为 0 的元素值为 0
        assert_equal(x[0], 0)
        # 确保索引为 1 的元素不被掩盖
        assert_(x[1] is not masked)
        # 确保索引为 1 的元素值为 3
        assert_equal(x[1], 3)
        # 确保索引为 2 的元素被掩盖
        assert_(x[2] is masked)
        # 确保索引为 3 的元素不被掩盖
        assert_(x[3] is not masked)
        # 确保索引为 3 的元素值为 0
        assert_equal(x[3], 0)

    def test_put_hardmask(self):
        # 测试在硬掩码上使用 put 方法
        # 创建一个包含数字 0 到 4 的数组
        d = arange(5)
        # 指定每个元素的分组情况，用于创建掩码
        n = [0, 0, 0, 1, 1]
        # 根据给定的分组情况创建掩码
        m = make_mask(n)
        # 创建一个带硬掩码的数组
        xh = array(d + 1, mask=m, hard_mask=True, copy=True)
        # 使用 put 方法设置指定索引处的值
        xh.put([4, 2, 0, 1, 3], [1, 2, 3, 4, 5])
        # 确保数组 xh 的实际数据与期望值相等
        assert_equal(xh._data, [3, 4, 2, 4, 5])
    def test_putmask(self):
        x = arange(6) + 1
        mx = array(x, mask=[0, 0, 0, 1, 1, 1])
        mask = [0, 0, 1, 0, 0, 1]
        # 创建副本 xx，并将 mask 应用于其中，用 99 替换被 mask 的位置
        xx = x.copy()
        putmask(xx, mask, 99)
        assert_equal(xx, [1, 2, 99, 4, 5, 99])
        # 创建 mx 的副本 mxx，并将 mask 应用于其中，用 99 替换被 mask 的位置
        mxx = mx.copy()
        putmask(mxx, mask, 99)
        # 验证 mxx 的数据部分等于期望值
        assert_equal(mxx._data, [1, 2, 99, 4, 5, 99])
        # 验证 mxx 的掩码部分等于期望值
        assert_equal(mxx._mask, [0, 0, 0, 1, 1, 0])
        # 创建副本 xx，并将 mask 应用于其中，用 values 替换被 mask 的位置
        values = array([10, 20, 30, 40, 50, 60], mask=[1, 1, 1, 0, 0, 0])
        xx = x.copy()
        putmask(xx, mask, values)
        # 验证 xx 的数据部分等于期望值
        assert_equal(xx._data, [1, 2, 30, 4, 5, 60])
        # 验证 xx 的掩码部分等于期望值
        assert_equal(xx._mask, [0, 0, 1, 0, 0, 0])
        # 创建 mx 的副本 mxx，并将 mask 应用于其中，用 values 替换被 mask 的位置
        mxx = mx.copy()
        putmask(mxx, mask, values)
        # 验证 mxx 的数据部分等于期望值
        assert_equal(mxx._data, [1, 2, 30, 4, 5, 60])
        # 验证 mxx 的掩码部分等于期望值
        assert_equal(mxx._mask, [0, 0, 1, 1, 1, 0])
        # 创建 mx 的副本 mxx，并将掩码硬化，然后将 mask 应用于其中，用 values 替换被 mask 的位置
        mxx = mx.copy()
        mxx.harden_mask()
        putmask(mxx, mask, values)
        # 验证 mxx 等于期望值
        assert_equal(mxx, [1, 2, 30, 4, 5, 60])

    def test_ravel(self):
        # 测试 ravel 方法
        a = array([[1, 2, 3, 4, 5]], mask=[[0, 1, 0, 0, 0]])
        # 对 a 进行 ravel 操作
        aravel = a.ravel()
        # 验证 aravel 的掩码形状与其本身形状相同
        assert_equal(aravel._mask.shape, aravel.shape)
        a = array([0, 0], mask=[1, 1])
        # 对 a 进行 ravel 操作
        aravel = a.ravel()
        # 验证 aravel 的掩码形状与 a 的形状相同
        assert_equal(aravel._mask.shape, a.shape)
        # 检查小掩码是否被保留
        a = array([1, 2, 3, 4], mask=[0, 0, 0, 0], shrink=False)
        # 对 a 进行 ravel 操作
        assert_equal(a.ravel()._mask, [0, 0, 0, 0])
        # 测试 fill_value 是否被保留
        a.fill_value = -99
        a.shape = (2, 2)
        ar = a.ravel()
        # 验证 ar 的掩码等于期望值
        assert_equal(ar._mask, [0, 0, 0, 0])
        # 验证 ar 的数据等于期望值
        assert_equal(ar._data, [1, 2, 3, 4])
        # 验证 ar 的 fill_value 等于期望值
        assert_equal(ar.fill_value, -99)
        # 测试索引顺序为 'C' 的情况
        assert_equal(a.ravel(order='C'), [1, 2, 3, 4])
        # 测试索引顺序为 'F' 的情况
        assert_equal(a.ravel(order='F'), [1, 3, 2, 4])

    @pytest.mark.parametrize("order", "AKCF")
    @pytest.mark.parametrize("data_order", "CF")
    def test_ravel_order(self, order, data_order):
        # Ravelling 操作必须始终以相同的顺序对数据和掩码进行操作，以避免在 ravel 结果中使两者不对齐。
        arr = np.ones((5, 10), order=data_order)
        arr[0, :] = 0
        mask = np.ones((10, 5), dtype=bool, order=data_order).T
        mask[0, :] = False
        x = array(arr, mask=mask)
        # 验证 x 的数据部分和掩码部分的 fnc 标志不相同
        assert x._data.flags.fnc != x._mask.flags.fnc
        # 验证 x 填充后的值等于 0
        assert (x.filled(0) == 0).all()
        # 对 x 进行 ravel 操作
        raveled = x.ravel(order)
        # 验证 raveled 填充后的值等于 0
        assert (raveled.filled(0) == 0).all()

        # 注意：如果 arr 的顺序既不是 'C' 也不是 'F'，并且 `order="K"`，则可能出错。
        assert_array_equal(arr.ravel(order), x.ravel(order)._data)
    # 定义测试方法：测试数组重塑操作
    def test_reshape(self):
        # 创建长度为4的数组
        x = arange(4)
        # 将第一个元素设置为掩码值
        x[0] = masked
        # 对数组进行重塑为2x2的形状
        y = x.reshape(2, 2)
        # 断言重塑后数组的形状为(2, 2)
        assert_equal(y.shape, (2, 2,))
        # 断言重塑后数组的掩码形状也为(2, 2)
        assert_equal(y._mask.shape, (2, 2,))
        # 断言原始数组x的形状为(4,)
        assert_equal(x.shape, (4,))
        # 断言原始数组x的掩码形状为(4,)
        assert_equal(x._mask.shape, (4,))

    # 定义测试方法：测试数组排序功能
    def test_sort(self):
        # 创建带有掩码的一维数组x
        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)

        # 对数组进行排序，默认升序
        sortedx = sort(x)
        # 断言排序后的数据部分正确
        assert_equal(sortedx._data, [1, 2, 3, 4])
        # 断言排序后的掩码部分正确
        assert_equal(sortedx._mask, [0, 0, 0, 1])

        # 对数组进行排序，指定降序排序
        sortedx = sort(x, endwith=False)
        # 断言排序后的数据部分正确
        assert_equal(sortedx._data, [4, 1, 2, 3])
        # 断言排序后的掩码部分正确
        assert_equal(sortedx._mask, [1, 0, 0, 0])

        # 原地对数组进行排序，默认升序
        x.sort()
        # 断言原地排序后的数据部分正确
        assert_equal(x._data, [1, 2, 3, 4])
        # 断言原地排序后的掩码部分正确
        assert_equal(x._mask, [0, 0, 0, 1])

        # 创建新的带有掩码的一维数组x
        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)
        # 原地对数组进行排序，指定降序排序
        x.sort(endwith=False)
        # 断言原地排序后的数据部分正确
        assert_equal(x._data, [4, 1, 2, 3])
        # 断言原地排序后的掩码部分正确
        assert_equal(x._mask, [1, 0, 0, 0])

        # 创建普通的一维数组x
        x = [1, 4, 2, 3]
        # 对普通数组进行排序
        sortedx = sort(x)
        # 断言返回的排序结果不是MaskedArray类型
        assert_(not isinstance(sorted, MaskedArray))

        # 创建带有掩码的一维数组x，但没有指定掩码
        x = array([0, 1, -1, -2, 2], mask=nomask, dtype=np.int8)
        # 对数组进行排序，指定不以掩码结尾
        sortedx = sort(x, endwith=False)
        # 断言排序后的数据部分正确
        assert_equal(sortedx._data, [-2, -1, 0, 1, 2])

        # 创建带有掩码的一维数组x，指定掩码
        x = array([0, 1, -1, -2, 2], mask=[0, 1, 0, 0, 1], dtype=np.int8)
        # 对数组进行排序，指定不以掩码结尾
        sortedx = sort(x, endwith=False)
        # 断言排序后的数据部分正确
        assert_equal(sortedx._data, [1, 2, -2, -1, 0])
        # 断言排序后的掩码部分正确
        assert_equal(sortedx._mask, [1, 1, 0, 0, 0])

        # 创建普通的一维数组x
        x = array([0, -1], dtype=np.int8)
        # 对数组进行稳定排序
        sortedx = sort(x, kind="stable")
        # 断言稳定排序后的数组结果正确
        assert_equal(sortedx, array([-1, 0], dtype=np.int8))

    # 定义测试方法：测试稳定排序
    def test_stable_sort(self):
        # 创建普通的一维数组x
        x = array([1, 2, 3, 1, 2, 3], dtype=np.uint8)
        # 期望的排序结果的索引
        expected = array([0, 3, 1, 4, 2, 5])
        # 计算稳定排序后的数组结果的索引
        computed = argsort(x, kind='stable')
        # 断言计算得到的索引结果与期望的索引结果相等
        assert_equal(computed, expected)

    # 定义测试方法：测试argsort函数与sort函数匹配
    def test_argsort_matches_sort(self):
        # 创建带有掩码的一维数组x
        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)

        # 对不同的排序参数组合进行循环测试
        for kwargs in [dict(),
                       dict(endwith=True),
                       dict(endwith=False),
                       dict(fill_value=2),
                       dict(fill_value=2, endwith=True),
                       dict(fill_value=2, endwith=False)]:
            # 对数组进行排序，并记录排序结果
            sortedx = sort(x, **kwargs)
            # 对数组进行argsort排序，并记录排序结果
            argsortedx = x[argsort(x, **kwargs)]
            # 断言排序后的数据部分相等
            assert_equal(sortedx._data, argsortedx._data)
            # 断言排序后的掩码部分相等
            assert_equal(sortedx._mask, argsortedx._mask)
    # 定义测试方法，测试二维数组的排序功能
    def test_sort_2d(self):
        # 检查对二维数组的排序，无遮罩
        a = masked_array([[8, 4, 1], [2, 0, 9]])
        # 按列排序
        a.sort(0)
        # 断言排序后数组的正确性
        assert_equal(a, [[2, 0, 1], [8, 4, 9]])
        
        # 重新初始化数组进行行排序
        a = masked_array([[8, 4, 1], [2, 0, 9]])
        a.sort(1)
        assert_equal(a, [[1, 4, 8], [0, 2, 9]])
        
        # 检查有遮罩的二维数组排序
        a = masked_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(0)
        assert_equal(a, [[2, 0, 1], [8, 4, 9]])
        # 断言排序后遮罩数组的正确性
        assert_equal(a._mask, [[0, 0, 0], [1, 0, 1]])
        
        # 重新初始化数组进行行排序
        a = masked_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(1)
        assert_equal(a, [[1, 4, 8], [0, 2, 9]])
        # 断言排序后遮罩数组的正确性
        assert_equal(a._mask, [[0, 0, 1], [0, 0, 1]])
        
        # 测试三维数组排序
        a = masked_array([[[7, 8, 9], [4, 5, 6], [1, 2, 3]],
                          [[1, 2, 3], [7, 8, 9], [4, 5, 6]],
                          [[7, 8, 9], [1, 2, 3], [4, 5, 6]],
                          [[4, 5, 6], [1, 2, 3], [7, 8, 9]]])
        # 将可被4整除的元素设置为遮罩
        a[a % 4 == 0] = masked
        am = a.copy()
        an = a.filled(99)
        # 按第一个轴排序
        am.sort(0)
        an.sort(0)
        # 断言排序结果的一致性
        assert_equal(am, an)
        
        am = a.copy()
        an = a.filled(99)
        # 按第二个轴排序
        am.sort(1)
        an.sort(1)
        # 断言排序结果的一致性
        assert_equal(am, an)
        
        am = a.copy()
        an = a.filled(99)
        # 按第三个轴排序
        am.sort(2)
        an.sort(2)
        # 断言排序结果的一致性
        assert_equal(am, an)

    # 测试灵活类型数据结构的排序
    def test_sort_flexible(self):
        a = array(
            data=[(3, 3), (3, 2), (2, 2), (2, 1), (1, 0), (1, 1), (1, 2)],
            mask=[(0, 0), (0, 1), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)],
            dtype=[('A', int), ('B', int)])
        mask_last = array(
            data=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2), (1, 0)],
            mask=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 0)],
            dtype=[('A', int), ('B', int)])
        mask_first = array(
            data=[(1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3)],
            mask=[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0)],
            dtype=[('A', int), ('B', int)])

        # 对数组进行排序，并断言结果正确
        test = sort(a)
        assert_equal(test, mask_last)
        assert_equal(test.mask, mask_last.mask)

        # 使用参数`endwith=False`对数组进行排序，并断言结果正确
        test = sort(a, endwith=False)
        assert_equal(test, mask_first)
        assert_equal(test.mask, mask_first.mask)

        # 测试具有子数组的数据类型排序（gh-8069）
        # 只需检查排序是否无误，结构化数组的子数组被视为字节字符串，因此排序行为会因字节顺序和`endwith`参数的不同而有所不同。
        dt = np.dtype([('v', int, 2)])
        a = a.view(dt)
        # 对数组进行排序
        test = sort(a)
        # 使用参数`endwith=False`对数组进行排序
        test = sort(a, endwith=False)

    # 测试argsort函数
    def test_argsort(self):
        # 测试argsort函数
        a = array([1, 5, 2, 4, 3], mask=[1, 0, 0, 1, 0])
        # 断言argsort结果的正确性
        assert_equal(np.argsort(a), argsort(a))
    def test_squeeze(self):
        # Check squeeze
        # 创建一个带掩码的数组，包含一个单行的数据
        data = masked_array([[1, 2, 3]])
        # 断言调用squeeze方法后的结果与预期的普通列表相同
        assert_equal(data.squeeze(), [1, 2, 3])
        # 创建一个带掩码的数组，包含一个单行的数据和一个全为真的掩码
        data = masked_array([[1, 2, 3]], mask=[[1, 1, 1]])
        # 断言调用squeeze方法后的结果与预期的普通列表相同
        assert_equal(data.squeeze(), [1, 2, 3])
        # 断言调用squeeze方法后的掩码属性与预期的掩码相同
        assert_equal(data.squeeze()._mask, [1, 1, 1])

        # 普通的 ndarray 返回一个视图
        arr = np.array([[1]])
        arr_sq = arr.squeeze()
        # 断言调用squeeze方法后的结果与预期的值相同
        assert_equal(arr_sq, 1)
        # 修改 arr_sq 的值
        arr_sq[...] = 2
        # 断言原始数组 arr 的值也被修改了
        assert_equal(arr[0,0], 2)

        # 因此，带掩码的数组也应该如此
        m_arr = masked_array([[1]], mask=True)
        m_arr_sq = m_arr.squeeze()
        # 断言调用squeeze方法后的结果不是 np.ma.masked
        assert_(m_arr_sq is not np.ma.masked)
        # 断言调用squeeze方法后的掩码属性与预期的掩码相同
        assert_equal(m_arr_sq.mask, True)
        # 修改 m_arr_sq 的值
        m_arr_sq[...] = 2
        # 断言原始数组 m_arr 的值也被修改了
        assert_equal(m_arr[0,0], 2)

    def test_swapaxes(self):
        # Tests swapaxes on MaskedArrays.
        # 创建一个普通的 numpy 数组 x
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        # 创建一个普通的 numpy 数组 m，用于创建带掩码的数组 mX
        m = np.array([0, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0])
        # 创建一个带掩码的数组 mX，并将其重塑为 6x6 的数组
        mX = array(x, mask=m).reshape(6, 6)
        # 将 mX 重塑为 3x2x2x3 的数组 mXX
        mXX = mX.reshape(3, 2, 2, 3)

        # 调用 swapaxes 方法交换轴 0 和 1
        mXswapped = mX.swapaxes(0, 1)
        # 断言交换后的结果与预期的一致
        assert_equal(mXswapped[-1], mX[:, -1])

        # 对 mXX 调用 swapaxes 方法交换轴 0 和 2
        mXXswapped = mXX.swapaxes(0, 2)
        # 断言交换后的形状与预期的一致
        assert_equal(mXXswapped.shape, (2, 2, 3, 3))

    def test_take(self):
        # Tests take
        # 创建一个带掩码的数组 x
        x = masked_array([10, 20, 30, 40], [0, 1, 0, 1])
        # 断言调用 take 方法后的结果与预期的带掩码数组一致
        assert_equal(x.take([0, 0, 3]), masked_array([10, 10, 40], [0, 0, 1]))
        # 断言调用 take 方法后的结果与预期的带掩码数组一致
        assert_equal(x.take([0, 0, 3]), x[[0, 0, 3]])
        # 断言调用 take 方法后的结果与预期的带掩码数组一致
        assert_equal(x.take([[0, 1], [0, 1]]),
                     masked_array([[10, 20], [10, 20]], [[0, 1], [0, 1]]))

        # 当传入 np.ma.mask 时，assert_equal 会报错
        assert_(x[1] is np.ma.masked)
        assert_(x.take(1) is np.ma.masked)

        # 创建一个普通数组 x，并带有一个掩码
        x = array([[10, 20, 30], [40, 50, 60]], mask=[[0, 0, 1], [1, 0, 0, ]])
        # 断言调用 take 方法后的结果与预期的带掩码数组一致
        assert_equal(x.take([0, 2], axis=1),
                     array([[10, 30], [40, 60]], mask=[[0, 1], [1, 0]]))
        # 断言调用 take 方法后的结果与预期的带掩码数组一致
        assert_equal(take(x, [0, 2], axis=1),
                     array([[10, 30], [40, 60]], mask=[[0, 1], [1, 0]]))
    def test_take_masked_indices(self):
        # 测试带有掩码索引的 take 函数

        # 创建一个包含五个元素的 NumPy 数组
        a = np.array((40, 18, 37, 9, 22))
        
        # 创建一个二维索引数组，用于获取 a 中的子集
        indices = np.arange(3)[None,:] + np.arange(5)[:, None]
        
        # 使用 indices 创建一个带有掩码的数组对象 mindices
        mindices = array(indices, mask=(indices >= len(a)))
        
        # 使用 take 函数，对 a 应用 mode='clip' 模式，获取测试结果
        test = take(a, mindices, mode='clip')
        
        # 创建控制结果数组 ctrl
        ctrl = array([[40, 18, 37],
                      [18, 37, 9],
                      [37, 9, 22],
                      [9, 22, 22],
                      [22, 22, 22]])
        
        # 断言测试结果与控制结果相等
        assert_equal(test, ctrl)
        
        # 使用 take 函数，对 a 应用默认模式，获取测试结果
        test = take(a, mindices)
        
        # 更新控制结果数组 ctrl，包含掩码值 masked
        ctrl = array([[40, 18, 37],
                      [18, 37, 9],
                      [37, 9, 22],
                      [9, 22, 40],
                      [22, 40, 40]])
        ctrl[3, 2] = ctrl[4, 1] = ctrl[4, 2] = masked
        
        # 断言测试结果与更新后的控制结果相等，并检查掩码
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        
        # 创建带有掩码输入的数组对象 a
        a = array((40, 18, 37, 9, 22), mask=(0, 1, 0, 0, 0))
        
        # 使用 take 函数，对带有掩码输入的 a 应用 mindices 获取测试结果
        test = take(a, mindices)
        
        # 更新控制结果数组 ctrl，包含掩码值 masked
        ctrl[0, 1] = ctrl[1, 0] = masked
        
        # 断言测试结果与更新后的控制结果相等，并检查掩码
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

    def test_tolist(self):
        # 测试 tolist 方法

        # 在一维数组上测试 tolist
        x = array(np.arange(12))
        
        # 将索引为 1 和倒数第二个元素设为 masked
        x[[1, -2]] = masked
        
        # 使用 tolist 方法转换为列表
        xlist = x.tolist()
        
        # 断言转换后的列表索引为 1 和倒数第二个元素为 None
        assert_(xlist[1] is None)
        assert_(xlist[-2] is None)
        
        # 在二维数组上测试 tolist
        x.shape = (3, 4)
        xlist = x.tolist()
        
        # 创建控制结果列表 ctrl
        ctrl = [[0, None, 2, 3], [4, 5, 6, 7], [8, 9, None, 11]]
        
        # 断言转换后的列表与控制结果列表相等
        assert_equal(xlist[0], [0, None, 2, 3])
        assert_equal(xlist[1], [4, 5, 6, 7])
        assert_equal(xlist[2], [8, 9, None, 11])
        assert_equal(xlist, ctrl)
        
        # 在包含掩码记录的结构化数组上测试 tolist
        x = array(list(zip([1, 2, 3],
                           [1.1, 2.2, 3.3],
                           ['one', 'two', 'thr'])),
                  dtype=[('a', int), ('b', float), ('c', '|S8')])
        
        # 将最后一个记录设为 masked
        x[-1] = masked
        
        # 断言转换后的列表与控制结果列表相等
        assert_equal(x.tolist(),
                     [(1, 1.1, b'one'),
                      (2, 2.2, b'two'),
                      (None, None, None)])
        
        # 在包含掩码字段的结构化数组上测试 tolist
        a = array([(1, 2,), (3, 4)], mask=[(0, 1), (0, 0)],
                  dtype=[('a', int), ('b', int)])
        
        # 使用 tolist 方法转换为列表
        test = a.tolist()
        
        # 创建控制结果列表 ctrl
        assert_equal(test, [[1, None], [3, 4]])
        
        # 在 mvoid 上测试 tolist
        a = a[0]
        test = a.tolist()
        
        # 断言转换后的列表与控制结果列表相等
        assert_equal(test, [1, None])
    def test_tolist_specialcase(self):
        # 测试 mvoid.tolist: 确保我们返回一个标准的 Python 对象
        a = array([(0, 1), (2, 3)], dtype=[('a', int), ('b', int)])
        # 没有掩码: 每个条目都是一个 np.void，其元素为标准的 Python 对象
        for entry in a:
            for item in entry.tolist():
                assert_(not isinstance(item, np.generic))
        # 有掩码: 每个条目都是一个 ma.void，其元素应该是标准的 Python 对象
        a.mask[0] = (0, 1)
        for entry in a:
            for item in entry.tolist():
                assert_(not isinstance(item, np.generic))

    def test_toflex(self):
        # 测试转换为 records
        data = arange(10)
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        ndtype = [('i', int), ('s', '|S3'), ('f', float)]
        data = array([(i, s, f) for (i, s, f) in zip(np.arange(10),
                                                     'ABCDEFGHIJKLM',
                                                     np.random.rand(10))],
                     dtype=ndtype)
        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        ndtype = np.dtype("int, (2,3)float, float")
        data = array([(i, f, ff) for (i, f, ff) in zip(np.arange(10),
                                                       np.random.rand(10),
                                                       np.random.rand(10))],
                     dtype=ndtype)
        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal_records(record['_data'], data._data)
        assert_equal_records(record['_mask'], data._mask)

    def test_fromflex(self):
        # 测试从记录重建带掩码数组
        a = array([1, 2, 3])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.mask, a.mask)

        a = array([1, 2, 3], mask=[0, 0, 1])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.mask, a.mask)

        a = array([(1, 1.), (2, 2.), (3, 3.)], mask=[(1, 0), (0, 0), (0, 1)],
                  dtype=[('A', int), ('B', float)])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.data, a.data)
    # 定义一个测试方法 test_arraymethod，用于测试 _arraymethod 方法，接受 self 参数
    def test_arraymethod(self):
        # 创建一个带掩码的数组 marray，掩码指定了数组中哪些元素被屏蔽
        marray = masked_array([[1, 2, 3, 4, 5]], mask=[0, 0, 1, 0, 0])
        # 创建一个掩码数组 control，与 marray 的转置结果相对应
        control = masked_array([[1], [2], [3], [4], [5]],
                               mask=[0, 0, 1, 0, 0])
        # 断言 marray 的转置结果等于 control，即验证转置操作是否正确
        assert_equal(marray.T, control)
        # 断言 marray 的 transpose 方法得到的结果等于 control，验证 transpose 方法是否正确
        assert_equal(marray.transpose(), control)

        # 使用 MaskedArray 的 cumsum 方法对 marray.T 沿着 axis=0 进行累加求和，结果与 control 的累加求和结果比较
        assert_equal(MaskedArray.cumsum(marray.T, 0), control.cumsum(0))

    # 定义一个测试方法 test_arraymethod_0d，用于测试对 0 维度的数组操作
    def test_arraymethod_0d(self):
        # 创建一个包含单个元素的掩码数组 x，元素值为 42，且被掩盖
        x = np.ma.array(42, mask=True)
        # 断言 x 的转置掩码与 x 的掩码相等
        assert_equal(x.T.mask, x.mask)
        # 断言 x 的转置数据与 x 的数据相等
        assert_equal(x.T.data, x.data)

    # 定义一个测试方法 test_transpose_view，用于测试数组转置视图
    def test_transpose_view(self):
        # 创建一个二维掩码数组 x
        x = np.ma.array([[1, 2, 3], [4, 5, 6]])
        # 将 x 的 (0,1) 位置的元素设置为掩盖值
        x[0,1] = np.ma.masked
        # 获取 x 的转置视图 xt
        xt = x.T

        # 修改 xt 的 (1,0) 位置的元素为 10
        xt[1,0] = 10
        # 将 xt 的 (0,1) 位置的元素设置为掩盖值
        xt[0,1] = np.ma.masked

        # 断言 x 的数据与 xt 的转置数据相等
        assert_equal(x.data, xt.T.data)
        # 断言 x 的掩码与 xt 的转置掩码相等
        assert_equal(x.mask, xt.T.mask)

    # 定义一个测试方法 test_diagonal_view，用于测试对角线视图
    def test_diagonal_view(self):
        # 创建一个全零掩码数组 x，形状为 (3,3)
        x = np.ma.zeros((3,3))
        # 将 x 的 (0,0) 位置的元素设置为 10
        x[0,0] = 10
        # 将 x 的 (1,1) 位置的元素设置为掩盖值
        x[1,1] = np.ma.masked
        # 将 x 的 (2,2) 位置的元素设置为 20
        x[2,2] = 20
        # 获取 x 的对角线视图 xd
        xd = x.diagonal()
        # 修改 x 的 (1,1) 位置的元素为 15
        x[1,1] = 15
        # 断言 xd 的掩码与 x 的对角线视图的掩码相等
        assert_equal(xd.mask, x.diagonal().mask)
        # 断言 xd 的数据与 x 的对角线视图的数据相等
        assert_equal(xd.data, x.diagonal().data)
class TestMaskedArrayMathMethods:

    def setup_method(self):
        # 定义基础数据。
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = array(data=x, mask=m2)
        m2X = array(data=X, mask=m2.reshape(X.shape))
        m2XX = array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_cumsumprod(self):
        # 测试在 MaskedArray 上的 cumsum 和 cumprod 方法。
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        
        # 在第一个维度上计算累积和，并验证结果是否与填充后的 MaskedArray 的累积和相等。
        mXcp = mX.cumsum(0)
        assert_equal(mXcp._data, mX.filled(0).cumsum(0))
        
        # 在第二个维度上计算累积和，并验证结果是否与填充后的 MaskedArray 的累积和相等。
        mXcp = mX.cumsum(1)
        assert_equal(mXcp._data, mX.filled(0).cumsum(1))

        # 在第一个维度上计算累积乘积，并验证结果是否与填充后的 MaskedArray 的累积乘积相等。
        mXcp = mX.cumprod(0)
        assert_equal(mXcp._data, mX.filled(1).cumprod(0))
        
        # 在第二个维度上计算累积乘积，并验证结果是否与填充后的 MaskedArray 的累积乘积相等。
        mXcp = mX.cumprod(1)
        assert_equal(mXcp._data, mX.filled(1).cumprod(1))

    def test_cumsumprod_with_output(self):
        # 测试带有输出参数的 cumsum 和 cumprod 方法。
        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked

        for funcname in ('cumsum', 'cumprod'):
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)

            # 使用 ndarray 作为显式输入参数
            output = np.empty((3, 4), dtype=float)
            output.fill(-9999)
            result = npfunc(xm, axis=0, out=output)
            # 结果应与给定的输出相等
            assert_(result is output)
            assert_equal(result, xmmeth(axis=0, out=output))

            output = empty((3, 4), dtype=int)
            result = xmmeth(axis=0, out=output)
            assert_(result is output)
    def test_ptp(self):
        # Tests ptp on MaskedArrays.
        # 解构元组 self.d
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        # 获取 X 的形状
        (n, m) = X.shape
        # 断言 mx 的峰值到峰值范围与 mx 中压缩数据的峰值到峰值范围相等
        assert_equal(mx.ptp(), np.ptp(mx.compressed()))
        # 创建一个大小为 n 的全零数组 rows 和一个大小为 m 的全零数组 cols
        rows = np.zeros(n, float)
        cols = np.zeros(m, float)
        # 计算 mX 每列的压缩数据的峰值到峰值范围并存储在 cols 数组中
        for k in range(m):
            cols[k] = np.ptp(mX[:, k].compressed())
        # 计算 mX 每行的压缩数据的峰值到峰值范围并存储在 rows 数组中
        for k in range(n):
            rows[k] = np.ptp(mX[k].compressed())
        # 断言 mX 按列计算的峰值到峰值范围等于 cols
        assert_equal(mX.ptp(0), cols)
        # 断言 mX 按行计算的峰值到峰值范围等于 rows
        assert_equal(mX.ptp(1), rows)

    def test_add_object(self):
        # 创建一个包含对象 'a', 'b' 的 MaskedArray，并对其进行加法运算
        x = masked_array(['a', 'b'], mask=[1, 0], dtype=object)
        y = x + 'x'
        # 断言 y 的第二个元素等于 'bx'
        assert_equal(y[1], 'bx')
        # 断言 y 的第一个元素被屏蔽
        assert_(y.mask[0])

    def test_sum_object(self):
        # 测试对象类型的数组求和
        a = masked_array([1, 2, 3], mask=[1, 0, 0], dtype=object)
        assert_equal(a.sum(), 5)
        a = masked_array([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.sum(axis=0), [5, 7, 9])

    def test_prod_object(self):
        # 测试对象类型的数组求积
        a = masked_array([1, 2, 3], mask=[1, 0, 0], dtype=object)
        assert_equal(a.prod(), 2 * 3)
        a = masked_array([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.prod(axis=0), [4, 10, 18])

    def test_meananom_object(self):
        # 测试对象类型的数组均值和偏差
        a = masked_array([1, 2, 3], dtype=object)
        assert_equal(a.mean(), 2)
        assert_equal(a.anom(), [-1, 0, 1])

    def test_anom_shape(self):
        # 测试对象类型的数组偏差的形状
        a = masked_array([1, 2, 3])
        assert_equal(a.anom().shape, a.shape)
        a.mask = True
        assert_equal(a.anom().shape, a.shape)
        assert_(np.ma.is_masked(a.anom()))

    def test_anom(self):
        # 测试 MaskedArray 的偏差计算
        a = masked_array(np.arange(1, 7).reshape(2, 3))
        assert_almost_equal(a.anom(),
                            [[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
        assert_almost_equal(a.anom(axis=0),
                            [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        assert_almost_equal(a.anom(axis=1),
                            [[-1., 0., 1.], [-1., 0., 1.]])
        # 屏蔽部分数据后，测试偏差计算的结果
        a.mask = [[0, 0, 1], [0, 1, 0]]
        mval = -99
        assert_almost_equal(a.anom().filled(mval),
                            [[-2.25, -1.25, mval], [0.75, mval, 2.75]])
        assert_almost_equal(a.anom(axis=0).filled(mval),
                            [[-1.5, 0.0, mval], [1.5, mval, 0.0]])
        assert_almost_equal(a.anom(axis=1).filled(mval),
                            [[-0.5, 0.5, mval], [-1.0, mval, 1.0]])
    def test_trace(self):
        # Tests trace on MaskedArrays.
        # 解构赋值以获取测试数据元组中的各个变量
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        # 获取 mX 对角线上的非屏蔽元素之和，并断言与 mX 的迹相等
        mXdiag = mX.diagonal()
        assert_equal(mX.trace(), mX.diagonal().compressed().sum())
        # 断言 mX 的迹与 X 的迹减去 mX 对角线上屏蔽元素乘积的和近似相等
        assert_almost_equal(mX.trace(),
                            X.trace() - sum(mXdiag.mask * X.diagonal(),
                                            axis=0))
        # 断言 np.trace(mX) 等于 mX 的迹

        assert_equal(np.trace(mX), mX.trace())

        # gh-5560
        # 创建一个形状为 (2, 4, 4) 的三维数组 arr，并生成一个相应的 MaskedArray m_arr
        arr = np.arange(2*4*4).reshape(2,4,4)
        m_arr = np.ma.masked_array(arr, False)
        # 断言 arr 沿第 1 和第 2 轴的迹与 m_arr 相同
        assert_equal(arr.trace(axis1=1, axis2=2), m_arr.trace(axis1=1, axis2=2))

    def test_dot(self):
        # Tests dot on MaskedArrays.
        # 解构赋值以获取测试数据元组中的各个变量
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        # 使用零填充的 mx 计算自身的点积，并断言结果近似等于原始数据的点积
        fx = mx.filled(0)
        r = mx.dot(mx)
        assert_almost_equal(r.filled(0), fx.dot(fx))
        # 断言 r 的掩码为空掩码
        assert_(r.mask is nomask)

        # 使用零填充的 mX 计算自身的点积，并断言结果近似等于原始数据的点积
        fX = mX.filled(0)
        r = mX.dot(mX)
        assert_almost_equal(r.filled(0), fX.dot(fX))
        # 断言 r 的特定位置有掩码
        assert_(r.mask[1,3])
        # 创建一个形状与 r 相同的空数组 r1，计算 mX 与 mX 的点积，并断言结果近似等于 r1
        r1 = empty_like(r)
        mX.dot(mX, out=r1)
        assert_almost_equal(r, r1)

        # 将 mXX 的最后两个轴交换，创建填充零的 mYY 和 fYY，并计算 mXX 与 mYY 的点积
        mYY = mXX.swapaxes(-1, -2)
        fXX, fYY = mXX.filled(0), mYY.filled(0)
        r = mXX.dot(mYY)
        assert_almost_equal(r.filled(0), fXX.dot(fYY))
        # 创建一个形状与 r 相同的空数组 r1，计算 mXX 与 mYY 的点积，并断言结果近似等于 r1
        r1 = empty_like(r)
        mXX.dot(mYY, out=r1)
        assert_almost_equal(r, r1)

    def test_dot_shape_mismatch(self):
        # regression test
        # 创建两个形状相同的 MaskedArray x 和 y，以及一个形状不同的 MaskedArray z
        x = masked_array([[1,2],[3,4]], mask=[[0,1],[0,0]])
        y = masked_array([[1,2],[3,4]], mask=[[0,1],[0,0]])
        z = masked_array([[0,1],[3,3]])
        # 计算 x 与 y 的点积，将结果存入 z，并断言结果近似等于预期的值和掩码
        x.dot(y, out=z)
        assert_almost_equal(z.filled(0), [[1, 0], [15, 16]])
        assert_almost_equal(z.mask, [[0, 1], [0, 0]])

    def test_varmean_nomask(self):
        # gh-5769
        # 创建两个浮点类型的数组 foo 和 bar
        foo = array([1,2,3,4], dtype='f8')
        bar = array([1,2,3,4], dtype='f8')
        # 断言 foo.mean() 和 foo.var() 的类型是 np.float64
        assert_equal(type(foo.mean()), np.float64)
        assert_equal(type(foo.var()), np.float64)
        # 断言 foo.mean() 等于 bar.mean() 是 True 类型的布尔值

        assert((foo.mean() == bar.mean()) is np.bool(True))

        # 检查数组类型是否保留，并验证 out 参数是否有效
        # 创建一个形状为 (4, 4) 的二维数组 foo，并创建一个相同形状的空数组 bar
        foo = array(np.arange(16).reshape((4,4)), dtype='f8')
        bar = empty(4, dtype='f4')
        # 断言 foo.mean(axis=1) 和 foo.var(axis=1) 的类型是 MaskedArray
        assert_equal(type(foo.mean(axis=1)), MaskedArray)
        assert_equal(type(foo.var(axis=1)), MaskedArray)
        # 断言 foo.mean(axis=1, out=bar) 和 foo.var(axis=1, out=bar) 返回的是 bar 自身
        assert_(foo.mean(axis=1, out=bar) is bar)
        assert_(foo.var(axis=1, out=bar) is bar)
    def test_varstd(self):
        # Tests var & std on MaskedArrays.
        # 获取测试数据元组的各个元素
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        # 检查未压缩的 mX 的方差是否与压缩后的一致
        assert_almost_equal(mX.var(axis=None), mX.compressed().var())
        # 检查未压缩的 mX 的标准差是否与压缩后的一致
        assert_almost_equal(mX.std(axis=None), mX.compressed().std())
        # 检查未压缩的 mX 的标准差（自由度为 1）是否与压缩后的一致
        assert_almost_equal(mX.std(axis=None, ddof=1),
                            mX.compressed().std(ddof=1))
        # 检查未压缩的 mX 的方差（自由度为 1）是否与压缩后的一致
        assert_almost_equal(mX.var(axis=None, ddof=1),
                            mX.compressed().var(ddof=1))
        # 检查 mXX 在第 3 维上的方差形状是否与 XX 的一致
        assert_equal(mXX.var(axis=3).shape, XX.var(axis=3).shape)
        # 检查 mX 的方差形状是否与 X 的一致
        assert_equal(mX.var().shape, X.var().shape)
        # 获取 mX 按轴 0 和 1 计算的方差结果
        (mXvar0, mXvar1) = (mX.var(axis=0), mX.var(axis=1))
        # 检查未压缩的 mX 的方差（自由度为 2）是否与压缩后的一致
        assert_almost_equal(mX.var(axis=None, ddof=2),
                            mX.compressed().var(ddof=2))
        # 检查未压缩的 mX 的标准差（自由度为 2）是否与压缩后的一致
        assert_almost_equal(mX.std(axis=None, ddof=2),
                            mX.compressed().std(ddof=2))
        # 遍历范围为 0 到 5 的 k 值，检查 mXvar1[k] 是否与 mX[k] 的压缩后方差一致
        for k in range(6):
            assert_almost_equal(mXvar1[k], mX[k].compressed().var())
            # 检查 mXvar0[k] 是否与 mX[:, k] 的压缩后方差一致
            assert_almost_equal(mXvar0[k], mX[:, k].compressed().var())
            # 检查 mXvar0[k] 的平方根是否与 mX[:, k] 的压缩后标准差一致
            assert_almost_equal(np.sqrt(mXvar0[k]),
                                mX[:, k].compressed().std())

    @suppress_copy_mask_on_assignment
    def test_varstd_specialcases(self):
        # Test a special case for var
        # 创建用于输出的特殊情况下的数组
        nout = np.array(-1, dtype=float)
        mout = array(-1, dtype=float)

        # 创建一个包含遮蔽值的数组 x
        x = array(arange(10), mask=True)
        # 针对 'var' 和 'std' 方法分别进行测试
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            # 检查未指定参数时方法的输出是否为遮蔽值
            assert_(method() is masked)
            # 检查指定 ddof=0 参数时方法的输出是否为遮蔽值
            assert_(method(0) is masked)
            # 检查指定 ddof=-1 参数时方法的输出是否为遮蔽值
            assert_(method(-1) is masked)
            # 使用遮蔽数组作为显式输出时，检查方法的输出是否不为遮蔽值
            method(out=mout)
            assert_(mout is not masked)
            # 检查输出数组的遮蔽值是否为真
            assert_equal(mout.mask, True)
            # 使用 ndarray 作为显式输出时，检查输出是否为 NaN
            method(out=nout)
            assert_(np.isnan(nout))

        # 修改 x 数组最后一个元素的值
        x = array(arange(10), mask=True)
        x[-1] = 9
        # 再次针对 'var' 和 'std' 方法分别进行测试
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            # 检查指定 ddof=1 参数时方法的输出是否为遮蔽值
            assert_(method(ddof=1) is masked)
            # 检查指定 ddof=0 参数时方法的输出是否为遮蔽值
            assert_(method(0, ddof=1) is masked)
            # 检查指定 ddof=-1 参数时方法的输出是否为遮蔽值
            assert_(method(-1, ddof=1) is masked)
            # 使用遮蔽数组作为显式输出时，检查方法的输出是否不为遮蔽值
            method(out=mout, ddof=1)
            assert_(mout is not masked)
            # 检查输出数组的遮蔽值是否为真
            assert_equal(mout.mask, True)
            # 使用 ndarray 作为显式输出时，检查输出是否为 NaN
            method(out=nout, ddof=1)
            assert_(np.isnan(nout))

    def test_varstd_ddof(self):
        # 创建一个带有遮蔽值的数组 a
        a = array([[1, 1, 0], [1, 1, 0]], mask=[[0, 0, 1], [0, 0, 1]])
        # 检查指定 ddof=0 时在轴 0 上的标准差结果
        test = a.std(axis=0, ddof=0)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        # 检查指定 ddof=1 时在轴 0 上的标准差结果
        test = a.std(axis=0, ddof=1)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        # 检查指定 ddof=2 时在轴 0 上的标准差结果
        test = a.std(axis=0, ddof=2)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [1, 1, 1])
    def test_diag(self):
        # 测试对角线提取函数
        x = arange(9).reshape((3, 3))  # 创建一个3x3的数组
        x[1, 1] = masked  # 将数组中索引为(1, 1)的元素设为masked
        out = np.diag(x)  # 提取数组的对角线元素
        assert_equal(out, [0, 4, 8])  # 检查对角线提取结果是否正确
        out = diag(x)  # 使用别名函数再次提取对角线元素
        assert_equal(out, [0, 4, 8])  # 检查别名函数的结果是否正确
        assert_equal(out.mask, [0, 1, 0])  # 检查结果的遮罩(mask)是否正确设置
        out = diag(out)  # 对已有的对角线数组再次提取对角线
        control = array([[0, 0, 0], [0, 4, 0], [0, 0, 8]],  # 期望的对角线数组结果
                        mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])  # 期望的对角线遮罩结果
        assert_equal(out, control)  # 检查最终提取结果是否与期望一致

    def test_axis_methods_nomask(self):
        # 测试不带遮罩的数组方法与轴的组合使用
        a = array([[1, 2, 3], [4, 5, 6]])  # 创建一个二维数组

        assert_equal(a.sum(0), [5, 7, 9])  # 检查按列求和的结果
        assert_equal(a.sum(-1), [6, 15])  # 检查按行求和的结果
        assert_equal(a.sum(1), [6, 15])  # 再次检查按行求和的结果

        assert_equal(a.prod(0), [4, 10, 18])  # 检查按列求积的结果
        assert_equal(a.prod(-1), [6, 120])  # 检查按行求积的结果
        assert_equal(a.prod(1), [6, 120])  # 再次检查按行求积的结果

        assert_equal(a.min(0), [1, 2, 3])  # 检查按列求最小值的结果
        assert_equal(a.min(-1), [1, 4])  # 检查按行求最小值的结果
        assert_equal(a.min(1), [1, 4])  # 再次检查按行求最小值的结果

        assert_equal(a.max(0), [4, 5, 6])  # 检查按列求最大值的结果
        assert_equal(a.max(-1), [3, 6])  # 检查按行求最大值的结果
        assert_equal(a.max(1), [3, 6])  # 再次检查按行求最大值的结果

    @requires_memory(free_bytes=2 * 10000 * 1000 * 2)
    def test_mean_overflow(self):
        # 测试带有遮罩数组中的溢出情况
        # gh-20272
        a = masked_array(np.full((10000, 10000), 65535, dtype=np.uint16),  # 创建一个大数组，填充为65535
                         mask=np.zeros((10000, 10000)))  # 创建一个与数组大小相同的全0遮罩
        assert_equal(a.mean(), 65535.0)  # 检查数组的均值是否为65535.0

    def test_diff_with_prepend(self):
        # GH 22465
        x = np.array([1, 2, 2, 3, 4, 2, 1, 1])  # 创建一个数组

        a = np.ma.masked_equal(x[3:], value=2)  # 创建一个从索引3开始的部分遮罩数组
        a_prep = np.ma.masked_equal(x[:3], value=2)  # 创建一个从索引0到索引2的部分遮罩数组
        diff1 = np.ma.diff(a, prepend=a_prep, axis=0)  # 对数组进行差分运算，使用前置遮罩数组

        b = np.ma.masked_equal(x, value=2)  # 创建整体遮罩数组
        diff2 = np.ma.diff(b, axis=0)  # 对整体遮罩数组进行差分运算

        assert_(np.ma.allequal(diff1, diff2))  # 检查两次差分运算结果是否完全相等

    def test_diff_with_append(self):
        # GH 22465
        x = np.array([1, 2, 2, 3, 4, 2, 1, 1])  # 创建一个数组

        a = np.ma.masked_equal(x[:3], value=2)  # 创建一个从索引0到索引2的部分遮罩数组
        a_app = np.ma.masked_equal(x[3:], value=2)  # 创建一个从索引3开始的部分遮罩数组
        diff1 = np.ma.diff(a, append=a_app, axis=0)  # 对数组进行差分运算，使用后置遮罩数组

        b = np.ma.masked_equal(x, value=2)  # 创建整体遮罩数组
        diff2 = np.ma.diff(b, axis=0)  # 对整体遮罩数组进行差分运算

        assert_(np.ma.allequal(diff1, diff2))  # 检查两次差分运算结果是否完全相等

    def test_diff_with_dim_0(self):
        with pytest.raises(
            ValueError,
            match="diff requires input that is at least one dimensional"
            ):
            np.ma.diff(np.array(1))  # 对一个零维数组进行差分运算，预期引发异常

    def test_diff_with_n_0(self):
        a = np.ma.masked_equal([1, 2, 2, 3, 4, 2, 1, 1], value=2)  # 创建一个部分遮罩数组
        diff = np.ma.diff(a, n=0, axis=0)  # 对数组进行差分运算，n=0

        assert_(np.ma.allequal(a, diff))  # 检查差分运算结果是否与原数组完全相等
# 定义一个测试类，用于测试 MaskedArray 的复杂数学方法
class TestMaskedArrayMathMethodsComplex:

    # 在每个测试方法执行前执行的设置方法
    def setup_method(self):
        # 定义基础数据
        x = np.array([8.375j, 7.545j, 8.828j, 8.5j, 1.757j, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479j,
                      7.189j, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993j])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        # 创建带屏蔽数组的对象 mx，mX，mXX
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        # 创建第二组带屏蔽数组的对象 m2x，m2X，m2XX
        m2x = array(data=x, mask=m2)
        m2X = array(data=X, mask=m2.reshape(X.shape))
        m2XX = array(data=XX, mask=m2.reshape(XX.shape))
        # 将所有数据存入实例变量 d 中
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    # 测试 MaskedArrays 上的 var 和 std 方法
    def test_varstd(self):
        # 从实例变量 d 中解包出所需的变量
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        # 检查未压缩的 mX 的 var 方法与压缩后的 var 方法的准确性
        assert_almost_equal(mX.var(axis=None), mX.compressed().var())
        # 检查未压缩的 mX 的 std 方法与压缩后的 std 方法的准确性
        assert_almost_equal(mX.std(axis=None), mX.compressed().std())
        # 检查 mXX 的 var 方法在指定轴上的形状与 XX 的 var 方法的形状是否相同
        assert_equal(mXX.var(axis=3).shape, XX.var(axis=3).shape)
        # 检查 mX 的 var 方法的形状与 X 的 var 方法的形状是否相同
        assert_equal(mX.var().shape, X.var().shape)
        # 解包出 mXvar0 和 mXvar1，分别是按行和按列计算的 var 结果
        (mXvar0, mXvar1) = (mX.var(axis=0), mX.var(axis=1))
        # 检查未压缩的 mX 的 var 方法（带 ddof 参数）与压缩后的 var 方法的准确性
        assert_almost_equal(mX.var(axis=None, ddof=2),
                            mX.compressed().var(ddof=2))
        # 检查未压缩的 mX 的 std 方法（带 ddof 参数）与压缩后的 std 方法的准确性
        assert_almost_equal(mX.std(axis=None, ddof=2),
                            mX.compressed().std(ddof=2))
        # 对于每个列 k，检查按行和按列压缩后的 var 和 std 方法的准确性
        for k in range(6):
            assert_almost_equal(mXvar1[k], mX[k].compressed().var())
            assert_almost_equal(mXvar0[k], mX[:, k].compressed().var())
            assert_almost_equal(np.sqrt(mXvar0[k]),
                                mX[:, k].compressed().std())


# 定义一个测试类，用于测试 MaskedArray 的函数
class TestMaskedArrayFunctions:

    # 在每个测试方法执行前执行的设置方法
    def setup_method(self):
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        # 创建带屏蔽数组的对象 xm 和 ym
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        # 设置 xm 的填充值
        xm.set_fill_value(1e+20)
        # 将 xm 和 ym 存入实例变量 info 中
        self.info = (xm, ym)
    # 测试函数：masked_where_bool
    def test_masked_where_bool(self):
        # 定义列表 x
        x = [1, 2]
        # 调用 masked_where 函数创建 masked 数组 y，使其不被掩盖
        y = masked_where(False, x)
        # 断言 y 应与原始列表 x 相等
        assert_equal(y, [1, 2])
        # 断言 y 中的第二个元素应为 2
        assert_equal(y[1], 2)

    # 测试函数：test_masked_equal_wlist
    def test_masked_equal_wlist(self):
        # 定义列表 x
        x = [1, 2, 3]
        # 使用 masked_equal 函数将列表 x 中的值为 3 的元素掩盖
        mx = masked_equal(x, 3)
        # 断言 mx 应与原始列表 x 相等
        assert_equal(mx, x)
        # 断言 mx 的掩码属性 _mask 应为 [0, 0, 1]
        assert_equal(mx._mask, [0, 0, 1])
        # 使用 masked_not_equal 函数将列表 x 中值不为 3 的元素掩盖
        mx = masked_not_equal(x, 3)
        # 断言 mx 应与原始列表 x 相等
        assert_equal(mx, x)
        # 断言 mx 的掩码属性 _mask 应为 [1, 1, 0]

    # 测试函数：test_masked_equal_fill_value
    def test_masked_equal_fill_value(self):
        # 定义列表 x
        x = [1, 2, 3]
        # 使用 masked_equal 函数将列表 x 中的值为 3 的元素掩盖
        mx = masked_equal(x, 3)
        # 断言 mx 的掩码属性 _mask 应为 [0, 0, 1]
        assert_equal(mx._mask, [0, 0, 1])
        # 断言 mx 的填充值属性 fill_value 应为 3
        assert_equal(mx.fill_value, 3)

    # 测试函数：test_masked_where_condition
    def test_masked_where_condition(self):
        # 测试掩盖函数的条件
        # 创建浮点数数组 x
        x = array([1., 2., 3., 4., 5.])
        # 在数组 x 中将索引为 2 的元素设置为 masked
        x[2] = masked
        # 断言调用 masked_where 函数，应与调用 masked_greater 函数结果相等
        assert_equal(masked_where(greater(x, 2), x), masked_greater(x, 2))
        # 断言调用 masked_where 函数，应与调用 masked_greater_equal 函数结果相等
        assert_equal(masked_where(greater_equal(x, 2), x),
                     masked_greater_equal(x, 2))
        # 断言调用 masked_where 函数，应与调用 masked_less 函数结果相等
        assert_equal(masked_where(less(x, 2), x), masked_less(x, 2))
        # 断言调用 masked_where 函数，应与调用 masked_less_equal 函数结果相等
        assert_equal(masked_where(less_equal(x, 2), x),
                     masked_less_equal(x, 2))
        # 断言调用 masked_where 函数，应与调用 masked_not_equal 函数结果相等
        assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2))
        # 断言调用 masked_where 函数，应与调用 masked_equal 函数结果相等
        assert_equal(masked_where(equal(x, 2), x), masked_equal(x, 2))
        # 再次断言调用 masked_where 函数，应与调用 masked_not_equal 函数结果相等
        assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2))
        # 断言调用 masked_where 函数，应返回列表 [99, 99, 3, 4, 5]
        assert_equal(masked_where([1, 1, 0, 0, 0], [1, 2, 3, 4, 5]),
                     [99, 99, 3, 4, 5])

    # 测试函数：test_masked_where_oddities
    def test_masked_where_oddities(self):
        # 测试一些通用特性
        # 创建形状为 (10, 10, 10) 的全一浮点数组 atest
        atest = ones((10, 10, 10), dtype=float)
        # 创建与 atest 形状相同的零数组 btest，类型为 MaskType
        btest = zeros(atest.shape, MaskType)
        # 调用 masked_where 函数，根据 btest 对 atest 进行掩盖
        ctest = masked_where(btest, atest)
        # 断言 atest 应与 ctest 相等
        assert_equal(atest, ctest)

    # 测试函数：test_masked_where_shape_constraint
    def test_masked_where_shape_constraint(self):
        # 创建数组 a，包含从 0 到 9 的整数
        a = arange(10)
        # 使用 assert_raises 断言在调用 masked_equal 函数时会引发 IndexError 异常
        with assert_raises(IndexError):
            masked_equal(1, a)
        # 调用 masked_equal 函数，将数组 a 中值为 1 的元素掩盖
        test = masked_equal(a, 1)
        # 断言 test 的掩码属性 mask 应为 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        assert_equal(test.mask, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    # 测试函数：test_masked_where_structured
    def test_masked_where_structured(self):
        # 测试在结构化数组上使用 masked_where 是否设置了结构化掩码（参见问题 #2972）
        # 创建结构化数组 a，包含 10 个元素，每个元素包含两个字段 A 和 B，分别为 float16 和 float32 类型
        a = np.zeros(10, dtype=[("A", "<f2"), ("B", "<f4")])
        # 使用 np.errstate 忽略溢出警告
        with np.errstate(over="ignore"):
            # 调用 masked_where 函数，在数组 a 的字段 A 小于 5 的位置进行掩盖
            am = np.ma.masked_where(a["A"] < 5, a)
        # 断言 am 的掩码属性 mask 的字段名应与 am 的 dtype 的字段名相同
        assert_equal(am.mask.dtype.names, am.dtype.names)
        # 断言 am 的字段 A 应为一个 masked_array，其掩码为全一数组，数据为全零数组
        assert_equal(am["A"],
                    np.ma.masked_array(np.zeros(10), np.ones(10)))

    # 测试函数：test_masked_where_mismatch
    def test_masked_where_mismatch(self):
        # 测试 gh-4520
        # 创建数组 x，包含从 0 到 9 的整数
        x = np.arange(10)
        # 创建数组 y，包含从 0 到 4 的整数
        y = np.arange(5)
        # 使用 assert_raises 断言在调用 masked_where 函数时会引发 IndexError 异常
        assert_raises(IndexError, np.ma.masked_where, y > 6, x)
    # 测试函数，用于测试 masked_inside 函数和 masked_outside 函数
    def test_masked_otherfunctions(self):
        # 断言测试 masked_inside 函数，检查其返回值是否符合预期
        assert_equal(masked_inside(list(range(5)), 1, 3),
                     [0, 199, 199, 199, 4])
        # 断言测试 masked_outside 函数，检查其返回值是否符合预期
        assert_equal(masked_outside(list(range(5)), 1, 3), [199, 1, 2, 3, 199])
        # 使用 mask 参数调用 masked_inside 函数，检查其返回的 mask 属性是否符合预期
        assert_equal(masked_inside(array(list(range(5)),
                                         mask=[1, 0, 0, 0, 0]), 1, 3).mask,
                     [1, 1, 1, 1, 0])
        # 使用 mask 参数调用 masked_outside 函数，检查其返回的 mask 属性是否符合预期
        assert_equal(masked_outside(array(list(range(5)),
                                          mask=[0, 1, 0, 0, 0]), 1, 3).mask,
                     [1, 1, 0, 0, 1])
        # 使用 mask 参数调用 masked_equal 函数，检查其返回的 mask 属性是否符合预期
        assert_equal(masked_equal(array(list(range(5)),
                                        mask=[1, 0, 0, 0, 0]), 2).mask,
                     [1, 0, 1, 0, 0])
        # 使用 mask 参数调用 masked_not_equal 函数，检查其返回的 mask 属性是否符合预期
        assert_equal(masked_not_equal(array([2, 2, 1, 2, 1],
                                            mask=[1, 0, 0, 0, 0]), 2).mask,
                     [1, 0, 1, 0, 1])

    # 测试 round 函数及相关功能
    def test_round(self):
        # 创建带 mask 的数组 a
        a = array([1.23456, 2.34567, 3.45678, 4.56789, 5.67890],
                  mask=[0, 1, 0, 0, 0])
        # 断言测试 a.round()，检查其返回值是否符合预期
        assert_equal(a.round(), [1., 2., 3., 5., 6.])
        # 断言测试带有小数点位数的 a.round(decimals=1)，检查其返回值是否符合预期
        assert_equal(a.round(1), [1.2, 2.3, 3.5, 4.6, 5.7])
        # 断言测试带有小数点位数的 a.round(decimals=3)，检查其返回值是否符合预期
        assert_equal(a.round(3), [1.235, 2.346, 3.457, 4.568, 5.679])
        # 创建空数组 b，用 a.round(out=b) 进行原位舍入操作，检查 b 是否符合预期
        b = empty_like(a)
        a.round(out=b)
        assert_equal(b, [1., 2., 3., 5., 6.])

        # 创建普通数组 x 和掩码数组 c，进行 where 函数的测试
        x = array([1., 2., 3., 4., 5.])
        c = array([1, 1, 1, 0, 0])
        # 将 x 中的第二个元素设为 masked
        x[2] = masked
        # 使用 where 函数根据 c 的值选择 x 或 -x，检查返回值是否符合预期
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        # 将 c 中的第一个元素设为 masked，再次测试 where 函数的返回值
        c[0] = masked
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        # 断言测试 z 中的第一个元素是否为 masked
        assert_(z[0] is masked)
        # 断言测试 z 中的第二个元素是否不为 masked
        assert_(z[1] is not masked)
        # 断言测试 z 中的第三个元素是否为 masked
        assert_(z[2] is masked)

    # 测试 round 函数的输出参数
    def test_round_with_output(self):
        # 测试带有显式输出的 round 函数

        # 创建随机数组 xm，并设置部分值为 masked
        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked

        # 创建一个输出数组 output，并填充初始值 -9999
        output = np.empty((3, 4), dtype=float)
        output.fill(-9999)
        # 使用 xm.round(decimals=2, out=output) 调用 round 函数，检查返回值是否为 output
        result = np.round(xm, decimals=2, out=output)
        assert_(result is output)
        assert_equal(result, xm.round(decimals=2, out=output))

        # 创建一个空数组 output，再次测试 round 函数的输出参数
        output = empty((3, 4), dtype=float)
        result = xm.round(decimals=2, out=output)
        assert_(result is output)
    def test_round_with_scalar(self):
        # 测试标量/零维输入的四舍五入功能
        # GH问题编号2244

        # 创建一个带有数据掩码的数组a，值为1.1
        a = array(1.1, mask=[False])
        # 断言a四舍五入后的结果为1
        assert_equal(a.round(), 1)

        # 创建一个带有数据掩码的数组a，值为1.1
        a = array(1.1, mask=[True])
        # 断言a四舍五入后的结果为掩码值（即masked）
        assert_(a.round() is masked)

        # 创建一个带有数据掩码的数组a，值为1.1
        a = array(1.1, mask=[False])
        # 创建一个空数组output，填充值为-9999，数据类型为float
        output = np.empty(1, dtype=float)
        output.fill(-9999)
        # 对数组a进行四舍五入操作，将结果写入output
        a.round(out=output)
        # 断言output的值与1相等
        assert_equal(output, 1)

        # 创建一个带有数据掩码的数组a，值为1.1
        a = array(1.1, mask=[False])
        # 创建一个带有掩码的数组output，值为-9999.0
        output = array(-9999., mask=[True])
        # 对数组a进行四舍五入操作，将结果写入output
        a.round(out=output)
        # 断言output的第一个元素值与1相等
        assert_equal(output[()], 1)

        # 创建一个带有数据掩码的数组a，值为1.1
        a = array(1.1, mask=[True])
        # 创建一个带有掩码的数组output，值为-9999.0
        output = array(-9999., mask=[False])
        # 对数组a进行四舍五入操作，将结果写入output
        a.round(out=output)
        # 断言output的第一个元素为掩码值（即masked）
        assert_(output[()] is masked)

    def test_identity(self):
        # 测试单位矩阵的生成函数identity

        # 生成一个大小为5的单位矩阵a
        a = identity(5)
        # 断言a的类型为MaskedArray
        assert_(isinstance(a, MaskedArray))
        # 断言a与numpy生成的大小为5的单位矩阵相等
        assert_equal(a, np.identity(5))

    def test_power(self):
        # 测试幂运算函数power的各种用例

        # 定义一个变量x为-1.1
        x = -1.1
        # 断言power(-1.1, 2.0)的结果约等于1.21
        assert_almost_equal(power(x, 2.), 1.21)
        # 断言power(-1.1, masked)的结果为掩码值（即masked）
        assert_(power(x, masked) is masked)

        # 创建一个带有数据掩码的数组x
        x = array([-1.1, -1.1, 1.1, 1.1, 0.])
        # 创建一个带有数据掩码的数组b
        b = array([0.5, 2., 0.5, 2., -1.], mask=[0, 0, 0, 0, 1])
        # 对数组x进行以b为指数的幂运算，结果保存在数组y中
        y = power(x, b)
        # 断言y的结果与预期值接近
        assert_almost_equal(y, [0, 1.21, 1.04880884817, 1.21, 0.])
        # 断言y的数据掩码与预期相符
        assert_equal(y._mask, [1, 0, 0, 0, 1])

        # 清除数组b的数据掩码
        b.mask = nomask
        # 再次对数组x进行以b为指数的幂运算
        y = power(x, b)
        # 断言y的数据掩码与预期相符
        assert_equal(y._mask, [1, 0, 0, 0, 1])

        # 对数组x和b进行直接幂运算，结果分别保存在z和y中
        z = x ** b
        # 断言z和y的数据掩码相同
        assert_equal(z._mask, y._mask)
        # 断言z和y的结果接近
        assert_almost_equal(z, y)
        # 断言z和y的数据部分相同
        assert_almost_equal(z._data, y._data)

        # 将数组x就地修改为以b为指数的幂运算结果
        x **= b
        # 断言x的数据掩码与y相同
        assert_equal(x._mask, y._mask)
        # 断言x和y的结果接近
        assert_almost_equal(x, y)
        # 断言x和y的数据部分相同
        assert_almost_equal(x._data, y._data)

    def test_power_with_broadcasting(self):
        # 测试带有广播功能的幂运算

        # 创建一个二维数组a2和对应的带有数据掩码的数组a2m
        a2 = np.array([[1., 2., 3.], [4., 5., 6.]])
        a2m = array(a2, mask=[[1, 0, 0], [0, 0, 1]])
        # 创建一维数组b1和对应的二维数组b2，以及带有数据掩码的数组b2m
        b1 = np.array([2, 4, 3])
        b2 = np.array([b1, b1])
        b2m = array(b2, mask=[[0, 1, 0], [0, 1, 0]])

        # 创建预期结果控制数组ctrl和对应的数据掩码
        ctrl = array([[1 ** 2, 2 ** 4, 3 ** 3], [4 ** 2, 5 ** 4, 6 ** 3]],
                     mask=[[1, 1, 0], [0, 1, 1]])

        # 测试无广播的情况，基数和指数都带有数据掩码
        test = a2m ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

        # 测试无广播的情况，基数带有数据掩码，指数不带掩码
        test = a2m ** b2
        assert_equal(test, ctrl)
        assert_equal(test.mask, a2m.mask)

        # 测试无广播的情况，基数不带掩码，指数带有数据掩码
        test = a2 ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, b2m.mask)

        # 创建另一个预期结果控制数组ctrl和对应的数据掩码
        ctrl = array([[2 ** 2, 4 ** 4, 3 ** 3], [2 ** 2, 4 ** 4, 3 ** 3]],
                     mask=[[0, 1, 0], [0, 1, 0]])

        # 测试基数不带数据掩码，指数带有数据掩码
        test = b1 ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

        # 测试基数带有数据掩码，指数不带掩码
        test = b2m ** b1
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
    def test_where(self):
        # Test the where function
        # 创建测试用例的输入数组 x 和 y
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        
        # 创建两个掩码 m1 和 m2
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        
        # 使用掩码创建 masked_array 对象 xm 和 ym
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        
        # 设置 xm 对象的填充值为 1e+20
        xm.set_fill_value(1e+20)

        # 对 xm 进行条件判断，大于 2 的用 xm 值，否则用 -9 填充，结果保存在 d 中
        d = where(xm > 2, xm, -9)
        assert_equal(d, [-9., -9., -9., -9., -9., 4.,
                         -9., -9., 10., -9., -9., 3.])
        assert_equal(d._mask, xm._mask)
        
        # 对 xm 和 ym 进行条件判断，大于 2 的用 -9 填充，否则用 ym 值，结果保存在 d 中
        d = where(xm > 2, -9, ym)
        assert_equal(d, [5., 0., 3., 2., -1., -9.,
                         -9., -10., -9., 1., 0., -9.])
        assert_equal(d._mask, [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        
        # 对 xm 进行条件判断，大于 2 的用 xm 值，否则用 masked 对象填充，结果保存在 d 中
        d = where(xm > 2, xm, masked)
        assert_equal(d, [-9., -9., -9., -9., -9., 4.,
                         -9., -9., 10., -9., -9., 3.])
        
        # 检查 d 的掩码是否正确
        tmp = xm._mask.copy()
        tmp[(xm <= 2).filled(True)] = True
        assert_equal(d._mask, tmp)

        # 在运行时捕获警告，尝试将 xm 转换为整数类型
        with np.errstate(invalid="warn"):
            with pytest.warns(RuntimeWarning, match="invalid value"):
                ixm = xm.astype(int)
        
        # 对 ixm 进行条件判断，大于 2 的用 ixm 值，否则用 masked 对象填充，结果保存在 d 中
        d = where(ixm > 2, ixm, masked)
        assert_equal(d, [-9, -9, -9, -9, -9, 4, -9, -9, 10, -9, -9, 3])
        assert_equal(d.dtype, ixm.dtype)

    def test_where_object(self):
        # 创建空的 np.array 对象 a
        a = np.array(None)
        
        # 创建空的 masked_array 对象 b
        b = masked_array(None)
        
        # 复制 b 对象到 r
        r = b.copy()
        
        # 测试 np.ma.where 对象
        assert_equal(np.ma.where(True, a, a), r)
        assert_equal(np.ma.where(True, b, b), r)

    def test_where_with_masked_choice(self):
        # 创建数组 x，其中第三个元素设为 masked
        x = arange(10)
        x[3] = masked
        
        # 创建条件数组 c，大于等于 8 的设为 True，否则为 False
        c = x >= 8
        
        # 对条件数组 c 进行判断，True 的位置用 x 值填充，False 的位置用 masked 填充，结果保存在 z 中
        z = where(c, x, masked)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is masked)
        assert_(z[7] is masked)
        assert_(z[8] is not masked)
        assert_(z[9] is not masked)
        assert_equal(x, z)
        
        # 对条件数组 c 进行判断，True 的位置用 masked 填充，False 的位置用 x 值填充，结果保存在 z 中
        z = where(c, masked, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is not masked)
        assert_(z[7] is not masked)
        assert_(z[8] is masked)
        assert_(z[9] is masked)
    def test_where_with_masked_condition(self):
        x = array([1., 2., 3., 4., 5.])  # 创建包含浮点数的数组 x
        c = array([1, 1, 1, 0, 0])  # 创建包含整数的数组 c
        x[2] = masked  # 将 x 数组的第三个元素标记为 masked（掩码值）
        z = where(c, x, -x)  # 使用 c 的条件，对 x 和 -x 进行条件运算，结果存储在 z 中
        assert_equal(z, [1., 2., 0., -4., -5])  # 断言 z 应该等于给定的列表
        c[0] = masked  # 将 c 数组的第一个元素标记为 masked
        z = where(c, x, -x)  # 再次使用 c 的条件，对 x 和 -x 进行条件运算，结果存储在 z 中
        assert_equal(z, [1., 2., 0., -4., -5])  # 断言 z 应该等于给定的列表
        assert_(z[0] is masked)  # 断言 z 的第一个元素是 masked
        assert_(z[1] is not masked)  # 断言 z 的第二个元素不是 masked
        assert_(z[2] is masked)  # 断言 z 的第三个元素是 masked

        x = arange(1, 6)  # 创建一个从 1 到 5 的数组 x
        x[-1] = masked  # 将 x 数组的最后一个元素标记为 masked
        y = arange(1, 6) * 10  # 创建一个从 10 到 50 的数组 y
        y[2] = masked  # 将 y 数组的第三个元素标记为 masked
        c = array([1, 1, 1, 0, 0], mask=[1, 0, 0, 0, 0])  # 创建一个带掩码的数组 c
        cm = c.filled(1)  # 使用填充值 1 填充掩码数组 c
        z = where(c, x, y)  # 使用 c 的条件，对 x 和 y 进行条件运算，结果存储在 z 中
        zm = where(cm, x, y)  # 使用填充后的 c 的条件，对 x 和 y 进行条件运算，结果存储在 zm 中
        assert_equal(z, zm)  # 断言 z 应该等于 zm
        assert_(getmask(zm) is nomask)  # 断言 zm 的掩码为 nomask
        assert_equal(zm, [1, 2, 3, 40, 50])  # 断言 zm 应该等于给定的列表
        z = where(c, masked, 1)  # 使用 c 的条件，将 masked 替换为 1，结果存储在 z 中
        assert_equal(z, [99, 99, 99, 1, 1])  # 断言 z 应该等于给定的列表
        z = where(c, 1, masked)  # 使用 c 的条件，将 masked 替换为 1，结果存储在 z 中
        assert_equal(z, [99, 1, 1, 99, 99])  # 断言 z 应该等于给定的列表

    def test_where_type(self):
        # 测试 where 函数在类型转换时的保持性
        x = np.arange(4, dtype=np.int32)  # 创建一个具有指定数据类型的数组 x
        y = np.arange(4, dtype=np.float32) * 2.2  # 创建一个具有指定数据类型的数组 y
        test = where(x > 1.5, y, x).dtype  # 使用 where 函数进行条件运算，并获取结果的数据类型
        control = np.result_type(np.int32, np.float32)  # 获取预期的结果数据类型
        assert_equal(test, control)  # 断言条件运算结果的数据类型应该等于预期的结果数据类型

    def test_where_broadcast(self):
        # 解决 Issue 8599
        x = np.arange(9).reshape(3, 3)  # 创建一个 3x3 的数组 x
        y = np.zeros(3)  # 创建一个全零数组 y
        core = np.where([1, 0, 1], x, y)  # 使用 where 函数对 x 和 y 进行条件运算
        ma = where([1, 0, 1], x, y)  # 同上，使用 where 函数对 x 和 y 进行条件运算

        assert_equal(core, ma)  # 断言 core 应该等于 ma
        assert_equal(core.dtype, ma.dtype)  # 断言 core 的数据类型应该等于 ma 的数据类型

    def test_where_structured(self):
        # 解决 Issue 8600
        dt = np.dtype([('a', int), ('b', int)])  # 创建一个结构化数据类型 dt
        x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)  # 创建一个结构化数组 x
        y = np.array((10, 20), dtype=dt)  # 创建一个结构化数组 y
        core = np.where([0, 1, 1], x, y)  # 使用 where 函数对 x 和 y 进行条件运算
        ma = np.where([0, 1, 1], x, y)  # 同上，使用 where 函数对 x 和 y 进行条件运算

        assert_equal(core, ma)  # 断言 core 应该等于 ma
        assert_equal(core.dtype, ma.dtype)  # 断言 core 的数据类型应该等于 ma 的数据类型

    def test_where_structured_masked(self):
        dt = np.dtype([('a', int), ('b', int)])  # 创建一个结构化数据类型 dt
        x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)  # 创建一个结构化数组 x

        ma = where([0, 1, 1], x, masked)  # 使用 where 函数对 x 和 masked 进行条件运算
        expected = masked_where([1, 0, 0], x)  # 根据指定条件创建一个预期的 masked 数组

        assert_equal(ma.dtype, expected.dtype)  # 断言 ma 的数据类型应该等于 expected 的数据类型
        assert_equal(ma, expected)  # 断言 ma 应该等于 expected
        assert_equal(ma.mask, expected.mask)  # 断言 ma 的掩码应该等于 expected 的掩码

    def test_masked_invalid_error(self):
        a = np.arange(5, dtype=object)  # 创建一个包含对象的数组 a
        a[3] = np.inf  # 将数组 a 的第四个元素设为无穷大
        a[2] = np.nan  # 将数组 a 的第三个元素设为 NaN
        with pytest.raises(TypeError, match="not supported for the input types"):
            np.ma.masked_invalid(a)  # 断言在处理无效值时应该抛出特定类型的异常
    # 定义测试函数，用于测试 pandas 的 masked_invalid 函数
    def test_masked_invalid_pandas(self):
        # getdata() 曾因其 _data 属性而对 pandas series 产生不良影响。
        # 此测试主要是一个回归测试，如果 getdata() 被调整，则可以移除此测试。
        # 创建一个模拟的 Series 类
        class Series():
            _data = "nonsense"

            # 定义 __array__ 方法，返回一个包含 NaN 和 Infinity 的 NumPy 数组
            def __array__(self, dtype=None, copy=None):
                return np.array([5, np.nan, np.inf])

        # 使用 np.ma.masked_invalid 对 Series 进行处理
        arr = np.ma.masked_invalid(Series())
        # 断言处理后的数据数组与原始 Series 数组相等
        assert_array_equal(arr._data, np.array(Series()))
        # 断言掩码数组与预期的 [False, True, True] 相等
        assert_array_equal(arr._mask, [False, True, True])

    # 使用 pytest.mark.parametrize 进行参数化测试
    @pytest.mark.parametrize("copy", [True, False])
    # 定义测试函数，测试 masked_invalid 函数对完全掩码情况的处理
    def test_masked_invalid_full_mask(self, copy):
        # Matplotlib 依赖于 masked_invalid 始终返回完全掩码
        # 创建一个包含数据的 ma 数组 a
        a = np.ma.array([1, 2, 3, 4])
        # 断言 a 的掩码为 nomask
        assert a._mask is nomask
        # 对数组 a 应用 masked_invalid 函数
        res = np.ma.masked_invalid(a, copy=copy)
        # 断言结果的掩码不是 nomask
        assert res.mask is not nomask
        # 断言原始数组 a 的掩码未被改变
        assert a.mask is nomask
        # 断言结果数组的数据与原始数组是否共享内存
        assert np.may_share_memory(a._data, res._data) != copy

    # 定义测试函数，测试 numpy 中的 choose 函数
    def test_choose(self):
        # 定义选择列表 choices
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        # 使用 choose 函数根据给定的索引数组选择元素
        chosen = choose([2, 3, 1, 0], choices)
        # 断言选择结果与预期的数组相等
        assert_equal(chosen, array([20, 31, 12, 3]))
        # 使用 mode='clip' 参数测试 choose 函数
        chosen = choose([2, 4, 1, 0], choices, mode='clip')
        # 断言选择结果与预期的数组相等
        assert_equal(chosen, array([20, 31, 12, 3]))
        # 使用 mode='wrap' 参数测试 choose 函数
        chosen = choose([2, 4, 1, 0], choices, mode='wrap')
        # 断言选择结果与预期的数组相等
        assert_equal(chosen, array([20, 1, 12, 3]))
        # 使用带有部分掩码索引的数组 indices_ 测试 choose 函数
        indices_ = array([2, 4, 1, 0], mask=[1, 0, 0, 1])
        chosen = choose(indices_, choices, mode='wrap')
        # 断言选择结果与预期的数组相等
        assert_equal(chosen, array([99, 1, 12, 99]))
        # 断言选择结果的掩码与预期的掩码相等
        assert_equal(chosen.mask, [1, 0, 0, 1])
        # 使用带有部分掩码选择列表的数组 choices 测试 choose 函数
        choices = array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        chosen = choose(indices_, choices, mode='wrap')
        # 断言选择结果与预期的数组相等
        assert_equal(chosen, array([20, 31, 12, 3]))
        # 断言选择结果的掩码与预期的掩码相等
        assert_equal(chosen.mask, [1, 0, 0, 1])
    def test_choose_with_out(self):
        # Test choose with an explicit out keyword

        # 定义选择列表
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        
        # 创建一个空的数组，用于存储选择的结果
        store = empty(4, dtype=int)
        
        # 使用 choose 函数进行选择操作，将结果存储到预先定义的数组中
        chosen = choose([2, 3, 1, 0], choices, out=store)
        
        # 断言存储的结果与预期结果相等
        assert_equal(store, array([20, 31, 12, 3]))
        
        # 断言存储的结果与返回的 chosen 对象相同
        assert_(store is chosen)
        
        # 使用带有遮罩索引和 out 参数的 choose 函数进行测试
        store = empty(4, dtype=int)
        indices_ = array([2, 3, 1, 0], mask=[1, 0, 0, 1])
        chosen = choose(indices_, choices, mode='wrap', out=store)
        
        # 断言存储的结果与预期结果相等
        assert_equal(store, array([99, 31, 12, 99]))
        
        # 断言存储的遮罩与预期的遮罩相等
        assert_equal(store.mask, [1, 0, 0, 1])
        
        # 使用带有部分遮罩选择和 out 参数的 choose 函数进行测试
        choices = array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        
        # 创建一个视图数组用于存储选择的结果
        store = empty(4, dtype=int).view(ndarray)
        
        # 使用 choose 函数进行选择操作，将结果存储到视图数组中
        chosen = choose(indices_, choices, mode='wrap', out=store)
        
        # 断言存储的结果与预期结果相等
        assert_equal(store, array([999999, 31, 12, 999999]))

    def test_reshape(self):
        # 创建一个包含 masked 值的数组
        a = arange(10)
        a[0] = masked
        
        # 测试默认的 reshape 操作
        b = a.reshape((5, 2))
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['C'])
        
        # 使用列表参数而不是元组进行 reshape 操作
        b = a.reshape(5, 2)
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['C'])
        
        # 使用指定顺序进行 reshape 操作
        b = a.reshape((5, 2), order='F')
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['F'])
        
        # 使用指定顺序进行 reshape 操作（使用列表参数）
        b = a.reshape(5, 2, order='F')
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['F'])

        # 使用 np.reshape 函数进行 reshape 操作
        c = np.reshape(a, (2, 5))
        
        # 断言结果数组的类型为 MaskedArray
        assert_(isinstance(c, MaskedArray))
        assert_equal(c.shape, (2, 5))
        
        # 断言结果数组中的第一个元素为 masked
        assert_(c[0, 0] is masked)
        
        # 断言结果数组是 C 风格存储的
        assert_(c.flags['C'])
    # 定义测试函数，用于测试 make_mask_descr 函数的不同输入情况
    def test_make_mask_descr(self):
        # 测试类型为 [('a', float), ('b', float)] 的情况
        ntype = [('a', float), ('b', float)]
        # 调用 make_mask_descr 函数生成测试结果
        test = make_mask_descr(ntype)
        # 断言测试结果与预期结果 [('a', bool), ('b', bool)] 相等
        assert_equal(test, [('a', bool), ('b', bool)])
        # 断言测试结果与自身的身份相同
        assert_(test is make_mask_descr(test))

        # 测试类型为 (float, 2) 的情况
        ntype = (float, 2)
        # 调用 make_mask_descr 函数生成测试结果
        test = make_mask_descr(ntype)
        # 断言测试结果与预期结果 (bool, 2) 相等
        assert_equal(test, (bool, 2))
        # 断言测试结果与自身的身份相同
        assert_(test is make_mask_descr(test))

        # 测试类型为 float 的情况
        ntype = float
        # 调用 make_mask_descr 函数生成测试结果
        test = make_mask_descr(ntype)
        # 断言测试结果与预期结果 np.dtype(bool) 相等
        assert_equal(test, np.dtype(bool))
        # 断言测试结果与自身的身份相同
        assert_(test is make_mask_descr(test))

        # 测试类型为 [('a', float), ('b', [('ba', float), ('bb', float)])] 的情况
        ntype = [('a', float), ('b', [('ba', float), ('bb', float)])]
        # 调用 make_mask_descr 函数生成测试结果
        test = make_mask_descr(ntype)
        # 构建预期结果
        control = np.dtype([('a', 'b1'), ('b', [('ba', 'b1'), ('bb', 'b1')])])
        # 断言测试结果与预期结果相等
        assert_equal(test, control)
        # 断言测试结果与自身的身份相同
        assert_(test is make_mask_descr(test))

        # 测试类型为 [('a', (float, 2))] 的情况
        ntype = [('a', (float, 2))]
        # 调用 make_mask_descr 函数生成测试结果
        test = make_mask_descr(ntype)
        # 断言测试结果与预期结果 np.dtype([('a', (bool, 2))]) 相等
        assert_equal(test, np.dtype([('a', (bool, 2))]))
        # 断言测试结果与自身的身份相同
        assert_(test is make_mask_descr(test))

        # 测试类型为 [(('A', 'a'), float)] 的情况
        ntype = [(('A', 'a'), float)]
        # 调用 make_mask_descr 函数生成测试结果
        test = make_mask_descr(ntype)
        # 断言测试结果与预期结果 np.dtype([(('A', 'a'), bool)]) 相等
        assert_equal(test, np.dtype([(('A', 'a'), bool)]))
        # 断言测试结果与自身的身份相同
        assert_(test is make_mask_descr(test))

        # 测试嵌套的 boolean 类型应保持身份不变的情况
        # 定义基础类型
        base_type = np.dtype([('a', int, 3)])
        # 调用 make_mask_descr 函数生成基础类型的掩码描述
        base_mtype = make_mask_descr(base_type)
        # 定义子类型
        sub_type = np.dtype([('a', int), ('b', base_mtype)])
        # 调用 make_mask_descr 函数生成测试结果
        test = make_mask_descr(sub_type)
        # 构建预期结果
        expected_result = np.dtype([('a', bool), ('b', [('a', bool, 3)])])
        # 断言测试结果与预期结果相等
        assert_equal(test, expected_result)
        # 断言测试结果中 'b' 字段与基础类型的掩码描述身份相同
        assert_(test.fields['b'][0] is base_mtype)
    def test_make_mask(self):
        # 定义测试函数 test_make_mask，这是一个测试类中的测试方法
        # 测试 make_mask 函数处理列表作为输入的情况
        mask = [0, 1]
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [0, 1])
        
        # 测试 make_mask 函数处理 ndarray 作为输入的情况
        mask = np.array([0, 1], dtype=bool)
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [0, 1])
        
        # 测试 make_mask 函数处理灵活类型的 ndarray 作为输入，使用默认行为
        mdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [1, 1])
        
        # 测试 make_mask 函数处理灵活类型的 ndarray 作为输入，使用输入的 dtype
        mdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test.dtype, mdtype)
        assert_equal(test, mask)
        
        # 测试 make_mask 函数处理灵活类型的 ndarray 作为输入，使用输入的 dtype 转换
        mdtype = [('a', float), ('b', float)]
        bdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test.dtype, bdtype)
        assert_equal(test, np.array([(0, 0), (0, 1)], dtype=bdtype))
        
        # 确保 make_mask 函数能处理 void 类型输入
        mask = np.array((False, True), dtype='?,?')[()]
        assert_(isinstance(mask, np.void))
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test, mask)
        assert_(test is not mask)
        
        # 测试 make_mask 函数能处理多维 void 类型输入
        mask = np.array((0, 1), dtype='i4,i4')[()]
        test2 = make_mask(mask, dtype=mask.dtype)
        assert_equal(test2, test)
        
        # 测试当输入为 nomask 时，确保返回 nomask
        bools = [True, False]
        dtypes = [MaskType, float]
        msgformat = 'copy=%s, shrink=%s, dtype=%s'
        for cpy, shr, dt in itertools.product(bools, bools, dtypes):
            res = make_mask(nomask, copy=cpy, shrink=shr, dtype=dt)
            assert_(res is nomask, msgformat % (cpy, shr, dt))
    def test_mask_or(self):
        # 初始化数据类型为 [('a', bool), ('b', bool)] 的结构化数组
        mtype = [('a', bool), ('b', bool)]
        # 创建一个包含四个元素的数组 mask，每个元素为一个元组，元组中有两个布尔值
        mask = np.array([(0, 0), (0, 1), (1, 0), (0, 0)], dtype=mtype)
        
        # 使用 nomask 作为输入测试 mask_or 函数
        test = mask_or(mask, nomask)
        # 断言 test 是否等于 mask
        assert_equal(test, mask)
        
        # 使用 nomask 和 mask 作为输入测试 mask_or 函数
        test = mask_or(nomask, mask)
        # 断言 test 是否等于 mask
        assert_equal(test, mask)
        
        # 使用 False 作为输入测试 mask_or 函数
        test = mask_or(mask, False)
        # 断言 test 是否等于 mask
        assert_equal(test, mask)
        
        # 创建另一个与 mask 具有相同数据类型的数组 other
        other = np.array([(0, 1), (0, 1), (0, 1), (0, 1)], dtype=mtype)
        # 使用 other 作为输入测试 mask_or 函数
        test = mask_or(mask, other)
        # 创建一个预期结果数组 control，其元素经过按位或运算得到
        control = np.array([(0, 1), (0, 1), (1, 1), (0, 1)], dtype=mtype)
        # 断言 test 是否等于 control
        assert_equal(test, control)
        
        # 创建具有不同数据类型 othertype 的数组 other
        othertype = [('A', bool), ('B', bool)]
        other = np.array([(0, 1), (0, 1), (0, 1), (0, 1)], dtype=othertype)
        # 尝试使用 other 作为输入测试 mask_or 函数，预期会引发 ValueError 异常
        try:
            test = mask_or(mask, other)
        except ValueError:
            pass
        
        # 创建一个嵌套结构的数据类型 dtype
        dtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
        # 创建两个具有 dtype 数据类型的数组 amask 和 bmask
        amask = np.array([(0, (1, 0)), (0, (1, 0))], dtype=dtype)
        bmask = np.array([(1, (0, 1)), (0, (0, 0))], dtype=dtype)
        # 创建一个预期结果数组 cntrl，其元素经过按位或运算得到
        cntrl = np.array([(1, (1, 1)), (0, (1, 0))], dtype=dtype)
        # 断言 mask_or(amask, bmask) 是否等于 cntrl
        assert_equal(mask_or(amask, bmask), cntrl)

    def test_flatten_mask(self):
        # 测试 flatten_mask 函数
        # 标准数据类型的 mask 数组
        mask = np.array([0, 0, 1], dtype=bool)
        # 断言 flatten_mask(mask) 是否等于 mask
        assert_equal(flatten_mask(mask), mask)
        
        # 灵活数据类型的 mask 数组
        mask = np.array([(0, 0), (0, 1)], dtype=[('a', bool), ('b', bool)])
        # 使用 flatten_mask 函数处理 mask 数组
        test = flatten_mask(mask)
        # 创建一个预期结果数组 control，其元素经过 flatten 处理后得到
        control = np.array([0, 0, 0, 1], dtype=bool)
        # 断言 test 是否等于 control
        assert_equal(test, control)

        # 创建一个复杂结构的数据类型 mdtype 和相应的数据 data
        mdtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
        data = [(0, (0, 0)), (0, (0, 1))]
        mask = np.array(data, dtype=mdtype)
        # 使用 flatten_mask 函数处理 mask 数组
        test = flatten_mask(mask)
        # 创建一个预期结果数组 control，其元素经过 flatten 处理后得到
        control = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        # 断言 test 是否等于 control
        assert_equal(test, control)

    def test_on_ndarray(self):
        # 在 ndarray 上测试函数
        # 创建一个 ndarray 数组 a
        a = np.array([1, 2, 3, 4])
        # 创建一个 mask=False 的 masked array m
        m = array(a, mask=False)
        # 测试 anom 函数
        test = anom(a)
        # 断言 test 是否等于 m.anom()
        assert_equal(test, m.anom())
        # 测试 reshape 函数
        test = reshape(a, (2, 2))
        # 断言 test 是否等于 m.reshape(2, 2)
        assert_equal(test, m.reshape(2, 2))

    def test_compress(self):
        # 在 ndarray 和 masked array 上测试 compress 函数
        # 解决 Github 问题 #2495
        # 创建一个 ndarray 数组 arr
        arr = np.arange(8)
        arr.shape = 4, 2
        # 创建一个条件数组 cond
        cond = np.array([True, False, True, True])
        # 创建一个预期结果数组 control，其元素经过 compress 处理后得到
        control = arr[[0, 2, 3]]
        # 使用 compress 函数处理 arr
        test = np.ma.compress(cond, arr, axis=0)
        # 断言 test 是否等于 control
        assert_equal(test, control)
        
        # 创建一个 masked array marr
        marr = np.ma.array(arr)
        # 使用 compress 函数处理 marr
        test = np.ma.compress(cond, marr, axis=0)
        # 断言 test 是否等于 control
        assert_equal(test, control)
    def test_compressed(self):
        # Test ma.compressed function.
        # 测试 ma.compressed 函数

        # Address gh-4026
        # 处理 GitHub 问题编号 gh-4026

        a = np.ma.array([1, 2])
        # 创建一个掩码数组 a，包含元素 [1, 2]

        test = np.ma.compressed(a)
        # 使用 ma.compressed 函数压缩数组 a
        assert_(type(test) is np.ndarray)
        # 断言压缩后的结果 test 类型为 np.ndarray

        # Test case when input data is ndarray subclass
        # 测试当输入数据为 ndarray 的子类时的情况
        class A(np.ndarray):
            pass

        a = np.ma.array(A(shape=0))
        # 创建一个掩码数组 a，其底层数据类型为 A，形状为空
        test = np.ma.compressed(a)
        # 使用 ma.compressed 函数压缩数组 a
        assert_(type(test) is A)
        # 断言压缩后的结果 test 类型为 A

        # Test that compress flattens
        # 测试 compress 方法的扁平化效果
        test = np.ma.compressed([[1],[2]])
        # 压缩二维数组 [[1],[2]]
        assert_equal(test.ndim, 1)
        # 断言压缩后的结果 test 的维度为 1

        test = np.ma.compressed([[[[[1]]]]])
        # 压缩多维数组 [[[[[1]]]]]
        assert_equal(test.ndim, 1)
        # 断言压缩后的结果 test 的维度为 1

        # Test case when input is MaskedArray subclass
        # 测试当输入为 MaskedArray 的子类时的情况
        class M(MaskedArray):
            pass

        test = np.ma.compressed(M([[[]], [[]]]))
        # 使用 ma.compressed 函数压缩 M 的实例，传入多维数组
        assert_equal(test.ndim, 1)
        # 断言压缩后的结果 test 的维度为 1

        # with .compressed() overridden
        # 使用重载的 .compressed() 方法
        class M(MaskedArray):
            def compressed(self):
                return 42

        test = np.ma.compressed(M([[[]], [[]]]))
        # 使用重载后的 .compressed() 方法压缩 M 的实例，传入多维数组
        assert_equal(test, 42)
        # 断言压缩后的结果为 42

    def test_convolve(self):
        a = masked_equal(np.arange(5), 2)
        # 创建一个掩码数组 a，使用 masked_equal 函数将元素 2 掩码化

        b = np.array([1, 1])
        # 创建数组 b，包含元素 [1, 1]

        result = masked_equal([0, 1, -1, -1, 7, 4], -1)
        # 创建一个掩码数组 result，将元素 -1 掩码化

        test = np.ma.convolve(a, b, mode='full')
        # 使用 ma.convolve 函数对数组 a 和 b 进行全模式卷积
        assert_equal(test, result)
        # 断言卷积后的结果与预期结果 result 相等

        test = np.ma.convolve(a, b, mode='same')
        # 使用 ma.convolve 函数对数组 a 和 b 进行同模式卷积
        assert_equal(test, result[:-1])
        # 断言卷积后的结果与预期结果 result 的前 n-1 个元素相等

        test = np.ma.convolve(a, b, mode='valid')
        # 使用 ma.convolve 函数对数组 a 和 b 进行有效模式卷积
        assert_equal(test, result[1:-1])
        # 断言卷积后的结果与预期结果 result 的中间部分元素相等

        result = masked_equal([0, 1, 1, 3, 7, 4], -1)
        # 更新 result 为将元素 -1 掩码化后的数组

        test = np.ma.convolve(a, b, mode='full', propagate_mask=False)
        # 使用 ma.convolve 函数对数组 a 和 b 进行全模式卷积，不传播掩码
        assert_equal(test, result)
        # 断言卷积后的结果与更新后的 result 相等

        test = np.ma.convolve(a, b, mode='same', propagate_mask=False)
        # 使用 ma.convolve 函数对数组 a 和 b 进行同模式卷积，不传播掩码
        assert_equal(test, result[:-1])
        # 断言卷积后的结果与更新后的 result 的前 n-1 个元素相等

        test = np.ma.convolve(a, b, mode='valid', propagate_mask=False)
        # 使用 ma.convolve 函数对数组 a 和 b 进行有效模式卷积，不传播掩码
        assert_equal(test, result[1:-1])
        # 断言卷积后的结果与更新后的 result 的中间部分元素相等

        test = np.ma.convolve([1, 1], [1, 1, 1])
        # 使用 ma.convolve 函数对数组 [1, 1] 和 [1, 1, 1] 进行卷积
        assert_equal(test, masked_equal([1, 2, 2, 1], -1))
        # 断言卷积后的结果与预期的掩码化数组相等

        a = [1, 1]
        # 创建数组 a，包含元素 [1, 1]

        b = masked_equal([1, -1, -1, 1], -1)
        # 创建一个掩码化数组 b，将元素 -1 掩码化

        test = np.ma.convolve(a, b, propagate_mask=False)
        # 使用 ma.convolve 函数对数组 a 和 b 进行卷积，不传播掩码
        assert_equal(test, masked_equal([1, 1, -1, 1, 1], -1))
        # 断言卷积后的结果与预期的掩码化数组相等

        test = np.ma.convolve(a, b, propagate_mask=True)
        # 使用 ma.convolve 函数对数组 a 和 b 进行卷积，传播掩码
        assert_equal(test, masked_equal([-1, -1, -1, -1, -1], -1))
        # 断言卷积后的结果与预期的掩码化数组相等
# 定义一个测试类 TestMaskedFields，用于测试记录掩码功能
class TestMaskedFields:

    # 设置测试方法的初始化方法
    def setup_method(self):
        # 初始化整数列表 ilist
        ilist = [1, 2, 3, 4, 5]
        # 初始化浮点数列表 flist
        flist = [1.1, 2.2, 3.3, 4.4, 5.5]
        # 初始化字符串列表 slist
        slist = ['one', 'two', 'three', 'four', 'five']
        # 定义结构化数据类型 ddtype
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        # 定义掩码数据类型 mdtype
        mdtype = [('a', bool), ('b', bool), ('c', bool)]
        # 初始化掩码列表 mask
        mask = [0, 1, 0, 0, 1]
        # 使用 array 函数创建 base 对象，设置数据和掩码
        base = array(list(zip(ilist, flist, slist)), mask=mask, dtype=ddtype)
        # 将初始化数据保存到 self.data 字典中
        self.data = dict(base=base, mask=mask, ddtype=ddtype, mdtype=mdtype)

    # 测试设置记录掩码的方法
    def test_set_records_masks(self):
        # 获取 self.data 中的 base 对象和 mdtype 数据类型
        base = self.data['base']
        mdtype = self.data['mdtype']
        # 设置为未掩码状态
        base.mask = nomask
        # 断言 base._mask 与全零数组的相等性
        assert_equal_records(base._mask, np.zeros(base.shape, dtype=mdtype))
        # 设置为已掩码状态
        base.mask = masked
        # 断言 base._mask 与全一数组的相等性
        assert_equal_records(base._mask, np.ones(base.shape, dtype=mdtype))
        # 使用简单布尔值设置掩码状态为未掩码
        base.mask = False
        # 断言 base._mask 与全零数组的相等性
        assert_equal_records(base._mask, np.zeros(base.shape, dtype=mdtype))
        # 使用简单布尔值设置掩码状态为已掩码
        base.mask = True
        # 断言 base._mask 与全一数组的相等性
        assert_equal_records(base._mask, np.ones(base.shape, dtype=mdtype))
        # 使用列表设置不同记录的掩码状态
        base.mask = [0, 0, 0, 1, 1]
        # 断言 base._mask 与指定数组的相等性
        assert_equal_records(base._mask,
                             np.array([(x, x, x) for x in [0, 0, 0, 1, 1]],
                                      dtype=mdtype))

    # 测试设置记录元素的方法
    def test_set_record_element(self):
        # 获取 self.data 中的 base 对象
        base = self.data['base']
        # 分别获取 base 中的字段 'a', 'b', 'c'
        (base_a, base_b, base_c) = (base['a'], base['b'], base['c'])
        # 设置第一个记录元素为 (pi, pi, 'pi')
        base[0] = (pi, pi, 'pi')

        # 断言字段 'a' 的数据类型为整数
        assert_equal(base_a.dtype, int)
        # 断言字段 'a' 的数据内容
        assert_equal(base_a._data, [3, 2, 3, 4, 5])

        # 断言字段 'b' 的数据类型为浮点数
        assert_equal(base_b.dtype, float)
        # 断言字段 'b' 的数据内容
        assert_equal(base_b._data, [pi, 2.2, 3.3, 4.4, 5.5])

        # 断言字段 'c' 的数据类型为字符串
        assert_equal(base_c.dtype, '|S8')
        # 断言字段 'c' 的数据内容
        assert_equal(base_c._data,
                     [b'pi', b'two', b'three', b'four', b'five'])

    # 测试设置记录切片的方法
    def test_set_record_slice(self):
        # 获取 self.data 中的 base 对象
        base = self.data['base']
        # 分别获取 base 中的字段 'a', 'b', 'c'
        (base_a, base_b, base_c) = (base['a'], base['b'], base['c'])
        # 将前三个记录元素设置为 (pi, pi, 'pi')
        base[:3] = (pi, pi, 'pi')

        # 断言字段 'a' 的数据类型为整数
        assert_equal(base_a.dtype, int)
        # 断言字段 'a' 的数据内容
        assert_equal(base_a._data, [3, 3, 3, 4, 5])

        # 断言字段 'b' 的数据类型为浮点数
        assert_equal(base_b.dtype, float)
        # 断言字段 'b' 的数据内容
        assert_equal(base_b._data, [pi, pi, pi, 4.4, 5.5])

        # 断言字段 'c' 的数据类型为字符串
        assert_equal(base_c.dtype, '|S8')
        # 断言字段 'c' 的数据内容
        assert_equal(base_c._data,
                     [b'pi', b'pi', b'pi', b'four', b'five'])

    # 测试记录元素掩码的方法
    def test_mask_element(self):
        # 检查记录访问
        base = self.data['base']
        # 将第一个记录元素设置为掩码状态
        base[0] = masked

        # 遍历字段 'a', 'b', 'c'，断言其掩码状态和数据内容
        for n in ('a', 'b', 'c'):
            assert_equal(base[n].mask, [1, 1, 0, 0, 1])
            assert_equal(base[n]._data, base._data[n])
    def test_getmaskarray(self):
        # Test getmaskarray on flexible dtype
        # 定义一个灵活数据类型的结构：包含 'a' 整数和 'b' 浮点数
        ndtype = [('a', int), ('b', float)]
        # 创建一个形状为 (3,) 的空数组，使用上述灵活数据类型
        test = empty(3, dtype=ndtype)
        # 断言 getmaskarray 函数对 test 的输出
        assert_equal(getmaskarray(test),
                     np.array([(0, 0), (0, 0), (0, 0)],
                              dtype=[('a', '|b1'), ('b', '|b1')]))
        # 将 test 数组中所有元素设为 masked
        test[:] = masked
        # 断言 getmaskarray 函数对所有元素均为 masked 的 test 的输出
        assert_equal(getmaskarray(test),
                     np.array([(1, 1), (1, 1), (1, 1)],
                              dtype=[('a', '|b1'), ('b', '|b1')]))

    def test_view(self):
        # Test view w/ flexible dtype
        # 创建一个包含 0 到 9 的整数和 10 个随机浮点数的迭代器
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        # 将迭代器转换为 NumPy 数组
        data = np.array(iterator)
        # 创建一个具有灵活数据类型 'a' 和 'b' 的数组
        a = array(iterator, dtype=[('a', float), ('b', float)])
        # 将 a 数组的第一个元素的 mask 设置为 (1, 0)
        a.mask[0] = (1, 0)
        # 创建一个长度为 20 的布尔数组，第一个元素为 True，其余为 False
        controlmask = np.array([1] + 19 * [0], dtype=bool)
        # 将 a 数组全局转换为简单的浮点数类型
        test = a.view(float)
        # 断言转换后的 test 数组内容与 data 数组展平后的内容相等
        assert_equal(test, data.ravel())
        # 断言 test 数组的 mask 与 controlmask 相等
        assert_equal(test.mask, controlmask)
        # 将 a 数组全局转换为元组 (float, 2) 类型
        test = a.view((float, 2))
        # 断言转换后的 test 数组与 data 数组相等
        assert_equal(test, data)
        # 断言 test 数组的 mask 与重新排列后的 controlmask 相等
        assert_equal(test.mask, controlmask.reshape(-1, 2))

    def test_getitem(self):
        # 定义一个包含 'a' 和 'b' 浮点数的灵活数据类型
        ndtype = [('a', float), ('b', float)]
        # 创建一个包含 0 到 9 的随机浮点数和对应整数的数组，使用上述数据类型
        a = array(list(zip(np.random.rand(10), np.arange(10))), dtype=ndtype)
        # 创建一个 mask 数组，指定哪些元素被掩盖
        a.mask = np.array(list(zip([0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1, 0])),
                          dtype=[('a', bool), ('b', bool)])

        def _test_index(i):
            # 断言 a[i] 的类型为 mvoid
            assert_equal(type(a[i]), mvoid)
            # 断言 a[i] 的数据与 a._data[i] 相等
            assert_equal_records(a[i]._data, a._data[i])
            # 断言 a[i] 的 mask 与 a._mask[i] 相等
            assert_equal_records(a[i]._mask, a._mask[i])

            # 断言 a[i, ...] 的类型为 MaskedArray
            assert_equal(type(a[i, ...]), MaskedArray)
            # 断言 a[i, ...] 的数据与 a._data[i, ...] 相等
            assert_equal_records(a[i, ...]._data, a._data[i, ...])
            # 断言 a[i, ...] 的 mask 与 a._mask[i, ...] 相等
            assert_equal_records(a[i, ...]._mask, a._mask[i, ...])

        _test_index(1)   # 没有 mask
        _test_index(0)   # 一个元素被 mask
        _test_index(-2)  # 所有元素被 mask
    def test_setitem(self):
        # Issue 4866: check that one can set individual items in [record][col]
        # and [col][record] order
        # 定义结构化数据类型，包含字段 'a' (浮点数) 和 'b' (整数)
        ndtype = np.dtype([('a', float), ('b', int)])
        # 创建一个带有掩码的 MaskedArray 对象，包含两行数据
        ma = np.ma.MaskedArray([(1.0, 1), (2.0, 2)], dtype=ndtype)
        # 修改第 1 行数据的 'a' 字段为 3.0
        ma['a'][1] = 3.0
        # 验证 'a' 字段的修改结果
        assert_equal(ma['a'], np.array([1.0, 3.0]))
        # 修改第 1 行数据的 'a' 字段为 4.0
        ma[1]['a'] = 4.0
        # 验证 'a' 字段的修改结果
        assert_equal(ma['a'], np.array([1.0, 4.0]))
        
        # Issue 2403
        # 定义结构化数据类型，包含两个布尔字段 'a' 和 'b'
        mdtype = np.dtype([('a', bool), ('b', bool)])
        # 创建一个控制数组，两行数据，每行分别为 (False, True) 和 (True, True)
        control = np.array([(False, True), (True, True)], dtype=mdtype)
        
        # 创建一个形状为 (2,) 的 MaskedArray 对象，所有元素为掩码状态
        a = np.ma.masked_all((2,), dtype=ndtype)
        # 修改第 0 行数据的 'a' 字段为 2
        a['a'][0] = 2
        # 验证掩码的修改结果
        assert_equal(a.mask, control)
        
        # 创建一个形状为 (2,) 的 MaskedArray 对象，所有元素为掩码状态
        a = np.ma.masked_all((2,), dtype=ndtype)
        # 修改第 0 行数据的 'a' 字段为 2
        a[0]['a'] = 2
        # 验证掩码的修改结果
        assert_equal(a.mask, control)
        
        # 创建一个形状为 (2,) 的 MaskedArray 对象，所有元素为掩码状态
        a = np.ma.masked_all((2,), dtype=ndtype)
        # 将掩码硬化（永久性应用）
        a.harden_mask()
        # 修改第 0 行数据的 'a' 字段为 2
        a['a'][0] = 2
        # 验证掩码的修改结果
        assert_equal(a.mask, control)
        
        # 创建一个形状为 (2,) 的 MaskedArray 对象，所有元素为掩码状态
        a = np.ma.masked_all((2,), dtype=ndtype)
        # 将掩码硬化（永久性应用）
        a.harden_mask()
        # 修改第 0 行数据的 'a' 字段为 2
        a[0]['a'] = 2
        # 验证掩码的修改结果
        assert_equal(a.mask, control)

    def test_setitem_scalar(self):
        # 8510
        # 创建一个标量的 MaskedArray 对象，值为 1，完全掩码
        mask_0d = np.ma.masked_array(1, mask=True)
        # 创建一个形状为 (3,) 的 MaskedArray 对象，元素值为 0, 1, 2
        arr = np.ma.arange(3)
        # 将第 0 个元素设置为 mask_0d
        arr[0] = mask_0d
        # 验证掩码的设置结果
        assert_array_equal(arr.mask, [True, False, False])

    def test_element_len(self):
        # check that len() works for mvoid (Github issue #576)
        # 遍历 self.data['base'] 中的每条记录
        for rec in self.data['base']:
            # 验证 len() 函数对 mvoid 类型的记录的正确工作
            assert_equal(len(rec), len(self.data['ddtype']))
class TestMaskedObjectArray:

    def test_getitem(self):
        # 创建一个带有两个 None 元素的屏蔽数组
        arr = np.ma.array([None, None])
        
        # 对于每种数据类型 float 和 object 进行循环测试
        for dt in [float, object]:
            # 创建并设置两个单位矩阵的数组，转换为指定数据类型
            a0 = np.eye(2).astype(dt)
            a1 = np.eye(3).astype(dt)
            
            # 将数组 a0 和 a1 分别赋值给屏蔽数组的第一个和第二个元素
            arr[0] = a0
            arr[1] = a1

            # 断言：确保 arr[0] 和 a0 是同一个对象
            assert_(arr[0] is a0)
            # 断言：确保 arr[1] 和 a1 是同一个对象
            assert_(arr[1] is a1)
            # 断言：确保 arr[0,...] 是一个 MaskedArray 类型对象
            assert_(isinstance(arr[0,...], MaskedArray))
            # 断言：确保 arr[1,...] 是一个 MaskedArray 类型对象
            assert_(isinstance(arr[1,...], MaskedArray))
            # 断言：确保 arr[0,...][()] 和 a0 是同一个对象
            assert_(arr[0,...][()] is a0)
            # 断言：确保 arr[1,...][()] 和 a1 是同一个对象
            assert_(arr[1,...][()] is a1)

            # 将 arr[0] 设为 np.ma.masked
            arr[0] = np.ma.masked

            # 断言：确保 arr[1] 和 a1 是同一个对象
            assert_(arr[1] is a1)
            # 断言：确保 arr[0,...] 是一个 MaskedArray 类型对象
            assert_(isinstance(arr[0,...], MaskedArray))
            # 断言：确保 arr[1,...] 是一个 MaskedArray 类型对象
            assert_(isinstance(arr[1,...], MaskedArray))
            # 断言：确保 arr[0,...].mask 是 True
            assert_equal(arr[0,...].mask, True)
            # 断言：确保 arr[1,...][()] 和 a1 是同一个对象
            assert_(arr[1,...][()] is a1)

            # gh-5962 - 对象数组的数组执行特殊操作
            # 断言：确保 arr[0].data 和 a0 是相等的
            assert_equal(arr[0].data, a0)
            # 断言：确保 arr[0].mask 是 True
            assert_equal(arr[0].mask, True)
            # 断言：确保 arr[0,...][()].data 和 a0 是相等的
            assert_equal(arr[0,...][()].data, a0)
            # 断言：确保 arr[0,...][()].mask 是 True
            assert_equal(arr[0,...][()].mask, True)

    def test_nested_ma(self):
        # 创建一个带有两个 None 元素的屏蔽数组
        arr = np.ma.array([None, None])
        
        # 将第一个对象设置为未屏蔽的屏蔽常量。有点繁琐
        arr[0,...] = np.array([np.ma.masked], object)[0,...]

        # 断言：检查上述行是否达到了预期的效果
        assert_(arr.data[0] is np.ma.masked)

        # 断言：测试获取的项是否返回了相同的对象
        assert_(arr[0] is np.ma.masked)

        # 现在屏蔽已屏蔽的值！
        arr[0] = np.ma.masked
        assert_(arr[0] is np.ma.masked)


class TestMaskedView:

    def setup_method(self):
        # 创建一个包含 10 个元素的迭代器，并与随机数数组合并为数据数组
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        # 使用元组结构创建一个结构化数组 a，带有两列 'a' 和 'b' 的数据类型为 float
        a = array(iterator, dtype=[('a', float), ('b', float)])
        # 将 a 的第一个元素的掩码设置为 (1, 0)
        a.mask[0] = (1, 0)
        # 创建一个包含 20 个元素，第一个元素为 True，其余为 False 的掩码数组
        controlmask = np.array([1] + 19 * [0], dtype=bool)
        # 将数据存储在实例变量中
        self.data = (data, a, controlmask)

    def test_view_to_nothing(self):
        # 解包实例变量
        (data, a, controlmask) = self.data
        # 创建 a 的视图对象 test
        test = a.view()
        # 断言：确保 test 是一个 MaskedArray 类型对象
        assert_(isinstance(test, MaskedArray))
        # 断言：确保 test 的数据与 a 的数据相同
        assert_equal(test._data, a._data)
        # 断言：确保 test 的掩码与 a 的掩码相同
        assert_equal(test._mask, a._mask)

    def test_view_to_type(self):
        # 解包实例变量
        (data, a, controlmask) = self.data
        # 创建以 np.ndarray 类型为目标的 a 的视图对象 test
        test = a.view(np.ndarray)
        # 断言：确保 test 不是一个 MaskedArray 类型对象
        assert_(not isinstance(test, MaskedArray))
        # 断言：确保 test 与 a 的数据相同
        assert_equal(test, a._data)
        # 断言：确保 test 与 data 的视图（去除维度为 1 的维度）相同
        assert_equal_records(test, data.view(a.dtype).squeeze())

    def test_view_to_simple_dtype(self):
        # 解包实例变量
        (data, a, controlmask) = self.data
        # 全局视图转换为 float 类型
        test = a.view(float)
        # 断言：确保 test 是一个 MaskedArray 类型对象
        assert_(isinstance(test, MaskedArray))
        # 断言：确保 test 与 data 的展平版本相同
        assert_equal(test, data.ravel())
        # 断言：确保 test 的掩码与 controlmask 相同
        assert_equal(test.mask, controlmask)
    def test_view_to_flexible_dtype(self):
        # 解构赋值，从 self.data 中获取 data, a, controlmask
        (data, a, controlmask) = self.data
        
        # 创建一个视图，包含字段 ('A', float) 和 ('B', float)
        test = a.view([('A', float), ('B', float)])
        # 断言视图的掩码 dtype 的字段名为 ('A', 'B')
        assert_equal(test.mask.dtype.names, ('A', 'B'))
        # 断言视图中的字段 'A' 与 a['a'] 相等
        assert_equal(test['A'], a['a'])
        # 断言视图中的字段 'B' 与 a['b'] 相等
        assert_equal(test['B'], a['b'])

        # 创建一个视图，仅包含第一个元素的字段 ('A', float) 和 ('B', float)
        test = a[0].view([('A', float), ('B', float)])
        # 断言 test 是 MaskedArray 类型的实例
        assert_(isinstance(test, MaskedArray))
        # 断言视图的掩码 dtype 的字段名为 ('A', 'B')
        assert_equal(test.mask.dtype.names, ('A', 'B'))
        # 断言视图中的字段 'A' 与 a['a'][0] 相等
        assert_equal(test['A'], a['a'][0])
        # 断言视图中的字段 'B' 与 a['b'][0] 相等
        assert_equal(test['B'], a['b'][0])

        # 创建一个视图，仅包含最后一个元素的字段 ('A', float) 和 ('B', float)
        test = a[-1].view([('A', float), ('B', float)])
        # 断言 test 是 MaskedArray 类型的实例
        assert_(isinstance(test, MaskedArray))
        # 断言视图的 dtype 的字段名为 ('A', 'B')
        assert_equal(test.dtype.names, ('A', 'B'))
        # 断言视图中的字段 'A' 与 a['a'][-1] 相等
        assert_equal(test['A'], a['a'][-1])
        # 断言视图中的字段 'B' 与 a['b'][-1] 相等
        assert_equal(test['B'], a['b'][-1])

    def test_view_to_subdtype(self):
        # 解构赋值，从 self.data 中获取 data, a, controlmask
        (data, a, controlmask) = self.data
        # 全局视图，元素类型为 (float, 2)
        test = a.view((float, 2))
        # 断言 test 是 MaskedArray 类型的实例
        assert_(isinstance(test, MaskedArray))
        # 断言视图与 data 相等
        assert_equal(test, data)
        # 断言视图的掩码与 controlmask 重塑为 (-1, 2) 后相等
        assert_equal(test.mask, controlmask.reshape(-1, 2))
        
        # 对第一个被掩盖元素创建视图，元素类型为 (float, 2)
        test = a[0].view((float, 2))
        # 断言 test 是 MaskedArray 类型的实例
        assert_(isinstance(test, MaskedArray))
        # 断言视图与 data[0] 相等
        assert_equal(test, data[0])
        # 断言视图的掩码为 (1, 0)
        assert_equal(test.mask, (1, 0))
        
        # 对最后一个未被掩盖元素创建视图，元素类型为 (float, 2)
        test = a[-1].view((float, 2))
        # 断言 test 是 MaskedArray 类型的实例
        assert_(isinstance(test, MaskedArray))
        # 断言视图与 data[-1] 相等
        assert_equal(test, data[-1])

    def test_view_to_dtype_and_type(self):
        # 解构赋值，从 self.data 中获取 data, a, controlmask
        (data, a, controlmask) = self.data
        
        # 创建一个视图，元素类型为 (float, 2)，类型为 np.recarray
        test = a.view((float, 2), np.recarray)
        # 断言 test 与 data 相等
        assert_equal(test, data)
        # 断言 test 是 np.recarray 类型的实例
        assert_(isinstance(test, np.recarray))
        # 断言 test 不是 MaskedArray 类型的实例
        assert_(not isinstance(test, MaskedArray))
class TestOptionalArgs:
    def test_ndarrayfuncs(self):
        # test axis arg behaves the same as ndarray (including multiple axes)

        # 创建一个形状为 (2, 3, 4) 的 ndarray，包含从 0 到 23 的浮点数
        d = np.arange(24.0).reshape((2,3,4))
        # 创建一个形状为 (2, 3, 4) 的布尔类型的 mask，将最后一个维度的最后一个元素标记为 True
        m = np.zeros(24, dtype=bool).reshape((2,3,4))
        m[:,:,-1] = True
        # 使用 mask 创建一个 masked array 对象
        a = np.ma.array(d, mask=m)

        def testaxis(f, a, d):
            numpy_f = numpy.__getattribute__(f)
            ma_f = np.ma.__getattribute__(f)

            # 测试 axis 参数
            assert_equal(ma_f(a, axis=1)[...,:-1], numpy_f(d[...,:-1], axis=1))
            assert_equal(ma_f(a, axis=(0,1))[...,:-1],
                         numpy_f(d[...,:-1], axis=(0,1)))

        def testkeepdims(f, a, d):
            numpy_f = numpy.__getattribute__(f)
            ma_f = np.ma.__getattribute__(f)

            # 测试 keepdims 参数
            assert_equal(ma_f(a, keepdims=True).shape,
                         numpy_f(d, keepdims=True).shape)
            assert_equal(ma_f(a, keepdims=False).shape,
                         numpy_f(d, keepdims=False).shape)

            # 同时测试 axis 和 keepdims 参数
            assert_equal(ma_f(a, axis=1, keepdims=True)[...,:-1],
                         numpy_f(d[...,:-1], axis=1, keepdims=True))
            assert_equal(ma_f(a, axis=(0,1), keepdims=True)[...,:-1],
                         numpy_f(d[...,:-1], axis=(0,1), keepdims=True))

        # 对于函数列表中的每个函数名 f，依次调用 testaxis 和 testkeepdims 函数
        for f in ['sum', 'prod', 'mean', 'var', 'std']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)

        # 对于函数列表中的每个函数名 f，仅调用 testaxis 函数
        for f in ['min', 'max']:
            testaxis(f, a, d)

        # 重新定义 d 为一个布尔类型的 ndarray，用于测试 'all' 和 'any' 函数
        d = (np.arange(24).reshape((2,3,4)) % 2 == 0)
        # 重新创建 masked array 对象 a，使用新的 d 和之前定义的 mask m
        a = np.ma.array(d, mask=m)
        # 对于函数列表中的每个函数名 f，依次调用 testaxis 和 testkeepdims 函数
        for f in ['all', 'any']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)
    # 定义测试方法 test_count，用于测试 np.ma.count 特性

    # 创建一个二维数组 d，包含数字 0 到 23，形状为 (2,3,4)
    d = np.arange(24.0).reshape((2,3,4))
    # 创建一个与 d 相同形状的布尔掩码数组 m，并将第一列设置为 True
    m = np.zeros(24, dtype=bool).reshape((2,3,4))
    m[:,0,:] = True
    # 使用 d 和 m 创建一个掩码数组 a
    a = np.ma.array(d, mask=m)

    # 测试 count 函数，断言其结果为 16
    assert_equal(count(a), 16)
    # 测试 count 函数在 axis=1 上的结果，期望是形状为 (2,4) 的全 2 数组
    assert_equal(count(a, axis=1), 2*ones((2,4)))
    # 测试 count 函数在 axis=(0,1) 上的结果，期望是形状为 (4,) 的全 4 数组
    assert_equal(count(a, axis=(0,1)), 4*ones((4,)))
    # 测试 count 函数在 keepdims=True 下的结果，期望是形状为 (1,1,1) 的全 16 数组
    assert_equal(count(a, keepdims=True), 16*ones((1,1,1)))
    # 测试 count 函数在 axis=1 和 keepdims=True 下的结果，期望是形状为 (2,1,4) 的全 2 数组
    assert_equal(count(a, axis=1, keepdims=True), 2*ones((2,1,4)))
    # 测试 count 函数在 axis=(0,1) 和 keepdims=True 下的结果，期望是形状为 (1,1,4) 的全 4 数组
    assert_equal(count(a, axis=(0,1), keepdims=True), 4*ones((1,1,4)))
    # 测试 count 函数在 axis=-2 上的结果，与 axis=1 相同，期望是形状为 (2,4) 的全 2 数组
    assert_equal(count(a, axis=-2), 2*ones((2,4)))
    # 测试 count 函数在非法的 axis=(1,1) 参数下是否引发 ValueError 异常
    assert_raises(ValueError, count, a, axis=(1,1))
    # 测试 count 函数在超出数组维度的 axis=3 参数下是否引发 AxisError 异常
    assert_raises(AxisError, count, a, axis=3)

    # 使用 nomask 创建一个未掩码的数组 a
    a = np.ma.array(d, mask=nomask)

    # 测试 count 函数在未掩码数组下的结果，期望是 24
    assert_equal(count(a), 24)
    # 测试 count 函数在 axis=1 上的结果，期望是形状为 (2,4) 的全 3 数组
    assert_equal(count(a, axis=1), 3*ones((2,4)))
    # 测试 count 函数在 axis=(0,1) 上的结果，期望是形状为 (4,) 的全 6 数组
    assert_equal(count(a, axis=(0,1)), 6*ones((4,)))
    # 测试 count 函数在 keepdims=True 下的结果，期望是形状为 (1,1,1) 的全 24 数组
    assert_equal(count(a, keepdims=True), 24*ones((1,1,1)))
    # 测试 count 函数在 keepdims=True 下返回结果的维度是否为 3
    assert_equal(np.ndim(count(a, keepdims=True)), 3)
    # 测试 count 函数在 axis=1 和 keepdims=True 下的结果，期望是形状为 (2,1,4) 的全 3 数组
    assert_equal(count(a, axis=1, keepdims=True), 3*ones((2,1,4)))
    # 测试 count 函数在 axis=(0,1) 和 keepdims=True 下的结果，期望是形状为 (1,1,4) 的全 6 数组
    assert_equal(count(a, axis=(0,1), keepdims=True), 6*ones((1,1,4)))
    # 测试 count 函数在 axis=-2 上的结果，与 axis=1 相同，期望是形状为 (2,4) 的全 3 数组
    assert_equal(count(a, axis=-2), 3*ones((2,4)))
    # 测试 count 函数在非法的 axis=(1,1) 参数下是否引发 ValueError 异常
    assert_raises(ValueError, count, a, axis=(1,1))
    # 测试 count 函数在超出数组维度的 axis=3 参数下是否引发 AxisError 异常
    assert_raises(AxisError, count, a, axis=3)

    # 测试 count 函数在空 masked 数组上的结果，期望为 0
    assert_equal(count(np.ma.masked), 0)

    # 测试 0 维数组不允许使用 axis > 0 参数是否引发 AxisError 异常
    assert_raises(AxisError, count, np.ma.array(1), axis=1)
class TestMaskedConstant:
    # 测试用例类，用于验证 numpy 中的 MaskedConstant 相关功能

    def _do_add_test(self, add):
        # 执行加法测试函数
        # 检查当第一个参数为 np.ma.masked 时，加法操作返回仍为 np.ma.masked
        assert_(add(np.ma.masked, 1) is np.ma.masked)

        # 使用向量进行加法测试
        vector = np.array([1, 2, 3])
        result = add(np.ma.masked, vector)

        # 验证加法结果不是 np.ma.masked，并且不是 MaskedConstant 类型
        assert_(result is not np.ma.masked)
        assert_(not isinstance(result, np.ma.core.MaskedConstant))
        assert_equal(result.shape, vector.shape)
        assert_equal(np.ma.getmask(result), np.ones(vector.shape, dtype=bool))

    def test_ufunc(self):
        # 测试使用 numpy 的通用函数（ufunc）
        self._do_add_test(np.add)

    def test_operator(self):
        # 测试使用 lambda 函数作为操作符
        self._do_add_test(lambda a, b: a + b)

    def test_ctor(self):
        # 测试创建 MaskedArray 实例
        m = np.ma.array(np.ma.masked)

        # 确保不会创建新的 MaskedConstant 实例
        assert_(not isinstance(m, np.ma.core.MaskedConstant))
        assert_(m is not np.ma.masked)

    def test_repr(self):
        # 测试 MaskedConstant 实例的 repr 表示
        # 检查 repr(np.ma.masked) 的输出应为 'masked'
        assert_equal(repr(np.ma.masked), 'masked')

        # 以奇怪的方式创建一个新实例
        masked2 = np.ma.MaskedArray.__new__(np.ma.core.MaskedConstant)
        assert_not_equal(repr(masked2), 'masked')

    def test_pickle(self):
        # 测试序列化和反序列化 MaskedConstant 实例
        from io import BytesIO

        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with BytesIO() as f:
                pickle.dump(np.ma.masked, f, protocol=proto)
                f.seek(0)
                res = pickle.load(f)
            assert_(res is np.ma.masked)

    def test_copy(self):
        # 测试复制操作
        # 检查 np.ma.masked 的复制行为，应返回自身
        assert_equal(
            np.ma.masked.copy() is np.ma.masked,
            np.True_.copy() is np.True_)

    def test__copy(self):
        # 测试 copy 模块的复制功能
        import copy
        assert_(
            copy.copy(np.ma.masked) is np.ma.masked)

    def test_deepcopy(self):
        # 测试 copy 模块的深度复制功能
        import copy
        assert_(
            copy.deepcopy(np.ma.masked) is np.ma.masked)

    def test_immutable(self):
        # 测试不可变性
        orig = np.ma.masked
        # 尝试修改 MaskedConstant 实例的数据或掩码，应引发异常
        assert_raises(np.ma.core.MaskError, operator.setitem, orig, (), 1)
        assert_raises(ValueError, operator.setitem, orig.data, (), 1)
        assert_raises(ValueError, operator.setitem, orig.mask, (), False)

        # 尝试修改 MaskedArray 视图的数据或掩码，应引发异常
        view = np.ma.masked.view(np.ma.MaskedArray)
        assert_raises(ValueError, operator.setitem, view, (), 1)
        assert_raises(ValueError, operator.setitem, view.data, (), 1)
        assert_raises(ValueError, operator.setitem, view.mask, (), False)

    def test_coercion_int(self):
        # 测试强制类型转换为整数时的行为
        a_i = np.zeros((), int)
        # 尝试将 np.ma.masked 赋值给整数数组，应引发 MaskError
        assert_raises(MaskError, operator.setitem, a_i, (), np.ma.masked)
        # 尝试将 np.ma.masked 转换为整数，应引发 MaskError
        assert_raises(MaskError, int, np.ma.masked)

    def test_coercion_float(self):
        # 测试强制类型转换为浮点数时的行为
        a_f = np.zeros((), float)
        # 尝试将 np.ma.masked 赋值给浮点数数组，应引发 UserWarning
        assert_warns(UserWarning, operator.setitem, a_f, (), np.ma.masked)
        # 检查赋值后的值应为 NaN
        assert_(np.isnan(a_f[()]))
    # 将此测试标记为预期失败，原因是 Github 问题编号 gh-9750
    @pytest.mark.xfail(reason="See gh-9750")
    def test_coercion_unicode(self):
        # 创建一个 U10 类型的空数组
        a_u = np.zeros((), 'U10')
        # 将数组的第一个元素赋值为 np.ma.masked
        a_u[()] = np.ma.masked
        # 断言数组的第一个元素为 '--'
        assert_equal(a_u[()], '--')
    
    # 将此测试标记为预期失败，原因是 Github 问题编号 gh-9750
    @pytest.mark.xfail(reason="See gh-9750")
    def test_coercion_bytes(self):
        # 创建一个 S10 类型的空数组
        a_b = np.zeros((), 'S10')
        # 将数组的第一个元素赋值为 np.ma.masked
        a_b[()] = np.ma.masked
        # 断言数组的第一个元素为 b'--'
        assert_equal(a_b[()], b'--')
    
    def test_subclass(self):
        # 创建一个继承自 np.ma.masked 类型的子类，参考 GitHub 问题编号 #6645
        class Sub(type(np.ma.masked)): pass
    
        # 实例化 Sub 类
        a = Sub()
        # 断言 a 是 Sub 类的实例
        assert_(a is Sub())
        # 断言 a 不是 np.ma.masked 的实例
        assert_(a is not np.ma.masked)
        # 断言 a 的字符串表示形式不等于 'masked'
        assert_not_equal(repr(a), 'masked')
    
    def test_attributes_readonly(self):
        # 断言尝试设置 np.ma.masked 的 'shape' 属性会抛出 AttributeError 异常
        assert_raises(AttributeError, setattr, np.ma.masked, 'shape', (1,))
        # 断言尝试设置 np.ma.masked 的 'dtype' 属性会抛出 AttributeError 异常
        assert_raises(AttributeError, setattr, np.ma.masked, 'dtype', np.int64)
class TestMaskedWhereAliases:
    # 定义一个测试类 TestMaskedWhereAliases，用于测试 masked_object、masked_equal 等功能

    # 定义测试方法 test_masked_values，测试 masked_values 函数的功能
    def test_masked_values(self):
        # 测试用例1：输入 np.array([-32768.0]) 和 np.int16(-32768)，期望返回结果的 mask 属性为 [True]
        res = masked_values(np.array([-32768.0]), np.int16(-32768))
        assert_equal(res.mask, [True])

        # 测试用例2：输入 np.inf 和 np.inf，期望返回结果的 mask 属性为 True
        res = masked_values(np.inf, np.inf)
        assert_equal(res.mask, True)

        # 测试用例3：使用 np.ma.masked_values 函数，输入 np.inf 和 -np.inf，期望返回结果的 mask 属性为 False
        res = np.ma.masked_values(np.inf, -np.inf)
        assert_equal(res.mask, False)

        # 测试用例4：使用 np.ma.masked_values 函数，输入 [1, 2, 3, 4] 和 5，并设置 shrink=True，期望返回结果的 mask 属性为 np.ma.nomask
        res = np.ma.masked_values([1, 2, 3, 4], 5, shrink=True)
        assert_(res.mask is np.ma.nomask)

        # 测试用例5：使用 np.ma.masked_values 函数，输入 [1, 2, 3, 4] 和 5，并设置 shrink=False，期望返回结果的 mask 属性为 [False, False, False, False]
        res = np.ma.masked_values([1, 2, 3, 4], 5, shrink=False)
        assert_equal(res.mask, [False] * 4)


# 定义测试函数 test_masked_array
def test_masked_array():
    # 创建一个 masked array a，数据为 [0, 1, 2, 3]，mask 为 [0, 0, 1, 0]
    a = np.ma.array([0, 1, 2, 3], mask=[0, 0, 1, 0])
    # 使用 np.argwhere 函数测试 masked array a，期望返回 [[1], [3]]
    assert_equal(np.argwhere(a), [[1], [3]])


# 定义测试函数 test_masked_array_no_copy
def test_masked_array_no_copy():
    # 测试使用 np.ma.masked_where 函数，检查是否在原地更新 nomask 数组
    a = np.ma.array([1, 2, 3, 4])
    _ = np.ma.masked_where(a == 3, a, copy=False)
    assert_array_equal(a.mask, [False, False, True, False])

    # 测试使用 np.ma.masked_where 函数，检查是否在原地更新 masked array
    a = np.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 0])
    _ = np.ma.masked_where(a == 3, a, copy=False)
    assert_array_equal(a.mask, [True, False, True, False])

    # 测试使用 np.ma.masked_invalid 函数，检查是否在原地更新 masked array
    a = np.ma.array([np.inf, 1, 2, 3, 4])
    _ = np.ma.masked_invalid(a, copy=False)
    assert_array_equal(a.mask, [True, False, False, False, False])


# 定义测试函数 test_append_masked_array
def test_append_masked_array():
    # 使用 np.ma.masked_equal 创建 masked array a 和 b
    a = np.ma.masked_equal([1,2,3], value=2)
    b = np.ma.masked_equal([4,3,2], value=2)

    # 测试 np.ma.append 函数，将 a 和 b 连接起来
    result = np.ma.append(a, b)
    expected_data = [1, 2, 3, 4, 3, 2]
    expected_mask = [False, True, False, False, False, True]
    assert_array_equal(result.data, expected_data)
    assert_array_equal(result.mask, expected_mask)

    # 创建 masked array a 和 b，a 为全 mask，b 为全 1
    a = np.ma.masked_all((2,2))
    b = np.ma.ones((3,1))

    # 测试 np.ma.append 函数，将 a 和 b 沿 axis=None 方向连接起来
    result = np.ma.append(a, b)
    expected_data = [1] * 3
    expected_mask = [True] * 4 + [False] * 3
    assert_array_equal(result.data[-3], expected_data)
    assert_array_equal(result.mask, expected_mask)

    # 再次测试 np.ma.append 函数，将 a 和 b 沿 axis=None 方向连接起来
    result = np.ma.append(a, b, axis=None)
    assert_array_equal(result.data[-3], expected_data)
    assert_array_equal(result.mask, expected_mask)


# 定义测试函数 test_append_masked_array_along_axis
def test_append_masked_array_along_axis():
    # 使用 np.ma.masked_equal 创建 masked array a 和 b
    a = np.ma.masked_equal([1,2,3], value=2)
    b = np.ma.masked_values([[4, 5, 6], [7, 8, 9]], 7)

    # 当指定 axis 参数时，values 的形状必须正确
    assert_raises(ValueError, np.ma.append, a, b, axis=0)

    # 测试 np.ma.append 函数，将 a 和 b 沿 axis=0 方向连接起来
    result = np.ma.append(a[np.newaxis,:], b, axis=0)
    expected = np.ma.arange(1, 10)
    expected[[1, 6]] = np.ma.masked
    expected = expected.reshape((3,3))
    assert_array_equal(result.data, expected.data)
    assert_array_equal(result.mask, expected.mask)


# 定义测试函数 test_default_fill_value_complex
def test_default_fill_value_complex():
    # 测试 default_fill_value 函数，对于复数 1 + 1j，期望返回默认填充值
    assert_(default_fill_value(1 + 1j) == 1.e20 + 0.0j)


# 定义测试函数 test_ufunc_with_output
def test_ufunc_with_output():
    # 检查给定输出参数时，是否始终返回该输出
    # 进行 gh-8416 的回归测试（可能是一个 GitHub 问题编号），验证以下代码的正确性
    x = array([1., 2., 3.], mask=[0, 0, 1])
    # 使用 numpy 的 add 函数将数组 x 的每个元素都加上 1，将结果存储回 x 中
    y = np.add(x, 1., out=x)
    # 断言检查 y 和 x 是同一个对象（引用相同）
    assert_(y is x)
# 定义一个函数用于测试 ufunc 的多种输出变量组合是否正常工作
def test_ufunc_with_out_varied():
    """ Test that masked arrays are immune to gh-10459 """
    # 创建一个带有屏蔽值的数组 a
    a        = array([ 1,  2,  3], mask=[1, 0, 0])
    # 创建一个带有屏蔽值的数组 b
    b        = array([10, 20, 30], mask=[1, 0, 0])
    # 创建一个带有屏蔽值的输出数组 out
    out      = array([ 0,  0,  0], mask=[0, 0, 1])
    # 创建一个预期的输出结果数组 expected
    expected = array([11, 22, 33], mask=[1, 0, 0])

    # 复制 out 到 out_pos
    out_pos = out.copy()
    # 使用 np.add 函数计算 a 和 b 的和，将结果保存在 out_pos 中
    res_pos = np.add(a, b, out_pos)

    # 复制 out 到 out_kw
    out_kw = out.copy()
    # 使用 np.add 函数计算 a 和 b 的和，通过关键字参数 out 将结果保存在 out_kw 中
    res_kw = np.add(a, b, out=out_kw)

    # 复制 out 到 out_tup
    out_tup = out.copy()
    # 使用 np.add 函数计算 a 和 b 的和，通过元组形式的 out 将结果保存在 out_tup 中
    res_tup = np.add(a, b, out=(out_tup,))

    # 断言结果 res_kw 的屏蔽值和数据与预期结果 expected 相等
    assert_equal(res_kw.mask,  expected.mask)
    assert_equal(res_kw.data,  expected.data)
    # 断言结果 res_tup 的屏蔽值和数据与预期结果 expected 相等
    assert_equal(res_tup.mask, expected.mask)
    assert_equal(res_tup.data, expected.data)
    # 断言结果 res_pos 的屏蔽值和数据与预期结果 expected 相等
    assert_equal(res_pos.mask, expected.mask)
    assert_equal(res_pos.data, expected.data)


# 定义一个函数用于测试 dtype 转换中的屏蔽值保持顺序是否正常工作
def test_astype_mask_ordering():
    descr = np.dtype([('v', int, 3), ('x', [('y', float)])])
    x = array([
        [([1, 2, 3], (1.0,)),  ([1, 2, 3], (2.0,))],
        [([1, 2, 3], (3.0,)),  ([1, 2, 3], (4.0,))]], dtype=descr)
    # 将 x 的第一个元素的 'v' 字段的第一个元素设置为屏蔽值
    x[0]['v'][0] = np.ma.masked

    # 使用 dtype 描述符对 x 进行类型转换，得到 x_a
    x_a = x.astype(descr)
    # 断言 x_a 的 dtype 的字段名与描述符 descr 的字段名相同
    assert x_a.dtype.names == np.dtype(descr).names
    # 断言 x_a 的屏蔽值 dtype 的字段名与描述符 descr 的字段名相同
    assert x_a.mask.dtype.names == np.dtype(descr).names
    # 断言 x_a 与原始数组 x 相等
    assert_equal(x, x_a)

    # 断言 x 是通过非复制方式进行自身 dtype 转换的
    assert_(x is x.astype(x.dtype, copy=False))
    # 断言 x.astype 后返回的类型为 np.ndarray
    assert_equal(type(x.astype(x.dtype, subok=False)), np.ndarray)

    # 使用 order='F' 参数对 x 进行 dtype 转换，得到 x_f
    x_f = x.astype(x.dtype, order='F')
    # 断言 x_f 是列优先存储的
    assert_(x_f.flags.f_contiguous)
    # 断言 x_f 的屏蔽值是列优先存储的
    assert_(x_f.mask.flags.f_contiguous)

    # 通过 np.array 函数间接测试，使用描述符 dtype 对 x 进行类型转换，得到 x_a2
    x_a2 = np.array(x, dtype=descr, subok=True)
    # 断言 x_a2 的 dtype 的字段名与描述符 descr 的字段名相同
    assert x_a2.dtype.names == np.dtype(descr).names
    # 断言 x_a2 的屏蔽值 dtype 的字段名与描述符 descr 的字段名相同
    assert x_a2.mask.dtype.names == np.dtype(descr).names
    # 断言 x_a2 与原始数组 x 相等
    assert_equal(x, x_a2)

    # 断言 x 是通过非复制方式进行 np.array 转换的
    assert_(x is np.array(x, dtype=descr, copy=None, subok=True))

    # 通过 np.array 函数间接测试，使用 x 的 dtype 和 order='F' 参数对 x 进行类型转换，得到 x_f2
    x_f2 = np.array(x, dtype=x.dtype, order='F', subok=True)
    # 断言 x_f2 是列优先存储的
    assert_(x_f2.flags.f_contiguous)
    # 断言 x_f2 的屏蔽值是列优先存储的


# 使用参数化测试 dt1 和 dt2 进行基本的 dtype 转换测试
@pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
@pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
@pytest.mark.filterwarnings('ignore::numpy.exceptions.ComplexWarning')
def test_astype_basic(dt1, dt2):
    # 参见 gh-12070
    # 创建一个带有填充值为 1 的长度为 3 的 dt1 类型的 masked array src
    src = np.ma.array(ones(3, dt1), fill_value=1)
    # 将 src 转换为 dt2 类型，得到 dst
    dst = src.astype(dt2)

    # 断言 src 的填充值为 1
    assert_(src.fill_value == 1)
    # 断言 src 的 dtype 为 dt1
    assert_(src.dtype == dt1)
    # 断言 src 的填充值的 dtype 为 dt1
    assert_(src.fill_value.dtype == dt1)

    # 断言 dst 的填充值为 1
    assert_(dst.fill_value == 1)
    # 断言 dst 的 dtype 为 dt2
    assert_(dst.dtype == dt2)
    # 断言 dst 的填充值的 dtype 为 dt2

    # 断言 src 和 dst 相等
    assert_equal(src, dst)


# 定义一个函数用于测试没有字段的 void dtype
def test_fieldless_void():
    dt = np.dtype([])  # 创建一个没有字段的 void dtype
    x = np.empty(4, dt)

    # 创建一个 x 的 masked array mx
    mx = np.ma.array(x)
    # 断言 mx 的 dtype 与 x 的 dtype 相同
    assert_equal(mx.dtype, x.dtype)
    # 断言 mx 的形状与 x 的形状相同
    assert_equal(mx.shape, x.shape)

    # 创建一个带有屏蔽值的 masked array mx
    mx = np.ma.array(x, mask=x)
    # 断言 mx 的 dtype 与 x 的 dtype 相同
    assert_equal(mx.dtype, x.dtype)
    # 断言 mx 的形状与 x 的形状相同


# 定义一个函数用于测试屏蔽值形状赋值不会破坏 masked array
def test_mask_shape_assignment_does_not_break_masked():
    a = np.ma.masked
    # 创建一个 NumPy 的掩码数组 `b`，其数据部分为 1，掩码与给定数组 `a` 的掩码相同
    b = np.ma.array(1, mask=a.mask)
    
    # 将 `b` 的形状修改为 (1,)
    b.shape = (1,)
    
    # 使用断言确保数组 `a` 的掩码形状为一个空元组 `()`
    assert_equal(a.mask.shape, ())
@pytest.mark.skipif(sys.flags.optimize > 1,
                    reason="no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1")
# 定义测试函数 test_doc_note，如果 Python 优化标志大于 1 则跳过测试，原因是缺少文档字符串无法检查
def test_doc_note():
    # 定义内部方法 method，包含多行文档字符串和 pass 语句
    def method(self):
        """This docstring

        Has multiple lines

        And notes

        Notes
        -----
        original note
        """
        pass

    # 预期的文档字符串
    expected_doc = """This docstring

Has multiple lines

And notes

Notes
-----
note

original note"""
    
    # 断言 method 的文档字符串经过处理后与预期的文档字符串相等
    assert_equal(np.ma.core.doc_note(method.__doc__, "note"), expected_doc)


# 定义测试函数 test_gh_22556，用于测试 numpy 中的特定功能
def test_gh_22556():
    # 创建包含对象数组的 numpy 掩码数组 source
    source = np.ma.array([0, [0, 1, 2]], dtype=object)
    # 深拷贝 source 到 deepcopy
    deepcopy = copy.deepcopy(source)
    # 修改 deepcopy 中的第二个元素的子列表
    deepcopy[1].append('this should not appear in source')
    # 断言 source 的第二个元素的长度为 3
    assert len(source[1]) == 3


# 定义测试函数 test_gh_21022，测试 numpy 中的特定功能并验证错误报告是否不存在
def test_gh_21022():
    # 创建具有掩码的 numpy 掩码数组 source
    source = np.ma.masked_array(data=[-1, -1], mask=True, dtype=np.float64)
    # 创建 axis 数组
    axis = np.array(0)
    # 在指定轴上计算 source 的乘积并存储为 result
    result = np.prod(source, axis=axis, keepdims=False)
    # 创建具有布尔掩码的 numpy 掩码数组 result
    result = np.ma.masked_array(result,
                                mask=np.ones(result.shape, dtype=np.bool))
    # 创建具有掩码的 numpy 掩码数组 array
    array = np.ma.masked_array(data=-1, mask=True, dtype=np.float64)
    # 对 array 和 result 进行深拷贝
    copy.deepcopy(array)
    copy.deepcopy(result)


# 定义测试函数 test_deepcopy_2d_obj，测试 numpy 中对二维对象数组的深拷贝
def test_deepcopy_2d_obj():
    # 创建二维对象数组 source 和其对应的掩码数组
    source = np.ma.array([[0, "dog"],
                          [1, 1],
                          [[1, 2], "cat"]],
                        mask=[[0, 1],
                              [0, 0],
                              [0, 0]],
                        dtype=object)
    # 对 source 进行深拷贝到 deepcopy
    deepcopy = copy.deepcopy(source)
    # 修改 deepcopy 中特定位置的元素并扩展其子列表
    deepcopy[2, 0].extend(['this should not appear in source', 3])
    # 断言 source 和 deepcopy 中特定元素的长度是否符合预期
    assert len(source[2, 0]) == 2
    assert len(deepcopy[2, 0]) == 4
    # 断言 deepcopy 的掩码与 source 的掩码是否相等
    assert_equal(deepcopy._mask, source._mask)
    # 修改 deepcopy 的掩码，并断言修改是否反映在 source 上
    deepcopy._mask[0, 0] = 1
    assert source._mask[0, 0] == 0


# 定义测试函数 test_deepcopy_0d_obj，测试 numpy 中对零维对象数组的深拷贝
def test_deepcopy_0d_obj():
    # 创建零维对象数组 source 和其对应的掩码
    source = np.ma.array(0, mask=[0], dtype=object)
    # 对 source 进行深拷贝到 deepcopy
    deepcopy = copy.deepcopy(source)
    # 修改 deepcopy 的所有元素为 17
    deepcopy[...] = 17
    # 断言 source 和 deepcopy 是否相等
    assert_equal(source, 0)
    assert_equal(deepcopy, 17)
```