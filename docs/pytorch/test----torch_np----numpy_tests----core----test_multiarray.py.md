# `.\pytorch\test\torch_np\numpy_tests\core\test_multiarray.py`

```
# Owner(s): ["module: dynamo"]

# 导入内置模块和第三方库
import builtins
import collections.abc
import ctypes
import functools
import io
import itertools
import mmap
import operator
import os
import sys
import tempfile
import warnings
import weakref
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from tempfile import mkstemp
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

# 导入 NumPy 和 Pytest 相关模块
import numpy
import pytest
from pytest import raises as assert_raises

# 导入 Torch Dynamo 测试相关工具
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    slowTest as slow,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
    xpassIfTorchDynamo,
)

# 如果需要使用 TorchDynamo 进行测试，则使用 NumPy 进行导入
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import (
        assert_,  # 断言函数，用于验证条件是否为真
        assert_allclose,  # 断言函数，用于比较两个数组是否几乎相等
        assert_almost_equal,  # 断言函数，用于比较两个数值是否几乎相等
        assert_array_almost_equal,  # 断言函数，用于比较两个数组是否几乎相等
        assert_array_equal,  # 断言函数，用于比较两个数组是否相等
        assert_array_less,  # 断言函数，用于验证一个数组是否小于另一个数组
        assert_equal,  # 断言函数，用于比较两个对象是否相等
        assert_raises_regex,  # 断言函数，用于验证是否引发指定异常和正则表达式匹配的消息
        assert_warns,  # 断言函数，用于验证是否引发指定警告
        suppress_warnings,  # 上下文管理器，用于抑制所有警告
    )

# 否则使用 Torch._numpy 进行导入
else:
    import torch._numpy as np
    from torch._numpy.testing import (
        assert_,  # 断言函数，用于验证条件是否为真
        assert_allclose,  # 断言函数，用于比较两个数组是否几乎相等
        assert_almost_equal,  # 断言函数，用于比较两个数值是否几乎相等
        assert_array_almost_equal,  # 断言函数，用于比较两个数组是否几乎相等
        assert_array_equal,  # 断言函数，用于比较两个数组是否相等
        assert_array_less,  # 断言函数，用于验证一个数组是否小于另一个数组
        assert_equal,  # 断言函数，用于比较两个对象是否相等
        assert_raises_regex,  # 断言函数，用于验证是否引发指定异常和正则表达式匹配的消息
        assert_warns,  # 断言函数，用于验证是否引发指定警告
        suppress_warnings,  # 上下文管理器，用于抑制所有警告
    )

# 定义 skip 作为 skipif 函数的偏函数，总是返回 True
skip = functools.partial(skipif, True)

# 初始化一些常量
IS_PYPY = False
IS_PYSTON = False
HAS_REFCOUNT = True

# 导入 NumPy 测试相关模块
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.testing._private.utils import _no_tracing, requires_memory


# #### stubs to make pytest pass the collections stage ####

# 定义 runstring 函数，执行给定代码字符串
def runstring(astr, dict):
    exec(astr, dict)


# 临时文件路径的上下文管理器
@contextmanager
def temppath(*args, **kwargs):
    """Context manager for temporary files.

    Context manager that returns the path to a closed temporary file. Its
    parameters are the same as for tempfile.mkstemp and are passed directly
    to that function. The underlying file is removed when the context is
    exited, so it should be closed at that time.

    Windows does not allow a temporary file to be opened if it is already
    open, so the underlying file must be closed after opening before it
    can be opened again.

    """
    fd, path = mkstemp(*args, **kwargs)  # 创建临时文件
    os.close(fd)  # 关闭文件描述符
    try:
        yield path  # 返回临时文件路径
    finally:
        os.remove(path)  # 删除临时文件

# FIXME: 将 np.asanyarray 和 np.asfortranarray 定义为 np.asarray 的别名
np.asanyarray = np.asarray
np.asfortranarray = np.asarray

# #### end stubs


# 定义函数 _aligned_zeros，分配一个具有对齐内存的新 ndarray
def _aligned_zeros(shape, dtype=float, order="C", align=None):
    """
    Allocate a new ndarray with aligned memory.

    Parameters:
    - shape: ndarray 的形状
    - dtype: 数据类型，默认为 float
    - order: 存储顺序，默认为 "C"
    - align: 内存对齐方式，默认为 None

    """
    # 根据给定的 dtype 创建一个 ndarray 对象，根据需要对其进行内存对齐
    def _empty_like_dispatcher(dtype, shape, order, *, align=None):
        # 将 dtype 转换为 numpy 的数据类型对象
        dtype = np.dtype(dtype)
        # 如果 dtype 是 object 类型，则无法指定对齐方式，因此使用标准分配
        if dtype == np.dtype(object):
            # 无法指定对象数组的对齐方式，因此抛出异常
            if align is not None:
                raise ValueError("object array alignment not supported")
            # 返回一个以 dtype 和 order 指定的零数组
            return np.zeros(shape, dtype=dtype, order=order)
        
        # 如果 align 没有指定，则使用 dtype 的默认对齐方式
        if align is None:
            align = dtype.alignment
        
        # 如果 shape 不是一个长度，转换为元组
        if not hasattr(shape, "__len__"):
            shape = (shape,)
        
        # 计算所需的内存大小
        size = functools.reduce(operator.mul, shape) * dtype.itemsize
        
        # 分配内存空间，额外分配 2 * align + 1 字节，类型为 uint8
        buf = np.empty(size + 2 * align + 1, np.uint8)
        
        # 获取 buf 的数据指针地址
        ptr = buf.__array_interface__["data"][0]
        # 计算当前指针地址相对于指定对齐方式的偏移量
        offset = ptr % align
        if offset != 0:
            offset = align - offset
        # 如果当前指针地址恰好是 2 * align 的倍数，则增加一个对齐偏移量
        if (ptr % (2 * align)) == 0:
            offset += align
        
        # 根据计算的偏移量，截取 buf 的一部分作为数据存储区域，填充为 0
        buf = buf[offset : offset + size + 1][:-1]
        buf.fill(0)
        
        # 使用 buf 创建一个新的 ndarray 对象作为结果数据，指定 dtype 和 order
        data = np.ndarray(shape, dtype, buf, order=order)
        return data
# 使用装饰器 @xpassIfTorchDynamo 并附带参数 reason="TODO: flags"，但未提供实际代码实现
@xpassIfTorchDynamo  # (reason="TODO: flags")

# 使用装饰器 @instantiate_parametrized_tests 实例化参数化测试类
@instantiate_parametrized_tests
class TestFlag(TestCase):

    # 在每个测试方法执行前执行的初始化方法
    def setUp(self):
        self.a = np.arange(10)

    # 标记为预期失败的测试方法
    @xfail
    def test_writeable(self):
        # 获取当前局部变量字典
        mydict = locals()
        # 设置数组属性为不可写，验证运行时会抛出 ValueError 异常
        self.a.flags.writeable = False
        assert_raises(ValueError, runstring, "self.a[0] = 3", mydict)
        assert_raises(ValueError, runstring, "self.a[0:1].itemset(3)", mydict)
        # 恢复数组属性为可写，并修改数组内容
        self.a.flags.writeable = True
        self.a[0] = 5
        self.a[0] = 0

    # 测试任何基类为可写时，数组标志位也为可写的情况
    def test_writeable_any_base(self):
        # 创建一个普通的 NumPy 数组
        arr = np.arange(10)

        # 创建一个继承自 np.ndarray 的子类
        class subclass(np.ndarray):
            pass

        # 创建视图 view1 和 view2
        view1 = arr.view(subclass)
        view2 = view1[...]

        # 设置数组及其视图的属性为不可写，然后再设置为可写
        arr.flags.writeable = False
        view2.flags.writeable = False
        view2.flags.writeable = True  # 可以再次设置为 True

        # 使用接口创建一个数组
        arr = np.arange(10)

        class frominterface:
            def __init__(self, arr):
                self.arr = arr
                self.__array_interface__ = arr.__array_interface__

        # 创建视图 view1 和 view2
        view1 = np.asarray(frominterface)
        view2 = view1[...]

        # 设置数组及其视图的属性为不可写，然后再设置为可写
        view2.flags.writeable = False
        view2.flags.writeable = True

        # 验证设置数组及其视图属性为不可写后，再设置为可写时会抛出 ValueError 异常
        view1.flags.writeable = False
        view2.flags.writeable = False
        with assert_raises(ValueError):
            view2.flags.writeable = True

    # 测试从只读缓冲区创建的数组的可写性
    def test_writeable_from_readonly(self):
        # 创建一个只包含 100 个字节的只读数据
        data = b"\x00" * 100
        vals = np.frombuffer(data, "B")
        # 验证无法将只读数组设置为可写
        assert_raises(ValueError, vals.setflags, write=True)
        types = np.dtype([("vals", "u1"), ("res3", "S4")])
        values = np.core.records.fromstring(data, types)
        vals = values["vals"]
        # 验证无法将只读数组设置为可写
        assert_raises(ValueError, vals.setflags, write=True)

    # 测试从缓冲区创建的数组的可写性
    def test_writeable_from_buffer(self):
        # 创建一个包含 100 个字节的可变字节数组
        data = bytearray(b"\x00" * 100)
        vals = np.frombuffer(data, "B")
        # 验证初始时数组为可写状态
        assert_(vals.flags.writeable)
        # 将数组设置为不可写，并验证设置成功
        vals.setflags(write=False)
        assert_(vals.flags.writeable is False)
        # 将数组设置为可写，并验证设置成功
        vals.setflags(write=True)
        assert_(vals.flags.writeable)

        types = np.dtype([("vals", "u1"), ("res3", "S4")])
        values = np.core.records.fromstring(data, types)
        vals = values["vals"]
        # 验证初始时数组为可写状态
        assert_(vals.flags.writeable)
        # 将数组设置为不可写，并验证设置成功
        vals.setflags(write=False)
        assert_(vals.flags.writeable is False)
        # 将数组设置为可写，并验证设置成功
        vals.setflags(write=True)
        assert_(vals.flags.writeable)

    # 根据条件跳过测试（例如在 PyPy 环境下跳过）
    @skipif(IS_PYPY, reason="PyPy always copies")
    def test_writeable_pickle(self):
        import pickle

        # Small arrays will be copied without setting base.
        # See condition for using PyArray_SetBaseObject in
        # array_setstate.
        # 创建一个包含1000个元素的NumPy数组
        a = np.arange(1000)
        # 使用不同的pickle协议序列化和反序列化数组，并验证其可写性和基础对象为字节流
        for v in range(pickle.HIGHEST_PROTOCOL):
            vals = pickle.loads(pickle.dumps(a, v))
            assert_(vals.flags.writeable)
            assert_(isinstance(vals.base, bytes))

    def test_warnonwrite(self):
        # 创建一个包含10个元素的NumPy数组
        a = np.arange(10)
        # 设置警告写入标志位为True
        a.flags._warn_on_write = True
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            # 修改数组中的元素，并验证警告是否只触发一次
            a[1] = 10
            a[2] = 10
            assert_(len(w) == 1)

    @parametrize(
        "flag, flag_value, writeable",
        [
            ("writeable", True, True),
            # 在废弃后删除_warn_on_write并简化参数化
            ("_warn_on_write", True, False),
            ("writeable", False, False),
        ],
    )
    def test_readonly_flag_protocols(self, flag, flag_value, writeable):
        # 创建一个包含10个元素的NumPy数组
        a = np.arange(10)
        # 设置数组的特定标志位
        setattr(a.flags, flag, flag_value)

        class MyArr:
            __array_struct__ = a.__array_struct__

        # 验证内存视图和数组接口中的只读属性
        assert memoryview(a).readonly is not writeable
        assert a.__array_interface__["data"][1] is not writeable
        # 验证从自定义数组结构创建的数组是否可写
        assert np.asarray(MyArr()).flags.writeable is writeable

    @xfail
    def test_otherflags(self):
        # 验证不同的数组标志位
        assert_equal(self.a.flags.carray, True)
        assert_equal(self.a.flags["C"], True)
        assert_equal(self.a.flags.farray, False)
        assert_equal(self.a.flags.behaved, True)
        assert_equal(self.a.flags.fnc, False)
        assert_equal(self.a.flags.forc, True)
        assert_equal(self.a.flags.owndata, True)
        assert_equal(self.a.flags.writeable, True)
        assert_equal(self.a.flags.aligned, True)
        assert_equal(self.a.flags.writebackifcopy, False)
        assert_equal(self.a.flags["X"], False)
        assert_equal(self.a.flags["WRITEBACKIFCOPY"], False)

    @xfail  # invalid dtype
    def test_string_align(self):
        # 创建一个字符串dtype的NumPy数组，并验证其对齐性
        a = np.zeros(4, dtype=np.dtype("|S4"))
        assert_(a.flags.aligned)
        # 对于非2的幂字节访问的情况，仍然被视为对齐
        a = np.zeros(5, dtype=np.dtype("|S4"))
        assert_(a.flags.aligned)

    @xfail  # structured dtypes
    def test_void_align(self):
        # 创建一个结构化dtype的NumPy数组，并验证其对齐性
        a = np.zeros(4, dtype=np.dtype([("a", "i4"), ("b", "i4")]))
        assert_(a.flags.aligned)
@xpassIfTorchDynamo  # (reason="TODO: hash")
class TestHash(TestCase):
    # see #3793
    # 测试整数类型的哈希函数
    def test_int(self):
        for st, ut, s in [
            (np.int8, np.uint8, 8),
            (np.int16, np.uint16, 16),
            (np.int32, np.uint32, 32),
            (np.int64, np.uint64, 64),
        ]:
            for i in range(1, s):
                assert_equal(
                    hash(st(-(2**i))), hash(-(2**i)), err_msg="%r: -2**%d" % (st, i)
                )
                assert_equal(
                    hash(st(2 ** (i - 1))),
                    hash(2 ** (i - 1)),
                    err_msg="%r: 2**%d" % (st, i - 1),
                )
                assert_equal(
                    hash(st(2**i - 1)),
                    hash(2**i - 1),
                    err_msg="%r: 2**%d - 1" % (st, i),
                )

                i = max(i - 1, 1)
                assert_equal(
                    hash(ut(2 ** (i - 1))),
                    hash(2 ** (i - 1)),
                    err_msg="%r: 2**%d" % (ut, i - 1),
                )
                assert_equal(
                    hash(ut(2**i - 1)),
                    hash(2**i - 1),
                    err_msg="%r: 2**%d - 1" % (ut, i),
                )


@xpassIfTorchDynamo  # (reason="TODO: hash")
class TestAttributes(TestCase):
    # 设置测试环境
    def setUp(self):
        self.one = np.arange(10)
        self.two = np.arange(20).reshape(4, 5)
        self.three = np.arange(60, dtype=np.float64).reshape(2, 5, 6)

    # 测试属性
    def test_attributes(self):
        assert_equal(self.one.shape, (10,))
        assert_equal(self.two.shape, (4, 5))
        assert_equal(self.three.shape, (2, 5, 6))
        self.three.shape = (10, 3, 2)
        assert_equal(self.three.shape, (10, 3, 2))
        self.three.shape = (2, 5, 6)
        assert_equal(self.one.strides, (self.one.itemsize,))
        num = self.two.itemsize
        assert_equal(self.two.strides, (5 * num, num))
        num = self.three.itemsize
        assert_equal(self.three.strides, (30 * num, 6 * num, num))
        assert_equal(self.one.ndim, 1)
        assert_equal(self.two.ndim, 2)
        assert_equal(self.three.ndim, 3)
        num = self.two.itemsize
        assert_equal(self.two.size, 20)
        assert_equal(self.two.nbytes, 20 * num)
        assert_equal(self.two.itemsize, self.two.dtype.itemsize)

    @xfailIfTorchDynamo  # use ndarray.tensor._base to track the base tensor
    # 测试属性（第二部分）
    def test_attributes_2(self):
        assert_equal(self.two.base, np.arange(20))

    # 测试 dtype 属性
    def test_dtypeattr(self):
        assert_equal(self.one.dtype, np.dtype(np.int_))
        assert_equal(self.three.dtype, np.dtype(np.float_))
        assert_equal(self.one.dtype.char, "l")
        assert_equal(self.three.dtype.char, "d")
        assert_(self.three.dtype.str[0] in "<>")
        assert_equal(self.one.dtype.str[1], "i")
        assert_equal(self.three.dtype.str[1], "f")
    # 测试函数：test_stridesattr
    def test_stridesattr(self):
        # 设置本地变量 x 等于 self.one
        x = self.one

        # 定义内部函数 make_array，用于创建 ndarray 对象
        def make_array(size, offset, strides):
            # 创建一个 numpy 数组，使用给定的参数和 buffer=x、dtype=int
            return np.ndarray(
                size,
                buffer=x,
                dtype=int,
                offset=offset * x.itemsize,  # 设置数组的偏移量
                strides=strides * x.itemsize,  # 设置数组的步幅
            )

        # 断言：调用 make_array(4, 4, -1) 应该返回 [4, 3, 2, 1]
        assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
        # 断言：调用 make_array(4, 4, -2) 应该引发 ValueError 异常
        assert_raises(ValueError, make_array, 4, 4, -2)
        # 断言：调用 make_array(4, 2, -1) 应该引发 ValueError 异常
        assert_raises(ValueError, make_array, 4, 2, -1)
        # 断言：调用 make_array(8, 3, 1) 应该引发 ValueError 异常
        assert_raises(ValueError, make_array, 8, 3, 1)
        # 断言：调用 make_array(8, 3, 0) 应该返回 [3, 3, 3, 3, 3, 3, 3, 3]
        assert_equal(make_array(8, 3, 0), np.array([3] * 8))
        # 检查在 gh-2503 中报告的行为
        assert_raises(ValueError, make_array, (2, 3), 5, np.array([-2, -3]))
        # 调用 make_array(0, 0, 10)，不做断言，仅测试是否能够成功执行

    # 测试函数：test_set_stridesattr
    def test_set_stridesattr(self):
        # 设置本地变量 x 等于 self.one
        x = self.one

        # 定义内部函数 make_array，用于创建 ndarray 对象
        def make_array(size, offset, strides):
            try:
                # 尝试创建一个 numpy 数组，使用给定的参数和 buffer=x、dtype=int
                r = np.ndarray([size], dtype=int, buffer=x, offset=offset * x.itemsize)
            except Exception as e:
                raise RuntimeError(e)  # 抛出 RuntimeError 异常
            # 设置数组的步幅为 strides * x.itemsize
            r.strides = strides = strides * x.itemsize
            return r

        # 断言：调用 make_array(4, 4, -1) 应该返回 [4, 3, 2, 1]
        assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
        # 断言：调用 make_array(7, 3, 1) 应该返回 [3, 4, 5, 6, 7, 8, 9]
        assert_equal(make_array(7, 3, 1), np.array([3, 4, 5, 6, 7, 8, 9]))
        # 断言：调用 make_array(4, 4, -2) 应该引发 ValueError 异常
        assert_raises(ValueError, make_array, 4, 4, -2)
        # 断言：调用 make_array(4, 2, -1) 应该引发 ValueError 异常
        assert_raises(ValueError, make_array, 4, 2, -1)
        # 断言：调用 make_array(8, 3, 1) 应该引发 RuntimeError 异常
        assert_raises(RuntimeError, make_array, 8, 3, 1)

        # 检查数组真实范围的使用。
        # 测试依赖于 as_strided 基础不暴露缓冲区。
        x = np.lib.stride_tricks.as_strided(np.arange(1), (10, 10), (0, 0))

        # 定义内部函数 set_strides，用于设置数组的步幅
        def set_strides(arr, strides):
            arr.strides = strides

        # 断言：调用 set_strides(x, (10 * x.itemsize, x.itemsize)) 应该引发 ValueError 异常
        assert_raises(ValueError, set_strides, x, (10 * x.itemsize, x.itemsize))

        # 测试偏移量计算：
        x = np.lib.stride_tricks.as_strided(
            np.arange(10, dtype=np.int8)[-1], shape=(10,), strides=(-1,)
        )
        # 断言：调用 set_strides(x[::-1], -1) 应该引发 ValueError 异常
        assert_raises(ValueError, set_strides, x[::-1], -1)
        # 对 x[::-1] 进行切片，并设置步幅为 1
        a = x[::-1]
        a.strides = 1
        # 对 a[::2] 进行切片，并设置步幅为 2
        a[::2].strides = 2

        # 测试 0 维数组
        arr_0d = np.array(0)
        # 设置 arr_0d 的步幅为空元组，应该引发 TypeError 异常
        arr_0d.strides = ()
        assert_raises(TypeError, set_strides, arr_0d, None)

    # 测试函数：test_fill
    def test_fill(self):
        # 遍历不同的数据类型 t
        for t in "?bhilqpBHILQPfdgFDGO":
            # 创建一个类型为 t、形状为 (3, 2, 1) 的空数组 x 和 y
            x = np.empty((3, 2, 1), t)
            y = np.empty((3, 2, 1), t)
            # 使用 fill 方法填充数组 x 和 y，将其值设为 1
            x.fill(1)
            y[...] = 1
            # 断言：数组 x 和 y 应该相等
            assert_equal(x, y)

    # 测试函数：test_fill_max_uint64
    def test_fill_max_uint64(self):
        # 创建一个类型为 np.uint64、形状为 (3, 2, 1) 的空数组 x 和 y
        x = np.empty((3, 2, 1), dtype=np.uint64)
        y = np.empty((3, 2, 1), dtype=np.uint64)
        value = 2**64 - 1
        # 将数组 y 的所有元素设为 value
        y[...] = value
        # 使用 fill 方法将数组 x 的所有元素设为 value
        x.fill(value)
        # 断言：数组 x 和 y 应该相等
        assert_array_equal(x, y)
    # 定义测试方法：填充结构化数组
    def test_fill_struct_array(self):
        # 从标量填充
        x = np.array([(0, 0.0), (1, 1.0)], dtype="i4,f8")
        x.fill(x[0])  # 使用第一个元素填充整个数组
        assert_equal(x["f1"][1], x["f1"][0])  # 断言：数组中第二个元素的第二列应与第一个元素的第二列相等

        # 从可以转换为标量的元组填充
        x = np.zeros(2, dtype=[("a", "f8"), ("b", "i4")])
        x.fill((3.5, -2))  # 使用元组 (3.5, -2) 填充数组
        assert_array_equal(x["a"], [3.5, 3.5])  # 断言：数组的 "a" 列应为 [3.5, 3.5]
        assert_array_equal(x["b"], [-2, -2])    # 断言：数组的 "b" 列应为 [-2, -2]

    # 定义测试方法：测试填充只读数组
    def test_fill_readonly(self):
        # gh-22922
        a = np.zeros(11)
        a.setflags(write=False)  # 将数组 a 设置为只读
        with pytest.raises(ValueError, match=".*read-only"):
            a.fill(0)  # 尝试填充只读数组 a，预期会抛出 ValueError 异常，错误消息中包含 "read-only"
@instantiate_parametrized_tests
class TestArrayConstruction(TestCase):
    def test_array(self):
        # 创建一个长度为6的全1数组
        d = np.ones(6)
        # 创建一个包含两个d数组的二维数组r，并验证其形状是否为(2, 6)
        r = np.array([d, d])
        assert_equal(r, np.ones((2, 6)))

        # 再次创建长度为6的全1数组d，并定义目标数组tgt为形状为(2, 6)的全1数组
        d = np.ones(6)
        tgt = np.ones((2, 6))
        # 创建一个包含两个d数组的二维数组r，并验证其是否与tgt相等
        r = np.array([d, d])
        assert_equal(r, tgt)
        # 修改tgt的第二行为全2，并创建一个包含两个不同d数组的r数组
        tgt[1] = 2
        r = np.array([d, d + 1])
        assert_equal(r, tgt)

        # 创建一个长度为6的全1数组d，并将其放入一个形状为(1, 2, 6)的数组r中
        d = np.ones(6)
        r = np.array([[d, d]])
        assert_equal(r, np.ones((1, 2, 6)))

        # 创建一个包含两个d数组的二维数组r，并验证其形状是否为(2, 2, 6)
        d = np.ones(6)
        r = np.array([[d, d], [d, d]])
        assert_equal(r, np.ones((2, 2, 6)))

        # 创建一个形状为(6, 6)的全1数组d，并将其放入一个包含两个d数组的二维数组r中
        d = np.ones((6, 6))
        r = np.array([d, d])
        assert_equal(r, np.ones((2, 6, 6)))

        # 创建一个形状为(2, 3)的全1布尔数组tgt，并按指定索引修改其值
        tgt = np.ones((2, 3), dtype=bool)
        tgt[0, 2] = False
        tgt[1, 0:2] = False
        # 创建一个预期的布尔数组r，并验证其与tgt是否相等
        r = np.array([[True, True, False], [False, False, True]])
        assert_equal(r, tgt)
        # 创建r的转置数组，并验证其是否与tgt的转置相等
        r = np.array([[True, False], [True, False], [False, True]])
        assert_equal(r, tgt.T)

    @skip(reason="object arrays")
    def test_array_object(self):
        # 创建一个长度为6的全1数组d，并包含在一个包含两个数组的数组r中，类型为object
        d = np.ones((6,))
        r = np.array([[d, d + 1], d + 2], dtype=object)
        # 验证r的长度是否为2，并且其元素与预期相等
        assert_equal(len(r), 2)
        assert_equal(r[0], [d, d + 1])
        assert_equal(r[1], d + 2)

    def test_array_empty(self):
        # 测试空参数传递给np.array是否引发TypeError异常
        assert_raises(TypeError, np.array)

    def test_0d_array_shape(self):
        # 测试传递一个0维数组作为参数时，返回数组的形状是否为(3,)
        assert np.ones(np.array(3)).shape == (3,)

    def test_array_copy_false(self):
        # 创建一个数组d，并用copy=False选项创建e作为其副本
        d = np.array([1, 2, 3])
        e = np.array(d, copy=False)
        # 修改d的一个元素，并验证e是否反映出这一修改
        d[1] = 3
        assert_array_equal(e, [1, 3, 3])

    @xpassIfTorchDynamo  # (reason="order='F'")
    def test_array_copy_false_2(self):
        # 创建一个数组d，并用copy=False和指定的order选项'F'创建e作为其副本
        d = np.array([1, 2, 3])
        e = np.array(d, copy=False, order="F")
        # 修改d的一个元素，并验证e是否反映出这一修改
        d[1] = 4
        assert_array_equal(e, [1, 4, 3])
        # 修改e的一个元素，并验证d是否反映出这一修改
        e[2] = 7
        assert_array_equal(d, [1, 4, 7])

    def test_array_copy_true(self):
        # 创建一个二维数组d，并用copy=True选项创建e作为其副本
        d = np.array([[1, 2, 3], [1, 2, 3]])
        e = np.array(d, copy=True)
        # 修改d和e的元素，并验证它们是否有所不同
        d[0, 1] = 3
        e[0, 2] = -7
        assert_array_equal(e, [[1, 2, -7], [1, 2, 3]])
        assert_array_equal(d, [[1, 3, 3], [1, 2, 3]])

    @xfail  # (reason="order='F'")
    def test_array_copy_true_2(self):
        # 创建一个二维数组d，并用copy=True和指定的order选项'F'创建e作为其副本
        d = np.array([[1, 2, 3], [1, 2, 3]])
        e = np.array(d, copy=True, order="F")
        # 修改d和e的元素，并验证它们是否有所不同
        d[0, 1] = 5
        e[0, 2] = 7
        assert_array_equal(e, [[1, 3, 7], [1, 2, 3]])
        assert_array_equal(d, [[1, 5, 3], [1, 2, 3]])

    @xfailIfTorchDynamo
    def test_array_cont(self):
        # 创建一个间隔为2的数组d，并验证其连续性属性
        d = np.ones(10)[::2]
        assert_(np.ascontiguousarray(d).flags.c_contiguous)
        assert_(np.ascontiguousarray(d).flags.f_contiguous)
        assert_(np.asfortranarray(d).flags.c_contiguous)
        # assert_(np.asfortranarray(d).flags.f_contiguous)   # XXX: f ordering
        # 创建一个间隔为2的二维数组d，并验证其连续性属性
        d = np.ones((10, 10))[::2, ::2]
        assert_(np.ascontiguousarray(d).flags.c_contiguous)
        # assert_(np.asfortranarray(d).flags.f_contiguous)
    @parametrize(
        "func",
        [
            subtest(np.array, name="array"),  # 参数化测试，使用 np.array 函数，命名为 "array"
            subtest(np.asarray, name="asarray"),  # 参数化测试，使用 np.asarray 函数，命名为 "asarray"
            subtest(np.asanyarray, name="asanyarray"),  # 参数化测试，使用 np.asanyarray 函数，命名为 "asanyarray"
            subtest(np.ascontiguousarray, name="ascontiguousarray"),  # 参数化测试，使用 np.ascontiguousarray 函数，命名为 "ascontiguousarray"
            subtest(np.asfortranarray, name="asfortranarray"),  # 参数化测试，使用 np.asfortranarray 函数，命名为 "asfortranarray"
        ],
    )
    # 定义测试方法，验证对于不合法的参数调用，是否会抛出 TypeError 异常
    def test_bad_arguments_error(self, func):
        # 测试调用 func(3, dtype="bad dtype")，期待抛出 TypeError 异常
        with pytest.raises(TypeError):
            func(3, dtype="bad dtype")
        # 测试调用 func()，缺少参数，期待抛出 TypeError 异常
        with pytest.raises(TypeError):
            func()  # missing arguments
        # 测试调用 func(1, 2, 3, 4, 5, 6, 7, 8)，传入过多的参数，期待抛出 TypeError 异常
        with pytest.raises(TypeError):
            func(1, 2, 3, 4, 5, 6, 7, 8)  # too many arguments

    # 跳过测试的原因是 "np.array w/keyword argument"
    @skip(reason="np.array w/keyword argument")
    @parametrize(
        "func",
        [
            subtest(np.array, name="array"),  # 参数化测试，使用 np.array 函数，命名为 "array"
            subtest(np.asarray, name="asarray"),  # 参数化测试，使用 np.asarray 函数，命名为 "asarray"
            subtest(np.asanyarray, name="asanyarray"),  # 参数化测试，使用 np.asanyarray 函数，命名为 "asanyarray"
            subtest(np.ascontiguousarray, name="ascontiguousarray"),  # 参数化测试，使用 np.ascontiguousarray 函数，命名为 "ascontiguousarray"
            subtest(np.asfortranarray, name="asfortranarray"),  # 参数化测试，使用 np.asfortranarray 函数，命名为 "asfortranarray"
        ],
    )
    # 定义测试方法，验证特定情况下 func 的行为
    def test_array_as_keyword(self, func):
        # 这里应该将 func 参数改为只接受位置参数，但是不要意外改变其名称。
        if func is np.array:
            # 当 func 是 np.array 时，尝试使用关键字参数调用 func，期望不抛出异常
            func(object=3)
        else:
            # 否则，使用 a=3 调用 func
            func(a=3)
class TestAssignment(TestCase):
    def test_assignment_broadcasting(self):
        a = np.arange(6).reshape(2, 3)

        # Broadcasting the input to the output
        # 将输入广播到输出
        a[...] = np.arange(3)
        assert_equal(a, [[0, 1, 2], [0, 1, 2]])
        
        # Broadcasting a different shape to the output
        # 将不同形状的内容广播到输出
        a[...] = np.arange(2).reshape(2, 1)
        assert_equal(a, [[0, 0, 0], [1, 1, 1]])

        # For compatibility with <= 1.5, a limited version of broadcasting
        # the output to the input.
        #
        # This behavior is inconsistent with NumPy broadcasting
        # in general, because it only uses one of the two broadcasting
        # rules (adding a new "1" dimension to the left of the shape),
        # applied to the output instead of an input. In NumPy 2.0, this kind
        # of broadcasting assignment will likely be disallowed.
        # 为了与 <= 1.5 版本兼容，将输出广播到输入的有限版本。
        #
        # 这种行为与通常的 NumPy 广播不一致，因为它仅使用了两个广播规则中的一个
        # （在形状左侧添加一个新的 "1" 维度），应用于输出而不是输入。
        # 在 NumPy 2.0 中，这种广播赋值可能会被禁止。
        a[...] = np.flip(np.arange(6)).reshape(1, 2, 3)
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])
        # The other type of broadcasting would require a reduction operation.
        # 另一种广播方式需要进行减少操作。

        def assign(a, b):
            a[...] = b

        assert_raises(
            (RuntimeError, ValueError), assign, a, np.arange(12).reshape(2, 2, 3)
        )

    def test_assignment_errors(self):
        # Address issue #2276
        class C:
            pass

        a = np.zeros(1)

        def assign(v):
            a[0] = v

        assert_raises((RuntimeError, TypeError), assign, C())
        # assert_raises((TypeError, ValueError), assign, [1])  # numpy raises, we do not
        # numpy raises, 我们不会

    @skip(reason="object arrays")
    def test_unicode_assignment(self):
        # gh-5049
        from numpy.core.numeric import set_string_function

        @contextmanager
        def inject_str(s):
            """replace ndarray.__str__ temporarily"""
            set_string_function(lambda x: s, repr=False)
            try:
                yield
            finally:
                set_string_function(None, repr=False)

        a1d = np.array(["test"])
        a0d = np.array("done")
        with inject_str("bad"):
            a1d[0] = a0d  # previously this would invoke __str__
        assert_equal(a1d[0], "done")

        # this would crash for the same reason
        # 出于相同的原因，这会导致崩溃
        np.array([np.array("\xe5\xe4\xf6")])

    @skip(reason="object arrays")
    def test_stringlike_empty_list(self):
        # gh-8902
        u = np.array(["done"])
        b = np.array([b"done"])

        class bad_sequence:
            def __getitem__(self, value):
                pass

            def __len__(self):
                raise RuntimeError

        assert_raises(ValueError, operator.setitem, u, 0, [])
        assert_raises(ValueError, operator.setitem, b, 0, [])

        assert_raises(ValueError, operator.setitem, u, 0, bad_sequence())
        assert_raises(ValueError, operator.setitem, b, 0, bad_sequence())

    @skip(reason="longdouble")
    def test_longdouble_assignment(self):
        # 如果 longdouble 比 float 大，则以下内容才相关
        # 我们在寻找精度丢失的情况

        for dtype in (np.longdouble, np.longcomplex):
            # 解决问题 gh-8902

            # 找到比 np.longdouble(0) 大的最接近的浮点数，并转换成指定类型 dtype
            tinyb = np.nextafter(np.longdouble(0), 1).astype(dtype)
            # 找到比 np.longdouble(0) 小的最接近的浮点数，并转换成指定类型 dtype
            tinya = np.nextafter(np.longdouble(0), -1).astype(dtype)

            # 构建包含 tinya 的一维数组
            tiny1d = np.array([tinya])
            assert_equal(tiny1d[0], tinya)

            # 将 tinyb 赋值给 tiny1d 中的第一个元素
            tiny1d[0] = tinyb
            assert_equal(tiny1d[0], tinyb)

            # 将 tinya 赋值给 tiny1d 中的第一个元素（处理零维数组赋值）
            tiny1d[0, ...] = tinya
            assert_equal(tiny1d[0], tinya)

            # 将 tinyb 的零维版本赋值给 tiny1d 中的第一个元素（处理零维数组赋值）
            tiny1d[0, ...] = tinyb[...]
            assert_equal(tiny1d[0], tinyb)

            # 将 tinyb 的零维版本赋值给 tiny1d 中的第一个元素（处理零维数组赋值）
            tiny1d[0] = tinyb[...]
            assert_equal(tiny1d[0], tinyb)

            # 创建一个数组，其中包含 tinya 的数组，然后检查第一个元素
            arr = np.array([np.array(tinya)])
            assert_equal(arr[0], tinya)

    @skip(reason="object arrays")
    def test_cast_to_string(self):
        # 将类型转换为字符串应该执行 "str(scalar)"，而不是 "str(scalar.item())"
        # 示例：在 Python2 中，str(float) 会截断，因此我们要避免使用 str(np.float64(...).item())，因为这会错误地截断。
        
        # 创建一个长度为 1 的零数组，类型为字符串，最大长度为 20
        a = np.zeros(1, dtype="S20")
        # 将数组 a 中的元素赋值为一个浮点数数组的字符串表示
        a[:] = np.array(["1.12345678901234567890"], dtype="f8")
        # 断言数组中的第一个元素与指定的字节对象相等
        assert_equal(a[0], b"1.1234567890123457")
class TestDtypedescr(TestCase):
    # 定义测试类 TestDtypedescr，继承自 TestCase 类

    def test_construction(self):
        # 测试构造函数 test_construction

        d1 = np.dtype("i4")
        # 创建一个数据类型对象 d1，表示 32 位整数
        assert_equal(d1, np.dtype(np.int32))
        # 断言 d1 的类型与 np.int32 相同

        d2 = np.dtype("f8")
        # 创建一个数据类型对象 d2，表示 64 位浮点数
        assert_equal(d2, np.dtype(np.float64))
        # 断言 d2 的类型与 np.float64 相同


@skip  # (reason="TODO: zero-rank?")   # FIXME: revert skip into xfail
# 标记整个测试类 TestZeroRank 为跳过测试，并附带注释和修正建议
class TestZeroRank(TestCase):
    # 定义测试类 TestZeroRank，继承自 TestCase 类

    def setUp(self):
        # 设置测试前的准备工作

        self.d = np.array(0), np.array("x", object)
        # 创建包含两个数组的元组 self.d，分别包括一个整数和一个字符串对象数组

    def test_ellipsis_subscript(self):
        # 测试省略号索引功能的方法 test_ellipsis_subscript

        a, b = self.d
        # 将 self.d 中的两个数组分别赋值给 a 和 b

        assert_equal(a[...], 0)
        # 断言使用省略号索引 a 时得到值 0
        assert_equal(b[...], "x")
        # 断言使用省略号索引 b 时得到值 "x"

        assert_(a[...].base is a)
        # 断言 a[...] 的基础数组是 a，在 numpy 版本小于 1.9 时应为 `a[...] is a`。
        assert_(b[...].base is b)
        # 断言 b[...] 的基础数组是 b，在 numpy 版本小于 1.9 时应为 `b[...] is b`。

    def test_empty_subscript(self):
        # 测试空元组索引功能的方法 test_empty_subscript

        a, b = self.d
        # 将 self.d 中的两个数组分别赋值给 a 和 b

        assert_equal(a[()], 0)
        # 断言使用空元组索引 a 时得到值 0
        assert_equal(b[()], "x")
        # 断言使用空元组索引 b 时得到值 "x"

        assert_(type(a[()]) is a.dtype.type)
        # 断言 a[()] 的类型是 a 的数据类型
        assert_(type(b[()]) is str)
        # 断言 b[()] 的类型是字符串类型

    def test_invalid_subscript(self):
        # 测试无效索引功能的方法 test_invalid_subscript

        a, b = self.d
        # 将 self.d 中的两个数组分别赋值给 a 和 b

        assert_raises(IndexError, lambda x: x[0], a)
        # 断言对 a 使用索引 0 会抛出 IndexError 异常
        assert_raises(IndexError, lambda x: x[0], b)
        # 断言对 b 使用索引 0 会抛出 IndexError 异常
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)
        # 断言对 a 使用空的整数数组索引会抛出 IndexError 异常
        assert_raises(IndexError, lambda x: x[np.array([], int)], b)
        # 断言对 b 使用空的整数数组索引会抛出 IndexError 异常

    def test_ellipsis_subscript_assignment(self):
        # 测试省略号索引赋值功能的方法 test_ellipsis_subscript_assignment

        a, b = self.d
        # 将 self.d 中的两个数组分别赋值给 a 和 b

        a[...] = 42
        # 将 a 的所有元素赋值为 42
        assert_equal(a, 42)
        # 断言 a 的值等于 42

        b[...] = ""
        # 将 b 的所有元素赋值为空字符串
        assert_equal(b.item(), "")
        # 断言 b 的元素的单个值为空字符串

    def test_empty_subscript_assignment(self):
        # 测试空元组索引赋值功能的方法 test_empty_subscript_assignment

        a, b = self.d
        # 将 self.d 中的两个数组分别赋值给 a 和 b

        a[()] = 42
        # 将 a 的空元组索引位置赋值为 42
        assert_equal(a, 42)
        # 断言 a 的值等于 42

        b[()] = ""
        # 将 b 的空元组索引位置赋值为空字符串
        assert_equal(b.item(), "")
        # 断言 b 的元素的单个值为空字符串

    def test_invalid_subscript_assignment(self):
        # 测试无效索引赋值功能的方法 test_invalid_subscript_assignment

        a, b = self.d

        def assign(x, i, v):
            x[i] = v

        assert_raises(IndexError, assign, a, 0, 42)
        # 断言尝试对 a 使用索引 0 赋值为 42 会抛出 IndexError 异常
        assert_raises(IndexError, assign, b, 0, "")
        # 断言尝试对 b 使用索引 0 赋值为空字符串会抛出 IndexError 异常
        assert_raises(ValueError, assign, a, (), "")
        # 断言尝试对 a 使用空元组索引赋值为空字符串会抛出 ValueError 异常

    def test_newaxis(self):
        # 测试 newaxis 功能的方法 test_newaxis

        a, b = self.d
        # 将 self.d 中的两个数组分别赋值给 a 和 b

        assert_equal(a[np.newaxis].shape, (1,))
        # 断言 a 使用 np.newaxis 后的形状为 (1,)
        assert_equal(a[..., np.newaxis].shape, (1,))
        # 断言 a 使用省略号加 np.newaxis 后的形状为 (1,)
        assert_equal(a[np.newaxis, ...].shape, (1,))
        # 断言 a 使用 np.newaxis 加省略号后的形状为 (1,)
        assert_equal(a[..., np.newaxis].shape, (1,))
        # 断言 a 使用省略号加 np.newaxis 后的形状为 (1,)
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))
        # 断言 a 同时使用两个 np.newaxis 后的形状为 (1, 1)
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))
        # 断言 a 使用省略号加两个 np.newaxis 后的形状为 (1, 1)
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))
        # 断言 a 使用两个 np.newaxis 加省略号后的形状为 (1, 1)
        assert_equal(a[(np.newaxis,) * 10].shape, (1,) * 10)
        # 断言 a 使用十个 np.newaxis 后的形状为 (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    def test_invalid_newaxis(self):
        # 测试无效 newaxis 功能的方法 test_invalid_newaxis

        a, b = self.d

        def subscript(x, i):
            x[i]

        assert_raises(IndexError, subscript, a, (np.newaxis, 0))
        # 断言尝试在 a 中使用 (np.newaxis, 0) 索引会抛出 IndexError 异常
        assert_raises(IndexError, subscript, a, (np.newaxis,) * 50)
        # 断言尝试在 a 中使用 50 个 np.newaxis 索引会抛出 IndexError 异常

    def test_constructor(self):
        # 测试构造函数功能的方法 test_constructor

        x = np.ndarray(())
        # 创建一个 shape 为空的 ndarray 对象 x
        x[()] = 5
        # 将 x 的空元组索引位置赋值为 5
        assert_equal(x[()], 5)
        # 断言 x 的空元组索引位置的值为 5

        y = np.ndarray((), buffer=x)
        # 使用 x 的缓冲区创建
    # 定义一个测试方法，用于测试特定条件下函数的输出
    def test_output(self):
        # 创建一个包含单个整数元素的 NumPy 数组 x
        x = np.array(2)
        # 断言调用 np.add 函数时会引发 ValueError 异常，因为 x 不是可修改的对象
        assert_raises(ValueError, np.add, x, [1], x)

    # 定义另一个测试方法，测试复数数组的实部和虚部处理
    def test_real_imag(self):
        # 创建一个包含复数 1j 的 NumPy 数组 x
        # contiguity checks are for gh-11245
        x = np.array(1j)
        # 获取 x 的实部 xr 和虚部 xi
        xr = x.real
        xi = x.imag

        # 断言 xr 的值等于 0
        assert_equal(xr, np.array(0))
        # 断言 xr 的类型是 NumPy 数组
        assert_(type(xr) is np.ndarray)
        # 断言 xr 是连续存储的（C 连续）
        assert_equal(xr.flags.contiguous, True)
        # 断言 xr 是列优先连续的（Fortran 连续）
        assert_equal(xr.flags.f_contiguous, True)

        # 断言 xi 的值等于 1
        assert_equal(xi, np.array(1))
        # 断言 xi 的类型是 NumPy 数组
        assert_(type(xi) is np.ndarray)
        # 断言 xi 是连续存储的（C 连续）
        assert_equal(xi.flags.contiguous, True)
        # 断言 xi 是列优先连续的（Fortran 连续）
        assert_equal(xi.flags.f_contiguous, True)
class TestScalarIndexing(TestCase):
    # 定义测试类 TestScalarIndexing，继承自 TestCase 类
    def setUp(self):
        # 设置测试环境，在每个测试方法执行前运行
        self.d = np.array([0, 1])[0]

    def test_ellipsis_subscript(self):
        # 测试省略号(...)作为下标的情况
        a = self.d
        assert_equal(a[...], 0)
        assert_equal(a[...].shape, ())

    def test_empty_subscript(self):
        # 测试空元组()作为下标的情况
        a = self.d
        assert_equal(a[()], 0)
        assert_equal(a[()].shape, ())

    def test_invalid_subscript(self):
        # 测试非法下标的情况
        a = self.d
        assert_raises(IndexError, lambda x: x[0], a)
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)

    def test_invalid_subscript_assignment(self):
        # 测试对非法下标赋值的情况
        a = self.d

        def assign(x, i, v):
            x[i] = v

        assert_raises((IndexError, TypeError), assign, a, 0, 42)

    def test_newaxis(self):
        # 测试 np.newaxis 的使用
        a = self.d
        assert_equal(a[np.newaxis].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ...].shape, (1,))
        assert_equal(a[..., np.newaxis].shape, (1,))
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))
        assert_equal(a[(np.newaxis,) * 10].shape, (1,) * 10)

    def test_invalid_newaxis(self):
        # 测试非法 np.newaxis 的使用
        a = self.d

        def subscript(x, i):
            x[i]

        assert_raises(IndexError, subscript, a, (np.newaxis, 0))

        # 下面的断言因为 50 > NPY_MAXDIMS = 32 而会失败
        # assert_raises(IndexError, subscript, a, (np.newaxis,)*50)

    @xfail  # (reason="pytorch disallows overlapping assignments")
    def test_overlapping_assignment(self):
        # 测试重叠赋值的情况
        # 正步长情况
        a = np.arange(4)
        a[:-1] = a[1:]
        assert_equal(a, [1, 2, 3, 3])

        a = np.arange(4)
        a[1:] = a[:-1]
        assert_equal(a, [0, 0, 1, 2])

        # 正负步长混合情况
        a = np.arange(4)
        a[:] = a[::-1]
        assert_equal(a, [3, 2, 1, 0])

        a = np.arange(6).reshape(2, 3)
        a[::-1, :] = a[:, ::-1]
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])

        a = np.arange(6).reshape(2, 3)
        a[::-1, ::-1] = a[:, ::-1]
        assert_equal(a, [[3, 4, 5], [0, 1, 2]])

        # 仅有一个元素重叠的情况
        a = np.arange(5)
        a[:3] = a[2:]
        assert_equal(a, [2, 3, 4, 3, 4])

        a = np.arange(5)
        a[2:] = a[:3]
        assert_equal(a, [0, 1, 0, 1, 2])

        a = np.arange(5)
        a[2::-1] = a[2:]
        assert_equal(a, [4, 3, 2, 3, 4])

        a = np.arange(5)
        a[2:] = a[2::-1]
        assert_equal(a, [0, 1, 2, 1, 0])

        a = np.arange(5)
        a[2::-1] = a[:1:-1]
        assert_equal(a, [2, 3, 4, 3, 4])

        a = np.arange(5)
        a[:1:-1] = a[2::-1]
        assert_equal(a, [0, 1, 0, 1, 2])


@skip(reason="object, void, structured dtypes")
@instantiate_parametrized_tests
class TestCreation(TestCase):
    """
    Test the np.array constructor
    """
    # 定义一个测试方法，用于测试从对象属性创建数组时的行为
    def test_from_attribute(self):
        # 定义一个简单的类 x，没有实现 __array__ 方法
        class x:
            def __array__(self, dtype=None):
                pass
        
        # 断言调用 np.array(x()) 会引发 ValueError 异常
        assert_raises(ValueError, np.array, x())

    # 定义一个测试方法，用于测试从字符串数组创建数组的行为
    def test_from_string(self):
        # 从 NumPy 中获取包含所有整数和浮点数类型码的列表
        types = np.typecodes["AllInteger"] + np.typecodes["Float"]
        # 定义一个包含两个相同字符串的列表
        nstr = ["123", "123"]
        # 创建一个期望的整数类型数组作为结果
        result = np.array([123, 123], dtype=int)
        # 遍历所有类型码，测试字符串转换为数组的结果
        for type in types:
            msg = f"String conversion for {type}"
            # 断言将 nstr 转换为指定类型的数组与预期结果相等
            assert_equal(np.array(nstr, dtype=type), result, err_msg=msg)

    # 定义一个测试方法，用于测试创建 void 类型数组的行为
    def test_void(self):
        # 创建一个空的 void 类型数组 arr
        arr = np.array([], dtype="V")
        # 断言 arr 的数据类型是 "V8"，即当前的默认 void 类型
        assert arr.dtype == "V8"
        
        # 创建一个包含两个相同长度的字节字符串的 void 类型数组 arr
        arr = np.array([b"1234", b"1234"], dtype="V")
        # 断言 arr 的数据类型是 "V4"
        assert arr.dtype == "V4"

        # 尝试创建包含不同长度的字节字符串的 void 类型数组，预期会引发 TypeError 异常
        with pytest.raises(TypeError):
            np.array([b"1234", b"12345"], dtype="V")
        with pytest.raises(TypeError):
            np.array([b"12345", b"1234"], dtype="V")

        # 尝试先创建对象类型数组，然后将其转换为 void 类型数组的行为测试
        arr = np.array([b"1234", b"1234"], dtype="O").astype("V")
        # 断言 arr 的数据类型是 "V4"
        assert arr.dtype == "V4"
        with pytest.raises(TypeError):
            np.array([b"1234", b"12345"], dtype="O").astype("V")

    # 使用 parametrize 装饰器定义多组参数化测试用例
    @parametrize(
        # 参数化测试用例，用 Ellipsis 和 () 作为索引
        "idx", [subtest(Ellipsis, name="arr"), subtest((), name="scalar")],
    )
    # 定义一个测试方法，用于测试结构化数组和 void 类型数组的提升行为
    def test_structured_void_promotion(self, idx):
        # 创建一个包含两个结构化数组元素的 void 类型数组 arr
        arr = np.array(
            [np.array(1, dtype="i,i")[idx], np.array(2, dtype="i,i")[idx]], dtype="V"
        )
        # 断言 arr 与预期的结构化数组相等
        assert_array_equal(arr, np.array([(1, 1), (2, 2)], dtype="i,i"))
        
        # 尝试创建包含不同结构化数组的 void 类型数组，预期会引发 TypeError 异常
        with pytest.raises(TypeError):
            np.array(
                [np.array(1, dtype="i,i")[idx], np.array(2, dtype="i,i,i")[idx]],
                dtype="V",
            )

    # 定义一个测试方法，用于测试创建过大数组时的错误处理行为
    def test_too_big_error(self):
        # 根据系统位数选择不同的数组维度来测试
        if np.iinfo("intp").max == 2**31 - 1:
            shape = (46341, 46341)  # 32 位系统
        elif np.iinfo("intp").max == 2**63 - 1:
            shape = (3037000500, 3037000500)  # 64 位系统
        else:
            return
        # 断言创建指定维度的 int8 类型数组会引发 ValueError 异常
        assert_raises(ValueError, np.empty, shape, dtype=np.int8)
        assert_raises(ValueError, np.zeros, shape, dtype=np.int8)
        assert_raises(ValueError, np.ones, shape, dtype=np.int8)

    # 使用 skipif 装饰器跳过特定条件下的测试
    @skipif(
        np.dtype(np.intp).itemsize != 8, reason="malloc may not fail on 32 bit systems"
    )
    # 定义一个测试函数，用于测试当内存分配失败时的情况
    def test_malloc_fails(self):
        # 使用 assert_raises 断言来验证是否会抛出 ArrayMemoryError 异常
        with assert_raises(np.core._exceptions._ArrayMemoryError):
            # 尝试分配一个过大的空数组，预期会失败
            np.empty(np.iinfo(np.intp).max, dtype=np.uint8)

    # 定义一个测试函数，用于测试 np.zeros 的不同用法
    def test_zeros(self):
        # 获取所有整数和浮点数类型码的组合
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        # 遍历每种数据类型
        for dt in types:
            # 创建一个形状为 (13,) 的零数组，指定数据类型为 dt
            d = np.zeros((13,), dtype=dt)
            # 断言数组中的非零元素数量为 0
            assert_equal(np.count_nonzero(d), 0)
            # 对于 IEEE 浮点数，断言数组所有元素之和为 0
            assert_equal(d.sum(), 0)
            # 断言数组中没有任何元素为 True
            assert_(not d.any())

            # 创建一个形状为 (2,) 的复合类型数组，元素类型为 "(2,4)i4"
            d = np.zeros(2, dtype="(2,4)i4")
            # 断言数组中的非零元素数量为 0
            assert_equal(np.count_nonzero(d), 0)
            # 断言数组所有元素之和为 0
            assert_equal(d.sum(), 0)
            # 断言数组中没有任何元素为 True
            assert_(not d.any())

            # 创建一个形状为 (2,) 的复合类型数组，元素类型为 "4i4"
            d = np.zeros(2, dtype="4i4")
            # 断言数组中的非零元素数量为 0
            assert_equal(np.count_nonzero(d), 0)
            # 断言数组所有元素之和为 0
            assert_equal(d.sum(), 0)
            # 断言数组中没有任何元素为 True
            assert_(not d.any())

            # 创建一个形状为 (2,) 的复合类型数组，元素类型为 "(2,4)i4, (2,4)i4"
            d = np.zeros(2, dtype="(2,4)i4, (2,4)i4")
            # 断言数组中的非零元素数量为 0
            assert_equal(np.count_nonzero(d), 0)

    # 定义一个带有 @slow 装饰器的测试函数，用于测试大数组的情况
    @slow
    def test_zeros_big(self):
        # test big array as they might be allocated different by the system
        # 获取所有整数和浮点数类型码的组合
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        # 遍历每种数据类型
        for dt in types:
            # 创建一个形状为 (30 * 1024**2,) 的零数组，数据类型为 dt
            d = np.zeros((30 * 1024**2,), dtype=dt)
            # 断言数组中没有任何元素为 True
            assert_(not d.any())
            # 在 32 位系统上，由于内存不足可能导致此测试失败。释放前一个数组可以增加成功的机会。
            del d

    # 定义一个测试函数，用于测试对象数组的初始化
    def test_zeros_obj(self):
        # 创建一个形状为 (13,) 的对象类型零数组
        d = np.zeros((13,), dtype=object)
        # 断言数组内容等于 [0] * 13
        assert_array_equal(d, [0] * 13)
        # 断言数组中的非零元素数量为 0
        assert_equal(np.count_nonzero(d), 0)

    # 定义一个测试函数，用于测试复合对象类型数组的初始化
    def test_zeros_obj_obj(self):
        # 创建一个形状为 (10,) 的复合对象类型数组，字段 "k" 类型为 object，长度为 2
        d = np.zeros(10, dtype=[("k", object, 2)])
        # 断言数组字段 "k" 的内容等于 0
        assert_array_equal(d["k"], 0)
    def test_zeros_like_like_zeros(self):
        # 测试 zeros_like 返回与 zeros 相同的结果
        for c in np.typecodes["All"]:
            # 遍历 NumPy 支持的所有类型代码
            if c == "V":
                continue
            # 创建一个 dtype 为 c 的 3x3 的零数组 d
            d = np.zeros((3, 3), dtype=c)
            # 断言 np.zeros_like(d) 返回与 d 相同的数组
            assert_array_equal(np.zeros_like(d), d)
            # 断言 np.zeros_like(d) 的 dtype 与 d 的 dtype 相同
            assert_equal(np.zeros_like(d).dtype, d.dtype)
        # 显式检查一些特殊情况
        # 创建一个 dtype 为 "S5" 的 3x3 的零数组 d
        d = np.zeros((3, 3), dtype="S5")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        # 创建一个 dtype 为 "U5" 的 3x3 的零数组 d
        d = np.zeros((3, 3), dtype="U5")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        # 创建一个 dtype 为 "<i4" 的 3x3 的零数组 d
        d = np.zeros((3, 3), dtype="<i4")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        # 创建一个 dtype 为 ">i4" 的 3x3 的零数组 d
        d = np.zeros((3, 3), dtype=">i4")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        # 创建一个 dtype 为 "<M8[s]" 的 3x3 的零数组 d
        d = np.zeros((3, 3), dtype="<M8[s]")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        # 创建一个 dtype 为 ">M8[s]" 的 3x3 的零数组 d
        d = np.zeros((3, 3), dtype=">M8[s]")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

        # 创建一个 dtype 为 "f4,f4" 的 3x3 的零数组 d
        d = np.zeros((3, 3), dtype="f4,f4")
        assert_array_equal(np.zeros_like(d), d)
        assert_equal(np.zeros_like(d).dtype, d.dtype)

    def test_empty_unicode(self):
        # 在垃圾内存上不会引发解码错误
        for i in range(5, 100, 5):
            # 创建一个长度为 i 的 Unicode 空数组 d
            d = np.empty(i, dtype="U")
            str(d)

    def test_sequence_non_homogeneous(self):
        # 断言非同质数组的 dtype 是 object
        assert_equal(np.array([4, 2**80]).dtype, object)
        assert_equal(np.array([4, 2**80, 4]).dtype, object)
        assert_equal(np.array([2**80, 4]).dtype, object)
        assert_equal(np.array([2**80] * 3).dtype, object)
        # 断言复数数组的 dtype 是 complex
        assert_equal(np.array([[1, 1], [1j, 1j]]).dtype, complex)
        assert_equal(np.array([[1j, 1j], [1, 1]]).dtype, complex)
        assert_equal(np.array([[1, 1, 1], [1, 1j, 1.0], [1, 1, 1]]).dtype, complex)

    def test_non_sequence_sequence(self):
        """不应该导致段错误。

        类 Fail 打破了新式类（即从 object 派生的类）的序列协议，即 __getitem__ 抛出 ValueError。
        类 Map 是一个映射类型，抛出 KeyError 表示。
        在某些情况下，Fail 的情况可能会引发警告而不是错误。

        """

        class Fail:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise ValueError

        class Map:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise KeyError

        # 创建包含 Map 对象的数组 a
        a = np.array([Map()])
        assert_(a.shape == (1,))
        assert_(a.dtype == np.dtype(object))
        # 断言 np.array([Fail()]) 会引发 ValueError
        assert_raises(ValueError, np.array, [Fail()])
    def test_no_len_object_type(self):
        # 定义一个名为 Point2 的类，用于模拟一个没有长度的可迭代对象
        class Point2:
            def __init__(self):
                pass

            # 定义 __getitem__ 方法，返回索引值，如果索引不是 0 或 1 则抛出 IndexError
            def __getitem__(self, ind):
                if ind in [0, 1]:
                    return ind
                else:
                    raise IndexError

        # 创建一个包含三个 Point2 实例的 numpy 数组
        d = np.array([Point2(), Point2(), Point2()])
        # 断言数组的数据类型是 object 类型
        assert_equal(d.dtype, np.dtype(object))

    def test_false_len_sequence(self):
        # 测试案例 gh-7264，示例中可能导致段错误
        class C:
            # 定义 __getitem__ 方法，抛出 IndexError
            def __getitem__(self, i):
                raise IndexError

            # 定义 __len__ 方法，返回固定值 42
            def __len__(self):
                return 42

        # 创建一个包含 C 实例的 numpy 数组
        a = np.array(C())  # 可能导致段错误？
        # 断言数组的长度为 0
        assert_equal(len(a), 0)

    def test_false_len_iterable(self):
        # 特殊情况，当 __getitem__ 方法问题时，会退而使用 __iter__ 方法
        class C:
            # 定义 __getitem__ 方法，抛出异常
            def __getitem__(self, x):
                raise Exception  # noqa: TRY002

            # 定义 __iter__ 方法，返回一个空迭代器
            def __iter__(self):
                return iter(())

            # 定义 __len__ 方法，返回固定值 2
            def __len__(self):
                return 2

        # 创建一个空的 numpy 数组
        a = np.empty(2)
        # 使用 assert_raises 检查赋值过程是否导致 ValueError 异常
        with assert_raises(ValueError):
            a[:] = C()  # 可能导致段错误！

        # 断言 np.array(C()) 和 list(C()) 结果相等
        assert_equal(np.array(C()), list(C()))

    def test_failed_len_sequence(self):
        # 测试案例 gh-7393
        class A:
            def __init__(self, data):
                self._data = data

            # 定义 __getitem__ 方法，返回一个新的 A 类型实例
            def __getitem__(self, item):
                return type(self)(self._data[item])

            # 定义 __len__ 方法，返回数据 _data 的长度
            def __len__(self):
                return len(self._data)

        # 创建一个 A 类型的实例 d，其中 _data 是一个包含 [1, 2, 3] 的列表
        d = A([1, 2, 3])
        # 断言 np.array(d) 的长度为 3
        assert_equal(len(np.array(d)), 3)

    def test_array_too_big(self):
        # 测试数组的创建是否能够成功，对于通过 intp 进行地址访问的数组可以成功，对于过大的数组则失败
        buf = np.zeros(100)

        max_bytes = np.iinfo(np.intp).max
        for dtype in ["intp", "S20", "b"]:
            dtype = np.dtype(dtype)
            itemsize = dtype.itemsize

            # 创建一个 numpy 数组，通过 buf 缓冲区、strides、shape 和 dtype 参数确定
            np.ndarray(
                buffer=buf, strides=(0,), shape=(max_bytes // itemsize,), dtype=dtype
            )
            # 使用 assert_raises 检查创建超过大小限制的数组是否会导致 ValueError 异常
            assert_raises(
                ValueError,
                np.ndarray,
                buffer=buf,
                strides=(0,),
                shape=(max_bytes // itemsize + 1,),
                dtype=dtype,
            )

    def _ragged_creation(self, seq):
        # 如果没有指定 dtype=object，则会引发 ValueError，用 pytest 检查并匹配错误信息
        with pytest.raises(ValueError, match=".*detected shape was"):
            # 使用 seq 创建一个 numpy 数组
            a = np.array(seq)

        # 返回一个 dtype=object 的 numpy 数组
        return np.array(seq, dtype=object)
    def test_ragged_ndim_object(self):
        # Lists of mismatching depths are treated as object arrays
        
        # 调用 _ragged_creation 方法，传入包含不同深度子列表的列表作为参数，返回数组 a
        a = self._ragged_creation([[1], 2, 3])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        # 同上，传入不同深度子列表的另一种情况
        a = self._ragged_creation([1, [2], 3])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        # 同上，传入不同深度子列表的另一种情况
        a = self._ragged_creation([1, 2, [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

    def test_ragged_shape_object(self):
        # The ragged dimension of a list is turned into an object array
        
        # 调用 _ragged_creation 方法，传入包含不同长度子列表的列表作为参数，返回数组 a
        a = self._ragged_creation([[1, 1], [2], [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        # 同上，传入包含不同长度子列表的另一种情况
        a = self._ragged_creation([[1], [2, 2], [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        # 同上，传入包含不同长度子列表的另一种情况
        a = self._ragged_creation([[1], [2], [3, 3]])
        assert a.shape == (3,)
        assert a.dtype == object

    def test_array_of_ragged_array(self):
        outer = np.array([None, None])
        outer[0] = outer[1] = np.array([1, 2, 3])
        assert np.array(outer).shape == (2,)
        assert np.array([outer]).shape == (1, 2)

        outer_ragged = np.array([None, None])
        outer_ragged[0] = np.array([1, 2, 3])
        outer_ragged[1] = np.array([1, 2, 3, 4])
        
        # 对外部数组进行操作，确保形状为 (2,)
        assert np.array(outer_ragged).shape == (2,)
        # 对包含外部数组的数组进行操作，确保形状为 (1, 2)
        assert np.array([outer_ragged]).shape == (1, 2)

    def test_deep_nonragged_object(self):
        # None of these should raise, even though they are missing dtype=object
        
        # 创建包含 Decimal 类型的数组，即使未指定 dtype=object 也不应该引发异常
        a = np.array([[[Decimal(1)]]])
        a = np.array([1, Decimal(1)])
        a = np.array([[1], [Decimal(1)]])

    @parametrize("dtype", [object, "O,O", "O,(3)O", "(2,3)O"])
    @parametrize(
        "function",
        [
            np.ndarray,
            np.empty,
            lambda shape, dtype: np.empty_like(np.empty(shape, dtype=dtype)),
        ],
    )
    def test_object_initialized_to_None(self, function, dtype):
        # NumPy has support for object fields to be NULL (meaning None)
        # but generally, we should always fill with the proper None, and
        # downstream may rely on that.  (For fully initialized arrays!)
        
        # 使用参数化测试，创建指定 dtype 的数组，应初始化为 None
        arr = function(3, dtype=dtype)
        # 我们期望填充值为 None，这不是 NULL：
        expected = np.array(None).tobytes()
        expected = expected * (arr.nbytes // len(expected))
        assert arr.tobytes() == expected
class TestBool(TestCase):
    @xfail  # 标记该测试用例为预期失败，原因是布尔类型不会被内部化
    def test_test_interning(self):
        # 创建 np.bool_ 类型的对象，表示 False
        a0 = np.bool_(0)
        # 创建 np.bool_ 类型的对象，表示 False
        b0 = np.bool_(False)
        # 断言 a0 和 b0 是同一个对象
        assert_(a0 is b0)
        # 创建 np.bool_ 类型的对象，表示 True
        a1 = np.bool_(1)
        # 创建 np.bool_ 类型的对象，表示 True
        b1 = np.bool_(True)
        # 断言 a1 和 b1 是同一个对象
        assert_(a1 is b1)
        # 断言 np.array([True])[0] 是 a1
        assert_(np.array([True])[0] is a1)
        # 断言 np.array(True)[()] 是 a1
        assert_(np.array(True)[()] is a1)

    def test_sum(self):
        # 创建一个长度为 101 的全为 True 的布尔数组
        d = np.ones(101, dtype=bool)
        # 断言数组 d 的和等于数组大小
        assert_equal(d.sum(), d.size)
        # 断言数组 d 中偶数索引位置的元素和等于偶数索引的数量
        assert_equal(d[::2].sum(), d[::2].size)
        # 注释掉的代码行，用于测试负步长的情况

    @xpassIfTorchDynamo  # 标记该测试用例为预期通过，原因是使用了 frombuffer
    def test_sum_2(self):
        # 从字节序列创建布尔数组 d
        d = np.frombuffer(b"\xff\xff" * 100, dtype=bool)
        # 断言数组 d 的和等于数组大小
        assert_equal(d.sum(), d.size)
        # 断言数组 d 中偶数索引位置的元素和等于偶数索引的数量
        assert_equal(d[::2].sum(), d[::2].size)
        # 断言数组 d 中逆序偶数索引位置的元素和等于逆序偶数索引的数量
        assert_equal(d[::-2].sum(), d[::-2].size)

    def check_count_nonzero(self, power, length):
        # 生成长度为 length 的幂集列表
        powers = [2**i for i in range(length)]
        # 遍历 0 到 2^power-1 的所有数值
        for i in range(2**power):
            # 创建长度为 length 的布尔数组 l
            l = [(i & x) != 0 for x in powers]
            # 创建 np.array 对象 a，从列表 l 中创建布尔数组
            a = np.array(l, dtype=bool)
            # 计算数组 a 中非零元素的数量，断言结果与内置 sum 函数的结果相等
            c = builtins.sum(l)
            assert_equal(np.count_nonzero(a), c)
            # 将数组 a 视为 np.uint8 类型的视图 av
            av = a.view(np.uint8)
            av *= 3
            # 再次断言数组 a 中非零元素的数量，结果应与之前的 c 相等
            assert_equal(np.count_nonzero(a), c)
            av *= 4
            # 再次断言数组 a 中非零元素的数量，结果应与之前的 c 相等
            assert_equal(np.count_nonzero(a), c)
            # 将 av 中非零元素设为 0xFF
            av[av != 0] = 0xFF
            # 最后一次断言数组 a 中非零元素的数量，结果应与之前的 c 相等
            assert_equal(np.count_nonzero(a), c)

    def test_count_nonzero(self):
        # 检查在长度为 17 的数组中的所有 12 位组合
        # 覆盖大多数 16 字节展开的情况
        self.check_count_nonzero(12, 17)

    @slow
    def test_count_nonzero_all(self):
        # 检查在长度为 17 的数组中的所有组合
        # 覆盖所有 16 字节展开的情况
        self.check_count_nonzero(17, 17)

    def test_count_nonzero_unaligned(self):
        # 防止像 gh-4060 这样的错误
        # 遍历 0 到 6 的所有值
        for o in range(7):
            # 创建一个长度为 18 的全为 False 的布尔数组 a
            a = np.zeros((18,), dtype=bool)[o + 1 :]
            # 将前 o 个元素设置为 True
            a[:o] = True
            # 断言数组 a 中非零元素的数量，结果应与内置 sum 函数的结果相等
            assert_equal(np.count_nonzero(a), builtins.sum(a.tolist()))
            # 创建一个长度为 18 的全为 True 的布尔数组 a
            a = np.ones((18,), dtype=bool)[o + 1 :]
            # 将前 o 个元素设置为 False
            a[:o] = False
            # 断言数组 a 中非零元素的数量，结果应与内置 sum 函数的结果相等
            assert_equal(np.count_nonzero(a), builtins.sum(a.tolist()))

    def _test_cast_from_flexible(self, dtype):
        # 空字符串转换为 False
        for n in range(3):
            # 从空字节串创建长度为 n 的数组 v，类型为 dtype
            v = np.array(b"", (dtype, n))
            assert_equal(bool(v), False)
            assert_equal(bool(v[()]), False)
            # 将数组 v 转换为布尔类型，断言结果为 False
            assert_equal(v.astype(bool), False)
            # 断言 v.astype(bool) 是一个 np.ndarray 对象
            assert_(isinstance(v.astype(bool), np.ndarray))
            # 断言 v[()].astype(bool) 是 np.False_
            assert_(v[()].astype(bool) is np.False_)

        # 任何其他值转换为 True
        for n in range(1, 4):
            for val in [b"a", b"0", b" "]:
                # 从字节串 val 创建长度为 n 的数组 v，类型为 dtype
                v = np.array(val, (dtype, n))
                assert_equal(bool(v), True)
                assert_equal(bool(v[()]), True)
                # 将数组 v 转换为布尔类型，断言结果为 True
                assert_equal(v.astype(bool), True)
                # 断言 v.astype(bool) 是一个 np.ndarray 对象
                assert_(isinstance(v.astype(bool), np.ndarray))
                # 断言 v[()].astype(bool) 是 np.True_
                assert_(v[()].astype(bool) is np.True_)
    # 使用装饰器 @skip，标记为跳过测试，原因是 "np.void"
    @skip(reason="np.void")
    def test_cast_from_void(self):
        # 调用内部方法 _test_cast_from_flexible，测试从 np.void 类型进行类型转换
        self._test_cast_from_flexible(np.void)
    
    # 使用装饰器 @xfail，标记为预期失败的测试，原因是 "See gh-9847"
    @xfail  # (reason="See gh-9847")
    def test_cast_from_unicode(self):
        # 调用内部方法 _test_cast_from_flexible，测试从 np.unicode_ 类型进行类型转换
        self._test_cast_from_flexible(np.unicode_)
    
    # 使用装饰器 @xfail，标记为预期失败的测试，原因是 "See gh-9847"
    @xfail  # (reason="See gh-9847")
    def test_cast_from_bytes(self):
        # 调用内部方法 _test_cast_from_flexible，测试从 np.bytes_ 类型进行类型转换
        self._test_cast_from_flexible(np.bytes_)
# 在测试方法上应用装饰器，实例化参数化测试
@instantiate_parametrized_tests
class TestMethods(TestCase):
    # 定义排序种类列表
    sort_kinds = ["quicksort", "heapsort", "stable"]

    # 定义一个带有 xpassIfTorchDynamo 装饰器的测试方法，用于测试 all 函数与 where 参数的组合使用
    @xpassIfTorchDynamo  # (reason="all(..., where=...)")
    def test_all_where(self):
        # 创建一个布尔数组 a
        a = np.array([[True, False, True], [False, False, False], [True, True, True]])
        # 创建 where 参数的全布尔值矩阵
        wh_full = np.array([[True, False, True], [False, False, False], [True, False, True]])
        # 创建 where 参数的部分布尔值矩阵
        wh_lower = np.array([[False], [False], [True]])

        # 循环遍历 _ax 的取值 [0, None]
        for _ax in [0, None]:
            # 使用 all 函数与 where 参数来检查数组 a 的满足条件，与通过切片操作得到的结果进行比较
            assert_equal(
                a.all(axis=_ax, where=wh_lower), np.all(a[wh_lower[:, 0], :], axis=_ax)
            )
            # 使用 np.all 函数与 where 参数检查数组 a 的所有元素，与通过切片操作得到的结果进行比较
            assert_equal(
                np.all(a, axis=_ax, where=wh_lower), a[wh_lower[:, 0], :].all(axis=_ax)
            )

        # 使用 all 函数与 where 参数来检查数组 a 是否全为 True，与预期结果进行比较
        assert_equal(a.all(where=wh_full), True)
        # 使用 np.all 函数与 where 参数检查数组 a 的所有元素是否全为 True，与预期结果进行比较
        assert_equal(np.all(a, where=wh_full), True)
        # 使用 all 函数与 where 参数来检查数组 a 是否全为 True（传入 False 作为 where 参数），与预期结果进行比较
        assert_equal(a.all(where=False), True)
        # 使用 np.all 函数与 where 参数检查数组 a 的所有元素是否全为 True（传入 False 作为 where 参数），与预期结果进行比较
        assert_equal(np.all(a, where=False), True)

    # 定义一个带有 xpassIfTorchDynamo 装饰器的测试方法，用于测试 any 函数与 where 参数的组合使用
    @xpassIfTorchDynamo  # (reason="any(..., where=...)")
    def test_any_where(self):
        # 创建一个布尔数组 a
        a = np.array([[True, False, True], [False, False, False], [True, True, True]])
        # 创建 where 参数的全布尔值矩阵
        wh_full = np.array([[False, True, False], [True, True, True], [False, False, False]])
        # 创建 where 参数的部分布尔值矩阵
        wh_middle = np.array([[False], [True], [False]])

        # 循环遍历 _ax 的取值 [0, None]
        for _ax in [0, None]:
            # 使用 any 函数与 where 参数来检查数组 a 的满足条件，与通过切片操作得到的结果进行比较
            assert_equal(
                a.any(axis=_ax, where=wh_middle),
                np.any(a[wh_middle[:, 0], :], axis=_ax),
            )
            # 使用 np.any 函数与 where 参数检查数组 a 的任意元素，与通过切片操作得到的结果进行比较
            assert_equal(
                np.any(a, axis=_ax, where=wh_middle),
                a[wh_middle[:, 0], :].any(axis=_ax),
            )

        # 使用 any 函数与 where 参数来检查数组 a 是否有任意 True 值，与预期结果进行比较
        assert_equal(a.any(where=wh_full), False)
        # 使用 np.any 函数与 where 参数检查数组 a 的任意元素是否有 True 值，与预期结果进行比较
        assert_equal(np.any(a, where=wh_full), False)
        # 使用 any 函数与 where 参数来检查数组 a 是否有任意 True 值（传入 False 作为 where 参数），与预期结果进行比较
        assert_equal(a.any(where=False), False)
        # 使用 np.any 函数与 where 参数检查数组 a 的任意元素是否有 True 值（传入 False 作为 where 参数），与预期结果进行比较
        assert_equal(np.any(a, where=False), False)

    # 定义一个带有 xpassIfTorchDynamo 装饰器的测试方法，用于测试 compress 函数
    @xpassIfTorchDynamo  # (reason="TODO: compress")
    def test_compress(self):
        # 定义目标数组 tgt 和初始数组 arr
        tgt = [[5, 6, 7, 8, 9]]
        arr = np.arange(10).reshape(2, 5)
        # 使用 compress 函数对数组 arr 进行压缩操作，沿着 axis=0 的方向，与预期结果进行比较
        out = arr.compress([0, 1], axis=0)
        assert_equal(out, tgt)

        # 更新目标数组 tgt 和初始数组 arr
        tgt = [[1, 3], [6, 8]]
        # 使用 compress 函数对数组 arr 进行压缩操作，沿着 axis=1 的方向，与预期结果进行比较
        out = arr.compress([0, 1, 0, 1, 0], axis=1)
        assert_equal(out, tgt)

        # 更新目标数组 tgt 和初始数组 arr
        tgt = [[1], [6]]
        # 使用 compress 函数对数组 arr 进行压缩操作，未指定 axis 参数，默认为 axis=0，与预期结果进行比较
        out = arr.compress([0, 1], axis=1)
        assert_equal(out, tgt)

        # 更新初始数组 arr
        arr = np.arange(10).reshape(2, 5)
        # 使用 compress 函数对数组 arr 进行压缩操作，未指定 axis 参数，默认为 axis=None，与预期结果进行比较
        out = arr.compress([0, 1])
        assert_equal(out, 1)
    # 定义测试函数test_choose，用于测试numpy中的choose函数
    def test_choose(self):
        # 创建包含元素全为2的长度为3的整数数组x
        x = 2 * np.ones((3,), dtype=int)
        # 创建包含元素全为3的长度为3的整数数组y
        y = 3 * np.ones((3,), dtype=int)
        # 创建包含元素全为2的2行3列的整数数组x2
        x2 = 2 * np.ones((2, 3), dtype=int)
        # 创建包含元素全为3的2行3列的整数数组y2
        y2 = 3 * np.ones((2, 3), dtype=int)
        # 创建包含元素为[0, 0, 1]的整数数组ind
        ind = np.array([0, 0, 1])

        # 使用ind数组从(x, y)中选择元素，组成数组A，预期结果为[2, 2, 3]
        A = ind.choose((x, y))
        assert_equal(A, [2, 2, 3])

        # 使用ind数组从(x2, y2)中选择元素，组成数组A，预期结果为[[2, 2, 3], [2, 2, 3]]
        A = ind.choose((x2, y2))
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

        # 使用ind数组从(x, y2)中选择元素，组成数组A，预期结果为[[2, 2, 3], [2, 2, 3]]
        A = ind.choose((x, y2))
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

        # 创建初始值为0的numpy数组out
        out = np.array(0)
        # 使用np.choose函数，从[10, 20, 30]中选择第1个元素，放入out中
        ret = np.choose(np.array(1), [10, 20, 30], out=out)
        # 断言out与ret为同一对象，断言out的值为20
        assert out is ret
        assert_equal(out[()], 20)

    @xpassIfTorchDynamo  # (reason="choose(..., mode=...) not implemented")
    # 定义测试函数test_choose_2，用于测试numpy中choose函数的另一种情况
    def test_choose_2(self):
        # 对gh-6272进行检查，使用np.arange创建长度为5的数组x
        x = np.arange(5)
        # 使用np.choose函数，从[x[:3], x[:3], x[:3]]中选择[0, 0, 0]的元素，放入x[1:4]中，使用"wrap"模式
        y = np.choose([0, 0, 0], [x[:3], x[:3], x[:3]], out=x[1:4], mode="wrap")
        # 断言y与预期的数组[0, 1, 2]相等
        assert_equal(y, np.array([0, 1, 2]))

    # 定义测试函数test_prod，用于测试numpy中的prod函数
    def test_prod(self):
        # 创建包含元素为[1, 2, 10, 11, 6, 5, 4]的列表ba
        ba = [1, 2, 10, 11, 6, 5, 4]
        # 创建包含子数组的列表ba2
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]

        # 遍历不同的数据类型ctype
        for ctype in [
            np.int16,
            np.int32,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]:
            # 根据ba创建numpy数组a，数据类型为ctype
            a = np.array(ba, ctype)
            # 根据ba2创建numpy数组a2，数据类型为ctype
            a2 = np.array(ba2, ctype)
            # 如果ctype在["1", "b"]中，断言调用a.prod和a2.prod(axis=1)会引发ArithmeticError异常
            if ctype in ["1", "b"]:
                assert_raises(ArithmeticError, a.prod)
                assert_raises(ArithmeticError, a2.prod, axis=1)
            else:
                # 否则，断言a沿axis=0的乘积为26400
                assert_equal(a.prod(axis=0), 26400)
                # 断言a2沿axis=0的乘积为np.array([50, 36, 84, 180], ctype)
                assert_array_equal(a2.prod(axis=0), np.array([50, 36, 84, 180], ctype))
                # 断言a2沿axis=-1的乘积为np.array([24, 1890, 600], ctype)
                assert_array_equal(a2.prod(axis=-1), np.array([24, 1890, 600], ctype))

    # 定义测试函数test_repeat，用于测试numpy中的repeat函数
    def test_repeat(self):
        # 创建包含元素为[1, 2, 3, 4, 5, 6]的numpy数组m
        m = np.array([1, 2, 3, 4, 5, 6])
        # 将m按(2, 3)形状重新排列得到m_rect
        m_rect = m.reshape((2, 3))

        # 将m按[1, 3, 2, 1, 1, 2]重复，组成数组A
        A = m.repeat([1, 3, 2, 1, 1, 2])
        assert_equal(A, [1, 2, 2, 2, 3, 3, 4, 5, 6, 6])

        # 将m整体重复两次，组成数组A
        A = m.repeat(2)
        assert_equal(A, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

        # 将m_rect按[2, 1]在axis=0上重复，组成数组A
        A = m_rect.repeat([2, 1], axis=0)
        assert_equal(A, [[1, 2, 3], [1, 2, 3], [4, 5, 6]])

        # 将m_rect按[1, 3, 2]在axis=1上重复，组成数组A
        A = m_rect.repeat([1, 3, 2], axis=1)
        assert_equal(A, [[1, 2, 2, 2, 3, 3], [4, 5, 5, 5, 6, 6]])

        # 将m_rect在axis=0上整体重复两次，组成数组A
        A = m_rect.repeat(2, axis=0)
        assert_equal(A, [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])

        # 将m_rect在axis=1上整体重复两次，组成数组A
        A = m_rect.repeat(2, axis=1)
        assert_equal(A, [[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]])
    def test_reshape(self):
        # 创建一个3x4的NumPy数组
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        # 将数组reshape为2行6列，期望得到的目标数组
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(arr.reshape(2, 6), tgt)

        # 将数组reshape为3行4列，期望得到的目标数组
        tgt = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        assert_equal(arr.reshape(3, 4), tgt)

        # 将数组reshape为3行4列，使用Fortran（列序优先）顺序，期望得到的目标数组
        tgt = [[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]]
        assert_equal(arr.reshape((3, 4), order="F"), tgt)

        # 将数组转置后reshape为3行4列，使用C（行序优先）顺序，期望得到的目标数组
        tgt = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
        assert_equal(arr.T.reshape((3, 4), order="C"), tgt)

    def test_round(self):
        def check_round(arr, expected, *round_args):
            # 检查数组arr使用指定参数round_args四舍五入后是否与期望的数组expected相等
            assert_equal(arr.round(*round_args), expected)
            
            # 创建一个与arr相同形状的零数组out，将arr使用指定参数round_args四舍五入后结果存入out
            out = np.zeros_like(arr)
            res = arr.round(*round_args, out=out)
            
            # 检查out数组是否与期望的数组expected相等，同时检查out和res是否是同一个对象
            assert_equal(out, expected)
            assert out is res

        # 测试不同情况下的四舍五入功能
        check_round(np.array([1.2, 1.5]), [1, 2])
        check_round(np.array(1.5), 2)
        check_round(np.array([12.2, 15.5]), [10, 20], -1)
        check_round(np.array([12.15, 15.51]), [12.2, 15.5], 1)
        
        # 复数情况下的四舍五入功能测试
        check_round(np.array([4.5 + 1.5j]), [4 + 2j])
        check_round(np.array([12.5 + 15.5j]), [10 + 20j], -1)

    def test_squeeze(self):
        # 创建一个三维的NumPy数组a
        a = np.array([[[1], [2], [3]]])
        
        # 测试数组a使用squeeze()函数压缩后的结果是否符合期望
        assert_equal(a.squeeze(), [1, 2, 3])
        
        # 测试数组a在指定轴(axis=(0,))上压缩后的结果是否符合期望
        assert_equal(a.squeeze(axis=(0,)), [[1], [2], [3]])
        
        # 测试数组a在指定轴(axis=(2,))上压缩后的结果是否符合期望
        assert_equal(a.squeeze(axis=(2,)), [[1, 2, 3]])

    def test_transpose(self):
        # 创建一个2x2的NumPy数组a
        a = np.array([[1, 2], [3, 4]])
        
        # 测试数组a进行转置操作后的结果是否符合期望
        assert_equal(a.transpose(), [[1, 3], [2, 4]])
        
        # 测试尝试使用错误参数调用转置操作时是否抛出异常
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0))
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 0))
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 1, 2))

    def test_sort(self):
        # 测试浮点数和包含NaN的复数排序的顺序
        # 这里只需检查小于比较，因此只需使用插入排序路径足以测试
        # 我们只测试双精度和复数双精度，因为逻辑相同
        
        # 检查双精度浮点数排序（包含NaN）
        msg = "Test real sort order with nans"
        a = np.array([np.nan, 1, 0])
        b = np.sort(a)
        assert_equal(b, np.flip(a), msg)

    @xpassIfTorchDynamo  # (reason="sort complex")
    def test_sort_complex_nans(self):
        # 检查复数排序（包含NaN）
        msg = "Test complex sort order with nans"
        a = np.zeros(9, dtype=np.complex128)
        a.real += [np.nan, np.nan, np.nan, 1, 0, 1, 1, 0, 0]
        a.imag += [np.nan, 1, 0, np.nan, np.nan, 1, 0, 1, 0]
        b = np.sort(a)
        assert_equal(b, a[::-1], msg)
    # 在小数组上，使用插入排序而不是快速排序和归并排序的算法选择。

    @parametrize("dtype", [np.uint8, np.float16, np.float32, np.float64])
    # 参数化装饰器，用于指定测试用例中的数据类型，包括无符号整数和浮点数类型
    def test_sort_unsigned(self, dtype):
        # 创建一个从0到100的数组a，数据类型为dtype
        a = np.arange(101, dtype=dtype)
        # 创建数组b作为a的反转副本
        b = np.flip(a)
        # 对于每种排序类型进行迭代
        for kind in self.sort_kinds:
            msg = f"scalar sort, kind={kind}"
            # 复制数组a到c，对c进行排序，使用指定的排序类型kind
            c = a.copy()
            c.sort(kind=kind)
            # 断言排序后的c与原始数组a相等，如果不相等则抛出异常，msg为断言失败时的信息
            assert_equal(c, a, msg)
            # 复制数组b到c，对c进行排序，使用指定的排序类型kind
            c = b.copy()
            c.sort(kind=kind)
            # 断言排序后的c与原始数组a相等，如果不相等则抛出异常，msg为断言失败时的信息
            assert_equal(c, a, msg)

    @parametrize(
        "dtype",
        [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64],
    )
    # 参数化装饰器，用于指定测试用例中的数据类型，包括有符号整数和浮点数类型
    def test_sort_signed(self, dtype):
        # 创建一个从-50到50的数组a，数据类型为dtype
        a = np.arange(-50, 51, dtype=dtype)
        # 创建数组b作为a的反转副本
        b = np.flip(a)
        # 对于每种排序类型进行迭代
        for kind in self.sort_kinds:
            msg = f"scalar sort, kind={kind}"
            # 复制数组a到c，对c进行排序，使用指定的排序类型kind
            c = a.copy()
            c.sort(kind=kind)
            # 断言排序后的c与原始数组a相等，如果不相等则抛出异常，msg为断言失败时的信息
            assert_equal(c, a, msg)
            # 复制数组b到c，对c进行排序，使用指定的排序类型kind
            c = b.copy()
            c.sort(kind=kind)
            # 断言排序后的c与原始数组a相等，如果不相等则抛出异常，msg为断言失败时的信息
            assert_equal(c, a, msg)

    @xpassIfTorchDynamo  # (reason="sort complex")
    @parametrize("dtype", [np.float32, np.float64])
    @parametrize("part", ["real", "imag"])
    # 参数化装饰器，用于指定测试用例中的数据类型为浮点数和复数的实部或虚部
    def test_sort_complex(self, part, dtype):
        # 测试复数排序。这些测试使用与标量排序相同的代码，但比较函数不同。
        cdtype = {
            np.single: np.csingle,
            np.double: np.cdouble,
        }[dtype]
        # 创建一个从-50到50的数组a，数据类型为dtype
        a = np.arange(-50, 51, dtype=dtype)
        # 创建a的反转副本b
        b = a[::-1].copy()
        # 创建复数数组ai和bi，ai是a加上虚部1，bi是b加上虚部1，数据类型为cdtype
        ai = (a * (1 + 1j)).astype(cdtype)
        bi = (b * (1 + 1j)).astype(cdtype)
        # 设置ai和bi的part属性为1，part可能是'real'或'imag'
        setattr(ai, part, 1)
        setattr(bi, part, 1)
        # 对于每种排序类型进行迭代
        for kind in self.sort_kinds:
            msg = f"complex sort, {part} part == 1, kind={kind}"
            # 复制数组ai到c，对c进行排序，使用指定的排序类型kind
            c = ai.copy()
            c.sort(kind=kind)
            # 断言排序后的c与原始数组ai相等，如果不相等则抛出异常，msg为断言失败时的信息
            assert_equal(c, ai, msg)
            # 复制数组bi到c，对c进行排序，使用指定的排序类型kind
            c = bi.copy()
            c.sort(kind=kind)
            # 断言排序后的c与原始数组ai相等，如果不相等则抛出异常，msg为断言失败时的信息
            assert_equal(c, ai, msg)

    def test_sort_axis(self):
        # 检查轴处理。这应该对所有类型的排序都是相同的，因此我们只检查一种类型和一种排序类型
        # 创建一个2x2的二维数组a
        a = np.array([[3, 2], [1, 0]])
        # 创建数组b和c作为a按不同轴排序后的期望结果
        b = np.array([[1, 0], [3, 2]])
        c = np.array([[2, 3], [0, 1]])
        # 复制数组a到d，按轴0进行排序
        d = a.copy()
        d.sort(axis=0)
        # 断言排序后的d与数组b相等，如果不相等则抛出异常，msg为断言失败时的信息
        assert_equal(d, b, "test sort with axis=0")
        # 复制数组a到d，按轴1进行排序
        d = a.copy()
        d.sort(axis=1)
        # 断言排序后的d与数组c相等，如果不相等则抛出异常，msg为断言失败时的信息
        assert_equal(d, c, "test sort with axis=1")
        # 复制数组a到d，按默认轴进行排序
        d = a.copy()
        d.sort()
        # 断言排序后的d与数组c相等，如果不相等则抛出异常，msg为断言失败时的信息
        assert_equal(d, c, "test sort with default axis")

    def test_sort_size_0(self):
        # 检查多维空数组的轴处理
        # 创建一个空数组a，并重塑为3x2x1x0的多维数组
        a = np.array([])
        a = a.reshape(3, 2, 1, 0)
        # 对每个可能的轴进行迭代
        for axis in range(-a.ndim, a.ndim):
            msg = f"test empty array sort with axis={axis}"
            # 断言按指定轴排序后的结果与原始数组a相等，如果不相等则抛出异常，msg为断言失败时的信息
            assert_equal(np.sort(a, axis=axis), a, msg)
        # 断言按None轴排序后的结果（扁平化数组）与原始数组a的扁平化结果相等，如果不相等则抛出异常
        msg = "test empty array sort with axis=None"
        assert_equal(np.sort(a, axis=None), a.ravel(), msg)

    @skip(reason="waaay tooo sloooow")
    # 跳过装饰器，理由是“太慢了”
    def test_sort_degraded(self):
        # 测试降级数据集，使用普通快速排序可能需要几分钟才能运行完成
        d = np.arange(1000000)
        do = d.copy()
        x = d
        # 创建一个中位数为3的杀手，其中每个中位数是快速排序分区的排序后第二个最后元素
        while x.size > 3:
            mid = x.size // 2
            x[mid], x[-2] = x[-2], x[mid]
            x = x[:-2]

        # 断言：使用 np.sort 对 d 进行排序后结果应与 do 相同
        assert_equal(np.sort(d), do)
        # 断言：使用 np.argsort 对 d 进行排序后结果应与 do 相同
        assert_equal(d[np.argsort(d)], do)

    @xfail  # (reason="order='F'")
    def test_copy(self):
        def assert_fortran(arr):
            assert_(arr.flags.fortran)
            assert_(arr.flags.f_contiguous)
            assert_(not arr.flags.c_contiguous)

        def assert_c(arr):
            assert_(not arr.flags.fortran)
            assert_(not arr.flags.f_contiguous)
            assert_(arr.flags.c_contiguous)

        a = np.empty((2, 2), order="F")
        # 测试复制 Fortran 数组
        assert_c(a.copy())
        assert_c(a.copy("C"))
        assert_fortran(a.copy("F"))
        assert_fortran(a.copy("A"))

        # 现在测试从 C 数组开始
        a = np.empty((2, 2), order="C")
        assert_c(a.copy())
        assert_c(a.copy("C"))
        assert_fortran(a.copy("F"))
        assert_c(a.copy("A"))

    @skip(reason="no .ctypes attribute")
    @parametrize("dtype", [np.int32])
    def test__deepcopy__(self, dtype):
        # 强制将 NULL 输入数组中
        a = np.empty(4, dtype=dtype)
        ctypes.memset(a.ctypes.data, 0, a.nbytes)

        # 确保不会引发错误，参见 gh-21833
        b = a.__deepcopy__({})

        a[0] = 42
        # 使用 pytest 断言，a 与 b 应该不相等
        with pytest.raises(AssertionError):
            assert_array_equal(a, b)

    def test_argsort(self):
        # 所有 C 标量 argsort 使用相同的代码，仅类型不同
        # 因此，用一个类型进行快速检查足以。排序项的数量必须大于 ~50，以检查实际算法，因为快速排序和归并排序会为小数组使用插入排序。

        for dtype in [np.int32, np.uint8, np.float32]:
            a = np.arange(101, dtype=dtype)
            b = np.flip(a)
            for kind in self.sort_kinds:
                msg = f"scalar argsort, kind={kind}, dtype={dtype}"
                # 断言：使用 argsort(kind=kind) 对 a 进行排序后结果应与 a 相同
                assert_equal(a.copy().argsort(kind=kind), a, msg)
                # 断言：使用 argsort(kind=kind) 对 b 进行排序后结果应与 b 相同
                assert_equal(b.copy().argsort(kind=kind), b, msg)

    @skip(reason="argsort complex")
    # 定义一个测试方法，测试复杂的 argsort 功能
    def test_argsort_complex(self):
        # 创建一个从 0 到 100 的浮点数数组
        a = np.arange(101, dtype=np.float32)
        # 创建一个翻转的数组 b
        b = np.flip(a)

        # 测试复杂 argsort。这些使用与标量相同的代码，但比较函数不同。
        # 创建复数数组 ai 和 bi，使用复数作为虚部并加上实部为 1
        ai = a * 1j + 1
        bi = b * 1j + 1
        # 遍历排序种类 self.sort_kinds
        for kind in self.sort_kinds:
            msg = f"complex argsort, kind={kind}"
            # 断言复数数组 ai 的排序结果与数组 a 相等
            assert_equal(ai.copy().argsort(kind=kind), a, msg)
            # 断言复数数组 bi 的排序结果与数组 b 相等
            assert_equal(bi.copy().argsort(kind=kind), b, msg)
        
        # 创建复数数组 ai 和 bi，使用实数作为虚部并加上虚部为 1
        ai = a + 1j
        bi = b + 1j
        # 再次遍历排序种类 self.sort_kinds
        for kind in self.sort_kinds:
            msg = f"complex argsort, kind={kind}"
            # 断言复数数组 ai 的排序结果与数组 a 相等
            assert_equal(ai.copy().argsort(kind=kind), a, msg)
            # 断言复数数组 bi 的排序结果与数组 b 相等
            assert_equal(bi.copy().argsort(kind=kind), b, msg)

        # 测试复杂数组的 argsort，需要进行字节交换，见 gh-5441
        # 遍历字节序列 "<>" 和复杂类型的类型码
        for endianness in "<>":
            for dt in np.typecodes["Complex"]:
                # 创建复杂数组 arr，指定字节序和数据类型
                arr = np.array([1 + 3.0j, 2 + 2.0j, 3 + 1.0j], dtype=endianness + dt)
                msg = f"byte-swapped complex argsort, dtype={dt}"
                # 断言数组 arr 的排序结果与序号数组相等
                assert_equal(arr.argsort(), np.arange(len(arr), dtype=np.intp), msg)

    @xpassIfTorchDynamo  # (reason="argsort axis TODO")
    # 定义测试方法，测试 argsort 的 axis 参数
    def test_argsort_axis(self):
        # 检查轴处理。对于所有类型特定的 argsort，应该是相同的，因此我们只检查一个类型和一种排序种类
        # 创建二维数组 a、b、c
        a = np.array([[3, 2], [1, 0]])
        b = np.array([[1, 1], [0, 0]])
        c = np.array([[1, 0], [1, 0]])
        # 断言数组 a 按轴 0 排序的结果与数组 b 相等
        assert_equal(a.copy().argsort(axis=0), b)
        # 断言数组 a 按轴 1 排序的结果与数组 c 相等
        assert_equal(a.copy().argsort(axis=1), c)
        # 断言数组 a 默认排序（未指定轴）的结果与数组 c 相等
        assert_equal(a.copy().argsort(), c)

        # 检查多维空数组的轴处理
        # 创建空数组 a，并将其重塑为 3x2x1x0 维度
        a = np.array([])
        a = a.reshape(3, 2, 1, 0)
        # 遍历所有可能的轴范围
        for axis in range(-a.ndim, a.ndim):
            msg = f"test empty array argsort with axis={axis}"
            # 断言对空数组 a 按指定轴排序的结果与全零数组相等
            assert_equal(np.argsort(a, axis=axis), np.zeros_like(a, dtype=np.intp), msg)
        msg = "test empty array argsort with axis=None"
        # 断言对空数组 a 执行默认排序（未指定轴）的结果与展开后的全零数组相等
        assert_equal(
            np.argsort(a, axis=None), np.zeros_like(a.ravel(), dtype=np.intp), msg
        )

        # 检查稳定排序的稳定性
        r = np.arange(100)
        # 标量
        a = np.zeros(100)
        # 断言零数组按稳定排序的结果与序号数组相等
        assert_equal(a.argsort(kind="m"), r)
        # 复数
        a = np.zeros(100, dtype=complex)
        # 断言零数组按稳定排序的结果与序号数组相等
        assert_equal(a.argsort(kind="m"), r)
        # 字符串
        a = np.array(["aaaaaaaaa" for i in range(100)])
        # 断言字符串数组按稳定排序的结果与序号数组相等
        assert_equal(a.argsort(kind="m"), r)
        # Unicode
        a = np.array(["aaaaaaaaa" for i in range(100)], dtype=np.unicode_)
        # 断言 Unicode 字符串数组按稳定排序的结果与序号数组相等
        assert_equal(a.argsort(kind="m"), r)

    @xpassIfTorchDynamo  # (reason="TODO: searchsorted with nans differs in pytorch")
    @parametrize(
        "a",
        [
            subtest(np.array([0, 1, np.nan], dtype=np.float16), name="f16"),
            subtest(np.array([0, 1, np.nan], dtype=np.float32), name="f32"),
            subtest(np.array([0, 1, np.nan]), name="default_dtype"),
        ],
        # 参数化测试方法，测试包含 NaN 的数组的 searchsorted
    )
    )
    # 定义一个测试方法 test_searchsorted_floats，用于测试包含 NaN 的浮点数数组。
    def test_searchsorted_floats(self, a):
        # 测试包含 NaN 的浮点数数组，明确测试半精度、单精度和双精度浮点数，
        # 确保 NaN 处理正确。
        msg = f"Test real ({a.dtype}) searchsorted with nans, side='l'"
        # 执行左侧搜索排序，并断言结果与预期相等
        b = a.searchsorted(a, side="left")
        assert_equal(b, np.arange(3), msg)
        msg = f"Test real ({a.dtype}) searchsorted with nans, side='r'"
        # 执行右侧搜索排序，并断言结果与预期相等
        b = a.searchsorted(a, side="right")
        assert_equal(b, np.arange(1, 4), msg)
        # 检查关键字参数的使用
        a.searchsorted(v=1)
        # 创建一个包含 NaN 的浮点数数组，并进行搜索排序
        x = np.array([0, 1, np.nan], dtype="float32")
        y = np.searchsorted(x, x[-1])
        assert_equal(y, 2)

    @xfail  # (
    #    reason="'searchsorted_out_cpu' not implemented for 'ComplexDouble'"
    # )
    # 定义一个测试方法 test_searchsorted_complex，用于测试包含 NaN 的复数数组。
    def test_searchsorted_complex(self):
        # 测试包含 NaN 的复数数组。
        # 搜索排序例程使用数组类型的比较函数，因此检查是否与排序顺序一致。
        # 检查双精度复数
        a = np.zeros(9, dtype=np.complex128)
        a.real += [0, 0, 1, 1, 0, 1, np.nan, np.nan, np.nan]
        a.imag += [0, 1, 0, 1, np.nan, np.nan, 0, 1, np.nan]
        msg = "Test complex searchsorted with nans, side='l'"
        # 执行左侧搜索排序，并断言结果与预期相等
        b = a.searchsorted(a, side="left")
        assert_equal(b, np.arange(9), msg)
        msg = "Test complex searchsorted with nans, side='r'"
        # 执行右侧搜索排序，并断言结果与预期相等
        b = a.searchsorted(a, side="right")
        assert_equal(b, np.arange(1, 10), msg)
        msg = "Test searchsorted with little endian, side='l'"
        # 使用小端序进行搜索排序，并断言结果与预期相等
        a = np.array([0, 128], dtype="<i4")
        b = a.searchsorted(np.array(128, dtype="<i4"))
        assert_equal(b, 1, msg)
        msg = "Test searchsorted with big endian, side='l'"
        # 使用大端序进行搜索排序，并断言结果与预期相等
        a = np.array([0, 128], dtype=">i4")
        b = a.searchsorted(np.array(128, dtype=">i4"))
        assert_equal(b, 1, msg)

    # 定义一个测试方法 test_searchsorted_n_elements，用于测试不同数量元素的搜索排序。
    def test_searchsorted_n_elements(self):
        # 检查 0 个元素的情况
        a = np.ones(0)
        b = a.searchsorted([0, 1, 2], "left")
        assert_equal(b, [0, 0, 0])
        b = a.searchsorted([0, 1, 2], "right")
        assert_equal(b, [0, 0, 0])
        a = np.ones(1)
        # 检查 1 个元素的情况
        b = a.searchsorted([0, 1, 2], "left")
        assert_equal(b, [0, 0, 1])
        b = a.searchsorted([0, 1, 2], "right")
        assert_equal(b, [0, 1, 1])
        # 检查所有元素相等的情况
        a = np.ones(2)
        b = a.searchsorted([0, 1, 2], "left")
        assert_equal(b, [0, 0, 2])
        b = a.searchsorted([0, 1, 2], "right")
        assert_equal(b, [0, 2, 2])

    @xpassIfTorchDynamo  # (
    #    reason="RuntimeError: self.storage_offset() must be divisible by 8"
    # )
    def test_searchsorted_unaligned_array(self):
        # 测试搜索非对齐数组

        # 创建一个长度为 10 的 NumPy 数组
        a = np.arange(10)
        
        # 创建一个长度为 a.itemsize * a.size + 1 的空数组，数据类型为 uint8
        aligned = np.empty(a.itemsize * a.size + 1, dtype="uint8")
        
        # 将 aligned 数组的第一个元素之后的部分视图化为与 a 相同类型的数组
        unaligned = aligned[1:].view(a.dtype)
        
        # 将数组 a 的数据复制到 unaligned 数组中
        unaligned[:] = a
        
        # 使用 "left" 方式测试搜索非对齐数组
        b = unaligned.searchsorted(a, "left")
        assert_equal(b, a)
        
        # 使用 "right" 方式测试搜索非对齐数组
        b = unaligned.searchsorted(a, "right")
        assert_equal(b, a + 1)
        
        # 使用 "left" 方式测试搜索非对齐键
        b = a.searchsorted(unaligned, "left")
        assert_equal(b, a)
        
        # 使用 "right" 方式测试搜索非对齐键
        b = a.searchsorted(unaligned, "right")
        assert_equal(b, a + 1)

    def test_searchsorted_resetting(self):
        # 测试二分搜索索引的智能重置

        # 创建一个长度为 5 的 NumPy 数组
        a = np.arange(5)
        
        # 使用 "left" 方式测试搜索 [6, 5, 4]
        b = a.searchsorted([6, 5, 4], "left")
        assert_equal(b, [5, 5, 4])
        
        # 使用 "right" 方式测试搜索 [6, 5, 4]
        b = a.searchsorted([6, 5, 4], "right")
        assert_equal(b, [5, 5, 5])

    def test_searchsorted_type_specific(self):
        # 测试所有特定类型的二分搜索函数

        # 获取所有整数和浮点数类型的类型码
        types = "".join((np.typecodes["AllInteger"], np.typecodes["Float"]))
        
        # 遍历每种数据类型
        for dt in types:
            if dt == "?":
                # 如果是布尔类型，创建长度为 2 的数组
                a = np.arange(2, dtype=dt)
                out = np.arange(2)
            else:
                # 否则，创建从 0 到 4 的数组，指定数据类型为 dt
                a = np.arange(0, 5, dtype=dt)
                out = np.arange(5)
            
            # 使用 "left" 方式测试搜索相同数组
            b = a.searchsorted(a, "left")
            assert_equal(b, out)
            
            # 使用 "right" 方式测试搜索相同数组
            b = a.searchsorted(a, "right")
            assert_equal(b, out + 1)

    @xpassIfTorchDynamo  # (reason="ndarray ctor")
    def test_searchsorted_type_specific_2(self):
        # 测试所有特定类型的二分搜索函数

        # 获取所有整数、浮点数和布尔类型的类型码
        types = "".join((np.typecodes["AllInteger"], np.typecodes["AllFloat"], "?"))
        
        # 遍历每种数据类型
        for dt in types:
            if dt == "?":
                # 如果是布尔类型，创建长度为 2 的数组
                a = np.arange(2, dtype=dt)
                out = np.arange(2)
            else:
                # 否则，创建从 0 到 4 的数组，指定数据类型为 dt
                a = np.arange(0, 5, dtype=dt)
                out = np.arange(5)

            # 创建一个空的 NumPy 数组，形状为 0，缓冲区为 b""，数据类型为 dt
            e = np.ndarray(shape=0, buffer=b"", dtype=dt)
            
            # 使用 "left" 方式测试空数组 e 对数组 a 的搜索
            b = e.searchsorted(a, "left")
            assert_array_equal(b, np.zeros(len(a), dtype=np.intp))
            
            # 使用 "left" 方式测试数组 a 对空数组 e 的搜索
            b = a.searchsorted(e, "left")
            assert_array_equal(b, np.zeros(0, dtype=np.intp))
    # 定义一个测试函数，用于测试在排序数组上使用非法排序器时的异常情况
    def test_searchsorted_with_invalid_sorter(self):
        # 创建一个 NumPy 数组
        a = np.array([5, 2, 1, 3, 4])
        # 对数组进行排序并获取排序索引
        s = np.argsort(a)
        # 断言：当传入的排序器参数包含非整数值时，期望抛出 TypeError 或 RuntimeError 异常
        assert_raises((TypeError, RuntimeError), np.searchsorted, a, 0, sorter=[1.1])
        # 断言：当传入的排序器参数不完全匹配排序数组的长度时，期望抛出 ValueError 或 RuntimeError 异常
        assert_raises(
            (ValueError, RuntimeError), np.searchsorted, a, 0, sorter=[1, 2, 3, 4]
        )
        assert_raises(
            (ValueError, RuntimeError), np.searchsorted, a, 0, sorter=[1, 2, 3, 4, 5, 6]
        )

        # bounds check : XXX torch does not raise
        # 以下是一些被注释掉的断言，用于测试超出边界值在特定情况下是否会引发异常
        # assert_raises(ValueError, np.searchsorted, a, 4, sorter=[0, 1, 2, 3, 5])
        # assert_raises(ValueError, np.searchsorted, a, 0, sorter=[-1, 0, 1, 2, 3])
        # assert_raises(ValueError, np.searchsorted, a, 0, sorter=[4, 0, -1, 2, 3])

    @xpassIfTorchDynamo  # (reason="self.storage_offset() must be divisible by 8")
    @xpassIfTorchDynamo  # (reason="TODO argpartition")
    @parametrize("dtype", "efdFDBbhil?")
    # 参数化测试函数，用于测试在 argpartition 函数中使用超出范围的 kth 值时是否会引发异常
    def test_argpartition_out_of_range(self, dtype):
        # 创建一个 NumPy 数组，以指定的数据类型
        d = np.arange(10).astype(dtype=dtype)
        # 断言：当传入的 kth 值超出数组长度时，期望抛出 ValueError 异常
        assert_raises(ValueError, d.argpartition, 10)
        assert_raises(ValueError, d.argpartition, -11)

    @xpassIfTorchDynamo  # (reason="TODO partition")
    @parametrize("dtype", "efdFDBbhil?")
    # 参数化测试函数，用于测试在 partition 函数中使用超出范围的 kth 值时是否会引发异常
    def test_partition_out_of_range(self, dtype):
        # 创建一个 NumPy 数组，以指定的数据类型
        d = np.arange(10).astype(dtype=dtype)
        # 断言：当传入的 kth 值超出数组长度时，期望抛出 ValueError 异常
        assert_raises(ValueError, d.partition, 10)
        assert_raises(ValueError, d.partition, -11)

    @xpassIfTorchDynamo  # (reason="TODO argpartition")
    # 标记为跳过，理由是还未实现对 argpartition 函数的测试
    def test_argpartition_integer(self):
        # 测试在 argpartition 函数中传入非整数值是否会引发 TypeError 异常
        d = np.arange(10)
        assert_raises(TypeError, d.argpartition, 9.0)
        # 还测试了泛型类型的 argpartition，该函数使用排序，不会对 kth 值进行边界检查
        d_obj = np.arange(10, dtype=object)
        assert_raises(TypeError, d_obj.argpartition, 9.0)

    @xpassIfTorchDynamo  # (reason="TODO partition")
    # 标记为跳过，理由是还未实现对 partition 函数的测试
    def test_partition_integer(self):
        # 测试在 partition 函数中传入非整数值是否会引发 TypeError 异常
        d = np.arange(10)
        assert_raises(TypeError, d.partition, 9.0)
        # 还测试了泛型类型的 partition，该函数使用排序，不会对 kth 值进行边界检查
        d_obj = np.arange(10, dtype=object)
        assert_raises(TypeError, d_obj.partition, 9.0)

    @xpassIfTorchDynamo  # (reason="TODO partition")
    @parametrize("kth_dtype", "Bbhil")
    # 参数化测试函数，用于测试在 partition 函数中使用特定数据类型的 kth 值时是否会引发异常
    # 定义一个测试方法，用于测试在空数组上的分区操作
    def test_partition_empty_array(self, kth_dtype):
        # 检查多维空数组的轴处理
        kth = np.array(0, dtype=kth_dtype)[()]  # 创建一个指定类型和值的NumPy数组
        a = np.array([])  # 创建一个空的NumPy数组
        a.shape = (3, 2, 1, 0)  # 修改数组的形状为多维空数组
        # 遍历数组的所有轴
        for axis in range(-a.ndim, a.ndim):
            msg = f"test empty array partition with axis={axis}"  # 设置测试消息
            # 断言分区操作后的结果与原数组相等
            assert_equal(np.partition(a, kth, axis=axis), a, msg)
        msg = "test empty array partition with axis=None"  # 设置测试消息
        # 断言在axis=None时，分区操作后数组展平后的结果与原数组展平相等
        assert_equal(np.partition(a, kth, axis=None), a.ravel(), msg)

    @xpassIfTorchDynamo  # (reason="TODO argpartition")
    @parametrize("kth_dtype", "Bbhil")
    # 定义一个测试方法，用于测试在空数组上的参数分区操作
    def test_argpartition_empty_array(self, kth_dtype):
        # 检查多维空数组的轴处理
        kth = np.array(0, dtype=kth_dtype)[()]  # 创建一个指定类型和值的NumPy数组
        a = np.array([])  # 创建一个空的NumPy数组
        a.shape = (3, 2, 1, 0)  # 修改数组的形状为多维空数组
        # 遍历数组的所有轴
        for axis in range(-a.ndim, a.ndim):
            msg = f"test empty array argpartition with axis={axis}"  # 设置测试消息
            # 断言参数分区操作后的结果与全零数组（与原数组相同形状）相等
            assert_equal(
                np.partition(a, kth, axis=axis), np.zeros_like(a, dtype=np.intp), msg
            )
        msg = "test empty array argpartition with axis=None"  # 设置测试消息
        # 断言在axis=None时，参数分区操作后数组展平后的结果与全零数组（与展平后原数组相同形状）相等
        assert_equal(
            np.partition(a, kth, axis=None),
            np.zeros_like(a.ravel(), dtype=np.intp),
            msg,
        )

    @xpassIfTorchDynamo  # (reason="TODO partition")
    # 定义一个方法，用于断言给定字典中的数据是否已经正确分区
    def assert_partitioned(self, d, kth):
        prev = 0  # 初始化前一个索引
        # 对排序后的kth值进行遍历
        for k in np.sort(kth):
            # 断言当前索引到k之间的元素小于第k个元素
            assert_array_less(d[prev:k], d[k], err_msg="kth %d" % k)
            # 断言从第k个元素开始后面的所有元素大于等于第k个元素
            assert_(
                (d[k:] >= d[k]).all(),
                msg="kth %d, %r not greater equal %d" % (k, d[k:], d[k]),
            )
            prev = k + 1  # 更新前一个索引

    @xpassIfTorchDynamo  # (reason="TODO partition")
    # 定义一个名为 test_partition_iterative 的测试方法，用于测试分区函数的迭代实现
    def test_partition_iterative(self):
        # 创建一个包含 0 到 16 的数组
        d = np.arange(17)
        # 定义 kth 变量为元组 (0, 1, 2, 429, 231)，这些值超出数组范围，预期会引发 ValueError 异常
        kth = (0, 1, 2, 429, 231)
        # 断言调用 d.partition(kth) 会引发 ValueError 异常
        assert_raises(ValueError, d.partition, kth)
        # 断言调用 d.argpartition(kth) 会引发 ValueError 异常
        assert_raises(ValueError, d.argpartition, kth)
        
        # 重新定义 d 为一个形状为 (2, 5) 的二维数组
        d = np.arange(10).reshape((2, 5))
        # 断言调用 d.partition(kth, axis=0) 会引发 ValueError 异常
        assert_raises(ValueError, d.partition, kth, axis=0)
        # 断言调用 d.partition(kth, axis=1) 会引发 ValueError 异常
        assert_raises(ValueError, d.partition, kth, axis=1)
        # 断言调用 np.partition(d, kth, axis=1) 会引发 ValueError 异常
        assert_raises(ValueError, np.partition, d, kth, axis=1)
        # 断言调用 np.partition(d, kth, axis=None) 会引发 ValueError 异常
        assert_raises(ValueError, np.partition, d, kth, axis=None)
        
        # 重新定义 d 为数组 [3, 4, 2, 1]
        d = np.array([3, 4, 2, 1])
        # 调用 np.partition(d, (0, 3)) 进行分区，返回结果 p
        p = np.partition(d, (0, 3))
        # 使用 self.assert_partitioned 验证 p 中的分区情况是否符合预期 (0, 3)
        self.assert_partitioned(p, (0, 3))
        # 使用 d[np.argpartition(d, (0, 3))] 获取根据索引分区后的数组，并使用 self.assert_partitioned 进行验证
        self.assert_partitioned(d[np.argpartition(d, (0, 3))], (0, 3))
        
        # 断言 np.partition(d, (-3, -1)) 的结果与 p 相等
        assert_array_equal(p, np.partition(d, (-3, -1)))
        # 断言 d[np.argpartition(d, (-3, -1))] 的结果与 p 相等
        assert_array_equal(p, d[np.argpartition(d, (-3, -1))])
        
        # 重新定义 d 为包含 0 到 16 的数组，并进行随机打乱
        d = np.arange(17)
        np.random.shuffle(d)
        # 调用 d.partition(range(d.size))，将数组自身分区排序
        d.partition(range(d.size))
        # 断言 d 的结果与 np.arange(17) 相等
        assert_array_equal(np.arange(17), d)
        
        # 再次随机打乱 d
        np.random.shuffle(d)
        # 使用 d.argpartition(range(d.size)) 对 d 进行索引分区排序
        assert_array_equal(np.arange(17), d[d.argpartition(range(d.size))])
        
        # 测试未排序的 kth
        d = np.arange(17)
        np.random.shuffle(d)
        keys = np.array([1, 3, 8, -2])
        np.random.shuffle(d)
        # 调用 np.partition(d, keys)，使用 keys 进行分区排序
        p = np.partition(d, keys)
        # 使用 self.assert_partitioned 验证 p 中的分区情况是否符合 keys
        self.assert_partitioned(p, keys)
        # 调用 d[np.argpartition(d, keys)]，使用 keys 进行索引分区排序，并使用 self.assert_partitioned 进行验证
        p = d[np.argpartition(d, keys)]
        self.assert_partitioned(p, keys)
        np.random.shuffle(keys)
        # 断言 np.partition(d, keys) 的结果与 p 相等
        assert_array_equal(np.partition(d, keys), p)
        # 断言 d[np.argpartition(d, keys)] 的结果与 p 相等
        assert_array_equal(d[np.argpartition(d, keys)], p)
        
        # 测试相等的 kth
        d = np.arange(20)[::-1]
        # 使用 np.partition(d, [5] * 4) 进行分区排序，预期结果为 [5] * 4
        self.assert_partitioned(np.partition(d, [5] * 4), [5])
        # 使用 np.partition(d, [5] * 4 + [6, 13]) 进行分区排序，预期结果为 [5] * 4 + [6, 13]
        self.assert_partitioned(np.partition(d, [5] * 4 + [6, 13]), [5] * 4 + [6, 13])
        # 使用 d[np.argpartition(d, [5] * 4)] 进行索引分区排序，预期结果为 [5]
        self.assert_partitioned(d[np.argpartition(d, [5] * 4)], [5])
        # 使用 d[np.argpartition(d, [5] * 4 + [6, 13])] 进行索引分区排序，预期结果为 [5] * 4 + [6, 13]
        self.assert_partitioned(
            d[np.argpartition(d, [5] * 4 + [6, 13])], [5] * 4 + [6, 13]
        )
        
        # 定义包含 12 个元素的数组 d，并进行随机打乱
        d = np.arange(12)
        np.random.shuffle(d)
        # 创建一个形状为 (4, 12) 的数组 d1，并对其每行进行随机打乱
        d1 = np.tile(np.arange(12), (4, 1))
        map(np.random.shuffle, d1)
        # 创建 d0 作为 d1 的转置
        d0 = np.transpose(d1)
        
        kth = (1, 6, 7, -1)
        # 使用 np.partition(d1, kth, axis=1) 进行二维数组 d1 的行内分区排序
        p = np.partition(d1, kth, axis=1)
        # 使用 d1.argpartition(kth, axis=1) 进行索引分区排序，并使用 np.transpose(d1) 重新索引得到 pa
        pa = d1[np.arange(d1.shape[0])[:, None], d1.argpartition(kth, axis=1)]
        # 断言 p 与 pa 相等
        assert_array_equal(p, pa)
        # 针对 p 的每一行，使用 self.assert_partitioned 进行验证是否符合 kth
        for i in range(d1.shape[0]):
            self.assert_partitioned(p[i, :], kth)
        
        # 使用 np.partition(d0, kth, axis=0) 进行二维数组 d0 的列内分区排序
        p = np.partition(d0, kth, axis=0)
        # 使用 np.argpartition(d0, kth, axis=0) 进行索引分区排序，并使用 np.transpose(d0) 重新索引得到 pa
        pa = d0[np.argpartition(d0, kth, axis=0), np.arange(d0.shape[1])[None, :]]
        # 断言 p 与 pa 相等
        assert_array_equal(p, pa)
        # 针对 p 的每一列，使用 self.assert_partitioned 进行验证是否符合 kth
        for i in range(d0.shape[1]):
            self.assert_partitioned(p[:, i], kth)
    # 定义一个名为 test_partition_fuzz 的测试方法
    def test_partition_fuzz(self):
        # 进行几轮随机数据测试
        for j in range(10, 30):
            for i in range(1, j - 2):
                # 创建长度为 j 的序列 d，并打乱其顺序
                d = np.arange(j)
                np.random.shuffle(d)
                # 将序列 d 中的每个元素对随机数取模，范围为 [2, 30)
                d = d % np.random.randint(2, 30)
                # 随机选择序列 d 中的一个索引
                idx = np.random.randint(d.size)
                # 创建一个包含特定索引值的列表 kth
                kth = [0, idx, i, i + 1]
                # 通过 np.partition 函数计算序列 d 的分区结果，并选取 kth 指定的元素
                tgt = np.sort(d)[kth]
                # 断言 np.partition 函数的结果与预期结果 tgt 相等，若不等则输出错误信息
                assert_array_equal(
                    np.partition(d, kth)[kth],
                    tgt,
                    err_msg=f"data: {d!r}\n kth: {kth!r}",
                )

    @xpassIfTorchDynamo  # (reason="TODO partition")
    # 使用 parametrize 装饰器，对 test_argpartition_gh5524 方法进行参数化测试
    @parametrize("kth_dtype", "Bbhil")
    def test_argpartition_gh5524(self, kth_dtype):
        # 测试 np.argpartition 在列表上的功能
        kth = np.array(1, dtype=kth_dtype)[()]
        # 创建列表 d
        d = [6, 7, 3, 2, 9, 0]
        # 对列表 d 执行 np.argpartition，得到分区后的索引 p
        p = np.argpartition(d, kth)
        # 断言 np.argpartition 的结果符合预期，即 p 应当指向第 kth 小的元素
        self.assert_partitioned(np.array(d)[p], [1])

    @xpassIfTorchDynamo  # (reason="TODO order='F'")
    # 定义测试方法 test_flatten，用于测试 np.ndarray.flatten 方法
    def test_flatten(self):
        # 创建不同形状的 numpy 数组
        x0 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], np.int32)
        # 创建预期的展平结果数组
        y0 = np.array([1, 2, 3, 4, 5, 6], np.int32)
        y0f = np.array([1, 4, 2, 5, 3, 6], np.int32)
        y1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.int32)
        y1f = np.array([1, 5, 3, 7, 2, 6, 4, 8], np.int32)
        # 断言 np.ndarray.flatten 方法的结果与预期的展平结果相等
        assert_equal(x0.flatten(), y0)
        assert_equal(x0.flatten("F"), y0f)
        assert_equal(x0.flatten("F"), x0.T.flatten())
        assert_equal(x1.flatten(), y1)
        assert_equal(x1.flatten("F"), y1f)
        assert_equal(x1.flatten("F"), x1.T.flatten())

    @parametrize("func", (np.dot, np.matmul))
    # 定义一个测试方法，用于测试多种数组乘法函数
    def test_arr_mult(self, func):
        # 创建四个二维数组，分别是单位矩阵、反对角线矩阵、指定形状的数组d以及两个预定义矩阵ddt和dtd
        a = np.array([[1, 0], [0, 1]])  # 单位矩阵
        b = np.array([[0, 1], [1, 0]])  # 反对角线矩阵
        c = np.array([[9, 1], [1, -9]])  # 指定的2x2数组
        d = np.arange(24).reshape(4, 6)  # 4x6的数组，从0到23
        ddt = np.array(  # 预定义的4x4数组ddt
            [
                [55, 145, 235, 325],
                [145, 451, 757, 1063],
                [235, 757, 1279, 1801],
                [325, 1063, 1801, 2539],
            ]
        )
        dtd = np.array(  # 预定义的6x6数组dtd
            [
                [504, 540, 576, 612, 648, 684],
                [540, 580, 620, 660, 700, 740],
                [576, 620, 664, 708, 752, 796],
                [612, 660, 708, 756, 804, 852],
                [648, 700, 752, 804, 856, 908],
                [684, 740, 796, 852, 908, 964],
            ]
        )

        # gemm vs syrk optimizations
        # 针对不同的数据类型进行优化和验证
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)  # 将数组a转换为指定类型et的数组eaf
            # 断言函数func对eaf和其转置的结果与eaf本身相等
            assert_equal(func(eaf, eaf), eaf)
            assert_equal(func(eaf.T, eaf), eaf)
            assert_equal(func(eaf, eaf.T), eaf)
            assert_equal(func(eaf.T, eaf.T), eaf)
            assert_equal(func(eaf.T.copy(), eaf), eaf)
            assert_equal(func(eaf, eaf.T.copy()), eaf)
            assert_equal(func(eaf.T.copy(), eaf.T.copy()), eaf)

        # syrk validations
        # 针对不同的数据类型进行验证
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)  # 将数组a转换为指定类型et的数组eaf
            ebf = b.astype(et)  # 将数组b转换为指定类型et的数组ebf
            # 断言函数func对ebf和其转置的结果与eaf相等
            assert_equal(func(ebf, ebf), eaf)
            assert_equal(func(ebf.T, ebf), eaf)
            assert_equal(func(ebf, ebf.T), eaf)
            assert_equal(func(ebf.T, ebf.T), eaf)
        
        # syrk - different shape
        # 针对不同形状的数组进行验证
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)  # 将数组d转换为指定类型et的数组edf
            eddtf = ddt.astype(et)  # 将数组ddt转换为指定类型et的数组eddtf
            edtdf = dtd.astype(et)  # 将数组dtd转换为指定类型et的数组edtdf
            # 断言函数func对edf和其转置的结果与eddtf相等
            assert_equal(func(edf, edf.T), eddtf)
            assert_equal(func(edf.T, edf), edtdf)

            # 断言函数func对切片后的数组进行计算的结果与拷贝后的切片数组进行计算的结果相等
            assert_equal(
                func(edf[: edf.shape[0] // 2, :], edf[::2, :].T),
                func(edf[: edf.shape[0] // 2, :].copy(), edf[::2, :].T.copy()),
            )
            assert_equal(
                func(edf[::2, :], edf[: edf.shape[0] // 2, :].T),
                func(edf[::2, :].copy(), edf[: edf.shape[0] // 2, :].T.copy()),
            )
    # 定义一个测试函数，用于测试接受一个函数参数的多维数组乘法函数
    def test_arr_mult_2(self, func):
        # syrk - 不同形状、步长和视图的验证
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            # 将数组 d 转换为指定的数据类型 et
            edf = d.astype(et)
            # 断言函数 func 对逆序和转置后的数组的处理结果相同，包括复制后的情况
            assert_equal(
                func(edf[::-1, :], edf.T), func(edf[::-1, :].copy(), edf.T.copy())
            )
            # 断言函数 func 对逆序列的列和转置后的数组的处理结果相同，包括复制后的情况
            assert_equal(
                func(edf[:, ::-1], edf.T), func(edf[:, ::-1].copy(), edf.T.copy())
            )
            # 断言函数 func 对原始数组和逆序行列的转置的处理结果相同，包括复制后的情况
            assert_equal(func(edf, edf[::-1, :].T), func(edf, edf[::-1, :].T.copy()))
            # 断言函数 func 对原始数组和逆序列的列的转置的处理结果相同，包括复制后的情况
            assert_equal(func(edf, edf[:, ::-1].T), func(edf, edf[:, ::-1].T.copy()))

    @parametrize("func", (np.dot, np.matmul))
    @parametrize("dtype", "ifdFD")
    # 定义一个测试函数，用于测试不包含 DGEMV 的情况
    def test_no_dgemv(self, func, dtype):
        # 检查 gemv 前向量参数是否连续
        # gh-12156
        # 创建一个二维数组 a，数据类型为指定的 dtype
        a = np.arange(8.0, dtype=dtype).reshape(2, 4)
        # 创建一个广播为 (4, 1) 的数组 b
        b = np.broadcast_to(1.0, (4, 1))
        # 对 a 和 b 进行函数运算，比较运算结果是否相同，包括对 b 的复制
        ret1 = func(a, b)
        ret2 = func(a, b.copy())
        assert_equal(ret1, ret2)

        # 对 b 的转置和 a 的转置进行函数运算，比较运算结果是否相同，包括对 b 的复制
        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T)
        assert_equal(ret1, ret2)

    @skip(reason="__array_interface__")
    @parametrize("func", (np.dot, np.matmul))
    @parametrize("dtype", "ifdFD")
    # 定义一个测试函数，用于测试不包含 DGEMV 的情况，第二个版本
    def test_no_dgemv_2(self, func, dtype):
        # 检查非对齐数据
        # 创建一个指定数据类型的空数组 a，并将其视图为指定数据类型的二维数组
        dt = np.dtype(dtype)
        a = np.zeros(8 * dt.itemsize // 2 + 1, dtype="int16")[1:].view(dtype)
        a = a.reshape(2, 4)
        # 取出数组 a 的第一行作为数组 b
        b = a[0]
        # 确保数组 a 的数据不对齐
        assert_(a.__array_interface__["data"][0] % dt.itemsize != 0)
        # 对 a 和 b 进行函数运算，比较运算结果是否相同，包括对 a 和 b 的复制
        ret1 = func(a, b)
        ret2 = func(a.copy(), b.copy())
        assert_equal(ret1, ret2)

        # 对 b 的转置和 a 的转置进行函数运算，比较运算结果是否相同，包括对 a 和 b 的复制
        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T.copy())
        assert_equal(ret1, ret2)

    # 定义一个测试 dot 函数的方法
    def test_dot(self):
        # 创建三个二维数组 a、b 和 c
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[0, 1], [1, 0]])
        c = np.array([[9, 1], [1, -9]])
        # 断言 np.dot(a, b) 等同于 a.dot(b)
        assert_equal(np.dot(a, b), a.dot(b))
        # 断言 np.dot(np.dot(a, b), c) 等同于 a.dot(b).dot(c)
        assert_equal(np.dot(np.dot(a, b), c), a.dot(b).dot(c))

        # 测试传入输出数组的情况
        c = np.zeros_like(a)
        a.dot(b, c)
        assert_equal(c, np.dot(a, b))

        # 测试关键字参数的情况
        c = np.zeros_like(a)
        a.dot(b=b, out=c)
        assert_equal(c, np.dot(a, b))

    @xpassIfTorchDynamo  # (reason="_aligned_zeros")
    def test_dot_out_mem_overlap(self):
        np.random.seed(1)

        # Test BLAS and non-BLAS code paths, including all dtypes
        # that dot() supports
        # 获取所有 dot() 支持的数据类型，排除 "USVM"
        dtypes = [np.dtype(code) for code in np.typecodes["All"] if code not in "USVM"]
        for dtype in dtypes:
            # 创建一个随机数组，指定数据类型为 dtype
            a = np.random.rand(3, 3).astype(dtype)

            # 创建一个对齐的全零数组，指定数据类型为 dtype
            b = _aligned_zeros((3, 3), dtype=dtype)
            b[...] = np.random.rand(3, 3)

            # 使用 np.dot() 计算 a 和 b 的矩阵乘积
            y = np.dot(a, b)
            # 使用指定的输出数组 b 计算 a 和 b 的矩阵乘积
            x = np.dot(a, b, out=b)
            # 断言输出结果 x 和 y 相等，用于验证计算的正确性
            assert_equal(x, y, err_msg=repr(dtype))

            # 检查使用无效的输出数组时是否会引发 ValueError
            assert_raises(ValueError, np.dot, a, b, out=b[::2])
            assert_raises(ValueError, np.dot, a, b, out=b.T)

    @xpassIfTorchDynamo  # (reason="TODO: overlapping memor in matmul")
    def test_matmul_out(self):
        # overlapping memory
        # 创建一个形状为 (2, 3, 3) 的数组 a
        a = np.arange(18).reshape(2, 3, 3)
        # 使用 np.matmul() 计算数组 a 与自身的矩阵乘积，并将结果赋给 b
        b = np.matmul(a, a)
        # 使用指定的输出数组 a 计算数组 a 与自身的矩阵乘积，并将结果赋给 c
        c = np.matmul(a, a, out=a)
        # 断言 c 和 b 是相同的对象
        assert_(c is a)
        # 断言 c 和 b 的值相等，用于验证计算的正确性
        assert_equal(c, b)
        # 重新初始化数组 a
        a = np.arange(18).reshape(2, 3, 3)
        # 使用指定的输出数组 a[::-1, ...] 计算数组 a 与自身的矩阵乘积，并将结果赋给 c
        c = np.matmul(a, a, out=a[::-1, ...])
        # 断言 c 的底层数据是数组 a 的底层数据
        assert_(c.base is a.base)
        # 断言 c 和 b 的值相等，用于验证计算的正确性
        assert_equal(c, b)

    def test_diagonal(self):
        # 创建一个形状为 (3, 4) 的数组 a，其元素从 0 到 11
        a = np.arange(12).reshape((3, 4))
        # 获取主对角线的元素
        assert_equal(a.diagonal(), [0, 5, 10])
        # 获取主对角线的元素，与 axis1=0, axis2=0 等效
        assert_equal(a.diagonal(0), [0, 5, 10])
        # 获取次对角线的元素
        assert_equal(a.diagonal(1), [1, 6, 11])
        # 获取次对角线的元素
        assert_equal(a.diagonal(-1), [4, 9])
        # 断言使用错误的轴参数时会引发 np.AxisError
        assert_raises(np.AxisError, a.diagonal, axis1=0, axis2=5)
        assert_raises(np.AxisError, a.diagonal, axis1=5, axis2=0)
        assert_raises(np.AxisError, a.diagonal, axis1=5, axis2=5)
        assert_raises((ValueError, RuntimeError), a.diagonal, axis1=1, axis2=1)

        # 创建一个形状为 (2, 2, 2) 的数组 b，其元素从 0 到 7
        b = np.arange(8).reshape((2, 2, 2))
        # 获取主对角线的元素
        assert_equal(b.diagonal(), [[0, 6], [1, 7]])
        # 获取主对角线的元素，与 axis1=0, axis2=0 等效
        assert_equal(b.diagonal(0), [[0, 6], [1, 7]])
        # 获取次对角线的元素
        assert_equal(b.diagonal(1), [[2], [3]])
        # 获取次对角线的元素
        assert_equal(b.diagonal(-1), [[4], [5]])
        # 断言使用错误的轴参数时会引发 np.AxisError
        assert_raises((ValueError, RuntimeError), b.diagonal, axis1=0, axis2=0)
        # 获取指定轴的对角线元素
        assert_equal(b.diagonal(0, 1, 2), [[0, 3], [4, 7]])
        assert_equal(b.diagonal(0, 0, 1), [[0, 6], [1, 7]])
        # 获取指定偏移和轴参数的对角线元素
        assert_equal(b.diagonal(offset=1, axis1=0, axis2=2), [[1], [3]])
        # 轴参数的顺序不影响结果
        assert_equal(b.diagonal(0, 2, 1), [[0, 3], [4, 7]])

    @xfail  # (reason="no readonly views")
    def test_diagonal_view_notwriteable(self):
        # 获取单位矩阵的主对角线，并断言其不可写
        a = np.eye(3).diagonal()
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)

        # 获取单位矩阵的主对角线，并断言其不可写
        a = np.diagonal(np.eye(3))
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)

        # 获取单位矩阵的主对角线，并断言其不可写
        a = np.diag(np.eye(3))
        assert_(not a.flags.writeable)
        assert_(not a.flags.owndata)
    # 测试对角线内存泄漏
    def test_diagonal_memleak(self):
        # 回归测试，检查是否存在某个bug
        a = np.zeros((100, 100))
        # 如果支持引用计数，则检查引用计数是否小于50
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a) < 50)
        # 循环遍历100次，获取对角线元素
        for i in range(100):
            a.diagonal()
        # 如果支持引用计数，则再次检查引用计数是否小于50
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a) < 50)

    # 测试大小为零的内存泄漏
    def test_size_zero_memleak(self):
        # 回归测试，针对issue 9615
        # 在cblasfuncs中特殊情况下的点积长度为零的情况（特定于浮点类型）
        a = np.array([], dtype=np.float64)
        x = np.array(2.0)
        # 循环遍历100次，计算点积
        for _ in range(100):
            np.dot(a, a, out=x)
        # 如果支持引用计数，则检查引用计数是否小于50
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(x) < 50)

    # 测试迹
    def test_trace(self):
        # 创建一个3x4的数组
        a = np.arange(12).reshape((3, 4))
        # 检查迹的值是否为15
        assert_equal(a.trace(), 15)
        # 检查迹的值是否为15
        assert_equal(a.trace(0), 15)
        # 检查迹的值是否为18
        assert_equal(a.trace(1), 18)
        # 检查迹的值是否为13
        assert_equal(a.trace(-1), 13)

        # 创建一个2x2x2的数组
        b = np.arange(8).reshape((2, 2, 2))
        # 检查迹的值是否为[6, 8]
        assert_equal(b.trace(), [6, 8])
        # 检查迹的值是否为[6, 8]
        assert_equal(b.trace(0), [6, 8])
        # 检查迹的值是否为[2, 3]
        assert_equal(b.trace(1), [2, 3])
        # 检查迹的值是否为[4, 5]
        assert_equal(b.trace(-1), [4, 5])
        # 检查迹的值是否为[6, 8]
        assert_equal(b.trace(0, 0, 1), [6, 8])
        # 检查迹的值是否为[5, 9]
        assert_equal(b.trace(0, 0, 2), [5, 9])
        # 检查迹的值是否为[3, 11]
        assert_equal(b.trace(0, 1, 2), [3, 11])
        # 检查迹的值是否为[1, 3]
        assert_equal(b.trace(offset=1, axis1=0, axis2=2), [1, 3])

        # 创建一个数组out为1
        out = np.array(1)
        # 调用trace方法，将结果赋给out
        ret = a.trace(out=out)
        # 断言ret和out是同一个对象
        assert ret is out

    # 测试put方法
    def test_put(self):
        # 获取所有整数和浮点数类型码
        icodes = np.typecodes["AllInteger"]
        fcodes = np.typecodes["AllFloat"]
        # 遍历整数和浮点数类型码
        for dt in icodes + fcodes:
            # 创建目标数组
            tgt = np.array([0, 1, 0, 3, 0, 5], dtype=dt)

            # 测试1维数组
            a = np.zeros(6, dtype=dt)
            a.put([1, 3, 5], [1, 3, 5])
            # 断言a和目标数组相等
            assert_equal(a, tgt)

            # 测试2维数组
            a = np.zeros((2, 3), dtype=dt)
            a.put([1, 3, 5], [1, 3, 5])
            # 断言a和目标数组reshape后相等
            assert_equal(a, tgt.reshape(2, 3))

        # 遍历布尔类型码
        for dt in "?":
            # 创建目标数组
            tgt = np.array([False, True, False, True, False, True], dtype=dt)

            # 测试1维数组
            a = np.zeros(6, dtype=dt)
            a.put([1, 3, 5], [True] * 3)
            # 断言a和目标数组相等
            assert_equal(a, tgt)

            # 测试2维数组
            a = np.zeros((2, 3), dtype=dt)
            a.put([1, 3, 5], [True] * 3)
            # 断言a和目标数组reshape后相等
            assert_equal(a, tgt.reshape(2, 3))

        # 当调用np.put时，确保如果对象不是ndarray，则会引发TypeError
        bad_array = [1, 2, 3]
        assert_raises(TypeError, np.put, bad_array, [0, 2], 5)

    @xpassIfTorchDynamo  # (reason="TODO: implement order='F'")
    @xfailIfTorchDynamo  # flags["OWNDATA"]
    def test_swapaxes(self):
        # 创建一个四维数组，用于测试
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).copy()
        # 获取数组的维度索引
        idx = np.indices(a.shape)
        # 断言数组是否拥有自己的数据副本
        assert_(a.flags["OWNDATA"])
        # 复制数组a到数组b
        b = a.copy()

        # 检查异常情况
        assert_raises(np.AxisError, a.swapaxes, -5, 0)
        assert_raises(np.AxisError, a.swapaxes, 4, 0)
        assert_raises(np.AxisError, a.swapaxes, 0, -5)
        assert_raises(np.AxisError, a.swapaxes, 0, 4)

        # 循环测试不同的轴交换组合
        for i in range(-4, 4):
            for j in range(-4, 4):
                for k, src in enumerate((a, b)):
                    # 执行数组的轴交换操作
                    c = src.swapaxes(i, j)
                    # 检查交换后的形状是否符合预期
                    shape = list(src.shape)
                    shape[i] = src.shape[j]
                    shape[j] = src.shape[i]
                    assert_equal(c.shape, shape, str((i, j, k)))
                    # 检查交换后数组内容是否正确
                    i0, i1, i2, i3 = (dim - 1 for dim in c.shape)
                    j0, j1, j2, j3 = (dim - 1 for dim in src.shape)
                    assert_equal(
                        src[idx[j0], idx[j1], idx[j2], idx[j3]],
                        c[idx[i0], idx[i1], idx[i2], idx[i3]],
                        str((i, j, k)),
                    )
                    # 检查返回的是否是视图而不是拥有数据的新数组，gh-5260
                    assert_(not c.flags["OWNDATA"], str((i, j, k)))
                    # 在非连续输入数组上进行检查
                    if k == 1:
                        b = c
    # 定义测试方法，测试复数类型转换
    def test__complex__(self):
        # 定义各种数据类型的数组
        dtypes = [
            "i1",   # int8
            "i2",   # int16
            "i4",   # int32
            "i8",   # int64
            "u1",   # uint8
            "f",    # float32
            "d",    # float64
            "F",    # complex64
            "D",    # complex128
            "?",    # bool
        ]
        # 遍历数据类型列表
        for dt in dtypes:
            # 创建不同数据类型的数组
            a = np.array(7, dtype=dt)
            b = np.array([7], dtype=dt)
            c = np.array([[[[[7]]]]], dtype=dt)

            # 准备错误消息，标明当前数据类型
            msg = f"dtype: {dt}"
            # 转换数组为复数，应与原数组相等
            ap = complex(a)
            assert_equal(ap, a, msg)
            bp = complex(b)
            assert_equal(bp, b, msg)
            cp = complex(c)
            assert_equal(cp, c, msg)

    # 定义测试方法，测试复数类型转换不应成功的情况
    def test__complex__should_not_work(self):
        # 定义各种数据类型的数组
        dtypes = [
            "i1",   # int8
            "i2",   # int16
            "i4",   # int32
            "i8",   # int64
            "u1",   # uint8
            "f",    # float32
            "d",    # float64
            "F",    # complex64
            "D",    # complex128
            "?",    # bool
        ]
        # 遍历数据类型列表
        for dt in dtypes:
            # 创建不同数据类型的数组，预期会引发错误
            a = np.array([1, 2, 3], dtype=dt)
            assert_raises((TypeError, ValueError), complex, a)

        # 创建复合数据类型的数组，预期会引发错误
        c = np.array([(1.0, 3), (2e-3, 7)], dtype=dt)
        assert_raises((TypeError, ValueError), complex, c)
class TestCequenceMethods(TestCase):
    # 定义测试类 TestCequenceMethods，继承自 TestCase

    def test_array_contains(self):
        # 定义测试方法 test_array_contains
        assert_(4.0 in np.arange(16.0).reshape(4, 4))
        # 断言：验证 4.0 是否在 reshape 后的 np.arange(16.0) 数组中

        assert_(20.0 not in np.arange(16.0).reshape(4, 4))
        # 断言：验证 20.0 是否不在 reshape 后的 np.arange(16.0) 数组中


class TestBinop(TestCase):
    # 定义测试类 TestBinop，继承自 TestCase

    def test_inplace(self):
        # 定义测试方法 test_inplace

        # test refcount 1 inplace conversion
        # 测试引用计数为 1 的原地转换
        assert_array_almost_equal(np.array([0.5]) * np.array([1.0, 2.0]), [0.5, 1.0])

        d = np.array([0.5, 0.5])[::2]
        # 创建数组 d，选取索引为 0 和 2 的元素，即 [0.5]
        assert_array_almost_equal(d * (d * np.array([1.0, 2.0])), [0.25, 0.5])

        a = np.array([0.5])
        b = np.array([0.5])
        c = a + b
        # 将 a 和 b 相加，结果存入 c
        c = a - b
        # 将 a 和 b 相减，结果存入 c
        c = a * b
        # 将 a 和 b 相乘，结果存入 c
        c = a / b
        # 将 a 和 b 相除，结果存入 c
        assert_equal(a, b)
        # 断言：验证 a 是否等于 b
        assert_almost_equal(c, 1.0)
        # 断言：验证 c 是否接近于 1.0

        c = a + b * 2.0 / b * a - a / b
        # 对 c 进行复杂的数学运算
        assert_equal(a, b)
        # 断言：验证 a 是否等于 b
        assert_equal(c, 0.5)
        # 断言：验证 c 是否等于 0.5

        # true divide
        # 真除法
        a = np.array([5])
        b = np.array([3])
        c = (a * a) / b
        # 计算 (a * a) / b 的结果
        assert_almost_equal(c, 25 / 3, decimal=5)
        # 断言：验证 c 是否接近于 25 / 3，精确到小数点后第五位
        assert_equal(a, 5)
        # 断言：验证 a 是否等于 5
        assert_equal(b, 3)
        # 断言：验证 b 是否等于 3


class TestSubscripting(TestCase):
    # 定义测试类 TestSubscripting，继承自 TestCase

    def test_test_zero_rank(self):
        # 定义测试方法 test_test_zero_rank
        x = np.array([1, 2, 3])
        # 创建数组 x
        assert_(isinstance(x[0], (np.int_, np.ndarray)))
        # 断言：验证 x[0] 是否是 np.int_ 或 np.ndarray 的实例
        assert_(type(x[0, ...]) is np.ndarray)
        # 断言：验证 x[0, ...] 的类型是否是 np.ndarray


class TestFancyIndexing(TestCase):
    # 定义测试类 TestFancyIndexing，继承自 TestCase

    def test_list(self):
        # 定义测试方法 test_list
        x = np.ones((1, 1))
        # 创建全为 1 的数组 x，形状为 (1, 1)
        x[:, [0]] = 2.0
        # 将 x 的第一列的所有行的第一个元素设为 2.0
        assert_array_equal(x, np.array([[2.0]]))
        # 断言：验证数组 x 是否等于 np.array([[2.0]])

        x = np.ones((1, 1, 1))
        # 创建全为 1 的数组 x，形状为 (1, 1, 1)
        x[:, :, [0]] = 2.0
        # 将 x 的第一维的所有元素的第三维的第一个元素设为 2.0
        assert_array_equal(x, np.array([[[2.0]]]))
        # 断言：验证数组 x 是否等于 np.array([[[2.0]]]))

    def test_tuple(self):
        # 定义测试方法 test_tuple
        x = np.ones((1, 1))
        # 创建全为 1 的数组 x，形状为 (1, 1)
        x[:, (0,)] = 2.0
        # 将 x 的第一列的所有行的第一个元素设为 2.0
        assert_array_equal(x, np.array([[2.0]]))
        # 断言：验证数组 x 是否等于 np.array([[2.0]])

        x = np.ones((1, 1, 1))
        # 创建全为 1 的数组 x，形状为 (1, 1, 1)
        x[:, :, (0,)] = 2.0
        # 将 x 的第一维的所有元素的第三维的第一个元素设为 2.0
        assert_array_equal(x, np.array([[[2.0]]]))
        # 断言：验证数组 x 是否等于 np.array([[[2.0]]]))

    def test_mask(self):
        # 定义测试方法 test_mask
        x = np.array([1, 2, 3, 4])
        # 创建数组 x，包含元素 [1, 2, 3, 4]
        m = np.array([0, 1, 0, 0], bool)
        # 创建布尔数组 m，指示哪些元素应包括在子集中
        assert_array_equal(x[m], np.array([2]))
        # 断言：验证 x 中根据布尔数组 m 所选择的元素是否等于 np.array([2])

    def test_mask2(self):
        # 定义测试方法 test_mask2
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        # 创建二维数组 x
        m = np.array([0, 1], bool)
        # 创建布尔数组 m，指示哪些行应包括在子集中
        m2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], bool)
        # 创建布尔数组 m2，指示哪些元素应包括在子集中
        m3 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool)
        # 创建布尔数组 m3，指示哪些元素应包括在子集中
        assert_array_equal(x[m], np.array([[5, 6, 7, 8]]))
        # 断言：验证 x 中根据布尔数组 m 所选择的行是否等于 np.array([[5, 6, 7, 8]])
        assert_array_equal(x[m2], np.array([2, 5]))
        # 断言：验证 x 中根据布尔数组 m2 所选择的元素是否等于 np.array([2, 5])
        assert_array_equal(x[m3], np.array([2]))
        # 断言：验证 x 中根据布尔数组 m3 所选择的元素是否等于 np.array([2])

    def test_assign_mask(self):
        # 定义测试方法 test_assign_mask
        x = np.array([1, 2, 3, 4])
        # 创建数组 x，包含元素 [1, 2, 3, 4]
        m = np.array([0, 1, 0, 0], bool)
        # 创建布尔数组 m，指示哪些元素应该更新
        x[m] = 5
        # 将 x 中根据布尔数组 m 所选择的元素更新为 5
        assert_array_equal(x, np.array([1, 5, 3, 4]))
        # 断言：验证数组 x 是否等于 np.array([1, 5, 3, 4])
    # 定义一个测试方法，用于测试数组的赋值操作在存在布尔掩码时的行为
    def test_assign_mask2(self):
        # 创建一个原始的二维 NumPy 数组
        xorig = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        # 创建第一个布尔掩码数组 m，用于指定哪些行要进行赋值操作
        m = np.array([0, 1], bool)
        # 创建第二个布尔掩码数组 m2，用于指定哪些元素要进行赋值操作
        m2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], bool)
        # 创建第三个布尔掩码数组 m3，用于指定哪些元素要进行赋值操作
        m3 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool)
        
        # 对原始数组进行复制，以免改变原始数据
        x = xorig.copy()
        # 使用第一个布尔掩码 m 对数组 x 进行赋值操作
        x[m] = 10
        # 断言数组 x 的结果与预期结果一致
        assert_array_equal(x, np.array([[1, 2, 3, 4], [10, 10, 10, 10]]))
        
        # 对原始数组进行复制，以免改变原始数据
        x = xorig.copy()
        # 使用第二个布尔掩码 m2 对数组 x 进行赋值操作
        x[m2] = 10
        # 断言数组 x 的结果与预期结果一致
        assert_array_equal(x, np.array([[1, 10, 3, 4], [10, 6, 7, 8]]))
        
        # 对原始数组进行复制，以免改变原始数据
        x = xorig.copy()
        # 使用第三个布尔掩码 m3 对数组 x 进行赋值操作
        x[m3] = 10
        # 断言数组 x 的结果与预期结果一致
        assert_array_equal(x, np.array([[1, 10, 3, 4], [5, 6, 7, 8]]))
# 使用装饰器实例化参数化测试类
@instantiate_parametrized_tests
class TestArgmaxArgminCommon(TestCase):
    # 定义不同维度的数组大小作为测试用例
    sizes = [
        (),
        (3,),
        (3, 2),
        (2, 3),
        (3, 3),
        (2, 3, 4),
        (4, 3, 2),
        (1, 2, 3, 4),
        (2, 3, 4, 1),
        (3, 4, 1, 2),
        (4, 1, 2, 3),
        (64,),
        (128,),
        (256,),
    ]

    # 使用@parametrize装饰器对测试方法参数进行参数化
    @parametrize(
        "size, axis",
        list(
            itertools.chain(
                *[
                    [
                        (size, axis)
                        for axis in list(range(-len(size), len(size))) + [None]
                    ]
                    for size in sizes
                ]
            )
        ),
    )
    # 使用@skipif装饰器定义条件跳过测试，条件为numpy版本小于1.23时
    @skipif(numpy.__version__ < "1.23", reason="keepdims is new in numpy 1.22")
    # 对方法参数进行参数化，包含np.argmax和np.argmin两种方法
    @parametrize("method", [np.argmax, np.argmin])
    # 定义测试函数，用于测试 NumPy 中 argmin 和 argmax 函数的 keepdims 参数
    def test_np_argmin_argmax_keepdims(self, size, axis, method):
        # 生成指定尺寸的正态分布随机数组
        arr = np.random.normal(size=size)
        # 如果 size 为 None 或者为空元组，则转换为 NumPy 数组
        if size is None or size == ():
            arr = np.asarray(arr)

        # 处理连续的数组
        if axis is None:
            # 创建一个维度为 size 长度的全为 1 的新形状
            new_shape = [1 for _ in range(len(size))]
        else:
            # 复制 size 的内容到新形状列表，将指定轴的长度设为 1
            new_shape = list(size)
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        # 调用给定的方法计算 arr 在指定轴上的结果
        _res_orig = method(arr, axis=axis)
        # 将结果 reshape 成新形状
        res_orig = _res_orig.reshape(new_shape)
        # 再次调用方法，保持维度
        res = method(arr, axis=axis, keepdims=True)
        # 断言保持维度后的结果与 reshape 后的结果相等
        assert_equal(res, res_orig)
        # 断言保持维度后的结果形状与新形状相等
        assert_(res.shape == new_shape)
        # 创建一个与 res 相同形状和 dtype 的空数组
        outarray = np.empty(res.shape, dtype=res.dtype)
        # 将方法的结果输出到 outarray，保持维度
        res1 = method(arr, axis=axis, out=outarray, keepdims=True)
        # 断言 res1 和 outarray 是同一个对象
        assert_(res1 is outarray)
        # 断言方法结果与 outarray 相等
        assert_equal(res, outarray)

        # 如果 size 长度大于 0
        if len(size) > 0:
            # 创建一个形状与 new_shape 类似的错误形状列表
            wrong_shape = list(new_shape)
            if axis is not None:
                # 如果指定了轴，将该轴的长度设为 2
                wrong_shape[axis] = 2
            else:
                # 否则将第一个轴的长度设为 2
                wrong_shape[0] = 2
            # 创建一个与 wrong_shape 形状相同的空数组
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            # 使用 pytest 断言会引发 ValueError 异常
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)

        # 处理非连续的数组
        if axis is None:
            # 创建一个维度为 size 长度的全为 1 的新形状
            new_shape = [1 for _ in range(len(size))]
        else:
            # 将 size 的内容倒序后复制到新形状列表，将指定轴的长度设为 1
            new_shape = list(size)[::-1]
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        # 调用给定方法计算 arr 的转置在指定轴上的结果
        _res_orig = method(arr.T, axis=axis)
        # 将结果 reshape 成新形状
        res_orig = _res_orig.reshape(new_shape)
        # 再次调用方法，保持维度
        res = method(arr.T, axis=axis, keepdims=True)
        # 断言保持维度后的结果与 reshape 后的结果相等
        assert_equal(res, res_orig)
        # 断言保持维度后的结果形状与新形状相等
        assert_(res.shape == new_shape)
        # 创建一个形状为 new_shape[::-1]，dtype 为 res.dtype 的空数组
        outarray = np.empty(new_shape[::-1], dtype=res.dtype)
        # 将 outarray 转置后赋值给 outarray
        outarray = outarray.T
        # 将方法的结果输出到 outarray，保持维度
        res1 = method(arr.T, axis=axis, out=outarray, keepdims=True)
        # 断言 res1 和 outarray 是同一个对象
        assert_(res1 is outarray)
        # 断言方法结果与 outarray 相等
        assert_equal(res, outarray)

        # 如果 size 长度大于 0
        if len(size) > 0:
            # 创建一个形状与 new_shape 类似的错误形状列表
            wrong_shape = list(new_shape)
            if axis is not None:
                # 如果指定了轴，将该轴的长度设为 2
                wrong_shape[axis] = 2
            else:
                # 否则将第一个轴的长度设为 2
                wrong_shape[0] = 2
            # 创建一个与 wrong_shape 形状相同的空数组
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            # 使用 pytest 断言会引发 ValueError 异常
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)
    # 定义一个测试方法，用于测试给定方法在多维数组上的行为
    def test_all(self, method):
        # 创建一个形状为 (4, 5, 6, 7, 8) 的正态分布随机数组
        a = np.random.normal(0, 1, (4, 5, 6, 7, 8))
        # 获取数组对象中的指定方法，例如 argmax 或 argmin
        arg_method = getattr(a, "arg" + method)
        val_method = getattr(a, method)
        # 遍历数组的维度
        for i in range(a.ndim):
            # 获取当前维度上的最大或最小值
            a_maxmin = val_method(i)
            # 获取当前维度上最大或最小值的索引
            aarg_maxmin = arg_method(i)
            # 创建一个轴的列表，并移除当前维度的索引
            axes = list(range(a.ndim))
            axes.remove(i)
            # 使用轴置换操作，检查最大或最小值索引是否正确
            assert_(np.all(a_maxmin == aarg_maxmin.choose(*a.transpose(i, *axes))))

    @parametrize("method", ["argmax", "argmin"])
    def test_output_shape(self, method):
        # 创建一个全为 1 的形状为 (10, 5) 的数组
        a = np.ones((10, 5))
        # 获取数组对象中的指定方法，例如 argmax 或 argmin
        arg_method = getattr(a, method)
        # 检查一些简单的形状不匹配情况
        out = np.ones(11, dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        out = np.ones((2, 5), dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        out = np.ones((1, 10), dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        out = np.ones(10, dtype=np.int_)
        # 使用输出参数获取最大或最小值索引，并断言输出是否正确
        arg_method(-1, out=out)
        assert_equal(out, arg_method(-1))

    @parametrize("ndim", [0, 1])
    @parametrize("method", ["argmax", "argmin"])
    def test_ret_is_out(self, ndim, method):
        # 创建一个全为 1 的形状为 (4, 256^ndim) 的数组
        a = np.ones((4,) + (256,) * ndim)
        # 获取数组对象中的指定方法，例如 argmax 或 argmin
        arg_method = getattr(a, method)
        # 创建一个与数组形状相同的空数组，用于存放输出结果
        out = np.empty((256,) * ndim, dtype=np.intp)
        # 调用方法获取最大或最小值索引，并断言返回的是输出数组本身
        ret = arg_method(axis=0, out=out)
        assert ret is out

    @parametrize(
        "arr_method, np_method", [("argmax", np.argmax), ("argmin", np.argmin)]
    )
    def test_np_vs_ndarray(self, arr_method, np_method):
        # 创建一个形状为 (2, 3) 的正态分布随机数组
        a = np.random.normal(size=(2, 3))
        # 获取数组对象中的指定方法，例如 argmax 或 argmin
        arg_method = getattr(a, arr_method)

        # 检查位置参数的情况
        out1 = np.zeros(2, dtype=int)
        out2 = np.zeros(2, dtype=int)
        # 比较 ndarray 方法和 numpy 方法的输出是否一致
        assert_equal(arg_method(1, out1), np_method(a, 1, out2))
        assert_equal(out1, out2)

        # 检查关键字参数的情况
        out1 = np.zeros(3, dtype=int)
        out2 = np.zeros(3, dtype=int)
        # 比较 ndarray 方法和 numpy 方法的输出是否一致
        assert_equal(arg_method(out=out1, axis=0), np_method(a, out=out2, axis=0))
        assert_equal(out1, out2)
# 使用装饰器@parametrize("data", nan_arr)将当前测试类中的测试方法参数化，以便使用多组测试数据运行相同的测试用例
@instantiate_parametrized_tests
class TestArgmax(TestCase):
    # 定义无符号整数测试数据
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 0),
        ([3, 3, 3, 3, 2, 2, 2, 2], 0),
        ([0, 1, 2, 3, 4, 5, 6, 7], 7),
        ([7, 6, 5, 4, 3, 2, 1, 0], 0),
    ]
    # 定义有符号整数测试数据，包含无符号整数测试数据
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 3),
        ([1, 2, 3, 4, -1, -2, -3, -4], 3),
    ]
    # 构建数据数组darr，每个元素是一个元组，包含numpy数组和预期结果
    darr = [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (itertools.product(usg_data, (np.uint8,)))
    ]
    # 继续构建数据数组darr，包含有符号整数测试数据
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
            )
        )
    ]
    # 继续构建数据数组darr，包含其他特定情况的测试数据
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                (
                    ([0, 1, 2, 3, np.nan], 4),
                    ([0, 1, 2, np.nan, 3], 3),
                    ([np.nan, 0, 1, 2, 3], 0),
                    ([np.nan, 0, np.nan, 2, 3], 0),
                    # To hit the tail of SIMD multi-level(x4, x1) inner loops
                    # on variant SIMD widthes
                    ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1),
                    ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1),
                    ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1),
                    ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1),
                    ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1),
                ),
                (np.float32, np.float64),
            )
        )
    ]
    # 构建包含NaN值的数组nan_arr，包含darr和其他特定情况的测试数据
    nan_arr = darr + [
        ([False, False, False, False, True], 4),
        ([False, False, False, True, False], 3),
        ([True, False, False, False, False], 0),
        ([True, False, True, False, False], 0),
    ]

    # 使用装饰器@parametrize("data", nan_arr)将当前测试方法参数化，以便使用多组测试数据运行相同的测试用例
    @parametrize("data", nan_arr)
    # 测试各种数据组合的函数，接受一个数据元组作为参数
    def test_combinations(self, data):
        # 解包数据元组，获取数组 arr 和位置 pos
        arr, pos = data
        # 使用 suppress_warnings 上下文管理器来过滤特定的运行时警告
        with suppress_warnings() as sup:
            # 设置过滤器以忽略 "invalid value encountered in reduce" 运行时警告
            sup.filter(RuntimeWarning, "invalid value encountered in reduce")
            # 计算数组 arr 的最大值
            val = np.max(arr)

        # 断言：数组 arr 中的最大值的索引应该等于 pos
        assert_equal(np.argmax(arr), pos, err_msg=f"{arr!r}")
        # 断言：数组 arr 中最大值的值应该等于之前计算的最大值 val
        assert_equal(arr[np.argmax(arr)], val, err_msg=f"{arr!r}")

        # 在数组 arr 上添加填充以测试 SIMD 循环
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        # 断言：重复数组 rarr 中的最大值的索引应该等于 rpos
        assert_equal(np.argmax(rarr), rpos, err_msg=f"{rarr!r}")
        # 断言：重复数组 rarr 中最大值的值应该等于之前计算的最大值 val
        assert_equal(rarr[np.argmax(rarr)], val, err_msg=f"{rarr!r}")

        # 创建一个填充数组 padd，元素为数组 arr 中的最小值，重复 513 次
        padd = np.repeat(np.min(arr), 513)
        # 将填充数组 padd 与原数组 arr 连接起来形成新的数组 rarr
        rarr = np.concatenate((arr, padd))
        rpos = pos
        # 断言：新数组 rarr 中的最大值的索引应该等于 rpos
        assert_equal(np.argmax(rarr), rpos, err_msg=f"{rarr!r}")
        # 断言：新数组 rarr 中最大值的值应该等于之前计算的最大值 val
        assert_equal(rarr[np.argmax(rarr)], val, err_msg=f"{rarr!r}")

    # 测试不同整数类型的最大有符号整数的函数
    def test_maximum_signed_integers(self):
        # 测试 np.int8 类型的数组
        a = np.array([1, 2**7 - 1, -(2**7)], dtype=np.int8)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        # 测试 np.int16 类型的数组
        a = np.array([1, 2**15 - 1, -(2**15)], dtype=np.int16)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        # 测试 np.int32 类型的数组
        a = np.array([1, 2**31 - 1, -(2**31)], dtype=np.int32)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        # 测试 np.int64 类型的数组
        a = np.array([1, 2**63 - 1, -(2**63)], dtype=np.int64)
        assert_equal(np.argmax(a), 1)
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)
# 使用参数化测试装饰器@parametrize，为测试类TestArgmin中的测试方法生成多组测试数据
@instantiate_parametrized_tests
class TestArgmin(TestCase):
    # 定义无符号整数数组的测试数据usg_data，每组数据包括一个列表和一个预期的最小值索引
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 8),
        ([3, 3, 3, 3, 2, 2, 2, 2], 4),
        ([0, 1, 2, 3, 4, 5, 6, 7], 0),
        ([7, 6, 5, 4, 3, 2, 1, 0], 7),
    ]
    # 将有符号整数数组的测试数据sg_data扩展为usg_data的数据加上额外的数据类型和值
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 4),
        ([1, 2, 3, 4, -1, -2, -3, -4], 7),
    ]
    # 生成测试数据数组darr，将每组数据的列表转换为NumPy数组，并加上预期的最小值索引
    darr = [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (itertools.product(usg_data, (np.uint8,)))
    ]
    # 继续扩展darr数组，包括sg_data的所有数据类型和部分数值的组合
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
            )
        )
    ]
    # 最后扩展darr数组，添加包含NaN值的浮点数测试数据，和不同的数据类型组合
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                (
                    ([0, 1, 2, 3, np.nan], 4),
                    ([0, 1, 2, np.nan, 3], 3),
                    ([np.nan, 0, 1, 2, 3], 0),
                    ([np.nan, 0, np.nan, 2, 3], 0),
                    # To hit the tail of SIMD multi-level(x4, x1) inner loops
                    # on variant SIMD widthes
                    ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1),
                    ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1),
                    ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1),
                    ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1),
                    ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1),
                ),
                (np.float32, np.float64),
            )
        )
    ]
    # 将包含NaN值的数组nan_arr扩展为darr的数据加上布尔值数组和预期的最小值索引
    nan_arr = darr + [
        # RuntimeError: "min_values_cpu" not implemented for 'ComplexDouble'
        #    ([0, 1, 2, 3, complex(0, np.nan)], 4),
        #    ([0, 1, 2, 3, complex(np.nan, 0)], 4),
        #    ([0, 1, 2, complex(np.nan, 0), 3], 3),
        #    ([0, 1, 2, complex(0, np.nan), 3], 3),
        #    ([complex(0, np.nan), 0, 1, 2, 3], 0),
        #    ([complex(np.nan, np.nan), 0, 1, 2, 3], 0),
        #    ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
        #    ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
        #    ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),
        #    ([complex(0, 0), complex(0, 2), complex(0, 1)], 0),
        #    ([complex(1, 0), complex(0, 2), complex(0, 1)], 2),
        #    ([complex(1, 0), complex(0, 2), complex(1, 1)], 1),
        ([True, True, True, True, False], 4),
        ([True, True, True, False, True], 3),
        ([False, True, True, True, True], 0),
        ([False, True, False, True, True], 0),
    ]

    # 使用@parametrize将测试方法"data"参数化，传入nan_arr中的各组测试数据进行测试
    @parametrize("data", nan_arr)
    # 定义一个测试方法，用于测试不同数据组合
    def test_combinations(self, data):
        # 解包数据元组
        arr, pos = data
        # 使用上下文管理器抑制特定类型的警告
        with suppress_warnings() as sup:
            # 过滤运行时警告："invalid value encountered in reduce"
            sup.filter(RuntimeWarning, "invalid value encountered in reduce")
            # 计算数组 arr 中的最小值
            min_val = np.min(arr)

        # 断言数组 arr 中最小值的索引等于给定位置 pos
        assert_equal(np.argmin(arr), pos, err_msg=f"{arr!r}")
        # 断言数组 arr 中最小值等于之前计算的最小值 min_val
        assert_equal(arr[np.argmin(arr)], min_val, err_msg=f"{arr!r}")

        # 在数组 arr 中每个元素重复 129 次，用于测试 SIMD 循环
        rarr = np.repeat(arr, 129)
        # 计算重复后的位置
        rpos = pos * 129
        # 断言重复数组 rarr 中最小值的索引等于 rpos
        assert_equal(np.argmin(rarr), rpos, err_msg=f"{rarr!r}")
        # 断言重复数组 rarr 中最小值等于之前计算的最小值 min_val
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg=f"{rarr!r}")

        # 创建一个新数组 padd，该数组将 arr 中的最大值重复 513 次
        padd = np.repeat(np.max(arr), 513)
        # 将 padd 追加到 arr 的末尾，形成新的数组 rarr
        rarr = np.concatenate((arr, padd))
        # 重置 rpos 的值为 pos
        rpos = pos
        # 断言新数组 rarr 中最小值的索引等于 rpos
        assert_equal(np.argmin(rarr), rpos, err_msg=f"{rarr!r}")
        # 断言新数组 rarr 中最小值等于之前计算的最小值 min_val
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg=f"{rarr!r}")

    # 定义一个测试方法，用于测试最小的有符号整数
    def test_minimum_signed_integers(self):
        # 创建 int8 类型的数组 a，包含最小值、最大值及其中间值
        a = np.array([1, -(2**7), -(2**7) + 1, 2**7 - 1], dtype=np.int8)
        # 断言数组 a 中最小值的索引为 1
        assert_equal(np.argmin(a), 1)
        # 将数组 a 中的每个元素重复 129 次，并断言最小值的索引为 129
        a = a.repeat(129)
        assert_equal(np.argmin(a), 129)

        # 创建 int16 类型的数组 a，包含最小值、最大值及其中间值
        a = np.array([1, -(2**15), -(2**15) + 1, 2**15 - 1], dtype=np.int16)
        # 断言数组 a 中最小值的索引为 1
        assert_equal(np.argmin(a), 1)
        # 将数组 a 中的每个元素重复 129 次，并断言最小值的索引为 129
        a = a.repeat(129)
        assert_equal(np.argmin(a), 129)

        # 创建 int32 类型的数组 a，包含最小值、最大值及其中间值
        a = np.array([1, -(2**31), -(2**31) + 1, 2**31 - 1], dtype=np.int32)
        # 断言数组 a 中最小值的索引为 1
        assert_equal(np.argmin(a), 1)
        # 将数组 a 中的每个元素重复 129 次，并断言最小值的索引为 129
        a = a.repeat(129)
        assert_equal(np.argmin(a), 129)

        # 创建 int64 类型的数组 a，包含最小值、最大值及其中间值
        a = np.array([1, -(2**63), -(2**63) + 1, 2**63 - 1], dtype=np.int64)
        # 断言数组 a 中最小值的索引为 1
        assert_equal(np.argmin(a), 1)
        # 将数组 a 中的每个元素重复 129 次，并断言最小值的索引为 129
        a = a.repeat(129)
        assert_equal(np.argmin(a), 129)
class TestMinMax(TestCase):
    # 定义一个测试类 TestMinMax，继承自 TestCase
    @xpassIfTorchDynamo
    def test_scalar(self):
        # 在通过 xpassIfTorchDynamo 装饰器标记的测试方法 test_scalar 中
        # 测试 np.amax 和 np.amin 对标量参数的行为
        assert_raises(np.AxisError, np.amax, 1, 1)
        assert_raises(np.AxisError, np.amin, 1, 1)

        # 测试 np.amax 和 np.amin 对标量参数在指定轴上的最大和最小值
        assert_equal(np.amax(1, axis=0), 1)
        assert_equal(np.amin(1, axis=0), 1)
        
        # 测试 np.amax 和 np.amin 对标量参数在不指定轴上的最大和最小值
        assert_equal(np.amax(1, axis=None), 1)
        assert_equal(np.amin(1, axis=None), 1)

    def test_axis(self):
        # 在测试方法 test_axis 中
        # 测试 np.amax 对数组参数指定超出范围的轴引发 AxisError 异常
        assert_raises(np.AxisError, np.amax, [1, 2, 3], 1000)
        
        # 测试 np.amax 对二维数组参数在指定轴上的最大值
        assert_equal(np.amax([[1, 2, 3]], axis=1), 3)


class TestNewaxis(TestCase):
    # 定义一个测试类 TestNewaxis，继承自 TestCase
    def test_basic(self):
        # 在测试方法 test_basic 中
        # 创建一个包含三个元素的一维数组 sk
        sk = np.array([0, -0.1, 0.1])
        
        # 使用 np.newaxis 扩展 sk 数组为二维数组，然后与标量进行乘法操作
        res = 250 * sk[:, np.newaxis]
        
        # 断言扁平化后的结果与扩展前的乘法结果相等
        assert_almost_equal(res.ravel(), 250 * sk)


class TestClip(TestCase):
    # 定义一个测试类 TestClip，继承自 TestCase
    def _check_range(self, x, cmin, cmax):
        # 定义一个辅助方法 _check_range，用于检查数组 x 是否在给定范围 [cmin, cmax] 内
        assert_(np.all(x >= cmin))
        assert_(np.all(x <= cmax))

    def _clip_type(
        self,
        type_group,
        array_max,
        clip_min,
        clip_max,
        inplace=False,
        expected_min=None,
        expected_max=None,
    ):
        # 定义一个方法 _clip_type，用于测试不同类型数据的 clip 方法行为
        if expected_min is None:
            expected_min = clip_min
        if expected_max is None:
            expected_max = clip_max

        # 遍历给定类型组中的数据类型
        for T in np.sctypes[type_group]:
            # 根据系统字节顺序确定可能的字节顺序
            if sys.byteorder == "little":
                byte_orders = ["=", ">"]
            else:
                byte_orders = ["<", "="]

            # 遍历可能的字节顺序
            for byteorder in byte_orders:
                # 创建指定字节顺序的数据类型
                dtype = np.dtype(T).newbyteorder(byteorder)

                # 创建一个长度为 1000 的随机数组 x，其元素为在 [0, array_max] 范围内的浮点数，并转换为指定类型 dtype
                x = (np.random.random(1000) * array_max).astype(dtype)
                
                if inplace:
                    # 如果 inplace 为 True，就在原地进行 clip 操作，需要使用 unsafe 转换避免警告
                    x.clip(clip_min, clip_max, x, casting="unsafe")
                else:
                    # 否则直接调用 clip 方法
                    x = x.clip(clip_min, clip_max)
                    byteorder = "="

                # 如果数据类型的字节顺序为 '|'，则设置 byteorder 为 '|'
                if x.dtype.byteorder == "|":
                    byteorder = "|"
                
                # 断言数组的字节顺序与预期字节顺序相等
                assert_equal(x.dtype.byteorder, byteorder)
                
                # 调用辅助方法检查数组 x 是否在预期范围内
                self._check_range(x, expected_min, expected_max)
        # 返回最后的数组 x
        return x

    @skip(reason="endianness")
    def test_basic(self):
        # 在测试方法 test_basic 中，跳过测试原因为 'endianness'
        # 遍历 inplace 参数为 False 和 True 时的情况，测试浮点数和整数类型的 clip 方法行为
        for inplace in [False, True]:
            self._clip_type("float", 1024, -12.8, 100.2, inplace=inplace)
            self._clip_type("float", 1024, 0, 0, inplace=inplace)

            self._clip_type("int", 1024, -120, 100, inplace=inplace)
            self._clip_type("int", 1024, 0, 0, inplace=inplace)

            self._clip_type("uint", 1024, 0, 0, inplace=inplace)
            self._clip_type("uint", 1024, -120, 100, inplace=inplace, expected_min=0)
    # 定义一个测试方法，用于测试 numpy 数组的 clip 方法
    def test_max_or_min(self):
        # 创建一个包含整数序列的 numpy 数组
        val = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        # 使用 clip 方法，将数组中小于 3 的元素替换为 3
        x = val.clip(3)
        # 断言：检查数组 x 中所有元素是否都大于等于 3
        assert_(np.all(x >= 3))
        # 使用 clip 方法，将数组中小于 3 的元素替换为 3（显式使用参数 min）
        x = val.clip(min=3)
        # 断言：检查数组 x 中所有元素是否都大于等于 3
        assert_(np.all(x >= 3))
        # 使用 clip 方法，将数组中大于 4 的元素替换为 4
        x = val.clip(max=4)
        # 断言：检查数组 x 中所有元素是否都小于等于 4
        assert_(np.all(x <= 4))

    # 定义一个测试方法，用于测试 numpy 数组处理 NaN 值时的 clip 方法
    def test_nan(self):
        # 创建一个包含 NaN 值的 numpy 数组
        input_arr = np.array([-2.0, np.nan, 0.5, 3.0, 0.25, np.nan])
        # 使用 clip 方法，将数组中小于 -1 的元素替换为 -1，大于 1 的元素替换为 1
        result = input_arr.clip(-1, 1)
        # 预期的结果数组，NaN 值保持不变
        expected = np.array([-1.0, np.nan, 0.5, 1.0, 0.25, np.nan])
        # 断言：检查 result 和 expected 是否相等
        assert_array_equal(result, expected)
# 声明一个装饰器，用于标记测试类，此处是一个占位符
@xpassIfTorchDynamo  # (reason="TODO")
# 定义一个测试类 TestCompress，继承自 TestCase
class TestCompress(TestCase):
    
    # 测试函数：测试 np.compress 在指定轴上的功能
    def test_axis(self):
        # 目标输出结果
        tgt = [[5, 6, 7, 8, 9]]
        # 创建一个 2x5 的数组
        arr = np.arange(10).reshape(2, 5)
        # 使用 np.compress 在 axis=0 上操作数组 arr
        out = np.compress([0, 1], arr, axis=0)
        # 断言输出结果是否符合预期目标
        assert_equal(out, tgt)

        # 下一个测试目标输出结果
        tgt = [[1, 3], [6, 8]]
        # 使用 np.compress 在 axis=1 上操作数组 arr
        out = np.compress([0, 1, 0, 1, 0], arr, axis=1)
        # 断言输出结果是否符合预期目标
        assert_equal(out, tgt)

    # 测试函数：测试 np.compress 在截断数组时的功能
    def test_truncate(self):
        # 目标输出结果
        tgt = [[1], [6]]
        # 创建一个 2x5 的数组
        arr = np.arange(10).reshape(2, 5)
        # 使用 np.compress 在 axis=1 上操作数组 arr
        out = np.compress([0, 1], arr, axis=1)
        # 断言输出结果是否符合预期目标
        assert_equal(out, tgt)

    # 测试函数：测试 np.compress 在扁平化数组时的功能
    def test_flatten(self):
        # 创建一个 2x5 的数组
        arr = np.arange(10).reshape(2, 5)
        # 使用 np.compress 扁平化数组
        out = np.compress([0, 1], arr)
        # 断言输出结果是否等于 1
        assert_equal(out, 1)


# 声明一个装饰器，用于标记测试类，此处是一个占位符
@xpassIfTorchDynamo  # (reason="TODO")
# 使用另一个装饰器，用于参数化测试类的实例化
@instantiate_parametrized_tests
# 定义一个测试类 TestPutmask，继承自 TestCase
class TestPutmask(TestCase):
    
    # 测试辅助函数：在数组 x 上应用 np.putmask，并断言结果是否符合预期
    def tst_basic(self, x, T, mask, val):
        np.putmask(x, mask, val)
        assert_equal(x[mask], np.array(val, T))

    # 测试函数：测试不同的数据类型对 np.putmask 的影响
    def test_ip_types(self):
        # 未经检查的数据类型列表
        unchecked_types = [bytes, str, np.void]

        # 创建一个随机数组 x，并生成一个与之相同大小的掩码 mask
        x = np.random.random(1000) * 100
        mask = x < 40

        # 遍历数值列表 [-100, 0, 15]
        for val in [-100, 0, 15]:
            # 遍历字符类型 "efdFDBbhil?" 中的每个类型
            for types in "efdFDBbhil?":
                # 遍历每个类型的每个 T
                for T in types:
                    # 如果 T 不在未检查的类型列表中
                    if T not in unchecked_types:
                        # 如果 val < 0 且 T 的数据类型是无符号整数
                        if val < 0 and np.dtype(T).kind == "u":
                            val = np.iinfo(T).max - 99
                        # 对 x 的副本应用 tst_basic 函数
                        self.tst_basic(x.copy().astype(T), T, mask, val)

            # 另外，测试一个长度不典型的字符串
            dt = np.dtype("S3")
            # 对 x 转换为 dt 类型后应用 tst_basic 函数
            self.tst_basic(x.astype(dt), dt.type, mask, dt.type(val)[:3])

    # 测试函数：测试 np.putmask 在处理大小不一致的掩码时是否引发 ValueError
    def test_mask_size(self):
        # 断言调用 np.putmask 时，传递不一致大小的掩码会引发 ValueError
        assert_raises(ValueError, np.putmask, np.array([1, 2, 3]), [True], 5)

    # 参数化测试函数：测试 np.putmask 在处理大端和小端字节顺序时的功能
    @parametrize("greater", (True, False))
    def test_byteorder(self, greater):
        # 根据参数 greater 决定 dtype 的字节顺序
        dtype = ">i4" if greater else "<i4"
        # 创建一个指定 dtype 的数组 x
        x = np.array([1, 2, 3], dtype)
        # 对 x 应用 np.putmask
        np.putmask(x, [True, False, True], -1)
        # 断言数组 x 是否符合预期结果
        assert_array_equal(x, [-1, 2, -1])

    # 测试函数：测试 np.putmask 在处理记录数组时的功能
    def test_record_array(self):
        # 注意：记录数组中混合的字节顺序
        rec = np.array(
            [(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
            dtype=[("x", "<f8"), ("y", ">f8"), ("z", "<f8")],
        )
        # 对记录数组的字段 rec["x"] 应用 np.putmask
        np.putmask(rec["x"], [True, False], 10)
        # 断言字段 rec["x"] 是否符合预期结果
        assert_array_equal(rec["x"], [10, 5])
        # 断言字段 rec["y"] 是否符合预期结果
        assert_array_equal(rec["y"], [2, 4])
        # 断言字段 rec["z"] 是否符合预期结果
        assert_array_equal(rec["z"], [3, 3])
        # 再次对字段 rec["y"] 应用 np.putmask
        np.putmask(rec["y"], [True, False], 11)
        # 断言字段 rec["x"] 是否保持不变
        assert_array_equal(rec["x"], [10, 5])
        # 断言字段 rec["y"] 是否符合更新后的预期结果
        assert_array_equal(rec["y"], [11, 4])
        # 断言字段 rec["z"] 是否保持不变
        assert_array_equal(rec["z"], [3, 3])

    # 测试函数：测试 np.putmask 在处理重叠掩码时的功能
    def test_overlaps(self):
        # gh-6272 检查重叠掩码
        x = np.array([True, False, True, False])
        # 在 x[1:4] 上应用重叠掩码
        np.putmask(x[1:4], [True, True, True], x[:3])
        # 断言数组 x 是否符合预期结果
        assert_equal(x, np.array([True, True, False, True]))

        x = np.array([True, False, True, False])
        # 在 x[1:4] 上应用另一种重叠掩码
        np.putmask(x[1:4], x[:3], [True, False, True])
        # 断言数组 x 是否符合预期结果
        assert_equal(x, np.array([True, True, True, True]))
    # 定义一个测试方法，用于验证数组不可写时的行为
    def test_writeable(self):
        # 创建一个包含0到4的NumPy数组
        a = np.arange(5)
        # 将数组的可写标志设置为False，即数组变为不可写
        a.flags.writeable = False

        # 使用pytest的断言检查是否会引发值错误
        with pytest.raises(ValueError):
            # 尝试在不可写数组上执行putmask操作，应该引发异常
            np.putmask(a, a >= 2, 3)

    # 定义一个测试方法，用于验证np.putmask函数的各种参数组合
    def test_kwargs(self):
        # 创建一个包含两个0的NumPy数组
        x = np.array([0, 0])
        # 使用putmask函数，将数组x中对应位置为0的值替换为-1和-2
        np.putmask(x, [0, 1], [-1, -2])
        # 使用assert_array_equal检查结果数组是否与预期相同
        assert_array_equal(x, [0, -2])

        # 重新初始化数组x
        x = np.array([0, 0])
        # 使用命名参数mask和values调用putmask函数，同样替换数组x中对应位置为0的值为-1和-2
        np.putmask(x, mask=[0, 1], values=[-1, -2])
        # 再次使用assert_array_equal检查结果数组是否与预期相同
        assert_array_equal(x, [0, -2])

        # 重新初始化数组x
        x = np.array([0, 0])
        # 使用values和mask参数的反向顺序调用putmask函数，效果应该相同，替换数组x中对应位置为0的值为-1和-2
        np.putmask(x, values=[-1, -2], mask=[0, 1])
        # 再次使用assert_array_equal检查结果数组是否与预期相同
        assert_array_equal(x, [0, -2])

        # 使用pytest的断言检查是否会引发类型错误
        with pytest.raises(TypeError):
            # 调用putmask函数时，使用未知的参数名'a'，应该引发类型错误
            np.putmask(a=x, values=[-1, -2], mask=[0, 1])
@instantiate_parametrized_tests
class TestTake(TestCase):
    # 定义测试方法，用于测试 np.take 的基本功能
    def tst_basic(self, x):
        # 创建索引列表，包含 x 数组的所有行索引
        ind = list(range(x.shape[0]))
        # 断言 np.take 操作后结果与原数组 x 相等
        assert_array_equal(np.take(x, ind, axis=0), x)

    # 测试不同数据类型的输入
    def test_ip_types(self):
        # 生成一个形状为 (2, 3, 4) 的随机数组 x
        x = np.random.random(24) * 100
        x = np.reshape(x, (2, 3, 4))
        # 对于字符集 "efdFDBbhil?" 中的每种类型 T，调用 self.tst_basic 方法测试 x 的副本
        for types in "efdFDBbhil?":
            for T in types:
                self.tst_basic(x.copy().astype(T))

    # 测试索引超出范围时是否引发 IndexError 异常
    def test_raise(self):
        # 生成一个形状为 (2, 3, 4) 的随机数组 x
        x = np.random.random(24) * 100
        x = np.reshape(x, (2, 3, 4))
        # 断言 np.take 在超出索引时抛出 IndexError 异常
        assert_raises(IndexError, np.take, x, [0, 1, 2], axis=0)
        assert_raises(IndexError, np.take, x, [-3], axis=0)
        # 断言 np.take 的正常索引操作是否正确
        assert_array_equal(np.take(x, [-1], axis=0)[0], x[1])

    @xpassIfTorchDynamo  # (reason="XXX: take(..., mode='clip')")
    # 测试 np.take 的剪裁模式是否正常工作
    def test_clip(self):
        # 生成一个形状为 (2, 3, 4) 的随机数组 x
        x = np.random.random(24) * 100
        x = np.reshape(x, (2, 3, 4))
        # 断言 np.take 在剪裁模式下是否正确取值
        assert_array_equal(np.take(x, [-1], axis=0, mode="clip")[0], x[0])
        assert_array_equal(np.take(x, [2], axis=0, mode="clip")[0], x[1])

    @xpassIfTorchDynamo  # (reason="XXX: take(..., mode='wrap')")
    # 测试 np.take 的环绕模式是否正常工作
    def test_wrap(self):
        # 生成一个形状为 (2, 3, 4) 的随机数组 x
        x = np.random.random(24) * 100
        x = np.reshape(x, (2, 3, 4))
        # 断言 np.take 在环绕模式下是否正确取值
        assert_array_equal(np.take(x, [-1], axis=0, mode="wrap")[0], x[1])
        assert_array_equal(np.take(x, [2], axis=0, mode="wrap")[0], x[0])
        assert_array_equal(np.take(x, [3], axis=0, mode="wrap")[0], x[1])

    @xpassIfTorchDynamo  # (reason="XXX: take(mode='wrap')")
    # 测试 np.take 的输出重叠情况是否正常
    def test_out_overlap(self):
        # 检查在输出重叠时的行为
        # 创建一个长度为 5 的一维数组 x
        x = np.arange(5)
        # 对 x 中的一部分进行 np.take 操作，指定 mode="wrap"，将结果保存在 x 的部分区间
        y = np.take(x, [1, 2, 3], out=x[2:5], mode="wrap")
        # 断言操作的结果与预期一致
        assert_equal(y, np.array([1, 2, 3]))

    @parametrize("shape", [(1, 2), (1,), ()])
    # 测试 np.take 返回的数组是否是输出数组的别名
    def test_ret_is_out(self, shape):
        # 生成一个长度为 5 的一维数组 x
        x = np.arange(5)
        # 创建索引数组 inds，全零，与 x 的 dtype 一致
        inds = np.zeros(shape, dtype=np.intp)
        # 创建一个与 x 类型相同的全零数组 out
        out = np.zeros(shape, dtype=x.dtype)
        # 使用 np.take 操作，指定输出数组为 out，返回结果保存在 ret 中
        ret = np.take(x, inds, out=out)
        # 断言返回的结果是输出数组的别名
        assert ret is out


@xpassIfTorchDynamo  # (reason="TODO")
@instantiate_parametrized_tests
class TestLexsort(TestCase):
    # 测试 np.lexsort 的基本用法
    @parametrize(
        "dtype",
        [
            np.uint8,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float16,
            np.float32,
            np.float64,
        ],
    )
    def test_basic(self, dtype):
        # 创建两个数组 a 和 b
        a = np.array([1, 2, 1, 3, 1, 5], dtype=dtype)
        b = np.array([0, 4, 5, 6, 2, 3], dtype=dtype)
        # 使用 np.lexsort 对 (b, a) 进行排序，并保存索引到 idx 中
        idx = np.lexsort((b, a))
        # 预期的排序后的索引结果
        expected_idx = np.array([0, 4, 2, 1, 3, 5])
        # 断言排序后的索引是否符合预期
        assert_array_equal(idx, expected_idx)
        # 断言按排序后的索引取出的 a 是否为升序排列
        assert_array_equal(a[idx], np.sort(a))

    # 测试 np.lexsort 在混合类型数组上的使用
    def test_mixed(self):
        # 创建一个包含混合类型的数组 a 和 b
        a = np.array([1, 2, 1, 3, 1, 5])
        b = np.array([0, 4, 5, 6, 2, 3], dtype="datetime64[D]")
        # 使用 np.lexsort 对 (b, a) 进行排序，并保存索引到 idx 中
        idx = np.lexsort((b, a))
        # 预期的排序后的索引结果
        expected_idx = np.array([0, 4, 2, 1, 3, 5])
        # 断言排序后的索引是否符合预期
        assert_array_equal(idx, expected_idx)
    # 定义测试函数 test_datetime
    def test_datetime(self):
        # 创建包含三个元素的 NumPy 数组，元素类型为 datetime64[D]
        a = np.array([0, 0, 0], dtype="datetime64[D]")
        # 创建包含三个元素的 NumPy 数组，元素类型为 datetime64[D]
        b = np.array([2, 1, 0], dtype="datetime64[D]")
        # 使用 lexsort 函数对数组 a 和 b 进行词典序排序，返回排序后的索引
        idx = np.lexsort((b, a))
        # 预期的排序后的索引数组
        expected_idx = np.array([2, 1, 0])
        # 断言排序后的索引数组与预期的索引数组相等
        assert_array_equal(idx, expected_idx)

        # 创建包含三个元素的 NumPy 数组，元素类型为 timedelta64[D]
        a = np.array([0, 0, 0], dtype="timedelta64[D]")
        # 创建包含三个元素的 NumPy 数组，元素类型为 timedelta64[D]
        b = np.array([2, 1, 0], dtype="timedelta64[D]")
        # 使用 lexsort 函数对数组 a 和 b 进行词典序排序，返回排序后的索引
        idx = np.lexsort((b, a))
        # 预期的排序后的索引数组
        expected_idx = np.array([2, 1, 0])
        # 断言排序后的索引数组与预期的索引数组相等
        assert_array_equal(idx, expected_idx)

    # 定义测试函数 test_object，标记为 gh-6312
    def test_object(self):  # gh-6312
        # 创建包含 1000 个随机整数的 NumPy 数组
        a = np.random.choice(10, 1000)
        # 创建包含 1000 个随机字符串的 NumPy 数组
        b = np.random.choice(["abc", "xy", "wz", "efghi", "qwst", "x"], 1000)

        # 对数组 a 和 b 进行迭代
        for u in a, b:
            # 使用 lexsort 函数对数组 u 转换为对象类型后进行排序，返回排序后的索引
            left = np.lexsort((u.astype("O"),))
            # 使用 argsort 函数对原始数组 u 进行排序，采用 mergesort 算法，返回排序后的索引
            right = np.argsort(u, kind="mergesort")
            # 断言 lexsort 和 argsort 的结果相等
            assert_array_equal(left, right)

        # 对数组 a 和 b 进行迭代
        for u, v in (a, b), (b, a):
            # 使用 lexsort 函数对数组 u 和 v 进行词典序排序，返回排序后的索引
            idx = np.lexsort((u, v))
            # 断言 lexsort 结果与将 u 转换为对象类型后进行 lexsort 的结果相等
            assert_array_equal(idx, np.lexsort((u.astype("O"), v)))
            # 断言 lexsort 结果与将 v 转换为对象类型后进行 lexsort 的结果相等
            assert_array_equal(idx, np.lexsort((u, v.astype("O"))))
            # 将数组 u 和 v 转换为对象类型，再使用 lexsort 函数进行排序，断言结果相等
            u, v = np.array(u, dtype="object"), np.array(v, dtype="object")
            assert_array_equal(idx, np.lexsort((u, v)))

    # 定义测试函数 test_invalid_axis，标记为 gh-7528
    def test_invalid_axis(self):  # gh-7528
        # 创建一个形状为 (42, 3) 的二维 NumPy 数组，包含从 0.0 到 1.0 的等差数列
        x = np.linspace(0.0, 1.0, 42 * 3).reshape(42, 3)
        # 断言调用 lexsort 函数时，传入超出范围的轴参数会抛出 AxisError 异常
        assert_raises(np.AxisError, np.lexsort, x, axis=2)
# 装饰器，用于跳过测试的原因是IO操作不相关
@skip(reason="dont worry about IO")
# 定义一个测试类，用于测试文件读写相关的功能：tofile, fromfile, tobytes, fromstring
class TestIO(TestCase):

    """Test tofile, fromfile, tobytes, and fromstring"""

    # 定义一个pytest的fixture，返回一个数组x，包含复杂数据类型
    @pytest.fixture()
    def x(self):
        shape = (2, 4, 3)
        rand = np.random.random
        x = rand(shape) + rand(shape).astype(complex) * 1j
        x[0, :, 1] = [np.nan, np.inf, -np.inf, np.nan]
        return x

    # 定义一个pytest的fixture，返回临时文件名，支持字符串和路径对象两种形式
    @pytest.fixture(params=["string", "path_obj"])
    def tmp_filename(self, tmp_path, request):
        # 此fixture覆盖两种情况：
        # 一种是文件名是字符串形式，
        # 另一种是文件名是pathlib对象形式
        filename = tmp_path / "file"
        if request.param == "string":
            filename = str(filename)
        return filename

    # 测试不存在文件时的情况，预期抛出OSError异常
    def test_nofile(self):
        # 测试从空BytesIO对象读取数据，期望抛出OSError异常
        b = io.BytesIO()
        assert_raises(OSError, np.fromfile, b, np.uint8, 80)
        d = np.ones(7)
        # 测试将数组d写入空BytesIO对象，期望抛出OSError异常
        assert_raises(OSError, lambda x: x.tofile(b), d)

    # 测试从字符串中解析布尔型数据
    def test_bool_fromstring(self):
        v = np.array([True, False, True, False], dtype=np.bool_)
        y = np.fromstring("1 0 -2.3 0.0", sep=" ", dtype=np.bool_)
        assert_array_equal(v, y)

    # 测试从字符串中解析64位无符号整数数据
    def test_uint64_fromstring(self):
        d = np.fromstring(
            "9923372036854775807 104783749223640", dtype=np.uint64, sep=" "
        )
        e = np.array([9923372036854775807, 104783749223640], dtype=np.uint64)
        assert_array_equal(d, e)

    # 测试从字符串中解析64位有符号整数数据
    def test_int64_fromstring(self):
        d = np.fromstring("-25041670086757 104783749223640", dtype=np.int64, sep=" ")
        e = np.array([-25041670086757, 104783749223640], dtype=np.int64)
        assert_array_equal(d, e)

    # 测试从字符串中解析数据，count设置为0，期望返回空数组
    def test_fromstring_count0(self):
        d = np.fromstring("1,2", sep=",", dtype=np.int64, count=0)
        assert d.shape == (0,)

    # 测试空文本文件的情况
    def test_empty_files_text(self, tmp_filename):
        with open(tmp_filename, "w") as f:
            pass
        y = np.fromfile(tmp_filename)
        assert_(y.size == 0, "Array not empty")

    # 测试空二进制文件的情况
    def test_empty_files_binary(self, tmp_filename):
        with open(tmp_filename, "wb") as f:
            pass
        y = np.fromfile(tmp_filename, sep=" ")
        assert_(y.size == 0, "Array not empty")

    # 测试数据的文件写入和读取的往返测试
    def test_roundtrip_file(self, x, tmp_filename):
        with open(tmp_filename, "wb") as f:
            x.tofile(f)
        # 注意：由于使用了C标准IO，flush+seek方式不起作用
        with open(tmp_filename, "rb") as f:
            y = np.fromfile(f, dtype=x.dtype)
        assert_array_equal(y, x.flat)

    # 测试数据的直接写入和读取的往返测试
    def test_roundtrip(self, x, tmp_filename):
        x.tofile(tmp_filename)
        y = np.fromfile(tmp_filename, dtype=x.dtype)
        assert_array_equal(y, x.flat)

    # 测试数据的写入和读取使用pathlib对象的往返测试
    def test_roundtrip_dump_pathlib(self, x, tmp_filename):
        p = Path(tmp_filename)
        x.dump(p)
        y = np.load(p, allow_pickle=True)
        assert_array_equal(y, x)
    # 测试将二进制数据转换为字符串，并再次转换为原始数据的一致性
    def test_roundtrip_binary_str(self, x):
        # 将数组转换为字节序列
        s = x.tobytes()
        # 从字节序列重新创建数组，使用原始的数据类型
        y = np.frombuffer(s, dtype=x.dtype)
        # 断言新创建的数组与原始数组在内容上完全一致
        assert_array_equal(y, x.flat)

        # 使用不同的内存布局（Fortran顺序）将数组转换为字节序列
        s = x.tobytes("F")
        # 从字节序列重新创建数组，使用原始的数据类型
        y = np.frombuffer(s, dtype=x.dtype)
        # 断言新创建的数组与原始数组在内容上完全一致
        assert_array_equal(y, x.flatten("F"))

    # 测试将实部数组转换为字符串，并再次转换为原始数据的一致性
    def test_roundtrip_str(self, x):
        # 将实部数组展平为一维，并用"@"符号连接成字符串
        x = x.real.ravel()
        s = "@".join(map(str, x))
        # 从字符串重新创建数组，使用指定分隔符"@"
        y = np.fromstring(s, sep="@")
        # 检查精度损失（NaN值）
        nan_mask = ~np.isfinite(x)
        # 断言NaN值处的元素在两个数组中相等
        assert_array_equal(x[nan_mask], y[nan_mask])
        # 断言非NaN值处的元素在两个数组中近似相等，精度为5位小数
        assert_array_almost_equal(x[~nan_mask], y[~nan_mask], decimal=5)

    # 测试将实部数组的字符串表示转换为数组，并与原始数组进行比较
    def test_roundtrip_repr(self, x):
        # 将实部数组展平为一维，并用"@"符号连接成字符串表示
        x = x.real.ravel()
        s = "@".join(map(repr, x))
        # 从字符串表示重新创建数组，使用指定分隔符"@"
        y = np.fromstring(s, sep="@")
        # 断言新创建的数组与原始数组在内容上完全一致
        assert_array_equal(x, y)

    # 测试从文件中读取数据时处理无法seek的情况
    def test_unseekable_fromfile(self, x, tmp_filename):
        # 将数组写入临时文件
        x.tofile(tmp_filename)

        # 定义一个函数用于模拟无法执行seek和tell操作的文件对象
        def fail(*args, **kwargs):
            raise OSError("Can not tell or seek")

        # 打开临时文件以二进制只读方式，禁用缓冲
        with open(tmp_filename, "rb", buffering=0) as f:
            # 将文件对象的seek和tell方法替换为模拟函数
            f.seek = fail
            f.tell = fail
            # 断言从文件中读取数据时会抛出OSError异常
            assert_raises(OSError, np.fromfile, f, dtype=x.dtype)

    # 测试在使用缓冲的情况下从文件中读取数据
    def test_io_open_unbuffered_fromfile(self, x, tmp_filename):
        # 将数组写入临时文件
        x.tofile(tmp_filename)
        # 打开临时文件以二进制只读方式，禁用缓冲
        with open(tmp_filename, "rb", buffering=0) as f:
            # 从文件中读取数据并重新创建数组，使用原始的数据类型
            y = np.fromfile(f, dtype=x.dtype)
            # 断言新创建的数组与原始数组在内容上完全一致
            assert_array_equal(y, x.flat)

    # 测试处理大文件时从文件中读取数据的正确性
    def test_largish_file(self, tmp_filename):
        # 创建一个4MB大小的零数组，并将其写入临时文件
        d = np.zeros(4 * 1024**2)
        d.tofile(tmp_filename)
        # 断言临时文件的大小等于数组的字节数
        assert_equal(os.path.getsize(tmp_filename), d.nbytes)
        # 断言从文件中读取的数据与原始数组在内容上完全一致
        assert_array_equal(d, np.fromfile(tmp_filename))
        
        # 检查偏移量功能
        with open(tmp_filename, "r+b") as f:
            # 将数据追加到文件末尾后，检查文件大小是否变为原来的两倍
            f.seek(d.nbytes)
            d.tofile(f)
            assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)
        
        # 检查追加模式（gh-8329）
        open(tmp_filename, "w").close()  # 清空文件内容
        with open(tmp_filename, "ab") as f:
            d.tofile(f)
        # 断言从文件中读取的数据与原始数组在内容上完全一致
        assert_array_equal(d, np.fromfile(tmp_filename))
        with open(tmp_filename, "ab") as f:
            d.tofile(f)
        # 断言文件大小等于原始数组的两倍
        assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)

    # 测试在使用有缓冲的情况下从文件中读取数据
    def test_io_open_buffered_fromfile(self, x, tmp_filename):
        # 将数组写入临时文件
        x.tofile(tmp_filename)
        # 打开临时文件以二进制只读方式，启用默认的缓冲
        with open(tmp_filename, "rb", buffering=-1) as f:
            # 从文件中读取数据并重新创建数组，使用原始的数据类型
            y = np.fromfile(f, dtype=x.dtype)
        # 断言新创建的数组与原始数组在内容上完全一致
        assert_array_equal(y, x.flat)
    # 测试函数，用于验证从文件读取数据后的文件位置（gh-4118）
    def test_file_position_after_fromfile(self, tmp_filename):
        # 定义不同的文件大小，以默认缓冲区大小的倍数为单位
        sizes = [
            io.DEFAULT_BUFFER_SIZE // 8,
            io.DEFAULT_BUFFER_SIZE,
            io.DEFAULT_BUFFER_SIZE * 8,
        ]

        # 遍历不同大小的文件
        for size in sizes:
            with open(tmp_filename, "wb") as f:
                # 将文件指针定位到指定位置减一处，写入一个空字节
                f.seek(size - 1)
                f.write(b"\0")

            # 遍历不同的文件打开模式
            for mode in ["rb", "r+b"]:
                err_msg = "%d %s" % (size, mode)

                with open(tmp_filename, mode) as f:
                    # 从文件中读取前两个字节
                    f.read(2)
                    # 从文件中读取一个 np.float64 类型的数据
                    np.fromfile(f, dtype=np.float64, count=1)
                    # 获取当前文件指针位置
                    pos = f.tell()

                # 断言当前文件指针位置为10
                assert_equal(pos, 10, err_msg=err_msg)

    # 测试函数，用于验证写入数据后的文件位置（gh-4118）
    def test_file_position_after_tofile(self, tmp_filename):
        # 定义不同的文件大小，以默认缓冲区大小的倍数为单位
        sizes = [
            io.DEFAULT_BUFFER_SIZE // 8,
            io.DEFAULT_BUFFER_SIZE,
            io.DEFAULT_BUFFER_SIZE * 8,
        ]

        # 遍历不同大小的文件
        for size in sizes:
            err_msg = "%d" % (size,)

            with open(tmp_filename, "wb") as f:
                # 将文件指针定位到指定位置减一处，写入一个空字节
                f.seek(size - 1)
                f.write(b"\0")
                # 将文件指针定位到位置10处，写入字节数据"12"
                f.seek(10)
                f.write(b"12")
                # 将 np.float64 类型的数组写入文件
                np.array([0], dtype=np.float64).tofile(f)
                # 获取当前文件指针位置
                pos = f.tell()

            # 断言当前文件指针位置为10 + 2 + 8
            assert_equal(pos, 10 + 2 + 8, err_msg=err_msg)

            with open(tmp_filename, "r+b") as f:
                # 从文件中读取前两个字节
                f.read(2)
                # 在读取和写入之间进行 ANSI C 所需的定位
                f.seek(0, 1)
                # 将 np.float64 类型的数组写入文件
                np.array([0], dtype=np.float64).tofile(f)
                # 获取当前文件指针位置
                pos = f.tell()

            # 断言当前文件指针位置为10
            assert_equal(pos, 10, err_msg=err_msg)

    # 测试函数，用于验证从文件中加载对象数组时的异常情况（gh-12300）
    def test_load_object_array_fromfile(self, tmp_filename):
        with open(tmp_filename, "w") as f:
            # 确保创建一个内容一致的空文件

        with open(tmp_filename, "rb") as f:
            # 断言尝试从文件中读取对象数组时抛出 ValueError 异常
            assert_raises_regex(
                ValueError,
                "Cannot read into object array",
                np.fromfile,
                f,
                dtype=object,
            )

        # 断言尝试从文件名指定的文件中读取对象数组时抛出 ValueError 异常
        assert_raises_regex(
            ValueError,
            "Cannot read into object array",
            np.fromfile,
            tmp_filename,
            dtype=object,
        )
    # 定义测试方法，用于测试从文件读取数据并验证 numpy 数组
    def test_fromfile_offset(self, x, tmp_filename):
        # 将数组 x 写入临时文件 tmp_filename
        with open(tmp_filename, "wb") as f:
            x.tofile(f)

        # 从文件中读取数据到数组 y，偏移量为 0
        with open(tmp_filename, "rb") as f:
            y = np.fromfile(f, dtype=x.dtype, offset=0)
            # 验证读取的数组 y 与原始数组 x 的扁平化版本是否相等
            assert_array_equal(y, x.flat)

        # 从文件中读取部分数据到数组 y，根据偏移量和数量计算读取的字节和项目
        with open(tmp_filename, "rb") as f:
            count_items = len(x.flat) // 8
            offset_items = len(x.flat) // 4
            offset_bytes = x.dtype.itemsize * offset_items
            y = np.fromfile(f, dtype=x.dtype, count=count_items, offset=offset_bytes)
            # 验证读取的数组 y 的部分与原始数组 x 的扁平化版本的相应部分是否相等
            assert_array_equal(y, x.flat[offset_items : offset_items + count_items])

            # 后续的偏移应当叠加
            offset_bytes = x.dtype.itemsize
            z = np.fromfile(f, dtype=x.dtype, offset=offset_bytes)
            # 验证读取的数组 z 与原始数组 x 的扁平化版本的另一部分是否相等
            assert_array_equal(z, x.flat[offset_items + count_items + 1 :])

        # 将数组 x 以逗号为分隔符写入临时文件 tmp_filename
        with open(tmp_filename, "wb") as f:
            x.tofile(f, sep=",")

        # 从文件中读取数据时，使用了不支持 offset 参数的 sep=","
        with open(tmp_filename, "rb") as f:
            assert_raises_regex(
                TypeError,
                "'offset' argument only permitted for binary files",
                np.fromfile,
                tmp_filename,
                dtype=x.dtype,
                sep=",",
                offset=1,
            )

    # 根据平台特性条件，跳过 PyPy 平台的测试
    @skipif(IS_PYPY, reason="bug in PyPy's PyNumber_AsSsize_t")
    # 测试从文件中读取数据时的异常情况处理
    def test_fromfile_bad_dup(self, x, tmp_filename):
        # 定义返回错误值的函数
        def dup_str(fd):
            return "abc"

        def dup_bigint(fd):
            return 2**68

        # 保存旧的 dup 函数，以便恢复
        old_dup = os.dup
        try:
            # 将数组 x 写入临时文件 tmp_filename
            with open(tmp_filename, "wb") as f:
                x.tofile(f)
                # 对于每种返回错误值的 dup 函数，检查是否引发相应的异常
                for dup, exc in ((dup_str, TypeError), (dup_bigint, OSError)):
                    os.dup = dup
                    assert_raises(exc, np.fromfile, f)
        finally:
            # 恢复原始的 dup 函数
            os.dup = old_dup

    # 检查从不同数据源（字符串或文件）读取数据的一致性
    def _check_from(self, s, value, filename, **kw):
        # 如果 kw 中没有包含 "sep" 参数，则使用 np.frombuffer() 读取 s 中的数据
        if "sep" not in kw:
            y = np.frombuffer(s, **kw)
        else:
            # 否则，使用 np.fromstring() 读取 s 中的数据
            y = np.fromstring(s, **kw)
        # 验证读取的数组 y 与给定的 value 是否相等
        assert_array_equal(y, value)

        # 将数据 s 写入文件 filename
        with open(filename, "wb") as f:
            f.write(s)
        # 从文件中读取数据到数组 y，并验证其与给定的 value 是否相等
        y = np.fromfile(filename, **kw)
        assert_array_equal(y, value)

    # Pytest 的 fixture，用于测试十进制分隔符的本地化问题
    @pytest.fixture(params=["period", "comma"])
    def decimal_sep_localization(self, request):
        """
        Including this fixture in a test will automatically
        execute it with both types of decimal separator.

        So::

            def test_decimal(decimal_sep_localization):
                pass

        is equivalent to the following two tests::

            def test_decimal_period_separator():
                pass

            def test_decimal_comma_separator():
                with CommaDecimalPointLocale():
                    pass
        """
        # 根据请求的参数决定执行哪种十进制分隔符的测试
        if request.param == "period":
            yield
        elif request.param == "comma":
            # 使用逗号作为十进制点的本地化设置
            with CommaDecimalPointLocale():
                yield
        else:
            # 如果请求的参数不是 "period" 或 "comma"，则抛出断言错误
            raise AssertionError(request.param)
    # 测试处理 NaN 值的方法
    def test_nan(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b"nan +nan -nan NaN nan(foo) +NaN(BAR) -NAN(q_u_u_x_)",
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            tmp_filename,
            sep=" ",
        )

    # 测试处理无穷大值的方法
    def test_inf(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b"inf +inf -inf infinity -Infinity iNfInItY -inF",
            [np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf],
            tmp_filename,
            sep=" ",
        )

    # 测试处理数字的方法
    def test_numbers(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b"1.234 -1.234 .3 .3e55 -123133.1231e+133",
            [1.234, -1.234, 0.3, 0.3e55, -123133.1231e133],
            tmp_filename,
            sep=" ",
        )

    # 测试处理二进制数据的方法
    def test_binary(self, tmp_filename):
        self._check_from(
            b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@",
            np.array([1, 2, 3, 4]),
            tmp_filename,
            dtype="<f4",
        )

    @slow  # 此测试需要超过 1 分钟的时间在机械硬盘上运行
    def test_big_binary(self):
        """测试解决 MSVC fwrite、fseek 和 ftell 的 32 位限制的方法

        这些通常会在进行类似操作时 hang。
        参见：https://github.com/numpy/numpy/issues/2256
        """
        if sys.platform != "win32" or "[GCC " in sys.version:
            return
        try:
            # 在应用解决方法之前，只有 2**32-1 大小的数据有效
            fourgbplus = 2**32 + 2**16
            testbytes = np.arange(8, dtype=np.int8)
            n = len(testbytes)
            flike = tempfile.NamedTemporaryFile()
            f = flike.file
            np.tile(testbytes, fourgbplus // testbytes.nbytes).tofile(f)
            flike.seek(0)
            a = np.fromfile(f, dtype=np.int8)
            flike.close()
            assert_(len(a) == fourgbplus)
            # 仅检查起始和结束以提高速度：
            assert_((a[:n] == testbytes).all())
            assert_((a[-n:] == testbytes).all())
        except (MemoryError, ValueError):
            pass

    # 测试处理字符串数据的方法
    def test_string(self, tmp_filename):
        self._check_from(b"1,2,3,4", [1.0, 2.0, 3.0, 4.0], tmp_filename, sep=",")

    # 测试处理带计数的字符串数据的方法
    def test_counted_string(self, tmp_filename, decimal_sep_localization):
        self._check_from(
            b"1,2,3,4", [1.0, 2.0, 3.0, 4.0], tmp_filename, count=4, sep=","
        )
        self._check_from(b"1,2,3,4", [1.0, 2.0, 3.0], tmp_filename, count=3, sep=",")
        self._check_from(
            b"1,2,3,4", [1.0, 2.0, 3.0, 4.0], tmp_filename, count=-1, sep=","
        )

    # 测试处理带空格的字符串数据的方法
    def test_string_with_ws(self, tmp_filename):
        self._check_from(
            b"1 2  3     4   ", [1, 2, 3, 4], tmp_filename, dtype=int, sep=" "
        )

    # 测试处理带计数和空格的字符串数据的方法
    def test_counted_string_with_ws(self, tmp_filename):
        self._check_from(
            b"1 2  3     4   ", [1, 2, 3], tmp_filename, count=3, dtype=int, sep=" "
        )
    # 测试 ASCII 数据的解析功能
    def test_ascii(self, tmp_filename, decimal_sep_localization):
        # 使用 _check_from 方法验证从字节数据解析出的结果是否符合预期
        self._check_from(b"1 , 2 , 3 , 4", [1.0, 2.0, 3.0, 4.0], tmp_filename, sep=",")

        # 使用 _check_from 方法验证从字节数据解析出的结果是否符合预期，指定 dtype 为 float
        self._check_from(
            b"1,2,3,4", [1.0, 2.0, 3.0, 4.0], tmp_filename, dtype=float, sep=","
        )

    # 测试异常数据的处理功能
    def test_malformed(self, tmp_filename, decimal_sep_localization):
        # 使用 assert_warns 检查是否会发出 DeprecationWarning 警告
        with assert_warns(DeprecationWarning):
            # 使用 _check_from 方法验证异常格式的字节数据解析出的结果是否符合预期
            self._check_from(b"1.234 1,234", [1.234, 1.0], tmp_filename, sep=" ")

    # 测试长分隔符的处理功能
    def test_long_sep(self, tmp_filename):
        # 使用 _check_from 方法验证带有长分隔符的字节数据解析出的结果是否符合预期
        self._check_from(b"1_x_3_x_4_x_5", [1, 3, 4, 5], tmp_filename, sep="_x_")

    # 测试指定数据类型的解析功能
    def test_dtype(self, tmp_filename):
        # 创建指定数据类型的 NumPy 数组
        v = np.array([1, 2, 3, 4], dtype=np.int_)
        # 使用 _check_from 方法验证从字节数据解析出的结果是否符合预期，指定数据类型为 np.int_
        self._check_from(b"1,2,3,4", v, tmp_filename, sep=",", dtype=np.int_)

    # 测试布尔类型数据的写入和读取功能
    def test_dtype_bool(self, tmp_filename):
        # 创建布尔类型的 NumPy 数组
        v = np.array([True, False, True, False], dtype=np.bool_)
        # 准备用于写入的字节数据
        s = b"1,0,-2.3,0"
        # 将字节数据写入临时文件
        with open(tmp_filename, "wb") as f:
            f.write(s)
        # 从临时文件读取数据，验证读取结果是否符合预期布尔数组
        y = np.fromfile(tmp_filename, sep=",", dtype=np.bool_)
        # 使用 assert_ 断言验证读取的数据类型是否为布尔类型
        assert_(y.dtype == "?")
        # 使用 assert_array_equal 断言验证读取的数据是否与预期的布尔数组一致
        assert_array_equal(y, v)

    # 测试将数组写入文件并从文件读取的功能
    def test_tofile_sep(self, tmp_filename, decimal_sep_localization):
        # 创建浮点数类型的 NumPy 数组
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        # 将数组写入以逗号分隔的文本文件
        with open(tmp_filename, "w") as f:
            x.tofile(f, sep=",")
        # 从文件读取数据
        with open(tmp_filename) as f:
            s = f.read()
        # 将读取的数据转换为数组，并使用 assert_array_equal 断言验证其与原始数组的一致性
        y = np.array([float(p) for p in s.split(",")])
        assert_array_equal(x, y)

    # 测试将数组以指定格式写入文件的功能
    def test_tofile_format(self, tmp_filename, decimal_sep_localization):
        # 创建浮点数类型的 NumPy 数组
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        # 将数组以指定格式（保留两位小数）写入文本文件
        with open(tmp_filename, "w") as f:
            x.tofile(f, sep=",", format="%.2f")
        # 从文件读取数据，并使用 assert_equal 断言验证其与预期的字符串格式一致
        with open(tmp_filename) as f:
            s = f.read()
        assert_equal(s, "1.51,2.00,3.51,4.00")

    # 测试写入文件时的清理功能
    def test_tofile_cleanup(self, tmp_filename):
        # 创建对象类型为 object 的 NumPy 数组
        x = np.zeros((10), dtype=object)
        # 使用 lambda 函数和 assert_raises 断言验证尝试使用非法分隔符时是否会引发 OSError 异常
        with open(tmp_filename, "wb") as f:
            assert_raises(OSError, lambda: x.tofile(f, sep=""))
        # 尝试删除文件，如果文件句柄未关闭，则在 Windows 操作系统上会失败
        os.remove(tmp_filename)

        # 同时确保 Python 文件句柄已关闭
        assert_raises(OSError, lambda: x.tofile(tmp_filename))
        os.remove(tmp_filename)

    # 测试从文件读取子数组的功能
    def test_fromfile_subarray_binary(self, tmp_filename):
        # 创建多维数组
        x = np.arange(24, dtype="i4").reshape(2, 3, 4)
        # 将数组以二进制形式写入临时文件
        x.tofile(tmp_filename)
        # 从文件中读取数据，并使用 assert_array_equal 断言验证其与原始数组的一致性
        res = np.fromfile(tmp_filename, dtype="(3,4)i4")
        assert_array_equal(x, res)

        # 将数组转换为字节流
        x_str = x.tobytes()
        with assert_warns(DeprecationWarning):
            # 使用 fromstring 方法从字节流中读取数据，此方法已被标记为弃用
            res = np.fromstring(x_str, dtype="(3,4)i4")
            assert_array_equal(x, res)
    # 测试不支持解析子数组数据类型的情况
    def test_parsing_subarray_unsupported(self, tmp_filename):
        # 准备一个重复数据字符串
        data = "12,42,13," * 50
        # 使用 np.fromstring 尝试解析子数组数据，期望抛出 ValueError 异常
        with pytest.raises(ValueError):
            expected = np.fromstring(data, dtype="(3,)i", sep=",")

        # 将数据写入临时文件
        with open(tmp_filename, "w") as f:
            f.write(data)

        # 使用 np.fromfile 从文件中读取数据，期望抛出 ValueError 异常
        with pytest.raises(ValueError):
            np.fromfile(tmp_filename, dtype="(3,)i", sep=",")

    # 测试当请求的值数量超过实际值数量时，不会导致问题
    # 尤其是在子数组维度被合并到数组维度时
    def test_read_shorter_than_count_subarray(self, tmp_filename):
        # 准备一个预期的数组，形状为 (5110,)
        expected = np.arange(511 * 10, dtype="i").reshape(-1, 10)

        # 将预期数组转换为二进制数据
        binary = expected.tobytes()
        # 使用 np.fromstring 尝试解析子数组数据，期望抛出 ValueError 异常
        with pytest.raises(ValueError):
            with pytest.warns(DeprecationWarning):
                np.fromstring(binary, dtype="(10,)i", count=10000)

        # 将预期数组写入临时文件
        expected.tofile(tmp_filename)
        # 使用 np.fromfile 从文件中读取数据，预期结果与预期数组相等
        res = np.fromfile(tmp_filename, dtype="(10,)i", count=10000)
        assert_array_equal(res, expected)
@xpassIfTorchDynamo  # (reason="TODO")
@instantiate_parametrized_tests
class TestFromBuffer(TestCase):
    @parametrize(
        "byteorder", [subtest("little", name="little"), subtest("big", name="big")]
    )
    @parametrize("dtype", [float, int, complex])
    def test_basic(self, byteorder, dtype):
        # 创建指定字节顺序和数据类型的 NumPy 数据类型对象
        dt = np.dtype(dtype).newbyteorder(byteorder)
        # 创建一个形状为 (4, 7) 的随机数组，类型为 dt
        x = (np.random.random((4, 7)) * 5).astype(dt)
        # 将数组转换为字节流
        buf = x.tobytes()
        # 检查从字节流中读取的数据是否与原始数组 x 的扁平化版本相等
        assert_array_equal(np.frombuffer(buf, dtype=dt), x.flat)

    #    @xpassIfTorchDynamo
    @parametrize(
        "obj", [np.arange(10), subtest("12345678", decorators=[xfailIfTorchDynamo])]
    )
    def test_array_base(self, obj):
        # 对象（包括 NumPy 数组），如果不使用 `release_buffer` 插槽，则应直接用作基对象。
        # 参见 gh-21612
        if isinstance(obj, str):
            # @parametrize 在字节对象上存在问题
            # 使用 latin-1 编码将字符串转换为字节对象
            obj = bytes(obj, encoding="latin-1")
        # 从对象中创建新的 NumPy 数组
        new = np.frombuffer(obj)
        # 断言新数组的基础对象与原始对象相同
        assert new.base is obj

    def test_empty(self):
        # 检查从空字节流创建的数组是否为空数组
        assert_array_equal(np.frombuffer(b""), np.array([]))

    @skip("fails on CI, we are unlikely to implement this")
    @skipif(
        IS_PYPY,
        reason="PyPy's memoryview currently does not track exports. See: "
        "https://foss.heptapod.net/pypy/pypy/-/issues/3724",
    )
    def test_mmap_close(self):
        # 旧的缓冲区协议对于新协议支持的某些情况不安全。但 `frombuffer` 长期以来始终使用旧协议。
        # 使用 memoryviews 检查 `frombuffer` 在新协议下的安全性
        with tempfile.TemporaryFile(mode="wb") as tmp:
            tmp.write(b"asdf")
            tmp.flush()
            # 创建内存映射对象
            mm = mmap.mmap(tmp.fileno(), 0)
            # 使用 mm 创建一个无符号 8 位整数数组
            arr = np.frombuffer(mm, dtype=np.uint8)
            # 使用数组时不能关闭内存映射对象，应该抛出 BufferError
            with pytest.raises(BufferError):
                mm.close()  # cannot close while array uses the buffer
            del arr
            mm.close()


@skip  # (reason="TODO")   # FIXME: skip -> xfail (a0.shape = (4, 5) raises)
class TestFlat(TestCase):
    def setUp(self):
        # 创建一个包含 20 个连续浮点数的 NumPy 数组，并将其形状重塑为 (4, 5)
        a0 = np.arange(20.0)
        a = a0.reshape(4, 5)
        # 将 a0 的形状设置为 (4, 5)，并设置其不可写
        a0.shape = (4, 5)
        a.flags.writeable = False
        # 设置测试类的属性
        self.a = a
        self.b = a[::2, ::2]
        self.a0 = a0
        self.b0 = a0[::2, ::2]

    def test_contiguous(self):
        testpassed = False
        try:
            # 尝试修改 self.a 中扁平化后索引为 12 的元素
            self.a.flat[12] = 100.0
        except ValueError:
            testpassed = True
        # 断言修改操作抛出 ValueError
        assert_(testpassed)
        # 断言 self.a 中扁平化后索引为 12 的元素是否为 12.0
        assert_(self.a.flat[12] == 12.0)

    def test_discontiguous(self):
        testpassed = False
        try:
            # 尝试修改 self.b 中扁平化后索引为 4 的元素
            self.b.flat[4] = 100.0
        except ValueError:
            testpassed = True
        # 断言修改操作抛出 ValueError
        assert_(testpassed)
        # 断言 self.b 中扁平化后索引为 4 的元素是否为 12.0
        assert_(self.b.flat[4] == 12.0)
    # 测试 __array__ 方法的返回值，并进行断言检查
    def test___array__(self):
        # 获取 self.a 的 flat 属性的 __array__ 方法返回值
        c = self.a.flat.__array__()
        # 获取 self.b 的 flat 属性的 __array__ 方法返回值
        d = self.b.flat.__array__()
        # 获取 self.a0 的 flat 属性的 __array__ 方法返回值
        e = self.a0.flat.__array__()
        # 获取 self.b0 的 flat 属性的 __array__ 方法返回值
        f = self.b0.flat.__array__()

        # 断言 c 的 flags.writeable 属性为 False
        assert_(c.flags.writeable is False)
        # 断言 d 的 flags.writeable 属性为 False
        assert_(d.flags.writeable is False)
        # 断言 e 的 flags.writeable 属性为 True
        assert_(e.flags.writeable is True)
        # 断言 f 的 flags.writeable 属性为 False
        assert_(f.flags.writeable is False)
        
        # 断言 c 的 flags.writebackifcopy 属性为 False
        assert_(c.flags.writebackifcopy is False)
        # 断言 d 的 flags.writebackifcopy 属性为 False
        assert_(d.flags.writebackifcopy is False)
        # 断言 e 的 flags.writebackifcopy 属性为 False
        assert_(e.flags.writebackifcopy is False)
        # 断言 f 的 flags.writebackifcopy 属性为 False
        assert_(f.flags.writebackifcopy is False)

    @skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    # 测试引用计数相关功能，同时检查 gh-13165 中的引用计数错误回归
    def test_refcount(self):
        # 定义不同的索引类型和引用计数类型
        inds = [np.intp(0), np.array([True] * self.a.size), np.array([0]), None]
        indtype = np.dtype(np.intp)
        # 获取 indtype 的引用计数
        rc_indtype = sys.getrefcount(indtype)
        
        # 遍历不同的索引类型
        for ind in inds:
            # 获取当前索引 ind 的引用计数
            rc_ind = sys.getrefcount(ind)
            # 尝试执行多次索引操作，捕获 IndexError 异常
            for _ in range(100):
                try:
                    self.a.flat[ind]
                except IndexError:
                    pass
            # 断言索引 ind 的引用计数变化在可接受范围内
            assert_(abs(sys.getrefcount(ind) - rc_ind) < 50)
            # 断言引用计数类型 indtype 的变化在可接受范围内
            assert_(abs(sys.getrefcount(indtype) - rc_indtype) < 50)

    # 测试 flat 迭代器的 index 属性的设置行为
    def test_index_getset(self):
        # 获取 self.a 的 flat 属性的迭代器对象 it
        it = np.arange(10).reshape(2, 1, 5).flat
        # 使用 pytest 检查设置 index 属性是否会引发 AttributeError 异常
        with pytest.raises(AttributeError):
            it.index = 10
        
        # 遍历 flat 迭代器 it
        for _ in it:
            pass
        # 检查 .index 属性的值是否正确更新，特别是在 big-endian 机器上的兼容性问题（见 gh-19153）
        assert it.index == it.base.size
class TestResize(TestCase):
    # 定义测试类 TestResize，继承自 TestCase 类

    @_no_tracing
    def test_basic(self):
        # 定义测试方法 test_basic，禁止追踪（decorator）
        
        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 创建一个 3x3 的 NumPy 数组 x

        if IS_PYPY:
            # 如果是在 PyPy 环境下
            x.resize((5, 5), refcheck=False)
            # 调整数组 x 的大小为 (5, 5)，关闭引用检查
        else:
            # 否则
            x.resize((5, 5))
            # 调整数组 x 的大小为 (5, 5)
        
        assert_array_equal(
            x.ravel()[:9], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).ravel()
        )
        # 断言：展开 x 的前9个元素与给定的数组的展开形式相等

        assert_array_equal(x[9:].ravel(), 0)
        # 断言：x 的第9个元素之后的所有元素展开后应该全为0

    @skip(reason="how to find if someone is refencing an array")
    # 跳过该测试方法，理由是如何判断某人是否正在引用一个数组

    def test_check_reference(self):
        # 定义测试方法 test_check_reference

        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 创建一个 3x3 的 NumPy 数组 x
        y = x
        # 将 x 赋值给 y，即 y 和 x 指向同一个数组对象

        assert_raises(ValueError, x.resize, (5, 1))
        # 断言：调整 x 的大小为 (5, 1) 时会抛出 ValueError 异常

        del y  # 避免 pyflakes 报未使用变量的警告

    @_no_tracing
    def test_int_shape(self):
        # 定义测试方法 test_int_shape，禁止追踪（decorator）

        x = np.eye(3)
        # 创建一个 3x3 的单位矩阵 x

        if IS_PYPY:
            # 如果是在 PyPy 环境下
            x.resize(3, refcheck=False)
            # 调整数组 x 的大小为 (3, 3)，关闭引用检查
        else:
            # 否则
            x.resize(3)
            # 调整数组 x 的大小为 (3, 3)

        assert_array_equal(x, np.eye(3)[0, :])
        # 断言：数组 x 应该与单位矩阵的第一行相等

    def test_none_shape(self):
        # 定义测试方法 test_none_shape

        x = np.eye(3)
        # 创建一个 3x3 的单位矩阵 x

        x.resize(None)
        # 调整数组 x 的大小为 None，即不改变大小

        assert_array_equal(x, np.eye(3))
        # 断言：数组 x 应该与单位矩阵相等

        x.resize()
        # 调整数组 x 的大小为默认大小，即不改变大小

        assert_array_equal(x, np.eye(3))
        # 断言：数组 x 应该与单位矩阵相等

    def test_0d_shape(self):
        # 定义测试方法 test_0d_shape

        # 多次测试以确保不破坏分配缓存 gh-9216
        for i in range(10):
            # 循环10次

            x = np.empty((1,))
            # 创建一个空的形状为 (1,) 的数组 x

            x.resize(())
            # 调整数组 x 的大小为 ()

            assert_equal(x.shape, ())
            # 断言：数组 x 的形状应该是 ()

            assert_equal(x.size, 1)
            # 断言：数组 x 的大小应该是 1

            x = np.empty(())
            # 创建一个空的形状为 () 的数组 x

            x.resize((1,))
            # 调整数组 x 的大小为 (1,)

            assert_equal(x.shape, (1,))
            # 断言：数组 x 的形状应该是 (1,)

            assert_equal(x.size, 1)
            # 断言：数组 x 的大小应该是 1

    def test_invalid_arguments(self):
        # 定义测试方法 test_invalid_arguments

        assert_raises(TypeError, np.eye(3).resize, "hi")
        # 断言：调用 np.eye(3).resize("hi") 应该抛出 TypeError 异常

        assert_raises(ValueError, np.eye(3).resize, -1)
        # 断言：调用 np.eye(3).resize(-1) 应该抛出 ValueError 异常

        assert_raises(TypeError, np.eye(3).resize, order=1)
        # 断言：调用 np.eye(3).resize(order=1) 应该抛出 TypeError 异常

        assert_raises((NotImplementedError, TypeError), np.eye(3).resize, refcheck="hi")
        # 断言：调用 np.eye(3).resize(refcheck="hi") 应该抛出 NotImplementedError 或 TypeError 异常

    @_no_tracing
    def test_freeform_shape(self):
        # 定义测试方法 test_freeform_shape，禁止追踪（decorator）

        x = np.eye(3)
        # 创建一个 3x3 的单位矩阵 x

        if IS_PYPY:
            # 如果是在 PyPy 环境下
            x.resize(3, 2, 1, refcheck=False)
            # 调整数组 x 的大小为 (3, 2, 1)，关闭引用检查
        else:
            # 否则
            x.resize(3, 2, 1)
            # 调整数组 x 的大小为 (3, 2, 1)

        assert_(x.shape == (3, 2, 1))
        # 断言：数组 x 的形状应该是 (3, 2, 1)

    @_no_tracing
    def test_zeros_appended(self):
        # 定义测试方法 test_zeros_appended，禁止追踪（decorator）

        x = np.eye(3)
        # 创建一个 3x3 的单位矩阵 x

        if IS_PYPY:
            # 如果是在 PyPy 环境下
            x.resize(2, 3, 3, refcheck=False)
            # 调整数组 x 的大小为 (2, 3, 3)，关闭引用检查
        else:
            # 否则
            x.resize(2, 3, 3)
            # 调整数组 x 的大小为 (2, 3, 3)

        assert_array_equal(x[0], np.eye(3))
        # 断言：数组 x 的第一个元素应该是单位矩阵

        assert_array_equal(x[1], np.zeros((3, 3)))
        # 断言：数组 x 的第二个元素应该是一个全为0的 3x3 数组

    def test_empty_view(self):
        # 定义测试方法 test_empty_view

        # 检查包含零的大小不会触发重新分配已经为空的数组

        x = np.zeros((10, 0), int)
        # 创建一个形状为 (10, 0) 的 int 类型全为零的数组 x

        x_view = x[...]
        # 创建 x 的视图 x_view

        x_view.resize((0, 10))
        # 调整 x_view 的大小为 (0, 10)

        x_view.resize((0, 100))
        # 调整 x_view 的大小为 (0, 100)

    @skip(reason="ignore weakrefs for ndarray.resize")
    # 跳过该测试方法，理由是在 ndarray.resize 中忽略 weakrefs

    def test_check_weakref(self):
        # 定义测试方法 test_check_weakref

        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 创建一个 3x3 的 NumPy 数组 x

        xref = weakref.ref(x)
        # 创建 x 的弱引用 xref

    # 定义一个测试方法，用于测试从数据类型到数据类型的映射
    def test_dtype_from_dtype(self):
        # 创建一个 3x3 的单位矩阵
        mat = np.eye(3)

        # 对于整数类型的统计
        # FIXME: 
        # 这需要定义，因为可能会沿着很多地方进行类型转换。

        # 对于每个测试函数
        # for f in self.funcs:
        #     对于所有整数类型的代码字符
        #     for c in np.typecodes['AllInteger']:
        #         目标类型为 numpy 中的 c 类型
        #         tgt = np.dtype(c).type
        #         调用函数 f 处理矩阵 mat，在 axis=1 时指定数据类型为 c，获取其返回的数据类型并获取其类型
        #         res = f(mat, axis=1, dtype=c).dtype.type
        #         断言 res 是 tgt
        #         assert_(res is tgt)
        #         # 标量情况
        #         调用函数 f 处理矩阵 mat，在 axis=None 时指定数据类型为 c，获取其返回的数据类型并获取其类型
        #         res = f(mat, axis=None, dtype=c).dtype.type
        #         断言 res 是 tgt

        # 对于浮点类型的统计
        for f in self.funcs:
            # 对于所有浮点类型的代码字符
            for c in np.typecodes["AllFloat"]:
                # 目标类型为 numpy 中的 c 类型
                tgt = np.dtype(c).type
                # 调用函数 f 处理矩阵 mat，在 axis=1 时指定数据类型为 c，获取其返回的数据类型并获取其类型
                res = f(mat, axis=1, dtype=c).dtype.type
                # 断言 res 是 tgt
                assert_(res is tgt)
                # 标量情况
                # 调用函数 f 处理矩阵 mat，在 axis=None 时指定数据类型为 c，获取其返回的数据类型并获取其类型
                res = f(mat, axis=None, dtype=c).dtype.type
                # 断言 res 是 tgt

    # 定义一个测试方法，用于测试自由度修正参数 ddof
    def test_ddof(self):
        # 对于每个函数 f
        for f in [_var]:
            # 对于 ddof 从 0 到 2 的范围
            for ddof in range(3):
                # 计算矩阵 self.rmat 沿 axis=1 的结果乘以维度 dim
                dim = self.rmat.shape[1]
                tgt = f(self.rmat, axis=1)
    # 测试计算均值函数的正确性
    def test_mean_values(self):
        # 对于每个矩阵（行矩阵和列矩阵）
        for mat in [self.rmat, self.cmat]:
            # 对于每个轴（0表示列，1表示行）
            for axis in [0, 1]:
                # 目标值是在指定轴上求和
                tgt = mat.sum(axis=axis)
                # 结果是调用_mean函数计算的平均值乘以矩阵轴的大小
                res = _mean(mat, axis=axis) * mat.shape[axis]
                # 断言结果与目标值几乎相等
                assert_almost_equal(res, tgt)
            # 对于未指定轴的情况
            for axis in [None]:
                # 目标值是在所有轴上求和
                tgt = mat.sum(axis=axis)
                # 结果是调用_mean函数计算的平均值乘以矩阵所有维度的乘积
                res = _mean(mat, axis=axis) * np.prod(mat.shape)
                # 断言结果与目标值几乎相等
                assert_almost_equal(res, tgt)

    # 测试在float16数据类型下计算均值的准确性
    def test_mean_float16(self):
        # 如果在_mean函数内部的求和使用float16而不是float32，则会失败
        assert_(_mean(np.ones(100000, dtype="float16")) == 1)

    # 测试当轴超出范围时，确保引发AxisError而不是IndexError
    def test_mean_axis_error(self):
        with assert_raises(np.AxisError):
            # 当轴超出范围时，调用np.arange(10).mean(axis=2)应该引发AxisError，而不是IndexError
            np.arange(10).mean(axis=2)

    @xpassIfTorchDynamo  # (reason="implement mean(..., where=...)")
    # 测试带有where参数的均值函数的行为
    def test_mean_where(self):
        a = np.arange(16).reshape((4, 4))
        wh_full = np.array(
            [
                [False, True, False, True],
                [True, False, True, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
        wh_partial = np.array([[False], [True], [True], [False]])
        _cases = [
            (1, True, [1.5, 5.5, 9.5, 13.5]),
            (0, wh_full, [6.0, 5.0, 10.0, 9.0]),
            (1, wh_full, [2.0, 5.0, 8.5, 14.5]),
            (0, wh_partial, [6.0, 7.0, 8.0, 9.0]),
        ]
        for _ax, _wh, _res in _cases:
            # 断言调用mean函数时，使用where参数的结果与预期结果相等
            assert_allclose(a.mean(axis=_ax, where=_wh), np.array(_res))
            # 断言调用np.mean函数时，使用where参数的结果与预期结果相等
            assert_allclose(np.mean(a, axis=_ax, where=_wh), np.array(_res))

        a3d = np.arange(16).reshape((2, 2, 4))
        _wh_partial = np.array([False, True, True, False])
        _res = [[1.5, 5.5], [9.5, 13.5]]
        # 断言调用mean函数处理三维数组时，使用where参数的结果与预期结果相等
        assert_allclose(a3d.mean(axis=2, where=_wh_partial), np.array(_res))
        # 断言调用np.mean处理三维数组时，使用where参数的结果与预期结果相等
        assert_allclose(np.mean(a3d, axis=2, where=_wh_partial), np.array(_res))

        with pytest.warns(RuntimeWarning) as w:
            # 断言在使用where参数时，平均函数对行矩阵a的行进行平均时会引发警告
            assert_allclose(
                a.mean(axis=1, where=wh_partial), np.array([np.nan, 5.5, 9.5, np.nan])
            )
        with pytest.warns(RuntimeWarning) as w:
            # 断言调用mean函数时，对矩阵a使用where参数为False时的结果为NaN
            assert_equal(a.mean(where=False), np.nan)
        with pytest.warns(RuntimeWarning) as w:
            # 断言调用np.mean函数时，对矩阵a使用where参数为False时的结果为NaN
            assert_equal(np.mean(a, where=False), np.nan)

    # 测试方差函数的正确性
    def test_var_values(self):
        # 对于每个矩阵（行矩阵和列矩阵）
        for mat in [self.rmat, self.cmat]:
            # 对于每个轴（0表示列，1表示行，None表示所有轴）
            for axis in [0, 1, None]:
                # 计算平方平均值
                msqr = _mean(mat * mat.conj(), axis=axis)
                # 计算平均值
                mean = _mean(mat, axis=axis)
                # 目标值是平方平均值减去平均值乘以共轭的平均值
                tgt = msqr - mean * mean.conjugate()
                # 结果是调用_var函数计算的方差
                res = _var(mat, axis=axis)
                # 断言结果与目标值几乎相等
                assert_almost_equal(res, tgt)

    @parametrize(
        "complex_dtype, ndec",
        (
            ("complex64", 6),
            ("complex128", 7),
        ),
    )
    # 定义一个测试函数，用于测试复杂变量类型的方差计算，ndec 参数指定精度
    def test_var_complex_values(self, complex_dtype, ndec):
        # 对每种内置复杂类型进行快速路径测试
        for axis in [0, 1, None]:
            # 复制并转换为指定复杂类型的矩阵
            mat = self.cmat.copy().astype(complex_dtype)
            # 计算平方平均值，支持指定轴向
            msqr = _mean(mat * mat.conj(), axis=axis)
            # 计算平均值，支持指定轴向
            mean = _mean(mat, axis=axis)
            # 计算方差的目标值
            tgt = msqr - mean * mean.conjugate()
            # 计算方差并进行断言检查
            res = _var(mat, axis=axis)
            assert_almost_equal(res, tgt, decimal=ndec)

    # 定义一个测试函数，验证复杂数维度引入的附加视图是否增加了维度
    def test_var_dimensions(self):
        # 创建三维矩阵，每个维度都是 self.cmat 的复制
        mat = np.stack([self.cmat] * 3)
        for axis in [0, 1, 2, -1, None]:
            # 计算平方平均值，支持指定轴向
            msqr = _mean(mat * mat.conj(), axis=axis)
            # 计算平均值，支持指定轴向
            mean = _mean(mat, axis=axis)
            # 计算方差的目标值
            tgt = msqr - mean * mean.conjugate()
            # 计算方差并进行断言检查
            res = _var(mat, axis=axis)
            assert_almost_equal(res, tgt)

    # 跳过测试，原因是复杂数组具有非本机字节顺序不会导致失败
    @skip(reason="endianness")
    def test_var_complex_byteorder(self):
        # 测试 var 快速路径在具有非本机字节顺序的复杂数组上不会导致失败
        cmat = self.cmat.copy().astype("complex128")
        cmat_swapped = cmat.astype(cmat.dtype.newbyteorder())
        assert_almost_equal(cmat.var(), cmat_swapped.var())

    # 定义一个测试函数，确保当轴超出范围时引发 AxisError 而不是 IndexError
    def test_var_axis_error(self):
        with assert_raises(np.AxisError):
            # 在轴超出范围时调用 np.arange(10).var(axis=2)，预期引发 AxisError
            np.arange(10).var(axis=2)

    # 标记为跳过测试，原因是尚未实现 var 函数的 where 参数支持
    @xpassIfTorchDynamo  # (reason="implement var(..., where=...)")
    # 测试函数，用于验证 numpy 的方差计算在指定条件下的行为
    def test_var_where(self):
        # 创建一个 5x5 的数组，包含 0 到 24 的整数，然后将其重塑为 5x5 的矩阵
        a = np.arange(25).reshape((5, 5))
        # 创建一个指示在方差计算中应用条件的布尔数组，形状为 5x5
        wh_full = np.array(
            [
                [False, True, False, True, True],
                [True, False, True, True, False],
                [True, True, False, False, True],
                [False, True, True, False, True],
                [True, False, True, True, False],
            ]
        )
        # 创建一个部分指示条件的布尔数组，形状为 5x1
        wh_partial = np.array([[False], [True], [True], [False], [True]])
        # 定义多个测试用例，每个用例包含 axis、where 和预期结果列表
        _cases = [
            (0, True, [50.0, 50.0, 50.0, 50.0, 50.0]),
            (1, True, [2.0, 2.0, 2.0, 2.0, 2.0]),
        ]
        # 遍历每个测试用例，验证 numpy 方差计算和 np.var() 函数的结果
        for _ax, _wh, _res in _cases:
            assert_allclose(a.var(axis=_ax, where=_wh), np.array(_res))
            assert_allclose(np.var(a, axis=_ax, where=_wh), np.array(_res))

        # 创建一个 3D 数组，包含 0 到 15 的整数，然后将其重塑为 2x2x4 的矩阵
        a3d = np.arange(16).reshape((2, 2, 4))
        # 创建一个部分指示条件的布尔数组，形状为 4x1
        _wh_partial = np.array([False, True, True, False])
        # 定义预期的结果矩阵
        _res = [[0.25, 0.25], [0.25, 0.25]]
        # 验证在 3D 数组上应用条件时 numpy 方差计算和 np.var() 函数的结果
        assert_allclose(a3d.var(axis=2, where=_wh_partial), np.array(_res))
        assert_allclose(np.var(a3d, axis=2, where=_wh_partial), np.array(_res))

        # 验证在矩阵的特定轴上应用条件时 numpy 方差计算和 np.var() 函数的结果
        assert_allclose(
            np.var(a, axis=1, where=wh_full), np.var(a[wh_full].reshape((5, 3)), axis=1)
        )
        assert_allclose(
            np.var(a, axis=0, where=wh_partial), np.var(a[wh_partial[:, 0]], axis=0)
        )

        # 验证当条件完全为 False 时，方差计算会引发 RuntimeWarning 并返回 NaN
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.var(where=False), np.nan)
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.var(a, where=False), np.nan)

    # 测试函数，用于验证 numpy 标准差计算的结果
    def test_std_values(self):
        # 遍历测试矩阵（self.rmat 和 self.cmat）和轴向（0、1 和 None），验证标准差计算的准确性
        for mat in [self.rmat, self.cmat]:
            for axis in [0, 1, None]:
                tgt = np.sqrt(_var(mat, axis=axis))
                res = _std(mat, axis=axis)
                assert_almost_equal(res, tgt)

    # 装饰器函数，用于跳过 TorchDynamo，目前未实现 std(..., where=...)
    @xpassIfTorchDynamo  # (reason="implement std(..., where=...)")
    # 定义测试函数 test_std_where，用于测试 NumPy 的标准差计算功能
    def test_std_where(self):
        # 创建一个 5x5 的数组 a，其元素为 0 到 24，然后将其倒序排列
        a = np.arange(25).reshape((5, 5))[::-1]
        
        # 定义布尔型索引数组 whf，用于指定在哪些位置计算标准差
        whf = np.array(
            [
                [False, True, False, True, True],
                [True, False, True, False, True],
                [True, True, False, True, False],
                [True, False, True, True, False],
                [False, True, False, True, True],
            ]
        )
        
        # 定义布尔型索引数组 whp，用于指定在哪些位置计算标准差
        whp = np.array([[False], [False], [True], [True], [False]])
        
        # 定义测试用例 _cases，包含不同的参数组合及预期结果
        _cases = [
            (0, True, 7.07106781 * np.ones(5)),
            (1, True, 1.41421356 * np.ones(5)),
            (0, whf, np.array([4.0824829, 8.16496581, 5.0, 7.39509973, 8.49836586])),
            (0, whp, 2.5 * np.ones(5)),
        ]
        
        # 遍历测试用例进行验证
        for _ax, _wh, _res in _cases:
            # 使用 assert_allclose 检查数组 a 按指定轴和条件计算的标准差是否与预期结果一致
            assert_allclose(a.std(axis=_ax, where=_wh), _res)
            # 使用 assert_allclose 检查使用 np.std 函数计算的标准差是否与预期结果一致
            assert_allclose(np.std(a, axis=_ax, where=_wh), _res)
        
        # 创建一个 3D 数组 a3d，形状为 (2, 2, 4)，元素为 0 到 15
        a3d = np.arange(16).reshape((2, 2, 4))
        
        # 定义布尔型索引数组 _wh_partial，用于指定在哪些位置计算标准差
        _wh_partial = np.array([False, True, True, False])
        
        # 定义预期结果 _res，表示在指定条件下按轴 2 计算得到的标准差
        _res = [[0.5, 0.5], [0.5, 0.5]]
        
        # 使用 assert_allclose 检查数组 a3d 按指定轴和条件计算的标准差是否与预期结果一致
        assert_allclose(a3d.std(axis=2, where=_wh_partial), np.array(_res))
        # 使用 assert_allclose 检查使用 np.std 函数计算的标准差是否与预期结果一致
        assert_allclose(np.std(a3d, axis=2, where=_wh_partial), np.array(_res))
        
        # 使用 assert_allclose 检查数组 a 按指定轴和条件计算的标准差是否与预期结果一致
        assert_allclose(
            a.std(axis=1, where=whf), np.std(a[whf].reshape((5, 3)), axis=1)
        )
        # 使用 assert_allclose 检查使用 np.std 函数计算的标准差是否与预期结果一致
        assert_allclose(
            np.std(a, axis=1, where=whf), (a[whf].reshape((5, 3))).std(axis=1)
        )
        
        # 使用 assert_allclose 检查数组 a 按指定轴和条件计算的标准差是否与预期结果一致
        assert_allclose(a.std(axis=0, where=whp), np.std(a[whp[:, 0]], axis=0))
        # 使用 assert_allclose 检查使用 np.std 函数计算的标准差是否与预期结果一致
        assert_allclose(np.std(a, axis=0, where=whp), (a[whp[:, 0]]).std(axis=0))
        
        # 使用 pytest.warns 检查运行时警告，确保调用 a.std(where=False) 时返回 np.nan
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.std(where=False), np.nan)
        
        # 使用 pytest.warns 检查运行时警告，确保调用 np.std(a, where=False) 时返回 np.nan
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.std(a, where=False), np.nan)
class TestVdot(TestCase):
    def test_basic(self):
        # 定义数值类型和复数类型的数据类型码
        dt_numeric = np.typecodes["AllFloat"] + np.typecodes["AllInteger"]
        dt_complex = np.typecodes["Complex"]

        # test real
        # 创建一个单位矩阵
        a = np.eye(3)
        # 遍历数值类型的数据类型码
        for dt in dt_numeric:
            # 将矩阵 a 转换为指定的数据类型 dt
            b = a.astype(dt)
            # 计算向量的内积
            res = np.vdot(b, b)
            # 断言返回值是标量
            assert_(np.isscalar(res))
            # 断言计算的内积值为 3
            assert_equal(np.vdot(b, b), 3)

        # test complex
        # 创建一个虚数单位矩阵
        a = np.eye(3) * 1j
        # 遍历复数类型的数据类型码
        for dt in dt_complex:
            # 将矩阵 a 转换为指定的数据类型 dt
            b = a.astype(dt)
            # 计算向量的内积
            res = np.vdot(b, b)
            # 断言返回值是标量
            assert_(np.isscalar(res))
            # 断言计算的内积值为 3
            assert_equal(np.vdot(b, b), 3)

        # test boolean
        # 创建一个布尔类型的单位矩阵
        b = np.eye(3, dtype=bool)
        # 计算向量的内积
        res = np.vdot(b, b)
        # 断言返回值是标量
        assert_(np.isscalar(res))
        # 断言计算的内积值为 True
        assert_equal(np.vdot(b, b), True)

    @xpassIfTorchDynamo  # (reason="implement order='F'")
    def test_vdot_array_order(self):
        # 创建两个不同存储顺序的数组
        a = np.array([[1, 2], [3, 4]], order="C")
        b = np.array([[1, 2], [3, 4]], order="F")
        # 计算向量的内积
        res = np.vdot(a, a)

        # 断言整数数组的精确性
        assert_equal(np.vdot(a, b), res)
        assert_equal(np.vdot(b, a), res)
        assert_equal(np.vdot(b, b), res)

    def test_vdot_uncontiguous(self):
        # 对不同大小的数组进行测试
        for size in [2, 1000]:
            # 创建多维零数组，并设置部分元素值
            a = np.zeros((size, 2, 2))
            b = np.zeros((size, 2, 2))
            a[:, 0, 0] = np.arange(size)
            b[:, 0, 0] = np.arange(size) + 1
            # 使数组 a 和 b 不连续
            a = a[..., 0]
            b = b[..., 0]

            # 断言不连续数组的内积等于将它们展平后的内积
            assert_equal(np.vdot(a, b), np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a, b.copy()), np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a.copy(), b), np.vdot(a.flatten(), b.flatten()))

    @xpassIfTorchDynamo  # (reason="implement order='F'")
    def test_vdot_uncontiguous_2(self):
        # 分别测试 order='F'
        for size in [2, 1000]:
            # 创建多维零数组，并设置部分元素值
            a = np.zeros((size, 2, 2))
            b = np.zeros((size, 2, 2))
            a[:, 0, 0] = np.arange(size)
            b[:, 0, 0] = np.arange(size) + 1
            # 使数组 a 和 b 不连续
            a = a[..., 0]
            b = b[..., 0]

            # 断言使用 order='F' 的不连续数组的内积等于将它们展平后的内积
            assert_equal(np.vdot(a.copy("F"), b), np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a, b.copy("F")), np.vdot(a.flatten(), b.flatten()))
    def setUp(self):
        # 设置随机数种子以确保可重复性
        np.random.seed(128)

        # 定义矩阵 A
        # 因为 Numpy 和 PyTorch 的随机数流不同，所以直接提供 numpy 1.24.1 中的值
        self.A = np.array(
            [
                [0.86663704, 0.26314485],
                [0.13140848, 0.04159344],
                [0.23892433, 0.6454746],
                [0.79059935, 0.60144244],
            ]
        )

        # 定义向量 b1
        self.b1 = np.array([[0.33429937], [0.11942846]])

        # 定义向量 b2
        self.b2 = np.array([0.30913305, 0.10972379])

        # 定义向量 b3
        self.b3 = np.array([[0.60211331, 0.25128496]])

        # 定义向量 b4
        self.b4 = np.array([0.29968129, 0.517116, 0.71520252, 0.9314494])

        # 设置精度参数 N
        self.N = 7

    def test_dotmatmat(self):
        # 测试矩阵乘法：A^T * A
        A = self.A
        res = np.dot(A.transpose(), A)
        tgt = np.array([[1.45046013, 0.86323640], [0.86323640, 0.84934569]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotmatvec(self):
        # 测试矩阵和向量乘法：A * b1
        A, b1 = self.A, self.b1
        res = np.dot(A, b1)
        tgt = np.array([[0.32114320], [0.04889721], [0.15696029], [0.33612621]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotmatvec2(self):
        # 测试矩阵和向量乘法：A * b2
        A, b2 = self.A, self.b2
        res = np.dot(A, b2)
        tgt = np.array([0.29677940, 0.04518649, 0.14468333, 0.31039293])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat(self):
        # 测试向量和矩阵乘法：b4 * A
        A, b4 = self.A, self.b4
        res = np.dot(b4, A)
        tgt = np.array([1.23495091, 1.12222648])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat2(self):
        # 测试向量和矩阵乘法：b3 * A^T
        b3, A = self.b3, self.A
        res = np.dot(b3, A.transpose())
        tgt = np.array([[0.58793804, 0.08957460, 0.30605758, 0.62716383]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecmat3(self):
        # 测试向量和矩阵乘法：A^T * b4
        A, b4 = self.A, self.b4
        res = np.dot(A.transpose(), b4)
        tgt = np.array([1.23495091, 1.12222648])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecvecouter(self):
        # 测试向量外积：b1 * b3
        b1, b3 = self.b1, self.b3
        res = np.dot(b1, b3)
        tgt = np.array([[0.20128610, 0.08400440], [0.07190947, 0.03001058]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotvecvecinner(self):
        # 测试向量内积：b3 * b1
        b1, b3 = self.b1, self.b3
        res = np.dot(b3, b1)
        tgt = np.array([[0.23129668]])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotcolumnvect1(self):
        # 测试列向量和标量乘法：b1 * [5.3]
        b1 = np.ones((3, 1))
        b2 = [5.3]
        res = np.dot(b1, b2)
        tgt = np.array([5.3, 5.3, 5.3])
        assert_almost_equal(res, tgt, decimal=self.N)

    def test_dotcolumnvect2(self):
        # 测试行向量和标量乘法：[6.2] * b1
        b1 = np.ones((3, 1)).transpose()
        b2 = [6.2]
        res = np.dot(b2, b1)
        tgt = np.array([6.2, 6.2, 6.2])
        assert_almost_equal(res, tgt, decimal=self.N)
    def test_dotvecscalar(self):
        np.random.seed(100)
        # 设置随机种子确保结果可重复
        # 使用 numpy 1.24.1 中的预设值，因为我们无法保证随机流与 numpy 相同
        b1 = np.array([[0.54340494]])  # 初始化矩阵 b1

        b2 = np.array([[0.27836939, 0.42451759, 0.84477613, 0.00471886]])  # 初始化矩阵 b2

        res = np.dot(b1, b2)  # 计算矩阵乘积
        tgt = np.array([[0.15126730, 0.23068496, 0.45905553, 0.00256425]])  # 期望的结果
        assert_almost_equal(res, tgt, decimal=self.N)  # 断言计算结果与期望结果的接近程度

    def test_dotvecscalar2(self):
        np.random.seed(100)
        b1 = np.array([[0.54340494], [0.27836939], [0.42451759], [0.84477613]])  # 初始化矩阵 b1

        b2 = np.array([[0.00471886]])  # 初始化矩阵 b2

        res = np.dot(b1, b2)  # 计算矩阵乘积
        tgt = np.array([[0.00256425], [0.00131359], [0.00200324], [0.00398638]])  # 期望的结果
        assert_almost_equal(res, tgt, decimal=self.N)  # 断言计算结果与期望结果的接近程度

    def test_all(self):
        dims = [(), (1,), (1, 1)]
        dout = [(), (1,), (1, 1), (1,), (), (1,), (1, 1), (1,), (1, 1)]
        for dim, (dim1, dim2) in zip(dout, itertools.product(dims, dims)):
            b1 = np.zeros(dim1)  # 初始化矩阵 b1
            b2 = np.zeros(dim2)  # 初始化矩阵 b2
            res = np.dot(b1, b2)  # 计算矩阵乘积
            tgt = np.zeros(dim)  # 期望的结果
            assert_(res.shape == tgt.shape)  # 断言结果的形状与期望的形状相同
            assert_almost_equal(res, tgt, decimal=self.N)  # 断言计算结果与期望结果的接近程度

    @skip(reason="numpy internals")
    def test_dot_2args(self):
        from numpy.core.multiarray import dot

        a = np.array([[1, 2], [3, 4]], dtype=float)  # 初始化矩阵 a
        b = np.array([[1, 0], [1, 1]], dtype=float)  # 初始化矩阵 b
        c = np.array([[3, 2], [7, 4]], dtype=float)  # 初始化矩阵 c

        d = dot(a, b)  # 使用 numpy 的 dot 函数计算矩阵乘积
        assert_allclose(c, d)  # 断言计算结果与期望结果的接近程度

    @skip(reason="numpy internals")
    def test_dot_3args(self):
        from numpy.core.multiarray import dot

        np.random.seed(22)
        f = np.random.random_sample((1024, 16))  # 初始化矩阵 f
        v = np.random.random_sample((16, 32))  # 初始化矩阵 v

        r = np.empty((1024, 32))  # 初始化矩阵 r
        for i in range(12):
            dot(f, v, r)  # 使用 numpy 的 dot 函数计算矩阵乘积，并将结果存入 r
        if HAS_REFCOUNT:
            assert_equal(sys.getrefcount(r), 2)  # 断言 r 的引用计数为 2
        r2 = dot(f, v, out=None)  # 使用 numpy 的 dot 函数计算矩阵乘积，不指定输出参数
        assert_array_equal(r2, r)  # 断言计算结果与预期结果 r 相同
        assert_(r is dot(f, v, out=r))  # 断言返回的结果与 r 相同

        v = v[:, 0].copy()  # 复制 v 的第一列
        r = r[:, 0].copy()  # 复制 r 的第一列
        r2 = dot(f, v)  # 使用 numpy 的 dot 函数计算矩阵乘积
        assert_(r is dot(f, v, r))  # 断言返回的结果与 r 相同
        assert_array_equal(r2, r)  # 断言计算结果与预期结果 r 相同

    @skip(reason="numpy internals")
    # 定义测试函数，用于测试 dot 函数在不同参数错误情况下的行为
    def test_dot_3args_errors(self):
        # 导入 numpy 的 dot 函数
        from numpy.core.multiarray import dot

        # 设定随机种子，确保结果可复现
        np.random.seed(22)
        
        # 创建两个随机数组，f 和 v
        f = np.random.random_sample((1024, 16))
        v = np.random.random_sample((16, 32))

        # 初始化一个空数组 r，形状为 (1024, 31)，预期会抛出 ValueError 异常
        r = np.empty((1024, 31))
        assert_raises(ValueError, dot, f, v, r)

        # 初始化一个空数组 r，形状为 (1024,)，预期会抛出 ValueError 异常
        r = np.empty((1024,))
        assert_raises(ValueError, dot, f, v, r)

        # 初始化一个空数组 r，形状为 (32,)，预期会抛出 ValueError 异常
        r = np.empty((32,))
        assert_raises(ValueError, dot, f, v, r)

        # 初始化一个空数组 r，形状为 (32, 1024)，预期会抛出 ValueError 异常
        r = np.empty((32, 1024))
        assert_raises(ValueError, dot, f, v, r)
        
        # 对 r 的转置进行 dot 运算，预期会抛出 ValueError 异常
        assert_raises(ValueError, dot, f, v, r.T)

        # 初始化一个空数组 r，形状为 (1024, 64)，预期会抛出 ValueError 异常
        r = np.empty((1024, 64))
        # 对 r 的每隔两列进行 dot 运算，预期会抛出 ValueError 异常
        assert_raises(ValueError, dot, f, v, r[:, ::2])
        # 对 r 的前 32 列进行 dot 运算，预期会抛出 ValueError 异常
        assert_raises(ValueError, dot, f, v, r[:, :32])

        # 初始化一个空数组 r，形状为 (1024, 32)，数据类型为 np.float32，预期会抛出 ValueError 异常
        r = np.empty((1024, 32), dtype=np.float32)
        assert_raises(ValueError, dot, f, v, r)

        # 初始化一个空数组 r，形状为 (1024, 32)，数据类型为 int，预期会抛出 ValueError 异常
        r = np.empty((1024, 32), dtype=int)
        assert_raises(ValueError, dot, f, v, r)

    # 标记测试函数，用于测试 dot 函数在数组顺序不同情况下的行为
    @xpassIfTorchDynamo  # (reason="TODO order='F'")
    def test_dot_array_order(self):
        # 创建两个二维数组 a 和 b，分别使用不同的存储顺序 'C' 和 'F'
        a = np.array([[1, 2], [3, 4]], order="C")
        b = np.array([[1, 2], [3, 4]], order="F")
        
        # 计算数组 a 与自身的点积，结果保存在 res 中
        res = np.dot(a, a)

        # 检查整数数组的精确性
        # 检验 a 和 b 的点积与 res 是否相等
        assert_equal(np.dot(a, b), res)
        assert_equal(np.dot(b, a), res)
        assert_equal(np.dot(b, b), res)

    # 标记跳过该测试函数，原因是 "TODO: nbytes, view, __array_interface__"
    @skip(reason="TODO: nbytes, view, __array_interface__")
    def test_accelerate_framework_sgemv_fix(self):
        # 定义一个测试函数，用于验证和修复SGEMV加速框架的问题

        def aligned_array(shape, align, dtype, order="C"):
            # 创建一个按照指定对齐方式对齐的数组
            d = dtype(0)
            N = np.prod(shape)
            tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
            address = tmp.__array_interface__["data"][0]
            # 确定数组在内存中的对齐偏移量
            for offset in range(align):
                if (address + offset) % align == 0:
                    break
            tmp = tmp[offset : offset + N * d.nbytes].view(dtype=dtype)
            return tmp.reshape(shape, order=order)

        def as_aligned(arr, align, dtype, order="C"):
            # 将给定数组按照指定的对齐方式对齐
            aligned = aligned_array(arr.shape, align, dtype, order)
            aligned[:] = arr[:]
            return aligned

        def assert_dot_close(A, X, desired):
            # 断言两个矩阵乘积的结果与期望值非常接近
            assert_allclose(np.dot(A, X), desired, rtol=1e-5, atol=1e-7)

        m = aligned_array(100, 15, np.float32)
        s = aligned_array((100, 100), 15, np.float32)
        np.dot(s, m)  # 如果存在错误，此处将总是段错误

        testdata = itertools.product((15, 32), (10000,), (200, 89), ("C", "F"))
        for align, m, n, a_order in testdata:
            # 在双精度中进行计算
            A_d = np.random.rand(m, n)
            X_d = np.random.rand(n)
            desired = np.dot(A_d, X_d)
            # 使用对齐的单精度进行计算
            A_f = as_aligned(A_d, align, np.float32, order=a_order)
            X_f = as_aligned(X_d, align, np.float32)
            assert_dot_close(A_f, X_f, desired)
            # A行按步长处理
            A_d_2 = A_d[::2]
            desired = np.dot(A_d_2, X_d)
            A_f_2 = A_f[::2]
            assert_dot_close(A_f_2, X_f, desired)
            # A列和X向量按步长处理
            A_d_22 = A_d_2[:, ::2]
            X_d_2 = X_d[::2]
            desired = np.dot(A_d_22, X_d_2)
            A_f_22 = A_f_2[:, ::2]
            X_f_2 = X_f[::2]
            assert_dot_close(A_f_22, X_f_2, desired)
            # 检查步长是否符合预期
            if a_order == "F":
                assert_equal(A_f_22.strides, (8, 8 * m))
            else:
                assert_equal(A_f_22.strides, (8 * n, 8))
            assert_equal(X_f_2.strides, (8,))
            # A行和列按步长处理，X只按列步长处理
            X_f_2c = as_aligned(X_f_2, align, np.float32)
            assert_dot_close(A_f_22, X_f_2c, desired)
            # 只有A列按步长处理
            A_d_12 = A_d[:, ::2]
            desired = np.dot(A_d_12, X_d_2)
            A_f_12 = A_f[:, ::2]
            assert_dot_close(A_f_12, X_f_2c, desired)
            # A列和X按步长处理
            assert_dot_close(A_f_12, X_f_2, desired)

    @slow
    @parametrize("dtype", [np.float64, np.complex128])
    @requires_memory(free_bytes=18e9)  # 复杂情况需要18GiB+
    # 定义测试函数，用于测试大向量点积
    def test_huge_vectordot(self, dtype):
        # 大向量乘法会使用32位 BLAS 进行分块处理
        # 测试分块处理是否正确，也参见 issue gh-22262
        
        # 创建一个数据数组，包含 2^30 + 100 个元素，数据类型为 dtype
        data = np.ones(2**30 + 100, dtype=dtype)
        
        # 计算数据数组与自身的点积
        res = np.dot(data, data)
        
        # 断言点积的结果应为 2^30 + 100
        assert res == 2**30 + 100
class MatmulCommon:
    """Common tests for '@' operator and numpy.matmul."""

    # 应该适用于这些类型。将来可能会添加"O"
    types = "?bhilBefdFD"

    def test_exceptions(self):
        # 不匹配的向量与向量
        dims = [
            ((1,), (2,)),
            # 不匹配的矩阵与向量
            ((2, 1), (2,)),
            # 不匹配的向量与矩阵
            ((2,), (1, 2)),
            # 不匹配的矩阵与矩阵
            ((1, 2), (3, 1)),
            # 向量与标量
            ((1,), ()),
            # 标量与向量
            ((), (1,)),
            # 矩阵与标量
            ((1, 1), ()),
            # 标量与矩阵
            ((), (1, 1)),
            # 无法广播的情况
            ((2, 2, 1), (3, 1, 2)),
        ]

        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            assert_raises((RuntimeError, ValueError), self.matmul, a, b)

    def test_shapes(self):
        # 广播第一个参数
        # 广播第二个参数
        # 矩阵堆叠大小相匹配
        dims = [
            ((1, 1), (2, 1, 1)),
            ((2, 1, 1), (1, 1)),
            ((2, 1, 1), (2, 1, 1)),
        ]

        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            res = self.matmul(a, b)
            assert_(res.shape == (2, 1, 1))

        # 向量向量返回标量。
        for dt in self.types:
            a = np.ones((2,), dtype=dt)
            b = np.ones((2,), dtype=dt)
            c = self.matmul(a, b)
            assert_(np.array(c).shape == ())

    def test_result_types(self):
        mat = np.ones((1, 1))
        vec = np.ones((1,))
        for dt in self.types:
            m = mat.astype(dt)
            v = vec.astype(dt)
            for arg in [(m, v), (v, m), (m, m)]:
                res = self.matmul(*arg)
                assert_(res.dtype == dt)

    @xpassIfTorchDynamo  # (reason="no scalars")
    def test_result_types_2(self):
        # 在 numpy 中，向量向量返回标量
        # 我们返回一个0维数组而不是标量

        for dt in self.types:
            v = np.ones((1,)).astype(dt)
            if dt != "O":
                res = self.matmul(v, v)
                assert_(type(res) is np.dtype(dt).type)

    def test_scalar_output(self):
        vec1 = np.array([2])
        vec2 = np.array([3, 4]).reshape(1, -1)
        tgt = np.array([6, 8])
        for dt in self.types[1:]:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            res = self.matmul(v1, v2)
            assert_equal(res, tgt)
            res = self.matmul(v2.T, v1)
            assert_equal(res, tgt)

        # 布尔类型
        vec = np.array([True, True], dtype="?").reshape(1, -1)
        res = self.matmul(vec[:, 0], vec)
        assert_equal(res, True)
    # 测试向量与向量的乘法，验证不同数据类型下的运算结果
    def test_vector_vector_values(self):
        # 创建第一个向量
        vec1 = np.array([1, 2])
        # 创建第二个向量，并将其转换为列向量形式
        vec2 = np.array([3, 4]).reshape(-1, 1)
        # 期望的第一次乘法结果
        tgt1 = np.array([11])
        # 期望的第二次乘法结果
        tgt2 = np.array([[3, 6], [4, 8]])
        # 遍历除了第一个数据类型外的其它数据类型
        for dt in self.types[1:]:
            # 将第一个向量转换为当前数据类型
            v1 = vec1.astype(dt)
            # 将第二个向量转换为当前数据类型
            v2 = vec2.astype(dt)
            # 进行矩阵乘法运算
            res = self.matmul(v1, v2)
            # 断言结果与期望值相等
            assert_equal(res, tgt1)
            # 因为没有广播，需要将 v1 转换为二维数组形式
            res = self.matmul(v2, v1.reshape(1, -1))
            # 断言结果与期望值相等
            assert_equal(res, tgt2)

        # 对于布尔类型
        vec = np.array([True, True], dtype="?")
        # 进行布尔类型的乘法运算
        res = self.matmul(vec, vec)
        # 断言结果与期望值相等
        assert_equal(res, True)

    # 测试向量与矩阵的乘法，验证不同数据类型下的运算结果
    def test_vector_matrix_values(self):
        # 创建向量
        vec = np.array([1, 2])
        # 创建第一个矩阵
        mat1 = np.array([[1, 2], [3, 4]])
        # 创建第二个矩阵，通过堆叠操作生成
        mat2 = np.stack([mat1] * 2, axis=0)
        # 期望的第一次乘法结果
        tgt1 = np.array([7, 10])
        # 期望的第二次乘法结果
        tgt2 = np.stack([tgt1] * 2, axis=0)
        # 遍历除了第一个数据类型外的其它数据类型
        for dt in self.types[1:]:
            # 将向量转换为当前数据类型
            v = vec.astype(dt)
            # 将第一个矩阵转换为当前数据类型
            m1 = mat1.astype(dt)
            # 将第二个矩阵转换为当前数据类型
            m2 = mat2.astype(dt)
            # 进行矩阵乘法运算
            res = self.matmul(v, m1)
            # 断言结果与期望值相等
            assert_equal(res, tgt1)
            # 进行矩阵乘法运算
            res = self.matmul(v, m2)
            # 断言结果与期望值相等
            assert_equal(res, tgt2)

        # 对于布尔类型
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1] * 2, axis=0)

        # 进行布尔类型的乘法运算
        res = self.matmul(vec, mat1)
        # 断言结果与期望值相等
        assert_equal(res, tgt1)
        # 进行布尔类型的乘法运算
        res = self.matmul(vec, mat2)
        # 断言结果与期望值相等
        assert_equal(res, tgt2)

    # 测试矩阵与向量的乘法，验证不同数据类型下的运算结果
    def test_matrix_vector_values(self):
        # 创建向量
        vec = np.array([1, 2])
        # 创建第一个矩阵
        mat1 = np.array([[1, 2], [3, 4]])
        # 创建第二个矩阵，通过堆叠操作生成
        mat2 = np.stack([mat1] * 2, axis=0)
        # 期望的第一次乘法结果
        tgt1 = np.array([5, 11])
        # 期望的第二次乘法结果
        tgt2 = np.stack([tgt1] * 2, axis=0)
        # 遍历除了第一个数据类型外的其它数据类型
        for dt in self.types[1:]:
            # 将向量转换为当前数据类型
            v = vec.astype(dt)
            # 将第一个矩阵转换为当前数据类型
            m1 = mat1.astype(dt)
            # 将第二个矩阵转换为当前数据类型
            m2 = mat2.astype(dt)
            # 进行矩阵乘法运算
            res = self.matmul(m1, v)
            # 断言结果与期望值相等
            assert_equal(res, tgt1)
            # 进行矩阵乘法运算
            res = self.matmul(m2, v)
            # 断言结果与期望值相等
            assert_equal(res, tgt2)

        # 对于布尔类型
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1] * 2, axis=0)
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1] * 2, axis=0)

        # 进行布尔类型的乘法运算
        res = self.matmul(vec, mat1)
        # 断言结果与期望值相等
        assert_equal(res, tgt1)
        # 进行布尔类型的乘法运算
        res = self.matmul(vec, mat2)
        # 断言结果与期望值相等
        assert_equal(res, tgt2)
    # 定义一个测试函数，用于测试矩阵乘法的不同情况
    def test_matrix_matrix_values(self):
        # 创建两个二维 NumPy 数组作为矩阵 mat1 和 mat2
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[1, 0], [1, 1]])

        # 将 mat1 和 mat2 沿着新的轴 0 进行堆叠，形成一个新的三维数组 mat12 和 mat21
        mat12 = np.stack([mat1, mat2], axis=0)
        mat21 = np.stack([mat2, mat1], axis=0)

        # 定义期望结果的矩阵 tgt11, tgt12, tgt21, tgt12_21, tgt11_12, tgt11_21
        tgt11 = np.array([[7, 10], [15, 22]])
        tgt12 = np.array([[3, 2], [7, 4]])
        tgt21 = np.array([[1, 2], [4, 6]])
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)

        # 遍历除了第一个类型之外的所有类型
        for dt in self.types[1:]:
            # 将 mat1, mat2, mat12, mat21 转换为当前数据类型 dt 的副本
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            m12 = mat12.astype(dt)
            m21 = mat21.astype(dt)

            # 矩阵乘法：m1 @ m2 和 m2 @ m1，并断言结果与 tgt12 和 tgt21 相等
            res = self.matmul(m1, m2)
            assert_equal(res, tgt12)
            res = self.matmul(m2, m1)
            assert_equal(res, tgt21)

            # 堆叠矩阵 @ 矩阵：m12 @ m1，断言结果与 tgt11_21 相等
            res = self.matmul(m12, m1)
            assert_equal(res, tgt11_21)

            # 矩阵 @ 堆叠矩阵：m1 @ m12，断言结果与 tgt11_12 相等
            res = self.matmul(m1, m12)
            assert_equal(res, tgt11_12)

            # 堆叠矩阵 @ 堆叠矩阵：m12 @ m21，断言结果与 tgt12_21 相等
            res = self.matmul(m12, m21)
            assert_equal(res, tgt12_21)

        # 布尔类型
        m1 = np.array([[1, 1], [0, 0]], dtype=np.bool_)
        m2 = np.array([[1, 0], [1, 1]], dtype=np.bool_)
        m12 = np.stack([m1, m2], axis=0)
        m21 = np.stack([m2, m1], axis=0)
        tgt11 = m1
        tgt12 = m1
        tgt21 = np.array([[1, 1], [1, 1]], dtype=np.bool_)
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)

        # 矩阵乘法：m1 @ m2 和 m2 @ m1，断言结果与 tgt12 和 tgt21 相等
        res = self.matmul(m1, m2)
        assert_equal(res, tgt12)
        res = self.matmul(m2, m1)
        assert_equal(res, tgt21)

        # 堆叠矩阵 @ 矩阵：m12 @ m1，断言结果与 tgt11_21 相等
        res = self.matmul(m12, m1)
        assert_equal(res, tgt11_21)

        # 矩阵 @ 堆叠矩阵：m1 @ m12，断言结果与 tgt11_12 相等
        res = self.matmul(m1, m12)
        assert_equal(res, tgt11_12)

        # 堆叠矩阵 @ 堆叠矩阵：m12 @ m21，断言结果与 tgt12_21 相等
        res = self.matmul(m12, m21)
        assert_equal(res, tgt12_21)
@instantiate_parametrized_tests
class TestMatmul(MatmulCommon, TestCase):
    # 使用参数化测试装饰器，创建测试类 TestMatmul，继承自 MatmulCommon 和 TestCase
    def setUp(self):
        # 设置测试方法的前置条件
        self.matmul = np.matmul

    def test_out_arg(self):
        # 定义测试方法 test_out_arg
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)
        tgt = np.dot(a, b)

        # test as positional argument
        msg = "out positional argument"
        out = np.zeros((5, 2), dtype=float)
        self.matmul(a, b, out)
        assert_array_equal(out, tgt, err_msg=msg)

        # test as keyword argument
        msg = "out keyword argument"
        out = np.zeros((5, 2), dtype=float)
        self.matmul(a, b, out=out)
        assert_array_equal(out, tgt, err_msg=msg)

        # test out with not allowed type cast (safe casting)
        msg = "Cannot cast"
        out = np.zeros((5, 2), dtype=np.int32)
        assert_raises_regex(TypeError, msg, self.matmul, a, b, out=out)

        # test out with type upcast to complex
        out = np.zeros((5, 2), dtype=np.complex128)
        c = self.matmul(a, b, out=out)
        assert_(c is out)
        #      with suppress_warnings() as sup:
        #          sup.filter(np.ComplexWarning, '')
        c = c.astype(tgt.dtype)
        assert_array_equal(c, tgt)

    def test_empty_out(self):
        # 检查输出不能广播，因此当外部维度（迭代器大小）为零时，它不能为大小为零。
        arr = np.ones((0, 1, 1))
        out = np.ones((1, 1, 1))
        assert self.matmul(arr, arr).shape == (0, 1, 1)

        with pytest.raises((RuntimeError, ValueError)):
            self.matmul(arr, arr, out=out)

    def test_out_contiguous(self):
        # 定义测试方法 test_out_contiguous
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)
        v = np.array([1, 3], dtype=float)
        tgt = np.dot(a, b)
        tgt_mv = np.dot(a, v)

        # test out non-contiguous
        out = np.ones((5, 2, 2), dtype=float)
        c = self.matmul(a, b, out=out[..., 0])
        assert_array_equal(c, tgt)
        c = self.matmul(a, v, out=out[:, 0, 0])
        assert_array_equal(c, tgt_mv)
        c = self.matmul(v, a.T, out=out[:, 0, 0])
        assert_array_equal(c, tgt_mv)

        # test out contiguous in only last dim
        out = np.ones((10, 2), dtype=float)
        c = self.matmul(a, b, out=out[::2, :])
        assert_array_equal(c, tgt)

        # test transposes of out, args
        out = np.ones((5, 2), dtype=float)
        c = self.matmul(b.T, a.T, out=out.T)
        assert_array_equal(out, tgt)

    @xfailIfTorchDynamo
    def test_out_contiguous_2(self):
        # 标记为如果 TorchDynamo 失败则跳过的测试方法
        a = np.ones((5, 2), dtype=float)
        b = np.array([[1, 3], [5, 7]], dtype=float)

        # test out non-contiguous
        out = np.ones((5, 2, 2), dtype=float)
        c = self.matmul(a, b, out=out[..., 0])
        assert c.tensor._base is out.tensor

    m1 = np.arange(15.0).reshape(5, 3)
    m2 = np.arange(21.0).reshape(3, 7)
    m3 = np.arange(30.0).reshape(5, 6)[:, ::2]  # 创建一个非连续的 NumPy 数组 m3，从一个 reshape 后的数组中取出部分列

    vc = np.arange(10.0)  # 创建一个包含 0 到 9 的连续向量 vc

    vr = np.arange(6.0)  # 创建一个包含 0 到 5 的连续向量 vr

    m0 = np.zeros((3, 0))  # 创建一个大小为 3x0 的零矩阵 m0

    @parametrize(
        "args",
        (
            # 矩阵-矩阵乘法测试用例
            subtest((m1, m2), name="mm1"),
            subtest((m2.T, m1.T), name="mm2"),
            subtest((m2.T.copy(), m1.T), name="mm3"),
            subtest((m2.T, m1.T.copy()), name="mm4"),
            # 矩阵-矩阵转置乘法测试用例，包括连续和非连续情况
            subtest((m1, m1.T), name="mmT1"),
            subtest((m1.T, m1), name="mmT2"),
            subtest((m1, m3.T), name="mmT3"),
            subtest((m3, m1.T), name="mmT4"),
            subtest((m3, m3.T), name="mmT5"),
            subtest((m3.T, m3), name="mmT6"),
            # 矩阵-矩阵非连续乘法测试用例
            subtest((m3, m2), name="mmN1"),
            subtest((m2.T, m3.T), name="mmN2"),
            subtest((m2.T.copy(), m3.T), name="mmN3"),
            # 向量-矩阵、矩阵-向量乘法测试用例，包括连续情况
            subtest((m1, vr[:3]), name="vm1"),
            subtest((vc[:5], m1), name="vm2"),
            subtest((m1.T, vc[:5]), name="vm3"),
            subtest((vr[:3], m1.T), name="vm4"),
            # 向量-矩阵、矩阵-向量乘法测试用例，包括向量非连续情况
            subtest((m1, vr[::2]), name="mvN1"),
            subtest((vc[::2], m1), name="mvN2"),
            subtest((m1.T, vc[::2]), name="mvN3"),
            subtest((vr[::2], m1.T), name="mvN4"),
            # 向量-矩阵、矩阵-向量乘法测试用例，包括矩阵非连续情况
            subtest((m3, vr[:3]), name="mvN5"),
            subtest((vc[:5], m3), name="mvN6"),
            subtest((m3.T, vc[:5]), name="mvN7"),
            subtest((vr[:3], m3.T), name="mvN8"),
            # 向量-矩阵、矩阵-向量乘法测试用例，包括向量和矩阵都非连续情况
            subtest((m3, vr[::2]), name="mvN9"),
            subtest((vc[::2], m3), name="mvn10"),
            subtest((m3.T, vc[::2]), name="mv11"),
            subtest((vr[::2], m3.T), name="mv12"),
            # 大小为0的情况
            subtest((m0, m0.T), name="s0_1"),
            subtest((m0.T, m0), name="s0_2"),
            subtest((m1, m0), name="s0_3"),
            subtest((m0.T, m1.T), name="s0_4"),
        ),
    )
    def test_dot_equivalent(self, args):
        r1 = np.matmul(*args)  # 使用 np.matmul 计算 args 中两个矩阵的乘积
        r2 = np.dot(*args)  # 使用 np.dot 计算 args 中两个数组的点积
        assert_equal(r1, r2)  # 断言 np.matmul 和 np.dot 的结果相等

        r3 = np.matmul(args[0].copy(), args[1].copy())  # 使用复制后的数组进行 np.matmul 计算
        assert_equal(r1, r3)  # 断言复制后的计算结果与原始结果相等

    @skip(reason="object arrays")
    def test_matmul_exception_multiply(self):
        # 测试如果缺少 __mul__ 方法，matmul 是否会失败
        class add_not_multiply:
            def __add__(self, other):
                return self

        a = np.full((3, 3), add_not_multiply())  # 创建一个填充对象数组
        with assert_raises(TypeError):  # 断言捕获 TypeError 异常
            b = np.matmul(a, a)  # 尝试对对象数组执行 np.matmul
    # 定义一个测试函数，用于测试矩阵乘法在缺少 `__add__` 方法时是否会失败
    def test_matmul_exception_add(self):
        # 定义一个类 multiply_not_add，缺少 `__add__` 方法，只有 `__mul__` 方法
        class multiply_not_add:
            def __mul__(self, other):
                return self
        
        # 创建一个 3x3 的 NumPy 数组，元素类型为 multiply_not_add 的实例
        a = np.full((3, 3), multiply_not_add())
        # 使用 assert_raises 来验证 matmul 函数调用时是否会引发 TypeError 异常
        with assert_raises(TypeError):
            b = np.matmul(a, a)

    # 定义一个测试函数，测试布尔类型数组的矩阵乘法
    def test_matmul_bool(self):
        # 创建一个 2x2 的布尔类型 NumPy 数组 a
        a = np.array([[1, 0], [1, 1]], dtype=bool)
        # 断言数组 a 中的最大值转换为 np.uint8 后是否为 1
        assert np.max(a.view(np.uint8)) == 1
        # 对数组 a 进行矩阵乘法运算，结果保存在数组 b 中
        b = np.matmul(a, a)
        # 断言数组 b 中的最大值转换为 np.uint8 后是否为 1
        # 布尔类型数组的矩阵乘法结果应该总是 0 或 1
        assert np.max(b.view(np.uint8)) == 1

        # 使用固定种子值（1234）生成随机数，创建一个 4x5 的布尔类型 NumPy 数组 d
        np.random.seed(1234)
        d = np.random.randint(2, size=(4, 5)) > 0

        # 对数组 d 和 d 转置后的结果进行矩阵乘法，分别保存在 out1 和 out2 中
        out1 = np.matmul(d, d.reshape(5, 4))
        out2 = np.dot(d, d.reshape(5, 4))
        # 使用 assert_equal 断言 out1 和 out2 是否相等
        assert_equal(out1, out2)

        # 创建两个空数组进行矩阵乘法运算，结果保存在数组 c 中
        c = np.matmul(np.zeros((2, 0), dtype=bool), np.zeros(0, dtype=bool))
        # 断言数组 c 中是否没有任何元素
        assert not np.any(c)
class TestMatmulOperator(MatmulCommon, TestCase):
    import operator  # 导入 Python 的 operator 模块

    matmul = operator.matmul  # 将 operator.matmul 赋值给类属性 matmul

    @skip(reason="no __array_priority__")  # 标记该测试跳过，原因是没有 __array_priority__

    def test_array_priority_override(self):
        class A:
            __array_priority__ = 1000  # 设置类 A 的 __array_priority__ 为 1000

            def __matmul__(self, other):  # 定义类 A 的 __matmul__ 方法
                return "A"

            def __rmatmul__(self, other):  # 定义类 A 的 __rmatmul__ 方法
                return "A"

        a = A()  # 创建 A 的实例 a
        b = np.ones(2)  # 创建一个包含两个元素的全为 1 的 numpy 数组 b
        assert_equal(self.matmul(a, b), "A")  # 断言 matmul 方法对 a 和 b 的运算结果为 "A"
        assert_equal(self.matmul(b, a), "A")  # 断言 matmul 方法对 b 和 a 的运算结果为 "A"

    def test_matmul_raises(self):
        assert_raises(
            (RuntimeError, TypeError, ValueError), self.matmul, np.int8(5), np.int8(5)
        )  # 断言 matmul 方法对 np.int8 类型的两个参数会引发 RuntimeError、TypeError 或 ValueError 中的一个异常

    @xpassIfTorchDynamo  # 标记测试通过条件是 Torch Dynamo 支持，因为 torch 支持就地 matmul，我们也支持
    @skipif(numpy.__version__ >= "1.26", reason="This is fixed in numpy 1.26")  # 如果 numpy 版本大于等于 1.26，跳过测试，因为问题已在该版本中修复
    def test_matmul_inplace(self):
        # It would be nice to support in-place matmul eventually, but for now
        # we don't have a working implementation, so better just to error out
        # and nudge people to writing "a = a @ b".
        a = np.eye(3)  # 创建一个 3x3 的单位矩阵 a
        b = np.eye(3)  # 创建一个 3x3 的单位矩阵 b
        assert_raises(TypeError, a.__imatmul__, b)  # 断言对 a 进行就地 matmul 会引发 TypeError 异常

    @xfail  # 标记这个测试为预期失败，因为在 Dynamo 下执行有问题
    def test_matmul_inplace_2(self):
        a = np.eye(3)  # 创建一个 3x3 的单位矩阵 a
        b = np.eye(3)  # 创建一个 3x3 的单位矩阵 b

        assert_raises(TypeError, operator.imatmul, a, b)  # 断言使用 operator.imatmul 对 a 和 b 进行就地 matmul 会引发 TypeError 异常
        assert_raises(TypeError, exec, "a @= b", globals(), locals())  # 断言执行 "a @= b" 会引发 TypeError 异常

    @xpassIfTorchDynamo  # 标记测试通过条件是 Torch Dynamo 支持，因为支持 matmul_axes
    def test_matmul_axes(self):
        a = np.arange(3 * 4 * 5).reshape(3, 4, 5)  # 创建一个形状为 (3, 4, 5) 的 numpy 数组 a
        c = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (1, 2)])  # 使用指定轴进行 matmul，得到数组 c
        assert c.shape == (3, 4, 4)  # 断言数组 c 的形状为 (3, 4, 4)
        d = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (0, 1)])  # 使用指定轴进行 matmul，得到数组 d
        assert d.shape == (4, 4, 3)  # 断言数组 d 的形状为 (4, 4, 3)
        e = np.swapaxes(d, 0, 2)  # 交换数组 d 的轴 0 和 2，得到数组 e
        assert_array_equal(e, c)  # 断言数组 e 和数组 c 相等
        f = np.matmul(a, np.arange(3), axes=[(1, 0), (0), (0)])  # 使用指定轴进行 matmul，得到数组 f
        assert f.shape == (4, 5)  # 断言数组 f 的形状为 (4, 5)


class TestInner(TestCase):
    def test_inner_scalar_and_vector(self):
        for dt in np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "?":
            sca = np.array(3, dtype=dt)[()]  # 创建一个标量 sca，类型为 dt
            vec = np.array([1, 2], dtype=dt)  # 创建一个向量 vec，类型为 dt
            desired = np.array([3, 6], dtype=dt)  # 创建一个期望结果 desired，类型为 dt
            assert_equal(np.inner(vec, sca), desired)  # 断言 np.inner 计算 vec 和 sca 的内积结果与 desired 相等
            assert_equal(np.inner(sca, vec), desired)  # 断言 np.inner 计算 sca 和 vec 的内积结果与 desired 相等

    def test_vecself(self):
        # Ticket 844.
        # Inner product of a vector with itself segfaults or give
        # meaningless result
        a = np.zeros(shape=(1, 80), dtype=np.float64)  # 创建一个形状为 (1, 80)，类型为 np.float64 的全零数组 a
        p = np.inner(a, a)  # 计算数组 a 和自身的内积，得到 p
        assert_almost_equal(p, 0, decimal=14)  # 断言 p 的值在小数点后 14 位精度下接近于 0
    # 定义一个测试方法，用于测试不同连续性条件下的内积运算
    def test_inner_product_with_various_contiguities(self):
        # github issue 6532
        # 遍历所有整数和浮点数数据类型，以及布尔类型
        for dt in np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "?":
            # 创建矩阵 A 和 B，数据类型为 dt
            A = np.array([[1, 2], [3, 4]], dtype=dt)
            B = np.array([[1, 3], [2, 4]], dtype=dt)
            # 创建向量 C，数据类型为 dt
            C = np.array([1, 1], dtype=dt)
            # 预期结果向量，数据类型为 dt
            desired = np.array([4, 6], dtype=dt)
            # 断言矩阵转置与向量 C 的内积等于预期结果
            assert_equal(np.inner(A.T, C), desired)
            # 断言向量 C 与矩阵转置的内积等于预期结果
            assert_equal(np.inner(C, A.T), desired)
            # 断言矩阵 B 与向量 C 的内积等于预期结果
            assert_equal(np.inner(B, C), desired)
            # 断言向量 C 与矩阵 B 的内积等于预期结果
            assert_equal(np.inner(C, B), desired)
            # 检查矩阵乘积，计算预期结果矩阵
            desired = np.array([[7, 10], [15, 22]], dtype=dt)
            assert_equal(np.inner(A, B), desired)
            # 检查 syrk 与 gemm 路径
            desired = np.array([[5, 11], [11, 25]], dtype=dt)
            assert_equal(np.inner(A, A), desired)
            assert_equal(np.inner(A, A.copy()), desired)

    # 标记为跳过测试，原因是不支持 [::-1] 的操作
    @skip(reason="[::-1] not supported")
    def test_inner_product_reversed_view(self):
        # 遍历所有整数和浮点数数据类型，以及布尔类型
        for dt in np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "?":
            # 创建数组 a，以及其反转视图 b
            a = np.arange(5).astype(dt)
            b = a[::-1]
            # 预期结果为标量 10，数据类型为 dt
            desired = np.array(10, dtype=dt).item()
            # 断言反转视图 b 与数组 a 的内积等于预期结果
            assert_equal(np.inner(b, a), desired)

    # 定义一个测试方法，用于测试三维张量的内积运算
    def test_3d_tensor(self):
        # 遍历所有整数和浮点数数据类型，以及布尔类型
        for dt in np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "?":
            # 创建两个三维张量 a 和 b，数据类型为 dt
            a = np.arange(24).reshape(2, 3, 4).astype(dt)
            b = np.arange(24, 48).reshape(2, 3, 4).astype(dt)
            # 创建预期结果的三维张量，数据类型为 dt
            desired = np.array(
                [
                    [
                        [[158, 182, 206], [230, 254, 278]],
                        [[566, 654, 742], [830, 918, 1006]],
                        [[974, 1126, 1278], [1430, 1582, 1734]],
                    ],
                    [
                        [[1382, 1598, 1814], [2030, 2246, 2462]],
                        [[1790, 2070, 2350], [2630, 2910, 3190]],
                        [[2198, 2542, 2886], [3230, 3574, 3918]],
                    ],
                ]
            ).astype(dt)
            # 断言三维张量 a 和 b 的内积等于预期结果
            assert_equal(np.inner(a, b), desired)
            # 断言三维张量 b 和 a 的转置结果的内积等于预期结果
            assert_equal(np.inner(b, a).transpose(2, 3, 0, 1), desired)
@instantiate_parametrized_tests
class TestChoose(TestCase):
    # 在测试类初始化前准备数据
    def setUp(self):
        self.x = 2 * np.ones((3,), dtype=int)  # 创建包含三个元素的整数数组 self.x
        self.y = 3 * np.ones((3,), dtype=int)  # 创建包含三个元素的整数数组 self.y
        self.x2 = 2 * np.ones((2, 3), dtype=int)  # 创建包含 2x3 元素的整数数组 self.x2
        self.y2 = 3 * np.ones((2, 3), dtype=int)  # 创建包含 2x3 元素的整数数组 self.y2
        self.ind = [0, 0, 1]  # 创建包含三个索引的列表 self.ind

    # 测试 np.choose 的基本用法
    def test_basic(self):
        A = np.choose(self.ind, (self.x, self.y))  # 使用 self.ind 和 self.x, self.y 作为参数调用 np.choose
        assert_equal(A, [2, 2, 3])  # 断言 A 应为 [2, 2, 3]

    # 测试 np.choose 在广播情况下的应用
    def test_broadcast1(self):
        A = np.choose(self.ind, (self.x2, self.y2))  # 使用 self.ind 和 self.x2, self.y2 作为参数调用 np.choose
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])  # 断言 A 应为 2x3 的二维数组

    # 测试 np.choose 在混合维度广播情况下的应用
    def test_broadcast2(self):
        A = np.choose(self.ind, (self.x, self.y2))  # 使用 self.ind 和 self.x, self.y2 作为参数调用 np.choose
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])  # 断言 A 应为 2x3 的二维数组

    # 标记为跳过测试，并添加跳过原因
    @skip(reason="XXX: revisit xfails when NEP 50 lands in numpy")
    # 参数化测试，测试 np.choose 的输出数据类型
    @parametrize(
        "ops",
        [
            (1000, np.array([1], dtype=np.uint8)),  # 参数组合1
            (-1, np.array([1], dtype=np.uint8)),    # 参数组合2
            (1.0, np.float32(3)),                   # 参数组合3
            (1.0, np.array([3], dtype=np.float32)),  # 参数组合4
        ],
    )
    def test_output_dtype(self, ops):
        expected_dt = np.result_type(*ops)  # 计算预期的输出数据类型
        assert np.choose([0], ops).dtype == expected_dt  # 断言 np.choose 的输出数据类型与预期一致

    # 测试 np.choose 的文档示例1
    def test_docstring_1(self):
        # 从文档中复制的示例，测试 np.choose 的应用
        choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        A = np.choose([2, 3, 1, 0], choices)  # 使用给定选择数组和索引调用 np.choose
        assert_equal(A, [20, 31, 12, 3])  # 断言 A 应为 [20, 31, 12, 3]

    # 测试 np.choose 的文档示例2
    def test_docstring_2(self):
        a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        choices = [-10, 10]
        A = np.choose(a, choices)  # 使用给定选择数组和索引调用 np.choose
        assert_equal(A, [[10, -10, 10], [-10, 10, -10], [10, -10, 10]])  # 断言 A 应为指定的二维数组

    # 测试 np.choose 的文档示例3
    def test_docstring_3(self):
        a = np.array([0, 1]).reshape((2, 1, 1))
        c1 = np.array([1, 2, 3]).reshape((1, 3, 1))
        c2 = np.array([-1, -2, -3, -4, -5]).reshape((1, 1, 5))
        A = np.choose(a, (c1, c2))  # 使用给定选择数组和索引调用 np.choose
        expected = np.array(
            [
                [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                [[-1, -2, -3, -4, -5], [-1, -2, -3, -4, -5], [-1, -2, -3, -4, -5]],
            ]
        )
        assert_equal(A, expected)  # 断言 A 应与预期的数组相等


class TestRepeat(TestCase):
    def setUp(self):
        self.m = np.array([1, 2, 3, 4, 5, 6])  # 创建包含六个元素的数组 self.m
        self.m_rect = self.m.reshape((2, 3))  # 创建一个 2x3 的二维数组 self.m_rect

    # 测试 np.repeat 的基本用法
    def test_basic(self):
        A = np.repeat(self.m, [1, 3, 2, 1, 1, 2])  # 对 self.m 应用指定的重复次数
        assert_equal(A, [1, 2, 2, 2, 3, 3, 4, 5, 6, 6])  # 断言 A 应为指定的数组

    # 测试 np.repeat 在广播情况下的应用
    def test_broadcast1(self):
        A = np.repeat(self.m, 2)  # 对 self.m 应用重复两次
        assert_equal(A, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])  # 断言 A 应为指定的数组

    # 测试指定轴上的 np.repeat
    def test_axis_spec(self):
        A = np.repeat(self.m_rect, [2, 1], axis=0)  # 在 axis=0 上重复指定次数
        assert_equal(A, [[1, 2, 3], [1, 2, 3], [4, 5, 6]])  # 断言 A 应为指定的二维数组

        A = np.repeat(self.m_rect, [1, 3, 2], axis=1)  # 在 axis=1 上重复指定次数
        assert_equal(A, [[1, 2, 2, 2, 3, 3], [4, 5, 5, 5, 6, 6]])  # 断言 A 应为指定的二维数组
    # 定义一个测试方法，用于测试 NumPy 的广播功能
    def test_broadcast2(self):
        # 对矩阵 self.m_rect 按指定轴向进行重复，axis=0 表示沿行的方向重复两次
        A = np.repeat(self.m_rect, 2, axis=0)
        # 断言重复后的矩阵 A 符合预期值
        assert_equal(A, [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])

        # 对矩阵 self.m_rect 按指定轴向进行重复，axis=1 表示沿列的方向重复两次
        A = np.repeat(self.m_rect, 2, axis=1)
        # 断言重复后的矩阵 A 符合预期值
        assert_equal(A, [[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]])
# 定义邻域模式字典，用于多维数组邻域操作
NEIGH_MODE = {"zero": 0, "one": 1, "constant": 2, "circular": 3, "mirror": 4}

# 根据装饰器 xpassIfTorchDynamo 进行测试，理由是 TODO
@xpassIfTorchDynamo  # (reason="TODO")
class TestWarnings(TestCase):
    def test_complex_warning(self):
        # 创建包含复数的 numpy 数组
        x = np.array([1, 2])
        y = np.array([1 - 2j, 1 + 2j])

        # 捕获复数警告，确保设置警告筛选器
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.ComplexWarning)
            # 验证设置复数时触发复数警告
            assert_raises(np.ComplexWarning, x.__setitem__, slice(None), y)
            # 验证 x 数组不变
            assert_equal(x, [1, 2])


class TestMinScalarType(TestCase):
    def test_usigned_shortshort(self):
        # 获取小于等于 255 的最小无符号整数类型
        dt = np.min_scalar_type(2**8 - 1)
        wanted = np.dtype("uint8")
        assert_equal(wanted, dt)

    # 根据 numpy 实现添加了下面三个测试
    def test_complex(self):
        # 获取复数 0+0j 的最小数据类型
        dt = np.min_scalar_type(0 + 0j)
        assert dt == np.dtype("complex64")

    def test_float(self):
        # 获取浮点数 0.1 的最小数据类型
        dt = np.min_scalar_type(0.1)
        assert dt == np.dtype("float16")

    def test_nonscalar(self):
        # 获取非标量数组 [0, 1, 2] 的最小数据类型
        dt = np.min_scalar_type([0, 1, 2])
        assert dt == np.dtype("int64")


from numpy.core._internal import _dtype_from_pep3118

# 根据 skip 装饰器的理由跳过测试，理由是 "dont worry about buffer protocol"
@skip(reason="dont worry about buffer protocol")
class TestPEP3118Dtype(TestCase):
    def _check(self, spec, wanted):
        # 创建 numpy 数据类型 wanted
        dt = np.dtype(wanted)
        # 使用 _dtype_from_pep3118 函数根据 PEP 3118 规范检查 spec 是否与 wanted 相等
        actual = _dtype_from_pep3118(spec)
        assert_equal(actual, dt, err_msg=f"spec {spec!r} != dtype {wanted!r}")

    def test_native_padding(self):
        # 获取整数 i 的对齐值
        align = np.dtype("i").alignment
        for j in range(8):
            if j == 0:
                s = "bi"
            else:
                s = "b%dxi" % j
            # 检查 native padding 使用 @s 格式
            self._check(
                "@" + s, {"f0": ("i1", 0), "f1": ("i", align * (1 + j // align))}
            )
            # 检查 native padding 使用 =s 格式
            self._check("=" + s, {"f0": ("i1", 0), "f1": ("i", 1 + j)})

    def test_native_padding_2(self):
        # native padding 也适用于结构体和子数组
        self._check("x3T{xi}", {"f0": (({"f0": ("i", 4)}, (3,)), 4)})
        self._check("^x3T{xi}", {"f0": (({"f0": ("i", 1)}, (3,)), 1)})
    def test_trailing_padding(self):
        # 测试尾部填充是否包含，并且如果在对齐模式下，项目大小应该与对齐一致
        align = np.dtype("i").alignment  # 获取整数类型的对齐方式
        size = np.dtype("i").itemsize  # 获取整数类型的大小

        def aligned(n):
            return align * (1 + (n - 1) // align)  # 返回对齐后的大小

        base = dict(formats=["i"], names=["f0"])  # 基本的数据格式字典

        self._check("ix", dict(itemsize=aligned(size + 1), **base))  # 检查包含对齐后大小的数据格式
        self._check("ixx", dict(itemsize=aligned(size + 2), **base))  # 检查包含对齐后大小的数据格式
        self._check("ixxx", dict(itemsize=aligned(size + 3), **base))  # 检查包含对齐后大小的数据格式
        self._check("ixxxx", dict(itemsize=aligned(size + 4), **base))  # 检查包含对齐后大小的数据格式
        self._check("i7x", dict(itemsize=aligned(size + 7), **base))  # 检查包含对齐后大小的数据格式

        self._check("^ix", dict(itemsize=size + 1, **base))  # 检查不对齐时的数据格式
        self._check("^ixx", dict(itemsize=size + 2, **base))  # 检查不对齐时的数据格式
        self._check("^ixxx", dict(itemsize=size + 3, **base))  # 检查不对齐时的数据格式
        self._check("^ixxxx", dict(itemsize=size + 4, **base))  # 检查不对齐时的数据格式
        self._check("^i7x", dict(itemsize=size + 7, **base))  # 检查不对齐时的数据格式

    def test_native_padding_3(self):
        dt = np.dtype(
            [("a", "b"), ("b", "i"), ("sub", np.dtype("b,i")), ("c", "i")], align=True
        )
        self._check("T{b:a:xxxi:b:T{b:f0:=i:f1:}:sub:xxxi:c:}", dt)  # 检查结构体的对齐方式

        dt = np.dtype(
            [
                ("a", "b"),
                ("b", "i"),
                ("c", "b"),
                ("d", "b"),
                ("e", "b"),
                ("sub", np.dtype("b,i", align=True)),
            ]
        )
        self._check("T{b:a:=i:b:b:c:b:d:b:e:T{b:f0:xxxi:f1:}:sub:}", dt)  # 检查结构体的对齐方式

    def test_padding_with_array_inside_struct(self):
        dt = np.dtype(
            [("a", "b"), ("b", "i"), ("c", "b", (3,)), ("d", "i")], align=True
        )
        self._check("T{b:a:xxxi:b:3b:c:xi:d:}", dt)  # 检查包含数组的结构体的对齐方式

    def test_byteorder_inside_struct(self):
        # 在 @T{=i} 后，字节顺序应为 '='，而不是 '@'。
        # 通过注意本地对齐的缺失来验证此问题。
        self._check("@T{^i}xi", {"f0": ({"f0": ("i", 0)}, 0), "f1": ("i", 5)})  # 检查结构体内部的字节顺序

    def test_intra_padding(self):
        # 本地对齐的子数组可能需要一些内部填充
        align = np.dtype("i").alignment  # 获取整数类型的对齐方式
        size = np.dtype("i").itemsize  # 获取整数类型的大小

        def aligned(n):
            return align * (1 + (n - 1) // align)  # 返回对齐后的大小

        self._check(
            "(3)T{ix}",
            (
                dict(
                    names=["f0"], formats=["i"], offsets=[0], itemsize=aligned(size + 1)
                ),
                (3,),
            ),
        )  # 检查结构体内部数组的对齐方式

    def test_char_vs_string(self):
        dt = np.dtype("c")  # 检查字符类型
        self._check("c", dt)  # 检查字符类型的数据格式

        dt = np.dtype([("f0", "S1", (4,)), ("f1", "S4")])
        self._check("4c4s", dt)  # 检查字符串类型的数据格式

    def test_field_order(self):
        # gh-9053 - 以前我们依赖于字典键的顺序
        self._check("(0)I:a:f:b:", [("a", "I", (0,)), ("b", "f")])  # 检查字段的顺序
        self._check("(0)I:b:f:a:", [("b", "I", (0,)), ("a", "f")])  # 检查字段的顺序
    # 定义测试方法 test_unnamed_fields，用于测试未命名字段的情况
    def test_unnamed_fields(self):
        # 调用 self._check 方法，验证输入格式为 "ii" 时的输出是否符合预期，期望的输出是 [("f0", "i"), ("f1", "i")]
        self._check("ii", [("f0", "i"), ("f1", "i")])
        # 调用 self._check 方法，验证输入格式为 "ii:f0:" 时的输出是否符合预期，期望的输出是 [("f1", "i"), ("f0", "i")]
        self._check("ii:f0:", [("f1", "i"), ("f0", "i")])

        # 调用 self._check 方法，验证输入格式为 "i" 时的输出是否符合预期，期望的输出是 "i"
        self._check("i", "i")
        # 调用 self._check 方法，验证输入格式为 "i:f0:" 时的输出是否符合预期，期望的输出是 [("f0", "i")]
        self._check("i:f0:", [("f0", "i")])
# NOTE: xpassIfTorchDynamo below
# 在下方的xpassIfTorchDynamo修饰器用于跳过与Torch Dynamo有关的测试用例

# 1. TODO: torch._numpy does not handle/model _CopyMode
# 对于torch._numpy，尚未处理或模拟_CopyMode功能

# 2. order= keyword not supported (probably won't be)
# 不支持order=关键字参数（可能不会支持）

# 3. Under TEST_WITH_TORCHDYNAMO many of these make it through due
#    to a graph break leaving the _CopyMode to only be handled by numpy.
# 在TEST_WITH_TORCHDYNAMO下，由于图的断开，许多这些情况通过，只有numpy来处理_CopyMode。

@skipif(numpy.__version__ < "1.23", reason="CopyMode is new in NumPy 1.22")
# 如果numpy版本低于1.23，则跳过这些测试用例，因为_CopyMode是在NumPy 1.22中引入的新功能

@xpassIfTorchDynamo
# 在Torch Dynamo环境下，跳过这些测试用例

@instantiate_parametrized_tests
# 实例化参数化测试用例

class TestArrayCreationCopyArgument(TestCase):
    # 测试用例类，用于测试数组创建时的复制参数设置

    class RaiseOnBool:
        def __bool__(self):
            raise ValueError

    # true_vals = [True, np._CopyMode.ALWAYS, np.True_]
    # false_vals = [False, np._CopyMode.IF_NEEDED, np.False_]
    true_vals = [True, 1, np.True_]
    false_vals = [False, 0, np.False_]

    def test_scalars(self):
        # 测试标量值
        # 测试numpy和python标量

        for dtype in np.typecodes["All"]:
            # 对于所有numpy数据类型进行迭代

            arr = np.zeros((), dtype=dtype)
            # 创建一个指定dtype的零维数组
            scalar = arr[()]
            # 从数组中获取标量值
            pyscalar = arr.item(0)
            # 通过索引获取python标量值

            # Test never-copy raises error:
            # 测试不复制时是否引发错误

            assert_raises(ValueError, np.array, scalar, copy=np._CopyMode.NEVER)
            # 使用np._CopyMode.NEVER时，确保不复制会引发值错误

            assert_raises(ValueError, np.array, pyscalar, copy=np._CopyMode.NEVER)
            # 使用np._CopyMode.NEVER时，确保不复制会引发值错误

            assert_raises(ValueError, np.array, pyscalar, copy=self.RaiseOnBool())
            # 使用自定义的RaiseOnBool对象时，确保不复制会引发值错误

            # Casting with a dtype (to unsigned integers) can be special:
            # 使用dtype进行转换（转换为无符号整数）可能会有特殊情况
            with pytest.raises(ValueError):
                np.array(pyscalar, dtype=np.int64, copy=np._CopyMode.NEVER)

    @xfail  # TODO: handle `_CopyMode` properly in torch._numpy
    # 标记为xfail，表示预期在torch._numpy中正确处理_CopyMode
    def test_compatible_cast(self):
        # Some types are compatible even though they are different, no
        # copy is necessary for them. This is mostly true for some integers
        
        # 定义一个生成器函数，生成所有整数类型及其无符号版本的数据类型
        def int_types(byteswap=False):
            int_types = np.typecodes["Integer"] + np.typecodes["UnsignedInteger"]
            for int_type in int_types:
                yield np.dtype(int_type)
                if byteswap:
                    yield np.dtype(int_type).newbyteorder()

        # 遍历所有整数类型及其无符号版本的组合
        for int1 in int_types():
            for int2 in int_types(True):
                # 创建一个整数类型为 int1 的长度为 10 的数组
                arr = np.arange(10, dtype=int1)

                # 遍历 self.true_vals 中的值
                for copy in self.true_vals:
                    # 创建一个数据类型为 int2 的数组 res，并断言其为 arr 的复制品，并且具有独立的内存
                    res = np.array(arr, copy=copy, dtype=int2)
                    assert res is not arr and res.flags.owndata
                    assert_array_equal(res, arr)

                # 如果 int1 和 int2 相同
                if int1 == int2:
                    # 在这种情况下，不需要进行类型转换，基本检查就足够了
                    for copy in self.false_vals:
                        # 创建一个数据类型为 int2 的数组 res，并断言其与 arr 共享基础数据
                        res = np.array(arr, copy=copy, dtype=int2)
                        assert res is arr or res.base is arr

                    # 创建一个数据类型为 int2 的数组 res，并断言其与 arr 共享基础数据
                    res = np.array(arr, copy=np._CopyMode.NEVER, dtype=int2)
                    assert res is arr or res.base is arr

                else:
                    # 在类型不同的情况下，需要进行类型转换，断言复制有效
                    for copy in self.false_vals:
                        # 创建一个数据类型为 int2 的数组 res，并断言其为 arr 的复制品，并且具有独立的内存
                        res = np.array(arr, copy=copy, dtype=int2)
                        assert res is not arr and res.flags.owndata
                        assert_array_equal(res, arr)

                    # 如果类型不同且指定了不复制，应该引发 ValueError
                    assert_raises(
                        ValueError, np.array, arr, copy=np._CopyMode.NEVER, dtype=int2
                    )
                    assert_raises(ValueError, np.array, arr, copy=None, dtype=int2)

    def test_buffer_interface(self):
        # Buffer interface gives direct memory access (no copy)
        # 创建一个长度为 10 的数组 arr
        arr = np.arange(10)
        # 创建 arr 的内存视图 view
        view = memoryview(arr)

        # 检查基础数据的共享，使用 may_share_memory 函数
        for copy in self.true_vals:
            # 创建一个从 view 中复制而来的数组 res，并断言其与 arr 不共享内存
            res = np.array(view, copy=copy)
            assert not np.may_share_memory(arr, res)
        for copy in self.false_vals:
            # 创建一个从 view 中复制而来的数组 res，并断言其与 arr 共享内存
            res = np.array(view, copy=copy)
            assert np.may_share_memory(arr, res)
        # 创建一个从 view 中复制而来的数组 res，并断言其与 arr 共享内存
        res = np.array(view, copy=np._CopyMode.NEVER)
        assert np.may_share_memory(arr, res)

    def test_array_interfaces(self):
        # Array interface gives direct memory access (much like a memoryview)
        # 创建一个基础数组 base_arr
        base_arr = np.arange(10)

        # 创建一个类 ArrayLike，其 __array_interface__ 与 base_arr 相同
        class ArrayLike:
            __array_interface__ = base_arr.__array_interface__

        # 创建一个 ArrayLike 的实例 arr
        arr = ArrayLike()

        # 遍历不同的复制模式和基值
        for copy, val in [
            (True, None),
            (np._CopyMode.ALWAYS, None),
            (False, arr),
            (np._CopyMode.IF_NEEDED, arr),
            (np._CopyMode.NEVER, arr),
        ]:
            # 创建一个从 arr 中复制而来的数组 res，并断言其基础是 val
            res = np.array(arr, copy=copy)
            assert res.base is val
    # 定义一个测试方法 test___array__，用于测试 __array__ 方法的行为
    def test___array__(self):
        # 创建一个基础的 numpy 数组，包含从 0 到 9 的整数
        base_arr = np.arange(10)

        # 定义一个类 ArrayLike，实现了 __array__ 方法
        class ArrayLike:
            # 自定义的 __array__ 方法，应返回一个副本，但 numpy 不一定能意识到这一点
            def __array__(self):
                return base_arr

        # 创建 ArrayLike 类的实例 arr
        arr = ArrayLike()

        # 对于每一个真值情况，即 self.true_vals 中的每个值
        for copy in self.true_vals:
            # 调用 np.array 方法，将 arr 转换为 numpy 数组，根据 copy 参数决定是否复制数据
            res = np.array(arr, copy=copy)
            # 使用 assert_array_equal 检查 res 是否与 base_arr 相等
            assert_array_equal(res, base_arr)
            # 在当前情况下，numpy 强制进行额外的复制，因此 res 不会和 base_arr 相同
            assert res is not base_arr

        # 对于每一个假值情况，即 self.false_vals 中的每个值
        for copy in self.false_vals:
            # 调用 np.array 方法，将 arr 转换为 numpy 数组，设置 copy=False，不复制数据
            res = np.array(arr, copy=False)
            # 使用 assert_array_equal 检查 res 是否与 base_arr 相等
            assert_array_equal(res, base_arr)
            # 在这种情况下，numpy 信任 ArrayLike，因此 res 与 base_arr 相同
            assert res is base_arr

        # 使用 pytest.raises 检查是否会抛出 ValueError 异常
        with pytest.raises(ValueError):
            # 调用 np.array 方法，将 arr 转换为 numpy 数组，设置 copy=np._CopyMode.NEVER
            np.array(arr, copy=np._CopyMode.NEVER)

    # 使用 parametrize 装饰器为测试方法 test___array__ 提供参数化测试数据
    @parametrize("arr", [np.ones(()), np.arange(81).reshape((9, 9))])
    @parametrize("order1", ["C", "F", None])
    @parametrize("order2", ["C", "F", "A", "K"])
    # 定义一个测试方法，用于检查数组在不同顺序条件下的行为
    def test_order_mismatch(self, arr, order1, order2):
        # order1 和 order2 是决定复制需求的关键因素之一

        # 根据 order1 复制数组，准备 C-order、F-order 和非连续数组：
        arr = arr.copy(order1)
        if order1 == "C":
            # 如果 order1 是 C，则断言数组是 C-contiguous
            assert arr.flags.c_contiguous
        elif order1 == "F":
            # 如果 order1 是 F，则断言数组是 F-contiguous
            assert arr.flags.f_contiguous
        elif arr.ndim != 0:
            # 如果数组不是零维，使其变成非连续数组
            arr = arr[::2, ::2]
            # 断言数组不是 forc contiguous（这里可能是打字错误，应该是非 contiguous）
            assert not arr.flags.forc

        # 根据 order2 判断是否需要复制数组：
        if order2 == "C":
            no_copy_necessary = arr.flags.c_contiguous
        elif order2 == "F":
            no_copy_necessary = arr.flags.f_contiguous
        else:
            # Keeporder 和 Anyorder 不要求输出是连续的。
            # 这与 `astype` 的行为不一致，后者对于 "A" 要求连续性，这可能是历史遗留问题，当时 "K" 不存在。
            no_copy_necessary = True

        # 对数组和内存视图进行测试
        for view in [arr, memoryview(arr)]:
            for copy in self.true_vals:
                # 创建一个按照 order2 的顺序复制的数组
                res = np.array(view, copy=copy, order=order2)
                # 断言 res 是新创建的数组且拥有数据
                assert res is not arr and res.flags.owndata
                # 断言 arr 和 res 相等
                assert_array_equal(arr, res)

            if no_copy_necessary:
                for copy in self.false_vals:
                    # 创建一个按照 order2 的顺序复制的数组
                    res = np.array(view, copy=copy, order=order2)
                    # 如果不是在 PyPy 下，断言 res 是 arr 或者 res 的基础对象是 arr 的内存视图
                    if not IS_PYPY:
                        assert res is arr or res.base.obj is arr

                # 使用 NEVER 模式创建数组，如果不是在 PyPy 下，断言 res 是 arr 或者 res 的基础对象是 arr 的内存视图
                res = np.array(view, copy=np._CopyMode.NEVER, order=order2)
                if not IS_PYPY:
                    assert res is arr or res.base.obj is arr
            else:
                for copy in self.false_vals:
                    # 创建一个按照 order2 的顺序复制的数组
                    res = np.array(arr, copy=copy, order=order2)
                    # 断言 arr 和 res 相等
                    assert_array_equal(arr, res)
                # 如果需要复制，断言调用时会引发 ValueError
                assert_raises(
                    ValueError, np.array, view, copy=np._CopyMode.NEVER, order=order2
                )
                assert_raises(ValueError, np.array, view, copy=None, order=order2)

    # 检查不允许使用 striding 的情况
    def test_striding_not_ok(self):
        arr = np.array([[1, 2, 4], [3, 4, 5]])
        # 断言尝试使用 np.array 创建不允许 striding 的数组时会引发 ValueError
        assert_raises(ValueError, np.array, arr.T, copy=np._CopyMode.NEVER, order="C")
        assert_raises(
            ValueError,
            np.array,
            arr.T,
            copy=np._CopyMode.NEVER,
            order="C",
            dtype=np.int64,
        )
        # 断言尝试使用 np.array 创建不允许 striding 的数组时会引发 ValueError
        assert_raises(ValueError, np.array, arr, copy=np._CopyMode.NEVER, order="F")
        assert_raises(
            ValueError,
            np.array,
            arr,
            copy=np._CopyMode.NEVER,
            order="F",
            dtype=np.int64,
        )
class TestArrayAttributeDeletion(TestCase):
    # 定义测试类 TestArrayAttributeDeletion，继承自 TestCase

    def test_multiarray_writable_attributes_deletion(self):
        # 测试删除多维数组可写属性的行为
        # ticket #2046, 应该不会导致序列错误，而是引发 AttributeError
        a = np.ones(2)
        # 创建一个包含 "shape", "strides", "data", "dtype", "real", "imag", "flat" 的属性列表
        attr = ["shape", "strides", "data", "dtype", "real", "imag", "flat"]
        # 使用 suppress_warnings 上下文管理器，过滤掉 DeprecationWarning 类型的警告
        with suppress_warnings() as sup:
            # 过滤掉 "Assigning the 'data' attribute" 的警告
            sup.filter(DeprecationWarning, "Assigning the 'data' attribute")
            # 遍历属性列表 attr
            for s in attr:
                # 断言删除对象 a 的属性 s 时引发 AttributeError 异常
                assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_not_writable_attributes_deletion(self):
        # 测试删除多维数组不可写属性的行为
        a = np.ones(2)
        # 创建一个包含多个不可写属性的列表
        attr = [
            "ndim",
            "flags",
            "itemsize",
            "size",
            "nbytes",
            "base",
            "ctypes",
            "T",
            "__array_interface__",
            "__array_struct__",
            "__array_priority__",
            "__array_finalize__",
        ]
        # 遍历属性列表 attr
        for s in attr:
            # 断言删除对象 a 的属性 s 时引发 AttributeError 异常
            assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_flags_writable_attribute_deletion(self):
        # 测试删除多维数组 flags 对象可写属性的行为
        a = np.ones(2).flags
        # 创建一个包含可写属性的列表
        attr = ["writebackifcopy", "updateifcopy", "aligned", "writeable"]
        # 遍历属性列表 attr
        for s in attr:
            # 断言删除对象 a 的属性 s 时引发 AttributeError 异常
            assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_flags_not_writable_attribute_deletion(self):
        # 测试删除多维数组 flags 对象不可写属性的行为
        a = np.ones(2).flags
        # 创建一个包含不可写属性的列表
        attr = [
            "contiguous",
            "c_contiguous",
            "f_contiguous",
            "fortran",
            "owndata",
            "fnc",
            "forc",
            "behaved",
            "carray",
            "farray",
            "num",
        ]
        # 遍历属性列表 attr
        for s in attr:
            # 断言删除对象 a 的属性 s 时引发 AttributeError 异常
            assert_raises(AttributeError, delattr, a, s)


@skip  # not supported, too brittle, too annoying
@instantiate_parametrized_tests
class TestArrayInterface(TestCase):
    # 定义测试类 TestArrayInterface，继承自 TestCase，用于测试数组接口

    class Foo:
        # 定义内部类 Foo

        def __init__(self, value):
            # Foo 类的初始化方法，接受一个 value 参数
            self.value = value
            self.iface = {"typestr": "f8"}
            # 初始化 iface 属性为包含一个键值对 {"typestr": "f8"} 的字典

        def __float__(self):
            # 定义 __float__ 方法，返回 self.value 的浮点数值
            return float(self.value)

        @property
        def __array_interface__(self):
            # 定义属性方法 __array_interface__，返回 self.iface 属性
            return self.iface

    f = Foo(0.5)
    # 创建 Foo 类的实例 f，初始值为 0.5

    @parametrize(
        "val, iface, expected",
        [
            (f, {}, 0.5),
            ([f], {}, [0.5]),
            ([f, f], {}, [0.5, 0.5]),
            (f, {"shape": ()}, 0.5),
            (f, {"shape": None}, TypeError),
            (f, {"shape": (1, 1)}, [[0.5]]),
            (f, {"shape": (2,)}, ValueError),
            (f, {"strides": ()}, 0.5),
            (f, {"strides": (2,)}, ValueError),
            (f, {"strides": 16}, TypeError),
        ],
    )
    # 使用 parametrize 装饰器，定义多个测试参数的组合及其期望值
    # 定义一个测试方法，用于测试标量在数组接口中的处理
    def test_scalar_interface(self, val, iface, expected):
        # 设置接口的类型字符串为双精度浮点数
        self.f.iface = {"typestr": "f8"}
        # 更新接口信息
        self.f.iface.update(iface)
        # 如果支持引用计数，获取双精度浮点数类型的初始引用计数
        if HAS_REFCOUNT:
            pre_cnt = sys.getrefcount(np.dtype("f8"))
        # 如果期望结果是一个类型，则断言数组化val会引发该类型的异常
        if isinstance(expected, type):
            assert_raises(expected, np.array, val)
        else:
            # 将val转换为NumPy数组
            result = np.array(val)
            # 断言数组化后的结果与期望值相等
            assert_equal(np.array(val), expected)
            # 断言结果的数据类型为双精度浮点数
            assert result.dtype == "f8"
            # 删除结果变量，释放其占用的内存
            del result
        # 如果支持引用计数，获取双精度浮点数类型的最终引用计数
        if HAS_REFCOUNT:
            post_cnt = sys.getrefcount(np.dtype("f8"))
            # 断言初始引用计数与最终引用计数相等
            assert_equal(pre_cnt, post_cnt)
class TestDelMisc(TestCase):
    @xpassIfTorchDynamo  # 定义装饰器 xpassIfTorchDynamo，用于标记该测试用例在特定条件下跳过
    def test_flat_element_deletion(self):
        it = np.ones(3).flat  # 创建一个 numpy 数组的迭代器对象 it，包含三个元素的全为 1 的数组
        try:
            del it[1]  # 尝试删除迭代器中索引为 1 的元素，但由于迭代器不支持此操作，抛出 TypeError 异常
            del it[1:2]  # 尝试删除迭代器中索引从 1 到 2 的元素范围，同样抛出 TypeError 异常
        except TypeError:
            pass  # 捕获 TypeError 异常并忽略
        except Exception:
            raise AssertionError from None  # 捕获其它异常，并抛出 AssertionError 异常，但不包含原异常信息


class TestConversion(TestCase):
    def test_array_scalar_relational_operation(self):
        # All integer
        for dt1 in np.typecodes["AllInteger"]:  # 遍历 numpy 所有整数类型编码
            assert_(1 > np.array(0, dtype=dt1), f"type {dt1} failed")  # 断言 1 大于以 dt1 类型创建的数组中的 0
            assert_(not 1 < np.array(0, dtype=dt1), f"type {dt1} failed")  # 断言 1 不小于以 dt1 类型创建的数组中的 0

            for dt2 in np.typecodes["AllInteger"]:  # 嵌套遍历 numpy 所有整数类型编码
                assert_(
                    np.array(1, dtype=dt1) > np.array(0, dtype=dt2),
                    f"type {dt1} and {dt2} failed",  # 断言以 dt1 类型创建的数组中的 1 大于以 dt2 类型创建的数组中的 0
                )
                assert_(
                    not np.array(1, dtype=dt1) < np.array(0, dtype=dt2),
                    f"type {dt1} and {dt2} failed",  # 断言以 dt1 类型创建的数组中的 1 不小于以 dt2 类型创建的数组中的 0
                )

        # Unsigned integers
        for dt1 in "B":  # 遍历无符号整数类型编码 "B"
            assert_(-1 < np.array(1, dtype=dt1), f"type {dt1} failed")  # 断言 -1 小于以 dt1 类型创建的数组中的 1
            assert_(not -1 > np.array(1, dtype=dt1), f"type {dt1} failed")  # 断言 -1 不大于以 dt1 类型创建的数组中的 1
            assert_(-1 != np.array(1, dtype=dt1), f"type {dt1} failed")  # 断言 -1 不等于以 dt1 类型创建的数组中的 1

            # Unsigned vs signed
            for dt2 in "bhil":  # 嵌套遍历有符号整数类型编码 "bhil"
                assert_(
                    np.array(1, dtype=dt1) > np.array(-1, dtype=dt2),
                    f"type {dt1} and {dt2} failed",  # 断言以 dt1 类型创建的数组中的 1 大于以 dt2 类型创建的数组中的 -1
                )
                assert_(
                    not np.array(1, dtype=dt1) < np.array(-1, dtype=dt2),
                    f"type {dt1} and {dt2} failed",  # 断言以 dt1 类型创建的数组中的 1 不小于以 dt2 类型创建的数组中的 -1
                )
                assert_(
                    np.array(1, dtype=dt1) != np.array(-1, dtype=dt2),
                    f"type {dt1} and {dt2} failed",  # 断言以 dt1 类型创建的数组中的 1 不等于以 dt2 类型创建的数组中的 -1
                )

        # Signed integers and floats
        for dt1 in "bhl" + np.typecodes["Float"]:  # 遍历有符号整数和浮点数类型编码
            assert_(1 > np.array(-1, dtype=dt1), f"type {dt1} failed")  # 断言 1 大于以 dt1 类型创建的数组中的 -1
            assert_(not 1 < np.array(-1, dtype=dt1), f"type {dt1} failed")  # 断言 1 不小于以 dt1 类型创建的数组中的 -1
            assert_(-1 == np.array(-1, dtype=dt1), f"type {dt1} failed")  # 断言 -1 等于以 dt1 类型创建的数组中的 -1

            for dt2 in "bhl" + np.typecodes["Float"]:  # 嵌套遍历有符号整数和浮点数类型编码
                assert_(
                    np.array(1, dtype=dt1) > np.array(-1, dtype=dt2),
                    f"type {dt1} and {dt2} failed",  # 断言以 dt1 类型创建的数组中的 1 大于以 dt2 类型创建的数组中的 -1
                )
                assert_(
                    not np.array(1, dtype=dt1) < np.array(-1, dtype=dt2),
                    f"type {dt1} and {dt2} failed",  # 断言以 dt1 类型创建的数组中的 1 不小于以 dt2 类型创建的数组中的 -1
                )
                assert_(
                    np.array(-1, dtype=dt1) == np.array(-1, dtype=dt2),
                    f"type {dt1} and {dt2} failed",  # 断言以 dt1 类型创建的数组中的 -1 等于以 dt2 类型创建的数组中的 -1
                )

    @skip(reason="object arrays")  # 跳过当前测试用例，原因为 "object arrays"
    # 定义测试方法，验证将 numpy 数组转换为布尔值的行为是否符合预期
    def test_to_bool_scalar(self):
        # 断言将包含一个 False 值的 numpy 数组转换为布尔值应为 False
        assert_equal(bool(np.array([False])), False)
        # 断言将包含一个 True 值的 numpy 数组转换为布尔值应为 True
        assert_equal(bool(np.array([True])), True)
        # 断言将包含一个二维数组，包含数值 42 的 numpy 数组转换为布尔值应为 True
        assert_equal(bool(np.array([[42]])), True)
        # 断言尝试将包含多个值的 numpy 数组转换为布尔值会引发 ValueError 异常
        assert_raises(ValueError, bool, np.array([1, 2]))

        # 定义一个无法进行布尔转换的类 NotConvertible
        class NotConvertible:
            def __bool__(self):
                raise NotImplementedError

        # 断言尝试将 NotConvertible 实例转换为布尔值会引发 NotImplementedError 异常
        assert_raises(NotImplementedError, bool, np.array(NotConvertible()))
        # 断言尝试将包含 NotConvertible 实例的 numpy 数组转换为布尔值会引发 NotImplementedError 异常
        assert_raises(NotImplementedError, bool, np.array([NotConvertible()]))
        
        # 如果运行环境是 Pyston，则跳过递归检查的测试
        if IS_PYSTON:
            raise SkipTest("Pyston disables recursion checking")

        # 创建一个包含自身引用的 numpy 数组
        self_containing = np.array([None])
        self_containing[0] = self_containing

        # 定义 RecursionError 为 Error 变量
        Error = RecursionError

        # 断言尝试将包含自身引用的 numpy 数组转换为布尔值会引发 RecursionError 异常
        assert_raises(Error, bool, self_containing)  # previously stack overflow
        # 解除包含自身引用的 numpy 数组的循环引用
        self_containing[0] = None  # resolve circular reference

    # 定义测试方法，验证将 numpy 数组转换为整数的行为是否符合预期
    def test_to_int_scalar(self):
        # 定义 int 和 lambda 函数为 int_func，用于测试
        int_funcs = (int, lambda x: x.__int__())
        for int_func in int_funcs:
            # 断言将包含数值 0 的 numpy 数组转换为整数应为 0
            assert_equal(int_func(np.array(0)), 0)
            # 断言将包含数值 1 的 numpy 数组转换为整数应为 1
            assert_equal(int_func(np.array([1])), 1)
            # 断言将包含一个二维数组，包含数值 42 的 numpy 数组转换为整数应为 42
            assert_equal(int_func(np.array([[42]])), 42)
            # 断言尝试将包含多个值的 numpy 数组转换为整数会引发 ValueError 或 TypeError 异常
            assert_raises((ValueError, TypeError), int_func, np.array([1, 2]))

    # 标记为跳过的测试方法，理由是测试对象数组的情况
    @skip(reason="object arrays")
    def test_to_int_scalar_2(self):
        # 定义 int 和 lambda 函数为 int_func，用于测试
        int_funcs = (int, lambda x: x.__int__())
        for int_func in int_funcs:
            # 断言将字符串 "4" 的 numpy 数组转换为整数应为 4
            assert_equal(4, int_func(np.array("4")))
            # 断言将字节串 b"5" 的 numpy 数组转换为整数应为 5
            assert_equal(5, int_func(np.bytes_(b"5")))
            # 断言将 Unicode 字符串 "6" 的 numpy 数组转换为整数应为 6
            assert_equal(6, int_func(np.unicode_("6")))

            # 如果 Python 版本低于 3.11，测试将 int() 委托给 __trunc__ 方法的行为
            if sys.version_info < (3, 11):

                # 定义一个具有 __trunc__ 方法的类 HasTrunc
                class HasTrunc:
                    def __trunc__(self):
                        return 3

                # 断言将包含 HasTrunc 实例的 numpy 数组转换为整数应为 3
                assert_equal(3, int_func(np.array(HasTrunc())))
                # 断言将包含 HasTrunc 实例的 numpy 数组转换为整数应为 3
                assert_equal(3, int_func(np.array([HasTrunc()])))
            else:
                pass

            # 定义一个无法进行整数转换的类 NotConvertible
            class NotConvertible:
                def __int__(self):
                    raise NotImplementedError

            # 断言尝试将 NotConvertible 实例转换为整数会引发 NotImplementedError 异常
            assert_raises(NotImplementedError, int_func, np.array(NotConvertible()))
            # 断言尝试将包含 NotConvertible 实例的 numpy 数组转换为整数会引发 NotImplementedError 异常
            assert_raises(NotImplementedError, int_func, np.array([NotConvertible()]))
# 定义一个名为 TestWhere 的测试类，继承自 TestCase 类
class TestWhere(TestCase):
    # 定义一个名为 test_basic 的测试方法
    def test_basic(self):
        # 定义一个包含不同数据类型的列表
        dts = [bool, np.int16, np.int32, np.int64, np.double, np.complex128]
        # 遍历数据类型列表中的每个数据类型
        for dt in dts:
            # 创建一个长度为 53 的布尔数组 c，元素全部为 True
            c = np.ones(53, dtype=bool)
            # 断言 np.where(c, dt(0), dt(1)) 返回的结果等于 dt(0)
            assert_equal(np.where(c, dt(0), dt(1)), dt(0))
            # 断言 np.where(~c, dt(0), dt(1)) 返回的结果等于 dt(1)
            assert_equal(np.where(~c, dt(0), dt(1)), dt(1))
            # 断言 np.where(True, dt(0), dt(1)) 返回的结果等于 dt(0)
            assert_equal(np.where(True, dt(0), dt(1)), dt(0))
            # 断言 np.where(False, dt(0), dt(1)) 返回的结果等于 dt(1)
            assert_equal(np.where(False, dt(0), dt(1)), dt(1))
            # 创建一个与 c 形状相同的由 dt 类型元素组成的数组 d
            d = np.ones_like(c).astype(dt)
            # 创建一个与 d 形状相同的零数组 e
            e = np.zeros_like(d)
            # 创建一个与 d 类型相同的数组 r，内容为 d 的拷贝
            r = d.astype(dt)
            # 将 c 的第 7 个元素设置为 False
            c[7] = False
            # 将 r 的第 7 个元素设置为 e 的第 7 个元素（即 0）
            r[7] = e[7]
            # 断言 np.where(c, e, e) 返回的结果等于 e
            assert_equal(np.where(c, e, e), e)
            # 断言 np.where(c, d, e) 返回的结果等于 r
            assert_equal(np.where(c, d, e), r)
            # 断言 np.where(c, d, e[0]) 返回的结果等于 r
            assert_equal(np.where(c, d, e[0]), r)
            # 断言 np.where(c, d[0], e) 返回的结果等于 r
            assert_equal(np.where(c, d[0], e), r)
            # 断言 np.where(c[::2], d[::2], e[::2]) 返回的结果等于 r[::2]
            assert_equal(np.where(c[::2], d[::2], e[::2]), r[::2])
            # 断言 np.where(c[1::2], d[1::2], e[1::2]) 返回的结果等于 r[1::2]
            assert_equal(np.where(c[1::2], d[1::2], e[1::2]), r[1::2])
            # 断言 np.where(c[::3], d[::3], e[::3]) 返回的结果等于 r[::3]
            assert_equal(np.where(c[::3], d[::3], e[::3]), r[::3])
            # 断言 np.where(c[1::3], d[1::3], e[1::3]) 返回的结果等于 r[1::3]
            assert_equal(np.where(c[1::3], d[1::3], e[1::3]), r[1::3])
        # 注释掉以下两行代码，因为它们引发 IndexError 异常，暂不执行
        # assert_equal(np.where(c[::-2], d[::-2], e[::-2]), r[::-2])
        # assert_equal(np.where(c[::-3], d[::-3], e[::-3]), r[::-3])

    # 定义一个名为 test_exotic 的测试方法
    def test_exotic(self):
        # 创建一个形状为 (0, 3) 的空布尔数组 m
        m = np.array([], dtype=bool).reshape(0, 3)
        # 创建一个形状为 (0, 3) 的空 np.float64 数组 b
        b = np.array([], dtype=np.float64).reshape(0, 3)
        # 断言 np.where(m, 0, b) 返回的结果等于形状为 (0, 3) 的空数组
        assert_array_equal(np.where(m, 0, b), np.array([]).reshape(0, 3))

    # 使用 skip 装饰器标记该测试方法跳过执行，原因是处理对象数组
    @skip(reason="object arrays")
    # 定义测试方法，测试 np.where 函数在特定情况下的行为
    def test_exotic_2(self):
        # 创建包含浮点数的 numpy 数组 d
        d = np.array(
            [
                -1.34,
                -0.16,
                -0.54,
                -0.31,
                -0.08,
                -0.95,
                0.000,
                0.313,
                0.547,
                -0.18,
                0.876,
                0.236,
                1.969,
                0.310,
                0.699,
                1.013,
                1.267,
                0.229,
                -1.39,
                0.487,
            ]
        )
        # 创建包含字符串和 NaN 的 numpy 数组 e
        nan = float("NaN")
        e = np.array(
            [
                "5z",
                "0l",
                nan,
                "Wz",
                nan,
                nan,
                "Xq",
                "cs",
                nan,
                nan,
                "QN",
                nan,
                nan,
                "Fd",
                nan,
                nan,
                "kp",
                nan,
                "36",
                "i1",
            ],
            dtype=object,
        )
        # 创建布尔类型的 numpy 数组 m
        m = np.array(
            [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], dtype=bool
        )

        # 复制 e 数组到 r
        r = e[:]
        # 根据 m 数组的 True 值，将 d 数组的对应元素复制到 r 中
        r[np.where(m)] = d[np.where(m)]
        # 断言 np.where 函数按预期返回的结果
        assert_array_equal(np.where(m, d, e), r)

        # 再次复制 e 数组到 r
        r = e[:]
        # 根据 m 数组的 False 值，将 d 数组的对应元素复制到 r 中
        r[np.where(~m)] = d[np.where(~m)]
        # 断言 np.where 函数按预期返回的结果
        assert_array_equal(np.where(m, e, d), r)

        # 断言 np.where(m, e, e) 返回的结果与 e 数组相同
        assert_array_equal(np.where(m, e, e), e)

        # 测试最小的 dtype 结果和 NaN 标量（例如 pandas 所需的情况）
        d = np.array([1.0, 2.0], dtype=np.float32)
        e = float("NaN")
        # 断言 np.where 函数返回的结果 dtype 为 np.float32
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float("Infinity")
        # 断言 np.where 函数返回的结果 dtype 为 np.float32
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float("-Infinity")
        # 断言 np.where 函数返回的结果 dtype 为 np.float32
        assert_equal(np.where(True, d, e).dtype, np.float32)
        # 还要检查向上转换的情况
        e = 1e150
        # 断言 np.where 函数返回的结果 dtype 为 np.float64
        assert_equal(np.where(True, d, e).dtype, np.float64)

    # 定义测试方法，测试 np.where 函数在多维数组上的行为
    def test_ndim(self):
        c = [True, False]
        # 创建形状为 (2, 25) 的全零数组 a 和全一数组 b
        a = np.zeros((2, 25))
        b = np.ones((2, 25))
        # 使用 np.where 在 c 数组的基础上，选择 a 或 b 中的对应元素形成结果数组 r
        r = np.where(np.array(c)[:, np.newaxis], a, b)
        # 断言结果数组 r 的第一行与数组 a 的第一行相等
        assert_array_equal(r[0], a[0])
        # 断言结果数组 r 的第二行与数组 b 的第一行相等
        assert_array_equal(r[1], b[0])

        # 将数组 a 和 b 转置
        a = a.T
        b = b.T
        # 使用 np.where 在 c 数组的基础上，选择 a 或 b 中的对应元素形成结果数组 r
        r = np.where(c, a, b)
        # 断言结果数组 r 的第一列与数组 a 的第一列相等
        assert_array_equal(r[:, 0], a[:, 0])
        # 断言结果数组 r 的第二列与数组 b 的第一列相等
        assert_array_equal(r[:, 1], b[:, 0])
    # 定义测试函数，测试混合数据类型的情况
    def test_dtype_mix(self):
        # 创建布尔数组 c
        c = np.array(
            [
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ]
        )
        # 创建标量 a，使用 np.uint8 类型
        a = np.uint8(1)
        # 创建数组 b，指定 dtype 为 np.float64
        b = np.array(
            [5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0],
            dtype=np.float64,
        )
        # 创建数组 r，指定 dtype 为 np.float64
        r = np.array(
            [5.0, 1.0, 3.0, 2.0, -1.0, -4.0, 1.0, -10.0, 10.0, 1.0, 1.0, 3.0],
            dtype=np.float64,
        )
        # 断言 np.where 函数的返回结果与 r 相等
        assert_equal(np.where(c, a, b), r)

        # 将 a 的数据类型转换为 np.float32
        a = a.astype(np.float32)
        # 将 b 的数据类型转换为 np.int64
        b = b.astype(np.int64)
        # 再次断言 np.where 函数的返回结果与 r 相等
        assert_equal(np.where(c, a, b), r)

        # 将 c 的数据类型转换为 int
        c = c.astype(int)
        # 修改 c 中非零元素的值为 34242324
        c[c != 0] = 34242324
        # 再次断言 np.where 函数的返回结果与 r 相等
        assert_equal(np.where(c, a, b), r)
        
        # 反转 c 数组的值
        tmpmask = c != 0
        c[c == 0] = 41247212
        c[tmpmask] = 0
        # 再次断言 np.where 函数的返回结果与 r 相等
        assert_equal(np.where(c, b, a), r)

    # 跳过测试的装饰器函数，理由是 "endianness"
    @skip(reason="endianness")
    def test_foreign(self):
        # 创建布尔数组 c
        c = np.array(
            [
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ]
        )
        # 创建数组 r，指定 dtype 为 np.float64
        r = np.array(
            [5.0, 1.0, 3.0, 2.0, -1.0, -4.0, 1.0, -10.0, 10.0, 1.0, 1.0, 3.0],
            dtype=np.float64,
        )
        # 创建包含一个元素的数组 a，数据类型为 ">i4"
        a = np.ones(1, dtype=">i4")
        # 创建数组 b，指定 dtype 为 np.float64
        b = np.array(
            [5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0],
            dtype=np.float64,
        )
        # 断言 np.where 函数的返回结果与 r 相等
        assert_equal(np.where(c, a, b), r)

        # 将 b 的数据类型转换为 ">f8"
        b = b.astype(">f8")
        # 再次断言 np.where 函数的返回结果与 r 相等
        assert_equal(np.where(c, a, b), r)

        # 将 a 的数据类型转换为 "<i4"
        a = a.astype("<i4")
        # 再次断言 np.where 函数的返回结果与 r 相等
        assert_equal(np.where(c, a, b), r)

        # 将 c 的数据类型转换为 ">i4"
        c = c.astype(">i4")
        # 再次断言 np.where 函数的返回结果与 r 相等
        assert_equal(np.where(c, a, b), r)

    # 测试错误情况的函数
    def test_error(self):
        # 创建包含两个元素的布尔列表 c
        c = [True, True]
        # 创建 4x5 全为 1 的数组 a
        a = np.ones((4, 5))
        # 创建 5x5 全为 1 的数组 b
        b = np.ones((5, 5))
        # 断言 np.where 函数会引发 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.where, c, a, a)
        # 断言 np.where 函数会引发 RuntimeError 或 ValueError 异常
        assert_raises((RuntimeError, ValueError), np.where, c[0], a, b)

    # 测试空结果情况的函数
    def test_empty_result(self):
        # 创建一个 1x1 全为 0 的数组 x
        x = np.zeros((1, 1))
        # 通过赋值操作传递空的 where 结果，这可以通过 valgrind 检测到错误，参见 gh-8922
        ibad = np.vstack(np.where(x == 99.0))
        # 断言 ibad 等于一个至少有两个维度的数组，其数据类型为 np.intp
        assert_array_equal(ibad, np.atleast_2d(np.array([[], []], dtype=np.intp)))

    # 测试大维度数组的函数
    def test_largedim(self):
        # 定义数组的形状为 [10, 2, 3, 4, 5, 6]
        shape = [10, 2, 3, 4, 5, 6]
        # 设置随机种子为 2
        np.random.seed(2)
        # 创建随机数组 array，形状为 shape
        array = np.random.rand(*shape)

        # 循环 10 次
        for i in range(10):
            # 使用 array 的 nonzero 方法创建 benchmark
            benchmark = array.nonzero()
            # 使用 array 的 nonzero 方法创建 result
            result = array.nonzero()
            # 断言 benchmark 等于 result
            assert_array_equal(benchmark, result)
    # 定义一个测试方法，用于测试 np.where 函数的关键字参数
    def test_kwargs(self):
        # 创建一个包含一个元素的零数组
        a = np.zeros(1)
        # 使用 assert_raises 上下文管理器来检查是否抛出 TypeError 异常
        with assert_raises(TypeError):
            # 调用 np.where 函数，传入数组 a，并指定 x 和 y 关键字参数为数组 a 自身
            np.where(a, x=a, y=a)
class TestHashing(TestCase):
    # 哈希测试类

    def test_arrays_not_hashable(self):
        # 测试数组不可哈希化
        x = np.ones(3)
        assert_raises(TypeError, hash, x)

    def test_collections_hashable(self):
        # 测试集合可哈希化
        x = np.array([])
        assert_(not isinstance(x, collections.abc.Hashable))


class TestFormat(TestCase):
    # 格式化测试类

    @xpassIfTorchDynamo  # (reason="TODO")
    def test_0d(self):
        # 测试0维情况
        a = np.array(np.pi)
        assert_equal(f"{a:0.3g}", "3.14")
        assert_equal(f"{a[()]:0.3g}", "3.14")

    def test_1d_no_format(self):
        # 测试1维情况（无格式化）
        a = np.array([np.pi])
        assert_equal(f"{a}", str(a))

    def test_1d_format(self):
        # 测试1维情况（格式化）
        # 直到gh-5543，确保行为与之前一致
        a = np.array([np.pi])
        assert_raises(TypeError, "{:30}".format, a)


from numpy.testing import IS_PYPY


class TestWritebackIfCopy(TestCase):
    # 使用WRITEBACKIFCOPY机制的测试类

    def test_argmax_with_out(self):
        # 测试带有输出参数的argmax函数
        mat = np.eye(5)
        out = np.empty(5, dtype="i2")
        res = np.argmax(mat, 0, out=out)
        assert_equal(res, range(5))

    def test_argmin_with_out(self):
        # 测试带有输出参数的argmin函数
        mat = -np.eye(5)
        out = np.empty(5, dtype="i2")
        res = np.argmin(mat, 0, out=out)
        assert_equal(res, range(5))

    @xpassIfTorchDynamo  # (reason="XXX: place()")
    def test_insert_noncontiguous(self):
        # 测试插入非连续数组的情况
        a = np.arange(6).reshape(2, 3).T  # 强制非C连续
        # 使用arr_insert
        np.place(a, a > 2, [44, 55])
        assert_equal(a, np.array([[0, 44], [1, 55], [2, 44]]))
        # 触发一个失败路径
        assert_raises(ValueError, np.place, a, a > 20, [])

    def test_put_noncontiguous(self):
        # 测试放置非连续数组的情况
        a = np.arange(6).reshape(2, 3).T  # 强制非C连续
        assert not a.flags["C_CONTIGUOUS"]  # 检查一致性
        np.put(a, [0, 2], [44, 55])
        assert_equal(a, np.array([[44, 3], [55, 4], [2, 5]]))

    @xpassIfTorchDynamo  # (reason="XXX: putmask()")
    def test_putmask_noncontiguous(self):
        # 测试放置掩码到非连续数组的情况
        a = np.arange(6).reshape(2, 3).T  # 强制非C连续
        # 使用arr_putmask
        np.putmask(a, a > 2, a**2)
        assert_equal(a, np.array([[0, 9], [1, 16], [2, 25]]))

    def test_take_mode_raise(self):
        # 测试使用raise模式的take函数
        a = np.arange(6, dtype="int")
        out = np.empty(2, dtype="int")
        np.take(a, [0, 2], out=out, mode="raise")
        assert_equal(out, np.array([0, 2]))

    def test_choose_mod_raise(self):
        # 测试使用raise模式的choose函数
        a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        out = np.empty((3, 3), dtype="int")
        choices = [-10, 10]
        np.choose(a, choices, out=out, mode="raise")
        assert_equal(out, np.array([[10, -10, 10], [-10, 10, -10], [10, -10, 10]]))

    @xpassIfTorchDynamo  # (reason="XXX: ndarray.flat")
    def test_flatiter__array__(self):
        # 测试flatiter的数组情况
        a = np.arange(9).reshape(3, 3)
        b = a.T.flat
        c = b.__array__()
        # 触发WRITEBACKIFCOPY解析，假设引用计数语义
        del c
    def test_dot_out(self):
        # 定义测试方法 test_dot_out，用于测试 np.dot 函数的 out 参数功能
        
        # 创建一个 3x3 的浮点数数组 a，数组元素从 0 到 8
        a = np.arange(9, dtype=float).reshape(3, 3)
        
        # 使用 np.dot 函数计算数组 a 与自身的矩阵乘法，将结果保存到数组 a 中
        b = np.dot(a, a, out=a)
        
        # 断言 b 的结果与给定的数组相等
        assert_equal(b, np.array([[15, 18, 21], [42, 54, 66], [69, 90, 111]]))
# 实例化参数化测试类
@instantiate_parametrized_tests
class TestArange(TestCase):
    # 测试无限范围的情况
    def test_infinite(self):
        # 断言抛出异常(RuntimeError, ValueError)，调用 np.arange 函数，传入参数 0, np.inf
        assert_raises(
            (RuntimeError, ValueError), np.arange, 0, np.inf  # "unsupported range",
        )

    # 测试步长为 NaN 的情况
    def test_nan_step(self):
        # 断言抛出异常(RuntimeError, ValueError)，调用 np.arange 函数，传入参数 0, 1, np.nan
        assert_raises(
            (RuntimeError, ValueError),  # "cannot compute length",
            np.arange,
            0,
            1,
            np.nan,
        )

    # 测试步长为 0 的情况
    def test_zero_step(self):
        # 断言抛出 ZeroDivisionError 异常，调用 np.arange 函数，传入参数 0, 10, 0
        assert_raises(ZeroDivisionError, np.arange, 0, 10, 0)
        # 断言抛出 ZeroDivisionError 异常，调用 np.arange 函数，传入参数 0.0, 10.0, 0.0
        assert_raises(ZeroDivisionError, np.arange, 0.0, 10.0, 0.0)

        # 空范围
        # 断言抛出 ZeroDivisionError 异常，调用 np.arange 函数，传入参数 0, 0, 0
        assert_raises(ZeroDivisionError, np.arange, 0, 0, 0)
        # 断言抛出 ZeroDivisionError 异常，调用 np.arange 函数，传入参数 0.0, 0.0, 0.0
        assert_raises(ZeroDivisionError, np.arange, 0.0, 0.0, 0.0)

    # 测试需要范围的情况
    def test_require_range(self):
        # 断言抛出 TypeError 异常，调用 np.arange 函数，不传入任何参数
        assert_raises(TypeError, np.arange)
        # 断言抛出 TypeError 异常，调用 np.arange 函数，传入参数 step=3
        assert_raises(TypeError, np.arange, step=3)
        # 断言抛出 TypeError 异常，调用 np.arange 函数，传入参数 dtype="int64"
        assert_raises(TypeError, np.arange, dtype="int64")

    @xpassIfTorchDynamo  # (reason="weird arange signature (optionals before required args)")
    def test_require_range_2(self):
        # 断言抛出 TypeError 异常，调用 np.arange 函数，传入参数 start=4
        assert_raises(TypeError, np.arange, start=4)

    # 测试使用关键字参数 start 和 stop 的情况
    def test_start_stop_kwarg(self):
        # 使用关键字参数 stop 调用 np.arange 函数
        keyword_stop = np.arange(stop=3)
        # 使用关键字参数 start 和 stop 调用 np.arange 函数
        keyword_zerotostop = np.arange(start=0, stop=3)
        # 使用关键字参数 start 和 stop 调用 np.arange 函数
        keyword_start_stop = np.arange(start=3, stop=9)

        # 断言关键字参数 stop 返回的数组长度为 3
        assert len(keyword_stop) == 3
        # 断言关键字参数 start 和 stop 返回的数组长度为 3
        assert len(keyword_zerotostop) == 3
        # 断言关键字参数 start 和 stop 返回的数组长度为 6
        assert len(keyword_start_stop) == 6
        # 断言关键字参数 stop 和 keyword_zerotostop 返回的数组相等
        assert_array_equal(keyword_stop, keyword_zerotostop)

    @skip(reason="arange for booleans: numpy maybe deprecates?")
    def test_arange_booleans(self):
        # 对布尔值进行 arange 操作，长度最多为 2
        # 但是 `arange(2, 4, dtype=bool)` 的操作很奇怪
        # 可能大部分或全部都会被弃用/移除
        # 使用 dtype=bool 调用 np.arange 函数
        res = np.arange(False, dtype=bool)
        # 断言返回的数组为空，数据类型为 bool
        assert_array_equal(res, np.array([], dtype="bool"))

        # 使用 dtype=bool 调用 np.arange 函数
        res = np.arange(True, dtype="bool")
        # 断言返回的数组为 [False]
        assert_array_equal(res, [False])

        # 使用 dtype=bool 调用 np.arange 函数
        res = np.arange(2, dtype="bool")
        # 断言返回的数组为 [False, True]
        assert_array_equal(res, [False, True])

        # 这种情况特别奇怪，但没有特殊情况会被忽略：
        # 使用 dtype=bool 调用 np.arange 函数
        res = np.arange(6, 8, dtype="bool")
        # 断言返回的数组为 [True, True]
        assert_array_equal(res, [True, True])

        # 使用 pytest.raises 断言抛出 TypeError 异常
        with pytest.raises(TypeError):
            np.arange(3, dtype="bool")

    @parametrize("which", [0, 1, 2])
    def test_error_paths_and_promotion(self, which):
        args = [0, 1, 2]  # start, stop, and step
        args[which] = np.float64(2.0)  # should ensure float64 output
        # 断言 np.arange 函数返回的数组数据类型为 float64
        assert np.arange(*args).dtype == np.float64

        # 重复非空范围
        args = [0, 8, 2]
        args[which] = np.float64(2.0)
        # 断言 np.arange 函数返回的数组数据类型为 float64
        assert np.arange(*args).dtype == np.float64

    @parametrize("dt", [np.float32, np.uint8, complex])
    def test_explicit_dtype(self, dt):
        # 断言 np.arange 函数返回的数组数据类型为指定的 dt
        assert np.arange(5.0, dtype=dt).dtype == dt


class TestRichcompareScalar(TestCase):
    @skip  # XXX: 脆弱，根据 NumPy 版本可能失败或通过
    def test_richcompare_scalar_boolean_singleton_return(self):
        # 当前确保返回布尔单例，但返回 NumPy 布尔值也可以接受：
        # 断言数组中的元素和字符串 "a" 的比较结果为 False
        assert (np.array(0) == "a") is False
        # 断言数组中的元素和字符串 "a" 的比较结果为 True
        assert (np.array(0) != "a") is True
        # 断言 np.int16 类型的 0 和字符串 "a" 的比较结果为 False
        assert (np.int16(0) == "a") is False
        # 断言 np.int16 类型的 0 和字符串 "a" 的比较结果为 True
        assert (np.int16(0) != "a") is True
@skip  # (reason="implement views/dtypes")
# 装饰器，用于跳过测试，原因是实现视图和数据类型相关功能尚未完成

class TestViewDtype(TestCase):
    """
    Verify that making a view of a non-contiguous array works as expected.
    """

    def test_smaller_dtype_multiple(self):
        # x is non-contiguous
        # 创建一个非连续的一维数组 x，元素类型为 "<i4"，步长为 2
        x = np.arange(10, dtype="<i4")[::2]
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="the last axis must be contiguous"):
            x.view("<i2")
        # 预期的结果数组
        expected = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]]
        # 使用 assert_array_equal 断言两个数组是否相等
        assert_array_equal(x[:, np.newaxis].view("<i2"), expected)

    def test_smaller_dtype_not_multiple(self):
        # x is non-contiguous
        # 创建一个非连续的一维数组 x，元素类型为 "<i4"，步长为 2
        x = np.arange(5, dtype="<i4")[::2]

        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="the last axis must be contiguous"):
            x.view("S3")
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="When changing to a smaller dtype"):
            x[:, np.newaxis].view("S3")

        # 预期的结果数组
        expected = [[b""], [b"\x02"], [b"\x04"]]
        # 使用 assert_array_equal 断言两个数组是否相等
        assert_array_equal(x[:, np.newaxis].view("S4"), expected)

    def test_larger_dtype_multiple(self):
        # x is non-contiguous in the first dimension, contiguous in the last
        # 创建一个非连续的二维数组 x，元素类型为 "<i2"，形状为 (10, 2)，每隔一行取一行
        x = np.arange(20, dtype="<i2").reshape(10, 2)[::2, :]
        # 预期的结果数组，转换成 "<i4" 类型
        expected = np.array(
            [[65536], [327684], [589832], [851980], [1114128]], dtype="<i4"
        )
        # 使用 assert_array_equal 断言两个数组是否相等
        assert_array_equal(x.view("<i4"), expected)

    def test_larger_dtype_not_multiple(self):
        # x is non-contiguous in the first dimension, contiguous in the last
        # 创建一个非连续的二维数组 x，元素类型为 "<i2"，形状为 (10, 2)，每隔一行取一行
        x = np.arange(20, dtype="<i2").reshape(10, 2)[::2, :]
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="When changing to a larger dtype"):
            x.view("S3")
        # 预期的结果数组
        expected = [
            [b"\x00\x00\x01"],
            [b"\x04\x00\x05"],
            [b"\x08\x00\t"],
            [b"\x0c\x00\r"],
            [b"\x10\x00\x11"],
        ]
        # 使用 assert_array_equal 断言两个数组是否相等
        assert_array_equal(x.view("S4"), expected)

    def test_f_contiguous(self):
        # x is F-contiguous
        # 创建一个 F 连续的二维数组 x，元素类型为 "<i4"，形状为 (3, 4)
        x = np.arange(4 * 3, dtype="<i4").reshape(4, 3).T
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="the last axis must be contiguous"):
            x.view("<i2")

    def test_non_c_contiguous(self):
        # x is contiguous in axis=-1, but not C-contiguous in other axes
        # 创建一个在 axis=-1 连续但不在 C 方向上连续的三维数组 x，元素类型为 "i1"，形状为 (2, 3, 4)
        x = np.arange(2 * 3 * 4, dtype="i1").reshape(2, 3, 4).transpose(1, 0, 2)
        # 预期的结果数组，转换成 "<i2" 类型
        expected = [
            [[256, 770], [3340, 3854]],
            [[1284, 1798], [4368, 4882]],
            [[2312, 2826], [5396, 5910]],
        ]
        # 使用 assert_array_equal 断言两个数组是否相等
        assert_array_equal(x.view("<i2"), expected)


@instantiate_parametrized_tests
# 装饰器，用于实例化参数化测试
class TestSortFloatMisc(TestCase):
    # Test various array sizes that hit different code paths in quicksort-avx512
    @parametrize(
        "N", [8, 16, 24, 32, 48, 64, 96, 128, 151, 191, 256, 383, 512, 1023, 2047]
    )
    # 定义一个测试方法，用于测试对浮点数数组的排序
    def test_sort_float(self, N):
        # 使用种子 42 初始化随机数生成器
        np.random.seed(42)
        # 生成长度为 N 的浮点数数组，取值范围在 [-0.5, 0.5) 之间，并将部分元素设为 NaN
        arr = -0.5 + np.random.sample(N).astype("f")
        arr[np.random.choice(arr.shape[0], 3)] = np.nan
        # 断言使用快速排序和堆排序的结果相等
        assert_equal(np.sort(arr, kind="quick"), np.sort(arr, kind="heap"))

        # 创建所有元素为 +INF 的浮点数数组，并将部分元素设为 -1.0
        infarr = np.inf * np.ones(N, dtype="f")
        infarr[np.random.choice(infarr.shape[0], 5)] = -1.0
        # 断言使用快速排序和堆排序的结果相等
        assert_equal(np.sort(infarr, kind="quick"), np.sort(infarr, kind="heap"))

        # 创建所有元素为 -INF 的浮点数数组，并将部分元素设为 1.0
        neginfarr = -np.inf * np.ones(N, dtype="f")
        neginfarr[np.random.choice(neginfarr.shape[0], 5)] = 1.0
        # 断言使用快速排序和堆排序的结果相等
        assert_equal(np.sort(neginfarr, kind="quick"), np.sort(neginfarr, kind="heap"))

        # 创建所有元素为 +INF 的浮点数数组，并将一半元素设为 -INF
        infarr = np.inf * np.ones(N, dtype="f")
        infarr[np.random.choice(infarr.shape[0], (int)(N / 2))] = -np.inf
        # 断言使用快速排序和堆排序的结果相等
        assert_equal(np.sort(infarr, kind="quick"), np.sort(infarr, kind="heap"))

    # 定义一个测试方法，用于测试对整数数组的排序
    def test_sort_int(self):
        # 使用种子 1234 初始化随机数生成器
        np.random.seed(1234)
        # 定义整数数组长度为 2047
        N = 2047
        # 获取 int32 类型的最小值和最大值
        minv = np.iinfo(np.int32).min
        maxv = np.iinfo(np.int32).max
        # 生成随机整数数组，取值范围在 [minv, maxv] 之间，并将部分元素设为最小值和最大值
        arr = np.random.randint(low=minv, high=maxv, size=N).astype("int32")
        arr[np.random.choice(arr.shape[0], 10)] = minv
        arr[np.random.choice(arr.shape[0], 10)] = maxv
        # 断言使用快速排序和堆排序的结果相等
        assert_equal(np.sort(arr, kind="quick"), np.sort(arr, kind="heap"))
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来执行测试
    run_tests()
```