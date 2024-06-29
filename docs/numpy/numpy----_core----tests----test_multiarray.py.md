# `.\numpy\numpy\_core\tests\test_multiarray.py`

```py
from __future__ import annotations
# 引入用于类型提示的特殊导入语法，使得类型提示可以在类定义中使用

import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap

import numpy as np
import numpy._core._multiarray_tests as _multiarray_tests
from numpy._core._rational_tests import rational
from numpy.exceptions import AxisError, ComplexWarning
from numpy.testing import (
    assert_, assert_raises, assert_warns, assert_equal, assert_almost_equal,
    assert_array_equal, assert_raises_regex, assert_array_almost_equal,
    assert_allclose, IS_PYPY, IS_WASM, IS_PYSTON, HAS_REFCOUNT,
    assert_array_less, runstring, temppath, suppress_warnings, break_cycles,
    _SUPPORTS_SVE, assert_array_compare,
)
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy._core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy._core.multiarray import _get_ndarray_c_version, dot
# 导入所需的各种模块和函数，包括numpy和其测试相关的功能

# Need to test an object that does not fully implement math interface
from datetime import timedelta, datetime
# 导入datetime模块的timedelta和datetime类，用于测试未完全实现数学接口的对象


def assert_arg_sorted(arr, arg):
    # 确保返回的数组在参数索引处按顺序排列，并且参数值是唯一的
    assert_equal(arr[arg], np.sort(arr))
    assert_equal(np.sort(arg), np.arange(len(arg)))
    # 使用numpy的函数进行断言，检查数组是否按照预期排序和参数是否唯一


def assert_arr_partitioned(kth, k, arr_part):
    # 确保数组分割后在第k个位置的值是kth
    assert_equal(arr_part[k], kth)
    assert_array_compare(operator.__le__, arr_part[:k], kth)
    assert_array_compare(operator.__ge__, arr_part[k:], kth)
    # 使用numpy的函数进行断言，检查数组是否按照预期分割


def _aligned_zeros(shape, dtype=float, order="C", align=None):
    """
    Allocate a new ndarray with aligned memory.

    The ndarray is guaranteed *not* aligned to twice the requested alignment.
    Eg, if align=4, guarantees it is not aligned to 8. If align=None uses
    dtype.alignment.
    """
    # 分配一个带有对齐内存的新ndarray

    dtype = np.dtype(dtype)
    if dtype == np.dtype(object):
        # 如果dtype是object类型，则无法使用对齐内存，回退到标准分配
        if align is not None:
            raise ValueError("object array alignment not supported")
        return np.zeros(shape, dtype=dtype, order=order)
    
    if align is None:
        align = dtype.alignment
    
    if not hasattr(shape, '__len__'):
        shape = (shape,)
    
    size = functools.reduce(operator.mul, shape) * dtype.itemsize
    buf = np.empty(size + 2*align + 1, np.uint8)
    # 分配足够大的缓冲区来确保有足够的空间对齐

    ptr = buf.__array_interface__['data'][0]
    offset = ptr % align
    if offset != 0:
        offset = align - offset
    if (ptr % (2*align)) == 0:
        offset += align
    # 计算偏移量，确保数据对齐

    buf = buf[offset:offset+size+1][:-1]
    buf.fill(0)
    data = np.ndarray(shape, dtype, buf, order=order)
    return data
    # 使用缓冲区创建ndarray，确保内存对齐


class TestFlags:
    pass
    # 定义一个空的测试类
    # 设置测试方法的初始化操作，创建一个包含0到9的numpy数组
    def setup_method(self):
        self.a = np.arange(10)

    # 测试数组是否可写，使用本地变量字典来运行字符串
    def test_writeable(self):
        mydict = locals()
        # 设置数组为不可写状态，并尝试修改第一个元素，预期抛出 ValueError 异常
        self.a.flags.writeable = False
        assert_raises(ValueError, runstring, 'self.a[0] = 3', mydict)
        # 将数组设置为可写状态，并修改第一个元素的值
        self.a.flags.writeable = True
        self.a[0] = 5
        # 再次将第一个元素设置为0
        self.a[0] = 0

    # 测试从任何基类是否可以使数组可写，特别是对于通过数组接口创建的数组
    def test_writeable_any_base(self):
        # 创建一个从0到9的numpy数组
        arr = np.arange(10)

        # 创建一个子类，使得基类不会被折叠，可以更改标志位
        class subclass(np.ndarray):
            pass

        # 创建一个视图，使得基类不会被折叠，可以更改标志位
        view1 = arr.view(subclass)
        view2 = view1[...]
        # 将原始数组设置为不可写状态
        arr.flags.writeable = False
        view2.flags.writeable = False
        # 将视图的标志位设置为可写状态，再次可以设置为True
        view2.flags.writeable = True

        # 重新创建一个从0到9的numpy数组
        arr = np.arange(10)

        # 创建一个类来模拟数组接口
        class frominterface:
            def __init__(self, arr):
                self.arr = arr
                self.__array_interface__ = arr.__array_interface__

        # 使用数组接口创建一个视图
        view1 = np.asarray(frominterface)
        view2 = view1[...]
        # 将视图的标志位设置为不可写状态，再设置为可写状态
        view2.flags.writeable = False
        view2.flags.writeable = True

        # 将第一个视图的标志位设置为不可写状态，第二个视图跟随设置为不可写状态，并断言抛出 ValueError 异常
        view1.flags.writeable = False
        view2.flags.writeable = False
        with assert_raises(ValueError):
            view2.flags.writeable = True

    # 测试从只读缓冲区创建的数组是否可以设置为可写
    def test_writeable_from_readonly(self):
        # 创建一个包含100个0的字节数据
        data = b'\x00' * 100
        # 从缓冲区创建一个无符号字节类型的numpy数组
        vals = np.frombuffer(data, 'B')
        # 断言尝试将只读数组的标志位设置为可写时会抛出 ValueError 异常
        assert_raises(ValueError, vals.setflags, write=True)
        # 使用给定类型从字符串数据创建numpy记录数组
        types = np.dtype( [('vals', 'u1'), ('res3', 'S4')] )
        values = np._core.records.fromstring(data, types)
        vals = values['vals']
        # 断言尝试将只读数组的标志位设置为可写时会抛出 ValueError 异常
        assert_raises(ValueError, vals.setflags, write=True)

    # 测试从缓冲区创建的数组是否可以设置为可写
    def test_writeable_from_buffer(self):
        # 创建一个包含100个0的字节数据
        data = bytearray(b'\x00' * 100)
        # 从缓冲区创建一个无符号字节类型的numpy数组
        vals = np.frombuffer(data, 'B')
        # 断言数组的标志位初始为可写状态
        assert_(vals.flags.writeable)
        # 将数组的标志位设置为不可写状态
        vals.setflags(write=False)
        # 断言数组的标志位已经不可写
        assert_(vals.flags.writeable is False)
        # 将数组的标志位重新设置为可写状态
        vals.setflags(write=True)
        # 再次断言数组的标志位已经可写
        assert_(vals.flags.writeable)
        
        # 使用给定类型从字符串数据创建numpy记录数组
        types = np.dtype( [('vals', 'u1'), ('res3', 'S4')] )
        values = np._core.records.fromstring(data, types)
        vals = values['vals']
        # 断言数组的标志位初始为可写状态
        assert_(vals.flags.writeable)
        # 将数组的标志位设置为不可写状态
        vals.setflags(write=False)
        # 断言数组的标志位已经不可写
        assert_(vals.flags.writeable is False)
        # 将数组的标志位重新设置为可写状态
        vals.setflags(write=True)
        # 再次断言数组的标志位已经可写
        assert_(vals.flags.writeable)

    # 标记测试用例为跳过，如果运行环境是 PyPy，则总是复制
    @pytest.mark.skipif(IS_PYPY, reason="PyPy always copies")
    # 测试可写的 pickle
    def test_writeable_pickle(self):
        import pickle
        # 小型数组在未设置基础时会被复制。
        # 查看 array_setstate 中使用 PyArray_SetBaseObject 的条件。
        a = np.arange(1000)
        # 使用不同的 pickle 协议版本进行序列化和反序列化测试
        for v in range(pickle.HIGHEST_PROTOCOL):
            # 反序列化并验证数组的可写性标志
            vals = pickle.loads(pickle.dumps(a, v))
            assert_(vals.flags.writeable)
            # 验证基础对象是否为字节流
            assert_(isinstance(vals.base, bytes))

    # 测试从 C 数据中获取可写性
    def test_writeable_from_c_data(self):
        # 测试对一个封装低级 C 数据、但不拥有其数据的数组，可更改其可写性标志。
        # 同时查看从 Python 更改此功能已被弃用的情况。
        from numpy._core._multiarray_tests import get_c_wrapping_array

        # 获取一个包装了可写 C 数据的数组
        arr_writeable = get_c_wrapping_array(True)
        assert not arr_writeable.flags.owndata
        assert arr_writeable.flags.writeable
        view = arr_writeable[...]

        # 在视图上切换可写性标志
        view.flags.writeable = False
        assert not view.flags.writeable
        view.flags.writeable = True
        assert view.flags.writeable
        # 可在 arr_writeable 上取消可写性标志
        arr_writeable.flags.writeable = False

        # 获取一个包装了只读 C 数据的数组
        arr_readonly = get_c_wrapping_array(False)
        assert not arr_readonly.flags.owndata
        assert not arr_readonly.flags.writeable

        # 对可写性和只读性数组进行相同的测试
        for arr in [arr_writeable, arr_readonly]:
            view = arr[...]
            # 确保视图是只读的
            view.flags.writeable = False
            arr.flags.writeable = False
            assert not arr.flags.writeable

            # 验证试图在只读模式下无法设置可写性标志
            with assert_raises(ValueError):
                view.flags.writeable = True

            # 验证在使用 DeprecationWarning 过滤器时，试图设置可写性标志会引发警告
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                with assert_raises(DeprecationWarning):
                    arr.flags.writeable = True

            # 使用 assert_warns 验证试图设置可写性标志会产生 DeprecationWarning 警告
            with assert_warns(DeprecationWarning):
                arr.flags.writeable = True

    # 测试警告写操作
    def test_warnonwrite(self):
        a = np.arange(10)
        # 设置数组的 _warn_on_write 标志为 True
        a.flags._warn_on_write = True
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always')
            # 修改数组元素，验证是否只警告一次
            a[1] = 10
            a[2] = 10
            assert_(len(w) == 1)

    # 使用 pytest 的参数化装饰器进行只读标志协议测试
    @pytest.mark.parametrize(["flag", "flag_value", "writeable"],
            [("writeable", True, True),
             # 在弃用后删除 _warn_on_write 并简化参数化
             ("_warn_on_write", True, False),
             ("writeable", False, False)])
    def test_readonly_flag_protocols(self, flag, flag_value, writeable):
        a = np.arange(10)
        # 设置数组的指定标志和值
        setattr(a.flags, flag, flag_value)

        # 创建一个使用数组结构的类 MyArr
        class MyArr():
            __array_struct__ = a.__array_struct__

        # 验证 memoryview 是否只读
        assert memoryview(a).readonly is not writeable
        # 验证数组的 __array_interface__ 中的数据是否只读
        assert a.__array_interface__['data'][1] is not writeable
        # 验证从 MyArr 创建的数组是否可写
        assert np.asarray(MyArr()).flags.writeable is writeable
    # 测试函数，用于验证对象的各种标志位
    def test_otherflags(self):
        # 断言对象的 carray 标志位为 True
        assert_equal(self.a.flags.carray, True)
        # 断言对象的 'C' 标志位为 True，等同于上一行的断言
        assert_equal(self.a.flags['C'], True)
        # 断言对象的 farray 标志位为 False
        assert_equal(self.a.flags.farray, False)
        # 断言对象的 behaved 标志位为 True
        assert_equal(self.a.flags.behaved, True)
        # 断言对象的 fnc 标志位为 False
        assert_equal(self.a.flags.fnc, False)
        # 断言对象的 forc 标志位为 True
        assert_equal(self.a.flags.forc, True)
        # 断言对象的 owndata 标志位为 True
        assert_equal(self.a.flags.owndata, True)
        # 断言对象的 writeable 标志位为 True
        assert_equal(self.a.flags.writeable, True)
        # 断言对象的 aligned 标志位为 True
        assert_equal(self.a.flags.aligned, True)
        # 断言对象的 writebackifcopy 标志位为 False
        assert_equal(self.a.flags.writebackifcopy, False)
        # 断言对象的 'X' 标志位为 False，等同于上一行的断言
        assert_equal(self.a.flags['X'], False)
        # 断言对象的 'WRITEBACKIFCOPY' 标志位为 False，等同于上两行的断言
        assert_equal(self.a.flags['WRITEBACKIFCOPY'], False)

    # 测试函数，用于验证字符串类型数组的对齐情况
    def test_string_align(self):
        # 创建一个大小为 4 的字符串类型数组，每个元素长度为 4
        a = np.zeros(4, dtype=np.dtype('|S4'))
        # 断言数组的 aligned 标志位为 True
        assert_(a.flags.aligned)
        # 创建一个大小为 5 的字符串类型数组，每个元素长度为 4
        a = np.zeros(5, dtype=np.dtype('|S4'))
        # 断言数组的 aligned 标志位为 True
        assert_(a.flags.aligned)

    # 测试函数，用于验证复合类型数组的对齐情况
    def test_void_align(self):
        # 创建一个大小为 4 的复合类型数组，每个元素包含两个字段 ('a', 'i4') 和 ('b', 'i4')
        a = np.zeros(4, dtype=np.dtype([("a", "i4"), ("b", "i4")]))
        # 断言数组的 aligned 标志位为 True
        assert_(a.flags.aligned)
class TestHash:
    # 定义一个测试类 TestHash，用于测试哈希值的正确性
    # see #3793  # 关联 GitHub Issue #3793

    # 测试整数类型的哈希值
    def test_int(self):
        # 针对不同的整数类型和位数进行测试
        for st, ut, s in [(np.int8, np.uint8, 8),   # 使用 np.int8 和 np.uint8 测试 8 位整数
                          (np.int16, np.uint16, 16),  # 使用 np.int16 和 np.uint16 测试 16 位整数
                          (np.int32, np.uint32, 32),  # 使用 np.int32 和 np.uint32 测试 32 位整数
                          (np.int64, np.uint64, 64)]:  # 使用 np.int64 和 np.uint64 测试 64 位整数
            # 对每种位数的整数进行多次测试
            for i in range(1, s):
                # 测试负整数的哈希值是否正确
                assert_equal(hash(st(-2**i)), hash(-2**i),
                             err_msg="%r: -2**%d" % (st, i))
                # 测试正整数的哈希值是否正确
                assert_equal(hash(st(2**(i - 1))), hash(2**(i - 1)),
                             err_msg="%r: 2**%d" % (st, i - 1))
                # 测试 2**i - 1 的哈希值是否正确
                assert_equal(hash(st(2**i - 1)), hash(2**i - 1),
                             err_msg="%r: 2**%d - 1" % (st, i))

                # 修正 i 的值，确保不会出现负索引
                i = max(i - 1, 1)
                # 测试无符号整数 2**(i - 1) 的哈希值是否正确
                assert_equal(hash(ut(2**(i - 1))), hash(2**(i - 1)),
                             err_msg="%r: 2**%d" % (ut, i - 1))
                # 测试无符号整数 2**i - 1 的哈希值是否正确
                assert_equal(hash(ut(2**i - 1)), hash(2**i - 1),
                             err_msg="%r: 2**%d - 1" % (ut, i))


class TestAttributes:
    # 定义一个测试类 TestAttributes，用于测试 numpy 数组的属性

    # 设置测试方法的前置条件
    def setup_method(self):
        # 创建三个不同的 numpy 数组对象
        self.one = np.arange(10)
        self.two = np.arange(20).reshape(4, 5)
        self.three = np.arange(60, dtype=np.float64).reshape(2, 5, 6)

    # 测试数组的 shape 属性
    def test_attributes(self):
        # 检查数组的形状是否与预期一致
        assert_equal(self.one.shape, (10,))
        assert_equal(self.two.shape, (4, 5))
        assert_equal(self.three.shape, (2, 5, 6))
        
        # 修改数组的 shape 属性，并再次进行检查
        self.three.shape = (10, 3, 2)
        assert_equal(self.three.shape, (10, 3, 2))
        self.three.shape = (2, 5, 6)
        
        # 检查数组的 strides 属性是否正确
        assert_equal(self.one.strides, (self.one.itemsize,))
        num = self.two.itemsize
        assert_equal(self.two.strides, (5*num, num))
        num = self.three.itemsize
        assert_equal(self.three.strides, (30*num, 6*num, num))
        
        # 检查数组的维度属性
        assert_equal(self.one.ndim, 1)
        assert_equal(self.two.ndim, 2)
        assert_equal(self.three.ndim, 3)
        
        # 检查数组的 size 属性
        num = self.two.itemsize
        assert_equal(self.two.size, 20)
        
        # 检查数组的 nbytes 属性
        assert_equal(self.two.nbytes, 20*num)
        
        # 检查数组的 itemsize 属性
        assert_equal(self.two.itemsize, self.two.dtype.itemsize)
        
        # 检查数组的 base 属性
        assert_equal(self.two.base, np.arange(20))

    # 测试 dtype 属性
    def test_dtypeattr(self):
        # 检查数组的 dtype 是否与预期一致
        assert_equal(self.one.dtype, np.dtype(np.int_))
        assert_equal(self.three.dtype, np.dtype(np.float64))
        
        # 检查数组的 dtype 的 char 属性
        assert_equal(self.one.dtype.char, np.dtype(int).char)
        assert self.one.dtype.char in "lq"
        assert_equal(self.three.dtype.char, 'd')
        
        # 检查数组的 dtype 的 str 属性
        assert_(self.three.dtype.str[0] in '<>')
        assert_equal(self.one.dtype.str[1], 'i')
        assert_equal(self.three.dtype.str[1], 'f')

    # 测试整数子类化
    def test_int_subclassing(self):
        # 回归测试 https://github.com/numpy/numpy/pull/3526

        # 创建一个 numpy 整数对象
        numpy_int = np.int_(0)

        # int_ 类型并不继承自 Python 的 int，因为它不是固定宽度的
        assert_(not isinstance(numpy_int, int))
    # 定义测试函数 test_stridesattr
    def test_stridesattr(self):
        # 从 self.one 中获取数据，并赋值给 x
        x = self.one

        # 定义内部函数 make_array，用于创建 ndarray 对象
        def make_array(size, offset, strides):
            # 创建一个 numpy ndarray 对象，使用 x 作为缓冲区，整数类型，指定偏移和步幅
            return np.ndarray(size, buffer=x, dtype=int,
                              offset=offset*x.itemsize,
                              strides=strides*x.itemsize)

        # 断言 make_array(4, 4, -1) 得到的结果等于 np.array([4, 3, 2, 1])
        assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
        # 断言调用 make_array(4, 4, -2) 会引发 ValueError 异常
        assert_raises(ValueError, make_array, 4, 4, -2)
        # 断言调用 make_array(4, 2, -1) 会引发 ValueError 异常
        assert_raises(ValueError, make_array, 4, 2, -1)
        # 断言调用 make_array(8, 3, 1) 会引发 ValueError 异常
        assert_raises(ValueError, make_array, 8, 3, 1)
        # 断言 make_array(8, 3, 0) 得到的结果等于 np.array([3]*8)
        assert_equal(make_array(8, 3, 0), np.array([3]*8))
        # 检查 GitHub 上报告的行为（gh-2503）
        assert_raises(ValueError, make_array, (2, 3), 5, np.array([-2, -3]))
        # 调用 make_array(0, 0, 10)，创建大小为 0 的数组

    # 定义测试函数 test_set_stridesattr
    def test_set_stridesattr(self):
        # 从 self.one 中获取数据，并赋值给 x
        x = self.one

        # 定义内部函数 make_array，用于创建 ndarray 对象
        def make_array(size, offset, strides):
            try:
                # 尝试创建一个 numpy ndarray 对象，使用 x 作为缓冲区，指定偏移和类型
                r = np.ndarray([size], dtype=int, buffer=x,
                               offset=offset*x.itemsize)
            except Exception as e:
                # 如果出现异常，抛出 RuntimeError 异常
                raise RuntimeError(e)
            # 设置 ndarray 对象的步幅值为 strides * x.itemsize
            r.strides = strides = strides*x.itemsize
            return r

        # 断言 make_array(4, 4, -1) 得到的结果等于 np.array([4, 3, 2, 1])
        assert_equal(make_array(4, 4, -1), np.array([4, 3, 2, 1]))
        # 断言 make_array(7, 3, 1) 得到的结果等于 np.array([3, 4, 5, 6, 7, 8, 9])
        assert_equal(make_array(7, 3, 1), np.array([3, 4, 5, 6, 7, 8, 9]))
        # 断言调用 make_array(4, 4, -2) 会引发 ValueError 异常
        assert_raises(ValueError, make_array, 4, 4, -2)
        # 断言调用 make_array(4, 2, -1) 会引发 ValueError 异常
        assert_raises(ValueError, make_array, 4, 2, -1)
        # 断言调用 make_array(8, 3, 1) 会引发 RuntimeError 异常
        assert_raises(RuntimeError, make_array, 8, 3, 1)

        # 设置 x 为 np.lib.stride_tricks.as_strided 创建的数组
        x = np.lib.stride_tricks.as_strided(np.arange(1), (10, 10), (0, 0))

        # 定义内部函数 set_strides，用于设置 ndarray 对象的步幅值
        def set_strides(arr, strides):
            # 尝试设置数组 arr 的步幅值为 strides
            arr.strides = strides

        # 断言调用 set_strides(x, (10*x.itemsize, x.itemsize)) 会引发 ValueError 异常

        # 测试偏移计算：
        x = np.lib.stride_tricks.as_strided(np.arange(10, dtype=np.int8)[-1],
                                                    shape=(10,), strides=(-1,))
        # 断言调用 set_strides(x[::-1], -1) 会引发 ValueError 异常
        assert_raises(ValueError, set_strides, x[::-1], -1)
        a = x[::-1]
        # 设置 a 的步幅值为 1
        a.strides = 1
        # 设置 a[::2] 的步幅值为 2

        # 测试 0d 数组
        arr_0d = np.array(0)
        # 设置 arr_0d 的步幅为空元组
        arr_0d.strides = ()
        # 断言调用 set_strides(arr_0d, None) 会引发 TypeError 异常

    # 定义测试函数 test_fill
    def test_fill(self):
        # 对每种数据类型 t 进行循环测试
        for t in "?bhilqpBHILQPfdgFDGO":
            # 创建一个空的 numpy 数组 x，指定数据类型为 t，形状为 (3, 2, 1)
            x = np.empty((3, 2, 1), t)
            # 创建一个相同形状的空的 numpy 数组 y，指定数据类型为 t
            y = np.empty((3, 2, 1), t)
            # 用值 1 填充数组 x
            x.fill(1)
            # 用值 1 填充数组 y（使用 y[...] = 1 的方式）
            y[...] = 1
            # 断言数组 x 和 y 相等
            assert_equal(x, y)

    # 定义测试函数 test_fill_max_uint64
    def test_fill_max_uint64(self):
        # 创建一个空的 numpy 数组 x，指定数据类型为 np.uint64，形状为 (3, 2, 1)
        x = np.empty((3, 2, 1), dtype=np.uint64)
        # 创建一个相同形状的空的 numpy 数组 y，指定数据类型为 np.uint64
        y = np.empty((3, 2, 1), dtype=np.uint64)
        # 设置 value 为 2^64 - 1
        value = 2**64 - 1
        # 使用 y[...] = value 的方式，用 value 填充数组 y
        y[...] = value
        # 使用 x.fill(value) 的方式，用 value 填充数组 x
        x.fill(value)
        # 断言数组 x 和 y 相等
        assert_array_equal(x, y)
    # 定义测试函数 test_fill_struct_array(self)，用于测试填充结构化数组的功能

    # 使用 numpy 创建一个结构化数组 x，包含两个元组，每个元组包括一个整数和一个浮点数，dtype 分别为 'i4' 和 'f8'
    x = np.array([(0, 0.0), (1, 1.0)], dtype='i4,f8')
    
    # 将数组 x 使用第一个元组的值填充整个数组
    x.fill(x[0])
    
    # 断言结构化数组 x 中的第一个和第二个元组的 'f1' 字段（即浮点数字段）相等
    assert_equal(x['f1'][1], x['f1'][0])
    
    # 使用 numpy 创建一个长度为 2 的全零数组 x，dtype 包括两个字段 'a'（浮点数类型）和 'b'（整数类型）
    x = np.zeros(2, dtype=[('a', 'f8'), ('b', 'i4')])
    
    # 使用元组 (3.5, -2) 填充数组 x 的所有元素
    x.fill((3.5, -2))
    
    # 断言数组 x 的 'a' 字段包含值 [3.5, 3.5]
    assert_array_equal(x['a'], [3.5, 3.5])
    
    # 断言数组 x 的 'b' 字段包含值 [-2, -2]
    assert_array_equal(x['b'], [-2, -2])

    # 定义测试函数 test_fill_readonly(self)，用于测试在只读模式下填充数组的行为
    
    # 创建一个长度为 11 的全零数组 a
    a = np.zeros(11)
    
    # 将数组 a 设置为只读模式，不可写
    a.setflags(write=False)
    
    # 使用 pytest 断言，当尝试在只读数组 a 上执行填充操作时，会引发 ValueError 异常，异常信息包含 "read-only"
    with pytest.raises(ValueError, match=".*read-only"):
        a.fill(0)
class TestArrayConstruction:
    # 测试数组构建的类
    def test_array(self):
        # 测试创建数组的方法
        d = np.ones(6)
        # 创建一个包含六个1的数组
        r = np.array([d, d])
        # 使用d创建一个二维数组r，包含两行六列
        assert_equal(r, np.ones((2, 6)))
        # 断言r与全1的二维数组（2行6列）相等

        d = np.ones(6)
        # 重新创建一个包含六个1的数组d
        tgt = np.ones((2, 6))
        # 创建一个目标数组tgt，包含两行六列，元素全为1
        r = np.array([d, d])
        # 使用d创建一个二维数组r，包含两行六列
        assert_equal(r, tgt)
        # 断言r与tgt相等
        tgt[1] = 2
        # 修改tgt的第二行为全2
        r = np.array([d, d + 1])
        # 使用d和d+1创建一个二维数组r
        assert_equal(r, tgt)
        # 断言r与修改后的tgt相等

        d = np.ones(6)
        # 重新创建一个包含六个1的数组d
        r = np.array([[d, d]])
        # 使用d创建一个二维数组r，包含一个元素，元素是一个包含两个数组的列表
        assert_equal(r, np.ones((1, 2, 6)))
        # 断言r与全1的三维数组（1行2列6深度）相等

        d = np.ones(6)
        # 重新创建一个包含六个1的数组d
        r = np.array([[d, d], [d, d]])
        # 使用d创建一个二维数组r，包含两行两列，每个元素都是一个包含两个数组的列表
        assert_equal(r, np.ones((2, 2, 6)))
        # 断言r与全1的三维数组（2行2列6深度）相等

        d = np.ones((6, 6))
        # 重新创建一个包含6行6列的全1数组d
        r = np.array([d, d])
        # 使用d创建一个二维数组r，包含两行六列六深度
        assert_equal(r, np.ones((2, 6, 6)))
        # 断言r与全1的三维数组（2行6列6深度）相等

        d = np.ones((6, ))
        # 重新创建一个包含六个1的一维数组d
        r = np.array([[d, d + 1], d + 2], dtype=object)
        # 使用d和d+1创建一个二维数组r，第一行是包含两个一维数组的列表，第二行是d+2
        assert_equal(len(r), 2)
        # 断言r的长度为2
        assert_equal(r[0], [d, d + 1])
        # 断言r的第一个元素与包含d和d+1的列表相等
        assert_equal(r[1], d + 2)
        # 断言r的第二个元素与d+2相等

        tgt = np.ones((2, 3), dtype=bool)
        # 创建一个包含2行3列的全1布尔类型数组tgt
        tgt[0, 2] = False
        # 修改tgt的第一行第三列为False
        tgt[1, 0:2] = False
        # 修改tgt的第二行的第一列和第二列为False
        r = np.array([[True, True, False], [False, False, True]])
        # 使用指定的值创建一个二维数组r
        assert_equal(r, tgt)
        # 断言r与修改后的tgt相等
        r = np.array([[True, False], [True, False], [False, True]])
        # 使用指定的值创建一个二维数组r
        assert_equal(r, tgt.T)
        # 断言r与tgt的转置相等

    def test_array_empty(self):
        # 测试空数组的方法
        assert_raises(TypeError, np.array)
        # 断言调用np.array时会抛出TypeError异常

    def test_0d_array_shape(self):
        # 测试0维数组形状的方法
        assert np.ones(np.array(3)).shape == (3,)
        # 断言np.ones(np.array(3))的形状为(3,)

    def test_array_copy_false(self):
        # 测试数组复制为False的方法
        d = np.array([1, 2, 3])
        # 创建一个包含1、2、3的数组d
        e = np.array(d, copy=False)
        # 使用d创建一个不复制的数组e
        d[1] = 3
        # 修改d的第二个元素为3
        assert_array_equal(e, [1, 3, 3])
        # 断言e与修改后的d相等
        np.array(d, copy=False, order='F')

    def test_array_copy_if_needed(self):
        # 测试按需复制数组的方法
        d = np.array([1, 2, 3])
        # 创建一个包含1、2、3的数组d
        e = np.array(d, copy=None)
        # 使用d创建一个根据需要复制的数组e
        d[1] = 3
        # 修改d的第二个元素为3
        assert_array_equal(e, [1, 3, 3])
        # 断言e与修改后的d相等
        e = np.array(d, copy=None, order='F')
        # 使用d创建一个根据需要复制的数组e，按F顺序存储
        d[1] = 4
        # 修改d的第二个元素为4
        assert_array_equal(e, [1, 4, 3])
        # 断言e与修改后的d相等
        e[2] = 7
        # 修改e的第三个元素为7
        assert_array_equal(d, [1, 4, 7])
        # 断言d与修改后的e相等

    def test_array_copy_true(self):
        # 测试数组复制为True的方法
        d = np.array([[1,2,3], [1, 2, 3]])
        # 创建一个二维数组d，包含两行三列，元素为1、2、3
        e = np.array(d, copy=True)
        # 使用d创建一个复制的数组e
        d[0, 1] = 3
        # 修改d的第一行第二列为3
        e[0, 2] = -7
        # 修改e的第一行第三列为-7
        assert_array_equal(e, [[1, 2, -7], [1, 2, 3]])
        # 断言e与修改后的数组相等
        assert_array_equal(d, [[1, 3, 3], [1, 2, 3]])
        # 断言d与修改后的数组相等
        e = np.array(d, copy=True, order='F')
        # 使用d创建一个复制的数组e，按F顺序存储
        d[0, 1] = 5
        # 修改d的第一行第二列为5
        e[0, 2] = 7
        # 修改e的第一行第三列为7
        assert_array_equal(e, [[1, 3, 7], [1, 2, 3]])
        # 断言e与修改后的数组相等
        assert_array_equal(d, [[1, 5, 3], [1,2,3]])
        # 断言d与修改后的数组相等

    def test_array_copy_str(self):
        # 测试字符串复制数组的方法
        with pytest.raises(
            ValueError,
            match="strings are not allowed for 'copy' keyword. "
                  "Use True/False/None instead."
        ):
            np.array([1, 2, 3], copy="always")
            # 断言调用np.array时会抛出值错误异常，匹配指定的错误消息

    def test_array_cont(self):
        # 测试连续数组的方法
        d = np.ones(10)[::2]
        # 创建一个包含10个1的数组d，选取步长为2的元素
        assert_(np.ascontiguousarray(d).
    # 使用 pytest 的 parametrize 装饰器为 test_bad_arguments_error 方法参数化，传入不同的 numpy 函数作为参数 func
    @pytest.mark.parametrize("func",
                [np.array,
                 np.asarray,
                 np.asanyarray,
                 np.ascontiguousarray,
                 np.asfortranarray])
    def test_bad_arguments_error(self, func):
        # 使用 pytest 的 raises 方法断言 TypeError 异常被抛出
        with pytest.raises(TypeError):
            # 调用 func 函数并传入不合法的参数
            func(3, dtype="bad dtype")
        with pytest.raises(TypeError):
            # 调用 func 函数并缺少参数
            func()  # missing arguments
        with pytest.raises(TypeError):
            # 调用 func 函数并传入过多的参数
            func(1, 2, 3, 4, 5, 6, 7, 8)  # too many arguments
    
    # 使用 pytest 的 parametrize 装饰器为 test_array_as_keyword 方法参数化，传入不同的 numpy 函数作为参数 func
    @pytest.mark.parametrize("func",
                [np.array,
                 np.asarray,
                 np.asanyarray,
                 np.ascontiguousarray,
                 np.asfortranarray])
    def test_array_as_keyword(self, func):
        # 对于 np.array 函数，尝试使用关键字参数 object=3 调用
        # 注意：此处可能应该将其改为仅接受位置参数，但不要意外更改其名称。
        if func is np.array:
            func(object=3)
        else:
            # 对于其它函数，尝试使用关键字参数 a=3 调用
            func(a=3)
class TestAssignment:
    def test_assignment_broadcasting(self):
        a = np.arange(6).reshape(2, 3)

        # Broadcasting the input to the output
        # 将输入广播到输出
        a[...] = np.arange(3)
        assert_equal(a, [[0, 1, 2], [0, 1, 2]])
        
        # Broadcasting a different shape to the output
        # 将不同形状的数组广播到输出
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
        # 为了与 <= 1.5 版本兼容，输出到输入的广播的有限版本
        #
        # 这种行为与一般的 NumPy 广播不一致，
        # 因为它只使用了两个广播规则中的一个（在形状左侧添加新的 "1" 维度），
        # 应用于输出而不是输入。在 NumPy 2.0 中，这种广播赋值可能会被禁止。
        a[...] = np.arange(6)[::-1].reshape(1, 2, 3)
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])
        # The other type of broadcasting would require a reduction operation.
        # 另一种类型的广播需要进行减少操作。

        def assign(a, b):
            a[...] = b

        assert_raises(ValueError, assign, a, np.arange(12).reshape(2, 2, 3))

    def test_assignment_errors(self):
        # Address issue #2276
        class C:
            pass
        a = np.zeros(1)

        def assign(v):
            a[0] = v

        assert_raises((AttributeError, TypeError), assign, C())
        assert_raises(ValueError, assign, [1])

    @pytest.mark.filterwarnings(
        "ignore:.*set_string_function.*:DeprecationWarning"
    )
    def test_unicode_assignment(self):
        # gh-5049
        from numpy._core.arrayprint import set_printoptions

        @contextmanager
        def inject_str(s):
            """ replace ndarray.__str__ temporarily """
            # 临时替换 ndarray.__str__
            set_printoptions(formatter={"all": lambda x: s})
            try:
                yield
            finally:
                set_printoptions()

        a1d = np.array(['test'])
        a0d = np.array('done')
        with inject_str('bad'):
            a1d[0] = a0d  # previously this would invoke __str__
        assert_equal(a1d[0], 'done')

        # this would crash for the same reason
        # 由于相同的原因，这会导致崩溃
        np.array([np.array('\xe5\xe4\xf6')])

    def test_stringlike_empty_list(self):
        # gh-8902
        u = np.array(['done'])
        b = np.array([b'done'])

        class bad_sequence:
            def __getitem__(self): pass
            def __len__(self): raise RuntimeError

        assert_raises(ValueError, operator.setitem, u, 0, [])
        assert_raises(ValueError, operator.setitem, b, 0, [])

        assert_raises(ValueError, operator.setitem, u, 0, bad_sequence())
        assert_raises(ValueError, operator.setitem, b, 0, bad_sequence())
    # 定义一个测试方法，用于验证长双精度浮点数的赋值行为
    def test_longdouble_assignment(self):
        # 只有在长双精度浮点数大于普通浮点数时才相关
        # 我们关注精度丢失的情况

        # 遍历长双精度和复数长双精度两种数据类型
        for dtype in (np.longdouble, np.clongdouble):
            # 创建一个接近 0 的最大值，转换为指定数据类型
            tinyb = np.nextafter(np.longdouble(0), 1).astype(dtype)
            # 创建一个接近 0 的最小值，转换为指定数据类型
            tinya = np.nextafter(np.longdouble(0), -1).astype(dtype)

            # 构造包含 tinya 值的一维数组
            tiny1d = np.array([tinya])
            # 断言数组中第一个元素与 tinya 相等
            assert_equal(tiny1d[0], tinya)

            # 将数组中第一个元素赋值为 tinyb
            tiny1d[0] = tinyb
            # 断言数组中第一个元素与 tinyb 相等
            assert_equal(tiny1d[0], tinyb)

            # 将数组中第一个元素的所有维度赋值为 tinya
            tiny1d[0, ...] = tinya
            # 断言数组中第一个元素与 tinya 相等
            assert_equal(tiny1d[0], tinya)

            # 将数组中第一个元素的所有维度赋值为 tinyb 的所有维度
            tiny1d[0, ...] = tinyb[...]
            # 断言数组中第一个元素与 tinyb 相等
            assert_equal(tiny1d[0], tinyb)

            # 将数组中第一个元素赋值为 tinyb 的所有维度
            tiny1d[0] = tinyb[...]
            # 断言数组中第一个元素与 tinyb 相等
            assert_equal(tiny1d[0], tinyb)

            # 创建一个数组，包含一个包含 tinya 值的数组
            arr = np.array([np.array(tinya)])
            # 断言数组中第一个元素与 tinya 相等
            assert_equal(arr[0], tinya)

    # 定义一个测试方法，验证将数据转换为字符串时的行为
    def test_cast_to_string(self):
        # 将数据转换为字符串应当执行 "str(scalar)"，而非 "str(scalar.item())"
        # 例如，在 Python2 中，str(float) 会被截断，因此我们要避免使用 str(np.float64(...).item())
        # 因为这会不正确地截断数据。

        # 创建一个长度为 1 的字符串类型数组
        a = np.zeros(1, dtype='S20')
        # 将数组中所有元素赋值为一个浮点数数组的字符串表示
        a[:] = np.array(['1.12345678901234567890'], dtype='f8')
        # 断言数组中第一个元素与指定的字节字符串相等
        assert_equal(a[0], b"1.1234567890123457")
class TestDtypedescr:
    # 测试数据类型构造函数
    def test_construction(self):
        # 创建一个整数类型的数据类型对象 d1
        d1 = np.dtype('i4')
        # 断言 d1 应该等于 np.int32 的数据类型对象
        assert_equal(d1, np.dtype(np.int32))
        # 创建一个双精度浮点数类型的数据类型对象 d2
        d2 = np.dtype('f8')
        # 断言 d2 应该等于 np.float64 的数据类型对象
        assert_equal(d2, np.dtype(np.float64))

    # 测试字节顺序
    def test_byteorders(self):
        # 断言小端字节顺序的整数类型不等于大端字节顺序的整数类型
        assert_(np.dtype('<i4') != np.dtype('>i4'))
        # 断言具有结构化字段的小端字节顺序数组类型不等于具有结构化字段的大端字节顺序数组类型
        assert_(np.dtype([('a', '<i4')]) != np.dtype([('a', '>i4')]))

    # 测试结构化数组非空类型
    def test_structured_non_void(self):
        # 定义结构化字段
        fields = [('a', '<i2'), ('b', '<i2')]
        # 创建一个结构化数组数据类型对象 dt_int
        dt_int = np.dtype(('i4', fields))
        # 断言 dt_int 的字符串表示应该是 "(numpy.int32, [('a', '<i2'), ('b', '<i2')])"
        assert_equal(str(dt_int), "(numpy.int32, [('a', '<i2'), ('b', '<i2')])")

        # gh-9821
        # 创建一个由 dt_int 数据类型构成的全零数组对象 arr_int
        arr_int = np.zeros(4, dt_int)
        # 断言 arr_int 的字符串表示应该是 "array([0, 0, 0, 0], dtype=(numpy.int32, [('a', '<i2'), ('b', '<i2')]))"
        assert_equal(repr(arr_int),
            "array([0, 0, 0, 0], dtype=(numpy.int32, [('a', '<i2'), ('b', '<i2')]))")


class TestZeroRank:
    # 设置测试环境方法
    def setup_method(self):
        # 初始化数组对象 d，包括一个整数数组和一个字符串数组
        self.d = np.array(0), np.array('x', object)

    # 测试省略号索引
    def test_ellipsis_subscript(self):
        # 拆包数组对象 d
        a, b = self.d
        # 断言 a 的省略号索引应该等于 0
        assert_equal(a[...], 0)
        # 断言 b 的省略号索引应该等于 'x'
        assert_equal(b[...], 'x')
        # 断言 a 的省略号索引的基础应该是 a 本身
        assert_(a[...].base is a)  # `a[...] is a` in numpy <1.9.
        # 断言 b 的省略号索引的基础应该是 b 本身
        assert_(b[...].base is b)  # `b[...] is b` in numpy <1.9.

    # 测试空索引
    def test_empty_subscript(self):
        # 拆包数组对象 d
        a, b = self.d
        # 断言 a 的空索引应该等于 0
        assert_equal(a[()], 0)
        # 断言 b 的空索引应该等于 'x'
        assert_equal(b[()], 'x')
        # 断言 a 的空索引的类型应该是其数据类型的类型
        assert_(type(a[()]) is a.dtype.type)
        # 断言 b 的空索引的类型应该是字符串的类型
        assert_(type(b[()]) is str)

    # 测试无效索引
    def test_invalid_subscript(self):
        # 拆包数组对象 d
        a, b = self.d
        # 断言对 a 使用索引 0 应该引发 IndexError 异常
        assert_raises(IndexError, lambda x: x[0], a)
        # 断言对 b 使用索引 0 应该引发 IndexError 异常
        assert_raises(IndexError, lambda x: x[0], b)
        # 断言对 a 使用空的整数数组索引应该引发 IndexError 异常
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)
        # 断言对 b 使用空的整数数组索引应该引发 IndexError 异常
        assert_raises(IndexError, lambda x: x[np.array([], int)], b)

    # 测试省略号索引赋值
    def test_ellipsis_subscript_assignment(self):
        # 拆包数组对象 d
        a, b = self.d
        # 对 a 使用省略号索引赋值为 42
        a[...] = 42
        # 断言 a 应该等于 42
        assert_equal(a, 42)
        # 对 b 使用省略号索引赋值为空字符串
        b[...] = ''
        # 断言 b 的单个元素应该为空字符串
        assert_equal(b.item(), '')

    # 测试空索引赋值
    def test_empty_subscript_assignment(self):
        # 拆包数组对象 d
        a, b = self.d
        # 对 a 使用空索引赋值为 42
        a[()] = 42
        # 断言 a 应该等于 42
        assert_equal(a, 42)
        # 对 b 使用空索引赋值为空字符串
        b[()] = ''
        # 断言 b 的单个元素应该为空字符串
        assert_equal(b.item(), '')

    # 测试无效索引赋值
    def test_invalid_subscript_assignment(self):
        # 拆包数组对象 d
        a, b = self.d

        def assign(x, i, v):
            x[i] = v

        # 断言对 a 使用索引 0 赋值为 42 应该引发 IndexError 异常
        assert_raises(IndexError, assign, a, 0, 42)
        # 断言对 b 使用索引 0 赋值为空字符串应该引发 IndexError 异常
        assert_raises(IndexError, assign, b, 0, '')
        # 断言对 a 使用空索引赋值为空字符串应该引发 ValueError 异常
        assert_raises(ValueError, assign, a, (), '')

    # 测试 newaxis
    def test_newaxis(self):
        # 拆包数组对象 d
        a, b = self.d
        # 断言 a 的新轴索引的形状应该是 (1,)
        assert_equal(a[np.newaxis].shape, (1,))
        # 断言 a 的省略号和新轴索引的形状应该是 (1,)
        assert_equal(a[..., np.newaxis].shape, (1,))
        # 断言 a 的新轴和省略号索引的形状应该是 (1,)
        assert_equal(a[np.newaxis, ...].shape, (1,))
        # 断言 a 的省略号和新轴索引的形状应该是 (1,)
        assert_equal(a[..., np.newaxis].shape, (1,))
        # 断言 a 的两个新轴索引的形状应该是 (1, 1)
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))
        # 断言 a 的省略号和两个新轴索引的形状应该是 (1, 1)
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))
        # 断言 a 的两个新轴和省略号索引的形状应该是 (1, 1)
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))
        # 断言 a 的十个新轴索引的形状应该是 (1,)*10
        assert_equal(a[(np.newaxis,)*10].shape, (1,)*10)

    # 测试无效 newaxis
    def test_invalid_newaxis(self):
        # 拆包数组对象 d
        a, b = self.d

        def subscript(x, i):
            x[i]

        # 断言对 a 使用 (newaxis, 0) 的索引应该引发 IndexError 异常
        assert_raises(IndexError, subscript, a, (
    # 定义测试方法，用于测试 ndarray 的构造函数和功能
    def test_constructor(self):
        # 创建一个空的 ndarray 对象 x，并赋值为 5
        x = np.ndarray(())
        x[()] = 5
        # 断言确保 x[()] 的值为 5
        assert_equal(x[()], 5)
        
        # 从现有的 ndarray 对象 x 创建一个新的 ndarray 对象 y
        y = np.ndarray((), buffer=x)
        y[()] = 6
        # 断言确保 x[()] 的值为 6，验证了 y 与 x 共享数据缓冲区
        assert_equal(x[()], 6)

        # 检查当 strides 参数为空而 shape 参数为非空时是否引发 ValueError 异常
        with pytest.raises(ValueError):
            np.ndarray((2,), strides=())
        with pytest.raises(ValueError):
            np.ndarray((), strides=(2,))

    # 定义测试方法，用于测试 np.add 函数在特定情况下的异常处理
    def test_output(self):
        # 创建一个标量 ndarray 对象 x，并尝试使用 np.add 函数在同一对象上操作
        x = np.array(2)
        assert_raises(ValueError, np.add, x, [1], x)

    # 定义测试方法，用于测试复数 ndarray 对象的 real 和 imag 属性
    def test_real_imag(self):
        # 创建一个复数 ndarray 对象 x，并分别获取其实部和虚部
        # 这里包含了用于 gh-11245 的连续性检查
        x = np.array(1j)
        xr = x.real
        xi = x.imag

        # 断言确保实部 xr 的值为 0，并且 xr 是 ndarray 类型，且连续性标志为 True
        assert_equal(xr, np.array(0))
        assert_(type(xr) is np.ndarray)
        assert_equal(xr.flags.contiguous, True)
        assert_equal(xr.flags.f_contiguous, True)

        # 断言确保虚部 xi 的值为 1，并且 xi 是 ndarray 类型，且连续性标志为 True
        assert_equal(xi, np.array(1))
        assert_(type(xi) is np.ndarray)
        assert_equal(xi.flags.contiguous, True)
        assert_equal(xi.flags.f_contiguous, True)


这段代码是针对 `numpy` 库中 `ndarray` 对象的几个功能进行了单元测试。每个方法都测试了不同的情况，通过断言来验证预期行为是否符合预期，同时包含了异常处理的测试。
class TestScalarIndexing:
    # 设置测试方法的初始化，创建包含单个标量元素的 NumPy 数组
    def setup_method(self):
        self.d = np.array([0, 1])[0]

    # 测试省略号索引操作
    def test_ellipsis_subscript(self):
        a = self.d
        assert_equal(a[...], 0)  # 断言省略号索引返回值为 0
        assert_equal(a[...].shape, ())  # 断言省略号索引的形状为单个元素的空元组

    # 测试空元组索引操作
    def test_empty_subscript(self):
        a = self.d
        assert_equal(a[()], 0)  # 断言空元组索引返回值为 0
        assert_equal(a[()].shape, ())  # 断言空元组索引的形状为单个元素的空元组

    # 测试无效索引操作
    def test_invalid_subscript(self):
        a = self.d
        assert_raises(IndexError, lambda x: x[0], a)  # 断言访问超出索引范围将引发 IndexError
        assert_raises(IndexError, lambda x: x[np.array([], int)], a)  # 断言使用空数组作为索引将引发 IndexError

    # 测试无效索引赋值操作
    def test_invalid_subscript_assignment(self):
        a = self.d

        def assign(x, i, v):
            x[i] = v

        assert_raises(TypeError, assign, a, 0, 42)  # 断言尝试在标量上进行索引赋值将引发 TypeError

    # 测试 newaxis 操作
    def test_newaxis(self):
        a = self.d
        assert_equal(a[np.newaxis].shape, (1,))  # 断言使用 np.newaxis 增加维度后的形状为 (1,)
        assert_equal(a[..., np.newaxis].shape, (1,))  # 断言使用 np.newaxis 在末尾增加维度后的形状为 (1,)
        assert_equal(a[np.newaxis, ...].shape, (1,))  # 断言使用 np.newaxis 在开头增加维度后的形状为 (1,)
        assert_equal(a[..., np.newaxis].shape, (1,))  # 断言再次使用 np.newaxis 在末尾增加维度后的形状为 (1,)
        assert_equal(a[np.newaxis, ..., np.newaxis].shape, (1, 1))  # 断言同时在两个位置增加维度后的形状为 (1, 1)
        assert_equal(a[..., np.newaxis, np.newaxis].shape, (1, 1))  # 断言在末尾连续增加两个维度后的形状为 (1, 1)
        assert_equal(a[np.newaxis, np.newaxis, ...].shape, (1, 1))  # 断言在开头连续增加两个维度后的形状为 (1, 1)
        assert_equal(a[(np.newaxis,)*10].shape, (1,)*10)  # 断言使用多个 np.newaxis 连续增加 10 维后的形状为 (1, 1, ..., 1) 十次

    # 测试无效 newaxis 操作
    def test_invalid_newaxis(self):
        a = self.d

        def subscript(x, i):
            x[i]

        assert_raises(IndexError, subscript, a, (np.newaxis, 0))  # 断言在标量上使用元组包含 np.newaxis 将引发 IndexError
        assert_raises(IndexError, subscript, a, (np.newaxis,)*70)  # 断言在标量上连续使用 70 次 np.newaxis 将引发 IndexError

    # 测试重叠赋值操作
    def test_overlapping_assignment(self):
        # 使用正向步长
        a = np.arange(4)
        a[:-1] = a[1:]  # 将数组末尾元素复制到前面，最终数组为 [1, 2, 3, 3]
        assert_equal(a, [1, 2, 3, 3])

        a = np.arange(4)
        a[1:] = a[:-1]  # 将数组前面元素复制到末尾，最终数组为 [0, 0, 1, 2]
        assert_equal(a, [0, 0, 1, 2])

        # 使用正向和负向步长
        a = np.arange(4)
        a[:] = a[::-1]  # 将数组反转赋值给自身，最终数组为 [3, 2, 1, 0]
        assert_equal(a, [3, 2, 1, 0])

        a = np.arange(6).reshape(2, 3)
        a[::-1,:] = a[:, ::-1]  # 将二维数组的每一行反转赋值给自身，最终数组为 [[5, 4, 3], [2, 1, 0]]
        assert_equal(a, [[5, 4, 3], [2, 1, 0]])

        a = np.arange(6).reshape(2, 3)
        a[::-1, ::-1] = a[:, ::-1]  # 将二维数组反转赋值给自身，最终数组为 [[3, 4, 5], [0, 1, 2]]
        assert_equal(a, [[3, 4, 5], [0, 1, 2]])

        # 仅存在一个元素重叠赋值
        a = np.arange(5)
        a[:3] = a[2:]  # 将数组后面的元素复制到前面，最终数组为 [2, 3, 4, 3, 4]
        assert_equal(a, [2, 3, 4, 3, 4])

        a = np.arange(5)
        a[2:] = a[:3]  # 将数组前面的元素复制到后面，最终数组为 [0, 1, 0, 1, 2]
        assert_equal(a, [0, 1, 0, 1, 2])

        a = np.arange(5)
        a[2::-1] = a[2:]  # 将数组后面的元素逆序复制到前面，最终数组为 [4, 3, 2, 3, 4]
        assert_equal(a, [4, 3, 2, 3, 4])

        a = np.arange(5)
        a[2:] = a[2::-1]  # 将数组前面的元素逆序复制到后面，最终数组为 [0, 1, 2, 1, 0]
        assert_equal(a, [0, 1, 2, 1, 0])

        a = np.arange(5)
        a[2::-1] = a[:1:-1]  # 将数组后面的元素逆序复制到前面，最终数组为 [2, 3, 4, 3, 4]
        assert_equal(a, [2, 3, 4, 3, 4])

        a = np.arange(5)
        a[:1:-1] = a[2::-1]  # 将数组前面的元素逆序复制到后面，最终数组为 [0, 1, 0, 1, 2]
        assert_equal(a, [0, 1, 0, 1, 2])


class TestCreation:
    """
    Test the np.array constructor
    """
    # 测试从类属性创建 np.array 对象时的异常
    def test_from_attribute(self):
        class x:
            def __array__(self, dtype=None, copy=None):
                pass

        assert_raises(ValueError, np.array, x())
    # 测试从字符串创建数组的函数
    def test_from_string(self):
        # 获取所有整数和浮点数类型的类型码
        types = np.typecodes['AllInteger'] + np.typecodes['Float']
        # 创建包含两个相同字符串的列表
        nstr = ['123', '123']
        # 创建预期结果为整数类型的数组
        result = np.array([123, 123], dtype=int)
        # 遍历每种数据类型进行测试
        for type in types:
            # 构建消息字符串
            msg = 'String conversion for %s' % type
            # 断言字符串数组与预期结果相等
            assert_equal(np.array(nstr, dtype=type), result, err_msg=msg)

    # 测试 void 类型数组的函数
    def test_void(self):
        # 创建空的 void 类型数组
        arr = np.array([], dtype='V')
        # 断言数组的数据类型为默认的 'V8'
        assert arr.dtype == 'V8'  # current default
        # 创建包含两个相同字节串的 void 类型数组
        arr = np.array([b"1234", b"1234"], dtype="V")
        # 断言数组的数据类型为 'V4'
        assert arr.dtype == "V4"

        # 尝试将不同长度的字节串转换为 void 类型数组会引发 TypeError（在1.20版本之前是有效的）
        with pytest.raises(TypeError):
            np.array([b"1234", b"12345"], dtype="V")
        with pytest.raises(TypeError):
            np.array([b"12345", b"1234"], dtype="V")

        # 检查强制类型转换路径
        arr = np.array([b"1234", b"1234"], dtype="O").astype("V")
        # 断言数组的数据类型为 'V4'
        assert arr.dtype == "V4"
        with pytest.raises(TypeError):
            np.array([b"1234", b"12345"], dtype="O").astype("V")

    # 使用参数化测试标记测试结构化 void 类型数组的促进
    @pytest.mark.parametrize("idx",
            [pytest.param(Ellipsis, id="arr"), pytest.param((), id="scalar")])
    def test_structured_void_promotion(self, idx):
        # 创建结构化的 void 类型数组
        arr = np.array(
            [np.array(1, dtype="i,i")[idx], np.array(2, dtype='i,i')[idx]],
            dtype="V")
        # 断言数组与预期结果相等
        assert_array_equal(arr, np.array([(1, 1), (2, 2)], dtype="i,i"))
        # 尝试创建不同长度的结构化 void 类型数组会引发 TypeError
        with pytest.raises(TypeError):
            np.array(
                [np.array(1, dtype="i,i")[idx], np.array(2, dtype='i,i,i')[idx]],
                dtype="V")

    # 测试超出尺寸限制时引发错误的函数
    def test_too_big_error(self):
        # 根据系统位数设置数组的形状
        if np.iinfo('intp').max == 2**31 - 1:
            shape = (46341, 46341)
        elif np.iinfo('intp').max == 2**63 - 1:
            shape = (3037000500, 3037000500)
        else:
            return
        # 断言尝试创建超出系统限制的数组会引发 ValueError
        assert_raises(ValueError, np.empty, shape, dtype=np.int8)
        assert_raises(ValueError, np.zeros, shape, dtype=np.int8)
        assert_raises(ValueError, np.ones, shape, dtype=np.int8)

    # 根据系统位数设置的内存分配失败时引发异常的测试
    @pytest.mark.skipif(np.dtype(np.intp).itemsize != 8,
                        reason="malloc may not fail on 32 bit systems")
    def test_malloc_fails(self):
        # 这个测试保证会因为过大的内存分配而失败
        with assert_raises(np._core._exceptions._ArrayMemoryError):
            np.empty(np.iinfo(np.intp).max, dtype=np.uint8)
    def test_zeros(self):
        # 获取所有整数和浮点数类型的 typecode，作为测试数据类型
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        # 遍历每种数据类型进行测试
        for dt in types:
            # 创建一个形状为 (13,) 的零数组，指定数据类型为 dt
            d = np.zeros((13,), dtype=dt)
            # 断言数组中非零元素的数量为 0
            assert_equal(np.count_nonzero(d), 0)
            # 对于 IEEE 浮点数，断言数组所有元素的和为 0
            assert_equal(d.sum(), 0)
            # 断言数组中没有任何元素为真（非零）
            assert_(not d.any())

            # 创建一个形状为 (2,) 的零数组，数据类型为 '(2,4)i4'
            d = np.zeros(2, dtype='(2,4)i4')
            # 断言数组中非零元素的数量为 0
            assert_equal(np.count_nonzero(d), 0)
            # 断言数组所有元素的和为 0
            assert_equal(d.sum(), 0)
            # 断言数组中没有任何元素为真（非零）
            assert_(not d.any())

            # 创建一个形状为 (2,) 的零数组，数据类型为 '4i4'
            d = np.zeros(2, dtype='4i4')
            # 断言数组中非零元素的数量为 0
            assert_equal(np.count_nonzero(d), 0)
            # 断言数组所有元素的和为 0
            assert_equal(d.sum(), 0)
            # 断言数组中没有任何元素为真（非零）
            assert_(not d.any())

            # 创建一个形状为 (2,) 的零数组，数据类型为 '(2,4)i4, (2,4)i4'
            d = np.zeros(2, dtype='(2,4)i4, (2,4)i4')
            # 断言数组中非零元素的数量为 0
            assert_equal(np.count_nonzero(d), 0)

    @pytest.mark.slow
    def test_zeros_big(self):
        # 测试大数组分配，因为它们可能会因为系统不同而分配不同
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        # 遍历每种数据类型进行测试
        for dt in types:
            # 创建一个形状为 (30 * 1024**2,) 的零数组，指定数据类型为 dt
            d = np.zeros((30 * 1024**2,), dtype=dt)
            # 断言数组中没有任何元素为真（非零）
            assert_(not d.any())
            # 在 32 位系统上，由于内存不足，此测试可能失败。删除前一个数组可以增加成功的机会。
            del(d)

    def test_zeros_obj(self):
        # 测试从 PyLong(0) 初始化
        # 创建一个形状为 (13,) 的零数组，数据类型为 object
        d = np.zeros((13,), dtype=object)
        # 断言数组中的所有元素都等于列表 [0] * 13
        assert_array_equal(d, [0] * 13)
        # 断言数组中非零元素的数量为 0
        assert_equal(np.count_nonzero(d), 0)

    def test_zeros_obj_obj(self):
        # 创建一个形状为 10 的零数组，数据类型为 [('k', object, 2)]
        d = np.zeros(10, dtype=[('k', object, 2)])
        # 断言数组的 'k' 列中的所有元素都等于 0
        assert_array_equal(d['k'], 0)

    def test_zeros_like_like_zeros(self):
        # 测试 zeros_like 返回与 zeros 相同的数组
        for c in np.typecodes['All']:
            if c == 'V':
                continue
            # 创建一个形状为 (3,3) 的零数组，数据类型为 c
            d = np.zeros((3,3), dtype=c)
            # 断言 np.zeros_like 返回与 d 相同的数组
            assert_array_equal(np.zeros_like(d), d)
            # 断言 np.zeros_like 返回的数组数据类型与 d 相同
            assert_equal(np.zeros_like(d).dtype, d.dtype)
        
        # 显式检查一些特殊情况
        # 创建一个形状为 (3,3) 的零数组，数据类型为 'S5'
        d = np.zeros((3,3), dtype='S5')
        # 断言 np.zeros_like 返回与 d 相同的数组
        assert_array_equal(np.zeros_like(d), d)
        # 断言 np.zeros_like 返回的数组数据类型与 d 相同
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        
        # 创建一个形状为 (3,3) 的零数组，数据类型为 'U5'
        d = np.zeros((3,3), dtype='U5')
        # 断言 np.zeros_like 返回与 d 相同的数组
        assert_array_equal(np.zeros_like(d), d)
        # 断言 np.zeros_like 返回的数组数据类型与 d 相同
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        
        # 创建一个形状为 (3,3) 的零数组，数据类型为 '<i4'
        d = np.zeros((3,3), dtype='<i4')
        # 断言 np.zeros_like 返回与 d 相同的数组
        assert_array_equal(np.zeros_like(d), d)
        # 断言 np.zeros_like 返回的数组数据类型与 d 相同
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        
        # 创建一个形状为 (3,3) 的零数组，数据类型为 '>i4'
        d = np.zeros((3,3), dtype='>i4')
        # 断言 np.zeros_like 返回与 d 相同的数组
        assert_array_equal(np.zeros_like(d), d)
        # 断言 np.zeros_like 返回的数组数据类型与 d 相同
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        
        # 创建一个形状为 (3,3) 的零数组，数据类型为 '<M8[s]'
        d = np.zeros((3,3), dtype='<M8[s]')
        # 断言 np.zeros_like 返回与 d 相同的数组
        assert_array_equal(np.zeros_like(d), d)
        # 断言 np.zeros_like 返回的数组数据类型与 d 相同
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        
        # 创建一个形状为 (3,3) 的零数组，数据类型为 '>M8[s]'
        d = np.zeros((3,3), dtype='>M8[s]')
        # 断言 np.zeros_like 返回与 d 相同的数组
        assert_array_equal(np.zeros_like(d), d)
        # 断言 np.zeros_like 返回的数组数据类型与 d 相同
        assert_equal(np.zeros_like(d).dtype, d.dtype)
        
        # 创建一个形状为 (3,3) 的零数组，数据类型为 'f4,f4'
        d = np.zeros((3,3), dtype='f4,f4')
        # 断言 np.zeros_like 返回与 d 相同的数组
        assert_array_equal(np.zeros_like(d), d)
        # 断言 np.zeros_like 返回的数组数据类型与 d 相同
        assert_equal(np.zeros_like(d).dtype, d.dtype)
    def test_empty_unicode(self):
        # 在处理垃圾内存时，不要抛出解码错误
        for i in range(5, 100, 5):
            # 创建指定长度的空 Unicode 字符串数组
            d = np.empty(i, dtype='U')
            # 将数组转换为字符串形式

    def test_sequence_non_homogeneous(self):
        # 断言：数组包含非同质元素时，数据类型应为 object
        assert_equal(np.array([4, 2**80]).dtype, object)
        assert_equal(np.array([4, 2**80, 4]).dtype, object)
        assert_equal(np.array([2**80, 4]).dtype, object)
        assert_equal(np.array([2**80] * 3).dtype, object)
        assert_equal(np.array([[1, 1],[1j, 1j]]).dtype, complex)
        assert_equal(np.array([[1j, 1j],[1, 1]]).dtype, complex)
        assert_equal(np.array([[1, 1, 1],[1, 1j, 1.], [1, 1, 1]]).dtype, complex)

    def test_non_sequence_sequence(self):
        """不应该导致段错误。

        Class Fail 打破了新式类的序列协议，即继承自 object 的类。Class Map 是映射类型，通过引发 ValueError 表示。在 Fail 情况下，可能会在某些时候改为引发警告而不是错误。

        """
        class Fail:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise ValueError()

        class Map:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise KeyError()

        a = np.array([Map()])
        assert_(a.shape == (1,))
        assert_(a.dtype == np.dtype(object))
        assert_raises(ValueError, np.array, [Fail()])

    def test_no_len_object_type(self):
        # gh-5100, 从没有 len() 方法的可迭代对象创建对象数组
        class Point2:
            def __init__(self):
                pass

            def __getitem__(self, ind):
                if ind in [0, 1]:
                    return ind
                else:
                    raise IndexError()
        d = np.array([Point2(), Point2(), Point2()])
        assert_equal(d.dtype, np.dtype(object))

    def test_false_len_sequence(self):
        # gh-7264, 此示例可能导致段错误
        class C:
            def __getitem__(self, i):
                raise IndexError
            def __len__(self):
                return 42

        a = np.array(C()) # 可能导致段错误？
        assert_equal(len(a), 0)

    def test_false_len_iterable(self):
        # 特殊情况，其中错误的 __getitem__ 方法使我们回退到 __iter__：
        class C:
            def __getitem__(self, x):
                raise Exception
            def __iter__(self):
                return iter(())
            def __len__(self):
                return 2

        a = np.empty(2)
        with assert_raises(ValueError):
            a[:] = C()  # Segfault!

        np.array(C()) == list(C())
    def test_failed_len_sequence(self):
        # 定义一个内部类 A，用于模拟一个支持 __getitem__ 和 __len__ 方法的类
        class A:
            def __init__(self, data):
                self._data = data
            def __getitem__(self, item):
                return type(self)(self._data[item])
            def __len__(self):
                return len(self._data)

        # 创建一个 A 类的实例 d，其内部数据为 [1,2,3]
        d = A([1,2,3])
        # 断言 np.array(d) 的长度为 3
        assert_equal(len(np.array(d)), 3)

    def test_array_too_big(self):
        # 测试当数组的大小超出限制时的行为，使用 np.iinfo(np.intp).max 来获取最大字节数
        buf = np.zeros(100)

        max_bytes = np.iinfo(np.intp).max
        for dtype in ["intp", "S20", "b"]:
            dtype = np.dtype(dtype)
            itemsize = dtype.itemsize

            # 创建一个 ndarray，使用给定的 buffer、strides 和 dtype 参数
            np.ndarray(buffer=buf, strides=(0,),
                       shape=(max_bytes//itemsize,), dtype=dtype)
            # 断言创建超出限制大小的数组会抛出 ValueError 异常
            assert_raises(ValueError, np.ndarray, buffer=buf, strides=(0,),
                          shape=(max_bytes//itemsize + 1,), dtype=dtype)

    def _ragged_creation(self, seq):
        # 当创建的数组中存在不同深度的嵌套列表时，如果没有指定 dtype=object，会抛出 ValueError 异常
        with pytest.raises(ValueError, match=".*detected shape was"):
            a = np.array(seq)

        # 返回指定 dtype=object 的数组
        return np.array(seq, dtype=object)

    def test_ragged_ndim_object(self):
        # 测试不同深度嵌套列表的行为，验证是否会转换为 object 类型的数组
        a = self._ragged_creation([[1], 2, 3])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([1, [2], 3])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([1, 2, [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

    def test_ragged_shape_object(self):
        # 测试列表的不规则维度如何转换为 object 类型的数组
        a = self._ragged_creation([[1, 1], [2], [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([[1], [2, 2], [3]])
        assert_equal(a.shape, (3,))
        assert_equal(a.dtype, object)

        a = self._ragged_creation([[1], [2], [3, 3]])
        assert a.shape == (3,)
        assert a.dtype == object

    def test_array_of_ragged_array(self):
        # 测试包含不规则数组的数组的行为
        outer = np.array([None, None])
        outer[0] = outer[1] = np.array([1, 2, 3])
        assert np.array(outer).shape == (2,)
        assert np.array([outer]).shape == (1, 2)

        outer_ragged = np.array([None, None])
        outer_ragged[0] = np.array([1, 2, 3])
        outer_ragged[1] = np.array([1, 2, 3, 4])
        # 断言 np.array(outer_ragged) 的形状为 (2,)
        assert np.array(outer_ragged).shape == (2,)
        # 断言 np.array([outer_ragged]) 的形状为 (1, 2)
        assert np.array([outer_ragged]).shape == (1, 2,)
    # 定义测试方法，用于测试包含深层嵌套且非不规则的对象的情况
    def test_deep_nonragged_object(self):
        # 下列数组的创建不应该引发异常，即使它们没有指定 dtype=object
        a = np.array([[[Decimal(1)]]])  # 创建一个三层深度的数组，包含 Decimal(1)
        a = np.array([1, Decimal(1)])   # 创建一个包含整数 1 和 Decimal(1) 的数组
        a = np.array([[1], [Decimal(1)]])  # 创建一个包含两个子数组的数组，每个子数组包含一个整数或一个 Decimal

    # 使用参数化测试框架 pytest.mark.parametrize 注解此方法
    @pytest.mark.parametrize("dtype", [object, "O,O", "O,(3,)O", "(2,3)O"])
    @pytest.mark.parametrize("function", [
            np.ndarray, np.empty,
            lambda shape, dtype: np.empty_like(np.empty(shape, dtype=dtype))])
    # 测试对象数组初始化为 None 的情况
    def test_object_initialized_to_None(self, function, dtype):
        # NumPy 支持将对象字段初始化为 NULL（即 None）
        # 但通常情况下，我们应该始终填充正确的 None，下游代码可能依赖于此（对于完全初始化的数组！）
        arr = function(3, dtype=dtype)
        # 我们期望的填充值是 None，它并不是 NULL：
        expected = np.array(None).tobytes()
        expected = expected * (arr.nbytes // len(expected))
        # 断言数组的二进制表示与期望值相同
        assert arr.tobytes() == expected

    # 使用参数化测试框架 pytest.mark.parametrize 注解此方法
    @pytest.mark.parametrize("func", [
        np.array, np.asarray, np.asanyarray, np.ascontiguousarray,
        np.asfortranarray])
    # 测试从 dtype 元信息创建数组的情况
    def test_creation_from_dtypemeta(self, func):
        dtype = np.dtype('i')  # 创建一个整数类型的 dtype
        arr1 = func([1, 2, 3], dtype=dtype)
        arr2 = func([1, 2, 3], dtype=type(dtype))
        # 断言两个数组是否相等
        assert_array_equal(arr1, arr2)
        # 断言 arr2 的 dtype 与 dtype 相同
        assert arr2.dtype == dtype
class TestStructured:
    def test_subarray_field_access(self):
        # 创建一个形状为 (3, 5) 的数组，每个元素是一个带有 ('a', ('i4', (2, 2))) 结构的数据类型
        a = np.zeros((3, 5), dtype=[('a', ('i4', (2, 2)))])
        # 使用 arange 方法生成一个包含 60 个元素的数组，并按照指定形状重新排列成 (3, 5, 2, 2) 的形式，将其赋值给字段 'a'
        a['a'] = np.arange(60).reshape(3, 5, 2, 2)

        # 因为子数组始终按照 C 顺序存储，所以转置不会交换子数组
        assert_array_equal(a.T['a'], a['a'].transpose(1, 0, 2, 3))

        # 在 Fortran 顺序中，子数组会被附加，不像其他情况下作为特殊情况被前置
        b = a.copy(order='F')
        # 检查字段 'a' 的形状是否相同
        assert_equal(a['a'].shape, b['a'].shape)
        # 检查转置后字段 'a' 的形状是否与复制后再转置的字段 'a' 的形状相同
        assert_equal(a.T['a'].shape, a.T.copy()['a'].shape)
    # 定义一个测试函数，用于测试子数组的比较操作
    def test_subarray_comparison(self):
        # 检查对具有多维字段类型的记录数组进行比较操作的正确性
        a = np.rec.fromrecords(
            [([1, 2, 3], 'a', [[1, 2], [3, 4]]), ([3, 3, 3], 'b', [[0, 0], [0, 0]])],
            dtype=[('a', ('f4', 3)), ('b', object), ('c', ('i4', (2, 2)))])
        b = a.copy()
        # 断言两个记录数组是否相等
        assert_equal(a == b, [True, True])
        # 断言两个记录数组是否不相等
        assert_equal(a != b, [False, False])
        # 修改 b 中的一个字段，检查比较操作的结果是否正确
        b[1].b = 'c'
        assert_equal(a == b, [True, False])
        assert_equal(a != b, [False, True])
        # 对数组中的特定元素进行迭代，并修改其中的值，验证比较操作的正确性
        for i in range(3):
            b[0].a = a[0].a
            b[0].a[i] = 5
            assert_equal(a == b, [False, False])
            assert_equal(a != b, [True, True])
        # 对数组中的特定子数组进行迭代，并修改其值，验证比较操作的正确性
        for i in range(2):
            for j in range(2):
                b = a.copy()
                b[0].c[i, j] = 10
                assert_equal(a == b, [False, True])
                assert_equal(a != b, [True, False])

        # 检查包含子数组的广播操作，包括需要升级类型才能进行比较的情况：
        a = np.array([[(0,)], [(1,)]], dtype=[('a', 'f8')])
        b = np.array([(0,), (0,), (1,)], dtype=[('a', 'f8')])
        assert_equal(a == b, [[True, True, False], [False, False, True]])
        assert_equal(b == a, [[True, True, False], [False, False, True]])
        a = np.array([[(0,)], [(1,)]], dtype=[('a', 'f8', (1,))])
        b = np.array([(0,), (0,), (1,)], dtype=[('a', 'f8', (1,))])
        assert_equal(a == b, [[True, True, False], [False, False, True]])
        assert_equal(b == a, [[True, True, False], [False, False, True]])
        a = np.array([[([0, 0],)], [([1, 1],)]], dtype=[('a', 'f8', (2,))])
        b = np.array([([0, 0],), ([0, 1],), ([1, 1],)], dtype=[('a', 'f8', (2,))])
        assert_equal(a == b, [[True, False, False], [False, False, True]])
        assert_equal(b == a, [[True, False, False], [False, False, True]])

        # 检查使用子数组的 Fortran 风格数组的广播操作
        a = np.array([[([0, 0],)], [([1, 1],)]], dtype=[('a', 'f8', (2,))], order='F')
        b = np.array([([0, 0],), ([0, 1],), ([1, 1],)], dtype=[('a', 'f8', (2,))])
        assert_equal(a == b, [[True, False, False], [False, False, True]])
        assert_equal(b == a, [[True, False, False], [False, False, True]])

        # 检查不兼容的子数组形状是否不会进行广播
        x = np.zeros((1,), dtype=[('a', ('f4', (1, 2))), ('b', 'i1')])
        y = np.zeros((1,), dtype=[('a', ('f4', (2,))), ('b', 'i1')])
        # 主要验证不会返回 True：
        with pytest.raises(TypeError):
            x == y

        x = np.zeros((1,), dtype=[('a', ('f4', (2, 1))), ('b', 'i1')])
        y = np.zeros((1,), dtype=[('a', ('f4', (2,))), ('b', 'i1')])
        # 主要验证不会返回 True：
        with pytest.raises(TypeError):
            x == y
    def test_empty_structured_array_comparison(self):
        # 检查空数组的比较，尤其是在具有复杂形状字段的情况下
        a = np.zeros(0, [('a', '<f8', (1, 1))])
        assert_equal(a, a)
        a = np.zeros(0, [('a', '<f8', (1,))])
        assert_equal(a, a)
        a = np.zeros((0, 0), [('a', '<f8', (1, 1))])
        assert_equal(a, a)
        a = np.zeros((1, 0, 1), [('a', '<f8', (1, 1))])
        assert_equal(a, a)

    @pytest.mark.parametrize("op", [operator.eq, operator.ne])
    def test_structured_array_comparison_bad_broadcasts(self, op):
        # 检查结构化数组比较时的错误广播情况
        a = np.zeros(3, dtype='i,i')
        b = np.array([], dtype="i,i")
        with pytest.raises(ValueError):
            op(a, b)

    def test_structured_comparisons_with_promotion(self):
        # 检查结构化数组可以进行比较，只要它们的 dtype 能够正确升级：
        a = np.array([(5, 42), (10, 1)], dtype=[('a', '>i8'), ('b', '<f8')])
        b = np.array([(5, 43), (10, 1)], dtype=[('a', '<i8'), ('b', '>f8')])
        assert_equal(a == b, [False, True])
        assert_equal(a != b, [True, False])

        a = np.array([(5, 42), (10, 1)], dtype=[('a', '>f8'), ('b', '<f8')])
        b = np.array([(5, 43), (10, 1)], dtype=[('a', '<i8'), ('b', '>i8')])
        assert_equal(a == b, [False, True])
        assert_equal(a != b, [True, False])

        # 包括嵌套的子数组 dtype（尽管子数组比较本身可能仍然有点奇怪，并比较原始数据）
        a = np.array([(5, 42), (10, 1)], dtype=[('a', '10>f8'), ('b', '5<f8')])
        b = np.array([(5, 43), (10, 1)], dtype=[('a', '10<i8'), ('b', '5>i8')])
        assert_equal(a == b, [False, True])
        assert_equal(a != b, [True, False])

    @pytest.mark.parametrize("op", [
            operator.eq, lambda x, y: operator.eq(y, x),
            operator.ne, lambda x, y: operator.ne(y, x)])
    def test_void_comparison_failures(self, op):
        # 原则上，如果比较不可能，可以决定返回一个全为 False 的数组。
        # 但是目前当涉及 "void" dtype 时，我们会返回 TypeError。
        x = np.zeros(3, dtype=[('a', 'i1')])
        y = np.zeros(3)
        # 无法比较非结构化数组和结构化数组：
        with pytest.raises(TypeError):
            op(x, y)

        # 添加了标题可以阻止升级，但是类型转换是可以的：
        y = np.zeros(3, dtype=[(('title', 'a'), 'i1')])
        assert np.can_cast(y.dtype, x.dtype)
        with pytest.raises(TypeError):
            op(x, y)

        x = np.zeros(3, dtype="V7")
        y = np.zeros(3, dtype="V8")
        with pytest.raises(TypeError):
            op(x, y)
    def test_casting(self):
        # 检查将结构化数组进行类型转换以改变其字节顺序是否有效
        a = np.array([(1,)], dtype=[('a', '<i4')])
        assert_(np.can_cast(a.dtype, [('a', '>i4')], casting='unsafe'))
        # 将数组 a 转换为指定的数据类型 [('a', '>i4')]
        b = a.astype([('a', '>i4')])
        # 对数组 a 进行字节交换
        a_tmp = a.byteswap()
        # 将交换字节后的数组视图改为指定字节顺序的视图
        a_tmp = a_tmp.view(a_tmp.dtype.newbyteorder())
        # 检查转换后的数组 b 是否与处理后的数组 a_tmp 相等
        assert_equal(b, a_tmp)
        # 检查数组 a 和 b 中的元素是否相等
        assert_equal(a['a'][0], b['a'][0])

        # 检查如果结构化数组在‘equiv’转换时是否可以进行相等性比较
        a = np.array([(5, 42), (10, 1)], dtype=[('a', '>i4'), ('b', '<f8')])
        b = np.array([(5, 42), (10, 1)], dtype=[('a', '<i4'), ('b', '>f8')])
        assert_(np.can_cast(a.dtype, b.dtype, casting='equiv'))
        # 检查数组 a 和 b 中的元素是否相等
        assert_equal(a == b, [True, True])

        # 检查‘equiv’转换是否可以改变字节顺序
        assert_(np.can_cast(a.dtype, b.dtype, casting='equiv'))
        c = a.astype(b.dtype, casting='equiv')
        # 检查数组 a 和 c 中的元素是否相等
        assert_equal(a == c, [True, True])

        # 检查‘safe’转换是否可以改变字节顺序并且可以升级字段
        t = [('a', '<i8'), ('b', '>f8')]
        assert_(np.can_cast(a.dtype, t, casting='safe'))
        c = a.astype(t, casting='safe')
        # 检查数组 c 是否与指定的数组相等
        assert_equal((c == np.array([(5, 42), (10, 1)], dtype=t)),
                     [True, True])

        # 检查‘same_kind’转换是否可以改变字节顺序并且在“kind”内部改变字段宽度
        t = [('a', '<i4'), ('b', '>f4')]
        assert_(np.can_cast(a.dtype, t, casting='same_kind'))
        c = a.astype(t, casting='same_kind')
        # 检查数组 c 是否与指定的数组相等
        assert_equal((c == np.array([(5, 42), (10, 1)], dtype=t)),
                     [True, True])

        # 检查如果在任何字段上的转换规则应该失败，则转换是否会失败
        t = [('a', '>i8'), ('b', '<f4')]
        assert_(not np.can_cast(a.dtype, t, casting='safe'))
        assert_raises(TypeError, a.astype, t, casting='safe')
        t = [('a', '>i2'), ('b', '<f8')]
        assert_(not np.can_cast(a.dtype, t, casting='equiv'))
        assert_raises(TypeError, a.astype, t, casting='equiv')
        t = [('a', '>i8'), ('b', '<i2')]
        assert_(not np.can_cast(a.dtype, t, casting='same_kind'))
        assert_raises(TypeError, a.astype, t, casting='same_kind')
        assert_(not np.can_cast(a.dtype, b.dtype, casting='no'))
        assert_raises(TypeError, a.astype, b.dtype, casting='no')

        # 检查非‘unsafe’转换是否不能改变字段名称集合
        for casting in ['no', 'safe', 'equiv', 'same_kind']:
            t = [('a', '>i4')]
            assert_(not np.can_cast(a.dtype, t, casting=casting))
            t = [('a', '>i4'), ('b', '<f8'), ('c', 'i4')]
            assert_(not np.can_cast(a.dtype, t, casting=casting))
    def test_objview(self):
        # 创建一个空的结构化 NumPy 数组，其中包含三个字段：'a' 和 'b' 是浮点型，'c' 是对象型
        a = np.array([], dtype=[('a', 'f'), ('b', 'f'), ('c', 'O')])
        # 尝试访问不存在的字段 'a' 和 'b'，可能会引发 TypeError
        a[['a', 'b']]  # TypeError?

        # 创建一个长度为 3 的零数组，其中包含两个字段：'A' 是整数型，'B' 是对象型
        dat2 = np.zeros(3, [('A', 'i'), ('B', '|O')])
        # 尝试访问字段 'B' 和 'A'，可能会引发 TypeError
        dat2[['B', 'A']]  # TypeError?

    def test_setfield(self):
        # 定义一个结构化数据类型 struct_dt，包含一个名为 'elem' 的整数型数组（长度为 5）
        struct_dt = np.dtype([('elem', 'i4', 5),])
        # 定义一个结构化数据类型 dt，包含两个字段：'field' 是整数型数组（长度为 10），'struct' 是前面定义的 struct_dt 类型
        dt = np.dtype([('field', 'i4', 10), ('struct', struct_dt)])
        # 创建一个形状为 (1,) 的零数组 x，使用定义的数据类型 dt
        x = np.zeros(1, dt)
        # 将 x[0]['field'] 赋值为一个包含 10 个整数 1 的数组
        x[0]['field'] = np.ones(10, dtype='i4')
        # 将 x[0]['struct'] 赋值为一个包含一个 struct_dt 类型的数组，其中每个元素都是整数 1
        x[0]['struct'] = np.ones(1, dtype=struct_dt)
        # 断言 x[0]['field'] 的值与包含 10 个整数 1 的数组相等
        assert_equal(x[0]['field'], np.ones(10, dtype='i4'))

    def test_setfield_object(self):
        # 创建一个形状为 (1,) 的零数组 b，包含一个名为 'x' 的对象型字段
        b = np.zeros(1, dtype=[('x', 'O')])
        # 将 b[0]['x'] 赋值为一个包含整数序列 0 到 2 的数组
        # 这行代码应当与 b['x'][0] = np.arange(3) 的行为相同
        b[0]['x'] = np.arange(3)
        # 断言 b[0]['x'] 的值与整数序列 0 到 2 的数组相等
        assert_equal(b[0]['x'], np.arange(3))

        # 创建一个形状为 (1,) 的零数组 c，包含一个名为 'x' 的对象型数组字段，长度为 5
        c = np.zeros(1, dtype=[('x', 'O', 5)])

        def testassign():
            # 尝试将 c[0]['x'] 赋值为一个长度不同于字段定义的数组，这应当引发 ValueError
            c[0]['x'] = np.arange(3)

        # 断言调用 testassign 函数会引发 ValueError
        assert_raises(ValueError, testassign)
    def test_zero_width_string(self):
        # Test for PR #6430 / issues #473, #4955, #2585
        
        # 定义一个结构化数据类型 `dt`，包含两个字段：'I' (整数) 和 'S' (零长度字符串)
        dt = np.dtype([('I', int), ('S', 'S0')])
        
        # 创建一个长度为4的零数组，使用定义好的数据类型 `dt`
        x = np.zeros(4, dtype=dt)
        
        # 断言字段 'S' 应该都是空字符串的字节表示，长度为0
        assert_equal(x['S'], [b'', b'', b'', b''])
        assert_equal(x['S'].itemsize, 0)
        
        # 将字段 'S' 的值设为 ['a', 'b', 'c', 'd']，但由于是零长度字符串，不应该改变
        assert_equal(x['S'], [b'', b'', b'', b''])
        assert_equal(x['I'], [0, 0, 0, 0])
        
        # 当 'I' 字段为0时，将 'S' 字段设为 'hello'，但由于 'S' 是零长度字符串，不应该改变
        x['S'][x['I'] == 0] = 'hello'
        assert_equal(x['S'], [b'', b'', b'', b''])
        assert_equal(x['I'], [0, 0, 0, 0])
        
        # 将 'S' 字段设为 'A'，同样不应该改变因为 'S' 是零长度字符串
        x['S'] = 'A'
        assert_equal(x['S'], [b'', b'', b'', b''])
        assert_equal(x['I'], [0, 0, 0, 0])
        
        # 使用 x['S'].dtype 创建另一个数组 y，其数据类型为零长度字符串的类型，验证其长度为0
        y = np.ndarray(4, dtype=x['S'].dtype)
        assert_equal(y.itemsize, 0)
        assert_equal(x['S'], y)
        
        # 更多测试：索引包含零长度字段的数组，验证字段 'a' 的长度为0
        assert_equal(np.zeros(4, dtype=[('a', 'S0,S0'), ('b', 'u1')])['a'].itemsize, 0)
        assert_equal(np.empty(3, dtype='S0,S0').itemsize, 0)
        assert_equal(np.zeros(4, dtype='S0,u1')['f0'].itemsize, 0)
        
        # 将 x['S'] 重塑为 (2, 2) 的数组 xx，验证其长度为0
        xx = x['S'].reshape((2, 2))
        assert_equal(xx.itemsize, 0)
        assert_equal(xx, [[b'', b''], [b'', b'']])
        
        # 验证由于查看 S0 数组，没有未初始化的内存
        assert_equal(xx[:].dtype, xx.dtype)
        assert_array_equal(eval(repr(xx), dict(np=np, array=np.array)), xx)
        
        # 将 xx 保存到 BytesIO 中，并加载验证其长度为0
        b = io.BytesIO()
        np.save(b, xx)
        
        b.seek(0)
        yy = np.load(b)
        assert_equal(yy.itemsize, 0)
        assert_equal(xx, yy)
        
        # 使用临时路径保存和加载 xx，验证其长度为0
        with temppath(suffix='.npy') as tmp:
            np.save(tmp, xx)
            yy = np.load(tmp)
            assert_equal(yy.itemsize, 0)
            assert_equal(xx, yy)

    def test_base_attr(self):
        # 创建一个包含 'i4' 和 'f4' 类型的数组 a
        a = np.zeros(3, dtype='i4,f4')
        
        # 获取数组 a 的第一个元素并赋值给 b，断言 b 的 base 属性指向 a
        b = a[0]
        assert_(b.base is a)
    def test_assignment(self):
        # 定义一个内部函数 testassign，用于测试数组赋值操作
        def testassign(arr, v):
            # 复制数组 arr
            c = arr.copy()
            # 使用索引赋值方式修改 c 的第一个元素为 v
            c[0] = v  # assign using setitem
            # 使用 "dtype_transfer" 代码路径赋值方式修改 c 的其余元素为 v
            c[1:] = v # assign using "dtype_transfer" code paths
            return c

        # 定义一个结构化数据类型 dt，包含两个字段 foo 和 bar，类型均为 'i8'（64位整数）
        dt = np.dtype([('foo', 'i8'), ('bar', 'i8')])
        # 创建一个包含两个元素的数组 arr，元素类型为 dt 所定义的结构化类型，初始值为全 1
        arr = np.ones(2, dt)
        # 定义多个结构化数组作为测试用例
        v1 = np.array([(2,3)], dtype=[('foo', 'i8'), ('bar', 'i8')])
        v2 = np.array([(2,3)], dtype=[('bar', 'i8'), ('foo', 'i8')])
        v3 = np.array([(2,3)], dtype=[('bar', 'i8'), ('baz', 'i8')])
        v4 = np.array([(2,)],  dtype=[('bar', 'i8')])
        v5 = np.array([(2,3)], dtype=[('foo', 'f8'), ('bar', 'f8')])
        # 创建一个视图 w，从 arr 中视图出 bar 字段，元素类型为 'i8'
        w = arr.view({'names': ['bar'], 'formats': ['i8'], 'offsets': [8]})

        # 预期的结果数组，包含两个元素，每个元素为 (2, 3)，类型为 dt 所定义的结构化类型
        ans = np.array([(2,3),(2,3)], dtype=dt)
        
        # 断言测试不同情况下 testassign 函数的返回结果与预期结果 ans 相等
        assert_equal(testassign(arr, v1), ans)
        assert_equal(testassign(arr, v2), ans)
        assert_equal(testassign(arr, v3), ans)
        # 断言传入 v4 时会引发 TypeError 异常
        assert_raises(TypeError, lambda: testassign(arr, v4))
        assert_equal(testassign(arr, v5), ans)
        
        # 将视图 w 的所有元素赋值为 4
        w[:] = 4
        # 断言数组 arr 的值等于预期的结果数组，包含两个元素，每个元素为 (1, 4)，类型为 dt 所定义的结构化类型
        assert_equal(arr, np.array([(1,4),(1,4)], dtype=dt))

        # 测试字段重排、按位置赋值和自我赋值
        # 创建一个结构化数组 a，包含一个元素 (1,2,3)，字段分别为 foo（'i8'）、bar（'i8'）、baz（'f4'）
        a = np.array([(1,2,3)],
                     dtype=[('foo', 'i8'), ('bar', 'i8'), ('baz', 'f4')])
        # 使用字段名称进行赋值，将 foo 和 bar 字段的值互换
        a[['foo', 'bar']] = a[['bar', 'foo']]
        # 断言数组 a 的第一个元素值等于 (2, 1, 3)
        assert_equal(a[0].item(), (2,1,3))

        # 测试即使是 'simple_unaligned' 结构体，字段顺序也会影响赋值
        # 创建一个结构化数组 a，包含一个元素 (1, 2)，字段分别为 a（'i4'）、b（'i4'）
        a = np.array([(1,2)], dtype=[('a', 'i4'), ('b', 'i4')])
        # 使用字段名称进行赋值，将 a 和 b 字段的值互换
        a[['a', 'b']] = a[['b', 'a']]
        # 断言数组 a 的第一个元素值等于 (2, 1)
        assert_equal(a[0].item(), (2,1))

    def test_structuredscalar_indexing(self):
        # 测试 GitHub 问题编号为 gh-7262
        # 创建一个空数组 x，包含一个元素，数据类型为复合类型 [("f0", '3S'), ("f1", '3U')]
        x = np.empty(shape=1, dtype="(2,)3S,(2,)3U")
        # 断言数组 x 的第一个元素的 'f0' 和 'f1' 字段与 x[0] 相等
        assert_equal(x[["f0","f1"]][0], x[0][["f0","f1"]])
        # 断言数组 x 的第一个元素与 x[0][()] 相等
        assert_equal(x[0], x[0][()])

    def test_multiindex_titles(self):
        # 创建一个包含 4 个元素的数组 a，数据类型为 [("a", 'i'), ("c", 'i'), ("d", 'i')]
        a = np.zeros(4, dtype=[(('a', 'b'), 'i'), ('c', 'i'), ('d', 'i')])
        # 断言使用索引包含不存在的字段 'a' 和 'c' 会引发 KeyError 异常
        assert_raises(KeyError, lambda : a[['a','c']])
        # 断言使用索引包含重复字段 'a' 和 'a' 会引发 KeyError 异常
        assert_raises(KeyError, lambda : a[['a','a']])
        # 断言使用索引包含已存在但重复的字段 'b' 和 'b' 会引发 ValueError 异常
        assert_raises(ValueError, lambda : a[['b','b']])  # field exists, but repeated
        a[['b','c']]  # 不会引发异常，正常索引
    # 定义测试方法，用于测试结构化数组类型转换和促升的情况
    def test_structured_cast_promotion_fieldorder(self):
        # gh-15494
        # 当字段名不同的dtype不能进行促升
        A = ("a", "<i8")
        B = ("b", ">i8")
        ab = np.array([(1, 2)], dtype=[A, B])
        ba = np.array([(1, 2)], dtype=[B, A])
        # 断言ab和ba无法进行连接
        assert_raises(TypeError, np.concatenate, ab, ba)
        # 断言ab.dtype和ba.dtype无法进行结果类型的推断
        assert_raises(TypeError, np.result_type, ab.dtype, ba.dtype)
        # 断言ab.dtype和ba.dtype无法进行类型的促升
        assert_raises(TypeError, np.promote_types, ab.dtype, ba.dtype)

        # 具有相同字段名和顺序但内存偏移和字节顺序不同的dtype可以促升为打包的网络字节序
        assert_equal(np.promote_types(ab.dtype, ba[['a', 'b']].dtype),
                     repack_fields(ab.dtype.newbyteorder('N')))

        # gh-13667
        # 具有不同字段名但可以进行类型转换的dtype可以转换
        assert_equal(np.can_cast(ab.dtype, ba.dtype), True)
        # 断言通过类型转换后的ab的dtype与ba的dtype相同
        assert_equal(ab.astype(ba.dtype).dtype, ba.dtype)
        # 断言可以从'f8,i8'转换到[('f0', 'f8'), ('f1', 'i8')]的dtype
        assert_equal(np.can_cast('f8,i8', [('f0', 'f8'), ('f1', 'i8')]), True)
        # 断言可以从'f8,i8'转换到[('f1', 'f8'), ('f0', 'i8')]的dtype
        assert_equal(np.can_cast('f8,i8', [('f1', 'f8'), ('f0', 'i8')]), True)
        # 断言无法从'f8,i8'转换到[('f1', 'i8'), ('f0', 'f8')]的dtype
        assert_equal(np.can_cast('f8,i8', [('f1', 'i8'), ('f0', 'f8')]), False)
        # 断言可以从'f8,i8'转换到[('f1', 'i8'), ('f0', 'f8')]的dtype（使用'unsafe'选项）
        assert_equal(np.can_cast('f8,i8', [('f1', 'i8'), ('f0', 'f8')],
                                 casting='unsafe'), True)

        ab[:] = ba  # 确保赋值仍然有效

        # 测试相应字段的类型促升
        dt1 = np.dtype([("", "i4")])
        dt2 = np.dtype([("", "i8")])
        # 断言i4和i8可以成功促升为i8
        assert_equal(np.promote_types(dt1, dt2), np.dtype([('f0', 'i8')]))
        # 断言i8和i4可以成功促升为i8
        assert_equal(np.promote_types(dt2, dt1), np.dtype([('f0', 'i8')]))
        # 断言i4和V3类型无法进行类型促升
        assert_raises(TypeError, np.promote_types, dt1, np.dtype([("", "V3")]))
        # 断言'f8,i8'可以成功促升为'i8,f4'
        assert_equal(np.promote_types('i4,f8', 'i8,f4'),
                     np.dtype([('f0', 'i8'), ('f1', 'f8')]))
        # 测试嵌套情况
        dt1nest = np.dtype([("", dt1)])
        dt2nest = np.dtype([("", dt2)])
        # 断言嵌套的dt1和dt2可以成功促升
        assert_equal(np.promote_types(dt1nest, dt2nest),
                     np.dtype([('f0', np.dtype([('f0', 'i8')]))]))

        # 注意，促升时会丢失偏移量
        dt = np.dtype({'names': ['x'], 'formats': ['i4'], 'offsets': [8]})
        a = np.ones(3, dtype=dt)
        # 断言连接后的dtype丢失了偏移量信息
        assert_equal(np.concatenate([a, a]).dtype, np.dtype([('x', 'i4')]))

    @pytest.mark.parametrize("dtype_dict", [
            dict(names=["a", "b"], formats=["i4", "f"], itemsize=100),
            dict(names=["a", "b"], formats=["i4", "f"],
                 offsets=[0, 12])])
    @pytest.mark.parametrize("align", [True, False])
    # 测试结构化数据类型在类型提升时的行为
    def test_structured_promotion_packs(self, dtype_dict, align):
        # 结构化数据类型在指定对齐方式后创建
        dtype = np.dtype(dtype_dict, align=align)
        # 移除非“规范化”的数据类型选项
        dtype_dict.pop("itemsize", None)
        dtype_dict.pop("offsets", None)
        # 根据更新后的 dtype_dict 和对齐方式创建期望的数据类型
        expected = np.dtype(dtype_dict, align=align)

        # 测试类型提升函数，确保提升后的类型与期望的一致
        res = np.promote_types(dtype, dtype)
        assert res.itemsize == expected.itemsize
        assert res.fields == expected.fields

        # 对于相同的期望类型，类型提升函数应当返回其本身
        res = np.promote_types(expected, expected)
        assert res is expected

    # 测试结构化数组的标量视图行为
    def test_structured_asarray_is_view(self):
        # 创建一个结构化数组，其中包含一个包含两个整数的结构
        arr = np.array([1], dtype="i,i")
        # 获取数组的第一个标量作为视图
        scalar = arr[0]
        assert not scalar.flags.owndata  # 标量是数组的视图，不拥有数据
        assert np.asarray(scalar).base is scalar  # 转换标量仍然保持视图关系

        # 当传入 dtype 参数时，视图关系将被破坏
        assert np.asarray(scalar, dtype=scalar.dtype).base is None

        # 拥有数据的标量不具备保持视图的特性，通常可以通过 pickle 来实现
        scalar = pickle.loads(pickle.dumps(scalar))
        assert scalar.flags.owndata  # 标量现在拥有自己的数据
        assert np.asarray(scalar).base is None
class TestBool:
    def test_test_interning(self):
        # 创建一个 np.bool 类型的对象，表示 False
        a0 = np.bool(0)
        # 创建一个 np.bool 类型的对象，表示 False
        b0 = np.bool(False)
        # 断言 a0 和 b0 是同一个对象
        assert_(a0 is b0)
        # 创建一个 np.bool 类型的对象，表示 True
        a1 = np.bool(1)
        # 创建一个 np.bool 类型的对象，表示 True
        b1 = np.bool(True)
        # 断言 a1 和 b1 是同一个对象
        assert_(a1 is b1)
        # 断言 np.array([True])[0] 和 a1 是同一个对象
        assert_(np.array([True])[0] is a1)
        # 断言 np.array(True)[()] 和 a1 是同一个对象
        assert_(np.array(True)[()] is a1)

    def test_sum(self):
        # 创建一个包含 101 个 True 的布尔类型的 numpy 数组
        d = np.ones(101, dtype=bool)
        # 断言数组 d 的元素之和等于数组的大小
        assert_equal(d.sum(), d.size)
        # 断言数组 d 中偶数索引位置的元素之和等于偶数索引位置元素的个数
        assert_equal(d[::2].sum(), d[::2].size)
        # 断言数组 d 中倒序偶数索引位置的元素之和等于倒序偶数索引位置元素的个数
        assert_equal(d[::-2].sum(), d[::-2].size)

        # 从字节序列创建一个布尔类型的 numpy 数组，每个元素为 True
        d = np.frombuffer(b'\xff\xff' * 100, dtype=bool)
        # 断言数组 d 的元素之和等于数组的大小
        assert_equal(d.sum(), d.size)
        # 断言数组 d 中偶数索引位置的元素之和等于偶数索引位置元素的个数
        assert_equal(d[::2].sum(), d[::2].size)
        # 断言数组 d 中倒序偶数索引位置的元素之和等于倒序偶数索引位置元素的个数
        assert_equal(d[::-2].sum(), d[::-2].size)

    def check_count_nonzero(self, power, length):
        # 创建长度为 length 的 powers 数组，包含从 2^0 到 2^(length-1) 的幂次
        powers = [2 ** i for i in range(length)]
        # 遍历 0 到 2^power 之间的数字
        for i in range(2**power):
            # 创建一个长度为 length 的布尔类型的 numpy 数组
            l = [(i & x) != 0 for x in powers]
            # 将列表转换为 numpy 数组
            a = np.array(l, dtype=bool)
            # 计算数组 a 中非零元素的个数，并与预期的 c 进行断言比较
            c = builtins.sum(l)
            assert_equal(np.count_nonzero(a), c)
            # 将数组 a 视图转换为 uint8 类型的数组 av
            av = a.view(np.uint8)
            # av 数组每个元素乘以 3，不影响数组 a 的非零元素个数
            av *= 3
            assert_equal(np.count_nonzero(a), c)
            # av 数组每个元素乘以 4，不影响数组 a 的非零元素个数
            av *= 4
            assert_equal(np.count_nonzero(a), c)
            # 将 av 数组中不为 0 的元素设为 0xFF，不影响数组 a 的非零元素个数
            av[av != 0] = 0xFF
            assert_equal(np.count_nonzero(a), c)

    def test_count_nonzero(self):
        # 检查在长度为 17 的数组中所有 12 位组合
        # 涵盖大多数 16 字节展开代码的情况
        self.check_count_nonzero(12, 17)

    @pytest.mark.slow
    def test_count_nonzero_all(self):
        # 检查在长度为 17 的数组中所有组合
        # 涵盖所有 16 字节展开代码的情况
        self.check_count_nonzero(17, 17)

    def test_count_nonzero_unaligned(self):
        # 防止像 gh-4060 中的错误
        # 遍历 0 到 6 的范围
        for o in range(7):
            # 创建一个长度为 18 的布尔类型的 numpy 数组，从索引 o+1 开始为 True
            a = np.zeros((18,), dtype=bool)[o+1:]
            a[:o] = True
            # 断言数组 a 中非零元素的个数等于 a.tolist() 中非零元素的个数
            assert_equal(np.count_nonzero(a), builtins.sum(a.tolist()))
            # 创建一个长度为 18 的布尔类型的 numpy 数组，从索引 o+1 开始为 False
            a = np.ones((18,), dtype=bool)[o+1:]
            a[:o] = False
            # 断言数组 a 中非零元素的个数等于 a.tolist() 中非零元素的个数
            assert_equal(np.count_nonzero(a), builtins.sum(a.tolist()))

    def _test_cast_from_flexible(self, dtype):
        # 空字符串 -> False
        # 遍历 0 到 2 的范围
        for n in range(3):
            # 创建一个长度为 n 的 dtype 类型的 numpy 数组，内容为空字符串
            v = np.array(b'', (dtype, n))
            # 断言数组 v 转换为布尔类型后为 False
            assert_equal(bool(v), False)
            assert_equal(bool(v[()]), False)
            # 断言数组 v 转换为布尔类型后为 False
            assert_equal(v.astype(bool), False)
            # 断言数组 v 转换为布尔类型后是一个 numpy 数组
            assert_(isinstance(v.astype(bool), np.ndarray))
            # 断言数组 v[()] 转换为布尔类型后为 np.False_
            assert_(v[()].astype(bool) is np.False_)

        # 任何其他值 -> True
        # 遍历 1 到 3 的范围
        for n in range(1, 4):
            # 对于每个值为 b'a', b'0', b' ' 的元素
            for val in [b'a', b'0', b' ']:
                # 创建一个长度为 n 的 dtype 类型的 numpy 数组，内容为 val
                v = np.array(val, (dtype, n))
                # 断言数组 v 转换为布尔类型后为 True
                assert_equal(bool(v), True)
                assert_equal(bool(v[()]), True)
                # 断言数组 v 转换为布尔类型后为 True
                assert_equal(v.astype(bool), True)
                # 断言数组 v 转换为布尔类型后是一个 numpy 数组
                assert_(isinstance(v.astype(bool), np.ndarray))
                # 断言数组 v[()] 转换为布尔类型后为 np.True_
                assert_(v[()].astype(bool) is np.True_)

    def test_cast_from_void(self):
        # 测试从可变类型转换
        self._test_cast_from_flexible(np.void)

    @pytest.mark.xfail(reason="See gh-9847")
    # 定义一个测试方法，用于测试从 Unicode 类型转换的功能
    def test_cast_from_unicode(self):
        # 调用 _test_cast_from_flexible 方法，测试从 np.str_ 类型的灵活转换
        self._test_cast_from_flexible(np.str_)
    
    # 使用 pytest.mark.xfail 装饰器标记的测试方法，原因是“查看 gh-9847”（预期测试失败）
    def test_cast_from_bytes(self):
        # 调用 _test_cast_from_flexible 方法，测试从 np.bytes_ 类型的灵活转换
        self._test_cast_from_flexible(np.bytes_)
class TestZeroSizeFlexible:
    @staticmethod
    def _zeros(shape, dtype=str):
        # 将 dtype 转换为 numpy 的数据类型对象
        dtype = np.dtype(dtype)
        if dtype == np.void:
            # 如果 dtype 是 np.void，则创建一个空的 np.void 类型的数组
            return np.zeros(shape, dtype=(dtype, 0))

        # 否则，将 dtype 包装成一个结构化数据类型，只有一个字段 'x'
        dtype = np.dtype([('x', dtype, 0)])
        # 创建一个结构化数组，并返回其中 'x' 字段的内容
        return np.zeros(shape, dtype=dtype)['x']

    def test_create(self):
        # 测试用例：测试 _zeros 方法的返回结果是否 itemsize 为 0
        zs = self._zeros(10, bytes)
        assert_equal(zs.itemsize, 0)
        zs = self._zeros(10, np.void)
        assert_equal(zs.itemsize, 0)
        zs = self._zeros(10, str)
        assert_equal(zs.itemsize, 0)

    def _test_sort_partition(self, name, kinds, **kwargs):
        # 测试用例：测试排序和分区方法是否正常运行
        # 对于每种数据类型 dt，使用 _zeros 方法创建数组并进行排序和分区操作
        # 使用 getattr 获取对应的排序方法和 np 中的排序函数
        for dt in [bytes, np.void, str]:
            zs = self._zeros(10, dt)
            sort_method = getattr(zs, name)
            sort_func = getattr(np, name)
            for kind in kinds:
                sort_method(kind=kind, **kwargs)
                sort_func(zs, kind=kind, **kwargs)

    def test_sort(self):
        # 测试用例：测试 sort 方法的排序行为
        self._test_sort_partition('sort', kinds='qhs')

    def test_argsort(self):
        # 测试用例：测试 argsort 方法的排序行为
        self._test_sort_partition('argsort', kinds='qhs')

    def test_partition(self):
        # 测试用例：测试 partition 方法的分区行为
        self._test_sort_partition('partition', kinds=['introselect'], kth=2)

    def test_argpartition(self):
        # 测试用例：测试 argpartition 方法的分区行为
        self._test_sort_partition('argpartition', kinds=['introselect'], kth=2)

    def test_resize(self):
        # 测试用例：测试 resize 方法的调整数组大小行为
        # 对于每种数据类型 dt，使用 _zeros 方法创建数组并进行 resize 操作
        for dt in [bytes, np.void, str]:
            zs = self._zeros(10, dt)
            zs.resize(25)  # 调整数组大小为 25
            zs.resize((10, 10))  # 调整数组大小为 (10, 10)

    def test_view(self):
        # 测试用例：测试 view 方法的数组视图行为
        for dt in [bytes, np.void, str]:
            zs = self._zeros(10, dt)

            # 视图为相同类型应允许
            assert_equal(zs.view(dt).dtype, np.dtype(dt))

            # 视图为任何非空类型应返回空结果
            assert_equal(zs.view((dt, 1)).shape, (0,))

    def test_dumps(self):
        # 测试用例：测试 dumps 方法的序列化和反序列化行为
        zs = self._zeros(10, int)
        assert_equal(zs, pickle.loads(zs.dumps()))

    def test_pickle(self):
        # 测试用例：测试 pickle 序列化和反序列化行为
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            for dt in [bytes, np.void, str]:
                zs = self._zeros(10, dt)
                p = pickle.dumps(zs, protocol=proto)
                zs2 = pickle.loads(p)

                assert_equal(zs.dtype, zs2.dtype)

    def test_pickle_empty(self):
        """Checking if an empty array pickled and un-pickled will not cause a
        segmentation fault"""
        # 测试用例：测试空数组的 pickle 序列化和反序列化行为
        arr = np.array([]).reshape(999999, 0)
        pk_dmp = pickle.dumps(arr)
        pk_load = pickle.loads(pk_dmp)

        assert pk_load.size == 0

    @pytest.mark.skipif(pickle.HIGHEST_PROTOCOL < 5,
                        reason="requires pickle protocol 5")
    # 定义一个测试方法，用于测试带有 buffer_callback 的 pickle 功能
    def test_pickle_with_buffercallback(self):
        # 创建一个包含 0 到 9 的 NumPy 数组
        array = np.arange(10)
        # 初始化一个空列表，用于存储 pickle 操作期间的缓冲区
        buffers = []
        # 使用 pickle.dumps 序列化数组，指定 buffer_callback 将缓冲区添加到 buffers 列表中
        bytes_string = pickle.dumps(array, buffer_callback=buffers.append,
                                    protocol=5)
        # 使用 pickle.loads 反序列化 bytes_string，同时传入之前收集的 buffers
        array_from_buffer = pickle.loads(bytes_string, buffers=buffers)
        # 当使用 pickle 协议 5 和 buffer_callback 时，
        # array_from_buffer 将从包含对初始数组数据视图的缓冲区中重构，
        # 因此修改数组中的元素应当同时反映在 array_from_buffer 中。
        array[0] = -1
        # 断言修改后的 array_from_buffer 第一个元素确实为 -1
        assert array_from_buffer[0] == -1, array_from_buffer[0]
class TestMethods:
    
    # 类变量，包含排序方法的列表
    sort_kinds = ['quicksort', 'heapsort', 'stable']

    # 测试 all 方法中的 where 参数
    def test_all_where(self):
        # 创建一个布尔型的 NumPy 数组
        a = np.array([[True, False, True],
                      [False, False, False],
                      [True, True, True]])
        # 创建两个用于 where 参数的布尔型 NumPy 数组
        wh_full = np.array([[True, False, True],
                            [False, False, False],
                            [True, False, True]])
        wh_lower = np.array([[False],
                             [False],
                             [True]])
        
        # 循环遍历 axis 参数的不同取值
        for _ax in [0, None]:
            # 使用 assert_equal 检查 all 方法的结果
            assert_equal(a.all(axis=_ax, where=wh_lower),
                         np.all(a[wh_lower[:,0],:], axis=_ax))
            assert_equal(np.all(a, axis=_ax, where=wh_lower),
                         a[wh_lower[:,0],:].all(axis=_ax))
        
        # 检查全局的 where 参数设置
        assert_equal(a.all(where=wh_full), True)
        assert_equal(np.all(a, where=wh_full), True)
        assert_equal(a.all(where=False), True)
        assert_equal(np.all(a, where=False), True)

    # 测试 any 方法中的 where 参数
    def test_any_where(self):
        # 创建一个布尔型的 NumPy 数组
        a = np.array([[True, False, True],
                      [False, False, False],
                      [True, True, True]])
        # 创建两个用于 where 参数的布尔型 NumPy 数组
        wh_full = np.array([[False, True, False],
                            [True, True, True],
                            [False, False, False]])
        wh_middle = np.array([[False],
                              [True],
                              [False]])
        
        # 循环遍历 axis 参数的不同取值
        for _ax in [0, None]:
            # 使用 assert_equal 检查 any 方法的结果
            assert_equal(a.any(axis=_ax, where=wh_middle),
                         np.any(a[wh_middle[:,0],:], axis=_ax))
            assert_equal(np.any(a, axis=_ax, where=wh_middle),
                         a[wh_middle[:,0],:].any(axis=_ax))
        
        # 检查全局的 where 参数设置
        assert_equal(a.any(where=wh_full), False)
        assert_equal(np.any(a, where=wh_full), False)
        assert_equal(a.any(where=False), False)
        assert_equal(np.any(a, where=False), False)

    # 使用 pytest.mark.parametrize 注解测试 any 和 all 方法的结果数据类型
    @pytest.mark.parametrize("dtype", ["i8", "U10", "object", "datetime64[ms]"])
    def test_any_and_all_result_dtype(self, dtype):
        # 创建指定数据类型的全为 1 的 NumPy 数组
        arr = np.ones(3, dtype=dtype)
        # 使用 assert 检查 any 和 all 方法的结果数据类型
        assert arr.any().dtype == np.bool
        assert arr.all().dtype == np.bool

    # 测试 any 和 all 方法在 object 类型数组上的行为
    def test_any_and_all_object_dtype(self):
        # 创建一个 object 类型的全为 1 的 NumPy 数组
        arr = np.ones(3, dtype=object)
        # 使用 assert 检查 any 和 all 方法在 object 类型数组上的行为
        assert arr.any(dtype=object, keepdims=True).dtype == object
        assert arr.all(dtype=object, keepdims=True).dtype == object
    # 定义一个测试函数 test_compress 用于测试 compress 方法
    def test_compress(self):
        tgt = [[5, 6, 7, 8, 9]]  # 目标压缩结果
        arr = np.arange(10).reshape(2, 5)  # 创建一个 2*5 的数组
        out = arr.compress([0, 1], axis=0)  # 沿着指定轴对数组进行压缩
        assert_equal(out, tgt)  # 断言压缩结果是否符合预期

        tgt = [[1, 3], [6, 8]]  # 目标压缩结果
        out = arr.compress([0, 1, 0, 1, 0], axis=1)  # 沿着指定轴对数组进行压缩
        assert_equal(out, tgt)  # 断言压缩结果是否符合预期

        tgt = [[1], [6]]  # 目标压缩结果
        arr = np.arange(10).reshape(2, 5)  # 创建一个 2*5 的数组
        out = arr.compress([0, 1], axis=1)  # 沿着指定轴对数组进行压缩
        assert_equal(out, tgt)  # 断言压缩结果是否符合预期

        arr = np.arange(10).reshape(2, 5)  # 创建一个 2*5 的数组
        out = arr.compress([0, 1])  # 对数组进行压缩
        assert_equal(out, 1)  # 断言压缩结果是否为 1

    # 定义一个测试函数 test_choose 用于测试 choose 方法
    def test_choose(self):
        x = 2*np.ones((3,), dtype=int)  # 创建一个由 2 组成的长度为 3 的数组
        y = 3*np.ones((3,), dtype=int)  # 创建一个由 3 组成的长度为 3 的数组
        x2 = 2*np.ones((2, 3), dtype=int)  # 创建一个 2*3 的数组
        y2 = 3*np.ones((2, 3), dtype=int)  # 创建一个 2*3 的数组
        ind = np.array([0, 0, 1])  # 创建一个索引数组

        A = ind.choose((x, y))  # 使用索引数组从元组中选择
        assert_equal(A, [2, 2, 3])  # 断言选择结果是否符合预期

        A = ind.choose((x2, y2))  # 使用索引数组从元组中选择
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])  # 断言选择结果是否符合预期

        A = ind.choose((x, y2))  # 使用索引数组从元组中选择
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])  # 断言选择结果是否符合预期

        oned = np.ones(1)  # 创建一个长度为 1 的数组
        # gh-12031, caused SEGFAULT
        assert_raises(TypeError, oned.choose, np.void(0), [oned])  # 检查异常类型是否符合预期

        out = np.array(0)  # 创建一个值为 0 的数组
        ret = np.choose(np.array(1), [10, 20, 30], out=out)  # 使用索引数组从可选项列表中选择
        assert out is ret  # 断言返回结果是否与输出结果相同
        assert_equal(out[()], 20)  # 断言输出结果是否符合预期

        # gh-6272 check overlap on out
        x = np.arange(5)  # 创建一个包含 0 到 4 的数组
        y = np.choose([0, 0, 0], [x[:3], x[:3], x[:3]], out=x[1:4], mode='wrap')  # 使用索引数组从可选项列表中选择，并指定输出数组
        assert_equal(y, np.array([0, 1, 2]))  # 断言选择结果是否符合预期

    # 定义一个测试函数 test_prod 用于测试 prod 方法
    def test_prod(self):
        ba = [1, 2, 10, 11, 6, 5, 4]  # 数组元素列表
        ba2 = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]  # 二维数组

        # 遍历不同的数据类型
        for ctype in [np.int16, np.uint16, np.int32, np.uint32,
                      np.float32, np.float64, np.complex64, np.complex128]:
            a = np.array(ba, ctype)  # 创建数组 a
            a2 = np.array(ba2, ctype)  # 创建数组 a2
            # 如果数据类型为 '1' 或 'b'，则检查是否抛出算术错误
            if ctype in ['1', 'b']:
                assert_raises(ArithmeticError, a.prod)
                assert_raises(ArithmeticError, a2.prod, axis=1)
            else:
                assert_equal(a.prod(axis=0), 26400)  # 沿着指定轴计算乘积并断言结果是否符合预期
                assert_array_equal(a2.prod(axis=0),
                                   np.array([50, 36, 84, 180], ctype))  # 沿着指定轴计算乘积并断言结果是否符合预期
                assert_array_equal(a2.prod(axis=-1),
                                   np.array([24, 1890, 600], ctype))  # 沿着指定轴计算乘积并断言结果是否符合预期

    @pytest.mark.parametrize('dtype', [None, object])  # 使用 pytest 的参数化装饰器
    # 定义测试方法，接受参数self和dtype
    def test_repeat(self, dtype):
        # 创建一个包含整数的NumPy数组m，使用指定的数据类型dtype
        m = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
        # 将m重新形状为2行3列的矩形数组m_rect
        m_rect = m.reshape((2, 3))

        # 使用指定的重复次数对数组m的元素进行重复，并将结果赋给A
        A = m.repeat([1, 3, 2, 1, 1, 2])
        # 断言A与预期的结果是否相等
        assert_equal(A, [1, 2, 2, 2, 3,
                         3, 4, 5, 6, 6])

        # 对数组m的每个元素重复两次，并将结果赋给A
        A = m.repeat(2)
        # 断言A与预期的结果是否相等
        assert_equal(A, [1, 1, 2, 2, 3, 3,
                         4, 4, 5, 5, 6, 6])

        # 按照指定的重复次数沿着指定的轴（axis=0，即行）对矩形数组m_rect进行重复，并将结果赋给A
        A = m_rect.repeat([2, 1], axis=0)
        # 断言A与预期的结果是否相等
        assert_equal(A, [[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6]])

        # 按照指定的重复次数沿着指定的轴（axis=1，即列）对矩形数组m_rect进行重复，并将结果赋给A
        A = m_rect.repeat([1, 3, 2], axis=1)
        # 断言A与预期的结果是否相等
        assert_equal(A, [[1, 2, 2, 2, 3, 3],
                         [4, 5, 5, 5, 6, 6]])

        # 按照指定的重复次数沿着指定的轴（axis=0，即行）对矩形数组m_rect进行重复，并将结果赋给A
        A = m_rect.repeat(2, axis=0)
        # 断言A与预期的结果是否相等
        assert_equal(A, [[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6],
                         [4, 5, 6]])

        # 按照指定的重复次数沿着指定的轴（axis=1，即列）对矩形数组m_rect进行重复，并将结果赋给A
        A = m_rect.repeat(2, axis=1)
        # 断言A与预期的结果是否相等
        assert_equal(A, [[1, 1, 2, 2, 3, 3],
                         [4, 4, 5, 5, 6, 6]])

    # 定义测试方法test_reshape
    def test_reshape(self):
        # 创建一个包含多维数组的NumPy数组arr
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        # 将数组arr重新形状为2行6列的数组tgt，将结果与预期结果tgt进行断言比较
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(arr.reshape(2, 6), tgt)

        # 将数组arr重新形状为3行4列的数组tgt，将结果与预期结果tgt进行断言比较
        tgt = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        assert_equal(arr.reshape(3, 4), tgt)

        # 按照指定的顺序（order='F'，即Fortran顺序）将数组arr重新形状为3行4列的数组tgt，将结果与预期结果tgt进行断言比较
        tgt = [[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]]
        assert_equal(arr.reshape((3, 4), order='F'), tgt)

        # 按照指定的顺序（order='C'，即C顺序）将数组arr进行转置后，再将其重新形状为3行4列的数组tgt，将结果与预期结果tgt进行断言比较
        tgt = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
        assert_equal(arr.T.reshape((3, 4), order='C'), tgt)

    # 定义测试方法test_round
    def test_round(self):
        # 定义内部函数check_round，用于验证四舍五入操作的结果
        def check_round(arr, expected, *round_args):
            # 断言对数组arr进行指定参数的四舍五入后，结果与预期的expected相等
            assert_equal(arr.round(*round_args), expected)
            # 创建一个与arr同样大小的零数组out，将结果数组赋给res
            out = np.zeros_like(arr)
            res = arr.round(*round_args, out=out)
            # 断言out数组与预期的expected相等
            assert_equal(out, expected)
            # 断言out与res是同一个对象
            assert out is res

        # 调用check_round验证四舍五入操作的结果
        check_round(np.array([1.2, 1.5]), [1, 2])
        check_round(np.array(1.5), 2)
        check_round(np.array([12.2, 15.5]), [10, 20], -1)
        check_round(np.array([12.15, 15.51]), [12.2, 15.5], 1)
        # 复杂数的四舍五入验证
        check_round(np.array([4.5 + 1.5j]), [4 + 2j])
        check_round(np.array([12.5 + 15.5j]), [10 + 20j], -1)

    # 定义测试方法test_squeeze
    def test_squeeze(self):
        # 创建一个包含三维数组的NumPy数组a
        a = np.array([[[1], [2], [3]]])
        # 对数组a执行squeeze操作，将结果与预期的[1, 2, 3]进行断言比较
        assert_equal(a.squeeze(), [1, 2, 3])
        # 对数组a沿着指定的轴（axis=(0,)，即第0轴）执行squeeze操作，将结果与预期的[[1], [2], [3]]进行断言比较
        assert_equal(a.squeeze(axis=(0,)), [[1], [2], [3]])
        # 断言在沿着指定的轴（axis=(1,)，即第1轴）执行squeeze操作时会抛出ValueError异常
        assert_raises(ValueError, a.squeeze, axis=(1,))
        # 对数组a沿着指定的轴（axis=(2,)，即第2轴）执行squeeze操作，将结果与预期的[[1, 2, 3]]进行断言比较
        assert_equal(a.squeeze(axis=(2,)), [[1, 2, 3]])

    # 定义测试方法test_transpose
    def test_transpose(self):
        # 创建一个包含二维数组的NumPy数组a
        a = np.array([[1, 2], [3, 4]])
        # 对数组a执行转置操作，将结果与预期的[[1, 3], [2, 4]]进行断言比较
        assert_equal(a.transpose(), [[1, 3], [2, 4]])
        # 断言在传递一个参数（0）给transpose时会抛出ValueError异常
        assert_raises(ValueError, lambda: a.transpose(0))
        # 断言在传递两个参数（0,
    def test_sort(self):
        # 测试浮点数和复数排序，包含 NaN 值。只需要检查小于比较即可，
        # 因此只需要进行插入排序路径的排序即可。我们只测试双精度和复数双精度因为逻辑相同。

        # 检查双精度
        msg = "Test real sort order with nans"
        a = np.array([np.nan, 1, 0])
        b = np.sort(a)
        assert_equal(b, a[::-1], msg)
        # 检查复数
        msg = "Test complex sort order with nans"
        a = np.zeros(9, dtype=np.complex128)
        a.real += [np.nan, np.nan, np.nan, 1, 0, 1, 1, 0, 0]
        a.imag += [np.nan, 1, 0, np.nan, np.nan, 1, 0, 1, 0]
        b = np.sort(a)
        assert_equal(b, a[::-1], msg)

        # 检查是否会抛出异常
        with assert_raises_regex(
            ValueError,
            "kind` and `stable` parameters can't be provided at the same time"
        ):
            np.sort(a, kind="stable", stable=True)

    # 所有的 C 标量排序使用相同的代码，只是类型不同
    # 因此只需要用一个类型快速检查即可。排序的项数必须大于 ~50 才能检查实际的
    # 算法，因为对于小数组，快速排序和归并排序会转换为插入排序。

    @pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.uint64,
                                       np.float16, np.float32, np.float64,
                                       np.longdouble])
    def test_sort_unsigned(self, dtype):
        a = np.arange(101, dtype=dtype)
        b = a[::-1].copy()
        for kind in self.sort_kinds:
            msg = "scalar sort, kind=%s" % kind
            c = a.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)

    @pytest.mark.parametrize('dtype',
                             [np.int8, np.int16, np.int32, np.int64, np.float16,
                              np.float32, np.float64, np.longdouble])
    def test_sort_signed(self, dtype):
        a = np.arange(-50, 51, dtype=dtype)
        b = a[::-1].copy()
        for kind in self.sort_kinds:
            msg = "scalar sort, kind=%s" % (kind)
            c = a.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy()
            c.sort(kind=kind)
            assert_equal(c, a, msg)

    @pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longdouble])
    @pytest.mark.parametrize('part', ['real', 'imag'])
    def test_sort_complex(self, part, dtype):
        # 测试复杂排序。这些使用与标量相同的代码，但比较函数不同。
        cdtype = {
            np.single: np.csingle,
            np.double: np.cdouble,
            np.longdouble: np.clongdouble,
        }[dtype]
        # 创建从 -50 到 50 的数组，使用给定的数据类型
        a = np.arange(-50, 51, dtype=dtype)
        # 创建逆序的数组副本
        b = a[::-1].copy()
        # 创建复数数组，将每个元素乘以 (1+1j)，并转换为指定的复数类型
        ai = (a * (1+1j)).astype(cdtype)
        bi = (b * (1+1j)).astype(cdtype)
        # 设置对象的属性为 1
        setattr(ai, part, 1)
        setattr(bi, part, 1)
        # 遍历排序种类
        for kind in self.sort_kinds:
            # 创建消息字符串，用于测试说明
            msg = "complex sort, %s part == 1, kind=%s" % (part, kind)
            # 对 ai 进行排序，使用指定的排序种类
            c = ai.copy()
            c.sort(kind=kind)
            # 断言排序后的数组与原始数组 ai 相等，否则输出消息
            assert_equal(c, ai, msg)
            # 对 bi 进行排序，使用指定的排序种类
            c = bi.copy()
            c.sort(kind=kind)
            # 断言排序后的数组与原始数组 ai 相等，否则输出消息
            assert_equal(c, ai, msg)

    def test_sort_complex_byte_swapping(self):
        # 测试需要字节交换的复杂数组排序，gh-5441
        # 遍历大小端序
        for endianness in '<>':
            # 遍历复杂类型的数据类型
            for dt in np.typecodes['Complex']:
                # 创建复数数组，使用指定的字节序和数据类型
                arr = np.array([1+3.j, 2+2.j, 3+1.j], dtype=endianness + dt)
                # 创建数组的副本
                c = arr.copy()
                # 对数组进行排序
                c.sort()
                # 创建消息字符串，用于测试说明
                msg = 'byte-swapped complex sort, dtype={0}'.format(dt)
                # 断言排序后的数组与原始数组 arr 相等，否则输出消息
                assert_equal(c, arr, msg)

    @pytest.mark.parametrize('dtype', [np.bytes_, np.str_])
    def test_sort_string(self, dtype):
        # 测试字符串数组排序
        # 创建字符串数组，使用给定的数据类型
        a = np.array(['aaaaaaaa' + chr(i) for i in range(101)], dtype=dtype)
        # 创建逆序的数组副本
        b = a[::-1].copy()
        # 遍历排序种类
        for kind in self.sort_kinds:
            # 创建消息字符串，用于测试说明
            msg = "kind=%s" % kind
            # 对 a 进行排序，使用指定的排序种类
            c = a.copy()
            c.sort(kind=kind)
            # 断言排序后的数组与原始数组 a 相等，否则输出消息
            assert_equal(c, a, msg)
            # 对 b 进行排序，使用指定的排序种类
            c = b.copy()
            c.sort(kind=kind)
            # 断言排序后的数组与原始数组 a 相等，否则输出消息
            assert_equal(c, a, msg)

    def test_sort_object(self):
        # 测试对象数组排序
        # 创建空的对象数组
        a = np.empty((101,), dtype=object)
        # 将数组元素设置为连续的整数值
        a[:] = list(range(101))
        # 创建逆序的数组副本
        b = a[::-1]
        # 遍历排序种类
        for kind in ['q', 'h', 'm']:
            # 创建消息字符串，用于测试说明
            msg = "kind=%s" % kind
            # 对 a 进行排序，使用指定的排序种类
            c = a.copy()
            c.sort(kind=kind)
            # 断言排序后的数组与原始数组 a 相等，否则输出消息
            assert_equal(c, a, msg)
            # 对 b 进行排序，使用指定的排序种类
            c = b.copy()
            c.sort(kind=kind)
            # 断言排序后的数组与原始数组 a 相等，否则输出消息
            assert_equal(c, a, msg)

    @pytest.mark.parametrize("dt", [
            np.dtype([('f', float), ('i', int)]),
            np.dtype([('f', float), ('i', object)])])
    @pytest.mark.parametrize("step", [1, 2])
    def test_sort_structured(self, dt, step):
        # test record array sorts.
        # 创建一个结构化数组 a，包含元组 (i, i)，其中 i 的范围是 0 到 101*step-1，数据类型为 dt
        a = np.array([(i, i) for i in range(101*step)], dtype=dt)
        # 创建数组 b，其内容是数组 a 的逆序
        b = a[::-1]
        # 遍历排序类型 ['q', 'h', 'm']
        for kind in ['q', 'h', 'm']:
            msg = "kind=%s" % kind
            # 复制数组 a 的每步步长为 step 的子数组到 c
            c = a.copy()[::step]
            # 根据指定的排序类型 kind 对 c 进行排序，并返回排序后的索引
            indx = c.argsort(kind=kind)
            c.sort(kind=kind)
            # 断言排序后的 c 是否与原始数组 a 每步步长为 step 的子数组相等，使用消息 msg
            assert_equal(c, a[::step], msg)
            # 断言按照 indx 排序后的 c 是否与原始数组 a 每步步长为 step 的子数组相等，使用消息 msg
            assert_equal(a[::step][indx], a[::step], msg)
            # 复制数组 b 的每步步长为 step 的子数组到 c
            c = b.copy()[::step]
            # 根据指定的排序类型 kind 对 c 进行排序，并返回排序后的索引
            indx = c.argsort(kind=kind)
            c.sort(kind=kind)
            # 断言排序后的 c 是否与数组 a 以步长 step-1 逆序的子数组相等，使用消息 msg
            assert_equal(c, a[step-1::step], msg)
            # 断言按照 indx 排序后的数组 b 的每步步长为 step 的子数组是否与数组 a 以步长 step-1 逆序的子数组相等，使用消息 msg
            assert_equal(b[::step][indx], a[step-1::step], msg)

    @pytest.mark.parametrize('dtype', ['datetime64[D]', 'timedelta64[D]'])
    def test_sort_time(self, dtype):
        # test datetime64 and timedelta64 sorts.
        # 创建一个从 0 到 101 的数组 a，数据类型为 dtype（可以是 'datetime64[D]' 或 'timedelta64[D]'）
        a = np.arange(0, 101, dtype=dtype)
        # 创建数组 b，其内容是数组 a 的逆序
        b = a[::-1]
        # 遍历排序类型 ['q', 'h', 'm']
        for kind in ['q', 'h', 'm']:
            msg = "kind=%s" % kind
            # 复制数组 a 到 c
            c = a.copy()
            # 根据指定的排序类型 kind 对 c 进行排序
            c.sort(kind=kind)
            # 断言排序后的 c 是否与原始数组 a 相等，使用消息 msg
            assert_equal(c, a, msg)
            # 复制数组 b 到 c
            c = b.copy()
            # 根据指定的排序类型 kind 对 c 进行排序
            c.sort(kind=kind)
            # 断言排序后的 c 是否与原始数组 a 相等，使用消息 msg
            assert_equal(c, a, msg)

    def test_sort_axis(self):
        # check axis handling. This should be the same for all type
        # specific sorts, so we only check it for one type and one kind
        # 创建一个二维数组 a
        a = np.array([[3, 2], [1, 0]])
        # 创建与 a 对应的排序后的数组 b
        b = np.array([[1, 0], [3, 2]])
        # 创建与 a 对应的在轴 1 排序后的数组 c
        c = np.array([[2, 3], [0, 1]])
        # 复制数组 a 到 d
        d = a.copy()
        # 按照轴 0 进行排序
        d.sort(axis=0)
        # 断言排序后的 d 是否与数组 b 相等，使用消息 "test sort with axis=0"
        assert_equal(d, b, "test sort with axis=0")
        # 复制数组 a 到 d
        d = a.copy()
        # 按照轴 1 进行排序
        d.sort(axis=1)
        # 断言排序后的 d 是否与数组 c 相等，使用消息 "test sort with axis=1"
        assert_equal(d, c, "test sort with axis=1")
        # 复制数组 a 到 d
        d = a.copy()
        # 默认情况下按照最后一个轴进行排序
        d.sort()
        # 断言排序后的 d 是否与数组 c 相等，使用消息 "test sort with default axis"
        assert_equal(d, c, "test sort with default axis")

    def test_sort_size_0(self):
        # check axis handling for multidimensional empty arrays
        # 创建一个空的多维数组 a，并设置其形状为 (3, 2, 1, 0)
        a = np.array([])
        a.shape = (3, 2, 1, 0)
        # 遍历数组的所有轴
        for axis in range(-a.ndim, a.ndim):
            msg = 'test empty array sort with axis={0}'.format(axis)
            # 断言按照指定的轴 axis 对数组 a 进行排序后是否与原数组 a 相等，使用消息 msg
            assert_equal(np.sort(a, axis=axis), a, msg)
        # 断言对空数组 a 进行默认轴排序后是否与其展平的结果相等，使用消息 "test empty array sort with axis=None"
        msg = 'test empty array sort with axis=None'
        assert_equal(np.sort(a, axis=None), a.ravel(), msg)

    def test_sort_bad_ordering(self):
        # test generic class with bogus ordering,
        # should not segfault.
        # 定义一个简单的类 Boom，用于测试不正确的排序
        class Boom:
            def __lt__(self, other):
                return True

        # 创建一个包含 100 个 Boom 类型对象的数组 a
        a = np.array([Boom()] * 100, dtype=object)
        # 遍历排序类型 self.sort_kinds
        for kind in self.sort_kinds:
            msg = "kind=%s" % kind
            # 复制数组 a 到 c
            c = a.copy()
            # 根据指定的排序类型 kind 对 c 进行排序
            c.sort(kind=kind)
            # 断言排序后的 c 是否与原始数组 a 相等，使用消息 msg
            assert_equal(c, a, msg)
    # 定义测试方法，用于测试排序功能在特定情况下的表现
    def test_void_sort(self):
        # 修复问题 gh-8210：在先前的情况下出现段错误
        for i in range(4):
            # 生成随机的 uint8 类型的数组
            rand = np.random.randint(256, size=4000, dtype=np.uint8)
            # 将数组视图转换为复合数据类型 'V4'
            arr = rand.view('V4')
            # 对数组进行逆序排序
            arr[::-1].sort()

        # 定义数据类型 dt，包含字段 'val'，每个元素为 i4 类型的一维数组
        dt = np.dtype([('val', 'i4', (1,))])
        for i in range(4):
            # 生成随机的 uint8 类型的数组
            rand = np.random.randint(256, size=4000, dtype=np.uint8)
            # 将数组视图转换为指定的复合数据类型 dt
            arr = rand.view(dt)
            # 对数组进行逆序排序
            arr[::-1].sort()

    # 定义测试方法，用于测试排序功能在抛出异常时的行为
    def test_sort_raises(self):
        # 问题 gh-9404
        arr = np.array([0, datetime.now(), 1], dtype=object)
        for kind in self.sort_kinds:
            # 断言排序时会抛出 TypeError 异常
            assert_raises(TypeError, arr.sort, kind=kind)
        
        # 问题 gh-3879
        # 定义一个类 Raiser，其方法和运算符都会抛出 TypeError 异常
        class Raiser:
            def raises_anything(*args, **kwargs):
                raise TypeError("SOMETHING ERRORED")
            __eq__ = __ne__ = __lt__ = __gt__ = __ge__ = __le__ = raises_anything
        # 生成一个包含 Raiser 对象和数字的数组，并展开成一维数组
        arr = np.array([[Raiser(), n] for n in range(10)]).reshape(-1)
        np.random.shuffle(arr)
        for kind in self.sort_kinds:
            # 断言排序时会抛出 TypeError 异常
            assert_raises(TypeError, arr.sort, kind=kind)

    # 定义测试方法，用于测试排序功能在处理大数据集时的性能
    def test_sort_degraded(self):
        # 测试使用普通快速排序可能耗费数分钟的数据集
        d = np.arange(1000000)
        do = d.copy()
        x = d
        # 创建一个中位数选取策略，其中每个中位数是快速排序分区的第二个排序的元素
        while x.size > 3:
            mid = x.size // 2
            x[mid], x[-2] = x[-2], x[mid]
            x = x[:-2]

        # 断言排序后的结果与原始数据一致
        assert_equal(np.sort(d), do)
        assert_equal(d[np.argsort(d)], do)

    # 定义测试方法，用于测试数组复制的各种情况
    def test_copy(self):
        # 定义辅助函数，用于断言数组是 Fortran 风格的
        def assert_fortran(arr):
            assert_(arr.flags.fortran)
            assert_(arr.flags.f_contiguous)
            assert_(not arr.flags.c_contiguous)

        # 定义辅助函数，用于断言数组是 C 风格的
        def assert_c(arr):
            assert_(not arr.flags.fortran)
            assert_(not arr.flags.f_contiguous)
            assert_(arr.flags.c_contiguous)

        # 创建一个 Fortran 风格的空数组
        a = np.empty((2, 2), order='F')
        # 测试复制 Fortran 风格数组时的行为
        assert_c(a.copy())
        assert_c(a.copy('C'))
        assert_fortran(a.copy('F'))
        assert_fortran(a.copy('A'))

        # 创建一个 C 风格的空数组
        a = np.empty((2, 2), order='C')
        # 测试复制 C 风格数组时的行为
        assert_c(a.copy())
        assert_c(a.copy('C'))
        assert_fortran(a.copy('F'))
        assert_c(a.copy('A'))

    # 使用 pytest 参数化装饰器定义测试方法，测试对象的深拷贝行为
    @pytest.mark.parametrize("dtype", ['O', np.int32, 'i,O'])
    def test__deepcopy__(self, dtype):
        # 创建一个指定类型 dtype 的空数组
        a = np.empty(4, dtype=dtype)
        # 强制将 NULL 值写入数组中
        ctypes.memset(a.ctypes.data, 0, a.nbytes)

        # 确保调用 __deepcopy__ 方法时不会引发错误，见问题 gh-21833
        b = a.__deepcopy__({})

        # 修改数组的第一个元素
        a[0] = 42
        # 断言深拷贝后的数组 b 与原始数组 a 不相等
        with pytest.raises(AssertionError):
            assert_array_equal(a, b)
    # 定义一个测试函数，用于测试__deepcopy__方法是否能够捕获异常
    def test__deepcopy__catches_failure(self):
        # 定义一个自定义类MyObj，覆盖其__deepcopy__方法以引发RuntimeError异常
        class MyObj:
            def __deepcopy__(self, *args, **kwargs):
                raise RuntimeError

        # 创建一个包含不同类型元素的NumPy数组
        arr = np.array([1, MyObj(), 3], dtype='O')
        # 使用pytest断言来验证是否捕获到RuntimeError异常
        with pytest.raises(RuntimeError):
            arr.__deepcopy__({})

    # 定义一个测试函数，用于测试排序数组的顺序
    def test_sort_order(self):
        # 创建三个不同的NumPy数组
        x1 = np.array([21, 32, 14])
        x2 = np.array(['my', 'first', 'name'])
        x3 = np.array([3.1, 4.5, 6.2])
        # 使用np.rec.fromarrays方法创建一个结构化数组r，包含字段'id'、'word'、'number'
        r = np.rec.fromarrays([x1, x2, x3], names='id,word,number')

        # 按'id'字段排序，并验证排序结果是否符合预期
        r.sort(order=['id'])
        assert_equal(r.id, np.array([14, 21, 32]))
        assert_equal(r.word, np.array(['name', 'my', 'first']))
        assert_equal(r.number, np.array([6.2, 3.1, 4.5]))

        # 按'word'字段排序，并验证排序结果是否符合预期
        r.sort(order=['word'])
        assert_equal(r.id, np.array([32, 21, 14]))
        assert_equal(r.word, np.array(['first', 'my', 'name']))
        assert_equal(r.number, np.array([4.5, 3.1, 6.2]))

        # 按'number'字段排序，并验证排序结果是否符合预期
        r.sort(order=['number'])
        assert_equal(r.id, np.array([21, 32, 14]))
        assert_equal(r.word, np.array(['my', 'first', 'name']))
        assert_equal(r.number, np.array([3.1, 4.5, 6.2]))

        # 使用lambda表达式和assert_raises_regex验证当排序字段有重复时是否抛出ValueError异常
        assert_raises_regex(ValueError, 'duplicate',
            lambda: r.sort(order=['id', 'id']))

        # 根据系统字节顺序设置数据类型，并创建新的NumPy数组r，验证按'col2'字段排序后结果是否符合预期
        if sys.byteorder == 'little':
            strtype = '>i2'
        else:
            strtype = '<i2'
        mydtype = [('name', 'U5'), ('col2', strtype)]
        r = np.array([('a', 1), ('b', 255), ('c', 3), ('d', 258)],
                     dtype=mydtype)
        r.sort(order='col2')
        assert_equal(r['col2'], [1, 3, 255, 258])
        assert_equal(r, np.array([('a', 1), ('c', 3), ('b', 255), ('d', 258)],
                                 dtype=mydtype))

    # 定义一个测试函数，用于测试排序Unicode字符串的行为
    def test_sort_unicode_kind(self):
        # 创建一个包含10个元素的NumPy数组
        d = np.arange(10)
        # 使用UTF-8解码生成一个Unicode字符串k
        k = b'\xc3\xa4'.decode("UTF8")
        # 使用assert_raises验证当尝试使用指定kind进行排序时是否抛出ValueError异常
        assert_raises(ValueError, d.sort, kind=k)
        assert_raises(ValueError, d.argsort, kind=k)

    # 使用pytest.mark.parametrize装饰器定义参数化测试，验证不同类型的浮点数数组的searchsorted方法行为
    @pytest.mark.parametrize('a', [
        np.array([0, 1, np.nan], dtype=np.float16),
        np.array([0, 1, np.nan], dtype=np.float32),
        np.array([0, 1, np.nan]),
    ])
    def test_searchsorted_floats(self, a):
        # 测试包含NaN值的浮点数数组的searchsorted方法，验证其在半精度、单精度和双精度浮点数上的行为
        msg = "Test real (%s) searchsorted with nans, side='l'" % a.dtype
        b = a.searchsorted(a, side='left')
        assert_equal(b, np.arange(3), msg)
        msg = "Test real (%s) searchsorted with nans, side='r'" % a.dtype
        b = a.searchsorted(a, side='right')
        assert_equal(b, np.arange(1, 4), msg)
        # 检查关键字参数是否正常工作
        a.searchsorted(v=1)
        x = np.array([0, 1, np.nan], dtype='float32')
        y = np.searchsorted(x, x[-1])
        assert_equal(y, 2)
    def test_searchsorted_complex(self):
        # 测试复杂数组包含 NaN 的情况。
        # 搜索排序算法使用数组类型的比较函数，因此这里检查排序顺序是否一致。
        # 检查双复数
        a = np.zeros(9, dtype=np.complex128)
        a.real += [0, 0, 1, 1, 0, 1, np.nan, np.nan, np.nan]
        a.imag += [0, 1, 0, 1, np.nan, np.nan, 0, 1, np.nan]
        msg = "Test complex searchsorted with nans, side='l'"
        b = a.searchsorted(a, side='left')
        assert_equal(b, np.arange(9), msg)
        msg = "Test complex searchsorted with nans, side='r'"
        b = a.searchsorted(a, side='right')
        assert_equal(b, np.arange(1, 10), msg)
        msg = "Test searchsorted with little endian, side='l'"
        a = np.array([0, 128], dtype='<i4')
        b = a.searchsorted(np.array(128, dtype='<i4'))
        assert_equal(b, 1, msg)
        msg = "Test searchsorted with big endian, side='l'"
        a = np.array([0, 128], dtype='>i4')
        b = a.searchsorted(np.array(128, dtype='>i4'))
        assert_equal(b, 1, msg)

    def test_searchsorted_n_elements(self):
        # 检查 0 元素的情况
        a = np.ones(0)
        b = a.searchsorted([0, 1, 2], 'left')
        assert_equal(b, [0, 0, 0])
        b = a.searchsorted([0, 1, 2], 'right')
        assert_equal(b, [0, 0, 0])
        a = np.ones(1)
        # 检查 1 元素的情况
        b = a.searchsorted([0, 1, 2], 'left')
        assert_equal(b, [0, 0, 1])
        b = a.searchsorted([0, 1, 2], 'right')
        assert_equal(b, [0, 1, 1])
        # 检查所有元素相等的情况
        a = np.ones(2)
        b = a.searchsorted([0, 1, 2], 'left')
        assert_equal(b, [0, 0, 2])
        b = a.searchsorted([0, 1, 2], 'right')
        assert_equal(b, [0, 2, 2])

    def test_searchsorted_unaligned_array(self):
        # 测试搜索非对齐数组的情况
        a = np.arange(10)
        aligned = np.empty(a.itemsize * a.size + 1, 'uint8')
        unaligned = aligned[1:].view(a.dtype)
        unaligned[:] = a
        # 测试搜索非对齐数组
        b = unaligned.searchsorted(a, 'left')
        assert_equal(b, a)
        b = unaligned.searchsorted(a, 'right')
        assert_equal(b, a + 1)
        # 测试搜索非对齐键值的情况
        b = a.searchsorted(unaligned, 'left')
        assert_equal(b, a)
        b = a.searchsorted(unaligned, 'right')
        assert_equal(b, a + 1)

    def test_searchsorted_resetting(self):
        # 测试二分搜索索引的智能重置
        a = np.arange(5)
        b = a.searchsorted([6, 5, 4], 'left')
        assert_equal(b, [5, 5, 4])
        b = a.searchsorted([6, 5, 4], 'right')
        assert_equal(b, [5, 5, 5])
    def test_searchsorted_type_specific(self):
        # 测试所有类型特定的二分查找函数

        # 拼接所有整数、浮点数、日期时间和布尔对象的类型码
        types = ''.join((np.typecodes['AllInteger'], np.typecodes['AllFloat'],
                         np.typecodes['Datetime'], '?O'))
        
        # 遍历每种数据类型
        for dt in types:
            # 如果数据类型是日期时间类型，修改为特定的日期时间格式
            if dt == 'M':
                dt = 'M8[D]'
            # 如果数据类型是布尔类型
            if dt == '?':
                # 创建一个长度为2的数组，数据类型为布尔类型
                a = np.arange(2, dtype=dt)
                # 创建一个长度为2的数组，数据类型为整数
                out = np.arange(2)
            else:
                # 创建一个从0到5的数组，步长为1，数据类型为当前类型码所表示的类型
                a = np.arange(0, 5, dtype=dt)
                # 创建一个长度为5的数组，数据类型为整数
                out = np.arange(5)
            
            # 在数组a中查找元素a，返回左边界索引
            b = a.searchsorted(a, 'left')
            # 断言左边界索引数组b与预期结果out相等
            assert_equal(b, out)
            
            # 在数组a中查找元素a，返回右边界索引
            b = a.searchsorted(a, 'right')
            # 断言右边界索引数组b与预期结果out加1相等
            assert_equal(b, out + 1)
            
            # 测试空数组情况，使用新的数组以便在valgrind中获得访问警告
            e = np.ndarray(shape=0, buffer=b'', dtype=dt)
            
            # 在空数组e中查找元素a，返回左边界索引
            b = e.searchsorted(a, 'left')
            # 断言左边界索引数组b全为0，长度与数组a相同，数据类型为intp
            assert_array_equal(b, np.zeros(len(a), dtype=np.intp))
            
            # 在数组a中查找元素e，返回左边界索引
            b = a.searchsorted(e, 'left')
            # 断言左边界索引数组b为空数组，数据类型为intp
            assert_array_equal(b, np.zeros(0, dtype=np.intp))

    def test_searchsorted_unicode(self):
        # 测试Unicode字符串的searchsorted函数

        # 1.6.1版本在arraytypes.c.src:UNICODE_compare()中存在字符串长度计算错误，
        # 导致searchsorted的结果不正确/不一致。
        
        # 创建一个Unicode字符串数组a
        a = np.array(['P:\\20x_dapi_cy3\\20x_dapi_cy3_20100185_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100186_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100187_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100189_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100190_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100191_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100192_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100193_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100194_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100195_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100196_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100197_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100198_1',
                      'P:\\20x_dapi_cy3\\20x_dapi_cy3_20100199_1'],
                     dtype=np.str_)
        
        # 创建一个与a相同长度的索引数组
        ind = np.arange(len(a))
        
        # 断言使用左边界模式查找每个元素在数组a中的索引与预期的索引数组ind相等
        assert_equal([a.searchsorted(v, 'left') for v in a], ind)
        
        # 断言使用右边界模式查找每个元素在数组a中的索引与预期的索引数组ind+1相等
        assert_equal([a.searchsorted(v, 'right') for v in a], ind + 1)
        
        # 断言使用左边界模式查找数组a中每个元素与其自身的索引与预期的索引数组ind相等
        assert_equal([a.searchsorted(a[i], 'left') for i in ind], ind)
        
        # 断言使用右边界模式查找数组a中每个元素与其自身的索引与预期的索引数组ind+1相等
        assert_equal([a.searchsorted(a[i], 'right') for i in ind], ind + 1)
    # 测试当排序器参数无效时的情况

    def test_searchsorted_with_invalid_sorter(self):
        # 创建一个 NumPy 数组
        a = np.array([5, 2, 1, 3, 4])
        # 对数组进行排序并获取排序后的索引
        s = np.argsort(a)
        # 测试排序器参数为不合法的类型时是否会引发 TypeError 异常
        assert_raises(TypeError, np.searchsorted, a, 0,
                      sorter=np.array((1, (2, 3)), dtype=object))
        assert_raises(TypeError, np.searchsorted, a, 0, sorter=[1.1])
        assert_raises(ValueError, np.searchsorted, a, 0, sorter=[1, 2, 3, 4])
        assert_raises(ValueError, np.searchsorted, a, 0, sorter=[1, 2, 3, 4, 5, 6])

        # 边界检查
        assert_raises(ValueError, np.searchsorted, a, 4, sorter=[0, 1, 2, 3, 5])
        assert_raises(ValueError, np.searchsorted, a, 0, sorter=[-1, 0, 1, 2, 3])
        assert_raises(ValueError, np.searchsorted, a, 0, sorter=[4, 0, -1, 2, 3])

    # 测试 searchsorted 方法返回类型的准确性

    def test_searchsorted_return_type(self):
        # 定义一个类 A，继承自 np.ndarray
        class A(np.ndarray):
            pass
        # 创建 A 类型的数组实例
        a = np.arange(5).view(A)
        b = np.arange(1, 3).view(A)
        s = np.arange(5).view(A)
        # 断言 searchsorted 方法的返回值类型不是 A 类型的实例
        assert_(not isinstance(a.searchsorted(b, 'left'), A))
        assert_(not isinstance(a.searchsorted(b, 'right'), A))
        assert_(not isinstance(a.searchsorted(b, 'left', s), A))
        assert_(not isinstance(a.searchsorted(b, 'right', s), A))

    # 使用 pytest 的参数化装饰器，测试 argpartition 方法的超出范围情况

    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_argpartition_out_of_range(self, dtype):
        # 创建一个包含所有数据类型的 NumPy 数组
        d = np.arange(10).astype(dtype=dtype)
        # 测试 kth 参数超出范围时是否会引发 ValueError 异常
        assert_raises(ValueError, d.argpartition, 10)
        assert_raises(ValueError, d.argpartition, -11)

    # 使用 pytest 的参数化装饰器，测试 partition 方法的超出范围情况

    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_partition_out_of_range(self, dtype):
        # 创建一个包含所有数据类型的 NumPy 数组
        d = np.arange(10).astype(dtype=dtype)
        # 测试 kth 参数超出范围时是否会引发 ValueError 异常
        assert_raises(ValueError, d.partition, 10)
        assert_raises(ValueError, d.partition, -11)

    # 测试 argpartition 方法中 kth 参数为整数的情况

    def test_argpartition_integer(self):
        # 创建一个包含整数的 NumPy 数组
        d = np.arange(10)
        # 测试 kth 参数为浮点数时是否会引发 TypeError 异常
        assert_raises(TypeError, d.argpartition, 9.)
        # 对于对象数组，也进行相似的测试
        d_obj = np.arange(10, dtype=object)
        assert_raises(TypeError, d_obj.argpartition, 9.)

    # 测试 partition 方法中 kth 参数为整数的情况

    def test_partition_integer(self):
        # 创建一个包含整数的 NumPy 数组
        d = np.arange(10)
        # 测试 kth 参数为浮点数时是否会引发 TypeError 异常
        assert_raises(TypeError, d.partition, 9.)
        # 对于对象数组，也进行相似的测试
        d_obj = np.arange(10, dtype=object)
        assert_raises(TypeError, d_obj.partition, 9.)

    # 使用 pytest 的参数化装饰器，测试 kth 参数为所有整数类型的情况

    @pytest.mark.parametrize("kth_dtype", np.typecodes["AllInteger"])
    # 定义测试空数组分区的函数，传入self和kth_dtype参数
    def test_partition_empty_array(self, kth_dtype):
        # 检查多维空数组的轴处理
        kth = np.array(0, dtype=kth_dtype)[()]
        # 创建一个空数组
        a = np.array([])
        # 设置数组的形状为(3, 2, 1, 0)
        a.shape = (3, 2, 1, 0)
        # 遍历数组的轴
        for axis in range(-a.ndim, a.ndim):
            msg = 'test empty array partition with axis={0}'.format(axis)
            # 使用断言检查分区函数对空数组的操作
            assert_equal(np.partition(a, kth, axis=axis), a, msg)
        msg = 'test empty array partition with axis=None'
        # 使用断言检查分区函数对轴为None时的操作
        assert_equal(np.partition(a, kth, axis=None), a.ravel(), msg)

    @pytest.mark.parametrize("kth_dtype", np.typecodes["AllInteger"])
    def test_argpartition_empty_array(self, kth_dtype):
        # 检查多维空数组的轴处理
        kth = np.array(0, dtype=kth_dtype)[()]
        # 创建一个空数组
        a = np.array([])
        # 设置数组的形状为(3, 2, 1, 0)
        a.shape = (3, 2, 1, 0)
        # 遍历数组的轴
        for axis in range(-a.ndim, a.ndim):
            msg = 'test empty array argpartition with axis={0}'.format(axis)
            # 使用断言检查参数分区函数对空数组的操作
            assert_equal(np.partition(a, kth, axis=axis),
                         np.zeros_like(a, dtype=np.intp), msg)
        msg = 'test empty array argpartition with axis=None'
        # 使用断言检查参数分区函数对轴为None时的操作
        assert_equal(np.partition(a, kth, axis=None),
                     np.zeros_like(a.ravel(), dtype=np.intp), msg)

    # 定义一个函数，用于断言数组是否被分区
    def assert_partitioned(self, d, kth):
        prev = 0
        # 遍历kth数组
        for k in np.sort(kth):
            # 使用断言检查数组是否被正确分区
            assert_array_compare(operator.__le__, d[prev:k], d[k],
                    err_msg='kth %d' % k)
            # 使用断言检查数组是否被正确分区
            assert_((d[k:] >= d[k]).all(),
                    msg="kth %d, %r not greater equal %r" % (k, d[k:], d[k]))
            prev = k + 1
    # 定义一个测试方法，用于测试分区操作的各种情况
    def test_partition_iterative(self):
        # 创建一个包含17个元素的NumPy数组
        d = np.arange(17)
        # 设置几个不合法的分区索引值，检查是否引发值错误异常
        kth = (0, 1, 2, 429, 231)
        assert_raises(ValueError, d.partition, kth)
        assert_raises(ValueError, d.argpartition, kth)
        
        # 将原始数组重新形状为2行5列的数组，并检查在指定轴上分区是否引发值错误异常
        d = np.arange(10).reshape((2, 5))
        assert_raises(ValueError, d.partition, kth, axis=0)
        assert_raises(ValueError, d.partition, kth, axis=1)
        assert_raises(ValueError, np.partition, d, kth, axis=1)
        assert_raises(ValueError, np.partition, d, kth, axis=None)

        # 创建一个新的数组，测试分区和argpartition函数的结果是否符合预期
        d = np.array([3, 4, 2, 1])
        p = np.partition(d, (0, 3))
        self.assert_partitioned(p, (0, 3))
        self.assert_partitioned(d[np.argpartition(d, (0, 3))], (0, 3))
        
        # 检查分区函数在使用负索引值时的行为
        assert_array_equal(p, np.partition(d, (-3, -1)))
        assert_array_equal(p, d[np.argpartition(d, (-3, -1))])

        # 对数组进行随机排序，并检查排序后数组是否与期望的顺序相同
        d = np.arange(17)
        np.random.shuffle(d)
        d.partition(range(d.size))
        assert_array_equal(np.arange(17), d)
        np.random.shuffle(d)
        assert_array_equal(np.arange(17), d[d.argpartition(range(d.size))])

        # 测试未排序的kth值对分区函数的影响
        d = np.arange(17)
        np.random.shuffle(d)
        keys = np.array([1, 3, 8, -2])
        np.random.shuffle(d)
        p = np.partition(d, keys)
        self.assert_partitioned(p, keys)
        p = d[np.argpartition(d, keys)]
        self.assert_partitioned(p, keys)
        np.random.shuffle(keys)
        assert_array_equal(np.partition(d, keys), p)
        assert_array_equal(d[np.argpartition(d, keys)], p)

        # 测试所有kth值相等的情况
        d = np.arange(20)[::-1]
        self.assert_partitioned(np.partition(d, [5]*4), [5])
        self.assert_partitioned(np.partition(d, [5]*4 + [6, 13]),
                                [5]*4 + [6, 13])
        self.assert_partitioned(d[np.argpartition(d, [5]*4)], [5])
        self.assert_partitioned(d[np.argpartition(d, [5]*4 + [6, 13])],
                                [5]*4 + [6, 13])

        # 创建一个数组，进行二维分区操作，并验证结果是否符合预期
        d = np.arange(12)
        np.random.shuffle(d)
        d1 = np.tile(np.arange(12), (4, 1))
        map(np.random.shuffle, d1)
        d0 = np.transpose(d1)

        kth = (1, 6, 7, -1)
        p = np.partition(d1, kth, axis=1)
        pa = d1[np.arange(d1.shape[0])[:, None],
                d1.argpartition(kth, axis=1)]
        assert_array_equal(p, pa)
        for i in range(d1.shape[0]):
            self.assert_partitioned(p[i,:], kth)
        p = np.partition(d0, kth, axis=0)
        pa = d0[np.argpartition(d0, kth, axis=0),
                np.arange(d0.shape[1])[None,:]]
        assert_array_equal(p, pa)
        for i in range(d0.shape[1]):
            self.assert_partitioned(p[:, i], kth)
    def test_partition_cdtype(self):
        # 创建一个包含姓名、身高和年龄的结构化 NumPy 数组
        d = np.array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
                      ('Lancelot', 1.9, 38)],
                     dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

        # 按照年龄和身高排序数组 d，并存储到 tgt 中
        tgt = np.sort(d, order=['age', 'height'])

        # 使用 np.partition 函数按照年龄和身高重新排序数组 d，然后与 tgt 进行比较
        assert_array_equal(np.partition(d, range(d.size),
                                        order=['age', 'height']),
                           tgt)

        # 使用 np.argpartition 函数找到按照年龄和身高重新排序后的索引，并与 tgt 进行比较
        assert_array_equal(d[np.argpartition(d, range(d.size),
                                             order=['age', 'height'])],
                           tgt)

        # 遍历数组 d 的大小
        for k in range(d.size):
            # 使用 np.partition 函数对数组 d 进行部分排序，并与 tgt 的相应部分进行比较
            assert_equal(np.partition(d, k, order=['age', 'height'])[k],
                         tgt[k])
            # 使用 np.argpartition 函数找到部分排序后的索引，并与 tgt 的相应部分进行比较
            assert_equal(d[np.argpartition(d, k, order=['age', 'height'])][k],
                         tgt[k])

        # 创建一个简单的字符串数组 d
        d = np.array(['Galahad', 'Arthur', 'zebra', 'Lancelot'])

        # 将数组 d 按字母顺序排序，存储到 tgt 中
        tgt = np.sort(d)

        # 使用 np.partition 函数对数组 d 进行重新排序，并与 tgt 进行比较
        assert_array_equal(np.partition(d, range(d.size)), tgt)

        # 遍历数组 d 的大小
        for k in range(d.size):
            # 使用 np.partition 函数对数组 d 进行部分排序，并与 tgt 的相应部分进行比较
            assert_equal(np.partition(d, k)[k], tgt[k])
            # 使用 np.argpartition 函数找到部分排序后的索引，并与 tgt 的相应部分进行比较
            assert_equal(d[np.argpartition(d, k)][k], tgt[k])


    def test_partition_unicode_kind(self):
        # 创建一个从 0 到 9 的整数数组 d
        d = np.arange(10)

        # 创建一个 UTF-8 编码的字符串 k
        k = b'\xc3\xa4'.decode("UTF8")

        # 断言在使用非法 kind 参数调用 d.partition 时会引发 ValueError
        assert_raises(ValueError, d.partition, 2, kind=k)

        # 断言在使用非法 kind 参数调用 d.argpartition 时会引发 ValueError
        assert_raises(ValueError, d.argpartition, 2, kind=k)


    def test_partition_fuzz(self):
        # 进行几轮随机数据测试
        for j in range(10, 30):
            for i in range(1, j - 2):
                # 创建一个长度为 j 的整数数组 d，并随机打乱顺序
                d = np.arange(j)
                np.random.shuffle(d)

                # 将 d 的值取余数，确保值在随机的边界内
                d = d % np.random.randint(2, 30)

                # 随机选择一个索引 idx
                idx = np.random.randint(d.size)

                # 创建 kth 数组，用于部分排序
                kth = [0, idx, i, i + 1]

                # 对数组 d 进行排序，并选择 kth 指定的元素，存储到 tgt 中
                tgt = np.sort(d)[kth]

                # 使用 np.partition 函数对数组 d 进行部分排序，并与 tgt 进行比较
                assert_array_equal(np.partition(d, kth)[kth], tgt,
                                   err_msg="data: %r\n kth: %r" % (d, kth))


    @pytest.mark.parametrize("kth_dtype", np.typecodes["AllInteger"])
    def test_argpartition_gh5524(self, kth_dtype):
        # 测试 argpartition 在列表上的功能

        # 创建一个包含整数的列表 d
        d = [6, 7, 3, 2, 9, 0]

        # 使用 np.argpartition 函数对列表 d 进行部分排序，并与预期的索引 [1] 进行比较
        p = np.argpartition(d, kth)
        self.assert_partitioned(np.array(d)[p], [1])


    def test_flatten(self):
        # 创建两个多维数组 x0 和 x1
        x0 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
        x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], np.int32)

        # 创建预期的展平结果 y0、y0f、y1、y1f
        y0 = np.array([1, 2, 3, 4, 5, 6], np.int32)
        y0f = np.array([1, 4, 2, 5, 3, 6], np.int32)
        y1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], np.int32)
        y1f = np.array([1, 5, 3, 7, 2, 6, 4, 8], np.int32)

        # 断言展平结果与预期结果相等
        assert_equal(x0.flatten(), y0)
        assert_equal(x0.flatten('F'), y0f)
        assert_equal(x0.flatten('F'), x0.T.flatten())
        assert_equal(x1.flatten(), y1)
        assert_equal(x1.flatten('F'), y1f)
        assert_equal(x1.flatten('F'), x1.T.flatten())
    # 定义测试方法，用于测试矩阵乘法函数的多种情况
    def test_arr_mult(self, func):
        # 创建四个 NumPy 数组作为测试用例
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[0, 1], [1, 0]])
        c = np.array([[9, 1], [1, -9]])
        d = np.arange(24).reshape(4, 6)
        
        # 预先计算好的结果数组，用于断言验证
        ddt = np.array(
            [[  55,  145,  235,  325],
             [ 145,  451,  757, 1063],
             [ 235,  757, 1279, 1801],
             [ 325, 1063, 1801, 2539]]
        )
        dtd = np.array(
            [[504, 540, 576, 612, 648, 684],
             [540, 580, 620, 660, 700, 740],
             [576, 620, 664, 708, 752, 796],
             [612, 660, 708, 756, 804, 852],
             [648, 700, 752, 804, 856, 908],
             [684, 740, 796, 852, 908, 964]]
        )

        # gemm vs syrk optimizations
        # 针对不同的数据类型进行优化测试
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            eaf = a.astype(et)
            # 断言矩阵乘法函数对相同矩阵的多种组合返回期望结果
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
            eaf = a.astype(et)
            ebf = b.astype(et)
            # 断言矩阵乘法函数对同一矩阵的转置返回期望结果
            assert_equal(func(ebf, ebf), eaf)
            assert_equal(func(ebf.T, ebf), eaf)
            assert_equal(func(ebf, ebf.T), eaf)
            assert_equal(func(ebf.T, ebf.T), eaf)

        # syrk - different shape, stride, and view validations
        # 针对不同形状、步长和视图的矩阵进行验证
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)
            # 断言矩阵乘法函数对不同切片的矩阵返回期望结果
            assert_equal(
                func(edf[::-1, :], edf.T),
                func(edf[::-1, :].copy(), edf.T.copy())
            )
            assert_equal(
                func(edf[:, ::-1], edf.T),
                func(edf[:, ::-1].copy(), edf.T.copy())
            )
            assert_equal(
                func(edf, edf[::-1, :].T),
                func(edf, edf[::-1, :].T.copy())
            )
            assert_equal(
                func(edf, edf[:, ::-1].T),
                func(edf, edf[:, ::-1].T.copy())
            )
            assert_equal(
                func(edf[:edf.shape[0] // 2, :], edf[::2, :].T),
                func(edf[:edf.shape[0] // 2, :].copy(), edf[::2, :].T.copy())
            )
            assert_equal(
                func(edf[::2, :], edf[:edf.shape[0] // 2, :].T),
                func(edf[::2, :].copy(), edf[:edf.shape[0] // 2, :].T.copy())
            )

        # syrk - different shape
        # 针对不同形状的矩阵进行验证
        for et in [np.float32, np.float64, np.complex64, np.complex128]:
            edf = d.astype(et)
            eddtf = ddt.astype(et)
            edtdf = dtd.astype(et)
            # 断言矩阵乘法函数对不同形状矩阵返回期望结果
            assert_equal(func(edf, edf.T), eddtf)
            assert_equal(func(edf.T, edf), edtdf)

    @pytest.mark.parametrize('func', (np.dot, np.matmul))
    @pytest.mark.parametrize('dtype', 'ifdFD')
    # 使用 pytest 的参数化装饰器，对 dtype 参数进行参数化测试，依次使用 'ifdFD' 中的每个值
    def test_no_dgemv(self, func, dtype):
        # 检查在执行 gemv 前向量参数是否连续
        # gh-12156，GitHub 上的 issue 编号
        a = np.arange(8.0, dtype=dtype).reshape(2, 4)
        # 创建一个浮点类型的数组 a，形状为 (2, 4)，元素值为 0.0 到 7.0
        b = np.broadcast_to(1., (4, 1))
        # 创建一个广播到 (4, 1) 形状的数组 b，元素值全部为 1.0
        ret1 = func(a, b)
        ret2 = func(a, b.copy())
        # 分别使用 a, b 和 a, b 的副本调用 func 函数，并将结果分别赋给 ret1 和 ret2
        assert_equal(ret1, ret2)

        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T)
        # 分别使用 b 的转置和 a 的转置调用 func 函数，并将结果分别赋给 ret1 和 ret2
        assert_equal(ret1, ret2)

        # 检查非对齐数据
        dt = np.dtype(dtype)
        # 创建一个指定 dtype 的 numpy 数据类型对象 dt
        a = np.zeros(8 * dt.itemsize // 2 + 1, dtype='int16')[1:].view(dtype)
        # 创建一个长度为 8*dt.itemsize//2+1 的 int16 类型数组 a，从索引 1 开始视图转换为 dtype 类型
        a = a.reshape(2, 4)
        # 将 a 重新调整为 (2, 4) 形状的数组
        b = a[0]
        # b 是 a 的第一行数据
        # 确保数据非对齐
        assert_(a.__array_interface__['data'][0] % dt.itemsize != 0)
        # 检查 a 的数据内存地址是否不是 dt.itemsize 的倍数
        ret1 = func(a, b)
        ret2 = func(a.copy(), b.copy())
        # 分别使用 a, b 和它们的副本调用 func 函数，并将结果分别赋给 ret1 和 ret2
        assert_equal(ret1, ret2)

        ret1 = func(b.T, a.T)
        ret2 = func(b.T.copy(), a.T.copy())
        # 分别使用 b 的转置和 a 的转置的副本调用 func 函数，并将结果分别赋给 ret1 和 ret2
        assert_equal(ret1, ret2)

    def test_dot(self):
        a = np.array([[1, 0], [0, 1]])
        b = np.array([[0, 1], [1, 0]])
        c = np.array([[9, 1], [1, -9]])
        # 函数与方法的对比
        assert_equal(np.dot(a, b), a.dot(b))
        # 断言 np.dot(a, b) 与 a.dot(b) 的结果是否相等
        assert_equal(np.dot(np.dot(a, b), c), a.dot(b).dot(c))
        # 断言 np.dot(np.dot(a, b), c) 与 a.dot(b).dot(c) 的结果是否相等

        # 测试传入输出数组
        c = np.zeros_like(a)
        a.dot(b, c)
        # 使用输出数组 c 调用 a.dot(b)
        assert_equal(c, np.dot(a, b))
        # 断言 c 是否与 np.dot(a, b) 的结果相等

        # 测试关键字参数
        c = np.zeros_like(a)
        a.dot(b=b, out=c)
        # 使用关键字参数 b 和输出数组 c 调用 a.dot()
        assert_equal(c, np.dot(a, b))
        # 断言 c 是否与 np.dot(a, b) 的结果相等

    def test_dot_type_mismatch(self):
        c = 1.
        A = np.array((1,1), dtype='i,i')
        # 断言 TypeError 是否会被抛出，当尝试对 c 和 A 使用 np.dot() 函数时
        assert_raises(TypeError, np.dot, c, A)
        assert_raises(TypeError, np.dot, A, c)

    def test_dot_out_mem_overlap(self):
        np.random.seed(1)

        # 测试 BLAS 和非-BLAS 代码路径，包括所有 dot() 支持的数据类型
        dtypes = [np.dtype(code) for code in np.typecodes['All']
                  if code not in 'USVM']
        for dtype in dtypes:
            a = np.random.rand(3, 3).astype(dtype)
            # 创建一个随机数据的 (3, 3) 形状的数组 a，类型为指定的 dtype

            # 有效的 dot() 输出数组必须对齐
            b = _aligned_zeros((3, 3), dtype=dtype)
            # 创建一个与 dtype 对齐的零数组 b
            b[...] = np.random.rand(3, 3)
            # 将随机数据赋值给 b

            y = np.dot(a, b)
            # 计算 np.dot(a, b) 的结果
            x = np.dot(a, b, out=b)
            # 将 np.dot(a, b) 的结果存储到输出数组 b 中
            assert_equal(x, y, err_msg=repr(dtype))
            # 断言 x 和 y 是否相等，用于验证输出数组功能的正确性

            # 检查无效的输出数组
            assert_raises(ValueError, np.dot, a, b, out=b[::2])
            # 断言调用 np.dot(a, b, out=b[::2]) 是否会引发 ValueError 异常
            assert_raises(ValueError, np.dot, a, b, out=b.T)
            # 断言调用 np.dot(a, b, out=b.T) 是否会引发 ValueError 异常

    def test_dot_matmul_out(self):
        # gh-9641，GitHub 上的 issue 编号

        class Sub(np.ndarray):
            pass
        # 定义一个继承自 np.ndarray 的子类 Sub

        a = np.ones((2, 2)).view(Sub)
        b = np.ones((2, 2)).view(Sub)
        out = np.ones((2, 2))
        # 创建形状为 (2, 2) 的全为 1 的数组 a, b 和输出数组 out

        # 确保 out 可以是任何 ndarray（不仅仅是输入的子类）
        np.dot(a, b, out=out)
        # 调用 np.dot(a, b)，将结果存储到输出数组 out 中
        np.matmul(a, b, out=out)
        # 调用 np.matmul(a, b)，将结果存储到输出数组 out 中
    def test_dot_matmul_inner_array_casting_fails(self):
        # 定义一个名为 A 的类，其中定义了 __array__() 方法，但抛出了 NotImplementedError
        class A:
            def __array__(self, *args, **kwargs):
                raise NotImplementedError

        # 测试 np.dot()、np.matmul() 和 np.inner() 是否会捕获到 NotImplementedError 异常
        assert_raises(NotImplementedError, np.dot, A(), A())
        assert_raises(NotImplementedError, np.matmul, A(), A())
        assert_raises(NotImplementedError, np.inner, A(), A())

    def test_matmul_out(self):
        # 创建一个 2x3x3 的三维数组 a，包含从 0 到 17 的整数
        a = np.arange(18).reshape(2, 3, 3)
        # 计算 a 与自身的矩阵乘积，存储结果在 b 中
        b = np.matmul(a, a)
        # 将 a 与自身的矩阵乘积存储在 a 中，使用 out 参数避免额外的内存分配
        c = np.matmul(a, a, out=a)
        # 断言 c 和 a 是同一个对象
        assert_(c is a)
        # 断言 c 的内容与 b 相同
        assert_equal(c, b)
        
        # 重新创建 a 数组
        a = np.arange(18).reshape(2, 3, 3)
        # 将 a 与自身的矩阵乘积存储在 a 的反向切片中
        c = np.matmul(a, a, out=a[::-1, ...])
        # 断言 c 和 a 共享相同的基础数据
        assert_(c.base is a.base)
        # 断言 c 的内容与 b 相同
        assert_equal(c, b)

    def test_diagonal(self):
        # 创建一个 3x4 的二维数组 a，包含从 0 到 11 的整数
        a = np.arange(12).reshape((3, 4))
        # 获取数组 a 的主对角线元素，期望结果为 [0, 5, 10]
        assert_equal(a.diagonal(), [0, 5, 10])
        # 同上，指定主对角线偏移为 0，期望结果为 [0, 5, 10]
        assert_equal(a.diagonal(0), [0, 5, 10])
        # 获取数组 a 的第一条上对角线元素，期望结果为 [1, 6, 11]
        assert_equal(a.diagonal(1), [1, 6, 11])
        # 获取数组 a 的第一条下对角线元素，期望结果为 [4, 9]
        assert_equal(a.diagonal(-1), [4, 9])
        # 断言调用 a.diagonal() 方法时指定了错误的轴参数，预期会抛出 AxisError 异常
        assert_raises(AxisError, a.diagonal, axis1=0, axis2=5)
        assert_raises(AxisError, a.diagonal, axis1=5, axis2=0)
        assert_raises(AxisError, a.diagonal, axis1=5, axis2=5)
        # 断言调用 a.diagonal() 方法时指定了相同的轴参数，预期会抛出 ValueError 异常
        assert_raises(ValueError, a.diagonal, axis1=1, axis2=1)

        # 创建一个 2x2x2 的三维数组 b，包含从 0 到 7 的整数
        b = np.arange(8).reshape((2, 2, 2))
        # 获取数组 b 的主对角线元素，期望结果为 [[0, 6], [1, 7]]
        assert_equal(b.diagonal(), [[0, 6], [1, 7]])
        # 同上，指定主对角线偏移为 0，期望结果为 [[0, 6], [1, 7]]
        assert_equal(b.diagonal(0), [[0, 6], [1, 7]])
        # 获取数组 b 的第一条上对角线元素，期望结果为 [[2], [3]]
        assert_equal(b.diagonal(1), [[2], [3]])
        # 获取数组 b 的第一条下对角线元素，期望结果为 [[4], [5]]
        assert_equal(b.diagonal(-1), [[4], [5]])
        # 断言调用 b.diagonal() 方法时指定了错误的轴参数，预期会抛出 ValueError 异常
        assert_raises(ValueError, b.diagonal, axis1=0, axis2=0)
        # 获取数组 b 沿第 0 和第 1 轴的对角线元素，期望结果为 [[0, 3], [4, 7]]
        assert_equal(b.diagonal(0, 1, 2), [[0, 3], [4, 7]])
        # 获取数组 b 沿第 0 和第 0 轴的对角线元素，期望结果为 [[0, 6], [1, 7]]
        assert_equal(b.diagonal(0, 0, 1), [[0, 6], [1, 7]])
        # 获取数组 b 沿第 0 和第 2 轴的对角线元素，期望结果为 [[1], [3]]
        assert_equal(b.diagonal(offset=1, axis1=0, axis2=2), [[1], [3]])
        # 断言调用 b.diagonal() 方法时轴参数的顺序不影响结果，期望结果为 [[0, 3], [4, 7]]
        assert_equal(b.diagonal(0, 2, 1), [[0, 3], [4, 7]])

    def test_diagonal_view_notwriteable(self):
        # 创建单位矩阵的主对角线视图 a
        a = np.eye(3).diagonal()
        # 断言 a 不可写入
        assert_(not a.flags.writeable)
        # 断言 a 不拥有数据内存
        assert_(not a.flags.owndata)

        # 获取单位矩阵的主对角线视图，并赋值给 a
        a = np.diagonal(np.eye(3))
        # 断言 a 不可写入
        assert_(not a.flags.writeable)
        # 断言 a 不拥有数据内存
        assert_(not a.flags.owndata)

        # 创建单位矩阵的主对角线视图，并赋值给 a
        a = np.diag(np.eye(3))
        # 断言 a 不可写入
        assert_(not a.flags.writeable)
        # 断言 a 不拥有数据内存
        assert_(not a.flags.owndata)

    def test_diagonal_memleak(self):
        # 创建一个形状为 (100, 100) 的全零数组 a
        a = np.zeros((100, 100))
        # 如果有引用计数功能，断言数组 a 的引用计数小于 50
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a) < 50)
        # 多次调用数组 a 的对角线视图，检查是否存在内存泄漏
        for i in range(100):
            a.diagonal()
        # 如果有引用计数功能，再次断言数组 a 的引用计数小于 50
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a) < 50)
    # 定`
    # 定义一个单元测试方法，用于检测大小为零的情况下是否存在内存泄漏
    def test_size_zero_memleak(self):
        # 回归测试用例，针对 issue 9615
        # 在 cblasfuncs 中，测试处理长度为零的特殊情况，针对浮点数数据类型
        a = np.array([], dtype=np.float64)
        x = np.array(2.0)
        # 进行100次迭代，调用 np.dot，对长度为零的情况进行特别处理
        for _ in range(100):
            np.dot(a, a, out=x)
        # 如果支持引用计数，则检查变量 x 的引用计数是否小于50
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(x) < 50)

    # 定义一个测试方法，用于测试 numpy 数组的迹（trace）方法
    def test_trace(self):
        # 创建一个形状为 (3, 4) 的 numpy 数组 a
        a = np.arange(12).reshape((3, 4))
        # 断言数组 a 的迹（主对角线元素之和）为 15
        assert_equal(a.trace(), 15)
        # 断言数组 a 沿着第0轴（行）的迹为 15
        assert_equal(a.trace(0), 15)
        # 断言数组 a 沿着第1轴（列）的迹为 18
        assert_equal(a.trace(1), 18)
        # 断言数组 a 沿着倒数第1轴（倒数第2维的迹）为 13
        assert_equal(a.trace(-1), 13)

        # 创建一个形状为 (2, 2, 2) 的 numpy 数组 b
        b = np.arange(8).reshape((2, 2, 2))
        # 断言数组 b 每个子数组的迹为 [6, 8]
        assert_equal(b.trace(), [6, 8])
        # 断言数组 b 沿着第0轴的迹为 [6, 8]
        assert_equal(b.trace(0), [6, 8])
        # 断言数组 b 沿着第1轴的迹为 [2, 3]
        assert_equal(b.trace(1), [2, 3])
        # 断言数组 b 沿着倒数第1轴的迹为 [4, 5]
        assert_equal(b.trace(-1), [4, 5])
        # 断言数组 b 在指定的轴索引上的迹为 [6, 8]
        assert_equal(b.trace(0, 0, 1), [6, 8])
        # 断言数组 b 在指定的轴索引上的迹为 [5, 9]
        assert_equal(b.trace(0, 0, 2), [5, 9])
        # 断言数组 b 在指定的轴索引上的迹为 [3, 11]
        assert_equal(b.trace(0, 1, 2), [3, 11])
        # 断言数组 b 在指定的轴索引上的迹为 [1, 3]
        assert_equal(b.trace(offset=1, axis1=0, axis2=2), [1, 3])

        # 创建一个形状为 (2, 2, 2) 的 numpy 数组 b 的视图，并指定其为自定义类 MyArray 的实例
        b = np.arange(8).reshape((2, 2, 2)).view(MyArray)
        # 调用 b 的迹方法，并断言返回结果的类型为 MyArray 类型
        t = b.trace()
        assert_(isinstance(t, MyArray))
    # 定义一个测试方法 test_put，用于测试 np.put 方法的功能
    def test_put(self):
        # 获取所有整数和浮点数的数据类型代码
        icodes = np.typecodes['AllInteger']
        fcodes = np.typecodes['AllFloat']
        # 遍历所有整数、浮点数和对象数据类型代码
        for dt in icodes + fcodes + 'O':
            # 创建目标数组 tgt，包含指定数据类型的元素
            tgt = np.array([0, 1, 0, 3, 0, 5], dtype=dt)

            # 测试 1 维数组情况
            a = np.zeros(6, dtype=dt)
            # 使用 np.put 方法将指定位置的元素更新为指定值
            a.put([1, 3, 5], [1, 3, 5])
            # 断言数组 a 是否与目标数组 tgt 相等
            assert_equal(a, tgt)

            # 测试 2 维数组情况
            a = np.zeros((2, 3), dtype=dt)
            # 使用 np.put 方法将指定位置的元素更新为指定值
            a.put([1, 3, 5], [1, 3, 5])
            # 断言数组 a 是否与按照目标形状重塑后的 tgt 相等
            assert_equal(a, tgt.reshape(2, 3))

        # 对于布尔值数据类型
        for dt in '?':
            # 创建目标数组 tgt，包含指定数据类型的元素
            tgt = np.array([False, True, False, True, False, True], dtype=dt)

            # 测试 1 维数组情况
            a = np.zeros(6, dtype=dt)
            # 使用 np.put 方法将指定位置的元素更新为 True
            a.put([1, 3, 5], [True]*3)
            # 断言数组 a 是否与目标数组 tgt 相等
            assert_equal(a, tgt)

            # 测试 2 维数组情况
            a = np.zeros((2, 3), dtype=dt)
            # 使用 np.put 方法将指定位置的元素更新为 True
            a.put([1, 3, 5], [True]*3)
            # 断言数组 a 是否与按照目标形状重塑后的 tgt 相等
            assert_equal(a, tgt.reshape(2, 3))

        # 检查数组必须可写
        a = np.zeros(6)
        # 将数组的写入标志设置为 False
        a.flags.writeable = False
        # 断言调用 np.put 方法时会抛出 ValueError 异常
        assert_raises(ValueError, a.put, [1, 3, 5], [1, 3, 5])

        # 当调用 np.put 时，确保如果对象不是 ndarray 类型，会抛出 TypeError 异常
        bad_array = [1, 2, 3]
        assert_raises(TypeError, np.put, bad_array, [0, 2], 5)

        # 当调用 np.put 时，确保如果数组为空，会抛出 IndexError 异常
        empty_array = np.asarray(list())
        # 使用 pytest 检查是否抛出预期的 IndexError 异常
        with pytest.raises(IndexError, match="cannot replace elements of an empty array"):
            np.put(empty_array, 1, 1, mode="wrap")
        with pytest.raises(IndexError, match="cannot replace elements of an empty array"):
            np.put(empty_array, 1, 1, mode="clip")


    # 定义测试方法 test_ravel_subclass，用于测试 np.ravel 方法在子类数组上的行为
    def test_ravel_subclass(self):
        # 定义一个数组子类 ArraySubclass，继承自 np.ndarray
        class ArraySubclass(np.ndarray):
            pass

        # 创建一个 ArraySubclass 的实例 a，包含整数序列，并调用 np.ravel 方法
        a = np.arange(10).view(ArraySubclass)
        # 断言 np.ravel 方法返回的对象是否仍然是 ArraySubclass 的实例
        assert_(isinstance(a.ravel('C'), ArraySubclass))
        assert_(isinstance(a.ravel('F'), ArraySubclass))
        assert_(isinstance(a.ravel('A'), ArraySubclass))
        assert_(isinstance(a.ravel('K'), ArraySubclass))

        # 创建一个 ArraySubclass 的实例 a，包含间隔取值的整数序列，并调用 np.ravel 方法
        a = np.arange(10)[::2].view(ArraySubclass)
        # 断言 np.ravel 方法返回的对象是否仍然是 ArraySubclass 的实例
        assert_(isinstance(a.ravel('C'), ArraySubclass))
        assert_(isinstance(a.ravel('F'), ArraySubclass))
        assert_(isinstance(a.ravel('A'), ArraySubclass))
        assert_(isinstance(a.ravel('K'), ArraySubclass))
    # 定义测试函数 test_swapaxes
    def test_swapaxes(self):
        # 创建一个四维数组 a，形状为 (1, 2, 3, 4)，并复制一份给变量 a
        a = np.arange(1*2*3*4).reshape(1, 2, 3, 4).copy()
        # 使用 np.indices 函数生成数组 a 的索引
        idx = np.indices(a.shape)
        # 断言数组 a 拥有自己的数据副本
        assert_(a.flags['OWNDATA'])
        # 复制数组 a 给数组 b
        b = a.copy()

        # 检查异常情况
        assert_raises(AxisError, a.swapaxes, -5, 0)
        assert_raises(AxisError, a.swapaxes, 4, 0)
        assert_raises(AxisError, a.swapaxes, 0, -5)
        assert_raises(AxisError, a.swapaxes, 0, 4)

        # 进行多重循环来测试 swapaxes 方法的不同参数组合
        for i in range(-4, 4):
            for j in range(-4, 4):
                for k, src in enumerate((a, b)):
                    # 对数组 src 使用 swapaxes 方法，交换轴 i 和 j
                    c = src.swapaxes(i, j)
                    # 检查交换后的形状是否符合预期
                    shape = list(src.shape)
                    shape[i] = src.shape[j]
                    shape[j] = src.shape[i]
                    assert_equal(c.shape, shape, str((i, j, k)))
                    # 检查交换后的数组内容是否正确
                    i0, i1, i2, i3 = [dim-1 for dim in c.shape]
                    j0, j1, j2, j3 = [dim-1 for dim in src.shape]
                    assert_equal(src[idx[j0], idx[j1], idx[j2], idx[j3]],
                                 c[idx[i0], idx[i1], idx[i2], idx[i3]],
                                 str((i, j, k)))
                    # 断言返回的是视图而非拥有数据
                    assert_(not c.flags['OWNDATA'], str((i, j, k)))
                    # 检查非连续输入数组的情况
                    if k == 1:
                        b = c

    # 定义测试函数 test_conjugate
    def test_conjugate(self):
        # 测试复数数组的共轭方法
        a = np.array([1-1j, 1+1j, 23+23.0j])
        ac = a.conj()
        # 检查实部是否相等
        assert_equal(a.real, ac.real)
        # 检查虚部是否相反
        assert_equal(a.imag, -ac.imag)
        # 检查共轭数组是否等于调用 conjugate 方法的结果
        assert_equal(ac, a.conjugate())
        # 检查共轭数组是否等于调用 np.conjugate 的结果
        assert_equal(ac, np.conjugate(a))

        # 使用 'F' 指定列优先存储的复数数组
        a = np.array([1-1j, 1+1j, 23+23.0j], 'F')
        ac = a.conj()
        assert_equal(a.real, ac.real)
        assert_equal(a.imag, -ac.imag)
        assert_equal(ac, a.conjugate())
        assert_equal(ac, np.conjugate(a))

        # 测试实数数组的共轭方法
        a = np.array([1, 2, 3])
        ac = a.conj()
        assert_equal(a, ac)
        assert_equal(ac, a.conjugate())
        assert_equal(ac, np.conjugate(a))

        # 测试浮点数数组的共轭方法
        a = np.array([1.0, 2.0, 3.0])
        ac = a.conj()
        assert_equal(a, ac)
        assert_equal(ac, a.conjugate())
        assert_equal(ac, np.conjugate(a))

        # 测试对象数组的共轭方法
        a = np.array([1-1j, 1, 2.0], object)
        ac = a.conj()
        assert_equal(ac, [k.conjugate() for k in a])
        assert_equal(ac, a.conjugate())
        assert_equal(ac, np.conjugate(a))

        # 对包含不同类型对象的数组进行类型错误的共轭方法测试
        a = np.array([1-1j, 1, 2.0, 'f'], object)
        assert_raises(TypeError, lambda: a.conj())
        assert_raises(TypeError, lambda: a.conjugate())

    # 定义测试函数 test_conjugate_out
    def test_conjugate_out(self):
        # 测试共轭方法的 out 参数传递情况
        # 注意：目前尚未记录能够传递 `out` 参数的功能！
        a = np.array([1-1j, 1+1j, 23+23.0j])
        out = np.empty_like(a)
        res = a.conjugate(out)
        assert res is out
        assert_array_equal(out, a.conjugate())
    # 定义一个测试方法，用于测试复数转换功能的各种情况
    def test__complex__(self):
        # 定义各种数据类型的列表，用于测试
        dtypes = ['i1', 'i2', 'i4', 'i8',
                  'u1', 'u2', 'u4', 'u8',
                  'f', 'd', 'g', 'F', 'D', 'G',
                  '?', 'O']
        # 遍历每种数据类型
        for dt in dtypes:
            # 创建不同形式的numpy数组，设置dtype为当前类型
            a = np.array(7, dtype=dt)
            b = np.array([7], dtype=dt)
            c = np.array([[[[[7]]]]], dtype=dt)

            # 准备测试消息，指示当前数据类型
            msg = 'dtype: {0}'.format(dt)
            # 尝试将a转换为复数，并断言转换后与原数组相等
            ap = complex(a)
            assert_equal(ap, a, msg)

            # 使用断言警告来测试，尝试将b转换为复数，预期产生DeprecationWarning
            with assert_warns(DeprecationWarning):
                bp = complex(b)
            assert_equal(bp, b, msg)

            # 使用断言警告来测试，尝试将c转换为复数，预期产生DeprecationWarning
            with assert_warns(DeprecationWarning):
                cp = complex(c)
            assert_equal(cp, c, msg)

    # 定义测试方法，用于测试复数转换不应工作的情况
    def test__complex__should_not_work(self):
        # 定义各种数据类型的列表，用于测试
        dtypes = ['i1', 'i2', 'i4', 'i8',
                  'u1', 'u2', 'u4', 'u8',
                  'f', 'd', 'g', 'F', 'D', 'G',
                  '?', 'O']
        # 遍历每种数据类型
        for dt in dtypes:
            # 创建数组a，尝试将其转换为复数，预期引发TypeError
            a = np.array([1, 2, 3], dtype=dt)
            assert_raises(TypeError, complex, a)

        # 创建结构化数据类型为('a', 'f8'), ('b', 'i1')的数组b，尝试将其转换为复数，预期引发TypeError
        dt = np.dtype([('a', 'f8'), ('b', 'i1')])
        b = np.array((1.0, 3), dtype=dt)
        assert_raises(TypeError, complex, b)

        # 创建结构化数据类型为('a', 'f8'), ('b', 'i1')的数组c，尝试将其转换为复数，预期引发TypeError
        c = np.array([(1.0, 3), (2e-3, 7)], dtype=dt)
        assert_raises(TypeError, complex, c)

        # 创建字符串数组d，尝试将其转换为复数，预期引发TypeError
        d = np.array('1+1j')
        assert_raises(TypeError, complex, d)

        # 创建Unicode数组e，尝试将其转换为复数，预期引发TypeError，并可能引发DeprecationWarning
        e = np.array(['1+1j'], 'U')
        with assert_warns(DeprecationWarning):
            assert_raises(TypeError, complex, e)
classpython
# 定义一个测试类 TestCequenceMethods
class TestCequenceMethods:
    # 定义测试数组是否包含元素的测试方法
    def test_array_contains(self):
        # 断言 4.0 是否在由 np.arange(16.) 生成的数组的reshape(4,4)中
        assert_(4.0 in np.arange(16.).reshape(4,4))
        # 断言 20.0 是否不在由 np.arange(16.) 生成的数组的reshape(4,4)中
        assert_(20.0 not in np.arange(16.).reshape(4,4))

# 定义一个测试类 TestBinop
class TestBinop:
    # 定义测试原地操作的测试方法
    def test_inplace(self):
        # 测试 refcount 1 的原地转换
        assert_array_almost_equal(np.array([0.5]) * np.array([1.0, 2.0]), [0.5, 1.0])

        # 创建数组 d，通过切片方式获取 np.array([0.5, 0.5]) 中的第一个元素，进行原地操作
        d = np.array([0.5, 0.5])[::2]
        assert_array_almost_equal(d * (d * np.array([1.0, 2.0])), [0.25, 0.5])

        # 创建数组 a 和 b，进行多种原地操作，并进行断言
        a = np.array([0.5])
        b = np.array([0.5])
        c = a + b
        c = a - b
        c = a * b
        c = a / b
        assert_equal(a, b)
        assert_almost_equal(c, 1.)

        # 进行复合操作，并进行断言
        c = a + b * 2. / b * a - a / b
        assert_equal(a, b)
        assert_equal(c, 0.5)

        # 测试真除法操作，并进行断言
        a = np.array([5])
        b = np.array([3])
        c = (a * a) / b
        assert_almost_equal(c, 25 / 3)
        assert_equal(a, 5)
        assert_equal(b, 3)

    # ndarray.__rop__ 总是调用 ufunc
    # ndarray.__iop__ 总是调用 ufunc
    # ndarray.__op__, __rop__:
    #   - 如果 other 有 __array_ufunc__ 且为 None，或者 other 不是子类且具有更高的 array 优先级，则推迟
    #           或者调用 ufunc
    @pytest.mark.xfail(IS_PYPY, reason="Bug in pypy3.{9, 10}-v7.3.13, #24862")
    @pytest.mark.parametrize("priority", [None, "runtime error"])
    # 测试 ufunc 二元操作的坏 array 优先级
    def test_ufunc_binop_bad_array_priority(self, priority):
        # 主要检查这不会导致崩溃。第二个数组的优先级低于 -1（"error value"）。
        # 如果 __radd__ 实际存在，可能会发生糟糕的事情（我认为是通过标量路径）。
        # 原则上，这两者可能都只是将来的错误。
        class BadPriority:
            @property
            def __array_priority__(self):
                if priority == "runtime error":
                    raise RuntimeError("RuntimeError in __array_priority__!")
                return priority

            def __radd__(self, other):
                return "result"

        class LowPriority(np.ndarray):
            __array_priority__ = -1000

        # 优先级失败使用与标量相同的优先级（较小的 -1000）。所以 LowPriority 在每个元素（内部操作）上都会赢得 'result'。
        res = np.arange(3).view(LowPriority) + BadPriority()
        assert res.shape == (3,)
        assert res[0] == 'result'
    # 定义测试函数 test_ufunc_override_normalize_signature
    def test_ufunc_override_normalize_signature(self):
        # GitHub issue 5674 的测试用例
        # 定义一个内部类 SomeClass，实现 __array_ufunc__ 方法
        class SomeClass:
            def __array_ufunc__(self, ufunc, method, *inputs, **kw):
                # 返回关键字参数 kw
                return kw
        
        # 创建 SomeClass 的实例 a
        a = SomeClass()
        
        # 调用 np.add 函数，传入 a 和 [1] 作为输入，kw 接收返回的关键字参数
        kw = np.add(a, [1])
        # 断言检查 kw 中不包含 'sig' 和 'signature' 键
        assert_('sig' not in kw and 'signature' not in kw)
        
        # 再次调用 np.add 函数，传入 a 和 [1] 作为输入，并指定 sig='ii->i'
        kw = np.add(a, [1], sig='ii->i')
        # 断言检查 kw 中不包含 'sig' 键，但包含 'signature' 键，并且其值为 'ii->i'
        assert_('sig' not in kw and 'signature' in kw)
        assert_equal(kw['signature'], 'ii->i')
        
        # 再次调用 np.add 函数，传入 a 和 [1] 作为输入，并指定 signature='ii->i'
        kw = np.add(a, [1], signature='ii->i')
        # 断言检查 kw 中不包含 'sig' 键，但包含 'signature' 键，并且其值为 'ii->i'
        assert_('sig' not in kw and 'signature' in kw)
        assert_equal(kw['signature'], 'ii->i')

    # 定义测试函数 test_array_ufunc_index
    def test_array_ufunc_index(self):
        # 检查 index 是否适当设置，以及只传递输出的情况（后者是针对 GitHub bug 4753 的另一个回归测试）
        # 这也隐含检查了 'out' 参数始终是一个元组的情况。
        # 定义内部类 CheckIndex，实现 __array_ufunc__ 方法
        class CheckIndex:
            def __array_ufunc__(self, ufunc, method, *inputs, **kw):
                # 遍历输入参数 inputs
                for i, a in enumerate(inputs):
                    # 如果某个输入参数 a 是 self，则返回其索引 i
                    if a is self:
                        return i
                # 如果执行到这里，说明必须是输出的情况
                # 再次遍历关键字参数 kw 中的 'out' 元组
                for j, a in enumerate(kw['out']):
                    # 如果某个输出参数 a 是 self，则返回元组 (j,)
                    if a is self:
                        return (j,)
        
        # 创建 CheckIndex 的实例 a
        a = CheckIndex()
        dummy = np.arange(2.)
        
        # 1 个输入，1 个输出
        assert_equal(np.sin(a), 0)
        assert_equal(np.sin(dummy, a), (0,))
        assert_equal(np.sin(dummy, out=a), (0,))
        assert_equal(np.sin(dummy, out=(a,)), (0,))
        assert_equal(np.sin(a, a), 0)
        assert_equal(np.sin(a, out=a), 0)
        assert_equal(np.sin(a, out=(a,)), 0)
        
        # 1 个输入，2 个输出
        assert_equal(np.modf(dummy, a), (0,))
        assert_equal(np.modf(dummy, None, a), (1,))
        assert_equal(np.modf(dummy, dummy, a), (1,))
        assert_equal(np.modf(dummy, out=(a, None)), (0,))
        assert_equal(np.modf(dummy, out=(a, dummy)), (0,))
        assert_equal(np.modf(dummy, out=(None, a)), (1,))
        assert_equal(np.modf(dummy, out=(dummy, a)), (1,))
        assert_equal(np.modf(a, out=(dummy, a)), 0)
        
        # 使用 assert_raises 检查 TypeError 异常
        with assert_raises(TypeError):
            # 因为有多个输出，所以 out 参数必须是一个元组
            np.modf(dummy, out=a)
        
        # 使用 assert_raises 检查 ValueError 异常
        assert_raises(ValueError, np.modf, dummy, out=(a,))
        
        # 2 个输入，1 个输出
        assert_equal(np.add(a, dummy), 0)
        assert_equal(np.add(dummy, a), 1)
        assert_equal(np.add(dummy, dummy, a), (0,))
        assert_equal(np.add(dummy, a, a), 1)
        assert_equal(np.add(dummy, dummy, out=a), (0,))
        assert_equal(np.add(dummy, dummy, out=(a,)), (0,))
        assert_equal(np.add(a, dummy, out=a), 0)
    def test_out_override(self):
        # regression test for github bug 4753
        # 定义一个继承自 np.ndarray 的特殊类 OutClass
        class OutClass(np.ndarray):
            # 重载 __array_ufunc__ 方法用于处理数组操作
            def __array_ufunc__(self, ufunc, method, *inputs, **kw):
                # 如果关键字参数中包含 'out'，则处理特定逻辑
                if 'out' in kw:
                    tmp_kw = kw.copy()
                    tmp_kw.pop('out')
                    # 调用 ufunc 的指定方法来更新 'out' 参数的值
                    func = getattr(ufunc, method)
                    kw['out'][0][...] = func(*inputs, **tmp_kw)

        # 创建一个 np.ndarray 的实例 A，并转为 OutClass 类型
        A = np.array([0]).view(OutClass)
        B = np.array([5])
        C = np.array([6])
        # 使用 np.multiply 函数，将 C 和 B 相乘，结果存入 A 中
        np.multiply(C, B, A)
        # 断言 A[0] 的值为 30
        assert_equal(A[0], 30)
        # 断言 A 是 OutClass 类的实例
        assert_(isinstance(A, OutClass))
        # 将 A[0] 的值重新设置为 0
        A[0] = 0
        # 使用 np.multiply 函数，将 C 和 B 相乘，将结果存入 A
        np.multiply(C, B, out=A)
        # 再次断言 A[0] 的值为 30
        assert_equal(A[0], 30)
        # 再次断言 A 是 OutClass 类的实例
        assert_(isinstance(A, OutClass))

    def test_pow_override_with_errors(self):
        # regression test for gh-9112
        # 定义一个继承自 np.ndarray 的特殊类 PowerOnly
        class PowerOnly(np.ndarray):
            # 重载 __array_ufunc__ 方法用于处理数组操作
            def __array_ufunc__(self, ufunc, method, *inputs, **kw):
                # 如果 ufunc 不是 np.power，则抛出 NotImplementedError
                if ufunc is not np.power:
                    raise NotImplementedError
                return "POWER!"

        # 创建一个 np.float64 类型的数组 a，并转为 PowerOnly 类型
        # 确保使用快速路径计算幂操作
        a = np.array(5., dtype=np.float64).view(PowerOnly)
        # 断言 a 的 2.5 次幂为字符串 "POWER!"
        assert_equal(a ** 2.5, "POWER!")
        # 使用 assert_raises 断言以下各项会抛出 NotImplementedError 异常
        with assert_raises(NotImplementedError):
            a ** 0.5
        with assert_raises(NotImplementedError):
            a ** 0
        with assert_raises(NotImplementedError):
            a ** 1
        with assert_raises(NotImplementedError):
            a ** -1
        with assert_raises(NotImplementedError):
            a ** 2

    def test_pow_array_object_dtype(self):
        # test pow on arrays of object dtype
        # 定义一个类 SomeClass
        class SomeClass:
            def __init__(self, num=None):
                self.num = num

            # 确保不会使用快速路径计算乘法
            def __mul__(self, other):
                raise AssertionError('__mul__ should not be called')

            # 确保不会使用快速路径计算除法
            def __div__(self, other):
                raise AssertionError('__div__ should not be called')

            # 重载幂运算操作
            def __pow__(self, exp):
                return SomeClass(num=self.num ** exp)

            # 重载相等比较操作
            def __eq__(self, other):
                if isinstance(other, SomeClass):
                    return self.num == other.num

            # 反向幂运算的重载
            __rpow__ = __pow__

        # 定义一个函数 pow_for，用于在数组 arr 上计算 exp 次幂
        def pow_for(exp, arr):
            return np.array([x ** exp for x in arr])

        # 创建一个对象类型为 SomeClass 的数组 obj_arr
        obj_arr = np.array([SomeClass(1), SomeClass(2), SomeClass(3)])

        # 断言 obj_arr 的 0.5 次幂结果与 pow_for(0.5, obj_arr) 相等
        assert_equal(obj_arr ** 0.5, pow_for(0.5, obj_arr))
        # 断言 obj_arr 的 0 次幂结果与 pow_for(0, obj_arr) 相等
        assert_equal(obj_arr ** 0, pow_for(0, obj_arr))
        # 断言 obj_arr 的 1 次幂结果与 pow_for(1, obj_arr) 相等
        assert_equal(obj_arr ** 1, pow_for(1, obj_arr))
        # 断言 obj_arr 的 -1 次幂结果与 pow_for(-1, obj_arr) 相等
        assert_equal(obj_arr ** -1, pow_for(-1, obj_arr))
        # 断言 obj_arr 的 2 次幂结果与 pow_for(2, obj_arr) 相等
        assert_equal(obj_arr ** 2, pow_for(2, obj_arr))
    # 定义一个测试方法，用于测试数组的操作函数重载
    def test_pos_array_ufunc_override(self):
        # 定义一个继承自 np.ndarray 的类 A
        class A(np.ndarray):
            # 定义 __array_ufunc__ 方法，用于处理数组操作函数的重载
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                # 调用 ufunc 对应的方法，并将输入转换为 np.ndarray 后传递给它
                return getattr(ufunc, method)(*[i.view(np.ndarray) for
                                                i in inputs], **kwargs)
        # 创建一个字符串 'foo' 的 np.ndarray，然后转换为类 A 的实例 tst
        tst = np.array('foo').view(A)
        # 使用 assert_raises 函数验证 TypeError 是否会被抛出
        with assert_raises(TypeError):
            # 尝试对 tst 执行一元加操作，期望引发 TypeError
            +tst
class TestTemporaryElide:
    # elision is only triggered on relatively large arrays

    def test_extension_incref_elide(self):
        # test extension (e.g. cython) calling PyNumber_* slots without
        # increasing the reference counts
        #
        # def incref_elide(a):
        #    d = input.copy() # refcount 1
        #    return d, d + d # PyNumber_Add without increasing refcount
        from numpy._core._multiarray_tests import incref_elide
        # 创建一个长度为 100000 的全一数组
        d = np.ones(100000)
        # 调用 incref_elide 函数处理数组 d
        orig, res = incref_elide(d)
        # d + d，计算数组 d 的加法结果但不保存
        d + d
        # 检查返回的 orig 是否与原始数组 d 相等
        assert_array_equal(orig, d)
        # 检查返回的 res 是否等于数组 d 加上自身的结果
        assert_array_equal(res, d + d)

    def test_extension_incref_elide_stack(self):
        # scanning if the refcount == 1 object is on the python stack to check
        # that we are called directly from python is flawed as object may still
        # be above the stack pointer and we have no access to the top of it
        #
        # def incref_elide_l(d):
        #    return l[4] + l[4] # PyNumber_Add without increasing refcount
        from numpy._core._multiarray_tests import incref_elide_l
        # 创建一个包含 5 个元素的列表，最后一个元素是长度为 100000 的全一数组
        l = [1, 1, 1, 1, np.ones(100000)]
        # 调用 incref_elide_l 函数处理列表 l
        res = incref_elide_l(l)
        # 检查返回的 res 是否等于列表 l 中的最后一个数组加上自身的结果
        assert_array_equal(l[4], np.ones(100000))
        assert_array_equal(res, l[4] + l[4])

    def test_temporary_with_cast(self):
        # check that we don't elide into a temporary which would need casting
        # 创建一个长度为 200000 的全一数组，数据类型为 np.int64
        d = np.ones(200000, dtype=np.int64)
        # 执行 d + d，然后与一个超出 np.int64 范围的整数数组相加
        r = ((d + d) + np.array(2**222, dtype='O'))
        # 检查 r 的数据类型是否为对象型 'O'
        assert_equal(r.dtype, np.dtype('O'))

        # 执行 d + d 后再除以 2
        r = ((d + d) / 2)
        # 检查 r 的数据类型是否为双精度浮点型 'f8'
        assert_equal(r.dtype, np.dtype('f8'))

        # 使用 np.true_divide 执行 d + d 再除以 2
        r = np.true_divide((d + d), 2)
        # 检查 r 的数据类型是否为双精度浮点型 'f8'
        assert_equal(r.dtype, np.dtype('f8'))

        # 执行 d + d 后再除以浮点数 2.0
        r = ((d + d) / 2.)
        # 检查 r 的数据类型是否为双精度浮点型 'f8'
        assert_equal(r.dtype, np.dtype('f8'))

        # 执行 d + d 后再整除 2
        r = ((d + d) // 2)
        # 检查 r 的数据类型是否为 np.int64
        assert_equal(r.dtype, np.dtype(np.int64))

        # 对于浮点数数组 f，执行 f + f，再加上 f 转换为 np.float64 后的结果
        f = np.ones(100000, dtype=np.float32)
        # 检查结果的数据类型是否为双精度浮点型 'f8'
        assert_equal(((f + f) + f.astype(np.float64)).dtype, np.dtype('f8'))

        # 创建一个双精度浮点型数组 d，将 f 转换为双精度浮点型后执行 f + f 再加上 d 的结果
        d = f.astype(np.float64)
        # 检查结果的数据类型是否与 d 的数据类型相同
        assert_equal(((f + f) + d).dtype, d.dtype)
        
        # 创建一个长双精度浮点型数组 l，执行 d + d 后再加上 l 的结果
        l = np.ones(100000, dtype=np.longdouble)
        # 检查结果的数据类型是否与 l 的数据类型相同
        assert_equal(((d + d) + l).dtype, l.dtype)

        # 测试不同输出数据类型的绝对值运算
        for dt in (np.complex64, np.complex128, np.clongdouble):
            c = np.ones(100000, dtype=dt)
            # 计算 c * 2.0 的绝对值
            r = abs(c * 2.0)
            # 检查结果的数据类型是否为对应的浮点型
            assert_equal(r.dtype, np.dtype('f%d' % (c.itemsize // 2)))
    # 测试在广播到更高维度时不省略
    # 在调试模式下触发省略代码路径，普通模式需要256KB大的匹配维度，需要大量内存
    d = np.ones((2000, 1), dtype=int)
    b = np.ones((2000), dtype=bool)
    r = (1 - d) + b
    assert_equal(r, 1)
    assert_equal(r.shape, (2000, 2000))

    # 检查就地操作不会从标量创建 ndarray
    a = np.bool()
    assert_(type(~(a & a)) is np.bool)

    # 实部为实数组的虚部是只读的。这需要通过 fast_scalar_power 处理，
    # 仅对+1、-1、0、0.5和2的幂调用，因此使用2。同时，虚部对于实数组可以得到有效的引用计数，
    # 用于省略操作不应该出错。
    a = np.empty(100000, dtype=np.float64)
    a.imag ** 2

    # 不要尝试省略只读临时变量
    r = np.asarray(np.broadcast_to(np.zeros(1), 100000).flat) * 0.0
    assert_equal(r, 0)

    # 测试更新 if-copy 操作
    a = np.ones(2**20)[::2]
    b = a.flat.__array__() + 1
    del b
    assert_equal(a, 1)
# 定义一个名为 TestCAPI 的测试类
class TestCAPI:
    # 定义一个名为 test_IsPythonScalar 的测试方法
    def test_IsPythonScalar(self):
        # 从 numpy 库中导入 IsPythonScalar 函数，并断言传入字节字符串 b'foobar' 返回 True
        from numpy._core._multiarray_tests import IsPythonScalar
        assert_(IsPythonScalar(b'foobar'))
        # 断言传入整数 1 返回 True
        assert_(IsPythonScalar(1))
        # 断言传入大整数 2**80 返回 True
        assert_(IsPythonScalar(2**80))
        # 断言传入浮点数 2. 返回 True
        assert_(IsPythonScalar(2.))
        # 断言传入字符串 "a" 返回 True
        assert_(IsPythonScalar("a"))

    # 使用 pytest 的参数化装饰器，定义一个名为 test_intp_sequence_converters 的测试方法，
    # 参数 converter 分别传入 _multiarray_tests.run_scalar_intp_converter 和 _multiarray_tests.run_scalar_intp_from_sequence
    @pytest.mark.parametrize("converter",
             [_multiarray_tests.run_scalar_intp_converter,
              _multiarray_tests.run_scalar_intp_from_sequence])
    def test_intp_sequence_converters(self, converter):
        # 测试简单值 (对于错误返回路径，-1 是特殊的)
        assert converter(10) == (10,)
        assert converter(-1) == (-1,)
        # 0 维数组看起来有点像序列，但必须走整数路径：
        assert converter(np.array(123)) == (123,)
        # 测试简单序列 (intp_from_sequence 仅支持长度为 1 的序列)：
        assert converter((10,)) == (10,)
        assert converter(np.array([11])) == (11,)

    # 使用 pytest 的参数化装饰器，定义一个名为 test_intp_sequence_converters_errors 的测试方法，
    # 参数 converter 同样传入 _multiarray_tests.run_scalar_intp_converter 和 _multiarray_tests.run_scalar_intp_from_sequence
    # 同时使用 pytest 的 skipif 装饰器，如果 IS_PYPY 为真且 Python 版本小于等于 (7, 3, 8)，则跳过测试
    @pytest.mark.parametrize("converter",
             [_multiarray_tests.run_scalar_intp_converter,
              _multiarray_tests.run_scalar_intp_from_sequence])
    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
            reason="PyPy bug in error formatting")
    def test_intp_sequence_converters_errors(self, converter):
        # 使用 pytest 的 raises 断言，期望抛出 TypeError 异常，并匹配给定的错误消息
        with pytest.raises(TypeError,
                match="expected a sequence of integers or a single integer, "):
            converter(object())
        with pytest.raises(TypeError,
                match="expected a sequence of integers or a single integer, "
                      "got '32.0'"):
            converter(32.)
        with pytest.raises(TypeError,
                match="'float' object cannot be interpreted as an integer"):
            converter([32.])
        with pytest.raises(ValueError,
                match="Maximum allowed dimension"):
            # 这些转换器当前将溢出转换为 ValueError
            converter(2**64)


# 定义一个名为 TestSubscripting 的测试类
class TestSubscripting:
    # 定义一个名为 test_test_zero_rank 的测试方法
    def test_test_zero_rank(self):
        # 创建一个 numpy 数组 x
        x = np.array([1, 2, 3])
        # 使用 assert_ 断言，验证 x[0] 的类型为 np.int_
        assert_(isinstance(x[0], np.int_))
        # 使用 assert_ 断言，验证 x[0, ...] 的类型为 np.ndarray


# 定义一个名为 TestPickling 的测试类
class TestPickling:
    # 使用 pytest 的 skipif 装饰器，如果 pickle.HIGHEST_PROTOCOL >= 5，则跳过测试
    @pytest.mark.skipif(pickle.HIGHEST_PROTOCOL >= 5,
                        reason=('this tests the error messages when trying to'
                                'protocol 5 although it is not available'))
    # 定义一个名为 test_correct_protocol5_error_message 的测试方法
    def test_correct_protocol5_error_message(self):
        # 创建一个 numpy 数组 array，序列化为 pickle 格式，尝试使用协议 5 时测试错误消息
        array = np.arange(10)
    # 测试使用对象类型数据的记录数组
    def test_record_array_with_object_dtype(self):
        # 创建一个 Python 对象
        my_object = object()

        # 创建包含对象类型的记录数组
        arr_with_object = np.array(
                [(my_object, 1, 2.0)],
                dtype=[('a', object), ('b', int), ('c', float)])
        
        # 创建不包含对象类型的记录数组
        arr_without_object = np.array(
                [('xxx', 1, 2.0)],
                dtype=[('a', str), ('b', int), ('c', float)])

        # 遍历从 pickle 模块支持的协议版本 2 到最高协议版本的范围
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 使用 pickle 序列化和反序列化包含对象类型的记录数组
            depickled_arr_with_object = pickle.loads(
                    pickle.dumps(arr_with_object, protocol=proto))
            
            # 使用 pickle 序列化和反序列化不包含对象类型的记录数组
            depickled_arr_without_object = pickle.loads(
                    pickle.dumps(arr_without_object, protocol=proto))

            # 断言序列化和反序列化后的记录数组的 dtype 相同
            assert_equal(arr_with_object.dtype,
                         depickled_arr_with_object.dtype)
            assert_equal(arr_without_object.dtype,
                         depickled_arr_without_object.dtype)

    # 使用 pytest 标记跳过条件，如果 pickle 模块的最高协议版本小于 5
    @pytest.mark.skipif(pickle.HIGHEST_PROTOCOL < 5,
                        reason="requires pickle protocol 5")
    # 测试 Fortran 连续数组
    def test_f_contiguous_array(self):
        # 创建 Fortran 连续数组
        f_contiguous_array = np.array([[1, 2, 3], [4, 5, 6]], order='F')
        # 创建空列表用于存储缓冲区
        buffers = []

        # 当使用 pickle 协议版本 5 时，Fortran 连续数组可以使用 out-of-band 缓冲区进行序列化
        bytes_string = pickle.dumps(f_contiguous_array, protocol=5,
                                    buffer_callback=buffers.append)

        # 断言缓冲区列表不为空
        assert len(buffers) > 0

        # 使用指定的缓冲区反序列化 Fortran 连续数组
        depickled_f_contiguous_array = pickle.loads(bytes_string,
                                                    buffers=buffers)

        # 断言反序列化后的数组与原数组相等
        assert_equal(f_contiguous_array, depickled_f_contiguous_array)

    # 测试非连续数组的序列化和反序列化
    def test_non_contiguous_array(self):
        # 创建非连续数组
        non_contiguous_array = np.arange(12).reshape(3, 4)[:, :2]
        # 断言数组不是 C 连续的也不是 Fortran 连续的
        assert not non_contiguous_array.flags.c_contiguous
        assert not non_contiguous_array.flags.f_contiguous

        # 确保可以使用任何协议序列化和反序列化非连续数组
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 使用指定协议版本序列化和反序列化非连续数组
            depickled_non_contiguous_array = pickle.loads(
                    pickle.dumps(non_contiguous_array, protocol=proto))

            # 断言反序列化后的数组与原数组相等
            assert_equal(non_contiguous_array, depickled_non_contiguous_array)
    # 测试序列化和反序列化过程的回路
    def test_roundtrip(self):
        # 遍历从2到最高协议版本的pickle协议
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            # 创建一个包含不同类型numpy数组的列表
            carray = np.array([[2, 9], [7, 0], [3, 8]])
            DATA = [
                carray,  # 原始数组
                np.transpose(carray),  # 转置后的数组
                np.array([('xxx', 1, 2.0)], dtype=[('a', (str, 3)), ('b', int), ('c', float)])  # 具有自定义dtype的数组
            ]

            # 创建弱引用列表以跟踪DATA中每个对象的引用
            refs = [weakref.ref(a) for a in DATA]
            # 对每个数组进行序列化和反序列化，并进行相等性断言
            for a in DATA:
                assert_equal(
                        a, pickle.loads(pickle.dumps(a, protocol=proto)),
                        err_msg="%r" % a)
            # 清理变量以及DATA、carray引用
            del a, DATA, carray
            # 打破循环引用
            break_cycles()
            # 检查是否有引用泄漏 (gh-12793)
            for ref in refs:
                assert ref() is None

    # 使用指定编码'latin1'反序列化pickle对象
    def _loads(self, obj):
        return pickle.loads(obj, encoding='latin1')

    # 测试版本0的int8类型数据反序列化
    # 版本0的pickle没有版本字段
    def test_version0_int8(self):
        # 定义一个包含int8类型数据的字节序列
        s = b"\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x04\x85cnumpy\ndtype\nq\x04U\x02i1K\x00K\x01\x87Rq\x05(U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x04\x01\x02\x03\x04tb."  # noqa
        # 创建预期的int8类型numpy数组
        a = np.array([1, 2, 3, 4], dtype=np.int8)
        # 反序列化字节序列s并进行相等性断言
        p = self._loads(s)
        assert_equal(a, p)

    # 测试版本0的float32类型数据反序列化
    def test_version0_float32(self):
        # 定义一个包含float32类型数据的字节序列
        s = b"\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x04\x85cnumpy\ndtype\nq\x04U\x02f4K\x00K\x01\x87Rq\x05(U\x01<NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@tb."  # noqa
        # 创建预期的float32类型numpy数组
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        # 反序列化字节序列s并进行相等性断言
        p = self._loads(s)
        assert_equal(a, p)

    # 测试版本0的object类型数据反序列化
    def test_version0_object(self):
        # 定义一个包含object类型数据的字节序列
        s = b"\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x02\x85cnumpy\ndtype\nq\x04U\x02O8K\x00K\x01\x87Rq\x05(U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89]q\x06(}q\x07U\x01aK\x01s}q\x08U\x01bK\x02setb."  # noqa
        # 创建预期的object类型numpy数组
        a = np.array([{'a': 1}, {'b': 2}])
        # 反序列化字节序列s并进行相等性断言
        p = self._loads(s)
        assert_equal(a, p)

    # 测试版本1的int8类型数据反序列化
    def test_version1_int8(self):
        # 定义一个包含版本1 int8类型数据的字节序列
        s = b"\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x04\x85cnumpy\ndtype\nq\x04U\x02i1K\x00K\x01\x87Rq\x05(K\x01U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x04\x01\x02\x03\x04tb."  # noqa
        # 创建预期的int8类型numpy数组
        a = np.array([1, 2, 3, 4], dtype=np.int8)
        # 反序列化字节序列s并进行相等性断言
        p = self._loads(s)
        assert_equal(a, p)
    # 测试函数，用于测试加载 float32 版本的数据
    def test_version1_float32(self):
        # 示例数据序列化后的字节流表示
        s = b"\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x04\x85cnumpy\ndtype\nq\x04U\x02f4K\x00K\x01\x87Rq\x05(K\x01U\x01<NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@tb."  # noqa
        # 创建一个包含四个元素的 float32 类型的 numpy 数组
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        # 使用自定义方法 _loads 反序列化 s，并将结果赋给变量 p
        p = self._loads(s)
        # 断言两个数组 a 和 p 是否相等
        assert_equal(a, p)

    # 测试函数，用于测试加载 object 类型的数据
    def test_version1_object(self):
        # 示例数据序列化后的字节流表示
        s = b"\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x02\x85cnumpy\ndtype\nq\x04U\x02O8K\x00K\x01\x87Rq\x05(K\x01U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89]q\x06(}q\x07U\x01aK\x01s}q\x08U\x01bK\x02setb."  # noqa
        # 创建一个包含两个字典元素的 numpy 数组
        a = np.array([{'a': 1}, {'b': 2}])
        # 使用自定义方法 _loads 反序列化 s，并将结果赋给变量 p
        p = self._loads(s)
        # 断言两个数组 a 和 p 是否相等
        assert_equal(a, p)

    # 测试函数，用于测试包含子数组和结构化 dtype 的数据加载
    def test_subarray_int_shape(self):
        # 示例数据序列化后的字节流表示
        s = b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\n(I0\ntp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\np7\n(S'V6'\np8\nI0\nI1\ntp9\nRp10\n(I3\nS'|'\np11\nN(S'a'\np12\ng3\ntp13\n(dp14\ng12\n(g7\n(S'V4'\np15\nI0\nI1\ntp16\nRp17\n(I3\nS'|'\np18\n(g7\n(S'i1'\np19\nI0\nI1\ntp20\nRp21\n(I3\nS'|'\np22\nNNNI-1\nI-1\nI0\ntp23\nb(I2\nI2\ntp24\ntp25\nNNI4\nI1\nI0\ntp26\nbI0\ntp27\nsg3\n(g7\n(S'V2'\np28\nI0\nI1\ntp29\nRp30\n(I3\nS'|'\np31\n(g21\nI2\ntp32\nNNI2\nI1\nI0\ntp33\nbI4\ntp34\nsI6\nI1\nI0\ntp35\nbI00\nS'\\x01\\x01\\x01\\x01\\x01\\x02'\np36\ntp37\nb."  # noqa
        # 创建一个结构化的 numpy 数组，包含两个字段 'a' 和 'b'，每个字段有特定的 dtype 和形状
        a = np.array([(1, (1, 2))], dtype=[('a', 'i1', (2, 2)), ('b', 'i1', 2)])
        # 使用自定义方法 _loads 反序列化 s，并将结果赋给变量 p
        p = self._loads(s)
        # 断言两个数组 a 和 p 是否相等
        assert_equal(a, p)

    # 测试函数，用于测试 datetime64 类型数据的字节顺序处理
    def test_datetime64_byteorder(self):
        # 创建一个包含单个日期时间字符串的 datetime64 数组
        original = np.array([['2015-02-24T00:00:00.000000000']], dtype='datetime64[ns]')
        # 复制原数组，并调整其字节顺序为 'K'
        original_byte_reversed = original.copy(order='K')
        # 修改复制后数组的 dtype 为与原数组相同，但字节顺序为 'S'
        original_byte_reversed.dtype = original_byte_reversed.dtype.newbyteorder('S')
        # 在原处字节交换数组 original_byte_reversed
        original_byte_reversed.byteswap(inplace=True)
        # 使用 pickle 将交换后的数组序列化并反序列化，赋值给新数组 new
        new = pickle.loads(pickle.dumps(original_byte_reversed))
        # 断言原数组和新数组的 dtype 是否相等
        assert_equal(original.dtype, new.dtype)
class TestFancyIndexing:
    def test_list(self):
        # 创建一个形状为 (1, 1) 的全 1 数组
        x = np.ones((1, 1))
        # 使用 fancy indexing 将第一列的所有元素设为 2.0
        x[:, [0]] = 2.0
        # 断言数组 x 等于给定的数组 [[2.0]]
        assert_array_equal(x, np.array([[2.0]]))

        # 创建一个形状为 (1, 1, 1) 的全 1 数组
        x = np.ones((1, 1, 1))
        # 使用 fancy indexing 将第一列的所有元素设为 2.0
        x[:, :, [0]] = 2.0
        # 断言数组 x 等于给定的数组 [[[2.0]]]
        assert_array_equal(x, np.array([[[2.0]]]))

    def test_tuple(self):
        # 创建一个形状为 (1, 1) 的全 1 数组
        x = np.ones((1, 1))
        # 使用 fancy indexing 将第一列的所有元素设为 2.0
        x[:, (0,)] = 2.0
        # 断言数组 x 等于给定的数组 [[2.0]]
        assert_array_equal(x, np.array([[2.0]]))

        # 创建一个形状为 (1, 1, 1) 的全 1 数组
        x = np.ones((1, 1, 1))
        # 使用 fancy indexing 将第一列的所有元素设为 2.0
        x[:, :, (0,)] = 2.0
        # 断言数组 x 等于给定的数组 [[[2.0]]]
        assert_array_equal(x, np.array([[[2.0]]]))

    def test_mask(self):
        # 创建一个包含整数的一维数组
        x = np.array([1, 2, 3, 4])
        # 创建一个布尔类型的掩码数组
        m = np.array([0, 1, 0, 0], bool)
        # 断言使用掩码 m 取出的数组元素等于给定的数组 [2]
        assert_array_equal(x[m], np.array([2]))

    def test_mask2(self):
        # 创建一个包含整数的二维数组
        x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        # 创建一个布尔类型的掩码数组
        m = np.array([0, 1], bool)
        # 创建一个布尔类型的二维掩码数组
        m2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], bool)
        # 创建一个布尔类型的二维掩码数组
        m3 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool)
        # 断言使用掩码 m 取出的数组等于给定的二维数组 [[5, 6, 7, 8]]
        assert_array_equal(x[m], np.array([[5, 6, 7, 8]]))
        # 断言使用掩码 m2 取出的数组等于给定的一维数组 [2, 5]
        assert_array_equal(x[m2], np.array([2, 5]))
        # 断言使用掩码 m3 取出的数组等于给定的一维数组 [2]
        assert_array_equal(x[m3], np.array([2]))

    def test_assign_mask(self):
        # 创建一个包含整数的一维数组
        x = np.array([1, 2, 3, 4])
        # 创建一个布尔类型的掩码数组
        m = np.array([0, 1, 0, 0], bool)
        # 使用掩码 m 将数组 x 的符合条件的元素设为 5
        x[m] = 5
        # 断言数组 x 等于给定的数组 [1, 5, 3, 4]
        assert_array_equal(x, np.array([1, 5, 3, 4]))

    def test_assign_mask2(self):
        # 创建一个包含整数的二维数组
        xorig = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        # 创建一个布尔类型的掩码数组
        m = np.array([0, 1], bool)
        # 创建一个布尔类型的二维掩码数组
        m2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], bool)
        # 创建一个布尔类型的二维掩码数组
        m3 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool)
        
        # 复制原始数组 xorig 到 x
        x = xorig.copy()
        # 使用掩码 m 将数组 x 的符合条件的元素设为 10
        x[m] = 10
        # 断言数组 x 等于给定的二维数组 [[1, 2, 3, 4], [10, 10, 10, 10]]
        assert_array_equal(x, np.array([[1, 2, 3, 4], [10, 10, 10, 10]]))
        
        # 复制原始数组 xorig 到 x
        x = xorig.copy()
        # 使用掩码 m2 将数组 x 的符合条件的元素设为 10
        x[m2] = 10
        # 断言数组 x 等于给定的二维数组 [[1, 10, 3, 4], [10, 6, 7, 8]]
        assert_array_equal(x, np.array([[1, 10, 3, 4], [10, 6, 7, 8]]))
        
        # 复制原始数组 xorig 到 x
        x = xorig.copy()
        # 使用掩码 m3 将数组 x 的符合条件的元素设为 10
        x[m3] = 10
        # 断言数组 x 等于给定的二维数组 [[1, 10, 3, 4], [5, 6, 7, 8]]
        assert_array_equal(x, np.array([[1, 10, 3, 4], [5, 6, 7, 8]]))


class TestStringCompare:
    def test_string(self):
        # 创建一个字符串数组 g1
        g1 = np.array(["This", "is", "example"])
        # 创建一个字符串数组 g2
        g2 = np.array(["This", "was", "example"])
        # 断言 g1 与 g2 的每个元素是否相等，返回布尔数组
        assert_array_equal(g1 == g2, [g1[i] == g2[i] for i in [0, 1, 2]])
        # 断言 g1 与 g2 的每个元素是否不相等，返回布尔数组
        assert_array_equal(g1 != g2, [g1[i] != g2[i] for i in [0, 1, 2]])
        # 断言 g1 的每个元素是否小于等于 g2 的对应元素，返回布尔数组
        assert_array_equal(g1 <= g2, [g1[i] <= g2[i] for i in [0, 1, 2]])
        # 断言 g1 的每个元素是否大于等于 g2 的对应元素，返回布尔数组
        assert_array_equal(g1 >= g2, [g1[i] >= g2[i] for i in [0, 1, 2]])
        # 断言 g1 的每个元素是否小于 g2 的对应元素，返回布尔数组
        assert_array_equal(g1 < g2, [g1[i] < g2[i] for i in [0, 1, 2]])
        # 断言 g1 的每个元素是否大于 g2 的对应元素，返回布尔数组
        assert_array_equal(g1 > g2, [g1[i] > g2[i] for i in [0, 1, 2]])

    def test_mixed(self):
        # 创建一个字符串数组 g1
        g1 = np.array(["spam", "spa", "spammer", "and eggs"])
        # 创建一个字符串 g2
        g2 = "spam"
        # 断言 g
    # 定义一个测试方法，用于测试 Unicode 字符串数组的比较操作
    def test_unicode(self):
        # 创建两个 NumPy 数组，包含 Unicode 字符串
        g1 = np.array(["This", "is", "example"])
        g2 = np.array(["This", "was", "example"])
        # 断言 g1 和 g2 数组的相等性，检查每个元素的比较结果是否符合预期
        assert_array_equal(g1 == g2, [g1[i] == g2[i] for i in [0, 1, 2]])
        # 断言 g1 和 g2 数组的不等性
        assert_array_equal(g1 != g2, [g1[i] != g2[i] for i in [0, 1, 2]])
        # 断言 g1 数组元素小于等于 g2 数组元素
        assert_array_equal(g1 <= g2, [g1[i] <= g2[i] for i in [0, 1, 2]])
        # 断言 g1 数组元素大于等于 g2 数组元素
        assert_array_equal(g1 >= g2, [g1[i] >= g2[i] for i in [0, 1, 2]])
        # 断言 g1 数组元素小于 g2 数组元素
        assert_array_equal(g1 < g2,  [g1[i] < g2[i] for i in [0, 1, 2]])
        # 断言 g1 数组元素大于 g2 数组元素
        assert_array_equal(g1 > g2,  [g1[i] > g2[i] for i in [0, 1, 2]])
# 定义一个测试类 TestArgmaxArgminCommon
class TestArgmaxArgminCommon:

    # 定义不同的数组大小作为测试用例
    sizes = [(), (3,), (3, 2), (2, 3),
             (3, 3), (2, 3, 4), (4, 3, 2),
             (1, 2, 3, 4), (2, 3, 4, 1),
             (3, 4, 1, 2), (4, 1, 2, 3),
             (64,), (128,), (256,)]

    # 使用 pytest 的 parametrize 标记为每个测试用例提供不同的 size 和 axis 参数组合
    @pytest.mark.parametrize("size, axis", itertools.chain(*[[(size, axis)
        # 对于每个 size，生成其维度范围内的所有轴和 None 的组合
        for axis in list(range(-len(size), len(size))) + [None]]
        # 对于每个 size 在 sizes 列表中进行迭代
        for size in sizes]))
    # 为每个测试用例标记 parametrize，使用 np.argmax 和 np.argmin 两种方法作为测试方法
    @pytest.mark.parametrize('method', [np.argmax, np.argmin])
    # 定义一个测试函数，用于测试 numpy 的 argmin 和 argmax 方法在保留维度的情况下的行为
    def test_np_argmin_argmax_keepdims(self, size, axis, method):
        # 生成一个指定大小的正态分布随机数组
        arr = np.random.normal(size=size)

        # 对于连续的数组，确定新的形状以保留指定的轴
        if axis is None:
            new_shape = [1 for _ in range(len(size))]  # 创建一个所有维度为1的形状列表
        else:
            new_shape = list(size)
            new_shape[axis] = 1
        new_shape = tuple(new_shape)  # 将列表转换为元组，作为新的形状

        # 使用给定的方法计算原始结果
        _res_orig = method(arr, axis=axis)
        # 将原始结果重新塑形为新的形状
        res_orig = _res_orig.reshape(new_shape)
        # 使用方法计算保留维度的结果
        res = method(arr, axis=axis, keepdims=True)
        # 断言保留维度的结果与重新塑形的结果相等
        assert_equal(res, res_orig)
        # 断言结果的形状与新的形状相同
        assert_(res.shape == new_shape)
        # 创建一个与结果相同形状的空数组
        outarray = np.empty(res.shape, dtype=res.dtype)
        # 使用方法计算保留维度的结果并将其存储到预先创建的数组中
        res1 = method(arr, axis=axis, out=outarray, keepdims=True)
        # 断言返回的结果与预先创建的数组是同一个对象
        assert_(res1 is outarray)
        # 断言保留维度的结果与预先创建的数组相等
        assert_equal(res, outarray)

        # 如果数组的维度大于0，则执行以下操作
        if len(size) > 0:
            # 创建一个错误的形状列表，用于测试错误的输出数组形状
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            # 创建一个与错误形状相同的空数组
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            # 使用 pytest 检查在给定的条件下是否会引发 ValueError 异常
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)

        # 对于非连续的数组，确定新的形状以保留指定的轴
        if axis is None:
            new_shape = [1 for _ in range(len(size))]  # 创建一个所有维度为1的形状列表
        else:
            new_shape = list(size)[::-1]  # 将维度顺序反转
            new_shape[axis] = 1
        new_shape = tuple(new_shape)  # 将列表转换为元组，作为新的形状

        # 使用给定的方法计算原始结果（转置后的数组）
        _res_orig = method(arr.T, axis=axis)
        # 将原始结果重新塑形为新的形状
        res_orig = _res_orig.reshape(new_shape)
        # 使用方法计算保留维度的结果（转置后的数组）
        res = method(arr.T, axis=axis, keepdims=True)
        # 断言保留维度的结果与重新塑形的结果相等
        assert_equal(res, res_orig)
        # 断言结果的形状与新的形状相同
        assert_(res.shape == new_shape)
        # 创建一个与新形状的反向形状相同的空数组
        outarray = np.empty(new_shape[::-1], dtype=res.dtype)
        outarray = outarray.T  # 对数组进行转置
        # 使用方法计算保留维度的结果并将其存储到预先创建的数组中
        res1 = method(arr.T, axis=axis, out=outarray, keepdims=True)
        # 断言返回的结果与预先创建的数组是同一个对象
        assert_(res1 is outarray)
        # 断言保留维度的结果与预先创建的数组相等
        assert_equal(res, outarray)

        # 如果数组的维度大于0，则执行以下操作
        if len(size) > 0:
            # 创建一个错误的形状列表，用于测试错误的输出数组形状
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            # 创建一个与错误形状相同的空数组
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            # 使用 pytest 检查在给定的条件下是否会引发 ValueError 异常
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)
    # 定义测试函数，用于测试指定方法在不同情况下的行为
    def test_all(self, method):
        # 创建一个形状为 (4, 5, 6, 7, 8) 的正态分布随机数组
        a = np.random.normal(0, 1, (4, 5, 6, 7, 8))
        # 获取数组对象的指定方法，例如 'argmax' 或 'argmin'
        arg_method = getattr(a, 'arg' + method)
        val_method = getattr(a, method)
        # 遍历数组的每个维度
        for i in range(a.ndim):
            # 获取当前维度上的最大或最小值
            a_maxmin = val_method(i)
            # 获取当前维度上最大或最小值的索引
            aarg_maxmin = arg_method(i)
            # 创建一个排除当前维度后的轴列表
            axes = list(range(a.ndim))
            axes.remove(i)
            # 断言当前维度上的最大或最小值等于通过轴重新排列后的最大或最小值
            assert_(np.all(a_maxmin == aarg_maxmin.choose(
                                        *a.transpose(i, *axes))))

    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    # 参数化测试函数，测试指定方法的输出形状
    def test_output_shape(self, method):
        # 创建一个全为 1 的形状为 (10, 5) 的数组
        a = np.ones((10, 5))
        arg_method = getattr(a, method)
        # 检查一些简单的形状不匹配情况
        out = np.ones(11, dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        out = np.ones((2, 5), dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        # 这些情况可能会放宽（之前的版本允许甚至是前一个）
        out = np.ones((1, 10), dtype=np.int_)
        assert_raises(ValueError, arg_method, -1, out)

        out = np.ones(10, dtype=np.int_)
        # 在指定输出数组的情况下调用方法
        arg_method(-1, out=out)
        assert_equal(out, arg_method(-1))

    @pytest.mark.parametrize('ndim', [0, 1])
    @pytest.mark.parametrize('method', ['argmax', 'argmin'])
    # 参数化测试函数，测试返回结果是否为指定输出数组的情况
    def test_ret_is_out(self, ndim, method):
        # 创建一个全为 1 的数组，形状为 (4, 256, 256) 或 (4, 256)
        a = np.ones((4,) + (256,)*ndim)
        arg_method = getattr(a, method)
        # 创建一个空的指定形状的输出数组
        out = np.empty((256,)*ndim, dtype=np.intp)
        # 调用方法并断言返回的结果与指定的输出数组相同
        ret = arg_method(axis=0, out=out)
        assert ret is out

    @pytest.mark.parametrize('np_array, method, idx, val',
        [(np.zeros, 'argmax', 5942, "as"),
         (np.ones, 'argmin', 6001, "0")])
    # 参数化测试函数，测试对 Unicode 数据的行为
    def test_unicode(self, np_array, method, idx, val):
        # 创建一个指定长度和类型的 Unicode 数组
        d = np_array(6031, dtype='<U9')
        arg_method = getattr(d, method)
        # 修改指定索引位置的值
        d[idx] = val
        # 断言方法返回的结果与指定索引相同
        assert_equal(arg_method(), idx)

    @pytest.mark.parametrize('arr_method, np_method',
        [('argmax', np.argmax),
         ('argmin', np.argmin)])
    # 参数化测试函数，测试 ndarray 方法与 numpy 方法的行为一致性
    def test_np_vs_ndarray(self, arr_method, np_method):
        # 创建一个正态分布的随机数组
        a = np.random.normal(size=(2, 3))
        arg_method = getattr(a, arr_method)

        # 检查位置参数的情况
        out1 = np.zeros(2, dtype=int)
        out2 = np.zeros(2, dtype=int)
        assert_equal(arg_method(1, out1), np_method(a, 1, out2))
        assert_equal(out1, out2)

        # 检查关键字参数的情况
        out1 = np.zeros(3, dtype=int)
        out2 = np.zeros(3, dtype=int)
        assert_equal(arg_method(out=out1, axis=0),
                     np_method(a, out=out2, axis=0))
        assert_equal(out1, out2)

    @pytest.mark.leaks_references(reason="replaces None with NULL.")
    @pytest.mark.parametrize('method, vals',
        [('argmax', (10, 30)),
         ('argmin', (30, 10))])
    # 参数化测试函数，测试对不同方法和值的行为
    # 定义一个测试方法，用于测试包含空值的对象方法
    def test_object_with_NULLs(self, method, vals):
        # 在NumPy中创建一个空对象数组，长度为4，数据类型为对象型
        a = np.empty(4, dtype='O')
        # 获取对象数组a的指定方法（method）的引用
        arg_method = getattr(a, method)
        # 使用ctypes库的memset函数将对象数组a的内存区域清零
        ctypes.memset(a.ctypes.data, 0, a.nbytes)
        # 断言调用arg_method方法后返回值为0
        assert_equal(arg_method(), 0)
        # 将vals列表中的第一个值赋给对象数组a的第3个元素
        a[3] = vals[0]
        # 断言调用arg_method方法后返回值为3
        assert_equal(arg_method(), 3)
        # 将vals列表中的第二个值赋给对象数组a的第1个元素
        a[1] = vals[1]
        # 断言调用arg_method方法后返回值为1
        assert_equal(arg_method(), 1)
class TestArgmax:
    # 定义一个静态数据集，包含多组用例，每组用例由一个列表和一个预期结果组成
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 0),  # 示例用例1
        ([3, 3, 3, 3, 2, 2, 2, 2], 0),  # 示例用例2
        ([0, 1, 2, 3, 4, 5, 6, 7], 7),  # 示例用例3
        ([7, 6, 5, 4, 3, 2, 1, 0], 0)   # 示例用例4
    ]
    # 创建一个新的数据集 sg_data，包含之前的 usg_data 并额外添加两组新的用例
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 3),    # 新用例1
        ([1, 2, 3, 4, -1, -2, -3, -4], 3)     # 新用例2
    ]
    # 创建一个包含不同数据类型的数据数组 darr，使用 itertools.product 进行组合
    darr = [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product(usg_data, (
            np.uint8, np.uint16, np.uint32, np.uint64
        ))
    )]
    # 将 sg_data 与不同数据类型的组合添加到 darr 中
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product(sg_data, (
            np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
        ))
    )]
    # 将包含 NaN 值的数组与不同数据类型的组合添加到 darr 中
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product((
            ([0, 1, 2, 3, np.nan], 4),
            ([0, 1, 2, np.nan, 3], 3),
            ([np.nan, 0, 1, 2, 3], 0),
            ([np.nan, 0, np.nan, 2, 3], 0),
            # To hit the tail of SIMD multi-level(x4, x1) inner loops
            # on variant SIMD widthes
            ([1] * (2*5-1) + [np.nan], 2*5-1),
            ([1] * (4*5-1) + [np.nan], 4*5-1),
            ([1] * (8*5-1) + [np.nan], 8*5-1),
            ([1] * (16*5-1) + [np.nan], 16*5-1),
            ([1] * (32*5-1) + [np.nan], 32*5-1)
        ), (
            np.float32, np.float64
        ))
    )]
    nan_arr = darr + [
        ([0, 1, 2, 3, complex(0, np.nan)], 4),      # Add a tuple with a list and an integer 4 to nan_arr
        ([0, 1, 2, 3, complex(np.nan, 0)], 4),      # Add another tuple with a list and an integer 4 to nan_arr
        ([0, 1, 2, complex(np.nan, 0), 3], 3),      # Add another tuple with a list and an integer 3 to nan_arr
        ([0, 1, 2, complex(0, np.nan), 3], 3),      # Add another tuple with a list and an integer 3 to nan_arr
        ([complex(0, np.nan), 0, 1, 2, 3], 0),      # Add another tuple with a list and an integer 0 to nan_arr
        ([complex(np.nan, np.nan), 0, 1, 2, 3], 0), # Add another tuple with a list and an integer 0 to nan_arr
        ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0), # Add a tuple with a list and an integer 0 to nan_arr
        ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0), # Add a tuple with a list and an integer 0 to nan_arr
        ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0), # Add a tuple with a list and an integer 0 to nan_arr
    
        ([complex(0, 0), complex(0, 2), complex(0, 1)], 1), # Add a tuple with a list and an integer 1 to nan_arr
        ([complex(1, 0), complex(0, 2), complex(0, 1)], 0), # Add another tuple with a list and an integer 0 to nan_arr
        ([complex(1, 0), complex(0, 2), complex(1, 1)], 2), # Add another tuple with a list and an integer 2 to nan_arr
    
        ([np.datetime64('1923-04-14T12:43:12'),   # Add a tuple with a list of datetime64 values and an integer 5 to nan_arr
          np.datetime64('1994-06-21T14:43:15'),
          np.datetime64('2001-10-15T04:10:32'),
          np.datetime64('1995-11-25T16:02:16'),
          np.datetime64('2005-01-04T03:14:12'),
          np.datetime64('2041-12-03T14:05:03')], 5),
    
        ([np.datetime64('1935-09-14T04:40:11'),   # Add a tuple with a list of datetime64 values and an integer 3 to nan_arr
          np.datetime64('1949-10-12T12:32:11'),
          np.datetime64('2010-01-03T05:14:12'),
          np.datetime64('2015-11-20T12:20:59'),
          np.datetime64('1932-09-23T10:10:13'),
          np.datetime64('2014-10-10T03:50:30')], 3),
    
        # Assorted tests with NaTs
        ([np.datetime64('NaT'),                   # Add a tuple with a list containing NaT values and an integer 0 to nan_arr
          np.datetime64('NaT'),
          np.datetime64('2010-01-03T05:14:12'),
          np.datetime64('NaT'),
          np.datetime64('2015-09-23T10:10:13'),
          np.datetime64('1932-10-10T03:50:30')], 0),
    
        ([np.datetime64('2059-03-14T12:43:12'),   # Add a tuple with a list of datetime64 values and an integer 2 to nan_arr
          np.datetime64('1996-09-21T14:43:15'),
          np.datetime64('NaT'),
          np.datetime64('2022-12-25T16:02:16'),
          np.datetime64('1963-10-04T03:14:12'),
          np.datetime64('2013-05-08T18:15:23')], 2),
    
        ([np.timedelta64(2, 's'),                 # Add a tuple with a list of timedelta64 values and an integer 2 to nan_arr
          np.timedelta64(1, 's'),
          np.timedelta64('NaT', 's'),
          np.timedelta64(3, 's')], 2),
    
        ([np.timedelta64('NaT', 's')] * 3, 0),     # Add a tuple with a list containing three NaT timedelta64 values and an integer 0 to nan_arr
    
        ([timedelta(days=5, seconds=14),           # Add a tuple with a list of timedelta objects and an integer 0 to nan_arr
          timedelta(days=2, seconds=35),
          timedelta(days=-1, seconds=23)], 0),
    
        ([timedelta(days=1, seconds=43),           # Add a tuple with a list of timedelta objects and an integer 1 to nan_arr
          timedelta(days=10, seconds=5),
          timedelta(days=5, seconds=14)], 1),
    
        ([timedelta(days=10, seconds=24),          # Add a tuple with a list of timedelta objects and an integer 2 to nan_arr
          timedelta(days=10, seconds=5),
          timedelta(days=10, seconds=43)], 2),
    
        ([False, False, False, False, True], 4),    # Add a tuple with a list of boolean values and an integer 4 to nan_arr
        ([False, False, False, True, False], 3),    # Add another tuple with a list of boolean values and an integer 3 to nan_arr
        ([True, False, False, False, False], 0),    # Add another tuple with a list of boolean values and an integer 0 to nan_arr
        ([True, False, True, False, False], 0),     # Add another tuple with a list of boolean values and an integer 0 to nan_arr
    ]
    
    @pytest.mark.parametrize('data', nan_arr)      # Use pytest's parametrize to run tests with data from nan_arr
    # 定义一个测试函数，用于测试给定数据的各种组合情况
    def test_combinations(self, data):
        # 解包数据元组，分别获取数组和位置信息
        arr, pos = data
        # 使用 suppress_warnings 上下文管理器，过滤运行时警告
        with suppress_warnings() as sup:
            # 过滤特定的 RuntimeWarning，避免其影响到后续代码执行
            sup.filter(RuntimeWarning,
                        "invalid value encountered in reduce")
            # 计算数组 arr 的最大值
            val = np.max(arr)

        # 断言：数组 arr 中最大值的索引应与预期位置 pos 相等，否则输出错误信息
        assert_equal(np.argmax(arr), pos, err_msg="%r" % arr)
        # 断言：数组 arr 中最大值应与之前计算得到的 val 相等，否则输出错误信息
        assert_equal(arr[np.argmax(arr)], val, err_msg="%r" % arr)

        # 添加填充以测试 SIMD 循环效果
        # 将数组 arr 中的每个元素重复 129 次形成新数组 rarr
        rarr = np.repeat(arr, 129)
        # 计算新的预期位置 rpos
        rpos = pos * 129
        # 断言：新数组 rarr 中最大值的索引应与 rpos 相等，否则输出错误信息
        assert_equal(np.argmax(rarr), rpos, err_msg="%r" % rarr)
        # 断言：新数组 rarr 中最大值应与之前计算得到的 val 相等，否则输出错误信息
        assert_equal(rarr[np.argmax(rarr)], val, err_msg="%r" % rarr)

        # 创建填充数组 padd，其中每个元素都是数组 arr 的最小值，重复 513 次
        padd = np.repeat(np.min(arr), 513)
        # 将填充数组 padd 与数组 arr 连接起来形成新数组 rarr
        rarr = np.concatenate((arr, padd))
        # 设置新的预期位置 rpos 为原始位置 pos
        rpos = pos
        # 断言：新数组 rarr 中最大值的索引应与 rpos 相等，否则输出错误信息
        assert_equal(np.argmax(rarr), rpos, err_msg="%r" % rarr)
        # 断言：新数组 rarr 中最大值应与之前计算得到的 val 相等，否则输出错误信息
        assert_equal(rarr[np.argmax(rarr)], val, err_msg="%r" % rarr)


    # 定义测试最大有符号整数的函数
    def test_maximum_signed_integers(self):

        # 使用 np.array 创建一个包含有符号整数的数组 a，数据类型为 np.int8
        a = np.array([1, 2**7 - 1, -2**7], dtype=np.int8)
        # 断言：数组 a 中最大值的索引应为 1，即第二个元素，否则输出错误信息
        assert_equal(np.argmax(a), 1)
        # 将数组 a 中的每个元素重复 129 次形成新数组，并断言其最大值的索引为 129
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        # 使用 np.array 创建一个包含有符号整数的数组 a，数据类型为 np.int16
        a = np.array([1, 2**15 - 1, -2**15], dtype=np.int16)
        # 断言：数组 a 中最大值的索引应为 1，即第二个元素，否则输出错误信息
        assert_equal(np.argmax(a), 1)
        # 将数组 a 中的每个元素重复 129 次形成新数组，并断言其最大值的索引为 129
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        # 使用 np.array 创建一个包含有符号整数的数组 a，数据类型为 np.int32
        a = np.array([1, 2**31 - 1, -2**31], dtype=np.int32)
        # 断言：数组 a 中最大值的索引应为 1，即第二个元素，否则输出错误信息
        assert_equal(np.argmax(a), 1)
        # 将数组 a 中的每个元素重复 129 次形成新数组，并断言其最大值的索引为 129
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)

        # 使用 np.array 创建一个包含有符号整数的数组 a，数据类型为 np.int64
        a = np.array([1, 2**63 - 1, -2**63], dtype=np.int64)
        # 断言：数组 a 中最大值的索引应为 1，即第二个元素，否则输出错误信息
        assert_equal(np.argmax(a), 1)
        # 将数组 a 中的每个元素重复 129 次形成新数组，并断言其最大值的索引为 129
        a = a.repeat(129)
        assert_equal(np.argmax(a), 129)
# 定义一个测试类 TestArgmin
class TestArgmin:
    # 定义用于测试的数据集 usg_data，包含多个列表和对应的预期最小值索引
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 8),
        ([3, 3, 3, 3, 2, 2, 2, 2], 4),
        ([0, 1, 2, 3, 4, 5, 6, 7], 0),
        ([7, 6, 5, 4, 3, 2, 1, 0], 7)
    ]
    # 扩展 usg_data 并命名为 sg_data，添加额外的测试数据
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 4),
        ([1, 2, 3, 4, -1, -2, -3, -4], 7)
    ]
    # 使用 itertools.product 组合 usg_data 中的数据和多个数据类型，创建 numpy 数组及其预期最小值索引的元组
    darr = [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product(usg_data, (
            np.uint8, np.uint16, np.uint32, np.uint64
        ))
    )]
    # 继续扩展 darr，使用 itertools.product 组合 sg_data 中的数据和多个数据类型，创建 numpy 数组及其预期最小值索引的元组
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product(sg_data, (
            np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
        ))
    )]
    # 继续扩展 darr，使用 itertools.product 组合特定的数据集和数据类型，创建 numpy 数组及其预期最小值索引的元组
    darr = darr + [(np.array(d[0], dtype=t), d[1]) for d, t in (
        itertools.product((
            ([0, 1, 2, 3, np.nan], 4),
            ([0, 1, 2, np.nan, 3], 3),
            ([np.nan, 0, 1, 2, 3], 0),
            ([np.nan, 0, np.nan, 2, 3], 0),
            # To hit the tail of SIMD multi-level(x4, x1) inner loops
            # on variant SIMD widthes
            ([1] * (2*5-1) + [np.nan], 2*5-1),
            ([1] * (4*5-1) + [np.nan], 4*5-1),
            ([1] * (8*5-1) + [np.nan], 8*5-1),
            ([1] * (16*5-1) + [np.nan], 16*5-1),
            ([1] * (32*5-1) + [np.nan], 32*5-1)
        ), (
            np.float32, np.float64
        ))
    )]
    # 定义包含各种数据类型的列表 `nan_arr`
    nan_arr = darr + [
        # 第一个元组：包含复数和NaN，作为数据和期望值的元组
        ([0, 1, 2, 3, complex(0, np.nan)], 4),
        # 第二个元组：包含复数和NaN，作为数据和期望值的元组
        ([0, 1, 2, 3, complex(np.nan, 0)], 4),
        # 第三个元组：包含复数和NaN，作为数据和期望值的元组
        ([0, 1, 2, complex(np.nan, 0), 3], 3),
        # 第四个元组：包含复数和NaN，作为数据和期望值的元组
        ([0, 1, 2, complex(0, np.nan), 3], 3),
        # 第五个元组：包含复数和NaN，作为数据和期望值的元组
        ([complex(0, np.nan), 0, 1, 2, 3], 0),
        # 第六个元组：包含复数和NaN，作为数据和期望值的元组
        ([complex(np.nan, np.nan), 0, 1, 2, 3], 0),
        # 第七个元组：包含复数和NaN，作为数据和期望值的元组
        ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
        # 第八个元组：包含复数和NaN，作为数据和期望值的元组
        ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
        # 第九个元组：包含复数和NaN，作为数据和期望值的元组
        ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),
    
        # 以下元组包含复数，作为数据和期望值的元组
        ([complex(0, 0), complex(0, 2), complex(0, 1)], 0),
        ([complex(1, 0), complex(0, 2), complex(0, 1)], 2),
        ([complex(1, 0), complex(0, 2), complex(1, 1)], 1),
    
        # 以下元组包含日期时间对象，作为数据和期望值的元组
        ([np.datetime64('1923-04-14T12:43:12'),
          np.datetime64('1994-06-21T14:43:15'),
          np.datetime64('2001-10-15T04:10:32'),
          np.datetime64('1995-11-25T16:02:16'),
          np.datetime64('2005-01-04T03:14:12'),
          np.datetime64('2041-12-03T14:05:03')], 0),
        ([np.datetime64('1935-09-14T04:40:11'),
          np.datetime64('1949-10-12T12:32:11'),
          np.datetime64('2010-01-03T05:14:12'),
          np.datetime64('2014-11-20T12:20:59'),
          np.datetime64('2015-09-23T10:10:13'),
          np.datetime64('1932-10-10T03:50:30')], 5),
    
        # 包含NaT值的混合测试
        ([np.datetime64('NaT'),
          np.datetime64('NaT'),
          np.datetime64('2010-01-03T05:14:12'),
          np.datetime64('NaT'),
          np.datetime64('2015-09-23T10:10:13'),
          np.datetime64('1932-10-10T03:50:30')], 0),
        ([np.datetime64('2059-03-14T12:43:12'),
          np.datetime64('1996-09-21T14:43:15'),
          np.datetime64('NaT'),
          np.datetime64('2022-12-25T16:02:16'),
          np.datetime64('1963-10-04T03:14:12'),
          np.datetime64('2013-05-08T18:15:23')], 2),
    
        # 包含NaT值的时间间隔测试
        ([np.timedelta64(2, 's'),
          np.timedelta64(1, 's'),
          np.timedelta64('NaT', 's'),
          np.timedelta64(3, 's')], 2),
        # 全部为NaT值的时间间隔测试
        ([np.timedelta64('NaT', 's')] * 3, 0),
    
        # 时间差值的混合测试
        ([timedelta(days=5, seconds=14), timedelta(days=2, seconds=35),
          timedelta(days=-1, seconds=23)], 2),
        ([timedelta(days=1, seconds=43), timedelta(days=10, seconds=5),
          timedelta(days=5, seconds=14)], 0),
        ([timedelta(days=10, seconds=24), timedelta(days=10, seconds=5),
          timedelta(days=10, seconds=43)], 1),
    
        # 布尔值测试
        ([True, True, True, True, False], 4),
        ([True, True, True, False, True], 3),
        ([False, True, True, True, True], 0),
        ([False, True, False, True, True], 0),
    ]
    
    # 使用pytest的参数化测试，对`data`参数进行参数化
    @pytest.mark.parametrize('data', nan_arr)
    # 测试给定数据的各种组合情况
    def test_combinations(self, data):
        # 解包数据元组
        arr, pos = data
        # 使用 suppress_warnings 上下文管理器，过滤特定的运行时警告信息
        with suppress_warnings() as sup:
            # 过滤特定的 RuntimeWarning，这里是 "invalid value encountered in reduce"
            sup.filter(RuntimeWarning, "invalid value encountered in reduce")
            # 计算数组 arr 的最小值
            min_val = np.min(arr)

        # 断言：np.argmin(arr) 应该等于 pos，如果不等则输出错误信息 %r % arr
        assert_equal(np.argmin(arr), pos, err_msg="%r" % arr)
        # 断言：arr[np.argmin(arr)] 应该等于 min_val，如果不等则输出错误信息 %r % arr
        assert_equal(arr[np.argmin(arr)], min_val, err_msg="%r" % arr)

        # 添加填充以测试 SIMD 循环
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        # 断言：np.argmin(rarr) 应该等于 rpos，如果不等则输出错误信息 %r % rarr
        assert_equal(np.argmin(rarr), rpos, err_msg="%r" % rarr)
        # 断言：rarr[np.argmin(rarr)] 应该等于 min_val，如果不等则输出错误信息 %r % rarr
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg="%r" % rarr)

        # 创建填充数组 padd，以便测试 SIMD 循环
        padd = np.repeat(np.max(arr), 513)
        # 将 arr 和 padd 连接起来形成新的 rarr
        rarr = np.concatenate((arr, padd))
        rpos = pos
        # 断言：np.argmin(rarr) 应该等于 rpos，如果不等则输出错误信息 %r % rarr
        assert_equal(np.argmin(rarr), rpos, err_msg="%r" % rarr)
        # 断言：rarr[np.argmin(rarr)] 应该等于 min_val，如果不等则输出错误信息 %r % rarr
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg="%r" % rarr)

    # 测试不同数据类型的最小有符号整数
    def test_minimum_signed_integers(self):
        # 创建 np.int8 类型的数组 a
        a = np.array([1, -2**7, -2**7 + 1, 2**7 - 1], dtype=np.int8)
        # 断言：np.argmin(a) 应该等于 1
        assert_equal(np.argmin(a), 1)
        # 将数组 a 重复 129 次
        a = a.repeat(129)
        # 断言：np.argmin(a) 应该等于 129
        assert_equal(np.argmin(a), 129)

        # 创建 np.int16 类型的数组 a
        a = np.array([1, -2**15, -2**15 + 1, 2**15 - 1], dtype=np.int16)
        # 断言：np.argmin(a) 应该等于 1
        assert_equal(np.argmin(a), 1)
        # 将数组 a 重复 129 次
        a = a.repeat(129)
        # 断言：np.argmin(a) 应该等于 129
        assert_equal(np.argmin(a), 129)

        # 创建 np.int32 类型的数组 a
        a = np.array([1, -2**31, -2**31 + 1, 2**31 - 1], dtype=np.int32)
        # 断言：np.argmin(a) 应该等于 1
        assert_equal(np.argmin(a), 1)
        # 将数组 a 重复 129 次
        a = a.repeat(129)
        # 断言：np.argmin(a) 应该等于 129
        assert_equal(np.argmin(a), 129)

        # 创建 np.int64 类型的数组 a
        a = np.array([1, -2**63, -2**63 + 1, 2**63 - 1], dtype=np.int64)
        # 断言：np.argmin(a) 应该等于 1
        assert_equal(np.argmin(a), 1)
        # 将数组 a 重复 129 次
        a = a.repeat(129)
        # 断言：np.argmin(a) 应该等于 129
        assert_equal(np.argmin(a), 129)
class TestMinMax:
    
    # 测试针对标量的最大最小值计算是否引发 AxisError 异常
    def test_scalar(self):
        assert_raises(AxisError, np.amax, 1, 1)
        assert_raises(AxisError, np.amin, 1, 1)
        
        # 验证对标量 1 沿着 axis=0 的最大值计算结果为 1
        assert_equal(np.amax(1, axis=0), 1)
        # 验证对标量 1 沿着 axis=0 的最小值计算结果为 1
        assert_equal(np.amin(1, axis=0), 1)
        # 验证对标量 1 没有指定轴的最大值计算结果为 1
        assert_equal(np.amax(1, axis=None), 1)
        # 验证对标量 1 没有指定轴的最小值计算结果为 1
        assert_equal(np.amin(1, axis=None), 1)

    # 测试针对轴参数的最大最小值计算是否引发 AxisError 异常
    def test_axis(self):
        assert_raises(AxisError, np.amax, [1, 2, 3], 1000)
        # 验证对二维数组 [[1, 2, 3]] 沿着 axis=1 的最大值计算结果为 3
        assert_equal(np.amax([[1, 2, 3]], axis=1), 3)

    # 测试针对日期时间类型的最大最小值计算
    def test_datetime(self):
        # 不忽略 NaT（不是时间的时间）
        for dtype in ('m8[s]', 'm8[Y]'):
            a = np.arange(10).astype(dtype)
            # 验证对日期时间数组的最小值计算
            assert_equal(np.amin(a), a[0])
            # 验证对日期时间数组的最大值计算
            assert_equal(np.amax(a), a[9])
            a[3] = 'NaT'
            # 验证包含 NaT 的日期时间数组的最小值计算
            assert_equal(np.amin(a), a[3])
            # 验证包含 NaT 的日期时间数组的最大值计算
            assert_equal(np.amax(a), a[3])


class TestNewaxis:
    
    # 测试基本的 newaxis 功能
    def test_basic(self):
        sk = np.array([0, -0.1, 0.1])
        # 对 sk 数组的每个元素应用 newaxis 扩展
        res = 250*sk[:, np.newaxis]
        # 验证扩展后的结果是否与原数组元素乘以 250 的结果一致
        assert_almost_equal(res.ravel(), 250*sk)


class TestClip:
    
    # 检查数组 x 是否在给定范围内
    def _check_range(self, x, cmin, cmax):
        assert_(np.all(x >= cmin))
        assert_(np.all(x <= cmax))

    # 对指定类型组的数据进行剪切操作，并验证结果
    def _clip_type(self, type_group, array_max,
                   clip_min, clip_max, inplace=False,
                   expected_min=None, expected_max=None):
        if expected_min is None:
            expected_min = clip_min
        if expected_max is None:
            expected_max = clip_max

        for T in np._core.sctypes[type_group]:
            if sys.byteorder == 'little':
                byte_orders = ['=', '>']
            else:
                byte_orders = ['<', '=']

            for byteorder in byte_orders:
                dtype = np.dtype(T).newbyteorder(byteorder)

                # 生成随机数组，类型为 dtype，并进行剪切操作
                x = (np.random.random(1000) * array_max).astype(dtype)
                if inplace:
                    # 调用 clip 方法进行原地剪切操作，需要显式传递 casting='unsafe' 避免警告
                    x.clip(clip_min, clip_max, x, casting='unsafe')
                else:
                    # 调用 clip 方法进行剪切操作
                    x = x.clip(clip_min, clip_max)
                    byteorder = '='

                # 处理不同字节顺序的情况
                if x.dtype.byteorder == '|':
                    byteorder = '|'
                assert_equal(x.dtype.byteorder, byteorder)
                # 验证剪切后数组的数值范围是否符合预期
                self._check_range(x, expected_min, expected_max)
        return x
    def test_basic(self):
        # 测试基本情况下的数值裁剪操作
        for inplace in [False, True]:
            # 对 'float' 类型进行裁剪操作
            self._clip_type('float', 1024, -12.8, 100.2, inplace=inplace)
            self._clip_type('float', 1024, 0, 0, inplace=inplace)

            # 对 'int' 类型进行裁剪操作
            self._clip_type('int', 1024, -120, 100, inplace=inplace)
            self._clip_type('int', 1024, 0, 0, inplace=inplace)

            # 对 'uint' 类型进行裁剪操作
            self._clip_type('uint', 1024, 0, 0, inplace=inplace)
            self._clip_type('uint', 1024, 10, 100, inplace=inplace)

    @pytest.mark.parametrize("inplace", [False, True])
    def test_int_range_error(self, inplace):
        # 测试整数裁剪操作中的范围错误情况
        # 尝试使用负数裁剪无符号整数将引发溢出错误
        # （随着 NEP 50 的改变可能会适应）
        # 类似于 `test_basic` 中的最后一个检查
        x = (np.random.random(1000) * 255).astype("uint8")
        with pytest.raises(OverflowError):
            x.clip(-1, 10, out=x if inplace else None)

        with pytest.raises(OverflowError):
            x.clip(0, 256, out=x if inplace else None)

    def test_record_array(self):
        # 测试记录数组的裁剪操作
        rec = np.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                       dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
        y = rec['x'].clip(-0.3, 0.5)
        self._check_range(y, -0.3, 0.5)

    def test_max_or_min(self):
        # 测试使用最大值或最小值进行裁剪操作
        val = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        x = val.clip(3)
        assert_(np.all(x >= 3))
        x = val.clip(min=3)
        assert_(np.all(x >= 3))
        x = val.clip(max=4)
        assert_(np.all(x <= 4))

    def test_nan(self):
        # 测试处理 NaN 值的裁剪操作
        input_arr = np.array([-2., np.nan, 0.5, 3., 0.25, np.nan])
        result = input_arr.clip(-1, 1)
        expected = np.array([-1., np.nan, 0.5, 1., 0.25, np.nan])
        assert_array_equal(result, expected)
class TestCompress:
    def test_axis(self):
        tgt = [[5, 6, 7, 8, 9]]
        arr = np.arange(10).reshape(2, 5)
        # 使用 np.compress 函数按指定轴压缩数组
        out = np.compress([0, 1], arr, axis=0)
        assert_equal(out, tgt)

        tgt = [[1, 3], [6, 8]]
        # 再次使用 np.compress 函数，这次按另一轴压缩数组
        out = np.compress([0, 1, 0, 1, 0], arr, axis=1)
        assert_equal(out, tgt)

    def test_truncate(self):
        tgt = [[1], [6]]
        arr = np.arange(10).reshape(2, 5)
        # 对数组进行轴向压缩，以删除部分元素
        out = np.compress([0, 1], arr, axis=1)
        assert_equal(out, tgt)

    def test_flatten(self):
        arr = np.arange(10).reshape(2, 5)
        # 在未指定轴的情况下，使用 np.compress 进行数组压缩，相当于展平数组
        out = np.compress([0, 1], arr)
        assert_equal(out, 1)


class TestPutmask:
    def tst_basic(self, x, T, mask, val):
        # 使用 np.putmask 函数根据条件(mask)向数组(x)中放置值(val)
        np.putmask(x, mask, val)
        # 断言被放置的值与预期值数组匹配
        assert_equal(x[mask], np.array(val, T))

    def test_ip_types(self):
        unchecked_types = [bytes, str, np.void]

        x = np.random.random(1000)*100
        mask = x < 40

        for val in [-100, 0, 15]:
            for types in np._core.sctypes.values():
                for T in types:
                    if T not in unchecked_types:
                        if val < 0 and np.dtype(T).kind == "u":
                            val = np.iinfo(T).max - 99
                        # 调用 tst_basic 方法测试不同数据类型的情况
                        self.tst_basic(x.copy().astype(T), T, mask, val)

            # 还测试一个长度不典型的字符串
            dt = np.dtype("S3")
            self.tst_basic(x.astype(dt), dt.type, mask, dt.type(val)[:3])

    def test_mask_size(self):
        # 断言当 mask 的大小与数组不匹配时，会引发 ValueError
        assert_raises(ValueError, np.putmask, np.array([1, 2, 3]), [True], 5)

    @pytest.mark.parametrize('dtype', ('>i4', '<i4'))
    def test_byteorder(self, dtype):
        x = np.array([1, 2, 3], dtype)
        # 测试不同字节顺序的情况下，np.putmask 的行为
        np.putmask(x, [True, False, True], -1)
        assert_array_equal(x, [-1, 2, -1])

    def test_record_array(self):
        # 注意混合字节顺序的记录数组的情况
        rec = np.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                      dtype=[('x', '<f8'), ('y', '>f8'), ('z', '<f8')])
        # 对记录数组的一个字段应用 putmask
        np.putmask(rec['x'], [True, False], 10)
        assert_array_equal(rec['x'], [10, 5])
        assert_array_equal(rec['y'], [2, 4])
        assert_array_equal(rec['z'], [3, 3])
        np.putmask(rec['y'], [True, False], 11)
        assert_array_equal(rec['x'], [10, 5])
        assert_array_equal(rec['y'], [11, 4])
        assert_array_equal(rec['z'], [3, 3])

    def test_overlaps(self):
        # 检查 putmask 在重叠区域的行为
        x = np.array([True, False, True, False])
        np.putmask(x[1:4], [True, True, True], x[:3])
        assert_equal(x, np.array([True, True, False, True]))

        x = np.array([True, False, True, False])
        np.putmask(x[1:4], x[:3], [True, False, True])
        assert_equal(x, np.array([True, True, True, True]))

    def test_writeable(self):
        a = np.arange(5)
        a.flags.writeable = False

        # 断言当数组不可写时，调用 putmask 会引发 ValueError
        with pytest.raises(ValueError):
            np.putmask(a, a >= 2, 3)
    # 定义测试方法 test_kwargs
    def test_kwargs(self):
        # 创建一个包含两个元素的 NumPy 数组 x，元素值为 [0, 0]
        x = np.array([0, 0])
        # 使用 np.putmask 函数，将 x 数组中索引为 [0, 1] 的位置分别赋值为 [-1, -2]
        np.putmask(x, [0, 1], [-1, -2])
        # 断言 x 数组是否与期望结果 [0, -2] 相等
        assert_array_equal(x, [0, -2])

        # 创建一个新的数组 x，元素值同样为 [0, 0]
        x = np.array([0, 0])
        # 使用 np.putmask 函数，通过关键字参数 mask 和 values 将 x 数组中索引为 [0, 1] 的位置分别赋值为 [-1, -2]
        np.putmask(x, mask=[0, 1], values=[-1, -2])
        # 断言 x 数组是否与期望结果 [0, -2] 相等
        assert_array_equal(x, [0, -2])

        # 创建一个新的数组 x，元素值同样为 [0, 0]
        x = np.array([0, 0])
        # 使用 np.putmask 函数，通过关键字参数 values 和 mask 将 x 数组中索引为 [0, 1] 的位置分别赋值为 [-1, -2]
        np.putmask(x, values=[-1, -2],  mask=[0, 1])
        # 断言 x 数组是否与期望结果 [0, -2] 相等
        assert_array_equal(x, [0, -2])

        # 使用 pytest 的 raises 方法检测是否抛出 TypeError 异常
        with pytest.raises(TypeError):
            # 调用 np.putmask 函数时，使用错误的关键字参数 a，预期会抛出 TypeError 异常
            np.putmask(a=x, values=[-1, -2],  mask=[0, 1])
class TestTake:
    # 测试基本的 take 操作，确保从数组中取出所有元素时结果不变
    def tst_basic(self, x):
        # 创建一个索引列表，覆盖整个数组的所有行
        ind = list(range(x.shape[0]))
        # 断言取出的数组与原数组相等
        assert_array_equal(x.take(ind, axis=0), x)

    # 测试不同数据类型的输入
    def test_ip_types(self):
        # 未经检查的数据类型列表
        unchecked_types = [bytes, str, np.void]

        # 创建一个随机数组，并设置其形状为 2x3x4
        x = np.random.random(24) * 100
        x.shape = 2, 3, 4
        
        # 遍历 NumPy 支持的所有数据类型
        for types in np._core.sctypes.values():
            for T in types:
                # 排除掉未经检查的数据类型
                if T not in unchecked_types:
                    # 对复制后转换为当前类型的数组执行基本的测试
                    self.tst_basic(x.copy().astype(T))

            # 对转换为字符串类型的数组执行基本的测试
            self.tst_basic(x.astype("S3"))

    # 测试索引超出范围时是否正确抛出 IndexError
    def test_raise(self):
        # 创建一个随机数组，并设置其形状为 2x3x4
        x = np.random.random(24) * 100
        x.shape = 2, 3, 4
        
        # 断言对超出范围的索引执行 take 操作会引发 IndexError
        assert_raises(IndexError, x.take, [0, 1, 2], axis=0)
        assert_raises(IndexError, x.take, [-3], axis=0)
        # 断言 take 操作对于负索引能正确返回数组中对应的元素
        assert_array_equal(x.take([-1], axis=0)[0], x[1])

    # 测试使用 'clip' 模式时的 take 操作
    def test_clip(self):
        # 创建一个随机数组，并设置其形状为 2x3x4
        x = np.random.random(24) * 100
        x.shape = 2, 3, 4
        
        # 断言使用 'clip' 模式时对索引为 -1 的取值
        assert_array_equal(x.take([-1], axis=0, mode='clip')[0], x[0])
        # 断言使用 'clip' 模式时对索引为 2 的取值
        assert_array_equal(x.take([2], axis=0, mode='clip')[0], x[1])

    # 测试使用 'wrap' 模式时的 take 操作
    def test_wrap(self):
        # 创建一个随机数组，并设置其形状为 2x3x4
        x = np.random.random(24) * 100
        x.shape = 2, 3, 4
        
        # 断言使用 'wrap' 模式时对索引为 -1 的取值
        assert_array_equal(x.take([-1], axis=0, mode='wrap')[0], x[1])
        # 断言使用 'wrap' 模式时对索引为 2 和 3 的取值
        assert_array_equal(x.take([2], axis=0, mode='wrap')[0], x[0])
        assert_array_equal(x.take([3], axis=0, mode='wrap')[0], x[1])

    # 使用 pytest 的参数化装饰器，测试不同字节顺序的 take 操作
    @pytest.mark.parametrize('dtype', ('>i4', '<i4'))
    def test_byteorder(self, dtype):
        # 创建一个整数数组，并指定其字节顺序
        x = np.array([1, 2, 3], dtype)
        # 断言对该数组执行 take 操作后得到的结果与预期一致
        assert_array_equal(x.take([0, 2, 1]), [1, 3, 2])

    # 测试记录数组的 take 操作
    def test_record_array(self):
        # 创建一个记录数组，其中字段具有混合的字节顺序
        rec = np.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                      dtype=[('x', '<f8'), ('y', '>f8'), ('z', '<f8')])
        # 对记录数组执行 take 操作，断言取出的字段值与预期一致
        rec1 = rec.take([1])
        assert_(rec1['x'] == 5.0 and rec1['y'] == 4.0)

    # 测试在 out 参数有重叠时的 take 操作
    def test_out_overlap(self):
        # 创建一个整数数组
        x = np.arange(5)
        # 对数组执行 take 操作，使用 'wrap' 模式，并将结果写入到重叠的 out 参数中
        y = np.take(x, [1, 2, 3], out=x[2:5], mode='wrap')
        # 断言 take 操作的结果与预期一致
        assert_equal(y, np.array([1, 2, 3]))

    # 使用 pytest 的参数化装饰器，测试不同形状的索引数组的 take 操作
    @pytest.mark.parametrize('shape', [(1, 2), (1,), ()])
    def test_ret_is_out(self, shape):
        # 创建一个整数数组
        x = np.arange(5)
        # 创建一个指定形状的索引数组，并创建一个与 x 相同类型的 out 数组
        inds = np.zeros(shape, dtype=np.intp)
        out = np.zeros(shape, dtype=x.dtype)
        # 对数组执行 take 操作，断言返回的结果对象与 out 数组是同一个对象
        ret = np.take(x, inds, out=out)
        assert ret is out


class TestLexsort:
    # 使用 pytest 的参数化装饰器，测试基本的 lexsort 操作
    @pytest.mark.parametrize('dtype',[
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.int8, np.int16, np.int32, np.int64,
        np.float16, np.float32, np.float64
    ])
    def test_basic(self, dtype):
        # 创建两个数组，分别作为 lexsort 操作的输入
        a = np.array([1, 2, 1, 3, 1, 5], dtype=dtype)
        b = np.array([0, 4, 5, 6, 2, 3], dtype=dtype)
        # 执行 lexsort 操作，并断言结果与预期一致
        idx = np.lexsort((b, a))
        expected_idx = np.array([0, 4, 2, 1, 3, 5])
        assert_array_equal(idx, expected_idx)
        # 断言对排序后的数组执行基本的排序操作，结果与预期一致
        assert_array_equal(a[idx], np.sort(a))
    def test_mixed(self):
        # 创建包含整数的 NumPy 数组
        a = np.array([1, 2, 1, 3, 1, 5])
        # 创建包含日期的 NumPy 数组，使用日期类型 'datetime64[D]'
        b = np.array([0, 4, 5, 6, 2, 3], dtype='datetime64[D]')

        # 使用 np.lexsort 对数组 a 和 b 进行字典序排序，返回排序后的索引
        idx = np.lexsort((b, a))
        # 预期的排序后的索引数组
        expected_idx = np.array([0, 4, 2, 1, 3, 5])
        # 断言排序后的索引数组与预期的索引数组相等
        assert_array_equal(idx, expected_idx)

    def test_datetime(self):
        # 创建包含日期的 NumPy 数组，使用日期类型 'datetime64[D]'
        a = np.array([0,0,0], dtype='datetime64[D]')
        b = np.array([2,1,0], dtype='datetime64[D]')
        
        # 使用 np.lexsort 对数组 a 和 b 进行字典序排序，返回排序后的索引
        idx = np.lexsort((b, a))
        # 预期的排序后的索引数组
        expected_idx = np.array([2, 1, 0])
        # 断言排序后的索引数组与预期的索引数组相等
        assert_array_equal(idx, expected_idx)

        # 创建包含时间间隔的 NumPy 数组，使用时间间隔类型 'timedelta64[D]'
        a = np.array([0,0,0], dtype='timedelta64[D]')
        b = np.array([2,1,0], dtype='timedelta64[D]')
        
        # 使用 np.lexsort 对数组 a 和 b 进行字典序排序，返回排序后的索引
        idx = np.lexsort((b, a))
        # 预期的排序后的索引数组
        expected_idx = np.array([2, 1, 0])
        # 断言排序后的索引数组与预期的索引数组相等
        assert_array_equal(idx, expected_idx)

    def test_object(self):  # gh-6312
        # 创建包含随机整数的 NumPy 数组
        a = np.random.choice(10, 1000)
        # 创建包含随机字符串的 NumPy 数组
        b = np.random.choice(['abc', 'xy', 'wz', 'efghi', 'qwst', 'x'], 1000)

        # 对 a 和 b 中的每一个数组进行排序，并进行断言验证
        for u in a, b:
            # 使用 np.lexsort 对数组 u 进行字典序排序，返回排序后的索引
            left = np.lexsort((u.astype('O'),))
            # 使用 np.argsort 对数组 u 进行稳定排序，返回排序后的索引
            right = np.argsort(u, kind='mergesort')
            # 断言 np.lexsort 和 np.argsort 得到的索引数组相等
            assert_array_equal(left, right)

        # 对 (a, b) 和 (b, a) 这两组数组进行排序，并进行断言验证
        for u, v in (a, b), (b, a):
            # 使用 np.lexsort 对数组 u 和 v 进行字典序排序，返回排序后的索引
            idx = np.lexsort((u, v))
            # 断言 np.lexsort 对数组 u 和 v 的排序结果与将它们转换为对象数组后的排序结果相等
            assert_array_equal(idx, np.lexsort((u.astype('O'), v)))
            assert_array_equal(idx, np.lexsort((u, v.astype('O'))))
            u, v = np.array(u, dtype='object'), np.array(v, dtype='object')
            # 断言 np.lexsort 对对象数组 u 和 v 的排序结果与之前的排序结果相等
            assert_array_equal(idx, np.lexsort((u, v)))

    def test_invalid_axis(self): # gh-7528
        # 创建一个形状为 (42, 3) 的 NumPy 数组，其中包含等间隔的浮点数
        x = np.linspace(0., 1., 42*3).reshape(42, 3)
        # 断言在使用 np.lexsort 时会引发 AxisError 异常
        assert_raises(AxisError, np.lexsort, x, axis=2)
# 定义一个测试类 TestIO，用于测试文件的输入输出操作
class TestIO:
    """Test tofile, fromfile, tobytes, and fromstring"""

    # 定义测试用例的 fixture x，返回一个随机生成的复数数组
    @pytest.fixture()
    def x(self):
        shape = (2, 4, 3)
        rand = np.random.random
        x = rand(shape) + rand(shape).astype(complex) * 1j
        x[0, :, 1] = [np.nan, np.inf, -np.inf, np.nan]
        return x

    # 定义测试用例的 fixture tmp_filename，返回一个临时文件名（可以是字符串或路径对象）
    @pytest.fixture(params=["string", "path_obj"])
    def tmp_filename(self, tmp_path, request):
        # 这个 fixture 覆盖了两种情况：
        # 一种是文件名为字符串，另一种是路径对象
        filename = tmp_path / "file"
        if request.param == "string":
            filename = str(filename)
        yield filename

    # 测试当文件不存在时的情况，应该引发适当的错误
    def test_nofile(self):
        # 创建一个 BytesIO 对象 b
        b = io.BytesIO()
        # 当从空的 BytesIO 对象中读取数据时，应该引发 OSError 错误
        assert_raises(OSError, np.fromfile, b, np.uint8, 80)
        # 创建一个包含全为 1 的数组 d
        d = np.ones(7)
        # 当尝试将数组 d 写入到 b 中时，应该引发 OSError 错误
        assert_raises(OSError, lambda x: x.tofile(b), d)

    # 测试从字符串中读取布尔值数组的情况
    def test_bool_fromstring(self):
        # 创建一个预期的布尔值数组 v
        v = np.array([True, False, True, False], dtype=np.bool)
        # 使用 fromstring 方法从字符串中解析出布尔值数组 y
        y = np.fromstring('1 0 -2.3 0.0', sep=' ', dtype=np.bool)
        assert_array_equal(v, y)

    # 测试从字符串中读取 uint64 数据类型数组的情况
    def test_uint64_fromstring(self):
        # 使用 fromstring 方法从字符串中解析出 uint64 类型的数组 d
        d = np.fromstring("9923372036854775807 104783749223640",
                          dtype=np.uint64, sep=' ')
        # 创建预期的 uint64 类型数组 e
        e = np.array([9923372036854775807, 104783749223640], dtype=np.uint64)
        assert_array_equal(d, e)

    # 测试从字符串中读取 int64 数据类型数组的情况
    def test_int64_fromstring(self):
        # 使用 fromstring 方法从字符串中解析出 int64 类型的数组 d
        d = np.fromstring("-25041670086757 104783749223640",
                          dtype=np.int64, sep=' ')
        # 创建预期的 int64 类型数组 e
        e = np.array([-25041670086757, 104783749223640], dtype=np.int64)
        assert_array_equal(d, e)

    # 测试从字符串中读取数据，并指定 count 参数为 0 的情况
    def test_fromstring_count0(self):
        # 使用 fromstring 方法从字符串中解析出 int64 类型的数组 d，并且指定 count 参数为 0
        d = np.fromstring("1,2", sep=",", dtype=np.int64, count=0)
        # 断言数组 d 的形状应为 (0,)
        assert d.shape == (0,)

    # 测试创建空文本文件的情况
    def test_empty_files_text(self, tmp_filename):
        # 使用 'w' 模式打开临时文件 tmp_filename
        with open(tmp_filename, 'w') as f:
            pass
        # 从文本文件 tmp_filename 中读取数据到数组 y
        y = np.fromfile(tmp_filename)
        # 断言数组 y 的大小应为 0，即文件为空
        assert_(y.size == 0, "Array not empty")

    # 测试创建空二进制文件的情况
    def test_empty_files_binary(self, tmp_filename):
        # 使用 'wb' 模式打开临时文件 tmp_filename
        with open(tmp_filename, 'wb') as f:
            pass
        # 从二进制文件 tmp_filename 中读取数据到数组 y，使用默认分隔符
        y = np.fromfile(tmp_filename, sep=" ")
        # 断言数组 y 的大小应为 0，即文件为空
        assert_(y.size == 0, "Array not empty")

    # 测试将数组写入文件并从文件中读取的往返操作
    def test_roundtrip_file(self, x, tmp_filename):
        # 使用 'wb' 模式打开临时文件 tmp_filename
        with open(tmp_filename, 'wb') as f:
            # 将数组 x 写入到文件 f
            x.tofile(f)
        # 使用 'rb' 模式打开临时文件 tmp_filename
        with open(tmp_filename, 'rb') as f:
            # 从文件 f 中读取数据到数组 y，数据类型为 x 的数据类型
            y = np.fromfile(f, dtype=x.dtype)
        # 断言数组 y 与原始数组 x 的扁平化版本相等
        assert_array_equal(y, x.flat)

    # 测试直接将数组写入文件并从文件中读取的往返操作
    def test_roundtrip(self, x, tmp_filename):
        # 将数组 x 直接写入到临时文件 tmp_filename
        x.tofile(tmp_filename)
        # 从临时文件 tmp_filename 中读取数据到数组 y，数据类型为 x 的数据类型
        y = np.fromfile(tmp_filename, dtype=x.dtype)
        # 断言数组 y 与原始数组 x 的扁平化版本相等
        assert_array_equal(y, x.flat)

    # 测试使用 Pathlib 对象进行数据的序列化和反序列化操作的情况
    def test_roundtrip_dump_pathlib(self, x, tmp_filename):
        # 创建一个 Path 对象 p，表示临时文件 tmp_filename
        p = pathlib.Path(tmp_filename)
        # 将数组 x 序列化并存储到文件 p
        x.dump(p)
        # 从文件 p 中加载数据到数组 y，允许使用 pickle
        y = np.load(p, allow_pickle=True)
        # 断言数组 y 与原始数组 x 相等
        assert_array_equal(y, x)
    # 测试将二进制数据转换为字符串并再次转换为数组的过程
    def test_roundtrip_binary_str(self, x):
        # 将数组转换为字节序列
        s = x.tobytes()
        # 从字节序列中重新解析为 NumPy 数组
        y = np.frombuffer(s, dtype=x.dtype)
        # 断言重新解析的数组与原始数组内容相等
        assert_array_equal(y, x.flat)

        # 将数组按照 Fortran 风格的顺序转换为字节序列
        s = x.tobytes('F')
        # 从字节序列中重新解析为 NumPy 数组
        y = np.frombuffer(s, dtype=x.dtype)
        # 断言重新解析的数组与按 Fortran 顺序展平后的数组内容相等
        assert_array_equal(y, x.flatten('F'))

    # 测试将字符串转换为数组并再次转换为字符串的过程
    def test_roundtrip_str(self, x):
        # 将实部展平为一维数组
        x = x.real.ravel()
        # 将数组中的每个元素转换为字符串，并用 "@" 连接成一个字符串
        s = "@".join(map(str, x))
        # 从字符串中解析出数组，使用 "@" 作为分隔符
        y = np.fromstring(s, sep="@")
        # 创建一个 NaN 掩码，用于检查非有限数值的一致性
        nan_mask = ~np.isfinite(x)
        # 断言 NaN 掩码中的值在解析后仍保持一致
        assert_array_equal(x[nan_mask], y[nan_mask])
        # 断言非 NaN 掩码中的值在解析后仍保持一致
        assert_array_equal(x[~nan_mask], y[~nan_mask])

    # 测试将对象的 repr 字符串转换为数组并再次转换为 repr 字符串的过程
    def test_roundtrip_repr(self, x):
        # 将实部展平为一维数组
        x = x.real.ravel()
        # 将数组中每个元素的 repr 字符串截取掉开头的部分和末尾的引号，并用 "@" 连接成一个字符串
        s = "@".join(map(lambda x: repr(x)[11:-1], x))
        # 从字符串中解析出数组，使用 "@" 作为分隔符
        y = np.fromstring(s, sep="@")
        # 断言数组的 repr 字符串解析后与原始数组内容相等
        assert_array_equal(x, y)

    # 测试不可寻址文件对象的 fromfile 方法的行为
    def test_unseekable_fromfile(self, x, tmp_filename):
        # 将数组写入临时文件
        x.tofile(tmp_filename)

        # 定义一个失败函数，模拟无法执行 seek 或 tell 操作
        def fail(*args, **kwargs):
            raise OSError('Can not tell or seek')

        # 使用不可寻址的文件对象进行测试
        with open(tmp_filename, 'rb', buffering=0) as f:
            # 替换文件对象的 seek 和 tell 方法为失败函数
            f.seek = fail
            f.tell = fail
            # 断言调用 fromfile 方法时会抛出 OSError 异常
            assert_raises(OSError, np.fromfile, f, dtype=x.dtype)

    # 测试使用不带缓冲区的文件对象的 fromfile 方法的行为
    def test_io_open_unbuffered_fromfile(self, x, tmp_filename):
        # 将数组写入临时文件
        x.tofile(tmp_filename)
        # 使用不带缓冲区的文件对象进行读取
        with open(tmp_filename, 'rb', buffering=0) as f:
            # 从文件中读取数据并解析为 NumPy 数组
            y = np.fromfile(f, dtype=x.dtype)
            # 断言读取的数组与原始数组内容相等
            assert_array_equal(y, x.flat)

    # 测试使用带缓冲区的文件对象的 fromfile 方法的行为
    def test_io_open_buffered_fromfile(self, x, tmp_filename):
        # 将数组写入临时文件
        x.tofile(tmp_filename)
        # 使用带缓冲区的文件对象进行读取
        with open(tmp_filename, 'rb', buffering=-1) as f:
            # 从文件中读取数据并解析为 NumPy 数组
            y = np.fromfile(f, dtype=x.dtype)
        # 断言读取的数组与原始数组内容相等
        assert_array_equal(y, x.flat)

    # 测试处理较大文件的行为
    def test_largish_file(self, tmp_filename):
        # 创建一个大小为 4MB 的零数组，并将其写入临时文件
        d = np.zeros(4 * 1024 ** 2)
        d.tofile(tmp_filename)
        # 断言临时文件的大小与数组的字节数相等
        assert_equal(os.path.getsize(tmp_filename), d.nbytes)
        # 断言从文件中读取的数组与原始数组内容相等
        assert_array_equal(d, np.fromfile(tmp_filename))
        
        # 检查文件偏移量的行为
        with open(tmp_filename, "r+b") as f:
            # 将数组写入文件的偏移量位置
            f.seek(d.nbytes)
            d.tofile(f)
            # 断言文件大小变为原来的两倍
            assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)
        
        # 检查追加模式的行为 (gh-8329)
        # 清空临时文件的内容
        open(tmp_filename, "w").close()
        with open(tmp_filename, "ab") as f:
            # 向文件追加数组内容
            d.tofile(f)
        # 断言从文件中读取的数组与原始数组内容相等
        assert_array_equal(d, np.fromfile(tmp_filename))
        with open(tmp_filename, "ab") as f:
            # 再次向文件追加数组内容
            d.tofile(f)
        # 断言文件大小变为原来的两倍
        assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)
    def test_file_position_after_fromfile(self, tmp_filename):
        # gh-4118
        # 定义不同的缓冲区大小，用于测试
        sizes = [io.DEFAULT_BUFFER_SIZE//8,
                 io.DEFAULT_BUFFER_SIZE,
                 io.DEFAULT_BUFFER_SIZE*8]

        # 遍历不同大小的缓冲区
        for size in sizes:
            # 使用 'wb' 模式打开临时文件
            with open(tmp_filename, 'wb') as f:
                # 将文件指针定位到指定位置，写入一个字节
                f.seek(size-1)
                f.write(b'\0')

            # 遍历不同的打开模式：'rb' 和 'r+b'
            for mode in ['rb', 'r+b']:
                # 构造错误消息，包含当前大小和模式
                err_msg = "%d %s" % (size, mode)

                # 使用指定模式打开临时文件
                with open(tmp_filename, mode) as f:
                    # 从文件中读取前两个字节
                    f.read(2)
                    # 使用 NumPy 从文件中读取一个 float64 类型的数据
                    np.fromfile(f, dtype=np.float64, count=1)
                    # 获取当前文件指针位置
                    pos = f.tell()
                # 断言当前文件指针位置为 10，如果不是则抛出错误消息
                assert_equal(pos, 10, err_msg=err_msg)

    def test_file_position_after_tofile(self, tmp_filename):
        # gh-4118
        # 定义不同的缓冲区大小，用于测试
        sizes = [io.DEFAULT_BUFFER_SIZE//8,
                 io.DEFAULT_BUFFER_SIZE,
                 io.DEFAULT_BUFFER_SIZE*8]

        # 遍历不同大小的缓冲区
        for size in sizes:
            # 构造错误消息，包含当前大小
            err_msg = "%d" % (size,)

            # 使用 'wb' 模式打开临时文件
            with open(tmp_filename, 'wb') as f:
                # 将文件指针定位到指定位置，写入一个字节
                f.seek(size-1)
                f.write(b'\0')
                # 将文件指针定位到位置 10，写入字节 '12'
                f.seek(10)
                f.write(b'12')
                # 使用 NumPy 将一个 float64 类型的数组写入文件
                np.array([0], dtype=np.float64).tofile(f)
                # 获取当前文件指针位置
                pos = f.tell()
            # 断言当前文件指针位置为 10 + 2 + 8，如果不是则抛出错误消息
            assert_equal(pos, 10 + 2 + 8, err_msg=err_msg)

            # 使用 'r+b' 模式打开临时文件
            with open(tmp_filename, 'r+b') as f:
                # 从文件中读取前两个字节
                f.read(2)
                # 在 ANSI C 中，读写操作之间需要进行 seek 操作
                f.seek(0, 1)  # seek between read&write required by ANSI C
                # 使用 NumPy 将一个 float64 类型的数组写入文件
                np.array([0], dtype=np.float64).tofile(f)
                # 获取当前文件指针位置
                pos = f.tell()
            # 断言当前文件指针位置为 10，如果不是则抛出错误消息
            assert_equal(pos, 10, err_msg=err_msg)

    def test_load_object_array_fromfile(self, tmp_filename):
        # gh-12300
        # 使用 'w' 模式打开临时文件，确保其具有一致的内容
        with open(tmp_filename, 'w') as f:
            pass

        # 使用 'rb' 模式打开临时文件
        with open(tmp_filename, 'rb') as f:
            # 断言从文件中读取对象数组时会引发 ValueError 错误
            assert_raises_regex(ValueError, "Cannot read into object array",
                                np.fromfile, f, dtype=object)

        # 断言从文件名为 tmp_filename 的对象中读取对象数组时会引发 ValueError 错误
        assert_raises_regex(ValueError, "Cannot read into object array",
                            np.fromfile, tmp_filename, dtype=object)
    # 定义测试函数，用于测试从文件读取数据的偏移量功能
    def test_fromfile_offset(self, x, tmp_filename):
        # 将数组 x 写入临时文件 tmp_filename
        with open(tmp_filename, 'wb') as f:
            x.tofile(f)

        # 从文件读取数据，无偏移量
        with open(tmp_filename, 'rb') as f:
            y = np.fromfile(f, dtype=x.dtype, offset=0)
            assert_array_equal(y, x.flat)

        # 从文件读取数据，设置偏移量和数据数量
        with open(tmp_filename, 'rb') as f:
            count_items = len(x.flat) // 8
            offset_items = len(x.flat) // 4
            offset_bytes = x.dtype.itemsize * offset_items
            y = np.fromfile(
                f, dtype=x.dtype, count=count_items, offset=offset_bytes
            )
            assert_array_equal(
                y, x.flat[offset_items:offset_items+count_items]
            )

            # 连续的 seek 操作应当生效
            offset_bytes = x.dtype.itemsize
            z = np.fromfile(f, dtype=x.dtype, offset=offset_bytes)
            assert_array_equal(z, x.flat[offset_items+count_items+1:])

        # 使用不同的分隔符将数组 x 写入临时文件
        with open(tmp_filename, 'wb') as f:
            x.tofile(f, sep=",")

        # 尝试从文件读取数据，使用了不允许的 offset 参数
        with open(tmp_filename, 'rb') as f:
            assert_raises_regex(
                    TypeError,
                    "'offset' argument only permitted for binary files",
                    np.fromfile, tmp_filename, dtype=x.dtype,
                    sep=",", offset=1)

    # 标记为跳过测试，如果运行环境是 PyPy 的话
    @pytest.mark.skipif(IS_PYPY, reason="bug in PyPy's PyNumber_AsSsize_t")
    # 测试从文件读取时出现的错误情况
    def test_fromfile_bad_dup(self, x, tmp_filename):
        # 定义一个返回固定字符串的 dup 函数
        def dup_str(fd):
            return 'abc'

        # 定义一个返回超大整数的 dup 函数
        def dup_bigint(fd):
            return 2**68

        # 备份原始的 os.dup 函数
        old_dup = os.dup
        try:
            # 将数组 x 写入临时文件 tmp_filename
            with open(tmp_filename, 'wb') as f:
                x.tofile(f)
                # 针对每个 dup 函数和对应的异常类型，执行断言测试
                for dup, exc in ((dup_str, TypeError), (dup_bigint, OSError)):
                    os.dup = dup
                    assert_raises(exc, np.fromfile, f)
        finally:
            # 恢复原始的 os.dup 函数
            os.dup = old_dup

    # 辅助函数，用于检查从字符串或文件中读取的数据是否与给定值相等
    def _check_from(self, s, value, filename, **kw):
        if 'sep' not in kw:
            # 使用 np.frombuffer 从字节串 s 中读取数据
            y = np.frombuffer(s, **kw)
        else:
            # 使用 np.fromstring 从字符串 s 中读取数据
            y = np.fromstring(s, **kw)
        # 断言读取的数据与给定的 value 相等
        assert_array_equal(y, value)

        # 将字节串 s 写入文件 filename
        with open(filename, 'wb') as f:
            f.write(s)
        # 从文件 filename 中读取数据，并与给定的 value 进行比较
        y = np.fromfile(filename, **kw)
        assert_array_equal(y, value)

    # pytest 的 fixture，用于测试不同的小数分隔符情况
    @pytest.fixture(params=["period", "comma"])
    def decimal_sep_localization(self, request):
        """
        包含此 fixture 到测试中，将会自动在两种小数分隔符下执行测试。

        所以::

            def test_decimal(decimal_sep_localization):
                pass

        相当于以下两个测试的结合::

            def test_decimal_period_separator():
                pass

            def test_decimal_comma_separator():
                with CommaDecimalPointLocale():
                    pass
        """
        if request.param == "period":
            yield
        elif request.param == "comma":
            with CommaDecimalPointLocale():
                yield
        else:
            assert False, request.param
    # 测试处理 NaN 值的方法
    def test_nan(self, tmp_filename, decimal_sep_localization):
        # 调用 _check_from 方法，传入包含 NaN 不同表示形式的字节字符串和期望的 NaN 数组
        self._check_from(
            b"nan +nan -nan NaN nan(foo) +NaN(BAR) -NAN(q_u_u_x_)",
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            tmp_filename,
            sep=' ')

    # 测试处理无穷大值的方法
    def test_inf(self, tmp_filename, decimal_sep_localization):
        # 调用 _check_from 方法，传入包含不同形式的无穷大表示的字节字符串和期望的无穷大数值数组
        self._check_from(
            b"inf +inf -inf infinity -Infinity iNfInItY -inF",
            [np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf],
            tmp_filename,
            sep=' ')

    # 测试处理数字的方法
    def test_numbers(self, tmp_filename, decimal_sep_localization):
        # 调用 _check_from 方法，传入包含不同数字表示的字节字符串和期望的浮点数数组
        self._check_from(
            b"1.234 -1.234 .3 .3e55 -123133.1231e+133",
            [1.234, -1.234, .3, .3e55, -123133.1231e+133],
            tmp_filename,
            sep=' ')

    # 测试处理二进制数据的方法
    def test_binary(self, tmp_filename):
        # 调用 _check_from 方法，传入二进制数据和期望的 numpy 浮点数数组
        self._check_from(
            b'\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@',
            np.array([1, 2, 3, 4]),
            tmp_filename,
            dtype='<f4')

    # 测试处理字符串的方法
    def test_string(self, tmp_filename):
        # 调用 _check_from 方法，传入包含逗号分隔的字符串和期望的浮点数数组
        self._check_from(b'1,2,3,4', [1., 2., 3., 4.], tmp_filename, sep=',')

    # 测试处理带计数参数的字符串的方法
    def test_counted_string(self, tmp_filename, decimal_sep_localization):
        # 调用 _check_from 方法，传入包含逗号分隔的字符串和期望的浮点数数组，以及不同的计数参数
        self._check_from(
            b'1,2,3,4', [1., 2., 3., 4.], tmp_filename, count=4, sep=',')
        self._check_from(
            b'1,2,3,4', [1., 2., 3.], tmp_filename, count=3, sep=',')
        self._check_from(
            b'1,2,3,4', [1., 2., 3., 4.], tmp_filename, count=-1, sep=',')

    # 测试处理带空白字符的字符串的方法
    def test_string_with_ws(self, tmp_filename):
        # 调用 _check_from 方法，传入包含空格分隔的字符串和期望的整数数组，指定数据类型为整数
        self._check_from(
            b'1 2  3     4   ', [1, 2, 3, 4], tmp_filename, dtype=int, sep=' ')

    # 测试处理带空白字符和计数参数的字符串的方法
    def test_counted_string_with_ws(self, tmp_filename):
        # 调用 _check_from 方法，传入包含空格分隔的字符串和期望的整数数组，以及指定的计数参数和数据类型
        self._check_from(
            b'1 2  3     4   ', [1, 2, 3], tmp_filename, count=3, dtype=int,
            sep=' ')

    # 测试处理 ASCII 字符串的方法
    def test_ascii(self, tmp_filename, decimal_sep_localization):
        # 调用 _check_from 方法，传入包含逗号分隔的字符串和期望的浮点数数组，指定数据类型为浮点数
        self._check_from(
            b'1 , 2 , 3 , 4', [1., 2., 3., 4.], tmp_filename, sep=',')
        self._check_from(
            b'1,2,3,4', [1., 2., 3., 4.], tmp_filename, dtype=float, sep=',')

    # 测试处理格式错误数据的方法
    def test_malformed(self, tmp_filename, decimal_sep_localization):
        # 使用 assert_warns 检查是否会发出 DeprecationWarning
        with assert_warns(DeprecationWarning):
            # 调用 _check_from 方法，传入包含不同形式数字的字符串和期望的浮点数数组
            self._check_from(
                b'1.234 1,234', [1.234, 1.], tmp_filename, sep=' ')

    # 测试处理使用长分隔符的字符串的方法
    def test_long_sep(self, tmp_filename):
        # 调用 _check_from 方法，传入包含使用长分隔符分隔的字符串和期望的整数数组
        self._check_from(
            b'1_x_3_x_4_x_5', [1, 3, 4, 5], tmp_filename, sep='_x_')

    # 测试处理指定数据类型的方法
    def test_dtype(self, tmp_filename):
        # 创建一个指定数据类型的 numpy 数组
        v = np.array([1, 2, 3, 4], dtype=np.int_)
        # 调用 _check_from 方法，传入包含逗号分隔的字符串和指定的 numpy 数组，以及数据类型和分隔符参数
        self._check_from(b'1,2,3,4', v, tmp_filename, sep=',', dtype=np.int_)
    # 测试布尔类型数组的写入和读取
    def test_dtype_bool(self, tmp_filename):
        # 创建一个布尔类型的 NumPy 数组
        v = np.array([True, False, True, False], dtype=np.bool)
        # 准备一个字节串作为测试数据
        s = b'1,0,-2.3,0'
        # 将测试数据写入临时文件
        with open(tmp_filename, 'wb') as f:
            f.write(s)
        # 从临时文件中读取数据并转换为布尔类型数组
        y = np.fromfile(tmp_filename, sep=',', dtype=np.bool)
        # 断言读取的数组类型为布尔类型
        assert_(y.dtype == '?')
        # 断言读取的数组内容与预期的布尔类型数组相等
        assert_array_equal(y, v)

    # 测试使用不同分隔符写入和读取数组
    def test_tofile_sep(self, tmp_filename, decimal_sep_localization):
        # 创建一个浮点类型的 NumPy 数组
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        # 将数组以逗号分隔符写入临时文件
        with open(tmp_filename, 'w') as f:
            x.tofile(f, sep=',')
        # 从临时文件中读取数据并转换为浮点类型数组
        with open(tmp_filename, 'r') as f:
            s = f.read()
        # 将读取的数据转换为浮点类型数组
        y = np.array([float(p) for p in s.split(',')])
        # 断言读取的数组内容与原始数组相等
        assert_array_equal(x, y)

    # 测试以指定格式写入和读取数组
    def test_tofile_format(self, tmp_filename, decimal_sep_localization):
        # 创建一个浮点类型的 NumPy 数组
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        # 将数组以逗号分隔符和指定格式写入临时文件
        with open(tmp_filename, 'w') as f:
            x.tofile(f, sep=',', format='%.2f')
        # 从临时文件中读取数据
        with open(tmp_filename, 'r') as f:
            s = f.read()
        # 断言读取的字符串与期望的格式化字符串相等
        assert_equal(s, '1.51,2.00,3.51,4.00')

    # 测试处理错误情况下的文件操作
    def test_tofile_cleanup(self, tmp_filename):
        # 创建一个对象类型的 NumPy 数组
        x = np.zeros((10), dtype=object)
        # 断言在写入时会引发 OSError 异常
        with open(tmp_filename, 'wb') as f:
            assert_raises(OSError, lambda: x.tofile(f, sep=''))
        # 移除临时文件，关闭文件句柄
        os.remove(tmp_filename)
        # 断言在关闭 Python 文件句柄后再次写入会引发 OSError 异常
        assert_raises(OSError, lambda: x.tofile(tmp_filename))
        # 再次移除临时文件
        os.remove(tmp_filename)

    # 测试从文件中读取子数组的二进制数据
    def test_fromfile_subarray_binary(self, tmp_filename):
        # 创建一个多维整型数组
        x = np.arange(24, dtype="i4").reshape(2, 3, 4)
        # 将数组以二进制形式写入临时文件
        x.tofile(tmp_filename)
        # 从临时文件中读取数据并将其解析为子数组形式
        res = np.fromfile(tmp_filename, dtype="(3,4)i4")
        # 断言读取的数组与原始数组相等
        assert_array_equal(x, res)
        # 将数组转换为字节串
        x_str = x.tobytes()
        # 断言使用 fromstring 方法读取数组的二进制数据会引发警告
        with assert_warns(DeprecationWarning):
            res = np.fromstring(x_str, dtype="(3,4)i4")
            assert_array_equal(x, res)

    # 测试不支持解析子数组数据类型的情况
    def test_parsing_subarray_unsupported(self, tmp_filename):
        # 准备包含重复数据的字符串
        data = "12,42,13," * 50
        # 断言解析时会引发 ValueError 异常
        with pytest.raises(ValueError):
            expected = np.fromstring(data, dtype="(3,)i", sep=",")
        # 将数据写入临时文件
        with open(tmp_filename, "w") as f:
            f.write(data)
        # 断言从文件中读取数据并解析时会引发 ValueError 异常
        with pytest.raises(ValueError):
            np.fromfile(tmp_filename, dtype="(3,)i", sep=",")
    # 定义一个测试方法，用于测试读取短于指定长度的子数组时的行为
    def test_read_shorter_than_count_subarray(self, tmp_filename):
        # 测试在请求更多数值时，与子数组维度合并到数组维度时不会出现问题
        expected = np.arange(511 * 10, dtype="i").reshape(-1, 10)

        # 将期望的数组转换为字节序列
        binary = expected.tobytes()

        # 使用 pytest 断言来验证在特定条件下是否会抛出 ValueError 异常
        with pytest.raises(ValueError):
            # 使用 pytest 断言来验证是否会发出 DeprecationWarning 警告
            with pytest.warns(DeprecationWarning):
                # 从二进制数据中解析出数组，指定数据类型为 "(10,)i"，尝试读取 10000 个元素
                np.fromstring(binary, dtype="(10,)i", count=10000)

        # 将期望的数组以二进制格式写入临时文件
        expected.tofile(tmp_filename)

        # 从临时文件中读取数据，数据类型为 "(10,)i"，尝试读取 10000 个元素
        res = np.fromfile(tmp_filename, dtype="(10,)i", count=10000)

        # 使用断言检查从文件中读取的结果是否与期望的数组一致
        assert_array_equal(res, expected)
# 定义一个测试类 TestFromBuffer，用于测试从缓冲区创建 NumPy 数组的功能
class TestFromBuffer:
    
    # 使用 pytest.mark.parametrize 装饰器，为 byteorder 参数指定两种取值 '<' 和 '>'
    # 同时为 dtype 参数指定三种数据类型：float, int, complex
    @pytest.mark.parametrize('byteorder', ['<', '>'])
    @pytest.mark.parametrize('dtype', [float, int, complex])
    def test_basic(self, byteorder, dtype):
        # 创建指定 byteorder 的新 NumPy 数据类型对象
        dt = np.dtype(dtype).newbyteorder(byteorder)
        # 生成一个随机的 NumPy 数组 x，元素是指定数据类型的随机数，并转换成指定的字节序
        x = (np.random.random((4, 7)) * 5).astype(dt)
        # 将数组 x 转换为字节流 buf
        buf = x.tobytes()
        # 使用 np.frombuffer 从 buf 中解析出一个新的 NumPy 数组，指定数据类型为 dt
        assert_array_equal(np.frombuffer(buf, dtype=dt), x.flat)

    # 使用 pytest.mark.parametrize 装饰器，为 obj 参数指定两种类型：np.arange(10) 和 b"12345678"
    def test_array_base(self, obj):
        # 测试当传入的对象 obj 不使用 release_buffer 插槽时，直接作为基础对象使用的情况
        # 参见 GitHub issue gh-21612
        new = np.frombuffer(obj)
        # 断言新创建的数组 new 的基础对象是 obj
        assert new.base is obj

    def test_empty(self):
        # 测试当传入空字节串 b'' 时，是否返回一个空的 NumPy 数组
        assert_array_equal(np.frombuffer(b''), np.array([]))

    # 使用 pytest.mark.skipif 装饰器，条件是 IS_PYPY 为 True
    # 给出跳过测试的理由，指向 PyPy 的问题追踪链接
    def test_mmap_close(self):
        # 测试 mmap 对象关闭时，使用 frombuffer 是否安全
        # 旧的缓冲区协议不适用于一些新功能，但是 frombuffer 长期以来一直使用旧协议
        # 使用新协议（使用 memoryviews）确保安全性
        with tempfile.TemporaryFile(mode='wb') as tmp:
            tmp.write(b"asdf")
            tmp.flush()
            # 创建一个 mmap 对象 mm，映射到临时文件 tmp 的文件描述符，大小为 0
            mm = mmap.mmap(tmp.fileno(), 0)
            # 使用 np.frombuffer 从 mmap 对象创建一个新的 NumPy 数组，数据类型为 np.uint8
            arr = np.frombuffer(mm, dtype=np.uint8)
            # 使用 pytest.raises 检查在数组 arr 使用缓冲区时关闭 mm 是否会引发 BufferError
            with pytest.raises(BufferError):
                mm.close()  # 不能在数组使用缓冲区时关闭 mmap 对象
            del arr
            # 安全地关闭 mmap 对象
            mm.close()

# 定义测试类 TestFlat，用于测试 NumPy 数组的扁平化操作
class TestFlat:
    
    # 在每个测试方法运行前设置初始状态
    def setup_method(self):
        # 创建一个连续存储的 NumPy 数组 a0，形状为 (4, 5)
        a0 = np.arange(20.0)
        a = a0.reshape(4, 5)
        # 将 a0 视图的形状修改为 (4, 5)，并设置为不可写
        a0.shape = (4, 5)
        a.flags.writeable = False
        self.a = a
        # 创建数组 a 的视图 b，步长为 2
        self.b = a[::2, ::2]
        self.a0 = a0
        # 创建 a0 视图的视图 b0，步长为 2
        self.b0 = a0[::2, ::2]

    def test_contiguous(self):
        # 测试在连续存储的数组 a 上修改 flat 属性的行为
        testpassed = False
        try:
            self.a.flat[12] = 100.0
        except ValueError:
            testpassed = True
        # 断言是否捕获到 ValueError 异常
        assert_(testpassed)
        # 断言数组 a 的 flat 属性中索引 12 的值为 12.0
        assert_(self.a.flat[12] == 12.0)

    def test_discontiguous(self):
        # 测试在非连续存储的数组 b 上修改 flat 属性的行为
        testpassed = False
        try:
            self.b.flat[4] = 100.0
        except ValueError:
            testpassed = True
        # 断言是否捕获到 ValueError 异常
        assert_(testpassed)
        # 断言数组 b 的 flat 属性中索引 4 的值为 12.0
        assert_(self.b.flat[4] == 12.0)

    def test___array__(self):
        # 测试 flat 属性的 __array__ 方法返回的数组的写入权限及写回复制标志
        c = self.a.flat.__array__()
        d = self.b.flat.__array__()
        e = self.a0.flat.__array__()
        f = self.b0.flat.__array__()

        # 断言数组 c 和 d 的写入权限为 False
        assert_(c.flags.writeable is False)
        assert_(d.flags.writeable is False)
        # 断言数组 e 和 f 的写入权限为 True 和 False
        assert_(e.flags.writeable is True)
        assert_(f.flags.writeable is False)
        # 断言数组 c、d、e、f 的写回复制标志均为 False
        assert_(c.flags.writebackifcopy is False)
        assert_(d.flags.writebackifcopy is False)
        assert_(e.flags.writebackifcopy is False)
        assert_(f.flags.writebackifcopy is False)

    # 使用 pytest.mark.skipif 装饰器，条件是 not HAS_REFCOUNT 为 True
    # 给出跳过测试的理由，说明 Python 缺少 refcounts
    def test_refcount(self):
        # 包括对引用计数错误 gh-13165 的回归测试
        inds = [np.intp(0), np.array([True]*self.a.size), np.array([0]), None]
        # 定义索引类型为 np.intp
        indtype = np.dtype(np.intp)
        # 获取索引类型的引用计数
        rc_indtype = sys.getrefcount(indtype)
        for ind in inds:
            # 获取当前索引对象的引用计数
            rc_ind = sys.getrefcount(ind)
            for _ in range(100):
                try:
                    # 访问 self.a 的平坦索引 ind
                    self.a.flat[ind]
                except IndexError:
                    pass
            # 断言当前索引对象的引用计数变化不超过 50
            assert_(abs(sys.getrefcount(ind) - rc_ind) < 50)
            # 断言索引类型的引用计数变化不超过 50
            assert_(abs(sys.getrefcount(indtype) - rc_indtype) < 50)

    def test_index_getset(self):
        # 创建一个平坦迭代器 it
        it = np.arange(10).reshape(2, 1, 5).flat
        # 使用 pytest 断言捕获 AttributeError 异常
        with pytest.raises(AttributeError):
            it.index = 10

        for _ in it:
            pass
        # 检查 `.index` 的值是否正确更新（参见 gh-19153）
        # 如果类型不正确，在大端序机器上会显示问题
        assert it.index == it.base.size

    def test_maxdims(self):
        # 当前平坦迭代器和属性受到限制，仅支持最多 32 维（在 2.0 版本后提升到 64 维）
        a = np.ones((1,) * 64)

        with pytest.raises(RuntimeError,
                match=".*32 dimensions but the array has 64"):
            a.flat
# 定义一个名为 TestResize 的测试类
class TestResize:

    # 装饰器函数，用于标记不需要追踪的测试方法
    @_no_tracing
    # 测试基本的数组调整功能
    def test_basic(self):
        # 创建一个二维数组
        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 根据平台判断是否使用 refcheck=False 调整数组大小
        if IS_PYPY:
            x.resize((5, 5), refcheck=False)
        else:
            x.resize((5, 5))
        # 断言数组的前9个元素与指定的数组相等
        assert_array_equal(x.flat[:9],
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).flat)
        # 断言数组的剩余元素都为0
        assert_array_equal(x[9:].flat, 0)

    # 检查引用关系的测试方法
    def test_check_reference(self):
        # 创建一个二维数组并将其赋给变量 y
        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y = x
        # 断言调整数组大小时会引发 ValueError 异常
        assert_raises(ValueError, x.resize, (5, 1))
        # 删除变量 y，避免 pyflakes 报告未使用变量的警告
        del y

    # 测试调整为整数形状的数组
    @_no_tracing
    def test_int_shape(self):
        # 创建一个单位矩阵
        x = np.eye(3)
        # 根据平台判断是否使用 refcheck=False 调整数组大小
        if IS_PYPY:
            x.resize(3, refcheck=False)
        else:
            x.resize(3)
        # 断言数组与预期的行相等
        assert_array_equal(x, np.eye(3)[0,:])

    # 测试调整为 None 形状的数组
    def test_none_shape(self):
        # 创建一个单位矩阵
        x = np.eye(3)
        # 调整数组大小为 None
        x.resize(None)
        # 断言数组与预期的单位矩阵相等
        assert_array_equal(x, np.eye(3))
        # 再次调整数组大小为默认形状
        x.resize()
        # 断言数组与预期的单位矩阵相等
        assert_array_equal(x, np.eye(3))

    # 测试调整为0维形状的数组
    def test_0d_shape(self):
        # 多次测试以确保不会破坏分配缓存 gh-9216
        for i in range(10):
            # 创建一个长度为1的空数组
            x = np.empty((1,))
            # 调整数组大小为0维
            x.resize(())
            # 断言数组的形状为 ()
            assert_equal(x.shape, ())
            # 断言数组的大小为 1
            assert_equal(x.size, 1)
            # 创建一个空数组
            x = np.empty(())
            # 调整数组大小为 (1,)
            x.resize((1,))
            # 断言数组的形状为 (1,)
            assert_equal(x.shape, (1,))
            # 断言数组的大小为 1
            assert_equal(x.size, 1)

    # 测试无效参数的情况
    def test_invalid_arguments(self):
        # 断言调用 np.eye(3).resize('hi') 时会引发 TypeError 异常
        assert_raises(TypeError, np.eye(3).resize, 'hi')
        # 断言调用 np.eye(3).resize(-1) 时会引发 ValueError 异常
        assert_raises(ValueError, np.eye(3).resize, -1)
        # 断言调用 np.eye(3).resize(order=1) 时会引发 TypeError 异常
        assert_raises(TypeError, np.eye(3).resize, order=1)
        # 断言调用 np.eye(3).resize(refcheck='hi') 时会引发 TypeError 异常
        assert_raises(TypeError, np.eye(3).resize, refcheck='hi')

    # 测试调整为自由形状的数组
    @_no_tracing
    def test_freeform_shape(self):
        # 创建一个单位矩阵
        x = np.eye(3)
        # 根据平台判断是否使用 refcheck=False 调整数组大小
        if IS_PYPY:
            x.resize(3, 2, 1, refcheck=False)
        else:
            x.resize(3, 2, 1)
        # 断言数组的形状为 (3, 2, 1)
        assert_(x.shape == (3, 2, 1))

    # 测试追加0的情况
    @_no_tracing
    def test_zeros_appended(self):
        # 创建一个单位矩阵
        x = np.eye(3)
        # 根据平台判断是否使用 refcheck=False 调整数组大小
        if IS_PYPY:
            x.resize(2, 3, 3, refcheck=False)
        else:
            x.resize(2, 3, 3)
        # 断言数组的第一行与单位矩阵相等
        assert_array_equal(x[0], np.eye(3))
        # 断言数组的第二行为3x3的全0矩阵
        assert_array_equal(x[1], np.zeros((3, 3)))

    # 测试对象数组的情况
    @_no_tracing
    def test_obj_obj(self):
        # 检查在调整大小时初始化内存，gh-4857
        # 创建一个元素类型为对象的全1数组
        a = np.ones(10, dtype=[('k', object, 2)])
        # 根据平台判断是否使用 refcheck=False 调整数组大小
        if IS_PYPY:
            a.resize(15, refcheck=False)
        else:
            a.resize(15,)
        # 断言数组的形状为 (15,)
        assert_equal(a.shape, (15,))
        # 断言数组末尾元素的 'k' 字段为0
        assert_array_equal(a['k'][-5:], 0)
        # 断言数组除了末尾的 'k' 字段都为1
        assert_array_equal(a['k'][:-5], 1)

    # 测试空视图的情况
    def test_empty_view(self):
        # 检查包含0的大小不会触发已空数组的重新分配
        # 创建一个大小为 (10, 0) 的全0数组
        x = np.zeros((10, 0), int)
        # 创建一个数组视图
        x_view = x[...]
        # 调整数组视图大小为 (0, 10)
        x_view.resize((0, 10))
        # 再次调整数组视图大小为 (0, 100)
        x_view.resize((0, 100))
    # 定义一个测试函数，用于检查弱引用
    def test_check_weakref(self):
        # 创建一个 NumPy 数组，包含三行三列的单位矩阵
        x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # 创建 x 的弱引用对象 xref
        xref = weakref.ref(x)
        # 使用 assert_raises 断言，期望调用 x.resize((5, 1)) 时引发 ValueError 异常
        assert_raises(ValueError, x.resize, (5, 1))
        # 删除 xref 引用，避免 pyflakes 报告未使用变量的警告。
        del xref
# 定义一个测试记录类 TestRecord
class TestRecord:
    
    # 定义测试方法 test_field_rename
    def test_field_rename(self):
        # 创建一个包含字段 'f' 和 'i' 的 NumPy 数据类型
        dt = np.dtype([('f', float), ('i', int)])
        # 将字段名称修改为 ['p', 'q']
        dt.names = ['p', 'q']
        # 断言字段名称已成功修改为 ['p', 'q']
        assert_equal(dt.names, ['p', 'q'])

    # 定义测试方法 test_multiple_field_name_occurrence
    def test_multiple_field_name_occurrence(self):
        
        # 定义内部函数 test_dtype_init
        def test_dtype_init():
            # 尝试创建包含重复字段名的 NumPy 数据类型
            np.dtype([("A", "f8"), ("B", "f8"), ("A", "f8")])

        # 断言当存在重复字段名时，会引发 ValueError 错误
        assert_raises(ValueError, test_dtype_init)

    # 定义测试方法 test_bytes_fields
    def test_bytes_fields(self):
        # 断言在 Python 3 中，不允许使用字节作为字段名，也不会被视为标题
        assert_raises(TypeError, np.dtype, [(b'a', int)])
        assert_raises(TypeError, np.dtype, [(('b', b'a'), int)])

        # 创建一个包含复合字段名的 NumPy 数据类型
        dt = np.dtype([((b'a', 'b'), int)])
        assert_raises(TypeError, dt.__getitem__, b'a')

        # 创建一个 NumPy 数组 x，使用上述数据类型 dt
        x = np.array([(1,), (2,), (3,)], dtype=dt)
        assert_raises(IndexError, x.__getitem__, b'a')

        # 获取数组 x 的第一个元素
        y = x[0]
        assert_raises(IndexError, y.__getitem__, b'a')

    # 定义测试方法 test_multiple_field_name_unicode
    def test_multiple_field_name_unicode(self):
        
        # 定义内部函数 test_dtype_unicode
        def test_dtype_unicode():
            # 尝试创建包含重复字段名（包括 Unicode 字符）的 NumPy 数据类型
            np.dtype([("\u20B9", "f8"), ("B", "f8"), ("\u20B9", "f8")])

        # 断言当存在重复字段名（包括 Unicode 字符）时，会引发 ValueError 错误
        assert_raises(ValueError, test_dtype_unicode)

    # 定义测试方法 test_fromarrays_unicode
    def test_fromarrays_unicode(self):
        # 在 Python 2 和 3 中，允许 fromarrays() 方法的字段名为 Unicode 字符串
        x = np._core.records.fromarrays(
            [[0], [1]], names='a,b', formats='i4,i4')
        # 断言从数组 x 中取出字段 'a' 的第一个值为 0
        assert_equal(x['a'][0], 0)
        # 断言从数组 x 中取出字段 'b' 的第一个值为 1
        assert_equal(x['b'][0], 1)

    # 定义测试方法 test_unicode_order
    def test_unicode_order(self):
        # 测试在 Python 2 和 3 中，可以使用 Unicode 字符作为排序字段名
        name = 'b'
        x = np.array([1, 3, 2], dtype=[(name, int)])
        # 使用字段名 'b' 对数组 x 进行排序
        x.sort(order=name)
        # 断言数组 x 中字段 'b' 的值为 [1, 2, 3]
        assert_equal(x['b'], np.array([1, 2, 3]))
    def test_field_names(self):
        # Test unicode and 8-bit / byte strings can be used
        # 创建一个包含一个元素的零数组，结构为一个字段为整数的元组和一个包含子字段的元组
        a = np.zeros((1,), dtype=[('f1', 'i4'),
                                  ('f2', 'i4'),
                                  ('f3', [('sf1', 'i4')])])
        # 使用字节字符串作为索引将会优雅地失败
        assert_raises(IndexError, a.__setitem__, b'f1', 1)
        assert_raises(IndexError, a.__getitem__, b'f1')
        assert_raises(IndexError, a['f1'].__setitem__, b'sf1', 1)
        assert_raises(IndexError, a['f1'].__getitem__, b'sf1')
        # 复制数组 a 到 b
        b = a.copy()
        fn1 = str('f1')
        # 修改字段 fn1 的值为 1，并验证是否相等
        b[fn1] = 1
        assert_equal(b[fn1], 1)
        fnn = str('not at all')
        # 尝试对不存在的字段进行设置和获取，应该会引发 ValueError
        assert_raises(ValueError, b.__setitem__, fnn, 1)
        assert_raises(ValueError, b.__getitem__, fnn)
        # 修改数组 b 中第一个元素的字段 fn1 的值为 2，并验证是否相等
        b[0][fn1] = 2
        assert_equal(b[fn1], 2)
        # 对子字段进行操作
        fn3 = str('f3')
        sfn1 = str('sf1')
        # 修改数组 b 中字段 fn3 的子字段 sfn1 的值为 1，并验证是否相等
        b[fn3][sfn1] = 1
        assert_equal(b[fn3][sfn1], 1)
        # 尝试对不存在的子字段进行设置和获取，应该会引发 ValueError
        assert_raises(ValueError, b[fn3].__setitem__, fnn, 1)
        assert_raises(ValueError, b[fn3].__getitem__, fnn)
        # 修改数组 b 中字段 fn2 的值为 3
        fn2 = str('f2')
        b[fn2] = 3

        # 验证取出多个字段的值并将其转换为列表是否正确
        assert_equal(b[['f1', 'f2']][0].tolist(), (2, 3))
        assert_equal(b[['f2', 'f1']][0].tolist(), (3, 2))
        assert_equal(b[['f1', 'f3']][0].tolist(), (2, (1,)))

        # 尝试对非 ASCII 字符的 Unicode 字段进行设置和获取，应该会引发 ValueError
        assert_raises(ValueError, a.__setitem__, '\u03e0', 1)
        assert_raises(ValueError, a.__getitem__, '\u03e0')

    def test_record_hash(self):
        # 创建包含元组的数组 a，元组中包含两个整数
        a = np.array([(1, 2), (1, 2)], dtype='i1,i2')
        a.flags.writeable = False
        # 创建包含命名字段的数组 b，字段为 num1 和 num2，各自为整数类型
        b = np.array([(1, 2), (3, 4)], dtype=[('num1', 'i1'), ('num2', 'i2')])
        b.flags.writeable = False
        # 创建包含命名字段的数组 c，字段为整数类型
        c = np.array([(1, 2), (3, 4)], dtype='i1,i2')
        c.flags.writeable = False
        # 验证哈希值是否相等，应该为 True
        assert_(hash(a[0]) == hash(a[1]))
        assert_(hash(a[0]) == hash(b[0]))
        # 验证哈希值是否不相等，应该为 True
        assert_(hash(a[0]) != hash(b[1]))
        # 验证哈希值和值是否相等，应该为 True
        assert_(hash(c[0]) == hash(a[0]) and c[0] == a[0])

    def test_record_no_hash(self):
        # 创建包含元组的数组 a，元组中包含两个整数
        a = np.array([(1, 2), (1, 2)], dtype='i1,i2')
        # 尝试对不可哈希对象进行哈希，应该会引发 TypeError
        assert_raises(TypeError, hash, a[0])

    def test_empty_structure_creation(self):
        # 确保以下操作不会引发错误 (gh-5631)
        # 创建一个包含空元组的数组，dtype 包含空的字段名、格式、偏移和项大小
        np.array([()], dtype={'names': [], 'formats': [],
                           'offsets': [], 'itemsize': 12})
        # 创建一个包含多个空元组的数组，dtype 包含空的字段名、格式、偏移和项大小
        np.array([(), (), (), (), ()], dtype={'names': [], 'formats': [],
                                           'offsets': [], 'itemsize': 12})
    # 定义一个测试方法，用于测试多字段索引视图
    def test_multifield_indexing_view(self):
        # 创建一个包含3个元素的数组，每个元素包含'a', 'b', 'c'三个字段，类型分别为整数、浮点数和无符号整数
        a = np.ones(3, dtype=[('a', 'i4'), ('b', 'f4'), ('c', 'u4')])
        
        # 通过多字段索引'a'和'c'创建一个新的视图v
        v = a[['a', 'c']]
        
        # 断言新视图v的基础对象是数组a本身
        assert_(v.base is a)
        
        # 断言新视图v的数据类型为指定的结构化数据类型，包括字段名为'a'和'c'，对应的格式为'i4'和'u4'，偏移量分别为0和8
        assert_(v.dtype == np.dtype({'names': ['a', 'c'],
                                     'formats': ['i4', 'u4'],
                                     'offsets': [0, 8]}))
        
        # 将新视图v的所有元素赋值为(4, 5)
        v[:] = (4, 5)
        
        # 断言数组a的第一个元素的值为(4, 1, 5)
        assert_equal(a[0].item(), (4, 1, 5))
class TestView:
    def test_basic(self):
        # 创建一个包含两个元组的 numpy 数组，每个元组有四个 np.int8 类型的字段
        x = np.array([(1, 2, 3, 4), (5, 6, 7, 8)],
                     dtype=[('r', np.int8), ('g', np.int8),
                            ('b', np.int8), ('a', np.int8)])
        # 将数组 x 转换为指定类型 '<i4' 的视图
        y = x.view(dtype='<i4')
        # 再次转换为 '<i4' 类型的视图，这次省略了 dtype 关键字
        z = x.view('<i4')
        # 断言两个视图 y 和 z 相等
        assert_array_equal(y, z)
        # 断言数组 y 的内容与预期值 [67305985, 134678021] 相等
        assert_array_equal(y, [67305985, 134678021])


def _mean(a, **args):
    # 计算数组 a 的均值，并返回结果
    return a.mean(**args)


def _var(a, **args):
    # 计算数组 a 的方差，并返回结果
    return a.var(**args)


def _std(a, **args):
    # 计算数组 a 的标准差，并返回结果
    return a.std(**args)


class TestStats:

    funcs = [_mean, _var, _std]

    def setup_method(self):
        # 设置随机种子为范围为 [0, 1, 2] 的整数数组
        np.random.seed(range(3))
        # 创建一个 4x5 的随机浮点数数组 rmat
        self.rmat = np.random.random((4, 5))
        # 创建一个与 rmat 形状相同的复数数组 cmat
        self.cmat = self.rmat + 1j * self.rmat
        # 将 rmat 的每个元素转换为 Decimal 类型，并将其扁平化成一维数组 omat
        self.omat = np.array([Decimal(str(r)) for r in self.rmat.flat])
        self.omat = self.omat.reshape(4, 5)

    def test_python_type(self):
        # 对不同类型的输入进行测试，确保计算结果符合预期
        for x in (np.float16(1.), 1, 1., 1+0j):
            assert_equal(np.mean([x]), 1.)
            assert_equal(np.std([x]), 0.)
            assert_equal(np.var([x]), 0.)

    def test_keepdims(self):
        # 测试保持维度参数 keepdims 在统计函数中的效果
        mat = np.eye(3)
        for f in self.funcs:
            for axis in [0, 1]:
                # 对矩阵 mat 在指定轴上应用函数 f，并保持维度
                res = f(mat, axis=axis, keepdims=True)
                # 断言返回结果的维度与原始矩阵一致
                assert_(res.ndim == mat.ndim)
                # 断言结果在指定轴上的长度为 1
                assert_(res.shape[axis] == 1)
            # 对 axis=None 的情况进行测试
            for axis in [None]:
                # 对矩阵 mat 在所有轴上应用函数 f，并保持维度
                res = f(mat, axis=axis, keepdims=True)
                # 断言返回结果的形状为 (1, 1)
                assert_(res.shape == (1, 1))

    def test_out(self):
        # 测试使用 out 参数来指定计算结果的存储位置
        mat = np.eye(3)
        for f in self.funcs:
            out = np.zeros(3)
            # 计算矩阵 mat 在指定轴上的函数值，并将结果存储到 out 中
            tgt = f(mat, axis=1)
            res = f(mat, axis=1, out=out)
            # 断言计算结果与预期的结果 out 相等
            assert_almost_equal(res, out)
            # 断言计算结果与预期的目标 tgt 相等
            assert_almost_equal(res, tgt)
        # 测试当 out 参数的形状与计算结果不匹配时引发 ValueError
        out = np.empty(2)
        assert_raises(ValueError, f, mat, axis=1, out=out)
        out = np.empty((2, 2))
        assert_raises(ValueError, f, mat, axis=1, out=out)
    # 定义一个测试方法，用于测试从输入数据推断数据类型的功能
    def test_dtype_from_input(self):

        # 获取所有整数类型的代码
        icodes = np.typecodes['AllInteger']
        # 获取所有浮点数类型的代码
        fcodes = np.typecodes['AllFloat']

        # object 类型的测试
        for f in self.funcs:
            # 创建一个包含 Decimal 数字的 3x3 矩阵
            mat = np.array([[Decimal(1)]*3]*3)
            # 目标数据类型为矩阵 mat 的数据类型
            tgt = mat.dtype.type
            # 调用函数 f 处理矩阵 mat，按行计算结果的数据类型
            res = f(mat, axis=1).dtype.type
            # 断言结果的数据类型与目标数据类型相同
            assert_(res is tgt)
            # 标量情况下的测试
            res = type(f(mat, axis=None))
            # 断言结果类型为 Decimal 类型
            assert_(res is Decimal)

        # 整数类型的测试
        for f in self.funcs:
            for c in icodes:
                # 创建一个以 c 类型为整数类型的 3x3 单位矩阵
                mat = np.eye(3, dtype=c)
                # 目标数据类型为 np.float64
                tgt = np.float64
                # 调用函数 f 处理矩阵 mat，按行计算结果的数据类型
                res = f(mat, axis=1).dtype.type
                # 断言结果的数据类型与目标数据类型相同
                assert_(res is tgt)
                # 标量情况下的测试
                res = f(mat, axis=None).dtype.type
                # 断言结果的数据类型与目标数据类型相同
                assert_(res is tgt)

        # 浮点数类型的均值测试
        for f in [_mean]:
            for c in fcodes:
                # 创建一个以 c 类型为浮点数类型的 3x3 单位矩阵
                mat = np.eye(3, dtype=c)
                # 目标数据类型为矩阵 mat 的数据类型
                tgt = mat.dtype.type
                # 调用函数 f 处理矩阵 mat，按行计算结果的数据类型
                res = f(mat, axis=1).dtype.type
                # 断言结果的数据类型与目标数据类型相同
                assert_(res is tgt)
                # 标量情况下的测试
                res = f(mat, axis=None).dtype.type
                # 断言结果的数据类型与目标数据类型相同
                assert_(res is tgt)

        # 浮点数类型的方差和标准差测试
        for f in [_var, _std]:
            for c in fcodes:
                # 创建一个以 c 类型为浮点数类型的 3x3 单位矩阵
                mat = np.eye(3, dtype=c)
                # 处理复数类型的情况，目标数据类型为矩阵实部的数据类型
                tgt = mat.real.dtype.type
                # 调用函数 f 处理矩阵 mat，按行计算结果的数据类型
                res = f(mat, axis=1).dtype.type
                # 断言结果的数据类型与目标数据类型相同
                assert_(res is tgt)
                # 标量情况下的测试
                res = f(mat, axis=None).dtype.type
                # 断言结果的数据类型与目标数据类型相同
                assert_(res is tgt)

    # 定义一个测试方法，用于测试从数据类型推断数据类型的功能
    def test_dtype_from_dtype(self):
        mat = np.eye(3)

        # 整数类型的统计测试
        # FIXME:
        # 此处需要定义，因为沿途可能存在多处类型转换的地方。

        # for f in self.funcs:
        #    for c in np.typecodes['AllInteger']:
        #        tgt = np.dtype(c).type
        #        res = f(mat, axis=1, dtype=c).dtype.type
        #        assert_(res is tgt)
        #        # 标量情况下的测试
        #        res = f(mat, axis=None, dtype=c).dtype.type
        #        assert_(res is tgt)

        # 浮点数类型的统计测试
        for f in self.funcs:
            for c in np.typecodes['AllFloat']:
                # 目标数据类型为 np.dtype(c).type
                tgt = np.dtype(c).type
                # 调用函数 f 处理矩阵 mat，按行计算结果的数据类型
                res = f(mat, axis=1, dtype=c).dtype.type
                # 断言结果的数据类型与目标数据类型相同
                assert_(res is tgt)
                # 标量情况下的测试
                res = f(mat, axis=None, dtype=c).dtype.type
                # 断言结果的数据类型与目标数据类型相同
                assert_(res is tgt)
    # 测试函数，验证不同自由度(ddof)对于方差和标准差计算的影响
    def test_ddof(self):
        # 遍历每个函数 `_var` 和 `_std`
        for f in [_var]:
            # 对于 ddof 在 0 到 2 之间的值进行循环
            for ddof in range(3):
                # 获取矩阵 `self.rmat` 的第二个维度大小
                dim = self.rmat.shape[1]
                # 计算目标值，使用函数 `f` 计算矩阵在 axis=1 上的结果，乘以维度大小
                tgt = f(self.rmat, axis=1) * dim
                # 计算实际结果，使用函数 `f` 计算矩阵在 axis=1 上的结果，并考虑自由度 ddof
                res = f(self.rmat, axis=1, ddof=ddof) * (dim - ddof)
        # 对于 `_std` 函数进行同样的循环验证
        for f in [_std]:
            for ddof in range(3):
                dim = self.rmat.shape[1]
                tgt = f(self.rmat, axis=1) * np.sqrt(dim)
                res = f(self.rmat, axis=1, ddof=ddof) * np.sqrt(dim - ddof)
                # 断言实际结果 `res` 与目标值 `tgt` 几乎相等
                assert_almost_equal(res, tgt)
                # 再次断言实际结果 `res` 与目标值 `tgt` 几乎相等
                assert_almost_equal(res, tgt)

    # 测试函数，验证当自由度(ddof)超出范围时的行为
    def test_ddof_too_big(self):
        # 获取矩阵 `self.rmat` 的第二个维度大小
        dim = self.rmat.shape[1]
        # 对于 `_var` 和 `_std` 函数进行循环验证
        for f in [_var, _std]:
            # 对于 ddof 超出 `dim` 至 `dim + 1` 范围的值进行循环
            for ddof in range(dim, dim + 2):
                # 捕获警告，确保发出运行时警告
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    # 计算函数 `f` 在 axis=1 上，ddof=ddof 的结果
                    res = f(self.rmat, axis=1, ddof=ddof)
                    # 断言结果中没有负数值
                    assert_(not (res < 0).any())
                    # 断言警告列表长度大于零
                    assert_(len(w) > 0)
                    # 断言第一个警告是 RuntimeWarning 的子类
                    assert_(issubclass(w[0].category, RuntimeWarning))

    # 测试函数，验证空矩阵时的函数行为
    def test_empty(self):
        # 创建一个零矩阵 `A`，形状为 (0, 3)
        A = np.zeros((0, 3))
        # 对于 `self.funcs` 中的每个函数进行循环验证
        for f in self.funcs:
            # 对于 axis 分别为 [0, None] 的情况进行循环
            for axis in [0, None]:
                # 捕获警告，确保发出运行时警告
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    # 断言函数 `f` 在给定 axis 上返回的结果都是 NaN
                    assert_(np.isnan(f(A, axis=axis)).all())
                    # 断言警告列表长度大于零
                    assert_(len(w) > 0)
                    # 断言第一个警告是 RuntimeWarning 的子类
                    assert_(issubclass(w[0].category, RuntimeWarning))
            # 对于 axis 为 1 的情况进行循环
            for axis in [1]:
                # 捕获警告，确保发出运行时警告
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    # 断言函数 `f` 在给定 axis 上返回的结果为一个空数组
                    assert_equal(f(A, axis=axis), np.zeros([]))

    # 测试函数，验证均值计算的正确性
    def test_mean_values(self):
        # 对于三个矩阵 `self.rmat`, `self.cmat`, `self.omat` 进行循环验证
        for mat in [self.rmat, self.cmat, self.omat]:
            # 对于 axis 分别为 [0, 1, None] 的情况进行循环
            for axis in [0, 1]:
                # 计算目标值，矩阵在给定 axis 上的和
                tgt = mat.sum(axis=axis)
                # 计算实际结果，使用 `_mean` 函数计算矩阵在给定 axis 上的均值，并乘以相应的维度大小或总元素数
                res = _mean(mat, axis=axis) * mat.shape[axis]
                # 断言实际结果 `res` 与目标值 `tgt` 几乎相等
                assert_almost_equal(res, tgt)
            # 对于 axis 为 None 的情况进行循环
            for axis in [None]:
                # 计算目标值，矩阵的总和
                tgt = mat.sum(axis=axis)
                # 计算实际结果，使用 `_mean` 函数计算矩阵的均值，并乘以总元素数
                res = _mean(mat, axis=axis) * np.prod(mat.shape)
                # 断言实际结果 `res` 与目标值 `tgt` 几乎相等
                assert_almost_equal(res, tgt)

    # 测试函数，验证在使用 float16 类型时均值计算的正确性
    def test_mean_float16(self):
        # 断言使用 `_mean` 函数计算 float16 类型数组全为 1 的均值为 1
        assert_(_mean(np.ones(100000, dtype='float16')) == 1)

    # 测试函数，验证当 axis 超出范围时是否引发 AxisError 而不是 IndexError
    def test_mean_axis_error(self):
        # 使用 `assert_raises` 断言在 axis 超出范围时引发 AxisError 异常
        with assert_raises(np.exceptions.AxisError):
            np.arange(10).mean(axis=2)
    # 定义测试函数 `test_mean_where`，用于测试 `mean` 方法和 `assert` 语句
    def test_mean_where(self):
        # 创建一个 4x4 的数组 a，其中包含 0 到 15 的连续整数
        a = np.arange(16).reshape((4, 4))
        
        # 创建一个布尔类型的数组 `wh_full`，表示在计算均值时哪些元素应包括
        wh_full = np.array([[False, True, False, True],
                            [True, False, True, False],
                            [True, True, False, False],
                            [False, False, True, True]])
        
        # 创建另一个布尔类型的数组 `wh_partial`，表示部分元素在计算均值时应包括
        wh_partial = np.array([[False],
                               [True],
                               [True],
                               [False]])
        
        # 创建一个包含多种测试用例的列表 `_cases`
        _cases = [(1, True, [1.5, 5.5, 9.5, 13.5]),
                  (0, wh_full, [6., 5., 10., 9.]),
                  (1, wh_full, [2., 5., 8.5, 14.5]),
                  (0, wh_partial, [6., 7., 8., 9.])]
        
        # 遍历 `_cases` 中的每个测试用例
        for _ax, _wh, _res in _cases:
            # 使用 `assert_allclose` 函数检查 `a.mean` 方法的返回结果是否与 `_res` 数组接近
            assert_allclose(a.mean(axis=_ax, where=_wh),
                            np.array(_res))
            # 使用 `assert_allclose` 函数检查 `np.mean` 函数的返回结果是否与 `_res` 数组接近
            assert_allclose(np.mean(a, axis=_ax, where=_wh),
                            np.array(_res))

        # 创建一个 3 维数组 `a3d`，其中包含 0 到 15 的连续整数，reshape 为 (2, 2, 4)
        a3d = np.arange(16).reshape((2, 2, 4))
        
        # 创建一个布尔类型的数组 `_wh_partial`，表示在计算均值时哪些元素应包括
        _wh_partial = np.array([False, True, True, False])
        
        # 创建一个预期结果数组 `_res`
        _res = [[1.5, 5.5], [9.5, 13.5]]
        
        # 使用 `assert_allclose` 函数检查 `a3d.mean` 方法的返回结果是否与 `_res` 数组接近
        assert_allclose(a3d.mean(axis=2, where=_wh_partial),
                        np.array(_res))
        
        # 使用 `assert_allclose` 函数检查 `np.mean` 函数的返回结果是否与 `_res` 数组接近
        assert_allclose(np.mean(a3d, axis=2, where=_wh_partial),
                        np.array(_res))

        # 使用 `pytest.warns` 捕获 RuntimeWarning，并在测试中断前检查是否如预期发出警告
        with pytest.warns(RuntimeWarning) as w:
            # 使用 `assert_allclose` 函数检查 `a.mean` 方法的返回结果是否与预期的数组接近
            assert_allclose(a.mean(axis=1, where=wh_partial),
                            np.array([np.nan, 5.5, 9.5, np.nan]))
        
        # 使用 `pytest.warns` 捕获 RuntimeWarning，并在测试中断前检查是否如预期发出警告
        with pytest.warns(RuntimeWarning) as w:
            # 使用 `assert_equal` 函数检查 `a.mean` 方法的返回结果是否等于预期的 np.nan
            assert_equal(a.mean(where=False), np.nan)
        
        # 使用 `pytest.warns` 捕获 RuntimeWarning，并在测试中断前检查是否如预期发出警告
        with pytest.warns(RuntimeWarning) as w:
            # 使用 `assert_equal` 函数检查 `np.mean` 函数的返回结果是否等于预期的 np.nan
            assert_equal(np.mean(a, where=False), np.nan)

    # 定义测试函数 `test_var_values`，用于测试 `_var` 函数的计算结果是否正确
    def test_var_values(self):
        # 遍历包含 `self.rmat`, `self.cmat`, `self.omat` 的列表，依次为每个矩阵执行测试
        for mat in [self.rmat, self.cmat, self.omat]:
            # 遍历坐标轴参数 [0, 1, None]，依次为每个坐标轴执行测试
            for axis in [0, 1, None]:
                # 计算矩阵 `mat` 元素的平方和，并求均值，得到 `msqr`
                msqr = _mean(mat * mat.conj(), axis=axis)
                # 计算矩阵 `mat` 元素的均值，得到 `mean`
                mean = _mean(mat, axis=axis)
                # 计算矩阵 `mat` 的方差，得到 `tgt`
                tgt = msqr - mean * mean.conjugate()
                # 调用 `_var` 函数计算矩阵 `mat` 的方差，并与 `tgt` 进行比较
                res = _var(mat, axis=axis)
                # 使用 `assert_almost_equal` 函数检查 `res` 是否与 `tgt` 接近
                assert_almost_equal(res, tgt)

    # 使用 `pytest.mark.parametrize` 装饰器设置参数化测试，用于测试复数数据类型的方差计算
    @pytest.mark.parametrize(('complex_dtype', 'ndec'), (
        ('complex64', 6),
        ('complex128', 7),
        ('clongdouble', 7),
    ))
    # 定义测试函数 `test_var_complex_values`，用于测试复数数据类型的方差计算
    def test_var_complex_values(self, complex_dtype, ndec):
        # 遍历坐标轴参数 [0, 1, None]，依次为每个坐标轴执行测试
        for axis in [0, 1, None]:
            # 复制 `self.cmat` 矩阵并将数据类型转换为指定的 `complex_dtype`
            mat = self.cmat.copy().astype(complex_dtype)
            # 计算矩阵 `mat` 元素的平方和，并求均值，得到 `msqr`
            msqr = _mean(mat * mat.conj(), axis=axis)
            # 计算矩阵 `mat` 元素的均值，得到 `mean`
            mean = _mean(mat, axis=axis)
            # 计算矩阵 `mat` 的方差，得到 `tgt`
            tgt = msqr - mean * mean.conjugate()
            # 调用 `_var` 函数计算矩阵 `mat` 的方差，并与 `tgt` 进行比较
            res = _var(mat, axis=axis)
            # 使用 `assert_almost_equal` 函数检查 `res` 是否与 `tgt` 接近，指定精度为 `ndec`
            assert_almost_equal(res, tgt, decimal=ndec)
    def test_var_dimensions(self):
        # 测试复数变量路径对视图引入增加维度的影响。确保这适用于更高的维度。
        mat = np.stack([self.cmat]*3)  # 创建一个包含三个副本的数组堆栈
        for axis in [0, 1, 2, -1, None]:
            msqr = _mean(mat * mat.conj(), axis=axis)  # 计算平方后的平均值
            mean = _mean(mat, axis=axis)  # 计算平均值
            tgt = msqr - mean * mean.conjugate()  # 计算方差
            res = _var(mat, axis=axis)  # 计算变量（方差）
            assert_almost_equal(res, tgt)  # 断言结果与目标值几乎相等

    def test_var_complex_byteorder(self):
        # 测试复杂数组在非本机字节顺序下，变量快速路径不会导致失败
        cmat = self.cmat.copy().astype('complex128')  # 复制并转换为复数类型
        cmat_swapped = cmat.astype(cmat.dtype.newbyteorder())  # 转换为新的字节顺序
        assert_almost_equal(cmat.var(), cmat_swapped.var())  # 断言两者的方差几乎相等

    def test_var_axis_error(self):
        # 确保当轴超出范围时，引发 AxisError 而不是 IndexError，参见 gh-15817
        with assert_raises(np.exceptions.AxisError):
            np.arange(10).var(axis=2)  # 尝试在轴超出范围时计算方差，预期引发异常

    def test_var_where(self):
        a = np.arange(25).reshape((5, 5))  # 创建一个5x5的数组
        wh_full = np.array([[False, True, False, True, True],  # 定义完全条件的布尔数组
                            [True, False, True, True, False],
                            [True, True, False, False, True],
                            [False, True, True, False, True],
                            [True, False, True, True, False]])
        wh_partial = np.array([[False],  # 定义部分条件的布尔数组
                               [True],
                               [True],
                               [False],
                               [True]])
        _cases = [(0, True, [50., 50., 50., 50., 50.]),  # 定义用于测试的轴和条件组合
                  (1, True, [2., 2., 2., 2., 2.])]
        for _ax, _wh, _res in _cases:
            assert_allclose(a.var(axis=_ax, where=_wh),  # 断言计算的方差与预期结果几乎相等
                            np.array(_res))
            assert_allclose(np.var(a, axis=_ax, where=_wh),  # 断言计算的全局方差与预期结果几乎相等
                            np.array(_res))

        a3d = np.arange(16).reshape((2, 2, 4))  # 创建一个2x2x4的三维数组
        _wh_partial = np.array([False, True, True, False])  # 定义部分条件的布尔数组
        _res = [[0.25, 0.25], [0.25, 0.25]]  # 预期的结果数组
        assert_allclose(a3d.var(axis=2, where=_wh_partial),  # 断言计算的三维方差与预期结果几乎相等
                        np.array(_res))
        assert_allclose(np.var(a3d, axis=2, where=_wh_partial),  # 断言计算的三维全局方差与预期结果几乎相等
                        np.array(_res))

        assert_allclose(np.var(a, axis=1, where=wh_full),  # 断言计算的行方差与预期结果几乎相等
                        np.var(a[wh_full].reshape((5, 3)), axis=1))
        assert_allclose(np.var(a, axis=0, where=wh_partial),  # 断言计算的列方差与预期结果几乎相等
                        np.var(a[wh_partial[:,0]], axis=0))
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.var(where=False), np.nan)  # 断言在条件全部为假时，计算结果为 NaN
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.var(a, where=False), np.nan)  # 断言在条件全部为假时，计算结果为 NaN
    # 测试函数，用于验证 _std 函数对于标准值计算的正确性
    def test_std_values(self):
        # 对于给定的矩阵列表 [self.rmat, self.cmat, self.omat]，分别在指定的轴或全局计算标准差
        for mat in [self.rmat, self.cmat, self.omat]:
            # 对于轴参数为 0, 1 或 None，分别计算期望的标准差
            for axis in [0, 1, None]:
                # 目标值为使用 numpy 的 _var 函数计算的标准差的平方根
                tgt = np.sqrt(_var(mat, axis=axis))
                # 计算实际的标准差值
                res = _std(mat, axis=axis)
                # 使用 numpy 的 assert_almost_equal 函数断言结果 res 应接近于目标值 tgt
                assert_almost_equal(res, tgt)

    # 测试函数，用于验证在指定条件下 _std 函数的行为
    def test_std_where(self):
        # 创建一个 5x5 的矩阵 a，其中元素从 24 到 0 递减填充
        a = np.arange(25).reshape((5,5))[::-1]
        # 创建布尔掩码数组，指定 a 中的条件
        whf = np.array([[False, True, False, True, True],
                        [True, False, True, False, True],
                        [True, True, False, True, False],
                        [True, False, True, True, False],
                        [False, True, False, True, True]])
        # 创建布尔掩码数组，指定 a 中的条件
        whp = np.array([[False],
                        [False],
                        [True],
                        [True],
                        [False]])
        # 定义测试用例列表，每个元组包含轴、条件和期望的标准差值数组
        _cases = [
            (0, True, 7.07106781*np.ones((5))),
            (1, True, 1.41421356*np.ones((5))),
            (0, whf,
             np.array([4.0824829 , 8.16496581, 5., 7.39509973, 8.49836586])),
            (0, whp, 2.5*np.ones((5)))
        ]
        # 对于每个测试用例，分别使用 numpy 的 assert_allclose 断言 _std 和 np.std 的结果应接近于期望的结果
        for _ax, _wh, _res in _cases:
            assert_allclose(a.std(axis=_ax, where=_wh), _res)
            assert_allclose(np.std(a, axis=_ax, where=_wh), _res)

        # 创建一个三维数组 a3d
        a3d = np.arange(16).reshape((2, 2, 4))
        # 创建部分布尔掩码数组 _wh_partial
        _wh_partial = np.array([False, True, True, False])
        # 期望的标准差结果数组 _res
        _res = [[0.5, 0.5], [0.5, 0.5]]
        # 使用 assert_allclose 断言 a3d 在指定轴和条件下的标准差应接近于 _res
        assert_allclose(a3d.std(axis=2, where=_wh_partial),
                        np.array(_res))
        assert_allclose(np.std(a3d, axis=2, where=_wh_partial),
                        np.array(_res))

        # 使用 assert_allclose 断言 a 在指定条件下的标准差应接近于将符合条件的元素重新排列后的标准差
        assert_allclose(a.std(axis=1, where=whf),
                        np.std(a[whf].reshape((5,3)), axis=1))
        assert_allclose(np.std(a, axis=1, where=whf),
                        (a[whf].reshape((5,3))).std(axis=1))
        # 使用 assert_allclose 断言 a 在指定条件下的标准差应接近于将符合条件的元素重新排列后的标准差
        assert_allclose(a.std(axis=0, where=whp),
                        np.std(a[whp[:,0]], axis=0))
        assert_allclose(np.std(a, axis=0, where=whp),
                        (a[whp[:,0]]).std(axis=0))
        # 使用 pytest 的 warn 语句检查运行时警告，并断言调用 a.std(where=False) 和 np.std(a, where=False) 应返回 NaN
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(a.std(where=False), np.nan)
        with pytest.warns(RuntimeWarning) as w:
            assert_equal(np.std(a, where=False), np.nan)

    # 测试函数，用于验证自定义子类 TestArray 的扩展功能
    def test_subclass(self):
        # 定义一个继承自 np.ndarray 的子类 TestArray
        class TestArray(np.ndarray):
            # 自定义 __new__ 方法，接收 data 和 info 参数，返回带有附加信息的 ndarray
            def __new__(cls, data, info):
                result = np.array(data)
                result = result.view(cls)
                result.info = info
                return result

            # 自定义 __array_finalize__ 方法，处理 ndarray 的继承和附加信息
            def __array_finalize__(self, obj):
                self.info = getattr(obj, "info", '')

        # 创建一个 TestArray 实例 dat
        dat = TestArray([[1, 2, 3, 4], [5, 6, 7, 8]], 'jubba')
        # 调用 mean 方法计算均值，断言结果的附加信息与 dat 的一致
        res = dat.mean(1)
        assert_(res.info == dat.info)
        # 调用 std 方法计算标准差，断言结果的附加信息与 dat 的一致
        res = dat.std(1)
        assert_(res.info == dat.info)
        # 调用 var 方法计算方差，断言结果的附加信息与 dat 的一致
        res = dat.var(1)
        assert_(res.info == dat.info)
class TestVdot:
    def test_basic(self):
        # 定义包含所有数值类型和整数类型的数据类型字符串
        dt_numeric = np.typecodes['AllFloat'] + np.typecodes['AllInteger']
        # 定义包含复数类型的数据类型字符串
        dt_complex = np.typecodes['Complex']

        # 测试实数类型
        a = np.eye(3)
        for dt in dt_numeric + 'O':
            # 将矩阵 a 转换为指定数据类型 dt 的数组 b
            b = a.astype(dt)
            # 计算数组 b 与自身的向量点积
            res = np.vdot(b, b)
            # 断言 res 是标量
            assert_(np.isscalar(res))
            # 断言向量点积的结果与预期值 3 相等
            assert_equal(np.vdot(b, b), 3)

        # 测试复数类型
        a = np.eye(3) * 1j
        for dt in dt_complex + 'O':
            # 将矩阵 a 转换为指定数据类型 dt 的数组 b
            b = a.astype(dt)
            # 计算数组 b 与自身的向量点积
            res = np.vdot(b, b)
            # 断言 res 是标量
            assert_(np.isscalar(res))
            # 断言向量点积的结果与预期值 3 相等
            assert_equal(np.vdot(b, b), 3)

        # 测试布尔类型
        b = np.eye(3, dtype=bool)
        # 计算布尔数组 b 与自身的向量点积
        res = np.vdot(b, b)
        # 断言 res 是标量
        assert_(np.isscalar(res))
        # 断言向量点积的结果与预期值 True 相等
        assert_equal(np.vdot(b, b), True)

    def test_vdot_array_order(self):
        # 创建两个不同存储顺序（'C' 和 'F'）的数组 a 和 b
        a = np.array([[1, 2], [3, 4]], order='C')
        b = np.array([[1, 2], [3, 4]], order='F')
        # 计算数组 a 与自身的向量点积
        res = np.vdot(a, a)

        # 整数数组是精确的
        # 断言不同存储顺序下的数组点积与 res 相等
        assert_equal(np.vdot(a, b), res)
        assert_equal(np.vdot(b, a), res)
        assert_equal(np.vdot(b, b), res)

    def test_vdot_uncontiguous(self):
        for size in [2, 1000]:
            # 不同大小的数组匹配 vdot 中的不同分支
            # 创建大小为 (size, 2, 2) 的零矩阵 a 和 b
            a = np.zeros((size, 2, 2))
            b = np.zeros((size, 2, 2))
            # 设置 a 和 b 的部分元素值
            a[:, 0, 0] = np.arange(size)
            b[:, 0, 0] = np.arange(size) + 1
            # 使 a 和 b 不连续
            a = a[..., 0]
            b = b[..., 0]

            # 断言不连续数组 a 和 b 的向量点积与展开后的点积结果相等
            assert_equal(np.vdot(a, b),
                         np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a, b.copy()),
                         np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a.copy(), b),
                         np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a.copy('F'), b),
                         np.vdot(a.flatten(), b.flatten()))
            assert_equal(np.vdot(a, b.copy('F')),
                         np.vdot(a.flatten(), b.flatten()))
    # 定义一个测试方法，计算矩阵 A 与向量 b2 的点积，并进行断言检查结果的近似性
    def test_dotmatvec2(self):
        # 从当前对象获取矩阵 A 和向量 b2
        A, b2 = self.A, self.b2
        # 计算矩阵 A 与向量 b2 的点积
        res = np.dot(A, b2)
        # 预期的点积结果
        tgt = np.array([0.29677940, 0.04518649, 0.14468333, 0.31039293])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算向量 b4 与矩阵 A 的点积，并进行断言检查结果的近似性
    def test_dotvecmat(self):
        # 从当前对象获取矩阵 A 和向量 b4
        A, b4 = self.A, self.b4
        # 计算向量 b4 与矩阵 A 的点积
        res = np.dot(b4, A)
        # 预期的点积结果
        tgt = np.array([1.23495091, 1.12222648])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算向量 b3 与矩阵 A 转置的点积，并进行断言检查结果的近似性
    def test_dotvecmat2(self):
        # 从当前对象获取向量 b3 和矩阵 A
        b3, A = self.b3, self.A
        # 计算向量 b3 与矩阵 A 转置的点积
        res = np.dot(b3, A.transpose())
        # 预期的点积结果
        tgt = np.array([[0.58793804, 0.08957460, 0.30605758, 0.62716383]])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算矩阵 A 转置与向量 b4 的点积，并进行断言检查结果的近似性
    def test_dotvecmat3(self):
        # 从当前对象获取矩阵 A 和向量 b4
        A, b4 = self.A, self.b4
        # 计算矩阵 A 转置与向量 b4 的点积
        res = np.dot(A.transpose(), b4)
        # 预期的点积结果
        tgt = np.array([1.23495091, 1.12222648])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算向量 b1 与向量 b3 的外积，并进行断言检查结果的近似性
    def test_dotvecvecouter(self):
        # 从当前对象获取向量 b1 和向量 b3
        b1, b3 = self.b1, self.b3
        # 计算向量 b1 与向量 b3 的外积
        res = np.dot(b1, b3)
        # 预期的外积结果
        tgt = np.array([[0.20128610, 0.08400440], [0.07190947, 0.03001058]])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算向量 b3 与向量 b1 的内积，并进行断言检查结果的近似性
    def test_dotvecvecinner(self):
        # 从当前对象获取向量 b1 和向量 b3
        b1, b3 = self.b1, self.b3
        # 计算向量 b3 与向量 b1 的内积
        res = np.dot(b3, b1)
        # 预期的内积结果
        tgt = np.array([[0.23129668]])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算矩阵 b1 与标量 b2 的乘积，并进行断言检查结果的近似性
    def test_dotcolumnvect1(self):
        # 创建一个形状为 (3, 1) 的全一矩阵 b1
        b1 = np.ones((3, 1))
        # 创建一个标量 b2
        b2 = [5.3]
        # 计算矩阵 b1 与标量 b2 的乘积
        res = np.dot(b1, b2)
        # 预期的乘积结果
        tgt = np.array([5.3, 5.3, 5.3])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算标量 b2 与矩阵 b1 的乘积，并进行断言检查结果的近似性
    def test_dotcolumnvect2(self):
        # 创建一个形状为 (1, 3) 的全一矩阵 b1
        b1 = np.ones((3, 1)).transpose()
        # 创建一个标量 b2
        b2 = [6.2]
        # 计算标量 b2 与矩阵 b1 的乘积
        res = np.dot(b2, b1)
        # 预期的乘积结果
        tgt = np.array([6.2, 6.2, 6.2])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算向量 b1 与矩阵 b2 的乘积，并进行断言检查结果的近似性
    def test_dotvecscalar(self):
        # 设定随机种子为 100
        np.random.seed(100)
        # 创建一个形状为 (1, 1) 的随机矩阵 b1
        b1 = np.random.rand(1, 1)
        # 创建一个形状为 (1, 4) 的随机矩阵 b2
        b2 = np.random.rand(1, 4)
        # 计算向量 b1 与矩阵 b2 的乘积
        res = np.dot(b1, b2)
        # 预期的乘积结果
        tgt = np.array([[0.15126730, 0.23068496, 0.45905553, 0.00256425]])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，计算矩阵 b1 与标量 b2 的乘积，并进行断言检查结果的近似性
    def test_dotvecscalar2(self):
        # 设定随机种子为 100
        np.random.seed(100)
        # 创建一个形状为 (4, 1) 的随机矩阵 b1
        b1 = np.random.rand(4, 1)
        # 创建一个形状为 (1, 1) 的随机矩阵 b2
        b2 = np.random.rand(1, 1)
        # 计算矩阵 b1 与标量 b2 的乘积
        res = np.dot(b1, b2)
        # 预期的乘积结果
        tgt = np.array([[0.00256425],[0.00131359],[0.00200324],[ 0.00398638]])
        # 断言检查计算结果与预期结果的近似性
        assert_almost_equal(res, tgt, decimal=self.N)

    # 定义一个测试方法，
    def test_vecobject(self):
        # 定义一个名为 Vec 的类，用于处理向量操作
        class Vec:
            def __init__(self, sequence=None):
                # 初始化函数，如果没有传入序列，则默认为空列表
                if sequence is None:
                    sequence = []
                # 将序列转换为 NumPy 数组，并赋值给实例变量 array
                self.array = np.array(sequence)

            def __add__(self, other):
                # 向量加法重载，创建一个新的 Vec 对象 out，并计算其数组为当前对象数组与另一个对象数组的和
                out = Vec()
                out.array = self.array + other.array
                return out

            def __sub__(self, other):
                # 向量减法重载，创建一个新的 Vec 对象 out，并计算其数组为当前对象数组与另一个对象数组的差
                out = Vec()
                out.array = self.array - other.array
                return out

            def __mul__(self, other):  # with scalar
                # 向量乘法重载（与标量相乘），创建一个新的 Vec 对象 out，并将当前对象数组乘以标量 other
                out = Vec(self.array.copy())
                out.array *= other
                return out

            def __rmul__(self, other):
                # 右乘法重载（反向乘法），即标量与向量的乘法
                return self * other

        # 创建一个非连续的数组 U_non_cont，通过转置来初始化
        U_non_cont = np.transpose([[1., 1.], [1., 2.]])
        # 将非连续数组转换为连续数组 U_cont
        U_cont = np.ascontiguousarray(U_non_cont)
        # 创建一个包含 Vec 对象的数组 x，每个 Vec 对象初始化为不同的向量
        x = np.array([Vec([1., 0.]), Vec([0., 1.])])
        # 创建一个包含 Vec 对象的数组 zeros，每个 Vec 对象初始化为零向量
        zeros = np.array([Vec([0., 0.]), Vec([0., 0.])])
        # 计算 U_cont 与 x 的矩阵乘法减去 U_non_cont 与 x 的矩阵乘法，结果存储在 zeros_test 中
        zeros_test = np.dot(U_cont, x) - np.dot(U_non_cont, x)
        # 断言第一个 Vec 对象的数组与 zeros_test 的第一个元素的数组相等
        assert_equal(zeros[0].array, zeros_test[0].array)
        # 断言第二个 Vec 对象的数组与 zeros_test 的第二个元素的数组相等
        assert_equal(zeros[1].array, zeros_test[1].array)

    def test_dot_2args(self):
        # 创建两个二维浮点数数组 a, b
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[1, 0], [1, 1]], dtype=float)
        # 创建期望的结果数组 c
        c = np.array([[3, 2], [7, 4]], dtype=float)
        # 计算矩阵乘法 a 和 b，并存储在 d 中
        d = dot(a, b)
        # 断言 d 与 c 数组内容近似相等
        assert_allclose(c, d)

    def test_dot_3args(self):
        # 设定随机数种子为 22
        np.random.seed(22)
        # 创建两个随机浮点数数组 f, v
        f = np.random.random_sample((1024, 16))
        v = np.random.random_sample((16, 32))
        # 创建一个空的结果数组 r
        r = np.empty((1024, 32))
        # 多次调用 dot 函数，每次将结果存储在 r 中
        for i in range(12):
            dot(f, v, r)
        # 如果支持引用计数功能，断言 r 的引用计数为 2
        if HAS_REFCOUNT:
            assert_equal(sys.getrefcount(r), 2)
        # 再次调用 dot 函数，将结果存储在 r2 中，然后断言 r2 与 r 相等
        r2 = dot(f, v, out=None)
        assert_array_equal(r2, r)
        # 断言 r 和 dot 函数返回的结果是同一个对象
        assert_(r is dot(f, v, out=r))

        # 将 v 的第一列复制给 v，使其变成一维数组
        v = v[:, 0].copy()  # v.shape == (16,)
        # 将 r 的第一列复制给 r，使其变成一维数组
        r = r[:, 0].copy()  # r.shape == (1024,)
        # 再次调用 dot 函数，将结果存储在 r2 中
        r2 = dot(f, v)
        # 断言 r 和 dot 函数返回的结果是同一个对象
        assert_(r is dot(f, v, r))
        # 断言 r2 与 r 相等
        assert_array_equal(r2, r)

    def test_dot_3args_errors(self):
        # 设定随机数种子为 22
        np.random.seed(22)
        # 创建两个随机浮点数数组 f, v
        f = np.random.random_sample((1024, 16))
        v = np.random.random_sample((16, 32))

        # 创建一个形状不匹配的结果数组 r，并断言调用 dot 函数会引发 ValueError 异常
        r = np.empty((1024, 31))
        assert_raises(ValueError, dot, f, v, r)

        # 创建一个形状不匹配的结果数组 r，并断言调用 dot 函数会引发 ValueError 异常
        r = np.empty((1024,))
        assert_raises(ValueError, dot, f, v, r)

        # 创建一个形状不匹配的结果数组 r，并断言调用 dot 函数会引发 ValueError 异常
        r = np.empty((32,))
        assert_raises(ValueError, dot, f, v, r)

        # 创建一个形状不匹配的结果数组 r，并断言调用 dot 函数会引发 ValueError 异常
        r = np.empty((32, 1024))
        assert_raises(ValueError, dot, f, v, r)
        # 断言调用 dot 函数传入 r 的转置会引发 ValueError 异常
        assert_raises(ValueError, dot, f, v, r.T)

        # 创建一个形状不匹配的结果数组 r，并断言调用 dot 函数会引发 ValueError 异常
        r = np.empty((1024, 64))
        assert_raises(ValueError, dot, f, v, r[:, ::2])
        # 断言调用 dot 函数传入的 r 的前 32 列会引发 ValueError 异常
        assert_raises(ValueError, dot, f, v, r[:, :32])

        # 创建一个形状匹配但数据类型不匹配的结果数组 r，并断言调用 dot 函数会引发 ValueError 异常
        r = np.empty((1024, 32), dtype=np.float32)
        assert_raises(ValueError, dot, f, v, r)

        # 创建一个形状匹配但数据类型不匹配的结果数组 r，并断言调用 dot 函数会引发 ValueError 异常
        r = np.empty((1024, 32), dtype=int)
        assert_raises(ValueError, dot, f, v, r)
    # 定义一个测试方法，测试 np.dot 函数的 out 参数的使用
    def test_dot_out_result(self):
        # 创建一个单元素数组 x，数据类型为 np.float16
        x = np.ones((), dtype=np.float16)
        # 创建一个长度为 5 的全为 1 的数组 y，数据类型为 np.float16
        y = np.ones((5,), dtype=np.float16)
        # 创建一个长度为 5 的全为 0 的数组 z，数据类型为 np.float16
        z = np.zeros((5,), dtype=np.float16)
        # 使用 x 和 y 计算点积，将结果存储在 z 中
        res = x.dot(y, out=z)
        # 断言 res 与 y 数组完全相等
        assert np.array_equal(res, y)
        # 断言 z 数组与 y 数组完全相等
        assert np.array_equal(z, y)

    # 定义一个测试方法，测试 np.dot 函数的 out 参数在别名情况下的行为
    def test_dot_out_aliasing(self):
        # 创建一个单元素数组 x，数据类型为 np.float16
        x = np.ones((), dtype=np.float16)
        # 创建一个长度为 5 的全为 1 的数组 y，数据类型为 np.float16
        y = np.ones((5,), dtype=np.float16)
        # 创建一个长度为 5 的全为 0 的数组 z，数据类型为 np.float16
        z = np.zeros((5,), dtype=np.float16)
        # 使用 x 和 y 计算点积，将结果存储在 z 中
        res = x.dot(y, out=z)
        # 修改 z 中的第一个元素为 2
        z[0] = 2
        # 断言 res 与 z 数组完全相等
        assert np.array_equal(res, z)

    # 定义一个测试方法，测试 np.dot 函数在不同数组顺序下的结果
    def test_dot_array_order(self):
        # 创建一个按行存储的 2x2 数组 a
        a = np.array([[1, 2], [3, 4]], order='C')
        # 创建一个按列存储的 2x2 数组 b
        b = np.array([[1, 2], [3, 4]], order='F')
        # 计算 a 和 a 的点积
        res = np.dot(a, a)

        # 断言 np.dot(a, b) 的结果与 res 相等
        assert_equal(np.dot(a, b), res)
        # 断言 np.dot(b, a) 的结果与 res 相等
        assert_equal(np.dot(b, a), res)
        # 断言 np.dot(b, b) 的结果与 res 相等
        assert_equal(np.dot(b, b), res)
    def test_accelerate_framework_sgemv_fix(self):
        # 定义测试函数，用于验证加速框架中的sgemv修复

        def aligned_array(shape, align, dtype, order='C'):
            # 创建指定形状、对齐方式、数据类型和顺序的数组
            d = dtype(0)
            N = np.prod(shape)
            tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
            address = tmp.__array_interface__["data"][0]
            for offset in range(align):
                if (address + offset) % align == 0:
                    break
            tmp = tmp[offset:offset+N*d.nbytes].view(dtype=dtype)
            return tmp.reshape(shape, order=order)

        def as_aligned(arr, align, dtype, order='C'):
            # 将给定数组对齐到指定的边界
            aligned = aligned_array(arr.shape, align, dtype, order)
            aligned[:] = arr[:]
            return aligned

        def assert_dot_close(A, X, desired):
            # 断言两个矩阵相乘的结果接近于期望值
            assert_allclose(np.dot(A, X), desired, rtol=1e-5, atol=1e-7)

        m = aligned_array(100, 15, np.float32)
        s = aligned_array((100, 100), 15, np.float32)
        np.dot(s, m)  # 如果存在bug，这里将始终导致段错误

        testdata = itertools.product((15, 32), (10000,), (200, 89), ('C', 'F'))
        for align, m, n, a_order in testdata:
            # 对于每组测试数据，执行以下操作
            # 在双精度下计算
            A_d = np.random.rand(m, n)
            X_d = np.random.rand(n)
            desired = np.dot(A_d, X_d)
            
            # 使用对齐的单精度计算
            A_f = as_aligned(A_d, align, np.float32, order=a_order)
            X_f = as_aligned(X_d, align, np.float32)
            assert_dot_close(A_f, X_f, desired)
            
            # 在A的行上进行跨步操作
            A_d_2 = A_d[::2]
            desired = np.dot(A_d_2, X_d)
            A_f_2 = A_f[::2]
            assert_dot_close(A_f_2, X_f, desired)
            
            # 在A的列和X向量上进行跨步操作
            A_d_22 = A_d_2[:, ::2]
            X_d_2 = X_d[::2]
            desired = np.dot(A_d_22, X_d_2)
            A_f_22 = A_f_2[:, ::2]
            X_f_2 = X_f[::2]
            assert_dot_close(A_f_22, X_f_2, desired)
            
            # 检查所期望的步幅
            if a_order == 'F':
                assert_equal(A_f_22.strides, (8, 8 * m))
            else:
                assert_equal(A_f_22.strides, (8 * n, 8))
            assert_equal(X_f_2.strides, (8,))
            
            # 在A的行和列上进行跨步操作
            X_f_2c = as_aligned(X_f_2, align, np.float32)
            assert_dot_close(A_f_22, X_f_2c, desired)
            
            # 仅在A的列上进行跨步操作
            A_d_12 = A_d[:, ::2]
            desired = np.dot(A_d_12, X_d_2)
            A_f_12 = A_f[:, ::2]
            assert_dot_close(A_f_12, X_f_2c, desired)
            
            # 在A的列和X向量上进行跨步操作
            assert_dot_close(A_f_12, X_f_2, desired)

    @pytest.mark.slow
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @requires_memory(free_bytes=18e9)  # 复杂情况需要18GiB以上的内存
    # 定义一个测试函数，用于测试大向量点积的情况，传入数据类型参数dtype
    def test_huge_vectordot(self, dtype):
        # 大向量乘法使用32位BLAS进行分块处理
        # 测试分块处理是否正确，参见gh-22262
        # 创建一个包含2^30+100个元素的全为1的数组，指定数据类型为dtype
        data = np.ones(2**30+100, dtype=dtype)
        # 计算数组data与自身的点积
        res = np.dot(data, data)
        # 断言点积的结果应为2^30+100
        assert res == 2**30+100

    # 定义一个测试函数，测试dtype发现失败的情况
    def test_dtype_discovery_fails(self):
        # 参见gh-14247，对于dtype发现失败时缺少错误检查
        # 定义一个BadObject类，实现__array__方法抛出TypeError异常
        class BadObject(object):
            def __array__(self, dtype=None, copy=None):
                raise TypeError("just this tiny mint leaf")

        # 使用pytest断言捕获到TypeError异常
        with pytest.raises(TypeError):
            np.dot(BadObject(), BadObject())

        # 使用pytest断言捕获到TypeError异常
        with pytest.raises(TypeError):
            np.dot(3.0, BadObject())
# 定义一个类 MatmulCommon，用于测试 '@' 操作符和 numpy.matmul 函数的通用性
class MatmulCommon:
    """Common tests for '@' operator and numpy.matmul."""

    # 可用的数据类型列表，包括布尔、整数、浮点数和复数类型
    types = "?bhilqBHILQefdgFDGO"

    # 测试异常情况的方法
    def test_exceptions(self):
        # 不匹配的维度对，用来测试是否抛出 ValueError 异常
        dims = [
            ((1,), (2,)),            # mismatched vector vector
            ((2, 1,), (2,)),         # mismatched matrix vector
            ((2,), (1, 2)),          # mismatched vector matrix
            ((1, 2), (3, 1)),        # mismatched matrix matrix
            ((1,), ()),              # vector scalar
            ((), (1)),               # scalar vector
            ((1, 1), ()),            # matrix scalar
            ((), (1, 1)),            # scalar matrix
            ((2, 2, 1), (3, 1, 2)),  # cannot broadcast
        ]

        # 对每种数据类型和维度对进行迭代测试
        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            # 断言是否抛出 ValueError 异常
            assert_raises(ValueError, self.matmul, a, b)

    # 测试结果形状的方法
    def test_shapes(self):
        # 不同维度对，用来测试矩阵乘法的形状
        dims = [
            ((1, 1), (2, 1, 1)),     # broadcast first argument
            ((2, 1, 1), (1, 1)),     # broadcast second argument
            ((2, 1, 1), (2, 1, 1)),  # matrix stack sizes match
        ]

        # 对每种数据类型和维度对进行迭代测试
        for dt, (dm1, dm2) in itertools.product(self.types, dims):
            a = np.ones(dm1, dtype=dt)
            b = np.ones(dm2, dtype=dt)
            # 执行矩阵乘法
            res = self.matmul(a, b)
            # 断言结果的形状是否为 (2, 1, 1)
            assert_(res.shape == (2, 1, 1))

        # 向量乘以向量应返回标量
        for dt in self.types:
            a = np.ones((2,), dtype=dt)
            b = np.ones((2,), dtype=dt)
            c = self.matmul(a, b)
            # 断言结果的形状为 ()
            assert_(np.array(c).shape == ())

    # 测试结果类型的方法
    def test_result_types(self):
        mat = np.ones((1,1))
        vec = np.ones((1,))
        # 对每种数据类型进行迭代测试
        for dt in self.types:
            m = mat.astype(dt)
            v = vec.astype(dt)
            # 对矩阵乘法的不同参数组合进行测试
            for arg in [(m, v), (v, m), (m, m)]:
                res = self.matmul(*arg)
                # 断言结果的数据类型是否与输入的数据类型一致
                assert_(res.dtype == dt)

            # 向量乘以向量应返回标量，检查返回类型是否正确
            if dt != "O":
                res = self.matmul(v, v)
                assert_(type(res) is np.dtype(dt).type)

    # 测试标量输出的方法
    def test_scalar_output(self):
        vec1 = np.array([2])
        vec2 = np.array([3, 4]).reshape(1, -1)
        tgt = np.array([6, 8])
        # 对每种数据类型进行迭代测试
        for dt in self.types[1:]:
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            # 执行标量输出的乘法操作，检查结果是否与目标值一致
            res = self.matmul(v1, v2)
            assert_equal(res, tgt)
            res = self.matmul(v2.T, v1)
            assert_equal(res, tgt)

        # 布尔类型的向量乘法应返回布尔值
        vec = np.array([True, True], dtype='?').reshape(1, -1)
        res = self.matmul(vec[:, 0], vec)
        assert_equal(res, True)
    # 测试向量和向量的乘积结果
    def test_vector_vector_values(self):
        # 创建两个numpy数组作为向量
        vec1 = np.array([1, 2])
        vec2 = np.array([3, 4]).reshape(-1, 1)
        # 目标结果
        tgt1 = np.array([11])
        tgt2 = np.array([[3, 6], [4, 8]])
        # 遍历除了第一个类型之外的所有类型
        for dt in self.types[1:]:
            # 将向量转换为当前数据类型
            v1 = vec1.astype(dt)
            v2 = vec2.astype(dt)
            # 调用matmul方法计算乘积
            res = self.matmul(v1, v2)
            # 断言结果等于目标值tgt1
            assert_equal(res, tgt1)
            # 没有广播，必须将v1转换为二维ndarray
            res = self.matmul(v2, v1.reshape(1, -1))
            # 断言结果等于目标值tgt2
            assert_equal(res, tgt2)

        # 布尔类型
        vec = np.array([True, True], dtype='?')
        # 调用matmul方法计算布尔值的乘积
        res = self.matmul(vec, vec)
        # 断言结果等于True
        assert_equal(res, True)

    # 测试向量和矩阵的乘积结果
    def test_vector_matrix_values(self):
        # 创建一个向量和两个矩阵
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1]*2, axis=0)
        # 目标结果
        tgt1 = np.array([7, 10])
        tgt2 = np.stack([tgt1]*2, axis=0)
        # 遍历除了第一个类型之外的所有类型
        for dt in self.types[1:]:
            # 将向量和矩阵转换为当前数据类型
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            # 调用matmul方法计算乘积
            res = self.matmul(v, m1)
            # 断言结果等于目标值tgt1
            assert_equal(res, tgt1)
            res = self.matmul(v, m2)
            # 断言结果等于目标值tgt2
            assert_equal(res, tgt2)

        # 布尔类型
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1]*2, axis=0)
        # 目标结果
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1]*2, axis=0)
        # 调用matmul方法计算布尔值的乘积
        res = self.matmul(vec, mat1)
        # 断言结果等于目标值tgt1
        assert_equal(res, tgt1)
        res = self.matmul(vec, mat2)
        # 断言结果等于目标值tgt2
        assert_equal(res, tgt2)

    # 测试矩阵和向量的乘积结果
    def test_matrix_vector_values(self):
        # 创建一个向量和两个矩阵
        vec = np.array([1, 2])
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.stack([mat1]*2, axis=0)
        # 目标结果
        tgt1 = np.array([5, 11])
        tgt2 = np.stack([tgt1]*2, axis=0)
        # 遍历除了第一个类型之外的所有类型
        for dt in self.types[1:]:
            # 将向量和矩阵转换为当前数据类型
            v = vec.astype(dt)
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            # 调用matmul方法计算乘积
            res = self.matmul(m1, v)
            # 断言结果等于目标值tgt1
            assert_equal(res, tgt1)
            res = self.matmul(m2, v)
            # 断言结果等于目标值tgt2
            assert_equal(res, tgt2)

        # 布尔类型
        vec = np.array([True, False])
        mat1 = np.array([[True, False], [False, True]])
        mat2 = np.stack([mat1]*2, axis=0)
        # 目标结果
        tgt1 = np.array([True, False])
        tgt2 = np.stack([tgt1]*2, axis=0)
        # 调用matmul方法计算布尔值的乘积
        res = self.matmul(vec, mat1)
        # 断言结果等于目标值tgt1
        assert_equal(res, tgt1)
        res = self.matmul(vec, mat2)
        # 断言结果等于目标值tgt2
        assert_equal(res, tgt2)
    # 定义一个测试方法，用于测试矩阵与矩阵相乘的不同情况
    def test_matrix_matrix_values(self):
        # 创建两个2x2的NumPy数组作为矩阵mat1和mat2
        mat1 = np.array([[1, 2], [3, 4]])
        mat2 = np.array([[1, 0], [1, 1]])
        
        # 将mat1和mat2按照指定轴（axis=0）进行堆叠，得到一个包含两个矩阵的数组mat12和mat21
        mat12 = np.stack([mat1, mat2], axis=0)
        mat21 = np.stack([mat2, mat1], axis=0)
        
        # 创建预期结果的NumPy数组tgt11、tgt12、tgt21、tgt12_21和tgt11_12，分别表示不同矩阵相乘的预期结果
        tgt11 = np.array([[7, 10], [15, 22]])
        tgt12 = np.array([[3, 2], [7, 4]])
        tgt21 = np.array([[1, 2], [4, 6]])
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)
        
        # 遍历self.types[1:]中的每个数据类型dt，并对mat1、mat2、mat12和mat21进行类型转换
        for dt in self.types[1:]:
            m1 = mat1.astype(dt)
            m2 = mat2.astype(dt)
            m12 = mat12.astype(dt)
            m21 = mat21.astype(dt)
            
            # 对不同类型和形状的矩阵进行矩阵乘法运算，并使用assert_equal断言进行结果验证
            # matrix @ matrix
            res = self.matmul(m1, m2)
            assert_equal(res, tgt12)
            res = self.matmul(m2, m1)
            assert_equal(res, tgt21)
            
            # stacked @ matrix
            res = self.matmul(m12, m1)
            assert_equal(res, tgt11_21)
            
            # matrix @ stacked
            res = self.matmul(m1, m12)
            assert_equal(res, tgt11_12)
            
            # stacked @ stacked
            res = self.matmul(m12, m21)
            assert_equal(res, tgt12_21)
        
        # 使用布尔类型创建矩阵m1、m2，并将它们按照指定轴进行堆叠，得到m12和m21
        m1 = np.array([[1, 1], [0, 0]], dtype=np.bool)
        m2 = np.array([[1, 0], [1, 1]], dtype=np.bool)
        m12 = np.stack([m1, m2], axis=0)
        m21 = np.stack([m2, m1], axis=0)
        
        # 创建布尔类型的预期结果矩阵tgt11、tgt12、tgt21、tgt12_21和tgt11_12
        tgt11 = m1
        tgt12 = m1
        tgt21 = np.array([[1, 1], [1, 1]], dtype=np.bool)
        tgt12_21 = np.stack([tgt12, tgt21], axis=0)
        tgt11_12 = np.stack((tgt11, tgt12), axis=0)
        tgt11_21 = np.stack((tgt11, tgt21), axis=0)
        
        # 对布尔类型的矩阵进行矩阵乘法运算，并使用assert_equal断言进行结果验证
        # matrix @ matrix
        res = self.matmul(m1, m2)
        assert_equal(res, tgt12)
        res = self.matmul(m2, m1)
        assert_equal(res, tgt21)
        
        # stacked @ matrix
        res = self.matmul(m12, m1)
        assert_equal(res, tgt11_21)
        
        # matrix @ stacked
        res = self.matmul(m1, m12)
        assert_equal(res, tgt11_12)
        
        # stacked @ stacked
        res = self.matmul(m12, m21)
        assert_equal(res, tgt12_21)
class TestMatmul(MatmulCommon):
    matmul = np.matmul  # 将 np.matmul 函数赋值给类变量 matmul

    def test_out_arg(self):
        a = np.ones((5, 2), dtype=float)  # 创建一个形状为 (5, 2) 的全 1 数组 a
        b = np.array([[1, 3], [5, 7]], dtype=float)  # 创建一个形状为 (2, 2) 的数组 b
        tgt = np.dot(a, b)  # 计算 a 和 b 的点积，赋值给 tgt

        # test as positional argument
        msg = "out positional argument"  # 错误消息
        out = np.zeros((5, 2), dtype=float)  # 创建一个形状为 (5, 2) 的全 0 数组 out
        self.matmul(a, b, out)  # 使用 matmul 进行矩阵乘法，将结果写入 out
        assert_array_equal(out, tgt, err_msg=msg)  # 断言 out 是否等于预期的 tgt

        # test as keyword argument
        msg = "out keyword argument"  # 错误消息
        out = np.zeros((5, 2), dtype=float)  # 创建一个形状为 (5, 2) 的全 0 数组 out
        self.matmul(a, b, out=out)  # 使用 matmul 进行矩阵乘法，将结果写入 out
        assert_array_equal(out, tgt, err_msg=msg)  # 断言 out 是否等于预期的 tgt

        # test out with not allowed type cast (safe casting)
        msg = "Cannot cast ufunc .* output"  # 错误消息
        out = np.zeros((5, 2), dtype=np.int32)  # 创建一个形状为 (5, 2) 的全 0 整数数组 out
        assert_raises_regex(TypeError, msg, self.matmul, a, b, out=out)  # 断言 matmul 在给定的类型下会引发 TypeError 异常

        # test out with type upcast to complex
        out = np.zeros((5, 2), dtype=np.complex128)  # 创建一个形状为 (5, 2) 的全 0 复数数组 out
        c = self.matmul(a, b, out=out)  # 使用 matmul 进行矩阵乘法，将结果写入 out，并赋值给 c
        assert_(c is out)  # 断言 c 和 out 是同一个对象
        with suppress_warnings() as sup:  # 屏蔽警告
            sup.filter(ComplexWarning, '')  # 过滤掉复数警告
            c = c.astype(tgt.dtype)  # 将 c 的数据类型转换为 tgt 的数据类型
        assert_array_equal(c, tgt)  # 断言 c 是否等于预期的 tgt

    def test_empty_out(self):
        # Check that the output cannot be broadcast, so that it cannot be
        # size zero when the outer dimensions (iterator size) has size zero.
        arr = np.ones((0, 1, 1))  # 创建一个形状为 (0, 1, 1) 的全 1 数组 arr
        out = np.ones((1, 1, 1))  # 创建一个形状为 (1, 1, 1) 的全 1 数组 out
        assert self.matmul(arr, arr).shape == (0, 1, 1)  # 断言 matmul 的输出形状为 (0, 1, 1)

        with pytest.raises(ValueError, match=r"non-broadcastable"):
            self.matmul(arr, arr, out=out)  # 断言 matmul 在给定的条件下会引发 ValueError 异常

    def test_out_contiguous(self):
        a = np.ones((5, 2), dtype=float)  # 创建一个形状为 (5, 2) 的全 1 数组 a
        b = np.array([[1, 3], [5, 7]], dtype=float)  # 创建一个形状为 (2, 2) 的数组 b
        v = np.array([1, 3], dtype=float)  # 创建一个形状为 (2,) 的数组 v
        tgt = np.dot(a, b)  # 计算 a 和 b 的点积，赋值给 tgt
        tgt_mv = np.dot(a, v)  # 计算 a 和 v 的点积，赋值给 tgt_mv

        # test out non-contiguous
        out = np.ones((5, 2, 2), dtype=float)  # 创建一个形状为 (5, 2, 2) 的全 1 数组 out
        c = self.matmul(a, b, out=out[..., 0])  # 使用 matmul 进行矩阵乘法，将结果写入 out 的第一维度
        assert c.base is out  # 断言 c 和 out 是同一个基础对象
        assert_array_equal(c, tgt)  # 断言 c 是否等于预期的 tgt
        c = self.matmul(a, v, out=out[:, 0, 0])  # 使用 matmul 进行矢量乘法，将结果写入 out 的第一维度的第一个元素
        assert_array_equal(c, tgt_mv)  # 断言 c 是否等于预期的 tgt_mv
        c = self.matmul(v, a.T, out=out[:, 0, 0])  # 使用 matmul 进行矢量乘法，将结果写入 out 的第一维度的第一个元素
        assert_array_equal(c, tgt_mv)  # 断言 c 是否等于预期的 tgt_mv

        # test out contiguous in only last dim
        out = np.ones((10, 2), dtype=float)  # 创建一个形状为 (10, 2) 的全 1 数组 out
        c = self.matmul(a, b, out=out[::2, :])  # 使用 matmul 进行矩阵乘法，将结果写入 out 的一部分
        assert_array_equal(c, tgt)  # 断言 c 是否等于预期的 tgt

        # test transposes of out, args
        out = np.ones((5, 2), dtype=float)  # 创建一个形状为 (5, 2) 的全 1 数组 out
        c = self.matmul(b.T, a.T, out=out.T)  # 使用 matmul 进行矩阵乘法，将结果写入 out 的转置
        assert_array_equal(out, tgt)  # 断言 out 是否等于预期的 tgt

    m1 = np.arange(15.).reshape(5, 3)  # 创建一个形状为 (5, 3) 的数组 m1
    m2 = np.arange(21.).reshape(3, 7)  # 创建一个形状为 (3, 7) 的数组 m2
    m3 = np.arange(30.).reshape(5, 6)[:, ::2]  # 创建一个形状为 (5, 3) 的数组 m3，仅保留偶数列
    vc = np.arange(10.)  # 创建一个包含 0 到 9 的一维数组 vc
    vr = np.arange(6.)  # 创建一个包含 0 到 5 的一维数组 vr
    m0 = np.zeros((3, 0))  # 创建一个形状为 (3, 0) 的全 0 数组 m0
    @pytest.mark.parametrize('args', (
        # 测试参数化：矩阵乘法
        (m1, m2), (m2.T, m1.T), (m2.T.copy(), m1.T), (m2.T, m1.T.copy()),
        # 矩阵乘法，包括转置，连续和非连续
        (m1, m1.T), (m1.T, m1), (m1, m3.T), (m3, m1.T),
        (m3, m3.T), (m3.T, m3),
        # 矩阵乘法，非连续
        (m3, m2), (m2.T, m3.T), (m2.T.copy(), m3.T),
        # 向量-矩阵，矩阵-向量，连续
        (m1, vr[:3]), (vc[:5], m1), (m1.T, vc[:5]), (vr[:3], m1.T),
        # 向量-矩阵，矩阵-向量，向量非连续
        (m1, vr[::2]), (vc[::2], m1), (m1.T, vc[::2]), (vr[::2], m1.T),
        # 向量-矩阵，矩阵-向量，矩阵非连续
        (m3, vr[:3]), (vc[:5], m3), (m3.T, vc[:5]), (vr[:3], m3.T),
        # 向量-矩阵，矩阵-向量，都非连续
        (m3, vr[::2]), (vc[::2], m3), (m3.T, vc[::2]), (vr[::2], m3.T),
        # 尺寸为0
        (m0, m0.T), (m0.T, m0), (m1, m0), (m0.T, m1.T),
    ))
    # 测试矩阵乘法的等效性
    def test_dot_equivalent(self, args):
        r1 = np.matmul(*args)
        r2 = np.dot(*args)
        assert_equal(r1, r2)

        r3 = np.matmul(args[0].copy(), args[1].copy())
        assert_equal(r1, r3)

    # 测试矩阵乘法与自定义对象
    def test_matmul_object(self):
        import fractions

        f = np.vectorize(fractions.Fraction)
        def random_ints():
            return np.random.randint(1, 1000, size=(10, 3, 3))
        M1 = f(random_ints(), random_ints())
        M2 = f(random_ints(), random_ints())

        M3 = self.matmul(M1, M2)

        [N1, N2, N3] = [a.astype(float) for a in [M1, M2, M3]]

        assert_allclose(N3, self.matmul(N1, N2))

    # 测试矩阵乘法与标量对象的类型
    def test_matmul_object_type_scalar(self):
        from fractions import Fraction as F
        v = np.array([F(2,3), F(5,7)])
        res = self.matmul(v, v)
        assert_(type(res) is F)

    # 测试矩阵乘法与空矩阵
    def test_matmul_empty(self):
        a = np.empty((3, 0), dtype=object)
        b = np.empty((0, 3), dtype=object)
        c = np.zeros((3, 3))
        assert_array_equal(np.matmul(a, b), c)

    # 测试矩阵乘法引发异常（乘法操作缺失）
    def test_matmul_exception_multiply(self):
        # 测试如果缺少 `__mul__` 方法，矩阵乘法是否会失败
        class add_not_multiply():
            def __add__(self, other):
                return self
        a = np.full((3,3), add_not_multiply())
        with assert_raises(TypeError):
            b = np.matmul(a, a)

    # 测试矩阵乘法引发异常（加法操作缺失）
    def test_matmul_exception_add(self):
        # 测试如果缺少 `__add__` 方法，矩阵乘法是否会失败
        class multiply_not_add():
            def __mul__(self, other):
                return self
        a = np.full((3,3), multiply_not_add())
        with assert_raises(TypeError):
            b = np.matmul(a, a)
    # 定义一个测试函数，用于测试布尔矩阵乘法
    def test_matmul_bool(self):
        # 说明：此测试用例涉及到 GitHub issue #14439

        # 创建一个布尔类型的二维数组 a
        a = np.array([[1, 0],[1, 1]], dtype=bool)
        # 断言：a 中的最大值转换为无符号 8 位整数后应该为 1
        assert np.max(a.view(np.uint8)) == 1

        # 计算布尔矩阵 a 与自身的矩阵乘法结果 b
        b = np.matmul(a, a)
        # 断言：b 转换为无符号 8 位整数后的最大值应为 1
        # 说明：布尔类型的矩阵乘法结果应总是 0 或 1
        assert np.max(b.view(np.uint8)) == 1

        # 创建一个指定种子的随机数生成器 rg
        rg = np.random.default_rng(np.random.PCG64(43))
        # 生成一个包含随机整数的一维数组 d，元素取值范围为 [0, 1]
        d = rg.integers(2, size=4*5, dtype=np.int8)
        # 将一维数组 d 转换为形状为 (4, 5) 的二维布尔数组
        d = d.reshape(4, 5) > 0

        # 计算布尔数组 d 与其转置的矩阵乘法结果 out1
        out1 = np.matmul(d, d.reshape(5, 4))
        # 使用 np.dot 函数计算布尔数组 d 与其转置的乘法结果 out2
        out2 = np.dot(d, d.reshape(5, 4))
        # 断言：out1 和 out2 应该相等
        assert_equal(out1, out2)

        # 创建两个形状为 (2, 0) 和 (0,) 的空布尔数组进行矩阵乘法运算 c
        c = np.matmul(np.zeros((2, 0), dtype=bool), np.zeros(0, dtype=bool))
        # 断言：数组 c 不应包含任何 True 值
        assert not np.any(c)
class TestMatmulOperator(MatmulCommon):
    import operator  # 导入operator模块，用于支持运算符重载
    matmul = operator.matmul  # 将operator.matmul赋值给类变量matmul

    def test_array_priority_override(self):
        # 定义内部类A，设置其数组优先级为1000
        class A:
            __array_priority__ = 1000

            def __matmul__(self, other):
                return "A"  # 定义矩阵乘法运算符@的行为，返回字符串"A"

            def __rmatmul__(self, other):
                return "A"  # 定义反向矩阵乘法运算符的行为，返回字符串"A"

        a = A()  # 创建类A的实例a
        b = np.ones(2)  # 创建一个包含全部为1的长度为2的数组b
        assert_equal(self.matmul(a, b), "A")  # 调用类变量matmul执行a @ b，断言结果为"A"
        assert_equal(self.matmul(b, a), "A")  # 调用类变量matmul执行b @ a，断言结果为"A"

    def test_matmul_raises(self):
        # 使用assert_raises断言TypeError异常被抛出，当matmul被应用于np.int8(5)和np.int8(5)
        assert_raises(TypeError, self.matmul, np.int8(5), np.int8(5))
        # 使用assert_raises断言TypeError异常被抛出，当matmul被应用于np.void(b'abc')和np.void(b'abc')
        assert_raises(TypeError, self.matmul, np.void(b'abc'), np.void(b'abc'))
        # 使用assert_raises断言TypeError异常被抛出，当matmul被应用于np.arange(10)和np.void(b'abc')
        assert_raises(TypeError, self.matmul, np.arange(10), np.void(b'abc'))


class TestMatmulInplace:
    DTYPES = {}  # 创建空字典DTYPES
    for i in MatmulCommon.types:  # 迭代MatmulCommon类的types属性
        for j in MatmulCommon.types:  # 再次迭代MatmulCommon类的types属性
            if np.can_cast(j, i):  # 如果可以将j转换为i的数据类型
                DTYPES[f"{i}-{j}"] = (np.dtype(i), np.dtype(j))  # 将(i-j)作为键，(np.dtype(i), np.dtype(j))作为值存入DTYPES字典

    @pytest.mark.parametrize("dtype1,dtype2", DTYPES.values(), ids=DTYPES)
    def test_basic(self, dtype1: np.dtype, dtype2: np.dtype) -> None:
        a = np.arange(10).reshape(5, 2).astype(dtype1)  # 创建一个10个元素的数组，并按dtype1的数据类型进行reshape
        a_id = id(a)  # 获取数组a的id
        b = np.ones((2, 2), dtype=dtype2)  # 创建一个dtype2数据类型的2x2数组，所有元素为1

        ref = a @ b  # 计算a和b的矩阵乘积，将结果赋给ref
        a @= b  # 就地计算a和b的矩阵乘积

        assert id(a) == a_id  # 断言a的id未变
        assert a.dtype == dtype1  # 断言a的数据类型为dtype1
        assert a.shape == (5, 2)  # 断言a的形状为(5, 2)
        if dtype1.kind in "fc":  # 如果dtype1的类型为浮点数或复数
            np.testing.assert_allclose(a, ref)  # 使用np.testing.assert_allclose断言a和ref在容差范围内相等
        else:
            np.testing.assert_array_equal(a, ref)  # 否则使用np.testing.assert_array_equal断言a和ref完全相等

    SHAPES = {
        "2d_large": ((10**5, 10), (10, 10)),  # 定义不同形状的矩阵乘积测试用例
        "3d_large": ((10**4, 10, 10), (1, 10, 10)),
        "1d": ((3,), (3,)),
        "2d_1d": ((3, 3), (3,)),
        "1d_2d": ((3,), (3, 3)),
        "2d_broadcast": ((3, 3), (3, 1)),
        "2d_broadcast_reverse": ((1, 3), (3, 3)),
        "3d_broadcast1": ((3, 3, 3), (1, 3, 1)),
        "3d_broadcast2": ((3, 3, 3), (1, 3, 3)),
        "3d_broadcast3": ((3, 3, 3), (3, 3, 1)),
        "3d_broadcast_reverse1": ((1, 3, 3), (3, 3, 3)),
        "3d_broadcast_reverse2": ((3, 1, 3), (3, 3, 3)),
        "3d_broadcast_reverse3": ((1, 1, 3), (3, 3, 3)),
    }

    @pytest.mark.parametrize("a_shape,b_shape", SHAPES.values(), ids=SHAPES)
    def test_shapes(self, a_shape: tuple[int, ...], b_shape: tuple[int, ...]):
        a_size = np.prod(a_shape)  # 计算数组a_shape的元素个数
        a = np.arange(a_size).reshape(a_shape).astype(np.float64)  # 创建一个指定形状和数据类型的数组a，并初始化为按顺序排列的浮点数

        a_id = id(a)  # 获取数组a的id

        b_size = np.prod(b_shape)  # 计算数组b_shape的元素个数
        b = np.arange(b_size).reshape(b_shape)  # 创建一个指定形状的数组b，并初始化为按顺序排列的整数

        ref = a @ b  # 计算a和b的矩阵乘积，将结果赋给ref
        if ref.shape != a_shape:  # 如果计算结果ref的形状不等于a_shape
            with pytest.raises(ValueError):  # 使用pytest.raises断言引发ValueError异常
                a @= b  # 就地计算a和b的矩阵乘积
            return
        else:
            a @= b  # 就地计算a和b的矩阵乘积

        assert id(a) == a_id  # 断言a的id未变
        assert a.dtype.type == np.float64  # 断言a的数据类型为np.float64
        assert a.shape == a_shape  # 断言a的形状为a_shape
        np.testing.assert_allclose(a, ref)  # 使用np.testing.assert_allclose断言a和ref在容差范围内相等


def test_matmul_axes():
    a = np.arange(3*4*5).reshape(3, 4, 5)  # 创建一个3x4x5的三维数组a，并初始化为按顺序排列的整数
    c = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (1, 2)])  # 计算a和a的矩阵乘积，指定轴顺序，将结果赋给c
    assert c.shape == (3, 4, 4)  # 断言c的形状为(3, 4, 4)
    d = np.matmul(a, a, axes=[(-2, -1), (-1, -2), (0, 1)])  # 计算a和a的矩阵乘积，指定轴顺序，将结果赋给d
    # 断言数组 d 的形状为 (4, 4, 3)
    assert d.shape == (4, 4, 3)
    # 使用 NumPy 中的 swapaxes 函数，交换数组 d 的第 0 和第 2 轴
    e = np.swapaxes(d, 0, 2)
    # 断言交换轴后的数组 e 与数组 c 相等
    assert_array_equal(e, c)
    # 使用 NumPy 中的 matmul 函数，计算矩阵 a 与一维数组 np.arange(3) 的矩阵乘积，
    # 指定轴的顺序为 [(1, 0), (0), (0)]
    f = np.matmul(a, np.arange(3), axes=[(1, 0), (0), (0)])
    # 断言数组 f 的形状为 (4, 5)
    assert f.shape == (4, 5)
class TestInner:

    def test_inner_type_mismatch(self):
        # 设置一个浮点数常量
        c = 1.
        # 创建一个整数数组，元组表示数据类型为两个整数
        A = np.array((1, 1), dtype='i,i')

        # 断言检查：预期会抛出 TypeError 异常，因为 np.inner 不支持浮点数与数组的内积
        assert_raises(TypeError, np.inner, c, A)
        assert_raises(TypeError, np.inner, A, c)

    def test_inner_scalar_and_vector(self):
        # 遍历所有整数和浮点数类型及布尔类型
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            # 创建一个标量数组，标量值为3，指定数据类型为当前迭代的类型
            sca = np.array(3, dtype=dt)[()]
            # 创建一个整数或浮点数数组，元素为1和2，数据类型为当前迭代的类型
            vec = np.array([1, 2], dtype=dt)
            # 预期的内积结果数组，元素值为3和6，数据类型与当前迭代的类型相同
            desired = np.array([3, 6], dtype=dt)
            # 断言检查：np.inner 计算向量与标量的内积，结果应与预期相等
            assert_equal(np.inner(vec, sca), desired)
            assert_equal(np.inner(sca, vec), desired)

    def test_vecself(self):
        # Ticket 844.
        # 向量与自身的内积，在某些情况下可能导致段错误或返回无意义的结果
        a = np.zeros(shape=(1, 80), dtype=np.float64)
        # 计算向量 a 与自身的内积
        p = np.inner(a, a)
        # 断言检查：预期内积 p 等于0，精确度为14位小数
        assert_almost_equal(p, 0, decimal=14)

    def test_inner_product_with_various_contiguities(self):
        # github issue 6532
        # 遍历所有整数和浮点数类型及布尔类型
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            # 创建一个2x2的数组 A，元素为1到4，数据类型为当前迭代的类型
            A = np.array([[1, 2], [3, 4]], dtype=dt)
            # 创建一个2x2的数组 B，元素为1, 3和2, 4，数据类型为当前迭代的类型
            B = np.array([[1, 3], [2, 4]], dtype=dt)
            # 创建一个长度为2的数组 C，元素为1和1，数据类型为当前迭代的类型
            C = np.array([1, 1], dtype=dt)
            # 预期的内积结果数组，元素为4和6，数据类型与当前迭代的类型相同
            desired = np.array([4, 6], dtype=dt)
            # 断言检查：np.inner 计算转置后的 A 与向量 C 的内积，结果应与预期相等
            assert_equal(np.inner(A.T, C), desired)
            assert_equal(np.inner(C, A.T), desired)
            # 断言检查：np.inner 计算 B 与向量 C 的内积，结果应与预期相等
            assert_equal(np.inner(B, C), desired)
            assert_equal(np.inner(C, B), desired)
            # 预期的内积结果数组，元素为7和10，数据类型与当前迭代的类型相同
            desired = np.array([[7, 10], [15, 22]], dtype=dt)
            # 断言检查：np.inner 计算 A 与 B 的矩阵乘积，结果应与预期相等
            assert_equal(np.inner(A, B), desired)
            # 预期的内积结果数组，元素为5和11，数据类型与当前迭代的类型相同
            desired = np.array([[5, 11], [11, 25]], dtype=dt)
            # 断言检查：np.inner 计算 A 与自身的内积，结果应与预期相等
            assert_equal(np.inner(A, A), desired)
            assert_equal(np.inner(A, A.copy()), desired)
            # 创建一个整数或浮点数数组 a，元素为0到4，数据类型为当前迭代的类型
            a = np.arange(5).astype(dt)
            # 创建一个 a 的逆序视图 b
            b = a[::-1]
            # 预期的内积结果为10，数据类型与当前迭代的类型相同
            desired = np.array(10, dtype=dt).item()
            # 断言检查：np.inner 计算 b 与 a 的内积，结果应与预期相等
            assert_equal(np.inner(b, a), desired)

    def test_3d_tensor(self):
        # 遍历所有整数和浮点数类型及布尔类型
        for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
            # 创建一个2x3x4的3维数组 a，元素为0到23，数据类型为当前迭代的类型
            a = np.arange(24).reshape(2, 3, 4).astype(dt)
            # 创建一个2x3x4的3维数组 b，元素为24到47，数据类型为当前迭代的类型
            b = np.arange(24, 48).reshape(2, 3, 4).astype(dt)
            # 预期的3维张量内积结果数组，具体数值根据数据类型变化
            desired = np.array(
                [[[[ 158,  182,  206],
                   [ 230,  254,  278]],

                  [[ 566,  654,  742],
                   [ 830,  918, 1006]],

                  [[ 974, 1126, 1278],
                   [1430, 1582, 1734]]],

                 [[[1382, 1598, 1814],
                   [2030, 2246, 2462]],

                  [[1790, 2070, 2350],
                   [2630, 2910, 3190]],

                  [[2198, 2542, 2886],
                   [3230, 3574, 3918]]]]
            ).astype(dt)
            # 断言检查：np.inner 计算 a 与 b 的3维张量内积，结果应与预期相等
            assert_equal(np.inner(a, b), desired)
            # 断言检查：np.inner 计算 b 与 a 的3维张量内积的转置，结果应与预期相等
            assert_equal(np.inner(b, a).transpose(2, 3, 0, 1), desired)

class TestChoose:
    # 设置测试环境的方法，在每个测试方法执行前调用
    def setup_method(self):
        # 创建一个包含三个元素，类型为整数的数组 self.x，元素值均为 2
        self.x = 2*np.ones((3,), dtype=int)
        # 创建一个包含三个元素，类型为整数的数组 self.y，元素值均为 3
        self.y = 3*np.ones((3,), dtype=int)
        # 创建一个包含两行三列，类型为整数的二维数组 self.x2，元素值均为 2
        self.x2 = 2*np.ones((2, 3), dtype=int)
        # 创建一个包含两行三列，类型为整数的二维数组 self.y2，元素值均为 3
        self.y2 = 3*np.ones((2, 3), dtype=int)
        # 创建一个包含三个元素的列表 self.ind，值分别为 [0, 0, 1]

    # 基本的测试方法，验证 np.choose 的基本用法
    def test_basic(self):
        # 使用 self.ind 数组选择 self.x 或 self.y 中的元素，构成数组 A
        A = np.choose(self.ind, (self.x, self.y))
        # 断言 A 应为 [2, 2, 3]
        assert_equal(A, [2, 2, 3])

    # 测试广播功能，验证 np.choose 在广播数组上的应用
    def test_broadcast1(self):
        # 使用 self.ind 数组选择 self.x2 或 self.y2 中的元素，构成二维数组 A
        A = np.choose(self.ind, (self.x2, self.y2))
        # 断言 A 应为 [[2, 2, 3], [2, 2, 3]]
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

    # 测试广播功能，验证 np.choose 在不同维度数组上的应用
    def test_broadcast2(self):
        # 使用 self.ind 数组选择 self.x 或 self.y2 中的元素，构成二维数组 A
        A = np.choose(self.ind, (self.x, self.y2))
        # 断言 A 应为 [[2, 2, 3], [2, 2, 3]]
        assert_equal(A, [[2, 2, 3], [2, 2, 3]])

    # 参数化测试方法，验证 np.choose 在不同数据类型下的输出类型
    @pytest.mark.parametrize("ops",
        [(1000, np.array([1], dtype=np.uint8)),
         (-1, np.array([1], dtype=np.uint8)),
         (1., np.float32(3)),
         (1., np.array([3], dtype=np.float32))],)
    def test_output_dtype(self, ops):
        # 计算 ops 中数组的预期数据类型
        expected_dt = np.result_type(*ops)
        # 断言使用 np.choose([0], ops) 的输出数据类型应为预期的数据类型
        assert(np.choose([0], ops).dtype == expected_dt)

    # 测试方法，验证 np.choose 在限制维度和参数个数的情况下的行为
    def test_dimension_and_args_limit(self):
        # 创建一个维度为 (1,) * 32，类型为 np.intp 的数组 a
        a = np.ones((1,) * 32, dtype=np.intp)
        # 使用 a.choose 进行选择操作，选择 a 自身及其他元素，预期抛出 ValueError 异常
        res = a.choose([0, a] + [2] * 61)
        # 断言捕获的异常信息包含指定的错误信息
        with pytest.raises(ValueError,
                match="Need at least 0 and at most 64 array objects"):
            # 调用 a.choose 时传入超出参数个数限制的参数列表，预期抛出 ValueError 异常
            a.choose([0, a] + [2] * 62)

        # 断言 res 与数组 a 相等
        assert_array_equal(res, a)
        # 创建一个维度为 (1,) * 60，类型为 np.intp 的数组 a
        a = np.ones((1,) * 60, dtype=np.intp)
        # 调用 a.choose 时传入超出维度限制的参数列表，预期抛出 RuntimeError 异常
        with pytest.raises(RuntimeError,
                match=".*32 dimensions but the array has 60"):
            a.choose([a, a])
class TestRepeat:
    # 设置测试环境的方法
    def setup_method(self):
        # 创建一个包含整数 1 到 6 的 NumPy 数组
        self.m = np.array([1, 2, 3, 4, 5, 6])
        # 将 self.m 重塑为 2x3 的矩阵
        self.m_rect = self.m.reshape((2, 3))

    # 测试基本的重复功能
    def test_basic(self):
        # 使用指定的重复次数对 self.m 中的元素进行重复
        A = np.repeat(self.m, [1, 3, 2, 1, 1, 2])
        # 断言重复后的数组是否符合预期
        assert_equal(A, [1, 2, 2, 2, 3,
                         3, 4, 5, 6, 6])

    # 测试广播重复（一维数组）
    def test_broadcast1(self):
        # 对 self.m 中的元素进行简单的重复，每个元素重复两次
        A = np.repeat(self.m, 2)
        # 断言重复后的数组是否符合预期
        assert_equal(A, [1, 1, 2, 2, 3, 3,
                         4, 4, 5, 5, 6, 6])

    # 测试指定轴向重复（二维数组）
    def test_axis_spec(self):
        # 沿着 axis=0 轴向重复 self.m_rect，指定每行的重复次数
        A = np.repeat(self.m_rect, [2, 1], axis=0)
        # 断言重复后的数组是否符合预期
        assert_equal(A, [[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6]])

        # 沿着 axis=1 轴向重复 self.m_rect，指定每列的重复次数
        A = np.repeat(self.m_rect, [1, 3, 2], axis=1)
        # 断言重复后的数组是否符合预期
        assert_equal(A, [[1, 2, 2, 2, 3, 3],
                         [4, 5, 5, 5, 6, 6]])

    # 测试广播重复（二维数组）
    def test_broadcast2(self):
        # 沿着 axis=0 轴向重复 self.m_rect，每行重复两次
        A = np.repeat(self.m_rect, 2, axis=0)
        # 断言重复后的数组是否符合预期
        assert_equal(A, [[1, 2, 3],
                         [1, 2, 3],
                         [4, 5, 6],
                         [4, 5, 6]])

        # 沿着 axis=1 轴向重复 self.m_rect，每列重复两次
        A = np.repeat(self.m_rect, 2, axis=1)
        # 断言重复后的数组是否符合预期
        assert_equal(A, [[1, 1, 2, 2, 3, 3],
                         [4, 4, 5, 5, 6, 6]])


# TODO: test for multidimensional
# 邻域模式的映射表
NEIGH_MODE = {'zero': 0, 'one': 1, 'constant': 2, 'circular': 3, 'mirror': 4}


@pytest.mark.parametrize('dt', [float, Decimal], ids=['float', 'object'])
class TestNeighborhoodIter:
    # 简单的二维测试
    def test_simple2d(self, dt):
        # 用于简单数据类型的零填充和一填充测试
        x = np.array([[0, 1], [2, 3]], dtype=dt)
        # 预期的结果数组列表，用于零填充模式
        r = [np.array([[0, 0, 0], [0, 0, 1]], dtype=dt),
             np.array([[0, 0, 0], [0, 1, 0]], dtype=dt),
             np.array([[0, 0, 1], [0, 2, 3]], dtype=dt),
             np.array([[0, 1, 0], [2, 3, 0]], dtype=dt)]
        # 调用测试函数，返回的结果列表与预期结果列表进行比较
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], x[0], NEIGH_MODE['zero'])
        # 断言返回的结果是否与预期一致
        assert_array_equal(l, r)

        # 预期的结果数组列表，用于一填充模式
        r = [np.array([[1, 1, 1], [1, 0, 1]], dtype=dt),
             np.array([[1, 1, 1], [0, 1, 1]], dtype=dt),
             np.array([[1, 0, 1], [1, 2, 3]], dtype=dt),
             np.array([[0, 1, 1], [2, 3, 1]], dtype=dt)]
        # 调用测试函数，返回的结果列表与预期结果列表进行比较
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], x[0], NEIGH_MODE['one'])
        # 断言返回的结果是否与预期一致
        assert_array_equal(l, r)

        # 预期的结果数组列表，用于常量填充模式
        r = [np.array([[4, 4, 4], [4, 0, 1]], dtype=dt),
             np.array([[4, 4, 4], [0, 1, 4]], dtype=dt),
             np.array([[4, 0, 1], [4, 2, 3]], dtype=dt),
             np.array([[0, 1, 4], [2, 3, 4]], dtype=dt)]
        # 调用测试函数，返回的结果列表与预期结果列表进行比较
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], 4, NEIGH_MODE['constant'])
        # 断言返回的结果是否与预期一致
        assert_array_equal(l, r)

        # 从中间位置开始的测试
        r = [np.array([[4, 0, 1], [4, 2, 3]], dtype=dt),
             np.array([[0, 1, 4], [2, 3, 4]], dtype=dt)]
        # 调用测试函数，返回的结果列表与预期结果列表进行比较
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], 4, NEIGH_MODE['constant'], 2)
        # 断言返回的结果是否与预期一致
        assert_array_equal(l, r)
    # 测试二维镜像模式
    def test_mirror2d(self, dt):
        # 创建一个二维数组 x，数据类型为 dt
        x = np.array([[0, 1], [2, 3]], dtype=dt)
        # 预期的结果数组 r，包含四个二维数组
        r = [np.array([[0, 0, 1], [0, 0, 1]], dtype=dt),
             np.array([[0, 1, 1], [0, 1, 1]], dtype=dt),
             np.array([[0, 0, 1], [2, 2, 3]], dtype=dt),
             np.array([[0, 1, 1], [2, 3, 3]], dtype=dt)]
        # 调用 _multiarray_tests.test_neighborhood_iterator 函数进行测试，
        # 使用 x 作为输入数组，[-1, 0, -1, 1] 作为邻域模式，x 的第一行作为填充值，使用镜像模式
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 0, -1, 1], x[0], NEIGH_MODE['mirror'])
        # 断言 l 和 r 数组相等
        assert_array_equal(l, r)

    # 简单的一维测试
    def test_simple(self, dt):
        # 测试使用常数值进行填充
        x = np.linspace(1, 5, 5).astype(dt)
        # 预期的结果数组 r，包含五个数组
        r = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0]]
        # 调用 _multiarray_tests.test_neighborhood_iterator 函数进行测试，
        # 使用 x 作为输入数组，[-1, 1] 作为邻域模式，x 的第一个元素作为填充值，使用零填充模式
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 1], x[0], NEIGH_MODE['zero'])
        # 断言 l 和 r 数组相等
        assert_array_equal(l, r)

        # 预期的结果数组 r
        r = [[1, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 1]]
        # 调用 _multiarray_tests.test_neighborhood_iterator 函数进行测试，
        # 使用 x 作为输入数组，[-1, 1] 作为邻域模式，x 的第一个元素作为填充值，使用常数 1 填充模式
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 1], x[0], NEIGH_MODE['one'])
        # 断言 l 和 r 数组相等
        assert_array_equal(l, r)

        # 预期的结果数组 r
        r = [[x[4], 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, x[4]]]
        # 调用 _multiarray_tests.test_neighborhood_iterator 函数进行测试，
        # 使用 x 作为输入数组，[-1, 1] 作为邻域模式，x 的最后一个元素作为填充值，使用常数填充模式
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-1, 1], x[4], NEIGH_MODE['constant'])
        # 断言 l 和 r 数组相等
        assert_array_equal(l, r)

    # 测试镜像模式
    def test_mirror(self, dt):
        # 创建一维数组 x，数据类型为 dt
        x = np.linspace(1, 5, 5).astype(dt)
        # 预期的结果数组 r，二维数组
        r = np.array([[2, 1, 1, 2, 3], [1, 1, 2, 3, 4], [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 5], [3, 4, 5, 5, 4]], dtype=dt)
        # 调用 _multiarray_tests.test_neighborhood_iterator 函数进行测试，
        # 使用 x 作为输入数组，[-2, 2] 作为邻域模式，x 的第二个元素作为填充值，使用镜像模式
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-2, 2], x[1], NEIGH_MODE['mirror'])
        # 断言 l 的所有元素的数据类型均为 dt
        assert_([i.dtype == dt for i in l])
        # 断言 l 和 r 数组相等
        assert_array_equal(l, r)

    # 环形模式测试
    def test_circular(self, dt):
        # 创建一维数组 x，数据类型为 dt
        x = np.linspace(1, 5, 5).astype(dt)
        # 预期的结果数组 r，二维数组
        r = np.array([[4, 5, 1, 2, 3], [5, 1, 2, 3, 4], [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 1], [3, 4, 5, 1, 2]], dtype=dt)
        # 调用 _multiarray_tests.test_neighborhood_iterator 函数进行测试，
        # 使用 x 作为输入数组，[-2, 2] 作为邻域模式，x 的第一个元素作为填充值，使用环形模式
        l = _multiarray_tests.test_neighborhood_iterator(
                x, [-2, 2], x[0], NEIGH_MODE['circular'])
        # 断言 l 和 r 数组相等
        assert_array_equal(l, r)
# Test stacking neighborhood iterators
# 测试堆叠邻域迭代器

class TestStackedNeighborhoodIter:
    # Simple, 1d test: stacking 2 constant-padded neigh iterators
    # 简单的一维测试：堆叠两个常数填充的邻域迭代器
    def test_simple_const(self):
        dt = np.float64
        # Test zero and one padding for simple data type
        # 测试简单数据类型的零填充和一填充
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([0], dtype=dt),
             np.array([0], dtype=dt),
             np.array([1], dtype=dt),
             np.array([2], dtype=dt),
             np.array([3], dtype=dt),
             np.array([0], dtype=dt),
             np.array([0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-2, 4], NEIGH_MODE['zero'], [0, 0], NEIGH_MODE['zero'])
        assert_array_equal(l, r)

        r = [np.array([1, 0, 1], dtype=dt),
             np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt),
             np.array([3, 0, 1], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-1, 1], NEIGH_MODE['one'])
        assert_array_equal(l, r)

    # 2nd simple, 1d test: stacking 2 neigh iterators, mixing const padding and
    # mirror padding
    # 第二个简单的一维测试：堆叠两个邻域迭代器，混合常数填充和镜像填充
    # 定义一个测试方法，用于测试简单镜像操作
    def test_simple_mirror(self):
        # 设置数据类型为 np.float64
        dt = np.float64
        # 创建一个包含三个元素的 numpy 数组，数据类型为 dt
        # x = [1, 2, 3]
        x = np.array([1, 2, 3], dtype=dt)
        # 预期的结果列表，包含五个 numpy 数组，数据类型为 dt
        r = [np.array([0, 1, 1], dtype=dt),
             np.array([1, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 3], dtype=dt),
             np.array([3, 3, 0], dtype=dt)]
        # 调用 _multiarray_tests 模块的 test_neighborhood_iterator_oob 方法，期望返回 l 与 r 相等
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['mirror'], [-1, 1], NEIGH_MODE['zero'])
        # 断言 l 与 r 相等
        assert_array_equal(l, r)

        # 创建一个包含三个元素的 numpy 数组，数据类型为 dt
        # x = [1, 2, 3]
        x = np.array([1, 2, 3], dtype=dt)
        # 预期的结果列表，包含五个 numpy 数组，数据类型为 dt
        r = [np.array([1, 0, 0], dtype=dt),
             np.array([0, 0, 1], dtype=dt),
             np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt)]
        # 调用 _multiarray_tests 模块的 test_neighborhood_iterator_oob 方法，期望返回 l 与 r 相等
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-2, 0], NEIGH_MODE['mirror'])
        # 断言 l 与 r 相等
        assert_array_equal(l, r)

        # 创建一个包含三个元素的 numpy 数组，数据类型为 dt
        # x = [1, 2, 3]
        x = np.array([1, 2, 3], dtype=dt)
        # 预期的结果列表，包含五个 numpy 数组，数据类型为 dt
        r = [np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt),
             np.array([3, 0, 0], dtype=dt),
             np.array([0, 0, 3], dtype=dt)]
        # 调用 _multiarray_tests 模块的 test_neighborhood_iterator_oob 方法，期望返回 l 与 r 相等
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [0, 2], NEIGH_MODE['mirror'])
        # 断言 l 与 r 相等
        assert_array_equal(l, r)

        # 创建一个包含三个元素的 numpy 数组，数据类型为 dt
        # x = [1, 2, 3]
        x = np.array([1, 2, 3], dtype=dt)
        # 预期的结果列表，包含五个 numpy 数组，数据类型为 dt
        r = [np.array([1, 0, 0, 1, 2], dtype=dt),
             np.array([0, 0, 1, 2, 3], dtype=dt),
             np.array([0, 1, 2, 3, 0], dtype=dt),
             np.array([1, 2, 3, 0, 0], dtype=dt),
             np.array([2, 3, 0, 0, 3], dtype=dt)]
        # 调用 _multiarray_tests 模块的 test_neighborhood_iterator_oob 方法，期望返回 l 与 r 相等
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-2, 2], NEIGH_MODE['mirror'])
        # 断言 l 与 r 相等
        assert_array_equal(l, r)

    # 3rd simple, 1d test: stacking 2 neigh iterators, mixing const padding and
    # circular padding
    # 定义一个测试函数，测试简单的循环邻域迭代器情况
    def test_simple_circular(self):
        dt = np.float64
        # 使用 np.float64 数据类型定义变量 dt
        
        # Stacking zero on top of mirror
        # 在数组 x=[1, 2, 3] 的基础上，使用循环邻域模式 NEIGH_MODE['circular'] 和 NEIGH_MODE['zero'] 进行测试
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([0, 3, 1], dtype=dt),
             np.array([3, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 1], dtype=dt),
             np.array([3, 1, 0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['circular'], [-1, 1], NEIGH_MODE['zero'])
        # 断言 l 与 r 相等
        assert_array_equal(l, r)

        # Stacking mirror on top of zero
        # 在数组 x=[1, 2, 3] 的基础上，使用循环邻域模式 NEIGH_MODE['zero'] 和 NEIGH_MODE['circular'] 进行测试
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([3, 0, 0], dtype=dt),
             np.array([0, 0, 1], dtype=dt),
             np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-2, 0], NEIGH_MODE['circular'])
        # 断言 l 与 r 相等
        assert_array_equal(l, r)

        # Stacking mirror on top of zero: 2nd
        # 在数组 x=[1, 2, 3] 的基础上，使用循环邻域模式 NEIGH_MODE['zero'] 和 NEIGH_MODE['circular'] 进行测试
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([0, 1, 2], dtype=dt),
             np.array([1, 2, 3], dtype=dt),
             np.array([2, 3, 0], dtype=dt),
             np.array([3, 0, 0], dtype=dt),
             np.array([0, 0, 1], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [0, 2], NEIGH_MODE['circular'])
        # 断言 l 与 r 相等
        assert_array_equal(l, r)

        # Stacking mirror on top of zero: 3rd
        # 在数组 x=[1, 2, 3] 的基础上，使用循环邻域模式 NEIGH_MODE['zero'] 和 NEIGH_MODE['circular'] 进行测试
        x = np.array([1, 2, 3], dtype=dt)
        r = [np.array([3, 0, 0, 1, 2], dtype=dt),
             np.array([0, 0, 1, 2, 3], dtype=dt),
             np.array([0, 1, 2, 3, 0], dtype=dt),
             np.array([1, 2, 3, 0, 0], dtype=dt),
             np.array([2, 3, 0, 0, 1], dtype=dt)]
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [-1, 3], NEIGH_MODE['zero'], [-2, 2], NEIGH_MODE['circular'])
        # 断言 l 与 r 相等
        assert_array_equal(l, r)

    # 4th simple, 1d test: stacking 2 neigh iterators, but with lower iterator
    # being strictly within the array
    # 第四个简单的一维测试：堆叠两个邻域迭代器，但较低的迭代器严格位于数组内部
    def test_simple_strict_within(self):
        dt = np.float64
        # 定义数据类型为 np.float64
        # 在数组内部严格内部堆叠零，第一个邻域严格在数组内部
        x = np.array([1, 2, 3], dtype=dt)
        # 预期的结果数组
        r = [np.array([1, 2, 3, 0], dtype=dt)]
        # 调用测试函数，测试邻域迭代器超出边界情况，期望使用零填充模式
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [1, 1], NEIGH_MODE['zero'], [-1, 2], NEIGH_MODE['zero'])
        # 断言两个数组是否相等
        assert_array_equal(l, r)

        # 在零的上面堆叠镜像，第一个邻域严格在数组内部
        x = np.array([1, 2, 3], dtype=dt)
        # 预期的结果数组
        r = [np.array([1, 2, 3, 3], dtype=dt)]
        # 调用测试函数，测试邻域迭代器超出边界情况，期望使用镜像填充模式
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [1, 1], NEIGH_MODE['zero'], [-1, 2], NEIGH_MODE['mirror'])
        # 断言两个数组是否相等
        assert_array_equal(l, r)

        # 在零的上面堆叠循环，第一个邻域严格在数组内部
        x = np.array([1, 2, 3], dtype=dt)
        # 预期的结果数组
        r = [np.array([1, 2, 3, 1], dtype=dt)]
        # 调用测试函数，测试邻域迭代器超出边界情况，期望使用循环填充模式
        l = _multiarray_tests.test_neighborhood_iterator_oob(
                x, [1, 1], NEIGH_MODE['zero'], [-1, 2], NEIGH_MODE['circular'])
        # 断言两个数组是否相等
        assert_array_equal(l, r)
class TestWarnings:
    # 定义测试类 TestWarnings

    def test_complex_warning(self):
        # 测试复杂警告情况
        x = np.array([1, 2])
        # 创建一个 NumPy 数组 x
        y = np.array([1-2j, 1+2j])
        # 创建一个复数类型的 NumPy 数组 y

        with warnings.catch_warnings():
            # 捕获警告信息
            warnings.simplefilter("error", ComplexWarning)
            # 设置复杂警告为错误级别
            assert_raises(ComplexWarning, x.__setitem__, slice(None), y)
            # 断言会触发 ComplexWarning 警告
            assert_equal(x, [1, 2])
            # 断言数组 x 的值为 [1, 2]

class TestMinScalarType:
    # 定义测试类 TestMinScalarType

    def test_usigned_shortshort(self):
        # 测试最小无符号短整数类型
        dt = np.min_scalar_type(2**8-1)
        # 计算给定值对应的最小数据类型
        wanted = np.dtype('uint8')
        # 期望的数据类型是无符号8位整数
        assert_equal(wanted, dt)
        # 断言计算得到的数据类型符合预期

    def test_usigned_short(self):
        # 测试最小无符号短整数类型
        dt = np.min_scalar_type(2**16-1)
        # 计算给定值对应的最小数据类型
        wanted = np.dtype('uint16')
        # 期望的数据类型是无符号16位整数
        assert_equal(wanted, dt)
        # 断言计算得到的数据类型符合预期

    def test_usigned_int(self):
        # 测试最小无符号整数类型
        dt = np.min_scalar_type(2**32-1)
        # 计算给定值对应的最小数据类型
        wanted = np.dtype('uint32')
        # 期望的数据类型是无符号32位整数
        assert_equal(wanted, dt)
        # 断言计算得到的数据类型符合预期

    def test_usigned_longlong(self):
        # 测试最小无符号长长整数类型
        dt = np.min_scalar_type(2**63-1)
        # 计算给定值对应的最小数据类型
        wanted = np.dtype('uint64')
        # 期望的数据类型是无符号64位整数
        assert_equal(wanted, dt)
        # 断言计算得到的数据类型符合预期

    def test_object(self):
        # 测试对象数据类型
        dt = np.min_scalar_type(2**64)
        # 计算给定值对应的最小数据类型
        wanted = np.dtype('O')
        # 期望的数据类型是对象类型
        assert_equal(wanted, dt)
        # 断言计算得到的数据类型符合预期

from numpy._core._internal import _dtype_from_pep3118

class TestPEP3118Dtype:
    # 定义测试类 TestPEP3118Dtype

    def _check(self, spec, wanted):
        # 内部方法，用于检查 PEP3118 规范对应的数据类型是否正确
        dt = np.dtype(wanted)
        # 创建 NumPy 数据类型对象
        actual = _dtype_from_pep3118(spec)
        # 使用 PEP3118 规范解析数据类型
        assert_equal(actual, dt,
                     err_msg="spec %r != dtype %r" % (spec, wanted))
        # 断言解析得到的数据类型与预期的一致

    def test_native_padding(self):
        # 测试本地填充（padding）
        align = np.dtype('i').alignment
        # 获取整数类型 'i' 的对齐方式
        for j in range(8):
            # 循环测试不同的对齐方式
            if j == 0:
                s = 'bi'
            else:
                s = 'b%dxi' % j
            self._check('@'+s, {'f0': ('i1', 0),
                                'f1': ('i', align*(1 + j//align))})
            # 调用内部方法检查本地填充是否正确
            self._check('='+s, {'f0': ('i1', 0),
                                'f1': ('i', 1+j)})

    def test_native_padding_2(self):
        # 测试结构体和子数组的本地填充
        # Native padding should work also for structs and sub-arrays
        self._check('x3T{xi}', {'f0': (({'f0': ('i', 4)}, (3,)), 4)})
        # 调用内部方法检查本地填充是否正确
        self._check('^x3T{xi}', {'f0': (({'f0': ('i', 1)}, (3,)), 1)})
        # 调用内部方法检查本地填充是否正确
    def test_trailing_padding(self):
        # Trailing padding should be included, *and*, the item size
        # should match the alignment if in aligned mode
        
        # 获取整数类型 'i' 的对齐方式
        align = np.dtype('i').alignment
        # 获取整数类型 'i' 的字节大小
        size = np.dtype('i').itemsize

        # 计算按照对齐方式对齐后的大小
        def aligned(n):
            return align*(1 + (n-1)//align)

        # 定义基础的字典结构
        base = dict(formats=['i'], names=['f0'])

        # 进行多种长度的字段测试，并返回包含对齐信息的字典
        self._check('ix',    dict(itemsize=aligned(size + 1), **base))
        self._check('ixx',   dict(itemsize=aligned(size + 2), **base))
        self._check('ixxx',  dict(itemsize=aligned(size + 3), **base))
        self._check('ixxxx', dict(itemsize=aligned(size + 4), **base))
        self._check('i7x',   dict(itemsize=aligned(size + 7), **base))

        # 进行 '^' 对齐模式的测试，返回包含不进行对齐的字典
        self._check('^ix',    dict(itemsize=size + 1, **base))
        self._check('^ixx',   dict(itemsize=size + 2, **base))
        self._check('^ixxx',  dict(itemsize=size + 3, **base))
        self._check('^ixxxx', dict(itemsize=size + 4, **base))
        self._check('^i7x',   dict(itemsize=size + 7, **base))

    def test_native_padding_3(self):
        # 创建一个自定义的结构化数据类型 'dt'，其中包含多个字段和子结构
        dt = np.dtype(
                [('a', 'b'), ('b', 'i'),
                    ('sub', np.dtype('b,i')), ('c', 'i')],
                align=True)
        # 使用自定义函数 '_check' 进行验证
        self._check("T{b:a:xxxi:b:T{b:f0:=i:f1:}:sub:xxxi:c:}", dt)

        # 创建另一个自定义的结构化数据类型 'dt'，包含多个字段和嵌套的子结构
        dt = np.dtype(
                [('a', 'b'), ('b', 'i'), ('c', 'b'), ('d', 'b'),
                    ('e', 'b'), ('sub', np.dtype('b,i', align=True))])
        # 使用自定义函数 '_check' 进行验证
        self._check("T{b:a:=i:b:b:c:b:d:b:e:T{b:f0:xxxi:f1:}:sub:}", dt)

    def test_padding_with_array_inside_struct(self):
        # 创建一个自定义的结构化数据类型 'dt'，包含多个字段和数组类型的子结构
        dt = np.dtype(
                [('a', 'b'), ('b', 'i'), ('c', 'b', (3,)),
                    ('d', 'i')],
                align=True)
        # 使用自定义函数 '_check' 进行验证
        self._check("T{b:a:xxxi:b:3b:c:xi:d:}", dt)

    def test_byteorder_inside_struct(self):
        # 在结构内部设置字节顺序
        # '@T{=i}' 后应该是 '=' 而不是 '@'
        # 通过没有本地对齐来检查这一点
        self._check('@T{^i}xi', {'f0': ({'f0': ('i', 0)}, 0),
                                 'f1': ('i', 5)})

    def test_intra_padding(self):
        # 嵌套对齐的子数组可能需要一些内部填充
        # 获取整数类型 'i' 的对齐方式和字节大小
        align = np.dtype('i').alignment
        size = np.dtype('i').itemsize

        # 计算按照对齐方式对齐后的大小
        def aligned(n):
            return (align*(1 + (n-1)//align))

        # 创建包含对齐信息的字典，用于描述子数组
        self._check('(3)T{ix}', (dict(
            names=['f0'],
            formats=['i'],
            offsets=[0],
            itemsize=aligned(size + 1)
        ), (3,)))

    def test_char_vs_string(self):
        # 创建字符和字符串的数据类型 'dt'
        dt = np.dtype('c')
        # 使用自定义函数 '_check' 进行验证
        self._check('c', dt)

        # 创建包含字符数组和字符串数组的结构化数据类型 'dt'
        dt = np.dtype([('f0', 'S1', (4,)), ('f1', 'S4')])
        # 使用自定义函数 '_check' 进行验证
        self._check('4c4s', dt)

    def test_field_order(self):
        # gh-9053 - 之前我们依赖于字典键的顺序
        # 使用自定义函数 '_check' 进行验证字段顺序
        self._check("(0)I:a:f:b:", [('a', 'I', (0,)), ('b', 'f')])
        self._check("(0)I:b:f:a:", [('b', 'I', (0,)), ('a', 'f')])
    # 定义一个名为 test_unnamed_fields 的测试方法
    def test_unnamed_fields(self):
        # 调用 _check 方法，检查输入 'ii' 的结果是否符合预期
        self._check('ii',     [('f0', 'i'), ('f1', 'i')])
        # 调用 _check 方法，检查输入 'ii:f0:' 的结果是否符合预期
        self._check('ii:f0:', [('f1', 'i'), ('f0', 'i')])

        # 调用 _check 方法，检查输入 'i' 的结果是否符合预期
        self._check('i', 'i')
        # 调用 _check 方法，检查输入 'i:f0:' 的结果是否符合预期
        self._check('i:f0:', [('f0', 'i')])
class TestNewBufferProtocol:
    """ Test PEP3118 buffers """

    # 检查对象通过内存视图和 NumPy 数组的往返转换
    def _check_roundtrip(self, obj):
        # 将对象转换为 NumPy 数组
        obj = np.asarray(obj)
        # 创建内存视图
        x = memoryview(obj)
        # 通过内存视图再次创建 NumPy 数组
        y = np.asarray(x)
        # 通过内存视图创建另一个 NumPy 数组
        y2 = np.array(x)
        # 断言 y 不拥有数据内存
        assert_(not y.flags.owndata)
        # 断言 y2 拥有数据内存
        assert_(y2.flags.owndata)

        # 断言 y 的数据类型与原始对象相同
        assert_equal(y.dtype, obj.dtype)
        # 断言 y 的形状与原始对象相同
        assert_equal(y.shape, obj.shape)
        # 断言 y 的数据与原始对象相等
        assert_array_equal(obj, y)

        # 断言 y2 的数据类型与原始对象相同
        assert_equal(y2.dtype, obj.dtype)
        # 断言 y2 的形状与原始对象相同
        assert_equal(y2.shape, obj.shape)
        # 断言 y2 的数据与原始对象相等
        assert_array_equal(obj, y2)

    # 测试不同类型的数据通过内存视图和 NumPy 数组的往返转换
    def test_roundtrip(self):
        # 测试整数数组的往返转换
        x = np.array([1, 2, 3, 4, 5], dtype='i4')
        self._check_roundtrip(x)

        # 测试浮点数数组的往返转换
        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        self._check_roundtrip(x)

        # 测试切片后的浮点数数组的往返转换
        x = np.zeros((3, 3, 3), dtype=np.float32)[:, 0, :]
        self._check_roundtrip(x)

        # 定义复杂数据类型列表
        dt = [('a', 'b'),
              ('b', 'h'),
              ('c', 'i'),
              ('d', 'l'),
              ('dx', 'q'),
              ('e', 'B'),
              ('f', 'H'),
              ('g', 'I'),
              ('h', 'L'),
              ('hx', 'Q'),
              ('i', np.single),
              ('j', np.double),
              ('k', np.longdouble),
              ('ix', np.csingle),
              ('jx', np.cdouble),
              ('kx', np.clongdouble),
              ('l', 'S4'),
              ('m', 'U4'),
              ('n', 'V3'),
              ('o', '?'),
              ('p', np.half),
              ]
        # 测试结构化数组的往返转换
        x = np.array(
                [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    b'aaaa', 'bbbb', b'xxx', True, 1.0)],
                dtype=dt)
        self._check_roundtrip(x)

        # 测试具有复杂形状的结构化数组的往返转换
        x = np.array(([[1, 2], [3, 4]],), dtype=[('a', (int, (2, 2)))])
        self._check_roundtrip(x)

        # 测试大端序整数的往返转换
        x = np.array([1, 2, 3], dtype='>i2')
        self._check_roundtrip(x)

        # 测试小端序整数的往返转换
        x = np.array([1, 2, 3], dtype='<i2')
        self._check_roundtrip(x)

        # 测试大端序整数的往返转换
        x = np.array([1, 2, 3], dtype='>i4')
        self._check_roundtrip(x)

        # 测试小端序整数的往返转换
        x = np.array([1, 2, 3], dtype='<i4')
        self._check_roundtrip(x)

        # 检查长长整型是否能作为非本机格式表示
        x = np.array([1, 2, 3], dtype='>q')
        self._check_roundtrip(x)

        # 只有本机字节顺序的数据类型可以通过缓冲区接口传递
        # 在本机字节顺序下测试大端序浮点数的往返转换
        if sys.byteorder == 'little':
            x = np.array([1, 2, 3], dtype='>g')
            assert_raises(ValueError, self._check_roundtrip, x)
            x = np.array([1, 2, 3], dtype='<g')
            self._check_roundtrip(x)
        else:
            x = np.array([1, 2, 3], dtype='>g')
            self._check_roundtrip(x)
            x = np.array([1, 2, 3], dtype='<g')
            assert_raises(ValueError, self._check_roundtrip, x)
    # 测试半精度浮点数数组的转换
    def test_roundtrip_half(self):
        # 定义半精度浮点数列表
        half_list = [
            1.0,
            -2.0,
            6.5504 * 10**4,  # (max half precision)
            2**-14,  # ~= 6.10352 * 10**-5 (minimum positive normal)
            2**-24,  # ~= 5.96046 * 10**-8 (minimum strictly positive subnormal)
            0.0,
            -0.0,
            float('+inf'),
            float('-inf'),
            0.333251953125,  # ~= 1/3
        ]

        # 创建半精度浮点数数组（大端序），并进行往返转换检查
        x = np.array(half_list, dtype='>e')
        self._check_roundtrip(x)
        
        # 创建半精度浮点数数组（小端序），并进行往返转换检查
        x = np.array(half_list, dtype='<e')
        self._check_roundtrip(x)

    # 测试各种单一数据类型的往返转换
    def test_roundtrip_single_types(self):
        for typ in np._core.sctypeDict.values():
            dtype = np.dtype(typ)

            # 跳过不能用于缓冲区的日期时间类型和空类型
            if dtype.char in 'Mm':
                continue
            if dtype.char == 'V':
                continue

            # 创建指定数据类型的零数组，并进行往返转换检查
            x = np.zeros(4, dtype=dtype)
            self._check_roundtrip(x)

            # 对于非复数数据类型，创建不同字节序的数组，并进行往返转换检查
            if dtype.char not in 'qQgG':
                dt = dtype.newbyteorder('<')
                x = np.zeros(4, dtype=dt)
                self._check_roundtrip(x)

                dt = dtype.newbyteorder('>')
                x = np.zeros(4, dtype=dt)
                self._check_roundtrip(x)

    # 测试标量值的往返转换
    def test_roundtrip_scalar(self):
        # Issue #4015.
        # 对于标量值0，进行往返转换检查
        self._check_roundtrip(0)

    # 测试无效的缓冲区格式
    def test_invalid_buffer_format(self):
        # datetime64 类型尚不能完全用于缓冲区
        # 在下一个 Numpy 主要版本中应该会修复这个问题
        dt = np.dtype([('a', 'uint16'), ('b', 'M8[s]')])
        a = np.empty(3, dt)
        # 断言应该触发 ValueError 或 BufferError 异常
        assert_raises((ValueError, BufferError), memoryview, a)
        assert_raises((ValueError, BufferError), memoryview, np.array((3), 'M8[D]'))

    # 测试导出简单一维数组
    def test_export_simple_1d(self):
        x = np.array([1, 2, 3, 4, 5], dtype='i')
        y = memoryview(x)
        # 断言导出的内存视图的格式、形状、维度、步幅、子偏移、项大小是否符合预期
        assert_equal(y.format, 'i')
        assert_equal(y.shape, (5,))
        assert_equal(y.ndim, 1)
        assert_equal(y.strides, (4,))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 4)

    # 测试导出简单多维数组
    def test_export_simple_nd(self):
        x = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = memoryview(x)
        # 断言导出的内存视图的格式、形状、维度、步幅、子偏移、项大小是否符合预期
        assert_equal(y.format, 'd')
        assert_equal(y.shape, (2, 2))
        assert_equal(y.ndim, 2)
        assert_equal(y.strides, (16, 8))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 8)

    # 测试导出不连续数组
    def test_export_discontiguous(self):
        x = np.zeros((3, 3, 3), dtype=np.float32)[:, 0, :]
        y = memoryview(x)
        # 断言导出的内存视图的格式、形状、维度、步幅、子偏移、项大小是否符合预期
        assert_equal(y.format, 'f')
        assert_equal(y.shape, (3, 3))
        assert_equal(y.ndim, 2)
        assert_equal(y.strides, (36, 4))
        assert_equal(y.suboffsets, ())
        assert_equal(y.itemsize, 4)
    # 定义一个测试方法，用于测试导出记录功能
    def test_export_record(self):
        # 定义一个结构化数据类型，包含不同的字段和它们的数据类型
        dt = [('a', 'b'),
              ('b', 'h'),
              ('c', 'i'),
              ('d', 'l'),
              ('dx', 'q'),
              ('e', 'B'),
              ('f', 'H'),
              ('g', 'I'),
              ('h', 'L'),
              ('hx', 'Q'),
              ('i', np.single),
              ('j', np.double),
              ('k', np.longdouble),
              ('ix', np.csingle),
              ('jx', np.cdouble),
              ('kx', np.clongdouble),
              ('l', 'S4'),
              ('m', 'U4'),
              ('n', 'V3'),
              ('o', '?'),
              ('p', np.half),
              ]
        # 创建一个 NumPy 数组，用指定的数据类型 dt 初始化
        x = np.array(
                [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    b'aaaa', 'bbbb', b'   ', True, 1.0)],
                dtype=dt)
        # 创建 x 的内存视图
        y = memoryview(x)
        # 断言内存视图的形状为 (1,)
        assert_equal(y.shape, (1,))
        # 断言内存视图的维度为 1
        assert_equal(y.ndim, 1)
        # 断言内存视图的子偏移为空元组
        assert_equal(y.suboffsets, ())

        # 计算数据类型 dt 中所有元素的总大小
        sz = sum([np.dtype(b).itemsize for a, b in dt])
        # 如果 'l' 类型的大小为 4，则断言内存视图的格式符合特定格式
        if np.dtype('l').itemsize == 4:
            assert_equal(y.format, 'T{b:a:=h:b:i:c:l:d:q:dx:B:e:@H:f:=I:g:L:h:Q:hx:f:i:d:j:^g:k:=Zf:ix:Zd:jx:^Zg:kx:4s:l:=4w:m:3x:n:?:o:@e:p:}')
        else:
            assert_equal(y.format, 'T{b:a:=h:b:i:c:q:d:q:dx:B:e:@H:f:=I:g:Q:h:Q:hx:f:i:d:j:^g:k:=Zf:ix:Zd:jx:^Zg:kx:4s:l:=4w:m:3x:n:?:o:@e:p:}')
        # 断言内存视图的步长为 (sz,)
        assert_equal(y.strides, (sz,))
        # 断言内存视图的单个元素大小为 sz
        assert_equal(y.itemsize, sz)

    # 定义一个测试方法，用于测试导出子数组功能
    def test_export_subarray(self):
        # 创建一个包含子数组的 NumPy 数组，指定了子数组的数据类型
        x = np.array(([[1, 2], [3, 4]],), dtype=[('a', ('i', (2, 2)))])
        # 创建 x 的内存视图
        y = memoryview(x)
        # 断言内存视图的格式为指定的格式
        assert_equal(y.format, 'T{(2,2)i:a:}')
        # 断言内存视图的形状为空元组
        assert_equal(y.shape, ())
        # 断言内存视图的维度为 0
        assert_equal(y.ndim, 0)
        # 断言内存视图的步长为空元组
        assert_equal(y.strides, ())
        # 断言内存视图的子偏移为空元组
        assert_equal(y.suboffsets, ())
        # 断言内存视图的单个元素大小为 16
        assert_equal(y.itemsize, 16)

    # 定义一个测试方法，用于测试导出大小端设置功能
    def test_export_endian(self):
        # 创建一个大端序的 NumPy 数组
        x = np.array([1, 2, 3], dtype='>i')
        # 创建 x 的内存视图
        y = memoryview(x)
        # 如果系统字节序为小端，则断言内存视图的格式为大端整数
        if sys.byteorder == 'little':
            assert_equal(y.format, '>i')
        else:
            assert_equal(y.format, 'i')

        # 创建一个小端序的 NumPy 数组
        x = np.array([1, 2, 3], dtype='<i')
        # 创建 x 的内存视图
        y = memoryview(x)
        # 如果系统字节序为小端，则断言内存视图的格式为小端整数
        if sys.byteorder == 'little':
            assert_equal(y.format, 'i')
        else:
            assert_equal(y.format, '<i')

    # 定义一个测试方法，用于测试导出标志位功能
    def test_export_flags(self):
        # 检查 SIMPLE 标志位，预期抛出 ValueError 异常
        assert_raises(ValueError,
                      _multiarray_tests.get_buffer_info,
                      np.arange(5)[::2], ('SIMPLE',))

    # 使用 pytest.mark.parametrize 注解，参数化不同的输入对象和预期的异常类型
    @pytest.mark.parametrize(["obj", "error"], [
            pytest.param(np.array([1, 2], dtype=rational), ValueError, id="array"),
            pytest.param(rational(1, 2), TypeError, id="scalar")])
    @pytest.mark.valgrind_error(reason="leaks buffer info cache temporarily.")
    # 定义一个测试函数，用于测试放宽的步幅（strides）
    def test_relaxed_strides(self, c=np.ones((1, 10, 10), dtype='i8')):
        # 注意：参数 c 被定义为持久的，以便于在泄漏检查时注意到缓冲信息缓存的泄漏（见 gh-16934）。

        # 设置步幅以便在导出时固定
        c.strides = (-1, 80, 8)  # strides need to be fixed at export

        # 断言内存视图的步幅是否为 (800, 80, 8)
        assert_(memoryview(c).strides == (800, 80, 8))

        # 将 C 连续数据写入一个 BytesIO 缓冲区应该正常工作
        fd = io.BytesIO()
        fd.write(c.data)

        # 创建 Fortran 顺序的数组视图
        fortran = c.T
        # 断言内存视图的步幅是否为 (8, 80, 800)
        assert_(memoryview(fortran).strides == (8, 80, 800))

        # 创建一个形状为 (1, 10) 的数组
        arr = np.ones((1, 10))
        # 如果数组是 F 连续的
        if arr.flags.f_contiguous:
            # 获取数组的缓冲区信息，检查 F 连续性
            shape, strides = _multiarray_tests.get_buffer_info(
                    arr, ['F_CONTIGUOUS'])
            # 断言步幅数组的第一个元素为 8
            assert_(strides[0] == 8)

            # 创建一个以 F 顺序存储的数组
            arr = np.ones((10, 1), order='F')
            # 获取数组的缓冲区信息，检查 C 连续性
            shape, strides = _multiarray_tests.get_buffer_info(
                    arr, ['C_CONTIGUOUS'])
            # 断言步幅数组的最后一个元素为 8
            assert_(strides[-1] == 8)
    def test_out_of_order_fields(self):
        # 定义一个自定义的 NumPy 数据类型 dt，包含两个字段 'one' 和 'two'，
        # 字段 'one' 在偏移量 4 处，字段 'two' 在偏移量 0 处，总长度为 8
        dt = np.dtype(dict(
            formats=['<i4', '<i4'],
            names=['one', 'two'],
            offsets=[4, 0],
            itemsize=8
        ))

        # 创建一个空数组 arr，使用自定义数据类型 dt
        # 由于字段存在重叠，因此无法被 PEP3118 表示
        arr = np.empty(1, dt)
        
        # 使用 assert_raises 检查是否会引发 ValueError 异常
        with assert_raises(ValueError):
            memoryview(arr)

    def test_max_dims(self):
        # 创建一个形状为 (1,)*32 的全为 1 的 NumPy 数组 a
        a = np.ones((1,) * 32)
        
        # 调用内部方法 _check_roundtrip，验证数组 a 的往返转换
        self._check_roundtrip(a)

    def test_error_pointer_type(self):
        # 创建一个 ctypes 的 unsigned char 指针对象，并使用 memoryview 封装
        m = memoryview(ctypes.pointer(ctypes.c_uint8()))
        
        # 检查 memoryview 对象的 format 属性是否包含 '&'
        assert_('&' in m.format)

        # 使用 assert_raises_regex 检查是否会引发 ValueError 异常，
        # 并且异常信息中包含 "format string"
        assert_raises_regex(
            ValueError, "format string",
            np.array, m)

    def test_error_message_unsupported(self):
        # 创建一个 ctypes 的 wchar 类型数组 t，长度为 4
        t = ctypes.c_wchar * 4
        
        # 使用 assert_raises 检查是否会引发 ValueError 异常
        with assert_raises(ValueError) as cm:
            np.array(t())

        # 获取异常对象
        exc = cm.exception
        
        # 使用 assert_raises_regex 检查是否会引发 NotImplementedError 异常，
        # 并且异常信息中包含 "Unrepresentable" 和 "'u' (UCS-2 strings)"
        with assert_raises_regex(
            NotImplementedError,
            r"Unrepresentable .* 'u' \(UCS-2 strings\)"
        ):
            # 触发异常的原因抛出
            raise exc.__cause__

    def test_ctypes_integer_via_memoryview(self):
        # 遍历 ctypes 的整数类型 {ctypes.c_int, ctypes.c_long, ctypes.c_longlong}
        for c_integer in {ctypes.c_int, ctypes.c_long, ctypes.c_longlong}:
            # 创建一个指定类型的整数对象 value，并尝试将其转换为 NumPy 数组
            value = c_integer(42)
            with warnings.catch_warnings(record=True):
                # 设置警告过滤器，捕获所有 ctypes 相关的运行时警告
                warnings.filterwarnings('always', r'.*\bctypes\b', RuntimeWarning)
                np.asarray(value)

    def test_ctypes_struct_via_memoryview(self):
        # 定义一个 ctypes 结构体 foo，包含字段 'a' 和 'b'
        class foo(ctypes.Structure):
            _fields_ = [('a', ctypes.c_uint8), ('b', ctypes.c_uint32)]
        
        # 创建结构体实例 f，并尝试将其转换为 NumPy 数组 arr
        f = foo(a=1, b=2)

        with warnings.catch_warnings(record=True):
            # 设置警告过滤器，捕获所有 ctypes 相关的运行时警告
            warnings.filterwarnings('always', r'.*\bctypes\b', RuntimeWarning)
            arr = np.asarray(f)

        # 使用 assert_equal 断言检查 arr 中字段 'a' 和 'b' 的值
        assert_equal(arr['a'], 1)
        assert_equal(arr['b'], 2)
        
        # 修改结构体实例 f 的字段 'a' 的值，并使用 assert_equal 断言验证
        f.a = 3
        assert_equal(arr['a'], 3)

    @pytest.mark.parametrize("obj", [np.ones(3), np.ones(1, dtype="i,i")[()]])
    def test_error_if_stored_buffer_info_is_corrupted(self, obj):
        """
        If a user extends a NumPy array before 1.20 and then runs it
        on NumPy 1.20+. A C-subclassed array might in theory modify
        the new buffer-info field. This checks that an error is raised
        if this happens (for buffer export), an error is written on delete.
        This is a sanity check to help users transition to safe code, it
        may be deleted at any point.
        """
        # 破坏存储的缓冲区信息：
        _multiarray_tests.corrupt_or_fix_bufferinfo(obj)
        
        # 获取对象的类型名
        name = type(obj)
        
        # 使用 pytest.raises 检查是否会引发 RuntimeError 异常
        # 并且异常信息中包含 "{name} appears to be C subclassed"
        with pytest.raises(RuntimeError,
                    match=f".*{name} appears to be C subclassed"):
            # 使用 memoryview 封装对象 obj，触发异常
            memoryview(obj)
        
        # 再次修复缓冲区信息，以免丢失内存
        _multiarray_tests.corrupt_or_fix_bufferinfo(obj)
    # 定义一个名为 test_no_suboffsets 的测试函数
    def test_no_suboffsets(self):
        try:
            # 尝试导入 _testbuffer 模块，用于测试
            import _testbuffer
        except ImportError:
            # 如果导入失败，抛出 pytest.skip 异常，表示跳过此测试
            raise pytest.skip("_testbuffer is not available")

        # 对于不同的数组形状进行迭代测试
        for shape in [(2, 3), (2, 3, 4)]:
            # 创建一个包含指定形状的数据列表
            data = list(range(np.prod(shape)))
            # 使用 _testbuffer.ndarray 创建一个数组缓冲区对象
            buffer = _testbuffer.ndarray(data, shape, format='i',
                                         flags=_testbuffer.ND_PIL)
            # 准备用于匹配的错误信息
            msg = "NumPy currently does not support.*suboffsets"
            
            # 使用 pytest.raises 检查是否会抛出 BufferError 异常，并匹配指定的错误信息
            with pytest.raises(BufferError, match=msg):
                # 尝试使用 np.asarray 将 buffer 转换为 NumPy 数组
                np.asarray(buffer)
            with pytest.raises(BufferError, match=msg):
                # 尝试使用 np.asarray 将包含 buffer 的列表转换为 NumPy 数组
                np.asarray([buffer])

            # 另外检查 np.frombuffer 是否会抛出 BufferError 异常
            with pytest.raises(BufferError):
                np.frombuffer(buffer)
    class TestArrayCreationCopyArgument(object):

        class RaiseOnBool:

            def __bool__(self):
                raise ValueError

        # 定义包含True值的列表，用于测试
        true_vals = [True, np._CopyMode.ALWAYS, np.True_]
        # 定义包含IF_NEEDED值的列表，用于测试
        if_needed_vals = [None, np._CopyMode.IF_NEEDED]
        # 定义包含False值的列表，用于测试
        false_vals = [False, np._CopyMode.NEVER, np.False_]

        def test_scalars(self):
            # 测试numpy和python标量
            for dtype in np.typecodes["All"]:
                # 创建dtype类型的空数组
                arr = np.zeros((), dtype=dtype)
                # 获取数组的标量值
                scalar = arr[()]
                # 获取数组的python标量值
                pyscalar = arr.item(0)

                # 测试不允许复制的情况抛出错误
                assert_raises(ValueError, np.array, pyscalar,
                                copy=self.RaiseOnBool())
                assert_raises(ValueError, _multiarray_tests.npy_ensurenocopy,
                                [1])
                for copy in self.false_vals:
                    # 测试不允许复制的情况抛出错误
                    assert_raises(ValueError, np.array, scalar, copy=copy)
                    assert_raises(ValueError, np.array, pyscalar, copy=copy)
                    # 对于指定dtype的类型转换（如无符号整数），可能有特殊情况：
                    with pytest.raises(ValueError):
                        np.array(pyscalar, dtype=np.int64, copy=copy)

        def test_compatible_cast(self):

            # 一些类型即使不同，也是兼容的，不需要复制。对于一些整数类型来说，这是成立的。
            def int_types(byteswap=False):
                int_types = (np.typecodes["Integer"] +
                             np.typecodes["UnsignedInteger"])
                for int_type in int_types:
                    yield np.dtype(int_type)
                    if byteswap:
                        yield np.dtype(int_type).newbyteorder()

            for int1 in int_types():
                for int2 in int_types(True):
                    # 创建int1类型的数组
                    arr = np.arange(10, dtype=int1)

                    for copy in self.true_vals:
                        # 测试使用不同dtype进行复制操作
                        res = np.array(arr, copy=copy, dtype=int2)
                        assert res is not arr and res.flags.owndata
                        assert_array_equal(res, arr)

                    if int1 == int2:
                        # 对于相同类型，不需要类型转换，基本检查足够
                        for copy in self.if_needed_vals:
                            res = np.array(arr, copy=copy, dtype=int2)
                            assert res is arr or res.base is arr

                        for copy in self.false_vals:
                            res = np.array(arr, copy=copy, dtype=int2)
                            assert res is arr or res.base is arr

                    else:
                        # 需要类型转换，确保复制操作有效
                        for copy in self.if_needed_vals:
                            res = np.array(arr, copy=copy, dtype=int2)
                            assert res is not arr and res.flags.owndata
                            assert_array_equal(res, arr)

                        # 断言不允许复制的情况抛出错误
                        assert_raises(ValueError, np.array,
                                      arr, copy=False,
                                      dtype=int2)
    def test_buffer_interface(self):
        # Buffer interface gives direct memory access (no copy)
        arr = np.arange(10)
        view = memoryview(arr)

        # Checking bases is a bit tricky since numpy creates another
        # memoryview, so use may_share_memory.
        for copy in self.true_vals:
            # Create a new numpy array from the memoryview, optionally copying data
            res = np.array(view, copy=copy)
            # Assert that res does not share memory with arr
            assert not np.may_share_memory(arr, res)
        for copy in self.false_vals:
            # Create a new numpy array from the memoryview, optionally copying data
            res = np.array(view, copy=copy)
            # Assert that res shares memory with arr
            assert np.may_share_memory(arr, res)
        # Create a numpy array from the memoryview without copying data
        res = np.array(view, copy=np._CopyMode.NEVER)
        # Assert that res does not share memory with arr
        assert np.may_share_memory(arr, res) == True

    def test_array_interfaces(self):
        base_arr = np.arange(10)

        # Array interface gives direct memory access (much like a memoryview)
        class ArrayLike:
            # Define the array interface for ArrayLike class
            __array_interface__ = base_arr.__array_interface__

        arr = ArrayLike()

        for copy, val in [(True, None), (np._CopyMode.ALWAYS, None),
                          (False, arr), (np._CopyMode.IF_NEEDED, arr),
                          (np._CopyMode.NEVER, arr)]:
            # Create a numpy array from ArrayLike object, optionally copying data
            res = np.array(arr, copy=copy)
            # Assert that the base attribute of res matches val
            assert res.base is val

    def test___array__(self):
        base_arr = np.arange(10)

        class ArrayLike:
            def __array__(self, dtype=None, copy=None):
                # Implement __array__ method to return base_arr
                return base_arr

        arr = ArrayLike()

        for copy in self.true_vals:
            # Create a numpy array from ArrayLike object, optionally copying data
            res = np.array(arr, copy=copy)
            # Assert that res is equal to base_arr
            assert_array_equal(res, base_arr)
            # Assert that res is base_arr due to the behavior of __array__ method
            assert res is base_arr

        for copy in self.if_needed_vals + self.false_vals:
            # Create a numpy array from ArrayLike object, optionally copying data
            res = np.array(arr, copy=copy)
            # Assert that res is equal to base_arr
            assert_array_equal(res, base_arr)
            # Assert that res is base_arr due to the behavior of __array__ method
            assert res is base_arr  # numpy trusts the ArrayLike
    # 定义测试方法，测试对象的 __array__ 方法实现
    def test___array__copy_arg(self):
        # 创建一个 10x10 的整数数组 a，所有元素为 1
        a = np.ones((10, 10), dtype=int)

        # 断言 a 和 a 的 __array__() 共享内存
        assert np.shares_memory(a, a.__array__())
        # 断言 a 和 a 的 __array__(float) 不共享内存
        assert not np.shares_memory(a, a.__array__(float))
        # 断言 a 和 a 的 __array__(float, copy=None) 不共享内存
        assert not np.shares_memory(a, a.__array__(float, copy=None))
        # 断言 a 和 a 的 __array__(copy=True) 不共享内存
        assert not np.shares_memory(a, a.__array__(copy=True))
        # 断言 a 和 a 的 __array__(copy=None) 共享内存
        assert np.shares_memory(a, a.__array__(copy=None))
        # 断言 a 和 a 的 __array__(copy=False) 共享内存
        assert np.shares_memory(a, a.__array__(copy=False))
        # 断言 a 和 a 的 __array__(int, copy=False) 共享内存
        assert np.shares_memory(a, a.__array__(int, copy=False))
        # 使用 pytest.raises 检查是否抛出 ValueError 异常
        with pytest.raises(ValueError):
            np.shares_memory(a, a.__array__(float, copy=False))

        # 创建一个基础数组 base_arr，包含整数范围为 0 到 9
        base_arr = np.arange(10)

        # 定义一个自定义类 ArrayLikeNoCopy
        class ArrayLikeNoCopy:
            # 实现 __array__ 方法，返回 base_arr
            def __array__(self, dtype=None):
                return base_arr

        # 创建一个 ArrayLikeNoCopy 的实例 a
        a = ArrayLikeNoCopy()

        # 使用 copy=None 明确传递参数不应引发警告
        arr = np.array(a, copy=None)
        # 断言 arr 和 base_arr 相等
        assert_array_equal(arr, base_arr)
        # 断言 arr 是 base_arr 的引用
        assert arr is base_arr

        # 在 NumPy 2.1 版本中，显式传递 copy=True 将触发 __array__ 方法的调用，并引发弃用警告
        with pytest.warns(DeprecationWarning,
                          match="__array__.*must implement.*'copy'"):
            arr = np.array(a, copy=True)
        # 断言 arr 和 base_arr 相等
        assert_array_equal(arr, base_arr)
        # 断言 arr 不是 base_arr 的引用
        assert arr is not base_arr

        # 传递 copy=False 将引发弃用警告，并抛出错误
        with pytest.warns(DeprecationWarning, match="__array__.*'copy'"):
            with pytest.raises(ValueError,
                    match=r"Unable to avoid copy(.|\n)*numpy_2_0_migration_guide.html"):
                np.array(a, copy=False)

    # 定义测试方法，测试对象的 __array__ 方法实现（复制一次）
    def test___array__copy_once(self):
        # 设置数组大小为 100
        size = 100
        # 创建大小为 (size, size) 的全零基础数组 base_arr 和复制数组 copy_arr
        base_arr = np.zeros((size, size))
        copy_arr = np.zeros((size, size))

        # 定义一个 ArrayRandom 类
        class ArrayRandom:
            # 构造方法初始化 true_passed 属性为 False
            def __init__(self):
                self.true_passed = False

            # 实现 __array__ 方法
            def __array__(self, dtype=None, copy=None):
                # 如果 copy 为真，则将 true_passed 设置为 True 并返回 copy_arr
                if copy:
                    self.true_passed = True
                    return copy_arr
                # 否则返回 base_arr
                else:
                    return base_arr

        # 创建 ArrayRandom 类的实例 arr_random
        arr_random = ArrayRandom()

        # 使用 copy=True 复制数组，断言 true_passed 为 True，并且结果是 copy_arr
        first_copy = np.array(arr_random, copy=True)
        assert arr_random.true_passed
        assert first_copy is copy_arr

        # 重新创建 ArrayRandom 类的实例 arr_random
        arr_random = ArrayRandom()

        # 使用 copy=False 不复制数组，断言 true_passed 为 False，并且结果是 base_arr
        no_copy = np.array(arr_random, copy=False)
        assert not arr_random.true_passed
        assert no_copy is base_arr

        # 重新创建 ArrayRandom 类的实例 arr_random
        arr_random = ArrayRandom()

        # 使用 copy=True 传递数组，不应触发 true_passed
        _ = np.array([arr_random], copy=True)
        assert not arr_random.true_passed

        # 重新创建 ArrayRandom 类的实例 arr_random
        arr_random = ArrayRandom()

        # 使用 copy=True 和 order="F" 复制数组，断言 true_passed 为 True，并且结果不是 copy_arr
        second_copy = np.array(arr_random, copy=True, order="F")
        assert arr_random.true_passed
        assert second_copy is not copy_arr
    # 定义一个测试函数，用于检查在特定情况下是否会发生数组引用泄漏
    def test__array__reference_leak(self):
        # 定义一个名为 NotAnArray 的类，该类没有实现 __array__ 方法
        class NotAnArray:
            def __array__(self, dtype=None, copy=None):
                raise NotImplementedError()

        # 创建 NotAnArray 的一个实例 x
        x = NotAnArray()

        # 获取 x 的引用计数
        refcount = sys.getrefcount(x)

        try:
            # 尝试将 x 转换为 numpy 数组，预期会抛出 NotImplementedError 异常
            np.array(x)
        except NotImplementedError:
            pass

        # 手动触发垃圾回收
        gc.collect()

        # 断言 x 的引用计数没有发生变化
        assert refcount == sys.getrefcount(x)

    @pytest.mark.parametrize(
            "arr", [np.ones(()), np.arange(81).reshape((9, 9))])
    @pytest.mark.parametrize("order1", ["C", "F", None])
    @pytest.mark.parametrize("order2", ["C", "F", "A", "K"])
    def test_order_mismatch(self, arr, order1, order2):
        # 主要讨论数组的顺序(order)可能导致无需复制失败的情况
        # 准备 C 顺序、F 顺序和非连续数组:
        arr = arr.copy(order1)

        if order1 == "C":
            # 断言数组是 C 连续的
            assert arr.flags.c_contiguous
        elif order1 == "F":
            # 断言数组是 F 连续的
            assert arr.flags.f_contiguous
        elif arr.ndim != 0:
            # 使数组变成非连续数组
            arr = arr[::2, ::2]
            # 断言数组不是强制连续的
            assert not arr.flags.forc

        # 根据数组的顺序(order2)判断是否需要复制
        if order2 == "C":
            no_copy_necessary = arr.flags.c_contiguous
        elif order2 == "F":
            no_copy_necessary = arr.flags.f_contiguous
        else:
            # Keeporder 和 Anyorder 对于非连续输出是可以接受的
            # 这与 astype 方法的行为不一致，astype 方法对 "A" 强制要求连续性
            # 这可能是历史遗留问题，"K" 选项之前不存在时可能会出现这种情况
            no_copy_necessary = True

        # 对数组和内存视图(memoryview)进行测试
        for view in [arr, memoryview(arr)]:
            for copy in self.true_vals:
                # 根据给定的复制选项和顺序(order2)创建一个新的数组 res
                res = np.array(view, copy=copy, order=order2)
                # 断言 res 不是 arr，且 res 拥有自己的数据副本
                assert res is not arr and res.flags.owndata
                # 断言 arr 和 res 相等
                assert_array_equal(arr, res)

            if no_copy_necessary:
                for copy in self.if_needed_vals + self.false_vals:
                    # 如果无需复制，创建新的数组 res
                    res = np.array(view, copy=copy, order=order2)
                    # 对于非 PyPy，断言 res 是 arr 或者 res.base.obj 是 arr
                    if not IS_PYPY:
                        assert res is arr or res.base.obj is arr
            else:
                for copy in self.if_needed_vals:
                    # 如果需要复制，创建新的数组 res
                    res = np.array(arr, copy=copy, order=order2)
                    # 断言 arr 和 res 相等
                    assert_array_equal(arr, res)
                for copy in self.false_vals:
                    # 如果不允许复制，断言会引发 ValueError 异常
                    assert_raises(ValueError, np.array,
                                  view, copy=copy, order=order2)
    # 定义一个测试方法，验证不支持的数组步幅情况
    def test_striding_not_ok(self):
        # 创建一个二维数组
        arr = np.array([[1, 2, 4], [3, 4, 5]])
        # 断言：尝试创建数组的转置，但不允许复制，使用 C 风格顺序存储
        assert_raises(ValueError, np.array,
                      arr.T, copy=np._CopyMode.NEVER,
                      order='C')
        # 断言：尝试创建数组的转置，但不允许复制，使用 C 风格顺序存储，指定数据类型为 np.int64
        assert_raises(ValueError, np.array,
                      arr.T, copy=np._CopyMode.NEVER,
                      order='C', dtype=np.int64)
        # 断言：尝试创建数组，但不允许复制，使用 Fortran 风格顺序存储
        assert_raises(ValueError, np.array,
                      arr, copy=np._CopyMode.NEVER,
                      order='F')
        # 断言：尝试创建数组，但不允许复制，使用 Fortran 风格顺序存储，指定数据类型为 np.int64
        assert_raises(ValueError, np.array,
                      arr, copy=np._CopyMode.NEVER,
                      order='F', dtype=np.int64)
class TestArrayAttributeDeletion:

    def test_multiarray_writable_attributes_deletion(self):
        # 创建一个长度为2的全1数组
        a = np.ones(2)
        # 定义要删除的可写属性列表
        attr = ['shape', 'strides', 'data', 'dtype', 'real', 'imag', 'flat']
        # 忽略警告上下文，捕获 DeprecationWarning 类型的警告
        with suppress_warnings() as sup:
            # 过滤掉关于 'data' 属性赋值的警告信息
            sup.filter(DeprecationWarning, "Assigning the 'data' attribute")
            # 遍历属性列表，验证删除每个属性时是否会引发 AttributeError 异常
            for s in attr:
                assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_not_writable_attributes_deletion(self):
        # 创建一个长度为2的全1数组
        a = np.ones(2)
        # 定义要删除的不可写属性列表
        attr = ["ndim", "flags", "itemsize", "size", "nbytes", "base",
                "ctypes", "T", "__array_interface__", "__array_struct__",
                "__array_priority__", "__array_finalize__"]
        # 遍历属性列表，验证删除每个属性时是否会引发 AttributeError 异常
        for s in attr:
            assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_flags_writable_attribute_deletion(self):
        # 创建一个长度为2的全1数组的 flags 对象
        a = np.ones(2).flags
        # 定义要删除的 flags 可写属性列表
        attr = ['writebackifcopy', 'updateifcopy', 'aligned', 'writeable']
        # 遍历属性列表，验证删除每个属性时是否会引发 AttributeError 异常
        for s in attr:
            assert_raises(AttributeError, delattr, a, s)

    def test_multiarray_flags_not_writable_attribute_deletion(self):
        # 创建一个长度为2的全1数组的 flags 对象
        a = np.ones(2).flags
        # 定义要删除的 flags 不可写属性列表
        attr = ["contiguous", "c_contiguous", "f_contiguous", "fortran",
                "owndata", "fnc", "forc", "behaved", "carray", "farray",
                "num"]
        # 遍历属性列表，验证删除每个属性时是否会引发 AttributeError 异常
        for s in attr:
            assert_raises(AttributeError, delattr, a, s)


class TestArrayInterface():
    class Foo:
        def __init__(self, value):
            self.value = value
            # 初始化接口字典，定义类型字符串为 'f8'
            self.iface = {'typestr': 'f8'}

        def __float__(self):
            return float(self.value)

        @property
        def __array_interface__(self):
            # 返回接口字典
            return self.iface


    f = Foo(0.5)

    @pytest.mark.parametrize('val, iface, expected', [
        # 参数化测试，验证不同情况下的数组接口行为
        (f, {}, 0.5),
        ([f], {}, [0.5]),
        ([f, f], {}, [0.5, 0.5]),
        (f, {'shape': ()}, 0.5),
        (f, {'shape': None}, TypeError),
        (f, {'shape': (1, 1)}, [[0.5]]),
        (f, {'shape': (2,)}, ValueError),
        (f, {'strides': ()}, 0.5),
        (f, {'strides': (2,)}, ValueError),
        (f, {'strides': 16}, TypeError),
        ])
    def test_scalar_interface(self, val, iface, expected):
        # 测试标量在数组接口内的强制转换行为
        self.f.iface = {'typestr': 'f8'}
        self.f.iface.update(iface)
        if HAS_REFCOUNT:
            # 如果支持引用计数，则获取 'f8' 数据类型的初始引用计数
            pre_cnt = sys.getrefcount(np.dtype('f8'))
        if isinstance(expected, type):
            # 预期结果是异常类型时，验证 np.array(val) 是否引发异常
            assert_raises(expected, np.array, val)
        else:
            # 否则，验证 np.array(val) 的结果与期望值相等，并且数据类型为 'f8'
            result = np.array(val)
            assert_equal(np.array(val), expected)
            assert result.dtype == 'f8'
            del result
        if HAS_REFCOUNT:
            # 如果支持引用计数，则验证操作后 'f8' 数据类型的引用计数是否不变
            post_cnt = sys.getrefcount(np.dtype('f8'))
            assert_equal(pre_cnt, post_cnt)

def test_interface_no_shape():
    class ArrayLike:
        # 创建一个包含整数1的数组
        array = np.array(1)
        # 定义数组接口为 array 的数组接口
        __array_interface__ = array.__array_interface__
    # 使用断言检查调用 np.array() 函数对空的 ArrayLike 对象的返回结果是否等于 1
    assert_equal(np.array(ArrayLike()), 1)
# 测试函数，用于验证数组接口中的 itemsize 属性
def test_array_interface_itemsize():
    # 标记 GitHub 问题编号为 gh-6361
    my_dtype = np.dtype({'names': ['A', 'B'], 'formats': ['f4', 'f4'],
                         'offsets': [0, 8], 'itemsize': 16})
    # 创建一个元素为 1 的数组，指定自定义数据类型 my_dtype
    a = np.ones(10, dtype=my_dtype)
    # 从数组接口中获取描述符信息并转换为数据类型对象
    descr_t = np.dtype(a.__array_interface__['descr'])
    # 从数组接口中获取类型字符串信息并转换为数据类型对象
    typestr_t = np.dtype(a.__array_interface__['typestr'])
    # 断言两种类型的 itemsize 属性相等
    assert_equal(descr_t.itemsize, typestr_t.itemsize)


# 测试函数，用于验证空 shape 的数组接口行为
def test_array_interface_empty_shape():
    # 标记 GitHub 问题编号为 gh-7994
    arr = np.array([1, 2, 3])
    # 创建数组接口的字典副本
    interface1 = dict(arr.__array_interface__)
    # 将接口的 shape 属性设为空元组
    interface1['shape'] = ()

    # 定义一个模拟数组类，使用修改后的接口字典
    class DummyArray1:
        __array_interface__ = interface1

    # 创建第二个接口字典副本，并修改其 data 属性为数组元素的字节表示
    interface2 = dict(interface1)
    interface2['data'] = arr[0].tobytes()

    # 定义第二个模拟数组类，使用修改后的接口字典
    class DummyArray2:
        __array_interface__ = interface2

    # 从模拟数组类创建数组，并进行断言比较
    arr1 = np.asarray(DummyArray1())
    arr2 = np.asarray(DummyArray2())
    arr3 = arr[:1].reshape(())
    assert_equal(arr1, arr2)
    assert_equal(arr1, arr3)


# 测试函数，用于验证数组接口中的 offset 属性
def test_array_interface_offset():
    arr = np.array([1, 2, 3], dtype='int32')
    # 创建数组接口的字典副本
    interface = dict(arr.__array_interface__)
    # 将接口的 data 属性设为数组的 memoryview 对象
    interface['data'] = memoryview(arr)
    # 修改接口的 shape 属性为元组 (2,)
    interface['shape'] = (2,)
    # 设置接口的 offset 属性为 4

    # 定义一个模拟数组类，使用修改后的接口字典
    class DummyArray:
        __array_interface__ = interface

    # 从模拟数组类创建数组，并进行断言比较
    arr1 = np.asarray(DummyArray())
    assert_equal(arr1, arr[1:])


# 测试函数，用于验证数组接口中的 typestr 属性
def test_array_interface_unicode_typestr():
    arr = np.array([1, 2, 3], dtype='int32')
    # 创建数组接口的字典副本
    interface = dict(arr.__array_interface__)
    # 将接口的 typestr 属性设为 Unicode 字符 '\N{check mark}'

    # 定义一个模拟数组类，使用修改后的接口字典
    class DummyArray:
        __array_interface__ = interface

    # 断言创建模拟数组类时会引发 TypeError 异常
    # 因为设置 shape=() 的情况下，如果数据是支持缓冲区接口的对象（如 Py2 的 str/Py3 的 bytes），
    # 则会触发此 bug 的测试，即不允许 shape=() 的情况
    with pytest.raises(TypeError):
        np.asarray(DummyArray())


# 测试函数，用于验证对一维数组元素删除的行为
def test_flat_element_deletion():
    # 获取一维数组的迭代器
    it = np.ones(3).flat
    try:
        # 尝试删除迭代器中的一个元素
        del it[1]
        # 尝试删除迭代器中的一个切片
        del it[1:2]
    except TypeError:
        pass
    except Exception:
        raise AssertionError


# 测试函数，用于验证对标量元素删除的行为
def test_scalar_element_deletion():
    a = np.zeros(2, dtype=[('x', 'int'), ('y', 'int')])
    # 断言删除标量元素 'x' 时会引发 ValueError 异常
    assert_raises(ValueError, a[0].__delitem__, 'x')


# TestAsCArray 类，用于测试 _multiarray_tests.test_as_c_array 函数的行为
class TestAsCArray:
    # 用于测试一维数组的转换行为
    def test_1darray(self):
        # 创建一个双精度浮点型的一维数组
        array = np.arange(24, dtype=np.double)
        # 调用 C 函数 _multiarray_tests.test_as_c_array 进行转换
        from_c = _multiarray_tests.test_as_c_array(array, 3)
        # 断言转换后的结果与原数组索引为 3 的元素相等
        assert_equal(array[3], from_c)

    # 用于测试二维数组的转换行为
    def test_2darray(self):
        # 创建一个双精度浮点型的二维数组
        array = np.arange(24, dtype=np.double).reshape(3, 8)
        # 调用 C 函数 _multiarray_tests.test_as_c_array 进行转换
        from_c = _multiarray_tests.test_as_c_array(array, 2, 4)
        # 断言转换后的结果与原数组索引为 (2, 4) 的元素相等
        assert_equal(array[2, 4], from_c)

    # 用于测试三维数组的转换行为
    def test_3darray(self):
        # 创建一个双精度浮点型的三维数组
        array = np.arange(24, dtype=np.double).reshape(2, 3, 4)
        # 调用 C 函数 _multiarray_tests.test_as_c_array 进行转换
        from_c = _multiarray_tests.test_as_c_array(array, 1, 2, 3)
        # 断言转换后的结果与原数组索引为 (1, 2, 3) 的元素相等
        assert_equal(array[1, 2, 3], from_c)
    # 定义一个测试函数，用于测试数组与标量的关系运算
    def test_array_scalar_relational_operation(self):
        # 对所有整数类型进行测试
        for dt1 in np.typecodes['AllInteger']:
            # 断言：1 大于数组形式的0（使用指定数据类型），如果不成立则输出失败信息
            assert_(1 > np.array(0, dtype=dt1), "type %s failed" % (dt1,))
            # 断言：1 不小于数组形式的0（使用指定数据类型），如果不成立则输出失败信息
            assert_(not 1 < np.array(0, dtype=dt1), "type %s failed" % (dt1,))

            # 对每个整数类型再次进行测试
            for dt2 in np.typecodes['AllInteger']:
                # 断言：数组形式的1（使用第一种数据类型）大于数组形式的0（使用第二种数据类型），如果不成立则输出失败信息
                assert_(np.array(1, dtype=dt1) > np.array(0, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                # 断言：数组形式的1（使用第一种数据类型）不小于数组形式的0（使用第二种数据类型），如果不成立则输出失败信息
                assert_(not np.array(1, dtype=dt1) < np.array(0, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))

        # 对无符号整数类型进行测试
        for dt1 in 'BHILQP':
            # 断言：-1 小于数组形式的1（使用指定数据类型），如果不成立则输出失败信息
            assert_(-1 < np.array(1, dtype=dt1), "type %s failed" % (dt1,))
            # 断言：-1 不大于数组形式的1（使用指定数据类型），如果不成立则输出失败信息
            assert_(not -1 > np.array(1, dtype=dt1), "type %s failed" % (dt1,))
            # 断言：-1 不等于数组形式的1（使用指定数据类型），如果不成立则输出失败信息
            assert_(-1 != np.array(1, dtype=dt1), "type %s failed" % (dt1,))

            # 对有符号整数与无符号整数进行测试
            for dt2 in 'bhilqp':
                # 断言：数组形式的1（使用第一种数据类型）大于数组形式的-1（使用第二种数据类型），如果不成立则输出失败信息
                assert_(np.array(1, dtype=dt1) > np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                # 断言：数组形式的1（使用第一种数据类型）不小于数组形式的-1（使用第二种数据类型），如果不成立则输出失败信息
                assert_(not np.array(1, dtype=dt1) < np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                # 断言：数组形式的1（使用第一种数据类型）不等于数组形式的-1（使用第二种数据类型），如果不成立则输出失败信息
                assert_(np.array(1, dtype=dt1) != np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))

        # 对有符号整数与浮点数类型进行测试
        for dt1 in 'bhlqp' + np.typecodes['Float']:
            # 断言：1 大于数组形式的-1（使用指定数据类型），如果不成立则输出失败信息
            assert_(1 > np.array(-1, dtype=dt1), "type %s failed" % (dt1,))
            # 断言：1 不小于数组形式的-1（使用指定数据类型），如果不成立则输出失败信息
            assert_(not 1 < np.array(-1, dtype=dt1), "type %s failed" % (dt1,))
            # 断言：-1 等于数组形式的-1（使用指定数据类型），如果不成立则输出失败信息
            assert_(-1 == np.array(-1, dtype=dt1), "type %s failed" % (dt1,))

            # 对每个整数与浮点数类型再次进行测试
            for dt2 in 'bhlqp' + np.typecodes['Float']:
                # 断言：数组形式的1（使用第一种数据类型）大于数组形式的-1（使用第二种数据类型），如果不成立则输出失败信息
                assert_(np.array(1, dtype=dt1) > np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                # 断言：数组形式的1（使用第一种数据类型）不小于数组形式的-1（使用第二种数据类型），如果不成立则输出失败信息
                assert_(not np.array(1, dtype=dt1) < np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
                # 断言：数组形式的-1（使用第一种数据类型）等于数组形式的-1（使用第二种数据类型），如果不成立则输出失败信息
                assert_(np.array(-1, dtype=dt1) == np.array(-1, dtype=dt2),
                        "type %s and %s failed" % (dt1, dt2))
    def test_to_bool_scalar(self):
        # 断言单个布尔数组为 False
        assert_equal(bool(np.array([False])), False)
        # 断言单个布尔数组为 True
        assert_equal(bool(np.array([True])), True)
        # 断言包含整数的数组为 True
        assert_equal(bool(np.array([[42]])), True)
        # 断言传入数组中包含多个元素引发 ValueError
        assert_raises(ValueError, bool, np.array([1, 2]))

        class NotConvertible:
            def __bool__(self):
                raise NotImplementedError

        # 断言自定义类 NotConvertible 的实例在转换为布尔值时引发 NotImplementedError
        assert_raises(NotImplementedError, bool, np.array(NotConvertible()))
        # 断言包含 NotConvertible 实例的数组在转换为布尔值时引发 NotImplementedError
        assert_raises(NotImplementedError, bool, np.array([NotConvertible()]))
        # 如果是 Pyston 平台，则跳过当前测试，因为其禁用递归检查
        if IS_PYSTON:
            pytest.skip("Pyston disables recursion checking")

        # 创建一个自引用的数组
        self_containing = np.array([None])
        self_containing[0] = self_containing

        Error = RecursionError

        # 断言尝试将自引用数组转换为布尔值时引发递归错误
        assert_raises(Error, bool, self_containing)  # previously stack overflow
        # 解决自引用的循环引用问题，将第一个元素设为 None
        self_containing[0] = None  # resolve circular reference

    def test_to_int_scalar(self):
        # gh-9972 意味着这些情况并不总是相同
        int_funcs = (int, lambda x: x.__int__())
        for int_func in int_funcs:
            # 断言将数组中的零转换为整数是零
            assert_equal(int_func(np.array(0)), 0)
            # 期望 DeprecationWarning 警告
            with assert_warns(DeprecationWarning):
                # 断言将包含单个整数的数组转换为整数得到该整数
                assert_equal(int_func(np.array([1])), 1)
            with assert_warns(DeprecationWarning):
                # 断言将包含整数的二维数组转换为整数得到该整数
                assert_equal(int_func(np.array([[42]])), 42)
            # 断言尝试将包含多个元素的数组转换为整数引发 TypeError
            assert_raises(TypeError, int_func, np.array([1, 2]))

            # gh-9972
            # 断言将字符串 '4' 转换为整数为 4
            assert_equal(4, int_func(np.array('4')))
            # 断言将字节序列 b'5' 转换为整数为 5
            assert_equal(5, int_func(np.bytes_(b'5')))
            # 断言将字符串 '6' 转换为整数为 6
            assert_equal(6, int_func(np.str_('6')))

            # Python 3.11 中将 int() 委托给 __trunc__ 已被弃用
            if sys.version_info < (3, 11):
                class HasTrunc:
                    def __trunc__(self):
                        return 3
                # 断言将 HasTrunc 类的实例转换为整数得到 3
                assert_equal(3, int_func(np.array(HasTrunc())))
                with assert_warns(DeprecationWarning):
                    # 断言将包含 HasTrunc 实例的数组转换为整数得到 3
                    assert_equal(3, int_func(np.array([HasTrunc()])))
            else:
                pass

            class NotConvertible:
                def __int__(self):
                    raise NotImplementedError
            # 断言将 NotConvertible 类的实例转换为整数时引发 NotImplementedError
            assert_raises(NotImplementedError, int_func, np.array(NotConvertible()))
            with assert_warns(DeprecationWarning):
                # 断言将包含 NotConvertible 实例的数组转换为整数时引发 NotImplementedError
                assert_raises(NotImplementedError, int_func, np.array([NotConvertible()]))
# 定义一个名为 TestWhere 的测试类
class TestWhere:
    # 定义测试基础功能的方法
    def test_basic(self):
        # 定义一组数据类型列表，包括布尔类型和不同的 NumPy 数据类型
        dts = [bool, np.int16, np.int32, np.int64, np.double, np.complex128,
               np.longdouble, np.clongdouble]
        # 遍历数据类型列表中的每个数据类型
        for dt in dts:
            # 创建一个长度为 53 的布尔类型的数组
            c = np.ones(53, dtype=bool)
            # 使用 np.where 进行条件判断，将符合条件的值替换为 dt(0)，不符合条件的替换为 dt(1)，并进行断言比较
            assert_equal(np.where( c, dt(0), dt(1)), dt(0))
            assert_equal(np.where(~c, dt(0), dt(1)), dt(1))
            assert_equal(np.where(True, dt(0), dt(1)), dt(0))
            assert_equal(np.where(False, dt(0), dt(1)), dt(1))
            # 创建一个与 c 形状相同并转换为当前数据类型 dt 的数组
            d = np.ones_like(c).astype(dt)
            # 创建一个与 d 形状相同的零数组 e
            e = np.zeros_like(d)
            # 将 d 转换为当前数据类型 dt 的数组，并将其中第 7 个元素的值设置为 e 的第 7 个元素的值
            r = d.astype(dt)
            c[7] = False
            r[7] = e[7]
            # 使用 np.where 进行条件判断，断言比较结果是否符合预期
            assert_equal(np.where(c, e, e), e)
            assert_equal(np.where(c, d, e), r)
            assert_equal(np.where(c, d, e[0]), r)
            assert_equal(np.where(c, d[0], e), r)
            assert_equal(np.where(c[::2], d[::2], e[::2]), r[::2])
            assert_equal(np.where(c[1::2], d[1::2], e[1::2]), r[1::2])
            assert_equal(np.where(c[::3], d[::3], e[::3]), r[::3])
            assert_equal(np.where(c[1::3], d[1::3], e[1::3]), r[1::3])
            assert_equal(np.where(c[::-2], d[::-2], e[::-2]), r[::-2])
            assert_equal(np.where(c[::-3], d[::-3], e[::-3]), r[::-3])
            assert_equal(np.where(c[1::-3], d[1::-3], e[1::-3]), r[1::-3])

    @pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
    def test_exotic(self):
        # 使用 np.where() 函数测试条件为 True 的情况，期望结果为 np.array(None)
        assert_array_equal(np.where(True, None, None), np.array(None))
        
        # 创建空的 numpy 数组，并按指定形状重新调整，期望结果为空数组
        m = np.array([], dtype=bool).reshape(0, 3)
        b = np.array([], dtype=np.float64).reshape(0, 3)
        assert_array_equal(np.where(m, 0, b), np.array([]).reshape(0, 3))

        # 创建包含 NaN 值的 numpy 数组，并进行条件测试和数组操作
        d = np.array([-1.34, -0.16, -0.54, -0.31, -0.08, -0.95, 0.000, 0.313,
                      0.547, -0.18, 0.876, 0.236, 1.969, 0.310, 0.699, 1.013,
                      1.267, 0.229, -1.39, 0.487])
        nan = float('NaN')
        e = np.array(['5z', '0l', nan, 'Wz', nan, nan, 'Xq', 'cs', nan, nan,
                     'QN', nan, nan, 'Fd', nan, nan, 'kp', nan, '36', 'i1'],
                     dtype=object)
        m = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1,
                      0, 1, 1, 0, 1, 1, 0, 1, 0, 0], dtype=bool)

        # 复制数组 e 的副本，并根据条件 m 对其中的元素进行替换操作
        r = e[:]
        r[np.where(m)] = d[np.where(m)]
        assert_array_equal(np.where(m, d, e), r)

        # 复制数组 e 的副本，并根据条件 ~m 对其中的元素进行替换操作
        r = e[:]
        r[np.where(~m)] = d[np.where(~m)]
        assert_array_equal(np.where(m, e, d), r)

        # 测试 np.where() 在条件全部为 True 时返回原始数组 e
        assert_array_equal(np.where(m, e, e), e)

        # 测试 np.where() 在返回最小 dtype 的结果，并验证返回结果的数据类型为 np.float32
        d = np.array([1., 2.], dtype=np.float32)
        e = float('NaN')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float('Infinity')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        e = float('-Infinity')
        assert_equal(np.where(True, d, e).dtype, np.float32)
        # 使用大数值 e 进行测试，预期会引发 RuntimeWarning 并返回 np.float32 类型结果
        e = float(1e150)
        with pytest.warns(RuntimeWarning, match="overflow"):
            res = np.where(True, d, e)
        assert res.dtype == np.float32

    def test_ndim(self):
        c = [True, False]
        a = np.zeros((2, 25))
        b = np.ones((2, 25))
        
        # 使用 np.where() 处理多维数组，验证条件为 True 时结果为 a，条件为 False 时结果为 b
        r = np.where(np.array(c)[:,np.newaxis], a, b)
        assert_array_equal(r[0], a[0])
        assert_array_equal(r[1], b[0])

        # 对 a 和 b 进行转置操作，再次使用 np.where() 处理条件数组 c
        r = np.where(c, a.T, b.T)
        assert_array_equal(r[:,0], a[:,0])
        assert_array_equal(r[:,1], b[:,0])

    def test_dtype_mix(self):
        c = np.array([False, True, False, False, False, False, True, False,
                     False, False, True, False])
        a = np.uint32(1)
        b = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.],
                      dtype=np.float64)
        r = np.array([5., 1., 3., 2., -1., -4., 1., -10., 10., 1., 1., 3.],
                     dtype=np.float64)
        
        # 测试混合 dtype 的情况下，使用 np.where() 进行条件替换操作
        assert_equal(np.where(c, a, b), r)

        # 将 a 转换为 np.float32 类型，b 转换为 np.int64 类型后继续测试
        a = a.astype(np.float32)
        b = b.astype(np.int64)
        assert_equal(np.where(c, a, b), r)

        # 非布尔类型的条件数组 c，进行相应操作后继续测试
        c = c.astype(int)
        c[c != 0] = 34242324
        assert_equal(np.where(c, a, b), r)
        
        # 反转条件数组 c，并进行测试
        tmpmask = c != 0
        c[c == 0] = 41247212
        c[tmpmask] = 0
        assert_equal(np.where(c, b, a), r)
    def test_foreign(self):
        # 创建一个布尔数组 c，表示条件
        c = np.array([False, True, False, False, False, False, True, False,
                     False, False, True, False])
        # 创建一个期望结果的数组 r
        r = np.array([5., 1., 3., 2., -1., -4., 1., -10., 10., 1., 1., 3.],
                     dtype=np.float64)
        # 创建一个大端字节顺序的整数数组 a
        a = np.ones(1, dtype='>i4')
        # 创建一个浮点数数组 b，作为备选结果
        b = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.],
                     dtype=np.float64)
        # 断言 np.where 的结果与预期结果 r 相等
        assert_equal(np.where(c, a, b), r)

        # 将数组 b 的数据类型转换为大端字节顺序的双精度浮点数
        b = b.astype('>f8')
        # 再次断言 np.where 的结果与预期结果 r 相等
        assert_equal(np.where(c, a, b), r)

        # 将数组 a 的数据类型转换为小端字节顺序的整数
        a = a.astype('<i4')
        # 再次断言 np.where 的结果与预期结果 r 相等
        assert_equal(np.where(c, a, b), r)

        # 将数组 c 的数据类型转换为大端字节顺序的整数
        c = c.astype('>i4')
        # 再次断言 np.where 的结果与预期结果 r 相等
        assert_equal(np.where(c, a, b), r)

    def test_error(self):
        # 创建一个包含两个布尔值的列表 c
        c = [True, True]
        # 创建两个不同形状的全为1的数组 a 和 b
        a = np.ones((4, 5))
        b = np.ones((5, 5))
        # 断言 np.where 在给定的条件下会引发 ValueError 异常
        assert_raises(ValueError, np.where, c, a, a)
        # 断言 np.where 在给定的条件下会引发 ValueError 异常
        assert_raises(ValueError, np.where, c[0], a, b)

    def test_string(self):
        # 测试字符串数组中 null 字符的填充情况
        # 创建字符串数组 a 和 b
        a = np.array("abc")
        b = np.array("x" * 753)
        # 断言 np.where 的结果与预期的字符串 "abc" 相等
        assert_equal(np.where(True, a, b), "abc")
        # 断言 np.where 的结果与预期的字符串 "abc" 相等
        assert_equal(np.where(False, b, a), "abc")

        # 再次测试原生数据类型大小的字符串
        # 创建字符串数组 a 和 b
        a = np.array("abcd")
        b = np.array("x" * 8)
        # 断言 np.where 的结果与预期的字符串 "abcd" 相等
        assert_equal(np.where(True, a, b), "abcd")
        # 断言 np.where 的结果与预期的字符串 "abcd" 相等
        assert_equal(np.where(False, b, a), "abcd")

    def test_empty_result(self):
        # 通过一个读取空数组数据的赋值来传递空的 where 结果，检测错误，详见 gh-8922
        # 创建一个全为零的数组 x
        x = np.zeros((1, 1))
        # 使用 np.where 寻找 x 中等于 99. 的位置，将结果堆叠成至少二维数组
        ibad = np.vstack(np.where(x == 99.))
        # 断言数组 ibad 与预期的至少二维空数组相等
        assert_array_equal(ibad,
                           np.atleast_2d(np.array([[],[]], dtype=np.intp)))

    def test_largedim(self):
        # 无效读取回归测试 gh-9304
        # 创建一个具有给定形状的随机数组 array
        shape = [10, 2, 3, 4, 5, 6]
        np.random.seed(2)
        array = np.random.rand(*shape)

        # 循环测试 array 的非零元素索引
        for i in range(10):
            benchmark = array.nonzero()
            result = array.nonzero()
            # 断言 benchmark 和 result 的非零元素索引相等
            assert_array_equal(benchmark, result)

    def test_kwargs(self):
        # 创建一个全为零的数组 a
        a = np.zeros(1)
        # 使用 assert_raises 检测带有未知参数的 np.where 调用是否引发 TypeError 异常
        with assert_raises(TypeError):
            np.where(a, x=a, y=a)
if not IS_PYPY:
    # 如果不是在 PyPy 环境下执行以下代码
    # sys.getsizeof() 在 PyPy 上不可用
    class TestSizeOf:
        # 测试类 TestSizeOf，用于测试数组大小

        def test_empty_array(self):
            # 测试空数组的大小
            x = np.array([])
            assert_(sys.getsizeof(x) > 0)

        def check_array(self, dtype):
            # 检查特定数据类型的数组大小
            elem_size = dtype(0).itemsize

            for length in [10, 50, 100, 500]:
                # 创建指定长度和数据类型的数组
                x = np.arange(length, dtype=dtype)
                assert_(sys.getsizeof(x) > length * elem_size)

        def test_array_int32(self):
            # 测试 int32 类型数组的大小
            self.check_array(np.int32)

        def test_array_int64(self):
            # 测试 int64 类型数组的大小
            self.check_array(np.int64)

        def test_array_float32(self):
            # 测试 float32 类型数组的大小
            self.check_array(np.float32)

        def test_array_float64(self):
            # 测试 float64 类型数组的大小
            self.check_array(np.float64)

        def test_view(self):
            # 测试数组切片后的大小
            d = np.ones(100)
            assert_(sys.getsizeof(d[...]) < sys.getsizeof(d))

        def test_reshape(self):
            # 测试数组重塑后的大小
            d = np.ones(100)
            assert_(sys.getsizeof(d) < sys.getsizeof(d.reshape(100, 1, 1).copy()))

        @_no_tracing
        def test_resize(self):
            # 测试数组调整大小后的效果
            d = np.ones(100)
            old = sys.getsizeof(d)
            d.resize(50)
            assert_(old > sys.getsizeof(d))
            d.resize(150)
            assert_(old < sys.getsizeof(d))

        def test_error(self):
            # 测试对数组类型的错误操作
            d = np.ones(100)
            assert_raises(TypeError, d.__sizeof__, "a")


class TestHashing:
    # 哈希测试类

    def test_arrays_not_hashable(self):
        # 测试数组不可哈希
        x = np.ones(3)
        assert_raises(TypeError, hash, x)

    def test_collections_hashable(self):
        # 测试集合可哈希
        x = np.array([])
        assert_(not isinstance(x, collections.abc.Hashable))


class TestArrayPriority:
    # 数组优先级测试类
    # 在 __array_priority__ 定义确定后将删除此类，现用于检查意外更改。

    op = operator
    binary_ops = [
        op.pow, op.add, op.sub, op.mul, op.floordiv, op.truediv, op.mod,
        op.and_, op.or_, op.xor, op.lshift, op.rshift, op.mod, op.gt,
        op.ge, op.lt, op.le, op.ne, op.eq
        ]

    class Foo(np.ndarray):
        # Foo 类继承自 np.ndarray，定义了 __array_priority__

        __array_priority__ = 100.

        def __new__(cls, *args, **kwargs):
            # 创建一个新的 Foo 类实例
            return np.array(*args, **kwargs).view(cls)

    class Bar(np.ndarray):
        # Bar 类继承自 np.ndarray，定义了 __array_priority__

        __array_priority__ = 101.

        def __new__(cls, *args, **kwargs):
            # 创建一个新的 Bar 类实例
            return np.array(*args, **kwargs).view(cls)
    # 定义一个名为 Other 的类
    class Other:
        # 设置特殊属性 __array_priority__ 为 1000.，表示在数组操作中的优先级
        __array_priority__ = 1000.

        # 定义一个私有方法 _all，接受参数 other，返回一个新的同类对象
        def _all(self, other):
            return self.__class__()

        # 使用 _all 方法重载以下运算符，使其对操作数执行相同的操作并返回新的对象
        __add__ = __radd__ = _all  # 加法操作
        __sub__ = __rsub__ = _all  # 减法操作
        __mul__ = __rmul__ = _all  # 乘法操作
        __pow__ = __rpow__ = _all  # 指数操作
        __div__ = __rdiv__ = _all  # 整数除法操作
        __mod__ = __rmod__ = _all  # 求模操作
        __truediv__ = __rtruediv__ = _all  # 真除法操作
        __floordiv__ = __rfloordiv__ = _all  # 向下整除操作
        __and__ = __rand__ = _all  # 位与操作
        __xor__ = __rxor__ = _all  # 位异或操作
        __or__ = __ror__ = _all  # 位或操作
        __lshift__ = __rlshift__ = _all  # 左移操作
        __rshift__ = __rrshift__ = _all  # 右移操作
        __eq__ = _all  # 等于比较操作
        __ne__ = _all  # 不等于比较操作
        __gt__ = _all  # 大于比较操作
        __ge__ = _all  # 大于等于比较操作
        __lt__ = _all  # 小于比较操作
        __le__ = _all  # 小于等于比较操作

    # 定义测试方法 test_ndarray_subclass，用于测试 ndarray 子类的操作
    def test_ndarray_subclass(self):
        # 创建一个包含 [1, 2] 的 numpy 数组 a
        a = np.array([1, 2])
        # 创建一个 self.Bar 类的对象 b，其类型不明确
        b = self.Bar([1, 2])
        # 遍历 self.binary_ops 中的每个操作函数 f
        for f in self.binary_ops:
            # 生成描述消息
            msg = repr(f)
            # 断言 f(a, b) 返回的对象是 self.Bar 类型，使用 msg 作为错误消息
            assert_(isinstance(f(a, b), self.Bar), msg)
            # 断言 f(b, a) 返回的对象是 self.Bar 类型，使用 msg 作为错误消息
            assert_(isinstance(f(b, a), self.Bar), msg)

    # 定义测试方法 test_ndarray_other，用于测试 ndarray 对象和 Other 类型对象之间的操作
    def test_ndarray_other(self):
        # 创建一个包含 [1, 2] 的 numpy 数组 a
        a = np.array([1, 2])
        # 创建一个 self.Other 类的对象 b
        b = self.Other()
        # 遍历 self.binary_ops 中的每个操作函数 f
        for f in self.binary_ops:
            # 生成描述消息
            msg = repr(f)
            # 断言 f(a, b) 返回的对象是 self.Other 类型，使用 msg 作为错误消息
            assert_(isinstance(f(a, b), self.Other), msg)
            # 断言 f(b, a) 返回的对象是 self.Other 类型，使用 msg 作为错误消息
            assert_(isinstance(f(b, a), self.Other), msg)

    # 定义测试方法 test_subclass_subclass，用于测试子类之间的操作
    def test_subclass_subclass(self):
        # 创建一个 self.Foo 类的对象 a，其内容为 [1, 2]
        a = self.Foo([1, 2])
        # 创建一个 self.Bar 类的对象 b，其内容为 [1, 2]
        b = self.Bar([1, 2])
        # 遍历 self.binary_ops 中的每个操作函数 f
        for f in self.binary_ops:
            # 生成描述消息
            msg = repr(f)
            # 断言 f(a, b) 返回的对象是 self.Bar 类型，使用 msg 作为错误消息
            assert_(isinstance(f(a, b), self.Bar), msg)
            # 断言 f(b, a) 返回的对象是 self.Bar 类型，使用 msg 作为错误消息
            assert_(isinstance(f(b, a), self.Bar), msg)

    # 定义测试方法 test_subclass_other，用于测试子类和 Other 类型对象之间的操作
    def test_subclass_other(self):
        # 创建一个 self.Foo 类的对象 a，其内容为 [1, 2]
        a = self.Foo([1, 2])
        # 创建一个 self.Other 类的对象 b
        b = self.Other()
        # 遍历 self.binary_ops 中的每个操作函数 f
        for f in self.binary_ops:
            # 生成描述消息
            msg = repr(f)
            # 断言 f(a, b) 返回的对象是 self.Other 类型，使用 msg 作为错误消息
            assert_(isinstance(f(a, b), self.Other), msg)
            # 断言 f(b, a) 返回的对象是 self.Other 类型，使用 msg 作为错误消息
            assert_(isinstance(f(b, a), self.Other), msg)
class TestBytestringArrayNonzero:

    def test_empty_bstring_array_is_falsey(self):
        # 确保空的字节字符串数组在逻辑上为假
        assert_(not np.array([''], dtype=str))

    def test_whitespace_bstring_array_is_truthy(self):
        # 创建包含一个元素为'spam'的字节字符串数组，并将其第一个元素设为空白字符和空字符
        a = np.array(['spam'], dtype=str)
        a[0] = '  \0\0'
        # 确保字节字符串数组在逻辑上为真
        assert_(a)

    def test_all_null_bstring_array_is_falsey(self):
        # 创建包含一个元素为'spam'的字节字符串数组，并将其第一个元素设为全空字符
        a = np.array(['spam'], dtype=str)
        a[0] = '\0\0\0\0'
        # 确保字节字符串数组在逻辑上为假
        assert_(not a)

    def test_null_inside_bstring_array_is_truthy(self):
        # 创建包含一个元素为'spam'的字节字符串数组，并将其第一个元素设为包含空字符的字符串
        a = np.array(['spam'], dtype=str)
        a[0] = ' \0 \0'
        # 确保字节字符串数组在逻辑上为真
        assert_(a)


class TestUnicodeEncoding:
    """
    Tests for encoding related bugs, such as UCS2 vs UCS4, round-tripping
    issues, etc
    """
    def test_round_trip(self):
        """ Tests that GETITEM, SETITEM, and PyArray_Scalar roundtrip """
        # 测试在不同编码（UCS2 vs UCS4）之间进行来回转换时的问题（gh-15363）
        arr = np.zeros(shape=(), dtype="U1")
        for i in range(1, sys.maxunicode + 1):
            expected = chr(i)
            arr[()] = expected
            # 确保在设置和获取数组项以及标量数组之间进行来回转换时保持一致
            assert arr[()] == expected
            assert arr.item() == expected

    def test_assign_scalar(self):
        # 测试将标量值分配给数组的所有元素（gh-3258）
        l = np.array(['aa', 'bb'])
        l[:] = np.str_('cc')
        # 确保数组的所有元素被正确赋值为'cc'
        assert_equal(l, ['cc', 'cc'])

    def test_fill_scalar(self):
        # 测试使用标量值填充数组的所有元素（gh-7227）
        l = np.array(['aa', 'bb'])
        l.fill(np.str_('cc'))
        # 确保数组的所有元素被正确填充为'cc'
        assert_equal(l, ['cc', 'cc'])


class TestUnicodeArrayNonzero:

    def test_empty_ustring_array_is_falsey(self):
        # 确保空的Unicode字符串数组在逻辑上为假
        assert_(not np.array([''], dtype=np.str_))

    def test_whitespace_ustring_array_is_truthy(self):
        # 创建包含一个元素为'eggs'的Unicode字符串数组，并将其第一个元素设为空白字符和空字符
        a = np.array(['eggs'], dtype=np.str_)
        a[0] = '  \0\0'
        # 确保Unicode字符串数组在逻辑上为真
        assert_(a)

    def test_all_null_ustring_array_is_falsey(self):
        # 创建包含一个元素为'eggs'的Unicode字符串数组，并将其第一个元素设为全空字符
        a = np.array(['eggs'], dtype=np.str_)
        a[0] = '\0\0\0\0'
        # 确保Unicode字符串数组在逻辑上为假
        assert_(not a)

    def test_null_inside_ustring_array_is_truthy(self):
        # 创建包含一个元素为'eggs'的Unicode字符串数组，并将其第一个元素设为包含空字符的字符串
        a = np.array(['eggs'], dtype=np.str_)
        a[0] = ' \0 \0'
        # 确保Unicode字符串数组在逻辑上为真
        assert_(a)


class TestFormat:

    def test_0d(self):
        # 测试0维数组的格式化输出
        a = np.array(np.pi)
        assert_equal('{:0.3g}'.format(a), '3.14')
        assert_equal('{:0.3g}'.format(a[()]), '3.14')

    def test_1d_no_format(self):
        # 测试1维数组的默认格式化输出
        a = np.array([np.pi])
        assert_equal('{}'.format(a), str(a))

    def test_1d_format(self):
        # 测试1维数组的格式化输出（直到gh-5543，确保行为与旧版本一致）
        a = np.array([np.pi])
        assert_raises(TypeError, '{:30}'.format, a)


from numpy.testing import IS_PYPY


class TestCTypes:

    def test_ctypes_is_available(self):
        # 测试numpy数组的ctypes属性是否可用（gh-3258）
        test_arr = np.array([[1, 2, 3], [4, 5, 6]])

        assert_equal(ctypes, test_arr.ctypes._ctypes)
        assert_equal(tuple(test_arr.ctypes.shape), (2, 3))
    def test_ctypes_is_not_available(self):
        # 从 numpy 内部引入 _internal 模块
        from numpy._core import _internal
        # 将 _internal 模块中的 ctypes 设置为 None，模拟 ctypes 不可用的情况
        _internal.ctypes = None
        try:
            # 创建一个测试数组 test_arr
            test_arr = np.array([[1, 2, 3], [4, 5, 6]])

            # 断言：test_arr 的 ctypes._ctypes 属性是 _internal._missing_ctypes 类型
            assert_(isinstance(test_arr.ctypes._ctypes,
                               _internal._missing_ctypes))
            # 断言：test_arr 的 ctypes.shape 属性为 (2, 3)
            assert_equal(tuple(test_arr.ctypes.shape), (2, 3))
        finally:
            # 恢复 _internal 模块的 ctypes 属性
            _internal.ctypes = ctypes

    def _make_readonly(x):
        # 将数组 x 的可写标志设为 False
        x.flags.writeable = False
        return x

    @pytest.mark.parametrize('arr', [
        np.array([1, 2, 3]),
        np.array([['one', 'two'], ['three', 'four']]),
        np.array((1, 2), dtype='i4,i4'),
        np.zeros((2,), dtype=
            np.dtype(dict(
                formats=['<i4', '<i4'],
                names=['a', 'b'],
                offsets=[0, 2],
                itemsize=6
            ))
        ),
        np.array([None], dtype=object),
        np.array([]),
        np.empty((0, 0)),
        _make_readonly(np.array([1, 2, 3])),
    ], ids=[
        '1d',
        '2d',
        'structured',
        'overlapping',
        'object',
        'empty',
        'empty-2d',
        'readonly'
    ])
    def test_ctypes_data_as_holds_reference(self, arr):
        # gh-9647
        # 创建 arr 的副本以确保 pytest 不会影响引用计数
        arr = arr.copy()

        # 使用 weakref 创建 arr 的弱引用
        arr_ref = weakref.ref(arr)

        # 获取 arr 的 ctypes 数据作为 ctypes.c_void_p 类型的指针
        ctypes_ptr = arr.ctypes.data_as(ctypes.c_void_p)

        # 断言：ctypes_ptr 应当持有 arr 的引用
        del arr
        break_cycles()
        assert_(arr_ref() is not None, "ctypes pointer did not hold onto a reference")

        # 当 ctypes_ptr 对象销毁时，arr 也应该被释放
        del ctypes_ptr
        if IS_PYPY:
            # Pypy 不会立即回收 arr 对象，手动触发垃圾回收以释放 arr
            break_cycles()
        assert_(arr_ref() is None, "unknowable whether ctypes pointer holds a reference")

    def test_ctypes_as_parameter_holds_reference(self):
        # 创建 np.array([None]) 的副本并赋给 arr
        arr = np.array([None]).copy()

        # 使用 weakref 创建 arr 的弱引用
        arr_ref = weakref.ref(arr)

        # 获取 arr 的 _as_parameter_ 属性作为 ctypes 指针
        ctypes_ptr = arr.ctypes._as_parameter_

        # 断言：ctypes_ptr 应当持有 arr 的引用
        del arr
        break_cycles()
        assert_(arr_ref() is not None, "ctypes pointer did not hold onto a reference")

        # 当 ctypes_ptr 对象销毁时，arr 也应该被释放
        del ctypes_ptr
        if IS_PYPY:
            break_cycles()
        assert_(arr_ref() is None, "unknowable whether ctypes pointer holds a reference")
class TestWritebackIfCopy:
    # 使用 WRITEBACKIFCOPY 机制的测试用例集合

    def test_argmax_with_out(self):
        # 创建一个 5x5 的单位矩阵
        mat = np.eye(5)
        # 创建一个形状为 (5,) 的空数组，数据类型为 'i2'
        out = np.empty(5, dtype='i2')
        # 沿着第一个轴（列）计算每列中最大元素的索引，将结果写入 out 数组
        res = np.argmax(mat, 0, out=out)
        # 断言 res 等于从 0 到 4 的范围数组
        assert_equal(res, range(5))

    def test_argmin_with_out(self):
        # 创建一个 5x5 的负单位矩阵
        mat = -np.eye(5)
        # 创建一个形状为 (5,) 的空数组，数据类型为 'i2'
        out = np.empty(5, dtype='i2')
        # 沿着第一个轴（列）计算每列中最小元素的索引，将结果写入 out 数组
        res = np.argmin(mat, 0, out=out)
        # 断言 res 等于从 0 到 4 的范围数组
        assert_equal(res, range(5))

    def test_insert_noncontiguous(self):
        # 创建一个 2x3 的数组并转置，强制其为非 C 连续存储
        a = np.arange(6).reshape(2, 3).T
        # 使用值替换，将满足条件 a>2 的元素替换为 [44, 55]
        np.place(a, a > 2, [44, 55])
        # 断言数组 a 等于指定的数组
        assert_equal(a, np.array([[0, 44], [1, 55], [2, 44]]))
        # 断言触发 ValueError 异常，因为条件 a>20 永远不成立
        assert_raises(ValueError, np.place, a, a > 20, [])

    def test_put_noncontiguous(self):
        # 创建一个 2x3 的数组并转置，强制其为非 C 连续存储
        a = np.arange(6).reshape(2, 3).T
        # 使用指定的索引和值在数组中放置元素
        np.put(a, [0, 2], [44, 55])
        # 断言数组 a 等于指定的数组
        assert_equal(a, np.array([[44, 3], [55, 4], [2, 5]]))

    def test_putmask_noncontiguous(self):
        # 创建一个 2x3 的数组并转置，强制其为非 C 连续存储
        a = np.arange(6).reshape(2, 3).T
        # 使用条件 a>2 和 a**2 来放置值
        np.putmask(a, a > 2, a ** 2)
        # 断言数组 a 等于指定的数组
        assert_equal(a, np.array([[0, 9], [1, 16], [2, 25]]))

    def test_take_mode_raise(self):
        # 创建一个包含 0 到 5 的整数数组
        a = np.arange(6, dtype='int')
        # 创建一个形状为 (2,) 的空数组，数据类型为 'int'
        out = np.empty(2, dtype='int')
        # 从数组 a 中获取指定索引的值，将结果写入 out 数组，使用 'raise' 模式
        np.take(a, [0, 2], out=out, mode='raise')
        # 断言 out 数组等于指定的数组
        assert_equal(out, np.array([0, 2]))

    def test_choose_mod_raise(self):
        # 创建一个 3x3 的数组
        a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        # 创建一个形状为 (3,3) 的空数组，数据类型为 'int'
        out = np.empty((3, 3), dtype='int')
        # 使用 a 数组作为索引从 choices 中选择值，将结果写入 out 数组，使用 'raise' 模式
        choices = [-10, 10]
        np.choose(a, choices, out=out, mode='raise')
        # 断言 out 数组等于指定的数组
        assert_equal(out, np.array([[10, -10, 10], [-10, 10, -10], [10, -10, 10]]))

    def test_flatiter__array__(self):
        # 创建一个 3x3 的数组
        a = np.arange(9).reshape(3, 3)
        # 获取数组 a 的转置后的迭代器对象，并将其转换为数组 c
        b = a.T.flat
        c = b.__array__()
        # 删除数组 c，触发 WRITEBACKIFCOPY 解析，假定引用计数语义
        del c

    def test_dot_out(self):
        # 如果有 CBLAS，将使用 WRITEBACKIFCOPY
        # 创建一个形状为 (3,3) 的浮点型数组
        a = np.arange(9, dtype=float).reshape(3, 3)
        # 计算数组 a 与自身的矩阵乘法，将结果写入数组 a
        b = np.dot(a, a, out=a)
        # 断言 b 等于指定的数组
        assert_equal(b, np.array([[15, 18, 21], [42, 54, 66], [69, 90, 111]]))

    def test_view_assign(self):
        # 导入 numpy 内部的函数
        from numpy._core._multiarray_tests import (
            npy_create_writebackifcopy, npy_resolve
        )

        # 创建一个 3x3 的数组并转置
        arr = np.arange(9).reshape(3, 3).T
        # 创建一个写回拷贝的数组
        arr_wb = npy_create_writebackifcopy(arr)
        # 断言 arr_wb 的标志为写回拷贝
        assert_(arr_wb.flags.writebackifcopy)
        # 断言 arr_wb 的基础数组是 arr
        assert_(arr_wb.base is arr)
        # 将 arr_wb 的所有元素赋值为 -100
        arr_wb[...] = -100
        # 解析 arr_wb，数组 arr 改变，尽管我们赋值给 arr_wb
        npy_resolve(arr_wb)
        # 断言 arr 等于 -100
        assert_equal(arr, -100)
        # 解析后，两个数组不再引用同一个对象
        assert_(arr_wb.ctypes.data != 0)
        assert_equal(arr_wb.base, None)
        # 对 arr_wb 赋值不会传递给 arr
        arr_wb[...] = 100
        assert_equal(arr, -100)
    # 使用 pytest.mark.leaks_references 装饰器标记测试函数，指定了内存泄漏相关的理由
    @pytest.mark.leaks_references(
            reason="increments self in dealloc; ignore since deprecated path.")
    # 测试对象的 dealloc 警告
    def test_dealloc_warning(self):
        # 使用 suppress_warnings 上下文管理器来捕获警告信息
        with suppress_warnings() as sup:
            # 记录 RuntimeWarning 类型的警告
            sup.record(RuntimeWarning)
            # 创建一个 3x3 的数组并转置
            arr = np.arange(9).reshape(3, 3)
            v = arr.T
            # 调用 _multiarray_tests.npy_abuse_writebackifcopy 函数
            _multiarray_tests.npy_abuse_writebackifcopy(v)
            # 确保只记录到一个警告
            assert len(sup.log) == 1

    # 测试视图丢弃引用计数
    def test_view_discard_refcount(self):
        # 导入需要的函数
        from numpy._core._multiarray_tests import (
            npy_create_writebackifcopy, npy_discard
        )

        # 创建一个 3x3 的数组并转置
        arr = np.arange(9).reshape(3, 3).T
        # 备份原始数组
        orig = arr.copy()
        # 如果支持引用计数，则记录数组的引用计数
        if HAS_REFCOUNT:
            arr_cnt = sys.getrefcount(arr)
        # 创建 writebackifcopy 视图
        arr_wb = npy_create_writebackifcopy(arr)
        # 确保 arr_wb 具有 writebackifcopy 标志
        assert_(arr_wb.flags.writebackifcopy)
        # 确保 arr_wb 的基础数组是 arr
        assert_(arr_wb.base is arr)
        # 修改 arr_wb 的内容
        arr_wb[...] = -100
        # 丢弃 writebackifcopy 视图
        npy_discard(arr_wb)
        # 断言丢弃后，arr 保持不变
        assert_equal(arr, orig)
        # 断言丢弃后，两个数组不再相互引用
        assert_(arr_wb.ctypes.data != 0)
        assert_equal(arr_wb.base, None)
        # 如果支持引用计数，则检查 arr 的引用计数是否恢复
        if HAS_REFCOUNT:
            assert_equal(arr_cnt, sys.getrefcount(arr))
        # 修改 arr_wb 不会传递到 arr
        arr_wb[...] = 100
        assert_equal(arr, orig)
class TestArange:
    # 测试当步长为正无穷时，抛出值错误异常
    def test_infinite(self):
        assert_raises_regex(
            ValueError, "size exceeded",
            np.arange, 0, np.inf
        )

    # 测试步长为 NaN 时，抛出值错误异常
    def test_nan_step(self):
        assert_raises_regex(
            ValueError, "cannot compute length",
            np.arange, 0, 1, np.nan
        )

    # 测试步长为零时，抛出零除错误异常
    def test_zero_step(self):
        assert_raises(ZeroDivisionError, np.arange, 0, 10, 0)
        assert_raises(ZeroDivisionError, np.arange, 0.0, 10.0, 0.0)

        # 空范围情况下，也应该抛出零除错误异常
        assert_raises(ZeroDivisionError, np.arange, 0, 0, 0)
        assert_raises(ZeroDivisionError, np.arange, 0.0, 0.0, 0.0)

    # 测试需要给定起始和结束参数的情况下，抛出类型错误异常
    def test_require_range(self):
        assert_raises(TypeError, np.arange)
        assert_raises(TypeError, np.arange, step=3)
        assert_raises(TypeError, np.arange, dtype='int64')
        assert_raises(TypeError, np.arange, start=4)

    # 测试使用关键字参数（start, stop）创建 arange 数组
    def test_start_stop_kwarg(self):
        keyword_stop = np.arange(stop=3)
        keyword_zerotostop = np.arange(0, stop=3)
        keyword_start_stop = np.arange(start=3, stop=9)

        assert len(keyword_stop) == 3
        assert len(keyword_zerotostop) == 3
        assert len(keyword_start_stop) == 6
        assert_array_equal(keyword_stop, keyword_zerotostop)

    # 测试布尔类型数组的 arange 行为
    def test_arange_booleans(self):
        # 对于布尔类型，arange 的行为在长度为2时比较合理，但是在其它情况下可能会有不同
        res = np.arange(False, dtype=bool)
        assert_array_equal(res, np.array([], dtype="bool"))

        res = np.arange(True, dtype="bool")
        assert_array_equal(res, [False])

        res = np.arange(2, dtype="bool")
        assert_array_equal(res, [False, True])

        # 特殊情况，长度为2时的奇怪行为
        res = np.arange(6, 8, dtype="bool")
        assert_array_equal(res, [True, True])

        # 对于布尔类型，不支持长度为3的情况应该抛出类型错误异常
        with pytest.raises(TypeError):
            np.arange(3, dtype="bool")

    # 测试不支持的数据类型，应该抛出类型错误异常
    @pytest.mark.parametrize("dtype", ["S3", "U", "5i"])
    def test_rejects_bad_dtypes(self, dtype):
        dtype = np.dtype(dtype)
        DType_name = re.escape(str(type(dtype)))
        with pytest.raises(TypeError,
                match=rf"arange\(\) not supported for inputs .* {DType_name}"):
            np.arange(2, dtype=dtype)

    # 测试字符串类型参数，应该抛出类型错误异常
    def test_rejects_strings(self):
        # 显式测试字符串参数会调用 "b" - "a" 的错误情况
        DType_name = re.escape(str(type(np.array("a").dtype)))
        with pytest.raises(TypeError,
                match=rf"arange\(\) not supported for inputs .* {DType_name}"):
            np.arange("a", "b")

    # 测试大端和小端序列的 arange 行为
    def test_byteswapped(self):
        res_be = np.arange(1, 1000, dtype=">i4")
        res_le = np.arange(1, 1000, dtype="<i4")
        assert res_be.dtype == ">i4"
        assert res_le.dtype == "<i4"
        assert_array_equal(res_le, res_be)

    # 参数化测试，用于测试不同情况下的 arange 行为
    @pytest.mark.parametrize("which", [0, 1, 2])
    # 定义测试函数，用于测试错误路径和类型提升
    def test_error_paths_and_promotion(self, which):
        # 设置参数列表：起始值、结束值和步长
        args = [0, 1, 2]
        # 将指定索引位置的参数转换为 np.float64 类型，以确保输出类型为 float64
        args[which] = np.float64(2.)

        # 断言 np.arange 函数生成的数组的数据类型为 np.float64
        assert np.arange(*args).dtype == np.float64

        # 处理更复杂的错误路径，目的是为了提高代码覆盖率
        args[which] = [None, []]
        with pytest.raises(ValueError):
            # 期望此处引发 ValueError 异常，因为无法推断起始值的数据类型
            np.arange(*args)
class TestArrayFinalize:
    """ Tests __array_finalize__ """

    def test_receives_base(self):
        # gh-11237
        # 定义一个继承自 np.ndarray 的类 SavesBase，重载 __array_finalize__ 方法
        class SavesBase(np.ndarray):
            def __array_finalize__(self, obj):
                # 在 finalize 方法中保存 self.base 到 saved_base 属性
                self.saved_base = self.base

        # 创建一个 np.array，并转换为 SavesBase 类型
        a = np.array(1).view(SavesBase)
        # 断言 saved_base 和 base 相同
        assert_(a.saved_base is a.base)

    def test_bad_finalize1(self):
        # 定义一个具有错误的 __array_finalize__ 属性的类 BadAttributeArray
        class BadAttributeArray(np.ndarray):
            @property
            def __array_finalize__(self):
                # 当调用 finalize 方法时，引发 RuntimeError
                raise RuntimeError("boohoo!")

        # 断言尝试转换为 BadAttributeArray 会引发 TypeError
        with pytest.raises(TypeError, match="not callable"):
            np.arange(10).view(BadAttributeArray)

    def test_bad_finalize2(self):
        # 定义一个没有 property 装饰器的 __array_finalize__ 方法的类 BadAttributeArray
        class BadAttributeArray(np.ndarray):
            def __array_finalize__(self):
                # 当调用 finalize 方法时，引发 RuntimeError
                raise RuntimeError("boohoo!")

        # 断言尝试转换为 BadAttributeArray 会引发 TypeError
        with pytest.raises(TypeError, match="takes 1 positional"):
            np.arange(10).view(BadAttributeArray)

    def test_bad_finalize3(self):
        # 定义一个在 __array_finalize__ 方法中引发 RuntimeError 的类 BadAttributeArray
        class BadAttributeArray(np.ndarray):
            def __array_finalize__(self, obj):
                # 当调用 finalize 方法时，引发 RuntimeError
                raise RuntimeError("boohoo!")

        # 断言尝试转换为 BadAttributeArray 会引发 RuntimeError
        with pytest.raises(RuntimeError, match="boohoo!"):
            np.arange(10).view(BadAttributeArray)

    def test_lifetime_on_error(self):
        # gh-11237
        # 定义一个在 finalize 方法中引发异常但保持对象存活的类 RaisesInFinalize
        class RaisesInFinalize(np.ndarray):
            def __array_finalize__(self, obj):
                # 引发异常，但保持对象存活
                raise Exception(self)

        # 定义一个普通对象 Dummy，无法使用 weakref
        class Dummy: pass

        # 创建一个包含 Dummy 对象的数组，并获取其内部对象的弱引用
        obj_arr = np.array(Dummy())
        obj_ref = weakref.ref(obj_arr[()])

        # 尝试创建一个 RaisesInFinalize 类型的子数组，应该引发异常
        with assert_raises(Exception) as e:
            obj_arr.view(RaisesInFinalize)

        # 获取异常对象作为子数组
        obj_subarray = e.exception.args[0]
        del e
        # 断言 obj_subarray 是 RaisesInFinalize 的实例
        assert_(isinstance(obj_subarray, RaisesInFinalize))

        # 破坏循环引用
        break_cycles()
        # 断言 obj_ref 还存在引用，对象尚未死亡
        assert_(obj_ref() is not None, "object should not already be dead")

        del obj_arr
        break_cycles()
        # 断言 obj_ref 还存在引用，obj_arr 未持有最后一个引用
        assert_(obj_ref() is not None, "obj_arr should not hold the last reference")

        del obj_subarray
        break_cycles()
        # 断言 obj_ref 没有任何引用，没有引用保持
        assert_(obj_ref() is None, "no references should remain")

    def test_can_use_super(self):
        # 定义一个使用 super().__array_finalize__ 的类 SuperFinalize
        class SuperFinalize(np.ndarray):
            def __array_finalize__(self, obj):
                # 使用 super() 调用父类的 finalize 方法，并保存结果到 saved_result
                self.saved_result = super().__array_finalize__(obj)

        # 创建一个 np.array，并转换为 SuperFinalize 类型
        a = np.array(1).view(SuperFinalize)
        # 断言 saved_result 为 None
        assert_(a.saved_result is None)


def test_orderconverter_with_nonASCII_unicode_ordering():
    # gh-7475
    # 创建一个包含 0 到 4 的 np.array
    a = np.arange(5)
    # 断言尝试使用非 ASCII Unicode 排序引发 ValueError
    assert_raises(ValueError, a.flatten, order='\xe2')


def test_equal_override():
    # gh-9153: ndarray.__eq__ 使用特殊逻辑处理结构化数组，不尊重 __array_priority__ 或 __array_ufunc__ 的覆盖。
    # 这个 PR 修复了 __array_priority__ 和 __array_ufunc__ = None 的情况。
    # 定义一个总是返回"eq"的自定义类，覆盖了"=="运算符
    class MyAlwaysEqual:
        # 重载"=="运算符
        def __eq__(self, other):
            return "eq"

        # 重载"!="运算符
        def __ne__(self, other):
            return "ne"

    # 继承自MyAlwaysEqual类，并设置特殊属性__array_priority__
    class MyAlwaysEqualOld(MyAlwaysEqual):
        __array_priority__ = 10000

    # 继承自MyAlwaysEqual类，并设置特殊属性__array_ufunc__
    class MyAlwaysEqualNew(MyAlwaysEqual):
        __array_ufunc__ = None

    # 创建一个NumPy数组，包含两个元组，每个元组内有两个整数，类型为'i4,i4'
    array = np.array([(0, 1), (2, 3)], dtype='i4,i4')

    # 遍历MyAlwaysEqualOld和MyAlwaysEqualNew两个类
    for my_always_equal_cls in MyAlwaysEqualOld, MyAlwaysEqualNew:
        # 实例化当前类的对象
        my_always_equal = my_always_equal_cls()

        # 断言：自定义对象与数组使用"=="运算符比较，结果应为"eq"
        assert_equal(my_always_equal == array, 'eq')
        
        # 断言：数组与自定义对象使用"=="运算符比较，结果应为"eq"
        assert_equal(array == my_always_equal, 'eq')
        
        # 断言：自定义对象与数组使用"!="运算符比较，结果应为"ne"
        assert_equal(my_always_equal != array, 'ne')
        
        # 断言：数组与自定义对象使用"!="运算符比较，结果应为"ne"
        assert_equal(array != my_always_equal, 'ne')
# 使用 pytest 提供参数化测试，测试以下运算符的行为：operator.eq, operator.ne
@pytest.mark.parametrize("op", [operator.eq, operator.ne])
# 参数化测试数据，包括不同的数据类型组合，每个组合使用一个元组表示
@pytest.mark.parametrize(["dt1", "dt2"], [
        ([("f", "i")], [("f", "i")]),  # 结构化比较（成功）
        ("M8", "d"),  # 不可能的比较：结果全部为 True 或 False
        ("d", "d"),  # 有效的比较
        ])
def test_equal_subclass_no_override(op, dt1, dt2):
    # 测试不同可能路径如何处理子类

    class MyArr(np.ndarray):
        called_wrap = 0

        def __array_wrap__(self, new, context=None, return_scalar=False):
            type(self).called_wrap += 1
            return super().__array_wrap__(new)

    # 创建 numpy 数组和自定义子类数组
    numpy_arr = np.zeros(5, dtype=dt1)
    my_arr = np.zeros(5, dtype=dt2).view(MyArr)

    # 断言返回的数组类型为 MyArr
    assert type(op(numpy_arr, my_arr)) is MyArr
    assert type(op(my_arr, numpy_arr)) is MyArr
    # 期望 __array_wrap__ 被调用 2 次（如果有更多字段则更多调用）
    assert MyArr.called_wrap == 2


@pytest.mark.parametrize(["dt1", "dt2"], [
        ("M8[ns]", "d"),
        ("M8[s]", "l"),
        ("m8[ns]", "d"),
        # 缺失的测试：("m8[ns]", "l")，因为 timedelta 当前会提升整数
        ("M8[s]", "m8[s]"),
        ("S5", "U5"),
        # 结构化/空类型数据有专门的路径，在这里不测试
])
def test_no_loop_gives_all_true_or_false(dt1, dt2):
    # 确保广播后的测试结果形状，使用随机值，实际值应忽略
    arr1 = np.random.randint(5, size=100).astype(dt1)
    arr2 = np.random.randint(5, size=99)[:, np.newaxis].astype(dt2)

    # 测试相等操作
    res = arr1 == arr2
    assert res.shape == (99, 100)
    assert res.dtype == bool
    assert not res.any()  # 所有元素应为 False

    # 测试不等操作
    res = arr1 != arr2
    assert res.shape == (99, 100)
    assert res.dtype == bool
    assert res.all()  # 所有元素应为 True

    # 不兼容的形状会触发 ValueError
    arr2 = np.random.randint(5, size=99).astype(dt2)
    with pytest.raises(ValueError):
        arr1 == arr2

    with pytest.raises(ValueError):
        arr1 != arr2

    # 另一个操作的基本测试:
    with pytest.raises(np._core._exceptions._UFuncNoLoopError):
        arr1 > arr2


@pytest.mark.parametrize("op", [
        operator.eq, operator.ne, operator.le, operator.lt, operator.ge,
        operator.gt])
def test_comparisons_forwards_error(op):
    # 测试当其中一个操作数不是数组时是否会引发 TypeError 异常

    class NotArray:
        def __array__(self, dtype=None, copy=None):
            raise TypeError("run you fools")

    # 检查是否引发预期的 TypeError 异常
    with pytest.raises(TypeError, match="run you fools"):
        op(np.arange(2), NotArray())

    with pytest.raises(TypeError, match="run you fools"):
        op(NotArray(), np.arange(2))


def test_richcompare_scalar_boolean_singleton_return():
    # 测试返回标量布尔值的情况

    # 当前保证返回布尔单例，但返回 NumPy 布尔也是可以的
    assert (np.array(0) == "a") is False
    assert (np.array(0) != "a") is True
    assert (np.int16(0) == "a") is False
    assert (np.int16(0) != "a") is True
@pytest.mark.parametrize("op", [
        operator.eq, operator.ne, operator.le, operator.lt, operator.ge,
        operator.gt])
def test_ragged_comparison_fails(op):
    # 定义一个测试函数，用于测试各种比较运算符在特定条件下的行为是否正确
    # 创建两个包含对象数组的 NumPy 数组，dtype 设置为 object
    a = np.array([1, np.array([1, 2, 3])], dtype=object)
    b = np.array([1, np.array([1, 2, 3])], dtype=object)

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，且异常消息匹配特定模式
    with pytest.raises(ValueError, match="The truth value.*ambiguous"):
        op(a, b)


@pytest.mark.parametrize(
    ["fun", "npfun"],
    [
        (_multiarray_tests.npy_cabs, np.absolute),
        (_multiarray_tests.npy_carg, np.angle)
    ]
)
@pytest.mark.parametrize("x", [1, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("y", [1, np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("test_dtype", np.complexfloating.__subclasses__())
def test_npymath_complex(fun, npfun, x, y, test_dtype):
    # 对 NumPy 中的复数运算函数进行基本测试
    z = test_dtype(complex(x, y))
    with np.errstate(invalid='ignore'):
        # 在忽略无效值错误状态下，测试特定的 npymath 函数
        # 一些实现可能会对 +-inf 发出警告（参见 gh-24876）：
        #     RuntimeWarning: invalid value encountered in absolute
        got = fun(z)
        expected = npfun(z)
        # 使用 assert_allclose 检查得到的结果与预期结果的接近程度
        assert_allclose(got, expected)


def test_npymath_real():
    # 对 NumPy 中的实数运算函数进行基本测试
    from numpy._core._multiarray_tests import (
        npy_log10, npy_cosh, npy_sinh, npy_tan, npy_tanh)

    # 函数字典，将 npymath 函数映射到 NumPy 的对应函数
    funcs = {npy_log10: np.log10,
             npy_cosh: np.cosh,
             npy_sinh: np.sinh,
             npy_tan: np.tan,
             npy_tanh: np.tanh}
    # 测试值的集合
    vals = (1, np.inf, -np.inf, np.nan)
    # 测试类型的集合
    types = (np.float32, np.float64, np.longdouble)

    with np.errstate(all='ignore'):
        for fun, npfun in funcs.items():
            for x, t in itertools.product(vals, types):
                z = t(x)
                # 调用 npymath 函数
                got = fun(z)
                expected = npfun(z)
                # 使用 assert_allclose 检查得到的结果与预期结果的接近程度
                assert_allclose(got, expected)


def test_uintalignment_and_alignment():
    # 检查 uintalignment_and_alignment 函数的测试要求：
    #  1. numpy 结构与 C 结构布局匹配
    #  2. 对齐访问在 ufuncs/casting 中是安全的
    #  3. 复制代码在 "uint aligned" 访问中是安全的
    #
    # 复杂类型是主要问题，它们的对齐可能与它们的 "uint alignment" 不同。
    #
    # 这个测试可能仅在某些平台上失败，其中 uint64 对齐与 complex64 对齐不相等。
    # 第二个和第三个测试只有在 DEBUG=1 时才会失败。

    # 创建具有指定对齐的 dtype 对象
    d1 = np.dtype('u1,c8', align=True)
    d2 = np.dtype('u4,c8', align=True)
    d3 = np.dtype({'names': ['a', 'b'], 'formats': ['u1', d1]}, align=True)

    # 检查指定 dtype 对象的对齐属性
    assert_equal(np.zeros(1, dtype=d1)['f1'].flags['ALIGNED'], True)
    assert_equal(np.zeros(1, dtype=d2)['f1'].flags['ALIGNED'], True)
    assert_equal(np.zeros(1, dtype='u1,c8')['f1'].flags['ALIGNED'], False)

    # 检查 C 结构是否与 numpy 结构的大小匹配
    s = _multiarray_tests.get_struct_alignments()
    # 使用 zip 函数同时迭代 d1, d2, d3 和 s 中的元组 (alignment, size)
    for d, (alignment, size) in zip([d1,d2,d3], s):
        # 断言当前 d 的对齐方式与预期的 alignment 一致
        assert_equal(d.alignment, alignment)
        # 断言当前 d 的元素大小与预期的 size 一致
        assert_equal(d.itemsize, size)

    # 检查在调试模式下，ufuncs（通用函数）不会报错
    # （如果上面的 aligned 标志为 true，这应该是没问题的）
    src = np.zeros((2,2), dtype=d1)['f1']  # 通常是4字节对齐的
    np.exp(src)  # 断言失败？

    # 检查在调试模式下，复制代码不会报错
    dst = np.zeros((2,2), dtype='c8')
    dst[:,1] = src[:,1]  # 低级 strided 循环中的 assert 失败？
# 定义一个测试类 TestAlignment，用于测试内存对齐对 numpy 的影响
class TestAlignment:
    # 从 scipy._lib.tests.test__util.test__aligned_zeros 适配而来
    # 检查不寻常的内存对齐是否会影响 numpy

    # 定义一个检查函数 check，用于验证不同参数下的内存对齐情况
    def check(self, shape, dtype, order, align):
        # 生成错误信息字符串，用于显示参数
        err_msg = repr((shape, dtype, order, align))
        
        # 调用 _aligned_zeros 函数创建一个数组 x，传入指定的形状、数据类型、存储顺序和对齐要求
        x = _aligned_zeros(shape, dtype, order, align=align)
        
        # 如果 align 为 None，则使用数据类型的默认对齐值
        if align is None:
            align = np.dtype(dtype).alignment
        
        # 断言 x 的数据起始地址是否能被 align 整除
        assert_equal(x.__array_interface__['data'][0] % align, 0)
        
        # 根据 shape 的类型（单个元素或者元组），断言 x 的形状与预期相符
        if hasattr(shape, '__len__'):
            assert_equal(x.shape, shape, err_msg)
        else:
            assert_equal(x.shape, (shape,), err_msg)
        
        # 断言 x 的数据类型与预期的 dtype 相符
        assert_equal(x.dtype, dtype)
        
        # 如果 order 为 "C"，断言 x 是 C 顺序连续存储
        if order == "C":
            assert_(x.flags.c_contiguous, err_msg)
        # 如果 order 为 "F"，断言 x 是 Fortran 顺序连续存储（如果数组大小大于 0）
        elif order == "F":
            if x.size > 0:
                assert_(x.flags.f_contiguous, err_msg)
        # 如果 order 为 None，断言 x 是 C 顺序连续存储
        elif order is None:
            assert_(x.flags.c_contiguous, err_msg)
        else:
            # 其他情况抛出 ValueError 异常
            raise ValueError()

    # 定义一个测试函数 test_various_alignments，测试不同的内存对齐方式
    def test_various_alignments(self):
        # 遍历不同的对齐值 align
        for align in [1, 2, 3, 4, 8, 12, 16, 32, 64, None]:
            # 遍历不同的 n 值
            for n in [0, 1, 3, 11]:
                # 遍历不同的存储顺序 order
                for order in ["C", "F", None]:
                    # 遍历不同的数据类型 dtype
                    for dtype in list(np.typecodes["All"]) + ['i4,i4,i4']:
                        if dtype == 'O':
                            # 对象类型的 dtype 无法进行错误对齐测试，跳过
                            continue
                        # 遍历不同的形状 shape
                        for shape in [n, (1, 2, 3, n)]:
                            # 调用 self.check 方法，测试当前参数组合的内存对齐情况
                            self.check(shape, np.dtype(dtype), order, align)

    # 定义一个测试函数 test_strided_loop_alignments，特别测试 complex64 和 float128 的正确对齐情况
    def test_strided_loop_alignments(self):
        # 遍历不同的对齐值 align
        for align in [1, 2, 4, 8, 12, 16, None]:
            # 创建一个 float64 类型的数组 xf64，使用 _aligned_zeros 函数
            xf64 = _aligned_zeros(3, np.float64)
            
            # 创建一个 complex64 类型的数组 xc64，使用指定的对齐值 align
            xc64 = _aligned_zeros(3, np.complex64, align=align)
            
            # 创建一个 float128 类型的数组 xf128，使用指定的对齐值 align
            xf128 = _aligned_zeros(3, np.longdouble, align=align)

            # 测试类型转换，从和到错位的复杂值
            with suppress_warnings() as sup:
                sup.filter(ComplexWarning, "Casting complex values")
                # 将 complex64 类型数组 xc64 转换为 float64 类型
                xc64.astype('f8')
            # 将 float64 类型数组 xf64 转换为 complex64 类型
            xf64.astype(np.complex64)
            # 测试加法运算，将 xc64 和 xf64 相加
            test = xc64 + xf64

            # 将 float128 类型数组 xf128 转换为 float64 类型
            xf128.astype('f8')
            # 将 float64 类型数组 xf64 转换为 float128 类型
            xf64.astype(np.longdouble)
            # 测试加法运算，将 xf128 和 xf64 相加
            test = xf128 + xf64

            # 测试加法运算，将 xf128 和 xc64 相加
            test = xf128 + xc64

            # 测试复制操作，对连续存储和错位存储的数组进行复制
            # 连续复制
            xf64[:] = xf64.copy()
            xc64[:] = xc64.copy()
            xf128[:] = xf128.copy()
            # 错位复制
            xf64[::2] = xf64[::2].copy()
            xc64[::2] = xc64[::2].copy()
            xf128[::2] = xf128[::2].copy()

def test_getfield():
    # 创建一个 uint16 类型的数组 a，元素从 0 到 31
    a = np.arange(32, dtype='uint16')
    
    # 如果系统的字节顺序是 little-endian
    if sys.byteorder == 'little':
        i = 0
        j = 1
    else:
        i = 1
        j = 0
    
    # 从数组 a 中获取 int8 类型的数据，使用 i 作为偏移量
    b = a.getfield('int8', i)
    
    # 断言 b 与数组 a 相等
    assert_equal(b, a)
    
    # 从数组 a 中获取 int8 类型的数据，使用 j 作为偏移量
    b = a.getfield('int8', j)
    # 断言检查变量 b 的值是否等于 0
    assert_equal(b, 0)
    # 使用 pytest 检查调用 a.getfield('uint8', -1) 是否会引发 ValueError 异常
    pytest.raises(ValueError, a.getfield, 'uint8', -1)
    # 使用 pytest 检查调用 a.getfield('uint8', 16) 是否会引发 ValueError 异常
    pytest.raises(ValueError, a.getfield, 'uint8', 16)
    # 使用 pytest 检查调用 a.getfield('uint64', 0) 是否会引发 ValueError 异常
    pytest.raises(ValueError, a.getfield, 'uint64', 0)
class TestViewDtype:
    """
    Verify that making a view of a non-contiguous array works as expected.
    """

    def test_smaller_dtype_multiple(self):
        # x is non-contiguous
        x = np.arange(10, dtype='<i4')[::2]
        # Assert that a ValueError is raised with specified error message
        with pytest.raises(ValueError,
                           match='the last axis must be contiguous'):
            # Attempt to view x as '<i2' dtype
            x.view('<i2')
        expected = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]]
        # Assert that x[:, np.newaxis] viewed as '<i2' matches expected
        assert_array_equal(x[:, np.newaxis].view('<i2'), expected)

    def test_smaller_dtype_not_multiple(self):
        # x is non-contiguous
        x = np.arange(5, dtype='<i4')[::2]

        with pytest.raises(ValueError,
                           match='the last axis must be contiguous'):
            # Attempt to view x as 'S3' dtype, expect ValueError
            x.view('S3')
        with pytest.raises(ValueError,
                           match='When changing to a smaller dtype'):
            # Attempt to view x[:, np.newaxis] as 'S3' dtype, expect ValueError
            x[:, np.newaxis].view('S3')

        # Make sure the problem is because of the dtype size
        expected = [[b''], [b'\x02'], [b'\x04']]
        # Assert that x[:, np.newaxis] viewed as 'S4' matches expected
        assert_array_equal(x[:, np.newaxis].view('S4'), expected)

    def test_larger_dtype_multiple(self):
        # x is non-contiguous in the first dimension, contiguous in the last
        x = np.arange(20, dtype='<i2').reshape(10, 2)[::2, :]
        expected = np.array([[65536], [327684], [589832],
                             [851980], [1114128]], dtype='<i4')
        # Assert that x viewed as '<i4' dtype matches expected
        assert_array_equal(x.view('<i4'), expected)

    def test_larger_dtype_not_multiple(self):
        # x is non-contiguous in the first dimension, contiguous in the last
        x = np.arange(20, dtype='<i2').reshape(10, 2)[::2, :]
        with pytest.raises(ValueError,
                           match='When changing to a larger dtype'):
            # Attempt to view x as 'S3' dtype, expect ValueError
            x.view('S3')
        # Make sure the problem is because of the dtype size
        expected = [[b'\x00\x00\x01'], [b'\x04\x00\x05'], [b'\x08\x00\t'],
                    [b'\x0c\x00\r'], [b'\x10\x00\x11']]
        # Assert that x viewed as 'S4' dtype matches expected
        assert_array_equal(x.view('S4'), expected)

    def test_f_contiguous(self):
        # x is F-contiguous
        x = np.arange(4 * 3, dtype='<i4').reshape(4, 3).T
        with pytest.raises(ValueError,
                           match='the last axis must be contiguous'):
            # Attempt to view x as '<i2' dtype, expect ValueError
            x.view('<i2')

    def test_non_c_contiguous(self):
        # x is contiguous in axis=-1, but not C-contiguous in other axes
        x = np.arange(2 * 3 * 4, dtype='i1').\
                    reshape(2, 3, 4).transpose(1, 0, 2)
        expected = [[[256, 770], [3340, 3854]],
                    [[1284, 1798], [4368, 4882]],
                    [[2312, 2826], [5396, 5910]]]
        # Assert that x viewed as '<i2' dtype matches expected
        assert_array_equal(x.view('<i2'), expected)


@pytest.mark.xfail(_SUPPORTS_SVE, reason="gh-22982")
# Test various array sizes that hit different code paths in quicksort-avx512
@pytest.mark.parametrize("N", np.arange(1, 512))
@pytest.mark.parametrize("dtype", ['e', 'f', 'd'])
def test_sort_float(N, dtype):
    # Regular data with nan sprinkled
    np.random.seed(42)
    # 创建一个包含 N 个随机数的数组，数值范围为 (-0.5, 0.5)，并将其转换为指定的数据类型
    arr = -0.5 + np.random.sample(N).astype(dtype)
    # 在数组中随机选择 3 个位置，将其值设为 NaN (Not a Number)
    arr[np.random.choice(arr.shape[0], 3)] = np.nan
    # 使用快速排序算法对数组进行排序，并与使用堆排序算法的结果进行断言比较
    assert_equal(np.sort(arr, kind='quick'), np.sort(arr, kind='heap'))

    # (2) with +INF
    # 创建一个包含 N 个元素的数组，所有元素的值为正无穷大
    infarr = np.inf*np.ones(N, dtype=dtype)
    # 在数组中随机选择 5 个位置，将其值设为 -1.0
    infarr[np.random.choice(infarr.shape[0], 5)] = -1.0
    # 使用快速排序算法对数组进行排序，并与使用堆排序算法的结果进行断言比较
    assert_equal(np.sort(infarr, kind='quick'), np.sort(infarr, kind='heap'))

    # (3) with -INF
    # 创建一个包含 N 个元素的数组，所有元素的值为负无穷大
    neginfarr = -np.inf*np.ones(N, dtype=dtype)
    # 在数组中随机选择 5 个位置，将其值设为 1.0
    neginfarr[np.random.choice(neginfarr.shape[0], 5)] = 1.0
    # 使用快速排序算法对数组进行排序，并与使用堆排序算法的结果进行断言比较
    assert_equal(np.sort(neginfarr, kind='quick'),
                 np.sort(neginfarr, kind='heap'))

    # (4) with +/-INF
    # 创建一个包含 N 个元素的数组，所有元素的值为正无穷大
    infarr = np.inf*np.ones(N, dtype=dtype)
    # 在数组中随机选择 N/2 个位置，将其值设为负无穷大
    infarr[np.random.choice(infarr.shape[0], (int)(N/2))] = -np.inf
    # 使用快速排序算法对数组进行排序，并与使用堆排序算法的结果进行断言比较
    assert_equal(np.sort(infarr, kind='quick'), np.sort(infarr, kind='heap'))
# 测试以 float16 排序的函数
def test_sort_float16():
    # 创建一个包含从 0 到 65535 的整数的数组，数据类型为 int16
    arr = np.arange(65536, dtype=np.int16)
    # 将数组转换为 float16 类型的临时数组
    temp = np.frombuffer(arr.tobytes(), dtype=np.float16)
    # 创建数据的副本
    data = np.copy(temp)
    # 对数据进行随机重排
    np.random.shuffle(data)
    # 创建数据的备份
    data_backup = data
    # 使用快速排序和堆排序分别对数据和其备份进行排序，并断言排序结果相等
    assert_equal(np.sort(data, kind='quick'),
                 np.sort(data_backup, kind='heap'))


# 使用参数化测试测试不同类型的整数排序
@pytest.mark.parametrize("N", np.arange(1, 512))
@pytest.mark.parametrize("dtype", ['h', 'H', 'i', 'I', 'l', 'L'])
def test_sort_int(N, dtype):
    # 使用给定类型的最小值和最大值生成随机数据
    minv = np.iinfo(dtype).min
    maxv = np.iinfo(dtype).max
    arr = np.random.randint(low=minv, high=maxv-1, size=N, dtype=dtype)
    # 将数组中的部分元素替换为最小值和最大值
    arr[np.random.choice(arr.shape[0], 10)] = minv
    arr[np.random.choice(arr.shape[0], 10)] = maxv
    # 使用快速排序对数组进行排序，并断言排序结果与堆排序结果相等
    assert_equal(np.sort(arr, kind='quick'), np.sort(arr, kind='heap'))


# 测试以 uint32 类型排序的函数
def test_sort_uint():
    # 使用默认随机数生成器和指定的大小生成 uint32 类型的随机数据
    rng = np.random.default_rng(42)
    N = 2047
    maxv = np.iinfo(np.uint32).max
    arr = rng.integers(low=0, high=maxv, size=N).astype('uint32')
    # 将数组中的部分元素替换为最大值
    arr[np.random.choice(arr.shape[0], 10)] = maxv
    # 使用快速排序对数组进行排序，并断言排序结果与堆排序结果相等
    assert_equal(np.sort(arr, kind='quick'), np.sort(arr, kind='heap'))


# 测试获取 ndarray C 版本的私有函数
def test_private_get_ndarray_c_version():
    # 断言 _get_ndarray_c_version() 返回的是整数
    assert isinstance(_get_ndarray_c_version(), int)


# 使用参数化测试测试不同类型和大小的浮点数数组排序
@pytest.mark.parametrize("N", np.arange(1, 512))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_argsort_float(N, dtype):
    rnd = np.random.RandomState(116112)
    # (1) 创建包含少量 NaN 的正常数据，不使用向量化排序
    arr = -0.5 + rnd.random(N).astype(dtype)
    arr[rnd.choice(arr.shape[0], 3)] = np.nan
    # 断言 arr 使用快速排序后的参数排序结果与 np.argsort 使用快速排序的结果相等
    assert_arg_sorted(arr, np.argsort(arr, kind='quick'))

    # (2) 创建包含正无穷的随机数据
    arr = -0.5 + rnd.rand(N).astype(dtype)
    arr[N-1] = np.inf
    # 断言 arr 使用快速排序后的参数排序结果与 np.argsort 使用快速排序的结果相等
    assert_arg_sorted(arr, np.argsort(arr, kind='quick'))


# 使用参数化测试测试不同类型和大小的整数数组排序
@pytest.mark.parametrize("N", np.arange(2, 512))
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_argsort_int(N, dtype):
    rnd = np.random.RandomState(1100710816)
    # (1) 创建包含最小值和最大值的随机数据
    minv = np.iinfo(dtype).min
    maxv = np.iinfo(dtype).max
    arr = rnd.randint(low=minv, high=maxv, size=N, dtype=dtype)
    i, j = rnd.choice(N, 2, replace=False)
    arr[i] = minv
    arr[j] = maxv
    # 断言 arr 使用快速排序后的参数排序结果与 np.argsort 使用快速排序的结果相等
    assert_arg_sorted(arr, np.argsort(arr, kind='quick'))

    # (2) 创建包含最大值的随机数据
    arr = rnd.randint(low=minv, high=maxv, size=N, dtype=dtype)
    arr[N-1] = maxv
    # 断言 arr 使用快速排序后的参数排序结果与 np.argsort 使用快速排序的结果相等
    assert_arg_sorted(arr, np.argsort(arr, kind='quick'))


# 如果系统不支持引用计数，跳过该测试
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_gh_22683():
    b = 777.68760986
    # 创建包含 10000 个相同对象的数组
    a = np.array([b] * 10000, dtype=object)
    # 记录开始时的引用计数
    refc_start = sys.getrefcount(b)
    # 使用 np.choose() 函数操作数组，不生成新对象
    np.choose(np.zeros(10000, dtype=int), [a], out=a)
    # 再次使用 np.choose() 函数操作数组，不生成新对象
    np.choose(np.zeros(10000, dtype=int), [a], out=a)
    # 记录结束时的引用计数
    refc_end = sys.getrefcount(b)
    # 断言确保 refc_end 和 refc_start 之间的差值小于 10
    assert refc_end - refc_start < 10
# 定义测试函数 test_gh_24459，用于测试 numpy 的 np.choose 函数在特定条件下是否引发 TypeError 异常
def test_gh_24459():
    # 创建一个形状为 (50, 3)，元素类型为 np.float64 的全零数组 a
    a = np.zeros((50, 3), dtype=np.float64)
    # 使用 pytest 模块确保 np.choose 函数在输入 a 时会抛出 TypeError 异常
    with pytest.raises(TypeError):
        np.choose(a, [3, -1])

# 使用 pytest 的 parametrize 装饰器，为 test_partition_int 函数传递参数 N 和 dtype
@pytest.mark.parametrize("N", np.arange(2, 512))
@pytest.mark.parametrize("dtype", [np.int16, np.uint16,
                        np.int32, np.uint32, np.int64, np.uint64])
def test_partition_int(N, dtype):
    # 创建一个指定类型和形状的随机数生成器
    rnd = np.random.RandomState(1100710816)
    
    # (1) 使用给定 dtype 的随机数据，确保数据中包含其最小和最大值
    minv = np.iinfo(dtype).min
    maxv = np.iinfo(dtype).max
    arr = rnd.randint(low=minv, high=maxv, size=N, dtype=dtype)
    # 从生成的数组中选择两个不同的索引，将这两个位置上的值分别设置为最小值和最大值
    i, j = rnd.choice(N, 2, replace=False)
    arr[i] = minv
    arr[j] = maxv
    # 随机选择一个索引 k
    k = rnd.choice(N, 1)[0]
    # 断言 np.partition 函数的结果与排序后的结果在索引 k 处相同
    assert_arr_partitioned(np.sort(arr)[k], k,
            np.partition(arr, k, kind='introselect'))
    # 断言 np.argpartition 函数的结果与排序后的结果在索引 k 处相同
    assert_arr_partitioned(np.sort(arr)[k], k,
            arr[np.argpartition(arr, k, kind='introselect')])

    # (2) 使用给定 dtype 的随机数据，确保数组中最大值位于数组末尾
    arr = rnd.randint(low=minv, high=maxv, size=N, dtype=dtype)
    arr[N-1] = maxv
    # 断言 np.partition 函数的结果与排序后的结果在索引 k 处相同
    assert_arr_partitioned(np.sort(arr)[k], k,
            np.partition(arr, k, kind='introselect'))
    # 断言 np.argpartition 函数的结果与排序后的结果在索引 k 处相同
    assert_arr_partitioned(np.sort(arr)[k], k,
            arr[np.argpartition(arr, k, kind='introselect')])

# 使用 pytest 的 parametrize 装饰器，为 test_partition_fp 函数传递参数 N 和 dtype
@pytest.mark.parametrize("N", np.arange(2, 512))
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_partition_fp(N, dtype):
    # 创建一个指定类型和形状的随机数生成器
    rnd = np.random.RandomState(1100710816)
    # 生成一个随机数组，元素类型为 dtype
    arr = -0.5 + rnd.random(N).astype(dtype)
    # 随机选择一个索引 k
    k = rnd.choice(N, 1)[0]
    # 断言 np.partition 函数的结果与排序后的结果在索引 k 处相同
    assert_arr_partitioned(np.sort(arr)[k], k,
            np.partition(arr, k, kind='introselect'))
    # 断言 np.argpartition 函数的结果与排序后的结果在索引 k 处相同
    assert_arr_partitioned(np.sort(arr)[k], k,
            arr[np.argpartition(arr, k, kind='introselect')])

# 定义测试函数 test_cannot_assign_data，用于测试 numpy 数组的 data 属性不能被直接赋值的情况
def test_cannot_assign_data():
    # 创建一个长度为 10 的 numpy 数组 a
    a = np.arange(10)
    # 创建一个与 a 相同长度的线性空间数组 b
    b = np.linspace(0, 1, 10)
    # 使用 pytest 模块确保试图给 a.data 赋值会抛出 AttributeError 异常
    with pytest.raises(AttributeError):
        a.data = b.data

# 定义测试函数 test_insufficient_width，测试当向 np.binary_repr 函数传递不足以表示数字的二进制宽度时会引发 ValueError 异常
def test_insufficient_width():
    # 使用 pytest 模块确保向 np.binary_repr 函数传递 width=2 时会抛出 ValueError 异常
    with pytest.raises(ValueError):
        np.binary_repr(10, width=2)
    # 使用 pytest 模块确保向 np.binary_repr 函数传递 width=2 时会抛出 ValueError 异常
    with pytest.raises(ValueError):
        np.binary_repr(-5, width=2)

# 定义测试函数 test_npy_char_raises，用于测试从 numpy._core._multiarray_tests 模块导入的 npy_char_deprecation 函数抛出 ValueError 异常的情况
def test_npy_char_raises():
    # 使用 pytest 模块确保调用 npy_char_deprecation 函数会抛出 ValueError 异常
    from numpy._core._multiarray_tests import npy_char_deprecation
    with pytest.raises(ValueError):
        npy_char_deprecation()

# 定义测试类 TestDevice，用于测试 numpy 数组的设备属性和 to_device 方法
class TestDevice:
    """
    Test arr.device attribute and arr.to_device() method.
    """
    # 使用 pytest 的 parametrize 装饰器为该类中的测试方法传递参数 func 和 arg
    @pytest.mark.parametrize("func, arg", [
        (np.arange, 5),
        (np.empty_like, []),
        (np.zeros, 5),
        (np.empty, (5, 5)),
        (np.asarray, []),
        (np.asanyarray, []),
    ])
    # 定义测试方法，用于测试给定函数 func 对参数 arg 的行为
    def test_device(self, func, arg):
        # 调用 func 处理参数 arg，并断言返回的数组设备为 "cpu"
        arr = func(arg)
        assert arr.device == "cpu"
        
        # 再次调用 func 处理参数 arg，设备为 None，断言返回的数组设备仍为 "cpu"
        arr = func(arg, device=None)
        assert arr.device == "cpu"
        
        # 再次调用 func 处理参数 arg，指定设备为 "cpu"，断言返回的数组设备为 "cpu"
        arr = func(arg, device="cpu")
        assert arr.device == "cpu"

        # 使用 assert_raises_regex 上下文管理器，测试 func 在设备参数为 "nonsense" 时是否抛出 ValueError 异常
        with assert_raises_regex(
            ValueError,
            r"Device not understood. Only \"cpu\" is allowed, "
            r"but received: nonsense"
        ):
            func(arg, device="nonsense")

        # 使用 assert_raises_regex 上下文管理器，测试试图设置数组 arr 的 device 属性为 "other" 是否抛出 AttributeError 异常
        with assert_raises_regex(
            AttributeError,
            r"attribute 'device' of '(numpy.|)ndarray' objects is "
            r"not writable"
        ):
            arr.device = "other"

    # 定义测试方法，用于测试 numpy 数组的 to_device 方法
    def test_to_device(self):
        # 创建一个长度为 5 的 numpy 数组 arr
        arr = np.arange(5)

        # 调用 arr 的 to_device 方法，设备为 "cpu"，断言返回的是原始数组 arr
        assert arr.to_device("cpu") is arr
        
        # 使用 assert_raises_regex 上下文管理器，测试调用 to_device 方法时传递了不支持的 stream 参数是否抛出 ValueError 异常
        with assert_raises_regex(
            ValueError,
            r"The stream argument in to_device\(\) is not supported"
        ):
            arr.to_device("cpu", stream=1)
```