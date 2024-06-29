# `.\numpy\numpy\_core\tests\test_dtype.py`

```py
# 导入系统相关模块
import sys
# 导入运算符模块，用于操作符的函数集合
import operator
# 导入 pytest 测试框架模块
import pytest
# ctypes 是一个提供 C 语言数据类型的 Python 扩展库
import ctypes
# gc 是 Python 的垃圾回收模块
import gc
# types 模块包含用于操作类型和创建新类型的函数
import types
# 导入 pickle 模块，用于序列化和反序列化 Python 对象
import pickle

# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 numpy 的数据类型模块
import numpy.dtypes
# 导入 numpy 的有理数模块
from numpy._core._rational_tests import rational
# 导入 numpy 的自定义字段数据类型创建模块
from numpy._core._multiarray_tests import create_custom_field_dtype
# 导入 numpy 的测试函数集
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_raises, HAS_REFCOUNT,
    IS_PYSTON, _OLD_PROMOTION)
# 导入 permutations 函数，用于生成迭代器的所有可能排列
from itertools import permutations
# 导入 random 模块，用于生成随机数
import random

# 导入 hypothesis 库，用于基于假设的快速测试
import hypothesis
# 导入 hypothesis.extra.numpy 模块，提供了对 numpy 数组的假设支持
from hypothesis.extra import numpy as hynp


# 定义一个函数，用于断言两个数据类型相等
def assert_dtype_equal(a, b):
    # 使用 assert_equal 函数断言 a 和 b 相等
    assert_equal(a, b)
    # 断言两个等价类型的哈希值也相等
    assert_equal(hash(a), hash(b),
                 "two equivalent types do not hash to the same value !")

# 定义一个函数，用于断言两个数据类型不相等
def assert_dtype_not_equal(a, b):
    # 断言 a 和 b 不相等
    assert_(a != b)
    # 断言两个不同类型的哈希值不相等
    assert_(hash(a) != hash(b),
            "two different types hash to the same value !")

# 定义一个测试类 TestBuiltin
class TestBuiltin:
    # 使用 pytest 的参数化装饰器，测试函数接受多个参数化的 t
    @pytest.mark.parametrize('t', [int, float, complex, np.int32, str, object])
    def test_run(self, t):
        """Only test hash runs at all."""
        # 创建一个 numpy 数据类型对象 dt
        dt = np.dtype(t)
        # 对 dt 进行哈希运算
        hash(dt)

    # 使用 pytest 的参数化装饰器，测试函数接受多个参数化的 t
    @pytest.mark.parametrize('t', [int, float])
    def test_dtype(self, t):
        # 创建一个 numpy 数据类型对象 dt
        dt = np.dtype(t)
        # 创建新的字节顺序为小端的数据类型对象 dt2
        dt2 = dt.newbyteorder("<")
        # 创建新的字节顺序为大端的数据类型对象 dt3
        dt3 = dt.newbyteorder(">")
        # 如果 dt 等于 dt2
        if dt == dt2:
            # 断言 dt 的字节顺序不等于 dt2 的字节顺序，验证测试是否无效
            assert_(dt.byteorder != dt2.byteorder, "bogus test")
            # 断言 dt 和 dt2 的数据类型相等
            assert_dtype_equal(dt, dt2)
        else:
            # 断言 dt 的字节顺序不等于 dt3 的字节顺序，验证测试是否无效
            assert_(dt.byteorder != dt3.byteorder, "bogus test")
            # 断言 dt 和 dt3 的数据类型相等
            assert_dtype_equal(dt, dt3)

    # 测试等效数据类型的哈希值
    def test_equivalent_dtype_hashing(self):
        # 创建一个无符号整数指针类型的 numpy 数据类型对象 uintp
        uintp = np.dtype(np.uintp)
        # 如果 uintp 的字节大小为 4
        if uintp.itemsize == 4:
            # 将 left 设为 uintp
            left = uintp
            # 将 right 设为 32 位无符号整数的数据类型对象
            right = np.dtype(np.uint32)
        else:
            # 将 left 设为 uintp
            left = uintp
            # 将 right 设为 64 位无符号长整数的数据类型对象
            right = np.dtype(np.ulonglong)
        # 断言 left 等于 right
        assert_(left == right)
        # 断言 left 和 right 的哈希值相等
        assert_(hash(left) == hash(right))
    def test_invalid_types(self):
        # 确保对于无效的类型字符串会引发错误

        # 检查特定类型字符串是否会引发 TypeError 异常
        assert_raises(TypeError, np.dtype, 'O3')
        assert_raises(TypeError, np.dtype, 'O5')
        assert_raises(TypeError, np.dtype, 'O7')
        assert_raises(TypeError, np.dtype, 'b3')
        assert_raises(TypeError, np.dtype, 'h4')
        assert_raises(TypeError, np.dtype, 'I5')
        assert_raises(TypeError, np.dtype, 'e3')
        assert_raises(TypeError, np.dtype, 'f5')

        # 根据 'g' 的 itemsize 属性确定条件
        if np.dtype('g').itemsize == 8 or np.dtype('g').itemsize == 16:
            # 当 'g' 的 itemsize 为 8 或 16 时，检查 'g12' 是否引发 TypeError 异常
            assert_raises(TypeError, np.dtype, 'g12')
        elif np.dtype('g').itemsize == 12:
            # 当 'g' 的 itemsize 为 12 时，检查 'g16' 是否引发 TypeError 异常
            assert_raises(TypeError, np.dtype, 'g16')

        # 根据 'l' 的 itemsize 属性确定条件
        if np.dtype('l').itemsize == 8:
            # 当 'l' 的 itemsize 为 8 时，检查 'l4' 和 'L4' 是否引发 TypeError 异常
            assert_raises(TypeError, np.dtype, 'l4')
            assert_raises(TypeError, np.dtype, 'L4')
        else:
            # 当 'l' 的 itemsize 不为 8 时，检查 'l8' 和 'L8' 是否引发 TypeError 异常
            assert_raises(TypeError, np.dtype, 'l8')
            assert_raises(TypeError, np.dtype, 'L8')

        # 根据 'q' 的 itemsize 属性确定条件
        if np.dtype('q').itemsize == 8:
            # 当 'q' 的 itemsize 为 8 时，检查 'q4' 和 'Q4' 是否引发 TypeError 异常
            assert_raises(TypeError, np.dtype, 'q4')
            assert_raises(TypeError, np.dtype, 'Q4')
        else:
            # 当 'q' 的 itemsize 不为 8 时，检查 'q8' 和 'Q8' 是否引发 TypeError 异常
            assert_raises(TypeError, np.dtype, 'q8')
            assert_raises(TypeError, np.dtype, 'Q8')

        # 确保负大小的 dtype 引发 TypeError 异常
        assert_raises(TypeError, np.dtype, 'S-1')
        assert_raises(TypeError, np.dtype, 'U-1')
        assert_raises(TypeError, np.dtype, 'V-1')

    def test_richcompare_invalid_dtype_equality(self):
        # 确保无法转换为有效 dtype 的对象在与有效 dtype 比较时返回 False/True
        # 这里的 7 无法转换为 dtype，因此比较时不应引发异常

        # 检查 np.int32 类型的 dtype 是否等于 7，应返回 False
        assert not np.dtype(np.int32) == 7, "dtype richcompare failed for =="
        # 检查 np.int32 类型的 dtype 是否不等于 7，应返回 True
        assert np.dtype(np.int32) != 7, "dtype richcompare failed for !="

    @pytest.mark.parametrize(
        'operation',
        [operator.le, operator.lt, operator.ge, operator.gt])
    def test_richcompare_invalid_dtype_comparison(self, operation):
        # 确保对于无效 dtype，比较操作符会引发 TypeError 异常
        # 这里的 7 是一个无效的 dtype。

        # 使用 pytest 的 raises 来检查在比较操作中是否会引发 TypeError 异常
        with pytest.raises(TypeError):
            operation(np.dtype(np.int32), 7)

    @pytest.mark.parametrize("dtype",
             ['Bool', 'Bytes0', 'Complex32', 'Complex64',
              'Datetime64', 'Float16', 'Float32', 'Float64',
              'Int8', 'Int16', 'Int32', 'Int64',
              'Object0', 'Str0', 'Timedelta64',
              'UInt8', 'UInt16', 'Uint32', 'UInt32',
              'Uint64', 'UInt64', 'Void0',
              "Float128", "Complex128"])
    def test_numeric_style_types_are_invalid(self, dtype):
        # 确保数值风格的类型字符串会引发 TypeError 异常

        # 使用 assert_raises 来检查是否会引发 TypeError 异常
        with assert_raises(TypeError):
            np.dtype(dtype)
    def test_expired_dtypes_with_bad_bytesize(self):
        # 定义正则表达式匹配模式，用于捕获 NumPy 2.0 中移除的数据类型错误
        match: str = r".*removed in NumPy 2.0.*"

        # 使用 pytest 检查以下数据类型定义是否抛出 TypeError，并匹配预定义的错误消息模式
        with pytest.raises(TypeError, match=match):
            np.dtype("int0")
        with pytest.raises(TypeError, match=match):
            np.dtype("uint0")
        with pytest.raises(TypeError, match=match):
            np.dtype("bool8")
        with pytest.raises(TypeError, match=match):
            np.dtype("bytes0")
        with pytest.raises(TypeError, match=match):
            np.dtype("str0")
        with pytest.raises(TypeError, match=match):
            np.dtype("object0")
        with pytest.raises(TypeError, match=match):
            np.dtype("void0")

    @pytest.mark.parametrize(
        'value',
        ['m8', 'M8', 'datetime64', 'timedelta64',
         'i4, (2,3)f8, f4', 'S3, 3u8, (3,4)S10',
         '>f', '<f', '=f', '|f',
        ])
    def test_dtype_bytes_str_equivalence(self, value):
        # 将字符串编码为 ASCII 字节表示
        bytes_value = value.encode('ascii')
        # 使用 ASCII 字节表示创建 NumPy dtype 对象
        from_bytes = np.dtype(bytes_value)
        # 直接使用字符串创建 NumPy dtype 对象
        from_str = np.dtype(value)
        # 断言两种创建方式得到的 dtype 对象相等
        assert_dtype_equal(from_bytes, from_str)

    def test_dtype_from_bytes(self):
        # 检查空的字节对象是否会抛出 TypeError
        assert_raises(TypeError, np.dtype, b'')
        # 检查带有字节顺序指示符但没有类型信息的字节对象是否会抛出 TypeError
        assert_raises(TypeError, np.dtype, b'|')

        # 使用单个字节字符 < NPY_NTYPES_LEGACY 的值作为索引，返回对应的类型
        assert_dtype_equal(np.dtype(bytes([0])), np.dtype('bool'))
        assert_dtype_equal(np.dtype(bytes([17])), np.dtype(object))

        # 使用单个字节表示有效的类型码 'f'，验证是否正确解析为 'float32'
        assert_dtype_equal(np.dtype(b'f'), np.dtype('float32'))

        # 检查带有非 ASCII 值的字节对象是否会抛出 TypeError
        assert_raises(TypeError, np.dtype, b'\xff')
        assert_raises(TypeError, np.dtype, b's\xff')

    def test_bad_param(self):
        # 检查给定过小的 itemsize 是否会抛出 ValueError
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i4', 'i1'],
                         'offsets':[0, 4],
                         'itemsize':4})
        # 如果启用了对齐，itemsize 必须是对齐值（4）的倍数
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i4', 'i1'],
                         'offsets':[0, 4],
                         'itemsize':9}, align=True)
        # 如果启用了对齐，各个字段必须按照对齐要求进行对齐
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i1', 'f4'],
                         'offsets':[0, 2]}, align=True)
    # 定义一个测试方法，用于测试字段顺序的相等性
    def test_field_order_equality(self):
        # 创建一个 NumPy 数据类型 x，包含字段名 'A' 和 'B'，对应格式为整数和浮点数，偏移量分别为 0 和 4
        x = np.dtype({'names': ['A', 'B'],
                      'formats': ['i4', 'f4'],
                      'offsets': [0, 4]})
        # 创建另一个 NumPy 数据类型 y，字段名 'B' 和 'A'，对应格式为整数和浮点数，偏移量分别为 4 和 0
        y = np.dtype({'names': ['B', 'A'],
                      'formats': ['i4', 'f4'],
                      'offsets': [4, 0]})
        # 断言 x 和 y 不相等
        assert_equal(x == y, False)
        # 使用 "safe" 类型转换，验证 x 是否可以安全地转换为 y
        assert np.can_cast(x, y, casting="safe")

    # 使用参数化测试装饰器，测试直接创建字符串类型数据类型
    @pytest.mark.parametrize(
        ["type_char", "char_size", "scalar_type"],
        [["U", 4, np.str_],
         ["S", 1, np.bytes_]])
    def test_create_string_dtypes_directly(
            self, type_char, char_size, scalar_type):
        # 获取相应的数据类型类对象
        dtype_class = type(np.dtype(type_char))

        # 创建指定大小的数据类型
        dtype = dtype_class(8)
        # 断言数据类型是期望的标量类型
        assert dtype.type is scalar_type
        # 断言数据类型的字节大小符合预期
        assert dtype.itemsize == 8*char_size

    # 测试创建无效字符串类型数据类型时的错误情况
    def test_create_invalid_string_errors(self):
        # 计算一个超出整数容量范围的值
        one_too_big = np.iinfo(np.intc).max + 1
        # 应该抛出 TypeError 异常
        with pytest.raises(TypeError):
            type(np.dtype("U"))(one_too_big // 4)

        # 应该抛出 TypeError 异常，用于非常大的数值
        with pytest.raises(TypeError):
            type(np.dtype("U"))(np.iinfo(np.intp).max // 4 + 1)

        # 如果超出整数容量范围，则应该抛出 TypeError 异常
        if one_too_big < sys.maxsize:
            with pytest.raises(TypeError):
                type(np.dtype("S"))(one_too_big)

        # 应该抛出 ValueError 异常，因为大小为负数
        with pytest.raises(ValueError):
            type(np.dtype("U"))(-1)

        # 在 32 位系统上可能会出现 OverflowError
        with pytest.raises((TypeError, OverflowError)):
            # 测试一个非常大的数值
            type(np.dtype("S"))(2**61)

        # 应该抛出 TypeError 异常，因为数据类型名中包含无效字符
        with pytest.raises(TypeError):
            np.dtype("S1234hello")

    # 测试解析时忽略前导零
    def test_leading_zero_parsing(self):
        # 创建数据类型 dt1 和 dt2，都表示长度为 10 的字符串，但忽略了 dt1 中的前导零
        dt1 = np.dtype('S010')
        dt2 = np.dtype('S10')

        # 断言 dt1 和 dt2 是相等的
        assert dt1 == dt2
        # 断言 dt1 的字符串表示为 "dtype('S10')"
        assert repr(dt1) == "dtype('S10')"
        # 断言 dt1 的字节大小为 10
        assert dt1.itemsize == 10
class TestRecord:
    def test_equivalent_record(self):
        """Test whether equivalent record dtypes hash the same."""
        # 定义包含单个字段 'yo' 的 dtype
        a = np.dtype([('yo', int)])
        # 定义另一个与前一个相同的 dtype
        b = np.dtype([('yo', int)])
        # 断言这两个 dtype 相等
        assert_dtype_equal(a, b)

    def test_different_names(self):
        # 在理论上，它们可能会哈希相同（碰撞）？
        # 定义包含 'yo' 和 'ye' 两个不同字段的 dtype
        a = np.dtype([('yo', int)])
        b = np.dtype([('ye', int)])
        # 断言这两个 dtype 不相等
        assert_dtype_not_equal(a, b)

    def test_different_titles(self):
        # 在理论上，它们可能会哈希相同（碰撞）？
        # 定义两个结构化 dtype，每个都包含 'r' 和 'b' 两个字段，但标题略有不同
        a = np.dtype({'names': ['r', 'b'],
                      'formats': ['u1', 'u1'],
                      'titles': ['Red pixel', 'Blue pixel']})
        b = np.dtype({'names': ['r', 'b'],
                      'formats': ['u1', 'u1'],
                      'titles': ['RRed pixel', 'Blue pixel']})
        # 断言这两个 dtype 不相等
        assert_dtype_not_equal(a, b)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_refcount_dictionary_setting(self):
        # 测试 dtype 对象的引用计数设置
        names = ["name1"]
        formats = ["f8"]
        titles = ["t1"]
        offsets = [0]
        # 创建一个包含字段名、格式、标题和偏移量的字典
        d = dict(names=names, formats=formats, titles=titles, offsets=offsets)
        # 记录字典中每个值的引用计数
        refcounts = {k: sys.getrefcount(i) for k, i in d.items()}
        # 创建一个新的 dtype 对象并再次记录引用计数
        np.dtype(d)
        refcounts_new = {k: sys.getrefcount(i) for k, i in d.items()}
        # 断言新旧两组引用计数相同
        assert refcounts == refcounts_new

    def test_mutate(self):
        # 改变 dtype 应该重置其缓存的哈希值。
        # 注意：改变操作应该被弃用，但新 API 已添加以替代它。
        # 定义初始的 dtype
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        c = np.dtype([('ye', int)])
        # 断言 a 和 b 相等，a 和 c 不相等
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)
        # 修改字段名称并再次断言 a 和 c 相等，a 和 b 不相等
        a.names = ['ye']
        assert_dtype_equal(a, c)
        assert_dtype_not_equal(a, b)
        # 获取 b 的状态并恢复 a 的状态，断言 a 和 b 再次相等，a 和 c 不相等
        state = b.__reduce__()[2]
        a.__setstate__(state)
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)

    def test_init_simple_structured(self):
        # 初始化简单结构化 dtype，检查字段名称
        dt1 = np.dtype("i, i")
        assert dt1.names == ("f0", "f1")

        dt2 = np.dtype("i,")
        assert dt2.names == ("f0",)

    def test_mutate_error(self):
        # 注意：改变操作应该被弃用，但新 API 已添加以替代它。
        # 定义初始 dtype 包含多个字段
        a = np.dtype("i,i")

        # 使用 pytest 断言应该触发 ValueError，因为逐个更改字段名已被弃用
        with pytest.raises(ValueError, match="must replace all names at once"):
            a.names = ["f0"]

        # 使用 pytest 断言应该触发 ValueError，因为包含非 Unicode 名称
        with pytest.raises(ValueError, match=".*and not string"):
            a.names = ["f0", b"not a unicode name"]

    def test_not_lists(self):
        """Test if an appropriate exception is raised when passing bad values to
        the dtype constructor.
        """
        # 使用 assert_raises 检查当向 dtype 构造函数传递错误值时是否引发了 TypeError
        assert_raises(TypeError, np.dtype,
                      dict(names={'A', 'B'}, formats=['f8', 'i4']))
        assert_raises(TypeError, np.dtype,
                      dict(names=['A', 'B'], formats={'f8', 'i4'}))
    def test_aligned_size(self):
        # 检查结构化数据类型是否被填充到对齐的大小
        dt = np.dtype('i4, i1', align=True)
        # 断言第一个数据类型的大小为8字节
        assert_equal(dt.itemsize, 8)
        
        dt = np.dtype([('f0', 'i4'), ('f1', 'i1')], align=True)
        # 断言第二个数据类型的大小为8字节
        assert_equal(dt.itemsize, 8)
        
        dt = np.dtype({'names':['f0', 'f1'],
                       'formats':['i4', 'u1'],
                       'offsets':[0, 4]}, align=True)
        # 断言第三个数据类型的大小为8字节
        assert_equal(dt.itemsize, 8)
        
        dt = np.dtype({'f0': ('i4', 0), 'f1':('u1', 4)}, align=True)
        # 断言第四个数据类型的大小为8字节
        assert_equal(dt.itemsize, 8)
        
        # 嵌套应保持相同的对齐
        dt1 = np.dtype([('f0', 'i4'),
                        ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]),
                        ('f2', 'i1')], align=True)
        # 断言第一个嵌套数据类型的大小为20字节
        assert_equal(dt1.itemsize, 20)
        
        dt2 = np.dtype({'names':['f0', 'f1', 'f2'],
                        'formats':['i4',
                                   [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')],
                                   'i1'],
                        'offsets':[0, 4, 16]}, align=True)
        # 断言第二个嵌套数据类型的大小为20字节
        assert_equal(dt2.itemsize, 20)
        
        dt3 = np.dtype({'f0': ('i4', 0),
                        'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4),
                        'f2': ('i1', 16)}, align=True)
        # 断言第三个嵌套数据类型的大小为20字节
        assert_equal(dt3.itemsize, 20)
        
        # 断言三个嵌套数据类型相等
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        
        # 嵌套应保持打包
        dt1 = np.dtype([('f0', 'i4'),
                        ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]),
                        ('f2', 'i1')], align=False)
        # 断言第一个非对齐的嵌套数据类型的大小为11字节
        assert_equal(dt1.itemsize, 11)
        
        dt2 = np.dtype({'names':['f0', 'f1', 'f2'],
                        'formats':['i4',
                                   [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')],
                                   'i1'],
                        'offsets':[0, 4, 10]}, align=False)
        # 断言第二个非对齐的嵌套数据类型的大小为11字节
        assert_equal(dt2.itemsize, 11)
        
        dt3 = np.dtype({'f0': ('i4', 0),
                        'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4),
                        'f2': ('i1', 10)}, align=False)
        # 断言第三个非对齐的嵌套数据类型的大小为11字节
        assert_equal(dt3.itemsize, 11)
        
        # 断言三个非对齐的嵌套数据类型相等
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        
        # 子类型的数组应保持对齐
        dt1 = np.dtype([('a', '|i1'),
                        ('b', [('f0', '<i2'),
                              ('f1', '<f4')], 2)], align=True)
        # 断言子类型的数组的描述符
        assert_equal(dt1.descr, [('a', '|i1'), ('', '|V3'),
                                 ('b', [('f0', '<i2'), ('', '|V2'),
                                       ('f1', '<f4')], (2,))])
    def test_empty_struct_alignment(self):
        # 测试空结构的对齐方式
        # 创建一个空的 dtype，指定对齐方式为 True，默认对齐到 1
        dt = np.dtype([], align=True)
        # 断言该 dtype 的对齐方式为 1
        assert_equal(dt.alignment, 1)

        # 创建一个包含一个空字段 'f0' 的 dtype，指定对齐方式为 True
        dt = np.dtype([('f0', [])], align=True)
        # 断言该 dtype 的对齐方式为 1
        assert_equal(dt.alignment, 1)

        # 创建一个空的结构字典 dtype，指定对齐方式为 True
        dt = np.dtype({'names': [],
                       'formats': [],
                       'offsets': []}, align=True)
        # 断言该 dtype 的对齐方式为 1
        assert_equal(dt.alignment, 1)

        # 创建一个包含字段 'f0' 的 dtype，其中该字段为空列表，指定对齐方式为 True
        dt = np.dtype({'names': ['f0'],
                       'formats': [[]],
                       'offsets': [0]}, align=True)
        # 断言该 dtype 的对齐方式为 1
        assert_equal(dt.alignment, 1)

    def test_union_struct(self):
        # 测试创建联合结构的 dtype
        # 创建一个包含三个字段 'f0', 'f1', 'f2' 的 dtype，各自指定不同的格式和偏移量，指定对齐方式为 True
        dt = np.dtype({'names':['f0', 'f1', 'f2'], 'formats':['<u4', '<u2', '<u2'],
                        'offsets':[0, 0, 2]}, align=True)
        # 断言该 dtype 的总字节大小为 4
        assert_equal(dt.itemsize, 4)

        # 创建一个 uint32 类型的数组 a，视图使用上面定义的 dtype
        a = np.array([3], dtype='<u4').view(dt)
        # 对字段 'f1' 和 'f2' 赋值
        a['f1'] = 10
        a['f2'] = 36
        # 断言字段 'f0' 的值应为 10 + 36 * 256 * 256
        assert_equal(a['f0'], 10 + 36*256*256)

        # 创建一个包含字段 'f0', 'f1', 'f2' 的 dtype，但指定的偏移量不按顺序排列，指定对齐方式为 True
        dt = np.dtype({'names':['f0', 'f1', 'f2'], 'formats':['<u4', '<u2', '<u2'],
                        'offsets':[4, 0, 2]}, align=True)
        # 断言该 dtype 的总字节大小为 8
        assert_equal(dt.itemsize, 8)

        # 创建另一个与上述 dtype 结构等效的 dtype，但字段名顺序不同，指定对齐方式为 True
        dt2 = np.dtype({'names':['f2', 'f0', 'f1'],
                        'formats':['<u4', '<u2', '<u2'],
                        'offsets':[4, 0, 2]}, align=True)
        # 创建数据数组 vals 和 vals2
        vals = [(0, 1, 2), (3, 2**15-1, 4)]
        vals2 = [(0, 1, 2), (3, 2**15-1, 4)]
        # 创建数组 a 和 b，分别使用 dt 和 dt2
        a = np.array(vals, dt)
        b = np.array(vals2, dt2)
        # 断言 a 转换为 dt2 结构与 b 相等
        assert_equal(a.astype(dt2), b)
        # 断言 b 转换为 dt 结构与 a 相等
        assert_equal(b.astype(dt), a)
        # 断言 a 的视图使用 dt2 结构与 b 相等
        assert_equal(a.view(dt2), b)
        # 断言 b 的视图使用 dt 结构与 a 相等
        assert_equal(b.view(dt), a)

        # 尝试创建与其他类型重叠的对象，预期会抛出 TypeError 异常
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['O', 'i1'],
                 'offsets':[0, 2]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['i4', 'O'],
                 'offsets':[0, 3]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':[[('a', 'O')], 'i1'],
                 'offsets':[0, 2]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['i4', [('a', 'O')]],
                 'offsets':[0, 3]})

        # 创建一个字段顺序不同但结构相同的 dtype，指定对齐方式为 True
        dt = np.dtype({'names':['f0', 'f1'],
                       'formats':['i1', 'O'],
                       'offsets':[np.dtype('intp').itemsize, 0]})
    def test_subarray_list(self, obj, dtype, expected):
        # 将 dtype 转换为 NumPy 的数据类型对象
        dtype = np.dtype(dtype)
        # 使用指定的 dtype 创建 NumPy 数组
        res = np.array(obj, dtype=dtype)

        if expected is None:
            # 如果期望结果为空，则迭代 1 维列表以填充数组
            expected = np.empty(len(obj), dtype=dtype)
            for i in range(len(expected)):
                expected[i] = obj[i]

        # 使用 assert_array_equal 函数比较 res 和 expected 数组是否相等
        assert_array_equal(res, expected)

    def test_parenthesized_single_number(self):
        # 使用 pytest 的断言检查是否引发了 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match="not understood"):
            np.dtype("(2)f4")

        # 下面这段代码用来测试 DeprecationWarning，已经被移动到 test_deprecations.py::TestDeprecatedDTypeParenthesizedRepeatCount 中
        # 保留在这里以便于将来转换为异常检查
        with pytest.warns(DeprecationWarning,
                          match="parenthesized single number"):
            np.dtype("(2)f4,")

    def test_comma_datetime(self):
        # 创建一个复合数据类型 dt，包含三个字段
        dt = np.dtype('M8[D],datetime64[Y],i8')
        # 使用 assert_equal 函数比较 dt 和指定的数据类型是否相等
        assert_equal(dt, np.dtype([('f0', 'M8[D]'),
                                   ('f1', 'datetime64[Y]'),
                                   ('f2', 'i8')]))

    def test_from_dictproxy(self):
        # 用于测试 PR #5920 的功能
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'f4']})
        # 使用 assert_dtype_equal 函数比较两个数据类型是否相等
        assert_dtype_equal(dt, np.dtype(dt.fields))
        # 创建一个新的数据类型 dt2，与 dt 相同
        dt2 = np.dtype((np.void, dt.fields))
        # 使用 assert_equal 函数比较两个数据类型的字段是否相等
        assert_equal(dt2.fields, dt.fields)

    def test_from_dict_with_zero_width_field(self):
        # 用于测试问题 #6430 / #2196 的回归测试
        dt = np.dtype([('val1', np.float32, (0,)), ('val2', int)])
        # 创建一个相同结构的数据类型 dt2
        dt2 = np.dtype({'names': ['val1', 'val2'],
                        'formats': [(np.float32, (0,)), int]})
        # 使用 assert_dtype_equal 函数比较两个数据类型是否相等
        assert_dtype_equal(dt, dt2)
        # 使用 assert_equal 函数检查字段 'val1' 的字节大小是否为 0
        assert_equal(dt.fields['val1'][0].itemsize, 0)
        # 使用 assert_equal 函数检查整个数据类型的字节大小是否正确
        assert_equal(dt.itemsize, dt.fields['val2'][0].itemsize)

    def test_bool_commastring(self):
        # 创建一个复合数据类型 d，包含三个字段，每个字段为布尔类型
        d = np.dtype('?,?,?')  # raises?
        # 使用 assert_equal 函数检查字段数量是否为 3
        assert_equal(len(d.names), 3)
        # 遍历字段名，使用 assert_equal 函数检查每个字段的数据类型是否为布尔类型
        for n in d.names:
            assert_equal(d.fields[n][0], np.dtype('?'))

    def test_nonint_offsets(self):
        # gh-8059
        # 定义一个函数 make_dtype，用于创建指定偏移量的数据类型
        def make_dtype(off):
            return np.dtype({'names': ['A'], 'formats': ['i4'],
                             'offsets': [off]})

        # 使用 assert_raises 函数检查 make_dtype 函数在输入错误类型时是否会引发异常
        assert_raises(TypeError, make_dtype, 'ASD')
        # 使用 assert_raises 函数检查 make_dtype 函数在输入超出范围的偏移量时是否会引发异常
        assert_raises(OverflowError, make_dtype, 2**70)
        # 使用 assert_raises 函数检查 make_dtype 函数在输入错误类型的偏移量时是否会引发异常
        assert_raises(TypeError, make_dtype, 2.3)
        # 使用 assert_raises 函数检查 make_dtype 函数在输入负数偏移量时是否会引发异常
        assert_raises(ValueError, make_dtype, -10)

        # 创建一个数据类型 dt，偏移量为 0，不应引发异常
        dt = make_dtype(np.uint32(0))
        # 调用 np.zeros 函数创建一个数组，使用指定的数据类型 dt
        np.zeros(1, dtype=dt)[0].item()

    def test_fields_by_index(self):
        # 创建一个复合数据类型 dt，包含两个字段
        dt = np.dtype([('a', np.int8), ('b', np.float32, 3)])
        # 使用 assert_dtype_equal 函数比较索引 0 处的字段数据类型是否为 np.int8
        assert_dtype_equal(dt[0], np.dtype(np.int8))
        # 使用 assert_dtype_equal 函数比较索引 1 处的字段数据类型是否为 (np.float32, 3)
        assert_dtype_equal(dt[1], np.dtype((np.float32, 3)))
        # 使用 assert_dtype_equal 函数比较负索引 -1 处的字段数据类型是否与索引 1 处相同
        assert_dtype_equal(dt[-1], dt[1])
        # 使用 assert_dtype_equal 函数比较负索引 -2 处的字段数据类型是否与索引 0 处相同
        assert_dtype_equal(dt[-2], dt[0])
        # 使用 assert_raises 函数检查索引 -3 是否引发 IndexError 异常
        assert_raises(IndexError, lambda: dt[-3])

        # 使用 assert_raises 函数检查传入非整数索引是否引发 TypeError 异常
        assert_raises(TypeError, operator.getitem, dt, 3.0)

        # 使用 assert_equal 函数比较索引 1 处的字段数据类型是否与索引 np.int8(1) 处相同
        assert_equal(dt[1], dt[np.int8(1)])
    @pytest.mark.parametrize('align_flag',[False, True])
    def test_multifield_index(self, align_flag):
        # 使用 pytest 的参数化标记来定义测试用例，测试多字段索引功能
        # align_flag 表示是否对齐的标志，在不同的测试参数下进行测试

        # 定义一个结构化数据类型 dt，包含字段 'title' 和 'col1'，以及 'A' 和 'B' 两个字段
        dt = np.dtype([
            (('title', 'col1'), '<U20'), ('A', '<f8'), ('B', '<f8')
        ], align=align_flag)

        # 通过列表索引获取子数据类型 dt_sub，包含 'B' 和 'col1' 两个字段
        dt_sub = dt[['B', 'col1']]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': ['B', 'col1'],
                'formats': ['<f8', '<U20'],
                'offsets': [88, 0],
                'titles': [None, 'title'],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        # 通过列表索引获取子数据类型 dt_sub，只包含 'B' 字段
        dt_sub = dt[['B']]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': ['B'],
                'formats': ['<f8'],
                'offsets': [88],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        # 通过空列表索引获取子数据类型 dt_sub，不包含任何字段
        dt_sub = dt[[]]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': [],
                'formats': [],
                'offsets': [],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        # 测试异常情况：尝试使用不支持的索引方式
        assert_raises(TypeError, operator.getitem, dt, ())
        assert_raises(TypeError, operator.getitem, dt, [1, 2, 3])
        assert_raises(TypeError, operator.getitem, dt, ['col1', 2])
        assert_raises(KeyError, operator.getitem, dt, ['fake'])
        assert_raises(KeyError, operator.getitem, dt, ['title'])
        assert_raises(ValueError, operator.getitem, dt, ['col1', 'col1'])

    def test_partial_dict(self):
        # 测试异常情况：创建结构化数据类型时缺少 'names' 字段
        assert_raises(ValueError, np.dtype,
                {'formats': ['i4', 'i4'], 'f0': ('i4', 0), 'f1':('i4', 4)})

    def test_fieldless_views(self):
        # 创建一个没有字段的数组 a
        a = np.zeros(2, dtype={'names':[], 'formats':[], 'offsets':[],
                               'itemsize':8})
        # 测试异常情况：尝试视图转换为没有字段的数据类型
        assert_raises(ValueError, a.view, np.dtype([]))

        # 创建一个没有基础数据类型的数据类型 d
        d = np.dtype((np.dtype([]), 10))
        assert_equal(d.shape, (10,))
        assert_equal(d.itemsize, 0)
        assert_equal(d.base, np.dtype([]))

        # 创建一个空的数组 arr，数据类型为没有字段的数据类型 []
        arr = np.fromiter((() for i in range(10)), [])
        assert_equal(arr.dtype, np.dtype([]))
        assert_raises(ValueError, np.frombuffer, b'', dtype=[])
        assert_equal(np.frombuffer(b'', dtype=[], count=2),
                     np.empty(2, dtype=[]))

        # 测试异常情况：尝试创建不支持的数据类型 ([], 'f8')
        assert_raises(ValueError, np.dtype, ([], 'f8'))
        assert_raises(ValueError, np.zeros(1, dtype='i4').view, [])

        # 对比两个没有字段的数组是否相等
        assert_equal(np.zeros(2, dtype=[]) == np.zeros(2, dtype=[]),
                     np.ones(2, dtype=bool))

        # 对比两个没有字段的数组是否相等
        assert_equal(np.zeros((1, 2), dtype=[]) == a,
                     np.ones((1, 2), dtype=bool))
    def test_nonstructured_with_object(self):
        # 使用说明：这是一个测试函数，用于测试非结构化的包含对象的情况
        # 在这里，如果下面关于 dtype 的断言失败，测试将变得毫无意义（这是可以接受的）
        
        # 创建一个空的记录数组，dtype 设置为对象类型
        arr = np.recarray((0,), dtype="O")
        
        # 断言数组的 dtype 中没有命名字段
        assert arr.dtype.names is None  # no fields
        
        # 断言数组的 dtype 声明包含对象
        assert arr.dtype.hasobject  # but claims to contain objects
        
        # 删除数组以释放资源，此操作之前曾经失败过
        del arr  # the deletion failed previously.
class TestSubarray:
    # 定义测试类 TestSubarray

    def test_single_subarray(self):
        # 定义测试方法 test_single_subarray
        a = np.dtype((int, (2)))
        # 创建一个 NumPy 数据类型 a，元素类型为 int，形状为 (2)
        b = np.dtype((int, (2,)))
        # 创建一个 NumPy 数据类型 b，元素类型为 int，形状为 (2,)
        assert_dtype_equal(a, b)
        # 断言 a 和 b 的数据类型相等

        assert_equal(type(a.subdtype[1]), tuple)
        # 断言 a 的第二个子数据类型为元组类型
        assert_equal(type(b.subdtype[1]), tuple)
        # 断言 b 的第二个子数据类型为元组类型

    def test_equivalent_record(self):
        """Test whether equivalent subarray dtypes hash the same."""
        # 测试等价子数组数据类型是否具有相同的哈希值
        a = np.dtype((int, (2, 3)))
        # 创建一个形状为 (2, 3) 的 NumPy 数据类型 a，元素类型为 int
        b = np.dtype((int, (2, 3)))
        # 创建一个形状为 (2, 3) 的 NumPy 数据类型 b，元素类型为 int
        assert_dtype_equal(a, b)
        # 断言 a 和 b 的数据类型相等

    def test_nonequivalent_record(self):
        """Test whether different subarray dtypes hash differently."""
        # 测试不同的子数组数据类型是否具有不同的哈希值
        a = np.dtype((int, (2, 3)))
        # 创建一个形状为 (2, 3) 的 NumPy 数据类型 a，元素类型为 int
        b = np.dtype((int, (3, 2)))
        # 创建一个形状为 (3, 2) 的 NumPy 数据类型 b，元素类型为 int
        assert_dtype_not_equal(a, b)
        # 断言 a 和 b 的数据类型不相等

        a = np.dtype((int, (2, 3)))
        # 创建一个形状为 (2, 3) 的 NumPy 数据类型 a，元素类型为 int
        b = np.dtype((int, (2, 2)))
        # 创建一个形状为 (2, 2) 的 NumPy 数据类型 b，元素类型为 int
        assert_dtype_not_equal(a, b)
        # 断言 a 和 b 的数据类型不相等

        a = np.dtype((int, (1, 2, 3)))
        # 创建一个形状为 (1, 2, 3) 的 NumPy 数据类型 a，元素类型为 int
        b = np.dtype((int, (1, 2)))
        # 创建一个形状为 (1, 2) 的 NumPy 数据类型 b，元素类型为 int
        assert_dtype_not_equal(a, b)
        # 断言 a 和 b 的数据类型不相等

    def test_shape_equal(self):
        """Test some data types that are equal"""
        # 测试一些相等的数据类型
        assert_dtype_equal(np.dtype('f8'), np.dtype(('f8', tuple())))
        # 断言两个数据类型 'f8' 和 ('f8', ()) 相等
        assert_dtype_equal(np.dtype('(1,)f8'), np.dtype(('f8', 1)))
        # 断言两个数据类型 '(1,)f8' 和 ('f8', 1) 相等
        assert np.dtype(('f8', 1)).shape == (1,)
        # 断言数据类型 ('f8', 1) 的形状为 (1,)
        assert_dtype_equal(np.dtype((int, 2)), np.dtype((int, (2,))))
        # 断言两个数据类型 (int, 2) 和 (int, (2,)) 相等
        assert_dtype_equal(np.dtype(('<f4', (3, 2))), np.dtype(('<f4', (3, 2))))
        # 断言两个数据类型 ('<f4', (3, 2)) 相等
        d = ([('a', 'f4', (1, 2)), ('b', 'f8', (3, 1))], (3, 2))
        assert_dtype_equal(np.dtype(d), np.dtype(d))
        # 断言自定义结构化数据类型 d 相等

    def test_shape_simple(self):
        """Test some simple cases that shouldn't be equal"""
        # 测试一些不应相等的简单情况
        assert_dtype_not_equal(np.dtype('f8'), np.dtype(('f8', (1,))))
        # 断言数据类型 'f8' 和 ('f8', (1,)) 不相等
        assert_dtype_not_equal(np.dtype(('f8', (1,))), np.dtype(('f8', (1, 1))))
        # 断言数据类型 ('f8', (1,)) 和 ('f8', (1, 1)) 不相等
        assert_dtype_not_equal(np.dtype(('f4', (3, 2))), np.dtype(('f4', (2, 3))))
        # 断言数据类型 ('f4', (3, 2)) 和 ('f4', (2, 3)) 不相等

    def test_shape_monster(self):
        """Test some more complicated cases that shouldn't be equal"""
        # 测试一些更复杂的情况，它们不应该相等
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', 'f4', (1, 2)), ('b', 'f8', (1, 3))], (2, 2))))
        # 断言两个复杂的数据类型不相等
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'i8', (1, 3))], (2, 2))))
        # 断言两个复杂的数据类型不相等
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('e', 'f8', (1, 3)), ('d', 'f4', (2, 1))], (2, 2))))
        # 断言两个复杂的数据类型不相等
        assert_dtype_not_equal(
            np.dtype(([('a', [('a', 'i4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', [('a', 'u4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))))
        # 断言两个复杂的数据类型不相等
    def test_shape_sequence(self):
        # 测试形状序列的函数
        # 任何整数序列都可以作为形状，但结果应该是一个包含基本类型整数的元组（不可变的）。

        # 创建一个包含整数的numpy数组，指定数据类型为int16
        a = np.array([1, 2, 3], dtype=np.int16)
        
        # 创建一个普通的Python列表
        l = [1, 2, 3]

        # 创建一个自定义的数据类型，其中'a'字段包含一个float32类型的数组
        dt = np.dtype([('a', 'f4', a)])
        
        # 断言'a'字段的形状是一个元组
        assert_(isinstance(dt['a'].shape, tuple))
        
        # 断言'a'字段形状元组的第一个元素是整数类型
        assert_(isinstance(dt['a'].shape[0], int))

        # 创建一个自定义的数据类型，其中'a'字段包含一个列表
        dt = np.dtype([('a', 'f4', l)])
        
        # 断言'a'字段的形状是一个元组
        assert_(isinstance(dt['a'].shape, tuple))
        
        # 未提供具体的断言信息，保留空注释行

        # 定义一个模拟整数的类
        class IntLike:
            def __index__(self):
                return 3

            def __int__(self):
                # (a PyNumber_Check fails without __int__)
                return 3

        # 创建一个自定义的数据类型，其中'a'字段包含一个IntLike类的实例
        dt = np.dtype([('a', 'f4', IntLike())])
        
        # 断言'a'字段的形状是一个元组
        assert_(isinstance(dt['a'].shape, tuple))
        
        # 断言'a'字段形状元组的第一个元素是整数类型
        assert_(isinstance(dt['a'].shape[0], int))

        # 创建一个自定义的数据类型，其中'a'字段包含一个元组(IntLike的实例)
        dt = np.dtype([('a', 'f4', (IntLike(),))])
        
        # 断言'a'字段的形状是一个元组
        assert_(isinstance(dt['a'].shape, tuple))
        
        # 断言'a'字段形状元组的第一个元素是整数类型
        assert_(isinstance(dt['a'].shape[0], int))

    def test_shape_matches_ndim(self):
        # 测试形状匹配维数的函数

        # 创建一个自定义的数据类型，其中'a'字段是一个空元组
        dt = np.dtype([('a', 'f4', ())])
        
        # 断言'a'字段的形状是一个空元组
        assert_equal(dt['a'].shape, ())
        
        # 断言'a'字段的维数是0
        assert_equal(dt['a'].ndim, 0)

        # 创建一个自定义的数据类型，其中'a'字段是一个float32类型
        dt = np.dtype([('a', 'f4')])
        
        # 断言'a'字段的形状是一个空元组
        assert_equal(dt['a'].shape, ())
        
        # 断言'a'字段的维数是0
        assert_equal(dt['a'].ndim, 0)

        # 创建一个自定义的数据类型，其中'a'字段是一个长度为4的float32类型数组
        dt = np.dtype([('a', 'f4', 4)])
        
        # 断言'a'字段的形状是一个包含一个元素4的元组
        assert_equal(dt['a'].shape, (4,))
        
        # 断言'a'字段的维数是1
        assert_equal(dt['a'].ndim, 1)

        # 创建一个自定义的数据类型，其中'a'字段是一个形状为(1, 2, 3)的float32类型数组
        dt = np.dtype([('a', 'f4', (1, 2, 3))])
        
        # 断言'a'字段的形状是一个包含(1, 2, 3)的元组
        assert_equal(dt['a'].shape, (1, 2, 3))
        
        # 断言'a'字段的维数是3
        assert_equal(dt['a'].ndim, 3)

    def test_shape_invalid(self):
        # 测试无效形状的函数
        # 检查形状是否有效。

        # 获取np.intc类型的最大整数
        max_int = np.iinfo(np.intc).max
        
        # 获取np.intp类型的最大整数
        max_intp = np.iinfo(np.intp).max
        
        # 过大的值（数据类型的一部分）
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_int // 4 + 1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_int + 1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', (max_int, 2))])
        
        # 采用不同的代码路径（更早失败：
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_intp + 1)])
        
        # 负值
        assert_raises(ValueError, np.dtype, [('a', 'f4', -1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', (-1, -1))])

    def test_alignment(self):
        # 检查子数组是否对齐的函数
        t1 = np.dtype('(1,)i4', align=True)
        t2 = np.dtype('2i4', align=True)
        
        # 断言t1和t2的对齐方式相同
        assert_equal(t1.alignment, t2.alignment)

    def test_aligned_empty(self):
        # 主要是gh-19696的回归测试：完全构造失败
        # 测试对齐空数据类型的函数

        # 创建一个空的数据类型，指定对齐方式为True
        dt = np.dtype([], align=True)
        
        # 断言dt等于空的数据类型
        assert dt == np.dtype([])
        
        # 创建一个空的数据类型，使用字典形式指定结构
        dt = np.dtype({"names": [], "formats": [], "itemsize": 0}, align=True)
        
        # 断言dt等于空的数据类型
        assert dt == np.dtype([])
    # 定义一个测试方法，验证子数组的基本项
    def test_subarray_base_item(self):
        # 创建一个包含三个元素的全一数组，每个元素为一个结构化数组的字段 "f"，包含一个长度为 3 的整数子数组
        arr = np.ones(3, dtype=[("f", "i", 3)])
        
        # 通过提取字段 "f"，将子数组转换为视图
        assert arr["f"].base is arr
        
        # 提取结构化数组的项，然后检查元组组成部分
        item = arr.item(0)
        assert type(item) is tuple and len(item) == 1
        assert item[0].base is arr

    # 定义一个测试方法，验证子数组的强制类型转换为对象时的复制行为
    def test_subarray_cast_copies(self):
        # 在旧版本的 NumPy 中，强制类型转换为对象时不会复制，但它们会出现所有权问题
        # 在 1.21 版本之后（我认为），这些问题基本上会导致崩溃。这里定义正确的行为为复制。
        arr = np.ones(3, dtype=[("f", "i", 3)])
        
        # 将数组强制类型转换为对象类型
        cast = arr.astype(object)
        
        # 遍历转换后的对象数组
        for fields in cast:
            # 每个元素应为一个元组，长度为 1
            assert type(fields) == tuple and len(fields) == 1
            
            # 获取子数组
            subarr = fields[0]
            
            # 子数组的基应为 None，即它不共享内存
            assert subarr.base is None
            
            # 检查子数组是否拥有自己的数据
            assert subarr.flags.owndata
def iter_struct_object_dtypes():
    """
    Iterates over a few complex dtypes and object pattern which
    fill the array with a given object (defaults to a singleton).

    Yields
    ------
    dtype : dtype
        Data type for NumPy arrays.
    pattern : tuple
        Structured tuple for use with `np.array`.
    count : int
        Number of objects stored in the dtype.
    singleton : object
        A singleton object. The returned pattern is constructed so that
        all objects inside the datatype are set to the singleton.
    """
    # 创建一个普通的 Python 对象
    obj = object()

    # 定义一个包含子数组的结构化数据类型
    dt = np.dtype([('b', 'O', (2, 3))])
    # 创建填充数据为单例对象的模式
    p = ([[obj] * 3] * 2,)
    # 使用 pytest 的 parametrize 生成器返回参数化测试的参数
    yield pytest.param(dt, p, 6, obj, id="<subarray>")

    # 定义包含字段和子数组的结构化数据类型
    dt = np.dtype([('a', 'i4'), ('b', 'O', (2, 3))])
    # 创建填充数据为单例对象的模式
    p = (0, [[obj] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<subarray in field>")

    # 定义包含结构化子数组的复杂结构化数据类型
    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'i1')], (2, 3))])
    # 创建填充数据为单例对象的模式
    p = (0, [[(obj, 0)] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<structured subarray 1>")

    # 定义包含两个对象字段的结构化数据类型
    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'O')], (2, 3))])
    # 创建填充数据为单例对象的模式
    p = (0, [[(obj, obj)] * 3] * 2)
    yield pytest.param(dt, p, 12, obj, id="<structured subarray 2>")


@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="Python 3.12 has immortal refcounts, this test will no longer "
           "work. See gh-23986"
)
@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
class TestStructuredObjectRefcounting:
    """These tests cover various uses of complicated structured types which
    include objects and thus require reference counting.
    """
    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    @pytest.mark.parametrize(["creation_func", "creation_obj"], [
        pytest.param(np.empty, None,
             # None is probably used for too many things
             marks=pytest.mark.skip("unreliable due to python's behaviour")),
        (np.ones, 1),
        (np.zeros, 0)])
    def test_structured_object_create_delete(self, dt, pat, count, singleton,
                                             creation_func, creation_obj):
        """Structured object reference counting in creation and deletion"""
        # 测试假设 0, 1 和 None 是单例对象
        gc.collect()
        # 获取创建对象前的引用计数
        before = sys.getrefcount(creation_obj)
        # 创建指定结构化数据类型的数组
        arr = creation_func(3, dt)

        # 获取创建对象后的引用计数
        now = sys.getrefcount(creation_obj)
        # 断言创建后引用计数增加了特定数量的对象
        assert now - before == count * 3
        # 删除数组对象
        del arr
        # 获取删除对象后的引用计数
        now = sys.getrefcount(creation_obj)
        # 断言删除后引用计数与创建前相同
        assert now == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    def test_structured_object_item_setting(self, dt, pat, count, singleton):
        """Structured object reference counting for simple item setting"""
        # 定义变量 `one` 并赋值为整数 `1`
        one = 1

        # 执行垃圾回收，获取设置前 `singleton` 的引用计数
        gc.collect()
        before = sys.getrefcount(singleton)
        # 创建一个包含多个 `pat` 副本的数组，并检查引用计数的变化
        arr = np.array([pat] * 3, dt)
        assert sys.getrefcount(singleton) - before == count * 3
        # 用 `1` 填充数组，并检查其是否正确替换：
        before2 = sys.getrefcount(one)
        arr[...] = one
        after2 = sys.getrefcount(one)
        assert after2 - before2 == count * 3
        # 删除数组 `arr`，执行垃圾回收
        del arr
        gc.collect()
        # 检查 `one` 的引用计数是否恢复到操作前的状态
        assert sys.getrefcount(one) == before2
        # 检查 `singleton` 的引用计数是否恢复到操作前的状态
        assert sys.getrefcount(singleton) == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    @pytest.mark.parametrize(
        ['shape', 'index', 'items_changed'],
        [((3,), ([0, 2],), 2),
         ((3, 2), ([0, 2], slice(None)), 4),
         ((3, 2), ([0, 2], [1]), 2),
         ((3,), ([True, False, True]), 2)])
    def test_structured_object_indexing(self, shape, index, items_changed,
                                        dt, pat, count, singleton):
        """Structured object reference counting for advanced indexing."""
        # 使用两个较小的负值（应为单例，但更不容易遇到竞争条件）。
        # 在某些线程环境中使用 0 和 1 可能会失败。如果再次失败，应删除所有显式检查，
        # 依赖于 `pytest-leaks` 的引用计数检查器。
        val0 = -4
        val1 = -5

        # 创建一个填充了 `val0` 的数组
        arr = np.full(shape, val0, dt)

        # 执行垃圾回收，获取设置前 `val0` 和 `val1` 的引用计数
        gc.collect()
        before_val0 = sys.getrefcount(val0)
        before_val1 = sys.getrefcount(val1)
        # 测试获取索引项
        part = arr[index]
        after_val0 = sys.getrefcount(val0)
        assert after_val0 - before_val0 == count * items_changed
        # 删除 `part`
        del part
        # 测试设置索引项
        arr[index] = val1
        gc.collect()
        after_val0 = sys.getrefcount(val0)
        after_val1 = sys.getrefcount(val1)
        assert before_val0 - after_val0 == count * items_changed
        assert after_val1 - before_val1 == count * items_changed

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    # 定义一个测试方法，用于测试结构化对象的取值和重复操作
    def test_structured_object_take_and_repeat(self, dt, pat, count, singleton):
        """Structured object reference counting for specialized functions.
        The older functions such as take and repeat use different code paths
        then item setting (when writing this).
        """
        # 定义一个索引列表
        indices = [0, 1]

        # 创建一个包含多个重复项的数组，并指定数据类型为 dt
        arr = np.array([pat] * 3, dt)
        # 手动执行垃圾回收
        gc.collect()
        # 获取操作前单例对象的引用计数
        before = sys.getrefcount(singleton)
        # 执行 take 操作，并获取结果
        res = arr.take(indices)
        # 获取操作后单例对象的引用计数
        after = sys.getrefcount(singleton)
        # 断言 take 操作后引用计数的变化是否符合预期
        assert after - before == count * 2
        # 对结果数组进行重复操作
        new = res.repeat(10)
        # 再次手动执行垃圾回收
        gc.collect()
        # 获取重复操作后单例对象的引用计数
        after_repeat = sys.getrefcount(singleton)
        # 断言重复操作后引用计数的变化是否符合预期
        assert after_repeat - after == count * 2 * 10
class TestStructuredDtypeSparseFields:
    """Tests subarray fields which contain sparse dtypes so that
    not all memory is used by the dtype work. Such dtype's should
    leave the underlying memory unchanged.
    """

    # 定义一个结构化的 NumPy 数据类型，包含一个名为 'a' 的字段，字段包含稀疏数据类型
    dtype = np.dtype([('a', {'names':['aa', 'ab'], 'formats':['f', 'f'],
                             'offsets':[0, 4]}, (2, 3))])

    # 定义一个稀疏的数据类型，是上述 dtype 中 'a' 字段的一部分，只包含名为 'ab' 的字段
    sparse_dtype = np.dtype([('a', {'names':['ab'], 'formats':['f'],
                                    'offsets':[4]}, (2, 3))])

    def test_sparse_field_assignment(self):
        # 创建一个形状为 (3,) 的零数组，使用定义好的 dtype
        arr = np.zeros(3, self.dtype)
        
        # 将 arr 视图为稀疏数据类型 sparse_dtype
        sparse_arr = arr.view(self.sparse_dtype)

        # 将 sparse_arr 的所有元素设置为 np.float32 类型的最大值
        sparse_arr[...] = np.finfo(np.float32).max
        
        # 断言通过索引 'a' 字段 'aa' 的值为形状为 (3, 2, 3) 的零数组
        assert_array_equal(arr["a"]["aa"], np.zeros((3, 2, 3)))

    def test_sparse_field_assignment_fancy(self):
        # 创建一个形状为 (3,) 的零数组，使用定义好的 dtype
        arr = np.zeros(3, self.dtype)
        
        # 将 arr 视图为稀疏数据类型 sparse_dtype
        sparse_arr = arr.view(self.sparse_dtype)

        # 对 sparse_arr 使用高级索引，将第 [0, 1, 2] 行设置为 np.float32 类型的最大值
        sparse_arr[[0, 1, 2]] = np.finfo(np.float32).max
        
        # 断言通过索引 'a' 字段 'aa' 的值为形状为 (3, 2, 3) 的零数组
        assert_array_equal(arr["a"]["aa"], np.zeros((3, 2, 3)))


class TestMonsterType:
    """Test deeply nested subtypes."""

    def test1(self):
        # 定义一个简单的结构化数据类型 simple1
        simple1 = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                            'titles': ['Red pixel', 'Blue pixel']})
        
        # 定义一个结构化数据类型 a，包含 'yo'、'ye' 和 'yi' 三个字段
        a = np.dtype([('yo', int), ('ye', simple1),
                      ('yi', np.dtype((int, (3, 2))))])
        
        # 定义一个结构化数据类型 b，与 a 相同
        b = np.dtype([('yo', int), ('ye', simple1),
                      ('yi', np.dtype((int, (3, 2))))])
        
        # 断言 a 和 b 结构化数据类型相等
        assert_dtype_equal(a, b)
        
        # 定义一个结构化数据类型 c，与 a 相同，但 'yi' 字段的类型为 a
        c = np.dtype([('yo', int), ('ye', simple1),
                      ('yi', np.dtype((a, (3, 2))))])
        
        # 定义一个结构化数据类型 d，与 c 相同
        d = np.dtype([('yo', int), ('ye', simple1),
                      ('yi', np.dtype((a, (3, 2))))])
        
        # 断言 c 和 d 结构化数据类型相等
        assert_dtype_equal(c, d)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_list_recursion(self):
        # 创建一个空列表 l
        l = list()
        
        # 向列表 l 添加一个包含 'f' 和列表 l 自身的元组
        l.append(('f', l))
        
        # 使用 pytest 的断言检查是否会引发 RecursionError
        with pytest.raises(RecursionError):
            np.dtype(l)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_tuple_recursion(self):
        # 定义变量 d 为 np.int32
        d = np.int32
        
        # 将 d 变量迭代增加到 100000 次，每次迭代 d 变量为一个包含 d 和 (1,) 的元组
        for i in range(100000):
            d = (d, (1,))
        
        # 使用 pytest 的断言检查是否会引发 RecursionError
        with pytest.raises(RecursionError):
            np.dtype(d)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_dict_recursion(self):
        # 定义一个字典 d，包含 'names' 键和值为列表 ['self']，'formats' 键和值为 [None]，'offsets' 键和值为 [0]
        d = dict(names=['self'], formats=[None], offsets=[0])
        
        # 将 d 字典中 'formats' 键的值设为 d 字典自身
        d['formats'][0] = d
        
        # 使用 pytest 的断言检查是否会引发 RecursionError
        with pytest.raises(RecursionError):
            np.dtype(d)


class TestMetadata:
    def test_no_metadata(self):
        # 创建一个简单的数据类型 d，为 np.int32
        d = np.dtype(int)
        
        # 使用 assert_ 函数检查 d 的 metadata 是否为 None
        assert_(d.metadata is None)

    def test_metadata_takes_dict(self):
        # 创建一个简单的数据类型 d，为 np.int32，同时定义 metadata 为 {'datum': 1}
        d = np.dtype(int, metadata={'datum': 1})
        
        # 使用 assert_ 函数检查 d 的 metadata 是否为 {'datum': 1}
        assert_(d.metadata == {'datum': 1})
    # 测试函数：验证 metadata 参数拒绝非字典类型的输入
    def test_metadata_rejects_nondict(self):
        # 断言抛出 TypeError 异常，当 metadata 参数为 'datum' 字符串时
        assert_raises(TypeError, np.dtype, int, metadata='datum')
        # 断言抛出 TypeError 异常，当 metadata 参数为整数 1 时
        assert_raises(TypeError, np.dtype, int, metadata=1)
        # 断言抛出 TypeError 异常，当 metadata 参数为 None 时
        assert_raises(TypeError, np.dtype, int, metadata=None)

    # 测试函数：验证嵌套 metadata 的情况
    def test_nested_metadata(self):
        # 创建一个结构化数据类型 d，包含字段 'a'，其数据类型为 int，同时具有 metadata {'datum': 1}
        d = np.dtype([('a', np.dtype(int, metadata={'datum': 1}))])
        # 断言字段 'a' 的 metadata 等于 {'datum': 1}
        assert_(d['a'].metadata == {'datum': 1})

    # 测试函数：验证基础数据类型的 metadata 被正确复制
    def test_base_metadata_copied(self):
        # 创建一个 void 类型的数据结构，包含两个 i4 类型字段，并且其 metadata 为 {'datum': 1}
        d = np.dtype((np.void, np.dtype('i4,i4', metadata={'datum': 1})))
        # 断言该数据结构的 metadata 等于 {'datum': 1}
        assert_(d.metadata == {'datum': 1})
# 定义一个名为 TestString 的类
class TestString:
    # 测试函数 test_repr_structured
    def test_repr_structured(self):
        # 创建结构化数据类型 dt
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        # 断言函数 repr(dt) 的返回值是否符合预期
        assert_equal(repr(dt),
                     "dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])])")

        # 重新赋值结构化数据类型 dt
        dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']},
                        align=True)
        # 断言函数 repr(dt) 的返回值是否符合预期
        assert_equal(repr(dt),
                    "dtype([(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')], align=True)")

    # 测试函数 test_repr_structured_not_packed
    def test_repr_structured_not_packed(self):
        dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'],
                       'formats': ['<u4', 'u1', 'u1', 'u1'],
                       'offsets': [0, 0, 1, 2],
                       'titles': ['Color', 'Red pixel',
                                  'Green pixel', 'Blue pixel']}, align=True)
        # 断言函数 repr(dt) 的返回值是否符合预期
        assert_equal(repr(dt),
                    "dtype({'names': ['rgba', 'r', 'g', 'b'],"
                    " 'formats': ['<u4', 'u1', 'u1', 'u1'],"
                    " 'offsets': [0, 0, 1, 2],"
                    " 'titles': ['Color', 'Red pixel', "
                                "'Green pixel', 'Blue pixel'],"
                    " 'itemsize': 4}, align=True)")

        # 重新赋值结构化数据类型 dt
        dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel'],
                        'itemsize': 4})
        # 断言函数 repr(dt) 的返回值是否符合预期
        assert_equal(repr(dt),
                    "dtype({'names': ['r', 'b'], "
                    "'formats': ['u1', 'u1'], "
                    "'offsets': [0, 2], "
                    "'titles': ['Red pixel', 'Blue pixel'], "
                    "'itemsize': 4})")

    # 测试函数 test_repr_structured_datetime
    def test_repr_structured_datetime(self):
        dt = np.dtype([('a', '<M8[D]'), ('b', '<m8[us]')])
        # 断言函数 repr(dt) 的返回值是否符合预期
        assert_equal(repr(dt),
                    "dtype([('a', '<M8[D]'), ('b', '<m8[us]')])")

    # 测试函数 test_repr_str_subarray
    def test_repr_str_subarray(self):
        dt = np.dtype(('<i2', (1,)))
        # 断言函数 repr(dt) 的返回值是否符合预期
        assert_equal(repr(dt), "dtype(('<i2', (1,)))")
        # 断言函数 str(dt) 的返回值是否符合预期
        assert_equal(str(dt), "('<i2', (1,))")

    # 测试函数 test_base_dtype_with_object_type
    def test_base_dtype_with_object_type(self):
        # 验证 Issue gh-2798，应该不报错
        np.array(['a'], dtype="O").astype(("O", [("name", "O")]))

    # 测试函数 test_empty_string_to_object
    def test_empty_string_to_object(self):
        # 验证 Pull request #4722
        np.array(["", ""]).astype(object)
    # 定义测试方法，用于测试 np.record 类型的 dtype

    def test_void_subclass_unsized(self):
        # 创建一个无大小限制的 np.record 类型的 dtype
        dt = np.dtype(np.record)
        # 断言返回值的表达式，应为 "dtype('V')"
        assert_equal(repr(dt), "dtype('V')")
        # 断言返回值的字符串表示，应为 '|V0'
        assert_equal(str(dt), '|V0')
        # 断言 dtype 的名称属性，应为 'record'
        assert_equal(dt.name, 'record')

    def test_void_subclass_sized(self):
        # 创建一个大小为 2 的 np.record 类型的 dtype
        dt = np.dtype((np.record, 2))
        # 断言返回值的表达式，应为 "dtype('V2')"
        assert_equal(repr(dt), "dtype('V2')")
        # 断言返回值的字符串表示，应为 '|V2'
        assert_equal(str(dt), '|V2')
        # 断言 dtype 的名称属性，应为 'record16'
        assert_equal(dt.name, 'record16')

    def test_void_subclass_fields(self):
        # 创建一个包含字段的 np.record 类型的 dtype，字段为 ('a', '<u2')
        dt = np.dtype((np.record, [('a', '<u2')]))
        # 断言返回值的表达式，应为 "dtype((numpy.record, [('a', '<u2')]))"
        assert_equal(repr(dt), "dtype((numpy.record, [('a', '<u2')]))")
        # 断言返回值的字符串表示，应为 "(numpy.record, [('a', '<u2')])"
        assert_equal(str(dt), "(numpy.record, [('a', '<u2')])")
        # 断言 dtype 的名称属性，应为 'record16'
        assert_equal(dt.name, 'record16')
class TestDtypeAttributeDeletion:

    def test_dtype_non_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["subdtype", "descr", "str", "name", "base", "shape",
                "isbuiltin", "isnative", "isalignedstruct", "fields",
                "metadata", "hasobject"]

        # 循环遍历不可写属性列表
        for s in attr:
            # 断言删除不可写属性时抛出 AttributeError 异常
            assert_raises(AttributeError, delattr, dt, s)

    def test_dtype_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["names"]
        
        # 循环遍历可写属性列表
        for s in attr:
            # 断言删除可写属性时抛出 AttributeError 异常
            assert_raises(AttributeError, delattr, dt, s)


class TestDtypeAttributes:
    def test_descr_has_trailing_void(self):
        # 见 issue gh-6359
        # 创建一个结构化 dtype
        dtype = np.dtype({
            'names': ['A', 'B'],
            'formats': ['f4', 'f4'],
            'offsets': [0, 8],
            'itemsize': 16})
        # 通过现有描述符创建新的 dtype
        new_dtype = np.dtype(dtype.descr)
        # 断言新 dtype 的 itemsize 属性为 16
        assert_equal(new_dtype.itemsize, 16)

    def test_name_dtype_subclass(self):
        # Ticket #4357
        # 定义一个继承自 np.void 的用户自定义子类
        class user_def_subcls(np.void):
            pass
        # 断言用户定义子类的 dtype 名称为 'user_def_subcls'
        assert_equal(np.dtype(user_def_subcls).name, 'user_def_subcls')

    def test_zero_stride(self):
        # 创建一个 int64 类型的数组
        arr = np.ones(1, dtype="i8")
        # 广播数组，使其大小为 10
        arr = np.broadcast_to(arr, 10)
        # 断言数组的 strides 属性为 (0,)
        assert arr.strides == (0,)
        # 使用 pytest 断言设置非法 dtype 时抛出 ValueError 异常
        with pytest.raises(ValueError):
            arr.dtype = "i1"

class TestDTypeMakeCanonical:
    def check_canonical(self, dtype, canonical):
        """
        Check most properties relevant to "canonical" versions of a dtype,
        which is mainly native byte order for datatypes supporting this.

        The main work is checking structured dtypes with fields, where we
        reproduce most the actual logic used in the C-code.
        """
        # 断言dtype和canonical的类型相同
        assert type(dtype) is type(canonical)

        # 断言可以在等价转换的情况下从dtype转换为canonical，反之亦然
        assert np.can_cast(dtype, canonical, casting="equiv")
        assert np.can_cast(canonical, dtype, casting="equiv")

        # 断言canonical的字节顺序是本机的
        assert canonical.isnative

        # 断言canonical的结果类型与自身相同（没有转换）
        assert np.result_type(canonical) == canonical

        if not dtype.names:
            # 对于非结构化dtype，标志位不会改变
            assert dtype.flags == canonical.flags
            return

        # 对于结构化dtype，必须设置所有需要的API标志位
        assert dtype.flags & 0b10000

        # 断言字段名相同，包括顺序和标题
        assert dtype.fields.keys() == canonical.fields.keys()

        def aligned_offset(offset, alignment):
            # 向上取整对齐偏移量
            return -(-offset // alignment) * alignment

        totalsize = 0
        max_alignment = 1
        for name in dtype.names:
            # 每个字段也必须是canonical的
            new_field_descr = canonical.fields[name][0]
            self.check_canonical(dtype.fields[name][0], new_field_descr)

            # 必须具有继承的对象相关标志位
            expected = 0b11011 & new_field_descr.flags
            assert (canonical.flags & expected) == expected

            if canonical.isalignedstruct:
                totalsize = aligned_offset(totalsize, new_field_descr.alignment)
                max_alignment = max(new_field_descr.alignment, max_alignment)

            # 断言字段的偏移量与totalsize相同
            assert canonical.fields[name][1] == totalsize
            # 如果存在标题，它们必须匹配（否则为空元组）
            assert dtype.fields[name][2:] == canonical.fields[name][2:]

            totalsize += new_field_descr.itemsize

        if canonical.isalignedstruct:
            totalsize = aligned_offset(totalsize, max_alignment)

        # 断言canonical的总大小和对齐方式与计算结果相同
        assert canonical.itemsize == totalsize
        assert canonical.alignment == max_alignment
    # 定义一个测试方法，测试简单的数据类型操作
    def test_simple(self):
        # 创建一个大端字节顺序的32位整数数据类型对象
        dt = np.dtype(">i4")
        # 断言该数据类型对象的结果类型是本机字节顺序
        assert np.result_type(dt).isnative
        # 断言该数据类型对象的结果类型的数字类型与原始类型相同
        assert np.result_type(dt).num == dt.num

        # 带有空间的结构化数据类型:
        struct_dt = np.dtype(">i4,<i1,i8,V3")[["f0", "f2"]]
        # 获取结构化数据类型的规范结果类型
        canonical = np.result_type(struct_dt)
        # 断言规范结果类型的字节大小等于子类型的总和
        assert canonical.itemsize == 4+8
        # 断言规范结果类型是本机字节顺序
        assert canonical.isnative

        # 带有对齐的结构化数据类型:
        struct_dt = np.dtype(">i1,<i4,i8,V3", align=True)[["f0", "f2"]]
        # 获取结构化数据类型的规范结果类型
        canonical = np.result_type(struct_dt)
        # 断言规范结果类型是对齐的结构化数据类型
        assert canonical.isalignedstruct
        # 断言规范结果类型的字节大小等于i8数据类型的对齐数加上8
        assert canonical.itemsize == np.dtype("i8").alignment + 8
        # 断言规范结果类型是本机字节顺序
        assert canonical.isnative

    # 测试对象标志未继承的情况
    def test_object_flag_not_inherited(self):
        # 创建一个包含整数、对象和整数的数组
        arr = np.ones(3, "i,O,i")[["f0", "f2"]]
        # 断言数组的数据类型具有对象类型
        assert arr.dtype.hasobject
        # 获取数组数据类型的规范结果类型
        canonical_dt = np.result_type(arr.dtype)
        # 断言规范结果类型不具有对象类型
        assert not canonical_dt.hasobject

    # 使用假设测试生成规范结果类型的方法
    @pytest.mark.slow
    @hypothesis.given(dtype=hynp.nested_dtypes())
    def test_make_canonical_hypothesis(self, dtype):
        # 获取数据类型的规范结果类型
        canonical = np.result_type(dtype)
        # 检查生成的规范结果类型是否正确
        self.check_canonical(dtype, canonical)
        # 使用两个相同的数据类型参数调用result_type应该得到相同的结果
        two_arg_result = np.result_type(dtype, dtype)
        # 断言两个规范结果类型可以进行无类型转换的转换
        assert np.can_cast(two_arg_result, canonical, casting="no")

    # 使用假设测试数组数据类型生成规范结果类型的方法
    @pytest.mark.slow
    @hypothesis.given(
            dtype=hypothesis.extra.numpy.array_dtypes(
                subtype_strategy=hypothesis.extra.numpy.array_dtypes(),
                min_size=5, max_size=10, allow_subarrays=True))
    # 定义一个测试方法，用于结构化数据类型的测试
    def test_structured(self, dtype):
        # 从给定的 dtype 中随机选择 4 个字段，这会在 dtype 中留下空白空间（因为此处不做规范化处理）
        field_subset = random.sample(dtype.names, k=4)
        # 从 dtype 中提取带有空白空间的子 dtype
        dtype_with_empty_space = dtype[field_subset]
        # 断言带有空白空间的 dtype 的字节大小与原始 dtype 的相同
        assert dtype_with_empty_space.itemsize == dtype.itemsize
        # 获取带有空白空间的 dtype 的规范化结果
        canonicalized = np.result_type(dtype_with_empty_space)
        # 调用方法检查带有空白空间的 dtype 是否与规范化结果一致
        self.check_canonical(dtype_with_empty_space, canonicalized)
        # 使用两个相同的参数进行类型提升应始终给出相同的结果：
        two_arg_result = np.promote_types(
                dtype_with_empty_space, dtype_with_empty_space)
        # 断言两个结果可以互相转换，且不进行类型转换
        assert np.can_cast(two_arg_result, canonicalized, casting="no")

        # 确保我们还检查对齐结构（检查相反情况，以防假设增加对 `align` 的支持）。然后重复测试：
        # 根据 dtype 的描述和对齐情况创建一个新的 dtype
        dtype_aligned = np.dtype(dtype.descr, align=not dtype.isalignedstruct)
        # 从新的对齐 dtype 中提取带有空白空间的子 dtype
        dtype_with_empty_space = dtype_aligned[field_subset]
        # 断言带有空白空间的 dtype 的字节大小与新的对齐 dtype 的相同
        assert dtype_with_empty_space.itemsize == dtype_aligned.itemsize
        # 获取带有空白空间的 dtype 的规范化结果
        canonicalized = np.result_type(dtype_with_empty_space)
        # 调用方法检查带有空白空间的 dtype 是否与规范化结果一致
        self.check_canonical(dtype_with_empty_space, canonicalized)
        # 使用两个相同的参数进行类型提升应始终给出相同的结果：
        two_arg_result = np.promote_types(
            dtype_with_empty_space, dtype_with_empty_space)
        # 断言两个结果可以互相转换，且不进行类型转换
        assert np.can_cast(two_arg_result, canonicalized, casting="no")
class TestPickling:

    def check_pickling(self, dtype):
        # 遍历所有协议版本进行序列化和反序列化测试
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # 使用指定协议版本将 dtype 序列化为字节流
            buf = pickle.dumps(dtype, proto)
            # 检查序列化后的字节流中不包含 "_DType_reconstruct" 字符串
            assert b"_DType_reconstruct" not in buf
            # 检查序列化后的字节流中包含 "dtype" 字符串
            assert b"dtype" in buf
            # 反序列化得到的对象应该与原始 dtype 相等
            pickled = pickle.loads(buf)
            assert_equal(pickled, dtype)
            # 检查反序列化后的对象的描述符与原始 dtype 的描述符相等
            assert_equal(pickled.descr, dtype.descr)
            # 如果原始 dtype 有元数据，则检查反序列化后的对象的元数据也相等
            if dtype.metadata is not None:
                assert_equal(pickled.metadata, dtype.metadata)
            # 检查重建的 dtype 是否能正常使用
            x = np.zeros(3, dtype=dtype)
            y = np.zeros(3, dtype=pickled)
            assert_equal(x, y)
            assert_equal(x[0], y[0])

    @pytest.mark.parametrize('t', [int, float, complex, np.int32, str, object,
                                   bool])
    def test_builtin(self, t):
        # 对于内置数据类型，进行序列化和反序列化测试
        self.check_pickling(np.dtype(t))

    def test_structured(self):
        # 测试结构化 dtype 的序列化和反序列化
        dt = np.dtype(([('a', '>f4', (2, 1)), ('b', '<f8', (1, 3))], (2, 2)))
        self.check_pickling(dt)

    def test_structured_aligned(self):
        # 测试结构化 dtype 的序列化和反序列化（对齐）
        dt = np.dtype('i4, i1', align=True)
        self.check_pickling(dt)

    def test_structured_unaligned(self):
        # 测试结构化 dtype 的序列化和反序列化（不对齐）
        dt = np.dtype('i4, i1', align=False)
        self.check_pickling(dt)

    def test_structured_padded(self):
        # 测试带填充的结构化 dtype 的序列化和反序列化
        dt = np.dtype({
            'names': ['A', 'B'],
            'formats': ['f4', 'f4'],
            'offsets': [0, 8],
            'itemsize': 16})
        self.check_pickling(dt)

    def test_structured_titles(self):
        # 测试带标题的结构化 dtype 的序列化和反序列化
        dt = np.dtype({'names': ['r', 'b'],
                       'formats': ['u1', 'u1'],
                       'titles': ['Red pixel', 'Blue pixel']})
        self.check_pickling(dt)

    @pytest.mark.parametrize('base', ['m8', 'M8'])
    @pytest.mark.parametrize('unit', ['', 'Y', 'M', 'W', 'D', 'h', 'm', 's',
                                      'ms', 'us', 'ns', 'ps', 'fs', 'as'])
    def test_datetime(self, base, unit):
        # 测试日期时间 dtype 的序列化和反序列化
        dt = np.dtype('%s[%s]' % (base, unit) if unit else base)
        self.check_pickling(dt)
        if unit:
            dt = np.dtype('%s[7%s]' % (base, unit))
            self.check_pickling(dt)

    def test_metadata(self):
        # 测试包含元数据的 dtype 的序列化和反序列化
        dt = np.dtype(int, metadata={'datum': 1})
        self.check_pickling(dt)

    @pytest.mark.parametrize("DType",
        [type(np.dtype(t)) for t in np.typecodes['All']] +
        [type(np.dtype(rational)), np.dtype])
    def test_pickle_dtype_class(self, DType):
        # 测试 dtype 类的序列化和反序列化
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_DType = pickle.loads(pickle.dumps(DType, proto))
            assert roundtrip_DType is DType

    @pytest.mark.parametrize("dt",
        [np.dtype(t) for t in np.typecodes['All']] +
        [np.dtype(rational)])
    def test_pickle_dtype_instance(self, dt):
        # 测试 dtype 实例的序列化和反序列化
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_dt = pickle.loads(pickle.dumps(dt, proto))
            assert roundtrip_dt == dt
    def test_pickle_dtype(self, dt):
        # 测试数据类型在通过 pickle 进行序列化和反序列化后能否保持一致，同时保持哈希值不变
        # 记录序列化前的哈希值
        pre_pickle_hash = hash(dt)
        # 遍历所有 pickle 协议版本
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # 使用指定协议版本进行序列化和反序列化
            roundtrip_dt = pickle.loads(pickle.dumps(dt, proto))
            # 断言反序列化后的对象与原始对象相等
            assert roundtrip_dt == dt
            # 断言反序列化后的对象的哈希值与原始对象的哈希值相等
            assert hash(dt) == pre_pickle_hash
# 定义一个测试类，用于测试复杂数据类型的提升（promotion）逻辑
class TestPromotion:
    """Test cases related to more complex DType promotions.  Further promotion
    tests are defined in `test_numeric.py`
    """

    # 使用 np._no_nep50_warning() 装饰器，标记不显示 NEP 50 的警告信息
    @np._no_nep50_warning()
    # 使用 pytest.mark.parametrize 装饰器，参数化测试方法的输入参数
    @pytest.mark.parametrize(["other", "expected", "expected_weak"],
            [(2**16-1, np.complex64, None),
             (2**32-1, np.complex128, np.complex64),
             (np.float16(2), np.complex64, None),
             (np.float32(2), np.complex64, None),
             (np.longdouble(2), np.complex64, np.clongdouble),
             # Base of the double value to sidestep any rounding issues:
             (np.longdouble(np.nextafter(1.7e308, 0.)),
                  np.complex128, np.clongdouble),
             # Additionally use "nextafter" so the cast can't round down:
             (np.longdouble(np.nextafter(1.7e308, np.inf)),
                  np.clongdouble, None),
             # repeat for complex scalars:
             (np.complex64(2), np.complex64, None),
             (np.clongdouble(2), np.complex64, np.clongdouble),
             # Base of the double value to sidestep any rounding issues:
             (np.clongdouble(np.nextafter(1.7e308, 0.) * 1j),
                  np.complex128, np.clongdouble),
             # Additionally use "nextafter" so the cast can't round down:
             (np.clongdouble(np.nextafter(1.7e308, np.inf)),
                  np.clongdouble, None),
             ])
    # 定义测试方法，测试根据其他值进行复杂类型推断的逻辑
    def test_complex_other_value_based(self,
            weak_promotion, other, expected, expected_weak):
        # 如果存在弱类型推断并且期望的弱类型不为空，则更新期望值
        if weak_promotion and expected_weak is not None:
            expected = expected_weak

        # 定义最小的复杂类型为 np.complex64
        min_complex = np.dtype(np.complex64)

        # 进行类型推断，比较结果是否符合预期
        res = np.result_type(other, min_complex)
        assert res == expected
        # 检查简单的 ufunc 调用是否使用相同的逻辑：
        res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
        assert res == expected

    # 使用 pytest.mark.parametrize 装饰器，参数化测试方法的输入参数
    @pytest.mark.parametrize(["other", "expected"],
                 [(np.bool, np.complex128),
                  (np.int64, np.complex128),
                  (np.float16, np.complex64),
                  (np.float32, np.complex64),
                  (np.float64, np.complex128),
                  (np.longdouble, np.clongdouble),
                  (np.complex64, np.complex64),
                  (np.complex128, np.complex128),
                  (np.clongdouble, np.clongdouble),
                  ])
    # 定义测试方法，测试根据标量复杂值进行推断的逻辑
    def test_complex_scalar_value_based(self, other, expected):
        # 定义复杂标量为虚数单位 1j
        complex_scalar = 1j

        # 进行类型推断，比较结果是否符合预期
        res = np.result_type(other, complex_scalar)
        assert res == expected
        # 检查简单的 ufunc 调用是否使用相同的逻辑：
        res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
        assert res == expected
    # 定义一个测试方法，用于测试复杂的数值类型提升到有理数的情况
    def test_complex_pyscalar_promote_rational(self):
        # 使用 pytest.raises 检查是否会抛出 TypeError 异常
        # 匹配错误信息，确保包含指定的错误信息提示
        with pytest.raises(TypeError,
                match=r".* no common DType exists for the given inputs"):
            # 调用 np.result_type 函数，测试 1j 和 rational 类型的组合
            np.result_type(1j, rational)

        with pytest.raises(TypeError,
                match=r".* no common DType exists for the given inputs"):
            # 调用 np.result_type 函数，测试 1j 和 rational(1, 2) 类型的组合
            np.result_type(1j, rational(1, 2))

    # 使用 pytest 的参数化标记，定义一个测试方法，测试 Python 整数的提升
    @pytest.mark.parametrize("val", [2, 2**32, 2**63, 2**64, 2*100])
    def test_python_integer_promotion(self, val):
        # 如果只传递标量（主要是 Python 标量），根据 NEP 50 规定，会得到默认整数类型
        expected_dtype = np.dtype(int)  # 默认整数类型
        # 断言调用 np.result_type 函数后返回的类型与期望的默认整数类型相同
        assert np.result_type(val, 0) == expected_dtype
        # 根据 NEP 50 规定，NumPy 标量将胜出：
        # 断言调用 np.result_type 函数后返回的类型与 np.int8 相同
        assert np.result_type(val, np.int8(0)) == np.int8

    # 使用 pytest 的参数化标记，定义一个测试方法，测试浮点数和整数提升到有理数的情况
    @pytest.mark.parametrize(["other", "expected"],
            [(1, rational), (1., np.float64)])
    @np._no_nep50_warning()
    def test_float_int_pyscalar_promote_rational(
            self, weak_promotion, other, expected):
        # 注意，有理数在与 float64 提升时有点尴尬，会与默认整数类型提升，但不与 float16 或 uint8/int8 提升（看起来不一致）
        # 新的提升修复了这个问题（部分修复？）
        if not weak_promotion and type(other) == float:
            # 如果不是弱提升且 other 是 float 类型，则检查在旧路径中是否会抛出 TypeError 异常
            with pytest.raises(TypeError,
                    match=r".* do not have a common DType"):
                np.result_type(other, rational)
        else:
            # 否则，断言调用 np.result_type 函数后返回的类型与期望的类型相同
            assert np.result_type(other, rational) == expected

        # 断言调用 np.result_type 函数后返回的类型与期望的类型相同
        assert np.result_type(other, rational(1, 2)) == expected

    # 使用 pytest 的参数化标记，定义一个测试方法，测试多种数据类型组合的提升
    @pytest.mark.parametrize(["dtypes", "expected"], [
             # 这些提升不是可交换的：
             ([np.uint16, np.int16, np.float16], np.float32),
             ([np.uint16, np.int8, np.float16], np.float32),
             ([np.uint8, np.int16, np.float16], np.float32),
             # 以下提升不是模糊的，但涵盖了抽象提升的代码路径（没有特定逻辑被测试）
             ([1, 1, np.float64], np.float64),
             ([1, 1., np.complex128], np.complex128),
             ([1, 1j, np.float64], np.complex128),
             ([1., 1., np.int64], np.float64),
             ([1., 1j, np.float64], np.complex128),
             ([1j, 1j, np.float64], np.complex128),
             ([1, True, np.bool], np.int_),
            ])
    # 定义一个测试方法，验证排列顺序对结果无影响
    def test_permutations_do_not_influence_result(self, dtypes, expected):
        # 对于给定的数据类型的排列进行迭代测试
        for perm in permutations(dtypes):
            # 断言：使用 numpy 计算结果类型，检查是否等于期望的结果类型
            assert np.result_type(*perm) == expected
def test_rational_dtype():
    # test for bug gh-5719
    # 创建一个包含单个元素 1111 的 NumPy 数组，并使用自定义的 rational 数据类型
    a = np.array([1111], dtype=rational).astype
    # 断言尝试将该数组转换为 'int8' 类型时会引发 OverflowError 异常
    assert_raises(OverflowError, a, 'int8')

    # test that dtype detection finds user-defined types
    # 创建一个 rational 类型的对象 x
    x = rational(1)
    # 断言创建包含两个 x 元素的 NumPy 数组的数据类型为 rational
    assert_equal(np.array([x,x]).dtype, np.dtype(rational))


def test_dtypes_are_true():
    # test for gh-6294
    # 断言 'f8' 数据类型的布尔值为真
    assert bool(np.dtype('f8'))
    # 断言 'i8' 数据类型的布尔值为真
    assert bool(np.dtype('i8'))
    # 断言包含两个字段 ('a', 'i8') 和 ('b', 'f4') 的结构化数据类型的布尔值为真
    assert bool(np.dtype([('a', 'i8'), ('b', 'f4')]))


def test_invalid_dtype_string():
    # test for gh-10440
    # 断言尝试使用无效的 dtype 字符串 'f8,i8,[f8,i8]' 会引发 TypeError 异常
    assert_raises(TypeError, np.dtype, 'f8,i8,[f8,i8]')
    # 断言尝试使用无效的 dtype 字符串 'Fl\xfcgel' 会引发 TypeError 异常
    assert_raises(TypeError, np.dtype, 'Fl\xfcgel')


def test_keyword_argument():
    # test for https://github.com/numpy/numpy/pull/16574#issuecomment-642660971
    # 断言传递 dtype=np.float64 参数时返回的数据类型为 np.float64
    assert np.dtype(dtype=np.float64) == np.dtype(np.float64)


class TestFromDTypeAttribute:
    def test_simple(self):
        class dt:
            dtype = np.dtype("f8")

        # 断言使用类 dt 的 dtype 属性创建的数据类型为 np.float64
        assert np.dtype(dt) == np.float64
        # 断言使用 dt 类的实例创建的数据类型为 np.float64
        assert np.dtype(dt()) == np.float64

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_recursion(self):
        class dt:
            pass

        dt.dtype = dt
        # 断言尝试创建循环引用的数据类型 dt 会引发 RecursionError 异常
        with pytest.raises(RecursionError):
            np.dtype(dt)

        dt_instance = dt()
        dt_instance.dtype = dt
        # 断言尝试创建循环引用的数据类型 dt_instance 会引发 RecursionError 异常
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)

    def test_void_subtype(self):
        class dt(np.void):
            # This code path is fully untested before, so it is unclear
            # what this should be useful for. Note that if np.void is used
            # numpy will think we are deallocating a base type [1.17, 2019-02].
            dtype = np.dtype("f,f")

        # 创建 void 子类型 dt 的数据类型
        np.dtype(dt)
        np.dtype(dt(1))

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_void_subtype_recursion(self):
        class vdt(np.void):
            pass

        vdt.dtype = vdt

        # 断言尝试创建循环引用的 void 子类型 vdt 会引发 RecursionError 异常
        with pytest.raises(RecursionError):
            np.dtype(vdt)

        with pytest.raises(RecursionError):
            np.dtype(vdt(1))


class TestDTypeClasses:
    @pytest.mark.parametrize("dtype", list(np.typecodes['All']) + [rational])
    # 测试基本数据类型的子类属性
    def test_basic_dtypes_subclass_properties(self, dtype):
        # 注意：除了 isinstance 和 type 检查外，这些属性当前被视为私有且可能会改变。
        
        # 将输入的 dtype 转换为 numpy 的 dtype 对象
        dtype = np.dtype(dtype)
        
        # 断言 dtype 是 np.dtype 的实例
        assert isinstance(dtype, np.dtype)
        
        # 断言 dtype 的类型不是 np.dtype 类型本身
        assert type(dtype) is not np.dtype
        
        # 如果 dtype 的类型名称不是 "rational"
        if dtype.type.__name__ != "rational":
            # 获取 dtype 的类型名称，并去除 "dtype" 后缀
            dt_name = type(dtype).__name__.lower().removesuffix("dtype")
            
            # 如果 dt_name 是 "uint" 或 "int"
            if dt_name == "uint" or dt_name == "int":
                # 在标量名称上添加 'c'，因为 "int" 是 Python 中的 int，而那是长整型...
                dt_name += "c"
            
            # 获取 dtype 的类型名称
            sc_name = dtype.type.__name__
            
            # 断言 dt_name 和 sc_name（去除下划线后）相同
            assert dt_name == sc_name.strip("_")
            
            # 断言 dtype 的类型模块是 "numpy.dtypes"
            assert type(dtype).__module__ == "numpy.dtypes"
            
            # 断言 numpy.dtypes 下的类型名称与 dtype 的类型相同
            assert getattr(numpy.dtypes, type(dtype).__name__) is type(dtype)
        
        else:
            # 如果 dtype 的类型名称是 "dtype[rational]"
            assert type(dtype).__name__ == "dtype[rational]"
            
            # 断言 dtype 的类型模块是 "numpy"
            assert type(dtype).__module__ == "numpy"
        
        # 断言 dtype 的类型不是抽象类型
        assert not type(dtype)._abstract
        
        # 对于灵活的 dtype 和 datetime/timedelta，它们有额外的参数
        # 这些参数不仅仅是存储信息，创建 dtype 时需要考虑这些参数
        parametric = (np.void, np.str_, np.bytes_, np.datetime64, np.timedelta64)
        
        # 如果 dtype 的类型不在 parametric 中
        if dtype.type not in parametric:
            # 断言 dtype 不是参数化的类型
            assert not type(dtype)._parametric
            
            # 断言 type(dtype)() 是 dtype 自身
            assert type(dtype)() is dtype
        
        else:
            # 如果 dtype 是参数化的类型
            assert type(dtype)._parametric
            
            # 使用 assert_raises 确保创建 type(dtype)() 会抛出 TypeError 异常
            with assert_raises(TypeError):
                type(dtype)()

    # 测试 dtype 的超类
    def test_dtype_superclass(self):
        # 断言 np.dtype 不是 type 类型本身
        assert type(np.dtype) is not type
        
        # 断言 np.dtype 是 type 类的实例
        assert isinstance(np.dtype, type)
        
        # 断言 np.dtype 的类型名称是 "_DTypeMeta"
        assert type(np.dtype).__name__ == "_DTypeMeta"
        
        # 断言 np.dtype 的类型模块是 "numpy"
        assert type(np.dtype).__module__ == "numpy"
        
        # 断言 np.dtype 是抽象类型
        assert np.dtype._abstract

    # 测试是否为数值类型
    def test_is_numeric(self):
        # 获取所有类型码和数值类型码的集合
        all_codes = set(np.typecodes['All'])
        numeric_codes = set(np.typecodes['AllInteger'] +
                            np.typecodes['AllFloat'] + '?')
        
        # 非数值类型码的集合
        non_numeric_codes = all_codes - numeric_codes
        
        # 对于数值类型码中的每个类型码
        for code in numeric_codes:
            # 断言 type(np.dtype(code))._is_numeric 是 True
            assert type(np.dtype(code))._is_numeric
        
        # 对于非数值类型码中的每个类型码
        for code in non_numeric_codes:
            # 断言 type(np.dtype(code))._is_numeric 是 False
            assert not type(np.dtype(code))._is_numeric

    # 测试整数别名名称
    @pytest.mark.parametrize("int_", ["UInt", "Int"])
    @pytest.mark.parametrize("size", [8, 16, 32, 64])
    def test_integer_alias_names(self, int_, size):
        # 获取 numpy.dtypes 中的 DType 类型
        DType = getattr(numpy.dtypes, f"{int_}{size}DType")
        
        # 获取 numpy 中的相应的数据类型
        sctype = getattr(numpy, f"{int_.lower()}{size}")
        
        # 断言 DType 的类型是 sctype
        assert DType.type is sctype
        
        # 断言 DType 的名称去除后缀 "dtype" 与 sctype 的名称相同
        assert DType.__name__.lower().removesuffix("dtype") == sctype.__name__

    # 测试浮点数别名名称
    @pytest.mark.parametrize("name",
            ["Half", "Float", "Double", "CFloat", "CDouble"])
    def test_float_alias_names(self, name):
        # 使用 pytest.raises 确保获取不存在的浮点数别名名称会引发 AttributeError 异常
        with pytest.raises(AttributeError):
            getattr(numpy.dtypes, name + "DType") is numpy.dtypes.Float16DType
class TestFromCTypes:

    @staticmethod
    def check(ctype, dtype):
        # 将 dtype 转换为 NumPy 数据类型对象
        dtype = np.dtype(dtype)
        # 断言 ctype 的 NumPy 数据类型与 dtype 相同
        assert np.dtype(ctype) == dtype
        # 断言 ctype 的实例化对象的 NumPy 数据类型与 dtype 相同
        assert np.dtype(ctype()) == dtype
        # 断言 ctype 的大小与 dtype 的字节大小相同
        assert ctypes.sizeof(ctype) == dtype.itemsize

    def test_array(self):
        # 定义一个 ctypes 类型 c_uint8
        c8 = ctypes.c_uint8
        # 调用 check 方法，验证 3 * c8 的类型和形状是否符合预期
        self.check(3 * c8, (np.uint8, (3,)))
        # 调用 check 方法，验证 1 * c8 的类型和形状是否符合预期
        self.check(1 * c8, (np.uint8, (1,)))
        # 调用 check 方法，验证 0 * c8 的类型和形状是否符合预期
        self.check(0 * c8, (np.uint8, (0,)))
        # 调用 check 方法，验证 1 * (3 * c8) 的类型和形状是否符合预期
        self.check(1 * (3 * c8), ((np.uint8, (3,)), (1,)))
        # 调用 check 方法，验证 3 * (1 * c8) 的类型和形状是否符合预期
        self.check(3 * (1 * c8), ((np.uint8, (1,)), (3,)))

    def test_padded_structure(self):
        # 定义一个包含字节对齐的结构体 PaddedStruct
        class PaddedStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        # 定义预期的 NumPy 数据类型对象 expected
        expected = np.dtype([
            ('a', np.uint8),
            ('b', np.uint16)
        ], align=True)
        # 调用 check 方法，验证 PaddedStruct 的类型和形状是否符合预期
        self.check(PaddedStruct, expected)

    def test_bit_fields(self):
        # 定义一个包含位域的结构体 BitfieldStruct
        class BitfieldStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_uint8, 7),
                ('b', ctypes.c_uint8, 1)
            ]
        # 断言抛出 TypeError 异常，因为无法将 BitfieldStruct 转换为 NumPy 数据类型
        assert_raises(TypeError, np.dtype, BitfieldStruct)
        # 断言抛出 TypeError 异常，因为无法将 BitfieldStruct 的实例化对象转换为 NumPy 数据类型
        assert_raises(TypeError, np.dtype, BitfieldStruct())

    def test_pointer(self):
        # 定义一个指向 ctypes.c_uint8 类型的指针 p_uint8
        p_uint8 = ctypes.POINTER(ctypes.c_uint8)
        # 断言抛出 TypeError 异常，因为无法将 p_uint8 转换为 NumPy 数据类型
        assert_raises(TypeError, np.dtype, p_uint8)

    def test_size_t(self):
        # 断言 np.uintp 的 NumPy 数据类型等同于 "N"
        assert np.dtype(np.uintp) is np.dtype("N")
        # 调用 check 方法，验证 ctypes.c_size_t 和 np.uintp 的类型和形状是否符合预期
        self.check(ctypes.c_size_t, np.uintp)

    def test_void_pointer(self):
        # 调用 check 方法，验证 ctypes.c_void_p 和 "P" 的类型和形状是否符合预期
        self.check(ctypes.c_void_p, "P")

    def test_union(self):
        # 定义一个联合体 Union
        class Union(ctypes.Union):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
            ]
        # 定义预期的 NumPy 数据类型对象 expected
        expected = np.dtype(dict(
            names=['a', 'b'],
            formats=[np.uint8, np.uint16],
            offsets=[0, 0],
            itemsize=2
        ))
        # 调用 check 方法，验证 Union 的类型和形状是否符合预期
        self.check(Union, expected)

    def test_union_with_struct_packed(self):
        # 定义一个紧凑结构体 Struct
        class Struct(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]

        # 定义一个包含结构体的联合体 Union
        class Union(ctypes.Union):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint32),
                ('d', Struct),
            ]
        # 定义预期的 NumPy 数据类型对象 expected
        expected = np.dtype(dict(
            names=['a', 'b', 'c', 'd'],
            formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]],
            offsets=[0, 0, 0, 0],
            itemsize=ctypes.sizeof(Union)
        ))
        # 调用 check 方法，验证 Union 的类型和形状是否符合预期
        self.check(Union, expected)
    # 定义一个测试方法，用于测试 Union 类型的打包方式
    def test_union_packed(self):
        # 定义一个嵌套的结构体类 Struct，包含两个字段：一个是 8 位无符号整数，一个是 32 位无符号整数
        class Struct(ctypes.Structure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1  # 设置结构体的打包对齐方式为 1 字节

        # 定义一个 Union 类，包含四个字段：一个是 8 位无符号整数，一个是 16 位无符号整数，一个是 32 位无符号整数，一个是结构体 Struct
        class Union(ctypes.Union):
            _pack_ = 1  # 设置联合体的打包对齐方式为 1 字节
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint32),
                ('d', Struct),
            ]

        # 构造一个预期的 NumPy dtype，包含字段名、格式、偏移量、结构体大小等信息
        expected = np.dtype(dict(
            names=['a', 'b', 'c', 'd'],
            formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]],
            offsets=[0, 0, 0, 0],
            itemsize=ctypes.sizeof(Union)
        ))

        # 调用自定义的检查方法，验证 Union 类型和预期的 dtype 是否一致
        self.check(Union, expected)

    # 定义一个测试方法，用于测试带打包方式的普通结构体 PackedStructure
    def test_packed_structure(self):
        # 定义一个带打包方式的结构体 PackedStructure，包含一个 8 位无符号整数和一个 16 位无符号整数
        class PackedStructure(ctypes.Structure):
            _pack_ = 1  # 设置结构体的打包对齐方式为 1 字节
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]

        # 构造一个预期的 NumPy dtype，包含字段名和对应的数据类型
        expected = np.dtype([
            ('a', np.uint8),
            ('b', np.uint16)
        ])

        # 调用自定义的检查方法，验证 PackedStructure 类型和预期的 dtype 是否一致
        self.check(PackedStructure, expected)

    # 定义一个测试方法，用于测试更大的带打包方式的结构体 PackedStructure
    def test_large_packed_structure(self):
        # 定义一个更大的带打包方式的结构体 PackedStructure，包含多个字段：8 位、16 位、32 位无符号整数
        class PackedStructure(ctypes.Structure):
            _pack_ = 2  # 设置结构体的打包对齐方式为 2 字节
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint8),
                ('d', ctypes.c_uint16),
                ('e', ctypes.c_uint32),
                ('f', ctypes.c_uint32),
                ('g', ctypes.c_uint8)
            ]

        # 构造一个预期的 NumPy dtype，包含字段名、数据类型、偏移量、结构体大小等信息
        expected = np.dtype(dict(
            formats=[np.uint8, np.uint16, np.uint8, np.uint16, np.uint32, np.uint32, np.uint8],
            offsets=[0, 2, 4, 6, 8, 12, 16],
            names=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            itemsize=18
        ))

        # 调用自定义的检查方法，验证 PackedStructure 类型和预期的 dtype 是否一致
        self.check(PackedStructure, expected)

    # 定义一个测试方法，用于测试带打包方式的大端字节序结构体 BigEndStruct
    def test_big_endian_structure_packed(self):
        # 定义一个带打包方式的大端字节序结构体 BigEndStruct，包含一个 8 位无符号整数和一个 32 位无符号整数
        class BigEndStruct(ctypes.BigEndianStructure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1  # 设置结构体的打包对齐方式为 1 字节

        # 构造一个预期的 NumPy dtype，包含字段名和对应的数据类型
        expected = np.dtype([('one', 'u1'), ('two', '>u4')])

        # 调用自定义的检查方法，验证 BigEndStruct 类型和预期的 dtype 是否一致
        self.check(BigEndStruct, expected)

    # 定义一个测试方法，用于测试带打包方式的小端字节序结构体 LittleEndStruct
    def test_little_endian_structure_packed(self):
        # 定义一个带打包方式的小端字节序结构体 LittleEndStruct，包含一个 8 位无符号整数和一个 32 位无符号整数
        class LittleEndStruct(ctypes.LittleEndianStructure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1  # 设置结构体的打包对齐方式为 1 字节

        # 构造一个预期的 NumPy dtype，包含字段名和对应的数据类型
        expected = np.dtype([('one', 'u1'), ('two', '<u4')])

        # 调用自定义的检查方法，验证 LittleEndStruct 类型和预期的 dtype 是否一致
        self.check(LittleEndStruct, expected)

    # 定义一个测试方法，用于测试不带打包方式的小端字节序结构体 PaddedStruct
    def test_little_endian_structure(self):
        # 定义一个不带打包方式的小端字节序结构体 PaddedStruct，包含一个 8 位无符号整数和一个 16 位无符号整数
        class PaddedStruct(ctypes.LittleEndianStructure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]

        # 构造一个预期的 NumPy dtype，包含字段名、数据类型和是否需要对齐等信息
        expected = np.dtype([
            ('a', '<B'),
            ('b', '<H')
        ], align=True)

        # 调用自定义的检查方法，验证 PaddedStruct 类型和预期的 dtype 是否一致
        self.check(PaddedStruct, expected)
    # 定义一个测试类方法，用于测试大端结构的情况
    def test_big_endian_structure(self):
        # 定义一个继承自 ctypes.BigEndianStructure 的内部类 PaddedStruct
        class PaddedStruct(ctypes.BigEndianStructure):
            # 定义结构体的字段
            _fields_ = [
                ('a', ctypes.c_uint8),   # 字段 a，无符号 8 位整数
                ('b', ctypes.c_uint16)   # 字段 b，无符号 16 位整数
            ]
        # 创建一个预期的 NumPy 数据类型，描述了字段 a 和 b 的大端字节顺序
        expected = np.dtype([
            ('a', '>B'),   # 字段 a，大端无符号 8 位整数
            ('b', '>H')    # 字段 b，大端无符号 16 位整数
        ], align=True)
        # 调用 self.check 方法，验证 PaddedStruct 类型与预期的数据类型 expected 是否匹配
        self.check(PaddedStruct, expected)

    # 定义一个测试方法，用于测试简单的大小端类型
    def test_simple_endian_types(self):
        # 验证 ctypes.c_uint16 的小端数据类型与 np.dtype('<u2') 是否匹配
        self.check(ctypes.c_uint16.__ctype_le__, np.dtype('<u2'))
        # 验证 ctypes.c_uint16 的大端数据类型与 np.dtype('>u2') 是否匹配
        self.check(ctypes.c_uint16.__ctype_be__, np.dtype('>u2'))
        # 验证 ctypes.c_uint8 的小端数据类型与 np.dtype('u1') 是否匹配
        self.check(ctypes.c_uint8.__ctype_le__, np.dtype('u1'))
        # 验证 ctypes.c_uint8 的大端数据类型与 np.dtype('u1') 是否匹配
        self.check(ctypes.c_uint8.__ctype_be__, np.dtype('u1'))

    # 使用 itertools 的 permutations 函数生成所有类型码的两两排列组合
    all_types = set(np.typecodes['All'])
    all_pairs = permutations(all_types, 2)

    # 使用 pytest 的 parametrize 装饰器，对 all_pairs 中的每对元素执行测试
    @pytest.mark.parametrize("pair", all_pairs)
    def test_pairs(self, pair):
        """
        Check that np.dtype('x,y') matches [np.dtype('x'), np.dtype('y')]
        Example: np.dtype('d,I') -> dtype([('f0', '<f8'), ('f1', '<u4')])
        """
        # gh-5645: check that np.dtype('i,L') can be used
        # 根据 pair 的两个类型码，创建一个复合数据类型 np.dtype('x,y')
        pair_type = np.dtype('{},{}'.format(*pair))
        # 创建一个预期的复合数据类型，其中包含两个字段 f0 和 f1，分别对应 pair 的两个类型
        expected = np.dtype([('f0', pair[0]), ('f1', pair[1])])
        # 断言 pair_type 是否与预期的 expected 相等
        assert_equal(pair_type, expected)
class TestUserDType:
    @pytest.mark.leaks_references(reason="dynamically creates custom dtype.")
    def test_custom_structured_dtype(self):
        # 定义一个空的类 mytype
        class mytype:
            pass

        # 创建一个包含单个字段的自定义 dtype 蓝图
        blueprint = np.dtype([("field", object)])
        # 使用 create_custom_field_dtype 函数创建自定义字段类型 dt
        dt = create_custom_field_dtype(blueprint, mytype, 0)
        # 断言自定义字段类型的类型是 mytype
        assert dt.type == mytype
        # 断言 np.dtype(mytype) 等于 np.dtype("O")
        # 这里是因为当前不能直接用 np.dtype 创建 mytype，因为 mytype 没有继承自 `np.generic` 类型，
        # 这看起来是一个不必要的限制，但这个限制已经存在很久了。
        assert np.dtype(mytype) == np.dtype("O")

        if HAS_REFCOUNT:
            # 创建一个数组并测试内存是否正确清理（gh-25949）
            o = object()
            a = np.array([o], dtype=dt)
            del a
            # 断言对象 o 的引用计数为 2
            assert sys.getrefcount(o) == 2

    def test_custom_structured_dtype_errors(self):
        # 定义一个空的类 mytype
        class mytype:
            pass

        # 创建一个包含单个字段的自定义 dtype 蓝图
        blueprint = np.dtype([("field", object)])

        with pytest.raises(ValueError):
            # 测试在创建过程中字段未设置会发生什么情况，这种情况当前会被拒绝，因为包含的对象未设置字段（见 PyArray_RegisterDataType）。
            create_custom_field_dtype(blueprint, mytype, 1)

        with pytest.raises(RuntimeError):
            # 测试 dtype 必须将其类型字段设置为 np.dtype 或内置实例。
            create_custom_field_dtype(blueprint, mytype, 2)


class TestClassGetItem:
    def test_dtype(self) -> None:
        # 创建一个 np.dtype[Any] 的别名
        alias = np.dtype[Any]
        # 断言别名是 types.GenericAlias 的实例
        assert isinstance(alias, types.GenericAlias)
        # 断言别名的原始类型是 np.dtype
        assert alias.__origin__ is np.dtype

    @pytest.mark.parametrize("code", np.typecodes["All"])
    def test_dtype_subclass(self, code: str) -> None:
        # 获取对应 typecode 的类
        cls = type(np.dtype(code))
        # 创建一个 cls[Any] 的别名
        alias = cls[Any]
        # 断言别名是 types.GenericAlias 的实例
        assert isinstance(alias, types.GenericAlias)
        # 断言别名的原始类型是 cls
        assert alias.__origin__ is cls

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        # 创建一个由 Any 组成的元组 arg_tup
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            # 断言 np.dtype[arg_tup] 是有效的
            assert np.dtype[arg_tup]
        else:
            # 测试当元组长度大于 1 时会抛出 TypeError
            with pytest.raises(TypeError):
                np.dtype[arg_tup]

    def test_subscript_scalar(self) -> None:
        # 断言 np.dtype[Any] 是有效的
        assert np.dtype[Any]


def test_result_type_integers_and_unitless_timedelta64():
    # 回归测试 gh-20077。以下调用 `result_type` 会导致段错误。
    td = np.timedelta64(4)
    # 调用 np.result_type 函数
    result = np.result_type(0, td)
    # 断言结果的 dtype 与 td 的 dtype 相等
    assert_dtype_equal(result, td.dtype)


def test_creating_dtype_with_dtype_class_errors():
    # 回归测试 #25031，用 np.dtype 自身调用会导致段错误。
    with pytest.raises(TypeError, match="Cannot convert np.dtype into a"):
        np.array(np.ones(10), dtype=np.dtype)
```