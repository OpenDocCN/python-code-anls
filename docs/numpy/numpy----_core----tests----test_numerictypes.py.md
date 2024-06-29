# `.\numpy\numpy\_core\tests\test_numerictypes.py`

```py
# 导入必要的模块和库
import sys  # 导入 sys 模块
import itertools  # 导入 itertools 模块

import pytest  # 导入 pytest 模块
import numpy as np  # 导入 NumPy 库并重命名为 np
import numpy._core.numerictypes as nt  # 导入 NumPy 的内部模块 numerictypes，并重命名为 nt
from numpy._core.numerictypes import (  # 从 numerictypes 模块导入多个函数和类
    issctype, sctype2char, maximum_sctype, sctypes
)
from numpy.testing import (  # 从 testing 模块导入多个函数和类
    assert_, assert_equal, assert_raises, assert_raises_regex, IS_PYPY
)

# This is the structure of the table used for plain objects:
#
# +-+-+-+
# |x|y|z|
# +-+-+-+

# Structure of a plain array description:
# 定义一个描述普通对象的结构
Pdescr = [
    ('x', 'i4', (2,)),  # 第一个字段 x，整数类型，形状为 (2,)
    ('y', 'f8', (2, 2)),  # 第二个字段 y，浮点数类型，形状为 (2, 2)
    ('z', 'u1')  # 第三个字段 z，无符号整数类型
]

# A plain list of tuples with values for testing:
# 用于测试的普通元组列表，包含了各个字段的数值
PbufferT = [
    # x     y                  z
    ([3, 2], [[6., 4.], [6., 4.]], 8),  # 第一组数据
    ([4, 3], [[7., 5.], [7., 5.]], 9)   # 第二组数据
]


# This is the structure of the table used for nested objects (DON'T PANIC!):
#
# +-+---------------------------------+-----+----------+-+-+
# |x|Info                             |color|info      |y|z|
# | +-----+--+----------------+----+--+     +----+-----+ | |
# | |value|y2|Info2           |name|z2|     |Name|Value| | |
# | |     |  +----+-----+--+--+    |  |     |    |     | | |
# | |     |  |name|value|y3|z3|    |  |     |    |     | | |
# +-+-----+--+----+-----+--+--+----+--+-----+----+-----+-+-+
#

# The corresponding nested array description:
# 定义一个描述嵌套对象的结构
Ndescr = [
    ('x', 'i4', (2,)),  # 第一个字段 x，整数类型，形状为 (2,)
    ('Info', [  # 第二个字段 Info，包含多个子字段
        ('value', 'c16'),  # value，复数类型
        ('y2', 'f8'),  # y2，浮点数类型
        ('Info2', [  # Info2 包含多个子字段
            ('name', 'S2'),  # name，字符串类型，长度为 2
            ('value', 'c16', (2,)),  # value，复数类型，形状为 (2,)
            ('y3', 'f8', (2,)),  # y3，浮点数类型，形状为 (2,)
            ('z3', 'u4', (2,))  # z3，无符号整数类型，形状为 (2,)
        ]),
        ('name', 'S2'),  # name，字符串类型，长度为 2
        ('z2', 'b1')  # z2，布尔类型
    ]),
    ('color', 'S2'),  # 第三个字段 color，字符串类型，长度为 2
    ('info', [  # 第四个字段 info，包含多个子字段
        ('Name', 'U8'),  # Name，Unicode 字符串类型，长度为 8
        ('Value', 'c16')  # Value，复数类型
    ]),
    ('y', 'f8', (2, 2)),  # 第五个字段 y，浮点数类型，形状为 (2, 2)
    ('z', 'u1')  # 第六个字段 z，无符号整数类型
]

NbufferT = [
    # x     Info                                                color info        y                  z
    #       value y2 Info2                            name z2         Name Value
    #                name   value    y3       z3
    ([3, 2], (6j, 6., (b'nn', [6j, 4j], [6., 4.], [1, 2]), b'NN', True),
     b'cc', ('NN', 6j), [[6., 4.], [6., 4.]], 8),  # 第一组数据
    ([4, 3], (7j, 7., (b'oo', [7j, 5j], [7., 5.], [2, 1]), b'OO', False),
     b'dd', ('OO', 7j), [[7., 5.], [7., 5.]], 9)  # 第二组数据
]


byteorder = {'little': '<', 'big': '>'}[sys.byteorder]  # 根据系统字节顺序选择对应的符号

def normalize_descr(descr):
    "Normalize a description adding the platform byteorder."
    # 将描述符描述中的字节顺序标准化为当前平台的字节顺序
    out = []
    # 遍历描述符列表中的每一项
    for item in descr:
        # 提取当前项的数据类型
        dtype = item[1]
        
        # 如果数据类型是字符串类型
        if isinstance(dtype, str):
            # 检查数据类型字符串的第一个字符，判断是否需要调整字节顺序
            if dtype[0] not in ['|', '<', '>']:
                # 检查是否为单字节数据或特定数据类型，若是则调整数据类型格式
                onebyte = dtype[1:] == "1"
                if onebyte or dtype[0] in ['S', 'V', 'b']:
                    dtype = "|" + dtype
                else:
                    dtype = byteorder + dtype
                    
            # 检查当前项是否包含附加信息，并且附加信息的元素数量大于1
            if len(item) > 2 and np.prod(item[2]) > 1:
                nitem = (item[0], dtype, item[2])  # 构建新的元组项
            else:
                nitem = (item[0], dtype)  # 构建新的元组项
            
            # 将新的元组项添加到输出列表中
            out.append(nitem)
        
        # 如果数据类型是列表类型
        elif isinstance(dtype, list):
            # 递归调用normalize_descr函数处理列表数据类型，并将处理结果作为元组项添加到输出列表中
            l = normalize_descr(dtype)
            out.append((item[0], l))
        
        # 如果数据类型既不是字符串类型也不是列表类型，则抛出异常
        else:
            raise ValueError("Expected a str or list and got %s" % (type(item)))
    
    # 返回处理后的输出列表
    return out
############################################################
#    Creation tests
############################################################

class CreateZeros:
    """Check the creation of heterogeneous arrays zero-valued"""

    def test_zeros0D(self):
        """Check creation of 0-dimensional objects"""
        # 创建一个零维数组对象，使用给定的描述符类型
        h = np.zeros((), dtype=self._descr)
        # 断言描述符已被规范化，并与数组的数据类型描述符相匹配
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        # 断言字段 'x' 的名称的前四个字符为 'void'
        assert_(h.dtype.fields['x'][0].name[:4] == 'void')
        # 断言字段 'x' 的字符为 'V'，表示它是 void 类型
        assert_(h.dtype.fields['x'][0].char == 'V')
        # 断言字段 'x' 的类型为 np.void
        assert_(h.dtype.fields['x'][0].type == np.void)
        # 对数据的一个小检查，确保数据正确初始化为零
        assert_equal(h['z'], np.zeros((), dtype='u1'))

    def test_zerosSD(self):
        """Check creation of single-dimensional objects"""
        # 创建一个单维数组对象，使用给定的描述符类型
        h = np.zeros((2,), dtype=self._descr)
        # 断言描述符已被规范化，并与数组的数据类型描述符相匹配
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        # 断言字段 'y' 的名称的前四个字符为 'void'
        assert_(h.dtype['y'].name[:4] == 'void')
        # 断言字段 'y' 的字符为 'V'，表示它是 void 类型
        assert_(h.dtype['y'].char == 'V')
        # 断言字段 'y' 的类型为 np.void
        assert_(h.dtype['y'].type == np.void)
        # 对数据的一个小检查，确保数据正确初始化为零
        assert_equal(h['z'], np.zeros((2,), dtype='u1'))

    def test_zerosMD(self):
        """Check creation of multi-dimensional objects"""
        # 创建一个多维数组对象，使用给定的描述符类型
        h = np.zeros((2, 3), dtype=self._descr)
        # 断言描述符已被规范化，并与数组的数据类型描述符相匹配
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        # 断言字段 'z' 的名称为 'uint8'
        assert_(h.dtype['z'].name == 'uint8')
        # 断言字段 'z' 的字符为 'B'，表示它是无符号字节类型
        assert_(h.dtype['z'].char == 'B')
        # 断言字段 'z' 的类型为 np.uint8
        assert_(h.dtype['z'].type == np.uint8)
        # 对数据的一个小检查，确保数据正确初始化为零
        assert_equal(h['z'], np.zeros((2, 3), dtype='u1'))


class TestCreateZerosPlain(CreateZeros):
    """Check the creation of heterogeneous arrays zero-valued (plain)"""
    _descr = Pdescr

class TestCreateZerosNested(CreateZeros):
    """Check the creation of heterogeneous arrays zero-valued (nested)"""
    _descr = Ndescr


class CreateValues:
    """Check the creation of heterogeneous arrays with values"""

    def test_tuple(self):
        """Check creation from tuples"""
        # 从元组创建数组对象，使用给定的描述符类型
        h = np.array(self._buffer, dtype=self._descr)
        # 断言描述符已被规范化，并与数组的数据类型描述符相匹配
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        # 如果存在多行数据，断言数组的形状为 (2,)
        if self.multiple_rows:
            assert_(h.shape == (2,))
        else:
            assert_(h.shape == ())

    def test_list_of_tuple(self):
        """Check creation from list of tuples"""
        # 从元组列表创建数组对象，使用给定的描述符类型
        h = np.array([self._buffer], dtype=self._descr)
        # 断言描述符已被规范化，并与数组的数据类型描述符相匹配
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        # 如果存在多行数据，断言数组的形状为 (1, 2)
        if self.multiple_rows:
            assert_(h.shape == (1, 2))
        else:
            assert_(h.shape == (1,))

    def test_list_of_list_of_tuple(self):
        """Check creation from list of list of tuples"""
        # 从元组列表的列表创建数组对象，使用给定的描述符类型
        h = np.array([[self._buffer]], dtype=self._descr)
        # 断言描述符已被规范化，并与数组的数据类型描述符相匹配
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        # 如果存在多行数据，断言数组的形状为 (1, 1, 2)
        if self.multiple_rows:
            assert_(h.shape == (1, 1, 2))
        else:
            assert_(h.shape == (1, 1))


class TestCreateValuesPlainSingle(CreateValues):
    """Check the creation of heterogeneous arrays (plain, single row)"""
    # 将 Pdescr 赋值给 _descr 变量
    _descr = Pdescr
    # 初始化 multiple_rows 变量为 0
    multiple_rows = 0
    # 从 PbufferT 列表中取出第一个元素，并赋值给 _buffer 变量
    _buffer = PbufferT[0]
class TestCreateValuesPlainMultiple(CreateValues):
    """Check the creation of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr  # 设置描述符为 Pdescr
    multiple_rows = 1  # 多行标志设为 1
    _buffer = PbufferT  # 缓冲区设为 PbufferT

class TestCreateValuesNestedSingle(CreateValues):
    """Check the creation of heterogeneous arrays (nested, single row)"""
    _descr = Ndescr  # 设置描述符为 Ndescr
    multiple_rows = 0  # 多行标志设为 0
    _buffer = NbufferT[0]  # 缓冲区设为 NbufferT 的第一个元素

class TestCreateValuesNestedMultiple(CreateValues):
    """Check the creation of heterogeneous arrays (nested, multiple rows)"""
    _descr = Ndescr  # 设置描述符为 Ndescr
    multiple_rows = 1  # 多行标志设为 1
    _buffer = NbufferT  # 缓冲区设为 NbufferT


############################################################
#    Reading tests
############################################################

class ReadValuesPlain:
    """Check the reading of values in heterogeneous arrays (plain)"""

    def test_access_fields(self):
        # 使用描述符和缓冲区创建 NumPy 数组 h
        h = np.array(self._buffer, dtype=self._descr)
        if not self.multiple_rows:
            assert_(h.shape == ())  # 断言 h 的形状为空
            assert_equal(h['x'], np.array(self._buffer[0], dtype='i4'))  # 断言 h 的 'x' 值等于缓冲区第一个元素的 'i4' 类型数组
            assert_equal(h['y'], np.array(self._buffer[1], dtype='f8'))  # 断言 h 的 'y' 值等于缓冲区第二个元素的 'f8' 类型数组
            assert_equal(h['z'], np.array(self._buffer[2], dtype='u1'))  # 断言 h 的 'z' 值等于缓冲区第三个元素的 'u1' 类型数组
        else:
            assert_(len(h) == 2)  # 断言 h 的长度为 2
            assert_equal(h['x'], np.array([self._buffer[0][0],
                                             self._buffer[1][0]], dtype='i4'))  # 断言 h 的 'x' 值等于缓冲区第一列 'i4' 类型数组
            assert_equal(h['y'], np.array([self._buffer[0][1],
                                             self._buffer[1][1]], dtype='f8'))  # 断言 h 的 'y' 值等于缓冲区第二列 'f8' 类型数组
            assert_equal(h['z'], np.array([self._buffer[0][2],
                                             self._buffer[1][2]], dtype='u1'))  # 断言 h 的 'z' 值等于缓冲区第三列 'u1' 类型数组


class TestReadValuesPlainSingle(ReadValuesPlain):
    """Check the creation of heterogeneous arrays (plain, single row)"""
    _descr = Pdescr  # 设置描述符为 Pdescr
    multiple_rows = 0  # 多行标志设为 0
    _buffer = PbufferT[0]  # 缓冲区设为 PbufferT 的第一个元素

class TestReadValuesPlainMultiple(ReadValuesPlain):
    """Check the values of heterogeneous arrays (plain, multiple rows)"""
    _descr = Pdescr  # 设置描述符为 Pdescr
    multiple_rows = 1  # 多行标志设为 1
    _buffer = PbufferT  # 缓冲区设为 PbufferT

class ReadValuesNested:
    """Check the reading of values in heterogeneous arrays (nested)"""
    def test_access_top_fields(self):
        """检查访问嵌套数组的顶层字段"""
        # 使用 self._buffer 和 self._descr 创建一个 NumPy 数组 h
        h = np.array(self._buffer, dtype=self._descr)
        # 如果不是多行数据
        if not self.multiple_rows:
            # 断言 h 的形状为空元组
            assert_(h.shape == ())
            # 断言 h 中字段 'x' 的值等于 self._buffer[0] 的整数数组
            assert_equal(h['x'], np.array(self._buffer[0], dtype='i4'))
            # 断言 h 中字段 'y' 的值等于 self._buffer[4] 的浮点数数组
            assert_equal(h['y'], np.array(self._buffer[4], dtype='f8'))
            # 断言 h 中字段 'z' 的值等于 self._buffer[5] 的无符号整数数组
            assert_equal(h['z'], np.array(self._buffer[5], dtype='u1'))
        else:
            # 断言 h 的长度为 2
            assert_(len(h) == 2)
            # 断言 h 中字段 'x' 的值等于 self._buffer 中每行的第一个元素的整数数组
            assert_equal(h['x'], np.array([self._buffer[0][0],
                                           self._buffer[1][0]], dtype='i4'))
            # 断言 h 中字段 'y' 的值等于 self._buffer 中每行的第五个元素的浮点数数组
            assert_equal(h['y'], np.array([self._buffer[0][4],
                                           self._buffer[1][4]], dtype='f8'))
            # 断言 h 中字段 'z' 的值等于 self._buffer 中每行的第六个元素的无符号整数数组
            assert_equal(h['z'], np.array([self._buffer[0][5],
                                           self._buffer[1][5]], dtype='u1'))

    def test_nested1_acessors(self):
        """检查访问嵌套数组的嵌套字段（第一层级）"""
        # 使用 self._buffer 和 self._descr 创建一个 NumPy 数组 h
        h = np.array(self._buffer, dtype=self._descr)
        # 如果不是多行数据
        if not self.multiple_rows:
            # 断言 h 中字段 'Info' 下的 'value' 等于 self._buffer[1][0] 的长度为 16 的字符数组
            assert_equal(h['Info']['value'],
                         np.array(self._buffer[1][0], dtype='c16'))
            # 断言 h 中字段 'Info' 下的 'y2' 等于 self._buffer[1][1] 的浮点数数组
            assert_equal(h['Info']['y2'],
                         np.array(self._buffer[1][1], dtype='f8'))
            # 断言 h 中字段 'info' 下的 'Name' 等于 self._buffer[3][0] 的长度为 2 的 Unicode 字符数组
            assert_equal(h['info']['Name'],
                         np.array(self._buffer[3][0], dtype='U2'))
            # 断言 h 中字段 'info' 下的 'Value' 等于 self._buffer[3][1] 的长度为 16 的字符数组
            assert_equal(h['info']['Value'],
                         np.array(self._buffer[3][1], dtype='c16'))
        else:
            # 断言 h 中字段 'Info' 下的 'value' 等于 self._buffer 中每行第二个元素的长度为 16 的字符数组
            assert_equal(h['Info']['value'],
                         np.array([self._buffer[0][1][0],
                                   self._buffer[1][1][0]],
                                  dtype='c16'))
            # 断言 h 中字段 'Info' 下的 'y2' 等于 self._buffer 中每行第二个元素的浮点数数组
            assert_equal(h['Info']['y2'],
                         np.array([self._buffer[0][1][1],
                                   self._buffer[1][1][1]],
                                  dtype='f8'))
            # 断言 h 中字段 'info' 下的 'Name' 等于 self._buffer 中每行第四个元素的长度为 2 的 Unicode 字符数组
            assert_equal(h['info']['Name'],
                         np.array([self._buffer[0][3][0],
                                   self._buffer[1][3][0]],
                                  dtype='U2'))
            # 断言 h 中字段 'info' 下的 'Value' 等于 self._buffer 中每行第四个元素的长度为 16 的字符数组
            assert_equal(h['info']['Value'],
                         np.array([self._buffer[0][3][1],
                                   self._buffer[1][3][1]],
                                  dtype='c16'))
    def test_nested2_acessors(self):
        """Check reading the nested fields of a nested array (2nd level)"""
        # 将 self._buffer 转换为 NumPy 数组 h，使用给定的数据类型 self._descr
        h = np.array(self._buffer, dtype=self._descr)
        # 如果不是多行数据：
        if not self.multiple_rows:
            # 断言验证 h['Info']['Info2']['value'] 是否等于指定的复数数组
            assert_equal(h['Info']['Info2']['value'],
                         np.array(self._buffer[1][2][1], dtype='c16'))
            # 断言验证 h['Info']['Info2']['z3'] 是否等于指定的无符号整数数组
            assert_equal(h['Info']['Info2']['z3'],
                         np.array(self._buffer[1][2][3], dtype='u4'))
        else:
            # 断言验证 h['Info']['Info2']['value'] 是否等于指定的复数数组列表
            assert_equal(h['Info']['Info2']['value'],
                         np.array([self._buffer[0][1][2][1],
                                   self._buffer[1][1][2][1]],
                                  dtype='c16'))
            # 断言验证 h['Info']['Info2']['z3'] 是否等于指定的无符号整数数组列表
            assert_equal(h['Info']['Info2']['z3'],
                         np.array([self._buffer[0][1][2][3],
                                   self._buffer[1][1][2][3]],
                                  dtype='u4'))

    def test_nested1_descriptor(self):
        """Check access nested descriptors of a nested array (1st level)"""
        # 将 self._buffer 转换为 NumPy 数组 h，使用给定的数据类型 self._descr
        h = np.array(self._buffer, dtype=self._descr)
        # 断言验证 h.dtype['Info']['value'] 的名称是否为 'complex128'
        assert_(h.dtype['Info']['value'].name == 'complex128')
        # 断言验证 h.dtype['Info']['y2'] 的名称是否为 'float64'
        assert_(h.dtype['Info']['y2'].name == 'float64')
        # 断言验证 h.dtype['info']['Name'] 的名称是否为 'str256'
        assert_(h.dtype['info']['Name'].name == 'str256')
        # 断言验证 h.dtype['info']['Value'] 的名称是否为 'complex128'
        assert_(h.dtype['info']['Value'].name == 'complex128')

    def test_nested2_descriptor(self):
        """Check access nested descriptors of a nested array (2nd level)"""
        # 将 self._buffer 转换为 NumPy 数组 h，使用给定的数据类型 self._descr
        h = np.array(self._buffer, dtype=self._descr)
        # 断言验证 h.dtype['Info']['Info2']['value'] 的名称是否为 'void256'
        assert_(h.dtype['Info']['Info2']['value'].name == 'void256')
        # 断言验证 h.dtype['Info']['Info2']['z3'] 的名称是否为 'void64'
        assert_(h.dtype['Info']['Info2']['z3'].name == 'void64')
class TestReadValuesNestedSingle(ReadValuesNested):
    """Check the values of heterogeneous arrays (nested, single row)"""
    # 设置描述符为Ndescr
    _descr = Ndescr
    # 多行标志设为False
    multiple_rows = False
    # 缓冲区为NbufferT的第一个元素
    _buffer = NbufferT[0]

class TestReadValuesNestedMultiple(ReadValuesNested):
    """Check the values of heterogeneous arrays (nested, multiple rows)"""
    # 设置描述符为Ndescr
    _descr = Ndescr
    # 多行标志设为True
    multiple_rows = True
    # 缓冲区为NbufferT
    _buffer = NbufferT

class TestEmptyField:
    def test_assign(self):
        # 创建一个包含10个浮点数的数组a
        a = np.arange(10, dtype=np.float32)
        # 重新设定数组a的数据类型为一个结构化数组，包含一个int字段和两个float字段
        a.dtype = [("int",   "<0i4"), ("float", "<2f4")]
        # 断言int字段的形状为(5, 0)
        assert_(a['int'].shape == (5, 0))
        # 断言float字段的形状为(5, 2)
        assert_(a['float'].shape == (5, 2))


class TestMultipleFields:
    def setup_method(self):
        # 创建一个结构化数组ary，包含两行四列的数据，每列的类型分别为i4, f4, i2, c8
        self.ary = np.array([(1, 2, 3, 4), (5, 6, 7, 8)], dtype='i4,f4,i2,c8')

    def _bad_call(self):
        # 尝试返回数组ary中的'f0'和'f1'字段，这会触发IndexError
        return self.ary['f0', 'f1']

    def test_no_tuple(self):
        # 断言调用_bad_call方法会引发IndexError异常
        assert_raises(IndexError, self._bad_call)

    def test_return(self):
        # 返回包含ary中'f0'和'f2'字段数据的列表
        res = self.ary[['f0', 'f2']].tolist()
        # 断言返回的列表与预期结果[(1, 3), (5, 7)]相等
        assert_(res == [(1, 3), (5, 7)])


class TestIsSubDType:
    # 标量类型可以被提升为dtype
    wrappers = [np.dtype, lambda x: x]

    def test_both_abstract(self):
        # 断言np.floating是np.inexact的子类型
        assert_(np.issubdtype(np.floating, np.inexact))
        # 断言np.inexact不是np.floating的子类型
        assert_(not np.issubdtype(np.inexact, np.floating))

    def test_same(self):
        # 针对np.float32和np.int32，通过不同的包装器组合，断言类型之间的子类型关系
        for cls in (np.float32, np.int32):
            for w1, w2 in itertools.product(self.wrappers, repeat=2):
                assert_(np.issubdtype(w1(cls), w2(cls)))

    def test_subclass(self):
        # 使用包装器，断言np.float32和np.float64是np.floating的子类型
        for w in self.wrappers:
            assert_(np.issubdtype(w(np.float32), np.floating))
            assert_(np.issubdtype(w(np.float64), np.floating))

    def test_subclass_backwards(self):
        # 使用包装器，断言np.floating不是np.float32或np.float64的子类型
        for w in self.wrappers:
            assert_(not np.issubdtype(np.floating, w(np.float32)))
            assert_(not np.issubdtype(np.floating, w(np.float64)))

    def test_sibling_class(self):
        # 使用不同的包装器组合，断言np.float32不是np.float64的子类型，反之亦然
        for w1, w2 in itertools.product(self.wrappers, repeat=2):
            assert_(not np.issubdtype(w1(np.float32), w2(np.float64)))
            assert_(not np.issubdtype(w1(np.float64), w2(np.float32)))
    def test_nondtype_nonscalartype(self):
        # 以下测试用例源自于 gh-9505，该问题由 gh-14619 和 gh-9505 引入了废弃修复。
        # 检查 np.float32 是否不是 'float64' 的子类型
        assert not np.issubdtype(np.float32, 'float64')
        # 检查 np.float32 是否不是 'f8' 的子类型
        assert not np.issubdtype(np.float32, 'f8')
        # 检查 np.int32 是否不是 str 的子类型
        assert not np.issubdtype(np.int32, str)
        # 检查 np.int32 是否不是 'int64' 的子类型
        assert not np.issubdtype(np.int32, 'int64')
        # 检查 np.str_ 是否不是 'void' 的子类型
        assert not np.issubdtype(np.str_, 'void')
        # 检查 np.int8 是否不是 int 的子类型，正确的是 np.integer
        assert not np.issubdtype(np.int8, int)
        # 检查 np.float32 是否不是 float 的子类型
        assert not np.issubdtype(np.float32, float)
        # 检查 np.complex64 是否不是 complex 的子类型
        assert not np.issubdtype(np.complex64, complex)
        # 检查 np.float32 是否不是 "float" 的子类型
        assert not np.issubdtype(np.float32, "float")
        # 检查 np.float64 是否是 "float" 的子类型
        assert np.issubdtype(np.float64, "float")

        # 测试 int、float、complex 的正确数据类型和抽象类型
        # 对于 int、float、complex，正确的数据类型应为 np.integer、np.floating、np.complexfloating
        assert np.issubdtype(np.float64, 'float64')
        assert np.issubdtype(np.float64, 'f8')
        assert np.issubdtype(np.str_, str)
        assert np.issubdtype(np.int64, 'int64')
        assert np.issubdtype(np.void, 'void')
        assert np.issubdtype(np.int8, np.integer)
        assert np.issubdtype(np.float32, np.floating)
        assert np.issubdtype(np.complex64, np.complexfloating)
        # 检查 np.float64 是否是 "float" 的子类型
        assert np.issubdtype(np.float64, "float")
        # 检查 np.float32 是否是 "f" 的子类型
        assert np.issubdtype(np.float32, "f")
class TestIsDType:
    """
    Check correctness of `np.isdtype`. The test considers different argument
    configurations: `np.isdtype(dtype, k1)` and `np.isdtype(dtype, (k1, k2))`
    with concrete dtypes and dtype groups.
    """

    # 定义 dtype 组的字典，映射到具体的 NumPy 类型数组
    dtype_group_dict = {
        "signed integer": sctypes["int"],
        "unsigned integer": sctypes["uint"],
        "integral": sctypes["int"] + sctypes["uint"],
        "real floating": sctypes["float"],
        "complex floating": sctypes["complex"],
        "numeric": (
            sctypes["int"] + sctypes["uint"] + sctypes["float"] +
            sctypes["complex"]
        )
    }

    @pytest.mark.parametrize(
        "dtype,close_dtype",
        [
            (np.int64, np.int32), (np.uint64, np.uint32),
            (np.float64, np.float32), (np.complex128, np.complex64)
        ]
    )
    @pytest.mark.parametrize(
        "dtype_group",
        [
            None, "signed integer", "unsigned integer", "integral",
            "real floating", "complex floating", "numeric"
        ]
    )
    # 测试 np.isdtype 函数的不同参数组合
    def test_isdtype(self, dtype, close_dtype, dtype_group):
        # 首先检查相同的 dtype 应返回 `True`，不同的应返回 `False`
        # 即使它们在 dtype 层次结构中接近也是如此！
        if dtype_group is None:
            assert np.isdtype(dtype, dtype)
            assert not np.isdtype(dtype, close_dtype)
            assert np.isdtype(dtype, (dtype, close_dtype))

        # 检查 dtype 是否属于指定的 dtype 组，应返回 `True`，否则返回 `False`
        elif dtype in self.dtype_group_dict[dtype_group]:
            assert np.isdtype(dtype, dtype_group)
            assert np.isdtype(dtype, (close_dtype, dtype_group))
        else:
            assert not np.isdtype(dtype, dtype_group)

    # 测试 np.isdtype 函数的无效参数
    def test_isdtype_invalid_args(self):
        with assert_raises_regex(TypeError, r".*must be a NumPy dtype.*"):
            np.isdtype("int64", np.int64)
        with assert_raises_regex(TypeError, r".*kind argument must.*"):
            np.isdtype(np.int64, 1)
        with assert_raises_regex(ValueError, r".*not a known kind name.*"):
            np.isdtype(np.int64, "int64")

    # 检查 sctypes 字典的完整性，特别是在 32 位系统上
    def test_sctypes_complete(self):
        assert np.int32 in sctypes['int']
        assert np.intc in sctypes['int']
        assert np.int64 in sctypes['int']
        assert np.uint32 in sctypes['uint']
        assert np.uintc in sctypes['uint']
        assert np.uint64 in sctypes['uint']


class TestSctypeDict:
    # 测试 np._core.sctypeDict 字典中的 longdouble 类型
    def test_longdouble(self):
        assert_(np._core.sctypeDict['float64'] is not np.longdouble)
        assert_(np._core.sctypeDict['complex128'] is not np.clongdouble)

    # 测试 np._core.sctypeDict 字典中的 ulong 类型
    def test_ulong(self):
        assert np._core.sctypeDict['ulong'] is np.ulong
        assert np.dtype(np.ulong) is np.dtype("ulong")
        assert np.dtype(np.ulong).itemsize == np.dtype(np.long).itemsize
@pytest.mark.filterwarnings("ignore:.*maximum_sctype.*:DeprecationWarning")
class TestMaximumSctype:
    # 定义一个测试类 TestMaximumSctype，用于测试 maximum_sctype 函数的各种数据类型情况

    # note that parametrizing with sctype['int'] and similar would skip types
    # with the same size (gh-11923)
    # 注意，使用 sctype['int'] 等进行参数化会跳过相同大小的类型 (gh-11923)

    @pytest.mark.parametrize(
        't', [np.byte, np.short, np.intc, np.long, np.longlong]
    )
    def test_int(self, t):
        # 测试整数类型的最大数据类型
        assert_equal(maximum_sctype(t), np._core.sctypes['int'][-1])

    @pytest.mark.parametrize(
        't', [np.ubyte, np.ushort, np.uintc, np.ulong, np.ulonglong]
    )
    def test_uint(self, t):
        # 测试无符号整数类型的最大数据类型
        assert_equal(maximum_sctype(t), np._core.sctypes['uint'][-1])

    @pytest.mark.parametrize('t', [np.half, np.single, np.double, np.longdouble])
    def test_float(self, t):
        # 测试浮点数类型的最大数据类型
        assert_equal(maximum_sctype(t), np._core.sctypes['float'][-1])

    @pytest.mark.parametrize('t', [np.csingle, np.cdouble, np.clongdouble])
    def test_complex(self, t):
        # 测试复数类型的最大数据类型
        assert_equal(maximum_sctype(t), np._core.sctypes['complex'][-1])

    @pytest.mark.parametrize('t', [np.bool, np.object_, np.str_, np.bytes_,
                                   np.void])
    def test_other(self, t):
        # 测试其他类型的最大数据类型
        assert_equal(maximum_sctype(t), t)


class Test_sctype2char:
    # 定义一个测试类 Test_sctype2char，用于测试 sctype2char 函数的各种数据类型情况
    # This function is old enough that we're really just documenting the quirks
    # at this point.
    # 这个函数已经足够老了，我们只是在记录当前阶段的一些怪异行为。

    def test_scalar_type(self):
        # 测试标量类型的 sctype2char 转换
        assert_equal(sctype2char(np.double), 'd')
        assert_equal(sctype2char(np.long), 'l')
        assert_equal(sctype2char(np.int_), np.array(0).dtype.char)
        assert_equal(sctype2char(np.str_), 'U')
        assert_equal(sctype2char(np.bytes_), 'S')

    def test_other_type(self):
        # 测试其他类型的 sctype2char 转换
        assert_equal(sctype2char(float), 'd')
        assert_equal(sctype2char(list), 'O')
        assert_equal(sctype2char(np.ndarray), 'O')

    def test_third_party_scalar_type(self):
        # 测试第三方标量类型的 sctype2char 转换
        from numpy._core._rational_tests import rational
        assert_raises(KeyError, sctype2char, rational)
        assert_raises(KeyError, sctype2char, rational(1))

    def test_array_instance(self):
        # 测试数组实例的 sctype2char 转换
        assert_equal(sctype2char(np.array([1.0, 2.0])), 'd')

    def test_abstract_type(self):
        # 测试抽象类型的 sctype2char 转换
        assert_raises(KeyError, sctype2char, np.floating)

    def test_non_type(self):
        # 测试非类型的 sctype2char 转换
        assert_raises(ValueError, sctype2char, 1)


@pytest.mark.parametrize("rep, expected", [
    (np.int32, True),
    (list, False),
    (1.1, False),
    (str, True),
    (np.dtype(np.float64), True),
    (np.dtype((np.int16, (3, 4))), True),
    (np.dtype([('a', np.int8)]), True),
    ])
def test_issctype(rep, expected):
    # ensure proper identification of scalar
    # data-types by issctype()
    # 确保 issctype() 正确识别标量数据类型
    actual = issctype(rep)
    assert_equal(actual, expected)


@pytest.mark.skipif(sys.flags.optimize > 1,
                    reason="no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1")
@pytest.mark.xfail(IS_PYPY,
                   reason="PyPy cannot modify tp_doc after PyType_Ready")
class TestDocStrings:
    # 定义一个测试类 TestDocStrings，用于测试在不同条件下文档字符串的处理行为
    def test_platform_dependent_aliases(self):
        # 检查 numpy 库中的 np.int64 是否和 np.int_ 相同
        if np.int64 is np.int_:
            # 如果相同，断言 'int64' 在 np.int_ 对象的文档字符串中出现
            assert_('int64' in np.int_.__doc__)
        # 如果 np.int64 和 np.longlong 相同
        elif np.int64 is np.longlong:
            # 断言 'int64' 在 np.longlong 对象的文档字符串中出现
            assert_('int64' in np.longlong.__doc__)
# 定义一个测试类 TestScalarTypeNames，用于测试数值类型的命名是否正确
class TestScalarTypeNames:
    # gh-9799

    # 定义一个包含各种数值类型的列表
    numeric_types = [
        np.byte, np.short, np.intc, np.long, np.longlong,
        np.ubyte, np.ushort, np.uintc, np.ulong, np.ulonglong,
        np.half, np.single, np.double, np.longdouble,
        np.csingle, np.cdouble, np.clongdouble,
    ]

    # 测试数值类型的名称是否唯一
    def test_names_are_unique(self):
        # 检查列表中的元素是否都不是别名
        assert len(set(self.numeric_types)) == len(self.numeric_types)

        # 检查数值类型的名称是否唯一
        names = [t.__name__ for t in self.numeric_types]
        assert len(set(names)) == len(names)

    # 使用 pytest 的 parametrize 装饰器，为每种数值类型 t 运行以下测试
    @pytest.mark.parametrize('t', numeric_types)
    def test_names_reflect_attributes(self, t):
        """ Test that names correspond to where the type is under ``np.`` """
        # 检查数值类型的名称是否对应于 np 下的类型
        assert getattr(np, t.__name__) is t

    # 使用 pytest 的 parametrize 装饰器，为每种数值类型 t 运行以下测试
    @pytest.mark.parametrize('t', numeric_types)
    def test_names_are_undersood_by_dtype(self, t):
        """ Test the dtype constructor maps names back to the type """
        # 检查 dtype 构造函数是否能将名称映射回类型
        assert np.dtype(t.__name__).type is t


# 定义一个测试类 TestBoolDefinition，用于测试布尔类型的定义
class TestBoolDefinition:
    # 测试布尔类型的定义
    def test_bool_definition(self):
        # 检查 nt.bool 是否等于 np.bool
        assert nt.bool is np.bool
```