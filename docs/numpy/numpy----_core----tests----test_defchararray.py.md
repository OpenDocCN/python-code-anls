# `.\numpy\numpy\_core\tests\test_defchararray.py`

```py
import pytest  # 导入 pytest 测试框架

import numpy as np  # 导入 NumPy 库
from numpy._core.multiarray import _vec_string  # 导入 NumPy 内部模块 _vec_string
from numpy.testing import (  # 导入 NumPy 测试模块中的多个函数
    assert_, assert_equal, assert_array_equal, assert_raises,
    assert_raises_regex
    )

kw_unicode_true = {'unicode': True}  # 定义关键字参数字典，用于启用 Unicode 支持，以便 2to3 正常工作
kw_unicode_false = {'unicode': False}

class TestBasic:
    def test_from_object_array(self):
        A = np.array([['abc', 2],  # 创建一个 NumPy 对象数组 A，包含字符串和整数
                      ['long   ', '0123456789']], dtype='O')
        B = np.char.array(A)  # 使用 np.char.array 将 A 转换为字符串数组 B
        assert_equal(B.dtype.itemsize, 10)  # 断言 B 的元素的字节大小为 10
        assert_array_equal(B, [[b'abc', b'2'],  # 断言 B 与预期的二维数组相等
                               [b'long', b'0123456789']])

    def test_from_object_array_unicode(self):
        A = np.array([['abc', 'Sigma \u03a3'],  # 创建一个包含 Unicode 字符串的 NumPy 对象数组 A
                      ['long   ', '0123456789']], dtype='O')
        assert_raises(ValueError, np.char.array, (A,))  # 断言调用 np.char.array(A) 抛出 ValueError 异常
        B = np.char.array(A, **kw_unicode_true)  # 使用关键字参数启用 Unicode 支持创建字符串数组 B
        assert_equal(B.dtype.itemsize, 10 * np.array('a', 'U').dtype.itemsize)  # 断言 B 中元素的字节大小与 Unicode 字符 'a' 的字节大小乘以 10 相等
        assert_array_equal(B, [['abc', 'Sigma \u03a3'],  # 断言 B 与预期的二维数组相等
                               ['long', '0123456789']])

    def test_from_string_array(self):
        A = np.array([[b'abc', b'foo'],  # 创建一个字节字符串数组 A
                      [b'long   ', b'0123456789']])
        assert_equal(A.dtype.type, np.bytes_)  # 断言 A 的数据类型是 np.bytes_
        B = np.char.array(A)  # 使用 np.char.array 将 A 转换为字符串数组 B
        assert_array_equal(B, A)  # 断言 B 与 A 相等
        assert_equal(B.dtype, A.dtype)  # 断言 B 的数据类型与 A 的数据类型相等
        assert_equal(B.shape, A.shape)  # 断言 B 的形状与 A 的形状相等
        B[0, 0] = 'changed'  # 修改 B 中的元素
        assert_(B[0, 0] != A[0, 0])  # 断言修改后 B 的元素不等于 A 的原始元素
        C = np.char.asarray(A)  # 使用 np.char.asarray 将 A 转换为字符串数组 C
        assert_array_equal(C, A)  # 断言 C 与 A 相等
        assert_equal(C.dtype, A.dtype)  # 断言 C 的数据类型与 A 的数据类型相等
        C[0, 0] = 'changed again'  # 修改 C 中的元素
        assert_(C[0, 0] != B[0, 0])  # 断言修改后 C 的元素不等于 B 的元素
        assert_(C[0, 0] == A[0, 0])  # 断言修改后 C 的元素等于 A 的原始元素

    def test_from_unicode_array(self):
        A = np.array([['abc', 'Sigma \u03a3'],  # 创建一个包含 Unicode 字符串的 NumPy 对象数组 A
                      ['long   ', '0123456789']])
        assert_equal(A.dtype.type, np.str_)  # 断言 A 的数据类型是 np.str_
        B = np.char.array(A)  # 使用 np.char.array 将 A 转换为字符串数组 B
        assert_array_equal(B, A)  # 断言 B 与 A 相等
        assert_equal(B.dtype, A.dtype)  # 断言 B 的数据类型与 A 的数据类型相等
        assert_equal(B.shape, A.shape)  # 断言 B 的形状与 A 的形状相等
        B = np.char.array(A, **kw_unicode_true)  # 使用关键字参数启用 Unicode 支持创建字符串数组 B
        assert_array_equal(B, A)  # 断言 B 与 A 相等
        assert_equal(B.dtype, A.dtype)  # 断言 B 的数据类型与 A 的数据类型相等
        assert_equal(B.shape, A.shape)  # 断言 B 的形状与 A 的形状相等

        def fail():
            np.char.array(A, **kw_unicode_false)  # 尝试使用禁用 Unicode 支持的关键字参数创建字符串数组

        assert_raises(UnicodeEncodeError, fail)  # 断言调用 fail() 抛出 UnicodeEncodeError 异常

    def test_unicode_upconvert(self):
        A = np.char.array(['abc'])  # 创建一个包含字节字符串的字符串数组 A
        B = np.char.array(['\u03a3'])  # 创建一个包含 Unicode 字符串的字符串数组 B
        assert_(issubclass((A + B).dtype.type, np.str_))  # 断言 (A + B) 的数据类型是 np.str_

    def test_from_string(self):
        A = np.char.array(b'abc')  # 创建一个字节字符串的字符串数组 A
        assert_equal(len(A), 1)  # 断言 A 的长度为 1
        assert_equal(len(A[0]), 3)  # 断言 A 的第一个元素的长度为 3
        assert_(issubclass(A.dtype.type, np.bytes_))  # 断言 A 的数据类型是 np.bytes_

    def test_from_unicode(self):
        A = np.char.array('\u03a3')  # 创建一个包含 Unicode 字符串的字符串数组 A
        assert_equal(len(A), 1)  # 断言 A 的长度为 1
        assert_equal(len(A[0]), 1)  # 断言 A 的第一个元素的长度为 1
        assert_equal(A.itemsize, 4)  # 断言 A 中元素的字节大小为 4
        assert_(issubclass(A.dtype.type, np.str_))  # 断言 A 的数据类型是 np.str_

class TestVecString:
    # 定义测试方法，用于测试不存在的方法
    def test_non_existent_method(self):

        # 定义一个会失败的函数，尝试调用 _vec_string 方法，但提供的方法名不存在
        def fail():
            _vec_string('a', np.bytes_, 'bogus')

        # 断言调用 fail 函数会抛出 AttributeError 异常
        assert_raises(AttributeError, fail)

    # 定义测试方法，用于测试非字符串数组作为输入时的情况
    def test_non_string_array(self):

        # 定义一个会失败的函数，尝试将整数作为输入传递给 _vec_string 方法
        def fail():
            _vec_string(1, np.bytes_, 'strip')

        # 断言调用 fail 函数会抛出 TypeError 异常
        assert_raises(TypeError, fail)

    # 定义测试方法，用于测试传递无效参数元组时的情况
    def test_invalid_args_tuple(self):

        # 定义一个会失败的函数，尝试传递包含非字符串元素的列表作为参数元组给 _vec_string 方法
        def fail():
            _vec_string(['a'], np.bytes_, 'strip', 1)

        # 断言调用 fail 函数会抛出 TypeError 异常
        assert_raises(TypeError, fail)

    # 定义测试方法，用于测试传递无效的类型描述符时的情况
    def test_invalid_type_descr(self):

        # 定义一个会失败的函数，尝试将无效的类型描述符 'BOGUS' 传递给 _vec_string 方法
        def fail():
            _vec_string(['a'], 'BOGUS', 'strip')

        # 断言调用 fail 函数会抛出 TypeError 异常
        assert_raises(TypeError, fail)

    # 定义测试方法，用于测试传递无效函数参数时的情况
    def test_invalid_function_args(self):

        # 定义一个会失败的函数，尝试将包含非单一元素元组作为函数参数传递给 _vec_string 方法
        def fail():
            _vec_string(['a'], np.bytes_, 'strip', (1,))

        # 断言调用 fail 函数会抛出 TypeError 异常
        assert_raises(TypeError, fail)

    # 定义测试方法，用于测试返回类型不正确时的情况
    def test_invalid_result_type(self):

        # 定义一个会失败的函数，尝试调用 _vec_string 方法，但期望返回类型为 np.int_ 而不是字符串
        def fail():
            _vec_string(['a'], np.int_, 'strip')

        # 断言调用 fail 函数会抛出 TypeError 异常
        assert_raises(TypeError, fail)

    # 定义测试方法，用于测试广播错误时的情况
    def test_broadcast_error(self):

        # 定义一个会失败的函数，尝试将包含不同长度子列表的列表作为参数广播给 _vec_string 方法
        def fail():
            _vec_string([['abc', 'def']], np.int_, 'find', (['a', 'd', 'j'],))

        # 断言调用 fail 函数会抛出 ValueError 异常
        assert_raises(ValueError, fail)
class TestWhitespace:
    # 设置测试前的准备工作，初始化两个二维字符数组 A 和 B
    def setup_method(self):
        self.A = np.array([['abc ', '123  '],
                           ['789 ', 'xyz ']]).view(np.char.chararray)
        self.B = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.char.chararray)

    # 测试相等性的方法
    def test1(self):
        assert_(np.all(self.A == self.B))  # 断言 A 等于 B 的所有元素
        assert_(np.all(self.A >= self.B))  # 断言 A 大于等于 B 的所有元素
        assert_(np.all(self.A <= self.B))  # 断言 A 小于等于 B 的所有元素
        assert_(not np.any(self.A > self.B))  # 断言 A 没有比 B 大的元素
        assert_(not np.any(self.A < self.B))  # 断言 A 没有比 B 小的元素
        assert_(not np.any(self.A != self.B))  # 断言 A 和 B 的所有元素都相等

class TestChar:
    # 设置测试前的准备工作，初始化一个字符数组 A
    def setup_method(self):
        self.A = np.array('abc1', dtype='c').view(np.char.chararray)

    # 测试字符数组的方法
    def test_it(self):
        assert_equal(self.A.shape, (4,))  # 断言字符数组 A 的形状为 (4,)
        assert_equal(self.A.upper()[:2].tobytes(), b'AB')  # 断言字符数组 A 大写后前两个字符为 'AB'

class TestComparisons:
    # 设置测试前的准备工作，初始化两个二维字符数组 A 和 B
    def setup_method(self):
        self.A = np.array([['abc', 'abcc', '123'],
                           ['789', 'abc', 'xyz']]).view(np.char.chararray)
        self.B = np.array([['efg', 'efg', '123  '],
                           ['051', 'efgg', 'tuv']]).view(np.char.chararray)

    # 测试不等于的方法
    def test_not_equal(self):
        assert_array_equal((self.A != self.B),  # 断言 A 和 B 的不等性
                           [[True, True, False], [True, True, True]])

    # 测试等于的方法
    def test_equal(self):
        assert_array_equal((self.A == self.B),  # 断言 A 和 B 的等性
                           [[False, False, True], [False, False, False]])

    # 测试大于等于的方法
    def test_greater_equal(self):
        assert_array_equal((self.A >= self.B),  # 断言 A 大于等于 B
                           [[False, False, True], [True, False, True]])

    # 测试小于等于的方法
    def test_less_equal(self):
        assert_array_equal((self.A <= self.B),  # 断言 A 小于等于 B
                           [[True, True, True], [False, True, False]])

    # 测试大于的方法
    def test_greater(self):
        assert_array_equal((self.A > self.B),  # 断言 A 大于 B
                           [[False, False, False], [True, False, True]])

    # 测试小于的方法
    def test_less(self):
        assert_array_equal((self.A < self.B),  # 断言 A 小于 B
                           [[True, True, False], [False, True, False]])

    # 测试类型比较的方法
    def test_type(self):
        out1 = np.char.equal(self.A, self.B)  # 比较 A 和 B 的元素是否相等，返回布尔数组
        out2 = np.char.equal('a', 'a')  # 比较单个字符 'a' 是否等于 'a'，返回布尔值
        assert_(isinstance(out1, np.ndarray))  # 断言 out1 是 NumPy 数组类型
        assert_(isinstance(out2, np.ndarray))  # 断言 out2 是 NumPy 数组类型

class TestComparisonsMixed1(TestComparisons):
    """Ticket #1276"""

    # 继承自 TestComparisons，重写 setup_method 方法
    def setup_method(self):
        TestComparisons.setup_method(self)
        self.B = np.array(
            [['efg', 'efg', '123  '],
             ['051', 'efgg', 'tuv']], np.str_).view(np.char.chararray)

class TestComparisonsMixed2(TestComparisons):
    """Ticket #1276"""

    # 继承自 TestComparisons，重写 setup_method 方法
    def setup_method(self):
        TestComparisons.setup_method(self)
        self.A = np.array(
            [['abc', 'abcc', '123'],
             ['789', 'abc', 'xyz']], np.str_).view(np.char.chararray)
    # 设置测试方法的初始化操作
    def setup_method(self):
        # 创建包含字符串数组的 numpy 数组 A，每个元素都是 chararray 类型
        self.A = np.array([[' abc ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]) \
                            .view(np.char.chararray)
        # 创建包含字符串数组的 numpy 数组 B，每个元素都是 chararray 类型，包含 Unicode 字符
        self.B = np.array([[' \u03a3 ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]) \
                            .view(np.char.chararray)
        # 创建包含长字符串的 numpy 数组 C，每个元素都是 chararray 类型
        # 这些字符串的长度超过了代码中的 MEMCHR_CUT_OFF
        self.C = (np.array(['ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                            '01234567890123456789012345'])
                  .view(np.char.chararray))

    # 测试字符串长度的方法
    def test_len(self):
        # 断言返回的字符串长度数组的 dtype 是整数类型的子类
        assert_(issubclass(np.char.str_len(self.A).dtype.type, np.integer))
        # 断言计算得到的字符串长度数组与预期结果相等
        assert_array_equal(np.char.str_len(self.A), [[5, 0], [5, 9], [12, 5]])
        assert_array_equal(np.char.str_len(self.B), [[3, 0], [5, 9], [12, 5]])

    # 测试字符串计数的方法
    def test_count(self):
        # 断言返回的空字符计数数组的 dtype 是整数类型的子类
        assert_(issubclass(self.A.count('').dtype.type, np.integer))
        # 断言计算得到的字符计数数组与预期结果相等
        assert_array_equal(self.A.count('a'), [[1, 0], [0, 1], [0, 0]])
        assert_array_equal(self.A.count('123'), [[0, 0], [1, 0], [1, 0]])
        # Python 似乎不支持计算空字符（'\0'）的数量
        # assert_array_equal(self.A.count('\0'), [[0, 0], [0, 0], [1, 0]])
        assert_array_equal(self.A.count('a', 0, 2), [[1, 0], [0, 0], [0, 0]])
        assert_array_equal(self.B.count('a'), [[0, 0], [0, 1], [0, 0]])
        assert_array_equal(self.B.count('123'), [[0, 0], [1, 0], [1, 0]])
        # assert_array_equal(self.B.count('\0'), [[0, 0], [0, 0], [1, 0]])

    # 测试字符串后缀匹配的方法
    def test_endswith(self):
        # 断言返回的空字符串后缀匹配数组的 dtype 是布尔类型的子类
        assert_(issubclass(self.A.endswith('').dtype.type, np.bool))
        # 断言字符串是否以指定的后缀结束，结果与预期相等
        assert_array_equal(self.A.endswith(' '), [[1, 0], [0, 0], [1, 0]])
        assert_array_equal(self.A.endswith('3', 0, 3), [[0, 0], [1, 0], [1, 0]])

        # 定义一个失败的函数，用于测试 TypeError 异常
        def fail():
            self.A.endswith('3', 'fdjk')

        # 断言调用失败函数会抛出 TypeError 异常
        assert_raises(TypeError, fail)

    # 参数化测试字符串查找的方法
    @pytest.mark.parametrize(
        "dtype, encode",
        [("U", str),  # 使用 Unicode 编码类型
         ("S", lambda x: x.encode('ascii')),  # 使用 ASCII 编码类型的 lambda 函数
         ])
    def test_find(self, dtype, encode):
        # 将数组 A 转换为指定 dtype 的数组
        A = self.A.astype(dtype)
        # 断言返回的查找结果数组的 dtype 是整数类型的子类
        assert_(issubclass(A.find(encode('a')).dtype.type, np.integer))
        # 断言查找的结果数组与预期结果相等
        assert_array_equal(A.find(encode('a')),
                           [[1, -1], [-1, 6], [-1, -1]])
        assert_array_equal(A.find(encode('3')),
                           [[-1, -1], [2, -1], [2, -1]])
        assert_array_equal(A.find(encode('a'), 0, 2),
                           [[1, -1], [-1, -1], [-1, -1]])
        assert_array_equal(A.find([encode('1'), encode('P')]),
                           [[-1, -1], [0, -1], [0, 1]])
        # 将数组 C 转换为指定 dtype 的数组
        C = self.C.astype(dtype)
        # 断言查找的结果数组与预期结果相等
        assert_array_equal(C.find(encode('M')), [12, -1])
    # 定义测试方法 test_index，用于测试索引相关的方法

        # 定义内部方法 fail，用于测试在 self.A 中查找 'a' 是否会引发 ValueError 异常
        def fail():
            self.A.index('a')

        # 使用 assert_raises 断言，验证调用 fail 方法会抛出 ValueError 异常
        assert_raises(ValueError, fail)
        
        # 使用 assert_ 断言，验证 np.char.index 方法在字符串 'abcba' 中查找 'b' 的索引为 1
        assert_(np.char.index('abcba', 'b') == 1)
        
        # 使用 assert_ 断言，验证 np.char.index 方法返回的结果的 dtype 是 np.integer 类型的子类
        assert_(issubclass(np.char.index('abcba', 'b').dtype.type, np.integer))

    # 定义测试方法 test_isalnum，用于测试字符串是否由字母和数字组成
    def test_isalnum(self):
        # 使用 assert_ 断言，验证 self.A.isalnum() 返回的结果的 dtype 是 np.bool 类型的子类
        assert_(issubclass(self.A.isalnum().dtype.type, np.bool))
        
        # 使用 assert_array_equal 断言，验证 self.A.isalnum() 对象的返回结果与预期的 [[False, False], [True, True], [False, True]] 相等
        assert_array_equal(self.A.isalnum(), [[False, False], [True, True], [False, True]])

    # 定义测试方法 test_isalpha，用于测试字符串是否只包含字母
    def test_isalpha(self):
        # 使用 assert_ 断言，验证 self.A.isalpha() 返回的结果的 dtype 是 np.bool 类型的子类
        assert_(issubclass(self.A.isalpha().dtype.type, np.bool))
        
        # 使用 assert_array_equal 断言，验证 self.A.isalpha() 对象的返回结果与预期的 [[False, False], [False, True], [False, True]] 相等
        assert_array_equal(self.A.isalpha(), [[False, False], [False, True], [False, True]])

    # 定义测试方法 test_isdigit，用于测试字符串是否只包含数字
    def test_isdigit(self):
        # 使用 assert_ 断言，验证 self.A.isdigit() 返回的结果的 dtype 是 np.bool 类型的子类
        assert_(issubclass(self.A.isdigit().dtype.type, np.bool))
        
        # 使用 assert_array_equal 断言，验证 self.A.isdigit() 对象的返回结果与预期的 [[False, False], [True, False], [False, False]] 相等
        assert_array_equal(self.A.isdigit(), [[False, False], [True, False], [False, False]])

    # 定义测试方法 test_islower，用于测试字符串是否只包含小写字母
    def test_islower(self):
        # 使用 assert_ 断言，验证 self.A.islower() 返回的结果的 dtype 是 np.bool 类型的子类
        assert_(issubclass(self.A.islower().dtype.type, np.bool))
        
        # 使用 assert_array_equal 断言，验证 self.A.islower() 对象的返回结果与预期的 [[True, False], [False, False], [False, False]] 相等
        assert_array_equal(self.A.islower(), [[True, False], [False, False], [False, False]])

    # 定义测试方法 test_isspace，用于测试字符串是否只包含空格
    def test_isspace(self):
        # 使用 assert_ 断言，验证 self.A.isspace() 返回的结果的 dtype 是 np.bool 类型的子类
        assert_(issubclass(self.A.isspace().dtype.type, np.bool))
        
        # 使用 assert_array_equal 断言，验证 self.A.isspace() 对象的返回结果与预期的 [[False, False], [False, False], [False, False]] 相等
        assert_array_equal(self.A.isspace(), [[False, False], [False, False], [False, False]])

    # 定义测试方法 test_istitle，用于测试字符串是否符合 title 格式（每个单词首字母大写）
    def test_istitle(self):
        # 使用 assert_ 断言，验证 self.A.istitle() 返回的结果的 dtype 是 np.bool 类型的子类
        assert_(issubclass(self.A.istitle().dtype.type, np.bool))
        
        # 使用 assert_array_equal 断言，验证 self.A.istitle() 对象的返回结果与预期的 [[False, False], [False, False], [False, False]] 相等
        assert_array_equal(self.A.istitle(), [[False, False], [False, False], [False, False]])

    # 定义测试方法 test_isupper，用于测试字符串是否只包含大写字母
    def test_isupper(self):
        # 使用 assert_ 断言，验证 self.A.isupper() 返回的结果的 dtype 是 np.bool 类型的子类
        assert_(issubclass(self.A.isupper().dtype.type, np.bool))
        
        # 使用 assert_array_equal 断言，验证 self.A.isupper() 对象的返回结果与预期的 [[False, False], [False, False], [False, True]] 相等
        assert_array_equal(self.A.isupper(), [[False, False], [False, False], [False, True]])

    # 定义测试方法 test_rfind，用于测试字符串从右往左查找子字符串的索引
    def test_rfind(self):
        # 使用 assert_ 断言，验证 self.A.rfind('a') 返回的结果的 dtype 是 np.integer 类型的子类
        assert_(issubclass(self.A.rfind('a').dtype.type, np.integer))
        
        # 使用 assert_array_equal 断言，验证 self.A.rfind('a') 对象的返回结果与预期的 [[1, -1], [-1, 6], [-1, -1]] 相等
        assert_array_equal(self.A.rfind('a'), [[1, -1], [-1, 6], [-1, -1]])
        
        # 使用 assert_array_equal 断言，验证 self.A.rfind('3') 对象的返回结果与预期的 [[-1, -1], [2, -1], [6, -1]] 相等
        assert_array_equal(self.A.rfind('3'), [[-1, -1], [2, -1], [6, -1]])
        
        # 使用 assert_array_equal 断言，验证 self.A.rfind('a', 0, 2) 对象的返回结果与预期的 [[1, -1], [-1, -1], [-1, -1]] 相等
        assert_array_equal(self.A.rfind('a', 0, 2), [[1, -1], [-1, -1], [-1, -1]])
        
        # 使用 assert_array_equal 断言，验证 self.A.rfind(['1', 'P']) 对象的返回结果与预期的 [[-1, -1], [0, -1], [0, 2]] 相等
        assert_array_equal(self.A.rfind(['1', 'P']), [[-1, -1], [0, -1], [0, 2]])

    # 定义测试方法 test_rindex，用于测试字符串从右往左查找子字符串的索引
    def test_rindex(self):
        
        # 定义内部方法 fail，用于测试在 self.A 中查找 'a' 是否会引发 ValueError 异常
        def fail():
            self.A.rindex('a')

        # 使用 assert_raises 断言，验证调用 fail 方法会抛出 ValueError 异常
        assert_raises(ValueError, fail)
        
        # 使用 assert_ 断言，验证 np.char.rindex 方法在字符串 'abcba' 中查找 'b' 的索引为 3
        assert_(np.char.rindex('abcba', 'b') == 3)
        
        # 使用 assert_ 断言，验证 np.char.rindex 方法返回的结果的 dtype 是 np.integer 类型的子类
        assert_(issubclass(np.char.rindex('abcba', 'b').dtype.type, np.integer))

    # 定义测试方法 test_startswith，用于测试字符串是否以指定前缀开头
    def test_startswith(self):
        # 使用 assert_ 断言，验证 self.A.startswith('') 返回的结果的 dtype 是 np.bool 类型的子类
        assert_(issubclass(self.A.startswith('').dtype.type, np.bool))
        
        # 使用 assert_array_equal 断言，验证 self.A.startswith(' ') 对象的返回结果与预期的 [[1, 0], [0, 0], [0, 0]] 相等
        assert_array_equal(self.A.startswith(' '), [[1, 0], [0, 0], [0, 0]])
        
        # 使用 assert_array_equal 断言，验证 self.A.startswith('1', 0, 3) 对象的返回结果与
class TestMethods:
    # 设置测试方法的初始化函数
    def setup_method(self):
        # 创建字符串数组 A，使用 np.char.chararray 将其视为字符数组
        self.A = np.array([[' abc ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']],
                          dtype='S').view(np.char.chararray)
        # 创建字符串数组 B，使用 np.char.chararray 将其视为字符数组
        self.B = np.array([[' \u03a3 ', ''],
                           ['12345', 'MixedCase'],
                           ['123 \t 345 \0 ', 'UPPER']]).view(
                                                            np.char.chararray)

    # 测试 capitalize 方法
    def test_capitalize(self):
        # 期望的结果 tgt，使用 np.bytes_ 类型检查结果的数据类型
        tgt = [[b' abc ', b''],
               [b'12345', b'Mixedcase'],
               [b'123 \t 345 \0 ', b'Upper']]
        assert_(issubclass(self.A.capitalize().dtype.type, np.bytes_))
        # 检查 capitalize 方法的输出是否与期望的结果 tgt 一致
        assert_array_equal(self.A.capitalize(), tgt)

        # 期望的结果 tgt，使用 np.str_ 类型检查结果的数据类型
        tgt = [[' \u03c3 ', ''],
               ['12345', 'Mixedcase'],
               ['123 \t 345 \0 ', 'Upper']]
        assert_(issubclass(self.B.capitalize().dtype.type, np.str_))
        # 检查 capitalize 方法的输出是否与期望的结果 tgt 一致
        assert_array_equal(self.B.capitalize(), tgt)

    # 测试 center 方法
    def test_center(self):
        # 使用 np.bytes_ 类型检查 center 方法的输出的数据类型
        assert_(issubclass(self.A.center(10).dtype.type, np.bytes_))
        # 创建数组 C，检查 center 方法的输出是否符合预期的长度
        C = self.A.center([10, 20])
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        # 使用指定填充符号进行 center 方法测试
        C = self.A.center(20, b'#')
        # 检查填充后的字符串是否以指定字符开始和结束
        assert_(np.all(C.startswith(b'#')))
        assert_(np.all(C.endswith(b'#')))

        # 使用 np.char.center 函数测试指定字符串的居中效果
        C = np.char.center(b'FOO', [[10, 20], [15, 8]])
        # 期望的结果 tgt，使用 np.bytes_ 类型检查结果的数据类型
        tgt = [[b'   FOO    ', b'        FOO         '],
               [b'      FOO      ', b'  FOO   ']]
        assert_(issubclass(C.dtype.type, np.bytes_))
        # 检查 np.char.center 方法的输出是否与期望的结果 tgt 一致
        assert_array_equal(C, tgt)

    # 测试 decode 方法
    def test_decode(self):
        # 创建数组 A，使用 decode 方法解码 Unicode 转义字符
        A = np.char.array([b'\\u03a3'])
        assert_(A.decode('unicode-escape')[0] == '\u03a3')

    # 测试 encode 方法
    def test_encode(self):
        # 使用 encode 方法对数组 B 执行 Unicode 转义编码
        B = self.B.encode('unicode_escape')
        assert_(B[0][0] == str(' \\u03a3 ').encode('latin1'))

    # 测试 expandtabs 方法
    def test_expandtabs(self):
        # 使用 expandtabs 方法展开数组 A 中的制表符
        T = self.A.expandtabs()
        assert_(T[2, 0] == b'123      345 \0')

    # 测试 join 方法
    def test_join(self):
        # 注意：list(b'123') == [49, 50, 51]
        # b','.join(b'123') 在 Python 3 上会导致错误
        A0 = self.A.decode('ascii')

        # 使用 join 方法连接数组 A0 中的字符串，指定连接符为 ',' 和 '#'
        A = np.char.join([',', '#'], A0)
        assert_(issubclass(A.dtype.type, np.str_))
        # 期望的结果 tgt，使用 np.array 创建连接后的字符串数组
        tgt = np.array([[' ,a,b,c, ', ''],
                        ['1,2,3,4,5', 'M#i#x#e#d#C#a#s#e'],
                        ['1,2,3, ,\t, ,3,4,5, ,\x00, ', 'U#P#P#E#R']])
        # 检查 np.char.join 方法的输出是否与期望的结果 tgt 一致
        assert_array_equal(np.char.join([',', '#'], A0), tgt)
    def test_ljust(self):
        # 断言：确保 self.A.ljust(10) 返回的类型是 np.bytes_ 的子类
        assert_(issubclass(self.A.ljust(10).dtype.type, np.bytes_))

        # 将 self.A 分别左对齐到长度为 10 和 20 的数组 C，并检查字符长度
        C = self.A.ljust([10, 20])
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        # 使用字符 '#' 将 self.A 左对齐到长度为 20 的数组 C，并检查开头和结尾
        C = self.A.ljust(20, b'#')
        assert_array_equal(C.startswith(b'#'), [
                [False, True], [False, False], [False, False]])
        assert_(np.all(C.endswith(b'#')))

        # 使用 np.char.ljust 将字节串 b'FOO' 左对齐到不同长度的数组，并断言结果是否正确
        C = np.char.ljust(b'FOO', [[10, 20], [15, 8]])
        tgt = [[b'FOO       ', b'FOO                 '],
               [b'FOO            ', b'FOO     ']]
        assert_(issubclass(C.dtype.type, np.bytes_))
        assert_array_equal(C, tgt)

    def test_lower(self):
        # 定义预期的小写转换结果并断言 self.A.lower() 的返回类型
        tgt = [[b' abc ', b''],
               [b'12345', b'mixedcase'],
               [b'123 \t 345 \0 ', b'upper']]
        assert_(issubclass(self.A.lower().dtype.type, np.bytes_))
        assert_array_equal(self.A.lower(), tgt)

        # 定义另一组预期的小写转换结果并断言 self.B.lower() 的返回类型
        tgt = [[' \u03c3 ', ''],
               ['12345', 'mixedcase'],
               ['123 \t 345 \0 ', 'upper']]
        assert_(issubclass(self.B.lower().dtype.type, np.str_))
        assert_array_equal(self.B.lower(), tgt)

    def test_lstrip(self):
        # 定义预期的去除左空格结果并断言 self.A.lstrip() 的返回类型
        tgt = [[b'abc ', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345 \0 ', b'UPPER']]
        assert_(issubclass(self.A.lstrip().dtype.type, np.bytes_))
        assert_array_equal(self.A.lstrip(), tgt)

        # 定义另一组预期的去除左字符 '1' 和 'M' 结果并断言 self.A.lstrip([b'1', b'M']) 的返回类型
        tgt = [[b' abc', b''],
               [b'2345', b'ixedCase'],
               [b'23 \t 345 \x00', b'UPPER']]
        assert_array_equal(self.A.lstrip([b'1', b'M']), tgt)

        # 定义另一组预期的去除左空格结果并断言 self.B.lstrip() 的返回类型
        tgt = [['\u03a3 ', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345 \0 ', 'UPPER']]
        assert_(issubclass(self.B.lstrip().dtype.type, np.str_))
        assert_array_equal(self.B.lstrip(), tgt)

    def test_partition(self):
        # 使用 [b'3', b'M'] 将 self.A 分区并断言返回结果的类型
        P = self.A.partition([b'3', b'M'])
        tgt = [[(b' abc ', b'', b''), (b'', b'', b'')],
               [(b'12', b'3', b'45'), (b'', b'M', b'ixedCase')],
               [(b'12', b'3', b' \t 345 \0 '), (b'UPPER', b'', b'')]]
        assert_(issubclass(P.dtype.type, np.bytes_))
        assert_array_equal(P, tgt)
    # 定义一个测试方法，用于测试字符串数组的替换操作
    def test_replace(self):
        # 调用数组对象的替换方法，将指定的子串替换为新的子串
        R = self.A.replace([b'3', b'a'],
                           [b'##########', b'@'])
        # 定义预期结果数组，用于验证替换操作的正确性
        tgt = [[b' abc ', b''],
               [b'12##########45', b'MixedC@se'],
               [b'12########## \t ##########45 \x00 ', b'UPPER']]
        # 断言替换后的数组的数据类型是 np.bytes_ 的子类
        assert_(issubclass(R.dtype.type, np.bytes_))
        # 断言替换后的数组与预期结果数组相等
        assert_array_equal(R, tgt)
        
        # 测试一些特殊情况，这些情况下应返回输入数组，因为替换是不可能或没有效果的
        # 测试替换长字节串时应返回原始数组
        S1 = self.A.replace(b'A very long byte string, longer than A', b'')
        assert_array_equal(S1, self.A)
        # 测试空串替换为空串应返回原始数组
        S2 = self.A.replace(b'', b'')
        assert_array_equal(S2, self.A)
        # 测试不做任何替换的情况下应返回原始数组
        S3 = self.A.replace(b'3', b'3')
        assert_array_equal(S3, self.A)
        # 测试不做任何替换的情况下应返回原始数组
        S4 = self.A.replace(b'3', b'', count=0)
        assert_array_equal(S4, self.A)

    # 定义一个测试方法，用于测试替换操作的计数和大小
    def test_replace_count_and_size(self):
        # 创建一个字符串数组对象 a，包含四个不同长度的字符串
        a = np.array(['0123456789' * i for i in range(4)]).view(np.char.chararray)
        # 对数组进行替换操作，将指定的子串替换为新的子串
        r1 = a.replace('5', 'ABCDE')
        # 断言替换后的数组元素大小符合预期
        assert r1.dtype.itemsize == (3*10 + 3*4) * 4
        # 断言替换后的数组与预期结果数组相等
        assert_array_equal(r1, np.array(['01234ABCDE6789' * i for i in range(4)]))
        
        # 对数组进行替换操作，指定替换的次数
        r2 = a.replace('5', 'ABCDE', count=1)
        # 断言替换后的数组元素大小符合预期
        assert r2.dtype.itemsize == (3*10 + 4) * 4
        
        # 对数组进行替换操作，不进行任何替换
        r3 = a.replace('5', 'ABCDE', count=0)
        # 断言替换后的数组元素大小与原始数组相同
        assert r3.dtype.itemsize == a.dtype.itemsize
        # 断言替换后的数组与原始数组相等
        assert_array_equal(r3, a)
        
        # 使用负数值表示替换所有匹配项
        r4 = a.replace('5', 'ABCDE', count=-1)
        # 断言替换后的数组元素大小符合预期
        assert r4.dtype.itemsize == (3*10 + 3*4) * 4
        # 断言替换后的数组与 r1 相等
        assert_array_equal(r4, r1)
        
        # 可以对每个元素进行独立的替换计数
        r5 = a.replace('5', 'ABCDE', count=[-1, -1, -1, 1])
        # 断言替换后的数组元素大小符合预期
        assert r5.dtype.itemsize == (3*10 + 4) * 4
        # 断言替换后的数组与预期结果数组相等
        assert_array_equal(r5, np.array(['01234ABCDE6789' * i for i in range(3)] + ['01234ABCDE6789' + '0123456789' * 2]))

    # 定义一个测试方法，用于测试替换操作的广播特性
    def test_replace_broadcasting(self):
        # 创建一个字符串数组对象 a
        a = np.array('0,0,0').view(np.char.chararray)
        # 对数组进行替换操作，替换指定子串为新的子串，并指定每个替换操作的次数
        r1 = a.replace('0', '1', count=np.arange(3))
        # 断言替换后的数组数据类型与原始数组相同
        assert r1.dtype == a.dtype
        # 断言替换后的数组与预期结果数组相等
        assert_array_equal(r1, np.array(['0,0,0', '1,0,0', '1,1,0']))
        
        # 对数组进行替换操作，使用二维数组作为新的子串进行替换，并指定每个替换操作的次数
        r2 = a.replace('0', [['1'], ['2']], count=np.arange(1, 4))
        # 断言替换后的数组与预期结果数组相等
        assert_array_equal(r2, np.array([['1,0,0', '1,1,0', '1,1,1'], ['2,0,0', '2,2,0', '2,2,2']]))
        
        # 对数组进行替换操作，替换多个子串为新的子串
        r3 = a.replace(['0', '0,0', '0,0,0'], 'X')
        # 断言替换后的数组与预期结果数组相等
        assert_array_equal(r3, np.array(['X,X,X', 'X,0', 'X']))
    def test_rjust(self):
        # 断言：self.A.rjust(10) 的返回值的类型是 np.bytes_
        assert_(issubclass(self.A.rjust(10).dtype.type, np.bytes_))

        # 将 self.A.rjust([10, 20]) 的结果赋给 C
        C = self.A.rjust([10, 20])
        # 断言：np.char.str_len(C) 的结果应与 [[10, 20], [10, 20], [12, 20]] 相等
        assert_array_equal(np.char.str_len(C), [[10, 20], [10, 20], [12, 20]])

        # 将 self.A.rjust(20, b'#') 的结果赋给 C
        C = self.A.rjust(20, b'#')
        # 断言：C 中所有元素均以 b'#' 开头
        assert_(np.all(C.startswith(b'#')))
        # 断言：C 的元素以 [[False, True], [False, False], [False, False]] 结尾
        assert_array_equal(C.endswith(b'#'),
                           [[False, True], [False, False], [False, False]])

        # 将 np.char.rjust(b'FOO', [[10, 20], [15, 8]]) 的结果赋给 C
        C = np.char.rjust(b'FOO', [[10, 20], [15, 8]])
        # 预期的结果 tgt
        tgt = [[b'       FOO', b'                 FOO'],
               [b'            FOO', b'     FOO']]
        # 断言：C 的类型是 np.bytes_
        assert_(issubclass(C.dtype.type, np.bytes_))
        # 断言：C 应与 tgt 相等
        assert_array_equal(C, tgt)

    def test_rpartition(self):
        # 将 self.A.rpartition([b'3', b'M']) 的结果赋给 P
        P = self.A.rpartition([b'3', b'M'])
        # 预期的结果 tgt
        tgt = [[(b'', b'', b' abc '), (b'', b'', b'')],
               [(b'12', b'3', b'45'), (b'', b'M', b'ixedCase')],
               [(b'123 \t ', b'3', b'45 \0 '), (b'', b'', b'UPPER')]]
        # 断言：P 的类型是 np.bytes_
        assert_(issubclass(P.dtype.type, np.bytes_))
        # 断言：P 应与 tgt 相等
        assert_array_equal(P, tgt)

    def test_rsplit(self):
        # 将 self.A.rsplit(b'3') 的结果赋给 A
        A = self.A.rsplit(b'3')
        # 预期的结果 tgt
        tgt = [[[b' abc '], [b'']],
               [[b'12', b'45'], [b'MixedCase']],
               [[b'12', b' \t ', b'45 \x00 '], [b'UPPER']]]
        # 断言：A 的类型是 np.object_
        assert_(issubclass(A.dtype.type, np.object_))
        # 断言：A 应与 tgt 相等
        assert_equal(A.tolist(), tgt)

    def test_rstrip(self):
        # 断言：self.A.rstrip() 的返回值的类型是 np.bytes_
        assert_(issubclass(self.A.rstrip().dtype.type, np.bytes_))

        # 预期的结果 tgt
        tgt = [[b' abc', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345', b'UPPER']]
        # 断言：self.A.rstrip() 应与 tgt 相等
        assert_array_equal(self.A.rstrip(), tgt)

        # 预期的结果 tgt
        tgt = [[b' abc ', b''],
               [b'1234', b'MixedCase'],
               [b'123 \t 345 \x00', b'UPP']
               ]
        # 断言：self.A.rstrip([b'5', b'ER']) 应与 tgt 相等
        assert_array_equal(self.A.rstrip([b'5', b'ER']), tgt)

        # 预期的结果 tgt
        tgt = [[' \u03a3', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345', 'UPPER']]
        # 断言：self.B.rstrip() 的返回值的类型是 np.str_
        assert_(issubclass(self.B.rstrip().dtype.type, np.str_))
        # 断言：self.B.rstrip() 应与 tgt 相等
        assert_array_equal(self.B.rstrip(), tgt)

    def test_strip(self):
        # 预期的结果 tgt
        tgt = [[b'abc', b''],
               [b'12345', b'MixedCase'],
               [b'123 \t 345', b'UPPER']]
        # 断言：self.A.strip() 的返回值的类型是 np.bytes_
        assert_(issubclass(self.A.strip().dtype.type, np.bytes_))
        # 断言：self.A.strip() 应与 tgt 相等
        assert_array_equal(self.A.strip(), tgt)

        # 预期的结果 tgt
        tgt = [[b' abc ', b''],
               [b'234', b'ixedCas'],
               [b'23 \t 345 \x00', b'UPP']]
        # 断言：self.A.strip([b'15', b'EReM']) 应与 tgt 相等
        assert_array_equal(self.A.strip([b'15', b'EReM']), tgt)

        # 预期的结果 tgt
        tgt = [['\u03a3', ''],
               ['12345', 'MixedCase'],
               ['123 \t 345', 'UPPER']]
        # 断言：self.B.strip() 的返回值的类型是 np.str_
        assert_(issubclass(self.B.strip().dtype.type, np.str_))
        # 断言：self.B.strip() 应与 tgt 相等
        assert_array_equal(self.B.strip(), tgt)

    def test_split(self):
        # 将 self.A.split(b'3') 的结果赋给 A
        A = self.A.split(b'3')
        # 预期的结果 tgt
        tgt = [
               [[b' abc '], [b'']],
               [[b'12', b'45'], [b'MixedCase']],
               [[b'12', b' \t ', b'45 \x00 '], [b'UPPER']]]
        # 断言：A 的类型是 np.object_
        assert_(issubclass(A.dtype.type, np.object_))
        # 断言：A 应与 tgt 相等
        assert_equal(A.tolist(), tgt)
    # 定义一个测试方法，测试 np.char.array 的 splitlines() 方法
    def test_splitlines(self):
        # 创建一个包含单个字符串 'abc\nfds\nwer' 的 np.char.array 对象，并对其调用 splitlines() 方法
        A = np.char.array(['abc\nfds\nwer']).splitlines()
        # 断言 A.dtype.type 是 np.object_ 的子类
        assert_(issubclass(A.dtype.type, np.object_))
        # 断言 A 的形状为 (1,)
        assert_(A.shape == (1,))
        # 断言 A[0] 的长度为 3
        assert_(len(A[0]) == 3)

    # 定义一个测试方法，测试 self.A 的 swapcase() 方法
    def test_swapcase(self):
        # 预期的目标结果列表，包含两个子列表，每个子列表包含两个 bytes 对象
        tgt = [[b' ABC ', b''],
               [b'12345', b'mIXEDcASE'],
               [b'123 \t 345 \0 ', b'upper']]
        # 断言 self.A.swapcase() 返回的数组的 dtype.type 是 np.bytes_ 的子类
        assert_(issubclass(self.A.swapcase().dtype.type, np.bytes_))
        # 断言 self.A.swapcase() 返回的数组与预期的 tgt 相等
        assert_array_equal(self.A.swapcase(), tgt)

        # 预期的目标结果列表，包含两个子列表，每个子列表包含两个字符串对象
        tgt = [[' \u03c3 ', ''],
               ['12345', 'mIXEDcASE'],
               ['123 \t 345 \0 ', 'upper']]
        # 断言 self.B.swapcase() 返回的数组的 dtype.type 是 np.str_ 的子类
        assert_(issubclass(self.B.swapcase().dtype.type, np.str_))
        # 断言 self.B.swapcase() 返回的数组与预期的 tgt 相等
        assert_array_equal(self.B.swapcase(), tgt)

    # 定义一个测试方法，测试 self.A 的 title() 方法
    def test_title(self):
        # 预期的目标结果列表，包含两个子列表，每个子列表包含两个 bytes 对象
        tgt = [[b' Abc ', b''],
               [b'12345', b'Mixedcase'],
               [b'123 \t 345 \0 ', b'Upper']]
        # 断言 self.A.title() 返回的数组的 dtype.type 是 np.bytes_ 的子类
        assert_(issubclass(self.A.title().dtype.type, np.bytes_))
        # 断言 self.A.title() 返回的数组与预期的 tgt 相等
        assert_array_equal(self.A.title(), tgt)

        # 预期的目标结果列表，包含两个子列表，每个子列表包含两个字符串对象
        tgt = [[' \u03a3 ', ''],
               ['12345', 'Mixedcase'],
               ['123 \t 345 \0 ', 'Upper']]
        # 断言 self.B.title() 返回的数组的 dtype.type 是 np.str_ 的子类
        assert_(issubclass(self.B.title().dtype.type, np.str_))
        # 断言 self.B.title() 返回的数组与预期的 tgt 相等
        assert_array_equal(self.B.title(), tgt)

    # 定义一个测试方法，测试 self.A 的 upper() 方法
    def test_upper(self):
        # 预期的目标结果列表，包含两个子列表，每个子列表包含两个 bytes 对象
        tgt = [[b' ABC ', b''],
               [b'12345', b'MIXEDCASE'],
               [b'123 \t 345 \0 ', b'UPPER']]
        # 断言 self.A.upper() 返回的数组的 dtype.type 是 np.bytes_ 的子类
        assert_(issubclass(self.A.upper().dtype.type, np.bytes_))
        # 断言 self.A.upper() 返回的数组与预期的 tgt 相等
        assert_array_equal(self.A.upper(), tgt)

        # 预期的目标结果列表，包含两个子列表，每个子列表包含两个字符串对象
        tgt = [[' \u03a3 ', ''],
               ['12345', 'MIXEDCASE'],
               ['123 \t 345 \0 ', 'UPPER']]
        # 断言 self.B.upper() 返回的数组的 dtype.type 是 np.str_ 的子类
        assert_(issubclass(self.B.upper().dtype.type, np.str_))
        # 断言 self.B.upper() 返回的数组与预期的 tgt 相等
        assert_array_equal(self.B.upper(), tgt)

    # 定义一个测试方法，测试 self.B 的 isnumeric() 方法
    def test_isnumeric(self):

        # 定义一个内部函数 fail()，尝试调用 self.A 的 isnumeric() 方法，预期会抛出 TypeError 异常
        def fail():
            self.A.isnumeric()

        # 断言调用 fail() 会抛出 TypeError 异常
        assert_raises(TypeError, fail)
        # 断言 self.B.isnumeric() 返回的数组的 dtype.type 是 np.bool 的子类
        assert_(issubclass(self.B.isnumeric().dtype.type, np.bool))
        # 断言 self.B.isnumeric() 返回的数组与预期的二维 bool 数组相等
        assert_array_equal(self.B.isnumeric(), [
                [False, False], [True, False], [False, False]])

    # 定义一个测试方法，测试 self.B 的 isdecimal() 方法
    def test_isdecimal(self):

        # 定义一个内部函数 fail()，尝试调用 self.A 的 isdecimal() 方法，预期会抛出 TypeError 异常
        def fail():
            self.A.isdecimal()

        # 断言调用 fail() 会抛出 TypeError 异常
        assert_raises(TypeError, fail)
        # 断言 self.B.isdecimal() 返回的数组的 dtype.type 是 np.bool 的子类
        assert_(issubclass(self.B.isdecimal().dtype.type, np.bool))
        # 断言 self.B.isdecimal() 返回的数组与预期的二维 bool 数组相等
        assert_array_equal(self.B.isdecimal(), [
                [False, False], [True, False], [False, False]])
# 定义一个测试类 TestOperations
class TestOperations:
    
    # 在每个测试方法执行前调用，初始化数组 A 和 B
    def setup_method(self):
        self.A = np.array([['abc', '123'],
                           ['789', 'xyz']]).view(np.char.chararray)
        self.B = np.array([['efg', '456'],
                           ['051', 'tuv']]).view(np.char.chararray)

    # 测试字符串数组的加法操作
    def test_add(self):
        # 预期的加法结果 AB
        AB = np.array([['abcefg', '123456'],
                       ['789051', 'xyztuv']]).view(np.char.chararray)
        # 断言两个数组是否相等
        assert_array_equal(AB, (self.A + self.B))
        # 断言加法结果的第一个元素的第一个字符的长度是否为6
        assert_(len((self.A + self.B)[0][0]) == 6)

    # 测试字符串数组的右加操作
    def test_radd(self):
        # 预期的右加结果 QA
        QA = np.array([['qabc', 'q123'],
                       ['q789', 'qxyz']]).view(np.char.chararray)
        # 断言两个数组是否相等
        assert_array_equal(QA, ('q' + self.A))

    # 测试字符串数组的乘法操作
    def test_mul(self):
        A = self.A
        # 遍历不同的乘数 r 进行测试
        for r in (2, 3, 5, 7, 197):
            # 预期的乘法结果 Ar
            Ar = np.array([[A[0, 0]*r, A[0, 1]*r],
                           [A[1, 0]*r, A[1, 1]*r]]).view(np.char.chararray)
            # 断言两个数组是否相等
            assert_array_equal(Ar, (self.A * r))

        # 对于不支持的对象类型进行异常处理
        for ob in [object(), 'qrs']:
            # 断言抛出 ValueError 异常
            with assert_raises_regex(ValueError,
                                     'Can only multiply by integers'):
                A*ob

    # 测试字符串数组的右乘操作
    def test_rmul(self):
        A = self.A
        # 遍历不同的乘数 r 进行测试
        for r in (2, 3, 5, 7, 197):
            # 预期的右乘结果 Ar
            Ar = np.array([[A[0, 0]*r, A[0, 1]*r],
                           [A[1, 0]*r, A[1, 1]*r]]).view(np.char.chararray)
            # 断言两个数组是否相等
            assert_array_equal(Ar, (r * self.A))

        # 对于不支持的对象类型进行异常处理
        for ob in [object(), 'qrs']:
            # 断言抛出 ValueError 异常
            with assert_raises_regex(ValueError,
                                     'Can only multiply by integers'):
                ob * A

    # 测试字符串数组的模运算操作
    def test_mod(self):
        """Ticket #856"""
        # 定义格式化字符串数组 F 和整数数组 C
        F = np.array([['%d', '%f'], ['%s', '%r']]).view(np.char.chararray)
        C = np.array([[3, 7], [19, 1]], dtype=np.int64)
        # 预期的模运算结果 FC
        FC = np.array([['3', '7.000000'],
                       ['19', 'np.int64(1)']]).view(np.char.chararray)
        # 断言两个数组是否相等
        assert_array_equal(FC, F % C)

        # 定义格式化字符串数组 A 和整数 1
        A = np.array([['%.3f', '%d'], ['%s', '%r']]).view(np.char.chararray)
        # 预期的模运算结果 A1
        A1 = np.array([['1.000', '1'],
                       ['1', repr(np.array(1)[()])]]).view(np.char.chararray)
        # 断言两个数组是否相等
        assert_array_equal(A1, (A % 1))

        # 预期的模运算结果 A2
        A2 = np.array([['1.000', '2'],
                       ['3', repr(np.array(4)[()])]]).view(np.char.chararray)
        # 断言两个数组是否相等
        assert_array_equal(A2, (A % [[1, 2], [3, 4]]))

    # 测试字符串数组的右模运算操作
    def test_rmod(self):
        # 断言字符串数组 A 的字符串格式化结果是否等于其字符串表示形式
        assert_(("%s" % self.A) == str(self.A))
        # 断言字符串数组 A 的原始表示形式是否等于其 repr 表示形式
        assert_(("%r" % self.A) == repr(self.A))

        # 对于不支持的操作数类型进行异常处理
        for ob in [42, object()]:
            # 断言抛出 TypeError 异常
            with assert_raises_regex(
                    TypeError, "unsupported operand type.* and 'chararray'"):
                ob % self.A
    def test_slice(self):
        """Regression test for https://github.com/numpy/numpy/issues/5982"""
        
        # 创建一个二维的字符数组 arr，每个元素长度为 4，视图转换为 chararray 类型
        arr = np.array([['abc ', 'def '], ['geh ', 'ijk ']],
                       dtype='S4').view(np.char.chararray)
        
        # 对整个数组进行切片，sl1 是 arr 的一个视图
        sl1 = arr[:]
        # 断言 sl1 与 arr 相等
        assert_array_equal(sl1, arr)
        # 断言 sl1 的基础对象是 arr
        assert_(sl1.base is arr)
        # 断言 sl1 的基础对象的基础对象是 arr 的基础对象
        assert_(sl1.base.base is arr.base)

        # 对整个数组进行完整切片，sl2 是 arr 的一个视图
        sl2 = arr[:, :]
        # 断言 sl2 与 arr 相等
        assert_array_equal(sl2, arr)
        # 断言 sl2 的基础对象是 arr
        assert_(sl2.base is arr)
        # 断言 sl2 的基础对象的基础对象是 arr 的基础对象
        assert_(sl2.base.base is arr.base)

        # 断言 arr 的第一个元素的值是 b'abc'
        assert_(arr[0, 0] == b'abc')

    @pytest.mark.parametrize('data', [['plate', '   ', 'shrimp'],
                                      [b'retro', b'  ', b'encabulator']])
    def test_getitem_length_zero_item(self, data):
        # Regression test for gh-26375.
        
        # 使用 data 创建一个字符数组 a
        a = np.char.array(data)
        
        # a.dtype.type() 将是一个空字符串或者 bytes 实例。
        # 如果 a[1] 类型不正确或长度不为 0，相等性测试将失败。
        # 断言 a[1] 等于 a 的 dtype 的类型
        assert_equal(a[1], a.dtype.type())
class TestMethodsEmptyArray:
    # 设置每个测试方法的初始化方法
    def setup_method(self):
        # 创建一个空的 Unicode 类型的 NumPy 数组
        self.U = np.array([], dtype='U')
        # 创建一个空的字节串（Bytes）类型的 NumPy 数组
        self.S = np.array([], dtype='S')

    # 测试 np.char.encode 方法
    def test_encode(self):
        # 对空的 Unicode 数组进行编码
        res = np.char.encode(self.U)
        # 断言编码后结果与空列表相等
        assert_array_equal(res, [])
        # 断言编码后结果的数据类型为字节串
        assert_(res.dtype.char == 'S')

    # 测试 np.char.decode 方法
    def test_decode(self):
        # 对空的字节串数组进行解码
        res = np.char.decode(self.S)
        # 断言解码后结果与空列表相等
        assert_array_equal(res, [])
        # 断言解码后结果的数据类型为 Unicode
        assert_(res.dtype.char == 'U')

    # 测试带有 reshape 的 np.char.decode 方法
    def test_decode_with_reshape(self):
        # 对空的字节串数组进行 reshape 并解码
        res = np.char.decode(self.S.reshape((1, 0, 1)))
        # 断言结果数组的形状为 (1, 0, 1)
        assert_(res.shape == (1, 0, 1))


class TestMethodsScalarValues:
    # 测试 np.char.mod 方法
    def test_mod(self):
        # 创建一个字节串类型的 NumPy 数组 A
        A = np.array([[' abc ', ''],
                      ['12345', 'MixedCase'],
                      ['123 \t 345 \0 ', 'UPPER']], dtype='S')
        # 预期的目标数组 tgt
        tgt = [[b'123 abc ', b'123'],
               [b'12312345', b'123MixedCase'],
               [b'123123 \t 345 \0 ', b'123UPPER']]
        # 断言使用 np.char.mod 方法后得到的结果与预期的目标数组相等
        assert_array_equal(np.char.mod(b"123%s", A), tgt)

    # 测试 np.char.decode 方法
    def test_decode(self):
        # 给定一个字节串
        bytestring = b'\x81\xc1\x81\xc1\x81\xc1'
        # 使用 cp037 编码解码字节串，预期结果为 'aAaAaA'
        assert_equal(np.char.decode(bytestring, encoding='cp037'),
                     'aAaAaA')

    # 测试 np.char.encode 方法
    def test_encode(self):
        # 给定一个 Unicode 字符串
        unicode = 'aAaAaA'
        # 使用 cp037 编码 Unicode 字符串，预期结果为对应的字节串
        assert_equal(np.char.encode(unicode, encoding='cp037'),
                     b'\x81\xc1\x81\xc1\x81\xc1')

    # 测试 np.char.expandtabs 方法
    def test_expandtabs(self):
        # 给定一个带有制表符的字符串 s
        s = "\tone level of indentation\n\t\ttwo levels of indentation"
        # 断言使用 np.char.expandtabs 方法后的结果与预期的字符串相等
        assert_equal(
            np.char.expandtabs(s, tabsize=2),
            "  one level of indentation\n    two levels of indentation"
        )

    # 测试 np.char.join 方法
    def test_join(self):
        # 给定分隔符数组 seps 和字符串 'hello'
        seps = np.array(['-', '_'])
        # 断言使用 np.char.join 方法后的结果与预期的列表相等
        assert_array_equal(np.char.join(seps, 'hello'),
                           ['h-e-l-l-o', 'h_e_l_l_o'])

    # 测试 np.char.partition 方法
    def test_partition(self):
        # 断言使用 np.char.partition 方法后的结果与预期的列表相等
        assert_equal(np.char.partition('This string', ' '),
                     ['This', ' ', 'string'])

    # 测试 np.char.rpartition 方法
    def test_rpartition(self):
        # 断言使用 np.char.rpartition 方法后的结果与预期的列表相等
        assert_equal(np.char.rpartition('This string here', ' '),
                     ['This string', ' ', 'here'])

    # 测试 np.char.replace 方法
    def test_replace(self):
        # 断言使用 np.char.replace 方法后的结果与预期的字符串相等
        assert_equal(np.char.replace('Python is good', 'good', 'great'),
                     'Python is great')


def test_empty_indexing():
    """Regression test for ticket 1948."""
    # 检查使用空列表索引 chararray 是否返回空的 chararray 而不是包含一个空字符串的 chararray
    s = np.char.chararray((4,))
    assert_(s[[]].size == 0)
```