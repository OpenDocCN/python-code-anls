# `.\numpy\numpy\_core\tests\test_arrayprint.py`

```py
# 导入必要的模块和库
import sys  # 导入系统相关的功能模块
import gc  # 导入垃圾回收模块
from hypothesis import given  # 从假设测试框架导入给定装饰器
from hypothesis.extra import numpy as hynp  # 导入假设测试框架的NumPy扩展
import pytest  # 导入 pytest 测试框架

import numpy as np  # 导入 NumPy 库
from numpy.testing import (  # 导入 NumPy 测试工具
    assert_, assert_equal, assert_raises, assert_warns, HAS_REFCOUNT,
    assert_raises_regex,
    )
from numpy._core.arrayprint import _typelessdata  # 导入 NumPy 数组打印相关功能
import textwrap  # 导入文本包装模块

class TestArrayRepr:
    def test_nan_inf(self):
        # 创建包含 NaN 和 Inf 的 NumPy 数组
        x = np.array([np.nan, np.inf])
        assert_equal(repr(x), 'array([nan, inf])')  # 断言数组的 repr 字符串

    def test_subclass(self):
        # 定义一个继承自 np.ndarray 的子类
        class sub(np.ndarray): pass

        # 一维数组示例
        x1d = np.array([1, 2]).view(sub)
        assert_equal(repr(x1d), 'sub([1, 2])')  # 断言子类数组的 repr 字符串

        # 二维数组示例
        x2d = np.array([[1, 2], [3, 4]]).view(sub)
        assert_equal(repr(x2d),
            'sub([[1, 2],\n'
            '     [3, 4]])')  # 断言子类二维数组的 repr 字符串

        # 带有灵活数据类型的二维数组示例
        xstruct = np.ones((2,2), dtype=[('a', '<i4')]).view(sub)
        assert_equal(repr(xstruct),
            "sub([[(1,), (1,)],\n"
            "     [(1,), (1,)]]"  # 断言带有结构化数据类型的子类数组的 repr 字符串
            ", dtype=[('a', '<i4')])"
        )

    @pytest.mark.xfail(reason="See gh-10544")  # 标记此测试用例为预期失败的情况
    def test_object_subclass(self):
        # 定义一个继承自 np.ndarray 的对象子类
        class sub(np.ndarray):
            def __new__(cls, inp):
                obj = np.asarray(inp).view(cls)
                return obj

            def __getitem__(self, ind):
                ret = super().__getitem__(ind)
                return sub(ret)

        # 测试对象 + 子类是否正常工作
        x = sub([None, None])
        assert_equal(repr(x), 'sub([None, None], dtype=object)')  # 断言对象子类的 repr 字符串
        assert_equal(str(x), '[None None]')  # 断言对象子类的字符串表示

        x = sub([None, sub([None, None])])
        assert_equal(repr(x),
            'sub([None, sub([None, None], dtype=object)], dtype=object)')  # 断言嵌套对象子类的 repr 字符串
        assert_equal(str(x), '[None sub([None, None], dtype=object)]')  # 断言嵌套对象子类的字符串表示
    def test_0d_object_subclass(self):
        # 确保返回 0ds 而不是标量的子类不会在 str 方法中引起无限递归

        class sub(np.ndarray):
            def __new__(cls, inp):
                # 将输入转换为 ndarray 并视图化为当前类的对象
                obj = np.asarray(inp).view(cls)
                return obj

            def __getitem__(self, ind):
                # 调用父类的 __getitem__ 方法获取索引处的值
                ret = super().__getitem__(ind)
                # 将获取的值转换为当前类的对象并返回
                return sub(ret)

        x = sub(1)
        assert_equal(repr(x), 'sub(1)')
        assert_equal(str(x), '1')

        x = sub([1, 1])
        assert_equal(repr(x), 'sub([1, 1])')
        assert_equal(str(x), '[1 1]')

        # 检查它在对象数组中的使用是否正常工作
        x = sub(None)
        assert_equal(repr(x), 'sub(None, dtype=object)')
        assert_equal(str(x), 'None')

        # 测试递归的对象数组（即使深度 > 1）
        y = sub(None)
        x[()] = y
        y[()] = x
        assert_equal(repr(x),
            'sub(sub(sub(..., dtype=object), dtype=object), dtype=object)')
        assert_equal(str(x), '...')
        x[()] = 0  # 解决循环引用以便垃圾收集器处理

        # 嵌套的 0d 子类对象
        x = sub(None)
        x[()] = sub(None)
        assert_equal(repr(x), 'sub(sub(None, dtype=object), dtype=object)')
        assert_equal(str(x), 'None')

        # gh-10663
        class DuckCounter(np.ndarray):
            def __getitem__(self, item):
                result = super().__getitem__(item)
                if not isinstance(result, DuckCounter):
                    result = result[...].view(DuckCounter)
                return result

            def to_string(self):
                return {0: 'zero', 1: 'one', 2: 'two'}.get(self.item(), 'many')

            def __str__(self):
                if self.shape == ():
                    return self.to_string()
                else:
                    fmt = {'all': lambda x: x.to_string()}
                    return np.array2string(self, formatter=fmt)

        dc = np.arange(5).view(DuckCounter)
        assert_equal(str(dc), "[zero one two many many]")
        assert_equal(str(dc[0]), "zero")

    def test_self_containing(self):
        # 测试包含自身引用的情况

        arr0d = np.array(None)
        arr0d[()] = arr0d
        assert_equal(repr(arr0d),
            'array(array(..., dtype=object), dtype=object)')
        arr0d[()] = 0  # 解决循环引用以便垃圾收集器处理

        arr1d = np.array([None, None])
        arr1d[1] = arr1d
        assert_equal(repr(arr1d),
            'array([None, array(..., dtype=object)], dtype=object)')
        arr1d[1] = 0  # 解决循环引用以便垃圾收集器处理

        first = np.array(None)
        second = np.array(None)
        first[()] = second
        second[()] = first
        assert_equal(repr(first),
            'array(array(array(..., dtype=object), dtype=object), dtype=object)')
        first[()] = 0  # 解决循环引用以便垃圾收集器处理
    # 定义测试方法，测试包含列表的情况

        # 创建一个包含两个元素的一维 NumPy 数组，初始值为 None
        arr1d = np.array([None, None])

        # 在数组的第一个位置插入一个包含两个元素的列表 [1, 2]
        arr1d[0] = [1, 2]

        # 在数组的第二个位置插入一个包含一个元素的列表 [3]
        arr1d[1] = [3]

        # 断言数组的字符串表示，确认包含了预期的列表结构
        assert_equal(repr(arr1d),
            'array([list([1, 2]), list([3])], dtype=object)')

    # 定义测试方法，测试空标量递归的情况

        # 运行 repr(np.void(b'test'))，测试是否会出现递归错误
        repr(np.void(b'test'))  # RecursionError ?

    # 定义测试方法，测试没有字段的结构化数组情况

        # 创建一个没有字段的数据类型对象
        no_fields = np.dtype([])

        # 创建一个包含 4 个元素的空数组，使用上面定义的无字段数据类型
        arr_no_fields = np.empty(4, dtype=no_fields)

        # 断言数组的字符串表示，确认数组中每个元素都是空元组
        assert_equal(repr(arr_no_fields), 'array([(), (), (), ()], dtype=[])')
class TestComplexArray:
    def test_str(self):
        # 定义实数和复数的测试值列表，包括整数、无穷大和NaN
        rvals = [0, 1, -1, np.inf, -np.inf, np.nan]
        # 生成复数的列表，包括所有可能的实部和虚部组合
        cvals = [complex(rp, ip) for rp in rvals for ip in rvals]
        # 定义测试的复数数据类型列表
        dtypes = [np.complex64, np.cdouble, np.clongdouble]
        # 生成实际结果的列表，包括每个复数在每种数据类型下的字符串表示
        actual = [str(np.array([c], dt)) for c in cvals for dt in dtypes]
        # 期望的结果列表，包含每个复数在每种数据类型下的预期字符串表示
        wanted = [
            '[0.+0.j]',    '[0.+0.j]',    '[0.+0.j]',
            '[0.+1.j]',    '[0.+1.j]',    '[0.+1.j]',
            '[0.-1.j]',    '[0.-1.j]',    '[0.-1.j]',
            '[0.+infj]',   '[0.+infj]',   '[0.+infj]',
            '[0.-infj]',   '[0.-infj]',   '[0.-infj]',
            '[0.+nanj]',   '[0.+nanj]',   '[0.+nanj]',
            '[1.+0.j]',    '[1.+0.j]',    '[1.+0.j]',
            '[1.+1.j]',    '[1.+1.j]',    '[1.+1.j]',
            '[1.-1.j]',    '[1.-1.j]',    '[1.-1.j]',
            '[1.+infj]',   '[1.+infj]',   '[1.+infj]',
            '[1.-infj]',   '[1.-infj]',   '[1.-infj]',
            '[1.+nanj]',   '[1.+nanj]',   '[1.+nanj]',
            '[-1.+0.j]',   '[-1.+0.j]',   '[-1.+0.j]',
            '[-1.+1.j]',   '[-1.+1.j]',   '[-1.+1.j]',
            '[-1.-1.j]',   '[-1.-1.j]',   '[-1.-1.j]',
            '[-1.+infj]',  '[-1.+infj]',  '[-1.+infj]',
            '[-1.-infj]',  '[-1.-infj]',  '[-1.-infj]',
            '[-1.+nanj]',  '[-1.+nanj]',  '[-1.+nanj]',
            '[inf+0.j]',   '[inf+0.j]',   '[inf+0.j]',
            '[inf+1.j]',   '[inf+1.j]',   '[inf+1.j]',
            '[inf-1.j]',   '[inf-1.j]',   '[inf-1.j]',
            '[inf+infj]',  '[inf+infj]',  '[inf+infj]',
            '[inf-infj]',  '[inf-infj]',  '[inf-infj]',
            '[inf+nanj]',  '[inf+nanj]',  '[inf+nanj]',
            '[-inf+0.j]',  '[-inf+0.j]',  '[-inf+0.j]',
            '[-inf+1.j]',  '[-inf+1.j]',  '[-inf+1.j]',
            '[-inf-1.j]',  '[-inf-1.j]',  '[-inf-1.j]',
            '[-inf+infj]', '[-inf+infj]', '[-inf+infj]',
            '[-inf-infj]', '[-inf-infj]', '[-inf-infj]',
            '[-inf+nanj]', '[-inf+nanj]', '[-inf+nanj]',
            '[nan+0.j]',   '[nan+0.j]',   '[nan+0.j]',
            '[nan+1.j]',   '[nan+1.j]',   '[nan+1.j]',
            '[nan-1.j]',   '[nan-1.j]',   '[nan-1.j]',
            '[nan+infj]',  '[nan+infj]',  '[nan+infj]',
            '[nan-infj]',  '[nan-infj]',  '[nan-infj]',
            '[nan+nanj]',  '[nan+nanj]',  '[nan+nanj]']

        # 对每个实际结果和期望结果进行断言比较
        for res, val in zip(actual, wanted):
            assert_equal(res, val)

class TestArray2String:
    def test_basic(self):
        """Basic test of array2string."""
        # 创建一个简单的整数数组
        a = np.arange(3)
        # 断言默认设置下整数数组的字符串表示
        assert_(np.array2string(a) == '[0 1 2]')
        # 断言在特定设置下整数数组的字符串表示，包括最大行宽和遗留模式
        assert_(np.array2string(a, max_line_width=4, legacy='1.13') == '[0 1\n 2]')
        # 断言在特定设置下整数数组的字符串表示，包括最大行宽
        assert_(np.array2string(a, max_line_width=4) == '[0\n 1\n 2]')
    # 定义测试函数：验证当传递给 array2string 函数一个未预期的关键字参数时，是否引发适当的 TypeError 异常
    def test_unexpected_kwarg(self):
        # 使用 assert_raises_regex 上下文管理器断言捕获的异常为 TypeError，并且异常消息中包含 'nonsense'
        with assert_raises_regex(TypeError, 'nonsense'):
            # 调用 np.array2string 函数，传递一个数组和一个未预期的关键字参数 nonsense=None
            np.array2string(np.array([1, 2, 3]),
                            nonsense=None)

    # 定义测试函数：测试自定义的格式化函数用于数组中每个元素
    def test_format_function(self):
        """Test custom format function for each element in array."""
        
        # 定义内部函数 _format_function，根据元素的绝对值选择不同的字符表示
        def _format_function(x):
            if np.abs(x) < 1:
                return '.'
            elif np.abs(x) < 2:
                return 'o'
            else:
                return 'O'

        # 创建一个包含 [0, 1, 2] 的整数数组 x
        x = np.arange(3)
        # 创建预期的十六进制字符串表示
        x_hex = "[0x0 0x1 0x2]"
        # 创建预期的八进制字符串表示
        x_oct = "[0o0 0o1 0o2]"
        
        # 使用 assert_ 断言 np.array2string 函数的结果符合预期
        assert_(np.array2string(x, formatter={'all':_format_function}) ==
                "[. o O]")
        assert_(np.array2string(x, formatter={'int_kind':_format_function}) ==
                "[. o O]")
        assert_(np.array2string(x, formatter={'all':lambda x: "%.4f" % x}) ==
                "[0.0000 1.0000 2.0000]")
        
        # 使用 assert_equal 断言 np.array2string 函数的结果与预期的十六进制字符串相等
        assert_equal(np.array2string(x, formatter={'int':lambda x: hex(x)}),
                     x_hex)
        # 使用 assert_equal 断言 np.array2string 函数的结果与预期的八进制字符串相等
        assert_equal(np.array2string(x, formatter={'int':lambda x: oct(x)}),
                     x_oct)

        # 创建一个包含 [0. 1. 2.] 的浮点数数组 x
        x = np.arange(3.)
        # 使用 assert_ 断言 np.array2string 函数的结果符合预期的浮点数格式
        assert_(np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x}) ==
                "[0.00 1.00 2.00]")
        assert_(np.array2string(x, formatter={'float':lambda x: "%.2f" % x}) ==
                "[0.00 1.00 2.00]")

        # 创建一个包含 ['abc', 'def'] 的字符串数组 s
        s = np.array(['abc', 'def'])
        # 使用 assert_ 断言 np.array2string 函数的结果符合预期的字符串加倍格式
        assert_(np.array2string(s, formatter={'numpystr':lambda s: s*2}) ==
                '[abcabc defdef]')
    def test_structure_format_mixed(self):
        # 定义结构化数据类型 dt，包含字段 'name' 和 'grades'
        dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
        
        # 创建一个结构化数组 x，包含两个条目：('Sarah', [8.0, 7.0]) 和 ('John', [6.0, 7.0])
        x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        
        # 使用 assert_equal 函数验证 np.array2string(x) 的输出是否符合预期
        assert_equal(np.array2string(x),
                "[('Sarah', [8., 7.]) ('John', [6., 7.])]")

        # 设置打印选项，使用 'legacy' 模式为 '1.13'
        np.set_printoptions(legacy='1.13')
        try:
            # 创建一个包含 'A' 字段的零填充数组 A，数据类型为 "M8[s]"
            A = np.zeros(shape=10, dtype=[("A", "M8[s]")])
            
            # 将数组 A 的从索引 5 开始的元素填充为 NaT (Not a Time)
            A[5:].fill(np.datetime64('NaT'))
            
            # 使用 assert_equal 函数验证 np.array2string(A) 的输出是否符合预期
            assert_equal(
                np.array2string(A),
                textwrap.dedent("""\
                [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
                 ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('NaT',) ('NaT',)
                 ('NaT',) ('NaT',) ('NaT',)]""")
            )
        finally:
            # 恢复非 legacy 打印选项
            np.set_printoptions(legacy=False)

        # 再次验证非 legacy 模式下的输出
        assert_equal(
            np.array2string(A),
            textwrap.dedent("""\
            [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
             ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)
             ('1970-01-01T00:00:00',) (                'NaT',)
             (                'NaT',) (                'NaT',)
             (                'NaT',) (                'NaT',)]""")
        )

        # 使用 timedeltas 的情况再次验证输出
        A = np.full(10, 123456, dtype=[("A", "m8[s]")])
        A[5:].fill(np.datetime64('NaT'))
        assert_equal(
            np.array2string(A),
            textwrap.dedent("""\
            [(123456,) (123456,) (123456,) (123456,) (123456,) ( 'NaT',) ( 'NaT',)
             ( 'NaT',) ( 'NaT',) ( 'NaT',)]""")
        )

    def test_structure_format_int(self):
        # See #8160
        # 创建一个包含整数数组的结构化数组 struct_int
        struct_int = np.array([([1, -1],), ([123, 1],)],
                dtype=[('B', 'i4', 2)])
        
        # 使用 assert_equal 函数验证 np.array2string(struct_int) 的输出是否符合预期
        assert_equal(np.array2string(struct_int),
                "[([  1,  -1],) ([123,   1],)]")
        
        # 创建一个包含二维整数数组的结构化数组 struct_2dint
        struct_2dint = np.array([([[0, 1], [2, 3]],), ([[12, 0], [0, 0]],)],
                dtype=[('B', 'i4', (2, 2))])
        
        # 使用 assert_equal 函数验证 np.array2string(struct_2dint) 的输出是否符合预期
        assert_equal(np.array2string(struct_2dint),
                "[([[ 0,  1], [ 2,  3]],) ([[12,  0], [ 0,  0]],)]")

    def test_structure_format_float(self):
        # See #8172
        # 创建一个包含浮点数的数组 array_scalar
        array_scalar = np.array(
                (1., 2.1234567890123456789, 3.), dtype=('f8,f8,f8'))
        
        # 使用 assert_equal 函数验证 np.array2string(array_scalar) 的输出是否符合预期
        assert_equal(np.array2string(array_scalar), "(1., 2.12345679, 3.)")
    # 测试 void 类型数组的表示形式
    def test_unstructured_void_repr(self):
        # 创建一个 numpy 数组，使用 'u1' 类型，视图类型为 'V8'，包含特定的字节序列
        a = np.array([27, 91, 50, 75,  7, 65, 10,  8,
                      27, 91, 51, 49,109, 82,101,100], dtype='u1').view('V8')
        # 断言第一个数组元素的 repr 结果
        assert_equal(repr(a[0]),
            r"np.void(b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08')")
        # 断言第一个数组元素的 str 结果
        assert_equal(str(a[0]), r"b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08'")
        # 断言整个数组的 repr 结果
        assert_equal(repr(a),
            r"array([b'\x1B\x5B\x32\x4B\x07\x41\x0A\x08'," "\n"
            r"       b'\x1B\x5B\x33\x31\x6D\x52\x65\x64'], dtype='|V8')")

        # 使用 eval 函数检查 repr(a) 的结果是否与原始数组 a 相等
        assert_equal(eval(repr(a), vars(np)), a)
        # 使用 eval 函数检查 repr(a[0]) 的结果是否与原始数组 a[0] 相等
        assert_equal(eval(repr(a[0]), dict(np=np)), a[0])

    # 测试 np.array2string 函数的 edgeitems 关键字参数
    def test_edgeitems_kwarg(self):
        # 创建一个包含三个整数零的数组
        arr = np.zeros(3, int)
        # 断言 np.array2string 函数对数组 arr 使用 edgeitems=1, threshold=0 的结果
        assert_equal(
            np.array2string(arr, edgeitems=1, threshold=0),
            "[0 ... 0]"
        )

    # 测试 1 维数组的 summarize 功能
    def test_summarize_1d(self):
        # 创建一个包含 0 到 1000 的整数的数组
        A = np.arange(1001)
        # 期望的字符串表示
        strA = '[   0    1    2 ...  998  999 1000]'
        # 断言数组 A 的字符串表示是否符合预期
        assert_equal(str(A), strA)

        # 期望的 repr 表示
        reprA = 'array([   0,    1,    2, ...,  998,  999, 1000])'
        # 断言数组 A 的 repr 表示是否符合预期
        assert_equal(repr(A), reprA)

    # 测试 2 维数组的 summarize 功能
    def test_summarize_2d(self):
        # 创建一个包含 0 到 1001 的整数，形状为 (2, 501) 的数组
        A = np.arange(1002).reshape(2, 501)
        # 期望的字符串表示
        strA = '[[   0    1    2 ...  498  499  500]\n' \
               ' [ 501  502  503 ...  999 1000 1001]]'
        # 断言数组 A 的字符串表示是否符合预期
        assert_equal(str(A), strA)

        # 期望的 repr 表示
        reprA = 'array([[   0,    1,    2, ...,  498,  499,  500],\n' \
                '       [ 501,  502,  503, ...,  999, 1000, 1001]])'
        # 断言数组 A 的 repr 表示是否符合预期
        assert_equal(repr(A), reprA)

    # 测试结构化数组的 summarize 功能
    def test_summarize_structure(self):
        # 创建一个结构化数组 A，包含两个字段，每个字段是包含 0 到 1000 的整数的数组
        A = (np.arange(2002, dtype="<i8").reshape(2, 1001)
             .view([('i', "<i8", (1001,))]))
        # 期望的字符串表示
        strA = ("[[([   0,    1,    2, ...,  998,  999, 1000],)]\n"
                " [([1001, 1002, 1003, ..., 1999, 2000, 2001],)]]")
        # 断言数组 A 的字符串表示是否符合预期
        assert_equal(str(A), strA)

        # 期望的 repr 表示
        reprA = ("array([[([   0,    1,    2, ...,  998,  999, 1000],)],\n"
                 "       [([1001, 1002, 1003, ..., 1999, 2000, 2001],)]],\n"
                 "      dtype=[('i', '<i8', (1001,))])")
        # 断言数组 A 的 repr 表示是否符合预期
        assert_equal(repr(A), reprA)

        # 创建一个结构化数组 B，包含一个字段，每个字段是包含 1 的数组
        B = np.ones(2002, dtype=">i8").view([('i', ">i8", (2, 1001))])
        # 期望的字符串表示
        strB = "[([[1, 1, 1, ..., 1, 1, 1], [1, 1, 1, ..., 1, 1, 1]],)]"
        # 断言数组 B 的字符串表示是否符合预期
        assert_equal(str(B), strB)

        # 期望的 repr 表示
        reprB = (
            "array([([[1, 1, 1, ..., 1, 1, 1], [1, 1, 1, ..., 1, 1, 1]],)],\n"
            "      dtype=[('i', '>i8', (2, 1001))])"
        )
        # 断言数组 B 的 repr 表示是否符合预期
        assert_equal(repr(B), reprB)

        # 创建一个结构化数组 C，包含两个字段，每个字段是包含 0 到 21 的整数的数组
        C = (np.arange(22, dtype="<i8").reshape(2, 11)
             .view([('i1', "<i8"), ('i10', "<i8", (10,))]))
        # 期望的字符串表示
        strC = "[[( 0, [ 1, ..., 10])]\n [(11, [12, ..., 21])]]"
        # 使用 np.array2string 函数，并设置 threshold=1, edgeitems=1
        # 断言数组 C 的字符串表示是否符合预期
        assert_equal(np.array2string(C, threshold=1, edgeitems=1), strC)
    # 定义测试方法，用于测试数组转换为字符串时的行宽限制
    def test_linewidth(self):
        # 创建一个包含六个元素的全1数组
        a = np.full(6, 1)

        # 定义内部函数make_str，将数组转换为字符串，控制最大行宽
        def make_str(a, width, **kw):
            return np.array2string(a, separator="", max_line_width=width, **kw)

        # 断言不同行宽下的字符串转换结果
        assert_equal(make_str(a, 8, legacy='1.13'), '[111111]')
        assert_equal(make_str(a, 7, legacy='1.13'), '[111111]')
        assert_equal(make_str(a, 5, legacy='1.13'), '[1111\n'
                                                    ' 11]')

        assert_equal(make_str(a, 8), '[111111]')
        assert_equal(make_str(a, 7), '[11111\n'
                                     ' 1]')
        assert_equal(make_str(a, 5), '[111\n'
                                     ' 111]')

        # 将a数组转换为形状为(1, 1, 6)的数组b
        b = a[None,None,:]

        # 断言不同行宽下的多维数组转换结果
        assert_equal(make_str(b, 12, legacy='1.13'), '[[[111111]]]')
        assert_equal(make_str(b,  9, legacy='1.13'), '[[[111111]]]')
        assert_equal(make_str(b,  8, legacy='1.13'), '[[[11111\n'
                                                     '   1]]]')

        assert_equal(make_str(b, 12), '[[[111111]]]')
        assert_equal(make_str(b,  9), '[[[111\n'
                                      '   111]]]')
        assert_equal(make_str(b,  8), '[[[11\n'
                                      '   11\n'
                                      '   11]]]')

    # 定义测试方法，用于测试数组中包含宽元素时的转换为字符串操作
    def test_wide_element(self):
        # 创建包含单个字符串元素的数组a
        a = np.array(['xxxxx'])
        
        # 断言在限制最大行宽为5时的数组转换结果
        assert_equal(
            np.array2string(a, max_line_width=5),
            "['xxxxx']"
        )
        
        # 断言在限制最大行宽为5且使用旧版本模式时的数组转换结果
        assert_equal(
            np.array2string(a, max_line_width=5, legacy='1.13'),
            "[ 'xxxxx']"
        )
    def test_multiline_repr(self):
        # 定义一个具有多行字符串表示的类 MultiLine
        class MultiLine:
            # 定义该类的 __repr__ 方法，返回多行字符串
            def __repr__(self):
                return "Line 1\nLine 2"

        # 创建一个包含 MultiLine 对象的 NumPy 数组 a
        a = np.array([[None, MultiLine()], [MultiLine(), None]])

        # 断言调用 np.array2string 函数后的返回结果
        assert_equal(
            np.array2string(a),
            '[[None Line 1\n'
            '       Line 2]\n'
            ' [Line 1\n'
            '  Line 2 None]]'
        )

        # 断言调用 np.array2string 函数（限定最大行宽为 5）后的返回结果
        assert_equal(
            np.array2string(a, max_line_width=5),
            '[[None\n'
            '  Line 1\n'
            '  Line 2]\n'
            ' [Line 1\n'
            '  Line 2\n'
            '  None]]'
        )

        # 断言调用 repr 函数后的返回结果
        assert_equal(
            repr(a),
            'array([[None, Line 1\n'
            '              Line 2],\n'
            '       [Line 1\n'
            '        Line 2, None]], dtype=object)'
        )

        # 定义一个更长的多行字符串表示的类 MultiLineLong
        class MultiLineLong:
            # 定义该类的 __repr__ 方法，返回更长的多行字符串
            def __repr__(self):
                return "Line 1\nLooooooooooongestLine2\nLongerLine 3"

        # 创建一个包含 MultiLineLong 对象的 NumPy 数组 a
        a = np.array([[None, MultiLineLong()], [MultiLineLong(), None]])

        # 断言调用 repr 函数后的返回结果
        assert_equal(
            repr(a),
            'array([[None, Line 1\n'
            '              LooooooooooongestLine2\n'
            '              LongerLine 3          ],\n'
            '       [Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          , None]], dtype=object)'
        )

        # 断言调用 np.array_repr 函数（限定每行最大字符数为 20）后的返回结果
        assert_equal(
            np.array_repr(a, 20),
            'array([[None,\n'
            '        Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          ],\n'
            '       [Line 1\n'
            '        LooooooooooongestLine2\n'
            '        LongerLine 3          ,\n'
            '        None]],\n'
            '      dtype=object)'
        )

    def test_nested_array_repr(self):
        # 创建一个空的 NumPy 数组 a，元素类型为对象
        a = np.empty((2, 2), dtype=object)
        # 在数组中填充元素
        a[0, 0] = np.eye(2)
        a[0, 1] = np.eye(3)
        a[1, 0] = None
        a[1, 1] = np.ones((3, 1))

        # 断言调用 repr 函数后的返回结果
        assert_equal(
            repr(a),
            'array([[array([[1., 0.],\n'
            '               [0., 1.]]), array([[1., 0., 0.],\n'
            '                                  [0., 1., 0.],\n'
            '                                  [0., 0., 1.]])],\n'
            '       [None, array([[1.],\n'
            '                     [1.],\n'
            '                     [1.]])]], dtype=object)'
        )

    @given(hynp.from_dtype(np.dtype("U")))
    # 定义一个测试方法，用于检查任意文本的处理
    def test_any_text(self, text):
        # 这个测试检查，对于可以表示为dtype("U")数组的任何值（即Unicode字符串）...
        a = np.array([text, text, text])
        # 将它们的列表转换为数组不会截断值，例如：
        assert_equal(a[0], text)
        # 使用原始的Python字符串来表示下面的repr
        text = text.item()  # 用于下面repr的原始Python字符串
        # 并且确保np.array2string在预期位置放置换行符
        expected_repr = "[{0!r} {0!r}\n {0!r}]".format(text)
        # 调用np.array2string生成结果字符串，限制行宽度为repr(text)长度的两倍加3
        result = np.array2string(a, max_line_width=len(repr(text)) * 2 + 3)
        # 断言生成的结果字符串符合预期的repr格式
        assert_equal(result, expected_repr)

    # 使用pytest的标记，如果没有引用计数，则跳过测试（因为Python缺少引用计数）
    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_refcount(self):
        # 确保由于递归闭包（gh-10620），我们不会保留对数组的引用
        gc.disable()
        # 创建一个包含0和1的数组a
        a = np.arange(2)
        # 获取a的引用计数r1
        r1 = sys.getrefcount(a)
        # 调用np.array2string(a)，不应该增加对a的引用计数
        np.array2string(a)
        np.array2string(a)
        # 再次获取a的引用计数r2
        r2 = sys.getrefcount(a)
        gc.collect()  # 手动垃圾回收
        gc.enable()
        # 断言在禁用和启用gc之间，a的引用计数没有变化
        assert_(r1 == r2)
    # 定义一个测试方法，用于测试带有符号输出的功能
    def test_with_sign(self):
        # 创建一个包含负数和正数值的 NumPy 数组
        a = np.array([-2, 0, 3])
        # 断言使用 '+' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='+'),
            '[-2 +0 +3]'
        )
        # 断言使用 '-' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='-'),
            '[-2  0  3]'
        )
        # 断言使用空格符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign=' '),
            '[-2  0  3]'
        )
        # 创建一个全部为非负数的 NumPy 数组
        a = np.array([2, 0, 3])
        # 断言使用 '+' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='+'),
            '[+2 +0 +3]'
        )
        # 断言使用 '-' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='-'),
            '[2 0 3]'
        )
        # 断言使用空格符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign=' '),
            '[ 2  0  3]'
        )
        # 创建一个全部为负数的 NumPy 数组
        a = np.array([-2, -1, -3])
        # 断言使用 '+' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='+'),
            '[-2 -1 -3]'
        )
        # 断言使用 '-' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='-'),
            '[-2 -1 -3]'
        )
        # 断言使用空格符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign=' '),
            '[-2 -1 -3]'
        )
        # 创建一个二维数组，混合负数和正数
        a = np.array([[10, -1, 1, 1], [10, 10, 10, 10]])
        # 断言使用 '+' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='+'),
            '[[+10  -1  +1  +1]\n [+10 +10 +10 +10]]'
        )
        # 断言使用 '-' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='-'),
            '[[10 -1  1  1]\n [10 10 10 10]]'
        )
        # 断言使用空格符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign=' '),
            '[[10 -1  1  1]\n [10 10 10 10]]'
        )
        # 创建一个二维数组，全部为正数
        a = np.array([[10, 0, 1, 1], [10, 10, 10, 10]])
        # 断言使用 '+' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='+'),
            '[[+10  +0  +1  +1]\n [+10 +10 +10 +10]]'
        )
        # 断言使用 '-' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='-'),
            '[[10  0  1  1]\n [10 10 10 10]]'
        )
        # 断言使用空格符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign=' '),
            '[[ 10   0   1   1]\n [ 10  10  10  10]]'
        )
        # 创建一个二维数组，全部为负数
        a = np.array([[-10, -1, -1, -1], [-10, -10, -10, -10]])
        # 断言使用 '+' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='+'),
            '[[-10  -1  -1  -1]\n [-10 -10 -10 -10]]'
        )
        # 断言使用 '-' 符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign='-'),
            '[[-10  -1  -1  -1]\n [-10 -10 -10 -10]]'
        )
        # 断言使用空格符号输出数组的字符串表示
        assert_equal(
            np.array2string(a, sign=' '),
            '[[-10  -1  -1  -1]\n [-10 -10 -10 -10]]'
        )
# 定义一个测试类 TestPrintOptions，用于测试获取和设置全局打印选项
class TestPrintOptions:
    """Test getting and setting global print options."""

    # 设置每个测试方法的初始化方法
    def setup_method(self):
        # 获取当前的打印选项并保存为旧选项
        self.oldopts = np.get_printoptions()

    # 设置每个测试方法的清理方法
    def teardown_method(self):
        # 恢复之前保存的旧打印选项
        np.set_printoptions(**self.oldopts)

    # 测试基本的打印精度设置
    def test_basic(self):
        # 创建一个包含浮点数的 NumPy 数组
        x = np.array([1.5, 0, 1.234567890])
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([1.5       , 0.        , 1.23456789])")
        # 设置新的打印选项，修改精度为 4
        np.set_printoptions(precision=4)
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([1.5   , 0.    , 1.2346])")

    # 测试精度设置为零时的打印输出
    def test_precision_zero(self):
        # 设置打印选项，精度设为 0
        np.set_printoptions(precision=0)
        # 迭代测试不同值和对应的字符串表示
        for values, string in (
                ([0.], "0."), ([.3], "0."), ([-.3], "-0."), ([.7], "1."),
                ([1.5], "2."), ([-1.5], "-2."), ([-15.34], "-15."),
                ([100.], "100."), ([.2, -1, 122.51], "  0.,  -1., 123."),
                ([0], "0"), ([-12], "-12"), ([complex(.3, -.7)], "0.-1.j")):
            # 创建包含特定值的 NumPy 数组
            x = np.array(values)
            # 断言数组的字符串表示与期望相符
            assert_equal(repr(x), "array([%s])" % string)

    # 测试自定义格式化函数的打印输出
    def test_formatter(self):
        # 创建一个整数范围的 NumPy 数组
        x = np.arange(3)
        # 设置自定义格式化函数，将所有元素减一后转为字符串
        np.set_printoptions(formatter={'all': lambda x: str(x - 1)})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([-1, 0, 1])")

    # 测试重置格式化函数后的打印输出
    def test_formatter_reset(self):
        # 创建一个整数范围的 NumPy 数组
        x = np.arange(3)
        # 设置自定义格式化函数，将所有元素减一后转为字符串
        np.set_printoptions(formatter={'all': lambda x: str(x - 1)})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([-1, 0, 1])")
        # 重置整数格式化函数为默认
        np.set_printoptions(formatter={'int': None})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([0, 1, 2])")

        # 重新设置格式化函数，将所有元素减一后转为字符串
        np.set_printoptions(formatter={'all': lambda x: str(x - 1)})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([-1, 0, 1])")
        # 清除所有格式化函数
        np.set_printoptions(formatter={'all': None})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([0, 1, 2])")

        # 设置整数格式化函数，将所有整数元素减一后转为字符串
        np.set_printoptions(formatter={'int': lambda x: str(x - 1)})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([-1, 0, 1])")
        # 清除整数类型的格式化函数
        np.set_printoptions(formatter={'int_kind': None})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([0, 1, 2])")

        # 创建一个浮点数范围的 NumPy 数组
        x = np.arange(3.)
        # 设置浮点数格式化函数，将所有元素减一后转为字符串
        np.set_printoptions(formatter={'float': lambda x: str(x - 1)})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([-1.0, 0.0, 1.0])")
        # 清除浮点数类型的格式化函数
        np.set_printoptions(formatter={'float_kind': None})
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([0., 1., 2.])")

    # 测试重写 repr 函数后的打印输出
    def test_override_repr(self):
        # 创建一个整数范围的 NumPy 数组
        x = np.arange(3)
        # 设置重写 repr 函数，使其返回固定字符串 "FOO"
        np.set_printoptions(override_repr=lambda x: "FOO")
        # 断言数组的字符串表示为 "FOO"
        assert_equal(repr(x), "FOO")
        # 清除重写的 repr 函数
        np.set_printoptions(override_repr=None)
        # 断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([0, 1, 2])")

        # 使用 with 语句设置重写 repr 函数，使其返回固定字符串 "BAR"
        with np.printoptions(override_repr=lambda x: "BAR"):
            # 断言数组的字符串表示为 "BAR"
            assert_equal(repr(x), "BAR")
        # 离开 with 语句后，断言数组的字符串表示与期望相符
        assert_equal(repr(x), "array([0, 1, 2])")
    def test_0d_arrays(self):
        # 测试对于0维数组的字符串表示，确保将字符串按照指定的格式 '<U4' 进行编码
        assert_equal(str(np.array('café', '<U4')), 'café')

        # 测试对于0维数组的表征，使用repr函数确保返回正确的数组表征字符串
        assert_equal(repr(np.array('café', '<U4')),
                     "array('café', dtype='<U4')")
        
        # 测试对于0维字符串数组的字符串表示
        assert_equal(str(np.array('test', np.str_)), 'test')

        # 创建一个dtype为 [('a', '<i4', (3,))] 的1维零数组，并确保其字符串表示正确
        a = np.zeros(1, dtype=[('a', '<i4', (3,))])
        assert_equal(str(a[0]), '([0, 0, 0],)')

        # 测试对于datetime64类型的0维数组，使用repr确保返回正确的日期字符串
        assert_equal(repr(np.datetime64('2005-02-25')[...]),
                     "array('2005-02-25', dtype='datetime64[D]')")

        # 测试对于timedelta64类型的0维数组，使用repr确保返回正确的时间间隔字符串
        assert_equal(repr(np.timedelta64('10', 'Y')[...]),
                     "array(10, dtype='timedelta64[Y]')")

        # 设置printoptions，测试0维数组的表征受其影响的情况
        x = np.array(1)
        np.set_printoptions(formatter={'all':lambda x: "test"})
        assert_equal(repr(x), "array(test)")
        # str函数不受影响
        assert_equal(str(x), "1")

        # 检查`style`参数引发DeprecationWarning警告
        assert_warns(DeprecationWarning, np.array2string,
                                         np.array(1.), style=repr)
        # 但在legacy模式下不会引发警告
        np.array2string(np.array(1.), style=repr, legacy='1.13')
        # gh-10934：在legacy模式下，style可能不起作用，检查其是否正常工作
        np.array2string(np.array(1.), legacy='1.13')

    def test_float_spacing(self):
        # 创建浮点数数组并检查其字符串表征
        x = np.array([1., 2., 3.])
        y = np.array([1., 2., -10.])
        z = np.array([100., 2., -1.])
        w = np.array([-100., 2., 1.])

        assert_equal(repr(x), 'array([1., 2., 3.])')
        assert_equal(repr(y), 'array([  1.,   2., -10.])')
        assert_equal(repr(np.array(y[0])), 'array(1.)')
        assert_equal(repr(np.array(y[-1])), 'array(-10.)')
        assert_equal(repr(z), 'array([100.,   2.,  -1.])')
        assert_equal(repr(w), 'array([-100.,    2.,    1.])')

        # 检查包含NaN和inf的数组的表征
        assert_equal(repr(np.array([np.nan, np.inf])), 'array([nan, inf])')
        assert_equal(repr(np.array([np.nan, -np.inf])), 'array([ nan, -inf])')

        # 设置精度为2，检查数组表征中浮点数的输出格式
        x = np.array([np.inf, 100000, 1.1234])
        y = np.array([np.inf, 100000, -1.1234])
        z = np.array([np.inf, 1.1234, -1e120])
        np.set_printoptions(precision=2)
        assert_equal(repr(x), 'array([     inf, 1.00e+05, 1.12e+00])')
        assert_equal(repr(y), 'array([      inf,  1.00e+05, -1.12e+00])')
        assert_equal(repr(z), 'array([       inf,  1.12e+000, -1.00e+120])')

    def test_bool_spacing(self):
        # 检查布尔数组的表征，确保正确的空格处理
        assert_equal(repr(np.array([True,  True])),
                     'array([ True,  True])')
        assert_equal(repr(np.array([True, False])),
                     'array([ True, False])')
        assert_equal(repr(np.array([True])),
                     'array([ True])')
        assert_equal(repr(np.array(True)),
                     'array(True)')
        assert_equal(repr(np.array(False)),
                     'array(False)')
    def test_sign_spacing(self):
        # 创建一个包含四个浮点数的数组
        a = np.arange(4.)
        # 创建一个包含单个浮点数的数组
        b = np.array([1.234e9])
        # 创建一个复数数组，指定数据类型为复数
        c = np.array([1.0 + 1.0j, 1.123456789 + 1.123456789j], dtype='c16')

        # 断言：验证数组的字符串表示是否与预期相符
        assert_equal(repr(a), 'array([0., 1., 2., 3.])')
        assert_equal(repr(np.array(1.)), 'array(1.)')
        assert_equal(repr(b), 'array([1.234e+09])')
        assert_equal(repr(np.array([0.])), 'array([0.])')
        assert_equal(repr(c),
            "array([1.        +1.j        , 1.12345679+1.12345679j])")
        assert_equal(repr(np.array([0., -0.])), 'array([ 0., -0.])')

        # 设置 NumPy 打印选项，使符号前面有空格
        np.set_printoptions(sign=' ')
        assert_equal(repr(a), 'array([ 0.,  1.,  2.,  3.])')
        assert_equal(repr(np.array(1.)), 'array( 1.)')
        assert_equal(repr(b), 'array([ 1.234e+09])')
        assert_equal(repr(c),
            "array([ 1.        +1.j        ,  1.12345679+1.12345679j])")
        assert_equal(repr(np.array([0., -0.])), 'array([ 0., -0.])')

        # 设置 NumPy 打印选项，使符号前面为加号
        np.set_printoptions(sign='+')
        assert_equal(repr(a), 'array([+0., +1., +2., +3.])')
        assert_equal(repr(np.array(1.)), 'array(+1.)')
        assert_equal(repr(b), 'array([+1.234e+09])')
        assert_equal(repr(c),
            "array([+1.        +1.j        , +1.12345679+1.12345679j])")

        # 设置 NumPy 打印选项，使用旧版（1.13）的格式
        np.set_printoptions(legacy='1.13')
        assert_equal(repr(a), 'array([ 0.,  1.,  2.,  3.])')
        assert_equal(repr(b),  'array([  1.23400000e+09])')
        assert_equal(repr(-b), 'array([ -1.23400000e+09])')
        assert_equal(repr(np.array(1.)), 'array(1.0)')
        assert_equal(repr(np.array([0.])), 'array([ 0.])')
        assert_equal(repr(c),
            "array([ 1.00000000+1.j        ,  1.12345679+1.12345679j])")
        # gh-10383
        # 断言：验证浮点数数组的字符串表示是否与预期相符
        assert_equal(str(np.array([-1., 10])), "[ -1.  10.]")

        # 断言：检查是否会抛出 TypeError 异常，用于测试 set_printoptions 函数的错误参数
        assert_raises(TypeError, np.set_printoptions, wrongarg=True)

    def test_float_overflow_nowarn(self):
        # 确保 FloatingFormat 内部计算不会警告溢出
        repr(np.array([1e4, 0.1], dtype='f2'))

    def test_sign_spacing_structured(self):
        # 创建一个结构化数组，每个元素包含两个浮点数
        a = np.ones(2, dtype='<f,<f')
        assert_equal(repr(a),
            "array([(1., 1.), (1., 1.)], dtype=[('f0', '<f4'), ('f1', '<f4')])")
        assert_equal(repr(a[0]),
            "np.void((1.0, 1.0), dtype=[('f0', '<f4'), ('f1', '<f4')])")

    def test_legacy_mode_scalars(self):
        # 在旧版模式下，浮点数的字符串表示被截断，复数标量使用 * 表示非有限虚部
        np.set_printoptions(legacy='1.13')
        assert_equal(str(np.float64(1.123456789123456789)), '1.12345678912')
        assert_equal(str(np.complex128(complex(1, np.nan))), '(1+nan*j)')

        # 关闭旧版模式
        np.set_printoptions(legacy=False)
        assert_equal(str(np.float64(1.123456789123456789)),
                     '1.1234567891234568')
        assert_equal(str(np.complex128(complex(1, np.nan))), '(1+nanj)')
    @pytest.mark.parametrize(
        ['native'],
        [
            ('bool',),
            ('uint8',),
            ('uint16',),
            ('uint32',),
            ('uint64',),
            ('int8',),
            ('int16',),
            ('int32',),
            ('int64',),
            ('float16',),
            ('float32',),
            ('float64',),
            ('U1',),     # 4-byte width string
        ],
    )
    def test_dtype_endianness_repr(self, native):
        '''
        there was an issue where
        repr(array([0], dtype='<u2')) and repr(array([0], dtype='>u2'))
        both returned the same thing:
        array([0], dtype=uint16)
        even though their dtypes have different endianness.
        '''
        # 将参数化测试用例标记为'test_dtype_endianness_repr'，传入单个参数'native'，其取值范围为以下类型
        native_dtype = np.dtype(native)
        # 使用参数'native'构建NumPy数据类型对象
        non_native_dtype = native_dtype.newbyteorder()
        # 获取'native_dtype'的非本地字节顺序版本的数据类型对象
        non_native_repr = repr(np.array([1], non_native_dtype))
        # 获取包含非本地字节顺序的数组的字符串表示形式
        native_repr = repr(np.array([1], native_dtype))
        # 获取包含本地字节顺序的数组的字符串表示形式
        # 保持默认的合理性，只在类型非标准时显示数据类型
        assert ('dtype' in native_repr) ^ (native_dtype in _typelessdata),\
                ("an array's repr should show dtype if and only if the type "
                 'of the array is NOT one of the standard types '
                 '(e.g., int32, bool, float64).')
        # 断言：如果非本地字节顺序的数据类型的项大小大于1字节，则其表示形式与本地字节顺序的表示形式不同
        if non_native_dtype.itemsize > 1:
            # 如果数据类型的大小大于1字节，则非本地字节顺序版本必须显示字节顺序
            assert non_native_repr != native_repr
            assert f"dtype='{non_native_dtype.byteorder}" in non_native_repr
    def test_linewidth_repr(self):
        # 创建一个长度为7的NumPy数组，每个元素填充为2
        a = np.full(7, fill_value=2)
        # 设置打印选项，限制每行字符数为17
        np.set_printoptions(linewidth=17)
        # 断言数组的字符串表示是否与指定的格式化字符串相等
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2,
                   2])""")
        )
        # 修改打印选项，保持限制每行字符数为17，使用1.13版的遗留模式
        np.set_printoptions(linewidth=17, legacy='1.13')
        # 再次断言数组的字符串表示是否符合新的格式化字符串
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2, 2])""")
        )

        # 创建一个长度为8的NumPy数组，每个元素填充为2
        a = np.full(8, fill_value=2)

        # 修改打印选项，限制每行字符数为18，不使用遗留模式
        np.set_printoptions(linewidth=18, legacy=False)
        # 断言数组的字符串表示是否与指定的格式化字符串相等
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2,
                   2, 2, 2,
                   2, 2])""")
        )

        # 修改打印选项，限制每行字符数为18，使用1.13版的遗留模式
        np.set_printoptions(linewidth=18, legacy='1.13')
        # 再次断言数组的字符串表示是否符合新的格式化字符串
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([2, 2, 2, 2,
                   2, 2, 2, 2])""")
        )

    def test_linewidth_str(self):
        # 创建一个长度为18的NumPy数组，每个元素填充为2
        a = np.full(18, fill_value=2)
        # 设置打印选项，限制每行字符数为18
        np.set_printoptions(linewidth=18)
        # 断言数组的字符串表示是否与指定的格式化字符串相等
        assert_equal(
            str(a),
            textwrap.dedent("""\
            [2 2 2 2 2 2 2 2
             2 2 2 2 2 2 2 2
             2 2]""")
        )
        # 修改打印选项，限制每行字符数为18，使用1.13版的遗留模式
        np.set_printoptions(linewidth=18, legacy='1.13')
        # 再次断言数组的字符串表示是否符合新的格式化字符串
        assert_equal(
            str(a),
            textwrap.dedent("""\
            [2 2 2 2 2 2 2 2 2
             2 2 2 2 2 2 2 2 2]""")
        )

    def test_edgeitems(self):
        # 设置打印选项，仅显示第一维的第一个元素和阈值以内的元素
        np.set_printoptions(edgeitems=1, threshold=1)
        # 创建一个形状为(3, 3, 3)的NumPy数组，填充从0到26的数值
        a = np.arange(27).reshape((3, 3, 3))
        # 断言数组的字符串表示是否与指定的格式化字符串相等
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([[[ 0, ...,  2],
                    ...,
                    [ 6, ...,  8]],

                   ...,

                   [[18, ..., 20],
                    ...,
                    [24, ..., 26]]])""")
        )

        # 创建一个形状为(3, 3, 1, 1)的全零NumPy数组
        b = np.zeros((3, 3, 1, 1))
        # 断言数组的字符串表示是否与指定的格式化字符串相等
        assert_equal(
            repr(b),
            textwrap.dedent("""\
            array([[[[0.]],

                    ...,

                    [[0.]]],


                   ...,


                   [[[0.]],

                    ...,

                    [[0.]]]])""")
        )

        # 修改打印选项，使用1.13版的遗留模式
        np.set_printoptions(legacy='1.13')

        # 再次断言数组的字符串表示是否符合新的格式化字符串
        assert_equal(
            repr(a),
            textwrap.dedent("""\
            array([[[ 0, ...,  2],
                    ..., 
                    [ 6, ...,  8]],

                   ..., 
                   [[18, ..., 20],
                    ..., 
                    [24, ..., 26]]])""")
        )

        # 再次断言数组的字符串表示是否符合新的格式化字符串
        assert_equal(
            repr(b),
            textwrap.dedent("""\
            array([[[[ 0.]],

                    ..., 
                    [[ 0.]]],


                   ..., 
                   [[[ 0.]],

                    ..., 
                    [[ 0.]]]])""")
        )
    # 定义一个测试方法，用于测试在设置特定打印选项后，结构化数组的表示是否正确
    def test_edgeitems_structured(self):
        # 设置打印选项，仅显示边缘元素，超过阈值的元素用省略号表示
        np.set_printoptions(edgeitems=1, threshold=1)
        # 创建一个结构化数组 A，包含从 0 到 29 的整数，形状为 (5, 2, 3)
        A = np.arange(5*2*3, dtype="<i8").view([('i', "<i8", (5, 2, 3))])
        # 预期的结构化数组 A 的字符串表示
        reprA = (
            "array([([[[ 0, ...,  2], [ 3, ...,  5]], ..., "
            "[[24, ..., 26], [27, ..., 29]]],)],\n"
            "      dtype=[('i', '<i8', (5, 2, 3))])"
        )
        # 断言结构化数组 A 的字符串表示是否与预期的 reprA 相等
        assert_equal(repr(A), reprA)

    # 定义一个测试方法，用于测试设置不正确参数时是否会引发异常
    def test_bad_args(self):
        # 断言设置 NaN 作为阈值时会引发 ValueError 异常
        assert_raises(ValueError, np.set_printoptions, threshold=float('nan'))
        # 断言设置字符串 '1' 作为阈值时会引发 TypeError 异常
        assert_raises(TypeError, np.set_printoptions, threshold='1')
        # 断言设置字节串 b'1' 作为阈值时会引发 TypeError 异常
        assert_raises(TypeError, np.set_printoptions, threshold=b'1')

        # 断言设置字符串 '1' 作为精度时会引发 TypeError 异常
        assert_raises(TypeError, np.set_printoptions, precision='1')
        # 断言设置浮点数 1.5 作为精度时会引发 TypeError 异常
        assert_raises(TypeError, np.set_printoptions, precision=1.5)
def test_unicode_object_array():
    # 预期的字符串表示形式
    expected = "array(['é'], dtype=object)"
    # 创建包含一个 Unicode 对象的 NumPy 数组
    x = np.array(['\xe9'], dtype=object)
    # 断言数组的字符串表示形式与预期相符
    assert_equal(repr(x), expected)


class TestContextManager:
    def test_ctx_mgr(self):
        # 测试上下文管理器是否正常工作
        with np.printoptions(precision=2):
            s = str(np.array([2.0]) / 3)
        # 断言计算结果的字符串表示形式是否正确
        assert_equal(s, '[0.67]')

    def test_ctx_mgr_restores(self):
        # 测试打印选项是否被正确还原
        opts = np.get_printoptions()
        with np.printoptions(precision=opts['precision'] - 1,
                             linewidth=opts['linewidth'] - 4):
            pass
        # 断言打印选项是否被恢复到初始状态
        assert_equal(np.get_printoptions(), opts)

    def test_ctx_mgr_exceptions(self):
        # 测试即使出现异常，打印选项也能正确还原
        opts = np.get_printoptions()
        try:
            with np.printoptions(precision=2, linewidth=11):
                raise ValueError
        except ValueError:
            pass
        # 断言打印选项是否被恢复到初始状态
        assert_equal(np.get_printoptions(), opts)

    def test_ctx_mgr_as_smth(self):
        # 测试上下文管理器作为某个对象的行为
        opts = {"precision": 2}
        with np.printoptions(**opts) as ctx:
            saved_opts = ctx.copy()
        # 断言保存的选项是否与预期一致
        assert_equal({k: saved_opts[k] for k in opts}, opts)


@pytest.mark.parametrize("dtype", "bhilqpBHILQPefdgFDG")
@pytest.mark.parametrize("value", [0, 1])
def test_scalar_repr_numbers(dtype, value):
    # 测试数值类型的 NEP 51 标量表示（和旧选项）
    dtype = np.dtype(dtype)
    scalar = np.array(value, dtype=dtype)[()]
    # 断言标量是否为 NumPy 的通用类型
    assert isinstance(scalar, np.generic)

    string = str(scalar)
    repr_string = string.strip("()")  # 复数类型可能有额外的括号
    representation = repr(scalar)
    if dtype.char == "g":
        assert representation == f"np.longdouble('{repr_string}')"
    elif dtype.char == 'G':
        assert representation == f"np.clongdouble('{repr_string}')"
    else:
        normalized_name = np.dtype(f"{dtype.kind}{dtype.itemsize}").type.__name__
        assert representation == f"np.{normalized_name}({repr_string})"

    with np.printoptions(legacy="1.25"):
        assert repr(scalar) == string


@pytest.mark.parametrize("scalar, legacy_repr, representation", [
        (np.True_, "True", "np.True_"),
        (np.bytes_(b'a'), "b'a'", "np.bytes_(b'a')"),
        (np.str_('a'), "'a'", "np.str_('a')"),
        (np.datetime64("2012"),
            "numpy.datetime64('2012')", "np.datetime64('2012')"),
        (np.timedelta64(1), "numpy.timedelta64(1)", "np.timedelta64(1)"),
        (np.void((True, 2), dtype="?,<i8"),
            "(True, 2)",
            "np.void((True, 2), dtype=[('f0', '?'), ('f1', '<i8')])"),
        (np.void((1, 2), dtype="<f8,>f4"),
            "(1., 2.)",
            "np.void((1.0, 2.0), dtype=[('f0', '<f8'), ('f1', '>f4')])"),
        (np.void(b'a'), r"void(b'\x61')", r"np.void(b'\x61')"),
    ])
def test_scalar_repr_special(scalar, legacy_repr, representation):
    # 测试特殊标量类型的字符串表示
    # 测试 NEP 51 标量的字符串表示形式（包括旧版选项）适用于数值类型
    assert repr(scalar) == representation
    # 使用旧版打印选项设置为 "1.25"，再次确认 NEP 51 标量的字符串表示形式
    with np.printoptions(legacy="1.25"):
        assert repr(scalar) == legacy_repr
# 定义一个测试函数，用于测试标量、空值、浮点数和字符串的处理
def test_scalar_void_float_str():
    # 创建一个 NumPy 的 void 类型对象 scalar，其内容为 (1.0, 2.0)
    # 使用 dtype 指定了数据类型为包含两个字段的结构化数组：
    # - 'f0' 字段为小端格式的 64 位浮点数
    # - 'f1' 字段为大端格式的 32 位浮点数
    scalar = np.void((1.0, 2.0), dtype=[('f0', '<f8'), ('f1', '>f4')])
    # 断言将该标量对象转换为字符串后结果应为 "(1.0, 2.0)"
    assert str(scalar) == "(1.0, 2.0)"
```