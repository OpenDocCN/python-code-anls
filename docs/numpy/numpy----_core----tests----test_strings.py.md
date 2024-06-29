# `.\numpy\numpy\_core\tests\test_strings.py`

```
import sys  # 导入 sys 模块，用于访问系统相关的变量和函数
import pytest  # 导入 pytest 模块，用于编写和运行测试用例

import operator  # 导入 operator 模块，提供了对内置运算符的函数形式的访问
import numpy as np  # 导入 NumPy 库，并使用 np 作为别名

from numpy.testing import assert_array_equal, assert_raises, IS_PYPY  # 从 NumPy 测试模块中导入几个断言函数和一个常量

COMPARISONS = [  # 定义一个包含比较运算符、NumPy 函数和符号的列表
    (operator.eq, np.equal, "=="),  # 等于运算符
    (operator.ne, np.not_equal, "!="),  # 不等于运算符
    (operator.lt, np.less, "<"),  # 小于运算符
    (operator.le, np.less_equal, "<="),  # 小于等于运算符
    (operator.gt, np.greater, ">"),  # 大于运算符
    (operator.ge, np.greater_equal, ">="),  # 大于等于运算符
]

MAX = np.iinfo(np.int64).max  # 定义一个常量 MAX，表示 np.int64 的最大值

IS_PYPY_LT_7_3_16 = IS_PYPY and sys.implementation.version < (7, 3, 16)  # 检查当前 Python 是否为 PyPy 且版本小于 7.3.16

@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
def test_mixed_string_comparison_ufuncs_fail(op, ufunc, sym):
    arr_string = np.array(["a", "b"], dtype="S")  # 创建一个包含字符串数组，类型为字节字符串
    arr_unicode = np.array(["a", "c"], dtype="U")  # 创建一个包含字符串数组，类型为 Unicode 字符串

    with pytest.raises(TypeError, match="did not contain a loop"):  # 使用 pytest 断言检查是否抛出 TypeError 异常
        ufunc(arr_string, arr_unicode)

    with pytest.raises(TypeError, match="did not contain a loop"):  # 使用 pytest 断言检查是否抛出 TypeError 异常
        ufunc(arr_unicode, arr_string)

@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
def test_mixed_string_comparisons_ufuncs_with_cast(op, ufunc, sym):
    arr_string = np.array(["a", "b"], dtype="S")  # 创建一个包含字符串数组，类型为字节字符串
    arr_unicode = np.array(["a", "c"], dtype="U")  # 创建一个包含字符串数组，类型为 Unicode 字符串

    # 虽然没有循环，但手动转换是可以接受的：
    res1 = ufunc(arr_string, arr_unicode, signature="UU->?", casting="unsafe")  # 使用 ufunc 执行操作，并指定签名和类型转换方式
    res2 = ufunc(arr_string, arr_unicode, signature="SS->?", casting="unsafe")  # 使用 ufunc 执行操作，并指定签名和类型转换方式

    expected = op(arr_string.astype("U"), arr_unicode)  # 计算预期结果
    assert_array_equal(res1, expected)  # 断言两个数组相等
    assert_array_equal(res2, expected)  # 断言两个数组相等

@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
@pytest.mark.parametrize("dtypes", [
        ("S2", "S2"), ("S2", "S10"),
        ("<U1", "<U1"), ("<U1", ">U1"), (">U1", ">U1"),
        ("<U1", "<U10"), ("<U1", ">U10")])
@pytest.mark.parametrize("aligned", [True, False])
def test_string_comparisons(op, ufunc, sym, dtypes, aligned):
    # 确保第一个视图使用本机字节顺序以保持在 Unicode 范围内
    native_dt = np.dtype(dtypes[0]).newbyteorder("=")
    arr = np.arange(2**15).view(native_dt).astype(dtypes[0])  # 创建一个数组并进行类型转换

    if not aligned:
        # 使数组不对齐：
        new = np.zeros(arr.nbytes + 1, dtype=np.uint8)[1:].view(dtypes[0])
        new[...] = arr
        arr = new

    arr2 = arr.astype(dtypes[1], copy=True)  # 复制并转换第二个数组的类型
    np.random.shuffle(arr2)  # 随机打乱第二个数组
    arr[0] = arr2[0]  # 确保第一个元素匹配

    expected = [op(d1, d2) for d1, d2 in zip(arr.tolist(), arr2.tolist())]  # 计算预期结果
    assert_array_equal(op(arr, arr2), expected)  # 断言两个数组相等
    assert_array_equal(ufunc(arr, arr2), expected)  # 断言两个数组相等
    assert_array_equal(
        np.char.compare_chararrays(arr, arr2, sym, False), expected
    )  # 使用 np.char.compare_chararrays 比较字符数组

    expected = [op(d2, d1) for d1, d2 in zip(arr.tolist(), arr2.tolist())]  # 计算预期结果
    assert_array_equal(op(arr2, arr), expected)  # 断言两个数组相等
    assert_array_equal(ufunc(arr2, arr), expected)  # 断言两个数组相等
    assert_array_equal(
        np.char.compare_chararrays(arr2, arr, sym, False), expected
    )  # 使用 np.char.compare_chararrays 比较字符数组
# 使用 pytest 模块标记此函数为参数化测试函数，测试不同的数据类型组合
@pytest.mark.parametrize("dtypes", [
        ("S2", "S2"), ("S2", "S10"), ("<U1", "<U1"), ("<U1", ">U10")])
def test_string_comparisons_empty(op, ufunc, sym, dtypes):
    # 创建一个空的多维数组 arr，使用给定的数据类型 dtypes[0]
    arr = np.empty((1, 0, 1, 5), dtype=dtypes[0])
    # 创建另一个空的多维数组 arr2，使用给定的数据类型 dtypes[1]
    arr2 = np.empty((100, 1, 0, 1), dtype=dtypes[1])

    # 计算广播后的形状，创建一个空的预期结果数组
    expected = np.empty(np.broadcast_shapes(arr.shape, arr2.shape), dtype=bool)
    # 断言调用 op 函数处理 arr 和 arr2 后的结果与预期结果相等
    assert_array_equal(op(arr, arr2), expected)
    # 断言调用 ufunc 函数处理 arr 和 arr2 后的结果与预期结果相等
    assert_array_equal(ufunc(arr, arr2), expected)
    # 断言调用 np.char.compare_chararrays 比较 arr 和 arr2 的结果与预期结果相等
    assert_array_equal(
        np.char.compare_chararrays(arr, arr2, sym, False), expected
    )


# 使用 pytest 模块标记此函数为参数化测试函数，测试不同的字符串数据类型和浮点数数据类型组合
@pytest.mark.parametrize("str_dt", ["S", "U"])
@pytest.mark.parametrize("float_dt", np.typecodes["AllFloat"])
def test_float_to_string_cast(str_dt, float_dt):
    # 将 float_dt 转换为 NumPy 的数据类型对象
    float_dt = np.dtype(float_dt)
    # 获取 float_dt 的浮点数信息
    fi = np.finfo(float_dt)
    # 创建一个包含 NaN、正无穷、负无穷、最大值和最小值的数组 arr，使用 float_dt 类型
    arr = np.array([np.nan, np.inf, -np.inf, fi.max, fi.min], dtype=float_dt)
    # 创建期望的字符串表示形式列表 expected
    expected = ["nan", "inf", "-inf", str(fi.max), str(fi.min)]
    # 如果 float_dt 的类型是复数，将期望值列表转换为复数形式字符串
    if float_dt.kind == "c":
        expected = [f"({r}+0j)" for r in expected]

    # 将 arr 转换为字符串类型 str_dt，并断言其与期望的结果数组 expected 相等
    res = arr.astype(str_dt)
    assert_array_equal(res, np.array(expected, dtype=str_dt))


# 使用 pytest 模块标记此类为参数化测试类，测试不同的字符串数据类型 dt
@pytest.mark.parametrize("dt", ["S", "U", "T"])
class TestMethods:

    # 使用 pytest 模块标记此方法为参数化测试方法，测试字符串的连接操作
    @pytest.mark.parametrize("in1,in2,out", [
        ("", "", ""),
        ("abc", "abc", "abcabc"),
        ("12345", "12345", "1234512345"),
        ("MixedCase", "MixedCase", "MixedCaseMixedCase"),
        ("12345 \0 ", "12345 \0 ", "12345 \0 12345 \0 "),
        ("UPPER", "UPPER", "UPPERUPPER"),
        (["abc", "def"], ["hello", "world"], ["abchello", "defworld"]),
    ])
    def test_add(self, in1, in2, out, dt):
        # 将输入字符串数组转换为给定的数据类型 dt
        in1 = np.array(in1, dtype=dt)
        in2 = np.array(in2, dtype=dt)
        out = np.array(out, dtype=dt)
        # 断言 np.strings.add 函数对 in1 和 in2 进行连接操作后的结果与期望的结果 out 相等
        assert_array_equal(np.strings.add(in1, in2), out)

    # 使用 pytest 模块标记此方法为参数化测试方法，测试字符串的重复操作
    @pytest.mark.parametrize("in1,in2,out", [
        ("abc", 3, "abcabcabc"),
        ("abc", 0, ""),
        ("abc", -1, ""),
        (["abc", "def"], [1, 4], ["abc", "defdefdefdef"]),
    ])
    def test_multiply(self, in1, in2, out, dt):
        # 将输入字符串数组转换为给定的数据类型 dt
        in1 = np.array(in1, dtype=dt)
        out = np.array(out, dtype=dt)
        # 断言 np.strings.multiply 函数对 in1 进行重复操作后的结果与期望的结果 out 相等
        assert_array_equal(np.strings.multiply(in1, in2), out)

    # 使用 pytest 模块标记此方法为测试方法，测试 np.strings.multiply 函数抛出异常的情况
    def test_multiply_raises(self, dt):
        # 断言 np.strings.multiply 函数对不支持的数据类型 dt 抛出 TypeError 异常
        with pytest.raises(TypeError, match="unsupported type"):
            np.strings.multiply(np.array("abc", dtype=dt), 3.14)

        # 断言 np.strings.multiply 函数对内存溢出情况抛出 MemoryError 异常
        with pytest.raises(MemoryError):
            np.strings.multiply(np.array("abc", dtype=dt), sys.maxsize)

    # 使用 pytest 模块标记此方法为参数化测试方法，测试整数数据类型与字符串的重复操作
    @pytest.mark.parametrize("i_dt", [np.int8, np.int16, np.int32,
                                      np.int64, np.int_])
    def test_multiply_integer_dtypes(self, i_dt, dt):
        # 创建一个数组 a，包含字符串 "abc"，使用给定的数据类型 dt
        a = np.array("abc", dtype=dt)
        # 创建一个整数数组 i，包含整数 3，使用给定的整数数据类型 i_dt
        i = np.array(3, dtype=i_dt)
        # 创建期望的结果数组 res，包含重复三次的字符串 "abcabcabc"，使用给定的数据类型 dt
        res = np.array("abcabcabc", dtype=dt)
        # 断言 np.strings.multiply 函数对 a 和 i 进行重复操作后的结果与期望的结果 res 相等
        assert_array_equal(np.strings.multiply(a, i), res)
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化函数，用于测试 np.strings.isalpha 方法
    @pytest.mark.parametrize("in_,out", [
        ("", False),             # 空字符串返回 False
        ("a", True),             # 单个字母返回 True
        ("A", True),             # 单个大写字母返回 True
        ("\n", False),           # 包含换行符返回 False
        ("abc", True),           # 全部是字母返回 True
        ("aBc123", False),       # 包含非字母字符返回 False
        ("abc\n", False),        # 包含换行符返回 False
        (["abc", "aBc123"], [True, False]),  # 多个输入分别测试
    ])
    def test_isalpha(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isalpha(in_), out)
    
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化函数，用于测试 np.strings.isalnum 方法
    @pytest.mark.parametrize("in_,out", [
        ('', False),             # 空字符串返回 False
        ('a', True),             # 单个字母返回 True
        ('A', True),             # 单个大写字母返回 True
        ('\n', False),           # 包含换行符返回 False
        ('123abc456', True),     # 包含数字和字母返回 True
        ('a1b3c', True),         # 包含字母和数字返回 True
        ('aBc000 ', False),      # 包含非字母或数字字符返回 False
        ('abc\n', False),        # 包含换行符返回 False
    ])
    def test_isalnum(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isalnum(in_), out)
    
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化函数，用于测试 np.strings.isdigit 方法
    @pytest.mark.parametrize("in_,out", [
        ("", False),             # 空字符串返回 False
        ("a", False),            # 单个字母返回 False
        ("0", True),             # 单个数字字符返回 True
        ("012345", True),        # 包含数字字符返回 True
        ("012345a", False),      # 包含非数字字符返回 False
        (["a", "012345"], [False, True]),  # 多个输入分别测试
    ])
    def test_isdigit(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isdigit(in_), out)
    
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化函数，用于测试 np.strings.isspace 方法
    @pytest.mark.parametrize("in_,out", [
        ("", False),             # 空字符串返回 False
        ("a", False),            # 单个字母返回 False
        ("1", False),            # 单个数字字符返回 False
        (" ", True),             # 单个空格返回 True
        ("\t", True),            # 包含制表符返回 True
        ("\r", True),            # 包含回车符返回 True
        ("\n", True),            # 包含换行符返回 True
        (" \t\r \n", True),      # 包含多种空白字符返回 True
        (" \t\r\na", False),     # 包含非空白字符返回 False
        (["\t1", " \t\r \n"], [False, True])  # 多个输入分别测试
    ])
    def test_isspace(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isspace(in_), out)
    
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化函数，用于测试 np.strings.islower 方法
    @pytest.mark.parametrize("in_,out", [
        ('', False),             # 空字符串返回 False
        ('a', True),             # 全部小写字母返回 True
        ('A', False),            # 包含大写字母返回 False
        ('\n', False),           # 包含换行符返回 False
        ('abc', True),           # 全部小写字母返回 True
        ('aBc', False),          # 包含大写字母返回 False
        ('abc\n', True),         # 包含换行符返回 True
    ])
    def test_islower(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.islower(in_), out)
    
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化函数，用于测试 np.strings.isupper 方法
    @pytest.mark.parametrize("in_,out", [
        ('', False),             # 空字符串返回 False
        ('a', False),            # 包含小写字母返回 False
        ('A', True),             # 全部大写字母返回 True
        ('\n', False),           # 包含换行符返回 False
        ('ABC', True),           # 全部大写字母返回 True
        ('AbC', False),          # 包含小写字母返回 False
        ('ABC\n', True),         # 包含换行符返回 True
    ])
    def test_isupper(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isupper(in_), out)
    
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化函数，用于测试 np.strings.istitle 方法
    @pytest.mark.parametrize("in_,out", [
        ('', False),                           # 空字符串返回 False
        ('a', False),                          # 单个字母返回 False
        ('A', False),                          # 单个大写字母返回 False
        ('\n', False),                         # 包含换行符返回 False
        ('abc', True),                         # 标题化字符串返回 True
        ('aBc', False),                        # 非标题化字符串返回 False
        ('abc\n', True),                       # 包含换行符的标题化字符串返回 True
        ('A Titlecased Line', True),           # 标题化行返回 True
        ('A\nTitlecased Line', True),          # 包含换行符的标题化行返回 True
        ('A Titlecased, Line', True),          # 包含逗号的标题化行返回 True
        ('Not a capitalized String', False),   # 非标题化字符串返回 False
        ('Not\ta Titlecase String', False),    # 非标题化字符串返回 False
        ('Not--a Titlecase String', False),    # 非标题化字符串返回 False
        ('NOT', False),                        # 全大写字符串返回 False
    ])
    def test_istitle(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.istitle(in_), out)
    @pytest.mark.parametrize("in_,out", [
        ("", 0),  # 参数化测试的输入为空字符串时，预期输出为0
        ("abc", 3),  # 参数化测试的输入为"abc"时，预期输出为3
        ("12345", 5),  # 参数化测试的输入为"12345"时，预期输出为5
        ("MixedCase", 9),  # 参数化测试的输入为"MixedCase"时，预期输出为9
        ("12345 \x00 ", 8),  # 参数化测试的输入为带有空字符的字符串时，预期输出为8
        ("UPPER", 5),  # 参数化测试的输入为"UPPER"时，预期输出为5
        (["abc", "12345 \x00 "], [3, 8]),  # 参数化测试的输入为包含两个字符串的列表时，预期输出为对应的长度列表[3, 8]
    ])
    def test_str_len(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)  # 将输入转换为指定数据类型的 NumPy 数组
        assert_array_equal(np.strings.str_len(in_), out)  # 断言调用 np.strings.str_len 方法后的输出与预期输出相等

    @pytest.mark.parametrize("a,sub,start,end,out", [
        ("abcdefghiabc", "abc", 0, None, 0),  # 参数化测试的输入为字符串 "abcdefghiabc"，查找子字符串 "abc"，从索引 0 开始，到末尾，预期输出为索引 0
        ("abcdefghiabc", "abc", 1, None, 9),  # 参数化测试的输入为字符串 "abcdefghiabc"，查找子字符串 "abc"，从索引 1 开始，到末尾，预期输出为索引 9
        ("abcdefghiabc", "def", 4, None, -1),  # 参数化测试的输入为字符串 "abcdefghiabc"，查找子字符串 "def"，从索引 4 开始，到末尾，预期输出为索引 -1
        ("abc", "", 0, None, 0),  # 参数化测试的输入为字符串 "abc"，查找空子字符串 ""，从索引 0 开始，到末尾，预期输出为索引 0
        ("abc", "", 3, None, 3),  # 参数化测试的输入为字符串 "abc"，查找空子字符串 ""，从索引 3 开始，到末尾，预期输出为索引 3
        ("abc", "", 4, None, -1),  # 参数化测试的输入为字符串 "abc"，查找空子字符串 ""，从索引 4 开始，到末尾，预期输出为索引 -1
        ("rrarrrrrrrrra", "a", 0, None, 2),  # 参数化测试的输入为字符串 "rrarrrrrrrrra"，查找子字符串 "a"，从索引 0 开始，到末尾，预期输出为索引 2
        ("rrarrrrrrrrra", "a", 4, None, 12),  # 参数化测试的输入为字符串 "rrarrrrrrrrra"，查找子字符串 "a"，从索引 4 开始，到末尾，预期输出为索引 12
        ("rrarrrrrrrrra", "a", 4, 6, -1),  # 参数化测试的输入为字符串 "rrarrrrrrrrra"，查找子字符串 "a"，从索引 4 开始，到索引 6 结束，预期输出为索引 -1
        ("", "", 0, None, 0),  # 参数化测试的输入为空字符串 ""，查找空子字符串 ""，从索引 0 开始，到末尾，预期输出为索引 0
        ("", "", 1, 1, -1),  # 参数化测试的输入为空字符串 ""，查找空子字符串 ""，从索引 1 开始，到索引 1 结束，预期输出为索引 -1
        ("", "", MAX, 0, -1),  # 参数化测试的输入为空字符串 ""，查找空子字符串 ""，从索引 MAX 开始，到索引 0 结束，预期输出为索引 -1
        ("", "xx", 0, None, -1),  # 参数化测试的输入为空字符串 ""，查找子字符串 "xx"，从索引 0 开始，到末尾，预期输出为索引 -1
        ("", "xx", 1, 1, -1),  # 参数化测试的输入为空字符串 ""，查找子字符串 "xx"，从索引 1 开始，到索引 1 结束，预期输出为索引 -1
        ("", "xx", MAX, 0, -1),  # 参数化测试的输入为空字符串 ""，查找子字符串 "xx"，从索引 MAX 开始，到索引 0 结束，预期输出为索引 -1
        pytest.param(99*"a" + "b", "b", 0, None, 99,
                     id="99*a+b-b-0-None-99"),  # 参数化测试的输入为字符串 "99*a+b" + "b"，查找子字符串 "b"，从索引 0 开始，到末尾，预期输出为索引 99，用例标识为 "99*a+b-b-0-None-99"
        pytest.param(98*"a" + "ba", "ba", 0, None, 98,
                     id="98*a+ba-ba-0-None-98"),  # 参数化测试的输入为字符串 "98*a+ba" + "ba"，查找子字符串 "ba"，从索引 0 开始，到末尾，预期输出为索引 98，用例标识为 "98*a+ba-ba-0-None-98"
        pytest.param(100*"a", "b", 0, None, -1,
                     id="100*a-b-0-None--1"),  # 参数化测试的输入为字符串 "100*a"，查找子字符串 "b"，从索引 0 开始，到末尾，预期输出为索引 -1，用例标识为 "100*a-b-0-None--1"
        pytest.param(30000*"a" + 100*"b", 100*"b", 0, None, 30000,
                     id="30000*a+100*b-100*b-0-None-30000"),  # 参数化测试的输入为字符串 "30000*a" + "100*b"，查找子字符串 "100*b"，从索引 0 开始，到末尾，预期输出为索引 30000，用例标识为 "30000*a+100*b-100*b-0-None-30000"
        pytest.param(30000*"a", 100*"b", 0, None, -1,
                     id="30000*a-100*b-0-None--1"),  # 参数化测试的输入为字符串 "30000*a"，查找子字符串 "100*b"，从索引 0 开始，到末尾，预期输出为索引 -1，用例标识为 "30000*a-100*b-0-None--1"
        pytest.param(15000*"a" + 15000*"b", 15000*"b", 0, None, 15000,
                     id="15000*a+15000*b-15000*b-0-None-15000"),  # 参数化测试的输入为字符串 "15000*a" + "15000*b"，查找子字符串 "15000*b"，从索引 0 开始，到末尾，预期输出为索引 15000，用例标识为 "15000*a+15000*b-15000*b-0-None-15000"
        pytest.param(15000*"a" + 15000*"b", 15000*"c", 0, None, -1,
                     id="15000*a+15000*b-15000*c-0-None--1"),  # 参数化测试的输入为字符串 "15000*a" + "15000*b"，查找子字符串 "15000*c"，从索引 0 开始，到末尾，预期输出为索引 -1，用例标识为 "15000*a+15000*b-15000*c-0-None--1"
        (["abcdefghiabc", "rrarrr
    # 定义一个测试方法，用于测试 np.strings.rfind 函数的行为
    def test_rfind(self, a, sub, start, end, out, dt):
        # 如果输入的数组包含非 ASCII 字符，并且数据类型为字符串 ("S")，则跳过测试
        if "😊" in a and dt == "S":
            pytest.skip("Bytes dtype does not support non-ascii input")
        # 将输入的 a 转换为 NumPy 数组，指定数据类型为 dt
        a = np.array(a, dtype=dt)
        # 将输入的 sub 转换为 NumPy 数组，指定数据类型为 dt
        sub = np.array(sub, dtype=dt)
        # 断言 np.strings.rfind 函数对 a 应用 sub、start 和 end 参数后的结果等于预期输出 out
        assert_array_equal(np.strings.rfind(a, sub, start, end), out)

    # 使用 pytest.mark.parametrize 注释来定义多组参数化测试
    @pytest.mark.parametrize("a,sub,start,end,out", [
        ("aaa", "a", 0, None, 3),  # 在字符串 "aaa" 中查找字符 "a"，从索引 0 开始，预期找到 3
        ("aaa", "b", 0, None, 0),  # 在字符串 "aaa" 中查找字符 "b"，从索引 0 开始，预期找到 0
        ("aaa", "a", 1, None, 2),  # 在字符串 "aaa" 中查找字符 "a"，从索引 1 开始，预期找到 2
        ("aaa", "a", 10, None, 0),  # 在字符串 "aaa" 中查找字符 "a"，从索引 10 开始，预期找到 0
        ("aaa", "a", -1, None, 1),  # 在字符串 "aaa" 中查找字符 "a"，从索引 -1 开始，预期找到 1
        ("aaa", "a", -10, None, 3),  # 在字符串 "aaa" 中查找字符 "a"，从索引 -10 开始，预期找到 3
        ("aaa", "a", 0, 1, 1),  # 在字符串 "aaa" 的索引 0 到 1 区间内查找字符 "a"，预期找到 1
        ("aaa", "a", 0, 10, 3),  # 在字符串 "aaa" 的索引 0 到 10 区间内查找字符 "a"，预期找到 3
        ("aaa", "a", 0, -1, 2),  # 在字符串 "aaa" 的索引 0 到 -1 区间内查找字符 "a"，预期找到 2
        ("aaa", "a", 0, -10, 0),  # 在字符串 "aaa" 的索引 0 到 -10 区间内查找字符 "a"，预期找到 0
        ("aaa", "", 1, None, 3),  # 在字符串 "aaa" 中查找空字符串，从索引 1 开始，预期找到 3
        ("aaa", "", 3, None, 1),  # 在字符串 "aaa" 中查找空字符串，从索引 3 开始，预期找到 1
        ("aaa", "", 10, None, 0),  # 在字符串 "aaa" 中查找空字符串，从索引 10 开始，预期找到 0
        ("aaa", "", -1, None, 2),  # 在字符串 "aaa" 中查找空字符串，从索引 -1 开始，预期找到 2
        ("aaa", "", -10, None, 4),  # 在字符串 "aaa" 中查找空字符串，从索引 -10 开始，预期找到 4
        ("aaa", "aaaa", 0, None, 0),  # 在字符串 "aaa" 中查找 "aaaa"，从索引 0 开始，预期找到 0
        pytest.param(98*"a" + "ba", "ba", 0, None, 1, id="98*a+ba-ba-0-None-1"),  # 在字符串 "98*a+ba" 中查找 "ba"，从索引 0 开始，预期找到 1
        pytest.param(30000*"a" + 100*"b", 100*"b", 0, None, 1, id="30000*a+100*b-100*b-0-None-1"),  # 在字符串 "30000*a+100*b" 中查找 "100*b"，从索引 0 开始，预期找到 1
        pytest.param(30000*"a", 100*"b", 0, None, 0, id="30000*a-100*b-0-None-0"),  # 在字符串 "30000*a" 中查找 "100*b"，从索引 0 开始，预期找到 0
        pytest.param(30000*"a" + 100*"ab", "ab", 0, None, 100, id="30000*a+100*ab-ab-0-None-100"),  # 在字符串 "30000*a+100*ab" 中查找 "ab"，从索引 0 开始，预期找到 100
        pytest.param(15000*"a" + 15000*"b", 15000*"b", 0, None, 1, id="15000*a+15000*b-15000*b-0-None-1"),  # 在字符串 "15000*a+15000*b" 中查找 "15000*b"，从索引 0 开始，预期找到 1
        pytest.param(15000*"a" + 15000*"b", 15000*"c", 0, None, 0, id="15000*a+15000*b-15000*c-0-None-0"),  # 在字符串 "15000*a+15000*b" 中查找 "15000*c"，从索引 0 开始，预期找到 0
        ("", "", 0, None, 1),  # 在空字符串中查找空字符串，从索引 0 开始，预期找到 1
        ("", "", 1, 1, 0),  # 在空字符串中查找空字符串，从索引 1 到 1 区间内，预期找到 0
        ("", "", MAX, 0, 0),  # 在空字符串中查找空字符串，从索引 MAX 开始，预期找到 0
        ("", "xx", 0, None, 0),  # 在空字符串中查找 "xx"，从索引 0 开始，预期找到 0
        ("", "xx", 1, 1, 0),  # 在空字符串中查找 "xx"，从索引 1 到 1 区间内，预期找到 0
        ("", "xx", MAX, 0, 0),  # 在空字符串中查找 "xx"，从索引 MAX 开始，预期找到 0
        (["aaa", ""], ["a", ""], [0, 0], None, [3, 1]),  # 在字符串数组 ["aaa", ""] 中查找 ["a", ""]，预期找到 [3, 1]
        ("Ae¢☃€ 😊" * 100, "😊", 0, None, 100),  # 在重复 "Ae¢☃€ 😊" 100 次的字符串中查找 "😊"，预期找到 100
    ])
    # 定义一个测试方法，用于测试 np.strings.count 函数的行为
    def test_count(self, a, sub, start, end, out, dt):
        # 如果输入的数组包含非 ASCII 字符，并且数据类型为字符串 ("S")，则跳过测试
        if "😊" in a and dt == "S":
            pytest.skip("Bytes dtype does not support non-ascii input")
        # 将输入的 a 转换为 NumPy 数组，指定数据类型为 dt
        a = np.array(a, dtype=dt)
        # 将输入的 sub 转换为 NumPy 数组，指定数据类型为 dt
        sub = np.array(sub, dtype=dt)
        # 断言 np.strings.count 函数对 a 应用 sub、start 和 end 参数后的结果等于预期输出 out
        assert_array_equal(np.strings.count(a, sub, start, end), out)
    @pytest.mark.parametrize("a,prefix,start,end,out", [
        # 参数化测试函数 test_startswith，测试字符串 a 是否以指定的 prefix 开头
        ("hello", "he", 0, None, True),  # 预期 "hello" 以 "he" 开头，返回 True
        ("hello", "hello", 0, None, True),  # 预期 "hello" 以 "hello" 开头，返回 True
        ("hello", "hello world", 0, None, False),  # 预期 "hello" 不以 "hello world" 开头，返回 False
        ("hello", "", 0, None, True),  # 预期 "hello" 以空字符串开头，返回 True
        ("hello", "ello", 0, None, False),  # 预期 "hello" 不以 "ello" 开头，返回 False
        ("hello", "ello", 1, None, True),  # 预期 "hello" 从索引 1 开始以 "ello" 开头，返回 True
        ("hello", "o", 4, None, True),  # 预期 "hello" 从索引 4 开始以 "o" 开头，返回 True
        ("hello", "o", 5, None, False),  # 预期 "hello" 不以 "o" 开头，因为索引 5 超出字符串长度，返回 False
        ("hello", "", 5, None, True),  # 预期 "hello" 以空字符串开头，返回 True
        ("hello", "lo", 6, None, False),  # 预期 "hello" 不以 "lo" 开头，因为索引 6 超出字符串长度，返回 False
        ("helloworld", "lowo", 3, None, True),  # 预期 "helloworld" 从索引 3 开始以 "lowo" 开头，返回 True
        ("helloworld", "lowo", 3, 7, True),  # 预期 "helloworld" 从索引 3 到 6 之间以 "lowo" 开头，返回 True
        ("helloworld", "lowo", 3, 6, False),  # 预期 "helloworld" 从索引 3 到 5 之间不以 "lowo" 开头，返回 False
        ("", "", 0, 1, True),  # 空字符串以空字符串开头，返回 True
        ("", "", 0, 0, True),  # 空字符串以空字符串开头，返回 True
        ("", "", 1, 0, False),  # 空字符串从索引 1 开始不以空字符串开头，返回 False
        ("hello", "he", 0, -1, True),  # 预期 "hello" 以 "he" 开头，返回 True
        ("hello", "he", -53, -1, True),  # 预期 "hello" 以 "he" 开头，返回 True
        ("hello", "hello", 0, -1, False),  # 预期 "hello" 不以 "hello" 开头，返回 False
        ("hello", "hello world", -1, -10, False),  # 预期 "hello" 不以 "hello world" 开头，返回 False
        ("hello", "ello", -5, None, False),  # 预期 "hello" 不以 "ello" 开头，索引 -5 超出字符串长度，返回 False
        ("hello", "ello", -4, None, True),  # 预期 "hello" 从倒数第 4 位开始以 "ello" 开头，返回 True
        ("hello", "o", -2, None, False),  # 预期 "hello" 不以 "o" 开头，因为索引 -2 对应字符 'l'，返回 False
        ("hello", "o", -1, None, True),  # 预期 "hello" 从倒数第 1 位开始以 "o" 开头，返回 True
        ("hello", "", -3, -3, True),  # 预期 "hello" 以空字符串开头，返回 True
        ("hello", "lo", -9, None, False),  # 预期 "hello" 不以 "lo" 开头，索引 -9 超出字符串长度，返回 False
        (["hello", ""], ["he", ""], [0, 0], None, [True, True]),  # 预期 ["hello", ""] 以 ["he", ""] 分别开头，返回 [True, True]
    ])
    def test_startswith(self, a, prefix, start, end, out, dt):
        # 将输入的 a 和 prefix 转换为 NumPy 数组，使用指定的数据类型 dt
        a = np.array(a, dtype=dt)
        prefix = np.array(prefix, dtype=dt)
        # 断言使用 np.strings.startswith 函数，比较结果是否与预期输出 out 相同
        assert_array_equal(np.strings.startswith(a, prefix, start, end), out)
    
    
    @pytest.mark.parametrize("a,suffix,start,end,out", [
        # 参数化测试函数 test_endswith，测试字符串 a 是否以指定的 suffix 结尾
        ("hello", "lo", 0, None, True),  # 预期 "hello" 以 "lo" 结尾，返回 True
        ("hello", "he", 0, None, False),  # 预期 "hello" 不以 "he" 结尾，返回 False
        ("hello", "", 0, None, True),  # 预期 "hello" 以空字符串结尾，返回 True
        ("hello", "hello world", 0, None, False),  # 预期 "hello" 不以 "hello world" 结尾，返回 False
        ("helloworld", "worl", 0, None, False),  # 预期 "helloworld" 不以 "worl" 结尾，返回 False
        ("helloworld", "worl", 3, 9, True),  # 预期 "helloworld" 从索引 3 到 8 之间以 "worl" 结尾，返回 True
        ("helloworld", "world", 3, 12, True),  # 预期 "helloworld" 从索引 3 到 11 之间以 "world" 结尾，返回 True
        ("helloworld", "lowo", 1, 7, True),  # 预期 "helloworld" 从索引 1 到 6 之间以 "lowo" 结尾，返回 True
        ("helloworld", "lowo", 2, 7, True),  # 预期 "helloworld" 从索引 2 到 6 之间以 "lowo" 结尾，返回 True
        ("helloworld", "lowo", 3, 7, True),  # 预期 "helloworld" 从索引 3 到 6 之间以 "lowo" 结尾，返回 True
        ("helloworld", "lowo", 4, 7, False),  # 预期 "helloworld" 从索引 4 到 6 之间不以 "lowo" 结尾，返回 False
        ("helloworld", "lowo", 3, 8, False),  # 预期 "helloworld" 从索引 3 到 7 之间不以 "lowo" 结尾，返回 False
        ("ab", "ab", 0, 1, False),  # 预期 "ab" 不以 "ab" 结尾，因为字符串长度为 2，返回 False
        ("ab", "ab", 0, 0, False),  # 预期 "ab" 不以 "ab" 结尾，因为索引 0 对应字符 'a'，返回 False
        ("", "", 0, 1, True),  # 空字符串以空字符串结尾，返回 True
        ("", "", 0, 0, True),  # 空字符串以空字符串结尾，返回 True
        ("", "", 1, 0, False),  # 空字符串从索引 1 开始不以空字符串结尾，返回 False
        ("hello", "lo", -2, None, True),  # 预期 "hello" 以 "lo" 结尾，返回 True
        ("hello", "he", -2, None, False),  # 预期 "hello" 不以 "he" 结尾，因为索引 -
    @pytest.mark.parametrize("a,chars,out", [
        # 参数化测试：a 是输入字符串或字符串列表，chars 是要剥离的字符集合，out 是预期的结果
        ("", None, ""),
        ("   hello   ", None, "hello   "),
        ("hello", None, "hello"),
        (" \t\n\r\f\vabc \t\n\r\f\v", None, "abc \t\n\r\f\v"),
        (["   hello   ", "hello"], None, ["hello   ", "hello"]),
        ("", "", ""),
        ("", "xyz", ""),
        ("hello", "", "hello"),
        (["hello    ", "abcdefghijklmnop"], None,
         ["hello", "abcdefghijklmnop"]),
        ("xyzzyhelloxyzzy", "xyz", "xyzzyhello"),
        ("hello", "xyz", "hello"),
        ("xyxz", "xyxz", ""),
        ("xyxzx", "x", "xyxz"),
        (["xyzzyhelloxyzzy", "hello"], ["xyz", "xyz"],
         ["xyzzyhello", "hello"]),
    ])
    def test_lstrip(self, a, chars, out, dt):
        # 将输入参数 a 转换为 numpy 数组，指定数据类型为 dt
        a = np.array(a, dtype=dt)
        # 如果 chars 不为 None，则将其转换为 numpy 数组，数据类型为 dt
        if chars is not None:
            chars = np.array(chars, dtype=dt)
        # 将预期输出 out 转换为 numpy 数组，数据类型为 dt
        out = np.array(out, dtype=dt)
        # 断言调用 np.strings.lstrip 方法后的结果与预期输出相等
        assert_array_equal(np.strings.lstrip(a, chars), out)

    @pytest.mark.parametrize("a,chars,out", [
        # 参数化测试：a 是输入字符串或字符串列表，chars 是要剥离的字符集合，out 是预期的结果
        ("", None, ""),
        ("   hello   ", None, "   hello"),
        ("hello", None, "hello"),
        (" \t\n\r\f\vabc \t\n\r\f\v", None, " \t\n\r\f\vabc"),
        (["   hello   ", "hello"], None, ["   hello", "hello"]),
        ("", "", ""),
        ("", "xyz", ""),
        ("hello", "", "hello"),
        (["hello    ", "abcdefghijklmnop"], None,
         ["hello", "abcdefghijklmnop"]),
        ("xyzzyhelloxyzzy", "xyz", "xyzzyhello"),
        ("hello", "xyz", "hello"),
        ("xyxz", "xyxz", ""),
        ("xyxzx", "x", "xyxz"),
        (["xyzzyhelloxyzzy", "hello"], ["xyz", "xyz"],
         ["xyzzyhello", "hello"]),
    ])
    def test_rstrip(self, a, chars, out, dt):
        # 将输入参数 a 转换为 numpy 数组，指定数据类型为 dt
        a = np.array(a, dtype=dt)
        # 如果 chars 不为 None，则将其转换为 numpy 数组，数据类型为 dt
        if chars is not None:
            chars = np.array(chars, dtype=dt)
        # 将预期输出 out 转换为 numpy 数组，数据类型为 dt
        out = np.array(out, dtype=dt)
        # 断言调用 np.strings.rstrip 方法后的结果与预期输出相等
        assert_array_equal(np.strings.rstrip(a, chars), out)

    @pytest.mark.parametrize("a,chars,out", [
        # 参数化测试：a 是输入字符串或字符串列表，chars 是要剥离的字符集合，out 是预期的结果
        ("", None, ""),
        ("   hello   ", None, "hello"),
        ("hello", None, "hello"),
        (" \t\n\r\f\vabc \t\n\r\f\v", None, "abc"),
        (["   hello   ", "hello"], None, ["hello", "hello"]),
        ("", "", ""),
        ("", "xyz", ""),
        ("hello", "", "hello"),
        ("xyzzyhelloxyzzy", "xyz", "hello"),
        ("hello", "xyz", "hello"),
        ("xyxz", "xyxz", ""),
        ("xyxzx", "x", "yxz"),
        (["xyzzyhelloxyzzy", "hello"], ["xyz", "xyz"],
         ["hello", "hello"]),
    ])
    def test_strip(self, a, chars, out, dt):
        # 将输入参数 a 转换为 numpy 数组，指定数据类型为 dt
        a = np.array(a, dtype=dt)
        # 如果 chars 不为 None，则将其转换为 numpy 数组，数据类型为 dt
        if chars is not None:
            chars = np.array(chars, dtype=dt)
        # 将预期输出 out 转换为 numpy 数组，数据类型为 dt
        out = np.array(out, dtype=dt)
        # 断言调用 np.strings.strip 方法后的结果与预期输出相等
        assert_array_equal(np.strings.strip(a, chars), out)
    def test_replace(self, buf, old, new, count, res, dt):
        # 如果 buf 中包含表情符号 "😊" 并且 dt 为 "S" 类型，则跳过测试，因为字节数据类型不支持非 ASCII 输入
        if "😊" in buf and dt == "S":
            pytest.skip("Bytes dtype does not support non-ascii input")
        # 将 buf 转换为 numpy 数组，指定数据类型为 dt
        buf = np.array(buf, dtype=dt)
        # 将 old 转换为 numpy 数组，指定数据类型为 dt
        old = np.array(old, dtype=dt)
        # 将 new 转换为 numpy 数组，指定数据类型为 dt
        new = np.array(new, dtype=dt)
        # 将 res 转换为 numpy 数组，指定数据类型为 dt
        res = np.array(res, dtype=dt)
        # 断言调用 np.strings.replace 函数后的结果是否与预期的 res 数组相等
        assert_array_equal(np.strings.replace(buf, old, new, count), res)

    @pytest.mark.parametrize("buf,sub,start,end,res", [
        # 参数化测试用例，测试空字符串替换，预期返回值为 0
        ("abcdefghiabc", "", 0, None, 0),
        # 参数化测试用例，测试子字符串 "def" 的索引查找，预期返回值为 3
        ("abcdefghiabc", "def", 0, None, 3),
        # 参数化测试用例，测试从头开始查找子字符串 "abc" 的索引，预期返回值为 0
        ("abcdefghiabc", "abc", 0, None, 0),
        # 参数化测试用例，测试从索引 1 开始查找子字符串 "abc" 的索引，预期返回值为 9
        ("abcdefghiabc", "abc", 1, None, 9),
    ])
    def test_index(self, buf, sub, start, end, res, dt):
        # 将 buf 转换为 numpy 数组，指定数据类型为 dt
        buf = np.array(buf, dtype=dt)
        # 将 sub 转换为 numpy 数组，指定数据类型为 dt
        sub = np.array(sub, dtype=dt)
        # 断言调用 np.strings.index 函数后的结果是否与预期的 res 值相等
        assert_array_equal(np.strings.index(buf, sub, start, end), res)

    @pytest.mark.parametrize("buf,sub,start,end", [
        # 参数化测试用例，测试在字符串中查找不存在的子字符串 "hib"，预期引发 ValueError 异常并提示 "substring not found"
        ("abcdefghiabc", "hib", 0, None),
        # 参数化测试用例，测试在字符串中查找子字符串 "abc"，从索引 1 开始，预期引发 ValueError 异常并提示 "substring not found"
        ("abcdefghiab", "abc", 1, None),
        # 参数化测试用例，测试在字符串中查找子字符串 "ghi"，从索引 8 开始，预期引发 ValueError 异常并提示 "substring not found"
        ("abcdefghi", "ghi", 8, None),
        # 参数化测试用例，测试在字符串中查找子字符串 "ghi"，从索引 -1 开始（从末尾向前查找），预期引发 ValueError 异常并提示 "substring not found"
        ("abcdefghi", "ghi", -1, None),
        # 参数化测试用例，测试在字符串中查找子字符串 "a"，从索引 4 到 6 区间内查找，预期引发 ValueError 异常并提示 "substring not found"
        ("rrarrrrrrrrra", "a", 4, 6),
    ])
    def test_index_raises(self, buf, sub, start, end, dt):
        # 将 buf 转换为 numpy 数组，指定数据类型为 dt
        buf = np.array(buf, dtype=dt)
        # 将 sub 转换为 numpy 数组，指定数据类型为 dt
        sub = np.array(sub, dtype=dt)
        # 使用 pytest 检查调用 np.strings.index 函数时是否引发 ValueError 异常，并检查异常信息是否包含 "substring not found"
        with pytest.raises(ValueError, match="substring not found"):
            np.strings.index(buf, sub, start, end)

    @pytest.mark.parametrize("buf,sub,start,end,res", [
        # 参数化测试用例，测试空字符串替换，预期返回值为 12（字符串长度）
        ("abcdefghiabc", "", 0, None, 12),
        # 参数化测试用例，测试子字符串 "def" 的反向索引查找，预期返回值为 3
        ("abcdefghiabc", "def", 0, None, 3),
        # 参数化测试用例，测试从头开始查找子字符串 "abc" 的反向索引，预期返回值为 9
        ("abcdefghiabc", "abc", 0, None, 9),
        # 参数化测试用例，测试从头开始查找子字符串 "abc" 的反向索引，但限制查找范围为索引 0 到 -1，预期返回值为 0
        ("abcdefghiabc", "abc", 0, -1, 0),
    ])
    def test_rindex(self, buf, sub, start, end, res, dt):
        # 将 buf 转换为 numpy 数组，指定数据类型为 dt
        buf = np.array(buf, dtype=dt)
        # 将 sub 转换为 numpy 数组，指定数据类型为 dt
        sub = np.array(sub, dtype=dt)
        # 断言调用 np.strings.rindex 函数后的结果是否与预期的 res 数组相等
        assert_array_equal(np.strings.rindex(buf, sub, start, end), res)

    @pytest.mark.parametrize("buf,sub,start,end", [
        # 参数化测试用例，测试在字符串中反向查找不存在的子字符串 "hib"，预期引发 ValueError 异常并提示 "substring not found"
        ("abcdefghiabc", "hib", 0, None),
        # 参数化测试用例，测试在字符串中反向查找子字符串 "def"，从索引 1 开始，预期引发 ValueError 异常并提示 "substring not found"
        ("defghiabc", "def", 1, None),
        # 参数化测试用例，测试在字符串中反向查找子字符串 "abc"，但限制查找范围为索引 0 到 -1，预期引发 ValueError 异常并提示 "substring not found"
        ("defghiabc", "abc", 0, -1),
        # 参数化测试用例，测试在字符串中反向查找子字符串 "ghi"，从索引 0 到 8 区间内查找，预期引发 ValueError 异常并提示 "substring not found"
        ("abcdefghi", "ghi", 0, 8),
        # 参数化测试用例，测试在字符串中反向查找子字符串 "ghi"，从索引 0 到 -1 区间内查找，预期引发 ValueError 异常并提示 "substring not found"
        ("abcdefghi", "ghi", 0, -1),
        # 参数化测试用例，测试在字符串中反向查找子字符串 "a"，从索引 4 到 6 区间内查找，预期引发 ValueError 异常并提示 "substring not found"
        ("rrarrrrrrrrra", "a", 4, 6),
    ])
    def test_rindex_raises(self, buf, sub, start, end, dt):
        # 将 buf 转换为 numpy 数组，指定数据类型为 dt
        buf = np.array(buf, dtype=dt)
        # 将 sub 转换为 numpy 数组，指定数据类型为 dt
        sub = np.array(sub, dtype=dt)
        # 使用 pytest 检查调用 np.strings.rindex 函数时是否引发 ValueError 异常，并检查异常信息是否包含 "substring not found"
        with pytest.raises(ValueError, match="substring not found"):
            np.strings.rindex(buf, sub, start, end)

    @pytest.mark.parametrize("buf,tabsize,res", [
        # 参数化测试用例，测试字符串展开制表符，制表符大小为 8，预期展开结果
        ("abc\rab\tdef\ng\thi", 8, "abc\rab      def\ng       hi"),
        # 参数化测试用例，测试字符串展开制表符，制表符大小为 4，预期展开结果
    # 测试函数，验证在使用超出字符串长度时，expandtabs 函数是否引发 OverflowError 异常
    def test_expandtabs_raises_overflow(self, dt):
        # 使用 pytest.raises 检查是否抛出 OverflowError 异常，并匹配特定的错误信息
        with pytest.raises(OverflowError, match="new string is too long"):
            # 调用 numpy 的 expandtabs 函数，传入包含制表符的字符串数组和系统最大大小
            np.strings.expandtabs(np.array("\ta\n\tb", dtype=dt), sys.maxsize)
            # 再次调用 expandtabs 函数，传入包含制表符的字符串数组和一个超过系统极限的值
            np.strings.expandtabs(np.array("\ta\n\tb", dtype=dt), 2**61)

    # 类级别的错误信息常量定义，用于填充字符长度不符合要求时的异常情况
    FILL_ERROR = "The fill character must be exactly one character long"

    # 测试函数，验证在使用多字符填充字符时，center 函数是否引发 TypeError 异常
    def test_center_raises_multiple_character_fill(self, dt):
        # 创建包含字符串 "abc" 的 numpy 数组
        buf = np.array("abc", dtype=dt)
        # 创建包含多字符 "**" 的 numpy 数组，用于测试填充字符长度不符合要求的异常情况
        fill = np.array("**", dtype=dt)
        # 使用 pytest.raises 检查是否抛出 TypeError 异常，并匹配预期的错误信息
        with pytest.raises(TypeError, match=self.FILL_ERROR):
            # 调用 numpy 的 center 函数，传入数组 buf、填充宽度 10 和多字符 fill
            np.strings.center(buf, 10, fill)

    # 测试函数，验证在使用多字符填充字符时，ljust 函数是否引发 TypeError 异常
    def test_ljust_raises_multiple_character_fill(self, dt):
        buf = np.array("abc", dtype=dt)
        fill = np.array("**", dtype=dt)
        with pytest.raises(TypeError, match=self.FILL_ERROR):
            np.strings.ljust(buf, 10, fill)

    # 测试函数，验证在使用多字符填充字符时，rjust 函数是否引发 TypeError 异常
    def test_rjust_raises_multiple_character_fill(self, dt):
        buf = np.array("abc", dtype=dt)
        fill = np.array("**", dtype=dt)
        with pytest.raises(TypeError, match=self.FILL_ERROR):
            np.strings.rjust(buf, 10, fill)

    # 参数化测试函数，验证 center 函数的多个参数化输入是否按预期运行
    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('abc', 10, ' ', '   abc    '),
        ('abc', 6, ' ', ' abc  '),
        ('abc', 3, ' ', 'abc'),
        ('abc', 2, ' ', 'abc'),
        ('abc', 10, '*', '***abc****'),
    ])
    def test_center(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        # 使用 assert_array_equal 检查 center 函数的结果是否与预期结果一致
        assert_array_equal(np.strings.center(buf, width, fillchar), res)

    # 参数化测试函数，验证 ljust 函数的多个参数化输入是否按预期运行
    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('abc', 10, ' ', 'abc       '),
        ('abc', 6, ' ', 'abc   '),
        ('abc', 3, ' ', 'abc'),
        ('abc', 2, ' ', 'abc'),
        ('abc', 10, '*', 'abc*******'),
    ])
    def test_ljust(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        # 使用 assert_array_equal 检查 ljust 函数的结果是否与预期结果一致
        assert_array_equal(np.strings.ljust(buf, width, fillchar), res)

    # 参数化测试函数，验证 rjust 函数的多个参数化输入是否按预期运行
    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('abc', 10, ' ', '       abc'),
        ('abc', 6, ' ', '   abc'),
        ('abc', 3, ' ', 'abc'),
        ('abc', 2, ' ', 'abc'),
        ('abc', 10, '*', '*******abc'),
    ])
    def test_rjust(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        # 使用 assert_array_equal 检查 rjust 函数的结果是否与预期结果一致
        assert_array_equal(np.strings.rjust(buf, width, fillchar), res)
    @pytest.mark.parametrize("buf,width,res", [
        ('123', 2, '123'),
        ('123', 3, '123'),
        ('0123', 4, '0123'),
        ('+123', 3, '+123'),
        ('+123', 4, '+123'),
        ('+123', 5, '+0123'),
        ('+0123', 5, '+0123'),
        ('-123', 3, '-123'),
        ('-123', 4, '-123'),
        ('-0123', 5, '-0123'),
        ('000', 3, '000'),
        ('34', 1, '34'),
        ('0034', 4, '0034'),
    ])
    # 参数化测试函数，测试 np.strings.zfill 方法的功能
    def test_zfill(self, buf, width, res, dt):
        # 将输入的 buf 和 res 转换为指定的数据类型 dt 的 NumPy 数组
        buf = np.array(buf, dtype=dt)
        res = np.array(res, dtype=dt)
        # 断言 np.strings.zfill 方法的输出与预期的 res 数组相等
        assert_array_equal(np.strings.zfill(buf, width), res)

    @pytest.mark.parametrize("buf,sep,res1,res2,res3", [
        ("this is the partition method", "ti", "this is the par",
            "ti", "tion method"),
        ("http://www.python.org", "://", "http", "://", "www.python.org"),
        ("http://www.python.org", "?", "http://www.python.org", "", ""),
        ("http://www.python.org", "http://://", "", "http://", "www.python.org"),
        ("http://www.python.org", "org", "http://www.python.", "org", ""),
        ("http://www.python.org", ["://", "?", "http://", "org"],
            ["http", "http://www.python.org", "", "http://www.python."],
            ["://", "", "http://", "org"],
            ["www.python.org", "", "www.python.org", ""]),
        ("mississippi", "ss", "mi", "ss", "issippi"),
        ("mississippi", "i", "m", "i", "ssissippi"),
        ("mississippi", "w", "mississippi", "", ""),
    ])
    # 参数化测试函数，测试 np.strings.partition 方法的功能
    def test_partition(self, buf, sep, res1, res2, res3, dt):
        # 将输入的 buf, sep 和 res1, res2, res3 转换为指定的数据类型 dt 的 NumPy 数组
        buf = np.array(buf, dtype=dt)
        sep = np.array(sep, dtype=dt)
        res1 = np.array(res1, dtype=dt)
        res2 = np.array(res2, dtype=dt)
        res3 = np.array(res3, dtype=dt)
        # 调用 np.strings.partition 方法，并分别断言其输出与预期的 res1, res2, res3 数组相等
        act1, act2, act3 = np.strings.partition(buf, sep)
        assert_array_equal(act1, res1)
        assert_array_equal(act2, res2)
        assert_array_equal(act3, res3)
        # 断言拼接后的字符串与原始 buf 数组相等
        assert_array_equal(act1 + act2 + act3, buf)

    @pytest.mark.parametrize("buf,sep,res1,res2,res3", [
        ("this is the partition method", "ti", "this is the parti",
            "ti", "on method"),
        ("http://www.python.org", "://", "http", "://", "www.python.org"),
        ("http://www.python.org", "?", "", "", "http://www.python.org"),
        ("http://www.python.org", "http://://", "", "http://", "www.python.org"),
        ("http://www.python.org", "org", "http://www.python.", "org", ""),
        ("http://www.python.org", ["://", "?", "http://", "org"],
            ["http", "", "", "http://www.python."],
            ["://", "", "http://", "org"],
            ["www.python.org", "http://www.python.org", "www.python.org", ""]),
        ("mississippi", "ss", "missi", "ss", "ippi"),
        ("mississippi", "i", "mississipp", "i", ""),
        ("mississippi", "w", "", "", "mississippi"),
    ])
    # 参数化测试函数，测试 np.strings.partition 方法的功能（不同的测试用例）
    # 定义一个测试方法，用于测试 np.strings.rpartition 函数的行为
    def test_rpartition(self, buf, sep, res1, res2, res3, dt):
        # 将输入的 buf 转换为指定数据类型的 NumPy 数组
        buf = np.array(buf, dtype=dt)
        # 将输入的 sep 转换为指定数据类型的 NumPy 数组
        sep = np.array(sep, dtype=dt)
        # 将输入的 res1 转换为指定数据类型的 NumPy 数组
        res1 = np.array(res1, dtype=dt)
        # 将输入的 res2 转换为指定数据类型的 NumPy 数组
        res2 = np.array(res2, dtype=dt)
        # 将输入的 res3 转换为指定数据类型的 NumPy 数组
        res3 = np.array(res3, dtype=dt)
        # 使用 np.strings.rpartition 函数对 buf 进行分割操作，返回分割后的结果 act1, act2, act3
        act1, act2, act3 = np.strings.rpartition(buf, sep)
        # 断言 act1 与预期的 res1 相等
        assert_array_equal(act1, res1)
        # 断言 act2 与预期的 res2 相等
        assert_array_equal(act2, res2)
        # 断言 act3 与预期的 res3 相等
        assert_array_equal(act3, res3)
        # 断言分割后的结果的合并与原始 buf 相等
        assert_array_equal(act1 + act2 + act3, buf)
@pytest.mark.parametrize("dt", ["U", "T"])
class TestMethodsWithUnicode:
    @pytest.mark.parametrize("in_,out", [
        ("", False),
        ("a", False),
        ("0", True),
        ("\u2460", False),  # CIRCLED DIGIT 1
        ("\xbc", False),  # VULGAR FRACTION ONE QUARTER
        ("\u0660", True),  # ARABIC_INDIC DIGIT ZERO
        ("012345", True),
        ("012345a", False),
        (["0", "a"], [True, False]),
    ])
    def test_isdecimal_unicode(self, in_, out, dt):
        buf = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isdecimal(buf), out)

    @pytest.mark.parametrize("in_,out", [
        ("", False),
        ("a", False),
        ("0", True),
        ("\u2460", True),  # CIRCLED DIGIT 1
        ("\xbc", True),  # VULGAR FRACTION ONE QUARTER
        ("\u0660", True),  # ARABIC_INDIC DIGIT ZERO
        ("012345", True),
        ("012345a", False),
        (["0", "a"], [True, False]),
    ])
    def test_isnumeric_unicode(self, in_, out, dt):
        buf = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isnumeric(buf), out)

    @pytest.mark.parametrize("buf,old,new,count,res", [
        ("...\u043c......<", "<", "&lt;", -1, "...\u043c......&lt;"),
        ("Ae¢☃€ 😊" * 2, "A", "B", -1, "Be¢☃€ 😊Be¢☃€ 😊"),
        ("Ae¢☃€ 😊" * 2, "😊", "B", -1, "Ae¢☃€ BAe¢☃€ B"),
    ])
    def test_replace_unicode(self, buf, old, new, count, res, dt):
        buf = np.array(buf, dtype=dt)
        old = np.array(old, dtype=dt)
        new = np.array(new, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.replace(buf, old, new, count), res)

    @pytest.mark.parametrize("in_", [
        '\U00010401',  # Supplementary Private Use Area-A character
        '\U00010427',  # Supplementary Private Use Area-A character
        '\U00010429',  # Supplementary Private Use Area-A character
        '\U0001044E',  # Supplementary Private Use Area-A character
        '\U0001D7F6',  # Mathematical Double-struck Digit Zero
        '\U00011066',  # Brahmi Digit Zero
        '\U000104A0',  # Supplementary Private Use Area-A character
        pytest.param('\U0001F107', marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISALNUM",
            strict=True)),
    ])
    def test_isalnum_unicode(self, in_, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isalnum(in_), True)

    @pytest.mark.parametrize("in_,out", [
        ('\u1FFc', False),  # Greek Small Letter Omega with Perispomeni and Ypogegrammeni
        ('\u2167', False),  # Roman Numeral Eight
        ('\U00010401', False),  # Supplementary Private Use Area-A character
        ('\U00010427', False),  # Supplementary Private Use Area-A character
        ('\U0001F40D', False),  # Snake Emoji
        ('\U0001F46F', False),  # Police Officer Emoji
        ('\u2177', True),  # Small Roman Numeral Eight
        pytest.param('\U00010429', True, marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISLOWER",
            strict=True)),
        ('\U0001044E', True),  # Supplementary Private Use Area-A character
    ])
    def test_islower_unicode(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.islower(in_), out)
    @pytest.mark.parametrize("in_,out", [
        ('\u1FFc', False),  # Unicode测试输入，预期输出为False
        ('\u2167', True),   # Unicode测试输入，预期输出为True
        ('\U00010401', True),  # Unicode测试输入，预期输出为True
        ('\U00010427', True),  # Unicode测试输入，预期输出为True
        ('\U0001F40D', False),  # Unicode测试输入，预期输出为False
        ('\U0001F46F', False),  # Unicode测试输入，预期输出为False
        ('\u2177', False),   # Unicode测试输入，预期输出为False
        pytest.param('\U00010429', False, marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISUPPER",
            strict=True)),  # Unicode测试输入，条件下预期输出为False，带有xfail标记
        ('\U0001044E', False),  # Unicode测试输入，预期输出为False
    ])
    def test_isupper_unicode(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.isupper(in_), out)  # 断言测试是否符合预期输出

    @pytest.mark.parametrize("in_,out", [
        ('\u1FFc', True),   # Unicode测试输入，预期输出为True
        ('Greek \u1FFcitlecases ...', True),  # Unicode测试输入，预期输出为True
        pytest.param('\U00010401\U00010429', True, marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISISTITLE",
            strict=True)),  # Unicode测试输入，条件下预期输出为True，带有xfail标记
        ('\U00010427\U0001044E', True),  # Unicode测试输入，预期输出为True
        pytest.param('\U00010429', False, marks=pytest.mark.xfail(
            sys.platform == 'win32' and IS_PYPY_LT_7_3_16,
            reason="PYPY bug in Py_UNICODE_ISISTITLE",
            strict=True)),  # Unicode测试输入，条件下预期输出为False，带有xfail标记
        ('\U0001044E', False),  # Unicode测试输入，预期输出为False
        ('\U0001F40D', False),  # Unicode测试输入，预期输出为False
        ('\U0001F46F', False),  # Unicode测试输入，预期输出为False
    ])
    def test_istitle_unicode(self, in_, out, dt):
        in_ = np.array(in_, dtype=dt)
        assert_array_equal(np.strings.istitle(in_), out)  # 断言测试是否符合预期输出

    @pytest.mark.parametrize("buf,sub,start,end,res", [
        ("Ae¢☃€ 😊" * 2, "😊", 0, None, 6),  # 测试字符串操作，预期输出为6
        ("Ae¢☃€ 😊" * 2, "😊", 7, None, 13),  # 测试字符串操作，预期输出为13
    ])
    def test_index_unicode(self, buf, sub, start, end, res, dt):
        buf = np.array(buf, dtype=dt)
        sub = np.array(sub, dtype=dt)
        assert_array_equal(np.strings.index(buf, sub, start, end), res)  # 断言测试是否符合预期输出

    def test_index_raises_unicode(self, dt):
        with pytest.raises(ValueError, match="substring not found"):  # 测试异常情况，预期抛出特定异常
            np.strings.index("Ae¢☃€ 😊", "😀")

    @pytest.mark.parametrize("buf,res", [
        ("Ae¢☃€ \t 😊", "Ae¢☃€    😊"),  # 测试字符串操作，预期输出为指定格式字符串
        ("\t\U0001044E", "        \U0001044E"),  # 测试字符串操作，预期输出为指定格式字符串
    ])
    def test_expandtabs(self, buf, res, dt):
        buf = np.array(buf, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.expandtabs(buf), res)  # 断言测试是否符合预期输出

    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('x', 2, '\U0001044E', 'x\U0001044E'),   # 测试字符串操作，预期输出为指定格式字符串
        ('x', 3, '\U0001044E', '\U0001044Ex\U0001044E'),  # 测试字符串操作，预期输出为指定格式字符串
        ('x', 4, '\U0001044E', '\U0001044Ex\U0001044E\U0001044E'),  # 测试字符串操作，预期输出为指定格式字符串
    ])
    def test_center(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)
        fillchar = np.array(fillchar, dtype=dt)
        res = np.array(res, dtype=dt)
        assert_array_equal(np.strings.center(buf, width, fillchar), res)  # 断言测试是否符合预期输出
    # 使用 pytest 的 mark.parametrize 装饰器定义参数化测试，每个测试用例会依次使用不同的参数进行测试
    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('x', 2, '\U0001044E', 'x\U0001044E'),  # 测试用例1：buf='x', width=2, fillchar='\U0001044E', 预期结果为'x\U0001044E'
        ('x', 3, '\U0001044E', 'x\U0001044E\U0001044E'),  # 测试用例2：buf='x', width=3, fillchar='\U0001044E', 预期结果为'x\U0001044E\U0001044E'
        ('x', 4, '\U0001044E', 'x\U0001044E\U0001044E\U0001044E'),  # 测试用例3：buf='x', width=4, fillchar='\U0001044E', 预期结果为'x\U0001044E\U0001044E\U0001044E'
    ])
    def test_ljust(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)  # 将buf转换为numpy数组，使用指定的数据类型dt
        fillchar = np.array(fillchar, dtype=dt)  # 将fillchar转换为numpy数组，使用指定的数据类型dt
        res = np.array(res, dtype=dt)  # 将res转换为numpy数组，使用指定的数据类型dt
        assert_array_equal(np.strings.ljust(buf, width, fillchar), res)  # 断言调用 np.strings.ljust 方法的结果与预期的res数组相等
    
    @pytest.mark.parametrize("buf,width,fillchar,res", [
        ('x', 2, '\U0001044E', '\U0001044Ex'),  # 测试用例1：buf='x', width=2, fillchar='\U0001044E', 预期结果为'\U0001044Ex'
        ('x', 3, '\U0001044E', '\U0001044E\U0001044Ex'),  # 测试用例2：buf='x', width=3, fillchar='\U0001044E', 预期结果为'\U0001044E\U0001044Ex'
        ('x', 4, '\U0001044E', '\U0001044E\U0001044E\U0001044Ex'),  # 测试用例3：buf='x', width=4, fillchar='\U0001044E', 预期结果为'\U0001044E\U0001044E\U0001044Ex'
    ])
    def test_rjust(self, buf, width, fillchar, res, dt):
        buf = np.array(buf, dtype=dt)  # 将buf转换为numpy数组，使用指定的数据类型dt
        fillchar = np.array(fillchar, dtype=dt)  # 将fillchar转换为numpy数组，使用指定的数据类型dt
        res = np.array(res, dtype=dt)  # 将res转换为numpy数组，使用指定的数据类型dt
        assert_array_equal(np.strings.rjust(buf, width, fillchar), res)  # 断言调用 np.strings.rjust 方法的结果与预期的res数组相等
    
    @pytest.mark.parametrize("buf,sep,res1,res2,res3", [
        ("āāāāĀĀĀĀ", "Ă", "āāāāĀĀĀĀ", "", ""),  # 测试用例1：buf="āāāāĀĀĀĀ", sep="Ă", 预期结果为("āāāāĀĀĀĀ", "", "")
        ("āāāāĂĀĀĀĀ", "Ă", "āāāā", "Ă", "ĀĀĀĀ"),  # 测试用例2：buf="āāāāĂĀĀĀĀ", sep="Ă", 预期结果为("āāāā", "Ă", "ĀĀĀĀ")
        ("āāāāĂĂĀĀĀĀ", "ĂĂ", "āāāā", "ĂĂ", "ĀĀĀĀ"),  # 测试用例3：buf="āāāāĂĂĀĀĀĀ", sep="ĂĂ", 预期结果为("āāāā", "ĂĂ", "ĀĀĀĀ")
        ("𐌁𐌁𐌁𐌁𐌀𐌀𐌀𐌀", "𐌂", "𐌁𐌁𐌁𐌁𐌀𐌀𐌀𐌀", "", ""),  # 测试用例4：buf="𐌁𐌁𐌁𐌁𐌀𐌀𐌀𐌀", sep="𐌂", 预期结果为("𐌁𐌁𐌁𐌁𐌀𐌀𐌀𐌀", "", "")
        ("𐌁𐌁𐌁𐌁𐌂𐌀𐌀𐌀𐌀", "𐌂", "𐌁𐌁𐌁𐌁", "𐌂", "𐌀𐌀𐌀𐌀"),  # 测试用例5：buf="𐌁𐌁𐌁𐌁𐌂𐌀𐌀𐌀𐌀", sep="𐌂", 预期结果为("𐌁𐌁𐌁𐌁", "𐌂", "𐌀𐌀𐌀𐌀")
        ("𐌁𐌁𐌁𐌁𐌂𐌂𐌀𐌀𐌀𐌀", "𐌂𐌂", "𐌁𐌁𐌁𐌁", "𐌂𐌂", "𐌀𐌀𐌀𐌀"),  # 测试用例6：buf="𐌁𐌁𐌁𐌁𐌂𐌂𐌀𐌀𐌀𐌀", sep="𐌂𐌂", 预期结果为("𐌁𐌁𐌁𐌁", "𐌂𐌂", "𐌀𐌀𐌀𐌀")
        ("𐌁𐌁𐌁𐌁𐌂𐌂𐌂𐌂𐌀𐌀𐌀𐌀", "𐌂𐌂𐌂𐌂", "𐌁𐌁𐌁𐌁", "𐌂𐌂𐌂𐌂", "𐌀𐌀𐌀𐌀"),  # 测试用例7：buf="𐌁𐌁𐌁𐌁𐌂𐌂𐌂𐌂𐌀𐌀𐌀𐌀", sep="𐌂𐌂𐌂𐌂", 预期结果为("𐌁𐌁𐌁𐌁", "𐌂𐌂𐌂𐌂", "𐌀𐌀𐌀𐌀")
    ])
    def test_partition(self, buf, sep, res1, res2, res3, dt):
        buf = np.array(buf, dtype=dt)  # 将buf转换为numpy数组，使用指定的数据类型dt
        sep = np.array(sep, dtype=dt)  # 将sep转换为numpy数组，使用指定的数据类型dt
        res1 = np.array(res1, dtype=dt)  # 将res1转换为numpy数组，使用指定的数据类型dt
        res2 = np.array(res2, dtype=dt)  # 将res2转换为numpy数组，使用指定的数据类型dt
        res3 = np.array(res3, dtype=dt)  # 将res3转换为numpy数组
class TestMixedTypeMethods:
    # 定义测试类 TestMixedTypeMethods，用于测试混合类型方法

    def test_center(self):
        # 测试字符串居中方法
        buf = np.array("😊", dtype="U")
        # 创建包含单个 Unicode 表情的 NumPy 字符串数组 buf
        fill = np.array("*", dtype="S")
        # 创建包含单个 ASCII 字符的 NumPy 字符串数组 fill
        res = np.array("*😊*", dtype="U")
        # 创建期望结果的 NumPy Unicode 字符串数组 res
        assert_array_equal(np.strings.center(buf, 3, fill), res)
        # 使用 np.strings.center 方法将 buf 居中填充为长度为 3 的字符串，检查结果是否与 res 相等

        buf = np.array("s", dtype="S")
        # 创建包含单个 ASCII 字符的 NumPy 字符串数组 buf
        fill = np.array("*", dtype="U")
        # 创建包含单个 Unicode 字符的 NumPy 字符串数组 fill
        res = np.array("*s*", dtype="S")
        # 创建期望结果的 NumPy ASCII 字符串数组 res
        assert_array_equal(np.strings.center(buf, 3, fill), res)
        # 使用 np.strings.center 方法将 buf 居中填充为长度为 3 的字符串，检查结果是否与 res 相等

        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            # 使用 pytest 检查是否抛出 ValueError 异常，异常消息匹配给定字符串
            buf = np.array("s", dtype="S")
            # 创建包含单个 ASCII 字符的 NumPy 字符串数组 buf
            fill = np.array("😊", dtype="U")
            # 创建包含单个 Unicode 表情的 NumPy 字符串数组 fill
            np.strings.center(buf, 3, fill)
            # 尝试使用 np.strings.center 方法将 buf 居中填充为长度为 3 的字符串，预期抛出异常

    def test_ljust(self):
        # 测试字符串左对齐方法
        buf = np.array("😊", dtype="U")
        # 创建包含单个 Unicode 表情的 NumPy 字符串数组 buf
        fill = np.array("*", dtype="S")
        # 创建包含单个 ASCII 字符的 NumPy 字符串数组 fill
        res = np.array("😊**", dtype="U")
        # 创建期望结果的 NumPy Unicode 字符串数组 res
        assert_array_equal(np.strings.ljust(buf, 3, fill), res)
        # 使用 np.strings.ljust 方法将 buf 左对齐填充为长度为 3 的字符串，检查结果是否与 res 相等

        buf = np.array("s", dtype="S")
        # 创建包含单个 ASCII 字符的 NumPy 字符串数组 buf
        fill = np.array("*", dtype="U")
        # 创建包含单个 Unicode 字符的 NumPy 字符串数组 fill
        res = np.array("s**", dtype="S")
        # 创建期望结果的 NumPy ASCII 字符串数组 res
        assert_array_equal(np.strings.ljust(buf, 3, fill), res)
        # 使用 np.strings.ljust 方法将 buf 左对齐填充为长度为 3 的字符串，检查结果是否与 res 相等

        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            # 使用 pytest 检查是否抛出 ValueError 异常，异常消息匹配给定字符串
            buf = np.array("s", dtype="S")
            # 创建包含单个 ASCII 字符的 NumPy 字符串数组 buf
            fill = np.array("😊", dtype="U")
            # 创建包含单个 Unicode 表情的 NumPy 字符串数组 fill
            np.strings.ljust(buf, 3, fill)
            # 尝试使用 np.strings.ljust 方法将 buf 左对齐填充为长度为 3 的字符串，预期抛出异常

    def test_rjust(self):
        # 测试字符串右对齐方法
        buf = np.array("😊", dtype="U")
        # 创建包含单个 Unicode 表情的 NumPy 字符串数组 buf
        fill = np.array("*", dtype="S")
        # 创建包含单个 ASCII 字符的 NumPy 字符串数组 fill
        res = np.array("**😊", dtype="U")
        # 创建期望结果的 NumPy Unicode 字符串数组 res
        assert_array_equal(np.strings.rjust(buf, 3, fill), res)
        # 使用 np.strings.rjust 方法将 buf 右对齐填充为长度为 3 的字符串，检查结果是否与 res 相等

        buf = np.array("s", dtype="S")
        # 创建包含单个 ASCII 字符的 NumPy 字符串数组 buf
        fill = np.array("*", dtype="U")
        # 创建包含单个 Unicode 字符的 NumPy 字符串数组 fill
        res = np.array("**s", dtype="S")
        # 创建期望结果的 NumPy ASCII 字符串数组 res
        assert_array_equal(np.strings.rjust(buf, 3, fill), res)
        # 使用 np.strings.rjust 方法将 buf 右对齐填充为长度为 3 的字符串，检查结果是否与 res 相等

        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            # 使用 pytest 检查是否抛出 ValueError 异常，异常消息匹配给定字符串
            buf = np.array("s", dtype="S")
            # 创建包含单个 ASCII 字符的 NumPy 字符串数组 buf
            fill = np.array("😊", dtype="U")
            # 创建包含单个 Unicode 表情的 NumPy 字符串数组 fill
            np.strings.rjust(buf, 3, fill)
            # 尝试使用 np.strings.rjust 方法将 buf 右对齐填充为长度为 3 的字符串，预期抛出异常


class TestUnicodeOnlyMethodsRaiseWithBytes:
    # 定义测试类 TestUnicodeOnlyMethodsRaiseWithBytes，用于测试只支持 Unicode 方法并且能处理字节异常

    def test_isdecimal_raises(self):
        # 测试 isdecimal 方法在处理字节时是否能正确抛出异常
        in_ = np.array(b"1")
        # 创建包含字节序列 b"1" 的 NumPy 数组 in_
        with assert_raises(TypeError):
            # 使用 assert_raises 检查是否抛出 TypeError 异常
            np.strings.isdecimal(in_)
            # 调用 np.strings.isdecimal 方法尝试处理字节输入

    def test_isnumeric_bytes(self):
        # 测试 isnumeric 方法在处理字节时是否能正确抛出异常
        in_ = np.array(b"1")
        # 创建包含字节序列 b"1" 的 NumPy 数组 in_
        with assert_raises(TypeError):
            # 使用 assert_raises 检查是否抛出 TypeError 异常
            np.strings.isnumeric(in_)
            # 调用 np.strings.isnumeric 方法尝试处理字节输入


def check_itemsize(n_elem, dt):
    # 定义函数 check_itemsize，用于检查数据类型的字节大小
    if dt == "T":
        return np.dtype(dt).itemsize
        # 如果数据类型为 "T"，返回其字节大小
    if dt == "S":
        return n_elem
        # 如果数据类型为 "S"，返回元素个数
    if dt == "U":
        return n_elem * 4
        # 如果数据类型为 "U"，返回元素个数乘以 4（Unicode 字符的字节大小）

@pytest.mark.parametrize("dt", ["S", "U", "T"])
# 使用 pytest.mark.parametrize 对参数 dt 进行多组测试参数化
class TestReplaceOnArrays:
    # 定义测试类 TestReplaceOnArrays，用于测试在数组上执行替换操作
    # 定义一个测试函数来检验replace方法对count和size的影响
    def test_replace_count_and_size(self, dt):
        # 创建一个包含不同长度字符串的数组
        a = np.array(["0123456789" * i for i in range(4)], dtype=dt)
        # 对数组中的所有元素进行替换
        r1 = np.strings.replace(a, "5", "ABCDE")
        # 检查替换后的数据类型大小是否符合预期
        assert r1.dtype.itemsize == check_itemsize(3*10 + 3*4, dt)
        # 创建预期的替换结果数组
        r1_res = np.array(["01234ABCDE6789" * i for i in range(4)], dtype=dt)
        # 检查替换后的数组是否符合预期
        assert_array_equal(r1, r1_res)
        # 在指定数量的字符中进行替换
        r2 = np.strings.replace(a, "5", "ABCDE", 1)
        # 检查替换后的数据类型大小是否符合预期
        assert r2.dtype.itemsize == check_itemsize(3*10 + 4, dt)
        # 在全部字符中进行替换
        r3 = np.strings.replace(a, "5", "ABCDE", 0)
        # 检查替换后的数据类型大小是否与原数组一致
        assert r3.dtype.itemsize == a.dtype.itemsize
        # 检查替换后的数组是否与原数组一致
        assert_array_equal(r3, a)
        # 使用负值表示替换所有匹配的字符
        r4 = np.strings.replace(a, "5", "ABCDE", -1)
        # 检查替换后的数据类型大小是否符合预期
        assert r4.dtype.itemsize == check_itemsize(3*10 + 3*4, dt)
        # 检查替换后的数组是否与r1相同
        assert_array_equal(r4, r1)
        # 可以对每个元素的替换数量进行逐个处理
        r5 = np.strings.replace(a, "5", "ABCDE", [-1, -1, -1, 1])
        # 检查替换后的数据类型大小是否符合预期
        assert r5.dtype.itemsize == check_itemsize(3*10 + 4, dt)
        # 检查替换后的数组是否符合预期
        assert_array_equal(r5, np.array(
            ["01234ABCDE6789" * i for i in range(3)]
            + ["01234ABCDE6789" + "0123456789" * 2], dtype=dt))

    # 测试replace方法的广播功能
    def test_replace_broadcasting(self, dt):
        # 创建一个包含字符串"0,0,0"的数组
        a = np.array("0,0,0", dtype=dt)
        # 对数组中的每个元素进行替换
        r1 = np.strings.replace(a, "0", "1", np.arange(3))
        # 检查替换后的数据类型是否与原数组一致
        assert r1.dtype == a.dtype
        # 检查替换后的数组是否符合预期
        assert_array_equal(r1, np.array(["0,0,0", "1,0,0", "1,1,0"], dtype=dt))
        # 对数组中的每个元素进行替换，同时进行广播操作
        r2 = np.strings.replace(a, "0", [["1"], ["2"]], np.arange(1, 4))
        # 检查替换后的数组是否符合预期
        assert_array_equal(r2, np.array([["1,0,0", "1,1,0", "1,1,1"],
                                         ["2,0,0", "2,2,0", "2,2,2"]],
                                        dtype=dt))
        # 对数组中的每个元素进行替换，用不同的替换字符串
        r3 = np.strings.replace(a, ["0", "0,0", "0,0,0"], "X")
        # 检查替换后的数组是否符合预期
        assert_array_equal(r3, np.array(["X,X,X", "X,0", "X"], dtype=dt))
```