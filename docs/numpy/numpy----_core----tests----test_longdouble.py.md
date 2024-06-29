# `.\numpy\numpy\_core\tests\test_longdouble.py`

```
# 引入警告模块，用于管理警告信息
import warnings
# 引入平台模块，用于获取平台相关信息
import platform
# 引入 pytest 测试框架
import pytest

# 引入 numpy 库，并导入一些特定的测试函数和变量
import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_warns, assert_array_equal,
    temppath, IS_MUSL
    )
# 从 numpy._core.tests._locales 模块导入 CommaDecimalPointLocale 类
from numpy._core.tests._locales import CommaDecimalPointLocale

# 获取长双精度浮点数的信息
LD_INFO = np.finfo(np.longdouble)
# 检查长双精度浮点数是否比双精度浮点数更精确
longdouble_longer_than_double = (LD_INFO.eps < np.finfo(np.double).eps)

# 定义一个临时变量 _o，用于检查字符串到长双精度浮点数的转换是否准确
_o = 1 + LD_INFO.eps
string_to_longdouble_inaccurate = (_o != np.longdouble(str(_o)))
del _o

def test_scalar_extraction():
    """确认提取值时不会转换为 Python 的浮点数"""
    o = 1 + LD_INFO.eps
    a = np.array([o, o, o])
    assert_equal(a[1], o)

# 字符串到长双精度浮点数的转换

# 计算 0.1 在二进制浮点表示中的精度
repr_precision = len(repr(np.longdouble(0.1)))
# 宏块从 scalartypes.c.src 文件大约第 842 行开始，额外增加 2 个精度。

# 根据条件跳过测试，当 IS_MUSL 为真时，测试在 musllinux 上不稳定
@pytest.mark.skipif(IS_MUSL,
                    reason="test flaky on musllinux")
# 根据条件跳过测试，当 repr 的精度不足以显示 eps 时
@pytest.mark.skipif(LD_INFO.precision + 2 >= repr_precision,
                    reason="repr precision not enough to show eps")
def test_str_roundtrip():
    # 只有在 repr 的精度能够显示 eps 时，我们才能在 repr 中看到 eps
    o = 1 + LD_INFO.eps
    assert_equal(np.longdouble(str(o)), o, "str was %s" % str(o))

# 根据条件跳过测试，当字符串到长双精度浮点数的转换不准确时
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
def test_str_roundtrip_bytes():
    o = 1 + LD_INFO.eps
    assert_equal(np.longdouble(str(o).encode("ascii")), o)

# 根据条件跳过测试，当字符串到长双精度浮点数的转换不准确时
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
# 参数化测试，测试不同类型的字符串（np.str_, np.bytes_, str, bytes）
@pytest.mark.parametrize("strtype", (np.str_, np.bytes_, str, bytes))
def test_array_and_stringlike_roundtrip(strtype):
    """
    测试长双精度浮点数的字符串表示在数组转换和标量强制转换中的往返，
    参见 gh-15608。
    """
    o = 1 + LD_INFO.eps

    if strtype in (np.bytes_, bytes):
        o_str = strtype(str(o).encode("ascii"))
    else:
        o_str = strtype(str(o))

    # 测试 `o` 是否能正确从字符串类转换
    assert o == np.longdouble(o_str)

    # 测试数组的往返转换是否正确：
    o_strarr = np.asarray([o] * 3, dtype=strtype)
    assert (o == o_strarr.astype(np.longdouble)).all()

    # 数组强制转换和字符串的强制转换应该与标量的 repr 相同：
    assert (o_strarr == o_str).all()
    assert (np.asarray([o] * 3).astype(strtype) == o_str).all()

def test_bogus_string():
    # 测试不合法的字符串转换是否会引发 ValueError 异常
    assert_raises(ValueError, np.longdouble, "spam")
    assert_raises(ValueError, np.longdouble, "1.0 flub")

# 根据条件跳过测试，当字符串到长双精度浮点数的转换不准确时
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
def test_fromstring():
    o = 1 + LD_INFO.eps
    s = (" " + str(o))*5
    a = np.array([o]*5)
    assert_equal(np.fromstring(s, sep=" ", dtype=np.longdouble), a,
                 err_msg="reading '%s'" % s)

def test_fromstring_complex():
    # 测试复杂字符串到长双精度浮点数的转换
    # 遍历指定的数据类型列表，依次进行以下操作
    for ctype in ["complex", "cdouble"]:
        # 使用 np.fromstring 函数将给定字符串解析为指定数据类型的 NumPy 数组，
        # 使用逗号作为分隔符，进行解析
        assert_equal(np.fromstring("1, 2 ,  3  ,4", sep=",", dtype=ctype),
                     np.array([1., 2., 3., 4.]))
        
        # 当实部未指定时，使用 np.fromstring 解析包含虚数部分的字符串，
        # 使用逗号作为分隔符，生成对应的复数数组
        assert_equal(np.fromstring("1j, -2j,  3j, 4e1j", sep=",", dtype=ctype),
                     np.array([1.j, -2.j, 3.j, 40.j]))
        
        # 当实部和虚部都指定时，使用 np.fromstring 解析包含实部和虚部的字符串，
        # 使用逗号作为分隔符，生成对应的复数数组
        assert_equal(np.fromstring("1+1j,2-2j, -3+3j,  -4e1+4j", sep=",", dtype=ctype),
                     np.array([1. + 1.j, 2. - 2.j, - 3. + 3.j, - 40. + 4j]))
        
        # 测试字符串中空格放置不当的情况，这些情况应该会引发 DeprecationWarning
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring("1+2 j,3", dtype=ctype, sep=","),
                         np.array([1.]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring("1+ 2j,3", dtype=ctype, sep=","),
                         np.array([1.]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring("1 +2j,3", dtype=ctype, sep=","),
                         np.array([1.]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring("1+j", dtype=ctype, sep=","),
                         np.array([1.]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring("1+", dtype=ctype, sep=","),
                         np.array([1.]))
        with assert_warns(DeprecationWarning):
            assert_equal(np.fromstring("1j+1", dtype=ctype, sep=","),
                         np.array([1j]))
# 定义一个测试函数，测试处理不正确字符串的情况
def test_fromstring_bogus():
    # 使用 assert_warns 检查是否会产生 DeprecationWarning 警告
    with assert_warns(DeprecationWarning):
        # 调用 np.fromstring 方法尝试解析字符串，期望返回一个浮点数数组
        assert_equal(np.fromstring("1. 2. 3. flop 4.", dtype=float, sep=" "),
                     np.array([1., 2., 3.]))

# 定义一个测试函数，测试处理空字符串的情况
def test_fromstring_empty():
    # 使用 assert_warns 检查是否会产生 DeprecationWarning 警告
    with assert_warns(DeprecationWarning):
        # 调用 np.fromstring 方法尝试解析字符串，期望返回一个空的数组
        assert_equal(np.fromstring("xxxxx", sep="x"),
                     np.array([]))

# 定义一个测试函数，测试处理字符串中有缺失数据的情况
def test_fromstring_missing():
    # 使用 assert_warns 检查是否会产生 DeprecationWarning 警告
    with assert_warns(DeprecationWarning):
        # 调用 np.fromstring 方法尝试解析字符串，期望返回只有一个元素的数组
        assert_equal(np.fromstring("1xx3x4x5x6", sep="x"),
                     np.array([1]))

# 定义一个文件基础测试类
class TestFileBased:
    # 计算 ldbl 变量的值，LD_INFO.eps 为一个预定义的值
    ldbl = 1 + LD_INFO.eps
    # 创建一个目标数组，包含 5 个 ldbl 元素
    tgt = np.array([ldbl]*5)
    # 将目标数组转换为一个包含每个元素加换行符的字符串
    out = ''.join([str(t) + '\n' for t in tgt])

    # 定义一个测试函数，测试从文件读取不正确字符串的情况
    def test_fromfile_bogus(self):
        # 使用 temppath() 上下文管理器创建临时文件路径
        with temppath() as path:
            # 将字符串写入临时文件
            with open(path, 'w') as f:
                f.write("1. 2. 3. flop 4.\n")

            # 使用 assert_warns 检查是否会产生 DeprecationWarning 警告
            # 调用 np.fromfile 方法尝试从文件中读取数据，期望返回一个浮点数数组
            with assert_warns(DeprecationWarning):
                res = np.fromfile(path, dtype=float, sep=" ")
        # 断言从文件中读取的数组与期望的数组相等
        assert_equal(res, np.array([1., 2., 3.]))
    def test_fromfile_complex(self):
        # 针对复数和复数双精度类型进行测试
        for ctype in ["complex", "cdouble"]:
            # 检查分隔符与仅指定实部之间的空格
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1, 2 ,  3  ,4\n")
                
                # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1., 2., 3., 4.]))

            # 未指定实部的情况
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1j, -2j,  3j, 4e1j\n")

                # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1.j, -2.j, 3.j, 40.j]))

            # 同时指定实部和虚部的情况
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+1j,2-2j, -3+3j,  -4e1+4j\n")

                # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1. + 1.j, 2. - 2.j, - 3. + 3.j, - 40. + 4j]))

            # 错误位置的空格
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+2 j,3\n")

                # 验证警告是否抛出 DeprecationWarning
                with assert_warns(DeprecationWarning):
                    # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                    res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1.]))

            # 错误位置的空格
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+ 2j,3\n")

                # 验证警告是否抛出 DeprecationWarning
                with assert_warns(DeprecationWarning):
                    # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                    res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1.]))

            # 错误位置的空格
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1 +2j,3\n")

                # 验证警告是否抛出 DeprecationWarning
                with assert_warns(DeprecationWarning):
                    # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                    res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1.]))

            # 错误位置的空格
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+j\n")

                # 验证警告是否抛出 DeprecationWarning
                with assert_warns(DeprecationWarning):
                    # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                    res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1.]))

            # 错误位置的空格
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1+\n")

                # 验证警告是否抛出 DeprecationWarning
                with assert_warns(DeprecationWarning):
                    # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                    res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1.]))

            # 错误位置的空格
            with temppath() as path:
                with open(path, 'w') as f:
                    f.write("1j+1\n")

                # 验证警告是否抛出 DeprecationWarning
                with assert_warns(DeprecationWarning):
                    # 从文件中读取数据并按指定的数据类型和分隔符创建 NumPy 数组
                    res = np.fromfile(path, dtype=ctype, sep=",")
            # 断言结果与期望的 NumPy 数组相等
            assert_equal(res, np.array([1.j]))
    # 在特定条件下跳过测试，条件是 string_to_longdouble_inaccurate 为真
    @pytest.mark.skipif(string_to_longdouble_inaccurate,
                        reason="Need strtold_l")
    # 定义一个测试函数 test_fromfile，测试从文件中读取数据并使用 np.fromfile 转换为长双精度数组
    def test_fromfile(self):
        # 使用 temppath() 创建临时文件路径，并在该路径下写入 self.out 的内容
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            # 使用 np.fromfile 从文件中读取数据，指定数据类型为 np.longdouble，数据分隔符为换行符
            res = np.fromfile(path, dtype=np.longdouble, sep="\n")
        # 断言测试结果与预期结果 self.tgt 相等
        assert_equal(res, self.tgt)
    
    # 在特定条件下跳过测试，条件是 string_to_longdouble_inaccurate 为真
    @pytest.mark.skipif(string_to_longdouble_inaccurate,
                        reason="Need strtold_l")
    # 定义一个测试函数 test_genfromtxt，测试从文件中读取数据并使用 np.genfromtxt 转换为长双精度数组
    def test_genfromtxt(self):
        # 使用 temppath() 创建临时文件路径，并在该路径下写入 self.out 的内容
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            # 使用 np.genfromtxt 从文件中读取数据，指定数据类型为 np.longdouble
            res = np.genfromtxt(path, dtype=np.longdouble)
        # 断言测试结果与预期结果 self.tgt 相等
        assert_equal(res, self.tgt)
    
    # 在特定条件下跳过测试，条件是 string_to_longdouble_inaccurate 为真
    @pytest.mark.skipif(string_to_longdouble_inaccurate,
                        reason="Need strtold_l")
    # 定义一个测试函数 test_loadtxt，测试从文件中读取数据并使用 np.loadtxt 转换为长双精度数组
    def test_loadtxt(self):
        # 使用 temppath() 创建临时文件路径，并在该路径下写入 self.out 的内容
        with temppath() as path:
            with open(path, 'w') as f:
                f.write(self.out)
            # 使用 np.loadtxt 从文件中读取数据，指定数据类型为 np.longdouble
            res = np.loadtxt(path, dtype=np.longdouble)
        # 断言测试结果与预期结果 self.tgt 相等
        assert_equal(res, self.tgt)
    
    # 在特定条件下跳过测试，条件是 string_to_longdouble_inaccurate 为真
    @pytest.mark.skipif(string_to_longdouble_inaccurate,
                        reason="Need strtold_l")
    # 定义一个测试函数 test_tofile_roundtrip，测试数据的写入和读取操作是否能够循环正确处理长双精度数据
    def test_tofile_roundtrip(self):
        # 使用 temppath() 创建临时文件路径，并将 self.tgt 的内容写入该路径下，数据分隔符为空格
        self.tgt.tofile(path, sep=" ")
        # 使用 np.fromfile 从文件中读取数据，指定数据类型为 np.longdouble，数据分隔符为空格
        res = np.fromfile(path, dtype=np.longdouble, sep=" ")
        # 断言测试结果与预期结果 self.tgt 相等
        assert_equal(res, self.tgt)
# Conversions long double -> string

# 定义测试函数，测试 long double 类型到字符串类型的转换
def test_str_exact():
    # 创建一个 long double 对象 o，其值为 LD_INFO.eps + 1
    o = 1 + LD_INFO.eps
    # 断言 o 转换为字符串后不等于 '1'
    assert_(str(o) != '1')


# 跳过条件性测试，如果 long double 类型的精度大于 double 类型
@pytest.mark.skipif(longdouble_longer_than_double, reason="BUG #2376")
# 跳过条件性测试，如果字符串到 long double 转换不精确，需要 strtold_l 函数
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
# 格式化测试函数
def test_format():
    # 创建一个 long double 对象 o，其值为 LD_INFO.eps + 1
    o = 1 + LD_INFO.eps
    # 断言格式化输出 "{0:.40g}" 的结果不等于 '1'
    assert_("{0:.40g}".format(o) != '1')


# 跳过条件性测试，如果 long double 类型的精度大于 double 类型
@pytest.mark.skipif(longdouble_longer_than_double, reason="BUG #2376")
# 跳过条件性测试，如果字符串到 long double 转换不精确，需要 strtold_l 函数
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
# 百分比格式化测试函数
def test_percent():
    # 创建一个 long double 对象 o，其值为 LD_INFO.eps + 1
    o = 1 + LD_INFO.eps
    # 断言百分比格式化输出 "%.40g" 的结果不等于 '1'
    assert_("%.40g" % o != '1')


# 跳过条件性测试，如果 long double 类型的精度大于 double 类型，由于数组表示问题
@pytest.mark.skipif(longdouble_longer_than_double, reason="array repr problem")
# 跳过条件性测试，如果字符串到 long double 转换不精确，需要 strtold_l 函数
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason="Need strtold_l")
# 数组表示测试函数
def test_array_repr():
    # 创建一个 long double 对象 o，其值为 LD_INFO.eps + 1
    o = 1 + LD_INFO.eps
    # 创建一个包含 o 的 numpy 数组 a
    a = np.array([o])
    # 创建一个包含 1 的 long double 类型的 numpy 数组 b
    b = np.array([1], dtype=np.longdouble)
    # 如果数组 a 与数组 b 的所有元素均相等，则抛出 ValueError 异常
    if not np.all(a != b):
        raise ValueError("precision loss creating arrays")
    # 断言数组 a 的字符串表示不等于数组 b 的字符串表示
    assert_(repr(a) != repr(b))


# 区域设置测试：标量类型的格式化输出应独立于区域设置
class TestCommaDecimalPointLocale(CommaDecimalPointLocale):

    # 测试用例：测试使用外国语言环境下的字符串转换为 long double 类型的精度
    def test_str_roundtrip_foreign(self):
        # 创建一个浮点数 o，其值为 1.5
        o = 1.5
        # 断言 o 转换为 long double 类型后与原始值相等
        assert_equal(o, np.longdouble(str(o)))

    # 测试用例：从字符串表示中创建 float 类型数组的准确性
    def test_fromstring_foreign_repr(self):
        # 浮点数 f 的值为 1.234
        f = 1.234
        # 从 f 的字符串表示中创建一个 float 类型的 numpy 数组 a
        a = np.fromstring(repr(f), dtype=float, sep=" ")
        # 断言数组 a 的第一个元素等于 f
        assert_equal(a[0], f)

    # 测试用例：从特定字符串中创建 float 类型数组，期望 DeprecationWarning 警告
    def test_fromstring_best_effort_float(self):
        # 期望 DeprecationWarning 警告
        with assert_warns(DeprecationWarning):
            # 从字符串 "1,234" 中创建一个 float 类型的数组，期望结果为 [1.]
            assert_equal(np.fromstring("1,234", dtype=float, sep=" "),
                         np.array([1.]))

    # 测试用例：从特定字符串中创建 long double 类型数组，期望 DeprecationWarning 警告
    def test_fromstring_best_effort(self):
        # 期望 DeprecationWarning 警告
        with assert_warns(DeprecationWarning):
            # 从字符串 "1,234" 中创建一个 long double 类型的数组，期望结果为 [1.]
            assert_equal(np.fromstring("1,234", dtype=np.longdouble, sep=" "),
                         np.array([1.]))

    # 测试用例：从特定字符串中创建 long double 类型数组
    def test_fromstring_foreign(self):
        # 字符串 s 的值为 "1.234"
        s = "1.234"
        # 从字符串 s 中创建一个 long double 类型的 numpy 数组 a
        a = np.fromstring(s, dtype=np.longdouble, sep=" ")
        # 断言数组 a 的第一个元素等于 long double 类型的 s
        assert_equal(a[0], np.longdouble(s))

    # 测试用例：从特定字符串中创建 long double 类型数组，指定分隔符为 ","
    def test_fromstring_foreign_sep(self):
        # 创建一个整数数组 a，值为 [1, 2, 3, 4]
        a = np.array([1, 2, 3, 4])
        # 从字符串 "1,2,3,4," 中创建一个 long double 类型的 numpy 数组 b，指定分隔符为 ","
        b = np.fromstring("1,2,3,4,", dtype=np.longdouble, sep=",")
        # 断言数组 a 与数组 b 相等
        assert_array_equal(a, b)

    # 测试用例：从特定字符串中创建 long double 类型数组，期望 DeprecationWarning 警告
    def test_fromstring_foreign_value(self):
        # 期望 DeprecationWarning 警告
        with assert_warns(DeprecationWarning):
            # 从字符串 "1,234" 中创建一个 long double 类型的数组 b，期望结果为 [1.]
            b = np.fromstring("1,234", dtype=np.longdouble, sep=" ")
            # 断言数组 b 的第一个元素等于 1
            assert_array_equal(b[0], 1)


# 使用参数化测试，测试 long double 类型从整数转换
@pytest.mark.parametrize("int_val", [
    # 讨论 gh-10723 和 gh-9968 中的案例
    2 ** 1024, 0])
def test_longdouble_from_int(int_val):
    # 对于问题 gh-9968
    # 将整数 int_val 转换为字符串 str_val
    str_val = str(int_val)
    # 对于大整数输入，在具有与 np.double 等效的 np.longdouble 平台上，我们期望 RuntimeWarning
    # 我们期望在具有 np.longdouble 等效于 np.double 的平台上看到 RuntimeWarning
    # 创建一个字符串表示时可能会有精度损失的情况
    # 使用 `warnings.catch_warnings` 上下文管理器捕获所有警告
    with warnings.catch_warnings(record=True) as w:
        # 设置警告过滤器，始终捕获 RuntimeWarning 类型的警告
        warnings.filterwarnings('always', '', RuntimeWarning)
        
        # 断言：某些平台上 np.longdouble(int_val) == np.longdouble(str_val) 可能为 True
        assert np.longdouble(int_val) == np.longdouble(str_val)
        
        # 检查是否可以在所有平台上直接比较 int 和最大 longdouble 值
        if np.allclose(np.finfo(np.longdouble).max,
                       np.finfo(np.double).max) and w:
            # 断言：如果条件成立且有警告产生，则确认第一个警告的类别为 RuntimeWarning
            assert w[0].category is RuntimeWarning
# 使用 pytest 的 @pytest.mark.parametrize 装饰器，用于多次运行 test_longdouble_from_bool 函数
@pytest.mark.parametrize("bool_val", [
    True, False])
def test_longdouble_from_bool(bool_val):
    # 断言对于 np.longdouble 类型，将布尔值转换为长双精度数应该等于将布尔值转换为整数后再转换为长双精度数
    assert np.longdouble(bool_val) == np.longdouble(int(bool_val))


# 使用 pytest 的 @pytest.mark.skipif 装饰器，条件不满足时跳过测试
@pytest.mark.skipif(
    not (IS_MUSL and platform.machine() == "x86_64"),
    reason="only need to run on musllinux_x86_64"
)
def test_musllinux_x86_64_signature():
    # 如果在非 musllinux_x86_64 架构上模拟该测试可能会失败，但在本地运行应该通过
    known_sigs = [b'\xcd\xcc\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf']
    sig = (np.longdouble(-1.0) / np.longdouble(10.0))
    # 将 sig 转换为小端字节序，并取其前 10 字节
    sig = sig.view(sig.dtype.newbyteorder('<')).tobytes()[:10]
    # 断言 sig 存在于已知签名列表中
    assert sig in known_sigs


# 检查 np.longdouble 类型的机器精度 eps 是否为正值
def test_eps_positive():
    # np.finfo(np.longdouble).eps 应在所有平台上为正值。如果不是，则可能 MachArLike 出了问题，
    # 比如 np._core.getlimits._discovered_machar 没有正常工作
    assert np.finfo(np.longdouble).eps > 0.
```