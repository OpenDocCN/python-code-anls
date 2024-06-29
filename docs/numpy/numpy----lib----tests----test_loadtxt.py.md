# `.\numpy\numpy\lib\tests\test_loadtxt.py`

```
"""
`np.loadtxt`的特定测试，用于在将loadtxt移至C代码后进行的补充测试。
这些测试是`test_io.py`中已有测试的补充。
"""

import sys  # 导入sys模块，用于系统相关操作
import os   # 导入os模块，用于操作系统相关功能
import pytest   # 导入pytest测试框架
from tempfile import NamedTemporaryFile, mkstemp   # 导入临时文件相关函数
from io import StringIO   # 导入StringIO用于内存中文件操作

import numpy as np   # 导入NumPy库
from numpy.ma.testutils import assert_equal   # 导入NumPy的测试工具函数
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY   # 导入NumPy的测试工具函数和相关常量


def test_scientific_notation():
    """测试科学计数法中 'e' 和 'E' 的解析是否正确。"""
    data = StringIO(
        (
            "1.0e-1,2.0E1,3.0\n"
            "4.0e-2,5.0E-1,6.0\n"
            "7.0e-3,8.0E1,9.0\n"
            "0.0e-4,1.0E-1,2.0"
        )
    )
    expected = np.array(
        [[0.1, 20., 3.0], [0.04, 0.5, 6], [0.007, 80., 9], [0, 0.1, 2]]
    )
    assert_array_equal(np.loadtxt(data, delimiter=","), expected)


@pytest.mark.parametrize("comment", ["..", "//", "@-", "this is a comment:"])
def test_comment_multiple_chars(comment):
    """测试多字符注释在加载数据时的处理。"""
    content = "# IGNORE\n1.5, 2.5# ABC\n3.0,4.0# XXX\n5.5,6.0\n"
    txt = StringIO(content.replace("#", comment))
    a = np.loadtxt(txt, delimiter=",", comments=comment)
    assert_equal(a, [[1.5, 2.5], [3.0, 4.0], [5.5, 6.0]])


@pytest.fixture
def mixed_types_structured():
    """
    提供具有结构化dtype的异构输入数据和相关结构化数组的fixture。
    """
    data = StringIO(
        (
            "1000;2.4;alpha;-34\n"
            "2000;3.1;beta;29\n"
            "3500;9.9;gamma;120\n"
            "4090;8.1;delta;0\n"
            "5001;4.4;epsilon;-99\n"
            "6543;7.8;omega;-1\n"
        )
    )
    dtype = np.dtype(
        [('f0', np.uint16), ('f1', np.float64), ('f2', 'S7'), ('f3', np.int8)]
    )
    expected = np.array(
        [
            (1000, 2.4, "alpha", -34),
            (2000, 3.1, "beta", 29),
            (3500, 9.9, "gamma", 120),
            (4090, 8.1, "delta", 0),
            (5001, 4.4, "epsilon", -99),
            (6543, 7.8, "omega", -1)
        ],
        dtype=dtype
    )
    return data, dtype, expected


@pytest.mark.parametrize('skiprows', [0, 1, 2, 3])
def test_structured_dtype_and_skiprows_no_empty_lines(
        skiprows, mixed_types_structured):
    """测试结构化dtype和跳过行数（无空行）的情况。"""
    data, dtype, expected = mixed_types_structured
    a = np.loadtxt(data, dtype=dtype, delimiter=";", skiprows=skiprows)
    assert_array_equal(a, expected[skiprows:])


def test_unpack_structured(mixed_types_structured):
    """测试结构化dtype在解包时的处理。"""
    data, dtype, expected = mixed_types_structured

    a, b, c, d = np.loadtxt(data, dtype=dtype, delimiter=";", unpack=True)
    assert_array_equal(a, expected["f0"])
    assert_array_equal(b, expected["f1"])
    assert_array_equal(c, expected["f2"])
    assert_array_equal(d, expected["f3"])


def test_structured_dtype_with_shape():
    """测试带形状的结构化dtype的情况。"""
    dtype = np.dtype([("a", "u1", 2), ("b", "u1", 2)])
    data = StringIO("0,1,2,3\n6,7,8,9\n")
    expected = np.array([((0, 1), (2, 3)), ((6, 7), (8, 9))], dtype=dtype)
    # 使用 numpy 库中的 loadtxt 函数读取数据文件，并使用指定的逗号分隔符和数据类型进行加载
    assert_array_equal(np.loadtxt(data, delimiter=",", dtype=dtype), expected)
def test_structured_dtype_with_multi_shape():
    # 定义一个结构化的 NumPy 数据类型，包含字段 'a'，每个元素是一个 2x2 的无符号整数数组
    dtype = np.dtype([("a", "u1", (2, 2))])
    # 创建一个包含数据的字符串流对象
    data = StringIO("0 1 2 3\n")
    # 期望的 NumPy 数组，包含一个元素，该元素是一个 2x2 的数组，元素值为 (0, 1, 2, 3)
    expected = np.array([(((0, 1), (2, 3)),)], dtype=dtype)
    # 断言使用 np.loadtxt 函数加载数据，并与期望的数组进行比较
    assert_array_equal(np.loadtxt(data, dtype=dtype), expected)


def test_nested_structured_subarray():
    # 测试来自 GitHub issue #16678
    # 定义一个结构化数据类型 'point'，包含字段 'x' 和 'y'，每个字段为浮点数
    point = np.dtype([('x', float), ('y', float)])
    # 定义一个结构化数据类型 'dt'，包含字段 'code' 和 'points'，'points' 是一个包含两个 'point' 结构的数组
    dt = np.dtype([('code', int), ('points', point, (2,))])
    # 创建一个包含数据的字符串流对象
    data = StringIO("100,1,2,3,4\n200,5,6,7,8\n")
    # 期望的 NumPy 数组，包含两个元素，每个元素包含一个整数和两个点的数组
    expected = np.array(
        [
            (100, [(1., 2.), (3., 4.)]),
            (200, [(5., 6.), (7., 8.)]),
        ],
        dtype=dt
    )
    # 断言使用 np.loadtxt 函数加载数据，并与期望的数组进行比较，指定分隔符为逗号
    assert_array_equal(np.loadtxt(data, dtype=dt, delimiter=","), expected)


def test_structured_dtype_offsets():
    # 一个对齐的结构化数据类型会有额外的填充
    # 定义一个结构化数据类型 'dt'，包含多个整数字段，对齐方式为 True
    dt = np.dtype("i1, i4, i1, i4, i1, i4", align=True)
    # 创建一个包含数据的字符串流对象
    data = StringIO("1,2,3,4,5,6\n7,8,9,10,11,12\n")
    # 期望的 NumPy 数组，包含两个元素，每个元素是一个包含整数的元组
    expected = np.array([(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)], dtype=dt)
    # 断言使用 np.loadtxt 函数加载数据，并与期望的数组进行比较，指定分隔符为逗号
    assert_array_equal(np.loadtxt(data, delimiter=",", dtype=dt), expected)


@pytest.mark.parametrize("param", ("skiprows", "max_rows"))
def test_exception_negative_row_limits(param):
    """skiprows 和 max_rows 应当对负参数抛出异常。"""
    # 使用 pytest.raises 检查 np.loadtxt 函数在读取文件时，对于负的参数值抛出 ValueError 异常
    with pytest.raises(ValueError, match="argument must be nonnegative"):
        np.loadtxt("foo.bar", **{param: -3})


@pytest.mark.parametrize("param", ("skiprows", "max_rows"))
def test_exception_noninteger_row_limits(param):
    # 测试 np.loadtxt 函数对于非整数参数值抛出 TypeError 异常
    with pytest.raises(TypeError, match="argument must be an integer"):
        np.loadtxt("foo.bar", **{param: 1.0})


@pytest.mark.parametrize(
    "data, shape",
    [
        ("1 2 3 4 5\n", (1, 5)),  # 单行数据
        ("1\n2\n3\n4\n5\n", (5, 1)),  # 单列数据
    ]
)
def test_ndmin_single_row_or_col(data, shape):
    # 创建一个包含数据的字符串流对象
    arr = np.array([1, 2, 3, 4, 5])
    # 将一维数组 arr 重塑成 shape 指定的形状的二维数组 arr2d
    arr2d = arr.reshape(shape)

    # 断言使用 np.loadtxt 函数加载数据，并与一维数组 arr 进行比较
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int), arr)
    # 断言使用 np.loadtxt 函数加载数据，并与一维数组 arr 进行比较，设置 ndmin=0
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=0), arr)
    # 断言使用 np.loadtxt 函数加载数据，并与一维数组 arr 进行比较，设置 ndmin=1
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=1), arr)
    # 断言使用 np.loadtxt 函数加载数据，并与二维数组 arr2d 进行比较，设置 ndmin=2
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=2), arr2d)


@pytest.mark.parametrize("badval", [-1, 3, None, "plate of shrimp"])
def test_bad_ndmin(badval):
    # 测试 np.loadtxt 函数对于非法的 ndmin 参数值抛出 ValueError 异常
    with pytest.raises(ValueError, match="Illegal value of ndmin keyword"):
        np.loadtxt("foo.bar", ndmin=badval)


@pytest.mark.parametrize(
    "ws",
    (
            " ",  # 空格
            "\t",  # 制表符
            "\u2003",  # 空白字符
            "\u00A0",  # 不间断空格
            "\u3000",  # 表意空格
    )
)
def test_blank_lines_spaces_delimit(ws):
    txt = StringIO(
        f"1 2{ws}30\n\n{ws}\n"
        f"4 5 60{ws}\n  {ws}  \n"
        f"7 8 {ws} 90\n  # comment\n"
        f"3 2 1"
    )
    # 注意：`  # comment` 应当成功。除非 delimiter=None，应当使用任意空白字符（也许
    # 应当更接近 Python 实现
    # 创建一个预期的 NumPy 数组，包含指定的整数值
    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])
    # 使用 NumPy 的 assert_equal 函数比较两个数组是否相等
    assert_equal(
        # 使用 np.loadtxt 从文本文件中加载数据，指定数据类型为整数，分隔符为任意空白，忽略以 '#' 开始的注释
        np.loadtxt(txt, dtype=int, delimiter=None, comments="#"),
        # 将加载的数据与预期的数组进行比较
        expected
    )
# 定义一个测试函数，用于测试带有空行和注释的文本的解析
def test_blank_lines_normal_delimiter():
    # 创建一个包含特定内容的内存文本流对象
    txt = StringIO('1,2,30\n\n4,5,60\n\n7,8,90\n# comment\n3,2,1')
    # 预期的结果是一个包含特定数值的二维 NumPy 数组
    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])
    # 断言加载文本内容后的结果与预期结果相等
    assert_equal(
        np.loadtxt(txt, dtype=int, delimiter=',', comments="#"), expected
    )


# 使用参数化测试来测试不同数据类型的加载行数限制
@pytest.mark.parametrize("dtype", (float, object))
def test_maxrows_no_blank_lines(dtype):
    # 创建一个包含特定内容的内存文本流对象
    txt = StringIO("1.5,2.5\n3.0,4.0\n5.5,6.0")
    # 加载并限制最大行数为 2，数据类型由参数 dtype 决定
    res = np.loadtxt(txt, dtype=dtype, delimiter=",", max_rows=2)
    # 断言加载结果的数据类型与预期参数 dtype 相等
    assert_equal(res.dtype, dtype)
    # 断言加载的结果与预期的 NumPy 数组相等
    assert_equal(res, np.array([["1.5", "2.5"], ["3.0", "4.0"]], dtype=dtype))


# 使用参数化测试来测试异常情况下的错误消息处理
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", (np.dtype("f8"), np.dtype("i2")))
def test_exception_message_bad_values(dtype):
    # 创建一个包含特定内容的内存文本流对象
    txt = StringIO("1,2\n3,XXX\n5,6")
    # 准备预期的错误消息
    msg = f"could not convert string 'XXX' to {dtype} at row 1, column 2"
    # 使用 pytest 断言捕获指定的 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(txt, dtype=dtype, delimiter=",")


# 测试使用自定义转换器处理数据的加载
def test_converters_negative_indices():
    # 创建一个包含特定内容的内存文本流对象
    txt = StringIO('1.5,2.5\n3.0,XXX\n5.5,6.0')
    # 定义一个转换器，根据特定规则转换数据，例如将 'XXX' 转换为 NaN
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    # 预期的结果是一个包含特定数值的二维 NumPy 数组
    expected = np.array([[1.5, 2.5], [3.0, np.nan], [5.5, 6.0]])
    # 使用转换器加载数据，并断言加载结果与预期结果相等
    res = np.loadtxt(txt, dtype=np.float64, delimiter=",", converters=conv)
    assert_equal(res, expected)


# 测试在使用 usecols 限定列数的情况下，加载数据并处理负索引的转换
def test_converters_negative_indices_with_usecols():
    # 创建一个包含特定内容的内存文本流对象
    txt = StringIO('1.5,2.5,3.5\n3.0,4.0,XXX\n5.5,6.0,7.5\n')
    # 定义一个转换器，根据特定规则转换数据，例如将 'XXX' 转换为 NaN
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    # 预期的结果是一个包含特定数值的二维 NumPy 数组
    expected = np.array([[1.5, 3.5], [3.0, np.nan], [5.5, 7.5]])
    # 使用 usecols 参数限定要加载的列，并使用转换器处理数据加载
    res = np.loadtxt(
        txt,
        dtype=np.float64,
        delimiter=",",
        converters=conv,
        usecols=[0, -1],
    )
    # 断言加载结果与预期结果相等
    assert_equal(res, expected)

    # 第二个测试用例，用于测试变量行数的情况
    res = np.loadtxt(StringIO('''0,1,2\n0,1,2,3,4'''), delimiter=",",
                     usecols=[0, -1], converters={-1: (lambda x: -1)})
    # 断言加载结果与预期结果相等
    assert_array_equal(res, [[0, -1], [0, -1]])


# 测试在不同行数列数不一致情况下是否能正确抛出 ValueError 异常
def test_ragged_error():
    # 准备包含不同行数的数据列表
    rows = ["1,2,3", "1,2,3", "4,3,2,1"]
    # 使用 pytest 断言捕获指定的 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError,
                       match="the number of columns changed from 3 to 4 at row 3"):
        np.loadtxt(rows, delimiter=",")


# 测试在不同行数列数不一致情况下是否能正确处理 usecols 参数
def test_ragged_usecols():
    # 测试即使在列数不一致的情况下，usecols 和负索引也能正确处理
    txt = StringIO("0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n")
    # 预期的结果是一个包含特定数值的二维 NumPy 数组
    expected = np.array([[0, 0], [0, 0], [0, 0]])
    # 使用 usecols 参数限定要加载的列，并使用负索引转换器处理数据加载
    res = np.loadtxt(txt, dtype=float, delimiter=",", usecols=[0, -2])
    # 断言加载结果与预期结果相等
    assert_equal(res, expected)

    # 准备另一个测试用例，包含不同行数和错误的 usecols 参数
    txt = StringIO("0,0,XXX\n0\n0,XXX,XXX,0,XXX\n")
    # 使用 pytest 断言捕获指定的 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError,
                       match="invalid column index -2 at row 2 with 1 columns"):
        # 加载数据时，将会抛出错误，因为第二行不存在负索引为 -2 的列
        np.loadtxt(txt, dtype=float, delimiter=",", usecols=[0, -2])


# 测试空 usecols 参数的情况
def test_empty_usecols():
    txt = StringIO("0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n")
    # 使用 NumPy 加载文本文件 `txt`，返回一个 NumPy 数组 `res`
    res = np.loadtxt(txt, dtype=np.dtype([]), delimiter=",", usecols=[])
    # 断言：确保数组 `res` 的形状为 (3,)
    assert res.shape == (3,)
    # 断言：确保数组 `res` 的数据类型为一个空的结构化 NumPy 数据类型
    assert res.dtype == np.dtype([])
@pytest.mark.parametrize("c1", ["a", "の", "🫕"])
@pytest.mark.parametrize("c2", ["a", "の", "🫕"])
def test_large_unicode_characters(c1, c2):
    # 创建包含大量 Unicode 字符的测试用例，c1 和 c2 覆盖 ASCII、16 位和 32 位字符范围。
    txt = StringIO(f"a,{c1},c,1.0\ne,{c2},2.0,g")
    # 将文本数据封装为 StringIO 对象
    res = np.loadtxt(txt, dtype=np.dtype('U12'), delimiter=",")
    # 使用 NumPy 加载文本数据到数组 res 中，使用 Unicode 类型，每个元素最多12个字符，使用逗号分隔
    expected = np.array(
        [f"a,{c1},c,1.0".split(","), f"e,{c2},2.0,g".split(",")],
        dtype=np.dtype('U12')
    )
    # 创建预期结果数组，每个元素也是最多12个字符的 Unicode 类型
    assert_equal(res, expected)
    # 断言实际结果与预期结果相等


def test_unicode_with_converter():
    # 测试带有转换器的 Unicode 处理
    txt = StringIO("cat,dog\nαβγ,δεζ\nabc,def\n")
    # 将文本数据封装为 StringIO 对象
    conv = {0: lambda s: s.upper()}
    # 定义转换器，将第一列字符转换为大写
    res = np.loadtxt(
        txt,
        dtype=np.dtype("U12"),
        converters=conv,
        delimiter=",",
        encoding=None
    )
    # 使用 NumPy 加载文本数据到数组 res 中，使用 Unicode 类型，应用转换器，逗号分隔
    expected = np.array([['CAT', 'dog'], ['ΑΒΓ', 'δεζ'], ['ABC', 'def']])
    # 创建预期结果数组，每个元素最多12个字符的 Unicode 类型
    assert_equal(res, expected)
    # 断言实际结果与预期结果相等


def test_converter_with_structured_dtype():
    # 测试结构化数据类型和转换器的使用
    txt = StringIO('1.5,2.5,Abc\n3.0,4.0,dEf\n5.5,6.0,ghI\n')
    # 将文本数据封装为 StringIO 对象
    dt = np.dtype([('m', np.int32), ('r', np.float32), ('code', 'U8')])
    # 定义结构化数据类型，包括整数、浮点数和 Unicode 字符串
    conv = {0: lambda s: int(10*float(s)), -1: lambda s: s.upper()}
    # 定义转换器，将第一列乘以10转换为整数，将最后一列转换为大写
    res = np.loadtxt(txt, dtype=dt, delimiter=",", converters=conv)
    # 使用 NumPy 加载文本数据到结构化数组 res 中，应用转换器，逗号分隔
    expected = np.array(
        [(15, 2.5, 'ABC'), (30, 4.0, 'DEF'), (55, 6.0, 'GHI')], dtype=dt
    )
    # 创建预期结果结构化数组
    assert_equal(res, expected)
    # 断言实际结果与预期结果相等


def test_converter_with_unicode_dtype():
    """
    当使用 'bytes' 编码时，标记 tokens 之前编码。这意味着转换器的输出可能是字节而不是 `read_rows` 预期的 Unicode。
    此测试检查以上场景的输出是否在由 `read_rows` 解析之前被正确解码。
    """
    txt = StringIO('abc,def\nrst,xyz')
    # 将文本数据封装为 StringIO 对象
    conv = bytes.upper
    # 定义转换器，将输入的字节转换为大写
    res = np.loadtxt(
            txt, dtype=np.dtype("U3"), converters=conv, delimiter=",",
            encoding="bytes")
    # 使用 NumPy 加载文本数据到数组 res 中，使用最多3个字符的 Unicode 类型，应用转换器，逗号分隔，使用字节编码
    expected = np.array([['ABC', 'DEF'], ['RST', 'XYZ']])
    # 创建预期结果数组
    assert_equal(res, expected)
    # 断言实际结果与预期结果相等


def test_read_huge_row():
    # 测试读取超大行数据
    row = "1.5, 2.5," * 50000
    # 创建一个超大的行字符串
    row = row[:-1] + "\n"
    # 将字符串结尾替换为换行符
    txt = StringIO(row * 2)
    # 将文本数据封装为 StringIO 对象
    res = np.loadtxt(txt, delimiter=",", dtype=float)
    # 使用 NumPy 加载文本数据到数组 res 中，逗号分隔，数据类型为浮点数
    assert_equal(res, np.tile([1.5, 2.5], (2, 50000)))
    # 断言实际结果与预期结果相等


@pytest.mark.parametrize("dtype", "edfgFDG")
def test_huge_float(dtype):
    # 测试处理大浮点数的情况，覆盖一个不经常发生的非优化路径
    field = "0" * 1000 + ".123456789"
    # 创建一个大数值字段
    dtype = np.dtype(dtype)
    # 定义数据类型
    value = np.loadtxt([field], dtype=dtype)[()]
    # 使用 NumPy 加载文本数据到数组 value 中，使用指定的数据类型
    assert value == dtype.type("0.123456789")
    # 断言实际结果与预期结果相等


@pytest.mark.parametrize(
    ("given_dtype", "expected_dtype"),
    [
        ("S", np.dtype("S5")),
        ("U", np.dtype("U5")),
    ],
)
def test_string_no_length_given(given_dtype, expected_dtype):
    """
    给定的数据类型只有 'S' 或 'U' 而没有长度。在这些情况下，结果的长度由文件中找到的最长字符串决定。
    """
    txt = StringIO("AAA,5-1\nBBBBB,0-3\nC,4-9\n")
    # 将文本数据封装为 StringIO 对象
    res = np.loadtxt(txt, dtype=given_dtype, delimiter=",")
    # 使用 NumPy 加载文本数据到数组 res 中，使用给定的数据类型，逗号分隔
    # 创建一个预期的 NumPy 数组，包含指定的数据和数据类型
    expected = np.array(
        [['AAA', '5-1'], ['BBBBB', '0-3'], ['C', '4-9']], dtype=expected_dtype
    )
    # 使用 assert_equal 函数比较两个对象 res 和 expected 是否相等
    assert_equal(res, expected)
    # 使用 assert_equal 函数比较对象 res 的数据类型是否与预期的数据类型 expected_dtype 相等
    assert_equal(res.dtype, expected_dtype)
# 测试浮点数转换的准确性，验证转换为 float64 是否与 Python 内置的 float 函数一致。
def test_float_conversion():
    """
    Some tests that the conversion to float64 works as accurately as the
    Python built-in `float` function. In a naive version of the float parser,
    these strings resulted in values that were off by an ULP or two.
    """
    # 定义待转换的字符串列表
    strings = [
        '0.9999999999999999',
        '9876543210.123456',
        '5.43215432154321e+300',
        '0.901',
        '0.333',
    ]
    # 将字符串列表写入内存中的文本流
    txt = StringIO('\n'.join(strings))
    # 使用 numpy 的 loadtxt 函数加载数据
    res = np.loadtxt(txt)
    # 构建预期结果的 numpy 数组，通过 float 函数转换每个字符串为 float 类型
    expected = np.array([float(s) for s in strings])
    # 使用 assert_equal 断言 res 和 expected 数组相等
    assert_equal(res, expected)


# 测试布尔值转换
def test_bool():
    # 通过整数测试布尔值的简单情况
    txt = StringIO("1, 0\n10, -1")
    # 使用 numpy 的 loadtxt 函数加载数据，指定数据类型为 bool，分隔符为逗号
    res = np.loadtxt(txt, dtype=bool, delimiter=",")
    # 断言结果数组的数据类型为 bool
    assert res.dtype == bool
    # 断言数组内容与预期数组相等
    assert_array_equal(res, [[True, False], [True, True]])
    # 确保在字节级别上只使用 1 和 0
    assert_array_equal(res.view(np.uint8), [[1, 0], [1, 1]])


# 测试整数符号的处理
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.filterwarnings("error:.*integer via a float.*:DeprecationWarning")
def test_integer_signs(dtype):
    # 将 dtype 转换为 numpy 的数据类型
    dtype = np.dtype(dtype)
    # 断言加载包含 "+2" 的数据返回值为 2
    assert np.loadtxt(["+2"], dtype=dtype) == 2
    # 如果数据类型为无符号整数，断言加载包含 "-1\n" 的数据会引发 ValueError 异常
    if dtype.kind == "u":
        with pytest.raises(ValueError):
            np.loadtxt(["-1\n"], dtype=dtype)
    else:
        # 断言加载包含 "-2\n" 的数据返回值为 -2
        assert np.loadtxt(["-2\n"], dtype=dtype) == -2

    # 对于不合法的符号组合，如 "++", "+-", "--", "-+"，断言加载时会引发 ValueError 异常
    for sign in ["++", "+-", "--", "-+"]:
        with pytest.raises(ValueError):
            np.loadtxt([f"{sign}2\n"], dtype=dtype)


# 测试隐式将浮点数转换为整数时的错误处理
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.filterwarnings("error:.*integer via a float.*:DeprecationWarning")
def test_implicit_cast_float_to_int_fails(dtype):
    # 定义包含浮点数和整数的文本流
    txt = StringIO("1.0, 2.1, 3.7\n4, 5, 6")
    # 断言加载时会引发 ValueError 异常
    with pytest.raises(ValueError):
        np.loadtxt(txt, dtype=dtype, delimiter=",")


# 测试复数的解析
@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("with_parens", (False, True))
def test_complex_parsing(dtype, with_parens):
    # 定义包含复数字符串的文本流
    s = "(1.0-2.5j),3.75,(7+-5.0j)\n(4),(-19e2j),(0)"
    if not with_parens:
        s = s.replace("(", "").replace(")", "")

    # 使用 numpy 的 loadtxt 函数加载数据，指定数据类型为复数类型，分隔符为逗号
    res = np.loadtxt(StringIO(s), dtype=dtype, delimiter=",")
    # 构建预期结果的 numpy 数组
    expected = np.array(
        [[1.0-2.5j, 3.75, 7-5j], [4.0, -1900j, 0]], dtype=dtype
    )
    # 使用 assert_equal 断言 res 和 expected 数组相等
    assert_equal(res, expected)


# 测试从生成器中读取数据
def test_read_from_generator():
    # 定义生成器函数
    def gen():
        for i in range(4):
            yield f"{i},{2*i},{i**2}"

    # 使用 numpy 的 loadtxt 函数加载生成器生成的数据，指定数据类型为整数，分隔符为逗号
    res = np.loadtxt(gen(), dtype=int, delimiter=",")
    # 构建预期结果的 numpy 数组
    expected = np.array([[0, 0, 0], [1, 2, 1], [2, 4, 4], [3, 6, 9]])
    # 使用 assert_equal 断言 res 和 expected 数组相等
    assert_equal(res, expected)


# 测试从生成器中读取多种类型的数据
def test_read_from_generator_multitype():
    # 定义生成器函数
    def gen():
        for i in range(3):
            yield f"{i} {i / 4}"

    # 使用 numpy 的 loadtxt 函数加载生成器生成的数据，指定数据类型为 "i, d"，分隔符为空格
    res = np.loadtxt(gen(), dtype="i, d", delimiter=" ")
    # 定义预期的 NumPy 数组，包含两列，第一列为整数类型，第二列为双精度浮点数类型
    expected = np.array([(0, 0.0), (1, 0.25), (2, 0.5)], dtype="i, d")
    # 使用 assert_equal 函数比较 res 和 expected，确保它们相等
    assert_equal(res, expected)
def test_read_from_bad_generator():
    # 定义一个生成器函数 `gen()`，生成器会依次产生字符串、字节串和整数
    def gen():
        yield from ["1,2", b"3, 5", 12738]

    # 使用 pytest 检查调用 `np.loadtxt()` 时抛出的 TypeError 异常，并验证异常消息
    with pytest.raises(
            TypeError, match=r"non-string returned while reading data"):
        np.loadtxt(gen(), dtype="i, i", delimiter=",")


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_object_cleanup_on_read_error():
    # 创建一个对象 sentinel 作为测试目的
    sentinel = object()
    # 初始化一个计数器 already_read，记录已经读取的次数
    already_read = 0

    # 定义一个转换函数 conv(x)，用于处理每一行的数据并返回 sentinel
    def conv(x):
        nonlocal already_read
        # 如果 already_read 大于 4999，抛出 ValueError 异常
        if already_read > 4999:
            raise ValueError("failed half-way through!")
        already_read += 1
        return sentinel

    # 创建一个包含大量数据的 StringIO 对象 txt
    txt = StringIO("x\n" * 10000)

    # 使用 pytest 检查调用 `np.loadtxt()` 时抛出的 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match="at row 5000, column 1"):
        np.loadtxt(txt, dtype=object, converters={0: conv})

    # 检查 sentinel 的引用计数是否为 2
    assert sys.getrefcount(sentinel) == 2


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_character_not_bytes_compatible():
    """Test exception when a character cannot be encoded as 'S'."""
    # 创建一个包含特殊字符 '–'（Unicode码点 \u2013）的 StringIO 对象 data
    data = StringIO("–")
    # 使用 pytest 检查调用 `np.loadtxt()` 时抛出的 ValueError 异常
    with pytest.raises(ValueError):
        np.loadtxt(data, dtype="S5")


@pytest.mark.parametrize("conv", (0, [float], ""))
def test_invalid_converter(conv):
    # 定义期望的错误消息
    msg = (
        "converters must be a dictionary mapping columns to converter "
        "functions or a single callable."
    )
    # 使用 pytest 检查调用 `np.loadtxt()` 时抛出的 TypeError 异常，并验证异常消息
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(StringIO("1 2\n3 4"), converters=conv)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_converters_dict_raises_non_integer_key():
    # 使用 pytest 检查调用 `np.loadtxt()` 时抛出的 TypeError 异常，并验证异常消息
    with pytest.raises(TypeError, match="keys of the converters dict"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={"a": int})
    # 使用 pytest 检查调用 `np.loadtxt()` 时抛出的 TypeError 异常，并验证异常消息
    with pytest.raises(TypeError, match="keys of the converters dict"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={"a": int}, usecols=0)


@pytest.mark.parametrize("bad_col_ind", (3, -3))
def test_converters_dict_raises_non_col_key(bad_col_ind):
    # 创建一个包含数据的 StringIO 对象 data
    data = StringIO("1 2\n3 4")
    # 使用 pytest 检查调用 `np.loadtxt()` 时抛出的 ValueError 异常，并验证异常消息
    with pytest.raises(ValueError, match="converter specified for column"):
        np.loadtxt(data, converters={bad_col_ind: int})


def test_converters_dict_raises_val_not_callable():
    # 使用 pytest 检查调用 `np.loadtxt()` 时抛出的 TypeError 异常，并验证异常消息
    with pytest.raises(TypeError,
                match="values of the converters dictionary must be callable"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={0: 1})


@pytest.mark.parametrize("q", ('"', "'", "`"))
def test_quoted_field(q):
    # 创建一个包含带引号字段的数据的 StringIO 对象 txt
    txt = StringIO(
        f"{q}alpha, x{q}, 2.5\n{q}beta, y{q}, 4.5\n{q}gamma, z{q}, 5.0\n"
    )
    # 定义期望的数据类型 dtype
    dtype = np.dtype([('f0', 'U8'), ('f1', np.float64)])
    # 定义期望的结果数组 expected
    expected = np.array(
        [("alpha, x", 2.5), ("beta, y", 4.5), ("gamma, z", 5.0)], dtype=dtype
    )

    # 调用 `np.loadtxt()` 加载数据，并将结果存储在 res 中
    res = np.loadtxt(txt, dtype=dtype, delimiter=",", quotechar=q)
    # 使用 assert_array_equal 检查 res 是否与期望的结果数组 expected 相等
    assert_array_equal(res, expected)


@pytest.mark.parametrize("q", ('"', "'", "`"))
def test_quoted_field_with_whitepace_delimiter(q):
    # 此测试未提供完整的代码示例，因此无法添加注释
    pass
    # 创建一个包含指定文本的内存中的文本流对象
    txt = StringIO(
        f"{q}alpha, x{q}     2.5\n{q}beta, y{q} 4.5\n{q}gamma, z{q}   5.0\n"
    )
    # 定义一个 NumPy 数据类型，包含两个字段：一个是最大长度为 8 的 Unicode 字符串，另一个是 64 位浮点数
    dtype = np.dtype([('f0', 'U8'), ('f1', np.float64)])
    # 创建一个 NumPy 数组，用于存储预期的数据，每个元素是一个元组，元组包含一个字符串和一个浮点数
    expected = np.array(
        [("alpha, x", 2.5), ("beta, y", 4.5), ("gamma, z", 5.0)], dtype=dtype
    )
    
    # 使用 np.loadtxt 从文本流中加载数据，并指定数据类型、分隔符和引用字符
    res = np.loadtxt(txt, dtype=dtype, delimiter=None, quotechar=q)
    # 使用 assert_array_equal 断言函数，检查加载的数据是否与预期数据一致
    assert_array_equal(res, expected)
def test_quoted_field_is_not_empty_nonstrict():
    # Same as test_quoted_field_is_not_empty but check that we are not strict
    # about missing closing quote (this is the `csv.reader` default also)
    # 创建包含数据的字符串文件对象
    txt = StringIO('1\n\n"4"\n"')
    # 期望的结果数组
    expected = np.array(["1", "4", ""], dtype="U1")
    # 使用 NumPy 的 loadtxt 函数从文本文件中加载数据
    res = np.loadtxt(txt, delimiter=",", dtype="U1", quotechar='"')
    # 断言，验证加载的数据 res 是否等于预期的数据 expected
    assert_equal(res, expected)
def test_consecutive_quotechar_escaped():
    # 创建一个字符串缓冲区，内容为包含连续引号的文本
    txt = StringIO('"Hello, my name is ""Monty""!"')
    # 创建预期的 NumPy 数组，包含解析后的字符串
    expected = np.array('Hello, my name is "Monty"!', dtype="U40")
    # 使用 np.loadtxt 从文本中加载数据到 res 变量中
    res = np.loadtxt(txt, dtype="U40", delimiter=",", quotechar='"')
    # 断言 res 和 expected 数组相等
    assert_equal(res, expected)


@pytest.mark.parametrize("data", ("", "\n\n\n", "# 1 2 3\n# 4 5 6\n"))
@pytest.mark.parametrize("ndmin", (0, 1, 2))
@pytest.mark.parametrize("usecols", [None, (1, 2, 3)])
def test_warn_on_no_data(data, ndmin, usecols):
    """检查当输入数据为空时是否发出 UserWarning。"""
    if usecols is not None:
        expected_shape = (0, 3)
    elif ndmin == 2:
        expected_shape = (0, 1)  # 猜测只有一列数据？！
    else:
        expected_shape = (0,)

    # 创建一个包含指定数据的字符串缓冲区
    txt = StringIO(data)
    # 使用 pytest 的 warn 环境，检查是否发出 UserWarning 并匹配指定消息
    with pytest.warns(UserWarning, match="input contained no data"):
        # 使用 np.loadtxt 从文本中加载数据到 res 变量中
        res = np.loadtxt(txt, ndmin=ndmin, usecols=usecols)
    # 断言加载后的数据形状与预期形状相同
    assert res.shape == expected_shape

    # 使用临时文件写入指定数据
    with NamedTemporaryFile(mode="w") as fh:
        fh.write(data)
        fh.seek(0)
        # 使用 pytest 的 warn 环境，检查是否发出 UserWarning 并匹配指定消息
        with pytest.warns(UserWarning, match="input contained no data"):
            # 使用 np.loadtxt 从文本中加载数据到 res 变量中
            res = np.loadtxt(txt, ndmin=ndmin, usecols=usecols)
        # 断言加载后的数据形状与预期形状相同
        assert res.shape == expected_shape


@pytest.mark.parametrize("skiprows", (2, 3))
def test_warn_on_skipped_data(skiprows):
    # 创建包含数据的字符串缓冲区
    data = "1 2 3\n4 5 6"
    txt = StringIO(data)
    # 使用 pytest 的 warn 环境，检查是否发出 UserWarning 并匹配指定消息
    with pytest.warns(UserWarning, match="input contained no data"):
        # 使用 np.loadtxt 从文本中加载数据，跳过指定行数
        np.loadtxt(txt, skiprows=skiprows)


@pytest.mark.parametrize(["dtype", "value"], [
        ("i2", 0x0001), ("u2", 0x0001),
        ("i4", 0x00010203), ("u4", 0x00010203),
        ("i8", 0x0001020304050607), ("u8", 0x0001020304050607),
        ("float16", 3.07e-05),
        ("float32", 9.2557e-41), ("complex64", 9.2557e-41+2.8622554e-29j),
        ("float64", -1.758571353180402e-24),
        ("complex128", repr(5.406409232372729e-29-1.758571353180402e-24j)),
        ("longdouble", 0x01020304050607),
        ("clongdouble", repr(0x01020304050607 + (0x00121314151617 * 1j))),
        ("U2", "\U00010203\U000a0b0c")])
@pytest.mark.parametrize("swap", [True, False])
def test_byteswapping_and_unaligned(dtype, value, swap):
    # 尝试创建具有 "有趣" 值的数据，确保在有效的 Unicode 范围内
    dtype = np.dtype(dtype)
    # 创建包含指定数据的列表
    data = [f"x,{value}\n"]
    # 如果 swap 为 True，则交换字节顺序
    if swap:
        dtype = dtype.newbyteorder()
    # 创建具有指定结构的 dtype
    full_dt = np.dtype([("a", "S1"), ("b", dtype)], align=False)
    # 确保 "b" 字段的对齐方式为非对齐
    assert full_dt.fields["b"][1] == 1
    # 使用 numpy 的 loadtxt 函数从数据中加载内容，指定数据类型为 full_dt，分隔符为逗号
    # 使用 max_rows 参数限制加载的行数，防止过度分配内存
    res = np.loadtxt(data, dtype=full_dt, delimiter=",", max_rows=1)

    # 使用断言确保 res 数组中字段 "b" 的值等于给定的 value 值
    assert res["b"] == dtype.type(value)
# 使用 pytest 的 parametrize 装饰器为单元测试函数提供多组参数化输入
@pytest.mark.parametrize("dtype",
        np.typecodes["AllInteger"] + "efdFD" + "?")
def test_unicode_whitespace_stripping(dtype):
    # 测试所有数字类型（包括布尔型）是否能正确去除空白字符
    # \u202F 是一个窄的不换行空格，`\n` 表示一个普通的换行符
    # 目前跳过 float128，因为它不总是支持此功能且没有“自定义”解析
    txt = StringIO(' 3 ,"\u202F2\n"')
    # 使用 np.loadtxt 函数从文本流中加载数据，并指定数据类型、分隔符和引号字符
    res = np.loadtxt(txt, dtype=dtype, delimiter=",", quotechar='"')
    # 断言加载的数据与预期的数组相等
    assert_array_equal(res, np.array([3, 2]).astype(dtype))


@pytest.mark.parametrize("dtype", "FD")
def test_unicode_whitespace_stripping_complex(dtype):
    # 复数有一些额外的情况，因为它有两个组件和括号
    line = " 1 , 2+3j , ( 4+5j ), ( 6+-7j )  , 8j , ( 9j ) \n"
    data = [line, line.replace(" ", "\u202F")]
    # 测试加载包含复数的数据时是否正确去除空白字符
    res = np.loadtxt(data, dtype=dtype, delimiter=',')
    # 断言加载的数据与预期的二维数组相等
    assert_array_equal(res, np.array([[1, 2+3j, 4+5j, 6-7j, 8j, 9j]] * 2))


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", "FD")
@pytest.mark.parametrize("field",
        ["1 +2j", "1+ 2j", "1+2 j", "1+-+3", "(1j", "(1", "(1+2j", "1+2j)"])
def test_bad_complex(dtype, field):
    # 使用 pytest.raises 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 测试加载包含错误格式的复数字符串时是否会抛出异常
        np.loadtxt([field + "\n"], dtype=dtype, delimiter=",")


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype",
            np.typecodes["AllInteger"] + "efgdFDG" + "?")
def test_nul_character_error(dtype):
    # 测试是否能正确识别 `\0` 字符，并抛出 ValueError 异常
    # 即使前面的内容是有效的（不是所有内容都能在内部解析）
    if dtype.lower() == "g":
        pytest.xfail("longdouble/clongdouble assignment may misbehave.")
    with pytest.raises(ValueError):
        np.loadtxt(["1\000"], dtype=dtype, delimiter=",", quotechar='"')


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype",
        np.typecodes["AllInteger"] + "efgdFDG" + "?")
def test_no_thousands_support(dtype):
    # 主要用于文档说明行为，Python 支持像 1_1 这样的千分位表示
    # （e 和 G 可能会使用不同的转换和支持，这是一个 bug 但确实发生了...）
    if dtype == "e":
        pytest.skip("half assignment currently uses Python float converter")
    if dtype in "eG":
        pytest.xfail("clongdouble assignment is buggy (uses `complex`?).")

    assert int("1_1") == float("1_1") == complex("1_1") == 11
    with pytest.raises(ValueError):
        np.loadtxt(["1_1\n"], dtype=dtype)


@pytest.mark.parametrize("data", [
    ["1,2\n", "2\n,3\n"],
    ["1,2\n", "2\r,3\n"]])
def test_bad_newline_in_iterator(data):
    # 在 NumPy <=1.22 中这是被接受的，因为换行符是完全
    # 设置错误消息字符串，用于匹配 pytest 抛出的 ValueError 异常
    msg = "Found an unquoted embedded newline within a single line"
    # 使用 pytest 提供的上下文管理器 `pytest.raises` 来捕获 ValueError 异常，
    # 并检查其异常消息是否与预设的 `msg` 相匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 numpy 的 loadtxt 函数来加载数据，指定分隔符为逗号 `,`
        np.loadtxt(data, delimiter=",")
@pytest.mark.parametrize("data", [
    ["1,2\n", "2,3\r\n"],  # 定义测试参数，包括包含不同换行符的数据
    ["1,2\n", "'2\n',3\n"],  # 含有引号的换行数据
    ["1,2\n", "'2\r',3\n"],  # 含有引号的回车数据
    ["1,2\n", "'2\r\n',3\n"],  # 含有引号的回车换行数据
])
def test_good_newline_in_iterator(data):
    # 在这里引号内的换行符不会被转换，但会被视为空白字符。
    res = np.loadtxt(data, delimiter=",", quotechar="'")  # 使用 numpy 的 loadtxt 函数加载数据
    assert_array_equal(res, [[1., 2.], [2., 3.]])


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n"])
def test_universal_newlines_quoted(newline):
    # 检查在引用字段中不应用通用换行符支持的情况下的情况
    # （注意，行必须以换行符结尾，否则引用字段将不包括换行符）
    data = ['1,"2\n"\n', '3,"4\n', '1"\n']
    data = [row.replace("\n", newline) for row in data]  # 替换每行的换行符为指定的换行符
    res = np.loadtxt(data, dtype=object, delimiter=",", quotechar='"')  # 使用 numpy 的 loadtxt 函数加载数据
    assert_array_equal(res, [['1', f'2{newline}'], ['3', f'4{newline}1']])


def test_null_character():
    # 检查 NUL 字符是否不具有特殊性的基本测试：
    res = np.loadtxt(["1\0002\0003\n", "4\0005\0006"], delimiter="\000")  # 使用 numpy 的 loadtxt 函数加载数据
    assert_array_equal(res, [[1, 2, 3], [4, 5, 6]])

    # 同样不作为字段的一部分（避免 Unicode/数组会将 \0 去掉）
    res = np.loadtxt(["1\000,2\000,3\n", "4\000,5\000,6"],
                     delimiter=",", dtype=object)  # 使用 numpy 的 loadtxt 函数加载数据
    assert res.tolist() == [["1\000", "2\000", "3"], ["4\000", "5\000", "6"]]


def test_iterator_fails_getting_next_line():
    class BadSequence:
        def __len__(self):
            return 100

        def __getitem__(self, item):
            if item == 50:
                raise RuntimeError("Bad things happened!")
            return f"{item}, {item+1}"

    with pytest.raises(RuntimeError, match="Bad things happened!"):
        np.loadtxt(BadSequence(), dtype=int, delimiter=",")  # 使用 numpy 的 loadtxt 函数加载数据


class TestCReaderUnitTests:
    # 这些是路径上不应该触发的内部测试，除非出现非常严重的问题。
    def test_not_an_filelike(self):
        with pytest.raises(AttributeError, match=".*read"):
            np._core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=True)

    def test_filelike_read_fails(self):
        # 只有当 loadtxt 打开文件时才能到达，所以很难通过公共接口实现
        # （尽管在当前的 "DataClass" 支持下可能不是不可能的）。
        class BadFileLike:
            counter = 0

            def read(self, size):
                self.counter += 1
                if self.counter > 20:
                    raise RuntimeError("Bad bad bad!")
                return "1,2,3\n"

        with pytest.raises(RuntimeError, match="Bad bad bad!"):
            np._core._multiarray_umath._load_from_filelike(
                BadFileLike(), dtype=np.dtype("i"), filelike=True)
    # 定义一个测试用例，用于测试当 read 方法返回非字符串类型时的情况
    def test_filelike_bad_read(self):
        # 如果 loadtxt 打开文件，则可以到达此处，所以很难通过公共接口完成
        # 虽然在当前的“DataClass”支持下可能并非不可能。

        # 定义一个模拟的文件类 BadFileLike
        class BadFileLike:
            counter = 0

            # 重载 read 方法，返回一个整数而不是字符串
            def read(self, size):
                return 1234  # not a string!

        # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配特定的错误信息
        with pytest.raises(TypeError,
                    match="non-string returned while reading data"):
            # 调用被测试的函数，传入 BadFileLike 实例作为文件对象
            np._core._multiarray_umath._load_from_filelike(
                BadFileLike(), dtype=np.dtype("i"), filelike=True)

    # 定义一个测试用例，用于测试当对象不是可迭代对象时的情况
    def test_not_an_iter(self):
        # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配特定的错误信息
        with pytest.raises(TypeError,
                    match="error reading from object, expected an iterable"):
            # 调用被测试的函数，传入普通对象而不是可迭代对象
            np._core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=False)

    # 定义一个测试用例，用于测试当 dtype 参数不正确时的情况
    def test_bad_type(self):
        # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配特定的错误信息
        with pytest.raises(TypeError, match="internal error: dtype must"):
            # 调用被测试的函数，传入错误的 dtype 类型
            np._core._multiarray_umath._load_from_filelike(
                object(), dtype="i", filelike=False)

    # 定义一个测试用例，用于测试当 encoding 参数不正确时的情况
    def test_bad_encoding(self):
        # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配特定的错误信息
        with pytest.raises(TypeError, match="encoding must be a unicode"):
            # 调用被测试的函数，传入错误的 encoding 类型
            np._core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=False, encoding=123)

    # 使用 pytest 的参数化功能定义一个测试用例，测试不同的 newline 参数
    @pytest.mark.parametrize("newline", ["\r", "\n", "\r\n"])
    def test_manual_universal_newlines(self, newline):
        # 这部分当前对用户不可用，因为我们应该始终以启用了 universal newlines 的方式打开文件 `newlines=None`
        # （从迭代器读取数据使用了稍微不同的代码路径）。
        # 我们对 `newline="\r"` 或 `newline="\n"` 没有真正的支持，因为用户不能指定这些选项。

        # 创建一个 StringIO 对象，模拟包含特定 newline 的数据
        data = StringIO('0\n1\n"2\n"\n3\n4 #\n'.replace("\n", newline),
                        newline="")

        # 调用被测试的函数，传入 StringIO 对象以及其他参数
        res = np._core._multiarray_umath._load_from_filelike(
            data, dtype=np.dtype("U10"), filelike=True,
            quote='"', comment="#", skiplines=1)
        
        # 使用 assert_array_equal 断言函数验证结果的正确性
        assert_array_equal(res[:, 0], ["1", f"2{newline}", "3", "4 "])
# 当分隔符与注释字符冲突时，应该抛出TypeError异常，提示控制字符不兼容
def test_delimiter_comment_collision_raises():
    # 使用 pytest 模块验证加载文本时抛出TypeError异常，异常消息中包含“control characters”和“incompatible”
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        # 使用 numpy 的 loadtxt 函数加载以逗号分隔的文本数据，指定分隔符为逗号，注释字符也为逗号
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", comments=",")


# 当分隔符与引用字符冲突时，应该抛出TypeError异常，提示控制字符不兼容
def test_delimiter_quotechar_collision_raises():
    # 使用 pytest 模块验证加载文本时抛出TypeError异常，异常消息中包含“control characters”和“incompatible”
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        # 使用 numpy 的 loadtxt 函数加载以逗号分隔的文本数据，指定分隔符为逗号，引用字符也为逗号
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", quotechar=",")


# 当注释字符与引用字符冲突时，应该抛出TypeError异常，提示控制字符不兼容
def test_comment_quotechar_collision_raises():
    # 使用 pytest 模块验证加载文本时抛出TypeError异常，异常消息中包含“control characters”和“incompatible”
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        # 使用 numpy 的 loadtxt 函数加载空格分隔的文本数据，指定注释字符为井号，引用字符也为井号
        np.loadtxt(StringIO("1 2 3"), comments="#", quotechar="#")


# 当分隔符与多个注释字符冲突时，应该抛出TypeError异常，提示注释字符不能包括分隔符
def test_delimiter_and_multiple_comments_collision_raises():
    # 使用 pytest 模块验证加载文本时抛出TypeError异常，异常消息中包含“Comment characters”和“cannot include the delimiter”
    with pytest.raises(
        TypeError, match="Comment characters.*cannot include the delimiter"
    ):
        # 使用 numpy 的 loadtxt 函数加载以逗号分隔的文本数据，指定分隔符为逗号，注释字符包括井号和逗号
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", comments=["#", ","])


# 使用 pytest.mark.parametrize 注册的参数化测试，测试空白字符与默认分隔符冲突时是否抛出TypeError异常
@pytest.mark.parametrize(
    "ws",
    (
        " ",  # 空格
        "\t",  # 制表符
        "\u2003",  # EM 空白
        "\u00A0",  # 不间断空白
        "\u3000",  # 表意字符空白
    )
)
def test_collision_with_default_delimiter_raises(ws):
    # 使用 pytest 模块验证加载文本时抛出TypeError异常，异常消息中包含“control characters”和“incompatible”
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        # 使用 numpy 的 loadtxt 函数加载带有空白字符分隔的文本数据，指定注释字符为当前空白字符
        np.loadtxt(StringIO(f"1{ws}2{ws}3\n4{ws}5{ws}6\n"), comments=ws)
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        # 使用 numpy 的 loadtxt 函数加载带有空白字符分隔的文本数据，指定引用字符为当前空白字符
        np.loadtxt(StringIO(f"1{ws}2{ws}3\n4{ws}5{ws}6\n"), quotechar=ws)


# 使用 pytest.mark.parametrize 注册的参数化测试，测试控制字符与换行符冲突时是否抛出TypeError异常
@pytest.mark.parametrize("nl", ("\n", "\r"))
def test_control_character_newline_raises(nl):
    # 准备包含换行符的文本数据
    txt = StringIO(f"1{nl}2{nl}3{nl}{nl}4{nl}5{nl}6{nl}{nl}")
    # 准备异常消息
    msg = "control character.*cannot be a newline"
    # 使用 pytest 模块验证加载文本时抛出TypeError异常，异常消息中包含“control character”和“cannot be a newline”
    with pytest.raises(TypeError, match=msg):
        # 使用 numpy 的 loadtxt 函数加载文本数据，指定分隔符为当前换行符
        np.loadtxt(txt, delimiter=nl)
    with pytest.raises(TypeError, match=msg):
        # 使用 numpy 的 loadtxt 函数加载文本数据，指定注释字符为当前换行符
        np.loadtxt(txt, comments=nl)
    with pytest.raises(TypeError, match=msg):
        # 使用 numpy 的 loadtxt 函数加载文本数据，指定引用字符为当前换行符
        np.loadtxt(txt, quotechar=nl)


# 使用 pytest.mark.parametrize 注册的参数化测试，测试用户指定的数据类型发现功能
@pytest.mark.parametrize(
    ("generic_data", "long_datum", "unitless_dtype", "expected_dtype"),
    [
        ("2012-03", "2013-01-15", "M8", "M8[D]"),  # 日期时间类型
        ("spam-a-lot", "tis_but_a_scratch", "U", "U17"),  # 字符串类型
    ],
)
@pytest.mark.parametrize("nrows", (10, 50000, 60000))  # 小于、等于、大于分块大小
def test_parametric_unit_discovery(
    generic_data, long_datum, unitless_dtype, expected_dtype, nrows
):
    """检查当用户指定无单位的日期时间时，从数据中正确识别单位（例如月、日、秒）。"""
    # 准备数据，包含重复数据和长日期时间数据
    data = [generic_data] * 50000 + [long_datum]
    expected = np.array(data, dtype=expected_dtype)

    # 准备文件对象路径
    txt = StringIO("\n".join(data))
    # 使用 numpy 的 loadtxt 函数加载文本数据，指定数据类型为无单位的日期时间类型
    a = np.loadtxt(txt, dtype=unitless_dtype)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)

    # 准备文件路径
    fd, fname = mkstemp()
    os.close(fd)
    with open(fname, "w") as fh:
        fh.write("\n".join(data))
    # 使用 numpy 的 loadtxt 函数加载文件中的文本数据，指定数据类型为无单位的日期时间类型
    a = np.loadtxt(fname, dtype=unitless_dtype)
    os.remove(fname)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)
def test_str_dtype_unit_discovery_with_converter():
    # 创建一个包含大量字符串的列表，其中包括一个特殊的字符串
    data = ["spam-a-lot"] * 60000 + ["XXXtis_but_a_scratch"]
    # 创建预期的 NumPy 数组，指定数据类型为 Unicode 字符串，长度为 17
    expected = np.array(["spam-a-lot"] * 60000 + ["tis_but_a_scratch"], dtype="U17")
    # 定义一个字符串转换器，去除字符串两端的 "XXX"
    conv = lambda s: s.strip("XXX")

    # 创建一个类似文件的路径，将数据作为文本流写入 StringIO 对象
    txt = StringIO("\n".join(data))
    # 使用 np.loadtxt 从文本流中加载数据，指定数据类型为 Unicode，应用字符串转换器
    a = np.loadtxt(txt, dtype="U", converters=conv)
    # 断言加载后的数组的数据类型与预期相符
    assert a.dtype == expected.dtype
    # 断言加载后的数组内容与预期相等
    assert_equal(a, expected)

    # 创建一个文件对象路径，写入数据并读取
    fd, fname = mkstemp()
    os.close(fd)
    with open(fname, "w") as fh:
        fh.write("\n".join(data))
    # 使用 np.loadtxt 从文件中加载数据，指定数据类型为 Unicode，应用字符串转换器
    a = np.loadtxt(fname, dtype="U", converters=conv)
    os.remove(fname)
    # 断言加载后的数组的数据类型与预期相符
    assert a.dtype == expected.dtype
    # 断言加载后的数组内容与预期相等
    assert_equal(a, expected)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_control_character_empty():
    # 使用 pytest 检测加载数据时的异常情况，期望抛出 TypeError
    with pytest.raises(TypeError, match="Text reading control character must"):
        np.loadtxt(StringIO("1 2 3"), delimiter="")
    with pytest.raises(TypeError, match="Text reading control character must"):
        np.loadtxt(StringIO("1 2 3"), quotechar="")
    # 使用 pytest 检测加载数据时的异常情况，期望抛出 ValueError
    with pytest.raises(ValueError, match="comments cannot be an empty string"):
        np.loadtxt(StringIO("1 2 3"), comments="")
    with pytest.raises(ValueError, match="comments cannot be an empty string"):
        np.loadtxt(StringIO("1 2 3"), comments=["#", ""])


def test_control_characters_as_bytes():
    """Byte control characters (comments, delimiter) are supported."""
    # 使用字节形式的控制字符（注释符号和分隔符）加载数据
    a = np.loadtxt(StringIO("#header\n1,2,3"), comments=b"#", delimiter=b",")
    # 断言加载后的数组内容与预期相等
    assert_equal(a, [1, 2, 3])


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_field_growing_cases():
    # 测试在每个字段仍然占据一个字符的情况下进行空字段的追加/增长
    res = np.loadtxt([""], delimiter=",", dtype=bytes)
    # 断言加载结果数组的长度为 0
    assert len(res) == 0

    # 循环测试不同长度的字段字符串，检查最终字段追加不会产生问题
    for i in range(1, 1024):
        res = np.loadtxt(["," * i], delimiter=",", dtype=bytes)
        # 断言加载结果数组的长度与预期相符
        assert len(res) == i+1
```