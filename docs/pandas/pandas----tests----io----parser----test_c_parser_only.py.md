# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_c_parser_only.py`

```
"""
Tests that apply specifically to the CParser. Unless specifically stated
as a CParser-specific issue, the goal is to eventually move as many of
these tests out of this module as soon as the Python parser can accept
further arguments when parsing.
"""

# 导入所需模块
from decimal import Decimal
from io import (
    BytesIO,
    StringIO,
    TextIOWrapper,
)
import mmap  # 导入 mmap 模块
import os    # 导入 os 模块
import tarfile  # 导入 tarfile 模块

import numpy as np  # 导入 numpy 库，并简化为 np
import pytest  # 导入 pytest 库

from pandas.compat import WASM  # 导入 WASM 从 pandas.compat 模块
from pandas.compat.numpy import np_version_gte1p24  # 导入 np_version_gte1p24 从 pandas.compat.numpy 模块
from pandas.errors import (  # 从 pandas.errors 导入 ParserError 和 ParserWarning
    ParserError,
    ParserWarning,
)
import pandas.util._test_decorators as td  # 导入 pandas.util._test_decorators，并简化为 td

from pandas import (  # 从 pandas 导入 DataFrame 和 concat
    DataFrame,
    concat,
)
import pandas._testing as tm  # 导入 pandas._testing，并简化为 tm


@pytest.mark.parametrize(  # 使用 pytest 的参数化测试
    "malformed",  # 参数名为 malformed
    ["1\r1\r1\r 1\r 1\r", "1\r1\r1\r 1\r 1\r11\r", "1\r1\r1\r 1\r 1\r11\r1\r"],  # 参数值为三个不同的字符串
    ids=["words pointer", "stream pointer", "lines pointer"],  # 对应参数化测试的标识符
)
def test_buffer_overflow(c_parser_only, malformed):  # 定义测试函数 test_buffer_overflow，接受参数 c_parser_only 和 malformed
    # see gh-9205: test certain malformed input files that cause
    # buffer overflows in tokenizer.c
    msg = "Buffer overflow caught - possible malformed input file."  # 定义错误消息字符串
    parser = c_parser_only  # 将参数 c_parser_only 赋值给 parser

    with pytest.raises(ParserError, match=msg):  # 使用 pytest 断言捕获 ParserError 异常，并匹配错误消息
        parser.read_csv(StringIO(malformed))  # 调用 parser 的 read_csv 方法读取 StringIO 格式的 malformed


def test_delim_whitespace_custom_terminator(c_parser_only):  # 定义测试函数 test_delim_whitespace_custom_terminator，接受参数 c_parser_only
    # See gh-12912
    data = "a b c~1 2 3~4 5 6~7 8 9"  # 定义包含特定分隔符和行终止符的数据字符串
    parser = c_parser_only  # 将参数 c_parser_only 赋值给 parser

    df = parser.read_csv(StringIO(data), lineterminator="~", sep=r"\s+")  # 调用 parser 的 read_csv 方法读取数据，指定行终止符和正则表达式作为分隔符
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])  # 预期的 DataFrame 结果
    tm.assert_frame_equal(df, expected)  # 使用 tm.assert_frame_equal 断言 df 和 expected 相等


def test_dtype_and_names_error(c_parser_only):  # 定义测试函数 test_dtype_and_names_error，接受参数 c_parser_only
    # see gh-8833: passing both dtype and names
    # resulting in an error reporting issue
    parser = c_parser_only  # 将参数 c_parser_only 赋值给 parser
    data = """
1.0 1
2.0 2
3.0 3
"""  # 定义数据字符串

    # base cases
    result = parser.read_csv(StringIO(data), sep=r"\s+", header=None)  # 调用 parser 的 read_csv 方法读取数据，指定正则表达式作为分隔符和不指定标题行
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]])  # 预期的 DataFrame 结果
    tm.assert_frame_equal(result, expected)  # 使用 tm.assert_frame_equal 断言 result 和 expected 相等

    result = parser.read_csv(StringIO(data), sep=r"\s+", header=None, names=["a", "b"])  # 调用 parser 的 read_csv 方法读取数据，指定正则表达式作为分隔符、不指定标题行，但指定列名
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=["a", "b"])  # 预期的 DataFrame 结果
    tm.assert_frame_equal(result, expected)  # 使用 tm.assert_frame_equal 断言 result 和 expected 相等

    # fallback casting
    result = parser.read_csv(  # 调用 parser 的 read_csv 方法读取数据，指定正则表达式作为分隔符、不指定标题行、指定列名和数据类型字典
        StringIO(data), sep=r"\s+", header=None, names=["a", "b"], dtype={"a": np.int32}
    )
    expected = DataFrame([[1, 1], [2, 2], [3, 3]], columns=["a", "b"])  # 预期的 DataFrame 结果
    expected["a"] = expected["a"].astype(np.int32)  # 将 DataFrame 中的列 'a' 转换为 np.int32 类型
    tm.assert_frame_equal(result, expected)  # 使用 tm.assert_frame_equal 断言 result 和 expected 相等

    data = """
1.0 1
nan 2
3.0 3
"""
    # fallback casting, but not castable
    warning = RuntimeWarning if np_version_gte1p24 else None  # 根据 numpy 版本决定警告类型
    # 如果不是在 WebAssembly 环境中（即WASM为False），则执行以下代码块
    if not WASM:
        # 使用 pytest 库断言捕获一个 ValueError 异常，并匹配错误信息 "cannot safely convert"
        with pytest.raises(ValueError, match="cannot safely convert"):
            # 使用 tm.assert_produces_warning 方法断言捕获一个警告
            # check_stacklevel=False 表示不检查调用堆栈层级
            with tm.assert_produces_warning(warning, check_stacklevel=False):
                # 调用 parser.read_csv 函数，解析给定的 CSV 数据
                # 使用 StringIO 将数据封装成文件对象
                # 设置分隔符为正则表达式 "\s+"，表示多个空白字符作为分隔符
                # 指定 header=None 表示数据没有头部信息
                # 指定 names=["a", "b"] 表示列的名称为 "a" 和 "b"
                # 指定 dtype={"a": np.int32} 表示列 "a" 的数据类型为 np.int32
                parser.read_csv(
                    StringIO(data),
                    sep=r"\s+",
                    header=None,
                    names=["a", "b"],
                    dtype={"a": np.int32},
                )
# 使用 pytest.mark.parametrize 装饰器，定义参数化测试函数 test_unsupported_dtype
@pytest.mark.parametrize(
    "match,kwargs",
    [
        # 第一种情况，指定了不支持的 dtype datetime64，建议使用 parse_dates 解析
        (
            (
                "the dtype datetime64 is not supported for parsing, "
                "pass this column using parse_dates instead"
            ),
            {"dtype": {"A": "datetime64", "B": "float64"}},
        ),
        # 第二种情况，同样指定了不支持的 dtype datetime64，但指定了 parse_dates 解析列
        (
            (
                "the dtype datetime64 is not supported for parsing, "
                "pass this column using parse_dates instead"
            ),
            {"dtype": {"A": "datetime64", "B": "float64"}, "parse_dates": ["B"]},
        ),
        # 第三种情况，指定了不支持的 dtype timedelta64
        (
            "the dtype timedelta64 is not supported for parsing",
            {"dtype": {"A": "timedelta64", "B": "float64"}},
        ),
        # 第四种情况，使用了特定的 dtype，但未被支持，根据 tm.ENDIAN 提示错误信息
        (
            f"the dtype {tm.ENDIAN}U8 is not supported for parsing",
            {"dtype": {"A": "U8"}},
        ),
    ],
    # 定义每组参数化测试的标识符
    ids=["dt64-0", "dt64-1", "td64", f"{tm.ENDIAN}U8"],
)
# 定义测试函数 test_unsupported_dtype，参数 c_parser_only 是测试的解析器实例
def test_unsupported_dtype(c_parser_only, match, kwargs):
    parser = c_parser_only
    # 创建一个 DataFrame 对象 df，包含随机生成的数据
    df = DataFrame(
        np.random.default_rng(2).random((5, 2)),
        columns=list("AB"),
        index=["1A", "1B", "1C", "1D", "1E"],
    )

    # 使用 tm.ensure_clean 上下文管理器，确保生成的文件路径 path 被清理
    with tm.ensure_clean("__unsupported_dtype__.csv") as path:
        # 将 DataFrame df 写入 CSV 文件
        df.to_csv(path)

        # 使用 pytest.raises 断言捕获 TypeError 异常，检查是否匹配预期的 match 字符串
        with pytest.raises(TypeError, match=match):
            # 调用 parser 的 read_csv 方法读取 CSV 文件，传入指定参数 kwargs
            parser.read_csv(path, index_col=0, **kwargs)


# 使用 td.skip_if_32bit 装饰器，跳过 32 位系统的测试
@td.skip_if_32bit
# 使用 pytest.mark.slow 装饰器，标记测试为慢速测试
# 参数化测试函数 test_precise_conversion，参数 num 取 1.0 到 2.0 之间的 21 个数值
@pytest.mark.parametrize("num", np.linspace(1.0, 2.0, num=21))
# 定义测试函数 test_precise_conversion，参数 c_parser_only 是测试的解析器实例，num 是测试数值
def test_precise_conversion(c_parser_only, num):
    parser = c_parser_only

    normal_errors = []
    precise_errors = []

    # 定义内部函数 error，用于计算误差
    def error(val: float, actual_val: Decimal) -> Decimal:
        return abs(Decimal(f"{val:.100}") - actual_val)

    # 设置文本数据 text，精确到 25 位小数
    text = f"a\n{num:.25}"

    # 使用 parser 的 read_csv 方法读取 text 文本数据，指定 float_precision 为 "legacy"
    normal_val = float(
        parser.read_csv(StringIO(text), float_precision="legacy")["a"][0]
    )
    # 使用 parser 的 read_csv 方法读取 text 文本数据，指定 float_precision 为 "high"
    precise_val = float(parser.read_csv(StringIO(text), float_precision="high")["a"][0])
    # 使用 parser 的 read_csv 方法读取 text 文本数据，指定 float_precision 为 "round_trip"
    roundtrip_val = float(
        parser.read_csv(StringIO(text), float_precision="round_trip")["a"][0]
    )
    # 将 text 文本数据转换为 Decimal 类型
    actual_val = Decimal(text[2:])

    # 计算误差并添加到 normal_errors 和 precise_errors 列表中
    normal_errors.append(error(normal_val, actual_val))
    precise_errors.append(error(precise_val, actual_val))

    # 断言 round-trip 操作后的值应与原始浮点数相等
    assert roundtrip_val == float(text[2:])

    # 断言高精度处理的误差之和不超过普通处理的误差之和
    assert sum(precise_errors) <= sum(normal_errors)
    # 断言高精度处理的最大误差不超过普通处理的最大误差
    assert max(precise_errors) <= max(normal_errors)


# 定义测试函数 test_usecols_dtypes，参数 c_parser_only 是测试的解析器实例
def test_usecols_dtypes(c_parser_only):
    parser = c_parser_only
    # 定义包含数据的字符串 data
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""

    # 使用 parser 的 read_csv 方法读取数据，并传入一系列参数：使用列 0 到 2，列名为 "a", "b", "c"，无头部行，
    # 将 "a" 列转换为字符串，"b" 列转换为整数，"c" 列转换为浮点数
    result = parser.read_csv(
        StringIO(data),
        usecols=(0, 1, 2),
        names=("a", "b", "c"),
        header=None,
        converters={"a": str},
        dtype={"b": int, "c": float},
    )
    # 调用 parser 对象的 read_csv 方法，解析 CSV 数据
    result2 = parser.read_csv(
        # 将数据作为字符串输入，使用 StringIO 将其转换为文件对象
        StringIO(data),
        # 仅使用列索引 0 和 2 的数据
        usecols=(0, 2),
        # 指定列名为 "a", "b", "c"
        names=("a", "b", "c"),
        # 没有头部行，因此指定 header=None
        header=None,
        # 将 "a" 列的数据转换为字符串
        converters={"a": str},
        # 指定 "b" 列为整数类型，"c" 列为浮点数类型
        dtype={"b": int, "c": float},
    )

    # 使用断言确保 result 对象的数据类型为 [object, int, float]
    assert (result.dtypes == [object, int, float]).all()
    # 使用断言确保 result2 对象的数据类型为 [object, float]
    assert (result2.dtypes == [object, float]).all()
# 测试禁用布尔解析功能
def test_disable_bool_parsing(c_parser_only):
    # 问题记录在 GitHub issue #2090

    # 使用 c_parser_only 创建解析器对象
    parser = c_parser_only

    # 定义包含布尔值的 CSV 数据
    data = """A,B,C
Yes,No,Yes
No,Yes,Yes
Yes,,Yes
No,No,No"""

    # 使用 parser 读取 CSV 数据，将所有列的数据类型设置为 object 类型
    result = parser.read_csv(StringIO(data), dtype=object)

    # 断言所有列的数据类型为 object
    assert (result.dtypes == object).all()

    # 再次使用 parser 读取 CSV 数据，禁用缺失值过滤器
    result = parser.read_csv(StringIO(data), dtype=object, na_filter=False)

    # 断言特定位置的值是否为空字符串
    assert result["B"][2] == ""


# 测试自定义行终止符
def test_custom_lineterminator(c_parser_only):
    # 使用 c_parser_only 创建解析器对象
    parser = c_parser_only

    # 定义包含自定义行终止符的 CSV 数据
    data = "a,b,c~1,2,3~4,5,6"

    # 使用 parser 读取 CSV 数据，指定自定义行终止符 "~"
    result = parser.read_csv(StringIO(data), lineterminator="~")

    # 期望的结果，将 "~" 替换为 "\n" 后再次使用 parser 读取
    expected = parser.read_csv(StringIO(data.replace("~", "\n")))

    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# 测试解析不规则的 CSV 数据
def test_parse_ragged_csv(c_parser_only):
    # 使用 c_parser_only 创建解析器对象
    parser = c_parser_only

    # 定义不规则的 CSV 数据，包含不同行数的列
    data = """1,2,3
1,2,3,4
1,2,3,4,5
1,2
1,2,3,4"""

    # 期望的数据，与 data 中的数据相比，添加了额外的逗号和空格
    nice_data = """1,2,3,,
1,2,3,4,
1,2,3,4,5
1,2,,,
1,2,3,4,"""

    # 使用 parser 读取 CSV 数据，指定不使用标题行，自定义列名
    result = parser.read_csv(
        StringIO(data), header=None, names=["a", "b", "c", "d", "e"]
    )

    # 使用 parser 读取期望的数据，也是不使用标题行，自定义列名
    expected = parser.read_csv(
        StringIO(nice_data), header=None, names=["a", "b", "c", "d", "e"]
    )

    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 太多的列，如果不小心可能导致段错误
    data = "1,2\n3,4,5"

    # 使用 parser 读取 CSV 数据，不使用标题行，自定义列名为 0 至 49
    result = parser.read_csv(StringIO(data), header=None, names=range(50))

    # 使用 parser 读取 CSV 数据，不使用标题行，自定义列名为 0 至 2，再重新索引到 0 至 49 列
    expected = parser.read_csv(StringIO(data), header=None, names=range(3)).reindex(
        columns=range(50)
    )

    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# 测试在引用情况下处理 CR 符号
def test_tokenize_CR_with_quoting(c_parser_only):
    # 问题记录在 GitHub issue #3453

    # 使用 c_parser_only 创建解析器对象
    parser = c_parser_only

    # 包含引号和 CR 符号的 CSV 数据
    data = ' a,b,c\r"a,b","e,d","f,f"'

    # 使用 parser 读取 CSV 数据，不使用标题行
    result = parser.read_csv(StringIO(data), header=None)

    # 期望的结果，将 CR 替换为换行符后再次使用 parser 读取
    expected = parser.read_csv(StringIO(data.replace("\r", "\n")), header=None)

    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 使用 parser 读取 CSV 数据，自动推断标题行
    result = parser.read_csv(StringIO(data))

    # 期望的结果，将 CR 替换为换行符后再次使用 parser 读取
    expected = parser.read_csv(StringIO(data.replace("\r", "\n")))

    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
@pytest.mark.parametrize("count", [3 * 2**n for n in range(6)])
def test_grow_boundary_at_cap(c_parser_only, count):
    # 问题记录在 GitHub issue #12494
    #
    # 错误的原因是 C 解析器在缓冲区容量达到极限时未增加缓冲区大小，
    # 这会在后续检查 CSV 流的 EOF 终止符时导致缓冲区溢出错误。
    # 发现当逗号数为 3 * 2^n 时会导致解析器崩溃

    # 使用 c_parser_only 创建解析器对象
    parser = c_parser_only

    # 使用 StringIO 创建包含 count 个逗号的字符串
    with StringIO("," * count) as s:
        # 期望的结果，创建一个列名为 "Unnamed: i" 的 DataFrame，i 从 0 到 count+1
        expected = DataFrame(columns=[f"Unnamed: {i}" for i in range(count + 1)])

        # 使用 parser 读取 CSV 数据
        df = parser.read_csv(s)

    # 断言读取的 DataFrame 是否与期望的 DataFrame 相等
    tm.assert_frame_equal(df, expected)


@pytest.mark.slow
@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_parse_trim_buffers(c_parser_only, encoding):
    # 此测试是 GitHub issue #13703 的修复的一部分。它尝试
    # 压力测试系统内存分配器，导致它移动流缓冲区，并让
    # 操作系统回收该区域，或让解析器的其他内存请求修改其内容
    # Set the parser to use only the C parser implementation (c_parser_only).
    parser = c_parser_only

    # Generate a large mixed-type CSV file on-the-fly (one record is
    # approx 1.5KiB).
    record_ = (
        """9999-9,99:99,,,,ZZ,ZZ,,,ZZZ-ZZZZ,.Z-ZZZZ,-9.99,,,9.99,Z"""
        """ZZZZZ,,-99,9,ZZZ-ZZZZ,ZZ-ZZZZ,,9.99,ZZZ-ZZZZZ,ZZZ-ZZZZZ,"""
        """ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,9"""
        """99,ZZZ-ZZZZ,,ZZ-ZZZZ,,,,,ZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,,,9,9,"""
        """9,9,99,99,999,999,ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,9,ZZ-ZZZZ,9."""
        """99,ZZ-ZZZZ,ZZ-ZZZZ,,,,ZZZZ,,,ZZ,ZZ,,,,,,,,,,,,,9,,,999."""
        """99,999.99,,,ZZZZZ,,,Z9,,,,,,,ZZZ,ZZZ,,,,,,,,,,,ZZZZZ,ZZ"""
        """ZZZ,ZZZ-ZZZZZZ,ZZZ-ZZZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZ"""
        """ZZ,,,999999,999999,ZZZ,ZZZ,,,ZZZ,ZZZ,999.99,999.99,,,,Z"""
        """ZZ-ZZZ,ZZZ-ZZZ,-9.99,-9.99,9,9,,99,,9.99,9.99,9,9,9.99,"""
        """9.99,,,,9.99,9.99,,99,,99,9.99,9.99,,,ZZZ,ZZZ,,999.99,,"""
        """999.99,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,ZZZZZ,ZZZZZ,ZZZ,ZZZ,9,9,"""
        """,,,,,ZZZ-ZZZZ,ZZZ999Z,,,999.99,,999.99,ZZZ-ZZZZ,,,9.999"""
        """,9.999,9.999,9.999,-9.999,-9.999,-9.999,-9.999,9.999,9."""
        """999,9.999,9.999,9.999,9.999,9.999,9.999,99999,ZZZ-ZZZZ,"""
        """,9.99,ZZZ,,,,,,,,ZZZ,,,,,9,,,,9,,,,,,,,,,ZZZ-ZZZZ,ZZZ-Z"""
        """ZZZ,,ZZZZZ,ZZZZZ,ZZZZZ,ZZZZZ,,,9.99,,ZZ-ZZZZ,ZZ-ZZZZ,ZZ"""
        """,999,,,,ZZ-ZZZZ,ZZZ,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,99.99,99.99"""
        """,,,9.99,9.99,9.99,9.99,ZZZ-ZZZZ,,,ZZZ-ZZZZZ,,,,,-9.99,-"""
        """9.99,-9.99,-9.99,,,,,,,,,ZZZ-ZZZZ,,9,9.99,9.99,99ZZ,,-9"""
        """.99,-9.99,ZZZ-ZZZZ,,,,,,,ZZZ-ZZZZ,9.99,9.99,9999,,,,,,,"""
        """,,,-9.9,Z/Z-ZZZZ,999.99,9.99,,999.99,ZZ-ZZZZ,ZZ-ZZZZ,9."""
        """99,9.99,9.99,9.99,9.99,9.99,,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZ"""
        """ZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ,ZZZ,ZZZ,ZZZ,9.99,,,-9.99,ZZ"""
        """-ZZZZ,-999.99,,-9999,,999.99,,,,999.99,99.99,,,ZZ-ZZZZZ"""
        """ZZZ,ZZ-ZZZZ-ZZZZZZZ,,,,ZZ-ZZ-ZZZZZZZZ,ZZZZZZZZ,ZZZ-ZZZZ"""
        """,9999,999.99,ZZZ-ZZZZ,-9.99,-9.99,ZZZ-ZZZZ,99:99:99,,99"""
        """,99,,9.99,,-99.99,,,,,,9.99,ZZZ-ZZZZ,-9.99,-9.99,9.99,9"""
        """.99,,ZZZ,,,,,,,ZZZ,ZZZ,,,,,"""
    )

    # Set the chunksize and number of lines to trigger `parser_trim_buffers`
    # when processing the CSV data.
    chunksize, n_lines = 128, 2 * 128 + 15
    csv_data = "\n".join([record_] * n_lines) + "\n"

    # We will use StringIO to load the CSV from this text buffer.
    # 使用 pd.read_csv() 逐块迭代文件，并最终读取一个非常小的剩余块。

    # 生成期望的输出：手动创建数据框架，通过逗号分隔并重复 `n_lines` 次的记录。
    row = tuple(val_ if val_ else np.nan for val_ in record_.split(","))
    expected = DataFrame(
        [row for _ in range(n_lines)], dtype=object, columns=None, index=None
    )

    # 在 `chunksize` 行的 CSV 文件中进行逐块迭代
    with parser.read_csv(
        StringIO(csv_data),
        header=None,
        dtype=object,
        chunksize=chunksize,
        encoding=encoding,
    ) as chunks_:
        # 将所有块按行拼接起来，忽略索引，形成最终的结果
        result = concat(chunks_, axis=0, ignore_index=True)

    # 如果没有段错误，检查数据是否损坏
    tm.assert_frame_equal(result, expected)
def test_internal_null_byte(c_parser_only):
    # see gh-14012
    #
    # The null byte ('\x00') should not be used as a
    # true line terminator, escape character, or comment
    # character, only as a placeholder to indicate that
    # none was specified.
    #
    # This test should be moved to test_common.py ONLY when
    # Python's csv class supports parsing '\x00'.
    # 用于指示 gh-14012 的测试，检验空字节 ('\x00') 在行终止符、转义字符、
    # 注释字符中的使用情况，仅作为未指定时的占位符。
    parser = c_parser_only

    names = ["a", "b", "c"]
    data = "1,2,3\n4,\x00,6\n7,8,9"
    expected = DataFrame([[1, 2.0, 3], [4, np.nan, 6], [7, 8, 9]], columns=names)

    # 使用解析器读取 CSV 数据，并验证解析结果是否与预期相符
    result = parser.read_csv(StringIO(data), names=names)
    tm.assert_frame_equal(result, expected)


def test_read_nrows_large(c_parser_only):
    # gh-7626 - Read only nrows of data in for large inputs (>262144b)
    # 用于测试 gh-7626，针对大输入（>262144b）只读取部分行数（nrows）的数据
    parser = c_parser_only
    header_narrow = "\t".join(["COL_HEADER_" + str(i) for i in range(10)]) + "\n"
    data_narrow = "\t".join(["somedatasomedatasomedata1" for _ in range(10)]) + "\n"
    header_wide = "\t".join(["COL_HEADER_" + str(i) for i in range(15)]) + "\n"
    data_wide = "\t".join(["somedatasomedatasomedata2" for _ in range(15)]) + "\n"
    test_input = header_narrow + data_narrow * 1050 + header_wide + data_wide * 2

    # 使用解析器读取 CSV 数据，限定只读取前 1010 行数据
    df = parser.read_csv(StringIO(test_input), sep="\t", nrows=1010)

    # 验证读取的 DataFrame 是否符合预期的大小
    assert df.size == 1010 * 10


def test_float_precision_round_trip_with_text(c_parser_only):
    # see gh-15140
    # 用于验证 gh-15140，测试在文本中使用浮点精度 "round_trip" 的情况
    parser = c_parser_only
    df = parser.read_csv(StringIO("a"), header=None, float_precision="round_trip")
    tm.assert_frame_equal(df, DataFrame({0: ["a"]}))


def test_large_difference_in_columns(c_parser_only):
    # see gh-14125
    # 用于查看 gh-14125，测试在列数差异大的情况下的解析行为
    parser = c_parser_only

    count = 10000
    large_row = ("X," * count)[:-1] + "\n"
    normal_row = "XXXXXX XXXXXX,111111111111111\n"
    test_input = (large_row + normal_row * 6)[:-1]

    # 使用解析器读取 CSV 数据，只保留第一列数据
    result = parser.read_csv(StringIO(test_input), header=None, usecols=[0])
    rows = test_input.split("\n")

    # 生成预期的 DataFrame，只包含每行的第一列数据
    expected = DataFrame([row.split(",")[0] for row in rows])
    tm.assert_frame_equal(result, expected)


def test_data_after_quote(c_parser_only):
    # see gh-15910
    # 用于观察 gh-15910，测试在引号后有数据的情况
    parser = c_parser_only

    data = 'a\n1\n"b"a'
    result = parser.read_csv(StringIO(data))

    # 预期的 DataFrame 包含列名为 'a' 的数据
    expected = DataFrame({"a": ["1", "ba"]})
    tm.assert_frame_equal(result, expected)


def test_comment_whitespace_delimited(c_parser_only):
    parser = c_parser_only
    test_input = """\
1 2
2 2 3
3 2 3 # 3 fields
4 2 3# 3 fields
5 2 # 2 fields
6 2# 2 fields
7 # 1 field, NaN
8# 1 field, NaN
9 2 3 # skipped line
# comment"""
    
    # 验证在有注释存在的情况下解析器的行为，应该跳过特定行并发出警告
    with tm.assert_produces_warning(
        ParserWarning, match="Skipping line", check_stacklevel=False
    ):
        df = parser.read_csv(
            StringIO(test_input),
            comment="#",
            header=None,
            delimiter="\\s+",
            skiprows=0,
            on_bad_lines="warn",
        )
    
    # 生成预期的 DataFrame，只包含前几行符合条件的数据
    expected = DataFrame([[1, 2], [5, 2], [6, 2], [7, np.nan], [8, np.nan]])
    tm.assert_frame_equal(df, expected)
def test_file_like_no_next(c_parser_only):
    # gh-16530: the file-like need not have a "next" or "__next__"
    # attribute despite having an "__iter__" attribute.
    #
    # NOTE: This is only true for the C engine, not Python engine.
    # 定义一个类 NoNextBuffer 继承自 StringIO，用于测试文件类对象是否需要 "next" 或 "__next__" 方法。
    class NoNextBuffer(StringIO):
        # 定义 __next__ 方法，抛出属性错误异常，模拟没有 "next" 方法的情况
        def __next__(self):
            raise AttributeError("No next method")
        
        # 将 next 属性设置为 __next__ 方法，保证与 Python 2 兼容性
        next = __next__

    # 获取传入的 c_parser_only 对象，用于解析 CSV 数据
    parser = c_parser_only
    # 准备测试数据
    data = "a\n1"

    # 准备预期结果，创建 DataFrame 对象
    expected = DataFrame({"a": [1]})
    # 调用解析器的 read_csv 方法，传入 NoNextBuffer 类型的数据
    result = parser.read_csv(NoNextBuffer(data))

    # 使用测试工具函数验证结果与预期是否相符
    tm.assert_frame_equal(result, expected)


def test_buffer_rd_bytes_bad_unicode(c_parser_only):
    # see gh-22748
    # 创建一个 BytesIO 对象 t，包含一个无效的 UTF-8 字符串
    t = BytesIO(b"\xb0")
    # 使用 TextIOWrapper 对象对 t 进行包装，指定编码为 UTF-8，并设置错误处理方式为 surrogateescape
    t = TextIOWrapper(t, encoding="UTF-8", errors="surrogateescape")
    # 准备异常消息字符串
    msg = "'utf-8' codec can't encode character"
    # 使用 pytest 的 raises 断言，验证读取 CSV 文件时是否会抛出 UnicodeError 异常，并检查异常消息
    with pytest.raises(UnicodeError, match=msg):
        c_parser_only.read_csv(t, encoding="UTF-8")


@pytest.mark.parametrize("tar_suffix", [".tar", ".tar.gz"])
def test_read_tarfile(c_parser_only, csv_dir_path, tar_suffix):
    # see gh-16530
    #
    # Unfortunately, Python's CSV library can't handle
    # tarfile objects (expects string, not bytes when
    # iterating through a file-like).
    # 获取传入的 c_parser_only 对象，用于解析 CSV 数据
    parser = c_parser_only
    # 构建完整的 tar 文件路径
    tar_path = os.path.join(csv_dir_path, "tar_csv" + tar_suffix)

    # 打开 tar 文件对象，并提取其中的 "tar_data.csv" 文件内容
    with tarfile.open(tar_path, "r") as tar:
        data_file = tar.extractfile("tar_data.csv")

        # 使用解析器读取 tar 文件中的 CSV 数据
        out = parser.read_csv(data_file)
        # 准备预期结果，创建 DataFrame 对象
        expected = DataFrame({"a": [1]})
        # 使用测试工具函数验证结果与预期是否相符
        tm.assert_frame_equal(out, expected)


def test_chunk_whitespace_on_boundary(c_parser_only):
    # see gh-9735: this issue is C parser-specific (bug when
    # parsing whitespace and characters at chunk boundary)
    #
    # This test case has a field too large for the Python parser / CSV library.
    # 获取传入的 c_parser_only 对象，用于解析 CSV 数据
    parser = c_parser_only

    # 准备测试数据，分两个块：chunk1 和 chunk2
    chunk1 = "a" * (1024 * 256 - 2) + "\na"
    chunk2 = "\n a"
    # 使用解析器读取 CSV 数据，不指定标题行
    result = parser.read_csv(StringIO(chunk1 + chunk2), header=None)

    # 准备预期结果，创建包含字符串列表的 DataFrame 对象
    expected = DataFrame(["a" * (1024 * 256 - 2), "a", " a"])
    # 使用测试工具函数验证结果与预期是否相符
    tm.assert_frame_equal(result, expected)


@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
def test_file_handles_mmap(c_parser_only, csv1):
    # gh-14418
    #
    # Don't close user provided file handles.
    # 获取传入的 c_parser_only 对象，用于解析 CSV 数据
    parser = c_parser_only

    # 打开指定路径的 CSV 文件，并使用 mmap 映射文件到内存
    with open(csv1, encoding="utf-8") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            # 使用解析器读取 mmap 对象中的 CSV 数据
            parser.read_csv(m)
            # 使用断言确认 mmap 对象未被关闭
            assert not m.closed


def test_file_binary_mode(c_parser_only):
    # see gh-23779
    # 获取传入的 c_parser_only 对象，用于解析 CSV 数据
    parser = c_parser_only
    # 准备预期结果，创建包含列表的 DataFrame 对象
    expected = DataFrame([[1, 2, 3], [4, 5, 6]])

    # 使用测试工具函数，在临时路径下创建并写入 CSV 数据
    with tm.ensure_clean() as path:
        with open(path, "w", encoding="utf-8") as f:
            f.write("1,2,3\n4,5,6")

        # 打开临时文件，以二进制模式读取数据
        with open(path, "rb") as f:
            # 使用解析器读取二进制文件中的 CSV 数据，不指定标题行
            result = parser.read_csv(f, header=None)
            # 使用测试工具函数验证结果与预期是否相符
            tm.assert_frame_equal(result, expected)


def test_unix_style_breaks(c_parser_only):
    # GH 11020
    # 获取传入的 c_parser_only 对象，用于解析 CSV 数据
    parser = c_parser_only
    # 使用 tm.ensure_clean() 上下文管理器确保在代码块结束时清理临时文件
    with tm.ensure_clean() as path:
        # 使用指定路径创建一个新文件，以写入模式打开，使用 UTF-8 编码，'\n' 作为换行符
        with open(path, "w", newline="\n", encoding="utf-8") as f:
            # 向文件写入初始内容 "blah\n\ncol_1,col_2,col_3\n\n"
            f.write("blah\n\ncol_1,col_2,col_3\n\n")
        # 使用 parser.read_csv() 方法读取 CSV 文件，跳过前两行，使用 UTF-8 编码和 'c' 引擎
        result = parser.read_csv(path, skiprows=2, encoding="utf-8", engine="c")
    # 创建预期的 DataFrame，包含列名为 "col_1", "col_2", "col_3" 的空数据框架
    expected = DataFrame(columns=["col_1", "col_2", "col_3"])
    # 使用 tm.assert_frame_equal() 断言检查 result 是否与 expected 相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
@pytest.mark.parametrize(
    "data,thousands,decimal",
    [
        (
            """A|B|C
1|2,334.01|5
10|13|10.
""",
            ",",
            ".",
        ),
        (
            """A|B|C
1|2.334,01|5
10|13|10,
""",
            ".",
            ",",
        ),
    ],
)
# 使用pytest的@parametrize装饰器，为test_1000_sep_with_decimal测试用例定义多组参数组合
def test_1000_sep_with_decimal(
    c_parser_only, data, thousands, decimal, float_precision
):
    # 获取解析器对象
    parser = c_parser_only
    # 期望的DataFrame结果
    expected = DataFrame({"A": [1, 10], "B": [2334.01, 13], "C": [5, 10.0]})
    
    # 调用解析器的read_csv方法解析CSV数据，并设置分隔符、千分位分隔符、小数点符号和浮点数精度
    result = parser.read_csv(
        StringIO(data),
        sep="|",
        thousands=thousands,
        decimal=decimal,
        float_precision=float_precision,
    )
    # 使用assert_frame_equal方法断言结果DataFrame与期望的DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_float_precision_options(c_parser_only):
    # GH 17154, 36228
    # 获取解析器对象
    parser = c_parser_only
    # 定义包含浮点数的字符串
    s = "foo\n243.164\n"
    # 使用默认浮点数精度解析CSV数据
    df = parser.read_csv(StringIO(s))
    # 使用高精度选项解析CSV数据
    df2 = parser.read_csv(StringIO(s), float_precision="high")

    # 断言两个DataFrame对象相等
    tm.assert_frame_equal(df, df2)

    # 使用遗留精度选项解析CSV数据
    df3 = parser.read_csv(StringIO(s), float_precision="legacy")

    # 断言第一个DataFrame的第一个元素不等于第三个DataFrame的第一个元素
    assert not df.iloc[0, 0] == df3.iloc[0, 0]

    # 定义错误的浮点数精度选项
    msg = "Unrecognized float_precision option: junk"

    # 使用pytest.raises断言解析CSV数据时引发值错误，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(s), float_precision="junk")
```