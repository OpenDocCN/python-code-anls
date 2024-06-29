# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_info.py`

```
# 从 io 模块导入 StringIO 类
from io import StringIO
# 导入正则表达式模块
import re
# 从 string 模块导入 ascii_uppercase 常量
from string import ascii_uppercase
# 导入 sys 模块
import sys
# 从 textwrap 模块导入 textwrap 函数
import textwrap

# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas.compat 模块导入 IS64 和 PYPY 常量
from pandas.compat import (
    IS64,
    PYPY,
)

# 从 pandas 模块导入多个类和函数
from pandas import (
    CategoricalIndex,
    DataFrame,
    MultiIndex,
    Series,
    date_range,
    option_context,
)
# 导入 pandas._testing 模块并重命名为 tm
import pandas._testing as tm


# 用作测试 fixture 的函数，返回具有重复列名的 DataFrame 对象
@pytest.fixture
def duplicate_columns_frame():
    """Dataframe with duplicate column names."""
    return DataFrame(
        np.random.default_rng(2).standard_normal((1500, 4)),
        columns=["a", "a", "b", "b"],
    )


# 测试函数：测试空 DataFrame 的 info 方法
def test_info_empty():
    # 创建一个空的 DataFrame 对象
    df = DataFrame()
    # 创建一个 StringIO 对象作为缓冲区
    buf = StringIO()
    # 对空的 DataFrame 调用 info 方法，将输出写入缓冲区
    df.info(buf=buf)
    # 从缓冲区中获取结果字符串
    result = buf.getvalue()
    # 预期的输出结果字符串
    expected = textwrap.dedent(
        """\
        <class 'pandas.DataFrame'>
        RangeIndex: 0 entries
        Empty DataFrame\n"""
    )
    # 断言结果与预期相符
    assert result == expected


# 测试函数：测试具有分类列的 DataFrame 的 info 方法
def test_info_categorical_column_smoke_test():
    # 设定随机数生成的种子值
    np.random.seed(2)
    n = 2500
    # 创建一个包含整数列和分类列的 DataFrame 对象
    df = DataFrame({"int64": np.random.randint(100, size=n)})
    df["category"] = Series(
        np.array(list("abcdefghij")).take(
            np.random.randint(0, 10, size=n)
        )
    ).astype("category")
    # 检查 DataFrame 是否有缺失值
    df.isna()
    # 创建一个 StringIO 对象作为缓冲区
    buf = StringIO()
    # 对 DataFrame 调用 info 方法，将输出写入缓冲区
    df.info(buf=buf)

    # 从原 DataFrame 中选取分类为 "d" 的子集
    df2 = df[df["category"] == "d"]
    # 创建一个新的 StringIO 对象作为缓冲区
    buf = StringIO()
    # 对子集 DataFrame 调用 info 方法，将输出写入缓冲区
    df2.info(buf=buf)


# 测试函数：使用参数化测试 fixture 进行 info 方法的测试
@pytest.mark.parametrize(
    "fixture_func_name",
    [
        "int_frame",
        "float_frame",
        "datetime_frame",
        "duplicate_columns_frame",
        "float_string_frame",
    ],
)
def test_info_smoke_test(fixture_func_name, request):
    # 使用 request 对象获取对应 fixture 的 DataFrame 对象
    frame = request.getfixturevalue(fixture_func_name)
    # 创建一个 StringIO 对象作为缓冲区
    buf = StringIO()
    # 对 DataFrame 调用 info 方法，将输出写入缓冲区
    frame.info(buf=buf)
    # 从缓冲区中获取结果字符串并按行分割为列表
    result = buf.getvalue().splitlines()
    # 断言结果列表的行数大于 10
    assert len(result) > 10

    # 创建一个新的 StringIO 对象作为缓冲区
    buf = StringIO()
    # 对 DataFrame 调用 info 方法（verbose 参数为 False），将输出写入缓冲区
    frame.info(buf=buf, verbose=False)


# 测试函数：测试具体 DataFrame 对象的 reindex 方法的 info 方法
def test_info_smoke_test2(float_frame):
    # 创建一个 StringIO 对象作为缓冲区
    buf = StringIO()
    # 使用 reindex 方法对 DataFrame 进行列重排，并调用 info 方法（verbose 参数为 False），将输出写入缓冲区
    float_frame.reindex(columns=["A"]).info(verbose=False, buf=buf)
    # 再次使用 reindex 方法对 DataFrame 进行列重排，并调用 info 方法（verbose 参数为 False），将输出写入缓冲区
    float_frame.reindex(columns=["A", "B"]).info(verbose=False, buf=buf)

    # 创建一个空的 DataFrame 对象，并调用 info 方法（verbose 参数为 False），将输出写入缓冲区
    DataFrame().info(buf=buf)


# 测试函数：测试不同的 display.max_info_columns 参数设置下的 info 方法
@pytest.mark.parametrize(
    "num_columns, max_info_columns, verbose",
    [
        (10, 100, True),
        (10, 11, True),
        (10, 10, True),
        (10, 9, False),
        (10, 1, False),
    ],
)
def test_info_default_verbose_selection(num_columns, max_info_columns, verbose):
    # 使用随机数生成种子值创建具有随机数值的 DataFrame 对象
    frame = DataFrame(np.random.standard_normal((5, num_columns)))
    # 使用 option_context 设置 display.max_info_columns 参数
    with option_context("display.max_info_columns", max_info_columns):
        # 创建一个 StringIO 对象作为缓冲区
        io_default = StringIO()
        # 对 DataFrame 调用 info 方法，将输出写入缓冲区
        frame.info(buf=io_default)
        # 从缓冲区中获取结果字符串
        result = io_default.getvalue()

        # 创建一个新的 StringIO 对象作为缓冲区
        io_explicit = StringIO()
        # 对 DataFrame 调用 info 方法（verbose 参数由参数化设置），将输出写入缓冲区
        frame.info(buf=io_explicit, verbose=verbose)
        # 获取显式设置 verbose 参数后的预期输出字符串
        expected = io_explicit.getvalue()

        # 断言结果与预期相符
        assert result == expected


# 测试函数：测试 info 方法在详细模式下的头部、分隔符和主体输出
def test_info_verbose_check_header_separator_body():
    # 创建一个 StringIO 对象作为缓冲区
    buf = StringIO()
    size = 1001
    start = 5
    # 创建一个 DataFrame 对象，其中包含从标准正态分布生成的随机数据，大小为 (3, size)
    frame = DataFrame(np.random.default_rng(2).standard_normal((3, size)))
    
    # 调用 DataFrame 的 info 方法，输出对象的详细信息到缓冲区 buf 中
    frame.info(verbose=True, buf=buf)
    
    # 从缓冲区 buf 中获取所有的内容，并存储在 res 变量中
    res = buf.getvalue()
    
    # 定义一个预期的表头字符串，用于后续断言检查
    header = " #     Column  Dtype  \n---    ------  -----  "
    
    # 确保 res 中包含预期的表头字符串 header
    assert header in res
    
    # 再次调用 DataFrame 的 info 方法，输出对象的详细信息到缓冲区 buf 中
    frame.info(verbose=True, buf=buf)
    
    # 将 buf 缓冲区的指针位置移动到文件开头（即偏移量为 0）
    buf.seek(0)
    
    # 逐行读取 buf 缓冲区中的内容，并存储在 lines 列表中
    lines = buf.readlines()
    
    # 确保 lines 列表中有内容，即至少有一行数据
    assert len(lines) > 0
    
    # 遍历 lines 列表中的每一行，同时获取行号 i 和行内容 line
    for i, line in enumerate(lines):
        # 检查行号 i 是否在指定的起始行 start 和结束行 start + size 之间
        if start <= i < start + size:
            # 构建期望的行号字符串，用于后续断言检查
            line_nr = f" {i - start} "
            # 确保当前行 line 以期望的行号字符串 line_nr 开头
            assert line.startswith(line_nr)
# 使用 pytest 模块的 mark.parametrize 装饰器，定义参数化测试用例
@pytest.mark.parametrize(
    "size, header_exp, separator_exp, first_line_exp, last_line_exp",
    [
        (
            4,
            " #   Column  Non-Null Count  Dtype  ",
            "---  ------  --------------  -----  ",
            " 0   0       3 non-null      float64",
            " 3   3       3 non-null      float64",
        ),
        (
            11,
            " #   Column  Non-Null Count  Dtype  ",
            "---  ------  --------------  -----  ",
            " 0   0       3 non-null      float64",
            " 10  10      3 non-null      float64",
        ),
        (
            101,
            " #    Column  Non-Null Count  Dtype  ",
            "---   ------  --------------  -----  ",
            " 0    0       3 non-null      float64",
            " 100  100     3 non-null      float64",
        ),
        (
            1001,
            " #     Column  Non-Null Count  Dtype  ",
            "---    ------  --------------  -----  ",
            " 0     0       3 non-null      float64",
            " 1000  1000    3 non-null      float64",
        ),
        (
            10001,
            " #      Column  Non-Null Count  Dtype  ",
            "---     ------  --------------  -----  ",
            " 0      0       3 non-null      float64",
            " 10000  10000   3 non-null      float64",
        ),
    ],
)
def test_info_verbose_with_counts_spacing(
    size, header_exp, separator_exp, first_line_exp, last_line_exp
):
    """Test header column, spacer, first line and last line in verbose mode."""
    # 使用 pandas 的 DataFrame 类，生成指定大小和随机数据的数据帧对象
    frame = DataFrame(np.random.default_rng(2).standard_normal((3, size)))
    # 使用 StringIO 创建缓冲区对象
    with StringIO() as buf:
        # 调用 DataFrame 的 info 方法，将信息输出到 buf 中
        frame.info(verbose=True, show_counts=True, buf=buf)
        # 获取 buf 中的所有行，并按换行符拆分成列表
        all_lines = buf.getvalue().splitlines()
    # 提取第 3 到倒数第 2 行之间的内容，这部分是表格的 header、separator 和数据行
    table = all_lines[3:-2]
    # 将表格分割为各个部分：header、separator、第一行、最后一行
    header, separator, first_line, *rest, last_line = table
    # 断言各部分与期望值相等
    assert header == header_exp
    assert separator == separator_exp
    assert first_line == first_line_exp
    assert last_line == last_line_exp


def test_info_memory():
    # 链接到 pandas GitHub 上的 issues 21056 的解释
    df = DataFrame({"a": Series([1, 2], dtype="i8")})
    buf = StringIO()
    # 使用 DataFrame 的 info 方法，将信息输出到 buf 中
    df.info(buf=buf)
    result = buf.getvalue()
    # 计算 DataFrame 内存使用量，并生成期望的输出结果
    bytes = float(df.memory_usage().sum())
    expected = textwrap.dedent(
        f"""\
    <class 'pandas.DataFrame'>
    RangeIndex: 2 entries, 0 to 1
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       2 non-null      int64
    dtypes: int64(1)
    memory usage: {bytes} bytes
    """
    )
    # 断言实际结果与期望结果相等
    assert result == expected


def test_info_wide():
    io = StringIO()
    # 使用 DataFrame 创建一个包含随机数据的数据帧对象
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 101)))
    # 使用 DataFrame 的 info 方法，将信息输出到 io 缓冲区中
    df.info(buf=io)

    io = StringIO()
    # 再次调用 info 方法，限制列的最大数量为 101，将信息输出到 io 缓冲区中
    df.info(buf=io, max_cols=101)
    result = io.getvalue()
    # 断言结果字符串按行拆分后的行数大于 100
    assert len(result.splitlines()) > 100
    
    # 将当前的结果字符串保存为预期结果
    expected = result
    
    # 设置上下文选项，使得在显示 DataFrame 信息时最大信息列数为 101
    with option_context("display.max_info_columns", 101):
        # 创建一个字符串缓冲区对象
        io = StringIO()
        
        # 将 DataFrame 的信息输出到缓冲区中
        df.info(buf=io)
        
        # 获取缓冲区中的内容作为结果字符串
        result = io.getvalue()
        
        # 断言当前的结果字符串与预期结果相等
        assert result == expected
def test_info_duplicate_columns_shows_correct_dtypes():
    # GH11761
    # 创建一个字符串 IO 对象
    io = StringIO()
    # 创建一个包含重复列名的 DataFrame
    frame = DataFrame([[1, 2.0]], columns=["a", "a"])
    # 将 DataFrame 的信息输出到 io 对象中
    frame.info(buf=io)
    # 将 io 对象中的内容按行分割为列表
    lines = io.getvalue().splitlines(True)
    # 断言第 6 行包含特定的信息
    assert " 0   a       1 non-null      int64  \n" == lines[5]
    # 断言第 7 行包含特定的信息
    assert " 1   a       1 non-null      float64\n" == lines[6]


def test_info_shows_column_dtypes():
    # 定义多种数据类型列表
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    # 创建一个空的数据字典
    data = {}
    n = 10
    # 使用不同数据类型填充数据字典
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    # 根据数据字典创建 DataFrame
    df = DataFrame(data)
    # 创建一个字符串 IO 对象
    buf = StringIO()
    # 将 DataFrame 的信息输出到 buf 对象中
    df.info(buf=buf)
    # 从 buf 对象获取输出的字符串
    res = buf.getvalue()
    # 定义信息表头
    header = (
        " #   Column  Non-Null Count  Dtype          \n"
        "---  ------  --------------  -----          "
    )
    # 断言信息表头在输出结果中
    assert header in res
    # 遍历数据类型列表，断言每种数据类型在输出结果中
    for i, dtype in enumerate(dtypes):
        name = f" {i:d}   {i:d}       {n:d} non-null     {dtype}"
        assert name in res


def test_info_max_cols():
    # 创建一个具有随机数据的 DataFrame
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
    # 针对不同的列数限制和详细输出选项进行测试
    for len_, verbose in [(5, None), (5, False), (12, True)]:
        # 当 max_info_columns 设置为 4 时
        with option_context("max_info_columns", 4):
            buf = StringIO()
            # 将 DataFrame 的信息输出到 buf 对象中，根据 verbose 设置决定是否详细输出
            df.info(buf=buf, verbose=verbose)
            # 获取输出字符串并断言行数是否符合预期
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

    # 针对不同的列数限制和详细输出选项进行测试
    for len_, verbose in [(12, None), (5, False), (12, True)]:
        # 当 max_info_columns 设置为 5 时
        with option_context("max_info_columns", 5):
            buf = StringIO()
            # 将 DataFrame 的信息输出到 buf 对象中，根据 verbose 设置决定是否详细输出
            df.info(buf=buf, verbose=verbose)
            # 获取输出字符串并断言行数是否符合预期
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

    # 针对不同的列数限制和 max_cols 设置进行测试
    for len_, max_cols in [(12, 5), (5, 4)]:
        # 当 max_info_columns 设置为 4 时
        with option_context("max_info_columns", 4):
            buf = StringIO()
            # 将 DataFrame 的信息输出到 buf 对象中，根据 max_cols 设置决定是否截断输出
            df.info(buf=buf, max_cols=max_cols)
            # 获取输出字符串并断言行数是否符合预期
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_

        # 当 max_info_columns 设置为 5 时
        with option_context("max_info_columns", 5):
            buf = StringIO()
            # 将 DataFrame 的信息输出到 buf 对象中，根据 max_cols 设置决定是否截断输出
            df.info(buf=buf, max_cols=max_cols)
            # 获取输出字符串并断言行数是否符合预期
            res = buf.getvalue()
            assert len(res.strip().split("\n")) == len_


def test_info_memory_usage():
    # 确保在最后一行显示内存使用情况
    # 定义多种数据类型列表
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    # 创建一个空的数据字典
    data = {}
    n = 10
    # 使用不同数据类型填充数据字典
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    # 根据数据字典创建 DataFrame
    df = DataFrame(data)
    # 创建一个字符串 IO 对象
    buf = StringIO()

    # 显示内存使用情况
    df.info(buf=buf, memory_usage=True)
    # 获取输出的所有行
    res = buf.getvalue().splitlines()
    # 断言最后一行包含内存使用信息
    assert "memory usage: " in res[-1]
    # 禁用内存使用信息，直接输出到缓冲区
    df.info(buf=buf, memory_usage=False)
    # 将缓冲区内容按行分割成列表
    res = buf.getvalue().splitlines()
    # 确保结果中不包含 "memory usage: "
    assert "memory usage: " not in res[-1]

    # 启用内存使用信息，并将结果输出到缓冲区
    df.info(buf=buf, memory_usage=True)
    # 将缓冲区内容按行分割成列表
    res = buf.getvalue().splitlines()

    # 确保最后一行匹配 "memory usage: 数字+ MB" 的模式
    assert re.match(r"memory usage: [^+]+\+", res[-1])

    # 对前五列的 DataFrame 启用内存使用信息，并将结果输出到缓冲区
    df.iloc[:, :5].info(buf=buf, memory_usage=True)
    # 将缓冲区内容按行分割成列表
    res = buf.getvalue().splitlines()

    # 由于排除了包含对象 dtype 的列，因此确保结果不匹配 "memory usage: 数字+ MB" 的模式
    assert not re.match(r"memory usage: [^+]+\+", res[-1])

    # 测试包含重复列的 DataFrame
    dtypes = ["int64", "int64", "int64", "float64"]
    data = {}
    n = 100
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.default_rng(2).integers(2, size=n).astype(dtype)
    # 创建 DataFrame，并指定列的数据类型
    df = DataFrame(data)
    df.columns = dtypes

    # 创建带有对象索引的 DataFrame，并启用内存使用信息输出到缓冲区
    df_with_object_index = DataFrame({"a": [1]}, index=["foo"])
    df_with_object_index.info(buf=buf, memory_usage=True)
    # 将缓冲区内容按行分割成列表
    res = buf.getvalue().splitlines()
    # 确保最后一行匹配 "memory usage: 数字+ MB" 的模式
    assert re.match(r"memory usage: [^+]+\+", res[-1])

    # 对带有对象索引的 DataFrame 启用深度内存使用信息输出到缓冲区
    df_with_object_index.info(buf=buf, memory_usage="deep")
    # 将缓冲区内容按行分割成列表
    res = buf.getvalue().splitlines()
    # 确保最后一行匹配 "memory usage: 数字+$" 的模式
    assert re.match(r"memory usage: [^+]+$", res[-1])

    # 确保 DataFrame 的实际大小与预期相符
    # (列数 * 行数 * 字节) + 索引大小
    df_size = df.memory_usage().sum()
    exp_size = len(dtypes) * n * 8 + df.index.nbytes
    assert df_size == exp_size

    # 确保 memory_usage 返回的列数与 DataFrame 中的列数相同
    size_df = np.size(df.columns.values) + 1  # index=True; 默认情况下包括索引
    assert size_df == np.size(df.memory_usage())

    # 确保仅对包含对象的列启用深度内存使用信息
    assert df.memory_usage().sum() == df.memory_usage(deep=True).sum()

    # 测试有效性
    DataFrame(1, index=["a"], columns=["A"]).memory_usage(index=True)
    DataFrame(1, index=["a"], columns=["A"]).index.nbytes
    df = DataFrame(
        data=1, index=MultiIndex.from_product([["a"], range(1000)]), columns=["A"]
    )
    df.index.nbytes
    df.memory_usage(index=True)
    df.index.values.nbytes

    # 计算 DataFrame 深度内存使用的总和，并确保大于 0
    mem = df.memory_usage(deep=True).sum()
    assert mem > 0
# 当条件 PYPY 为真时跳过测试，因为在 PyPy 上设置 deep=True 不会改变结果
@pytest.mark.skipif(PYPY, reason="on PyPy deep=True doesn't change result")
def test_info_memory_usage_deep_not_pypy():
    # 创建一个具有对象索引的 DataFrame
    df_with_object_index = DataFrame({"a": [1]}, index=["foo"])
    # 检查使用 deep=True 计算的内存使用量是否大于默认计算方式的总和
    assert (
        df_with_object_index.memory_usage(index=True, deep=True).sum()
        > df_with_object_index.memory_usage(index=True).sum()
    )

    # 创建一个包含对象类型数据的 DataFrame
    df_object = DataFrame({"a": ["a"]})
    # 检查使用 deep=True 计算的内存使用量是否大于默认计算方式的总和
    assert df_object.memory_usage(deep=True).sum() > df_object.memory_usage().sum()


# 当条件 PYPY 不为真时预期测试失败，因为在 PyPy 上设置 deep=True 不会改变结果
@pytest.mark.xfail(not PYPY, reason="on PyPy deep=True does not change result")
def test_info_memory_usage_deep_pypy():
    # 创建一个具有对象索引的 DataFrame
    df_with_object_index = DataFrame({"a": [1]}, index=["foo"])
    # 检查使用 deep=True 计算的内存使用量是否等于默认计算方式的总和
    assert (
        df_with_object_index.memory_usage(index=True, deep=True).sum()
        == df_with_object_index.memory_usage(index=True).sum()
    )

    # 创建一个包含对象类型数据的 DataFrame
    df_object = DataFrame({"a": ["a"]})
    # 检查使用 deep=True 计算的内存使用量是否等于默认计算方式的总和
    assert df_object.memory_usage(deep=True).sum() == df_object.memory_usage().sum()


# 当条件 PYPY 为真时跳过测试，因为 PyPy 上的 sys.getsizeof() 会设计上失败
@pytest.mark.skipif(PYPY, reason="PyPy getsizeof() fails by design")
def test_usage_via_getsizeof():
    # 创建一个具有多层索引的 DataFrame
    df = DataFrame(
        data=1, index=MultiIndex.from_product([["a"], range(1000)]), columns=["A"]
    )
    # 计算 DataFrame 深度内存使用量的总和
    mem = df.memory_usage(deep=True).sum()
    # sys.getsizeof 将调用 .memory_usage(deep=True)，并添加一些垃圾回收的开销
    diff = mem - sys.getsizeof(df)
    # 断言差值的绝对值小于 100
    assert abs(diff) < 100


def test_info_memory_usage_qualified():
    # 创建一个 DataFrame，并输出到指定缓冲区
    buf = StringIO()
    df = DataFrame(1, columns=list("ab"), index=[1, 2, 3])
    df.info(buf=buf)
    # 断言缓冲区中不包含 "+"
    assert "+" not in buf.getvalue()

    buf = StringIO()
    df = DataFrame(1, columns=list("ab"), index=list("ABC"))
    df.info(buf=buf)
    # 断言缓冲区中包含 "+"
    assert "+" in buf.getvalue()

    buf = StringIO()
    df = DataFrame(
        1, columns=list("ab"), index=MultiIndex.from_product([range(3), range(3)])
    )
    df.info(buf=buf)
    # 断言缓冲区中不包含 "+"
    assert "+" not in buf.getvalue()

    buf = StringIO()
    df = DataFrame(
        1, columns=list("ab"), index=MultiIndex.from_product([range(3), ["foo", "bar"]])
    )
    df.info(buf=buf)
    # 断言缓冲区中包含 "+"
    assert "+" in buf.getvalue()


def test_info_memory_usage_bug_on_multiindex():
    # GH 14308
    # 内存使用量检查不应该实体化 .values

    def memory_usage(f):
        return f.memory_usage(deep=True).sum()

    N = 100
    M = len(ascii_uppercase)
    # 创建一个具有多层索引的 DataFrame
    index = MultiIndex.from_product(
        [list(ascii_uppercase), date_range("20160101", periods=N)],
        names=["id", "date"],
    )
    df = DataFrame(
        {"value": np.random.default_rng(2).standard_normal(N * M)}, index=index
    )

    # 对 DataFrame 进行 "id" 列解堆
    unstacked = df.unstack("id")
    # 断言原始 DataFrame 和解堆后 DataFrame 的值的字节大小相等
    assert df.values.nbytes == unstacked.values.nbytes
    # 断言原始 DataFrame 的深度内存使用量大于解堆后 DataFrame 的深度内存使用量
    assert memory_usage(df) > memory_usage(unstacked)

    # 高上界
    # 断言解堆后 DataFrame 的深度内存使用量减去原始 DataFrame 的深度内存使用量小于 2000
    assert memory_usage(unstacked) - memory_usage(df) < 2000


def test_info_categorical():
    # GH14298
    # 创建一个具有分类索引的 DataFrame
    idx = CategoricalIndex(["a", "b"])
    df = DataFrame(np.zeros((2, 2)), index=idx, columns=idx)

    buf = StringIO()
    # 将 DataFrame 的信息输出到指定缓冲区
    df.info(buf=buf)
@pytest.mark.xfail(not IS64, reason="GH 36579: fail on 32-bit system")
# 标记为预期失败，条件为非64位系统，原因是GH 36579: 在32位系统上失败
def test_info_int_columns():
    # GH#37245
    # 创建一个DataFrame，包含两列和两行的数据框，列名分别为1和2，行索引为["A", "B"]
    df = DataFrame({1: [1, 2], 2: [2, 3]}, index=["A", "B"])
    # 创建一个StringIO对象作为缓冲区
    buf = StringIO()
    # 打印DataFrame的信息到缓冲区，包括列的非空计数
    df.info(show_counts=True, buf=buf)
    # 获取缓冲区的输出结果
    result = buf.getvalue()
    # 预期的输出结果，使用textwrap.dedent去除多余的空格和缩进
    expected = textwrap.dedent(
        """\
        <class 'pandas.DataFrame'>
        Index: 2 entries, A to B
        Data columns (total 2 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   1       2 non-null      int64
         1   2       2 non-null      int64
        dtypes: int64(2)
        memory usage: 48.0+ bytes
        """
    )
    # 断言实际结果与预期结果相同
    assert result == expected


def test_memory_usage_empty_no_warning():
    # GH#50066
    # 创建一个空的DataFrame，行索引为["a", "b"]
    df = DataFrame(index=["a", "b"])
    # 使用tm.assert_produces_warning(None)断言不会产生警告
    with tm.assert_produces_warning(None):
        # 获取DataFrame的内存使用情况
        result = df.memory_usage()
    # 预期的结果，创建一个Series，根据IS64的真假确定使用16或8作为值，索引为["Index"]
    expected = Series(16 if IS64 else 8, index=["Index"])
    # 使用tm.assert_series_equal进行Series的断言比较
    tm.assert_series_equal(result, expected)


@pytest.mark.single_cpu
# 标记为单CPU测试
def test_info_compute_numba():
    # GH#51922
    # 导入numba模块，如果导入失败则跳过该测试
    pytest.importorskip("numba")
    # 创建一个包含两个列表的DataFrame
    df = DataFrame([[1, 2], [3, 4]])

    with option_context("compute.use_numba", True):
        # 使用Numba进行计算时设置上下文，捕获输出到buf
        buf = StringIO()
        # 打印DataFrame的信息到buf
        df.info(buf=buf)
        # 获取buf的输出结果
        result = buf.getvalue()

    # 重新创建一个StringIO对象作为buf
    buf = StringIO()
    # 再次打印DataFrame的信息到buf
    df.info(buf=buf)
    # 获取buf的输出结果
    expected = buf.getvalue()
    # 断言两次打印的结果相同
    assert result == expected


@pytest.mark.parametrize(
    "row, columns, show_counts, result",
    [
        [20, 20, None, True],
        [20, 20, True, True],
        [20, 20, False, False],
        [5, 5, None, False],
        [5, 5, True, False],
        [5, 5, False, False],
    ],
)
# 参数化测试，测试不同的行、列、show_counts参数组合
def test_info_show_counts(row, columns, show_counts, result):
    # 将所有数据设置为1，创建一个DataFrame，列为0到9，行为0到9，并将第二列的数据类型设置为浮点型
    df = DataFrame(1, columns=range(10), index=range(10)).astype({1: "float"})
    # 将DataFrame的(1, 1)位置设置为NaN
    df.iloc[1, 1] = np.nan

    with option_context(
        "display.max_info_rows", row, "display.max_info_columns", columns
    ):
        # 使用StringIO作为缓冲区buf
        with StringIO() as buf:
            # 打印DataFrame的信息到buf，并指定show_counts参数
            df.info(buf=buf, show_counts=show_counts)
            # 断言buf中是否包含"non-null"，结果应与result相符
            assert ("non-null" in buf.getvalue()) is result
```