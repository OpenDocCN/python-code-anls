# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_to_html.py`

```
from datetime import datetime  # 导入datetime模块中的datetime类，用于处理日期和时间
from io import StringIO  # 导入StringIO类，用于在内存中读写字符串
import itertools  # 导入itertools模块，提供了用于创建和操作迭代器的函数
import re  # 导入re模块，用于支持正则表达式的操作
import textwrap  # 导入textwrap模块，用于格式化文本段落

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

import pandas as pd  # 导入Pandas库，用于数据分析和处理
from pandas import (  # 从Pandas库中导入特定的子模块和函数
    DataFrame,  # 用于创建和操作二维表格数据
    Index,  # 用于表示和操作索引
    MultiIndex,  # 用于表示和操作多级索引
    get_option,  # 获取Pandas库的全局选项值
    option_context,  # 为执行某个上下文设置Pandas库的选项
)

import pandas.io.formats.format as fmt  # 导入Pandas格式化模块中的format对象

lorem_ipsum = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex "
    "ea commodo consequat. Duis aute irure dolor in reprehenderit in "
    "voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur "
    "sint occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum."
)  # 定义一个长字符串，用于文本测试和示例

def expected_html(datapath, name):
    """
    Read HTML file from formats data directory.

    Parameters
    ----------
    datapath : pytest fixture
        The datapath fixture injected into a test by pytest.
    name : str
        The name of the HTML file without the suffix.

    Returns
    -------
    str : contents of HTML file.
    """
    filename = ".".join([name, "html"])  # 根据给定的name参数生成HTML文件名
    filepath = datapath("io", "formats", "data", "html", filename)  # 拼接文件路径
    with open(filepath, encoding="utf-8") as f:  # 使用utf-8编码打开文件
        html = f.read()  # 读取文件内容
    return html.rstrip()  # 返回去除右侧空白的HTML内容

@pytest.fixture(params=["mixed", "empty"])
def biggie_df_fixture(request):
    """Fixture for a big mixed Dataframe and an empty Dataframe"""
    if request.param == "mixed":
        df = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(200),  # 创建一个带有正态分布数据的DataFrame列
                "B": Index([f"{i}?!" for i in range(200)]),  # 创建一个带有自定义索引的DataFrame列
            },
            index=np.arange(200),  # 设置DataFrame的整数索引
        )
        df.loc[:20, "A"] = np.nan  # 将前20行'A'列设置为NaN
        df.loc[:20, "B"] = np.nan  # 将前20行'B'列设置为NaN
        return df  # 返回生成的DataFrame对象
    elif request.param == "empty":
        df = DataFrame(index=np.arange(200))  # 创建一个空的DataFrame对象
        return df  # 返回生成的DataFrame对象

@pytest.fixture(params=fmt.VALID_JUSTIFY_PARAMETERS)
def justify(request):
    return request.param  # 返回从fmt.VALID_JUSTIFY_PARAMETERS参数列表中选择的参数值

@pytest.mark.parametrize("col_space", [30, 50])
def test_to_html_with_col_space(col_space):
    df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))  # 创建一个具有随机数据的DataFrame对象
    # 检查col_space是否影响HTML生成，并非常脆弱地进行检查。
    result = df.to_html(col_space=col_space)  # 使用指定的col_space生成HTML表格
    hdrs = [x for x in result.split(r"\n") if re.search(r"<th[>\s]", x)]  # 检查生成的HTML表头
    assert len(hdrs) > 0  # 断言确保至少有一个表头存在
    for h in hdrs:
        assert "min-width" in h  # 断言确保表头中包含'min-width'样式设置
        assert str(col_space) in h  # 断言确保表头中包含指定的col_space值

def test_to_html_with_column_specific_col_space_raises():
    df = DataFrame(
        np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
    )  # 创建一个具有指定列名和随机数据的DataFrame对象

    msg = (
        "Col_space length\\(\\d+\\) should match "
        "DataFrame number of columns\\(\\d+\\)"
    )  # 设置异常消息正则表达式模式
    with pytest.raises(ValueError, match=msg):  # 断言捕获指定异常消息
        df.to_html(col_space=[30, 40])  # 尝试使用不匹配列数的col_space参数调用to_html方法

    with pytest.raises(ValueError, match=msg):  # 断言捕获指定异常消息
        df.to_html(col_space=[30, 40, 50, 60])  # 尝试使用不匹配列数的col_space参数调用to_html方法
    # 设置错误消息文本
    msg = "unknown column"
    # 使用 pytest 的上下文管理器，检查是否会抛出 ValueError 异常，并且异常消息匹配预期的 "unknown column"
    with pytest.raises(ValueError, match=msg):
        # 调用 DataFrame 的 to_html 方法，传入一个非法的 col_space 参数，预期会抛出异常
        df.to_html(col_space={"a": "foo", "b": 23, "d": 34})
def test_to_html_with_column_specific_col_space():
    # 创建一个包含随机数据的 DataFrame，有三行三列
    df = DataFrame(
        np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
    )

    # 使用指定的列宽选项将 DataFrame 转换为 HTML，并检查生成的表头部分
    result = df.to_html(col_space={"a": "2em", "b": 23})
    # 提取包含<th>标签的行，检查每列的最小宽度设置
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    assert 'min-width: 2em;">a</th>' in hdrs[1]
    assert 'min-width: 23px;">b</th>' in hdrs[2]
    assert "<th>c</th>" in hdrs[3]

    # 使用列表形式的列宽选项再次将 DataFrame 转换为 HTML
    result = df.to_html(col_space=["1em", 2, 3])
    # 提取包含<th>标签的行，检查每列的最小宽度设置
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    assert 'min-width: 1em;">a</th>' in hdrs[1]
    assert 'min-width: 2px;">b</th>' in hdrs[2]
    assert 'min-width: 3px;">c</th>' in hdrs[3]


def test_to_html_with_empty_string_label():
    # 检查空字符串标签在 to_html 方法中的处理（GH 3547）
    data = {"c1": ["a", "b"], "c2": ["a", ""], "data": [1, 2]}
    df = DataFrame(data).set_index(["c1", "c2"])
    result = df.to_html()
    # 断言结果中不包含 rowspan 属性，确保空字符串标签被正确处理
    assert "rowspan" not in result


@pytest.mark.parametrize(
    "df_data,expected",
    [
        ({"\u03c3": np.arange(10.0)}, "unicode_1"),
        ({"A": ["\u03c3"]}, "unicode_2"),
    ],
)
def test_to_html_unicode(df_data, expected, datapath):
    # 测试 DataFrame 包含 Unicode 数据时的 to_html 方法
    df = DataFrame(df_data)
    # 从预期的 HTML 文件中加载期望的输出
    expected = expected_html(datapath, expected)
    result = df.to_html()
    assert result == expected


def test_to_html_encoding(float_frame, tmp_path):
    # GH 28663，测试 DataFrame 到 HTML 的编码设置
    path = tmp_path / "test.html"
    float_frame.to_html(path, encoding="gbk")
    with open(str(path), encoding="gbk") as f:
        assert float_frame.to_html() == f.read()


def test_to_html_decimal(datapath):
    # GH 12031，测试使用特定小数分隔符转换 DataFrame 到 HTML
    df = DataFrame({"A": [6.0, 3.1, 2.2]})
    result = df.to_html(decimal=",")
    # 从预期的 HTML 文件中加载期望的输出
    expected = expected_html(datapath, "gh12031_expected_output")
    assert result == expected


@pytest.mark.parametrize(
    "kwargs,string,expected",
    [
        ({}, "<type 'str'>", "escaped"),
        ({"escape": False}, "<b>bold</b>", "escape_disabled"),
    ],
)
def test_to_html_escaped(kwargs, string, expected, datapath):
    # 测试包含特殊字符和 HTML 转义的 DataFrame 到 HTML 方法
    a = "str<ing1 &amp;"
    b = "stri>ng2 &amp;"
    test_dict = {"co<l1": {a: string, b: string}, "co>l2": {a: string, b: string}}
    result = DataFrame(test_dict).to_html(**kwargs)
    # 从预期的 HTML 文件中加载期望的输出
    expected = expected_html(datapath, expected)
    assert result == expected


@pytest.mark.parametrize("index_is_named", [True, False])
def test_to_html_multiindex_index_false(index_is_named, datapath):
    # GH 8452，测试多级索引在不显示索引的情况下转换为 HTML
    df = DataFrame(
        {"a": range(2), "b": range(3, 5), "c": range(5, 7), "d": range(3, 5)}
    )
    df.columns = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    if index_is_named:
        df.index = Index(df.index.values, name="idx")
    result = df.to_html(index=False)
    # 从预期的 HTML 文件中加载期望的输出
    expected = expected_html(datapath, "gh8452_expected_output")
    assert result == expected
    # 列表包含四个元组，每个元组包含一个布尔值和一个字符串
    [
        # 第一个元组，布尔值为 False，字符串为 "multiindex_sparsify_false_multi_sparse_1"
        (False, "multiindex_sparsify_false_multi_sparse_1"),
        # 第二个元组，布尔值为 False，字符串为 "multiindex_sparsify_false_multi_sparse_2"
        (False, "multiindex_sparsify_false_multi_sparse_2"),
        # 第三个元组，布尔值为 True，字符串为 "multiindex_sparsify_1"
        (True, "multiindex_sparsify_1"),
        # 第四个元组，布尔值为 True，字符串为 "multiindex_sparsify_2"
        (True, "multiindex_sparsify_2"),
    ],
# 定义一个测试函数，测试多重索引情况下稀疏化处理生成 HTML 表格的功能
def test_to_html_multiindex_sparsify(multi_sparse, expected, datapath):
    # 创建一个多重索引对象，包含两级索引
    index = MultiIndex.from_arrays([[0, 0, 1, 1], [0, 1, 0, 1]], names=["foo", None])
    # 创建一个数据框，使用上述索引作为行索引，包含四行数据
    df = DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]], index=index)
    # 如果预期结果字符串以 "2" 结尾，则将数据框的列名设置为索引的每隔两个元素
    if expected.endswith("2"):
        df.columns = index[::2]
    # 设置显示选项 "display.multi_sparse"，并生成数据框 df 的 HTML 表格
    with option_context("display.multi_sparse", multi_sparse):
        result = df.to_html()
    # 使用预期输出数据的 HTML 格式
    expected = expected_html(datapath, expected)
    # 断言生成的 HTML 表格与预期的 HTML 结果相等
    assert result == expected


# 使用参数化测试框架标记此测试函数，测试在不同参数下生成 HTML 表格的情况
@pytest.mark.parametrize(
    "max_rows,expected",
    [
        (60, "gh14882_expected_output_1"),
        # 测试在截断时 "..." 是否出现在中间层
        (56, "gh14882_expected_output_2"),
    ],
)
def test_to_html_multiindex_odd_even_truncate(max_rows, expected, datapath):
    # GH 14882 - 处理奇数长度数据框截断的问题
    # 创建一个多重索引对象，包含三个级别的完全组合
    index = MultiIndex.from_product(
        [[100, 200, 300], [10, 20, 30], [1, 2, 3, 4, 5, 6, 7]], names=["a", "b", "c"]
    )
    # 创建一个数据框，包含一个列 "n"，索引为上述多重索引
    df = DataFrame({"n": range(len(index))}, index=index)
    # 生成数据框 df 的 HTML 表格，限制最大行数为 max_rows
    result = df.to_html(max_rows=max_rows)
    # 使用预期输出数据的 HTML 格式
    expected = expected_html(datapath, expected)
    # 断言生成的 HTML 表格与预期的 HTML 结果相等
    assert result == expected


# 使用参数化测试框架标记此测试函数，测试在不同格式化器下生成 HTML 表格的情况
@pytest.mark.parametrize(
    "df,formatters,expected",
    [
        (
            # 创建一个数据框，包含整数、浮点数、可空整数、字符串、布尔值、类别、对象类型数据
            DataFrame(
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                columns=Index(["foo", None], dtype=object),
                index=np.arange(4),
            ),
            # 设置格式化器 "__index__" 以将索引值转换为 "abcd" 中的字符
            {"__index__": lambda x: "abcd"[x]},
            "index_formatter",
        ),
        (
            # 创建一个数据框，包含日期时间数据，格式为 "%Y-%m"
            DataFrame({"months": [datetime(2016, 1, 1), datetime(2016, 2, 2)]}),
            # 设置格式化器 "months" 以将日期时间格式化为 "%Y-%m"
            {"months": lambda x: x.strftime("%Y-%m")},
            "datetime64_monthformatter",
        ),
        (
            # 创建一个数据框，包含时间数据，格式为 "%H:%M"
            DataFrame(
                {
                    "hod": pd.to_datetime(
                        ["10:10:10.100", "12:12:12.120"], format="%H:%M:%S.%f"
                    )
                }
            ),
            # 设置格式化器 "hod" 以将时间格式化为 "%H:%M"
            {"hod": lambda x: x.strftime("%H:%M")},
            "datetime64_hourformatter",
        ),
        (
            # 创建一个数据框，包含各种数据类型的列，并应用七个相同的格式化器
            DataFrame(
                {
                    "i": pd.Series([1, 2], dtype="int64"),
                    "f": pd.Series([1, 2], dtype="float64"),
                    "I": pd.Series([1, 2], dtype="Int64"),
                    "s": pd.Series([1, 2], dtype="string"),
                    "b": pd.Series([True, False], dtype="boolean"),
                    "c": pd.Series(["a", "b"], dtype=pd.CategoricalDtype(["a", "b"])),
                    "o": pd.Series([1, "2"], dtype=object),
                }
            ),
            # 设置七个格式化器，每个都将输入 x 格式化为字符串 "formatted"
            [lambda x: "formatted"] * 7,
            "various_dtypes_formatted",
        ),
    ],
)
def test_to_html_formatters(df, formatters, expected, datapath):
    # 使用预期输出数据的 HTML 格式
    expected = expected_html(datapath, expected)
    # 生成数据框 df 的 HTML 表格，应用指定的格式化器
    result = df.to_html(formatters=formatters)
    # 断言生成的 HTML 表格与预期的 HTML 结果相等
    assert result == expected


# 定义一个回归测试函数，测试 GH6098 的修复
def test_to_html_regression_GH6098():
    # 创建一个 DataFrame 对象，包含四列数据：clé1, clé2, données1, données2
    df = DataFrame(
        {
            "clé1": ["a", "a", "b", "b", "a"],
            "clé2": ["1er", "2ème", "1er", "2ème", "1er"],
            "données1": np.random.default_rng(2).standard_normal(5),
            "données2": np.random.default_rng(2).standard_normal(5),
        }
    )

    # 对 DataFrame 执行 pivot_table 操作，以'clé1'为行索引，'clé2'为列索引，计算数据的平均值
    # 调用 _repr_html_() 方法将生成的表格转换为 HTML 格式的字符串并返回
    df.pivot_table(index=["clé1"], columns=["clé2"])._repr_html_()
# 定义测试函数，将DataFrame转换为HTML表格并截断显示行和列
def test_to_html_truncate(datapath):
    # 创建一个日期索引，从2001年1月1日开始，每日频率，总共20个周期
    index = pd.date_range(start="20010101", freq="D", periods=20)
    # 创建一个DataFrame，行索引为日期，列索引为0到19的整数
    df = DataFrame(index=index, columns=range(20))
    # 将DataFrame转换为HTML格式的表格，最多显示8行和4列
    result = df.to_html(max_rows=8, max_cols=4)
    # 获取预期的HTML结果，根据给定的datapath和"truncate"参数
    expected = expected_html(datapath, "truncate")
    # 断言实际结果与预期结果相等
    assert result == expected


# 使用pytest的参数化测试，测试当formatters参数长度与DataFrame列数不匹配时是否引发ValueError异常
@pytest.mark.parametrize("size", [1, 5])
def test_html_invalid_formatters_arg_raises(size):
    # 创建一个空的DataFrame，列为["a", "b", "c"]
    df = DataFrame(columns=["a", "b", "c"])
    # 设置异常消息格式
    msg = "Formatters length({}) should match DataFrame number of columns(3)"
    # 使用pytest的raises断言检查是否抛出预期的ValueError异常，异常消息包含size的值
    with pytest.raises(ValueError, match=re.escape(msg.format(size))):
        # 调用to_html方法，设置formatters参数为长度为size的字符串格式化函数列表
        df.to_html(formatters=["{}".format] * size)


# 测试将DataFrame转换为HTML表格时，使用自定义格式化函数和最大列数为3的情况
def test_to_html_truncate_formatter(datapath):
    # 创建包含字典数据的列表
    data = [
        {"A": 1, "B": 2, "C": 3, "D": 4},
        {"A": 5, "B": 6, "C": 7, "D": 8},
        {"A": 9, "B": 10, "C": 11, "D": 12},
        {"A": 13, "B": 14, "C": 15, "D": 16},
    ]
    # 创建DataFrame，使用data作为数据源
    df = DataFrame(data)
    # 创建格式化函数，将每个值转换为字符串并添加"_mod"后缀
    fmt = lambda x: str(x) + "_mod"
    # 创建格式化函数列表，对应DataFrame的每列
    formatters = [fmt, fmt, None, None]
    # 将DataFrame转换为HTML表格，使用自定义格式化函数和最大列数为3
    result = df.to_html(formatters=formatters, max_cols=3)
    # 获取预期的HTML结果，根据给定的datapath和"truncate_formatter"参数
    expected = expected_html(datapath, "truncate_formatter")
    # 断言实际结果与预期结果相等
    assert result == expected


# 使用pytest的参数化测试，测试将多级索引DataFrame转换为HTML表格时的不同情况
@pytest.mark.parametrize(
    "sparsify,expected",
    [(True, "truncate_multi_index"), (False, "truncate_multi_index_sparse_off")],
)
def test_to_html_truncate_multi_index(sparsify, expected, datapath):
    # 创建包含两级索引的数组
    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    ]
    # 创建具有两级索引的DataFrame
    df = DataFrame(index=arrays, columns=arrays)
    # 将DataFrame转换为HTML表格，最多显示7行和7列，并根据sparsify参数处理稀疏性
    result = df.to_html(max_rows=7, max_cols=7, sparsify=sparsify)
    # 获取预期的HTML结果，根据给定的datapath和expected参数
    expected = expected_html(datapath, expected)
    # 断言实际结果与预期结果相等
    assert result == expected


# 使用pytest的参数化测试，测试DataFrame转换为HTML表格时不同的选项和结果
@pytest.mark.parametrize(
    "option,result,expected",
    [
        (None, lambda df: df.to_html(), "1"),
        (None, lambda df: df.to_html(border=2), "2"),
        (2, lambda df: df.to_html(), "2"),
        (2, lambda df: df._repr_html_(), "2"),
    ],
)
def test_to_html_border(option, result, expected):
    # 创建包含列"A"和值[1, 2]的DataFrame
    df = DataFrame({"A": [1, 2]})
    # 根据选项情况，调用DataFrame的to_html方法或_repr_html_方法，并设置相关选项
    if option is None:
        result = result(df)
    else:
        with option_context("display.html.border", option):
            result = result(df)
    # 构造预期的HTML字符串，包含相应的border属性值
    expected = f'border="{expected}"'
    # 断言实际结果包含预期的HTML字符串
    assert expected in result


# 使用pytest的参数化测试，测试DataFrame转换为HTML表格时处理空DataFrame的情况
@pytest.mark.parametrize("biggie_df_fixture", ["mixed"], indirect=True)
def test_to_html(biggie_df_fixture):
    # 获取由biggie_df_fixture装饰器提供的DataFrame
    df = biggie_df_fixture
    # 调用DataFrame的to_html方法，不传递buf参数时返回HTML字符串s
    s = df.to_html()
    # 创建一个StringIO对象作为缓冲区
    buf = StringIO()
    # 调用DataFrame的to_html方法，传递buf参数后，返回值retval为None
    retval = df.to_html(buf=buf)
    # 断言retval为None
    assert retval is None
    # 断言buf对象的值与s相同
    assert buf.getvalue() == s
    # 断言s是字符串类型
    assert isinstance(s, str)
    
    # 测试DataFrame的to_html方法，分别指定列和列间距(col_space)参数
    df.to_html(columns=["B", "A"], col_space=17)
    df.to_html(columns=["B", "A"], formatters={"A": lambda x: f"{x:.1f}"})
    
    # 测试DataFrame的to_html方法，指定列和浮点格式(float_format)参数
    df.to_html(columns=["B", "A"], float_format=str)
    df.to_html(columns=["B", "A"], col_space=12, float_format=str)
    # 将变量 biggie_df_fixture 的值赋给变量 df，通常是一个大型数据框对象
    df = biggie_df_fixture
    # 将数据框 df 转换成 HTML 格式的字符串并输出，用于在网页中显示数据
    df.to_html()
# 定义一个测试函数，将 DataFrame 转换为 HTML 并验证输出文件名
def test_to_html_filename(biggie_df_fixture, tmpdir):
    # 从测试夹具中获取 DataFrame 实例
    df = biggie_df_fixture
    # 生成预期的 HTML 字符串
    expected = df.to_html()
    # 在临时目录下创建名为 "test.html" 的文件路径对象
    path = tmpdir.join("test.html")
    # 将 DataFrame 转换为 HTML 并保存到指定路径
    df.to_html(path)
    # 从文件中读取生成的 HTML 内容
    result = path.read()
    # 断言生成的 HTML 内容与预期相符
    assert result == expected


# 定义一个测试函数，验证在 HTML 中不会使用 <strong> 标签
def test_to_html_with_no_bold():
    # 创建一个包含随机数据的 DataFrame
    df = DataFrame({"x": np.random.default_rng(2).standard_normal(5)})
    # 将 DataFrame 转换为 HTML，不使用加粗行选项
    html = df.to_html(bold_rows=False)
    # 从生成的 HTML 中截取到 "</thead>" 结束标签后的内容
    result = html[html.find("</thead>")]
    # 断言结果中不包含 "<strong" 标签
    assert "<strong" not in result


# 定义一个测试函数，验证在生成的 HTML 中不包含指定列
def test_to_html_columns_arg(float_frame):
    # 将 DataFrame 转换为 HTML，仅包含指定的列 "A"
    result = float_frame.to_html(columns=["A"])
    # 断言生成的 HTML 中不包含 "<th>B</th>" 标签
    assert "<th>B</th>" not in result


# 使用参数化测试装饰器，定义多个测试参数的测试函数
@pytest.mark.parametrize(
    "columns,justify,expected",
    [
        (
            # 创建一个多级索引，命名为 "CL0" 和 "CL1"
            MultiIndex.from_arrays(
                [np.arange(2).repeat(2), np.mod(range(4), 2)],
                names=["CL0", "CL1"],
            ),
            "left",  # 设定对齐方式为左对齐
            "multiindex_1",  # 预期的 HTML 结果名称为 "multiindex_1"
        ),
        (
            # 创建一个多级索引，包含两个默认的整数数组
            MultiIndex.from_arrays([np.arange(4), np.mod(range(4), 2)]),
            "right",  # 设定对齐方式为右对齐
            "multiindex_2",  # 预期的 HTML 结果名称为 "multiindex_2"
        ),
    ],
)
def test_to_html_multiindex(columns, justify, expected, datapath):
    # 创建一个包含多级索引的 DataFrame 实例
    df = DataFrame([list("abcd"), list("efgh")], columns=columns)
    # 将 DataFrame 转换为 HTML，并指定对齐方式
    result = df.to_html(justify=justify)
    # 根据数据路径和预期结果名称，获取预期的 HTML 内容
    expected = expected_html(datapath, expected)
    # 断言生成的 HTML 内容与预期相符
    assert result == expected


# 定义一个测试函数，验证在指定对齐方式时生成的 HTML 结果
def test_to_html_justify(justify, datapath):
    # 创建一个包含指定列和数据的 DataFrame 实例
    df = DataFrame(
        {"A": [6, 30000, 2], "B": [1, 2, 70000], "C": [223442, 0, 1]},
        columns=["A", "B", "C"],
    )
    # 将 DataFrame 转换为 HTML，并根据给定的对齐方式格式化预期结果
    result = df.to_html(justify=justify)
    expected = expected_html(datapath, "justify").format(justify=justify)
    # 断言生成的 HTML 内容与预期相符
    assert result == expected


# 使用参数化测试装饰器，定义多个无效对齐方式的测试函数
@pytest.mark.parametrize(
    "justify", ["super-right", "small-left", "noinherit", "tiny", "pandas"]
)
def test_to_html_invalid_justify(justify):
    # 创建一个空的 DataFrame 实例
    df = DataFrame()
    msg = "Invalid value for justify parameter"
    # 使用 pytest 的断言检查，验证当传入无效的对齐方式时是否会抛出 ValueError 异常，并包含预期的错误信息
    with pytest.raises(ValueError, match=msg):
        df.to_html(justify=justify)


# 定义一个测试类 TestHTMLIndex
class TestHTMLIndex:
    # 定义一个测试夹具，返回一个包含索引的 DataFrame 实例
    @pytest.fixture
    def df(self):
        index = ["foo", "bar", "baz"]
        df = DataFrame(
            {"A": [1, 2, 3], "B": [1.2, 3.4, 5.6], "C": ["one", "two", np.nan]},
            columns=["A", "B", "C"],
            index=index,
        )
        return df

    # 定义一个测试夹具，返回不包含索引的预期 HTML 结果
    @pytest.fixture
    def expected_without_index(self, datapath):
        return expected_html(datapath, "index_2")

    # 定义一个测试函数，验证在没有索引名称的情况下生成的 HTML 结果
    def test_to_html_flat_index_without_name(
        self, datapath, df, expected_without_index
    ):
        # 获取带有索引名称的预期 HTML 结果
        expected_with_index = expected_html(datapath, "index_1")
        # 断言使用默认设置时生成的 HTML 与带有索引名称的预期结果相符
        assert df.to_html() == expected_with_index
        # 将 DataFrame 转换为 HTML，不包含索引
        result = df.to_html(index=False)
        # 验证生成的 HTML 不包含任何索引值
        for i in df.index:
            assert i not in result
        # 断言生成的 HTML 与不包含索引的预期结果相符
        assert result == expected_without_index
    # 定义测试方法，将 DataFrame 转换为带有名称的单层索引的 HTML 表格
    def test_to_html_flat_index_with_name(self, datapath, df, expected_without_index):
        # 将 DataFrame 的索引设置为单层索引，名称为 "idx"
        df.index = Index(["foo", "bar", "baz"], name="idx")
        # 获取带有索引的预期 HTML 表格内容
        expected_with_index = expected_html(datapath, "index_3")
        # 断言 DataFrame 转换为 HTML 的结果与带索引的预期内容相等
        assert df.to_html() == expected_with_index
        # 断言 DataFrame 转换为 HTML，但不包含索引列时的结果与预期内容相等
        assert df.to_html(index=False) == expected_without_index

    # 定义测试方法，将 DataFrame 转换为多层索引但不带名称的 HTML 表格
    def test_to_html_multiindex_without_names(
        self, datapath, df, expected_without_index
    ):
        # 创建包含元组的 MultiIndex，用于 DataFrame 的索引
        tuples = [("foo", "car"), ("foo", "bike"), ("bar", "car")]
        df.index = MultiIndex.from_tuples(tuples)
        # 获取带有索引的预期 HTML 表格内容
        expected_with_index = expected_html(datapath, "index_4")
        # 断言 DataFrame 转换为 HTML 的结果与带索引的预期内容相等
        assert df.to_html() == expected_with_index

        # 获取不包含索引列的 HTML 表格内容，并断言结果中不包含指定的元素
        result = df.to_html(index=False)
        for i in ["foo", "bar", "car", "bike"]:
            assert i not in result
        
        # 断言不包含索引列的 HTML 表格内容与预期的不包含索引的内容相等
        assert result == expected_without_index

    # 定义测试方法，将 DataFrame 转换为带有名称的多层索引的 HTML 表格
    def test_to_html_multiindex_with_names(self, datapath, df, expected_without_index):
        # 创建包含元组的 MultiIndex，同时指定索引的名称为 "idx1" 和 "idx2"
        tuples = [("foo", "car"), ("foo", "bike"), ("bar", "car")]
        df.index = MultiIndex.from_tuples(tuples, names=["idx1", "idx2"])
        # 获取带有索引的预期 HTML 表格内容
        expected_with_index = expected_html(datapath, "index_5")
        # 断言 DataFrame 转换为 HTML 的结果与带索引的预期内容相等
        assert df.to_html() == expected_with_index
        # 断言 DataFrame 转换为 HTML，但不包含索引列时的结果与预期内容相等
        assert df.to_html(index=False) == expected_without_index
@pytest.mark.parametrize("classes", ["sortable draggable", ["sortable", "draggable"]])
# 使用 pytest 的参数化装饰器，为测试函数 test_to_html_with_classes 提供两组参数化输入
def test_to_html_with_classes(classes, datapath):
    # 创建一个空的 DataFrame 对象
    df = DataFrame()
    # 使用指定的数据路径和标识加载预期的 HTML 输出
    expected = expected_html(datapath, "with_classes")
    # 调用 DataFrame 对象的 to_html 方法，传入 classes 参数生成结果
    result = df.to_html(classes=classes)
    # 断言结果与预期输出相等
    assert result == expected


def test_to_html_no_index_max_rows(datapath):
    # GH 14998
    # 创建一个包含列"A"的 DataFrame 对象
    df = DataFrame({"A": [1, 2, 3, 4]})
    # 调用 DataFrame 对象的 to_html 方法，传入 index=False 和 max_rows=1 参数生成结果
    result = df.to_html(index=False, max_rows=1)
    # 使用指定的数据路径和标识加载预期的 HTML 输出
    expected = expected_html(datapath, "gh14998_expected_output")
    # 断言结果与预期输出相等
    assert result == expected


def test_to_html_multiindex_max_cols(datapath):
    # GH 6131
    # 创建一个包含 MultiIndex 索引和列的 DataFrame 对象
    index = MultiIndex(
        levels=[["ba", "bb", "bc"], ["ca", "cb", "cc"]],
        codes=[[0, 1, 2], [0, 1, 2]],
        names=["b", "c"],
    )
    columns = MultiIndex(
        levels=[["d"], ["aa", "ab", "ac"]],
        codes=[[0, 0, 0], [0, 1, 2]],
        names=[None, "a"],
    )
    data = np.array(
        [[1.0, np.nan, np.nan], [np.nan, 2.0, np.nan], [np.nan, np.nan, 3.0]]
    )
    df = DataFrame(data, index, columns)
    # 调用 DataFrame 对象的 to_html 方法，传入 max_cols=2 参数生成结果
    result = df.to_html(max_cols=2)
    # 使用指定的数据路径和标识加载预期的 HTML 输出
    expected = expected_html(datapath, "gh6131_expected_output")
    # 断言结果与预期输出相等
    assert result == expected


def test_to_html_multi_indexes_index_false(datapath):
    # GH 22579
    # 创建一个包含 MultiIndex 索引和标准列的 DataFrame 对象
    df = DataFrame(
        {"a": range(10), "b": range(10, 20), "c": range(10, 20), "d": range(10, 20)}
    )
    # 设置 DataFrame 对象的列为 MultiIndex 格式
    df.columns = MultiIndex.from_product([["a", "b"], ["c", "d"]])
    # 设置 DataFrame 对象的索引为 MultiIndex 格式
    df.index = MultiIndex.from_product([["a", "b"], ["c", "d", "e", "f", "g"]])
    # 调用 DataFrame 对象的 to_html 方法，传入 index=False 参数生成结果
    result = df.to_html(index=False)
    # 使用指定的数据路径和标识加载预期的 HTML 输出
    expected = expected_html(datapath, "gh22579_expected_output")
    # 断言结果与预期输出相等
    assert result == expected


@pytest.mark.parametrize("index_names", [True, False])
@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize(
    "column_index, column_type",
    [
        (Index([0, 1]), "unnamed_standard"),
        (Index([0, 1], name="columns.name"), "named_standard"),
        (MultiIndex.from_product([["a"], ["b", "c"]]), "unnamed_multi"),
        (
            MultiIndex.from_product(
                [["a"], ["b", "c"]], names=["columns.name.0", "columns.name.1"]
            ),
            "named_multi",
        ),
    ],
)
@pytest.mark.parametrize(
    "row_index, row_type",
    [
        (Index([0, 1]), "unnamed_standard"),
        (Index([0, 1], name="index.name"), "named_standard"),
        (MultiIndex.from_product([["a"], ["b", "c"]]), "unnamed_multi"),
        (
            MultiIndex.from_product(
                [["a"], ["b", "c"]], names=["index.name.0", "index.name.1"]
            ),
            "named_multi",
        ),
    ],
)
# 使用 pytest 的参数化装饰器，为测试函数 test_to_html_basic_alignment 提供多组参数化输入
def test_to_html_basic_alignment(
    datapath, row_index, row_type, column_index, column_type, index, header, index_names
):
    # GH 22747, GH 22579
    # 创建一个包含指定索引和列的 DataFrame 对象
    df = DataFrame(np.zeros((2, 2), dtype=int), index=row_index, columns=column_index)
    # 调用 DataFrame 对象的 to_html 方法，传入 index, header, index_names 参数生成结果
    result = df.to_html(index=index, header=header, index_names=index_names)
    # 如果 index 为空，则将行类型设为 "none"
    if not index:
        row_type = "none"
    
    # 如果 index_names 为空且行类型以 "named" 开头，则将行类型设为其反义词形式
    elif not index_names and row_type.startswith("named"):
        row_type = "un" + row_type

    # 如果 header 为空，则将列类型设为 "none"
    if not header:
        column_type = "none"
    
    # 如果 index_names 为空且列类型以 "named" 开头，则将列类型设为其反义词形式
    elif not index_names and column_type.startswith("named"):
        column_type = "un" + column_type

    # 组合文件名，包括行类型和列类型
    filename = "index_" + row_type + "_columns_" + column_type
    
    # 根据 datapath 和 filename 预期得到 HTML 数据
    expected = expected_html(datapath, filename)
    
    # 断言结果与预期相等
    assert result == expected
@pytest.mark.parametrize("index_names", [True, False])
@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize(
    "column_index, column_type",
    [
        (Index(np.arange(8)), "unnamed_standard"),  # 创建一个8个元素的索引对象，未命名
        (Index(np.arange(8), name="columns.name"), "named_standard"),  # 创建一个带命名的8个元素的索引对象
        (
            MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]]),  # 创建一个多级索引对象，未命名
            "unnamed_multi",
        ),
        (
            MultiIndex.from_product(
                [["a", "b"], ["c", "d"], ["e", "f"]], names=["foo", None, "baz"]
            ),  # 创建一个多级索引对象，带有部分命名
            "named_multi",
        ),
    ],
)
@pytest.mark.parametrize(
    "row_index, row_type",
    [
        (Index(np.arange(8)), "unnamed_standard"),  # 创建一个8个元素的行索引对象，未命名
        (Index(np.arange(8), name="index.name"), "named_standard"),  # 创建一个带命名的8个元素的行索引对象
        (
            MultiIndex.from_product([["a", "b"], ["c", "d"], ["e", "f"]]),  # 创建一个多级行索引对象，未命名
            "unnamed_multi",
        ),
        (
            MultiIndex.from_product(
                [["a", "b"], ["c", "d"], ["e", "f"]], names=["foo", None, "baz"]
            ),  # 创建一个多级行索引对象，带有部分命名
            "named_multi",
        ),
    ],
)
def test_to_html_alignment_with_truncation(
    datapath, row_index, row_type, column_index, column_type, index, header, index_names
):
    # GH 22747, GH 22579
    # 创建一个DataFrame对象，填充为0到63的整数，行索引和列索引由参数指定
    df = DataFrame(np.arange(64).reshape(8, 8), index=row_index, columns=column_index)
    # 将DataFrame对象转换为HTML格式，进行截断处理，根据参数设置是否包含索引和标题
    result = df.to_html(
        max_rows=4, max_cols=4, index=index, header=header, index_names=index_names
    )

    # 根据条件修改行索引类型字符串
    if not index:
        row_type = "none"
    elif not index_names and row_type.startswith("named"):
        row_type = "un" + row_type

    # 根据条件修改列索引类型字符串
    if not header:
        column_type = "none"
    elif not index_names and column_type.startswith("named"):
        column_type = "un" + column_type

    # 根据修改后的行列索引类型构建预期的文件名字符串
    filename = "trunc_df_index_" + row_type + "_columns_" + column_type
    # 根据数据路径和文件名获取预期的HTML内容
    expected = expected_html(datapath, filename)
    # 断言结果与预期相符
    assert result == expected


@pytest.mark.parametrize("index", [False, 0])
def test_to_html_truncation_index_false_max_rows(datapath, index):
    # GH 15019
    # 创建一个包含浮点数的DataFrame对象
    data = [
        [1.764052, 0.400157],
        [0.978738, 2.240893],
        [1.867558, -0.977278],
        [0.950088, -0.151357],
        [-0.103219, 0.410599],
    ]
    df = DataFrame(data)
    # 将DataFrame对象转换为HTML格式，根据参数设置是否包含索引
    result = df.to_html(max_rows=4, index=index)
    # 获取预期的HTML内容
    expected = expected_html(datapath, "gh15019_expected_output")
    # 断言结果与预期相符
    assert result == expected


@pytest.mark.parametrize("index", [False, 0])
@pytest.mark.parametrize(
    "col_index_named, expected_output",
    [(False, "gh22783_expected_output"), (True, "gh22783_named_columns_index")],
)
def test_to_html_truncation_index_false_max_cols(
    datapath, index, col_index_named, expected_output
):
    # GH 22783
    # 创建一个包含浮点数的DataFrame对象
    data = [
        [1.764052, 0.400157, 0.978738, 2.240893, 1.867558],
        [-0.977278, 0.950088, -0.151357, -0.103219, 0.410599],
    ]
    df = DataFrame(data)
    # 如果col_index_named为True，则为DataFrame对象的列重命名
    if col_index_named:
        df.columns.rename("columns.name", inplace=True)
    # 将 DataFrame 转换为 HTML 表格格式，最多显示4列，包括指定的索引
    result = df.to_html(max_cols=4, index=index)
    # 根据给定的数据路径和预期输出生成预期的 HTML 内容
    expected = expected_html(datapath, expected_output)
    # 断言实际生成的 HTML 内容与预期的 HTML 内容相等
    assert result == expected
@pytest.mark.parametrize("notebook", [True, False])
def test_to_html_notebook_has_style(notebook):
    # 创建一个 DataFrame 包含一列名为 "A" 的数据
    df = DataFrame({"A": [1, 2, 3]})
    # 调用 to_html 方法将 DataFrame 转换为 HTML 字符串，根据 notebook 参数选择不同的输出样式
    result = df.to_html(notebook=notebook)

    # 根据 notebook 参数断言不同的 HTML 输出结果
    if notebook:
        assert "tbody tr th:only-of-type" in result
        assert "vertical-align: middle;" in result
        assert "thead th" in result
    else:
        assert "tbody tr th:only-of-type" not in result
        assert "vertical-align: middle;" not in result
        assert "thead th" not in result


def test_to_html_with_index_names_false():
    # GH 16493
    # 创建一个带有索引名为 "myindexname" 的 DataFrame
    df = DataFrame({"A": [1, 2]}, index=Index(["a", "b"], name="myindexname"))
    # 调用 to_html 方法将 DataFrame 转换为 HTML 字符串，禁用索引名显示
    result = df.to_html(index_names=False)
    # 断言结果中不包含索引名 "myindexname"
    assert "myindexname" not in result


def test_to_html_with_id():
    # GH 8496
    # 创建一个带有索引名为 "myindexname" 的 DataFrame
    df = DataFrame({"A": [1, 2]}, index=Index(["a", "b"], name="myindexname"))
    # 调用 to_html 方法将 DataFrame 转换为 HTML 字符串，设置表格 ID 为 "TEST_ID"，并禁用索引名显示
    result = df.to_html(index_names=False, table_id="TEST_ID")
    # 断言结果中包含 ' id="TEST_ID"'
    assert ' id="TEST_ID"' in result


@pytest.mark.parametrize(
    "value,float_format,expected",
    [
        (0.19999, "%.3f", "gh21625_expected_output"),
        (100.0, "%.0f", "gh22270_expected_output"),
    ],
)
def test_to_html_float_format_no_fixed_width(value, float_format, expected, datapath):
    # GH 21625, GH 22270
    # 创建一个包含数值为 value 的 DataFrame
    df = DataFrame({"x": [value]})
    # 使用预期的 HTML 输出结果
    expected = expected_html(datapath, expected)
    # 调用 to_html 方法将 DataFrame 转换为 HTML 字符串，使用指定的浮点格式 float_format
    result = df.to_html(float_format=float_format)
    # 断言结果与预期相符
    assert result == expected


@pytest.mark.parametrize(
    "render_links,expected",
    [(True, "render_links_true"), (False, "render_links_false")],
)
def test_to_html_render_links(render_links, expected, datapath):
    # GH 2679
    # 创建一个包含链接数据的 DataFrame
    data = [
        [0, "https://pandas.pydata.org/?q1=a&q2=b", "pydata.org"],
        [0, "www.pydata.org", "pydata.org"],
    ]
    df = DataFrame(data, columns=Index(["foo", "bar", None], dtype=object))

    # 调用 to_html 方法将 DataFrame 转换为 HTML 字符串，根据 render_links 参数决定是否渲染链接
    result = df.to_html(render_links=render_links)
    # 使用预期的 HTML 输出结果
    expected = expected_html(datapath, expected)
    # 断言结果与预期相符
    assert result == expected


@pytest.mark.parametrize(
    "method,expected",
    [
        ("to_html", lambda x: lorem_ipsum),
        ("_repr_html_", lambda x: lorem_ipsum[: x - 4] + "..."),  # 回归测试案例
    ],
)
@pytest.mark.parametrize("max_colwidth", [10, 20, 50, 100])
def test_ignore_display_max_colwidth(method, expected, max_colwidth):
    # see gh-17004
    # 创建一个包含 Lorem Ipsum 文本的 DataFrame
    df = DataFrame([lorem_ipsum])
    # 设置显示选项，限制列宽为 max_colwidth
    with option_context("display.max_colwidth", max_colwidth):
        # 调用方法（to_html 或 _repr_html_）将 DataFrame 转换为 HTML 字符串
        result = getattr(df, method)()
    # 使用预期的 HTML 输出结果
    expected = expected(max_colwidth)
    # 断言结果中包含预期的内容
    assert expected in result


@pytest.mark.parametrize("classes", [True, 0])
def test_to_html_invalid_classes_type(classes):
    # GH 25608
    # 创建一个空的 DataFrame
    df = DataFrame()
    # 设置预期的异常消息
    msg = "classes must be a string, list, or tuple"

    # 断言调用 to_html 方法时，传入无效的 classes 参数会抛出 TypeError 异常，并且异常消息符合预期
    with pytest.raises(TypeError, match=msg):
        df.to_html(classes=classes)


def test_to_html_round_column_headers():
    # GH 17280
    # 创建一个包含浮点数列名的 DataFrame
    df = DataFrame([1], columns=[0.55555])
    # 设置显示选项，限制小数精度为 3
    with option_context("display.precision", 3):
        # 调用 to_html 方法将 DataFrame 转换为 HTML 字符串，根据 notebook 参数选择不同的输出样式
        html = df.to_html(notebook=False)
        notebook = df.to_html(notebook=True)
    # 确保字符串 "0.55555" 存在于变量 html 中
    assert "0.55555" in html
    # 确保字符串 "0.556" 存在于变量 notebook 中
    assert "0.556" in notebook
@pytest.mark.parametrize("unit", ["100px", "10%", "5em", 150])
def test_to_html_with_col_space_units(unit):
    # 测试用例：测试 DataFrame 转换为 HTML 时的列空间单位设置
    # 创建一个包含随机数据的 DataFrame
    df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
    # 调用 DataFrame 的 to_html 方法，设置列空间为 unit，并获取结果字符串
    result = df.to_html(col_space=unit)
    # 将结果字符串按 "tbody" 进行分割，只保留前面部分
    result = result.split("tbody")[0]
    # 从结果中提取所有包含 <th> 标签的行
    hdrs = [x for x in result.split("\n") if re.search(r"<th[>\s]", x)]
    # 如果 unit 是整数类型，将其转换为字符串形式加上 "px"
    if isinstance(unit, int):
        unit = str(unit) + "px"
    # 遍历所有的 <th> 标签行，验证每行中是否包含预期的最小宽度样式设置
    for h in hdrs:
        expected = f'<th style="min-width: {unit};">'
        assert expected in h


class TestReprHTML:
    def test_html_repr_min_rows_default(self, datapath):
        # 测试用例：测试默认情况下 DataFrame 的 HTML 表示方法对最小行数的处理

        # 创建一个包含 20 行数据的 DataFrame
        df = DataFrame({"a": range(20)})
        # 调用 DataFrame 的 _repr_html_ 方法，获取结果 HTML 字符串
        result = df._repr_html_()
        # 根据预期数据路径和文件名获取期望的 HTML 字符串
        expected = expected_html(datapath, "html_repr_min_rows_default_no_truncation")
        # 断言实际结果与预期结果相等
        assert result == expected

        # 创建一个包含 61 行数据的 DataFrame
        df = DataFrame({"a": range(61)})
        # 再次调用 DataFrame 的 _repr_html_ 方法，获取结果 HTML 字符串
        result = df._repr_html_()
        # 根据预期数据路径和文件名获取另一个预期的 HTML 字符串（此时会触发截断）
        expected = expected_html(datapath, "html_repr_min_rows_default_truncated")
        # 断言实际结果与预期结果相等
        assert result == expected

    @pytest.mark.parametrize(
        "max_rows,min_rows,expected",
        [
            # 截断前两行
            (10, 4, "html_repr_max_rows_10_min_rows_4"),
            # 当 min_rows 设为 None 时，使用 max_rows 的值
            (12, None, "html_repr_max_rows_12_min_rows_None"),
            # 当 min_rows 大于 max_rows 时，使用 min_rows 的值
            (10, 12, "html_repr_max_rows_10_min_rows_12"),
            # max_rows 设为 None 时，不截断
            (None, 12, "html_repr_max_rows_None_min_rows_12"),
        ],
    )
    def test_html_repr_min_rows(self, datapath, max_rows, min_rows, expected):
        # 测试用例：测试 DataFrame 的 HTML 表示方法对最小行数和最大行数的处理

        # 创建一个包含 61 行数据的 DataFrame
        df = DataFrame({"a": range(61)})
        # 根据预期数据路径和文件名获取预期的 HTML 字符串
        expected = expected_html(datapath, expected)
        # 使用 option_context 设置 display.max_rows 和 display.min_rows 的值，并获取结果 HTML 字符串
        with option_context("display.max_rows", max_rows, "display.min_rows", min_rows):
            result = df._repr_html_()
        # 断言实际结果与预期结果相等
        assert result == expected

    def test_repr_html_ipython_config(self, ip):
        # 测试用例：测试 DataFrame 的 HTML 表示方法在 IPython 配置下的执行情况

        # 定义包含 IPython 代码的字符串
        code = textwrap.dedent(
            """\
        from pandas import DataFrame
        df = DataFrame({"A": [1, 2]})
        df._repr_html_()

        cfg = get_ipython().config
        cfg['IPKernelApp']['parent_appname']
        df._repr_html_()
        """
        )
        # 执行 IPython 代码，并静默处理执行结果
        result = ip.run_cell(code, silent=True)
        # 断言执行结果没有错误
        assert not result.error_in_exec
    def test_info_repr_html(self):
        max_rows = 60
        max_cols = 20
        # 设置最大行数和最大列数
        h, w = max_rows + 1, max_cols - 1
        # 创建一个 DataFrame 对象，其中每列是一个范围数组，键为列的索引
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        # 断言 HTML 表示中不包含 "<class"
        assert r"&lt;class" not in df._repr_html_()
        # 使用上下文管理器设置显示大数据情况下的选项
        with option_context("display.large_repr", "info"):
            # 断言 HTML 表示中包含 "<class"
            assert r"&lt;class" in df._repr_html_()

        # 设置最大行数和最大列数
        h, w = max_rows - 1, max_cols + 1
        # 创建一个 DataFrame 对象，其中每列是一个范围数组，键为列的索引
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        # 断言 HTML 表示中不包含 "<class"
        assert "<class" not in df._repr_html_()
        # 使用上下文管理器设置显示大数据情况下的选项
        with option_context(
            "display.large_repr", "info", "display.max_columns", max_cols
        ):
            # 断言 HTML 表示中包含 "<class"
            assert "&lt;class" in df._repr_html_()

    def test_fake_qtconsole_repr_html(self, float_frame):
        df = float_frame

        def get_ipython():
            return {"config": {"KernelApp": {"parent_appname": "ipython-qtconsole"}}}

        # 获取 DataFrame 的 HTML 表示字符串
        repstr = df._repr_html_()
        # 断言 HTML 表示字符串不为 None
        assert repstr is not None

        # 使用上下文管理器设置显示最大行数和最大列数的选项
        with option_context("display.max_rows", 5, "display.max_columns", 2):
            # 获取 DataFrame 的 HTML 表示字符串
            repstr = df._repr_html_()

        # 断言 HTML 表示字符串中包含 "class"，作为信息回退的标志
        assert "class" in repstr  # info fallback

    def test_repr_html(self, float_frame):
        df = float_frame
        # 获取 DataFrame 的 HTML 表示字符串
        df._repr_html_()

        # 使用上下文管理器设置显示最大行数和最大列数的选项
        with option_context("display.max_rows", 1, "display.max_columns", 1):
            # 获取 DataFrame 的 HTML 表示字符串
            df._repr_html_()

        # 使用上下文管理器设置不使用 notebook 的 HTML 表示
        with option_context("display.notebook_repr_html", False):
            # 获取 DataFrame 的 HTML 表示字符串
            df._repr_html_()

        # 创建一个 DataFrame 对象
        df = DataFrame([[1, 2], [3, 4]])
        # 使用上下文管理器设置显示尺寸信息的选项
        with option_context("display.show_dimensions", True):
            # 断言 HTML 表示字符串中包含 "2 rows"
            assert "2 rows" in df._repr_html_()
        # 使用上下文管理器设置不显示尺寸信息的选项
        with option_context("display.show_dimensions", False):
            # 断言 HTML 表示字符串中不包含 "2 rows"
            assert "2 rows" not in df._repr_html_()

    def test_repr_html_mathjax(self):
        # 创建一个 DataFrame 对象
        df = DataFrame([[1, 2], [3, 4]])
        # 断言 HTML 表示字符串中不包含 "tex2jax_ignore"
        assert "tex2jax_ignore" not in df._repr_html_()

        # 使用上下文管理器设置不使用 MathJax 渲染的选项
        with option_context("display.html.use_mathjax", False):
            # 断言 HTML 表示字符串中包含 "tex2jax_ignore"
            assert "tex2jax_ignore" in df._repr_html_()

    def test_repr_html_wide(self):
        max_cols = 20
        # 创建一个宽度接近最大列数的 DataFrame 对象
        df = DataFrame([["a" * 25] * (max_cols - 1)] * 10)
        # 使用上下文管理器设置显示最大行数和最大列数的选项
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            # 断言 HTML 表示字符串中不包含 "..."
            assert "..." not in df._repr_html_()

        # 创建一个超过最大列数的宽 DataFrame 对象
        wide_df = DataFrame([["a" * 25] * (max_cols + 1)] * 10)
        # 使用上下文管理器设置显示最大行数和最大列数的选项
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            # 断言 HTML 表示字符串中包含 "..."
            assert "..." in wide_df._repr_html_()
    # 定义测试函数，用于测试多级索引列展示在 HTML 中的表现
    def test_repr_html_wide_multiindex_cols(self):
        # 设置最大列数为20
        max_cols = 20
        
        # 创建一个包含两级索引的 MultiIndex 对象，每级索引包含的元素是 [0, max_cols // 2) 范围内的数字和 ["foo", "bar"] 两个字符串
        mcols = MultiIndex.from_product(
            [np.arange(max_cols // 2), ["foo", "bar"]], names=["first", "second"]
        )
        
        # 创建一个 DataFrame，每列都包含相同的长度为 25 的字符串 "a"*25，共计 10 行，列索引为 mcols
        df = DataFrame([["a" * 25] * len(mcols)] * 10, columns=mcols)
        
        # 调用 DataFrame 的 _repr_html_ 方法生成 HTML 表示，并将结果保存在 reg_repr 变量中
        reg_repr = df._repr_html_()
        
        # 断言 HTML 表示中不包含 "..."
        assert "..." not in reg_repr
        
        # 重新设置 mcols，使其包含更多的列，列数为 1 + (max_cols // 2)，并保持二级索引为 ["foo", "bar"]
        mcols = MultiIndex.from_product(
            (np.arange(1 + (max_cols // 2)), ["foo", "bar"]), names=["first", "second"]
        )
        
        # 重新创建 DataFrame，设置与之前相同的内容结构
        df = DataFrame([["a" * 25] * len(mcols)] * 10, columns=mcols)
        
        # 使用 option_context 设置显示选项，限制最大行数为 60，最大列数为 20
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            # 调用 DataFrame 的 _repr_html_ 方法生成 HTML 表示，并断言其中包含 "..."
            assert "..." in df._repr_html_()

    # 定义测试函数，用于测试长 DataFrame 的 HTML 表示
    def test_repr_html_long(self):
        # 使用 option_context 设置显示选项，限制最大行数为 60
        with option_context("display.max_rows", 60):
            # 获取当前显示选项中的最大行数
            max_rows = get_option("display.max_rows")
            
            # 计算 DataFrame 长度为 max_rows - 1，创建 DataFrame 包含两列 "A" 和 "B"
            h = max_rows - 1
            df = DataFrame({"A": np.arange(1, 1 + h), "B": np.arange(41, 41 + h)})
            
            # 调用 DataFrame 的 _repr_html_ 方法生成 HTML 表示，并断言其中不包含 ".."，以及特定数字在 HTML 表示中出现
            reg_repr = df._repr_html_()
            assert ".." not in reg_repr
            assert str(41 + max_rows // 2) in reg_repr
            
            # 重新设置 h，使其大于当前最大行数，创建更长的 DataFrame
            h = max_rows + 1
            df = DataFrame({"A": np.arange(1, 1 + h), "B": np.arange(41, 41 + h)})
            
            # 调用 DataFrame 的 _repr_html_ 方法生成 HTML 表示，并断言其中包含 ".."，以及特定文本和数字在 HTML 表示中出现
            long_repr = df._repr_html_()
            assert ".." in long_repr
            assert str(41 + max_rows // 2) not in long_repr
            assert f"{h} rows " in long_repr
            assert "2 columns" in long_repr

    # 定义测试函数，用于测试包含浮点索引的 DataFrame 的 HTML 表示
    def test_repr_html_float(self):
        # 使用 option_context 设置显示选项，限制最大行数为 60
        with option_context("display.max_rows", 60):
            # 获取当前显示选项中的最大行数
            max_rows = get_option("display.max_rows")
            
            # 计算 DataFrame 长度为 max_rows - 1，创建包含浮点索引的 DataFrame
            h = max_rows - 1
            df = DataFrame(
                {
                    "idx": np.linspace(-10, 10, h),
                    "A": np.arange(1, 1 + h),
                    "B": np.arange(41, 41 + h),
                }
            ).set_index("idx")
            
            # 调用 DataFrame 的 _repr_html_ 方法生成 HTML 表示，并断言其中不包含 ".."，以及特定字符串在 HTML 表示中出现
            reg_repr = df._repr_html_()
            assert ".." not in reg_repr
            assert f"<td>{40 + h}</td>" in reg_repr
            
            # 重新设置 h，使其大于当前最大行数，创建更长的 DataFrame
            h = max_rows + 1
            df = DataFrame(
                {
                    "idx": np.linspace(-10, 10, h),
                    "A": np.arange(1, 1 + h),
                    "B": np.arange(41, 41 + h),
                }
            ).set_index("idx")
            
            # 调用 DataFrame 的 _repr_html_ 方法生成 HTML 表示，并断言其中包含 ".."，以及特定文本在 HTML 表示中出现
            long_repr = df._repr_html_()
            assert ".." in long_repr
            assert "<td>31</td>" not in long_repr
            assert f"{h} rows " in long_repr
            assert "2 columns" in long_repr
    # 测试生成具有长多索引的 HTML 表示。
    def test_repr_html_long_multiindex(self):
        # 设置每个级别的最大行数
        max_rows = 60
        # 计算第一级别的最大行数
        max_L1 = max_rows // 2

        # 创建一个元组列表，包含所有可能的索引组合
        tuples = list(itertools.product(np.arange(max_L1), ["foo", "bar"]))
        # 从元组列表创建多级索引对象，设置索引名称
        idx = MultiIndex.from_tuples(tuples, names=["first", "second"])
        # 创建一个数据帧，其中元素来自标准正态分布随机数，使用指定的索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((max_L1 * 2, 2)),
            index=idx,
            columns=["A", "B"],
        )
        # 在指定的上下文下设置显示选项，生成数据帧的 HTML 表示
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            # 获取数据帧的 HTML 表示
            reg_repr = df._repr_html_()
        # 断言生成的 HTML 表示中不包含省略号
        assert "..." not in reg_repr

        # 创建另一个元组列表，其中第一级别的最大行数超过前面设置的最大行数
        tuples = list(itertools.product(np.arange(max_L1 + 1), ["foo", "bar"]))
        # 从元组列表创建多级索引对象，设置索引名称
        idx = MultiIndex.from_tuples(tuples, names=["first", "second"])
        # 创建另一个数据帧，其中元素来自标准正态分布随机数，使用指定的索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal(((max_L1 + 1) * 2, 2)),
            index=idx,
            columns=["A", "B"],
        )
        # 获取数据帧的 HTML 表示
        long_repr = df._repr_html_()
        # 断言生成的 HTML 表示中包含省略号
        assert "..." in long_repr

    # 测试生成同时具有长和宽的 HTML 表示。
    def test_repr_html_long_and_wide(self):
        # 设置最大列数和最大行数
        max_cols = 20
        max_rows = 60

        # 计算使用的行数和列数
        h, w = max_rows - 1, max_cols - 1
        # 创建一个数据帧，其中每列都包含从 1 到指定最大行数的整数
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        # 在指定的上下文下设置显示选项，生成数据帧的 HTML 表示
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            # 断言生成的 HTML 表示中不包含省略号
            assert "..." not in df._repr_html_()

        # 计算使用的行数和列数，此时行数和列数均超过了前面设置的最大值
        h, w = max_rows + 1, max_cols + 1
        # 创建另一个数据帧，其中每列都包含从 1 到指定最大行数的整数
        df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
        # 在指定的上下文下设置显示选项，生成数据帧的 HTML 表示
        with option_context("display.max_rows", 60, "display.max_columns", 20):
            # 断言生成的 HTML 表示中包含省略号
            assert "..." in df._repr_html_()
# 定义测试函数，将多级索引 DataFrame 作为参数传入
def test_to_html_multilevel(multiindex_year_month_day_dataframe_random_data):
    # 将传入的多级索引 DataFrame 赋值给变量 ymd
    ymd = multiindex_year_month_day_dataframe_random_data

    # 设置 ymd DataFrame 的列名为 "foo"
    ymd.columns.name = "foo"
    
    # 将 ymd DataFrame 转换为 HTML 格式（但未使用结果）
    ymd.to_html()
    
    # 将 ymd DataFrame 的转置结果转换为 HTML 格式（但未使用结果）
    ymd.T.to_html()


# 使用参数化测试装饰器，测试不同的 na_rep 值和数据路径
@pytest.mark.parametrize("na_rep", ["NaN", "Ted"])
def test_to_html_na_rep_and_float_format(na_rep, datapath):
    # 创建 DataFrame 对象 df，包含两列：["Group", "Data"]
    df = DataFrame(
        [
            ["A", 1.2225],
            ["A", None],
        ],
        columns=["Group", "Data"],
    )
    
    # 将 DataFrame df 转换为 HTML 格式，设置 na_rep 和 float_format 格式
    result = df.to_html(na_rep=na_rep, float_format="{:.2f}".format)
    
    # 使用预期的 HTML 结果填充 expected 变量
    expected = expected_html(datapath, "gh13828_expected_output")
    
    # 使用 na_rep 替换 expected 中的特定占位符
    expected = expected.format(na_rep=na_rep)
    
    # 断言结果与预期是否相同
    assert result == expected


# 测试处理非标量数据的情况，使用参数化测试装饰器和数据路径参数
def test_to_html_na_rep_non_scalar_data(datapath):
    # 创建包含非标量数据的 DataFrame 对象 df
    df = DataFrame([{"a": 1, "b": [1, 2, 3]}])
    
    # 将 DataFrame df 转换为 HTML 格式，设置 na_rep 为 "-"
    result = df.to_html(na_rep="-")
    
    # 使用预期的 HTML 结果填充 expected 变量
    expected = expected_html(datapath, "gh47103_expected_output")
    
    # 断言结果与预期是否相同
    assert result == expected


# 测试处理包含对象列的情况，使用参数化测试装饰器和数据路径参数
def test_to_html_float_format_object_col(datapath):
    # 创建包含对象列的 DataFrame 对象 df
    df = DataFrame(data={"x": [1000.0, "test"]})
    
    # 将 DataFrame df 转换为 HTML 格式，使用自定义的 float_format 函数
    result = df.to_html(float_format=lambda x: f"{x:,.0f}")
    
    # 使用预期的 HTML 结果填充 expected 变量
    expected = expected_html(datapath, "gh40024_expected_output")
    
    # 断言结果与预期是否相同
    assert result == expected


# 测试处理多级索引列，并设置列宽度的情况
def test_to_html_multiindex_col_with_colspace():
    # 创建包含多级索引列的 DataFrame 对象 df
    df = DataFrame([[1, 2]])
    df.columns = MultiIndex.from_tuples([(1, 1), (2, 1)])
    
    # 将 DataFrame df 转换为 HTML 格式，设置列宽度为 100
    result = df.to_html(col_space=100)
    
    # 设置预期的 HTML 结果，包含特定的表头和数据行
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        "    <tr>\n"
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '      <th style="min-width: 100px;">2</th>\n'
        "    </tr>\n"
        "    <tr>\n"
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        '      <th style="min-width: 100px;">1</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>"
    )
    
    # 断言结果与预期是否相同
    assert result == expected


# 测试处理包含元组列，并设置列宽度的情况
def test_to_html_tuple_col_with_colspace():
    # 创建包含元组列的 DataFrame 对象 df
    df = DataFrame({("a", "b"): [1], "b": [2]})
    
    # 将 DataFrame df 转换为 HTML 格式，设置列宽度为 100
    result = df.to_html(col_space=100)
    
    # 设置预期的 HTML 结果，包含特定的表头和数据行
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">(a, b)</th>\n'
        '      <th style="min-width: 100px;">b</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "    <tr>\n"
        "      <th>0</th>\n"
        "      <td>1</td>\n"
        "      <td>2</td>\n"
        "    </tr>\n"
        "  </tbody>\n"
        "</table>"
    )
    
    # 断言结果与预期是否相同
    assert result == expected


# 测试处理空复杂数组的情况
def test_to_html_empty_complex_array():
    # 该测试函数尚未完成
    # 创建一个空的 DataFrame 对象，其中包含一个名为 'x' 的列，列数据类型为复数数组
    df = DataFrame({"x": np.array([], dtype="complex")})
    
    # 将 DataFrame 对象转换为 HTML 表格表示形式，并设置列之间的最小宽度为 100 像素
    result = df.to_html(col_space=100)
    
    # 期望的 HTML 表格字符串，包含表头和空的表体部分，表头中 'x' 列的最小宽度也设置为 100 像素
    expected = (
        '<table border="1" class="dataframe">\n'
        "  <thead>\n"
        '    <tr style="text-align: right;">\n'
        '      <th style="min-width: 100px;"></th>\n'
        '      <th style="min-width: 100px;">x</th>\n'
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        "  </tbody>\n"
        "</table>"
    )
    
    # 使用断言来验证生成的 HTML 表格字符串是否符合预期的格式
    assert result == expected
```