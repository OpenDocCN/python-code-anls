# `D:\src\scipysrc\pandas\pandas\tests\io\test_html.py`

```
# 导入必要的模块和库
from collections.abc import Iterator
from functools import partial
from io import (
    BytesIO,
    StringIO,
)
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError

import numpy as np
import pytest

# 导入 pandas 相关模块和函数
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
    NA,
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    read_csv,
    read_html,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,
    StringArray,
)
from pandas.io.common import file_path_to_url


# 参数化测试用例，用于 HTML 编码测试文件名
@pytest.fixture(
    params=[
        "chinese_utf-16.html",
        "chinese_utf-32.html",
        "chinese_utf-8.html",
        "letz_latin1.html",
    ]
)
def html_encoding_file(request, datapath):
    """Parametrized fixture for HTML encoding test filenames."""
    return datapath("io", "data", "html_encoding", request.param)


# 检查两个列表中的 DataFrame 是否相等
def assert_framelist_equal(list1, list2, *args, **kwargs):
    assert len(list1) == len(list2), (
        "lists are not of equal size "
        f"len(list1) == {len(list1)}, "
        f"len(list2) == {len(list2)}"
    )
    msg = "not all list elements are DataFrames"
    both_frames = all(
        map(
            lambda x, y: isinstance(x, DataFrame) and isinstance(y, DataFrame),
            list1,
            list2,
        )
    )
    assert both_frames, msg
    for frame_i, frame_j in zip(list1, list2):
        tm.assert_frame_equal(frame_i, frame_j, *args, **kwargs)
        assert not frame_i.empty, "frames are both empty"


# 测试当 BeautifulSoup4 版本不符合要求时的情况
def test_bs4_version_fails(monkeypatch, datapath):
    bs4 = pytest.importorskip("bs4")
    pytest.importorskip("html5lib")

    monkeypatch.setattr(bs4, "__version__", "4.2")
    with pytest.raises(ImportError, match="Pandas requires version"):
        read_html(datapath("io", "data", "html", "spam.html"), flavor="bs4")


# 测试不支持的 HTML 解析器 Flavor
def test_invalid_flavor():
    url = "google.com"
    flavor = "invalid flavor"
    msg = r"\{" + flavor + r"\} is not a valid set of flavors"

    with pytest.raises(ValueError, match=msg):
        read_html(StringIO(url), match="google", flavor=flavor)


# 测试使用相同参数的不同解析器解析 HTML 文件结果是否相同
def test_same_ordering(datapath):
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")
    pytest.importorskip("html5lib")

    filename = datapath("io", "data", "html", "valid_markup.html")
    dfs_lxml = read_html(filename, index_col=0, flavor=["lxml"])
    dfs_bs4 = read_html(filename, index_col=0, flavor=["bs4"])
    assert_framelist_equal(dfs_lxml, dfs_bs4)


# 参数化测试用例，用于指定不同的 HTML 解析器 Flavor
@pytest.fixture(
    params=[
        pytest.param("bs4", marks=[td.skip_if_no("bs4"), td.skip_if_no("html5lib")]),
        pytest.param("lxml", marks=td.skip_if_no("lxml")),
    ],
)
def flavor_read_html(request):
    return partial(read_html, flavor=request.param)


# 定义 TestReadHtml 类，用于存放与 HTML 解析相关的测试方法
class TestReadHtml:
    pass
    # 测试函数：检查文本HTML格式的解析函数是否正确处理文件未找到的异常
    def test_literal_html_deprecation(self, flavor_read_html):
        # 定义匹配的错误消息，用于验证抛出的异常类型和消息内容是否匹配
        msg = r"\[Errno 2\] No such file or director"

        # 使用 pytest 的上下文管理器，期望抛出 FileNotFoundError 异常，并验证其错误消息匹配指定的正则表达式 msg
        with pytest.raises(FileNotFoundError, match=msg):
            # 调用 flavor_read_html 函数，尝试解析以下HTML字符串
            flavor_read_html(
                """<table>
                <thead>
                    <tr>
                        <th>A</th>
                        <th>B</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1</td>
                        <td>2</td>
                    </tr>
                </tbody>
                <tbody>
                    <tr>
                        <td>3</td>
                        <td>4</td>
                    </tr>
                </tbody>
            </table>"""
            )

    # Pytest 的装置函数：提供HTML文件路径作为测试数据
    @pytest.fixture
    def spam_data(self, datapath):
        return datapath("io", "data", "html", "spam.html")

    # Pytest 的装置函数：提供HTML文件路径作为测试数据
    @pytest.fixture
    def banklist_data(self, datapath):
        return datapath("io", "data", "html", "banklist.html")

    # 测试函数：验证 DataFrame 转换为HTML字符串的兼容性
    def test_to_html_compat(self, flavor_read_html):
        # 创建一个随机数生成的 DataFrame
        df = (
            DataFrame(
                np.random.default_rng(2).random((4, 3)),
                columns=pd.Index(list("abc"), dtype=object),
            )
            # 将 DataFrame 中的数值转换为指定格式的字符串
            .map("{:.3f}".format)
            # 将字符串数据类型转换为浮点数
            .astype(float)
        )
        # 调用 DataFrame 的 to_html 方法，将 DataFrame 转换为HTML字符串
        out = df.to_html()
        # 调用 flavor_read_html 函数，尝试解析 StringIO 中的HTML字符串，指定额外的属性
        res = flavor_read_html(
            StringIO(out), attrs={"class": "dataframe"}, index_col=0
        )[0]
        # 使用测试工具 tm.assert_frame_equal 检查解析结果 res 是否与原始 DataFrame df 相等
        tm.assert_frame_equal(res, df)
    # 定义一个测试方法，用于测试不同的数据类型后端和存储方式
    def test_dtype_backend(self, string_storage, dtype_backend, flavor_read_html):
        # GH#50286: GitHub issue reference
        
        # 创建一个包含多列数据的 DataFrame 对象
        df = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),   # 创建 Int64 类型的 Series 列 'a'
                "b": Series([1, 2, 3], dtype="Int64"),        # 创建 Int64 类型的 Series 列 'b'
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),  # 创建 Float64 类型的 Series 列 'c'
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),     # 创建 Float64 类型的 Series 列 'd'
                "e": [True, False, None],   # 创建布尔类型的列表 'e'，包含 True、False 和 None
                "f": [True, False, True],   # 创建布尔类型的列表 'f'，包含 True、False 和 True
                "g": ["a", "b", "c"],       # 创建字符串类型的列表 'g'，包含 'a'、'b' 和 'c'
                "h": ["a", "b", None],      # 创建字符串类型的列表 'h'，包含 'a'、'b' 和 None
            }
        )

        # 根据 string_storage 的值选择不同的字符串存储方式
        if string_storage == "python":
            # 使用 Python 的字符串数组来创建 StringArray 对象，包含 ['a', 'b', 'c'] 和 ['a', 'b', NA]
            string_array = StringArray(np.array(["a", "b", "c"], dtype=np.object_))
            string_array_na = StringArray(np.array(["a", "b", NA], dtype=np.object_))
        elif dtype_backend == "pyarrow":
            # 如果使用 pyarrow 作为后端，则导入 pyarrow 库，并创建 ArrowExtensionArray 对象
            pa = pytest.importorskip("pyarrow")
            from pandas.arrays import ArrowExtensionArray
            string_array = ArrowExtensionArray(pa.array(["a", "b", "c"]))       # 使用 Arrow 扩展数组创建 'a', 'b', 'c' 字符串数组
            string_array_na = ArrowExtensionArray(pa.array(["a", "b", None]))   # 使用 Arrow 扩展数组创建 'a', 'b', None 字符串数组
        else:
            # 如果未选择 Python 或者 pyarrow，则默认使用 pyarrow 创建 ArrowStringArray 对象
            pa = pytest.importorskip("pyarrow")
            string_array = ArrowStringArray(pa.array(["a", "b", "c"]))    # 使用 ArrowStringArray 创建 'a', 'b', 'c' 字符串数组
            string_array_na = ArrowStringArray(pa.array(["a", "b", None]))  # 使用 ArrowStringArray 创建 'a', 'b', None 字符串数组

        # 将 DataFrame 转换为 HTML 字符串，并去除索引列
        out = df.to_html(index=False)
        
        # 设置 pandas 的上下文选项，调整字符串存储方式
        with pd.option_context("mode.string_storage", string_storage):
            # 使用 flavor_read_html 函数解析 HTML 字符串，获取解析结果的第一个元素
            result = flavor_read_html(StringIO(out), dtype_backend=dtype_backend)[0]

        # 创建期望的 DataFrame 对象，与解析结果进行比较
        expected = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),    # 创建期望的 Int64 类型的 Series 列 'a'
                "b": Series([1, 2, 3], dtype="Int64"),         # 创建期望的 Int64 类型的 Series 列 'b'
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),  # 创建期望的 Float64 类型的 Series 列 'c'
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),     # 创建期望的 Float64 类型的 Series 列 'd'
                "e": Series([True, False, NA], dtype="boolean"),   # 创建期望的布尔类型的 Series 列 'e'
                "f": Series([True, False, True], dtype="boolean"),  # 创建期望的布尔类型的 Series 列 'f'
                "g": string_array,   # 期望的字符串数组 'g'
                "h": string_array_na,   # 期望的字符串数组 'h'
            }
        )

        # 如果 dtype_backend 是 "pyarrow"，则进一步处理期望的 DataFrame
        if dtype_backend == "pyarrow":
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray
            
            # 将每列数据转换为 ArrowExtensionArray 类型
            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                    for col in expected.columns
                }
            )

        # 使用 pytest 框架的 assert_frame_equal 方法比较解析结果和期望结果是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.network
    @pytest.mark.single_cpu
    # 测试 banklist_url 方法，在 httpserver 上使用 banklist_data 和 flavor_read_html 进行测试
    def test_banklist_url(self, httpserver, banklist_data, flavor_read_html):
        # 打开 banklist_data 文件，使用 UTF-8 编码
        with open(banklist_data, encoding="utf-8") as f:
            # 将文件内容作为 http 响应内容返回给 httpserver
            httpserver.serve_content(content=f.read())
            # 使用 flavor_read_html 从 httpserver.url 读取匹配 "First Federal Bank of Florida" 的数据
            df1 = flavor_read_html(
                httpserver.url,
                match="First Federal Bank of Florida",  # attrs={"class": "dataTable"}
            )
            # 使用 flavor_read_html 从 httpserver.url 读取匹配 "Metcalf Bank" 的数据
            df2 = flavor_read_html(
                httpserver.url,
                match="Metcalf Bank",
            )  # attrs={"class": "dataTable"})

        # 断言 df1 和 df2 的数据框是否相等
        assert_framelist_equal(df1, df2)

    # 标记为网络测试和单 CPU 测试
    @pytest.mark.network
    @pytest.mark.single_cpu
    # 测试 spam_url 方法，在 httpserver 上使用 spam_data 和 flavor_read_html 进行测试
    def test_spam_url(self, httpserver, spam_data, flavor_read_html):
        # 打开 spam_data 文件，使用 UTF-8 编码
        with open(spam_data, encoding="utf-8") as f:
            # 将文件内容作为 http 响应内容返回给 httpserver
            httpserver.serve_content(content=f.read())
            # 使用 flavor_read_html 从 httpserver.url 读取匹配 ".*Water.*" 的数据
            df1 = flavor_read_html(httpserver.url, match=".*Water.*")
            # 使用 flavor_read_html 从 httpserver.url 读取匹配 "Unit" 的数据
            df2 = flavor_read_html(httpserver.url, match="Unit")

        # 断言 df1 和 df2 的数据框是否相等
        assert_framelist_equal(df1, df2)

    # 标记为慢速测试
    @pytest.mark.slow
    # 测试 banklist 方法，使用 banklist_data 和 flavor_read_html 进行测试
    def test_banklist(self, banklist_data, flavor_read_html):
        # 使用 flavor_read_html 从 banklist_data 读取匹配 ".*Florida.*" 的数据，且属性为 {"id": "table"}
        df1 = flavor_read_html(
            banklist_data, match=".*Florida.*", attrs={"id": "table"}
        )
        # 使用 flavor_read_html 从 banklist_data 读取匹配 "Metcalf Bank" 的数据，且属性为 {"id": "table"}
        df2 = flavor_read_html(
            banklist_data, match="Metcalf Bank", attrs={"id": "table"}
        )

        # 断言 df1 和 df2 的数据框是否相等
        assert_framelist_equal(df1, df2)

    # 测试 spam 方法，使用 spam_data 和 flavor_read_html 进行测试
    def test_spam(self, spam_data, flavor_read_html):
        # 使用 flavor_read_html 从 spam_data 读取匹配 ".*Water.*" 的数据
        df1 = flavor_read_html(spam_data, match=".*Water.*")
        # 使用 flavor_read_html 从 spam_data 读取匹配 "Unit" 的数据
        df2 = flavor_read_html(spam_data, match="Unit")
        
        # 断言 df1 和 df2 的数据框是否相等
        assert_framelist_equal(df1, df2)

        # 断言 df1 第一个元素的第一行第一列是否为 "Proximates"
        assert df1[0].iloc[0, 0] == "Proximates"
        # 断言 df1 的第一个数据框的第一列是否为 "Nutrient"
        assert df1[0].columns[0] == "Nutrient"

    # 测试 spam_no_match 方法，使用 spam_data 和 flavor_read_html 进行测试
    def test_spam_no_match(self, spam_data, flavor_read_html):
        # 使用 flavor_read_html 从 spam_data 读取所有匹配的数据框
        dfs = flavor_read_html(spam_data)
        # 遍历每个数据框 df，断言其是否为 DataFrame 类型
        for df in dfs:
            assert isinstance(df, DataFrame)

    # 测试 banklist_no_match 方法，使用 banklist_data 和 flavor_read_html 进行测试
    def test_banklist_no_match(self, banklist_data, flavor_read_html):
        # 使用 flavor_read_html 从 banklist_data 读取所有匹配的数据框，且属性为 {"id": "table"}
        dfs = flavor_read_html(banklist_data, attrs={"id": "table"})
        # 遍历每个数据框 df，断言其是否为 DataFrame 类型
        for df in dfs:
            assert isinstance(df, DataFrame)

    # 测试 spam_header 方法，使用 spam_data 和 flavor_read_html 进行测试
    def test_spam_header(self, spam_data, flavor_read_html):
        # 使用 flavor_read_html 从 spam_data 读取匹配 ".*Water.*" 的数据，且跳过前两行
        df = flavor_read_html(spam_data, match=".*Water.*", header=2)[0]
        # 断言 df 的第一列是否为 "Proximates"
        assert df.columns[0] == "Proximates"
        # 断言 df 是否不为空
        assert not df.empty

    # 测试 skiprows_int 方法，使用 spam_data 和 flavor_read_html 进行测试
    def test_skiprows_int(self, spam_data, flavor_read_html):
        # 使用 flavor_read_html 从 spam_data 读取匹配 ".*Water.*" 的数据，且跳过第一行
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=1)
        # 使用 flavor_read_html 从 spam_data 读取匹配 "Unit" 的数据，且跳过第一行
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=1)

        # 断言 df1 和 df2 的数据框是否相等
        assert_framelist_equal(df1, df2)

    # 测试 skiprows_range 方法，使用 spam_data 和 flavor_read_html 进行测试
    def test_skiprows_range(self, spam_data, flavor_read_html):
        # 使用 flavor_read_html 从 spam_data 读取匹配 ".*Water.*" 的数据，且跳过前两行
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=range(2))
        # 使用 flavor_read_html 从 spam_data 读取匹配 "Unit" 的数据，且跳过前两行
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=range(2))

        # 断言 df1 和 df2 的数据框是否相等
        assert_framelist_equal(df1, df2)
    # 测试函数，使用指定的数据源和HTML解析器函数读取数据，根据匹配条件和跳过行的参数创建数据框
    def test_skiprows_list(self, spam_data, flavor_read_html):
        # 使用 flavor_read_html 函数从 spam_data 中读取匹配 .*Water.* 的数据，跳过行号为 [1, 2] 的行
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=[1, 2])
        # 使用 flavor_read_html 函数从 spam_data 中读取匹配 Unit 的数据，跳过行号为 [2, 1] 的行
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=[2, 1])

        # 断言 df1 和 df2 的数据框内容相等
        assert_framelist_equal(df1, df2)

    # 测试函数，使用集合作为跳过行的参数
    def test_skiprows_set(self, spam_data, flavor_read_html):
        # 使用 flavor_read_html 函数从 spam_data 中读取匹配 .*Water.* 的数据，跳过行号为 {1, 2} 的行
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows={1, 2})
        # 使用 flavor_read_html 函数从 spam_data 中读取匹配 Unit 的数据，跳过行号为 {2, 1} 的行
        df2 = flavor_read_html(spam_data, match="Unit", skiprows={2, 1})

        # 断言 df1 和 df2 的数据框内容相等
        assert_framelist_equal(df1, df2)

    # 测试函数，使用切片作为跳过行的参数
    def test_skiprows_slice(self, spam_data, flavor_read_html):
        # 使用 flavor_read_html 函数从 spam_data 中读取匹配 .*Water.* 的数据，跳过第 1 行之前的所有行
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=1)
        # 使用 flavor_read_html 函数从 spam_data 中读取匹配 Unit 的数据，跳过第 1 行之前的所有行
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=1)

        # 断言 df1 和 df2 的数据框内容相等
        assert_framelist_equal(df1, df2)

    # 测试函数，使用切片对象作为跳过行的参数，跳过前 2 行
    def test_skiprows_slice_short(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=slice(2))
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=slice(2))

        assert_framelist_equal(df1, df2)

    # 测试函数，使用切片对象作为跳过行的参数，跳过第 2 至 5 行
    def test_skiprows_slice_long(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=slice(2, 5))
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=slice(4, 1, -1))

        assert_framelist_equal(df1, df2)

    # 测试函数，使用 ndarray 作为跳过行的参数，跳过前 2 行
    def test_skiprows_ndarray(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", skiprows=np.arange(2))
        df2 = flavor_read_html(spam_data, match="Unit", skiprows=np.arange(2))

        assert_framelist_equal(df1, df2)

    # 测试函数，使用非法类型作为跳过行的参数，应该引发 TypeError 异常
    def test_skiprows_invalid(self, spam_data, flavor_read_html):
        with pytest.raises(TypeError, match=("is not a valid type for skipping rows")):
            flavor_read_html(spam_data, match=".*Water.*", skiprows="asdf")

    # 测试函数，使用指定的数据源和HTML解析器函数读取数据，并设置索引列为第一列
    def test_index(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", index_col=0)
        df2 = flavor_read_html(spam_data, match="Unit", index_col=0)
        assert_framelist_equal(df1, df2)

    # 测试函数，使用指定的数据源和HTML解析器函数读取数据，设置头部行为第一行，并将第一列作为索引列
    def test_header_and_index_no_types(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", header=1, index_col=0)
        df2 = flavor_read_html(spam_data, match="Unit", header=1, index_col=0)
        assert_framelist_equal(df1, df2)

    # 测试函数，使用指定的数据源和HTML解析器函数读取数据，设置头部行为第一行，并将第一列作为索引列
    def test_header_and_index_with_types(self, spam_data, flavor_read_html):
        df1 = flavor_read_html(spam_data, match=".*Water.*", header=1, index_col=0)
        df2 = flavor_read_html(spam_data, match="Unit", header=1, index_col=0)
        assert_framelist_equal(df1, df2)

    # 测试函数，使用指定的数据源和HTML解析器函数读取数据，并推断索引列类型
    def test_infer_types(self, spam_data, flavor_read_html):
        # 10892 infer_types removed
        df1 = flavor_read_html(spam_data, match=".*Water.*", index_col=0)
        df2 = flavor_read_html(spam_data, match="Unit", index_col=0)
        assert_framelist_equal(df1, df2)
    @pytest.mark.slow
    def test_string_io(self, spam_data, flavor_read_html):
        # 打开文件并将其内容封装到StringIO对象中
        with open(spam_data, encoding="UTF-8") as f:
            data1 = StringIO(f.read())

        # 再次打开文件并将其内容封装到另一个StringIO对象中
        with open(spam_data, encoding="UTF-8") as f:
            data2 = StringIO(f.read())

        # 使用flavor_read_html函数处理data1，匹配包含".*Water.*"的内容，返回DataFrame对象df1
        df1 = flavor_read_html(data1, match=".*Water.*")
        # 使用flavor_read_html函数处理data2，匹配包含"Unit"的内容，返回DataFrame对象df2
        df2 = flavor_read_html(data2, match="Unit")
        # 断言df1和df2的DataFrame对象相等
        assert_framelist_equal(df1, df2)

    def test_string(self, spam_data, flavor_read_html):
        # 打开文件并读取其内容到data字符串中
        with open(spam_data, encoding="UTF-8") as f:
            data = f.read()

        # 使用flavor_read_html函数处理StringIO对象（包含data内容），匹配包含".*Water.*"的内容，返回DataFrame对象df1
        df1 = flavor_read_html(StringIO(data), match=".*Water.*")
        # 使用flavor_read_html函数处理StringIO对象（包含data内容），匹配包含"Unit"的内容，返回DataFrame对象df2
        df2 = flavor_read_html(StringIO(data), match="Unit")

        # 断言df1和df2的DataFrame对象相等
        assert_framelist_equal(df1, df2)

    def test_file_like(self, spam_data, flavor_read_html):
        # 打开文件并直接使用flavor_read_html处理文件对象f，匹配包含".*Water.*"的内容，返回DataFrame对象df1
        with open(spam_data, encoding="UTF-8") as f:
            df1 = flavor_read_html(f, match=".*Water.*")

        # 再次打开文件并使用flavor_read_html处理文件对象f，匹配包含"Unit"的内容，返回DataFrame对象df2
        with open(spam_data, encoding="UTF-8") as f:
            df2 = flavor_read_html(f, match="Unit")

        # 断言df1和df2的DataFrame对象相等
        assert_framelist_equal(df1, df2)

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_bad_url_protocol(self, httpserver, flavor_read_html):
        # 设置HTTP服务器响应内容和状态码
        httpserver.serve_content("urlopen error unknown url type: git", code=404)
        
        # 使用flavor_read_html函数尝试访问非法的URL，预期抛出URLError并匹配指定的错误信息
        with pytest.raises(URLError, match="urlopen error unknown url type: git"):
            flavor_read_html("git://github.com", match=".*Water.*")

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_invalid_url(self, httpserver, flavor_read_html):
        # 设置HTTP服务器响应内容和状态码
        httpserver.serve_content("Name or service not known", code=404)
        
        # 使用flavor_read_html函数尝试访问无效的URL，预期抛出URLError或ValueError并匹配指定的错误信息
        with pytest.raises((URLError, ValueError), match="HTTP Error 404: NOT FOUND"):
            flavor_read_html(httpserver.url, match=".*Water.*")

    @pytest.mark.slow
    def test_file_url(self, banklist_data, flavor_read_html):
        # 获取银行列表数据的URL
        url = banklist_data
        
        # 使用flavor_read_html函数处理文件路径对应的URL，匹配包含"First"的内容，返回DataFrame对象的列表dfs
        dfs = flavor_read_html(
            file_path_to_url(os.path.abspath(url)), match="First", attrs={"id": "table"}
        )
        
        # 断言dfs是一个列表，并且列表中的每个元素都是DataFrame对象
        assert isinstance(dfs, list)
        for df in dfs:
            assert isinstance(df, DataFrame)

    @pytest.mark.slow
    def test_invalid_table_attrs(self, banklist_data, flavor_read_html):
        # 获取银行列表数据的URL
        url = banklist_data
        
        # 使用flavor_read_html函数尝试处理URL，匹配包含"First Federal Bank of Florida"的内容，
        # 但指定的表格属性"id"为"tasdfable"是无效的，预期抛出ValueError并匹配指定的错误信息
        with pytest.raises(ValueError, match="No tables found"):
            flavor_read_html(
                url, match="First Federal Bank of Florida", attrs={"id": "tasdfable"}
            )

    @pytest.mark.slow
    def test_multiindex_header(self, banklist_data, flavor_read_html):
        # 使用flavor_read_html函数处理银行列表数据，匹配包含"Metcalf"的内容，指定表格属性"id"为"table"，
        # 并且指定表头为多级索引[0, 1]，返回DataFrame对象的列表，取第一个元素df
        df = flavor_read_html(
            banklist_data, match="Metcalf", attrs={"id": "table"}, header=[0, 1]
        )[0]
        
        # 断言df的列是一个MultiIndex对象
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    def test_multiindex_index(self, banklist_data, flavor_read_html):
        # 使用flavor_read_html函数处理银行列表数据，匹配包含"Metcalf"的内容，指定表格属性"id"为"table"，
        # 并且指定索引为多级索引[0, 1]，返回DataFrame对象的列表，取第一个元素df
        df = flavor_read_html(
            banklist_data, match="Metcalf", attrs={"id": "table"}, index_col=[0, 1]
        )[0]
        
        # 断言df的行索引是一个MultiIndex对象
        assert isinstance(df.index, MultiIndex)
    # 测试函数，验证读取 HTML 表格数据并处理为 DataFrame，要求使用复合索引作为列名和行索引
    def test_multiindex_header_index(self, banklist_data, flavor_read_html):
        # 调用读取 HTML 表格数据的函数，匹配特定内容"Metcalf"，指定id属性为"table"
        df = flavor_read_html(
            banklist_data,
            match="Metcalf",
            attrs={"id": "table"},
            # 指定表格的复合标题行作为列名和行索引
            header=[0, 1],
            index_col=[0, 1],
        )[0]
        # 断言列名为 MultiIndex 类型
        assert isinstance(df.columns, MultiIndex)
        # 断言行索引为 MultiIndex 类型
        assert isinstance(df.index, MultiIndex)

    @pytest.mark.slow
    # 测试函数，验证读取 HTML 表格数据并处理为 DataFrame，要求使用复合索引作为列名，跳过首行
    def test_multiindex_header_skiprows_tuples(self, banklist_data, flavor_read_html):
        # 调用读取 HTML 表格数据的函数，匹配特定内容"Metcalf"，指定id属性为"table"
        df = flavor_read_html(
            banklist_data,
            match="Metcalf",
            attrs={"id": "table"},
            # 指定表格的复合标题行作为列名，跳过首行
            header=[0, 1],
            skiprows=1,
        )[0]
        # 断言列名为 MultiIndex 类型
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    # 测试函数，验证读取 HTML 表格数据并处理为 DataFrame，要求使用复合索引作为列名，跳过首行
    def test_multiindex_header_skiprows(self, banklist_data, flavor_read_html):
        # 调用读取 HTML 表格数据的函数，匹配特定内容"Metcalf"，指定id属性为"table"
        df = flavor_read_html(
            banklist_data,
            match="Metcalf",
            attrs={"id": "table"},
            # 指定表格的复合标题行作为列名，跳过首行
            header=[0, 1],
            skiprows=1,
        )[0]
        # 断言列名为 MultiIndex 类型
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    # 测试函数，验证读取 HTML 表格数据并处理为 DataFrame，要求使用复合索引作为列名和行索引，跳过首行
    def test_multiindex_header_index_skiprows(self, banklist_data, flavor_read_html):
        # 调用读取 HTML 表格数据的函数，匹配特定内容"Metcalf"，指定id属性为"table"
        df = flavor_read_html(
            banklist_data,
            match="Metcalf",
            attrs={"id": "table"},
            # 指定表格的复合标题行作为列名和行索引，跳过首行
            header=[0, 1],
            index_col=[0, 1],
            skiprows=1,
        )[0]
        # 断言行索引为 MultiIndex 类型
        assert isinstance(df.index, MultiIndex)
        # 断言列名为 MultiIndex 类型
        assert isinstance(df.columns, MultiIndex)

    @pytest.mark.slow
    # 测试函数，验证读取 HTML 表格数据并处理为 DataFrame，使用正则表达式匹配 URL 中包含"Florida"的内容
    def test_regex_idempotency(self, banklist_data, flavor_read_html):
        # 获取银行列表数据的 URL
        url = banklist_data
        # 转换文件路径为 URL，读取 HTML 表格数据，并使用正则表达式匹配表格id为"table"
        dfs = flavor_read_html(
            file_path_to_url(os.path.abspath(url)),
            match=re.compile(re.compile("Florida")),
            attrs={"id": "table"},
        )
        # 断言返回的数据为列表类型
        assert isinstance(dfs, list)
        # 验证列表中的每个元素是 DataFrame 类型
        for df in dfs:
            assert isinstance(df, DataFrame)

    # 测试函数，验证传递负值给 skiprows 参数时是否引发 ValueError 异常
    def test_negative_skiprows(self, spam_data, flavor_read_html):
        # 定义异常消息
        msg = r"\(you passed a negative value\)"
        # 使用 pytest 断言检查是否引发 ValueError 异常，消息匹配预期的异常信息
        with pytest.raises(ValueError, match=msg):
            flavor_read_html(spam_data, match="Water", skiprows=-1)

    @pytest.fixture
    @pytest.mark.network
    @pytest.mark.single_cpu
    # 测试函数，验证读取 HTML 表格数据并处理为 DataFrame，匹配多个"Python"关键字
    def test_multiple_matches(self, python_docs, httpserver, flavor_read_html):
        # 启动 HTTP 服务器，提供 Python 文档内容
        httpserver.serve_content(content=python_docs)
        # 读取 HTTP 服务器上的内容，匹配包含"Python"关键字的表格
        dfs = flavor_read_html(httpserver.url, match="Python")
        # 断言返回的 DataFrame 列表长度大于 1
        assert len(dfs) > 1

    @pytest.mark.network
    @pytest.mark.single_cpu
    # 测试函数，验证读取 HTTP 服务器上 Python 文档中包含"Python"关键字的表格内容
    def test_python_docs_table(self, python_docs, httpserver, flavor_read_html):
        # 启动 HTTP 服务器，提供 Python 文档内容
        httpserver.serve_content(content=python_docs)
        # 读取 HTTP 服务器上的内容，匹配包含"Python"关键字的表格
        dfs = flavor_read_html(httpserver.url, match="Python")
        # 提取每个 DataFrame 的第一行第一列的前四个字符
        zz = [df.iloc[0, 0][0:4] for df in dfs]
        # 断言提取的字符列表按字母顺序排列为["Pyth", "What"]
        assert sorted(zz) == ["Pyth", "What"]
    def test_empty_tables(self, flavor_read_html):
        """
        Make sure that read_html ignores empty tables.
        """
        html = """
            <table>
                <thead>
                    <tr>
                        <th>A</th>
                        <th>B</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1</td>
                        <td>2</td>
                    </tr>
                </tbody>
            </table>
            <table>
                <tbody>
                </tbody>
            </table>
        """
        # 调用 flavor_read_html 函数，解析 HTML 字符串，忽略空表格，返回结果
        result = flavor_read_html(StringIO(html))
        # 断言结果列表长度为1，确保只解析了非空表格
        assert len(result) == 1

    def test_multiple_tbody(self, flavor_read_html):
        # GH-20690
        # Read all tbody tags within a single table.
        # 调用 flavor_read_html 函数，解析包含多个 tbody 的 HTML 字符串，返回结果列表
        result = flavor_read_html(
            StringIO(
                """<table>
            <thead>
                <tr>
                    <th>A</th>
                    <th>B</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>1</td>
                    <td>2</td>
                </tr>
            </tbody>
            <tbody>
                <tr>
                    <td>3</td>
                    <td>4</td>
                </tr>
            </tbody>
        </table>"""
            )
        )[0]

        # 期望的 DataFrame 结果，包含两行数据，列为 ["A", "B"]
        expected = DataFrame(data=[[1, 2], [3, 4]], columns=["A", "B"])

        # 断言解析的 DataFrame 结果与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_header_and_one_column(self, flavor_read_html):
        """
        Don't fail with bs4 when there is a header and only one column
        as described in issue #9178
        """
        # 调用 flavor_read_html 函数，解析包含表头和单列的 HTML 字符串，返回结果列表中的第一个 DataFrame
        result = flavor_read_html(
            StringIO(
                """<table>
                <thead>
                    <tr>
                        <th>Header</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>first</td>
                    </tr>
                </tbody>
            </table>"""
            )
        )[0]

        # 期望的 DataFrame 结果，包含单列 "Header"，值为 "first"，索引为 [0]
        expected = DataFrame(data={"Header": "first"}, index=[0])

        # 断言解析的 DataFrame 结果与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    def test_thead_without_tr(self, flavor_read_html):
        """
        Ensure parser adds <tr> within <thead> on malformed HTML.
        """
        # 构造包含缺失 <tr> 的 <thead> 的 HTML 表格，并使用指定的 flavor_read_html 进行解析
        result = flavor_read_html(
            StringIO(
                """<table>
            <thead>
                <tr>
                    <th>Country</th>
                    <th>Municipality</th>
                    <th>Year</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Ukraine</td>
                    <th>Odessa</th>
                    <td>1944</td>
                </tr>
            </tbody>
        </table>"""
            )
        )[0]

        # 期望的 DataFrame 结果，表示正确解析的表格内容
        expected = DataFrame(
            data=[["Ukraine", "Odessa", 1944]],
            columns=["Country", "Municipality", "Year"],
        )

        # 断言解析结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    def test_tfoot_read(self, flavor_read_html):
        """
        Make sure that read_html reads tfoot, containing td or th.
        Ignores empty tfoot
        """
        # HTML 模板，包含 <tfoot> 部分，用于测试读取带和不带内容的 tfoot 的情况
        data_template = """<table>
            <thead>
                <tr>
                    <th>A</th>
                    <th>B</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>bodyA</td>
                    <td>bodyB</td>
                </tr>
            </tbody>
            <tfoot>
                {footer}
            </tfoot>
        </table>"""

        # 期望的 DataFrame 结果，表示没有 <tfoot> 的情况
        expected1 = DataFrame(data=[["bodyA", "bodyB"]], columns=["A", "B"])

        # 期望的 DataFrame 结果，表示带有 <tfoot> 的情况
        expected2 = DataFrame(
            data=[["bodyA", "bodyB"], ["footA", "footB"]], columns=["A", "B"]
        )

        # 构造两种不同的 HTML 数据，一种不含 <tfoot>，一种含有特定的 <tfoot>
        data1 = data_template.format(footer="")
        data2 = data_template.format(footer="<tr><td>footA</td><th>footB</th></tr>")

        # 使用指定的 flavor_read_html 进行解析，并分别断言结果与期望相等
        result1 = flavor_read_html(StringIO(data1))[0]
        result2 = flavor_read_html(StringIO(data2))[0]

        tm.assert_frame_equal(result1, expected1)
        tm.assert_frame_equal(result2, expected2)

    def test_parse_header_of_non_string_column(self, flavor_read_html):
        # GH5048: if header is specified explicitly, an int column should be
        # parsed as int while its header is parsed as str
        # 构造包含非字符串列的表格，确保在指定 header=0 的情况下正确解析
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td>S</td>
                    <td>I</td>
                </tr>
                <tr>
                    <td>text</td>
                    <td>1944</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        # 期望的 DataFrame 结果，显示解析出的列类型和列名
        expected = DataFrame([["text", 1944]], columns=("S", "I"))

        # 断言解析结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.slow
    # 定义测试方法，验证银行列表数据的表头
    def test_banklist_header(self, banklist_data, datapath, flavor_read_html):
        # 导入私有函数_remove_whitespace，用于尝试移除字符串中的空白字符
        from pandas.io.html import _remove_whitespace

        # 定义尝试移除空白字符的函数try_remove_ws，处理可能的AttributeError异常
        def try_remove_ws(x):
            try:
                return _remove_whitespace(x)
            except AttributeError:
                return x

        # 使用flavor_read_html函数从banklist_data中读取包含"Metcalf"的第一个表格作为DataFrame df
        df = flavor_read_html(banklist_data, match="Metcalf", attrs={"id": "table"})[0]
        
        # 从datapath中读取CSV文件'banklist.csv'，将其转换为DataFrame ground_truth
        ground_truth = read_csv(
            datapath("io", "data", "csv", "banklist.csv"),
            converters={"Updated Date": Timestamp, "Closing Date": Timestamp},
        )
        
        # 断言df的形状与ground_truth的形状相同
        assert df.shape == ground_truth.shape
        
        # 定义旧银行名称列表old和新银行名称列表new，用于映射旧名称到新名称
        old = [
            "First Vietnamese American Bank In Vietnamese",
            "Westernbank Puerto Rico En Espanol",
            "R-G Premier Bank of Puerto Rico En Espanol",
            "Eurobank En Espanol",
            "Sanderson State Bank En Espanol",
            "Washington Mutual Bank (Including its subsidiary Washington "
            "Mutual Bank FSB)",
            "Silver State Bank En Espanol",
            "AmTrade International Bank En Espanol",
            "Hamilton Bank, NA En Espanol",
            "The Citizens Savings Bank Pioneer Community Bank, Inc.",
        ]
        new = [
            "First Vietnamese American Bank",
            "Westernbank Puerto Rico",
            "R-G Premier Bank of Puerto Rico",
            "Eurobank",
            "Sanderson State Bank",
            "Washington Mutual Bank",
            "Silver State Bank",
            "AmTrade International Bank",
            "Hamilton Bank, NA",
            "The Citizens Savings Bank",
        ]
        
        # 使用try_remove_ws函数，将df中的旧银行名称替换为新银行名称，生成新的DataFrame dfnew
        dfnew = df.map(try_remove_ws).replace(old, new)
        
        # 使用try_remove_ws函数，移除ground_truth中的空白字符，生成新的DataFrame gtnew
        gtnew = ground_truth.map(try_remove_ws)
        
        # 将dfnew中的日期列"Closing Date"和"Updated Date"转换为datetime格式
        converted = dfnew
        date_cols = ["Closing Date", "Updated Date"]
        converted[date_cols] = converted[date_cols].apply(to_datetime)
        
        # 断言转换后的DataFrame converted与gtnew相等
        tm.assert_frame_equal(converted, gtnew)

    # 标记为慢速测试
    @pytest.mark.slow
    # 定义测试Gold Canyon是否在银行列表数据中的方法
    def test_gold_canyon(self, banklist_data, flavor_read_html):
        gc = "Gold Canyon"
        
        # 使用UTF-8编码打开banklist_data文件，读取其内容为raw_text
        with open(banklist_data, encoding="utf-8") as f:
            raw_text = f.read()

        # 断言raw_text中包含字符串"Gold Canyon"
        assert gc in raw_text
        
        # 使用flavor_read_html函数从banklist_data中读取包含"Gold Canyon"的第一个表格作为DataFrame df
        df = flavor_read_html(
            banklist_data, match="Gold Canyon", attrs={"id": "table"}
        )[0]
        
        # 断言df.to_string()中包含字符串"Gold Canyon"
        assert gc in df.to_string()
    # 定义一个测试方法，用于测试读取 HTML 表格的不同列数情况
    def test_different_number_of_cols(self, flavor_read_html):
        # 定义预期结果，调用 flavor_read_html 函数读取第一个表格内容并返回
        expected = flavor_read_html(
            StringIO(
                """<table>
                        <thead>
                            <tr style="text-align: right;">
                            <th></th>
                            <th>C_l0_g0</th>
                            <th>C_l0_g1</th>
                            <th>C_l0_g2</th>
                            <th>C_l0_g3</th>
                            <th>C_l0_g4</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                            <th>R_l0_g0</th>
                            <td> 0.763</td>
                            <td> 0.233</td>
                            <td> nan</td>
                            <td> nan</td>
                            <td> nan</td>
                            </tr>
                            <tr>
                            <th>R_l0_g1</th>
                            <td> 0.244</td>
                            <td> 0.285</td>
                            <td> 0.392</td>
                            <td> 0.137</td>
                            <td> 0.222</td>
                            </tr>
                        </tbody>
                    </table>"""
            ),
            index_col=0,
        )[0]

        # 定义实际结果，调用 flavor_read_html 函数读取第二个表格内容并返回
        result = flavor_read_html(
            StringIO(
                """<table>
                    <thead>
                        <tr style="text-align: right;">
                        <th></th>
                        <th>C_l0_g0</th>
                        <th>C_l0_g1</th>
                        <th>C_l0_g2</th>
                        <th>C_l0_g3</th>
                        <th>C_l0_g4</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                        <th>R_l0_g0</th>
                        <td> 0.763</td>
                        <td> 0.233</td>
                        </tr>
                        <tr>
                        <th>R_l0_g1</th>
                        <td> 0.244</td>
                        <td> 0.285</td>
                        <td> 0.392</td>
                        <td> 0.137</td>
                        <td> 0.222</td>
                        </tr>
                    </tbody>
                 </table>"""
            ),
            index_col=0,
        )[0]

        # 使用测试工具比较实际结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)
    # 测试表格的合并单元格（colspan）和跨行单元格（rowspan）的情况
    def test_colspan_rowspan_1(self, flavor_read_html):
        # GH17054
        
        # 使用给定的 HTML 内容创建一个表格，并通过 flavor_read_html 解析得到结果
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <th>A</th>
                    <th colspan="1">B</th>
                    <th rowspan="1">C</th>
                </tr>
                <tr>
                    <td>a</td>
                    <td>b</td>
                    <td>c</td>
                </tr>
            </table>
        """
            )
        )[0]

        # 创建预期的 DataFrame，包含一行数据["a", "b", "c"]，列名为["A", "B", "C"]
        expected = DataFrame([["a", "b", "c"]], columns=["A", "B", "C"])

        # 使用 pandas 的 assert_frame_equal 函数比较结果和预期，确认它们是否相同
        tm.assert_frame_equal(result, expected)

    # 测试表格中的合并单元格（colspan）和复制值的情况
    def test_colspan_rowspan_copy_values(self, flavor_read_html):
        # GH17054

        # 创建包含合并单元格和跨行单元格的表格，并通过 flavor_read_html 解析得到结果
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td colspan="2">X</td>
                    <td>Y</td>
                    <td rowspan="2">Z</td>
                    <td>W</td>
                </tr>
                <tr>
                    <td>A</td>
                    <td colspan="2">B</td>
                    <td>C</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        # 创建预期的 DataFrame，包含一行数据["A", "B", "B", "Z", "C"]，列名为["X", "X.1", "Y", "Z", "W"]
        expected = DataFrame(
            data=[["A", "B", "B", "Z", "C"]], columns=["X", "X.1", "Y", "Z", "W"]
        )

        # 使用 pandas 的 assert_frame_equal 函数比较结果和预期，确认它们是否相同
        tm.assert_frame_equal(result, expected)

    # 测试表格中同时存在合并单元格和跨行单元格的情况
    def test_colspan_rowspan_both_not_1(self, flavor_read_html):
        # GH17054

        # 创建包含多个合并单元格和跨行单元格的表格，并通过 flavor_read_html 解析得到结果
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td rowspan="2">A</td>
                    <td rowspan="2" colspan="3">B</td>
                    <td>C</td>
                </tr>
                <tr>
                    <td>D</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        # 创建预期的 DataFrame，包含一行数据["A", "B", "B", "B", "D"]，列名为["A", "B", "B.1", "B.2", "C"]
        expected = DataFrame(
            data=[["A", "B", "B", "B", "D"]], columns=["A", "B", "B.1", "B.2", "C"]
        )

        # 使用 pandas 的 assert_frame_equal 函数比较结果和预期，确认它们是否相同
        tm.assert_frame_equal(result, expected)

    # 测试表格中跨行单元格位于行尾的情况
    def test_rowspan_at_end_of_row(self, flavor_read_html):
        # GH17054

        # 创建包含跨行单元格位于行尾的表格，并通过 flavor_read_html 解析得到结果
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td>A</td>
                    <td rowspan="2">B</td>
                </tr>
                <tr>
                    <td>C</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        # 创建预期的 DataFrame，包含一行数据["C", "B"]，列名为["A", "B"]
        expected = DataFrame(data=[["C", "B"]], columns=["A", "B"])

        # 使用 pandas 的 assert_frame_equal 函数比较结果和预期，确认它们是否相同
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，测试仅包含行合并的表格情况
    def test_rowspan_only_rows(self, flavor_read_html):
        # GH17054

        # 使用指定的HTML内容进行表格解析，返回结果的第一个DataFrame
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <td rowspan="3">A</td>
                    <td rowspan="3">B</td>
                </tr>
            </table>
        """
            ),
            header=0,
        )[0]

        # 期望的结果DataFrame，包含两行数据 ["A", "B"], ["A", "B"]，列名为 ["A", "B"]
        expected = DataFrame(data=[["A", "B"], ["A", "B"]], columns=["A", "B"])

        # 断言结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试从行中的<th>标签推断表头的情况
    def test_header_inferred_from_rows_with_only_th(self, flavor_read_html):
        # GH17054

        # 使用指定的HTML内容进行表格解析，返回结果的第一个DataFrame
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <th>A</th>
                    <th>B</th>
                </tr>
                <tr>
                    <th>a</th>
                    <th>b</th>
                </tr>
                <tr>
                    <td>1</td>
                    <td>2</td>
                </tr>
            </table>
        """
            )
        )[0]

        # 期望的结果DataFrame，包含一行数据 [1, 2]，列名为多级索引 ["A", "B"] 和 ["a", "b"]
        columns = MultiIndex(levels=[["A", "B"], ["a", "b"]], codes=[[0, 1], [0, 1]])
        expected = DataFrame(data=[[1, 2]], columns=columns)

        # 断言结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试解析包含日期的HTML表格的情况
    def test_parse_dates_list(self, flavor_read_html):
        # 创建一个包含日期数据的DataFrame
        df = DataFrame({"date": date_range("1/1/2001", periods=10)})

        # 创建一个期望的DataFrame副本，并将日期列转换为秒单位
        expected = df[:]
        expected["date"] = expected["date"].dt.as_unit("s")

        # 将DataFrame转换为HTML字符串
        str_df = df.to_html()

        # 使用解析日期参数解析HTML字符串，返回结果的第一个DataFrame
        res = flavor_read_html(StringIO(str_df), parse_dates=[1], index_col=0)
        # 断言结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(expected, res[0])

        # 再次使用解析日期参数解析HTML字符串，这次通过列名解析日期列，返回结果的第一个DataFrame
        res = flavor_read_html(StringIO(str_df), parse_dates=["date"], index_col=0)
        # 断言结果DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(expected, res[0])

    # 定义测试函数，测试解析维基百科州表格的情况
    def test_wikipedia_states_table(self, datapath, flavor_read_html):
        # 获取维基百科州表格的数据路径
        data = datapath("io", "data", "html", "wikipedia_states.html")
        # 断言数据文件存在
        assert os.path.isfile(data), f"{data!r} is not a file"
        # 断言数据文件非空
        assert os.path.getsize(data), f"{data!r} is an empty file"

        # 使用指定的数据文件进行表格解析，匹配“Arizona”，指定表头所在行为第1行，返回结果的第一个DataFrame
        result = flavor_read_html(data, match="Arizona", header=1)[0]

        # 断言结果DataFrame的形状为 (60, 12)
        assert result.shape == (60, 12)
        # 断言结果DataFrame的最后一列包含"Unnamed"
        assert "Unnamed" in result.columns[-1]
        # 断言结果DataFrame的"sq mi"列的数据类型为float64
        assert result["sq mi"].dtype == np.dtype("float64")
        # 断言结果DataFrame的第一行的"sq mi"列值接近于665384.04
        assert np.allclose(result.loc[0, "sq mi"], 665384.04)

    # 定义测试函数，测试解析维基百科州表格的情况，并使用多级索引
    def test_wikipedia_states_multiindex(self, datapath, flavor_read_html):
        # 获取维基百科州表格的数据路径
        data = datapath("io", "data", "html", "wikipedia_states.html")
        
        # 使用指定的数据文件进行表格解析，匹配“Arizona”，将第一列作为索引列，返回结果的第一个DataFrame
        result = flavor_read_html(data, match="Arizona", index_col=0)[0]

        # 断言结果DataFrame的形状为 (60, 11)
        assert result.shape == (60, 11)
        # 断言结果DataFrame的最后一列包含"Unnamed"
        assert "Unnamed" in result.columns[-1][1]
        # 断言结果DataFrame的列级别数为2
        assert result.columns.nlevels == 2
        # 断言结果DataFrame中"Alaska"行的("Total area[2]", "sq mi")单元格的值接近于665384.04
        assert np.allclose(result.loc["Alaska", ("Total area[2]", "sq mi")], 665384.04)
    # 测试在空表头行上解析器是否报错
    def test_parser_error_on_empty_header_row(self, flavor_read_html):
        # 使用给定的HTML内容创建flavor_read_html对象，并传入header参数
        result = flavor_read_html(
            StringIO(
                """
                <table>
                    <thead>
                        <tr><th></th><th></tr>  # 表头中有一个空的<th>标签，导致空表头行
                        <tr><th>A</th><th>B</th></tr>  # 第二行表头定义
                    </thead>
                    <tbody>
                        <tr><td>a</td><td>b</td></tr>  # 表体数据行
                    </tbody>
                </table>
            """
            ),
            header=[0, 1],  # 指定列索引作为header参数
        )
        # 预期的DataFrame对象，包含一个数据行，列名为MultiIndex类型
        expected = DataFrame(
            [["a", "b"]],
            columns=MultiIndex.from_tuples(
                [("Unnamed: 0_level_0", "A"), ("Unnamed: 1_level_0", "B")]
            ),
        )
        # 使用assert_frame_equal方法断言测试结果与预期是否相同
        tm.assert_frame_equal(result[0], expected)

    # 测试解析包含小数的行
    def test_decimal_rows(self, flavor_read_html):
        # 使用给定的HTML内容创建flavor_read_html对象，指定decimal参数为"#"
        result = flavor_read_html(
            StringIO(
                """<html>
            <body>
             <table>
                <thead>
                    <tr>
                        <th>Header</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1100#101</td>  # 包含小数的数据行
                    </tr>
                </tbody>
            </table>
            </body>
        </html>"""
            ),
            decimal="#",
        )[0]

        # 预期的DataFrame对象，包含一个名为"Header"的列，数据类型为float64
        expected = DataFrame(data={"Header": 1100.101}, index=[0])

        # 使用assert语句验证结果中"Header"列的数据类型为float64
        assert result["Header"].dtype == np.dtype("float64")
        # 使用assert_frame_equal方法断言测试结果与预期是否相同
        tm.assert_frame_equal(result, expected)

    # 测试布尔型header参数的情况
    @pytest.mark.parametrize("arg", [True, False])
    def test_bool_header_arg(self, spam_data, arg, flavor_read_html):
        # GH 6114 GitHub上的issue编号
        msg = re.escape(
            "Passing a bool to header is invalid. Use header=None for no header or "
            "header=int or list-like of ints to specify the row(s) making up the "
            "column names"
        )
        # 使用pytest.raises断言测试是否抛出TypeError异常，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            flavor_read_html(spam_data, header=arg)

    # 测试使用转换器converters参数的情况
    def test_converters(self, flavor_read_html):
        # GH 13461 GitHub上的issue编号
        # 使用给定的HTML内容创建flavor_read_html对象，并指定converters参数为{"a": str}
        result = flavor_read_html(
            StringIO(
                """<table>
                 <thead>
                   <tr>
                     <th>a</th>
                    </tr>
                 </thead>
                 <tbody>
                   <tr>
                     <td> 0.763</td>
                   </tr>
                   <tr>
                     <td> 0.244</td>
                   </tr>
                 </tbody>
               </table>"""
            ),
            converters={"a": str},  # 将"a"列的数据转换为字符串类型
        )[0]

        # 预期的DataFrame对象，包含一列名为"a"的列，数据为字符串类型列表
        expected = DataFrame({"a": ["0.763", "0.244"]})

        # 使用assert_frame_equal方法断言测试结果与预期是否相同
        tm.assert_frame_equal(result, expected)
    # 测试处理缺失值的功能，针对 flavor_read_html 方法
    def test_na_values(self, flavor_read_html):
        # GH 13461 GitHub issue 号
        # 构造包含表格的 HTML 字符串，模拟数据表格
        result = flavor_read_html(
            StringIO(
                """<table>
                 <thead>
                   <tr>
                     <th>a</th>
                   </tr>
                 </thead>
                 <tbody>
                   <tr>
                     <td> 0.763</td>
                   </tr>
                   <tr>
                     <td> 0.244</td>
                   </tr>
                 </tbody>
               </table>"""
            ),
            na_values=[0.244],  # 指定要视为缺失值的数值列表
        )[0]

        # 期望的 DataFrame 结果
        expected = DataFrame({"a": [0.763, np.nan]})

        # 使用测试工具函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    # 测试保留默认的缺失值表示功能，针对 flavor_read_html 方法
    def test_keep_default_na(self, flavor_read_html):
        # 构造包含表格的 HTML 字符串，模拟数据表格
        html_data = """<table>
                        <thead>
                            <tr>
                            <th>a</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                            <td> N/A</td>
                            </tr>
                            <tr>
                            <td> NA</td>
                            </tr>
                        </tbody>
                    </table>"""

        # 期望的 DataFrame 结果
        expected_df = DataFrame({"a": ["N/A", "NA"]})
        # 测试不保留默认缺失值表示时的处理结果
        html_df = flavor_read_html(StringIO(html_data), keep_default_na=False)[0]
        tm.assert_frame_equal(expected_df, html_df)

        # 期望的 DataFrame 结果
        expected_df = DataFrame({"a": [np.nan, np.nan]})
        # 测试保留默认缺失值表示时的处理结果
        html_df = flavor_read_html(StringIO(html_data), keep_default_na=True)[0]
        tm.assert_frame_equal(expected_df, html_df)

    # 测试保留空行的功能，针对 flavor_read_html 方法
    def test_preserve_empty_rows(self, flavor_read_html):
        # 构造包含表格的 HTML 字符串，模拟数据表格
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <th>A</th>
                    <th>B</th>
                </tr>
                <tr>
                    <td>a</td>
                    <td>b</td>
                </tr>
                <tr>
                    <td></td>
                    <td></td>
                </tr>
            </table>
        """
            )
        )[0]

        # 期望的 DataFrame 结果，包括保留的空行
        expected = DataFrame(data=[["a", "b"], [np.nan, np.nan]], columns=["A", "B"])

        # 使用测试工具函数检查结果是否符合预期
        tm.assert_frame_equal(result, expected)
    def test_ignore_empty_rows_when_inferring_header(self, flavor_read_html):
        # 使用指定的flavor函数读取HTML内容并返回DataFrame列表中的第一个DataFrame
        result = flavor_read_html(
            StringIO(
                """
            <table>
                <thead>
                    <tr><th></th><th></tr>  # 表头的第一行，两个空的表头单元格
                    <tr><th>A</th><th>B</th></tr>  # 表头的第二行，指定A和B两列
                    <tr><th>a</th><th>b</th></tr>  # 表头的第三行，指定a和b两列
                </thead>
                <tbody>
                    <tr><td>1</td><td>2</td></tr>  # 表体的第一行，包含1和2两个数据单元格
                </tbody>
            </table>
        """
            )
        )[0]

        columns = MultiIndex(levels=[["A", "B"], ["a", "b"]], codes=[[0, 1], [0, 1]])
        expected = DataFrame(data=[[1, 2]], columns=columns)

        # 断言结果DataFrame与预期DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_multiple_header_rows(self, flavor_read_html):
        # 测试多行表头的情况
        # 创建预期的DataFrame，包含三列数据：姓名、年龄、政党，并设定多级表头
        expected_df = DataFrame(
            data=[("Hillary", 68, "D"), ("Bernie", 74, "D"), ("Donald", 69, "R")]
        )
        expected_df.columns = [
            ["Unnamed: 0_level_0", "Age", "Party"],  # 第一级表头
            ["Name", "Unnamed: 1_level_1", "Unnamed: 2_level_1"],  # 第二级表头
        ]
        # 将预期的DataFrame转换为HTML字符串
        html = expected_df.to_html(index=False)
        # 使用flavor函数读取HTML字符串并返回DataFrame列表中的第一个DataFrame
        html_df = flavor_read_html(StringIO(html))[0]
        # 断言预期的DataFrame与从HTML解析得到的DataFrame相等
        tm.assert_frame_equal(expected_df, html_df)

    def test_works_on_valid_markup(self, datapath, flavor_read_html):
        # 测试在有效的HTML标记上正常工作的情况
        filename = datapath("io", "data", "html", "valid_markup.html")
        # 使用flavor函数读取HTML文件，并期望返回一个DataFrame列表
        dfs = flavor_read_html(filename, index_col=0)
        # 断言返回的dfs是一个列表
        assert isinstance(dfs, list)
        # 断言列表中第一个元素是一个DataFrame
        assert isinstance(dfs[0], DataFrame)

    @pytest.mark.slow
    def test_fallback_success(self, datapath, flavor_read_html):
        # 测试在特定条件下的后备成功情况
        banklist_data = datapath("io", "data", "html", "banklist.html")
        # 使用flavor函数读取银行列表HTML数据，匹配含有"Water"的数据行，尝试多种解析方法
        flavor_read_html(banklist_data, match=".*Water.*", flavor=["lxml", "html5lib"])

    def test_to_html_timestamp(self):
        # 测试DataFrame对象转换为HTML时，是否包含时间戳信息
        rng = date_range("2000-01-01", periods=10)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=rng)

        # 将DataFrame转换为HTML字符串
        result = df.to_html()
        # 断言结果中包含指定的时间戳信息
        assert "2000-01-01" in result

    def test_to_html_borderless(self):
        # 测试DataFrame对象转换为HTML时，边框参数的影响
        df = DataFrame([{"A": 1, "B": 2}])
        
        # 分别测试不同边框参数下生成的HTML字符串
        out_border_default = df.to_html()
        out_border_true = df.to_html(border=True)
        out_border_explicit_default = df.to_html(border=1)
        out_border_nondefault = df.to_html(border=2)
        out_border_zero = df.to_html(border=0)
        out_border_false = df.to_html(border=False)

        # 断言不同参数生成的HTML字符串符合预期
        assert ' border="1"' in out_border_default
        assert out_border_true == out_border_default
        assert out_border_default == out_border_explicit_default
        assert out_border_default != out_border_nondefault
        assert ' border="2"' in out_border_nondefault
        assert ' border="0"' not in out_border_zero
        assert " border" not in out_border_false
        assert out_border_zero == out_border_false
    @pytest.mark.parametrize(
        "displayed_only,exp0,exp1",
        [
            (True, ["foo"], None),  # 参数化测试，测试显示隐藏内容为True时的期望输出
            (False, ["foo  bar  baz  qux"], DataFrame(["foo"])),  # 参数化测试，测试显示隐藏内容为False时的期望输出
        ],
    )
    def test_displayed_only(self, displayed_only, exp0, exp1, flavor_read_html):
        # GH 20027
        data = """<html>
          <body>
            <table>
              <tr>
                <td>
                  foo
                  <span style="display:none;text-align:center">bar</span>  # 隐藏的span标签内容不应显示
                  <span style="display:none">baz</span>  # 隐藏的span标签内容不应显示
                  <span style="display: none">qux</span>  # 隐藏的span标签内容不应显示
                </td>
              </tr>
            </table>
            <table style="display: none">
              <tr>
                <td>foo</td>
              </tr>
            </table>
          </body>
        </html>"""

        exp0 = DataFrame(exp0)  # 设置期望的DataFrame对象
        dfs = flavor_read_html(StringIO(data), displayed_only=displayed_only)  # 使用给定的flavor读取HTML内容并解析为DataFrame对象列表
        tm.assert_frame_equal(dfs[0], exp0)  # 断言第一个DataFrame与期望的DataFrame相等

        if exp1 is not None:
            tm.assert_frame_equal(dfs[1], exp1)  # 如果exp1不为空，断言第二个DataFrame与期望的DataFrame相等
        else:
            assert len(dfs) == 1  # 如果exp1为空，断言DataFrame列表的长度为1，表明不应解析隐藏的表格

    @pytest.mark.parametrize("displayed_only", [True, False])
    def test_displayed_only_with_many_elements(self, displayed_only, flavor_read_html):
        html_table = """
        <table>
            <tr>
                <th>A</th>
                <th>B</th>
            </tr>
            <tr>
                <td>1</td>
                <td>2</td>
            </tr>
            <tr>
                <td><span style="display:none"></span>4</td>  # 隐藏的span标签内容不应显示
                <td>5</td>
            </tr>
        </table>
        """
        result = flavor_read_html(StringIO(html_table), displayed_only=displayed_only)[0]  # 使用给定的flavor读取HTML内容并解析为DataFrame对象
        expected = DataFrame({"A": [1, 4], "B": [2, 5]})  # 设置期望的DataFrame对象
        tm.assert_frame_equal(result, expected)  # 断言解析的DataFrame对象与期望的DataFrame对象相等

    @pytest.mark.filterwarnings(
        "ignore:You provided Unicode markup but also provided a value for "
        "from_encoding.*:UserWarning"
    )
    # 定义一个测试方法，用于测试编码功能
    def test_encode(self, html_encoding_file, flavor_read_html):
        # 提取 HTML 文件路径中的基本文件名（不包含路径）
        base_path = os.path.basename(html_encoding_file)
        # 获取基本文件名去除扩展名后的部分作为根名称
        root = os.path.splitext(base_path)[0]
        # 从根名称中提取编码部分（假设根名称形如 filename_encoding）

        _, encoding = root.split("_")

        try:
            # 使用 'rb' 模式打开 HTML 文件，将文件内容封装为 BytesIO 对象，并使用给定编码读取 HTML 内容
            with open(html_encoding_file, "rb") as fobj:
                from_string = flavor_read_html(
                    BytesIO(fobj.read()), encoding=encoding, index_col=0
                ).pop()

            # 使用 'rb' 模式再次打开 HTML 文件，将文件内容封装为 BytesIO 对象，并使用给定编码读取 HTML 内容
            with open(html_encoding_file, "rb") as fobj:
                from_file_like = flavor_read_html(
                    BytesIO(fobj.read()), encoding=encoding, index_col=0
                ).pop()

            # 直接使用文件路径读取 HTML 内容，使用给定编码读取 HTML 内容
            from_filename = flavor_read_html(
                html_encoding_file, encoding=encoding, index_col=0
            ).pop()
            
            # 使用 pandas 测试工具比较三种读取方式得到的 DataFrame 是否相等
            tm.assert_frame_equal(from_string, from_file_like)
            tm.assert_frame_equal(from_string, from_filename)
        
        except Exception:
            # 捕获任何异常，通常是编码相关或文件读取失败导致的
            # 在 Windows 平台上，针对 UTF-16/32 编码可能会失败
            if is_platform_windows():
                if "16" in encoding or "32" in encoding:
                    # 如果是 UTF-16 或 UTF-32 编码且在 Windows 上，跳过测试
                    pytest.skip()
            # 如果不是上述情况，则抛出异常
            raise

    # 定义一个测试方法，用于测试解析不可寻址对象的失败情况
    def test_parse_failure_unseekable(self, flavor_read_html):
        # Issue #17975

        # 如果 flavor_read_html 的关键字中包含 flavor 属性且其值为 'lxml'，则跳过测试
        if flavor_read_html.keywords.get("flavor") == "lxml":
            pytest.skip("Not applicable for lxml")

        # 定义一个不可寻址的 StringIO 子类对象，用于模拟不可寻址的文件对象
        class UnseekableStringIO(StringIO):
            def seekable(self):
                return False

        # 创建一个包含 HTML 内容的不可寻址的 StringIO 对象
        bad = UnseekableStringIO(
            """
            <table><tr><td>spam<foobr />eggs</td></tr></table>"""
        )

        # 使用 flavor_read_html 方法解析不可寻址的文件对象，应该成功
        assert flavor_read_html(bad)

        # 使用 flavor_read_html 方法再次尝试解析不可寻址的文件对象，预期抛出 ValueError 异常
        with pytest.raises(ValueError, match="passed a non-rewindable file object"):
            flavor_read_html(bad)

    # 定义一个测试方法，用于测试解析可以回绕的失败情况
    def test_parse_failure_rewinds(self, flavor_read_html):
        # Issue #17975

        # 定义一个 MockFile 类，模拟可以回绕的文件对象
        class MockFile:
            def __init__(self, data) -> None:
                self.data = data
                self.at_end = False

            # 模拟读取文件内容的方法
            def read(self, size=None):
                data = "" if self.at_end else self.data
                self.at_end = True
                return data

            # 模拟文件指针移动到指定位置的方法
            def seek(self, offset):
                self.at_end = False

            # 模拟判断文件对象是否可寻址的方法
            def seekable(self):
                return True

            # 未定义迭代器方法，故此处略去

        # 创建两个 MockFile 对象，一个包含有效 HTML 内容，另一个包含无效 HTML 内容
        good = MockFile("<table><tr><td>spam<br />eggs</td></tr></table>")
        bad = MockFile("<table><tr><td>spam<foobr />eggs</td></tr></table>")

        # 使用 flavor_read_html 方法解析可以回绕的文件对象，应该成功
        assert flavor_read_html(good)
        # 使用 flavor_read_html 方法解析无法回绕的文件对象，应该成功
        assert flavor_read_html(bad)

    # 为测试方法添加标记，表示该测试执行较慢且只在单 CPU 模式下运行
    @pytest.mark.slow
    @pytest.mark.single_cpu
    def test_importcheck_thread_safety(self, datapath, flavor_read_html):
        # see gh-16928
        # 测试线程安全性，验证多线程执行时是否能正确处理异常

        class ErrorThread(threading.Thread):
            # 定义一个自定义的线程类，用于捕获异常
            def run(self):
                try:
                    super().run()  # 调用父类的 run 方法
                except Exception as err:
                    self.err = err  # 如果出现异常，记录异常信息
                else:
                    self.err = None  # 没有异常则将异常信息设为 None

        filename = datapath("io", "data", "html", "valid_markup.html")
        # 获取测试文件的路径

        helper_thread1 = ErrorThread(target=flavor_read_html, args=(filename,))
        helper_thread2 = ErrorThread(target=flavor_read_html, args=(filename,))
        # 创建两个线程实例，分别执行 flavor_read_html 函数，传入相同的文件路径参数

        helper_thread1.start()  # 启动线程1
        helper_thread2.start()  # 启动线程2

        while helper_thread1.is_alive() or helper_thread2.is_alive():
            pass
        # 等待两个线程执行完成

        assert None is helper_thread1.err is helper_thread2.err
        # 断言：两个线程的异常信息都应该是 None，表示都没有捕获到异常

    def test_parse_path_object(self, datapath, flavor_read_html):
        # GH 37705
        # 测试解析路径对象，验证以路径对象和路径字符串作为参数时的解析效果

        file_path_string = datapath("io", "data", "html", "spam.html")
        # 获取测试文件的路径字符串

        file_path = Path(file_path_string)
        # 创建 Path 对象

        df1 = flavor_read_html(file_path_string)[0]
        # 使用路径字符串调用 flavor_read_html 函数，获取返回的 DataFrame 对象列表的第一个元素

        df2 = flavor_read_html(file_path)[0]
        # 使用 Path 对象调用 flavor_read_html 函数，获取返回的 DataFrame 对象列表的第一个元素

        tm.assert_frame_equal(df1, df2)
        # 断言：两个 DataFrame 对象应该相等

    def test_parse_br_as_space(self, flavor_read_html):
        # GH 29528: pd.read_html() convert <br> to space
        # 测试 pd.read_html() 将 <br> 转换为空格的功能

        result = flavor_read_html(
            StringIO(
                """
            <table>
                <tr>
                    <th>A</th>
                </tr>
                <tr>
                    <td>word1<br>word2</td>
                </tr>
            </table>
        """
            )
        )[0]
        # 调用 flavor_read_html 函数，传入包含 <br> 标签的 HTML 字符串作为参数，获取返回的 DataFrame 对象列表的第一个元素

        expected = DataFrame(data=[["word1 word2"]], columns=["A"])
        # 创建预期的 DataFrame 对象，期望单元格内容为 "word1 word2"

        tm.assert_frame_equal(result, expected)
        # 断言：返回的 DataFrame 对象应该与预期的 DataFrame 对象相等

    @pytest.mark.parametrize("arg", ["all", "body", "header", "footer"])
    # 使用 pytest 的参数化装饰器，定义测试函数的参数为 "all", "body", "header", "footer"
    # 定义一个单元测试方法，用于测试从 HTML 数据中提取链接的功能
    def test_extract_links(self, arg, flavor_read_html):
        # 模拟一个包含表格的 HTML 数据
        gh_13141_data = """
          <table>
            <tr>
              <th>HTTP</th>
              <th>FTP</th>
              <th><a href="https://en.wiktionary.org/wiki/linkless">Linkless</a></th>
            </tr>
            <tr>
              <td><a href="https://en.wikipedia.org/">Wikipedia</a></td>
              <td>SURROUNDING <a href="ftp://ftp.us.debian.org/">Debian</a> TEXT</td>
              <td>Linkless</td>
            </tr>
            <tfoot>
              <tr>
                <td><a href="https://en.wikipedia.org/wiki/Page_footer">Footer</a></td>
                <td>
                  Multiple <a href="1">links:</a> <a href="2">Only first captured.</a>
                </td>
              </tr>
            </tfoot>
          </table>
          """
        
        # 预期的链接提取结果，包含了各部分需要忽略和需要提取的链接信息
        gh_13141_expected = {
            "head_ignore": ["HTTP", "FTP", "Linkless"],
            "head_extract": [
                ("HTTP", None),
                ("FTP", None),
                ("Linkless", "https://en.wiktionary.org/wiki/linkless"),
            ],
            "body_ignore": ["Wikipedia", "SURROUNDING Debian TEXT", "Linkless"],
            "body_extract": [
                ("Wikipedia", "https://en.wikipedia.org/"),
                ("SURROUNDING Debian TEXT", "ftp://ftp.us.debian.org/"),
                ("Linkless", None),
            ],
            "footer_ignore": [
                "Footer",
                "Multiple links: Only first captured.",
                None,
            ],
            "footer_extract": [
                ("Footer", "https://en.wikipedia.org/wiki/Page_footer"),
                ("Multiple links: Only first captured.", "1"),
                None,
            ],
        }
        
        # 根据参数设置要比较的期望值
        data_exp = gh_13141_expected["body_ignore"]
        foot_exp = gh_13141_expected["footer_ignore"]
        head_exp = gh_13141_expected["head_ignore"]
        if arg == "all":
            data_exp = gh_13141_expected["body_extract"]
            foot_exp = gh_13141_expected["footer_extract"]
            head_exp = gh_13141_expected["head_extract"]
        elif arg == "body":
            data_exp = gh_13141_expected["body_extract"]
        elif arg == "footer":
            foot_exp = gh_13141_expected["footer_extract"]
        elif arg == "header":
            head_exp = gh_13141_expected["head_extract"]
        
        # 调用被测函数 `flavor_read_html` 从模拟的 HTML 数据中提取链接，并获取结果
        result = flavor_read_html(StringIO(gh_13141_data), extract_links=arg)[0]
        
        # 创建期望的 DataFrame 结果，根据提取的数据、忽略的数据和需要作为列名的数据
        expected = DataFrame([data_exp, foot_exp], columns=head_exp)
        
        # 将缺失值填充为 NaN
        expected = expected.fillna(np.nan)
        
        # 使用 `tm.assert_frame_equal` 断言实际结果与期望结果一致
        tm.assert_frame_equal(result, expected)
    
    # 定义一个测试方法，用于测试当传递错误的 `extract_links` 参数时是否会引发 ValueError 异常
    def test_extract_links_bad(self, spam_data):
        # 准备错误消息
        msg = (
            "`extract_links` must be one of "
            '{None, "header", "footer", "body", "all"}, got "incorrect"'
        )
        
        # 使用 pytest 断言确实会抛出 ValueError 异常，并且异常消息符合预期
        with pytest.raises(ValueError, match=msg):
            read_html(spam_data, extract_links="incorrect")
    # 测试从 HTML 数据中提取所有链接，不包括表头
    def test_extract_links_all_no_header(self, flavor_read_html):
        # GH 48316
        # 定义包含 HTML 表格和链接的字符串数据
        data = """
        <table>
          <tr>
            <td>
              <a href='https://google.com'>Google.com</a>
            </td>
          </tr>
        </table>
        """
        # 调用 flavor_read_html 方法从 StringIO 对象中解析 HTML 数据，并提取所有链接
        result = flavor_read_html(StringIO(data), extract_links="all")[0]
        # 定义预期结果的 DataFrame
        expected = DataFrame([[("Google.com", "https://google.com")]])
        # 使用 pytest 的 assert_frame_equal 断言方法比较结果和预期值是否相等
        tm.assert_frame_equal(result, expected)

    # 测试读取 HTML 数据时，指定无效的 dtype_backend 参数
    def test_invalid_dtype_backend(self):
        # 定义错误消息
        msg = (
            "dtype_backend numpy is invalid, only 'numpy_nullable' and "
            "'pyarrow' are allowed."
        )
        # 使用 pytest.raises 断言捕获 ValueError 异常，并验证异常消息是否匹配预期错误消息
        with pytest.raises(ValueError, match=msg):
            read_html("test", dtype_backend="numpy")

    # 测试处理包含 style 标签的 HTML 数据
    def test_style_tag(self, flavor_read_html):
        # GH 48316
        # 定义包含 HTML 表格和样式标签的字符串数据
        data = """
        <table>
            <tr>
                <th>
                    <style>.style</style>
                    A
                    </th>
                <th>B</th>
            </tr>
            <tr>
                <td>A1</td>
                <td>B1</td>
            </tr>
            <tr>
                <td>A2</td>
                <td>B2</td>
            </tr>
        </table>
        """
        # 调用 flavor_read_html 方法从 StringIO 对象中解析 HTML 数据
        result = flavor_read_html(StringIO(data))[0]
        # 定义预期结果的 DataFrame
        expected = DataFrame(data=[["A1", "B1"], ["A2", "B2"]], columns=["A", "B"])
        # 使用 pytest 的 assert_frame_equal 断言方法比较结果和预期值是否相等
        tm.assert_frame_equal(result, expected)
```