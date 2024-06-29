# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_nlargest.py`

```
"""
Note: for naming purposes, most tests are title with as e.g. "test_nlargest_foo"
but are implicitly also testing nsmallest_foo.
"""

# 导入所需的模块和库
from string import ascii_lowercase

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 测试框架

import pandas as pd  # 导入 Pandas 库
import pandas._testing as tm  # 导入 Pandas 测试模块
from pandas.util.version import Version  # 导入 Pandas 版本模块


@pytest.fixture
def df_main_dtypes():
    """
    创建并返回一个包含多种数据类型的 DataFrame，用于测试
    """
    return pd.DataFrame(
        {
            "group": [1, 1, 2],
            "int": [1, 2, 3],
            "float": [4.0, 5.0, 6.0],
            "string": list("abc"),
            "category_string": pd.Series(list("abc")).astype("category"),
            "category_int": [7, 8, 9],
            "datetime": pd.date_range("20130101", periods=3),
            "datetimetz": pd.date_range("20130101", periods=3, tz="US/Eastern"),
            "timedelta": pd.timedelta_range("1 s", periods=3, freq="s"),
        },
        columns=[
            "group",
            "int",
            "float",
            "string",
            "category_string",
            "category_int",
            "datetime",
            "datetimetz",
            "timedelta",
        ],
    )


class TestNLargestNSmallest:
    # ----------------------------------------------------------------------
    # Top / bottom
    @pytest.mark.parametrize(
        "order",
        [
            ["a"],
            ["c"],
            ["a", "b"],
            ["a", "c"],
            ["b", "a"],
            ["b", "c"],
            ["a", "b", "c"],
            ["c", "a", "b"],
            ["c", "b", "a"],
            ["b", "c", "a"],
            ["b", "a", "c"],
            # dups!
            ["b", "c", "c"],
        ],
    )
    @pytest.mark.parametrize("n", range(1, 11))
    def test_nlargest_n(self, nselect_method, n, order):
        # GH#10393
        # 创建一个包含随机数据的 DataFrame，用于测试 nlargest 和 nsmallest 方法
        df = pd.DataFrame(
            {
                "a": np.random.default_rng(2).permutation(10),
                "b": list(ascii_lowercase[:10]),
                "c": np.random.default_rng(2).permutation(10).astype("float64"),
            }
        )
        if "b" in order:
            # 如果排序列中包含 'b' 列，抛出 TypeError 异常，因为 'b' 列的 dtype 是 object 或 string
            error_msg = (
                f"Column 'b' has dtype (object|string), "
                f"cannot use method '{nselect_method}' with this dtype"
            )
            with pytest.raises(TypeError, match=error_msg):
                getattr(df, nselect_method)(n, order)
        else:
            ascending = nselect_method == "nsmallest"
            # 调用 nlargest 或 nsmallest 方法，并进行结果比较
            result = getattr(df, nselect_method)(n, order)
            expected = df.sort_values(order, ascending=ascending).head(n)
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "columns", [["group", "category_string"], ["group", "string"]]
    )
    def test_nlargest_error(self, df_main_dtypes, nselect_method, columns):
        # 从参数中获取主数据框
        df = df_main_dtypes
        # 获取第二列的列名
        col = columns[1]
        # 构建错误消息字符串，描述列数据类型不支持指定方法的情况
        error_msg = (
            f"Column '{col}' has dtype {df[col].dtype}, "
            f"cannot use method '{nselect_method}' with this dtype"
        )
        # 转义错误消息中可能出现的特殊字符
        error_msg = (
            error_msg.replace("(", "\\(")
            .replace(")", "\\)")
            .replace("[", "\\[")
            .replace("]", "\\]")
        )
        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=error_msg):
            # 调用主数据框的指定方法，并传递参数
            getattr(df, nselect_method)(2, columns)

    def test_nlargest_all_dtypes(self, df_main_dtypes):
        # 从参数中获取主数据框
        df = df_main_dtypes
        # 对主数据框中除去指定列集合外的所有列，使用 nsmallest 方法获取最小的 2 行数据
        df.nsmallest(2, list(set(df) - {"category_string", "string"}))
        # 对主数据框中除去指定列集合外的所有列，使用 nlargest 方法获取最大的 2 行数据
        df.nlargest(2, list(set(df) - {"category_string", "string"}))

    def test_nlargest_duplicates_on_starter_columns(self):
        # 为 GH#22752 编写的回归测试

        # 创建一个包含重复值的 DataFrame
        df = pd.DataFrame({"a": [2, 2, 2, 1, 1, 1], "b": [1, 2, 3, 3, 2, 1]})

        # 使用 nlargest 方法获取基于指定列的前 4 行数据，并进行预期结果的比较
        result = df.nlargest(4, columns=["a", "b"])
        expected = pd.DataFrame(
            {"a": [2, 2, 2, 1], "b": [3, 2, 1, 3]}, index=[2, 1, 0, 3]
        )
        tm.assert_frame_equal(result, expected)

        # 使用 nsmallest 方法获取基于指定列的前 4 行数据，并进行预期结果的比较
        result = df.nsmallest(4, columns=["a", "b"])
        expected = pd.DataFrame(
            {"a": [1, 1, 1, 2], "b": [1, 2, 3, 1]}, index=[5, 4, 3, 0]
        )
        tm.assert_frame_equal(result, expected)

    def test_nlargest_n_identical_values(self):
        # 为 GH#15297 编写的回归测试
        # 创建一个包含相同值的 DataFrame
        df = pd.DataFrame({"a": [1] * 5, "b": [1, 2, 3, 4, 5]})

        # 使用 nlargest 方法获取基于指定列的前 3 行数据，并进行预期结果的比较
        result = df.nlargest(3, "a")
        expected = pd.DataFrame({"a": [1] * 3, "b": [1, 2, 3]}, index=[0, 1, 2])
        tm.assert_frame_equal(result, expected)

        # 使用 nsmallest 方法获取基于指定列的前 3 行数据，并进行预期结果的比较
        result = df.nsmallest(3, "a")
        expected = pd.DataFrame({"a": [1] * 3, "b": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "order",
        [["a", "b", "c"], ["c", "b", "a"], ["a"], ["b"], ["a", "b"], ["c", "b"]],
    )
    @pytest.mark.parametrize("n", range(1, 6))
    def test_nlargest_n_duplicate_index(self, n, order, request):
        # GH#13412
        # 创建一个包含重复索引的 DataFrame，用于测试 nlargest 和 nsmallest 方法
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 4], "b": [1, 1, 1, 1, 1], "c": [0, 1, 2, 5, 4]},
            index=[0, 0, 1, 1, 1],
        )
        # 使用 nlargest 方法获取最大的 n 个元素
        result = df.nsmallest(n, order)
        # 预期结果是按指定顺序排序后的前 n 行
        expected = df.sort_values(order).head(n)
        tm.assert_frame_equal(result, expected)

        # 使用 nlargest 方法获取最大的 n 个元素
        result = df.nlargest(n, order)
        # 预期结果是按指定顺序降序排序后的前 n 行
        expected = df.sort_values(order, ascending=False).head(n)
        
        # 如果 numpy 的版本 >= 1.25 并且满足条件 (order == ["a"] and n in (1, 2, 3, 4)) 或者 (order == ["a", "b"] and n == 5)
        # 则标记该测试为预期失败，原因是 pandas 默认对重复值的排序不稳定，与 numpy >= 1.25 以及 AVX 指令有关
        if Version(np.__version__) >= Version("1.25") and (
            (order == ["a"] and n in (1, 2, 3, 4)) or (order == ["a", "b"]) and n == 5
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "pandas default unstable sorting of duplicates"
                        "issue with numpy>=1.25 with AVX instructions"
                    ),
                    strict=False,
                )
            )
        tm.assert_frame_equal(result, expected)

    def test_nlargest_duplicate_keep_all_ties(self):
        # GH#16818
        # 创建一个包含重复值的 DataFrame，测试 nlargest 方法中 keep="all" 参数的效果
        df = pd.DataFrame(
            {"a": [5, 4, 4, 2, 3, 3, 3, 3], "b": [10, 9, 8, 7, 5, 50, 10, 20]}
        )
        # 使用 nlargest 方法获取最大的 4 个元素，保留所有重复值
        result = df.nlargest(4, "a", keep="all")
        # 预期结果是按 "a" 列的值排序后的前 4 行，保留所有重复值
        expected = pd.DataFrame(
            {
                "a": {0: 5, 1: 4, 2: 4, 4: 3, 5: 3, 6: 3, 7: 3},
                "b": {0: 10, 1: 9, 2: 8, 4: 5, 5: 50, 6: 10, 7: 20},
            }
        )
        tm.assert_frame_equal(result, expected)

        # 使用 nsmallest 方法获取最小的 2 个元素，保留所有重复值
        result = df.nsmallest(2, "a", keep="all")
        # 预期结果是按 "a" 列的值排序后的前 2 行，保留所有重复值
        expected = pd.DataFrame(
            {
                "a": {3: 2, 4: 3, 5: 3, 6: 3, 7: 3},
                "b": {3: 7, 4: 5, 5: 50, 6: 10, 7: 20},
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_nlargest_multiindex_column_lookup(self):
        # Check whether tuples are correctly treated as multi-level lookups.
        # GH#23033
        # 创建一个包含多级索引的 DataFrame，测试 nlargest 方法对多级索引列的处理
        df = pd.DataFrame(
            columns=pd.MultiIndex.from_product([["x"], ["a", "b"]]),
            data=[[0.33, 0.13], [0.86, 0.25], [0.25, 0.70], [0.85, 0.91]],
        )

        # 使用 nsmallest 方法获取最小的 3 个元素，按 ("x", "a") 列进行排序
        result = df.nsmallest(3, ("x", "a"))
        # 预期结果是按 ("x", "a") 列的值排序后的前 3 行
        expected = df.iloc[[2, 0, 3]]
        tm.assert_frame_equal(result, expected)

        # 使用 nlargest 方法获取最大的 3 个元素，按 ("x", "b") 列进行排序
        result = df.nlargest(3, ("x", "b"))
        # 预期结果是按 ("x", "b") 列的值排序后的前 3 行
        expected = df.iloc[[3, 2, 1]]
        tm.assert_frame_equal(result, expected)

    def test_nlargest_nan(self):
        # GH#43060
        # 创建一个包含 NaN 值的 DataFrame，测试 nlargest 方法对 NaN 值的处理
        df = pd.DataFrame([np.nan, np.nan, 0, 1, 2, 3])
        # 使用 nlargest 方法获取最大的 5 个元素，按第一列进行排序
        result = df.nlargest(5, 0)
        # 预期结果是按第一列的值降序排序后的前 5 行
        expected = df.sort_values(0, ascending=False).head(5)
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，测试DataFrame的nsmallest方法处理NaN值和取得指定元素后的行为
    def test_nsmallest_nan_after_n_element(self):
        # 注释: GH#46589 表示这是与GitHub上某个issue相关的测试
        # 创建一个包含三列的DataFrame，包括整数和NaN值
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, None, 7],
                "b": [7, 6, 5, 4, 3, 2, 1],
                "c": [1, 1, 2, 2, 3, 3, 3],
            },
            index=range(7),
        )
        # 调用DataFrame的nsmallest方法，按照指定列"a"和"b"，取最小的5行
        result = df.nsmallest(5, columns=["a", "b"])
        # 创建一个预期的DataFrame，包含同样的列和索引，且列"a"转换为浮点数类型
        expected = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [7, 6, 5, 4, 3],
                "c": [1, 1, 2, 2, 3],
            },
            index=range(5),
        ).astype({"a": "float"})
        # 使用测试工具tm.assert_frame_equal来比较结果DataFrame和预期DataFrame是否相同
        tm.assert_frame_equal(result, expected)
```