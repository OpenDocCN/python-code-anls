# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_filter.py`

```
# 导入 NumPy 库，并使用 np 作为别名
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas 库，并从中导入 DataFrame 类
import pandas as pd
from pandas import DataFrame
# 导入 pandas 内部测试模块，用于数据框测试
import pandas._testing as tm


# 定义 TestDataFrameFilter 类，用于数据框过滤测试
class TestDataFrameFilter:
    def test_filter(self, float_frame, float_string_frame):
        # 使用指定列名列表对浮点数数据框进行过滤
        filtered = float_frame.filter(["A", "B", "E"])
        # 断言过滤后的列数为2
        assert len(filtered.columns) == 2
        # 断言过滤后不包含列名"E"
        assert "E" not in filtered

        # 使用指定列名列表对浮点数数据框进行过滤，指定轴为列
        filtered = float_frame.filter(["A", "B", "E"], axis="columns")
        # 断言过滤后的列数为2
        assert len(filtered.columns) == 2
        # 断言过滤后不包含列名"E"
        assert "E" not in filtered

        # 使用指定索引值对浮点数数据框进行行过滤
        idx = float_frame.index[0:4]
        filtered = float_frame.filter(idx, axis="index")
        # 生成预期的过滤后的数据框
        expected = float_frame.reindex(index=idx)
        # 断言两个数据框内容相等
        tm.assert_frame_equal(filtered, expected)

        # 复制浮点数数据框
        fcopy = float_frame.copy()
        # 在复制后的数据框中添加新列"AA"并赋值为1

        filtered = fcopy.filter(like="A")
        # 断言过滤后的列数为2
        assert len(filtered.columns) == 2
        # 断言过滤后包含列名"AA"
        assert "AA" in filtered

        # 使用包含整数列名的数据框，根据部分列名进行过滤
        df = DataFrame(0.0, index=[0, 1, 2], columns=[0, 1, "_A", "_B"])
        filtered = df.filter(like="_")
        # 断言过滤后的列数为2
        assert len(filtered.columns) == 2

        # 使用包含整数和字符串列名的数据框，根据正则表达式进行过滤
        df = DataFrame(0.0, index=[0, 1, 2], columns=["A1", 1, "B", 2, "C"])
        expected = DataFrame(
            0.0, index=[0, 1, 2], columns=pd.Index([1, 2], dtype=object)
        )
        filtered = df.filter(regex="^[0-9]+$")
        # 断言过滤后的数据框内容与预期相等
        tm.assert_frame_equal(filtered, expected)

        expected = DataFrame(0.0, index=[0, 1, 2], columns=[0, "0", 1, "1"])
        # 使用正则表达式不过滤任何列
        filtered = expected.filter(regex="^[0-9]+$")
        # 断言过滤后的数据框内容与预期相等
        tm.assert_frame_equal(filtered, expected)

        # 测试传入空参数的情况
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter()
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter(items=None)
        with pytest.raises(TypeError, match="Must pass"):
            float_frame.filter(axis=1)

        # 测试互斥参数的情况
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$", like="bbi")
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$", axis=1)
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], regex="e$")
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], like="bbi", axis=0)
        with pytest.raises(TypeError, match="mutually exclusive"):
            float_frame.filter(items=["one", "three"], like="bbi")

        # 对包含字符串列的数据框进行过滤
        filtered = float_string_frame.filter(like="foo")
        # 断言过滤后的数据框中包含列名"foo"
        assert "foo" in filtered

        # 对包含Unicode列名的浮点数数据框进行过滤，不会进行ASCII编码
        df = float_frame.rename(columns={"B": "\u2202"})
        filtered = df.filter(like="C")
        # 断言过滤后的数据框中包含列名"C"
        assert "C" in filtered
    def test_filter_regex_search(self, float_frame):
        # 复制传入的 DataFrame，以免修改原始数据
        fcopy = float_frame.copy()
        # 向复制的 DataFrame 中新增一列名为 "AA" 的列，赋值为 1
        fcopy["AA"] = 1

        # 使用正则表达式过滤 DataFrame 的列名，保留以大写字母 A 开头的列
        filtered = fcopy.filter(regex="[A]+")
        # 断言过滤后的 DataFrame 列数为 2
        assert len(filtered.columns) == 2
        # 断言 "AA" 列在过滤后的 DataFrame 中
        assert "AA" in filtered

        # 创建一个新的 DataFrame，包含不同以及组合的列名
        df = DataFrame(
            {"aBBa": [1, 2], "BBaBB": [1, 2], "aCCa": [1, 2], "aCCaBB": [1, 2]}
        )

        # 使用正则表达式过滤 DataFrame 的列名，保留包含 "BB" 的列
        result = df.filter(regex="BB")
        # 期望的 DataFrame 只包含包含 "BB" 的列
        exp = df[[x for x in df.columns if "BB" in x]]
        # 使用测试框架中的函数比较结果与期望值是否一致
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize(
        "name,expected_data",
        [
            ("a", {"a": [1, 2]}),
            ("あ", {"あ": [3, 4]}),
        ],
    )
    def test_filter_unicode(self, name, expected_data):
        # GH13101
        # 创建一个包含 Unicode 列名的 DataFrame
        df = DataFrame({"a": [1, 2], "あ": [3, 4]})
        # 创建一个期望的 DataFrame，根据传入的期望数据
        expected = DataFrame(expected_data)

        # 使用字符串搜索（like）功能过滤 DataFrame，保留列名包含传入的 name 参数的列
        tm.assert_frame_equal(df.filter(like=name), expected)
        # 使用正则表达式搜索（regex）功能过滤 DataFrame，保留列名符合传入的 name 参数的列
        tm.assert_frame_equal(df.filter(regex=name), expected)

    def test_filter_bytestring(self):
        # GH13101
        # 定义一个名称为 "a" 的字节字符串
        name = "a"
        # 创建一个包含字节字符串列名的 DataFrame
        df = DataFrame({b"a": [1, 2], b"b": [3, 4]})
        # 创建一个期望的 DataFrame，只包含列名为 b"a" 的列
        expected = DataFrame({b"a": [1, 2]})

        # 使用字符串搜索（like）功能过滤 DataFrame，保留列名包含字节字符串 name 的列
        tm.assert_frame_equal(df.filter(like=name), expected)
        # 使用正则表达式搜索（regex）功能过滤 DataFrame，保留列名符合字节字符串 name 的列
        tm.assert_frame_equal(df.filter(regex=name), expected)

    def test_filter_corner(self):
        # 创建一个空的 DataFrame
        empty = DataFrame()

        # 使用空列表过滤 DataFrame，结果应与空 DataFrame 相同
        result = empty.filter([])
        tm.assert_frame_equal(result, empty)

        # 使用字符串搜索（like）功能过滤空 DataFrame，结果应与空 DataFrame 相同
        result = empty.filter(like="foo")
        tm.assert_frame_equal(result, empty)

    def test_filter_regex_non_string(self):
        # GH#5798 尝试在非字符串列上进行过滤应该丢弃，而不是引发错误
        # 创建一个具有随机数的 DataFrame，列名包括字符串和整数
        df = DataFrame(np.random.default_rng(2).random((3, 2)), columns=["STRING", 123])
        # 使用正则表达式过滤 DataFrame，保留列名包含 "STRING" 的列
        result = df.filter(regex="STRING")
        # 期望的 DataFrame 只包含列名为 "STRING" 的列
        expected = df[["STRING"]]
        # 使用测试框架中的函数比较结果与期望值是否一致
        tm.assert_frame_equal(result, expected)

    def test_filter_keep_order(self):
        # GH#54980
        # 创建一个 DataFrame，包含列 "A" 和 "B"
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # 使用指定的列顺序过滤 DataFrame，只保留列 "B" 和 "A"
        result = df.filter(items=["B", "A"])
        # 期望的 DataFrame 包含按照指定顺序的列 "B" 和 "A"
        expected = df[["B", "A"]]
        # 使用测试框架中的函数比较结果与期望值是否一致
        tm.assert_frame_equal(result, expected)

    def test_filter_different_dtype(self):
        # GH#54980
        # 创建一个 DataFrame，列名为整数 1 和 2
        df = DataFrame({1: [1, 2, 3], 2: [4, 5, 6]})
        # 使用指定的列名过滤 DataFrame，期望结果为空 DataFrame
        result = df.filter(items=["B", "A"])
        # 期望的 DataFrame 为空，因为过滤的列名不存在
        expected = df[[]]
        # 使用测试框架中的函数比较结果与期望值是否一致
        tm.assert_frame_equal(result, expected)
```