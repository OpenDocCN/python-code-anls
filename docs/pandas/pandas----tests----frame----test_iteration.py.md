# `D:\src\scipysrc\pandas\pandas\tests\frame\test_iteration.py`

```
# 导入 datetime 模块，用于处理日期和时间
import datetime

# 导入 numpy 库，并重命名为 np，用于数值计算
import numpy as np

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 从 pandas.compat 中导入 IS64 和 is_platform_windows 函数
from pandas.compat import (
    IS64,
    is_platform_windows,
)

# 从 pandas 库中导入必要的类和函数
from pandas import (
    Categorical,
    DataFrame,
    Series,
    date_range,
)

# 导入 pandas._testing 模块，并重命名为 tm，用于测试辅助函数
import pandas._testing as tm


# 定义一个测试类 TestIteration，用于测试迭代相关功能
class TestIteration:

    # 测试 float_frame 对象的 keys() 方法
    def test_keys(self, float_frame):
        assert float_frame.keys() is float_frame.columns

    # 测试 DataFrame 对象的 iteritems() 方法
    def test_iteritems(self):
        # 创建一个 DataFrame 对象 df，包含两行数据和重复列名
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
        # 使用 iteritems() 方法迭代 df 的列
        for k, v in df.items():
            # 断言迭代出的 v 是 DataFrame._constructor_sliced 类型
            assert isinstance(v, DataFrame._constructor_sliced)

    # 测试 DataFrame 对象的 items() 方法
    def test_items(self):
        # GH#17213, GH#13918
        # 定义列名列表 cols
        cols = ["a", "b", "c"]
        # 创建一个 DataFrame 对象 df，包含两行数据和自定义列名
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=cols)
        # 使用 items() 方法迭代 df 的列
        for c, (k, v) in zip(cols, df.items()):
            # 断言迭代出的列名 k 等于预期的列名 c
            assert c == k
            # 断言迭代出的 v 是 Series 类型
            assert isinstance(v, Series)
            # 断言 df[k] 等于 v 中的所有值
            assert (df[k] == v).all()

    # 测试 float_string_frame 对象的 items() 方法
    def test_items_names(self, float_string_frame):
        # 使用 items() 方法迭代 float_string_frame 的列
        for k, v in float_string_frame.items():
            # 断言 v 的名称等于其对应的列名 k
            assert v.name == k

    # 测试 float_frame 对象的 iter() 方法
    def test_iter(self, float_frame):
        # 断言 list(float_frame) 等于 float_frame.columns 的列表形式
        assert list(float_frame) == list(float_frame.columns)

    # 测试 float_frame 和 float_string_frame 对象的 iterrows() 方法
    def test_iterrows(self, float_frame, float_string_frame):
        # 使用 iterrows() 方法迭代 float_frame 的行
        for k, v in float_frame.iterrows():
            # 获取 float_frame 中索引为 k 的期望 Series
            exp = float_frame.loc[k]
            # 使用辅助函数 assert_series_equal 检查 v 和 exp 是否相等
            tm.assert_series_equal(v, exp)

        # 使用 iterrows() 方法迭代 float_string_frame 的行
        for k, v in float_string_frame.iterrows():
            # 获取 float_string_frame 中索引为 k 的期望 Series
            exp = float_string_frame.loc[k]
            # 使用辅助函数 assert_series_equal 检查 v 和 exp 是否相等
            tm.assert_series_equal(v, exp)

    # 测试 DataFrame 对象的 iterrows() 方法，以 ISO 8601 格式日期为例
    def test_iterrows_iso8601(self):
        # GH#19671
        # 创建一个包含非 ISO 8601 和 ISO 8601 格式日期的 DataFrame 对象 s
        s = DataFrame(
            {
                "non_iso8601": ["M1701", "M1802", "M1903", "M2004"],
                "iso8601": date_range("2000-01-01", periods=4, freq="ME"),
            }
        )
        # 使用 iterrows() 方法迭代 s 的行
        for k, v in s.iterrows():
            # 获取 s 中索引为 k 的期望 Series
            exp = s.loc[k]
            # 使用辅助函数 assert_series_equal 检查 v 和 exp 是否相等
            tm.assert_series_equal(v, exp)

    # 测试 DataFrame 对象的 iterrows() 方法，处理特殊情况
    def test_iterrows_corner(self):
        # GH#12222
        # 创建一个包含不同类型数据的 DataFrame 对象 df
        df = DataFrame(
            {
                "a": [datetime.datetime(2015, 1, 1)],
                "b": [None],
                "c": [None],
                "d": [""],
                "e": [[]],
                "f": [set()],
                "g": [{}],
            }
        )
        # 创建期望的 Series 对象 expected
        expected = Series(
            [datetime.datetime(2015, 1, 1), None, None, "", [], set(), {}],
            index=list("abcdefg"),
            name=0,
            dtype="object",
        )
        # 使用 next() 函数获取 df.iterrows() 的第一个结果
        _, result = next(df.iterrows())
        # 使用辅助函数 assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 测试 float_frame 对象的 itertuples() 方法
    def test_itertuples(self, float_frame):
        # 使用 itertuples() 方法迭代 float_frame 的行
        for i, tup in enumerate(float_frame.itertuples()):
            # 创建一个 DataFrame._constructor_sliced 类型的 Series 对象 ser
            ser = DataFrame._constructor_sliced(tup[1:])
            # 设置 ser 的名称为 tup 的第一个元素
            ser.name = tup[0]
            # 获取 float_frame 中第 i 行的期望 Series 对象 expected
            expected = float_frame.iloc[i, :].reset_index(drop=True)
            # 使用辅助函数 assert_series_equal 检查 ser 和 expected 是否相等
            tm.assert_series_equal(ser, expected)
    # 定义测试方法：验证在不包含索引的情况下使用DataFrame的itertuples方法
    def test_itertuples_index_false(self):
        # 创建一个包含随机浮点数和整数的DataFrame
        df = DataFrame(
            {"floats": np.random.default_rng(2).standard_normal(5), "ints": range(5)},
            columns=["floats", "ints"],
        )

        # 遍历DataFrame的行，每一行作为一个命名元组返回
        for tup in df.itertuples(index=False):
            assert isinstance(tup[1], int)

    # 定义测试方法：验证在包含重复列名的情况下使用DataFrame的itertuples方法
    def test_itertuples_duplicate_cols(self):
        # 创建一个包含重复列名的DataFrame
        df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
        dfaa = df[["a", "a"]]

        # 验证itertuples方法返回的命名元组中包含重复列名的情况
        assert list(dfaa.itertuples()) == [(0, 1, 1), (1, 2, 2), (2, 3, 3)]

        # 验证在32位Windows环境或非64位平台上，使用itertuples方法返回的结果的表示
        if not (is_platform_windows() or not IS64):
            assert (
                repr(list(df.itertuples(name=None)))
                == "[(0, 1, 4), (1, 2, 5), (2, 3, 6)]"
            )

    # 定义测试方法：验证使用DataFrame的itertuples方法时设置元组名
    def test_itertuples_tuple_name(self):
        # 创建一个包含列'a'和'b'的DataFrame
        df = DataFrame(data={"a": [1, 2, 3], "b": [4, 5, 6]})
        # 获取第一个命名元组，并设置元组名为'TestName'
        tup = next(df.itertuples(name="TestName"))
        # 验证命名元组的字段名是否正确
        assert tup._fields == ("Index", "a", "b")
        # 验证命名元组的内容是否与其字段名对应
        assert (tup.Index, tup.a, tup.b) == tup
        # 验证命名元组的类型名是否为'TestName'
        assert type(tup).__name__ == "TestName"

    # 定义测试方法：验证在包含不允许的列标签的情况下使用DataFrame的itertuples方法
    def test_itertuples_disallowed_col_labels(self):
        # 创建一个包含列名'def'和'return'的DataFrame
        df = DataFrame(data={"def": [1, 2, 3], "return": [4, 5, 6]})
        # 获取第一个命名元组，并设置元组名为'TestName'
        tup2 = next(df.itertuples(name="TestName"))
        # 验证命名元组的内容是否正确
        assert tup2 == (0, 1, 4)
        # 验证命名元组的字段名是否正确
        assert tup2._fields == ("Index", "_1", "_2")

    # 定义测试方法：验证在Python 2/3兼容的环境下使用DataFrame的itertuples方法
    @pytest.mark.parametrize("limit", [254, 255, 1024])
    @pytest.mark.parametrize("index", [True, False])
    def test_itertuples_py2_3_field_limit_namedtuple(self, limit, index):
        # GH#28282
        # 创建一个包含指定限制数量字段的字典的DataFrame
        df = DataFrame([{f"foo_{i}": f"bar_{i}" for i in range(limit)}])
        # 获取第一个命名元组，根据索引设置是否包含索引
        result = next(df.itertuples(index=index))
        # 验证结果是否为元组类型
        assert isinstance(result, tuple)
        # 验证结果是否包含'_fields'属性
        assert hasattr(result, "_fields")

    # 定义测试方法：验证使用分类数据类型的DataFrame进行序列化操作
    def test_sequence_like_with_categorical(self):
        # GH#7839
        # 创建一个包含'id'和'raw_grade'列的DataFrame
        df = DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
        )
        # 将'raw_grade'列转换为分类数据类型'grade'
        df["grade"] = Categorical(df["raw_grade"])

        # 基本序列化测试
        result = list(df.grade.values)
        expected = np.array(df.grade.values).tolist()
        # 检查序列化结果是否近似相等
        tm.assert_almost_equal(result, expected)

        # 迭代操作
        for t in df.itertuples(index=False):
            str(t)

        for row, s in df.iterrows():
            str(s)

        for c, col in df.items():
            str(col)
```