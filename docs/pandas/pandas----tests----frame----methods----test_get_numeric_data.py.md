# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_get_numeric_data.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

import pandas as pd  # 导入 Pandas 库，用于数据分析和处理
from pandas import (  # 导入 Pandas 的特定子模块和类
    Categorical,
    DataFrame,
    Index,
    Series,
    Timestamp,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.core.arrays import IntervalArray  # 导入 Pandas 的 IntervalArray 类


class TestGetNumericData:
    def test_get_numeric_data_preserve_dtype(self):
        # get the numeric data
        obj = DataFrame({"A": [1, "2", 3.0]}, columns=Index(["A"], dtype="object"))
        result = obj._get_numeric_data()  # 获取 DataFrame 中的数值数据
        expected = DataFrame(dtype=object, index=pd.RangeIndex(3), columns=[])
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果的正确性

    def test_get_numeric_data(self, using_infer_string):
        datetime64name = np.dtype("M8[s]").name  # 获取 numpy 的 datetime64 类型的名称
        objectname = np.dtype(np.object_).name  # 获取 numpy 的 object 类型的名称

        df = DataFrame(
            {"a": 1.0, "b": 2, "c": "foo", "f": Timestamp("20010102")},
            index=np.arange(10),
        )
        result = df.dtypes  # 获取 DataFrame 各列的数据类型
        expected = Series(
            [
                np.dtype("float64"),
                np.dtype("int64"),
                np.dtype(objectname) if not using_infer_string else "string",
                np.dtype(datetime64name),
            ],
            index=["a", "b", "c", "f"],
        )
        tm.assert_series_equal(result, expected)  # 使用测试模块验证 Series 的相等性

        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                "d": np.array([1.0] * 10, dtype="float32"),
                "e": np.array([1] * 10, dtype="int32"),
                "f": np.array([1] * 10, dtype="int16"),
                "g": Timestamp("20010102"),
            },
            index=np.arange(10),
        )

        result = df._get_numeric_data()  # 获取 DataFrame 中的数值数据
        expected = df.loc[:, ["a", "b", "d", "e", "f"]]
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果的正确性

        only_obj = df.loc[:, ["c", "g"]]
        result = only_obj._get_numeric_data()  # 获取 DataFrame 中的数值数据
        expected = df.loc[:, []]
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果的正确性

        df = DataFrame.from_dict({"a": [1, 2], "b": ["foo", "bar"], "c": [np.pi, np.e]})
        result = df._get_numeric_data()  # 获取 DataFrame 中的数值数据
        expected = DataFrame.from_dict({"a": [1, 2], "c": [np.pi, np.e]})
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果的正确性

        df = result.copy()
        result = df._get_numeric_data()  # 获取 DataFrame 中的数值数据
        expected = df
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证结果的正确性

    def test_get_numeric_data_mixed_dtype(self):
        # numeric and object columns

        df = DataFrame(
            {
                "a": [1, 2, 3],
                "b": [True, False, True],
                "c": ["foo", "bar", "baz"],
                "d": [None, None, None],
                "e": [3.14, 0.577, 2.773],
            }
        )
        result = df._get_numeric_data()  # 获取 DataFrame 中的数值数据
        tm.assert_index_equal(result.columns, Index(["a", "b", "e"]))  # 使用测试模块验证结果的正确性
    # 定义一个测试方法，用于测试获取数据框中数值数据的扩展数据类型
    def test_get_numeric_data_extension_dtype(self):
        # 标记：GH#22290，表示这是与 GitHub 问题编号为22290 相关的测试用例
        # 创建一个数据框 DataFrame，包含不同数据类型的列
        df = DataFrame(
            {
                "A": pd.array([-10, np.nan, 0, 10, 20, 30], dtype="Int64"),
                "B": Categorical(list("abcabc")),
                "C": pd.array([0, 1, 2, 3, np.nan, 5], dtype="UInt8"),
                "D": IntervalArray.from_breaks(range(7)),
            }
        )
        # 调用数据框的方法 _get_numeric_data()，获取其中的数值数据
        result = df._get_numeric_data()
        # 创建一个期望的数据框 expected，只包含列'A'和'C'
        expected = df.loc[:, ["A", "C"]]
        # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
```