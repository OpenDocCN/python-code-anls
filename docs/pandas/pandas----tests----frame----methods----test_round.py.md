# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_round.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试

import pandas as pd  # 导入 Pandas 库，用于数据分析
from pandas import (  # 从 Pandas 中导入 DataFrame、Series 和 date_range 函数
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块

class TestDataFrameRound:
    def test_round_numpy(self):
        # GH#12600
        # 创建一个包含浮点数的 DataFrame
        df = DataFrame([[1.53, 1.36], [0.06, 7.01]])
        # 使用 NumPy 的 round 函数对 DataFrame 进行四舍五入到整数
        out = np.round(df, decimals=0)
        # 创建预期的四舍五入结果的 DataFrame
        expected = DataFrame([[2.0, 1.0], [0.0, 7.0]])
        # 使用测试工具检查 out 和 expected 是否相等
        tm.assert_frame_equal(out, expected)

        # 准备一个错误消息
        msg = "the 'out' parameter is not supported"
        # 使用 Pytest 来检查是否抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            np.round(df, decimals=0, out=df)

    def test_round_numpy_with_nan(self):
        # See GH#14197
        # 创建一个包含 NaN 值的 Series，并将其转换为 DataFrame
        df = Series([1.53, np.nan, 0.06]).to_frame()
        # 使用测试工具禁用 NaN 警告，然后进行四舍五入
        with tm.assert_produces_warning(None):
            result = df.round()
        # 创建预期的四舍五入结果的 DataFrame
        expected = Series([2.0, np.nan, 0.0]).to_frame()
        # 使用测试工具检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_round_mixed_type(self):
        # GH#11885
        # 创建一个包含不同数据类型的 DataFrame
        df = DataFrame(
            {
                "col1": [1.1, 2.2, 3.3, 4.4],
                "col2": ["1", "a", "c", "f"],
                "col3": date_range("20111111", periods=4),
            }
        )
        # 创建一个预期的四舍五入到整数的 DataFrame
        round_0 = DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0],
                "col2": ["1", "a", "c", "f"],
                "col3": date_range("20111111", periods=4),
            }
        )
        # 使用测试工具检查不同参数设置下的四舍五入结果
        tm.assert_frame_equal(df.round(), round_0)
        tm.assert_frame_equal(df.round(1), df)
        tm.assert_frame_equal(df.round({"col1": 1}), df)
        tm.assert_frame_equal(df.round({"col1": 0}), round_0)
        tm.assert_frame_equal(df.round({"col1": 0, "col2": 1}), round_0)
        tm.assert_frame_equal(df.round({"col3": 1}), df)

    def test_round_with_duplicate_columns(self):
        # GH#11611
        # 创建一个具有重复列名的随机值 DataFrame
        df = DataFrame(
            np.random.default_rng(2).random([3, 3]),
            columns=["A", "B", "C"],
            index=["first", "second", "third"],
        )
        # 将两个相同的 DataFrame 拼接在一起
        dfs = pd.concat((df, df), axis=1)
        # 对拼接后的 DataFrame 进行四舍五入
        rounded = dfs.round()
        # 使用测试工具检查结果的索引是否相等
        tm.assert_index_equal(rounded.index, dfs.index)

        # 准备一个包含重复列名的 Series 作为小数位参数
        decimals = Series([1, 0, 2], index=["A", "B", "A"])
        msg = "Index of decimals must be unique"
        # 使用 Pytest 来检查是否抛出 ValueError，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            df.round(decimals)

    def test_round_builtin(self):
        # GH#11763
        # 准备一个测试用的 DataFrame
        df = DataFrame({"col1": [1.123, 2.123, 3.123], "col2": [1.234, 2.234, 3.234]})

        # 默认四舍五入到整数 (即 decimals=0)
        expected_rounded = DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [1.0, 2.0, 3.0]})
        # 使用测试工具检查默认四舍五入结果是否与预期相等
        tm.assert_frame_equal(round(df), expected_rounded)
    # 定义测试方法，用于测试非唯一分类数据的四舍五入操作
    def test_round_nonunique_categorical(self):
        # 查看GitHub问题编号 GH#21809
        idx = pd.CategoricalIndex(["low"] * 3 + ["hi"] * 3)
        # 创建一个包含随机数据的DataFrame，列名为 'abc'
        df = DataFrame(np.random.default_rng(2).random((6, 3)), columns=list("abc"))

        # 期望的DataFrame结果，数据四舍五入保留三位小数
        expected = df.round(3)
        # 将索引设置为分类索引idx
        expected.index = idx

        # 复制DataFrame并设置分类索引为idx
        df_categorical = df.copy().set_index(idx)
        # 断言df_categorical的形状为(6, 3)
        assert df_categorical.shape == (6, 3)
        # 对df_categorical进行四舍五入保留三位小数
        result = df_categorical.round(3)
        # 断言result的形状为(6, 3)
        assert result.shape == (6, 3)

        # 使用测试工具方法tm.assert_frame_equal比较result和expected
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试区间类别列的四舍五入操作
    def test_round_interval_category_columns(self):
        # GitHub问题编号 GH#30063
        # 创建一个包含区间的分类索引
        columns = pd.CategoricalIndex(pd.interval_range(0, 2))
        # 创建一个包含数据的DataFrame，数据保留两位小数
        df = DataFrame([[0.66, 1.1], [0.3, 0.25]], columns=columns)

        # 对DataFrame进行四舍五入操作
        result = df.round()
        # 期望的DataFrame结果，所有数据四舍五入为整数
        expected = DataFrame([[1.0, 1.0], [0.0, 0.0]], columns=columns)
        # 使用测试工具方法tm.assert_frame_equal比较result和expected
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试空DataFrame的四舍五入操作
    def test_round_empty_not_input(self):
        # GitHub问题编号 GH#51032
        # 创建一个空的DataFrame
        df = DataFrame()
        # 对空DataFrame进行四舍五入操作
        result = df.round()
        # 使用测试工具方法tm.assert_frame_equal比较df和result，确保它们相等
        tm.assert_frame_equal(df, result)
        # 断言df和result不是同一个对象
        assert df is not result
```