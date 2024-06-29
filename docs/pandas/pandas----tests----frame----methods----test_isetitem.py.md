# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_isetitem.py`

```
`
import pytest  # 导入 pytest 模块

from pandas import (  # 导入 pandas 模块中的 DataFrame 和 Series 类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块 pandas._testing


class TestDataFrameSetItem:
    def test_isetitem_ea_df(self):
        # GH#49922
        # 创建一个包含两行三列的 DataFrame 对象
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        # 创建一个包含两行两列的 DataFrame 对象，指定数据类型为 Int64
        rhs = DataFrame([[11, 12], [13, 14]], dtype="Int64")

        # 使用 isetitem 方法将 rhs 的值设置到 df 中的指定位置
        df.isetitem([0, 1], rhs)
        # 创建期望的 DataFrame 对象
        expected = DataFrame(
            {
                0: Series([11, 13], dtype="Int64"),
                1: Series([12, 14], dtype="Int64"),
                2: [3, 6],
            }
        )
        # 断言 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

    def test_isetitem_ea_df_scalar_indexer(self):
        # GH#49922
        # 创建一个包含两行三列的 DataFrame 对象
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        # 创建一个包含两行一列的 DataFrame 对象，指定数据类型为 Int64
        rhs = DataFrame([[11], [13]], dtype="Int64")

        # 使用 isetitem 方法将 rhs 的值设置到 df 中的指定位置
        df.isetitem(2, rhs)
        # 创建期望的 DataFrame 对象
        expected = DataFrame(
            {
                0: [1, 4],
                1: [2, 5],
                2: Series([11, 13], dtype="Int64"),
            }
        )
        # 断言 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

    def test_isetitem_dimension_mismatch(self):
        # GH#51701
        # 创建一个包含三列的 DataFrame 对象
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        # 复制 df 到 value
        value = df.copy()
        # 使用 isetitem 方法尝试将 value 的值设置到 df 中的指定位置，期望抛出 ValueError 异常
        with pytest.raises(ValueError, match="Got 2 positions but value has 3 columns"):
            df.isetitem([1, 2], value)

        # 再次复制 df 到 value
        value = df.copy()
        # 使用 isetitem 方法尝试将 value 的值设置到 df 中的指定位置，期望抛出 ValueError 异常
        with pytest.raises(ValueError, match="Got 2 positions but value has 1 columns"):
            df.isetitem([1, 2], value[["a"]])
```