# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_equals.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 从 Pandas 库中导入 DataFrame 和 date_range 函数
    DataFrame,
    date_range,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


class TestEquals:
    def test_dataframe_not_equal(self):
        # see GH#28839
        df1 = DataFrame({"a": [1, 2], "b": ["s", "d"]})  # 创建 DataFrame df1
        df2 = DataFrame({"a": ["s", "d"], "b": [1, 2]})  # 创建 DataFrame df2
        assert df1.equals(df2) is False  # 断言 df1 不等于 df2

    def test_equals_different_blocks(self, using_infer_string):
        # GH#9330
        df0 = DataFrame({"A": ["x", "y"], "B": [1, 2], "C": ["w", "z"]})  # 创建 DataFrame df0
        df1 = df0.reset_index()[["A", "B", "C"]]  # 重置索引并选择特定列，创建 DataFrame df1
        if not using_infer_string:
            # this assert verifies that the above operations have
            # induced a block rearrangement
            assert df0._mgr.blocks[0].dtype != df1._mgr.blocks[0].dtype  # 验证操作是否引起了块的重新排列

        # do the real tests
        tm.assert_frame_equal(df0, df1)  # 使用 Pandas 测试模块验证 df0 和 df1 是否相等
        assert df0.equals(df1)  # 断言 df0 等于 df1
        assert df1.equals(df0)  # 断言 df1 等于 df0

    def test_equals(self):
        # Add object dtype column with nans
        index = np.random.default_rng(2).random(10)  # 生成随机索引
        df1 = DataFrame(  # 创建 DataFrame df1
            np.random.default_rng(2).random(10), index=index, columns=["floats"]
        )
        df1["text"] = "the sky is so blue. we could use more chocolate.".split()  # 添加文本列
        df1["start"] = date_range("2000-1-1", periods=10, freq="min")  # 添加时间起始列
        df1["end"] = date_range("2000-1-1", periods=10, freq="D")  # 添加时间结束列
        df1["diff"] = df1["end"] - df1["start"]  # 计算时间差列
        # Explicitly cast to object, to avoid implicit cast when setting np.nan
        df1["bool"] = (np.arange(10) % 3 == 0).astype(object)  # 添加布尔列
        df1.loc[::2] = np.nan  # 将每隔两行设为 NaN
        df2 = df1.copy()  # 复制 DataFrame df1 到 df2
        assert df1["text"].equals(df2["text"])  # 断言 df1 的文本列等于 df2 的文本列
        assert df1["start"].equals(df2["start"])  # 断言 df1 的起始时间列等于 df2 的起始时间列
        assert df1["end"].equals(df2["end"])  # 断言 df1 的结束时间列等于 df2 的结束时间列
        assert df1["diff"].equals(df2["diff"])  # 断言 df1 的时间差列等于 df2 的时间差列
        assert df1["bool"].equals(df2["bool"])  # 断言 df1 的布尔列等于 df2 的布尔列
        assert df1.equals(df2)  # 断言 df1 等于 df2
        assert not df1.equals(object)  # 断言 df1 不等于对象类型

        # different dtype
        different = df1.copy()  # 复制 DataFrame df1 到 different
        different["floats"] = different["floats"].astype("float32")  # 修改不同数据类型的列
        assert not df1.equals(different)  # 断言 df1 不等于 different

        # different index
        different_index = -index  # 创建不同的索引
        different = df2.set_index(different_index)  # 使用不同索引设置 df2
        assert not df1.equals(different)  # 断言 df1 不等于 different

        # different columns
        different = df2.copy()  # 复制 DataFrame df2 到 different
        different.columns = df2.columns[::-1]  # 修改不同的列名顺序
        assert not df1.equals(different)  # 断言 df1 不等于 different

        # DatetimeIndex
        index = date_range("2000-1-1", periods=10, freq="min")  # 创建时间索引
        df1 = df1.set_index(index)  # 使用时间索引设置 df1
        df2 = df1.copy()  # 复制 DataFrame df1 到 df2
        assert df1.equals(df2)  # 断言 df1 等于 df2

        # MultiIndex
        df3 = df1.set_index(["text"], append=True)  # 使用多重索引设置 df3
        df2 = df1.set_index(["text"], append=True)  # 使用多重索引设置 df2
        assert df3.equals(df2)  # 断言 df3 等于 df2

        df2 = df1.set_index(["floats"], append=True)  # 使用多重索引设置 df2
        assert not df3.equals(df2)  # 断言 df3 不等于 df2

        # NaN in index
        df3 = df1.set_index(["floats"], append=True)  # 使用多重索引设置 df3
        df2 = df1.set_index(["floats"], append=True)  # 使用多重索引设置 df2
        assert df3.equals(df2)  # 断言 df3 等于 df2
```