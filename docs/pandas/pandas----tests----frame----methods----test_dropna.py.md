# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_dropna.py`

```
# 导入 datetime 模块，用于处理日期和时间
import datetime

# 导入 dateutil 模块，提供了强大的日期和时间处理工具
import dateutil
# 导入 numpy 库，并使用 np 作为别名，用于数值计算和数组操作
import numpy as np
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas 库，并使用 pd 作为别名，用于数据分析和处理
import pandas as pd
# 从 pandas 中导入 DataFrame 和 Series 类
from pandas import (
    DataFrame,
    Series,
)
# 导入 pandas._testing 模块，提供了一些用于测试的实用工具函数和类
import pandas._testing as tm


class TestDataFrameMissingData:
    def test_dropEmptyRows(self, float_frame):
        # 获取 float_frame 的行数 N
        N = len(float_frame.index)
        # 使用随机数生成器生成 N 个标准正态分布的随机数
        mat = np.random.default_rng(2).standard_normal(N)
        # 将前 5 个随机数设为 NaN
        mat[:5] = np.nan

        # 创建 DataFrame 对象 frame，其中包含一列名为 "foo" 的数据 mat
        frame = DataFrame({"foo": mat}, index=float_frame.index)
        # 创建 Series 对象 original，与 frame["foo"] 具有相同的索引和数据
        original = Series(mat, index=float_frame.index, name="foo")
        # 通过 dropna() 方法删除 NaN 值，得到期望的 Series 对象 expected
        expected = original.dropna()
        # 复制 frame，得到 inplace_frame1 和 inplace_frame2 两个副本
        inplace_frame1, inplace_frame2 = frame.copy(), frame.copy()

        # 使用 dropna() 方法删除所有 NaN 行，得到较小的 DataFrame 对象 smaller_frame
        smaller_frame = frame.dropna(how="all")
        # 检查 frame["foo"] 与 original 是否相等
        tm.assert_series_equal(frame["foo"], original)
        # 返回值为 None，因为 inplace=True，修改了 inplace_frame1
        return_value = inplace_frame1.dropna(how="all", inplace=True)
        # 检查 smaller_frame["foo"] 与 expected 是否相等
        tm.assert_series_equal(smaller_frame["foo"], expected)
        # 检查 inplace_frame1["foo"] 与 expected 是否相等
        tm.assert_series_equal(inplace_frame1["foo"], expected)
        # 确保 return_value 为 None
        assert return_value is None

        # 使用 dropna() 方法删除所有 NaN 行，只在 "foo" 列中检查 NaN 值，得到 smaller_frame
        smaller_frame = frame.dropna(how="all", subset=["foo"])
        # 返回值为 None，因为 inplace=True，修改了 inplace_frame2
        return_value = inplace_frame2.dropna(how="all", subset=["foo"], inplace=True)
        # 检查 smaller_frame["foo"] 与 expected 是否相等
        tm.assert_series_equal(smaller_frame["foo"], expected)
        # 检查 inplace_frame2["foo"] 与 expected 是否相等
        tm.assert_series_equal(inplace_frame2["foo"], expected)
        # 确保 return_value 为 None
        assert return_value is None

    def test_dropIncompleteRows(self, float_frame):
        # 获取 float_frame 的行数 N
        N = len(float_frame.index)
        # 使用随机数生成器生成 N 个标准正态分布的随机数
        mat = np.random.default_rng(2).standard_normal(N)
        # 将前 5 个随机数设为 NaN
        mat[:5] = np.nan

        # 创建 DataFrame 对象 frame，其中包含一列名为 "foo" 的数据 mat
        frame = DataFrame({"foo": mat}, index=float_frame.index)
        # 在 frame 中添加一列名为 "bar"，并设置其所有值为 5
        frame["bar"] = 5
        # 创建 Series 对象 original，与 frame["foo"] 具有相同的索引和数据
        original = Series(mat, index=float_frame.index, name="foo")
        # 复制 frame，得到 inp_frame1 和 inp_frame2 两个副本
        inp_frame1, inp_frame2 = frame.copy(), frame.copy()

        # 使用 dropna() 方法删除所有包含 NaN 的行，得到较小的 DataFrame 对象 smaller_frame
        smaller_frame = frame.dropna()
        # 检查 frame["foo"] 与 original 是否相等
        tm.assert_series_equal(frame["foo"], original)
        # 返回值为 None，因为 inplace=True，修改了 inp_frame1
        return_value = inp_frame1.dropna(inplace=True)

        # 创建期望的 Series 对象 exp，包含 mat[5:] 的数据
        exp = Series(mat[5:], index=float_frame.index[5:], name="foo")
        # 检查 smaller_frame["foo"] 与 exp 是否相等
        tm.assert_series_equal(smaller_frame["foo"], exp)
        # 检查 inp_frame1["foo"] 与 exp 是否相等
        tm.assert_series_equal(inp_frame1["foo"], exp)
        # 确保 return_value 为 None
        assert return_value is None

        # 使用 dropna() 方法删除所有包含 NaN 的行，只在 "bar" 列中检查 NaN 值，得到 samesize_frame
        samesize_frame = frame.dropna(subset=["bar"])
        # 检查 frame["foo"] 与 original 是否相等
        tm.assert_series_equal(frame["foo"], original)
        # 确保 frame["bar"] 中的所有值均为 5
        assert (frame["bar"] == 5).all()
        # 返回值为 None，因为 inplace=True，修改了 inp_frame2
        return_value = inp_frame2.dropna(subset=["bar"], inplace=True)
        # 检查 samesize_frame 和 float_frame 的索引是否相等
        tm.assert_index_equal(samesize_frame.index, float_frame.index)
        # 检查 inp_frame2 和 float_frame 的索引是否相等
        tm.assert_index_equal(inp_frame2.index, float_frame.index)
        # 确保 return_value 为 None
        assert return_value is None
    # 定义一个单元测试方法，用于测试数据框中的缺失值删除功能
    def test_dropna(self):
        # 创建一个 6x4 的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)))
        # 将前两行第三列设置为 NaN
        df.iloc[:2, 2] = np.nan

        # 删除所有包含 NaN 的列，返回删除后的数据框
        dropped = df.dropna(axis=1)
        # 创建期望的数据框，只包含第 0、1、3 列
        expected = df.loc[:, [0, 1, 3]]
        # 复制数据框
        inp = df.copy()
        # 原地删除所有包含 NaN 的列
        return_value = inp.dropna(axis=1, inplace=True)
        # 断言删除后的数据框与期望的数据框相等
        tm.assert_frame_equal(dropped, expected)
        # 断言原数据框也与期望的数据框相等
        tm.assert_frame_equal(inp, expected)
        # 确保 inplace 操作返回 None
        assert return_value is None

        # 删除所有包含 NaN 的行，返回删除后的数据框
        dropped = df.dropna(axis=0)
        # 创建期望的数据框，只包含索引为 2 到 5 的行
        expected = df.loc[list(range(2, 6))]
        # 复制数据框
        inp = df.copy()
        # 原地删除所有包含 NaN 的行
        return_value = inp.dropna(axis=0, inplace=True)
        # 断言删除后的数据框与期望的数据框相等
        tm.assert_frame_equal(dropped, expected)
        # 断言原数据框也与期望的数据框相等
        tm.assert_frame_equal(inp, expected)
        # 确保 inplace 操作返回 None
        assert return_value is None

        # 删除包含 NaN 的列，但要求至少有 5 个非 NaN 值
        dropped = df.dropna(axis=1, thresh=5)
        # 创建期望的数据框，只包含第 0、1、3 列
        expected = df.loc[:, [0, 1, 3]]
        # 复制数据框
        inp = df.copy()
        # 原地删除包含 NaN 的列，但要求至少有 5 个非 NaN 值
        return_value = inp.dropna(axis=1, thresh=5, inplace=True)
        # 断言删除后的数据框与期望的数据框相等
        tm.assert_frame_equal(dropped, expected)
        # 断言原数据框也与期望的数据框相等
        tm.assert_frame_equal(inp, expected)
        # 确保 inplace 操作返回 None
        assert return_value is None

        # 删除包含 NaN 的行，但要求至少有 4 个非 NaN 值
        dropped = df.dropna(axis=0, thresh=4)
        # 创建期望的数据框，只包含索引为 2 到 5 的行
        expected = df.loc[range(2, 6)]
        # 复制数据框
        inp = df.copy()
        # 原地删除包含 NaN 的行，但要求至少有 4 个非 NaN 值
        return_value = inp.dropna(axis=0, thresh=4, inplace=True)
        # 断言删除后的数据框与期望的数据框相等
        tm.assert_frame_equal(dropped, expected)
        # 断言原数据框也与期望的数据框相等
        tm.assert_frame_equal(inp, expected)
        # 确保 inplace 操作返回 None
        assert return_value is None

        # 删除所有列，因为每列都至少有 4 个非 NaN 值
        dropped = df.dropna(axis=1, thresh=4)
        # 断言删除后的数据框与原数据框相等
        tm.assert_frame_equal(dropped, df)

        # 删除所有列，因为每列都至少有 3 个非 NaN 值
        dropped = df.dropna(axis=1, thresh=3)
        # 断言删除后的数据框与原数据框相等
        tm.assert_frame_equal(dropped, df)

        # 删除包含 NaN 的行，但只考虑第 0、1、3 列
        dropped = df.dropna(axis=0, subset=[0, 1, 3])
        # 复制数据框
        inp = df.copy()
        # 原地删除包含 NaN 的行，但只考虑第 0、1、3 列
        return_value = inp.dropna(axis=0, subset=[0, 1, 3], inplace=True)
        # 断言删除后的数据框与原数据框相等
        tm.assert_frame_equal(dropped, df)
        # 断言原数据框也与期望的数据框相等
        tm.assert_frame_equal(inp, df)
        # 确保 inplace 操作返回 None
        assert return_value is None

        # 删除所有列，因为没有全是 NaN 的列
        dropped = df.dropna(axis=1, how="all")
        # 断言删除后的数据框与原数据框相等
        tm.assert_frame_equal(dropped, df)

        # 将第 2 列所有值设置为 NaN
        df[2] = np.nan
        # 删除所有列，因为没有全是 NaN 的列
        dropped = df.dropna(axis=1, how="all")
        # 创建期望的数据框，只包含第 0、1、3 列
        expected = df.loc[:, [0, 1, 3]]
        # 断言删除后的数据框与期望的数据框相等
        tm.assert_frame_equal(dropped, expected)

        # 测试不良输入，删除不存在的轴 3
        msg = "No axis named 3 for object type DataFrame"
        # 确保引发 ValueError 异常并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            df.dropna(axis=3)
    def test_drop_and_dropna_caching(self):
        # 测试缓存更新是否正常工作

        # 创建原始Series和期望的Series
        original = Series([1, 2, np.nan], name="A")
        expected = Series([1, 2], dtype=original.dtype, name="A")

        # 创建DataFrame，并复制一份
        df = DataFrame({"A": original.values.copy()})
        df2 = df.copy()

        # 调用 dropna() 方法，不改变原始数据
        df["A"].dropna()
        tm.assert_series_equal(df["A"], original)

        # 获取 Series 对象并调用 dropna() 方法，同时修改了原始数据
        ser = df["A"]
        return_value = ser.dropna(inplace=True)
        tm.assert_series_equal(ser, expected)
        tm.assert_series_equal(df["A"], original)
        assert return_value is None

        # 调用 drop() 方法，不改变原始数据
        df2["A"].drop([1])
        tm.assert_series_equal(df2["A"], original)

        # 获取 Series 对象并调用 drop() 方法，同时修改了原始数据
        ser = df2["A"]
        return_value = ser.drop([1], inplace=True)
        tm.assert_series_equal(ser, original.drop([1]))
        tm.assert_series_equal(df2["A"], original)
        assert return_value is None

    def test_dropna_corner(self, float_frame):
        # 测试异常输入情况

        # 使用 pytest 检测是否抛出预期的 ValueError 异常
        msg = "invalid how option: foo"
        with pytest.raises(ValueError, match=msg):
            float_frame.dropna(how="foo")

        # 使用 pytest 检测是否抛出预期的 KeyError 异常
        with pytest.raises(KeyError, match=r"^\['X'\]$"):
            float_frame.dropna(subset=["A", "X"])

    def test_dropna_multiple_axes(self):
        # 测试在多个轴上应用 dropna() 方法的情况

        # 使用 pytest 检测是否抛出预期的 TypeError 异常
        df = DataFrame(
            [
                [1, np.nan, 2, 3],
                [4, np.nan, 5, 6],
                [np.nan, np.nan, np.nan, np.nan],
                [7, np.nan, 8, 9],
            ]
        )

        # 对于多个轴参数，应抛出 TypeError 异常
        with pytest.raises(TypeError, match="supplying multiple axes"):
            df.dropna(how="all", axis=[0, 1])
        with pytest.raises(TypeError, match="supplying multiple axes"):
            df.dropna(how="all", axis=(0, 1))

        # 在 inplace=True 模式下，同样应该抛出 TypeError 异常
        inp = df.copy()
        with pytest.raises(TypeError, match="supplying multiple axes"):
            inp.dropna(how="all", axis=(0, 1), inplace=True)

    def test_dropna_tz_aware_datetime(self):
        # 测试包含时区信息的日期时间数据的 dropna() 方法

        # 创建包含时区信息的日期时间数据
        df = DataFrame()
        dt1 = datetime.datetime(2015, 1, 1, tzinfo=dateutil.tz.tzutc())
        dt2 = datetime.datetime(2015, 2, 2, tzinfo=dateutil.tz.tzutc())
        df["Time"] = [dt1]

        # 执行 dropna() 操作，并检查结果是否符合预期
        result = df.dropna(axis=0)
        expected = DataFrame({"Time": [dt1]})
        tm.assert_frame_equal(result, expected)

        # 测试包含空值的日期时间数据的 dropna() 方法
        df = DataFrame({"Time": [dt1, None, np.nan, dt2]})
        result = df.dropna(axis=0)
        expected = DataFrame([dt1, dt2], columns=["Time"], index=[0, 3])
        tm.assert_frame_equal(result, expected)

    def test_dropna_categorical_interval_index(self):
        # 测试包含分类和区间索引的数据的 dropna() 方法

        # 创建带有区间索引和分类的 DataFrame
        ii = pd.IntervalIndex.from_breaks([0, 2.78, 3.14, 6.28])
        ci = pd.CategoricalIndex(ii)
        df = DataFrame({"A": list("abc")}, index=ci)

        # 执行 dropna() 操作，并检查结果是否符合预期
        expected = df
        result = df.dropna()
        tm.assert_frame_equal(result, expected)
    def test_dropna_with_duplicate_columns(self):
        df = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(5),
                "B": np.random.default_rng(2).standard_normal(5),
                "C": np.random.default_rng(2).standard_normal(5),
                "D": ["a", "b", "c", "d", "e"],
            }
        )
        df.iloc[2, [0, 1, 2]] = np.nan  # 在第三行（索引为2）的A、B、C列设置为NaN
        df.iloc[0, 0] = np.nan  # 在第一行（索引为0）的A列设置为NaN
        df.iloc[1, 1] = np.nan  # 在第二行（索引为1）的B列设置为NaN
        df.iloc[:, 3] = np.nan  # 第四列（索引为3，即D列）设置为NaN
        expected = df.dropna(subset=["A", "B", "C"], how="all")  # 按照"A", "B", "C"列任意一个NaN就删除行

        df.columns = ["A", "A", "B", "C"]  # 修改列名，将第一列和第二列的名称改为"A"

        result = df.dropna(subset=["A", "C"], how="all")  # 按照"A", "C"列任意一个NaN就删除行
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

    def test_set_single_column_subset(self):
        # GH 41021
        df = DataFrame({"A": [1, 2, 3], "B": list("abc"), "C": [4, np.nan, 5]})
        expected = DataFrame(
            {"A": [1, 3], "B": list("ac"), "C": [4.0, 5.0]}, index=[0, 2]
        )
        result = df.dropna(subset="C")  # 删除"C"列中包含NaN的行
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

    def test_single_column_not_present_in_axis(self):
        # GH 41021
        df = DataFrame({"A": [1, 2, 3]})

        # Column not present
        with pytest.raises(KeyError, match="['D']"):
            df.dropna(subset="D", axis=0)  # 在axis=0轴上，由于"D"列不存在，预期抛出KeyError异常

    def test_subset_is_nparray(self):
        # GH 41021
        df = DataFrame({"A": [1, 2, np.nan], "B": list("abc"), "C": [4, np.nan, 5]})
        expected = DataFrame({"A": [1.0], "B": ["a"], "C": [4.0]})
        result = df.dropna(subset=np.array(["A", "C"]))  # 按照"A", "C"列任意一个NaN就删除行
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

    def test_no_nans_in_frame(self, axis):
        # GH#41965
        df = DataFrame([[1, 2], [3, 4]], columns=pd.RangeIndex(0, 2))
        expected = df.copy()
        result = df.dropna(axis=axis)  # 删除所有包含NaN的行或列，取决于axis参数
        tm.assert_frame_equal(result, expected, check_index_type=True)  # 断言结果与期望相等

    def test_how_thresh_param_incompatible(self):
        # GH46575
        df = DataFrame([1, 2, pd.NA])
        msg = "You cannot set both the how and thresh arguments at the same time"
        with pytest.raises(TypeError, match=msg):
            df.dropna(how="all", thresh=2)  # 同时设置how和thresh参数时抛出TypeError异常

        with pytest.raises(TypeError, match=msg):
            df.dropna(how="any", thresh=2)  # 同时设置how和thresh参数时抛出TypeError异常

        with pytest.raises(TypeError, match=msg):
            df.dropna(how=None, thresh=None)  # 同时设置how和thresh参数时抛出TypeError异常

    @pytest.mark.parametrize("val", [1, 1.5])
    def test_dropna_ignore_index(self, val):
        # GH#31725
        df = DataFrame({"a": [1, 2, val]}, index=[3, 2, 1])
        result = df.dropna(ignore_index=True)  # 删除所有包含NaN的行，并重置索引
        expected = DataFrame({"a": [1, 2, val]})
        tm.assert_frame_equal(result, expected)  # 断言结果与期望相等

        df.dropna(ignore_index=True, inplace=True)  # 在原DataFrame上进行操作，删除所有包含NaN的行，并重置索引
        tm.assert_frame_equal(df, expected)  # 断言结果与期望相等
```