# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_reset_index.py`

```
from datetime import datetime  # 导入 datetime 模块中的 datetime 类

import numpy as np  # 导入 numpy 库，并使用 np 别名
import pytest  # 导入 pytest 库

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import (  # 从 pandas 库中导入多个类和函数
    DataFrame,  # 数据帧类
    Index,  # 索引类
    MultiIndex,  # 多重索引类
    RangeIndex,  # 范围索引类
    Series,  # 系列类
    date_range,  # 日期范围函数
    option_context,  # 选项上下文函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestResetIndex:  # 定义测试类 TestResetIndex
    def test_reset_index_dti_round_trip(self):  # 定义测试方法 test_reset_index_dti_round_trip
        dti = date_range(start="1/1/2001", end="6/1/2001", freq="D")._with_freq(None)  # 创建日期范围对象 dti
        d1 = DataFrame({"v": np.random.default_rng(2).random(len(dti))}, index=dti)  # 创建数据帧 d1
        d2 = d1.reset_index()  # 重置 d1 的索引，并赋值给 d2
        assert d2.dtypes.iloc[0] == np.dtype("M8[ns]")  # 断言 d2 的第一个列的数据类型为日期时间类型
        d3 = d2.set_index("index")  # 将 d2 的 "index" 列设置为新的索引，赋值给 d3
        tm.assert_frame_equal(d1, d3, check_names=False)  # 使用测试模块中的函数比较 d1 和 d3，忽略索引名称的检查

        # GH#2329
        stamp = datetime(2012, 11, 22)  # 创建一个 datetime 对象 stamp
        df = DataFrame([[stamp, 12.1]], columns=["Date", "Value"])  # 创建数据帧 df
        df = df.set_index("Date")  # 将 df 的 "Date" 列设置为索引

        assert df.index[0] == stamp  # 断言 df 的索引的第一个元素与 stamp 相等
        assert df.reset_index()["Date"].iloc[0] == stamp  # 断言重置 df 索引后的第一个 "Date" 列元素与 stamp 相等

    def test_reset_index(self):  # 定义测试方法 test_reset_index
        df = DataFrame(  # 创建数据帧 df
            1.1 * np.arange(120).reshape((30, 4)),  # 生成一个数组并进行 reshape 后乘以 1.1
            columns=Index(list("ABCD"), dtype=object),  # 设置列索引为 A, B, C, D
            index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 设置行索引为 i-0 到 i-29
        )[:5]  # 取前 5 行
        ser = df.stack()  # 对 df 进行堆叠操作，得到一个系列 ser
        ser.index.names = ["hash", "category"]  # 设置 ser 的索引名称为 "hash" 和 "category"

        ser.name = "value"  # 设置 ser 的名称为 "value"
        df = ser.reset_index()  # 重置 ser 的索引，赋值给 df
        assert "value" in df  # 断言 "value" 列存在于 df 中

        df = ser.reset_index(name="value2")  # 重置 ser 的索引，并将列名命名为 "value2"，赋值给 df
        assert "value2" in df  # 断言 "value2" 列存在于 df 中

        # check inplace
        s = ser.reset_index(drop=True)  # 重置 ser 的索引，不保留原索引，赋值给 s
        s2 = ser  # 将 ser 赋值给 s2
        return_value = s2.reset_index(drop=True, inplace=True)  # 在原地重置 s2 的索引，不保留原索引，返回值赋给 return_value
        assert return_value is None  # 断言 return_value 为 None
        tm.assert_series_equal(s, s2)  # 使用测试模块中的函数比较 s 和 s2

        # level
        index = MultiIndex(  # 创建多重索引对象 index
            levels=[["bar"], ["one", "two", "three"], [0, 1]],  # 设置多重索引的三个级别的值
            codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],  # 设置多重索引的编码
        )
        s = Series(np.random.default_rng(2).standard_normal(6), index=index)  # 创建系列 s
        rs = s.reset_index(level=1)  # 重置 s 的第二级别索引，赋值给 rs
        assert len(rs.columns) == 2  # 断言 rs 的列数为 2

        rs = s.reset_index(level=[0, 2], drop=True)  # 重置 s 的第一和第三级别索引，不保留原索引，赋值给 rs
        tm.assert_index_equal(rs.index, Index(index.get_level_values(1)))  # 使用测试模块中的函数比较 rs 的索引和 index 的第二级别值的索引
        assert isinstance(rs, Series)  # 断言 rs 是 Series 类型的对象

    def test_reset_index_name(self):  # 定义测试方法 test_reset_index_name
        s = Series([1, 2, 3], index=Index(range(3), name="x"))  # 创建具有命名索引的系列 s
        assert s.reset_index().index.name is None  # 断言重置 s 索引后的索引名为 None
        assert s.reset_index(drop=True).index.name is None  # 断言在不保留原索引的情况下重置 s 索引后的索引名为 None
    # 定义测试方法，测试 DataFrame 的 reset_index 方法在不同场景下的表现
    def test_reset_index_level(self):
        # 创建一个 DataFrame 包含两行三列的数据
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])

        # 遍历两组不同的 levels 参数进行测试
        for levels in ["A", "B"], [0, 1]:
            # 使用 MultiIndex 设置 DataFrame 索引，并选择其中的列 "C"
            s = df.set_index(["A", "B"])["C"]

            # 测试 reset_index 方法，根据指定的 level 重置索引
            result = s.reset_index(level=levels[0])
            tm.assert_frame_equal(result, df.set_index("B"))

            # 再次测试 reset_index 方法，传入一个包含单个 level 的列表
            result = s.reset_index(level=levels[:1])
            tm.assert_frame_equal(result, df.set_index("B"))

            # 测试 reset_index 方法，使用包含多个 levels 的列表
            result = s.reset_index(level=levels)
            tm.assert_frame_equal(result, df)

            # 测试 set_index 后再 reset_index 方法，带有 drop=True 参数
            result = df.set_index(["A", "B"]).reset_index(level=levels, drop=True)
            tm.assert_frame_equal(result, df[["C"]])

            # 测试 reset_index 方法时期望引发 KeyError 异常
            with pytest.raises(KeyError, match="Level E "):
                s.reset_index(level=["A", "E"])

            # 使用单层索引设置 DataFrame，并选择其中的列 "B"
            s = df.set_index("A")["B"]

            # 测试 reset_index 方法，根据指定的 level 重置索引
            result = s.reset_index(level=levels[0])
            tm.assert_frame_equal(result, df[["A", "B"]])

            # 再次测试 reset_index 方法，传入一个包含单个 level 的列表
            result = s.reset_index(level=levels[:1])
            tm.assert_frame_equal(result, df[["A", "B"]])

            # 测试 reset_index 方法，带有 drop=True 参数
            result = s.reset_index(level=levels[0], drop=True)
            tm.assert_series_equal(result, df["B"])

            # 测试 reset_index 方法时期望引发 IndexError 异常
            with pytest.raises(IndexError, match="Too many levels"):
                s.reset_index(level=[0, 1, 2])

        # 测试 .reset_index([],drop=True) 方法，不应该引发异常
        result = Series(range(4)).reset_index([], drop=True)
        expected = Series(range(4))
        tm.assert_series_equal(result, expected)

    # 定义测试方法，测试 Series 的 reset_index 方法在特定情况下的表现
    def test_reset_index_range(self):
        # 创建一个 Series 包含两个整数的范围，设置名称和数据类型
        s = Series(range(2), name="A", dtype="int64")
        # 使用 reset_index 方法重置 Series 的索引
        series_result = s.reset_index()
        # 断言返回的结果索引类型为 RangeIndex
        assert isinstance(series_result.index, RangeIndex)
        # 创建期望的 DataFrame 结果
        series_expected = DataFrame(
            [[0, 0], [1, 1]], columns=["index", "A"], index=RangeIndex(stop=2)
        )
        tm.assert_frame_equal(series_result, series_expected)

    # 定义测试方法，测试 Series 的 reset_index 方法在处理错误时的表现
    def test_reset_index_drop_errors(self):
        #  GH 20925

        # 当指定的 level 名称不存在时，期望引发 KeyError 异常
        s = Series(range(4))
        with pytest.raises(KeyError, match="does not match index name"):
            s.reset_index("wrong", drop=True)
        with pytest.raises(KeyError, match="does not match index name"):
            s.reset_index("wrong")

        # 当指定的 level 不存在时，期望引发 KeyError 异常
        s = Series(range(4), index=MultiIndex.from_product([[1, 2]] * 2))
        with pytest.raises(KeyError, match="not found"):
            s.reset_index("wrong", drop=True)
    def test_reset_index_with_drop(self):
        # 创建一个包含多个子列表的列表，每个子列表表示索引的不同级别的值
        arrays = [
            ["bar", "bar", "baz", "baz", "qux", "qux", "foo", "foo"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        # 使用 zip 函数将数组转换为元组的列表
        tuples = zip(*arrays)
        # 从元组列表创建一个 MultiIndex 对象作为索引
        index = MultiIndex.from_tuples(tuples)
        # 生成一个包含随机标准正态分布数据的 Series 对象
        data = np.random.default_rng(2).standard_normal(8)
        ser = Series(data, index=index)
        # 将序列中的第三个元素设置为 NaN
        ser.iloc[3] = np.nan

        # 对序列进行 reset_index 操作，返回一个新的 DataFrame 对象
        deleveled = ser.reset_index()
        # 断言返回结果的类型为 DataFrame
        assert isinstance(deleveled, DataFrame)
        # 断言返回结果的列数等于原始序列索引级别数加一
        assert len(deleveled.columns) == len(ser.index.levels) + 1
        # 断言返回结果的索引名称与原始序列的索引名称相同
        assert deleveled.index.name == ser.index.name

        # 对序列进行带有 drop=True 参数的 inplace reset_index 操作
        deleveled = ser.reset_index(drop=True)
        # 断言返回结果的类型为 Series
        assert isinstance(deleveled, Series)
        # 断言返回结果的索引名称与原始序列的索引名称相同
        assert deleveled.index.name == ser.index.name

    def test_reset_index_inplace_and_drop_ignore_name(self):
        # GH#44575
        # 创建一个名为 "old" 的 Series 对象，其值为 [0, 1]
        ser = Series(range(2), name="old")
        # 对该 Series 进行 inplace reset_index 操作，丢弃原有索引，设置新的列名为 "new"
        ser.reset_index(name="new", drop=True, inplace=True)
        # 创建一个期望的 Series 对象，其值也为 [0, 1]，名称为 "old"
        expected = Series(range(2), name="old")
        # 使用断言函数检查两个 Series 对象是否相等
        tm.assert_series_equal(ser, expected)

    def test_reset_index_drop_infer_string(self):
        # GH#56160
        # 导入 pytest，并跳过如果没有安装 pyarrow 模块
        pytest.importorskip("pyarrow")
        # 创建一个包含字符串的 Series 对象，数据类型为 object
        ser = Series(["a", "b", "c"], dtype=object)
        # 在设置 future.infer_string 选项为 True 的上下文中，进行 reset_index 操作，丢弃原有索引
        result = ser.reset_index(drop=True)
        # 使用断言函数检查返回的结果是否与原始 Series 相等
        tm.assert_series_equal(result, ser)
# 使用 pytest.mark.parametrize 装饰器定义测试参数化，测试函数接收 array 和 dtype 作为参数
@pytest.mark.parametrize(
    "array, dtype",
    [
        (["a", "b"], object),  # 参数示例1：array 是包含字符串 'a', 'b' 的列表，dtype 是 object 类型
        (
            pd.period_range("12-1-2000", periods=2, freq="Q-DEC"),  # 参数示例2：array 是日期范围，dtype 是 PeriodDtype 类型
            pd.PeriodDtype(freq="Q-DEC"),
        ),
    ],
)
# 定义测试函数 test_reset_index_dtypes_on_empty_series_with_multiindex，接收 array, dtype 和 using_infer_string 作为参数
def test_reset_index_dtypes_on_empty_series_with_multiindex(
    array, dtype, using_infer_string
):
    # GH 19602 - 保持在空 Series 中的 MultiIndex 上的 dtype
    # 创建 MultiIndex 对象 idx，包含多层索引，其中包括 0, 1，0.5, 1.0 和 array 中的值
    idx = MultiIndex.from_product([[0, 1], [0.5, 1.0], array])
    # 创建空 Series 对象，指定 dtype 为 object，并重置索引，获取重置后的索引的数据类型
    result = Series(dtype=object, index=idx)[:0].reset_index().dtypes
    # 根据 using_infer_string 的值确定期望的类型 exp
    exp = "string" if using_infer_string else object
    # 创建期望的 Series 对象 expected，包含特定的列名和对应的数据类型
    expected = Series(
        {
            "level_0": np.int64,
            "level_1": np.float64,
            "level_2": exp if dtype == object else dtype,
            0: object,
        }
    )
    # 使用 pytest 的 tm.assert_series_equal 函数比较 result 和 expected
    tm.assert_series_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器定义测试参数化，测试函数接收 names 和 expected_names 作为参数
@pytest.mark.parametrize(
    "names, expected_names",
    [
        (["A", "A"], ["A", "A"]),  # 参数示例1：names 和 expected_names 都是包含字符串 'A', 'A' 的列表
        (["level_1", None], ["level_1", "level_1"]),  # 参数示例2：names 中包含 'level_1' 和 None，expected_names 为 ['level_1', 'level_1']
    ],
)
# 使用 pytest.mark.parametrize 装饰器定义测试参数化，测试函数接收 allow_duplicates 作为参数
@pytest.mark.parametrize("allow_duplicates", [False, True])
# 定义测试函数 test_column_name_duplicates，接收 names, expected_names 和 allow_duplicates 作为参数
def test_column_name_duplicates(names, expected_names, allow_duplicates):
    # GH#44755 reset_index 处理具有重复列标签的情况
    # 创建带有 MultiIndex 的 Series 对象 s，其中索引为 names 指定的多层结构，值为 [1]
    s = Series([1], index=MultiIndex.from_arrays([[1], [1]], names=names))
    # 如果 allow_duplicates 为 True，则执行重置索引并比较结果与期望的 DataFrame
    if allow_duplicates:
        result = s.reset_index(allow_duplicates=True)
        expected = DataFrame([[1, 1, 1]], columns=expected_names + [0])
        tm.assert_frame_equal(result, expected)
    else:
        # 如果 allow_duplicates 为 False，则确保在重置索引时抛出 ValueError 异常
        with pytest.raises(ValueError, match="cannot insert"):
            s.reset_index()
```