# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_scalar.py`

```
# 导入所需的模块和库

"""test scalar indexing, including at and iat"""
# 测试标量索引，包括 at 和 iat 方法

from datetime import (
    datetime,
    timedelta,
)
# 导入 datetime 和 timedelta 类

import itertools
# 导入 itertools 模块，用于迭代工具

import numpy as np
# 导入 NumPy 库，并使用别名 np

import pytest
# 导入 pytest 测试框架

from pandas import (
    DataFrame,
    Index,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
# 从 pandas 库中导入 DataFrame、Index、Series、Timedelta、Timestamp 和 date_range

import pandas._testing as tm
# 导入 pandas 内部测试模块 tm

def generate_indices(f, values=False):
    """
    generate the indices
    if values is True , use the axis values
    is False, use the range
    """
    # 生成索引的函数
    axes = f.axes
    # 获取数据结构 f 的轴

    if values:
        axes = (list(range(len(ax))) for ax in axes)
        # 如果 values 为 True，使用轴的值

    return itertools.product(*axes)
    # 返回所有轴的笛卡尔积

class TestScalar:
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    # 使用 pytest.mark.parametrize 装饰器，参数化测试用例的数据类型

    def test_iat_set_ints(self, dtype, frame_or_series):
        # 测试 iat 方法设置整数
        f = frame_or_series(range(3), index=Index([0, 1, 2], dtype=dtype))
        # 创建一个由 frame_or_series 生成的对象 f，使用指定的整数索引
        indices = generate_indices(f, True)
        # 生成索引集合

        for i in indices:
            f.iat[i] = 1
            # 使用 iat 方法设置元素值为 1
            expected = f.values[i]
            # 获取预期的值为 f 中索引 i 对应的值
            tm.assert_almost_equal(expected, 1)
            # 使用 tm.assert_almost_equal 断言预期值与 1 几乎相等

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcd"), dtype=object),
            date_range("20130101", periods=4),
            Index(range(0, 8, 2), dtype=np.float64),
        ],
    )
    # 参数化测试用例的索引

    def test_iat_set_other(self, index, frame_or_series):
        # 测试 iat 方法设置其他类型的索引
        f = frame_or_series(range(len(index)), index=index)
        # 创建一个由 frame_or_series 生成的对象 f，使用指定的索引
        msg = "iAt based indexing can only have integer indexers"
        # 提示消息

        idx = next(generate_indices(f, False))
        # 获取生成索引集合中的下一个索引

        with pytest.raises(ValueError, match=msg):
            f.iat[idx] = 1
            # 使用 iat 方法设置元素值为 1，预期会引发 ValueError 异常

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcd"), dtype=object),
            date_range("20130101", periods=4),
            Index(range(0, 8, 2), dtype=np.float64),
            Index(range(0, 8, 2), dtype=np.uint64),
            Index(range(0, 8, 2), dtype=np.int64),
        ],
    )
    # 参数化测试用例的索引

    def test_at_set_ints_other(self, index, frame_or_series):
        # 测试 at 方法设置整数和其他类型的索引
        f = frame_or_series(range(len(index)), index=index)
        # 创建一个由 frame_or_series 生成的对象 f，使用指定的索引
        indices = generate_indices(f, False)
        # 生成索引集合

        for i in indices:
            f.at[i] = 1
            # 使用 at 方法设置元素值为 1
            expected = f.loc[i]
            # 获取预期的值为 f 中索引 i 对应的值
            tm.assert_almost_equal(expected, 1)
            # 使用 tm.assert_almost_equal 断言预期值与 1 几乎相等

class TestAtAndiAT:
    # 测试 at 和 iat 方法的类

    def test_float_index_at_iat(self):
        # 测试浮点数索引的 at 和 iat 方法
        ser = Series([1, 2, 3], index=[0.1, 0.2, 0.3])
        # 创建一个 Series 对象 ser，使用指定的浮点数索引

        for el, item in ser.items():
            assert ser.at[el] == item
            # 断言 ser 的 at 方法根据索引 el 返回的值等于 item

        for i in range(len(ser)):
            assert ser.iat[i] == i + 1
            # 断言 ser 的 iat 方法根据索引 i 返回的值等于 i + 1

    def test_at_iat_coercion(self):
        # 测试 at 和 iat 方法的类型强制转换
        dates = date_range("1/1/2000", periods=8)
        # 创建一个日期范围 dates

        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 4)),
            index=dates,
            columns=["A", "B", "C", "D"],
        )
        # 创建一个 DataFrame 对象 df，使用指定的随机标准正态分布值和日期索引

        s = df["A"]
        # 获取 df 的 A 列 Series 对象 s

        result = s.at[dates[5]]
        # 使用 s 的 at 方法根据日期 dates[5] 返回的值
        xp = s.values[5]
        # 获取 s 中索引 5 对应的值

        assert result == xp
        # 断言结果等于预期值 xp
    @pytest.mark.parametrize(
        "ser, expected",
        [  # 参数化测试的参数列表
            [
                Series(["2014-01-01", "2014-02-02"], dtype="datetime64[ns]"),  # 创建包含日期时间的 Series 对象
                Timestamp("2014-02-02"),  # 期望的时间戳对象
            ],
            [
                Series(["1 days", "2 days"], dtype="timedelta64[ns]"),  # 创建包含时间间隔的 Series 对象
                Timedelta("2 days"),  # 期望的时间间隔对象
            ],
        ],
    )
    def test_iloc_iat_coercion_datelike(self, indexer_ial, ser, expected):
        # GH 7729
        # 确保返回值被封装
        result = indexer_ial(ser)[1]  # 使用给定的 indexer_ial 函数处理 ser，并获取第二个返回值
        assert result == expected  # 断言处理结果与期望值相等

    def test_imethods_with_dups(self):
        # GH6493
        # 处理带有重复索引的 iat/iloc 操作

        s = Series(range(5), index=[1, 1, 2, 2, 3], dtype="int64")  # 创建带有重复索引的 Series 对象
        result = s.iloc[2]  # 使用位置索引 iloc 获取第三个元素
        assert result == 2  # 断言结果为 2
        result = s.iat[2]  # 使用快速访问方法 iat 获取第三个元素
        assert result == 2  # 断言结果为 2

        # 测试超出索引范围的异常情况
        msg = "index 10 is out of bounds for axis 0 with size 5"
        with pytest.raises(IndexError, match=msg):  # 断言捕获到 IndexError 异常，并匹配特定的错误消息
            s.iat[10]
        msg = "index -10 is out of bounds for axis 0 with size 5"
        with pytest.raises(IndexError, match=msg):  # 断言捕获到 IndexError 异常，并匹配特定的错误消息
            s.iat[-10]

        result = s.iloc[[2, 3]]  # 使用 iloc 获取多个元素
        expected = Series([2, 3], index=[2, 2], dtype="int64")  # 期望的结果 Series 对象
        tm.assert_series_equal(result, expected)  # 断言两个 Series 对象相等

        df = s.to_frame()  # 将 Series 转换为 DataFrame
        result = df.iloc[2]  # 使用 iloc 获取 DataFrame 的第三行
        expected = Series(2, index=[0], name=2)  # 期望的结果 Series 对象
        tm.assert_series_equal(result, expected)  # 断言两个 Series 对象相等

        result = df.iat[2, 0]  # 使用 iat 快速访问 DataFrame 的第三行第一列
        assert result == 2  # 断言结果为 2

    def test_frame_at_with_duplicate_axes(self):
        # GH#33041
        arr = np.random.default_rng(2).standard_normal(6).reshape(3, 2)  # 创建一个随机数组
        df = DataFrame(arr, columns=["A", "A"])  # 创建带有重复列名的 DataFrame 对象

        result = df.at[0, "A"]  # 使用标签访问方法 at 获取指定位置的值
        expected = df.iloc[0].copy()  # 复制 DataFrame 的第一行作为期望结果

        tm.assert_series_equal(result, expected)  # 断言两个 Series 对象相等

        result = df.T.at["A", 0]  # 转置后使用 at 方法获取指定位置的值
        tm.assert_series_equal(result, expected)  # 断言两个 Series 对象相等

        # 设置器操作
        df.at[1, "A"] = 2  # 使用 at 方法设置指定位置的值
        expected = Series([2.0, 2.0], index=["A", "A"], name=1)  # 期望的结果 Series 对象
        tm.assert_series_equal(df.iloc[1], expected)  # 断言两个 Series 对象相等

    def test_at_getitem_dt64tz_values(self):
        # gh-15822
        df = DataFrame(
            {
                "name": ["John", "Anderson"],
                "date": [
                    Timestamp(2017, 3, 13, 13, 32, 56),  # 创建 Timestamp 对象
                    Timestamp(2017, 2, 16, 12, 10, 3),  # 创建 Timestamp 对象
                ],
            }
        )
        df["date"] = df["date"].dt.tz_localize("Asia/Shanghai")  # 为日期列添加时区信息

        expected = Timestamp("2017-03-13 13:32:56+0800", tz="Asia/Shanghai")  # 期望的结果 Timestamp 对象

        result = df.loc[0, "date"]  # 使用 loc 方法获取指定位置的值
        assert result == expected  # 断言结果与期望值相等

        result = df.at[0, "date"]  # 使用 at 方法获取指定位置的值
        assert result == expected  # 断言结果与期望值相等
    # 定义测试函数：混合索引下的 at、iat、loc、iloc 操作对 Series 的测试
    def test_mixed_index_at_iat_loc_iloc_series(self):
        # GH 19860
        # 创建一个 Series 对象，包含混合索引
        s = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        
        # 遍历 Series 的每个元素和对应的值
        for el, item in s.items():
            # 断言 at、loc 方法获取的值与当前值相等
            assert s.at[el] == s.loc[el] == item
        
        # 遍历 Series 的索引范围
        for i in range(len(s)):
            # 断言 iat、iloc 方法获取的值与当前索引值加一相等
            assert s.iat[i] == s.iloc[i] == i + 1
        
        # 使用 pytest 断言，验证索引不存在时使用 at 方法抛出 KeyError 异常
        with pytest.raises(KeyError, match="^4$"):
            s.at[4]
        
        # 使用 pytest 断言，验证索引不存在时使用 loc 方法抛出 KeyError 异常
        with pytest.raises(KeyError, match="^4$"):
            s.loc[4]

    # 定义测试函数：混合索引下的 at、iat、loc、iloc 操作对 DataFrame 的测试
    def test_mixed_index_at_iat_loc_iloc_dataframe(self):
        # GH 19860
        # 创建一个 DataFrame 对象，包含混合索引
        df = DataFrame(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], columns=["a", "b", "c", 1, 2]
        )
        
        # 遍历 DataFrame 的每行索引和每个元素
        for rowIdx, row in df.iterrows():
            # 遍历每行的每个元素和对应的值
            for el, item in row.items():
                # 断言 at、loc 方法获取的值与当前值相等
                assert df.at[rowIdx, el] == df.loc[rowIdx, el] == item
        
        # 遍历 DataFrame 的每行和每列索引范围
        for row in range(2):
            for i in range(5):
                # 断言 iat、iloc 方法获取的值与当前值相等
                assert df.iat[row, i] == df.iloc[row, i] == row * 5 + i
        
        # 使用 pytest 断言，验证索引不存在时使用 at 方法抛出 KeyError 异常
        with pytest.raises(KeyError, match="^3$"):
            df.at[0, 3]
        
        # 使用 pytest 断言，验证索引不存在时使用 loc 方法抛出 KeyError 异常
        with pytest.raises(KeyError, match="^3$"):
            df.loc[0, 3]

    # 定义测试函数：测试 DataFrame 中使用 iat 方法进行不兼容的赋值操作
    def test_iat_setter_incompatible_assignment(self):
        # GH 23236
        # 创建一个 DataFrame 对象
        result = DataFrame({"a": [0.0, 1.0], "b": [4, 5]})
        
        # 使用 iat 方法进行赋值操作
        result.iat[0, 0] = None
        
        # 创建预期结果的 DataFrame 对象
        expected = DataFrame({"a": [None, 1], "b": [4, 5]})
        
        # 使用 pandas 测试工具（tm）进行 DataFrame 对象的内容比较
        tm.assert_frame_equal(result, expected)
def test_iat_dont_wrap_object_datetimelike():
    # GH#32809 .iat calls go through DataFrame._get_value, should not
    #  call maybe_box_datetimelike

    # 创建一个日期范围对象，从"2016-01-01"开始，持续3个周期
    dti = date_range("2016-01-01", periods=3)
    # 计算日期范围对象的时间差
    tdi = dti - dti
    # 创建一个包含日期时间对象的Series，数据类型为object
    ser = Series(dti.to_pydatetime(), dtype=object)
    # 创建一个包含时间差对象的Series，数据类型为object
    ser2 = Series(tdi.to_pytimedelta(), dtype=object)
    # 创建一个DataFrame，列"A"包含ser的日期时间对象，列"B"包含ser2的时间差对象
    df = DataFrame({"A": ser, "B": ser2})
    # 断言DataFrame的所有列数据类型为object
    assert (df.dtypes == object).all()

    # 针对四种索引方式分别进行断言
    for result in [df.at[0, "A"], df.iat[0, 0], df.loc[0, "A"], df.iloc[0, 0]]:
        # 断言结果与ser的第一个元素相等
        assert result is ser[0]
        # 断言结果是datetime类型
        assert isinstance(result, datetime)
        # 断言结果不是Timestamp类型
        assert not isinstance(result, Timestamp)

    # 针对四种索引方式分别进行断言
    for result in [df.at[1, "B"], df.iat[1, 1], df.loc[1, "B"], df.iloc[1, 1]]:
        # 断言结果与ser2的第二个元素相等
        assert result is ser2[1]
        # 断言结果是timedelta类型
        assert isinstance(result, timedelta)
        # 断言结果不是Timedelta类型
        assert not isinstance(result, Timedelta)


def test_at_with_tuple_index_get():
    # GH 26989
    # DataFrame.at getter works with Index of tuples

    # 创建一个DataFrame，列"a"包含两个元素1和2，索引为[(1, 2), (3, 4)]
    df = DataFrame({"a": [1, 2]}, index=[(1, 2), (3, 4)])
    # 断言DataFrame的索引级别为1
    assert df.index.nlevels == 1
    # 使用DataFrame.at获取索引为(1, 2)的"a"列值，断言为1
    assert df.at[(1, 2), "a"] == 1

    # 创建一个Series，从DataFrame的"a"列中取出索引为(1, 2)的值
    series = df["a"]
    # 断言Series的索引级别为1
    assert series.index.nlevels == 1
    # 使用Series.at获取索引为(1, 2)的值，断言为1
    assert series.at[(1, 2)] == 1


@pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
def test_at_with_tuple_index_set():
    # GH 26989
    # DataFrame.at setter works with Index of tuples

    # 创建一个DataFrame，列"a"包含两个元素1和2，索引为[(1, 2), (3, 4)]
    df = DataFrame({"a": [1, 2]}, index=[(1, 2), (3, 4)])
    # 断言DataFrame的索引级别为1
    assert df.index.nlevels == 1
    # 使用DataFrame.at设置索引为(1, 2)的"a"列值为2
    df.at[(1, 2), "a"] = 2
    # 使用DataFrame.at获取索引为(1, 2)的"a"列值，断言为2
    assert df.at[(1, 2), "a"] == 2

    # 创建一个Series，从DataFrame的"a"列中取出
    series = df["a"]
    # 断言Series的索引级别为1
    assert series.index.nlevels == 1
    # 使用Series.at设置索引为(1, 2)的值为3
    series.at[1, 2] = 3
    # 使用Series.at获取索引为(1, 2)的值，断言为3
    assert series.at[1, 2] == 3


class TestMultiIndexScalar:
    def test_multiindex_at_get(self):
        # GH 26989
        # DataFrame.at and DataFrame.loc getter works with MultiIndex

        # 创建一个DataFrame，列"a"包含两个元素1和2，多级索引为[[1, 2], [3, 4]]
        df = DataFrame({"a": [1, 2]}, index=[[1, 2], [3, 4]])
        # 断言DataFrame的索引级别为2
        assert df.index.nlevels == 2
        # 使用DataFrame.at获取多级索引为(1, 3)的"a"列值，断言为1
        assert df.at[(1, 3), "a"] == 1
        # 使用DataFrame.loc获取多级索引为(1, 3)的"a"列值，断言为1
        assert df.loc[(1, 3), "a"] == 1

        # 创建一个Series，从DataFrame的"a"列中取出
        series = df["a"]
        # 断言Series的索引级别为2
        assert series.index.nlevels == 2
        # 使用Series.at获取多级索引为(1, 3)的值，断言为1
        assert series.at[1, 3] == 1
        # 使用Series.loc获取多级索引为(1, 3)的值，断言为1
        assert series.loc[1, 3] == 1

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_multiindex_at_set(self):
        # GH 26989
        # DataFrame.at and DataFrame.loc setter works with MultiIndex

        # 创建一个DataFrame，列"a"包含两个元素1和2，多级索引为[[1, 2], [3, 4]]
        df = DataFrame({"a": [1, 2]}, index=[[1, 2], [3, 4]])
        # 断言DataFrame的索引级别为2
        assert df.index.nlevels == 2
        # 使用DataFrame.at设置多级索引为(1, 3)的"a"列值为3
        df.at[(1, 3), "a"] = 3
        # 使用DataFrame.at获取多级索引为(1, 3)的"a"列值，断言为3
        assert df.at[(1, 3), "a"] == 3
        # 使用DataFrame.loc设置多级索引为(1, 3)的"a"列值为4
        df.loc[(1, 3), "a"] = 4
        # 使用DataFrame.loc获取多级索引为(1, 3)的"a"列值，断言为4
        assert df.loc[(1, 3), "a"] == 4

        # 创建一个Series，从DataFrame的"a"列中取出
        series = df["a"]
        # 断言Series的索引级别为2
        assert series.index.nlevels == 2
        # 使用Series.at设置多级索引为(1, 3)的值为5
        series.at[1, 3] = 5
        # 使用Series.at获取多级索引为(1, 3)的值，断言为5
        assert series.at[1, 3] == 5
        # 使用Series.loc设置多级索引为(1, 3)的值为6
        series.loc[1, 3] = 6
        # 使用Series.loc获取多级索引为(1, 3)的值，断言为6
        assert series.loc[1, 3] == 6
    def test_multiindex_at_get_one_level(self):
        # 定义一个测试方法，用于测试多级索引的 `at` 方法获取单层索引的情况
        # GH#38053 是 GitHub 上的 issue 编号，指明了这个测试的相关背景或问题
        # 创建一个 Series 对象 `s2`，其元素为 (0, 1)，索引为一个二级列表 [[False, True]]
        s2 = Series((0, 1), index=[[False, True]])
        # 使用 `at` 方法获取索引为 `False` 的元素，期望结果是 0
        result = s2.at[False]
        # 断言获取的结果是否与期望值相等
        assert result == 0
```