# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_partial_indexing.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,  # DataFrame 数据结构，用于处理二维数据
    IndexSlice,  # IndexSlice 用于多级索引的切片
    MultiIndex,  # MultiIndex 多级索引对象
    date_range,  # date_range 用于生成时间范围
)
import pandas._testing as tm  # 导入 pandas 内部的测试工具模块


@pytest.fixture  # 使用 pytest 的 fixture 装饰器，用于提供测试数据
def df():
    """
    #                        c1
    # 2016-01-01 00:00:00 a   0
    #                     b   1
    #                     c   2
    # 2016-01-01 12:00:00 a   3
    #                     b   4
    #                     c   5
    # 2016-01-02 00:00:00 a   6
    #                     b   7
    #                     c   8
    # 2016-01-02 12:00:00 a   9
    #                     b  10
    #                     c  11
    # 2016-01-03 00:00:00 a  12
    #                     b  13
    #                     c  14
    """
    dr = date_range("2016-01-01", "2016-01-03", freq="12h")  # 创建一个时间范围，每12小时一次
    abc = ["a", "b", "c"]  # 定义一个列表，包含字符串 'a', 'b', 'c'
    mi = MultiIndex.from_product([dr, abc])  # 创建一个多级索引对象，行为笛卡尔积结果
    frame = DataFrame({"c1": range(15)}, index=mi)  # 创建一个 DataFrame 对象，包含一列 'c1'，索引为 mi
    return frame  # 返回生成的 DataFrame 对象


def test_partial_string_matching_single_index(df):
    """
    # partial string matching on a single index
    """
    for df_swap in [df.swaplevel(), df.swaplevel(0), df.swaplevel(0, 1)]:  # 循环测试不同索引交换后的 DataFrame
        df_swap = df_swap.sort_index()  # 对交换索引后的 DataFrame 进行排序
        just_a = df_swap.loc["a"]  # 选取索引为 'a' 的部分
        result = just_a.loc["2016-01-01"]  # 选取日期为 '2016-01-01' 的部分
        expected = df.loc[IndexSlice[:, "a"], :].iloc[0:2]  # 从原始 DataFrame 中选取符合条件的预期部分
        expected.index = expected.index.droplevel(1)  # 删除第二级索引
        tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果是否符合预期


def test_get_loc_partial_timestamp_multiindex(df):
    """
    # GH10331
    """
    mi = df.index  # 获取 DataFrame 的索引
    key = ("2016-01-01", "a")  # 设置一个索引键
    loc = mi.get_loc(key)  # 获取索引键在索引中的位置

    expected = np.zeros(len(mi), dtype=bool)  # 创建一个布尔数组，长度为索引的长度
    expected[[0, 3]] = True  # 设置预期的位置为 True
    tm.assert_numpy_array_equal(loc, expected)  # 使用测试工具比较结果是否符合预期

    key2 = ("2016-01-02", "a")  # 设置第二个索引键
    loc2 = mi.get_loc(key2)  # 获取第二个索引键在索引中的位置
    expected2 = np.zeros(len(mi), dtype=bool)  # 创建一个布尔数组，长度为索引的长度
    expected2[[6, 9]] = True  # 设置预期的位置为 True
    tm.assert_numpy_array_equal(loc2, expected2)  # 使用测试工具比较结果是否符合预期

    key3 = ("2016-01", "a")  # 设置第三个索引键
    loc3 = mi.get_loc(key3)  # 获取第三个索引键在索引中的位置
    expected3 = np.zeros(len(mi), dtype=bool)  # 创建一个布尔数组，长度为索引的长度
    expected3[mi.get_level_values(1).get_loc("a")] = True  # 设置预期的位置为 True
    tm.assert_numpy_array_equal(loc3, expected3)  # 使用测试工具比较结果是否符合预期

    key4 = ("2016", "a")  # 设置第四个索引键
    loc4 = mi.get_loc(key4)  # 获取第四个索引键在索引中的位置
    expected4 = expected3  # 第四个预期结果与第三个相同
    tm.assert_numpy_array_equal(loc4, expected4)  # 使用测试工具比较结果是否符合预期

    # non-monotonic
    taker = np.arange(len(mi), dtype=np.intp)  # 创建一个索引数组
    taker[::2] = taker[::-2]  # 对索引数组进行非单调处理
    mi2 = mi.take(taker)  # 根据索引数组重新生成索引对象
    loc5 = mi2.get_loc(key)  # 获取索引键在新索引中的位置
    expected5 = np.zeros(len(mi2), dtype=bool)  # 创建一个布尔数组，长度为新索引的长度
    expected5[[3, 14]] = True  # 设置预期的位置为 True
    tm.assert_numpy_array_equal(loc5, expected5)  # 使用测试工具比较结果是否符合预期


def test_partial_string_timestamp_multiindex(df):
    """
    # GH10331
    # indexing with IndexSlice
    """
    df_swap = df.swaplevel(0, 1).sort_index()  # 交换索引级别并对索引排序
    SLC = IndexSlice  # 创建一个 IndexSlice 对象，用于切片

    # indexing with IndexSlice
    result = df.loc[SLC["2016-01-01":"2016-02-01", :], :]  # 使用 IndexSlice 进行切片操作
    expected = df  # 预期结果为整个 DataFrame
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果是否符合预期

    # match on secondary index
    result = df_swap.loc[SLC[:, "2016-01-01":"2016-01-01"], :]  # 使用 IndexSlice 进行切片操作
    expected = df_swap.iloc[[0, 1, 5, 6, 10, 11]]  # 根据位置选取预期的部分
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果是否符合预期

    # partial string match on year only
    result = df.loc["2016"]  # 选择特定年份的数据
    expected = df  # 预期结果为整个 DataFrame
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果是否符合预期
    # 针对日期的部分字符串匹配
    result = df.loc["2016-01-01"]
    # 期望结果是从第一个到第六个位置的数据
    expected = df.iloc[0:6]
    # 使用测试工具检查结果是否相等
    tm.assert_frame_equal(result, expected)

    # 针对日期和小时的部分字符串匹配，从中间开始
    result = df.loc["2016-01-02 12"]
    # 按小时解析，与索引的第一级相同，因此不在该级别上切片，该级别将被删除
    expected = df.iloc[9:12].droplevel(0)
    # 使用测试工具检查结果是否相等
    tm.assert_frame_equal(result, expected)

    # 针对次级索引的部分字符串匹配
    result = df_swap.loc[SLC[:, "2016-01-02"], :]
    expected = df_swap.iloc[[2, 3, 7, 8, 12, 13]]
    # 使用测试工具检查结果是否相等
    tm.assert_frame_equal(result, expected)

    # 使用日期部分字符串进行元组选择器
    # "2016-01-01" 具有每日分辨率，因此在第一级上是一个切片。
    result = df.loc[("2016-01-01", "a"), :]
    expected = df.iloc[[0, 3]]
    expected = df.iloc[[0, 3]].droplevel(1)
    # 使用测试工具检查结果是否相等
    tm.assert_frame_equal(result, expected)

    # 在第一级别上切片日期应该会失败，因为 DTI 是 df_swap 的第二级别
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df_swap.loc["2016-01-01"]
# 当前函数用于测试数据框（DataFrame）在多级索引情况下使用部分字符串时间戳的行为是否符合预期。

def test_partial_string_timestamp_multiindex_str_key_raises(df):
    # 即使这种语法在单索引上有效，但这在多级索引中可能会产生歧义，
    # 我们不希望将这种行为推广到多级索引上。这将导致从列中选择一个标量值。
    # 使用 pytest 来断言是否会抛出 KeyError 异常，并且异常信息中包含 "'2016-01-01'"
    with pytest.raises(KeyError, match="'2016-01-01'"):
        df["2016-01-01"]


def test_partial_string_timestamp_multiindex_daily_resolution(df):
    # 解决了 GH12685（部分字符串与每日或更低分辨率一起使用）
    # 从数据框中选择在指定时间范围内的数据，使用 IndexSlice 进行多级索引的切片
    result = df.loc[IndexSlice["2013-03":"2013-03", :], :]
    # 预期结果是选择数据框中从索引位置 118 到 180 的行
    expected = df.iloc[118:180]
    # 使用测试工具 tm.assert_frame_equal 检查结果和预期是否相等
    tm.assert_frame_equal(result, expected)
```