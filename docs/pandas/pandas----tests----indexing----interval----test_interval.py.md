# `D:\src\scipysrc\pandas\pandas\tests\indexing\interval\test_interval.py`

```
import numpy as np  # 导入NumPy库，用于处理数值数据
import pytest  # 导入pytest库，用于编写和运行测试

from pandas._libs import index as libindex  # 导入pandas的内部索引模块
from pandas.compat import IS64  # 导入pandas的兼容性模块，用于判断是否为64位系统

import pandas as pd  # 导入pandas库，用于数据处理和分析
from pandas import (  # 从pandas库中导入以下对象
    DataFrame,  # 数据框对象
    IntervalIndex,  # 区间索引对象
    Series,  # 系列对象
)
import pandas._testing as tm  # 导入pandas的测试模块，用于编写测试函数

class TestIntervalIndex:
    @pytest.fixture  # 使用pytest的装饰器定义测试的fixture
    def series_with_interval_index(self):  # 定义生成具有区间索引的系列的fixture
        return Series(np.arange(5), IntervalIndex.from_breaks(np.arange(6)))  # 返回一个具有区间索引的系列对象

    def test_getitem_with_scalar(self, series_with_interval_index, indexer_sl):  # 测试用例：使用标量进行索引
        ser = series_with_interval_index.copy()  # 复制区间索引系列对象

        expected = ser.iloc[:3]  # 预期结果为前三个元素
        tm.assert_series_equal(expected, indexer_sl(ser)[:3])  # 使用测试模块的函数比较结果
        tm.assert_series_equal(expected, indexer_sl(ser)[:2.5])  # 使用测试模块的函数比较结果
        tm.assert_series_equal(expected, indexer_sl(ser)[0.1:2.5])  # 使用测试模块的函数比较结果
        if indexer_sl is tm.loc:
            tm.assert_series_equal(expected, ser.loc[-1:3])  # 使用测试模块的函数比较结果

        expected = ser.iloc[1:4]  # 预期结果为第1到第3个元素
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 2.5, 3.5]])  # 使用测试模块的函数比较结果
        tm.assert_series_equal(expected, indexer_sl(ser)[[2, 3, 4]])  # 使用测试模块的函数比较结果
        tm.assert_series_equal(expected, indexer_sl(ser)[[1.5, 3, 4]])  # 使用测试模块的函数比较结果

        expected = ser.iloc[2:5]  # 预期结果为第2到第4个元素
        tm.assert_series_equal(expected, indexer_sl(ser)[ser >= 2])  # 使用测试模块的函数比较结果

    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])  # 参数化测试：测试不同的方向
    def test_getitem_nonoverlapping_monotonic(self, direction, closed, indexer_sl):
        tpls = [(0, 1), (2, 3), (4, 5)]  # 定义区间的元组列表
        if direction == "decreasing":  # 如果方向是递减的，反转元组列表
            tpls = tpls[::-1]

        idx = IntervalIndex.from_tuples(tpls, closed=closed)  # 创建区间索引对象
        ser = Series(list("abc"), idx)  # 创建系列对象，使用区间索引

        for key, expected in zip(idx.left, ser):  # 遍历左端点和系列的键值对
            if idx.closed_left:
                assert indexer_sl(ser)[key] == expected  # 使用测试模块的函数进行断言
            else:
                with pytest.raises(KeyError, match=str(key)):  # 断言抛出KeyError异常
                    indexer_sl(ser)[key]

        for key, expected in zip(idx.right, ser):  # 遍历右端点和系列的键值对
            if idx.closed_right:
                assert indexer_sl(ser)[key] == expected  # 使用测试模块的函数进行断言
            else:
                with pytest.raises(KeyError, match=str(key)):  # 断言抛出KeyError异常
                    indexer_sl(ser)[key]

        for key, expected in zip(idx.mid, ser):  # 遍历中点和系列的键值对
            assert indexer_sl(ser)[key] == expected  # 使用测试模块的函数进行断言

    def test_getitem_non_matching(self, series_with_interval_index, indexer_sl):
        ser = series_with_interval_index.copy()  # 复制区间索引系列对象

        # this is a departure from our current
        # indexing scheme, but simpler
        with pytest.raises(KeyError, match=r"\[-1\] not in index"):  # 断言抛出KeyError异常，匹配指定的正则表达式
            indexer_sl(ser)[[-1, 3, 4, 5]]  # 使用测试模块的函数进行索引操作

        with pytest.raises(KeyError, match=r"\[-1\] not in index"):  # 断言抛出KeyError异常，匹配指定的正则表达式
            indexer_sl(ser)[[-1, 3]]  # 使用测试模块的函数进行索引操作
    # 测试用例：测试在大型序列中使用 loc 进行切片操作
    def test_loc_getitem_large_series(self, monkeypatch):
        # 设置序列的大小截断值为 20
        size_cutoff = 20
        # 使用 monkeypatch 创建上下文环境
        with monkeypatch.context():
            # 设置 libindex 模块中的 _SIZE_CUTOFF 属性为 size_cutoff
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            # 创建一个 Series 对象，使用 np.arange(size_cutoff) 作为数据，使用 IntervalIndex 作为索引
            ser = Series(
                np.arange(size_cutoff),
                index=IntervalIndex.from_breaks(np.arange(size_cutoff + 1)),
            )

            # 使用不同方式进行 loc 操作，获取结果
            result1 = ser.loc[:8]  # 使用默认步长的切片操作
            result2 = ser.loc[0:8]  # 显式指定起始和结束的切片操作
            result3 = ser.loc[0:8:1]  # 显式指定起始、结束和步长的切片操作

        # 断言结果的一致性
        tm.assert_series_equal(result1, result2)
        tm.assert_series_equal(result1, result3)

    # 测试用例：测试在 DataFrame 中使用 loc 进行索引操作
    def test_loc_getitem_frame(self):
        # 创建一个 DataFrame 对象，包含列 'A'，并使用 pd.cut 对 'A' 进行分段
        df = DataFrame({"A": range(10)})
        ser = pd.cut(df.A, 5)  # 使用 pd.cut 对 'A' 列进行分段，并将结果保存到 ser 中
        df["B"] = ser  # 将分段结果添加为 DataFrame 的新列 'B'
        df = df.set_index("B")  # 将 'B' 列设置为索引列

        # 使用 loc 对 DataFrame 进行索引操作
        result = df.loc[4]  # 使用单个标签进行索引
        expected = df.iloc[4:6]  # 使用 iloc 进行位置索引
        tm.assert_frame_equal(result, expected)  # 断言结果的一致性

        # 测试索引不存在的情况，预期抛出 KeyError 异常
        with pytest.raises(KeyError, match="10"):
            df.loc[10]

        # 使用单个列表形式的标签进行索引操作
        result = df.loc[[4]]
        expected = df.iloc[4:6]
        tm.assert_frame_equal(result, expected)

        # 使用非唯一的标签进行索引操作
        result = df.loc[[4, 5]]
        expected = df.take([4, 5, 4, 5])
        tm.assert_frame_equal(result, expected)

        # 测试部分标签不存在的情况，预期抛出 KeyError 异常
        msg = (
            r"None of \[Index\(\[10\], dtype='object', name='B'\)\] "
            r"are in the \[index\]"
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[10]]

        # 测试部分标签不存在的情况，预期抛出 KeyError 异常
        with pytest.raises(KeyError, match=r"\[10\] not in index"):
            df.loc[[10, 4]]

    # 测试用例：测试在 IntervalIndex 中使用 loc 进行索引操作，处理 NaN 值情况
    def test_getitem_interval_with_nans(self, frame_or_series, indexer_sl):
        # GH#41831

        # 创建一个包含 NaN 值的 IntervalIndex
        index = IntervalIndex([np.nan, np.nan])
        key = index[:-1]

        # 根据 frame_or_series 的类型，创建相应的对象 obj
        obj = frame_or_series(range(2), index=index)
        if frame_or_series is DataFrame and indexer_sl is tm.setitem:
            obj = obj.T

        # 使用 indexer_sl 对 obj 进行索引操作，获取结果
        result = indexer_sl(obj)[key]
        expected = obj

        # 断言结果的一致性
        tm.assert_equal(result, expected)

    # 测试用例：测试在 IntervalIndex 中使用 loc 进行切片操作，并设置值
    def test_setitem_interval_with_slice(self):
        # GH#54722
        # 创建一个 IntervalIndex
        ii = IntervalIndex.from_breaks(range(4, 15))
        ser = Series(range(10), index=ii)

        orig = ser.copy()  # 备份原始的 Series 对象

        # 进行切片并设置值，预期为不引发异常
        ser.loc[1:3] = 20
        tm.assert_series_equal(ser, orig)  # 断言结果的一致性

        # 进行切片并设置值，验证预期结果
        ser.loc[6:8] = 19
        orig.iloc[1:4] = 19
        tm.assert_series_equal(ser, orig)

        # 创建一个新的 Series 对象，使用 ii 的部分索引
        ser2 = Series(range(5), index=ii[::2])
        orig2 = ser2.copy()  # 备份原始的 Series 对象

        # 进行切片并设置值，预期不引发异常
        ser2.loc[6:8] = 22
        orig2.iloc[1] = 22
        tm.assert_series_equal(ser2, orig2)

        # 进行切片并设置值，验证预期结果
        ser2.loc[5:7] = 21
        orig2.iloc[:2] = 21
        tm.assert_series_equal(ser2, orig2)
# 定义一个测试类 TestIntervalIndexInsideMultiIndex，用于测试带有 IntervalIndex 的 MultiIndex 切片操作
class TestIntervalIndexInsideMultiIndex:

    # 测试在 MultiIndex 中使用 IntervalIndex 进行标量切片
    def test_mi_intervalindex_slicing_with_scalar(self):
        # GH#27456: 这是 GitHub 上的 issue 编号

        # 创建一个 IntervalIndex 对象 ii，从给定的数组创建
        ii = pd.IntervalIndex.from_arrays(
            [0, 1, 10, 11, 0, 1, 10, 11],  # IntervalIndex 的左端点数组
            [1, 2, 11, 12, 1, 2, 11, 12],  # IntervalIndex 的右端点数组
            name="MP"  # IntervalIndex 的名称为 "MP"
        )

        # 创建一个 MultiIndex 对象 idx，从给定的数组创建
        idx = pd.MultiIndex.from_arrays(
            [
                pd.Index(["FC", "FC", "FC", "FC", "OWNER", "OWNER", "OWNER", "OWNER"]),  # 第一级索引
                pd.Index(["RID1", "RID1", "RID2", "RID2", "RID1", "RID1", "RID2", "RID2"]),  # 第二级索引
                ii,  # 第三级索引使用创建的 IntervalIndex 对象 ii
            ],
            names=["Item", "RID", "MP"]  # 设置 MultiIndex 对象的各级名称
        )

        # 将 DataFrame df 的索引设置为 idx
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8]})
        df.index = idx

        # 创建一个查询 DataFrame query_df
        query_df = pd.DataFrame(
            {
                "Item": ["FC", "OWNER", "FC", "OWNER", "OWNER"],  # Item 列
                "RID": ["RID1", "RID1", "RID1", "RID2", "RID2"],  # RID 列
                "MP": [0.2, 1.5, 1.6, 11.1, 10.9],  # MP 列
            }
        )

        # 对查询 DataFrame query_df 按索引排序
        query_df = query_df.sort_index()

        # 从 query_df 的 Item、RID 和 MP 列创建一个新的 MultiIndex 对象 idx
        idx = pd.MultiIndex.from_arrays([query_df.Item, query_df.RID, query_df.MP])
        query_df.index = idx

        # 使用 df 的 value 列，通过 query_df 的索引进行查询，并将结果存储在 result 中
        result = df.value.loc[query_df.index]

        # 对 IntervalIndex 进行切片操作，返回一个新的 IntervalIndex 对象 sliced_level
        sliced_level = ii.take([0, 1, 1, 3, 2])

        # 创建一个预期的 MultiIndex 对象 expected_index，使用 query_df 的各级值和切片后的 IntervalIndex
        expected_index = pd.MultiIndex.from_arrays(
            [idx.get_level_values(0), idx.get_level_values(1), sliced_level]
        )

        # 创建一个预期的 Series 对象 expected，其索引为 expected_index，名称为 "value"
        expected = pd.Series([1, 6, 2, 8, 7], index=expected_index, name="value")

        # 使用 pandas 的 tm 模块断言 result 和 expected 的 Series 相等
        tm.assert_series_equal(result, expected)

    # 标记为预期失败的测试用例，条件是 IS64 不为真，原因是 GH 23440
    @pytest.mark.xfail(not IS64, reason="GH 23440")
    @pytest.mark.parametrize("base", [101, 1010])
    # 测试带有 IntervalIndex 的重新索引行为
    def test_reindex_behavior_with_interval_index(self, base):
        # GH 51826: 这是 GitHub 上的 issue 编号

        # 创建一个 Series 对象 ser，使用范围为 base 的整数作为数据，IntervalIndex 作为索引
        ser = pd.Series(
            range(base),  # 数据范围从 0 到 base-1
            index=pd.IntervalIndex.from_arrays(range(base), range(1, base + 1)),  # 使用 IntervalIndex 作为索引
        )

        # 创建一个预期的 Series 对象 expected_result，包含两个 NaN 值，索引为 [NaN, 1.0]，数据类型为 float
        expected_result = pd.Series([np.nan, 0], index=[np.nan, 1.0], dtype=float)

        # 对 ser 进行重新索引操作，索引为 [NaN, 1.0]，将结果存储在 result 中
        result = ser.reindex(index=[np.nan, 1.0])

        # 使用 pandas 的 tm 模块断言 result 和 expected_result 的 Series 相等
        tm.assert_series_equal(result, expected_result)
```