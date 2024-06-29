# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_multiindex.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

import pandas._libs.index as libindex  # 导入pandas库的内部索引模块

import pandas as pd  # 导入pandas库，并使用pd作为别名
from pandas import (  # 从pandas库中导入多个特定类或函数
    CategoricalDtype,
    DataFrame,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm  # 导入pandas库的测试模块
from pandas.core.arrays.boolean import BooleanDtype  # 导入pandas库的布尔类型数组

class TestMultiIndexBasic:
    def test_multiindex_perf_warn(self, performance_warning):
        df = DataFrame(  # 创建DataFrame对象，包含三列数据：'jim', 'joe', 'jolie'
            {
                "jim": [0, 0, 1, 1],
                "joe": ["x", "x", "z", "y"],
                "jolie": np.random.default_rng(2).random(4),  # 使用随机数填充'jolie'列
            }
        ).set_index(["jim", "joe"])  # 将'jim'和'joe'列设置为多级索引

        with tm.assert_produces_warning(performance_warning):  # 使用tm.assert_produces_warning检测性能警告
            df.loc[(1, "z")]  # 使用.loc访问索引为(1, 'z')的行数据

        df = df.iloc[[2, 1, 3, 0]]  # 根据位置重新排列DataFrame的行
        with tm.assert_produces_warning(performance_warning):  # 使用tm.assert_produces_warning检测性能警告
            df.loc[(0,)]  # 使用.loc访问索引为(0,)的行数据

    @pytest.mark.parametrize("offset", [-5, 5])
    def test_indexing_over_hashtable_size_cutoff(self, monkeypatch, offset):
        size_cutoff = 20  # 设置大小截止值为20
        n = size_cutoff + offset  # 根据偏移量计算n的值

        with monkeypatch.context():  # 使用monkeypatch来修改全局变量
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)  # 设置libindex模块的_SIZE_CUTOFF变量为size_cutoff
            s = Series(np.arange(n), MultiIndex.from_arrays((["a"] * n, np.arange(n))))  # 创建带有多级索引的Series对象

            # hai it works!  # 断言检查索引为('a', 5)的元素是否为5
            assert s[("a", 5)] == 5
            assert s[("a", 6)] == 6
            assert s[("a", 7)] == 7

    def test_multi_nan_indexing(self):
        # GH 3588  # GitHub issue 3588
        df = DataFrame(  # 创建DataFrame对象，包含三列数据：'a', 'b', 'c'，其中包含NaN值
            {
                "a": ["R1", "R2", np.nan, "R4"],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            }
        )
        result = df.set_index(["a", "b"], drop=False)  # 将'a'和'b'列设置为索引，不删除原始列
        expected = DataFrame(  # 创建期望的DataFrame对象，保留原始索引
            {
                "a": ["R1", "R2", np.nan, "R4"],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            },
            index=[
                Index(["R1", "R2", np.nan, "R4"], name="a"),
                Index(["C1", "C2", "C3", "C4"], name="b"),
            ],
        )
        tm.assert_frame_equal(result, expected)  # 使用tm.assert_frame_equal比较结果与期望的DataFrame对象

    def test_exclusive_nat_column_indexing(self):
        # GH 38025  # GitHub issue 38025
        # test multi indexing when one column exclusively contains NaT values
        df = DataFrame(  # 创建DataFrame对象，包含三列数据：'a', 'b', 'c'，其中'a'列仅包含NaT值
            {
                "a": [pd.NaT, pd.NaT, pd.NaT, pd.NaT],
                "b": ["C1", "C2", "C3", "C4"],
                "c": [10, 15, np.nan, 20],
            }
        )
        df = df.set_index(["a", "b"])  # 将'a'和'b'列设置为索引
        expected = DataFrame(  # 创建期望的DataFrame对象，保留原始索引
            {
                "c": [10, 15, np.nan, 20],
            },
            index=[
                Index([pd.NaT, pd.NaT, pd.NaT, pd.NaT], name="a"),
                Index(["C1", "C2", "C3", "C4"], name="b"),
            ],
        )
        tm.assert_frame_equal(df, expected)  # 使用tm.assert_frame_equal比较结果与期望的DataFrame对象
    def test_nested_tuples_duplicates(self):
        # GH#30892
        # 创建包含重复日期时间和索引的多重索引对象
        dti = pd.to_datetime(["20190101", "20190101", "20190102"])
        idx = Index(["a", "a", "c"])
        mi = MultiIndex.from_arrays([dti, idx], names=["index1", "index2"])

        # 创建包含指定索引的 DataFrame，其中包括 NaN 值
        df = DataFrame({"c1": [1, 2, 3], "c2": [np.nan, np.nan, np.nan]}, index=mi)

        # 创建预期的 DataFrame，其中一个列的值被修改
        expected = DataFrame({"c1": df["c1"], "c2": [1.0, 1.0, np.nan]}, index=mi)

        # 深拷贝原始 DataFrame，并修改其中一个值，然后比较预期结果
        df2 = df.copy(deep=True)
        df2.loc[(dti[0], "a"), "c2"] = 1.0
        tm.assert_frame_equal(df2, expected)

        # 深拷贝原始 DataFrame，并修改其中一个值（使用列表形式），然后比较预期结果
        df3 = df.copy(deep=True)
        df3.loc[[(dti[0], "a")], "c2"] = 1.0
        tm.assert_frame_equal(df3, expected)

    def test_multiindex_with_datatime_level_preserves_freq(self):
        # https://github.com/pandas-dev/pandas/issues/35563
        # 创建包含日期时间和索引的多重索引对象，确保频率保持不变
        idx = Index(range(2), name="A")
        dti = pd.date_range("2020-01-01", periods=7, freq="D", name="B")
        mi = MultiIndex.from_product([idx, dti])
        df = DataFrame(np.random.default_rng(2).standard_normal((14, 2)), index=mi)

        # 选择第一级索引为 0 的数据，比较其索引结果和预期的日期时间索引
        result = df.loc[0].index
        tm.assert_index_equal(result, dti)
        assert result.freq == dti.freq

    def test_multiindex_complex(self):
        # GH#42145
        # 创建一个复杂数据的 DataFrame，包括复数和非复数数据，设置多重索引
        complex_data = [1 + 2j, 4 - 3j, 10 - 1j]
        non_complex_data = [3, 4, 5]
        result = DataFrame(
            {
                "x": complex_data,
                "y": non_complex_data,
                "z": non_complex_data,
            }
        )
        result.set_index(["x", "y"], inplace=True)

        # 创建预期的 DataFrame，设置相同的多重索引
        expected = DataFrame(
            {"z": non_complex_data},
            index=MultiIndex.from_arrays(
                [complex_data, non_complex_data],
                names=("x", "y"),
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_rename_multiindex_with_duplicates(self):
        # GH 38015
        # 创建一个带有重复条目的多重索引对象，然后对其进行重命名操作
        mi = MultiIndex.from_tuples([("A", "cat"), ("B", "cat"), ("B", "cat")])
        df = DataFrame(index=mi)
        df = df.rename(index={"A": "Apple"}, level=0)

        # 创建预期的 DataFrame，确保重命名操作正确
        mi2 = MultiIndex.from_tuples([("Apple", "cat"), ("B", "cat"), ("B", "cat")])
        expected = DataFrame(index=mi2)
        tm.assert_frame_equal(df, expected)

    def test_series_align_multiindex_with_nan_overlap_only(self):
        # GH 38439
        # 创建两个带有 NaN 值重叠的多重索引的 Series，并对齐它们
        mi1 = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        mi2 = MultiIndex.from_arrays([[np.nan, 82.0], [np.nan, np.nan]])
        ser1 = Series([1, 2], index=mi1)
        ser2 = Series([1, 2], index=mi2)
        result1, result2 = ser1.align(ser2)

        # 创建预期的 Series 对象，确保对齐操作正确
        mi = MultiIndex.from_arrays([[81.0, 82.0, np.nan], [np.nan, np.nan, np.nan]])
        expected1 = Series([1.0, np.nan, 2.0], index=mi)
        expected2 = Series([np.nan, 2.0, 1.0], index=mi)

        tm.assert_series_equal(result1, expected1)
        tm.assert_series_equal(result2, expected2)
    def test_series_align_multiindex_with_nan(self):
        # GH 38439
        # 创建第一个多级索引对象 mi1，包含 NaN 值
        mi1 = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        # 创建第二个多级索引对象 mi2，包含 NaN 值
        mi2 = MultiIndex.from_arrays([[np.nan, 81.0], [np.nan, np.nan]])
        # 使用第一个多级索引对象创建 Series 对象 ser1
        ser1 = Series([1, 2], index=mi1)
        # 使用第二个多级索引对象创建 Series 对象 ser2
        ser2 = Series([1, 2], index=mi2)
        # 对 ser1 和 ser2 进行索引对齐操作
        result1, result2 = ser1.align(ser2)

        # 创建期望的 Series 对象 expected1 和 expected2，以 mi 作为索引
        mi = MultiIndex.from_arrays([[81.0, np.nan], [np.nan, np.nan]])
        expected1 = Series([1, 2], index=mi)
        expected2 = Series([2, 1], index=mi)

        # 断言 result1 和 expected1 相等
        tm.assert_series_equal(result1, expected1)
        # 断言 result2 和 expected2 相等
        tm.assert_series_equal(result2, expected2)

    def test_nunique_smoke(self):
        # GH 34019
        # 创建一个 DataFrame，并设置其索引，然后计算唯一索引值的数量
        n = DataFrame([[1, 2], [1, 2]]).set_index([0, 1]).index.nunique()
        # 断言唯一索引值的数量为 1
        assert n == 1

    def test_multiindex_repeated_keys(self):
        # GH19414
        # 创建一个具有重复键的 Series 对象，并进行索引操作
        tm.assert_series_equal(
            Series([1, 2], MultiIndex.from_arrays([["a", "b"]])).loc[
                ["a", "a", "b", "b"]
            ],
            Series([1, 1, 2, 2], MultiIndex.from_arrays([["a", "a", "b", "b"]])),
        )

    def test_multiindex_with_na_missing_key(self):
        # GH46173
        # 从字典创建一个 DataFrame，其中包含具有 None 键的列
        df = DataFrame.from_dict(
            {
                ("foo",): [1, 2, 3],
                ("bar",): [5, 6, 7],
                (None,): [8, 9, 0],
            }
        )
        # 使用 pytest 检测期望的 KeyError 异常，确保捕获到 "missing_key" 的异常信息
        with pytest.raises(KeyError, match="missing_key"):
            df[[("missing_key",)]]

    def test_multiindex_dtype_preservation(self):
        # GH51261
        # 创建一个具有多级索引的 DataFrame，并将其中一列转换为 category 类型
        columns = MultiIndex.from_tuples([("A", "B")], names=["lvl1", "lvl2"])
        df = DataFrame(["value"], columns=columns).astype("category")
        # 获取非多级索引 DataFrame 中 "B" 列的 dtype，并断言其为 CategoricalDtype 类型
        df_no_multiindex = df["A"]
        assert isinstance(df_no_multiindex["B"].dtype, CategoricalDtype)

        # geopandas 1763 analogue
        # 创建一个包含布尔类型列的 DataFrame
        df = DataFrame(
            [[1, 0], [0, 1]],
            columns=[
                ["foo", "foo"],
                ["location", "location"],
                ["x", "y"],
            ],
        ).assign(bools=Series([True, False], dtype="boolean"))
        # 断言 "bools" 列的 dtype 为 BooleanDtype 类型
        assert isinstance(df["bools"].dtype, BooleanDtype)

    def test_multiindex_from_tuples_with_nan(self):
        # GH#23578
        # 创建一个包含 NaN 值的 MultiIndex 对象
        result = MultiIndex.from_tuples([("a", "b", "c"), np.nan, ("d", "", "")])
        # 创建期望的 MultiIndex 对象，其中 NaN 值用 np.nan 表示
        expected = MultiIndex.from_tuples(
            [("a", "b", "c"), (np.nan, np.nan, np.nan), ("d", "", "")]
        )
        # 断言 result 和 expected 相等
        tm.assert_index_equal(result, expected)
```