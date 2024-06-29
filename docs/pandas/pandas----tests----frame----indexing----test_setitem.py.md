# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_setitem.py`

```
    from datetime import datetime  # 导入 datetime 模块中的 datetime 类

    import numpy as np  # 导入 NumPy 库并使用别名 np
    import pytest  # 导入 pytest 测试框架

    from pandas.core.dtypes.base import _registry as ea_registry  # 导入 pandas 库中的 _registry 模块，并使用别名 ea_registry
    from pandas.core.dtypes.common import is_object_dtype  # 导入 pandas 库中的 is_object_dtype 函数
    from pandas.core.dtypes.dtypes import (  # 导入 pandas 库中的各种数据类型
        CategoricalDtype,
        DatetimeTZDtype,
        IntervalDtype,
        PeriodDtype,
    )

    import pandas as pd  # 导入 pandas 库并使用别名 pd
    from pandas import (  # 从 pandas 库中导入多个类和函数
        Categorical,
        DataFrame,
        DatetimeIndex,
        Index,
        Interval,
        IntervalIndex,
        MultiIndex,
        NaT,
        Period,
        PeriodIndex,
        Series,
        Timestamp,
        cut,
        date_range,
        notna,
        period_range,
    )
    import pandas._testing as tm  # 导入 pandas 测试模块，并使用别名 tm
    from pandas.core.arrays import SparseArray  # 导入 pandas 库中的 SparseArray 类

    from pandas.tseries.offsets import BDay  # 导入 pandas 时间序列偏移量 BDay 类


    class TestDataFrameSetItem:  # 定义 TestDataFrameSetItem 类，用于测试 DataFrame 的设置项功能

        def test_setitem_str_subclass(self):  # 测试字符串子类作为索引设置项的功能

            # GH#37366
            class mystring(str):  # 定义一个名为 mystring 的字符串子类
                __slots__ = ()  # 空的 __slots__，用于优化内存使用

            data = ["2020-10-22 01:21:00+00:00"]  # 时间数据列表
            index = DatetimeIndex(data)  # 创建时间索引对象
            df = DataFrame({"a": [1]}, index=index)  # 创建 DataFrame 对象 df，带有时间索引和列 'a'
            df["b"] = 2  # 向 DataFrame 添加新列 'b'，赋值为 2
            df[mystring("c")] = 3  # 向 DataFrame 添加新列 'c'，使用 mystring 类型的索引，赋值为 3
            expected = DataFrame({"a": [1], "b": [2], mystring("c"): [3]}, index=index)  # 预期的 DataFrame 结果
            tm.assert_equal(df, expected)  # 使用测试模块中的 assert_equal 函数比较 df 和 expected 是否相等

        @pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器定义参数化测试
            "dtype", ["int32", "int64", "uint32", "uint64", "float32", "float64"]
        )
        def test_setitem_dtype(self, dtype, float_frame):  # 测试设置 DataFrame 中特定数据类型列的功能
            # Use integers since casting negative floats to uints is undefined
            arr = np.random.default_rng(2).integers(1, 10, len(float_frame))  # 生成随机整数数组 arr

            float_frame[dtype] = np.array(arr, dtype=dtype)  # 向 float_frame 中添加名为 dtype 的列，数据类型为 dtype
            assert float_frame[dtype].dtype.name == dtype  # 断言新添加的列的数据类型与指定的 dtype 一致

        def test_setitem_list_not_dataframe(self, float_frame):  # 测试向 DataFrame 中添加非 DataFrame 类型的列表的功能
            data = np.random.default_rng(2).standard_normal((len(float_frame), 2))  # 生成符合正态分布的随机数数组 data
            float_frame[["A", "B"]] = data  # 向 float_frame 中添加列 'A' 和 'B'，赋值为 data
            tm.assert_almost_equal(float_frame[["A", "B"]].values, data)  # 使用测试模块中的 assert_almost_equal 函数比较结果

        def test_setitem_error_msmgs(self):  # 测试设置项功能中的错误处理信息

            # GH 7432
            df = DataFrame(  # 创建 DataFrame 对象 df
                {"bar": [1, 2, 3], "baz": ["d", "e", "f"]},  # 列 'bar' 和 'baz' 的数据
                index=Index(["a", "b", "c"], name="foo"),  # 使用指定索引和名称 'foo'
            )
            ser = Series(  # 创建 Series 对象 ser
                ["g", "h", "i", "j"],  # 数据为列表
                index=Index(["a", "b", "c", "a"], name="foo"),  # 使用指定索引和名称 'foo'
                name="fiz",  # 设置 Series 的名称为 'fiz'
            )
            msg = "cannot reindex on an axis with duplicate labels"  # 错误信息提示
            with pytest.raises(ValueError, match=msg):  # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误信息 msg
                df["newcol"] = ser  # 向 df 中添加新列 'newcol'，赋值为 ser

            # GH 4107, more descriptive error message
            df = DataFrame(  # 创建 DataFrame 对象 df
                np.random.default_rng(2).integers(0, 2, (4, 4)),  # 随机生成的整数数组作为数据
                columns=["a", "b", "c", "d"],  # 列名称为 'a', 'b', 'c', 'd'
            )

            msg = "Cannot set a DataFrame with multiple columns to the single column gr"  # 错误信息提示
            with pytest.raises(ValueError, match=msg):  # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误信息 msg
                df["gr"] = df.groupby(["b", "c"]).count()  # 尝试向 df 中添加新列 'gr'，赋值为根据 ['b', 'c'] 分组计数的结果

            # GH 55956, specific message for zero columns
            msg = "Cannot set a DataFrame without columns to the column gr"  # 错误信息提示
            with pytest.raises(ValueError, match=msg):  # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误信息 msg
                df["gr"] = DataFrame()  # 尝试向 df 中添加新列 'gr'，赋值为空 DataFrame 对象
    # 测试向 DataFrame 中插入数据的性能
    def test_setitem_benchmark(self):
        # 设置测试数据大小
        N = 10
        K = 5
        # 创建一个空的 DataFrame，指定行索引范围
        df = DataFrame(index=range(N))
        # 生成一个随机数列作为新列的数据
        new_col = np.random.default_rng(2).standard_normal(N)
        # 循环将新列数据插入 DataFrame 中
        for i in range(K):
            df[i] = new_col
        # 生成期望的 DataFrame，重复新列数据 K 次后重新排列
        expected = DataFrame(np.repeat(new_col, K).reshape(N, K), index=range(N))
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # 测试插入不同数据类型列时的行为
    def test_setitem_different_dtype(self):
        # 创建一个具有随机数数据的 DataFrame，指定索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=np.arange(5),
            columns=["c", "b", "a"],
        )
        # 在第 0 列位置插入一个新列，该列数据来源于已有列 "a"
        df.insert(0, "foo", df["a"])
        # 在第 2 列位置插入一个新列，该列数据来源于已有列 "c"
        df.insert(2, "bar", df["c"])

        # 插入新的列 "x"，其数据类型为 float32
        df["x"] = df["a"].astype("float32")
        # 检查结果的数据类型
        result = df.dtypes
        # 期望的数据类型 Series
        expected = Series(
            [np.dtype("float64")] * 5 + [np.dtype("float32")],
            index=["foo", "c", "bar", "b", "a", "x"],
        )
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

        # 替换已有列 "a" 的数据类型为 float32
        df["a"] = df["a"].astype("float32")
        # 检查结果的数据类型
        result = df.dtypes
        # 期望的数据类型 Series
        expected = Series(
            [np.dtype("float64")] * 4 + [np.dtype("float32")] * 2,
            index=["foo", "c", "bar", "b", "a", "x"],
        )
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

        # 插入新的列 "y"，其数据类型为 int32
        df["y"] = df["a"].astype("int32")
        # 检查结果的数据类型
        result = df.dtypes
        # 期望的数据类型 Series
        expected = Series(
            [np.dtype("float64")] * 4 + [np.dtype("float32")] * 2 + [np.dtype("int32")],
            index=["foo", "c", "bar", "b", "a", "x", "y"],
        )
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    # 测试在空 DataFrame 中插入列是否正常
    def test_setitem_empty_columns(self):
        # 创建一个空的 DataFrame，指定行索引为 ["A", "B", "C"]
        df = DataFrame(index=["A", "B", "C"])
        # 在 DataFrame 中插入一个列 "X"，其数据等于索引值
        df["X"] = df.index
        # 重新赋值列 "X" 的数据
        df["X"] = ["x", "y", "z"]
        # 期望的结果 DataFrame
        exp = DataFrame(data={"X": ["x", "y", "z"]}, index=["A", "B", "C"])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, exp)

    # 测试在具有 datetime64 索引的空 DataFrame 中插入列是否正常
    def test_setitem_dt64_index_empty_columns(self):
        # 生成一个时间范围
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        # 创建一个空的 DataFrame，指定行索引为时间序列长度
        df = DataFrame(index=np.arange(len(rng)))

        # 在 DataFrame 中插入一个列 "A"，其数据为时间序列
        df["A"] = rng
        # 断言该列的数据类型为 datetime64
        assert df["A"].dtype == np.dtype("M8[ns]")

    # 测试在具有时间戳索引的空 DataFrame 中插入列是否正常
    def test_setitem_timestamp_empty_columns(self):
        # 创建一个空的 DataFrame，指定行索引为 [0, 1, 2]
        df = DataFrame(index=range(3))
        # 在 DataFrame 中插入一个列 "now"，其数据为指定的时间戳
        df["now"] = Timestamp("20130101", tz="UTC")

        # 期望的结果 DataFrame
        expected = DataFrame(
            [[Timestamp("20130101", tz="UTC")]] * 3, index=[0, 1, 2], columns=["now"]
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    # 测试插入长度不匹配的分类数据列时是否引发异常
    def test_setitem_wrong_length_categorical_dtype_raises(self):
        # 创建一个分类数据对象
        cat = Categorical.from_codes([0, 1, 1, 0, 1, 2], ["a", "b", "c"])
        # 创建一个整数数据的 DataFrame，列名为 "bar"
        df = DataFrame(range(10), columns=["bar"])

        # 准备断言失败时的错误信息
        msg = (
            rf"Length of values \({len(cat)}\) "
            rf"does not match length of index \({len(df)}\)"
        )
        # 期望引发 ValueError 异常，并且异常信息与预期匹配
    # 定义测试函数，测试将稀疏数组设置为 DataFrame 的新列
    def test_setitem_with_sparse_value(self):
        # GH#8131：GitHub issue 号码，标识此测试的背景和相关讨论
        df = DataFrame({"c_1": ["a", "b", "c"], "n_1": [1.0, 2.0, 3.0]})
        # 创建稀疏数组对象
        sp_array = SparseArray([0, 0, 1])
        # 将稀疏数组设置为 DataFrame 的新列
        df["new_column"] = sp_array

        # 准备预期的 Series 对象，用于比较
        expected = Series(sp_array, name="new_column")
        # 断言 DataFrame 中的新列与预期的 Series 相等
        tm.assert_series_equal(df["new_column"], expected)

    # 定义测试函数，测试将不对齐的稀疏 Series 设置为 DataFrame 的新列
    def test_setitem_with_unaligned_sparse_value(self):
        df = DataFrame({"c_1": ["a", "b", "c"], "n_1": [1.0, 2.0, 3.0]})
        # 创建带索引的稀疏 Series 对象
        sp_series = Series(SparseArray([0, 0, 1]), index=[2, 1, 0])

        # 将稀疏 Series 设置为 DataFrame 的新列
        df["new_column"] = sp_series
        # 准备预期的 Series 对象，用于比较
        expected = Series(SparseArray([1, 0, 0]), name="new_column")
        # 断言 DataFrame 中的新列与预期的 Series 相等
        tm.assert_series_equal(df["new_column"], expected)

    # 定义测试函数，测试将 Period 对象设置为 DataFrame 的新列，并保留数据类型
    def test_setitem_period_preserves_dtype(self):
        # GH: 26861：GitHub issue 号码，标识此测试的背景和相关讨论
        data = [Period("2003-12", "D")]
        result = DataFrame([])
        # 将 Period 对象设置为 DataFrame 的新列
        result["a"] = data

        # 准备预期的 DataFrame 对象，用于比较
        expected = DataFrame({"a": data})
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试将字典形式的数据设置为 DataFrame，并保留数据类型
    def test_setitem_dict_preserves_dtypes(self):
        # https://github.com/pandas-dev/pandas/issues/34573：链接到相关 GitHub issue
        expected = DataFrame(
            {
                "a": Series([0, 1, 2], dtype="int64"),
                "b": Series([1, 2, 3], dtype=float),
                "c": Series([1, 2, 3], dtype=float),
                "d": Series([1, 2, 3], dtype="uint32"),
            }
        )
        df = DataFrame(
            {
                "a": Series([], dtype="int64"),
                "b": Series([], dtype=float),
                "c": Series([], dtype=float),
                "d": Series([], dtype="uint32"),
            }
        )
        # 循环迭代并向 DataFrame 中添加数据
        for idx, b in enumerate([1, 2, 3]):
            df.loc[df.shape[0]] = {
                "a": int(idx),
                "b": float(b),
                "c": float(b),
                "d": np.uint32(b),
            }
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)

    # 定义参数化测试函数，测试不同扩展类型的对象设置为 DataFrame 的新列
    @pytest.mark.parametrize(
        "obj,dtype",
        [
            (Period("2020-01"), PeriodDtype("M")),
            (Interval(left=0, right=5), IntervalDtype("int64", "right")),
            (
                Timestamp("2011-01-01", tz="US/Eastern"),
                DatetimeTZDtype(unit="s", tz="US/Eastern"),
            ),
        ],
    )
    def test_setitem_extension_types(self, obj, dtype):
        # GH: 34832：GitHub issue 号码，标识此测试的背景和相关讨论
        expected = DataFrame({"idx": [1, 2, 3], "obj": Series([obj] * 3, dtype=dtype)})

        df = DataFrame({"idx": [1, 2, 3]})
        # 将扩展类型的对象设置为 DataFrame 的新列
        df["obj"] = obj

        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)
    def test_setitem_with_ea_name(self, ea_name):
        # GH 38386
        # 创建一个包含单个元素的DataFrame对象
        result = DataFrame([0])
        # 设置DataFrame对象中的列名为ea_name的值为[1]
        result[ea_name] = [1]
        # 创建一个期望的DataFrame对象，包含列名为0和ea_name的值
        expected = DataFrame({0: [0], ea_name: [1]})
        # 使用测试工具比较result和expected是否相等
        tm.assert_frame_equal(result, expected)

    def test_setitem_dt64_ndarray_with_NaT_and_diff_time_units(self):
        # GH#7492
        # 创建一个包含datetime64[ns]类型数组的Series，并转换为DataFrame对象
        data_ns = np.array([1, "nat"], dtype="datetime64[ns]")
        result = Series(data_ns).to_frame()
        # 将data_ns数组作为"new"列添加到result DataFrame对象中
        result["new"] = data_ns
        # 创建一个期望的DataFrame对象，包含列名为0和"new"的值，数据类型为datetime64[ns]
        expected = DataFrame({0: [1, None], "new": [1, None]}, dtype="datetime64[ns]")
        # 使用测试工具比较result和expected是否相等
        tm.assert_frame_equal(result, expected)

        # 确保在设定data_s数组时不会出现OutOfBoundsDatetime错误；自2.0版本起，我们保留"M8[s]"类型
        # 创建一个包含datetime64[s]类型数组的data_s
        data_s = np.array([1, "nat"], dtype="datetime64[s]")
        # 将data_s作为"new"列添加到result DataFrame对象中
        result["new"] = data_s
        # 使用测试工具比较result和expected的第一个列是否相等
        tm.assert_series_equal(result[0], expected[0])
        # 使用测试工具比较result的"new"列的numpy数组与data_s是否相等
        tm.assert_numpy_array_equal(result["new"].to_numpy(), data_s)

    @pytest.mark.parametrize("unit", ["h", "m", "s", "ms", "D", "M", "Y"])
    def test_frame_setitem_datetime64_col_other_units(self, unit):
        # 检查非纳秒级别的dt64值在设定时会被转换为dt64类型
        #  在一个尚未存在的列中进行设定
        n = 100

        # 创建一个对应单位的dtype
        dtype = np.dtype(f"M8[{unit}]")
        # 创建一个包含指定单位的vals数组
        vals = np.arange(n, dtype=np.int64).view(dtype)
        if unit in ["s", "ms"]:
            # 如果单位在支持的范围内
            ex_vals = vals
        else:
            # 获取最接近的支持单位，即"s"
            ex_vals = vals.astype("datetime64[s]")

        # 创建一个DataFrame对象，包含一个整数列"ints"和索引
        df = DataFrame({"ints": np.arange(n)}, index=np.arange(n))
        # 将vals作为单位列(unit)添加到df DataFrame对象中
        df[unit] = vals

        # 断言df的单位列的数据类型等于ex_vals的数据类型
        assert df[unit].dtype == ex_vals.dtype
        # 断言df的单位列的值数组与ex_vals的值数组是否完全相等
        assert (df[unit].values == ex_vals).all()

    @pytest.mark.parametrize("unit", ["h", "m", "s", "ms", "D", "M", "Y"])
    def test_frame_setitem_existing_datetime64_col_other_units(self, unit):
        # 检查非纳秒级别的dt64值在设定时会被转换为dt64类型
        #  在一个已存在的dt64列中进行设定
        n = 100

        # 创建一个对应单位的dtype
        dtype = np.dtype(f"M8[{unit}]")
        # 创建一个包含指定单位的vals数组
        vals = np.arange(n, dtype=np.int64).view(dtype)
        # 将vals转换为datetime64[ns]类型作为期望的值
        ex_vals = vals.astype("datetime64[ns]")

        # 创建一个DataFrame对象，包含一个整数列"ints"和一个已存在的日期列"dates"
        df = DataFrame({"ints": np.arange(n)}, index=np.arange(n))
        df["dates"] = np.arange(n, dtype=np.int64).view("M8[ns]")

        # 使用vals数组覆盖现有的"dates"列
        df["dates"] = vals
        # 断言df的"dates"列的值数组与ex_vals的值数组是否完全相等
        assert (df["dates"].values == ex_vals).all()
    # 测试函数，用于测试设置带有日期时间和时区的数据框
    def test_setitem_dt64tz(self, timezone_frame):
        # 将传入的数据框赋值给变量 df
        df = timezone_frame
        # 从数据框 df 的列 "B" 中提取数据，并重命名为 "foo"，赋值给变量 idx
        idx = df["B"].rename("foo")

        # 设置数据框 df 中的列 "C"，将 idx 的数据赋给 "C"
        df["C"] = idx
        # 断言数据框 df 中列 "C" 的数据与 Series(idx, name="C") 相等
        tm.assert_series_equal(df["C"], Series(idx, name="C"))

        # 设置数据框 df 中的列 "D" 为字符串 "foo"
        df["D"] = "foo"
        # 再次设置数据框 df 中的列 "D"，将 idx 的数据赋给 "D"
        df["D"] = idx
        # 断言数据框 df 中列 "D" 的数据与 Series(idx, name="D") 相等
        tm.assert_series_equal(df["D"], Series(idx, name="D"))
        # 删除数据框 df 中的列 "D"
        del df["D"]

        # 断言数据框 df 的第 1 块和第 2 块的值相等，用于检查是否为拷贝
        v1 = df._mgr.blocks[1].values
        v2 = df._mgr.blocks[2].values
        tm.assert_extension_array_equal(v1, v2)
        # 获取第 1 块和第 2 块的基础 ndarray，并断言它们的 id 相同
        v1base = v1._ndarray.base
        v2base = v2._ndarray.base
        assert id(v1base) == id(v2base)

        # 创建数据框 df 的副本 df2
        df2 = df.copy()
        # 修改 df2 中的某些单元格为 NaT
        df2.iloc[1, 1] = NaT
        df2.iloc[1, 2] = NaT
        # 从修改后的 df2 中提取列 "B" 的结果，并断言其非缺失值
        result = df2["B"]
        tm.assert_series_equal(notna(result), Series([True, False, True], name="B"))
        # 断言 df2 和 df 的数据类型相等
        tm.assert_series_equal(df2.dtypes, df.dtypes)

    # 测试函数，用于测试在 PeriodIndex 中设置数据
    def test_setitem_periodindex(self):
        # 创建一个 PeriodIndex
        rng = period_range("1/1/2000", periods=5, name="index")
        # 使用随机生成的数据创建数据框 df，其索引为 rng
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=rng)

        # 设置数据框 df 中的列 "Index"，将 rng 的数据赋给 "Index"
        df["Index"] = rng
        # 创建一个 Index 对象 rs，用于存储 df 中的列 "Index"
        rs = Index(df["Index"])
        # 断言 rs 和 rng 相等，不检查名称
        tm.assert_index_equal(rs, rng, check_names=False)
        # 断言 rs 的名称为 "Index"
        assert rs.name == "Index"
        # 断言 rng 的名称为 "index"
        assert rng.name == "index"

        # 对数据框 df 进行重置索引，并将索引设置为 "index"，存储在 rs 中
        rs = df.reset_index().set_index("index")
        # 断言 rs 的索引类型为 PeriodIndex
        assert isinstance(rs.index, PeriodIndex)
        # 断言 rs 的索引与 rng 相等
        tm.assert_index_equal(rs.index, rng)

    # 测试函数，用于测试将数组完整地设置为数据框的列
    def test_setitem_complete_column_with_array(self):
        # GH#37954
        # 创建包含两列的数据框 df 和一个数组 arr
        df = DataFrame({"a": ["one", "two", "three"], "b": [1, 2, 3]})
        arr = np.array([[1, 1], [3, 1], [5, 1]])
        # 将数组 arr 设置为数据框 df 的列 "c" 和 "d"
        df[["c", "d"]] = arr
        # 创建预期的数据框 expected
        expected = DataFrame(
            {
                "a": ["one", "two", "three"],
                "b": [1, 2, 3],
                "c": [1, 3, 5],
                "d": [1, 1, 1],
            }
        )
        # 将 expected 的列 "c" 和 "d" 的数据类型转换为 arr 的数据类型
        expected["c"] = expected["c"].astype(arr.dtype)
        expected["d"] = expected["d"].astype(arr.dtype)
        # 断言 expected 的列 "c" 和 "d" 的数据类型与 arr 的数据类型相等
        assert expected["c"].dtype == arr.dtype
        assert expected["d"].dtype == arr.dtype
        # 断言数据框 df 和预期的数据框 expected 相等
        tm.assert_frame_equal(df, expected)

    # 测试函数，用于测试在 PeriodIndex 中设置数据时的 dtype
    def test_setitem_period_d_dtype(self):
        # GH 39763
        # 创建一个 PeriodIndex rng
        rng = period_range("2016-01-01", periods=9, freq="D", name="A")
        # 创建结果数据框 result
        result = DataFrame(rng)
        # 创建预期的数据框 expected
        expected = DataFrame(
            {"A": ["NaT", "NaT", "NaT", "NaT", "NaT", "NaT", "NaT", "NaT", "NaT"]},
            dtype="period[D]",
        )
        # 将 result 的所有行设置为 rng 的 _na_value
        result.iloc[:] = rng._na_value
        # 断言 result 和 expected 相等
        tm.assert_frame_equal(result, expected)

    # 参数化测试，测试不同的 dtype
    @pytest.mark.parametrize("dtype", ["f8", "i8", "u8"])
    def test_setitem_bool_with_numeric_index(self, dtype):
        # GH#36319
        # 创建一个索引对象，包含整数元素 1, 2, 3，指定数据类型为 dtype
        cols = Index([1, 2, 3], dtype=dtype)
        # 创建一个 3x3 的随机数 DataFrame，列索引使用上述创建的 cols
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=cols)

        # 向 DataFrame df 中添加一个布尔类型的列，索引为 False，值为字符串列表 ["a", "b", "c"]
        df[False] = ["a", "b", "c"]

        # 创建一个预期的列索引对象，包含整数元素 1, 2, 3, False，数据类型为 object
        expected_cols = Index([1, 2, 3, False], dtype=object)
        # 如果 dtype 为 "f8"，则将预期的列索引对象改为包含浮点数元素 1.0, 2.0, 3.0, False，数据类型为 object
        if dtype == "f8":
            expected_cols = Index([1.0, 2.0, 3.0, False], dtype=object)

        # 使用测试工具函数检查 df 的列索引是否与预期的列索引对象 expected_cols 相等
        tm.assert_index_equal(df.columns, expected_cols)

    @pytest.mark.parametrize("indexer", ["B", ["B"]])
    def test_setitem_frame_length_0_str_key(self, indexer):
        # GH#38831
        # 创建一个空的 DataFrame，列索引为 ["A", "B"]
        df = DataFrame(columns=["A", "B"])
        # 创建另一个 DataFrame，包含列 "B" 和对应的数据 [1, 2]
        other = DataFrame({"B": [1, 2]})
        # 使用 indexer 作为列索引，将 other DataFrame 中的数据添加到 df 中
        df[indexer] = other
        # 创建一个预期的 DataFrame，包含列 "A" 全为 NaN，列 "B" 包含数据 [1, 2]
        expected = DataFrame({"A": [np.nan] * 2, "B": [1, 2]})
        # 将预期 DataFrame 的列 "A" 数据类型改为 object
        expected["A"] = expected["A"].astype("object")
        # 使用测试工具函数检查 df 是否与预期 DataFrame expected 相等
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_duplicate_columns(self):
        # GH#15695
        # 创建一个 DataFrame，行索引为 range(3)，列索引为 ["A", "B", "C", "A", "B", "C"]
        cols = ["A", "B", "C"] * 2
        df = DataFrame(index=range(3), columns=cols)
        # 修改第 0 行 "A" 列的值为元组 (0, 3)
        df.loc[0, "A"] = (0, 3)
        # 修改所有行 "B" 列的值为元组 (1, 4)
        df.loc[:, "B"] = (1, 4)
        # 添加一列 "C"，其值为元组 (2, 5)
        df["C"] = (2, 5)
        # 创建一个预期的 DataFrame，包含相同的数据结构但数据类型为 object
        expected = DataFrame(
            [
                [0, 1, 2, 3, 4, 5],
                [np.nan, 1, 2, np.nan, 4, 5],
                [np.nan, 1, 2, np.nan, 4, 5],
            ],
            dtype="object",
        )

        # 将预期 DataFrame 的第 2 列和第 5 列数据类型改为 np.int64
        expected[2] = expected[2].astype(np.int64)
        expected[5] = expected[5].astype(np.int64)
        # 将预期 DataFrame 的列索引改为 cols
        expected.columns = cols

        # 使用测试工具函数检查 df 是否与预期 DataFrame expected 相等
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_duplicate_columns_size_mismatch(self):
        # GH#39510
        # 创建一个 DataFrame，行索引为 range(3)，列索引为 ["A", "B", "C", "A", "B", "C"]
        cols = ["A", "B", "C"] * 2
        df = DataFrame(index=range(3), columns=cols)
        # 使用 pytest 的异常断言，检查尝试用长度不匹配的元组 (0, 3, 5) 替换列 "A" 时是否触发 ValueError 异常
        with pytest.raises(ValueError, match="Columns must be same length as key"):
            df[["A"]] = (0, 3, 5)

        # 创建一个子集 df2，包含前三列
        df2 = df.iloc[:, :3]  # unique columns
        # 使用 pytest 的异常断言，检查尝试用长度不匹配的元组 (0, 3, 5) 替换列 "A" 时是否触发 ValueError 异常
        with pytest.raises(ValueError, match="Columns must be same length as key"):
            df2[["A"]] = (0, 3, 5)

    @pytest.mark.parametrize("cols", [["a", "b", "c"], ["a", "a", "a"]])
    def test_setitem_df_wrong_column_number(self, cols):
        # GH#38604
        # 创建一个 DataFrame，包含单行数据 [1, 2, 3]，列索引为 cols
        df = DataFrame([[1, 2, 3]], columns=cols)
        # 创建另一个 DataFrame，包含单行数据 [10, 11]，列索引为 ["d", "e"]
        rhs = DataFrame([[10, 11]], columns=["d", "e"])
        # 使用 pytest 的异常断言，检查尝试用 rhs 替换列 "a" 时是否触发 ValueError 异常
        msg = "Columns must be same length as key"
        with pytest.raises(ValueError, match=msg):
            df["a"] = rhs

    def test_setitem_listlike_indexer_duplicate_columns(self):
        # GH#38604
        # 创建一个 DataFrame，包含单行数据 [1, 2, 3]，列索引为 ["a", "b", "b"]
        df = DataFrame([[1, 2, 3]], columns=["a", "b", "b"])
        # 创建另一个 DataFrame，包含单行数据 [10, 11, 12]，列索引为 ["a", "b", "b"]
        rhs = DataFrame([[10, 11, 12]], columns=["a", "b", "b"])
        # 使用 rhs 替换列 "a" 和 "b"
        df[["a", "b"]] = rhs
        # 创建一个预期的 DataFrame，包含单行数据 [10, 11, 12]，列索引为 ["a", "b", "b"]
        expected = DataFrame([[10, 11, 12]], columns=["a", "b", "b"])
        # 使用测试工具函数检查 df 是否与预期 DataFrame expected 相等
        tm.assert_frame_equal(df, expected)

        # 向 df 添加一列 "c"，并使用 rhs 的列 "b" 替换
        df[["c", "b"]] = rhs
        # 创建一个预期的 DataFrame，包含单行数据 [10, 11, 12, 10]，列索引为 ["a", "b", "b", "c"]
        expected = DataFrame([[10, 11, 12, 10]], columns=["a", "b", "b", "c"])
        # 使用测试工具函数检查 df 是否与预期 DataFrame expected 相等
        tm.assert_frame_equal(df, expected)
    # 测试设置列表式索引器的情况，处理重复列名但长度不相等的情况
    def test_setitem_listlike_indexer_duplicate_columns_not_equal_length(self):
        # 创建包含重复列名的 DataFrame 对象
        df = DataFrame([[1, 2, 3]], columns=["a", "b", "b"])
        # 创建右侧 DataFrame 对象，用于赋值
        rhs = DataFrame([[10, 11]], columns=["a", "b"])
        # 准备错误消息，用于断言异常
        msg = "Columns must be same length as key"
        # 使用 pytest 断言抛出 ValueError 异常并匹配指定消息
        with pytest.raises(ValueError, match=msg):
            # 尝试将 rhs 赋值给 df 的列 "a" 和 "b"
            df[["a", "b"]] = rhs

    # 测试设置区间数据的情况
    def test_setitem_intervals(self):
        # 创建包含列 "A" 的 DataFrame 对象
        df = DataFrame({"A": range(10)})
        # 使用 cut 函数将 "A" 列划分为 5 个区间
        ser = cut(df["A"], 5)
        # 断言 ser 的 categories 属性为 IntervalIndex 类型
        assert isinstance(ser.cat.categories, IntervalIndex)

        # 将不同形式的 Series 赋值给 df 的不同列

        # 列 "B" 和 "D" 被转换为 Categorical 类型
        df["B"] = ser
        df["C"] = np.array(ser)
        df["D"] = ser.values
        df["E"] = np.array(ser.values)
        df["F"] = ser.astype(object)

        # 断言列的类型及其子类型

        # 列 "B" 应为 CategoricalDtype 类型
        assert isinstance(df["B"].dtype, CategoricalDtype)
        # 列 "B" 的 categories 应为 IntervalDtype 类型
        assert isinstance(df["B"].cat.categories.dtype, IntervalDtype)
        # 列 "D" 应为 CategoricalDtype 类型
        assert isinstance(df["D"].dtype, CategoricalDtype)
        # 列 "D" 的 categories 应为 IntervalDtype 类型
        assert isinstance(df["D"].cat.categories.dtype, IntervalDtype)

        # 这些列经过 Series 构造函数处理后会被推断为 IntervalDtype 类型

        # 列 "C" 应为 IntervalDtype 类型
        assert isinstance(df["C"].dtype, IntervalDtype)
        # 列 "E" 应为 IntervalDtype 类型
        assert isinstance(df["E"].dtype, IntervalDtype)

        # 但 Series 构造函数不会对 Series 对象执行推断，
        # 因此设置 df["F"] 不会被转换为 IntervalDtype 类型
        assert is_object_dtype(df["F"])

        # 当转换为 numpy 对象后，它们作为 Index 可以相等比较

        # 定义一个函数 c 用于创建 Index 对象
        c = lambda x: Index(np.array(x))
        # 断言以下索引相等
        tm.assert_index_equal(c(df.B), c(df.B))
        tm.assert_index_equal(c(df.B), c(df.C), check_names=False)
        tm.assert_index_equal(c(df.B), c(df.D), check_names=False)
        tm.assert_index_equal(c(df.C), c(df.D), check_names=False)

        # 列 "B" 和 "D" 是相同的 Series 对象
        tm.assert_series_equal(df["B"], df["B"])
        tm.assert_series_equal(df["B"], df["D"], check_names=False)

        # 列 "C" 和 "E" 是相同的 Series 对象
        tm.assert_series_equal(df["C"], df["C"])
        tm.assert_series_equal(df["C"], df["E"], check_names=False)

    # 测试设置分类数据的情况
    def test_setitem_categorical(self):
        # 创建包含分类数据的 DataFrame 对象
        df = DataFrame({"h": Series(list("mn")).astype("category")})
        # 重新排序列 "h" 的分类顺序
        df.h = df.h.cat.reorder_categories(["n", "m"])
        # 创建期望的 DataFrame 对象，包含已排序的分类数据
        expected = DataFrame(
            {"h": Categorical(["m", "n"]).reorder_categories(["n", "m"])}
        )
        # 使用 assert_frame_equal 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)

    # 测试使用空的列表式赋值的情况
    def test_setitem_with_empty_listlike(self):
        # 创建一个空的索引对象
        index = Index([], name="idx")
        # 创建一个具有指定列名和空索引的 DataFrame 对象
        result = DataFrame(columns=["A"], index=index)
        # 将空列表赋值给列 "A"
        result["A"] = []
        # 创建期望的 DataFrame 对象，具有相同的列名和索引结构
        expected = DataFrame(columns=["A"], index=index)
        # 使用 assert_index_equal 断言 result 的索引与 expected 相等
        tm.assert_index_equal(result.index, expected.index)
    @pytest.mark.parametrize(
        "cols, values, expected",
        [
            (["C", "D", "D", "a"], [1, 2, 3, 4], 4),  # 使用pytest的参数化装饰器，定义了多组测试参数，包括列名、数值和预期结果
            (["D", "C", "D", "a"], [1, 2, 3, 4], 4),  # 测试参数包括重复列名的情况
            (["C", "B", "B", "a"], [1, 2, 3, 4], 4),  # 测试参数包括其他列名重复的情况
            (["C", "B", "a"], [1, 2, 3], 3),  # 测试参数没有重复列名的情况
            (["B", "C", "a"], [3, 2, 1], 1),  # 测试参数按字母顺序排列
            (["C", "a", "B"], [3, 2, 1], 2),  # 测试参数列名在中间位置
        ],
    )
    def test_setitem_same_column(self, cols, values, expected):
        # GH#23239
        # 创建一个DataFrame对象，使用给定的列名和数值
        df = DataFrame([values], columns=cols)
        # 将列'a'设置为自身，不会改变数据
        df["a"] = df["a"]
        # 获取结果并断言其与预期结果相等
        result = df["a"].values[0]
        assert result == expected

    def test_setitem_multi_index(self):
        # GH#7655, test that assigning to a sub-frame of a frame
        # with multi-index columns aligns both rows and columns
        # 定义一个多级索引的列，并创建一个DataFrame对象
        it = ["jim", "joe", "jolie"], ["first", "last"], ["left", "center", "right"]
        cols = MultiIndex.from_product(it)
        index = date_range("20141006", periods=20)
        vals = np.random.default_rng(2).integers(1, 1000, (len(index), len(cols)))
        df = DataFrame(vals, columns=cols, index=index)

        # 复制索引值和列名
        i, j = df.index.values.copy(), it[-1][:]

        # 混淆索引值i
        np.random.default_rng(2).shuffle(i)
        # 将列'jim'设置为'jolie'列的逆序
        df["jim"] = df["jolie"].loc[i, ::-1]
        # 断言DataFrame的'jim'列与'jolie'列相等
        tm.assert_frame_equal(df["jim"], df["jolie"])

        # 混淆列名j
        np.random.default_rng(2).shuffle(j)
        # 将('joe', 'first')列设置为('jolie', 'last')列的部分数据
        df[("joe", "first")] = df[("jolie", "last")].loc[i, j]
        # 断言DataFrame的('joe', 'first')列与('jolie', 'last')列相等
        tm.assert_frame_equal(df[("joe", "first")], df[("jolie", "last")])

        # 再次混淆列名j
        np.random.default_rng(2).shuffle(j)
        # 将('joe', 'last')列设置为('jolie', 'first')列的部分数据
        df[("joe", "last")] = df[("jolie", "first")].loc[i, j]
        # 断言DataFrame的('joe', 'last')列与('jolie', 'first')列相等
        tm.assert_frame_equal(df[("joe", "last")], df[("jolie", "first")])
    @pytest.mark.parametrize(
        "columns,box,expected",
        [  # 参数化测试的参数：列名列表、箱子值、期望的数据帧
            (
                ["A", "B", "C", "D"],  # 列名列表
                7,  # 箱子值
                DataFrame(  # 期望的数据帧
                    [[7, 7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7]],  # 数据帧的内容
                    columns=["A", "B", "C", "D"],  # 数据帧的列名
                ),
            ),
            (
                ["C", "D"],  # 列名列表
                [7, 8],  # 箱子值
                DataFrame(  # 期望的数据帧
                    [[1, 2, 7, 8], [3, 4, 7, 8], [5, 6, 7, 8]],  # 数据帧的内容
                    columns=["A", "B", "C", "D"],  # 数据帧的列名
                ),
            ),
            (
                ["A", "B", "C"],  # 列名列表
                np.array([7, 8, 9], dtype=np.int64),  # 箱子值
                DataFrame(  # 期望的数据帧
                    [[7, 8, 9], [7, 8, 9], [7, 8, 9]],  # 数据帧的内容
                    columns=["A", "B", "C"],  # 数据帧的列名
                ),
            ),
            (
                ["B", "C", "D"],  # 列名列表
                [[7, 8, 9], [10, 11, 12], [13, 14, 15]],  # 箱子值
                DataFrame(  # 期望的数据帧
                    [[1, 7, 8, 9], [3, 10, 11, 12], [5, 13, 14, 15]],  # 数据帧的内容
                    columns=["A", "B", "C", "D"],  # 数据帧的列名
                ),
            ),
            (
                ["C", "A", "D"],  # 列名列表
                np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.int64),  # 箱子值
                DataFrame(  # 期望的数据帧
                    [[8, 2, 7, 9], [11, 4, 10, 12], [14, 6, 13, 15]],  # 数据帧的内容
                    columns=["A", "B", "C", "D"],  # 数据帧的列名
                ),
            ),
            (
                ["A", "C"],  # 列名列表
                DataFrame([[7, 8], [9, 10], [11, 12]], columns=["A", "C"]),  # 箱子值
                DataFrame(  # 期望的数据帧
                    [[7, 2, 8], [9, 4, 10], [11, 6, 12]],  # 数据帧的内容
                    columns=["A", "B", "C"],  # 数据帧的列名
                ),
            ),
        ],
    )
    def test_setitem_list_missing_columns(self, columns, box, expected):
        # GH#29334
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=["A", "B"])
        df[columns] = box  # 在数据帧中使用列名列表进行赋值操作
        tm.assert_frame_equal(df, expected)  # 断言数据帧与期望的数据帧相等

    def test_setitem_list_of_tuples(self, float_frame):
        tuples = list(zip(float_frame["A"], float_frame["B"]))  # 创建由列'A'和列'B'组成的元组列表
        float_frame["tuples"] = tuples  # 在数据帧中创建名为'tuples'的列，并赋值为元组列表

        result = float_frame["tuples"]  # 获取名为'tuples'的列
        expected = Series(tuples, index=float_frame.index, name="tuples")  # 创建期望的序列对象
        tm.assert_series_equal(result, expected)  # 断言序列对象与期望的序列对象相等

    def test_setitem_iloc_generator(self):
        # GH#39614
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        indexer = (x for x in [1, 2])  # 创建生成器对象
        df.iloc[indexer] = 1  # 使用生成器对象在数据帧的iloc位置进行赋值操作
        expected = DataFrame({"a": [1, 1, 1], "b": [4, 1, 1]})  # 期望的数据帧
        tm.assert_frame_equal(df, expected)  # 断言数据帧与期望的数据帧相等

    def test_setitem_iloc_two_dimensional_generator(self):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        indexer = (x for x in [1, 2])  # 创建生成器对象
        df.iloc[indexer, 1] = 1  # 使用生成器对象在数据帧的iloc位置进行赋值操作
        expected = DataFrame({"a": [1, 2, 3], "b": [4, 1, 1]})  # 期望的数据帧
        tm.assert_frame_equal(df, expected)  # 断言数据帧与期望的数据帧相等
    # 测试设置 DataFrame 列的数据类型为字节类型转为对象类型
    def test_setitem_dtypes_bytes_type_to_object(self):
        # GH 20734
        # 创建一个名为 "id" 的 Series，数据类型为字节字符串 "S24"
        index = Series(name="id", dtype="S24")
        # 创建一个 DataFrame，使用上述 Series 作为索引
        df = DataFrame(index=index)
        # 设置列 "a"，其数据类型为无符号 32 位整数的 Series
        df["a"] = Series(name="a", index=index, dtype=np.uint32)
        # 设置列 "b" 和 "c"，它们的数据类型为长度为 64 的字节字符串
        df["b"] = Series(name="b", index=index, dtype="S64")
        df["c"] = Series(name="c", index=index, dtype="S64")
        # 设置列 "d"，其数据类型为无符号 8 位整数的 Series
        df["d"] = Series(name="d", index=index, dtype=np.uint8)
        # 获取 DataFrame 的列数据类型
        result = df.dtypes
        # 预期的列数据类型 Series，包含 np.uint32, object, object, np.uint8，对应列名为 "abcd"
        expected = Series([np.uint32, object, object, np.uint8], index=list("abcd"))
        # 断言 DataFrame 的列数据类型与预期的相等
        tm.assert_series_equal(result, expected)

    # 测试布尔掩码在可空 int64 上的应用
    def test_boolean_mask_nullable_int64(self):
        # GH 28928
        # 创建一个 DataFrame，包含两列 "a" 和 "b"，数据类型分别为 int64 和 Int64
        result = DataFrame({"a": [3, 4], "b": [5, 6]}).astype(
            {"a": "int64", "b": "Int64"}
        )
        # 创建一个布尔掩码 Series，与 result 的索引相同，所有值为 False
        mask = Series(False, index=result.index)
        # 使用布尔掩码更新 result 的列 "a" 和 "b"
        result.loc[mask, "a"] = result["a"]
        result.loc[mask, "b"] = result["b"]
        # 预期的 DataFrame，结构与 result 相同，数据类型为 int64 和 Int64
        expected = DataFrame({"a": [3, 4], "b": [5, 6]}).astype(
            {"a": "int64", "b": "Int64"}
        )
        # 断言 result 与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 测试设置 DataFrame 列的数据类型为 Series 的右侧对象
    def test_setitem_ea_dtype_rhs_series(self):
        # GH#47425
        # 创建一个 DataFrame，包含一列 "a"，数据为 [1, 2]
        df = DataFrame({"a": [1, 2]})
        # 将列 "a" 的数据类型设置为 Int64 的 Series，数据为 [1, 2]
        df["a"] = Series([1, 2], dtype="Int64")
        # 预期的 DataFrame，包含一列 "a"，数据类型为 Int64，数据为 [1, 2]
        expected = DataFrame({"a": [1, 2]}, dtype="Int64")
        # 断言 DataFrame df 与预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)

    # 测试设置 DataFrame 列的数据类型为 NumPy 矩阵的 2D 数组
    def test_setitem_npmatrix_2d(self):
        # GH#42376
        # 创建预期的 DataFrame，包含两列 "np-array" 和 "np-matrix"，索引为 [0, 1, ..., 9]
        expected = DataFrame(
            {"np-array": np.ones(10), "np-matrix": np.ones(10)}, index=np.arange(10)
        )

        # 创建一个形状为 (10, 1) 的 NumPy 数组 a
        a = np.ones((10, 1))
        # 创建一个索引为 [0, 1, ..., 9] 的 DataFrame
        df = DataFrame(index=np.arange(10))
        # 将数组 a 赋值给列 "np-array"
        df["np-array"] = a

        # 使用 NumPy 矩阵 a 给列 "np-matrix" 赋值
        with tm.assert_produces_warning(
            PendingDeprecationWarning,
            match="matrix subclass is not the recommended way to represent matrices",
        ):
            df["np-matrix"] = np.matrix(a)

        # 断言 DataFrame df 与预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)

    # 使用参数化测试，测试将字典与索引对齐并与 DataFrame 对象进行设置
    @pytest.mark.parametrize("vals", [{}, {"d": "a"}])
    def test_setitem_aligning_dict_with_index(self, vals):
        # GH#47216
        # 创建一个 DataFrame，包含列 "a" 和 "b"，以及额外的字典 vals
        df = DataFrame({"a": [1, 2], "b": [3, 4], **vals})
        # 使用字典 {1: 100, 0: 200} 更新列 "a"
        df.loc[:, "a"] = {1: 100, 0: 200}
        # 使用字典 {0: 5, 1: 6} 更新列 "c"
        df.loc[:, "c"] = {0: 5, 1: 6}
        # 使用字典 {1: 5} 更新列 "e"
        df.loc[:, "e"] = {1: 5}
        # 预期的 DataFrame，包含列 "a", "b", "c", "e"，数据与更新后的 df 相对应
        expected = DataFrame(
            {"a": [200, 100], "b": [3, 4], **vals, "c": [5, 6], "e": [np.nan, 5]}
        )
        # 断言 DataFrame df 与预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)

    # 测试将 DataFrame 列设置为另一个 DataFrame
    def test_setitem_rhs_dataframe(self):
        # GH#47578
        # 创建一个 DataFrame，包含一列 "a"，数据为 [1, 2]
        df = DataFrame({"a": [1, 2]})
        # 将列 "a" 的数据设置为另一个 DataFrame，该 DataFrame 包含一列 "a"，索引为 [1, 2]
        df["a"] = DataFrame({"a": [10, 11]}, index=[1, 2])
        # 预期的 DataFrame，包含一列 "a"，数据为 [nan, 10]
        expected = DataFrame({"a": [np.nan, 10]})
        # 断言 DataFrame df 与预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)

        # 再次创建一个 DataFrame，包含一列 "a"，数据为 [1, 2]
        df = DataFrame({"a": [1, 2]})
        # 使用 isetitem 方法，将列 "a" 的数据设置为另一个 DataFrame，该 DataFrame 包含一列 "a"，索引为 [1, 2]
        df.isetitem(0, DataFrame({"a": [10, 11]}, index=[1, 2]))
        # 断言 DataFrame df 与预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)
    # 测试函数：test_setitem_frame_overwrite_with_ea_dtype
    def test_setitem_frame_overwrite_with_ea_dtype(self, any_numeric_ea_dtype):
        # 创建一个具有两列数据的DataFrame对象
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        # 用另一个DataFrame对象覆盖df的"a"列，指定数据类型为any_numeric_ea_dtype
        df["a"] = DataFrame({"a": [10, 11]}, dtype=any_numeric_ea_dtype)
        # 期望的DataFrame对象，包含更新后的"a"列和未变更的"b"列
        expected = DataFrame(
            {
                "a": Series([10, 11], dtype=any_numeric_ea_dtype),
                "b": [2, 4],
            }
        )
        # 断言df和expected是否相等
        tm.assert_frame_equal(df, expected)

    # 测试函数：test_setitem_string_option_object_index
    def test_setitem_string_option_object_index(self):
        # 如果没有安装pyarrow模块，则跳过此测试
        pytest.importorskip("pyarrow")
        # 创建一个包含"a"列的DataFrame对象
        df = DataFrame({"a": [1, 2]})
        # 在上下文中设置选项"future.infer_string"为True，将Index对象赋给"b"列
        with pd.option_context("future.infer_string", True):
            df["b"] = Index(["a", "b"], dtype=object)
        # 期望的DataFrame对象，包含更新后的"b"列和未变更的"a"列
        expected = DataFrame({"a": [1, 2], "b": Series(["a", "b"], dtype=object)})
        # 断言df和expected是否相等
        tm.assert_frame_equal(df, expected)

    # 测试函数：test_setitem_frame_midx_columns
    def test_setitem_frame_midx_columns(self):
        # 创建一个包含多级列名的DataFrame对象
        df = DataFrame({("a", "b"): [10]})
        # 复制df对象到expected
        expected = df.copy()
        # 定义列名col_name为("a", "b")，将df的[[col_name]]的值更新到df的col_name列
        df[col_name] = df[[col_name]]
        # 断言df和expected是否相等
        tm.assert_frame_equal(df, expected)

    # 测试函数：test_loc_setitem_ea_dtype
    def test_loc_setitem_ea_dtype(self):
        # 创建一个包含"a"列的DataFrame对象，列数据类型为'i8'
        df = DataFrame({"a": np.array([10], dtype="i8")})
        # 使用.loc方法更新"a"列的值，数据类型为Int64
        df.loc[:, "a"] = Series([11], dtype="Int64")
        # 期望的DataFrame对象，更新后的"a"列数据类型保持不变
        expected = DataFrame({"a": np.array([11], dtype="i8")})
        # 断言df和expected是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个包含"a"列的DataFrame对象，列数据类型为'i8'
        df = DataFrame({"a": np.array([10], dtype="i8")})
        # 使用.iloc方法更新第一列的值，数据类型为Int64
        df.iloc[:, 0] = Series([11], dtype="Int64")
        # 断言df和expected是否相等
        tm.assert_frame_equal(df, expected)

    # 测试函数：test_setitem_index_object_dtype_not_inferring
    def test_setitem_index_object_dtype_not_inferring(self):
        # 创建一个Index对象，包含一个日期时间戳，数据类型为object
        idx = Index([Timestamp("2019-12-31")], dtype=object)
        # 创建一个包含"a"列的DataFrame对象
        df = DataFrame({"a": [1]})
        # 使用.loc方法更新"b"列和"c"列的值为idx对象
        df.loc[:, "b"] = idx
        df["c"] = idx

        # 期望的DataFrame对象，包含更新后的"b"列和"c"列，以及未变更的"a"列
        expected = DataFrame(
            {
                "a": [1],
                "b": idx,
                "c": idx,
            }
        )
        # 断言df和expected是否相等
        tm.assert_frame_equal(df, expected)
class TestSetitemTZAwareValues:
    @pytest.fixture
    def idx(self):
        # 创建一个包含两个日期时间字符串的 DatetimeIndex 对象，时区为 'US/Pacific'
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        idx = naive.tz_localize("US/Pacific")
        return idx

    @pytest.fixture
    def expected(self, idx):
        # 创建一个 Series 对象，包含与 idx 相同的日期时间对象，数据类型为 'object'
        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")
        assert expected.dtype == idx.dtype  # 确保 expected 的数据类型与 idx 相同
        return expected

    def test_setitem_dt64series(self, idx, expected):
        # 将 DataFrame 的 'B' 列设置为 idx，然后转换为 Series 对象并移除时区信息
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 1)), columns=["A"])
        df["B"] = idx
        df["B"] = idx.to_series(index=[0, 1]).dt.tz_convert(None)

        result = df["B"]
        # 创建一个 Series 对象，其数据是将 idx 转换为 UTC 时区后去除时区信息的结果
        comp = Series(idx.tz_convert("UTC").tz_localize(None), name="B")
        tm.assert_series_equal(result, comp)

    def test_setitem_datetimeindex(self, idx, expected):
        # 设置一个包含 tzaware 时间索引的 DataFrame 列，保持数据类型不变
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 1)), columns=["A"])

        # 将 idx 赋给 DataFrame 的 'B' 列
        df["B"] = idx
        result = df["B"]
        tm.assert_series_equal(result, expected)

    def test_setitem_object_array_of_tzaware_datetimes(self, idx, expected):
        # 设置一个包含 tzaware 时间对象数组的 DataFrame 列，保持数据类型不变
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 1)), columns=["A"])

        # 将 idx 转换为 Python datetime 对象数组，并将其设置为 DataFrame 的 'B' 列
        df["B"] = idx.to_pydatetime()
        result = df["B"]
        # 将期望结果的时间单位转换为微秒
        expected = expected.dt.as_unit("us")
        tm.assert_series_equal(result, expected)


class TestDataFrameSetItemWithExpansion:
    def test_setitem_listlike_views(self):
        # 测试 DataFrame 中列的视图语义，添加新列到 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [4, 4, 6]})

        # 获取 DataFrame 中 'a' 列的视图
        ser = df["a"]

        # 使用数组添加两列到 DataFrame
        df[["c", "d"]] = np.array([[0.1, 0.2], [0.3, 0.4], [0.4, 0.5]])

        # 直接在原地修改第一列，以检查视图语义
        df.iloc[0, 0] = 100

        expected = Series([1, 2, 3], name="a")
        tm.assert_series_equal(ser, expected)

    def test_setitem_string_column_numpy_dtype_raising(self):
        # 测试通过字符串设置 DataFrame 的列名，并引发对应的 NumPy 数据类型问题
        df = DataFrame([[1, 2], [3, 4]])

        # 设置 DataFrame 的 "0 - Name" 列
        df["0 - Name"] = [5, 6]

        expected = DataFrame([[1, 2, 5], [3, 4, 6]], columns=[0, 1, "0 - Name"])
        tm.assert_frame_equal(df, expected)

    def test_setitem_empty_df_duplicate_columns(self):
        # 测试在空 DataFrame 中设置具有重复列名的列
        df = DataFrame(columns=["a", "b", "b"], dtype="float64")

        # 在 DataFrame 的 "a" 列上设置值
        df.loc[:, "a"] = list(range(2))

        expected = DataFrame(
            [[0, np.nan, np.nan], [1, np.nan, np.nan]], columns=["a", "b", "b"]
        )
        tm.assert_frame_equal(df, expected)
    # 测试函数：设置带有扩展分类数据类型的项
    def test_setitem_with_expansion_categorical_dtype(self):
        # 创建包含随机整数的 DataFrame 对象
        df = DataFrame(
            {
                "value": np.array(
                    np.random.default_rng(2).integers(0, 10000, 100), dtype="int32"
                )
            }
        )
        # 创建分类数据类型的标签
        labels = Categorical([f"{i} - {i + 499}" for i in range(0, 10000, 500)])

        # 根据'value'列对 DataFrame 进行升序排序
        df = df.sort_values(by=["value"], ascending=True)
        # 利用cut函数对'value'列进行切分，并将结果存储在 Series 对象中
        ser = cut(df.value, range(0, 10500, 500), right=False, labels=labels)
        # 获取切分后的分类值
        cat = ser.values

        # 使用分类值设置 DataFrame 的'D'列
        df["D"] = cat
        # 检查 DataFrame 的列数据类型是否符合预期
        result = df.dtypes
        expected = Series(
            [np.dtype("int32"), CategoricalDtype(categories=labels, ordered=False)],
            index=["value", "D"],
        )
        tm.assert_series_equal(result, expected)

        # 使用 Series 对象设置 DataFrame 的'E'列
        df["E"] = ser
        # 再次检查 DataFrame 的列数据类型是否符合预期
        result = df.dtypes
        expected = Series(
            [
                np.dtype("int32"),
                CategoricalDtype(categories=labels, ordered=False),
                CategoricalDtype(categories=labels, ordered=False),
            ],
            index=["value", "D", "E"],
        )
        tm.assert_series_equal(result, expected)

        # 检查设置后的'D'列是否与预期的分类值相等
        result1 = df["D"]
        result2 = df["E"]
        tm.assert_categorical_equal(result1._mgr.array, cat)

        # 对'E'列进行排序，并与原始的 Series 对象进行比较
        ser.name = "E"
        tm.assert_series_equal(result2.sort_index(), ser.sort_index())

    # 测试函数：设置标量值到没有索引的列
    def test_setitem_scalars_no_index(self):
        # 创建一个空的 DataFrame 对象
        df = DataFrame()
        # 向列'foo'设置值为1
        df["foo"] = 1
        # 创建预期结果的 DataFrame 对象，并指定其列类型为np.int64
        expected = DataFrame(columns=["foo"]).astype(np.int64)
        tm.assert_frame_equal(df, expected)

    # 测试函数：使用元组作为新列的键
    def test_setitem_newcol_tuple_key(self, float_frame):
        # 检查("A", "B")是否不在 float_frame 的列中
        assert (
            "A",
            "B",
        ) not in float_frame.columns
        # 将'A'列的值赋给('A', 'B')列
        float_frame["A", "B"] = float_frame["A"]
        # 再次检查("A", "B")是否在 float_frame 的列中
        assert ("A", "B") in float_frame.columns

        # 获取('A', 'B')列的结果，并与'A'列进行比较，忽略列名检查
        result = float_frame["A", "B"]
        expected = float_frame["A"]
        tm.assert_series_equal(result, expected, check_names=False)

    # 测试函数：使用时间戳作为新列的键
    def test_frame_setitem_newcol_timestamp(self):
        # 创建日期范围为工作日的列
        columns = date_range(start="1/1/2012", end="2/1/2012", freq=BDay())
        # 创建 DataFrame 对象，指定列和索引
        data = DataFrame(columns=columns, index=range(10))
        t = datetime(2012, 11, 1)
        ts = Timestamp(t)
        # 将 NaN 值设置到时间戳对应的列中，进行简单的测试
        data[ts] = np.nan  # works, mostly a smoke-test
        # 检查是否所有值都是 NaN
        assert np.isnan(data[ts]).all()

    # 测试函数：使用 RangeIndex 作为新列的键
    def test_frame_setitem_rangeindex_into_new_col(self):
        # 创建包含'a'列的 DataFrame 对象
        df = DataFrame({"a": ["a", "b"]})
        # 将 RangeIndex 赋给'b'列
        df["b"] = df.index
        # 根据条件选择将100赋给'b'列的行，并与预期结果进行比较
        df.loc[[False, True], "b"] = 100
        result = df.loc[[1], :]
        expected = DataFrame({"a": ["b"], "b": [100]}, index=[1])
        tm.assert_frame_equal(result, expected)
    # 测试函数，验证在保持现有数据类型的情况下设置 DataFrame 的新列
    def test_setitem_frame_keep_ea_dtype(self, any_numeric_ea_dtype):
        # 创建一个具有两列的 DataFrame，包含两行数据
        df = DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        # 在 DataFrame 中添加一个新列 'c'，其数据来自另一个 DataFrame，并指定数据类型为 any_numeric_ea_dtype
        df["c"] = DataFrame({"a": [10, 11]}, dtype=any_numeric_ea_dtype)
        # 创建预期的 DataFrame，包含三列，其中第三列 'c' 是一个 Series 对象，数据类型与 any_numeric_ea_dtype 相同
        expected = DataFrame(
            {
                "a": [1, 3],
                "b": [2, 4],
                "c": Series([10, 11], dtype=any_numeric_ea_dtype),
            }
        )
        # 使用测试框架中的方法验证 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

    # 测试函数，验证使用时间增量类型（timedelta）在 loc 操作时的行为
    def test_loc_expansion_with_timedelta_type(self):
        # 创建一个空的 DataFrame，包含列 'a', 'b', 'c'
        result = DataFrame(columns=list("abc"))
        # 在索引为 0 的位置插入一行数据，其中包含时间增量类型的数据
        result.loc[0] = {
            "a": pd.to_timedelta(5, unit="s"),
            "b": pd.to_timedelta(72, unit="s"),
            "c": "23",
        }
        # 创建预期的 DataFrame，包含一行数据，其中第一列 'a' 和 'b' 是时间增量类型，第三列 'c' 是字符串类型
        expected = DataFrame(
            [[pd.Timedelta("0 days 00:00:05"), pd.Timedelta("0 days 00:01:12"), "23"]],
            index=Index([0]),  # 指定索引为 0
            columns=(["a", "b", "c"]),  # 指定列为 'a', 'b', 'c'
        )
        # 使用测试框架中的方法验证 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
class TestDataFrameSetItemSlicing:
    # 定义测试类 TestDataFrameSetItemSlicing，用于测试 DataFrame 的切片赋值操作

    def test_setitem_slice_position(self):
        # GH#31469
        # 创建一个包含 100 行、1 列的 DataFrame，所有元素初始化为 0
        df = DataFrame(np.zeros((100, 1)))
        
        # 将倒数第四行到最后一行的所有列赋值为 1
        df[-4:] = 1
        
        # 创建一个和 df 相同形状的数组，所有元素初始化为 0
        arr = np.zeros((100, 1))
        
        # 将 arr 中倒数第四行到最后一行的所有列赋值为 1
        arr[-4:] = 1
        
        # 期望的结果 DataFrame，与 arr 相同形状
        expected = DataFrame(arr)
        
        # 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("box", [Series, np.array, list, pd.array])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_setitem_slice_indexer_broadcasting_rhs(self, n, box, indexer_si):
        # GH#40440
        # 创建一个包含多行的 DataFrame，列名为 ["a", "b", "c"]，初始化值为 [[1, 3, 5], [2, 4, 6]] * n
        df = DataFrame([[1, 3, 5]] + [[2, 4, 6]] * n, columns=["a", "b", "c"])
        
        # 使用 indexer_si 返回的索引器，对 df 的第一行之后的所有行进行赋值，值为 box([10, 11, 12])
        indexer_si(df)[1:] = box([10, 11, 12])
        
        # 期望的结果 DataFrame，赋值后的第一行之后的所有行变为 [[1, 3, 5], [10, 11, 12]] * n
        expected = DataFrame([[1, 3, 5]] + [[10, 11, 12]] * n, columns=["a", "b", "c"])
        
        # 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("box", [Series, np.array, list, pd.array])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_setitem_list_indexer_broadcasting_rhs(self, n, box):
        # GH#40440
        # 创建一个包含多行的 DataFrame，列名为 ["a", "b", "c"]，初始化值为 [[1, 3, 5], [2, 4, 6]] * n
        df = DataFrame([[1, 3, 5]] + [[2, 4, 6]] * n, columns=["a", "b", "c"])
        
        # 使用 iloc 方法，对 df 的第 1 到 n 行进行赋值，值为 box([10, 11, 12])
        df.iloc[list(range(1, n + 1))] = box([10, 11, 12])
        
        # 期望的结果 DataFrame，赋值后的第 1 到 n 行变为 [[1, 3, 5], [10, 11, 12]] * n
        expected = DataFrame([[1, 3, 5]] + [[10, 11, 12]] * n, columns=["a", "b", "c"])
        
        # 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("box", [Series, np.array, list, pd.array])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_setitem_slice_broadcasting_rhs_mixed_dtypes(self, n, box, indexer_si):
        # GH#40440
        # 创建一个包含多行的 DataFrame，列名为 ["a", "b", "c"]，包含混合数据类型的值
        df = DataFrame(
            [[1, 3, 5], ["x", "y", "z"]] + [[2, 4, 6]] * n, columns=["a", "b", "c"]
        )
        
        # 使用 indexer_si 返回的索引器，对 df 的第一行之后的所有行进行赋值，值为 box([10, 11, 12])
        indexer_si(df)[1:] = box([10, 11, 12])
        
        # 期望的结果 DataFrame，赋值后的第一行之后的所有行变为 [[1, 3, 5]] + [[10, 11, 12]] * (n + 1)
        expected = DataFrame(
            [[1, 3, 5]] + [[10, 11, 12]] * (n + 1),
            columns=["a", "b", "c"],
            dtype="object",
        )
        
        # 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)


class TestDataFrameSetItemCallable:
    # 定义测试类 TestDataFrameSetItemCallable，用于测试 DataFrame 的可调用对象赋值操作

    def test_setitem_callable(self):
        # GH#12533
        # 创建一个包含两列的 DataFrame，列名为 "A" 和 "B"，初始化值为 [1, 2, 3, 4] 和 [5, 6, 7, 8]
        df = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
        
        # 使用 lambda 函数选择 "A" 列，并将其赋值为 [11, 12, 13, 14]
        df[lambda x: "A"] = [11, 12, 13, 14]
        
        # 期望的结果 DataFrame，"A" 列变为 [11, 12, 13, 14]
        exp = DataFrame({"A": [11, 12, 13, 14], "B": [5, 6, 7, 8]})
        
        # 断言 df 和 exp 相等
        tm.assert_frame_equal(df, exp)

    def test_setitem_other_callable(self):
        # GH#13299
        # 定义一个增加函数 inc
        def inc(x):
            return x + 1
        
        # 创建一个包含两行两列的 DataFrame，所有元素初始化为 -1，数据类型为 object
        df = DataFrame([[-1, 1], [1, -1]], dtype=object)
        
        # 使用布尔索引，将 df 大于 0 的元素赋值为 inc 函数
        df[df > 0] = inc
        
        # 期望的结果 DataFrame，将大于 0 的元素应用 inc 函数后
        expected = DataFrame([[-1, inc], [inc, -1]])
        
        # 断言 df 和 expected 相等
        tm.assert_frame_equal(df, expected)


class TestDataFrameSetItemBooleanMask:
    # 定义测试类 TestDataFrameSetItemBooleanMask，用于测试 DataFrame 的布尔掩码赋值操作

    @pytest.mark.parametrize(
        "mask_type",
        [lambda df: df > np.abs(df) / 2, lambda df: (df > np.abs(df) / 2).values],
        ids=["dataframe", "array"],
    )
    def test_setitem_boolean_mask(self, mask_type, float_frame):
        # Test for issue #18582
        # 复制浮点数据框以防止修改原始数据
        df = float_frame.copy()
        # 根据给定的掩码类型生成掩码
        mask = mask_type(df)

        # 使用布尔掩码对数据框进行索引
        result = df.copy()
        result[mask] = np.nan

        # 复制预期结果
        expected = df.values.copy()
        expected[np.array(mask)] = np.nan
        expected = DataFrame(expected, index=df.index, columns=df.columns)
        # 断言操作结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(reason="Currently empty indexers are treated as all False")
    @pytest.mark.parametrize("box", [list, np.array, Series])
    def test_setitem_loc_empty_indexer_raises_with_non_empty_value(self, box):
        # GH#37672
        # 创建包含字符串和整数的数据框
        df = DataFrame({"a": ["a"], "b": [1], "c": [1]})
        # 根据不同的箱子类型生成空索引器
        if box == Series:
            indexer = box([], dtype="object")
        else:
            indexer = box([])
        msg = "Must have equal len keys and value when setting with an iterable"
        # 使用空索引器设置非空值应引发值错误
        with pytest.raises(ValueError, match=msg):
            df.loc[indexer, ["b"]] = [1]

    @pytest.mark.parametrize("box", [list, np.array, Series])
    def test_setitem_loc_only_false_indexer_dtype_changed(self, box):
        # GH#37550
        # 创建包含字符串和整数的数据框
        df = DataFrame({"a": ["a"], "b": [1], "c": [1]})
        # 创建只包含 False 布尔值的索引器
        indexer = box([False])
        # 使用 loc 根据索引器设置 b 列的值
        df.loc[indexer, ["b"]] = 10 - df["c"]
        expected = DataFrame({"a": ["a"], "b": [1], "c": [1]})
        # 断言数据框与预期结果相等
        tm.assert_frame_equal(df, expected)

        # 再次使用 loc 设置 b 列的值
        df.loc[indexer, ["b"]] = 9
        # 断言数据框与预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_setitem_boolean_mask_aligning(self, indexer_sl):
        # GH#39931
        # 创建包含整数的数据框
        df = DataFrame({"a": [1, 4, 2, 3], "b": [5, 6, 7, 8]})
        expected = df.copy()
        # 创建布尔掩码
        mask = df["a"] >= 3
        # 根据掩码对数据框的子集进行排序
        indexer_sl(df)[mask] = indexer_sl(df)[mask].sort_values("a")
        # 断言数据框与预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_setitem_mask_categorical(self):
        # assign multiple rows (mixed values) (-> array) -> exp_multi_row
        # changed multiple rows
        # 创建包含分类数据的数据框
        cats2 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        idx2 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values2 = [1, 1, 2, 2, 1, 1, 1]
        exp_multi_row = DataFrame({"cats": cats2, "values": values2}, index=idx2)

        catsf = Categorical(
            ["a", "a", "c", "c", "a", "a", "a"], categories=["a", "b", "c"]
        )
        idxf = Index(["h", "i", "j", "k", "l", "m", "n"])
        valuesf = [1, 1, 3, 3, 1, 1, 1]
        # 创建包含分类数据的数据框
        df = DataFrame({"cats": catsf, "values": valuesf}, index=idxf)

        exp_fancy = exp_multi_row.copy()
        # 更新预期结果中的分类值
        exp_fancy["cats"] = exp_fancy["cats"].cat.set_categories(["a", "b", "c"])

        # 创建布尔掩码
        mask = df["cats"] == "c"
        # 根据掩码更新数据框的值
        df[mask] = ["b", 2]
        # 断言数据框与预期结果相等
        tm.assert_frame_equal(df, exp_fancy)

    @pytest.mark.parametrize("dtype", ["float", "int64"])
    # 使用 pytest 的参数化功能，为以下测试用例提供多组参数
    @pytest.mark.parametrize("kwargs", [{}, {"index": [1]}, {"columns": ["A"]}])
    def test_setitem_empty_frame_with_boolean(self, dtype, kwargs):
        # 解释：设置参数字典中的 dtype 键值对
        kwargs["dtype"] = dtype
        # 创建一个 DataFrame 对象，使用传入的参数
        df = DataFrame(**kwargs)

        # 复制 DataFrame 对象，生成一个新的副本
        df2 = df.copy()
        # 根据条件设置 DataFrame 中符合条件的值为 47
        df[df > df2] = 47
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, df2)

    def test_setitem_boolean_indexing(self):
        # 创建一个索引列表和列名列表
        idx = list(range(3))
        cols = ["A", "B", "C"]
        # 创建第一个 DataFrame 对象，包含指定的索引、列名和数据
        df1 = DataFrame(
            index=idx,
            columns=cols,
            data=np.array(
                [[0.0, 0.5, 1.0], [1.5, 2.0, 2.5], [3.0, 3.5, 4.0]], dtype=float
            ),
        )
        # 创建第二个 DataFrame 对象，包含指定的索引和列名，数据全为 1
        df2 = DataFrame(index=idx, columns=cols, data=np.ones((len(idx), len(cols))))

        # 创建预期的 DataFrame 对象，根据条件设置符合条件的值为 -1
        expected = DataFrame(
            index=idx,
            columns=cols,
            data=np.array([[0.0, 0.5, 1.0], [1.5, 2.0, -1], [-1, -1, -1]], dtype=float),
        )

        # 根据条件设置 df1 中符合条件的值为 -1
        df1[df1 > 2.0 * df2] = -1
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df1, expected)

        # 使用 pytest 的断言检查是否抛出 ValueError 异常，匹配给定的错误信息
        with pytest.raises(ValueError, match="Item wrong length"):
            # 根据条件设置 df1 中索引除最后一个外符合条件的值为 -1
            df1[df1.index[:-1] > 2] = -1

    def test_loc_setitem_all_false_boolean_two_blocks(self):
        # GH#40885
        # 创建一个包含指定列的 DataFrame 对象
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": "a"})
        # 创建预期结果的 DataFrame 对象，复制原始 df
        expected = df.copy()
        # 创建一个 Series 对象，用于作为 loc 方法的索引器
        indexer = Series([False, False], name="c")
        # 使用 loc 方法根据索引器和指定的列名设置值
        df.loc[indexer, ["b"]] = DataFrame({"b": [5, 6]}, index=[0, 1])
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)

    def test_setitem_ea_boolean_mask(self):
        # GH#47125
        # 创建一个包含指定数据的 DataFrame 对象
        df = DataFrame([[-1, 2], [3, -4]])
        # 创建预期结果的 DataFrame 对象，数据根据条件设置为 0
        expected = DataFrame([[0, 2], [3, 0]])
        # 创建一个布尔索引器的 DataFrame 对象，用于根据条件设置 df 中的值为 0
        boolean_indexer = DataFrame(
            {
                0: Series([True, False], dtype="boolean"),
                1: Series([pd.NA, True], dtype="boolean"),
            }
        )
        # 根据布尔索引器设置 df 中符合条件的值为 0
        df[boolean_indexer] = 0
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)
    # 定义一个测试类，用于测试DataFrame的赋值语义
    class TestDataFrameSetitemCopyViewSemantics:
        
        # 测试赋值操作是否总是复制
        def test_setitem_always_copy(self, float_frame):
            # 断言列"A"不在float_frame的列中
            assert "E" not in float_frame.columns
            # 复制float_frame中"A"列的数据到变量s
            s = float_frame["A"].copy()
            # 将复制的数据s赋值给float_frame的"E"列
            float_frame["E"] = s
            
            # 将float_frame中"E"列第5到第10行的值设置为NaN
            float_frame.iloc[5:10, float_frame.columns.get_loc("E")] = np.nan
            # 断言s的第5到第10行没有全部为NaN
            assert notna(s[5:10]).all()

        @pytest.mark.parametrize("consolidate", [True, False])
        # 测试部分列的就地赋值操作
        def test_setitem_partial_column_inplace(self, consolidate):
            # 这个设置应该是就地进行的，无论frame是单块还是多块
            # GH#304 以前这个设置不正确地不是就地的，在这种情况下我们需要确保清除_item_cache。
            
            # 创建一个DataFrame对象df，包含三列"x", "y", "z"，其中"z"列初始为NaN
            df = DataFrame(
                {"x": [1.1, 2.1, 3.1, 4.1], "y": [5.1, 6.1, 7.1, 8.1]}, index=[0, 1, 2, 3]
            )
            # 在第2列位置插入一列"z"，初始值为NaN
            df.insert(2, "z", np.nan)
            if consolidate:
                # 如果consolidate为True，则就地整理df
                df._consolidate_inplace()
                # 断言df内部数据块的数量为1
                assert len(df._mgr.blocks) == 1
            else:
                # 否则，断言df内部数据块的数量为2
                assert len(df._mgr.blocks) == 2

            # 将df中第2行及以后的"z"列赋值为42
            df.loc[2:, "z"] = 42

            # 创建一个期望的Series对象，包含与df索引相同的四个元素，名为"z"，前两个元素为NaN，后两个为42
            expected = Series([np.nan, np.nan, 42, 42], index=df.index, name="z")
            # 断言df的"z"列与期望的Series对象相等
            tm.assert_series_equal(df["z"], expected)

        # 测试重复列名的赋值操作是否不是就地的
        def test_setitem_duplicate_columns_not_inplace(self):
            # GH#39510
            # 创建一个DataFrame对象df，包含一行全为0.0的两列"A"和"B"，每列各重复两次
            cols = ["A", "B"] * 2
            df = DataFrame(0.0, index=[0], columns=cols)
            # 复制df对象到df_copy
            df_copy = df.copy()
            # 创建一个df的视图df_view
            df_view = df[:]
            # 将"B"列的值设置为(2, 5)
            df["B"] = (2, 5)

            # 创建一个期望的DataFrame对象，包含一行[[0.0, 2, 0.0, 5]]，列名为cols
            expected = DataFrame([[0.0, 2, 0.0, 5]], columns=cols)
            # 断言df_view与df_copy相等
            tm.assert_frame_equal(df_view, df_copy)
            # 断言df与期望的DataFrame对象相等
            tm.assert_frame_equal(df, expected)

        @pytest.mark.parametrize(
            "value", [1, np.array([[1], [1]], dtype="int64"), [[1], [1]]]
        )
        # 测试赋值相同数据类型的操作是否不是就地的
        def test_setitem_same_dtype_not_inplace(self, value):
            # GH#39510
            # 创建一个DataFrame对象df，包含两行全为0的"A"和"B"列
            cols = ["A", "B"]
            df = DataFrame(0, index=[0, 1], columns=cols)
            # 复制df对象到df_copy
            df_copy = df.copy()
            # 创建一个df的视图df_view
            df_view = df[:]
            # 将"B"列的值设置为value
            df[["B"]] = value

            # 创建一个期望的DataFrame对象，包含两行[[0, 1], [0, 1]]，列名为cols
            expected = DataFrame([[0, 1], [0, 1]], columns=cols)
            # 断言df与期望的DataFrame对象相等
            tm.assert_frame_equal(df, expected)
            # 断言df_view与df_copy相等
            tm.assert_frame_equal(df_view, df_copy)

        @pytest.mark.parametrize("value", [1.0, np.array([[1.0], [1.0]]), [[1.0], [1.0]]])
        # 测试赋值标量值给列表类型键是否不是就地的
        def test_setitem_listlike_key_scalar_value_not_inplace(self, value):
            # GH#39510
            # 创建一个DataFrame对象df，包含两行全为0的"A"和"B"列
            cols = ["A", "B"]
            df = DataFrame(0, index=[0, 1], columns=cols)
            # 复制df对象到df_copy
            df_copy = df.copy()
            # 创建一个df的视图df_view
            df_view = df[:]
            # 将"B"列的值设置为value
            df[["B"]] = value

            # 创建一个期望的DataFrame对象，包含两行[[0, 1.0], [0, 1.0]]，列名为cols
            expected = DataFrame([[0, 1.0], [0, 1.0]], columns=cols)
            # 断言df_view与df_copy相等
            tm.assert_frame_equal(df_view, df_copy)
            # 断言df与期望的DataFrame对象相等
            tm.assert_frame_equal(df, expected)

        @pytest.mark.parametrize(
            "indexer",
            [
                "a",
                ["a"],
                pytest.param(
                    [True, False],
                    marks=pytest.mark.xfail(
                        reason="Boolean indexer incorrectly setting inplace",
                        strict=False,  # passing on some builds, no obvious pattern
                    ),
                ),
            ],
        )
    @pytest.mark.parametrize(
        "value, set_value",
        [  # 定义测试参数，包括整数、浮点数、时间戳和字符串
            (1, 5),
            (1.0, 5.0),
            (Timestamp("2020-12-31"), Timestamp("2021-12-31")),
            ("a", "b"),
        ],
    )
    def test_setitem_not_operating_inplace(self, value, set_value, indexer):
        # GH#43406
        # 创建一个 DataFrame，包含单列数据，并设置索引
        df = DataFrame({"a": value}, index=[0, 1])
        # 复制预期结果 DataFrame
        expected = df.copy()
        # 创建 DataFrame 的视图
        view = df[:]
        # 使用索引器设置 DataFrame 中的值
        df[indexer] = set_value
        # 断言视图与预期结果的 DataFrame 相等
        tm.assert_frame_equal(view, expected)

    def test_setitem_column_update_inplace(self):
        # https://github.com/pandas-dev/pandas/issues/47172

        # 创建一个 DataFrame，其中列名由 "c0" 到 "c9"
        labels = [f"c{i}" for i in range(10)]
        df = DataFrame({col: np.zeros(len(labels)) for col in labels}, index=labels)
        # 获取 DataFrame 内部的数据块值
        values = df._mgr.blocks[0].values

        # 使用链式赋值错误捕获上下文
        with tm.raises_chained_assignment_error():
            # 遍历 DataFrame 的列，并在每个列的标签处设置值为 1
            for label in df.columns:
                df[label][label] = 1
        # 断言原始 DataFrame 没有更新
        assert np.all(values[np.arange(10), np.arange(10)] == 0)

    def test_setitem_column_frame_as_category(self):
        # GH31581
        # 创建一个包含整数的 DataFrame
        df = DataFrame([1, 2, 3])
        # 将第一列设为类别类型的 DataFrame
        df["col1"] = DataFrame([1, 2, 3], dtype="category")
        # 将第二列设为类别类型的 Series
        df["col2"] = Series([1, 2, 3], dtype="category")

        # 创建预期结果的数据类型 Series
        expected_types = Series(
            ["int64", "category", "category"], index=[0, "col1", "col2"], dtype=object
        )
        # 断言 DataFrame 的数据类型与预期结果相等
        tm.assert_series_equal(df.dtypes, expected_types)

    @pytest.mark.parametrize("dtype", ["int64", "Int64"])
    def test_setitem_iloc_with_numpy_array(self, dtype):
        # GH-33828
        # 创建一个包含整数 1 的 DataFrame，数据类型为 int64 或 Int64
        df = DataFrame({"a": np.ones(3)}, dtype=dtype)
        # 使用 numpy 数组设置 DataFrame 的 iloc
        df.iloc[np.array([0]), np.array([0])] = np.array([[2]])

        # 创建预期结果的 DataFrame
        expected = DataFrame({"a": [2, 1, 1]}, dtype=dtype)
        # 断言 DataFrame 与预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_dup_cols_dtype(self):
        # GH#53143
        # 创建一个包含重复列的 DataFrame
        df = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=["a", "b", "a", "c"])
        # 创建一个包含重复列的右手边 DataFrame
        rhs = DataFrame([[0, 1.5], [2, 2.5]], columns=["a", "a"])
        # 将左手边 DataFrame 的 "a" 列设为右手边 DataFrame
        df["a"] = rhs
        # 创建预期结果的 DataFrame
        expected = DataFrame(
            [[0, 2, 1.5, 4], [2, 5, 2.5, 7]], columns=["a", "b", "a", "c"]
        )
        # 断言 DataFrame 与预期结果相等
        tm.assert_frame_equal(df, expected)

        # 创建一个包含重复列的 DataFrame
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
        # 创建一个包含重复列的右手边 DataFrame
        rhs = DataFrame([[0, 1.5], [2, 2.5]], columns=["a", "a"])
        # 将左手边 DataFrame 的 "a" 列设为右手边 DataFrame
        df["a"] = rhs
        # 创建预期结果的 DataFrame
        expected = DataFrame([[0, 1.5, 3], [2, 2.5, 6]], columns=["a", "a", "b"])
        # 断言 DataFrame 与预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_frame_setitem_empty_dataframe(self):
        # GH#28871
        # 创建一个时间索引对象
        dti = DatetimeIndex(["2000-01-01"], dtype="M8[ns]", name="date")
        # 创建一个 DataFrame，以日期为列名，并设置索引为日期
        df = DataFrame({"date": dti}).set_index("date")
        # 复制一个空的 DataFrame
        df = df[0:0].copy()

        # 设置列名为 "3010" 的值为 None
        df["3010"] = None
        # 设置列名为 "2010" 的值为 None
        df["2010"] = None

        # 创建预期结果的空 DataFrame
        expected = DataFrame(
            [],
            columns=["3010", "2010"],
            index=dti[:0],
        )
        # 断言 DataFrame 与预期结果相等
        tm.assert_frame_equal(df, expected)
def test_full_setter_loc_incompatible_dtype():
    # https://github.com/pandas-dev/pandas/issues/55791
    # 创建一个包含列'a'的DataFrame，初始值为[1, 2]
    df = DataFrame({"a": [1, 2]})
    # 断言设置值时会产生FutureWarning警告，警告内容包含"incompatible dtype"
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 将DataFrame列'a'的所有元素设置为True
        df.loc[:, "a"] = True
    # 期望的DataFrame，列'a'中所有元素变为True
    expected = DataFrame({"a": [True, True]})
    # 断言df与期望的DataFrame相等
    tm.assert_frame_equal(df, expected)

    # 创建一个包含列'a'的DataFrame，初始值为[1, 2]
    df = DataFrame({"a": [1, 2]})
    # 断言设置值时会产生FutureWarning警告，警告内容包含"incompatible dtype"
    with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
        # 将DataFrame列'a'的所有元素设置为字典{0: 3.5, 1: 4.5}
        df.loc[:, "a"] = {0: 3.5, 1: 4.5}
    # 期望的DataFrame，列'a'中的元素变为[3.5, 4.5]
    expected = DataFrame({"a": [3.5, 4.5]})
    # 断言df与期望的DataFrame相等
    tm.assert_frame_equal(df, expected)

    # 创建一个包含列'a'的DataFrame，初始值为[1, 2]
    df = DataFrame({"a": [1, 2]})
    # 将DataFrame列'a'的元素根据字典{0: 3, 1: 4}进行设置
    df.loc[:, "a"] = {0: 3, 1: 4}
    # 期望的DataFrame，列'a'中的元素变为[3, 4]
    expected = DataFrame({"a": [3, 4]})
    # 断言df与期望的DataFrame相等
    tm.assert_frame_equal(df, expected)


def test_setitem_partial_row_multiple_columns():
    # https://github.com/pandas-dev/pandas/issues/56503
    # 创建一个包含列'A'和'B'的DataFrame，初始值为[1, 2, 3]和[4.0, 5, 6]
    df = DataFrame({"A": [1, 2, 3], "B": [4.0, 5, 6]})
    # 不应该产生警告
    # 将满足条件 df.index <= 1 的行，列['F', 'G'] 设置为(1, 'abc')
    df.loc[df.index <= 1, ["F", "G"]] = (1, "abc")
    # 期望的DataFrame，包含列'A', 'B', 'F', 'G'，其中满足条件的行被设置，未设置的列值为NaN
    expected = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4.0, 5, 6],
            "F": [1.0, 1, float("nan")],
            "G": ["abc", "abc", float("nan")],
        }
    )
    # 断言df与期望的DataFrame相等
    tm.assert_frame_equal(df, expected)
```