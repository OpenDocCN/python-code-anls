# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_setitem.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 PyTest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据分析和处理
from pandas import (  # 从 Pandas 库中导入多个模块和函数
    DataFrame,  # 用于创建和操作二维数据表
    MultiIndex,  # 用于多级索引对象
    Series,  # 用于操作一维标签数组，支持任意数据类型
    date_range,  # 用于生成日期范围
    isna,  # 检查缺失值（NaN）
    notna,  # 检查非缺失值
)
import pandas._testing as tm  # 导入 Pandas 测试工具模块

# 定义一个函数，用于比较两个值是否相等，不等则抛出异常
def assert_equal(a, b):
    assert a == b

# 定义一个测试类 TestMultiIndexSetItem
class TestMultiIndexSetItem:
    # 定义一个方法 check，用于设置 DataFrame 的元素，并验证设置后的结果
    def check(self, target, indexers, value, compare_fn=assert_equal, expected=None):
        target.loc[indexers] = value  # 设置目标 DataFrame 的指定位置为给定值
        result = target.loc[indexers]  # 获取设置后的结果
        if expected is None:
            expected = value
        compare_fn(result, expected)  # 比较结果与期望值

    # 定义一个测试方法，测试在多级索引 DataFrame 中设置元素
    def test_setitem_multiindex(self):
        # GH#7190
        cols = ["A", "w", "l", "a", "x", "X", "d", "profit"]  # 列名列表
        index = MultiIndex.from_product(  # 使用 MultiIndex 的方法生成多级索引对象
            [np.arange(0, 100), np.arange(0, 80)], names=["time", "firm"]
        )
        t, n = 0, 2  # 定义 t 和 n 的值

        # 创建一个值为 NaN 的 DataFrame，指定列名和索引
        df = DataFrame(
            np.nan,
            columns=cols,
            index=index,
        )
        self.check(target=df, indexers=((t, n), "X"), value=0)  # 测试设置值为 0

        # 创建一个值为 -999 的 DataFrame，指定列名和索引
        df = DataFrame(-999, columns=cols, index=index)
        self.check(target=df, indexers=((t, n), "X"), value=1)  # 测试设置值为 1

        # 创建一个空的 DataFrame，指定列名和索引
        df = DataFrame(columns=cols, index=index)
        self.check(target=df, indexers=((t, n), "X"), value=2)  # 测试设置值为 2

        # gh-7218: 使用 0 维数组进行赋值
        df = DataFrame(-999, columns=cols, index=index)
        self.check(
            target=df,
            indexers=((t, n), "X"),
            value=np.array(3),
            expected=3,
        )  # 测试使用 0 维数组赋值为 3

    # 定义另一个测试方法，测试在 DataFrame 中设置元素
    def test_setitem_multiindex2(self):
        # GH#5206
        df = DataFrame(  # 创建一个 5x5 的 DataFrame，填充值为 0 到 24
            np.arange(25).reshape(5, 5), columns="A,B,C,D,E".split(","), dtype=float
        )
        df["F"] = 99  # 新增一列 "F"，填充值为 99
        row_selection = df["A"] % 2 == 0  # 选择满足条件的行
        col_selection = ["B", "C"]  # 选择的列
        df.loc[row_selection, col_selection] = df["F"]  # 在选定的行和列上设置值为 "F"
        output = DataFrame(99.0, index=[0, 2, 4], columns=["B", "C"])  # 期望的输出 DataFrame
        tm.assert_frame_equal(df.loc[row_selection, col_selection], output)  # 比较结果与期望值
        self.check(
            target=df,
            indexers=(row_selection, col_selection),
            value=df["F"],
            compare_fn=tm.assert_frame_equal,
            expected=output,
        )  # 使用自定义的比较函数比较结果与期望值
    def test_setitem_multiindex3(self):
        # GH#11372
        # 创建一个多级索引，行索引包含 ["A", "B", "C"] 和 2015年1月到4月每月第一个工作日的日期
        idx = MultiIndex.from_product(
            [["A", "B", "C"], date_range("2015-01-01", "2015-04-01", freq="MS")]
        )
        # 创建一个多级索引，列索引包含 ["foo", "bar"] 和 2016年1月到2月每月第一个工作日的日期
        cols = MultiIndex.from_product(
            [["foo", "bar"], date_range("2016-01-01", "2016-02-01", freq="MS")]
        )

        # 创建一个12行4列的DataFrame，填充随机数，行索引为idx，列索引为cols
        df = DataFrame(
            np.random.default_rng(2).random((12, 4)), index=idx, columns=cols
        )

        # 创建一个子级索引，行索引包含 ["A", "A"] 和 2015年1月到2月每月第一个工作日的日期
        subidx = MultiIndex.from_arrays(
            [["A", "A"], date_range("2015-01-01", "2015-02-01", freq="MS")]
        )
        # 创建一个子级索引，列索引包含 ["foo", "foo"] 和 2016年1月到2月每月第一个工作日的日期
        subcols = MultiIndex.from_arrays(
            [["foo", "foo"], date_range("2016-01-01", "2016-02-01", freq="MS")]
        )

        # 创建一个2行2列的DataFrame，填充随机数，行索引为subidx，列索引为subcols
        vals = DataFrame(
            np.random.default_rng(2).random((2, 2)), index=subidx, columns=subcols
        )
        
        # 使用自定义的检查函数检查DataFrame设置操作的结果是否正确
        self.check(
            target=df,
            indexers=(subidx, subcols),
            value=vals,
            compare_fn=tm.assert_frame_equal,
        )

        # 替换所有列
        vals = DataFrame(
            np.random.default_rng(2).random((2, 4)), index=subidx, columns=cols
        )
        self.check(
            target=df,
            indexers=(subidx, slice(None, None, None)),
            value=vals,
            compare_fn=tm.assert_frame_equal,
        )

        # 复制DataFrame对象
        copy = df.copy()
        self.check(
            target=df,
            indexers=(df.index, df.columns),
            value=df,
            compare_fn=tm.assert_frame_equal,
            expected=copy,
        )

    def test_multiindex_setitem(self):
        # GH 3738
        # 使用多级索引设置右侧操作
        arrays = [
            np.array(["bar", "bar", "baz", "qux", "qux", "bar"]),
            np.array(["one", "two", "one", "one", "two", "one"]),
            np.arange(0, 6, 1),
        ]

        # 创建一个包含随机数的DataFrame，行索引为arrays，列索引为 ["A", "B", "C"]，并按索引排序
        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((6, 3)),
            index=arrays,
            columns=["A", "B", "C"],
        ).sort_index()

        # 预期的DataFrame是df_orig中索引为["bar"]的行的值乘以2
        expected = df_orig.loc[["bar"]] * 2
        df = df_orig.copy()
        df.loc[["bar"]] *= 2
        # 使用测试模块中的函数检查df中索引为["bar"]的行是否与预期相等
        tm.assert_frame_equal(df.loc[["bar"]], expected)

        # 由于这些具有不同的级别，会触发错误
        msg = "cannot align on a multi-index with out specifying the join levels"
        # 使用pytest检查是否引发了TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            df.loc["bar"] *= 2
    def test_multiindex_setitem2(self):
        # 从 Stack Overflow 上的问题中获取的测试用例
        # https://stackoverflow.com/questions/24572040/pandas-access-the-level-of-multiindex-for-inplace-operation
        
        # 创建一个 DataFrame，包含多级索引，表示不同国家、能源类型和指标
        df_orig = DataFrame.from_dict(
            {
                "price": {
                    ("DE", "Coal", "Stock"): 2,
                    ("DE", "Gas", "Stock"): 4,
                    ("DE", "Elec", "Demand"): 1,
                    ("FR", "Gas", "Stock"): 5,
                    ("FR", "Solar", "SupIm"): 0,
                    ("FR", "Wind", "SupIm"): 0,
                }
            }
        )
        # 将索引设置为 MultiIndex，指定索引名称为 ["Sit", "Com", "Type"]
        df_orig.index = MultiIndex.from_tuples(
            df_orig.index, names=["Sit", "Com", "Type"]
        )

        # 复制原始 DataFrame 作为预期结果
        expected = df_orig.copy()
        # 选择部分行并对其值进行倍增
        expected.iloc[[0, 1, 3]] *= 2

        # 使用 pd.IndexSlice 创建一个索引切片对象
        idx = pd.IndexSlice
        # 复制原始 DataFrame
        df = df_orig.copy()
        # 使用 loc 和多级索引进行选择，并对所有列进行倍增操作
        df.loc[idx[:, :, "Stock"], :] *= 2
        # 断言操作后的 DataFrame 等于预期结果
        tm.assert_frame_equal(df, expected)

        # 复制原始 DataFrame
        df = df_orig.copy()
        # 使用 loc 和多级索引选择特定列 "price"，并对其进行倍增操作
        df.loc[idx[:, :, "Stock"], "price"] *= 2
        # 断言操作后的 DataFrame 等于预期结果
        tm.assert_frame_equal(df, expected)

    def test_multiindex_assignment(self):
        # GH3777 part 2

        # 混合数据类型的 DataFrame 创建
        df = DataFrame(
            np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3),
            columns=list("abc"),
            index=[[4, 4, 8], [8, 10, 12]],
        )
        # 新增一列 "d"，并赋值为 NaN
        df["d"] = np.nan
        # 创建一个数组 arr
        arr = np.array([0.0, 1.0])

        # 使用 loc 对索引值为 4 的行的 "d" 列进行赋值
        df.loc[4, "d"] = arr
        # 断言操作后的 Series 等于预期结果，Series 的索引为 [8, 10]，名称为 "d"
        tm.assert_series_equal(df.loc[4, "d"], Series(arr, index=[8, 10], name="d"))
    def test_multiindex_assignment_single_dtype(self):
        # GH3777 part 2b
        # single dtype
        # 创建一个包含两个元素的 numpy 数组，元素为浮点数 0.0 和 1.0
        arr = np.array([0.0, 1.0])

        # 创建一个 3x3 的 DataFrame：
        # - 使用 np.random.default_rng(2) 生成随机整数填充
        # - 列名为 "abc"
        # - 行索引为多重索引 [4, 4, 8], [8, 10, 12]
        # - 数据类型为 np.int64
        df = DataFrame(
            np.random.default_rng(2).integers(5, 10, size=9).reshape(3, 3),
            columns=list("abc"),
            index=[[4, 4, 8], [8, 10, 12]],
            dtype=np.int64,
        )

        # 将 arr 赋值给 df 的 loc[4, "c"]，因为 arr 可以无损地转换为整数
        df.loc[4, "c"] = arr
        # 创建一个期望的 Series，包含 arr 的值，索引为 [8, 10]，数据类型为 int64
        exp = Series(arr, index=[8, 10], name="c", dtype="int64")
        # 获取 df.loc[4, "c"] 的结果
        result = df.loc[4, "c"]
        # 断言 result 与期望的 exp 相等
        tm.assert_series_equal(result, exp)

        # arr + 0.5 不能无损地转换为整数，因此会进行类型提升
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            # 将 arr + 0.5 赋值给 df 的 loc[4, "c"]
            df.loc[4, "c"] = arr + 0.5
        # 获取 df.loc[4, "c"] 的结果
        result = df.loc[4, "c"]
        # 更新期望的 exp，将其每个元素加上 0.5
        exp = exp + 0.5
        # 断言 result 与更新后的 exp 相等
        tm.assert_series_equal(result, exp)

        # 将标量 10 赋值给 df 的 loc[4, "c"]
        df.loc[4, "c"] = 10
        # 创建一个期望的 Series，值为 10，索引为 [8, 10]，数据类型为 float64
        exp = Series(10, index=[8, 10], name="c", dtype="float64")
        # 断言 df.loc[4, "c"] 与期望的 exp 相等
        tm.assert_series_equal(df.loc[4, "c"], exp)

        # 无效的赋值操作，期望引发 ValueError 异常，错误信息为 "Must have equal len keys and value when setting with an iterable"
        with pytest.raises(ValueError, match=msg):
            df.loc[4, "c"] = [0, 1, 2, 3]

        with pytest.raises(ValueError, match=msg):
            df.loc[4, "c"] = [0]

        # 使用长度为 1 的列表索引器，赋值行为等同于 df.loc[4, "c"] = 0
        df.loc[4, ["c"]] = [0]
        # 断言 df.loc[4, "c"] 的所有值都为 0
        assert (df.loc[4, "c"] == 0).all()

    def test_groupby_example(self):
        # groupby example
        # 定义常量 NUM_ROWS 和 NUM_COLS
        NUM_ROWS = 100
        NUM_COLS = 10
        # 创建列名列表 col_names，以 "A0" 到 "A9" 命名
        col_names = ["A" + num for num in map(str, np.arange(NUM_COLS).tolist())]
        # 索引列为 col_names 的前五个元素
        index_cols = col_names[:5]

        # 创建一个 NUM_ROWS x NUM_COLS 的 DataFrame：
        # - 使用 np.random.default_rng(2) 生成 5 到 9 之间的随机整数填充
        # - 数据类型为 np.int64
        # - 列名为 col_names
        df = DataFrame(
            np.random.default_rng(2).integers(5, size=(NUM_ROWS, NUM_COLS)),
            dtype=np.int64,
            columns=col_names,
        )
        # 将 index_cols 设置为索引，并按索引排序
        df = df.set_index(index_cols).sort_index()
        # 对 df 进行按前四个索引级别分组
        grp = df.groupby(level=index_cols[:4])
        # 在 df 中创建一个名为 "new_col" 的新列，初始值为 NaN
        df["new_col"] = np.nan

        # 遍历 grp 中的每个分组（name, df2）
        for name, df2 in grp:
            # 创建一个新的值数组，长度为 df2 的行数
            new_vals = np.arange(df2.shape[0])
            # 将 new_vals 赋值给 df 中的 loc[name, "new_col"]
            df.loc[name, "new_col"] = new_vals

    def test_series_setitem(self, multiindex_year_month_day_dataframe_random_data):
        # 获取 multiindex_year_month_day_dataframe_random_data 中的 "A" 列
        ymd = multiindex_year_month_day_dataframe_random_data
        s = ymd["A"]

        # 将 NaN 赋值给 s 的 (2000, 3) 位置
        s[2000, 3] = np.nan
        # 断言 s 的值数组中从索引 42 到 64 的所有值都为 NaN
        assert isna(s.values[42:65]).all()
        # 断言 s 的值数组中从索引 0 到 41 的所有值都不为 NaN
        assert notna(s.values[:42]).all()
        # 断言 s 的值数组中从索引 65 到末尾的所有值都不为 NaN
        assert notna(s.values[65:]).all()

        # 将 NaN 赋值给 s 的 (2000, 3, 10) 位置
        s[2000, 3, 10] = np.nan
        # 断言 s 的 iloc[49] 值为 NaN
        assert isna(s.iloc[49])

        # 使用 KeyError，期望引发异常，异常消息中包含 "49"
        with pytest.raises(KeyError, match="49"):
            # 当前索引 49 是整数时，不应回退到位置索引
            s[49]
    def test_frame_getitem_setitem_boolean(self, multiindex_dataframe_random_data):
        # 使用传入的多级索引数据框创建一个数据框副本
        frame = multiindex_dataframe_random_data
        # 转置数据框并创建其副本
        df = frame.T.copy()
        # 复制数据框的值
        values = df.values.copy()

        # 获取所有大于0的元素所在位置的数据框
        result = df[df > 0]
        # 用where方法获取所有大于0的元素所在位置的期望结果数据框
        expected = df.where(df > 0)
        # 断言两个数据框相等
        tm.assert_frame_equal(result, expected)

        # 将所有大于0的元素设置为5
        df[df > 0] = 5
        # 将所有大于0的值设置为5
        values[values > 0] = 5
        # 断言两个数据框的值几乎相等
        tm.assert_almost_equal(df.values, values)

        # 将所有等于5的元素设置为0
        df[df == 5] = 0
        # 将所有等于5的值设置为0
        values[values == 5] = 0
        # 断言两个数据框的值几乎相等
        tm.assert_almost_equal(df.values, values)

        # 需要首先对数据框进行对齐的情况下，将所有小于0的元素设置为2
        df[df[:-1] < 0] = 2
        # 将所有小于0的值设置为2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        # 断言两个数据框的值几乎相等
        tm.assert_almost_equal(df.values, values)

        # 使用 pytest 检查是否引发 TypeError，并匹配错误信息 "boolean values only"
        with pytest.raises(TypeError, match="boolean values only"):
            df[df * 0] = 2

    def test_frame_getitem_setitem_multislice(self):
        # 定义多级索引的层次结构和编码
        levels = [["t1", "t2"], ["a", "b", "c"]]
        codes = [[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]]
        # 使用层次结构和编码创建多级索引对象
        midx = MultiIndex(codes=codes, levels=levels, names=[None, "id"])
        # 创建数据框，指定索引和值
        df = DataFrame({"value": [1, 2, 3, 7, 8]}, index=midx)

        # 获取索引为 "value" 的系列数据
        result = df.loc[:, "value"]
        # 断言数据框的 "value" 列与结果系列相等
        tm.assert_series_equal(df["value"], result)

        # 获取索引为 1 到 3 的数据框的 "value" 列的系列数据
        result = df.loc[df.index[1:3], "value"]
        # 断言数据框的 "value" 列的子集与结果系列相等
        tm.assert_series_equal(df["value"][1:3], result)

        # 获取整个数据框
        result = df.loc[:, :]
        # 断言整个数据框与结果数据框相等
        tm.assert_frame_equal(df, result)

        # 将数据框的 "value" 列设置为10
        result = df
        df.loc[:, "value"] = 10
        result["value"] = 10
        # 断言数据框与结果数据框相等
        tm.assert_frame_equal(df, result)

        # 将整个数据框的值设置为10
        df.loc[:, :] = 10
        # 断言数据框与结果数据框相等
        tm.assert_frame_equal(df, result)

    def test_frame_setitem_multi_column(self):
        # 使用随机数生成器创建一个形状为 (10, 4) 的数据框，指定列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=[["a", "a", "b", "b"], [0, 1, 0, 1]],
        )

        # 创建数据框的副本
        cp = df.copy()
        # 将 "a" 列的值设置为 "b" 列的值
        cp["a"] = cp["b"]
        # 断言数据框的 "a" 列与 "b" 列相等
        tm.assert_frame_equal(cp["a"], cp["b"])

        # 使用 ndarray 将 "a" 列的值设置为 "b" 列的值
        cp = df.copy()
        cp["a"] = cp["b"].values
        # 断言数据框的 "a" 列与 "b" 列相等
        tm.assert_frame_equal(cp["a"], cp["b"])

    def test_frame_setitem_multi_column2(self):
        # ---------------------------------------
        # GH#1803
        # 创建多级索引的列
        columns = MultiIndex.from_tuples([("A", "1"), ("A", "2"), ("B", "1")])
        # 创建索引为 [1, 3, 5]，列为多级索引的数据框
        df = DataFrame(index=[1, 3, 5], columns=columns)

        # 设置 "A" 列的所有值为 0.0，但实际上增加了一列，而不是更新现有的两列
        df["A"] = 0.0  # 不起作用
        assert (df["A"].values == 0).all()

        # 广播操作，将 "B" 列的 "1" 子列设置为 [1, 2, 3]
        df["B", "1"] = [1, 2, 3]
        # 将 "B" 列的值设置为 "A" 列的值
        df["A"] = df["B", "1"]

        # 获取列 "A" 的子列 "1" 的系列数据
        sliced_a1 = df["A", "1"]
        # 获取列 "A" 的子列 "2" 的系列数据
        sliced_a2 = df["A", "2"]
        # 获取列 "B" 的子列 "1" 的系列数据
        sliced_b1 = df["B", "1"]
        # 断言系列数据的相等性，忽略名称检查
        tm.assert_series_equal(sliced_a1, sliced_b1, check_names=False)
        tm.assert_series_equal(sliced_a2, sliced_b1, check_names=False)
        # 断言系列数据的名称
        assert sliced_a1.name == ("A", "1")
        assert sliced_a2.name == ("A", "2")
        assert sliced_b1.name == ("B", "1")
    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    # 使用 pytest 标记，忽略特定警告信息：Setting a value on a view:FutureWarning
    def test_loc_getitem_setitem_slice_integers(self, frame_or_series):
        # 创建多索引对象，包含随机数据框
        index = MultiIndex(
            levels=[[0, 1, 2], [0, 2]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]
        )

        # 创建包含随机标准正态分布数据的数据框
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 4)),
            index=index,
            columns=["a", "b", "c", "d"],
        )
        # 获取 frame_or_series 对象的数据
        obj = tm.get_obj(obj, frame_or_series)

        # 使用 loc 方法进行切片操作
        res = obj.loc[1:2]
        # 重新索引 obj 到其索引的第二项开始
        exp = obj.reindex(obj.index[2:])
        # 比较结果和期望，使用测试模块中的等值函数
        tm.assert_equal(res, exp)

        # 使用 loc 方法设置切片范围的值为 7
        obj.loc[1:2] = 7
        # 断言 obj 中切片范围的所有值都等于 7
        assert (obj.loc[1:2] == 7).values.all()

    def test_setitem_change_dtype(self, multiindex_dataframe_random_data):
        # 获取随机数据框
        frame = multiindex_dataframe_random_data
        # 转置数据框
        dft = frame.T
        # 选择 foo, two 列作为序列 s
        s = dft["foo", "two"]
        # 将 foo, two 列设置为大于其中位数的布尔值
        dft["foo", "two"] = s > s.median()
        # 比较结果和期望，使用测试模块中的等值函数
        tm.assert_series_equal(dft["foo", "two"], s > s.median())
        # 重新索引 dft 到只包含 ("foo", "two") 列
        reindexed = dft.reindex(columns=[("foo", "two")])
        # 比较重新索引后的结果和期望，使用测试模块中的等值函数
        tm.assert_series_equal(reindexed["foo", "two"], s > s.median())

    def test_set_column_scalar_with_loc(self, multiindex_dataframe_random_data):
        # 获取随机数据框
        frame = multiindex_dataframe_random_data
        # 选择索引中的子集
        subset = frame.index[[1, 4, 5]]

        # 使用 loc 方法将 subset 所有行的值设置为 99
        frame.loc[subset] = 99
        # 断言 subset 所有行的值都等于 99
        assert (frame.loc[subset].values == 99).all()

        # 复制 frame 数据框
        frame_original = frame.copy()
        # 获取 "B" 列作为序列 col
        col = frame["B"]
        # 使用 loc 方法将 subset 所有行的 "B" 列的值设置为 97
        col[subset] = 97
        # 断言 frame 和 frame_original 相等，使用测试模块中的等值函数
        # 链式 setitem 操作不适用于 CoW（Copy-on-Write）模式
        tm.assert_frame_equal(frame, frame_original)

    def test_nonunique_assignment_1750(self):
        # 创建包含特定值的数据框
        df = DataFrame(
            [[1, 1, "x", "X"], [1, 1, "y", "Y"], [1, 2, "z", "Z"]],
            columns=list("ABCD")
        )

        # 将 "A", "B" 列设置为索引
        df = df.set_index(["A", "B"])
        # 创建多索引对象 mi
        mi = MultiIndex.from_tuples([(1, 1)])

        # 使用 loc 方法将 mi 对应行的 "C" 列设置为 "_"
        df.loc[mi, "C"] = "_"

        # 断言 df 中 (1, 1) 行的 "C" 列所有值都等于 "_"
        assert (df.xs((1, 1))["C"] == "_").all()

    def test_astype_assignment_with_dups(self):
        # GH 4686
        # 创建具有对象类型的包含索引和列的多索引对象 cols 的数据框 df
        # 进行 dtype 改变的重复项赋值
        cols = MultiIndex.from_tuples([("A", "1"), ("B", "1"), ("A", "2")])
        df = DataFrame(np.arange(3).reshape((1, 3)), columns=cols, dtype=object)
        # 复制 df 的索引
        index = df.index.copy()

        # 将 df 中 "A" 列的类型转换为 np.float64
        df["A"] = df["A"].astype(np.float64)
        # 比较 df 的索引和复制的索引，使用测试模块中的等值函数
        tm.assert_index_equal(df.index, index)
    # 定义一个测试方法，用于测试 DataFrame 的 loc 方法在非单调性索引设置时的行为
    def test_setitem_nonmonotonic(self):
        # 创建一个多级索引对象，包含元组 ("a", "c"), ("b", "x"), ("a", "d")，并指定索引名称为 ["l1", "l2"]
        index = MultiIndex.from_tuples(
            [("a", "c"), ("b", "x"), ("a", "d")], names=["l1", "l2"]
        )
        # 创建一个 DataFrame 对象，数据为 [0, 1, 2]，索引为上面创建的多级索引，列名为 ["e"]
        df = DataFrame(data=[0, 1, 2], index=index, columns=["e"])
        # 使用 loc 方法将索引为 "a" 的行中的 "e" 列设置为 np.arange(99, 101, dtype="int64") 所生成的数组
        df.loc["a", "e"] = np.arange(99, 101, dtype="int64")
        # 创建一个预期的 DataFrame 对象，数据为 {"e": [99, 1, 100]}，索引与原始的 index 相同
        expected = DataFrame({"e": [99, 1, 100]}, index=index)
        # 使用测试工具方法 assert_frame_equal 来比较 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)
class TestSetitemWithExpansionMultiIndex:
    # 测试在多索引 DataFrame 中设置新列，包含混合深度
    def test_setitem_new_column_mixed_depth(self):
        arrays = [
            ["a", "top", "top", "routine1", "routine1", "routine2"],
            ["", "OD", "OD", "result1", "result2", "result1"],
            ["", "wx", "wy", "", "", ""],
        ]

        # 对数组进行排序并转换为元组列表
        tuples = sorted(zip(*arrays))
        # 使用元组列表创建多索引对象
        index = MultiIndex.from_tuples(tuples)
        # 创建随机数据的 DataFrame，列使用上述多索引
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

        # 复制 DataFrame 以进行测试
        result = df.copy()
        expected = df.copy()
        # 在 result 中设置新列 "b"
        result["b"] = [1, 2, 3, 4]
        # 在 expected 中设置新列 ("b", "", "")
        expected["b", "", ""] = [1, 2, 3, 4]
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试在多索引 DataFrame 中设置新列，所有值为 NaN
    def test_setitem_new_column_all_na(self):
        # GH#1534
        # 创建指定多索引的 DataFrame
        mix = MultiIndex.from_tuples([("1a", "2a"), ("1a", "2b"), ("1a", "2c")])
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=mix)
        # 创建 Series 对象
        s = Series({(1, 1): 1, (1, 2): 2})
        # 在 df 中设置新列 "new"
        df["new"] = s
        # 断言 "new" 列是否全部为 NaN
        assert df["new"].isna().all()

    # 测试在扩展 DataFrame 时保持索引名称不变
    def test_setitem_enlargement_keep_index_names(self):
        # GH#53053
        # 创建具有指定索引名称的多索引对象
        mi = MultiIndex.from_tuples([(1, 2, 3)], names=["i1", "i2", "i3"])
        # 创建具有指定数据和索引的 DataFrame
        df = DataFrame(data=[[10, 20, 30]], index=mi, columns=["A", "B", "C"])
        # 根据索引复制数据到新行
        df.loc[(0, 0, 0)] = df.loc[(1, 2, 3)]
        # 创建期望的多索引对象
        mi_expected = MultiIndex.from_tuples(
            [(1, 2, 3), (0, 0, 0)], names=["i1", "i2", "i3"]
        )
        # 创建期望的 DataFrame
        expected = DataFrame(
            data=[[10, 20, 30], [10, 20, 30]],
            index=mi_expected,
            columns=["A", "B", "C"],
        )
        # 断言 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)
```