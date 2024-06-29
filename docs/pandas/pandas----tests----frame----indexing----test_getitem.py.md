# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_getitem.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

from pandas import (  # 从pandas库中导入多个子模块和类
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DateOffset,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    get_dummies,
    period_range,
)
import pandas._testing as tm  # 导入pandas测试模块
from pandas.core.arrays import SparseArray  # 从pandas核心数组模块导入SparseArray类


class TestGetitem:
    def test_getitem_unused_level_raises(self):
        # GH#20410: 测试访问未使用的层级时是否引发KeyError异常
        mi = MultiIndex(
            levels=[["a_lot", "onlyone", "notevenone"], [1970, ""]],
            codes=[[1, 0], [1, 0]],
        )
        df = DataFrame(-1, index=range(3), columns=mi)

        with pytest.raises(KeyError, match="notevenone"):
            df["notevenone"]

    def test_getitem_periodindex(self):
        rng = period_range("1/1/2000", periods=5)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)

        ts = df[rng[0]]  # 获取特定PeriodIndex的列数据
        tm.assert_series_equal(ts, df.iloc[:, 0])  # 断言Series相等性

        ts = df["1/1/2000"]  # 使用日期字符串获取PeriodIndex的列数据
        tm.assert_series_equal(ts, df.iloc[:, 0])  # 断言Series相等性

    def test_getitem_list_of_labels_categoricalindex_cols(self):
        # GH#16115: 测试从CategoricalIndex列获取数据的情况
        cats = Categorical([Timestamp("12-31-1999"), Timestamp("12-31-2000")])

        expected = DataFrame([[1, 0], [0, 1]], dtype="bool", index=[0, 1], columns=cats)
        dummies = get_dummies(cats)
        result = dummies[list(dummies.columns)]
        tm.assert_frame_equal(result, expected)

    def test_getitem_sparse_column_return_type_and_dtype(self):
        # https://github.com/pandas-dev/pandas/issues/23559
        data = SparseArray([0, 1])
        df = DataFrame({"A": data})
        expected = Series(data, name="A")
        result = df["A"]  # 获取稀疏列数据
        tm.assert_series_equal(result, expected)  # 断言Series相等性

        # 同时检查iloc和loc的情况
        result = df.iloc[:, 0]  # 使用iloc获取稀疏列数据
        tm.assert_series_equal(result, expected)  # 断言Series相等性

        result = df.loc[:, "A"]  # 使用loc获取稀疏列数据
        tm.assert_series_equal(result, expected)  # 断言Series相等性

    def test_getitem_string_columns(self):
        # GH#46185: 测试使用字符串类型的列名访问DataFrame列数据
        df = DataFrame([[1, 2]], columns=Index(["A", "B"], dtype="string"))
        result = df.A  # 通过属性访问列"A"
        expected = df["A"]
        tm.assert_series_equal(result, expected)


class TestGetitemListLike:
    def test_getitem_list_missing_key(self):
        # GH#13822: 测试在具有非唯一列名的情况下，访问缺失列名是否引发正确的KeyError异常
        df = DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
        df.columns = ["x", "x", "z"]

        # 检查确保在KeyError中得到正确的错误信息
        with pytest.raises(KeyError, match=r"\['y'\] not in index"):
            df[["x", "y", "z"]]
    # 定义测试函数，用于测试从 DataFrame 中获取带有重复列名的列
    def test_getitem_list_duplicates(self):
        # GH#1943
        # 创建一个 4x4 的随机数 DataFrame，列名为 ['A', 'A', 'B', 'C']，并设置列名为 'foo'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), columns=list("AABC")
        )
        df.columns.name = "foo"

        # 选择列 'B' 和 'C'，返回结果 DataFrame
        result = df[["B", "C"]]
        # 断言选择结果的列名为 'foo'
        assert result.columns.name == "foo"

        # 预期选择的结果为 df 的所有行和列索引为 2 及其后的列
        expected = df.iloc[:, 2:]
        # 使用测试工具比较结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试 DataFrame 中包含重复列名的情况下获取不存在的列
    def test_getitem_dupe_cols(self):
        # 创建一个包含重复列名的 DataFrame
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
        # 预期会引发 KeyError 异常，异常消息中包含指定的信息
        msg = "\"None of [Index(['baf'], dtype="
        # 使用 pytest 的断言检查是否引发了预期的异常，并匹配异常消息
        with pytest.raises(KeyError, match=re.escape(msg)):
            df[["baf"]]

    # 使用 pytest 的参数化装饰器定义多个参数化测试，测试不同类型的索引
    @pytest.mark.parametrize(
        "idx_type",
        [
            list,
            iter,
            Index,
            set,
            lambda keys: dict(zip(keys, range(len(keys)))),
            lambda keys: dict(zip(keys, range(len(keys)))).keys(),
        ],
        ids=["list", "iter", "Index", "set", "dict", "dict_keys"],
    )
    @pytest.mark.parametrize("levels", [1, 2])
    def test_getitem_listlike(self, idx_type, levels, float_frame):
        # GH#21294

        if levels == 1:
            # 如果 levels 为 1，则使用 float_frame，并设置 missing 变量为 "food"
            frame, missing = float_frame, "food"
        else:
            # 否则，创建一个 MultiIndex 列的 DataFrame
            frame = DataFrame(
                np.random.default_rng(2).standard_normal((8, 3)),
                columns=Index(
                    [("foo", "bar"), ("baz", "qux"), ("peek", "aboo")],
                    name=("sth", "sth2"),
                ),
            )
            missing = ("good", "food")

        # 选择 frame 的第 1 列和第 0 列作为 keys
        keys = [frame.columns[1], frame.columns[0]]
        # 使用 idx_type 创建索引 idx 和 idx_check
        idx = idx_type(keys)
        idx_check = list(idx_type(keys))

        # 如果 idx 类型为 set 或 dict，则预期会引发 TypeError 异常
        if isinstance(idx, (set, dict)):
            with pytest.raises(TypeError, match="as an indexer is not supported"):
                frame[idx]
            return
        else:
            # 否则，获取 frame 中对应 idx 的数据
            result = frame[idx]

        # 预期结果为 frame 的所有行和 idx_check 列
        expected = frame.loc[:, idx_check]
        # 设置预期结果的列名与 frame 的列名相同
        expected.columns.names = frame.columns.names

        # 使用测试工具比较结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 将 keys 添加 missing 后作为 idx
        idx = idx_type(keys + [missing])
        # 预期会引发 KeyError 异常，异常消息中包含 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            frame[idx]

    # 定义测试函数，测试通过生成器作为索引的 iloc 操作
    def test_getitem_iloc_generator(self):
        # GH#39614
        # 创建一个包含两列 'a' 和 'b' 的 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # 创建生成器作为索引器
        indexer = (x for x in [1, 2])
        # 使用 iloc 进行索引操作，返回结果 DataFrame
        result = df.iloc[indexer]
        # 创建预期的 DataFrame，包含索引为 1 和 2 的数据
        expected = DataFrame({"a": [2, 3], "b": [5, 6]}, index=[1, 2])
        # 使用测试工具比较结果和预期 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试通过生成器作为索引的 iloc 操作，同时选择指定的列
    def test_getitem_iloc_two_dimensional_generator(self):
        # 创建一个包含两列 'a' 和 'b' 的 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # 创建生成器作为索引器
        indexer = (x for x in [1, 2])
        # 使用 iloc 进行索引操作，选择列 'b'，返回结果 Series
        result = df.iloc[indexer, 1]
        # 创建预期的 Series，包含索引为 1 和 2 的列 'b' 的数据
        expected = Series([5, 6], name="b", index=[1, 2])
        # 使用测试工具比较结果和预期 Series 是否相等
        tm.assert_series_equal(result, expected)
    # 定义测试函数，验证针对日期偏移量的索引功能
    def test_getitem_iloc_dateoffset_days(self):
        # GH 46671：参考GitHub问题编号
        # 创建包含数字列表的DataFrame，使用日期范围作为索引，日期间隔为一天
        df = DataFrame(
            list(range(10)),
            index=date_range("01-01-2022", periods=10, freq=DateOffset(days=1)),
        )
        # 对DataFrame进行日期范围的loc操作，获取"2022-01-01"到"2022-01-03"之间的数据
        result = df.loc["2022-01-01":"2022-01-03"]
        # 创建期望的DataFrame，包含预期的数据和日期索引
        expected = DataFrame(
            [0, 1, 2],
            index=DatetimeIndex(
                ["2022-01-01", "2022-01-02", "2022-01-03"],
                dtype="datetime64[ns]",
                freq=DateOffset(days=1),
            ),
        )
        # 使用测试工具比较结果DataFrame和期望的DataFrame
        tm.assert_frame_equal(result, expected)

        # 创建另一个DataFrame，使用日期范围作为索引，日期间隔为一天两小时
        df = DataFrame(
            list(range(10)),
            index=date_range(
                "01-01-2022", periods=10, freq=DateOffset(days=1, hours=2)
            ),
        )
        # 对DataFrame进行日期范围的loc操作，获取"2022-01-01"到"2022-01-03"之间的数据
        result = df.loc["2022-01-01":"2022-01-03"]
        # 创建期望的DataFrame，包含预期的数据和日期索引
        expected = DataFrame(
            [0, 1, 2],
            index=DatetimeIndex(
                ["2022-01-01 00:00:00", "2022-01-02 02:00:00", "2022-01-03 04:00:00"],
                dtype="datetime64[ns]",
                freq=DateOffset(days=1, hours=2),
            ),
        )
        # 使用测试工具比较结果DataFrame和期望的DataFrame
        tm.assert_frame_equal(result, expected)

        # 创建另一个DataFrame，使用日期范围作为索引，日期间隔为三分钟
        df = DataFrame(
            list(range(10)),
            index=date_range("01-01-2022", periods=10, freq=DateOffset(minutes=3)),
        )
        # 对DataFrame进行日期范围的loc操作，获取"2022-01-01"到"2022-01-03"之间的数据
        result = df.loc["2022-01-01":"2022-01-03"]
        # 使用测试工具比较结果DataFrame和原始DataFrame，期望它们相等
        tm.assert_frame_equal(result, df)
class TestGetitemCallable:
    def test_getitem_callable(self, float_frame):
        # GH#12533
        # 使用 lambda 函数从 float_frame 中获取 "A" 列的数据
        result = float_frame[lambda x: "A"]
        # 用 loc 方法获取 float_frame 中所有行的 "A" 列数据作为期望结果
        expected = float_frame.loc[:, "A"]
        # 断言 result 和 expected 序列相等
        tm.assert_series_equal(result, expected)

        # 使用 lambda 函数从 float_frame 中获取 "A" 和 "B" 列的数据
        result = float_frame[lambda x: ["A", "B"]]
        # 用 loc 方法获取 float_frame 中所有行的 "A" 和 "B" 列数据作为期望结果
        expected = float_frame.loc[:, ["A", "B"]]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, float_frame.loc[:, ["A", "B"]])

        # 从 float_frame 中取前三行数据
        df = float_frame[:3]
        # 使用 lambda 函数选择出第 1 和第 3 行数据
        result = df[lambda x: [True, False, True]]
        # 用 iloc 方法选择 float_frame 中第 1 和第 3 行作为期望结果
        expected = float_frame.iloc[[0, 2], :]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

    def test_loc_multiindex_columns_one_level(self):
        # GH#29749
        # 创建一个具有多级列索引的数据帧 df
        df = DataFrame([[1, 2]], columns=[["a", "b"]])
        # 创建一个只包含 "a" 列的期望结果数据帧
        expected = DataFrame([1], columns=[["a"]])

        # 通过列名 "a" 获取 df 中的数据
        result = df["a"]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

        # 使用 loc 方法通过列名 "a" 获取 df 中的数据
        result = df.loc[:, "a"]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)


class TestGetitemBooleanMask:
    def test_getitem_bool_mask_categorical_index(self):
        # 创建一个具有分类索引的数据帧 df3
        df3 = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
            },
            index=CategoricalIndex(
                [1, 1, 2, 1, 3, 2],
                dtype=CategoricalDtype([3, 2, 1], ordered=True),
                name="B",
            ),
        )
        # 创建另一个具有分类索引的数据帧 df4
        df4 = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
            },
            index=CategoricalIndex(
                [1, 1, 2, 1, 3, 2],
                dtype=CategoricalDtype([3, 2, 1], ordered=False),
                name="B",
            ),
        )

        # 通过索引值 "a" 从 df3 中选择数据
        result = df3[df3.index == "a"]
        # 创建一个空的期望结果数据帧
        expected = df3.iloc[[]]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

        # 通过索引值 "a" 从 df4 中选择数据
        result = df4[df4.index == "a"]
        # 创建一个空的期望结果数据帧
        expected = df4.iloc[[]]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

        # 通过索引值 1 从 df3 中选择数据
        result = df3[df3.index == 1]
        # 用 iloc 方法选择 df3 中索引为 0, 1, 3 的行作为期望结果
        expected = df3.iloc[[0, 1, 3]]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

        # 通过索引值 1 从 df4 中选择数据
        result = df4[df4.index == 1]
        # 用 iloc 方法选择 df4 中索引为 0, 1, 3 的行作为期望结果
        expected = df4.iloc[[0, 1, 3]]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

        # 对于有序分类索引

        # 由于有序分类索引的存在
        result = df3[df3.index < 2]
        # 用 iloc 方法选择 df3 中索引为 4 的行作为期望结果
        expected = df3.iloc[[4]]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

        # 通过索引值大于 1 从 df3 中选择数据
        result = df3[df3.index > 1]
        # 创建一个空的期望结果数据帧
        expected = df3.iloc[[]]
        # 断言 result 和 expected 数据帧相等
        tm.assert_frame_equal(result, expected)

        # 对于无序分类索引

        # 无法比较

        # 由于分类索引是无序的
        msg = "Unordered Categoricals can only compare equality or not"
        # 使用 pytest 来断言引发 TypeError 异常并匹配消息
        with pytest.raises(TypeError, match=msg):
            df4[df4.index < 2]
        with pytest.raises(TypeError, match=msg):
            df4[df4.index > 1]
    @pytest.mark.parametrize(
        "data1,data2,expected_data",
        (
            (
                [[1, 2], [3, 4]],
                [[0.5, 6], [7, 8]],
                [[np.nan, 3.0], [np.nan, 4.0], [np.nan, 7.0], [6.0, 8.0]],
            ),
            (
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[np.nan, 3.0], [np.nan, 4.0], [5, 7], [6, 8]],
            ),
        ),
    )
    def test_getitem_bool_mask_duplicate_columns_mixed_dtypes(
        self,
        data1,
        data2,
        expected_data,
    ):
        # GH#31954
        # 使用 pytest 的 parametrize 装饰器，允许在单个测试方法中多次运行相同的测试逻辑
        df1 = DataFrame(np.array(data1))  # 创建 DataFrame df1，使用 data1 的数据
        df2 = DataFrame(np.array(data2))  # 创建 DataFrame df2，使用 data2 的数据
        df = concat([df1, df2], axis=1)    # 将 df1 和 df2 沿着列方向连接成一个 DataFrame df

        result = df[df > 2]  # 对 df 中大于 2 的元素进行布尔索引，得到一个新的 DataFrame result

        exdict = {i: np.array(col) for i, col in enumerate(expected_data)}  # 创建预期的字典 exdict，包含 expected_data 的数据
        expected = DataFrame(exdict).rename(columns={2: 0, 3: 1})  # 根据 exdict 创建 DataFrame expected，并重新命名列名
        tm.assert_frame_equal(result, expected)  # 断言 result 和 expected 的 DataFrame 结果是否相等

    @pytest.fixture
    def df_dup_cols(self):
        dups = ["A", "A", "C", "D"]
        df = DataFrame(np.arange(12).reshape(3, 4), columns=dups, dtype="float64")  # 创建具有重复列的 DataFrame df
        return df

    def test_getitem_boolean_frame_unaligned_with_duplicate_columns(self, df_dup_cols):
        # `df.A > 6` is a DataFrame with a different shape from df
        # 当使用 df.A > 6 进行布尔索引时，返回的 DataFrame 形状与 df 不同

        # boolean with the duplicate raises
        df = df_dup_cols  # 使用 df_dup_cols 作为测试用的 DataFrame df
        msg = "cannot reindex on an axis with duplicate labels"
        with pytest.raises(ValueError, match=msg):  # 使用 pytest.raises 检测是否抛出 ValueError 异常，并匹配错误信息 msg
            df[df.A > 6]  # 在 df 上进行布尔索引，期望抛出 ValueError 异常

    def test_getitem_boolean_series_with_duplicate_columns(self, df_dup_cols):
        # boolean indexing
        # GH#4879
        df = DataFrame(
            np.arange(12).reshape(3, 4), columns=["A", "B", "C", "D"], dtype="float64"
        )  # 创建一个具有不同列名的 DataFrame df
        expected = df[df.C > 6]  # 使用 df.C > 6 进行布尔索引，创建预期的 DataFrame expected
        expected.columns = df_dup_cols.columns  # 将 expected 的列名修改为 df_dup_cols 的列名

        df = df_dup_cols  # 使用 df_dup_cols 作为测试用的 DataFrame df
        result = df[df.C > 6]  # 在 df 上进行布尔索引，得到结果 DataFrame result

        tm.assert_frame_equal(result, expected)  # 断言 result 和 expected 的 DataFrame 结果是否相等

    def test_getitem_boolean_frame_with_duplicate_columns(self, df_dup_cols):
        # where
        df = DataFrame(
            np.arange(12).reshape(3, 4), columns=["A", "B", "C", "D"], dtype="float64"
        )  # 创建一个具有不同列名的 DataFrame df
        # `df > 6` is a DataFrame with the same shape+alignment as df
        expected = df[df > 6]  # 使用 df > 6 进行布尔索引，创建预期的 DataFrame expected
        expected.columns = df_dup_cols.columns  # 将 expected 的列名修改为 df_dup_cols 的列名

        df = df_dup_cols  # 使用 df_dup_cols 作为测试用的 DataFrame df
        result = df[df > 6]  # 在 df 上进行布尔索引，得到结果 DataFrame result

        tm.assert_frame_equal(result, expected)  # 断言 result 和 expected 的 DataFrame 结果是否相等

    def test_getitem_empty_frame_with_boolean(self):
        # Test for issue GH#11859
        # 测试 GH#11859 的问题，测试空 DataFrame 使用布尔索引

        df = DataFrame()  # 创建一个空的 DataFrame df
        df2 = df[df > 0]  # 在空的 DataFrame 上进行布尔索引
        tm.assert_frame_equal(df, df2)  # 断言 df 和 df2 的结果是否相等

    def test_getitem_returns_view_when_column_is_unique_in_df(self):
        # GH#45316
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])  # 创建具有重复列的 DataFrame df
        df_orig = df.copy()  # 复制一份 DataFrame df 的原始副本
        view = df["b"]  # 获取 df 的 "b" 列作为视图 view
        view.loc[:] = 100  # 将 view 的所有元素设为 100
        expected = df_orig  # 期望的结果是 df 的原始副本
        tm.assert_frame_equal(df, expected)  # 断言 df 和 expected 的结果是否相等
    # 定义一个测试函数，用于测试从DataFrame中获取特定frozenset作为列索引的行数据
    def test_getitem_frozenset_unique_in_column(self):
        # GH#41062，指明这个测试用例的编号或相关问题的参考号码
        # 创建一个DataFrame，包含一行数据：[1, 2, 3, 4]，列索引分别为[frozenset(["KEY"]), "B", "C", "C"]
        df = DataFrame([[1, 2, 3, 4]], columns=[frozenset(["KEY"]), "B", "C", "C"])
        # 从DataFrame中获取指定的frozenset作为列索引，返回结果为一个Series对象
        result = df[frozenset(["KEY"])]
        # 创建预期的Series对象，其值为[1]，名称为frozenset(["KEY"])
        expected = Series([1], name=frozenset(["KEY"]))
        # 使用测试模块中的函数来比较两个Series对象是否相等
        tm.assert_series_equal(result, expected)
class TestGetitemSlice:
    # 定义测试类 TestGetitemSlice

    def test_getitem_slice_float64(self, frame_or_series):
        # 测试方法：测试浮点数类型的切片操作
        values = np.arange(10.0, 50.0, 2)
        # 创建包含浮点数序列的 numpy 数组
        index = Index(values)
        # 使用该数组创建索引对象 Index

        start, end = values[[5, 15]]
        # 从 values 数组中获取索引为 5 和 15 的两个值作为切片的起始和结束位置

        data = np.random.default_rng(2).standard_normal((20, 3))
        # 使用随机数生成器创建一个形状为 (20, 3) 的标准正态分布数组
        if frame_or_series is not DataFrame:
            # 如果 frame_or_series 不是 DataFrame 类型
            data = data[:, 0]
            # 则将数据数组限制在第一列

        obj = frame_or_series(data, index=index)
        # 使用数据和索引创建 frame_or_series 对象

        result = obj[start:end]
        # 对对象进行切片操作，使用起始和结束位置进行切片
        expected = obj.iloc[5:16]
        # 使用 iloc 方法获取预期的切片对象

        tm.assert_equal(result, expected)
        # 使用测试框架中的 assert_equal 方法验证 result 和 expected 是否相等

        result = obj.loc[start:end]
        # 使用 loc 方法进行标签（索引值）切片操作
        tm.assert_equal(result, expected)
        # 再次使用 assert_equal 方法验证结果是否符合预期

    def test_getitem_datetime_slice(self):
        # 测试方法：测试日期时间类型的切片操作
        # GH#43223

        df = DataFrame(
            {"a": 0},
            index=DatetimeIndex(
                [
                    "11.01.2011 22:00",
                    "11.01.2011 23:00",
                    "12.01.2011 00:00",
                    "2011-01-13 00:00",
                ]
            ),
        )
        # 创建一个 DataFrame，指定列 'a' 和日期时间索引

        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            # 使用 pytest 框架的 raises 方法，期望引发 KeyError，并验证错误信息
            df["2011-01-01":"2011-11-01"]
            # 尝试对非单调递增索引进行基于值的部分切片操作

    def test_getitem_slice_same_dim_only_one_axis(self):
        # 测试方法：测试仅有一个轴的相同维度切片操作
        # GH#54622

        df = DataFrame(np.random.default_rng(2).standard_normal((10, 8)))
        # 创建一个形状为 (10, 8) 的 DataFrame，使用随机数生成器生成标准正态分布数据

        result = df.iloc[(slice(None, None, 2),)]
        # 使用 iloc 方法对 DataFrame 进行切片操作，每隔两个取一个元素
        assert result.shape == (5, 8)
        # 断言结果的形状应为 (5, 8)

        expected = df.iloc[slice(None, None, 2), slice(None)]
        # 使用 iloc 方法获取预期的切片结果，每隔两个取一个元素
        tm.assert_frame_equal(result, expected)
        # 使用测试框架中的 assert_frame_equal 方法验证 result 和 expected 是否相等


class TestGetitemDeprecatedIndexers:
    # 定义测试类 TestGetitemDeprecatedIndexers

    @pytest.mark.parametrize("key", [{"a", "b"}, {"a": "a"}])
    # 使用 pytest 的 parametrize 装饰器，传入不同的参数 key 进行参数化测试
    def test_getitem_dict_and_set_deprecated(self, key):
        # 测试方法：测试已弃用的字典和集合作为索引器
        # GH#42825 enforced in 2.0

        df = DataFrame(
            [[1, 2], [3, 4]], columns=MultiIndex.from_tuples([("a", 1), ("b", 2)])
        )
        # 创建一个 DataFrame，包含多级索引的列

        with pytest.raises(TypeError, match="as an indexer is not supported"):
            # 使用 pytest 框架的 raises 方法，期望引发 TypeError，并验证错误信息
            df[key]
            # 尝试使用字典或集合作为索引器访问 DataFrame
```