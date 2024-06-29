# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_fillna.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas._config import using_pyarrow_string_dtype  # 导入使用 PyArrow 的配置信息

from pandas import (  # 从 Pandas 库中导入多个类和函数
    Categorical,
    DataFrame,
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.tests.frame.common import _check_mixed_float  # 导入用于混合浮点数检查的函数


class TestFillNA:
    def test_fillna_dict_inplace_nonunique_columns(self):
        df = DataFrame(
            {"A": [np.nan] * 3, "B": [NaT, Timestamp(1), NaT], "C": [np.nan, "foo", 2]}
        )
        df.columns = ["A", "A", "A"]  # 设置 DataFrame 的列名为非唯一值
        orig = df[:]  # 备份原始 DataFrame

        df.fillna({"A": 2}, inplace=True)  # 对列"A"进行填充，原地修改
        # 第一列和第三列可以原地设置，而第二列不能。

        expected = DataFrame(
            {"A": [2.0] * 3, "B": [2, Timestamp(1), 2], "C": [2, "foo", 2]}
        )
        expected.columns = ["A", "A", "A"]
        tm.assert_frame_equal(df, expected)  # 使用测试模块检查 DataFrame 是否符合预期
        assert not tm.shares_memory(df.iloc[:, 1], orig.iloc[:, 1])  # 检查第二列是否与原始数据共享内存

    def test_fillna_on_column_view(self):
        # 避免不必要的复制
        arr = np.full((40, 50), np.nan)  # 创建一个大小为 (40, 50) 的 NaN 数组
        df = DataFrame(arr, copy=False)  # 创建一个 DataFrame，不复制数组数据

        with tm.raises_chained_assignment_error():  # 使用测试模块检查是否抛出链式赋值错误
            df[0].fillna(-1, inplace=True)  # 对 DataFrame 的第一列进行填充，原地修改
        assert np.isnan(arr[:, 0]).all()  # 检查原始数组的第一列是否仍然全为 NaN

        # 即没有创建一个新的 49 列数据块
        assert len(df._mgr.blocks) == 1  # 检查 DataFrame 内部的数据块数是否为 1
        assert np.shares_memory(df.values, arr)  # 检查 DataFrame 的值是否与原始数组共享内存

    def test_fillna_datetime(self, datetime_frame):
        tf = datetime_frame  # 使用传入的日期时间 DataFrame
        tf.loc[tf.index[:5], "A"] = np.nan  # 将前五行的"A"列设置为 NaN
        tf.loc[tf.index[-5:], "A"] = np.nan  # 将后五行的"A"列设置为 NaN

        zero_filled = datetime_frame.fillna(0)  # 对 DataFrame 进行填充，使用 0 替代 NaN
        assert (zero_filled.loc[zero_filled.index[:5], "A"] == 0).all()  # 检查填充后的前五行是否全部为 0

        padded = datetime_frame.ffill()  # 前向填充 DataFrame
        assert np.isnan(padded.loc[padded.index[:5], "A"]).all()  # 检查前向填充后前五行的"A"列是否全部为 NaN

        msg = r"missing 1 required positional argument: 'value'"
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 检查是否抛出类型错误异常，并匹配特定消息
            datetime_frame.fillna()

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't fill 0 in string")
    def test_fillna_mixed_type(self, float_string_frame):
        mf = float_string_frame  # 使用传入的混合类型 DataFrame
        mf.loc[mf.index[5:20], "foo"] = np.nan  # 将索引为 5 到 19 的"foo"列设置为 NaN
        mf.loc[mf.index[-10:], "A"] = np.nan  # 将后十行的"A"列设置为 NaN
        # TODO: 在此处添加更强的断言，GH 25640
        mf.fillna(value=0)  # 对 DataFrame 进行填充，使用 0 替代 NaN
        mf.ffill()  # 前向填充 DataFrame

    def test_fillna_mixed_float(self, mixed_float_frame):
        # 混合数字类型（但不包括 float16）
        mf = mixed_float_frame.reindex(columns=["A", "B", "D"])  # 重新索引以选择特定列
        mf.loc[mf.index[-10:], "A"] = np.nan  # 将后十行的"A"列设置为 NaN
        result = mf.fillna(value=0)  # 对 DataFrame 进行填充，使用 0 替代 NaN
        _check_mixed_float(result, dtype={"C": None})  # 使用检查函数检查混合浮点数
        result = mf.ffill()  # 前向填充 DataFrame
        _check_mixed_float(result, dtype={"C": None})  # 使用检查函数再次检查混合浮点数
    def test_fillna_different_dtype(self, using_infer_string):
        # with different dtype (GH#3386)
        # 创建一个包含不同数据类型的DataFrame，其中包括NaN值
        df = DataFrame(
            [["a", "a", np.nan, "a"], ["b", "b", np.nan, "b"], ["c", "c", np.nan, "c"]]
        )

        if using_infer_string:
            # 如果 using_infer_string 为True，期望产生一个未来警告
            with tm.assert_produces_warning(FutureWarning, match="Downcasting"):
                result = df.fillna({2: "foo"})
        else:
            # 否则，直接填充NaN值为字符串'foo'
            result = df.fillna({2: "foo"})
        # 期望的填充后的DataFrame
        expected = DataFrame(
            [["a", "a", "foo", "a"], ["b", "b", "foo", "b"], ["c", "c", "foo", "c"]]
        )
        # 验证填充后的结果与期望是否相同
        tm.assert_frame_equal(result, expected)

        if using_infer_string:
            # 如果 using_infer_string 为True，期望产生一个未来警告，并且原地填充
            with tm.assert_produces_warning(FutureWarning, match="Downcasting"):
                return_value = df.fillna({2: "foo"}, inplace=True)
        else:
            # 否则，原地填充NaN值为字符串'foo'
            return_value = df.fillna({2: "foo"}, inplace=True)
        # 验证原地填充后的DataFrame与期望是否相同
        tm.assert_frame_equal(df, expected)
        # 原地填充操作返回None
        assert return_value is None

    def test_fillna_limit_and_value(self):
        # limit and value
        # 创建一个包含随机数据的DataFrame，然后对部分数据设为NaN
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
        df.iloc[2:7, 0] = np.nan
        df.iloc[3:5, 2] = np.nan

        expected = df.copy()
        # 将特定位置的NaN值填充为999，但只限制填充一次
        expected.iloc[2, 0] = 999
        expected.iloc[3, 2] = 999
        result = df.fillna(999, limit=1)
        # 验证填充后的结果与期望是否相同
        tm.assert_frame_equal(result, expected)

    def test_fillna_datelike(self):
        # with datelike
        # 使用日期类型填充NaN值，解决GH#6344
        df = DataFrame(
            {
                "Date": [NaT, Timestamp("2014-1-1")],
                "Date2": [Timestamp("2013-1-1"), NaT],
            }
        )

        expected = df.copy()
        # 使用 Date2 列的值填充 Date 列的NaN值
        expected["Date"] = expected["Date"].fillna(df.loc[df.index[0], "Date2"])
        result = df.fillna(value={"Date": df["Date2"]})
        # 验证填充后的结果与期望是否相同
        tm.assert_frame_equal(result, expected)

    def test_fillna_tzaware(self):
        # with timezone
        # 使用时区信息填充NaN值，解决GH#15855
        df = DataFrame({"A": [Timestamp("2012-11-11 00:00:00+01:00"), NaT]})
        exp = DataFrame(
            {
                "A": [
                    Timestamp("2012-11-11 00:00:00+01:00"),
                    Timestamp("2012-11-11 00:00:00+01:00"),
                ]
            }
        )
        # 使用前向填充NaN值
        res = df.ffill()
        # 验证填充后的结果与期望是否相同
        tm.assert_frame_equal(res, exp)

        df = DataFrame({"A": [NaT, Timestamp("2012-11-11 00:00:00+01:00")]})
        exp = DataFrame(
            {
                "A": [
                    Timestamp("2012-11-11 00:00:00+01:00"),
                    Timestamp("2012-11-11 00:00:00+01:00"),
                ]
            }
        )
        # 使用后向填充NaN值
        res = df.bfill()
        # 验证填充后的结果与期望是否相同
        tm.assert_frame_equal(res, exp)
    def test_fillna_tzaware_different_column(self):
        # 定义一个测试函数，用于测试在不同列中填充缺失值时的行为
        # GH#15522 表示这个测试与GitHub上的Issue编号15522相关联
        df = DataFrame(
            {
                "A": date_range("20130101", periods=4, tz="US/Eastern"),
                "B": [1, 2, np.nan, np.nan],
            }
        )
        # 对DataFrame进行前向填充缺失值
        result = df.ffill()
        # 期望的结果DataFrame
        expected = DataFrame(
            {
                "A": date_range("20130101", periods=4, tz="US/Eastern"),
                "B": [1.0, 2.0, 2.0, 2.0],
            }
        )
        # 断言两个DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_na_actions_categorical(self):
        # 创建一个包含分类数据和缺失值的DataFrame，用于测试不同的缺失值处理方法
        cat = Categorical([1, 2, 3, np.nan], categories=[1, 2, 3])
        vals = ["a", "b", np.nan, "d"]
        df = DataFrame({"cats": cat, "vals": vals})
        # 创建期望填充缺失值后的DataFrame
        cat2 = Categorical([1, 2, 3, 3], categories=[1, 2, 3])
        vals2 = ["a", "b", "b", "d"]
        df_exp_fill = DataFrame({"cats": cat2, "vals": vals2})
        # 创建期望删除特定类别后的DataFrame
        cat3 = Categorical([1, 2, 3], categories=[1, 2, 3])
        vals3 = ["a", "b", np.nan]
        df_exp_drop_cats = DataFrame({"cats": cat3, "vals": vals3})
        # 创建期望删除所有含有缺失值的行后的DataFrame
        cat4 = Categorical([1, 2], categories=[1, 2, 3])
        vals4 = ["a", "b"]
        df_exp_drop_all = DataFrame({"cats": cat4, "vals": vals4})

        # 使用指定值填充缺失值
        res = df.fillna(value={"cats": 3, "vals": "b"})
        # 断言填充后的结果是否与期望的DataFrame相等
        tm.assert_frame_equal(res, df_exp_fill)

        # 使用不允许的新类别值进行填充，验证是否触发TypeError异常
        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(TypeError, match=msg):
            df.fillna(value={"cats": 4, "vals": "c"})

        # 对DataFrame进行前向填充缺失值
        res = df.ffill()
        # 断言填充后的结果是否与期望的DataFrame相等
        tm.assert_frame_equal(res, df_exp_fill)

        # 删除包含特定列缺失值的行
        res = df.dropna(subset=["cats"])
        # 断言删除后的结果是否与期望的DataFrame相等
        tm.assert_frame_equal(res, df_exp_drop_cats)

        # 删除所有含有缺失值的行
        res = df.dropna()
        # 断言删除后的结果是否与期望的DataFrame相等
        tm.assert_frame_equal(res, df_exp_drop_all)

        # 确保fillna方法正确处理缺失值
        c = Categorical([np.nan, "b", np.nan], categories=["a", "b"])
        df = DataFrame({"cats": c, "vals": [1, 2, 3]})
        # 创建期望填充缺失值后的DataFrame
        cat_exp = Categorical(["a", "b", "a"], categories=["a", "b"])
        df_exp = DataFrame({"cats": cat_exp, "vals": [1, 2, 3]})
        # 使用指定值填充缺失值
        res = df.fillna("a")
        # 断言填充后的结果是否与期望的DataFrame相等
        tm.assert_frame_equal(res, df_exp)
    def test_fillna_categorical_nan(self):
        # GH#14021
        # np.nan should always be a valid filler
        # 创建一个包含 Categorical 数据的对象，包括两个 np.nan 和一个整数 2
        cat = Categorical([np.nan, 2, np.nan])
        # 创建一个包含 Categorical 数据的对象，全部是 np.nan
        val = Categorical([np.nan, np.nan, np.nan])
        # 创建一个 DataFrame 对象，包含两列数据，cats 列使用 cat，vals 列使用 val
        df = DataFrame({"cats": cat, "vals": val})

        # GH#32950 df.median() is poorly behaved because there is no
        #  Categorical.median
        # 创建一个包含两个键值对的 Series 对象，cats 键对应值为 2.0，vals 键对应值为 np.nan
        median = Series({"cats": 2.0, "vals": np.nan})

        # 对 DataFrame 进行填充操作，用 median 中的值来填充缺失值
        res = df.fillna(median)
        # 预期的 vals 列数据应该都是 np.nan，cats 列数据都是 2
        v_exp = [np.nan, np.nan, np.nan]
        df_exp = DataFrame({"cats": [2, 2, 2], "vals": v_exp}, dtype="category")
        # 断言填充后的结果和预期的 DataFrame 相等
        tm.assert_frame_equal(res, df_exp)

        # 对 cats 列进行填充，用 np.nan 来填充缺失值
        result = df.cats.fillna(np.nan)
        # 断言填充后的结果和原始的 df.cats 列相等
        tm.assert_series_equal(result, df.cats)

        # 对 vals 列进行填充，用 np.nan 来填充缺失值
        result = df.vals.fillna(np.nan)
        # 断言填充后的结果和原始的 df.vals 列相等
        tm.assert_series_equal(result, df.vals)

        # 创建一个包含 DatetimeIndex 的对象，包含五个元素，其中两个是 NaT
        idx = DatetimeIndex(
            ["2011-01-01 09:00", "2016-01-01 23:45", "2011-01-01 09:00", NaT, NaT]
        )
        # 创建一个 DataFrame 对象，包含一列 a，使用 Categorical(idx) 来填充
        df = DataFrame({"a": Categorical(idx)})
        # 断言填充后的结果和原始的 DataFrame 相等
        tm.assert_frame_equal(df.fillna(value=NaT), df)

        # 创建一个包含 PeriodIndex 的对象，包含五个元素，其中两个是 NaT
        idx = PeriodIndex(["2011-01", "2011-01", "2011-01", NaT, NaT], freq="M")
        # 创建一个 DataFrame 对象，包含一列 a，使用 Categorical(idx) 来填充
        df = DataFrame({"a": Categorical(idx)})
        # 断言填充后的结果和原始的 DataFrame 相等
        tm.assert_frame_equal(df.fillna(value=NaT), df)

        # 创建一个包含 TimedeltaIndex 的对象，包含五个元素，其中两个是 NaT
        idx = TimedeltaIndex(["1 days", "2 days", "1 days", NaT, NaT])
        # 创建一个 DataFrame 对象，包含一列 a，使用 Categorical(idx) 来填充
        df = DataFrame({"a": Categorical(idx)})
        # 断言填充后的结果和原始的 DataFrame 相等
        tm.assert_frame_equal(df.fillna(value=NaT), df)

    def test_fillna_no_downcast(self, frame_or_series):
        # GH#45603 preserve object dtype
        # 创建一个对象，包含三个整数，数据类型为 "object"
        obj = frame_or_series([1, 2, 3], dtype="object")
        # 对该对象进行填充操作，用空字符串来填充缺失值
        result = obj.fillna("")
        # 断言填充后的结果和原始的对象相等
        tm.assert_equal(result, obj)

    @pytest.mark.parametrize("columns", [["A", "A", "B"], ["A", "A"]])
    def test_fillna_dictlike_value_duplicate_colnames(self, columns):
        # GH#43476
        # 创建一个 DataFrame 对象，全部元素为 NaN，行索引为 [0, 1]，列索引为 columns
        df = DataFrame(np.nan, index=[0, 1], columns=columns)
        # 忽略警告进行填充操作，用字典形式 {"A": 0} 来填充缺失值
        with tm.assert_produces_warning(None):
            result = df.fillna({"A": 0})

        # 创建一个预期的 DataFrame 对象，将原始的 df 复制一份，将 "A" 列填充为 0.0
        expected = df.copy()
        expected["A"] = 0.0
        # 断言填充后的结果和预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_fillna_dtype_conversion(self, using_infer_string):
        # make sure that fillna on an empty frame works
        # 创建一个空的 DataFrame 对象，行索引为 ["A", "B", "C"]，列索引为 [1, 2, 3, 4, 5]
        df = DataFrame(index=["A", "B", "C"], columns=[1, 2, 3, 4, 5])
        # 获取 DataFrame 中各列的数据类型
        result = df.dtypes
        # 创建一个预期的 Series 对象，数据类型全部为 "object"
        expected = Series([np.dtype("object")] * 5, index=[1, 2, 3, 4, 5])
        # 断言获取的结果和预期的数据类型相等
        tm.assert_series_equal(result, expected)
        # 对空的 DataFrame 进行填充操作，用整数 1 来填充缺失值
        result = df.fillna(1)
        # 创建一个预期的 DataFrame 对象，全部元素为 1，行索引为 ["A", "B", "C"]，列索引为 [1, 2, 3, 4, 5]，数据类型为 "object"
        expected = DataFrame(
            1, index=["A", "B", "C"], columns=[1, 2, 3, 4, 5], dtype=object
        )
        # 断言填充后的结果和预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 创建一个空的 DataFrame 对象，行索引为 range(3)，列索引为 ["A", "B"]，数据类型为 "float64"
        df = DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
        # 对该 DataFrame 进行填充操作，用字符串 "nan" 来填充缺失值
        result = df.fillna("nan")
        # 创建一个预期的 DataFrame 对象，全部元素为字符串 "nan"，行索引为 range(3)，列索引为 ["A", "B"]
        expected = DataFrame("nan", index=range(3), columns=["A", "B"])
        # 断言填充后的结果和预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试在填充 NaN 值时，数据类型转换与使用替换方法的等效性
    def test_fillna_dtype_conversion_equiv_replace(self, val):
        # 创建一个 DataFrame，包含两列"A"和"B"，"A"列有一个 NaN 值
        df = DataFrame({"A": [1, np.nan], "B": [1.0, 2.0]})
        # 用 val 替换 DataFrame 中的 NaN 值，作为预期结果
        expected = df.replace(np.nan, val)
        # 使用 fillna 方法填充 DataFrame 中的 NaN 值，生成结果
        result = df.fillna(val)
        # 断言填充后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试在填充日期时间列时的行为
    def test_fillna_datetime_columns(self):
        # 创建一个 DataFrame，包含四列"A"、"B"、"C"和"D"，其中"A"列有一个 NaN 值
        df = DataFrame(
            {
                "A": [-1, -2, np.nan],
                "B": date_range("20130101", periods=3),
                "C": ["foo", "bar", None],
                "D": ["foo2", "bar2", None],
            },
            index=date_range("20130110", periods=3),
        )
        # 使用 "?" 填充 DataFrame 中的 NaN 值，生成结果
        result = df.fillna("?")
        # 创建预期的 DataFrame，用 "?" 替换 NaN 值
        expected = DataFrame(
            {
                "A": [-1, -2, "?"],
                "B": date_range("20130101", periods=3),
                "C": ["foo", "bar", "?"],
                "D": ["foo2", "bar2", "?"],
            },
            index=date_range("20130110", periods=3),
        )
        # 断言填充后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建另一个 DataFrame，包含四列"A"、"B"、"C"和"D"，其中"A"列有一个 NaN 值
        df = DataFrame(
            {
                "A": [-1, -2, np.nan],
                "B": [Timestamp("2013-01-01"), Timestamp("2013-01-02"), NaT],
                "C": ["foo", "bar", None],
                "D": ["foo2", "bar2", None],
            },
            index=date_range("20130110", periods=3),
        )
        # 使用 "?" 填充 DataFrame 中的 NaN 值，生成结果
        result = df.fillna("?")
        # 创建预期的 DataFrame，用 "?" 替换 NaN 值
        expected = DataFrame(
            {
                "A": [-1, -2, "?"],
                "B": [Timestamp("2013-01-01"), Timestamp("2013-01-02"), "?"],
                "C": ["foo", "bar", "?"],
                "D": ["foo2", "bar2", "?"],
            },
            index=date_range("20130110", periods=3),
        )
        # 断言填充后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，测试 DataFrame 对象的前向填充行为
    def test_ffill(self, datetime_frame):
        # 将 datetime_frame 的前五行"A"列设为 NaN
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        # 将 datetime_frame 的最后五行"A"列设为 NaN
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan
        # 使用前向填充方法填充 DataFrame，生成结果
        alt = datetime_frame.ffill()
        # 断言使用前向填充方法两次得到的结果相等
        tm.assert_frame_equal(datetime_frame.ffill(), alt)

    # 定义一个测试方法，测试 DataFrame 对象的后向填充行为
    def test_bfill(self, datetime_frame):
        # 将 datetime_frame 的前五行"A"列设为 NaN
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        # 将 datetime_frame 的最后五行"A"列设为 NaN
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan
        # 使用后向填充方法填充 DataFrame，生成结果
        alt = datetime_frame.bfill()
        # 断言使用后向填充方法两次得到的结果相等
        tm.assert_frame_equal(datetime_frame.bfill(), alt)

    # 定义一个测试方法，测试在重新索引并限制填充和后向填充的行为
    def test_frame_pad_backfill_limit(self):
        # 创建一个 10x4 大小的 DataFrame，填充随机正态分布的数据
        index = np.arange(10)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=index)
        
        # 使用 "pad" 方法和限制数值 5 填充前两行，生成结果
        result = df[:2].reindex(index, method="pad", limit=5)
        # 创建预期的 DataFrame，使用前向填充来填充前两行，并在末尾三行加入 NaN 值
        expected = df[:2].reindex(index).ffill()
        expected.iloc[-3:] = np.nan
        # 断言填充后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 使用 "backfill" 方法和限制数值 5 后向填充最后两行，生成结果
        result = df[-2:].reindex(index, method="backfill", limit=5)
        # 创建预期的 DataFrame，使用后向填充来填充最后两行，并在开头三行加入 NaN 值
        expected = df[-2:].reindex(index).bfill()
        expected.iloc[:3] = np.nan
        # 断言填充后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，测试DataFrame的fillna方法在限制条件下的行为
    def test_frame_fillna_limit(self):
        # 创建一个包含 0 到 9 的索引
        index = np.arange(10)
        # 创建一个形状为 (10, 4) 的随机数DataFrame，索引为上述创建的索引
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=index)

        # 从df的前两行创建一个新的DataFrame，然后重新索引为原始索引，并填充缺失值，最多填充5个值
        result = df[:2].reindex(index)
        result = result.ffill(limit=5)

        # 从df的最后两行创建一个新的DataFrame，然后重新索引为原始索引，并向后填充缺失值，最多填充5个值
        result = df[-2:].reindex(index)
        result = result.bfill(limit=5)

    # 定义一个测试方法，测试DataFrame的fillna方法在跳过特定数据块时的行为
    def test_fillna_skip_certain_blocks(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)).astype(int))

        # 使用np.nan填充整个DataFrame，即使原始数据块是布尔值或整数类型
        df.fillna(np.nan)

    # 使用pytest参数化装饰器，测试在正整数限制条件下DataFrame的fillna方法的行为
    @pytest.mark.parametrize("type", [int, float])
    def test_fillna_positive_limit(self, type):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4))).astype(type)

        # 当限制条件小于等于0时，填充值必须引发ValueError异常，匹配错误消息"Limit must be greater than 0"
        msg = "Limit must be greater than 0"
        with pytest.raises(ValueError, match=msg):
            df.fillna(0, limit=-5)

    # 使用pytest参数化装饰器，测试在整数限制条件下DataFrame的fillna方法的行为
    @pytest.mark.parametrize("type", [int, float])
    def test_fillna_integer_limit(self, type):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4))).astype(type)

        # 当限制条件不是整数时，填充值必须引发ValueError异常，匹配错误消息"Limit must be an integer"
        msg = "Limit must be an integer"
        with pytest.raises(ValueError, match=msg):
            df.fillna(0, limit=0.5)

    # 测试DataFrame的fillna方法的就地填充行为
    def test_fillna_inplace(self):
        # 创建一个形状为 (10, 4) 的随机数DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 将第一列的前五行和第四列的后四行设置为NaN
        df.loc[:4, 1] = np.nan
        df.loc[-4:, 3] = np.nan

        # 用值0填充NaN，并返回新的DataFrame对象，与原始对象不同
        expected = df.fillna(value=0)
        assert expected is not df

        # 就地用值0填充NaN
        df.fillna(value=0, inplace=True)
        tm.assert_frame_equal(df, expected)

        # 就地用字典 {0: 0} 填充NaN，返回None
        expected = df.fillna(value={0: 0}, inplace=True)
        assert expected is None

        # 用前向填充NaN，并返回新的DataFrame对象，与原始对象不同
        expected = df.ffill()
        assert expected is not df

        # 就地用前向填充NaN
        df.ffill(inplace=True)
        tm.assert_frame_equal(df, expected)

    # 测试DataFrame的fillna方法使用字典填充不同列的NaN值的行为
    def test_fillna_dict_series(self):
        # 创建一个包含NaN的DataFrame
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            }
        )

        # 使用字典填充指定列的NaN值
        result = df.fillna({"a": 0, "b": 5})

        # 创建预期的DataFrame，对列'a'和'b'分别使用0和5填充NaN
        expected = df.copy()
        expected["a"] = expected["a"].fillna(0)
        expected["b"] = expected["b"].fillna(5)
        tm.assert_frame_equal(result, expected)

        # 使用字典填充指定列的NaN值，包括不存在的列'd'
        result = df.fillna({"a": 0, "b": 5, "d": 7})

        # Series被当作字典处理
        result = df.fillna(df.max())

        # 用每列的最大值填充NaN，预期结果与使用最大值字典填充NaN相同
        expected = df.fillna(df.max().to_dict())
        tm.assert_frame_equal(result, expected)

        # 目前禁用这个功能
        with pytest.raises(NotImplementedError, match="column by column"):
            df.fillna(df.max(axis=1), axis=1)
    # 定义一个测试方法，用于测试填充 DataFrame 中的空值操作
    def test_fillna_dataframe(self):
        # GH#8377：引用 GitHub 上的 issue 编号
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],  # 创建包含 NaN 值的列 'a'
                "b": [1, 2, 3, np.nan, np.nan],  # 创建包含 NaN 值的列 'b'
                "c": [np.nan, 1, 2, 3, 4],  # 创建包含 NaN 值的列 'c'
            },
            index=list("VWXYZ"),  # 设置索引为 'VWXYZ'
        )

        # df2 可能具有不同的索引和列
        df2 = DataFrame(
            {
                "a": [np.nan, 10, 20, 30, 40],  # 创建包含 NaN 值的列 'a'
                "b": [50, 60, 70, 80, 90],  # 创建列 'b'，填充数字
                "foo": ["bar"] * 5,  # 创建名为 'foo' 的列，并填充字符串 'bar'
            },
            index=list("VWXuZ"),  # 设置索引为 'VWXuZ'
        )

        result = df.fillna(df2)  # 使用 df2 来填充 df 中的 NaN 值

        # 只有共享的列和索引会被填充
        expected = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, 40],  # 期望的列 'a' 值
                "b": [1, 2, 3, np.nan, 90],  # 期望的列 'b' 值
                "c": [np.nan, 1, 2, 3, 4],  # 期望的列 'c' 值
            },
            index=list("VWXYZ"),  # 期望的索引为 'VWXYZ'
        )

        tm.assert_frame_equal(result, expected)  # 断言填充后的结果与期望相同

    # 定义一个测试方法，用于测试在 DataFrame 中按列填充空值的操作
    def test_fillna_columns(self):
        arr = np.random.default_rng(2).standard_normal((10, 10))  # 创建一个 10x10 的随机数组，含 NaN
        arr[:, ::2] = np.nan  # 每一行的偶数列设置为 NaN
        df = DataFrame(arr)  # 使用随机数组创建 DataFrame

        result = df.ffill(axis=1)  # 按行前向填充 NaN 值
        expected = df.T.ffill().T  # 转置后按列前向填充 NaN 值再转置回来

        tm.assert_frame_equal(result, expected)  # 断言填充后的结果与期望相同

        df.insert(6, "foo", 5)  # 在第 6 列插入名为 'foo' 的列，并填充值为 5
        result = df.ffill(axis=1)  # 再次按行前向填充 NaN 值
        expected = df.astype(float).ffill(axis=1)  # 将 DataFrame 转换为 float 类型后按行前向填充 NaN 值

        tm.assert_frame_equal(result, expected)  # 断言填充后的结果与期望相同

    # 定义一个测试方法，用于测试填充 DataFrame 中无效值的情况
    def test_fillna_invalid_value(self, float_frame):
        # list 类型的值填充，期望抛出 TypeError 异常
        msg = '"value" parameter must be a scalar or dict, but you passed a "{}"'
        with pytest.raises(TypeError, match=msg.format("list")):
            float_frame.fillna([1, 2])

        # tuple 类型的值填充，期望抛出 TypeError 异常
        with pytest.raises(TypeError, match=msg.format("tuple")):
            float_frame.fillna((1, 2))

        # DataFrame 类型的值填充，期望抛出 TypeError 异常
        msg = (
            '"value" parameter must be a scalar, dict or Series, but you '
            'passed a "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            float_frame.iloc[:, 0].fillna(float_frame)

    # 定义一个测试方法，用于测试填充 DataFrame 中列重新排序的情况
    def test_fillna_col_reordering(self):
        cols = ["COL." + str(i) for i in range(5, 0, -1)]  # 创建列名为 'COL.5' 到 'COL.1'
        data = np.random.default_rng(2).random((20, 5))  # 创建一个 20x5 的随机数组
        df = DataFrame(index=range(20), columns=cols, data=data)  # 使用随机数组创建 DataFrame
        filled = df.ffill()  # 对 DataFrame 进行前向填充 NaN 值
        assert df.columns.tolist() == filled.columns.tolist()  # 断言填充后的列名顺序与原始 DataFrame 相同

    # 使用 pytest.mark.xfail 标记的测试方法，用于测试在特定情况下填充 DataFrame 的行为
    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't fill 0 in string")
    def test_fill_corner(self, float_frame, float_string_frame):
        mf = float_string_frame  # 使用 float_string_frame 进行测试
        mf.loc[mf.index[5:20], "foo"] = np.nan  # 将索引为 5 到 19 的行的 'foo' 列设置为 NaN
        mf.loc[mf.index[-10:], "A"] = np.nan  # 将倒数第 10 行到最后一行的 'A' 列设置为 NaN

        filled = float_string_frame.fillna(value=0)  # 使用值 0 填充 DataFrame 中的 NaN 值
        assert (filled.loc[filled.index[5:20], "foo"] == 0).all()  # 断言索引为 5 到 19 的行的 'foo' 列都填充为 0
        del float_string_frame["foo"]  # 删除 DataFrame 中的 'foo' 列

        float_frame.reindex(columns=[]).fillna(value=0)  # 重新索引为一个空列表后，使用值 0 填充 NaN 值
    # 定义一个测试方法，用于测试在指定轴上填充缺失值，并限制填充的次数和填充的值
    def test_fillna_with_columns_and_limit(self):
        # GH40989: 此处是 GitHub issue #40989 的相关测试
        # 创建一个 DataFrame 对象，包含带有缺失值的数据和列名
        df = DataFrame(
            [
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5],
                [np.nan, 3, np.nan, 4],
            ],
            columns=list("ABCD"),
        )
        # 在指定轴上使用不同的值进行填充，并限制填充的次数
        result = df.fillna(axis=1, value=100, limit=1)
        result2 = df.fillna(axis=1, value=100, limit=2)

        # 预期结果 DataFrame，包含填充后的值和原始索引
        expected = DataFrame(
            {
                "A": Series([100, 3, 100, 100], dtype="float64"),
                "B": [2, 4, np.nan, 3],
                "C": [np.nan, 100, np.nan, np.nan],
                "D": Series([0, 1, 5, 4], dtype="float64"),
            },
            index=[0, 1, 2, 3],
        )
        expected2 = DataFrame(
            {
                "A": Series([100, 3, 100, 100], dtype="float64"),
                "B": Series([2, 4, 100, 3], dtype="float64"),
                "C": [100, 100, np.nan, 100],
                "D": Series([0, 1, 5, 4], dtype="float64"),
            },
            index=[0, 1, 2, 3],
        )

        # 断言填充后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected2)

    # 定义一个测试方法，测试在 DataFrame 中使用 inplace 参数进行日期填充
    def test_fillna_datetime_inplace(self):
        # GH#48863: 此处是 GitHub issue #48863 的相关测试
        # 创建一个包含日期数据的 DataFrame
        df = DataFrame(
            {
                "date1": to_datetime(["2018-05-30", None]),
                "date2": to_datetime(["2018-09-30", None]),
            }
        )
        # 复制预期结果
        expected = df.copy()
        # 使用 inplace=True 进行填充缺失值
        df.fillna(np.nan, inplace=True)
        # 断言填充后的 DataFrame 与预期结果相等
        tm.assert_frame_equal(df, expected)

    # 定义一个测试方法，测试在 DataFrame 中使用 inplace 参数进行指定轴上的填充，并限制填充次数和填充的值
    def test_fillna_inplace_with_columns_limit_and_value(self):
        # GH40989: 此处是 GitHub issue #40989 的相关测试
        # 创建一个 DataFrame 对象，包含带有缺失值的数据和列名
        df = DataFrame(
            [
                [np.nan, 2, np.nan, 0],
                [3, 4, np.nan, 1],
                [np.nan, np.nan, np.nan, 5],
                [np.nan, 3, np.nan, 4],
            ],
            columns=list("ABCD"),
        )
        # 用于在指定轴上填充缺失值，并限制填充的次数和填充的值，并返回填充后的 DataFrame
        expected = df.fillna(axis=1, value=100, limit=1)
        # 确保填充后的结果与原始对象不是同一个对象
        assert expected is not df
        # 使用 inplace=True 在原始 DataFrame 上进行填充
        df.fillna(axis=1, value=100, limit=1, inplace=True)
        # 断言填充后的 DataFrame 与预期结果相等
        tm.assert_frame_equal(df, expected)

    # 定义一个参数化测试方法，测试在 DataFrame 中使用 inplace 参数进行字典更新的填充
    @pytest.mark.parametrize("val", [-1, {"x": -1, "y": -1}])
    def test_inplace_dict_update_view(self, val):
        # GH#47188: 此处是 GitHub issue #47188 的相关测试
        # 创建一个包含 NaN 值的 DataFrame
        df = DataFrame({"x": [np.nan, 2], "y": [np.nan, 2]})
        # 复制原始 DataFrame
        df_orig = df.copy()
        # 创建一个 DataFrame 视图
        result_view = df[:]
        # 使用 inplace=True 进行填充缺失值，并更新 DataFrame
        df.fillna(val, inplace=True)
        # 预期的填充结果 DataFrame
        expected = DataFrame({"x": [-1, 2.0], "y": [-1.0, 2]})
        # 断言填充后的 DataFrame 与预期结果相等
        tm.assert_frame_equal(df, expected)
        # 断言结果视图与原始 DataFrame 相等
        tm.assert_frame_equal(result_view, df_orig)
    def test_single_block_df_with_horizontal_axis(self):
        # 测试单个数据块的 DataFrame 在水平轴上填充缺失值的情况
        df = DataFrame(
            {
                "col1": [5, 0, np.nan, 10, np.nan],
                "col2": [7, np.nan, np.nan, 5, 3],
                "col3": [12, np.nan, 1, 2, 0],
                "col4": [np.nan, 1, 1, np.nan, 18],
            }
        )
        # 使用指定值填充 DataFrame 中的缺失值，每行最多填充一次，操作在水平轴上
        result = df.fillna(50, limit=1, axis=1)
        # 期望的填充后的 DataFrame
        expected = DataFrame(
            [
                [5.0, 7.0, 12.0, 50.0],
                [0.0, 50.0, np.nan, 1.0],
                [50.0, np.nan, 1.0, 1.0],
                [10.0, 5.0, 2.0, 50.0],
                [50.0, 3.0, 0.0, 18.0],
            ],
            columns=["col1", "col2", "col3", "col4"],
        )
        # 检验填充后的结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    def test_fillna_with_multi_index_frame(self):
        # 测试多层索引 DataFrame 的填充缺失值情况
        pdf = DataFrame(
            {
                ("x", "a"): [np.nan, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, np.nan],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        # 期望的填充后的 DataFrame
        expected = DataFrame(
            {
                ("x", "a"): [-1.0, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, -1.0],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        # 使用指定的字典填充 DataFrame 中的缺失值，检查结果是否与期望一致
        tm.assert_frame_equal(pdf.fillna({"x": -1}), expected)
        # 使用不同的填充字典再次检查填充后的结果是否与期望一致
        tm.assert_frame_equal(pdf.fillna({"x": -1, ("x", "b"): -2}), expected)

        expected = DataFrame(
            {
                ("x", "a"): [-1.0, 2.0, 3.0],
                ("x", "b"): [1.0, 2.0, -2.0],
                ("y", "c"): [1.0, 2.0, np.nan],
            }
        )
        # 使用不同的填充字典再次检查填充后的结果是否与期望一致
        tm.assert_frame_equal(pdf.fillna({("x", "b"): -2, "x": -1}), expected)
# 定义测试函数，用于测试填充缺失值的功能，特别是非合并的框架
def test_fillna_nonconsolidated_frame():
    # 引用GitHub上的问题链接，描述了此测试函数的背景
    # https://github.com/pandas-dev/pandas/issues/36495
    # 创建一个DataFrame对象，包含几行数据和列名
    df = DataFrame(
        [
            [1, 1, 1, 1.0],
            [2, 2, 2, 2.0],
            [3, 3, 3, 3.0],
        ],
        columns=["i1", "i2", "i3", "f1"],
    )
    # 使用pivot方法将DataFrame进行重塑，设定i1和i2为索引和列
    df_nonconsol = df.pivot(index="i1", columns="i2")
    # 对重塑后的DataFrame进行缺失值填充，将缺失值用0替换
    result = df_nonconsol.fillna(0)
    # 断言：确认填充后的结果中不存在任何缺失值
    assert result.isna().sum().sum() == 0


def test_fillna_nones_inplace():
    # GH 48480
    # 创建一个包含空值的DataFrame，用于测试fillna方法的inplace参数
    df = DataFrame(
        [[None, None], [None, None]],
        columns=["A", "B"],
    )
    # 使用fillna方法填充空值，通过字典指定每列对应的填充值，inplace=True表示在原对象上进行修改
    df.fillna(value={"A": 1, "B": 2}, inplace=True)

    # 预期的DataFrame结果，所有空值被填充为指定的值
    expected = DataFrame([[1, 2], [1, 2]], columns=["A", "B"], dtype=object)
    # 使用assert_frame_equal函数检查填充后的DataFrame与预期结果是否一致
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "data, expected_data, method, kwargs",
    (
        # 第一个例子：输入列表1和期望输出列表2，使用 "ffill" 方法填充缺失值
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside"},  # 附加参数指定填充限制区域为内部
        ),
        # 第二个例子：输入列表1和期望输出列表2，使用 "ffill" 方法填充缺失值，限制填充次数为1
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 3.0, np.nan, np.nan, 7.0, np.nan, np.nan],
            "ffill",
            {"limit_area": "inside", "limit": 1},  # 附加参数指定填充限制区域为内部，并设置填充次数为1
        ),
        # 第三个例子：输入列表1和期望输出列表2，使用 "ffill" 方法填充缺失值，限制填充区域为外部
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, 7.0],
            "ffill",
            {"limit_area": "outside"},  # 附加参数指定填充限制区域为外部
        ),
        # 第四个例子：输入列表1和期望输出列表2，使用 "ffill" 方法填充缺失值，限制填充区域为外部，并设置填充次数为1
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, 7.0, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},  # 附加参数指定填充限制区域为外部，并设置填充次数为1
        ),
        # 第五个例子：输入全为 NaN 的列表1和期望输出全为 NaN 的列表2，使用 "ffill" 方法填充缺失值，限制填充区域为外部，并设置填充次数为1
        (
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ffill",
            {"limit_area": "outside", "limit": 1},  # 附加参数指定填充限制区域为外部，并设置填充次数为1
        ),
        # 第六个例子：输入为 0 到 4 的 range，期望输出为相同的 range，使用 "ffill" 方法填充缺失值，限制填充区域为外部，并设置填充次数为1
        (
            range(5),
            range(5),
            "ffill",
            {"limit_area": "outside", "limit": 1},  # 附加参数指定填充限制区域为外部，并设置填充次数为1
        ),
        # 第七个例子：输入列表1和期望输出列表2，使用 "bfill" 方法填充缺失值，限制填充区域为内部
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, 7.0, 7.0, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside"},  # 附加参数指定填充限制区域为内部
        ),
        # 第八个例子：输入列表1和期望输出列表2，使用 "bfill" 方法填充缺失值，限制填充区域为内部，并设置填充次数为1
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, 3.0, np.nan, np.nan, 7.0, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "inside", "limit": 1},  # 附加参数指定填充限制区域为内部，并设置填充次数为1
        ),
        # 第九个例子：输入列表1和期望输出列表2，使用 "bfill" 方法填充缺失值，限制填充区域为外部
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside"},  # 附加参数指定填充限制区域为外部
        ),
        # 第十个例子：输入列表1和期望输出列表2，使用 "bfill" 方法填充缺失值，限制填充区域为外部，并设置填充次数为1
        (
            [np.nan, np.nan, 3, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, 3.0, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan, np.nan],
            "bfill",
            {"limit_area": "outside", "limit": 1},  # 附加参数指定填充限制区域为外部，并设置填充次数为1
        ),
    ),
# 定义一个测试函数，用于测试 DataFrame 对象的填充方法（前向填充、后向填充、限制填充区域）
def test_ffill_bfill_limit_area(data, expected_data, method, kwargs):
    # GH#56492
    # 创建 DataFrame 对象，使用给定的数据
    df = DataFrame(data)
    # 创建期望的 DataFrame 对象，使用期望的数据
    expected = DataFrame(expected_data)
    # 调用 DataFrame 对象的指定填充方法，使用给定的参数
    result = getattr(df, method)(**kwargs)
    # 断言结果 DataFrame 与期望的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


# 使用参数化装饰器，定义一个参数化测试函数，用于测试填充 NaN 值时的行为
@pytest.mark.parametrize("test_frame", [True, False])
@pytest.mark.parametrize("dtype", ["float", "object"])
def test_fillna_with_none_object(test_frame, dtype):
    # GH#57723
    # 创建一个 Series 对象，包含整数、NaN 和整数，使用给定的 dtype 类型
    obj = Series([1, np.nan, 3], dtype=dtype)
    # 如果 test_frame 为 True，则将 Series 对象转换为 DataFrame 对象
    if test_frame:
        obj = obj.to_frame()
    # 调用 fillna 方法，将 NaN 值填充为 None
    result = obj.fillna(value=None)
    # 创建期望的 Series 对象，将 NaN 值填充为 None，使用给定的 dtype 类型
    expected = Series([1, None, 3], dtype=dtype)
    # 如果 test_frame 为 True，则将期望的 Series 对象转换为 DataFrame 对象
    if test_frame:
        expected = expected.to_frame()
    # 断言结果对象与期望对象相等
    tm.assert_equal(result, expected)
```