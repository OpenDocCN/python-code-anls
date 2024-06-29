# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_indexing.py`

```
from collections import namedtuple
from datetime import (
    datetime,
    timedelta,
)
from decimal import Decimal
import re

import numpy as np
import pytest

from pandas._libs import iNaT
from pandas.errors import InvalidIndexError

from pandas.core.dtypes.common import is_integer

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    to_datetime,
)
import pandas._testing as tm

# We pass through a TypeError raised by numpy
_slice_msg = "slice indices must be integers or None or have an __index__ method"


class TestDataFrameIndexing:
    # 测试数据帧的索引功能
    def test_getitem(self, float_frame):
        # 切片操作
        sl = float_frame[:20]
        assert len(sl.index) == 20

        # 遍历每列
        for _, series in sl.items():
            assert len(series.index) == 20
            tm.assert_index_equal(series.index, sl.index)

        # 遍历数据帧的每一列
        for key, _ in float_frame._series.items():
            assert float_frame[key] is not None

        # 检查不存在的列
        assert "random" not in float_frame
        # 用 pytest 检查是否抛出 KeyError 异常，且异常消息包含 "random"
        with pytest.raises(KeyError, match="random"):
            float_frame["random"]

    # 测试使用数字进行索引，不应该回退到位置索引
    def test_getitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
        # GH51053
        dtype = any_numeric_dtype
        idx = Index([1, 0, 1], dtype=dtype)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
        result = df[1]
        expected = DataFrame([[1, 3], [4, 6]], columns=Index([1, 1], dtype=dtype))
        tm.assert_frame_equal(result, expected, check_exact=True)

    def test_getitem2(self, float_frame):
        df = float_frame.copy()
        
        # 添加名为 "$10" 的列，并赋予随机数值
        df["$10"] = np.random.default_rng(2).standard_normal(len(df))

        # 添加名为 "@awesome_domain" 的列，并赋予随机数值
        ad = np.random.default_rng(2).standard_normal(len(df))
        df["@awesome_domain"] = ad

        # 使用 pytest 检查是否抛出 KeyError 异常，异常消息中包含 "'df[\"$10\"]'"
        with pytest.raises(KeyError, match=re.escape("'df[\"$10\"]'")):
            df.__getitem__('df["$10"]')

        # 获取 "@awesome_domain" 列的值，与预期的随机数组 ad 进行比较
        res = df["@awesome_domain"]
        tm.assert_numpy_array_equal(ad, res.values)

    # 测试设置数字索引时不应该回退到位置索引
    def test_setitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
        # GH51053
        dtype = any_numeric_dtype
        idx = Index([1, 0, 1], dtype=dtype)
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
        
        # 设置 df[1] = 10
        df[1] = 10
        
        # 设置预期结果
        expected = DataFrame([[10, 2, 10], [10, 5, 10]], columns=idx)
        tm.assert_frame_equal(df, expected, check_exact=True)
    # 在 float_frame 中添加新的列 'E'，赋值为字符串 "foo"
    float_frame["E"] = "foo"
    # 从 float_frame 中选择列 'A' 和 'B'，生成一个新的 DataFrame data
    data = float_frame[["A", "B"]]
    # 将 data 中的列 'A' 和 'B' 的顺序交换，并赋值给 float_frame 的相应列
    float_frame[["B", "A"]] = data

    # 使用断言验证 float_frame 中的列 'B' 是否与 data 中的列 'A' 相等，不检查列名
    tm.assert_series_equal(float_frame["B"], data["A"], check_names=False)
    # 使用断言验证 float_frame 中的列 'A' 是否与 data 中的列 'B' 相等，不检查列名
    tm.assert_series_equal(float_frame["A"], data["B"], check_names=False)

    # 准备错误消息，用于指示在将 'A' 列的长度与 float_frame 的 ['A', 'B'] 列长度不同时抛出 ValueError
    msg = "Columns must be same length as key"
    # 使用断言验证在赋值时 data['A'] = float_frame[['A', 'B']] 是否会引发 ValueError，匹配错误消息
    with pytest.raises(ValueError, match=msg):
        data[["A"]] = float_frame[["A", "B"]]

    # 准备新列数据，其长度比 data.index 少 1
    newcolumndata = range(len(data.index) - 1)
    # 准备错误消息，用于指示在长度不匹配时抛出 ValueError
    msg = (
        rf"Length of values \({len(newcolumndata)}\) "
        rf"does not match length of index \({len(data)}\)"
    )
    # 使用断言验证在赋值时 data['A'] = newcolumndata 是否会引发 ValueError，匹配错误消息
    with pytest.raises(ValueError, match=msg):
        data["A"] = newcolumndata

# 创建一个测试方法 test_setitem_list2，不接受任何参数
def test_setitem_list2():
    # 创建一个 DataFrame df，所有元素为整数 0，行索引为 range(3)，列为 ["tt1", "tt2"]
    df = DataFrame(0, index=range(3), columns=["tt1", "tt2"], dtype=int)
    # 在 df 的第二行，列 'tt1' 和 'tt2' 处分别赋值为 1 和 2
    df.loc[1, ["tt1", "tt2"]] = [1, 2]

    # 选择 df 的第二行，并指定列为 ["tt1", "tt2"]，将结果保存到 result
    result = df.loc[df.index[1], ["tt1", "tt2"]]
    # 创建一个预期的 Series expected，内容为 [1, 2]，列名与 df 的列名相同，数据类型为整数，名称为 1
    expected = Series([1, 2], df.columns, dtype=int, name=1)
    # 使用断言验证 result 是否等于 expected
    tm.assert_series_equal(result, expected)

    # 将 df 的列 'tt1' 和 'tt2' 全部赋值为字符串 "0"
    df["tt1"] = df["tt2"] = "0"
    # 在 df 的第二行，列 'tt1' 和 'tt2' 处分别赋值为字符串 "1" 和 "2"
    df.loc[df.index[1], ["tt1", "tt2"]] = ["1", "2"]
    # 选择 df 的第二行，并指定列为 ["tt1", "tt2"]，将结果保存到 result
    result = df.loc[df.index[1], ["tt1", "tt2"]]
    # 创建一个预期的 Series expected，内容为 ["1", "2"]，列名与 df 的列名相同，名称为 1
    expected = Series(["1", "2"], df.columns, name=1)
    # 使用断言验证 result 是否等于 expected
    tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于测试布尔索引功能
    def test_getitem_boolean(self, mixed_float_frame, mixed_int_frame, datetime_frame):
        # 从 datetime_frame 的索引中选择中间位置的日期时间对象
        d = datetime_frame.index[len(datetime_frame) // 2]
        # 创建一个布尔型索引器，用于选择大于 d 的日期时间索引
        indexer = datetime_frame.index > d
        # 将布尔型索引器转换为对象类型
        indexer_obj = indexer.astype(object)

        # 使用布尔型索引器选择子索引和子框架
        subindex = datetime_frame.index[indexer]
        subframe = datetime_frame[indexer]

        # 断言子索引和子框架的索引相等
        tm.assert_index_equal(subindex, subframe.index)
        # 使用 pytest 断言引发 ValueError 异常，匹配特定错误消息
        with pytest.raises(ValueError, match="Item wrong length"):
            datetime_frame[indexer[:-1]]

        # 使用对象类型的布尔型索引器选择子框架，再次断言框架相等
        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)

        # 使用 pytest 断言引发 TypeError 异常，匹配特定错误消息
        with pytest.raises(TypeError, match="Boolean array expected"):
            datetime_frame[datetime_frame]

        # 将对象类型的布尔型索引器转换为 Series，使用该 Series 选择子框架，再次断言框架相等
        indexer_obj = Series(indexer_obj, datetime_frame.index)
        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)

        # 测试 Series 索引器重新索引功能
        # 生成一个警告，因为传递的布尔键与给定的索引不同，将重新索引
        with tm.assert_produces_warning(UserWarning, match="will be reindexed"):
            # 将索引器重新索引为倒序，选择子框架，并断言框架相等
            indexer_obj = indexer_obj.reindex(datetime_frame.index[::-1])
            subframe_obj = datetime_frame[indexer_obj]
            tm.assert_frame_equal(subframe_obj, subframe)

        # 测试 df[df > 0] 的功能
        for df in [
            datetime_frame,
            mixed_float_frame,
            mixed_int_frame,
        ]:
            # 获取数值数据，创建一个新的布尔型框架
            data = df._get_numeric_data()
            bif = df[df > 0]
            # 用 np.where 创建一个包含 NaN 的 DataFrame
            bifw = DataFrame(
                {c: np.where(data[c] > 0, data[c], np.nan) for c in data.columns},
                index=data.index,
                columns=data.columns,
            )

            # 将其他列添加回来以进行比较
            for c in df.columns:
                if c not in bifw:
                    bifw[c] = df[c]
            # 重新索引列为 df 的列，并与 bif 进行比较
            bifw = bifw.reindex(columns=df.columns)

            # 断言 bif 和 bifw 的框架相等，忽略数据类型的检查
            tm.assert_frame_equal(bif, bifw, check_dtype=False)
            # 对于每列，如果 bif 的数据类型与 bifw 的不同，则断言 bif 的数据类型与 df 的数据类型相同
            for c in df.columns:
                if bif[c].dtype != bifw[c].dtype:
                    assert bif[c].dtype == df[c].dtype
    def test_getitem_boolean_casting(self, datetime_frame):
        # don't upcast if we don't need to
        # 复制传入的日期框架，以避免直接修改原始数据
        df = datetime_frame.copy()

        # 添加新列"E"，并将其值设为1
        df["E"] = 1
        # 将列"E"的数据类型转换为'int32'
        df["E"] = df["E"].astype("int32")
        # 复制列"E"的值到新列"E1"
        df["E1"] = df["E"].copy()

        # 添加新列"F"，并将其值设为1
        df["F"] = 1
        # 将列"F"的数据类型转换为'int64'
        df["F"] = df["F"].astype("int64")
        # 复制列"F"的值到新列"F1"
        df["F1"] = df["F"].copy()

        # 选取大于0的部分数据
        casted = df[df > 0]
        # 获取选取后各列的数据类型
        result = casted.dtypes
        # 期望的数据类型列表
        expected = Series(
            [np.dtype("float64")] * 4
            + [np.dtype("int32")] * 2
            + [np.dtype("int64")] * 2,
            index=["A", "B", "C", "D", "E", "E1", "F", "F1"],
        )
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

        # 对int类型的数据进行切片操作
        df.loc[df.index[1:3], ["E1", "F1"]] = 0
        # 重新选取大于0的部分数据
        casted = df[df > 0]
        # 获取选取后各列的数据类型
        result = casted.dtypes
        # 期望的数据类型列表
        expected = Series(
            [np.dtype("float64")] * 4
            + [np.dtype("int32")]
            + [np.dtype("float64")]
            + [np.dtype("int64")]
            + [np.dtype("float64")],
            index=["A", "B", "C", "D", "E", "E1", "F", "F1"],
        )
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "lst", [[True, False, True], [True, True, True], [False, False, False]]
    )
    def test_getitem_boolean_list(self, lst):
        # 创建一个3行4列的DataFrame，数值为0到11的序列
        df = DataFrame(np.arange(12).reshape(3, 4))
        # 根据布尔列表选取DataFrame中的行
        result = df[lst]
        # 根据布尔列表选取DataFrame中的行，期望的结果
        expected = df.loc[df.index[lst]]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

    def test_getitem_boolean_iadd(self):
        # 创建一个5行5列的随机数组
        arr = np.random.default_rng(2).standard_normal((5, 5))

        # 创建一个DataFrame，使用上述数组作为数据，列名为"A"到"E"
        df = DataFrame(arr.copy(), columns=["A", "B", "C", "D", "E"])

        # 将DataFrame中小于0的元素加1
        df[df < 0] += 1
        # 将数组中小于0的元素加1
        arr[arr < 0] += 1

        # 断言DataFrame的值与数组的值几乎相等
        tm.assert_almost_equal(df.values, arr)

    def test_boolean_index_empty_corner(self):
        # #2096
        # 创建一个空的DataFrame，列名为"A"，索引为空的时间戳索引
        blah = DataFrame(np.empty([0, 1]), columns=["A"], index=DatetimeIndex([]))

        # 创建一个空的布尔数组
        k = np.array([], bool)

        # 根据空的布尔数组选取DataFrame中的行
        blah[k]
        # 将DataFrame中根据空的布尔数组选取的行的值设为0
        blah[k] = 0

    def test_getitem_ix_mixed_integer(self):
        # 创建一个4行3列的随机数组，行索引为[1, 10, "C", "E"]，列索引为[1, 2, 3]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 3)),
            index=[1, 10, "C", "E"],
            columns=[1, 2, 3],
        )

        # 选取除最后一行外的所有行
        result = df.iloc[:-1]
        # 选取除最后一行外的所有行，期望的结果
        expected = df.loc[df.index[:-1]]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 根据混合整数列表选取行
        result = df.loc[[1, 10]]
        # 根据混合整数列表选取行，期望的结果
        expected = df.loc[pd.Index([1, 10])]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

    def test_getitem_ix_mixed_integer2(self):
        # 11320
        # 创建一个DataFrame，包含4列，列名为"rna", -1000, 0, 1000
        df = DataFrame(
            {
                "rna": (1.5, 2.2, 3.2, 4.5),
                -1000: [11, 21, 36, 40],
                0: [10, 22, 43, 34],
                1000: [0, 10, 20, 30],
            },
            columns=["rna", -1000, 0, 1000],
        )

        # 根据列名"1000"选取列
        result = df[[1000]]
        # 根据列索引[3]选取列，期望的结果
        expected = df.iloc[:, [3]]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)

        # 根据列名"-1000"选取列
        result = df[[-1000]]
        # 根据列索引[1]选取列，期望的结果
        expected = df.iloc[:, [1]]
        # 断言结果与期望相等
        tm.assert_frame_equal(result, expected)
    # 测试通过属性访问和索引访问确保 DataFrame 中的列引用一致性
    def test_getattr(self, float_frame):
        # 使用 assert_series_equal 检查属性访问和索引访问的列是否相等
        tm.assert_series_equal(float_frame.A, float_frame["A"])
        # 准备错误消息，测试访问不存在的列是否会引发 AttributeError 异常
        msg = "'DataFrame' object has no attribute 'NONEXISTENT_NAME'"
        with pytest.raises(AttributeError, match=msg):
            # 尝试访问不存在的列触发异常
            float_frame.NONEXISTENT_NAME

    # 测试通过 setattr 设置 DataFrame 列的值
    def test_setattr_column(self):
        # 创建包含一个列 'foobar' 的 DataFrame
        df = DataFrame({"foobar": 1}, index=range(10))

        # 使用 setattr 设置 'foobar' 列的值为 5
        df.foobar = 5
        # 断言 'foobar' 列的所有值都等于 5
        assert (df.foobar == 5).all()

    # 测试通过 setitem 设置 DataFrame 的条目（列）
    def test_setitem(self, float_frame, using_infer_string):
        # 从 DataFrame 中选择 'A' 列的偶数行作为新的 series
        series = float_frame["A"][::2]
        # 将新的 series 添加为 'col5' 列
        float_frame["col5"] = series
        # 断言 'col5' 列已经存在于 DataFrame 中
        assert "col5" in float_frame

        # 断言新 series 的长度为 15
        assert len(series) == 15
        # 断言整个 DataFrame 的长度为 30
        assert len(float_frame) == 30

        # 准备预期的 series，包括新添加的 'col5' 列
        exp = np.ravel(np.column_stack((series.values, [np.nan] * 15)))
        exp = Series(exp, index=float_frame.index, name="col5")
        # 使用 assert_series_equal 检查 'col5' 列的值是否符合预期
        tm.assert_series_equal(float_frame["col5"], exp)

        # 将 'A' 列添加为 'col6' 列
        series = float_frame["A"]
        float_frame["col6"] = series
        # 使用 assert_series_equal 检查 'col6' 列的值是否与 'A' 列相同，忽略列名检查
        tm.assert_series_equal(series, float_frame["col6"], check_names=False)

        # 使用 ndarray 设置 'col9' 列的值
        arr = np.random.default_rng(2).standard_normal(len(float_frame))
        float_frame["col9"] = arr
        # 断言 'col9' 列的所有值与 ndarray arr 的所有值相等
        assert (float_frame["col9"] == arr).all()

        # 设置 'col7' 列的值为标量 5
        float_frame["col7"] = 5
        # 断言 'col7' 列的所有值都等于 5
        assert (float_frame["col7"] == 5).all()

        # 设置 'col0' 列的值为浮点数 3.14
        float_frame["col0"] = 3.14
        # 断言 'col0' 列的所有值都等于 3.14
        assert (float_frame["col0"] == 3.14).all()

        # 设置 'col8' 列的值为字符串 "foo"
        float_frame["col8"] = "foo"
        # 断言 'col8' 列的所有值都等于 "foo"
        assert (float_frame["col8"] == "foo").all()

        # 创建 DataFrame 的切片 smaller，包含前两行
        smaller = float_frame[:2]
        # 将 'col10' 列设置为字符串列表 ["1", "2"]
        smaller["col10"] = ["1", "2"]

        # 如果 using_infer_string 为真，断言 'col10' 列的数据类型为 "string"
        if using_infer_string:
            assert smaller["col10"].dtype == "string"
        else:
            # 否则，断言 'col10' 列的数据类型为 np.object_
            assert smaller["col10"].dtype == np.object_
        # 断言 'col10' 列的所有值都等于 ["1", "2"]
        assert (smaller["col10"] == ["1", "2"]).all()

    # 测试通过 setitem 设置 DataFrame 条目，检查 dtype 变化的情况
    def test_setitem2(self):
        # 创建包含一个整数的 DataFrame
        df = DataFrame([[0, 0]])
        # 使用 iloc 设置第一行的所有值为 NaN
        df.iloc[0] = np.nan
        # 创建预期的 DataFrame，所有值为 NaN
        expected = DataFrame([[np.nan, np.nan]])
        # 使用 assert_frame_equal 检查 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建包含一个整数的 DataFrame
        df = DataFrame([[0, 0]])
        # 使用 loc 设置第一行的所有值为 NaN
        df.loc[0] = np.nan
        # 使用 assert_frame_equal 检查 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)
    # 测试 DataFrame 的布尔索引设置
    def test_setitem_boolean(self, float_frame):
        # 复制 float_frame 以避免修改原始数据
        df = float_frame.copy()
        # 复制 float_frame 的值数组
        values = float_frame.values.copy()

        # 将满足条件 df["A"] > 0 的行设置为 4
        df[df["A"] > 0] = 4
        # 将满足条件 values[:, 0] > 0 的行设置为 4
        values[values[:, 0] > 0] = 4
        # 检验 df 和 values 数组是否近似相等
        tm.assert_almost_equal(df.values, values)

        # 测试列重新索引是否正常工作
        series = df["A"] == 4
        # 将 series 根据 df.index 逆序重新索引
        series = series.reindex(df.index[::-1])
        # 将满足条件 series 的行设置为 1
        df[series] = 1
        # 将满足条件 values[:, 0] == 4 的行设置为 1
        values[values[:, 0] == 4] = 1
        # 检验 df 和 values 数组是否近似相等
        tm.assert_almost_equal(df.values, values)

        # 将满足条件 df > 0 的元素设置为 5
        df[df > 0] = 5
        # 将满足条件 values > 0 的元素设置为 5
        values[values > 0] = 5
        # 检验 df 和 values 数组是否近似相等
        tm.assert_almost_equal(df.values, values)

        # 将满足条件 df == 5 的元素设置为 0
        df[df == 5] = 0
        # 将满足条件 values == 5 的元素设置为 0
        values[values == 5] = 0
        # 检验 df 和 values 数组是否近似相等
        tm.assert_almost_equal(df.values, values)

        # 需要首先对齐的 DataFrame
        df[df[:-1] < 0] = 2
        # 将 values[:-1] 中满足条件 values[:-1] < 0 的元素设置为 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        # 检验 df 和 values 数组是否近似相等
        tm.assert_almost_equal(df.values, values)

        # 用行逆序的 df 中满足条件 df[::-1] == 2 的元素设置为 3
        df[df[::-1] == 2] = 3
        # 将满足条件 values == 2 的元素设置为 3
        values[values == 2] = 3
        # 检验 df 和 values 数组是否近似相等
        tm.assert_almost_equal(df.values, values)

        # 引发异常，测试用例必须使用布尔值的 DataFrame 或 2 维 ndarray
        msg = "Must pass DataFrame or 2-d ndarray with boolean values only"
        with pytest.raises(TypeError, match=msg):
            df[df * 0] = 2

        # 使用 DataFrame 进行索引
        df_orig = df.copy()
        # 创建布尔掩码，标记 df 中大于其绝对值的元素位置
        mask = df > np.abs(df)
        # 将 df 中大于其绝对值的元素设置为 NaN
        df[df > np.abs(df)] = np.nan
        # 复制原始 values 数组
        values = df_orig.values.copy()
        # 将 mask 中为 True 的位置设置为 NaN
        values[mask.values] = np.nan
        # 创建预期的 DataFrame，用于与 df 进行比较
        expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
        # 检验 df 和 expected DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 使用 DataFrame 设置值
        df[df > np.abs(df)] = df * 2
        # 将 mask 中为 True 的位置设置为 df.values * 2
        np.putmask(values, mask.values, df.values * 2)
        # 创建预期的 DataFrame，用于与 df 进行比较
        expected = DataFrame(values, index=df_orig.index, columns=df_orig.columns)
        # 检验 df 和 expected DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
    def test_setitem_corner(self, float_frame, using_infer_string):
        # corner case
        # 创建一个 DataFrame 包含列 'B' 和 'C'，索引为 np.arange(3)
        df = DataFrame({"B": [1.0, 2.0, 3.0], "C": ["a", "b", "c"]}, index=np.arange(3))
        # 删除列 'B'
        del df["B"]
        # 重新设置列 'B' 的值为 [1.0, 2.0, 3.0]
        df["B"] = [1.0, 2.0, 3.0]
        # 断言列 'B' 在 DataFrame 中
        assert "B" in df
        # 断言 DataFrame 的列数为 2
        assert len(df.columns) == 2

        # 设置新的列 'A'、'E'、'D' 和当前时间为列名，值为字符串
        df["A"] = "beginning"
        df["E"] = "foo"
        df["D"] = "bar"
        df[datetime.now()] = "date"
        df[datetime.now()] = 5.0

        # 当索引为空但存在列 'A' 和 'B' 的 DataFrame
        dm = DataFrame(index=float_frame.index)
        # 设置列 'A' 和 'B' 的值为字符串
        dm["A"] = "foo"
        dm["B"] = "bar"
        # 断言 DataFrame 的列数为 2
        assert len(dm.columns) == 2
        # 断言 DataFrame 的值类型为 np.object_
        assert dm.values.dtype == np.object_

        # 类型提升操作
        dm["C"] = 1
        # 断言列 'C' 的数据类型为 np.int64
        assert dm["C"].dtype == np.int64

        dm["E"] = 1.0
        # 断言列 'E' 的数据类型为 np.float64
        assert dm["E"].dtype == np.float64

        # 设置已存在的列 'A' 的值为 "bar"
        dm["A"] = "bar"
        assert "bar" == dm["A"].iloc[0]

        # 创建一个索引为 np.arange(3) 的 DataFrame
        dm = DataFrame(index=np.arange(3))
        dm["A"] = 1
        dm["foo"] = "bar"
        # 删除列 'foo'
        del dm["foo"]
        # 重新设置列 'foo' 的值为 "bar"
        dm["foo"] = "bar"
        # 根据 using_infer_string 的值断言列 'foo' 的数据类型
        if using_infer_string:
            assert dm["foo"].dtype == "string"
        else:
            assert dm["foo"].dtype == np.object_

        # 设置列 'coercible' 的值为列表 ["1", "2", "3"]
        dm["coercible"] = ["1", "2", "3"]
        # 根据 using_infer_string 的值断言列 'coercible' 的数据类型
        if using_infer_string:
            assert dm["coercible"].dtype == "string"
        else:
            assert dm["coercible"].dtype == np.object_

    def test_setitem_corner2(self):
        # 使用混合类型数据时的困难情况
        # 创建一个包含混合类型数据的 DataFrame
        data = {
            "title": ["foobar", "bar", "foobar"] + ["foobar"] * 17,
            "cruft": np.random.default_rng(2).random(20),
        }
        df = DataFrame(data)
        # 获取所有 'title' 列值为 "bar" 的行索引
        ix = df[df["title"] == "bar"].index
        # 将这些行的 'title' 列值设置为 "foobar"
        df.loc[ix, ["title"]] = "foobar"
        # 将这些行的 'cruft' 列值设置为 0
        df.loc[ix, ["cruft"]] = 0
        # 断言指定位置的 'title' 列的值为 "foobar"
        assert df.loc[1, "title"] == "foobar"
        # 断言指定位置的 'cruft' 列的值为 0
        assert df.loc[1, "cruft"] == 0

    def test_setitem_ambig(self, using_infer_string):
        # Difficulties with mixed-type data
        # Created as float type
        # 创建一个行索引为 range(3)，列索引为 range(3) 的 DataFrame
        dm = DataFrame(index=range(3), columns=range(3))
        # 设置第一列的值为 np.ones(3)
        dm[0] = np.ones(3)
        # 断言 DataFrame 的列数为 3
        assert len(dm.columns) == 3

        # 创建一个 Series，包含三个 Decimal(1) 元素，索引为 range(3)
        coercable_series = Series([Decimal(1) for _ in range(3)], index=range(3))
        # 设置第二列的值为 coercable_series
        dm[1] = coercable_series
        # 断言 DataFrame 的列数为 3
        assert len(dm.columns) == 3

        # 创建一个 Series，包含三个字符串元素，索引为 range(3)
        uncoercable_series = Series(["foo", "bzr", "baz"], index=range(3))
        # 设置第三列的值为 uncoercable_series
        dm[2] = uncoercable_series
        # 断言 DataFrame 的列数为 3
        assert len(dm.columns) == 3
        # 根据 using_infer_string 的值断言第三列的数据类型
        if using_infer_string:
            assert dm[2].dtype == "string"
        else:
            assert dm[2].dtype == np.object_

    def test_setitem_None(self, float_frame, using_infer_string):
        # GH #766
        # 将 float_frame["A"] 的值赋给 float_frame 的最后一列（None 列）
        float_frame[None] = float_frame["A"]
        # 如果 using_infer_string 为 False，则 key 为 np.nan
        key = None if not using_infer_string else np.nan
        # 使用 tm.assert_series_equal 断言列名为 "A" 的 Series 相等
        tm.assert_series_equal(
            float_frame.iloc[:, -1], float_frame["A"], check_names=False
        )
        # 使用 tm.assert_series_equal 断言列名为 key 的 Series 相等
        tm.assert_series_equal(
            float_frame.loc[:, key], float_frame["A"], check_names=False
        )
        # 使用 tm.assert_series_equal 断言列名为 key 的 Series 相等
        tm.assert_series_equal(float_frame[key], float_frame["A"], check_names=False)
    # 测试用例：测试 loc 方法在布尔掩码全为 False 时的设置行为
    def test_loc_setitem_boolean_mask_allfalse(self):
        # 创建一个包含三列的 DataFrame
        df = DataFrame(
            {"a": ["1", "2", "3"], "b": ["11", "22", "33"], "c": ["111", "222", "333"]}
        )

        # 复制 DataFrame
        result = df.copy()
        # 使用 loc 方法根据条件将 "a" 列中 NaN 值位置替换为原始 "a" 列的值
        result.loc[result.b.isna(), "a"] = result.a.copy()
        # 断言结果与原始 DataFrame 相等
        tm.assert_frame_equal(result, df)

    # 测试用例：测试获取空切片
    def test_getitem_slice_empty(self):
        # 创建一个包含一个元素的 DataFrame，并定义预期结果
        df = DataFrame([[1]], columns=MultiIndex.from_product([["A"], ["a"]]))
        result = df[:]

        expected = DataFrame([[1]], columns=MultiIndex.from_product([["A"], ["a"]]))

        # 断言结果与预期结果相等
        tm.assert_frame_equal(result, expected)
        # 确保 df[:] 返回 df 的视图，而不是相同的对象
        assert result is not df

    # 测试用例：测试使用整数步长的复杂切片
    def test_getitem_fancy_slice_integers_step(self):
        # 创建一个具有随机标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))

        # 正确操作：获取前8行中步长为2的行
        df.iloc[:8:2]
        # 将前8行中步长为2的行设置为 NaN
        df.iloc[:8:2] = np.nan
        # 断言所有设置为 NaN 的值
        assert isna(df.iloc[:8:2]).values.all()

    # 测试用例：测试使用整数切片键的获取和设置
    def test_getitem_setitem_integer_slice_keyerrors(self):
        # 创建一个具有随机标准正态分布数据的 DataFrame，使用偶数索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 5)), index=range(0, 20, 2)
        )

        # 正确操作：复制 DataFrame，并将索引为4到9的行设置为0
        cp = df.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).values.all()

        # 正确操作：复制 DataFrame，并将索引为3到10的行设置为0
        cp = df.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()

        # 获取索引为2到6的子 DataFrame，获取索引为3到11的子 DataFrame，并定义预期结果
        result = df.iloc[2:6]
        result2 = df.loc[3:11]
        expected = df.reindex([4, 6, 8, 10])

        # 断言结果与预期结果相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # 非单调索引，引发 KeyError
        df2 = df.iloc[list(range(5)) + list(range(5, 10))[::-1]]
        with pytest.raises(KeyError, match=r"^3$"):
            df2.loc[3:11]
        with pytest.raises(KeyError, match=r"^3$"):
            df2.loc[3:11] = 0

    # 测试用例：测试复杂获取和设置混合操作
    def test_fancy_getitem_slice_mixed(self, float_frame, float_string_frame):
        # 使用 iloc 获取最后三列的切片，并断言其数据类型为 np.float64
        sliced = float_string_frame.iloc[:, -3:]
        assert sliced["D"].dtype == np.float64

        # 获取单个块的视图，并设置其值，触发设置副本操作
        original = float_frame.copy()
        sliced = float_frame.iloc[:, -3:]

        # 断言切片的 "C" 列与原始 DataFrame 的 "C" 列共享内存
        assert np.shares_memory(sliced["C"]._values, float_frame["C"]._values)

        # 将切片的 "C" 列设置为4.0，并断言修改后的 DataFrame 与原始 DataFrame 相等
        sliced.loc[:, "C"] = 4.0
        tm.assert_frame_equal(float_frame, original)

    # 测试用例：测试使用非整数标签的获取和设置
    def test_getitem_setitem_non_ix_labels(self):
        # 创建一个包含20个元素的 DataFrame，使用日期索引
        df = DataFrame(range(20), index=date_range("2020-01-01", periods=20))

        # 获取从第5到第10个索引的子 DataFrame，并定义预期结果
        start, end = df.index[[5, 10]]
        result = df.loc[start:end]
        result2 = df[start:end]
        expected = df[5:11]

        # 断言结果与预期结果相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # 复制 DataFrame，并将从第5到第10个索引的行设置为0，并定义预期结果
        result = df.copy()
        result.loc[start:end] = 0
        result2 = df.copy()
        result2[start:end] = 0
        expected = df.copy()
        expected[5:11] = 0

        # 断言结果与预期结果相等
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)
    # 测试函数，验证在使用多个索引操作时的行为
    def test_ix_multi_take(self):
        # 创建一个随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        # 使用 loc 方法获取特定行的子集
        rs = df.loc[df.index == 0, :]
        # 使用 reindex 方法重新索引 DataFrame
        xp = df.reindex([0])
        # 使用 assert_frame_equal 断言比较两个 DataFrame 是否相等
        tm.assert_frame_equal(rs, xp)

        # GH#1321
        # 再次创建一个随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
        # 使用 loc 方法获取特定行和列的子集
        rs = df.loc[df.index == 0, df.columns == 1]
        # 使用 reindex 方法重新索引 DataFrame
        xp = df.reindex(index=[0], columns=[1])
        # 使用 assert_frame_equal 断言比较两个 DataFrame 是否相等
        tm.assert_frame_equal(rs, xp)

    # 测试函数，验证使用 fancy 索引获取单个元素的行为
    def test_getitem_fancy_scalar(self, float_frame):
        # 获取 float_frame 并赋值给变量 f
        f = float_frame
        # 获取 loc 方法并赋值给 ix
        ix = f.loc

        # 遍历每一列
        for col in f.columns:
            # 获取列 ts，并赋值给变量 ts
            ts = f[col]
            # 遍历索引的每隔五个值
            for idx in f.index[::5]:
                # 使用 loc 方法获取单个元素并断言相等
                assert ix[idx, col] == ts[idx]

    # 测试函数，验证使用 fancy 索引设置单个元素的行为
    def test_setitem_fancy_scalar(self, float_frame):
        # 获取 float_frame 并赋值给变量 f
        f = float_frame
        # 复制 float_frame 到 expected
        expected = float_frame.copy()
        # 获取 loc 方法并赋值给 ix
        ix = f.loc

        # 遍历每一列的索引和列名
        for j, col in enumerate(f.columns):
            # 获取列 col
            f[col]
            # 遍历索引的每隔五个值
            for idx in f.index[::5]:
                # 获取索引 idx 在索引中的位置并赋值给 i
                i = f.index.get_loc(idx)
                # 生成一个随机标准正态分布的值并赋值给 val
                val = np.random.default_rng(2).standard_normal()
                # 修改 expected 中的元素值
                expected.iloc[i, j] = val

                # 使用 loc 方法设置单个元素的值
                ix[idx, col] = val
                # 使用 assert_frame_equal 断言比较两个 DataFrame 是否相等
                tm.assert_frame_equal(f, expected)

    # 测试函数，验证使用 fancy 索引获取布尔数组的行为
    def test_getitem_fancy_boolean(self, float_frame):
        # 获取 float_frame 并赋值给变量 f
        f = float_frame
        # 获取 loc 方法并赋值给 ix
        ix = f.loc

        # 使用 reindex 方法创建一个期望的 DataFrame，并赋值给 expected
        expected = f.reindex(columns=["B", "D"])
        # 使用 loc 方法获取布尔数组的子集并断言相等
        result = ix[:, [False, True, False, True]]
        tm.assert_frame_equal(result, expected)

        # 使用 reindex 方法创建一个期望的 DataFrame，并赋值给 expected
        expected = f.reindex(index=f.index[5:10], columns=["B", "D"])
        # 使用 loc 方法获取布尔数组的子集并断言相等
        result = ix[f.index[5:10], [False, True, False, True]]
        tm.assert_frame_equal(result, expected)

        # 创建一个布尔向量 boolvec
        boolvec = f.index > f.index[7]
        # 使用 reindex 方法创建一个期望的 DataFrame，并赋值给 expected
        expected = f.reindex(index=f.index[boolvec])
        # 使用 loc 方法获取布尔数组的子集并断言相等
        result = ix[boolvec]
        tm.assert_frame_equal(result, expected)
        # 使用 loc 方法获取布尔数组的子集并断言相等
        result = ix[boolvec, :]
        tm.assert_frame_equal(result, expected)

        # 使用 loc 方法获取布尔数组和列名的子集并断言相等
        result = ix[boolvec, f.columns[2:]]
        expected = f.reindex(index=f.index[boolvec], columns=["C", "D"])
        tm.assert_frame_equal(result, expected)

    # 测试函数，验证使用 fancy 索引设置布尔数组的行为
    def test_setitem_fancy_boolean(self, float_frame):
        # 复制 float_frame 到 frame
        frame = float_frame.copy()
        # 复制 float_frame 到 expected
        expected = float_frame.copy()
        # 复制 expected 的值到 values
        values = expected.values.copy()

        # 创建一个布尔掩码 mask
        mask = frame["A"] > 0
        # 使用 loc 方法根据布尔掩码设置值
        frame.loc[mask] = 0.0
        # 根据布尔掩码更新 values 的值
        values[mask.values] = 0.0
        # 创建一个新的 DataFrame expected，并赋值给 expected
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        # 使用 assert_frame_equal 断言比较两个 DataFrame 是否相等
        tm.assert_frame_equal(frame, expected)

        # 复制 float_frame 到 frame
        frame = float_frame.copy()
        # 复制 float_frame 到 expected
        expected = float_frame.copy()
        # 复制 expected 的值到 values
        values = expected.values.copy()
        # 使用 loc 方法根据布尔掩码和列名设置值
        frame.loc[mask, ["A", "B"]] = 0.0
        # 根据布尔掩码和列名更新 values 的值
        values[mask.values, :2] = 0.0
        # 创建一个新的 DataFrame expected，并赋值给 expected
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        # 使用 assert_frame_equal 断言比较两个 DataFrame 是否相等
        tm.assert_frame_equal(frame, expected)
    # 测试从 DataFrame 中使用 fancy indexing 提取特定行，通过整数列表索引
    def test_getitem_fancy_ints(self, float_frame):
        # 使用整数列表索引提取指定行的数据
        result = float_frame.iloc[[1, 4, 7]]
        # 使用整数列表索引提取指定行的数据，然后使用 loc 方法根据索引提取相应行
        expected = float_frame.loc[float_frame.index[[1, 4, 7]]]
        # 检查提取的结果与期望是否相等
        tm.assert_frame_equal(result, expected)

        # 使用整数列表索引提取特定列的数据
        result = float_frame.iloc[:, [2, 0, 1]]
        # 使用整数列表索引提取特定列的数据，然后使用 loc 方法根据列名提取相应列
        expected = float_frame.loc[:, float_frame.columns[[2, 0, 1]]]
        # 检查提取的结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # 测试在 DataFrame 中使用布尔索引，处理标签不对齐的情况
    def test_getitem_setitem_boolean_misaligned(self, float_frame):
        # 创建一个布尔掩码，用于选择列"A"中大于1的元素，并且对索引进行反转
        mask = float_frame["A"][::-1] > 1

        # 根据布尔掩码选择相应的行数据
        result = float_frame.loc[mask]
        # 根据反转后的布尔掩码选择相应的行数据
        expected = float_frame.loc[mask[::-1]]
        # 检查提取的结果与期望是否相等
        tm.assert_frame_equal(result, expected)

        # 复制 DataFrame
        cp = float_frame.copy()
        expected = float_frame.copy()
        # 根据布尔掩码将选定行的数据设为0
        cp.loc[mask] = 0
        # 根据布尔掩码将选定行的数据设为0
        expected.loc[mask] = 0
        # 检查复制后的 DataFrame 和期望的是否相等
        tm.assert_frame_equal(cp, expected)

    # 测试在 DataFrame 中使用多维布尔索引进行获取和设置操作
    def test_getitem_setitem_boolean_multi(self):
        # 创建一个随机的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))

        # 创建两个布尔数组作为行和列的索引
        k1 = np.array([True, False, True])
        k2 = np.array([False, True])
        # 使用多维布尔索引获取指定行和列的数据
        result = df.loc[k1, k2]
        # 使用 loc 方法根据行和列索引提取相应的数据
        expected = df.loc[[0, 2], [1]]
        # 检查提取的结果与期望是否相等
        tm.assert_frame_equal(result, expected)

        # 复制 DataFrame
        expected = df.copy()
        # 使用多维布尔索引设置指定行和列的数据为5
        df.loc[np.array([True, False, True]), np.array([False, True])] = 5
        # 使用 loc 方法根据行和列索引设置相应的数据为5
        expected.loc[[0, 2], [1]] = 5
        # 检查设置后的 DataFrame 和期望的是否相等
        tm.assert_frame_equal(df, expected)

    # 测试在 DataFrame 中使用浮点标签进行位置索引获取操作
    def test_getitem_float_label_positional(self):
        # GH 53338
        # 创建一个带有浮点数索引的 Index
        index = Index([1.5, 2])
        # 创建一个带有浮点数索引的 DataFrame
        df = DataFrame(range(2), index=index)
        # 使用浮点数索引进行位置索引获取数据
        result = df[1:2]
        # 使用浮点数索引进行位置索引获取数据，期望的索引会被转换成浮点数
        expected = DataFrame([1], index=[2.0])
        # 检查提取的结果与期望是否相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试带有浮点数标签的索引和切片操作
    def test_getitem_setitem_float_labels(self):
        # 创建一个索引对象，包含浮点数作为索引
        index = Index([1.5, 2, 3, 4, 5])
        # 创建一个随机生成的 5x5 的数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)

        # 使用 loc 方法进行切片操作，选择索引范围为 1.5 到 4 的数据
        result = df.loc[1.5:4]
        # 期望的结果是重新索引为 [1.5, 2, 3, 4] 的数据框
        expected = df.reindex([1.5, 2, 3, 4])
        # 使用测试工具函数确认 result 和 expected 相等
        tm.assert_frame_equal(result, expected)
        # 确保结果长度为 4
        assert len(result) == 4

        # 使用 loc 方法选择索引范围为 4 到 5 的数据
        result = df.loc[4:5]
        # 期望的结果是重新索引为 [4, 5] 的数据框（使用整数）
        expected = df.reindex([4, 5])
        # 使用测试工具函数确认 result 和 expected 相等，忽略索引类型检查
        tm.assert_frame_equal(result, expected, check_index_type=False)
        # 确保结果长度为 2
        assert len(result) == 2

        # 使用 loc 方法选择索引范围为 4 到 5 的数据
        result = df.loc[4:5]
        # 期望的结果是重新索引为 [4.0, 5.0] 的数据框（使用浮点数）
        expected = df.reindex([4.0, 5.0])
        # 使用测试工具函数确认 result 和 expected 相等
        tm.assert_frame_equal(result, expected)
        # 确保结果长度为 2
        assert len(result) == 2

        # 使用 loc 方法选择索引范围为 1 到 2 的数据
        result = df.loc[1:2]
        # 期望的结果是使用 iloc 方法选取的前两行数据
        expected = df.iloc[0:2]
        # 使用测试工具函数确认 result 和 expected 相等
        tm.assert_frame_equal(result, expected)

        # 创建另一个索引对象，包含浮点数作为索引
        index = Index([1.0, 2.5, 3.5, 4.5, 5.0])
        # 创建一个随机生成的 5x5 的数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)

        # 通过 iloc 方法进行位置切片操作，预期会引发 TypeError 错误
        msg = (
            "cannot do positional indexing on Index with "
            r"these indexers \[1.0\] of type float"
        )
        with pytest.raises(TypeError, match=msg):
            df.iloc[1.0:5]

        # 使用 iloc 方法选择索引范围为 4 到 5 的数据
        result = df.iloc[4:5]
        # 期望的结果是重新索引为 [5.0] 的数据框
        expected = df.reindex([5.0])
        # 使用测试工具函数确认 result 和 expected 相等
        tm.assert_frame_equal(result, expected)
        # 确保结果长度为 1
        assert len(result) == 1

        # 复制数据框 df 到 cp
        cp = df.copy()

        # 通过 iloc 方法尝试进行赋值操作，预期会引发 TypeError 错误
        with pytest.raises(TypeError, match=_slice_msg):
            cp.iloc[1.0:5] = 0

        # 通过 iloc 方法进行切片操作，并与 0 比较，预期结果为 True
        with pytest.raises(TypeError, match=msg):
            result = cp.iloc[1.0:5] == 0

        assert result.values.all()
        # 确保与原始数据相等
        assert (cp.iloc[0:1] == df.iloc[0:1]).values.all()

        # 复制数据框 df 到 cp
        cp = df.copy()
        # 通过 iloc 方法将索引范围为 4 到 5 的值设置为 0
        cp.iloc[4:5] = 0
        # 确保索引范围为 4 到 5 的值为 0
        assert (cp.iloc[4:5] == 0).values.all()
        # 确保索引范围为 0 到 4 的值与原始数据相等
        assert (cp.iloc[0:4] == df.iloc[0:4]).values.all()

        # 使用 loc 方法选择索引范围为 1.0 到 5 的数据
        result = df.loc[1.0:5]
        # 期望的结果是与原始数据相同的数据框
        expected = df
        # 使用测试工具函数确认 result 和 expected 相等
        tm.assert_frame_equal(result, expected)
        # 确保结果长度为 5
        assert len(result) == 5

        # 使用 loc 方法选择索引范围为 1.1 到 5 的数据
        result = df.loc[1.1:5]
        # 期望的结果是重新索引为 [2.5, 3.5, 4.5, 5.0] 的数据框
        expected = df.reindex([2.5, 3.5, 4.5, 5.0])
        # 使用测试工具函数确认 result 和 expected 相等
        tm.assert_frame_equal(result, expected)
        # 确保结果长度为 4
        assert len(result) == 4

        # 使用 loc 方法选择索引范围为 4.51 到 5 的数据
        result = df.loc[4.51:5]
        # 期望的结果是重新索引为 [5.0] 的数据框
        expected = df.reindex([5.0])
        # 使用测试工具函数确认 result 和 expected 相等
        tm.assert_frame_equal(result, expected)
        # 确保结果长度为 1
        assert len(result) == 1

        # 使用 loc 方法选择索引范围为 1.0 到 5.0 的数据
        result = df.loc[1.0:5.0]
        # 期望的结果是重新索引为 [1.0, 2.5, 3.5, 4.5, 5.0] 的数据框
        expected = df.reindex([1.0, 2.5, 3.5, 4.5, 5.0])
        # 使用测试工具函数确认 result 和 expected 相等
        tm.assert_frame_equal(result, expected)
        # 确保结果长度为 5
        assert len(result) == 5

        # 复制数据框 df 到 cp
        cp = df.copy()
        # 通过 loc 方法将索引范围为 1.0 到 5.0 的值设置为 0
        cp.loc[1.0:5.0] = 0
        # 使用 loc 方法选择索引范围为 1.0 到 5.0 的数据，并确保所有值为 0
        result = cp.loc[1.0:5.0]
        assert (result == 0).values.all()
    # 测试在DataFrame中设置单列，包含混合类型的日期时间数据
    def test_setitem_single_column_mixed_datetime(self):
        # 创建一个5行3列的DataFrame，填充随机正态分布数据，指定索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["foo", "bar", "baz"],
        )

        # 将"timestamp"列设置为固定的Timestamp对象 "20010102"
        df["timestamp"] = Timestamp("20010102")

        # 检查DataFrame的数据类型
        result = df.dtypes
        # 创建预期的数据类型Series，包含浮点数和datetime64类型
        expected = Series(
            [np.dtype("float64")] * 3 + [np.dtype("datetime64[s]")],
            index=["foo", "bar", "baz", "timestamp"],
        )
        tm.assert_series_equal(result, expected)

        # 在特定位置（"b", "timestamp"）设置iNaT，并验证警告和数据的正确性
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            df.loc["b", "timestamp"] = iNaT
        assert not isna(df.loc["b", "timestamp"])
        assert df["timestamp"].dtype == np.object_
        assert df.loc["b", "timestamp"] == iNaT

        # 允许设置np.nan到指定位置（"c", "timestamp"）
        df.loc["c", "timestamp"] = np.nan
        assert isna(df.loc["c", "timestamp"])

        # 允许将np.nan设置到整行"d"的所有列
        df.loc["d", :] = np.nan
        assert not isna(df.loc["c", :]).all()

    # 测试在DataFrame中设置包含混合类型的日期时间数据
    def test_setitem_mixed_datetime(self):
        # 创建预期的DataFrame，包含整数和日期时间数据类型
        expected = DataFrame(
            {
                "a": [0, 0, 0, 0, 13, 14],
                "b": [
                    datetime(2012, 1, 1),
                    1,
                    "x",
                    "y",
                    datetime(2013, 1, 1),
                    datetime(2014, 1, 1),
                ],
            }
        )
        # 创建一个初始化为0的DataFrame，指定列名"a", "b"，索引为0到5
        df = DataFrame(0, columns=list("ab"), index=range(6))
        # 将"b"列设置为pd.NaT
        df["b"] = pd.NaT
        # 在特定位置（0, "b"）设置日期时间值
        df.loc[0, "b"] = datetime(2012, 1, 1)
        # 验证警告和数据的正确性
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            df.loc[1, "b"] = 1
        # 设置多行（2, 3）的"b"列为"x", "y"
        df.loc[[2, 3], "b"] = "x", "y"
        # 设置多行（4, 5）的"a", "b"列为给定数组A
        A = np.array(
            [
                [13, np.datetime64("2013-01-01T00:00:00")],
                [14, np.datetime64("2014-01-01T00:00:00")],
            ]
        )
        df.loc[[4, 5], ["a", "b"]] = A
        # 验证DataFrame是否与预期相等
        tm.assert_frame_equal(df, expected)

    # 测试在DataFrame中设置浮点数数据
    def test_setitem_frame_float(self, float_frame):
        # 从float_frame中选择部分数据，并将其设置到float_frame的最后两行
        piece = float_frame.loc[float_frame.index[:2], ["A", "B"]]
        float_frame.loc[float_frame.index[-2] :, ["A", "B"]] = piece.values
        # 检查设置后的结果与预期的值接近
        result = float_frame.loc[float_frame.index[-2:], ["A", "B"]].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    # 测试在DataFrame中设置包含混合类型的数据
    def test_setitem_frame_mixed(self, float_string_frame):
        # GH 3216

        # 复制float_string_frame，并将一部分数据piece设置到对应位置
        f = float_string_frame.copy()
        piece = DataFrame(
            [[1.0, 2.0], [3.0, 4.0]], index=f.index[0:2], columns=["A", "B"]
        )
        key = (f.index[slice(None, 2)], ["A", "B"])
        f.loc[key] = piece
        # 检查设置后的结果与piece的值接近
        tm.assert_almost_equal(f.loc[f.index[0:2], ["A", "B"]].values, piece.values)
    ```python`
        def test_setitem_frame_mixed_rows_unaligned(self, float_string_frame):
            # 测试在行索引不对齐的情况下设置 DataFrame 的值，示例来自 GH#3216
            f = float_string_frame.copy()  # 复制输入的 DataFrame
            piece = DataFrame(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],  # 创建包含数据的 DataFrame
                index=list(f.index[0:2]) + ["foo", "bar"],  # 设置不对齐的索引
                columns=["A", "B"],  # 设置列名
            )
            key = (f.index[slice(None, 2)], ["A", "B"])  # 创建包含行索引和列名的键
            f.loc[key] = piece  # 将 piece DataFrame 的值赋给 f 的指定位置
            tm.assert_almost_equal(
                f.loc[f.index[0:2:], ["A", "B"]].values, piece.values[0:2]
            )  # 验证赋值后的结果是否正确
    
        def test_setitem_frame_mixed_key_unaligned(self, float_string_frame):
            # 测试在键不对齐的情况下设置 DataFrame 的值，示例来自 GH#3216
            f = float_string_frame.copy()  # 复制输入的 DataFrame
            piece = f.loc[f.index[:2], ["A"]]  # 从 DataFrame 中提取一部分数据
            piece.index = f.index[-2:]  # 修改 piece 的索引，使其不对齐
            key = (f.index[slice(-2, None)], ["A", "B"])  # 创建包含行索引和列名的键
            f.loc[key] = piece  # 将 piece 的值赋给 f 的指定位置
            piece["B"] = np.nan  # 为 piece 的 B 列赋值为 NaN
            tm.assert_almost_equal(f.loc[f.index[-2:], ["A", "B"]].values, piece.values)  # 验证赋值后的结果是否正确
    
        def test_setitem_frame_mixed_ndarray(self, float_string_frame):
            # 测试使用 ndarray 赋值 DataFrame 的值，示例来自 GH#3216
            f = float_string_frame.copy()  # 复制输入的 DataFrame
            piece = float_string_frame.loc[f.index[:2], ["A", "B"]]  # 从 DataFrame 中提取一部分数据
            key = (f.index[slice(-2, None)], ["A", "B"])  # 创建包含行索引和列名的键
            f.loc[key] = piece.values  # 将 piece 的值（ndarray）赋给 f 的指定位置
            tm.assert_almost_equal(f.loc[f.index[-2:], ["A", "B"]].values, piece.values)  # 验证赋值后的结果是否正确
    
        def test_setitem_frame_upcast(self):
            # 测试在需要类型提升的情况下设置 DataFrame 的值
            df = DataFrame([[1, 2, "foo"], [3, 4, "bar"]], columns=["A", "B", "C"])  # 创建一个 DataFrame，包含不同类型的列
            df2 = df.copy()  # 复制 df 为 df2
            with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):  # 预期产生未来警告
                df2.loc[:, ["A", "B"]] = df.loc[:, ["A", "B"]] + 0.5  # 将 df 的 A 和 B 列加上 0.5，赋值给 df2
            expected = df.reindex(columns=["A", "B"])  # 生成期望的结果 DataFrame
            expected += 0.5  # 对期望结果的 A 和 B 列进行加 0.5 操作
            expected["C"] = df["C"]  # 保留原始的 C 列数据
            tm.assert_frame_equal(df2, expected)  # 验证 df2 是否等于期望结果
    
        def test_setitem_frame_align(self, float_frame):
            # 测试在行索引对齐的情况下设置 DataFrame 的值
            piece = float_frame.loc[float_frame.index[:2], ["A", "B"]]  # 从 float_frame 中提取一部分数据
            piece.index = float_frame.index[-2:]  # 修改 piece 的索引，使其对齐
            piece.columns = ["A", "B"]  # 设置 piece 的列名
            float_frame.loc[float_frame.index[-2:], ["A", "B"]] = piece  # 将 piece 的值赋给 float_frame 的指定位置
            result = float_frame.loc[float_frame.index[-2:], ["A", "B"]].values  # 获取赋值后的结果
            expected = piece.values  # 设置期望结果为 piece 的值
            tm.assert_almost_equal(result, expected)  # 验证赋值后的结果是否正确
    
        def test_getitem_setitem_ix_duplicates(self):
            # 测试在索引重复的情况下获取和设置 DataFrame 的值，示例来自 #1201
            df = DataFrame(
                np.random.default_rng(2).standard_normal((5, 3)),
                index=["foo", "foo", "bar", "baz", "bar"],  # 设置具有重复索引的 DataFrame
            )
    
            result = df.loc["foo"]  # 获取索引为 "foo" 的行
            expected = df[:2]  # 设置期望结果为前两行
            tm.assert_frame_equal(result, expected)  # 验证获取的结果是否正确
    
            result = df.loc["bar"]  # 获取索引为 "bar" 的行
            expected = df.iloc[[2, 4]]  # 设置期望结果为第 3 行和第 5 行
            tm.assert_frame_equal(result, expected)  # 验证获取的结果是否正确
    
            result = df.loc["baz"]  # 获取索引为 "baz" 的行
            expected = df.iloc[3]  # 设置期望结果为第 4 行
            tm.assert_series_equal(result, expected)  # 验证获取的结果是否正确
    # 定义一个测试方法，测试DataFrame的.loc和.iloc操作中处理布尔索引、重复索引和多个索引的情况
    def test_getitem_ix_boolean_duplicates_multiple(self):
        # 创建一个5行3列的DataFrame，数据是从标准正态分布中生成的随机数，同时设置索引，其中"foo"和"bar"有重复
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["foo", "foo", "bar", "baz", "bar"],
        )

        # 测试.loc方法对重复索引的处理，应返回所有匹配的行
        result = df.loc[["bar"]]
        exp = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, exp)

        # 测试.loc方法根据布尔条件进行索引，应返回符合条件的行
        result = df.loc[df[1] > 0]
        exp = df[df[1] > 0]
        tm.assert_frame_equal(result, exp)

        # 测试.loc方法根据布尔条件进行索引，应返回符合条件的行
        result = df.loc[df[0] > 0]
        exp = df[df[0] > 0]
        tm.assert_frame_equal(result, exp)

    # 使用pytest的参数化功能，测试DataFrame的.loc操作中处理布尔键引发的KeyError
    @pytest.mark.parametrize("bool_value", [True, False])
    def test_getitem_setitem_ix_bool_keyerror(self, bool_value):
        # 创建一个包含列"a"的DataFrame
        df = DataFrame({"a": [1, 2, 3]})
        # 设置预期的错误消息
        message = f"{bool_value}: boolean label can not be used without a boolean index"
        # 测试.loc方法在使用布尔键时是否引发KeyError，并验证错误消息
        with pytest.raises(KeyError, match=message):
            df.loc[bool_value]

        # 设置预期的错误消息
        msg = "cannot use a single bool to index into setitem"
        # 测试.loc方法在使用布尔键进行设置时是否引发KeyError，并验证错误消息
        with pytest.raises(KeyError, match=msg):
            df.loc[bool_value] = 0

    # 测试DataFrame的.loc和.iloc操作中处理单个元素索引的情况
    def test_single_element_ix_dont_upcast(self, float_frame):
        # 向DataFrame中的列"E"设置数值1
        float_frame["E"] = 1
        # 断言"E"列的数据类型是否是int或np.integer类型
        assert issubclass(float_frame["E"].dtype.type, (int, np.integer))

        # 测试.loc方法获取单个元素的值，验证返回结果是否为整数类型
        result = float_frame.loc[float_frame.index[5], "E"]
        assert is_integer(result)

        # GH 11617
        # 创建一个包含列"a"的DataFrame
        df = DataFrame({"a": [1.23]})
        # 向DataFrame中添加列"b"，设置数值666
        df["b"] = 666

        # 测试.loc方法获取指定位置的元素值，并验证返回结果是否为整数类型
        result = df.loc[0, "b"]
        assert is_integer(result)

        # 设置预期的Series对象，包含索引为0的值为666，列名为"b"
        expected = Series([666], [0], name="b")
        # 测试.loc方法根据索引列表获取多个元素值，验证返回结果是否与预期一致
        result = df.loc[[0], "b"]
        tm.assert_series_equal(result, expected)

    # 测试DataFrame的.iloc操作中，调用可调用对象作为参数时是否引发ValueError
    def test_iloc_callable_tuple_return_value_raises(self):
        # 创建一个包含40个元素的DataFrame，数据是0到39的整数，形状是10行4列，同时设置索引为0到18，步长为2
        df = DataFrame(np.arange(40).reshape(10, 4), index=range(0, 20, 2))
        # 设置预期的错误消息
        msg = "Returning a tuple from"
        # 测试.iloc方法在传递可调用对象作为参数时是否引发ValueError，并验证错误消息
        with pytest.raises(ValueError, match=msg):
            df.iloc[lambda _: (0,)]
        with pytest.raises(ValueError, match=msg):
            df.iloc[lambda _: (0,)] = 1

    # 测试DataFrame的.iloc操作中，获取行的不同方式（整数索引、切片、列表）
    def test_iloc_row(self):
        # 创建一个包含10行4列的DataFrame，数据是从标准正态分布中生成的随机数，同时设置索引为0到18，步长为2
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2)
        )

        # 测试.iloc方法获取指定位置的行，验证返回结果是否与.loc方法获取相同索引的行一致
        result = df.iloc[1]
        exp = df.loc[2]
        tm.assert_series_equal(result, exp)

        # 测试.iloc方法获取指定位置的行，验证返回结果是否与.loc方法获取相同索引的行一致
        result = df.iloc[2]
        exp = df.loc[4]
        tm.assert_series_equal(result, exp)

        # 测试.iloc方法获取指定范围的行（切片），验证返回结果是否与.loc方法获取相同范围的行一致
        result = df.iloc[slice(4, 8)]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)

        # 测试.iloc方法获取指定位置列表的行，验证返回结果是否与.loc方法获取相同索引的行一致
        result = df.iloc[[1, 2, 4, 6]]
        expected = df.reindex(df.index[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)
    def test_iloc_row_slice_view(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), index=range(0, 20, 2)
        )
        original = df.copy()

        # verify slice is view
        # 确保切片是视图
        subset = df.iloc[slice(4, 8)]

        assert np.shares_memory(df[2], subset[2])

        exp_col = original[2].copy()
        # 修改 subset 的列 2
        subset.loc[:, 2] = 0.0
        # 验证 df 的列 2 是否不变
        tm.assert_series_equal(df[2], exp_col)

    def test_iloc_col(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2)
        )

        result = df.iloc[:, 1]
        exp = df.loc[:, 2]
        tm.assert_series_equal(result, exp)

        result = df.iloc[:, 2]
        exp = df.loc[:, 4]
        tm.assert_series_equal(result, exp)

        # slice
        # 切片操作
        result = df.iloc[:, slice(4, 8)]
        expected = df.loc[:, 8:14]
        tm.assert_frame_equal(result, expected)

        # list of integers
        # 整数列表操作
        result = df.iloc[:, [1, 2, 4, 6]]
        expected = df.reindex(columns=df.columns[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_col_slice_view(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2)
        )
        original = df.copy()
        subset = df.iloc[:, slice(4, 8)]

        # verify slice is view
        # 确保切片是视图
        assert np.shares_memory(df[8]._values, subset[8]._values)
        subset[8] = 0.0
        # subset 变化了
        assert (subset[8] == 0).all()
        # 但 df 本身没有变化（setitem 替换整个列）
        tm.assert_frame_equal(df, original)

    def test_loc_duplicates(self):
        # gh-17105

        # insert a duplicate element to the index
        # 在索引中插入重复元素
        trange = date_range(
            start=Timestamp(year=2017, month=1, day=1),
            end=Timestamp(year=2017, month=1, day=5),
        )

        trange = trange.insert(loc=5, item=Timestamp(year=2017, month=1, day=5))

        df = DataFrame(0, index=trange, columns=["A", "B"])
        bool_idx = np.array([False, False, False, False, False, True])

        # assignment
        # 赋值操作
        df.loc[trange[bool_idx], "A"] = 6

        expected = DataFrame(
            {"A": [0, 0, 0, 0, 6, 6], "B": [0, 0, 0, 0, 0, 0]}, index=trange
        )
        tm.assert_frame_equal(df, expected)

        # in-place
        # 就地操作
        df = DataFrame(0, index=trange, columns=["A", "B"])
        df.loc[trange[bool_idx], "A"] += 6
        tm.assert_frame_equal(df, expected)
    def test_setitem_with_unaligned_tz_aware_datetime_column(self):
        # GH 12981
        # Assignment of unaligned offset-aware datetime series.
        # Make sure timezone isn't lost

        # 创建一个带有时区信息的日期时间序列，起始于 '2015-01-01'，共3个时间点，时区为 'utc'
        column = Series(date_range("2015-01-01", periods=3, tz="utc"), name="dates")
        # 创建一个DataFrame，将日期时间序列作为其中的一列
        df = DataFrame({"dates": column})
        # 通过索引顺序为 [1, 0, 2] 的方式重新赋值 'dates' 列，测试时区是否丢失
        df["dates"] = column[[1, 0, 2]]
        # 断言重新赋值后的 'dates' 列与原始列保持相同
        tm.assert_series_equal(df["dates"], column)

        # 再次创建一个DataFrame，将日期时间序列作为其中的一列
        df = DataFrame({"dates": column})
        # 使用 loc 方法，通过行索引 [0, 1, 2] 和列名 'dates' 的方式重新赋值，测试时区是否丢失
        df.loc[[0, 1, 2], "dates"] = column[[1, 0, 2]]
        # 断言重新赋值后的 'dates' 列与原始列保持相同
        tm.assert_series_equal(df["dates"], column)

    def test_loc_setitem_datetimelike_with_inference(self):
        # GH 7592
        # assignment of timedeltas with NaT

        # 定义一个小时的 timedelta
        one_hour = timedelta(hours=1)
        # 创建一个以日期时间为索引的DataFrame，共4行
        df = DataFrame(index=date_range("20130101", periods=4))
        # 为 'A' 列赋值，使用数组 [1小时, 1小时, 1小时, 1小时]，数据类型为 'm8[ns]'
        df["A"] = np.array([1 * one_hour] * 4, dtype="m8[ns]")
        # 使用 loc 方法，为 'B' 列赋值，使用数组 [2小时, 2小时, 2小时, 2小时]，数据类型为 'm8[ns]'
        df.loc[:, "B"] = np.array([2 * one_hour] * 4, dtype="m8[ns]")
        # 使用 loc 方法，为前3行的 'C' 列赋值，使用数组 [3小时, 3小时, 3小时]，数据类型为 'm8[ns]'
        df.loc[df.index[:3], "C"] = np.array([3 * one_hour] * 3, dtype="m8[ns]")
        # 使用 loc 方法，为 'D' 列赋值，使用数组 [4小时, 4小时, 4小时, 4小时]，数据类型为 'm8[ns]'
        df.loc[:, "D"] = np.array([4 * one_hour] * 4, dtype="m8[ns]")
        # 使用 loc 方法，为前3行的 'E' 列赋值，使用数组 [5小时, 5小时, 5小时]，数据类型为 'm8[ns]'
        df.loc[df.index[:3], "E"] = np.array([5 * one_hour] * 3, dtype="m8[ns]")
        # 创建 'F' 列，并赋值为 NaT（Not a Time，即无效时间）
        df["F"] = np.timedelta64("NaT")
        # 使用 loc 方法，为除最后一行外的 'F' 列赋值，使用数组 [6小时, 6小时, 6小时]，数据类型为 'm8[ns]'
        df.loc[df.index[:-1], "F"] = np.array([6 * one_hour] * 3, dtype="m8[ns]")
        # 使用 loc 方法，为最后3行的 'G' 列赋值，使用日期时间序列从 '20130101' 开始的3个时间点
        df.loc[df.index[-3:], "G"] = date_range("20130101", periods=3)
        # 创建 'H' 列，并赋值为 NaT（Not a Time，即无效时间）
        df["H"] = np.datetime64("NaT")
        # 检查DataFrame各列的数据类型，并与预期结果进行比较，生成一个 Series 对象
        result = df.dtypes
        # 创建一个预期的数据类型 Series，包含了 DataFrame 中各列的数据类型信息
        expected = Series(
            [np.dtype("timedelta64[ns]")] * 6 + [np.dtype("datetime64[ns]")] * 2,
            index=list("ABCDEFGH"),
        )
        # 断言实际的数据类型 Series 与预期的数据类型 Series 相等
        tm.assert_series_equal(result, expected)
    def test_getitem_interval_index_partial_indexing(self):
        # 创建一个包含三行四列的 DataFrame，每个元素都是 1，列由 IntervalIndex 组成
        df = DataFrame(
            np.ones((3, 4)), columns=pd.IntervalIndex.from_breaks(np.arange(5))
        )

        # 期望结果是取出 DataFrame 的第一列作为 Series
        expected = df.iloc[:, 0]

        # 使用数值索引直接访问列，并与期望结果进行比较
        res = df[0.5]
        tm.assert_series_equal(res, expected)

        # 使用 loc 方法通过数值索引访问列，并与期望结果进行比较
        res = df.loc[:, 0.5]
        tm.assert_series_equal(res, expected)
    # 测试设置 DataFrame 中数组作为单元格值的情况
    def test_setitem_array_as_cell_value(self):
        # GH#43422：测试特定 GitHub 问题编号的情况

        # 创建一个包含两列的 DataFrame，数据类型为 object
        df = DataFrame(columns=["a", "b"], dtype=object)
        
        # 将数组作为单元格值设置到指定行（0）中的列 "a" 和 "b"
        df.loc[0] = {"a": np.zeros((2,)), "b": np.zeros((2, 2))}
        
        # 创建预期的 DataFrame，用于与当前 df 进行比较
        expected = DataFrame({"a": [np.zeros((2,))], "b": [np.zeros((2, 2))]})
        
        # 使用测试工具比较当前 df 和预期的 DataFrame
        tm.assert_frame_equal(df, expected)

    # 测试通过 iloc 设置二维值的情况，值可以是 nullable
    def test_iloc_setitem_nullable_2d_values(self):
        # 创建一个包含列 "A" 的 DataFrame，数据类型为 "Int64"
        df = DataFrame({"A": [1, 2, 3]}, dtype="Int64")
        
        # 备份原始 DataFrame
        orig = df.copy()

        # 使用 df.values 的逆序设置 df 中的所有行
        df.loc[:] = df.values[:, ::-1]
        
        # 使用测试工具比较当前 df 和原始的 DataFrame
        tm.assert_frame_equal(df, orig)

        # 使用 NumpyExtensionArray 类型的逆序设置 df 中的所有行
        df.loc[:] = pd.core.arrays.NumpyExtensionArray(df.values[:, ::-1])
        
        # 使用测试工具再次比较当前 df 和原始的 DataFrame
        tm.assert_frame_equal(df, orig)

        # 使用 iloc[:] 复制当前 df 中的所有行到 df
        df.iloc[:] = df.iloc[:, :].copy()
        
        # 使用测试工具最后一次比较当前 df 和原始的 DataFrame
        tm.assert_frame_equal(df, orig)

    # 测试在空的 DataFrame 中使用 getitem 导致的 segfault
    def test_getitem_segfault_with_empty_like_object(self):
        # GH#46848：测试特定 GitHub 问题编号的情况
        
        # 创建一个包含一个元素的空对象类型 DataFrame
        df = DataFrame(np.empty((1, 1), dtype=object))
        
        # 使用 np.empty_like(df[0]) 设置 df 中第一列的值
        df[0] = np.empty_like(df[0])
        
        # 以下代码导致 segfault
        df[[0]]

    # 使用 pytest 参数化装饰器测试不匹配的 NaN 插入 nullable 数据类型失败的情况
    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    @pytest.mark.parametrize(
        "null", [pd.NaT, pd.NaT.to_numpy("M8[ns]"), pd.NaT.to_numpy("m8[ns]")]
    )
    def test_setting_mismatched_na_into_nullable_fails(
        self, null, any_numeric_ea_dtype
    ):
        # GH#44514：测试特定 GitHub 问题编号的情况，不将不匹配的 null 转换为 pd.NA
        
        # 创建一个包含列 "A" 的 DataFrame，数据类型为 any_numeric_ea_dtype
        df = DataFrame({"A": [1, 2, 3]}, dtype=any_numeric_ea_dtype)
        
        # 备份列 "A" 的 Series
        ser = df["A"].copy()
        
        # 获取列 "A" 的内部值数组
        arr = ser._values
        
        # 设置 arr[0] 为 null，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        msg = "|".join(
            [
                r"timedelta64\[ns\] cannot be converted to (Floating|Integer)Dtype",
                r"datetime64\[ns\] cannot be converted to (Floating|Integer)Dtype",
                "'values' contains non-numeric NA",
                r"Invalid value '.*' for dtype (U?Int|Float)\d{1,2}",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            arr[0] = null
        
        # 设置 arr[:2] 为 [null, null]，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            arr[:2] = [null, null]
        
        # 设置 ser[0] 为 null，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            ser[0] = null
        
        # 设置 ser[:2] 为 [null, null]，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            ser[:2] = [null, null]
        
        # 设置 ser.iloc[0] 为 null，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            ser.iloc[0] = null
        
        # 设置 ser.iloc[:2] 为 [null, null]，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            ser.iloc[:2] = [null, null]
        
        # 设置 df.iloc[0, 0] 为 null，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            df.iloc[0, 0] = null
        
        # 设置 df.iloc[:2, 0] 为 [null, null]，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            df.iloc[:2, 0] = [null, null]
        
        # 创建 df2 作为 df 的副本，并在 df2 中添加列 "B"，内容与 ser 相同
        df2 = df.copy()
        df2["B"] = ser.copy()
        
        # 设置 df2.iloc[0, 0] 为 null，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            df2.iloc[0, 0] = null
        
        # 设置 df2.iloc[:2, 0] 为 [null, null]，预期会引发 TypeError 异常，匹配特定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            df2.iloc[:2, 0] = [null, null]

    # 测试在扩展空 DataFrame 时保留索引名称的情况
    def test_loc_expand_empty_frame_keep_index_name(self):
        # GH#45621：测试特定 GitHub 问题编号的情况
        
        # 创建一个空的 DataFrame，包含列 "b"，以空索引名称 "a" 作为索引
        df = DataFrame(columns=["b"], index=Index([], name="a"))
        
        # 设置 df 中索引为 0 的行的值为 1
        df.loc[0] = 1
        
        # 创建预期的 DataFrame，带有与 df 相同的数据，但具有新的索引名称 "a"
        expected = DataFrame({"b": [1]}, index=Index([0], name="a"))
        
        # 使用测试工具比较当前 df 和预期的 DataFrame
        tm.assert_frame_equal(df, expected)
    def test_loc_expand_empty_frame_keep_midx_names(self):
        # GH#46317
        # 创建一个空的DataFrame，具有指定的列名和一个空的多级索引
        df = DataFrame(
            columns=["d"], index=MultiIndex.from_tuples([], names=["a", "b", "c"])
        )
        # 在DataFrame中的特定位置设置一个新值
        df.loc[(1, 2, 3)] = "foo"
        # 创建一个预期的DataFrame，具有相同的列名和指定的多级索引，与设置的值
        expected = DataFrame(
            {"d": ["foo"]},
            index=MultiIndex.from_tuples([(1, 2, 3)], names=["a", "b", "c"]),
        )
        # 检查预期的DataFrame与实际的DataFrame是否相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "val, idxr",
        [
            ("x", "a"),
            ("x", ["a"]),
            (1, "a"),
            (1, ["a"]),
        ],
    )
    def test_loc_setitem_rhs_frame(self, idxr, val):
        # GH#47578
        # 创建一个包含一个列"a"的DataFrame，其中包含两行数据
        df = DataFrame({"a": [1, 2]})

        # 在设置DataFrame值时，检查是否会产生一个警告
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            # 设置指定列的值为另一个DataFrame的某些行
            df.loc[:, idxr] = DataFrame({"a": [val, 11]}, index=[1, 2])
        # 创建一个预期的DataFrame，其中第一列的第一行为NaN，第二行为给定的值
        expected = DataFrame({"a": [np.nan, val]})
        # 检查预期的DataFrame与实际的DataFrame是否相等
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_enlarge_no_warning(self):
        # GH#47381
        # 创建一个空的DataFrame，具有两列"a"和"b"
        df = DataFrame(columns=["a", "b"])
        # 复制DataFrame以获取预期结果
        expected = df.copy()
        # 创建一个DataFrame视图
        view = df[:]
        # 使用iloc设置指定列的值，扩展DataFrame，但不应产生警告
        df.iloc[:, 0] = np.array([1, 2], dtype=np.float64)
        # 检查视图DataFrame与预期DataFrame是否相等
        tm.assert_frame_equal(view, expected)

    def test_loc_internals_not_updated_correctly(self):
        # GH#47867 all steps are necessary to reproduce the initial bug
        # 创建一个包含多列数据的DataFrame，包括一个布尔列
        # 创建一个具有多级索引的索引对象
        df = DataFrame(
            {"bool_col": True, "a": 1, "b": 2.5},
            index=MultiIndex.from_arrays([[1, 2], [1, 2]], names=["idx1", "idx2"]),
        )
        # 创建一个索引列表
        idx = [(1, 1)]

        # 设置新列"c"的值为3
        df["c"] = 3
        # 使用loc设置指定索引的列"c"的值为0
        df.loc[idx, "c"] = 0

        # 获取特定索引的列"c"的值
        df.loc[idx, "c"]
        # 获取特定索引的列"a"和"b"的值
        df.loc[idx, ["a", "b"]]

        # 使用loc设置特定索引的列"c"的值为15
        df.loc[idx, "c"] = 15
        # 获取特定索引的列"c"的结果
        result = df.loc[idx, "c"]
        # 创建一个预期的Series，具有相同的索引和名称，且所有值为15
        expected = Series(
            15,
            index=MultiIndex.from_arrays([[1], [1]], names=["idx1", "idx2"]),
            name="c",
        )
        # 检查预期的Series与实际结果的Series是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [None, [None], pd.NA, [pd.NA]])
    def test_iloc_setitem_string_list_na(self, val):
        # GH#45469
        # 创建一个包含一个列"a"的DataFrame，列类型为字符串
        df = DataFrame({"a": ["a", "b", "c"]}, dtype="string")
        # 使用iloc设置指定行的所有列为给定的值
        df.iloc[[0], :] = val
        # 创建一个预期的DataFrame，其中第一行的值为pd.NA，其余行不变
        expected = DataFrame({"a": [pd.NA, "b", "c"]}, dtype="string")
        # 检查预期的DataFrame与实际的DataFrame是否相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("val", [None, pd.NA])
    def test_iloc_setitem_string_na(self, val):
        # GH#45469
        # 创建一个包含一个列"a"的DataFrame，列类型为字符串
        df = DataFrame({"a": ["a", "b", "c"]}, dtype="string")
        # 使用iloc设置指定位置的值为给定的值
        df.iloc[0, :] = val
        # 创建一个预期的DataFrame，其中第一行的值为pd.NA，其余行不变
        expected = DataFrame({"a": [pd.NA, "b", "c"]}, dtype="string")
        # 检查预期的DataFrame与实际的DataFrame是否相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("func", [list, Series, np.array])
    def test_iloc_setitem_ea_null_slice_length_one_list(self, func):
        # GH#48016
        # 创建一个包含一个列"a"的DataFrame，列类型为Int64
        df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
        # 使用iloc设置指定列的所有行的值为给定的值
        df.iloc[:, func([0])] = 5
        # 创建一个预期的DataFrame，其中所有行的值为5
        expected = DataFrame({"a": [5, 5, 5]}, dtype="Int64")
        # 检查预期的DataFrame与实际的DataFrame是否相等
        tm.assert_frame_equal(df, expected)
    def test_loc_named_tuple_for_midx(self):
        # 测试用例 GH#48124：测试 DataFrame 的 loc 方法使用命名元组索引 MultiIndex

        # 创建一个包含 MultiIndex 的 DataFrame，其中 MultiIndex 由产品形式生成
        df = DataFrame(
            index=MultiIndex.from_product(
                [["A", "B"], ["a", "b", "c"]], names=["first", "second"]
            )
        )

        # 根据 DataFrame 的索引名称创建一个命名元组索引对象
        indexer_tuple = namedtuple("Indexer", df.index.names)
        idxr = indexer_tuple(first="A", second=["a", "b"])

        # 使用命名元组索引对象 idxr，从 DataFrame 中选择对应的行
        result = df.loc[idxr, :]

        # 创建预期的 DataFrame，其中 MultiIndex 由元组形式生成
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                [("A", "a"), ("A", "b")], names=["first", "second"]
            )
        )

        # 使用测试框架的方法验证结果 DataFrame 与预期的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("indexer", [["a"], "a"])
    @pytest.mark.parametrize("col", [{}, {"b": 1}])
    def test_set_2d_casting_date_to_int(self, col, indexer):
        # 测试用例 GH#49159：测试将日期列强制类型转换为整数

        # 创建一个 DataFrame 包含日期列 'a' 和其他列（如果有）
        df = DataFrame(
            {"a": [Timestamp("2022-12-29"), Timestamp("2022-12-30")], **col},
        )

        # 将 DataFrame 中第一行的指定索引 indexer 的列值设置为日期列 'a' 加一天后的值
        df.loc[[1], indexer] = df["a"] + pd.Timedelta(days=1)

        # 创建预期的 DataFrame，其中日期列 'a' 的值加一天，其他列不变
        expected = DataFrame(
            {"a": [Timestamp("2022-12-29"), Timestamp("2022-12-31")], **col},
        )

        # 使用测试框架的方法验证结果 DataFrame 与预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("col", [{}, {"name": "a"}])
    def test_loc_setitem_reordering_with_all_true_indexer(self, col):
        # 测试用例 GH#48701：测试 loc 方法使用全部为 True 的索引器进行重新排序

        # 创建一个包含列 'x' 和 'y' 的 DataFrame，以及其他可能的列（如果有）
        n = 17
        df = DataFrame({**col, "x": range(n), "y": range(n)})

        # 复制预期结果与当前 DataFrame 相同
        expected = df.copy()

        # 使用 loc 方法，将当前 DataFrame 中所有行的列 'x' 和 'y' 的值重新赋值为它们自身
        df.loc[n * [True], ["x", "y"]] = df[["x", "y"]]

        # 使用测试框架的方法验证结果 DataFrame 与预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_loc_rhs_empty_warning(self):
        # 测试用例 GH#48480：测试 loc 方法在右侧为空时的警告

        # 创建一个空的 DataFrame，只包含列 'a' 和 'b'
        df = DataFrame(columns=["a", "b"])

        # 复制预期结果与当前 DataFrame 相同
        expected = df.copy()

        # 创建一个右侧 DataFrame，只包含列 'a'
        rhs = DataFrame(columns=["a"])

        # 使用 loc 方法将当前 DataFrame 的所有行的列 'a' 的值设置为右侧 DataFrame 的值
        with tm.assert_produces_warning(None):
            df.loc[:, "a"] = rhs

        # 使用测试框架的方法验证结果 DataFrame 与预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_iloc_ea_series_indexer(self):
        # 测试用例 GH#49521：测试 iloc 方法使用 Series 作为索引器

        # 创建一个包含两行的 DataFrame，每行包含五个整数
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

        # 创建一个整数型 Series 索引器，指定行索引
        indexer = Series([0, 1], dtype="Int64")
        
        # 创建一个整数型 Series 行索引器
        row_indexer = Series([1], dtype="Int64")

        # 使用 iloc 方法，根据行索引器和列索引器选择子 DataFrame
        result = df.iloc[row_indexer, indexer]

        # 创建预期的 DataFrame，包含选择的子集
        expected = DataFrame([[5, 6]], index=[1])

        # 使用测试框架的方法验证结果 DataFrame 与预期的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 iloc 方法，根据行索引器和列索引器值选择子 DataFrame
        result = df.iloc[row_indexer.values, indexer.values]

        # 使用测试框架的方法验证结果 DataFrame 与预期的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_iloc_ea_series_indexer_with_na(self):
        # 测试用例 GH#49521：测试 iloc 方法使用包含 NA 值的 Series 作为索引器

        # 创建一个包含两行的 DataFrame，每行包含五个整数
        df = DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

        # 创建一个包含 NA 值的整数型 Series 索引器
        indexer = Series([0, pd.NA], dtype="Int64")

        # 准备一个错误消息字符串
        msg = "cannot convert"

        # 使用 pytest 的异常断言检查使用索引器在 iloc 方法中的操作是否会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.iloc[:, indexer]

        # 使用 pytest 的异常断言检查使用索引器值在 iloc 方法中的操作是否会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.iloc[:, indexer.values]

    @pytest.mark.parametrize("indexer", [True, (True,)])
    @pytest.mark.parametrize("dtype", [bool, "boolean"])
    # 定义一个测试方法，用于测试 loc 方法在多索引条件下的行为
    def test_loc_bool_multiindex(self, performance_warning, dtype, indexer):
        # 创建一个多级索引对象 midx，包含两个级别的布尔类型数据列
        midx = MultiIndex.from_arrays(
            [
                Series([True, True, False, False], dtype=dtype),  # 第一级别数据列
                Series([True, False, True, False], dtype=dtype),  # 第二级别数据列
            ],
            names=["a", "b"],  # 设置索引级别名称
        )
        # 使用创建的多级索引对象 midx 构建 DataFrame df，其中包含一列数据 'c'
        df = DataFrame({"c": [1, 2, 3, 4]}, index=midx)
        # 使用 with 语句调用 maybe_produces_warning 函数，检查性能警告并记录
        with tm.maybe_produces_warning(performance_warning, isinstance(indexer, tuple)):
            # 使用 loc 方法根据 indexer 参数选择行，并将结果赋给 result
            result = df.loc[indexer]
        # 创建预期的 DataFrame 对象 expected，包含一列数据 'c'，使用 Index 对象作为索引
        expected = DataFrame(
            {"c": [1, 2]}, index=Index([True, False], name="b", dtype=dtype)
        )
        # 使用 assert_frame_equal 函数比较 result 和 expected，确保它们相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化标记定义一个测试方法，测试 loc 方法中关于日期时间赋值时 dtype 的保持行为
    @pytest.mark.parametrize("utc", [False, True])
    @pytest.mark.parametrize("indexer", ["date", ["date"]])
    def test_loc_datetime_assignment_dtype_does_not_change(self, utc, indexer):
        # 创建 DataFrame df，包含两列数据 'date' 和 'update'，其中 'date' 列为日期时间对象
        df = DataFrame(
            {
                "date": to_datetime(
                    [datetime(2022, 1, 20), datetime(2022, 1, 22)], utc=utc
                ),
                "update": [True, False],  # 'update' 列为布尔类型数据
            }
        )
        # 复制 df 的内容到 expected
        expected = df.copy(deep=True)

        # 从 df 中选择 'update' 列为 True 的行，结果存储在 update_df 中
        update_df = df[df["update"]]

        # 使用 loc 方法根据 'update' 列为 True 的行，将 'indexer' 参数指定列的值更新为 update_df 中的 'date' 列的值
        df.loc[df["update"], indexer] = update_df["date"]

        # 使用 assert_frame_equal 函数比较 df 和 expected，确保它们相等
        tm.assert_frame_equal(df, expected)

    # 使用 pytest 的参数化标记定义一个测试方法，测试设置项时值强制类型转换的行为
    @pytest.mark.parametrize("indexer, idx", [(tm.loc, 1), (tm.iloc, 2)])
    def test_setitem_value_coercing_dtypes(self, indexer, idx):
        # 创建 DataFrame df，包含三行两列的对象数组，数据类型为对象
        df = DataFrame([["1", np.nan], ["2", np.nan], ["3", np.nan]], dtype=object)
        # 创建 DataFrame rhs，包含两行两列的对象数组
        rhs = DataFrame([[1, np.nan], [2, np.nan]])
        # 使用 indexer 函数选择 df 的部分区域，并将 rhs 的值赋给这部分区域
        indexer(df)[:idx, :] = rhs
        # 创建预期的 DataFrame 对象 expected，确保部分区域被正确地强制类型转换
        expected = DataFrame([[1, np.nan], [2, np.nan], ["3", np.nan]], dtype=object)
        # 使用 assert_frame_equal 函数比较 df 和 expected，确保它们相等
        tm.assert_frame_equal(df, expected)

    # 定义一个测试方法，测试大端格式支持下选择列的行为
    def test_big_endian_support_selecting_columns(self):
        # 创建列名为 'a'，数据为大端格式的数组的字典
        columns = ["a"]
        data = [np.array([1, 2], dtype=">f8")]
        # 使用 dict(zip(columns, data)) 创建 DataFrame df
        df = DataFrame(dict(zip(columns, data)))
        # 使用 df.columns 选择所有列，并赋给 result
        result = df[df.columns]
        # 创建预期的 DataFrame 对象 dfexp，包含一列数据 'a'，数据类型为大端格式
        dfexp = DataFrame({"a": [1, 2]}, dtype=">f8")
        # 使用 assert_frame_equal 函数比较 result 和 expected，确保它们相等
        expected = dfexp[dfexp.columns]
        tm.assert_frame_equal(result, expected)
class TestDataFrameIndexingUInt64:
    def test_setitem(self):
        # 创建一个 DataFrame 对象，包含两列，其中一列是 np.uint64 类型
        df = DataFrame(
            {"A": np.arange(3), "B": [2**63, 2**63 + 5, 2**63 + 10]},
            dtype=np.uint64,
        )
        # 从 DataFrame 的列 "A" 创建一个重命名为 "foo" 的索引对象
        idx = df["A"].rename("foo")

        # setitem 操作测试
        # 断言 DataFrame 中没有列 "C"
        assert "C" not in df.columns
        # 将列 "C" 设置为索引对象 idx
        df["C"] = idx
        # 断言设置后列 "C" 的内容与预期的 Series 相等
        tm.assert_series_equal(df["C"], Series(idx, name="C"))

        # 断言 DataFrame 中没有列 "D"
        assert "D" not in df.columns
        # 将列 "D" 设置为字符串 "foo"
        df["D"] = "foo"
        # 再将列 "D" 设置为索引对象 idx
        df["D"] = idx
        # 断言设置后列 "D" 的内容与预期的 Series 相等
        tm.assert_series_equal(df["D"], Series(idx, name="D"))
        # 删除列 "D"
        del df["D"]

        # 包含 NaN 值的情况测试：由于 uint64 类型不支持 NaN，所以列应转换为对象类型
        df2 = df.copy()
        # 使用 assert_produces_warning 上下文管理器检查是否产生 FutureWarning 警告，匹配指定的正则表达式
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            # 将某些位置设置为 pd.NaT
            df2.iloc[1, 1] = pd.NaT
            df2.iloc[1, 2] = pd.NaT
        # 获取 "B" 列的结果 Series，并断言其中的非 NaN 值
        result = df2["B"]
        tm.assert_series_equal(notna(result), Series([True, False, True], name="B"))
        # 断言 df2 的 dtypes 属性与预期的 Series 相等
        tm.assert_series_equal(
            df2.dtypes,
            Series(
                [np.dtype("uint64"), np.dtype("O"), np.dtype("O")],
                index=["A", "B", "C"],
            ),
        )
    # 定义一个方法用于生成一个包含类别数据和数值的DataFrame，类别数据进行了部分更改
    def exp_parts_cats_col(self):
        # 创建类别列数据，只包含"a"和"b"两个类别
        cats3 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        # 创建索引数据
        idx3 = Index(["h", "i", "j", "k", "l", "m", "n"])
        # 创建数值列数据
        values3 = [1, 1, 1, 1, 1, 1, 1]
        # 生成DataFrame对象，包含"cats"和"values"两列，使用idx3作为索引
        exp_parts_cats_col = DataFrame({"cats": cats3, "values": values3}, index=idx3)
        return exp_parts_cats_col

    @pytest.fixture
    # 定义一个测试用的fixture，返回一个包含单个类别值更改后的DataFrame对象
    def exp_single_cats_value(self):
        # 创建类别列数据，只包含"a"和"b"两个类别
        cats4 = Categorical(["a", "a", "b", "a", "a", "a", "a"], categories=["a", "b"])
        # 创建索引数据
        idx4 = Index(["h", "i", "j", "k", "l", "m", "n"])
        # 创建数值列数据
        values4 = [1, 1, 1, 1, 1, 1, 1]
        # 生成DataFrame对象，包含"cats"和"values"两列，使用idx4作为索引
        exp_single_cats_value = DataFrame(
            {"cats": cats4, "values": values4}, index=idx4
        )
        return exp_single_cats_value

    # 定义一个测试方法，测试DataFrame的loc/iloc设置多行数据的情况
    def test_loc_iloc_setitem_list_of_lists(self, orig, indexer_li):
        # 复制原始DataFrame对象
        df = orig.copy()

        # 设置切片键
        key = slice(2, 4)
        # 如果indexer_li是tm.loc，则设置为对应索引的切片
        if indexer_li is tm.loc:
            key = slice("j", "k")

        # 使用indexer_li设置指定位置的多行数据
        indexer_li(df)[key, :] = [["b", 2], ["b", 2]]

        # 创建预期结果的类别数据，只包含"a"和"b"两个类别
        cats2 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        # 创建预期结果的索引数据
        idx2 = Index(["h", "i", "j", "k", "l", "m", "n"])
        # 创建预期结果的数值数据
        values2 = [1, 1, 2, 2, 1, 1, 1]
        # 生成预期结果的DataFrame对象，包含"cats"和"values"两列，使用idx2作为索引
        exp_multi_row = DataFrame({"cats": cats2, "values": values2}, index=idx2)
        # 断言设置后的DataFrame与预期结果DataFrame相等
        tm.assert_frame_equal(df, exp_multi_row)

        # 复制原始DataFrame对象
        df = orig.copy()
        # 使用pytest的断言，测试设置非预期类别值时的异常情况
        with pytest.raises(TypeError, match=msg1):
            indexer_li(df)[key, :] = [["c", 2], ["c", 2]]

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc, tm.at, tm.iat])
    # 使用参数化装饰器定义测试方法，测试DataFrame的loc/iloc/at/iat设置单个类别值的情况
    def test_loc_iloc_at_iat_setitem_single_value_in_categories(
        self, orig, exp_single_cats_value, indexer
    ):
        # 复制原始DataFrame对象
        df = orig.copy()

        # 设置键值对
        key = (2, 0)
        # 如果indexer是tm.loc或tm.at，则设置为对应索引和列名的键值对
        if indexer in [tm.loc, tm.at]:
            key = (df.index[2], df.columns[0])

        # 使用indexer设置指定位置的单个类别值
        indexer(df)[key] = "b"
        # 断言设置后的DataFrame与预期结果DataFrame相等
        tm.assert_frame_equal(df, exp_single_cats_value)

        # 使用pytest的断言，测试设置非预期类别值时的异常情况
        with pytest.raises(TypeError, match=msg1):
            indexer(df)[key] = "c"

    # 定义一个测试方法，测试DataFrame的loc/iloc设置掩码位置单个类别值的情况
    def test_loc_iloc_setitem_mask_single_value_in_categories(
        self, orig, exp_single_cats_value, indexer_li
    ):
        # 复制原始DataFrame对象
        df = orig.copy()

        # 创建掩码，选择指定索引为"j"的位置
        mask = df.index == "j"
        key = 0
        # 如果indexer_li是tm.loc，则设置为对应列名的键
        if indexer_li is tm.loc:
            key = df.columns[key]

        # 使用indexer_li设置掩码位置的单个类别值
        indexer_li(df)[mask, key] = "b"
        # 断言设置后的DataFrame与预期结果DataFrame相等
        tm.assert_frame_equal(df, exp_single_cats_value)
    # 测试函数：测试在非分类列右侧使用 loc 或 iloc 进行完整行赋值
    def test_loc_iloc_setitem_full_row_non_categorical_rhs(self, orig, indexer_li):
        # 复制原始数据框
        df = orig.copy()

        # 确定要使用的索引键
        key = 2
        if indexer_li is tm.loc:
            key = df.index[2]

        # 在非分类 dtype 的列中，将指定行的值替换为 ["b", 2]
        indexer_li(df)[key, :] = ["b", 2]

        # 预期的单行数据框
        cats1 = Categorical(["a", "a", "b", "a", "a", "a", "a"], categories=["a", "b"])
        idx1 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values1 = [1, 1, 2, 1, 1, 1, 1]
        exp_single_row = DataFrame({"cats": cats1, "values": values1}, index=idx1)

        # 断言数据框是否与预期单行数据框相等
        tm.assert_frame_equal(df, exp_single_row)

        # 尝试将非存在于 df["cat"] 中的值 "c" 替换为 2，预期引发 TypeError 异常
        with pytest.raises(TypeError, match=msg1):
            indexer_li(df)[key, :] = ["c", 2]

    # 测试函数：测试使用 loc 或 iloc 在分类列右侧部分列赋值
    def test_loc_iloc_setitem_partial_col_categorical_rhs(
        self, orig, exp_parts_cats_col, indexer_li
    ):
        # 复制原始数据框
        df = orig.copy()

        # 确定要使用的索引键
        key = (slice(2, 4), 0)
        if indexer_li is tm.loc:
            key = (slice("j", "k"), df.columns[0])

        # 将指定列的部分值替换为已定义的分类
        compat = Categorical(["b", "b"], categories=["a", "b"])
        indexer_li(df)[key] = compat

        # 断言数据框是否与预期部分列分类数据框相等
        tm.assert_frame_equal(df, exp_parts_cats_col)

        # 尝试将不符合 df["cat"] 中已定义分类的值替换为 semi_compat，预期引发 TypeError 异常
        semi_compat = Categorical(list("bb"), categories=list("abc"))
        with pytest.raises(TypeError, match=msg2):
            indexer_li(df)[key] = semi_compat

        # 尝试将不符合 df["cat"] 中已定义分类的值替换为 incompat，预期引发 TypeError 异常
        incompat = Categorical(list("cc"), categories=list("abc"))
        with pytest.raises(TypeError, match=msg2):
            indexer_li(df)[key] = incompat

    # 测试函数：测试在非分类列右侧使用 loc 或 iloc 进行部分列赋值
    def test_loc_iloc_setitem_non_categorical_rhs(
        self, orig, exp_parts_cats_col, indexer_li
    ):
        # 复制原始数据框
        df = orig.copy()

        # 确定要使用的索引键
        key = (slice(2, 4), 0)
        if indexer_li is tm.loc:
            key = (slice("j", "k"), df.columns[0])

        # 将非分类 dtype 列的部分值替换为 ["b", "b"]
        indexer_li(df)[key] = ["b", "b"]

        # 断言数据框是否与预期部分列分类数据框相等
        tm.assert_frame_equal(df, exp_parts_cats_col)

        # 尝试将不符合 df["cat"] 中已定义分类的值替换为 "c"，预期引发 TypeError 异常
        with pytest.raises(TypeError, match=msg1):
            indexer_li(df)[key] = ["c", "c"]

    # 参数化测试：使用不同的索引器进行数据框操作
    @pytest.mark.parametrize("indexer", [tm.getitem, tm.loc, tm.iloc])
    # 测试用例：测试在保留日期对象索引时，使用指定的索引器获取数据
    def test_getitem_preserve_object_index_with_dates(self, indexer):
        # https://github.com/pandas-dev/pandas/pull/42950 - 当从数据框中选择列时，
        # 不要在 Series 构造时尝试推断对象 dtype 索引
        # 创建一个日期范围索引，并将其转换为对象类型
        idx = date_range("2012", periods=3).astype(object)
        # 创建一个包含指定索引的数据框
        df = DataFrame({0: [1, 2, 3]}, index=idx)
        # 断言数据框索引的 dtype 是对象类型
        assert df.index.dtype == object

        # 根据传入的索引器获取数据
        if indexer is tm.getitem:
            # 使用索引器获取数据框的第一列作为 Series
            ser = indexer(df)[0]
        else:
            # 使用索引器获取数据框的第一列的第一行作为 Series
            ser = indexer(df)[:, 0]

        # 断言 Series 的索引的 dtype 是对象类型
        assert ser.index.dtype == object

    # 测试用例：测试在多级索引的一个级别上使用 loc 方法
    def test_loc_on_multiindex_one_level(self):
        # GH#45779
        # 创建一个包含单个列的数据框，使用多级索引，其中第一个级别命名为 "first"
        df = DataFrame(
            data=[[0], [1]],
            index=MultiIndex.from_tuples([("a",), ("b",)], names=["first"]),
        )
        # 创建期望的结果数据框，使用与原始数据框相同的第一个级别索引 "a"
        expected = DataFrame(
            data=[[0]], index=MultiIndex.from_tuples([("a",)], names=["first"])
        )
        # 使用 loc 方法获取多级索引中第一个级别为 "a" 的数据
        result = df.loc["a"]
        # 使用断言检查结果数据框与期望的数据框是否相等
        tm.assert_frame_equal(result, expected)
class TestDeprecatedIndexers:
    @pytest.mark.parametrize(
        "key", [{1}, {1: 1}, ({1}, "a"), ({1: 1}, "a"), (1, {"a"}), (1, {"a": "a"})]
    )
    def test_getitem_dict_and_set_deprecated(self, key):
        # 使用 DataFrame 创建一个包含两行两列的数据框，列名为 ["a", "b"]
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        # 断言使用 loc 访问时会抛出 TypeError，匹配字符串 "as an indexer is not supported"
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key]

    @pytest.mark.parametrize(
        "key",
        [
            {1},
            {1: 1},
            (({1}, 2), "a"),
            (({1: 1}, 2), "a"),
            ((1, 2), {"a"}),
            ((1, 2), {"a": "a"}),
        ],
    )
    def test_getitem_dict_and_set_deprecated_multiindex(self, key):
        # 使用 MultiIndex 创建一个包含两行两列的数据框，列名为 ["a", "b"]，索引为 [(1, 2), (3, 4)]
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=["a", "b"],
            index=MultiIndex.from_tuples([(1, 2), (3, 4)]),
        )
        # 断言使用 loc 访问时会抛出 TypeError，匹配字符串 "as an indexer is not supported"
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key]

    @pytest.mark.parametrize(
        "key", [{1}, {1: 1}, ({1}, "a"), ({1: 1}, "a"), (1, {"a"}), (1, {"a": "a"})]
    )
    def test_setitem_dict_and_set_disallowed(self, key):
        # 使用 DataFrame 创建一个包含两行两列的数据框，列名为 ["a", "b"]
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        # 断言使用 loc 设置值时会抛出 TypeError，匹配字符串 "as an indexer is not supported"
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key] = 1

    @pytest.mark.parametrize(
        "key",
        [
            {1},
            {1: 1},
            (({1}, 2), "a"),
            (({1: 1}, 2), "a"),
            ((1, 2), {"a"}),
            ((1, 2), {"a": "a"}),
        ],
    )
    def test_setitem_dict_and_set_disallowed_multiindex(self, key):
        # 使用 MultiIndex 创建一个包含两行两列的数据框，列名为 ["a", "b"]，索引为 [(1, 2), (3, 4)]
        df = DataFrame(
            [[1, 2], [3, 4]],
            columns=["a", "b"],
            index=MultiIndex.from_tuples([(1, 2), (3, 4)]),
        )
        # 断言使用 loc 设置值时会抛出 TypeError，匹配字符串 "as an indexer is not supported"
        with pytest.raises(TypeError, match="as an indexer is not supported"):
            df.loc[key] = 1


def test_adding_new_conditional_column() -> None:
    # 创建一个包含一列 "x" 的数据框
    df = DataFrame({"x": [1]})
    # 在满足条件 df["x"] == 1 的行中，增加列 "y"，赋值为 "1"
    df.loc[df["x"] == 1, "y"] = "1"
    # 创建一个期望结果数据框，包含列 "x" 和 "y"，y 列值为 ["1"]
    expected = DataFrame({"x": [1], "y": ["1"]})
    # 断言 df 和 expected 数据框相等
    tm.assert_frame_equal(df, expected)

    df = DataFrame({"x": [1]})
    # 尝试插入 numpy 会存储为 'object' 类型的值
    value = lambda x: x
    df.loc[df["x"] == 1, "y"] = value
    # 创建一个期望结果数据框，包含列 "x" 和 "y"，y 列值为 [value]
    expected = DataFrame({"x": [1], "y": [value]})
    # 断言 df 和 expected 数据框相等
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    ("dtype", "infer_string"),
    [
        (object, False),
        ("string[pyarrow_numpy]", True),
    ],
)
def test_adding_new_conditional_column_with_string(dtype, infer_string) -> None:
    # 引入 pyarrow，如果不存在则跳过测试
    pytest.importorskip("pyarrow")

    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    with pd.option_context("future.infer_string", infer_string):
        # 在满足条件 df["a"] == 1 的行中，增加列 "c"，赋值为 "1"
        df.loc[df["a"] == 1, "c"] = "1"
    # 创建一个预期的 DataFrame 对象，包含列 'a', 'b', 'c'，并定义各列的数据类型
    expected = DataFrame({"a": [1, 2], "b": [3, 4], "c": ["1", float("nan")]})
        .astype({"a": "int64", "b": "int64", "c": dtype})
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较当前的 DataFrame (df) 和预期的 DataFrame (expected) 是否相等
    tm.assert_frame_equal(df, expected)
# 在 pytest 中导入 pyarrow 模块，如果未安装则跳过当前测试
def test_add_new_column_infer_string():
    # GH#55366
    pytest.importorskip("pyarrow")
    # 创建一个包含单列 'x' 的 DataFrame
    df = DataFrame({"x": [1]})
    # 使用上下文管理器设置 'future.infer_string' 选项为 True
    with pd.option_context("future.infer_string", True):
        # 将符合条件的行的 'y' 列设置为字符串 "1"
        df.loc[df["x"] == 1, "y"] = "1"
    # 创建期望的 DataFrame，包含 'x' 和 'y' 两列，'y' 列的数据类型为字符串数组
    expected = DataFrame(
        {"x": [1], "y": Series(["1"], dtype="string[pyarrow_numpy]")},
        columns=Index(["x", "y"], dtype=object),
    )
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(df, expected)


class TestSetitemValidation:
    # 本测试类改编自 pandas/tests/arrays/masked/test_indexing.py，用于检查警告而非错误
    def _check_setitem_invalid(self, df, invalid, indexer, warn):
        # 设置错误消息的正则表达式模式
        msg = "Setting an item of incompatible dtype is deprecated"
        msg = re.escape(msg)

        # 备份原始 DataFrame
        orig_df = df.copy()

        # 使用 iloc 进行索引
        with tm.assert_produces_warning(warn, match=msg):
            df.iloc[indexer, 0] = invalid
            df = orig_df.copy()

        # 使用 loc 进行索引
        with tm.assert_produces_warning(warn, match=msg):
            df.loc[indexer, "a"] = invalid
            df = orig_df.copy()

    # 不兼容标量值的集合
    _invalid_scalars = [
        1 + 2j,
        "True",
        "1",
        "1.0",
        pd.NaT,
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ]
    # 索引器的集合
    _indexers = [0, [0], slice(0, 1), [True, False, False], slice(None, None, None)]

    @pytest.mark.parametrize(
        "invalid", _invalid_scalars + [1, 1.0, np.int64(1), np.float64(1)]
    )
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_bool(self, invalid, indexer):
        # 创建一个布尔类型的 DataFrame，用于测试
        df = DataFrame({"a": [True, False, False]}, dtype="bool")
        # 调用内部方法检查设置错误
        self._check_setitem_invalid(df, invalid, indexer, FutureWarning)

    @pytest.mark.parametrize("invalid", _invalid_scalars + [True, 1.5, np.float64(1.5)])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_int(self, invalid, any_int_numpy_dtype, indexer):
        # 创建一个整数类型的 DataFrame，用于测试
        df = DataFrame({"a": [1, 2, 3]}, dtype=any_int_numpy_dtype)
        # 根据值是否是 NaT、pd.NaT 或 np.isnat(invalid) 来决定是否产生警告
        if isna(invalid) and invalid is not pd.NaT and not np.isnat(invalid):
            warn = None
        else:
            warn = FutureWarning
        # 调用内部方法检查设置错误
        self._check_setitem_invalid(df, invalid, indexer, warn)

    @pytest.mark.parametrize("invalid", _invalid_scalars + [True])
    @pytest.mark.parametrize("indexer", _indexers)
    def test_setitem_validation_scalar_float(self, invalid, float_numpy_dtype, indexer):
        # 创建一个浮点数类型的 DataFrame，用于测试
        df = DataFrame({"a": [1, 2, None]}, dtype=float_numpy_dtype)
        # 调用内部方法检查设置错误
        self._check_setitem_invalid(df, invalid, indexer, FutureWarning)
```