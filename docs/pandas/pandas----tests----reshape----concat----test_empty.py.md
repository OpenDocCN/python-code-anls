# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_empty.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据
import pytest  # 导入 pytest 库，用于编写和运行测试

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 中导入特定模块和函数
    DataFrame,
    RangeIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestEmptyConcat:
    def test_handle_empty_objects(self, sort, using_infer_string):
        df = DataFrame(  # 创建 DataFrame 对象，包含随机数据
            np.random.default_rng(2).standard_normal((10, 4)), columns=list("abcd")
        )

        dfcopy = df[:5].copy()  # 复制 DataFrame 的前5行数据
        dfcopy["foo"] = "bar"  # 在复制的 DataFrame 中添加新列 'foo' 并赋值为 'bar'
        empty = df[5:5]  # 创建一个空的 DataFrame

        frames = [dfcopy, empty, empty, df[5:]]  # 构建一个 DataFrame 列表
        concatted = concat(frames, axis=0, sort=sort)  # 沿行方向拼接 DataFrame 列表

        expected = df.reindex(columns=["a", "b", "c", "d", "foo"])  # 重新索引 DataFrame 列
        expected["foo"] = expected["foo"].astype(  # 将列 'foo' 的数据类型转换为对象或指定类型
            object if not using_infer_string else "string[pyarrow_numpy]"
        )
        expected.loc[0:4, "foo"] = "bar"  # 在前5行设置列 'foo' 的值为 'bar'

        tm.assert_frame_equal(concatted, expected)  # 断言拼接后的 DataFrame 是否符合预期

        # empty as first element with time series
        # GH3259
        df = DataFrame(  # 创建包含时间序列的 DataFrame
            {"A": range(10000)}, index=date_range("20130101", periods=10000, freq="s")
        )
        empty = DataFrame()  # 创建一个空的 DataFrame
        result = concat([df, empty], axis=1)  # 沿列方向拼接两个 DataFrame
        tm.assert_frame_equal(result, df)  # 断言拼接后的 DataFrame 是否符合预期
        result = concat([empty, df], axis=1)  # 沿列方向拼接两个 DataFrame
        tm.assert_frame_equal(result, df)  # 断言拼接后的 DataFrame 是否符合预期

        result = concat([df, empty])  # 默认沿行方向拼接两个 DataFrame
        tm.assert_frame_equal(result, df)  # 断言拼接后的 DataFrame 是否符合预期
        result = concat([empty, df])  # 默认沿行方向拼接两个 DataFrame
        tm.assert_frame_equal(result, df)  # 断言拼接后的 DataFrame 是否符合预期

    def test_concat_empty_series(self):
        # GH 11082
        s1 = Series([1, 2, 3], name="x")  # 创建具有名称 'x' 的 Series 对象
        s2 = Series(name="y", dtype="float64")  # 创建具有名称 'y' 和 float64 类型的空 Series 对象
        res = concat([s1, s2], axis=1)  # 沿列方向拼接两个 Series
        exp = DataFrame(  # 创建预期的 DataFrame
            {"x": [1, 2, 3], "y": [np.nan, np.nan, np.nan]},
            index=RangeIndex(3),
        )
        tm.assert_frame_equal(res, exp)  # 断言拼接后的 DataFrame 是否符合预期

        s1 = Series([1, 2, 3], name="x")  # 创建具有名称 'x' 的 Series 对象
        s2 = Series(name="y", dtype="float64")  # 创建具有名称 'y' 和 float64 类型的空 Series 对象
        res = concat([s1, s2], axis=0)  # 沿行方向拼接两个 Series
        # name will be reset
        exp = Series([1, 2, 3], dtype="float64")  # 创建预期的 Series 对象
        tm.assert_series_equal(res, exp)  # 断言拼接后的 Series 是否符合预期

        # empty Series with no name
        s1 = Series([1, 2, 3], name="x")  # 创建具有名称 'x' 的 Series 对象
        s2 = Series(name=None, dtype="float64")  # 创建无名称和 float64 类型的空 Series 对象
        res = concat([s1, s2], axis=1)  # 沿列方向拼接两个 Series
        exp = DataFrame(  # 创建预期的 DataFrame
            {"x": [1, 2, 3], 0: [np.nan, np.nan, np.nan]},
            columns=["x", 0],
            index=RangeIndex(3),
        )
        tm.assert_frame_equal(res, exp)  # 断言拼接后的 DataFrame 是否符合预期

    @pytest.mark.parametrize("tz", [None, "UTC"])  # 参数化测试，参数为 tz，取值为 None 和 "UTC"
    @pytest.mark.parametrize("values", [[], [1, 2, 3]])  # 参数化测试，参数为 values，取值为空列表和 [1, 2, 3]
    def test_concat_empty_series_timelike(self, tz, values):
        # GH 18447
        # 在空 Series 上测试时间相关操作

        # 创建一个空的 Series，使用指定时区进行本地化
        first = Series([], dtype="M8[ns]").dt.tz_localize(tz)

        # 根据是否有值来确定第二个 Series 的数据类型
        dtype = None if values else np.float64
        second = Series(values, dtype=dtype)

        # 创建期望的 DataFrame，包含两列：第一列是空的时间数据，第二列是给定的 values 数据
        expected = DataFrame(
            {
                0: Series([pd.NaT] * len(values), dtype="M8[ns]").dt.tz_localize(tz),
                1: values,
            }
        )

        # 执行 concat 操作，沿着列方向进行连接
        result = concat([first, second], axis=1)

        # 断言结果 DataFrame 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "left,right,expected",
        [
            # booleans
            (np.bool_, np.int32, np.object_),  # changed from int32 in 2.0 GH#39817
            (np.bool_, np.float32, np.object_),
            # datetime-like
            ("m8[ns]", np.bool_, np.object_),
            ("m8[ns]", np.int64, np.object_),
            ("M8[ns]", np.bool_, np.object_),
            ("M8[ns]", np.int64, np.object_),
            # categorical
            ("category", "category", "category"),
            ("category", "object", "object"),
        ],
    )
    def test_concat_empty_series_dtypes(self, left, right, expected):
        # GH#39817, GH#45101
        # 在空 Series 上测试不同数据类型的连接

        # 执行 concat 操作，连接两个 Series，分别设置它们的数据类型为 left 和 right
        result = concat([Series(dtype=left), Series(dtype=right)])

        # 断言连接后的结果的数据类型符合期望
        assert result.dtype == expected

    @pytest.mark.parametrize(
        "dtype", ["float64", "int8", "uint8", "bool", "m8[ns]", "M8[ns]"]
    )
    def test_concat_empty_series_dtypes_match_roundtrips(self, dtype):
        # 在空 Series 上测试数据类型匹配的往返操作

        dtype = np.dtype(dtype)

        # 执行 concat 操作，创建一个指定数据类型的 Series
        result = concat([Series(dtype=dtype)])

        # 断言连接后的结果的数据类型与指定的数据类型相同
        assert result.dtype == dtype

        # 执行 concat 操作，连接两个相同数据类型的 Series
        result = concat([Series(dtype=dtype), Series(dtype=dtype)])

        # 断言连接后的结果的数据类型与指定的数据类型相同
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", ["float64", "int8", "uint8", "m8[ns]", "M8[ns]"])
    @pytest.mark.parametrize(
        "dtype2",
        ["float64", "int8", "uint8", "m8[ns]", "M8[ns]"],
    )
    def test_concat_empty_series_dtypes_roundtrips(self, dtype, dtype2):
        # 检查是否为相同的数据类型，如果是，则跳过测试
        if dtype == dtype2:
            pytest.skip("same dtype is not applicable for test")

        # 内部函数：确定两个整数数据类型的结果类型
        def int_result_type(dtype, dtype2):
            typs = {dtype.kind, dtype2.kind}
            # 如果类型集合不包含除了整数以外的类型，且至少一个是整数，则返回整数类型
            if not len(typs - {"i", "u", "b"}) and (
                dtype.kind == "i" or dtype2.kind == "i"
            ):
                return "i"
            elif not len(typs - {"u", "b"}) and (
                dtype.kind == "u" or dtype2.kind == "u"
            ):
                return "u"
            return None

        # 内部函数：确定两个浮点数数据类型的结果类型
        def float_result_type(dtype, dtype2):
            typs = {dtype.kind, dtype2.kind}
            # 如果类型集合不包含除了浮点数以外的类型，且至少一个是浮点数，则返回浮点数类型
            if not len(typs - {"f", "i", "u"}) and (
                dtype.kind == "f" or dtype2.kind == "f"
            ):
                return "f"
            return None

        # 内部函数：获取两个数据类型的结果类型
        def get_result_type(dtype, dtype2):
            result = float_result_type(dtype, dtype2)
            if result is not None:
                return result
            result = int_result_type(dtype, dtype2)
            if result is not None:
                return result
            return "O"  # 默认返回对象类型

        # 将输入的数据类型转换为NumPy的数据类型对象
        dtype = np.dtype(dtype)
        dtype2 = np.dtype(dtype2)
        # 调用函数获取预期的结果类型
        expected = get_result_type(dtype, dtype2)
        # 创建两个空Series并连接，获取连接后的数据类型
        result = concat([Series(dtype=dtype), Series(dtype=dtype2)]).dtype
        # 断言连接后的数据类型的kind与预期的结果类型一致
        assert result.kind == expected

    def test_concat_empty_series_dtypes_triple(self):
        # 测试连续连接三个空Series，预期结果的数据类型为对象类型
        assert (
            concat(
                [Series(dtype="M8[ns]"), Series(dtype=np.bool_), Series(dtype=np.int64)]
            ).dtype
            == np.object_
        )

    def test_concat_empty_series_dtype_category_with_array(self):
        # GH#18515 测试类别数据类型和浮点数类型的Series连接
        assert (
            concat(
                [Series(np.array([]), dtype="category"), Series(dtype="float64")]
            ).dtype
            == "float64"
        )

    def test_concat_empty_series_dtypes_sparse(self):
        # 测试稀疏Series的连接操作

        # 测试两个稀疏Series连接，预期结果的数据类型为稀疏浮点数
        result = concat(
            [
                Series(dtype="float64").astype("Sparse"),
                Series(dtype="float64").astype("Sparse"),
            ]
        )
        assert result.dtype == "Sparse[float64]"

        # 测试一个稀疏Series和一个非稀疏Series连接，预期结果的数据类型为对应的稀疏类型
        result = concat(
            [Series(dtype="float64").astype("Sparse"), Series(dtype="float64")]
        )
        expected = pd.SparseDtype(np.float64)
        assert result.dtype == expected

        # 测试一个稀疏Series和一个对象Series连接，预期结果的数据类型为对应的稀疏类型
        result = concat(
            [Series(dtype="float64").astype("Sparse"), Series(dtype="object")]
        )
        expected = pd.SparseDtype("object")
        assert result.dtype == expected

    def test_concat_empty_df_object_dtype(self):
        # GH 9149 测试DataFrame对象列数据类型为空对象类型的情况
        df_1 = DataFrame({"Row": [0, 1, 1], "EmptyCol": np.nan, "NumberCol": [1, 2, 3]})
        df_2 = DataFrame(columns=df_1.columns)
        # 将两个DataFrame对象沿行方向连接
        result = concat([df_1, df_2], axis=0)
        # 预期结果将df_1转换为所有列为对象类型的DataFrame
        expected = df_1.astype(object)
        # 断言连接后的DataFrame与预期结果相等
        tm.assert_frame_equal(result, expected)
    def test_concat_empty_dataframe_dtypes(self):
        # 创建一个空的 DataFrame，列名为 ['a', 'b', 'c']
        df = DataFrame(columns=list("abc"))
        # 将 'a' 列转换为布尔类型
        df["a"] = df["a"].astype(np.bool_)
        # 将 'b' 列转换为 32 位整数类型
        df["b"] = df["b"].astype(np.int32)
        # 将 'c' 列转换为双精度浮点数类型
        df["c"] = df["c"].astype(np.float64)

        # 对两个 df 进行连接操作
        result = concat([df, df])
        # 断言结果中 'a' 列的数据类型为布尔类型
        assert result["a"].dtype == np.bool_
        # 断言结果中 'b' 列的数据类型为 32 位整数类型
        assert result["b"].dtype == np.int32
        # 断言结果中 'c' 列的数据类型为双精度浮点数类型
        assert result["c"].dtype == np.float64

        # 对包含浮点数类型的 df 进行连接操作
        result = concat([df, df.astype(np.float64)])
        # 断言结果中 'a' 列的数据类型为对象类型
        assert result["a"].dtype == np.object_
        # 断言结果中 'b' 列的数据类型为双精度浮点数类型
        assert result["b"].dtype == np.float64
        # 断言结果中 'c' 列的数据类型为双精度浮点数类型
        assert result["c"].dtype == np.float64

    def test_concat_inner_join_empty(self):
        # GH 15328
        # 创建一个空的 DataFrame
        df_empty = DataFrame()
        # 创建一个包含列 'a' 的 DataFrame，索引为 [0, 1]
        df_a = DataFrame({"a": [1, 2]}, index=[0, 1], dtype="int64")
        # 创建一个预期的空 DataFrame，只有列 'a'
        df_expected = DataFrame({"a": []}, index=RangeIndex(0), dtype="int64")

        # 对 df_a 和 df_empty 进行内连接操作
        result = concat([df_a, df_empty], axis=1, join="inner")
        # 断言连接后的结果与预期的空 DataFrame 相等
        tm.assert_frame_equal(result, df_expected)

        # 对 df_a 和 df_empty 进行外连接操作
        result = concat([df_a, df_empty], axis=1, join="outer")
        # 断言连接后的结果与 df_a 相等
        tm.assert_frame_equal(result, df_a)

    def test_empty_dtype_coerce(self):
        # xref to #12411
        # xref to #12045
        # xref to #11594
        # see below

        # 10571
        # 创建包含空值的 DataFrame df1 和 df2，列名为 ['a', 'b']
        df1 = DataFrame(data=[[1, None], [2, None]], columns=["a", "b"])
        df2 = DataFrame(data=[[3, None], [4, None]], columns=["a", "b"])
        # 连接 df1 和 df2
        result = concat([df1, df2])
        # 断言连接后的结果的列数据类型与 df1 的列数据类型相等
        expected = df1.dtypes
        tm.assert_series_equal(result.dtypes, expected)

    def test_concat_empty_dataframe(self):
        # 39037
        # 创建两个空 DataFrame，列名分别为 ['a', 'b'] 和 ['b', 'c']
        df1 = DataFrame(columns=["a", "b"])
        df2 = DataFrame(columns=["b", "c"])
        # 连接 df1、df2 和 df1
        result = concat([df1, df2, df1])
        # 创建一个预期的空 DataFrame，列名为 ['a', 'b', 'c']
        expected = DataFrame(columns=["a", "b", "c"])
        # 断言连接后的结果与预期的空 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 创建两个空 DataFrame，列名分别为 ['a', 'b'] 和 ['b']
        df3 = DataFrame(columns=["a", "b"])
        df4 = DataFrame(columns=["b"])
        # 连接 df3 和 df4
        result = concat([df3, df4])
        # 创建一个预期的空 DataFrame，列名为 ['a', 'b']
        expected = DataFrame(columns=["a", "b"])
        # 断言连接后的结果与预期的空 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_concat_empty_dataframe_different_dtypes(self, using_infer_string):
        # 39037
        # 创建包含整型和对象列的 DataFrame df1 和只包含整型列的 DataFrame df2
        df1 = DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        df2 = DataFrame({"a": [1, 2, 3]})

        # 连接 df1[:0] 和 df2[:0]
        result = concat([df1[:0], df2[:0]])
        # 断言连接后的结果中 'a' 列的数据类型为 int64
        assert result["a"].dtype == np.int64
        # 断言连接后的结果中 'b' 列的数据类型为对象类型或字符串类型（根据 using_infer_string 的值决定）

    def test_concat_to_empty_ea(self):
        """48510 `concat` to an empty EA should maintain type EA dtype."""
        # 创建一个包含整型数据的空 DataFrame，列名为 ['a']，使用 pd.Int64Dtype()
        df_empty = DataFrame({"a": pd.array([], dtype=pd.Int64Dtype())})
        # 创建一个包含整型数据的 DataFrame，列名为 ['a']，使用 pd.Int64Dtype()
        df_new = DataFrame({"a": pd.array([1, 2, 3], dtype=pd.Int64Dtype())})
        # 复制 df_new，作为预期结果
        expected = df_new.copy()
        # 连接 df_empty 和 df_new
        result = concat([df_empty, df_new])
        # 断言连接后的结果与预期结果相等
        tm.assert_frame_equal(result, expected)
```