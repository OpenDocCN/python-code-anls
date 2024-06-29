# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_astype.py`

```
# 导入 re 模块，用于正则表达式操作
import re

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pytest 库
import pytest

# 导入 pandas.util._test_decorators 模块，并使用 td 别名
import pandas.util._test_decorators as td

# 导入 pandas 库，并使用 pd 别名
import pandas as pd

# 从 pandas 中导入特定对象
from pandas import (
    Categorical,
    CategoricalDtype,
    DataFrame,
    DatetimeTZDtype,
    Index,
    Interval,
    IntervalDtype,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    option_context,
)

# 导入 pandas._testing 模块，并使用 tm 别名
import pandas._testing as tm


def _check_cast(df, v):
    """
    检查 DataFrame 中所有列的数据类型是否都等于 v
    """
    # 使用 assert 来检查条件是否成立，如果不成立则抛出异常
    assert all(s.dtype.name == v for _, s in df.items())


class TestAstype:
    def test_astype_float(self, float_frame):
        # 将 DataFrame 中的值转换为 int 类型，并创建期望的 DataFrame
        casted = float_frame.astype(int)
        expected = DataFrame(
            float_frame.values.astype(int),
            index=float_frame.index,
            columns=float_frame.columns,
        )
        # 使用 pandas._testing.assert_frame_equal 检查两个 DataFrame 是否相等
        tm.assert_frame_equal(casted, expected)

        # 将 DataFrame 中的值转换为 np.int32 类型，并创建期望的 DataFrame
        casted = float_frame.astype(np.int32)
        expected = DataFrame(
            float_frame.values.astype(np.int32),
            index=float_frame.index,
            columns=float_frame.columns,
        )
        tm.assert_frame_equal(casted, expected)

        # 在 float_frame 中新增一列 'foo' 并将其值设为字符串 '5'
        float_frame["foo"] = "5"
        # 将 DataFrame 中的值转换为 int 类型，并创建期望的 DataFrame
        casted = float_frame.astype(int)
        expected = DataFrame(
            float_frame.values.astype(int),
            index=float_frame.index,
            columns=float_frame.columns,
        )
        tm.assert_frame_equal(casted, expected)

    def test_astype_mixed_float(self, mixed_float_frame):
        # 对混合类型的 DataFrame 进行列重索引，并将指定列转换为 'float32' 类型
        casted = mixed_float_frame.reindex(columns=["A", "B"]).astype("float32")
        # 调用 _check_cast 函数，检查所有列的数据类型是否都为 'float32'
        _check_cast(casted, "float32")

        # 对混合类型的 DataFrame 进行列重索引，并将指定列转换为 'float16' 类型
        casted = mixed_float_frame.reindex(columns=["A", "B"]).astype("float16")
        _check_cast(casted, "float16")

    def test_astype_mixed_type(self):
        # 创建一个包含多种数据类型的 DataFrame
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                "float32": np.array([1.0] * 10, dtype="float32"),
                "int32": np.array([1] * 10, dtype="int32"),
            },
            index=np.arange(10),
        )
        # 复制 DataFrame 中的数值数据，并仅保留数值列
        mn = df._get_numeric_data().copy()
        mn["little_float"] = np.array(12345.0, dtype="float16")
        mn["big_float"] = np.array(123456789101112.0, dtype="float64")

        # 将 DataFrame 中的数值数据转换为 'float64' 类型
        casted = mn.astype("float64")
        _check_cast(casted, "float64")

        # 将 DataFrame 中的数值数据转换为 'int64' 类型
        casted = mn.astype("int64")
        _check_cast(casted, "int64")

        # 对 DataFrame 进行列重索引，并将指定列转换为 'float16' 类型
        casted = mn.reindex(columns=["little_float"]).astype("float16")
        _check_cast(casted, "float16")

        # 将 DataFrame 中的数值数据转换为 'float32' 类型
        casted = mn.astype("float32")
        _check_cast(casted, "float32")

        # 将 DataFrame 中的数值数据转换为 'int32' 类型
        casted = mn.astype("int32")
        _check_cast(casted, "int32")

        # 将 DataFrame 中的数值数据转换为 'object' 类型
        casted = mn.astype("O")
        _check_cast(casted, "object")
    # 使用 `float_frame` 的副本创建 DataFrame `df`
    df = float_frame.copy()
    # 将 `float_frame` 转换为整数类型，预期结果存储在 `expected` 中
    expected = float_frame.astype(int)
    # 在 `df` 中新增一列名为 "string" 的列，赋值为字符串 "foo"
    df["string"] = "foo"
    # 将 `df` 转换为整数类型，忽略错误，结果存储在 `casted` 中
    casted = df.astype(int, errors="ignore")

    # 将 `expected` 中的 "string" 列也赋值为 "foo"
    expected["string"] = "foo"
    # 使用测试工具比较 `casted` 和 `expected` 的内容是否相等
    tm.assert_frame_equal(casted, expected)

    # 使用 `float_frame` 的副本创建 DataFrame `df`
    df = float_frame.copy()
    # 将 `float_frame` 转换为 np.int32 类型，预期结果存储在 `expected` 中
    expected = float_frame.astype(np.int32)
    # 在 `df` 中新增一列名为 "string" 的列，赋值为字符串 "foo"
    df["string"] = "foo"
    # 将 `df` 转换为 np.int32 类型，忽略错误，结果存储在 `casted` 中
    casted = df.astype(np.int32, errors="ignore")

    # 将 `expected` 中的 "string" 列也赋值为 "foo"
    expected["string"] = "foo"
    # 使用测试工具比较 `casted` 和 `expected` 的内容是否相等
    tm.assert_frame_equal(casted, expected)

    # 使用 `float_frame` 的副本创建 DataFrame `df`
    df = float_frame.copy()
    # 对 `float_frame` 进行四舍五入并转换为 np.int32 类型，结果存储在 `tf` 中
    tf = np.round(float_frame).astype(np.int32)
    # 将 `tf` 转换为 np.float32 类型（此处可能是一种误用，因为结果未被赋值或进一步使用）

    # TODO(wesm): 需要验证这一步操作的正确性？

    # 将 `float_frame` 转换为 np.float64 类型，结果存储在 `tf` 中
    tf = float_frame.astype(np.float64)
    # 将 `tf` 转换为 np.int64 类型，结果存储在 `tf` 中

    # 使用 `mixed_float_frame` 重新索引列 "A", "B", "C"，结果存储在 `tf` 中
    tf = mixed_float_frame.reindex(columns=["A", "B", "C"])

    # 将 `tf` 转换为 np.int64 类型，结果未存储或使用
    tf.astype(np.int64)
    # 将 `tf` 转换为 np.float32 类型，结果未存储或使用

    # 使用参数化测试，`val` 可以是 np.nan 或 np.inf
    def test_astype_cast_nan_inf_int(self, val, any_int_numpy_dtype):
        # 查看 GitHub issue#14265
        #
        # 检查 NaN 和 inf --> 当转换为整数时抛出错误
        msg = "Cannot convert non-finite values \\(NA or inf\\) to integer"
        # 创建包含单个值 `val` 的 DataFrame `df`
        df = DataFrame([val])

        # 使用 pytest 检查转换为 `any_int_numpy_dtype` 类型时是否抛出 ValueError 错误，并匹配 `msg`
        with pytest.raises(ValueError, match=msg):
            df.astype(any_int_numpy_dtype)

    # 测试将 DataFrame 中的所有元素转换为字符串类型
    def test_astype_str(self):
        # 创建多列数据 Series
        a = Series(date_range("2010-01-04", periods=5))
        b = Series(date_range("3/6/2012 00:00", periods=5, tz="US/Eastern"))
        c = Series([Timedelta(x, unit="D") for x in range(5)])
        d = Series(range(5))
        e = Series([0.0, 0.2, 0.4, 0.6, 0.8])

        # 创建包含多列数据 Series 的 DataFrame `df`
        df = DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})

        # 将 `df` 中的所有列转换为字符串类型，结果存储在 `result` 中
        result = df.astype(str)

        # 创建预期结果的 DataFrame `expected`
        expected = DataFrame(
            {
                "a": list(map(str, (Timestamp(x)._date_repr for x in a._values))),
                "b": list(map(str, map(Timestamp, b._values))),
                "c": [Timedelta(x)._repr_base() for x in c._values],
                "d": list(map(str, d._values)),
                "e": list(map(str, e._values)),
            },
            dtype="object",
        )

        # 使用测试工具比较 `result` 和 `expected` 的内容是否相等
        tm.assert_frame_equal(result, expected)

    # 测试将包含 NaN 的 DataFrame 转换为字符串类型
    def test_astype_str_float(self):
        # 将包含 NaN 的 DataFrame 转换为字符串类型，结果存储在 `result` 中
        result = DataFrame([np.nan]).astype(str)
        # 创建预期结果的 DataFrame `expected`
        expected = DataFrame(["nan"], dtype="object")

        # 使用测试工具比较 `result` 和 `expected` 的内容是否相等
        tm.assert_frame_equal(result, expected)

        # 将包含一个浮点数的 DataFrame 转换为字符串类型，结果存储在 `result` 中
        result = DataFrame([1.12345678901234567890]).astype(str)

        # 预期结果的值
        val = "1.1234567890123457"
        # 创建预期结果的 DataFrame `expected`
        expected = DataFrame([val], dtype="object")

        # 使用测试工具比较 `result` 和 `expected` 的内容是否相等
        tm.assert_frame_equal(result, expected)

    # 使用参数化测试，`dtype_class` 可以是 dict 或 Series
    # 定义一个测试方法，用于测试数据类型转换操作，接受一个 dtype_class 参数
    def test_astype_dict_like(self, dtype_class):
        # GH7271 & GH16717
        # 创建一个包含日期范围的 Series 对象 a，从 "2010-01-04" 开始，共计 5 个时间点
        a = Series(date_range("2010-01-04", periods=5))
        # 创建一个包含整数范围的 Series 对象 b，从 0 开始，共计 5 个整数
        b = Series(range(5))
        # 创建一个包含浮点数列表的 Series 对象 c
        c = Series([0.0, 0.2, 0.4, 0.6, 0.8])
        # 创建一个包含字符串列表的 Series 对象 d
        d = Series(["1.0", "2", "3.14", "4", "5.4"])
        # 创建一个 DataFrame 对象 df，包含以上创建的四个 Series 对象作为列，列名分别为 "a", "b", "c", "d"
        df = DataFrame({"a": a, "b": b, "c": c, "d": d})
        # 复制 df 生成一个 original DataFrame，深度复制以防止原始数据被修改
        original = df.copy(deep=True)

        # 修改部分列的数据类型
        # 使用 dtype_class 定义的数据类型字典 dt1，将 "b" 列改为字符串类型，"d" 列改为 float32 类型
        dt1 = dtype_class({"b": "str", "d": "float32"})
        # 对 df 应用 dt1 定义的数据类型转换，得到 result DataFrame
        result = df.astype(dt1)
        # 预期的结果 DataFrame，"b" 列变为字符串类型，"d" 列变为 float32 类型
        expected = DataFrame(
            {
                "a": a,
                "b": Series(["0", "1", "2", "3", "4"], dtype="object"),
                "c": c,
                "d": Series([1.0, 2.0, 3.14, 4.0, 5.4], dtype="float32"),
            }
        )
        # 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
        # 检查 df 是否保持不变
        tm.assert_frame_equal(df, original)

        # 修改所有列的数据类型
        # 使用 dtype_class 定义的数据类型字典 dt2，将 "b" 列改为 float32 类型，"c" 列改为 float32 类型，"d" 列改为 float64 类型
        dt2 = dtype_class({"b": np.float32, "c": "float32", "d": np.float64})
        # 对 df 应用 dt2 定义的数据类型转换，得到 result DataFrame
        result = df.astype(dt2)
        # 预期的结果 DataFrame，"b" 列变为 float32 类型，"c" 列和 "d" 列按照预期转换为指定的浮点数类型
        expected = DataFrame(
            {
                "a": a,
                "b": Series([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float32"),
                "c": Series([0.0, 0.2, 0.4, 0.6, 0.8], dtype="float32"),
                "d": Series([1.0, 2.0, 3.14, 4.0, 5.4], dtype="float64"),
            }
        )
        # 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
        # 检查 df 是否保持不变
        tm.assert_frame_equal(df, original)

        # 修改所有列的数据类型为字符串类型
        # 使用 dtype_class 定义的数据类型字典 dt3，将所有列改为字符串类型
        dt3 = dtype_class({"a": str, "b": str, "c": str, "d": str})
        # 检查 df 转换为 dt3 定义的数据类型后是否与转换为全字符串类型的结果一致
        tm.assert_frame_equal(df.astype(dt3), df.astype(str))
        # 检查 df 是否保持不变
        tm.assert_frame_equal(df, original)

        # 在 dtype 字典中使用非列名的键会引发错误
        # 定义 dt4 和 dt5，分别包含一个非列名键和一个未知列名键的数据类型字典
        dt4 = dtype_class({"b": str, 2: str})
        dt5 = dtype_class({"e": str})
        # 定义错误消息模板
        msg_frame = (
            "Only a column name can be used for the key in a dtype mappings argument. "
            "'{}' not found in columns."
        )
        # 使用 pytest 检查对 df 应用 dt4 和 dt5 定义的数据类型转换是否引发 KeyError，并匹配对应的错误消息
        with pytest.raises(KeyError, match=msg_frame.format(2)):
            df.astype(dt4)
        with pytest.raises(KeyError, match=msg_frame.format("e")):
            df.astype(dt5)
        # 检查 df 是否保持不变
        tm.assert_frame_equal(df, original)

        # 如果提供的数据类型与原始数据类型相同，则结果 DataFrame 应与原始 DataFrame 相同
        # 使用 dtype_class 根据 df 的每列当前数据类型创建数据类型字典 dt6
        dt6 = dtype_class({col: df[col].dtype for col in df.columns})
        # 将 df 转换为 dt6 定义的数据类型，得到 equiv DataFrame
        equiv = df.astype(dt6)
        # 检查 equiv 和 df 是否相等
        tm.assert_frame_equal(df, equiv)
        # 检查 df 是否保持不变
        tm.assert_frame_equal(df, original)

        # 如果提供的数据类型为空，结果 DataFrame 应与原始 DataFrame 相同
        # 根据 dtype_class 是否为 dict 创建空的数据类型字典 dt7，保持 DataFrame 不变
        dt7 = dtype_class({}) if dtype_class is dict else dtype_class({}, dtype=object)
        # 将 df 转换为 dt7 定义的数据类型，得到 equiv DataFrame
        equiv = df.astype(dt7)
        # 检查 equiv 和 df 是否相等
        tm.assert_frame_equal(df, equiv)
        # 检查 df 是否保持不变
        tm.assert_frame_equal(df, original)
    # 测试在数据框中转换重复列为字符串类型的情况
    def test_astype_duplicate_col(self):
        # 创建第一个序列 'a1'，包含整数数据 [1, 2, 3, 4, 5]，列名为 "a"
        a1 = Series([1, 2, 3, 4, 5], name="a")
        # 创建第二个序列 'b'，包含浮点数数据 [0.1, 0.2, 0.4, 0.6, 0.8]，列名为 "b"
        b = Series([0.1, 0.2, 0.4, 0.6, 0.8], name="b")
        # 创建第三个序列 'a2'，包含整数数据 [0, 1, 2, 3, 4]，列名为 "a"
        a2 = Series([0, 1, 2, 3, 4], name="a")
        # 将上述序列合并成数据框 'df'，沿着列方向合并
        df = concat([a1, b, a2], axis=1)

        # 将数据框 'df' 中的所有列转换为字符串类型，保存在 'result'
        result = df.astype(str)
        # 创建预期的数据框 'expected'，包含每列转换为字符串类型后的预期结果
        a1_str = Series(["1", "2", "3", "4", "5"], dtype="str", name="a")
        b_str = Series(["0.1", "0.2", "0.4", "0.6", "0.8"], dtype=str, name="b")
        a2_str = Series(["0", "1", "2", "3", "4"], dtype="str", name="a")
        expected = concat([a1_str, b_str, a2_str], axis=1)
        # 使用测试框架检查 'result' 和 'expected' 是否相等
        tm.assert_frame_equal(result, expected)

        # 将数据框 'df' 中列 'a' 转换为字符串类型，保存在 'result'
        result = df.astype({"a": "str"})
        # 创建预期的数据框 'expected'，只有列 'a' 被转换为字符串类型
        expected = concat([a1_str, b, a2_str], axis=1)
        # 使用测试框架检查 'result' 和 'expected' 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试在数据框中使用系列参数转换重复列的情况
    def test_astype_duplicate_col_series_arg(self):
        # 生成随机正态分布数据矩阵 'vals'，3行4列
        vals = np.random.default_rng(2).standard_normal((3, 4))
        # 创建包含 'vals' 数据的数据框 'df'，列名为 ["A", "B", "C", "A"]
        df = DataFrame(vals, columns=["A", "B", "C", "A"])
        # 获取 'df' 的列数据类型
        dtypes = df.dtypes
        # 将 'A' 列的数据类型更改为字符串类型
        dtypes.iloc[0] = str
        # 将 'C' 列的数据类型更改为 Float64
        dtypes.iloc[2] = "Float64"

        # 使用 'dtypes' 将数据框 'df' 中的各列数据类型进行转换，保存在 'result'
        result = df.astype(dtypes)
        # 创建预期的数据框 'expected'，每列根据 'dtypes' 进行相应转换
        expected = DataFrame(
            {
                0: Series(vals[:, 0].astype(str), dtype=object),
                1: vals[:, 1],
                2: pd.array(vals[:, 2], dtype="Float64"),
                3: vals[:, 3],
            }
        )
        # 设置 'expected' 的列名与 'df' 一致
        expected.columns = df.columns
        # 使用测试框架检查 'result' 和 'expected' 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用参数化测试来测试将数据框中的列转换为分类类型的情况
    @pytest.mark.parametrize(
        "dtype",
        [
            "category",
            CategoricalDtype(),
            CategoricalDtype(ordered=True),
            CategoricalDtype(ordered=False),
            CategoricalDtype(categories=list("abcdef")),
            CategoricalDtype(categories=list("edba"), ordered=False),
            CategoricalDtype(categories=list("edcb"), ordered=True),
        ],
        ids=repr,
    )
    def test_astype_categorical(self, dtype):
        # 创建包含字典 'd' 的数据框 'df'
        d = {"A": list("abbc"), "B": list("bccd"), "C": list("cdde")}
        df = DataFrame(d)
        # 将数据框 'df' 中的各列数据类型转换为 'dtype' 指定的分类类型，保存在 'result'
        result = df.astype(dtype)
        # 创建预期的数据框 'expected'，其中每列数据类型均为 'dtype' 指定的分类类型
        expected = DataFrame({k: Categorical(v, dtype=dtype) for k, v in d.items()})
        # 使用测试框架检查 'result' 和 'expected' 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试将数据框中的列转换为分类类型时，如果传递的类不是实例化对象，则引发异常
    @pytest.mark.parametrize("cls", [CategoricalDtype, DatetimeTZDtype, IntervalDtype])
    def test_astype_categoricaldtype_class_raises(self, cls):
        # 创建包含列 'A' 的数据框 'df'
        df = DataFrame({"A": ["a", "a", "b", "c"]})
        # 准备用于异常消息的字符串 'xpr'
        xpr = f"Expected an instance of {cls.__name__}"
        # 使用测试框架检查在尝试将列 'A' 转换为 'cls' 指定的类型时是否引发 TypeError 异常
        with pytest.raises(TypeError, match=xpr):
            df.astype({"A": cls})

        # 使用测试框架检查在尝试将列 'A' 的数据类型转换为 'cls' 指定的类型时是否引发 TypeError 异常
        with pytest.raises(TypeError, match=xpr):
            df["A"].astype(cls)
    # 定义测试方法，测试数据类型转换功能，使用参数化标记传入任意整数扩展类型
    def test_astype_extension_dtypes(self, any_int_ea_dtype):
        # GH#22578: GitHub issue reference

        # 将数据类型设置为传入的任意整数扩展类型
        dtype = any_int_ea_dtype

        # 创建一个包含浮点数的数据框
        df = DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["a", "b"])

        # 创建预期结果1：将数据框的列转换为指定的扩展数据类型
        expected1 = DataFrame(
            {
                "a": pd.array([1, 3, 5], dtype=dtype),
                "b": pd.array([2, 4, 6], dtype=dtype),
            }
        )

        # 断言转换后的数据框与预期结果1相等
        tm.assert_frame_equal(df.astype(dtype), expected1)

        # 连续两次类型转换后，断言数据框与预期结果1相等
        tm.assert_frame_equal(df.astype("int64").astype(dtype), expected1)

        # 再次类型转换为浮点数后，断言数据框与初始数据框相等
        tm.assert_frame_equal(df.astype(dtype).astype("float64"), df)

        # 修改数据框的一列为指定的扩展数据类型
        df["b"] = df["b"].astype(dtype)

        # 创建预期结果2：预期数据框的一列转换为指定的扩展数据类型
        expected2 = DataFrame(
            {"a": [1.0, 3.0, 5.0], "b": pd.array([2, 4, 6], dtype=dtype)}
        )

        # 断言数据框与预期结果2相等
        tm.assert_frame_equal(df, expected2)

        # 再次类型转换为指定的扩展数据类型后，断言数据框与预期结果1相等
        tm.assert_frame_equal(df.astype(dtype), expected1)

        # 连续两次类型转换后，断言数据框与预期结果1相等
        tm.assert_frame_equal(df.astype("int64").astype(dtype), expected1)

    # 定义测试方法，测试一维数据类型转换功能，使用参数化标记传入任意整数扩展类型
    def test_astype_extension_dtypes_1d(self, any_int_ea_dtype):
        # GH#22578: GitHub issue reference

        # 将数据类型设置为传入的任意整数扩展类型
        dtype = any_int_ea_dtype

        # 创建包含一列浮点数的数据框
        df = DataFrame({"a": [1.0, 2.0, 3.0]})

        # 创建预期结果1：将数据框的一列转换为指定的扩展数据类型
        expected1 = DataFrame({"a": pd.array([1, 2, 3], dtype=dtype)})

        # 断言数据框转换为指定的扩展数据类型后与预期结果1相等
        tm.assert_frame_equal(df.astype(dtype), expected1)

        # 连续两次类型转换后，断言数据框与预期结果1相等
        tm.assert_frame_equal(df.astype("int64").astype(dtype), expected1)

        # 修改数据框的一列为指定的扩展数据类型
        df["a"] = df["a"].astype(dtype)

        # 创建预期结果2：预期数据框的一列转换为指定的扩展数据类型
        expected2 = DataFrame({"a": pd.array([1, 2, 3], dtype=dtype)})

        # 断言数据框与预期结果2相等
        tm.assert_frame_equal(df, expected2)

        # 再次类型转换为指定的扩展数据类型后，断言数据框与预期结果1相等
        tm.assert_frame_equal(df.astype(dtype), expected1)

        # 连续两次类型转换后，断言数据框与预期结果1相等
        tm.assert_frame_equal(df.astype("int64").astype(dtype), expected1)

    # 使用参数化标记传入数据类型参数，测试数据框列的类型转换功能
    @pytest.mark.parametrize("dtype", ["category", "Int64"])
    def test_astype_extension_dtypes_duplicate_col(self, dtype):
        # GH#24704: GitHub issue reference

        # 创建两个具有相同名称但不同值的系列对象
        a1 = Series([0, np.nan, 4], name="a")
        a2 = Series([np.nan, 3, 5], name="a")

        # 将两个系列对象合并为数据框
        df = concat([a1, a2], axis=1)

        # 将数据框的列转换为指定的数据类型
        result = df.astype(dtype)

        # 创建预期结果：预期数据框的列转换为指定的数据类型
        expected = concat([a1.astype(dtype), a2.astype(dtype)], axis=1)

        # 断言数据框与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 使用参数化标记传入数据类型参数，测试数据框列元数据的类型转换功能
    @pytest.mark.parametrize(
        "dtype", [{100: "float64", 200: "uint64"}, "category", "float64"]
    )
    def test_astype_column_metadata(self, dtype):
        # GH#19920: GitHub issue reference

        # 创建一个具有特定列索引类型的数据框
        columns = Index([100, 200, 300], dtype=np.uint64, name="foo")
        df = DataFrame(np.arange(15).reshape(5, 3), columns=columns)

        # 将数据框的列转换为指定的数据类型
        df = df.astype(dtype)

        # 断言数据框的列索引与初始索引相等
        tm.assert_index_equal(df.columns, columns)
    # 定义一个测试方法，用于将数据框中的对象类型转换为指定单位的日期时间类型
    def test_astype_from_object_to_datetime_unit(self, unit):
        # 定义包含日期字符串的值列表
        vals = [
            ["2015-01-01", "2015-01-02", "2015-01-03"],
            ["2017-01-01", "2017-01-02", "2017-02-03"],
        ]
        # 创建一个数据框，指定数据类型为对象类型
        df = DataFrame(vals, dtype=object)
        # 构造错误消息，指示意外的 'dtype' 值
        msg = (
            rf"Unexpected value for 'dtype': 'datetime64\[{unit}\]'. "
            r"Must be 'datetime64\[s\]', 'datetime64\[ms\]', 'datetime64\[us\]', "
            r"'datetime64\[ns\]' or DatetimeTZDtype"
        )
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            df.astype(f"M8[{unit}]")

    # 使用 pytest 标记参数化测试，测试从对象类型到指定单位的时间增量类型的转换
    @pytest.mark.parametrize("unit", ["Y", "M", "W", "D", "h", "m"])
    def test_astype_from_object_to_timedelta_unit(self, unit):
        # 定义包含时间增量字符串的值列表
        vals = [
            ["1 Day", "2 Days", "3 Days"],
            ["4 Days", "5 Days", "6 Days"],
        ]
        # 创建一个数据框，指定数据类型为对象类型
        df = DataFrame(vals, dtype=object)
        # 构造错误消息，指示无法从 timedelta64[ns] 转换为 timedelta64[*]，并列出支持的分辨率
        msg = (
            r"Cannot convert from timedelta64\[ns\] to timedelta64\[.*\]. "
            "Supported resolutions are 's', 'ms', 'us', 'ns'"
        )
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            # TODO: this is ValueError while for DatetimeArray it is TypeError;
            #  get these consistent
            df.astype(f"m8[{unit}]")

    # 使用 pytest 标记参数化测试，测试从日期时间类似类型到对象类型的转换
    @pytest.mark.parametrize("dtype", ["M8", "m8"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_astype_from_datetimelike_to_object(self, dtype, unit):
        # tests astype to object dtype
        # GH#19223 / GH#12425
        # 构造数据类型字符串，格式为 "{dtype}[{unit}]"
        dtype = f"{dtype}[{unit}]"
        # 创建一个包含数值的 NumPy 数组，指定数据类型为构造的日期时间类似类型
        arr = np.array([[1, 2, 3]], dtype=dtype)
        # 创建一个数据框，以数组为数据源
        df = DataFrame(arr)
        # 将数据框转换为对象类型
        result = df.astype(object)
        # 断言结果数据框中所有列的数据类型为对象类型
        assert (result.dtypes == object).all()

        # 如果数据类型以 "M8" 开头，则断言第一个元素为 Timestamp 类型，单位为指定的单位
        if dtype.startswith("M8"):
            assert result.iloc[0, 0] == Timestamp(1, unit=unit)
        else:
            # 否则，断言第一个元素为 Timedelta 类型，单位为指定的单位
            assert result.iloc[0, 0] == Timedelta(1, unit=unit)

    # 使用 pytest 标记参数化测试，测试从任意真实的 NumPy 数据类型到指定日期时间类型的转换
    @pytest.mark.parametrize("dtype", ["M8", "m8"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_astype_to_datetimelike_unit(self, any_real_numpy_dtype, dtype, unit):
        # tests all units from numeric origination
        # GH#19223 / GH#12425
        # 构造数据类型字符串，格式为 "{dtype}[{unit}]"
        dtype = f"{dtype}[{unit}]"
        # 创建一个包含数值的 NumPy 数组，指定数据类型为任意真实的 NumPy 数据类型
        arr = np.array([[1, 2, 3]], dtype=any_real_numpy_dtype)
        # 创建一个数据框，以数组为数据源
        df = DataFrame(arr)
        # 将数据框转换为指定的日期时间类型
        result = df.astype(dtype)
        # 创建一个预期的数据框，数据类型与结果相同
        expected = DataFrame(arr.astype(dtype))
        # 使用测试工具比较结果数据框与预期数据框是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 标记参数化测试，测试所有单位的日期时间类型
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    # 定义一个测试函数，用于将数据类型转换为 datetime 对象的单元测试
    def test_astype_to_datetime_unit(self, unit):
        # 测试从 datetime 起源的所有单位
        # GH#19223

        # 构建 datetime 类型的字符串表示，例如 'M8[unit]'
        dtype = f"M8[{unit}]"

        # 创建一个包含单行数据的 NumPy 数组，并指定数据类型为 dtype
        arr = np.array([[1, 2, 3]], dtype=dtype)

        # 使用 NumPy 数组创建一个 DataFrame
        df = DataFrame(arr)

        # 从 DataFrame 中选择第一列，创建一个 Series 对象
        ser = df.iloc[:, 0]

        # 使用 Series 对象创建一个 Index 对象
        idx = Index(ser)

        # 提取 Series 对象的原始值，即内部的 NumPy 数组
        dta = ser._values

        # 如果单位是 ["ns", "us", "ms", "s"] 中的一个
        if unit in ["ns", "us", "ms", "s"]:
            # GH#48928
            # 将 DataFrame 转换为指定的 datetime 数据类型
            result = df.astype(dtype)
        else:
            # 否则，抛出类型错误异常，并匹配自定义的错误消息
            msg = rf"Cannot cast DatetimeArray to dtype datetime64\[{unit}\]"

            # 使用 pytest 检查 DataFrame 的类型转换是否引发预期的异常
            with pytest.raises(TypeError, match=msg):
                df.astype(dtype)

            # 使用 pytest 检查 Series 的类型转换是否引发预期的异常
            with pytest.raises(TypeError, match=msg):
                ser.astype(dtype)

            # 使用 pytest 检查 Index 的类型转换是否引发预期的异常
            with pytest.raises(TypeError, match=msg.replace("Array", "Index")):
                idx.astype(dtype)

            # 使用 pytest 检查原始值数组的类型转换是否引发预期的异常
            with pytest.raises(TypeError, match=msg):
                dta.astype(dtype)

            # 如果出现异常，结束函数执行
            return

        # 创建预期的 DataFrame，以便与转换后的结果进行比较
        exp_df = DataFrame(arr.astype(dtype))

        # 断言转换后的 DataFrame 中的数据类型是否全部符合预期
        assert (exp_df.dtypes == dtype).all()

        # 使用 pytest 的 assert_frame_equal 检查结果 DataFrame 是否与预期的 DataFrame 相等
        tm.assert_frame_equal(result, exp_df)

        # 将 Series 对象转换为指定的 datetime 数据类型
        res_ser = ser.astype(dtype)

        # 从预期的 DataFrame 中提取转换后的 Series 对象
        exp_ser = exp_df.iloc[:, 0]

        # 断言转换后的 Series 对象的数据类型是否符合预期
        assert exp_ser.dtype == dtype

        # 使用 pytest 的 assert_series_equal 检查转换后的 Series 是否与预期的 Series 相等
        tm.assert_series_equal(res_ser, exp_ser)

        # 创建预期的 Index 对象，以便与转换后的结果进行比较
        exp_index = Index(exp_ser)

        # 将 Index 对象转换为指定的 datetime 数据类型
        res_index = idx.astype(dtype)

        # 断言转换后的 Index 对象的数据类型是否符合预期
        assert exp_index.dtype == dtype

        # 使用 pytest 的 assert_index_equal 检查转换后的 Index 是否与预期的 Index 相等
        tm.assert_index_equal(res_index, exp_index)

        # 将原始值数组转换为指定的 datetime 数据类型
        res_dta = dta.astype(dtype)

        # 断言转换后的原始值数组的数据类型是否符合预期
        assert exp_dta.dtype == dtype

        # 使用 pytest 的 assert_extension_array_equal 检查转换后的原始值数组是否与预期的原始值数组相等
        tm.assert_extension_array_equal(res_dta, exp_dta)

    # 定义一个测试函数，用于将 timedelta 数据类型转换为 datetime 数据类型的单元测试
    def test_astype_to_timedelta_unit_ns(self):
        # 保持 timedelta 转换
        # GH#19223

        # 指定目标数据类型为 'm8[ns]'
        dtype = "m8[ns]"

        # 创建一个包含单行数据的 NumPy 数组，并指定数据类型为 dtype
        arr = np.array([[1, 2, 3]], dtype=dtype)

        # 使用 NumPy 数组创建一个 DataFrame
        df = DataFrame(arr)

        # 将 DataFrame 转换为指定的 timedelta 数据类型
        result = df.astype(dtype)

        # 创建预期的 DataFrame，以便与转换后的结果进行比较
        expected = DataFrame(arr.astype(dtype))

        # 使用 pytest 的 assert_frame_equal 检查结果 DataFrame 是否与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器，为 unit 参数设置多个值进行测试
    @pytest.mark.parametrize("unit", ["us", "ms", "s", "h", "m", "D"])
    def test_astype_to_timedelta_unit(self, unit):
        # 将单位强制转换为浮点数
        # 在 2.0 版本之前，将其强制转换为浮点数
        dtype = f"m8[{unit}]"  # 定义日期时间类型，根据指定单位
        arr = np.array([[1, 2, 3]], dtype=dtype)  # 创建包含日期时间数据的 NumPy 数组
        df = DataFrame(arr)  # 使用数组创建数据帧
        ser = df.iloc[:, 0]  # 获取数据帧的第一列作为序列
        tdi = Index(ser)  # 使用序列创建索引
        tda = tdi._values  # 获取索引的值数组

        if unit in ["us", "ms", "s"]:
            assert (df.dtypes == dtype).all()  # 断言数据帧的所有列都是指定的日期时间类型
            result = df.astype(dtype)  # 将数据帧转换为指定的日期时间类型
        else:
            # 如果单位不在支持的列表中，则使用最接近的支持单位，即 "s"
            assert (df.dtypes == "m8[s]").all()  # 断言数据帧的所有列都是 "m8[s]" 类型

            msg = (
                rf"Cannot convert from timedelta64\[s\] to timedelta64\[{unit}\]. "
                "Supported resolutions are 's', 'ms', 'us', 'ns'"
            )
            with pytest.raises(ValueError, match=msg):  # 断言抛出 ValueError 异常，并匹配指定的消息
                df.astype(dtype)
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)
            with pytest.raises(ValueError, match=msg):
                tdi.astype(dtype)
            with pytest.raises(ValueError, match=msg):
                tda.astype(dtype)

            return

        result = df.astype(dtype)  # 将数据帧转换为指定的日期时间类型
        # 转换是一个无操作，因此我们只是得到一个副本
        expected = df  # 期望的结果是数据帧本身
        tm.assert_frame_equal(result, expected)  # 断言转换后的结果与期望的结果相等

    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_astype_to_incorrect_datetimelike(self, unit):
        # 尝试将 m 转换为 M，或者反过来
        # GH#19224
        dtype = f"M8[{unit}]"  # 定义日期时间类型，根据指定单位
        other = f"m8[{unit}]"  # 定义另一种日期时间类型，根据相同的单位

        df = DataFrame(np.array([[1, 2, 3]], dtype=dtype))  # 创建数据帧，使用指定的日期时间类型
        msg = rf"Cannot cast DatetimeArray to dtype timedelta64\[{unit}\]"  # 准备用于断言的消息
        with pytest.raises(TypeError, match=msg):  # 断言抛出 TypeError 异常，并匹配指定的消息
            df.astype(other)

        msg = rf"Cannot cast TimedeltaArray to dtype datetime64\[{unit}\]"  # 准备用于断言的消息
        df = DataFrame(np.array([[1, 2, 3]], dtype=other))  # 创建数据帧，使用另一种日期时间类型
        with pytest.raises(TypeError, match=msg):  # 断言抛出 TypeError 异常，并匹配指定的消息
            df.astype(dtype)

    def test_astype_arg_for_errors(self):
        # GH#14878

        df = DataFrame([1, 2, 3])  # 创建包含整数数据的数据帧

        msg = (
            "Expected value of kwarg 'errors' to be one of "
            "['raise', 'ignore']. Supplied value is 'True'"
        )
        with pytest.raises(ValueError, match=re.escape(msg)):  # 断言抛出 ValueError 异常，并匹配指定的消息
            df.astype(np.float64, errors=True)

        df.astype(np.int8, errors="ignore")  # 将数据帧中的数据类型转换为 np.int8，忽略错误

    def test_astype_invalid_conversion(self):
        # GH#47571
        df = DataFrame({"a": [1, 2, "text"], "b": [1, 2, 3]})  # 创建包含混合数据类型的数据帧

        msg = (
            "invalid literal for int() with base 10: 'text': "
            "Error while type casting for column 'a'"
        )

        with pytest.raises(ValueError, match=re.escape(msg)):  # 断言抛出 ValueError 异常，并匹配指定的消息
            df.astype({"a": int})  # 尝试将数据帧中指定列的数据类型转换为整数类型
    def test_astype_arg_for_errors_dictlist(self):
        # 定义一个测试函数，测试 DataFrame 的 astype 方法处理错误字典列表的情况
        # GH#25905 是 GitHub 上的一个问题跟踪编号
        df = DataFrame(
            [
                {"a": "1", "b": "16.5%", "c": "test"},
                {"a": "2.2", "b": "15.3", "c": "another_test"},
            ]
        )
        # 创建期望的 DataFrame 对象，包含处理后的数据类型转换
        expected = DataFrame(
            [
                {"a": 1.0, "b": "16.5%", "c": "test"},
                {"a": 2.2, "b": "15.3", "c": "another_test"},
            ]
        )
        # 将期望中的 'c' 列转换为 object 类型
        expected["c"] = expected["c"].astype("object")
        # 指定要转换的数据类型字典
        type_dict = {"a": "float64", "b": "float64", "c": "object"}

        # 使用 astype 方法进行数据类型转换，忽略错误
        result = df.astype(dtype=type_dict, errors="ignore")

        # 使用测试工具比较结果和期望的 DataFrame
        tm.assert_frame_equal(result, expected)

    def test_astype_dt64tz(self, timezone_frame):
        # 测试 DataFrame 的 astype 方法处理时区日期时间对象的情况
        # 定义期望的数据数组，包含不同时区的时间戳对象
        expected = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        # 创建期望的 DataFrame 对象，保留 object 数据类型
        expected = DataFrame(
            expected,
            index=timezone_frame.index,
            columns=timezone_frame.columns,
            dtype=object,
        )
        # 使用 astype 方法将 DataFrame 的所有列转换为 object 类型
        result = timezone_frame.astype(object)
        # 使用测试工具比较结果和期望的 DataFrame
        tm.assert_frame_equal(result, expected)

        # 定义错误消息，用于预期的异常情况
        msg = "Cannot use .astype to convert from timezone-aware dtype to timezone-"
        # 使用 pytest 来验证特定异常是否被引发
        with pytest.raises(TypeError, match=msg):
            # 尝试使用 astype 方法将时区日期时间转换为 datetime64[ns] 类型，预期引发 TypeError 异常
            timezone_frame.astype("datetime64[ns]")
    def test_astype_dt64tz_to_str(self, timezone_frame):
        # 将数据框中的日期时间数据转换为字符串格式
        result = timezone_frame.astype(str)
        # 预期的结果数据框，包含特定的日期时间格式字符串
        expected = DataFrame(
            [
                [
                    "2013-01-01",
                    "2013-01-01 00:00:00-05:00",
                    "2013-01-01 00:00:00+01:00",
                ],
                ["2013-01-02", "NaT", "NaT"],
                [
                    "2013-01-03",
                    "2013-01-03 00:00:00-05:00",
                    "2013-01-03 00:00:00+01:00",
                ],
            ],
            columns=timezone_frame.columns,
            dtype="object",
        )
        # 使用测试框架验证结果数据框与预期数据框是否相等
        tm.assert_frame_equal(result, expected)

        # 使用特定的显示选项上下文，将数据框转换为字符串并进行验证
        with option_context("display.max_columns", 20):
            result = str(timezone_frame)
            # 验证特定的字符串存在于结果字符串中
            assert (
                "0 2013-01-01 2013-01-01 00:00:00-05:00 2013-01-01 00:00:00+01:00"
            ) in result
            assert (
                "1 2013-01-02                       NaT                       NaT"
            ) in result
            assert (
                "2 2013-01-03 2013-01-03 00:00:00-05:00 2013-01-03 00:00:00+01:00"
            ) in result

    def test_astype_empty_dtype_dict(self):
        # 在下面的问题线程中提到的问题
        # https://github.com/pandas-dev/pandas/issues/33113
        # 创建一个空的数据框
        df = DataFrame()
        # 使用空的数据类型字典来转换数据框的类型
        result = df.astype({})
        # 使用测试框架验证结果数据框与原数据框是否相等
        tm.assert_frame_equal(result, df)
        # 验证结果数据框与原数据框对象不是同一个对象
        assert result is not df

    @pytest.mark.parametrize(
        "data, dtype",
        [
            (["x", "y", "z"], "string[python]"),
            pytest.param(
                ["x", "y", "z"],
                "string[pyarrow]",
                marks=td.skip_if_no("pyarrow"),
            ),
            (["x", "y", "z"], "category"),
            (3 * [Timestamp("2020-01-01", tz="UTC")], None),
            (3 * [Interval(0, 1)], None),
        ],
    )
    @pytest.mark.parametrize("errors", ["raise", "ignore"])
    def test_astype_ignores_errors_for_extension_dtypes(self, data, dtype, errors):
        # https://github.com/pandas-dev/pandas/issues/35471
        # 创建包含特定数据类型的数据框
        df = DataFrame(Series(data, dtype=dtype))
        # 根据错误处理方式进行数据类型转换，使用测试框架验证结果
        if errors == "ignore":
            expected = df
            result = df.astype(float, errors=errors)
            tm.assert_frame_equal(result, expected)
        else:
            msg = "(Cannot cast)|(could not convert)"
            # 验证在特定错误情况下是否抛出异常
            with pytest.raises((ValueError, TypeError), match=msg):
                df.astype(float, errors=errors)

    def test_astype_tz_conversion(self):
        # GH 35973
        # 创建包含时区信息的数据框
        val = {"tz": date_range("2020-08-30", freq="d", periods=2, tz="Europe/London")}
        df = DataFrame(val)
        # 将数据框中的时区信息转换为指定的日期时间类型
        result = df.astype({"tz": "datetime64[ns, Europe/Berlin]"})

        # 预期的结果数据框，转换后的时区信息应该进行时区转换
        expected = df
        expected["tz"] = expected["tz"].dt.tz_convert("Europe/Berlin")
        # 使用测试框架验证结果数据框与预期数据框是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz", ["UTC", "Europe/Berlin"])
    # 测试时使用不同的时区参数 tz
    def test_astype_tz_object_conversion(self, tz):
        # GH 35973
        # 创建包含时区信息的字典数据
        val = {"tz": date_range("2020-08-30", freq="d", periods=2, tz="Europe/London")}
        # 创建预期的 DataFrame 对象
        expected = DataFrame(val)

        # 将预期的 tz 列从其他时区字符串转换为对象数据类型（已独立测试）
        result = expected.astype({"tz": f"datetime64[ns, {tz}]"})
        # 将 tz 列转换为对象数据类型
        result = result.astype({"tz": "object"})

        # 进行真实测试：将对象数据类型的 tz 列转换为指定的时区，与构造时的时区不同
        result = result.astype({"tz": "datetime64[ns, Europe/London]"})
        # 断言 result 与 expected 相等
        tm.assert_frame_equal(result, expected)

    # 将日期时间数据类型转换为字符串数据类型
    def test_astype_dt64_to_string(
        self, frame_or_series, tz_naive_fixture, using_infer_string
    ):
        # GH#41409
        # 获取时区信息
        tz = tz_naive_fixture

        # 创建带有时区信息的日期范围对象
        dti = date_range("2016-01-01", periods=3, tz=tz)
        # 获取内部数据数组
        dta = dti._data
        # 将第一个元素设置为 NaT
        dta[0] = NaT

        # 根据内部数据创建 Series 或 DataFrame 对象
        obj = frame_or_series(dta)
        # 将对象转换为字符串数据类型
        result = obj.astype("string")

        # 检查 Series/DataFrame.astype 是否与 DatetimeArray.astype 匹配
        expected = frame_or_series(dta.astype("string"))
        tm.assert_equal(result, expected)

        # 获取结果中的第一个元素
        item = result.iloc[0]
        # 如果 frame_or_series 是 DataFrame，则再获取一次第一个元素
        if frame_or_series is DataFrame:
            item = item.iloc[0]
        # 如果使用推断字符串转换，断言 item 是 NaN
        if using_infer_string:
            assert item is np.nan
        else:
            assert item is pd.NA

        # 对于非 NaN 值，应该与转换为普通字符串后的结果匹配
        alt = obj.astype(str)
        assert np.all(alt.iloc[1:] == result.iloc[1:])

    # 将时间差数据类型转换为字符串数据类型
    def test_astype_td64_to_string(self, frame_or_series):
        # GH#41409
        # 创建时间差范围对象
        tdi = pd.timedelta_range("1 Day", periods=3)
        # 根据时间差范围创建 Series 或 DataFrame 对象
        obj = frame_or_series(tdi)

        # 创建期望的 Series 或 DataFrame 对象，使用字符串数据类型
        expected = frame_or_series(["1 days", "2 days", "3 days"], dtype="string")
        # 将对象转换为字符串数据类型
        result = obj.astype("string")
        tm.assert_equal(result, expected)

    # 将 DataFrame 的字符串列转换为字节串类型
    def test_astype_bytes(self):
        # GH#39474
        # 将 DataFrame 对象的字符串列转换为字节串类型
        result = DataFrame(["foo", "bar", "baz"]).astype(bytes)
        # 断言结果的第一个列的数据类型为 S3
        assert result.dtypes[0] == np.dtype("S3")

    # 对非连续的索引切片进行数据类型转换测试
    @pytest.mark.parametrize(
        "index_slice",
        [
            np.s_[:2, :2],
            np.s_[:1, :2],
            np.s_[:2, :1],
            np.s_[::2, ::2],
            np.s_[::1, ::2],
            np.s_[::2, ::1],
        ],
    )
    def test_astype_noncontiguous(self, index_slice):
        # GH#42396
        # 创建一个 4x4 的数据数组
        data = np.arange(16).reshape(4, 4)
        # 创建 DataFrame 对象
        df = DataFrame(data)

        # 对非连续索引切片的部分数据进行 int16 类型的数据类型转换
        result = df.iloc[index_slice].astype("int16")
        # 获取期望的 DataFrame 对象
        expected = df.iloc[index_slice]
        # 断言 result 与 expected 在数据类型不检查的情况下相等
        tm.assert_frame_equal(result, expected, check_dtype=False)

    # 测试在保留属性的情况下进行数据类型转换
    def test_astype_retain_attrs(self, any_numpy_dtype):
        # GH#44414
        # 创建具有属性的 DataFrame 对象
        df = DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        df.attrs["Location"] = "Michigan"

        # 将 DataFrame 对象的 'a' 列转换为指定的任意 numpy 数据类型，并获取其属性
        result = df.astype({"a": any_numpy_dtype}).attrs
        # 获取期望的属性
        expected = df.attrs

        # 断言 result 和 expected 的属性相等
        tm.assert_dict_equal(expected, result)
class TestAstypeCategorical:
    def test_astype_from_categorical3(self):
        df = DataFrame({"cats": [1, 2, 3, 4, 5, 6], "vals": [1, 2, 3, 4, 5, 6]})
        cats = Categorical([1, 2, 3, 4, 5, 6])
        exp_df = DataFrame({"cats": cats, "vals": [1, 2, 3, 4, 5, 6]})
        # 将 "cats" 列转换为 pandas 中的 Categorical 类型
        df["cats"] = df["cats"].astype("category")
        # 比较转换后的 DataFrame 是否符合预期
        tm.assert_frame_equal(exp_df, df)

    def test_astype_from_categorical4(self):
        df = DataFrame(
            {"cats": ["a", "b", "b", "a", "a", "d"], "vals": [1, 2, 3, 4, 5, 6]}
        )
        cats = Categorical(["a", "b", "b", "a", "a", "d"])
        exp_df = DataFrame({"cats": cats, "vals": [1, 2, 3, 4, 5, 6]})
        # 将 "cats" 列转换为 pandas 中的 Categorical 类型
        df["cats"] = df["cats"].astype("category")
        # 比较转换后的 DataFrame 是否符合预期
        tm.assert_frame_equal(exp_df, df)

    def test_categorical_astype_to_int(self, any_int_dtype):
        # GH#39402
        df = DataFrame(data={"col1": pd.array([2.0, 1.0, 3.0])})
        # 将 "col1" 列转换为 pandas 中的 Categorical 类型
        df.col1 = df.col1.astype("category")
        # 将 "col1" 列的类型转换为指定的整数类型（通过参数传入）
        df.col1 = df.col1.astype(any_int_dtype)
        expected = DataFrame({"col1": pd.array([2, 1, 3], dtype=any_int_dtype)})
        # 比较转换后的 DataFrame 是否符合预期
        tm.assert_frame_equal(df, expected)

    def test_astype_categorical_to_string_missing(self):
        # https://github.com/pandas-dev/pandas/issues/41797
        df = DataFrame(["a", "b", np.nan])
        expected = df.astype(str)
        # 将 DataFrame 转换为字符串类型的 DataFrame
        cat = df.astype("category")
        result = cat.astype(str)
        # 比较转换后的 DataFrame 是否符合预期
        tm.assert_frame_equal(result, expected)


class IntegerArrayNoCopy(pd.core.arrays.IntegerArray):
    # GH 42501

    def copy(self):
        raise NotImplementedError


class Int16DtypeNoCopy(pd.Int16Dtype):
    # GH 42501

    @classmethod
    def construct_array_type(cls):
        return IntegerArrayNoCopy


def test_frame_astype_no_copy():
    # GH 42501
    df = DataFrame({"a": [1, 4, None, 5], "b": [6, 7, 8, 9]}, dtype=object)
    result = df.astype({"a": Int16DtypeNoCopy()})
    # 断言转换后 "a" 列的数据类型是否为 pd.Int16Dtype()
    assert result.a.dtype == pd.Int16Dtype()
    # 断言转换后 "b" 列是否与原始 DataFrame 的 "b" 列共享内存
    assert np.shares_memory(df.b.values, result.b.values)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_astype_copies(dtype):
    # GH#50984
    pytest.importorskip("pyarrow")
    df = DataFrame({"a": [1, 2, 3]}, dtype=dtype)
    result = df.astype("int64[pyarrow]")
    df.iloc[0, 0] = 100
    expected = DataFrame({"a": [1, 2, 3]}, dtype="int64[pyarrow]")
    # 比较转换后的 DataFrame 是否符合预期
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val", [None, 1, 1.5, np.nan, NaT])
def test_astype_to_string_not_modifying_input(string_storage, val):
    # GH#51073
    df = DataFrame({"a": ["a", "b", val]})
    expected = df.copy()
    with option_context("mode.string_storage", string_storage):
        # 将 DataFrame 转换为字符串类型的 DataFrame
        df.astype("string")
    # 比较转换后的 DataFrame 是否与原始 DataFrame 相同
    tm.assert_frame_equal(df, expected)
```